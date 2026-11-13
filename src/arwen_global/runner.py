"""Experiment runner, gates, checkpoints, and failure receipts for Arwen Global."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import struct
import math
from pathlib import Path
import time

from arwen_global.spectral.diffusion import ExponentialHyperdiffusion
from arwen_global.spectral.transform import SphericalHarmonicTransform

from .checkpoint import (
    TRACKER_KEYS,
    PinnedSnapshot,
    bundle_arrays,
    checkpoint_metadata,
    normalize_trackers,
    read_checkpoint,
    state_from_checkpoint,
    trajectory_from_checkpoint,
    write_checkpoint,
    write_checkpoint_arrays,
)
from .config import (
    DEFAULT_EULERIAN_STEP_RULE_FRACTION,
    ArwenGlobalConfig,
    default_eulerian_step_s,
    eulerian_wind_ceiling_m_s,
)
from .constants import EARTH_RADIUS_M
from .device_memory import (
    device_memory_receipt,
    disable_fft_plan_cache,
    select_device_allocator,
    start_device_peak_tracking,
)
from .dynamics import MoistHybridModel
from .initial_conditions import analytic_initial_state
from .insitu.ledger import InsituLedger, insitu_owned_files
from .longrun_gates import LONG_RUN_THRESHOLD_S, receipt_block as longrun_receipt_block
from .physics import NativeArwenPhysicsBridge, ReferencePhysics
from .pins import pins_hash, pins_receipt
from .profile import PROFILE_NAME, StepProfiler, attach_profiler, calibrate
from .receipt import software_stack, world_receipt_rows, write_receipt
from .semi_implicit import build_semi_implicit
from .semilag.step import SEMILAG_INTEGRATORS
from .water import OCEAN_EXCHANGE_NAME


RECEIPT_NAME = "arwen-global-receipt.json"
DIAGNOSTICS_NAME = "diagnostics.jsonl"
CHECKPOINT_PREFIX = "arwen_global_step"

# The mass/water drift gates compare post-fix means against the very targets
# the fixers reset every step, so they cannot see a conservation leak while
# the fixers are on. These budgets gate what the fixers and the physics local
# water repair ABSORBED, per step, so an unbounded leak fails the run even
# while the fixers hold the mean. The limits are per-precision because the
# healthy ceiling is arithmetic roundoff and roundoff scales with the float:
#
# float64, measured on the six-hour T7 reference campaign (360 steps,
# conservative physics, exchange-clamp closure): mass fixer |log offset|
# <= 1.64e-7, water fixer <= 3.9e-8 relative, physics local repair
# <= 1.14e-13 kg/m2.
#
# float32, measured on the six-hour T63 GDAS canary on the RTX 5090
# (72 steps, 2026-08-31): 4.14e-5, 1.51e-6 relative, and 1.22e-4 kg/m2 -
# that last is exactly 2^-13, the fp32 ulp of the ~500 kg/m2 ledger.
#
# The leaks these exist to catch measured 5e-3 (0.5%/step surface-pressure
# sink), 2.4e-4 relative/step (1%-per-call qv deletion), and 8.2e-4
# kg/m2/step (the unfixed diffusion operator) - every limit sits >=4x above
# its precision's measured healthy ceiling and remains below the leak scale.
#
# These envelopes were measured on the REFERENCE suite, whose exchange-clamp
# closure has no kernel-internal error channel.  (They were measured under
# the v4 uniform reservoir levy; under the v5 atmosphere-internal fixer the
# clamps no longer move water between reservoir and atmosphere, so the water
# fixer absorbs only transport/projection residuals - audit 2026-09-01
# VTW-1.)  The arwen-native suite does:
# Morrison's rainncv is credited 1:1 to the reservoir while the kernel's own
# fp32 sedimentation/process arithmetic sets the actual column debit, and the
# post-kernel negative-species clamp silently creates up to ~1e-7 kg/kg per
# level-species.  Both are genuine fp32 accumulation, not booking routes
# (CPU repro water-repair-262, 2026-09-01: every Noah/PBL booking route
# settles to <=2x2^-14 by construction), so a native run config may carry its
# own MEASURED gates.physics_water_repair_max_step_kg_m2 - the config-level
# override below - with the measurement and sizing arithmetic quoted in the
# config.  Reference configs keep this envelope.
_FIXER_ABSORPTION_LIMITS = {
    "float64": {
        "mass_log_offset": 1.0e-6,
        "water_relative": 1.0e-6,
        "physics_repair_kg_m2": 1.0e-4,
    },
    "float32": {
        "mass_log_offset": 2.0e-4,
        "water_relative": 1.0e-5,
        "physics_repair_kg_m2": 5.0e-4,
    },
}


# The vapor fixer (dynamics._fill_column_holes) closes the vapor clamps
# inside each column: the global-mean column water the clips create per
# step, divided by the global-mean atmospheric water column, is the
# fraction of the atmosphere's water the clips created and the columns
# paid back from their own vapor that step.  Under the v5 per-level global
# rescale (retired 2026-09-02 with the spectral condensate) the same ratio
# was 7.6e-5..1.2e-4 sustained on the T63 GDAS reference config and 9.0e-6
# on the T3 smoke, most of it condensate ringing; with vapor the only
# spectral water field it measures 3.4e-7 on the 10-step T255 native case
# (largest column fraction 0.06) and exactly zero on the T3 smoke, so the
# limit now sits four orders above the healthy figure.  The breakage the
# gate names is unchanged: a transport or physics defect that drives whole
# columns negative every step (the class of leak
# test_receipt_fails_when_the_positivity_fixer_carries_a_sign_defect
# installs, a quarter of the vapor) leaves those columns nothing to pay
# with, the clip creates the water (counted as unfillable) and the drift
# gates read only the projection residual because no reservoir moved.
POSITIVITY_FIXER_RELATIVE_LIMIT = 5.0e-3


def fixer_absorption_limits(precision: str) -> dict[str, float]:
    try:
        return _FIXER_ABSORPTION_LIMITS[precision]
    except KeyError:
        raise ValueError(
            f"no measured fixer-absorption envelope for precision "
            f"{precision!r}; measure one before running it"
        ) from None


def vertical_grid_receipt(cfg: ArwenGlobalConfig) -> dict[str, object]:
    """The vertical grid a run states about itself in its receipt."""
    return {"coordinate": cfg.vertical_coordinate, **cfg.vertical.describe()}


def vertical_grid_sentence(cfg: ArwenGlobalConfig) -> str:
    grid = vertical_grid_receipt(cfg)
    return (
        f"vertical {grid['coordinate']} nlev={grid['nlev']} "
        f"p_top={grid['p_top_pa'] / 100.0:g} hPa: first full level "
        f"{grid['first_full_level_height_m']:.1f} m AGL at "
        f"{grid['reference_surface_pa'] / 100.0:g} hPa, bottom layer "
        f"{grid['bottom_layer_thickness_pa'] / 100.0:.2f} hPa, "
        f"{grid['full_levels_above_50hpa']} full levels above 50 hPa"
    )


def build_transform(cfg: ArwenGlobalConfig) -> SphericalHarmonicTransform:
    # legendre_band and streaming are the transform's own memory levers.
    # Until this call carried them a config could name either and the
    # transform was built on the defaults regardless, which is the flag
    # parsed and silently ignored, not a lever.
    return SphericalHarmonicTransform.create(
        cfg.truncation,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
        dealias_factor=cfg.dealias_factor,
        backend=cfg.backend,
        precision=cfg.precision,
        tensor_core_contractions=cfg.tensor_core_contractions,
        legendre_band=cfg.legendre_band,
        streaming=cfg.streaming,
    )


def build_diffusion(cfg: ArwenGlobalConfig):
    """The drain the config names: ``None`` when disabled, the shipped
    fixed-shape hyperdiffusion, or the spectral eddy viscosity of closure
    theory (``[diffusion] closure = "spectral_eddy_viscosity"``,
    arwen_global.spectral.eddy_viscosity) with the constants the config
    carries."""
    if not cfg.diffusion_enabled:
        return None
    if cfg.diffusion_closure == "spectral_eddy_viscosity":
        from arwen_global.spectral.eddy_viscosity import SpectralEddyViscosity

        return SpectralEddyViscosity(
            plateau=cfg.closure_plateau,
            cusp_amplitude=cfg.closure_cusp_amplitude,
            cusp_decay=cfg.closure_cusp_decay,
            eddy_prandtl=cfg.closure_eddy_prandtl,
            tail_degrees=cfg.closure_tail_degrees,
            preserve_degree=cfg.diffusion_preserve_degree,
        )
    return ExponentialHyperdiffusion(
        order=cfg.diffusion_order,
        e_folding_time_s_at_truncation=cfg.diffusion_efold_s,
        preserve_degree=cfg.diffusion_preserve_degree,
    )


def build_physics(cfg: ArwenGlobalConfig, backend, clock=None, grid=None):
    """The physics suite, run on the forecast ``clock``
    (arwen_global.clock.ForecastClock).  Without one the clock is resolved
    from the config alone: its stated start_time_utc, else the idealized
    fixture (a native suite with neither is refused by the clock).
    ``grid`` is the model's Gaussian grid, which the orographic drag's
    statistics are checked against (built from the config when absent).
    A world that changes the sun hands it to the suite here, or is refused
    by name (arwen_global.world.radiation.attach_sun); an Earth run never
    reaches world code.  A world that changes the air is refused here under
    a suite that radiates no trace gas (physics.native_options.
    world_air_physics_refusal), and its gases reach the native radiation
    through _build_physics."""
    world = world_of(cfg)
    if not world.is_earth:
        from .physics.native_options import world_air_physics_refusal

        reason = world_air_physics_refusal(world, cfg.physics_mode)
        if reason is not None:
            raise ValueError(reason)
    physics = _build_physics(cfg, backend, clock, grid)
    if not world.is_earth:
        from .world.radiation import attach_sun

        physics = attach_sun(physics, world)
    return physics


def _build_physics(cfg: ArwenGlobalConfig, backend, clock=None, grid=None):
    from .clock import resolve_forecast_clock

    if cfg.physics_mode == "none":
        return None
    if clock is None:
        if cfg.initial_mode == "analysis":
            # THE BREAKAGE THIS PREVENTS: without the analysis's clock the
            # reference suite would radiate on the undated fixture sun and
            # the native bridge on whatever literal the config carries,
            # which is GI-1 again behind a caller that forgot the clock.
            raise ValueError(
                "an analysis-initialised physics suite runs on the analysis "
                "valid time, and build_physics was handed no forecast clock; "
                "build it from the analysis (arwen_global.clock) and pass it")
        clock = resolve_forecast_clock(cfg, None)
    if cfg.physics_mode == "reference":
        return ReferencePhysics(
            backend, cfg.reference_physics,
            start_utc=None if clock is None else clock.start_utc,
        )
    if cfg.physics_mode == "arwen-native":
        # FOUR SIGNATURE REFUSALS STOOD HERE UNTIL 2026-09-09, and they are
        # retired rather than weakened.  Each named one engine callable this
        # suite hands an argument a published 2.7.0 does not accept: the
        # surface layer's vegetation fraction, radiation's effective-size
        # bounding, and the two cumulus constructors' column chunk.  Against
        # such an engine the config was accepted, the card priced and
        # allocated, the first steps integrated, and the argument vector
        # refused inside the physics -- so the refusal was moved forward to
        # here, the first point where the answer is "a native forecast is
        # being built".
        #
        # All four callables are now carried in `arwen_global.core`, cut from
        # the tree this model was graded in.  There is no installed engine
        # that can present a different signature for them, so the question
        # the guard asked has no second answer, and a guard whose defect is
        # fixed teaches a reader to expect a failure that cannot happen.
        # What CAN still move is the engine underneath the carried code: that
        # is the seam manifest's job, and the doctor reports it by file.
        options = dict(cfg.native_adapter_options)
        world = world_of(cfg)
        if not world.is_earth:
            # A world's [world.air] gases (CO2, methane, nitrous oxide) onto
            # the radiation's own options; never into the config, whose hash
            # already carries them through the world's identity.
            from .physics.native_options import world_air_options

            options = world_air_options(options, world)
        if clock is not None and clock.dated:
            # The radiation's declination and hour angle run on the
            # forecast clock: the analysis valid time when the run starts
            # from one (a stated start_time_utc has already been checked
            # equal to it), never a literal the analysis has moved away from.
            options["start_time_utc"] = clock.iso
        bridge = NativeArwenPhysicsBridge(cfg.native_adapter_name or "", options)
        from .physics.orographic_drag import attach_orographic_statistics

        attach_orographic_statistics(bridge.adapter, cfg, grid)
        return bridge
    raise AssertionError(cfg.physics_mode)


def world_of(cfg):
    """The world a config runs on: ``cfg.world``, or Earth for a config
    object built without one (a hand-made test double)."""
    from .world.spec import EARTH

    world = getattr(cfg, "world", None)
    return EARTH if world is None else world


def frame_world_reason(world) -> str | None:
    """Why ``world`` needs an analysis start, or None when it does not.

    A world that moves the terrain, the sea or the coast, or invents its
    surface, is built by editing the analysis frame (the hooks in
    ``analysis_initial.analysis_initial_state``); an analytic or
    baroclinic-wave start has no frame to edit, and running it would label
    an Earth state with that world.  A world that changes only the sun, the
    day length or the air acts in the physics and the dynamics and starts
    from any initial state."""
    if world.is_earth:
        return None
    reasons = []
    if world.changes_terrain:
        reasons.append("its terrain")
    if world.changes_sea:
        reasons.append("its sea")
    if world.is_invented:
        reasons.append("its invented surface")
    # Not the rotation: its re-balanced start (packages 2.2/3.2) edits an
    # analysis frame when there is one, and an idealized start of another
    # rotation (package 1.8's ladder) is a test case's own business.
    if not reasons:
        return None
    return ", ".join(reasons)


def adopt_checkpoint_clock(model, state, label: str) -> None:
    """Hold a state read from a checkpoint to the model's forecast clock:
    refused by name when the checkpoint was integrated from another start
    (arwen_global.clock.check_checkpoint_clock), stamped with the clock when
    it was written before checkpoints carried one, so every checkpoint the
    resumed run writes is dated like the uninterrupted run's.

    The same door holds the state to the model's WORLD
    (checkpoint.check_checkpoint_world): every reader that rebuilds a model
    from a config and adopts a checkpoint (a restart, a cycle, a tape or a
    parent export) is refused by name when the checkpoint was integrated on
    another world.  A checkpoint with no world record reads as Earth."""
    from .checkpoint import check_checkpoint_world
    from .clock import FORECAST_CLOCK_KEY, check_checkpoint_clock

    world = getattr(model, "world_spec", None)
    if world is not None:
        check_checkpoint_world(state.physics_state.metadata, world, label)
    clock = getattr(model, "forecast_clock", None)
    if clock is None:
        return
    metadata = state.physics_state.metadata
    check_checkpoint_clock(metadata, clock, label)
    if clock.dated and FORECAST_CLOCK_KEY not in metadata:
        # Not silent: the checkpoint names no start of its own, so the
        # config's clock is taken on trust (an analysis file re-fetched
        # for another day under the same path would date it wrongly).
        # The stamp says so in every checkpoint the run writes from it,
        # and the operator is told once.
        metadata[FORECAST_CLOCK_KEY] = {
            **clock.metadata(),
            "stamped_on_checkpoint_without_clock": str(label),
        }
        import sys

        print(
            f"gpuwm-global: {label} carries no forecast clock (written before "
            f"checkpoints carried one); it is taken to start at {clock.iso}, "
            "this config's clock, unchecked",
            file=sys.stderr,
        )


def _card_gather_of(model, transform):
    """The checkpoint's whole-globe assembler, or None on one card."""
    exchange = getattr(model.pipeline, "exchange", None)
    if exchange is None or int(getattr(exchange, "world", 1)) <= 1:
        return None
    from .cards import gather_named_arrays

    grid = transform.grid

    def gather(arrays):
        return gather_named_arrays(
            exchange.session, arrays, grid.nlat, grid.nlon)

    return gather


def _row_exchange(session):
    from .cards import RowExchange

    return RowExchange(session)


def resolve_latitude_bands(cfg: ArwenGlobalConfig) -> int:
    """The band count this run will use: the config's, or the sizer's.

    ``latitude_bands = 0`` (the shipped default) hands the choice to
    :func:`sizing.auto_latitude_bands`, which reads the card's free bytes
    and returns one when the resident run fits and the smallest count
    that fits when it does not.  That is what makes a BARE run at a
    truncation the card cannot hold resident start rather than die in the
    allocator, which is the whole point of the streaming.
    """
    from .sizing import auto_latitude_bands

    declared = int(getattr(cfg, "latitude_bands", 0) or 0)
    if declared > 0:
        return declared
    return auto_latitude_bands(cfg)


def latitude_bands_receipt(model, cfg, sized_by_door: bool, plan=None
                           ) -> dict[str, object]:
    """The band schedule grid space streamed through, for a receipt.

    The count, the rows it cut them into, and who chose it.  It is NOT
    part of any identity (it changes no arithmetic), so it is recorded
    where a reader can see what the run did rather than where a hash
    would refuse a restart for it.

    Every door that sizes a run writes this, because a door that priced a
    band count and a receipt that does not name one is a run nobody can
    check afterwards.

    ``chosen_by`` is ``"fallback"`` when the count is the lighter plan a
    run moved to after the card ran out of memory (:func:`run`).
    """
    if getattr(plan, "bands_chosen_by", None) == "fallback":
        chosen = "fallback"
    elif sized_by_door:
        chosen = "sizer"
    else:
        chosen = ("config" if int(getattr(cfg, "latitude_bands", 0) or 0)
                  else "sizer")
    return {
        **model.pipeline.receipt(),
        # The door writes the count it priced into the config it hands
        # the run, so the config carrying one does not say who chose it;
        # the door says so instead.
        "chosen_by": chosen,
    }


def place_rank_for(cfg) -> dict:
    """Put this rank of a multi-card run on its own card (MG-4).

    Called by the door before the sizer reads the card and by :func:`run`
    before the first device byte; a no-op for one card and for NumPy."""
    from . import cards as cards_module

    return cards_module.place_rank(
        int(getattr(cfg, "card_rank", 0) or 0),
        int(getattr(cfg, "cards", 1) or 1),
        tuple(getattr(cfg, "card_addresses", ()) or ()),
        str(getattr(cfg, "backend", "numpy")),
    )


def open_card_session(cfg: ArwenGlobalConfig, transform, bands: int):
    """The card session this rank runs behind, or the single-card one.

    ONE CARD IS THE DEFAULT and it still gets a real session rather than a
    ``None``: the single-card transport is the multi-card one with a world
    of one, so the step executes one code path and there is nothing that
    only a two-card run exercises.

    The band assignment follows the cards' MEASURED speeds when a per-band
    profile is available and splits the rows evenly when it is not; either
    way it changes who computes a row and never what the row is (gate
    BIT-6).  ``NCCL_IB_DISABLE=1`` is set here as well as by the launcher,
    so a rank started by hand carries it too and the receipt never has to
    guess which transport a rate belongs to.
    """
    from . import cards as cards_module

    world = int(getattr(cfg, "cards", 1) or 1)
    if world <= 1:
        return cards_module.single_card_session(transform.grid.nlat, bands)
    axis = str(getattr(cfg, "card_axis", "band"))
    if axis == "order":
        raise ValueError(
            "memory.card_axis='order' is not built for a model RUN.  The "
            "order axis is the CAPACITY axis: it partitions the Legendre "
            "orders so each card holds a fraction of the table, which is the "
            "only path past the whole-table wall (design section 10), and it "
            "is proven bit-identical to one card at the transform "
            "(SphericalHarmonicTransform.forward_orders / inverse_orders, "
            "gate ORDER-BIT).  Wiring it end-to-end through the model step "
            "means running grid space, physics and the FFT whole on every "
            "card, and it buys nothing until a truncation whose table does "
            "not fit one card is wanted -- which no card in the building "
            "runs.  Use the 'band' axis, which serves every truncation the "
            "hardware fits and is the speed axis"
        )
    for key, value in cards_module.launch_environment().items():
        os.environ.setdefault(key, value)
    rank = int(getattr(cfg, "card_rank", 0) or 0)
    addresses = tuple(getattr(cfg, "card_addresses", ()) or ())
    # One rank, one card (MG-4).  The door placed this rank before the
    # sizer read the card; asked again here for a library caller that
    # skipped the door, and it returns the door's decision unchanged.
    placement = cards_module.place_rank(
        rank, world, addresses, transform.backend.name)
    transport = cards_module.open_transport(
        rank, world, addresses,
        prefer=str(getattr(cfg, "card_transport", "auto")),
        device=transform.backend.name == "cupy",
    )
    # Every rank's card, all-gathered before anything else crosses the
    # wire: two ranks on one card are refused here by UUID, by name.
    devices = cards_module.check_card_placement(
        transport, cards_module.card_identity(transform.backend.xp))
    declared = tuple(getattr(cfg, "card_weights", ()) or ())
    band_ms = None
    weights = None
    if declared:
        if len(declared) != world:
            raise ValueError(
                f"{len(declared)} card weights for {world} cards: one weight "
                "per rank, in rank order, or none to measure them"
            )
        weights = tuple(float(v) for v in declared)
    else:
        # The per-band cost profile: every rank times the same band-shaped
        # work on its own card and the ranks all-gather the milliseconds,
        # so the assignment follows what the cards MEASURE rather than a
        # table with no tense.
        rows = max(1, transform.grid.nlat // max(1, bands))
        mine = cards_module.measure_band_cost_ms(transform, rows)
        pieces = transport.all_gather(
            "band-profile", struct.pack("!d", float(mine)))
        band_ms = tuple(
            struct.unpack("!d", piece)[0] for piece in pieces
        )
    session = cards_module.CardSession(
        transport, transform.grid.nlat, bands,
        exchange=str(getattr(cfg, "card_exchange", "gather")),
        weights=weights, band_ms=band_ms,
        agreement=str(getattr(cfg, "card_agreement", "refuse")),
    )
    session.devices = devices
    session.placement = placement
    # A gather run assembles a Fourier waist out of latitude rows computed
    # on both cards, so it reproduces the single-card answer only where the
    # cards return the same bits (lane 6's finding).  The check rides the
    # run's own contractions (cards.ContractionAgreement, called by the
    # transform once per distinct shape) rather than a probe at fixed
    # widths here: the fixed-width probe refused this pair at an
    # eight-plane synthesis the T255 L40 step never presents, while every
    # contraction the step does present agrees (MEASURED 2026-09-07).
    return session


def books_open_water_exchange(cfg: ArwenGlobalConfig) -> bool:
    """Whether a run books open water to the exchange account: the native
    suite does, whose kernels never read the surface reservoir.  The
    reference suite caps one call's evaporation by its reservoir and keeps
    it, and a run without physics exchanges no surface water.  The native
    option ``open_water_exchange = false`` keeps the finite reservoir, the
    bookkeeping of 0.1.8 and earlier."""
    if cfg.physics_mode != "arwen-native":
        return False
    return bool(dict(cfg.native_adapter_options or {}).get(
        "open_water_exchange", True))


def carried_physics_arrays(cfg: ArwenGlobalConfig) -> frozenset[str]:
    """The arrays of ``checkpoint.LATER_CARRIED_PHYSICS_ARRAYS`` a run of
    ``cfg`` carries between calls: Noah's snow age under native
    ``snow_ageing = true``, nothing otherwise.  A restart archive that
    lacks one of these is a gap the receipt and stderr name; one that lacks
    an array the run does not carry is missing nothing."""
    if cfg.physics_mode != "arwen-native":
        return frozenset()
    from .physics.native_options import NativePhysicsOptions
    from .physics.native_state import NOAH_SNOW_AGE

    ageing = dict(cfg.native_adapter_options or {}).get(
        "snow_ageing", NativePhysicsOptions.snow_ageing)
    return frozenset({NOAH_SNOW_AGE}) if ageing is True else frozenset()


def _world_spin(cfg: ArwenGlobalConfig, world) -> dict[str, float]:
    """The dynamics' rotation rate under a world whose day is not 24 hours
    (``arwen_global.world.clock.PlanetClock``), and the radiation interval
    that restores the config's own degrees of sun per call, said once on
    stderr when the configured one moves the sun further."""
    import sys

    from .world.clock import PlanetClock

    planet = PlanetClock.from_world(world)
    if planet is None:
        return {}
    if cfg.physics_mode == "arwen-native":
        from .physics.native_options import NativePhysicsOptions

        interval = dict(cfg.native_adapter_options or {}).get(
            "radiation_interval_s", NativePhysicsOptions.radiation_interval_s)
        advice = planet.radiation_interval_advice(float(interval), float(cfg.dt_s))
        if advice is not None:
            print(f"gpuwm-global: world {world.label!r}: {advice['sentence']}",
                  file=sys.stderr)
    return {"rotation_rate_s": planet.rotation_rate_s}


def build_model_and_cold_state(cfg: ArwenGlobalConfig, transform=None, *,
                               scratch_destination: str | Path | None = None,
                               card_session=None):
    """The model and its cold state.  ``scratch_destination`` is the output
    directory of the work being built (a run's outdir, an export's
    output): the analysis route's GRIB decode stages its frame stream
    beside it rather than in the system temp."""
    transform = transform or build_transform(cfg)
    bands = resolve_latitude_bands(cfg)
    session = card_session
    if session is None and int(getattr(cfg, "cards", 1) or 1) > 1:
        session = open_card_session(cfg, transform, bands)
    world = world_of(cfg)
    needs_frame = frame_world_reason(world)
    if needs_frame is not None and cfg.initial_mode != "analysis":
        raise ValueError(
            f"world {world.label!r} changes {needs_frame}, which is built by "
            "editing the analysis frame, and this config starts from "
            f"initial mode {cfg.initial_mode!r}, which has no frame: set "
            "[initial] mode = \"analysis\" (a world that changes only the sun "
            "or the air starts from any initial state)")
    if cfg.initial_mode == "analysis":
        from .analysis_initial import analysis_initial_state

        cold, surface_geopotential, initial_provenance = analysis_initial_state(
            cfg, transform, scratch_destination=scratch_destination
        )
    elif cfg.initial_mode == "baroclinic_wave":
        from .testcases import baroclinic_wave_initial_state

        cold, surface_geopotential, initial_provenance = (
            baroclinic_wave_initial_state(cfg, transform)
        )
    else:
        cold, surface_geopotential, initial_provenance = analytic_initial_state(
            cfg, transform
        )
    # The forecast clock (arwen_global.clock): the analysis valid time
    # when the run starts from one, refused by name when a stated physics
    # start_time_utc disagrees with it.  Resolved before the physics is
    # built so the radiation can only ever run on the analysis's sun.
    from .clock import FORECAST_CLOCK_KEY, resolve_forecast_clock

    clock = resolve_forecast_clock(cfg, initial_provenance)
    if clock.dated:
        # Every checkpoint of this run carries the clock it was integrated
        # under, so a restart or a render is checked against it.
        cold.physics_state.metadata[FORECAST_CLOCK_KEY] = clock.metadata()
    if not world.is_earth:
        # The world, on the clock's precedent: every checkpoint of this run
        # carries the record of the world it was integrated on, so a
        # restart, a cycle or a render under another world is refused by
        # name (checkpoint.check_checkpoint_world).  Earth stamps nothing,
        # so its checkpoints are byte-identical to those before worlds.
        from .checkpoint import WORLD_RECORD_KEY

        cold.physics_state.metadata[WORLD_RECORD_KEY] = world.record()
    diffusion = build_diffusion(cfg)
    # A world with another day length spins the dynamics at the rate its
    # sun's hour angle is derived from (arwen_global.world.clock, one
    # function for both).  An Earth run passes nothing and keeps the
    # model's own rotation rate.
    spin = {}
    if world.changes_rotation:
        spin = _world_spin(cfg, world)
    model = MoistHybridModel(
        **spin,
        transform=transform,
        vertical=cfg.vertical,
        surface_geopotential=surface_geopotential,
        physics=build_physics(cfg, transform.backend, clock, transform.grid),
        physics_split=cfg.physics_split,
        integrator=cfg.integrator,
        diffusion=diffusion,
        divergence_diffusion_strength=cfg.divergence_diffusion_strength,
        pressure_diffusion_strength=cfg.pressure_diffusion_strength,
        water_diffusion_strength=cfg.water_diffusion_strength,
        semi_implicit=build_semi_implicit(
            cfg.semi_implicit_scheme,
            enabled=cfg.semi_implicit_enabled,
            weight=cfg.semi_implicit_weight,
            external_wave_speed_m_s=cfg.external_wave_speed_m_s,
            reference_temperature_k=cfg.semi_implicit_reference_temperature_k,
            reference_surface_pressure_pa=cfg.semi_implicit_reference_surface_pressure_pa,
            off_centring_weight=cfg.semi_implicit_off_centring_weight,
        ),
        mass_fixer=cfg.mass_fixer,
        water_fixer=cfg.water_fixer,
        open_water_exchange=books_open_water_exchange(cfg),
        positivity_repair=cfg.positivity_repair,
        maximum_cfl=cfg.maximum_cfl,
        minimum_fold_determinant=cfg.minimum_fold_determinant,
        semilag=cfg.semilag,
        sponge_base_pa=cfg.sponge_base_pa,
        sponge_lid_relaxation_time_s=cfg.sponge_lid_relaxation_time_s,
        sponge_heat_booking=cfg.sponge_heat_booking,
        sponge_mean_drag_base_pa=cfg.sponge_mean_drag_base_pa,
        sponge_mean_drag_time_s=cfg.sponge_mean_drag_time_s,
        initial_provenance=initial_provenance,
        spectral_chunk=cfg.spectral_chunk,
        synthesis_memo=cfg.synthesis_memo,
        skip_unread_planes=cfg.skip_unread_planes,
        divergence_only_mass_flux=cfg.divergence_only_mass_flux,
        latitude_bands=bands,
        cards=None if session is None else _row_exchange(session),
        card_halo_rows=int(getattr(cfg, "card_halo_rows", 16)),
    )
    model.forecast_clock = clock
    # Not "world": the card exchange already uses that word for its rank count.
    model.world_spec = world
    cold, _, _, _ = model._repair_positivity(cold)
    model.initialize_mass_target(cold.atmosphere)
    model.initialize_water_target(cold)
    model.enforce(cold)
    cold = attach_host_tier(cfg, model, cold)
    return model, cold


def attach_host_tier(cfg: ArwenGlobalConfig, model, bundle):
    """Give ``model`` its pinned host tier and park what it holds.

    The slices are chosen by the MINIMUM-SPILL rule against the peak the
    run will actually take -- the band count's peak, not the resident one
    -- and the census is read off the state's own arrays rather than
    fitted (``sizing.spill_census``).  With nothing to park this returns
    the bundle unchanged and the model keeps no tier, so a run that does
    not need the tier is the run that shipped before it existed.
    """
    from .sizing import SPILL_SLICES, plan_run_memory, spill_census
    from .spill import HostTier

    mode = str(getattr(cfg, "host_spill", "auto"))
    if mode == "off":
        return bundle
    if getattr(model, "host_tier", None) is not None:
        # A second call (the restart path) parks into the tier the builder
        # already made, so the slots the cold state opened are the slots
        # the restart writes rather than a second tier's.
        return model.park_persistent(bundle)
    census = spill_census(bundle, cfg)
    if mode == "on":
        slices = tuple(n for n in SPILL_SLICES if int(census.get(n, 0)) > 0)
    else:
        # THE SAME FUNCTION THE DOOR USED, with a better census: the door
        # priced the three slices from the estimate's terms before an
        # array existed, and this reads them off the arrays themselves.
        # Two policies here and at the door is how a run gets sized for
        # one plan and built for another.
        free = _free_device_bytes_for_spill(cfg)
        plan = plan_run_memory(cfg, free, census=census)
        slices = plan.spill_slices
    if not slices:
        return bundle
    model.host_tier = HostTier(model.transform.backend.xp)
    model.host_tier.slices = list(slices)
    model.spill_slices = tuple(slices)
    return model.park_persistent(bundle)


def _host_spill_receipt(cfg, model, *, wall_steps=0, wall_seconds=0.0,
                        plan=None):
    """What the pinned host tier held, moved, and cost: gate OVERLAP-1.

    ``transfer_share`` is the time the tier's own streams spent copying,
    measured with CUDA events around every transfer, over the wall the
    steps took.  A spill configuration whose transfers are NOT hidden
    reports it here rather than running a campaign fifteen percent slow
    with nothing in the receipt to say why.
    """
    # WHO CHOSE IT, and the plan is asked first.  The door writes the mode
    # it priced into the config it hands the run, so a config reading
    # "off" after the door has spoken does NOT mean a person asked for
    # off; the plan is the only record of which it was.
    chosen_by = (getattr(plan, "spill_chosen_by", None)
                 or ("config"
                     if str(getattr(cfg, "host_spill", "auto")) != "auto"
                     else "sizer"))
    tier = getattr(model, "host_tier", None)
    if tier is None:
        return {
            "mode": str(getattr(cfg, "host_spill", "auto")),
            "chosen_by": chosen_by,
            "slices": [],
            "parked_gib": 0.0,
            "reason": "nothing parked: the run fits the card without the tier",
        }
    receipt = dict(tier.receipt())
    receipt["mode"] = str(getattr(cfg, "host_spill", "auto"))
    receipt["chosen_by"] = chosen_by
    steps = int(wall_steps or receipt.get("steps", 0) or 0)
    seconds = float(wall_seconds or 0.0)
    receipt["step_wall_seconds"] = round(seconds, 6)
    receipt["steps_timed"] = steps
    if steps and seconds > 0.0:
        step_s = seconds / steps
        receipt["step_s"] = round(step_s, 6)
        receipt["transfer_share"] = round(
            (receipt.get("transfer_s_per_step", 0.0)) / step_s, 6
        ) if step_s > 0 else None
    return receipt


def _step_wall_receipt(steps: int, seconds: float) -> dict:
    """The steps' own wall: the count timed, their seconds, one step's."""
    steps = int(steps or 0)
    seconds = float(seconds or 0.0)
    return {
        "steps_timed": steps,
        "step_wall_seconds": round(seconds, 6),
        "step_s": round(seconds / steps, 6) if steps > 0 else None,
        "measures": ("perf_counter around model.step only: the build, the "
                     "cold start, the diagnostics and the checkpoint "
                     "writes are outside it"),
    }


def _free_device_bytes_for_spill(cfg) -> int | None:
    if getattr(cfg, "backend", "numpy") != "cupy":
        return None
    from .sizing import _free_device_bytes

    return _free_device_bytes()


def _owned_files(outdir: Path) -> list[Path]:
    return [
        outdir / RECEIPT_NAME,
        outdir / DIAGNOSTICS_NAME,
        *sorted(outdir.glob(f"{CHECKPOINT_PREFIX}*.npz")),
        *insitu_owned_files(outdir),
    ]


def _prepare_outdir(
    outdir: Path, overwrite: bool, *, keep: frozenset[Path] = frozenset()
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in _owned_files(outdir) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Arwen Global output exists in {outdir}; pass --overwrite to replace owned files"
        )
    if overwrite:
        for path in existing:
            if path.resolve() in keep:
                continue
            path.unlink()


def _checkpoint_path(outdir: Path, step: int) -> Path:
    return outdir / f"{CHECKPOINT_PREFIX}{int(step):08d}.npz"


def _append_diagnostics(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


# Receipt-only trackers: the checkpoint tracker schema is pinned to eight
# keys, so these accumulate per run segment and are not inherited on restart.
SUPPLEMENTARY_TRACKER_KEYS = (
    "maximum_repaired_negative_number_per_kg",
    "maximum_positivity_fixer_relative",
    "maximum_positivity_fixer_water_kg_m2",
    "maximum_positivity_fixer_rescale",
    # The semi-Lagrangian core's three (zero on every other path).  They
    # are receipt-only for the reason the four above are: the checkpoint
    # tracker schema is pinned to eight keys and normalize_trackers
    # REFUSES any dict whose key set differs, so a ninth entry would make
    # every checkpoint ever written unreadable.
    "maximum_semilag_lipschitz",
    "maximum_semilag_trajectory_move_cells",
    "maximum_semilag_tracer_mass_fixer_relative",
    "maximum_semilag_tracer_mass_fixer_water_relative",
    # The trajectory fold gate's own number, a MINIMUM: it starts at one
    # (the identity map, no step measured) and every other path keeps it
    # there.  Receipt-only, for the reason above.
    "minimum_semilag_fold_determinant",
    # A COUNT: the steps whose departure search missed the convergence
    # test at the configured iterations and was searched again at the
    # most (semilag.trajectory.converged_departure_points).  Receipt-only.
    "semilag_trajectory_retried_steps",
)

#: The value a supplementary tracker starts a run at, where it is not zero.
SUPPLEMENTARY_TRACKER_STARTS = {"minimum_semilag_fold_determinant": 1.0}


def fresh_supplementary() -> dict[str, float]:
    """The supplementary trackers at the start of a run."""
    return {
        name: float(SUPPLEMENTARY_TRACKER_STARTS.get(name, 0.0))
        for name in SUPPLEMENTARY_TRACKER_KEYS
    }


#: The per-species rows the semi-Lagrangian tracer fixer writes each step.
_SPECIES_FIXER_PREFIXES = (
    "semilag_tracer_mass_fixer_relative__",
    "semilag_tracer_mass_fixer_kg_m2__",
    "semilag_tracer_clip_share__",
    # The mass the positivity floor created, which is where the fixer's
    # correction comes from on a species that is zero nearly everywhere.
    # Without this row a receipt says how much was corrected and not what
    # made the correction necessary, and the two arms that separate the
    # limiter from the floor cannot be told apart at all.
    "semilag_tracer_positivity_clamp_kg_m2__",
    # Where the correction landed, which the relative magnitude above
    # cannot say: the uniform factor applied to EVERY point holding the
    # species (the whole correction under 'mass_proportional', 0.116 for
    # graupel in the worst step of the 2026-10-01 receipts; only the
    # remainder under 'bermejo_conde') and the fraction of the species'
    # mass at points changed by more than one percent of their own value.
    "semilag_tracer_fixer_uniform_rescale__",
    "semilag_tracer_fixer_touched_mass_fraction__",
    # The share of the correction the local stage placed (the weighted
    # Bermejo-Conde stage, or the additive form's clip record); the
    # uniform rescale above carries only what it could not place.
    "semilag_tracer_fixer_local_share__",
    # The species' mass after the fix against its target, relative: the
    # conservation the fixer exists for, read back from the field it
    # returned rather than assumed from its construction.
    "semilag_tracer_fixer_residual_relative__",
)


def _greenhouse_gas_receipt(cfg: ArwenGlobalConfig, clock=None) -> dict | None:
    """The native radiation's greenhouse-gas record (source named), or
    None when the run radiates nothing through RRTMGP.

    The gases are dated by the forecast ``clock`` the physics ran on
    (build_physics fills the suite's start_time_utc from it), not by the
    config's literal: a config that follows its analysis states no
    start_time_utc, as the clock's own refusal tells the reader to do, and
    reading the date off the config alone refused every such run at its
    receipt, after the whole forecast had been integrated."""
    if cfg.physics_mode != "arwen-native":
        return None
    from .physics.native_options import NativePhysicsOptions

    raw = dict(cfg.native_adapter_options)
    world = world_of(cfg)
    if not world.is_earth:
        # The gases the radiation was built with: a world's air included, so
        # the receipt names what was radiated.  Earth never reaches world code.
        from .physics.native_options import world_air_options

        raw = world_air_options(raw, world)
    if clock is not None and clock.dated:
        raw["start_time_utc"] = clock.iso
    options = NativePhysicsOptions.from_mapping(raw)
    if options.radiation != "rrtmgp":
        return None
    # The whole run's span (duration_s is in the identity, so every segment
    # of a restarted run states the same schedule).
    return options.greenhouse_gas_record(cfg.duration_s)


class PositivityFixerLedger:
    """The run's account of where the vapor positivity fixer moved water.

    The four vapor clamps of a step (two positivity repairs, two physics
    exchange clamps) clip the spectral ringing of vapor and pay the water
    the clip creates out of the same column, first from the layers next to
    each hole and then from the rest of the column in proportion.  Each
    step's metrics carry the global-mean water the fill added to and took
    from every level; this ledger sums them over the run segment, so the
    receipt and every diagnostics record say how much water the fixer
    moved vertically and across which interfaces, instead of leaving the
    fixer's vertical transfer unattributed.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.created_kg_m2 = 0.0
        self.unfillable_kg_m2 = 0.0
        self.neighbour_kg_m2 = 0.0
        self.gain: list[float] | None = None
        self.loss: list[float] | None = None

    def add(self, metrics: dict[str, object]) -> None:
        gain = metrics.get("positivity_fixer_level_gain_kg_m2")
        loss = metrics.get("positivity_fixer_level_loss_kg_m2")
        self.steps += 1
        self.created_kg_m2 += float(metrics.get("positivity_fixer_water_kg_m2", 0.0))
        self.unfillable_kg_m2 += float(
            metrics.get("positivity_fixer_unfillable_kg_m2", 0.0))
        self.neighbour_kg_m2 += float(
            metrics.get("positivity_fixer_neighbour_kg_m2", 0.0))
        if gain is None or loss is None:
            return
        if self.gain is None:
            self.gain = [0.0] * len(gain)
            self.loss = [0.0] * len(loss)
        self.gain = [a + float(b) for a, b in zip(self.gain, gain)]
        self.loss = [a + float(b) for a, b in zip(self.loss, loss)]

    def record(self) -> dict[str, object]:
        """Run-segment totals, global-mean kg/m2, levels top first.

        ``upward_transfer_kg_m2[i]`` is the water the fill moved upward
        across the interface below level ``i`` (negative: downward), the
        running sum of the net gain from the top down.
        """
        out: dict[str, object] = {
            "steps": self.steps,
            "clip_created_kg_m2": self.created_kg_m2,
            "unfillable_kg_m2": self.unfillable_kg_m2,
            "paid_by_adjacent_layer_kg_m2": self.neighbour_kg_m2,
            "level_order": "top_first",
        }
        if self.gain is None:
            return out
        net = [g - l for g, l in zip(self.gain, self.loss)]
        upward = []
        running = 0.0
        for value in net:
            running += value
            upward.append(running)
        out.update({
            "level_gain_kg_m2": list(self.gain),
            "level_loss_kg_m2": list(self.loss),
            "level_net_kg_m2": net,
            "upward_transfer_kg_m2": upward,
            "maximum_upward_transfer_kg_m2": max(upward),
            "vertically_moved_kg_m2": 0.5 * sum(abs(v) for v in net),
        })
        return out


class StratosphericFloorLedger:
    """The run's bill for the native suite's stratospheric floor.

    The floor adds enthalpy to air above ``stratospheric_floor_pa`` that
    is colder than ``stratospheric_floor_k``.  Each physics call reports
    the area-mean enthalpy it added (J/m2) and how many grid points it
    warmed; this ledger sums the heating and keeps the largest point count
    over the run segment, so the receipt and every diagnostics record carry
    what the floor did instead of only that it was configured.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.active_calls = 0
        self.heating_j_m2 = 0.0
        self.maximum_points = 0.0

    def add(self, metrics: dict[str, object]) -> None:
        for half in ("first_half_physics", "second_half_physics"):
            value = metrics.get(half, {})
            physics = value.get("physics") if isinstance(value, dict) else None
            if not isinstance(physics, dict):
                continue
            heating = float(physics.get("mean_stratospheric_floor_heating_j_m2", 0.0))
            points = float(physics.get("stratospheric_floor_points", 0.0))
            self.calls += 1
            self.heating_j_m2 += heating
            if points > 0.0:
                self.active_calls += 1
            self.maximum_points = max(self.maximum_points, points)

    def record(self) -> dict[str, object]:
        return {
            "physics_calls": self.calls,
            "calls_that_warmed": self.active_calls,
            "area_mean_heating_j_m2": self.heating_j_m2,
            "maximum_points_warmed_in_a_call": self.maximum_points,
        }


def _update_trackers(
    trackers: dict[str, float],
    supplementary: dict[str, float],
    metrics: dict[str, object],
    species: dict[str, float] | None = None,
) -> None:
    trackers["maximum_spectral_cfl"] = max(
        trackers["maximum_spectral_cfl"], float(metrics["spectral_cfl"])
    )
    trackers["maximum_mass_fixer_log_offset"] = max(
        trackers["maximum_mass_fixer_log_offset"],
        abs(float(metrics["mass_fixer_log_offset"])),
    )
    trackers["maximum_global_water_fixer_kg_m2"] = max(
        trackers["maximum_global_water_fixer_kg_m2"],
        abs(float(metrics["global_water_fixer_kg_m2"])),
    )
    trackers["maximum_repaired_negative_mixing_ratio"] = max(
        trackers["maximum_repaired_negative_mixing_ratio"],
        abs(float(metrics["maximum_repaired_negative_mixing_ratio"])),
    )
    supplementary["maximum_repaired_negative_number_per_kg"] = max(
        supplementary["maximum_repaired_negative_number_per_kg"],
        abs(float(metrics["maximum_repaired_negative_number_per_kg"])),
    )
    for key, metric in (
        ("maximum_positivity_fixer_relative", "positivity_fixer_relative"),
        ("maximum_positivity_fixer_water_kg_m2", "positivity_fixer_water_kg_m2"),
        ("maximum_positivity_fixer_rescale", "positivity_fixer_max_rescale"),
        ("maximum_semilag_lipschitz", "semilag_lipschitz"),
        ("maximum_semilag_trajectory_move_cells",
         "semilag_trajectory_move_max_cells"),
        ("maximum_semilag_tracer_mass_fixer_relative",
         "semilag_tracer_mass_fixer_relative"),
        ("maximum_semilag_tracer_mass_fixer_water_relative",
         "semilag_tracer_mass_fixer_water_relative"),
    ):
        supplementary[key] = max(
            supplementary[key], abs(float(metrics.get(metric, 0.0)))
        )
    supplementary["minimum_semilag_fold_determinant"] = min(
        supplementary["minimum_semilag_fold_determinant"],
        float(metrics.get("semilag_fold_determinant", 1.0)),
    )
    supplementary["semilag_trajectory_retried_steps"] += float(
        metrics.get("semilag_trajectory_retried", 0.0))
    # Per species, so a receipt that reports a large correction says which
    # species carried it.  A run that moves four percent of the graupel and
    # nothing else per step is a different finding from one that moves four
    # percent of the cloud water, and the single maximum cannot tell them
    # apart.  Receipt-only, for the reason in SUPPLEMENTARY_TRACKER_KEYS.
    if species is not None:
        for name, value in metrics.items():
            if name.startswith(_SPECIES_FIXER_PREFIXES):
                species[name] = max(species.get(name, 0.0), abs(float(value)))
    trackers["maximum_semi_implicit_divergence_increment_s1"] = max(
        trackers["maximum_semi_implicit_divergence_increment_s1"],
        abs(float(metrics["semi_implicit_max_divergence_increment_s1"])),
    )
    repairs = []
    for half in ("first_half_physics", "second_half_physics"):
        value = metrics.get(half, {})
        if isinstance(value, dict):
            physics = value.get("physics", {})
            if isinstance(physics, dict):
                repairs.append(float(physics.get("maximum_local_water_repair_kg_m2", 0.0)))
    trackers["maximum_physics_water_repair_kg_m2"] = max(
        trackers["maximum_physics_water_repair_kg_m2"],
        *(repairs or [0.0]),
    )
    native_water = []
    native_energy = []
    for half in ("first_half_physics", "second_half_physics"):
        value = metrics.get(half, {})
        if isinstance(value, dict):
            physics = value.get("physics", {})
            if isinstance(physics, dict):
                native_water.append(float(
                    physics.get("maximum_native_water_residual_kg_m2", 0.0)
                ))
                native_energy.append(float(
                    physics.get("maximum_native_energy_change_j_m2", 0.0)
                ))
    trackers["maximum_native_water_residual_kg_m2"] = max(
        trackers["maximum_native_water_residual_kg_m2"],
        *(native_water or [0.0]),
    )
    trackers["maximum_native_energy_residual_j_m2"] = max(
        trackers["maximum_native_energy_residual_j_m2"],
        *(native_energy or [0.0]),
    )


#: Per-step correction the semi-Lagrangian tracer fixer may make, as a
#: fraction of the atmosphere's OWN column water.
#:
#: This is the number the breakage is about.  A tracer scheme that is not
#: conservative silently loses or gains water over a forecast day, which
#: is the class of defect the grid tracers were introduced to retire, and
#: what says whether that is happening is how much water the fixer has to
#: put back, not how large that is beside one trace species' own mass.
#:
#: Priced from measurement, not from an estimate.  MEASURED 2026-09-06,
#: the 16 GB host RTX 5070 Ti, T255 L40 with the full native suite at dt = 300 s:
#: 3.44e-5 over a whole forecast day (288 steps) from the GDAS 2026-09-01
#: 00Z analysis, and 2.17e-5 over the six-hour arms, which run the OTHER
#: case this tree carries (GDAS 2026-08-30 18Z,
#: configs/verify/arwen_global_gdas_t255_native_sl_si_6h.toml) -- two
#: initial states, named because a reading of six hours and a reading of a
#: day here are not two points on one run.  The six-hour number is the same
#: 2.17e-5 whichever form the fixer runs, because it is the ADVECTION's
#: error and not the fixer's.  The limit is 1e-4, three times the day's
#: reading.
#:
#: It interlocks with the two water gates rather than duplicating them.  A
#: run sitting at this limit with every correction the same sign would move
#: 2.9 percent of the column over 288 steps, which
#: ``total_water_relative_drift`` (1e-3) would refuse; a run that hid it in
#: the surface reservoir instead would be refused by
#: ``water_fixer_max_step_relative``.  What this row adds is the reading
#: BEFORE either of those absorbs it, so a scheme whose water error is
#: growing is visible while it is still being closed.
SEMILAG_TRACER_FIXER_WATER_RELATIVE_LIMIT = 1.0e-4

#: Per-step relative mass the fixer may move within ONE species.  This is
#: an accuracy diagnostic of the advection, not a conservation gate: the
#: fixer closes the mass either way, and what a large value says is that
#: the species' field is being shaped by the fixer's weighting as much as
#: by the flow.
#:
#: Read what the number is before reading the limit.  ``fix_mass``
#: computes it as ``|mass(before) - mass(advected)| / mass(before)`` on
#: the field the gather produced and BEFORE any correction is applied,
#: and it computes it for every scheme including ``"none"``.  So it
#: measures the TRANSPORT's mass error and no choice of fixer can move
#: it: the multiplicative form this module ships is, on a non-negative
#: field, algebraically the additive Bermejo-Conde form with
#: mass-proportional weights (``q (1 + d/W) == q + d q/W``), so "take the
#: additive form instead" is not a remedy for this number, it is the same
#: operator.
#:
#: Priced from measurement.  The 1e-4 of the specification of record was an
#: estimate written before the scheme ran; it is off by three orders of
#: magnitude and this is what the arms read.
#:
#: MEASURED 2026-09-06, the 16 GB host, T255 L40 native: 1.01e-1 for graupel over
#: the forecast day (GDAS 2026-09-01 00Z) and 5.96e-2 to 6.35e-2 over the
#: six-hour arms (GDAS 2026-08-30 18Z, the other case), the six-hour
#: reading the same to two figures under every fixer form AND with the
#: quasi-monotone limiter turned off (6.43e-2).  What the number measures
#: is what POSITIVITY costs, not what the limiter costs: a cubic through a
#: field that is zero at three of its four stencil points undershoots
#: below zero on the shoulder of every maximum, something has to lift it
#: back, and turning the limiter off only moves that lift from the
#: gather's clip to the fixer's own floor.  Turning both off stops it and
#: the run is refused at its first step, on a cloud water of -1.49e-5
#: against a maximum of 1.02e-3.
#:
#: The limit is 0.25.  Named breakage: a species whose per-step correction
#: approaches its own mass is a species the fixer is placing rather than
#: the flow transporting, and a quarter of it per step is where that
#: begins to be true over the few steps a convective species lives in one
#: column.  2.5 times the day's worst reading.
#:
#: What the number is a function of, so the headroom is not read as an
#: invariant.  MEASURED 2026-09-06 on the numpy specification, a synthetic
#: condensate blob displaced by a third of a cell in each direction on a
#: 40-level stack: the correction falls from 1.35e-1 for a species ONE
#: model layer deep to 1.9e-2 for one ten layers deep, and from 1.16e-1
#: for a blob two cells across to 2.6e-2 for one ten cells across, while
#: the UNLIMITED gather's own mass error stays at 4.6e-3 throughout.  So
#: the reading is what positivity costs FOR A SPECIES A FEW CELLS ACROSS
#: AND ONE TO THREE LAYERS DEEP, which is what a condensate species is at
#: T255 on forty levels; it is not a property of the scheme alone, and a
#: run at a sharper truncation or on a thinner ladder reads higher.  The
#: 2.5x headroom above the T255 L40 day is headroom AT THAT OPERATING
#: POINT and the limit owes a re-reading at T533 and at T799.
SEMILAG_TRACER_FIXER_RELATIVE_LIMIT = 0.25


def cfl_headroom_receipt(cfg: ArwenGlobalConfig, trackers: dict[str, float]) -> dict[str, object]:
    """What the day's strongest wind left of the step, written into every
    receipt so a refusal on a stronger day is explained by the numbers.

    The gate is the spectral CFL dt |V|max sqrt(N(N+1))/a against
    ``[time] maximum_cfl``; on the Eulerian path it refuses, on the
    semi-Lagrangian path it is measured and the trajectory fold gate
    refuses instead.  The block carries the day's maximum, its fraction of the
    gate, the wind it implies, the largest step this day would have
    admitted at the gate and at the shipped rule, and the shipped step of
    this truncation so a reader sees whether the run took it.
    """
    maximum = float(trackers.get("maximum_spectral_cfl", 0.0))
    dt = float(cfg.dt_s)
    gate = float(cfg.maximum_cfl)
    truncation = int(cfg.truncation)
    factor = math.sqrt(truncation * (truncation + 1.0)) / EARTH_RADIUS_M
    rate = maximum / dt if dt > 0.0 else 0.0
    semi_lagrangian = cfg.integrator in SEMILAG_INTEGRATORS
    rule = DEFAULT_EULERIAN_STEP_RULE_FRACTION
    try:
        shipped = default_eulerian_step_s(truncation, gate)
    except ValueError:
        # A gate no candidate step satisfies (a test's near-zero gate, or a
        # truncation past the rule's reach): the receipt still carries the
        # day's own numbers, and the shipped step reads as none.
        shipped = None
    block: dict[str, object] = {
        "role": (
            "measurement: the semi-Lagrangian path has no advective bound, "
            "the trajectory fold gate refuses instead"
            if semi_lagrangian else "refusal: the Eulerian path refuses above the gate"
        ),
        "dt_s": dt,
        "truncation": truncation,
        "gate": gate,
        "maximum_spectral_cfl": maximum,
        "fraction_of_gate": maximum / gate if gate > 0.0 else None,
        "headroom_fraction_of_gate": 1.0 - maximum / gate if gate > 0.0 else None,
        "implied_maximum_wind_m_s": rate / factor if factor > 0.0 else None,
        "largest_step_at_gate_s": gate / rate if rate > 0.0 else None,
        "largest_step_at_shipped_rule_s": rule * gate / rate if rate > 0.0 else None,
        "shipped_rule": {
            "fraction_of_gate": rule,
            "wind_ceiling_m_s": eulerian_wind_ceiling_m_s(truncation),
            "shipped_eulerian_step_s": shipped,
            "this_run_took_the_shipped_step": (
                shipped is not None and not semi_lagrangian and abs(dt - shipped) < 1.0e-9
            ),
        },
    }
    if semi_lagrangian:
        block["sentence"] = (
            f"spectral CFL reached {maximum:.3f} at dt {dt:g} s on the semi-Lagrangian "
            f"path (no advective bound; the trajectory fold gate refuses instead); the "
            f"Eulerian core would have admitted dt <= {block['largest_step_at_gate_s']:.1f} s "
            f"on this day" if rate > 0.0 else "no CFL reading"
        )
    else:
        block["sentence"] = (
            f"spectral CFL reached {maximum:.3f} of the {gate:g} gate at dt {dt:g} s "
            f"({100.0 * maximum / gate:.0f} percent, implied maximum wind "
            f"{block['implied_maximum_wind_m_s']:.1f} m/s); this day admits dt <= "
            f"{block['largest_step_at_gate_s']:.1f} s at the gate and <= "
            f"{block['largest_step_at_shipped_rule_s']:.1f} s at the shipped {rule:g}-of-gate "
            f"rule, whose step at T{truncation} is "
            + ("none (no candidate step satisfies this gate)" if shipped is None else f"{shipped:g} s")
            if rate > 0.0 else "no CFL reading"
        )
    return block


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


class _CheckpointWriter:
    """Checkpoints published off the model thread.

    The device-to-host copy of the state (``bundle_arrays``) runs on the
    caller's thread, so the arrays are the step's own; the hashing,
    compression, fsync and rename (``write_checkpoint_arrays``, the same
    function and the same bytes as a synchronous write) run on one
    worker while the next steps integrate.  At most one write is in
    flight: the next submission first joins the previous one, so a
    failed write raises on the model thread at the next checkpoint or
    at :meth:`close`, never silently.  Why: a checkpoint's zlib and
    SHA-256 passes held the card idle on the model thread (the step
    profiler's ``checkpoint`` section, 2026-09-04).

    ``snapshot`` (``checkpoint.PinnedSnapshot``, ``[output] snapshot_buffer
    = "pinned"`` on the card) is the one page-locked buffer the copy lands
    in.  The join at the top of :meth:`submit` is what makes one buffer
    safe: the previous write has finished reading its views before the
    next copy overwrites them.  ``[output] checkpoint_compression`` picks
    stored or deflated members; the arrays and their hashes are the same.
    """

    def __init__(self, cfg, to_numpy, trajectory=None, card_gather=None,
                 snapshot=None):
        self.cfg = cfg
        self.snapshot = snapshot
        self.compression = str(getattr(cfg, "checkpoint_compression", "deflate"))
        self.to_numpy = to_numpy
        # A callable, read at submit time: the second time level of a
        # two-time-level integrator is replaced every step, and the
        # archive must carry the one that belongs to the state it holds.
        self.trajectory = trajectory
        # The multi-card row gather runs on the CALLER's thread with the
        # device-to-host copy, never on the worker: it is a collective and
        # both cards have to reach it in the same order as every other
        # exchange of the step.
        self.card_gather = card_gather
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None
        self.last_arrays = None
        # Called on the caller's thread with each archive's host arrays and
        # record before the write is handed off (the long-run products,
        # arwen_global.longrun_ops); the arrays are the checkpoint's own.
        self.on_arrays = None

    def submit(self, path: Path, state, trackers: dict, *, want_metadata: bool = False):
        """Publish ``state`` to ``path`` off the model thread; returns the
        path, or ``(path, metadata)`` with the record the archive will
        carry (``checkpoint.checkpoint_metadata``, computed here on the
        caller's thread from the same arrays) when ``want_metadata`` is
        set: the cycle door names a background by its ``self_sha256``
        before the write has finished."""
        self.join()
        arrays = bundle_arrays(
            state, self.to_numpy,
            None if self.trajectory is None else self.trajectory(),
            card_gather=self.card_gather, snapshot=self.snapshot,
        )
        cfg = self.cfg
        record = dict(
            step=int(state.step), time_s=float(state.time_s),
            physics_state_schema=state.physics_state.schema,
            physics_metadata=json.loads(json.dumps(
                state.physics_state.metadata, sort_keys=True, allow_nan=False,
            )),
            config_hash=cfg.config_hash, trackers=dict(trackers),
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        metadata = checkpoint_metadata(arrays, **record) if want_metadata else None
        if self.on_arrays is not None:
            self.on_arrays(path, arrays, record)
        self._pending = self._executor.submit(
            write_checkpoint_arrays, path, arrays, metadata=metadata,
            compression=self.compression, **record,
        )
        # The host arrays this checkpoint is written from, for a tape of the
        # same time (_TapeWriter): read only, by both workers.
        self.last_arrays = arrays
        return (path, metadata) if want_metadata else path

    def join(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        try:
            self.join()
        finally:
            self._executor.shutdown(wait=True)


@dataclass(frozen=True)
class TapeRequest:
    """``run --tapes DIR --tape-nlat N --tape-nlon M``: the wrfout tape of
    every output time, written by the run itself beside the forecast.
    Outside the config identity, like ``--until-s``."""

    outdir: Path
    nlat: int = 360
    nlon: int = 720


def tape_refusal(cfg, model) -> str | None:
    """Why this run cannot write its own tapes, or None.

    The device half of a tape synthesizes from the state where it lives;
    a state parked in the pinned host tier, or spread over several cards,
    is not where that synthesis reads it, so such a run leaves its tapes
    to ``export`` as before.
    """
    if getattr(model, "host_tier", None) is not None:
        return ("this run parks state in the pinned host tier "
                f"(slices {list(getattr(model, 'spill_slices', ()))})")
    exchange = getattr(model.pipeline, "exchange", None)
    if exchange is not None and int(getattr(exchange, "world", 1)) > 1:
        return "this run spans more than one card"
    return None


class _TapeWriter:
    """The run's own wrfout tapes (``run --tapes``), written off the model
    thread.

    At an output time the model thread synthesizes the six spectral fields
    a tape needs (``TapeGrid.synthesize``: on the device, copied off it one
    at a time; the state is only read) and hands them, with the host
    arrays the checkpoint of that time is written from, to one worker.
    The worker rebuilds the checkpoint's bundle from those arrays on the
    host and runs the export's own host half (``TapeGrid.write``): the
    same function on the same values as ``export`` of that checkpoint, so
    the same bytes, and nothing of the export's own process (no second
    model, no cold state, no checkpoint read, no second card context).

    At most one tape is in flight.  A tape that fails is a printed line and
    a receipt row, never the forecast's failure: its checkpoint is on disk
    and ``export`` writes it.  The tape is published under a temporary
    name and renamed (``WrfoutWriter``), so a folder watcher sees only
    whole files.
    """

    def __init__(self, cfg, model, transform, request: TapeRequest):
        import numpy as np

        from .spectral.backend import Backend
        from .wrfout_export import TapeGrid, _require_tape_writer

        # Refused here, before the first step, rather than at the first
        # output time: no tape writer, no render kernels, or an undated run
        # (tape_start asks for the date the export would be given).
        _require_tape_writer()
        self.grid = TapeGrid(cfg, model, transform,
                             nlat=int(request.nlat), nlon=int(request.nlon))
        self.outdir = Path(request.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.request = request
        backend = transform.backend
        self.host_backend = Backend(
            "numpy", np, backend.float_dtype, backend.complex_dtype)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None
        self.rows: list[dict] = []
        self.failures: list[dict] = []
        self.synthesis_seconds = 0.0

    def submit(self, state, arrays, checkpoint) -> None:
        """Synthesize ``state``'s fields here, and queue its tape."""
        self.join()
        started = time.perf_counter()
        step = int(state.step)
        try:
            fields = self.grid.synthesize(state.atmosphere)
            record = {
                "time_s": float(state.time_s),
                "step": step,
                "physics_state_schema": state.physics_state.schema,
                # The checkpoint's own record of it (_CheckpointWriter).
                "physics_metadata": json.loads(json.dumps(
                    state.physics_state.metadata, sort_keys=True,
                    allow_nan=False)),
            }
        except Exception as exc:  # noqa: BLE001 - a tape never ends the forecast
            self._failed(step, checkpoint, exc)
            return
        synthesis = time.perf_counter() - started
        self.synthesis_seconds += synthesis
        self._pending = self._executor.submit(
            self._write, record, arrays, fields, checkpoint, synthesis)

    def _write(self, record, arrays, fields, checkpoint, synthesis):
        started = time.perf_counter()
        step = int(record["step"])
        try:
            bundle = state_from_checkpoint(record, arrays, self.host_backend)
            path, row = self.grid.write(
                bundle, fields, record["time_s"], self.outdir,
                overwrite=True, checkpoint=checkpoint)
        except Exception as exc:  # noqa: BLE001 - a tape never ends the forecast
            self._failed(step, checkpoint, exc)
            return
        wall = time.perf_counter() - started
        row = {**row, "step": step,
               "synthesis_s": round(synthesis, 4), "host_s": round(wall, 4)}
        self.rows.append(row)
        # The export's receipt, rewritten after every tape so the folder
        # always describes the tapes it holds.
        try:
            write_receipt(self.outdir / _export_receipt_name(),
                          self.grid.receipt(self.rows))
        except Exception as exc:  # noqa: BLE001
            self._failed(step, checkpoint, exc)
            return
        print(f"tape={Path(row['tape']).name} step={step} "
              f"synthesis={synthesis:.2f}s host={wall:.2f}s", flush=True)

    def _failed(self, step, checkpoint, exc) -> None:
        row = {"step": int(step), "checkpoint": str(checkpoint),
               "error_type": type(exc).__name__, "error_message": str(exc)}
        self.failures.append(row)
        print(f"tape failed step={int(step)}: {type(exc).__name__}: {exc} "
              f"(its checkpoint {checkpoint} is kept for export)", flush=True)

    def join(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        try:
            self.join()
        finally:
            self._executor.shutdown(wait=True)

    def receipt(self) -> dict[str, object]:
        return {
            "outdir": str(self.outdir),
            "nlat": int(self.request.nlat),
            "nlon": int(self.request.nlon),
            "written": [row["tape"] for row in self.rows],
            "synthesis_seconds": round(self.synthesis_seconds, 4),
            "host_seconds": round(sum(row["host_s"] for row in self.rows), 4),
            "failures": list(self.failures),
        }


def _export_receipt_name() -> str:
    from .wrfout_export import EXPORT_RECEIPT_NAME

    return EXPORT_RECEIPT_NAME


def _snapshot_for(cfg, backend):
    """The run's pinned checkpoint buffer, or None (numpy backend, or
    ``[output] snapshot_buffer = "pageable"``)."""
    if getattr(backend, "name", "numpy") != "cupy":
        return None
    if str(getattr(cfg, "snapshot_buffer", "pageable")) != "pinned":
        return None
    return PinnedSnapshot(backend.xp)


class SurfaceServices:
    """The run's lower-boundary services: the SST and sea-ice provider on
    the boundary machine (arwen_global.surface_boundary) and the greenness
    march (arwen_global.land_climatology), each None when the run holds
    what it would write."""

    def __init__(self, cfg, boundary=None, provider=None, greenness=None, identity=None):
        self.cfg = cfg
        self.boundary = boundary
        self.provider = provider
        self.greenness = greenness
        self.identity = identity

    @property
    def active(self) -> bool:
        return self.boundary is not None or self.greenness is not None

    def apply(self, state, *, model=None) -> None:
        if self.boundary is not None:
            self.boundary.apply(state, model=model)
        if self.greenness is not None:
            self.greenness.apply(state)

    def diagnostics(self, state) -> dict:
        reader = getattr(self.provider, "diagnostics", None)
        return {} if reader is None else reader(state)

    def initialize(self, state) -> None:
        """A cold start's prognostic boundary arrays (the slab ocean's, or a
        calibration's accumulators) into ``state``'s physics namespace;
        nothing for a provider that keeps none."""
        start = getattr(self.provider, "initialize", None)
        if start is not None:
            start(state)

    def check_restart_state(self, state, label: str) -> None:
        check = getattr(self.provider, "check_state", None)
        if check is not None:
            check(state, label)

    def receipt(self) -> dict | None:
        if not self.active:
            return None
        out = {} if self.boundary is None else self.boundary.receipt()
        out["greenness"] = ("held at the start" if self.greenness is None
                            else self.greenness.receipt())
        return out

    def boundary_receipt(self) -> dict:
        ocean = self.cfg.ocean
        out = ocean.receipt()
        if ocean.implicit and ocean.mode == "hold":
            out["ocean_mode"] = "hold (implicit)"
        out["greenness"] = self.cfg.land.resolved(ocean)
        if self.identity is not None:
            out["restart_identity"] = self.identity
        out["future_surface_records"] = 0 if ocean.mode != "series" else None
        return out


def _surface_services_for(cfg, model, cold) -> SurfaceServices:
    """The lower-boundary services of this run, built from the COLD state
    (the analysis the run starts from, rebuilt by every segment), so the
    starting anomaly of a restarted segment is the straight run's."""
    import numpy as np

    from .surface_boundary import SurfaceBoundary

    ocean = cfg.ocean
    greenness_daily = cfg.land.resolved(ocean) == "daily"
    rule = ocean.mode in ("anomaly", "climatology") or ocean.ice_mode in (
        "anomaly", "climatology")
    series = ocean.mode == "series"
    slab = ocean.mode == "slab"
    calibrate = bool(getattr(ocean, "qflux_calibration", False))
    real = getattr(cfg.statics, "source", None) == "real"
    if not (rule or series or slab or (greenness_daily and real)):
        return SurfaceServices(cfg)
    if (slab or calibrate) and cfg.physics_mode not in ("reference", "arwen-native"):
        raise ValueError(
            f"[ocean] mode = {ocean.mode!r}"
            + (" with qflux_calibration" if calibrate else "")
            + " reads the surface fluxes of a physics suite, and physics.mode = "
            f"{cfg.physics_mode!r} computes none")
    clock = getattr(model, "forecast_clock", None)
    start = None if clock is None else getattr(clock, "start_utc", None)
    if start is None:
        raise ValueError(
            "the ocean rule, the surface series and the greenness march are dated "
            "and this run has no forecast clock (an undated idealized run holds its "
            "surface)")
    grid = model.transform.grid
    statics_fields = statics_provenance = None
    if real:
        from .statics import load_statics

        statics_fields, statics_provenance = load_statics(cfg.statics, grid)
    provider = None
    coupler = None
    if slab:
        from .ocean_coupler import SlabProvider, physics_identity
        from .ocean_qflux import load_qflux
        from .ocean_rules import RuleProvider, cold_surface_planes, reference_sst
        from .surface_climatology import load_climatology

        climatology = load_climatology(ocean.climatology)
        if climatology.sha256 != ocean.climatology_sha256:
            raise ValueError(
                f"the climatology {ocean.climatology} changed since the config was "
                "read (its SHA-256 is not the one in the config identity)")
        qflux = load_qflux(ocean.slab_qflux)
        if qflux.sha256 != ocean.slab_qflux_sha256:
            raise ValueError(
                f"the Q-flux file {ocean.slab_qflux} changed since the config was read "
                "(its SHA-256 is not the one in the config identity)")
        planes = cold_surface_planes(cold.surface)
        target = None
        if ocean.needs_anomaly:
            from dataclasses import replace as _replace

            reference = None
            if ocean.anomaly_reference == "mean-7d":
                reference = reference_sst(ocean.anomaly_reference_series, start=start,
                                          nlat=int(grid.nlat), nlon=int(grid.nlon))
            target = RuleProvider(
                _replace(ocean, mode="anomaly",
                         sea_ice=_replace(ocean.sea_ice, mode="hold")),
                climatology, grid, start_time=start, cold_surface=planes,
                model_truncation=int(cfg.truncation), reference=reference)
        provider = SlabProvider(
            ocean, climatology, qflux, grid, start_time=start, cold_surface=planes,
            model_truncation=int(cfg.truncation), rule=target,
            physics_mode=cfg.physics_mode, physics_identity=physics_identity(cfg),
            duration_s=float(cfg.duration_s))
        print(f"gpuwm-global: ocean slab (depth {ocean.slab_depth}, restoring "
              f"{ocean.slab_relaxation_days or 'none'} d toward {ocean.slab_target}), sea "
              f"ice {ocean.ice_mode}, climatology {ocean.climatology} (sha256 "
              f"{climatology.sha256[:12]}), Q-flux {ocean.slab_qflux} (sha256 "
              f"{qflux.sha256[:12]})", flush=True)
    elif series:
        from .surface_series import SeriesProvider

        provider = SeriesProvider(cfg.surface_series, start_time=start,
                                  duration_s=float(cfg.duration_s))
        print(f"gpuwm-global: surface series {provider.path} "
              f"({provider.provenance.get('source')}, {len(provider.valid)} records "
              f"{provider.valid[0].isoformat()} .. {provider.valid[-1].isoformat()}): "
              "SST every step, sea ice per record", flush=True)
    elif rule:
        from .ocean_rules import RuleProvider, cold_surface_planes, reference_sst
        from .surface_climatology import load_climatology

        climatology = load_climatology(ocean.climatology)
        if climatology.sha256 != ocean.climatology_sha256:
            raise ValueError(
                f"the climatology {ocean.climatology} changed since the config was "
                "read (its SHA-256 is not the one in the config identity)")
        reference = None
        if ocean.needs_anomaly and ocean.anomaly_reference == "mean-7d":
            reference = reference_sst(ocean.anomaly_reference_series, start=start,
                                      nlat=int(grid.nlat), nlon=int(grid.nlon))
        provider = RuleProvider(
            ocean, climatology, grid, start_time=start,
            cold_surface=cold_surface_planes(cold.surface),
            model_truncation=int(cfg.truncation), reference=reference)
        print(f"gpuwm-global: ocean {ocean.mode}, sea ice {ocean.ice_mode}, from "
              f"{ocean.climatology} (sha256 {climatology.sha256[:12]}); no surface "
              "record later than the start is read", flush=True)
        if calibrate:
            from .ocean_coupler import SlabProvider, physics_identity

            coupler = SlabProvider(
                ocean, climatology, None, grid, start_time=start,
                cold_surface=cold_surface_planes(cold.surface),
                model_truncation=int(cfg.truncation), rule=provider, calibrate=True,
                physics_mode=cfg.physics_mode, physics_identity=physics_identity(cfg),
                duration_s=float(cfg.duration_s))
            print("gpuwm-global: Q-flux calibration: the slab's flux is deposited by "
                  f"month after {ocean.qflux_calibration_spinup_days:g} days of spin-up",
                  flush=True)
    boundary = None
    if provider is not None:
        boundary = SurfaceBoundary(
            provider, statics_fields=statics_fields,
            statics_provenance=statics_provenance,
            terrain_height_m=np.zeros((int(grid.nlat), int(grid.nlon))),
            latitude_deg=np.asarray(grid.latitude_deg, dtype=np.float64),
            coupler=coupler)
        if coupler is not None:
            # The rows and the identity are the calibration's (the rule's,
            # with what the calibration adds).
            provider = coupler
    greenness = None
    if greenness_daily and real:
        from .land_climatology import GreennessMarch

        greenness = GreennessMarch(statics_fields, start_time=start)
    identity = provider.identity_block() if hasattr(provider, "identity_block") else None
    services = SurfaceServices(cfg, boundary, provider, greenness, identity)
    model.surface_update = services
    return services


def _surface_update_receipt(model):
    update = getattr(model, "surface_update", None)
    return None if update is None else update.receipt()


def _surface_boundary_receipt(model, cfg):
    update = getattr(model, "surface_update", None)
    if update is None:
        update = SurfaceServices(cfg)
    return update.boundary_receipt()


def _section(profiler, name: str):
    """A profiler section, or a no-op when no profiler is attached."""
    if profiler is None:
        from .profile import NULL_PROFILER

        return NULL_PROFILER.section(name)
    return profiler.section(name)


def _finish_profile(profiler, output: Path, progress) -> None:
    """Write the profile once its last step has been read, and print it."""
    if getattr(profiler, "_written", False):
        return
    path = profiler.write(output / PROFILE_NAME)
    profiler._written = True
    if progress is not None:
        print(
            f"profile: {profiler.profiled_steps} steps after "
            f"{profiler.warmup} warm-up steps, written to {path}"
        )
        print(profiler.table())


def hold_pool_to_plan(tracker, door_plan, allocator=None
                      ) -> dict[str, object] | None:
    """Hold the run's pool to the pool figure of the plan the door priced.

    A run no door priced (a library caller, a test) carries no figure and
    its pool is left as found; so is a numpy run, which has no tracker,
    and a run under the driver's async pool or the slab, each of which
    bounds itself its own way and is priced at its own runs.
    """
    if tracker is None:
        return None
    if door_plan is None:
        return tracker.hold_pool_to(
            None, reason="no door priced this run, so its pool is not held")
    named = "default"
    if allocator is not None:
        named = str(allocator.receipt().get("allocator", "default"))
    if named != "default":
        return tracker.hold_pool_to(
            None, reason=f"the {named} allocator takes no limit from the "
                         "run, so its pool is not held")
    return tracker.hold_pool_to(getattr(door_plan, "pool_limit_bytes", None))


#: ``_run_tracked``'s default for a prediction its caller did not make
#: (None is a real answer: no prediction could be made).
_NOT_PREDICTED = object()


def _sizing_model_peak_bytes(cfg: ArwenGlobalConfig,
                             free_bytes: int | None = None) -> int | None:
    """The pre-run prediction, for the receipt's device_memory row and the
    slab's first reservation.

    ``free_bytes`` is a reading of the card the caller already holds (the
    door's, taken before anything was allocated); without one the card is
    probed here.  Each probe is a short-lived subprocess, MEASURED 0.4 to
    0.9 s beside an RTX 5070 Ti (2026-11-12), and a run used to take three: the door's,
    this one at run start and this one again for the receipt.

    Priced at the PLAN the run will take, band count and pinned tier
    together: a prediction made for a resident run would size the slab's
    arena for a peak the banded run never reaches, and one that ignored
    the tier would size it for bytes the card is never asked for.
    MEASURED 2026-09-06 at T383 L40 on an RTX 5070 Ti, the bare run: the
    run's live peak read 12,470,658,560 B, and the plan on the tree that
    ships prices that shape at 12,691,484,204 -- 1.8 percent above it,
    which is the direction a figure the slab is sized from has to err in.
    (The 12,467,386,320 B the receipt of that leg carries was priced
    before the census scaling and the fragmentation classes landed, so it
    is not the figure this function returns today.)
    """
    from .sizing import (banded_device_peak_bytes, estimate_global_memory,
                         plan_run_memory, _free_device_bytes)

    try:
        estimate = estimate_global_memory(cfg)
    except Exception:  # noqa: BLE001 - a prediction that cannot be made is not a run failure
        return None
    if estimate.backend != "cupy":
        return None
    try:
        if free_bytes is None:
            free_bytes = _free_device_bytes()
        plan = plan_run_memory(cfg, free_bytes, estimate)
        if plan.bands is not None:
            return int(plan.live_peak_bytes)
    except Exception:  # noqa: BLE001 - the run's own refusal, not the prediction's
        pass
    try:
        bands = resolve_latitude_bands(cfg)
    except Exception:  # noqa: BLE001 - the run's own refusal, not the prediction's
        bands = 1
    return int(banded_device_peak_bytes(estimate, bands))


def run(
    cfg: ArwenGlobalConfig,
    outdir: str | Path,
    *,
    restart: str | Path | None = None,
    overwrite: bool = False,
    progress=None,
    until_s: float | None = None,
    profile_steps: int = 0,
    profile_warmup: int = 2,
    sized_by_door: bool = False,
    door_plan=None,
    tapes: TapeRequest | None = None,
    forcing=None,
    forcing_record=None,
    branch_config: bool = False,
    notice=None,
) -> dict[str, object]:
    """Integrate ``cfg`` into ``outdir``.

    AN OUT-OF-MEMORY MID-RUN IS NOT THE END OF THE RUN.  When the card
    refuses an allocation (CuPy's ``OutOfMemoryError`` or a CUDA library's
    out-of-memory error) on a one-card cupy run, the run gives its memory
    back, reads the card again, and carries on from its newest checkpoint
    on the next plan priced below the one that ran out
    (``sizing.fallback_plan``: more latitude bands and/or the pinned host
    tier).  Neither is in the config identity and neither changes a bit,
    so the checkpoints it writes are the ones the uninterrupted run would
    have written.  It says so in one line (``notice``, a callable taking
    the sentence; stderr when ``None``) and in the receipt's ``recovery``
    row.  When no lighter plan fits the card it stops with
    :class:`device_memory.DeviceMemoryExhausted`, whose message names the
    cause, the free memory seen, what else holds the card and the flags
    to resume with.

    ``tapes`` (:class:`TapeRequest`) has the run write the wrfout tape of
    every checkpoint it writes, from its own state and beside the forecast
    (:class:`_TapeWriter`): the bytes ``export`` writes from that
    checkpoint.  Outside the config identity; the forecast's arithmetic
    is the same with and without it.

    ``forcing`` (opt-in, default None) is ``(record_path, ForcingOptions)``:
    the run is pulled toward that stored record of large scales before
    every step (arwen_global.forcing) and writes the pull's ledger beside
    the receipt.  ``forcing_record`` (opt-in) is ``(out_path, options)``:
    the run writes a record of its own low degrees at a cadence.  Neither
    is part of the config identity; both are recorded in the receipt, and
    a run that names neither is the shipped run.

    ``branch_config`` (default False) admits a ``restart`` checkpoint
    written under ANOTHER config hash: an exact twin that differs from its
    parent run by a deliberate config change.  It is an explicit opt-in
    and the receipt records it under ``branch`` with both hashes; without
    it a restart whose hash is not this config's is refused as before.

    ``until_s`` ends the integration at that model time (seconds from the
    run's start) instead of ``cfg.duration_s``: a cycling segment, whose
    last checkpoint the next segment restarts from.  It is outside the
    config identity (the receipt records ``segment_until_s``), so every
    segment of a cycle and the forecast from its final analysis share one
    config hash and one checkpoint lineage.

    ``profile_steps`` > 0 attaches the step profiler
    (arwen_global.profile) for that many steps after
    ``profile_warmup`` unprofiled ones, calibrates it first on the run's
    backend, writes ``profile.json`` beside the receipt and prints the
    per-operator table; the profiled run is bit-identical to an
    unprofiled one (the hooks add no arithmetic).
    """
    # One rank, one card, before the first device byte (MG-4): a rank of a
    # multi-card run narrows CUDA_VISIBLE_DEVICES to its own card.  The
    # door has already done it; this is for a caller that came in here.
    place_rank_for(cfg)
    # ONE prediction per run, priced against the door's own reading of the
    # card when the door took one: the door read it before anything was
    # allocated, which is the instant a pre-run prediction is defined at.
    # Both receipts carry this figure; they used to probe the card again
    # after the run, when this process held the run's own memory.
    predicted_peak_bytes = _sizing_model_peak_bytes(
        cfg, free_bytes=getattr(door_plan, "free_bytes", None))
    # The run's allocator is chosen and installed before the first device
    # byte, because the slab's arena has to be the first thing on the card
    # for the run's bytes to come out of it rather than beside it.
    allocator = select_device_allocator(
        cfg.device_allocator, cfg.backend,
        predicted_peak_bytes=predicted_peak_bytes,
    )
    # The allocator hook goes in before the first device byte (the
    # Legendre tables are the first and, at T533, among the largest) and
    # comes out in the finally below, so the receipt's device peak is the
    # whole run's and the process's allocator is left as it was found.
    tracker = start_device_peak_tracking(cfg.backend)
    # The pool is held to the figure the door priced this plan at, before
    # the first device byte, so it cannot grow past what the run was
    # admitted at (DevicePeakTracker.hold_pool_to; released with the hook).
    hold_pool_to_plan(tracker, door_plan, allocator)
    # The cuFFT plan cache is off for the run (memory only; see
    # device_memory.disable_fft_plan_cache) and restored with the
    # allocator afterwards.
    restore_plan_cache = disable_fft_plan_cache(cfg.backend)
    try:
        recovery: list[dict] = []
        written: list[str] = []
        carried = None
        current_cfg, plan, resume = cfg, door_plan, restart
        sized = bool(sized_by_door) or (
            door_plan is not None
            and getattr(door_plan, "bands_chosen_by", None) == "sizer")
        while True:
            attempt: dict = {
                "restart": resume, "checkpoints": [],
                "started": time.perf_counter(),
                # A refusal recorded before this attempt is not its cause.
                "refusals_before": 0 if tracker is None else tracker.refusals,
            }
            try:
                return _run_tracked(
                    current_cfg, outdir, restart=resume, overwrite=overwrite,
                    progress=progress, tracker=tracker, until_s=until_s,
                    profile_steps=int(profile_steps),
                    profile_warmup=int(profile_warmup),
                    allocator=allocator, sized_by_door=sized,
                    door_plan=plan, attempt=attempt, carried=carried,
                    recovery=recovery or None,
                    resume_in_place=bool(recovery),
                    earlier_checkpoints=tuple(written),
                    tapes=tapes,
                    # The one pre-run prediction belongs to the plan the
                    # door admitted; a lighter plan is priced afresh.
                    predicted_peak_bytes=(
                        _NOT_PREDICTED if recovery else predicted_peak_bytes),
                    forcing=forcing,
                    forcing_record=forcing_record,
                    # A branch reads its parent under another hash only
                    # until the run has a checkpoint of its own to resume.
                    branch_config=bool(branch_config) and resume == restart,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised unless the card ran out and a lighter plan may fit
                if not _may_fall_back(current_cfg, attempt, exc):
                    raise
                attempt["error"] = f"{type(exc).__name__}: {exc}"[:600]
            # Out of the handler, so the dead attempt's frames (its model,
            # its state, every device array they held) are released before
            # the card is read again.
            carried = attempt.get("carried") or carried
            # The dead attempt's checkpoints are the run's too (the next
            # attempt carries on from the newest of them), so the receipt
            # lists them before the ones the next attempt writes.
            written.extend(str(path) for path in attempt.get("checkpoints") or ()
                           if Path(path).is_file() and str(path) not in written)
            current_cfg, plan, resume = _fall_back(
                current_cfg, plan, attempt, recovery, tracker, allocator,
                until_s=until_s, original_restart=restart, notice=notice)
            sized = False
    finally:
        restore_plan_cache()
        if tracker is not None:
            tracker.uninstall()
        if allocator is not None:
            allocator.uninstall()
            close = getattr(allocator, "close", None)
            if close is not None:
                # The arena's bytes go back when the last array that sits
                # in it goes away, not here; this drops the allocator's
                # own hold.
                close()


#: How many times one run may move to a lighter plan.  Each move is to a
#: plan priced strictly below the last, so the ladder ends the moves by
#: itself; this bounds a card whose other tenants keep growing.
MAX_FALLBACKS = 4


def _may_fall_back(cfg, attempt: dict, error) -> bool:
    """Whether a run that raised ``error`` may carry on on a lighter plan:
    the card ran out of memory, on a one-card cupy run, before its steps
    were all taken (a death after the last checkpoint has nothing left to
    integrate)."""

    from .device_memory import is_out_of_memory

    if str(getattr(cfg, "backend", "numpy")) != "cupy":
        return False
    if int(getattr(cfg, "cards", 1) or 1) > 1:
        # The other ranks are mid-collective; one rank cannot change plan
        # alone.
        return False
    if attempt.get("loop_done"):
        return False
    return is_out_of_memory(error)


def _newest_checkpoint(attempt: dict):
    """The newest checkpoint the attempt wrote that is on disk, else the
    checkpoint it started from (``None`` for a cold start)."""

    for path in reversed(list(attempt.get("checkpoints") or ())):
        if Path(path).is_file():
            return path
    return attempt.get("restart")


def _step_of_checkpoint(path) -> int | None:
    if path is None:
        return 0
    name = Path(path).stem
    digits = name[len(CHECKPOINT_PREFIX):] if name.startswith(
        CHECKPOINT_PREFIX) else ""
    return int(digits) if digits.isdigit() else None


def _drop_diagnostics_after(path, step) -> None:
    """Drop the diagnostics rows past ``step`` from ``path``.

    A death between a step's diagnostics row and its checkpoint (in the
    checkpoint's device-to-host copy, say) leaves a row for a step the run
    carries on from before; the attempt that carries on writes that row
    again, so the dead attempt's is dropped and every step keeps one row.
    """

    if path is None or step is None:
        return
    target = Path(path)
    if not target.is_file():
        return
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = []
    for line in lines:
        try:
            row_step = int(json.loads(line)["step"])
        except Exception:  # noqa: BLE001 - a row this cannot read is kept as it is
            kept.append(line)
            continue
        if row_step <= int(step):
            kept.append(line)
    if len(kept) == len(lines):
        return
    staged = target.with_name(target.name + ".tmp")
    with staged.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(kept))
    os.replace(staged, target)


def _plan_words(bands, slices) -> str:
    bands = int(bands or 1)
    tier = ("the host tier holding " + ", ".join(slices)) if slices \
        else "nothing parked"
    return f"{bands} latitude band{'' if bands == 1 else 's'} with {tier}"


def _plan_flags(bands, spill_on) -> str:
    return (f"--latitude-bands {int(bands)} --host-spill "
            f"{'on' if spill_on else 'off'}")


def _user_pool_limit_bytes(pool, environ=None) -> int | None:
    """The limit ``CUPY_GPU_MEMORY_LIMIT`` put on the pool, in bytes as the
    pool reads it (CuPy resolves a percentage itself), or ``None`` when the
    environment sets none, sets ``0`` (CuPy's "no limit") or the pool will
    not say.  The run's own limit is released before this is read, so a
    limit the pool still carries is the user's (``hold_pool_to``)."""

    from .device_memory import environment_pool_limit

    if pool is None or environment_pool_limit(environ) is None:
        return None
    try:
        limit = int(pool.get_limit())
    except Exception:  # noqa: BLE001 - a pool that will not say has no limit to weigh
        return None
    return limit or None


def _refusal_words(refusal: dict | None, error: str,
                   user_limit: int | None = None) -> str:
    """``the card refused 810 MiB with 738 MiB free while the pool held
    13.05 GiB``, or the error itself when the pool saw no refusal.

    When the request did not fit under ``user_limit`` (the pool limit
    ``CUPY_GPU_MEMORY_LIMIT`` set) it was that limit, not the card, that
    refused it, and the words say so: MEASURED 2026-11-13 at T383 L40 on
    the 16 GB card under a 6 GiB limit, the line said the card refused
    101 MiB with 9,284 MiB of it free."""

    if not refusal:
        return f"the CUDA call failed with {error.rstrip('.')}"
    request = int(refusal["request_bytes"])
    total = refusal.get("pool_total_bytes")
    if user_limit and total is not None and int(total) + request > user_limit:
        text = (f"the pool limit CUPY_GPU_MEMORY_LIMIT sets "
                f"({user_limit / 2**30:.2f} GiB) refused {request >> 20:,} MiB")
    else:
        text = f"the card refused {request >> 20:,} MiB"
    free = refusal.get("card_free_bytes")
    if free is not None:
        text += f" with {int(free) >> 20:,} MiB free"
    total, used = refusal.get("pool_total_bytes"), refusal.get("pool_used_bytes")
    if total is not None and used is not None:
        text += (f" while this run's pool held {int(total) / 2**30:.2f} GiB, "
                 f"{int(used) / 2**30:.2f} GiB of it in use")
    return text


def _fall_back(cfg, plan, attempt: dict, recovery: list, tracker, allocator,
               *, until_s, original_restart, notice):
    """Give the dead attempt's memory back, read the card, and choose the
    plan the run carries on with; ``(cfg, plan, resume_checkpoint)``, or
    :class:`DeviceMemoryExhausted` when none fits."""

    import dataclasses
    import gc
    import sys

    from .device_memory import (DeviceMemoryExhausted, card_neighbours,
                                installed_pool, neighbours_phrase)
    from .sizing import (GIB, OUT_OF_POOL_BYTES, _free_device_bytes,
                         estimate_global_memory, fallback_plan, lighter_plans)

    gc.collect()
    pool = None
    cupy = sys.modules.get("cupy")
    if cupy is not None and str(getattr(cfg, "backend", "")) == "cupy":
        try:
            pool = installed_pool(cupy)
            release = getattr(pool, "free_all_blocks", None)
            if release is not None:
                release()
        except Exception:  # noqa: BLE001 - nothing handed back is weighed as such
            pool = None
    if tracker is not None:
        tracker.release_pool_limit()
    refusal = None
    if tracker is not None and tracker.refusals > int(
            attempt.get("refusals_before", 0)):
        refusal = tracker.last_refusal
    error = str(attempt.get("error") or "an out-of-memory error")
    bands = int(attempt.get("bands") or getattr(cfg, "latitude_bands", 0) or 1)
    slices = tuple(attempt.get("spill_slices") or ())
    resume = _newest_checkpoint(attempt)
    resume_step = _step_of_checkpoint(resume)
    _drop_diagnostics_after(attempt.get("diagnostics"), resume_step)
    died_at = attempt.get("step")
    neighbours = card_neighbours()
    free = _free_device_bytes()
    credit = 0
    if free and neighbours and neighbours.get("own_bytes") is not None:
        # This process's context and libraries are on the card now and were
        # not when the door read it; the plan's out-of-pool term already
        # charges them, so they are credited back, up to that term.
        held = 0
        try:
            held = int(pool.total_bytes()) if pool is not None else 0
        except Exception:  # noqa: BLE001 - nothing to credit on doubt
            held = int(neighbours["own_bytes"])
        credit = max(0, min(int(neighbours["own_bytes"]) - held,
                            int(OUT_OF_POOL_BYTES)))
    usable = None if not free else int(free) + credit
    # A pool limit the user set binds a lighter plan as the card does: its
    # pool part has to fit under the limit, so the plans are weighed against
    # the limit plus the out-of-pool term their card figure carries.
    # Without this, MEASURED 2026-11-13 at T383 L40 on the 16 GB card under
    # a 6 GiB limit, the run rebuilt on 2, 4 and 8 bands, each priced
    # against the card's 15.2 GiB and each refused by the limit, and then
    # told the user to free a card nothing else held.
    user_limit = _user_pool_limit_bytes(pool)
    weighed = usable
    if usable and user_limit:
        weighed = min(int(usable), int(user_limit) + int(OUT_OF_POOL_BYTES))
    try:
        estimate = estimate_global_memory(cfg)
        lighter = lighter_plans(cfg, bands, slices, estimate)
    except Exception:  # noqa: BLE001 - a plan that cannot be priced is no plan
        estimate, lighter = None, []
    where = ("" if died_at is None else f" after step {int(died_at)}")
    cause = (f"the card ran out of memory{where} on "
             f"{_plan_words(bands, slices)}: "
             f"{_refusal_words(refusal, error, user_limit)}")
    seen = ("the card could not be read again" if not usable else
            f"with this run's memory given back the card has "
            f"{usable / GIB:.2f} GiB free to it")
    limit_words = ""
    if user_limit:
        limit_words = (f"; CUPY_GPU_MEMORY_LIMIT holds this run's pool to "
                       f"{user_limit / GIB:.2f} GiB")
    others = neighbours_phrase(neighbours) + limit_words
    nxt = None
    why = ""
    if len(recovery) >= MAX_FALLBACKS:
        why = f"the run has already moved to a lighter plan {len(recovery)} times"
    elif estimate is not None and weighed:
        nxt = fallback_plan(
            cfg, weighed, bands, slices, estimate,
            why=f"after the card ran out of memory{where}")
    if nxt is None:
        if not why:
            if not lighter:
                why = "no plan is priced below this one"
            elif not usable:
                why = "so no lighter plan could be weighed against it"
            else:
                count, spill, card = min(lighter, key=lambda row: row[2])
                tier = " with the host tier on" if spill else ""
                fits = "fits it"
                if user_limit and weighed < usable:
                    fits = "fits under that limit"
                why = (f"no lighter plan {fits} (the lightest, {count} "
                       f"latitude band{'' if count == 1 else 's'}{tier}, "
                       f"prices at {card / GIB:.2f} GiB)")
        if lighter:
            count, spill, _card = lighter[0]
            flags = _plan_flags(count, spill)
        else:
            flags = _plan_flags(bands, bool(slices))
        resume_text = (f"--restart {resume}" if resume is not None
                       else "no checkpoint (start it again)")
        until = "" if until_s is None else f" --until-s {until_s:g}"
        remedy = ("Raise or unset CUPY_GPU_MEMORY_LIMIT" if user_limit
                  else "Free the card of what else holds it")
        message = (
            f"{cause}; {seen}, and {others}; {why}.  The run stopped "
            f"cleanly.  {remedy} and resume "
            f"with: {resume_text} {flags}{until}")
        raise DeviceMemoryExhausted(message)
    spill_on = bool(nxt.spill_slices)
    new_cfg = dataclasses.replace(
        cfg, latitude_bands=int(nxt.bands),
        host_spill="on" if spill_on else "off")
    hold_pool_to_plan(tracker, nxt, allocator)
    row = {
        "last_completed_step": None if died_at is None else int(died_at),
        "error": error,
        "refusal": refusal,
        "from_plan": {"latitude_bands": bands, "host_spill_slices": list(slices)},
        "to_plan": nxt.receipt(),
        "card_free_bytes_after_release": None if free is None else int(free),
        "own_out_of_pool_credit_bytes": int(credit),
        "user_pool_limit_bytes": user_limit,
        "neighbours": neighbours,
        "resumed_from": None if resume is None else str(resume),
        "resumed_from_step": resume_step,
        "original_restart": (None if original_restart is None
                             else str(original_restart)),
        "attempt_wall_seconds": round(
            time.perf_counter() - float(attempt.get("started", 0.0)), 3),
    }
    recovery.append(row)
    sentence = (
        f"{cause}; {others}.  Carrying on from "
        + (f"step {resume_step}" if resume is not None else "the start")
        + f" on {_plan_words(nxt.bands, nxt.spill_slices)}, priced at "
        f"{nxt.card_bytes / GIB:.2f} GiB of card against {usable / GIB:.2f} "
        "GiB free (same forecast, same bits; "
        f"{_plan_flags(nxt.bands, spill_on)} is this plan)")
    if notice is None:
        print(f"gpuwm-global: {sentence}", file=sys.stderr, flush=True)
    else:
        notice(sentence)
    return new_cfg, nxt, resume


def _carries_assimilation_chain(metadata: dict) -> bool:
    """Whether a checkpoint's physics metadata holds an assimilation chain
    with at least one cycle (the key the door writes,
    ``assimilate.ASSIMILATION_HISTORY_KEY``; named here by value so the
    runner does not import the door)."""
    physics = metadata.get("physics_metadata") or {}
    chain = physics.get("assimilation_history")
    if not isinstance(chain, dict):
        return False
    cycles = chain.get("cycles")
    return isinstance(cycles, list) and len(cycles) > 0


def _release_cached_device_blocks(backend) -> None:
    """Hand the pool's cached free blocks back to the device once.

    Memory only.  Initialisation (analysis ingest, the cold-state repair,
    the first diagnostics) leaves the pool holding blocks in its own
    shapes that the step never re-requests exactly: measured 19 GiB of
    cached free blocks beside 7 GiB live after a T533 forty-level init
    (2026-09-02).  Returning them lets the step take its working set
    from the device in the shapes it actually asks for instead of
    fragments of the ingest's.
    """
    if backend.name == "cupy":
        # The pool the run's allocator actually spends, not whichever pool
        # the process started with: a run under the async pool or the slab
        # left its cached blocks where they were when this read the default
        # pool by name.
        from .device_memory import installed_pool

        pool = installed_pool(backend.xp)
        release = getattr(pool, "free_all_blocks", None)
        if release is not None:
            release()


def _forcing_setup(cfg, model, transform, forcing, forcing_record):
    """The run's forcing operator and record writer (arwen_global.forcing),
    each None unless asked for.  Refused where the pull would touch state
    the operator does not see whole: a multi-card run, or a host tier that
    parks spectral fields."""
    from .forcing import (
        ForcingOperator,
        ForcingOptions,
        ForcingRecord,
        ForcingRecordWriter,
        level_reference_pressure_pa,
    )

    if getattr(model.pipeline, "exchange", None) is not None:
        raise ValueError("the forcing operator and the forcing record run on one card only")
    if getattr(model, "host_tier", None) is not None:
        raise ValueError(
            "the forcing operator and the forcing record need the spectral state resident; "
            "this run parks it on a host tier (set host_spill = \"off\" or a smaller truncation)")
    forcing_op = None
    if forcing is not None:
        record_path, options = forcing
        if options is None:
            options = ForcingOptions()
        record = ForcingRecord.read(record_path)
        forcing_op = ForcingOperator(
            record, options, transform=transform, dt_s=cfg.dt_s, nlev=cfg.vertical.nlev,
            level_pressure_pa=level_reference_pressure_pa(cfg))
    record_writer = None
    record_out = None
    if forcing_record is not None:
        record_out, writer_options = forcing_record
        writer_options = dict(writer_options or {})
        writer_options.setdefault("every_s", float(cfg.output_interval_s))
        record_writer = ForcingRecordWriter(
            source={"run": cfg.name, "config_hash": cfg.config_hash,
                    "truncation": int(cfg.truncation), "nlev": int(cfg.vertical.nlev)},
            **writer_options)
    return forcing_op, record_writer, record_out


def _run_tracked(
    cfg: ArwenGlobalConfig,
    outdir: str | Path,
    *,
    restart,
    overwrite: bool,
    progress,
    tracker,
    until_s: float | None = None,
    profile_steps: int = 0,
    profile_warmup: int = 2,
    allocator=None,
    sized_by_door: bool = False,
    door_plan=None,
    tapes: TapeRequest | None = None,
    predicted_peak_bytes=_NOT_PREDICTED,
    forcing=None,
    forcing_record=None,
    branch_config: bool = False,
    attempt: dict | None = None,
    carried: dict | None = None,
    recovery: list | None = None,
    resume_in_place: bool = False,
    earlier_checkpoints: tuple = (),
) -> dict[str, object]:
    """One attempt at the run (:func:`run` calls it again, on a lighter
    plan, after the card runs out of memory).

    ``attempt`` is filled in as the attempt goes (the plan it built, the
    checkpoints it wrote, the step it reached) so a fallback knows where to
    carry on from.  ``carried`` is the first attempt's conservation targets
    and segment-start diagnostics, held by every later attempt so it
    continues THAT run (a restart from an assimilation chain would
    otherwise re-target to the later checkpoint's means).  ``recovery`` is
    the receipt's row of fallbacks taken; ``resume_in_place`` leaves the
    run directory the dead attempt prepared as it is;
    ``earlier_checkpoints`` are the checkpoints earlier attempts wrote, which
    the receipt lists before this attempt's.
    """
    if predicted_peak_bytes is _NOT_PREDICTED:
        predicted_peak_bytes = _sizing_model_peak_bytes(
            cfg, free_bytes=getattr(door_plan, "free_bytes", None))
    output = Path(outdir)
    if branch_config and restart is None:
        raise ValueError(
            "--branch-from-other-config names a branch from a parent "
            "checkpoint; give that checkpoint with --restart"
        )
    from .boundary_options import long_run_refusal

    refusal = long_run_refusal(cfg)
    if refusal is not None:
        raise ValueError(refusal)
    if until_s is not None:
        until_s = float(until_s)
        if not math.isfinite(until_s) or until_s <= 0.0:
            raise ValueError("--until-s must be a positive, finite model time")
        if until_s > cfg.duration_s + 1.0e-9:
            raise ValueError(
                f"--until-s {until_s:g} lies beyond the config's duration_s "
                f"{cfg.duration_s:g}; a segment cannot outrun the run it "
                "belongs to"
            )
        if abs(until_s / cfg.dt_s - round(until_s / cfg.dt_s)) > 1.0e-6:
            raise ValueError(
                f"--until-s {until_s:g} is not a whole number of {cfg.dt_s:g} s "
                "steps; the segment must end on a step boundary the next "
                "segment can restart from"
            )
    restart_metadata = None
    restart_arrays = None
    if restart is not None:
        # Read the restart source before the overwrite sweep: the sweep owns
        # every arwen_global_step*.npz in outdir, so a resume-in-place would
        # otherwise delete the checkpoint it is about to read. The source is
        # also excluded from the sweep so the receipt's restart row keeps
        # pointing at an existing file.
        # The pin is checked against THIS run's scheme, not the build's
        # default: an archive of the external era resumes under
        # scheme = "external" (its pin is that era's, and the arithmetic is
        # shipped bit-identical), and a checkpoint of either scheme never
        # resumes under the other.
        # A branch under another config (the explicit opt-in) reads the
        # parent with every other check intact and the hash comparison
        # left to the receipt, which names both.
        # The world is checked before the config hash, which would refuse a
        # checkpoint of another world too but without naming why.
        restart_metadata, restart_arrays = read_checkpoint(
            restart,
            expected_config_hash=None if branch_config else cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
            expected_world=world_of(cfg),
        )
        carried_arrays = carried_physics_arrays(cfg)
        restart_metadata["absent_carried_physics_arrays"] = [
            name for name in restart_metadata.get(
                "absent_carried_physics_arrays", [])
            if name.removeprefix("physics__") in carried_arrays
        ]
        from .longrun_ops import refuse_time_mean

        refuse_time_mean(restart_metadata, f"restart checkpoint {restart}")
    from .longrun_ops import launch_gate

    launch_estimate = launch_gate(
        cfg, output,
        start_s=0.0 if restart_metadata is None else float(restart_metadata["time_s"]),
        until_s=until_s)
    if launch_estimate is not None:
        wall = launch_estimate.get("wall_s")
        print(f"gpuwm-global: this segment writes about "
              f"{launch_estimate['disk_bytes'] / 1e9:.2f} GB "
              f"({launch_estimate['checkpoints_on_disk']} checkpoints of "
              f"{launch_estimate['checkpoint_bytes_each'] / 1e9:.2f} GB at most on disk); "
              f"{launch_estimate['free_bytes'] / 1e9:.1f} GB free"
              + ("" if wall is None else
                 f"; about {wall / 3600.0:.1f} h of stepping at the planning rate"
                 " (cold build and export not counted)"),
              flush=True)
    if resume_in_place:
        # The dead attempt prepared this directory and wrote the
        # checkpoints this attempt carries on from: nothing is swept.
        output.mkdir(parents=True, exist_ok=True)
    else:
        _prepare_outdir(
            output,
            overwrite,
            keep=(
                frozenset()
                if restart is None
                else frozenset({Path(restart).resolve()})
            ),
        )
    diagnostics_path = output / DIAGNOSTICS_NAME
    receipt_path = output / RECEIPT_NAME
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=output)
    cold_diag = model.diagnostics(cold)
    # The boundary rule is built from the cold state on every segment, and a
    # restart written under another rule is refused here, before any of its
    # arrays becomes the run's state.
    surface_update = _surface_services_for(cfg, model, cold)
    if restart is not None:
        from .ocean_rules import check_restart_identity

        check_restart_identity(surface_update.identity,
                               restart_metadata.get("physics_metadata"),
                               f"restart checkpoint {restart}")
    elif surface_update.identity is not None:
        from .ocean_rules import IDENTITY_KEY

        cold.physics_state.metadata[IDENTITY_KEY] = surface_update.identity
    if restart is None:
        surface_update.initialize(cold)
    target_mass = cold_diag["global_mean_surface_pressure_pa"]
    target_water = cold_diag["global_mean_total_water_kg_m2"]
    trackers = normalize_trackers()
    supplementary = fresh_supplementary()
    species_fixer: dict[str, float] = {}
    fixer_ledger = PositivityFixerLedger()
    floor_ledger = StratosphericFloorLedger()
    restart_targets = None
    if restart is None:
        state = cold
    else:
        state = state_from_checkpoint(
            restart_metadata, restart_arrays, transform.backend,
            carried_physics_arrays=carried_physics_arrays(cfg),
        )
        adopt_checkpoint_clock(model, state, f"restart checkpoint {restart}")
        surface_update.check_restart_state(state, f"restart checkpoint {restart}")
        # The second time level rides with the state.  Without it the
        # resumed run would take a non-extrapolated start-up step in the
        # middle of a forecast, which is not the continuation of the
        # uninterrupted run, and the device-qualification pin asserts
        # exactly that it is.
        model.set_trajectory_state(
            trajectory_from_checkpoint(
                restart_metadata, restart_arrays, transform.backend
            )
        )
        trackers = normalize_trackers(restart_metadata["run_trackers"])
        if state.time_s >= cfg.duration_s - 1.0e-9:
            raise ValueError(
                "restart checkpoint is already at or beyond configured duration"
            )
        if until_s is not None and state.time_s >= until_s - 1.0e-9:
            raise ValueError(
                f"restart checkpoint at {state.time_s:g} s is already at or "
                f"beyond the segment end --until-s {until_s:g}"
            )
        model.enforce(state)
        if _carries_assimilation_chain(restart_metadata):
            # A state with an assimilation chain opens a new conservation
            # epoch: its analysis increments are deliberate external
            # sources of mass and water, not drift, and the fixers would
            # otherwise remove them on the first step and trip their
            # absorption gates (measured on the T255 hourly cycle: the
            # moisture update's 0.15 kg/m2 global-mean water read 2.1e-4
            # of the column against the water fixer's 1e-5 step gate on
            # the first restarted step).  The targets become the restart
            # state's own global means; a restart without a chain keeps
            # the cold start's, bit for bit as before.
            model.initialize_mass_target(state.atmosphere)
            model.initialize_water_target(state)
            epoch = model.diagnostics(state)
            restart_targets = {
                "reason": (
                    "the restart checkpoint carries an assimilation chain; "
                    "its increments are deliberate sources, so the mass and "
                    "water targets are the restart state's own global means"
                ),
                "cold_mass_target_pa": target_mass,
                "cold_total_water_target_kg_m2": target_water,
                "mass_target_pa": epoch["global_mean_surface_pressure_pa"],
                "total_water_target_kg_m2": epoch["global_mean_total_water_kg_m2"],
            }
            target_mass = epoch["global_mean_surface_pressure_pa"]
            target_water = epoch["global_mean_total_water_kg_m2"]
    # The open-water exchange account (native physics, water.
    # OCEAN_EXCHANGE_NAMES): a cold start and a checkpoint written before
    # the account existed are given a zero one here, before the first
    # checkpoint, and the receipt says which; a state that carries one into
    # a run without it refuses.
    exchange_account_fill = model.ensure_exchange_account(state)
    # The cold state's only readers above are done; on a cold start it IS
    # ``state`` and on a restart it is dead weight (one spectral state,
    # 1.19 GiB at T533 with 40 levels, held for the whole run otherwise).
    del cold, restart_arrays
    if restart is not None:
        # The cold state was parked by the builder; a restart arrives from
        # the checkpoint reader on the card and is parked here, into the
        # same named slots.
        state = attach_host_tier(cfg, model, state)
    if carried is not None:
        # A later attempt of the same run: the first attempt's targets,
        # exactly, so the fixers hold what the uninterrupted run held.
        model.set_conservation_targets(*carried["model_targets"])
        target_mass, target_water = carried["receipt_targets"]
        restart_targets = carried["restart_targets"]
    if attempt is not None:
        attempt["bands"] = int(model.pipeline.receipt()["latitude_bands"])
        attempt["spill_slices"] = tuple(getattr(model, "spill_slices", ()) or ())
    _release_cached_device_blocks(transform.backend)

    # The in-situ ledger watches every step from here on (default on; the
    # [insitu] table is outside the config identity).
    ledger = (
        InsituLedger(cfg, model, output).attach() if cfg.insitu.enabled else None
    )
    # The step profiler, calibrated on this backend before it reads
    # anything (profile.calibrate refuses an instrument that fails).
    profiler = None
    if profile_steps > 0:
        profiler = StepProfiler(
            transform.backend, steps=profile_steps, warmup=profile_warmup,
        )
        profiler.calibration = calibrate(transform.backend)
        attach_profiler(model, profiler)
    transform_check = transform.transform_check(seed=19)
    card_gather = _card_gather_of(model, transform)
    writer = _CheckpointWriter(
        cfg, transform.backend.to_numpy,
        trajectory=model.trajectory_state if model.semi_lagrangian else None,
        card_gather=card_gather,
        snapshot=_snapshot_for(cfg, transform.backend),
    )
    tape_writer = None
    tapes_refused = None
    if tapes is not None:
        tapes_refused = tape_refusal(cfg, model)
        if tapes_refused is None:
            try:
                tape_writer = _TapeWriter(cfg, model, transform, tapes)
            except Exception as exc:  # noqa: BLE001 - no tape writer is not the forecast's failure
                # No tape writer or render kernels on this machine, an
                # undated run (no instant to stamp a tape with), a grid
                # the sampler refuses: the forecast runs and export says
                # the same thing in its own words if it is asked.
                tapes_refused = f"{type(exc).__name__}: {exc}"
        if tapes_refused is not None:
            # Not the forecast's failure: the checkpoints are written as
            # always and ``export`` writes their tapes.
            print(f"tapes off: {tapes_refused}; export writes this run's "
                  "tapes from its checkpoints", flush=True)
    output_every = int(round(cfg.output_interval_s / cfg.dt_s))
    total_steps = int(round(cfg.duration_s / cfg.dt_s))
    if until_s is not None:
        total_steps = int(round(until_s / cfg.dt_s))
    checkpoints: list[str] = []
    from .longrun_ops import CheckpointRetention, LongRunProducts, SummaryGeometry

    clock_start = getattr(getattr(model, "forecast_clock", None), "start_utc", None)
    retention = CheckpointRetention(
        output, cfg.output, dt_s=cfg.dt_s, start=clock_start,
        protected=() if restart is None else (restart,))
    # The daily summary's derived planes (surface stress, sea-level pressure,
    # 850 hPa vorticity, water paths) are made on the host from the
    # checkpoint's own arrays through a numpy synthesis of the run's grid;
    # the model's transform, state and card are never touched.
    summary_geometry, geometry_note = None, None
    if cfg.output.daily_summary:
        try:
            summary_geometry = SummaryGeometry.for_run(
                cfg, transform, transform.backend.to_numpy(model.surface_geopotential))
            if summary_geometry.surface_geopotential.shape != tuple(
                    summary_geometry.grid.shape):
                raise ValueError("the terrain plane is not the whole grid")
        except Exception as exc:  # noqa: BLE001 - the summary keeps its planes
            summary_geometry = None
            geometry_note = f"off: {type(exc).__name__}: {exc}"
    products = LongRunProducts(
        output, cfg.output, start=clock_start, dt_s=cfg.dt_s,
        config_hash=cfg.config_hash, physics_state_schema=state.physics_state.schema,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
        geometry=summary_geometry)
    if geometry_note is not None:
        products.geometry_note = geometry_note
    if products.enabled and clock_start is None:
        raise ValueError("[output] mean_state and daily_summary are dated by the "
                         "forecast clock, and this run has none")
    products_resume = (products.resume(Path(restart)) if restart is not None
                       else ("fresh" if products.enabled else "off"))

    def _long_run_products(path, arrays, record):
        if not products.enabled:
            return
        import numpy as np

        products.add(arrays, {
            "step": record["step"], "time_s": record["time_s"],
            "physics_metadata": record["physics_metadata"],
            "trackers": json.loads(json.dumps(record["trackers"], default=float)),
            "dtypes": {name: str(np.asarray(value).dtype)
                       for name, value in arrays.items()},
        })
        products.write_sidecar(path)

    writer.on_arrays = _long_run_products
    # The software stack a replay of any window has to reproduce, read once
    # before the first step (receipt.software_stack); descriptive only.
    stack = software_stack(cfg.backend)
    start_wall = time.perf_counter()
    if carried is not None:
        # The step this attempt starts at already has its row in the file
        # (the dead attempt wrote it beside the checkpoint), and the
        # segment started where the first attempt did.
        initial_segment_diag = carried["segment_start_diagnostics"]
    else:
        initial_segment_diag = model.diagnostics(state)
        _append_diagnostics(diagnostics_path, initial_segment_diag)
    if attempt is not None:
        attempt["checkpoints"] = checkpoints
        attempt["diagnostics"] = str(diagnostics_path)
        attempt["step"] = int(state.step)
        attempt["carried"] = carried or {
            "model_targets": model.conservation_targets(),
            "receipt_targets": (target_mass, target_water),
            "restart_targets": restart_targets,
            "segment_start_diagnostics": initial_segment_diag,
        }
    if restart is None:
        path = write_checkpoint(
            _checkpoint_path(output, state.step),
            state,
            config_hash=cfg.config_hash,
            to_numpy=transform.backend.to_numpy,
            trackers=trackers,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
            trajectory=model.trajectory_state(),
            card_gather=card_gather,
            compression=str(getattr(cfg, "checkpoint_compression", "deflate")),
        )
        checkpoints.append(str(path))
        if tape_writer is not None:
            tape_writer.submit(
                state, bundle_arrays(state, transform.backend.to_numpy), path)

    # The wall the STEPS took, separate from the run's: gate OVERLAP-1
    # divides the tier's measured transfer time by it, and a build, a
    # cold start and a first diagnostics call are not steps.
    step_wall_seconds = 0.0
    step_count = 0
    # The loop below never reads a state after stepping it, so the model
    # may drop its input namespace mid-step (memory only, same bits).
    model.release_input_namespace = True
    if not surface_update.active:
        surface_update = None
    # The forcing operator and the record writer exist only when the run
    # asks for them (arwen_global.forcing); a run that names neither
    # takes none of the branches below.
    forcing_op = None
    record_writer = None
    record_out = None
    if forcing is not None or forcing_record is not None:
        forcing_op, record_writer, record_out = _forcing_setup(
            cfg, model, transform, forcing, forcing_record)
        if record_writer is not None:
            record_writer.offer(state, transform.backend.to_numpy)
    try:
        while state.step < total_steps:
            if profiler is not None:
                # Numbered by the step it produces, like the checkpoints.
                profiler.begin_step(state.step + 1)
            step_started = time.perf_counter()
            if surface_update is not None:
                with _section(profiler, "surface_update"):
                    surface_update.apply(state, model=model)
            if forcing_op is not None:
                with _section(profiler, "forcing"):
                    state = forcing_op.apply(state, model)
            state, metrics = model.step(state, cfg.dt_s)
            if record_writer is not None:
                record_writer.offer(state, transform.backend.to_numpy)
            step_wall_seconds += time.perf_counter() - step_started
            step_count += 1
            if attempt is not None:
                attempt["step"] = int(state.step)
            if model.pipeline.exchange is not None:
                model.pipeline.exchange.session.mark_step()
            _update_trackers(trackers, supplementary, metrics,
                             species_fixer)
            fixer_ledger.add(metrics)
            floor_ledger.add(metrics)
            due = state.step % output_every == 0 or state.step == total_steps
            if due:
                with _section(profiler, "diagnostics"):
                    diag = model.diagnostics(state)
                diag["step_metrics"] = metrics
                diag["positivity_fixer_ledger"] = fixer_ledger.record()
                diag["stratospheric_floor_ledger"] = floor_ledger.record()
                if surface_update is not None:
                    diag.update(surface_update.diagnostics(state))
                _append_diagnostics(diagnostics_path, diag)
                if tape_writer is not None:
                    # The previous tape reads the host arrays its checkpoint
                    # was copied into; it finishes before this checkpoint's
                    # copy, so a checkpoint writer that reuses one host
                    # buffer for every copy never overwrites a tape's input.
                    with _section(profiler, "tape"):
                        tape_writer.join()
                with _section(profiler, "checkpoint"):
                    checkpoint = writer.submit(
                        _checkpoint_path(output, state.step), state, trackers,
                    )
                    retention.prune(in_flight=checkpoint)
                checkpoints.append(str(checkpoint))
                if tape_writer is not None:
                    with _section(profiler, "tape"):
                        tape_writer.submit(state, writer.last_arrays, checkpoint)
                writer.last_arrays = None
                if progress is not None:
                    progress(diag)
            if profiler is not None:
                profiler.end_step()
                if profiler.profiled_steps == profiler.steps and not profiler.active:
                    _finish_profile(profiler, output, progress)
    except Exception as exc:
        # A checkpoint still in flight is joined so the failure receipt
        # lists only files that exist; a join that fails itself is the
        # same class of failure and rides as this exception's context.
        try:
            writer.close()
        except Exception as writer_exc:  # noqa: BLE001 - recorded beside the model's own failure
            exc.__context__ = writer_exc
        if tape_writer is not None:
            # The tape in flight finishes (it fails softly by itself) so
            # the folder holds whole tapes only.
            try:
                tape_writer.close()
            except Exception:  # noqa: BLE001 - never above the model's own failure
                pass
        failure = {
            "name": cfg.name,
            "status": "error",
            "config_hash": cfg.config_hash,
            # The world the dead run was on; absent for Earth.
            **world_receipt_rows(world_of(cfg)),
            "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
            "pins": pins_receipt(cfg.semi_implicit_scheme, cfg.integrator),
            "model": "moist-hybrid-spectral",
            "physics_mode": cfg.physics_mode,
            "physics_split": cfg.physics_split,
            "vertical": vertical_grid_receipt(cfg),
            # The schedule the failing run was on.  An out-of-memory
            # death is exactly when a reader needs to know which band
            # count was streaming and who chose it, and the successful
            # receipt carried it while this one did not.
            "latitude_bands": latitude_bands_receipt(model, cfg, sized_by_door,
                                                     plan=door_plan),
        "surface_update": _surface_update_receipt(model),
        "surface_boundary": _surface_boundary_receipt(model, cfg),
            # And the other half of the same decision.  A run that died
            # recorded which band count was streaming and neither what the
            # pinned host tier was holding nor what the sizer priced, so
            # the one artifact a reader has of a dead run could not say
            # whether the tier was even on.  MEASURED 2026-09-07: the
            # T533 forecast day raised on a parked slot in its first step
            # and its receipt read host_spill null and sizer null, while
            # the plan was on stdout only.  Both are descriptive, in no
            # identity, and the successful receipt already carried them.
            "host_spill": _host_spill_receipt(cfg, model, plan=door_plan),
            "sizer": None if door_plan is None else door_plan.receipt(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "software_stack": stack,
            "completed_step": int(state.step),
            "completed_time_s": float(state.time_s),
            "run_trackers": trackers,
            "supplementary_trackers": supplementary,
            # The step's headroom up to the failure: a CFL refusal on a
            # day stronger than the shipped step was set against is
            # explained here, in the step this day would have admitted.
            "cfl": cfl_headroom_receipt(cfg, trackers),
            **({"semilag_tracer_fixer_by_species": species_fixer}
               if species_fixer else {}),
            "positivity_fixer_ledger": fixer_ledger.record(),
            "stratospheric_floor_ledger": floor_ledger.record(),
            # What the radiation ran with up to the failure: a dead run's
            # receipt is the one artifact that has to say it.
            "radiation_greenhouse_gases": _greenhouse_gas_receipt(
                cfg, getattr(model, "forecast_clock", None)),
            "checkpoints": [*earlier_checkpoints, *checkpoints],
            **({} if tapes is None else {"tapes": (
                tape_writer.receipt() if tape_writer is not None
                else {"outdir": str(tapes.outdir), "refused": tapes_refused})}),
            # The peak up to the failure: an out-of-memory death is
            # exactly when this number is wanted.
            "device_memory": device_memory_receipt(
                tracker, cfg.backend,
                sizing_model_peak_bytes=predicted_peak_bytes,
                allocator=allocator,
            ),
            "restart": None if restart_metadata is None else {
                "path": str(restart),
                "checkpoint_self_sha256": restart_metadata["self_sha256"],
                "absent_carried_physics_arrays": restart_metadata.get(
                    "absent_carried_physics_arrays", []),
            },
            "wall_seconds": float(time.perf_counter() - start_wall),
            "step_wall": _step_wall_receipt(step_count, step_wall_seconds),
            # The lighter plans this run moved to after the card ran out
            # of memory (run), when it moved at all.
            **({"recovery": list(recovery)} if recovery else {}),
        }
        if ledger is not None:
            # The failure receipt is the post-mortem; a ledger fault here
            # must not cost the receipt of the model's own failure.
            try:
                failure["insitu"] = ledger.on_failure(state)
            except Exception as ledger_exc:  # noqa: BLE001 - recorded, not raised
                failure["insitu"] = {
                    "error_type": type(ledger_exc).__name__,
                    "error_message": str(ledger_exc),
                }
        write_receipt(receipt_path, failure)
        raise

    if attempt is not None:
        attempt["loop_done"] = True
    # Every checkpoint is on disk before the receipt that lists it.
    writer.close()
    # And every tape.
    if tape_writer is not None:
        tape_writer.close()
    forcing_rows: dict[str, object] = {}
    if forcing_op is not None:
        forcing_rows["forcing"] = forcing_op.finish(output, end_s=state.time_s)
    if record_writer is not None:
        forcing_rows["forcing_record"] = record_writer.write(record_out)
    products.finish(run_complete=state.step >= int(round(cfg.duration_s / cfg.dt_s)))
    retention.prune()
    final_diag = model.diagnostics(state)
    # The cards block is built BEFORE the gates, because gate WIRE-1 is
    # one of them and it reads the achieved rate out of this block.
    cards_receipt = (
        model.pipeline.exchange.session.receipt()
        if getattr(model.pipeline, "exchange", None) is not None
        else {"cards": 1, "transport": "single", "card_exchange": "gather"}
    )
    gates = run_gates(
        cfg, transform_check, final_diag, target_mass, target_water,
        trackers, supplementary, cards=cards_receipt,
    )
    status = "pass" if all(row["passed"] for row in gates.values()) else "fail"
    receipt = {
        "name": cfg.name,
        "status": status,
        "config_hash": cfg.config_hash,
        "config": cfg.config_identity,
        # The world this run integrated on (label, identity, settings),
        # beside the config identity it already joins.  ABSENT for Earth, so
        # an Earth receipt has the key set it had before worlds existed.
        **world_receipt_rows(world_of(cfg)),
        # The pin of the arithmetic this run integrated: the scheme's
        # (pins.SEMI_IMPLICIT_PINS), the same pin its checkpoints carry.
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "pins": pins_receipt(cfg.semi_implicit_scheme, cfg.integrator),
        "model": "moist-hybrid-spectral",
        "physics_mode": cfg.physics_mode,
        "physics_split": cfg.physics_split,
        "vertical": vertical_grid_receipt(cfg),
        # The scheme and, under vertical_modes, the reference operator's
        # phase speeds and equivalent depths the run integrated with.
        "semi_implicit": model.semi_implicit.describe(model.vertical),
        "physics_identity": None if model.physics is None else model.physics.identity,
        "initial": {
            "mode": cfg.initial_mode,
            "provenance": model.initial_provenance,
        },
        # The instant model time zero stands for and where it came from
        # (arwen_global.clock): the one copy every reader of this run's
        # valid times takes.
        "forecast_clock": (
            None if model.forecast_clock is None
            else model.forecast_clock.receipt()
        ),
        # The planet the land surface ran on, named at the top level so a
        # reader never has to know the synthetic arm is a provenance leaf.
        "statics": (
            None if not isinstance(model.initial_provenance, dict)
            else model.initial_provenance.get("statics")
        ),
        "transform": transform.identity,
        # The four memory levers as this run was built, DESCRIPTIVE and in
        # no hash: two of them are deliberately absent from every identity
        # (synthesis_memo, streaming) and a run that names them left no
        # trace at all -- a streamed run, whose analyses cost 15,144x a
        # resident-table call, read back identical to a resident one.
        "memory": {
            "spectral_chunk": int(cfg.spectral_chunk),
            "synthesis_memo": bool(cfg.synthesis_memo),
            "legendre_band": int(cfg.legendre_band),
            "streaming": bool(cfg.streaming),
        },
        # The band schedule grid space streamed through: the count, the
        # rows it cut them into, and whether the sizer chose it.  It is
        # NOT part of the identity (it changes no arithmetic), so it is
        # recorded where a reader can see what the run did rather than
        # where a hash would refuse a restart for it.
        "latitude_bands": latitude_bands_receipt(model, cfg, sized_by_door,
                                                     plan=door_plan),
        "surface_update": _surface_update_receipt(model),
        "surface_boundary": _surface_boundary_receipt(model, cfg),
        "host_spill": _host_spill_receipt(cfg, model, wall_steps=step_count,
                                          wall_seconds=step_wall_seconds,
                                          plan=door_plan),
        # THE WALL THE STEPS TOOK, on every receipt.  The tier's block
        # above carries a per-step figure only when something is parked;
        # a resident run had none, and its step could only be read off
        # the whole run's wall with the build, the cold start and the
        # checkpoint writes inside it.  The capacity rows and the two-card
        # ratio are defined on a step, so every receipt carries one
        # (2026-09-07, the cards lane's T383 pair measurement).
        "step_wall": _step_wall_receipt(step_count, step_wall_seconds),
        # WHAT THE SIZER CHOSE, AND AGAINST WHAT.  The band count and the
        # host tier are one decision (sizing.plan_run_memory), and this is
        # the door's own record of it: the free bytes it read before a
        # byte was allocated, the quarter-free budget it weighed against,
        # the fragmentation and out-of-pool terms it charged, the card
        # figure that came out, and one sentence saying what it took and
        # why.  A bare run's receipt says what a bare run did, which is
        # what a reader who set no flag needs and could not get before.
        # Absent only when the run was not started through a door.
        "sizer": None if door_plan is None else door_plan.receipt(),
        # The cards this rank shared the band schedule with, the transport
        # it used, and what crossed the wire.  Gate WIRE-1 reads
        # "achieved_gb_s" and refuses a run below the floor: every
        # interconnect figure the design was priced on was taken on IDLE
        # cards, and this is the same quantity taken while both cards
        # compute.  A single-card run records the same block with a world
        # of one and zero bytes, so a reader never has to ask whether the
        # field was simply absent.
        "cards": cards_receipt,
        "transform_check": transform_check,
        "cold_start_diagnostics": cold_diag,
        "segment_start_diagnostics": initial_segment_diag,
        "final_diagnostics": final_diag,
        "mass_target_pa": target_mass,
        "total_water_target_kg_m2": target_water,
        "gates": gates,
        "run_trackers": trackers,
        "supplementary_trackers": supplementary,
        # The day's CFL against the gate, the wind it implies and the step
        # it would have admitted at the gate and at the shipped rule.
        "cfl": cfl_headroom_receipt(cfg, trackers),
        # The spectral eddy viscosity's last reading (nu at the plateau per
        # level, the energy at the truncation it was read from); None under
        # the hyperdiffusion.
        "closure": getattr(model, "closure_record", None),
        # What the native radiation radiated: the greenhouse-gas mole
        # fractions of the run's valid year and the table they came from.
        "radiation_greenhouse_gases": _greenhouse_gas_receipt(
                cfg, getattr(model, "forecast_clock", None)),
        **({"semilag_tracer_fixer_by_species": species_fixer}
           if species_fixer else {}),
        "positivity_fixer_ledger": fixer_ledger.record(),
        "stratospheric_floor_ledger": floor_ledger.record(),
        # Gates LR-0 to LR-9 over every diagnostics row this folder holds,
        # on runs longer than 384 h only.  Read-only and outside ``status``:
        # their thresholds are engineering proposals (longrun_gates).
        **({"longrun_gates": longrun_receipt_block(
            diagnostics_path,
            getattr(model.forecast_clock, "start_utc", None))}
           if cfg.duration_s > LONG_RUN_THRESHOLD_S else {}),
        "checkpoints": [*earlier_checkpoints, *checkpoints],
        # The run's own tapes (run --tapes), in no identity: where they
        # went, which were written, what each cost, which failed (left to
        # export), or why the run wrote none.
        **({} if tapes is None else {"tapes": (
            tape_writer.receipt() if tape_writer is not None
            else {"outdir": str(tapes.outdir), "refused": tapes_refused})}),
        "long_run_outputs": {
            "retention": retention.receipt(),
            "products": {**products.receipt(), "resume": products_resume},
            "launch_estimate": launch_estimate,
        },
        "software_stack": stack,
        "segment_until_s": until_s,
        **({"branch": {
            "opt_in": "branch_config (--branch-from-other-config)",
            "parent_config_hash": restart_metadata["config_hash"],
            "config_hash": cfg.config_hash,
            "config_differs": restart_metadata["config_hash"] != cfg.config_hash,
            "parent_checkpoint_self_sha256": restart_metadata["self_sha256"],
        }} if branch_config else {}),
        "restart_targets": restart_targets,
        # Where open water's surface water is booked, and whether this
        # segment had to give the state a zero exchange account (a cold
        # start, or a checkpoint from before the account existed).
        "open_water_exchange": {
            "enabled": bool(model.open_water_exchange),
            "account": OCEAN_EXCHANGE_NAME if model.open_water_exchange else None,
            "filled": exchange_account_fill,
        },
        "restart": None if restart_metadata is None else {
            "path": str(restart),
            "checkpoint_self_sha256": restart_metadata["self_sha256"],
            "inherited_run_trackers": restart_metadata["run_trackers"],
            "absent_carried_physics_arrays": restart_metadata.get(
                "absent_carried_physics_arrays", []),
        },
        "wall_seconds": float(time.perf_counter() - start_wall),
        # The lighter plans this run moved to after the card ran out of
        # memory, each with what the card showed (run); absent when it
        # never moved.  The trackers above come from the checkpoints and
        # cover the whole run; the receipt-only ledgers and the wall cover
        # this last attempt, from the checkpoint it carried on from.
        **({"recovery": list(recovery)} if recovery else {}),
        "insitu": None if ledger is None else ledger.close(),
        # MEASURED at the allocator over the whole run (tables, initial
        # state, every step); the sizing model's pre-run prediction rides
        # beside it under its own name.  See device_memory.py.
        "device_memory": device_memory_receipt(
            tracker, cfg.backend,
            sizing_model_peak_bytes=predicted_peak_bytes,
            allocator=allocator,
        ),
        # Present only on a run that asked for the forcing operator or a
        # record of itself (arwen_global.forcing); a run that did not
        # carries neither key.
        **forcing_rows,
    }
    write_receipt(receipt_path, receipt)
    checked = json.loads(receipt_path.read_text(encoding="utf-8"))
    checked["receipt_path"] = str(receipt_path)
    return checked


def wire_gate_row(cards) -> dict | None:
    """Gate WIRE-1's row, or None on a single card.

    Over TCP the rate is the run's own: bytes over the time the sender
    threads spent in ``sendall`` while both cards compute.  Over NCCL that
    quotient is not a link rate: a collective's device time includes its
    wait on the slowest rank, so a rank that finished its compute first
    reads its peer's compute as wire time (MEASURED 2026-10-05: 0.03 GB/s
    on the waiting rank of a T31 run), and the floor would fail a healthy
    link.  There the gate reads the device link probe the transport took
    at session open, every rank released together
    (cards.NcclCards._probe_link), and the in-run rate stays in the
    receipt as the lower bound it is.
    """
    wire = dict((cards or {}).get("wire") or {})
    if int((cards or {}).get("cards", 1) or 1) <= 1 or not wire:
        return None
    floor = float(wire.get("wire_floor_gb_s", 0.0))
    probe = (cards or {}).get("link_probe_gb_s")
    if (cards or {}).get("transport") == "nccl" and probe is not None:
        value, measured = float(probe), "device link probe at session open"
    else:
        value = float(wire.get("achieved_gb_s", 0.0))
        measured = "bytes over sender time while both cards compute"
    return {
        "value": value,
        "limit": floor,
        "direction": "floor",
        "measured": measured,
        "passed": bool(value >= floor),
    }


def run_gates(
    cfg: ArwenGlobalConfig,
    transform_check: dict,
    final_diag: dict,
    target_mass: float,
    target_water: float,
    trackers: dict[str, float],
    supplementary: dict[str, float],
    cards: dict | None = None,
) -> dict[str, dict[str, object]]:
    """The receipt's gates, each with its value, limit and verdict: the
    transform controls, the mass and water drift against the epoch's
    targets, what the fixers and the physics repair absorbed per step,
    and on a multi-card run the rate the wire actually achieved.  One
    function so a run and a cycle judge themselves alike.

    Every row but one is a CEILING: the value must not exceed the limit.
    Gate WIRE-1 is a FLOOR and carries ``direction`` to say so."""
    mass_drift = _relative(
        final_diag["global_mean_surface_pressure_pa"], target_mass
    )
    water_drift = _relative(
        final_diag["global_mean_total_water_kg_m2"], target_water
    )
    gates = {
        "transform_roundtrip_relative_linf": {
            "value": transform_check["roundtrip_relative_linf"],
            "limit": cfg.gate_transform_roundtrip,
        },
        "transform_parseval_relative_error": {
            "value": transform_check["parseval_relative_error"],
            "limit": cfg.gate_transform_parseval,
        },
        "mass_relative_drift": {
            "value": mass_drift,
            "limit": cfg.gate_mass_relative_drift,
        },
        "total_water_relative_drift": {
            "value": water_drift,
            "limit": cfg.gate_total_water_relative_drift,
        },
        "mass_fixer_max_step_log_offset": {
            "value": trackers["maximum_mass_fixer_log_offset"],
            "limit": fixer_absorption_limits(cfg.precision)["mass_log_offset"],
        },
        "water_fixer_max_step_relative": {
            "value": trackers["maximum_global_water_fixer_kg_m2"]
            / max(abs(target_water), 1.0e-30),
            "limit": fixer_absorption_limits(cfg.precision)["water_relative"],
        },
        "physics_water_repair_max_step_kg_m2": {
            "value": trackers["maximum_physics_water_repair_kg_m2"],
            # The config-level override exists for suites with a measured
            # kernel-internal closure floor (see _FIXER_ABSORPTION_LIMITS);
            # unset, the precision envelope stands.
            "limit": (
                fixer_absorption_limits(cfg.precision)["physics_repair_kg_m2"]
                if cfg.gate_physics_water_repair_max_step_kg_m2 is None
                else cfg.gate_physics_water_repair_max_step_kg_m2
            ),
        },
        "positivity_fixer_max_step_relative": {
            "value": supplementary["maximum_positivity_fixer_relative"],
            "limit": (
                POSITIVITY_FIXER_RELATIVE_LIMIT
                if cfg.gate_positivity_fixer_max_step_relative is None
                else cfg.gate_positivity_fixer_max_step_relative
            ),
        },
    }
    if cfg.integrator in SEMILAG_INTEGRATORS:
        # Only a run that HAS a semi-Lagrangian tracer fixer carries these
        # rows, so every receipt written by every other integrator keeps
        # the gate inventory it had.  Two rows, because the fixer has two
        # separable failures and one number cannot see both.  The water
        # row is the conservation gate and names the breakage: a tracer
        # scheme that is not conservative silently loses or gains water
        # over a forecast day, the class of defect the grid tracers were
        # introduced to retire.  The per-species row is an accuracy gate
        # and names a different one: a species whose field owes more to
        # the fixer's weighting than to the flow forecasts a plausible
        # amount of the wrong thing in the wrong place.
        gates["semilag_tracer_mass_fixer_max_step_water_relative"] = {
            "value": supplementary[
                "maximum_semilag_tracer_mass_fixer_water_relative"
            ],
            "limit": SEMILAG_TRACER_FIXER_WATER_RELATIVE_LIMIT,
        }
        gates["semilag_tracer_mass_fixer_max_step_relative"] = {
            "value": supplementary["maximum_semilag_tracer_mass_fixer_relative"],
            "limit": SEMILAG_TRACER_FIXER_RELATIVE_LIMIT,
        }
        gates["semilag_trajectory_convergence_cells"] = {
            "value": supplementary["maximum_semilag_trajectory_move_cells"],
            "limit": float(cfg.semilag.trajectory_convergence_cells),
        }
        # The trajectory fold gate (semilag.trajectory.refuse_trajectory_
        # fold), a FLOOR: the smallest det(I -+ (dt/2) J) the run met.  It
        # replaced a ceiling on the deformation norm, which read shear and
        # rotation as folding; that norm stays in the receipt as
        # maximum_semilag_lipschitz and is not gated.
        gates["semilag_trajectory_fold_determinant"] = {
            "value": supplementary["minimum_semilag_fold_determinant"],
            "limit": float(cfg.minimum_fold_determinant),
            "direction": "floor",
        }
    for row in gates.values():
        direction = row.setdefault("direction", "ceiling")
        if direction == "floor":
            row["passed"] = bool(float(row["value"]) >= float(row["limit"]))
        else:
            row["passed"] = bool(float(row["value"]) <= float(row["limit"]))
    # Gate WIRE-1.  It exists on a MULTI-CARD run only, because a single
    # card moves no bytes and a floor on zero would fail every run that
    # never opened a socket.
    #
    # The breakage it prevents, named: every interconnect figure this
    # design was priced on was taken on IDLE cards (3.09 GB/s MEASURED),
    # and the whole two-card case rests on the wire keeping up with the
    # compute.  MEASURED 2026-09-06 on the T255 ten-step probe, the rate
    # ACHIEVED while both cards compute fell to 1.82 to 2.31 GB/s across
    # four rank-legs -- a quarter below the priced rate, and three of the
    # four below this floor.  Without this row a run reports a speedup
    # priced on a link it did not get, and the receipt says "pass".
    row = wire_gate_row(cards)
    if row is not None:
        gates["two_card_wire_achieved_gb_s"] = row
    # A gather run reproduces one card only where its cards return the same
    # bits for every contraction the step presents.  Under
    # card_agreement="record" a disagreeing pair carries on and this row is
    # what says the answer is neither card's: the count of disagreeing
    # shapes, ceiling zero.  Under "refuse" the run never reaches a receipt
    # with a disagreement in it, so the row reads zero and passes.
    agreement = dict((cards or {}).get("card_agreement") or {})
    if int((cards or {}).get("cards", 1) or 1) > 1 and agreement:
        disagreeing = len(agreement.get("disagreements") or [])
        gates["two_card_contractions_agree"] = {
            "value": disagreeing,
            "limit": 0,
            "direction": "ceiling",
            "passed": bool(disagreeing == 0),
        }
    return gates


__all__ = [
    "adopt_checkpoint_clock",
    "build_model_and_cold_state", "build_physics", "build_transform",
    "frame_world_reason", "run", "run_gates", "world_of",
]
