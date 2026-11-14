"""The slab ocean and thermodynamic sea ice in the run: state, step, records.

Design: docs/ARWEN_GLOBAL_OCEAN.md.  One provider of the boundary service
(arwen_global.surface_boundary), :class:`SlabProvider`, with one more
method than the rules have, :meth:`SlabProvider.advance`, which the
service calls first in every ``apply`` (before the model's step):

1. advance the mixed layer and the ice mass from the last update to the
   state's time, with the fluxes the physics held over that interval
   (the previous step's, evaluated at the skin the previous step saw);
2. at an ice record (``[sea_ice] record_interval_hours``), hand the
   prognostic fraction and thickness to the service, which writes them
   and re-classes a column that crosses the ice threshold, as it does for
   the rules;
3. write the mixed-layer temperature as the skin of every open-ocean
   column (water class, not a lake, inside the climatology's ocean).

Everything prognostic lives in the physics state under the names below,
float64, so a checkpoint carries it (``physics__<name>``), a restart
resumes it bit for bit, and a run with the ocean off carries none of it.
The calibration of the heat-flux correction (``[ocean] mode =
"climatology"`` with ``qflux_calibration = true``) runs the same flux
arithmetic through :class:`SlabProvider` in a mode that writes nothing to
the surface and only deposits (arwen_global.ocean_qflux.deposit); the
service then carries the climatology rule as its provider and this one as
its coupler.

The ocean columns are the rules' ice columns (arwen_global.ocean_rules):
inside the climatology's ocean, land fraction at most one half, not an
inland lake.  The flux the slab takes on a column is F_open, the net heat
into open water over the interval:

* the native suite: ``swdown (1 - albedo) + emissivity (glw - sigma
  skin^4) - hfx - L_v qfx`` from the planes it holds and the albedo,
  emissivity and skin the radiation saw; on a frozen-class column (the
  surface's ice fraction A_s at least one half) the lead tile's, ``(1 -
  A_s)`` times the same balance with ``lead_hfx``, ``lead_qfx``, the
  open-water albedo 0.08, emissivity 0.98 and the lead skin at the
  column's base temperature;
* the reference suite: the net surface flux it held for the interval
  (:data:`HELD_NET_FLUX`, written by physics/reference.py only when this
  plane is in the state, which only a slab or a calibration puts there).
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import math

import numpy as np

#: physics_state.arrays names, (nlat, nlon) float64 unless stated.
#: The mixed-layer temperature (K; 0 off the ocean columns).
SLAB_TEMPERATURE = "ocean_mixed_layer_temperature_k"
#: h at the last update (m), held for the tape.
SLAB_DEPTH = "ocean_mixed_layer_depth_m"
#: Q at the last update (W/m2), held for the tape and the rows.
SLAB_QFLUX = "ocean_qflux_w_m2"
#: The prognostic ice: fraction (0-1) and cell-mean volume (m).
ICE_FRACTION = "ocean_ice_fraction"
ICE_VOLUME = "ocean_ice_volume_m"
#: The energy books (J/m2, cumulative from the start): what entered the
#: ocean-and-ice-mass system, sum of dt (F_open + Q + R + A (M_top -
#: G_up)), and what it stored, sum of rho_w c_w h dT - rho_i L_f dV.  Gate
#: OC-3 holds them equal.
ENERGY_SOURCES = "ocean_energy_sources_j_m2"
ENERGY_STORED = "ocean_energy_stored_j_m2"
#: The calibration accumulators, (12, nlat, nlon) float64, present only
#: under ``qflux_calibration``.
QCAL_SUM = "ocean_qflux_calibration_sum_j_m2"
QCAL_WEIGHT = "ocean_qflux_calibration_weight_s"
SLAB_ARRAYS = (SLAB_TEMPERATURE, SLAB_DEPTH, SLAB_QFLUX, ICE_FRACTION, ICE_VOLUME,
               ENERGY_SOURCES, ENERGY_STORED)
CALIBRATION_ARRAYS = (QCAL_SUM, QCAL_WEIGHT)
#: The reference suite's net surface flux over the last step (W/m2, positive
#: down), held only when the slab or the calibration put this plane in the
#: state at the cold start (physics/reference.py writes it where present).
HELD_NET_FLUX = "ocean_held_net_surface_flux_w_m2"
#: physics_state.metadata key: {"last_update_s": float, "advances": int,
#: "record": the ice record last written (None before the first)}.
METADATA_KEY = "ocean_slab"
#: Tape fields written only when the slab is on (a run with the ocean off
#: writes the tape it wrote before, byte for byte).  SEAICE and ICEDEPTH
#: are already on every tape, from the surface state.
TAPE_SST = "SST"
TAPE_MIXED_LAYER_DEPTH = "HML"
TAPE_QFLUX = "ARWEN_QFLUX"
#: The open water of the leads (physics/frozen_surface.py, WRF's values).
LEAD_ALBEDO = 0.08
LEAD_EMISSIVITY = 0.98
SECONDS_PER_DAY = 86400.0


def _resident(xp, value):
    from .spill import resident, spilled

    return resident(xp, value) if spilled(value) else value


def _host(value):
    from .surface_boundary import _host as host

    return host(value)


def _module_of(value):
    """The array module a state plane lives on (a parked plane is worked on
    the card, as the boundary service does)."""
    from .surface_boundary import SurfaceBoundary

    return SurfaceBoundary._xp(value)


def _put_array(physics_state, name: str, value) -> None:
    """Write a whole physics-state plane: the parked slot through the tier,
    otherwise a new array in the namespace (a state's arrays are never
    written in place: a checkpoint's writer thread may be hashing them)."""
    from .spill import spilled

    target = physics_state.arrays.get(name)
    if target is not None and spilled(target):
        target.store(value)
    else:
        physics_state.arrays[name] = value


def physics_identity(cfg) -> dict:
    """The identity of the physics a Q-flux file is valid for: the native
    adapter's identity of its options
    (arwen_global.physics.registry.identity_of_global_physics_options), or
    the reference suite's options identity."""
    if cfg.physics_mode == "arwen-native" and getattr(cfg, "native_adapter_name", None):
        from .physics.builtin_adapters import ensure_builtin_global_physics_adapters
        from .physics.registry import identity_of_global_physics_options

        ensure_builtin_global_physics_adapters()
        return {"mode": "arwen-native", "adapter": cfg.native_adapter_name,
                "options": identity_of_global_physics_options(
                    cfg.native_adapter_name, cfg.native_adapter_options)}
    if cfg.physics_mode == "reference":
        return dict(cfg.reference_physics.identity)
    return {"mode": str(cfg.physics_mode)}


def _sha(payload) -> str:
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class SlabProvider:
    """The slab ocean as a record provider of the boundary service.

    Provider contract (surface_boundary.py): ``record_index``, ``sst_at``,
    ``ice_record``, ``record_sst_host``, ``record_valid``, ``last_info``,
    ``receipt``; plus ``identity_block`` and ``diagnostics`` as the rule
    provider has them, and :meth:`advance`.
    """

    kind = "slab"

    def __init__(self, options, climatology, qflux, grid, *, start_time,
                 cold_surface: dict, model_truncation: int, rule=None,
                 calibrate: bool = False, physics_mode: str = "arwen-native",
                 physics_identity: dict | None = None, duration_s: float | None = None):
        """``options`` the run's OceanOptions; ``climatology`` a
        SurfaceClimatology (with ``mixed_layer_depth_m`` when the depth is
        ``"woa18"``); ``qflux`` a QFluxFile (None when calibrating);
        ``cold_surface`` the host planes ocean_rules.cold_surface_planes
        reads; ``rule`` the RuleProvider whose SST is the restoring target
        under ``slab_target = "anomaly"`` (None otherwise), or, calibrating,
        the climatology rule the run is driven by.  ``physics_mode`` says
        which flux planes the run holds (the reference suite holds
        :data:`HELD_NET_FLUX`); ``physics_identity`` and ``duration_s`` are
        what a Q-flux file is checked against (and what a calibration
        records)."""
        from .ocean_slab import SlabParameters
        from .statics import lake_columns
        from .surface_series import _utc

        if climatology is None:
            raise ValueError("the slab ocean and its calibration read a climatology")
        self.options = options
        self.calibrate = bool(calibrate)
        self.clim = climatology
        self.qflux_file = qflux
        self.grid = grid
        self.rule = rule
        self.start = _utc(start_time)
        self.physics_mode = str(physics_mode)
        self.physics_identity = physics_identity
        self.model_truncation = int(model_truncation)
        lat = np.asarray(grid.latitude_deg, dtype=np.float64)
        lon = np.asarray(grid.longitude_deg, dtype=np.float64)
        self.latitude_deg, self.longitude_deg = lat, lon
        climatology.check_grid(lat, lon)
        f = {name: np.asarray(value, dtype=np.float64) for name, value in cold_surface.items()}
        self.cold = f
        land = f["land_fraction"].astype(np.float32)
        lake = f["lake_fraction"].astype(np.float32)
        ice0 = f["sea_ice_fraction"].astype(np.float32)
        lake0 = np.asarray(lake_columns(land, lake, np, sea_ice_fraction=ice0))
        lake_any = np.asarray(lake_columns(land, lake, np))
        clim_ocean = np.asarray(climatology.ocean, dtype=bool)
        #: The ocean columns: the rules' ice columns.
        self.ocean = clim_ocean & (land <= np.float32(0.5)) & ~lake_any & ~lake0
        self.record_s = float(options.sea_ice.record_interval_hours) * 3600.0
        self.new_ice_thickness_m = float(options.sea_ice.new_ice_thickness_m)
        self.spinup_s = float(options.qflux_calibration_spinup_days) * SECONDS_PER_DAY
        self.params = None
        if not self.calibrate:
            depth = options.slab_depth
            if depth == "woa18" and climatology.mixed_layer_depth_m is None:
                raise ValueError(
                    f"ocean.slab_depth = \"woa18\" reads the climatology's mixed-layer "
                    f"depth and {climatology.path} carries none; build it with "
                    "`gpuwm-global climatology build CONFIG --with-mld`, or give the "
                    "depth in metres")
            self.params = SlabParameters(
                depth=depth if depth == "woa18" else float(depth),
                depth_bounds_m=tuple(float(b) for b in options.slab_depth_bounds_m),
                relaxation_days=options.slab_relaxation_days,
                target=options.slab_target)
            if qflux is None:
                raise ValueError("[ocean] mode = \"slab\" reads a Q-flux file")
            if options.slab_target == "anomaly" and options.slab_relaxation_days is not None \
                    and rule is None:
                raise ValueError("slab_target = \"anomaly\" restores toward the anomaly "
                                 "rule's SST, and no rule was given")
            from .ocean_qflux import months_needed

            months = (set() if duration_s is None
                      else set(months_needed(self.start, float(duration_s))))
            self.months = sorted(months)
            qflux.check(latitude_deg=lat, longitude_deg=lon,
                        physics_identity=physics_identity or {},
                        climatology_sha256=climatology.sha256, months=months)
        elif rule is None:
            raise ValueError("a calibration runs under the climatology rule, and none "
                             "was given")
        self._href = None
        self._device: dict = {}
        self._state = None
        self._last_t = None
        self.stats: dict[str, object] = {
            "ocean_columns": int(self.ocean.sum()),
            "advances_this_segment": 0, "deposits_this_segment": 0,
            "records_written": 0, "records_already_written": 0,
        }

    # -- helpers -------------------------------------------------------------

    @property
    def reference_suite(self) -> bool:
        return self.physics_mode == "reference"

    def when(self, t: float):
        return self.start + timedelta(seconds=float(t))

    def _on(self, xp, name, value):
        key = (xp.__name__, name)
        if key not in self._device:
            self._device[key] = xp.asarray(value)
        return self._device[key]

    def _ice_reference(self, xp):
        if self._href is None:
            from .sea_ice_thermo import ice_reference_thickness

            self._href = np.asarray(ice_reference_thickness(
                self.latitude_deg, int(self.longitude_deg.size), np), dtype=np.float64)
        return self._on(xp, "href", self._href)

    def _depth(self, when, xp):
        from .ocean_slab import mixed_layer_depth

        ocean = self._on(xp, "ocean", self.ocean)
        return xp.asarray(mixed_layer_depth(self.clim, when, self.params, ocean, xp),
                          dtype=xp.float64)

    def _qflux(self, when, depth, xp, iced, fraction, until=None):
        """Q on the ocean columns, over the step from ``when`` to ``until``
        (the slopes are the climatology curve's change over it; None: its
        slope at ``when``).

        Where the slab holds ice (``iced``) the mixed layer sits at the
        freezing point and every watt of the ocean's heat goes to the ice, so
        the climatology's SST slope (the heat that warms or cools open
        water) is left out there: given to the pack, it melts or grows ice
        at a rate the climatology's ice slope does not ask for. The ice
        slope's term stays everywhere.

        Under ice a negative Q is also carried only in the share
        ocean_slab.under_ice_cooling_share gives for the slab's ice
        ``fraction`` against the climatology's: that cooling is the heat
        the calibration's leads took in and the ocean carried off, and a
        pack more closed than the climatology's takes in less of it. Charged
        whole, it grew the southern summer pack from below (-78 W/m2 in
        December, -87 in January, on the 180-day coupled run's cells where
        both held ice), and the new ice it formed closed the leads further."""
        from .ocean_slab import climatology_tendency, qflux, under_ice_cooling_share

        sst_rate, ice_rate = climatology_tendency(self.clim, when, xp, until=until)
        iced = xp.asarray(iced, dtype=bool)
        sst_rate = xp.where(iced, 0.0, sst_rate)
        value = qflux(depth_m=depth, sst_rate_k_s=sst_rate, ice_fraction_rate_s=ice_rate,
                      ice_reference_thickness_m=self._ice_reference(xp),
                      calibration_flux_w_m2=self.qflux_file.at(when, xp), xp=xp)
        value = xp.nan_to_num(xp.asarray(value, dtype=xp.float64))
        share = under_ice_cooling_share(
            ice_fraction=fraction,
            climatology_ice_fraction=self.clim.at("sea_ice_fraction", when, xp), xp=xp)
        value = xp.where(iced & (value < 0.0), value * share, value)
        ocean = self._on(xp, "ocean", self.ocean)
        return xp.where(ocean, value, 0.0)

    def _target(self, t: float, when, xp):
        if self.options.slab_target == "anomaly":
            target = self.rule.sst_host(t)
        else:
            target = self.clim.at("sst_k", when).astype(np.float64)
        ocean = self.ocean
        return xp.asarray(np.where(ocean, np.nan_to_num(target, nan=0.0), 0.0))

    # -- the cold start ------------------------------------------------------

    def initial_arrays(self, cold_surface: dict) -> dict:
        """The :data:`SLAB_ARRAYS` (and, calibrating, the
        :data:`CALIBRATION_ARRAYS`) of a cold start, host float64; with the
        reference suite, the zero :data:`HELD_NET_FLUX` plane it fills."""
        f = {name: np.asarray(value, dtype=np.float64) for name, value in cold_surface.items()}
        shape = f["land_fraction"].shape
        out: dict[str, np.ndarray] = {}
        if self.calibrate:
            out[QCAL_SUM] = np.zeros((12, *shape), dtype=np.float64)
            out[QCAL_WEIGHT] = np.zeros((12, *shape), dtype=np.float64)
        else:
            from .ocean_slab import initial_temperature
            from .sea_ice_thermo import initial_ice

            ocean = self.ocean
            ice = np.where(ocean, f["sea_ice_fraction"], 0.0)
            fraction, volume = initial_ice(ice, f["sea_ice_thickness_m"], ocean,
                                           self.new_ice_thickness_m, np)
            fraction = np.asarray(fraction, dtype=np.float64)
            out[SLAB_TEMPERATURE] = np.where(ocean, np.asarray(initial_temperature(
                f["temperature_k"], ocean, fraction, np), dtype=np.float64), 0.0)
            depth = self._depth(self.start, np)
            out[SLAB_DEPTH] = np.where(ocean, depth, 0.0)
            out[ICE_FRACTION] = np.where(ocean, fraction, 0.0)
            out[ICE_VOLUME] = np.where(ocean, np.asarray(volume, dtype=np.float64), 0.0)
            out[SLAB_QFLUX] = self._qflux(self.start, depth, np,
                                          ocean & (out[ICE_VOLUME] > 0.0), out[ICE_FRACTION])
            out[ENERGY_SOURCES] = np.zeros(shape, dtype=np.float64)
            out[ENERGY_STORED] = np.zeros(shape, dtype=np.float64)
        if self.reference_suite:
            out[HELD_NET_FLUX] = np.zeros(shape, dtype=np.float64)
        return out

    def initialize(self, state) -> None:
        """Put the cold-start arrays and the metadata into ``state``'s
        physics namespace, on the array module its surface lives on."""
        from .ocean_rules import cold_surface_planes

        xp = _module_of(state.surface.temperature_k)
        for name, value in self.initial_arrays(cold_surface_planes(state.surface)).items():
            _put_array(state.physics_state, name, xp.asarray(value))
        state.physics_state.metadata[METADATA_KEY] = {
            "last_update_s": float(state.time_s), "advances": 0, "record": None}

    def check_state(self, state, label: str) -> None:
        """Refuse a restart state that does not carry this provider's
        arrays (the identity block is checked before this)."""
        names = CALIBRATION_ARRAYS if self.calibrate else SLAB_ARRAYS
        missing = [name for name in names if name not in state.physics_state.arrays]
        if self.reference_suite and HELD_NET_FLUX not in state.physics_state.arrays:
            missing.append(HELD_NET_FLUX)
        if missing or METADATA_KEY not in state.physics_state.metadata:
            raise ValueError(
                f"{label} carries no {'calibration' if self.calibrate else 'slab'} state "
                f"({', '.join(missing) or METADATA_KEY} missing); a restart must "
                "continue the ocean it was written with")

    # -- step 1: advance -----------------------------------------------------

    def _open_water_flux(self, state, xp, frozen):
        """F_open on every column (masked by the caller), W/m2 float64."""
        arrays = state.physics_state.arrays
        if HELD_NET_FLUX in arrays:
            return xp.asarray(_resident(xp, arrays[HELD_NET_FLUX]), dtype=xp.float64)
        from .ocean_slab import open_water_flux

        def plane(name):
            return xp.asarray(_resident(xp, arrays[name]), dtype=xp.float64)

        def surface(name):
            return xp.asarray(_resident(xp, getattr(state.surface, name)), dtype=xp.float64)

        swdown, glw = plane("swdown"), plane("glw")
        water = open_water_flux(
            swdown=swdown, albedo=surface("albedo"), glw=glw,
            emissivity=surface("emissivity"), skin_k=surface("temperature_k"),
            hfx=plane("hfx"), qfx=plane("qfx"), xp=xp)
        if frozen is None:
            return xp.asarray(water, dtype=xp.float64)
        from .physics.frozen_surface import SEA_ICE_BOTTOM_K

        cover = xp.clip(surface("sea_ice_fraction"), 0.0, 1.0)
        lead = open_water_flux(
            swdown=swdown, albedo=xp.full_like(swdown, LEAD_ALBEDO), glw=glw,
            emissivity=xp.full_like(swdown, LEAD_EMISSIVITY),
            skin_k=xp.full_like(swdown, SEA_ICE_BOTTOM_K),
            hfx=plane("lead_hfx"), qfx=plane("lead_qfx"), xp=xp)
        return xp.where(frozen, (1.0 - cover) * xp.asarray(lead, dtype=xp.float64),
                        xp.asarray(water, dtype=xp.float64))

    def _ice_column_fluxes(self, state, xp, frozen):
        """``(G_up, M_top)`` per m2 of ice on the frozen-class columns, 0
        elsewhere, or ``(None, None)`` when no column holds an ice column
        (the reference suite steps none)."""
        arrays = state.physics_state.arrays
        if frozen is None or HELD_NET_FLUX in arrays or "ice_hfx" not in arrays:
            return None, None
        from .sea_ice_thermo import basal_conduction, top_melt_surplus

        def plane(name):
            return xp.asarray(_resident(xp, arrays[name]), dtype=xp.float64)

        layers = xp.asarray(_resident(xp, state.surface.soil_temperature_k), dtype=xp.float64)
        thick = xp.asarray(_resident(xp, state.surface.sea_ice_thickness_m), dtype=xp.float64)
        snow = plane("noah_snowh") if "noah_snowh" in arrays else xp.zeros_like(thick)
        basal = basal_conduction(layers_k=layers, thickness_m=thick, snow_depth_m=snow, xp=xp)
        melt = top_melt_surplus(layers_k=layers, thickness_m=thick, snow_depth_m=snow,
                                swdown=plane("swdown"), glw=plane("glw"),
                                ice_hfx=plane("ice_hfx"), ice_qfx=plane("ice_qfx"), xp=xp)
        return (xp.where(frozen, xp.asarray(basal, dtype=xp.float64), 0.0),
                xp.where(frozen, xp.asarray(melt, dtype=xp.float64), 0.0))

    def _frozen(self, state, xp):
        """The frozen-class ocean columns (the surface's ice at least one
        half), or None when there are none."""
        from .statics import SEA_ICE_THRESHOLD

        ice = xp.asarray(_resident(xp, state.surface.sea_ice_fraction), dtype=xp.float32)
        frozen = self._on(xp, "ocean", self.ocean) & (ice >= np.float32(SEA_ICE_THRESHOLD))
        return frozen if bool(xp.any(frozen)) else None

    def advance(self, state, *, model=None) -> None:
        """Step 1 of the module docstring, on ``state``'s physics arrays
        (whole planes replaced, a parked plane through its tier)."""
        self._state = state
        meta = state.physics_state.metadata.get(METADATA_KEY)
        if meta is None:
            raise ValueError("the ocean coupler found no ocean_slab record in the physics "
                             "state; the run was not started with this ocean")
        t = float(state.time_s)
        t0 = float(meta["last_update_s"])
        dt = t - t0
        if dt <= 0.0:
            return
        arrays = state.physics_state.arrays
        xp = _module_of(state.surface.temperature_k)
        when = self.when(t0)
        ocean = self._on(xp, "ocean", self.ocean)
        frozen = self._frozen(state, xp)
        f_open = xp.where(ocean, self._open_water_flux(state, xp, frozen), 0.0)
        basal, melt = self._ice_column_fluxes(state, xp, frozen)
        deposited = False
        if self.calibrate:
            flux = f_open
            if basal is not None:
                cover = xp.clip(xp.asarray(_resident(xp, state.surface.sea_ice_fraction),
                                           dtype=xp.float64), 0.0, 1.0)
                flux = flux + xp.where(ocean, cover * (melt - basal), 0.0)
            if t0 >= self.spinup_s - 1.0e-9:
                from .ocean_qflux import deposit

                total = xp.array(_resident(xp, arrays[QCAL_SUM]), dtype=xp.float64, copy=True)
                weight = xp.array(_resident(xp, arrays[QCAL_WEIGHT]), dtype=xp.float64,
                                  copy=True)
                deposit(total, weight, flux_w_m2=flux, columns=ocean, when=when, dt_s=dt,
                        xp=xp)
                _put_array(state.physics_state, QCAL_SUM, total)
                _put_array(state.physics_state, QCAL_WEIGHT, weight)
                deposited = True
                self.stats["deposits_this_segment"] += 1
        else:
            self._advance_slab(state, xp, t0, dt, when, ocean, f_open, basal, melt,
                               until=self.when(t))
        state.physics_state.metadata[METADATA_KEY] = {
            **meta, "last_update_s": t, "advances": int(meta.get("advances", 0)) + 1,
            **({"deposits": int(meta.get("deposits", 0)) + 1} if deposited else {})}
        self.stats["advances_this_segment"] += 1

    def _advance_slab(self, state, xp, t0, dt, when, ocean, f_open, basal, melt, *,
                      until) -> None:
        from .ocean_slab import RHO_CP_SEAWATER_J_M3_K, restoring_flux, slab_step
        from .sea_ice_thermo import ice_step

        arrays = state.physics_state.arrays

        def plane(name):
            return xp.asarray(_resident(xp, arrays[name]), dtype=xp.float64)

        temperature = plane(SLAB_TEMPERATURE)
        fraction, volume = plane(ICE_FRACTION), plane(ICE_VOLUME)
        depth = xp.where(ocean, self._depth(when, xp), 0.0)
        safe_depth = xp.where(ocean, depth, 1.0)
        iced = ocean & (volume > 0.0)
        q = self._qflux(when, depth, xp, iced, fraction, until=until)
        restoring = xp.where(ocean, xp.asarray(restoring_flux(
            temperature_k=temperature, target_k=self._target(t0, when, xp), depth_m=depth,
            relaxation_days=self.params.relaxation_days, xp=xp), dtype=xp.float64), 0.0)
        ocean_flux = f_open + q + restoring
        # The mixed layer steps where it holds no ice; under ice it is at the
        # freezing point and the ocean's heat goes to the ice.
        stepped, deficit = slab_step(
            temperature_k=xp.where(ocean, temperature, 1.0e3), depth_m=safe_depth,
            net_flux_w_m2=xp.where(ocean & ~iced, ocean_flux, 0.0), dt_s=dt, xp=xp)
        stepped = xp.where(ocean & ~iced, xp.asarray(stepped, dtype=xp.float64), temperature)
        deficit = xp.where(ocean & ~iced, xp.asarray(deficit, dtype=xp.float64), 0.0)
        zero = xp.zeros_like(temperature)
        basal = zero if basal is None else xp.where(iced, basal, 0.0)
        melt = zero if melt is None else xp.where(iced, melt, 0.0)
        ice = ice_step(fraction=fraction, volume_m=volume,
                       ocean_flux_w_m2=xp.where(iced, ocean_flux, 0.0),
                       basal_conduction_w_m2=basal, top_melt_w_m2=melt,
                       freeze_deficit_j_m2=deficit, dt_s=dt,
                       new_ice_thickness_m=self.new_ice_thickness_m, xp=xp)
        heat = xp.where(ocean, xp.asarray(ice.mixed_layer_heat_j_m2, dtype=xp.float64), 0.0)
        capacity = RHO_CP_SEAWATER_J_M3_K * safe_depth
        new_temperature = xp.where(ocean, stepped + heat / capacity, 0.0)
        sources = plane(ENERGY_SOURCES) + xp.where(
            ocean, dt * (ocean_flux + fraction * (melt - basal)), 0.0)
        stored = plane(ENERGY_STORED) + xp.where(
            ocean, capacity * (new_temperature - temperature)
            + xp.asarray(ice.stored_j_m2, dtype=xp.float64), 0.0)
        physics = state.physics_state
        _put_array(physics, SLAB_TEMPERATURE, new_temperature)
        _put_array(physics, SLAB_DEPTH, depth)
        _put_array(physics, SLAB_QFLUX, q)
        _put_array(physics, ICE_FRACTION,
                   xp.where(ocean, xp.asarray(ice.fraction, dtype=xp.float64), 0.0))
        _put_array(physics, ICE_VOLUME,
                   xp.where(ocean, xp.asarray(ice.volume_m, dtype=xp.float64), 0.0))
        _put_array(physics, ENERGY_SOURCES, sources)
        _put_array(physics, ENERGY_STORED, stored)

    # -- steps 2 and 3: the records and the skin -----------------------------

    def record_index(self, t: float) -> int:
        if self.calibrate:
            return self.rule.record_index(t)
        return int(float(t) // self.record_s + 1.0e-9)

    def sst_at(self, t: float, xp, dtype):
        """``(T as dtype, ocean columns)``; ``(None, None)`` when
        calibrating."""
        self._last_t = float(t)
        if self.calibrate or self._state is None:
            return None, None
        temperature = _resident(xp, self._state.physics_state.arrays[SLAB_TEMPERATURE])
        return (xp.asarray(temperature).astype(dtype, copy=False),
                self._on(xp, "ocean", self.ocean))

    def ice_record(self, k: int, xp, surface, get):
        """``(fraction, thickness, ocean columns)`` from the prognostic ice;
        ``(None, None, None)`` when calibrating, and for a record this run
        already wrote before the checkpoint it restarted from (the surface
        carries it)."""
        if self.calibrate or self._state is None:
            return None, None, None
        from .sea_ice_thermo import thickness

        meta = self._state.physics_state.metadata[METADATA_KEY]
        if meta.get("record") is not None and int(meta["record"]) == int(k):
            self.stats["records_already_written"] += 1
            return None, None, None
        arrays = self._state.physics_state.arrays
        fraction = xp.asarray(_resident(xp, arrays[ICE_FRACTION]), dtype=xp.float64)
        volume = xp.asarray(_resident(xp, arrays[ICE_VOLUME]), dtype=xp.float64)
        depth = xp.asarray(thickness(fraction, volume, xp), dtype=xp.float64)
        dtype = get(xp, surface, "sea_ice_fraction").dtype
        self._state.physics_state.metadata[METADATA_KEY] = {**meta, "record": int(k)}
        self.stats["records_written"] += 1
        return (fraction.astype(dtype), depth.astype(dtype), self._on(xp, "ocean", self.ocean))

    def record_sst_host(self, k: int):
        if self.calibrate:
            return self.rule.record_sst_host(k)
        from .surface_series import SEA_WATER_FREEZING_K

        temperature = _host(self._state.physics_state.arrays[SLAB_TEMPERATURE])
        return np.where(self.ocean, temperature.astype(np.float64),
                        np.float64(SEA_WATER_FREEZING_K))

    def record_valid(self, k: int):
        if self.calibrate:
            return self.rule.record_valid(k)
        return self.when(k * self.record_s)

    def last_info(self, t: float) -> dict:
        meta = ({} if self._state is None
                else self._state.physics_state.metadata.get(METADATA_KEY, {}))
        return {"time_s": float(t), "record": self.record_index(t),
                "advances": meta.get("advances"), "last_update_s": meta.get("last_update_s")}

    # -- identity, receipt and the rows --------------------------------------

    def identity_block(self) -> dict:
        """The restart identity (physics_metadata["surface_boundary"]):
        ``ocean_mode = "slab"`` (or the calibrating rule's mode with
        ``qflux_calibration``), ``sea_ice_mode``, the climatology's and the
        Q-flux file's SHA-256, the start, every slab and ice parameter."""
        if self.calibrate:
            block = self.rule.identity_block()
            block["qflux_calibration"] = {
                "spinup_days": self.options.qflux_calibration_spinup_days,
                "grid": [int(self.latitude_deg.size), int(self.longitude_deg.size)],
                "physics_identity": self.physics_identity,
                "ocean_columns_sha256": _sha(self.ocean.astype(int).tolist()),
            }
            return block
        parameters = self.options.receipt()
        for name in ("climatology", "anomaly_reference_series", "slab_qflux"):
            parameters.pop(name, None)
        block = {
            "ocean_mode": "slab", "sea_ice_mode": self.options.ice_mode,
            "start_utc": self.start.isoformat().replace("+00:00", "Z"),
            "climatology_sha256": self.clim.sha256,
            "qflux_sha256": self.qflux_file.sha256,
            "parameters": parameters,
        }
        if self.rule is not None:
            anomaly = self.rule.identity_block().get("start_anomaly_sha256")
            if anomaly is not None:
                block["start_anomaly_sha256"] = anomaly
        return block

    def coverage(self) -> list:
        """Each month's coverage of the ocean columns by the Q-flux file."""
        if self.qflux_file is None:
            return []
        coverage = np.asarray(self.qflux_file.coverage_s, dtype=np.float64)
        rows = []
        for month in range(coverage.shape[0]):
            covered = (coverage[month] > 0.0) & self.ocean
            rows.append({
                "month": month + 1,
                "covered_ocean_columns": int(covered.sum()),
                "ocean_columns": int(self.ocean.sum()),
                "mean_seconds_on_ocean": (float(coverage[month][self.ocean].mean())
                                          if self.ocean.any() else 0.0),
            })
        return rows

    def receipt(self) -> dict:
        if self.calibrate:
            out = self.rule.receipt()
            out["qflux_calibration"] = {
                "spinup_days": self.options.qflux_calibration_spinup_days,
                "flux": "F_open, plus A_s (M_top - G_up) on frozen-class columns, "
                        "deposited by month weights after the spin-up",
                **{key: self.stats[key] for key in ("advances_this_segment",
                                                    "deposits_this_segment")},
            }
            return out
        return {
            "provider": "slab",
            **self.options.receipt(),
            "climatology_sha256": self.clim.sha256,
            "qflux_provenance": dict(getattr(self.qflux_file, "provenance", {}) or {}),
            "qflux_coverage": self.coverage(),
            "months_reached": [m + 1 for m in getattr(self, "months", [])],
            "physics_flux_source": ("the reference suite's held net surface flux"
                                    if self.reference_suite else
                                    "the native suite's held radiation and surface-layer "
                                    "planes (lead tile on frozen-class columns)"),
            "sst": "the mixed-layer temperature on open ocean, every step",
            "sea_ice": "prognostic fraction and thickness at each record",
            "lakes": "not written: the skin follows its own budget, the ice is held",
            "start": dict(self.stats),
        }

    def diagnostics(self, state) -> dict:
        """The output-step rows: open-ocean mean SST and its departure from
        the climatology, ice extent and area, plus
        ``ocean_mean_qflux_w_m2``, ``ocean_mean_mixed_layer_depth_m``,
        ``ocean_energy_residual_j_m2`` (the largest |stored - sources|),
        ``ocean_ice_volume_nh_km3`` and ``ocean_ice_volume_sh_km3``."""
        from .ocean_rules import TROPICS_DEG, EXTRATROPICS_DEG, _area_weights, area_mean

        if self.calibrate:
            rows = self.rule.diagnostics(state)
            weight = _host(state.physics_state.arrays[QCAL_WEIGHT])
            total = _host(state.physics_state.arrays[QCAL_SUM])
            seconds = weight.sum(axis=0)
            mask = self.ocean & (seconds > 0.0)
            rows["ocean_qflux_calibration_deposited_s"] = (
                float(seconds[self.ocean].max()) if self.ocean.any() else 0.0)
            rows["ocean_qflux_calibration_mean_flux_w_m2"] = area_mean(
                self.grid, np.where(mask, total.sum(axis=0) / np.where(mask, seconds, 1.0),
                                    0.0), mask)
            return rows
        grid = self.grid
        arrays = state.physics_state.arrays
        t = float(state.time_s)
        temperature = _host(arrays[SLAB_TEMPERATURE])
        fraction = _host(arrays[ICE_FRACTION])
        volume = _host(arrays[ICE_VOLUME])
        ocean = self.ocean
        open_ocean = ocean & (volume <= 0.0)
        clim = np.nan_to_num(self.clim.at("sst_k", self.when(t)).astype(np.float64))
        departure = np.where(open_ocean, temperature - clim, 0.0)
        lat = self.latitude_deg[:, None] * np.ones((1, self.longitude_deg.size))
        residual = np.abs(_host(arrays[ENERGY_STORED]) - _host(arrays[ENERGY_SOURCES]))
        rows: dict[str, object] = {
            "boundary_provider": f"slab/{self.options.ice_mode}",
            "boundary_valid_utc": self.when(t).isoformat().replace("+00:00", "Z"),
            "boundary_open_ocean_mean_sst_k": area_mean(grid, temperature, open_ocean),
            "boundary_mean_sst_anomaly_k": area_mean(grid, departure, open_ocean),
            "boundary_tropical_mean_sst_anomaly_k": area_mean(
                grid, departure, open_ocean & (np.abs(lat) <= TROPICS_DEG)),
            "boundary_extratropical_mean_sst_anomaly_k": area_mean(
                grid, departure, open_ocean & (np.abs(lat) >= EXTRATROPICS_DEG)),
            "ocean_mean_qflux_w_m2": area_mean(grid, _host(arrays[SLAB_QFLUX]), ocean),
            "ocean_mean_mixed_layer_depth_m": area_mean(grid, _host(arrays[SLAB_DEPTH]),
                                                        ocean),
            "ocean_energy_residual_j_m2": float(residual[ocean].max()) if ocean.any() else 0.0,
        }
        areas_m2 = 4.0 * math.pi * float(grid.radius_m) ** 2 * (
            0.5 * _area_weights(grid)) / int(grid.nlon)
        for hemisphere, rows_mask in (("nh", lat > 0.0), ("sh", lat < 0.0)):
            for threshold, label in ((0.5, "050"), (0.15, "015")):
                cover = ocean & rows_mask & (fraction >= threshold)
                rows[f"boundary_ice_extent_{hemisphere}_{label}_mkm2"] = float(
                    areas_m2[cover].sum() / 1.0e12)
                rows[f"boundary_ice_area_{hemisphere}_{label}_mkm2"] = float(
                    (areas_m2 * fraction)[cover].sum() / 1.0e12)
            rows[f"ocean_ice_volume_{hemisphere}_km3"] = float(
                (areas_m2 * volume)[ocean & rows_mask].sum() / 1.0e9)
        return rows


def tape_fields(arrays: dict, skin_k, ocean_columns) -> dict:
    """The slab's tape fields from a checkpoint's arrays (``physics__``
    names, or a physics namespace's own names): SST (the mixed-layer
    temperature on ocean columns, the skin elsewhere, K), HML (m, 0 off
    the ocean) and ARWEN_QFLUX (W/m2); an empty dict when the arrays carry
    no slab.  ``ocean_columns`` None takes the columns where the mixed
    layer has a temperature (it is 0 off the ocean)."""
    def find(name):
        for key in (f"physics__{name}", name):
            if key in arrays:
                return np.asarray(_host(arrays[key]), dtype=np.float64)
        return None

    temperature = find(SLAB_TEMPERATURE)
    if temperature is None:
        return {}
    ocean = (temperature > 0.0) if ocean_columns is None else np.asarray(ocean_columns, bool)
    skin = np.asarray(_host(skin_k), dtype=np.float64)
    depth = find(SLAB_DEPTH)
    flux = find(SLAB_QFLUX)
    return {
        TAPE_SST: np.where(ocean, temperature, skin),
        TAPE_MIXED_LAYER_DEPTH: np.where(ocean, depth if depth is not None else 0.0, 0.0),
        TAPE_QFLUX: np.where(ocean, flux if flux is not None else 0.0, 0.0),
    }


#: The tape fields' units and descriptions (wrfout_export puts them on the
#: tape's variables).
TAPE_FIELD_ATTRS = {
    TAPE_SST: {"units": "K", "description": "sea surface temperature: the slab ocean's "
               "mixed-layer temperature on ocean columns, the skin temperature elsewhere"},
    TAPE_MIXED_LAYER_DEPTH: {"units": "m", "description": "the slab ocean's mixed-layer "
                             "depth (0 off the ocean)"},
    TAPE_QFLUX: {"units": "W m-2", "description": "the slab ocean's heat-flux correction "
                 "Q, positive into the ocean (0 off the ocean)"},
}


__all__ = [
    "CALIBRATION_ARRAYS", "ENERGY_SOURCES", "ENERGY_STORED", "HELD_NET_FLUX",
    "ICE_FRACTION", "ICE_VOLUME", "METADATA_KEY", "QCAL_SUM", "QCAL_WEIGHT", "SLAB_ARRAYS",
    "SLAB_DEPTH", "SLAB_QFLUX", "SLAB_TEMPERATURE", "SlabProvider", "TAPE_FIELD_ATTRS",
    "TAPE_MIXED_LAYER_DEPTH", "TAPE_QFLUX", "TAPE_SST", "physics_identity", "tape_fields",
]
