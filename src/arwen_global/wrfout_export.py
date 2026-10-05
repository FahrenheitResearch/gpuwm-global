"""Render tapes: global lat-lon wrfout files from Arwen Global checkpoints.

The production renderer reads wrfout NetCDF and already draws unrotated
MAP_PROJ=6 global frames (the MPAS bridge's proven path), so weather-field
maps of this model go through a wrfout tape and ``gpuwm render`` rather
than any Python plotting.  ``export_parent`` stays the exact-sampling
artifact door; this door synthesizes on the Gaussian grid and regrids
bilinearly, which is map-pixel accurate and orders of magnitude cheaper
than direct spherical evaluation at every regular-grid point.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import numpy as np

from arwen_global.spectral.sampling import regular_latlon_coordinates
from arwen_global.spectral.vector import VorticityDivergenceOperator
from gpuwm.io.wrfout import WrfoutWriter, wrfout_filename

from .checkpoint import read_checkpoint, state_from_checkpoint
from .constants import (
    CONVECTIVE_RAIN_ACCUMULATOR,
    GRAVITY_M_S2,
    KAPPA,
    LATENT_HEAT_VAPORIZATION,
    REFERENCE_PRESSURE_PA,
    WATER_SPECIES,
)
from .physics.native_runtime import NATIVE_SURFACE_DIAGNOSTICS_SOURCE
from .physics.surface_diagnostics import (
    SIMILARITY_SOURCE,
    SOURCE_METADATA_KEY,
    effective_surface_humidity,
    similarity_surface_diagnostics,
)
from .pins import pins_hash
from .receipt import write_receipt
from .runner import (
    adopt_checkpoint_clock, build_model_and_cold_state, build_transform,
)

_THETA_OFFSET_K = 300.0
_EARTH_RADIUS_RENDER_M = 6_370_000.0
_SOIL_LAYERS = 4
#: Sidecar receipt beside the tapes: which screen-level source each carries.
EXPORT_RECEIPT_NAME = "arwen-global-export-receipt.json"
#: wrfout global attribute stamped with the same label.
SURFACE_DIAGNOSTICS_ATTR = "ARWEN_SURFACE_DIAGNOSTICS"
#: The model the engine's renderer names in each map's metadata row.  A
#: tape imports under the renderer's generic wrfout identity, so without
#: this attribute every global map said "WRF" (the engine reads it from
#: 2.8.1; an older engine ignores it and draws as before).
MODEL_LABEL_ATTR = "GPUWM_MODEL_LABEL"
#: The public name: maps and every sendable call the model WOOF Global, and
#: the standalone package stamped the internal name on every tape, so its
#: published maps would have named the model differently from the WOOF
#: assembly's (which rewrote the literal).
MODEL_LABEL = "WOOF Global"
EXPORT_FALLBACK_SOURCE = "export-similarity-fallback"
SURFACE_DIAGNOSTICS_SOURCES = {
    NATIVE_SURFACE_DIAGNOSTICS_SOURCE: (
        "U10/V10 from the native surface layer (sfclay u10/v10) on every "
        "column; T2/Q2 from sfclay on water columns and from WRF's SFCDIAGS "
        "after Noah on land columns (T2 = TSK - HFX/(rho cp CQS2), Q2 = QSFC "
        "- QFX/(rho CQS2)), persisted in the checkpoint's physics state"
    ),
    SIMILARITY_SOURCE: (
        "T2/Q2/U10/V10 from the reference suite's Monin-Obukhov similarity "
        "diagnostic, persisted in the checkpoint's physics state"
    ),
    EXPORT_FALLBACK_SOURCE: (
        "checkpoint carried no screen-level state; T2/Q2/U10/V10 diagnosed at "
        "export by Monin-Obukhov similarity from the skin and the lowest full "
        "level (never the lowest level itself)"
    ),
}



#: wrfout global attribute naming the water fields' convention (GI-5).
WATER_CONVENTION_ATTR = "ARWEN_WATER_CONVENTION"
#: WRF's QVAPOR, QCLOUD, QRAIN, QICE, QSNOW, QGRAUP and Q2 are DRY MIXING
#: RATIOS, and every WRF-shaped reader of the tape (the renderer's dewpoint
#: and relative humidity, the ABI reference columns) reads them so.  The
#: model carries specific humidity, so the tape converts with the model's
#: own boundary conversion (physics/native_batch).  Writing q under WRF's
#: names read 2 percent dry at q = 0.02, a 0.33 K dewpoint bias (GI-5,
#: audit 2026-10-05).
TAPE_WATER_CONVENTION = (
    "dry mixing ratio (kg per kg of dry air): QVAPOR..QGRAUP = q_x / (1 - q_v) "
    "and Q2 = q2 / (1 - q2) from the model's specific humidities"
)


def tape_water_fields(species, q2):
    """The tape's water species and Q2 as WRF's dry mixing ratios.

    ``species`` maps every water species to the model's specific
    humidity (kg per kg of moist air) and must carry ``"qv"``; ``q2`` is
    the 2 m specific humidity.  Returns ``(species, q2)`` converted.
    """
    from .physics.native_batch import mixing_ratio_from_specific_humidity

    q2 = np.asarray(q2, dtype=np.float64)
    return mixing_ratio_from_specific_humidity(species), q2 / (1.0 - q2)


def _require_tape_writer() -> None:
    """The tape writer, before the first frame is assembled.

    `WrfoutWriter` drives the netcdf-writer cdylib, and on a machine without
    it the engine refuses in its own words with a `cargo build` for a
    checkout no user of a wheel has, at exit 1. A door that is not there is
    exit 3 and the bundle that publishes it, and the check belongs before the
    regridding rather than after several minutes of it.

    An operator who has selected the engine's netCDF4 writer is left alone:
    that route is the engine's own stated workaround, and refusing it here
    would break a road somebody deliberately took.
    """

    import os

    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import WRFOUT_WRITER_ENV

    from .doors import missing_door_refusal

    if os.environ.get(WRFOUT_WRITER_ENV, "rust") != "rust":
        return
    try:
        reason = nc_writer_bridge.unavailable_reason()
    except Exception:  # noqa: BLE001 - an engine whose bridge names it otherwise
        return
    if reason is not None:
        raise missing_door_refusal("netcdf_writer", str(reason))
def _gaussian_to_regular(grid, target_lat, target_lon):
    """Bilinear sampling from the Gaussian grid at regular lat-lon centres.

    ``target_lat`` names the tape's rows and ``target_lon`` its columns in
    the tape's own order.  The sampling runs in the Rust render kernels
    (`arwen_global.render_kernels`); the NumPy version it replaced is the
    test oracle in ``tests/render_kernel_oracle.py``.  The returned callable
    takes the field and an optional ``post`` of ``"floor_zero"`` (the water
    species' clip at zero) or ``"exp"`` (surface pressure from its log).
    """
    from .render_kernels import GaussianRegridder

    return GaussianRegridder(
        grid.latitude_deg, grid.longitude_deg, target_lat, target_lon)


def _stagger_x_periodic(field: np.ndarray) -> np.ndarray:
    nx = field.shape[-1]
    out = np.empty((*field.shape[:-1], nx + 1), dtype=field.dtype)
    out[..., 1:nx] = 0.5 * (field[..., :-1] + field[..., 1:])
    out[..., 0] = 0.5 * (field[..., -1] + field[..., 0])
    out[..., nx] = out[..., 0]
    return out


def _stagger_x_clamped(field: np.ndarray) -> np.ndarray:
    """Edge-clamped x stagger for a longitude subset (no wrap column)."""
    nx = field.shape[-1]
    out = np.empty((*field.shape[:-1], nx + 1), dtype=field.dtype)
    out[..., 1:nx] = 0.5 * (field[..., :-1] + field[..., 1:])
    out[..., 0] = field[..., 0]
    out[..., nx] = field[..., -1]
    return out


def _stagger_y(field: np.ndarray) -> np.ndarray:
    ny = field.shape[-2]
    out = np.empty((*field.shape[:-2], ny + 1, field.shape[-1]), dtype=field.dtype)
    out[..., 1:ny, :] = 0.5 * (field[..., :-1, :] + field[..., 1:, :])
    out[..., 0, :] = field[..., 0, :]
    out[..., ny, :] = field[..., -1, :]
    return out


#: Surface energy and land-state planes of the native suite, in WRF's
#: history names, exported when the checkpoint's physics state carries
#: them (a state without them, the reference suite or a cold start, writes
#: none; the receipt lists which tapes carry them).  LH is Noah's energy
#: flux on land and XLV * QFX on water, where the surface layer's is the
#: only one; GRDFLX keeps WRF's sign (into the surface from the soil).
SURFACE_ENERGY_FIELDS = (
    ("HFX", "hfx"), ("QFX", "qfx"), ("GRDFLX", "noah_grdflx"),
    ("SWDOWN", "swdown"), ("GLW", "glw"), ("OLR", "olr"),
    ("UST", "ust"), ("ZNT", "znt"), ("PBLH", "pblh"),
)
SURFACE_STATE_FIELDS = (
    ("ALBEDO", "albedo"), ("EMISS", "emissivity"), ("VEGFRA", "vegetation_fraction"),
    ("LAI", "leaf_area_index"),
)


def _surface_energy_planes(bundle, regrid) -> dict[str, np.ndarray]:
    arrays = bundle.physics_state.arrays
    if not all(name in arrays for _tape, name in SURFACE_ENERGY_FIELDS):
        return {}
    surface = bundle.surface
    planes = {tape: regrid(arrays[name]).astype(np.float32) for tape, name in SURFACE_ENERGY_FIELDS}
    for tape, name in SURFACE_STATE_FIELDS:
        value = getattr(surface, name, None)
        if value is not None:
            planes[tape] = regrid(value).astype(np.float32)
    if "noah_lh" in arrays:
        land = regrid(surface.land_fraction) >= 0.5
        lh_land = regrid(arrays["noah_lh"])
        lh_water = LATENT_HEAT_VAPORIZATION * regrid(arrays["qfx"])
        planes["LH"] = np.where(land, lh_land, lh_water).astype(np.float32)
    if "VEGFRA" in planes:
        planes["VEGFRA"] = (100.0 * planes["VEGFRA"]).astype(np.float32)   # percent, as WRF
    return planes


def parse_start_date(start_date: str | None) -> datetime | None:
    """``--start-date`` (``YYYY-MM-DD_HH:MM:SS``, UTC) as an aware instant,
    or None when it was not given."""
    if start_date is None or not str(start_date).strip():
        return None
    try:
        start = datetime.strptime(str(start_date).strip(), "%Y-%m-%d_%H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            "start-date must be YYYY-MM-DD_HH:MM:SS (the analysis valid time)"
        ) from exc
    return start.replace(tzinfo=timezone.utc)


def tape_start(clock, given: datetime | None) -> datetime:
    """The instant every tape's model time is offset from (naive UTC, the
    tape convention).

    A dated run (an analysis start, or a config that states
    ``start_time_utc``) IS its own start: ``given`` may be left out, and a
    ``given`` that disagrees is refused by name, because every tape's
    Times, START_DATE and file name, and every map drawn from it, would be
    stamped from an instant the forecast did not start at.  An undated run
    (the idealized fixture) has no instant to take, so ``given`` is then
    required and labels the tapes."""
    from .clock import ClockMismatchError, CONFIG_START_TIME, mismatch_sentence

    if clock is not None and clock.dated:
        if given is not None and given != clock.start_utc:
            name = (
                "the config's physics start_time_utc"
                if clock.source == CONFIG_START_TIME
                else "the analysis valid time"
            )
            raise ClockMismatchError(
                mismatch_sentence("--start-date", given, clock.start_utc, name)
                + "; leave --start-date out to stamp the tapes from the run's "
                "own clock"
            )
        return clock.start_utc.replace(tzinfo=None)
    if given is None:
        source = "unstated" if clock is None else clock.source
        raise ValueError(
            f"this run's clock is {source}: it starts from no analysis and "
            "states no physics start_time_utc, so its checkpoints carry no "
            "instant a tape's valid time could be taken from.  Pass "
            "--start-date to label the tapes"
        )
    return given.replace(tzinfo=None)


def export_wrfout(
    cfg,
    checkpoints,
    outdir: str | Path,
    *,
    nlat: int,
    nlon: int,
    start_date: str | None = None,
    overwrite: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    extra_planes=None,
) -> list[Path]:
    """Write one wrfout tape per checkpoint on a regular ``nlat`` x ``nlon``
    grid (or its ``bbox`` window).  ``extra_planes(checkpoint, bundle,
    regrid)`` may return further 2-D planes on the Gaussian grid or already
    regridded (a plane of the tape's shape is taken as it is) to store under
    their own names: the renderer draws any stored 2-D variable as a
    ``var:<name>`` product, which is how the surface-energy instrument's
    bias maps reach it.

    The tapes' valid times are the run's forecast clock plus each
    checkpoint's model time (:func:`tape_start`): ``start_date`` may be
    left out of a dated run and is refused when it disagrees."""
    given = parse_start_date(start_date)
    # The render kernels before anything else: a machine without them
    # refuses here, with the missing-door exit and the command that stages
    # them, rather than after the model is built.
    from . import render_kernels

    render_kernels.library()
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=Path(outdir))
    clock = model.forecast_clock
    start = tape_start(clock, given)
    backend = transform.backend
    vector = VorticityDivergenceOperator(transform)

    lat, lon = regular_latlon_coordinates(nlat, nlon, include_poles=False)
    # The renderer's proven global frames run -180..180-dlon like the MPAS
    # bridge; roll the ring so the wrap column is not duplicated.
    lon_signed = np.where(lon >= 180.0, lon - 360.0, lon)
    order = np.argsort(lon_signed)
    lon_out = lon_signed[order]
    # Each tape column is sampled at its own unsigned longitude, in the
    # tape's rolled order, and a bbox window samples only its own rows and
    # columns: every value depends on its own row and column alone, so this
    # is the whole-ring sample rolled and cut, without the whole ring.
    lon_sample = lon[order]

    lat_sel = lon_sel = None
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in bbox)
        if not (lat_min < lat_max and lon_min < lon_max):
            raise ValueError(
                "bbox must be (lat_min, lat_max, lon_min, lon_max) with "
                "min < max on both axes"
            )
        lat_sel = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_sel = np.where((lon_out >= lon_min) & (lon_out <= lon_max))[0]
        if lat_sel.size < 2 or lon_sel.size < 2:
            raise ValueError(
                f"bbox {bbox} selects {lat_sel.size}x{lon_sel.size} points; "
                "a render tape needs at least 2x2"
            )
        lat = lat[lat_sel]
        lon_out = lon_out[lon_sel]
        lon_sample = lon_sample[lon_sel]
    sampler = _gaussian_to_regular(transform.grid, lat, lon_sample)

    def regrid(values, post: str = "none") -> np.ndarray:
        return sampler(backend.to_numpy(values), post)

    xlong, xlat = np.meshgrid(lon_out, lat)
    ny_out, nx_out = xlat.shape
    stagger_x = _stagger_x_periodic if bbox is None else _stagger_x_clamped
    dx_m = (360.0 / nlon) * math.pi / 180.0 * _EARTH_RADIUS_RENDER_M
    a_half = np.asarray(cfg.a_half_pa, dtype=np.float64)
    b_half = np.asarray(cfg.b_half, dtype=np.float64)
    nz = a_half.size - 1
    phi_surface = regrid(model.surface_geopotential)
    terrain_m = phi_surface / GRAVITY_M_S2

    global_attrs = {
        "START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
        "SIMULATION_START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
        "MAP_PROJ": np.int32(6),
        "MAP_PROJ_CHAR": "Cylindrical Equidistant",
        "POLE_LAT": np.float32(90.0),
        "POLE_LON": np.float32(0.0),
        "TRUELAT1": np.float32(0.0),
        "TRUELAT2": np.float32(0.0),
        "STAND_LON": np.float32(0.0),
        "CEN_LON": np.float32(float(lon_out.mean())),
        "CEN_LAT": np.float32(float(lat.mean()) if bbox is not None else 0.0),
        "GRID_ID": np.int32(1),
        "PARENT_ID": np.int32(0),
        "DT": np.float32(cfg.dt_s),
        MODEL_LABEL_ATTR: MODEL_LABEL,
    }

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tapes: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        metadata, arrays = read_checkpoint(
            checkpoint,
            expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        bundle = state_from_checkpoint(metadata, arrays, backend)
        # A checkpoint integrated from another start than this config's
        # clock is refused by name rather than stamped with the wrong
        # valid times.
        adopt_checkpoint_clock(model, bundle, f"checkpoint {checkpoint}")
        model.enforce(bundle)
        atmosphere = bundle.atmosphere
        seconds = float(metadata["time_s"])
        if abs(seconds - round(seconds)) > 1.0e-6:
            raise ValueError(
                f"checkpoint time {seconds} s is not a whole second"
            )
        valid = start + timedelta(seconds=round(seconds))

        theta = regrid(transform.inverse(atmosphere.theta))
        ps = regrid(transform.inverse(atmosphere.log_surface_pressure), "exp")
        u_grid, v_grid = vector.wind_from_vordiv(
            atmosphere.vorticity, atmosphere.divergence
        )
        u = regrid(u_grid)
        v = regrid(v_grid)
        # Vapor is synthesized from its coefficients; the condensate
        # species are grid fields already.
        species = {
            name: regrid(
                transform.inverse(atmosphere.qv) if name == "qv"
                else getattr(atmosphere, name),
                "floor_zero",
            )
            for name in WATER_SPECIES
        }

        # The tape's column: hybrid pressure, temperature and virtual
        # temperature, and the hydrostatic geopotential integrated up from
        # the terrain, all in the Rust render kernels.
        p_half, p_full = render_kernels.hybrid_pressure(a_half, b_half, ps)
        temperature, virtual = render_kernels.virtual_temperature(
            theta, species, p_full,
            reference_pressure_pa=REFERENCE_PRESSURE_PA, kappa=KAPPA)
        phi_half = render_kernels.hydrostatic(
            virtual, p_half, phi_surface, gas_constant=model.gas_constant)

        def up(field: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(field[::-1]).astype(np.float32)

        surface = bundle.surface
        screen, screen_source = _screen_level_fields(
            cfg, bundle, backend, regrid, temperature, species, u, v, p_half, p_full
        )
        # WRF's moisture names carry WRF's convention (GI-5); the virtual
        # temperature and the screen fallback above use the model's q.
        tape_species, tape_q2 = tape_water_fields(species, screen["q2"])
        rain = regrid(surface.accumulated_rain_kg_m2)
        snow = regrid(surface.accumulated_snow_kg_m2)
        graupel = regrid(surface.accumulated_graupel_kg_m2)
        convective = _convective_rain(bundle, regrid, (ny_out, nx_out))
        land = regrid(surface.land_fraction)
        energy = _surface_energy_planes(bundle, regrid)
        extras = {}
        if extra_planes is not None:
            for name, value in (extra_planes(checkpoint, bundle, regrid) or {}).items():
                plane = np.asarray(value, dtype=np.float64)
                if plane.shape != (ny_out, nx_out):
                    plane = regrid(plane)
                extras[str(name)] = plane.astype(np.float32)
        frame = {
            "XLAT": xlat.astype(np.float32),
            "XLONG": xlong.astype(np.float32),
            "T": up(theta - _THETA_OFFSET_K),
            "P": np.zeros((nz, ny_out, nx_out), dtype=np.float32),
            "PB": up(p_full),
            "PH": np.zeros((nz + 1, ny_out, nx_out), dtype=np.float32),
            "PHB": up(phi_half),
            "QVAPOR": up(tape_species["qv"]),
            "QCLOUD": up(tape_species["qc"]),
            "QRAIN": up(tape_species["qr"]),
            "QICE": up(tape_species["qi"]),
            "QSNOW": up(tape_species["qs"]),
            "QGRAUP": up(tape_species["qg"]),
            "U": stagger_x(up(u)),
            "V": _stagger_y(up(v)),
            # Screen-level diagnostics: the physics suite's own 2 m / 10 m
            # fields when the checkpoint carries them, else the similarity
            # diagnostic from the skin and the lowest level; never the
            # lowest level itself (audit 2026-09-01, task 1b).  The source
            # is stamped on the tape and in the export receipt.
            "T2": screen["t2"].astype(np.float32),
            "Q2": tape_q2.astype(np.float32),
            "U10": screen["u10"].astype(np.float32),
            "V10": screen["v10"].astype(np.float32),
            "PSFC": ps.astype(np.float32),
            "TSK": regrid(surface.temperature_k).astype(np.float32),
            "HGT": terrain_m.astype(np.float32),
            "LANDMASK": (land >= 0.5).astype(np.float32),
            "SINALPHA": np.zeros((ny_out, nx_out), dtype=np.float32),
            "COSALPHA": np.ones((ny_out, nx_out), dtype=np.float32),
            # WRF's split: RAINC is the cumulus scheme's accumulator, RAINNC
            # the microphysics accumulators.
            "RAINC": convective.astype(np.float32),
            "RAINNC": (rain + snow + graupel).astype(np.float32),
            "SNOWNC": snow.astype(np.float32),
            "GRAUPELNC": graupel.astype(np.float32),
            "TSLB": regrid(surface.soil_temperature_k)[:_SOIL_LAYERS].astype(np.float32),
            "SMOIS": regrid(surface.soil_water_fraction)[:_SOIL_LAYERS].astype(np.float32),
            **energy,
            **extras,
            # The seeded surface under WRF's names: SEAICE and ICEDEPTH from
            # the surface state, SNOW (kg/m2), SNOWH (m) and SNOWC from
            # Noah's store when the checkpoint carries it (zero planes on a
            # state without a native physics namespace).
            "SEAICE": regrid(surface.sea_ice_fraction).astype(np.float32),
            "ICEDEPTH": regrid(surface.sea_ice_thickness_m).astype(np.float32),
            **_snow_planes(bundle, regrid, (ny_out, nx_out)),
        }

        _require_tape_writer()
        path = output / wrfout_filename(valid)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"render tape {path} exists; pass --overwrite to replace it"
            )
        writer = WrfoutWriter(
            path,
            nx=nx_out,
            ny=ny_out,
            nz=nz,
            dx=dx_m,
            dy=dx_m,
            title="Arwen Global research model",
            global_attrs={
                **global_attrs, SURFACE_DIAGNOSTICS_ATTR: screen_source,
                WATER_CONVENTION_ATTR: TAPE_WATER_CONVENTION,
            },
            soil_layers=_SOIL_LAYERS,
        )
        try:
            writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), frame)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        written.append(path)
        tapes.append({
            "tape": str(path),
            "checkpoint": str(checkpoint),
            "valid": valid.strftime("%Y-%m-%d_%H:%M:%S"),
            "surface_diagnostics": screen_source,
            "surface_energy_fields": sorted(energy),
            "extra_planes": sorted(extras),
        })
    write_receipt(output / EXPORT_RECEIPT_NAME, {
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "physics_mode": cfg.physics_mode,
        "vertical": {"coordinate": cfg.vertical_coordinate, **cfg.vertical.describe()},
        "screen_level_sources": SURFACE_DIAGNOSTICS_SOURCES,
        # Where every tape's valid time came from (arwen_global.clock).
        "start_date": start.strftime("%Y-%m-%d_%H:%M:%S"),
        "start_date_source": (
            clock.source if clock is not None and clock.dated
            else "start-date argument"
        ),
        "water_convention": TAPE_WATER_CONVENTION,
        "tapes": tapes,
    })
    return written


def _snow_planes(bundle, regrid, shape):
    """Noah's snow store, depth and cover flag on the tape (WRF SNOW,
    SNOWH, SNOWC), zero where the physics namespace has none."""
    arrays = bundle.physics_state.arrays
    out = {}
    for tape_name, store in (("SNOW", "noah_snow"), ("SNOWH", "noah_snowh"), ("SNOWC", "noah_snowc")):
        value = arrays.get(store)
        if value is None:
            out[tape_name] = np.zeros(shape, dtype=np.float32)
        else:
            out[tape_name] = regrid(value).astype(np.float32)
    return out


def _convective_rain(bundle, regrid, shape):
    """Accumulated convective precipitation (WRF RAINC) for one tape.

    The cumulus scheme books it in the physics state under
    CONVECTIVE_RAIN_ACCUMULATOR (native runtime _cumulus_step); a state
    without the array - physics mode "none", or a suite with no cumulus
    scheme - exports zeros, WRF's own convention for a run without one.
    """
    value = bundle.physics_state.arrays.get(CONVECTIVE_RAIN_ACCUMULATOR)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(regrid(value), dtype=np.float64)


def _screen_level_fields(cfg, bundle, backend, regrid, temperature, species, u, v, p_half, p_full):
    """T2/Q2/U10/V10 for one tape and the label naming where they came from."""
    arrays = bundle.physics_state.arrays
    if all(name in arrays for name in ("t2", "q2", "u10", "v10")):
        source = str(bundle.physics_state.metadata.get(SOURCE_METADATA_KEY, "physics-state"))
        return {
            name: regrid(arrays[name]) for name in ("t2", "q2", "u10", "v10")
        }, source
    # Fallback: the checkpoint carries no screen-level state (physics mode
    # "none", or a tape written before the suites persisted them), so the
    # similarity diagnostic runs here on the regridded skin and lowest level.
    surface = bundle.surface
    land = regrid(surface.land_fraction)
    wetness = (
        regrid(surface.soil_water_fraction)[0]
        / cfg.reference_physics.soil_wetness_capacity
    )
    skin = regrid(surface.temperature_k)
    humidity = effective_surface_humidity(
        skin, p_half[-1], land, wetness, species["qv"][-1], np
    )
    out = similarity_surface_diagnostics(
        u_lowest=u[-1], v_lowest=v[-1], temperature_lowest=temperature[-1],
        qv_lowest=species["qv"][-1], p_full_lowest_pa=p_full[-1],
        p_surface_pa=p_half[-1], skin_temperature_k=skin,
        surface_humidity=humidity, roughness_m=regrid(surface.roughness_m), xp=np,
    )
    return {name: out[name] for name in ("t2", "q2", "u10", "v10")}, EXPORT_FALLBACK_SOURCE


__all__ = [
    "EXPORT_RECEIPT_NAME", "SURFACE_ENERGY_FIELDS", "TAPE_WATER_CONVENTION",
    "WATER_CONVENTION_ATTR", "export_wrfout", "parse_start_date",
    "tape_start", "tape_water_fields",
]
