"""Check an ERA5 start against ERA5 itself: hour 0, and the tapes after it.

``python tools/arwen_global_era5_check.py read YYYY-MM-DDTHH OUT.npz``
``python tools/arwen_global_era5_check.py state CONFIG OUT.npz [--scratch DIR]``
``python tools/arwen_global_era5_check.py hour0 STATE.npz ERA5.npz [--json OUT]``
``python tools/arwen_global_era5_check.py tapes TAPE=ERA5_NC [TAPE=ERA5_NC ...] [--config CONFIG] [--json OUT]``

``read`` reads one ERA5 hour a second way: zarr-python straight from
Google's public ARCO store, the time index found from the store's own time
coordinate.  It shares nothing with the engine's ARCO retrieval or with this
package, and needs ``zarr`` with ``fsspec`` and ``aiohttp``, which this
package does not depend on (``pip install zarr fsspec aiohttp``).  It writes
the surface fields and the five pressure-level fields the comparison reads
(about 1 GB as ``.npz``).

``state`` runs the cold start of an ERA5 configuration once on the CPU, with
no time step, and writes the hour-0 state on the model's own grid and levels:
temperature, specific humidity and winds on the model levels, the full and
half level pressures, the surface geopotential, and the surface and Noah
planes.  It imports this package; it needs no device.

``hour0`` compares the two.  Nothing in it is the model's or the engine's
code: the periodic bilinear sampling of ERA5 onto the model's columns, the
interpolation in ln p onto the model's levels, the hypsometric move of
ERA5's surface pressure onto the model terrain, the hydrostatic heights, the
precipitable water and the depth overlap of ERA5's soil layers onto Noah's
are written again here.  It prints area-weighted bias and RMS of the model
state minus ERA5 by field and layer, the orientation test (terrain
correlated with ERA5 as it is, flipped north-south and shifted 180 degrees),
and the surface: open-water skin against ERA5's SST, sea ice, snow, and
soil by Noah layer on interior land.  These are the rows of the hour-0 table
in docs/ARWEN_GLOBAL_ERA5.md.

``tapes`` reads render tapes (``gpuwm-global export``) each paired with the
engine's ERA5 file of the same valid time (``gpuwm-global fetch-analysis
--source era5``), in any order; the tape of the earliest valid time is the
start.  For every tape it prints the non-finite count, the global surface
pressure, 500 hPa height, 2 m temperature and precipitable water against
ERA5, and the skin and 2 m temperature against ERA5 by the tape's own
surface classes, the classes fixed by the start tape: land ice
(``LU_INDEX`` equal to the tape's ``ISICE``), sea ice, open water, snow-free
land, snowy land that is not land ice, and of it the land that started with
more than 1000 kg m-2 of snow water.  The classes are read off the tape,
whose land-use planes are the nearest model column's and whose sea ice and
snow are interpolated, so a coastal column can read sea ice and the snow of
the glacier beside it at once: the sea-ice rows' snow is the coast's, and
the model's own pack is checked on its own grid by ``hour0``.  Z500 is
interpolated in ln p between
the tape's own layer pressures and heights; with ``--config`` the half-level
pressures come from that configuration's hybrid coefficients instead.

``read`` and ``state`` are the slow halves (about a minute each at T255);
``hour0`` and ``tapes`` need numpy, and ``tapes`` netCDF4.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np

ARCO_STORE = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
              "full_37-1h-0p25deg-chunk-1.zarr-v3")
#: ERA5's own constant for geopotential to height.
G_ERA5 = 9.80665
RD = 287.0
EPS_VIRTUAL = 0.608

READ_SURFACE = (
    "surface_pressure", "geopotential_at_surface", "land_sea_mask", "skin_temperature",
    "sea_surface_temperature", "sea_ice_cover", "snow_depth", "snow_density",
    "soil_temperature_level_1", "soil_temperature_level_2", "soil_temperature_level_3",
    "soil_temperature_level_4", "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4", "2m_temperature", "total_column_water_vapour",
)
READ_PRESSURE = ("temperature", "specific_humidity", "u_component_of_wind",
                 "v_component_of_wind", "geopotential")

#: ERA5 (HTESSEL) soil layers and Noah's, metres.
ERA5_SOIL_LAYERS_M = ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89))
NOAH_SOIL_LAYERS_M = ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
#: The MODIS land-use class of land ice, when a tape names none.
DEFAULT_ISICE = 15
#: Snow water above which land counts as snowy, and as deep, kg m-2.
SNOWY_KG_M2 = 10.0
DEEP_SNOW_KG_M2 = 1000.0


# ------------------------------------------------------------------- read

def read_era5_hour(when: dt.datetime, out_path: str, *, store: str = ARCO_STORE) -> dict:
    """One ERA5 hour straight from the ARCO store, written to ``out_path``."""

    import zarr

    root = zarr.open_group(store, mode="r")
    time = root["time"]
    units = str(time.attrs["units"])
    if not units.startswith("hours since 1900-01-01"):
        raise ValueError(f"the store's time units are {units!r}")
    hours = int((when - dt.datetime(1900, 1, 1)).total_seconds() // 3600)
    index = hours - int(time[0])
    if index < 0 or int(time[index]) != hours:
        raise ValueError(f"{when:%Y-%m-%d %HZ} is not in the store")
    out = {"time_hours_since_1900": np.asarray(hours)}
    for name in ("latitude", "longitude", "level"):
        out[name] = np.asarray(root[name][:])
    units_seen = {}
    for name in READ_SURFACE + READ_PRESSURE:
        array = root[name]
        out[name] = np.asarray(array[index])
        units_seen[name] = dict(array.attrs).get("units")
    out["units_json"] = np.asarray(json.dumps(units_seen))
    np.savez(out_path, **out)
    return {"time_index": index, "hours_since_1900": hours, "units": units_seen}


# ------------------------------------------------------------------ state

def write_cold_start_state(config_path: str, out_path: str, scratch: str | None = None) -> dict:
    """The cold start of ``config_path`` on the CPU, no step, as grid arrays."""

    import dataclasses
    from pathlib import Path

    from arwen_global.analysis_initial import analysis_initial_state
    from arwen_global.config import load_config
    from arwen_global.lnps_reference import LNPS_REFERENCE
    from arwen_global.runner import build_transform
    from arwen_global.spectral.constants import KAPPA, REFERENCE_PRESSURE_PA
    from arwen_global.spectral.vector import VorticityDivergenceOperator

    cfg = dataclasses.replace(load_config(config_path), backend="numpy")
    transform = build_transform(cfg)
    scratch_dir = Path(scratch) if scratch else Path(out_path).resolve().parent
    state, phi, provenance = analysis_initial_state(
        cfg, transform, scratch_destination=scratch_dir)
    backend = transform.backend

    def grid(values):
        return np.asarray(backend.to_numpy(values), dtype=np.float64)

    def inverse(coefficients):
        return grid(transform.inverse(coefficients))

    atmosphere = state.atmosphere
    log_ps = inverse(atmosphere.log_surface_pressure)
    # The state holds ln ps either whole or about LNPS_REFERENCE, depending
    # on the build; which one is read from the value.
    ps = np.exp(log_ps)
    if ps.mean() < 2000.0:
        ps = np.exp(log_ps + LNPS_REFERENCE)
    pressure = cfg.vertical.pressure(ps, backend)
    p_full = grid(pressure["p_full"])
    p_half = grid(pressure["p_half"])
    temperature = inverse(atmosphere.theta) * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    u, v = VorticityDivergenceOperator(transform).wind_from_vordiv(
        atmosphere.vorticity, atmosphere.divergence)
    out = {
        "lat": np.asarray(transform.grid.latitude_deg, dtype=np.float64),
        "lon": np.asarray(transform.grid.longitude_deg, dtype=np.float64),
        "ps": ps, "p_full": p_full, "p_half": p_half, "T": temperature,
        "q": inverse(atmosphere.qv), "u": grid(u), "v": grid(v), "phi": grid(phi),
    }
    surface = state.surface
    for name in ("temperature_k", "land_fraction", "sea_ice_fraction",
                 "sea_ice_thickness_m", "soil_temperature_k", "soil_water_fraction",
                 "landuse_category"):
        if hasattr(surface, name):
            out["sfc_" + name] = grid(getattr(surface, name))
    for name, values in state.physics_state.arrays.items():
        out["phys_" + name] = grid(values)
    out["valid_time"] = np.asarray(str(provenance.get("valid_time")))
    out["input_sha256"] = np.asarray(str(provenance.get("input_sha256")))
    np.savez(out_path, **out)
    return {"valid_time": provenance.get("valid_time"),
            "input_sha256": provenance.get("input_sha256"),
            "era5_completion": provenance.get("era5_completion")}


# ------------------------------------------------------------------ hour0

class Sampler:
    """Periodic bilinear sampling of a regular 0.25-degree ERA5 plane at
    model columns.  ERA5's rows run north to south from 90; its columns
    start at ``lon0`` and close the ring."""

    def __init__(self, era_lat, era_lon, lat, lon):
        era_lat = np.asarray(era_lat, dtype=np.float64)
        era_lon = np.asarray(era_lon, dtype=np.float64)
        step_y = float(era_lat[0] - era_lat[1])
        step_x = float(era_lon[1] - era_lon[0])
        if step_y <= 0.0 or step_x <= 0.0:
            raise ValueError("ERA5 rows must run north to south and columns east")
        if abs(era_lon.size * step_x - 360.0) > 1e-6:
            raise ValueError("ERA5 longitudes must close the ring")
        yi = (era_lat[0] - np.asarray(lat, dtype=np.float64)) / step_y
        self.y0 = np.clip(np.floor(yi).astype(int), 0, era_lat.size - 2)
        self.fy = (yi - self.y0)[:, None]
        xi = ((np.asarray(lon, dtype=np.float64) - era_lon[0]) % 360.0) / step_x
        self.x0 = np.floor(xi).astype(int) % era_lon.size
        self.x1 = (self.x0 + 1) % era_lon.size
        self.fx = (xi - np.floor(xi))[None, :]

    def __call__(self, field):
        f = np.asarray(field, dtype=np.float64)
        a = f[..., self.y0, :][..., self.x0]
        b = f[..., self.y0, :][..., self.x1]
        c = f[..., self.y0 + 1, :][..., self.x0]
        d = f[..., self.y0 + 1, :][..., self.x1]
        fy, fx = self.fy, self.fx
        return (1 - fy) * (1 - fx) * a + (1 - fy) * fx * b + fy * (1 - fx) * c + fy * fx * d

    def masked(self, field, mask):
        """The weighted mean over the stencil's masked points; NaN where it
        has none.  Returns ``(values, weight)``."""
        weight = self(np.asarray(mask, dtype=np.float64))
        total = self(np.where(mask, field, 0.0))
        safe = np.where(weight > 1e-9, weight, 1.0)
        return np.where(weight > 1e-9, total / safe, np.nan), weight


def area_stats(model, reference, weights, where=None) -> dict | None:
    """Area-weighted bias, RMS, max |difference| and correlation."""

    model = np.asarray(model, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = model - reference
    ok = np.isfinite(difference)
    if where is not None:
        ok &= np.broadcast_to(where, difference.shape)
    if not ok.any():
        return None
    w = np.broadcast_to(weights, difference.shape)[ok]
    d = difference[ok]
    a, b = model[ok], reference[ok]
    corr = (float(np.corrcoef(a, b)[0, 1])
            if a.size > 1 and a.std() > 0 and b.std() > 0 else None)
    return {
        "n": int(ok.sum()),
        "model_mean": float((a * w).sum() / w.sum()),
        "era5_mean": float((b * w).sum() / w.sum()),
        "bias": float((d * w).sum() / w.sum()),
        "rms": float(np.sqrt((d * d * w).sum() / w.sum())),
        "max_abs": float(np.abs(d).max()),
        "corr": corr,
    }


def soil_overlap_weights(source=ERA5_SOIL_LAYERS_M, target=NOAH_SOIL_LAYERS_M) -> np.ndarray:
    """Row k: the share of target layer k each source layer covers."""

    return np.array([[max(0.0, min(t1, s1) - max(t0, s0)) / (t1 - t0)
                      for (s0, s1) in source] for (t0, t1) in target])


def height_at_pressure(target_pa, p_half, z_half, virtual_temperature):
    """Hydrostatic height of ``target_pa`` inside the layer that holds it,
    from the half-level pressures and heights and the layer's virtual
    temperature; NaN where no layer holds it."""

    out = np.full(p_half.shape[1:], np.nan)
    for k in range(p_half.shape[0] - 1):
        inside = (p_half[k] <= target_pa) & (p_half[k + 1] > target_pa)
        z = z_half[k + 1] + RD * virtual_temperature[k] * np.log(p_half[k + 1] / target_pa) / G_ERA5
        out = np.where(inside, z, out)
    return out


def compare_hour0(model: dict, era5: dict) -> dict:
    """Hour 0 of the model state against ERA5; returns sections of rows."""

    elat = np.asarray(era5["latitude"], dtype=np.float64)
    elon = np.asarray(era5["longitude"], dtype=np.float64)
    elev = np.asarray(era5["level"], dtype=np.float64) * 100.0
    if elev[0] > elev[-1]:
        raise ValueError("ERA5 levels must run top first, as the store holds them")
    lat, lon = np.asarray(model["lat"]), np.asarray(model["lon"])
    sample = Sampler(elat, elon, lat, lon)
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    rows: dict[str, dict] = {}

    def row(section, label, a, b, where=None):
        rows.setdefault(section, {})[label] = area_stats(a, b, weights, where)

    ps, p_full, p_half = model["ps"], model["p_full"], model["p_half"]
    if not p_full[0].mean() < p_full[-1].mean():
        raise ValueError("the state's levels must run top first")
    zs_model = model["phi"] / G_ERA5
    zs_era5 = sample(era5["geopotential_at_surface"]) / G_ERA5
    ps_era5 = sample(era5["surface_pressure"])

    # Orientation: the terrain against ERA5 as it is and moved.
    base = sample(np.asarray(era5["geopotential_at_surface"], dtype=np.float64) / G_ERA5)
    corr = lambda x: float(np.corrcoef(zs_model.ravel(), x.ravel())[0, 1])  # noqa: E731
    orientation = {
        "as_is": corr(base), "flipped_north_south": corr(base[::-1]),
        "shifted_180": corr(np.roll(base, lon.size // 2, axis=1)),
        "shifted_one_column_east": corr(np.roll(base, 1, axis=1)),
    }

    row("terrain and surface pressure", "terrain height, m", zs_model, zs_era5)
    virtual_low = model["T"][-1] * (1 + EPS_VIRTUAL * np.clip(model["q"][-1], 0, None))
    ps_moved = ps_era5 * np.exp(G_ERA5 * (zs_era5 - zs_model) / (RD * virtual_low))
    row("terrain and surface pressure", "surface pressure, ERA5's as it is, hPa", ps / 100, ps_era5 / 100)
    row("terrain and surface pressure", "surface pressure, ERA5's moved to the model terrain, hPa",
        ps / 100, ps_moved / 100)

    ln_levels = np.log(elev)
    ln_p = np.log(p_full)
    k = np.clip((ln_p[..., None] >= ln_levels).sum(-1) - 1, 0, elev.size - 2)
    frac = (ln_p - ln_levels[k]) / (ln_levels[k + 1] - ln_levels[k])
    inside = (p_full >= elev[0]) & (p_full <= np.minimum(elev[-1], ps_era5[None]))
    bands = ((100, 1000, "1-10 hPa"), (1000, 10000, "10-100 hPa"),
             (10000, 30000, "100-300 hPa"), (30000, 70000, "300-700 hPa"),
             (70000, 110000, "700 hPa to the ground"))
    for name, era_name, unit, scale in (
            ("T", "temperature", "K", 1.0), ("q", "specific_humidity", "g/kg", 1000.0),
            ("u", "u_component_of_wind", "m/s", 1.0), ("v", "v_component_of_wind", "m/s", 1.0)):
        source = sample(era5[era_name])
        lower = np.take_along_axis(source, k, axis=0)
        upper = np.take_along_axis(source, k + 1, axis=0)
        reference = lower + frac * (upper - lower)
        for low, high, label in bands:
            row("atmosphere on model levels", f"{name} {label}, {unit}",
                model[name] * scale, reference * scale,
                inside & (p_full >= low) & (p_full < high))

    dp = p_half[1:] - p_half[:-1]
    q_clipped = np.clip(model["q"], 0, None)
    row("columns", "precipitable water against ERA5's column total, kg m-2",
        (q_clipped * dp).sum(0) / G_ERA5, sample(era5["total_column_water_vapour"]))
    virtual = model["T"] * (1 + EPS_VIRTUAL * q_clipped)
    z_half = np.empty_like(p_half)
    z_half[-1] = zs_model
    for kk in range(p_full.shape[0] - 1, -1, -1):
        z_half[kk] = z_half[kk + 1] + RD * virtual[kk] * np.log(p_half[kk + 1] / p_half[kk]) / G_ERA5
    for target in (5000.0, 25000.0, 50000.0, 85000.0):
        index = int(np.argmin(np.abs(elev - target)))
        if elev[index] != target:
            continue
        row("columns", f"Z{int(target / 100)} from the state against ERA5 z / g, m",
            height_at_pressure(target, p_half, z_half, virtual),
            sample(era5["geopotential"][index]) / G_ERA5)

    lsm = np.asarray(era5["land_sea_mask"], dtype=np.float64)
    ice_era5 = np.asarray(era5["sea_ice_cover"], dtype=np.float64)
    sst = np.asarray(era5["sea_surface_temperature"], dtype=np.float64)
    land = model["sfc_land_fraction"]
    ice = model["sfc_sea_ice_fraction"]
    skin = model["sfc_temperature_k"]
    frozen = ice.astype(np.float32) >= np.float32(0.5)
    open_water = (land <= 0.5) & ~frozen
    sst_ref, sst_weight = sample.masked(sst, np.isfinite(sst) & (np.nan_to_num(ice_era5) == 0.0))
    row("surface", "open-water skin against ERA5's SST (stencil all ice-free sea), K",
        skin, sst_ref, open_water & (sst_weight > 0.999))
    row("surface", "open-water skin against ERA5's SST (any ice-free sea in the stencil), K",
        skin, sst_ref, open_water & (sst_weight > 0))
    offset = np.where(open_water & (sst_weight > 0), skin - sst_ref, np.nan)
    sst_counts = {f"more_than_{n}_k": int(np.count_nonzero(np.abs(offset) > n)) for n in (1, 2, 5)}
    ice_ref, ice_weight = sample.masked(ice_era5, np.isfinite(ice_era5))
    row("surface", "sea-ice fraction (stencil all sea)", ice, ice_ref, ice_weight > 0.999)
    row("surface", "land skin against ERA5's skin (land above 0.99 in both), K",
        skin, sample(era5["skin_temperature"]), (land > 0.99) & (sample(lsm) > 0.99))

    snow_water = np.asarray(era5["snow_depth"], dtype=np.float64) * 1000.0
    density = np.asarray(era5["snow_density"], dtype=np.float64)
    land_points = lsm >= 0.5
    swe_ref, land_weight = sample.masked(snow_water, land_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_era5 = np.where(snow_water > 0, snow_water / density, 0.0)
    depth_ref, _ = sample.masked(depth_era5, land_points)
    interior = (land > 0.99) & (land_weight > 0.999)
    if "phys_noah_snow" in model:
        row("surface", "snow water, interior land, kg m-2", model["phys_noah_snow"], swe_ref, interior)
    if "phys_noah_snowh" in model:
        row("surface", "snow depth, interior land, m", model["phys_noah_snowh"], depth_ref, interior)
    sea_snow = frozen & (land <= 0.5)
    snow_on_pack = None
    if "phys_noah_snow" in model and sea_snow.any():
        snow_on_pack = {"frozen_water_columns": int(sea_snow.sum()),
                        "with_snow": int(np.count_nonzero(model["phys_noah_snow"][sea_snow] > 0))}

    overlap = soil_overlap_weights()
    land_ice = (np.rint(model["sfc_landuse_category"]) == DEFAULT_ISICE
                if "sfc_landuse_category" in model else np.zeros_like(interior))
    plain = interior & ~land_ice
    if "phys_noah_snow" in model:
        plain &= model["phys_noah_snow"] < 1.0
    for era_name, label, name, unit in (
            ("soil_temperature_level_%d", "soil temperature", "sfc_soil_temperature_k", "K"),
            ("volumetric_soil_water_layer_%d", "soil water", "sfc_soil_water_fraction", "m3 m-3")):
        if name not in model:
            continue
        stack = np.stack([sample.masked(np.asarray(era5[era_name % (n + 1)], dtype=np.float64),
                                        land_points)[0] for n in range(4)])
        for layer in range(4):
            reference = (overlap[layer][:, None, None] * stack).sum(0)
            row("soil, interior snow-free land", f"{label}, Noah layer {layer + 1}, {unit}",
                model[name][layer], reference, plain)

    return {"orientation": orientation, "rows": rows,
            "open_water_columns_off_era5_sst": sst_counts,
            "snow_on_the_pack": snow_on_pack,
            "model_grid": [int(lat.size), int(lon.size)],
            "valid_time": str(model.get("valid_time", "")),
            "input_sha256": str(model.get("input_sha256", ""))}


def print_hour0(result: dict, stream=sys.stdout) -> None:
    print(f"hour 0, model state minus ERA5; model grid {result['model_grid']}, "
          f"valid {result['valid_time']}", file=stream)
    o = result["orientation"]
    print(f"orientation, terrain against ERA5's: as it is {o['as_is']:.4f}, "
          f"flipped north-south {o['flipped_north_south']:.4f}, shifted 180 degrees "
          f"{o['shifted_180']:.4f}, one column east {o['shifted_one_column_east']:.4f}",
          file=stream)
    for section, rows in result["rows"].items():
        print(f"\n== {section}", file=stream)
        for label, s in rows.items():
            if s is None:
                print(f"  {label:70s} no points", file=stream)
                continue
            corr = "" if s["corr"] is None else f" corr {s['corr']:.5f}"
            print(f"  {label:70s} n {s['n']:8d} bias {s['bias']:11.4g} rms {s['rms']:10.4g} "
                  f"max|d| {s['max_abs']:10.4g}{corr}", file=stream)
    counts = result["open_water_columns_off_era5_sst"]
    print(f"\nopen-water columns off ERA5's SST: {counts}", file=stream)
    if result["snow_on_the_pack"] is not None:
        print(f"snow on frozen water columns: {result['snow_on_the_pack']}", file=stream)


# ------------------------------------------------------------------ tapes

def _plane(dataset, name):
    return np.asarray(np.ma.filled(np.ma.asarray(dataset[name][0], dtype=np.float64), np.nan))


def read_era5_file(path: str) -> dict:
    """The planes ``tapes`` compares, from the engine's ERA5 file."""

    from netCDF4 import Dataset

    with Dataset(path) as data:
        level = np.asarray(data["level"][:], dtype=np.float64) * 100.0
        out = {
            "lat": np.asarray(data["latitude"][:], dtype=np.float64),
            "lon": np.asarray(data["longitude"][:], dtype=np.float64),
            "level": level,
            "PSFC": _plane(data, "PSFC"), "T2": _plane(data, "T2"),
            "SKINTEMP": _plane(data, "SKINTEMP"),
            "Z": np.asarray(data["Z"][0], dtype=np.float64),
            "SPFH": np.asarray(data["SPFH"][0], dtype=np.float64),
        }
        times = data["time"]
        from netCDF4 import num2date

        stamp = num2date(np.asarray(times[:])[0], times.units,
                         only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        out["valid"] = dt.datetime(stamp.year, stamp.month, stamp.day, stamp.hour)
    return out


def era5_precipitable_water(level_pa, q, ps):
    """q dp / g over ERA5's levels above the ground, the 1000 hPa value
    held below the deepest level."""

    order = np.argsort(level_pa)
    lev, qs = level_pa[order], q[order]
    total = np.zeros_like(ps)
    for k in range(lev.size - 1):
        bottom = np.minimum(lev[k + 1], ps)
        total += 0.5 * (qs[k] + qs[k + 1]) * np.clip(bottom - lev[k], 0.0, None) / G_ERA5
    total += qs[-1] * np.clip(ps - lev[-1], 0.0, None) / G_ERA5
    return total


def read_tape(path: str, hybrid=None) -> dict:
    from netCDF4 import Dataset

    with Dataset(path) as data:
        get = lambda name: np.asarray(data[name][0], dtype=np.float64)  # noqa: E731
        times = b"".join(data["Times"][0].tolist()).decode().strip()
        valid = dt.datetime.strptime(times[:13], "%Y-%m-%d_%H")
        ps = get("PSFC")
        pressure = get("PB") + (get("P") if "P" in data.variables else 0.0)
        phi = get("PHB") + (get("PH") if "PH" in data.variables else 0.0)
        qv = get("QVAPOR")
        nonfinite = sum(int(np.count_nonzero(~np.isfinite(np.asarray(data[name][:], dtype=np.float64))))
                        for name in data.variables if data[name].dtype.kind == "f")
        isice = int(getattr(data, "ISICE", DEFAULT_ISICE))
        tape = {
            "path": path, "valid": valid, "lat": get("XLAT"), "lon": get("XLONG"),
            "PSFC": ps, "T2": get("T2"), "TSK": get("TSK"), "SNOW": get("SNOW"),
            "SEAICE": get("SEAICE"), "XLAND": get("XLAND"), "LU_INDEX": get("LU_INDEX"),
            "nonfinite": nonfinite, "isice": isice,
        }
    # Half-level pressures, bottom first like the tape.
    if hybrid is not None:
        a_half, b_half = hybrid
        p_half = (a_half[:, None, None] + b_half[:, None, None] * ps[None])[::-1]
        z_half = phi / G_ERA5
        ln_p = np.log(np.maximum(p_half, 1.0))
        z_levels = z_half
    else:
        ln_p = np.log(pressure)
        z_levels = 0.5 * (phi[:-1] + phi[1:]) / G_ERA5
        p_half = None
    target = np.log(50000.0)
    z500 = np.full(ps.shape, np.nan)
    for k in range(ln_p.shape[0] - 1):
        deep, high = ln_p[k], ln_p[k + 1]
        inside = (deep >= target) & (high <= target) & np.isnan(z500)
        share = np.where(deep > high, (deep - target) / np.where(deep > high, deep - high, 1.0), 0.0)
        z500 = np.where(inside, z_levels[k] + share * (z_levels[k + 1] - z_levels[k]), z500)
    tape["Z500"] = z500
    q = qv / (1.0 + qv)
    if p_half is None:
        # Layer thickness in pressure from the mass levels: interfaces at
        # the midpoints, the surface at the bottom and nothing above the top
        # mass level counted.
        mids = 0.5 * (pressure[:-1] + pressure[1:])
        p_half = np.concatenate([ps[None], mids, np.zeros_like(ps)[None]], axis=0)
    tape["TPW"] = np.sum(q * (p_half[:-1] - p_half[1:]), axis=0) / G_ERA5
    return tape


def surface_classes(start: dict) -> dict:
    """The surface classes of the start tape, held for every later tape."""

    lat = start["lat"]
    water = start["XLAND"] >= 1.5
    ice = start["SEAICE"] >= 0.5
    land = ~water & ~ice
    land_ice = land & (np.rint(start["LU_INDEX"]).astype(int) == start["isice"])
    snow = start["SNOW"]
    return {
        "land ice, south of 60S": land_ice & (lat < -60),
        "land ice, north of 55N": land_ice & (lat > 55),
        "sea ice, north": ice & (lat > 0),
        "sea ice, south": ice & (lat < 0),
        "open water": water & ~ice,
        "snow-free land": land & ~land_ice & (snow <= 0.0),
        f"snowy land that is not land ice (start > {SNOWY_KG_M2:g} kg m-2)":
            land & ~land_ice & (snow > SNOWY_KG_M2),
        f"  of it, start > {DEEP_SNOW_KG_M2:g} kg m-2":
            land & ~land_ice & (snow > DEEP_SNOW_KG_M2),
    }


def compare_tapes(pairs, hybrid=None) -> dict:
    tapes = []
    for tape_path, era5_path in pairs:
        tape = read_tape(tape_path, hybrid)
        era5 = read_era5_file(era5_path)
        if era5["valid"] != tape["valid"]:
            raise ValueError(f"{tape_path} is valid at {tape['valid']} and {era5_path} "
                             f"at {era5['valid']}; each tape needs ERA5 of its own hour")
        tapes.append((tape, era5))
    tapes.sort(key=lambda pair: pair[0]["valid"])
    start = tapes[0][0]
    classes = surface_classes(start)
    out = []
    for tape, era5 in tapes:
        lat, lon = tape["lat"], tape["lon"]
        sample = Sampler(era5["lat"], era5["lon"], lat[:, 0], lon[0])
        weights = np.cos(np.deg2rad(lat))
        i500 = int(np.argmin(np.abs(era5["level"] - 50000.0)))
        reference = {
            "surface pressure, hPa": (tape["PSFC"] / 100.0, sample(era5["PSFC"]) / 100.0),
            "500 hPa height, m": (tape["Z500"], sample(era5["Z"][i500]) / G_ERA5),
            "2 m temperature, K": (tape["T2"], sample(era5["T2"])),
            "precipitable water, kg m-2": (
                tape["TPW"], sample(era5_precipitable_water(era5["level"], era5["SPFH"], era5["PSFC"]))),
        }
        globe = {label: area_stats(a, b, weights) for label, (a, b) in reference.items()}
        t2_ref, skin_ref = sample(era5["T2"]), sample(era5["SKINTEMP"])
        by_class = {}
        for label, mask in classes.items():
            if not mask.any():
                by_class[label] = None
                continue
            w = weights[mask]
            by_class[label] = {
                "n": int(mask.sum()),
                "skin": area_stats(tape["TSK"], skin_ref, weights, mask),
                "t2": area_stats(tape["T2"], t2_ref, weights, mask),
                "snow_mean_kg_m2": float((tape["SNOW"][mask] * w).sum() / w.sum()),
                "snow_change_kg_m2": float(((tape["SNOW"] - start["SNOW"])[mask] * w).sum() / w.sum()),
            }
        out.append({"tape": tape["path"], "valid": tape["valid"].isoformat(),
                    "lead_h": (tape["valid"] - start["valid"]).total_seconds() / 3600.0,
                    "nonfinite": tape["nonfinite"], "global": globe, "by_class": by_class})
    return {"tapes": out, "classes_from": start["path"]}


def print_tapes(result: dict, stream=sys.stdout) -> None:
    for tape in result["tapes"]:
        print(f"\n{tape['valid']} (+{tape['lead_h']:g} h)  {tape['tape']}  "
              f"non-finite values: {tape['nonfinite']}", file=stream)
        for label, s in tape["global"].items():
            print(f"  {label:32s} ERA5 {s['era5_mean']:9.3f} tape {s['model_mean']:9.3f} "
                  f"bias {s['bias']:8.3f} rms {s['rms']:7.3f}", file=stream)
        for label, s in tape["by_class"].items():
            if s is None:
                print(f"  {label:58s} no columns", file=stream)
                continue
            print(f"  {label:58s} n {s['n']:6d} skin bias {s['skin']['bias']:6.2f} rms "
                  f"{s['skin']['rms']:5.2f} | T2 bias {s['t2']['bias']:6.2f} rms {s['t2']['rms']:5.2f} "
                  f"| snow {s['snow_mean_kg_m2']:8.1f} kg m-2, change {s['snow_change_kg_m2']:+8.2f}",
                  file=stream)


# ------------------------------------------------------------------- main

def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("read", help="one ERA5 hour straight from the ARCO store (zarr-python)")
    p.add_argument("cycle", metavar="YYYY-MM-DDTHH")
    p.add_argument("out")
    p = sub.add_parser("state", help="the cold start of an ERA5 configuration on the CPU, no step")
    p.add_argument("config")
    p.add_argument("out")
    p.add_argument("--scratch", help="where the engine stages the decode (default: beside OUT)")
    p = sub.add_parser("hour0", help="the hour-0 state against ERA5 read by `read`")
    p.add_argument("state")
    p.add_argument("era5")
    p.add_argument("--json")
    p = sub.add_parser("tapes", help="render tapes against ERA5 of their valid times")
    p.add_argument("pairs", nargs="+", metavar="TAPE=ERA5_NC")
    p.add_argument("--config", help="take the half-level pressures from this configuration")
    p.add_argument("--json")
    args = parser.parse_args(argv)

    if args.command == "read":
        when = dt.datetime.strptime(args.cycle, "%Y-%m-%dT%H")
        record = read_era5_hour(when, args.out)
        print(json.dumps(record, indent=1))
        return 0
    if args.command == "state":
        record = write_cold_start_state(args.config, args.out, args.scratch)
        print(json.dumps(record, indent=1, default=str))
        return 0
    if args.command == "hour0":
        result = compare_hour0(_load_npz(args.state), _load_npz(args.era5))
        print_hour0(result)
    else:
        hybrid = None
        if args.config:
            from arwen_global.config import load_config

            cfg = load_config(args.config)
            hybrid = (np.asarray(cfg.a_half_pa, dtype=np.float64),
                      np.asarray(cfg.b_half, dtype=np.float64))
        pairs = []
        for pair in args.pairs:
            tape, sep, era5 = pair.rpartition("=")
            if not sep:
                parser.error(f"{pair!r} is not TAPE=ERA5_NC")
            pairs.append((tape, era5))
        result = compare_tapes(pairs, hybrid)
        print_tapes(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
