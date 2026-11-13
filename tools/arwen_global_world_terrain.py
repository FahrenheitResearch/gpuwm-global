"""Measure an invented world's terrain for the dynamics.

Two commands, both reading the module ``arwen_global.world.terrain``:

``report``  averages an elevation map to a truncation's grid, filters it
            with each filter named, and prints the false-height report of
            each (CPU only, float64)::

    python tools/arwen_global_world_terrain.py report MAP.nc \\
        --truncation 255 --cutoff none --cutoff 1.0 --cutoff 0.7 \\
        --out report.json

``frame``   samples the filtered terrain onto regular pole-to-pole frames
            (``--step-deg``, default 1 and 0.25), puts each back through
            the cold start's own rule (bilinear onto the Gaussian grid,
            then the truncation) and prints how far it lands from the
            terrain it came from (CPU only)::

    python tools/arwen_global_world_terrain.py frame MAP.nc \\
        --truncation 255 --step-deg 1 --step-deg 0.25 --out frame.json

``rest``    an atmosphere at rest, physics off, over the ring mountain or
            over a map's filtered terrain, stepped with the dynamics of a
            run config (its grid, levels, integrator, step, diffusion,
            semi-implicit settings and fixers), and the largest wind of
            each day::

    python tools/arwen_global_world_terrain.py rest \\
        --config src/arwen_global/configs/arwen_global_gdas_t255_sl_si_dry.toml \\
        --terrain ring --days 6 --out ring.json

The map is a netCDF file with one-dimensional latitude and longitude
variables and a two-dimensional elevation variable (``--lat``, ``--lon``,
``--elevation`` name them; the defaults are the PALEOMAP files' own).
The config's ``[initial]`` table is replaced by a resting analytic start
and its ``[physics]`` by ``mode = "none"``; nothing else of it changes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import tomllib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arwen_global.constants import GRAVITY_M_S2  # noqa: E402
from arwen_global.world.terrain import (  # noqa: E402
    NO_FILTER, TerrainFilter, dynamics_terrain, rest_test, resting_state,
    ring_mountain,
)


def _read_map(path, lat_name, lon_name, elevation_name):
    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        z = np.ma.filled(np.ma.asarray(ds.variables[elevation_name][:],
                                       dtype=np.float64), np.nan)
    return z, lat, lon


def _filters(values):
    out = []
    for value in values or ["none", "default"]:
        if value == "none":
            out.append(NO_FILTER)
        elif value == "default":
            out.append(TerrainFilter())
        else:
            out.append(TerrainFilter(cutoff_fraction=float(value)))
    return out


def _host_transform(truncation):
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    return SphericalHarmonicTransform.create(
        int(truncation), backend="numpy", precision="float64")


def report(args) -> int:
    z, lat, lon = _read_map(args.map, args.lat, args.lon, args.elevation)
    transform = _host_transform(args.truncation)
    rows = []
    for terrain_filter in _filters(args.cutoff):
        built = dynamics_terrain(z, lat, lon, transform,
                                 sea_level_m=args.sea_level_m,
                                 terrain_filter=terrain_filter)
        print(built.report.text(), flush=True)
        rows.append(built.report.record())
    text = json.dumps({"map": Path(args.map).name, "rows": rows}, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


def frame(args) -> int:
    """The filtered terrain sampled onto a regular pole-to-pole frame, put
    back through the cold start's own rule (periodic bilinear onto the
    Gaussian grid, the NumPy oracle the Rust door is byte-identical to,
    then a sharp truncation), and compared with the terrain it came from."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from coldstart_oracle import numpy_global_regridder
    from arwen_global.world.terrain import false_height_report

    z, lat, lon = _read_map(args.map, args.lat, args.lon, args.elevation)
    transform = _host_transform(args.truncation)
    grid = transform.grid
    built = dynamics_terrain(z, lat, lon, transform,
                             sea_level_m=args.sea_level_m,
                             terrain_filter=_filters([args.cutoff])[0])
    print(built.report.text(), flush=True)
    rows = []
    for step in args.step_deg:
        nlat = int(round(180.0 / step)) + 1
        nlon = int(round(360.0 / step))
        flat = np.linspace(-90.0, 90.0, nlat)
        ring = np.arange(nlon) * (360.0 / nlon)
        planes = built.on_latlon(flat, ring)
        regrid = numpy_global_regridder(flat, ring, grid)
        back = np.asarray(transform.inverse(transform.forward(regrid(planes))))
        difference = back - built.height_m
        again = false_height_report(
            back, built.averaged_m, built.land_fraction, grid,
            built.terrain_filter, map_peak_m=built.report.map_peak_m,
            map_mean_m=built.report.map_mean_m)
        row = {
            "frame_step_deg": float(step),
            "frame_shape": [nlat, nlon],
            "max_abs_difference_m": float(np.max(np.abs(difference))),
            "rms_difference_m": float(grid.rms(difference)),
            "after_round_trip": again.record(),
        }
        print(f"frame {step:g} deg ({nlat}x{nlon}): back on the model grid "
              f"differs by up to {row['max_abs_difference_m']:.1f} m (rms "
              f"{row['rms_difference_m']:.2f} m); pit over water "
              f"{again.lowest_over_water_m:.1f} m, peak "
              f"{again.model_peak_m:.0f} m", flush=True)
        rows.append(row)
    text = json.dumps({"map": Path(args.map).name,
                       "terrain": built.report.record(), "frames": rows},
                      indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}"
                                for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def _toml_tables(table, prefix=""):
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in table.items() if isinstance(v, dict)}
    lines = []
    if prefix and (scalars or not tables):
        lines.append(f"[{prefix}]")
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables.items():
        lines.extend(_toml_tables(value, f"{prefix}.{key}" if prefix else key))
    return lines


def _scalar(text):
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def rest_config(base_path, outdir: Path, backend=None, memory=()) -> Path:
    """The base config with a resting analytic start and physics off
    (and, when given, that backend and those ``[memory]`` settings: the
    memory levers change no arithmetic and enter no identity, so a banded
    or spilled run measures the same dynamics in less of the card)."""
    raw = tomllib.loads(Path(base_path).read_text(encoding="utf-8"))
    if backend:
        raw.setdefault("arwen_global", {})["backend"] = str(backend)
    for item in memory or ():
        key, _, value = str(item).partition("=")
        if not key or not _:
            raise SystemExit(f"--memory takes KEY=VALUE, got {item!r}")
        raw.setdefault("memory", {})[key.strip()] = _scalar(value)
    raw["initial"] = {
        "mode": "analytic", "surface_pressure_pa": 100000.0,
        "surface_temperature_k": 288.0, "top_temperature_k": 215.0,
        "qv_surface": 0.0, "zonal_wind_m_s": 0.0,
        "perturbation_amplitude": 0.0, "zonal_wavenumber": 1,
        "terrain_amplitude_m": 0.0, "surface_water_kg_m2": 0.0,
    }
    raw["physics"] = {"mode": "none"}
    for drop in ("statics", "world"):
        raw.pop(drop, None)
    raw.setdefault("arwen_global", {})["name"] = "arwen-global-world-terrain-rest"
    path = outdir / "rest-config.toml"
    path.write_text("\n".join(_toml_tables(raw)) + "\n", encoding="utf-8")
    return path


def _peak_device_bytes(backend):
    if backend.name != "cupy":
        return None
    import cupy

    return int(cupy.get_default_memory_pool().total_bytes())


def rest(args) -> int:
    from arwen_global.config import load_config
    from arwen_global.runner import attach_host_tier, build_model_and_cold_state

    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp())
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(rest_config(args.config, outdir, args.backend,
                                  args.memory))
    model, cold = build_model_and_cold_state(cfg)
    transform = model.transform
    backend = transform.backend
    terrain_record = {"terrain": args.terrain}
    if args.terrain == "ring":
        height = ring_mountain(transform.grid)
    elif args.terrain == "flat":
        height = np.zeros(transform.grid.shape)
    else:
        z, lat, lon = _read_map(args.terrain, args.lat, args.lon,
                                args.elevation)
        terrain_filter = (NO_FILTER if args.cutoff == "none"
                          else TerrainFilter() if args.cutoff == "default"
                          else TerrainFilter(cutoff_fraction=float(args.cutoff)))
        built = dynamics_terrain(z, lat, lon, transform,
                                 sea_level_m=args.sea_level_m,
                                 terrain_filter=terrain_filter)
        print(built.report.text(), flush=True)
        terrain_record = {"terrain": Path(args.terrain).name,
                          "report": built.report.record()}
        height = built.height_m
    bundle, phis = resting_state(transform, cfg.vertical,
                                 GRAVITY_M_S2 * height, cold.surface,
                                 cold.physics_state)
    model.surface_geopotential = phis
    bundle, _, _, _ = model._repair_positivity(bundle)
    model.initialize_mass_target(bundle.atmosphere)
    model.initialize_water_target(bundle)
    model.enforce(bundle)
    bundle = attach_host_tier(cfg, model, bundle)
    terrain_record["model_peak_m"] = float(
        np.max(backend.to_numpy(phis))) / GRAVITY_M_S2

    def progress(sample):
        print(f"  hour {sample['hour']:7.1f}  max wind "
              f"{sample['max_wind_m_s']:.4e} m/s (level {sample['level']}, "
              f"{sample['lat_deg']:.2f}N {sample['lon_deg']:.2f}E)  rms "
              f"{sample['rms_wind_m_s']:.3e}", flush=True)

    started = time.perf_counter()
    result = rest_test(model, bundle, dt_s=float(cfg.dt_s), days=args.days,
                       samples_per_day=args.samples_per_day,
                       progress=progress)
    result.pop("final_state")
    summary = {
        "config": Path(args.config).name,
        "truncation": int(cfg.truncation),
        "nlev": int(cfg.vertical.nlev),
        "integrator": cfg.integrator,
        "backend": backend.name,
        "precision": str(np.dtype(backend.float_dtype)),
        "memory": list(args.memory),
        **terrain_record,
        **result,
        "total_wall_s": float(time.perf_counter() - started),
        "device_pool_bytes": _peak_device_bytes(backend),
    }
    gate = args.gate_m_s
    if gate is not None:
        summary["gate_m_s"] = float(gate)
        summary["gate_passed"] = bool(summary["largest_m_s"] < float(gate))
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    if gate is not None and not summary["gate_passed"]:
        return 1
    if not math.isfinite(summary["largest_m_s"]):
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def map_arguments(sub):
        sub.add_argument("--lat", default="lat")
        sub.add_argument("--lon", default="lon")
        sub.add_argument("--elevation", default="z")
        sub.add_argument("--sea-level-m", type=float, default=0.0)

    rep = commands.add_parser("report")
    rep.add_argument("map")
    rep.add_argument("--truncation", type=int, default=255)
    rep.add_argument("--cutoff", action="append",
                     help="'none', 'default' or a cutoff fraction; repeat")
    rep.add_argument("--out", default=None)
    map_arguments(rep)
    rep.set_defaults(handler=report)

    frm = commands.add_parser("frame")
    frm.add_argument("map")
    frm.add_argument("--truncation", type=int, default=255)
    frm.add_argument("--cutoff", default="default")
    frm.add_argument("--step-deg", type=float, action="append",
                     default=None)
    frm.add_argument("--out", default=None)
    map_arguments(frm)
    frm.set_defaults(handler=frame)

    rst = commands.add_parser("rest")
    rst.add_argument("--config", required=True)
    rst.add_argument("--terrain", default="ring",
                     help="'ring', 'flat' or a map file")
    rst.add_argument("--cutoff", default="default")
    rst.add_argument("--days", type=float, default=6.0)
    rst.add_argument("--samples-per-day", type=int, default=24)
    rst.add_argument("--gate-m-s", type=float, default=None)
    rst.add_argument("--backend", default=None,
                     help="override the config's backend (numpy or cupy)")
    rst.add_argument("--memory", action="append", default=[],
                     help="a [memory] setting, KEY=VALUE; repeat "
                          "(e.g. latitude_bands=8, host_spill=on)")
    rst.add_argument("--outdir", default=None)
    rst.add_argument("--out", default=None)
    map_arguments(rst)
    rst.set_defaults(handler=rest)

    args = parser.parse_args(argv)
    if getattr(args, "step_deg", "absent") is None:
        args.step_deg = [1.0, 0.25]
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
