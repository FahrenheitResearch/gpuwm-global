"""Two CPU benchmarks for other worlds: an atmosphere at rest over Earth's
terrain, and a steady rotating state on a ladder of rotation rates.

Both run the dynamical core alone (physics off, the shipped semi-Lagrangian
core, its hyperdiffusion, sponge and fixers) at the normal 300 s step, one
model step at a time, and read the wind and the surface pressure every few
model hours.

*   ``rest``: temperature a function of pressure only, no wind, surface
    pressure hydrostatic at the ground (``arwen_global.testcases``).  The
    continuous equations hold it still over any terrain, so the wind that
    grows is the core's own error over mountains.  Run over Earth's terrain
    at full height, half height and none: with no terrain the wind is
    rounding, and the number per day over full and half terrain is the
    floor every no-mountain twin's difference map sits on.

*   ``ladder``: the Jablonowski and Williamson 2006 steady state built for
    0.5, 1, 2, 4 and 8 times Earth's rotation.  By default the jet is
    scaled by one over the factor, which holds the state's temperature
    contrast at Earth's (thermal wind: f times the shear is what the
    temperature gradient balances), so every rung is the same planet's
    air on a faster or slower spin.  A rung "stays steady" when it runs
    every step with no refusal and its surface-pressure drift stays within
    ten times the Earth rung's (or 10 Pa, whichever is larger): the rule is
    a judgment set before the runs, printed with the numbers.  A rung that
    is unstable at the normal step is refused at once by the core's own
    guards, or blows up by orders of magnitude, so the margin is wide on
    purpose.

The model's rotation enters the dynamics only through its Coriolis plane
(``MoistHybridModel.coriolis``); this tool builds the model at the rung's
rate with ``dataclasses.replace`` and touches nothing in the package.
Passing a world's rotation to a real run is package 3.1's.

Usage::

    python tools/arwen_global_world_benchmarks.py rest \\
        --terrain era5-combined.nc --terrain-variable SOILGEO \\
        --scales 1,0.5,0 --days 6 --outdir OUT
    python tools/arwen_global_world_benchmarks.py ladder \\
        --factors 0.5,1,2,4,8 --days 6 --outdir OUT

Both default to the shipped T85 L40 Jablonowski-Williamson semi-Lagrangian
config (dt 300 s) on numpy in that config's float32; ``--truncation``,
``--backend`` and ``--precision`` derive another config from it, written
into the output folder so the run can be repeated from it.
``--diffusion no-pressure`` and ``--diffusion off`` turn the hyperdiffusion's
surface-pressure term, or the whole hyperdiffusion, off: it acts along the
terrain-following levels, so over mountains it is a source of motion in
its own right and these two say how much of the rest test's wind is its.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arwen_global.config import load_config  # noqa: E402
from arwen_global.constants import EARTH_ROTATION_RATE_S, GRAVITY_M_S2  # noqa: E402
from arwen_global.runner import build_model_and_cold_state  # noqa: E402
from arwen_global.testcases import (  # noqa: E402
    JW_U0,
    baroclinic_wave_initial_state,
    rest_atmosphere_initial_state,
    terrain_on_model_grid,
)

#: The shipped config both benchmarks derive theirs from.
DEFAULT_CONFIG = "arwen_global_jw06_t85_sl_si_steady.toml"
#: The ladder's rungs and the rest test's terrain scales.
DEFAULT_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_SCALES = (1.0, 0.5, 0.0)
#: The steadiness rule: a rung's final RMS surface-pressure drift may be at
#: most this many times the Earth rung's, with this floor in Pa.
STEADY_RATIO = 10.0
STEADY_FLOOR_PA = 10.0


# ----------------------------------------------------------------- config

def derived_config_text(text: str, *, backend=None, precision=None,
                        truncation=None, dt_s=None) -> str:
    """The shipped config's text with ``backend``, ``precision``,
    ``truncation`` and ``dt_s`` replaced where given.  Each key occurs once
    in the shipped file, and a key that is not found is refused rather than
    silently left alone."""
    for key, value in (("backend", backend), ("precision", precision),
                       ("truncation", truncation), ("dt_s", dt_s)):
        if value is None:
            continue
        if key == "truncation":
            literal = str(int(value))
        elif key == "dt_s":
            literal = repr(float(value))
        else:
            literal = f'"{value}"'
        pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
        if len(pattern.findall(text)) != 1:
            raise ValueError(f"config text has no single '{key} =' line")
        text = pattern.sub(f"{key} = {literal}", text)
    return text


#: What ``--diffusion`` can ask of the shipped hyperdiffusion: as shipped,
#: with the surface-pressure term off, or off altogether.  The two
#: departures exist to say how much of the wind over terrain is the
#: diffusion's (it acts on terrain-following levels) and how much is the
#: rest of the core's.
DIFFUSION_CHOICES = ("shipped", "no-pressure", "off")


def diffusion_config_text(text: str, choice: str) -> str:
    """``text`` with its ``[diffusion]`` table changed as ``choice`` says."""
    if choice == "shipped":
        return text
    if choice not in DIFFUSION_CHOICES:
        raise ValueError(f"diffusion must be one of {DIFFUSION_CHOICES}")
    match = re.search(r"^\[diffusion\]\n(.*?)(?=^\[|\Z)", text,
                      re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("config text has no [diffusion] table")
    body = match.group(1)
    if choice == "off":
        body, count = re.subn(r"^enabled\s*=.*$", "enabled = false", body,
                              flags=re.MULTILINE)
        if count != 1:
            raise ValueError("[diffusion] has no single 'enabled =' line")
    else:
        body = re.sub(r"^pressure_strength\s*=.*\n?", "", body, flags=re.MULTILINE)
        body = "pressure_strength = 0.0\n" + body
    return text[:match.start(1)] + body + text[match.end(1):]


def prepare_config(args, outdir: Path):
    if args.config:
        source = Path(args.config)
    else:
        from arwen_global.configs_dir import config_root

        source = Path(config_root()) / DEFAULT_CONFIG
    text = derived_config_text(
        source.read_text(encoding="utf-8"), backend=args.backend,
        precision=args.precision, truncation=args.truncation,
        dt_s=getattr(args, "dt_s", None))
    text = diffusion_config_text(text, getattr(args, "diffusion", "shipped"))
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "benchmark-config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path), path


# ------------------------------------------------------------------ model

def world_model(cfg, build_state, *, rotation_rate_s: float):
    """The run's model and cold state for a state the runner does not
    build: ``build_state(transform)`` returns ``(bundle,
    surface_geopotential, provenance)``.

    The model is the runner's own for ``cfg`` (same core, diffusion,
    sponge, fixers), re-made with the case's terrain and rotation rate, and
    the cold state is prepared the way the runner prepares its own.
    """
    model, _ = build_model_and_cold_state(cfg)
    bundle, surface_geopotential, provenance = build_state(model.transform)
    clock = getattr(model, "forecast_clock", None)
    model = dataclasses.replace(
        model, surface_geopotential=surface_geopotential,
        rotation_rate_s=float(rotation_rate_s),
        initial_provenance=provenance,
    )
    model.forecast_clock = clock
    bundle, _, _, _ = model._repair_positivity(bundle)
    model.initialize_mass_target(bundle.atmosphere)
    model.initialize_water_target(bundle)
    model.enforce(bundle)
    return model, bundle


# ---------------------------------------------------------------- reading

def _cell_weights(transform):
    weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    cell = weights[:, None] / (2.0 * transform.grid.nlon)
    return np.broadcast_to(cell, transform.grid.shape)


def read_state(model, bundle):
    """``(u, v, ps)`` on the grid, float64 on the host."""
    host = model.transform.backend.to_numpy
    g = model.grid_state(bundle.atmosphere, only=("u", "v", "ps"))
    return (np.asarray(host(g["u"]), dtype=np.float64),
            np.asarray(host(g["v"]), dtype=np.float64),
            np.asarray(host(g["ps"]), dtype=np.float64))


def eddy_spectrum(field, grid, *, truncation=None) -> dict:
    """Where the zonal power of a change field ``(level, lat, lon)`` sits:
    the zonal wavenumber that holds the most of it (0 excluded), and the
    share in the top third of the resolved wavenumbers.  A state that only
    a physical instability disturbs puts its power at the instability's
    own scale; a numerical one puts it at the grid's."""
    power = np.abs(np.fft.rfft(np.asarray(field, dtype=np.float64), axis=-1)) ** 2
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    spectrum = np.einsum("klm,l->m", power, weights)
    top = int(truncation) if truncation else spectrum.size - 1
    spectrum = spectrum[1:top + 1]
    total = float(spectrum.sum())
    if total <= 0.0:
        return {"eddy_peak_wavenumber": 0, "eddy_top_third_share": 0.0}
    cut = (2 * top) // 3
    return {
        "eddy_peak_wavenumber": int(np.argmax(spectrum)) + 1,
        "eddy_top_third_share": float(spectrum[cut:].sum() / total),
    }


def sample(model, bundle, start, *, step: int, dt_s: float) -> dict:
    """One reading against the start ``(u0, v0, ps0)``."""
    u, v, ps = read_state(model, bundle)
    u0, v0, ps0 = start
    cell = _cell_weights(model.transform)
    grid = model.transform.grid
    speed = np.sqrt(u * u + v * v)
    where = np.unravel_index(int(np.argmax(speed)), speed.shape)
    du = u - u0
    dv = v - v0
    dps = ps - ps0
    change = np.sqrt(du * du + dv * dv)
    lowest = speed[-1]
    return {
        "step": int(step),
        "hour": float(step * dt_s / 3600.0),
        "finite": bool(np.isfinite(speed).all() and np.isfinite(ps).all()),
        "max_wind_m_s": float(speed.max()),
        "max_wind_level": int(where[0]),
        "max_wind_lat_deg": float(np.asarray(grid.latitude_deg)[where[1]]),
        "max_wind_lon_deg": float(np.asarray(grid.longitude_deg)[where[2]]),
        "max_wind_lowest_level_m_s": float(lowest.max()),
        "rms_wind_m_s": float(math.sqrt(float(np.mean(
            np.sum(speed * speed * cell, axis=(1, 2)))))),
        "max_wind_change_m_s": float(change.max()),
        "rms_wind_change_m_s": float(math.sqrt(float(np.mean(
            np.sum(change * change * cell, axis=(1, 2)))))),
        "max_meridional_wind_m_s": float(np.abs(v).max()),
        "max_ps_change_pa": float(np.abs(dps).max()),
        "rms_ps_change_pa": float(math.sqrt(float(np.sum(dps * dps * cell)))),
        "global_mean_ps_pa": float(np.sum(ps * cell)),
        **eddy_spectrum(dv, model.transform.grid, truncation=getattr(
            model.transform, "truncation", None)),
    }


def integrate(model, bundle, *, dt_s: float, steps: int, sample_every: int,
              label: str = "", progress=print) -> dict:
    """Step ``steps`` times, reading every ``sample_every`` steps and at the
    end.  A refusal by the core or a non-finite reading ends the run and is
    recorded with its reason, never raised."""
    start = read_state(model, bundle)
    rows = [sample(model, bundle, start, step=0, dt_s=dt_s)]
    peaks: dict[str, float] = {}
    refusal = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            bundle, metrics = model.step(bundle, dt_s)
        except Exception as error:  # noqa: BLE001 -- recorded, not raised
            refusal = {"step": step, "hour": step * dt_s / 3600.0,
                       "type": type(error).__name__, "reason": str(error)[:2000]}
            break
        for key, value in dict(metrics).items():
            if ("lipschitz" in key or "cfl" in key or "fold" in key) and isinstance(
                    value, (int, float, np.floating)):
                value = float(value)
                if key.startswith("min") or "determinant" in key:
                    peaks[key] = min(peaks.get(key, value), value)
                else:
                    peaks[key] = max(peaks.get(key, value), value)
        if step % sample_every == 0 or step == steps:
            row = sample(model, bundle, start, step=step, dt_s=dt_s)
            rows.append(row)
            progress(f"  {label} h {row['hour']:7.1f}  max wind "
                     f"{row['max_wind_m_s']:.4e}  rms dps "
                     f"{row['rms_ps_change_pa']:.4e}  "
                     f"{time.perf_counter() - started:7.0f} s", flush=True)
            if not row["finite"]:
                refusal = {"step": step, "hour": row["hour"],
                           "type": "NonFinite", "reason": "a non-finite wind or pressure"}
                break
    pool_bytes = None
    if model.transform.backend.name == "cupy":
        # What the memory pool holds at the run's end.  The pool keeps
        # every block it has handed out, so this is the run's reserved
        # peak so far in the process: the number a card lease is sized from.
        pool_bytes = int(model.transform.backend.xp.get_default_memory_pool().total_bytes())
    return {
        "device_pool_bytes": pool_bytes,
        "steps_requested": int(steps),
        "steps_run": int(rows[-1]["step"]) if refusal is None else int(refusal["step"]) - 1,
        "completed": refusal is None,
        "refusal": refusal,
        "wall_s": float(time.perf_counter() - started),
        "step_metric_extremes": peaks,
        "rows": rows,
    }


def per_day(rows, key: str, dt_s: float) -> list[dict]:
    """The largest ``key`` in each model day, from the readings."""
    out: dict[int, float] = {}
    for row in rows:
        if row["step"] == 0:
            continue
        day = int(math.ceil(row["hour"] / 24.0 - 1.0e-9))
        out[day] = max(out.get(day, -math.inf), row[key])
    return [{"day": day, key: value} for day, value in sorted(out.items())]


# ------------------------------------------------------------------- rest

def read_terrain(path: str, variable: str | None):
    """``(latitude_deg, longitude_deg, height_m)`` from an ``.npz`` with
    those three names, or from a netCDF variable (geopotential in m2 s-2,
    which is divided by g, or a height in m)."""
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source) as data:
            return (np.asarray(data["latitude_deg"]), np.asarray(data["longitude_deg"]),
                    np.asarray(data["terrain_height_m"], dtype=np.float64))
    import netCDF4

    with netCDF4.Dataset(source) as data:
        name = variable or "SOILGEO"
        var = data.variables[name]
        values = np.asarray(np.ma.filled(var[...], np.nan), dtype=np.float64).squeeze()
        units = str(getattr(var, "units", "m")).replace("**", "").replace("^", "")
        lat = np.asarray(data.variables["latitude"][...], dtype=np.float64)
        lon = np.asarray(data.variables["longitude"][...], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"terrain variable {name} carries non-finite values")
    if units.replace(" ", "") in ("m2s-2", "m2/s2"):
        values = values / GRAVITY_M_S2
    elif units.strip() != "m":
        raise ValueError(f"terrain variable {name} has units {units!r}: "
                         "want a height in m or a geopotential in m2 s-2")
    return lat, lon, values


def rest_case(cfg, terrain_model_grid_m, scale: float):
    """Model and cold state of the rest benchmark at one terrain scale."""

    def build(transform):
        height = (np.zeros(transform.grid.shape) if terrain_model_grid_m is None
                  else float(scale) * terrain_model_grid_m(transform.grid))
        return rest_atmosphere_initial_state(cfg, transform, height)

    return world_model(cfg, build, rotation_rate_s=EARTH_ROTATION_RATE_S)


def run_rest(args) -> int:
    outdir = Path(args.outdir)
    cfg, config_path = prepare_config(args, outdir)
    terrain = None
    terrain_source = None
    if args.terrain:
        lat, lon, height = read_terrain(args.terrain, args.terrain_variable)
        terrain_source = {"path_name": Path(args.terrain).name,
                          "variable": args.terrain_variable,
                          "source_max_m": float(height.max()),
                          "source_min_m": float(height.min())}

        def terrain(grid):
            return terrain_on_model_grid(lat, lon, height, grid)
    dt = float(cfg.dt_s)
    steps = int(round(args.days * 86400.0 / dt))
    every = max(1, int(round(args.sample_hours * 3600.0 / dt)))
    report = {"benchmark": "rest-over-terrain", "config": config_path.name,
              "truncation": cfg.truncation, "nlev": cfg.vertical.nlev,
              "dt_s": dt, "backend": cfg.backend, "precision": cfg.precision,
              "integrator": cfg.integrator, "days": args.days,
              "diffusion": args.diffusion,
              "terrain": terrain_source, "runs": []}
    for scale in parse_list(args.scales):
        if scale != 0.0 and terrain is None:
            raise SystemExit("a terrain scale other than 0 needs --terrain")
        model, bundle = rest_case(cfg, terrain, scale)
        run = integrate(model, bundle, dt_s=dt, steps=steps, sample_every=every,
                        label=f"scale {scale:g}")
        run["terrain_scale"] = scale
        run["terrain_max_m"] = float(
            np.max(model.transform.backend.to_numpy(model.surface_geopotential))
            / GRAVITY_M_S2)
        run["largest_wind_per_day"] = per_day(run["rows"], "max_wind_m_s", dt)
        run["largest_lowest_level_wind_per_day"] = per_day(
            run["rows"], "max_wind_lowest_level_m_s", dt)
        report["runs"].append(run)
        write_report(outdir / f"rest{args.tag}.json", report)
    print(json.dumps(summary_rest(report), indent=2))
    return 0


def summary_rest(report) -> dict:
    return {
        "benchmark": report["benchmark"],
        "runs": [{
            "terrain_scale": run["terrain_scale"],
            "terrain_max_m": run["terrain_max_m"],
            "completed": run["completed"],
            "refusal": run["refusal"],
            "wall_s": run["wall_s"],
            "device_pool_bytes": run["device_pool_bytes"],
            "largest_wind_per_day": run["largest_wind_per_day"],
        } for run in report["runs"]],
    }


# ----------------------------------------------------------------- ladder

def ladder_case(cfg, factor: float, *, hold_jet: bool = False):
    rate = EARTH_ROTATION_RATE_S * float(factor)
    jet = JW_U0 if hold_jet else JW_U0 / float(factor)

    def build(transform):
        return baroclinic_wave_initial_state(
            cfg, transform, rotation_rate_s=rate, u0_m_s=jet)

    model, bundle = world_model(cfg, build, rotation_rate_s=rate)
    return model, bundle, rate, jet


def steadiness(runs, *, earth_factor: float = 1.0) -> None:
    """Stamp each rung with the steadiness rule's verdict (in place)."""
    earth = [run for run in runs if run["rotation_factor"] == earth_factor]
    reference = None
    if earth and earth[0]["completed"]:
        reference = earth[0]["rows"][-1]["rms_ps_change_pa"]
    for run in runs:
        drift = run["rows"][-1]["rms_ps_change_pa"]
        run["final_rms_ps_change_pa"] = drift
        if reference is None:
            run["steady"] = None
            run["steady_limit_pa"] = None
            continue
        limit = max(STEADY_RATIO * reference, STEADY_FLOOR_PA)
        run["steady_limit_pa"] = limit
        run["steady"] = bool(run["completed"] and drift <= limit)


def run_ladder(args) -> int:
    outdir = Path(args.outdir)
    cfg, config_path = prepare_config(args, outdir)
    dt = float(cfg.dt_s)
    steps = int(round(args.days * 86400.0 / dt))
    every = max(1, int(round(args.sample_hours * 3600.0 / dt)))
    report = {"benchmark": "rotation-ladder", "config": config_path.name,
              "truncation": cfg.truncation, "nlev": cfg.vertical.nlev,
              "dt_s": dt, "backend": cfg.backend, "precision": cfg.precision,
              "integrator": cfg.integrator, "days": args.days,
              "diffusion": args.diffusion,
              "jet": "held at JW_U0" if args.hold_jet else "JW_U0 / factor",
              "steady_rule": {"ratio_to_earth": STEADY_RATIO,
                              "floor_pa": STEADY_FLOOR_PA},
              "runs": []}
    for factor in parse_list(args.factors):
        model, bundle, rate, jet = ladder_case(cfg, factor, hold_jet=args.hold_jet)
        run = integrate(model, bundle, dt_s=dt, steps=steps, sample_every=every,
                        label=f"x{factor:g}")
        run.update({
            "rotation_factor": factor,
            "rotation_rate_s": rate,
            "day_length_h": 24.0 / factor,
            "jet_m_s": jet,
            "coriolis_dt_at_pole": 2.0 * rate * dt,
        })
        run["largest_ps_change_per_day"] = per_day(run["rows"], "max_ps_change_pa", dt)
        run["largest_wind_change_per_day"] = per_day(run["rows"], "max_wind_change_m_s", dt)
        report["runs"].append(run)
        steadiness(report["runs"])
        write_report(outdir / f"ladder{args.tag}.json", report)
    print(json.dumps(summary_ladder(report), indent=2))
    return 0


def summary_ladder(report) -> dict:
    return {
        "benchmark": report["benchmark"],
        "runs": [{
            key: run.get(key) for key in (
                "rotation_factor", "day_length_h", "jet_m_s", "coriolis_dt_at_pole",
                "completed", "refusal", "final_rms_ps_change_pa", "steady_limit_pa",
                "steady", "wall_s", "device_pool_bytes", "largest_ps_change_per_day",
                "largest_wind_change_per_day", "step_metric_extremes")
        } for run in report["runs"]],
    }


# ------------------------------------------------------------------- main

def parse_list(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _common(parser) -> None:
    parser.add_argument("--config", default=None,
                        help=f"default: the shipped {DEFAULT_CONFIG}")
    parser.add_argument("--truncation", type=int, default=None)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--diffusion", default="shipped", choices=DIFFUSION_CHOICES)
    parser.add_argument("--dt-s", type=float, default=None,
                        help="another step than the config's (the ladder's dt test)")
    parser.add_argument("--days", type=float, default=6.0)
    parser.add_argument("--sample-hours", type=float, default=6.0)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tag", default="",
                        help="suffix of the report's file name")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    rest = commands.add_parser("rest")
    _common(rest)
    rest.add_argument("--terrain", default=None)
    rest.add_argument("--terrain-variable", default=None)
    rest.add_argument("--scales", default=",".join(f"{s:g}" for s in DEFAULT_SCALES))
    rest.set_defaults(handler=run_rest)

    ladder = commands.add_parser("ladder")
    _common(ladder)
    ladder.add_argument("--factors", default=",".join(f"{f:g}" for f in DEFAULT_FACTORS))
    ladder.add_argument("--hold-jet", action="store_true")
    ladder.set_defaults(handler=run_ladder)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
