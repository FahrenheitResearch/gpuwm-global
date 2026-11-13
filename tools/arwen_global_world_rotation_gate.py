"""The CPU gate of the re-balanced rotation start (other-worlds package 3.2).

Dry, at a small grid, at r times Earth's rotation, for days: does a start
whose winds were re-balanced by :mod:`arwen_global.world.rotation_start`
shed less divergence than the same start with Earth's winds?

Arms, each a dry adiabatic run (physics none) built from CONFIG's dynamics
(the shipped arwen_global_held_suarez_t42_sl_si.toml is the intended one:
sl_si, 40 levels, hyperdiffusion, mass fixer) on a NumPy float64 transform
at ``--truncation``:

  E1  the start's own winds at Earth's rotation (the reference),
  E2  the start's own winds at r times Earth's rotation,
  R2  the re-balanced winds at r times Earth's rotation.

Two starts.  ``jw``: the Jablonowski and Williamson steady state, which
balances Earth's rotation exactly.  ``era5``: a real analysis, an ERA5
combined NetCDF (the variables T, U, V on pressure levels, PSFC, SOILGEO)
put on the model by a NumPy mini cold start written here (bilinear onto
the Gaussian grid, linear in ln p onto the model levels, temperature
continued below the bottom level, surface pressure moved
hypsometrically onto the spectral terrain, no moisture).  That mini cold
start is NOT the model's cold start; both arms share it, and the rule
acts on the analysis frame itself, as the cold start's hook does.  The
thermal-wind imbalance of package 2.2's ``thermal_wind_imbalance`` is
printed when ``arwen_global.world.start`` is importable.

Printed as JSON lines every 6 model hours, and written to OUTJSON: the
largest |divergence|, its area-RMS, the RMS divergence over the RMS
vorticity (free of the factor r, which scales both), and the mean
absolute 3-hour surface-pressure change.  No checkpoint is written.

usage: python tools/arwen_global_world_rotation_gate.py CONFIG ERA5NC OUTJSON
       [--case jw|era5|both] [--arms E1,E2,R2] [--days 5] [--dt 900]
       [--factor 2] [--truncation 42] [--stride 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from types import SimpleNamespace

import numpy as np


def build_model(cfg, transform, phis, rotation):
    from arwen_global.dynamics import MoistHybridModel
    from arwen_global.runner import build_diffusion
    from arwen_global.semi_implicit import build_semi_implicit

    return MoistHybridModel(
        transform=transform, vertical=cfg.vertical,
        surface_geopotential=transform.backend.asarray(phis, dtype=transform.backend.float_dtype),
        physics=None, rotation_rate_s=rotation,
        physics_split=cfg.physics_split, integrator=cfg.integrator,
        diffusion=build_diffusion(cfg),
        divergence_diffusion_strength=cfg.divergence_diffusion_strength,
        pressure_diffusion_strength=cfg.pressure_diffusion_strength,
        water_diffusion_strength=cfg.water_diffusion_strength,
        semi_implicit=build_semi_implicit(
            cfg.semi_implicit_scheme, enabled=cfg.semi_implicit_enabled,
            weight=cfg.semi_implicit_weight,
            external_wave_speed_m_s=cfg.external_wave_speed_m_s,
            reference_temperature_k=cfg.semi_implicit_reference_temperature_k,
            reference_surface_pressure_pa=cfg.semi_implicit_reference_surface_pressure_pa,
            off_centring_weight=cfg.semi_implicit_off_centring_weight),
        mass_fixer=cfg.mass_fixer, water_fixer=False,
        positivity_repair=cfg.positivity_repair, maximum_cfl=cfg.maximum_cfl,
        minimum_fold_determinant=cfg.minimum_fold_determinant, semilag=cfg.semilag,
        sponge_base_pa=cfg.sponge_base_pa,
        sponge_lid_relaxation_time_s=cfg.sponge_lid_relaxation_time_s,
        sponge_heat_booking=cfg.sponge_heat_booking,
        sponge_mean_drag_base_pa=cfg.sponge_mean_drag_base_pa,
        sponge_mean_drag_time_s=cfg.sponge_mean_drag_time_s,
    )


def divergence_stats(model, bundle):
    transform = model.transform
    d = np.asarray(transform.backend.to_numpy(transform.inverse(bundle.atmosphere.divergence)))
    z = np.asarray(transform.backend.to_numpy(transform.inverse(bundle.atmosphere.vorticity)))
    w = np.cos(np.asarray(transform.grid.lat_rad))[None, :, None] * np.ones_like(d)
    rms_d = float(np.sqrt(np.sum(w * d * d) / np.sum(w)))
    # Relative vorticity with its zonal mean removed: the eddy rotation,
    # the scale the divergence is read against (it scales with the winds
    # as the divergence does, so the ratio is free of the factor r).
    ze = z - z.mean(axis=-1, keepdims=True)
    rms_z = float(np.sqrt(np.sum(w * z * z) / np.sum(w)))
    rms_ze = float(np.sqrt(np.sum(w * ze * ze) / np.sum(w)))
    return {
        "max_abs_div_s": float(np.max(np.abs(d))),
        "rms_div_s": rms_d,
        "rms_vort_s": rms_z,
        "rms_eddy_vort_s": rms_ze,
        "div_over_vort": rms_d / rms_z if rms_z > 0 else None,
        "finite": bool(np.all(np.isfinite(d))),
    }, None


def surface_pressure(model, bundle):
    g = model.grid_state(bundle.atmosphere, only=("ps",))
    return np.asarray(model.transform.backend.to_numpy(g["ps"]), dtype=np.float64)


def run_arm(name, cfg, transform, phis, cold, rotation, days, dt, out):
    model = build_model(cfg, transform, phis, rotation)
    state = cold
    state, *_ = model._repair_positivity(state)
    model.initialize_mass_target(state.atmosphere)
    model.enforce(state)
    steps = int(round(days * 86400.0 / dt))
    every = int(round(21600.0 / dt))
    three_h = int(round(10800.0 / dt))
    worst = 0.0
    rms_sum = 0.0
    rms_n = 0
    ratio_sum = 0.0
    lines = []
    t0 = time.time()
    try:
        ps_prev = surface_pressure(model, state)
    except Exception:  # noqa: BLE001
        ps_prev = None
    dps = []
    stats, _ = divergence_stats(model, state)
    stats.update(arm=name, hour=0.0)
    lines.append(stats)
    for k in range(1, steps + 1):
        state, _m = model.step(state, dt)
        if ps_prev is not None and k % three_h == 0:
            ps_now = surface_pressure(model, state)
            dps.append(float(np.mean(np.abs(ps_now - ps_prev))) / 100.0)
            ps_prev = ps_now
        if k % every == 0 or k == steps:
            stats, _ = divergence_stats(model, state)
            stats.update(arm=name, hour=k * dt / 3600.0)
            lines.append(stats)
            worst = max(worst, stats["max_abs_div_s"])
            rms_sum += stats["rms_div_s"]
            ratio_sum += stats["div_over_vort"] or 0.0
            rms_n += 1
            print(json.dumps(stats), flush=True)
            if not stats["finite"]:
                break
    summary = {
        "arm": name, "rotation_rate_s": rotation, "days": days, "dt_s": dt,
        "largest_max_abs_div_s": worst,
        "mean_rms_div_s": rms_sum / max(rms_n, 1),
        "mean_div_over_vort": ratio_sum / max(rms_n, 1),
        "start": lines[0],
        "mean_abs_3h_dps_hpa_first_day": float(np.mean(dps[:8])) if dps else None,
        "mean_abs_3h_dps_hpa_all": float(np.mean(dps)) if dps else None,
        "wall_s": time.time() - t0,
        "series": lines,
    }
    out.append(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "series"}), flush=True)
    return summary


# ------------------------------------------------------------------ era5

def _bilinear(values, src_lat, src_lon, dst_lat, dst_lon):
    """Bilinear from a regular lat-lon grid (any leading axes) onto a
    Gaussian grid; periodic in longitude."""
    lat = np.asarray(src_lat, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        vals = vals[..., ::-1, :]
    lon = np.asarray(src_lon, dtype=np.float64)
    dlon = lon[1] - lon[0]
    j = np.clip(np.searchsorted(lat, dst_lat) - 1, 0, lat.size - 2)
    wy = (dst_lat - lat[j]) / (lat[j + 1] - lat[j])
    x = ((dst_lon - lon[0]) % 360.0) / dlon
    i0 = np.floor(x).astype(int) % lon.size
    i1 = (i0 + 1) % lon.size
    wx = x - np.floor(x)
    a = vals[..., j, :]
    row = a * (1 - wy)[:, None] + vals[..., j + 1, :] * wy[:, None]
    return row[..., i0] * (1 - wx) + row[..., i1] * wx


def _remap(values, ln_src, ln_tgt, extrapolate_below=False):
    """Linear in ln p from source levels (ascending, top first) onto model
    levels; held above the top; below the bottom held, or continued with
    the bottom layer's gradient."""
    nl = ln_src.size
    idx = np.clip(np.searchsorted(ln_src, ln_tgt), 1, nl - 1)
    lo = np.take_along_axis(values, idx - 1, axis=0)
    hi = np.take_along_axis(values, idx, axis=0)
    w = (ln_tgt - ln_src[idx - 1]) / (ln_src[idx] - ln_src[idx - 1])
    if not extrapolate_below:
        w = np.clip(w, 0.0, 1.0)
    else:
        w = np.maximum(w, 0.0)
    return lo + w * (hi - lo)


def era5_start(cfg, transform, path, stride, factor):
    import netCDF4

    from arwen_global.constants import DRY_AIR_GAS_CONSTANT, KAPPA, REFERENCE_PRESSURE_PA
    from arwen_global.lnps_reference import LNPS_REFERENCE, lnps_from_anomaly_coefficients
    from arwen_global.spectral.vector import VorticityDivergenceOperator
    from arwen_global.state import ArwenGlobalState, MoistHybridState
    from arwen_global.testcases import _dry_surface
    from arwen_global.world.rotation_start import rotation_start_replacements

    d = netCDF4.Dataset(path)
    lat = np.asarray(d["latitude"][::stride], dtype=np.float64)
    lon = np.asarray(d["longitude"][::stride], dtype=np.float64)
    levels_hpa = np.asarray(d["level"][:], dtype=np.float64)
    order = np.argsort(levels_hpa)
    levels = levels_hpa[order] * 100.0

    def read(name, three_d):
        if three_d:
            a = np.asarray(d[name][0, :, ::stride, ::stride], dtype=np.float64)[order]
        else:
            a = np.asarray(d[name][0, ::stride, ::stride], dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"{name} carries missing values")
        return a

    temperature = read("T", True)
    u = read("U", True)
    v = read("V", True)
    ps_src_ll = read("PSFC", False)
    phis_ll = read("SOILGEO", False)
    d.close()
    frame = SimpleNamespace(
        latitude=lat, longitude=lon, vertical_values=levels,
        fields={"eastward_wind": SimpleNamespace(values=u),
                "northward_wind": SimpleNamespace(values=v),
                "air_temperature": SimpleNamespace(values=temperature),
                "surface_pressure": SimpleNamespace(values=ps_src_ll)})
    replacements, record = rotation_start_replacements(frame, factor)

    imbalance = {}
    try:
        from arwen_global.world.start import thermal_wind_imbalance
        from arwen_global.constants import EARTH_ROTATION_RATE_S as OM

        def shifted(uu, vv):
            return SimpleNamespace(latitude=lat, longitude=lon, fields={
                "air_temperature": frame.fields["air_temperature"],
                "eastward_wind": SimpleNamespace(values=uu),
                "northward_wind": SimpleNamespace(values=vv),
                "surface_pressure": frame.fields["surface_pressure"]})
        imbalance = {
            "analysis_at_earth": thermal_wind_imbalance(shifted(u, v), levels, rotation_rate_s=OM),
            "analysis_at_world": thermal_wind_imbalance(shifted(u, v), levels, rotation_rate_s=factor * OM),
            "rebalanced_at_world": thermal_wind_imbalance(
                shifted(replacements["eastward_wind"], replacements["northward_wind"]),
                levels, rotation_rate_s=factor * OM),
        }
        for value in imbalance.values():
            value.pop("levels_used_pa", None)
    except ImportError as exc:
        imbalance = {"not_measured": str(exc)}

    grid = transform.grid
    glat = np.asarray(grid.latitude_deg, dtype=np.float64)
    glon = np.asarray(grid.longitude_deg, dtype=np.float64)

    def to_grid(a):
        return _bilinear(a, lat, lon, glat, glon)

    backend = transform.backend
    phi_src = to_grid(phis_ll)
    phi_model = np.asarray(backend.to_numpy(transform.inverse(transform.forward(phi_src))))
    ps_src = to_grid(ps_src_ll)
    t_src = to_grid(temperature)
    ln_src = np.log(levels)
    t_surf = _remap(t_src, ln_src, np.log(ps_src)[None], extrapolate_below=True)[0]
    ps_model = ps_src * np.exp((phi_src - phi_model) / (DRY_AIR_GAS_CONSTANT * t_surf))
    p_full = np.asarray(backend.to_numpy(cfg.vertical.pressure(ps_model, backend)["p_full"]))
    ln_tgt = np.log(p_full)
    t_model = _remap(t_src, ln_src, ln_tgt, extrapolate_below=True)
    theta = t_model / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    vector = VorticityDivergenceOperator(transform)
    dtype = backend.float_dtype
    nlev = cfg.vertical.nlev
    zeros_grid = np.zeros((nlev, *grid.shape))
    zeros_spec = np.zeros((nlev, *transform.spectral_shape), dtype=backend.complex_dtype)
    lnps = lnps_from_anomaly_coefficients(
        transform, transform.forward(np.log(ps_model) - LNPS_REFERENCE))
    theta_spec = transform.forward(theta)

    def bundle_from(uu_ll, vv_ll):
        uu = _remap(to_grid(uu_ll), ln_src, ln_tgt)
        vv = _remap(to_grid(vv_ll), ln_src, ln_tgt)
        vort, div = vector.vordiv_from_wind(backend.asarray(uu, dtype=dtype),
                                            backend.asarray(vv, dtype=dtype))
        atmosphere = MoistHybridState(
            vorticity=vort, divergence=div, theta=theta_spec,
            log_surface_pressure=lnps, qv=zeros_spec.copy(),
            **{n: zeros_grid.copy() for n in ("qc", "qr", "qi", "qs", "qg",
                                              "nc", "nr", "ni", "ns", "ng")})
        return ArwenGlobalState(atmosphere, _dry_surface(transform, t_model[-1]))

    earth = bundle_from(u, v)
    rebalanced = bundle_from(replacements["eastward_wind"], replacements["northward_wind"])
    info = {"frame_record": record, "thermal_wind_imbalance": imbalance,
            "source": path, "stride": stride,
            "ps_model_range_pa": [float(ps_model.min()), float(ps_model.max())]}
    return phi_model, earth, rebalanced, info


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("era5")
    p.add_argument("out")
    p.add_argument("--case", default="both")
    p.add_argument("--days", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--truncation", type=int, default=42)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--arms", default="E1,E2,R2")
    a = p.parse_args(argv)

    from arwen_global.config import load_config
    from arwen_global.constants import EARTH_ROTATION_RATE_S as OM
    from arwen_global.spectral.transform import SphericalHarmonicTransform
    from arwen_global.testcases import baroclinic_wave_initial_state
    from arwen_global.world.rotation_start import rebalanced_state

    cfg = load_config(a.config)
    transform = SphericalHarmonicTransform.create(a.truncation, backend="numpy", precision="float64")
    arms = a.arms.split(",")
    results = {"config": a.config, "truncation": a.truncation, "factor": a.factor,
               "dt_s": a.dt, "days": a.days, "integrator": cfg.integrator, "cases": {}}
    cases = ["jw", "era5"] if a.case == "both" else [a.case]
    for case in cases:
        out = []
        info = {}
        if case == "jw":
            cold, phis, _prov = baroclinic_wave_initial_state(
                SimpleNamespace(vertical=cfg.vertical, perturbation_amplitude=0.0), transform)
            phis = np.asarray(transform.backend.to_numpy(phis))
            probe = build_model(cfg, transform, phis, OM)
            rebalanced, info = rebalanced_state(probe, cold, a.factor)
            earth = cold
        else:
            phis, earth, rebalanced, info = era5_start(cfg, transform, a.era5, a.stride, a.factor)
        print(json.dumps({"case": case, "info": info}, default=str), flush=True)
        if "E1" in arms:
            run_arm(f"{case}:E1", cfg, transform, phis, earth, OM, a.days, a.dt, out)
        if "E2" in arms:
            run_arm(f"{case}:E2", cfg, transform, phis, earth, a.factor * OM, a.days, a.dt, out)
        if "R2" in arms:
            run_arm(f"{case}:R2", cfg, transform, phis, rebalanced, a.factor * OM, a.days, a.dt, out)
        results["cases"][case] = {"info": info, "arms": out}
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
