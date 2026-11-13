"""Long-run operations (arwen_global.longrun_ops) on known answers.

* Retention keeps the newest N checkpoints and, under ``+month-ends``, the
  last checkpoint of every calendar month, 29 February 2028 included.
* A day's mean file is the numpy mean of that day's checkpoints, exactly.
* A restarted run writes the straight run's mean and summary files, bit
  for bit (the sums ride in a sidecar beside each checkpoint).
* A mean file is refused as a restart and labels an exported tape.
* Pruning during the segment keeps at most N durable checkpoints plus the
  one being written.
* The launch gate refuses when the estimate exceeds the free disk less
  the reserve; the checkpoint size is scaled from the one measured T383
  checkpoint.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from arwen_global.boundary_options import OutputOptions
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.longrun_ops import (
    MEASURED_CHECKPOINT_BYTES, checkpoint_bytes, checkpoints_to_prune, estimate,
    sidecar_of, time_mean_attrs,
)

SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def test_retention_keeps_the_newest_two_and_every_month_end_of_a_leap_year():
    start = datetime(2028, 1, 1, tzinfo=timezone.utc)
    dt = 6 * 3600.0
    steps = list(range(0, 366 * 4 + 1))   # 6-hourly through 1 January 2029 00Z
    gone = set(checkpoints_to_prune(steps, dt_s=dt, start=start, keep_newest=2,
                                    keep_month_ends=True))
    kept = sorted(set(steps) - gone)
    valid = [start + timedelta(seconds=s * dt) for s in kept]
    month_ends = [start + timedelta(seconds=s * dt) for s in steps[:-1]
                  if (start + timedelta(seconds=(s + 1) * dt)).month
                  != (start + timedelta(seconds=s * dt)).month]
    assert len(month_ends) == 12
    assert datetime(2028, 2, 29, 18, tzinfo=timezone.utc) in month_ends
    assert valid == sorted(set(month_ends) | {start + timedelta(seconds=s * dt)
                                               for s in steps[-2:]})
    newest = checkpoints_to_prune(steps, dt_s=dt, start=start, keep_newest=2,
                                  keep_month_ends=False)
    assert sorted(set(steps) - set(newest)) == steps[-2:]
    assert checkpoints_to_prune(steps[:5], dt_s=dt, start=start, keep_newest=2,
                                keep_month_ends=False, protected=[0]) == [1, 2]


def test_the_checkpoint_size_scales_from_the_measured_t383_archive():
    assert checkpoint_bytes(576, 1152, 40) == MEASURED_CHECKPOINT_BYTES
    # T255 (384 x 768) is 4/9 of the T383 grid: 0.677 GB.
    assert checkpoint_bytes(384, 768, 40) == int(MEASURED_CHECKPOINT_BYTES * 4 / 9)


def _dated_smoke(tmp_path, monkeypatch, start, output):
    import arwen_global.clock as clock

    from arwen_global.config import load_config

    monkeypatch.setattr(clock, "resolve_forecast_clock",
                        lambda cfg, prov: clock.ForecastClock(
                            start, clock.CONFIG_START_TIME, declared_start_time=start))
    cfg = load_config(SMOKE)
    return replace(cfg, duration_s=80.0, output_interval_s=10.0, output=output)


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: np.array(data[k]) for k in data.files if k != "__metadata__"}


def test_a_day_mean_is_the_mean_of_its_checkpoints_and_a_restart_reproduces_it(
        tmp_path, monkeypatch):
    from arwen_global.checkpoint import read_checkpoint
    from arwen_global.runner import run

    # 23:59:30: steps 1 to 2 are the 31st, steps 3 to 8 the 1st of January.
    start = datetime(2026, 12, 31, 23, 59, 30, tzinfo=timezone.utc)
    cfg = _dated_smoke(tmp_path, monkeypatch, start,
                       OutputOptions(mean_state="monthly+daily", daily_summary=True))
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = run(cfg, full)
    written = result["long_run_outputs"]["products"]["written"]
    assert "arwen_global_mean_2026-12-31.npz" in written
    assert "arwen_global_mean_2027-01-01.npz" in written
    assert "arwen_global_mean_2026-12.npz" in written
    assert sidecar_of(full / "arwen_global_step00000004.npz").is_file()
    meta, mean = read_checkpoint(full / "arwen_global_mean_2026-12-31.npz")
    assert meta["physics_metadata"]["time_mean"]["samples"] == 2
    members = [read_checkpoint(full / f"arwen_global_step{s:08d}.npz")[1] for s in (1, 2)]
    for name, value in mean.items():
        if name.startswith(("atmosphere__", "surface__")) and np.issubdtype(
                value.dtype, np.inexact):
            wide = np.complex128 if np.iscomplexobj(value) else np.float64
            expect = (members[0][name].astype(wide) + members[1][name].astype(wide)) / 2
            np.testing.assert_array_equal(value, expect.astype(value.dtype), err_msg=name)
    assert time_mean_attrs(meta)["ARWEN_TIME_MEAN"].startswith("{")
    # The daily summary's skin mean is the mean of the day's six output
    # instants (steps 3 to 8), summed in float64 in step order.
    day = _arrays(full / "arwen_global_daily_2027-01-01.npz")
    total = None
    for step in range(3, 9):
        skin = read_checkpoint(full / f"arwen_global_step{step:08d}.npz")[1][
            "surface__surface_temperature_k"].astype(np.float64)
        total = skin.copy() if total is None else total + skin
    np.testing.assert_array_equal(day["skin_temperature_mean_k"],
                                  (total / 6).astype(np.float32))
    assert "sea_ice_fraction_mean" in day
    # A mean file is not a restart.
    with pytest.raises(ValueError, match="time-mean"):
        run(cfg, tmp_path / "bad", restart=full / "arwen_global_mean_2026-12-31.npz")
    # Restarted from step 4, inside the second day's window: the same files.
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    for name in ("arwen_global_mean_2027-01-01.npz", "arwen_global_daily_2027-01-01.npz"):
        a, b = _arrays(full / name), _arrays(resumed / name)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name} {key}")


def test_pruning_during_the_segment_keeps_the_newest_two_plus_one_in_flight(
        tmp_path, monkeypatch):
    from arwen_global import longrun_ops
    from arwen_global.runner import run

    start = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
    cfg = _dated_smoke(tmp_path, monkeypatch, start,
                       OutputOptions(keep_checkpoints="newest:2"))
    seen = []
    original = longrun_ops.CheckpointRetention.prune

    def counted(self, in_flight=None):
        gone = original(self, in_flight)
        seen.append(len(list(self.outdir.glob("arwen_global_step*.npz"))))
        return gone

    monkeypatch.setattr(longrun_ops.CheckpointRetention, "prune", counted)
    out = tmp_path / "run"
    result = run(cfg, out)
    assert max(seen) <= 3
    left = sorted(p.name for p in out.glob("arwen_global_step*.npz"))
    assert left == ["arwen_global_step00000007.npz", "arwen_global_step00000008.npz"]
    assert result["long_run_outputs"]["retention"]["deleted"] == 7
    # Every pruned checkpoint left its header: the same step, time and
    # hashes as the twin run that kept all of them (a CPU run is
    # deterministic), in deletion order.
    retention = result["long_run_outputs"]["retention"]
    assert retention["pruned_record"] == longrun_ops.PRUNED_RECORD
    assert retention["pruned_records_written"] == 7
    lines = (out / longrun_ops.PRUNED_RECORD).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["step"] for r in records] == list(range(7))
    kept = tmp_path / "kept"
    run(replace(cfg, output=OutputOptions()), kept)
    for record in records:
        twin = kept / record["checkpoint"]
        with np.load(twin, allow_pickle=False) as archive:
            meta = json.loads(str(archive["__metadata__"].item()))
        assert record["self_sha256"] == meta["self_sha256"]
        assert record["array_sha256"] == {n: e["sha256"] for n, e in meta["arrays"].items()}
        assert record["time_s"] == meta["time_s"] and record["bytes"] == twin.stat().st_size
        assert record["config_hash"] == meta["config_hash"] == cfg.config_hash
        assert record["valid_utc"] == (start + timedelta(seconds=meta["time_s"])).isoformat(
        ).replace("+00:00", "Z")
    # An unreadable header is recorded as unreadable, not skipped.
    broken = tmp_path / "arwen_global_step00000099.npz"
    broken.write_bytes(b"not a checkpoint")
    header = longrun_ops.pruned_header(broken)
    assert header["checkpoint"] == broken.name and "unreadable" in header


def test_the_receipt_carries_the_software_stack(tmp_path):
    from arwen_global.config import load_config
    from arwen_global.receipt import software_stack
    from arwen_global.runner import run

    import numpy

    result = run(load_config(SMOKE), tmp_path / "run")
    stack = result["software_stack"]
    assert stack["backend"] == "numpy" and stack["device"] is None
    assert stack["libraries"]["numpy"] == numpy.__version__
    assert stack["python"] and stack["platform"]["system"]
    assert isinstance(stack["cuda_wheels"], dict)
    # On the card the device block names the card, CUDA and the driver; a
    # field it cannot read says so instead of raising.
    pytest.importorskip("cupy")
    device = software_stack("cupy")["device"]
    for key in ("cupy", "cuda_runtime", "cuda_driver_api", "nvrtc", "card",
                "nvidia_driver"):
        assert key in device


def test_the_launch_gate_refuses_when_the_disk_cannot_hold_the_run(tmp_path, monkeypatch):
    from arwen_global import longrun_ops
    from arwen_global.config import load_config

    cfg = replace(load_config(SMOKE), output=OutputOptions(keep_checkpoints="newest:2"))
    figure = estimate(cfg)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(longrun_ops.shutil, "disk_usage",
                        lambda path: usage._replace(free=int(20e9) + figure["disk_bytes"] - 1))
    with pytest.raises(ValueError, match="free"):
        longrun_ops.launch_gate(cfg, tmp_path, start_s=0.0, until_s=None)
    monkeypatch.setattr(longrun_ops.shutil, "disk_usage",
                        lambda path: usage._replace(free=int(20e9) + figure["disk_bytes"]))
    assert longrun_ops.launch_gate(cfg, tmp_path, start_s=0.0, until_s=None) is not None
    # A short run that states no long-run output is not gated.
    assert longrun_ops.launch_gate(load_config(SMOKE), tmp_path, start_s=0.0,
                                   until_s=None) is None


def test_a_t383_year_with_retention_is_estimated_from_its_parts():
    from arwen_global import longrun_ops
    from arwen_global.config import load_config

    cfg = load_config(str(_shipped_configs() / "arwen_global_gdas_t383_native_24h.toml"))
    year = replace(cfg, duration_s=365 * 86400.0, output_interval_s=6 * 3600.0,
                   output=OutputOptions(keep_checkpoints="newest:2", mean_state="monthly",
                                        daily_summary=True))
    figure = estimate(year)
    per = MEASURED_CHECKPOINT_BYTES
    assert figure["checkpoints_on_disk"] == 3
    planes = longrun_ops.DAILY_SUMMARY_PLANES
    summary = planes * 576 * 1152 * 4 * 366
    sidecar = 2 * per + planes * 576 * 1152 * 8
    hand = 3 * per + 3 * sidecar + 15 * per + summary
    assert abs(figure["disk_bytes"] - hand) <= 0.10 * hand
    assert figure["wall_s"] == pytest.approx(365 * 12.0 * 60.0)


# ------------------------------------------------- the daily summary's planes


def _summary_geometry(truncation=21, nlev=10):
    from arwen_global.longrun_ops import SummaryGeometry
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    transform = SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64", streaming=True)
    eta = np.linspace(0.0, 1.0, nlev + 1)
    b = eta ** 1.7
    a = 100.0 * (1.0 - b)
    a[-1], b[-1] = 0.0, 1.0
    return SummaryGeometry(transform, a, b, np.zeros(transform.grid.shape))


def _summary_products(tmp_path, geometry, start):
    from arwen_global.longrun_ops import LongRunProducts

    return LongRunProducts(
        tmp_path, OutputOptions(daily_summary=True), start=start, dt_s=21600.0,
        config_hash="0" * 64, physics_state_schema="x", semi_implicit_scheme="x",
        integrator="x", geometry=geometry)


def _feed(products, instants):
    """``instants``: (time_s, arrays, physics_metadata) in time order."""
    for step, (time_s, arrays, physics_metadata) in enumerate(instants, start=1):
        products.add(arrays, {"step": step, "time_s": time_s,
                              "physics_metadata": physics_metadata,
                              "trackers": {}, "dtypes": {}})
    products.finish(run_complete=True)


def _lnps(geometry, ps):
    return geometry.transform.forward(np.log(np.broadcast_to(ps, geometry.grid.shape)))


def test_the_daily_surface_stress_is_the_mean_of_rho_ustar_squared_along_the_wind(tmp_path):
    """Two instants a day with u* of 0.2 and 0.4 m/s under a 3, -4 m/s wind:
    the day's stress is rho <u*^2> = rho 0.1 along (3, -4)/5, the mean of
    the products and not the product of the means (rho 0.09), positive
    along the wind; a calm column carries none."""
    from arwen_global.constants import DRY_AIR_GAS_CONSTANT, EPSILON
    from arwen_global.longrun_ops import SURFACE_STRESS_NAMES

    geometry = _summary_geometry()
    shape = geometry.grid.shape
    start = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
    u10 = np.full(shape, 3.0)
    u10[0, 0] = 0.0
    v10 = np.full(shape, -4.0)
    v10[0, 0] = 0.0

    def arrays(ust):
        return {"atmosphere__log_surface_pressure": _lnps(geometry, 1.0e5),
                "physics__ust": np.full(shape, ust), "physics__u10": u10,
                "physics__v10": v10, "physics__t2": np.full(shape, 290.0),
                "physics__q2": np.full(shape, 0.01)}

    products = _summary_products(tmp_path, geometry, start)
    _feed(products, [(21600.0, arrays(0.2), {}), (43200.0, arrays(0.4), {})])
    day = _arrays(tmp_path / "arwen_global_daily_2026-11-11.npz")
    rho = 1.0e5 / (DRY_AIR_GAS_CONSTANT * 290.0 * (1.0 + (1.0 / EPSILON - 1.0) * 0.01))
    east, north = (day[name] for name in SURFACE_STRESS_NAMES)
    assert east.dtype == np.float32
    np.testing.assert_allclose(east[1:], rho * 0.1 * 3.0 / 5.0, rtol=1e-6)
    np.testing.assert_allclose(north[1:], -rho * 0.1 * 4.0 / 5.0, rtol=1e-6)
    assert east[0, 0] == 0.0 and north[0, 0] == 0.0
    meta = json.loads(str(np.load(tmp_path / "arwen_global_daily_2026-11-11.npz")["metadata"]))
    assert "positive eastward" in meta["rules"][SURFACE_STRESS_NAMES[0]]
    assert meta["derived_planes"] == "on"
    # Without a host synthesis the stress is not made (and not zero-filled).
    bare = _summary_products(tmp_path / "bare", None, start)
    (tmp_path / "bare").mkdir()
    _feed(bare, [(21600.0, arrays(0.2), {})])
    assert SURFACE_STRESS_NAMES[0] not in _arrays(tmp_path / "bare" / "arwen_global_daily_2026-11-11.npz")


def test_sea_level_pressure_and_850_hpa_vorticity_are_read_back_in_closed_form(tmp_path):
    """Vorticity (a + b ln p) sin(lat) on every level under a uniform
    1000 hPa surface: the 850 hPa plane is (a + b ln 85000) sin(lat) to
    rounding (linear in ln p is exact), its cyclonic maximum |that|, and no
    column is below ground.  Sea level pressure equals ps over zero terrain
    and the reduction's closed form over 1000 m; a 800 hPa surface puts
    850 hPa under ground everywhere and holds the lowest level's value."""
    from arwen_global import longrun_ops as lo
    from arwen_global.constants import GRAVITY_M_S2

    geometry = _summary_geometry()
    grid = geometry.grid
    terrain = np.zeros(grid.shape)
    terrain[: grid.nlat // 2] = 1000.0
    geometry.surface_geopotential = terrain * GRAVITY_M_S2
    sin = np.sin(np.deg2rad(np.asarray(grid.latitude_deg)))[:, None] * np.ones(grid.shape)
    a, b = 3.0e-5, 2.0e-6

    def vorticity(ps):
        p_half = geometry.a_half + geometry.b_half * ps
        p_full = np.sqrt(p_half[:-1] * p_half[1:])
        return np.stack([geometry.transform.forward((a + b * math.log(p)) * sin)
                         for p in p_full])

    def arrays(ps):
        return {"atmosphere__log_surface_pressure": _lnps(geometry, ps),
                "atmosphere__vorticity": vorticity(ps),
                "physics__t2": np.full(grid.shape, 280.0)}

    start = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
    products = _summary_products(tmp_path, geometry, start)
    _feed(products, [(21600.0, arrays(1.0e5), {}), (43200.0, arrays(1.0e5), {})])
    day = _arrays(tmp_path / "arwen_global_daily_2026-11-11.npz")
    expected = (a + b * math.log(85000.0)) * sin
    np.testing.assert_allclose(day[lo.VORTICITY_MEAN], expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(day[lo.VORTICITY_CYCLONIC_MAX], np.abs(expected),
                               rtol=1e-6, atol=1e-12)
    assert not day[lo.BELOW_GROUND].any()
    tmean = 280.0 + 0.0065 * 1000.0 / 2.0
    reduced = 1.0e5 * math.exp(9.80665 * 1000.0 / (287.05 * tmean))
    np.testing.assert_allclose(day[lo.SLP_MEAN][grid.nlat // 2:], 1.0e5, rtol=1e-6)
    np.testing.assert_allclose(day[lo.SLP_MEAN][: grid.nlat // 2], reduced, rtol=1e-6)
    np.testing.assert_array_equal(day[lo.SLP_MIN], day[lo.SLP_MEAN])
    # 800 hPa surface: 850 hPa is under every column; the lowest level holds.
    low = tmp_path / "low"
    low.mkdir()
    products = _summary_products(low, geometry, start)
    _feed(products, [(21600.0, arrays(8.0e4), {})])
    day = _arrays(low / "arwen_global_daily_2026-11-11.npz")
    p_half = geometry.a_half[-2:] * 0.0 + geometry.b_half[-2:] * 8.0e4 + geometry.a_half[-2:]
    lowest = math.sqrt(p_half[0] * p_half[1])
    np.testing.assert_allclose(day[lo.VORTICITY_MEAN], (a + b * math.log(lowest)) * sin,
                               rtol=1e-6, atol=1e-12)
    assert day[lo.BELOW_GROUND].all()


def _energy_instant(geometry, time_s):
    """Fixed fluxes whose accumulators grow linearly with the radiation
    seconds (here the model time), and 2.9 mm/day of rain."""
    shape = geometry.grid.shape
    fluxes = {"acc_sw_down_top_j_m2": 340.0, "acc_sw_up_top_j_m2": 100.0,
              "acc_lw_up_top_j_m2": 239.0, "acc_sw_down_surface_j_m2": 180.0,
              "acc_sw_up_surface_j_m2": 20.0, "acc_lw_down_surface_j_m2": 330.0,
              "acc_lw_up_surface_j_m2": 390.0, "acc_total_cloud_cover_s": 0.6}
    arrays = {f"physics__{name}": np.full(shape, value * time_s)
              for name, value in fluxes.items()}
    nlev = len(geometry.a_half) - 1
    arrays.update({
        "physics__hfx": np.full(shape, 20.0), "physics__qfx": np.full(shape, 3.0e-5),
        "atmosphere__log_surface_pressure": _lnps(geometry, 1.0e5),
        "atmosphere__qc": np.full((nlev, *shape), 1.0e-4),
        "atmosphere__qi": np.full((nlev, *shape), 2.0e-5),
        "surface__accumulated_rain_kg_m2": np.full(shape, 2.9 * time_s / 86400.0),
        "surface__accumulated_snow_kg_m2": np.zeros(shape),
        "surface__accumulated_graupel_kg_m2": np.zeros(shape),
    })
    return time_s, arrays, {"radiation_accumulated_s": time_s}


def test_the_daily_energy_terms_are_exact_interval_means_and_survive_a_restart(tmp_path):
    """Day two (the day before is in the record): every radiation stream is
    its fixed flux to rounding, TOA absorbed 240 and net 1, surface net 100,
    cloud cover 0.6; H 20 and LE L_v 3e-5 = 75; the water paths are q
    (ps - p_top)/g; the global means give the albedo 100/340, the surface's
    net energy 100 - 20 - 75 = 5 W/m2 and 2.9 mm/day.  Day one has no
    radiation interval and says so.  A summary resumed from a sidecar
    written mid-day writes the straight summary's file, bit for bit."""
    from arwen_global import longrun_ops as lo
    from arwen_global.constants import GRAVITY_M_S2

    geometry = _summary_geometry()
    start = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
    instants = [_energy_instant(geometry, h * 3600.0) for h in (6, 24, 30, 42, 48)]
    products = _summary_products(tmp_path, geometry, start)
    _feed(products, instants)
    day = _arrays(tmp_path / "arwen_global_daily_2026-11-12.npz")
    expect = {"toa_incoming_shortwave_w_m2": 340.0, "toa_reflected_shortwave_w_m2": 100.0,
              "toa_outgoing_longwave_w_m2": 239.0, lo.TOA_ABSORBED: 240.0, lo.TOA_NET: 1.0,
              "surface_downward_shortwave_w_m2": 180.0,
              "surface_upward_shortwave_w_m2": 20.0,
              "surface_downward_longwave_w_m2": 330.0,
              "surface_upward_longwave_w_m2": 390.0, lo.SURFACE_NET_RADIATION: 100.0,
              "cloud_cover_fraction": 0.6, lo.SENSIBLE: 20.0, lo.LATENT: 75.0,
              lo.LIQUID_WATER_PATH: 1.0e-4 * (1.0e5 - 100.0) / GRAVITY_M_S2,
              lo.ICE_WATER_PATH: 2.0e-5 * (1.0e5 - 100.0) / GRAVITY_M_S2,
              "precipitation_kg_m2": 2.9}
    for name, value in expect.items():
        np.testing.assert_allclose(day[name], value, rtol=2e-6, atol=1e-6, err_msg=name)
    with np.load(tmp_path / "arwen_global_daily_2026-11-12.npz") as data:
        meta = json.loads(str(data["metadata"]))
    assert meta["radiation_interval"] == {"from_time_s": 86400.0, "to_time_s": 172800.0,
                                          "seconds": 86400.0}
    means = meta["global_means"]
    assert means["planetary_albedo"] == pytest.approx(100.0 / 340.0, rel=1e-6)
    assert means["surface_net_energy_w_m2"] == pytest.approx(5.0, abs=1e-4)
    assert means["precipitation_mm_day"] == pytest.approx(2.9, rel=1e-6)
    assert means[lo.TOA_NET] == pytest.approx(1.0, abs=1e-5)
    assert "clear_sky_radiation" in meta["not_written"]
    assert set(meta["rules"]) >= set(expect) - {"precipitation_kg_m2"}
    first = _arrays(tmp_path / "arwen_global_daily_2026-11-11.npz")
    assert lo.TOA_NET not in first and lo.SENSIBLE in first
    # Restart: a sidecar written after the third instant (inside day two).
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    head = _summary_products(resumed, geometry, start)
    for step, (time_s, arrays, physics) in enumerate(instants[:3], start=1):
        head.add(arrays, {"step": step, "time_s": time_s, "physics_metadata": physics,
                          "trackers": {}, "dtypes": {}})
    checkpoint = resumed / "arwen_global_step00000003.npz"
    head.write_sidecar(checkpoint)
    tail = _summary_products(resumed, geometry, start)
    assert tail.resume(checkpoint).startswith("resumed")
    for step, (time_s, arrays, physics) in enumerate(instants[3:], start=4):
        tail.add(arrays, {"step": step, "time_s": time_s, "physics_metadata": physics,
                          "trackers": {}, "dtypes": {}})
    tail.finish(run_complete=True)
    a = _arrays(tmp_path / "arwen_global_daily_2026-11-12.npz")
    b = _arrays(resumed / "arwen_global_daily_2026-11-12.npz")
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
