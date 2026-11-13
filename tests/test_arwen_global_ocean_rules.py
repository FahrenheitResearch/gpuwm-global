"""The ocean and sea-ice rules (arwen_global.ocean_rules) on known answers.

Each rule is a closed form in model time, so each test writes down the
number the formula gives and asks the provider, or the boundary service
on a state, for it:

* anomaly: C_s(t) + Abar + A' exp(-r t), with r the extratropical rate
  poleward of 30 degrees, zero on the equator in the Pacific and the
  extratropical rate on the equator in the Indian Ocean, and half the
  extratropical rate at 20 degrees in the Pacific (the half-cosine ramp);
* climatology: within (|Abar| + |A'|) e^-10 of C_s(t) at 30 days with a
  3-day ramp;
* hold: nothing written;
* the T_f floor, the zero anomaly of a column that starts as ice, the
  asymmetric ice clip, the warm-water ice rule and the new-ice thickness;
* the T42 truncation of A' keeps a degree-10 harmonic and removes a
  degree-50 one;
* a restarted run reproduces the straight run bit for bit, from an 18 UTC
  start, across a mid-month instant and an ice record;
* the config identity, the long-run refusal and the restart identity block.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.boundary_options import (
    LONG_RUN_THRESHOLD_S, OceanOptions, SeaIceOptions, long_run_refusal, parse_ocean,
)
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.ocean_rules import (
    RuleProvider, check_restart_identity, damping_rate, truncate,
)
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.surface_climatology import (
    interpolate, load_climatology, write_climatology,
)

T_F = 271.35
DAY = 86400.0
START = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
GDAS = str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")


def _climatology(tmp_path, grid, *, sst=None, ice=None, ocean=None, name="clim.npz"):
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    months = np.arange(12, dtype=np.float64)[:, None, None]
    if sst is None:
        sst = 280.0 + months + np.zeros((12, nlat, nlon))
    if ice is None:
        ice = np.zeros((12, nlat, nlon))
    if ocean is None:
        ocean = np.ones((nlat, nlon), dtype=bool)
    sst = np.where(ocean[None], sst, np.nan)
    ice = np.where(ocean[None], ice, np.nan)
    path = tmp_path / name
    write_climatology(path, {"sst_k": sst, "sea_ice_fraction": ice}, ocean=ocean,
                      latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
                      provenance={"schema": "gpuwm.arwen-global-surface-climatology/v1",
                                  "test": True})
    return load_climatology(path)


def _options(mode="anomaly", **kw):
    ice = kw.pop("sea_ice", SeaIceOptions())
    base = dict(declared=True, implicit=False, mode=mode, anomaly_reference="start",
                anomaly_truncation=None, climatology="x", climatology_sha256="0" * 64)
    base.update(kw)
    return OceanOptions(sea_ice=ice, **base)


def _cold(grid, skin, *, land=None, ice=None, lake=None):
    shape = (int(grid.nlat), int(grid.nlon))
    zeros = np.zeros(shape)
    return {"temperature_k": skin, "land_fraction": zeros if land is None else land,
            "lake_fraction": zeros if lake is None else lake,
            "sea_ice_fraction": zeros if ice is None else ice,
            "sea_ice_thickness_m": zeros.copy()}


def _area_mean(grid, plane, mask):
    w = np.asarray(grid.quadrature_weights)[:, None] * np.ones((1, int(grid.nlon)))
    return float((plane * w * mask).sum() / (w * mask).sum())


def test_the_anomaly_rule_is_its_closed_form(tmp_path):
    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid)
    lat, lon = np.meshgrid(grid.latitude_deg, grid.longitude_deg, indexing="ij")
    c0 = interpolate(clim.sst_k, START).astype(np.float64)
    a0 = 0.4 + 1.5 * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    provider = RuleProvider(_options(), clim, grid, start_time=START,
                            cold_surface=_cold(grid, c0 + a0), model_truncation=21)
    ocean = np.ones(lat.shape, dtype=bool)
    abar = _area_mean(grid, a0, ocean)
    assert provider.stats["abar_k"] == pytest.approx(abar, abs=1e-12)
    residual = a0 - abar
    t = 45.0 * DAY
    c_t = interpolate(clim.sst_k, START + timedelta(seconds=t)).astype(np.float64)
    sst, defined = provider.sst_at(t, np, np.float64)
    assert defined.all()
    # Poleward of 30 degrees: damped at 90 days.
    i = int(np.argmin(np.abs(grid.latitude_deg - 50.0)))
    expect = c_t[i] + abar + residual[i] * math.exp(-45.0 / 90.0)
    np.testing.assert_allclose(sst[i], expect, rtol=0, atol=1e-9)
    # On the equator in the Pacific (120 E to 80 W): undamped.
    e = int(np.argmin(np.abs(grid.latitude_deg)))
    assert abs(grid.latitude_deg[e]) < 10.0
    pacific = (lon[e] >= 120.0) & (lon[e] <= 280.0)
    np.testing.assert_allclose(sst[e, pacific], c_t[e, pacific] + abar + residual[e, pacific],
                               rtol=0, atol=1e-9)
    # On the equator in the Indian Ocean (70 E): damped at 90 days.
    j = int(np.argmin(np.abs(grid.longitude_deg - 70.0)))
    np.testing.assert_allclose(
        sst[e, j], c_t[e, j] + abar + residual[e, j] * math.exp(-0.5), rtol=0, atol=1e-9)


def test_the_tropical_ramp_halves_the_rate_at_20_degrees_in_the_pacific():
    options = _options()
    r = damping_rate(np.array([20.0, -20.0, 0.0, 30.0]), np.array([200.0, 70.0]), options)
    r_x = 1.0 / (90.0 * DAY)
    np.testing.assert_allclose(r[:2, 0], 0.5 * r_x, rtol=1e-12)
    np.testing.assert_allclose(r[:2, 1], r_x, rtol=1e-12)
    assert r[2, 0] == 0.0 and r[2, 1] == pytest.approx(r_x, rel=1e-12)
    np.testing.assert_allclose(r[3], r_x, rtol=1e-12)
    finite = damping_rate(np.array([0.0]), np.array([200.0]),
                          replace(options, tropical_anomaly_damping_days=180.0))
    assert finite[0, 0] == pytest.approx(1.0 / (180.0 * DAY), rel=1e-12)


def test_the_climatology_rule_has_lost_the_anomaly_after_ten_ramps(tmp_path):
    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid)
    c0 = interpolate(clim.sst_k, START).astype(np.float64)
    lat, lon = np.meshgrid(grid.latitude_deg, grid.longitude_deg, indexing="ij")
    a0 = 2.0 + np.sin(np.radians(lon))
    provider = RuleProvider(_options("climatology"), clim, grid, start_time=START,
                            cold_surface=_cold(grid, c0 + a0), model_truncation=21)
    t = 30.0 * DAY
    c_t = interpolate(clim.sst_k, START + timedelta(seconds=t)).astype(np.float64)
    sst, _ = provider.sst_at(t, np, np.float64)
    bound = (abs(provider.stats["abar_k"]) + np.abs(provider.residual).max()) * math.exp(-10.0)
    assert np.abs(sst - c_t).max() <= bound + 1e-12


def test_hold_writes_nothing_and_the_floor_holds(tmp_path):
    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid, sst=272.0 + np.zeros((12, *S)))
    skin = np.full(S, 265.0)  # 7 K below the climatology: the floor binds
    hold = RuleProvider(_options("hold"), clim, grid, start_time=START,
                        cold_surface=_cold(grid, skin), model_truncation=21)
    assert hold.sst_at(DAY, np, np.float32) == (None, None)
    anomaly = RuleProvider(_options(), clim, grid, start_time=START,
                           cold_surface=_cold(grid, skin), model_truncation=21)
    for t in (0.0, 10.0 * DAY, 200.0 * DAY):
        sst, _ = anomaly.sst_at(t, np, np.float64)
        assert sst.min() >= T_F


def test_a_column_that_starts_as_ice_takes_the_climatology_when_it_melts(tmp_path):
    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    ice = np.zeros((12, *S))
    ice[:, :4, :] = 0.9  # a southern cap in the climatology, all year
    clim = _climatology(tmp_path, grid, ice=ice)
    c0 = interpolate(clim.sst_k, START).astype(np.float64)
    start_ice = np.zeros(S)
    start_ice[:6, :] = 0.95  # two rows more ice at the start than the climatology
    skin = np.where(start_ice >= 0.5, 250.0, c0 + 1.25)
    provider = RuleProvider(_options(), clim, grid, start_time=START,
                            cold_surface=_cold(grid, skin, ice=start_ice), model_truncation=21)
    assert (provider.offset[:6] == 0.0).all() and (provider.residual[:6] == 0.0).all()
    t = 30.0 * DAY
    c_t = interpolate(clim.sst_k, START + timedelta(seconds=t)).astype(np.float64)
    sst = provider.sst_host(t)
    np.testing.assert_allclose(sst[4:6], c_t[4:6], rtol=0, atol=1e-12)
    assert (sst[4:6] > T_F).all()


def test_the_ice_rule_clips_asymmetrically_removes_warm_ice_and_seeds_new_ice(tmp_path):
    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    ice = np.zeros((12, *S))
    ice[:, 0, :] = 1.0          # climatology 1: a positive anomaly vanishes
    ice[:, 1, :] = 0.0          # climatology 0: a negative one vanishes
    ice[:, 2, :] = 0.9
    sst = np.full((12, *S), T_F + 0.5)   # cold water, the warm rule idle
    sst[:, 3, :] = T_F + 3.0                  # warm water: ice removed
    clim = _climatology(tmp_path, grid, sst=sst, ice=ice)
    start = np.zeros(S)
    start[0], start[1], start[2], start[3] = 0.6, 0.3, 0.2, 0.9
    options = _options("hold", sea_ice=SeaIceOptions(mode="anomaly"))
    provider = RuleProvider(options, clim, grid, start_time=START,
                            cold_surface=_cold(grid, np.full(S, T_F + 0.5),
                                               ice=start), model_truncation=21)
    k = 60
    t = k * provider.record_s
    decay = math.exp(-t / (60.0 * DAY))
    fraction, thickness, new = provider.ice_host(k, start, np.full(S, 2.0))
    np.testing.assert_allclose(fraction[0], min(1.0, 1.0 + (0.6 - 1.0) * decay), atol=1e-12)
    np.testing.assert_allclose(fraction[1], max(0.0, 0.0 + 0.3 * decay), atol=1e-12)
    assert (fraction[1] > 0.0).all()          # survives over C_i = 0 on cold water
    np.testing.assert_allclose(fraction[2], 0.9 + (0.2 - 0.9) * decay, atol=1e-12)
    assert (fraction[3] == 0.0).all()          # warm water keeps no ice
    # Row 2 freezes over (0.2 -> above one half): the new-ice prior.
    frozen_now = fraction[2] >= 0.5
    assert frozen_now.all() and new == int(frozen_now.sum())
    assert (thickness[2] == 1.0).all() and (thickness[0] == 2.0).all()
    # A record before any decay is the start's own ice.
    zero, _, _ = provider.ice_host(0, start, np.zeros(S))
    np.testing.assert_allclose(zero[:3], start[:3], atol=1e-12)


def test_the_t42_truncation_keeps_degree_10_and_removes_degree_50():
    grid = GaussianGrid.create(63)
    lat, lon = np.meshgrid(np.radians(grid.latitude_deg), np.radians(grid.longitude_deg),
                           indexing="ij")
    low = np.cos(lat) ** 10 * np.cos(10 * lon)
    high = np.cos(lat) ** 50 * np.cos(50 * lon)
    out = truncate(low + high, grid, 42, 63)
    np.testing.assert_allclose(out, low, atol=1e-9)
    assert truncate(high, grid, None, 63) is not None
    np.testing.assert_array_equal(truncate(high, grid, 63, 63), high)


def test_lakes_are_never_written_by_the_rule(tmp_path):
    from arwen_global.surface_boundary import SurfaceBoundary

    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid, ice=np.full((12, *S), 0.8))
    lake = np.zeros(S)
    lake[16, 10:14] = 0.9
    land = np.zeros(S)
    c0 = interpolate(clim.sst_k, START).astype(np.float64)
    skin = c0 + 0.5
    skin[16, 10:14] = 290.0
    provider = RuleProvider(_options(sea_ice=SeaIceOptions(mode="anomaly")), clim, grid,
                            start_time=START, cold_surface=_cold(grid, skin, lake=lake),
                            model_truncation=21)
    surface = SimpleNamespace(
        temperature_k=skin.astype(np.float32), land_fraction=land.astype(np.float32),
        lake_fraction=lake.astype(np.float32),
        sea_ice_fraction=np.zeros(S, np.float32),
        sea_ice_thickness_m=np.zeros(S, np.float32))
    state = SimpleNamespace(surface=surface, time_s=5.0 * DAY)
    SurfaceBoundary(provider).apply(state)
    assert (surface.temperature_k[16, 10:14] == np.float32(290.0)).all()
    assert (surface.sea_ice_fraction[16, 10:14] == 0.0).all()
    assert (surface.temperature_k[0] != skin[0].astype(np.float32)).any()


def test_identity_moves_only_when_arithmetic_does(tmp_path):
    from arwen_global.config import load_config

    clim = tmp_path / "c.npz"
    clim.write_bytes(b"climatology-bytes")
    moved = tmp_path / "elsewhere" / "c.npz"
    moved.parent.mkdir()
    shutil.copy(clim, moved)
    text = Path(GDAS).read_text(encoding="utf-8")

    def cfg(extra, name="a.toml"):
        path = tmp_path / name
        path.write_text(text + "\n" + extra, encoding="utf-8")
        return load_config(path)

    base = cfg("").config_hash
    assert cfg('[ocean]\nmode = "hold"\n').config_hash == base
    anomaly = cfg(f'[ocean]\nmode = "anomaly"\nclimatology = "{clim.as_posix()}"\n'
                  'anomaly_reference = "start"\n')
    assert anomaly.config_hash != base
    assert cfg(f'[ocean]\nmode = "anomaly"\nclimatology = "{moved.as_posix()}"\n'
               'anomaly_reference = "start"\n', "b.toml").config_hash == anomaly.config_hash
    for key, value in (("anomaly_damping_days", "60.0"),
                       ("tropical_anomaly_damping_days", "180.0"),
                       ("tropical_persist_region", '"global"'),
                       ("large_scale_offset", '"damp"'),
                       ("anomaly_truncation", '"none"')):
        other = cfg(f'[ocean]\nmode = "anomaly"\nclimatology = "{clim.as_posix()}"\n'
                    f'anomaly_reference = "start"\n{key} = {value}\n', "c.toml")
        assert other.config_hash != anomaly.config_hash, key
    with pytest.raises(ValueError, match="slab"):
        cfg('[ocean]\nmode = "slab"\n')
    with pytest.raises(ValueError, match="thermodynamic"):
        cfg('[ocean]\nmode = "hold"\n[sea_ice]\nmode = "thermodynamic"\n')
    with pytest.raises(ValueError, match="anomaly_reference_series"):
        cfg(f'[ocean]\nmode = "anomaly"\nclimatology = "{clim.as_posix()}"\n')
    # The greenness key joins only where it says what the mode would not do.
    assert cfg('[land]\ngreenness = "start"\n').config_hash == base
    assert cfg('[land]\ngreenness = "daily"\n').config_hash != base
    # [output] is an instrument.
    assert cfg('[output]\nkeep_checkpoints = "newest:2+month-ends"\nmean_state = "monthly"\n'
               'daily_summary = true\n').config_hash == base


def test_a_long_dated_run_with_no_ocean_rule_is_refused(tmp_path):
    from arwen_global.config import load_config

    text = Path(GDAS).read_text(encoding="utf-8")
    path = tmp_path / "long.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    long = replace(cfg, duration_s=LONG_RUN_THRESHOLD_S + 3600.0)
    message = long_run_refusal(long)
    assert message is not None
    for mode in ("anomaly", "climatology", "series", "hold"):
        assert f'"{mode}"' in message
    assert long_run_refusal(replace(cfg, duration_s=LONG_RUN_THRESHOLD_S)) is None
    stated = replace(long, ocean=replace(long.ocean, implicit=False))
    assert long_run_refusal(stated) is None
    smoke = load_config(SMOKE)
    assert long_run_refusal(replace(smoke, duration_s=LONG_RUN_THRESHOLD_S + 10.0)) is None


def test_a_restart_under_another_boundary_is_refused():
    block = {"ocean_mode": "anomaly", "sea_ice_mode": "anomaly",
             "climatology_sha256": "a" * 64, "start_utc": "2026-11-10T18:00:00Z",
             "start_anomaly_sha256": "b" * 64}
    check_restart_identity(block, {"surface_boundary": dict(block)}, "x")
    check_restart_identity(None, {}, "x")
    for key, value in (("ocean_mode", "climatology"), ("climatology_sha256", "c" * 64),
                       ("start_anomaly_sha256", "d" * 64)):
        with pytest.raises(ValueError, match="another ocean boundary"):
            check_restart_identity({**block, key: value}, {"surface_boundary": block}, "x")
    with pytest.raises(ValueError, match="another ocean boundary"):
        check_restart_identity(block, {}, "x")


def _smoke_rule_config(tmp_path, monkeypatch, start, *, mode="anomaly"):
    """The reference-physics smoke on a dated clock with an ocean rule.

    The smoke is an analytic planet with no clock, so the clock is given
    to it (the run's own resolver, patched), and the options are set on
    the loaded config: the parser admits a rule only on a run that has an
    analysis or real statics, which the T3 smoke has neither of."""
    import arwen_global.clock as clock

    from arwen_global.config import load_config

    real = clock.resolve_forecast_clock

    def dated(cfg, provenance):
        return clock.ForecastClock(start, clock.CONFIG_START_TIME, declared_start_time=start)

    monkeypatch.setattr(clock, "resolve_forecast_clock", dated)
    cfg = load_config(SMOKE)
    grid = GaussianGrid.create(int(cfg.truncation), dealias_factor=float(cfg.dealias_factor))
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    lat = np.asarray(grid.latitude_deg)[None, :, None]
    months = np.arange(12)[:, None, None]
    sst = 282.0 + 6.0 * np.cos(np.radians(lat)) + 0.8 * months + np.zeros((12, nlat, nlon))
    ice = np.clip((np.abs(lat) - 40.0) / 30.0, 0.0, 1.0) + np.zeros((12, nlat, nlon))
    clim = _climatology(tmp_path, grid, sst=sst, ice=ice)
    options = OceanOptions(
        declared=True, implicit=False, mode=mode, climatology=str(clim.path),
        climatology_sha256=clim.sha256, anomaly_reference="start", anomaly_truncation=None,
        sea_ice=SeaIceOptions(mode="match", record_interval_hours=30.0 / 3600.0))
    assert real is not None
    return replace(cfg, ocean=options, duration_s=80.0, output_interval_s=40.0)


def test_a_restarted_rule_run_reproduces_the_straight_run_bit_for_bit(tmp_path, monkeypatch):
    from arwen_global.checkpoint import read_checkpoint
    from arwen_global.runner import run

    # 11:59:20 on 16 January is 40 s before January's mid-month instant
    # (12 UTC on the 16th): the restart at step 4 sits on it, inside the ice
    # record that began at 30 s, and the run starts at an hour that is not 00.
    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    cfg = _smoke_rule_config(tmp_path, monkeypatch, start)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = run(cfg, full)
    assert result["surface_boundary"]["ocean_mode"] == "anomaly"
    assert result["surface_update"]["sst_steps"] == 8
    assert result["surface_update"]["ice_records"] >= 3
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000008.npz")
    meta_b, arrays_b = read_checkpoint(resumed / "arwen_global_step00000008.npz")
    assert meta_a["physics_metadata"]["surface_boundary"] == \
        meta_b["physics_metadata"]["surface_boundary"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    # The rule moved the skin: the boundary is not a held one.
    _meta0, arrays_0 = read_checkpoint(full / "arwen_global_step00000000.npz")
    assert not np.array_equal(arrays_0["surface__surface_temperature_k"],
                              arrays_a["surface__surface_temperature_k"])
    rows = [line for line in (full / "diagnostics.jsonl").read_text().splitlines() if line]
    assert any('"boundary_max_abs_sst_departure_k"' in line for line in rows)
    # A restart under another rule is refused before it runs.
    other = replace(cfg, ocean=replace(cfg.ocean, mode="climatology"))
    with pytest.raises(ValueError):
        run(other, tmp_path / "other", restart=full / "arwen_global_step00000004.npz")


def test_a_start_sst_below_the_freezing_point_is_not_an_anomaly(tmp_path):
    grid = GaussianGrid.create(21)
    S = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid, sst=272.0 + np.zeros((12, *S)))
    skin = np.full(S, 272.5)
    skin[-2:, :] = 241.0   # an ice skin on open-water columns (GDAS 2026-11-10)
    provider = RuleProvider(_options(), clim, grid, start_time=START,
                            cold_surface=_cold(grid, skin), model_truncation=21)
    assert provider.stats["columns_whose_start_sst_was_raised_to_t_f"] == 2 * S[1]
    a0 = np.where(skin < T_F, T_F, skin) - 272.0
    w = np.asarray(grid.quadrature_weights)[:, None] * np.ones((1, S[1]))
    assert provider.stats["abar_k"] == pytest.approx(float((a0 * w).sum() / w.sum()), abs=1e-12)
    assert provider.sst_host(0.0).min() >= T_F


def test_a_class_change_reaches_the_runtime_behind_the_physics_bridge():
    """The model's physics is the bridge and the runtime is on its adapter:
    the invalidation must drop the masks there, or a melted column keeps
    the frozen-surface step."""
    from arwen_global.physics.arwen_bridge import NativeArwenPhysicsBridge
    from arwen_global.physics.native_runtime import _BandCache
    from arwen_global.surface_series import invalidate_runtime_masks

    assert "adapter" in NativeArwenPhysicsBridge.__post_init__.__code__.co_names
    cache = _BandCache()
    cache.frozen, cache.frozen_count = np.ones((2, 2), bool), 4
    cache.lake, cache.statics_checked = np.zeros((2, 2), bool), True
    bridge = SimpleNamespace(adapter=SimpleNamespace(_runtime=SimpleNamespace(
        _bands={None: cache})))
    assert invalidate_runtime_masks(SimpleNamespace(physics=bridge)) == 1
    assert cache.frozen is None and cache.frozen_count is None
    assert cache.lake is None and cache.statics_checked is False
