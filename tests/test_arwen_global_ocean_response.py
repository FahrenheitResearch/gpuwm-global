"""The mixed-layer cooling response of a prescribed sea surface
(arwen_global.ocean_response), on known answers.

* Off by default: no [ocean] key moves a config hash while the response is
  off, a run with it off carries no array or metadata of it, and the keys
  are refused without the switch, under the slab and with the calibration.
* The arithmetic: a column whose flux never changes is never moved; one
  step of an event moves D by dt (F - F_ref) / (rho_w c_w h); relaxation
  alone decays as (1 - dt/tau)^n; D is bounded; the reference is the plain
  mean, then the exponential one, of the limited flux, held in its range; a
  diurnal flux averages to zero.
* The boundary: only open ocean moves (land, ice and lake columns keep
  their skin and D = 0), the skin is the held skin plus D, floored at T_f.
* The T3 reference smoke under hold and under the anomaly rule: straight
  and restarted, every checkpoint array equal byte for byte, float64 and
  float32; a branch switches the response on at a checkpoint written
  without it and is refused without --branch-from-other-config.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global import ocean_response as orr
from arwen_global.boundary_options import OceanOptions, SeaIceOptions
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.ocean_coupler import HELD_NET_FLUX
from arwen_global.ocean_rules import RuleProvider
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.state import PhysicsState
from arwen_global.surface_boundary import SurfaceBoundary
from arwen_global.surface_climatology import load_climatology, write_climatology

T_F = 271.35
RHO_CP = 1026.0 * 3990.0
DAY = 86400.0
START = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
GDAS = str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")


def _response_options(mode="hold", **kw):
    ice = kw.pop("sea_ice", SeaIceOptions())
    base = dict(declared=True, implicit=False, mode=mode, anomaly_reference="start",
                anomaly_truncation=None, mixed_layer_response=True)
    base.update(kw)
    return OceanOptions(sea_ice=ice, **base)


# -- the parser and the identity ----------------------------------------------

def _config(tmp_path, extra, name):
    from arwen_global.config import load_config

    path = tmp_path / name
    path.write_text(Path(GDAS).read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    return load_config(path)


def test_off_by_default_and_every_hash_unmoved_while_off(tmp_path):
    clim = tmp_path / "c.npz"
    clim.write_bytes(b"climatology-bytes")
    anomaly = (f'[ocean]\nmode = "anomaly"\nclimatology = "{clim.as_posix()}"\n'
               'anomaly_reference = "start"\n')
    base = _config(tmp_path, "", "a.toml")
    assert not base.ocean.mixed_layer_response
    assert _config(tmp_path, '[ocean]\nmode = "hold"\n', "b.toml").config_hash \
        == base.config_hash
    off = _config(tmp_path, '[ocean]\nmode = "hold"\nmixed_layer_response = false\n', "c.toml")
    assert off.config_hash == base.config_hash and off.ocean.identity is None
    plain = _config(tmp_path, anomaly, "d.toml")
    assert _config(tmp_path, anomaly + "mixed_layer_response = false\n",
                   "e.toml").config_hash == plain.config_hash
    assert "mixed_layer_response" not in plain.ocean.receipt()
    assert "mixed_layer_response" not in plain.ocean.identity
    # On: the hash moves, and with each key.
    on = _config(tmp_path, '[ocean]\nmode = "hold"\nmixed_layer_response = true\n', "f.toml")
    assert on.config_hash != base.config_hash
    assert on.ocean.mixed_layer_depth_m == 40.0
    assert on.ocean.mixed_layer_relaxation_days == 10.0
    assert on.ocean.identity["mixed_layer_response"]["max_departure_k"] == 5.0
    for index, (key, value) in enumerate((("mixed_layer_depth_m", "30.0"),
                                          ("mixed_layer_relaxation_days", "5.0"),
                                          ("mixed_layer_max_departure_k", "3.0"),
                                          ("mixed_layer_reference_days", "60.0"))):
        other = _config(tmp_path, '[ocean]\nmode = "hold"\nmixed_layer_response = true\n'
                        f"{key} = {value}\n", f"g{index}.toml")
        assert other.config_hash != on.config_hash, key
    on_anomaly = _config(tmp_path, anomaly + "mixed_layer_response = true\n", "h.toml")
    assert on_anomaly.config_hash != plain.config_hash
    series = tmp_path / "s.npz"
    series.write_bytes(b"series-bytes")
    refused = (
        ('[ocean]\nmode = "hold"\nmixed_layer_depth_m = 30.0\n', "mixed_layer_response = true"),
        ('[ocean]\nmode = "hold"\nmixed_layer_response = "yes"\n', "true or false"),
        ('[ocean]\nmode = "hold"\nmixed_layer_response = true\nmixed_layer_depth_m = -1.0\n',
         "positive"),
        (anomaly + "qflux_calibration = true\nmixed_layer_response = true\n", "climatology"),
        (anomaly.replace('"anomaly"', '"climatology"', 1)
         + "qflux_calibration = true\nmixed_layer_response = true\n", "qflux_calibration"),
    )
    for index, (text, match) in enumerate(refused):
        with pytest.raises((ValueError, FileNotFoundError), match=match):
            _config(tmp_path, text, f"r{index}.toml")
    q = tmp_path / "q.npz"
    q.write_bytes(b"q")
    with pytest.raises(ValueError, match="already answers"):
        _config(tmp_path, f'[ocean]\nmode = "slab"\nclimatology = "{clim.as_posix()}"\n'
                f'slab_qflux = "{q.as_posix()}"\nmixed_layer_response = true\n', "s.toml")


def test_the_off_options_carry_no_response_key():
    for mode in ("hold", "series", "anomaly", "climatology"):
        options = OceanOptions(declared=True, implicit=False, mode=mode)
        assert options.response_receipt() == {}
        assert "mixed_layer_response" not in options.receipt()
        identity = options.identity
        assert identity is None or "mixed_layer_response" not in identity


# -- the arithmetic -----------------------------------------------------------

def test_a_steady_flux_never_moves_the_sea():
    ref = np.zeros(3)
    dep = np.zeros(3)
    flux = np.array([-350.0, 120.0, -40.0])
    for n in range(1, 200):
        ref = orr.reference_flux_step(ref, flux, dt_s=300.0, elapsed_s=300.0 * n,
                                      reference_days=30.0, xp=np)
        dep = orr.departure_step(dep, flux, ref, depth_m=40.0, relaxation_days=10.0,
                                 max_departure_k=5.0, dt_s=300.0, xp=np)
        assert np.array_equal(ref, flux)
        assert np.array_equal(dep, np.zeros(3))


def test_one_step_of_an_event_moves_d_by_dt_f_over_rho_c_h():
    dep = orr.departure_step(np.zeros(1), np.array([-2400.0]), np.array([-150.0]),
                             depth_m=40.0, relaxation_days=10.0, max_departure_k=5.0,
                             dt_s=300.0, xp=np)
    assert dep[0] == pytest.approx(300.0 * (-2250.0) / (RHO_CP * 40.0), rel=1e-14)
    # The reference is held in its range: a -2000 W/m2 "normal" is -600.
    dep = orr.departure_step(np.zeros(1), np.array([-2400.0]), np.array([-2000.0]),
                             depth_m=40.0, relaxation_days=10.0, max_departure_k=5.0,
                             dt_s=300.0, xp=np)
    assert dep[0] == pytest.approx(300.0 * (-1800.0) / (RHO_CP * 40.0), rel=1e-14)


def test_relaxation_alone_decays_and_the_bound_holds():
    dep = np.array([-3.0, 2.0])
    for _ in range(100):
        dep = orr.departure_step(dep, np.zeros(2), np.zeros(2), depth_m=40.0,
                                 relaxation_days=10.0, max_departure_k=5.0, dt_s=300.0,
                                 xp=np)
    factor = (1.0 - 300.0 / (10.0 * DAY)) ** 100
    assert np.allclose(dep, [-3.0 * factor, 2.0 * factor], rtol=1e-12)
    dep = np.zeros(2)
    for _ in range(2000):
        dep = orr.departure_step(dep, np.array([-5000.0, 5000.0]), np.zeros(2),
                                 depth_m=20.0, relaxation_days=100.0, max_departure_k=4.0,
                                 dt_s=300.0, xp=np)
    assert dep.tolist() == [-4.0, 4.0]


def test_the_reference_is_the_plain_then_the_exponential_mean_of_the_limited_flux():
    fluxes = [-100.0, -300.0, -1600.0, 200.0]
    ref = np.zeros(1)
    for n, f in enumerate(fluxes, start=1):
        ref = orr.reference_flux_step(ref, np.array([f]), dt_s=DAY, elapsed_s=n * DAY,
                                      reference_days=30.0, xp=np)
    limited = [-100.0, -300.0, -1000.0, 200.0]
    assert ref[0] == pytest.approx(sum(limited) / 4.0, rel=1e-14)
    # Past the window: an exponential mean with weight dt / window.
    after = orr.reference_flux_step(ref, np.array([500.0]), dt_s=DAY, elapsed_s=40 * DAY,
                                    reference_days=30.0, xp=np)
    assert after[0] == pytest.approx(ref[0] + (500.0 - ref[0]) / 30.0, rel=1e-14)


def test_a_diurnal_flux_moves_the_sea_by_hundredths_and_averages_to_zero():
    dt = 1800.0
    ref, dep = np.zeros(1), np.zeros(1)
    kept = []
    for n in range(1, int(120 * DAY / dt) + 1):
        t = n * dt
        flux = np.array([-50.0 + 400.0 * math.sin(2.0 * math.pi * t / DAY)])
        ref = orr.reference_flux_step(ref, flux, dt_s=dt, elapsed_s=t, reference_days=30.0,
                                      xp=np)
        dep = orr.departure_step(dep, flux, ref, depth_m=40.0, relaxation_days=10.0,
                                 max_departure_k=5.0, dt_s=dt, xp=np)
        kept.append(float(dep[0]))
    # The first days' phase leaves the young mean a few W/m2 off, which the
    # exponential mean forgets: about 0.02 K at a month, 0.002 K by day 90.
    early, late = kept[int(30 * DAY / dt):int(60 * DAY / dt)], kept[int(90 * DAY / dt):]
    assert max(abs(v) for v in kept) < 0.15
    assert max(abs(v) for v in late) < 0.06
    assert abs(sum(early) / len(early)) < 0.02
    assert abs(sum(late) / len(late)) < 0.005


# -- the boundary ---------------------------------------------------------------

def _hold_state(grid, skin):
    shape = (int(grid.nlat), int(grid.nlon))
    land = np.zeros(shape)
    lake = np.zeros(shape)
    ice = np.zeros(shape)
    land[:, :8] = 1.0          # a continent
    lake[10:14, 20:24] = 1.0   # an inland lake (no land, all lake)
    ice[:3, :] = 1.0           # a polar pack
    cold = {"temperature_k": np.broadcast_to(skin, shape).astype(np.float64),
            "land_fraction": land, "lake_fraction": lake, "sea_ice_fraction": ice,
            "sea_ice_thickness_m": ice.copy()}
    surface = SimpleNamespace(
        temperature_k=cold["temperature_k"].astype(np.float32),
        land_fraction=land.astype(np.float32), lake_fraction=lake.astype(np.float32),
        sea_ice_fraction=ice.astype(np.float32),
        sea_ice_thickness_m=ice.astype(np.float32),
        albedo=np.full(shape, 0.08, np.float32), emissivity=np.full(shape, 0.98, np.float32))
    state = SimpleNamespace(surface=surface, physics_state=PhysicsState(), time_s=0.0)
    return cold, state


def _hold_response(grid, cold, options=None):
    options = options or _response_options()
    base = RuleProvider(options, None, grid, start_time=START, cold_surface=cold,
                        model_truncation=21)
    return orr.MixedLayerResponse(base, options, grid, start_time=START, cold_surface=cold,
                                  physics_mode="reference")


def test_only_open_ocean_moves_and_the_skin_is_the_held_skin_plus_d():
    grid = GaussianGrid.create(21)
    cold, state = _hold_state(grid, 300.0)
    response = _hold_response(grid, cold)
    response.initialize(state)
    assert HELD_NET_FLUX in state.physics_state.arrays
    boundary = SurfaceBoundary(response)
    held = state.surface.temperature_k.copy()
    boundary.apply(state)                     # t = 0: nothing to advance
    assert np.array_equal(state.surface.temperature_k, held)
    for t, flux in ((300.0, -200.0), (600.0, -1200.0)):
        state.physics_state.arrays[HELD_NET_FLUX] = np.full(held.shape, flux)
        state.time_s = t
        boundary.apply(state)
    dep = state.physics_state.arrays[orr.RESPONSE_DEPARTURE]
    ref = state.physics_state.arrays[orr.RESPONSE_REFERENCE_FLUX]
    open_ocean = response.open0
    assert open_ocean.sum() == held.size - 8 * held.shape[0] - 16 - 3 * (held.shape[1] - 8)
    # Step 1: F = F_ref, D = 0.  Step 2: F_ref = mean(-200, -1000) = -600.
    expected = 300.0 * (-1200.0 + 600.0) / (RHO_CP * 40.0)
    assert np.allclose(ref[open_ocean], -600.0, rtol=0, atol=1e-12)
    assert np.allclose(dep[open_ocean], expected, rtol=1e-13, atol=0)
    assert (dep[~open_ocean] == 0.0).all() and (ref[~open_ocean] == 0.0).all()
    skin = state.surface.temperature_k
    assert np.array_equal(skin[~open_ocean], held[~open_ocean])
    assert np.array_equal(skin[open_ocean],
                          (300.0 + dep[open_ocean]).astype(np.float32))
    rows = response.diagnostics(state)
    assert rows["ocean_response_mean_departure_k"] == pytest.approx(expected, rel=1e-12)
    assert rows["ocean_response_mean_reference_flux_w_m2"] == pytest.approx(-600.0)
    assert response.identity_block()["mixed_layer_response"]["depth_m"] == 40.0
    assert response.identity_without_response() is None


def test_the_skin_is_floored_at_the_freezing_point():
    grid = GaussianGrid.create(21)
    cold, state = _hold_state(grid, 273.0)
    response = _hold_response(grid, cold, _response_options(mixed_layer_max_departure_k=3.0,
                                                            mixed_layer_depth_m=10.0))
    response.initialize(state)
    boundary = SurfaceBoundary(response)
    for n in range(1, 400):
        state.physics_state.arrays[HELD_NET_FLUX] = np.full(
            state.surface.temperature_k.shape, -400.0 if n < 3 else -3000.0)
        state.time_s = 300.0 * n
        boundary.apply(state)
    dep = state.physics_state.arrays[orr.RESPONSE_DEPARTURE]
    assert dep[response.open0].min() == -3.0
    assert (state.surface.temperature_k[response.open0] == np.float32(T_F)).all()


# -- the T3 reference smoke -------------------------------------------------------

def _smoke(monkeypatch, start, ocean, precision="float64"):
    import arwen_global.clock as clock
    from arwen_global.config import load_config

    def dated(cfg, provenance):
        return clock.ForecastClock(start, clock.CONFIG_START_TIME, declared_start_time=start)

    monkeypatch.setattr(clock, "resolve_forecast_clock", dated)
    cfg = load_config(SMOKE)
    return replace(cfg, ocean=ocean, duration_s=80.0, output_interval_s=40.0,
                   precision=precision)


def _same(a, b):
    from arwen_global.checkpoint import read_checkpoint

    meta_a, arrays_a = read_checkpoint(a)
    meta_b, arrays_b = read_checkpoint(b)
    assert meta_a["physics_metadata"] == meta_b["physics_metadata"]
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    return meta_a, arrays_a


def _smoke_climatology(tmp_path, cfg):
    grid = GaussianGrid.create(int(cfg.truncation), dealias_factor=float(cfg.dealias_factor))
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    lat = np.asarray(grid.latitude_deg)[None, :, None]
    months = np.arange(12)[:, None, None]
    sst = 282.0 + 6.0 * np.cos(np.radians(lat)) + 0.8 * months + np.zeros((12, nlat, nlon))
    ocean = np.ones((nlat, nlon), dtype=bool)
    path = tmp_path / "clim.npz"
    write_climatology(path, {"sst_k": sst, "sea_ice_fraction": np.zeros((12, nlat, nlon))},
                      ocean=ocean, latitude_deg=grid.latitude_deg,
                      longitude_deg=grid.longitude_deg,
                      provenance={"schema": "gpuwm.arwen-global-surface-climatology/v1",
                                  "test": True})
    return load_climatology(path)


def test_with_the_response_off_the_smoke_carries_nothing_of_it(tmp_path):
    from arwen_global.checkpoint import read_checkpoint
    from arwen_global.config import load_config
    from arwen_global.runner import run

    run(load_config(SMOKE), tmp_path / "off")
    meta, arrays = read_checkpoint(tmp_path / "off" / "arwen_global_step00000004.npz")
    assert not any("ocean_response" in name for name in arrays), sorted(arrays)
    assert orr.METADATA_KEY not in meta["physics_metadata"]


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("mode", ["hold", "anomaly"])
def test_a_restarted_response_run_reproduces_the_straight_run_bit_for_bit(
        tmp_path, monkeypatch, mode, precision):
    from arwen_global.checkpoint import read_checkpoint
    from arwen_global.config import load_config
    from arwen_global.runner import run

    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    extra = {}
    if mode == "anomaly":
        clim = _smoke_climatology(tmp_path, load_config(SMOKE))
        extra = dict(climatology=str(clim.path), climatology_sha256=clim.sha256)
    # A thin, fast mixed layer so eight smoke steps move it measurably.
    options = _response_options(mode, mixed_layer_depth_m=1.0, mixed_layer_relaxation_days=1.0,
                                **extra)
    cfg = _smoke(monkeypatch, start, options, precision)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = run(cfg, full)
    assert "mixed_layer_response" in result["surface_update"]
    assert result["surface_update"]["sst_steps"] == 8
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    meta, arrays = _same(full / "arwen_global_step00000008.npz",
                         resumed / "arwen_global_step00000008.npz")
    # Steps 1..7 advanced (the step-0 apply has no interval behind it).
    assert meta["physics_metadata"][orr.METADATA_KEY]["advances"] == 7
    assert "mixed_layer_response" in meta["physics_metadata"]["surface_boundary"]
    dep = arrays[f"physics__{orr.RESPONSE_DEPARTURE}"]
    assert dep.dtype == np.float64 and np.abs(dep).max() > 0.0
    _m0, arrays0 = read_checkpoint(full / "arwen_global_step00000000.npz")
    assert not np.abs(arrays0[f"physics__{orr.RESPONSE_DEPARTURE}"]).any()
    rows = [line for line in (full / "diagnostics.jsonl").read_text().splitlines() if line]
    assert any('"ocean_response_mean_departure_k"' in line for line in rows)
    # A restart with the response off, or with another depth, is refused.
    for other in (replace(cfg.ocean, mixed_layer_response=False),
                  replace(cfg.ocean, mixed_layer_depth_m=2.0)):
        with pytest.raises(ValueError):
            run(replace(cfg, ocean=other), tmp_path / "other",
                restart=full / "arwen_global_step00000004.npz")


def test_a_branch_switches_the_response_on_at_a_held_checkpoint(tmp_path, monkeypatch):
    import json

    from arwen_global.runner import run

    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    held = OceanOptions(declared=True, implicit=False, mode="hold")
    parent = replace(_smoke(monkeypatch, start, held), output_interval_s=20.0)
    run(parent, tmp_path / "parent")
    on = replace(parent, ocean=_response_options(mixed_layer_depth_m=1.0))
    checkpoint = tmp_path / "parent" / "arwen_global_step00000004.npz"
    with pytest.raises(ValueError):
        run(on, tmp_path / "refused", restart=checkpoint)
    result = run(on, tmp_path / "branch", restart=checkpoint, branch_config=True)
    assert result["surface_update"]["mixed_layer_response"]["started_at_branch"]
    receipt = json.loads((tmp_path / "branch" / "arwen-global-receipt.json").read_text())
    boundary = receipt["branch"]["surface_boundary"]
    assert boundary["parent"] is None and "mixed_layer_response" in boundary["this"]
    from arwen_global.checkpoint import read_checkpoint

    meta, arrays = read_checkpoint(tmp_path / "branch" / "arwen_global_step00000008.npz")
    assert meta["physics_metadata"][orr.METADATA_KEY]["started_s"] == 40.0
    assert np.abs(arrays[f"physics__{orr.RESPONSE_DEPARTURE}"]).max() > 0.0
    # The branch's own checkpoints restart as the response, without the flag.
    run(on, tmp_path / "again", restart=tmp_path / "branch" / "arwen_global_step00000006.npz")
    _same(tmp_path / "branch" / "arwen_global_step00000008.npz",
          tmp_path / "again" / "arwen_global_step00000008.npz")
