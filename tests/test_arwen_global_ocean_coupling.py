"""The slab ocean in the run (arwen_global.ocean_coupler), on known answers.

Package C of docs/ARWEN_GLOBAL_OCEAN.md: the switch, the coupling in the
boundary service, the physics-state arrays, the restart and the tape.  The
arithmetic of the mixed layer, the Q-flux file and the ice mass is the other
packages' (ocean_slab, ocean_qflux, sea_ice_thermo).  The tests were
written while those were stubs and stand in a plain transcription of the
formula a docstring states for any function that is still one; all of them
are built now, so the real functions run (the first test below holds that).

* OC-5 (the coupler's part): a constant held flux with a fixed depth moves T
  linearly, dt F / (rho_w c_w h) a step; restoring alone decays as
  (1 - dt/tau)^n; cooling past the freezing point holds T at T_f and books
  the deficit as ice; the calibration deposits F dt into the two month bins
  with the month weights and nothing during the spin-up; a record written
  before a restart is not written again; the tape fields.
* OC-3 (the books): on every ocean column |stored - sources| stays at
  float64 rounding.
* OC-1: the T3 reference smoke with no [ocean] table carries no ocean array,
  no held flux plane and no slab metadata, and every shipped config hash is
  unmoved (the parser leaves an absent table where it was).
* OC-2: the T3 smoke under the slab and under the calibration, straight and
  restarted at a step inside an ice record: every checkpoint array, the
  physics metadata and the trackers equal byte for byte.
* The parser: every combination the design page refuses.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global import ocean_coupler as oc
from arwen_global.boundary_options import OceanOptions, SeaIceOptions
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.state import PhysicsState
from arwen_global.surface_climatology import load_climatology, month_weights, write_climatology

T_F = 271.35
RHO_CP = 1026.0 * 3990.0
RHO_L = 917.0 * 3.337e5
DAY = 86400.0
START = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
GDAS = str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")


# -- stand-ins for the other packages' functions while they are stubs --------

def _is_stub(func) -> bool:
    code = getattr(func, "__code__", None)
    return code is not None and code.co_names == ("NotImplementedError",)


def _mixed_layer_depth(climatology, when, params, ocean, xp):
    if params.depth != "woa18":
        return xp.where(ocean, float(params.depth), 0.0)
    i0, i1, w = month_weights(when)
    mld = climatology.mixed_layer_depth_m
    h = (1.0 - w) * np.asarray(mld[i0], np.float64) + w * np.asarray(mld[i1], np.float64)
    lo, hi = params.depth_bounds_m
    h = np.where(np.isfinite(h), np.clip(h, lo, hi), lo)
    return xp.where(ocean, xp.asarray(h), 0.0)


def _climatology_tendency(climatology, when, xp):
    shape = climatology.sst_k.shape[1:]
    return xp.zeros(shape), xp.zeros(shape)


def _open_water_flux(*, swdown, albedo, glw, emissivity, skin_k, hfx, qfx, xp):
    return (swdown * (1.0 - albedo) + emissivity * (glw - 5.670374419e-8 * skin_k ** 4)
            - hfx - 2.5e6 * qfx)


def _qflux(*, depth_m, sst_rate_k_s, ice_fraction_rate_s, ice_reference_thickness_m,
           calibration_flux_w_m2, xp):
    return (RHO_CP * depth_m * sst_rate_k_s
            - RHO_L * ice_reference_thickness_m * ice_fraction_rate_s - calibration_flux_w_m2)


def _restoring_flux(*, temperature_k, target_k, depth_m, relaxation_days, xp):
    if relaxation_days is None:
        return xp.zeros_like(temperature_k)
    return RHO_CP * depth_m * (target_k - temperature_k) / (relaxation_days * DAY)


def _slab_step(*, temperature_k, depth_m, net_flux_w_m2, dt_s, xp):
    unheld = temperature_k + dt_s * net_flux_w_m2 / (RHO_CP * depth_m)
    deficit = xp.where(unheld < T_F, RHO_CP * depth_m * (T_F - unheld), 0.0)
    return xp.maximum(unheld, T_F), deficit


def _initial_temperature(cold_skin_k, ocean, ice_fraction, xp):
    t = xp.where(ice_fraction > 0.0, T_F, xp.maximum(cold_skin_k, T_F))
    return xp.where(ocean, t, 0.0)


def _deposit(sum_j_m2, weight_s, *, flux_w_m2, columns, when, dt_s, xp):
    i0, i1, w = month_weights(when)
    for index, share in ((i0, 1.0 - w), (i1, w)):
        sum_j_m2[index] += xp.where(columns, share * flux_w_m2 * dt_s, 0.0)
        weight_s[index] += xp.where(columns, share * dt_s, 0.0)


def _months_needed(start, duration_s):
    return set(range(12))


def _ice_reference_thickness(latitude_deg, nlon, xp):
    lat = np.asarray(latitude_deg)[:, None] * np.ones((1, nlon))
    return xp.asarray(np.where(lat >= 0.0, 2.0, 1.0))


def _initial_ice(fraction, thickness_m, ocean, new_ice_thickness_m, xp):
    a = xp.where(ocean, xp.clip(fraction, 0.0, 1.0), 0.0)
    h = xp.where(thickness_m > 0.0, thickness_m, new_ice_thickness_m)
    return a, xp.where(a > 0.0, a * h, 0.0)


def _thickness(fraction, volume_m, xp):
    return xp.where(fraction > 0.0, volume_m / xp.where(fraction > 0.0, fraction, 1.0), 0.0)


def _zero_flux(**kw):
    return kw["xp"].zeros_like(kw["thickness_m"])


def _ice_step(*, fraction, volume_m, ocean_flux_w_m2, basal_conduction_w_m2, top_melt_w_m2,
              freeze_deficit_j_m2, dt_s, new_ice_thickness_m, xp):
    from arwen_global.sea_ice_thermo import IceStep

    change = (dt_s * (fraction * (basal_conduction_w_m2 - top_melt_w_m2) - ocean_flux_w_m2)
              + freeze_deficit_j_m2) / RHO_L
    raw = volume_m + change
    new_area = xp.where(freeze_deficit_j_m2 > 0.0,
                        freeze_deficit_j_m2 / RHO_L / new_ice_thickness_m, 0.0)
    area = xp.minimum(1.0, fraction + new_area)
    melting = (raw < volume_m) & (volume_m > 0.0)
    area = xp.where(melting, fraction * xp.sqrt(xp.maximum(raw, 0.0)
                                                / xp.where(melting, volume_m, 1.0)), area)
    heat = xp.where(raw < 0.0, -raw * RHO_L, 0.0)
    volume = xp.maximum(raw, 0.0)
    area = xp.where(volume > 0.0, area, 0.0)
    return IceStep(area, volume, heat, -RHO_L * (volume - volume_m))


_STAND_INS = {
    "arwen_global.ocean_slab": {
        "mixed_layer_depth": _mixed_layer_depth, "climatology_tendency": _climatology_tendency,
        "open_water_flux": _open_water_flux, "qflux": _qflux,
        "restoring_flux": _restoring_flux, "slab_step": _slab_step,
        "initial_temperature": _initial_temperature,
    },
    "arwen_global.ocean_qflux": {"deposit": _deposit, "months_needed": _months_needed},
    "arwen_global.sea_ice_thermo": {
        "ice_reference_thickness": _ice_reference_thickness, "initial_ice": _initial_ice,
        "thickness": _thickness, "basal_conduction": _zero_flux,
        "top_melt_surplus": _zero_flux, "ice_step": _ice_step,
    },
}


@pytest.fixture
def parts(monkeypatch):
    """The other packages' functions, a stand-in for each one still a stub."""
    import importlib

    stood_in = []
    for module_name, functions in _STAND_INS.items():
        module = importlib.import_module(module_name)
        for name, stand_in in functions.items():
            if _is_stub(getattr(module, name)):
                monkeypatch.setattr(module, name, stand_in)
                stood_in.append(f"{module_name}.{name}")
    return stood_in


class FakeQFlux:
    """A Q-flux file's reading surface (ocean_qflux.QFluxFile) with a
    constant F_cal: the coupler's arithmetic is tested, not the file's."""

    def __init__(self, shape, *, flux=0.0, sha256="f" * 64):
        self.sha256 = sha256
        self.flux = float(flux)
        self.coverage_s = np.ones((12, *shape))
        self.flux_w_m2 = np.full((12, *shape), self.flux)
        self.provenance = {"schema": "test"}
        self.checked = None

    def at(self, when, xp=np):
        return xp.full(self.coverage_s.shape[1:], self.flux)

    def check(self, **kw):
        self.checked = kw


# -- fixtures ----------------------------------------------------------------

def _climatology(tmp_path, grid, *, sst=None, ice=None, ocean=None, mld=None,
                 name="clim.npz"):
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    if sst is None:
        sst = 285.0 + np.zeros((12, nlat, nlon))
    if ice is None:
        ice = np.zeros((12, nlat, nlon))
    if ocean is None:
        ocean = np.ones((nlat, nlon), dtype=bool)
    planes = {"sst_k": np.where(ocean[None], sst, np.nan),
              "sea_ice_fraction": np.where(ocean[None], ice, np.nan)}
    if mld is not None:
        planes["mixed_layer_depth_m"] = np.where(ocean[None], mld, np.nan)
    path = tmp_path / name
    write_climatology(path, planes, ocean=ocean, latitude_deg=grid.latitude_deg,
                      longitude_deg=grid.longitude_deg,
                      provenance={"schema": "gpuwm.arwen-global-surface-climatology/v1",
                                  "test": True})
    return load_climatology(path)


def _slab_options(**kw):
    ice = kw.pop("sea_ice", SeaIceOptions(mode="match", new_ice_thickness_m=0.5))
    base = dict(declared=True, implicit=False, mode="slab", climatology="x",
                climatology_sha256="0" * 64, slab_qflux="q", slab_qflux_sha256="f" * 64,
                slab_depth=50.0, slab_relaxation_days=None, anomaly_reference="start",
                anomaly_truncation=None)
    base.update(kw)
    return OceanOptions(sea_ice=ice, **base)


def _cold(shape, skin, *, ice=None, thickness=None):
    zeros = np.zeros(shape)
    return {"temperature_k": np.broadcast_to(skin, shape).astype(np.float64),
            "land_fraction": zeros, "lake_fraction": zeros.copy(),
            "sea_ice_fraction": zeros.copy() if ice is None else ice,
            "sea_ice_thickness_m": zeros.copy() if thickness is None else thickness}


def _state(cold, time_s=0.0):
    surface = SimpleNamespace(
        temperature_k=cold["temperature_k"].astype(np.float32),
        land_fraction=cold["land_fraction"].astype(np.float32),
        lake_fraction=cold["lake_fraction"].astype(np.float32),
        sea_ice_fraction=cold["sea_ice_fraction"].astype(np.float32),
        sea_ice_thickness_m=cold["sea_ice_thickness_m"].astype(np.float32),
        albedo=np.full(cold["land_fraction"].shape, 0.08, np.float32),
        emissivity=np.full(cold["land_fraction"].shape, 0.98, np.float32))
    return SimpleNamespace(surface=surface, physics_state=PhysicsState(), time_s=time_s)


def _provider(tmp_path, *, options=None, skin=290.0, ice=None, thickness=None, flux=0.0,
              clim=None, calibrate=False, rule=None):
    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    clim = clim or _climatology(tmp_path, grid)
    cold = _cold(shape, skin, ice=ice, thickness=thickness)
    provider = oc.SlabProvider(
        options or _slab_options(), clim, None if calibrate else FakeQFlux(shape, flux=flux),
        grid, start_time=START, cold_surface=cold, model_truncation=21, rule=rule,
        calibrate=calibrate, physics_mode="reference", duration_s=10 * DAY)
    state = _state(cold)
    provider.initialize(state)
    return provider, state, grid


def _march(provider, state, held, steps, dt=300.0):
    """``steps`` advances with the reference suite's held flux ``held``."""
    for _ in range(steps):
        state.physics_state.arrays[oc.HELD_NET_FLUX] = np.broadcast_to(
            held, state.surface.temperature_k.shape).astype(np.float64)
        state.time_s = float(state.time_s) + dt
        provider.advance(state)


def _residual(state):
    a = state.physics_state.arrays
    return float(np.abs(a[oc.ENERGY_STORED] - a[oc.ENERGY_SOURCES]).max())


def test_every_package_function_is_built_so_nothing_is_stood_in(parts):
    """With packages A and B merged, the coupling tests run their real
    arithmetic: no function the coupler calls is still the design's stub."""
    assert parts == []


# -- OC-5: the coupler's known answers ----------------------------------------

def test_a_constant_flux_with_a_fixed_depth_moves_the_slab_linearly(tmp_path, parts):
    provider, state, _grid = _provider(tmp_path)
    arrays = state.physics_state.arrays
    assert set(oc.SLAB_ARRAYS) | {oc.HELD_NET_FLUX} == set(arrays)
    assert all(arrays[name].dtype == np.float64 for name in arrays)
    t0 = arrays[oc.SLAB_TEMPERATURE].copy()
    np.testing.assert_array_equal(t0, 290.0)
    provider.advance(state)                       # the cold start: an empty interval
    np.testing.assert_array_equal(arrays[oc.SLAB_TEMPERATURE], t0)
    _march(provider, state, 100.0, 12)
    expect = 290.0 + 12 * 300.0 * 100.0 / (RHO_CP * 50.0)
    np.testing.assert_allclose(state.physics_state.arrays[oc.SLAB_TEMPERATURE], expect,
                               rtol=0, atol=1e-10)
    meta = state.physics_state.metadata[oc.METADATA_KEY]
    assert meta["advances"] == 12 and meta["last_update_s"] == 3600.0
    np.testing.assert_allclose(state.physics_state.arrays[oc.ENERGY_SOURCES],
                               12 * 300.0 * 100.0, rtol=1e-12)
    assert _residual(state) <= 1e-9 * 12 * 300.0 * 100.0 + 1e-3
    # The SST the service writes is T, on the ocean columns.
    sst, defined = provider.sst_at(3600.0, np, np.float32)
    assert defined.all() and sst.dtype == np.float32
    np.testing.assert_array_equal(sst, np.float32(expect))


def test_the_q_flux_enters_as_minus_f_cal(tmp_path, parts):
    """A column driven by exactly F_cal on a flat climatology stays put."""
    provider, state, _grid = _provider(tmp_path, flux=-40.0)   # F_cal = -40: Q = +40
    _march(provider, state, -40.0, 6)
    np.testing.assert_allclose(state.physics_state.arrays[oc.SLAB_TEMPERATURE], 290.0,
                               rtol=0, atol=1e-11)
    np.testing.assert_allclose(state.physics_state.arrays[oc.SLAB_QFLUX], 40.0, rtol=1e-12)


def test_restoring_alone_decays_as_one_minus_dt_over_tau_to_the_n(tmp_path, parts):
    options = _slab_options(slab_relaxation_days=10.0)
    provider, state, _grid = _provider(tmp_path, options=options, skin=286.0)
    n, dt = 40, 300.0
    _march(provider, state, 0.0, n, dt)
    expect = 285.0 + 1.0 * (1.0 - dt / (10.0 * DAY)) ** n
    np.testing.assert_allclose(state.physics_state.arrays[oc.SLAB_TEMPERATURE], expect,
                               rtol=0, atol=1e-10)
    assert _residual(state) <= 1e-3   # OC-3: float64 rounding of rho c h dT at 286 K


def test_the_anomaly_target_keeps_the_starting_anomaly_the_rule_keeps(tmp_path, parts):
    """Under slab_target = "anomaly" the restoring pulls toward the anomaly
    rule's SST (here the climatology plus a persisted +2 K), so a slab that
    starts on it and takes no flux stays there; toward the climatology it
    decays."""
    from arwen_global.ocean_rules import RuleProvider

    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid)
    options = _slab_options(slab_relaxation_days=10.0, slab_target="anomaly",
                            climatology_sha256=clim.sha256)
    rule = RuleProvider(replace(options, mode="anomaly",
                                sea_ice=replace(options.sea_ice, mode="hold")),
                        clim, grid, start_time=START, cold_surface=_cold(shape, 287.0),
                        model_truncation=21)
    assert rule.stats["abar_k"] == pytest.approx(2.0, abs=1e-6)
    anomaly, state, _ = _provider(tmp_path, options=options, clim=clim, rule=rule, skin=287.0)
    _march(anomaly, state, 0.0, 20)
    np.testing.assert_allclose(state.physics_state.arrays[oc.SLAB_TEMPERATURE], 287.0,
                               rtol=0, atol=1e-6)
    assert "start_anomaly_sha256" in anomaly.identity_block()
    toward, other, _ = _provider(tmp_path, options=replace(options, slab_target="climatology"),
                                 clim=clim, skin=287.0)
    _march(toward, other, 0.0, 20)
    assert (other.physics_state.arrays[oc.SLAB_TEMPERATURE] < 287.0 - 1e-3).all()


def test_cooling_past_freezing_holds_t_f_and_books_the_deficit_as_ice(tmp_path, parts):
    provider, state, grid = _provider(tmp_path, skin=T_F + 0.001, options=_slab_options(
        slab_depth=10.0))
    flux = -400.0
    # The cold skin reaches the slab through the float32 surface.
    above = state.physics_state.arrays[oc.SLAB_TEMPERATURE] - T_F
    assert (above > 0.0).all()
    _march(provider, state, flux, 1)
    arrays = state.physics_state.arrays
    np.testing.assert_allclose(arrays[oc.SLAB_TEMPERATURE], T_F, rtol=0, atol=1e-12)
    deficit = -flux * 300.0 - RHO_CP * 10.0 * above
    np.testing.assert_allclose(arrays[oc.ICE_VOLUME], deficit / RHO_L, rtol=1e-9)
    assert (arrays[oc.ICE_FRACTION] > 0.0).all()
    # Under ice the mixed layer stays at T_f and the flux goes to the ice.
    _march(provider, state, flux, 3)
    np.testing.assert_allclose(arrays[oc.SLAB_TEMPERATURE], T_F, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.physics_state.arrays[oc.ICE_VOLUME],
                               (deficit + 3 * 400.0 * 300.0) / RHO_L, rtol=1e-9)
    assert _residual(state) <= 1e-9 * 4 * 400.0 * 300.0 + 1e-3
    # The record hands the prognostic ice to the service once.
    fraction, thick, columns = provider.ice_record(0, np, state.surface,
                                                   lambda xp, s, n: getattr(s, n))
    assert columns.all() and fraction.dtype == np.float32 and (fraction > 0).all()
    np.testing.assert_allclose(thick, _thickness(
        state.physics_state.arrays[oc.ICE_FRACTION], state.physics_state.arrays[oc.ICE_VOLUME],
        np), rtol=1e-6)
    assert provider.ice_record(0, np, state.surface, lambda xp, s, n: getattr(s, n)) == (
        None, None, None)


def test_a_melting_pack_warms_the_mixed_layer_with_what_is_left(tmp_path, parts):
    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    provider, state, _ = _provider(tmp_path, skin=T_F, ice=np.full(shape, 0.6),
                                   thickness=np.full(shape, 0.01),
                                   options=_slab_options(slab_depth=20.0))
    arrays = state.physics_state.arrays
    # A times the analysed thickness, both read through the float32 surface.
    volume = float(np.float32(0.6)) * float(np.float32(0.01))
    np.testing.assert_allclose(arrays[oc.ICE_VOLUME], volume, rtol=1e-12)
    np.testing.assert_array_equal(arrays[oc.SLAB_TEMPERATURE], T_F)
    _march(provider, state, 10000.0, 1)              # 3e6 J/m2 melts 0.0098 m of ice
    arrays = state.physics_state.arrays
    assert (arrays[oc.ICE_VOLUME] == 0.0).all() and (arrays[oc.ICE_FRACTION] == 0.0).all()
    left = 10000.0 * 300.0 - volume * RHO_L
    np.testing.assert_allclose(arrays[oc.SLAB_TEMPERATURE], T_F + left / (RHO_CP * 20.0),
                               rtol=0, atol=1e-10)
    assert _residual(state) <= 1e-9 * 3e6 + 1e-3


def test_under_ice_the_q_flux_leaves_out_the_climatology_sst_slope(tmp_path, parts):
    """A column driven by exactly F_cal on a climatology whose SST warms
    1 K a month and whose ice is flat: open water follows the slope, and a
    pack keeps its volume, because the heat that warms open water in the
    climatology is not handed to the ice (the first coupled card run melted
    the southern pack at twice the climatology's rate after mid-November,
    when that slope was given to the ice)."""
    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    sst = 280.0 + np.arange(12, dtype=np.float64)[:, None, None] + np.zeros((12, *shape))
    clim = _climatology(tmp_path, grid, sst=sst)
    from arwen_global.ocean_slab import month_bracket

    _i0, _i1, t0, t1 = month_bracket(START)
    rate = 1.0 / (t1 - t0).total_seconds()
    options = _slab_options(slab_depth=40.0)
    water, wstate, _ = _provider(tmp_path, options=options, skin=284.0, flux=-40.0, clim=clim)
    _march(water, wstate, -40.0, 12)
    np.testing.assert_allclose(wstate.physics_state.arrays[oc.SLAB_TEMPERATURE],
                               284.0 + 12 * 300.0 * rate, rtol=0, atol=1e-10)
    ice, istate, _ = _provider(tmp_path, options=options, skin=T_F, flux=-40.0, clim=clim,
                               ice=np.full(shape, 0.8), thickness=np.full(shape, 1.5))
    arrays = istate.physics_state.arrays
    volume = arrays[oc.ICE_VOLUME].copy()
    np.testing.assert_allclose(arrays[oc.SLAB_QFLUX], 40.0, rtol=1e-12)
    _march(ice, istate, -40.0, 12)
    arrays = istate.physics_state.arrays
    np.testing.assert_allclose(arrays[oc.ICE_VOLUME], volume, rtol=1e-12)
    np.testing.assert_array_equal(arrays[oc.SLAB_TEMPERATURE], T_F)
    np.testing.assert_allclose(arrays[oc.SLAB_QFLUX], 40.0, rtol=1e-12)
    assert _residual(istate) <= 1e-9 * 12 * 300.0 * 40.0 + 1e-3


def test_the_calibration_deposits_by_month_weight_after_the_spin_up(tmp_path, parts):
    from arwen_global.ocean_rules import RuleProvider

    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid)
    options = OceanOptions(declared=True, implicit=False, mode="climatology",
                           climatology="x", climatology_sha256=clim.sha256,
                           anomaly_reference="start", anomaly_truncation=None,
                           qflux_calibration=True,
                           qflux_calibration_spinup_days=600.0 / DAY)
    cold = _cold(shape, 288.0)
    rule = RuleProvider(options, clim, grid, start_time=START, cold_surface=cold,
                        model_truncation=21)
    provider, state, _ = _provider(tmp_path, options=options, clim=clim, calibrate=True,
                                   rule=rule, skin=288.0)
    assert set(state.physics_state.arrays) == set(oc.CALIBRATION_ARRAYS) | {oc.HELD_NET_FLUX}
    assert provider.sst_at(0.0, np, np.float32) == (None, None)
    _march(provider, state, 25.0, 5)
    arrays = state.physics_state.arrays
    # Intervals starting at 0 s and 300 s lie in the spin-up; three are booked.
    np.testing.assert_allclose(arrays[oc.QCAL_WEIGHT].sum(axis=0), 900.0, rtol=1e-12)
    i0, i1, w = month_weights(START + timedelta(seconds=600.0))
    assert i0 == 9 and i1 == 10
    np.testing.assert_allclose(arrays[oc.QCAL_SUM][i1] / arrays[oc.QCAL_WEIGHT][i1], 25.0,
                               rtol=1e-12)
    assert arrays[oc.QCAL_WEIGHT][i0].max() > 0.0
    assert np.count_nonzero(arrays[oc.QCAL_WEIGHT].sum(axis=(1, 2))) == 2
    block = provider.identity_block()
    assert block["ocean_mode"] == "climatology" and "qflux_calibration" in block
    rows = provider.diagnostics(SimpleNamespace(
        surface=state.surface, physics_state=state.physics_state, time_s=state.time_s))
    assert rows["ocean_qflux_calibration_deposited_s"] == pytest.approx(900.0)
    assert rows["ocean_qflux_calibration_mean_flux_w_m2"] == pytest.approx(25.0)


def test_the_tape_fields(tmp_path):
    t = np.array([[290.0, 0.0], [T_F, 0.0]])
    arrays = {f"physics__{oc.SLAB_TEMPERATURE}": t,
              f"physics__{oc.SLAB_DEPTH}": np.array([[30.0, 0.0], [10.0, 0.0]]),
              f"physics__{oc.SLAB_QFLUX}": np.array([[5.0, 0.0], [-2.0, 0.0]])}
    skin = np.array([[289.0, 300.0], [250.0, 260.0]])
    out = oc.tape_fields(arrays, skin, None)
    np.testing.assert_array_equal(out["SST"], [[290.0, 300.0], [T_F, 260.0]])
    np.testing.assert_array_equal(out["HML"], [[30.0, 0.0], [10.0, 0.0]])
    np.testing.assert_array_equal(out["ARWEN_QFLUX"], [[5.0, 0.0], [-2.0, 0.0]])
    bare = {oc.SLAB_TEMPERATURE: t}
    assert set(oc.tape_fields(bare, skin, np.ones((2, 2), bool))) == {"SST", "HML",
                                                                      "ARWEN_QFLUX"}
    assert oc.tape_fields({"physics__swdown": t}, skin, None) == {}


def test_the_export_writes_the_slab_fields_only_when_the_state_carries_them():
    from arwen_global.wrfout_export import _slab_ocean_planes

    skin = np.full((2, 2), 280.0)
    bundle = SimpleNamespace(surface=SimpleNamespace(temperature_k=skin),
                             physics_state=PhysicsState(arrays={"hfx": skin}))
    assert _slab_ocean_planes(bundle, lambda v: np.asarray(v)) == {}
    bundle.physics_state.arrays[oc.SLAB_TEMPERATURE] = np.full((2, 2), 285.0)
    planes = _slab_ocean_planes(bundle, lambda v: np.asarray(v))
    assert sorted(planes) == ["ARWEN_QFLUX", "HML", "SST"]
    assert all(p.dtype == np.float32 for p in planes.values())


# -- the parser ------------------------------------------------------------------

def _config_text(tmp_path, extra, name="a.toml"):
    from arwen_global.config import load_config

    path = tmp_path / name
    path.write_text(Path(GDAS).read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    return load_config(path)


def test_the_parser_admits_the_slab_and_refuses_what_the_page_refuses(tmp_path):
    clim = tmp_path / "c.npz"
    clim.write_bytes(b"climatology-bytes")
    q = tmp_path / "q.npz"
    q.write_bytes(b"qflux-bytes")
    slab = (f'[ocean]\nmode = "slab"\nclimatology = "{clim.as_posix()}"\n'
            f'slab_qflux = "{q.as_posix()}"\n')
    cfg = _config_text(tmp_path, slab)
    ocean = cfg.ocean
    assert ocean.mode == "slab" and ocean.ice_mode == "thermodynamic"
    assert ocean.sea_ice.new_ice_thickness_m == 0.5
    assert ocean.slab_depth == "woa18" and ocean.slab_relaxation_days == 365.0
    assert ocean.slab_qflux_sha256 == hashlib.sha256(b"qflux-bytes").hexdigest()
    base = _config_text(tmp_path, "", "b.toml").config_hash
    assert cfg.config_hash != base
    for key, value in (("slab_depth", "50.0"), ("slab_relaxation_days", '"none"'),
                       ("slab_depth_bounds_m", "[20.0, 150.0]")):
        other = _config_text(tmp_path, slab + f"{key} = {value}\n", "c.toml")
        assert other.config_hash != cfg.config_hash, key
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / "q.npz").write_bytes(b"qflux-bytes")
    same = _config_text(tmp_path, slab.replace(q.as_posix(), (moved / "q.npz").as_posix()),
                        "d.toml")
    assert same.config_hash == cfg.config_hash
    thermo = _config_text(tmp_path, slab + '[sea_ice]\nmode = "thermodynamic"\n', "e.toml")
    assert thermo.config_hash == cfg.config_hash
    calibration = _config_text(
        tmp_path, f'[ocean]\nmode = "climatology"\nclimatology = "{clim.as_posix()}"\n'
        'anomaly_reference = "start"\nqflux_calibration = true\n', "f.toml")
    plain = _config_text(
        tmp_path, f'[ocean]\nmode = "climatology"\nclimatology = "{clim.as_posix()}"\n'
        'anomaly_reference = "start"\n', "g.toml")
    assert calibration.ocean.qflux_calibration and not plain.ocean.qflux_calibration
    assert calibration.config_hash != plain.config_hash
    refused = (
        (f'[ocean]\nmode = "slab"\nclimatology = "{clim.as_posix()}"\n', "slab_qflux"),
        (f'[ocean]\nmode = "anomaly"\nclimatology = "{clim.as_posix()}"\n'
         f'anomaly_reference = "start"\nslab_depth = 50.0\n', "slab"),
        ('[ocean]\nmode = "hold"\n[sea_ice]\nmode = "thermodynamic"\n', "thermodynamic"),
        (slab + '[sea_ice]\nmode = "anomaly"\n', "prescribe ice"),
        (f'[ocean]\nmode = "anomaly"\nclimatology = "{clim.as_posix()}"\n'
         f'anomaly_reference = "start"\nqflux_calibration = true\n', "climatology"),
        (slab + 'slab_depth = "deep"\n', "woa18"),
        (slab + "slab_depth_bounds_m = [200.0, 10.0]\n", "lower < upper"),
        (slab + 'slab_target = "series"\n', "slab_target"),
        (f'[ocean]\nmode = "slab"\nslab_qflux = "{q.as_posix()}"\n', "climatology"),
    )
    for index, (text, match) in enumerate(refused):
        with pytest.raises((ValueError, FileNotFoundError), match=match):
            _config_text(tmp_path, text, f"r{index}.toml")


def test_no_rule_payload_gains_a_slab_key():
    """OC-1 at the parser: the identity of a run with no slab and no
    calibration is the payload it was, so every config hash written before
    the slab stays as it was (the rule tests pin those payloads' moves)."""
    assert OceanOptions().identity is None
    for mode in ("anomaly", "climatology"):
        payload = OceanOptions(declared=True, implicit=False, mode=mode).identity
        assert not any(key.startswith(("slab_", "qflux_")) for key in payload), mode
        assert "new_ice_thickness_m" in payload and payload["new_ice_thickness_m"] == 1.0


# -- OC-1 and OC-2 on the T3 reference smoke ------------------------------------

def _smoke(tmp_path, monkeypatch, start, ocean_options, precision="float64"):
    import arwen_global.clock as clock
    from arwen_global.config import load_config

    def dated(cfg, provenance):
        return clock.ForecastClock(start, clock.CONFIG_START_TIME, declared_start_time=start)

    monkeypatch.setattr(clock, "resolve_forecast_clock", dated)
    cfg = load_config(SMOKE)
    return replace(cfg, ocean=ocean_options, duration_s=80.0, output_interval_s=40.0,
                   precision=precision)


def _smoke_climatology(tmp_path, cfg_truncation, dealias):
    grid = GaussianGrid.create(int(cfg_truncation), dealias_factor=float(dealias))
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    lat = np.asarray(grid.latitude_deg)[None, :, None]
    months = np.arange(12)[:, None, None]
    sst = 282.0 + 6.0 * np.cos(np.radians(lat)) + 0.8 * months + np.zeros((12, nlat, nlon))
    ice = np.clip((np.abs(lat) - 40.0) / 30.0, 0.0, 1.0) + np.zeros((12, nlat, nlon))
    return grid, sst, ice


def _same_checkpoints(a, b):
    from arwen_global.checkpoint import read_checkpoint

    meta_a, arrays_a = read_checkpoint(a)
    meta_b, arrays_b = read_checkpoint(b)
    assert meta_a["physics_metadata"] == meta_b["physics_metadata"]
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    return meta_a, arrays_a


def test_with_the_ocean_off_the_smoke_carries_nothing_of_it(tmp_path):
    from arwen_global.checkpoint import read_checkpoint
    from arwen_global.config import load_config
    from arwen_global.runner import run

    run(load_config(SMOKE), tmp_path / "off")
    meta, arrays = read_checkpoint(tmp_path / "off" / "arwen_global_step00000004.npz")
    assert not any("ocean_" in name and "exchange" not in name for name in arrays), \
        sorted(arrays)
    assert oc.METADATA_KEY not in meta["physics_metadata"]
    assert f"physics__{oc.HELD_NET_FLUX}" not in arrays


# float32 is the card runs' precision: a restart there read the slab's float64
# arrays in it and was not the straight run (the first coupled card run).
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_restarted_slab_run_reproduces_the_straight_run_bit_for_bit(
        tmp_path, monkeypatch, parts, precision):
    from arwen_global import ocean_qflux
    from arwen_global.runner import run

    # 40 s before January's mid-month instant, at an hour that is not 00:
    # the restart at step 4 sits on the instant, inside the ice record that
    # began at 30 s.
    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    import arwen_global.clock as clock
    from arwen_global.config import load_config

    base = load_config(SMOKE)
    grid, sst, ice = _smoke_climatology(tmp_path, base.truncation, base.dealias_factor)
    clim = _climatology(tmp_path, grid, sst=sst, ice=ice)
    qpath = tmp_path / "q.npz"
    qpath.write_bytes(b"test q-flux")
    qsha = hashlib.sha256(b"test q-flux").hexdigest()
    shape = (int(grid.nlat), int(grid.nlon))
    monkeypatch.setattr(ocean_qflux, "load_qflux",
                        lambda path: FakeQFlux(shape, flux=-20.0, sha256=qsha))
    options = OceanOptions(
        declared=True, implicit=False, mode="slab", climatology=str(clim.path),
        climatology_sha256=clim.sha256, slab_qflux=str(qpath), slab_qflux_sha256=qsha,
        slab_depth=10.0, slab_relaxation_days=30.0, anomaly_reference="start",
        anomaly_truncation=None,
        sea_ice=SeaIceOptions(mode="match", record_interval_hours=30.0 / 3600.0,
                              new_ice_thickness_m=0.5))
    cfg = _smoke(tmp_path, monkeypatch, start, options, precision)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = run(cfg, full)
    assert result["surface_boundary"]["ocean_mode"] == "slab"
    assert result["surface_boundary"]["sea_ice_mode"] == "thermodynamic"
    assert result["surface_update"]["sst_steps"] == 8
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    meta, arrays = _same_checkpoints(full / "arwen_global_step00000008.npz",
                                     resumed / "arwen_global_step00000008.npz")
    assert meta["physics_metadata"]["surface_boundary"]["ocean_mode"] == "slab"
    assert meta["physics_metadata"][oc.METADATA_KEY]["advances"] == 7
    from arwen_global.checkpoint import read_checkpoint

    _meta0, arrays0 = read_checkpoint(full / "arwen_global_step00000000.npz")
    t0 = arrays0[f"physics__{oc.SLAB_TEMPERATURE}"]
    t8 = arrays[f"physics__{oc.SLAB_TEMPERATURE}"]
    ocean = t0 > 0.0
    assert ocean.any() and not np.array_equal(t0[ocean], t8[ocean])
    # The skin on open ocean is the slab's T (written before step 8, then
    # stepped once by the reference suite's own skin update).
    residual = np.abs(arrays[f"physics__{oc.ENERGY_STORED}"]
                      - arrays[f"physics__{oc.ENERGY_SOURCES}"])
    scale = np.abs(arrays[f"physics__{oc.ENERGY_SOURCES}"]).max()
    assert residual.max() <= 1e-9 * max(scale, 1.0) + 1e-3
    rows = [line for line in (full / "diagnostics.jsonl").read_text().splitlines() if line]
    assert any('"ocean_energy_residual_j_m2"' in line for line in rows)
    # A restart under another ocean is refused before it runs.
    other = replace(cfg, ocean=replace(cfg.ocean, slab_depth=20.0))
    with pytest.raises(ValueError):
        run(other, tmp_path / "other", restart=full / "arwen_global_step00000004.npz")


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_restarted_calibration_reproduces_the_straight_one_bit_for_bit(
        tmp_path, monkeypatch, parts, precision):
    from arwen_global.runner import run
    from arwen_global.config import load_config

    start = datetime(2026, 1, 16, 11, 59, 20, tzinfo=timezone.utc)
    base = load_config(SMOKE)
    grid, sst, ice = _smoke_climatology(tmp_path, base.truncation, base.dealias_factor)
    clim = _climatology(tmp_path, grid, sst=sst, ice=ice)
    options = OceanOptions(
        declared=True, implicit=False, mode="climatology", climatology=str(clim.path),
        climatology_sha256=clim.sha256, anomaly_reference="start", anomaly_truncation=None,
        qflux_calibration=True, qflux_calibration_spinup_days=20.0 / DAY,
        sea_ice=SeaIceOptions(mode="match", record_interval_hours=30.0 / 3600.0))
    cfg = _smoke(tmp_path, monkeypatch, start, options, precision)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = run(cfg, full)
    assert "coupler" in result["surface_update"]
    run(cfg, resumed, restart=full / "arwen_global_step00000004.npz")
    meta, arrays = _same_checkpoints(full / "arwen_global_step00000008.npz",
                                     resumed / "arwen_global_step00000008.npz")
    assert "qflux_calibration" in meta["physics_metadata"]["surface_boundary"]
    weight = arrays[f"physics__{oc.QCAL_WEIGHT}"]
    # Steps 1..7 advanced; the intervals starting at 0 s and 10 s are spin-up.
    assert weight.sum(axis=0).max() == pytest.approx(50.0)
    # Across January's mid-month instant the deposits go to December and
    # January before it and to January and February after it.
    months = np.nonzero(weight.sum(axis=(1, 2)))[0].tolist()
    assert months == [0, 1, 11]
    per_month = weight.sum(axis=(1, 2)) / weight.sum(axis=0).max()
    assert per_month[0] > 0.9 * per_month.sum()
    # A plain climatology run may not resume a calibration's checkpoint.
    plain = replace(cfg, ocean=replace(cfg.ocean, qflux_calibration=False))
    with pytest.raises(ValueError):
        run(plain, tmp_path / "plain", restart=full / "arwen_global_step00000004.npz")
