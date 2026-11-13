"""The slab mixed layer (arwen_global.ocean_slab) on known answers.

Gate OC-5's slab parts and the known answer of gate OC-3, each the number
a closed form gives:

* a constant flux into a fixed depth moves T linearly, n dt F / (rho c h);
* restoring alone decays the departure as (1 - dt / tau)^n;
* the freezing hold returns T_f and the withheld energy, and the energy
  stored plus the energy withheld is the energy that came in;
* a column driven by exactly F_cal follows the climatology's SST to 1e-9 K
  at the mid-month instants, on the climatology's own varying depth;
* the depth, the tendency, the open-water balance, the Q-flux and the
  cold-start temperature, each against its formula.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from arwen_global.ocean_qflux import QFluxFile
from arwen_global.ocean_slab import (
    FREEZING_K, LATENT_HEAT_VAPORIZATION_J_KG, RHO_CP_SEAWATER_J_M3_K,
    STEFAN_BOLTZMANN_W_M2_K4, SlabParameters, climatology_tendency, initial_temperature,
    mixed_layer_depth, month_bracket, open_water_flux, qflux, restoring_flux, slab_step,
)
from arwen_global.sea_ice_thermo import ICE_FUSION_J_M3
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.surface_climatology import (
    interpolate, load_climatology, mid_month, month_weights, write_climatology,
)

DT = 300.0
START = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
RHO_C = 1026.0 * 3990.0


def _climatology(tmp_path, grid, *, ocean=None, mld=True):
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    lat = np.asarray(grid.latitude_deg)[None, :, None]
    lon = np.asarray(grid.longitude_deg)[None, None, :]
    month = np.arange(12, dtype=np.float64)[:, None, None]
    sst = 285.0 + 10.0 * np.cos(np.radians(lat)) + 2.0 * np.sin(2 * np.pi * month / 12) \
        + 0.5 * np.cos(np.radians(lon)) + np.zeros((12, nlat, nlon))
    ice = np.zeros((12, nlat, nlon))
    # From about 4 m at the equator to about 290 m at the poles, so both
    # bounds clamp.
    depth = (3.0 + 250.0 * np.sin(np.radians(lat)) ** 2) \
        * (1.0 + 0.3 * np.cos(2 * np.pi * month / 12)) + np.zeros((12, nlat, nlon))
    if ocean is None:
        ocean = np.ones((nlat, nlon), dtype=bool)
    planes = {"sst_k": np.where(ocean[None], sst, np.nan),
              "sea_ice_fraction": np.where(ocean[None], ice, np.nan)}
    if mld:
        planes["mixed_layer_depth_m"] = np.where(ocean[None], depth, np.nan)
    path = tmp_path / "clim.npz"
    write_climatology(path, planes, ocean=ocean, latitude_deg=grid.latitude_deg,
                      longitude_deg=grid.longitude_deg,
                      provenance={"schema": "gpuwm.arwen-global-surface-climatology/v1",
                                  "test": True})
    return load_climatology(path)


def test_the_heat_capacity_is_the_design_number():
    assert RHO_CP_SEAWATER_J_M3_K == RHO_C
    assert abs(RHO_CP_SEAWATER_J_M3_K - 4.094e6) < 1.0e3
    assert FREEZING_K == 271.35


def test_a_constant_flux_into_a_fixed_depth_moves_t_linearly():
    t0 = np.array([280.0, 290.0, 300.0])
    depth = np.array([10.0, 50.0, 200.0])
    flux = np.array([100.0, -40.0, 7.5])
    t = t0.copy()
    for _ in range(288):
        t, deficit = slab_step(temperature_k=t, depth_m=depth, net_flux_w_m2=flux,
                               dt_s=DT, xp=np)
        assert not deficit.any()
    expected = t0 + 288 * DT * flux / (RHO_C * depth)
    np.testing.assert_allclose(t, expected, rtol=0.0, atol=1.0e-10)
    # 100 W/m2 for a day into 10 m: 0.211 K, the design page's arithmetic scale.
    assert abs((t[0] - t0[0]) - 8.64e6 / 4.0937e7) < 1.0e-4


def test_restoring_alone_decays_as_one_minus_dt_over_tau_to_the_n():
    target = np.array([285.0, 285.0])
    t0 = np.array([280.0, 291.0])
    depth = np.array([30.0, 30.0])
    days = 5.0
    t = t0.copy()
    for _ in range(1000):
        r = restoring_flux(temperature_k=t, target_k=target, depth_m=depth,
                           relaxation_days=days, xp=np)
        t, _ = slab_step(temperature_k=t, depth_m=depth, net_flux_w_m2=r, dt_s=DT, xp=np)
    expected = target + (t0 - target) * (1.0 - DT / (days * 86400.0)) ** 1000
    np.testing.assert_allclose(t, expected, rtol=0.0, atol=1.0e-11)


def test_restoring_is_off_with_none_and_zero_off_the_ocean():
    r = restoring_flux(temperature_k=np.array([280.0, 0.0]),
                       target_k=np.array([290.0, np.nan]), depth_m=np.array([50.0, 0.0]),
                       relaxation_days=None, xp=np)
    np.testing.assert_array_equal(r, [0.0, 0.0])
    r = restoring_flux(temperature_k=np.array([280.0, 0.0]),
                       target_k=np.array([290.0, np.nan]), depth_m=np.array([50.0, 0.0]),
                       relaxation_days=365.0, xp=np)
    # 50 m, 365 days: 6.49 W m-2 per kelvin, the page's figure.
    assert abs(r[0] / 10.0 - 6.49) < 0.01
    assert r[1] == 0.0


def test_the_freezing_hold_returns_t_f_and_the_withheld_energy():
    t0 = np.array([271.4, 271.36, 275.0])
    depth = np.array([10.0, 20.0, 10.0])
    flux = np.array([-2000.0, -5000.0, -100.0])
    dt = 3600.0
    new, deficit = slab_step(temperature_k=t0, depth_m=depth, net_flux_w_m2=flux,
                             dt_s=dt, xp=np)
    unheld = t0 + dt * flux / (RHO_C * depth)
    assert unheld[0] < FREEZING_K and unheld[1] < FREEZING_K and unheld[2] > FREEZING_K
    np.testing.assert_array_equal(new[:2], [FREEZING_K, FREEZING_K])
    assert new[2] == unheld[2]
    np.testing.assert_allclose(deficit[:2], RHO_C * depth[:2] * (FREEZING_K - unheld[:2]),
                               rtol=1.0e-12)
    assert deficit[2] == 0.0 and (deficit >= 0.0).all()
    # Energy in = energy stored in the water - energy handed to the ice.
    np.testing.assert_allclose(RHO_C * depth * (new - t0) - deficit, dt * flux, rtol=1.0e-9)


def test_a_column_of_zero_depth_is_returned_as_it_came():
    new, deficit = slab_step(temperature_k=np.array([0.0, 280.0]),
                             depth_m=np.array([0.0, 10.0]),
                             net_flux_w_m2=np.array([-500.0, 0.0]), dt_s=DT, xp=np)
    np.testing.assert_array_equal(new, [0.0, 280.0])
    np.testing.assert_array_equal(deficit, [0.0, 0.0])


def test_the_open_water_balance_is_its_formula():
    out = open_water_flux(swdown=np.array([400.0]), albedo=0.08, glw=np.array([350.0]),
                          emissivity=0.98, skin_k=np.array([290.0]), hfx=np.array([15.0]),
                          qfx=np.array([4.0e-5]), xp=np)
    expected = (400.0 * 0.92 + 0.98 * (350.0 - STEFAN_BOLTZMANN_W_M2_K4 * 290.0 ** 4)
                - 15.0 - LATENT_HEAT_VAPORIZATION_J_KG * 4.0e-5)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [expected], rtol=1.0e-14)
    assert LATENT_HEAT_VAPORIZATION_J_KG == 2.5e6


def test_the_qflux_is_its_formula():
    q = qflux(depth_m=np.array([50.0]), sst_rate_k_s=np.array([1.0e-7]),
              ice_fraction_rate_s=np.array([2.0e-8]), ice_reference_thickness_m=np.array([2.0]),
              calibration_flux_w_m2=np.array([-30.0]), xp=np)
    expected = RHO_C * 50.0 * 1.0e-7 - ICE_FUSION_J_M3 * 2.0 * 2.0e-8 + 30.0
    np.testing.assert_allclose(q, [expected], rtol=1.0e-14)


def test_month_bracket_is_the_month_weights_rule():
    for when in (START, datetime(2026, 11, 16, tzinfo=timezone.utc),
                 datetime(2026, 12, 31, 23, tzinfo=timezone.utc),
                 datetime(2028, 2, 15, 12, tzinfo=timezone.utc)):
        i0, i1, t0, t1 = month_bracket(when)
        j0, j1, w = month_weights(when)
        assert (i0, i1) == (j0, j1)
        assert t0 <= when < t1
        assert w == (when - t0).total_seconds() / (t1 - t0).total_seconds()
    # The mid-month instants fall at 00 or 12 UTC, on a 300 s step boundary.
    for month in range(1, 13):
        assert mid_month(2026, month).timestamp() % 43200.0 == 0.0


def test_the_depth_is_the_climatology_clamped_with_gaps_at_the_lower_bound(tmp_path):
    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid)
    ocean = np.ones(shape, dtype=bool)
    ocean[0, :5] = False
    params = SlabParameters(depth_bounds_m=(10.0, 120.0))
    h = mixed_layer_depth(clim, START, params, ocean, np)
    raw = interpolate(clim.mixed_layer_depth_m.astype(np.float64), START)
    assert h.dtype == np.float64
    np.testing.assert_array_equal(h[ocean], np.clip(raw, 10.0, 120.0)[ocean])
    np.testing.assert_array_equal(h[~ocean], 0.0)
    assert (raw < 10.0).any() and (raw > 120.0).any()
    # A gap in the climatology on a run's ocean column takes the lower bound.
    gap = clim.mixed_layer_depth_m.copy()
    gap[:, 3, 7] = np.nan
    clim.mixed_layer_depth_m = gap
    assert mixed_layer_depth(clim, START, params, ocean, np)[3, 7] == 10.0
    # A fixed depth is that depth everywhere on the ocean, unclamped.
    fixed = mixed_layer_depth(clim, START, SlabParameters(depth=250.0), ocean, np)
    np.testing.assert_array_equal(fixed, np.where(ocean, 250.0, 0.0))


def test_woa18_depth_needs_the_mld_plane(tmp_path):
    grid = GaussianGrid.create(21)
    clim = _climatology(tmp_path, grid, mld=False)
    with pytest.raises(ValueError, match="--with-mld"):
        mixed_layer_depth(clim, START, SlabParameters(),
                          np.ones((int(grid.nlat), int(grid.nlon)), dtype=bool), np)


def test_the_parameters_refuse_what_the_switch_does_not_offer():
    assert SlabParameters(depth=50).depth == 50.0
    assert SlabParameters(depth_bounds_m=[10, 200]).depth_bounds_m == (10.0, 200.0)
    for bad in ({"depth": "deep"}, {"depth": -1.0}, {"depth": float("nan")},
                {"depth_bounds_m": (0.0, 10.0)}, {"depth_bounds_m": (50.0, 10.0)},
                {"relaxation_days": 0.0}, {"target": "series"}):
        with pytest.raises(ValueError):
            SlabParameters(**bad)


def test_the_tendency_is_the_slope_between_mid_month_instants(tmp_path):
    grid = GaussianGrid.create(21)
    clim = _climatology(tmp_path, grid)
    ds, di = climatology_tendency(clim, START, np)
    i0, i1, t0, t1 = month_bracket(START)
    span = (t1 - t0).total_seconds()
    expected = (clim.sst_k[i1].astype(np.float64) - clim.sst_k[i0].astype(np.float64)) / span
    np.testing.assert_array_equal(ds, expected)
    np.testing.assert_array_equal(di, 0.0)
    # The slope times an interval inside the bracket is the interpolated change.
    later = START + timedelta(hours=30)
    change = (interpolate(clim.sst_k.astype(np.float64), later)
              - interpolate(clim.sst_k.astype(np.float64), START))
    np.testing.assert_allclose(ds * 30 * 3600.0, change, rtol=0.0, atol=1.0e-9)


def test_the_tendency_reads_zero_off_the_climatology_ocean(tmp_path):
    grid = GaussianGrid.create(21)
    ocean = np.ones((int(grid.nlat), int(grid.nlon)), dtype=bool)
    ocean[:4] = False
    clim = _climatology(tmp_path, grid, ocean=ocean)
    ds, di = climatology_tendency(clim, START, np)
    assert np.isfinite(ds).all() and np.isfinite(di).all()
    np.testing.assert_array_equal(ds[:4], 0.0)


def test_the_cold_start_temperature():
    skin = np.array([290.0, 265.0, 280.0, 300.0, np.nan])
    ocean = np.array([True, True, True, False, True])
    ice = np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    t = initial_temperature(skin, ocean, ice, np)
    np.testing.assert_array_equal(t, [290.0, FREEZING_K, FREEZING_K, 0.0, FREEZING_K])


def test_a_column_driven_by_exactly_f_cal_follows_the_climatology(tmp_path):
    """OC-3's known answer: F_open = F_cal on every step, so the net flux is
    rho c h dC_s/dt and T follows C_s; at the mid-month instants of 16
    November 00 UTC and 16 December 12 UTC it equals the month's plane to
    1e-9 K, on the climatology's own (varying, clamped) depth."""
    grid = GaussianGrid.create(21)
    shape = (int(grid.nlat), int(grid.nlon))
    clim = _climatology(tmp_path, grid)
    rng = np.random.default_rng(5)
    f_cal = rng.normal(0.0, 60.0, (12, *shape))
    qfile = QFluxFile(path=Path("synthetic"), sha256="0" * 64, flux_w_m2=f_cal,
                      coverage_s=np.ones((12, *shape)), latitude_deg=grid.latitude_deg,
                      longitude_deg=grid.longitude_deg, provenance={})
    ocean = np.ones(shape, dtype=bool)
    params = SlabParameters()
    h_ref = np.ones(shape)
    when = START
    t = interpolate(clim.sst_k.astype(np.float64), START)
    marks = {mid_month(2026, 11): 10, mid_month(2026, 12): 11}
    seen = 0
    while when < mid_month(2026, 12):
        h = mixed_layer_depth(clim, when, params, ocean, np)
        ds, di = climatology_tendency(clim, when, np)
        fc = qfile.at(when, np)
        q = qflux(depth_m=h, sst_rate_k_s=ds, ice_fraction_rate_s=di,
                  ice_reference_thickness_m=h_ref, calibration_flux_w_m2=fc, xp=np)
        t, deficit = slab_step(temperature_k=t, depth_m=h, net_flux_w_m2=fc + q, dt_s=DT,
                               xp=np)
        assert not deficit.any()
        when = when + timedelta(seconds=DT)
        if when in marks:
            seen += 1
            plane = clim.sst_k[marks[when]].astype(np.float64)
            assert float(np.max(np.abs(t - plane))) < 1.0e-9
    assert seen == 2
