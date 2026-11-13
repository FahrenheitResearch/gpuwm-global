"""Thermodynamic sea ice (arwen_global.sea_ice_thermo) on known answers.

Each test writes down the number the zero-layer balance gives in closed
form and asks the module for it (gate OC-5, the ice parts):

* freeze-up: a deficit on open water forms ice of the new-ice thickness;
* basal growth: the conduction the column draws from its base, less the
  ocean's heat, grows the volume linearly over n steps;
* top melt: the surplus at the melting point melts the volume and the
  square-root rule lowers the area with it;
* a negative ocean flux under a partial pack covers open area at the
  new-ice thickness, and the area stops at one;
* a complete melt sets volume and area to zero and returns the energy it
  did not need to the mixed layer;
* an ice-free column with no deficit is unchanged whatever its ocean flux;
* the energy identity stored + returned = -E on random columns;
* the base conduction and the top-melt surplus against the column's own
  layout, and the top node frozen_surface.column_step caps at the melting
  point is read as at the melting point;
* the cold-start ice, the thickness and the reference-thickness plane.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arwen_global import sea_ice_thermo as sit
from arwen_global.physics import frozen_surface as fs

RHO_LF = 917.0 * 3.337e5
DAY = 86400.0


def _step(**kw):
    base = dict(
        fraction=np.zeros(1), volume_m=np.zeros(1), ocean_flux_w_m2=np.zeros(1),
        basal_conduction_w_m2=np.zeros(1), top_melt_w_m2=np.zeros(1),
        freeze_deficit_j_m2=np.zeros(1), dt_s=300.0, new_ice_thickness_m=0.5, xp=np,
    )
    for key, value in kw.items():
        base[key] = np.atleast_1d(np.asarray(value, dtype=np.float64)) if key not in (
            "dt_s", "new_ice_thickness_m", "xp") else value
    return sit.ice_step(**base)


def test_constants_are_the_design_page_and_the_column_values() -> None:
    assert sit.ICE_FUSION_J_M3 == RHO_LF
    assert abs(sit.ICE_FUSION_J_M3 - 3.06e8) < 0.01e8
    assert sit.MELT_POINT_K == fs.MELT_POINT_K
    assert sit.FREEZING_K == 271.35
    assert sit.DEFAULT_NEW_ICE_THICKNESS_M == 0.5
    assert sit.ICE_REFERENCE_THICKNESS_M == {"nh": 2.0, "sh": 1.0}


def test_freeze_up_forms_new_ice_at_the_new_ice_thickness() -> None:
    deficit = RHO_LF * 0.1  # 0.1 m of ice per m2 of cell
    out = _step(freeze_deficit_j_m2=deficit, ocean_flux_w_m2=-250.0)
    assert out.volume_m[0] == pytest.approx(0.1, rel=1e-14)
    assert out.fraction[0] == pytest.approx(0.2, rel=1e-14)
    assert sit.thickness(out.fraction, out.volume_m, np)[0] == pytest.approx(0.5, rel=1e-14)
    assert out.mixed_layer_heat_j_m2[0] == 0.0
    assert out.stored_j_m2[0] == pytest.approx(-deficit, rel=1e-14)
    # a thinner new-ice thickness covers more area with the same volume
    thin = _step(freeze_deficit_j_m2=deficit, new_ice_thickness_m=0.25)
    assert thin.fraction[0] == pytest.approx(0.4, rel=1e-14)


def test_basal_growth_is_linear_over_n_steps() -> None:
    a, v = np.array([1.0, 0.5]), np.array([1.0, 0.5])
    g_up, f_o, dt, n = 20.0, 2.0, 1800.0, 48
    for _ in range(n):
        out = _step(fraction=a, volume_m=v, basal_conduction_w_m2=[g_up, g_up],
                    ocean_flux_w_m2=[f_o, 0.0], dt_s=dt)
        a, v = out.fraction, out.volume_m
    expect_full = 1.0 + n * dt * (1.0 * g_up - f_o) / RHO_LF
    expect_half = 0.5 + n * dt * (0.5 * g_up) / RHO_LF
    assert v[0] == pytest.approx(expect_full, rel=1e-13)
    assert v[1] == pytest.approx(expect_half, rel=1e-13)
    # base growth thickens the pack; it does not open or close area
    assert a[0] == 1.0 and a[1] == 0.5


def test_top_melt_lowers_area_by_the_square_root_rule() -> None:
    a0, v0, m_top, dt = 1.0, 2.0, 100.0, 3600.0
    out = _step(fraction=a0, volume_m=v0, top_melt_w_m2=m_top, dt_s=dt)
    v1 = v0 - dt * m_top / RHO_LF
    assert out.volume_m[0] == pytest.approx(v1, rel=1e-14)
    assert out.fraction[0] == pytest.approx(a0 * math.sqrt(v1 / v0), rel=1e-14)
    assert out.stored_j_m2[0] == pytest.approx(dt * m_top, rel=1e-9)


def test_ocean_heat_melts_a_partial_pack_by_the_square_root_rule() -> None:
    a0, v0, f_o = 0.8, 0.8, 50.0
    out = _step(fraction=a0, volume_m=v0, ocean_flux_w_m2=f_o, dt_s=DAY)
    v1 = v0 - DAY * f_o / RHO_LF
    assert out.volume_m[0] == pytest.approx(v1, rel=1e-14)
    assert out.fraction[0] == pytest.approx(a0 * math.sqrt(v1 / v0), rel=1e-14)
    # the thickness falls with the square root of the volume
    assert sit.thickness(out.fraction, out.volume_m, np)[0] == pytest.approx(
        math.sqrt(v1 / v0), rel=1e-13)


def test_negative_ocean_flux_under_a_partial_pack_covers_open_area() -> None:
    out = _step(fraction=0.5, volume_m=0.5, ocean_flux_w_m2=-100.0, dt_s=3600.0)
    freezing = 3600.0 * 100.0
    assert out.volume_m[0] == pytest.approx(0.5 + freezing / RHO_LF, rel=1e-14)
    assert out.fraction[0] == pytest.approx(0.5 + freezing / (RHO_LF * 0.5), rel=1e-14)


def test_area_stops_at_one_and_the_rest_thickens() -> None:
    deficit = RHO_LF * 0.5 * 0.5  # would cover 0.5 of the cell at 0.5 m
    out = _step(fraction=0.9, volume_m=1.8, freeze_deficit_j_m2=deficit)
    assert out.fraction[0] == sit.MAX_FRACTION
    assert out.volume_m[0] == pytest.approx(2.05, rel=1e-14)


def test_complete_melt_returns_the_surplus_to_the_mixed_layer() -> None:
    a0, v0, f_o = 0.3, 0.03, 500.0
    out = _step(fraction=a0, volume_m=v0, ocean_flux_w_m2=f_o, top_melt_w_m2=40.0, dt_s=DAY)
    used = RHO_LF * v0
    supplied = DAY * (f_o + a0 * 40.0)
    assert out.volume_m[0] == 0.0
    assert out.fraction[0] == 0.0
    assert out.stored_j_m2[0] == pytest.approx(used, rel=1e-14)
    assert out.mixed_layer_heat_j_m2[0] == pytest.approx(supplied - used, rel=1e-13)


def test_an_ice_free_column_with_no_deficit_is_unchanged() -> None:
    out = _step(fraction=0.0, volume_m=0.0, ocean_flux_w_m2=-300.0,
                basal_conduction_w_m2=50.0, top_melt_w_m2=80.0)
    assert out.fraction[0] == 0.0 and out.volume_m[0] == 0.0
    assert out.mixed_layer_heat_j_m2[0] == 0.0 and out.stored_j_m2[0] == 0.0
    warm = _step(fraction=0.0, volume_m=0.0, ocean_flux_w_m2=300.0)
    assert warm.mixed_layer_heat_j_m2[0] == 0.0 and warm.volume_m[0] == 0.0


def test_energy_identity_on_random_columns() -> None:
    rng = np.random.default_rng(20261112)
    n = 4096
    a = rng.uniform(0.0, 1.0, n) * (rng.uniform(size=n) > 0.3)
    v = a * rng.uniform(0.05, 3.0, n)
    f_o = rng.normal(0.0, 300.0, n)
    g_up = rng.uniform(0.0, 60.0, n)
    m_top = rng.uniform(0.0, 200.0, n) * (rng.uniform(size=n) > 0.5)
    deficit = rng.uniform(0.0, 5e6, n) * (rng.uniform(size=n) > 0.6)
    dt = 7200.0
    out = sit.ice_step(fraction=a, volume_m=v, ocean_flux_w_m2=f_o,
                       basal_conduction_w_m2=g_up, top_melt_w_m2=m_top,
                       freeze_deficit_j_m2=deficit, dt_s=dt, new_ice_thickness_m=0.5, xp=np)
    ice = v > 0.0
    minus_e = np.where(ice, dt * (f_o + a * (m_top - g_up)), 0.0) - deficit
    residual = out.stored_j_m2 + out.mixed_layer_heat_j_m2 - minus_e
    scale = np.abs(np.where(ice, dt * f_o, 0.0)) + np.abs(np.where(ice, dt * a * m_top, 0.0)) \
        + np.abs(np.where(ice, dt * a * g_up, 0.0)) + deficit + RHO_LF * v
    assert np.all(np.abs(residual) <= 1e-12 * scale + 1e-3)
    # the state stays physical: 0 <= A <= 1, V >= 0, area exactly where volume is
    assert np.all((out.fraction >= 0.0) & (out.fraction <= 1.0))
    assert np.all(out.volume_m >= 0.0)
    assert np.array_equal(out.fraction > 0.0, out.volume_m > 0.0)
    assert np.all(out.mixed_layer_heat_j_m2 >= 0.0)
    # heat comes back only where the whole volume went
    assert np.all(out.volume_m[out.mixed_layer_heat_j_m2 > 0.0] == 0.0)
    for name in out._fields:
        assert getattr(out, name).dtype == np.float64


def test_float32_input_is_widened_and_bad_thickness_refused() -> None:
    out = sit.ice_step(
        fraction=np.float32([0.5]), volume_m=np.float32([0.5]),
        ocean_flux_w_m2=np.float32([1.0]), basal_conduction_w_m2=np.float32([0.0]),
        top_melt_w_m2=np.float32([0.0]), freeze_deficit_j_m2=np.float32([0.0]),
        dt_s=300.0, new_ice_thickness_m=0.5, xp=np)
    assert out.volume_m.dtype == np.float64
    with pytest.raises(ValueError):
        _step(new_ice_thickness_m=0.0)


def _bare_layout(ice_m):
    dz = fs.sea_ice_layer_thickness(np.float32([ice_m]), np.float32([0.0]), np)
    return dz.astype(np.float64)[:, 0]


def test_basal_conduction_is_the_column_base_conductance() -> None:
    # bare ice of 2 m: skin node 0.1 m, the rest 1.9/3 m each, all ice
    dz = _bare_layout(2.0)
    assert dz[0] == pytest.approx(0.1, rel=1e-6)
    assert dz[3] == pytest.approx(1.9 / 3.0, rel=1e-6)
    layers = np.array([[250.0], [255.0], [258.0], [260.0]])
    g = sit.basal_conduction(layers_k=layers, thickness_m=np.array([2.0]),
                             snow_depth_m=np.array([0.0]), xp=np)
    k_ice = float(np.float32(2.03))
    expect = (271.36 - 260.0) * k_ice / (0.5 * dz[3])
    assert g[0] == pytest.approx(expect, rel=1e-12)
    assert g.dtype == np.float64
    # snow of 0.3 m over 1 m of ice: the nodes follow the snow, the bottom
    # node is half the ice, ice conductivity
    g_snow = sit.basal_conduction(layers_k=layers, thickness_m=np.array([1.0]),
                                  snow_depth_m=np.array([0.3]), xp=np)
    assert g_snow[0] == pytest.approx((271.36 - 260.0) * 2.03 / 0.25, rel=1e-6)
    # a bottom node at the base temperature draws nothing
    flat = np.full((4, 1), float(np.float32(271.36)))
    g0 = sit.basal_conduction(layers_k=flat, thickness_m=np.array([2.0]),
                              snow_depth_m=np.array([0.0]), xp=np)
    assert abs(g0[0]) < 1e-3


def test_top_melt_surplus_is_the_balance_at_the_melting_point() -> None:
    tm = float(np.float32(273.15))
    layers = np.array([[tm, 272.0], [272.0, 272.0], [271.8, 271.8], [271.5, 271.5]])
    kw = dict(thickness_m=np.array([2.0, 2.0]), snow_depth_m=np.array([0.0, 0.0]),
              swdown=np.array([600.0, 600.0]), glw=np.array([300.0, 300.0]),
              ice_hfx=np.array([-10.0, -10.0]), ice_qfx=np.array([1e-6, 1e-6]), xp=np)
    m = sit.top_melt_surplus(layers_k=layers, **kw)
    dz = _bare_layout(2.0)
    k_ice = float(np.float32(2.03))
    g01 = 1.0 / (0.5 * dz[0] / k_ice + 0.5 * dz[1] / k_ice)
    balance = (600.0 * 0.35 + 0.98 * 300.0 - 0.98 * 5.670374419e-8 * tm ** 4
               + 10.0 - 2.834e6 * 1e-6 - g01 * (tm - 272.0))
    assert balance > 0.0
    assert m[0] == pytest.approx(balance, rel=1e-12)
    # a top node below the melting point discards nothing
    assert m[1] == 0.0
    # a negative balance at the melting point is no surplus
    night = sit.top_melt_surplus(layers_k=layers, **{**kw, "swdown": np.zeros(2)})
    assert night[0] == 0.0


def test_a_node_column_step_capped_is_read_as_at_the_melting_point() -> None:
    dz = fs.sea_ice_layer_thickness(np.float32([2.0]), np.float32([0.0]), np)
    layers = np.float32([[272.9], [272.0], [271.8], [271.5]])
    forcing = dict(swdown=np.float32([900.0]), glw=np.float32([350.0]),
                   hfx=np.float32([-30.0]), qfx=np.float32([0.0]))
    new = fs.column_step(
        layers=layers, dz=dz, snow_depth_m=np.float32([0.0]), bottom_k=np.float32([271.36]),
        bottom_extra_resistance=np.float32([0.0]), albedo=np.float32([fs.SEA_ICE_ALBEDO]),
        emissivity=np.float32([fs.SEA_ICE_EMISSIVITY]), dt_s=300.0, xp=np, **forcing,
    )
    assert new[0, 0] == np.float32(fs.MELT_POINT_K)
    m = sit.top_melt_surplus(
        layers_k=new.astype(np.float64), thickness_m=np.array([2.0]),
        snow_depth_m=np.array([0.0]), swdown=forcing["swdown"], glw=forcing["glw"],
        ice_hfx=forcing["hfx"], ice_qfx=forcing["qfx"], xp=np,
    )
    assert m[0] > 0.0


def test_initial_ice_thickness_and_reference_plane() -> None:
    fraction = np.array([1.2, 0.5, 0.3, np.nan, 0.4, 0.0])
    thick = np.array([1.5, 0.0, 2.0, 1.0, np.nan, 3.0])
    ocean = np.array([True, True, False, True, True, True])
    a, v = sit.initial_ice(fraction, thick, ocean, 0.5, np)
    np.testing.assert_array_equal(a, [1.0, 0.5, 0.0, 0.0, 0.4, 0.0])
    np.testing.assert_allclose(v, [1.5, 0.25, 0.0, 0.0, 0.2, 0.0], rtol=1e-15)
    assert a.dtype == np.float64 and v.dtype == np.float64
    h = sit.thickness(a, v, np)
    np.testing.assert_allclose(h, [1.5, 0.5, 0.0, 0.0, 0.5, 0.0], rtol=1e-15)
    plane = sit.ice_reference_thickness(np.array([60.0, 10.0, -10.0, -60.0]), 3, np)
    assert plane.shape == (4, 3) and plane.dtype == np.float64
    np.testing.assert_array_equal(plane[:, 0], [2.0, 2.0, 1.0, 1.0])
    np.testing.assert_array_equal(plane[:, 2], plane[:, 0])
