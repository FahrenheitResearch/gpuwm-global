"""Package 3.2 of the other-worlds track: the re-balanced rotation start.

The rule (``arwen_global.world.rotation_start``) keeps the analysis's mass
field and gives it the winds that hold the same force against the world's
rotation.  These gates pin the rule's closed forms on arrays, its frame
edit, its refusal, and the dynamical claim at a tiny grid on the CPU: the
Jablonowski and Williamson steady state, built to balance Earth's
rotation, run dry at twice Earth's rotation, keeps a divergence far below
the same state run with Earth's winds.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.constants import EARTH_RADIUS_M, EARTH_ROTATION_RATE_S  # noqa: E402
from arwen_global.world.rotation_start import (  # noqa: E402
    ROTATION_START_SCHEMA,
    SLOWEST_REBALANCED_ROTATION,
    rebalanced_state,
    rebalanced_winds,
    rebalanced_zonal_mean,
    rotation_start_replacements,
)

OMEGA = EARTH_ROTATION_RATE_S
A = EARTH_RADIUS_M


def _gradient_wind_force(u, lat_deg, rotation):
    phi = np.deg2rad(lat_deg)
    return (2.0 * rotation * np.sin(phi) + u * np.tan(phi) / A) * u


@pytest.mark.parametrize("factor", [2.0, 1.5, 0.5, 0.8])
def test_the_zonal_mean_holds_earths_force_at_the_worlds_rotation(factor):
    lat = np.array([-80.0, -45.0, -30.0, -10.0, -2.0, 2.0, 10.0, 30.0, 45.0, 80.0])[:, None]
    u = np.array([-10.0, 0.0, 10.0, 35.0, 60.0])[None, :]
    new, irregular = rebalanced_zonal_mean(u, lat, factor)
    assert not irregular.any()
    earth = _gradient_wind_force(u, lat, OMEGA)
    world = _gradient_wind_force(new, lat, factor * OMEGA)
    np.testing.assert_allclose(world, earth, rtol=1e-11, atol=1e-16)


def test_the_zonal_mean_is_continuous_through_the_equator():
    """The Coriolis and curvature terms both vanish at the equator, and
    the reduced form's root there is the limit of its neighbours' roots,
    within the curvature share ``u / (2 Omega a)`` of ``u / r``."""
    u = np.array([[-12.0, 0.0, 8.0, 25.0]])
    at_equator, irregular = rebalanced_zonal_mean(u, np.array([[0.0]]), 2.0)
    assert not irregular.any() and np.all(np.isfinite(at_equator))
    near, _ = rebalanced_zonal_mean(u, np.array([[1.0e-3]]), 2.0)
    np.testing.assert_allclose(near, at_equator, rtol=1e-9)
    share = np.abs(u) / (2.0 * OMEGA * A)
    assert np.all(np.abs(at_equator - u / 2.0) <= share * np.abs(u) + 1e-15)


def test_geostrophic_limit_is_u_over_r():
    """A slow wind far from the pole: the curvature term is negligible and
    the balanced wind is Earth's over the factor (thermal-wind balance at
    the new rate with the same temperature)."""
    new, _ = rebalanced_zonal_mean(np.array([[1.0e-3]]), np.array([[45.0]]), 2.0)
    np.testing.assert_allclose(new, 0.5e-3, rtol=1e-5)


def test_eddies_keep_earths_coriolis_force_exactly():
    rng = np.random.default_rng(32)
    lat = np.linspace(-87.5, 87.5, 36)
    u = 20.0 * np.cos(np.deg2rad(lat))[None, :, None] + rng.normal(0, 8, (3, 36, 72))
    v = rng.normal(0, 6, (3, 36, 72))
    new_u, new_v, record = rebalanced_winds(u, v, lat, 2.0)
    np.testing.assert_array_equal(new_v, v / 2.0)
    eddy = u - u.mean(axis=-1, keepdims=True)
    new_eddy = new_u - new_u.mean(axis=-1, keepdims=True)
    f = 2.0 * OMEGA * np.sin(np.deg2rad(lat))[None, :, None]
    np.testing.assert_allclose(2.0 * f * new_eddy, f * eddy, rtol=1e-12, atol=1e-12)
    new_mean, _ = rebalanced_zonal_mean(u.mean(axis=-1), lat[None, :], 2.0)
    np.testing.assert_allclose(new_u.mean(axis=-1), new_mean, rtol=1e-12, atol=1e-12)
    assert record["schema"] == ROTATION_START_SCHEMA
    assert record["rotation_factor"] == 2.0
    assert record["rows_off_the_regular_branch"] == 0


def test_earths_rotation_returns_the_winds_unchanged():
    rng = np.random.default_rng(7)
    lat = np.linspace(-90.0, 90.0, 19)
    u = rng.normal(0, 15, (4, 19, 36))
    v = rng.normal(0, 5, (4, 19, 36))
    new_u, new_v, _ = rebalanced_winds(u, v, lat, 1.0)
    np.testing.assert_array_equal(new_u, u)
    np.testing.assert_array_equal(new_v, v)


def test_polar_rows_and_the_anomalous_branch_fall_back_to_u_over_r():
    lat = np.array([-90.0, 0.0, 89.99, 90.0])
    u = np.zeros((1, 4, 8))
    u[0, 2, :] = -400.0  # easterly beyond Omega a cos(lat): anomalous branch
    u[0, 0, :] = 5.0
    u[0, 3, :] = -5.0
    new_u, _v, record = rebalanced_winds(u, np.zeros_like(u), lat, 2.0)
    np.testing.assert_allclose(new_u[0, [0, 2, 3]], u[0, [0, 2, 3]] / 2.0)
    assert record["rows_off_the_regular_branch"] == 3


@pytest.mark.parametrize("factor", [0.49, 0.1, 0.0, -1.0, float("nan")])
def test_a_slow_or_invalid_rotation_is_refused_by_name(factor):
    with pytest.raises(ValueError, match="rotation"):
        rebalanced_winds(np.zeros((2, 3, 4)), np.zeros((2, 3, 4)), np.zeros(3), factor)


def test_the_slowest_rotation_is_a_48_hour_day():
    assert SLOWEST_REBALANCED_ROTATION == 0.5
    rebalanced_winds(np.ones((1, 3, 4)), np.zeros((1, 3, 4)), np.array([-30.0, 0.0, 30.0]), 0.5)


def _frame(nlat=19, nlon=36, nlev=3, seed=3):
    rng = np.random.default_rng(seed)
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)

    def field(values):
        return SimpleNamespace(values=values, source="primary")

    fields = {
        "eastward_wind": field(rng.normal(10, 10, (nlev, nlat, nlon))),
        "northward_wind": field(rng.normal(0, 4, (nlev, nlat, nlon))),
        "air_temperature": field(rng.normal(250, 10, (nlev, nlat, nlon))),
        "surface_pressure": field(rng.normal(1.0e5, 500, (nlat, nlon))),
    }
    return SimpleNamespace(latitude=lat, longitude=lon, fields=fields)


def test_the_frame_edit_replaces_the_two_winds_and_nothing_else():
    frame = _frame()
    before = {name: f.values.copy() for name, f in frame.fields.items()}
    replacements, record = rotation_start_replacements(frame, 2.0)
    assert set(replacements) == {"eastward_wind", "northward_wind"}
    for name, values in before.items():
        np.testing.assert_array_equal(frame.fields[name].values, values)
    np.testing.assert_array_equal(replacements["northward_wind"],
                                  before["northward_wind"] / 2.0)
    expected_u, _v, _r = rebalanced_winds(
        before["eastward_wind"], before["northward_wind"], frame.latitude, 2.0)
    np.testing.assert_array_equal(replacements["eastward_wind"], expected_u)
    assert record["largest_speed_m_s"]["start"] < record["largest_speed_m_s"]["analysis"]


def test_the_frame_edit_refuses_a_partial_longitude_ring():
    frame = _frame()
    frame.longitude = frame.longitude[:-1]
    for name in ("eastward_wind", "northward_wind"):
        frame.fields[name].values = frame.fields[name].values[..., :-1]
    with pytest.raises(ValueError, match="whole circle"):
        rotation_start_replacements(frame, 2.0)


# ------------------------------------------------------------- the dynamics

def _jw_model(transform, vertical, phis, rotation):
    from arwen_global.dynamics import MoistHybridModel
    from arwen_global.semi_implicit import VerticalModeSemiImplicit

    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=phis, physics=None, rotation_rate_s=rotation,
        diffusion=None, semi_implicit=VerticalModeSemiImplicit(),
        mass_fixer=False, water_fixer=False, positivity_repair=False,
        maximum_cfl=1.0e9, sponge_base_pa=0.0,
    )


def _largest_divergence(model, cold, hours, dt):
    transform = model.transform
    state = cold
    worst = 0.0
    for _ in range(int(round(hours * 3600.0 / dt))):
        state, _m = model.step(state, dt)
        d = np.asarray(transform.inverse(state.atmosphere.divergence))
        assert np.all(np.isfinite(d))
        worst = max(worst, float(np.max(np.abs(d))))
    return worst


def test_jw_state_at_twice_earths_rotation_stays_balanced_once_rebalanced():
    """The analytic steady state balances Earth's rotation.  Run dry at
    twice that rotation, the state with Earth's winds sheds a divergence
    of the size of its whole Coriolis imbalance in hours; the same mass
    field with the re-balanced winds keeps the divergence the state has
    at Earth's own rotation, to a small factor."""
    from arwen_global.spectral.transform import SphericalHarmonicTransform
    from arwen_global.testcases import baroclinic_wave_initial_state
    from arwen_global.vertical import HybridCoordinate

    transform = SphericalHarmonicTransform.create(15, backend="numpy", precision="float64")
    vertical = HybridCoordinate.surface_stretched(20, 100.0)
    cold, phis, _prov = baroclinic_wave_initial_state(
        SimpleNamespace(vertical=vertical, perturbation_amplitude=0.0), transform)
    dt, hours = 1200.0, 12.0
    earth_model = _jw_model(transform, vertical, phis, OMEGA)
    world_model = _jw_model(transform, vertical, phis, 2.0 * OMEGA)
    rebalanced, record = rebalanced_state(earth_model, cold, 2.0)
    assert record["rotation_factor"] == 2.0
    # The mass field is the input's own object.
    assert rebalanced.atmosphere.theta is cold.atmosphere.theta
    assert rebalanced.atmosphere.log_surface_pressure is cold.atmosphere.log_surface_pressure

    reference = _largest_divergence(earth_model, cold, hours, dt)
    with_earths_winds = _largest_divergence(world_model, cold, hours, dt)
    with_rebalanced = _largest_divergence(world_model, rebalanced, hours, dt)
    assert with_rebalanced < 0.1 * with_earths_winds, (with_rebalanced, with_earths_winds)
    assert with_rebalanced < 3.0 * reference + 1.0e-9, (with_rebalanced, reference)
    assert math.isfinite(with_earths_winds)


# ------------------------------------------------- the hook (package 2.2's)

def test_the_start_hook_rebalances_a_day_length_world_and_leaves_earth_alone():
    """Through ``world/start.py``'s ``world_start_frame`` (package 2.2's
    file, which calls this rule once its patch from package 3.2 is in): a
    12-hour day edits the two winds and records the rule; Earth, and a
    world whose day is Earth's, get the frame back as the same object."""
    start = pytest.importorskip("arwen_global.world.start")
    from arwen_global.world import EARTH, parse_world

    frame = _frame()
    same, record = start.world_start_frame(EARTH, frame)
    assert same is frame and record is None
    tilted = parse_world({"label": "tilt", "sun": {"obliquity_deg": 90.0}})
    same, _record = start.world_start_frame(tilted, frame)
    assert same.fields["eastward_wind"] is frame.fields["eastward_wind"]
    fast = parse_world({"label": "twelve-hour-day", "sun": {"day_length_h": 12.0}})
    edited, record = start.world_start_frame(fast, frame)
    assert record["rotation_start"]["rotation_factor"] == 2.0
    expected, _r = rotation_start_replacements(frame, 2.0)
    for name in ("eastward_wind", "northward_wind"):
        np.testing.assert_array_equal(edited.fields[name].values, expected[name])
    assert edited.fields["air_temperature"] is frame.fields["air_temperature"]
