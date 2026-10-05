"""GI-6 (audit 2026-10-05): the DA 2 m temperature operator reads the
model's surface layer, as the 10 m wind operator does.

Before this the 2 m temperature was the lowest model level moved down by a
fixed 6.5 K/km lapse rate and the dewpoint was the lowest level's vapour:
neither saw the skin, so under a nocturnal inversion the operator predicted
the lowest level plus a tenth of a kelvin and the filter read the whole
inversion as innovation.  Now the temperature comes from the same
Monin-Obukhov similarity diagnostic that writes U10/V10, and only the
model-terrain to station-elevation difference keeps the lapse rate.  The
dewpoint keeps the lowest level's vapour, by measurement
(assimilate.DEWPOINT_OPERATOR).
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pytest

from arwen_global.assimilate import (
    SCREEN_SENSITIVITY_FLOOR,
    SURFACE_LAPSE_K_M,
    _screen_level,
    _Family,
    _ModelSpace,
    _to_numpy_spectral,
)
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.constants import GRAVITY_M_S2
from arwen_global.da.operators import MemberOperators
from arwen_global.obs_table import ObsRow
from arwen_global.runner import build_model_and_cold_state, build_transform
from arwen_global.spectral.sampling import sample_scalar

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
LATS = np.asarray([-40.0, -10.0, 15.0, 45.0])
LONS = np.asarray([20.0, 110.0, 200.0, 300.0])


@pytest.fixture(scope="module")
def cold():
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    model, state = build_model_and_cold_state(cfg, transform)
    terrain = _to_numpy_spectral(
        transform.backend, transform.forward(model.surface_geopotential))
    z_model = sample_scalar(transform, terrain, LATS, LONS) / GRAVITY_M_S2
    return cfg, transform, model, state, terrain, np.asarray(z_model, dtype=np.float64)


def _shifted_skin(surface, kelvin):
    return dataclasses.replace(surface, temperature_k=surface.temperature_k + kelvin)


def _family(variable, elevation):
    return _Family([
        ObsRow(
            source="probe", station_id=f"P{k}", latitude_deg=float(LATS[k]),
            longitude_deg=float(LONS[k]), elevation_m=float(elevation[k]),
            level_pa=None, valid_time=dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc),
            variable=variable, value=0.0, error=1.0,
        )
        for k in range(LATS.size)
    ])


def _space(cfg, transform, terrain, surface):
    return _ModelSpace(
        transform, cfg.vertical, terrain, surface,
        cfg.reference_physics.soil_wetness_capacity,
    )


def test_the_door_2m_temperature_follows_the_skin_under_an_inversion(cold):
    cfg, transform, _model, state, terrain, z_model = cold
    out = {}
    for name, shift in (("warm", 0.0), ("inversion", -15.0)):
        space = _space(cfg, transform, terrain, _shifted_skin(state.surface, shift))
        values = space.evaluate(state.atmosphere, _family("temperature_k", z_model))
        out[name] = (values["temperature_k"], values["dewpoint_k"])
    # A 15 K colder skin cools the 2 m air the operator predicts: it is
    # the surface layer, not the lowest level lapsed down.
    assert np.all(out["inversion"][0] < out["warm"][0] - 1.0)
    # The dewpoint stays the lowest level's vapour (DEWPOINT_OPERATOR: the
    # bucket surface humidity read the model's own 2 m dewpoint 1.45 K
    # moist), so the skin does not move it.
    np.testing.assert_array_equal(out["inversion"][1], out["warm"][1])
    assert np.all(np.isfinite(out["inversion"][0]))


def test_the_door_2m_temperature_is_the_similarity_t2_moved_to_the_station(cold):
    cfg, transform, _model, state, terrain, z_model = cold
    space = _space(cfg, transform, terrain, state.surface)
    at_terrain = space.evaluate(state.atmosphere, _family("temperature_k", z_model))
    raised = space.evaluate(state.atmosphere, _family("temperature_k", z_model + 300.0))
    screen = space.screen_level(state.atmosphere, LATS, LONS)
    np.testing.assert_allclose(at_terrain["temperature_k"], screen["t2"], rtol=1.0e-12)
    # Only the terrain mismatch keeps the lapse rate.
    np.testing.assert_allclose(
        at_terrain["temperature_k"] - raised["temperature_k"],
        SURFACE_LAPSE_K_M * 300.0, rtol=1.0e-9)


def test_the_ensemble_operator_agrees_with_the_door_and_follows_the_skin(cold):
    cfg, transform, model, state, terrain, z_model = cold
    operators = MemberOperators.for_model(model, transform, cfg)
    door = _space(cfg, transform, terrain, state.surface).evaluate(
        state.atmosphere, _family("temperature_k", z_model))
    values, _ = operators.evaluate(
        [state], LATS, LONS, z_model, np.full(LATS.size, np.nan),
        variables=("temperature_k", "dewpoint_k"))
    np.testing.assert_allclose(values["temperature_k"][0], door["temperature_k"], rtol=1.0e-9)
    np.testing.assert_allclose(values["dewpoint_k"][0], door["dewpoint_k"], rtol=1.0e-9)
    chilled = dataclasses.replace(state, surface=_shifted_skin(state.surface, -15.0))
    cold_values, _ = operators.evaluate(
        [chilled], LATS, LONS, z_model, np.full(LATS.size, np.nan),
        variables=("temperature_k",))
    assert np.all(cold_values["temperature_k"][0] < values["temperature_k"][0] - 1.0)


def test_a_station_at_the_pole_reads_a_finite_2m_temperature(cold):
    cfg, transform, model, state, terrain, _z = cold
    operators = MemberOperators.for_model(model, transform, cfg)
    lat = np.asarray([-90.0, 90.0])
    lon = np.asarray([0.0, 0.0])
    values, _ = operators.evaluate(
        [state], lat, lon, np.zeros(2), np.full(2, np.nan),
        variables=("temperature_k", "dewpoint_k"))
    assert np.all(np.isfinite(values["temperature_k"]))
    assert np.all(np.isfinite(values["dewpoint_k"]))
    space = _space(cfg, transform, terrain, state.surface)
    family = _Family([
        ObsRow(source="probe", station_id=f"Q{k}", latitude_deg=float(lat[k]),
               longitude_deg=0.0, elevation_m=0.0, level_pa=None,
               valid_time=dt.datetime(2026, 8, 31, 12, tzinfo=dt.timezone.utc),
               variable="temperature_k", value=0.0, error=1.0)
        for k in range(2)
    ])
    door = space.evaluate(state.atmosphere, family)
    assert np.all(np.isfinite(door["temperature_k"]))


def test_the_2m_gain_carries_the_operators_sensitivity_to_the_lowest_level(cold):
    """The door spreads a 2 m innovation into the lowest level, which the
    screen-level operator follows by only h kelvin per kelvin; the gain is
    the OI gain of that sensitivity (assimilate.SCREEN_TEMPERATURE_GAIN).
    The measured h is the operator's own centred difference, and it is
    what a lowest-level change actually does to the operator's answer."""
    cfg, transform, _model, state, terrain, z_model = cold
    space = _space(cfg, transform, terrain, state.surface)
    family = _family("temperature_k", z_model)
    h = space.screen_temperature_sensitivity(state.atmosphere, family)
    assert h.shape == (LATS.size,)
    assert np.all(h >= SCREEN_SENSITIVITY_FLOOR) and np.all(h <= 1.0)
    ctx = space.surface_context(state.atmosphere, LATS, LONS)
    screen = space.screen_level(state.atmosphere, LATS, LONS)
    from arwen_global.assimilate import _sample_grid, screen_wind_latitude
    u_low, v_low = space.lowest_wind(state.atmosphere, screen_wind_latitude(LATS), LONS)
    grid = transform.grid
    nudged = _screen_level(
        u_low, v_low, ctx["t_low"] + 0.1, ctx["qv_low"], ctx["p_full_low"],
        ctx["ps"], _sample_grid(space.skin_k, grid, LATS, LONS),
        np.clip(_sample_grid(space.land_fraction, grid, LATS, LONS), 0.0, 1.0),
        _sample_grid(space.soil_wetness, grid, LATS, LONS),
        _sample_grid(space.roughness_m, grid, LATS, LONS))
    measured = (np.asarray(nudged["t2"]) - np.asarray(screen["t2"])) / 0.1
    unfloored = measured > SCREEN_SENSITIVITY_FLOOR
    np.testing.assert_allclose(h[unfloored], measured[unfloored], atol=0.02)
    # The screen level is below the lowest level: it follows it by less
    # than one kelvin per kelvin wherever the skin has a say.
    assert np.all(measured < 1.0)
    # An aloft row reads the profile itself: sensitivity one.
    aloft = _Family([dataclasses.replace(row, level_pa=50_000.0) for row in family.rows])
    np.testing.assert_array_equal(
        space.screen_temperature_sensitivity(state.atmosphere, aloft), 1.0)
