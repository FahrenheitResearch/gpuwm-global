"""The sea of a world and the zonal-mean start (worlds package 2.2).

Every test here runs on the CPU in well under a second and needs no door:
the cold start's open-water skin is formed by its own function through an
identity regrid (target grid = analysis grid), and the surface seeding is
run on the edited frame the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from arwen_global.analysis_initial import open_water_skin_temperature
from arwen_global.constants import DRY_AIR_GAS_CONSTANT, EARTH_RADIUS_M, EARTH_ROTATION_RATE_S
from arwen_global.statics import water_columns
from arwen_global.surface_seeding import seed_surface_from_analysis
from arwen_global.world.sea import (
    SEA_WATER_FREEZING_K,
    ZERO_C_K,
    analysis_open_water_columns,
    aquaplanet_profile_c,
    aquaplanet_profile_k,
    edit_sea_ice_frame,
    world_open_water_skin,
    zonal_mean_of_sea,
)
from arwen_global.world.spec import EARTH, SEA_TEMPERATURE_PROFILES, parse_world
from arwen_global.world.start import (
    needs_zonal_mean_start,
    thermal_wind_imbalance,
    world_start_frame,
    zonal_mean_atmosphere,
)

PROFILES = SEA_TEMPERATURE_PROFILES


@dataclass
class _Field:
    values: np.ndarray
    source: str = "primary"


@dataclass
class _Frame:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_values: np.ndarray
    fields: dict


LEVELS_PA = np.asarray([10000.0, 20000.0, 30000.0, 50000.0, 70000.0, 85000.0, 100000.0])


def _frame(nlat=73, nlon=144, *, flat=False, land_box=True, balanced=True, eddy_v=0.0):
    """A small analysis frame on a regular 2.5-degree grid, north to south.

    The temperature is 280 - 40 sin^2(lat) on every level and the zonal
    wind is its exact thermal wind at Earth's rotation,
    u = (40 R / (a Omega)) cos(lat) ln(p0 / p); ``eddy_v`` adds an
    unbalanced meridional wind of wavenumber 2.  One land box (unless
    ``land_box`` is False) sits at 10-40 N, 0-60 E with a 2,000 m plateau
    unless ``flat``.  Sea ice covers the water poleward of 70 degrees,
    half of it at 0.6 and half at 1.0.
    """
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    lat2 = np.broadcast_to(lat[:, None], (nlat, nlon))
    lon2 = np.broadcast_to(lon[None, :], (nlat, nlon))
    phi = np.deg2rad(lat2)
    nlev = LEVELS_PA.size
    temperature = np.broadcast_to(280.0 - 40.0 * np.sin(phi) ** 2, (nlev, nlat, nlon)).copy()
    coefficient = 40.0 * DRY_AIR_GAS_CONSTANT / (EARTH_RADIUS_M * EARTH_ROTATION_RATE_S)
    ln_ratio = np.log(100000.0 / LEVELS_PA)[:, None, None]
    u = (coefficient * np.cos(phi))[None] * ln_ratio if balanced else np.zeros((nlev, nlat, nlon))
    u = np.broadcast_to(u, (nlev, nlat, nlon)).copy()
    # Sheared with height, so it is out of thermal-wind balance.
    v = eddy_v * (np.sin(2.0 * np.deg2rad(lon2)) * np.cos(phi))[None] * ln_ratio
    humidity = 0.012 * (LEVELS_PA / 100000.0)[:, None, None] ** 3 * np.cos(phi)[None] ** 2
    humidity = humidity + 1.0e-4 * (1.0 + np.cos(np.deg2rad(lon2)))[None]
    land = np.zeros((nlat, nlon))
    if land_box:
        land[(lat2 >= 10.0) & (lat2 <= 40.0) & (lon2 >= 0.0) & (lon2 <= 60.0)] = 1.0
    terrain = np.where(land > 0.5, 0.0 if flat else 2000.0, 0.0)
    ps = 101325.0 * np.exp(-terrain / 8000.0) + 300.0 * np.cos(np.deg2rad(lon2))
    # A sea whose temperature varies along each row, so a zonal mean is a change.
    skin = 300.0 - 35.0 * np.sin(phi) ** 2 + 2.0 * np.cos(np.deg2rad(lon2))
    skin = np.where(land > 0.5, skin + 8.0, skin)
    polar = (land < 0.5) & (np.abs(lat2) >= 70.0)
    ice = np.where(polar, np.where(lon2 < 180.0, 0.6, 1.0), 0.0)
    thickness = np.where(ice > 0.0, 1.5, 0.0)
    skin = np.where(ice > 0.0, 250.0, skin)
    open_water = (land < 0.5) & (ice < 0.5)
    swe = np.where(open_water, np.nan, np.where(ice > 0.5, 20.0, 0.0))
    depth = np.where(open_water, np.nan, np.where(ice > 0.5, 0.1, 0.0))
    fields = {
        "air_temperature": _Field(temperature),
        "specific_humidity": _Field(humidity),
        "eastward_wind": _Field(u),
        "northward_wind": _Field(v),
        "surface_pressure": _Field(ps),
        "terrain_height": _Field(terrain),
        "skin_temperature": _Field(skin),
        "land_fraction": _Field(land),
        "sea_ice_fraction": _Field(ice),
        "sea_ice_thickness": _Field(thickness),
        "snow_water_equivalent": _Field(swe),
        "snow_depth": _Field(depth),
    }
    return _Frame(lat, lon, "pressure", LEVELS_PA.copy(), fields)


def _identity(values):
    return np.asarray(values, dtype=np.float64)


def _cold_start_sea(frame):
    """The cold start's own open-water skin on the analysis grid, with the
    run's open water and the analysis's own open water (the hook's three
    inputs), through an identity regrid."""
    land = frame.fields["land_fraction"].values
    seeded = seed_surface_from_analysis(frame, _identity, target_land_fraction=land)
    open_water = water_columns(land, sea_ice_fraction=seeded.sea_ice_fraction)
    skin, _record = open_water_skin_temperature(
        frame.fields["skin_temperature"].values, land, _identity, open_water)
    analysis_water = analysis_open_water_columns(land, _identity, open_water)
    return skin, open_water, analysis_water, seeded


def _world(**sea):
    return parse_world({"label": "sea-test", "sea": sea})


# -- the four published profiles ------------------------------------------


@pytest.mark.parametrize("name", PROFILES)
def test_each_published_profile_is_27_c_at_the_equator_and_0_c_at_and_beyond_60(name):
    assert aquaplanet_profile_c(name, 0.0) == pytest.approx(27.0, abs=1e-12)
    beyond = np.asarray([60.0, 60.5, 70.0, 89.0, 90.0])
    assert np.all(aquaplanet_profile_c(name, beyond) == 0.0)
    assert np.all(aquaplanet_profile_c(name, -beyond) == 0.0)
    # Continuous at the edge, symmetric, and falling toward each pole.
    assert aquaplanet_profile_c(name, 59.999) == pytest.approx(0.0, abs=1e-3)
    lats = np.linspace(0.0, 60.0, 241)
    values = aquaplanet_profile_c(name, lats)
    assert np.all(np.diff(values) <= 1e-12)
    assert np.array_equal(values, aquaplanet_profile_c(name, -lats))
    assert np.allclose(aquaplanet_profile_k(name, lats), values + ZERO_C_K)


def test_the_profiles_have_their_published_shapes():
    lat = np.asarray([15.0, 30.0, 45.0])
    s2 = np.sin(np.deg2rad(1.5 * lat)) ** 2
    control = aquaplanet_profile_c("aqua-control", lat)
    flat = aquaplanet_profile_c("aqua-flat", lat)
    assert np.allclose(control, 27.0 * (1.0 - s2))
    assert np.allclose(flat, 27.0 * (1.0 - s2 ** 2))
    assert np.allclose(aquaplanet_profile_c("aqua-peaked", lat), 27.0 * (1.0 - lat / 60.0))
    assert np.allclose(aquaplanet_profile_c("aqua-qobs", lat), 0.5 * (control + flat))
    # At 30 degrees: control 13.5 C, flat 20.25 C, peaked 13.5 C, qobs 16.875 C.
    assert control[1] == pytest.approx(13.5)
    assert flat[1] == pytest.approx(20.25)
    with pytest.raises(ValueError, match="aqua-control"):
        aquaplanet_profile_c("aqua-warm", lat)


# -- the temperature on the cold start's open-water skin -------------------


def test_the_analysis_sea_is_the_cold_starts_own_skin_object():
    frame = _frame()
    skin, open_water, analysis_water, _ = _cold_start_sea(frame)
    out, record = world_open_water_skin(
        _world(temperature="analysis", ice="none").sea, skin,
        open_water=open_water, analysis_open_water=analysis_water,
        latitude_deg=frame.latitude)
    assert out is skin and record["temperature"] == "analysis"


@pytest.mark.parametrize("name", PROFILES)
def test_a_profile_sea_has_the_profile_as_its_zonal_mean_and_leaves_land_and_ice(name):
    frame = _frame()
    skin, open_water, analysis_water, _ = _cold_start_sea(frame)
    out, record = world_open_water_skin(
        _world(temperature=name).sea, skin, open_water=open_water,
        analysis_open_water=analysis_water, latitude_deg=frame.latitude)
    zonal = zonal_mean_of_sea(out, open_water)
    rows = np.isfinite(zonal)
    expected = aquaplanet_profile_k(name, frame.latitude)
    assert np.allclose(zonal[rows], expected[rows], rtol=0.0, atol=1e-12)
    assert np.array_equal(out[~open_water], skin[~open_water])
    assert record["sea_columns"] == int(open_water.sum())
    assert record["reference"] == "Neale and Hoskins (2000)"


def test_the_zonal_mean_sea_is_the_mean_of_the_analysis_own_open_water_by_row():
    frame = _frame()
    skin, open_water, analysis_water, _ = _cold_start_sea(frame)
    out, record = world_open_water_skin(
        _world(temperature="zonal-mean").sea, skin, open_water=open_water,
        analysis_open_water=analysis_water, latitude_deg=frame.latitude)
    # The rule's input: the row mean of the analysis's own open water.
    for j in range(frame.latitude.size):
        here = analysis_water[j]
        if here.any():
            expected = max(float(skin[j][here].mean()), SEA_WATER_FREEZING_K)
            assert np.allclose(out[j][open_water[j]], expected, rtol=0.0, atol=1e-9)
    # The zonal mean of the result over the sea is that input, and the
    # sea no longer varies along a row.
    zonal = zonal_mean_of_sea(out, open_water)
    reference = zonal_mean_of_sea(skin, analysis_water)
    rows = np.isfinite(reference)
    assert np.allclose(zonal[rows], np.maximum(reference[rows], SEA_WATER_FREEZING_K),
                       rtol=0.0, atol=1e-9)
    sea_rows = open_water.any(axis=1)
    sea = np.where(open_water, out, np.nan)[sea_rows]
    assert np.max(np.nanmax(sea, axis=1) - np.nanmin(sea, axis=1)) < 1e-9
    assert record["rows_with_analysis_open_water"] == int(rows.sum())
    assert np.array_equal(out[~open_water], skin[~open_water])


def test_a_flooded_column_takes_its_row_and_is_not_averaged_into_it():
    # The world's open water reaches columns the analysis calls land (a
    # flood): they receive the row's value, which is the mean over the
    # analysis's OWN open water only.
    frame = _frame()
    skin, open_water, analysis_water, _ = _cold_start_sea(frame)
    flooded = frame.fields["land_fraction"].values > 0.5
    world_water = open_water | flooded
    hot = np.where(flooded, 330.0, skin)  # a land skin the search might carry
    out, _record = world_open_water_skin(
        _world(temperature="zonal-mean").sea, hot, open_water=world_water,
        analysis_open_water=analysis_open_water_columns(
            frame.fields["land_fraction"].values, _identity, world_water),
        latitude_deg=frame.latitude)
    j = int(np.argmin(np.abs(frame.latitude - 25.0)))
    assert flooded[j].any()
    expected = skin[j][analysis_water[j]].mean()
    assert np.allclose(out[j][world_water[j]], expected, atol=1e-9)
    assert float(out[flooded].max()) < 330.0


def test_rows_with_no_analysis_open_water_read_their_neighbours_and_never_freeze():
    frame = _frame()
    skin, open_water, analysis_water, _ = _cold_start_sea(frame)
    out, record = world_open_water_skin(
        _world(temperature="zonal-mean").sea, skin, open_water=open_water,
        analysis_open_water=analysis_water & (np.abs(frame.latitude) < 50.0)[:, None],
        latitude_deg=frame.latitude)
    assert record["rows_filled_by_latitude"] > 0
    north = frame.latitude >= 50.0
    # Held beyond the last row that has open water: one value per row,
    # the same for every row poleward of 50 N, never below freezing.
    values = np.where(open_water, out, np.nan)[north]
    finite = values[np.isfinite(values)]
    assert finite.size and np.ptp(finite) < 1e-9
    assert float(np.nanmin(np.where(open_water, out, np.nan))) >= SEA_WATER_FREEZING_K


# -- the ice, on the analysis frame ----------------------------------------


def test_analysis_ice_returns_the_frame_object():
    frame = _frame()
    out, record = edit_sea_ice_frame(frame, _world(temperature="zonal-mean").sea)
    assert out is frame and record["ice"] == "analysis"


def test_no_ice_opens_every_pack_at_no_less_than_freezing_and_leaves_the_input():
    frame = _frame()
    before = {name: np.array(f.values, copy=True) for name, f in frame.fields.items()}
    out, record = edit_sea_ice_frame(frame, _world(ice="none").sea)
    for name, values in before.items():
        assert np.array_equal(frame.fields[name].values, values, equal_nan=True)
    ice = out.fields["sea_ice_fraction"].values
    land = frame.fields["land_fraction"].values > 0.5
    assert np.all(ice == 0.0) and np.all(out.fields["sea_ice_thickness"].values == 0.0)
    was_ice = before["sea_ice_fraction"] > 0.0
    skin = out.fields["skin_temperature"].values
    assert np.all(skin[was_ice] >= SEA_WATER_FREEZING_K)
    assert np.array_equal(skin[~was_ice], before["skin_temperature"][~was_ice])
    assert np.array_equal(skin[land], before["skin_temperature"][land])
    assert record["analysis_points_frozen_over"] > 0
    # The grid and everything else are the frame's own; the edit is recorded.
    assert out.latitude is frame.latitude and out.vertical_values is frame.vertical_values
    assert out.fields["air_temperature"] is frame.fields["air_temperature"]
    assert out.world_edits["sea_ice"] is record
    # The cold start's own seeding takes it: no frozen column anywhere.
    _skin, open_water, _aw, seeded = _cold_start_sea(out)
    assert not np.any(seeded.sea_ice_fraction >= 0.5)
    assert np.all(open_water == ~land)


def test_zonal_ice_carries_each_rows_mean_fraction_and_seeds_without_refusal():
    frame = _frame()
    out, record = edit_sea_ice_frame(frame, _world(temperature="zonal-mean", ice="zonal").sea)
    ice = out.fields["sea_ice_fraction"].values
    water = frame.fields["land_fraction"].values < 0.5
    original = frame.fields["sea_ice_fraction"].values
    for j in range(frame.latitude.size):
        if water[j].any():
            assert np.allclose(ice[j][water[j]], original[j][water[j]].mean())
            skin_row = out.fields["skin_temperature"].values[j][water[j]]
            if original[j][water[j]].any():
                # A row with ice: one composite skin along it.
                assert np.ptp(skin_row) < 1e-9
            else:
                # A row with none keeps the analysis's sea.
                assert np.array_equal(
                    skin_row, frame.fields["skin_temperature"].values[j][water[j]])
    assert record["rows_with_ice"] > 0
    # Half the polar row was 0.6 and half 1.0: the zonal pack is 0.8, frozen.
    assert record["largest_row_fraction"] == pytest.approx(0.8)
    assert np.allclose(out.fields["sea_ice_thickness"].values[ice > 0.0], 1.5)
    _skin, _ow, _aw, seeded = _cold_start_sea(out)
    frozen = seeded.sea_ice_fraction >= 0.5
    assert frozen.any() and np.all(seeded.snow_water_kg_m2[frozen] == 0.0)


# -- the zonal-mean start ---------------------------------------------------


def test_the_zonal_mean_start_refuses_a_frame_with_mountains():
    with pytest.raises(ValueError, match=r"flat frame.*2000 m.*scale = 0"):
        zonal_mean_atmosphere(_frame())


def test_the_zonal_mean_start_is_zonal_keeps_each_rows_air_and_nothing_else():
    frame = _frame(flat=True)
    out, record = zonal_mean_atmosphere(frame)
    for name in ("air_temperature", "specific_humidity", "eastward_wind",
                 "northward_wind", "surface_pressure"):
        values = out.fields[name].values
        original = frame.fields[name].values
        assert np.ptp(values, axis=-1).max() == 0.0
        assert np.allclose(values.mean(axis=-1), original.mean(axis=-1), rtol=0.0, atol=1e-9)
    ps = record["global_mean_surface_pressure_pa"]
    assert ps["start"] == pytest.approx(ps["analysis"], rel=1e-14)
    assert record["largest_departure_removed"]["surface_pressure"] == pytest.approx(300.0)
    for name in ("skin_temperature", "land_fraction", "terrain_height", "sea_ice_fraction"):
        assert out.fields[name] is frame.fields[name]


def test_the_zonal_mean_start_carries_only_the_zonal_part_of_the_imbalance():
    # A balanced jet plus an unbalanced wavenumber-2 meridional wind.
    frame = _frame(flat=True, eddy_v=20.0)
    full = thermal_wind_imbalance(frame, LEVELS_PA)
    start, _record = zonal_mean_atmosphere(frame)
    zonal = thermal_wind_imbalance(start, LEVELS_PA)
    assert full["ratio"] > 0.3
    assert zonal["ratio"] < 0.01
    assert zonal["rms_residual_m_s2"] < full["rms_residual_m_s2"]


def test_the_imbalance_is_zero_to_truncation_for_an_exact_thermal_wind():
    balanced = thermal_wind_imbalance(_frame(flat=True), LEVELS_PA)
    assert balanced["ratio"] < 0.01
    assert balanced["points"] > 0
    calm = thermal_wind_imbalance(_frame(flat=True, balanced=False), LEVELS_PA)
    assert calm["ratio"] == pytest.approx(1.0, rel=1e-12)
    # At twice the rotation the same jet is half the balanced one.
    doubled = thermal_wind_imbalance(_frame(flat=True), LEVELS_PA,
                                     rotation_rate_s=2.0 * EARTH_ROTATION_RATE_S)
    assert doubled["ratio"] == pytest.approx(1.0, rel=0.02)


# -- the frame hook's one entry --------------------------------------------


def test_earth_and_an_earth_sea_leave_the_frame_object():
    frame = _frame()
    out, record = world_start_frame(EARTH, frame)
    assert out is frame and record is None
    sun_only = parse_world({"label": "tilted", "sun": {"obliquity_deg": 90.0}})
    out, record = world_start_frame(sun_only, frame)
    assert out is frame and record is None
    assert not needs_zonal_mean_start(EARTH) and not needs_zonal_mean_start(sun_only)


def test_a_waterworld_frame_is_zonal_and_ice_free():
    world = parse_world({
        "label": "waterworld",
        "terrain": {"scale": 0.0},
        "sea": {"land": "none", "temperature": "zonal-mean", "ice": "none"},
    })
    assert needs_zonal_mean_start(world)
    frame = _frame(flat=True)
    out, record = world_start_frame(world, frame)
    assert set(record) == {"zonal_mean_start", "sea_ice"}
    assert np.ptp(out.fields["air_temperature"].values, axis=-1).max() == 0.0
    assert np.all(out.fields["sea_ice_fraction"].values == 0.0)
    assert set(out.world_edits) == {"zonal_mean_start", "sea_ice"}
    with pytest.raises(ValueError, match="flat frame"):
        world_start_frame(world, _frame())


def test_the_held_constants_are_the_models_own():
    from arwen_global.physics import frozen_surface
    from arwen_global import statics
    from arwen_global.world import sea

    assert sea.SEA_WATER_FREEZING_K == frozen_surface.SEA_ICE_BOTTOM_K
    assert sea.SEA_ICE_THRESHOLD == statics.SEA_ICE_THRESHOLD
