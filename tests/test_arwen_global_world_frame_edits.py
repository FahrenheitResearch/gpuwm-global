"""The pressure mover and the terrain edit of a world (plan package 1.1).

Known answers on small CPU frames: the closed form of a constant-lapse
atmosphere at 500, 2,000 and 4,500 m to 10 Pa, Earth's terrain returned bit
for bit, lowering then raising back to 1 Pa, the below-ground rule, the two
air-mass rules, the land skin and the snow rule, and the box taper.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest

from arwen_global.constants import GRAVITY_M_S2
from arwen_global.world import EARTH, parse_world
from arwen_global.world.frame_edits import (
    DRY_AIR_GAS_CONSTANT, LAPSE_RATE_K_M, WorldFrame, air_mass_factor,
    apply_terrain_edit, below_ground_air, box_weight, cell_area_weights,
    closed_form_constant_lapse, edited_terrain, move_surface_pressure,
)

#: A GDAS-like isobaric ladder (Pa), top first.
LEVELS = np.asarray([
    100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0,
    4000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0,
    35000.0, 40000.0, 45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0,
    75000.0, 80000.0, 85000.0, 90000.0, 92500.0, 95000.0, 97500.0, 100000.0,
])
C65 = DRY_AIR_GAS_CONSTANT * LAPSE_RATE_K_M / GRAVITY_M_S2


@dataclass
class _Field:
    values: np.ndarray


@dataclass
class _Frame:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_values: np.ndarray
    fields: dict
    mapping_sha256: str = "test-mapping"
    input_sha256: str = "test-input"
    source_cycle: datetime = datetime(2026, 11, 10, 18)
    valid_time: datetime = datetime(2026, 11, 10, 18)


def _constant_lapse_column(ps_pa, surface_temperature_k, lapse_k_m):
    """A dry column of constant lapse anchored at the surface, on LEVELS
    (the levels below the surface continue the same lapse)."""
    c = DRY_AIR_GAS_CONSTANT * lapse_k_m / GRAVITY_M_S2
    return surface_temperature_k * (LEVELS / ps_pa) ** c


def _frame(nlat=19, nlon=36, *, seed=0, lon0=0.0):
    """A small GDAS-shaped frame: a mountain block with snow, an ocean with
    zero terrain, temperature with a lapse that varies with height and
    latitude, moist low levels, and the below-ground levels holding a
    centre's own (different) extrapolation."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(90.0, -90.0, nlat)
    lon = lon0 + np.arange(nlon) * (360.0 / nlon)
    lat2 = np.deg2rad(lat)[:, None] * np.ones((1, nlon))
    lon2 = np.deg2rad(lon)[None, :] * np.ones((nlat, 1))
    # Terrain: a 4,000 m block plus a ridge, zero over the ocean half.
    land = (np.cos(lon2) > -0.2) & (np.abs(np.rad2deg(lat2)) < 75.0)
    terrain = np.where(
        land,
        300.0 + 3700.0 * np.exp(-((np.rad2deg(lat2) - 30.0) / 15.0) ** 2)
        * np.cos(lon2).clip(0.0, None) ** 2
        + 50.0 * rng.random((nlat, nlon)),
        0.0)
    t_surface_sea = 300.0 - 45.0 * np.sin(lat2) ** 2
    ps_sea = 101325.0 + 800.0 * np.cos(3.0 * lat2)
    # Surface pressure from the standard lapse; the profile above it uses a
    # height-varying lapse so the column is not a closed form.
    t_ground = t_surface_sea - LAPSE_RATE_K_M * terrain
    ps = ps_sea * (t_ground / t_surface_sea) ** (1.0 / C65)
    lnp = np.log(LEVELS)[:, None, None]
    lnps = np.log(ps)[None]
    temperature = t_ground[None] + 32.0 * (lnp - lnps) + 2.5 * (lnp - lnps) ** 2
    temperature = np.maximum(temperature, 195.0)
    # A centre's below-ground values that differ from the world's rule.
    below = LEVELS[:, None, None] > ps[None]
    temperature = np.where(below, temperature + 3.0, temperature)
    humidity = 0.016 * np.exp(-((LEVELS[:, None, None] - 100000.0) / 25000.0) ** 2) \
        * np.cos(lat2)[None] ** 2 + 2e-6
    humidity = np.where(below, humidity * 1.7, humidity)
    u = 15.0 * np.cos(lat2)[None] * np.ones((LEVELS.size, 1, 1)) + 0.1 * lnp
    v = 2.0 * np.sin(2 * lon2)[None] * np.ones((LEVELS.size, 1, 1))
    skin = np.where(land, t_ground - 1.5, t_surface_sea)
    soil = np.where(land[None], skin[None] + np.asarray([0.5, 1.0, 1.5, 2.0])[:, None, None],
                    np.nan)
    soil_m = np.where(land[None], 0.3, np.nan) * np.ones((4, 1, 1))
    # Snow on the high ground (which thaws when lowered) and on the land
    # poleward of 50 degrees (which stays below freezing when lowered).
    snowy = land & ((terrain > 1000.0) | (np.abs(np.rad2deg(lat2)) >= 50.0))
    swe = np.where(land, np.where(snowy, 60.0, 0.0), np.nan)
    depth = np.where(land, np.where(snowy, 0.3, 0.0), np.nan)
    ice = np.zeros((nlat, nlon))
    fields = {
        "air_temperature": _Field(temperature),
        "specific_humidity": _Field(humidity),
        "eastward_wind": _Field(u),
        "northward_wind": _Field(v),
        "surface_pressure": _Field(ps),
        "terrain_height": _Field(terrain),
        "skin_temperature": _Field(skin),
        "land_fraction": _Field(land.astype(np.float64)),
        "soil_temperature": _Field(soil),
        "volumetric_soil_moisture": _Field(soil_m),
        "sea_ice_fraction": _Field(ice),
        "sea_ice_thickness": _Field(ice.copy()),
        "snow_water_equivalent": _Field(swe),
        "snow_depth": _Field(depth),
    }
    return _Frame(lat, lon, "pressure", LEVELS.copy(), fields)


def _world(text):
    import tomllib

    return parse_world(tomllib.loads(text)["world"])


FLAT_FILL = _world('[world]\nlabel = "flat"\n[world.terrain]\nscale = 0.0\nair_mass = "fill"\n')
FLAT_CONSERVE = _world('[world]\nlabel = "flat"\n[world.terrain]\nscale = 0.0\n')


# -- the closed form --------------------------------------------------------

@pytest.mark.parametrize("drop_m", [500.0, 2000.0, 4500.0])
def test_lowering_matches_the_constant_lapse_closed_form_to_10_pa(drop_m):
    """A dry atmosphere of 6.5 K/km whose ground at 4,500 m is lowered by
    500, 2,000 and 4,500 m: the moved pressure is the closed form's."""
    z_old = 4500.0
    t_sea, p_sea = 288.15, 101325.0
    t_ground = t_sea - LAPSE_RATE_K_M * z_old
    ps = p_sea * (t_ground / t_sea) ** (1.0 / C65)
    column = _constant_lapse_column(ps, t_ground, LAPSE_RATE_K_M)
    t, q, _, _, _ = below_ground_air(
        LEVELS, column[:, None], np.zeros((LEVELS.size, 1)), None, None,
        np.asarray([ps]))
    moved = move_surface_pressure(LEVELS, t, q, np.asarray([ps]),
                                  np.asarray([z_old]), np.asarray([z_old - drop_m]))
    expected = closed_form_constant_lapse(ps, t_ground, -drop_m)
    assert abs(float(moved[0]) - expected) <= 10.0
    # The layer integral is exact for a constant lapse, so the agreement is
    # far inside the gate; this pins the arithmetic, not only the gate.
    assert abs(float(moved[0]) - expected) <= 1e-3


@pytest.mark.parametrize("drop_m", [500.0, 2000.0, 4500.0])
def test_lowering_through_the_frame_edit_matches_the_closed_form(drop_m):
    """The same column through apply_terrain_edit: a cap of 4,500 - drop
    lowers the 4,500 m ground by ``drop_m``; air_mass "fill" keeps it."""
    frame = _frame(nlat=3, nlon=4)
    z_old = 4500.0
    t_sea, p_sea = 288.15, 101325.0
    t_ground = t_sea - LAPSE_RATE_K_M * z_old
    ps = p_sea * (t_ground / t_sea) ** (1.0 / C65)
    shape = (3, 4)
    column = _constant_lapse_column(ps, t_ground, LAPSE_RATE_K_M)
    frame.fields["terrain_height"] = _Field(np.full(shape, z_old))
    frame.fields["surface_pressure"] = _Field(np.full(shape, ps))
    frame.fields["air_temperature"] = _Field(column[:, None, None] * np.ones((1,) + shape))
    frame.fields["specific_humidity"] = _Field(np.zeros((LEVELS.size,) + shape))
    world = _world('[world]\nlabel = "capped"\n[world.terrain]\n'
                   f'cap_m = {z_old - drop_m}\nair_mass = "fill"\n')
    edited, record = apply_terrain_edit(frame, world, LEVELS)
    expected = closed_form_constant_lapse(ps, t_ground, -drop_m)
    got = edited.fields["surface_pressure"].values
    assert np.max(np.abs(got - expected)) <= 10.0
    assert record["columns_lowered"] == 12


@pytest.mark.parametrize("lapse", [0.004, 0.0065, 0.0085])
@pytest.mark.parametrize("rise_m", [500.0, 2000.0, 4500.0])
def test_raising_matches_the_closed_form_through_the_column_own_lapse(lapse, rise_m):
    """Raising ground integrates through the air that is there, whatever its
    lapse, and matches that lapse's closed form.  The ground sits between
    two levels (below the deepest level the column would continue at the
    world's 6.5 K/km, not at this column's lapse)."""
    ps, t_ground = 98000.0, 300.0
    column = _constant_lapse_column(ps, t_ground, lapse)
    moved = move_surface_pressure(
        LEVELS, column[:, None], np.zeros((LEVELS.size, 1)), np.asarray([ps]),
        np.asarray([0.0]), np.asarray([rise_m]))
    expected = closed_form_constant_lapse(ps, t_ground, rise_m, lapse_k_m=lapse)
    assert abs(float(moved[0]) - expected) <= 10.0
    assert abs(float(moved[0]) - expected) <= 1e-3


def test_moist_air_moves_by_its_virtual_temperature():
    """Humidity enters as Tv = T (1 + 0.608 q): a moist isothermal column
    moves as a dry one at the virtual temperature."""
    ps, t = 100000.0, 290.0
    q = 0.015
    tv = t * (1.0 + 0.608 * q)
    moved = move_surface_pressure(
        LEVELS, np.full((LEVELS.size, 1), t), np.full((LEVELS.size, 1), q),
        np.asarray([ps]), np.asarray([0.0]), np.asarray([1000.0]))
    expected = ps * np.exp(-GRAVITY_M_S2 * 1000.0 / (DRY_AIR_GAS_CONSTANT * tv))
    assert abs(float(moved[0]) - expected) <= 1e-6


# -- Earth returns its input -----------------------------------------------

def test_earth_terrain_returns_the_frame_object_itself():
    frame = _frame()
    out, record = apply_terrain_edit(frame, EARTH, LEVELS)
    assert out is frame and record is None
    scale_one = _world('[world]\nlabel = "sun"\n[world.sun]\nbrightness = 1.1\n')
    out, record = apply_terrain_edit(frame, scale_one, LEVELS)
    assert out is frame and record is None
    z = frame.fields["terrain_height"].values
    assert edited_terrain(z, frame.latitude, frame.longitude, EARTH.terrain) is z


def test_an_unchanged_column_keeps_its_pressure_bit_for_bit():
    frame = _frame()
    ps = frame.fields["surface_pressure"].values
    z = frame.fields["terrain_height"].values
    moved = move_surface_pressure(
        LEVELS, frame.fields["air_temperature"].values,
        frame.fields["specific_humidity"].values, ps, z, z.copy())
    assert moved.tobytes() == ps.tobytes()
    for dtype in (np.float32, np.float64):
        ps_typed = ps.astype(dtype)
        again = move_surface_pressure(
            LEVELS, frame.fields["air_temperature"].values,
            frame.fields["specific_humidity"].values, ps_typed, z, z)
        assert again.dtype == dtype and again.tobytes() == ps_typed.tobytes()


def test_columns_outside_a_box_are_untouched_bit_for_bit():
    frame = _frame()
    world = _world('[world]\nlabel = "box"\n[world.terrain]\nscale = 0.0\n'
                   'air_mass = "fill"\n[world.terrain.box]\nsouth_deg = 10.0\n'
                   'north_deg = 50.0\nwest_deg = -40.0\neast_deg = 40.0\ntaper_deg = 10.0\n')
    edited, record = apply_terrain_edit(frame, world, LEVELS)
    weight = box_weight(frame.latitude, frame.longitude, world.terrain.box)
    outside = weight == 0.0
    assert outside.any() and (~outside).any()
    for name, field in frame.fields.items():
        before = np.asarray(field.values)
        after = np.asarray(edited.fields[name].values)
        if before.ndim == 3:
            before, after = before[:, outside], after[:, outside]
        else:
            before, after = before[outside], after[outside]
        assert before.tobytes() == after.tobytes(), name
    assert record["columns_moved"] == int(np.count_nonzero(
        edited.fields["terrain_height"].values != frame.fields["terrain_height"].values))


# -- there and back -----------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_lowering_then_raising_returns_the_start_to_1_pa(seed):
    """Halve the terrain, then double it back (exact in floating point):
    the surface pressure returns to within 1 Pa on every column."""
    frame = _frame(seed=seed)
    half = _world('[world]\nlabel = "half"\n[world.terrain]\nscale = 0.5\nair_mass = "fill"\n')
    double = _world('[world]\nlabel = "double"\n[world.terrain]\nscale = 2.0\nair_mass = "fill"\n')
    lowered, _ = apply_terrain_edit(frame, half, LEVELS)
    back, record = apply_terrain_edit(lowered, double, LEVELS)
    z0 = frame.fields["terrain_height"].values
    assert np.array_equal(back.fields["terrain_height"].values, z0)
    ps0 = frame.fields["surface_pressure"].values
    ps2 = back.fields["surface_pressure"].values
    assert record["columns_raised"] > 0
    assert np.max(np.abs(ps2 - ps0)) <= 1.0
    # And to the flat world and back through the mover itself.
    flat, _ = apply_terrain_edit(frame, FLAT_FILL, LEVELS)
    ps_back = move_surface_pressure(
        LEVELS, flat.fields["air_temperature"].values,
        flat.fields["specific_humidity"].values,
        flat.fields["surface_pressure"].values,
        flat.fields["terrain_height"].values, z0)
    assert np.max(np.abs(ps_back - ps0)) <= 1.0


# -- the air under removed ground -------------------------------------------

def test_air_under_lowered_ground_follows_the_world_rule():
    frame = _frame()
    edited, record = apply_terrain_edit(frame, FLAT_FILL, LEVELS)
    ps = frame.fields["surface_pressure"].values
    z = frame.fields["terrain_height"].values
    lowered = z > 0.0
    t0 = frame.fields["air_temperature"].values
    t1 = edited.fields["air_temperature"].values
    q0 = frame.fields["specific_humidity"].values
    q1 = edited.fields["specific_humidity"].values
    u1 = edited.fields["eastward_wind"].values
    checked = 0
    for j, i in zip(*np.nonzero(lowered)):
        anchor = int(np.count_nonzero(LEVELS <= ps[j, i])) - 1
        for k in range(LEVELS.size):
            if k <= anchor:
                assert t1[k, j, i] == t0[k, j, i] and q1[k, j, i] == q0[k, j, i]
            else:
                expected = t0[anchor, j, i] * (LEVELS[k] / LEVELS[anchor]) ** C65
                assert t1[k, j, i] == pytest.approx(expected, rel=1e-14)
                assert q1[k, j, i] == q0[anchor, j, i]
                assert u1[k, j, i] == frame.fields["eastward_wind"].values[anchor, j, i]
                checked += 1
    assert checked == record["below_ground_levels_rewritten"] > 0
    # Columns that do not move keep every level, below ground included.
    assert t1[:, ~lowered].tobytes() == t0[:, ~lowered].tobytes()
    assert q1[:, ~lowered].tobytes() == q0[:, ~lowered].tobytes()


def test_the_rule_does_not_depend_on_the_centre_below_ground_values():
    """Two analyses that differ only below ground (one centre holds
    relative humidity, another specific humidity) start the same world."""
    a = _frame()
    b = _frame()
    ps = a.fields["surface_pressure"].values
    below = LEVELS[:, None, None] > ps[None]
    b.fields["air_temperature"] = _Field(
        np.where(below, a.fields["air_temperature"].values + 7.0,
                 a.fields["air_temperature"].values))
    b.fields["specific_humidity"] = _Field(
        np.where(below, a.fields["specific_humidity"].values * 3.0,
                 a.fields["specific_humidity"].values))
    for world in (FLAT_FILL, FLAT_CONSERVE):
        ea, _ = apply_terrain_edit(a, world, LEVELS)
        eb, _ = apply_terrain_edit(b, world, LEVELS)
        for name in ("surface_pressure", "air_temperature", "specific_humidity"):
            lowered = a.fields["terrain_height"].values > 0.0
            va = np.asarray(ea.fields[name].values)
            vb = np.asarray(eb.fields[name].values)
            if va.ndim == 3:
                va, vb = va[:, lowered], vb[:, lowered]
            else:
                va, vb = va[lowered], vb[lowered]
            assert va.tobytes() == vb.tobytes(), name


# -- the amount of air --------------------------------------------------------

def test_conserve_keeps_the_amount_of_air_and_fill_adds_it():
    frame = _frame()
    lat = frame.latitude
    weights = cell_area_weights(lat, frame.longitude.size)
    assert weights.sum() == pytest.approx(2.0 * frame.longitude.size, rel=1e-12)
    ps0 = frame.fields["surface_pressure"].values
    mass0 = np.sum(weights * ps0)
    fill, rec_fill = apply_terrain_edit(frame, FLAT_FILL, LEVELS)
    cons, rec_cons = apply_terrain_edit(frame, FLAT_CONSERVE, LEVELS)
    assert rec_fill["air_mass_factor"] == 1.0
    assert np.sum(weights * fill.fields["surface_pressure"].values) > mass0 * 1.01
    assert np.sum(weights * cons.fields["surface_pressure"].values) == pytest.approx(
        mass0, rel=1e-13)
    factor = rec_cons["air_mass_factor"]
    assert factor < 1.0
    assert factor == air_mass_factor(fill.fields["surface_pressure"].values, ps0, lat)
    # One number for every column, the unmoved ocean included.
    assert np.allclose(cons.fields["surface_pressure"].values,
                       fill.fields["surface_pressure"].values * factor, rtol=0, atol=1e-9)
    # The isobaric air is the same under either rule.
    assert (cons.fields["air_temperature"].values.tobytes()
            == fill.fields["air_temperature"].values.tobytes())


# -- land skin, soil and snow -----------------------------------------------

def test_land_warms_by_the_lapse_and_snow_goes_where_the_skin_thaws():
    frame = _frame()
    edited, record = apply_terrain_edit(frame, FLAT_CONSERVE, LEVELS)
    z = frame.fields["terrain_height"].values
    land = frame.fields["land_fraction"].values >= 0.5
    skin0 = frame.fields["skin_temperature"].values
    skin1 = edited.fields["skin_temperature"].values
    assert np.allclose(skin1[land], skin0[land] + LAPSE_RATE_K_M * z[land], rtol=0, atol=1e-10)
    assert skin1[~land].tobytes() == skin0[~land].tobytes()
    soil0 = frame.fields["soil_temperature"].values
    soil1 = edited.fields["soil_temperature"].values
    assert np.allclose(soil1[:, land], soil0[:, land] + LAPSE_RATE_K_M * z[land],
                       rtol=0, atol=1e-10)
    assert np.isnan(soil1[:, ~land]).all()
    swe0 = frame.fields["snow_water_equivalent"].values
    swe1 = edited.fields["snow_water_equivalent"].values
    snowy = land & (swe0 > 0.0)
    thawed = snowy & (skin1 > 273.15)
    kept = snowy & (skin1 <= 273.15)
    assert thawed.any() and kept.any()
    assert (swe1[thawed] == 0.0).all() and (edited.fields["snow_depth"].values[thawed] == 0.0).all()
    assert (swe1[kept] == swe0[kept]).all()
    assert record["snow_removed_columns"] == int(np.count_nonzero(thawed))
    assert record["snow_removed_global_mean_kg_m2"] > 0.0
    # Soil moisture, ice and the land mask are not the terrain edit's.
    for name in ("volumetric_soil_moisture", "sea_ice_fraction", "land_fraction"):
        assert edited.fields[name] is frame.fields[name]


def test_raised_land_cools_and_keeps_its_snow():
    frame = _frame()
    world = _world('[world]\nlabel = "tall"\n[world.terrain]\nscale = 1.5\n'
                   'air_mass = "fill"\n')
    edited, record = apply_terrain_edit(frame, world, LEVELS)
    z = frame.fields["terrain_height"].values
    land = frame.fields["land_fraction"].values >= 0.5
    skin0 = frame.fields["skin_temperature"].values
    skin1 = edited.fields["skin_temperature"].values
    assert np.allclose(skin1[land], skin0[land] - LAPSE_RATE_K_M * 0.5 * z[land],
                       rtol=0, atol=1e-10)
    assert record["snow_removed_columns"] == 0
    assert record["columns_lowered"] == 0 and record["below_ground_levels_rewritten"] == 0
    assert (edited.fields["surface_pressure"].values[land]
            < frame.fields["surface_pressure"].values[land]).all()


# -- the box --------------------------------------------------------------------

def test_box_weight_is_one_inside_and_tapers_to_zero():
    world = _world('[world]\nlabel = "box"\n[world.terrain]\nscale = 0.0\n'
                   '[world.terrain.box]\nsouth_deg = 20.0\nnorth_deg = 40.0\n'
                   'west_deg = 60.0\neast_deg = 100.0\ntaper_deg = 10.0\n')
    box = world.terrain.box
    lat = np.asarray([45.0, 40.0, 30.0, 15.0, 10.0, 5.0])
    lon = np.asarray([30.0, 50.0, 55.0, 80.0, 105.0, 110.0, 120.0])
    w = box_weight(lat, lon, box)
    assert w[2, 3] == 1.0 and w[1, 3] == 1.0
    assert w[0, 3] == pytest.approx(0.5)        # 5 degrees north of the box
    assert w[3, 3] == pytest.approx(0.5)        # 5 degrees south
    assert w[4, 3] == 0.0 and w[5, 3] == 0.0    # at and beyond the taper
    assert w[2, 2] == pytest.approx(0.5) and w[2, 4] == pytest.approx(0.5)
    assert w[2, 1] == 0.0 and w[2, 5] == 0.0 and w[2, 6] == 0.0 and w[2, 0] == 0.0
    assert w[0, 2] == pytest.approx(0.25)


def test_box_across_the_dateline_and_either_longitude_convention():
    world = _world('[world]\nlabel = "box"\n[world.terrain]\nscale = 0.0\n'
                   '[world.terrain.box]\nsouth_deg = -10.0\nnorth_deg = 10.0\n'
                   'west_deg = 170.0\neast_deg = -170.0\ntaper_deg = 0.0\n')
    box = world.terrain.box
    lat = np.asarray([0.0])
    east_convention = np.arange(0.0, 360.0, 5.0)
    signed_convention = np.where(east_convention > 180.0, east_convention - 360.0,
                                 east_convention)
    w_east = box_weight(lat, east_convention, box)[0]
    w_signed = box_weight(lat, signed_convention, box)[0]
    assert np.array_equal(w_east, w_signed)
    inside = (east_convention >= 170.0) & (east_convention <= 190.0)
    assert np.array_equal(w_east, inside.astype(float))


def test_cap_only_lowers_terrain_above_it():
    z = np.asarray([[-20.0, 100.0, 1500.0, 6000.0]])
    world = _world('[world]\nlabel = "cap"\n[world.terrain]\ncap_m = 1000.0\n')
    out = edited_terrain(z, np.asarray([0.0]), np.arange(4.0), world.terrain)
    assert out.tolist() == [[-20.0, 100.0, 1000.0, 1000.0]]
    scaled = _world('[world]\nlabel = "cap"\n[world.terrain]\nscale = 0.5\ncap_m = 1000.0\n')
    out = edited_terrain(z, np.asarray([0.0]), np.arange(4.0), scaled.terrain)
    assert out.tolist() == [[-10.0, 50.0, 750.0, 1000.0]]


# -- the frame the cold start reads --------------------------------------------

def test_the_world_frame_reads_through_to_the_frame_it_wraps():
    frame = _frame()
    frame.snow_on_sea_ice = False
    edited, record = apply_terrain_edit(frame, FLAT_CONSERVE, LEVELS)
    assert isinstance(edited, WorldFrame)
    assert edited.world_terrain is record
    assert edited.latitude is frame.latitude and edited.vertical_values is frame.vertical_values
    assert edited.valid_time == frame.valid_time and edited.snow_on_sea_ice is False
    assert edited.fields["surface_pressure"].source == "primary"
    # The input frame is not written.
    fresh = _frame()
    for name, field in frame.fields.items():
        assert np.asarray(field.values).tobytes() == np.asarray(
            fresh.fields[name].values).tobytes(), name
    import json

    json.dumps(record, allow_nan=False)
    assert record["rules"]["air_mass"].startswith("every column's surface pressure scaled")


def _analysis_cfg(tmp_path):
    """The moist smoke grid in analysis mode on the declared synthetic
    planet, as the cold start's own gates build it."""
    from pathlib import Path

    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root

    text = (Path(config_root()) / "arwen_global_moist_smoke.toml").read_text(encoding="utf-8")
    replaced = text.replace(
        "[initial]\n",
        '[initial]\nmode = "analysis"\nanalysis_grib = "unused.grib"\n'
        'analysis_mapping = "gdas-global"\n', 1)
    kept = [line for line in replaced.splitlines() if not any(
        line.strip().startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m"))]
    path = tmp_path / "analysis.toml"
    path.write_text("\n".join(kept) + '\n[statics]\nsource = "synthetic"\n', encoding="utf-8")
    return load_config(path)


def test_the_cold_start_reads_a_flattened_frame(tmp_path):
    """The edited frame goes through the unchanged cold start: a flat world
    starts with zero surface geopotential and a finite state whose surface
    pressure is the moved one's."""
    from arwen_global.analysis_initial import analysis_initial_state
    from arwen_global.runner import build_transform

    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = _frame()
    edited, record = apply_terrain_edit(frame, FLAT_CONSERVE, LEVELS)
    state, phi, provenance = analysis_initial_state(cfg, transform, frame=edited)
    b = transform.backend
    assert np.max(np.abs(b.to_numpy(phi))) == 0.0
    lnps = b.to_numpy(transform.inverse(state.atmosphere.log_surface_pressure))
    assert np.isfinite(lnps).all()
    ps_model = np.exp(lnps)
    assert record["surface_pressure_pa"]["after"]["min"] - 3000.0 < float(ps_model.min())
    assert float(ps_model.max()) < record["surface_pressure_pa"]["after"]["max"] + 3000.0


def test_the_hook_entry_is_the_terrain_edit():
    """``edit_frame(frame, world, *, levels_pa, grid)`` is the name and
    signature the cold start's hook calls (package 1.2)."""
    from arwen_global.world.frame_edits import edit_frame

    frame = _frame()
    a, rec_a = edit_frame(frame, FLAT_CONSERVE, levels_pa=LEVELS, grid=object())
    b, rec_b = apply_terrain_edit(frame, FLAT_CONSERVE, LEVELS)
    assert rec_a == rec_b
    for name in frame.fields:
        assert np.asarray(a.fields[name].values).tobytes() == np.asarray(
            b.fields[name].values).tobytes()
    with pytest.raises(TypeError):
        edit_frame(frame, FLAT_CONSERVE, LEVELS)


def test_constants_are_the_cold_start_and_statics_own():
    from arwen_global import analysis_initial, statics

    assert DRY_AIR_GAS_CONSTANT == analysis_initial.DRY_AIR_GAS_CONSTANT
    assert LAPSE_RATE_K_M == statics.TMN_LAPSE_K_PER_M
