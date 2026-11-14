"""The world statement and the preflight (worlds plan, package 1.6).

Before any card time a world is read against the frame it starts from:
each refusal fires by name with its text, the predicted start matches the
closed form of a constant-lapse atmosphere, Earth reads nothing, and the
statement is the same text and the same hash every time it is made.

The real-frame check at the end runs only when ``ARWEN_WORLD_GDAS_FRAME``
names a GDAS 0.25 degree pgrb2 analysis and the engine's decoder is
installed: a terrain scale of 1.5 on it is refused before any card time.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from arwen_global.constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2
from arwen_global.vertical import HybridCoordinate
from arwen_global.world import EARTH, parse_world
from arwen_global.world.preflight import (
    LIMITS,
    MARGIN_KEYS,
    REFUSALS,
    FrameView,
    WorldRefused,
    box_weight,
    edited_terrain,
    predict_surface_pressure,
    preflight,
)
from arwen_global.world.statement import (
    EARTH_CLASS,
    ILLUSTRATION,
    NOT_SUPPORTED_YET,
    SCIENCE,
    describe,
    world_class,
)

LAPSE = 0.0065
T_SEA = 288.15
P_SEA = 101_325.0
EXPONENT = GRAVITY_M_S2 / (DRY_AIR_GAS_CONSTANT * LAPSE)
LEVELS_PA = np.array([1000.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0,
                      30000.0, 35000.0, 40000.0, 45000.0, 50000.0, 55000.0, 60000.0,
                      65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 92500.0,
                      95000.0, 97500.0, 100000.0])
LAT = np.array([-60.0, -20.0, 20.0, 60.0])
LON = np.arange(8) * 45.0


def pressure_at(z_m):
    """A dry atmosphere with 6.5 K/km from 288.15 K at 1013.25 hPa."""
    return P_SEA * (1.0 - LAPSE * np.asarray(z_m, dtype=np.float64) / T_SEA) ** EXPONENT


def temperature_at(p_pa):
    return T_SEA * (np.asarray(p_pa, dtype=np.float64) / P_SEA) ** (1.0 / EXPONENT)


def lapse_frame(terrain_m, *, temperature_offset=None, humidity=0.0):
    """A constant-lapse frame on ``terrain_m`` (lat, lon)."""
    terrain = np.asarray(terrain_m, dtype=np.float64)
    shape = (LEVELS_PA.size,) + terrain.shape
    temperature = np.broadcast_to(temperature_at(LEVELS_PA)[:, None, None], shape).copy()
    if temperature_offset is not None:
        temperature = temperature + temperature_offset
    return FrameView(
        latitude_deg=LAT, longitude_deg=LON, levels_pa=LEVELS_PA,
        temperature_k=temperature,
        specific_humidity=np.full(shape, humidity),
        surface_pressure_pa=pressure_at(terrain),
        terrain_m=terrain,
        skin_temperature_k=np.full(terrain.shape, 300.0),
        land_fraction=(terrain > 0.0).astype(np.float64),
    )


def world(text: str):
    import tomllib

    return parse_world(tomllib.loads(text)["world"])


FLAT_FILL = world('[world]\nlabel = "flat"\n[world.terrain]\nscale = 0.0\nair_mass = "fill"\n')
FLAT = world('[world]\nlabel = "flat-earth"\n[world.terrain]\nscale = 0.0\n')
TALL = world('[world]\nlabel = "tall"\n[world.terrain]\nscale = 1.5\n')
SIDE = world('[world]\nlabel = "on-its-side"\n[world.sun]\nobliquity_deg = 90.0\n')
VERTICAL = HybridCoordinate.surface_stretched(40, 100.0)


def one_mountain(height_m):
    terrain = np.zeros((LAT.size, LON.size))
    terrain[2, 3] = height_m
    return terrain


# -- the prediction against its closed form --------------------------------


@pytest.mark.parametrize("height_m", [500.0, 2000.0, 4500.0])
def test_lowering_matches_the_closed_form_of_a_constant_lapse_atmosphere(height_m):
    view = lapse_frame(one_mountain(height_m))
    ps = predict_surface_pressure(FLAT_FILL, view, np.zeros_like(view.terrain_m))
    assert abs(ps[2, 3] - P_SEA) < 1.0
    # Every other column kept its pressure bit for bit.
    others = np.ones(ps.shape, dtype=bool)
    others[2, 3] = False
    assert np.array_equal(ps[others], view.surface_pressure_pa[others])


@pytest.mark.parametrize("raise_m", [500.0, 2000.0, 4500.0])
def test_raising_integrates_the_analysis_air_to_its_closed_form(raise_m):
    view = lapse_frame(one_mountain(1000.0))
    target = view.terrain_m.copy()
    target[2, 3] += raise_m
    ps = predict_surface_pressure(FLAT_FILL, view, target)
    assert abs(ps[2, 3] - pressure_at(1000.0 + raise_m)) < 10.0


def test_a_world_that_moves_no_terrain_returns_the_frame_pressure_bit_for_bit():
    view = lapse_frame(one_mountain(3000.0))
    ps = predict_surface_pressure(FLAT, view, view.terrain_m)
    assert np.array_equal(ps, view.surface_pressure_pa)


def test_conserve_keeps_the_global_mean_surface_pressure():
    view = lapse_frame(one_mountain(3000.0))
    ps = predict_surface_pressure(FLAT, view, np.zeros_like(view.terrain_m))
    w = view.area_weights
    assert math.isclose(np.sum(w * ps), np.sum(w * view.surface_pressure_pa), rel_tol=1e-14)
    filled = predict_surface_pressure(FLAT_FILL, view, np.zeros_like(view.terrain_m))
    assert np.sum(w * filled) > np.sum(w * ps)


def test_held_humidity_makes_the_new_air_lighter_and_the_pressure_lower():
    dry = lapse_frame(one_mountain(3000.0))
    moist = lapse_frame(one_mountain(3000.0), humidity=0.01)
    zero = np.zeros_like(dry.terrain_m)
    assert (predict_surface_pressure(FLAT_FILL, moist, zero)[2, 3]
            < predict_surface_pressure(FLAT_FILL, dry, zero)[2, 3])


# -- the terrain rule ------------------------------------------------------


def test_scale_cap_and_box_edit_the_terrain_and_nothing_else():
    terrain = np.full((LAT.size, LON.size), 2000.0)
    capped = world('[world]\nlabel = "cap"\n[world.terrain]\ncap_m = 1500.0\n')
    assert np.all(edited_terrain(capped, terrain, LAT, LON) == 1500.0)
    assert np.array_equal(edited_terrain(EARTH, terrain, LAT, LON), terrain)
    boxed = world('[world]\nlabel = "box"\n[world.terrain]\nscale = 0.0\n'
                  '[world.terrain.box]\nsouth_deg = 10.0\nnorth_deg = 30.0\n'
                  'west_deg = 80.0\neast_deg = 100.0\ntaper_deg = 0.0\n')
    edited = edited_terrain(boxed, terrain, LAT, LON)
    assert edited[2, 2] == 0.0          # 20 N 90 E, inside
    assert edited[2, 3] == 2000.0       # 20 N 135 E, outside
    assert edited[1, 2] == 2000.0       # 20 S 90 E, outside


def test_the_box_tapers_linearly_and_crosses_the_dateline():
    taper = world('[world]\nlabel = "taper"\n[world.terrain]\nscale = 0.0\n'
                  '[world.terrain.box]\nsouth_deg = -10.0\nnorth_deg = 10.0\n'
                  'west_deg = 170.0\neast_deg = -170.0\ntaper_deg = 20.0\n')
    lat = np.array([0.0, 15.0, 30.0])
    lon = np.array([180.0, 175.0, 160.0, 0.0])
    w = box_weight(taper, lat, lon)
    assert w[0, 0] == 1.0 and w[0, 1] == 1.0
    assert w[1, 0] == pytest.approx(0.75)     # 5 degrees north of the box
    assert w[0, 2] == pytest.approx(0.5)      # 10 degrees west of it
    assert w[2, 0] == 0.0 and w[0, 3] == 0.0


# -- the refusals ----------------------------------------------------------


def test_too_tall_terrain_is_refused_by_name_with_its_text():
    view = lapse_frame(one_mountain(6000.0))
    report = preflight(TALL, view=view, vertical=VERTICAL, duration_s=432000.0)
    assert [r.key for r in report.refusals] == ["terrain_too_tall"]
    text = report.refusals[0].text
    assert text.startswith(REFUSALS["terrain_too_tall"])
    assert "45,000 Pa floor" in text and "20.00N 135.00E" in text
    assert "world.terrain.scale" in text
    with pytest.raises(WorldRefused, match="terrain_too_tall") as caught:
        report.raise_if_refused()
    assert caught.value.label == "tall"


def test_earth_s_own_peaks_are_never_refused_as_too_tall():
    # A frame whose own peak already sits under the floor: a world that
    # lowers terrain elsewhere must not be refused for Earth's own column.
    terrain = one_mountain(7000.0)
    terrain[1, 1] = 8000.0
    view = lapse_frame(terrain)
    assert view.surface_pressure_pa[2, 3] < LIMITS["surface_pressure_floor_pa"].value
    lowered = world('[world]\nlabel = "cap"\n[world.terrain]\ncap_m = 7500.0\n'
                    'air_mass = "fill"\n')
    report = preflight(lowered, view=view, vertical=VERTICAL)
    assert report.prediction["columns_moved"] == 1
    assert report.refusals == ()


def test_a_start_over_the_radiation_ceiling_is_refused_with_the_remedy():
    # A cold plateau, 4,000 m at about 216 K, flattened under "fill" reads
    # about 111.8 kPa: past the radiation tables' 109,663 Pa and under the
    # 115,000 Pa the radiation accepts since 0.2.1, so it is no longer refused.
    view = lapse_frame(one_mountain(4000.0), temperature_offset=-45.0)
    report = preflight(FLAT_FILL, view=view, vertical=VERTICAL)
    assert report.refusals == ()
    assert 109_663.0 < report.prediction["surface_pressure_max_pa"] < 115_000.0
    # Colder and taller, 4,500 m at about 206 K: about 117 kPa, refused.
    terrain = one_mountain(4500.0)
    view = lapse_frame(terrain, temperature_offset=-55.0)
    report = preflight(FLAT_FILL, view=view, vertical=VERTICAL)
    keys = [r.key for r in report.refusals]
    assert keys == ["pressure_over_ceiling"]
    text = report.refusals[0].text
    assert text.startswith(REFUSALS["pressure_over_ceiling"])
    assert "115,000 Pa" in text and "air_mass = 'conserve'" in text
    assert report.prediction["surface_pressure_max_pa"] > LIMITS[
        "surface_pressure_ceiling_pa"].value


def test_the_mover_s_own_output_is_what_the_refusal_reads_when_given():
    view = lapse_frame(one_mountain(1000.0))
    over = view.surface_pressure_pa.copy()
    over[0, 0] = 116_000.0
    report = preflight(FLAT, view=view, edited_surface_pressure_pa=over)
    assert [r.key for r in report.refusals] == ["pressure_over_ceiling"]
    assert report.prediction["pressure_rule"] == "the pressure mover's own output"


def test_earth_reads_nothing_and_refuses_nothing():
    class Exploding:
        @property
        def fields(self):
            raise AssertionError("the preflight read Earth's frame")

    report = preflight(EARTH, frame=Exploding(), levels_pa=LEVELS_PA)
    assert report.refusals == () and report.margins == () and report.prediction is None


# -- the margins -----------------------------------------------------------


def test_the_four_margins_come_in_order_with_a_rule_each():
    view = lapse_frame(one_mountain(3000.0))
    skin_k = view.skin_temperature_k.copy()
    land = view.land_fraction.copy()
    skin_k[1, 1], land[1, 1] = 330.0, 1.0    # a hot desert at sea level
    view = dataclasses.replace(view, skin_temperature_k=skin_k, land_fraction=land)
    report = preflight(FLAT, view=view, vertical=VERTICAL, duration_s=5 * 86400.0)
    assert tuple(m.key for m in report.margins) == MARGIN_KEYS
    assert all(m.rule for m in report.margins)
    skin, cold, water, bottom = report.margins
    # The plateau's skin, warmed 19.5 K, stays under the desert's.
    assert report.prediction["world_land_skin_max_k"] == 330.0
    assert skin.value == pytest.approx(336.58) and skin.status == "inside"
    assert cold.limit == 140.0 and cold.status == "inside"
    assert water.margin == pytest.approx(23.9 - 5.0)
    assert bottom.value == pytest.approx(
        VERTICAL.delta_a[-1] + VERTICAL.delta_b[-1] * report.prediction[
            "surface_pressure_min_pa"])


def test_a_start_skin_warmed_past_the_limit_is_refused_and_is_the_margin():
    view = lapse_frame(one_mountain(3000.0))
    skin_k = view.skin_temperature_k.copy()
    skin_k[2, 3] = 340.0                       # a sunlit plateau, warmed 19.5 K
    view = dataclasses.replace(view, skin_temperature_k=skin_k)
    report = preflight(FLAT, view=view, vertical=VERTICAL)
    skin = report.margins[0]
    assert skin.value == pytest.approx(340.0 + LAPSE * 3000.0)
    assert skin.status == "over" and "stops the first radiation call" in skin.rule
    assert "20.00N 135.00E" in skin.rule
    assert [r.key for r in report.refusals] == ["start_skin_over_limit"]
    text = report.refusals[0].text
    assert text.startswith(REFUSALS["start_skin_over_limit"])
    assert "359.50 K at 20.00N 135.00E" in text and "3,000 m removed" in text
    # The same skin with the terrain left alone is Earth's, and not refused.
    assert preflight(SIDE, view=view).refusals == ()


def test_a_run_longer_than_the_water_store_lasts_is_over_but_not_refused():
    report = preflight(SIDE, duration_s=40 * 86400.0)
    water = report.margins[2]
    assert water.status == "over" and water.margin < 0.0
    assert not report.refused


def test_a_large_tilt_leaves_the_hot_and_cold_edges_unbounded_and_says_why():
    report = preflight(SIDE, duration_s=5 * 86400.0)
    skin, cold = report.margins[:2]
    assert skin.status == "unknown" and "355 K within a season" in skin.rule
    assert cold.status == "unknown" and "polar night" in cold.rule


def test_a_brighter_sun_moves_the_margins_by_its_fourth_root():
    bright = world('[world]\nlabel = "bright"\n[world.sun]\nbrightness = 1.1\n')
    skin, cold, water, _ = preflight(bright, duration_s=86400.0).margins
    assert skin.value == pytest.approx(336.58 * 1.1 ** 0.25)
    assert cold.value == pytest.approx(175.17 * 1.1 ** 0.25)
    assert water.value == pytest.approx(23.9 / 1.1)


# -- the statement ---------------------------------------------------------


def test_the_labels_follow_section_two_of_the_plan():
    assert world_class(EARTH) == EARTH_CLASS
    assert world_class(FLAT) == SCIENCE
    assert world_class(SIDE) == ILLUSTRATION
    assert world_class(TALL) == ILLUSTRATION
    co2 = world('[world]\nlabel = "hot"\n[world.air]\nco2_ppm = 3000.0\n')
    statement = describe(co2, duration_s=86400.0)
    assert "unvalidated above eight times pre-industrial" in statement.text()
    water = world('[world]\nlabel = "water"\n[world.terrain]\nscale = 0.0\n'
                  '[world.sea]\nland = "none"\ntemperature = "aqua-control"\nice = "none"\n')
    assert world_class(water) == ILLUSTRATION
    assert all(v.label != SCIENCE for v in describe(water).verdicts)


def test_a_refused_world_is_labelled_not_supported_yet():
    view = lapse_frame(one_mountain(6000.0))
    statement = describe(TALL, view=view, vertical=VERTICAL, duration_s=86400.0)
    assert statement.world_class == NOT_SUPPORTED_YET and statement.refused
    assert "REFUSED before any card time" in statement.text()


def test_the_statement_names_every_part_and_hashes_the_same_every_time():
    view = lapse_frame(one_mountain(3000.0))
    first = describe(FLAT, view=view, vertical=VERTICAL, duration_s=5 * 86400.0)
    second = describe(FLAT, view=lapse_frame(one_mountain(3000.0)), vertical=VERTICAL,
                      duration_s=5 * 86400.0)
    text = first.text()
    assert text == second.text() and first.sha256 == second.sha256
    for needle in ("World 'flat-earth': science", FLAT.identity_hash,
                   "This is the weather of the first 5 days after Earth was changed, "
                   "over a sea surface that does not respond. It is not this "
                   "world's climate.",
                   "terrain.scale: 0 (Earth: 1)", "6.5 K per km",
                   "specific humidity is held", "air mass conserved",
                   "Margins:", "355 K", "140 K", "500 kg m-2", "bottom layer",
                   "Not refused."):
        assert needle in text, needle
    record = first.record()
    assert json.loads(json.dumps(record)) == record
    assert record["identity"] == FLAT.identity_hash and record["class"] == SCIENCE
    assert [m["key"] for m in record["margins"]] == list(MARGIN_KEYS)


def test_a_different_world_or_frame_moves_the_statement_hash():
    view = lapse_frame(one_mountain(3000.0))
    base = describe(FLAT, view=view, vertical=VERTICAL, duration_s=86400.0).sha256
    assert describe(FLAT_FILL, view=view, vertical=VERTICAL, duration_s=86400.0).sha256 != base
    assert describe(FLAT, view=lapse_frame(one_mountain(3100.0)), vertical=VERTICAL,
                    duration_s=86400.0).sha256 != base


def test_earth_s_statement_says_nothing_was_changed():
    statement = describe(EARTH, duration_s=86400.0)
    assert statement.world_class == EARTH_CLASS
    assert "Every setting is Earth's" in statement.text()
    assert statement.record()["identity"] == "none"


def test_the_statement_text_carries_no_host_or_path():
    folder = world('[world]\nlabel = "pangea"\n[world.surface]\n'
                   'folder = "/somewhere/private/pangea-250ma.world"\nsoil_wetness = 0.3\n'
                   '[world.sea]\ntemperature = "zonal-mean"\nice = "none"\n')
    text = describe(folder, duration_s=20 * 86400.0).text()
    assert "pangea-250ma.world" in text and "/somewhere" not in text
    assert "Earth's air over this map" in text


def test_describe_path_reads_a_run_config_with_its_length_and_grid(tmp_path):
    from arwen_global.configs_dir import config_root
    from arwen_global.world.statement import describe_path

    smoke = (config_root() / "arwen_global_moist_smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "flat.toml"
    path.write_text(smoke + "\n" + FLAT.to_toml(), encoding="utf-8")
    statement = describe_path(path, read_frame=True)   # no analysis: no frame
    assert statement.world == FLAT and statement.world_class == SCIENCE
    bottom = statement.report.margins[3]
    assert bottom.status == "unknown" and "no frame was read" in bottom.rule
    earth = tmp_path / "earth.toml"
    earth.write_text(smoke, encoding="utf-8")
    assert describe_path(earth, read_frame=True).world_class == EARTH_CLASS


# -- a real frame ----------------------------------------------------------


REAL_FRAME = os.environ.get("ARWEN_WORLD_GDAS_FRAME")


@pytest.mark.skipif(not REAL_FRAME, reason="set ARWEN_WORLD_GDAS_FRAME to a GDAS "
                    "0.25 degree pgrb2 analysis to run the real-frame check")
def test_a_terrain_scale_of_one_and_a_half_on_a_real_frame_is_refused(tmp_path):
    from arwen_global.analysis_initial import analysis_levels_pa, resolve_analysis_mapping
    from arwen_global.mapped_source_compat import decode_through_engine

    mapping = resolve_analysis_mapping("gdas-global")
    decoded = decode_through_engine(mapping, [Path(REAL_FRAME)],
                                    scratch_destination=tmp_path)
    frame = decoded.frames[0]
    levels, _unit = analysis_levels_pa(np.asarray(frame.vertical_values), mapping)
    view = FrameView.from_frame(frame, levels)
    report = preflight(TALL, view=view, vertical=VERTICAL, duration_s=86400.0)
    assert "terrain_too_tall" in [r.key for r in report.refusals]
    flat = preflight(FLAT, view=view, vertical=VERTICAL, duration_s=86400.0)
    assert "terrain_too_tall" not in [r.key for r in flat.refusals]
    print(describe(TALL, report=report, duration_s=86400.0).text())
    print(describe(FLAT, report=flat, duration_s=86400.0).text())
