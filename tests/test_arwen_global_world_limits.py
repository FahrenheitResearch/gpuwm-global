"""The limits as data (worlds package 3.5): every fixture world earns its
label and its rules, the data agrees with the [world] table parser, and a
row of the data file, not code, moves a verdict."""
from __future__ import annotations

import copy
from importlib.resources import files
import json
import math

import pytest

from arwen_global.world import EARTH, NOT_SUPPORTED_YET as SPEC_NOT_SUPPORTED, parse_world
from arwen_global.world import spec
from arwen_global.world.limits import (
    EARTH_CLASS,
    ILLUSTRATION,
    LABEL_ORDER,
    LIMITS_FILE,
    NOT_SUPPORTED_YET,
    SCIENCE,
    TABLE_RANGE_RULE,
    check_fixtures,
    classify,
    contract_keys,
    label_of,
    limits,
    load_table,
    measured_earth,
    parse_table,
    worse,
)


def _raw() -> dict:
    text = files("arwen_global").joinpath("data", LIMITS_FILE).read_text(encoding="utf-8")
    return json.loads(text)


# -- the table ------------------------------------------------------------------


def test_the_shipped_table_parses_and_every_number_is_finite():
    table = load_table()
    assert table.labels == LABEL_ORDER
    assert len(table.rules) >= 30
    for key, row in table.limits.items():
        assert math.isfinite(row.value), key
        assert row.unit and row.name and row.source, key
    for key, row in table.measured_earth.items():
        assert math.isfinite(row.value), key
        assert row.unit and row.source, key


def test_about_thirty_fixture_worlds_each_earn_their_label_and_rules():
    results = check_fixtures()
    assert len(results) >= 30
    bad = [r.describe() for r in results if not r.ok]
    assert not bad, "\n".join(bad)


def test_every_rule_is_pinned_by_a_fixture():
    table = load_table()
    pinned = {rule for f in table.fixtures for rule in f.rules}
    unpinned = [rule.id for rule in table.rules if rule.id not in pinned]
    assert not unpinned, unpinned


def test_every_label_is_earned_by_some_fixture():
    assert {f.label for f in load_table().fixtures} == set(LABEL_ORDER)


# -- the plan's own examples, written here and not read from the fixtures --------


def test_mars_is_refused_by_three_named_rules():
    mars = {
        "label": "mars", "gravity_m_s2": 3.72, "radius_km": 3389.5,
        "year_length_d": 687.0,
        "air": {"composition": "co2", "surface_pressure_pa": 610.0},
        "sun": {"obliquity_deg": 25.19, "brightness": 0.431, "day_length_h": 24.62},
    }
    got = classify(mars)
    assert got.label == NOT_SUPPORTED_YET
    assert {hit.id for hit in got.refusals} == {"planet_constants", "thin_air", "year_length"}
    # The settings the [world] table can hold are within its ranges: the
    # refusal is the data's, not the parser's.
    assert got.contract_error is None


def test_eight_times_co2_is_labelled_unvalidated():
    got = classify({"label": "co2-x8", "air": {"co2_ppm": 8 * 280.0}})
    assert got.label == ILLUSTRATION
    assert "co2_unvalidated" in got.rules
    texts = " ".join(hit.limit for hit in got.hits)
    assert "unvalidated above eight times pre-industrial" in texts
    # Four times pre-industrial is inside the range the plan shows.
    assert "co2_unvalidated" not in classify({"label": "x4", "air": {"co2_ppm": 1120.0}}).rules


def test_no_mountains_is_inside_and_is_science():
    table = {"label": "flat-earth", "terrain": {"scale": 0.0}}
    for request in (table, parse_world(table)):
        got = classify(request)
        assert got.label == SCIENCE
        assert got.rules == ("mountains_lowered",)
        assert got.contract_error is None


def test_pangea_is_an_illustration():
    pangea = parse_world({
        "label": "pangea-250ma",
        "sea": {"temperature": "zonal-mean", "ice": "none"},
        "surface": {"folder": "pangea-250ma.world", "soil_wetness": 0.3},
    })
    assert label_of(pangea) == ILLUSTRATION


@pytest.mark.parametrize("request_", [None, {}, EARTH,
                                      {"terrain": {"air_mass": "fill"}},
                                      {"run": {"regional_zoom": True}}])
def test_earth_is_earth_with_no_rule(request_):
    got = classify(request_)
    assert got.label == EARTH_CLASS
    assert got.hits == ()


# -- the data agrees with the parser ------------------------------------------------


def test_a_world_the_labels_let_run_is_one_the_table_parses():
    for result in check_fixtures():
        if result.got.label != NOT_SUPPORTED_YET:
            assert result.got.contract_error is None, result.fixture.id


def test_a_refusal_on_table_keys_alone_is_also_the_parsers():
    table = load_table()
    contract = contract_keys()
    for result in check_fixtures():
        for hit in result.got.refusals:
            if hit.id == TABLE_RANGE_RULE:
                continue
            if table.rule(hit.id).keys() <= contract:
                assert result.got.contract_error is not None, (result.fixture.id, hit.id)


def test_no_fixture_falls_through_to_the_parser_or_the_fallback():
    table = load_table()
    for result in check_fixtures():
        assert TABLE_RANGE_RULE not in result.got.rules, result.fixture.id
        assert table.fallback.id not in result.got.rules, result.fixture.id


def test_a_parser_refusal_no_row_names_is_still_refused_by_name():
    got = classify({"label": "odd-orbit", "sun": {"orbit": "eccentric", "eccentricity": 0.5}})
    assert got.label == NOT_SUPPORTED_YET
    assert got.rules[-1] == TABLE_RANGE_RULE
    assert "world.sun.eccentricity" in got.hits[-1].limit


def test_every_setting_the_table_refuses_by_name_is_a_request_key_a_rule_reads():
    table = load_table()
    read = set().union(*(rule.keys() for rule in table.rules))
    for key in SPEC_NOT_SUPPORTED:
        assert key in table.request_keys, key
        assert key in read, key


def test_the_table_ranges_are_the_parsers():
    rows = limits()
    assert rows["day_length_min_h"].value == spec.DAY_LENGTH_RANGE_H[0]
    assert rows["day_length_max_h"].value == spec.DAY_LENGTH_RANGE_H[1]
    assert rows["brightness_min"].value == spec.BRIGHTNESS_RANGE[0]
    assert rows["brightness_max"].value == spec.BRIGHTNESS_RANGE[1]
    assert rows["terrain_scale_max"].value == spec.TERRAIN_SCALE_RANGE[1]
    assert rows["co2_radiation_max_ppm"].value == spec.CO2_RANGE_PPM[1]
    assert rows["trace_gas_max_ppb"].value == spec.TRACE_RANGE_PPB[1]


def test_a_parsed_world_and_its_table_earn_the_same_rules():
    for fixture in load_table().fixtures:
        flat = json.dumps(fixture.request)
        if any(k in fixture.request for k in ("facts", "run")) or fixture.label == NOT_SUPPORTED_YET:
            continue
        if any(word in flat for word in ("gravity", "radius", "year_length", "composition",
                                         "surface_pressure", "star", "locked", "ocean",
                                         "ice_grows")):
            continue
        assert classify(parse_world(fixture.request)).rules == classify(fixture.request).rules, (
            fixture.id)


# -- limits and measured figures ---------------------------------------------------


def test_the_preflights_rows_are_here_unchanged():
    rows = limits()
    assert rows["skin_temperature_max_k"].value == 355.0
    assert rows["skin_temperature_max_k"].caution == 5.0
    assert rows["air_temperature_min_k"].value == 140.0
    assert rows["radiation_table_floor_k"].value == 160.0
    assert rows["surface_pressure_ceiling_pa"].value == 109_663.0
    assert rows["surface_pressure_ceiling_pa"].caution == 1_116.0
    assert rows["surface_pressure_floor_pa"].value == 45_000.0
    assert rows["ocean_water_store_kg_m2"].value == 500.0
    earth = measured_earth()
    assert earth["skin_temperature_max_k"].value == 336.58
    assert earth["air_temperature_min_k"].value == 175.17
    assert earth["surface_pressure_min_pa"].value == 50_168.9
    assert earth["surface_pressure_max_pa"].value == 107_577.2
    assert earth["ocean_store_days"].value == 23.9


def test_a_row_and_not_code_moves_a_verdict():
    raw = _raw()
    mars = next(f["request"] for f in raw["fixtures"] if f["id"] == "mars")
    assert "thin_air" in classify(mars).rules
    raw["limits"]["thin_air_surface_pressure_pa"]["value"] = 500.0
    moved = parse_table(raw)
    assert "thin_air" not in classify(mars, moved).rules
    # A longer measured store lets a 30-day no-mountains run through.
    raw = _raw()
    request = {"label": "flat-earth", "terrain": {"scale": 0.0}, "facts": {"run_days": 30.0}}
    assert classify(request).label == NOT_SUPPORTED_YET
    raw["measured_earth"]["ocean_store_days"]["value"] = 40.0
    assert classify(request, parse_table(raw)).label == SCIENCE


def test_a_change_no_row_names_is_an_illustration_by_the_fallback():
    raw = _raw()
    raw["rules"] = [r for r in raw["rules"] if r["id"] not in ("tilt", "tilt_high_with_land")]
    raw["fixtures"] = []
    got = classify({"label": "tilted", "sun": {"obliquity_deg": 40.0}}, parse_table(raw))
    assert got.label == ILLUSTRATION
    assert got.rules == ("unlisted_change",)


# -- refusals of a bad table or request -------------------------------------------


def test_an_unknown_request_key_is_refused():
    with pytest.raises(ValueError, match="sun.obliqity_deg"):
        classify({"label": "typo", "sun": {"obliqity_deg": 90.0}})


@pytest.mark.parametrize("mutate, match", [
    (lambda raw: raw["rules"][0]["when"]["all"][1].update(key="sea.lund"), "unknown key"),
    (lambda raw: raw["rules"][2]["when"].update(limit="no_such_limit"), "unknown limit"),
    (lambda raw: raw["rules"][1].update(id=raw["rules"][0]["id"]), "not unique"),
    (lambda raw: raw["rules"][1].update(label="earth"), "has label"),
    (lambda raw: raw["fixtures"][0].update(rules=["no_such_rule"]), "unknown rules"),
    (lambda raw: raw.update(labels=list(reversed(raw["labels"]))), "labels must be"),
    (lambda raw: raw["rules"][1]["when"].update(op="~"), "unknown op"),
])
def test_a_bad_table_is_refused_on_load(mutate, match):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        parse_table(raw)


def test_a_label_only_gets_worse():
    assert worse(SCIENCE, ILLUSTRATION) == ILLUSTRATION
    assert worse(ILLUSTRATION, SCIENCE) == ILLUSTRATION
    assert worse(EARTH_CLASS, NOT_SUPPORTED_YET) == NOT_SUPPORTED_YET
    for a in LABEL_ORDER:
        for b in LABEL_ORDER:
            assert LABEL_ORDER.index(worse(a, b)) >= LABEL_ORDER.index(a)
    with pytest.raises(ValueError):
        worse("fine", SCIENCE)


def test_the_record_is_plain_json():
    got = classify({"label": "on-its-side", "sun": {"obliquity_deg": 90.0}})
    record = got.record()
    assert json.loads(json.dumps(record)) == record
    assert record["label"] == ILLUSTRATION
    assert [r["id"] for r in record["rules"]] == ["tilt", "tilt_high_with_land"]
