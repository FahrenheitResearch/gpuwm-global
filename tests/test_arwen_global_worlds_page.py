"""The worlds page states only what the code and its tests hold.

``docs/ARWEN_GLOBAL_WORLDS.md`` is the page a reader opens to learn what a
world can be, what refuses it and what was measured.  Every number and name
on it is a copy of something in the package, and a copy goes stale the
first time the original moves.  So each copy is read back here against the
thing it copies:

* every test file the page names exists, so a reader can run what the page
  cites;
* every ``[world]`` example on the page parses;
* the key table names exactly the keys the parser accepts, with the
  choices and the ranges the parser holds them to;
* the preset table is the shipped presets, with their summaries and labels;
* the refusals and the limits are the preflight's own rows;
* the two world experiments it names ship;
* the sentence every result carries is the one the code prints;
* the credits on the page and in NOTICE name the record and the licence the
  map reader uses.

CPU only, no network, no card.
"""
from __future__ import annotations

from pathlib import Path
import re
import tomllib

import pytest

from arwen_global import configs_dir
from arwen_global.world import spec

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "ARWEN_GLOBAL_WORLDS.md"
NOTICE = ROOT / "NOTICE"
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _table_after(text: str, heading_cell: str) -> list[list[str]]:
    """The rows of the markdown table whose header row starts with ``heading_cell``."""

    lines = text.splitlines()
    for number, line in enumerate(lines):
        if (line.startswith("| " + heading_cell + " |")
                and number + 1 < len(lines)
                and lines[number + 1].startswith("|---")):
            rows = []
            for row in lines[number + 2:]:
                if not row.startswith("|"):
                    break
                rows.append([cell.strip() for cell in row.strip().strip("|").split("|")])
            return rows
    raise AssertionError(f"the page has no table headed {heading_cell!r}")


def _ticked(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def _fmt(value: float) -> str:
    return format(value, ",g")


def test_every_test_file_the_page_names_exists(page):
    named = sorted(set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", page)))
    assert named, "the page names no test; this gate would pass vacuously"
    missing = [name for name in named if not (ROOT / name).is_file()]
    assert missing == [], f"the page cites tests a reader cannot run: {missing}"


def test_every_test_function_the_page_names_is_in_the_suite(page):
    cited = re.findall(r"`(test_[a-z0-9_]+)`", page)
    assert cited
    suite = "\n".join(path.read_text(encoding="utf-8")
                      for path in (ROOT / "tests").glob("test_*.py"))
    missing = [name for name in cited if f"def {name}(" not in suite]
    assert missing == [], f"the page names test functions that do not exist: {missing}"


def test_every_module_the_page_names_is_in_this_tree(page):
    """The checks below skip when their module is missing; this one does not.

    A page that describes a part of the package the tree does not carry is
    wrong, whichever part it is, so the absence is a failure here and the
    skips below only keep each check's own message about its own copy.
    """

    named = set(re.findall(r"\b(arwen_global\.world(?:\.[a-z_]+)+)\b", page))
    named |= {"arwen_global.world.twin", "arwen_global.world.preflight",
              "arwen_global.world.sea", "arwen_global.world.sources.paleodem"}
    missing = []
    for name in sorted(named):
        relative = Path("src", *name.split("."))
        if not ((ROOT / relative.with_suffix(".py")).is_file()
                or (ROOT / relative / "__init__.py").is_file()):
            missing.append(name)
    assert missing == [], f"the page describes modules this tree does not carry: {missing}"


def test_every_world_example_on_the_page_parses(page):
    blocks = re.findall(r"```toml\n(.*?)```", page, flags=re.DOTALL)
    worlds = [block for block in blocks if "[world]" in block]
    assert worlds, "the page shows no [world] table"
    for block in worlds:
        world = spec.parse_world(tomllib.loads(block)["world"])
        assert not world.is_earth


def _documented_keys(rows: list[list[str]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        names = _ticked(row[0])
        prefix = names[0].rpartition(".")[0]
        for name in names:
            keys.add(name if "." in name or not prefix else f"{prefix}.{name}")
    return keys


def test_the_key_table_names_exactly_the_keys_the_parser_accepts(page):
    accepted = set()
    for table, names in spec.WORLD_KEYS.items():
        for name in names:
            full = f"{table}.{name}" if table else name
            if any(full == other for other in spec.WORLD_KEYS):
                continue  # a sub-table, not a key
            accepted.add(full)
    documented = _documented_keys(_table_after(page, "Key"))
    assert documented == accepted, (
        f"on the page and not parsed: {sorted(documented - accepted)}; "
        f"parsed and not on the page: {sorted(accepted - documented)}")


def test_the_key_table_carries_the_parsers_choices_and_ranges(page):
    rows = {_ticked(row[0])[0]: row for row in _table_after(page, "Key")}

    def accepted(key: str) -> str:
        return rows[key][2]

    for key, choices in (("terrain.air_mass", spec.AIR_MASS_RULES),
                         ("sun.orbit", spec.ORBITS),
                         ("sea.land", spec.SEA_LAND_RULES),
                         ("sea.temperature", spec.SEA_TEMPERATURE_SOURCES),
                         ("sea.ice", spec.SEA_ICE_RULES),
                         ("surface.land_cover", spec.LAND_COVERS)):
        assert _ticked(accepted(key)) == [f'"{c}"' for c in choices], key
    for key, (lo, hi) in (("terrain.scale", spec.TERRAIN_SCALE_RANGE),
                          ("terrain.cap_m", spec.TERRAIN_CAP_RANGE_M),
                          ("terrain.box.taper_deg", spec.BOX_TAPER_RANGE_DEG),
                          ("sun.obliquity_deg", spec.OBLIQUITY_RANGE_DEG),
                          ("sun.brightness", spec.BRIGHTNESS_RANGE),
                          ("sun.day_length_h", spec.DAY_LENGTH_RANGE_H),
                          ("sea.level_m", spec.SEA_LEVEL_RANGE_M),
                          ("surface.soil_wetness", spec.SOIL_WETNESS_RANGE)):
        assert accepted(key).startswith(f"{_fmt(lo)} to {_fmt(hi)}"), key
    assert f"eccentricity {_fmt(spec.ECCENTRICITY_RANGE[0])} to " \
           f"{_fmt(spec.ECCENTRICITY_RANGE[1])}" in accepted("sun.eccentricity")
    assert f"under {_fmt(spec.CO2_RANGE_PPM[1])}" in accepted("air.co2_ppm")
    assert f"under {_fmt(spec.TRACE_RANGE_PPB[1])}" in accepted("air.ch4_ppb")
    for key, earth in (("terrain.scale", spec.EARTH_TERRAIN_SCALE),
                       ("sun.obliquity_deg", spec.EARTH_OBLIQUITY_DEG),
                       ("sun.brightness", spec.EARTH_BRIGHTNESS),
                       ("sun.day_length_h", spec.EARTH_DAY_LENGTH_H)):
        assert float(rows[key][1]) == earth, key


def test_the_reserved_names_on_the_page_are_the_parsers(page):
    sentence = page.split("The names are fixed.", 1)[1].split("because", 1)[0]
    named = set(_ticked(sentence)) - {"[world]"}
    assert named == set(spec.LAUNCHER_RESERVED_KEYS)


def test_the_page_counts_the_settings_refused_by_name(page):
    words = {5: "Five", 6: "Six", 7: "Seven", 4: "Four"}
    count = len(spec.NOT_SUPPORTED_YET)
    assert f"{words[count]} settings are refused by name" in page


def test_the_preset_table_is_the_shipped_presets(page):
    twin = pytest.importorskip("arwen_global.world.twin")
    rows = _table_after(page, "Preset")
    on_page = {_ticked(row[0])[0]: (row[1], row[2]) for row in rows}
    shipped = {name: (p.summary, p.label_class) for name, p in twin.PRESETS.items()}
    assert on_page == shipped


def test_the_refusals_and_limits_are_the_preflights(page):
    preflight = pytest.importorskip("arwen_global.world.preflight")
    rows = _table_after(page, "Refusal")
    assert [_ticked(row[0])[0] for row in rows] == list(preflight.REFUSALS)
    when = {_ticked(row[0])[0]: row[1] for row in rows}
    limits = preflight.LIMITS
    assert f"under {_fmt(limits['surface_pressure_floor_pa'].value)} Pa" in when["terrain_too_tall"]
    assert f"over {_fmt(limits['surface_pressure_ceiling_pa'].value)} Pa" in when["pressure_over_ceiling"]
    assert f"passes {_fmt(limits['skin_temperature_max_k'].value)} K" in when["start_skin_over_limit"]
    for key, unit in (("skin_temperature_max_k", "K"), ("air_temperature_min_k", "K"),
                      ("radiation_table_floor_k", "K"),
                      ("ocean_water_store_kg_m2", "kg m-2")):
        assert f"{_fmt(limits[key].value)} {unit}" in page, key
    earth = preflight.MEASURED_EARTH
    for key, unit in (("skin_temperature_max_k", "K"), ("air_temperature_min_k", "K"),
                      ("ocean_store_days", "days")):
        assert f"{_fmt(earth[key].value)} {unit}" in page, key


def test_the_world_experiments_the_page_names_ship(page):
    named = set(re.findall(r"`(arwen_global_[a-z0-9_]+)`", page))
    assert named
    missing = sorted(named - set(configs_dir.list_configs()))
    assert missing == [], f"the page names experiments that do not ship: {missing}"


def test_the_one_sentence_is_the_one_the_code_prints(page):
    twin = pytest.importorskip("arwen_global.world.twin")
    sentence = twin.ONE_SENTENCE.format(days="N days")
    quoted = re.sub(r"\n>[ \t]*", " ", page)
    assert sentence in " ".join(quoted.split())


def test_the_aquaplanet_numbers_are_the_seas(page):
    sea = pytest.importorskip("arwen_global.world.sea")
    assert (f"each is {_fmt(sea.AQUAPLANET_PEAK_C)} C at the equator and 0 C at "
            f"and beyond {_fmt(sea.AQUAPLANET_EDGE_DEG)} degrees") in " ".join(page.split())
    rows = _table_after(page, "`sea.temperature`")
    assert [_ticked(row[0])[0] for row in rows] == list(spec.SEA_TEMPERATURE_PROFILES)


def test_the_credits_name_the_record_and_licence_the_reader_uses(page):
    paleodem = pytest.importorskip("arwen_global.world.sources.paleodem")
    notice = NOTICE.read_text(encoding="utf-8")
    doi = f"https://doi.org/{paleodem.RECORD_DOI}"
    assert paleodem.LICENCE == "CC-BY-4.0"
    for text, where in ((page, "the worlds page"), (notice, "NOTICE")):
        flat = " ".join(text.split())
        assert doi in flat, where
        assert "Scotese, C. R. and Wright, N. M. (2018)" in flat, where
        assert "Attribution 4.0 International" in flat, where
        assert "doi:10.1006/asle.2000.0019" in flat, where
    assert doi in paleodem.CREDIT


def test_the_readme_links_the_page():
    assert "](docs/ARWEN_GLOBAL_WORLDS.md)" in README.read_text(encoding="utf-8")
