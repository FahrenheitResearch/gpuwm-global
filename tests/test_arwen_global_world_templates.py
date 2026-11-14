"""The two shipped world templates (worlds plan, package 1.7).

``arwen_global_world_t255_no_mountains`` and
``arwen_global_world_t255_on_its_side`` are the T255 forecast-day
experiment ``arwen_global_gdas_t255_native_sl_si_24h`` with a ``[world]``
table appended and nothing else changed but the run name, as that day
read before 0.2.3 turned its orographic drag on: the drag reads Earth's
sub-grid terrain and is refused under a world that moves the terrain, so
both templates keep the drag off and are held to the base without it.  A
launcher picks a template by name from ``gpuwm-global configs
--paths`` and sets seven keys in it by pattern, each of which has to occur
exactly once outside a comment; these tests hold both templates to that
and to the world each one says it is.
"""
from __future__ import annotations

import re
import tomllib

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root, list_configs
from arwen_global.world import EARTH

BASE = "arwen_global_gdas_t255_native_sl_si_24h"

#: Template name -> (label, the settings that differ from Earth's).
WORLDS = {
    "arwen_global_world_t255_no_mountains": (
        "no-mountains", {"terrain.scale": 0.0}),
    "arwen_global_world_t255_on_its_side": (
        "on-its-side", {"sun.obliquity_deg": 90.0}),
}

#: The keys a launcher sets by pattern (a ``set_toml_value`` that
#: replaces one line): each must occur exactly once outside a comment.
#: ``analysis_mapping`` is set only for an ERA5 start, and is there once.
LAUNCHER_KEYS = ("name", "duration_s", "output_interval_s", "analysis_grib",
               "start_time_utc", "analysis_mapping", "radiation_column_chunk")


def _text(name: str) -> str:
    return (config_root() / f"{name}.toml").read_text(encoding="utf-8")


def _body(text: str) -> str:
    """The file from its first table on: the header comment left out."""

    return text[text.index("[arwen_global]"):]


def _without_world(text: str) -> str:
    """The file with its trailing ``[world]`` block (and comment) removed."""

    return text[:text.index("\n# The planet this forecast is on.")]


#: What 0.2.3 added to the base day and the templates do not carry.
DRAG_OPTION = ', orographic_drag = "kim-gwdo"'
DRAG_NOTE = '# orographic_drag = "kim-gwdo" (WRF gwd_opt = 1), on since 0.2.3'


def _without_the_drag(text: str) -> str:
    """The base day as it read before 0.2.3: its drag note and option out."""

    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(DRAG_NOTE))
    end = next(i for i in range(start, len(lines)) if lines[i] == "[physics]")
    assert DRAG_OPTION in text
    return "\n".join(lines[:start] + lines[end:]).replace(DRAG_OPTION, "", 1)


def _launcher_count(text: str, key: str) -> int:
    """How many lines a launcher's own pattern would set ``key`` on."""

    pattern = (r'((?:^\s*|[{,]\s*)' + re.escape(key)
               + r'\s*=\s*)("[^"]*"|[^,}#\n]+)')
    return sum(1 for line in text.split("\n")
               if not line.lstrip().startswith("#") and re.search(pattern, line))


def test_both_world_templates_ship():
    shipped = set(list_configs())
    for name in WORLDS:
        assert name in shipped, name


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_a_world_template_is_its_base_experiment_plus_one_world_table(name):
    base = _without_the_drag(_text(BASE))
    text = _text(name)
    expected_name = name.replace("_", "-")
    stripped = _without_world(_body(text)).replace(
        f'name = "{expected_name}"',
        'name = "arwen-global-gdas-t255-native-sl-si-24h"', 1)
    assert stripped == _body(base)
    # ...and the appended block holds nothing but the [world] table.
    appended = text[text.index("\n# The planet this forecast is on."):]
    document = tomllib.loads(appended)
    assert list(document) == ["world"]


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_a_world_template_is_the_world_it_says_it_is(name):
    label, settings = WORLDS[name]
    cfg = load_config(config_root() / f"{name}.toml")
    world = cfg.world
    assert not world.is_earth
    assert world.label == label
    assert world.changed_settings() == list(settings)
    part, _, key = next(iter(settings)).partition(".")
    assert getattr(getattr(world, part), key) == settings[f"{part}.{key}"]
    # No mountains conserves Earth's amount of air (the owner's choice);
    # on its side keeps Earth's terrain, coast, sea and air.
    if name.endswith("no_mountains"):
        assert world.terrain.air_mass == "conserve"
        assert world.changes_terrain and not world.changes_sun
    else:
        assert world.changes_sun and not world.changes_terrain
        assert not world.changes_rotation
    assert not world.changes_coast and not world.changes_sea
    assert not world.changes_air and not world.is_invented
    assert cfg.config_identity["world"] == world.identity


@pytest.mark.parametrize("name", sorted(WORLDS))
def test_without_its_world_a_template_hashes_as_its_base(name, tmp_path):
    """The only physics a template changes is the world's."""

    base_path = tmp_path / "base" / f"{BASE}.toml"
    base_path.parent.mkdir()
    base_path.write_text(_without_the_drag(_text(BASE)), encoding="utf-8")
    base = load_config(base_path)
    text = _without_world(_text(name)).replace(
        f'name = "{name.replace("_", "-")}"',
        'name = "arwen-global-gdas-t255-native-sl-si-24h"', 1)
    path = tmp_path / f"{BASE}.toml"
    path.write_text(text, encoding="utf-8")
    earth = load_config(path)
    assert earth.world == EARTH
    assert earth.config_hash == base.config_hash
    world = load_config(config_root() / f"{name}.toml")
    assert world.config_hash != base.config_hash


def test_the_two_worlds_have_different_identities():
    a, b = (load_config(config_root() / f"{name}.toml").world
            for name in sorted(WORLDS))
    assert a.identity_hash != b.identity_hash
    assert a.token != b.token


@pytest.mark.parametrize("name", sorted(WORLDS) + [BASE])
@pytest.mark.parametrize("key", LAUNCHER_KEYS)
def test_a_launcher_finds_each_key_it_sets_exactly_once(name, key):
    assert _launcher_count(_text(name), key) == 1, (name, key)
