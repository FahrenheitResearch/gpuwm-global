"""Which shipped configs filter their terrain by default (0.2.5):
``[initial] orography_filter_half_degree``.

The configs in SETS_THE_FILTER state the key and move their hash; every
other shipped config keeps its 0.2.5 hash exactly; and every shipped config
with the key taken out (or stated as 0.0) is back to its 0.2.5 hash, the
hash recorded at rel/0.2.5 (e9e202a, before the key existed) in
tests/data/orography_filter_hashes_before.json.  So a checkpoint of such a
config written before the default restarts with ``orography_filter_half_degree
= 0.0`` added to its [initial] table.

To give another config the filter: add the TOML line to it and its name to
SETS_THE_FILTER.  The fixture is not touched (it is the hash before any
default).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.config import load_config  # noqa: E402
from arwen_global.configs_dir import config_root, list_configs  # noqa: E402

KEY = "orography_filter_half_degree"
PINNED = json.loads((Path(__file__).parent / "data" / "orography_filter_hashes_before.json")
                    .read_text(encoding="utf-8"))
BEFORE = PINNED["hashes_0_2_5"]
DO_NOT_LOAD = set(PINNED["do_not_load"])
#: The shipped configs that state the filter, by name.  T1023 only for now.
SETS_THE_FILTER = {
    "arwen_global_gdas_t1023_native_sl_si_24h",
    "arwen_global_gdas_t799_native_sl_si_wall",
    "arwen_global_gdas_t533_native_sl_si_24h",
    "arwen_global_gdas_t533_native_sl_si_wall",
}
_KEY_LINE = re.compile(rf"^\s*{KEY}\s*=")
#: The days 0.2.5 also gives the orographic drag after these hashes
#: (tests/test_arwen_global_drag_default.py): it is taken back out too.
DRAG_AFTER_THE_FIXTURE = {
    "arwen_global_gdas_t1023_native_sl_si_24h",
    "arwen_global_gdas_t799_native_sl_si_wall",
    "arwen_global_gdas_t533_native_sl_si_24h",
    "arwen_global_gdas_t533_native_sl_si_wall",
}
_DRAG = (', orographic_drag = "kim-gwdo" }', " }")
from test_arwen_global_gf_default_d import NEWTON_LS_BACK, NEWTON_LS_FROM_0_2_5  # noqa: E402


def _names():
    return sorted(name.removesuffix(".toml") for name in list_configs())


def _text(name):
    return (config_root() / f"{name}.toml").read_text(encoding="utf-8")


def _states_the_key(name) -> bool:
    return any(_KEY_LINE.match(line) for line in _text(name).splitlines())


def _off(tmp_path, name, spelling=None):
    """``name`` with every line of the key removed, or with it stated as
    ``spelling`` in its place."""
    lines = []
    for line in _text(name).splitlines(keepends=True):
        if _KEY_LINE.match(line):
            if spelling is not None:
                lines.append(f"{KEY} = {spelling}\n")
            continue
        if name in DRAG_AFTER_THE_FIXTURE and line.startswith("native_adapter_options = {"):
            assert line.count(_DRAG[0]) == 1, name
            line = line.replace(*_DRAG)
        if name in NEWTON_LS_FROM_0_2_5:
            line = line.replace(*NEWTON_LS_BACK)
        lines.append(line)
    path = tmp_path / f"{name}.toml"
    path.write_text("".join(lines), encoding="utf-8")
    return load_config(path)


def test_the_fixture_covers_every_shipped_config():
    assert set(_names()) == set(BEFORE) | DO_NOT_LOAD
    assert not set(BEFORE) & DO_NOT_LOAD


def test_the_configs_that_state_the_key_are_exactly_the_listed_ones():
    assert {name for name in _names() if _states_the_key(name)} == SETS_THE_FILTER
    for name in _names():
        if name in DO_NOT_LOAD:
            continue
        value = getattr(load_config(config_root() / f"{name}.toml"), KEY)
        assert (value > 0.0) == (name in SETS_THE_FILTER), (name, value)


def test_the_t1023_day_filters_at_degree_350():
    cfg = load_config(config_root() / "arwen_global_gdas_t1023_native_sl_si_24h.toml")
    assert getattr(cfg, KEY) == 350.0


def test_the_shipped_files_that_are_not_run_configs_still_do_not_load():
    for name in sorted(DO_NOT_LOAD):
        with pytest.raises(Exception):
            load_config(config_root() / f"{name}.toml")


def test_a_config_without_the_key_keeps_its_0_2_5_hash():
    for name, digest in BEFORE.items():
        if name in SETS_THE_FILTER:
            continue
        assert load_config(config_root() / f"{name}.toml").config_hash == digest, name


def test_a_config_that_sets_the_key_moves_its_hash():
    for name in SETS_THE_FILTER:
        assert load_config(config_root() / f"{name}.toml").config_hash != BEFORE[name], name


def test_every_config_with_the_filter_off_is_back_to_its_0_2_5_hash(tmp_path):
    for name, digest in BEFORE.items():
        assert _off(tmp_path, name).config_hash == digest, name
    for name in SETS_THE_FILTER:
        for spelling in ("0.0", "0"):
            assert _off(tmp_path, name, spelling).config_hash == BEFORE[name], (name, spelling)
