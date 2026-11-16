"""The Grell-Freitas default D (0.2.5): gf_deep_experiment = 7,
gf_deep_physics = 34, gf_momentum_transport = true.

Every shipped config that runs Grell-Freitas moves its hash with the new
default, no other config moves, and stating the three options at their 0.2.4
values brings back each of those configs' hash before the flip exactly (the
0.2.4 hash; the T1023 day, new in 0.2.5, its hash before the flip), so a
0.2.4 checkpoint of such a config restarts under 0.2.5 with those three lines
added.  The hashes are pinned in tests/data/gf_default_d_hashes_before.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root, list_configs
from arwen_global.physics.native_options import NativePhysicsOptions

PINNED = json.loads((Path(__file__).parent / "data" / "gf_default_d_hashes_before.json")
                    .read_text(encoding="utf-8"))
BEFORE = {**PINNED["hashes_0_2_4"], **PINNED["new_in_0_2_5_before_the_flip"]}
#: The 0.2.4 values of the three options, appended to a config's native_adapter_options.
BACK = ", gf_deep_experiment = 0, gf_deep_physics = 0, gf_momentum_transport = false }"
GF_0_2_4 = {"gf_deep_experiment": 0, "gf_deep_physics": 0, "gf_momentum_transport": False}
#: The configs 0.2.5 also gives extrapolation = "settls_departure" (their
#: hashes before both changes need the shipped "settls" back too).
#: The configs 0.2.5 moves from the shipped "newton" search to "newton_ls"
#: (the T1023 day: with the drag on, "newton" refused its first step).
NEWTON_LS_FROM_0_2_5 = {"arwen_global_gdas_t1023_native_sl_si_24h"}
NEWTON_LS_BACK = ('trajectory_solver = "newton_ls"', 'trajectory_solver = "newton"')
SETTLS_DEPARTURE_FROM_0_2_5 = {
    "arwen_global_gdas_t383_native_sl_si_24h", "arwen_global_gdas_t383_native_sl_si_wall",
    "arwen_global_gdas_t533_native_sl_si_24h", "arwen_global_gdas_t533_native_sl_si_wall",
    "arwen_global_gdas_t799_native_sl_si_wall",
}
#: The days 0.2.5 also gives the orographic drag, after these pins
#: (tests/test_arwen_global_drag_default.py): it is taken back out too.
DRAG_AFTER_THE_PINS = {
    "arwen_global_gdas_t533_native_sl_si_24h", "arwen_global_gdas_t533_native_sl_si_wall",
    "arwen_global_gdas_t799_native_sl_si_wall", "arwen_global_gdas_t1023_native_sl_si_24h",
}


def gf_0_2_4(cfg):
    """``cfg`` with the three Grell-Freitas options at their 0.2.4 values: a
    pinned hash recorded before the 0.2.5 default D is compared with this.
    A config without native options is returned as it is."""
    from dataclasses import replace

    if cfg.physics_mode != "arwen-native" or not cfg.native_adapter_options:
        return cfg
    return replace(cfg, native_adapter_options={**dict(cfg.native_adapter_options), **GF_0_2_4})


def _runs_gf(cfg) -> bool:
    if cfg.physics_mode != "arwen-native" or not cfg.native_adapter_options:
        return False
    return str(cfg.native_adapter_options.get("cumulus", "gf")).lower() == "gf"


def _back(tmp_path, name):
    text = (config_root() / f"{name}.toml").read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith("native_adapter_options = {")]
    assert len(hits) == 1, name
    line = lines[hits[0]].rstrip("\n")
    assert line.endswith(" }"), name
    if name in DRAG_AFTER_THE_PINS:
        assert line.count(', orographic_drag = "kim-gwdo" }') == 1, name
        line = line.replace(', orographic_drag = "kim-gwdo" }', " }")
    lines[hits[0]] = line[:-2] + BACK + "\n"
    if name in SETTLS_DEPARTURE_FROM_0_2_5:
        lines = [ln.replace('extrapolation = "settls_departure"', 'extrapolation = "settls"')
                 for ln in lines]
    # The 0.2.5 terrain filter default (the T1023 day,
    # tests/test_arwen_global_orography_filter_defaults.py) came after the
    # pinned hashes, so it is taken back out too.
    lines = [ln for ln in lines if not ln.lstrip().startswith("orography_filter_half_degree")]
    # The T1023 day's newton_ls solver (0.2.5) came after the pins too.
    if name in NEWTON_LS_FROM_0_2_5:
        lines = [ln.replace(*NEWTON_LS_BACK) for ln in lines]
    path = tmp_path / f"{name}.toml"
    path.write_text("".join(lines), encoding="utf-8")
    return load_config(path)


def test_the_defaults_are_d():
    day = load_config(config_root() / "arwen_global_gdas_t255_native_sl_si_24h.toml")
    options = day.native_adapter_options
    assert options["gf_deep_experiment"] == 7
    assert options["gf_deep_physics"] == 34
    assert options["gf_momentum_transport"] is True
    defaults = NativePhysicsOptions.__dataclass_fields__
    assert defaults["gf_deep_experiment"].default == 7
    assert defaults["gf_deep_physics"].default == 34
    assert defaults["gf_momentum_transport"].default is True


def test_exactly_the_configs_that_run_gf_are_pinned():
    runs_gf = set()
    for name in list_configs():
        try:
            cfg = load_config(config_root() / f"{name.removesuffix('.toml')}.toml")
        except Exception:  # the four fixtures that are not run configs
            continue
        if _runs_gf(cfg):
            runs_gf.add(name.removesuffix(".toml"))
    assert runs_gf == set(BEFORE)


def test_each_gf_config_moves_and_the_three_0_2_4_values_bring_its_hash_back(tmp_path):
    for name, digest in BEFORE.items():
        shipped = load_config(config_root() / f"{name}.toml")
        assert shipped.config_hash != digest, name
        assert _back(tmp_path, name).config_hash == digest, name


def test_the_helper_and_the_three_written_lines_give_the_same_hash(tmp_path):
    name = "arwen_global_gdas_t255_native_sl_si_24h"
    shipped = load_config(config_root() / f"{name}.toml")
    assert gf_0_2_4(shipped).config_hash == _back(tmp_path, name).config_hash == BEFORE[name]
