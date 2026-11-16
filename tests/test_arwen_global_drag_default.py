"""Which shipped configs run the orographic drag (0.2.3, 0.2.5).

``orographic_drag = "kim-gwdo"`` is on in the two forecast days a user is
pointed at, T255 and T383 on the semi-Lagrangian core, and in the opt-in
day-one config; every other shipped config keeps 0.2.2's arithmetic and
hash.  MEASURED (forecast scorecard, 16 starts, 120 h, T255, GFS analysis
truth): NH Z500 RMSE at 120 h 54.3 -> 52.0 m on 14 of 16 starts.  0.2.5
also turns it on in the semi-Lagrangian T533, T799 and T1023 days
(HIRES_DRAG_0_2_5; CHANGELOG 0.2.5 gives what it measured at T799).
"""
from __future__ import annotations

from test_arwen_global_gf_default_d import NEWTON_LS_BACK, NEWTON_LS_FROM_0_2_5, gf_0_2_4

from types import SimpleNamespace

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.physics.orographic_drag import (config_turns_the_drag_on,
                                                  prepare_orographic_statistics)

#: The semi-Lagrangian days above T383, which 0.2.5 also gives the drag
#: (GFS, IFS and KIM all run orographic drag at 9-13 km), with their hashes
#: at rel/0.2.5 85cf765, before the drag: stating it off brings each back.
HIRES_DRAG_0_2_5 = {
    "arwen_global_gdas_t533_native_sl_si_24h":
        "02b281ac4ad48674c4fee1c20622163024c3c32075f00c27bec45da262c04cb6",
    "arwen_global_gdas_t533_native_sl_si_wall":
        "887bd338248e2ddf06831a70b60e57989c06bc3ea2d4fd60ff6a9647b8f89622",
    "arwen_global_gdas_t799_native_sl_si_wall":
        "f647d32050ece4d9aa4c3d028f5de74e0ce448d86db08148d04aff438d00fae7",
    "arwen_global_gdas_t1023_native_sl_si_24h":
        "9629e8af20048a2f6366197225247b8e402fa882e9cf92ec7a110f211ee5da1e",
}
DRAG_ON = {
    "arwen_global_gdas_t255_native_sl_si_24h",
    "arwen_global_gdas_t383_native_sl_si_24h",
    "arwen_global_gdas_t255_native_sl_si_24h_dayone",
    *HIRES_DRAG_0_2_5,
}


def _loadable():
    for path in sorted(config_root().glob("*.toml")):
        try:
            yield path.stem, load_config(path)
        except Exception:  # noqa: BLE001 - the four fixtures that are not run configs
            continue


def test_the_drag_is_on_in_the_forecast_days_and_nowhere_else():
    on = {name for name, cfg in _loadable() if config_turns_the_drag_on(cfg)}
    assert on == DRAG_ON


def test_the_forecast_days_read_real_statics():
    for name in DRAG_ON:
        cfg = load_config(config_root() / f"{name}.toml")
        assert cfg.statics.source == "real", name


def test_a_config_without_the_drag_builds_nothing(tmp_path):
    off = SimpleNamespace(physics_mode="arwen-native", native_adapter_options={})
    assert not config_turns_the_drag_on(off)
    assert prepare_orographic_statistics(off) is None
    reference = SimpleNamespace(physics_mode="reference",
                                native_adapter_options={"orographic_drag": "kim-gwdo"})
    assert not config_turns_the_drag_on(reference)


def _copy_with(tmp_path, name, *edits):
    text = (config_root() / f"{name}.toml").read_text(encoding="utf-8")
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    # The orography filter default (0.2.5, after these releases' hashes) is
    # taken out too: off is the terrain path those hashes were written under.
    text = "".join(ln for ln in text.splitlines(keepends=True)
                   if not ln.lstrip().startswith("orography_filter_half_degree"))
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


DRAG_OFF = ('orographic_drag = "kim-gwdo" }', 'orographic_drag = "none" }')
SETTLS_BACK = ('extrapolation = "settls_departure"', 'extrapolation = "settls"')


def test_what_brings_back_each_forecast_day_s_0_2_2_hash(tmp_path):
    """What the release pages tell a user restarting a 0.2.2 checkpoint.

    The T255 day is 0.2.2's plus the drag, so the drag off is 0.2.2's hash.
    The T383 day also moved its radiation interval (2100 -> 3000 s), so the
    drag off alone is neither release's hash; with 2100 s back it is 0.2.2's
    once the time extrapolation is the shipped "settls" again (0.2.5 gives
    the T383 day settls_departure).
    """
    t255 = _copy_with(tmp_path, "arwen_global_gdas_t255_native_sl_si_24h", DRAG_OFF)
    assert gf_0_2_4(t255).config_hash == (
        "c7a000d28c206ace5f88312d92320f4c30e72f7f0eb6d6108587022d935b4bbf")
    t383_022 = "a30d9e528428975535741609d953ee4418b452201230f40e2631d7a8dc26b768"
    drag_only = _copy_with(tmp_path, "arwen_global_gdas_t383_native_sl_si_24h",
                           DRAG_OFF)
    assert drag_only.config_hash != t383_022
    radiation_only = _copy_with(tmp_path, "arwen_global_gdas_t383_native_sl_si_24h", DRAG_OFF,
                                ("radiation_interval_s = 3000.0",
                                 "radiation_interval_s = 2100.0"))
    assert gf_0_2_4(radiation_only).config_hash != t383_022
    both = _copy_with(tmp_path, "arwen_global_gdas_t383_native_sl_si_24h", DRAG_OFF,
                      ("radiation_interval_s = 3000.0", "radiation_interval_s = 2100.0"),
                      SETTLS_BACK)
    assert gf_0_2_4(both).config_hash == t383_022


def test_run_plan_builds_the_drag_cache_before_it_prices_the_card(monkeypatch, tmp_path):
    """``run-plan``'s forecast stage does what ``run`` does before sizing."""
    from arwen_global import cli, runplan
    from arwen_global.physics import orographic_drag

    calls = []

    class Priced(Exception):
        pass

    def prepare(cfg, grid=None, *, say=None):
        calls.append(("prepare", cfg))

    def size(args, cfg, door):
        calls.append(("size", cfg))
        raise Priced

    monkeypatch.setattr(orographic_drag, "prepare_orographic_statistics", prepare)
    monkeypatch.setattr(cli, "_size_the_run", size)
    cfg = object()
    plan = SimpleNamespace(run_options={})
    with pytest.raises(Priced):
        runplan._forecast(plan, cfg, tmp_path / "day.toml",
                          runner=SimpleNamespace(run_dir=tmp_path))
    assert calls == [("prepare", cfg), ("size", cfg)]


def test_the_shipped_settls_line_brings_back_each_t383_and_finer_day_s_0_2_4_hash(tmp_path):
    """0.2.5 gives the semi-Lagrangian days from T383 up settls_departure; a
    user resuming a 0.2.4 checkpoint of one sets the shipped "settls" back,
    with 0.2.4's three Grell-Freitas values, and that is 0.2.4's hash exactly."""
    hashes_024 = {
        "arwen_global_gdas_t383_native_sl_si_24h":
            "0b7d0643f1d964b2237180c9403409450e7ae8bc2d5c2d6b34824dee18941579",
        "arwen_global_gdas_t383_native_sl_si_wall":
            "393801d1727905445b808365ef2eaf44819e153dc88959b9b688aac6a3e10ec6",
        "arwen_global_gdas_t533_native_sl_si_24h":
            "b287fdb4b6715535b19d0e6ac23c6d3992ae35bbfc8b0e75c5ba9dcf071219fe",
        "arwen_global_gdas_t533_native_sl_si_wall":
            "4c9e2321a34f3b79376e1969e21ad4c9c473ec176209b1117693c6c40d8605bd",
        "arwen_global_gdas_t799_native_sl_si_wall":
            "07703038a907051bb83f1ed9814f528c8d19b332310002975d9bed6fb2f8f4de",
    }
    for name, digest in hashes_024.items():
        shipped = load_config(config_root() / f"{name}.toml")
        assert shipped.semilag.extrapolation == "settls_departure", name
        assert shipped.config_hash != digest, name
        # the days above T383 also had no drag in 0.2.4
        edits = (SETTLS_BACK, DRAG_OFF) if name in HIRES_DRAG_0_2_5 else (SETTLS_BACK,)
        assert gf_0_2_4(_copy_with(tmp_path, name, *edits)).config_hash == digest, name


def test_stating_the_drag_off_brings_back_each_high_resolution_day_s_0_2_5_hash(tmp_path):
    """A checkpoint of a T533, T799 or T1023 day written before the drag
    default restarts with ``orographic_drag = "none"`` (or the key taken out
    of native_adapter_options): either spelling is that day's hash at
    rel/0.2.5 85cf765 exactly, terrain filter and all."""
    for name, digest in HIRES_DRAG_0_2_5.items():
        text = (config_root() / f"{name}.toml").read_text(encoding="utf-8")
        assert text.count(DRAG_OFF[0]) == 1, name
        if name in NEWTON_LS_FROM_0_2_5:  # its solver also came after the pins
            text = text.replace(*NEWTON_LS_BACK)
        shipped = load_config(config_root() / f"{name}.toml")
        assert config_turns_the_drag_on(shipped), name
        assert shipped.config_hash != digest, name
        for spelling, (old, new) in (("none", DRAG_OFF),
                                     ("key out", (", " + DRAG_OFF[0], " }"))):
            path = tmp_path / f"{name}.toml"
            path.write_text(text.replace(old, new), encoding="utf-8")
            off = load_config(path)
            assert not config_turns_the_drag_on(off), (name, spelling)
            assert off.config_hash == digest, (name, spelling)
