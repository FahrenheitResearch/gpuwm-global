"""Which shipped configs run the orographic drag (0.2.3).

``orographic_drag = "kim-gwdo"`` is on in the two forecast days a user is
pointed at, T255 and T383 on the semi-Lagrangian core, and in the opt-in
day-one config; every other shipped config keeps 0.2.2's arithmetic and
hash.  MEASURED (forecast scorecard, 16 starts, 120 h, T255, GFS analysis
truth): NH Z500 RMSE at 120 h 54.3 -> 52.0 m on 14 of 16 starts.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.physics.orographic_drag import (config_turns_the_drag_on,
                                                  prepare_orographic_statistics)

DRAG_ON = {
    "arwen_global_gdas_t255_native_sl_si_24h",
    "arwen_global_gdas_t383_native_sl_si_24h",
    "arwen_global_gdas_t255_native_sl_si_24h_dayone",
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
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


DRAG_OFF = ('orographic_drag = "kim-gwdo" }', 'orographic_drag = "none" }')


def test_what_brings_back_each_forecast_day_s_0_2_2_hash(tmp_path):
    """What the release pages tell a user restarting a 0.2.2 checkpoint.

    The T255 day is 0.2.2's plus the drag, so the drag off is 0.2.2's hash.
    The T383 day also moved its radiation interval (2100 -> 3000 s), so the
    drag off alone is neither release's hash; with 2100 s back it is 0.2.2's.
    """
    t255 = _copy_with(tmp_path, "arwen_global_gdas_t255_native_sl_si_24h", DRAG_OFF)
    assert t255.config_hash == (
        "c7a000d28c206ace5f88312d92320f4c30e72f7f0eb6d6108587022d935b4bbf")
    t383_022 = "a30d9e528428975535741609d953ee4418b452201230f40e2631d7a8dc26b768"
    drag_only = _copy_with(tmp_path, "arwen_global_gdas_t383_native_sl_si_24h",
                           DRAG_OFF)
    assert drag_only.config_hash != t383_022
    both = _copy_with(tmp_path, "arwen_global_gdas_t383_native_sl_si_24h", DRAG_OFF,
                      ("radiation_interval_s = 3000.0", "radiation_interval_s = 2100.0"))
    assert both.config_hash == t383_022


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
