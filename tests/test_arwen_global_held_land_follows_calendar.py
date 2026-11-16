"""A held sea does not hold the land on a long run (round 9, the jet report).

Until 0.2.5 ``[land] greenness`` defaulted to "start" under ``[ocean] mode =
"hold"``, so a held run of weeks kept the start date's vegetation fraction,
leaf area and background albedo: a run from 1 March kept March vegetation
in May.  A dated analysis run longer than 16 days now follows the calendar
under hold too; a forecast of 16 days or less keeps the start, and no
config hash moves (a held run checkpointed under 0.2.4 restarts).
"""
from __future__ import annotations

from types import SimpleNamespace

from arwen_global.boundary_options import (LONG_RUN_THRESHOLD_S, LandOptions, OceanOptions,
                                           long_dated_run)


def _cfg(duration_s, initial_mode="analysis", source="real"):
    return SimpleNamespace(duration_s=duration_s, initial_mode=initial_mode,
                           statics=SimpleNamespace(source=source))


def test_only_a_dated_analysis_run_on_real_statics_past_16_days_is_long():
    assert long_dated_run(_cfg(LONG_RUN_THRESHOLD_S + 600.0))
    assert not long_dated_run(_cfg(LONG_RUN_THRESHOLD_S))
    assert not long_dated_run(_cfg(86400.0))
    assert not long_dated_run(_cfg(75 * 86400.0, initial_mode="idealized"))
    assert not long_dated_run(_cfg(75 * 86400.0, source="analytic"))


def test_a_held_long_run_follows_the_calendar_and_a_held_forecast_keeps_the_start():
    hold = OceanOptions(mode="hold")
    anomaly = OceanOptions(mode="anomaly")
    assert LandOptions().resolved(hold) == "start"
    assert LandOptions().resolved(hold, long_run=False) == "start"
    assert LandOptions().resolved(hold, long_run=True) == "daily"
    assert LandOptions().resolved(anomaly) == "daily"
    assert LandOptions().resolved(anomaly, long_run=True) == "daily"
    # A stated greenness is what it says, long or not.
    assert LandOptions("start").resolved(hold, long_run=True) == "start"
    assert LandOptions("daily").resolved(hold) == "daily"


def test_the_default_never_joins_the_identity():
    hold = OceanOptions(mode="hold")
    assert LandOptions().identity(hold) is None
    assert LandOptions("start").identity(hold) is None
    assert LandOptions("daily").identity(hold) == {"greenness": "daily"}
    assert LandOptions("daily").identity(OceanOptions(mode="anomaly")) is None


def test_an_explicit_start_on_a_held_long_run_never_shares_the_default_s_identity():
    hold = OceanOptions(mode="hold")
    # The long-run default follows the calendar and keeps the 0.2.4 identity.
    assert LandOptions().identity(hold, long_run=True) is None
    # A stated "start" keeps the start's land there, so it must hash apart.
    assert LandOptions("start").identity(hold, long_run=True) == {"greenness": "start"}
    # On a forecast "start" is what the default does, as before.
    assert LandOptions("start").identity(hold, long_run=False) is None
    assert LandOptions("daily").identity(hold, long_run=True) == {"greenness": "daily"}
    assert LandOptions("start").identity(OceanOptions(mode="anomaly"), long_run=True) \
        == {"greenness": "start"}


def test_the_config_hash_of_a_held_long_run_tells_a_stated_start_from_the_default(tmp_path):
    from dataclasses import replace
    from pathlib import Path

    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root

    text = (config_root() / "arwen_global_gdas_t255_native_24h.toml").read_text(encoding="utf-8")

    def cfg(extra, name):
        path = tmp_path / name
        path.write_text(text + "\n" + extra, encoding="utf-8")
        return load_config(Path(path))

    held = cfg('[ocean]\nmode = "hold"\n', "a.toml")
    start = cfg('[ocean]\nmode = "hold"\n[land]\ngreenness = "start"\n', "b.toml")
    assert held.statics.source == "real" and held.initial_mode == "analysis"
    # A forecast: the stated start is the default, one hash (as in 0.2.4).
    assert start.config_hash == held.config_hash
    long_held = replace(held, duration_s=LONG_RUN_THRESHOLD_S + 3600.0)
    long_start = replace(start, duration_s=LONG_RUN_THRESHOLD_S + 3600.0)
    assert long_dated_run(long_held) and long_dated_run(long_start)
    assert long_start.config_hash != long_held.config_hash
    # The default's identity is the one a 0.2.4 held long run wrote.
    assert "land" not in long_held.config_identity
    assert long_start.config_identity["land"] == {"greenness": "start"}
