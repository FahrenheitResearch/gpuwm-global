"""A long run radiates the greenhouse gases of each call's valid year.

THE BREAKAGE THIS PREVENTS.  The native radiation selected CO2, CH4 and N2O
once, from the run's start time, and held them for the whole run: a run
started on 1 November 2024 and run for a year radiated 2024's 422.79 ppm
through all of 2025, whose published mean is 425.64 ppm.  Under
``[radiation] ghg_year = "valid"`` the gases are re-selected at every
radiation call from the call's valid year; the default, ``"start"``, keeps
the old selection and every config hash written under it.

What is pinned here, on known answers:

* the value used on given dates, on both sides of 31 December and over a
  leap 29 February, and past the table's last year (held, and the receipt
  says so in a line of its own);
* that the selection lands in the radiation the call reads: the carried
  driver's ``select_trace_gases`` writes the slots the gas fill reads, in
  the constructor's order;
* that a run crossing 31 December (and one crossing 29 February) restarted
  part-way through reproduces the straight run bit for bit, through the
  suite, its runtime and the checkpoint's array-and-JSON round trip, with a
  stand-in radiation whose heating is proportional to the CO2 it is handed;
* that the config identity is unchanged when the key is unset or "start",
  and moves when it is "valid".

The real driver's solar geometry and ozone over the same edges, and the
real suite's restart across them, are in
``test_greenhouse_gases_by_year_cuda.py``.
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
from arwen_global.physics.greenhouse_gases import (
    GHG_YEAR_RULES,
    greenhouse_gas_record,
    greenhouse_gas_schedule,
    present_day_greenhouse_gases,
)
from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.state import PhysicsState

UTC = timezone.utc
T255_NATIVE = config_root() / "arwen_global_gdas_t255_native_24h.toml"


def _utc(*parts):
    return datetime(*parts, tzinfo=UTC)


def _options(**extra):
    return NativePhysicsOptions.from_mapping({
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": "2024-11-01T00:00:00Z",
        **extra,
    })


# ---------------------------------------------------------------------------
# The value used on given dates.
# ---------------------------------------------------------------------------

def test_the_year_end_switches_co2_to_the_next_years_row():
    """31 December 23:00 reads 2024's row; 1 January 01:00 reads 2025's."""
    before = present_day_greenhouse_gases(_utc(2024, 12, 31, 23))
    after = present_day_greenhouse_gases(_utc(2025, 1, 1, 1))
    assert before == {"co2": 422.79 * 1e-6, "ch4": 1929.41 * 1e-9, "n2o": 337.71 * 1e-9}
    assert after == {"co2": 425.64 * 1e-6, "ch4": 1935.94 * 1e-9, "n2o": 338.85 * 1e-9}
    # The switch is at 00:00:00 UTC exactly, not a day late or a step early.
    assert present_day_greenhouse_gases(
        _utc(2024, 12, 31, 23, 59, 59, 999_999))["co2"] == 422.79 * 1e-6
    assert present_day_greenhouse_gases(_utc(2025, 1, 1))["co2"] == 425.64 * 1e-6


def test_the_runtime_hands_each_calls_valid_year_under_valid_and_the_start_year_under_start():
    valid = _options(ghg_year="valid")
    start = _options()
    late = _utc(2025, 1, 1, 1)
    assert valid.greenhouse_gas_overrides(late)["co2"] == 425.64 * 1e-6
    # Without a valid time the overrides are the start's, under either rule:
    # it is what the constructor is handed, and what "start" holds all run.
    assert valid.greenhouse_gas_overrides()["co2"] == 422.79 * 1e-6
    assert start.greenhouse_gas_overrides()["co2"] == 422.79 * 1e-6
    # A declared CO2 still wins; CH4 and N2O still follow the year.
    fixed = _options(ghg_year="valid", trace_co2_ppm=369.55)
    assert fixed.greenhouse_gas_overrides(late) == {
        "co2": 369.55 * 1e-6, "ch4": 1935.94 * 1e-9, "n2o": 338.85 * 1e-9}


def test_a_leap_day_and_the_month_ends_are_ordinary_days_of_their_year():
    for instant in (_utc(2024, 2, 28, 23), _utc(2024, 2, 29, 0),
                    _utc(2024, 2, 29, 23, 59, 59), _utc(2024, 3, 1, 1),
                    _utc(2024, 1, 31, 23, 30), _utc(2024, 4, 30, 23, 30)):
        assert present_day_greenhouse_gases(instant)["co2"] == 422.79 * 1e-6, instant


def test_a_year_past_the_table_holds_its_last_row_and_says_so():
    held = present_day_greenhouse_gases(_utc(2026, 7, 1))
    assert held == {"co2": 425.64 * 1e-6, "ch4": 1935.94 * 1e-9, "n2o": 338.85 * 1e-9}
    assert present_day_greenhouse_gases(_utc(2031, 1, 1))["co2"] == 425.64 * 1e-6
    # The default rule on a 2026 start: the line names the year and the value.
    record = _options(start_time_utc="2026-09-01T00:00:00Z").greenhouse_gas_record()
    assert record["ghg_year"] == "start"
    assert record["table_year"] == {"co2": 2025, "ch4": 2025, "n2o": 2025}
    assert record["held_past_table"].startswith("valid year 2026 is past")
    assert "CO2 425.64 ppm (2025)" in record["held_past_table"]
    assert "CH4 1935.94 ppb (2025)" in record["held_past_table"]
    assert "N2O 338.85 ppb (2025)" in record["held_past_table"]
    assert "nothing is projected forward" in record["held_past_table"]


def test_a_run_inside_the_table_carries_no_hold_line():
    record = _options().greenhouse_gas_record()
    assert record["ghg_year"] == "start" and "held_past_table" not in record
    assert record["valid_year"] == 2024
    assert record["mole_fraction"]["co2"] == 422.79 * 1e-6


def test_the_valid_rule_records_the_year_by_year_schedule_of_the_whole_run():
    """A V2-shaped run: 1 November 2025, 365 days, into 2026."""
    options = _options(ghg_year="valid", start_time_utc="2025-11-01T00:00:00Z")
    record = options.greenhouse_gas_record(365 * 86_400.0)
    assert record["ghg_year"] == "valid"
    assert [row["valid_year"] for row in record["by_valid_year"]] == [2025, 2026]
    first, second = record["by_valid_year"]
    assert first["from_utc"] == "2025-11-01T00:00:00Z"
    assert second["from_utc"] == "2026-01-01T00:00:00Z"
    assert first["mole_fraction"]["co2"] == 425.64 * 1e-6
    assert second["table_year"] == {"co2": 2025, "ch4": 2025, "n2o": 2025}
    assert record["held_past_table"].startswith("valid year 2026 is past")
    # Further out, the span is named as a span.
    long = _options(ghg_year="valid", start_time_utc="2026-11-01T00:00:00Z")
    line = long.greenhouse_gas_record(400 * 86_400.0)["held_past_table"]
    assert line.startswith("valid years 2026-2027 are past")
    # And the record refuses to guess a run's end.
    with pytest.raises(ValueError, match="duration"):
        options.greenhouse_gas_record()


def test_the_schedule_starts_each_year_at_midnight_and_carries_overrides():
    rows = greenhouse_gas_schedule(_utc(2023, 12, 31, 23), _utc(2025, 1, 1, 1))
    assert [r["from_utc"] for r in rows] == [
        "2023-12-31T23:00:00Z", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    assert [r["mole_fraction"]["co2"] for r in rows] == [
        419.35 * 1e-6, 422.79 * 1e-6, 425.64 * 1e-6]
    record = greenhouse_gas_record(
        _utc(2024, 12, 31, 23), {"co2": 369.55 * 1e-6}, rule="valid",
        end=_utc(2025, 1, 1, 1))
    assert [r["mole_fraction"]["co2"] for r in record["by_valid_year"]] == [
        369.55 * 1e-6, 369.55 * 1e-6]
    assert [r["mole_fraction"]["ch4"] for r in record["by_valid_year"]] == [
        1929.41 * 1e-9, 1935.94 * 1e-9]
    assert record["overridden"] == ["co2"]
    with pytest.raises(ValueError, match="ghg_year"):
        greenhouse_gas_record(_utc(2024, 1, 1), rule="calendar")


def test_the_option_admits_two_rules_and_refuses_anything_else():
    assert GHG_YEAR_RULES == ("start", "valid")
    assert NativePhysicsOptions.__dataclass_fields__["ghg_year"].default == "start"
    with pytest.raises(ValueError, match="ghg_year"):
        _options(ghg_year="calendar")


# ---------------------------------------------------------------------------
# The selection lands where the gas fill reads it (the carried driver).
# ---------------------------------------------------------------------------

def _bare_driver(rrtmgp):
    """The carried driver with only the gas slots its fill reads: built
    without the constructor, which needs a card for the tables."""
    driver = rrtmgp.RRTMGPRadiation.__new__(rrtmgp.RRTMGPRadiation)
    driver.trace_vmr = {gas: 1.0 * 1e-9 for gas in rrtmgp._RFMIP_GAS_NAMES}
    return driver


def test_the_carried_driver_reselects_the_slots_the_gas_fill_reads():
    from arwen_global.core import rrtmgp

    driver = _bare_driver(rrtmgp)
    order = list(driver.trace_vmr)
    options = _options(ghg_year="valid")
    for instant, co2 in ((_utc(2024, 12, 31, 23), 422.79 * 1e-6),
                         (_utc(2025, 1, 1, 1), 425.64 * 1e-6),
                         (_utc(2024, 12, 31, 23), 422.79 * 1e-6)):
        selected = driver.select_trace_gases(
            instant, options.greenhouse_gas_overrides(instant))
        assert selected["co2"] == co2 and driver.trace_vmr["co2"] == co2
    assert driver.trace_vmr["ch4"] == 1929.41 * 1e-9
    # The write order the fused fill keeps is the constructor's.
    assert list(driver.trace_vmr) == order
    # Without overrides it is the carried table's own CO2, by the same year.
    assert driver.select_trace_gases(_utc(2025, 1, 1, 1))["co2"] == 425.64 * 1e-6
    with pytest.raises(ValueError, match="unknown trace gas"):
        driver.select_trace_gases(_utc(2025, 1, 1), {"sf6": 1.0e-11})


def test_the_solar_constant_runs_on_wrfs_365_day_orbit_across_the_year_end():
    """1370 * ECCFAC (Paltridge and Platt) on WRF's 0-based julian day over
    365 days: 1418.0185 W/m2 at 1 January 00Z (julian 0: 1370 * (1.000110 +
    0.034221 + 0.000719)); a leap year's 31 December 12Z (julian 365.5) is
    the following 1 January 12Z (julian 0.5), 1418.0335; a leap day is a
    common year's 1 March (julian 59.5).  WRF's convention, kept: it
    moves no run, and the slip it makes is at most one day of orbit."""
    from arwen_global.core.rrtmgp import RRTMGPRadiation

    solcon = RRTMGPRadiation._solar_constant
    assert solcon(_utc(2025, 1, 1)) == pytest.approx(1418.0185, abs=1e-9)
    assert solcon(_utc(2024, 12, 31, 12)) == pytest.approx(
        solcon(_utc(2025, 1, 1, 12)), abs=1e-9)
    assert solcon(_utc(2025, 1, 1, 12)) == pytest.approx(1418.0335265, abs=1e-6)
    assert solcon(_utc(2024, 2, 29, 12)) == pytest.approx(
        solcon(_utc(2023, 3, 1, 12)), abs=1e-9)
    assert solcon(_utc(2024, 2, 29, 12)) == pytest.approx(1395.6566996, abs=1e-6)
    # The docstring's own pin: julian 92.75.
    assert solcon(_utc(2025, 4, 3, 18)) == pytest.approx(1369.704, abs=1e-3)


# ---------------------------------------------------------------------------
# Restart exactness across 31 December and 29 February, through the suite.
# ---------------------------------------------------------------------------

#: The stand-in radiation's longwave heating per unit CO2 mole fraction,
#: K/s.  2024's and 2025's rows then heat by 4.2279e-4 and 4.2564e-4 K/s,
#: 1.4e-5 K apart over a 5 s step, above float32's spacing at 300 K.
HEATING_PER_CO2 = 1.0


def _recording_modules(seen):
    from arwen_global.core import rrtmgp
    from test_arwen_global_level5_native import _fake_modules

    modules = _fake_modules()
    stock = modules["arwen_global.core.rrtmgp"].RRTMGPRadiation

    class Radiation(stock):
        # The carried method itself, on a stand-in whose gas slots start
        # where the constructor leaves them (start-year overrides).
        select_trace_gases = rrtmgp.RRTMGPRadiation.select_trace_gases

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.trace_vmr = {gas: 1.0 * 1e-9 for gas in rrtmgp._RFMIP_GAS_NAMES}
            self.trace_vmr.update(kwargs["trace_gas_overrides"])

        def __call__(self, *, atmosphere, fields, state, cfg):
            result = super().__call__(
                atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
            co2 = self.trace_vmr["co2"]
            seen.append(co2)
            result.rthratenlw = np.full(
                atmosphere["temperature"].shape, HEATING_PER_CO2 * co2,
                np.float32)
            return result

    modules["arwen_global.core.rrtmgp"] = SimpleNamespace(RRTMGPRadiation=Radiation)
    return modules


def _suite_options(start, rule):
    return {
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": start,
        "radiation_interval_s": 10.0,
        "land_surface_interval_s": 10.0,
        "ghg_year": rule,
    }


def _through_checkpoint(physics_state):
    """The checkpoint's shape: every array under physics__ in an npz, the
    metadata through JSON (checkpoint.bundle_arrays)."""
    buffer = io.BytesIO()
    np.savez(buffer, **{
        f"physics__{name}": np.asarray(value)
        for name, value in physics_state.arrays.items()
    })
    buffer.seek(0)
    stored = np.load(buffer)
    return PhysicsState(
        schema=physics_state.schema,
        arrays={n.removeprefix("physics__"): stored[n] for n in stored.files},
        metadata=json.loads(json.dumps(physics_state.metadata)),
    )


def _run(start, rule, *, steps=3, restart_after=None):
    """``steps`` radiation calls 10 s apart from ``start``, through one suite
    or, with ``restart_after``, through a fresh suite, runtime and
    radiation after that many calls with the state through a checkpoint."""
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite
    from test_arwen_global_level5_native import _advance, _exchange

    seen = []

    def fresh():
        return ArwenCudaColumnSuite(
            _suite_options(start, rule), array_module=np,
            modules=_recording_modules(seen))

    exchange = _exchange()
    suite = fresh()
    result = None
    for call in range(steps):
        current = exchange if result is None else _advance(exchange, result, 10.0 * call)
        if restart_after is not None and call == restart_after:
            suite = fresh()
            current = replace(current, physics_state=_through_checkpoint(
                current.physics_state))
        result = suite.step(current)
    return result, seen


def _assert_same(straight, resumed):
    from arwen_global.constants import WATER_SPECIES

    for name in ("u", "v", "theta", *WATER_SPECIES):
        assert np.array_equal(np.asarray(getattr(straight, name)),
                              np.asarray(getattr(resumed, name))), name
    resumed_surface = resumed.surface.arrays()
    for name, value in straight.surface.arrays().items():
        assert np.array_equal(np.asarray(value),
                              np.asarray(resumed_surface[name])), name
    assert straight.physics_state.arrays.keys() == resumed.physics_state.arrays.keys()
    for name, value in straight.physics_state.arrays.items():
        assert np.array_equal(np.asarray(value),
                              np.asarray(resumed.physics_state.arrays[name])), name
    assert straight.physics_state.metadata == resumed.physics_state.metadata


YEAR_END = "2024-12-31T23:59:50Z"


def test_a_run_crossing_31_december_switches_at_the_first_call_of_the_new_year():
    _, valid = _run(YEAR_END, "valid")
    _, start = _run(YEAR_END, "start")
    # Calls at 23:59:50, 00:00:00 and 00:00:10.
    assert valid == [422.79 * 1e-6, 425.64 * 1e-6, 425.64 * 1e-6]
    assert start == [422.79 * 1e-6, 422.79 * 1e-6, 422.79 * 1e-6]


def test_inside_one_year_the_valid_rule_is_the_start_rule_bit_for_bit():
    straight, _ = _run("2024-06-30T23:59:50Z", "start")
    valid, _ = _run("2024-06-30T23:59:50Z", "valid")
    _assert_same(straight, valid)


def test_the_two_rules_part_where_the_year_changes():
    start, _ = _run(YEAR_END, "start")
    valid, _ = _run(YEAR_END, "valid")
    assert not np.array_equal(np.asarray(start.theta), np.asarray(valid.theta))


@pytest.mark.parametrize("restart_after", [1, 2])
def test_a_run_crossing_31_december_restarts_bit_for_bit(restart_after):
    """Restarted before the crossing (the resumed segment crosses) and
    after it (the resumed driver is built at the start year and must
    re-select the new one at its first call)."""
    straight, seen_straight = _run(YEAR_END, "valid")
    resumed, seen_resumed = _run(YEAR_END, "valid", restart_after=restart_after)
    assert seen_resumed == seen_straight
    _assert_same(straight, resumed)


@pytest.mark.parametrize("start", ["2024-02-28T23:59:50Z", "2024-02-29T23:59:50Z"])
def test_a_run_crossing_29_february_restarts_bit_for_bit(start):
    straight, seen = _run(start, "valid")
    resumed, _ = _run(start, "valid", restart_after=1)
    assert seen == [422.79 * 1e-6] * 3
    _assert_same(straight, resumed)


# ---------------------------------------------------------------------------
# The config: [radiation] ghg_year, and the identity.
# ---------------------------------------------------------------------------

def _with_radiation(tmp_path: Path, source: Path, body: str, *, physics=None) -> Path:
    text = source.read_text(encoding="utf-8")
    if physics is not None:
        text = text.replace(*physics)
    target = tmp_path / source.name
    target.write_text(text + "\n" + body, encoding="utf-8")
    return target


def test_the_identity_is_unchanged_unset_or_start_and_moves_under_valid(tmp_path):
    base = load_config(T255_NATIVE)
    (tmp_path / "start").mkdir()
    (tmp_path / "valid").mkdir()
    start = load_config(_with_radiation(
        tmp_path / "start", T255_NATIVE, '[radiation]\nghg_year = "start"\n'))
    valid = load_config(_with_radiation(
        tmp_path / "valid", T255_NATIVE, '[radiation]\nghg_year = "valid"\n'))
    assert "ghg_year" not in base.config_identity["native_adapter_options"]
    assert start.config_hash == base.config_hash
    assert valid.config_hash != base.config_hash
    assert valid.config_identity["native_adapter_options"]["ghg_year"] == "valid"
    # The runtime reads the value the operator wrote.
    assert base.native_adapter_options["ghg_year"] == "start"
    assert valid.native_adapter_options["ghg_year"] == "valid"


def test_the_key_is_refused_where_it_would_change_nothing_or_is_spelled_twice(tmp_path):
    smoke = config_root() / "arwen_global_moist_smoke.toml"
    with pytest.raises(ValueError, match="radiates no greenhouse gas"):
        load_config(_with_radiation(tmp_path, smoke, '[radiation]\nghg_year = "valid"\n'))
    (tmp_path / "bad").mkdir()
    with pytest.raises(ValueError, match="radiation.ghg_year must be one of"):
        load_config(_with_radiation(
            tmp_path / "bad", T255_NATIVE, '[radiation]\nghg_year = "calendar"\n'))
    (tmp_path / "other").mkdir()
    with pytest.raises(ValueError, match=r"unknown keys in \[radiation\]"):
        load_config(_with_radiation(
            tmp_path / "other", T255_NATIVE, '[radiation]\nsolar_constant = 1361\n'))
    (tmp_path / "twice").mkdir()
    text = T255_NATIVE.read_text(encoding="utf-8")
    assert "native_adapter_options = {" in text
    # The native options' spelling is refused alone as well as beside
    # [radiation]: one key, one place to read it.
    for body in ("", '[radiation]\nghg_year = "valid"\n'):
        with pytest.raises(ValueError, match="spelled once"):
            load_config(_with_radiation(
                tmp_path / "twice", T255_NATIVE, body,
                physics=("native_adapter_options = {",
                         'native_adapter_options = { ghg_year = "valid",')))


def test_the_receipt_states_the_schedule_of_a_valid_run(tmp_path):
    from arwen_global.runner import _greenhouse_gas_receipt

    cfg = load_config(_with_radiation(
        tmp_path, T255_NATIVE, '[radiation]\nghg_year = "valid"\n'))
    clock = SimpleNamespace(dated=True, iso="2024-12-31T12:00:00Z")
    # The shipped 24 h run, dated by its clock 12 h before a year end.
    assert cfg.duration_s == 86_400.0
    record = _greenhouse_gas_receipt(cfg, clock)
    assert record["ghg_year"] == "valid"
    assert [row["valid_year"] for row in record["by_valid_year"]] == [2024, 2025]
    assert [row["mole_fraction"]["co2"] for row in record["by_valid_year"]] == [
        422.79 * 1e-6, 425.64 * 1e-6]
    assert record["by_valid_year"][1]["from_utc"] == "2025-01-01T00:00:00Z"
    assert math.isclose(record["mole_fraction"]["co2"], 422.79 * 1e-6)
    assert "held_past_table" not in record
