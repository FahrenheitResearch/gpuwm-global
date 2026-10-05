"""The native radiation radiates the greenhouse gases of the run's year.

THE BREAKAGE THIS PREVENTS.  The native suite handed RRTMGP a fixed 369.55
ppm CO2 override on every run (the air of about 2000, roughly 56 ppm below
2026), and its CH4 and N2O were the RFMIP values of 2014.  These tests hold
that a default run radiates the dated NOAA GML annual means of its valid
year, that the source is named in what a receipt records, and that an
explicit CO2 override still wins for an arm that declares one.
"""
from __future__ import annotations

import inspect
import math

import pytest

from arwen_global.physics.greenhouse_gases import (
    CH4_PPB, CO2_PPM, GREENHOUSE_GAS_SOURCE, N2O_PPB,
    present_day_greenhouse_gases,
)
from arwen_global.physics.native_options import NativePhysicsOptions


def _options(**extra):
    return NativePhysicsOptions.from_mapping({
        "acknowledgement": "device-pending-arwen-native-physics-v1",
        "start_time_utc": "2026-09-01T00:00:00Z",
        **extra,
    })


def test_a_default_run_radiates_the_present_day_gases_of_its_year():
    gases = _options().greenhouse_gas_overrides()
    # 2026's annual mean is not published until the year ends, so a 2026
    # run reads the latest published year, 2025.
    assert math.isclose(gases["co2"], 425.64e-6, rel_tol=1e-12)
    assert math.isclose(gases["ch4"], 1935.94e-9, rel_tol=1e-12)
    assert math.isclose(gases["n2o"], 338.85e-9, rel_tol=1e-12)


def test_the_default_carries_no_fixed_co2():
    assert NativePhysicsOptions.__dataclass_fields__["trace_co2_ppm"].default is None
    assert _options().greenhouse_gas_overrides()["co2"] > 420.0e-6


def test_the_table_is_keyed_on_the_valid_year():
    from datetime import date

    assert math.isclose(
        present_day_greenhouse_gases(date(2019, 7, 1))["co2"], 410.07e-6)
    assert math.isclose(
        present_day_greenhouse_gases(date(2019, 7, 1))["ch4"], 1866.59e-9)
    # Before a table's first year the first entry holds.
    assert math.isclose(
        present_day_greenhouse_gases(date(1990, 1, 1))["n2o"],
        N2O_PPB[min(N2O_PPB)] * 1e-9)
    assert min(CO2_PPM) == 1979 and min(CH4_PPB) == 1984


def test_an_explicit_co2_override_still_wins():
    gases = _options(trace_co2_ppm=369.55).greenhouse_gas_overrides()
    assert math.isclose(gases["co2"], 369.55e-6)
    assert math.isclose(gases["ch4"], 1935.94e-9)
    with pytest.raises(ValueError, match="trace_co2_ppm"):
        _options(trace_co2_ppm=-1.0)


def test_the_record_names_the_source_and_the_years_read():
    record = _options().greenhouse_gas_record()
    assert record["source"] == GREENHOUSE_GAS_SOURCE
    assert "NOAA GML" in record["source"] and "10.15138/9N0H-ZH07" in record["source"]
    assert record["valid_year"] == 2026
    assert record["table_year"] == {"co2": 2025, "ch4": 2025, "n2o": 2025}
    assert record["overridden"] == []


def test_the_runtime_hands_rrtmgp_the_dated_gases():
    from arwen_global.physics import native_runtime

    source = inspect.getsource(native_runtime.NativePhysicsRuntime._radiation_step)
    assert "trace_gas_overrides=self.options.greenhouse_gas_overrides()" in source
    assert "trace_co2_ppm * 1.0e-6" not in source


def test_the_run_receipt_carries_the_record():
    from arwen_global import runner

    # Both the success and the failure receipt: a run that dies must
    # still say what it radiated with, dated by the run's forecast clock
    # (a config that follows its analysis states no start_time_utc).
    import re
    calls = re.findall(
        r'"radiation_greenhouse_gases": _greenhouse_gas_receipt\(\s*'
        r'cfg, getattr\(model, "forecast_clock", None\)\)',
        inspect.getsource(runner))
    assert len(calls) == 2, calls


def test_the_co2_table_is_the_carried_rrtmgps_own():
    """Two dated CO2 tables in one tree drift apart; this one overrides the
    carried RRTMGP's selection, so it must carry the same values."""
    from arwen_global.core.rrtmgp import _NOAA_GML_CO2_ANNUAL_PPM
    from arwen_global.physics.greenhouse_gases import CO2_PPM

    assert dict(CO2_PPM) == dict(_NOAA_GML_CO2_ANNUAL_PPM)
