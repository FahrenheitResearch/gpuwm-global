"""The native stratospheric floor's bill rides on every call and the receipt.

THE BREAKAGE THIS PREVENTS.  The floor adds enthalpy to air above 50 hPa
colder than its floor temperature.  Its per-call heating was reduced as a
flat mean over the Gaussian grid's rows (a polar column, where the floor
acts, weighed as much as an equatorial one), how many points it warmed was
never reported, and no receipt carried either: a run's receipt said the
floor was configured and not what it did.  These tests hold an area-mean
heating, a warmed-point count on every call, and a run ledger in the
receipt and the diagnostics records.
"""
from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from arwen_global.physics.native_suite import ArwenCudaColumnSuite
from arwen_global.runner import StratosphericFloorLedger


def _chilled(floor_k):
    from test_arwen_global_level5_native import _exchange

    exchange = _exchange()
    p_full = np.asarray(exchange.p_full, dtype=np.float64)
    exner = np.asarray(exchange.exner, dtype=np.float64)
    mask = p_full < 5000.0
    assert mask.any()
    temperature = np.asarray(exchange.theta, dtype=np.float64) * exner
    temperature[mask] = floor_k - 45.0
    return replace(exchange, theta=(temperature / exner)), mask


def _suite(**extra):
    from test_arwen_global_level5_native import _fake_modules, _options

    return ArwenCudaColumnSuite(
        {**_options(), **extra}, array_module=np, modules=_fake_modules())


def test_every_call_reports_the_points_it_warmed_and_an_area_mean():
    chilled, mask = _chilled(195.0)
    floored = _suite(stratospheric_floor_k=195.0, stratospheric_floor_pa=5000.0).step(chilled)
    unfloored = _suite(stratospheric_floor_k=0.0).step(chilled)
    assert floored.diagnostics["stratospheric_floor_points"] == float(mask.sum())
    assert unfloored.diagnostics["stratospheric_floor_points"] == 0.0
    exner = np.asarray(chilled.exner, dtype=np.float64)
    added = (
        np.asarray(floored.theta, np.float64) - np.asarray(unfloored.theta, np.float64)
    ) * exner
    from arwen_global.constants import DRY_AIR_CP, GRAVITY_M_S2

    column = (added * np.asarray(chilled.dp, np.float64)).sum(axis=0) * DRY_AIR_CP / GRAVITY_M_S2
    weight = np.cos(np.deg2rad(np.asarray(chilled.latitude_deg, np.float64)))
    expected = float((column * weight).sum() / weight.sum())
    assert floored.diagnostics["mean_stratospheric_floor_heating_j_m2"] == pytest.approx(
        expected, rel=2e-3)


def test_the_run_ledger_sums_the_bill():
    ledger = StratosphericFloorLedger()
    half = {"physics": {"mean_stratospheric_floor_heating_j_m2": 2.5,
                        "stratospheric_floor_points": 7.0}}
    idle = {"physics": {"mean_stratospheric_floor_heating_j_m2": 0.0,
                        "stratospheric_floor_points": 0.0}}
    ledger.add({"first_half_physics": half, "second_half_physics": idle})
    ledger.add({"first_half_physics": half, "second_half_physics": half})
    assert ledger.record() == {
        "physics_calls": 4, "calls_that_warmed": 3,
        "area_mean_heating_j_m2": 7.5, "maximum_points_warmed_in_a_call": 7.0,
    }


def test_the_receipt_and_the_diagnostics_carry_the_ledger():
    from arwen_global import runner

    source = inspect.getsource(runner)
    assert source.count('"stratospheric_floor_ledger": floor_ledger.record()') == 2
    assert 'diag["stratospheric_floor_ledger"] = floor_ledger.record()' in source


def test_the_floor_arm_leaves_analysed_vortex_air_alone():
    """The reach of an arm that turns the floor on is the model top.

    MEASURED 2026-10-05: the GDAS 2026-09-01 00Z analysis puts 4.6 percent
    of the points above 50 hPa below 195 K (south cap mean 192.1 K at 30 to
    50 hPa, minimum 181.3 K), and the 50 hPa reach warmed that analysed air
    toward 195 K on a 30-minute timescale.  The top sag the floor exists
    for lives above 5 hPa.  Here a 9.8 hPa level held at 150 K is analysed
    air the default must not touch, and the same level moved above 5 hPa
    is the sag it must still catch.
    """
    chilled, mask = _chilled(195.0)
    p_full = np.asarray(chilled.p_full, dtype=np.float64)
    assert float(p_full[mask].max()) > 500.0
    default = _suite(stratospheric_floor_k=195.0).step(chilled)
    unfloored = _suite(stratospheric_floor_k=0.0).step(chilled)
    np.testing.assert_array_equal(
        np.asarray(default.theta), np.asarray(unfloored.theta))
    assert default.diagnostics["stratospheric_floor_points"] == 0.0

    lifted = replace(chilled, p_full=np.where(mask, 400.0, p_full))
    caught = _suite(stratospheric_floor_k=195.0).step(lifted)
    assert caught.diagnostics["stratospheric_floor_points"] == float(mask.sum())
    assert caught.diagnostics["mean_stratospheric_floor_heating_j_m2"] > 0.0


def test_the_floor_defaults_name_the_top_reach():
    from arwen_global.physics.native_options import NativePhysicsOptions

    fields = NativePhysicsOptions.__dataclass_fields__
    assert fields["stratospheric_floor_pa"].default == 500.0


def test_the_default_suite_runs_no_floor():
    """The floor is retired from the default (2026-10-05).

    THE BREAKAGE THIS PREVENTS.  A default-on pull of the model top toward
    195 K is a heat source no breakage justifies any more: the hour-83.4
    top-sag death it was installed for does not reproduce in a 240 h
    floor-off run of the T255 GDAS 2026-09-01 00Z case on this tree (top
    level at 189 K at 240 h, every gate green; the measurement is on
    NativePhysicsOptions).  Here a top held 45 K under the floor's old
    temperature, above its 5 hPa reach, is left exactly as the unfloored
    arm leaves it, and the ledger books nothing.
    """
    from arwen_global.physics.native_options import NativePhysicsOptions

    assert NativePhysicsOptions.__dataclass_fields__[
        "stratospheric_floor_k"].default <= 0.0
    chilled, mask = _chilled(195.0)
    p_full = np.asarray(chilled.p_full, dtype=np.float64)
    lifted = replace(chilled, p_full=np.where(mask, 400.0, p_full))
    default = _suite().step(lifted)
    unfloored = _suite(stratospheric_floor_k=0.0).step(lifted)
    np.testing.assert_array_equal(
        np.asarray(default.theta), np.asarray(unfloored.theta))
    assert default.diagnostics["stratospheric_floor_points"] == 0.0
    assert default.diagnostics["mean_stratospheric_floor_heating_j_m2"] == 0.0
