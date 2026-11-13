"""The real radiation over a year end and a leap day: geometry, ozone, gases.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU).  A
months-long run crosses month ends, 31 December and, in a leap year,
29 February.  What the carried RRTMGP does there is WRF's `radconst`
convention and stays so (design section 8.2): its orbit, eccentricity and
ozone calendar run on a fixed 365-day year with WRF's 0-based julian day,
so in a leap year every date from 1 March on sits one day ahead of the
same date in a common year, and 31 December of a leap year repeats the
geometry and the ozone of the 1 January that follows.  Pinned here on
known answers, with the North Pole's cosine of the zenith angle as the
declination's sine (the hour angle drops out there):

* 31 December 2024 12Z and 1 January 2025 12Z: the same geometry, bit for
  bit (julian 365.5 and 0.5), sin(declination) = -0.39062005;
* 29 February 2024 12Z and 1 March 2023 12Z: the same (julian 59.5),
  -0.13781259;
* a common year end, 31 December 2025 12Z to 1 January 2026 12Z, steps by
  one ordinary day: -0.39194105 to -0.39062005;
* the ozone date weights repeat in the same way.

Then the real native suite on the tiny grid (Noah stood in, see
``_stand_in_land``), restarted part-way through a run that crosses
31 December (and one that crosses 29 February) under
``ghg_year = "valid"``, reproduces the straight run bit for bit, while the
two year rules agree before the crossing and part at it.  The CPU half is
``test_greenhouse_gases_by_year.py``.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from gpuwm.local_gpu import no_local_gpu

UTC = timezone.utc


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip("GPUWM_NO_LOCAL_GPU is set: no device for the radiation")
    return cp


def _utc(*parts):
    return datetime(*parts, tzinfo=UTC)


@pytest.fixture
def polar_driver(cupy):
    from arwen_global.core.rrtmgp import RRTMGPRadiation

    return RRTMGPRadiation(
        start_time=_utc(2024, 11, 1),
        latitude_deg=np.array([[90.0], [-90.0], [0.0]]),
        longitude_deg=np.zeros((3, 1)),
    )


def _mu(driver, instant):
    return driver._cosine_zenith(instant).get()


def test_a_leap_years_31_december_repeats_the_next_1_january(polar_driver):
    leap_end = _mu(polar_driver, _utc(2024, 12, 31, 12))
    new_year = _mu(polar_driver, _utc(2025, 1, 1, 12))
    assert np.array_equal(leap_end, new_year)
    # North Pole: sin(declination); South Pole: its negative.
    assert leap_end[0, 0] == pytest.approx(-0.39062005, abs=2e-7)
    assert leap_end[1, 0] == pytest.approx(0.39062005, abs=2e-7)


def test_a_leap_day_takes_the_geometry_of_a_common_years_1_march(polar_driver):
    assert np.array_equal(_mu(polar_driver, _utc(2024, 2, 29, 12)),
                          _mu(polar_driver, _utc(2023, 3, 1, 12)))
    assert _mu(polar_driver, _utc(2024, 2, 29, 12))[0, 0] == pytest.approx(
        -0.13781259, abs=2e-7)
    # And the next day moves on by one ordinary day.
    assert _mu(polar_driver, _utc(2024, 3, 1, 12))[0, 0] == pytest.approx(
        -0.13135133, abs=2e-7)


def test_a_common_year_end_steps_by_one_ordinary_day(polar_driver):
    assert _mu(polar_driver, _utc(2025, 12, 31, 12))[0, 0] == pytest.approx(
        -0.39194105, abs=2e-7)
    assert _mu(polar_driver, _utc(2026, 1, 1, 12))[0, 0] == pytest.approx(
        -0.39062005, abs=2e-7)


def test_the_ozone_calendar_repeats_the_same_way(polar_driver):
    def date_of(instant):
        polar_driver._set_ozone_date(instant)
        return dict(polar_driver.ozone_date), polar_driver._ozone_date_table.get()

    leap_end, table_a = date_of(_utc(2024, 12, 31, 12))
    new_year, table_b = date_of(_utc(2025, 1, 1, 12))
    assert leap_end["months"] == new_year["months"]
    assert leap_end["weights"] == new_year["weights"]
    assert np.array_equal(table_a, table_b)
    leap_day, table_c = date_of(_utc(2024, 2, 29, 12))
    common, table_d = date_of(_utc(2023, 3, 1, 12))
    assert leap_day["weights"] == common["weights"]
    assert np.array_equal(table_c, table_d)
    # Month ends move continuously: 31 January to 1 February, one day.
    jan, _ = date_of(_utc(2025, 1, 31, 12))
    feb, _ = date_of(_utc(2025, 2, 1, 12))
    assert jan["months"] == feb["months"]
    assert jan["weights"] != feb["weights"]


# ---------------------------------------------------------------------------
# The real native suite, restarted across the crossings.
# ---------------------------------------------------------------------------

def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _on_device(cp, exchange):
    """The exchange as the model hands it on the card: every array field,
    the surface's and the physics state's, as a device array."""
    import dataclasses

    from arwen_global.state import PhysicsState, SurfaceState

    def move(value):
        return cp.asarray(value) if isinstance(value, np.ndarray) else value

    fields = {
        field.name: move(getattr(exchange, field.name))
        for field in dataclasses.fields(exchange)
        if isinstance(getattr(exchange, field.name), np.ndarray)
    }
    surface = SurfaceState(**{
        field.name: move(getattr(exchange.surface, field.name))
        for field in dataclasses.fields(SurfaceState)
    })
    state = exchange.physics_state
    physics_state = PhysicsState(
        schema=state.schema,
        arrays={name: move(value) for name, value in state.arrays.items()},
        metadata=state.metadata,
    )
    return replace(exchange, **fields, surface=surface,
                   physics_state=physics_state)


def _stand_in_land():
    """Noah's no-op stand-in from the CPU harness (the real tables, no
    kernel).  MEASURED 2026-11-12 on the 4090: the real Noah on the smoke
    config's synthetic planet takes the skin from 284.6 K to 368.1 K in its
    first 5 s call, which RRTMGP's admitted 160-355 K surface range then
    refuses at the second call, at 4, 12 and 20 levels alike.  Everything
    else here is the real scheme: RRTMGP, the surface layer, YSU,
    Grell-Freitas and Morrison."""
    from test_arwen_global_level5_native import _fake_modules

    return {"arwen_global.core.noah": _fake_modules()["arwen_global.core.noah"]}


def _run(cupy, start, rule, *, steps=3, restart_after=None):
    from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite
    from arwen_global.state import PhysicsState
    from test_arwen_global_level5_native import _advance, _exchange
    from test_greenhouse_gases_by_year import _through_checkpoint

    options = {
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": start,
        "radiation_interval_s": 10.0,
        "land_surface_interval_s": 10.0,
        "ghg_year": rule,
    }
    exchange = _on_device(cupy, _exchange())

    def fresh():
        return ArwenCudaColumnSuite(options, modules=_stand_in_land())

    suite = fresh()
    result = None
    for call in range(steps):
        current = exchange if result is None else _advance(exchange, result, 10.0 * call)
        if restart_after is not None and call == restart_after:
            suite = fresh()
            on_host = PhysicsState(
                schema=current.physics_state.schema,
                arrays={name: _host(value) for name, value
                        in current.physics_state.arrays.items()},
                metadata=current.physics_state.metadata,
            )
            stored = _through_checkpoint(on_host)
            current = replace(current, physics_state=PhysicsState(
                schema=stored.schema,
                arrays={name: cupy.asarray(value)
                        for name, value in stored.arrays.items()},
                metadata=stored.metadata,
            ))
        result = suite.step(current)
    return result


def _assert_same(straight, resumed):
    from arwen_global.constants import WATER_SPECIES

    for name in ("u", "v", "theta", *WATER_SPECIES):
        assert np.array_equal(_host(getattr(straight, name)),
                              _host(getattr(resumed, name))), name
    resumed_surface = resumed.surface.arrays()
    for name, value in straight.surface.arrays().items():
        assert np.array_equal(_host(value), _host(resumed_surface[name])), name
    assert straight.physics_state.arrays.keys() == resumed.physics_state.arrays.keys()
    for name, value in straight.physics_state.arrays.items():
        assert np.array_equal(_host(value),
                              _host(resumed.physics_state.arrays[name])), name


@pytest.mark.gpu
@pytest.mark.parametrize("restart_after", [1, 2])
def test_the_real_suite_restarts_bit_for_bit_across_31_december(cupy, restart_after):
    start = "2024-12-31T23:59:50Z"
    straight = _run(cupy, start, "valid")
    resumed = _run(cupy, start, "valid", restart_after=restart_after)
    _assert_same(straight, resumed)


@pytest.mark.gpu
@pytest.mark.parametrize("start", ["2024-02-28T23:59:50Z", "2024-02-29T23:59:50Z"])
def test_the_real_suite_restarts_bit_for_bit_across_29_february(cupy, start):
    straight = _run(cupy, start, "valid")
    resumed = _run(cupy, start, "valid", restart_after=1)
    _assert_same(straight, resumed)


@pytest.mark.gpu
def test_the_two_rules_agree_before_the_crossing_and_part_after_it(cupy):
    start = "2024-12-31T23:59:50Z"
    one_valid = _run(cupy, start, "valid", steps=1)
    one_start = _run(cupy, start, "start", steps=1)
    _assert_same(one_start, one_valid)
    valid = _run(cupy, start, "valid")
    held = _run(cupy, start, "start")
    lw_valid = _host(valid.physics_state.arrays["rad_rthratenlw"])
    lw_held = _host(held.physics_state.arrays["rad_rthratenlw"])
    assert not np.array_equal(lw_valid, lw_held)
