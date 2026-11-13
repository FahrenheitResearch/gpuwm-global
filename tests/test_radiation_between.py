"""The shortwave between radiation calls (physics.radiation_between).

Known answers on the CPU: WRF's swint fit recovers a planted power law,
falls back to the cosine ratio where WRF does, zeroes night columns; the
WRF cosine reads the summer pole's sine of the tilt; and through the real
suite with a fake radiation whose fluxes are planted functions of the
sun, a call between two radiation calls carries the planes at the
current sun under "surface" and "column" and the call's planes under the
default "hold", which adds no array and keeps the identity.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.physics import radiation_between as rb
from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.physics.native_suite import ArwenCudaColumnSuite

from test_arwen_global_level5_native import (  # noqa: E402
    _advance, _exchange, _fake_modules, _options)

F = np.float32
START = datetime(2024, 5, 21, tzinfo=timezone.utc)
INTERVAL_S = 3000.0
SWDOWN_SCALE = 1000.0
SWDNT_SCALE = 1361.0
SWUPT_SCALE = 300.0
SW_HEATING_SCALE = 1.0e-5


def _planes(*values):
    return [np.asarray([v], F) for v in values]


# ---------------------------------------------------------------------------
# Options and identity
# ---------------------------------------------------------------------------

def _native(**extra):
    from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
    return NativePhysicsOptions.from_mapping(
        {"acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT, **extra})


def test_hold_is_the_default_and_stays_out_of_the_identity():
    default = _native()
    assert default.radiation_between == "hold"
    assert "radiation_between" not in default.identity
    assert _native(radiation_between="hold").identity == default.identity


@pytest.mark.parametrize("mode", ["surface", "column"])
def test_an_update_mode_joins_the_identity(mode):
    options = _native(radiation_between=mode)
    assert options.identity["radiation_between"] == mode
    assert options.identity != _native().identity


@pytest.mark.parametrize("bad", ["", "swint", "Hold", 1, None])
def test_an_unknown_mode_is_refused(bad):
    with pytest.raises(ValueError, match="radiation_between"):
        _native(radiation_between=bad)


# ---------------------------------------------------------------------------
# WRF's fit and evaluation
# ---------------------------------------------------------------------------

def _fresh_planes(n=1):
    return {name: np.zeros(n, F) for name in ("swdown_ref", "coszen_ref", "gx", "gg")}


def test_first_call_fit_is_the_cosine_ratio():
    p = _fresh_planes()
    (cz, czl, flux) = _planes(0.5, 0.45, 600.0)
    rb.fit(np, coszen=cz, coszen_now=czl, swdown=flux, **p)
    # WRF's first guess (Gx not yet positive) gives exponent one exactly.
    assert p["gg"][0] == pytest.approx(1.0, abs=1e-6)
    assert p["gx"][0] == pytest.approx(1200.0, rel=1e-6)
    assert p["coszen_ref"][0] == F(0.5) and p["swdown_ref"][0] == F(600.0)
    now = rb.evaluate(np, coszen_now=np.asarray([0.3], F), **p)
    assert now[0] == pytest.approx(360.0, rel=1e-5)


def test_second_call_recovers_a_planted_power_law():
    exponent, amplitude = 1.3, 1100.0
    p = _fresh_planes()
    for cz in (0.5, 0.6):
        flux = np.asarray([amplitude * cz ** exponent], F)
        rb.fit(np, coszen=np.asarray([cz], F), coszen_now=np.asarray([cz], F),
               swdown=flux, **p)
    assert p["gg"][0] == pytest.approx(exponent, abs=2e-4)
    assert p["gx"][0] == pytest.approx(amplitude, rel=5e-4)
    now = rb.evaluate(np, coszen_now=np.asarray([0.7], F), **p)
    assert now[0] == pytest.approx(amplitude * 0.7 ** exponent, rel=5e-4)


def test_a_clamped_exponent_falls_back_to_the_ratio():
    p = _fresh_planes()
    rb.fit(np, coszen=np.asarray([0.5], F), coszen_now=np.asarray([0.5], F),
           swdown=np.asarray([400.0], F), **p)
    # A cloud cleared: the flux tripled while the sun rose ten percent,
    # an exponent past 2.5 that WRF clamps and then evaluates by ratio.
    rb.fit(np, coszen=np.asarray([0.55], F), coszen_now=np.asarray([0.55], F),
           swdown=np.asarray([1200.0], F), **p)
    assert p["gg"][0] == F(2.5)
    now = rb.evaluate(np, coszen_now=np.asarray([0.6], F), **p)
    assert now[0] == pytest.approx(1200.0 * 0.6 / 0.55, rel=1e-6)


def test_night_columns_are_zero_and_carry_no_fit():
    p = _fresh_planes(3)
    cz = np.asarray([0.4, 0.0, 0.4], F)
    now = np.asarray([0.4, 0.3, -0.01], F)
    rb.fit(np, coszen=cz, coszen_now=now, swdown=np.asarray([500.0, 0.0, 500.0], F), **p)
    assert p["gx"][1] == 0.0 and p["gg"][1] == 0.0
    assert p["gx"][2] == 0.0 and p["gg"][2] == 0.0
    out = rb.evaluate(np, coszen_now=now, **p)
    assert out[0] == pytest.approx(500.0, rel=1e-6)
    assert out[1] == 0.0 and out[2] == 0.0


def test_the_summer_pole_sees_the_sine_of_the_tilt():
    when = datetime(2026, 6, 21, 7, 30, tzinfo=timezone.utc)
    lat = np.asarray([[90.0, -90.0]], F)
    lon = np.asarray([[0.0, 0.0]], F)
    mu = rb.cosine_zenith(np, lat, lon, when)
    assert mu.dtype == np.float32
    assert mu[0, 0] == pytest.approx(np.sin(np.deg2rad(23.5)), abs=1e-3)
    assert mu[0, 1] == pytest.approx(-np.sin(np.deg2rad(23.5)), abs=1e-3)


def test_local_noon_on_the_equator_at_the_equinox_is_overhead():
    when = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    # The equation of time moves solar noon about 7 minutes off 12 UTC on
    # the prime meridian; a quarter-degree of longitude is one minute.
    mu = rb.cosine_zenith(np, np.zeros((1, 2), F),
                          np.asarray([[0.0, 180.0]], F), when)
    assert mu[0, 0] > 0.99
    assert mu[0, 1] < -0.99


# ---------------------------------------------------------------------------
# Through the suite
# ---------------------------------------------------------------------------

def _sun_modules():
    """The level-5 fakes with a radiation whose shortwave is a planted
    function of the call's own (interval-midpoint) sun: SWDOWN, the top
    planes and the heating all proportional to the cosine."""
    modules = _fake_modules()

    class Radiation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, *, atmosphere, fields, state, cfg):
            shape = atmosphere["temperature"].shape
            when = (self.kwargs["start_time"]
                    + timedelta(seconds=state.elapsed_seconds + 0.5 * INTERVAL_S))
            mu = rb.cosine_zenith(np, self.kwargs["latitude_deg"],
                                  self.kwargs["longitude_deg"], when)
            lit = np.maximum(mu, F(0.0))
            return SimpleNamespace(
                rthratenlw=np.zeros(shape, F),
                rthratensw=np.broadcast_to(F(SW_HEATING_SCALE) * lit, shape).astype(F),
                swdown=(F(SWDOWN_SCALE) * lit).astype(F),
                glw=np.full(shape[1:], 300.0, F),
                olr=np.full(shape[1:], 236.0, F),
                gsw=(F(SWDOWN_SCALE) * lit * F(0.8)).astype(F),
                coszen=mu.astype(F),
                swdnt=(F(SWDNT_SCALE) * lit).astype(F),
                swupt=(F(SWUPT_SCALE) * lit).astype(F),
            )

    modules["arwen_global.core.rrtmgp"] = SimpleNamespace(RRTMGPRadiation=Radiation)
    return modules


def _run(mode, times=(0.0, 600.0)):
    options = {**_options(), "radiation_interval_s": INTERVAL_S,
               "radiation_between": mode}
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(options, array_module=np, modules=_sun_modules())
    results = [suite.step(exchange)]
    for t in times[1:]:
        results.append(suite.step(_advance(exchange, results[-1], t)))
    return exchange, results


def _sun_now(exchange, time_s):
    from arwen_global.physics.native_batch import NativeColumnBatch

    batch = NativeColumnBatch.from_exchange(exchange, np)
    when = START + timedelta(seconds=time_s)
    return rb.cosine_zenith(np, batch.arrays["latitude_deg"],
                            batch.arrays["longitude_deg"], when)


def test_hold_adds_no_array_and_keeps_the_call_planes():
    exchange, (first, second) = _run("hold")
    arrays = second.physics_state.arrays
    assert not [name for name in arrays if name.startswith("radbetween_")]
    np.testing.assert_array_equal(arrays["swdown"], first.physics_state.arrays["swdown"])
    assert second.physics_state.metadata["radiation_calls"] == 1


@pytest.mark.parametrize("mode", ["surface", "column"])
def test_the_surface_follows_the_sun_between_calls(mode):
    exchange, (first, second) = _run(mode)
    arrays = second.physics_state.arrays
    assert second.physics_state.metadata["radiation_calls"] == 1
    mu_now = _sun_now(exchange, 600.0)
    mu_call = arrays["coszen"]
    lit = (mu_now > rb.COSZEN_MIN) & (mu_call > rb.COSZEN_MIN)
    assert lit.any() and (~lit).any()
    # One call so far: WRF's first-guess fit, exponent one, so the flux is
    # the planted 1000 * cosine at the current sun on every lit column.
    expected = np.where(lit, F(SWDOWN_SCALE) * mu_now, F(0.0))
    np.testing.assert_allclose(arrays["swdown"], expected, rtol=2e-6, atol=1e-3)
    albedo = np.asarray(second.surface.albedo, F)
    np.testing.assert_allclose(arrays["gsw"], expected * (F(1.0) - albedo),
                               rtol=2e-6, atol=1e-3)
    # The radiation integrals advanced by what was applied: the first
    # call's sun for 5 s plus the current sun for 5 s.
    mu_first = _sun_now(exchange, 0.0)
    lit_first = (mu_first > rb.COSZEN_MIN) & (mu_call > rb.COSZEN_MIN)
    applied_first = np.where(lit_first, F(SWDOWN_SCALE) * mu_first, F(0.0))
    np.testing.assert_allclose(
        second.physics_state.arrays["acc_sw_down_surface_j_m2"],
        5.0 * (applied_first + expected), rtol=1e-5, atol=1e-2)
    if mode == "surface":
        np.testing.assert_array_equal(
            arrays["rad_rthratensw"], first.physics_state.arrays["rad_rthratensw"])
        np.testing.assert_array_equal(arrays["swdnt"], first.physics_state.arrays["swdnt"])
        assert "radbetween_rthratensw_ref" not in arrays
    else:
        ratio = np.where(lit, mu_now / np.where(lit, mu_call, F(1.0)), F(0.0))
        np.testing.assert_allclose(
            arrays["rad_rthratensw"],
            np.broadcast_to(F(SW_HEATING_SCALE) * np.maximum(mu_call, 0) * ratio,
                            arrays["rad_rthratensw"].shape),
            rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(arrays["swdnt"], F(SWDNT_SCALE) * np.maximum(mu_call, 0) * ratio,
                                   rtol=2e-6, atol=1e-3)


def test_a_restarted_suite_continues_bit_for_bit():
    times = (0.0, 600.0, 1200.0)
    exchange, straight = _run("column", times)
    # A fresh suite (no radiation driver, nothing cached) picks up the
    # namespace the second call left, as a restart from its checkpoint does.
    options = {**_options(), "radiation_interval_s": INTERVAL_S,
               "radiation_between": "column"}
    fresh = ArwenCudaColumnSuite(options, array_module=np, modules=_sun_modules())
    resumed = fresh.step(_advance(exchange, straight[1], times[2]))
    for name, value in straight[2].physics_state.arrays.items():
        np.testing.assert_array_equal(
            np.asarray(resumed.physics_state.arrays[name]), np.asarray(value), err_msg=name)
    np.testing.assert_array_equal(resumed.theta, straight[2].theta)
