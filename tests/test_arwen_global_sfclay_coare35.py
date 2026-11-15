"""The opt-in COARE 3.5 open-water momentum roughness (sfclay isftcflx 3):
the float64 mirror returns z0 = alpha u*^2/g + 0.11 nu/u* with Edson et al.
(2013) alpha = 0.0017 U10N - 0.005, leaves the default arm alone, lowers the
roughness below the WRF Charnock 0.0185 at trade-wind speeds, and the
runtime option is off by default and outside the identity there."""
import numpy as np
import pytest

from arwen_global.core.npref import np_sfclay
from arwen_global.physics.native_options import NativePhysicsOptions


def _ocean(isftcflx, wind=7.0, z0=1.0e-4, ust=0.25):
    one = np.ones((1, 1))
    return np_sfclay(
        wind * one, 0.0 * one, 299.0 * one, 0.017 * one, 99800.0 * one,
        46.0 * one, 101000.0 * one, 301.0 * one, z0 * one, 800.0 * one,
        one, 2.0 * one, option=1, ust=ust * one, dx=52100.0, isftcflx=isftcflx)


def _z0(out):
    for key in ("znt", "znt_out", "z0"):
        if key in out:
            return float(np.asarray(out[key]).ravel()[0])
    raise KeyError(sorted(out))


def test_coare35_roughness_is_the_edson_formula():
    out = _ocean(3, z0=1.0e-4)
    ust = float(np.asarray(out["ust"]).ravel()[0])
    u10n = min(ust / 0.4 * np.log(10.0 / 1.0e-4), 19.0)
    alpha = max(0.0017 * u10n - 0.005, 0.0)
    want = min(alpha * ust * ust / 9.81 + 0.11 * 1.5e-5 / max(ust, 0.01), 2.85e-3)
    assert _z0(out) == pytest.approx(want, rel=2e-3)


def test_coare35_is_smoother_than_wrf_charnock_at_trade_winds():
    assert _z0(_ocean(3)) < _z0(_ocean(0))


def test_default_arm_is_unchanged_and_off_by_default():
    out = _ocean(0)
    ust = float(np.asarray(out["ust"]).ravel()[0])
    assert _z0(out) == pytest.approx(
        min(0.0185 * ust * ust / 9.81 + 0.11 * 1.5e-5 / ust, 2.85e-3), rel=2e-3)
    opts = NativePhysicsOptions(acknowledgement="device-pending-arwen-native-physics-v1")
    assert opts.sfclay_ocean_roughness == "wrf"
