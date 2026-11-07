"""The longwave radiates the air above a 1 hPa model top.

WRF v4.6.1's RRTMG longwave appends nint(p_top / 4 hPa) buffer layers above
the model top, and when that rounds to none it moves the model's top
interface to zero pressure (module_ra_rrtmg_lw.F, "Add zero as top
level"), so the air above the top is always radiated.  The carried driver
transcribed the count but not the zero top, so under the global model's
1 hPa lid the longwave saw no air above the model at all: the top model
layer received no downward longwave from above and cooled to space about
9 K a day too fast (CHANGELOG 0.1.3).  The driver then carried one
longwave layer from the model top to the coefficient floor, the shape the
shortwave already used.

That one layer spanned 1 hPa to 0.01 hPa (4.6 in ln p) at a single
temperature, and on the 100 RFMIP sites it still cooled the 1.0-1.44 hPa
layer 1.16 K/day of T (about 8 K/day of theta) faster than the sites' own
atmosphere above 1 hPa resolved on 24 layers (2026-11-06,
tools/arwen_global_lid_columns.py).  A top below 2 hPa now carries
log-spaced cap layers no thicker than ABOVE_MODEL_MAX_DLNP for both solvers;
a top of 2 hPa or more keeps WRF's buffer layers exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

rrtmgp = pytest.importorskip("arwen_global.core.rrtmgp")


def test_a_1_hpa_top_carries_a_log_spaced_cap_to_the_floor():
    floor = rrtmgp.RRTMGP_TOA_PRESSURE_PA
    count = int(np.ceil(np.log(100.0 / floor) / rrtmgp.ABOVE_MODEL_MAX_DLNP))
    assert count == 14
    assert rrtmgp.rrtmgp_above_model_layer_counts(100.0) == (count, count)
    assert rrtmgp.above_model_log_spaced(100.0)
    # 2 hPa is WRF's first top with a buffer layer: WRF's rule from there.
    assert not rrtmgp.above_model_log_spaced(200.0)
    assert rrtmgp.rrtmgp_above_model_layer_counts(200.0) == (1, 1)


def test_a_regional_top_keeps_wrfs_buffer_layers():
    # 50 hPa: thirteen (Fortran NINT(12.5)) 4 hPa buffer layers, one shortwave layer.
    assert rrtmgp.rrtmgp_above_model_layer_counts(5000.0) == (13, 1)
    assert rrtmgp.rrtmgp_above_model_layer_counts(1000.0) == (3, 1)
    assert rrtmgp.rrtmgp_above_model_layer_counts(400.0) == (1, 1)


def test_a_top_at_the_coefficient_floor_needs_no_layer():
    floor = rrtmgp.RRTMGP_TOA_PRESSURE_PA
    assert rrtmgp.rrtmgp_above_model_layer_counts(floor) == (0, 0)
    assert rrtmgp.rrtmgp_above_model_layer_counts(1.5 * floor) == (0, 0)


def test_the_longwave_layer_spans_the_model_top_to_the_floor():
    ncol, nlay = 3, 4
    plev = np.tile(np.array([100000.0, 50000.0, 1000.0, 300.0, 100.0]), (ncol, 1))
    play = 0.5 * (plev[:, 1:] + plev[:, :-1])
    tlev = np.tile(np.array([290.0, 250.0, 230.0, 250.0, 265.0]), (ncol, 1))
    tlay = 0.5 * (tlev[:, 1:] + tlev[:, :-1])
    qv = np.full((ncol, nlay), 4.0e-6)
    floor = rrtmgp.RRTMGP_TOA_PRESSURE_PA
    for kind in ("lw", "sw"):
        profile = rrtmgp._extend_above_model_profile(
            play, plev, tlay, tlev, qv, p_top=100.0, kind=kind, xp=np)
        count = profile.upper_nlay
        assert profile.model_nlay == nlay and count == 14
        np.testing.assert_array_equal(
            profile.plev[:, :nlay + 1], plev.astype(np.float32))
        np.testing.assert_allclose(profile.plev[:, -1], floor, rtol=1.0e-6)
        steps = np.diff(np.log(profile.plev[:, nlay:].astype(np.float64)), axis=1)
        assert np.all(steps < 0.0)
        assert np.all(-steps <= rrtmgp.ABOVE_MODEL_MAX_DLNP + 1.0e-5)
        # The cap follows the standard-atmosphere shape down from the
        # model's own top interface temperature, inside the tables.
        # (WRF's table peaks at 0.76 hPa, so the first cap interfaces sit a
        # kelvin above the top; the floor is the mesosphere, far colder.)
        np.testing.assert_array_equal(
            profile.tlev[:, nlay], tlev[:, -1].astype(np.float32))
        assert np.all(profile.tlev[:, -1] < tlev[:, -1].astype(np.float32) - 50.0)
        assert np.all(profile.tlev[:, nlay:] >= 160.0)
        # The uniform-top cache and the per-column ladder agree bitwise.
        cached = rrtmgp._extend_above_model_profile(
            play, plev, tlay, tlev, qv, p_top=100.0, kind=kind, xp=np,
            uniform_top=True)
        np.testing.assert_array_equal(cached.plev, profile.plev)
        np.testing.assert_array_equal(cached.tlev, profile.tlev)
        np.testing.assert_array_equal(cached.play, profile.play)
    profile = rrtmgp._extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=100.0, kind="lw", xp=np)
    # The layer's temperature is WRF's standard-atmosphere shape shifted to
    # meet the model's top interface: finite, and its lower edge is the
    # model's own top temperature.
    assert np.all(np.isfinite(profile.tlay[:, -1]))
    np.testing.assert_array_equal(profile.tlev[:, nlay], tlev[:, -1].astype(np.float32))
