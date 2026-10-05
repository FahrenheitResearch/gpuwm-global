"""The longwave radiates the air above a 1 hPa model top.

WRF v4.6.1's RRTMG longwave appends nint(p_top / 4 hPa) buffer layers above
the model top, and when that rounds to none it moves the model's top
interface to zero pressure (module_ra_rrtmg_lw.F, "Add zero as top
level"), so the air above the top is always radiated.  The carried driver
transcribed the count but not the zero top, so under the global model's
1 hPa lid the longwave saw no air above the model at all: the top model
layer received no downward longwave from above and cooled to space about
9 K a day too fast (CHANGELOG 0.1.3).  The driver now carries at least one
longwave layer from the model top to the coefficient floor, the shape the
shortwave already used.
"""
from __future__ import annotations

import numpy as np
import pytest

rrtmgp = pytest.importorskip("arwen_global.core.rrtmgp")


def test_a_1_hpa_top_carries_one_longwave_layer_to_the_floor():
    assert rrtmgp.rrtmgp_above_model_layer_counts(100.0) == (1, 1)


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
    profile = rrtmgp._extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=100.0, kind="lw", xp=np)
    assert profile.model_nlay == nlay and profile.upper_nlay == 1
    np.testing.assert_array_equal(profile.plev[:, :nlay + 1], plev.astype(np.float32))
    np.testing.assert_allclose(
        profile.plev[:, -1], rrtmgp.RRTMGP_TOA_PRESSURE_PA, rtol=1.0e-6)
    np.testing.assert_allclose(
        profile.play[:, -1], 0.5 * (100.0 + rrtmgp.RRTMGP_TOA_PRESSURE_PA), rtol=1.0e-6)
    # The layer's temperature is WRF's standard-atmosphere shape shifted to
    # meet the model's top interface: finite, and its lower edge is the
    # model's own top temperature.
    assert np.all(np.isfinite(profile.tlay[:, -1]))
    np.testing.assert_array_equal(profile.tlev[:, nlay], tlev[:, -1].astype(np.float32))
