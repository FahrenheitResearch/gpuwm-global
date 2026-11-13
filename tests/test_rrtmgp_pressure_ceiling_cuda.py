"""The pressure-ceiling rows that need a card (see test_rrtmgp_pressure_ceiling.py).

* a layer inside the tables reads, bit for bit, what the untouched inline
  oracle kernel reads, so no run that stays inside the tables moves;
* a layer between the tables' deepest pressure and 1,150 hPa reads, on the
  device, exactly the coordinates the deepest pressure reads, and only those
  cells moved against the oracle;
* the device gas optics take the year run's stop and match the float64 mirror.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

from arwen_global.core import npref, rrtmgp  # noqa: E402
from test_rrtmgp_pressure_ceiling import _profile, _tables  # noqa: E402


def _prepass(kernel_name, tables, play, tlay):
    from arwen_global.core.kernels import get_kernel

    play = cp.ascontiguousarray(cp.asarray(play, dtype=np.float32))
    tlay = cp.ascontiguousarray(cp.asarray(tlay, dtype=np.float32))
    d = tables.to_device()
    integer = tuple(cp.empty(play.shape, dtype=cp.int32) for _ in range(3))
    fraction = tuple(cp.empty(play.shape, dtype=cp.float32) for _ in range(2))
    n = play.size
    get_kernel("rrtmgp_gas", kernel_name)(
        ((n + 255) // 256,), (256,), (
            play, tlay, d.press_ref, d.temp_ref,
            np.float32(tables.press_ref_trop), *integer, *fraction,
            np.int32(n), np.int32(tables.ntemp), np.int32(tables.npres)))
    return [cp.asnumpy(a) for a in (*integer, *fraction)]


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_an_in_table_layer_reads_what_the_untouched_oracle_reads(kind):
    tables = _tables(kind)
    rng = np.random.default_rng(20261113)
    lo, hi = np.log(float(np.min(tables.press_ref))), np.log(
        float(np.max(tables.press_ref)))
    play = np.exp(rng.uniform(lo, hi, (257, 40))).astype(np.float32)
    play[0, :3] = np.float32(np.max(tables.press_ref))  # the edge itself
    play[1, :3] = np.float32(np.min(tables.press_ref))
    tlay = rng.uniform(float(np.min(tables.temp_ref)),
                       float(np.max(tables.temp_ref)),
                       play.shape).astype(np.float32)
    fixed = _prepass("rrtmgp_interpolation_prepass", tables, play, tlay)
    oracle = _prepass("rrtmgp_interpolation_inline_reference", tables,
                      play, tlay)
    for a, b in zip(fixed, oracle):
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_a_deep_layer_reads_the_deepest_pressure_on_the_device(kind):
    tables = _tables(kind)
    play, plev, tlay, vmr = _profile(tables)
    top = np.float32(np.max(tables.press_ref))
    clipped = play.astype(np.float32)
    clipped[:, 0] = np.minimum(clipped[:, 0], top)
    deep = _prepass("rrtmgp_interpolation_prepass", tables, play, tlay)
    at_top = _prepass("rrtmgp_interpolation_prepass", tables, clipped, tlay)
    for a, b in zip(deep, at_top):
        assert a.tobytes() == b.tobytes()
    # The old statements extrapolate: the fix moved exactly these cells.
    oracle = _prepass("rrtmgp_interpolation_inline_reference", tables,
                      play, tlay)
    fpress_fixed, fpress_old = deep[4], oracle[4]
    moved = fpress_fixed != fpress_old
    assert moved[:, 0].tolist() == [True, False, True, False]
    assert not moved[:, 1:].any()


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_the_device_gas_optics_take_the_stop_and_match_the_mirror(kind):

    tables = _tables(kind)
    play, plev, tlay, vmr = _profile(tables)
    device = rrtmgp.gas_optics(tables, cp.asarray(play, dtype=np.float32),
                               cp.asarray(plev, dtype=np.float32),
                               cp.asarray(tlay, dtype=np.float32),
                               cp.asarray(vmr, dtype=np.float32))
    mirror = npref.np_rrtmgp_gas_optics(tables, play, plev, tlay, vmr)
    tau = cp.asnumpy(device.tau)
    assert np.all(np.isfinite(tau))
    np.testing.assert_allclose(tau, mirror.tau, rtol=2e-3,
                               atol=1e-6 * float(np.max(mirror.tau)))
    bad = play.copy()
    bad[0, 0] = 115_001.0
    with pytest.raises(ValueError, match="outside allowed range"):
        rrtmgp.gas_optics(tables, cp.asarray(bad, dtype=np.float32),
                          cp.asarray(plev + 2_000.0, dtype=np.float32),
                          cp.asarray(tlay, dtype=np.float32),
                          cp.asarray(vmr, dtype=np.float32))
