"""The column filler's neighbour sweep in one launch is the loop's bits.

``core.column_borrow`` against the level loop it replaces, run through the
same entry point with the kernel refused: float32 and float64, masses with
negative lobes at the top, the bottom and inside, signed zeros and NaNs,
and the moved plane.

Every number, signed zero and infinity is compared bit for bit; a NaN is
compared by place, not by payload.  With two NaN operands the card keeps
one by operand position and the kernel's compiler and CuPy's order it
differently: on an RTX 4090 (sm 8.9) the float64 NaN layer differed in
payload only (0x7fffffffe0000000 against 0x7ff8000000000000), and a bare
``a - c`` raw kernel differs from CuPy's subtract the same way.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import column_borrow  # noqa: E402
from arwen_global.dynamics import MoistHybridModel  # noqa: E402


def _case(dtype, nlev=40, shape=(23, 61), seed=11):
    rng = np.random.default_rng(seed)
    host = (rng.standard_normal((nlev, *shape)) * 1e-3 + 2e-4).astype(dtype)
    host[0, ::3, ::2] = -5e-4
    host[-1, 1::4] = -2e-3
    host[7:9, 5] = -1e-2
    host[10, 0, :7] = -0.0
    host[11, 0, :7] = 0.0
    host[12, 2, 3] = np.nan
    host[3, 4, 4] = np.inf
    return host


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_one_launch_sweep_is_the_level_loop(dtype, monkeypatch):
    host = _case(dtype)
    fused = cp.asarray(host)
    assert column_borrow.fusable(cp, fused)
    moved_fused = MoistHybridModel._borrow_from_neighbours(cp, fused)
    loop = cp.asarray(host)
    monkeypatch.setattr(column_borrow, "fusable", lambda *a, **k: False)
    moved_loop = MoistHybridModel._borrow_from_neighbours(cp, loop)
    _same_bits_nan_by_place(cp.asnumpy(fused), cp.asnumpy(loop))
    _same_bits_nan_by_place(cp.asnumpy(moved_fused), cp.asnumpy(moved_loop))


def _same_bits_nan_by_place(a, b):
    word = np.uint32 if a.dtype == np.float32 else np.uint64
    nan = np.isnan(a)
    assert np.array_equal(nan, np.isnan(b))
    assert np.array_equal(a[~nan].view(word), b[~nan].view(word))


def test_noncontiguous_mass_takes_the_loop():
    mass = cp.zeros((4, 6, 8), dtype=cp.float32)[:, :, ::2]
    assert not column_borrow.fusable(cp, mass)
