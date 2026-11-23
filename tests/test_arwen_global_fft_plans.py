"""Kept cuFFT plans (arwen_global.spectral.fft_plans): same bits, no
resident work area, and nothing kept off a cupy run."""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global import device_memory
from arwen_global.spectral import fft_plans


def test_a_numpy_array_gets_a_no_op_context():
    restore = device_memory.keep_fft_plans("numpy")
    try:
        assert fft_plans.active() is None
        with fft_plans.kept(np.zeros((2, 8), dtype=np.float32), 8):
            pass
    finally:
        restore()


def test_the_switch_turns_kept_plans_off():
    restore = device_memory.keep_fft_plans(
        "cupy", environ={device_memory.FFT_PLAN_CACHE_ENV: "0"}
    ) if _cupy() else (lambda: None)
    try:
        assert fft_plans.active() is None
    finally:
        restore()


def _cupy():
    try:
        import cupy as cp

        return cp if cp.cuda.runtime.getDeviceCount() > 0 else None
    except Exception:  # noqa: BLE001 - no device, no cupy test
        return None


@pytest.mark.skipif(_cupy() is None, reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kept_plans_return_the_replanned_bits_and_hold_no_work_area(dtype):
    cp = _cupy()
    from cupyx.scipy import fft as device_fft

    rng = np.random.default_rng(7)
    shapes = [(3, 40, 96, 1152), (40, 17, 1152), (6, 40, 64, 3072), (5, 2400)]
    fields = [cp.asarray(rng.standard_normal(s).astype(dtype)) for s in shapes]

    def transforms():
        out = []
        for f in fields:
            n = f.shape[-1]
            with fft_plans.kept(f, n):
                spec = cp.fft.rfft(f, axis=-1)
            work = spec.copy()
            with fft_plans.kept(work, n, inverse=True):
                back = device_fft.irfft(work, n=n, axis=-1, overwrite_x=True)
            out.append((cp.asnumpy(spec), cp.asnumpy(back)))
        return out

    restore = device_memory.disable_fft_plan_cache("cupy")
    try:
        fresh = transforms()
    finally:
        restore()
    pool = cp.get_default_memory_pool()
    restore = device_memory.keep_fft_plans("cupy")
    try:
        cache = fft_plans.active()
        assert cache is not None
        first = transforms()
        cp.cuda.Device().synchronize()
        held = pool.used_bytes()
        second = transforms()
        third = transforms()
        cp.cuda.Device().synchronize()
        receipt = cache.receipt()
        # every shape built once (forward and inverse), then served kept
        assert receipt["plans_built"] == 2 * len(shapes)
        assert receipt["hits"] == 2 * 2 * len(shapes)
        assert receipt["largest_work_bytes"] > 0
        # nothing a kept plan owns stays in the pool between calls
        assert pool.used_bytes() == held
    finally:
        restore()
    assert fft_plans.active() is None
    for a, b, c, d in zip(fresh, first, second, third):
        for i in range(2):
            assert a[i].tobytes() == b[i].tobytes() == c[i].tobytes() \
                == d[i].tobytes()
