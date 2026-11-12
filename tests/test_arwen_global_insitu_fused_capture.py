"""The fused component-capture mark records the chain's vectors.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU).  The same
sequence of marks is recorded twice on the card, once through the fused
kernel and once through the CuPy chain it replaced (``_fusable`` forced
false), at the T21 smoke shape and the T255 shape, whole-grid and in
bands.  The max-abs readings are a maximum, exact in any order, so they
must be the chain's bits; the means are float64 sums in another order and
must agree to a relative 1e-12 (insitu/capture.py docstring).
"""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global.insitu import capture as capture_module
from arwen_global.insitu.capture import CAPTURE_FIELDS, CAPTURE_NAMES, ComponentCapture
from gpuwm.local_gpu import no_local_gpu

MEAN_RTOL = 1.0e-12


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip("GPUWM_NO_LOCAL_GPU is set: no device for the fused mark")
    return cp


def _weights(nlat):
    lat = np.linspace(-1.5, 1.5, nlat)
    weights = np.cos(lat)
    return weights / weights.sum() * 2.0


def _record(cp, shape, bands, dtype, *, fused, seed, monkeypatch):
    rng = np.random.default_rng(seed)
    nlev, nlat, nlon = shape
    capture = ComponentCapture(_weights(nlat))
    capture.begin_step()
    with monkeypatch.context() as patch:
        if not fused:
            patch.setattr(capture_module, "_fusable", lambda xp, fields, previous: False)
        host = {
            key: (rng.standard_normal(shape) * scale + offset).astype(dtype)
            for key, scale, offset in (
                ("theta", 4.0, 280.0), ("qv", 3.0e-3, 5.0e-3),
                ("u", 8.0, 2.0), ("v", 6.0, -1.0))
        }
        steps = [
            {key: (rng.standard_normal(shape) * scale + shift).astype(dtype)
             for key, scale, shift in (
                 ("theta", 0.05, 0.01), ("qv", 1.0e-6, -2.0e-7),
                 ("u", 0.1, 0.03), ("v", 0.1, -0.02))}
            for _ in range(3)
        ]
        edges = np.linspace(0, nlat, bands + 1).astype(int)
        for call in range(2):
            for first, last in zip(edges[:-1], edges[1:]):
                band = None if bands == 1 else (int(first), int(last))
                fields = {
                    key: cp.ascontiguousarray(cp.asarray(host[key][:, first:last]))
                    for key in CAPTURE_FIELDS
                }
                capture.mark("start", fields, cp, band=band)
                for index, step in enumerate(steps):
                    for key in CAPTURE_FIELDS:
                        # In place, as the native kernels update the batch.
                        fields[key] += cp.asarray(step[key][:, first:last]) * (call + 1)
                    capture.mark(f"component{index}", fields, cp, band=band)
            if bands != 1:
                capture.close_call(cp)
    return [(call, name, cp.asnumpy(vector)) for call, name, vector in capture.drain()]


def _compare(fused, chain):
    assert [(c, n) for c, n, _ in fused] == [(c, n) for c, n, _ in chain]
    worst = 0.0
    for (_, name, left), (_, _, right) in zip(fused, chain):
        assert left.dtype == right.dtype == np.float64
        stats_left = dict(zip(CAPTURE_NAMES, left))
        stats_right = dict(zip(CAPTURE_NAMES, right))
        for stat in CAPTURE_NAMES:
            a, b = stats_left[stat], stats_right[stat]
            if stat.endswith("max_abs_change"):
                assert a == b, (name, stat, a, b)
            else:
                assert b != 0.0, (name, stat)
                relative = abs(a - b) / abs(b)
                worst = max(worst, relative)
                assert relative <= MEAN_RTOL, (name, stat, a, b, relative)
    return worst


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bands", [1, 3])
@pytest.mark.parametrize("shape", [(8, 32, 64), (40, 384, 768)], ids=["t21", "t255"])
def test_fused_mark_records_the_chain_vectors(cupy, monkeypatch, shape, bands, dtype):
    if dtype == np.float64 and shape[1] > 100:
        pytest.skip("float64 is checked at the T21 shape; the model runs float32")
    fused = _record(cupy, shape, bands, dtype, fused=True, seed=7, monkeypatch=monkeypatch)
    chain = _record(cupy, shape, bands, dtype, fused=False, seed=7, monkeypatch=monkeypatch)
    assert len(fused) == 6
    worst = _compare(fused, chain)
    print(f"shape={shape} bands={bands} dtype={np.dtype(dtype).name} "
          f"worst mean relative difference {worst:.3e}")


def test_fused_mark_holds_the_current_field_and_propagates_nan(cupy):
    cp = cupy
    capture = ComponentCapture(_weights(6))
    capture.begin_step()
    fields = {key: cp.zeros((3, 6, 10), dtype=cp.float32) for key in CAPTURE_FIELDS}
    capture.mark("start", fields, cp)
    assert capture_module._fusable(cp, fields, capture._previous)
    fields["theta"] += cp.float32(2.0)
    fields["qv"][1, 2, 3] = cp.float32(float("nan"))
    capture.mark("a", fields, cp)
    for key in CAPTURE_FIELDS:
        assert bool(cp.array_equal(capture._previous[key], fields[key], equal_nan=True)), key
        assert capture._previous[key] is not fields[key]
    stats = dict(zip(CAPTURE_NAMES, cp.asnumpy(capture.drain()[0][2])))
    assert stats["theta_max_abs_change"] == 2.0
    assert stats["theta_mean_change"] == pytest.approx(2.0, rel=1.0e-15)
    assert np.isnan(stats["qv_max_abs_change"]) and np.isnan(stats["qv_mean_change"])
    assert stats["u_max_abs_change"] == 0.0 and stats["u_mean_change"] == 0.0


@pytest.mark.parametrize("shape", [(8, 32, 64), (40, 384, 768)], ids=["t21", "t255"])
def test_the_batch_flip_and_conversion_are_the_chain_bits_on_the_card(cupy, shape):
    # The native batch's one-pass level flip and humidity conversion
    # (native_batch._flipped_quotient, global_prognostics) under CuPy's
    # own elementwise divide; the NumPy half runs in
    # test_arwen_global_native_shared_readings.
    from test_arwen_global_native_shared_readings import flip_and_conversion_bits

    flip_and_conversion_bits(cupy, shape)


def test_a_strided_field_takes_the_chain(cupy):
    cp = cupy
    fields = {key: cp.zeros((3, 6, 10), dtype=cp.float32) for key in CAPTURE_FIELDS}
    previous = {key: value.copy() for key, value in fields.items()}
    assert capture_module._fusable(cp, fields, previous)
    fields["u"] = cp.zeros((3, 6, 20), dtype=cp.float32)[:, :, ::2]
    assert not capture_module._fusable(cp, fields, previous)
    assert not capture_module._fusable(np, {k: np.zeros((3, 6, 10)) for k in fields},
                                       {k: np.zeros((3, 6, 10)) for k in fields})
