"""cal_cldfra1 in one launch is the operation-by-operation chain's bits.

The fused kernel (rrtmgp._cldfra1_kernel) against the chain it replaces
(rrtmgp._CLDFRA1_FUSED = False), word for word, on all three moisture arms
(qc+qi+qs, the P3 qc+qi arm, qc alone), over saturated, subsaturated, cloud
free, below-qcldmin, cold and warm cells, with zeros, a negative condensate
and NaNs planted.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import rrtmgp  # noqa: E402


def _profiles(seed=5, ncol=3001, nlay=41):
    rng = np.random.default_rng(seed)
    shape = (ncol, nlay)
    play = rng.uniform(100.0, 105000.0, shape).astype(np.float32)
    tlay = rng.uniform(150.0, 320.0, shape).astype(np.float32)
    qv = (rng.uniform(0.0, 0.025, shape) * rng.random(shape) ** 3).astype(np.float32)
    qc = np.where(rng.random(shape) < 0.4, rng.uniform(0, 2e-3, shape), 0).astype(np.float32)
    qi = np.where(rng.random(shape) < 0.3, rng.uniform(0, 1e-3, shape), 0).astype(np.float32)
    qs = np.where(rng.random(shape) < 0.2, rng.uniform(0, 1e-3, shape), 0).astype(np.float32)
    qc[0, :5] = 1e-13
    qi[1, :5] = -1e-9
    qs[2, 3] = -0.0
    qv[3, :] = 0.5        # supersaturated column
    tlay[4, 2] = np.float32(273.15)
    qv[5, 7] = np.nan
    play[6, 1] = np.nan
    return [cp.asarray(a) for a in (qv, qc, qi, qs, tlay, play)]


@pytest.mark.parametrize("flags", [
    dict(f_qc=True, f_qi=True, f_qs=True),
    dict(f_qc=True, f_qi=True, f_qs=False),
    dict(f_qc=True, f_qi=False, f_qs=False),
])
def test_fused_cldfra1_is_the_chain_word_for_word(flags, monkeypatch):
    args = _profiles()
    fused = rrtmgp.cal_cldfra1(*args, **flags)
    monkeypatch.setattr(rrtmgp, "_CLDFRA1_FUSED", False)
    chain = rrtmgp.cal_cldfra1(*args, **flags)
    assert fused.dtype == chain.dtype == np.float32
    assert fused.shape == chain.shape and fused.flags.c_contiguous
    a = cp.asnumpy(fused).view(np.uint32)
    b = cp.asnumpy(chain).view(np.uint32)
    assert np.array_equal(a, b), int(np.count_nonzero(a != b))
    # The case is not trivial: clear, partial and overcast cells all occur.
    values = cp.asnumpy(chain)
    assert (values == 0).any() and (values == 1).any()
    assert ((values > 0) & (values < 1)).any()


def _flat_words(value, prefix=""):
    """Every array (and scalar) a HydrometeorPaths carries, by name, as words."""
    import dataclasses

    out = {}
    if value is None:
        return out
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            out.update(_flat_words(getattr(value, f.name), f"{prefix}{f.name}."))
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flat_words(v, f"{prefix}{k}."))
        return out
    if isinstance(value, cp.ndarray):
        host = cp.asnumpy(value)
        host = np.ascontiguousarray(host).reshape(-1)
        out[prefix] = (str(host.dtype), host.view(np.uint8) if host.dtype != np.bool_ else host)
        return out
    out[prefix] = repr(value)
    return out


@pytest.mark.parametrize("effective", [False, True])
@pytest.mark.parametrize("path_sums", [False, True])
@pytest.mark.parametrize("fused_flags", [("_MORRISON_PATHS_FUSED", "_BOUND_SIZES_FUSED"), ("_MORRISON_PATHS_FUSED",), ("_BOUND_SIZES_FUSED",)])
def test_fused_morrison_paths_are_the_chain_word_for_word(effective, path_sums, fused_flags, monkeypatch):
    rng = np.random.default_rng(9)
    qv, qc, qi, qs, tlay, play = _profiles()
    for a in (qc, qi, qs, tlay, play):
        a[cp.isnan(a)] = np.float32(0.0)
    qc = cp.maximum(qc, np.float32(0.0))
    qi = cp.maximum(qi, np.float32(0.0))
    qs = cp.maximum(qs, np.float32(0.0))
    ncol, nlay = qc.shape
    plev = cp.asarray(np.sort(rng.uniform(100.0, 105000.0, (ncol, nlay + 1)),
                              axis=1)[:, ::-1].astype(np.float32).copy())
    shape = qc.shape

    def number(scale, zero_share):
        n = rng.uniform(0, scale, shape) * (rng.random(shape) > zero_share)
        return cp.asarray(n.astype(np.float32))

    kw = dict(microphysics="morrison", play=play, tlay=tlay,
              nc=number(3e8, 0.3), nr=number(1e4, 0.5), ni=number(1e6, 0.3),
              ns=number(1e5, 0.3), cldfra=rrtmgp.cal_cldfra1(qv, qc, qi, qs, tlay, play),
              validate=False, path_sums=path_sums)
    if effective:
        def radius(lo, hi):
            r = rng.uniform(lo, hi, shape).astype(np.float32)
            r[rng.random(shape) < 0.2] = np.float32(rrtmgp.MORRISON_NO_MASS_RADIUS_UM)
            return cp.asarray(r)
        kw.update(effc=radius(2.0, 30.0), effr=radius(50.0, 500.0),
                  effi=radius(5.0, 150.0), effs=radius(20.0, 1500.0))
    for flag in ("_MORRISON_PATHS_FUSED", "_BOUND_SIZES_FUSED"):
        monkeypatch.setattr(rrtmgp, flag, flag in fused_flags)
    fused = _flat_words(rrtmgp.hydrometeor_paths(plev, qc, qs=qs, qi=qi, **kw))
    monkeypatch.setattr(rrtmgp, "_MORRISON_PATHS_FUSED", False)
    monkeypatch.setattr(rrtmgp, "_BOUND_SIZES_FUSED", False)
    chain = _flat_words(rrtmgp.hydrometeor_paths(plev, qc, qs=qs, qi=qi, **kw))
    assert fused.keys() == chain.keys()
    for name in chain:
        if isinstance(chain[name], tuple):
            assert fused[name][0] == chain[name][0], name
            assert np.array_equal(fused[name][1], chain[name][1]), name
        else:
            assert fused[name] == chain[name], name
