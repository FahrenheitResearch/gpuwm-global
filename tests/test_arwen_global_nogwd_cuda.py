"""The fused float32 kernel of the non-orographic drag against its CuPy reference.

``nonorographic_gwd.apply_nonorographic_gwd`` has two paths: the fused CUDA
column kernel (the default on CuPy float32 batches) and the vectorised CuPy
form it replaced (``use_cuda=False``, ``ARWEN_NOGWD_CUDA=0``).  They do the
same arithmetic in a different order (the kernel sums the sixteen phase
speeds per azimuth in registers, the reference reduces a (4, 16, ncol) stack),
so they agree to float32 summation rounding, not bit for bit.  This holds
them together on 80-level columns with a jet, critical levels and saturation,
both latitude profiles and the heat booking on and off, and holds a run split
into latitude bands to the one-band result (each column launches at its own
level).

MEASURED 2026-11-23 on an RTX 5070 Ti, on this file's synthetic columns at
the T255 L80 size (384 x 768 x 80), 30 calls after 5 warm-ups: 121.5 ms a
call for the reference, 1.68 ms for the kernel.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

from arwen_global.physics import nonorographic_gwd as ng  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402

RD, CP_AIR, G = 287.04, 1004.64, 9.80616


def _columns(nlat=24, nlon=48, nz=80, seed=5):
    rng = np.random.default_rng(seed)
    coord = HybridCoordinate.ifs_class(nz, 100.0)
    a = np.asarray(coord.a_half_pa, dtype=np.float64)   # top -> bottom
    b = np.asarray(coord.b_half, dtype=np.float64)
    lat = np.linspace(-85.0, 85.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat2, lon2 = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    ps = 101_000.0 - 1_500.0 * np.cos(2 * lon2) * np.cos(lat2) + 200.0 * rng.standard_normal(lat2.shape)
    p_half_tb = a[:, None, None] + b[:, None, None] * ps[None]          # top -> bottom, nz + 1
    p_half = p_half_tb[::-1]                                             # bottom -> top
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    zeta = np.log(1.0e5 / p_full)
    # Temperature: troposphere lapse to a 215 K tropopause near 200 hPa, warming above.
    t = np.where(p_full > 2.0e4, 288.0 + 15.0 * np.cos(lat2) - 45.0 * zeta / np.log(5.0),
                 215.0 + 8.0 * (np.log(2.0e4 / p_full)))
    exner = (p_full / 1.0e5) ** (RD / CP_AIR)
    theta = t / exner
    dz = RD * t / G * np.log(p_half[:-1] / p_half[1:])
    # A westerly jet near 250 hPa that turns easterly in the stratosphere in the
    # tropics (critical levels for the eastward waves), plus noise.
    jet = 35.0 * np.exp(-((np.log(p_full / 2.5e4)) / 0.6) ** 2) * np.cos(lat2) ** 2
    strat = -20.0 * np.exp(-((lat2 / 0.4) ** 2)) * np.clip(zeta - 2.0, 0.0, None)
    u = 5.0 + jet + strat + 2.0 * rng.standard_normal(p_full.shape)
    v = 3.0 * np.sin(3 * lon2) * np.cos(lat2) + 1.0 * rng.standard_normal(p_full.shape)
    f32 = lambda x: cp.asarray(np.ascontiguousarray(x), dtype=cp.float32)
    batch = types.SimpleNamespace(xp=cp, arrays={
        "u": f32(u), "v": f32(v), "theta": f32(theta), "p_full": f32(p_full), "exner": f32(exner)})
    return batch, f32(dz), lat


def _run(batch, dz, lat, use_cuda, **kw):
    nz, nlat, nlon = batch.arrays["u"].shape
    out = {k: cp.zeros((nz, nlat, nlon), dtype=cp.float32) for k in ("du", "dv", "dtheta")}
    meta: dict = {}
    ng.apply_nonorographic_gwd(batch, out, dz=dz, latitude_deg=cp.asarray(lat, dtype=cp.float32),
                               metadata=meta, use_cuda=use_cuda, **kw)
    return {k: cp.asnumpy(v) for k, v in out.items()}, meta


@pytest.mark.parametrize("lat_profile", ["ifs", "uniform"])
@pytest.mark.parametrize("heat_booking", [True, False])
def test_the_kernel_is_the_reference_to_float32_summation(lat_profile, heat_booking):
    batch, dz, lat = _columns()
    ref, mref = _run(batch, dz, lat, False, lat_profile=lat_profile, heat_booking=heat_booking)
    ker, mker = _run(batch, dz, lat, True, lat_profile=lat_profile, heat_booking=heat_booking)
    assert mref == mker
    assert mker["nonorographic_gwd"] == "scinocca" and mker["nogwd_launch_pa"] == 45000.0
    # The case exercises the scheme: real drag, in both directions.
    assert np.max(np.abs(ref["du"])) > 1.0e-6 and np.min(ref["du"]) < 0.0 < np.max(ref["du"])
    for name in ("du", "dv", "dtheta"):
        r, k = ref[name], ker[name]
        assert np.all(np.isfinite(k)), name
        scale = float(np.max(np.abs(r)))
        if name == "dtheta" and not heat_booking:
            assert scale == 0.0 and float(np.max(np.abs(k))) == 0.0
            continue
        assert float(np.max(np.abs(k - r))) <= 2.0e-5 * scale, name
        rel_rms = float(np.sqrt(np.mean((k - r) ** 2)) / np.sqrt(np.mean(r ** 2)))
        assert rel_rms < 1.0e-5, (name, rel_rms)


@pytest.mark.parametrize("use_cuda", [True, False])
def test_a_latitude_band_reads_what_the_whole_globe_reads(use_cuda):
    """The launch level is each column's own: two bands of rows return the
    one-band result bit for bit, and the call's metadata is the same in both
    (an earlier form chose one level from the batch's mean pressure, 26
    in one T255 L80 band and 27 in the next, and the banded run refused)."""
    batch, dz, lat = _columns(nlat=24, nlon=48)
    # Mountains: a column whose surface sits at 600 hPa launches lower down.
    batch.arrays["p_full"][:, 5:8, 10:20] *= cp.float32(0.6)
    whole, meta = _run(batch, dz, lat, use_cuda)
    parts, metas = [], []
    for rows in (slice(0, 11), slice(11, 24)):
        sub = types.SimpleNamespace(xp=cp, arrays={k: cp.ascontiguousarray(v[:, rows]) for k, v in batch.arrays.items()})
        out, m = _run(sub, cp.ascontiguousarray(dz[:, rows]), lat[rows], use_cuda)
        parts.append(out)
        metas.append(m)
    for name in ("du", "dv", "dtheta"):
        assert np.array_equal(np.concatenate([p[name] for p in parts], axis=1), whole[name]), name
    assert metas[0] == metas[1] == meta


def test_the_environment_switch_selects_the_reference(monkeypatch):
    batch, dz, lat = _columns(nlat=8, nlon=16)
    monkeypatch.setenv("ARWEN_NOGWD_CUDA", "0")
    by_env, _ = _run(batch, dz, lat, None)
    ref, _ = _run(batch, dz, lat, False)
    for name in ("du", "dv", "dtheta"):
        assert np.array_equal(by_env[name], ref[name])
