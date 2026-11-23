"""The one-launch column packing (core.column_pack) moves bits, nothing else.

The pack is the lane-by-lane ``ascontiguousarray(lane[:, lo:hi].T)`` per
lane, a None lane is zeros, and the unpacks are the per-plane strided
assignments, word for word (NaN payloads and signed zeros included); and
the Grell-Freitas adapter, run once on the packed path and once with the
packing refused, returns the same bits in every output and every column
diagnostic over several chunks and a ragged last one.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import column_pack, gf  # noqa: E402


def _words(a):
    return cp.asnumpy(cp.ascontiguousarray(a)).view(np.uint32)


def _noisy(rng, shape):
    host = rng.standard_normal(shape).astype(np.float32)
    flat = host.reshape(-1)
    flat[::97] = np.float32(-0.0)
    flat[5::131] = np.frombuffer(np.uint32(0x7FC01234).tobytes(), np.float32)[0]
    return cp.asarray(host)


@pytest.mark.parametrize("lo,n", [(0, 64), (37, 1000), (4000, 96)])
def test_pack_and_unpack_are_the_lane_by_lane_words(lo, n):
    rng = np.random.default_rng(7)
    nz, ny, nx = 41, 13, 317
    ncol = ny * nx
    lanes = [_noisy(rng, (nz, ny, nx)) for _ in range(5)]
    lanes.insert(2, None)
    assert column_pack.packable(cp, lanes, (nz, ny, nx))
    block = cp.empty((n, len(lanes), nz), dtype=cp.float32)
    column_pack.pack_lanes(cp, column_pack.pointer_table(cp, lanes),
                           len(lanes), block, lo=lo, n=n, ncol=ncol, nz=nz)
    expected = cp.empty_like(block)
    for j, lane in enumerate(lanes):
        if lane is None:
            expected[:, j, :] = np.float32(0.0)
        else:
            expected[:, j, :] = cp.ascontiguousarray(
                lane.reshape(nz, ncol)[:, lo:lo + n].T)
    assert np.array_equal(_words(block), _words(expected))

    planes = [cp.full((nz, ncol), 3.0, dtype=cp.float32) for _ in range(3)]
    reference = [p.copy() for p in planes]
    pick = [4, 0, 5]
    column_pack.unpack_lanes(cp, block, column_pack.index_table(cp, pick),
                             column_pack.pointer_table(cp, planes), 3,
                             lo=lo, n=n, ncol=ncol, nz=nz)
    for plane, ref, j in zip(planes, reference, pick):
        ref[:, lo:lo + n] = block[:, j, :].T
        assert np.array_equal(_words(plane), _words(ref))

    table = _noisy(rng, (n, 9))
    itable = cp.asarray(rng.integers(-5, 5, (n, 4)).astype(np.int32))
    flat = [cp.full((ncol,), 1.5, dtype=cp.float32) for _ in range(3)]
    flat_i = [cp.full((ncol,), 9, dtype=cp.int32) for _ in range(2)]
    ref_f = [p.copy() for p in flat]
    ref_i = [p.copy() for p in flat_i]
    column_pack.unpack_table(cp, table, column_pack.index_table(cp, [8, 0, 3]),
                             column_pack.pointer_table(cp, flat), 3, lo=lo, n=n)
    column_pack.unpack_table(cp, itable, column_pack.index_table(cp, [1, 3]),
                             column_pack.pointer_table(cp, flat_i), 2, lo=lo, n=n)
    for plane, ref, j in zip(flat, ref_f, [8, 0, 3]):
        ref[lo:lo + n] = table[:, j]
        assert np.array_equal(_words(plane), _words(ref))
    for plane, ref, j in zip(flat_i, ref_i, [1, 3]):
        ref[lo:lo + n] = itable[:, j]
        assert np.array_equal(cp.asnumpy(plane), cp.asnumpy(ref))


def test_packable_refuses_what_the_kernels_cannot_read():
    a = cp.zeros((4, 3, 5), dtype=cp.float32)
    assert column_pack.packable(cp, [a, None], (4, 3, 5))
    assert not column_pack.packable(cp, [a.astype(cp.float64)], (4, 3, 5))
    assert not column_pack.packable(cp, [a[:, :, ::-1]], (4, 3, 5))
    assert not column_pack.packable(cp, [a[:3]], (4, 3, 5))


def _gf_case(nz=40, ny=24, nx=96, seed=3):
    rng = np.random.default_rng(seed)
    p_int = np.linspace(100000.0, 5000.0, nz + 1, dtype=np.float32)
    shape = (nz, ny, nx)

    def vol(base, spread):
        return cp.asarray((base + spread * rng.standard_normal(shape)).astype(np.float32))

    p_full = 0.5 * (p_int[:-1] + p_int[1:])
    pressure = cp.asarray(np.broadcast_to(p_full[:, None, None], shape).astype(np.float32).copy())
    temperature = cp.asarray(
        np.broadcast_to((300.0 - 60.0 * np.linspace(0, 1, nz))[:, None, None], shape)
        .astype(np.float32).copy()) + vol(0.0, 1.0)
    qv = cp.maximum(cp.asarray(np.broadcast_to(
        (0.016 * np.exp(-np.linspace(0, 6, nz)))[:, None, None], shape)
        .astype(np.float32).copy()) * (1 + vol(0.0, 0.05)), np.float32(1e-6))
    p_interface = cp.asarray(np.broadcast_to(p_int[:, None, None], (nz + 1, ny, nx))
                             .astype(np.float32).copy())
    atmosphere = {
        "u": vol(5.0, 3.0), "v": vol(0.0, 3.0), "temperature": temperature,
        "qv": qv, "pressure": pressure,
        "exner": (pressure / np.float32(100000.0)) ** np.float32(0.2857),
        "rho": pressure / (np.float32(287.0) * temperature),
        "dz": vol(400.0, 10.0), "p_interface": p_interface,
    }
    state = SimpleNamespace(
        p=pressure, w=cp.asarray((0.2 * rng.standard_normal((nz + 1, ny, nx))).astype(np.float32)),
        ht=cp.asarray((300 * rng.random((ny, nx))).astype(np.float32)), qi=atmosphere["qv"] * 0)
    fields = {
        "hfx": cp.asarray((100 * rng.random((ny, nx))).astype(np.float32)),
        "qfx": cp.asarray((1e-4 * rng.random((ny, nx))).astype(np.float32)),
        "xland": cp.asarray(rng.choice([1.0, 2.0], (ny, nx)).astype(np.float32)),
        "kpbl": cp.asarray(rng.integers(2, 10, (ny, nx)).astype(np.int32)),
    }
    cfg = SimpleNamespace(dt=300.0, ishallow=1, clos_choice=0, dx=34700.0)
    return atmosphere, fields, state, cfg


def _gf_outputs(seam, case):
    atmosphere, fields, state, cfg = case
    result = seam(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    out = {name: getattr(result, name) for name in (
        "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rainc")}
    out.update(seam.last_column_diagnostics)
    return {name: cp.asnumpy(v).view(np.uint32) for name, v in out.items()
            if v is not None}


def test_grell_freitas_packed_path_is_the_lane_by_lane_bits(monkeypatch):
    case = _gf_case()
    # A chunk far smaller than the grid: several packing passes and a
    # ragged last one, the lanes read at nonzero column offsets.
    packed = _gf_outputs(gf.GrellFreitas(column_chunk=1000), case)
    monkeypatch.setattr(column_pack, "packable", lambda *a, **k: False)
    plain = _gf_outputs(gf.GrellFreitas(column_chunk=1000), case)
    assert packed.keys() == plain.keys()
    for name in packed:
        assert np.array_equal(packed[name], plain[name]), name
