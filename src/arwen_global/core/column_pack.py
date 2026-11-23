"""Column packing in one launch: level lanes to (columns, lanes, nz) blocks and back.

A column scheme that runs on a ``(columns, lanes, nz)`` block (gf.cu's
``gf_gfdrv_stage`` is the one this was written for) used to build the block
lane by lane: one ``ascontiguousarray`` of a transposed view and one strided
assignment per lane per chunk, then the reverse per output lane and per
scalar reading, about seventy launches a chunk before the scheme's own.
These kernels do the same moves in one launch per direction.

COPIES ONLY.  Every value is loaded and stored as a 32-bit word, nothing is
converted or rounded, so the packed block and the unpacked planes are the
lane-by-lane path's bits, NaN payloads included.  Callers hand the kernels
C-contiguous float32 (or int32) arrays and keep the lane-by-lane path for
anything else.
"""
from __future__ import annotations

import numpy as np

from ..spectral.backend import device_cache_key

_TILE = 32
_ROWS = 8

_SOURCE = r"""
#define TILE 32
#define ROWS 8
// (nz, ncol) lanes -> out[c, j, k] for columns [lo, lo + n): a tiled
// transpose per lane (blockIdx.z), coalesced on both sides.  A null lane
// writes zeros.
extern "C" __global__ void arwen_pack_lanes(
    const unsigned long long* __restrict__ src, unsigned int* __restrict__ out,
    const long long ncol, const long long lo, const long long n,
    const int nlanes, const int nz)
{
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int j = blockIdx.z;
    const long long c0 = (long long)blockIdx.x * TILE;
    const int k0 = blockIdx.y * TILE;
    const unsigned int* s = (const unsigned int*)src[j];
    for (int r = threadIdx.y; r < TILE; r += ROWS) {
        const int k = k0 + r;
        const long long c = c0 + threadIdx.x;
        unsigned int v = 0u;
        if (s != 0 && k < nz && c < n) v = s[(long long)k * ncol + lo + c];
        tile[r][threadIdx.x] = v;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < TILE; r += ROWS) {
        const long long c = c0 + r;
        const int k = k0 + threadIdx.x;
        if (k < nz && c < n)
            out[(c * nlanes + j) * (long long)nz + k] = tile[threadIdx.x][r];
    }
}

// block[c, lane[i], k] -> dst[i][k, lo + c]: the inverse move for the
// output lanes the caller names (blockIdx.z = i).
extern "C" __global__ void arwen_unpack_lanes(
    const unsigned int* __restrict__ block, const int* __restrict__ lanes,
    const unsigned long long* __restrict__ dst,
    const long long ncol, const long long lo, const long long n,
    const int nlanes, const int nz)
{
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int i = blockIdx.z;
    const int j = lanes[i];
    const long long c0 = (long long)blockIdx.x * TILE;
    const int k0 = blockIdx.y * TILE;
    unsigned int* d = (unsigned int*)dst[i];
    for (int r = threadIdx.y; r < TILE; r += ROWS) {
        const long long c = c0 + r;
        const int k = k0 + threadIdx.x;
        unsigned int v = 0u;
        if (k < nz && c < n) v = block[(c * nlanes + j) * (long long)nz + k];
        tile[r][threadIdx.x] = v;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < TILE; r += ROWS) {
        const int k = k0 + r;
        const long long c = c0 + threadIdx.x;
        if (k < nz && c < n) d[(long long)k * ncol + lo + c] = tile[threadIdx.x][r];
    }
}

// table[c, col[i]] -> dst[i][lo + c] for an (n, width) table of 32-bit
// words (blockIdx.y = i).
extern "C" __global__ void arwen_unpack_table(
    const unsigned int* __restrict__ table, const int* __restrict__ cols,
    const unsigned long long* __restrict__ dst,
    const long long lo, const long long n, const int width)
{
    const int i = blockIdx.y;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    ((unsigned int*)dst[i])[lo + c] = table[c * width + cols[i]];
}
"""

_KERNELS: dict[tuple, dict] = {}


def _kernels(xp):
    key = device_cache_key(xp)
    found = _KERNELS.get(key)
    if found is None:
        module = xp.RawModule(code=_SOURCE)
        found = {name: module.get_function(name) for name in (
            "arwen_pack_lanes", "arwen_unpack_lanes", "arwen_unpack_table")}
        _KERNELS[key] = found
    return found


def packable(xp, arrays, shape, dtypes=(np.float32,)) -> bool:
    """True when every non-None array is a C-contiguous ``shape`` array of
    one of ``dtypes`` on the device (the kernels' contract)."""
    if not hasattr(xp, "RawModule"):
        return False
    for a in arrays:
        if a is None:
            continue
        if (not isinstance(a, xp.ndarray) or tuple(a.shape) != tuple(shape)
                or a.dtype not in tuple(np.dtype(d) for d in dtypes)
                or not a.flags.c_contiguous):
            return False
    return True


def pointer_table(xp, arrays):
    """The device table of the arrays' base pointers (0 for None) the kernels
    read: build it once per call, it holds for every chunk of that call."""
    return xp.asarray(np.array(
        [0 if a is None else int(a.data.ptr) for a in arrays],
        dtype=np.uint64))


def index_table(xp, values):
    """A device int32 table (lane or column indices), built once per call."""
    return xp.asarray(np.asarray(values, dtype=np.int32))


def pack_lanes(xp, pointers, nlanes: int, out, *, lo: int, n: int,
               ncol: int, nz: int) -> None:
    """``out[c, j, :] = lane_j.reshape(nz, ncol)[:, lo + c]`` for every lane
    of ``pointers`` (:func:`pointer_table`; zeros for a None lane) in one
    launch.  ``out`` is a C-contiguous ``(n, nlanes, nz)`` block of 32-bit
    words; each lane is C-contiguous with ``nz * ncol`` 32-bit elements."""
    grid = ((int(n) + _TILE - 1) // _TILE, (int(nz) + _TILE - 1) // _TILE,
            int(nlanes))
    _kernels(xp)["arwen_pack_lanes"](
        grid, (_TILE, _ROWS),
        (pointers, out, np.int64(ncol), np.int64(lo), np.int64(n),
         np.int32(nlanes), np.int32(nz)))


def unpack_lanes(xp, block, lane_index, pointers, nplanes: int, *, lo: int,
                 n: int, ncol: int, nz: int) -> None:
    """``plane_i.reshape(nz, ncol)[:, lo:lo + n] = block[:, lane_index[i], :].T``
    for every plane of ``pointers`` in one launch (``lane_index`` from
    :func:`index_table`)."""
    grid = ((int(n) + _TILE - 1) // _TILE, (int(nz) + _TILE - 1) // _TILE,
            int(nplanes))
    _kernels(xp)["arwen_unpack_lanes"](
        grid, (_TILE, _ROWS),
        (block, lane_index, pointers, np.int64(ncol), np.int64(lo),
         np.int64(n), np.int32(block.shape[1]), np.int32(nz)))


def unpack_table(xp, table, columns, pointers, nplanes: int, *, lo: int,
                 n: int) -> None:
    """``plane_i[lo:lo + n] = table[:, columns[i]]`` for every plane of
    ``pointers`` in one launch (32-bit words; ``table`` is C-contiguous
    ``(n, width)``)."""
    threads = 256
    grid = ((int(n) + threads - 1) // threads, int(nplanes))
    _kernels(xp)["arwen_unpack_table"](
        grid, (threads,),
        (table, columns, pointers, np.int64(lo), np.int64(n),
         np.int32(table.shape[1])))


__all__ = ["packable", "pointer_table", "index_table", "pack_lanes",
           "unpack_lanes", "unpack_table"]
