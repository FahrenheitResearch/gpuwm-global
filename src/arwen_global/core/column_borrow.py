"""The column filler's neighbour sweep as one launch per call.

``MoistHybridModel._borrow_from_neighbours`` walks the levels top down and,
for every level, takes from the level above and then the level below:
seven elementwise launches per (level, neighbour) pair, 546 a call at 40
levels and four calls a step (the two positivity repairs and the two
exchange clamps), about 2,200 launches a step whatever the resolution.

Here one thread owns one column and runs the same sweep in the same order
with the same operations: CuPy's own float maximum and minimum (NaN wins,
else ``max``/``min``), its negation, and one rounded subtraction, addition
and accumulation per transfer, in the array's dtype.  There is no multiply,
so nothing can contract into an FMA, and the kernel is compiled with
``-fmad=false`` regardless.  The swept masses and the moved plane are the
loop's bits, NaN payloads apart: a NaN stays a NaN in the same place, but
when both operands of a subtraction are NaN the card keeps one of them by
operand position, and the two compilers order it differently (on sm 8.9 a
float64 NaN layer comes back 0x7fffffffe0000000 from this kernel and
0x7ff8000000000000 from the loop; a bare ``a - c`` raw kernel does the same).
No forecast reads a payload: a non-finite state stops the run.
"""
from __future__ import annotations

import numpy as np

from ..spectral.backend import device_cache_key

_SOURCE = r"""
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
__device__ __forceinline__ TYPE cupy_max_(TYPE a, TYPE b) {
    return (isnan(a) | isnan(b)) ? (TYPE)(NAN) : (TYPE)(max(a, b));
}
__device__ __forceinline__ TYPE cupy_min_(TYPE a, TYPE b) {
    return (isnan(a) | isnan(b)) ? (TYPE)(NAN) : (TYPE)(min(a, b));
}
extern "C" __global__ void arwen_borrow_from_neighbours_TYPE(
    TYPE* __restrict__ mass, TYPE* __restrict__ moved,
    const long long plane, const int nlev)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= plane) return;
    TYPE m = (TYPE)0;
    for (int k = 0; k < nlev; ++k) {
        for (int side = 0; side < 2; ++side) {
            const int j = side == 0 ? k - 1 : k + 1;
            if (j < 0 || j >= nlev) continue;
            TYPE* mj = mass + (long long)j * plane + p;
            TYPE* mk = mass + (long long)k * plane + p;
            const TYPE give = cupy_min_(cupy_max_(*mj, (TYPE)0),
                                        cupy_max_(-(*mk), (TYPE)0));
            *mj = *mj - give;
            *mk = *mk + give;
            m = m + give;
        }
    }
    moved[p] = m;
}
"""

_KERNELS: dict[tuple, object] = {}


def _kernel(xp, dtype):
    name = "double" if np.dtype(dtype) == np.float64 else "float"
    key = (*device_cache_key(xp), name)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = xp.RawKernel(
            _SOURCE.replace("TYPE", name),
            f"arwen_borrow_from_neighbours_{name}", options=("-fmad=false",))
        _KERNELS[key] = kernel
    return kernel


def fusable(xp, mass) -> bool:
    """A C-contiguous float32/float64 CuPy array of at least one level."""
    return (hasattr(xp, "RawKernel") and isinstance(mass, xp.ndarray)
            and mass.dtype in (xp.float32, xp.float64)
            and mass.flags.c_contiguous and mass.ndim >= 1
            and mass.shape[0] >= 1 and mass.size > 0)


def borrow_from_neighbours(xp, mass):
    """The sweep in place on ``mass`` (nlev, ...); returns the moved plane."""
    nlev = int(mass.shape[0])
    plane = int(mass.size // nlev)
    moved = xp.empty(mass.shape[1:], dtype=mass.dtype)
    threads = 256
    _kernel(xp, mass.dtype)(
        ((plane + threads - 1) // threads,), (threads,),
        (mass, moved, np.int64(plane), np.int32(nlev)))
    return moved


__all__ = ["fusable", "borrow_from_neighbours"]
