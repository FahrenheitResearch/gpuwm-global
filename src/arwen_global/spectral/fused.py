"""CuPy ElementwiseKernel fast paths for the spectral/dynamics hot chains.

Every kernel here is a backend-conditional replacement for a numpy
expression that remains in place at its call site as the specification:
the fused path computes the same arithmetic in the same association (only
compiler contraction of multiply-add may move last bits).  The kernels are
built lazily so this module imports cleanly on machines without CuPy, and
each one is compared against its numpy specification on random and
adversarial inputs by the standalone GPU self-test that ships with the
fusion work (per-op max abs/rel diffs, nonzero exit beyond fp tolerance).

Call-site census that motivated each kernel (T533, one step, measured):
``project`` ran 470 times at 4 backend calls each; the RHS momentum/
bernoulli/scalar-tendency assemblies and the vertical upstream flux chain
were the remaining dynamics-side elementwise launches after the Legendre
m-loops were batched into strided GEMMs.
"""
from __future__ import annotations

import numpy as np

from .backend import device_cache_key

#: Compiled kernels, one entry per (name, module, DEVICE).  The device id
#: is in the key because a module-level cache that hands a kernel built on
#: one card to a launch on another is the multi-card defect
#: :func:`~arwen_global.spectral.backend.device_cache_key` documents; gate
#: CARD-1 builds every one of these on two device ids in one process and
#: asserts the entries are distinct.
_CACHE: dict[tuple, object] = {}


def _kernel(xp, name: str, in_params: str, out_params: str, operation: str):
    key = (name, *device_cache_key(xp))
    ker = _CACHE.get(key)
    if ker is None:
        ker = xp.ElementwiseKernel(in_params, out_params, operation, name)
        _CACHE[key] = ker
    return ker


def project_kernel(xp):
    """Fused triangular projection: mask + m=0 realify + fresh output.

    ``code`` is 0 above the triangle (coefficient zeroed), 2 on the m=0
    column inside the triangle (imaginary part dropped), 1 elsewhere
    inside the triangle (copied).  Specification: the numpy branch of
    ``SphericalHarmonicTransform.project``.  Matches it exactly for every
    finite input; a NON-FINITE above-triangle entry becomes an exact zero
    here where the specification's mask multiply would yield NaN.
    """
    return _kernel(
        xp,
        "arwen_spectral_project",
        "T x, int8 code",
        "T y",
        "y = (code == 0) ? T(0) : ((code == 2) ? T(x.real()) : x);",
    )


def complex_split_kernel(xp):
    """Real and imaginary parts of a (strided) complex array, in one pass.

    Specification (numpy branch of ``_batched_contract``)::

        re = ascontiguousarray(z.real)
        im = ascontiguousarray(z.imag)

    Pure data movement, no arithmetic, so every bit pattern (NaN payloads
    included) arrives as it left: the two copies read the complex operand
    twice at half its width each, this reads it once.
    """
    return _kernel(
        xp,
        "arwen_complex_split",
        "C z",
        "F re, F im",
        "re = z.real(); im = z.imag();",
    )


def expand_band_kernel(xp, layout: str):
    """One packed Legendre band written into the scratch, cleared slice and all.

    Specification (``transform.expand_band``)::

        'jn': dense[:, :, previous:m0] = 0; dense[:, :, m0:] = block
        'nj': dense[:, previous:m0, :] = 0; dense[:, m0:, :] = block

    The output is ``dense[..., previous:, ...]`` (``w = T+1-previous`` wide on
    the degree axis), the block is ``wb = T+1-m0`` wide there.  Moves only.
    """
    if layout == "jn":
        body = (
            "const ptrdiff_t row = i / w; const int n = prev + (int)(i % w);"
            "y = (n < m0) ? T(0) : block[row * (ptrdiff_t)wb + (n - m0)];"
        )
    else:
        body = (
            "const int j = (int)(i % nlat); const ptrdiff_t q = i / nlat;"
            "const int n = prev + (int)(q % w); const ptrdiff_t b = q / w;"
            "y = (n < m0) ? T(0) : block[(b * (ptrdiff_t)wb + (n - m0)) * nlat + j];"
        )
    return _kernel(
        xp,
        f"arwen_expand_band_{layout}",
        "raw T block, int32 prev, int32 m0, int32 w, int32 wb, int32 nlat",
        "T y",
        body,
    )


_TRANSPOSE_SOURCE = r"""
// Tiled layout moves for the Legendre GEMM operands: pure data movement,
// no arithmetic, through a 32x33 shared tile so both the read and the write
// are coalesced (the elementwise forms read or write with the order axis
// outermost, one 32-byte sector per element).
extern "C" __global__ void arwen_split_t_@T@(
    const @T2@* __restrict__ in, @T@* __restrict__ re, @T@* __restrict__ im,
    long long B, long long K, long long M)
{
    // in (B, K, M) contiguous complex -> re, im (M, B, K) contiguous
    __shared__ @T2@ tile[32][33];
    const long long m0 = (long long)blockIdx.x * 32;
    const long long k0 = (long long)blockIdx.y * 32;
    for (long long b = blockIdx.z; b < B; b += gridDim.z) {
        for (int j = threadIdx.y; j < 32; j += blockDim.y) {
            const long long k = k0 + j, m = m0 + threadIdx.x;
            if (k < K && m < M) tile[j][threadIdx.x] = in[(b * K + k) * M + m];
        }
        __syncthreads();
        for (int j = threadIdx.y; j < 32; j += blockDim.y) {
            const long long m = m0 + j, k = k0 + threadIdx.x;
            if (k < K && m < M) {
                const @T2@ v = tile[threadIdx.x][j];
                const long long o = (m * B + b) * K + k;
                re[o] = v.x;
                im[o] = v.y;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void arwen_join_t_@T@(
    const @T@* __restrict__ re, const @T@* __restrict__ im, @T2@* __restrict__ out,
    long long B, long long N, long long M, int realify)
{
    // re, im (M, B, N) contiguous -> out (B, N, M) contiguous complex;
    // realify: the m = 0 column's imaginary part is written +0, which is
    // what out[..., :, 0] = out[..., :, 0].real stored there.
    __shared__ @T@ tr[32][33];
    __shared__ @T@ ti[32][33];
    const long long n0 = (long long)blockIdx.x * 32;
    const long long m0 = (long long)blockIdx.y * 32;
    for (long long b = blockIdx.z; b < B; b += gridDim.z) {
        for (int j = threadIdx.y; j < 32; j += blockDim.y) {
            const long long m = m0 + j, n = n0 + threadIdx.x;
            if (m < M && n < N) {
                const long long i = (m * B + b) * N + n;
                tr[j][threadIdx.x] = re[i];
                ti[j][threadIdx.x] = (realify && m == 0) ? (@T@)0 : im[i];
            }
        }
        __syncthreads();
        for (int j = threadIdx.y; j < 32; j += blockDim.y) {
            const long long n = n0 + j, m = m0 + threadIdx.x;
            if (m < M && n < N) {
                @T2@ v;
                v.x = tr[threadIdx.x][j];
                v.y = ti[threadIdx.x][j];
                out[(b * N + n) * M + m] = v;
            }
        }
        __syncthreads();
    }
}
"""


def _transpose_kernel(xp, which: str, dtype):
    real = "float" if np.dtype(dtype) == np.float32 else "double"
    key = (f"transpose_{which}_{real}", *device_cache_key(xp))
    ker = _CACHE.get(key)
    if ker is None:
        source = _TRANSPOSE_SOURCE.replace("@T2@", real + "2").replace("@T@", real)
        ker = xp.RawKernel(source, f"arwen_{which}_t_{real}")
        _CACHE[key] = ker
    return ker


def _tile_grid(x: int, y: int, batch: int):
    return ((x + 31) // 32, (y + 31) // 32, max(1, min(int(batch), 65535)))


def split_transposed(xp, values, real_dtype):
    """``(B, K, M)`` C-contiguous complex -> two ``(M, B, K)`` real arrays.

    Specification: ``s = moveaxis(values, -1, 0).reshape(M, -1, K)``,
    ``ascontiguousarray(s.real)``, ``ascontiguousarray(s.imag)``.  Moves only.
    """
    *lead, k, m = values.shape
    b = 1
    for size in lead:
        b *= int(size)
    re = xp.empty((m, b, k), dtype=real_dtype)
    im = xp.empty_like(re)
    if re.size:
        _transpose_kernel(xp, "split", real_dtype)(
            _tile_grid(m, k, b), (32, 8, 1),
            (values, re, im, xp.int64(b), xp.int64(k), xp.int64(m)))
    return re, im


def join_transposed(xp, re, im, complex_dtype, *, realify: bool):
    """Two ``(M, B, N)`` real arrays -> ``(B, N, M)`` C-contiguous complex.

    Specification: ``out = empty((M, B, N)); out.real = re; out.imag = im;
    moveaxis(out, 0, -1)`` (values equal, the layout C-contiguous), and with
    ``realify`` the m = 0 column's imaginary part stored +0 as
    ``out[..., :, 0] = out[..., :, 0].real`` stores it.  Moves only.
    """
    m, b, n = re.shape
    out = xp.empty((b, n, m), dtype=complex_dtype)
    if out.size:
        _transpose_kernel(xp, "join", re.dtype)(
            _tile_grid(n, m, b), (32, 8, 1),
            (re, im, out, xp.int64(b), xp.int64(n), xp.int64(m),
             xp.int32(1 if realify else 0)))
    return out


def momentum_bernoulli_kernel(xp):
    """RHS momentum-forcing pair and Bernoulli function in one launch.

    Specification (numpy branch of ``MoistHybridModel.rhs``)::

        av   = vorticity + coriolis
        pf   = gas_constant * virtual_temperature * pressure_gradient
        mu   = av * v - wu - pf * grad_lnps_east
        mv   = -av * u - wv - pf * grad_lnps_north
        bern = 0.5 * (u**2 + v**2) + geopotential

    ``pg`` is the full-level factor ``grad(ln p_k) / grad(ln ps)`` from
    ``MoistHybridModel._pressure_gradient_factor`` (3-D); ``ge``/``gn`` are
    the 2-D gradient of ``ln ps`` and broadcast over the level axis (DN-3).
    """
    return _kernel(
        xp,
        "arwen_rhs_momentum_bernoulli",
        "T zeta, T cor, T u, T v, T wu, T wv, T tv, T pg, T ge, T gn, "
        "T phi, T rgas",
        "T mu, T mv, T bern",
        """
        const T av = zeta + cor;
        const T pf = rgas * tv * pg;
        mu = av * v - wu - pf * ge;
        mv = -av * u - wv - pf * gn;
        bern = (T)0.5 * (u * u + v * v) + phi;
        """,
    )


def tracer_flux_kernel(xp):
    """Horizontal mass-flux pair ``(dp*s*u, dp*s*v)`` in one launch.

    Specification: the numpy branch of ``MoistHybridModel._scalar_tendency``
    computes ``dp * scalar * u`` and ``dp * scalar * v``; the shared
    ``dp * scalar`` product is associated first in both paths.
    """
    return _kernel(
        xp,
        "arwen_rhs_tracer_flux",
        "T dp, T s, T u, T v",
        "T fx, T fy",
        """
        const T m = dp * s;
        fx = m * u;
        fy = m * v;
        """,
    )


def scalar_tendency_kernel(xp):
    """Flux-form scalar tendency combine ``(-h - vert - s*dp_t) / dp``.

    Specification: the numpy branch of ``MoistHybridModel._scalar_tendency``.
    """
    return _kernel(
        xp,
        "arwen_rhs_scalar_tendency",
        "T h, T vert, T s, T dpt, T dp",
        "T out",
        "out = (-h - vert - s * dpt) / dp;",
    )


def _limited_interface_flux(name: str, j: str) -> str:
    """C for the van Leer limited flux ``{name}`` through interface ``j``.

    Mirrors the numpy specification statement by statement, in the same
    association: upstream layer ``u`` (above when ``w >= 0``), its van
    Leer gradient ``2 * ga * gb / (ga + gb)`` when both one-sided
    gradients share a sign (zero otherwise; the boundary layers take the
    one-sided gradient to their one neighbour, DYC-1), the face value ``s[u] + grad * (p_half[j] - p_full[u])``
    bounded by the two adjacent layer values, times ``w``.  Interfaces
    0 and nlev carry zero flux.  ``base`` is the first index of the
    current tracer block, so no access crosses a tracer boundary.
    """
    return f"""
        T {name} = (T)0;
        {{
            const long long j = {j};
            if (j > 0 && j < nlev) {{
                const T wj = w[j * horiz + col];
                const long long u = (wj >= (T)0) ? (j - 1) : j;
                const T su = s[base + u * horiz + col];
                T grad = (T)0;
                if (u > 0 && u + 1 < nlev) {{
                    const T sa = s[base + (u - 1) * horiz + col];
                    const T sb = s[base + (u + 1) * horiz + col];
                    const T pa = pf[(u - 1) * horiz + col];
                    const T pu = pf[u * horiz + col];
                    const T pb = pf[(u + 1) * horiz + col];
                    const T ga = (su - sa) / (pu - pa);
                    const T gb = (sb - su) / (pb - pu);
                    const T prod = ga * gb;
                    if (prod > (T)0) {{
                        grad = (T)2 * prod / (ga + gb);
                    }}
                }} else if (u == 0 && nlev >= 2) {{
                    grad = (s[base + horiz + col] - su)
                         / (pf[horiz + col] - pf[col]);
                }} else if (u == nlev - 1 && nlev >= 2) {{
                    grad = (su - s[base + (u - 1) * horiz + col])
                         / (pf[u * horiz + col] - pf[(u - 1) * horiz + col]);
                }}
                T face = su + grad * (ph[j * horiz + col] - pf[u * horiz + col]);
                const T above = s[base + (j - 1) * horiz + col];
                const T below = s[base + j * horiz + col];
                const T lo = (above < below) ? above : below;
                const T hi = (above > below) ? above : below;
                face = (face > lo) ? face : lo;
                face = (face < hi) ? face : hi;
                {name} = wj * face;
            }}
        }}
    """


VERTICAL_FLUX_DIVERGENCE_OPERATION = (
    """
        const long long ii = (long long)i;
        const long long col = ii % horiz;
        const long long k = (ii / horiz) % nlev;
        const long long base = ii - (k * horiz + col);
    """
    + _limited_interface_flux("up", "k + 1")
    + _limited_interface_flux("dn", "k")
    + """
        out = up - dn;
    """
)


def vertical_flux_divergence_kernel(xp):
    """Van Leer limited vertical flux divergence, one launch, all interfaces.

    Specification: the numpy branch of
    ``MoistHybridModel._vertical_scalar_flux_divergence`` (monotone
    second-order MUSCL flux from the upstream layer; DN-4).  For output
    level ``k`` the result is ``flux[k+1] - flux[k]`` with interface
    fluxes built by ``_limited_interface_flux``.  ``s`` carries any
    leading (tracer) axes ahead of ``(nlev, nlat, nlon)``; ``w`` is the
    shared ``(nlev+1, nlat, nlon)`` interface pressure velocity, ``pf``
    the ``(nlev, nlat, nlon)`` full-level and ``ph`` the ``(nlev+1, nlat,
    nlon)`` half-level pressure; ``horiz = nlat * nlon``.  The level
    guards keep every neighbour access inside the same tracer block.
    """
    return _kernel(
        xp,
        "arwen_rhs_vertical_flux_divergence_vanleer",
        "raw T s, raw T w, raw T pf, raw T ph, int64 nlev, int64 horiz",
        "T out",
        VERTICAL_FLUX_DIVERGENCE_OPERATION,
    )


__all__ = [
    "VERTICAL_FLUX_DIVERGENCE_OPERATION",
    "momentum_bernoulli_kernel",
    "project_kernel",
    "scalar_tendency_kernel",
    "tracer_flux_kernel",
    "vertical_flux_divergence_kernel",
]
