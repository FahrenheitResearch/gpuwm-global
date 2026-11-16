/* Semi-Lagrangian departure points and 3-D interpolation on a Gaussian grid.
 *
 * Two entry families, both written once as a templated device body and
 * exposed through extern "C" wrappers, so there is one compiled kernel per
 * (scalar type, quasi-monotone, deficit-reporting) triple and no name
 * mangling to reason about.
 *
 *   sl_gather_*      tricubic 4x4x4 gather of F fields that share one index
 *                    and weight computation.  The measured shape: the
 *                    vertical tap loop is PARTIALLY unrolled.  Fully
 *                    unrolling it holds 64 stencil values live at once and
 *                    read 1.75x slow on the 5070 Ti; not unrolling at all
 *                    spilled the weights to local memory and read 3.4x slow.
 *   sl_gather_qmd_*  the same, plus the signed mass the quasi-monotone
 *                    limiter moved at every point.  The additive mass
 *                    fixer needs it to put a species' correction back
 *                    where the limiter took it rather than spreading it
 *                    over every point that holds the species.
 *   sl_gather5_*     the QUINTIC-horizontal gather: six-point Lagrange in
 *                    the zonal and meridional directions and the same
 *                    four-point cubic in the vertical (6x6x4 = 144 taps
 *                    against 64), for the dynamical bundle.  MEASURED
 *                    2026-09-06 on the T255 forecast day: the cubic gather
 *                    every 300 s step kept 17 percent of the Eulerian
 *                    core's 500 hPa vorticity power at total wavenumbers
 *                    181 to 230, and the four-point Lagrange weights at a
 *                    fifth of a cell lose 7.6 percent of a 3.8-cell wave's
 *                    amplitude per pass, which over 288 passes a day is
 *                    an effective drain 25 times the shipped
 *                    hyperdiffusion's at that scale.  The six-point
 *                    weights lose about half of that.  Same template, a
 *                    second table set (three reflected rows a pole rather
 *                    than two), and the cubic entry points untouched.
 *   sl_linear_*      trilinear interpolation inside the limiter's own cell,
 *                    the low-order interpolant the Bermejo-Conde mass
 *                    fixer weighs the high-order one against.
 *   sl_departure_*   the spherical fixed-point departure-point search, all
 *                    of its iterations in one launch, with the wind read as
 *                    geocentric Cartesian components so the poles need no
 *                    local basis and no sign convention.
 *
 * A POSITIVE-DEFINITE shape, [0, cell maximum] instead of [cell minimum,
 * cell maximum], was built here and MEASURED on 2026-09-06 and is not in
 * this file: on a condensate-shaped field (blobs over a zero background)
 * it returned the same array as the quasi-monotone shape at every point,
 * because every undershoot that shape lifts is already below zero there,
 * and on a field with a positive floor its mass error was 2.15e-3 against
 * the quasi-monotone shape's 8.04e-5.  Identical where it would have
 * helped and 27 times worse where it would not.
 *
 * Grid conventions this file assumes, every one of them checked on the
 * Python side before a launch:
 *   - longitudes are uniform, lon[i] = i * dlam, dlam = 2*pi/ni, ni EVEN
 *     (the polar reflection shifts by exactly ni/2 columns)
 *   - latitudes are the Gauss-Legendre nodes, ASCENDING, handed in as an
 *     EXTENDED table lat_ext[nj+4] whose first two entries are the two rows
 *     reflected through the south pole and whose last two are the two rows
 *     reflected through the north pole, so the table is strictly increasing
 *     over its whole length and a 4-point stencil always fits
 *   - rowoff[nj+4] and rowshift[nj+4] map an extended row to the offset of
 *     the DATA row it reads and to the zonal shift (0 or ni/2) it reads with
 *   - full level k sits at continuous index k, k in [0, nk-1], nk >= 4;
 *     departure levels are clamped to the boundary interfaces,
 *     [-1/2, nk - 1/2]
 *
 * The weights of all three directions are NORMALIZED by their own sum.  That
 * is what makes the zero-displacement identity exact: at a grid point three
 * of the four Lagrange weights are exactly zero (each carries the vanishing
 * factor) and the fourth is divided by itself.  Without the normalization
 * the meridional weight at a node is 1 + O(eps) rather than 1, because it is
 * a product of node differences and their reciprocals rather than a closed
 * form, and the identity would fail by a few ulp at every point.
 */

#define SL_HALF_PI_D 1.57079632679489661923

template <typename T> struct sl_traits;

template <> struct sl_traits<float> {
    __device__ static float floor_(float x) { return floorf(x); }
    __device__ static float asin_(float x) { return asinf(x); }
    __device__ static float atan2_(float y, float x) { return atan2f(y, x); }
    __device__ static void sincos_(float x, float* s, float* c) { sincosf(x, s, c); }
    __device__ static float rsqrt_(float x) { return rsqrtf(x); }
    __device__ static float sqrt_(float x) { return sqrtf(x); }
    __device__ static float abs_(float x) { return fabsf(x); }
    __device__ static float big() { return 3.402823466e38f; }
};

template <> struct sl_traits<double> {
    __device__ static double floor_(double x) { return floor(x); }
    __device__ static double asin_(double x) { return asin(x); }
    __device__ static double atan2_(double y, double x) { return atan2(y, x); }
    __device__ static void sincos_(double x, double* s, double* c) { sincos(x, s, c); }
    __device__ static double rsqrt_(double x) { return rsqrt(x); }
    __device__ static double sqrt_(double x) { return sqrt(x); }
    __device__ static double abs_(double x) { return fabs(x); }
    __device__ static double big() { return 1.7976931348623157e308; }
};

/* Lagrange weights on the equispaced nodes 0,1,2,3 evaluated at t, then
   normalized.  At t = 0,1,2,3 exactly one weight is 1 and the rest are 0,
   in floating point as well as in exact arithmetic. */
template <typename T>
__device__ __forceinline__ void sl_cubic_equispaced(T t, T* w)
{
    const T a = t;
    const T b = t - (T)1;
    const T c = t - (T)2;
    const T d = t - (T)3;
    w[0] = -b * c * d * (T)(1.0 / 6.0);
    w[1] =  a * c * d * (T)0.5;
    w[2] = -a * b * d * (T)0.5;
    w[3] =  a * b * c * (T)(1.0 / 6.0);
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= inv; w[1] *= inv; w[2] *= inv; w[3] *= inv;
}

/* Lagrange weights on the equispaced nodes 0..5 evaluated at t, normalized
   the same way: at an integer t exactly one weight survives. */
template <typename T>
__device__ __forceinline__ void sl_quintic_equispaced(T t, T* w)
{
    const T d0 = t;
    const T d1 = t - (T)1;
    const T d2 = t - (T)2;
    const T d3 = t - (T)3;
    const T d4 = t - (T)4;
    const T d5 = t - (T)5;
    w[0] = -d1 * d2 * d3 * d4 * d5 * (T)(1.0 / 120.0);
    w[1] =  d0 * d2 * d3 * d4 * d5 * (T)(1.0 / 24.0);
    w[2] = -d0 * d1 * d3 * d4 * d5 * (T)(1.0 / 12.0);
    w[3] =  d0 * d1 * d2 * d4 * d5 * (T)(1.0 / 12.0);
    w[4] = -d0 * d1 * d2 * d3 * d5 * (T)(1.0 / 24.0);
    w[5] =  d0 * d1 * d2 * d3 * d4 * (T)(1.0 / 120.0);
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]);
    w[0] *= inv; w[1] *= inv; w[2] *= inv;
    w[3] *= inv; w[4] *= inv; w[5] *= inv;
}

/* Latitude of a unit vector, by atan2 against the horizontal radius rather
   than asin of the vertical component.  d(asin)/dz is 1/cos(phi), which is
   156 on the outermost T255 ring, so the last bit of a float32 unit vector
   becomes 120 m of latitude there and the departure point of a resting
   atmosphere is not the arrival point.  atan2(z, hypot(x,y)) carries the
   same relative precision at every latitude, including exactly at a pole. */
template <typename T>
__device__ __forceinline__ T sl_latitude(T x, T y, T z)
{
    return sl_traits<T>::atan2_(z, sl_traits<T>::sqrt_(x * x + y * y));
}

/* The extended-row bracket m0 with lat_ext[m0] <= p < lat_ext[m0+1], clamped
   so the 4-point stencil m0-1 .. m0+2 stays inside the table.  The lookup
   table lands within a row or so; the two loops then make it exact for any
   node distribution, so a change of quadrature cannot silently mislocate. */
template <typename T>
__device__ __forceinline__ int sl_bracket(
    T p, const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    int nb, int nj, T lut_scale)
{
    int b = (int)((p + (T)SL_HALF_PI_D) * lut_scale);
    b = min(max(b, 0), nb - 1);
    int m0 = mlut[b];
    while (m0 > 1 && p < lat_ext[m0]) --m0;
    while (m0 < nj + 1 && p >= lat_ext[m0 + 1]) ++m0;
    return m0;
}

/* Meridional Lagrange weights on the true nodes, normalized. */
template <typename T>
__device__ __forceinline__ void sl_meridional_weights(
    T p, int sst, const T* __restrict__ lat_ext,
    const T* __restrict__ mrden, T* w)
{
    const T d0 = p - lat_ext[sst];
    const T d1 = p - lat_ext[sst + 1];
    const T d2 = p - lat_ext[sst + 2];
    const T d3 = p - lat_ext[sst + 3];
    const T* r = mrden + 4 * sst;
    w[0] = d1 * d2 * d3 * r[0];
    w[1] = d0 * d2 * d3 * r[1];
    w[2] = d0 * d1 * d3 * r[2];
    w[3] = d0 * d1 * d2 * r[3];
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3]);
    w[0] *= inv; w[1] *= inv; w[2] *= inv; w[3] *= inv;
}

/* The six-point bracket: lat_ext6[m0] <= p < lat_ext6[m0+1], m0 clamped to
   [2, nj+2] so the stencil m0-2 .. m0+3 stays inside the wider table. */
template <typename T>
__device__ __forceinline__ int sl_bracket6(
    T p, const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    int nb, int nj, T lut_scale)
{
    int b = (int)((p + (T)SL_HALF_PI_D) * lut_scale);
    b = min(max(b, 0), nb - 1);
    int m0 = mlut[b];
    while (m0 > 2 && p < lat_ext[m0]) --m0;
    while (m0 < nj + 2 && p >= lat_ext[m0 + 1]) ++m0;
    return m0;
}

/* Six-point meridional Lagrange weights on the true nodes, normalized. */
template <typename T>
__device__ __forceinline__ void sl_meridional_weights6(
    T p, int sst, const T* __restrict__ lat_ext,
    const T* __restrict__ mrden, T* w)
{
    T d[6];
#pragma unroll
    for (int m = 0; m < 6; ++m) d[m] = p - lat_ext[sst + m];
    const T* r = mrden + 6 * sst;
#pragma unroll
    for (int m = 0; m < 6; ++m) {
        T prod = r[m];
#pragma unroll
        for (int q = 0; q < 6; ++q) if (q != m) prod *= d[q];
        w[m] = prod;
    }
    const T inv = (T)1 / (w[0] + w[1] + w[2] + w[3] + w[4] + w[5]);
#pragma unroll
    for (int m = 0; m < 6; ++m) w[m] *= inv;
}

/* One body for both horizontal widths.  NH = 4 is the tricubic gather the
   gates of record were measured on and its arithmetic is untouched: the
   NH = 6 branches are compile-time and the four-point weights, bracket and
   address arithmetic are the same expressions in the same order. */
/* BAND = 1 is the latitude-band form (semilag.tables.BandWindow): the
   departure coordinates and the output cover the band's nja rows, the
   source fields hold only the band's njs rows plus halo, and rowoff maps
   an extended row to its offset INSIDE those held rows, or to -1 for a row
   the band does not hold.  A read of such a row is an escape: it is
   flagged in *esc and redirected to row 0 of the held rows so the read
   stays in bounds, and the caller refuses the result.  The weights, the
   bracket and every floating-point expression are the BAND = 0 ones in the
   same order; only the integer address of a data row differs, so a band
   returns the whole-grid gather's bits at every point it computes. */
template <typename T, int QM, int UNROLL, int DEFICIT, int NH, int BAND>
__device__ __forceinline__ void sl_gather_body(
    const T* __restrict__ src,
    const T* __restrict__ xis, const T* __restrict__ phi,
    const T* __restrict__ zk,
    const T* __restrict__ lat_ext, const T* __restrict__ mrden,
    const int* __restrict__ mlut, const int* __restrict__ rowoff,
    const int* __restrict__ rowshift,
    T* __restrict__ out, T* __restrict__ dev,
    int F, int nk, int nj, int ni, int nb, T lut_scale,
    int njs, int nja, int* __restrict__ esc)
{
    const int rows_out = BAND ? nja : nj;
    const int rows_src = BAND ? njs : nj;
    const long long n = (long long)nk * (long long)rows_out * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    T p = phi[gid];
    p = min(max(p, -(T)SL_HALF_PI_D), (T)SL_HALF_PI_D);
    /* Clamped at the lid and ground interfaces (DYC-1); a point in the
       half layer outside the outermost full level reads the one-sided
       cubic there, held between the outermost level's value and the
       linear reconstruction at the interface for the unlimited fields. */
    T xk = zk[gid];
    xk = min(max(xk, (T)-0.5), (T)nk - (T)0.5);
    const bool outside = (xk < (T)0) || (xk > (T)(nk - 1));
    const int c_edge = (xk < (T)0) ? 0 : 3;
    const int c_next = (xk < (T)0) ? 1 : 2;
    const T xi = xis[gid];

    /* zonal: uniform nodes, periodic, never clamped */
    const T fi = sl_traits<T>::floor_(xi);
    const int i0 = (int)fi - (NH / 2 - 1);
    T wx[NH];
    if (NH == 6) sl_quintic_equispaced<T>(xi - (T)i0, wx);
    else         sl_cubic_equispaced<T>(xi - (T)i0, wx);
    int a[NH], b[NH];
    a[0] = i0 % ni; if (a[0] < 0) a[0] += ni;
#pragma unroll
    for (int q = 1; q < NH; ++q) { a[q] = a[q - 1] + 1; if (a[q] >= ni) a[q] -= ni; }
    const int half = ni >> 1;
#pragma unroll
    for (int q = 0; q < NH; ++q) { b[q] = a[q] + half; if (b[q] >= ni) b[q] -= ni; }

    /* meridional: true Gauss-Legendre nodes, poles by reflection */
    int sst;
    T wy[NH];
    if (NH == 6) {
        const int m0 = sl_bracket6<T>(p, lat_ext, mlut, nb, nj, lut_scale);
        sst = m0 - 2;
        sl_meridional_weights6<T>(p, sst, lat_ext, mrden, wy);
    } else {
        const int m0 = sl_bracket<T>(p, lat_ext, mlut, nb, nj, lut_scale);
        sst = m0 - 1;
        sl_meridional_weights<T>(p, sst, lat_ext, mrden, wy);
    }

    /* vertical: uniform in the level index, clamped to the data range */
    const int kb = min(max((int)sl_traits<T>::floor_(xk), 0), nk - 2);
    int k0 = kb - 1;
    k0 = min(max(k0, 0), nk - 4);
    const int kbox = kb - k0;
    T wz[4];
    sl_cubic_equispaced<T>(xk - (T)k0, wz);

    int adr[NH * NH];
#pragma unroll
    for (int m = 0; m < NH; ++m) {
        int ro = rowoff[sst + m];
        if (BAND && ro < 0) { atomicOr(esc, 1); ro = 0; }
        const int sh = rowshift[sst + m];
#pragma unroll
        for (int q = 0; q < NH; ++q) adr[NH * m + q] = ro + (sh ? b[q] : a[q]);
    }
    /* the limiter's cell: the two rows and two columns around the point */
    const int mid = NH / 2 - 1;

    const long long plane = (long long)rows_src * (long long)ni;
    const long long nsrc = (long long)nk * plane;
    /* The stencil's own centre node, read once per field: every tap is
       summed as its DIFFERENCE from it and the centre is added back once.
       A tap sum written as sum w v rounds at the size of v, the same way
       at the same point every step of a steady flow, so it walks a
       steady state off by a sub-ulp of |v| per step (ln ps near 11.5,
       theta near 1800 K at the lid); summed about the centre it rounds at
       the size of the field's variation across one stencil.  The centre
       is the stencil node NEAREST the point in each direction, so at a
       node it is that node, every difference and every off-node weight is
       exactly zero and the zero-displacement identity stays exact. */
    const int cx = (xi - fi >= (T)0.5) ? mid + 1 : mid;
    const int cy = (p - lat_ext[sst + mid] >= lat_ext[sst + mid + 1] - p)
                       ? mid + 1 : mid;
    const int kn = min(max((int)sl_traits<T>::floor_(xk + (T)0.5), 0), nk - 1);
    const long long centre = (long long)kn * plane + adr[NH * cy + cx];
    for (int f = 0; f < F; ++f) {
        const T* s = src + (long long)f * nsrc;
        const T base = s[centre];
        T acc = (T)0;
        T lo = sl_traits<T>::big();
        T hi = -sl_traits<T>::big();
        T f_edge = (T)0, f_next = (T)0;
#pragma unroll UNROLL
        for (int c = 0; c < 4; ++c) {
            const T* sp = s + (long long)(k0 + c) * plane;
            T rowsum = (T)0;
#pragma unroll
            for (int m = 0; m < NH; ++m) {
                T v[NH];
#pragma unroll
                for (int q = 0; q < NH; ++q) v[q] = sp[adr[NH * m + q]];
                if (QM && (c == kbox || c == kbox + 1)
                        && (m == mid || m == mid + 1)) {
                    lo = min(lo, min(v[mid], v[mid + 1]));
                    hi = max(hi, max(v[mid], v[mid + 1]));
                }
#pragma unroll
                for (int q = 0; q < NH; ++q) v[q] -= base;
                T colsum = wx[0] * v[0] + wx[1] * v[1]
                         + wx[2] * v[2] + wx[3] * v[3];
                if (NH == 6) colsum += wx[4] * v[4] + wx[5] * v[5];
                rowsum += wy[m] * colsum;
            }
            acc += wz[c] * rowsum;
            if (c == c_edge) f_edge = rowsum;
            if (c == c_next) f_next = rowsum;
        }
        if (!QM && outside) {
            /* Half layer between the outermost full level and its
             * boundary interface: the one-sided cubic is held between the
             * outermost level's value and the linear reconstruction at
             * the interface, so the extrapolation hands the sponge no
             * extremum the column's own end slope does not imply.  The
             * limited fields keep their cell box, which already bounds
             * them by the outermost two levels. */
            const T f_face = f_edge + (T)0.5 * (f_edge - f_next);
            const T blo = min(f_edge, f_face);
            const T bhi = max(f_edge, f_face);
            acc = min(max(acc, blo), bhi);
        }
        /* Back to the field's own values; the limiter's box was read off
           the raw taps, so it bounds the raw result exactly. */
        acc += base;
        T kept = acc;
        if (QM) kept = min(max(acc, lo), hi);
        out[(long long)f * n + gid] = kept;
        /* The signed mass the clip moved at this point, raw minus kept:
         * positive where the limiter cut an overshoot away (mass the
         * gather lost) and negative where it lifted an undershoot (mass
         * the gather gained).  It is the ONLY place the interpolation
         * knows where its mass went, and the additive mass fixer puts the
         * correction back exactly there instead of spreading it over
         * every point that holds the species. */
        if (DEFICIT) dev[(long long)f * n + gid] = acc - kept;
    }
}

extern "C" __global__ void sl_gather_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 0, 2, 0, 4, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                 rowoff, rowshift, out, (float*)0, F, nk, nj,
                                 ni, nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather_qm_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 1, 2, 0, 4, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                 rowoff, rowshift, out, (float*)0, F, nk, nj,
                                 ni, nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 0, 2, 0, 4, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                  rowoff, rowshift, out, (double*)0, F, nk, nj,
                                  ni, nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather_qm_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 1, 2, 0, 4, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                  rowoff, rowshift, out, (double*)0, F, nk, nj,
                                  ni, nb, lut_scale, 0, 0, (int*)0); }

/* The quasi-monotone gather that also reports where its limiter moved
 * mass.  Same arithmetic in ``out``, bit for bit, as sl_gather_qm_*: the
 * clip is computed either way and this variant writes the difference it
 * would otherwise have discarded.  The gate that says so is
 * test_the_deficit_gather_returns_the_same_values_as_the_plain_one. */
extern "C" __global__ void sl_gather_qmd_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out, float* dev,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 1, 2, 1, 4, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                 rowoff, rowshift, out, dev, F, nk, nj, ni,
                                 nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather_qmd_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out, double* dev,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 1, 2, 1, 4, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                  rowoff, rowshift, out, dev, F, nk, nj, ni,
                                  nb, lut_scale, 0, 0, (int*)0); }

/* The quintic-horizontal gather.  Its table arguments are the SIX-POINT
 * tables (lat_ext6, mrden6, mlut6, rowoff6, rowshift6, their bin count and
 * scale); handing it the four-point tables reads the wrong rows, which is
 * why interpolate.gather_batch selects the set from the order and never
 * from the caller. */
extern "C" __global__ void sl_gather5_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 0, 2, 0, 6, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                    rowoff, rowshift, out, (float*)0, F, nk,
                                    nj, ni, nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather5_qm_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_gather_body<float, 1, 2, 0, 6, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                    rowoff, rowshift, out, (float*)0, F, nk,
                                    nj, ni, nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather5_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 0, 2, 0, 6, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                     rowoff, rowshift, out, (double*)0, F, nk,
                                     nj, ni, nb, lut_scale, 0, 0, (int*)0); }

extern "C" __global__ void sl_gather5_qm_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_gather_body<double, 1, 2, 0, 6, 0>(src, xis, phi, zk, lat_ext, mrden, mlut,
                                     rowoff, rowshift, out, (double*)0, F, nk,
                                     nj, ni, nb, lut_scale, 0, 0, (int*)0); }

/* The latitude-band entry points (BAND = 1, semilag.tables.BandWindow):
 * the same bodies, with the held source rows njs, the computed rows
 * nja and the escape flag appended.  The deficit pointer is the one
 * extra argument the qmd variants carry, exactly as above. */
extern "C" __global__ void sl_gather_band_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<float, 0, 2, 0, 4, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (float*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather_qm_band_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<float, 1, 2, 0, 4, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (float*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather_band_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<double, 0, 2, 0, 4, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (double*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather_qm_band_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<double, 1, 2, 0, 4, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (double*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather_qmd_band_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out, float* dev,
    int F, int nk, int nj, int ni, int nb, float lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<float, 1, 2, 1, 4, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, dev, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather_qmd_band_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out, double* dev,
    int F, int nk, int nj, int ni, int nb, double lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<double, 1, 2, 1, 4, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, dev, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather5_band_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<float, 0, 2, 0, 6, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (float*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather5_qm_band_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const float* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<float, 1, 2, 0, 6, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (float*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather5_band_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<double, 0, 2, 0, 6, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (double*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

extern "C" __global__ void sl_gather5_qm_band_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const double* mrden, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale,
    int njs, int nja, int* esc)
{ sl_gather_body<double, 1, 2, 0, 6, 1>(src, xis, phi, zk, lat_ext, mrden,
    mlut, rowoff, rowshift, out, (double*)0, F, nk, nj, ni, nb, lut_scale,
    njs, nja, esc); }

/* ------------------------------------------------------------------ */
/* The LOW-ORDER interpolant the Bermejo-Conde mass fixer weighs by.     */
/*
 * Trilinear interpolation at the departure point inside the limiter's own
 * cell: the two columns floor(xi) and floor(xi)+1, the bracketing extended
 * rows m0 and m0+1 (the cubic gather's inner pair, poles by the same
 * reflection), and the levels kb and kb+1 with kb the cubic gather's own
 * clamped base.  Its value therefore lies inside the quasi-monotone box
 * [lo, hi] the cubic gather clips to, it is nonnegative wherever the
 * source is, and at a zero displacement it returns the field exactly.
 *
 * The fixer's weight is max(0, sgn(d) (q_L - q_H)) with q_H the limited
 * cubic value, d the mass to restore
 * and q_L this one (the Bermejo-Conde fixer, Mon. Wea. Rev. 130, 2002, in
 * the form the IFS tracer mass fixers use): the difference between a high
 * and a low order interpolant is large exactly where the field is not
 * resolved, which is where the interpolation made its mass error.
 */
template <typename T>
__device__ __forceinline__ void sl_linear_body(
    const T* __restrict__ src,
    const T* __restrict__ xis, const T* __restrict__ phi,
    const T* __restrict__ zk,
    const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    const int* __restrict__ rowoff, const int* __restrict__ rowshift,
    T* __restrict__ out, int F, int nk, int nj, int ni, int nb, T lut_scale)
{
    const long long n = (long long)nk * (long long)nj * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    T p = phi[gid];
    p = min(max(p, -(T)SL_HALF_PI_D), (T)SL_HALF_PI_D);
    T xk = zk[gid];
    xk = min(max(xk, (T)0), (T)(nk - 1));
    const T xi = xis[gid];

    const T fi = sl_traits<T>::floor_(xi);
    const T ax = xi - fi;
    int c0 = ((int)fi) % ni; if (c0 < 0) c0 += ni;
    int c1 = c0 + 1; if (c1 >= ni) c1 -= ni;
    const int half = ni >> 1;
    int d0 = c0 + half; if (d0 >= ni) d0 -= ni;
    int d1 = c1 + half; if (d1 >= ni) d1 -= ni;

    const int m0 = sl_bracket<T>(p, lat_ext, mlut, nb, nj, lut_scale);
    const T la = lat_ext[m0];
    const T lb = lat_ext[m0 + 1];
    T ay = (p - la) / (lb - la);
    ay = min(max(ay, (T)0), (T)1);
    const int r0 = rowoff[m0], s0 = rowshift[m0];
    const int r1 = rowoff[m0 + 1], s1 = rowshift[m0 + 1];
    const int i00 = r0 + (s0 ? d0 : c0), i01 = r0 + (s0 ? d1 : c1);
    const int i10 = r1 + (s1 ? d0 : c0), i11 = r1 + (s1 ? d1 : c1);

    const int kb = min(max((int)sl_traits<T>::floor_(xk), 0), nk - 2);
    const T az = xk - (T)kb;

    const long long plane = (long long)nj * (long long)ni;
    const long long o0 = (long long)kb * plane;
    const long long o1 = o0 + plane;
    const T bx = (T)1 - ax, by = (T)1 - ay, bz = (T)1 - az;
    for (int f = 0; f < F; ++f) {
        const T* s = src + (long long)f * n;
        const T t0 = by * (bx * s[o0 + i00] + ax * s[o0 + i01])
                   + ay * (bx * s[o0 + i10] + ax * s[o0 + i11]);
        const T t1 = by * (bx * s[o1 + i00] + ax * s[o1 + i01])
                   + ay * (bx * s[o1 + i10] + ax * s[o1 + i11]);
        out[(long long)f * n + gid] = bz * t0 + az * t1;
    }
}

extern "C" __global__ void sl_linear_f32(
    const float* src, const float* xis, const float* phi, const float* zk,
    const float* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift, float* out,
    int F, int nk, int nj, int ni, int nb, float lut_scale)
{ sl_linear_body<float>(src, xis, phi, zk, lat_ext, mlut, rowoff, rowshift,
                        out, F, nk, nj, ni, nb, lut_scale); }

extern "C" __global__ void sl_linear_f64(
    const double* src, const double* xis, const double* phi, const double* zk,
    const double* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift, double* out,
    int F, int nk, int nj, int ni, int nb, double lut_scale)
{ sl_linear_body<double>(src, xis, phi, zk, lat_ext, mlut, rowoff, rowshift,
                         out, F, nk, nj, ni, nb, lut_scale); }

/* ------------------------------------------------------------------ */
/* trilinear gather of four fields at one point, used by the search    */

template <typename T, int BAND>
__device__ __forceinline__ void sl_trilinear4(
    const T* __restrict__ f0, const T* __restrict__ f1,
    const T* __restrict__ f2, const T* __restrict__ f3,
    T lamd, T phid, T xkd,
    const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    const int* __restrict__ rowoff, const int* __restrict__ rowshift,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,
    T* out, int njs, int* __restrict__ esc)
{
    const T p = min(max(phid, -(T)SL_HALF_PI_D), (T)SL_HALF_PI_D);
    const T xk = min(max(xkd, (T)0), (T)(nk - 1));
    const T xi = lamd * inv_dlam;

    const T fi = sl_traits<T>::floor_(xi);
    const T ax = xi - fi;
    int c0 = ((int)fi) % ni; if (c0 < 0) c0 += ni;
    int c1 = c0 + 1; if (c1 >= ni) c1 -= ni;
    const int half = ni >> 1;
    int d0 = c0 + half; if (d0 >= ni) d0 -= ni;
    int d1 = c1 + half; if (d1 >= ni) d1 -= ni;

    const int m0 = sl_bracket<T>(p, lat_ext, mlut, nb, nj, lut_scale);
    const T la = lat_ext[m0];
    const T lb = lat_ext[m0 + 1];
    const T ay = (p - la) / (lb - la);
    int r0 = rowoff[m0];
    int r1 = rowoff[m0 + 1];
    if (BAND && (r0 < 0 || r1 < 0)) {
        atomicOr(esc, 1);
        if (r0 < 0) r0 = 0;
        if (r1 < 0) r1 = 0;
    }
    const int s0 = rowshift[m0];
    const int s1 = rowshift[m0 + 1];
    const int i00 = r0 + (s0 ? d0 : c0), i01 = r0 + (s0 ? d1 : c1);
    const int i10 = r1 + (s1 ? d0 : c0), i11 = r1 + (s1 ? d1 : c1);

    int k0 = (int)sl_traits<T>::floor_(xk);
    k0 = min(max(k0, 0), nk - 2);
    const T az = xk - (T)k0;

    const long long plane = (long long)(BAND ? njs : nj) * (long long)ni;
    const long long o0 = (long long)k0 * plane;
    const long long o1 = o0 + plane;
    const T bx = (T)1 - ax, by = (T)1 - ay, bz = (T)1 - az;

    const T* fs[4] = {f0, f1, f2, f3};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const T* s = fs[q];
        const T t0 = by * (bx * s[o0 + i00] + ax * s[o0 + i01])
                   + ay * (bx * s[o0 + i10] + ax * s[o0 + i11]);
        const T t1 = by * (bx * s[o1 + i00] + ax * s[o1 + i01])
                   + ay * (bx * s[o1 + i10] + ax * s[o1 + i11]);
        out[q] = bz * t0 + az * t1;
    }
}

/* The spherical fixed-point departure-point search, all iterations in one
 * launch.  The wind arrives as geocentric Cartesian components of the
 * horizontal (tangent) velocity in m/s, and the vertical rate as level
 * indices per second at full levels.  The extrapolated fields are read at
 * the ARRIVAL point, so they need no interpolation at all.
 *
 * Writes the departure longitude and latitude in radians, the departure
 * continuous level index, and the size of the LAST iteration's move in
 * metres and in level indices, which is what the convergence gate reads.
 */
/* BAND = 1: the arrival points are the nja rows starting at grid row ja0,
   the advecting fields (vx, vy, vz, sv) hold the njs rows starting sja rows
   below the first arrival row, and the extrapolated fields and every output
   are arrival-shaped.  The arithmetic is the BAND = 0 arithmetic. */
template <typename T, int BAND>
__device__ __forceinline__ void sl_departure_body(
    const T* __restrict__ vx, const T* __restrict__ vy,
    const T* __restrict__ vz, const T* __restrict__ sv,
    const T* __restrict__ vxe, const T* __restrict__ vye,
    const T* __restrict__ vze, const T* __restrict__ sve,
    const T* __restrict__ lat_ext,
    const int* __restrict__ mlut, const int* __restrict__ rowoff,
    const int* __restrict__ rowshift,
    T* __restrict__ xi_d, T* __restrict__ phi_d, T* __restrict__ zk_d,
    T* __restrict__ move_m, T* __restrict__ move_k,
    int iters, T dt, T radius, T dlam,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,
    int njs, int nja, int ja0, int sja, int* __restrict__ esc)
{
    const int rows_out = BAND ? nja : nj;
    const long long n = (long long)nk * (long long)rows_out * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const long long plane = (long long)rows_out * (long long)ni;
    const int i = (int)(gid % ni);
    const int j = (int)((gid / ni) % rows_out);
    const int k = (int)(gid / plane);
    /* the arrival point's own index in the advecting fields */
    const long long here = BAND
        ? ((long long)k * (long long)njs + (long long)(j + sja)) * (long long)ni + i
        : gid;

    const T phia = lat_ext[j + (BAND ? ja0 : 0) + 2];
    const T lama = (T)i * dlam;
    T sph, cph, sla, cla;
    sl_traits<T>::sincos_(phia, &sph, &cph);
    sl_traits<T>::sincos_(lama, &sla, &cla);
    const T ax = cph * cla, ay = cph * sla, az = sph;

    const T ex = vxe[gid], ey = vye[gid], ez = vze[gid], es = sve[gid];
    const T inv_r = (T)1 / radius;

    /* first guess: one explicit backward step on the time-n wind at the
       arrival point, which is exactly what the arrival index holds */
    T qx = ax - dt * vx[here] * inv_r;
    T qy = ay - dt * vy[here] * inv_r;
    T qz = az - dt * vz[here] * inv_r;
    T rn = sl_traits<T>::rsqrt_(qx * qx + qy * qy + qz * qz);
    qx *= rn; qy *= rn; qz *= rn;
    /* The departure level is clamped at the lid and ground INTERFACES,
       index -1/2 and nk - 1/2, not at the outermost full levels: a parcel
       arriving at level 0 under descent departs from the half layer above
       it, and clamping it to level 0 made it read its own value, so the
       lid cooled under ascent and never warmed under descent (DYC-1).
       The wind in that half layer is read at the outermost full level
       (sl_trilinear4 keeps its own [0, nk-1] clamp). */
    const T klo = (T)-0.5, khi = (T)nk - (T)0.5;
    T qk = (T)k - dt * sv[here];
    qk = min(max(qk, klo), khi);

    T last_m = (T)0, last_k = (T)0;
    const T hdt = (T)0.5 * dt;
    for (int it = 0; it < iters; ++it) {
        const T lamd = sl_traits<T>::atan2_(qy, qx);
        const T phid = sl_latitude<T>(qx, qy, qz);
        T w[4];
        sl_trilinear4<T, BAND>(vx, vy, vz, sv, lamd, phid, qk, lat_ext, mlut,
                               rowoff, rowshift, nb, nk, nj, ni, inv_dlam,
                               lut_scale, w, njs, esc);
        T px = ax - hdt * (ex + w[0]) * inv_r;
        T py = ay - hdt * (ey + w[1]) * inv_r;
        T pz = az - hdt * (ez + w[2]) * inv_r;
        rn = sl_traits<T>::rsqrt_(px * px + py * py + pz * pz);
        px *= rn; py *= rn; pz *= rn;
        T pk = (T)k - hdt * (es + w[3]);
        pk = min(max(pk, klo), khi);

        const T cx = px - qx, cy = py - qy, cz = pz - qz;
        const T chord = sl_traits<T>::sqrt_(cx * cx + cy * cy + cz * cz);
        last_m = radius * (T)2 * sl_traits<T>::asin_(
            min(max((T)0.5 * chord, (T)0), (T)1));
        last_k = sl_traits<T>::abs_(pk - qk);
        qx = px; qy = py; qz = pz; qk = pk;
    }

    xi_d[gid] = sl_traits<T>::atan2_(qy, qx) * inv_dlam;
    phi_d[gid] = sl_latitude<T>(qx, qy, qz);
    zk_d[gid] = qk;
    move_m[gid] = last_m;
    move_k[gid] = last_k;
}

/* ------------------------------------------------------------------ */
/* The Newton form of the same search (semilag.trajectory_solver =      */
/* "newton").  The equation is the fixed-point search's own,            */
/*     r_D = normalize(r_A - (dt/2)[V_ex(r_A) + V(r_D)] / a),           */
/*     k_D = k_A - (dt/2)[s_ex + s(r_D, k_D)],                          */
/* read through the same trilinear gather, so the two solvers share one */
/* fixed point.  The fixed-point pass contracts by the spectral radius  */
/* of (dt/2) J, which the fold gate does not bound: det(I + (dt/2) J)   */
/* stays at one for a rotation or for the jet's vertical-shear coupling */
/* whatever their size, and the pass then converges slowly or not at    */
/* all.  Newton converges wherever det(I + (dt/2) J) is away from zero, */
/* which is what the gate does bound.  The Jacobian is the trilinear    */
/* interpolant's own derivative, taken in the local east / north / level */
/* frame at the current point; where its determinant is too small to    */
/* trust (or at the pole itself) the step falls back to the plain pass. */

template <typename T, int BAND>
__device__ __forceinline__ void sl_trilinear4_grad(
    const T* __restrict__ f0, const T* __restrict__ f1,
    const T* __restrict__ f2, const T* __restrict__ f3,
    T lamd, T phid, T xkd,
    const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    const int* __restrict__ rowoff, const int* __restrict__ rowshift,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,
    T* out, T* d_lam, T* d_phi, T* d_k, int njs, int* __restrict__ esc)
{
    const T p = min(max(phid, -(T)SL_HALF_PI_D), (T)SL_HALF_PI_D);
    const T xk = min(max(xkd, (T)0), (T)(nk - 1));
    const bool kfree = (xkd > (T)0) && (xkd < (T)(nk - 1));
    const T xi = lamd * inv_dlam;

    const T fi = sl_traits<T>::floor_(xi);
    const T ax = xi - fi;
    int c0 = ((int)fi) % ni; if (c0 < 0) c0 += ni;
    int c1 = c0 + 1; if (c1 >= ni) c1 -= ni;
    const int half = ni >> 1;
    int d0 = c0 + half; if (d0 >= ni) d0 -= ni;
    int d1 = c1 + half; if (d1 >= ni) d1 -= ni;

    const int m0 = sl_bracket<T>(p, lat_ext, mlut, nb, nj, lut_scale);
    const T la = lat_ext[m0];
    const T lb = lat_ext[m0 + 1];
    const T ay = (p - la) / (lb - la);
    const T inv_dphi = (T)1 / (lb - la);
    int r0 = rowoff[m0];
    int r1 = rowoff[m0 + 1];
    if (BAND && (r0 < 0 || r1 < 0)) {
        atomicOr(esc, 1);
        if (r0 < 0) r0 = 0;
        if (r1 < 0) r1 = 0;
    }
    const int s0 = rowshift[m0];
    const int s1 = rowshift[m0 + 1];
    const int i00 = r0 + (s0 ? d0 : c0), i01 = r0 + (s0 ? d1 : c1);
    const int i10 = r1 + (s1 ? d0 : c0), i11 = r1 + (s1 ? d1 : c1);

    int k0 = (int)sl_traits<T>::floor_(xk);
    k0 = min(max(k0, 0), nk - 2);
    const T az = xk - (T)k0;

    const long long plane = (long long)(BAND ? njs : nj) * (long long)ni;
    const long long o0 = (long long)k0 * plane;
    const long long o1 = o0 + plane;
    const T bx = (T)1 - ax, by = (T)1 - ay, bz = (T)1 - az;

    const T* fs[4] = {f0, f1, f2, f3};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const T* s = fs[q];
        const T a00 = s[o0 + i00], a01 = s[o0 + i01];
        const T a10 = s[o0 + i10], a11 = s[o0 + i11];
        const T b00 = s[o1 + i00], b01 = s[o1 + i01];
        const T b10 = s[o1 + i10], b11 = s[o1 + i11];
        const T lo0 = bx * a00 + ax * a01, hi0 = bx * a10 + ax * a11;
        const T lo1 = bx * b00 + ax * b01, hi1 = bx * b10 + ax * b11;
        const T t0 = by * lo0 + ay * hi0;
        const T t1 = by * lo1 + ay * hi1;
        out[q] = bz * t0 + az * t1;
        const T gx0 = by * (a01 - a00) + ay * (a11 - a10);
        const T gx1 = by * (b01 - b00) + ay * (b11 - b10);
        d_lam[q] = (bz * gx0 + az * gx1) * inv_dlam;
        d_phi[q] = (bz * (hi0 - lo0) + az * (hi1 - lo1)) * inv_dphi;
        d_k[q] = kfree ? (t1 - t0) : (T)0;
    }
}

template <typename T, int BAND>
__device__ __forceinline__ void sl_departure_newton_body(
    const T* __restrict__ vx, const T* __restrict__ vy,
    const T* __restrict__ vz, const T* __restrict__ sv,
    const T* __restrict__ vxe, const T* __restrict__ vye,
    const T* __restrict__ vze, const T* __restrict__ sve,
    const T* __restrict__ lat_ext,
    const int* __restrict__ mlut, const int* __restrict__ rowoff,
    const int* __restrict__ rowshift,
    T* __restrict__ xi_d, T* __restrict__ phi_d, T* __restrict__ zk_d,
    T* __restrict__ move_m, T* __restrict__ move_k,
    int iters, T dt, T radius, T dlam,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,
    int njs, int nja, int ja0, int sja, int* __restrict__ esc)
{
    const int rows_out = BAND ? nja : nj;
    const long long n = (long long)nk * (long long)rows_out * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const long long plane = (long long)rows_out * (long long)ni;
    const int i = (int)(gid % ni);
    const int j = (int)((gid / ni) % rows_out);
    const int k = (int)(gid / plane);
    const long long here = BAND
        ? ((long long)k * (long long)njs + (long long)(j + sja)) * (long long)ni + i
        : gid;

    const T phia = lat_ext[j + (BAND ? ja0 : 0) + 2];
    const T lama = (T)i * dlam;
    T sph, cph, sla, cla;
    sl_traits<T>::sincos_(phia, &sph, &cph);
    sl_traits<T>::sincos_(lama, &sla, &cla);
    const T ax = cph * cla, ay = cph * sla, az = sph;

    const T ex = vxe[gid], ey = vye[gid], ez = vze[gid], es = sve[gid];
    const T inv_r = (T)1 / radius;

    /* the fixed-point search's own first guess */
    T qx = ax - dt * vx[here] * inv_r;
    T qy = ay - dt * vy[here] * inv_r;
    T qz = az - dt * vz[here] * inv_r;
    T rn = sl_traits<T>::rsqrt_(qx * qx + qy * qy + qz * qz);
    qx *= rn; qy *= rn; qz *= rn;
    const T klo = (T)-0.5, khi = (T)nk - (T)0.5;
    T qk = (T)k - dt * sv[here];
    qk = min(max(qk, klo), khi);

    T last_m = (T)0, last_k = (T)0;
    T alpha = (T)1, res_prev = sl_traits<T>::big();
    const T hdt = (T)0.5 * dt;
    const T c = hdt * inv_r;
    for (int it = 0; it < iters; ++it) {
        const T lamd = sl_traits<T>::atan2_(qy, qx);
        const T phid = sl_latitude<T>(qx, qy, qz);
        T w[4], gl[4], gp[4], gk[4];
        sl_trilinear4_grad<T, BAND>(vx, vy, vz, sv, lamd, phid, qk, lat_ext,
                                    mlut, rowoff, rowshift, nb, nk, nj, ni,
                                    inv_dlam, lut_scale, w, gl, gp, gk, njs,
                                    esc);
        T ux = ax - c * (ex + w[0]);
        T uy = ay - c * (ey + w[1]);
        T uz = az - c * (ez + w[2]);
        const T un = sl_traits<T>::rsqrt_(ux * ux + uy * uy + uz * uz);
        const T px = ux * un, py = uy * un, pz = uz * un;
        T pk = (T)k - hdt * (es + w[3]);
        const bool kclamped = (pk <= klo) || (pk >= khi);
        pk = min(max(pk, klo), khi);

        /* the local frame at the current point */
        T sp, cp, sl, cl;
        sl_traits<T>::sincos_(phid, &sp, &cp);
        sl_traits<T>::sincos_(lamd, &sl, &cl);
        const T exx = -sl, exy = cl;
        const T nxx = -sp * cl, nxy = -sp * sl, nxz = cp;
        /* residual G(q) - q in that frame: radians east, north, levels */
        const T dx = px - qx, dy = py - qy, dz = pz - qz;
        const T re = exx * dx + exy * dy;
        const T rnn = nxx * dx + nxy * dy + nxz * dz;
        const T rk = pk - qk;

        T de = re, dn = rnn, dk = rk;
        /* The interpolant is piecewise trilinear, so its Jacobian jumps at
           cell faces and a Newton step can cycle between two cells around a
           solution that sits near a face.  Where the residual (in grid
           cells) did not fall by a tenth since the last pass the step is
           halved, down to an eighth; where Newton converges the residual
           falls far faster than that and the step is the full one. */
        {
            const T cells0 = (T)nj * (T)(0.5 / SL_HALF_PI_D);
            const T res = sl_traits<T>::sqrt_(
                (re * re + rnn * rnn) * cells0 * cells0 + rk * rk);
            if (it > 0 && res > (T)0.9 * res_prev)
                alpha = max(alpha * (T)0.5, (T)0.125);
            res_prev = res;
        }
        if (cp > (T)1e-6) {
            const T icp = (T)1 / cp;
            const T g = -c * un;
            /* M = dG / d(east, north, level) */
            const T m00 = g * (exx * gl[0] + exy * gl[1]) * icp;
            const T m01 = g * (exx * gp[0] + exy * gp[1]);
            const T m02 = g * (exx * gk[0] + exy * gk[1]);
            const T m10 = g * (nxx * gl[0] + nxy * gl[1] + nxz * gl[2]) * icp;
            const T m11 = g * (nxx * gp[0] + nxy * gp[1] + nxz * gp[2]);
            const T m12 = g * (nxx * gk[0] + nxy * gk[1] + nxz * gk[2]);
            const T m20 = kclamped ? (T)0 : -hdt * gl[3] * icp;
            const T m21 = kclamped ? (T)0 : -hdt * gp[3];
            const T m22 = kclamped ? (T)0 : -hdt * gk[3];
            /* (I - M) d = r, by Cramer */
            const T a00 = (T)1 - m00, a01 = -m01, a02 = -m02;
            const T a10 = -m10, a11 = (T)1 - m11, a12 = -m12;
            const T a20 = -m20, a21 = -m21, a22 = (T)1 - m22;
            const T c00 = a11 * a22 - a12 * a21;
            const T c01 = a12 * a20 - a10 * a22;
            const T c02 = a10 * a21 - a11 * a20;
            const T det = a00 * c00 + a01 * c01 + a02 * c02;
            if (isfinite(det) && det > (T)0.05) {
                const T idet = (T)1 / det;
                const T ne = (c00 * re
                              + (a02 * a21 - a01 * a22) * rnn
                              + (a01 * a12 - a02 * a11) * rk) * idet;
                const T nn = (c01 * re
                              + (a00 * a22 - a02 * a20) * rnn
                              + (a02 * a10 - a00 * a12) * rk) * idet;
                const T nk2 = (c02 * re
                               + (a01 * a20 - a00 * a21) * rnn
                               + (a00 * a11 - a01 * a10) * rk) * idet;
                /* a Newton step more than five times the plain one is a
                   linearization across a cell edge, not a solve: cap it */
                /* both moves in grid cells: horizontal in meridional
                   spacings (nj / pi per radian), vertical in levels */
                const T cells = (T)nj * (T)(0.5 / SL_HALF_PI_D);
                const T rr = sl_traits<T>::sqrt_(
                    (re * re + rnn * rnn) * cells * cells + rk * rk);
                const T nr = sl_traits<T>::sqrt_(
                    (ne * ne + nn * nn) * cells * cells + nk2 * nk2);
                const T s = nr / (rr + (T)1e-12);
                const T cap = (T)5;
                const T scale = s > cap ? cap / s : (T)1;
                if (isfinite(ne) && isfinite(nn) && isfinite(nk2)) {
                    de = ne * scale; dn = nn * scale; dk = nk2 * scale;
                }
            }
        }
        /* the convergence gate reads the UNDAMPED step: a halved step is
           not a converged one */
        const T full_m = radius * sl_traits<T>::sqrt_(de * de + dn * dn);
        const T full_k = sl_traits<T>::abs_(dk);
        de *= alpha; dn *= alpha; dk *= alpha;
        T nx = qx + de * exx + dn * nxx;
        T ny = qy + de * exy + dn * nxy;
        T nz = qz + dn * nxz;
        rn = sl_traits<T>::rsqrt_(nx * nx + ny * ny + nz * nz);
        nx *= rn; ny *= rn; nz *= rn;
        T nkk = min(max(qk + dk, klo), khi);

        const T cx = nx - qx, cy = ny - qy, cz = nz - qz;
        const T chord = sl_traits<T>::sqrt_(cx * cx + cy * cy + cz * cz);
        last_m = radius * (T)2 * sl_traits<T>::asin_(
            min(max((T)0.5 * chord, (T)0), (T)1));
        last_k = sl_traits<T>::abs_(nkk - qk);
        if (alpha < (T)1) {
            last_m = max(last_m, full_m);
            last_k = max(last_k, full_k);
        }
        qx = nx; qy = ny; qz = nz; qk = nkk;
    }

    xi_d[gid] = sl_traits<T>::atan2_(qy, qx) * inv_dlam;
    phi_d[gid] = sl_latitude<T>(qx, qy, qz);
    zk_d[gid] = qk;
    move_m[gid] = last_m;
    move_k[gid] = last_k;
}

#define SL_NEWTON_ENTRY(NAME, T, BAND)                                       \
extern "C" __global__ void NAME(                                             \
    const T* vx, const T* vy, const T* vz, const T* sv,                      \
    const T* vxe, const T* vye, const T* vze, const T* sve,                  \
    const T* lat_ext, const int* mlut,                                       \
    const int* rowoff, const int* rowshift,                                  \
    T* xi_d, T* phi_d, T* zk_d, T* move_m, T* move_k,                        \
    int iters, T dt, T radius, T dlam,                                       \
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,                 \
    int njs, int nja, int ja0, int sja, int* esc)                            \
{ sl_departure_newton_body<T, BAND>(vx, vy, vz, sv, vxe, vye, vze, sve,     \
    lat_ext, mlut, rowoff, rowshift, xi_d, phi_d, zk_d, move_m, move_k,      \
    iters, dt, radius, dlam, nb, nk, nj, ni, inv_dlam, lut_scale, njs, nja,  \
    ja0, sja, esc); }

SL_NEWTON_ENTRY(sl_departure_newton_f32, float, 0)
SL_NEWTON_ENTRY(sl_departure_newton_f64, double, 0)
SL_NEWTON_ENTRY(sl_departure_newton_band_f32, float, 1)
SL_NEWTON_ENTRY(sl_departure_newton_band_f64, double, 1)

/* ------------------------------------------------------------------ */
/* The globalized Newton search (semilag.trajectory_solver = "newton_ls"). */
/*
 * The same equation, the same trilinear gather and the same interpolant
 * Jacobian as sl_departure_newton_body.  What differs is how a step is
 * accepted and what the convergence gate reads.
 *
 * Why the plain Newton form stalls at a long step.  The interpolant is
 * piecewise trilinear, so the Jacobian it hands Newton is one-sided inside
 * each cell and jumps at every face.  Where (dt/2)|J| approaches one (the
 * regime a 450-600 s step puts a 13 km jet or a resolved updraft in) the
 * full Newton step from one cell lands in the next, whose own step points
 * back, and the point cycles.  The plain form answered with a damping
 * factor that only ever halves (to an eighth, never restored) and a gate
 * that reads the UNDAMPED step, so a cycling point was still "moving" by
 * most of a full step after eight iterations however close it sat to the
 * root.  The fixed-point pass contracts by the spectral radius of
 * (dt/2) J and does not contract at all past one.
 *
 * The cure is the textbook one for a piecewise-smooth system: a line
 * search on the residual.  Each iteration takes the Newton direction and
 * halves it (at most four times) until the residual |G(q) - q| measured
 * in grid cells falls (Armijo, sigma = 1e-4).  The Newton direction is a
 * descent direction of the residual inside the cell it was linearized in,
 * so a short enough step always falls; a cycle between two cells cannot
 * survive a strictly falling residual.  Where the 3x3 is singular or at
 * the pole the direction is the plain fixed-point one, as before.  The
 * trial point the line search accepts has already been evaluated, so an
 * accepted full step costs exactly one gather, as one Newton pass does.
 *
 * The gate reads the Newton step AT THE RETURNED POINT: the linearized
 * distance from the departure point this kernel returns to the root of
 * the equation it solves, (I - M)^-1 (G(q) - q), and the plain residual
 * wherever that system was not solved.  That is the error of the answer,
 * where the fixed-point pass's "last move" is the residual one iterate
 * earlier.  A point whose residual is below 1e-4 cells stops iterating.
 */
template <typename T, int BAND>
__device__ __forceinline__ void sl_newton_frame(
    T qx, T qy, T qz, T* exx, T* exy, T* nxx, T* nxy, T* nxz, T* cp_out)
{
    const T lamd = sl_traits<T>::atan2_(qy, qx);
    const T phid = sl_latitude<T>(qx, qy, qz);
    T sp, cp, sl, cl;
    sl_traits<T>::sincos_(phid, &sp, &cp);
    sl_traits<T>::sincos_(lamd, &sl, &cl);
    *exx = -sl; *exy = cl;
    *nxx = -sp * cl; *nxy = -sp * sl; *nxz = cp;
    *cp_out = cp;
}

/* One evaluation at q: the residual G(q) - q in q's local frame (radians
   east and north, levels), its size in grid cells, and the direction the
   next step takes (the Newton solve, or the residual itself where the
   system is not solved). */
template <typename T, int BAND>
__device__ __forceinline__ T sl_newton_ls_eval(
    const T* __restrict__ vx, const T* __restrict__ vy,
    const T* __restrict__ vz, const T* __restrict__ sv,
    const T* __restrict__ lat_ext, const int* __restrict__ mlut,
    const int* __restrict__ rowoff, const int* __restrict__ rowshift,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale, int njs,
    int* __restrict__ esc,
    T qx, T qy, T qz, T qk, T ax, T ay, T az, int k,
    T ex, T ey, T ez, T es, T c, T hdt, T klo, T khi, T cells,
    T* de, T* dn, T* dk, int* solved)
{
    const T lamd = sl_traits<T>::atan2_(qy, qx);
    const T phid = sl_latitude<T>(qx, qy, qz);
    T w[4], gl[4], gp[4], gk[4];
    sl_trilinear4_grad<T, BAND>(vx, vy, vz, sv, lamd, phid, qk, lat_ext,
                                mlut, rowoff, rowshift, nb, nk, nj, ni,
                                inv_dlam, lut_scale, w, gl, gp, gk, njs, esc);
    const T ux = ax - c * (ex + w[0]);
    const T uy = ay - c * (ey + w[1]);
    const T uz = az - c * (ez + w[2]);
    const T un = sl_traits<T>::rsqrt_(ux * ux + uy * uy + uz * uz);
    const T px = ux * un, py = uy * un, pz = uz * un;
    T pk = (T)k - hdt * (es + w[3]);
    const bool kclamped = (pk <= klo) || (pk >= khi);
    pk = min(max(pk, klo), khi);

    T exx, exy, nxx, nxy, nxz, cp;
    sl_newton_frame<T, BAND>(qx, qy, qz, &exx, &exy, &nxx, &nxy, &nxz, &cp);
    const T dx = px - qx, dy = py - qy, dz = pz - qz;
    const T re = exx * dx + exy * dy;
    const T rnn = nxx * dx + nxy * dy + nxz * dz;
    const T rk = pk - qk;
    *de = re; *dn = rnn; *dk = rk; *solved = 0;
    const T rr = sl_traits<T>::sqrt_(
        (re * re + rnn * rnn) * cells * cells + rk * rk);
    if (cp > (T)1e-6) {
        const T icp = (T)1 / cp;
        const T g = -c * un;
        const T m00 = g * (exx * gl[0] + exy * gl[1]) * icp;
        const T m01 = g * (exx * gp[0] + exy * gp[1]);
        const T m02 = g * (exx * gk[0] + exy * gk[1]);
        const T m10 = g * (nxx * gl[0] + nxy * gl[1] + nxz * gl[2]) * icp;
        const T m11 = g * (nxx * gp[0] + nxy * gp[1] + nxz * gp[2]);
        const T m12 = g * (nxx * gk[0] + nxy * gk[1] + nxz * gk[2]);
        const T m20 = kclamped ? (T)0 : -hdt * gl[3] * icp;
        const T m21 = kclamped ? (T)0 : -hdt * gp[3];
        const T m22 = kclamped ? (T)0 : -hdt * gk[3];
        const T a00 = (T)1 - m00, a01 = -m01, a02 = -m02;
        const T a10 = -m10, a11 = (T)1 - m11, a12 = -m12;
        const T a20 = -m20, a21 = -m21, a22 = (T)1 - m22;
        const T c00 = a11 * a22 - a12 * a21;
        const T c01 = a12 * a20 - a10 * a22;
        const T c02 = a10 * a21 - a11 * a20;
        const T det = a00 * c00 + a01 * c01 + a02 * c02;
        if (isfinite(det) && det > (T)0.05) {
            const T idet = (T)1 / det;
            const T ne = (c00 * re
                          + (a02 * a21 - a01 * a22) * rnn
                          + (a01 * a12 - a02 * a11) * rk) * idet;
            const T nn = (c01 * re
                          + (a00 * a22 - a02 * a20) * rnn
                          + (a02 * a10 - a00 * a12) * rk) * idet;
            const T nk2 = (c02 * re
                           + (a01 * a20 - a00 * a21) * rnn
                           + (a00 * a11 - a01 * a10) * rk) * idet;
            /* det > 0.05 bounds the Newton step at 20x the residual */
            const T nr = sl_traits<T>::sqrt_(
                (ne * ne + nn * nn) * cells * cells + nk2 * nk2);
            const T s = nr / (rr + (T)1e-30);
            const T cap = (T)20;
            const T scale = s > cap ? cap / s : (T)1;
            if (isfinite(ne) && isfinite(nn) && isfinite(nk2)) {
                *de = ne * scale; *dn = nn * scale; *dk = nk2 * scale;
                *solved = 1;
            }
        }
    }
    return rr;
}

template <typename T, int BAND>
__device__ __forceinline__ void sl_departure_newton_ls_body(
    const T* __restrict__ vx, const T* __restrict__ vy,
    const T* __restrict__ vz, const T* __restrict__ sv,
    const T* __restrict__ vxe, const T* __restrict__ vye,
    const T* __restrict__ vze, const T* __restrict__ sve,
    const T* __restrict__ lat_ext,
    const int* __restrict__ mlut, const int* __restrict__ rowoff,
    const int* __restrict__ rowshift,
    T* __restrict__ xi_d, T* __restrict__ phi_d, T* __restrict__ zk_d,
    T* __restrict__ move_m, T* __restrict__ move_k,
    int iters, T dt, T radius, T dlam,
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,
    int njs, int nja, int ja0, int sja, int* __restrict__ esc)
{
    const int rows_out = BAND ? nja : nj;
    const long long n = (long long)nk * (long long)rows_out * (long long)ni;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const long long plane = (long long)rows_out * (long long)ni;
    const int i = (int)(gid % ni);
    const int j = (int)((gid / ni) % rows_out);
    const int k = (int)(gid / plane);
    const long long here = BAND
        ? ((long long)k * (long long)njs + (long long)(j + sja)) * (long long)ni + i
        : gid;

    const T phia = lat_ext[j + (BAND ? ja0 : 0) + 2];
    const T lama = (T)i * dlam;
    T sph, cph, sla, cla;
    sl_traits<T>::sincos_(phia, &sph, &cph);
    sl_traits<T>::sincos_(lama, &sla, &cla);
    const T ax = cph * cla, ay = cph * sla, az = sph;

    const T ex = vxe[gid], ey = vye[gid], ez = vze[gid], es = sve[gid];
    const T inv_r = (T)1 / radius;

    /* the fixed-point search's own first guess */
    T qx = ax - dt * vx[here] * inv_r;
    T qy = ay - dt * vy[here] * inv_r;
    T qz = az - dt * vz[here] * inv_r;
    T rn = sl_traits<T>::rsqrt_(qx * qx + qy * qy + qz * qz);
    qx *= rn; qy *= rn; qz *= rn;
    const T klo = (T)-0.5, khi = (T)nk - (T)0.5;
    T qk = (T)k - dt * sv[here];
    qk = min(max(qk, klo), khi);

    const T hdt = (T)0.5 * dt;
    const T c = hdt * inv_r;
    const T cells = (T)nj * (T)(0.5 / SL_HALF_PI_D);
    T de, dn, dk;
    int solved;
    T res = sl_newton_ls_eval<T, BAND>(
        vx, vy, vz, sv, lat_ext, mlut, rowoff, rowshift, nb, nk, nj, ni,
        inv_dlam, lut_scale, njs, esc, qx, qy, qz, qk, ax, ay, az, k,
        ex, ey, ez, es, c, hdt, klo, khi, cells, &de, &dn, &dk, &solved);
    for (int it = 0; it < iters; ++it) {
        if (res < (T)1e-4) break;
        T exx, exy, nxx, nxy, nxz, cp;
        sl_newton_frame<T, BAND>(qx, qy, qz, &exx, &exy, &nxx, &nxy, &nxz,
                                 &cp);
        T alpha = (T)1;
        T tx, ty, tz, tk, tres, tde, tdn, tdk;
        int tsolved;
        for (int ls = 0; ; ++ls) {
            const T se = alpha * de, sn = alpha * dn, sk = alpha * dk;
            tx = qx + se * exx + sn * nxx;
            ty = qy + se * exy + sn * nxy;
            tz = qz + sn * nxz;
            rn = sl_traits<T>::rsqrt_(tx * tx + ty * ty + tz * tz);
            tx *= rn; ty *= rn; tz *= rn;
            tk = min(max(qk + sk, klo), khi);
            tres = sl_newton_ls_eval<T, BAND>(
                vx, vy, vz, sv, lat_ext, mlut, rowoff, rowshift, nb, nk, nj,
                ni, inv_dlam, lut_scale, njs, esc, tx, ty, tz, tk, ax, ay,
                az, k, ex, ey, ez, es, c, hdt, klo, khi, cells,
                &tde, &tdn, &tdk, &tsolved);
            if (tres <= ((T)1 - (T)1e-4 * alpha) * res || ls >= 4) break;
            alpha *= (T)0.5;
        }
        qx = tx; qy = ty; qz = tz; qk = tk; res = tres;
        de = tde; dn = tdn; dk = tdk; solved = tsolved;
    }

    xi_d[gid] = sl_traits<T>::atan2_(qy, qx) * inv_dlam;
    phi_d[gid] = sl_latitude<T>(qx, qy, qz);
    zk_d[gid] = qk;
    /* the distance from the returned point to the root, linearized there */
    move_m[gid] = radius * sl_traits<T>::sqrt_(de * de + dn * dn);
    move_k[gid] = sl_traits<T>::abs_(dk);
}

#define SL_NEWTON_LS_ENTRY(NAME, T, BAND)                                    \
extern "C" __global__ void NAME(                                             \
    const T* vx, const T* vy, const T* vz, const T* sv,                      \
    const T* vxe, const T* vye, const T* vze, const T* sve,                  \
    const T* lat_ext, const int* mlut,                                       \
    const int* rowoff, const int* rowshift,                                  \
    T* xi_d, T* phi_d, T* zk_d, T* move_m, T* move_k,                        \
    int iters, T dt, T radius, T dlam,                                       \
    int nb, int nk, int nj, int ni, T inv_dlam, T lut_scale,                 \
    int njs, int nja, int ja0, int sja, int* esc)                            \
{ sl_departure_newton_ls_body<T, BAND>(vx, vy, vz, sv, vxe, vye, vze, sve,  \
    lat_ext, mlut, rowoff, rowshift, xi_d, phi_d, zk_d, move_m, move_k,      \
    iters, dt, radius, dlam, nb, nk, nj, ni, inv_dlam, lut_scale, njs, nja,  \
    ja0, sja, esc); }

SL_NEWTON_LS_ENTRY(sl_departure_newton_ls_f32, float, 0)
SL_NEWTON_LS_ENTRY(sl_departure_newton_ls_f64, double, 0)
SL_NEWTON_LS_ENTRY(sl_departure_newton_ls_band_f32, float, 1)
SL_NEWTON_LS_ENTRY(sl_departure_newton_ls_band_f64, double, 1)

extern "C" __global__ void sl_departure_f32(
    const float* vx, const float* vy, const float* vz, const float* sv,
    const float* vxe, const float* vye, const float* vze, const float* sve,
    const float* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift,
    float* xi_d, float* phi_d, float* zk_d, float* move_m, float* move_k,
    int iters, float dt, float radius, float dlam,
    int nb, int nk, int nj, int ni, float inv_dlam, float lut_scale)
{ sl_departure_body<float, 0>(vx, vy, vz, sv, vxe, vye, vze, sve, lat_ext,
                           mlut, rowoff, rowshift, xi_d, phi_d, zk_d,
                           move_m, move_k, iters, dt, radius, dlam, nb, nk,
                           nj, ni, inv_dlam, lut_scale,
                           0, 0, 0, 0, (int*)0); }

extern "C" __global__ void sl_departure_f64(
    const double* vx, const double* vy, const double* vz, const double* sv,
    const double* vxe, const double* vye, const double* vze, const double* sve,
    const double* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift,
    double* xi_d, double* phi_d, double* zk_d, double* move_m, double* move_k,
    int iters, double dt, double radius, double dlam,
    int nb, int nk, int nj, int ni, double inv_dlam, double lut_scale)
{ sl_departure_body<double, 0>(vx, vy, vz, sv, vxe, vye, vze, sve, lat_ext,
                            mlut, rowoff, rowshift, xi_d, phi_d, zk_d,
                            move_m, move_k, iters, dt, radius, dlam, nb, nk,
                            nj, ni, inv_dlam, lut_scale,
                           0, 0, 0, 0, (int*)0); }

extern "C" __global__ void sl_departure_band_f32(
    const float* vx, const float* vy, const float* vz, const float* sv,
    const float* vxe, const float* vye, const float* vze, const float* sve,
    const float* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift,
    float* xi_d, float* phi_d, float* zk_d, float* move_m, float* move_k,
    int iters, float dt, float radius, float dlam,
    int nb, int nk, int nj, int ni, float inv_dlam, float lut_scale,
    int njs, int nja, int ja0, int sja, int* esc)
{ sl_departure_body<float, 1>(vx, vy, vz, sv, vxe, vye, vze, sve, lat_ext,
    mlut, rowoff, rowshift, xi_d, phi_d, zk_d, move_m, move_k, iters, dt,
    radius, dlam, nb, nk, nj, ni, inv_dlam, lut_scale, njs, nja, ja0, sja,
    esc); }

extern "C" __global__ void sl_departure_band_f64(
    const double* vx, const double* vy, const double* vz, const double* sv,
    const double* vxe, const double* vye, const double* vze, const double* sve,
    const double* lat_ext, const int* mlut,
    const int* rowoff, const int* rowshift,
    double* xi_d, double* phi_d, double* zk_d, double* move_m, double* move_k,
    int iters, double dt, double radius, double dlam,
    int nb, int nk, int nj, int ni, double inv_dlam, double lut_scale,
    int njs, int nja, int ja0, int sja, int* esc)
{ sl_departure_body<double, 1>(vx, vy, vz, sv, vxe, vye, vze, sve, lat_ext,
    mlut, rowoff, rowshift, xi_d, phi_d, zk_d, move_m, move_k, iters, dt,
    radius, dlam, nb, nk, nj, ni, inv_dlam, lut_scale, njs, nja, ja0, sja,
    esc); }
