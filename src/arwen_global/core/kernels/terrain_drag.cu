// Sub-grid terrain drag from WRF v4.7.1: the orographic drag scheme
// gwd_opt = 1, the one this model runs.
//
//   gwdo_column       gwd_opt = 1: phys/module_bl_gwdo.F (gwdo) calling
//                     phys/physics_mmm/bl_gwdo.F90 (bl_gwdo_run), the KIM
//                     orographic gravity-wave drag with flow blocking
//                     (Choi and Hong 2015).
//
// Carried from the engine's kernels/terrain_drag.cu (gpuwm 2.9.13), whose
// gwdo_column and wind-sector helpers below are its words unchanged.  The
// engine's file also holds topo_wind_static, terrain_pbl_top and
// gwdo_gsl_column (gwd_opt = 3, the GSL drag suite, which reads glibc_sinf);
// this package runs none of them and carries none of them, so the module
// assembles from glibc_flt32.cuh and this file alone.  The two schemes are
// WRF v4.6.1 and v4.7.1 code alike: the KIM scheme's two Fortran files are
// byte-identical in the two releases.
//
// Every statement follows WRF's own, in WRF's order of evaluation; the module
// is compiled with -fmad=false so no product is fused into a sum, as gfortran
// fuses none.  Fortran's real powers (x**2., x**(-1.2), ...) are glibc's
// powf (gfk_pow; gfortran calls powf for a real exponent even when it is
// 2.0), integer powers are products, expf and logf are glibc's (gfk_exp,
// gfk_log), sinf is glibc's (glibc_sinf).  atan2f is td_atan2f below, WOOF's
// own float32 atan2f, whose words are the reference platform's (Ubuntu 24.04,
// glibc 2.39, x86-64): see its comment.  The oracles that hold these kernels
// to WRF are tools/terrain_drag_wrf471_oracle
// (tests/test_terrain_drag_wrf471_parity.py) and, for gwd_opt = 1 on the
// reference platform, tools/gwdo_kim_wrf461_oracle
// (tests/test_gwdo_kim_wrf461_parity.py).
//
// Layout: 3-D fields are (nz, ny, nx) mass levels (p_interface nz+1), 2-D
// fields (ny, nx); the four directional statistics OA1-4 and OL1-4 are
// (4, ny, nx).  One thread owns one column.

#define TD_KMAX 128

__device__ __forceinline__ float td_max(float a, float b)
{ return fmaxf(a, b); }
__device__ __forceinline__ float td_min(float a, float b)
{ return fminf(a, b); }

// atan2f for the wind-sector choice, bl_gwdo.F90:338 and
// module_bl_gwdo_gsl.F's `wdir = atan2(ubar,vbar) + pi`: the angle only picks
// one of eight sectors (which OA/OL statistic and grid length the drag uses),
// so its last bit matters only on a sector boundary, and there it decides
// the whole column.
//
// WHY NOT atan2() IN DOUBLE.  In gpuwm 2.8.5 this read
// __double2float_rn(atan2((double)y, (double)x)), the correctly rounded
// angle, which is what glibc 2.43 returns (the WRF v4.7.1 oracle's
// platform).  The reference platform of this verification (Ubuntu 24.04,
// glibc 2.39; the MYNN, YSU and Kessler oracles' platform) returns the
// classic atanf-based atan2f instead, 1 ULP off the correctly rounded angle
// on 16% of random wind pairs (measured on x86-64), and on a sector
// boundary that 1 ULP picks the other sector: in the WRF v4.6.1 column
// oracle (tests/data/oracles/gwdo_kim) one sector-boundary column per grid
// took idir 1 under WRF and 8 under the double atan2 (50 levels), 5 and 4
// (90 levels), and every drag word of those columns differed (up to
// 0.41 m/s2).
//
// td_atanf and td_atan2f are WOOF's own float32 code: td_atanf is the same
// algorithm and the same words as the MYNN unit's mynn_glibc_atanf
// (kernels/mynn_pbl.cu), and td_atan2f reduces to it through one IEEE
// division y/x and a quadrant fix-up with float32 pi constants.  They are
// held to the reference platform's atanf over EVERY float32 and to its
// atan2f on boundary, special and random pairs by
// tools/gwdo_kim_wrf461_oracle/sector_sweep.py.  Every operation is one
// plain float32 operation; the module compiles with -fmad=false.
__constant__ unsigned int TD_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,   // atan hi parts
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,   // atan lo parts
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,   // odd series
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

__device__ float td_atanf(float x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int)hx;
    int id;
#define TD_ATAN_HI(i) __uint_as_float(TD_ATANF_TAB[(i)])
#define TD_ATAN_LO(i) __uint_as_float(TD_ATANF_TAB[4 + (i)])
#define TD_ATAN_T(i)  __uint_as_float(TD_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {                     // |x| >= 2**25
        if (ix > 0x7F800000u) return x + x;      // NaN
        if (signed_hx > 0) return TD_ATAN_HI(3) + TD_ATAN_LO(3);
        return -TD_ATAN_HI(3) - TD_ATAN_LO(3);
    }
    if (ix < 0x3EE00000u) {                      // |x| < 0.4375
        if (ix < 0x31000000u) return x;          // |x| < 2**-29
        id = -1;
    } else {
        x = __uint_as_float(ix);                 // |x|
        if (ix < 0x3F980000u) {                  // |x| < 1.1875
            if (ix < 0x3F300000u) {              // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0f * x - 1.0f) / (2.0f + x);
            } else {                             // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0f) / (x + 1.0f);
            }
        } else if (ix < 0x401C0000u) {           // |x| < 2.4375
            id = 2;
            x = (x - 1.5f) / (1.0f + 1.5f * x);
        } else {                                 // 2.4375 <= |x| < 2**25
            id = 3;
            x = -1.0f / x;
        }
    }
    float z = x * x;
    float w = z * z;
    float s1 = w * TD_ATAN_T(10);
    s1 = w * (TD_ATAN_T(8) + s1);
    s1 = w * (TD_ATAN_T(6) + s1);
    s1 = w * (TD_ATAN_T(4) + s1);
    s1 = w * (TD_ATAN_T(2) + s1);
    s1 = z * (TD_ATAN_T(0) + s1);
    float s2 = w * TD_ATAN_T(9);
    s2 = w * (TD_ATAN_T(7) + s2);
    s2 = w * (TD_ATAN_T(5) + s2);
    s2 = w * (TD_ATAN_T(3) + s2);
    s2 = w * (TD_ATAN_T(1) + s2);
    float s = s1 + s2;
    if (id < 0) return x - x * s;
    float r = TD_ATAN_HI(id) - ((x * s - TD_ATAN_LO(id)) - x);
    return (signed_hx < 0) ? -r : r;
#undef TD_ATAN_HI
#undef TD_ATAN_LO
#undef TD_ATAN_T
}

__device__ float td_atan2f(float y, float x)
{
    const float tiny = 1.0e-30f;
    const float pi_o_4 = __uint_as_float(0x3F490FDBu);
    const float pi_o_2 = __uint_as_float(0x3FC90FDBu);
    const float pi = __uint_as_float(0x40490FDBu);
    const float pi_lo = __uint_as_float(0xB3BBBD2Eu);   // pi - float(pi)
    int hx = __float_as_int(x), hy = __float_as_int(y);
    int ix = hx & 0x7FFFFFFF, iy = hy & 0x7FFFFFFF;
    if (ix > 0x7F800000 || iy > 0x7F800000) return x + y;   // NaN
    if (hx == 0x3F800000) return td_atanf(y);               // x = 1
    int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);            // 2*sign(x)+sign(y)
    if (iy == 0) {                                          // y = +-0
        if (m <= 1) return y;
        return (m == 2) ? pi + tiny : -pi - tiny;
    }
    if (ix == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    if (ix == 0x7F800000) {                                 // x = +-Inf
        if (iy == 0x7F800000) {
            switch (m) {
            case 0: return pi_o_4 + tiny;
            case 1: return -pi_o_4 - tiny;
            case 2: return 3.0f * pi_o_4 + tiny;
            default: return -3.0f * pi_o_4 - tiny;
            }
        }
        switch (m) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return pi + tiny;
        default: return -pi - tiny;
        }
    }
    if (iy == 0x7F800000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    int k = (iy - ix) >> 23;
    float z;
    if (k > 60) z = pi_o_2 + 0.5f * pi_lo;                  // |y/x| > 2**60
    else if (hx < 0 && k < -60) z = 0.0f;                   // |y|/x < -2**60
    else z = td_atanf(fabsf(y / x));
    switch (m) {
    case 0: return z;
    case 1: return -z;
    case 2: return pi - (z - pi_lo);
    default: return (z - pi_lo) - pi;
    }
}

// Fortran NINT: round half away from zero.
__device__ __forceinline__ int td_nint(float x) { return (int)roundf(x); }

// --------------------------------------------------------------------------
// gwd_opt = 1: bl_gwdo_run (phys/physics_mmm/bl_gwdo.F90), through WRF's
// wrapper module_bl_gwdo.F::gwdo (rublten += rotated drag).
// --------------------------------------------------------------------------
// The scheme's per-level arrays are recomputed from the inputs where it reads
// them rather than stored: every recomputation is the same statement on the
// same words, so the value is the one WRF stored.  usqj below the blending
// height and the averaged usqj(1) are written by WRF and never read again, so
// they are not formed.
extern "C" __global__
void gwdo_column(const real *u, const real *v, const real *t,
                 const real *qv, const real *p, const real *p_interface,
                 const real *pi, const real *z,
                 const real *var, const real *oc1, const real *oa4,
                 const real *ol4, const real *sina, const real *cosa,
                 real *du, real *dv,
                 real *dtaux3d, real *dtauy3d, real *dusfcg, real *dvsfcg,
                 real dxmeter, real deltim, real g_, real cp_, real rd_,
                 real fv_, real pi_, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > TD_KMAX) return;
    const real ric = 0.25f, dw2min = 1.0f, rimin = -100.0f, bnv2min = 1.0e-5f;
    const real efmin = 0.0f, efmax = 10.0f, gmax = 1.0f;
    const real veleps = 1.0f, frc = 1.0f, ce = 0.8f, cg = 0.5f;
    const int kpblmin = 2;
    const real frmax = 10.0f, olmin = 1.0e-5f, odmin = 0.1f, odmax = 10.0f;
    const int kte = nz;                    // Fortran kts = 1 .. kte = nz
    const int mdir = 8;
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    real ca = cosa[col], sa = sina[col];

    // Fortran-indexed accessors (k = 1..nz).
    #define Q(k) ((size_t)((k) - 1) * st + col)
    #define U1(k) (u[Q(k)] * ca - v[Q(k)] * sa)
    #define V1(k) (u[Q(k)] * sa + v[Q(k)] * ca)
    #define VTJ(k) (t[Q(k)] * (1.0f + fv_ * qv[Q(k)]))
    #define VTK(k) (VTJ(k) / pi[Q(k)])
    #define RHO(k) (1.0f / rd_ * p[Q(k)] / VTJ(k))
    #define PRSI(k) (p_interface[(size_t)((k) - 1) * st + col])
    #define PRSL(k) (p[Q(k)])
    #define DEL(k) (PRSI(k) - PRSI((k) + 1))
    #define ZL(k) (z[Q(k)])

    const real fdir = (real)mdir / (2.0f * pi_);
    real delx = dxmeter, dely = dxmeter;
    real dxy4[4], dxy4p[4];
    dxy4[0] = delx;
    dxy4[1] = dely;
    dxy4[2] = sqrtf(gfk_pow(delx, 2.0f) + gfk_pow(dely, 2.0f));
    dxy4[3] = dxy4[2];
    dxy4p[0] = dxy4[1];
    dxy4p[1] = dxy4[0];
    dxy4p[2] = dxy4[3];
    dxy4p[3] = dxy4[2];
    real cleff = dxmeter;
    real varc = var[col];

    // klowtop: first level whose height above the lowest reaches 2*var.
    real zlowtop = 2.0f * varc;
    int klowtop = 0;
    if (zlowtop > 0.0f) {
        for (int k = 2; k <= kte; ++k) {
            if (ZL(k) - ZL(1) >= zlowtop) { klowtop = k + 1; break; }
        }
    }
    const int kpblmax = kte;
    int kbl = klowtop;
    kbl = max(min(kbl, kpblmax), kpblmin);
    real delks = 1.0f / (PRSI(1) - PRSI(kbl));
    real delks1 = 1.0f / (PRSL(1) - PRSL(kbl));
    real ubar = 0.0f, vbar = 0.0f, rhobar = 0.0f;
    for (int k = 1; k <= kpblmax; ++k) {
        if (k < kbl) {
            real rcsks = DEL(k) * delks;
            real rdelks = DEL(k) * delks;
            ubar = ubar + rcsks * U1(k);
            vbar = vbar + rcsks * V1(k);
            rhobar = rhobar + rdelks * RHO(k);
        }
    }
    // wind direction sector
    real oa, ol, olp, od, dxy, dxyp;
    {
        real oa4c[4], ol4c[4];
        for (int m = 0; m < 4; ++m) {
            oa4c[m] = oa4[(size_t)m * st + col];
            ol4c[m] = ol4[(size_t)m * st + col];
        }
        real wdir = td_atan2f(ubar, vbar) + pi_;
        int idir = td_nint(fdir * wdir) % mdir + 1;
        int nwd = nwdir[idir - 1];
        int m = (nwd - 1) % 4;
        oa = (real)(1 - 2 * ((nwd - 1) / 4)) * oa4c[m];
        ol = ol4c[m];
        real ol4p[4] = {ol4c[1], ol4c[0], ol4c[3], ol4c[2]};
        olp = ol4p[m];
        od = olp / td_max(ol, olmin);
        od = td_min(od, odmax);
        od = td_max(od, odmin);
        dxy = dxy4[m];
        dxyp = dxy4p[m];
    }

    // usqj(k), bnv2(k) for k = 1..kte-1 (bnv2 unclamped in this scheme);
    // bnv2(kte) keeps its initial 0.
    #define TI(k) (2.0f / (t[Q(k)] + t[Q((k) + 1)]))
    #define RDZ(k) (1.0f / (ZL((k) + 1) - ZL(k)))
    auto usqj_at = [&](int k) -> real {
        real ti = TI(k);
        real rdz = RDZ(k);
        real tem1 = U1(k) - U1(k + 1);
        real tem2 = V1(k) - V1(k + 1);
        real dw2 = tem1 * tem1 + tem2 * tem2;
        real shr2 = td_max(dw2, dw2min) * rdz * rdz;
        real bvf2 = g_ * (g_ / cp_ + rdz * (VTJ(k + 1) - VTJ(k))) * ti;
        return td_max(bvf2 / shr2, rimin);
    };
    auto bnv2_at = [&](int k) -> real {
        if (k >= kte) return 0.0f;
        real rdz = RDZ(k);
        return 2.0f * g_ * rdz * (VTK(k + 1) - VTK(k))
               / (VTK(k + 1) + VTK(k));
    };

    real ulow = td_max(sqrtf(ubar * ubar + vbar * vbar), 1.0f);
    real rulow = 1.0f / ulow;
    auto velco_at = [&](int k) -> real {
        real vc = 0.5f * ((U1(k) + U1(k + 1)) * ubar
                          + (V1(k) + V1(k + 1)) * vbar);
        vc = vc * rulow;
        if (vc < veleps && vc > 0.0f) vc = veleps;
        return vc;
    };

    bool ldrag = velco_at(1) <= 0.0f;
    for (int k = kpblmin; k <= kpblmax; ++k)
        if (k < kbl) ldrag = ldrag || velco_at(k) <= 0.0f;
    real wtkbj = (PRSL(1) - PRSL(2)) * delks1;
    real bnv2_1 = wtkbj * bnv2_at(1);
    for (int k = kpblmin; k <= kpblmax; ++k) {
        if (k < kbl) {
            real rdelks = (PRSL(k) - PRSL(k + 1)) * delks1;
            bnv2_1 = bnv2_1 + bnv2_at(k) * rdelks;
        }
    }
    ldrag = ldrag || bnv2_1 <= 0.0f;
    ldrag = ldrag || ulow == 1.0f;
    ldrag = ldrag || varc <= 0.0f;

    real bnv = 0.0f, fr = 0.0f, xn = 0.0f, yn = 0.0f;
    if (!ldrag) {
        bnv = sqrtf(bnv2_1);
        fr = bnv * rulow * varc * od;
        fr = td_min(fr, frmax);
        xn = ubar * rulow;
        yn = vbar * rulow;
    }
    real taub, coefm = 0.0f;
    if (!ldrag) {
        real efact = gfk_pow(oa + 2.0f, ce * fr / frc);
        efact = td_min(td_max(efact, efmin), efmax);
        coefm = gfk_pow(1.0f + ol, oa + 1.0f);
        real xlinv = coefm / cleff;
        real tem = fr * fr * oc1[col];
        real gfobnv = gmax * tem / ((tem + cg) * bnv);
        taub = xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact;
    } else {
        taub = 0.0f;
        xn = 0.0f;
        yn = 0.0f;
    }

    // Vertical structure of the stress.
    real taup[TD_KMAX + 1];
    for (int k = 1; k <= kte + 1; ++k) taup[k - 1] = 0.0f;
    for (int k = 1; k <= kpblmax; ++k)
        if (k <= kbl) taup[k - 1] = taub;
    bool icrilv = false;
    real brvf = 0.0f;
    for (int k = kpblmin; k <= kte - 1; ++k) {
        int kp1 = k + 1;
        if (k >= kbl) {
            real usqj = usqj_at(k);
            icrilv = icrilv || (usqj < ric) || (velco_at(k) <= 0.0f);
            brvf = td_max(bnv2_at(k), bnv2min);
            brvf = sqrtf(brvf);
        }
        if (k >= kbl && !ldrag) {
            if (!icrilv && taup[k - 1] > 0.0f) {
                real velco = velco_at(k);
                real usqj = usqj_at(k);
                real temv = 1.0f / velco;
                real tem1 = coefm / dxy * (RHO(kp1) + RHO(k)) * brvf * velco
                            * 0.5f;
                real hd = sqrtf(taup[k - 1] / tem1);
                real fro = brvf * hd * temv;
                real tem2 = sqrtf(usqj);
                real tem = 1.0f + tem2 * fro;
                real rim = usqj * (1.0f - fro) / (tem * tem);
                if (rim <= ric) {
                    if ((oa <= 0.0f) || (kp1 >= kpblmin)) {
                        real temc = 2.0f + 1.0f / tem2;
                        hd = velco * (2.0f * sqrtf(temc) - temc) / brvf;
                        taup[kp1 - 1] = tem1 * hd * hd;
                    }
                } else {
                    taup[kp1 - 1] = taup[k - 1];
                }
            }
        }
    }
    // Flow blocking, summed into the stress.
    if (!ldrag) {
        int kblk = 0;
        real fbdpe = 0.0f, fbdke = 0.0f, zblk = 0.0f;
        for (int k = kte; k >= kpblmin; --k) {
            if (kblk == 0 && k <= kbl) {
                fbdpe = fbdpe + bnv2_at(k) * (ZL(kbl) - ZL(k)) * DEL(k) / g_
                        / RHO(k);
                fbdke = 0.5f * (gfk_pow(U1(k), 2.0f) + gfk_pow(V1(k), 2.0f));
                if (fbdpe >= fbdke) {
                    kblk = k;
                    kblk = min(kblk, kbl);
                    zblk = ZL(kblk) - ZL(1);
                }
            }
        }
        if (kblk != 0) {
            real fbdcd = td_max(2.0f - 1.0f / od, 0.0f);
            real taufb = 0.5f * rhobar * coefm / (dxmeter * dxmeter) * fbdcd
                         * dxyp * olp * zblk * (ulow * ulow);
            real tautem = taufb / (real)(kblk - 1);
            taup[0] = taup[0] + taufb;
            for (int k = 2; k <= kblk; ++k) {
                taufb = taufb - tautem;
                taup[k - 1] = taup[k - 1] + taufb;
            }
            // taufb is 0 above kblk: taup there gains + 0.
            for (int k = kblk + 1; k <= kte + 1; ++k)
                taup[k - 1] = taup[k - 1] + 0.0f;
        }
    }
    // -(g) d(tau)/dp and the critical-line limiter.
    auto taud_at = [&](int k) -> real {
        return 1.0f * (taup[k] - taup[k - 1]) * g_ / DEL(k);
    };
    real dtfac = 1.0f;
    for (int k = 1; k <= kpblmax - 1; ++k) {
        if (k <= kbl) {
            real td = taud_at(k);
            if (td != 0.0f)
                dtfac = td_min(dtfac, fabsf(velco_at(k) / (deltim * td)));
        }
    }
    real dusfc = 0.0f, dvsfc = 0.0f;
    for (int k = 1; k <= kte; ++k) {
        real td = taud_at(k) * dtfac;
        real dtaux = td * xn;
        real dtauy = td * yn;
        dusfc = dusfc + dtaux * DEL(k);
        dvsfc = dvsfc + dtauy * DEL(k);
        size_t q = Q(k);
        // module_bl_gwdo.F hands bl_gwdo_run rublten and copies it back;
        // bl_gwdo.F90 rotates the drag to the model grid and adds it.
        du[q] = du[q] + dtaux * ca + dtauy * sa;
        dv[q] = dv[q] - dtaux * sa + dtauy * ca;
        if (dtaux3d != nullptr) {
            dtaux3d[q] = dtaux * ca + dtauy * sa;
            dtauy3d[q] = -dtaux * sa + dtauy * ca;
        }
    }
    dusfc = (-1.0f / g_) * dusfc;
    dvsfc = (-1.0f / g_) * dvsfc;
    if (dusfcg != nullptr) {
        dusfcg[col] = dusfc * ca + dvsfc * sa;
        dvsfcg[col] = -dusfc * sa + dvsfc * ca;
    }
    #undef Q
    #undef U1
    #undef V1
    #undef VTJ
    #undef VTK
    #undef RHO
    #undef PRSI
    #undef PRSL
    #undef DEL
    #undef ZL
    #undef TI
    #undef RDZ
}
