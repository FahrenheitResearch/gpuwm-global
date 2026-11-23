"""Fast FP32 GPU port of the CCPP GFS SA-TKE-EDMF PBL scheme (satmedmf).

Eliminates the three bottlenecks of the FP64 reference path on consumer and
Workstation GPUs (e.g. RTX PRO 6000 Blackwell sm_120, 1:64 FP64:FP32 ratio):
  1. Executes column physics in native FP32 (with optional FP64 Thomas
     tridiagonal solves for strict conditioning).
  2. Uses block-coalesced SoA workspace layout (stride = blockDim.x = 64)
     instead of strided per-column AoS workspace, turning uncoalesced 224 KB
     strided loads into 100% coalesced 128-byte warp transactions.
  3. Reads/writes ArWen Global C-contiguous (nz, nlat, nlon) float32 tensors
     directly on device with zero float64 conversions, transposes, or packing.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

# cupy is imported where it is used, not at module scope: it is an optional
# extra of this package (gpu-cu12 / gpu-cu13), and a module-scope import made
# `import arwen_global.physics.native_runtime` need it on a CPU install
# (tests/test_packaging_declaration.py).
import numpy as np

from arwen_global.constants import DRY_AIR_CP, DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2

_SM_BLOCK = 64
_WORKSPACE_CACHE: dict[tuple[int, int, int], cp.ndarray] = {}


def _load_column_source_fp32(tridi_fp64: bool = False) -> str:
    """Transform the pinned CCPP satmedmf_column.cuh into FP32 CUDA source."""
    import gpuwm.core.kernels as gk

    kdir = Path(gk.__file__).resolve().parent
    raw = (kdir / "satmedmf_column.cuh").read_text(encoding="utf-8")

    # Separate tridiagonal solvers (lines before sm_mfpbltq) if hybrid FP64 tridiagonal requested
    if tridi_fp64:
        marker = "__device__ void sm_mfpbltq("
        idx_decl = raw.find(marker)
        idx_def = raw.find(marker, idx_decl + 1)
        if idx_def < 0:
            raise RuntimeError("Could not locate sm_mfpbltq definition in satmedmf_column.cuh")
        col_body = raw[idx_def:]
    else:
        col_body = raw

    # Suffix all hex float literals with 'f' so no implicit FP64 promotions occur.
    qs_expr = ("sm_max<float>(qmin,sm_div((float)(sm_mul((float)(eps),(float)(es))),"
               "(float)(sm_add((float)(plyr(i,k)),(float)(sm_mul((float)(epsm1),(float)(es)))))))")

    def _to_fp32_text(text: str) -> str:
        text = re.sub(r"\bdouble\b", "float", text)
        text = re.sub(r"(0x[0-9a-fA-F.]+p[+-]?[0-9]+)(?![0-9fFlL])", r"\1f", text)
        text = text.replace(qs_expr, "sm_qs_safe(eps, es, plyr(i,k), epsm1, qmin)")
        return text

    # The stratopause guard is a TEXT substitution into the engine's pinned
    # transcription.  If the engine's file stops spelling the expression this
    # way the substitution silently does nothing and float32 divides by
    # plyr + epsm1*es == 0 near 2.6 hPa / 275 K (the NaN the guard exists
    # for), so a file without it is refused by name.
    if re.sub(r"\bdouble\b", "float", col_body).count(qs_expr) < 1:
        raise RuntimeError(
            "satmedmf_column.cuh no longer carries the saturation-humidity "
            "expression the float32 port guards (sm_qs_safe); the installed "
            "gpuwm's satmedmf transcription moved, and the port must be "
            "re-derived against it")

    body_fp32 = _to_fp32_text(col_body)

    if tridi_fp64:
        # Provide FP32-interface, FP64-internal Thomas tridiagonal solvers (tiny: only 3 calls/column)
        tridi_src = r"""
__device__ void sm_tridi1(int l, int n, float* raw_cl, float* raw_cm, float* raw_cu, float* raw_r1, float* raw_au, float* raw_a1, SMWorkspace& ws) {
  SMFrame frame(ws);
  SMView<float> cl(raw_cl, l, n - 1, 1, 1, 2, 1);
  SMView<float> cm(raw_cm, l, n, 1, 1, 1, 1);
  SMView<float> cu(raw_cu, l, n - 1, 1, 1, 1, 1);
  SMView<float> r1(raw_r1, l, n, 1, 1, 1, 1);
  SMView<float> au(raw_au, l, n - 1, 1, 1, 1, 1);
  SMView<float> a1(raw_a1, l, n, 1, 1, 1, 1);
  double fk = 1.0 / (double)cm(1, 1);
  double au_prev = fk * (double)cu(1, 1);
  double a1_prev = fk * (double)r1(1, 1);
  au(1, 1) = (float)au_prev;
  a1(1, 1) = (float)a1_prev;
  for (int k = 2; k <= n - 1; ++k) {
    double cl_k = (double)cl(1, k);
    fk = 1.0 / ((double)cm(1, k) - cl_k * au_prev);
    au_prev = fk * (double)cu(1, k);
    a1_prev = fk * ((double)r1(1, k) - cl_k * a1_prev);
    au(1, k) = (float)au_prev;
    a1(1, k) = (float)a1_prev;
  }
  {
    double cl_n = (double)cl(1, n);
    fk = 1.0 / ((double)cm(1, n) - cl_n * au_prev);
    a1_prev = fk * ((double)r1(1, n) - cl_n * a1_prev);
    a1(1, n) = (float)a1_prev;
  }
  for (int k = n - 1; k >= 1; --k) {
    a1_prev = (double)a1(1, k) - (double)au(1, k) * a1_prev;
    a1(1, k) = (float)a1_prev;
  }
}

__device__ void sm_tridi2(int l, int n, float* raw_cl, float* raw_cm, float* raw_cu, float* raw_r1, float* raw_r2, float* raw_au, float* raw_a1, float* raw_a2, SMWorkspace& ws) {
  SMFrame frame(ws);
  SMView<float> cl(raw_cl, l, n - 1, 1, 1, 2, 1);
  SMView<float> cm(raw_cm, l, n, 1, 1, 1, 1);
  SMView<float> cu(raw_cu, l, n - 1, 1, 1, 1, 1);
  SMView<float> r1(raw_r1, l, n, 1, 1, 1, 1);
  SMView<float> r2(raw_r2, l, n, 1, 1, 1, 1);
  SMView<float> au(raw_au, l, n - 1, 1, 1, 1, 1);
  SMView<float> a1(raw_a1, l, n, 1, 1, 1, 1);
  SMView<float> a2(raw_a2, l, n, 1, 1, 1, 1);
  double fk = 1.0 / (double)cm(1, 1);
  double au_prev = fk * (double)cu(1, 1);
  double a1_prev = fk * (double)r1(1, 1);
  double a2_prev = fk * (double)r2(1, 1);
  au(1, 1) = (float)au_prev;
  a1(1, 1) = (float)a1_prev;
  a2(1, 1) = (float)a2_prev;
  for (int k = 2; k <= n - 1; ++k) {
    double cl_k = (double)cl(1, k);
    fk = 1.0 / ((double)cm(1, k) - cl_k * au_prev);
    au_prev = fk * (double)cu(1, k);
    a1_prev = fk * ((double)r1(1, k) - cl_k * a1_prev);
    a2_prev = fk * ((double)r2(1, k) - cl_k * a2_prev);
    au(1, k) = (float)au_prev;
    a1(1, k) = (float)a1_prev;
    a2(1, k) = (float)a2_prev;
  }
  {
    double cl_n = (double)cl(1, n);
    fk = 1.0 / ((double)cm(1, n) - cl_n * au_prev);
    a1_prev = fk * ((double)r1(1, n) - cl_n * a1_prev);
    a2_prev = fk * ((double)r2(1, n) - cl_n * a2_prev);
    a1(1, n) = (float)a1_prev;
    a2(1, n) = (float)a2_prev;
  }
  for (int k = n - 1; k >= 1; --k) {
    double auk = (double)au(1, k);
    a1_prev = (double)a1(1, k) - auk * a1_prev;
    a2_prev = (double)a2(1, k) - auk * a2_prev;
    a1(1, k) = (float)a1_prev;
    a2(1, k) = (float)a2_prev;
  }
}

__device__ void sm_tridin(int l, int n, int nt, float* raw_cl, float* raw_cm, float* raw_cu, float* raw_r1, float* raw_r2, float* raw_au, float* raw_a1, float* raw_a2, SMWorkspace& ws) {
  SMFrame frame(ws);
  SMView<float> cl(raw_cl, l, n - 1, 1, 1, 2, 1);
  SMView<float> cm(raw_cm, l, n, 1, 1, 1, 1);
  SMView<float> cu(raw_cu, l, n - 1, 1, 1, 1, 1);
  SMView<float> r1(raw_r1, l, n, 1, 1, 1, 1);
  SMView<float> r2(raw_r2, l, n * nt, 1, 1, 1, 1);
  SMView<float> au(raw_au, l, n - 1, 1, 1, 1, 1);
  SMView<float> a1(raw_a1, l, n, 1, 1, 1, 1);
  SMView<float> a2(raw_a2, l, n * nt, 1, 1, 1, 1);
  SMView<float> fkk(ws.alloc<float>(n - 2), l, n - 2, 1, 1, 2, 1);
  double fk1 = 1.0 / (double)cm(1, 1);
  double au_prev = fk1 * (double)cu(1, 1);
  double a1_prev = fk1 * (double)r1(1, 1);
  au(1, 1) = (float)au_prev;
  a1(1, 1) = (float)a1_prev;
  for (int k = 2; k <= n - 1; ++k) {
    double cl_k = (double)cl(1, k);
    double fk = 1.0 / ((double)cm(1, k) - cl_k * au_prev);
    fkk(1, k) = (float)fk;
    au_prev = fk * (double)cu(1, k);
    a1_prev = fk * ((double)r1(1, k) - cl_k * a1_prev);
    au(1, k) = (float)au_prev;
    a1(1, k) = (float)a1_prev;
  }
  double fkn = 1.0 / ((double)cm(1, n) - (double)cl(1, n) * au_prev);
  a1_prev = fkn * ((double)r1(1, n) - (double)cl(1, n) * a1_prev);
  a1(1, n) = (float)a1_prev;
  for (int k = n - 1; k >= 1; --k) {
    a1_prev = (double)a1(1, k) - (double)au(1, k) * a1_prev;
    a1(1, k) = (float)a1_prev;
  }
  for (int kk = 1; kk <= nt; ++kk) {
    int is = (kk - 1) * n;
    double prev = fk1 * (double)r2(1, 1 + is);
    a2(1, 1 + is) = (float)prev;
    for (int k = 2; k <= n - 1; ++k) {
      prev = (double)fkk(1, k) * ((double)r2(1, k + is) - (double)cl(1, k) * prev);
      a2(1, k + is) = (float)prev;
    }
    prev = fkn * ((double)r2(1, n + is) - (double)cl(1, n) * prev);
    a2(1, n + is) = (float)prev;
    for (int k = n - 1; k >= 1; --k) {
      prev = (double)a2(1, k + is) - (double)au(1, k) * prev;
      a2(1, k + is) = (float)prev;
    }
  }
}

__device__ void sm_tridit(int l, int n, int nt, float* raw_cl, float* raw_cm, float* raw_cu, float* raw_rt, float* raw_au, float* raw_at, SMWorkspace& ws) {
  SMFrame frame(ws);
  SMView<float> cl(raw_cl, l, n - 1, 1, 1, 2, 1);
  SMView<float> cm(raw_cm, l, n, 1, 1, 1, 1);
  SMView<float> cu(raw_cu, l, n - 1, 1, 1, 1, 1);
  SMView<float> rt(raw_rt, l, n * nt, 1, 1, 1, 1);
  SMView<float> au(raw_au, l, n - 1, 1, 1, 1, 1);
  SMView<float> at(raw_at, l, n * nt, 1, 1, 1, 1);
  SMView<float> fkk(ws.alloc<float>(n - 2), l, n - 2, 1, 1, 2, 1);
  double fk1 = 1.0 / (double)cm(1, 1);
  double au_prev = fk1 * (double)cu(1, 1);
  au(1, 1) = (float)au_prev;
  for (int k = 2; k <= n - 1; ++k) {
    double cl_k = (double)cl(1, k);
    double fk = 1.0 / ((double)cm(1, k) - cl_k * au_prev);
    fkk(1, k) = (float)fk;
    au_prev = fk * (double)cu(1, k);
    au(1, k) = (float)au_prev;
  }
  double fkn = 1.0 / ((double)cm(1, n) - (double)cl(1, n) * au_prev);
  for (int kk = 1; kk <= nt; ++kk) {
    int is = (kk - 1) * n;
    double prev = fk1 * (double)rt(1, 1 + is);
    at(1, 1 + is) = (float)prev;
    for (int k = 2; k <= n - 1; ++k) {
      prev = (double)fkk(1, k) * ((double)rt(1, k + is) - (double)cl(1, k) * prev);
      at(1, k + is) = (float)prev;
    }
    prev = fkn * ((double)rt(1, n + is) - (double)cl(1, n) * prev);
    at(1, n + is) = (float)prev;
    for (int k = n - 1; k >= 1; --k) {
      prev = (double)at(1, k + is) - (double)au(1, k) * prev;
      at(1, k + is) = (float)prev;
    }
  }
}
"""
        decl_prefix = _to_fp32_text(raw[:idx_decl])
        # Add forward decls for mfpbltq, mfscuq, satmedmfvdifq_run from top of file
        decls_end = raw.find("__device__ void sm_tridi1(", raw.find("__device__ void sm_tridi1(") + 1)
        decls_fp32 = _to_fp32_text(raw[:decls_end])
        return decls_fp32 + "\n" + tridi_src + "\n" + body_fp32

    return body_fp32


def _build_kernel_source(tridi_fp64: bool = False) -> str:
    col_fp32 = _load_column_source_fp32(tridi_fp64=tridi_fp64)
    header = f"""
typedef unsigned int uint32_t;
static constexpr int SM_STRIDE = {_SM_BLOCK};

__device__ __forceinline__ float sm_add(float a, float b) {{ return __fadd_rn(a, b); }}
__device__ __forceinline__ float sm_sub(float a, float b) {{ return __fsub_rn(a, b); }}
__device__ __forceinline__ float sm_mul(float a, float b) {{ return __fmul_rn(a, b); }}
__device__ __forceinline__ float sm_div(float a, float b) {{ return __fdiv_rn(a, b); }}
template<class T> __device__ __forceinline__ T sm_max(T a, T b) {{ return (a > b || a != a) ? a : b; }}
template<class T> __device__ __forceinline__ T sm_min(T a, T b) {{ return (a < b || a != a) ? a : b; }}
__device__ __forceinline__ float sm_abs(float a) {{ return fabsf(a); }}
__device__ __forceinline__ float sm_sqrt(float a) {{ return __fsqrt_rn(a); }}
__device__ __forceinline__ float sm_exp(float a) {{ return expf(a); }}
__device__ __forceinline__ float sm_log(float a) {{ return logf(a); }}
__device__ __forceinline__ float sm_pow(float a, float b) {{ return powf(a, b); }}
template<class T> __device__ __forceinline__ T sm_powi(T x, int m) {{
  unsigned int n = m < 0 ? (unsigned)(-m) : (unsigned)m;
  T y = (n & 1u) ? x : (T)1;
  while (n >>= 1) {{ x = sm_mul(x, x); if (n & 1u) y = sm_mul(y, x); }}
  return m < 0 ? sm_div((T)1, y) : y;
}}

static constexpr float sm_g     = 9.80665f;
static constexpr float sm_rd    = 287.05f;
static constexpr float sm_cp    = 1004.6f;
static constexpr float sm_rv    = 461.5f;
static constexpr float sm_hvap  = 2500000.0f;
static constexpr float sm_hfus  = 333580.0f;
static constexpr float sm_eps   = 287.05f / 461.5f;
static constexpr float sm_epsm1 = (287.05f / 461.5f) - 1.0f;
static constexpr float sm_fvirt = (461.5f / 287.05f) - 1.0f;

__device__ __forceinline__ float sm_fpvs(float t, const float* table) {{
  float xj = sm_min(sm_max(sm_add(-8999.0f, sm_mul(50.0f, t)), 1.0f), 7501.0f);
  int jx = (int)sm_min(xj, 7500.0f);
  return sm_add(table[jx - 1], sm_mul(sm_sub(xj, (float)jx), sm_sub(table[jx], table[jx - 1])));
}}

__device__ __forceinline__ float sm_qs_safe(float eps, float es, float plyr, float epsm1, float qmin) {{
  float denom = sm_add(plyr, sm_mul(epsm1, es));
  if (denom <= 1.0e-5f * plyr) return qmin;
  return sm_min(sm_max(qmin, sm_div(sm_mul(eps, es), denom)), 1.0e4f);
}}

struct SMWorkspace {{
  uint32_t* base;
  const float* table;
  int offset = 0;
  int capacity;
  int highwater = 0;
  __device__ __forceinline__ SMWorkspace(uint32_t* p, const float* t, int n)
      : base(p), table(t), capacity(n) {{}}
  template<class T> __device__ __forceinline__ T* alloc(int n) {{
    T* p = reinterpret_cast<T*>(base + (size_t)offset * SM_STRIDE);
    offset += n;
    if (offset > highwater) highwater = offset;
    return p;
  }}
}};

struct SMFrame {{
  SMWorkspace& ws;
  int mark;
  __device__ __forceinline__ SMFrame(SMWorkspace& w) : ws(w), mark(w.offset) {{}}
  __device__ __forceinline__ ~SMFrame() {{ ws.offset = mark; }}
}};

template<class T> struct SMView {{
  T* p;
  int n1, n2, n3, l1, l2, l3;
  __device__ __forceinline__ SMView(T* q, int a, int b=1, int c=1, int x=1, int y=1, int z=1)
      : p(q), n1(a), n2(b), n3(c), l1(x), l2(y), l3(z) {{}}
  __device__ __forceinline__ T& operator()(int i, int j=1, int k=1) {{
    size_t idx = (size_t)((i - l1) + n1 * ((j - l2) + n2 * (k - l3)));
    return *reinterpret_cast<T*>(reinterpret_cast<uint32_t*>(p) + idx * SM_STRIDE);
  }}
}};

template<class T> __device__ __forceinline__ T& sm_slot(T* p, int idx) {{
  return *reinterpret_cast<T*>(reinterpret_cast<uint32_t*>(p) + (size_t)idx * SM_STRIDE);
}}
"""

    entry = r"""
extern "C" __global__ void satmedmf_fp32_direct(
    int col_offset, int nc, int n_total, int km, int capacity,
    float dt, float dx_m, float dry_rd, float dry_kappa, float g0,
    const float* __restrict__ u_in,
    const float* __restrict__ v_in,
    const float* __restrict__ theta_in,
    const float* __restrict__ qv_in,
    const float* __restrict__ qc_in,
    const float* __restrict__ qr_in,
    const float* __restrict__ qi_in,
    const float* __restrict__ p_full_in,
    const float* __restrict__ p_half_in,
    const float* __restrict__ exner_in,
    const float* __restrict__ dz_in,
    const float* __restrict__ rad_sw_in,
    const float* __restrict__ rad_lw_in,
    const float* __restrict__ tke_in,
    const float* __restrict__ znt_in,
    const float* __restrict__ vegfra_in,
    const float* __restrict__ xland_in,
    const float* __restrict__ br_in,
    const float* __restrict__ tsk_in,
    const float* __restrict__ hfx_in,
    const float* __restrict__ qfx_in,
    const float* __restrict__ ust_in,
    const float* __restrict__ wspd_in,
    const float* __restrict__ u10_in,
    const float* __restrict__ v10_in,
    const float* __restrict__ fm_in,
    const float* __restrict__ fh_in,
    float* __restrict__ du_out,
    float* __restrict__ dv_out,
    float* __restrict__ dtheta_out,
    float* __restrict__ dqv_out,
    float* __restrict__ dqc_out,
    float* __restrict__ dqi_out,
    float* __restrict__ tke_out,
    float* __restrict__ hpbl_out,
    uint32_t* __restrict__ scratch,
    const float* __restrict__ table,
    int* __restrict__ max_highwater,
    int* __restrict__ err_flag
) {
  int local_c = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_c >= nc) return;
  int col = col_offset + local_c;
  const int ntrac = 5;

  uint32_t* slab = scratch + (size_t)blockIdx.x * SM_STRIDE * (size_t)capacity + (size_t)threadIdx.x;
  SMWorkspace ws(slab, table, capacity);

  float* u1    = ws.alloc<float>(km);
  float* v1    = ws.alloc<float>(km);
  float* t1    = ws.alloc<float>(km);
  float* q1    = ws.alloc<float>(ntrac * km);
  float* swh   = ws.alloc<float>(km);
  float* hlw   = ws.alloc<float>(km);
  float* prsi  = ws.alloc<float>(km + 1);
  float* prsl  = ws.alloc<float>(km);
  float* prslk = ws.alloc<float>(km);
  float* phii  = ws.alloc<float>(km + 1);
  float* phil  = ws.alloc<float>(km);

  float zi_cur = 0.0f;
  sm_slot(phii, 0) = 0.0f;
  for (int j = 0; j < km; ++j) {
    size_t idx3 = (size_t)j * (size_t)n_total + (size_t)col;
    float ex = exner_in[idx3];
    float rv = qv_in[idx3];
    float inv_den = 1.0f / (1.0f + rv);
    sm_slot(u1, j) = u_in[idx3];
    sm_slot(v1, j) = v_in[idx3];
    sm_slot(t1, j) = theta_in[idx3] * ex;
    sm_slot(q1, j + 0 * km) = rv * inv_den;
    sm_slot(q1, j + 1 * km) = qc_in[idx3] * inv_den;
    sm_slot(q1, j + 2 * km) = qr_in[idx3] * inv_den;
    sm_slot(q1, j + 3 * km) = qi_in[idx3] * inv_den;
    sm_slot(q1, j + 4 * km) = fmaxf(tke_in[idx3], 1.0e-6f);
    sm_slot(swh, j) = rad_sw_in[idx3] * ex;
    sm_slot(hlw, j) = rad_lw_in[idx3] * ex;
    sm_slot(prsi, j) = p_half_in[idx3];
    sm_slot(prsl, j) = p_full_in[idx3];
    sm_slot(prslk, j) = ex;
    float dzj = dz_in[idx3];
    float zi_next = zi_cur + dzj;
    sm_slot(phii, j + 1) = zi_next * g0;
    sm_slot(phil, j) = 0.5f * (zi_cur + zi_next) * g0;
    zi_cur = zi_next;
  }
  sm_slot(prsi, km) = p_half_in[(size_t)km * (size_t)n_total + (size_t)col];

  float p0 = sm_slot(prsl, 0);
  float pi0 = sm_slot(prsi, 0);
  float t0 = sm_slot(t1, 0);
  float rho0 = p0 / (dry_rd * fmaxf(t0, 100.0f));
  float rough = znt_in[col];
  float green = vegfra_in[col] * 0.01f;
  float rf = fminf(fmaxf((rough - 0.1f) / 0.9f, 0.0f), 1.0f);
  float zv = (xland_in[col] < 1.5f) ? sqrtf(rf * fmaxf(green, 0.1f)) : 0.0f;
  float ustv = ust_in[col];

  float* psk    = ws.alloc<float>(1); sm_slot(psk, 0)    = powf(pi0 / 100000.0f, dry_kappa);
  float* rbsoil = ws.alloc<float>(1); sm_slot(rbsoil, 0) = br_in[col];
  float* zorl   = ws.alloc<float>(1); sm_slot(zorl, 0)   = 100.0f * rough;
  float* tsea   = ws.alloc<float>(1); sm_slot(tsea, 0)   = tsk_in[col];
  float* heat   = ws.alloc<float>(1); sm_slot(heat, 0)   = hfx_in[col] / (rho0 * 1004.6f);
  float* evap   = ws.alloc<float>(1); sm_slot(evap, 0)   = qfx_in[col] / rho0;
  float* stress = ws.alloc<float>(1); sm_slot(stress, 0) = ustv * ustv;
  float* spd1   = ws.alloc<float>(1); sm_slot(spd1, 0)   = wspd_in[col];
  float* u10m   = ws.alloc<float>(1); sm_slot(u10m, 0)   = u10_in[col];
  float* v10m   = ws.alloc<float>(1); sm_slot(v10m, 0)   = v10_in[col];
  float* fm     = ws.alloc<float>(1); sm_slot(fm, 0)     = fm_in[col];
  float* fh     = ws.alloc<float>(1); sm_slot(fh, 0)     = fh_in[col];
  float* xmu    = ws.alloc<float>(1); sm_slot(xmu, 0)    = 1.0f;
  float* garea  = ws.alloc<float>(1); sm_slot(garea, 0)  = dx_m * dx_m;
  float* zvfun  = ws.alloc<float>(1); sm_slot(zvfun, 0)  = zv;
  float* sigmaf = ws.alloc<float>(1); sm_slot(sigmaf, 0) = green;

  float* du    = ws.alloc<float>(km);
  float* dv    = ws.alloc<float>(km);
  float* tdt   = ws.alloc<float>(km);
  float* rtg   = ws.alloc<float>(ntrac * km);
  float* hpbl  = ws.alloc<float>(1); sm_slot(hpbl, 0) = 0.0f;
  float* dkt   = ws.alloc<float>(km);
  float* dku   = ws.alloc<float>(km);
  float* del   = ws.alloc<float>(km);
  for (int j = 0; j < km; ++j) {
    sm_slot(du, j) = 0.0f;
    sm_slot(dv, j) = 0.0f;
    sm_slot(tdt, j) = 0.0f;
    sm_slot(dkt, j) = 0.0f;
    sm_slot(dku, j) = 0.0f;
    sm_slot(del, j) = sm_sub(sm_slot(prsi, j), sm_slot(prsi, j + 1));
  }
  for (int j = 0; j < ntrac * km; ++j) sm_slot(rtg, j) = 0.0f;

  float* usfco = ws.alloc<float>(1); sm_slot(usfco, 0) = 0.0f;
  float* vsfco = ws.alloc<float>(1); sm_slot(vsfco, 0) = 0.0f;
  float* dusfc = ws.alloc<float>(1); sm_slot(dusfc, 0) = 0.0f;
  float* dvsfc = ws.alloc<float>(1); sm_slot(dvsfc, 0) = 0.0f;
  float* dtsfc = ws.alloc<float>(1); sm_slot(dtsfc, 0) = 0.0f;
  float* dqsfc = ws.alloc<float>(1); sm_slot(dqsfc, 0) = 0.0f;
  int* kpbl    = ws.alloc<int>(1);   sm_slot(kpbl, 0) = 0;
  int* kinver  = ws.alloc<int>(1);   sm_slot(kinver, 0) = km;
  int* dtidx   = ws.alloc<int>(100 + ntrac);
  for (int j = 0; j < 100 + ntrac; ++j) sm_slot(dtidx, j) = 0;
  float* dtend = ws.alloc<float>(ntrac * km);
  for (int j = 0; j < ntrac * km; ++j) sm_slot(dtend, j) = 0.0f;

  char errmsg[1];
  int errflg = 0;
  sm_satmedmfvdifq_run(
      1, km, ntrac, 2, 3, 4, ntrac,
      sm_g, sm_rd, sm_cp, sm_rv, sm_hvap, sm_hfus, sm_fvirt, sm_eps, sm_epsm1,
      dv, du, tdt, rtg, u1, v1, t1, q1,
      usfco, vsfco, 0, swh, hlw, xmu, garea, zvfun, sigmaf,
      psk, rbsoil, zorl, u10m, v10m, fm, fh, tsea, heat, evap, stress, spd1,
      kpbl, prsi, del, prsl, prslk, phii, phil, dt, false,
      dusfc, dvsfc, dtsfc, dqsfc, hpbl, dkt, dku, kinver,
      1.0f, 0.01f, 0.2f, 1.0f, 0.13f, 0.13f, 300.0f, 300.0f,
      1, 0, 1, dtend, dtidx, 1, 2, 3, 1, false, false, errmsg, errflg, ws
  );

  // Recover physics rates directly into (nz, n) C-contiguous float32 outputs
  float inv_dt = 1.0f / dt;
  for (int j = 0; j < km; ++j) {
    size_t idx3 = (size_t)j * (size_t)n_total + (size_t)col;
    du_out[idx3] = sm_slot(du, j);
    dv_out[idx3] = sm_slot(dv, j);
    dtheta_out[idx3] = sm_slot(tdt, j) / sm_slot(prslk, j);

    float old_qv = sm_slot(q1, j + 0 * km);
    float new_qv = old_qv + dt * sm_slot(rtg, j + 0 * km);
    float den_old = 1.0f - old_qv;
    float den_new = 1.0f - new_qv;
    if (den_new <= 0.0f) {
      atomicExch(err_flag, 1);
      den_new = 1.0e-6f;
    }
    float inv_do = 1.0f / den_old;
    float inv_dn = 1.0f / den_new;

    dqv_out[idx3] = (new_qv * inv_dn - old_qv * inv_do) * inv_dt;

    float old_qc = sm_slot(q1, j + 1 * km);
    float new_qc = old_qc + dt * sm_slot(rtg, j + 1 * km);
    dqc_out[idx3] = (new_qc * inv_dn - old_qc * inv_do) * inv_dt;

    float old_qi = sm_slot(q1, j + 3 * km);
    float new_qi = old_qi + dt * sm_slot(rtg, j + 3 * km);
    dqi_out[idx3] = (new_qi * inv_dn - old_qi * inv_do) * inv_dt;

    float tke_new = tke_in[idx3] + dt * sm_slot(rtg, j + 4 * km);
    tke_out[idx3] = fminf(fmaxf(tke_new, 1.0e-6f), 50.0f);
  }
  hpbl_out[col] = sm_slot(hpbl, 0);
  if (ws.highwater > capacity) {
    atomicMax(max_highwater, ws.highwater);
  }
}
"""
    return header + "\n" + col_fp32 + "\n" + entry


@functools.lru_cache(maxsize=4)
def _get_module(tridi_fp64: bool = False) -> cp.RawModule:
    import cupy as cp

    src = _build_kernel_source(tridi_fp64=tridi_fp64)
    return cp.RawModule(
        code=src,
        options=("-std=c++17", "--prec-div=true", "--prec-sqrt=true", "--maxrregcount=128"),
        backend="nvrtc",
    )


@functools.lru_cache(maxsize=2)
def _fpvs_table_f32() -> cp.ndarray:
    import cupy as cp
    import gpuwm.core.satmedmf as satmedmf_ref
    return cp.ascontiguousarray(satmedmf_ref._table().astype(cp.float32))


def _capacity_for_levels(nz: int) -> int:
    # Exact highwater measured at nz=80, ntrac=5 is 10003 (~125*nz); 140*nz + 512
    # provides +17% headroom while trimming 28% of per-column workspace VRAM.
    return 140 * nz + 512


def _get_workspace(device_id: int, n_blocks: int, capacity: int) -> cp.ndarray:
    import cupy as cp

    key = (device_id, n_blocks, capacity)
    buf = _WORKSPACE_CACHE.get(key)
    needed = n_blocks * _SM_BLOCK * capacity
    if buf is None or buf.size < needed:
        _WORKSPACE_CACHE.clear()
        buf = cp.zeros(needed, dtype=cp.uint32)
        _WORKSPACE_CACHE[key] = buf
    return buf


def run_satmedmf_fp32(
    u, v, theta, qv, qc, qr, qi,
    p_full, p_half, exner, dz,
    rad_sw, rad_lw, tke_edmf,
    znt, vegfra, xland, br, tsk, hfx, qfx, ust, wspd, u10, v10, fm, fh,
    *,
    dx_m: float,
    dt_s: float,
    tridi_fp64: bool = True,
    chunk_cols: int | None = None,
) -> tuple[dict[str, cp.ndarray], cp.ndarray]:
    """Run direct FP32 GPU SA-TKE-EDMF PBL on (nz, nlat, nlon) float32 tensors."""
    import cupy as cp

    nz, nlat, nlon = u.shape
    n = nlat * nlon
    f32 = lambda a: cp.ascontiguousarray(a, dtype=cp.float32)

    u_c = f32(u); v_c = f32(v); th_c = f32(theta)
    qv_c = f32(qv); qc_c = f32(qc); qr_c = f32(qr); qi_c = f32(qi)
    pf_c = f32(p_full); ph_c = f32(p_half); ex_c = f32(exner); dz_c = f32(dz)
    sw_c = f32(rad_sw); lw_c = f32(rad_lw); tke_c = f32(tke_edmf)

    znt_c = f32(znt); veg_c = f32(vegfra); xl_c = f32(xland); br_c = f32(br)
    tsk_c = f32(tsk); hfx_c = f32(hfx); qfx_c = f32(qfx); ust_c = f32(ust)
    wspd_c = f32(wspd); u10_c = f32(u10); v10_c = f32(v10); fm_c = f32(fm); fh_c = f32(fh)

    du_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    dv_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    dth_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    dqv_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    dqc_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    dqi_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    tke_out = cp.empty((nz, nlat, nlon), dtype=cp.float32)
    hpbl_out = cp.empty((nlat, nlon), dtype=cp.float32)

    mod = _get_module(tridi_fp64=tridi_fp64)
    kern = mod.get_function("satmedmf_fp32_direct")
    table = _fpvs_table_f32()
    capacity = _capacity_for_levels(nz)

    import os
    if chunk_cols is None:
        chunk_cols = int(os.environ.get("ARWEN_SATMEDMF_CHUNK_COLS", "65536"))
    max_chunk = min(n, int(chunk_cols))
    n_blocks_max = (max_chunk + _SM_BLOCK - 1) // _SM_BLOCK
    dev_id = cp.cuda.Device().id
    scratch = _get_workspace(dev_id, n_blocks_max, capacity)
    status = cp.zeros(2, dtype=cp.int32)
    max_hw = status[:1]
    err_flag = status[1:]

    dry_kappa = float(DRY_AIR_GAS_CONSTANT / DRY_AIR_CP)

    for col0 in range(0, n, max_chunk):
        nc = min(max_chunk, n - col0)
        n_blocks = (nc + _SM_BLOCK - 1) // _SM_BLOCK
        kern(
            (n_blocks,),
            (_SM_BLOCK,),
            (
                np.int32(col0), np.int32(nc), np.int32(n), np.int32(nz), np.int32(capacity),
                np.float32(dt_s), np.float32(dx_m),
                np.float32(DRY_AIR_GAS_CONSTANT), np.float32(dry_kappa), np.float32(GRAVITY_M_S2),
                u_c, v_c, th_c, qv_c, qc_c, qr_c, qi_c,
                pf_c, ph_c, ex_c, dz_c, sw_c, lw_c, tke_c,
                znt_c, veg_c, xl_c, br_c, tsk_c, hfx_c, qfx_c, ust_c, wspd_c, u10_c, v10_c, fm_c, fh_c,
                du_out, dv_out, dth_out, dqv_out, dqc_out, dqi_out, tke_out, hpbl_out,
                scratch, table, max_hw, err_flag,
            ),
        )

    st_h = status.get()
    if int(st_h[0]) > capacity:
        raise RuntimeError(f"satmedmf_fp32 workspace overflow: highwater={int(st_h[0])} > capacity={capacity}")
    if int(st_h[1]) != 0:
        raise FloatingPointError("satmedmf_fp32 humidity increment reached unit specific humidity")

    return (
        {
            "du": du_out,
            "dv": dv_out,
            "dtheta": dth_out,
            "dqv": dqv_out,
            "dqc": dqc_out,
            "dqi": dqi_out,
            "hpbl": hpbl_out,
        },
        tke_out,
    )
