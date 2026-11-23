"""GPU non-orographic gravity-wave drag, a simplified spectral scheme after
Scinocca (2003) and Warner and McIntyre (1999), with the launch flux and
latitude shaping of Orr et al. (2010).

Written from the papers below, not ported from any model's code (the IFS and
CMAM implementations are not under open licences and are not to be read to
port it; licences.md).  It is not those schemes and not bitwise to either: its
spectrum is a smooth plateau with a c^-2 tail on sixteen log-spaced phase
speeds, not Scinocca's, and it carries no structure of the IFS routine.  It launches an
isotropic 4-azimuth (E, W, N, S) pseudomomentum flux spectrum at ``launch_pa``
(~450 hPa) in each column, propagates it upward with critical-level absorption
and saturation, and deposits the momentum convergence into the horizontal wind
tendencies (plus optional kinetic energy dissipation heating).

References: Scinocca, J. F. (2003), J. Atmos. Sci. 60, 667-682; Warner, C. D.
and McIntyre, M. E. (1999), Earth Planets Space 51, 475-484; Orr, A. et al.
(2010), J. Climate 23, 5905-5926.

All array inputs follow ``arwen_global.physics.native_batch`` bottom-to-top FP32
layout: ``(nz, nlat, nlon)`` with ``k = 0`` lowest full level and ``k = nz - 1``
top full level.

Performance architecture (CUDA direct FP32 column kernel):
  On CuPy / CUDA backends with FP32 arrays, the entire vertical spectral sweep
  (Brunt-Vaisala N(z), 4-azimuth x 16-phase-speed propagation, critical-level
  absorption, nonlinear saturation, momentum flux convergence, tendency clamping,
  and KE dissipation heating) runs in a single coalesced FP32 CUDA kernel (one
  thread per column, zero 3-D/4-D temporary allocations, zero per-step host
  synchronisations).  Each thread finds its own column's launch level, so the
  result does not depend on how the globe is split into latitude bands or
  cards.  ``ARWEN_NOGWD_CUDA=0`` selects the reference vectorised CuPy/NumPy
  path (a named referee, held to the kernel by
  tests/test_arwen_global_nogwd_cuda.py).
"""
from __future__ import annotations

import os
import numpy as np

from ..constants import DRY_AIR_CP, DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2

#: Characteristic horizontal wavenumber k* = 2*pi / 300 km (m^-1) for
#: Warner-McIntyre / Scinocca spectral saturation density.
NOGWD_OMEGA_STAR_S_INV = 1.5e-4
#: Intrinsic phase-speed spectral peak c* (m/s), Orr et al. (2010).
NOGWD_CSTAR_M_S = 8.0
#: Number of discrete log-spaced intrinsic phase speeds per azimuth.
NOGWD_NC = 16
#: Minimum and maximum launch intrinsic phase speeds (m/s).
NOGWD_CMIN_M_S = 0.5
NOGWD_CMAX_M_S = 50.0
#: Maximum permitted wind tendency magnitude (m/s^2) ~ 250 m/s/day (IFS safety clamp).
NOGWD_MAX_TENDENCY_M_S2 = 250.0 / 86400.0


def _phase_speed_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-spaced intrinsic launch phase speeds c_0, widths dc, and unit-area weights."""
    edges = np.geomspace(NOGWD_CMIN_M_S, NOGWD_CMAX_M_S, NOGWD_NC + 1, dtype=np.float64)
    c0 = 0.5 * (edges[:-1] + edges[1:])
    dc = edges[1:] - edges[:-1]
    # Scinocca (2003) / Orr et al. (2010) s=1, p=1 launch spectrum:
    # F_0(c) ~ 1 / (1 + (c / c*)^2)  (smooth low-c plateau, c^-2 high-c tail)
    raw = 1.0 / (1.0 + (c0 / NOGWD_CSTAR_M_S) ** 2)
    weights = raw / float(np.sum(raw * dc))  # integral(weights * dc) == 1.0
    return c0.astype(np.float32), dc.astype(np.float32), weights.astype(np.float32)


_C0_HOST, _DC_HOST, _W_HOST = _phase_speed_grid()

_NOGWD_CUDA_SOURCE = r"""
#define NOGWD_NC 16
#define DEG2RAD_F 0.017453292519943295f

extern "C" __global__ void scinocca_nogwd_fp32_direct(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ theta,
    const float* __restrict__ p_full,
    const float* __restrict__ exner,
    const float* __restrict__ dz,
    const float* __restrict__ lat_deg,
    const float* __restrict__ c0_arr,
    const float* __restrict__ dc_arr,
    const float* __restrict__ w0_arr,
    float launch_pa,
    float* __restrict__ du_out,
    float* __restrict__ dv_out,
    float* __restrict__ dth_out,
    int nz,
    int nlat,
    int nlon,
    int lat_mode,          // 0=uniform/none, 1=1D(nlat), 2=2D(nlat,nlon)
    int heat_booking,
    float launch_flux_pa,
    float sat_coeff,
    float max_tend,
    float rd,
    float cp,
    float grav
) {
    int ncol = nlat * nlon;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;

    // The launch level is the column's own: the highest full level still at or
    // below launch_pa (pressure falls with k).  It depends on this column
    // alone, so a run split into latitude bands or across cards launches
    // every column where a one-band run does.
    size_t stride0 = (size_t)ncol;
    int k_best = -1;
    for (int k = 0; k < nz; ++k) {
        if (p_full[(size_t)k * stride0 + (size_t)col] >= launch_pa) k_best = k;
    }
    int k_launch = (k_best >= 0) ? k_best : ((nz / 4) > 1 ? (nz / 4) : 1);
    if (k_launch < 1) k_launch = 1;
    if (k_launch > nz - 3) k_launch = nz - 3;

    // Load small 16-element spectral tables into registers
    float c0[NOGWD_NC];
    float dc[NOGWD_NC];
    float w0[NOGWD_NC];
    #pragma unroll
    for (int j = 0; j < NOGWD_NC; ++j) {
        c0[j] = c0_arr[j];
        dc[j] = dc_arr[j];
        w0[j] = w0_arr[j];
    }

    // Latitudinal modulation of launch momentum flux
    float tau_col = launch_flux_pa;
    if (lat_mode != 0 && lat_deg != nullptr) {
        float ldeg = (lat_mode == 1) ? lat_deg[col / nlon] : lat_deg[col];
        float lrad = ldeg * DEG2RAD_F;
        float s2 = sinf(2.0f * lrad);
        float t = lrad / 0.28f;
        float fac = 0.80f + 0.45f * (s2 * s2) + 0.25f * expf(-(t * t));
        tau_col *= fac;
    }

    size_t stride = (size_t)ncol;
    size_t idx_l = (size_t)k_launch * stride + (size_t)col;
    float u_l = u[idx_l];
    float v_l = v[idx_l];
    float u_launch[4] = { u_l, -u_l, v_l, -v_l };

    // Flux density per azimuth and phase speed
    float flux[4][NOGWD_NC];
    #pragma unroll
    for (int j = 0; j < NOGWD_NC; ++j) {
        float f0 = w0[j] * tau_col;
        flux[0][j] = f0;
        flux[1][j] = f0;
        flux[2][j] = f0;
        flux[3][j] = f0;
    }

    // Prime sliding stencil for centered Brunt-Vaisala N(z)
    float th_prev = theta[idx_l];
    float dz_prev = dz[idx_l];
    size_t idx_curr = idx_l + stride;
    float th_curr = theta[idx_curr];
    float dz_curr = dz[idx_curr];

    for (int k = k_launch + 1; k < nz; ++k) {
        size_t idx = (size_t)k * stride + (size_t)col;
        float uk = u[idx];
        float vk = v[idx];
        float pfk = p_full[idx];
        float exk = exner[idx];

        float dth, dz_c;
        float th_next = th_curr;
        float dz_next = dz_curr;
        if (k < nz - 1) {
            th_next = theta[idx + stride];
            dz_next = dz[idx + stride];
            dth = th_next - th_prev;
            dz_c = dz_curr + 0.5f * (dz_next + dz_prev);
        } else {
            dth = th_curr - th_prev;
            dz_c = 0.5f * (dz_curr + dz_prev);
        }

        float tk = th_curr * exk;
        float rhok = pfk / (rd * fmaxf(tk, 100.0f));
        float n2_sq = (grav / fmaxf(th_curr, 150.0f)) * (dth / fmaxf(dz_c, 10.0f));
        float bvk = sqrtf(fmaxf(n2_sq, 2.5e-5f));
        float fac_k = sat_coeff * rhok / bvk;

        float u_az[4] = { uk, -uk, vk, -vk };
        float tdep[4];

        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            float du_l_a = u_launch[a] - u_az[a];
            float sum_dep = 0.0f;
            #pragma unroll
            for (int j = 0; j < NOGWD_NC; ++j) {
                float ch = du_l_a + c0[j];
                float fj = flux[a][j];
                float nf = (ch > 0.0f) ? fminf(fj, fac_k * ch) : 0.0f;
                sum_dep += (fj - nf) * dc[j];
                flux[a][j] = nf;
            }
            tdep[a] = sum_dep;
        }

        float inv_mass = 1.0f / fmaxf(rhok * dz_curr, 1.0e-3f);
        float du_k = (tdep[0] - tdep[1]) * inv_mass;
        float dv_k = (tdep[2] - tdep[3]) * inv_mass;
        du_k = fminf(fmaxf(du_k, -max_tend), max_tend);
        dv_k = fminf(fmaxf(dv_k, -max_tend), max_tend);

        du_out[idx] += du_k;
        dv_out[idx] += dv_k;

        if (heat_booking && dth_out != nullptr) {
            float diss = fmaxf(0.0f, -(uk * du_k + vk * dv_k));
            dth_out[idx] += diss / (cp * fmaxf(exk, 0.05f));
        }

        th_prev = th_curr;
        dz_prev = dz_curr;
        th_curr = th_next;
        dz_curr = dz_next;
    }
}
"""

_CUDA_CACHE: dict[int, tuple] = {}


def _get_cuda_bundle():
    import cupy as cp

    dev_id = int(cp.cuda.Device().id)
    bundle = _CUDA_CACHE.get(dev_id)
    if bundle is None:
        mod = cp.RawModule(code=_NOGWD_CUDA_SOURCE, options=("-std=c++14",))
        k_run = mod.get_function("scinocca_nogwd_fp32_direct")
        c0_d = cp.asarray(_C0_HOST, dtype=cp.float32)
        dc_d = cp.asarray(_DC_HOST, dtype=cp.float32)
        w0_d = cp.asarray(_W_HOST, dtype=cp.float32)
        bundle = (k_run, c0_d, dc_d, w0_d)
        _CUDA_CACHE[dev_id] = bundle
    return bundle


def _record(metadata: dict, launch_pa: float) -> None:
    """The call's metadata: the scheme and its launch pressure, which are the
    same in every latitude band and on every card.  The launch LEVEL is each
    column's own and is not a per-call reading (a per-call level read off the
    batch's mean pressure differed between bands, 26 against 27 at T255 L80,
    and the banded run refused, 2026-11-22)."""
    metadata["nonorographic_gwd"] = "scinocca"
    metadata["nogwd_launch_pa"] = float(launch_pa)
    metadata["nogwd_calls"] = int(metadata.get("nogwd_calls", 0)) + 1


def _lat_flux_factor(latitude_deg, xp, profile: str) -> object:
    """Latitudinal modulation of launch momentum flux (IFS Cy35r3-style storm-track boost)."""
    if profile == "uniform" or latitude_deg is None:
        return None
    lat_rad = xp.deg2rad(xp.asarray(latitude_deg, dtype=xp.float32))
    sin2 = xp.sin(2.0 * lat_rad) ** 2
    trop = xp.exp(-((lat_rad / xp.float32(0.28)) ** 2))
    factor = xp.float32(0.80) + xp.float32(0.45) * sin2 + xp.float32(0.25) * trop
    return factor[:, None] if factor.ndim == 1 else factor


def _apply_nonorographic_gwd_cuda(
    u, v, theta, p_full, exner, dz, pbl_out: dict,
    *,
    launch_flux_pa: float,
    launch_pa: float,
    cstar_scale: float,
    heat_booking: bool,
    lat_profile: str,
    latitude_deg,
    metadata: dict | None,
) -> None:
    import cupy as cp

    nz, nlat, nlon = u.shape
    ncol = nlat * nlon
    f32 = lambda a: cp.ascontiguousarray(a, dtype=cp.float32)

    u_c = f32(u)
    v_c = f32(v)
    th_c = f32(theta)
    pf_c = f32(p_full)
    ex_c = f32(exner)
    dz_c = f32(dz)

    du_out = pbl_out["du"]
    dv_out = pbl_out["dv"]
    if not du_out.flags.c_contiguous or du_out.dtype != cp.float32:
        du_out = cp.ascontiguousarray(du_out, dtype=cp.float32)
        pbl_out["du"] = du_out
    if not dv_out.flags.c_contiguous or dv_out.dtype != cp.float32:
        dv_out = cp.ascontiguousarray(dv_out, dtype=cp.float32)
        pbl_out["dv"] = dv_out

    do_heat = bool(heat_booking and "dtheta" in pbl_out)
    if do_heat:
        dth_out = pbl_out["dtheta"]
        if not dth_out.flags.c_contiguous or dth_out.dtype != cp.float32:
            dth_out = cp.ascontiguousarray(dth_out, dtype=cp.float32)
            pbl_out["dtheta"] = dth_out
    else:
        dth_out = np.uint64(0)

    if lat_profile == "uniform" or latitude_deg is None:
        lat_mode = 0
        lat_arr = np.uint64(0)
    else:
        lat_arr = f32(latitude_deg)
        lat_mode = 1 if lat_arr.ndim == 1 else 2

    k_run, c0_d, dc_d, w0_d = _get_cuda_bundle()

    sat_coeff = float(0.08 * float(cstar_scale) * NOGWD_OMEGA_STAR_S_INV)
    tpb = 128
    blocks = (ncol + tpb - 1) // tpb
    k_run(
        (blocks,),
        (tpb,),
        (
            u_c, v_c, th_c, pf_c, ex_c, dz_c, lat_arr,
            c0_d, dc_d, w0_d, np.float32(launch_pa),
            du_out, dv_out, dth_out,
            np.int32(nz), np.int32(nlat), np.int32(nlon),
            np.int32(lat_mode), np.int32(1 if do_heat else 0),
            np.float32(launch_flux_pa), np.float32(sat_coeff),
            np.float32(NOGWD_MAX_TENDENCY_M_S2),
            np.float32(DRY_AIR_GAS_CONSTANT), np.float32(DRY_AIR_CP),
            np.float32(GRAVITY_M_S2),
        ),
    )

    if metadata is not None:
        _record(metadata, launch_pa)


def apply_nonorographic_gwd(
    batch,
    pbl_out: dict,
    *,
    dz,
    launch_flux_pa: float = 0.00375,
    launch_pa: float = 45000.0,
    cstar_scale: float = 1.0,
    heat_booking: bool = True,
    lat_profile: str = "ifs",
    latitude_deg=None,
    metadata: dict | None = None,
    use_cuda: bool | None = None,
) -> None:
    """Add Scinocca/WMS non-orographic GWD tendencies into ``pbl_out`` in place."""
    xp = batch.xp
    u = batch.arrays["u"]          # (nz, nlat, nlon), bottom-to-top
    v = batch.arrays["v"]
    theta = batch.arrays["theta"]
    p_full = batch.arrays["p_full"]
    exner = batch.arrays["exner"]

    if use_cuda is None:
        use_cuda = os.environ.get("ARWEN_NOGWD_CUDA", "1").strip() != "0"

    if use_cuda and xp.__name__ == "cupy" and u.dtype == xp.float32:
        _apply_nonorographic_gwd_cuda(
            u, v, theta, p_full, exner, dz, pbl_out,
            launch_flux_pa=launch_flux_pa,
            launch_pa=launch_pa,
            cstar_scale=cstar_scale,
            heat_booking=heat_booking,
            lat_profile=lat_profile,
            latitude_deg=latitude_deg,
            metadata=metadata,
        )
        return

    nz, nlat, nlon = u.shape
    ncol = nlat * nlon

    # Reshape to (nz, ncol) for clean vectorized GPU/CPU ops
    u2 = u.reshape(nz, ncol)
    v2 = v.reshape(nz, ncol)
    th2 = theta.reshape(nz, ncol)
    pf2 = p_full.reshape(nz, ncol)
    ex2 = exner.reshape(nz, ncol)
    dz2 = xp.asarray(dz, dtype=xp.float32).reshape(nz, ncol)

    # Temperature & density
    t2 = th2 * ex2
    rho2 = pf2 / (xp.float32(DRY_AIR_GAS_CONSTANT) * xp.maximum(t2, xp.float32(100.0)))

    # Brunt-Vaisala frequency N at full levels (centered differences)
    dth = xp.empty_like(th2)
    dz_c = xp.empty_like(dz2)
    dth[1:-1] = th2[2:] - th2[:-2]
    dz_c[1:-1] = dz2[1:-1] + xp.float32(0.5) * (dz2[2:] + dz2[:-2])
    dth[0] = th2[1] - th2[0]
    dz_c[0] = xp.float32(0.5) * (dz2[0] + dz2[1])
    dth[-1] = th2[-1] - th2[-2]
    dz_c[-1] = xp.float32(0.5) * (dz2[-1] + dz2[-2])
    n2_sq = (xp.float32(GRAVITY_M_S2) / xp.maximum(th2, xp.float32(150.0))) * (
        dth / xp.maximum(dz_c, xp.float32(10.0))
    )
    bv2 = xp.sqrt(xp.maximum(n2_sq, xp.float32(2.5e-5)))  # N >= 0.005 s^-1

    # Each column's launch level: the highest full level still at or below
    # launch_pa (pressure falls with k), as the kernel takes it.  The lowest
    # launch level of the batch is where the sweep starts; a column whose own
    # level is higher joins the sweep there.
    count = xp.sum(pf2 >= xp.float32(launch_pa), axis=0).astype(xp.int32)   # (ncol,)
    k_col = xp.where(count > 0, count - 1, max(1, nz // 4)).astype(xp.int32)
    k_col = xp.clip(k_col, 1, nz - 3)
    k_first = int(k_col.min())

    # Spectral phase-speed arrays on device: shape (nc, 1)
    c0 = xp.asarray(_C0_HOST, dtype=xp.float32)[:, None]
    dc = xp.asarray(_DC_HOST, dtype=xp.float32)[:, None]
    w0 = xp.asarray(_W_HOST, dtype=xp.float32)[:, None]

    # Launch total flux per azimuth: (1, ncol)
    tau_base = xp.full((nlat, nlon), xp.float32(launch_flux_pa), dtype=xp.float32)
    lat_fac = _lat_flux_factor(latitude_deg, xp, lat_profile)
    if lat_fac is not None:
        tau_base = tau_base * lat_fac
    tau_col = tau_base.reshape(1, ncol)

    # Exact Warner-McIntyre (1999) / Scinocca (2003) saturation density in c_hat:
    # F_sat(c_hat, z) = C_sat * rho(z) * (omega* / N(z)) * c_hat  [Pa / (m/s)]
    sat_coeff = xp.float32(0.08 * float(cstar_scale) * NOGWD_OMEGA_STAR_S_INV)

    # 4 orthogonal azimuths stacked as (4, ncol) at each level:
    # 0: +u (East), 1: -u (West), 2: +v (North), 3: -v (South)
    cols = xp.arange(ncol)
    u_l = u2[k_col, cols]
    v_l = v2[k_col, cols]
    u_launch = xp.stack((u_l, -u_l, v_l, -v_l), axis=0)  # (4, ncol)
    # Ground-relative phase speeds: (4, nc, ncol)
    c_ground = u_launch[:, None, :] + c0[None, :, :]
    # Initial flux density at launch level: (4, nc, ncol)
    flux = xp.broadcast_to(w0[None, :, :] * tau_col[None, :, :], (4, NOGWD_NC, ncol)).copy()

    du_acc = xp.zeros((nz, ncol), dtype=xp.float32)
    dv_acc = xp.zeros((nz, ncol), dtype=xp.float32)
    max_tend = xp.float32(NOGWD_MAX_TENDENCY_M_S2)

    # Propagate upward from each column's launch level + 1 to nz - 1
    for k in range(k_first + 1, nz):
        above = (k > k_col)[None, None, :]  # (1, 1, ncol): the column has launched
        u_az = xp.stack((u2[k], -u2[k], v2[k], -v2[k]), axis=0)  # (4, ncol)
        c_hat = c_ground - u_az[:, None, :]  # (4, nc, ncol)
        # Critical-level absorption (c_hat <= 0)
        alive = c_hat > xp.float32(0.0)
        c_pos = xp.where(alive, c_hat, xp.float32(0.0))
        # Saturation threshold at level k
        fac_k = (sat_coeff * rho2[k] / bv2[k])[None, None, :]  # (1, 1, ncol)
        f_sat = fac_k * c_pos
        new_flux = xp.where(alive, xp.minimum(flux, f_sat), xp.float32(0.0))
        new_flux = xp.where(above, new_flux, flux)

        # Deposited momentum flux in layer k: df = (flux - new_flux) >= 0
        # Integrated over phase speeds: tau_dep = sum_j (flux - new_flux) * dc_j  -> (4, ncol)
        tau_dep = xp.sum((flux - new_flux) * dc[None, :, :], axis=1)
        inv_mass = xp.float32(1.0) / xp.maximum(rho2[k] * dz2[k], xp.float32(1.0e-3))

        # Eastward wave (+0) accelerates +u; Westward wave (+1) accelerates -u
        # Northward wave (+2) accelerates +v; Southward wave (+3) accelerates -v
        du_k = (tau_dep[0] - tau_dep[1]) * inv_mass
        dv_k = (tau_dep[2] - tau_dep[3]) * inv_mass
        du_acc[k] = xp.clip(du_k, -max_tend, max_tend)
        dv_acc[k] = xp.clip(dv_k, -max_tend, max_tend)
        flux = new_flux

    du_3d = du_acc.reshape(nz, nlat, nlon)
    dv_3d = dv_acc.reshape(nz, nlat, nlon)
    pbl_out["du"] += du_3d
    pbl_out["dv"] += dv_3d

    if heat_booking and "dtheta" in pbl_out:
        # Kinetic-energy dissipation returned as positive heating:
        # cp * dT/dt = max(0, -(u * du + v * dv))
        diss = xp.maximum(xp.float32(0.0), -(u * du_3d + v * dv_3d))
        pbl_out["dtheta"] += diss / (xp.float32(DRY_AIR_CP) * xp.maximum(exner, xp.float32(0.05)))

    if metadata is not None:
        _record(metadata, launch_pa)
