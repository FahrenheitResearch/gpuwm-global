"""One ocean step (design 6): density, baroclinic momentum (AB3), the
split-explicit barotropic mode, KPP, GM, tracers, books.

The state is a plain dict of arrays (names in :data:`STATE_ARRAYS`), so the
coupler can keep it in the physics state and a checkpoint carries it bit
for bit.  Momentum is float32 (u, v and their AB3 history); temperature,
salinity, the free surface, the barotropic transports and every volume,
transport and tracer update are float64, so heat, salt and volume close to
float64 rounding whatever the momentum precision.

Step order, one ocean step of ``dt`` (3600 s):

1. density from the Wright EOS at the nominal level pressure; rho' against
   the fixed reference profile; hydrostatic phi'.
2. KPP and the interior mixing from the state at n.
3. baroclinic tendencies: vector-invariant momentum (Sadourny
   energy-conserving Coriolis-vorticity on the C grid, kinetic-energy
   gradient, centred vertical advection), the z* pressure gradient
   (-grad phi' - g rho'/rho0 grad z), AB3 (Euler, AB2 for the first two
   steps); Laplacian viscosity (max of Munk, Smagorinsky and a floor,
   anisotropic within 10 deg of the equator) forward in time; implicit
   vertical viscosity with the surface stress, ice-ocean drag and
   quadratic bottom drag.
4. barotropic substeps (forward-backward, eta first, alternating Coriolis
   order, a 0.2 forward extrapolation of the new eta in the pressure
   gradient for damping); the free surface steps with the mean of the
   substep transports, the baroclinic velocities take the final
   transport's depth mean.
5. GM (Ferrari et al. 2010 boundary-value problem per column, kappa 800).
6. tracers: dimensionally split flux-limited (Lax-Wendroff with the MC
   limiter, Sweby 1984) advection with intermediate volumes, horizontal
   diffusion 200 m2 s-1, surface heat and salt with two-band shortwave
   (Jerlov IB: R 0.67, 1.0 m, 17 m) and KPP's non-local flux, implicit
   vertical diffusion, the northern sponge, SSS restoring, frazil.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math

import numpy as np

from . import eos
from .column import by_rows, implicit_diffusion, thomas
from .grid import CP0, GRAVITY, RHO0
from .kpp import kpp as kpp_scheme

#: Prognostic arrays and their dtypes.
STATE_DTYPES = {
    "theta": "float64", "salt": "float64",
    "u": "float32", "v": "float32",
    "gu1": "float32", "gu2": "float32", "gv1": "float32", "gv2": "float32",
    "w": "float32",
    "eta": "float64", "ubt": "float64", "vbt": "float64",
    "hbl": "float32", "rho_ref": "float64", "books": "float64",
}
STATE_ARRAYS = tuple(STATE_DTYPES)
#: Books vector layout.
BOOKS = ("steps", "heat_content0_j", "heat_sources_j", "salt_content0", "salt_sources",
         "volume0_m3", "frazil_j", "sponge_heat_j", "restoring_heat_j", "sss_salt",
         "salt_flux_removed", "remap_residual_heat_j", "ab_steps", "sponge_salt", "spare2", "spare3")
FREEZING_C = -1.8  # 271.35 K, the slab's and the ice code's freezing point
#: Above this Courant number the vertical tracer transport through an
#: interface is advected implicitly (upwind) for the excess.
VERTICAL_COURANT_MAX = 0.5
KELVIN = 273.15


@dataclass(frozen=True)
class OceanParams:
    dt: float = 3600.0
    gm_kappa: float = 800.0
    gm_slope_max: float = 0.01
    kappa_h: float = 200.0
    visc_a_min: float = 500.0
    visc_a_y_equatorial: float = 2.0e3
    visc_smag: float = 2.0
    cd_bottom: float = 1.0e-3
    u_tide: float = 0.05
    c_io: float = 5.5e-3
    ch_io: float = 0.006
    ustar_io_min: float = 0.002
    bebt: float = 0.2
    sss_piston_m_per_day: float | None = 50.0 / 300.0
    sst_restoring_days: float | None = None
    sponge_lat0: float = 60.0
    sponge_lat1: float = 65.0
    sponge_days: float = 30.0
    jerlov_r: float = 0.67
    jerlov_z1: float = 1.0
    jerlov_z2: float = 17.0
    s_ref: float = 35.0
    s_ice: float = 4.0
    rho_ice: float = 917.0
    gm: bool = True
    kpp: bool = True
    advection: bool = True   # momentum advection (off for linear tests)
    viscosity: bool = True
    tracer_advection: bool = True
    surface_fluxes: bool = True
    sponge: bool = True
    substeps: int | None = None

    def identity(self) -> dict:
        return asdict(self)


def _f32(x):
    return np.float32(x)


class OceanModel:
    def __init__(self, grid, params: OceanParams | None = None, xp=np, targets=None):
        self.grid = grid
        self.p = params or OceanParams()
        self.xp = xp
        g = grid.on(xp)
        self.g = g
        self.g32 = {k: (v.astype(xp.float32) if getattr(v, "dtype", None) is not None
                        and v.dtype == xp.float64 else v) for k, v in g.items()}
        self.nsub = int(self.p.substeps or grid.substeps)
        #: Row blocks for the column work on a card (memory only: every
        #: column's arithmetic is the same at any block count).
        self.chunks = 1 if xp is np else 8
        self.targets = targets  # dict with optional sponge / sss targets (callables)
        nz = grid.nz
        self.p_ref = xp.asarray(RHO0 * GRAVITY * grid.z_t_nom).reshape(nz, 1, 1)
        zc = np.cumsum(grid.dzt, axis=0) - 0.5 * grid.dzt
        self.zc = xp.asarray(np.where(grid.tmask, zc, 0.0))
        # depth of the interface below each level, actual-static.
        self.zw = xp.asarray(np.where(grid.tmask, np.cumsum(grid.dzt, axis=0), 0.0))
        self._viscosity_static()
        lat_abs = np.abs(grid.lat_t)
        self.lat_abs2 = xp.asarray(np.broadcast_to(lat_abs[:, None], (grid.ny, grid.nx)).copy())
        self.f_abs2 = xp.asarray(np.abs(np.broadcast_to(grid.f_t, (grid.ny, grid.nx))).copy())
        lam = np.zeros(grid.ny)
        if self.p.sponge:
            frac = np.clip((grid.lat_t - self.p.sponge_lat0)
                           / (self.p.sponge_lat1 - self.p.sponge_lat0), 0.0, 1.0)
            lam = frac / (self.p.sponge_days * 86400.0)
        self.sponge_rate = xp.asarray(lam[:, None])
        self.sponge_rows = np.nonzero(lam > 0)[0]

    # -- shifts (x periodic; y walls) ----------------------------------------
    def ex(self, a):
        return self.xp.roll(a, -1, axis=-1)

    def wx(self, a):
        return self.xp.roll(a, 1, axis=-1)

    def nj(self, a):
        out = self.xp.zeros_like(a)
        out[..., :-1, :] = a[..., 1:, :]
        return out

    def sj(self, a):
        out = self.xp.zeros_like(a)
        out[..., 1:, :] = a[..., :-1, :]
        return out

    # -- static pieces -------------------------------------------------------
    def _viscosity_static(self):
        grid, p, xp = self.grid, self.p, self.xp
        dx = grid.dxt[:, 0]
        dy = grid.dyt[:, 0]
        beta = np.abs(grid.beta_t[:, 0])
        munk = beta * dx ** 3
        ax = np.maximum(munk, p.visc_a_min)
        ay = np.where(np.abs(grid.lat_t) < 10.0, p.visc_a_y_equatorial,
                      np.where(np.abs(grid.lat_t) < 20.0,
                               p.visc_a_y_equatorial + (ax - p.visc_a_y_equatorial)
                               * (np.abs(grid.lat_t) - 10.0) / 10.0, ax))
        ay = np.maximum(ay, p.visc_a_min)
        # Forward-Euler stability: A dt (1/dx^2 + 1/dy^2) <= 0.3.
        lim = 0.3 / (p.dt * (1.0 / dx ** 2 + 1.0 / dy ** 2))
        self.visc_lim = xp.asarray(lim[:, None]).astype(xp.float32)
        self.ax_static = xp.asarray(np.minimum(ax, lim)[:, None]).astype(xp.float32)
        self.ay_static = xp.asarray(np.minimum(ay, lim)[:, None]).astype(xp.float32)

    # -- state ---------------------------------------------------------------
    def new_state(self, theta_c, salt, u=None, v=None, eta=None, rho_ref=None) -> dict:
        """A state at rest (or with the given velocities and free surface)
        from temperature (deg C) and salinity, (nz, ny, nx)."""
        xp, grid = self.xp, self.grid
        g = self.g
        tm = g["tmask"]
        theta = xp.where(tm, xp.asarray(theta_c, dtype=xp.float64), 0.0)
        salt = xp.where(tm, xp.asarray(salt, dtype=xp.float64), 0.0)
        shape3 = (grid.nz, grid.ny, grid.nx)
        zeros32 = xp.zeros(shape3, dtype=xp.float32)
        uu = zeros32.copy() if u is None else xp.where(g["umask"], xp.asarray(u, xp.float32), 0)
        vv = zeros32.copy() if v is None else xp.where(g["vmask"], xp.asarray(v, xp.float32), 0)
        ee = xp.zeros((grid.ny, grid.nx)) if eta is None else xp.where(g["wet"], xp.asarray(eta, xp.float64), 0.0)
        if rho_ref is None:
            rho = eos.density(theta, salt, self.p_ref)
            vol = g["dzt"] * g["area"]
            w = vol.sum(axis=(1, 2))
            rho_ref = (xp.where(tm, rho, 0.0) * vol).sum(axis=(1, 2)) / xp.where(w > 0, w, 1.0)
            # Levels below every column take the deepest defined value.
            r = xp.asnumpy(rho_ref) if hasattr(xp, "asnumpy") else np.asarray(rho_ref)
            wn = xp.asnumpy(w) if hasattr(xp, "asnumpy") else np.asarray(w)
            last = r[wn > 0][-1] if (wn > 0).any() else RHO0
            r = np.where(wn > 0, r, last)
            rho_ref = xp.asarray(r)
        s = 1.0 + ee / xp.where(g["wet"], g["H"], 1.0)
        su = 0.5 * (s + self.ex(s))
        sv = 0.5 * (s + self.nj(s))
        hu = g["dzu"].astype(xp.float64) * su[None]
        hv = g["dzv"].astype(xp.float64) * sv[None]
        state = {
            "theta": theta, "salt": salt, "u": uu, "v": vv,
            "gu1": zeros32.copy(), "gu2": zeros32.copy(), "gv1": zeros32.copy(), "gv2": zeros32.copy(),
            "w": zeros32.copy(), "eta": ee,
            "ubt": (hu * uu).sum(axis=0), "vbt": (hv * vv).sum(axis=0),
            "hbl": xp.zeros((grid.ny, grid.nx), dtype=xp.float32),
            "rho_ref": xp.asarray(rho_ref, dtype=xp.float64),
            "books": xp.zeros(len(BOOKS), dtype=xp.float64),
        }
        heat, salt_c, vol = self.contents(state)
        b = np.zeros(len(BOOKS))
        b[BOOKS.index("heat_content0_j")] = heat
        b[BOOKS.index("salt_content0")] = salt_c
        b[BOOKS.index("volume0_m3")] = vol
        state["books"] = xp.asarray(b)
        return state

    def contents(self, state):
        xp, g = self.xp, self.g
        s = 1.0 + state["eta"] / xp.where(g["wet"], g["H"], 1.0)
        vol = g["dzt"] * s[None] * g["area"]
        heat = float((vol * state["theta"]).sum()) * RHO0 * CP0
        salt = float((vol * state["salt"]).sum())
        return heat, salt, float(vol.sum())

    # -- pieces --------------------------------------------------------------
    def _density(self, theta, salt, rho_ref):
        rho, r_t, r_s = eos.density_derivatives(theta, salt, self.p_ref)
        tm = self.g["tmask"]
        rhop = self.xp.where(tm, rho - rho_ref.reshape(-1, 1, 1), 0.0)
        return rhop, r_t, r_s

    def _n2(self, theta, salt, r_t, r_s, h):
        """N^2 at the interface below each level (0 where no level below)."""
        xp = self.xp
        below_h = xp.zeros_like(h)
        below_h[:-1] = h[1:]
        delta = 0.5 * (h + below_h)
        dth = xp.zeros_like(theta)
        ds = xp.zeros_like(salt)
        dth[:-1] = theta[:-1] - theta[1:]
        ds[:-1] = salt[:-1] - salt[1:]
        rt = r_t.copy()
        rs = r_s.copy()
        rt[:-1] = 0.5 * (r_t[:-1] + r_t[1:])
        rs[:-1] = 0.5 * (r_s[:-1] + r_s[1:])
        ok = (h > 0) & (below_h > 0)
        n2 = -(GRAVITY / RHO0) * (rt * dth + rs * ds) / xp.where(ok, delta, 1.0)
        return xp.where(ok, n2, 0.0)

    def _sw_fraction(self, depth):
        p, xp = self.p, self.xp
        return p.jerlov_r * xp.exp(-depth / p.jerlov_z1) + (1.0 - p.jerlov_r) * xp.exp(-depth / p.jerlov_z2)

    # -- the step ------------------------------------------------------------
    def step(self, st: dict, fc: dict) -> tuple[dict, dict]:
        """One step.  ``fc`` (ocean grid, (ny, nx) float64, time means over
        the step): ``taux``, ``tauy`` (N m-2 at T points, the air stress
        on the open part), ``ice`` (fraction), ``q_ns``, ``q_sw`` (W m-2
        into the ocean), ``fw`` (kg m-2 s-1 into the ocean, P - E - ...),
        ``ice_growth_m`` (ice volume gained over the step, m), optional
        ``sst_target_c``.  Returns the new state and the step's
        diagnostics (frazil plane, F_bot potential, books increments)."""
        xp, g, g32, p = self.xp, self.g, self.g32, self.p
        dt = p.dt
        f32 = xp.float32
        ex, wx, nj, sj = self.ex, self.wx, self.nj, self.sj
        nz = self.grid.nz
        theta, salt = st["theta"], st["salt"]
        u, v = st["u"], st["v"]
        eta = st["eta"]
        wet = g["wet"]
        tm, um, vm = g["tmask"], g["umask"], g["vmask"]
        Hs = xp.where(wet, g["H"], 1.0)
        s = xp.where(wet, 1.0 + eta / Hs, 1.0)
        h = g["dzt"] * s[None]
        su = 0.5 * (s + ex(s))
        sv = 0.5 * (s + nj(s))
        hu64 = g["dzu"].astype(xp.float64) * su[None]
        hv64 = g["dzv"].astype(xp.float64) * sv[None]
        hu = hu64.astype(f32)
        hv = hv64.astype(f32)
        hu_safe = xp.where(um, hu, f32(1.0))
        hv_safe = xp.where(vm, hv, f32(1.0))

        # 1. density
        rhop, r_t, r_s = self._density(theta, salt, st["rho_ref"])
        # The baroclinic pressure gradient is taken on the resting levels (s = 1,
        # eta = 0): its eta-dependent part is a fraction rho'/rho0 of the fast
        # gravity-wave force, and evaluated explicitly at n with a 3600 s step
        # (omega dt ~ 45) it grew a barotropic mode (eta to 29 m in 2.5 days of
        # the GODAS spin-up before this).  The free surface acts through the
        # barotropic -g D grad(eta) alone; the error is O(eta / H).
        rh = rhop * g["dzt"]
        phi = (GRAVITY / RHO0) * (xp.cumsum(rh, axis=0) - 0.5 * rh)
        zc = -self.zc
        # 2. forcing at T, surface stress magnitude, KPP
        zero2 = xp.zeros_like(eta)
        taux = fc.get("taux", zero2)
        tauy = fc.get("tauy", zero2)
        ice = xp.clip(fc.get("ice", zero2), 0.0, 1.0)
        q_ns = fc.get("q_ns", zero2) if p.surface_fluxes else zero2
        q_sw = fc.get("q_sw", zero2) if p.surface_fluxes else zero2
        fw = fc.get("fw", zero2) if p.surface_fluxes else zero2
        uT = 0.5 * (u + wx(u))
        vT = 0.5 * (v + sj(v))
        u1 = xp.sqrt(uT[0].astype(xp.float64) ** 2 + vT[0].astype(xp.float64) ** 2)
        tau_io = RHO0 * p.c_io * ice * u1 * u1
        ustar = xp.sqrt((xp.sqrt(taux ** 2 + tauy ** 2) + tau_io) / RHO0)
        # salt flux (virtual), global mean removed
        sflux = -p.s_ref * fw / RHO0
        if p.surface_fluxes and "salt_flux" in fc:  # a given salt flux (psu m s-1)
            sflux = sflux + fc["salt_flux"]
        sflux = sflux + (p.s_ref - p.s_ice) * (p.rho_ice / RHO0) * fc.get("ice_growth_m", zero2) / dt
        area2 = g["area"] * wet
        mean_s = (sflux * area2).sum() / area2.sum()
        sflux = xp.where(wet, sflux - mean_s, 0.0)
        depth_w = self.zw * s[None]
        depth_t = self.zc * s[None]
        sw_w = xp.where(tm, self._sw_fraction(depth_w), 0.0)
        n2 = self._n2(theta, salt, r_t, r_s, h)
        if p.kpp:
            mix = by_rows(kpp_scheme, self.chunks, self.grid.ny, xp,
                             theta=theta, salt=salt, uT=uT, vT=vT, h=h, depth_t=depth_t, depth_w=depth_w,
                             n2=n2, drho_dt=r_t, drho_ds=r_s, ustar=ustar, q_ns=q_ns, q_sw=q_sw,
                             salt_flux=sflux, sw_frac_w=sw_w, f_abs=self.f_abs2,
                             lat_abs=self.lat_abs2, wet=wet, xp=xp)
        else:
            from .kpp import interior

            below = xp.zeros_like(h)
            below[:-1] = h[1:]
            iface = tm & (below > 0)
            ks, km = interior(n2, xp.zeros_like(n2), self.lat_abs2, xp)
            mix = {"kappa_s": xp.where(iface, ks, 0.0), "kappa_m": xp.where(iface, km, 0.0),
                   "nonlocal": xp.zeros_like(h), "hbl": zero2}
        # (the card's transient memory is what bounds T383: every large
        # temporary is dropped at its last use)
        del depth_t, depth_w, uT, vT

        # 3. baroclinic tendencies (float32)
        pgf_u = -((ex(phi) - phi) + (GRAVITY / RHO0) * 0.5 * (rhop + ex(rhop)) * (ex(zc) - zc)) / g["dxu"]
        Gu = xp.where(um, pgf_u, 0.0).astype(f32)
        del pgf_u
        pgf_v = -((nj(phi) - phi) + (GRAVITY / RHO0) * 0.5 * (rhop + nj(rhop)) * (nj(zc) - zc)) / g["dyv"]
        Gv = xp.where(vm, pgf_v, 0.0).astype(f32)
        del pgf_v, phi, zc, rh
        dxu, dyu, dxv, dyv = g32["dxu"], g32["dyu"], g32["dxv"], g32["dyv"]
        dxt, dyt = g32["dxt"], g32["dyt"]
        qmask = um & nj(um) & vm & ex(vm)
        # The planetary part of the Coriolis term acts on the BAROCLINIC
        # velocity only: the barotropic flow's Coriolis term is the
        # substeps' (alternating, neutral).  Extrapolating the barotropic
        # part with AB3 and subtracting it at n leaves (dt/2) f^2 U, which
        # grows barotropic inertial motion (e-folding ~22 h at 35N:
        # measured as the slump-test blow-up before this split).
        umb0 = g["Hu"] > 0
        vmb0 = g["Hv"] > 0
        ubar0 = ((st["ubt"] / xp.where(umb0, hu64.sum(axis=0), 1.0))).astype(f32)
        vbar0 = ((st["vbt"] / xp.where(vmb0, hv64.sum(axis=0), 1.0))).astype(f32)
        # Sadourny's energy-conserving form: q = (f + zeta) / h_q at the
        # corners, u_t = avg_y(q avg_x(V h)), v_t = -avg_x(q avg_y(U h)).
        # Dividing by the U-face thickness instead is not energy-conserving
        # where face thicknesses differ (partial cells, steps) and made
        # energy there (measured: a rest state grew 4x a day at a shelf
        # step near 59.5N 6W before this form).
        nface = (um.astype(f32) + nj(um).astype(f32) + vm.astype(f32) + ex(vm).astype(f32))
        h_q = (hu + nj(hu) + hv + ex(hv)) / xp.maximum(nface, f32(1.0))
        inv_hq = xp.where(nface > 0, f32(1.0) / xp.maximum(h_q, f32(1.0e-3)), f32(0.0))
        del nface, h_q
        Vt = xp.where(vm, (v - vbar0[None]) * hv, f32(0.0))
        Ut = xp.where(um, (u - ubar0[None]) * hu, f32(0.0))
        q = g32["f_q"] * inv_hq
        cor_u = f32(0.5) * (q * f32(0.5) * (Vt + ex(Vt)) + sj(q) * f32(0.5) * (sj(Vt) + ex(sj(Vt))))
        cor_v = -f32(0.5) * (q * f32(0.5) * (Ut + nj(Ut)) + wx(q) * f32(0.5) * (wx(Ut) + nj(wx(Ut))))
        Gu = Gu + xp.where(um, cor_u, f32(0.0))
        Gv = Gv + xp.where(vm, cor_v, f32(0.0))
        del cor_u, cor_v, Vt, Ut, q
        if p.advection:
            # relative vorticity of the full flow, the same form
            zeta = xp.where(qmask, (dyv * (ex(v) - v) - (nj(u * dxu) - u * dxu)) / g32["area_q"],
                            f32(0.0))
            q = zeta * inv_hq
            del zeta
            Vt = v * hv
            Ut = u * hu
            cor_u = f32(0.5) * (q * f32(0.5) * (Vt + ex(Vt)) + sj(q) * f32(0.5) * (sj(Vt) + ex(sj(Vt))))
            cor_v = -f32(0.5) * (q * f32(0.5) * (Ut + nj(Ut)) + wx(q) * f32(0.5) * (wx(Ut) + nj(wx(Ut))))
            Gu = Gu + xp.where(um, cor_u, f32(0.0))
            Gv = Gv + xp.where(vm, cor_v, f32(0.0))
            del cor_u, cor_v, Vt, Ut, q
        del inv_hq
        if p.advection:
            K = f32(0.25) * (u * u + wx(u * u) + v * v + sj(v * v))
            Gu = Gu - xp.where(um, (ex(K) - K) / dxu, f32(0.0))
            Gv = Gv - xp.where(vm, (nj(K) - K) / dyv, f32(0.0))
            del K
        # Vertical advection of momentum is implicit upwind, inside the
        # vertical viscosity solve (centred and explicit it was unstable
        # where strong flow crosses steep topography: w dt / h reaches
        # several there).  w at the U and V level centres:
        w_adv_u = w_adv_v = None
        if p.advection:
            w = st["w"]
            for comp, (mask, wface) in (("u", (um, f32(0.5) * (w + ex(w)))),
                                        ("v", (vm, f32(0.5) * (w + nj(w))))):
                wbot = wface * xp.concatenate([mask[1:], xp.zeros_like(mask[:1])]).astype(f32)
                wtop = xp.concatenate([xp.zeros_like(wbot[:1]), wbot[:-1]])
                wc = xp.where(mask, f32(0.5) * (wbot + wtop), f32(0.0))
                if comp == "u":
                    w_adv_u = wc
                else:
                    w_adv_v = wc
            del wbot, wtop, wface, wc
        n_ab = int(round(float(st["books"][BOOKS.index("ab_steps")])))
        if n_ab <= 0:
            Au, Av = Gu, Gv
        elif n_ab == 1:
            Au = f32(1.5) * Gu - f32(0.5) * st["gu1"]
            Av = f32(1.5) * Gv - f32(0.5) * st["gv1"]
        else:
            Au = (f32(23.0) * Gu - f32(16.0) * st["gu1"] + f32(5.0) * st["gu2"]) / f32(12.0)
            Av = (f32(23.0) * Gv - f32(16.0) * st["gv1"] + f32(5.0) * st["gv2"]) / f32(12.0)
        if p.viscosity:
            vu, vv = self._viscosity(u, v, um, vm, tm, qmask)
            Au = Au + vu
            Av = Av + vv
        us = xp.where(um, u + f32(dt) * Au, f32(0.0))
        vs = xp.where(vm, v + f32(dt) * Av, f32(0.0))
        # implicit vertical viscosity, surface stress, drags
        nu_t = mix["kappa_m"].astype(f32)
        nu_u = f32(0.5) * (nu_t + ex(nu_t))
        nu_v = f32(0.5) * (nu_t + nj(nu_t))
        taux_u = (0.5 * (taux + ex(taux))).astype(f32)
        tauy_v = (0.5 * (tauy + nj(tauy))).astype(f32)
        ice_u = (0.5 * (ice + ex(ice))).astype(f32)
        ice_v = (0.5 * (ice + nj(ice))).astype(f32)
        kb_u = xp.clip(um.sum(axis=0) - 1, 0, nz - 1)
        kb_v = xp.clip(vm.sum(axis=0) - 1, 0, nz - 1)
        ub = xp.take_along_axis(u, kb_u[None], axis=0)[0]
        vb = xp.take_along_axis(v, kb_v[None], axis=0)[0]
        r_bu = f32(p.cd_bottom) * xp.sqrt(ub * ub + f32(p.u_tide ** 2))
        r_bv = f32(p.cd_bottom) * xp.sqrt(vb * vb + f32(p.u_tide ** 2))
        r_iu = f32(p.c_io) * ice_u * xp.abs(u[0])
        r_iv = f32(p.c_io) * ice_v * xp.abs(v[0])
        ny = self.grid.ny
        uss = by_rows(implicit_diffusion, self.chunks, ny, xp, us, hu, nu_u, f32(dt), xp,
                      top_flux=taux_u / f32(RHO0), bottom_drag=r_bu, top_drag=r_iu,
                      w_adv=w_adv_u)
        vss = by_rows(implicit_diffusion, self.chunks, ny, xp, vs, hv, nu_v, f32(dt), xp,
                      top_flux=tauy_v / f32(RHO0), bottom_drag=r_bv, top_drag=r_iv,
                      w_adv=w_adv_v)
        del us, vs, nu_t, nu_u, nu_v, Au, Av, hu, hv, hu_safe, hv_safe, w_adv_u, w_adv_v
        mix.pop("kappa_m", None)
        if p.viscosity:
            del vu, vv

        # 4. barotropic mode (float64)
        Du = hu64.sum(axis=0)
        Dv = hv64.sum(axis=0)
        U0, V0 = st["ubt"], st["vbt"]
        umb = g["Hu"] > 0
        vmb = g["Hv"] > 0
        f_u, f_v = g["f_u"], g["f_v"]

        # Energy-conserving barotropic Coriolis (the same Sadourny form as
        # the baroclinic one): q = f / D_q at corners, D_q the mean depth of
        # the wet faces around the corner; dU/dt = D_u avg_y(q avg_x(V)).
        nf = (umb.astype(xp.float64) + nj(umb).astype(xp.float64) + vmb.astype(xp.float64)
              + ex(vmb).astype(xp.float64))
        Dq = (Du + nj(Du) + Dv + ex(Dv)) / xp.maximum(nf, 1.0)
        q_bt = xp.where(nf > 0, g["f_q"] / xp.maximum(Dq, 1.0e-3), 0.0)
        del nf, Dq

        def vbar(V):
            return 0.5 * (q_bt * 0.5 * (V + ex(V)) + sj(q_bt) * 0.5 * (sj(V) + ex(sj(V)))) * Du

        def ubar(U):
            return 0.5 * (q_bt * 0.5 * (U + nj(U)) + wx(q_bt) * 0.5 * (wx(U) + nj(wx(U)))) * Dv

        # (the baroclinic tendencies carry no planetary Coriolis term of the
        # barotropic flow, so nothing is subtracted here)
        GU = ((hu64 * uss).sum(axis=0) - (hu64 * u).sum(axis=0)) / dt
        GV = ((hv64 * vss).sum(axis=0) - (hv64 * v).sum(axis=0)) / dt
        GU = xp.where(umb, GU, 0.0)
        GV = xp.where(vmb, GV, 0.0)
        if getattr(self, "keep_debug", False):
            self.debug = {"GU": GU, "GV": GV, "U0": U0, "V0": V0, "Du": Du, "Dv": Dv}
        M = self.nsub
        dtb = dt / M
        U, V = U0, V0
        Usum = xp.zeros_like(U0)
        Vsum = xp.zeros_like(V0)
        em = eta
        area, dxv64, dyu64 = g["area"], g["dxv"], g["dyu"]
        for m in range(M):
            Usum = Usum + U
            Vsum = Vsum + V
            Uf = U * dyu64
            Vf = V * dxv64
            div = (Uf - wx(Uf) + Vf - sj(Vf)) / area
            en = xp.where(wet, em - dtb * div, 0.0)
            ep = en + p.bebt * (en - em)
            if m % 2 == 0:
                U = xp.where(umb, U + dtb * (-GRAVITY * Du * (ex(ep) - ep) / g["dxu"]
                                             + vbar(V) + GU), 0.0)
                V = xp.where(vmb, V + dtb * (-GRAVITY * Dv * (nj(ep) - ep) / g["dyv"]
                                             - ubar(U) + GV), 0.0)
            else:
                V = xp.where(vmb, V + dtb * (-GRAVITY * Dv * (nj(ep) - ep) / g["dyv"]
                                             - ubar(U) + GV), 0.0)
                U = xp.where(umb, U + dtb * (-GRAVITY * Du * (ex(ep) - ep) / g["dxu"]
                                             + vbar(V) + GU), 0.0)
            em = en
        Um = Usum / M
        Vm = Vsum / M
        Uf = Um * dyu64
        Vf = Vm * dxv64
        div = (Uf - wx(Uf) + Vf - sj(Vf)) / area
        eta_new = xp.where(wet, eta - dt * div, 0.0)
        s_new = xp.where(wet, 1.0 + eta_new / Hs, 1.0)
        su_n = 0.5 * (s_new + ex(s_new))
        sv_n = 0.5 * (s_new + nj(s_new))
        hu_n = g["dzu"].astype(xp.float64) * su_n[None]
        hv_n = g["dzv"].astype(xp.float64) * sv_n[None]
        Du_n = xp.where(umb, hu_n.sum(axis=0), 1.0)
        Dv_n = xp.where(vmb, hv_n.sum(axis=0), 1.0)
        corr_u = (U - (hu_n * uss).sum(axis=0)) / Du_n
        corr_v = (V - (hv_n * vss).sum(axis=0)) / Dv_n
        u_new = xp.where(um, uss + corr_u.astype(f32)[None], f32(0.0))
        v_new = xp.where(vm, vss + corr_v.astype(f32)[None], f32(0.0))
        del uss, vss, hu_n, hv_n

        # tracer transports consistent with <U>
        Du_s = xp.where(umb, Du, 1.0)
        Dv_s = xp.where(vmb, Dv, 1.0)
        un64 = u_new.astype(xp.float64)
        vn64 = v_new.astype(xp.float64)
        ubar_n = (hu64 * un64).sum(axis=0) / Du_s
        vbar_n = (hv64 * vn64).sum(axis=0) / Dv_s
        uh = xp.where(um, hu64 * (un64 - ubar_n[None]) + hu64 * (Um / Du_s)[None], 0.0)
        vh = xp.where(vm, hv64 * (vn64 - vbar_n[None]) + hv64 * (Vm / Dv_s)[None], 0.0)
        del un64, vn64

        # 5. GM
        if p.gm and p.gm_kappa > 0:
            uh = uh + self._gm(rhop, r_t, r_s, theta, salt, n2, hu64, um, "x")
            vh = vh + self._gm(rhop, r_t, r_s, theta, salt, n2, hv64, vm, "y")
        del rhop, r_t, r_s, n2, hu64, hv64

        # 6. tracers
        h_new = g["dzt"] * s_new[None]
        vol0 = h * area
        Tx = uh * dyu64
        del uh
        Ty = vh * dxv64
        del vh
        divh = (Tx - wx(Tx) + Ty - sj(Ty)) / area
        dh = (h_new - h) / dt
        # W at the bottom interface of each level, upward, from the floor up.
        wb = xp.zeros_like(h)
        acc = xp.zeros_like(eta)
        for k in range(nz - 1, -1, -1):
            wb[k] = acc
            acc = xp.where(tm[k], acc - divh[k] - dh[k], 0.0)
        w_store = wb.astype(f32)
        vol_res = acc  # surface residual (rounding)
        # Adaptive implicit vertical advection: up to a Courant number of
        # VERTICAL_COURANT_MAX through an interface the transport is
        # advected explicitly (flux-limited, below); the excess is implicit
        # upwind in the vertical solve, so steep topography under strong
        # flow (w dt / h of several) stays stable.  After Shchepetkin
        # (2015), in the simplest split.
        h_below = xp.concatenate([h_new[1:], xp.zeros_like(h_new[:1])])
        thin = xp.minimum(h_new, h_below)
        cour = xp.abs(wb) * dt / xp.maximum(thin, 1.0e-6)
        frac = xp.where(cour > VERTICAL_COURANT_MAX, VERTICAL_COURANT_MAX / xp.maximum(cour, 1e-30), 1.0)
        w_imp = wb * (1.0 - frac)
        Tz = wb * frac * area
        self.last_vertical_courant_max = cour.max()
        if not p.tracer_advection:
            w_imp = xp.zeros_like(w_imp)
        del divh, dh, wb, h, h_below, thin, frac, cour
        if p.tracer_advection:
            order = int(round(float(st["books"][0]))) % 2
            theta_a, salt_a, vol_a = self._advect(theta, salt, vol0, Tx, Ty, Tz, um, vm, tm, order)
        else:
            theta_a, salt_a, vol_a = theta, salt, vol0
        del Tx, Ty, Tz, vol0
        vol_n = h_new * area
        vol_a = xp.where(tm, vol_a, 1.0)
        # horizontal diffusion
        if p.kappa_h > 0:
            hu_n = g["dzu"].astype(xp.float64) * su_n[None]
            hv_n = g["dzv"].astype(xp.float64) * sv_n[None]
            heat_th = theta_a * vol_a
            heat_s = salt_a * vol_a
            for name in ("theta", "salt"):
                q = theta_a if name == "theta" else salt_a
                fxd = -p.kappa_h * (hu_n * dyu64) * (ex(q) - q) / g["dxu"]
                fxd = xp.where(um, fxd, 0.0)
                fyd = -p.kappa_h * (hv_n * dxv64) * (nj(q) - q) / g["dyv"]
                fyd = xp.where(vm, fyd, 0.0)
                d = -dt * (fxd - wx(fxd) + fyd - sj(fyd))
                if name == "theta":
                    heat_th = heat_th + d
                else:
                    heat_s = heat_s + d
            theta_a = xp.where(tm, heat_th / vol_a, 0.0)
            salt_a = xp.where(tm, heat_s / vol_a, 0.0)
            del heat_th, heat_s, fxd, fyd, d, q, hu_n, hv_n
        hh = xp.where(tm, vol_a / area, 0.0)
        del vol_a
        # surface heat: non-SW in the top cell, SW by the two bands
        rc = RHO0 * CP0
        top_w = xp.concatenate([xp.ones_like(sw_w[:1]), sw_w[:-1]])
        kmt = g["kmt"]
        is_bottom = (xp.arange(nz).reshape(-1, 1, 1) == (kmt - 1)[None]) & tm
        sw_bot = xp.where(is_bottom, 0.0, sw_w)
        sw_cell = xp.where(tm, top_w - sw_bot, 0.0)
        src = q_sw[None] * sw_cell
        src[0] = src[0] + q_ns
        # KPP non-local flux of the non-penetrating heat and the salt flux
        nl = mix["nonlocal"]
        nl_up = xp.concatenate([xp.zeros_like(nl[:1]), nl[:-1]])
        src = src + q_ns[None] * (nl_up - nl)
        heat = hh * theta_a + dt * src / rc
        ssrc = sflux[None] * (nl_up - nl)
        ssrc[0] = ssrc[0] + sflux
        sal = hh * salt_a + dt * ssrc
        hsafe = xp.where(tm, hh, 1.0)
        theta_b = xp.where(tm, heat / hsafe, 0.0)
        salt_b = xp.where(tm, sal / hsafe, 0.0)
        del heat, sal, src, ssrc, nl, nl_up, sw_cell, sw_bot, top_w, is_bottom, theta_a, salt_a, hsafe
        mix.pop("nonlocal", None)
        # the implicit part of the vertical transport changes h_mid to h_new
        w_imp_top = xp.concatenate([xp.zeros_like(w_imp[:1]), w_imp[:-1]])
        hh_new = xp.where(tm, hh + dt * (w_imp - w_imp_top), 0.0)
        del w_imp_top
        theta_c = by_rows(implicit_diffusion, self.chunks, ny, xp, theta_b, hh_new, mix["kappa_s"],
                          dt, xp, w_flux=w_imp, h_old=hh)
        del theta_b
        salt_c = by_rows(implicit_diffusion, self.chunks, ny, xp, salt_b, hh_new, mix["kappa_s"],
                         dt, xp, w_flux=w_imp, h_old=hh)
        del salt_b, w_imp
        hh = hh_new
        del hh_new
        # sponge, SSS restoring, SST restoring
        sponge_heat = 0.0
        restoring_heat = 0.0
        sss_salt = 0.0
        sponge_salt = 0.0
        tg = self.targets or {}
        if p.sponge and self.sponge_rows.size and "sponge" in tg:
            t_star, s_star = tg["sponge"](fc.get("time_s", 0.0))
            rows = slice(int(self.sponge_rows[0]), int(self.sponge_rows[-1]) + 1)
            lam = dt * self.sponge_rate[rows]
            th_r = theta_c[:, rows]
            sa_r = salt_c[:, rows]
            th_new = (th_r + lam * t_star) / (1.0 + lam)
            sa_new = (sa_r + lam * s_star) / (1.0 + lam)
            mk = tm[:, rows]
            th_new = xp.where(mk, th_new, 0.0)
            sa_new = xp.where(mk, sa_new, 0.0)
            vr = vol_n[:, rows]
            sponge_heat = float(((th_new - th_r) * vr).sum()) * rc
            sponge_salt = float(((sa_new - sa_r) * vr).sum())
            theta_c = theta_c.copy()
            salt_c = salt_c.copy()
            theta_c[:, rows] = th_new
            salt_c[:, rows] = sa_new
        if p.sss_piston_m_per_day and "sss" in tg:
            s_star = tg["sss"](fc.get("time_s", 0.0))
            rate = p.sss_piston_m_per_day / 86400.0
            h1 = xp.where(wet, hh[0], 1.0)
            new = xp.where(wet, (salt_c[0] + dt * rate / h1 * s_star) / (1.0 + dt * rate / h1), 0.0)
            sss_salt = float(((new - salt_c[0]) * vol_n[0]).sum())
            salt_c = salt_c.copy()
            salt_c[0] = new
        if p.sst_restoring_days and "sst_target_c" in fc:
            lam = dt / (p.sst_restoring_days * 86400.0)
            t_star = fc["sst_target_c"]
            new = xp.where(wet, (theta_c[0] + lam * t_star) / (1.0 + lam), 0.0)
            restoring_heat = float(((new - theta_c[0]) * vol_n[0]).sum()) * rc
            theta_c = theta_c.copy()
            theta_c[0] = new
        # frazil
        cold = wet & (theta_c[0] < FREEZING_C)
        frazil = xp.where(cold, rc * hh[0] * (FREEZING_C - theta_c[0]), 0.0)
        theta_c = theta_c.copy()
        theta_c[0] = xp.where(cold, FREEZING_C, theta_c[0])
        # F_bot potential (W per m2 of ice): rho0 cp c_h u*_io (theta_1 - T_f)
        ustar_io = xp.maximum(math.sqrt(p.c_io) * u1, p.ustar_io_min)
        f_bot = xp.where(wet, rc * p.ch_io * ustar_io * xp.maximum(theta_c[0] - FREEZING_C, 0.0), 0.0)

        # books
        b = st["books"]
        bn = b.copy()
        ii = BOOKS.index
        surface_heat = float(((q_ns + q_sw) * area2).sum()) * dt
        frazil_j = float((frazil * area2).sum())
        bn[ii("steps")] = b[ii("steps")] + 1
        bn[ii("ab_steps")] = b[ii("ab_steps")] + 1
        bn[ii("heat_sources_j")] = b[ii("heat_sources_j")] + surface_heat + frazil_j + sponge_heat + restoring_heat
        bn[ii("frazil_j")] = b[ii("frazil_j")] + frazil_j
        bn[ii("sponge_heat_j")] = b[ii("sponge_heat_j")] + sponge_heat
        bn[ii("restoring_heat_j")] = b[ii("restoring_heat_j")] + restoring_heat
        bn[ii("salt_sources")] = (b[ii("salt_sources")] + float((sflux * area2).sum()) * dt + sss_salt
                                  + sponge_salt)
        bn[ii("sponge_salt")] = b[ii("sponge_salt")] + sponge_salt
        bn[ii("sss_salt")] = b[ii("sss_salt")] + sss_salt
        bn[ii("salt_flux_removed")] = b[ii("salt_flux_removed")] + float(mean_s) * float(area2.sum()) * dt
        new = {
            "theta": xp.where(tm, theta_c, 0.0), "salt": xp.where(tm, salt_c, 0.0),
            "u": u_new, "v": v_new, "gu1": Gu, "gu2": st["gu1"], "gv1": Gv, "gv2": st["gv1"],
            "w": w_store, "eta": eta_new, "ubt": U, "vbt": V,
            "hbl": mix["hbl"].astype(f32), "rho_ref": st["rho_ref"], "books": bn,
        }
        diag = {"frazil_j_m2": frazil, "f_bot_w_m2": f_bot, "volume_residual": vol_res}
        return new, diag

    # -- viscosity -------------------------------------------------------------
    def _viscosity(self, u, v, um, vm, tm, qmask):
        xp, g32, p = self.xp, self.g32, self.p
        f32 = xp.float32
        ex, wx, nj, sj = self.ex, self.wx, self.nj, self.sj
        dxt, dyt, dxu, dyu, dxv, dyv = (g32[k] for k in ("dxt", "dyt", "dxu", "dyu", "dxv", "dyv"))
        ax = self.ax_static
        ay = self.ay_static
        if p.visc_smag > 0:
            dT = (u - wx(u)) / dxt - (v - sj(v)) / dyt
            dS = xp.where(qmask, (nj(u) - u) / dyv + (ex(v) - v) / dxv, f32(0.0))
            dS2 = f32(0.25) * (dS * dS + wx(dS * dS) + sj(dS * dS) + wx(sj(dS * dS)))
            amag = f32((p.visc_smag / math.pi) ** 2) * dxt * dyt * xp.sqrt(dT * dT + dS2)
            # Smagorinsky acts on the zonal stress only and at most a third of
            # the forward-Euler limit: at the full limit and in both directions
            # it fed a two-step oscillation of a 30 m shelf cell at the Amazon
            # mouth (the GODAS spin-up blew up on day 10), and in y it would
            # override the 2e3 m2 s-1 the equatorial waves need.
            amag = xp.minimum(amag, f32(1.0 / 3.0) * self.visc_lim)
            axT = xp.maximum(ax, amag)
            ayT = xp.broadcast_to(ay, u.shape)
        else:
            axT = xp.broadcast_to(ax, u.shape)
            ayT = xp.broadcast_to(ay, u.shape)
        axq = f32(0.25) * (axT + ex(axT) + nj(axT) + nj(ex(axT)))
        ayq = f32(0.25) * (ayT + ex(ayT) + nj(ayT) + nj(ex(ayT)))
        fx = xp.where(tm, axT * (u - wx(u)) / dxt, f32(0.0))
        fy = xp.where(um & nj(um), ayq * (nj(u) - u) / dyv, f32(0.0))
        vu = (ex(fx) - fx) / dxu + (fy - sj(fy)) / dyu
        gx = xp.where(vm & ex(vm), axq * (ex(v) - v) / dxv, f32(0.0))
        gy = xp.where(tm, ayT * (v - sj(v)) / dyt, f32(0.0))
        vv = (gx - wx(gx)) / dxv + (nj(gy) - gy) / dyv
        return xp.where(um, vu, f32(0.0)), xp.where(vm, vv, f32(0.0))

    # -- GM --------------------------------------------------------------------
    def _gm(self, rhop, r_t, r_s, theta, salt, n2, hf, fmask, direction):
        """Eddy-induced transport per unit width through each face level
        (m2 s-1), Ferrari et al. (2010): (c^2 d_zz - N^2) psi = kappa (g/rho0)
        d_x rho on interfaces, psi = 0 at the surface and the floor."""
        # A parameterisation: worked in float32 (memory), psi widened to
        # float64 before it is differenced, so each column's eddy transport
        # sums to zero exactly (differences of float32 values are exact in
        # float64 and their running sums telescope).
        xp, p, g = self.xp, self.p, self.g
        f32 = xp.float32
        if direction == "x":
            nb = self.ex
            dist = g["dxu"].astype(f32)
        else:
            nb = self.nj
            dist = g["dyv"].astype(f32)
        # locally referenced horizontal density gradient at levels
        rt = (0.5 * (r_t + nb(r_t))).astype(f32)
        drho = rt * (nb(theta) - theta).astype(f32)
        del rt
        rs = (0.5 * (r_s + nb(r_s))).astype(f32)
        drho = drho + rs * (nb(salt) - salt).astype(f32)
        del rs
        drho = xp.where(fmask, drho / dist, f32(0.0))
        # at interfaces (below level k), defined where level k+1 is wet at the face
        fb = xp.concatenate([fmask[1:], xp.zeros_like(fmask[:1])])
        rhs = xp.where(fb, f32(0.5 * p.gm_kappa * GRAVITY / RHO0)
                       * (drho + xp.concatenate([drho[1:], drho[-1:]])), f32(0.0))
        del drho
        n2f = xp.where(fb, xp.maximum(0.5 * (n2 + nb(n2)), 0.0).astype(f32), f32(0.0))
        hf32 = hf.astype(f32)
        hb = xp.concatenate([hf32[1:], xp.zeros_like(hf32[:1])])
        ds = xp.where(fb, f32(0.5) * (hf32 + hb), f32(1.0))
        c = xp.clip((xp.sqrt(n2f) * xp.where(fb, ds, f32(0.0))).sum(axis=0) / f32(math.pi),
                    f32(0.1), f32(3.0))
        c2 = (c * c)[None]
        a = xp.where(fb, c2 / (xp.where(fmask, hf32, f32(1.0)) * ds), f32(0.0))
        del hf32
        cc = xp.where(fb, c2 / (xp.where(fb, hb, f32(1.0)) * ds), f32(0.0))
        del hb, ds
        b = xp.where(fb, -a - cc - n2f, f32(1.0))
        del n2f
        a[0] = 0.0
        psi = by_rows(thomas, self.chunks, self.grid.ny, xp, a, b, cc, rhs, xp)
        del a, b, cc, rhs
        lim = f32(p.gm_kappa * p.gm_slope_max)
        psi = xp.where(fb, xp.clip(psi, -lim, lim), f32(0.0)).astype(xp.float64)
        psi_top = xp.concatenate([xp.zeros_like(psi[:1]), psi[:-1]])
        return xp.where(fmask, psi_top - psi, 0.0)

    # -- advection -------------------------------------------------------------
    @staticmethod
    def _mc(r, xp):
        f = r.dtype.type
        return xp.maximum(f(0.0), xp.minimum(xp.minimum(f(2.0) * r, f(0.5) * (f(1.0) + r)), f(2.0)))

    def _face_flux(self, q, T, vol, nb, nb2, pb, fmask, fmask_prev, fmask_next, dt):
        """Flux-limited flux through the faces between a cell and its
        neighbour ``nb`` (T > 0 toward the neighbour).  The limiter's ratio
        and Courant number are float32 (the flux is a single value used by
        both cells, so conservation does not depend on its precision); the
        face value is float64."""
        xp = self.xp
        f32 = xp.float32
        pos = T >= 0.0
        qR = nb(q)
        up = xp.where(pos, q, qR)
        dn = xp.where(pos, qR, q)
        uu = xp.where(pos, xp.where(fmask_prev, pb(q), q), xp.where(fmask_next, nb2(q), qR))
        del qR
        dd = dn - up
        del dn
        nz_ = xp.abs(dd) > 0
        r = ((up - uu) / xp.where(nz_, dd, 1.0)).astype(f32)
        del uu
        vup = xp.where(pos, vol, nb(vol))
        c = xp.minimum((xp.abs(T) * dt / xp.maximum(vup, 1.0e-30)).astype(f32), f32(1.0))
        del vup, pos
        lim = xp.where(nz_, self._mc(r, xp), f32(0.0))
        del r
        corr = (f32(0.5) * (f32(1.0) - c) * lim).astype(xp.float64)
        del c, lim
        face = up + corr * dd
        del up, corr, dd
        return xp.where(fmask, T * face, 0.0)

    def _advect(self, theta, salt, vol, Tx, Ty, Tz, um, vm, tm, order):
        xp = self.xp
        dt = self.p.dt
        ex, wx, nj, sj = self.ex, self.wx, self.nj, self.sj
        nz = theta.shape[0]

        def up_k(a):  # value of level k+1 at level k (below)
            return xp.concatenate([a[1:], xp.zeros_like(a[:1])])

        def down_k(a):  # value of level k-1 at level k
            return xp.concatenate([xp.zeros_like(a[:1]), a[:-1]])

        tm_below = up_k(tm)
        face_z = tm & tm_below
        # Unsplit: every face flux from the start-of-step field and volume,
        # one update.  (Dimensional splitting with intermediate volumes was
        # the first form: in a strait cell where strong horizontal
        # convergence is balanced by vertical outflow the x sweep alone can
        # drain the cell, and the GODAS spin-up went non-finite on day 3.)
        # The new volume is h_new * area to rounding, never an intermediate.
        sweeps = (
            (Tx, wx, (ex, lambda a: ex(ex(a)), wx, um, wx(um), ex(um))),
            (Ty, sj, (nj, lambda a: nj(nj(a)), sj, vm, sj(vm), nj(vm))),
            # interface below level k; _face_flux takes T > 0 toward the
            # level below, so the upward transport enters negated.
            (-Tz, down_k, (up_k, lambda a: up_k(up_k(a)), down_k, face_z, down_k(face_z),
                           up_k(face_z))),
        )
        v = xp.where(tm, vol, 0.0)
        vsafe = xp.where(tm, vol, 1.0e-30)
        dv = xp.zeros_like(v)
        for T, back, _ in sweeps:
            dv = dv + (T - back(T))
        v_new = v - dt * dv
        del dv
        safe = xp.where(tm, v_new, 1.0)
        out = []
        for q in (theta, salt):   # one tracer at a time (memory)
            acc = xp.zeros_like(q)
            for T, back, args in sweeps:
                f = self._face_flux(q, T, vsafe, *args, dt)
                acc = acc + (f - back(f))
                del f
            out.append(xp.where(tm, (q * v - dt * acc) / safe, 0.0))
            del acc
        return out[0], out[1], xp.where(tm, v_new, 0.0)


def vertical_modes(grid, n2_profile, nmodes=2):
    """Phase speeds of the first baroclinic modes on the grid's own levels for
    an N^2 profile at the interfaces (flat bottom, rigid lid): the
    eigenproblem d/dz (1/N^2 dp/dz) = -p / c^2 in finite differences."""
    dz = np.diff(grid.z_edges_m)
    nz = dz.size
    n2 = np.asarray(n2_profile, dtype=np.float64)[: nz - 1]
    dzi = 0.5 * (dz[:-1] + dz[1:])
    A = np.zeros((nz, nz))
    for k in range(nz):
        if k > 0:
            c_up = 1.0 / (n2[k - 1] * dzi[k - 1] * dz[k])
            A[k, k - 1] += c_up
            A[k, k] -= c_up
        if k < nz - 1:
            c_dn = 1.0 / (n2[k] * dzi[k] * dz[k])
            A[k, k + 1] += c_dn
            A[k, k] -= c_dn
    lam = np.sort(np.linalg.eigvals(A).real)[::-1]
    lam = lam[lam < -1e-14]
    return [float(1.0 / math.sqrt(-x)) for x in lam[:nmodes]]


__all__ = ["BOOKS", "FREEZING_C", "KELVIN", "OceanModel", "OceanParams", "STATE_ARRAYS",
           "STATE_DTYPES", "vertical_modes"]
