"""The K-profile parameterisation (Large, McWilliams and Doney 1994,
Rev. Geophys. 32, 363; "LMD94") with its interior scheme.

* Boundary-layer depth h: the shallowest depth at which the bulk Richardson
  number Ri_b(d) = (B_r - B(d)) d / (|V_r - V(d)|^2 + V_t^2(d)) reaches
  Ri_c = 0.3, interpolated between level centres; the reference is the top
  cell; V_t^2 the unresolved-shear term (LMD94 eq. 23, C_v = 1.6,
  beta_T = -0.2); in stabilising forcing h is limited by the Ekman depth
  0.7 u*/f and the Monin-Obukhov length.
* Inside h: K_x(sigma) = h w_x(sigma) G(sigma), the turbulent velocity
  scales w_m, w_s from the stability functions phi_m, phi_s (LMD94 B1-B2),
  sigma capped at epsilon = 0.1 in unstable forcing, and G a cubic with
  G(0) = 0, G'(0) = 1 matched in value to the interior diffusivity at h
  (the derivative match is dropped, a common simplification, stated).
* Non-local flux of heat and salt in unstable forcing: C_s G0(sigma) times
  the surface flux, C_s = C* kappa (c_s kappa epsilon)^(1/3) and
  G0 = sigma (1 - sigma)^2.
* Interior: shear mixing nu = nu0 (1 - (Ri/Ri0)^2)^3 (Ri0 = 0.7, nu0 =
  5e-3 m2 s-1), convection 0.1 m2 s-1 where N^2 < 0, background
  diffusivity 1e-5 m2 s-1 (1e-6 within 2 deg of the equator) and viscosity
  1e-4 m2 s-1.  No Langmuir term, tidal mixing or double diffusion
  (stage 2).
"""
from __future__ import annotations

from .grid import GRAVITY, RHO0

VK = 0.4
RIC = 0.3
EPS = 0.1
CV = 1.6
BETA_T = -0.2
ZETA_M, A_M, C_M = -0.2, 1.26, 8.38
ZETA_S, A_S, C_S = -1.0, -28.86, 98.96
CSTAR = 10.0
NU0 = 5.0e-3
RI0 = 0.7
KAPPA_CONV = 0.1
KAPPA_BG = 1.0e-5
KAPPA_BG_EQ = 1.0e-6
NU_BG = 1.0e-4
USTAR_MIN = 1.0e-5
#: C_s = C* kappa (c_s kappa eps)^(1/3) (LMD94 eq. 20).
CS_NONLOCAL = CSTAR * VK * (C_S * VK * EPS) ** (1.0 / 3.0)
#: V_t^2 prefactor: C_v sqrt(-beta_T) / (Ri_c kappa^2) (c_s eps)^(-1/2).
VT_COEFF = CV * (-BETA_T) ** 0.5 / (RIC * VK * VK) * (C_S * EPS) ** -0.5


def _cbrt(x, xp):
    return xp.sign(x) * xp.abs(x) ** (1.0 / 3.0)


def velocity_scales(sigma_h, ustar, bflux, xp):
    """``(w_m, w_s)`` at depth sigma*h (m) for friction velocity ``ustar``
    and surface buoyancy input ``bflux`` (m2 s-3, > 0 stabilising)."""
    u3 = ustar ** 3
    # zeta = d / L = d kappa B / u*^3, written so u* -> 0 stays finite.
    kb = VK * sigma_h * bflux
    us = xp.maximum(ustar, USTAR_MIN)
    zeta = kb / (us ** 3)
    stable = zeta >= 0.0
    ws_stable = VK * ustar / (1.0 + 5.0 * xp.where(stable, zeta, 0.0))
    ws_mid = VK * ustar * xp.sqrt(xp.maximum(1.0 - 16.0 * zeta, 1.0))
    ws_conv = VK * _cbrt(A_S * u3 - C_S * kb, xp)
    w_s = xp.where(stable, ws_stable, xp.where(zeta > ZETA_S, ws_mid, ws_conv))
    wm_mid = VK * ustar * xp.maximum(1.0 - 16.0 * zeta, 1.0) ** 0.25
    wm_conv = VK * _cbrt(A_M * u3 - C_M * kb, xp)
    w_m = xp.where(stable, ws_stable, xp.where(zeta > ZETA_M, wm_mid, wm_conv))
    return xp.maximum(w_m, 1.0e-10), xp.maximum(w_s, 1.0e-10)


def interior(n2, shear2, lat_abs, xp):
    """``(kappa, nu)`` at interfaces from N^2 and |dV/dz|^2."""
    ri = n2 / xp.maximum(shear2, 1.0e-14)
    frac = xp.clip(ri / RI0, 0.0, 1.0)
    nu_sh = NU0 * (1.0 - frac * frac) ** 3
    nu_sh = xp.where(ri <= 0.0, NU0, nu_sh)
    bg = xp.where(lat_abs < 2.0, KAPPA_BG_EQ, KAPPA_BG)
    conv = n2 < 0.0
    kappa = bg + nu_sh + xp.where(conv, KAPPA_CONV, 0.0)
    nu = NU_BG + nu_sh + xp.where(conv, KAPPA_CONV, 0.0)
    return kappa, nu


def kpp(*, theta, salt, uT, vT, h, depth_t, depth_w, n2, drho_dt, drho_ds, ustar,
        q_ns, q_sw, salt_flux, sw_frac_w, f_abs, lat_abs, wet, xp):
    """The KPP diffusivities.

    ``h`` (nz, ny, nx) cell thicknesses (0 below the floor); ``depth_t`` the
    centre depths and ``depth_w`` the depths of the interface below each
    level (positive down); ``n2`` (nz, ...) N^2 at the interface below each
    level; ``drho_dt``/``drho_ds`` at the level centres; ``q_ns``/``q_sw``
    (W m-2 into the ocean), ``salt_flux`` (psu m s-1 into the ocean);
    ``sw_frac_w`` the shortwave fraction reaching each interface.

    Returns ``dict(kappa_s, kappa_m, nonlocal, hbl)``: interface
    diffusivities for scalars and momentum, the non-local fraction at each
    interface (apply to the non-penetrating surface flux, unstable forcing
    only) and the boundary-layer depth."""
    from .grid import CP0

    g = GRAVITY
    nz = theta.shape[0]
    uT = xp.asarray(uT, dtype=xp.float64)
    vT = xp.asarray(vT, dtype=xp.float64)
    # Buoyancy relative to the top cell, at each level's pressure:
    # B_r - B_k = g / rho0 (rho_k - rho(theta_r, S_r)) ~ (g/rho0)
    # (d rho/dT (theta_r - theta_k) ... ) with the local derivatives.
    dB = -(g / RHO0) * (drho_dt * (theta[0:1] - theta) + drho_ds * (salt[0:1] - salt))
    dV2 = (uT[0:1] - uT) ** 2 + (vT[0:1] - vT) ** 2
    alpha_top = -drho_dt[0] / RHO0
    beta_top = drho_ds[0] / RHO0
    b_ns = g * (alpha_top * q_ns / (RHO0 * CP0) - beta_top * salt_flux)
    b_sw = g * alpha_top * q_sw / (RHO0 * CP0)
    # SW absorbed above each centre (linear between interfaces).
    frac_above_w = 1.0 - sw_frac_w
    frac_top_w = xp.concatenate([xp.zeros_like(frac_above_w[:1]), frac_above_w[:-1]])
    absorbed_t = 0.5 * (frac_top_w + frac_above_w)
    bflux_t = b_ns + b_sw * absorbed_t
    _, ws_t = velocity_scales(EPS * depth_t, ustar, bflux_t, xp)
    n_t = xp.sqrt(xp.maximum(0.5 * (n2 + xp.concatenate([n2[:1], n2[:-1]])), 0.0))
    vt2 = VT_COEFF * depth_t * n_t * ws_t
    rib = dB * depth_t / xp.maximum(dV2 + vt2, 1.0e-14)
    wet_t = h > 0.0
    rib = xp.where(wet_t, rib, 1.0e9)
    rib[0] = 0.0
    over = rib > RIC
    kmt = wet_t.sum(axis=0)
    first = xp.argmax(over, axis=0)
    found = over.any(axis=0)
    kk = xp.clip(first, 1, nz - 1)
    take = lambda a, idx: xp.take_along_axis(a, idx[None], axis=0)[0]
    r1 = take(rib, kk)
    r0 = take(rib, kk - 1)
    d1 = take(depth_t, kk)
    d0 = take(depth_t, kk - 1)
    t = xp.clip((RIC - r0) / xp.where(r1 - r0 > 0, r1 - r0, 1.0), 0.0, 1.0)
    hbl = d0 + t * (d1 - d0)
    bottom = take(depth_w, xp.clip(kmt - 1, 0, nz - 1))
    hbl = xp.where(found, hbl, bottom)
    # Stable limits.
    bfl = b_ns + b_sw * (1.0 - take(sw_frac_w, xp.clip(first, 0, nz - 1)))
    us = xp.maximum(ustar, USTAR_MIN)
    ekman = 0.7 * us / xp.maximum(f_abs, 1.0e-10)
    mo = us ** 3 / (VK * xp.maximum(bfl, 1.0e-30))
    hbl = xp.where(bfl > 0.0, xp.minimum(hbl, xp.minimum(ekman, mo)), hbl)
    hbl = xp.clip(hbl, xp.minimum(depth_t[0], bottom), bottom)
    hbl = xp.where(wet, xp.maximum(hbl, 1.0e-3), 1.0)
    # Interior mixing.
    kap_i, nu_i = interior(n2, _shear2(uT, vT, h, xp), lat_abs, xp)
    sigma = depth_w / hbl[None]
    inside = (sigma < 1.0) & wet_t
    unstable = bfl < 0.0
    sig_ws = xp.where(unstable[None], xp.minimum(sigma, EPS), sigma) * hbl[None]
    wm, ws = velocity_scales(sig_ws, ustar[None], bfl[None], xp)
    wm1, ws1 = velocity_scales(xp.where(unstable, EPS, 1.0) * hbl, ustar, bfl, xp)
    # Interior values at h (interface just below or at h).
    kh = xp.clip(xp.argmax(depth_w >= hbl[None], axis=0), 0, nz - 1)
    kap_h = take(kap_i, kh)
    nu_h = take(nu_i, kh)

    def shape(g1, s):
        a2 = -2.0 + 3.0 * g1
        a3 = 1.0 - 2.0 * g1
        return s * (1.0 + s * (a2[None] + a3[None] * s))

    gs1 = xp.clip(kap_h / (hbl * ws1), 0.0, 1.0)
    gm1 = xp.clip(nu_h / (hbl * wm1), 0.0, 1.0)
    s = xp.clip(sigma, 0.0, 1.0)
    # Inside h the boundary-layer profile, never below the interior value
    # (convection inside the layer keeps its 0.1 m2 s-1).
    k_s = xp.where(inside, xp.maximum(hbl[None] * ws * shape(gs1, s), kap_i), kap_i)
    k_m = xp.where(inside, xp.maximum(hbl[None] * wm * shape(gm1, s), nu_i), nu_i)
    nonlocal_ = xp.where(inside & unstable[None], CS_NONLOCAL * s * (1.0 - s) ** 2, 0.0)
    below = xp.zeros_like(h)
    below[:-1] = h[1:]
    iface = wet_t & (below > 0.0)
    return {"kappa_s": xp.where(iface, k_s, 0.0), "kappa_m": xp.where(iface, k_m, 0.0),
            "nonlocal": xp.where(iface, nonlocal_, 0.0), "hbl": xp.where(wet, hbl, 0.0)}


def _shear2(uT, vT, h, xp):
    below = xp.zeros_like(h)
    below[:-1] = h[1:]
    delta = 0.5 * (h + below)
    du = xp.zeros_like(uT)
    dv = xp.zeros_like(vT)
    du[:-1] = uT[:-1] - uT[1:]
    dv[:-1] = vT[:-1] - vT[1:]
    return (du * du + dv * dv) / xp.maximum(delta, 1.0e-3) ** 2


__all__ = ["CS_NONLOCAL", "interior", "kpp", "velocity_scales"]
