"""Equation of state: Wright (1997), J. Atmos. Oceanic Technol. 14, 735-740,
the full-range rational fit in potential temperature (deg C), practical
salinity and pressure (Pa):

    rho = (p + p0(T, S)) / (lambda(T, S) + alpha0(T, S) (p + p0(T, S)))

with alpha0 = a0 + a1 T + a2 S, p0 = b0 + b1 T + b2 T^2 + b3 T^3 + b4 S +
b5 T S and lambda = c0 + c1 T + c2 T^2 + c3 T^3 + c4 S + c5 T S (the paper's
full-range coefficients, its Table 1).  The pressure is the Boussinesq
reference pressure rho0 g z of the nominal level depth.  Analytic
d rho / d T and d rho / d S for KPP's buoyancy and GM's slopes.
"""
from __future__ import annotations

A0, A1, A2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
B0, B1, B2, B3, B4, B5 = 5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3
C0, C1, C2, C3, C4, C5 = 1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464


def density(t_c, s, p_pa, xp=None):
    """In-situ density (kg m-3) from potential temperature (deg C), salinity
    and pressure (Pa)."""
    alpha0 = A0 + A1 * t_c + A2 * s
    p0 = B0 + t_c * (B1 + t_c * (B2 + B3 * t_c)) + s * (B4 + B5 * t_c)
    lam = C0 + t_c * (C1 + t_c * (C2 + C3 * t_c)) + s * (C4 + C5 * t_c)
    pp = p_pa + p0
    return pp / (lam + alpha0 * pp)


def density_derivatives(t_c, s, p_pa, xp=None):
    """``(rho, d rho/d T, d rho/d S)``."""
    alpha0 = A0 + A1 * t_c + A2 * s
    p0 = B0 + t_c * (B1 + t_c * (B2 + B3 * t_c)) + s * (B4 + B5 * t_c)
    lam = C0 + t_c * (C1 + t_c * (C2 + C3 * t_c)) + s * (C4 + C5 * t_c)
    pp = p_pa + p0
    den = lam + alpha0 * pp
    rho = pp / den
    p0_t = B1 + t_c * (2.0 * B2 + 3.0 * B3 * t_c) + B5 * s
    p0_s = B4 + B5 * t_c
    lam_t = C1 + t_c * (2.0 * C2 + 3.0 * C3 * t_c) + C5 * s
    lam_s = C4 + C5 * t_c
    den_t = lam_t + A1 * pp + alpha0 * p0_t
    den_s = lam_s + A2 * pp + alpha0 * p0_s
    inv2 = 1.0 / (den * den)
    return rho, (p0_t * den - pp * den_t) * inv2, (p0_s * den - pp * den_s) * inv2


__all__ = ["density", "density_derivatives"]
