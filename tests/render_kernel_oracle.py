"""The NumPy oracle of the Rust render kernels.

Each function here is the NumPy implementation that ran on the default
render and regional paths of WOOF Global at 7548983, kept verbatim in its
arithmetic so the Rust kernels (`arwen_global.render_kernels`) can be held
to it byte for byte.  Nothing in the package imports this module: it is
the test oracle and nothing else.

``libm`` mode.  NumPy's ``exp``, ``log`` and ``power`` dispatch on the
CPU: from the C library on an x86-64 machine without AVX-512, from NumPy's
own vector code with it, and the two round the last bit differently.  The
Rust kernels take all three from the C library one element at a time.  With
``LIBM = True`` this oracle does the same, which makes the comparison exact
on every machine; with it off, the oracle is the shipped NumPy expression
as written, exact on a machine whose NumPy takes the C library's.
"""
from __future__ import annotations

import math

import numpy as np

#: Take exp, log and pow from the C library element by element.
LIBM = True


def _exp(x):
    x = np.asarray(x, dtype=np.float64)
    if not LIBM:
        return np.exp(x)
    return np.vectorize(math.exp, otypes=[np.float64])(x)


def _log(x):
    x = np.asarray(x, dtype=np.float64)
    if not LIBM:
        return np.log(x)
    return np.vectorize(_c_log, otypes=[np.float64])(x)


def _c_log(value: float) -> float:
    # math.log refuses zero and negatives where C returns -inf and NaN.
    if value > 0.0:
        return math.log(value)
    if value == 0.0:
        return -math.inf
    return math.nan


def _pow(x, y: float):
    x = np.asarray(x, dtype=np.float64)
    if not LIBM:
        return x ** y
    return np.vectorize(lambda v: math.pow(v, y) if v >= 0.0 else math.nan,
                        otypes=[np.float64])(x)


# ------------------------------------------------- render tape (wrfout_export)

def gaussian_to_regular(source_latitude_deg, source_longitude_deg,
                        target_lat, target_lon):
    """``wrfout_export._gaussian_to_regular`` at 7548983."""
    lat = np.asarray(source_latitude_deg, dtype=np.float64)
    lon = np.asarray(source_longitude_deg, dtype=np.float64)
    fy = np.interp(target_lat, lat, np.arange(lat.size, dtype=np.float64))
    y0 = np.minimum(fy.astype(np.int64), lat.size - 2)
    wy = np.clip(fy - y0, 0.0, 1.0)
    dlon = 360.0 / lon.size
    fx = np.mod(target_lon - lon[0], 360.0) / dlon
    x0 = np.mod(fx.astype(np.int64), lon.size)
    wx = fx - np.floor(fx)
    x1 = np.mod(x0 + 1, lon.size)

    def regrid(values: np.ndarray) -> np.ndarray:
        field = np.asarray(values, dtype=np.float64)
        yl = y0[:, None]
        yu = (y0 + 1)[:, None]
        wyc = wy[:, None]
        wxc = wx[None, :]
        return (
            (1.0 - wyc) * (
                (1.0 - wxc) * field[..., yl, x0[None, :]]
                + wxc * field[..., yl, x1[None, :]]
            )
            + wyc * (
                (1.0 - wxc) * field[..., yu, x0[None, :]]
                + wxc * field[..., yu, x1[None, :]]
            )
        )

    return regrid


def tape_column(theta, species, ps, phi_surface, a_half, b_half, *,
                reference_pressure_pa, kappa, gas_constant):
    """The tape's column at 7548983: hybrid pressure, T, Tv, half-level phi."""
    a_half = np.asarray(a_half, dtype=np.float64)
    b_half = np.asarray(b_half, dtype=np.float64)
    nz = a_half.size - 1
    p_half = a_half[:, None, None] + b_half[:, None, None] * ps[None]
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    temperature = theta * _pow(p_full / reference_pressure_pa, kappa)
    virtual = temperature * (
        1.0 + 0.61 * species["qv"]
        - sum(species[name] for name in ("qc", "qr", "qi", "qs", "qg"))
    )
    phi_half = np.empty((nz + 1, *ps.shape), dtype=np.float64)
    phi_half[nz] = phi_surface
    for k in range(nz - 1, -1, -1):
        phi_half[k] = phi_half[k + 1] + gas_constant * virtual[k] * _log(
            p_half[k + 1] / p_half[k]
        )
    return p_half, p_full, temperature, virtual, phi_half


# ------------------------------------------- regional (interpolation, translate)

def periodic_bilinear(source_latitude_deg, source_longitude_deg, values,
                      target_latitude_deg, target_longitude_deg):
    """``regional.interpolation.periodic_bilinear`` at 7548983, arithmetic only
    (its refusals stay in the package)."""
    lon = np.asarray(source_longitude_deg, np.float64)
    return periodic_bilinear_core(
        source_latitude_deg, float(lon[0]), float(np.diff(lon)[0]), lon.size,
        values, target_latitude_deg, target_longitude_deg)


def periodic_bilinear_core(source_latitude_deg, lon0, spacing, nlon, values,
                           target_latitude_deg, target_longitude_deg):
    """The same arithmetic, given the first longitude, the spacing and the
    count it reads out of the longitude row."""
    lat = np.asarray(source_latitude_deg, np.float64)
    field = np.asarray(values)
    target_lat = np.asarray(target_latitude_deg, np.float64)
    target_lon = np.asarray(target_longitude_deg, np.float64)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[..., ::-1, :]
    flat_lat = target_lat.reshape(-1)
    j1 = np.searchsorted(lat, flat_lat, side="right")
    j1 = np.clip(j1, 1, lat.size - 1)
    j0 = j1 - 1
    wy = (flat_lat - lat[j0]) / (lat[j1] - lat[j0])

    x = ((target_lon.reshape(-1) - lon0) % 360.0) / spacing
    i0 = np.floor(x).astype(np.int64) % nlon
    i1 = (i0 + 1) % nlon
    wx = x - np.floor(x)

    f00 = field[..., j0, i0]
    f01 = field[..., j0, i1]
    f10 = field[..., j1, i0]
    f11 = field[..., j1, i1]
    lower = f00 * (1.0 - wx) + f01 * wx
    upper = f10 * (1.0 - wx) + f11 * wx
    result = lower * (1.0 - wy) + upper * wy
    return result.reshape((*field.shape[:-2], *target_lat.shape))


def standard_lapse_theta_below(source_pressure, source_theta, target_pressure, *,
                               reference_pressure_pa, kappa, gas_constant,
                               gravity, lapse_rate):
    """``regional.interpolation.standard_lapse_theta_below`` at 7548983."""
    source_p = np.asarray(source_pressure, np.float64)
    theta = np.asarray(source_theta, np.float64)
    target_p = np.asarray(target_pressure, np.float64)
    bottom_p = source_p[-1][None]
    bottom_theta = theta[-1][None]
    bottom_t = bottom_theta * _pow(bottom_p / reference_pressure_pa, kappa)
    depth_m = (gas_constant * bottom_t / gravity) * _log(target_p / bottom_p)
    continued_t = bottom_t + lapse_rate * depth_m
    return continued_t * _pow(reference_pressure_pa / target_p, kappa)


def log_pressure_interpolate(source_pressure, source_values, target_pressure,
                             bottom_values=None):
    """``regional.interpolation.log_pressure_interpolate`` at 7548983,
    arithmetic only: the per-column np.interp loop."""
    source_p = np.asarray(source_pressure, np.float64)
    source = np.asarray(source_values, np.float64)
    target_p = np.asarray(target_pressure, np.float64)
    flat_sp = _log(source_p.reshape(source_p.shape[0], -1))
    flat_sv = source.reshape(source.shape[0], -1)
    flat_tp = _log(target_p.reshape(target_p.shape[0], -1))
    out = np.empty_like(flat_tp)
    for column in range(flat_sp.shape[1]):
        out[:, column] = np.interp(
            flat_tp[:, column], flat_sp[:, column], flat_sv[:, column],
            left=flat_sv[0, column], right=flat_sv[-1, column],
        )
    result = out.reshape(target_p.shape)
    if bottom_values is not None:
        continued = np.asarray(bottom_values, np.float64)
        result = np.where(target_p > source_p[-1][None], continued, result)
    return result


def hydrostatic_half(virtual_temperature, p_half, terrain_geopotential, *,
                     gas_constant):
    """``regional.translate._hydrostatic_half`` at 7548983."""
    nlev = virtual_temperature.shape[0]
    half = np.empty((nlev + 1, *terrain_geopotential.shape), np.float64)
    half[-1] = terrain_geopotential
    for k in range(nlev - 1, -1, -1):
        half[k] = half[k + 1] + gas_constant * virtual_temperature[k] * _log(
            p_half[k + 1] / p_half[k]
        )
    full = np.empty_like(virtual_temperature)
    for k in range(nlev):
        p_full = np.sqrt(p_half[k] * p_half[k + 1])
        full[k] = half[k + 1] + gas_constant * virtual_temperature[k] * _log(
            p_half[k + 1] / p_full
        )
    return half, full


# --------------------------------------- the seam's own API, backed by the oracle

class OracleSeam:
    """Stand-ins with `arwen_global.render_kernels`' signatures, computing
    what the 7548983 NumPy computed, so a whole export or translation can
    be run once through the library and once through the oracle and the
    two outputs compared bit for bit."""

    class GaussianRegridder:
        def __init__(self, source_latitude_deg, source_longitude_deg,
                     target_latitude_deg, target_longitude_deg):
            self._regrid = gaussian_to_regular(
                source_latitude_deg, source_longitude_deg,
                np.asarray(target_latitude_deg, np.float64),
                np.asarray(target_longitude_deg, np.float64))

        def __call__(self, values, post="none"):
            out = self._regrid(values)
            if post == "exp":
                return _exp(out)
            if post == "floor_zero":
                return np.clip(out, 0.0, None)
            return out

    @staticmethod
    def library():
        return None

    @staticmethod
    def periodic_bilinear(source_latitude_deg, longitude0_deg, spacing_deg, nlon,
                          values, target_latitude_deg, target_longitude_deg):
        return periodic_bilinear_core(
            source_latitude_deg, float(longitude0_deg), float(spacing_deg),
            int(nlon), values, target_latitude_deg, target_longitude_deg)

    @staticmethod
    def log_pressure_interpolate(source_pressure, source_values, target_pressure,
                                 bottom_values=None):
        return log_pressure_interpolate(source_pressure, source_values,
                                        target_pressure, bottom_values)

    @staticmethod
    def standard_lapse_theta_below(source_pressure, source_theta, target_pressure,
                                   **kwargs):
        return standard_lapse_theta_below(source_pressure, source_theta,
                                          target_pressure, **kwargs)

    @staticmethod
    def hybrid_pressure(a_half, b_half, surface_pressure):
        a = np.asarray(a_half, np.float64)
        b = np.asarray(b_half, np.float64)
        ps = np.asarray(surface_pressure, np.float64)
        shape = (slice(None),) + (None,) * ps.ndim
        p_half = a[shape] + b[shape] * ps[None]
        return p_half, np.sqrt(p_half[:-1] * p_half[1:])

    @staticmethod
    def virtual_temperature(theta, species, p_full, *, reference_pressure_pa, kappa):
        temperature = theta * _pow(np.asarray(p_full) / reference_pressure_pa, kappa)
        condensate = sum(species[name] for name in ("qc", "qr", "qi", "qs", "qg"))
        return temperature, temperature * (1.0 + 0.61 * species["qv"] - condensate)

    @staticmethod
    def hydrostatic(virtual_temperature_k, p_half, surface_geopotential, *,
                    gas_constant, full_levels=False):
        half, full = hydrostatic_half(
            np.asarray(virtual_temperature_k, np.float64),
            np.asarray(p_half, np.float64),
            np.asarray(surface_geopotential, np.float64),
            gas_constant=gas_constant)
        return (half, full) if full_levels else half
