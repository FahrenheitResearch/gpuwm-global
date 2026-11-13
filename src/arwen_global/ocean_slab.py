"""The slab (mixed-layer) ocean: one temperature per ocean column.

Design: docs/ARWEN_GLOBAL_OCEAN.md.  This module is arithmetic only: every
function takes planes and returns planes on the array module it is handed
(numpy on the host, cupy on the card), float64 throughout, and reads no
run state.  The coupling (arwen_global.ocean_coupler) decides which
columns, which planes and when.

Per ocean column, over one coupling interval dt (one model step), with T
the mixed-layer temperature and h(t) its depth:

    rho_w c_w h dT/dt = F_open + Q + R

* F_open, :func:`open_water_flux`: the net heat the atmosphere put into the
  water this interval (absorbed shortwave, absorbed longwave less emission,
  less sensible and latent heat), positive into the ocean;
* Q, :func:`qflux`: the heat-flux correction.  From the climatology and a
  calibration file of the model's own prescribed-SST run,

      Q = rho_w c_w h dC_s/dt - rho_i L_f h_ref dC_i/dt - F_cal(t)

  so that a column whose fluxes equal the calibration run's follows the
  climatology exactly (ocean_qflux.py makes and reads F_cal);
* R, :func:`restoring_flux`: rho_w c_w h (T* - T) / tau_r, the optional
  weak restoring toward the target SST T* (zero when tau_r is None).

The step is explicit (:func:`slab_step`); a column cooled below the
freezing point of sea water is held at it and the deficit is handed to the
sea-ice step (arwen_global.sea_ice_thermo), which turns it into ice.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from .constants import LATENT_HEAT_VAPORIZATION
from .surface_series import SEA_WATER_FREEZING_K

#: Density and specific heat of sea water; their product is the volumetric
#: heat capacity of the mixed layer (4.09e6 J m-3 K-1).
RHO_SEAWATER_KG_M3 = 1026.0
CP_SEAWATER_J_KG_K = 3990.0
RHO_CP_SEAWATER_J_M3_K = RHO_SEAWATER_KG_M3 * CP_SEAWATER_J_KG_K
#: Latent heat of vaporization the open-water latent flux is booked with
#: (the package constant the native suite's lake step books ``qfx`` with).
LATENT_HEAT_VAPORIZATION_J_KG = LATENT_HEAT_VAPORIZATION
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
#: The defaults of the [ocean] slab keys (boundary_options.py parses them).
DEFAULT_DEPTH = "woa18"
DEFAULT_DEPTH_BOUNDS_M = (10.0, 200.0)
DEFAULT_RELAXATION_DAYS = 365.0
DEFAULT_TARGET = "climatology"
SLAB_TARGETS = ("climatology", "anomaly")
#: The freezing point the mixed layer is held at (271.35 K).
FREEZING_K = SEA_WATER_FREEZING_K
SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class SlabParameters:
    """The resolved slab keys of one run.

    ``depth`` is ``"woa18"`` (the climatology's monthly mixed-layer depth,
    interpolated like the SST and clamped to ``depth_bounds_m``) or a fixed
    depth in metres.  ``relaxation_days`` None is no restoring.
    """

    depth: str | float = DEFAULT_DEPTH
    depth_bounds_m: tuple = DEFAULT_DEPTH_BOUNDS_M
    relaxation_days: float | None = DEFAULT_RELAXATION_DAYS
    target: str = DEFAULT_TARGET

    def __post_init__(self) -> None:
        depth = self.depth
        if isinstance(depth, str):
            if depth != "woa18":
                raise ValueError(f"slab depth {depth!r} is neither \"woa18\" nor metres")
        else:
            if isinstance(depth, bool) or not math.isfinite(float(depth)) or float(depth) <= 0.0:
                raise ValueError(f"a fixed slab depth is a positive number of metres, "
                                 f"not {depth!r}")
            object.__setattr__(self, "depth", float(depth))
        bounds = tuple(float(b) for b in self.depth_bounds_m)
        if len(bounds) != 2 or not all(math.isfinite(b) for b in bounds) \
                or not 0.0 < bounds[0] <= bounds[1]:
            raise ValueError(f"slab depth bounds are [lower, upper] metres with "
                             f"0 < lower <= upper, not {list(self.depth_bounds_m)}")
        object.__setattr__(self, "depth_bounds_m", bounds)
        days = self.relaxation_days
        if days is not None:
            if isinstance(days, bool) or not math.isfinite(float(days)) or float(days) <= 0.0:
                raise ValueError(f"slab relaxation is a positive number of days or None, "
                                 f"not {days!r}")
            object.__setattr__(self, "relaxation_days", float(days))
        if self.target not in SLAB_TARGETS:
            raise ValueError(f"slab target {self.target!r} is not one of "
                             + ", ".join(SLAB_TARGETS))


def month_bracket(when) -> tuple[int, int, datetime, datetime]:
    """``(i0, i1, t0, t1)``: the month planes (0-11) whose mid-month instants
    ``t0 <= when < t1`` bracket ``when``, by the rule
    :func:`arwen_global.surface_climatology.month_weights` interpolates
    with (the weight of ``i1`` is ``(when - t0) / (t1 - t0)``)."""
    from .surface_climatology import _utc, mid_month

    when = _utc(when)
    year, month = when.year, when.month
    if when >= mid_month(year, month):
        a = (year, month)
        b = (year + 1, 1) if month == 12 else (year, month + 1)
    else:
        a = (year - 1, 12) if month == 1 else (year, month - 1)
        b = (year, month)
    return a[1] - 1, b[1] - 1, mid_month(*a), mid_month(*b)


def _f64(value, xp):
    return xp.asarray(value, dtype=xp.float64)


def mixed_layer_depth(climatology, when, params: SlabParameters, ocean, xp):
    """h(t) in metres on the (nlat, nlon) grid, float64: the fixed depth, or
    the climatology's ``mixed_layer_depth_m`` at ``when`` (the SST's month
    weights, arwen_global.surface_climatology.month_weights) clamped to
    ``params.depth_bounds_m``; a gap in the climatology on an ``ocean``
    column takes the lower bound, and a column off ``ocean`` reads 0.

    The two month planes are read as float64 and weighted in float64 (the
    file stores float32)."""
    from .surface_climatology import month_weights

    ocean = xp.asarray(ocean, dtype=bool)
    if params.depth != "woa18":
        return xp.where(ocean, float(params.depth), 0.0).astype(xp.float64)
    planes = getattr(climatology, "mixed_layer_depth_m", None)
    if planes is None:
        raise ValueError(
            f"slab_depth = \"woa18\" reads the climatology's mixed_layer_depth_m plane and "
            f"{getattr(climatology, 'path', 'this climatology')} carries none; build it "
            "with `gpuwm-global climatology build --with-mld`")
    i0, i1, w = month_weights(when)
    h = _f64(planes[i0], xp) * (1.0 - w) + _f64(planes[i1], xp) * w
    lower, upper = params.depth_bounds_m
    h = xp.where(xp.isfinite(h), xp.clip(h, lower, upper), lower)
    return xp.where(ocean, h, 0.0)


def climatology_tendency(climatology, when, xp):
    """``(dC_s/dt, dC_i/dt)`` at ``when``, K/s and 1/s, float64 planes: the
    slopes of the piecewise-linear climatology between the two mid-month
    instants that bracket ``when`` (zero off the climatology's ocean).

    The slope is constant between two mid-month instants and steps at each,
    so a step that ends on a mid-month instant carries exactly the change of
    the climatology over it; a model step of 300 s divides the 12 h
    resolution of the mid-month instants (each is at 00 or 12 UTC)."""
    i0, i1, t0, t1 = month_bracket(when)
    span = (t1 - t0).total_seconds()
    out = []
    for name in ("sst_k", "sea_ice_fraction"):
        planes = getattr(climatology, name)
        rate = (_f64(planes[i1], xp) - _f64(planes[i0], xp)) / span
        out.append(xp.where(xp.isfinite(rate), rate, 0.0))
    return out[0], out[1]


def open_water_flux(*, swdown, albedo, glw, emissivity, skin_k, hfx, qfx, xp):
    """Net heat into open water, W/m2, positive down:

        swdown (1 - albedo) + emissivity (glw - sigma skin^4) - hfx - L_v qfx

    from the planes the surface layer and the radiation held for the
    interval (``qfx`` in kg m-2 s-1).  Each input may be a plane or a
    number (the leads' 0.08 and 0.98); the arithmetic is float64."""
    skin = _f64(skin_k, xp)
    emissivity = _f64(emissivity, xp)
    shortwave = _f64(swdown, xp) * (1.0 - _f64(albedo, xp))
    longwave = emissivity * (_f64(glw, xp) - STEFAN_BOLTZMANN_W_M2_K4 * skin ** 4)
    return (shortwave + longwave - _f64(hfx, xp)
            - LATENT_HEAT_VAPORIZATION_J_KG * _f64(qfx, xp))


def qflux(*, depth_m, sst_rate_k_s, ice_fraction_rate_s, ice_reference_thickness_m,
          calibration_flux_w_m2, xp):
    """Q in W/m2, float64:

        rho_w c_w depth dC_s/dt - rho_i L_f h_ref dC_i/dt - F_cal

    ``ice_reference_thickness_m`` is the plane of h_ref
    (arwen_global.sea_ice_thermo.ice_reference_thickness), and
    ``calibration_flux_w_m2`` is F_cal at the same instant
    (arwen_global.ocean_qflux.QFluxFile.at)."""
    from .sea_ice_thermo import ICE_FUSION_J_M3

    return (RHO_CP_SEAWATER_J_M3_K * _f64(depth_m, xp) * _f64(sst_rate_k_s, xp)
            - ICE_FUSION_J_M3 * _f64(ice_reference_thickness_m, xp)
            * _f64(ice_fraction_rate_s, xp)
            - _f64(calibration_flux_w_m2, xp))


def restoring_flux(*, temperature_k, target_k, depth_m, relaxation_days, xp):
    """R = rho_w c_w depth (target - T) / (relaxation_days * 86400), W/m2;
    zeros when ``relaxation_days`` is None.  A column of zero depth (off
    the ocean) reads 0."""
    depth = _f64(depth_m, xp)
    if relaxation_days is None:
        return xp.zeros_like(depth)
    tau = float(relaxation_days) * SECONDS_PER_DAY
    gap = _f64(target_k, xp) - _f64(temperature_k, xp)
    # Off the ocean the target is the climatology's NaN and the depth 0.
    return xp.where(depth > 0.0, RHO_CP_SEAWATER_J_M3_K * depth * gap / tau, 0.0)


def slab_step(*, temperature_k, depth_m, net_flux_w_m2, dt_s, xp):
    """One explicit step of the mixed layer.

    Returns ``(new_temperature_k, freeze_deficit_j_m2)``: T + dt F /
    (rho_w c_w depth), held at or above :data:`FREEZING_K`, and the energy
    the hold withheld, rho_w c_w depth (T_f - T_unheld) where the step went
    below the freezing point and 0 elsewhere (never negative).

    A column of zero depth (off the ocean) is returned as it came, with no
    deficit, so the zeros the state holds there stay zeros."""
    temperature = _f64(temperature_k, xp)
    depth = _f64(depth_m, xp)
    on = depth > 0.0
    capacity = RHO_CP_SEAWATER_J_M3_K * xp.where(on, depth, 1.0)
    unheld = temperature + float(dt_s) * _f64(net_flux_w_m2, xp) / capacity
    below = on & (unheld < FREEZING_K)
    new = xp.where(on, xp.where(below, FREEZING_K, unheld), temperature)
    deficit = xp.where(below, capacity * (FREEZING_K - unheld), 0.0)
    return new, deficit


def initial_temperature(cold_skin_k, ocean, ice_fraction, xp):
    """T at a cold start: the cold-start skin on ``ocean`` columns, held at
    or above :data:`FREEZING_K`; :data:`FREEZING_K` where ``ice_fraction`` is
    above zero; 0 off ``ocean``."""
    skin = _f64(cold_skin_k, xp)
    ocean = xp.asarray(ocean, dtype=bool)
    iced = _f64(ice_fraction, xp) > 0.0
    held = xp.where(skin >= FREEZING_K, skin, FREEZING_K)
    return xp.where(ocean, xp.where(iced, FREEZING_K, held), 0.0)


__all__ = [
    "CP_SEAWATER_J_KG_K", "DEFAULT_DEPTH", "DEFAULT_DEPTH_BOUNDS_M",
    "DEFAULT_RELAXATION_DAYS", "DEFAULT_TARGET", "FREEZING_K",
    "LATENT_HEAT_VAPORIZATION_J_KG", "RHO_CP_SEAWATER_J_M3_K", "RHO_SEAWATER_KG_M3",
    "SECONDS_PER_DAY", "SLAB_TARGETS", "STEFAN_BOLTZMANN_W_M2_K4", "SlabParameters",
    "climatology_tendency", "initial_temperature", "mixed_layer_depth", "month_bracket",
    "open_water_flux", "qflux", "restoring_flux", "slab_step",
]
