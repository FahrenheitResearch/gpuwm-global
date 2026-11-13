"""Radiation between its full calls: the shortwave follows the sun.

RRTMGP fires once per ``radiation_interval_s`` bucket (3000 s, ten 300 s
steps, in the shipped semi-Lagrangian configs) and its planes are held
until the next bucket (``native_runtime._radiation_step``).  The call's
sun is the interval's midpoint sun (``core/rrtmgp.py``, WRF's
``xtime + radt * 0.5``), so a held plane is wrong by up to half an
interval of sun motion: a column the call saw at sunrise keeps its
midpoint shortwave for the whole interval and a column past sunset keeps
heating.

``radiation_between`` (a :class:`~arwen_global.physics.native_options.
NativePhysicsOptions` field) chooses what happens between calls:

``"hold"`` (the default)
    The call's planes are applied unchanged until the next call, as every
    run before the option did.  No array is added, no arithmetic changes,
    and the option is left out of the adapter identity, so every config
    hash and checkpoint keeps its bytes.

``"surface"``
    WRF's ``swint_opt = 1`` on the surface downward shortwave, as the
    regional engine runs it (``gpuwm/core/swint.py`` there, after WRF
    v4.6.1 ``module_radiation_driver.F`` ``update_swinterp_parameters`` /
    ``interp_sw_radiation``): each full call fits the column's surface
    downward shortwave as ``SWDOWN = Gx * coszen**gg``, the exponent from
    the log ratio between this call and the previous one, clamped to
    [-0.5, 2.5], a clamped or zero exponent falling back to the ratio
    ``coszen_now / coszen_ref``; every physics call then evaluates the fit
    at the current sun, a night column (either cosine at or under 1e-4)
    zero, and the absorbed part is ``GSW = SWDOWN * (1 - albedo)`` on the
    step's live albedo.  Noah, the sea-ice and lake surfaces read the
    updated planes.  The global solver hands back no direct-beam plane, so
    only WRF's global-flux half of the fit is carried (``SWDDIR`` and its
    fit have no reader here).  The heating rates stay the call's.

``"column"``
    ``"surface"``, plus the shortwave heating rate and the two top-of-
    atmosphere shortwave planes scaled by ``coszen_now / coszen_ref`` (the
    ratio of the incident top-of-atmosphere flux, exact for ``swdnt`` and
    the first-order update for the absorbed and reflected parts), zero
    where either cosine is at or under 1e-4.  A column dark at the call
    stays dark until the next call (its reference is zero), as under WRF's
    rule.

Clouds do not move between calls in either mode.  No update of the cloud
part is cheap here: the shortwave sees the cloud through McICA samples
seeded from each column's lowest pressures (``core/rrtmgp.py``,
``docs/CARRIED-PHYSICS-DIVERGENCE.md`` row ``RRTMGP-16``), so a partly
cloudy column's flux is not a smooth function of its cloud fields that a
per-step delta could follow, and a full solve on a subset of columns is a
different build.

The current sun is evaluated at the physics call's model time ``time_s``,
as WRF evaluates it at the step's ``xtime``.  Under the default Strang
split a step's second physics half and the next step's first half both
carry the time between them and together integrate the step around it,
so the instant is the centre of what the pair applies; under
``physics_split = "merged"`` (one call after the dynamics, stamped at the
step's end) it is half a step late.  The statements are the WRF
``radconst`` / ``calc_coszen`` ones the radiation call uses (``rrtmgp.RRTMGPRadiation._cosine_zenith``) on float32 latitude and
longitude.  Every reference the update reads is a persistent array of the
physics namespace, so a checkpoint carries it and a restart in the middle
of an interval continues bit for bit.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

F = np.float32

#: The option values, default first.
MODES = ("hold", "surface", "column")
DEFAULT_MODE = MODES[0]

#: coszen_min in both WRF routines (module_radiation_driver.F).
COSZEN_MIN = F(1.0e-4)

#: WRF's exponent clamp (update_swinterp_parameters).
EXPONENT_MIN = F(-0.5)
EXPONENT_MAX = F(2.5)

#: Persistent arrays the update adds to the physics namespace, by mode.
#: The four WRF fit planes (``swdown_ref``, ``coszen_ref``, ``Gx``,
#: ``gg``) under ``"surface"``; the call's own shortwave heating and top
#: planes as references under ``"column"`` as well.
SURFACE_ARRAYS = (
    "radbetween_swdown_ref", "radbetween_coszen_ref",
    "radbetween_gx", "radbetween_gg",
)
COLUMN_SURFACE_ARRAYS = ("radbetween_swdnt_ref", "radbetween_swupt_ref")
COLUMN_VOLUME_ARRAYS = ("radbetween_rthratensw_ref",)


def validate_mode(mode) -> str:
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(
            f"radiation_between must be one of {list(MODES)}, got {mode!r}")
    return mode


def surface_arrays(mode: str) -> tuple[str, ...]:
    """The (ny, nx) persistent arrays ``mode`` adds (none under hold)."""
    if mode == "surface":
        return SURFACE_ARRAYS
    if mode == "column":
        return SURFACE_ARRAYS + COLUMN_SURFACE_ARRAYS
    return ()


def volume_arrays(mode: str) -> tuple[str, ...]:
    """The (nz, ny, nx) persistent arrays ``mode`` adds."""
    return COLUMN_VOLUME_ARRAYS if mode == "column" else ()


# ---------------------------------------------------------------------------
# The sun
# ---------------------------------------------------------------------------

def cosine_zenith(xp, latitude_deg, longitude_deg, valid_time):
    """WRF ``radconst`` / ``calc_coszen`` at ``valid_time``, float32.

    The statements of ``rrtmgp.RRTMGPRadiation._cosine_zenith`` with no
    hour offset (the radiation call adds the half interval; here the
    caller passes the instant it wants), on float32 latitude and
    longitude as the driver holds them, through ``xp``.
    """
    lat = xp.asarray(latitude_deg, dtype=F)
    lon = xp.asarray(longitude_deg, dtype=F)
    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0
            + valid_time.microsecond / 3.6e9)
    julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
    degrad = np.pi / 180.0
    dpd = 360.0 / 365.0
    if julian >= 80.0:
        solar_longitude = dpd * (julian - 80.0)
    else:
        solar_longitude = dpd * (julian + 285.0)
    declination = np.arcsin(
        np.sin(23.5 * degrad) * np.sin(solar_longitude * degrad))
    da = 2.0 * np.pi * (julian - 1.0) / 365.0
    equation = 229.18 * (
        0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
        - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))
    solar_minutes = 60.0 * hour + equation + 4.0 * lon
    hour_angle = xp.deg2rad(solar_minutes / 4.0 - 180.0)
    latitude = xp.deg2rad(lat)
    mu = (xp.sin(latitude) * F(np.sin(declination))
          + xp.cos(latitude) * F(np.cos(declination))
          * xp.cos(hour_angle))
    return xp.clip(mu, F(-1.0), F(1.0)).astype(F, copy=False)


# ---------------------------------------------------------------------------
# WRF's fit and evaluation, whole planes through xp
# ---------------------------------------------------------------------------

def _exponent(xp, flux, flux_0, coszen, coszen_0):
    """One WRF fit exponent, elementwise (update_swinterp_parameters)."""
    one = F(1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_c0 = xp.where(coszen_0 != F(0.0), coszen_0, one)
        ratio = (coszen / safe_c0).astype(F, copy=False)
        log_flux = xp.log(
            xp.maximum(one, flux) / xp.maximum(one, flux_0)).astype(F, copy=False)
        below = xp.log(xp.minimum(one - F(1.0e-4), ratio))
        above = xp.log(xp.maximum(one + F(1.0e-4), ratio))
        b = xp.where(ratio < one, log_flux / below,
                     xp.where(ratio > one, log_flux / above, F(0.0)))
    return xp.clip(b.astype(F, copy=False), EXPONENT_MIN, EXPONENT_MAX)


def fit(xp, *, coszen, coszen_now, swdown, swdown_ref, coszen_ref, gx, gg):
    """WRF ``update_swinterp_parameters``, the global-flux half, in place.

    ``coszen`` is the call's own (interval-midpoint) cosine and ``swdown``
    the call's surface downward shortwave; ``coszen_now`` the sun at the
    instant the call is applied.  ``swdown_ref``, ``coszen_ref``, ``gx``
    and ``gg`` are the persistent planes and are updated in place: the
    fit against the previous call's reference (a first guess at
    ``coszen_now`` on a column whose ``Gx`` is not yet positive), then
    this call stored as the reference.
    """
    lit = (coszen > COSZEN_MIN) & (coszen_now > COSZEN_MIN)
    first = gx <= F(0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_cz = xp.where(coszen != F(0.0), coszen, F(1.0))
        guess = ((coszen_now / safe_cz).astype(F, copy=False) * swdown).astype(F, copy=False)
        flux_0 = xp.where(first, guess, swdown_ref)
        coszen_0 = xp.where(first, coszen_now, coszen_ref)
        g = _exponent(xp, swdown, flux_0, coszen, coszen_0)
        new_gx = (swdown / xp.power(xp.where(lit, coszen, F(1.0)), g)).astype(F, copy=False)
    gg[...] = xp.where(lit, g, F(0.0))
    gx[...] = xp.where(lit, new_gx, F(0.0))
    coszen_ref[...] = coszen
    swdown_ref[...] = swdown


def ratio_rule(xp, exponent):
    """WRF v4.6.1's test for the ratio fallback: a clamped exponent or an
    exponent of exactly zero (WRF cc02f989f)."""
    return ((exponent == EXPONENT_MIN) | (exponent == EXPONENT_MAX)
            | (exponent == F(0.0)))


def lit_ratio(xp, coszen_now, coszen_ref):
    """``coszen_now / coszen_ref`` where both are above 1e-4, else 0."""
    lit = (coszen_ref > COSZEN_MIN) & (coszen_now > COSZEN_MIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = xp.where(lit, coszen_ref, F(1.0))
        return xp.where(lit, (coszen_now / safe).astype(F, copy=False), F(0.0))


def evaluate(xp, *, coszen_now, coszen_ref, swdown_ref, gx, gg):
    """WRF ``interp_sw_radiation``, global flux: SWDOWN at ``coszen_now``."""
    ratio = lit_ratio(xp, coszen_now, coszen_ref)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        base = xp.where(ratio > F(0.0), coszen_now, F(1.0))
        powered = (gx * xp.power(base, gg)).astype(F, copy=False)
    by_ratio = (ratio * swdown_ref).astype(F, copy=False)
    value = xp.where(ratio_rule(xp, gg), by_ratio, powered)
    return xp.where(ratio > F(0.0), value, F(0.0)).astype(F, copy=False)


# ---------------------------------------------------------------------------
# The runtime's entry point
# ---------------------------------------------------------------------------

def apply(batch, persistent, *, mode: str, start_time, due: bool) -> None:
    """Rewrite the held shortwave planes at the current sun, in place.

    Called by ``native_runtime._radiation_step`` on every physics call
    under a mode other than ``"hold"``, after a due call has written its
    planes and before the heating is applied and the radiation integrals
    advance, so every reader of the planes and the integrals see the
    updated values.
    """
    if mode == "hold":
        return
    xp = batch.xp
    arrays = persistent.arrays
    valid_time = start_time + timedelta(seconds=float(batch.time_s))
    coszen_now = cosine_zenith(
        xp, batch.arrays["latitude_deg"], batch.arrays["longitude_deg"],
        valid_time)
    swdown_ref = arrays["radbetween_swdown_ref"]
    coszen_ref = arrays["radbetween_coszen_ref"]
    gx = arrays["radbetween_gx"]
    gg = arrays["radbetween_gg"]
    if due:
        fit(xp, coszen=arrays["coszen"], coszen_now=coszen_now,
            swdown=arrays["swdown"], swdown_ref=swdown_ref,
            coszen_ref=coszen_ref, gx=gx, gg=gg)
        if mode == "column":
            arrays["radbetween_rthratensw_ref"][...] = arrays["rad_rthratensw"]
            arrays["radbetween_swdnt_ref"][...] = arrays["swdnt"]
            arrays["radbetween_swupt_ref"][...] = arrays["swupt"]
    swdown = evaluate(xp, coszen_now=coszen_now, coszen_ref=coszen_ref,
                      swdown_ref=swdown_ref, gx=gx, gg=gg)
    albedo = xp.asarray(batch.surface.albedo, dtype=F)
    arrays["swdown"][...] = swdown
    arrays["gsw"][...] = (swdown * (F(1.0) - albedo)).astype(F, copy=False)
    if mode == "column":
        ratio = lit_ratio(xp, coszen_now, coszen_ref)
        arrays["rad_rthratensw"][...] = (
            arrays["radbetween_rthratensw_ref"] * ratio[None, :, :])
        arrays["swdnt"][...] = arrays["radbetween_swdnt_ref"] * ratio
        arrays["swupt"][...] = arrays["radbetween_swupt_ref"] * ratio


__all__ = [
    "COSZEN_MIN", "DEFAULT_MODE", "MODES", "apply", "cosine_zenith",
    "evaluate", "fit", "lit_ratio", "ratio_rule", "surface_arrays",
    "validate_mode", "volume_arrays",
]
