"""The re-balanced rotation start (plan section 4.7, package 3.2).

Every wind in an Earth analysis balances Earth's rotation.  Start a world
that turns twice as fast from those winds and the Coriolis force doubles
under them: a 30 m/s jet is left with an unbalanced acceleration of about
f u, the size of the whole pressure-gradient force it was holding, and the
first hours of the run are a geostrophic-adjustment shock.

The rule here keeps the analysis's own mass field (temperature, humidity,
surface pressure and terrain are not touched) and gives it the winds that
hold the SAME force against the world's rotation:

*   The zonal mean of the eastward wind is put back into gradient-wind
    balance at the new rate.  Earth's balance on each latitude row and
    level is ``(f + u tan(lat) / a) u = G``, where ``G`` is the
    pressure-gradient force the analysis's own zonal-mean wind holds; the
    world's zonal-mean wind is the root of
    ``(r f + u' tan(lat) / a) u' = G`` that is continuous with Earth's
    (``r`` the world's rotation over Earth's).  Written in the reduced
    variable ``x = u / (2 Omega a cos(lat))`` the root is finite on every
    row, the equator included, where both terms vanish and the root is
    the limit of its neighbours' (within ``x`` of ``u / r``).
*   Everything else, the eddies of the eastward wind and the whole
    northward wind, is divided by ``r``, which keeps the Coriolis force on
    it, ``f v``, exactly what it was on Earth.

In geostrophic balance this is the wind thermal-wind balance gives at the
new rate with the analysis's own temperature (``r f du'/dln p = f du/dln
p``), so the start's thermal-wind residual at the world's rotation is the
analysis's own at Earth's.  What is not kept is the advective part of the
balance: the wind's own advection scales as ``1 / r**2``, so the start's
imbalance at the new rate is the analysis's Rossby-number part, not its
whole Coriolis force.  The plan's first wording (section 4.7) started from
the zonal-mean temperature; that start is the analysis's zonal mean, which
package 2.2 refuses over mountains, and a day-length world keeps Earth's
mountains.  On a world that already starts from the zonal mean (a
waterworld) this rule acts on that zonal-mean frame and is exactly the
plan's: the zonal wind of a zonally uniform temperature, re-derived at the
new rate.

The rule is defined for a world that turns at least half as fast as Earth
(a day of 48 hours or less).  A slower world would need winds more than
twice Earth's to hold Earth's pressure gradients, which is not a start but
another atmosphere, and its zonal mean leaves the regular gradient-wind
branch first; it is refused by name until a rule for it is measured.

:func:`rebalanced_winds` is the rule on arrays, :func:`rotation_start_replacements`
the edit of an analysis frame (the cold start's frame hook calls it
through ``world/start.py``), and :func:`rebalanced_state` the same rule on
a model state, which the CPU gate of this package integrates.
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import EARTH_RADIUS_M, EARTH_ROTATION_RATE_S

ROTATION_START_SCHEMA = "arwen-global.world-rotation-start/v1"

#: The slowest rotation the rule is defined for, over Earth's: a 48-hour
#: day.  Below it the winds that hold Earth's pressure gradients are more
#: than twice Earth's.
SLOWEST_REBALANCED_ROTATION = 0.5

#: Rows whose cosine of latitude is below this are polar rows, where a
#: zonal mean of the eastward wind on a latitude-longitude grid is not a
#: wind; there the zonal mean is divided by ``r`` like the eddies.
POLAR_COSINE = 1.0e-6

RULE_TEXT = (
    "the analysis's temperature, humidity, surface pressure and terrain "
    "are kept; the zonal mean of the eastward wind on every level and "
    "latitude row is put back into gradient-wind balance at the world's "
    "rotation with the pressure-gradient force the analysis's own wind "
    "holds; the eddies of the eastward wind and the whole northward wind "
    "are divided by the world's rotation over Earth's, which keeps the "
    "Coriolis force on them exactly Earth's"
)


def _check_factor(rotation_factor: float) -> float:
    factor = float(rotation_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            f"the world's rotation over Earth's must be a positive number, "
            f"got {rotation_factor!r}")
    if factor < SLOWEST_REBALANCED_ROTATION:
        raise ValueError(
            "the re-balanced rotation start is defined for a world that "
            f"turns at least {SLOWEST_REBALANCED_ROTATION:g} times as fast "
            f"as Earth (a day of {24.0 / SLOWEST_REBALANCED_ROTATION:g} "
            f"hours or less); this one turns {factor:g} times as fast "
            f"(a {24.0 / factor:g}-hour day), where the winds that hold "
            "Earth's pressure gradients would be "
            f"{1.0 / factor:.1f} times Earth's: that is another atmosphere, "
            "not a start, and no rule for it has been measured")
    return factor


def rebalanced_zonal_mean(zonal_mean_u, latitude_deg, rotation_factor, *,
                          radius_m: float = EARTH_RADIUS_M,
                          rotation_rate_s: float = EARTH_ROTATION_RATE_S):
    """The gradient-wind zonal mean at ``rotation_factor`` times Earth's
    rotation that holds the force Earth's zonal mean ``zonal_mean_u``
    holds at Earth's.  ``latitude_deg`` broadcasts against
    ``zonal_mean_u`` (latitude on axis -2 when the mean keeps its
    longitude axis).  Returns ``(u, irregular)``: ``irregular`` is the
    boolean mask of points whose Earth wind is on the anomalous branch or
    whose new discriminant is negative, where the result is ``u / r``.
    """
    r = _check_factor(rotation_factor)
    u = np.asarray(zonal_mean_u, dtype=np.float64)
    cos_lat = np.cos(np.deg2rad(np.asarray(latitude_deg, dtype=np.float64)))
    polar = np.abs(cos_lat) < POLAR_COSINE
    safe_cos = np.where(polar, 1.0, cos_lat)
    # x = u / (2 Omega a cos(lat)): Earth's balance is
    # 2 Omega sin (1 + x) u = G and the world's
    # 2 Omega sin (r + x') u' = G, so (r + x') x' = (1 + x) x.
    x = u / (2.0 * float(rotation_rate_s) * float(radius_m) * safe_cos)
    regular = 1.0 + 2.0 * x
    discriminant = regular * regular + r * r - 1.0
    irregular = polar | (regular <= 0.0) | (discriminant < 0.0)
    root = np.sqrt(np.where(irregular, 1.0, discriminant))
    denominator = r + np.where(irregular, 1.0, root)
    balanced = 2.0 * u * (1.0 + x) / denominator
    result = u.copy() if r == 1.0 else np.where(irregular, u / r, balanced)
    return result, np.broadcast_to(irregular, result.shape)


def rebalanced_winds(u, v, latitude_deg, rotation_factor, *,
                     radius_m: float = EARTH_RADIUS_M,
                     rotation_rate_s: float = EARTH_ROTATION_RATE_S):
    """The rule on arrays: ``(u, v, record)`` with the world's winds.

    ``u`` and ``v`` are ``(..., latitude, longitude)`` on a grid whose
    longitudes are evenly spaced (so the plain row mean is the zonal
    mean); ``latitude_deg`` is the 1-D latitude of axis -2.  At a factor
    of exactly 1 the inputs come back as the same values.
    """
    r = _check_factor(rotation_factor)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim < 2:
        raise ValueError("u and v must share one (..., latitude, longitude) shape")
    lat = np.asarray(latitude_deg, dtype=np.float64)
    if lat.ndim != 1 or lat.size != u.shape[-2]:
        raise ValueError("latitude_deg must be the 1-D latitude of the winds' axis -2")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("the winds carry missing values; the rule needs every point")
    if r == 1.0:
        return u.copy(), v.copy(), _record(r, u, v, u, v, np.zeros(u.shape[:-1], bool))
    mean_u = u.mean(axis=-1, keepdims=True)
    new_mean, irregular = rebalanced_zonal_mean(
        mean_u, lat[:, None], r, radius_m=radius_m, rotation_rate_s=rotation_rate_s)
    new_u = new_mean + (u - mean_u) / r
    new_v = v / r
    return new_u, new_v, _record(r, u, v, new_u, new_v, irregular[..., 0])


def _record(r, u, v, new_u, new_v, irregular) -> dict:
    speed = np.hypot(u, v)
    new_speed = np.hypot(new_u, new_v)
    return {
        "schema": ROTATION_START_SCHEMA,
        "rotation_factor": float(r),
        "rule": RULE_TEXT,
        "largest_speed_m_s": {"analysis": float(np.max(speed)),
                              "start": float(np.max(new_speed))},
        "largest_zonal_mean_u_m_s": {
            "analysis": float(np.max(np.abs(u.mean(axis=-1)))),
            "start": float(np.max(np.abs(new_u.mean(axis=-1)))),
        },
        "rows_off_the_regular_branch": int(np.count_nonzero(irregular)),
    }


def rotation_start_replacements(frame, rotation_factor, *,
                                radius_m: float = EARTH_RADIUS_M,
                                rotation_rate_s: float = EARTH_ROTATION_RATE_S):
    """The rule on an analysis frame (source grid, pressure levels), for
    the cold start's frame hook.  Returns ``(replacements, record)``:
    ``replacements`` maps ``eastward_wind`` and ``northward_wind`` to the
    world's winds; every other plane is the frame's own.  The input frame
    is not modified.  The frame's longitudes must be evenly spaced around
    the whole circle, which the global analyses are."""
    missing = [name for name in ("eastward_wind", "northward_wind")
               if name not in frame.fields]
    if missing:
        raise ValueError(
            f"the re-balanced rotation start needs the analysis fields {', '.join(missing)}")
    longitude = np.asarray(frame.longitude, dtype=np.float64)
    if longitude.ndim != 1 or longitude.size < 4:
        raise ValueError("the re-balanced rotation start needs a 1-D longitude row")
    step = np.diff(longitude)
    if not np.allclose(step, step[0], rtol=0.0, atol=1.0e-6 * max(abs(step[0]), 1.0)) \
            or abs(abs(step[0]) * longitude.size - 360.0) > 1.0e-6 * 360.0:
        raise ValueError(
            "the re-balanced rotation start reads a zonal mean, which needs "
            "longitudes evenly spaced around the whole circle")
    u, v, record = rebalanced_winds(
        frame.fields["eastward_wind"].values, frame.fields["northward_wind"].values,
        np.asarray(frame.latitude, dtype=np.float64), rotation_factor,
        radius_m=radius_m, rotation_rate_s=rotation_rate_s)
    return {"eastward_wind": u, "northward_wind": v}, record


def rebalanced_state(model, bundle, rotation_factor):
    """The rule on a model state: ``(bundle, record)`` whose vorticity and
    divergence carry the world's winds; every other field is the input's
    own object.  ``model`` is any :class:`~arwen_global.dynamics.MoistHybridModel`
    on the state's transform (its own rotation rate is not read)."""
    from ..state import ArwenGlobalState

    transform = model.transform
    backend = transform.backend
    grid = model.grid_state(bundle.atmosphere, only=("u", "v"))
    u = np.asarray(backend.to_numpy(grid["u"]), dtype=np.float64)
    v = np.asarray(backend.to_numpy(grid["v"]), dtype=np.float64)
    latitude = np.rad2deg(np.asarray(transform.grid.lat_rad, dtype=np.float64))
    new_u, new_v, record = rebalanced_winds(
        u, v, latitude, rotation_factor, radius_m=float(transform.grid.radius_m))
    dtype = backend.float_dtype
    vorticity, divergence = model.vector.vordiv_from_wind(
        backend.asarray(new_u, dtype=dtype), backend.asarray(new_v, dtype=dtype))
    atmosphere = bundle.atmosphere.with_fields((
        vorticity, divergence, bundle.atmosphere.theta, bundle.atmosphere.log_surface_pressure,
        bundle.atmosphere.qv,
    ))
    return ArwenGlobalState(atmosphere, bundle.surface, bundle.physics_state), record


__all__ = [
    "POLAR_COSINE", "ROTATION_START_SCHEMA", "RULE_TEXT",
    "SLOWEST_REBALANCED_ROTATION", "rebalanced_state", "rebalanced_winds",
    "rebalanced_zonal_mean", "rotation_start_replacements",
]
