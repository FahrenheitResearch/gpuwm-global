"""The zonal-mean start: Earth's air averaged around each latitude circle.

A world with no Earth land (a waterworld, ``world.sea.land = "none"``)
starts from the analysis's zonal-mean atmosphere, so nothing in its first
state is anchored to a continent it does not have.  An invented world
(package 2.6, ``world/frame.py``) takes the same zonal-mean air and raises
it onto its own terrain with the pressure mover.

The rule (:func:`zonal_mean_atmosphere`), on the analysis frame before the
cold start reads a field: temperature, specific humidity, both wind
components on every pressure level, and surface pressure, are each replaced
by their mean along the frame's latitude row.  Every other plane is the
frame's own.  The mean is the plain mean over the row, which on the
regular latitude-longitude analysis grid is the area mean, so each row
keeps its mass of air exactly.  The rule reads a FLAT frame: a frame whose
terrain is not flat is refused, because averaging surface pressure across
mountains mixes air columns of different depths; the terrain edit
(package 1.1, scale 0) lowers it first.

The zonal mean of a field that is in thermal-wind balance is in balance
too (the balance is linear and its coefficients vary with latitude only),
so the start carries only the zonal-mean part of the analysis's own
imbalance and no more.  :func:`thermal_wind_imbalance` measures it on any
frame, the full analysis and its zonal mean alike, at any rotation rate
(the re-balanced rotation start of package 3.2 reads it at the world's).

``world_start_frame`` is the one entry the cold start's frame hook calls
for this package: the zonal-mean start where the world has no Earth land,
the re-balanced rotation start where the day is not Earth's
(:mod:`arwen_global.world.rotation_start`, package 3.2), then the sea-ice
rule (:mod:`arwen_global.world.sea`).
"""
from __future__ import annotations

import numpy as np

from ..constants import DRY_AIR_GAS_CONSTANT, EARTH_RADIUS_M, EARTH_ROTATION_RATE_S
from .sea import edit_sea_ice_frame, edited_frame

START_SCHEMA = "arwen-global.world-start/v1"

#: The fields the zonal-mean start averages: every atmospheric plane the
#: cold start reads except the terrain, which the rule requires flat.
ZONAL_MEAN_FIELDS = (
    "air_temperature", "specific_humidity", "eastward_wind",
    "northward_wind", "surface_pressure",
)
#: The terrain a flat frame may still carry, metres.  The terrain edit at
#: scale 0 writes exact zeros; an analysis's open ocean carries a few
#: metres of geoid or rounding noise at most.
FLAT_TERRAIN_TOLERANCE_M = 10.0
#: Where the imbalance is measured by default: away from the equator,
#: where the Coriolis parameter vanishes and geostrophy says nothing, and
#: from the top of the boundary layer to the tropopause.
IMBALANCE_LATITUDE_BAND_DEG = (20.0, 80.0)
IMBALANCE_PRESSURE_BAND_PA = (20000.0, 85000.0)


def needs_zonal_mean_start(world) -> bool:
    """True for a world with no Earth land: a waterworld, or an invented
    world (whose frame package 2.6 builds from the same zonal-mean air)."""
    return (not world.is_earth) and (world.sea.land == "none" or world.is_invented)


def zonal_mean_atmosphere(frame, *, flat_tolerance_m: float = FLAT_TERRAIN_TOLERANCE_M):
    """The frame's atmosphere replaced by its zonal mean.  Returns
    ``(frame, record)``; the input frame is not modified.  Refuses a frame
    whose terrain is not flat, and a field with a missing value."""
    missing = [name for name in (*ZONAL_MEAN_FIELDS, "terrain_height") if name not in frame.fields]
    if missing:
        raise ValueError(
            f"the zonal-mean start needs the analysis fields {', '.join(missing)}")
    terrain = np.asarray(frame.fields["terrain_height"].values, dtype=np.float64)
    highest = float(np.max(np.abs(terrain)))
    if highest > flat_tolerance_m:
        row, column = np.unravel_index(int(np.argmax(np.abs(terrain))), terrain.shape)
        raise ValueError(
            "the zonal-mean start reads a flat frame and this one carries "
            f"terrain of {highest:.0f} m (at {float(np.asarray(frame.latitude)[row]):.2f}, "
            f"{float(np.asarray(frame.longitude)[column]):.2f}); averaging surface "
            "pressure across mountains mixes columns of different depths, so "
            "lower the terrain first ([world.terrain] scale = 0)")
    replacements = {}
    removed = {}
    for name in ZONAL_MEAN_FIELDS:
        values = np.asarray(frame.fields[name].values, dtype=np.float64)
        if values.shape[-2:] != terrain.shape:
            raise ValueError(f"analysis field {name} is not on the frame's grid")
        bad = int(values.size - np.count_nonzero(np.isfinite(values)))
        if bad:
            raise ValueError(
                f"analysis field {name} carries {bad} missing values; its "
                "zonal mean would not be the analysis's")
        mean = values.mean(axis=-1, keepdims=True)
        replacements[name] = np.broadcast_to(mean, values.shape).copy()
        removed[name] = float(np.max(np.abs(values - mean)))
    ps_before = np.asarray(frame.fields["surface_pressure"].values, dtype=np.float64)
    weights = np.cos(np.deg2rad(np.asarray(frame.latitude, dtype=np.float64)))
    weights = np.clip(weights, 0.0, None)

    def global_mean(plane):
        return float(np.sum(plane.mean(axis=-1) * weights) / np.sum(weights))

    record = {
        "schema": START_SCHEMA,
        "rule": ("temperature, specific humidity, both wind components on every "
                 "pressure level and surface pressure each replaced by their "
                 "mean along the analysis's latitude row (the area mean on its "
                 "regular grid, so each row keeps its mass of air); every other "
                 "plane is the analysis's own"),
        "fields": list(ZONAL_MEAN_FIELDS),
        "largest_departure_removed": removed,
        "highest_terrain_m": highest,
        "global_mean_surface_pressure_pa": {
            "analysis": global_mean(ps_before),
            "start": global_mean(replacements["surface_pressure"]),
        },
    }
    return edited_frame(frame, replacements, "zonal_mean_start", record), record


def thermal_wind_imbalance(frame, levels_pa, *, rotation_rate_s: float = EARTH_ROTATION_RATE_S,
                           latitude_band_deg=IMBALANCE_LATITUDE_BAND_DEG,
                           pressure_band_pa=IMBALANCE_PRESSURE_BAND_PA) -> dict:
    """How far a frame's winds are from thermal-wind balance with its
    temperature, on its own pressure levels.

    The balance, at rotation rate Omega (f = 2 Omega sin(lat)), on a
    pressure surface:

        f du/dln p = (R/a) dT/dlat
        f dv/dln p = -(R / (a cos(lat))) dT/dlon

    The residual is the vector of the left side minus the right side
    (m s-2); the scale is the right side's size.  Both are reduced to an
    area-weighted root mean square over ``latitude_band_deg`` (both
    hemispheres, by absolute latitude) and the levels inside
    ``pressure_band_pa``, at points above the analysis's own ground
    (level pressure at most the surface pressure).  Temperature is the
    analysis's, not virtual: the moisture term is under 1 % of the
    gradient at these levels and the same in every frame compared.

    Returns ``{"rms_residual_m_s2", "rms_thermal_m_s2", "ratio", "points",
    ...}``: ``ratio`` is the residual over the scale, 0 for a balanced
    state.
    """
    levels = np.asarray(levels_pa, dtype=np.float64)
    temperature = np.asarray(frame.fields["air_temperature"].values, dtype=np.float64)
    u = np.asarray(frame.fields["eastward_wind"].values, dtype=np.float64)
    v = np.asarray(frame.fields["northward_wind"].values, dtype=np.float64)
    ps = np.asarray(frame.fields["surface_pressure"].values, dtype=np.float64)
    if temperature.ndim != 3 or temperature.shape[0] != levels.size:
        raise ValueError("temperature must be (levels, latitude, longitude) on levels_pa")
    lat = np.asarray(frame.latitude, dtype=np.float64)
    lon = np.asarray(frame.longitude, dtype=np.float64)
    phi = np.deg2rad(lat)
    ln_p = np.log(levels)
    f = (2.0 * rotation_rate_s * np.sin(phi))[None, :, None]
    du = np.gradient(u, ln_p, axis=0)
    dv = np.gradient(v, ln_p, axis=0)
    dt_dphi = np.gradient(temperature, phi, axis=1)
    dlam = np.deg2rad(float(np.mean(np.diff(lon))))
    dt_dlam = (np.roll(temperature, -1, axis=2) - np.roll(temperature, 1, axis=2)) / (2.0 * dlam)
    cos_phi = np.cos(phi)[None, :, None]
    coefficient = DRY_AIR_GAS_CONSTANT / EARTH_RADIUS_M
    thermal_u = coefficient * dt_dphi
    with np.errstate(divide="ignore", invalid="ignore"):
        thermal_v = -coefficient * dt_dlam / cos_phi
    residual_u = f * du - thermal_u
    residual_v = f * dv - thermal_v
    abs_lat = np.abs(lat)
    in_lat = (abs_lat >= latitude_band_deg[0]) & (abs_lat <= latitude_band_deg[1])
    in_p = (levels >= pressure_band_pa[0]) & (levels <= pressure_band_pa[1])
    mask = (in_p[:, None, None] & in_lat[None, :, None]
            & (levels[:, None, None] <= ps[None, :, :]))
    weight = np.broadcast_to(cos_phi, mask.shape) * mask
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("no point of the frame lies inside the measured band")
    residual_sq = np.where(mask, residual_u ** 2 + residual_v ** 2, 0.0)
    thermal_sq = np.where(mask, thermal_u ** 2 + thermal_v ** 2, 0.0)
    rms_residual = float(np.sqrt(np.sum(weight * residual_sq) / total))
    rms_thermal = float(np.sqrt(np.sum(weight * thermal_sq) / total))
    return {
        "rms_residual_m_s2": rms_residual,
        "rms_thermal_m_s2": rms_thermal,
        "ratio": rms_residual / rms_thermal if rms_thermal > 0.0 else float("inf"),
        "points": int(mask.sum()),
        "rotation_rate_s": float(rotation_rate_s),
        "latitude_band_deg": [float(x) for x in latitude_band_deg],
        "pressure_band_pa": [float(x) for x in pressure_band_pa],
        "levels_used_pa": [float(p) for p in levels[in_p]],
    }


def world_start_frame(world, frame):
    """This package's edit of the analysis frame, for the cold start's
    frame hook (after the terrain edit, before the cold start reads a
    field).  Returns ``(frame, record)``; on Earth, or on a world whose sea
    and land are Earth's, the frame is the input object and the record is
    None.

    In order: the zonal-mean atmosphere where the world has no Earth land
    and is not invented (an invented world's frame is built by package
    2.6, which takes :func:`zonal_mean_atmosphere` itself); then, where
    the day is not Earth's, the re-balanced rotation start of package 3.2
    (:mod:`arwen_global.world.rotation_start`), which gives the frame's
    mass field the winds that hold the same force at the world's rotation
    (on a zonal-mean frame, the plan's zonal wind re-derived at the new
    rate); then the world's sea-ice rule.  The sea TEMPERATURE acts later,
    on the cold start's open-water skin
    (:func:`arwen_global.world.sea.world_open_water_skin`).
    """
    if world.is_earth:
        return frame, None
    record = {}
    if world.sea.land == "none" and not world.is_invented:
        frame, record["zonal_mean_start"] = zonal_mean_atmosphere(frame)
    if world.changes_rotation:
        from .rotation_start import rotation_start_replacements

        replacements, record["rotation_start"] = rotation_start_replacements(
            frame, world.sun.rotation_factor)
        frame = edited_frame(frame, replacements, "rotation_start",
                             record["rotation_start"])
    if world.sea.ice != "analysis":
        frame, record["sea_ice"] = edit_sea_ice_frame(frame, world.sea)
    return frame, (record or None)


__all__ = [
    "FLAT_TERRAIN_TOLERANCE_M", "IMBALANCE_LATITUDE_BAND_DEG",
    "IMBALANCE_PRESSURE_BAND_PA", "START_SCHEMA", "ZONAL_MEAN_FIELDS",
    "needs_zonal_mean_start", "thermal_wind_imbalance", "world_start_frame",
    "zonal_mean_atmosphere",
]
