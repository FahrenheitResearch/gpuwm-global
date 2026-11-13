"""The invented-world frame (plan package 2.6).

An invented world (a ``[world.surface] folder``: Pangea, a painted map) has
no analysis of its own.  Its start is built here as an ordinary analysis
frame, so the cold start (``analysis_initial.analysis_initial_state``, 1,500
lines of measured rules) reads it exactly as it reads a GDAS or ERA5 frame
and nothing in it changes.  The frame is built by these stated rules:

1. **Earth's zonal-mean air.**  On every analysis pressure level and every
   latitude row of the Earth analysis, temperature, specific humidity and
   both wind components are averaged over the columns where that level is
   above the ground (:data:`MIN_ABOVE_GROUND_SHARE` of the row at least).
   A level that is below the ground on too much of its row takes the world's
   below-ground rule from the level above it: temperature continues downward
   at :data:`LAPSE_K_PER_M` (6.5 K per km) and specific humidity and wind are
   held.  The row's sea-level pressure is the mean of every column's surface
   pressure reduced to sea level through the same 6.5 K per km from its own
   surface air.  The rows are then interpolated in latitude onto the world.
2. **Raised onto the world's terrain by the pressure mover**
   (:func:`move_surface_pressure`): starting from the row's sea-level
   pressure at height 0, the hydrostatic equation is integrated layer by
   layer through the column's own virtual temperature, each layer at the
   constant lapse its two levels imply, to the world's terrain height.  At
   a height of exactly 0 it returns the sea-level pressure bit for bit.
3. **Air mass.**  ``world.terrain.air_mass = "conserve"`` (the default)
   multiplies the surface pressure by one global number so the world holds
   the Earth analysis's own amount of air (area-weighted mean surface
   pressure); ``"fill"`` keeps the sea-level pressure of rule 1.
4. **Land skin and soil from the air.**  The land skin is the zonal-mean
   air temperature at the world's surface pressure.  The soil layers run
   linearly in depth from that skin toward the deep-soil temperature at
   Noah's lower boundary (:data:`NOAH_ZBOT_M`, 8 m).
5. **Deep soil from Earth's zonal mean at sea level.**  The deep-soil
   temperature of a latitude is the zonal mean over Earth's land of the
   1-degree deep-soil climatology (``SOILTEMP``, which WPS carries at sea
   level and real.exe lowers by 0.0065 K/m of terrain), or, when no Earth
   statics are handed in, of the analysis's bottom soil layer raised to
   sea level by the same 0.0065 K/m.  Latitudes with no Earth land take the
   nearest rows' values by linear interpolation in latitude.  The world's
   value is that profile less 0.0065 K/m of the world's terrain, real.exe's
   own rule (:func:`deep_soil_plane`, which is also what package 2.4 writes
   as the world's ``SOILTEMP``, at sea level).
6. **Typed soil wetness.**  ``world.surface.soil_wetness`` is a fraction of
   saturation: the volumetric soil water of every layer is
   ``soil_wetness * MAXSMC`` of the world's soil class (uniform loam, STAS 6,
   MAXSMC 0.439), never below that class's dry limit ``DRYSMC`` (0.066).
7. **No snow and no land ice** anywhere.  Sea ice follows
   ``world.sea.ice``: ``"none"`` is ice-free; ``"zonal"`` is the zonal mean
   of the analysis's sea-ice fraction over its water columns, with the
   zonal-mean thickness of its frozen columns.
8. **The sea** follows ``world.sea.temperature``: ``"zonal-mean"`` is the
   zonal mean of the analysis's own open-water skin (never below sea
   water's freezing point, 271.35 K); the four published aquaplanet
   profiles (Neale and Hoskins 2000) are their closed forms.

The frame lives on the world map's own regular latitude-longitude grid
(the cold start's regridder accepts nothing else), carries every field the
cold start requires, and a ``world_frame`` record of every rule and number
above for the receipt.

Nothing here runs for Earth or for an edited Earth: the builder refuses a
world that is not invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math

import numpy as np

from ..constants import GRAVITY_M_S2

#: The gas constant and the virtual-temperature factor the cold start's own
#: hypsometric reduction uses (analysis_initial.DRY_AIR_GAS_CONSTANT and
#: surface_virtual_temperature), so a step-0 move onto the model's terrain
#: and this frame's mover integrate the same air.
DRY_AIR_GAS_CONSTANT = 287.0
VIRTUAL_FACTOR = 0.608
#: The world's below-ground rule and real.exe's deep-soil terrain
#: correction, both 6.5 K per km.
LAPSE_K_PER_M = 0.0065
#: A level enters a row's zonal mean only where it is above the ground on at
#: least this share of the row's columns; below that it takes the
#: below-ground rule from the level above.
MIN_ABOVE_GROUND_SHARE = 0.1
#: Noah's lower boundary, where the deep-soil temperature is held (WRF
#: Noah ZBOT = -8 m).
NOAH_ZBOT_M = 8.0
#: Sea water's freezing point; no zonal-mean sea is colder.
SEA_WATER_FREEZING_K = 271.35
#: The invented world's one soil class: STAS 6, loam (plan 4.4).
LOAM_SOIL_CATEGORY = 6
#: Zero Celsius, for the aquaplanet profiles.
ZERO_CELSIUS_K = 273.15
#: The aquaplanet profiles' equatorial maximum, degrees Celsius.
AQUA_PEAK_C = 27.0

FRAME_SCHEMA = "arwen-global.world-frame/v1"


# -- small helpers ---------------------------------------------------------


@dataclass
class WorldField:
    """One field of a frame, shaped as the decoder's are."""

    values: np.ndarray
    source: str = "primary"


@dataclass
class WorldMap:
    """An invented world's terrain and land on a regular global grid.

    ``latitude_deg`` and ``longitude_deg`` are 1-D, regular, reaching both
    poles and closing the longitude ring (the cold start's regridder refuses
    anything else); ``terrain_m`` and ``land_fraction`` are
    ``(nlat, nlon)``.  ``terrain_m`` is the terrain the dynamics will carry
    (package 2.5's filtered map); this module takes it as given."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    terrain_m: np.ndarray
    land_fraction: np.ndarray

    def __post_init__(self):
        self.latitude_deg = np.asarray(self.latitude_deg, dtype=np.float64)
        self.longitude_deg = np.asarray(self.longitude_deg, dtype=np.float64)
        self.terrain_m = np.asarray(self.terrain_m, dtype=np.float64)
        self.land_fraction = np.clip(
            np.asarray(self.land_fraction, dtype=np.float64), 0.0, 1.0)
        shape = (self.latitude_deg.size, self.longitude_deg.size)
        for name in ("terrain_m", "land_fraction"):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(
                    f"world map {name} is {value.shape}, not the {shape} its "
                    "latitudes and longitudes name")
            if not np.isfinite(value).all():
                raise ValueError(f"world map {name} carries non-finite values")
        lat = self.latitude_deg
        ascending = lat[::-1] if lat[0] > lat[-1] else lat
        if (lat.size < 3 or abs(ascending[0] + 90.0) > 1e-6
                or abs(ascending[-1] - 90.0) > 1e-6):
            raise ValueError(
                "world map latitudes must run pole to pole on a regular grid: "
                "the cold start's regridder reads only such a frame")
        steps = np.diff(ascending)
        if np.max(np.abs(steps - steps[0])) > 1e-6:
            raise ValueError("world map latitudes are not regular")
        dlon = np.diff(self.longitude_deg)
        if (np.max(np.abs(dlon - dlon[0])) > 1e-6
                or abs(dlon[0] * self.longitude_deg.size - 360.0) > 1e-3):
            raise ValueError("world map longitudes must be a regular ring of 360 degrees")


def world_map_from_elevation(world, elevation_m, latitude_deg, longitude_deg,
                             transform, *, step_deg=None):
    """``(WorldMap, report)`` from the world's elevation map, through
    package 2.5 (``arwen_global.world.terrain``): the map averaged onto the
    model's cells and spectrally filtered, sampled exactly on a regular
    quarter-degree frame, with the map's land averaged onto the frame's
    cells.  ``report`` is the false-height report the build prints.  A
    frame on the map's own one-degree grid would land up to 249 m from the
    filtered terrain once the cold start interpolates and truncates it
    again (measured at T255 on the 250 Ma map); a quarter degree, 21 m."""
    from .terrain import FRAME_STEP_DEG, dynamics_terrain_for_world, frame_planes

    built = dynamics_terrain_for_world(
        world, elevation_m, latitude_deg, longitude_deg, transform)
    lat, lon, terrain, land = frame_planes(
        built, elevation_m, latitude_deg, longitude_deg,
        sea_level_m=float(world.sea.level_m),
        step_deg=FRAME_STEP_DEG if step_deg is None else step_deg)
    return WorldMap(lat, lon, terrain, land), built.report


def cell_area_weights(latitude_deg) -> np.ndarray:
    """Relative area of each row's cell on a regular pole-to-pole grid:
    the sine band between the cell's edges (half cells at the poles)."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    order = np.argsort(lat)
    ascending = lat[order]
    edges = np.concatenate([
        [-90.0], 0.5 * (ascending[1:] + ascending[:-1]), [90.0]])
    band = np.diff(np.sin(np.deg2rad(edges)))
    weights = np.empty_like(band)
    weights[order] = band
    return weights


def area_mean(plane, latitude_deg) -> float:
    """Area-weighted global mean of a ``(nlat, nlon)`` plane on a regular
    pole-to-pole grid."""
    values = np.asarray(plane, dtype=np.float64)
    weights = cell_area_weights(latitude_deg)
    return float(np.sum(values.mean(axis=1) * weights) / np.sum(weights))


def _ascending_rows(frame):
    """Indices that put a frame's latitude rows south to north."""
    lat = np.asarray(frame.latitude, dtype=np.float64)
    return np.argsort(lat, kind="stable"), np.sort(lat)


def _bracket(levels, pressure):
    """For each pressure, the index ``k`` of the lower (higher-pressure)
    level of the layer it falls in, clipped to the level range, on ascending
    (top-first) ``levels``."""
    index = np.searchsorted(levels, pressure, side="left")
    return np.clip(index, 1, levels.size - 1)


def _power_law_at(levels, values, pressure):
    """``values`` (nlev, ncol) at ``pressure`` (ncol), the layer between
    the two bracketing levels taken at constant lapse (value a power of p);
    below the bottom level, continued at :data:`LAPSE_K_PER_M`; above the top
    level, held.  Exact on a constant-lapse column."""
    k = _bracket(levels, pressure)
    cols = np.arange(pressure.size)
    upper_p, lower_p = levels[k - 1], levels[k]
    upper_v, lower_v = values[k - 1, cols], values[k, cols]
    exponent = np.log(upper_v / lower_v) / np.log(upper_p / lower_p)
    inside = lower_v * (pressure / lower_p) ** exponent
    bottom_p, bottom_v = levels[-1], values[-1]
    below = bottom_v * (pressure / bottom_p) ** (
        DRY_AIR_GAS_CONSTANT * LAPSE_K_PER_M / GRAVITY_M_S2)
    out = np.where(pressure > bottom_p, below, inside)
    return np.where(pressure < levels[0], values[0], out)


def _linear_ln_p_at(levels, values, pressure):
    """``values`` (nlev, ncol) at ``pressure``: linear in ln p between
    levels, held beyond either end (the cold start's rule for humidity and
    wind)."""
    k = _bracket(levels, pressure)
    cols = np.arange(pressure.size)
    ln_upper, ln_lower = np.log(levels[k - 1]), np.log(levels[k])
    weight = (np.log(pressure) - ln_upper) / (ln_lower - ln_upper)
    weight = np.clip(weight, 0.0, 1.0)
    return values[k - 1, cols] * (1.0 - weight) + values[k, cols] * weight


# -- the pressure mover ------------------------------------------------------


def _log_mean(a, b):
    """The logarithmic mean of two positive arrays: the mean temperature of
    a constant-lapse layer per unit ln p (exactly ``a`` where ``a == b``)."""
    ratio = a / b
    close = np.abs(ratio - 1.0) < 1e-12
    safe = np.where(close, 2.0, ratio)
    return np.where(close, 0.5 * (a + b), (a - b) / np.log(safe))


def move_surface_pressure(levels_pa, virtual_temperature_k, p_from_pa,
                          z_from_m, z_to_m) -> np.ndarray:
    """Surface pressure moved hydrostatically from height ``z_from_m`` (where
    it is ``p_from_pa``) to ``z_to_m`` through the column's own air.

    ``levels_pa`` is the 1-D ascending (top-first) pressure-level set;
    ``virtual_temperature_k`` is ``(nlev, ...)`` on it; the other arguments
    broadcast to the surface shape.  Each layer between two levels is taken
    at the constant lapse its two virtual temperatures imply, so a
    constant-lapse column moves by its closed form exactly; below the bottom
    level the air continues at 6.5 K per km.  Where ``z_to_m == z_from_m``
    the input pressure is returned bit for bit.  A target above the top
    level is refused.  Works in both directions (lowering and raising)."""
    levels = np.asarray(levels_pa, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 2 or np.any(np.diff(levels) <= 0):
        raise ValueError("pressure levels must be 1-D and ascending (top first)")
    tv = np.asarray(virtual_temperature_k, dtype=np.float64)
    surface_shape = np.broadcast_shapes(
        tv.shape[1:], np.shape(p_from_pa), np.shape(z_from_m), np.shape(z_to_m))
    tv = np.broadcast_to(tv, (levels.size, *surface_shape)).reshape(levels.size, -1)
    p_from = np.broadcast_to(np.asarray(p_from_pa, dtype=np.float64), surface_shape)
    z_from = np.broadcast_to(np.asarray(z_from_m, dtype=np.float64), surface_shape)
    z_to = np.broadcast_to(np.asarray(z_to_m, dtype=np.float64), surface_shape)
    p0 = p_from.reshape(-1)
    rd_g = DRY_AIR_GAS_CONSTANT / GRAVITY_M_S2

    # Heights of every level above the bottom level.
    thickness = rd_g * np.log(levels[1:] / levels[:-1])[:, None] * _log_mean(
        tv[:-1], tv[1:])
    height = np.zeros_like(tv)
    height[:-1] = np.cumsum(thickness[::-1], axis=0)[::-1]

    # Height of the starting surface above the bottom level.
    tv_from = _power_law_at(levels, tv, p0)
    k = _bracket(levels, p0)
    cols = np.arange(p0.size)
    above_bottom = rd_g * np.log(levels[k] / p0) * _log_mean(tv[k, cols], tv_from) \
        + height[k, cols]
    below_bottom = -rd_g * np.log(p0 / levels[-1]) * _log_mean(tv[-1], tv_from)
    z_start = np.where(p0 > levels[-1], below_bottom, above_bottom)
    target = z_start + (z_to - z_from).reshape(-1)
    if np.any(target > height[0]):
        worst = float(np.max(target - height[0]))
        raise ValueError(
            "the pressure mover was asked to raise the surface above the "
            f"analysis top level ({float(levels[0]):g} Pa) by {worst:.0f} m; "
            "no column carries air there")

    # The layer the target falls in: its lower level is the first one (from
    # the top) whose height is at or below the target.
    lower = np.argmax(height <= target[None], axis=0)
    lower = np.where(target < 0.0, levels.size - 1, np.maximum(lower, 1))
    upper = lower - 1
    z_lower, z_upper = height[lower, cols], height[upper, cols]
    t_lower, t_upper = tv[lower, cols], tv[upper, cols]
    lapse = np.where(target < 0.0, LAPSE_K_PER_M,
                     (t_lower - t_upper) / np.maximum(z_upper - z_lower, 1e-9))
    rise = target - z_lower
    isothermal = np.abs(lapse) < 1e-9
    safe_lapse = np.where(isothermal, 1.0, lapse)
    t_target = t_lower - lapse * rise
    p_lapse = levels[lower] * (t_target / t_lower) ** (1.0 / (rd_g * safe_lapse))
    p_isothermal = levels[lower] * np.exp(-rise / (rd_g * t_lower))
    moved = np.where(isothermal, p_isothermal, p_lapse)
    unchanged = (z_to == z_from).reshape(-1)
    return np.where(unchanged, p0, moved).reshape(surface_shape)


# -- Earth's zonal-mean air ---------------------------------------------------


@dataclass
class ZonalColumn:
    """Earth's zonal-mean air at sea level, by latitude row (south first)."""

    latitude_deg: np.ndarray        # (nrow,)
    levels_pa: np.ndarray           # (nlev,) ascending
    temperature_k: np.ndarray       # (nlev, nrow)
    specific_humidity: np.ndarray   # (nlev, nrow)
    eastward_wind: np.ndarray       # (nlev, nrow)
    northward_wind: np.ndarray      # (nlev, nrow)
    sea_level_pressure_pa: np.ndarray  # (nrow,)
    below_ground_filled: int = 0    # (level, row) entries the rule filled
    record: dict = field(default_factory=dict)

    def at(self, latitude_deg):
        """The column interpolated linearly in latitude onto ``latitude_deg``:
        ``(temperature, humidity, u, v)`` each ``(nlev, n)`` and the
        sea-level pressure ``(n,)``."""
        lat = np.asarray(latitude_deg, dtype=np.float64)

        def rows(stack):
            return np.stack([np.interp(lat, self.latitude_deg, level) for level in stack])

        return (rows(self.temperature_k), np.clip(rows(self.specific_humidity), 0.0, None),
                rows(self.eastward_wind), rows(self.northward_wind),
                np.interp(lat, self.latitude_deg, self.sea_level_pressure_pa))


def earth_zonal_mean_air(frame, levels_pa) -> ZonalColumn:
    """Rule 1 of the module docstring, on the Earth analysis ``frame`` whose
    pressure levels (Pa, ascending) are ``levels_pa``."""
    levels = np.asarray(levels_pa, dtype=np.float64)
    order, lat = _ascending_rows(frame)

    def stack(name):
        values = np.asarray(frame.fields[name].values, dtype=np.float64)
        return values[:, order, :]

    temperature = stack("air_temperature")
    humidity = np.clip(stack("specific_humidity"), 0.0, None)
    u = stack("eastward_wind")
    v = stack("northward_wind")
    ps = np.asarray(frame.fields["surface_pressure"].values, dtype=np.float64)[order]
    terrain = np.asarray(frame.fields["terrain_height"].values, dtype=np.float64)[order]
    nlev, nrow, nlon = temperature.shape

    above = levels[:, None, None] <= ps[None]
    count = above.sum(axis=2)
    usable = count >= max(1, math.ceil(MIN_ABOVE_GROUND_SHARE * nlon))

    def masked_mean(values):
        total = np.where(above, values, 0.0).sum(axis=2)
        return np.where(usable, total / np.maximum(count, 1), np.nan)

    t_zm, q_zm, u_zm, v_zm = (masked_mean(x) for x in (temperature, humidity, u, v))
    if not usable[0].all():
        raise ValueError(
            "the analysis's top level is below the ground somewhere; it is not "
            "an isobaric analysis this rule can read")
    filled = 0
    exponent = DRY_AIR_GAS_CONSTANT * LAPSE_K_PER_M / GRAVITY_M_S2
    for k in range(1, nlev):
        hole = ~usable[k]
        if hole.any():
            filled += int(hole.sum())
            t_zm[k, hole] = t_zm[k - 1, hole] * (levels[k] / levels[k - 1]) ** exponent
            q_zm[k, hole] = q_zm[k - 1, hole]
            u_zm[k, hole] = u_zm[k - 1, hole]
            v_zm[k, hole] = v_zm[k - 1, hole]
            usable[k, hole] = True

    # Every column's surface pressure reduced to sea level from its own
    # surface air at 6.5 K per km, then averaged along the row.
    flat_ps = ps.reshape(-1)
    tv_levels = (temperature * (1.0 + VIRTUAL_FACTOR * humidity)).reshape(nlev, -1)
    tv_surface = _power_law_at(levels, tv_levels, flat_ps).reshape(nrow, nlon)
    sea_level = ps * (1.0 + LAPSE_K_PER_M * terrain / tv_surface) ** (1.0 / exponent)
    p0 = sea_level.mean(axis=1)
    return ZonalColumn(
        latitude_deg=lat, levels_pa=levels, temperature_k=t_zm,
        specific_humidity=q_zm, eastward_wind=u_zm, northward_wind=v_zm,
        sea_level_pressure_pa=p0, below_ground_filled=filled,
        record={
            "rule": (
                "zonal mean over the columns where each level is above the "
                f"ground (at least {MIN_ABOVE_GROUND_SHARE:g} of the row); a "
                "level below the ground on more of its row continues the "
                f"level above at {LAPSE_K_PER_M * 1000:g} K/km with specific "
                "humidity and wind held"),
            "sea_level_pressure_rule": (
                "row mean of every column's surface pressure reduced to sea "
                f"level from its surface virtual temperature at "
                f"{LAPSE_K_PER_M * 1000:g} K/km"),
            "below_ground_level_rows_filled": filled,
            "earth_rows": int(nrow),
            "earth_columns_per_row": int(nlon),
            "sea_level_pressure_pa": {"min": float(p0.min()), "max": float(p0.max())},
        },
    )


# -- the sea, the ice, the deep soil -----------------------------------------


def _fill_rows(lat, values, have):
    """``values`` on the rows where ``have``, linearly interpolated in
    latitude onto the others (held beyond the outermost)."""
    if not np.any(have):
        return None
    return np.interp(lat, lat[have], values[have])


def aquaplanet_sea_temperature_k(profile: str, latitude_deg) -> np.ndarray:
    """The four published aquaplanet sea-surface temperatures (Neale and
    Hoskins 2000): 27 C at the equator and 0 C at and beyond 60 degrees."""
    phi = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    inside = np.abs(phi) < np.pi / 3.0
    s = np.sin(1.5 * phi) ** 2
    control = 1.0 - s
    flat = 1.0 - s ** 2
    shapes = {
        "aqua-control": control,
        "aqua-flat": flat,
        "aqua-qobs": 0.5 * (control + flat),
        "aqua-peaked": 1.0 - 3.0 * np.abs(phi) / np.pi,
    }
    if profile not in shapes:
        raise ValueError(f"no aquaplanet profile named {profile!r}")
    return ZERO_CELSIUS_K + AQUA_PEAK_C * np.where(inside, shapes[profile], 0.0)


def earth_sea_by_latitude(frame, latitude_deg, *, temperature: str, ice: str):
    """The sea of rule 7 and 8 on ``latitude_deg``: ``(open_water_skin_k,
    ice_fraction, ice_thickness_m, ice_skin_k, record)`` per latitude."""
    order, lat = _ascending_rows(frame)
    land = np.asarray(frame.fields["land_fraction"].values, dtype=np.float64)[order] >= 0.5
    skin = np.asarray(frame.fields["skin_temperature"].values, dtype=np.float64)[order]
    ice_src = np.asarray(frame.fields["sea_ice_fraction"].values, dtype=np.float64)[order]
    ice_src = np.where(np.isfinite(ice_src), np.clip(ice_src, 0.0, 1.0), 0.0)
    thick_src = np.asarray(frame.fields["sea_ice_thickness"].values, dtype=np.float64)[order]
    thick_src = np.where(np.isfinite(thick_src), np.maximum(thick_src, 0.0), 0.0)
    water = ~land
    frozen = water & (ice_src >= 0.5)
    open_water = water & ~frozen
    target = np.asarray(latitude_deg, dtype=np.float64)
    record = {"temperature": temperature, "ice": ice}

    if temperature == "zonal-mean":
        count = open_water.sum(axis=1)
        mean = np.where(open_water, skin, 0.0).sum(axis=1) / np.maximum(count, 1)
        row = _fill_rows(lat, mean, count > 0)
        if row is None:
            raise ValueError("the analysis has no open water to take a zonal-mean sea from")
        sea = np.maximum(np.interp(target, lat, row), SEA_WATER_FREEZING_K)
        record["temperature_rule"] = (
            "zonal mean of the analysis's open-water skin (land fraction "
            "below 0.5, sea-ice fraction below 0.5), rows with none "
            "interpolated in latitude, never below "
            f"{SEA_WATER_FREEZING_K} K")
    elif temperature in ("aqua-control", "aqua-peaked", "aqua-flat", "aqua-qobs"):
        sea = aquaplanet_sea_temperature_k(temperature, target)
        record["temperature_rule"] = (
            f"the published aquaplanet profile {temperature} (Neale and "
            "Hoskins 2000)")
    else:
        raise ValueError(
            f"an invented world's sea cannot be {temperature!r}: it has no "
            "analysis sea of its own")

    if ice == "none":
        fraction = np.zeros_like(target)
        thickness = np.zeros_like(target)
        ice_skin = np.full_like(target, SEA_WATER_FREEZING_K)
        record["ice_rule"] = "no sea ice"
    elif ice == "zonal":
        count = water.sum(axis=1)
        mean = np.where(water, ice_src, 0.0).sum(axis=1) / np.maximum(count, 1)
        row = _fill_rows(lat, mean, count > 0)
        fraction = np.clip(np.interp(target, lat, row), 0.0, 1.0)
        fcount = frozen.sum(axis=1)
        thick_row = np.where(frozen, thick_src, 0.0).sum(axis=1) / np.maximum(fcount, 1)
        skin_row = np.where(frozen, skin, 0.0).sum(axis=1) / np.maximum(fcount, 1)
        if np.any(fcount > 0):
            thickness = np.interp(target, lat, _fill_rows(lat, thick_row, fcount > 0))
            ice_skin = np.minimum(
                np.interp(target, lat, _fill_rows(lat, skin_row, fcount > 0)),
                SEA_WATER_FREEZING_K)
        else:
            thickness = np.zeros_like(target)
            ice_skin = np.full_like(target, SEA_WATER_FREEZING_K)
        thickness = np.where(fraction > 0.0, thickness, 0.0)
        record["ice_rule"] = (
            "zonal mean of the analysis's sea-ice fraction over its water "
            "columns; thickness and skin the zonal means over its frozen "
            "columns (skin capped at sea water's freezing point)")
    else:
        raise ValueError(
            f"an invented world's sea ice cannot be {ice!r}: it has no "
            "analysis ice of its own")
    return sea, fraction, thickness, ice_skin, record


def sea_level_deep_soil_by_latitude(latitude_deg, deep_soil_k, land, *,
                                    terrain_m=None):
    """Earth's deep-soil temperature at sea level, zonally averaged over
    land, on ``latitude_deg`` rows (any order) of a ``(nlat, nlon)`` plane.

    ``deep_soil_k`` is either the WPS climatology ``SOILTEMP`` (already at
    sea level: leave ``terrain_m`` None) or a temperature at the ground
    (give ``terrain_m`` and it is raised by 0.0065 K/m).  Rows with no land
    take the nearest rows' values, linear in latitude.  Returns
    ``(ascending latitudes, profile)``."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    order = np.argsort(lat, kind="stable")
    values = np.asarray(deep_soil_k, dtype=np.float64)[order]
    if terrain_m is not None:
        values = values + LAPSE_K_PER_M * np.asarray(terrain_m, dtype=np.float64)[order]
    land_mask = (np.asarray(land, dtype=np.float64)[order] >= 0.5) & np.isfinite(values)
    count = land_mask.sum(axis=1)
    mean = np.where(land_mask, values, 0.0).sum(axis=1) / np.maximum(count, 1)
    profile = _fill_rows(lat[order], mean, count > 0)
    if profile is None:
        raise ValueError("Earth's deep soil has no land column to average")
    return lat[order], profile


def deep_soil_plane(profile_latitude_deg, profile_k, latitude_deg, terrain_m,
                    *, at_sea_level=False) -> np.ndarray:
    """The world's deep-soil temperature on its ``(nlat, nlon)`` grid: the
    sea-level profile interpolated to the world's latitudes, less 0.0065 K/m
    of the world's terrain (real.exe's rule).  ``at_sea_level=True`` leaves
    the terrain out: that is the plane package 2.4 writes as the world's
    ``SOILTEMP``, which the statics resolver lowers by the same rule."""
    terrain = np.asarray(terrain_m, dtype=np.float64)
    row = np.interp(np.asarray(latitude_deg, dtype=np.float64),
                    np.asarray(profile_latitude_deg, dtype=np.float64),
                    np.asarray(profile_k, dtype=np.float64))
    plane = np.broadcast_to(row[:, None], terrain.shape).copy()
    if not at_sea_level:
        plane = plane - LAPSE_K_PER_M * terrain
    return plane


def soil_water_from_wetness(soil_wetness: float, soil_category: int = LOAM_SOIL_CATEGORY,
                            tables=None) -> tuple[float, dict]:
    """Volumetric soil water for a typed soil wetness (fraction of
    saturation): ``wetness * MAXSMC``, never below ``DRYSMC``, from the Noah
    soil table's STAS row of ``soil_category``."""
    if tables is None:
        from ..core.noah import load_tables

        tables = load_tables()
    index = int(soil_category) - 1
    maxsmc = float(tables.maxsmc[index])
    drysmc = float(tables.drysmc[index])
    wetness = float(soil_wetness)
    if not 0.0 <= wetness <= 1.0:
        raise ValueError(f"soil wetness must lie in [0, 1], got {wetness:g}")
    water = max(wetness * maxsmc, drysmc)
    return water, {
        "soil_wetness": wetness,
        "rule": "volumetric soil water = soil_wetness * MAXSMC, never below DRYSMC",
        "soil_category": int(soil_category),
        "maxsmc": maxsmc,
        "drysmc": drysmc,
        "volumetric_soil_water": water,
    }


# -- the frame ------------------------------------------------------------------


@dataclass
class WorldFrame:
    """A frame the cold start reads as it reads a decoded analysis."""

    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_values: np.ndarray
    fields: dict
    mapping_sha256: object = None
    input_sha256: object = None
    source_cycle: datetime | None = None
    valid_time: datetime | None = None
    snow_on_sea_ice: bool = True
    fill: object = None
    #: Every rule and number of this frame, for the receipt.
    world_frame: dict = field(default_factory=dict)
    #: The pressure mover's output on the world grid, before nothing else
    #: touches it: the frame's surface pressure is this array.
    mover_surface_pressure_pa: np.ndarray | None = None


def invented_world_frame(world, earth_frame, world_map: WorldMap, *,
                         levels_pa=None, soil_layer_bounds_m=None,
                         earth_deep_soil=None,
                         soil_category: int = LOAM_SOIL_CATEGORY) -> WorldFrame:
    """The invented world's start, as an analysis frame (module docstring).

    ``world`` is the run's :class:`~arwen_global.world.spec.WorldSpec` and
    must be invented.  ``earth_frame`` is the decoded Earth analysis of the
    start time (its valid time becomes the world's), ``world_map`` the
    world's terrain and land.  ``levels_pa`` are the analysis levels in Pa
    (default: ``earth_frame.vertical_values`` taken as Pa);
    ``soil_layer_bounds_m`` the depth bounds of the analysis mapping's soil
    layers (default Noah's four); ``earth_deep_soil`` an optional
    ``(latitude_deg, SOILTEMP, LANDMASK)`` from Earth's real statics cache
    (default: the analysis's bottom soil layer raised to sea level)."""
    if world is None or not getattr(world, "is_invented", False):
        raise ValueError(
            "the invented-world frame is built only for a world with its own "
            "surface ([world.surface] folder); Earth and an edited Earth start "
            "from the analysis")
    levels = np.asarray(
        earth_frame.vertical_values if levels_pa is None else levels_pa,
        dtype=np.float64)
    if levels.ndim != 1 or np.any(np.diff(levels) <= 0):
        raise ValueError("analysis pressure levels must be ascending top-to-bottom")
    if levels[-1] < 50000.0:
        raise ValueError(
            f"the analysis's deepest level reads {levels[-1]:g}, which is not "
            "a pressure in Pa; pass levels_pa from analysis_levels_pa, which "
            "reads the unit the mapping declares")
    if soil_layer_bounds_m is None:
        from ..analysis_initial import noah_soil_layer_bounds_m

        soil_layer_bounds_m = noah_soil_layer_bounds_m()
    bounds = tuple((float(a), float(b)) for a, b in soil_layer_bounds_m)

    lat_w = world_map.latitude_deg
    nlat, nlon = lat_w.size, world_map.longitude_deg.size
    terrain = world_map.terrain_m
    land = world_map.land_fraction >= 0.5

    # Rule 1: Earth's zonal-mean air on the world's rows.
    column = earth_zonal_mean_air(earth_frame, levels)
    t_rows, q_rows, u_rows, v_rows, p0_rows = column.at(lat_w)
    nlev = levels.size

    def broadcast(rows):
        return np.repeat(rows[:, :, None], nlon, axis=2)

    temperature = broadcast(t_rows)
    humidity = broadcast(q_rows)
    eastward = broadcast(u_rows)
    northward = broadcast(v_rows)
    tv = temperature * (1.0 + VIRTUAL_FACTOR * humidity)
    sea_level = np.repeat(p0_rows[:, None], nlon, axis=1)

    # Rules 2 and 3: the mover, then the air-mass rule.
    raised = move_surface_pressure(levels, tv, sea_level, 0.0, terrain)
    order, lat_e = _ascending_rows(earth_frame)
    earth_ps = np.asarray(earth_frame.fields["surface_pressure"].values,
                          dtype=np.float64)[order]
    earth_mean = area_mean(earth_ps, lat_e)
    raised_mean = area_mean(raised, lat_w)
    air_mass = world.terrain.air_mass
    if air_mass == "conserve":
        factor = earth_mean / raised_mean
        surface_pressure = raised * factor
    elif air_mass == "fill":
        factor = 1.0
        surface_pressure = raised
    else:
        raise ValueError(f"unknown air-mass rule {air_mass!r}")

    # Rule 8 and 7: the sea and its ice.
    sea_k, ice_rows, thick_rows, ice_skin_rows, sea_record = earth_sea_by_latitude(
        earth_frame, lat_w, temperature=world.sea.temperature, ice=world.sea.ice)
    water = ~land
    ice_fraction = np.where(water, ice_rows[:, None], 0.0)
    ice_thickness = np.where(water, thick_rows[:, None], 0.0)
    frozen = ice_fraction >= 0.5

    # Rule 4: the land skin is the air at the surface.
    flat_ps = surface_pressure.reshape(-1)
    surface_air = _power_law_at(levels, temperature.reshape(nlev, -1), flat_ps).reshape(
        nlat, nlon)
    skin = np.where(land, surface_air,
                    np.where(frozen, ice_skin_rows[:, None], sea_k[:, None]))

    # Rule 5: the deep soil, and the soil layers between skin and deep soil.
    if earth_deep_soil is not None:
        e_lat, e_soiltemp, e_land = earth_deep_soil
        p_lat, profile = sea_level_deep_soil_by_latitude(e_lat, e_soiltemp, e_land)
        deep_source = "Earth's statics SOILTEMP (sea level), zonal mean over land"
    else:
        soil_src = np.asarray(earth_frame.fields["soil_temperature"].values,
                              dtype=np.float64)
        e_terrain = np.asarray(earth_frame.fields["terrain_height"].values, dtype=np.float64)
        e_land = np.asarray(earth_frame.fields["land_fraction"].values, dtype=np.float64)
        p_lat, profile = sea_level_deep_soil_by_latitude(
            earth_frame.latitude, soil_src[-1], e_land, terrain_m=e_terrain)
        deep_source = (
            "the analysis's bottom soil layer raised to sea level at "
            f"{LAPSE_K_PER_M:g} K/m, zonal mean over land")
    deep = deep_soil_plane(p_lat, profile, lat_w, terrain)
    mids = np.asarray([0.5 * (a + b) for a, b in bounds])
    weight = np.clip(mids / NOAH_ZBOT_M, 0.0, 1.0)[:, None, None]
    soil_temperature = skin[None] + (deep[None] - skin[None]) * weight
    soil_temperature = np.where(land[None], soil_temperature, np.nan)

    # Rule 6: typed soil wetness.
    water_content, wetness_record = soil_water_from_wetness(
        world.surface.soil_wetness, soil_category)
    soil_moisture = np.where(land[None], water_content,
                             np.nan) * np.ones((len(bounds), 1, 1))

    # Rule 7: no snow, anywhere.
    zeros = np.zeros((nlat, nlon))
    soil_source = getattr(earth_frame.fields["soil_temperature"], "source", "primary")

    fields = {
        "air_temperature": WorldField(temperature),
        "specific_humidity": WorldField(humidity),
        "eastward_wind": WorldField(eastward),
        "northward_wind": WorldField(northward),
        "surface_pressure": WorldField(surface_pressure),
        "terrain_height": WorldField(terrain.copy()),
        "skin_temperature": WorldField(skin),
        "land_fraction": WorldField(world_map.land_fraction.copy()),
        # The soil keeps the source label of the analysis soil it stands in
        # for, so the cold start reads its layer depths from the same
        # mapping the bounds above came from.
        "soil_temperature": WorldField(soil_temperature, soil_source),
        "volumetric_soil_moisture": WorldField(soil_moisture, soil_source),
        "sea_ice_fraction": WorldField(ice_fraction),
        "sea_ice_thickness": WorldField(ice_thickness),
        "snow_water_equivalent": WorldField(zeros.copy()),
        "snow_depth": WorldField(zeros.copy()),
    }
    for name, value in fields.items():
        finite = np.isfinite(value.values)
        if name in ("soil_temperature", "volumetric_soil_moisture"):
            finite = finite | ~land[None]
        if not np.all(finite):
            raise ValueError(f"the invented-world frame's {name} is not finite")

    record = {
        "schema": FRAME_SCHEMA,
        "world": world.record() if hasattr(world, "record") else None,
        "grid": {"rows": int(nlat), "columns": int(nlon)},
        "air": column.record,
        "mover": (
            "hydrostatic, layer by layer through the zonal-mean column's "
            "virtual temperature, each layer at the constant lapse its two "
            "levels imply, from the row's sea-level pressure at 0 m to the "
            "world's terrain"),
        "air_mass": {
            "rule": air_mass,
            "factor": float(factor),
            "earth_mean_surface_pressure_pa": earth_mean,
            "raised_mean_surface_pressure_pa": raised_mean,
            "world_mean_surface_pressure_pa": area_mean(surface_pressure, lat_w),
        },
        "surface_pressure_pa": {
            "min": float(surface_pressure.min()), "max": float(surface_pressure.max())},
        "terrain_m": {"min": float(terrain.min()), "max": float(terrain.max())},
        "land_share": float(area_mean(land.astype(np.float64), lat_w)),
        "land_skin": "the zonal-mean air temperature at the world's surface pressure",
        "soil_temperature": (
            "linear in depth from the skin toward the deep-soil temperature at "
            f"{NOAH_ZBOT_M:g} m, at the mid-depth of each layer {list(bounds)}"),
        "deep_soil": {
            "source": deep_source,
            "terrain_rule": f"sea-level profile less {LAPSE_K_PER_M:g} K/m of the world's terrain",
            "land_min_k": float(deep[land].min()) if land.any() else None,
            "land_max_k": float(deep[land].max()) if land.any() else None,
        },
        "soil_water": wetness_record,
        "snow": "none anywhere",
        "land_ice": "none",
        "sea": sea_record,
        "sea_ice_columns": int(frozen.sum()),
        "start_valid_time": None if getattr(earth_frame, "valid_time", None) is None
        else str(earth_frame.valid_time),
    }
    return WorldFrame(
        latitude=lat_w.copy(),
        longitude=world_map.longitude_deg.copy(),
        vertical_kind="pressure",
        vertical_values=np.asarray(earth_frame.vertical_values).copy(),
        fields=fields,
        mapping_sha256=getattr(earth_frame, "mapping_sha256", None),
        input_sha256=getattr(earth_frame, "input_sha256", None),
        source_cycle=getattr(earth_frame, "source_cycle", None),
        valid_time=getattr(earth_frame, "valid_time", None),
        snow_on_sea_ice=True,
        world_frame=record,
        mover_surface_pressure_pa=surface_pressure,
    )


__all__ = [
    "DRY_AIR_GAS_CONSTANT", "FRAME_SCHEMA", "LAPSE_K_PER_M", "LOAM_SOIL_CATEGORY",
    "MIN_ABOVE_GROUND_SHARE", "NOAH_ZBOT_M", "SEA_WATER_FREEZING_K",
    "WorldField", "WorldFrame", "WorldMap", "ZonalColumn",
    "aquaplanet_sea_temperature_k", "area_mean", "cell_area_weights",
    "deep_soil_plane", "earth_sea_by_latitude", "earth_zonal_mean_air",
    "invented_world_frame", "move_surface_pressure", "world_map_from_elevation",
    "sea_level_deep_soil_by_latitude", "soil_water_from_wetness",
]
