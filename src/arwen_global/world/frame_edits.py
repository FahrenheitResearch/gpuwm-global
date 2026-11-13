"""The terrain edit of a decoded analysis frame and the pressure mover.

A world that changes the terrain (``[world.terrain]``: a scale, a height
cap, optionally inside a tapered box) is applied to the DECODED analysis
frame, on the analysis's own grid, before the cold start reads a field
(plan section 4.3).  The cold start itself is not changed: it reads an
analysis whose terrain, surface pressure, below-ground air and land skin
are the world's, and does with them exactly what it does with Earth's.

What the edit does, in order:

1. **New terrain.**  ``target = min(scale * z, cap_m)`` (the cap only when
   one is set); inside a box the terrain moves the whole way to the
   target, and outside it the move tapers to nothing over ``taper_deg``
   with a cosine in latitude and in longitude (degrees, not kilometres).
2. **The air under removed terrain** (columns whose ground goes down):
   on every isobaric level below the old ground, temperature continues
   downward at :data:`LAPSE_RATE_K_M` from the lowest level that was above
   ground, and specific humidity and the wind are held at that level's
   value.  The rule is the world's, not the analysis centre's, so the same
   world starts the same way from GDAS (which holds relative humidity
   below ground) and ERA5 (which holds specific humidity); the owner chose
   to hold specific humidity, which adds no water Earth did not have.
3. **The pressure mover** (:func:`move_surface_pressure`): the surface
   pressure moves from the old ground to the new one by integrating the
   hydrostatic equation layer by layer through the column's own virtual
   temperature.  Between two isobaric levels the virtual temperature is
   taken as a constant lapse (``ln Tv`` linear in ``ln p``), which makes
   each layer's thickness ``R * log-mean(Tv) * dln p`` and the move exact
   for any atmosphere of constant lapse.  Below the deepest level the
   column continues at :data:`LAPSE_RATE_K_M`; above the top it is held.
   A column whose terrain does not change keeps its surface pressure bit
   for bit.  The mover works in both directions: lowering terrain (no
   mountains) and raising it (Earth's air laid onto an invented world's
   mountains, plan package 2.6).
4. **The amount of air** (``air_mass``): ``"conserve"`` scales every
   column's surface pressure by one number so the world holds the
   analysis's own amount of air (cell-area weighted on the frame's grid);
   ``"fill"`` keeps the moved pressures, which on lowered ground keeps the
   sea-level pressure map and adds air.
5. **Land skin and soil** move by :data:`LAPSE_RATE_K_M` per metre of
   height removed (warmer on lowered ground, colder on raised ground), the
   lapse the statics use for the deep soil (``statics.TMN_LAPSE_K_PER_M``).
   Land is the frame's own ``land_fraction`` at or above one half; water
   columns keep their skin, which the cold start holds as the sea
   temperature.  Snow is removed where the moved skin is above freezing on
   warmed land, and the record counts it for the world statement.

Every rule's sentence is in the record :func:`apply_terrain_edit` returns,
which the cold start carries into the provenance and the receipt.
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import GRAVITY_M_S2

#: The dry-air gas constant the cold start's hypsometric reduction uses
#: (``analysis_initial.DRY_AIR_GAS_CONSTANT``); a test holds them equal.
DRY_AIR_GAS_CONSTANT = 287.0
#: Virtual temperature factor, ``Tv = T (1 + 0.608 q)``, as the cold start.
VIRTUAL_FACTOR = 0.608
#: The world's lapse rate below removed ground, and the change of the land
#: skin and soil per metre of height moved: 6.5 K per km, the lapse the
#: statics use for the deep soil (``statics.TMN_LAPSE_K_PER_M``).
LAPSE_RATE_K_M = 0.0065
#: Snow on warmed land goes where the moved skin is above this.
FREEZING_K = 273.15
#: A column is land when the frame's land fraction is at least this.
LAND_THRESHOLD = 0.5
#: Columns moved per pass of the mover (a 0.25 degree frame has about a
#: million; a pass of this size holds a few hundred MB at 41 levels).
COLUMN_BATCH = 65536

#: The record's sentences, verbatim.
TERRAIN_RULE = (
    "target terrain = min(scale * terrain, cap_m) (cap only when set); "
    "inside the box the terrain moves the whole way to the target and "
    "outside it the move tapers to nothing over taper_deg by a cosine in "
    "latitude times a cosine in longitude, both in degrees"
)
BELOW_GROUND_RULE = (
    "on columns whose ground is lowered, every isobaric level below the old "
    "ground takes the temperature of the lowest level above ground continued "
    "downward at 6.5 K/km, and that level's specific humidity and wind"
)
MOVER_RULE = (
    "surface pressure moved from the old ground to the new by integrating "
    "the hydrostatic equation layer by layer through the column's virtual "
    "temperature, ln Tv linear in ln p between levels (each layer's "
    "thickness is R times the log-mean Tv times dln p), 6.5 K/km below the "
    "deepest level, held above the top; an unchanged column keeps its "
    "pressure bit for bit"
)
AIR_MASS_RULE = {
    "conserve": (
        "every column's surface pressure scaled by one number so the "
        "cell-area-weighted mean surface pressure (the amount of air) "
        "equals the analysis's"
    ),
    "fill": (
        "the moved surface pressure kept as it is: the sea-level pressure "
        "map is kept and lowered ground gains air"
    ),
}
LAND_RULE = (
    "on land (frame land fraction >= 0.5) the skin and every soil layer "
    "move by 6.5 K per km of height removed (colder where height is added); "
    "snow water and depth are removed where warmed land's moved skin is "
    "above 273.15 K; water columns keep their skin"
)


# -- the new terrain ------------------------------------------------------

def _cosine_taper(distance_deg, taper_deg):
    """1 at distance 0, 0 at and beyond ``taper_deg``, a half cosine
    between; a zero taper is a hard edge."""
    distance = np.asarray(distance_deg, dtype=np.float64)
    if taper_deg <= 0.0:
        return np.where(distance <= 0.0, 1.0, 0.0)
    s = np.clip(distance / taper_deg, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * s))


def box_weight(latitude_deg, longitude_deg, box):
    """The weight of the terrain edit at each frame point: 1 inside
    ``box``, a cosine taper to 0 over ``box.taper_deg`` outside it, the
    product of the latitude and longitude tapers.  ``box`` None is the whole
    globe.  Longitudes may be given in any convention; ``west_deg`` greater
    than ``east_deg`` is a box across the 180th meridian.  Returns a
    ``(nlat, nlon)`` array."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    if box is None:
        return np.ones((lat.size, lon.size))
    d_lat = np.maximum(np.maximum(box.south_deg - lat, lat - box.north_deg), 0.0)
    west, east = float(box.west_deg), float(box.east_deg)
    width = east - west if east > west else east - west + 360.0
    if width >= 360.0:
        d_lon = np.zeros_like(lon)
    else:
        offset = np.mod(lon - west, 360.0)
        d_lon = np.where(offset <= width, 0.0,
                         np.minimum(offset - width, 360.0 - offset))
    taper = float(box.taper_deg)
    return _cosine_taper(d_lat, taper)[:, None] * _cosine_taper(d_lon, taper)[None, :]


def edited_terrain(terrain_m, latitude_deg, longitude_deg, terrain):
    """The world's terrain on the frame's grid from Earth's ``terrain_m``
    ``(nlat, nlon)`` and the world's :class:`~arwen_global.world.Terrain`.

    An Earth terrain part returns the input object itself."""
    if terrain.is_earth:
        return terrain_m
    z = np.asarray(terrain_m, dtype=np.float64)
    target = float(terrain.scale) * z
    if terrain.cap_m is not None:
        target = np.minimum(target, float(terrain.cap_m))
    if terrain.box is None:
        return target
    weight = box_weight(latitude_deg, longitude_deg, terrain.box)
    return np.where(weight >= 1.0, target,
                    np.where(weight <= 0.0, z, z + weight * (target - z)))


# -- the air under removed ground ------------------------------------------

def _lapse_exponent(lapse_k_m=LAPSE_RATE_K_M):
    """``c`` in ``T(p) = T0 (p / p0) ** c`` for a constant lapse."""
    return DRY_AIR_GAS_CONSTANT * lapse_k_m / GRAVITY_M_S2


def below_ground_air(levels_pa, temperature, humidity, u, v, ps_pa):
    """The world's rule for the air under the ground of each column.

    ``temperature``, ``humidity``, ``u`` and ``v`` are ``(nlev, ncol)`` on
    ascending ``levels_pa`` (top first); ``ps_pa`` ``(ncol,)`` is the
    ground.  Every level with ``p > ps`` takes the temperature of the
    lowest level with ``p <= ps`` continued downward at
    :data:`LAPSE_RATE_K_M` (``T (p / p_k) ** c``), and that level's
    humidity and wind.  Returns the four arrays (new arrays; the inputs are
    not written; a wind given as None comes back None) and the boolean
    ``(nlev, ncol)`` mask of the rewritten levels.  Applying the rule twice
    gives the same air to rounding: a level the rule wrote continues the
    same lapse from the same anchor."""
    levels = np.asarray(levels_pa, dtype=np.float64)
    ps = np.asarray(ps_pa, dtype=np.float64)
    below = levels[:, None] > ps[None, :]
    # The lowest level at or above ground; a column whose ground lies above
    # the top level would have no air and is refused.
    above_count = np.count_nonzero(~below, axis=0)
    if np.any(above_count == 0):
        raise ValueError(
            "a column's ground lies above the analysis's top level; the "
            "below-ground rule has no level above ground to continue from")
    anchor = above_count - 1
    columns = np.arange(ps.size)
    c = _lapse_exponent()
    t_anchor = np.asarray(temperature, dtype=np.float64)[anchor, columns]
    p_anchor = levels[anchor]
    t_rule = t_anchor[None, :] * (levels[:, None] / p_anchor[None, :]) ** c
    out = []
    for name, values in (("temperature", temperature), ("humidity", humidity),
                         ("u", u), ("v", v)):
        if values is None:
            out.append(None)
            continue
        values = np.asarray(values, dtype=np.float64)
        if name == "temperature":
            rule = t_rule
        else:
            rule = np.broadcast_to(values[anchor, columns][None, :], values.shape)
        out.append(np.where(below, rule, values))
    return (*out, below)


# -- the pressure mover ---------------------------------------------------

def _exprel(a):
    """``expm1(a) / a``, 1 at 0."""
    a = np.asarray(a, dtype=np.float64)
    small = np.abs(a) < 1e-8
    safe = np.where(small, 1.0, a)
    return np.where(small, 1.0 + 0.5 * a, np.expm1(safe) / safe)


def _log1prel(a):
    """``log1p(a) / a``, 1 at 0."""
    a = np.asarray(a, dtype=np.float64)
    small = np.abs(a) < 1e-8
    safe = np.where(small, 1.0, a)
    return np.where(small, 1.0 - 0.5 * a, np.log1p(safe) / safe)


def _move_batch(ln_p, temperature, humidity, ps_old, dz):
    """The mover on one batch of columns that all move (``dz != 0``)."""
    nlev, ncol = temperature.shape
    q = np.clip(humidity, 0.0, None)
    ln_tv = np.log(temperature * (1.0 + VIRTUAL_FACTOR * q))
    dx = np.diff(ln_p)                                      # (nlev-1,)
    slope = np.diff(ln_tv, axis=0) / dx[:, None]            # (nlev-1, ncol)
    below_slope = _lapse_exponent()
    # F(x) = integral of Tv dln p from the top level; F at every level.
    tv = np.exp(ln_tv)
    layer = tv[:-1] * dx[:, None] * _exprel(slope * dx[:, None])
    f_level = np.concatenate([np.zeros((1, ncol)), np.cumsum(layer, axis=0)])

    def segment_values(index):
        """Anchor level, its ln p, its Tv, the slope and F there, for a
        segment index in -1 (above the top) .. nlev-1 (below the deepest)."""
        cols = np.arange(ncol)
        anchor = np.clip(index, 0, nlev - 1)
        slope_ext = np.concatenate(
            [slope, np.full((1, ncol), below_slope)], axis=0)
        beta = np.where(index < 0, 0.0, slope_ext[anchor, cols])
        return ln_p[anchor], tv[anchor, cols], beta, f_level[anchor, cols]

    def integral_from_anchor(x, index):
        x_a, tv_a, beta, f_a = segment_values(index)
        d = x - x_a
        return f_a + tv_a * d * _exprel(beta * d)

    x_old = np.log(ps_old)
    seg_old = np.count_nonzero(ln_p[:, None] <= x_old[None, :], axis=0) - 1
    f_old = integral_from_anchor(x_old, seg_old)
    # Phi(new) - Phi(old) = g dz = -R * (F(new) - F(old)).
    f_new = f_old - GRAVITY_M_S2 * dz / DRY_AIR_GAS_CONSTANT
    seg_new = np.count_nonzero(f_level <= f_new[None, :], axis=0) - 1
    x_a, tv_a, beta, f_a = segment_values(seg_new)
    r = (f_new - f_a) / tv_a
    x_new = x_a + r * _log1prel(beta * r)
    return np.exp(x_new)


def move_surface_pressure(levels_pa, temperature, humidity, ps_pa,
                          terrain_old_m, terrain_new_m, *, batch=COLUMN_BATCH):
    """Move each column's surface pressure from ``terrain_old_m`` to
    ``terrain_new_m`` hydrostatically through its own virtual temperature.

    ``temperature`` and ``humidity`` are ``(nlev, ...)`` isobaric stacks
    on ascending ``levels_pa`` (top first); ``ps_pa``, ``terrain_old_m``
    and ``terrain_new_m`` have the trailing shape.  Between levels the
    virtual temperature is a constant lapse (``ln Tv`` linear in ``ln p``),
    below the deepest level it continues at :data:`LAPSE_RATE_K_M`, above
    the top it is held; within each layer the move is solved in closed
    form, so a constant-lapse atmosphere moves exactly.  A column whose
    terrain is unchanged returns its input value bit for bit.

    The values below the old ground are whatever the stacks hold there:
    :func:`apply_terrain_edit` has already applied the world's below-ground
    rule (:func:`below_ground_air`) before it calls this.
    """
    levels = np.asarray(levels_pa, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 2 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("levels_pa must ascend top to bottom")
    ps = np.asarray(ps_pa)
    shape = ps.shape
    t = np.asarray(temperature, dtype=np.float64).reshape(levels.size, -1)
    q = np.asarray(humidity, dtype=np.float64).reshape(levels.size, -1)
    z_old = np.asarray(terrain_old_m, dtype=np.float64).reshape(-1)
    z_new = np.asarray(terrain_new_m, dtype=np.float64).reshape(-1)
    ps_flat = ps.reshape(-1)
    out = ps_flat.copy()
    moved = np.flatnonzero(z_new != z_old)
    ln_p = np.log(levels)
    for start in range(0, moved.size, batch):
        idx = moved[start:start + batch]
        out[idx] = _move_batch(
            ln_p, t[:, idx], q[:, idx],
            ps_flat[idx].astype(np.float64), z_new[idx] - z_old[idx])
    return out.reshape(shape)


# -- the amount of air ------------------------------------------------------

def cell_area_weights(latitude_deg, nlon):
    """Relative cell areas ``(nlat, nlon)`` of a latitude-longitude grid
    whose rows are ``latitude_deg`` (either order): each row spans from the
    midpoints to its neighbours, the end rows to the poles."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    order = np.argsort(lat)
    sorted_lat = lat[order]
    edges = np.concatenate(
        [[-90.0], 0.5 * (sorted_lat[1:] + sorted_lat[:-1]), [90.0]])
    band = np.diff(np.sin(np.deg2rad(edges)))
    weights = np.empty_like(lat)
    weights[order] = band
    return np.repeat(weights[:, None], nlon, axis=1)


def air_mass_factor(ps_moved, ps_analysis, latitude_deg):
    """The one number ``conserve`` multiplies every surface pressure by:
    the analysis's area-weighted mean surface pressure over the moved one."""
    ps_moved = np.asarray(ps_moved, dtype=np.float64)
    weights = cell_area_weights(latitude_deg, ps_moved.shape[-1])
    return float(np.sum(weights * np.asarray(ps_analysis, dtype=np.float64))
                 / np.sum(weights * ps_moved))


# -- the frame --------------------------------------------------------------

class _WorldField:
    """One edited plane (or stack) of the frame, keeping the source label
    the cold start reads off a composite frame's soil group."""

    __slots__ = ("values", "source")

    def __init__(self, values, source):
        self.values = values
        self.source = source


class WorldFrame:
    """A decoded analysis frame with the world's fields in place of some of
    Earth's.  Everything not edited (grid, levels, times, digests, any
    attribute a frame type adds) is read through from the frame it wraps;
    ``world_terrain`` is the record of the edit."""

    def __init__(self, base, fields: dict, record: dict):
        self._base = base
        self.fields = fields
        self.world_terrain = record

    def __getattr__(self, name):
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)


def _replace(fields, name, values):
    old = fields[name]
    fields[name] = _WorldField(values, getattr(old, "source", "primary"))


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.min(values)), "max": float(np.max(values))}


def apply_terrain_edit(frame, world, levels_pa):
    """Apply ``world``'s terrain to a decoded analysis ``frame``.

    ``levels_pa`` are the frame's isobaric levels in Pa, ascending (the cold
    start's ``analysis_levels_pa``).  Returns ``(frame, record)``: a
    :class:`WorldFrame` with the edited terrain, surface pressure, isobaric
    temperature, humidity and wind below lowered ground, land skin and soil
    temperature and snow, and the JSON-safe record of what moved.  A world
    whose terrain is Earth's returns the frame object itself and ``None``:
    no world code runs on Earth.
    """
    terrain = world.terrain
    if terrain.is_earth:
        return frame, None
    levels = np.asarray(levels_pa, dtype=np.float64)
    fields = frame.fields
    lat = np.asarray(frame.latitude, dtype=np.float64)
    lon = np.asarray(frame.longitude, dtype=np.float64)
    z_old = np.asarray(fields["terrain_height"].values, dtype=np.float64)
    ps_old = np.asarray(fields["surface_pressure"].values, dtype=np.float64)
    z_new = edited_terrain(z_old, lat, lon, terrain)
    dz = z_new - z_old
    moved = dz != 0.0
    lowered = dz < 0.0
    nlev = levels.size
    spatial = ps_old.shape
    ncolumns = int(np.prod(spatial))

    stacks = {}
    for name in ("air_temperature", "specific_humidity",
                 "eastward_wind", "northward_wind"):
        stacks[name] = np.array(fields[name].values, dtype=np.float64,
                                copy=True).reshape(nlev, ncolumns)
    flat_lowered = np.flatnonzero(lowered.reshape(-1))
    flat_moved = np.flatnonzero(moved.reshape(-1))
    rewritten_levels = 0
    ps_flat = ps_old.reshape(-1)
    for start in range(0, flat_lowered.size, COLUMN_BATCH):
        idx = flat_lowered[start:start + COLUMN_BATCH]
        t, q, u, v, below = below_ground_air(
            levels,
            stacks["air_temperature"][:, idx],
            stacks["specific_humidity"][:, idx],
            stacks["eastward_wind"][:, idx],
            stacks["northward_wind"][:, idx],
            ps_flat[idx],
        )
        stacks["air_temperature"][:, idx] = t
        stacks["specific_humidity"][:, idx] = q
        stacks["eastward_wind"][:, idx] = u
        stacks["northward_wind"][:, idx] = v
        rewritten_levels += int(np.count_nonzero(below))
    # The mover integrates every moved column through the world's rule
    # under its old ground: a raised column's stacks keep the analysis's
    # values there (that air stays under ground and the cold start never
    # reads it), but the short stretch between its lowest level above
    # ground and the ground itself is integrated through the rule, so the
    # move never depends on which centre extrapolated the analysis.
    raised = np.flatnonzero((dz > 0.0).reshape(-1))
    dz_flat = dz.reshape(-1)
    ln_p = np.log(levels)
    ps_moved = ps_flat.copy()
    for start in range(0, flat_moved.size, COLUMN_BATCH):
        idx = flat_moved[start:start + COLUMN_BATCH]
        t, q, _, _, _ = below_ground_air(
            levels, stacks["air_temperature"][:, idx],
            stacks["specific_humidity"][:, idx], None, None, ps_flat[idx])
        ps_moved[idx] = _move_batch(ln_p, t, q, ps_flat[idx], dz_flat[idx])
    ps_moved = ps_moved.reshape(spatial)
    if terrain.air_mass == "conserve":
        factor = air_mass_factor(ps_moved, ps_old, lat)
        ps_new = ps_moved * factor
    else:
        factor = 1.0
        ps_new = ps_moved

    edited = dict(fields)
    _replace(edited, "terrain_height", z_new)
    _replace(edited, "surface_pressure", ps_new)
    for name, stack in stacks.items():
        _replace(edited, name, stack.reshape((nlev,) + spatial))

    # Land skin, soil and snow.
    land = np.asarray(fields["land_fraction"].values, dtype=np.float64) >= LAND_THRESHOLD
    land_moved = land & moved
    warming = np.where(land_moved, -LAPSE_RATE_K_M * dz, 0.0)
    skin_old = np.asarray(fields["skin_temperature"].values, dtype=np.float64)
    skin_new = skin_old + warming
    _replace(edited, "skin_temperature", skin_new)
    soil_old = np.asarray(fields["soil_temperature"].values, dtype=np.float64)
    _replace(edited, "soil_temperature", soil_old + warming[None])
    swe = np.asarray(fields["snow_water_equivalent"].values, dtype=np.float64)
    melt = land_moved & (warming > 0.0) & (skin_new > FREEZING_K) & (swe > 0.0)
    weights = cell_area_weights(lat, lon.size)
    snow_removed_columns = int(np.count_nonzero(melt))
    snow_removed_mean = float(
        np.sum(np.where(melt, swe * weights, 0.0)) / np.sum(weights))
    if snow_removed_columns:
        _replace(edited, "snow_water_equivalent", np.where(melt, 0.0, swe))
        depth = np.asarray(fields["snow_depth"].values, dtype=np.float64)
        _replace(edited, "snow_depth", np.where(melt, 0.0, depth))

    area = float(np.sum(weights))
    record = {
        "settings": {
            "scale": float(terrain.scale),
            "cap_m": None if terrain.cap_m is None else float(terrain.cap_m),
            "box": None if terrain.box is None else {
                "south_deg": terrain.box.south_deg,
                "north_deg": terrain.box.north_deg,
                "west_deg": terrain.box.west_deg,
                "east_deg": terrain.box.east_deg,
                "taper_deg": terrain.box.taper_deg,
            },
            "air_mass": terrain.air_mass,
        },
        "rules": {
            "terrain": TERRAIN_RULE,
            "below_ground": BELOW_GROUND_RULE,
            "mover": MOVER_RULE,
            "air_mass": AIR_MASS_RULE[terrain.air_mass],
            "land": LAND_RULE,
        },
        "lapse_rate_k_per_m": LAPSE_RATE_K_M,
        "columns": ncolumns,
        "columns_moved": int(flat_moved.size),
        "columns_lowered": int(flat_lowered.size),
        "columns_raised": int(raised.size),
        "below_ground_levels_rewritten": rewritten_levels,
        "terrain_change_m": _stats(dz[moved]),
        "terrain_max_m": {"before": float(np.max(z_old)), "after": float(np.max(z_new))},
        "surface_pressure_pa": {
            "before": _stats(ps_old),
            "moved": _stats(ps_moved),
            "after": _stats(ps_new),
        },
        "surface_pressure_change_pa": _stats(ps_new - ps_old),
        "air_mass_factor": factor,
        "mean_surface_pressure_pa": {
            "before": float(np.sum(weights * ps_old) / area),
            "after": float(np.sum(weights * ps_new) / area),
        },
        "land_temperature_change_k": _stats(warming[land_moved]),
        "snow_removed_columns": snow_removed_columns,
        "snow_removed_global_mean_kg_m2": snow_removed_mean,
    }
    return WorldFrame(frame, edited, record), record


def edit_frame(frame, world, *, levels_pa, grid=None):
    """The cold start's hook (package 1.2's contract,
    ``analysis_initial.WORLD_FRAME_EDITOR``): called only when
    ``world.changes_terrain``, on the decoded frame after the ERA5
    completion and the fill, with the levels already in Pa.  Returns
    ``(frame, record)`` as :func:`apply_terrain_edit`.  The edit acts on
    the analysis's own grid; ``grid`` (the model's) is accepted for the
    contract and not used, because the cold start's own spectral smoothing
    and hypsometric move then take the edited terrain onto the model grid
    exactly as they take Earth's."""
    del grid
    return apply_terrain_edit(frame, world, levels_pa)


def closed_form_constant_lapse(ps_pa, surface_temperature_k, dz_m,
                               lapse_k_m=LAPSE_RATE_K_M):
    """Surface pressure after the ground moves ``dz_m`` (negative: lowered)
    under a dry atmosphere of constant lapse whose surface air is
    ``surface_temperature_k`` at ``ps_pa``: ``ps (T_new / T) ** (g / (R
    lapse))`` with ``T_new = T - lapse dz``.  The reference the mover is
    tested against."""
    t_new = surface_temperature_k - lapse_k_m * dz_m
    return ps_pa * (t_new / surface_temperature_k) ** (
        GRAVITY_M_S2 / (DRY_AIR_GAS_CONSTANT * lapse_k_m))


__all__ = [
    "AIR_MASS_RULE", "BELOW_GROUND_RULE", "COLUMN_BATCH", "DRY_AIR_GAS_CONSTANT",
    "FREEZING_K", "LAND_RULE", "LAND_THRESHOLD", "LAPSE_RATE_K_M", "MOVER_RULE",
    "TERRAIN_RULE", "VIRTUAL_FACTOR", "WorldFrame", "air_mass_factor",
    "apply_terrain_edit", "below_ground_air", "box_weight", "cell_area_weights",
    "closed_form_constant_lapse", "edit_frame", "edited_terrain",
    "move_surface_pressure",
]
