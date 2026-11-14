"""The preflight: does this world's start fit the model, before any card time?

A world is checked on the CPU, from the analysis frame it will start from,
before the cold start reads a field (worlds plan, sections 4.8 and 4.11,
package 1.6).  Two things come out:

- **refusals**, each by name and with its text, for a start the model is
  known not to survive: terrain so tall that the predicted lowest surface
  pressure falls under the floor the model's vertical grid and dynamics
  were built for (``terrain_too_tall``), and a predicted surface pressure
  over the radiation's pressure ceiling (``pressure_over_ceiling``), and a
  land skin the terrain edit warms (6.5 K per km of height removed, plan
  4.3 rule 5) past the radiation's 355 K range (``start_skin_over_limit``):
  each would stop the run in its first radiation call;
- **the four margins** of the world statement (``world.statement``): the
  hottest skin expected against the radiation's 355 K limit, the coldest
  air against the dynamics' 140 K limit, the days until the first ocean
  column empties its 500 kg m-2 water store against the run's length, and
  the bottom layer's thickness at the lowest surface pressure.

The prediction is the world's own rule for moving surface pressure onto
new terrain (plan 4.3), in closed form, on the frame's own grid:

- below the old surface (terrain lowered) the air is the world's: its
  temperature continues downward at 6.5 K per km from the lowest analysis
  level above ground and its specific humidity is held at that level's
  value, so ``ps' = p0 * (A + Gv (z - z') / Tv0) ** (g / (R Gv))`` with
  ``A = (ps / p0) ** (R Gv / g)``, ``Gv`` the lapse of virtual temperature
  and ``p0, Tv0`` the lowest level above ground;
- above it (terrain raised) the analysis's own air is integrated upward,
  layer by layer, by the hypsometric equation at each layer's mean
  virtual temperature;
- ``air_mass = "conserve"`` then scales every column by the one number that
  keeps the global mean surface pressure (Earth's amount of air), and
  ``"fill"`` leaves the sea-level pressure map as it is.

The frame's grid (0.25 degrees for GDAS) is sharper than the model's
T255 terrain, so the prediction is conservative in both directions: its
lowest pressure under a raised peak is lower, and its highest pressure
over a removed plateau is higher, than the model's own start.  The
pressure mover itself (package 1.1, ``world/frame_edits.py``) is the
arithmetic of record: when its output is passed as
``edited_surface_pressure_pa`` the refusals are read off it instead.

Every limit and every measured Earth figure the margins use is a row of
:data:`LIMITS` or :data:`MEASURED_EARTH` with where it came from, so a new
measurement changes a row and not code (the rows are read from
``data/world-limits.json``, worlds package 3.5).

Nothing here runs for Earth: :func:`preflight` of an Earth world returns an
empty report without reading the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2
from .limits import Limit, Measured
from .limits import limits as _limit_rows, measured_earth as _measured_rows
from .spec import EARTH_OBLIQUITY_DEG, WorldSpec

#: The world's lapse for air under removed terrain and for warming the land
#: skin where terrain was removed (plan 4.3, rules 3 and 5): the standard
#: atmosphere's, which is also the lapse the statics use for deep soil and
#: the one GDAS continues below ground (worlds package 0.2: exactly
#: 6.500 K/km on 100 % of the underground columns of GDAS 2026-11-10 18Z).
LAPSE_K_PER_M = 0.0065
#: Virtual-temperature factor, as ``analysis_initial.surface_virtual_temperature``.
VIRTUAL_FACTOR = 0.608
#: A land column for the skin margin: land fraction at least this.
LAND_FRACTION_LAND = 0.5


#: The model's limits and Earth's own figures measured with this model at
#: T255, by key.  Each is a row of ``arwen_global/data/world-limits.json``
#: (worlds package 3.5) with where it came from, so a new measurement
#: changes a row of that file and not code.
LIMITS: dict[str, Limit] = dict(_limit_rows())
MEASURED_EARTH: dict[str, Measured] = dict(_measured_rows())


# -- refusals --------------------------------------------------------------


#: Every refusal the preflight can raise, by name, with the sentence that
#: starts its text.
REFUSALS: dict[str, str] = {
    "terrain_too_tall": "the terrain is too tall for the model",
    "pressure_over_ceiling": (
        "the predicted surface pressure is over the radiation's pressure ceiling"),
    "start_skin_over_limit": (
        "the start's land skin, warmed where terrain was removed, is over the "
        "radiation's 355 K limit"),
}


@dataclass(frozen=True)
class Refusal:
    key: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "text": self.text}


class WorldRefused(ValueError):
    """A world the preflight refuses.  ``refusals`` lists every reason."""

    def __init__(self, label: str, refusals: tuple[Refusal, ...]):
        self.label = label
        self.refusals = tuple(refusals)
        lines = [f"world {label!r} is refused before any card time:"]
        lines.extend(f"- {r.key}: {r.text}" for r in self.refusals)
        super().__init__("\n".join(lines))


# -- margins ---------------------------------------------------------------

#: The four margins, in the order the statement prints them.
MARGIN_KEYS = ("hottest_skin", "coldest_air", "water_store", "bottom_layer")

INSIDE, CLOSE, OVER, UNKNOWN = "inside", "close", "over", "unknown"


@dataclass(frozen=True)
class Margin:
    """One margin of the world statement.

    ``value`` is this world's expected figure, ``limit`` what it is held
    against, ``margin`` the room left (positive is inside, in ``unit``),
    and ``status`` one of inside / close / over / unknown.  ``rule`` says
    how ``value`` was reached, from what."""

    key: str
    title: str
    value: float | None
    limit: float | None
    unit: str
    margin: float | None
    status: str
    rule: str

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key, "title": self.title,
            "value": _rounded(self.value), "limit": self.limit, "unit": self.unit,
            "margin": _rounded(self.margin), "status": self.status, "rule": self.rule,
        }


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _status(margin: float, caution: float) -> str:
    if margin < 0.0:
        return OVER
    return CLOSE if margin < caution else INSIDE


# -- the frame -------------------------------------------------------------


@dataclass(frozen=True)
class FrameView:
    """The parts of an analysis frame the preflight reads, on the frame's
    own regular latitude-longitude grid.

    ``levels_pa`` ascend from the top (smallest pressure) to the bottom;
    ``temperature`` and ``specific_humidity`` are ``(level, lat, lon)``;
    the rest are ``(lat, lon)``.  ``skin_temperature`` and
    ``land_fraction`` are optional (the skin margin then reads the
    measured Earth figure alone)."""

    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    levels_pa: np.ndarray
    temperature_k: np.ndarray
    specific_humidity: np.ndarray
    surface_pressure_pa: np.ndarray
    terrain_m: np.ndarray
    skin_temperature_k: np.ndarray | None = None
    land_fraction: np.ndarray | None = None

    def __post_init__(self):
        levels = np.asarray(self.levels_pa, dtype=np.float64)
        if levels.ndim != 1 or levels.size < 2 or np.any(np.diff(levels) <= 0.0):
            raise ValueError("frame levels must ascend from the top in Pa")
        shape = (levels.size, np.size(self.latitude_deg), np.size(self.longitude_deg))
        for name in ("temperature_k", "specific_humidity"):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"frame {name} has shape {np.shape(getattr(self, name))}, "
                                 f"expected {shape}")
        for name in ("surface_pressure_pa", "terrain_m", "skin_temperature_k",
                     "land_fraction"):
            value = getattr(self, name)
            if value is not None and np.shape(value) != shape[1:]:
                raise ValueError(f"frame {name} has shape {np.shape(value)}, "
                                 f"expected {shape[1:]}")

    @classmethod
    def from_frame(cls, frame, levels_pa) -> "FrameView":
        """From a decoded analysis frame (``frame.fields[name].values``,
        ``frame.latitude``, ``frame.longitude``) and its levels in Pa, as
        ``analysis_initial.analysis_levels_pa`` returns them."""

        def values(name, required=True):
            if name not in frame.fields:
                if required:
                    raise ValueError(f"analysis frame lacks {name}")
                return None
            return np.asarray(frame.fields[name].values, dtype=np.float64)

        return cls(
            latitude_deg=np.asarray(frame.latitude, dtype=np.float64),
            longitude_deg=np.asarray(frame.longitude, dtype=np.float64),
            levels_pa=np.asarray(levels_pa, dtype=np.float64),
            temperature_k=values("air_temperature"),
            specific_humidity=np.clip(values("specific_humidity"), 0.0, None),
            surface_pressure_pa=values("surface_pressure"),
            terrain_m=values("terrain_height"),
            skin_temperature_k=values("skin_temperature", required=False),
            land_fraction=values("land_fraction", required=False),
        )

    @property
    def area_weights(self) -> np.ndarray:
        lat = np.radians(np.asarray(self.latitude_deg, dtype=np.float64))
        return np.broadcast_to(np.cos(lat)[:, None],
                               (lat.size, np.size(self.longitude_deg)))

    def where(self, flat_index: int) -> tuple[float, float]:
        j, i = np.unravel_index(int(flat_index), np.shape(self.surface_pressure_pa))
        lon = float(np.asarray(self.longitude_deg)[i])
        lon = lon - 360.0 if lon > 180.0 else lon
        return float(np.asarray(self.latitude_deg)[j]), lon


def _columns(count: int) -> str:
    return f"{count:,} column" + ("" if count == 1 else "s")


def _place(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0.0 else "S"
    ew = "E" if lon >= 0.0 else "W"
    return f"{abs(lat):.2f}{ns} {abs(lon):.2f}{ew}"


# -- the terrain rule ------------------------------------------------------


def box_weight(world: WorldSpec, latitude_deg, longitude_deg) -> np.ndarray | float:
    """How much of the terrain edit acts at each point: 1 inside the box,
    falling linearly to 0 over ``taper_deg`` outside it, 1 everywhere when
    the world has no box.  Distance outside the box is the larger of the
    latitude and the (wrapped) longitude distance, in degrees."""
    box = world.terrain.box
    if box is None:
        return 1.0
    lat = np.asarray(latitude_deg, dtype=np.float64)[:, None]
    lon = np.mod(np.asarray(longitude_deg, dtype=np.float64), 360.0)[None, :]
    west, east = box.west_deg % 360.0, box.east_deg % 360.0
    width = (east - west) % 360.0
    offset = (lon - west) % 360.0
    inside_lon = offset <= width
    lon_out = np.where(inside_lon, 0.0,
                       np.minimum((west - lon) % 360.0, (lon - east) % 360.0))
    lat_out = np.maximum(np.maximum(box.south_deg - lat, lat - box.north_deg), 0.0)
    distance = np.maximum(lon_out, lat_out)
    if box.taper_deg <= 0.0:
        return np.where(distance <= 0.0, 1.0, 0.0)
    return np.clip(1.0 - distance / box.taper_deg, 0.0, 1.0)


def edited_terrain(world: WorldSpec, terrain_m, latitude_deg, longitude_deg) -> np.ndarray:
    """The world's terrain on the frame's grid by the scale, cap and box of
    ``[world.terrain]``: ``z' = z + w (min(scale z, cap) - z)`` with ``w``
    the box weight.  Earth's terrain comes back as the same array.

    This is the preflight's reading of the rule; the terrain edit of record
    is package 1.1's (``world/frame_edits.py``), and its output can be
    passed to :func:`preflight` as ``edited_terrain_m``."""
    z = np.asarray(terrain_m, dtype=np.float64)
    if world.terrain.is_earth:
        return z
    target = z * world.terrain.scale
    if world.terrain.cap_m is not None:
        target = np.minimum(target, world.terrain.cap_m)
    weight = box_weight(world, latitude_deg, longitude_deg)
    if np.isscalar(weight) and weight == 1.0:
        return target
    return z + weight * (target - z)


# -- the pressure prediction -----------------------------------------------


def predict_surface_pressure(world: WorldSpec, view: FrameView,
                             new_terrain_m: np.ndarray) -> np.ndarray:
    """The surface pressure the world's rule puts on ``new_terrain_m``
    (see the module docstring), with the air-mass rule applied.  A column
    whose terrain did not move keeps its pressure exactly."""
    ps = np.asarray(view.surface_pressure_pa, dtype=np.float64).reshape(-1)
    z0 = np.asarray(view.terrain_m, dtype=np.float64).reshape(-1)
    z1 = np.asarray(new_terrain_m, dtype=np.float64).reshape(-1)
    if z1.shape != z0.shape:
        raise ValueError(f"edited terrain has {z1.size} columns, the frame {z0.size}")
    levels = np.asarray(view.levels_pa, dtype=np.float64)
    nlev = levels.size
    temperature = np.asarray(view.temperature_k, dtype=np.float64).reshape(nlev, -1)
    # Clipped at zero where read, as the cold start reads it.
    humidity = np.asarray(view.specific_humidity, dtype=np.float64).reshape(nlev, -1)

    out = ps.copy()
    moved = np.flatnonzero(z1 != z0)
    if moved.size:
        g, r = GRAVITY_M_S2, DRY_AIR_GAS_CONSTANT
        ps_m, z0_m, z1_m = ps[moved], z0[moved], z1[moved]
        # The lowest analysis level above ground in each column.
        k0 = np.sum(levels[:, None] < ps_m[None, :], axis=0) - 1
        if np.any(k0 < 0):
            raise ValueError("a column's surface lies above the frame's top level")
        p0 = levels[k0]
        q0 = np.clip(humidity[k0, moved], 0.0, None)
        tv0 = temperature[k0, moved] * (1.0 + VIRTUAL_FACTOR * q0)
        gv = LAPSE_K_PER_M * (1.0 + VIRTUAL_FACTOR * q0)
        n = g / (r * gv)
        base = (ps_m / p0) ** (1.0 / n)
        # Height of the level k0 above the old surface under the same lapse.
        dz_k0 = tv0 / gv * (base - 1.0)
        bottom = z1_m - z0_m <= dz_k0
        new = np.empty_like(ps_m)
        new[bottom] = p0[bottom] * (base[bottom] + gv[bottom] * (z0_m[bottom] - z1_m[bottom])
                                    / tv0[bottom]) ** n[bottom]
        up = np.flatnonzero(~bottom)
        if up.size:
            new[up] = _integrate_upward(
                levels, temperature[:, moved[up]], humidity[:, moved[up]], k0[up],
                z0_m[up] + dz_k0[up], z1_m[up])
        out[moved] = new
    if world.terrain.air_mass == "conserve" and moved.size:
        weights = view.area_weights.reshape(-1)
        out = out * (np.sum(weights * ps) / np.sum(weights * out))
    return out.reshape(np.shape(view.surface_pressure_pa))


def _integrate_upward(levels, temperature, humidity, k0, z_k0, z_target):
    """ln p at ``z_target`` by the hypsometric equation upward from the
    level ``k0`` (at height ``z_k0``) through the analysis's own levels."""
    g, r = GRAVITY_M_S2, DRY_AIR_GAS_CONSTANT
    ncol = z_target.size
    cols = np.arange(ncol)
    tv = temperature * (1.0 + VIRTUAL_FACTOR * np.clip(humidity, 0.0, None))
    ln_p = np.log(levels)
    result = np.full(ncol, np.nan)
    z_low = z_k0.copy()
    k = k0.copy()
    for _ in range(levels.size):
        live = np.isnan(result) & (k > 0)
        if not live.any():
            break
        idx = cols[live]
        kk = k[idx]
        tv_mean = 0.5 * (tv[kk, idx] + tv[kk - 1, idx])
        dz = r * tv_mean / g * (ln_p[kk] - ln_p[kk - 1])
        z_high = z_low[idx] + dz
        hit = z_target[idx] <= z_high
        found = idx[hit]
        result[found] = np.exp(ln_p[k[found]] - g * (z_target[found] - z_low[found])
                               / (r * tv_mean[hit]))
        z_low[idx] = z_high
        k[idx] = kk - 1
    if np.isnan(result).any():
        raise ValueError("an edited terrain height lies above the frame's top level")
    return result


# -- the report ------------------------------------------------------------


@dataclass(frozen=True)
class PreflightReport:
    """What the preflight found for one world.  ``prediction`` holds the
    frame figures (None without a frame); ``margins`` the four margins;
    ``refusals`` every refusal."""

    world: WorldSpec
    prediction: dict | None = None
    margins: tuple[Margin, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def refused(self) -> bool:
        return bool(self.refusals)

    def raise_if_refused(self) -> "PreflightReport":
        if self.refusals:
            raise WorldRefused(self.world.label, self.refusals)
        return self

    def as_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "margins": [m.as_dict() for m in self.margins],
            "refusals": [r.as_dict() for r in self.refusals],
            "notes": list(self.notes),
        }


def preflight(world: WorldSpec, *, frame=None, levels_pa=None,
              view: FrameView | None = None, vertical=None,
              duration_s: float | None = None,
              edited_terrain_m=None,
              edited_surface_pressure_pa=None) -> PreflightReport:
    """Check ``world`` before any card time.

    ``frame`` with ``levels_pa`` (or a ready ``view``) is the analysis the
    run starts from; without one only the frame-free margins are given.
    ``vertical`` is the run's ``HybridCoordinate`` (``cfg.vertical``),
    ``duration_s`` the run's length.  ``edited_terrain_m`` and
    ``edited_surface_pressure_pa`` are the frame edit's own output on the
    frame grid (package 1.1, or an invented world's terrain); given, they
    replace the preflight's own reading of the rule.

    Returns a :class:`PreflightReport`; call ``raise_if_refused()`` to stop
    a run.  An Earth world returns an empty report and reads nothing."""
    if world.is_earth:
        return PreflightReport(world=world)
    if view is None and frame is not None:
        if levels_pa is None:
            raise ValueError("preflight of a frame needs its levels in Pa")
        view = FrameView.from_frame(frame, levels_pa)

    notes: list[str] = []
    refusals: list[Refusal] = []
    prediction = None
    if view is not None:
        prediction, frame_refusals, frame_notes = _frame_checks(
            world, view, vertical, edited_terrain_m, edited_surface_pressure_pa)
        refusals.extend(frame_refusals)
        notes.extend(frame_notes)
    else:
        if world.changes_terrain or world.is_invented:
            notes.append(
                "the terrain moves, and no analysis frame was read: the "
                "surface-pressure refusals run on the frame at the start of "
                "the run, before the cold start")
    margins = (
        _hottest_skin(world, prediction),
        _coldest_air(world),
        _water_store(world, duration_s),
        _bottom_layer(world, prediction, vertical),
    )
    return PreflightReport(world=world, prediction=prediction, margins=margins,
                           refusals=tuple(refusals), notes=tuple(notes))


def _frame_checks(world, view, vertical, edited_terrain_m, edited_ps):
    refusals: list[Refusal] = []
    notes: list[str] = []
    ps_earth = np.asarray(view.surface_pressure_pa, dtype=np.float64)
    z_earth = np.asarray(view.terrain_m, dtype=np.float64)
    if edited_terrain_m is not None:
        z_world = np.asarray(edited_terrain_m, dtype=np.float64)
        terrain_rule = "the frame edit's own terrain"
    else:
        z_world = edited_terrain(world, z_earth, view.latitude_deg, view.longitude_deg)
        terrain_rule = "the preflight's reading of [world.terrain]"
    if edited_ps is not None:
        ps_world = np.asarray(edited_ps, dtype=np.float64)
        pressure_rule = "the pressure mover's own output"
    else:
        ps_world = predict_surface_pressure(world, view, z_world)
        pressure_rule = ("the world's rule in closed form on the frame grid "
                         f"(air_mass = {world.terrain.air_mass!r})")
    if ps_world.shape != ps_earth.shape:
        raise ValueError(f"edited surface pressure has shape {ps_world.shape}, "
                         f"the frame {ps_earth.shape}")

    weights = view.area_weights
    i_min, i_max = int(np.argmin(ps_world)), int(np.argmax(ps_world))
    ps_min, ps_max = float(ps_world.reshape(-1)[i_min]), float(ps_world.reshape(-1)[i_max])
    earth_min, earth_max = float(ps_earth.min()), float(ps_earth.max())
    prediction: dict[str, object] = {
        "terrain_rule": terrain_rule,
        "pressure_rule": pressure_rule,
        "grid": f"{np.size(view.latitude_deg)} x {np.size(view.longitude_deg)}",
        "surface_pressure_min_pa": round(ps_min, 1),
        "surface_pressure_min_at": _place(*view.where(i_min)),
        "terrain_at_min_m": round(float(z_world.reshape(-1)[i_min]), 1),
        "surface_pressure_max_pa": round(ps_max, 1),
        "surface_pressure_max_at": _place(*view.where(i_max)),
        "terrain_at_max_m": round(float(z_world.reshape(-1)[i_max]), 1),
        "earth_surface_pressure_min_pa": round(earth_min, 1),
        "earth_surface_pressure_max_pa": round(earth_max, 1),
        "terrain_max_m": round(float(z_world.max()), 1),
        "earth_terrain_max_m": round(float(z_earth.max()), 1),
        "columns_moved": int(np.count_nonzero(z_world != z_earth)),
        "earth_mean_surface_pressure_pa": round(
            float(np.sum(weights * ps_earth) / np.sum(weights)), 1),
        "world_mean_surface_pressure_pa": round(
            float(np.sum(weights * ps_world) / np.sum(weights)), 1),
    }

    # Refusal 1: too tall.  Read only where the world is lower in pressure
    # than Earth's own frame, so Earth's own peaks are never refused.
    floor = LIMITS["surface_pressure_floor_pa"]
    deeper = ps_world < np.minimum(ps_earth, floor.value)
    reasons: list[str] = []
    if deeper.any():
        reasons.append(
            f"the predicted lowest surface pressure is {ps_min:,.0f} Pa at "
            f"{prediction['surface_pressure_min_at']} on {z_world.reshape(-1)[i_min]:,.0f} m "
            f"of terrain, under the {floor.value:,.0f} Pa floor for taller "
            f"terrain ({_columns(int(deeper.sum()))}); Earth's own frame "
            f"reaches {earth_min:,.0f} Pa")
    if vertical is not None and ps_min < earth_min:
        p_half = vertical.a_half_pa + vertical.b_half * ps_min
        if np.any(np.diff(p_half) <= 0.0):
            k = int(np.argmax(np.diff(p_half) <= 0.0))
            reasons.append(
                f"at {ps_min:,.0f} Pa the run's vertical grid folds: layer {k} "
                "has no positive thickness")
    if reasons:
        refusals.append(Refusal("terrain_too_tall", (
            f"{REFUSALS['terrain_too_tall']}: " + "; and ".join(reasons)
            + ". Lower world.terrain.scale or set world.terrain.cap_m.")))

    # Refusal 2: over the ceiling.
    ceiling = LIMITS["surface_pressure_ceiling_pa"]
    over = ps_world > ceiling.value
    if over.any():
        rule = world.terrain.air_mass
        remedy = ("set world.terrain.air_mass = 'conserve'" if rule == "fill"
                  else "edit less terrain, or limit the edit with world.terrain.box")
        refusals.append(Refusal("pressure_over_ceiling", (
            f"{REFUSALS['pressure_over_ceiling']}: {ps_max:,.0f} Pa at "
            f"{prediction['surface_pressure_max_at']} ({_columns(int(over.sum()))}) "
            f"against {ceiling.value:,.0f} Pa, where the first radiation call "
            f"would stop the run; Earth's own frame reaches {earth_max:,.0f} Pa. "
            f"With air_mass = {rule!r}: {remedy}.")))
    elif ceiling.value - ps_max < ceiling.caution:
        notes.append(
            f"the predicted highest surface pressure, {ps_max:,.0f} Pa at "
            f"{prediction['surface_pressure_max_at']}, is within "
            f"{ceiling.caution:,.0f} Pa of the radiation ceiling "
            f"({ceiling.value:,.0f} Pa); a T255 run has raised its highest "
            "surface pressure by 558 Pa in 10 days")

    # The land skin the edit warms: 6.5 K per km of height removed.
    if view.skin_temperature_k is not None and view.land_fraction is not None:
        skin = np.asarray(view.skin_temperature_k, dtype=np.float64)
        land = np.asarray(view.land_fraction, dtype=np.float64) >= LAND_FRACTION_LAND
        if world.sea.land == "none":
            land = np.zeros_like(land)
        warmed = skin + LAPSE_K_PER_M * np.maximum(z_earth - z_world, 0.0)
        if land.any():
            hottest = int(np.argmax(np.where(land, warmed, -np.inf)))
            prediction["earth_land_skin_max_k"] = round(float(skin[land].max()), 2)
            prediction["world_land_skin_max_k"] = round(float(warmed.reshape(-1)[hottest]), 2)
            prediction["world_land_skin_max_at"] = _place(*view.where(hottest))
            # Refusal 3: a start skin over the radiation's range, read only
            # where the edit warmed it past Earth's own frame.
            limit = LIMITS["skin_temperature_max_k"]
            start = float(warmed.reshape(-1)[hottest])
            hot = land & (warmed > limit.value) & (warmed > skin)
            if hot.any():
                removed = float((z_earth - z_world).reshape(-1)[hottest])
                refusals.append(Refusal("start_skin_over_limit", (
                    f"{REFUSALS['start_skin_over_limit']}: {start:.2f} K at "
                    f"{prediction['world_land_skin_max_at']} ({_columns(int(hot.sum()))}), "
                    f"the frame's {float(skin.reshape(-1)[hottest]):.2f} K warmed 6.5 K "
                    f"per km of the {removed:,.0f} m removed, against "
                    f"{limit.value:g} K, where the first radiation call would stop "
                    "the run. Start from an analysis time when that land is dark, "
                    "or leave that terrain with world.terrain.box or cap_m.")))
        sea = ~land & np.isfinite(skin)
        if sea.any():
            prediction["sea_skin_max_k"] = round(float(skin[sea].max()), 2)
    return prediction, refusals, notes


# -- the four margins ------------------------------------------------------


#: The hottest sea a published aquaplanet profile prescribes (27 C at the
#: equator for all four of Neale and Hoskins 2000, plan package 2.2).
AQUAPLANET_SEA_MAX_K = 300.15


def _hottest_skin(world: WorldSpec, prediction) -> Margin:
    limit = LIMITS["skin_temperature_max_k"]
    earth = MEASURED_EARTH["skin_temperature_max_k"]
    title = "hottest skin expected, against the radiation's 355 K limit"

    def margin(value, rule, status=None):
        room = limit.value - value
        return Margin("hottest_skin", title, value, limit.value, "K", room,
                      status or _status(room, limit.caution), rule)

    if world.sea.land == "none":
        if world.sea.temperature in ("analysis", "zonal-mean"):
            if prediction is not None and "sea_skin_max_k" in prediction:
                value = float(prediction["sea_skin_max_k"])
                return margin(value, "no land: the hottest skin is the held sea's, "
                              "its frame maximum (a zonal mean is no warmer)")
            return Margin("hottest_skin", title, None, limit.value, "K", None, UNKNOWN,
                          "no land: the hottest skin is the held sea's; read it "
                          "from the frame")
        return margin(AQUAPLANET_SEA_MAX_K,
                      "no land: the hottest skin is the held aquaplanet sea, 27 C")
    if world.sun.obliquity_deg > EARTH_OBLIQUITY_DEG or world.changes_rotation \
            or world.sun.orbit == "eccentric":
        why = []
        if world.sun.obliquity_deg > EARTH_OBLIQUITY_DEG:
            why.append(f"a {world.sun.obliquity_deg:g} degree tilt brings summer sun "
                       "nearer the poles than Earth's")
        if world.changes_rotation:
            why.append(f"a {world.sun.day_length_h:g} h day changes how long land "
                       "heats before night")
        if world.sun.orbit == "eccentric":
            why.append("an eccentric orbit brightens the sun near closest approach")
        rule = ("not bounded: " + "; ".join(why) + ". No run of this model has "
                "measured it; Earth's own figure is "
                f"{earth.value:.2f} K")
        if world.sun.obliquity_deg > 45.0 and not world.is_invented:
            rule += (". Above about 45 degrees with land, a summer continent can "
                     "pass 355 K within a season, and the run then stops by name")
        return Margin("hottest_skin", title, None, limit.value, "K", None, UNKNOWN, rule)

    value = earth.value
    parts = [f"Earth's measured {earth.value:.2f} K at T255 (not northern summer)"]
    if world.sun.brightness != 1.0:
        factor = world.sun.brightness ** 0.25
        value *= factor
        parts[0] += (f", times {factor:.4f}, the fourth root of the sun's brightness "
                     "(radiative-equilibrium scaling, a first-order estimate)")
    if prediction is not None and "world_land_skin_max_k" in prediction:
        start = float(prediction["world_land_skin_max_k"])
        parts.append(
            f"the start's hottest land skin on the frame, after warming 6.5 K per "
            f"km of height removed, is {start:.2f} K at {prediction['world_land_skin_max_at']} "
            f"(the frame's own {float(prediction['earth_land_skin_max_k']):.2f} K)")
        if start > value:
            value = start
            parts.append("the start is the hotter")
        if start > limit.value:
            parts.append("a skin over 355 K at the start stops the first radiation call")
    if world.is_invented:
        parts.append("an invented continent's interior has not been run; Earth's "
                     "figure is used")
    return margin(value, "; ".join(parts))


def _coldest_air(world: WorldSpec) -> Margin:
    limit = LIMITS["air_temperature_min_k"]
    floor = LIMITS["radiation_table_floor_k"]
    earth = MEASURED_EARTH["air_temperature_min_k"]
    title = "coldest air expected, against the dynamics' 140 K limit"
    if world.sun.obliquity_deg > EARTH_OBLIQUITY_DEG or world.sun.day_length_h > 24.0:
        why = []
        if world.sun.obliquity_deg > EARTH_OBLIQUITY_DEG:
            why.append("a larger tilt gives a longer and wider polar night")
        if world.sun.day_length_h > 24.0:
            why.append(f"a {world.sun.day_length_h:g} h day gives longer nights")
        return Margin("coldest_air", title, None, limit.value, "K", None, UNKNOWN, (
            "not bounded: " + "; ".join(why) + f". Earth's own coldest air at T255 "
            f"is {earth.value:.2f} K; the radiation radiates air under "
            f"{floor.value:g} K as {floor.value:g} K and the dynamics stop at "
            f"{limit.value:g} K"))
    value = earth.value
    rule = f"Earth's measured {earth.value:.2f} K at T255"
    if world.sun.brightness != 1.0:
        factor = world.sun.brightness ** 0.25
        value *= factor
        rule += (f", times {factor:.4f}, the fourth root of the sun's brightness "
                 "(a first-order estimate)")
    rule += (f"; the radiation tables end at {floor.value:g} K and colder air "
             "radiates as that")
    room = value - limit.value
    return Margin("coldest_air", title, value, limit.value, "K", room,
                  _status(room, limit.caution), rule)


def _water_store(world: WorldSpec, duration_s) -> Margin:
    limit = LIMITS["ocean_water_store_kg_m2"]
    earth = MEASURED_EARTH["ocean_store_days"]
    title = ("days until the first ocean column empties its 500 kg m-2 water "
             "store, against the run's length")
    horizon = earth.value
    rule = f"Earth's measured {earth.value:g} days on the 0.1.6 line"
    if world.sun.brightness != 1.0:
        horizon /= world.sun.brightness
        rule += (f", divided by the sun's brightness {world.sun.brightness:g} "
                 "(evaporation taken in proportion to sunlight, a first-order "
                 "estimate)")
    if world.changes_coast or world.changes_sea:
        rule += ("; this world's sea is not Earth's and has not been run, so "
                 "Earth's figure stands in")
    if duration_s is None:
        return Margin("water_store", title, horizon, None, "days", None,
                      UNKNOWN, rule + "; no run length given")
    days = float(duration_s) / 86400.0
    room = horizon - days
    rule += f"; the run is {days:g} day" + ("" if days == 1.0 else "s")
    return Margin("water_store", title, horizon, days, "days", room,
                  _status(room, limit.caution), rule)


def _bottom_layer(world: WorldSpec, prediction, vertical) -> Margin:
    floor = LIMITS["surface_pressure_floor_pa"]
    earth = MEASURED_EARTH["surface_pressure_min_pa"]
    title = "bottom layer thickness at the lowest surface pressure"
    if vertical is None:
        return Margin("bottom_layer", title, None, floor.value, "Pa", None, UNKNOWN,
                      "no vertical grid given")
    da = float(vertical.delta_a[-1])
    db = float(vertical.delta_b[-1])

    def thickness(ps):
        return da + db * ps

    at_floor = thickness(floor.value)
    if prediction is None:
        if world.changes_terrain or world.is_invented:
            return Margin("bottom_layer", title, None, at_floor, "Pa", None, UNKNOWN,
                          "the terrain moves and no frame was read; at the "
                          f"{floor.value:,.0f} Pa floor the bottom layer is "
                          f"{at_floor:.1f} Pa thick")
        ps_min = earth.value
        source = f"Earth's measured lowest {earth.value:,.0f} Pa at T255"
    else:
        ps_min = float(prediction["surface_pressure_min_pa"])
        source = (f"the predicted lowest {ps_min:,.0f} Pa at "
                  f"{prediction['surface_pressure_min_at']} (on the frame grid)")
    value = thickness(ps_min)
    rule = (f"{source}: the bottom layer is {value:.1f} Pa thick, against "
            f"{thickness(101_325.0):.1f} Pa at 1013.25 hPa and {at_floor:.1f} Pa at "
            f"the {floor.value:,.0f} Pa floor for taller terrain")
    room = value - at_floor
    return Margin("bottom_layer", title, value, at_floor, "Pa", room,
                  _status(room, 0.0), rule)


__all__ = [
    "AQUAPLANET_SEA_MAX_K", "CLOSE", "FrameView", "INSIDE", "LAPSE_K_PER_M",
    "LIMITS", "Limit", "MARGIN_KEYS", "MEASURED_EARTH", "Margin", "Measured",
    "OVER", "PreflightReport", "REFUSALS", "Refusal", "UNKNOWN", "WorldRefused",
    "box_weight", "edited_terrain", "predict_surface_pressure", "preflight",
]
