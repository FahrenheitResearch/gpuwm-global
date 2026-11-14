"""The ``[world]`` table and the world object it is parsed into.

This is the contract every other part of the worlds track codes against.
The key names below are FROZEN: a renamed key breaks every template, the
launcher that sets keys by pattern and every test that writes a world.

What a user writes (every key optional; an absent key is Earth's)::

    [world]
    label = "flat-earth"           # printed on every receipt, tape and picture

    [world.terrain]
    scale = 0.0                    # 0 = no mountains, 1 = Earth
    air_mass = "conserve"          # keep Earth's amount of air; "fill" keeps sea-level pressure
    cap_m = 1500.0                 # optional: no terrain higher than this
    [world.terrain.box]            # optional: scale and cap act only inside this box
    south_deg = 20.0
    north_deg = 50.0
    west_deg = 60.0
    east_deg = 110.0
    taper_deg = 5.0                # back to Earth's terrain over this many degrees

    [world.sun]
    obliquity_deg = 23.5
    brightness = 1.0               # times the model's own solar constant
    day_length_h = 24.0            # the rotation rate follows it
    orbit = "earth"                # or "perpetual-equinox", or "eccentric"
    eccentricity = 0.1             # "eccentric" only
    perihelion_deg = 0.0           # "eccentric" only

    [world.sea]
    level_m = 0.0
    land = "keep"                  # "none" = waterworld
    temperature = "analysis"       # or "zonal-mean", or a named published profile
    ice = "analysis"               # or "zonal", "none"

    [world.air]
    co2_ppm = "dated"              # or a number
    ch4_ppb = "dated"              # or a number
    n2o_ppb = "dated"              # or a number

    [world.surface]                # only for an invented world
    folder = "pangea-250ma.world"  # the user's map and everything built from it
    land_cover = "open-shrubland"  # or "barren"
    soil_wetness = 0.3             # required with a folder: fraction of saturation

The identity rule: a world whose every setting is Earth's has the identity
``"none"`` and joins nothing, so a config with no ``[world]`` table and a
config with an all-Earth one hash the same, write the same checkpoints and
the same receipts.  Any other world's identity is the sha256 of its
canonical settings (the label is a name and is left out), and that hash
joins the config hash, the receipt, the checkpoint metadata and, for a
world with its own surface, the statics cache name.

A setting that has no effect on the arithmetic is left out of the identity
(the precedent of ``ArwenGlobalConfig.config_identity``): the air-mass rule
acts only where the terrain moves, so on a world that keeps Earth's
terrain and surface it is dropped.

An invented world's folder enters the identity by its NAME, never its path
(a path is a fact about one host, and the run's own config copy sits in
another directory).  The folder's content hash is not part of the identity,
on the precedent of the real statics cache: the fields travel with every
checkpoint, and the receipt records the folder's own hashes beside the run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re

#: The version of the world record.  It is part of every non-Earth
#: identity so a later change of meaning moves every world hash with it.
WORLD_SCHEMA = "arwen-global.world/v1"

#: The identity of every world whose settings are all Earth's.
EARTH_IDENTITY = "none"

#: Key names a launcher sets by pattern in a config file and
#: requires to occur exactly once.  No key anywhere under ``[world]`` may
#: carry one of these names (plan section 4.1).
LAUNCHER_RESERVED_KEYS = frozenset({
    "name", "duration_s", "output_interval_s", "analysis_grib",
    "start_time_utc", "analysis_mapping", "radiation_column_chunk",
})

#: Every key the table accepts, by sub-table ("" is ``[world]`` itself).
WORLD_KEYS: dict[str, frozenset[str]] = {
    "": frozenset({"label", "terrain", "sun", "sea", "air", "surface"}),
    "terrain": frozenset({"scale", "air_mass", "cap_m", "box"}),
    "terrain.box": frozenset({
        "south_deg", "north_deg", "west_deg", "east_deg", "taper_deg"}),
    "sun": frozenset({
        "obliquity_deg", "brightness", "day_length_h", "orbit",
        "eccentricity", "perihelion_deg"}),
    "sea": frozenset({"level_m", "land", "temperature", "ice"}),
    "air": frozenset({"co2_ppm", "ch4_ppb", "n2o_ppb"}),
    "surface": frozenset({"folder", "land_cover", "soil_wetness"}),
}

#: Settings refused by name until the wave that builds them (plan 4.11):
#: each is a key a user is likely to try, and the refusal says why.
NOT_SUPPORTED_YET: dict[str, str] = {
    "gravity_m_s2": (
        "another gravity is not supported yet: about 240 lines of the "
        "engine hold Earth's constants and are not wired to a world"),
    "radius_km": (
        "another radius is not supported yet: the grid, the spectral "
        "operators and the physics hold Earth's radius"),
    "year_length_d": (
        "another year length is not supported yet: the radiation clock "
        "runs WRF's fixed 365-day orbit"),
    "air.composition": (
        "another air composition is not supported yet: the gas constants, "
        "heat capacities and the radiation tables are Earth air's"),
    "air.surface_pressure_pa": (
        "another amount of air is not supported yet: the convection and "
        "microphysics hold absolute-pressure thresholds that are being "
        "audited (docs/arwen-global-worlds-pressure-audit.md) and the "
        "radiation refuses above 115,000 Pa"),
}

#: Air-mass rules for moving the surface pressure under changed terrain.
AIR_MASS_RULES = ("conserve", "fill")
#: Sun orbits.  "earth" is the model's own WRF orbit.
ORBITS = ("earth", "perpetual-equinox", "eccentric")
#: What becomes of the land.
SEA_LAND_RULES = ("keep", "none")
#: The four published aquaplanet sea-temperature profiles
#: (Neale and Hoskins 2000), by the names the world table uses.
SEA_TEMPERATURE_PROFILES = ("aqua-control", "aqua-peaked", "aqua-flat", "aqua-qobs")
#: Every sea-temperature source.
SEA_TEMPERATURE_SOURCES = ("analysis", "zonal-mean") + SEA_TEMPERATURE_PROFILES
#: Sea-ice rules.
SEA_ICE_RULES = ("analysis", "zonal", "none")
#: Land-cover classes an invented world may be given.
LAND_COVERS = ("open-shrubland", "barren")
#: The value a trace gas takes when it follows the run's dated table.
DATED = "dated"

#: A label is printed on tapes, pictures and file names.
_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")

# Range checks.  Each bound is a parse-time sanity bound with its reason;
# the world statement and the preflight (package 1.6) refuse narrower,
# per-world, from the frame they are given.
TERRAIN_SCALE_RANGE = (0.0, 2.0)
TERRAIN_CAP_RANGE_M = (0.0, 9000.0)
BOX_TAPER_RANGE_DEG = (0.0, 30.0)
OBLIQUITY_RANGE_DEG = (0.0, 180.0)
BRIGHTNESS_RANGE = (0.1, 4.0)
#: The shortest day accepted: the stability bound of the model's rotation
#: term at faster rotation is not measured yet (package 1.8), and the plan
#: holds anything faster than a 12-hour day as refused until it is.
DAY_LENGTH_RANGE_H = (12.0, 2400.0)
ECCENTRICITY_RANGE = (0.0, 0.3)
SEA_LEVEL_RANGE_M = (0.0, 200.0)
CO2_RANGE_PPM = (0.0, 1.0e4)   # exclusive, as physics/native_options.py
TRACE_RANGE_PPB = (0.0, 1.0e5)  # exclusive
SOIL_WETNESS_RANGE = (0.0, 1.0)

#: Earth's values, which are also the defaults.  Obliquity 23.5 degrees is
#: the value the model's own sun uses (WRF ``radconst``, clock.py).
EARTH_TERRAIN_SCALE = 1.0
EARTH_OBLIQUITY_DEG = 23.5
EARTH_BRIGHTNESS = 1.0
EARTH_DAY_LENGTH_H = 24.0


@dataclass(frozen=True)
class TerrainBox:
    """A latitude-longitude box the terrain edit acts inside, tapering back
    to Earth's terrain over ``taper_deg``.  ``west_deg`` greater than
    ``east_deg`` means the box crosses the 180th meridian."""

    south_deg: float
    north_deg: float
    west_deg: float
    east_deg: float
    taper_deg: float = 0.0


@dataclass(frozen=True)
class Terrain:
    scale: float = EARTH_TERRAIN_SCALE
    air_mass: str = "conserve"
    cap_m: float | None = None
    box: TerrainBox | None = None

    @property
    def is_earth(self) -> bool:
        """True when the terrain is Earth's (the air-mass rule then has
        nothing to act on)."""
        return self.scale == EARTH_TERRAIN_SCALE and self.cap_m is None


@dataclass(frozen=True)
class Sun:
    obliquity_deg: float = EARTH_OBLIQUITY_DEG
    brightness: float = EARTH_BRIGHTNESS
    day_length_h: float = EARTH_DAY_LENGTH_H
    orbit: str = "earth"
    eccentricity: float | None = None
    perihelion_deg: float | None = None

    @property
    def is_earth(self) -> bool:
        return self == Sun()

    @property
    def rotation_factor(self) -> float:
        """The planet's rotation rate over Earth's: exactly 1.0 on Earth.
        Day length changes the hour angle and the rotation rate together,
        from this one number (plan 4.6)."""
        return EARTH_DAY_LENGTH_H / self.day_length_h


@dataclass(frozen=True)
class Sea:
    level_m: float = 0.0
    land: str = "keep"
    temperature: str = "analysis"
    ice: str = "analysis"

    @property
    def is_earth(self) -> bool:
        return self == Sea()


@dataclass(frozen=True)
class Air:
    co2_ppm: float | str = DATED
    ch4_ppb: float | str = DATED
    n2o_ppb: float | str = DATED

    @property
    def is_earth(self) -> bool:
        return self == Air()


@dataclass(frozen=True)
class Surface:
    """An invented world's surface.  ``folder`` is kept as written; its
    NAME is what enters the identity (``folder_name``)."""

    folder: str | None = None
    land_cover: str | None = None
    soil_wetness: float | None = None

    @property
    def is_earth(self) -> bool:
        return self.folder is None

    @property
    def folder_name(self) -> str | None:
        if self.folder is None:
            return None
        text = self.folder.rstrip("/\\")
        return PureWindowsPath(text).name if "\\" in text else PurePosixPath(text).name


@dataclass(frozen=True)
class WorldSpec:
    """One world, parsed and range-checked.  Immutable and hashable.

    Every other package reads a world through this object:

    - ``is_earth``: every setting is Earth's.  World code paths may run
      only when this is False; with it True a run must be bit-identical
      to a run with no ``[world]`` table.
    - ``identity`` (dict or None), ``identity_hash`` ("none" or 64 hex),
      ``token`` ("none" or 12 hex, for cache and file names).
    - the per-part predicates ``changes_terrain``, ``changes_sun``,
      ``changes_rotation``, ``changes_sea``, ``changes_coast``,
      ``changes_air``, ``is_invented``.
    - ``changed_settings()``: the dotted keys that differ from Earth's.
    - ``record()``: the dict a receipt, a checkpoint or a tape stores.
    - ``to_toml()``: the ``[world]`` table text that parses back to this.
    """

    label: str = "earth"
    terrain: Terrain = field(default_factory=Terrain)
    sun: Sun = field(default_factory=Sun)
    sea: Sea = field(default_factory=Sea)
    air: Air = field(default_factory=Air)
    surface: Surface = field(default_factory=Surface)

    # -- the identity rule ---------------------------------------------

    @property
    def is_earth(self) -> bool:
        return (self.terrain.is_earth and self.sun.is_earth
                and self.sea.is_earth and self.air.is_earth
                and self.surface.is_earth)

    @property
    def identity(self) -> dict[str, object] | None:
        """The canonical settings that define this world's arithmetic, or
        None for Earth.  The label is a name, not a setting, and is left
        out; so is any setting that acts on nothing (see the module
        docstring)."""
        if self.is_earth:
            return None
        terrain = asdict(self.terrain)
        if self.terrain.is_earth and self.surface.is_earth:
            # The air-mass rule moves pressure only where terrain moves.
            del terrain["air_mass"]
        surface = asdict(self.surface)
        surface["folder"] = self.surface.folder_name
        return {
            "schema": WORLD_SCHEMA,
            "terrain": terrain,
            "sun": asdict(self.sun),
            "sea": asdict(self.sea),
            "air": asdict(self.air),
            "surface": surface,
        }

    @property
    def identity_hash(self) -> str:
        identity = self.identity
        if identity is None:
            return EARTH_IDENTITY
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def token(self) -> str:
        """Short form of ``identity_hash`` for cache and file names."""
        digest = self.identity_hash
        return digest if digest == EARTH_IDENTITY else digest[:12]

    # -- what changed --------------------------------------------------

    @property
    def changes_terrain(self) -> bool:
        return not self.terrain.is_earth

    @property
    def changes_sun(self) -> bool:
        return not self.sun.is_earth

    @property
    def changes_rotation(self) -> bool:
        return self.sun.day_length_h != EARTH_DAY_LENGTH_H

    @property
    def changes_sea(self) -> bool:
        return not self.sea.is_earth

    @property
    def changes_coast(self) -> bool:
        """The land mask differs from Earth's: a flood, a waterworld or an
        invented world.  Drawing code must then use the tape's own mask."""
        return (self.sea.level_m != 0.0 or self.sea.land != "keep"
                or self.is_invented)

    @property
    def changes_air(self) -> bool:
        return not self.air.is_earth

    @property
    def is_invented(self) -> bool:
        return not self.surface.is_earth

    def changed_settings(self) -> list[str]:
        """Dotted names of every setting that differs from Earth's, in table
        order, e.g. ``["terrain.scale", "sun.obliquity_deg"]``."""
        earth = WorldSpec()
        out: list[str] = []
        for part in ("terrain", "sun", "sea", "air", "surface"):
            mine, theirs = getattr(self, part), getattr(earth, part)
            for item in fields(mine):
                if getattr(mine, item.name) != getattr(theirs, item.name):
                    out.append(f"{part}.{item.name}")
        if "terrain.air_mass" in out and self.terrain.is_earth and self.surface.is_earth:
            out.remove("terrain.air_mass")
        return out

    def record(self) -> dict[str, object]:
        """What a receipt, a checkpoint or a tape stores about the world:
        the label, the identity hash and the settings (None for Earth)."""
        return {
            "label": self.label,
            "identity": self.identity_hash,
            "settings": self.identity,
        }

    def to_toml(self) -> str:
        """The ``[world]`` table that parses back to this world.  Earth's
        values are written out in full, so the text is also a template."""
        lines = ["[world]", f"label = {_toml_value(self.label)}"]
        for part in ("terrain", "sun", "sea", "air", "surface"):
            value = getattr(self, part)
            rows = [(item.name, getattr(value, item.name)) for item in fields(value)]
            rows = [(k, v) for k, v in rows if v is not None and k != "box"]
            if part == "sun" and value.orbit == "perpetual-equinox":
                rows = [(k, v) for k, v in rows if k != "obliquity_deg"]
            if part == "surface" and value.is_earth:
                continue
            lines.append("")
            lines.append(f"[world.{part}]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in rows)
            if part == "terrain" and self.terrain.box is not None:
                lines.append("")
                lines.append("[world.terrain.box]")
                for item in fields(self.terrain.box):
                    lines.append(
                        f"{item.name} = {_toml_value(getattr(self.terrain.box, item.name))}")
        return "\n".join(lines) + "\n"


#: Earth.  ``WorldSpec()`` is equal to it.
EARTH = WorldSpec()


def _toml_value(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(float(value))


# -- parsing --------------------------------------------------------------


def _where(path: str, key: str) -> str:
    return f"world.{path}.{key}" if path else f"world.{key}"


def _check_keys(table: dict, path: str) -> None:
    for key in table:
        dotted = f"{path}.{key}" if path else key
        if key in LAUNCHER_RESERVED_KEYS:
            raise ValueError(
                f"{_where(path, key)} is not a world setting: the key name "
                f"{key!r} is set by pattern by a launcher, which requires it "
                "to occur once in the file, so no key under [world] may use it")
        if dotted in NOT_SUPPORTED_YET:
            raise ValueError(
                f"{_where(path, key)}: {NOT_SUPPORTED_YET[dotted]}")
    unknown = sorted(set(table) - WORLD_KEYS[path])
    if unknown:
        where = f"[world.{path}]" if path else "[world]"
        raise ValueError(
            f"unknown keys in {where}: {', '.join(unknown)}; accepted: "
            + ", ".join(sorted(WORLD_KEYS[path])))


def _sub(table: dict, path: str, key: str) -> dict:
    value = table.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{_where(path, key)}] must be a TOML table")
    return value


def _number(value, name: str, lo: float, hi: float, *,
            open_low: bool = False, open_high: bool = False,
            why: str = "") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    low_ok = number > lo if open_low else number >= lo
    high_ok = number < hi if open_high else number <= hi
    if not (low_ok and high_ok):
        left = "(" if open_low else "["
        right = ")" if open_high else "]"
        raise ValueError(
            f"{name} must lie in {left}{lo:g}, {hi:g}{right}, got {number:g}"
            + (f": {why}" if why else ""))
    return number


def _choice(value, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    text = value.strip().lower()
    if text not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return text


def _gas(value, name: str, lo: float, hi: float) -> float | str:
    if isinstance(value, str):
        if value.strip().lower() != DATED:
            raise ValueError(
                f"{name} must be {DATED!r} (the run's dated table) or a number, "
                f"got {value!r}")
        return DATED
    return _number(value, name, lo, hi, open_low=True, open_high=True)


def parse_world(table: dict | None) -> WorldSpec:
    """Parse and range-check a ``[world]`` table (as ``tomllib`` returns
    it).  ``None`` (no table) is Earth.  Every refusal names the key."""
    if table is None:
        return EARTH
    if not isinstance(table, dict):
        raise ValueError("[world] must be a TOML table")
    _check_keys(table, "")
    terrain_t = _sub(table, "", "terrain")
    sun_t = _sub(table, "", "sun")
    sea_t = _sub(table, "", "sea")
    air_t = _sub(table, "", "air")
    surface_t = _sub(table, "", "surface")
    for path, sub in (("terrain", terrain_t), ("sun", sun_t), ("sea", sea_t),
                      ("air", air_t), ("surface", surface_t)):
        _check_keys(sub, path)

    # [world.terrain]
    scale = _number(terrain_t.get("scale", EARTH_TERRAIN_SCALE), "world.terrain.scale",
                    *TERRAIN_SCALE_RANGE,
                    why="0 removes the mountains, 1 is Earth's; the preflight "
                        "refuses a taller world its frame cannot carry")
    air_mass = _choice(terrain_t.get("air_mass", "conserve"),
                       "world.terrain.air_mass", AIR_MASS_RULES)
    cap_m = None
    if "cap_m" in terrain_t:
        cap_m = _number(terrain_t["cap_m"], "world.terrain.cap_m", *TERRAIN_CAP_RANGE_M)
    box = None
    if "box" in terrain_t:
        box_t = _sub(terrain_t, "terrain", "box")
        _check_keys(box_t, "terrain.box")
        for key in ("south_deg", "north_deg", "west_deg", "east_deg"):
            if key not in box_t:
                raise ValueError(f"world.terrain.box.{key} is required")
        south = _number(box_t["south_deg"], "world.terrain.box.south_deg", -90.0, 90.0)
        north = _number(box_t["north_deg"], "world.terrain.box.north_deg", -90.0, 90.0)
        if north <= south:
            raise ValueError("world.terrain.box.north_deg must be north of south_deg")
        west = _number(box_t["west_deg"], "world.terrain.box.west_deg", -180.0, 360.0)
        east = _number(box_t["east_deg"], "world.terrain.box.east_deg", -180.0, 360.0)
        if west == east:
            raise ValueError("world.terrain.box west_deg and east_deg may not be equal")
        taper = _number(box_t.get("taper_deg", 0.0), "world.terrain.box.taper_deg",
                        *BOX_TAPER_RANGE_DEG)
        box = TerrainBox(south, north, west, east, taper)
        if scale == EARTH_TERRAIN_SCALE and cap_m is None:
            raise ValueError(
                "[world.terrain.box] limits the terrain edit to a box, and "
                "this table edits nothing: set scale or cap_m")
    terrain = Terrain(scale=scale, air_mass=air_mass, cap_m=cap_m, box=box)

    # [world.sun]
    orbit = _choice(sun_t.get("orbit", "earth"), "world.sun.orbit", ORBITS)
    if orbit == "perpetual-equinox":
        if "obliquity_deg" in sun_t:
            raise ValueError(
                "world.sun.obliquity_deg has no meaning under orbit = "
                "'perpetual-equinox': the sun stays over the equator")
        obliquity = 0.0
    else:
        obliquity = _number(sun_t.get("obliquity_deg", EARTH_OBLIQUITY_DEG),
                            "world.sun.obliquity_deg", *OBLIQUITY_RANGE_DEG)
    brightness = _number(sun_t.get("brightness", EARTH_BRIGHTNESS),
                         "world.sun.brightness", *BRIGHTNESS_RANGE)
    day = _number(sun_t.get("day_length_h", EARTH_DAY_LENGTH_H),
                  "world.sun.day_length_h", *DAY_LENGTH_RANGE_H,
                  why="a day shorter than 12 h waits for the measured stability "
                      "bound of the rotation term, and a day "
                      "longer than 100 days waits for the top absorber's "
                      "setting for slowly rotating worlds")
    eccentricity = perihelion = None
    if orbit == "eccentric":
        if "eccentricity" not in sun_t:
            raise ValueError("world.sun.eccentricity is required with orbit = 'eccentric'")
        eccentricity = _number(sun_t["eccentricity"], "world.sun.eccentricity",
                               *ECCENTRICITY_RANGE)
        perihelion = _number(sun_t.get("perihelion_deg", 0.0), "world.sun.perihelion_deg",
                             0.0, 360.0, open_high=True)
    else:
        for key in ("eccentricity", "perihelion_deg"):
            if key in sun_t:
                raise ValueError(f"world.sun.{key} is read only with orbit = 'eccentric'")
    sun = Sun(obliquity_deg=obliquity, brightness=brightness, day_length_h=day,
              orbit=orbit, eccentricity=eccentricity, perihelion_deg=perihelion)

    # [world.sea]
    level_value = sea_t.get("level_m", 0.0)
    if (not isinstance(level_value, bool) and isinstance(level_value, (int, float))
            and level_value < 0.0):
        raise ValueError(
            "world.sea.level_m below 0 is not supported yet: a lower sea "
            "uncovers land the edited-Earth surface has no rule to build")
    level = _number(level_value, "world.sea.level_m", *SEA_LEVEL_RANGE_M)
    land = _choice(sea_t.get("land", "keep"), "world.sea.land", SEA_LAND_RULES)
    temperature = _choice(sea_t.get("temperature", "analysis"),
                          "world.sea.temperature", SEA_TEMPERATURE_SOURCES)
    ice = _choice(sea_t.get("ice", "analysis"), "world.sea.ice", SEA_ICE_RULES)
    sea = Sea(level_m=level, land=land, temperature=temperature, ice=ice)

    # [world.air]
    air = Air(
        co2_ppm=_gas(air_t.get("co2_ppm", DATED), "world.air.co2_ppm", *CO2_RANGE_PPM),
        ch4_ppb=_gas(air_t.get("ch4_ppb", DATED), "world.air.ch4_ppb", *TRACE_RANGE_PPB),
        n2o_ppb=_gas(air_t.get("n2o_ppb", DATED), "world.air.n2o_ppb", *TRACE_RANGE_PPB),
    )

    # [world.surface]
    folder = surface_t.get("folder")
    if folder is None:
        for key in ("land_cover", "soil_wetness"):
            if key in surface_t:
                raise ValueError(
                    f"world.surface.{key} is read only with world.surface.folder: "
                    "an edited Earth keeps Earth's land cover and soil")
        surface = Surface()
    else:
        if not isinstance(folder, str) or not folder.strip():
            raise ValueError("world.surface.folder must be a non-empty path")
        folder = folder.strip()
        if "soil_wetness" not in surface_t:
            raise ValueError(
                "world.surface.soil_wetness is required with a folder: the "
                "starting soil wetness (fraction of saturation, 0 to 1) decides "
                "much of an invented world's first weeks and has no Earth value")
        surface = Surface(
            folder=folder,
            land_cover=_choice(surface_t.get("land_cover", "open-shrubland"),
                               "world.surface.land_cover", LAND_COVERS),
            soil_wetness=_number(surface_t["soil_wetness"], "world.surface.soil_wetness",
                                 *SOIL_WETNESS_RANGE),
        )
        if not surface.folder_name or surface.folder_name in (".", ".."):
            raise ValueError("world.surface.folder must name a folder")

    # Combinations.
    if land == "none":
        if level != 0.0:
            raise ValueError(
                "world.sea.level_m has no coast to move on a waterworld "
                "(world.sea.land = 'none')")
        if scale != 0.0 or cap_m is not None or box is not None:
            raise ValueError(
                "a waterworld (world.sea.land = 'none') is flat: set "
                "[world.terrain] scale = 0 and no cap_m or box")
    if surface.folder is not None:
        if scale != EARTH_TERRAIN_SCALE or cap_m is not None or box is not None:
            raise ValueError(
                "an invented world's terrain comes from its own map: "
                "[world.terrain] scale, cap_m and box are read only on an "
                "edited Earth")
        if level != 0.0 or land != "keep":
            raise ValueError(
                "an invented world's coast comes from its own map: "
                "world.sea.level_m and world.sea.land are read only on an "
                "edited Earth")
        if temperature == "analysis" or ice == "analysis":
            raise ValueError(
                "an invented world's sea has no analysis to take: set "
                "world.sea.temperature to 'zonal-mean' or a published profile "
                "and world.sea.ice to 'zonal' or 'none'")

    world = WorldSpec(label="earth", terrain=terrain, sun=sun, sea=sea, air=air,
                      surface=surface)

    # [world] label
    if "label" in table:
        label = table["label"]
        if not isinstance(label, str) or not _LABEL.match(label.strip()):
            raise ValueError(
                "world.label must be 1 to 48 characters of lower-case letters, "
                f"digits, '.', '_' or '-', starting with a letter or digit, got {label!r}")
        label = label.strip()
    elif world.is_earth:
        label = "earth"
    else:
        raise ValueError(
            "world.label is required for a world that is not Earth's: it is "
            "printed on every receipt, tape and picture")
    if label == "earth" and not world.is_earth:
        raise ValueError(
            "world.label 'earth' is kept for Earth; this world changes "
            + ", ".join(world.changed_settings()))
    return WorldSpec(label=label, terrain=terrain, sun=sun, sea=sea, air=air,
                     surface=surface)


def world_identity_of(record) -> str:
    """The world identity a stored record (checkpoint metadata, receipt)
    carries.  A record written before worlds existed has none and reads as
    Earth.  Accepts the dict ``WorldSpec.record()`` returns, a bare
    identity string, a mapping holding either under ``"world"``, or None."""
    if record is None:
        return EARTH_IDENTITY
    if isinstance(record, str):
        return record or EARTH_IDENTITY
    if isinstance(record, dict):
        if "identity" in record:
            return str(record["identity"] or EARTH_IDENTITY)
        if "world" in record:
            return world_identity_of(record["world"])
        return EARTH_IDENTITY
    raise TypeError(f"not a world record: {type(record).__name__}")
