"""The world statement: what a world is, printed before any card time.

Printed by ``gpuwm-global world describe``, stored in the run receipt and
stamped (by its hash) on every tape (worlds plan, section 4.8).  It says,
for one world:

- **changed**: every setting that is not Earth's;
- **invented**: what the model had to make up, and by which rule;
- **held**: what stays Earth's or stays fixed for the whole run;
- **the label** of plan section 2: ``science``, ``illustration`` or
  ``not supported yet``, with the limit that goes with it;
- **the one sentence** that goes on every result;
- **the four margins** and any **refusal** from the preflight
  (``world.preflight``).

The text is a pure function of the world, the run's length and the
preflight's findings: no clock, host or path enters it, so the same world
on the same frame gives the same statement and the same hash.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .preflight import PreflightReport, preflight
from .spec import (
    DATED,
    EARTH_DAY_LENGTH_H,
    EARTH_OBLIQUITY_DEG,
    WorldSpec,
)

#: The version of the statement record.
STATEMENT_SCHEMA = "arwen-global.world-statement/v1"

#: The three labels of plan section 2, the owner's own words, plus Earth's.
SCIENCE = "science"
ILLUSTRATION = "illustration"
NOT_SUPPORTED_YET = "not supported yet"
EARTH_CLASS = "earth"
#: Worst last: a world takes the worst label of any of its changes.
_CLASS_ORDER = (EARTH_CLASS, SCIENCE, ILLUSTRATION, NOT_SUPPORTED_YET)

#: Pre-industrial CO2, ppm, for the multiples the labels name.
PREINDUSTRIAL_CO2_PPM = 280.0
#: The CO2 range plan section 2.2 lists as an illustration (180 to 1,120 ppm).
CO2_ILLUSTRATION_RANGE_PPM = (180.0, 1120.0)
#: Above eight times pre-industrial a CO2 world is unvalidated (plan 2.2).
CO2_VALIDATED_MULTIPLE = 8.0


@dataclass(frozen=True)
class Verdict:
    """One reason a world carries the label it does."""

    label: str
    setting: str
    limit: str


def verdicts(world: WorldSpec) -> tuple[Verdict, ...]:
    """The label each change earns under plan section 2, in table order."""
    out: list[Verdict] = []
    t, sun, sea, air = world.terrain, world.sun, world.sea, world.air
    if world.is_invented:
        out.append(Verdict(ILLUSTRATION, "surface.folder", (
            "an invented continent: sea temperature, land cover and soil are "
            "set by stated rules, and the starting soil wetness the user typed "
            "decides the continental interior for more than a month. Good for "
            "Earth's air over this map; it is not that world's past or climate, "
            "and days 0 to 20 are spin-up")))
    if world.changes_terrain and sea.land != "none":
        if t.scale > 1.0:
            out.append(Verdict(ILLUSTRATION, "terrain.scale", (
                "taller mountains: refused when the predicted lowest surface "
                "pressure is under 450 hPa, where the model's bottom layer thins")))
        else:
            out.append(Verdict(SCIENCE, "terrain", (
                "mountains lowered or removed: science for days 2 to 6 near the "
                "removed ranges. Days 0 to 2 are the air falling into the space "
                "where rock was; after about a week, and anywhere far from the "
                "ranges, a difference is shown only where it exceeds the noise "
                "floor of a perturbed control")))
    if sun.obliquity_deg != EARTH_OBLIQUITY_DEG and sun.orbit != "perpetual-equinox":
        limit = ("another tilt: correct sun geometry, with ozone and vegetation "
                 "on Earth's calendar; for the first weeks from an analysis")
        if sun.obliquity_deg > 45.0 and sea.land != "none":
            limit += ("; above about 45 degrees with land, a summer continent can "
                      "pass the radiation's 355 K skin limit within a season and "
                      "the run stops by name")
        out.append(Verdict(ILLUSTRATION, "sun.obliquity_deg", limit))
    if sun.orbit != "earth":
        out.append(Verdict(ILLUSTRATION, "sun.orbit", (
            f"a {sun.orbit} orbit: the sun's geometry is the world's, everything "
            "keyed to the calendar is Earth's")))
    if sun.brightness != 1.0:
        out.append(Verdict(ILLUSTRATION, "sun.brightness", (
            "another sun: over a held ocean this is the land and atmosphere's "
            "fast adjustment, and a 10-day difference map from one twin is "
            "mostly chaos")))
    if sun.day_length_h != EARTH_DAY_LENGTH_H:
        out.append(Verdict(ILLUSTRATION, "sun.day_length_h", (
            "another day length: an adjustment from a state balanced for "
            "Earth's rotation, not comparable with published equilibrium "
            "results that took 1,500 days of spin-up")))
    if sea.land == "none":
        out.append(Verdict(ILLUSTRATION, "sea.land", (
            "a waterworld: the prescribed sea temperature largely decides the "
            "answer, and no ice can grow or melt")))
    elif sea.level_m != 0.0:
        out.append(Verdict(ILLUSTRATION, "sea.level_m", (
            "a higher sea: new coastlines under today's weather; the ice sheets "
            "whose melting would raise the sea are still there")))
    if sea.temperature != "analysis" or sea.ice != "analysis":
        if not world.is_invented and sea.land != "none":
            out.append(Verdict(ILLUSTRATION, "sea", (
                "another sea surface: held for the whole run, so it decides much "
                "of the answer")))
    for gas, value in (("co2_ppm", air.co2_ppm), ("ch4_ppb", air.ch4_ppb),
                       ("n2o_ppb", air.n2o_ppb)):
        if value == DATED:
            continue
        if gas == "co2_ppm":
            out.append(Verdict(ILLUSTRATION, "air.co2_ppm", _co2_limit(float(value))))
        else:
            out.append(Verdict(ILLUSTRATION, f"air.{gas}", (
                "another trace gas: over a held ocean this is the fast "
                "adjustment, and its forcing is a number, not a weather map")))
    return tuple(out)


def _co2_limit(ppm: float) -> str:
    multiple = ppm / PREINDUSTRIAL_CO2_PPM
    lo, hi = CO2_ILLUSTRATION_RANGE_PPM
    text = (f"CO2 at {ppm:g} ppm ({multiple:.2f} times pre-industrial): over a "
            "held ocean this is the land and atmosphere's fast adjustment; four "
            "times CO2 can warm the troposphere by at most about 0.06 K a day")
    if multiple > CO2_VALIDATED_MULTIPLE:
        text += "; unvalidated above eight times pre-industrial"
    elif ppm > hi:
        text += (f"; above the {lo:g} to {hi:g} ppm range shown as an illustration, "
                 "inside the eight times pre-industrial the radiation was checked to")
    elif ppm < lo:
        text += f"; below the {lo:g} ppm of the coldest glacial air"
    return text


def world_class(world: WorldSpec) -> str:
    """The world's label: the worst of its changes' labels."""
    if world.is_earth:
        return EARTH_CLASS
    found = verdicts(world)
    if not found:
        return ILLUSTRATION
    return max((v.label for v in found), key=_CLASS_ORDER.index)


def invented(world: WorldSpec) -> tuple[str, ...]:
    """What the model makes up for this world, each with its rule."""
    out: list[str] = []
    t, sun, sea = world.terrain, world.sun, world.sea
    if world.changes_terrain:
        out.append(
            "new terrain: " + ", ".join(filter(None, (
                f"scale {t.scale:g} times Earth's" if t.scale != 1.0 else "",
                f"no higher than {t.cap_m:g} m" if t.cap_m is not None else "",
                (f"inside {t.box.south_deg:g} to {t.box.north_deg:g} degrees "
                 f"latitude and {t.box.west_deg:g} to {t.box.east_deg:g} degrees "
                 f"longitude, back to Earth's over {t.box.taper_deg:g} degrees")
                if t.box is not None else "",
            ))))
        if t.scale < 1.0 or t.cap_m is not None:
            out.append(
                "air under removed terrain: temperature continues downward at "
                "6.5 K per km from the lowest analysis level above ground, and "
                "specific humidity is held at that level's value (no water Earth "
                "did not have)")
            out.append(
                "land skin and soil warmed by 6.5 K per km of height removed; "
                "snow removed where the moved skin is above freezing")
        if t.scale > 1.0:
            out.append("surface pressure on raised terrain: integrated upward "
                       "through the analysis's own air")
    if world.changes_terrain or world.is_invented:
        if t.air_mass == "conserve":
            out.append("air mass conserved: surface pressure scaled by one global "
                       "number so the world keeps Earth's amount of air (the "
                       "pressure map then sits uniformly off the control's, and "
                       "difference pictures remove and print that offset)")
        else:
            out.append("air mass filled: the sea-level pressure map is kept, which "
                       "adds air")
    if world.is_invented:
        surface = world.surface
        out.append(
            f"surface from the folder {surface.folder_name!r}: uniform loam soil, "
            f"one land-cover class ({surface.land_cover}), vegetation and snow "
            "albedo from the WRF tables by class, no land ice, no inland lakes, "
            "land found by flooding from the largest ocean")
        out.append(f"starting soil wetness {surface.soil_wetness:g} of saturation, "
                   "as typed")
        out.append("air: Earth's zonal-mean air from the analysis, laid onto the "
                   "world's terrain")
    if sea.land == "none":
        out.append("no land: every column is sea, flat")
    elif sea.level_m != 0.0:
        out.append(f"sea level raised {sea.level_m:g} m: every column at or below "
                   "it becomes sea")
    if sea.temperature == "zonal-mean":
        out.append("sea temperature: the zonal mean of the analysis's own open "
                   "water, by latitude")
    elif sea.temperature != "analysis":
        out.append(f"sea temperature: the published aquaplanet profile "
                   f"{sea.temperature!r} (Neale and Hoskins 2000)")
    if sea.ice == "zonal":
        out.append("sea ice: made zonal")
    elif sea.ice == "none":
        out.append("sea ice: removed")
    if sun.obliquity_deg != EARTH_OBLIQUITY_DEG or sun.orbit != "earth":
        if sun.orbit == "perpetual-equinox":
            out.append("sun: a perpetual equinox, over the equator all year")
        else:
            out.append(f"sun: a {sun.obliquity_deg:g} degree tilt"
                       + (f", eccentricity {sun.eccentricity:g} with closest "
                          f"approach at {sun.perihelion_deg:g} degrees"
                          if sun.orbit == "eccentric" else ""))
    if sun.brightness != 1.0:
        out.append(f"sun: {sun.brightness:g} times the model's own solar constant")
    if world.changes_rotation:
        out.append(f"day: {sun.day_length_h:g} h; the rotation rate is "
                   f"{sun.rotation_factor:g} times Earth's, from the same number as "
                   "the sun's hour angle; the start is re-balanced to it")
    air = world.air
    for gas, unit, value in (("CO2", "ppm", air.co2_ppm), ("CH4", "ppb", air.ch4_ppb),
                             ("N2O", "ppb", air.n2o_ppb)):
        if value != DATED:
            out.append(f"{gas} held at {float(value):g} {unit} for the whole run")
    return tuple(out)


def held(world: WorldSpec) -> tuple[str, ...]:
    """What stays fixed or stays Earth's for the whole run."""
    out = ["the sea surface temperature and sea ice, held at their starting "
           "values for the whole run (no ocean answers the atmosphere)"]
    if not world.is_invented:
        out.append("everything not listed as changed: Earth's analysis and Earth's "
                   "surface")
    if world.changes_sun:
        out.append("ozone, vegetation and the land-use season, on Earth's calendar")
    if world.changes_rotation:
        out.append("time labels on the ordinary calendar; each tape carries the "
                   "sun's true sub-solar longitude and declination")
    if world.sea.level_m != 0.0:
        out.append("Earth's ice sheets")
    out.append("Earth's gravity, radius, air composition and 365-day year")
    return tuple(out)


def result_sentence(world: WorldSpec, duration_s: float | None) -> str:
    """The sentence that goes on every result (plan section 1)."""
    if duration_s is None:
        days = "first N days"
    elif float(duration_s) == 86400.0:
        days = "first day"
    else:
        days = f"first {float(duration_s) / 86400.0:g} days"
    if world.is_invented:
        start = f"This is the weather of the {days} of Earth's air over this map"
    else:
        start = f"This is the weather of the {days} after Earth was changed"
    return (f"{start}, over a sea surface that does not respond. It is not this "
            "world's climate.")


@dataclass(frozen=True)
class WorldStatement:
    """One world's statement.  ``text()`` is what ``world describe``
    prints; ``record()`` what the receipt stores; ``sha256`` what a tape
    carries."""

    world: WorldSpec
    world_class: str
    verdicts: tuple[Verdict, ...]
    changed: tuple[str, ...]
    invented: tuple[str, ...]
    held: tuple[str, ...]
    sentence: str
    report: PreflightReport

    @property
    def refused(self) -> bool:
        return self.report.refused

    def record(self) -> dict[str, object]:
        """The statement as stored: JSON-ready, no clock, no host, no path."""
        return {
            "schema": STATEMENT_SCHEMA,
            "label": self.world.label,
            "identity": self.world.identity_hash,
            "class": self.world_class,
            "verdicts": [{"label": v.label, "setting": v.setting, "limit": v.limit}
                         for v in self.verdicts],
            "changed": list(self.changed),
            "invented": list(self.invented),
            "held": list(self.held),
            "sentence": self.sentence,
            **self.report.as_dict(),
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.record(), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def text(self) -> str:
        w = self.world
        lines = [f"World {w.label!r}: {self.world_class}"]
        if w.is_earth:
            lines.append("Every setting is Earth's: nothing is changed, invented "
                         "or checked, and the run is the model's own Earth.")
            return "\n".join(lines) + "\n"
        lines.append(f"identity {w.identity_hash}")
        lines.append("")
        lines.append(self.sentence)
        sections = (
            ("Label", [f"{v.label}, for {v.setting}: {v.limit}" for v in self.verdicts]),
            ("Changed", list(self.changed)),
            ("Invented, and by which rule", list(self.invented)),
            ("Held", list(self.held)),
        )
        for title, rows in sections:
            lines.append("")
            lines.append(f"{title}:")
            if rows:
                lines.extend(f"  - {row}" for row in rows)
            else:
                lines.append("  (none)")
        lines.append("")
        lines.append("Margins:")
        for m in self.report.margins:
            if m.value is None:
                figure = "not bounded"
            else:
                figure = f"{m.value:,.2f} {m.unit}"
                if m.margin is not None:
                    figure += f", {m.margin:+,.2f} {m.unit} of room"
            lines.append(f"  - {m.title}: {figure} [{m.status}]")
            lines.append(f"      {m.rule}")
        if self.report.prediction is not None:
            p = self.report.prediction
            lines.append("")
            lines.append(f"Start, predicted on the frame grid ({p['grid']}, "
                         f"{p['pressure_rule']}):")
            lines.append(f"  - lowest surface pressure {p['surface_pressure_min_pa']:,.0f} Pa "
                         f"at {p['surface_pressure_min_at']} (Earth's frame "
                         f"{p['earth_surface_pressure_min_pa']:,.0f} Pa)")
            lines.append(f"  - highest surface pressure {p['surface_pressure_max_pa']:,.0f} Pa "
                         f"at {p['surface_pressure_max_at']} (Earth's frame "
                         f"{p['earth_surface_pressure_max_pa']:,.0f} Pa)")
            lines.append(f"  - mean surface pressure {p['world_mean_surface_pressure_pa']:,.0f} Pa "
                         f"(Earth's frame {p['earth_mean_surface_pressure_pa']:,.0f} Pa)")
            lines.append(f"  - highest terrain {p['terrain_max_m']:,.0f} m "
                         f"(Earth's {p['earth_terrain_max_m']:,.0f} m); "
                         f"{p['columns_moved']:,} columns moved")
        for note in self.report.notes:
            lines.append(f"Note: {note}")
        lines.append("")
        if self.report.refusals:
            lines.append("REFUSED before any card time:")
            lines.extend(f"  - {r.key}: {r.text}" for r in self.report.refusals)
        else:
            lines.append("Not refused.")
        return "\n".join(lines) + "\n"


def describe(world: WorldSpec, *, duration_s: float | None = None,
             report: PreflightReport | None = None, **preflight_args) -> WorldStatement:
    """The statement of ``world``.  ``report`` is a preflight already run;
    otherwise the preflight runs here with ``preflight_args`` (``frame``
    and ``levels_pa``, or ``view``; ``vertical``; the frame edit's
    ``edited_terrain_m`` and ``edited_surface_pressure_pa``)."""
    if report is None:
        report = preflight(world, duration_s=duration_s, **preflight_args)
    label = NOT_SUPPORTED_YET if report.refused else world_class(world)
    return WorldStatement(
        world=world,
        world_class=label,
        verdicts=verdicts(world),
        changed=tuple(_changed_rows(world)),
        invented=invented(world),
        held=held(world),
        sentence=result_sentence(world, duration_s),
        report=report,
    )


def describe_config(cfg, **preflight_args) -> WorldStatement:
    """The statement of a loaded run config's world, with its vertical grid
    and run length."""
    preflight_args.setdefault("vertical", cfg.vertical)
    return describe(cfg.world, duration_s=cfg.duration_s, **preflight_args)


def describe_path(config_path, *, read_frame: bool = False,
                  scratch_destination=None) -> WorldStatement:
    """The statement of the world in the run config at ``config_path``:
    what ``world describe`` prints.  With ``read_frame`` the config's
    analysis is decoded (on the CPU, through the engine's decoder) and the
    frame checks and refusals run on it; without, the statement says which
    margins wait for the frame.  Earth never reads its frame."""
    from ..config import load_config

    cfg = load_config(config_path)
    grib = getattr(cfg, "analysis_grib", None)
    if not read_frame or cfg.world.is_earth or not grib:
        return describe_config(cfg)
    frame, levels = decode_analysis_frame(cfg, scratch_destination)
    return describe_config(cfg, frame=frame, levels_pa=levels)


def decode_analysis_frame(cfg, scratch_destination=None):
    """``(frame, levels_pa)`` of the config's analysis, decoded the way the
    cold start decodes it (ERA5 completed by its stated rules)."""
    from pathlib import Path

    import numpy as np

    from ..analysis_initial import analysis_levels_pa, resolve_analysis_mapping
    from ..era5_analysis import complete_era5_frame, is_era5_mapping
    from ..mapped_source_compat import decode_through_engine

    mapping = resolve_analysis_mapping(cfg.analysis_mapping)
    grib = Path(cfg.analysis_grib)
    if not grib.exists():
        raise FileNotFoundError(f"analysis GRIB {grib} does not exist")
    frames = decode_through_engine(mapping, [grib],
                                   scratch_destination=scratch_destination).frames
    if len(frames) != 1:
        raise ValueError(f"a world statement reads one valid time, decoded {len(frames)}")
    frame = frames[0]
    if is_era5_mapping(mapping):
        frame, _completion = complete_era5_frame(frame, grib)
    levels, _unit = analysis_levels_pa(np.asarray(frame.vertical_values, dtype=np.float64),
                                       mapping)
    return frame, levels


def _changed_rows(world: WorldSpec) -> list[str]:
    earth = WorldSpec()
    rows = []
    for key in world.changed_settings():
        part, name = key.split(".", 1)
        mine = getattr(getattr(world, part), name)
        theirs = getattr(getattr(earth, part), name)
        if part == "surface" and name == "folder":
            mine = world.surface.folder_name
        rows.append(f"{key}: {_show(mine)} (Earth: {_show(theirs)})")
    return rows


def _show(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    if hasattr(value, "__dataclass_fields__"):
        return ", ".join(f"{k} {_show(getattr(value, k))}" for k in value.__dataclass_fields__)
    return str(value)


__all__ = [
    "CO2_ILLUSTRATION_RANGE_PPM", "EARTH_CLASS", "ILLUSTRATION", "NOT_SUPPORTED_YET",
    "SCIENCE", "STATEMENT_SCHEMA", "Verdict", "WorldStatement", "decode_analysis_frame",
    "describe", "describe_config", "describe_path", "held", "invented", "result_sentence", "verdicts",
    "world_class",
]
