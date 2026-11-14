"""Render tapes: global lat-lon wrfout files from Arwen Global checkpoints.

The production renderer reads wrfout NetCDF and already draws unrotated
MAP_PROJ=6 global frames (the MPAS bridge's proven path), so weather-field
maps of this model go through a wrfout tape and ``gpuwm render`` rather
than any Python plotting.  ``export_parent`` stays the exact-sampling
artifact door; this door synthesizes on the Gaussian grid and regrids
bilinearly, which is map-pixel accurate and orders of magnitude cheaper
than direct spherical evaluation at every regular-grid point.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import numpy as np

from arwen_global.spectral.sampling import regular_latlon_coordinates
from arwen_global.spectral.vector import VorticityDivergenceOperator
from gpuwm.io.wrfout import WrfoutWriter, wrfout_filename

from .checkpoint import read_checkpoint, state_from_checkpoint
from .longrun_ops import time_mean_attrs
from .constants import (
    CONVECTIVE_RAIN_ACCUMULATOR,
    GRAVITY_M_S2,
    KAPPA,
    LATENT_HEAT_VAPORIZATION,
    REFERENCE_PRESSURE_PA,
    WATER_SPECIES,
)
from .mapped_parent_export import diagnose_interface_w
from .physics.native_runtime import NATIVE_SURFACE_DIAGNOSTICS_SOURCE
from .physics.surface_diagnostics import (
    SIMILARITY_SOURCE,
    SOURCE_METADATA_KEY,
    effective_surface_humidity,
    similarity_surface_diagnostics,
)
from .pins import pins_hash
from .receipt import write_receipt
from .runner import (
    adopt_checkpoint_clock, build_model_and_cold_state, build_transform,
)
from .statics import SURFACE_STATICS_METADATA_KEY, water_columns

_THETA_OFFSET_K = 300.0
_EARTH_RADIUS_RENDER_M = 6_370_000.0
_SOIL_LAYERS = 4
#: Sidecar receipt beside the tapes: which screen-level source each carries.
EXPORT_RECEIPT_NAME = "arwen-global-export-receipt.json"
#: wrfout global attribute stamped with the same label.
SURFACE_DIAGNOSTICS_ATTR = "ARWEN_SURFACE_DIAGNOSTICS"
#: The model the engine's renderer names in each map's metadata row.  A
#: tape imports under the renderer's generic wrfout identity, so without
#: this attribute every global map said "WRF" (the engine reads it from
#: 2.8.1; an older engine ignores it and draws as before).
MODEL_LABEL_ATTR = "GPUWM_MODEL_LABEL"
#: The public name: maps and every sendable call the model WOOF Global, and
#: the standalone package stamped the internal name on every tape, so its
#: published maps would have named the model differently from the WOOF
#: assembly's (which rewrote the literal).
MODEL_LABEL = "WOOF Global"
#: The longest model name the renderer prints.  MEASURED 2026-11-13 with
#: engine 2.9.13's rw_wrfbatch on a T127 world tape: a label of 32
#: characters is printed, one of 33 is dropped and the map says "WRF"
#: instead, and so is any label holding a parenthesis.
RENDERER_MODEL_LABEL_MAX = 32
EXPORT_FALLBACK_SOURCE = "export-similarity-fallback"
SURFACE_DIAGNOSTICS_SOURCES = {
    NATIVE_SURFACE_DIAGNOSTICS_SOURCE: (
        "U10/V10 from the native surface layer (sfclay u10/v10) on every "
        "column; T2/Q2 from sfclay on water columns and from WRF's SFCDIAGS "
        "after Noah on land columns (T2 = TSK - HFX/(rho cp CQS2), Q2 = QSFC "
        "- QFX/(rho CQS2)), persisted in the checkpoint's physics state"
    ),
    SIMILARITY_SOURCE: (
        "T2/Q2/U10/V10 from the reference suite's Monin-Obukhov similarity "
        "diagnostic, persisted in the checkpoint's physics state"
    ),
    EXPORT_FALLBACK_SOURCE: (
        "checkpoint carried no screen-level state; T2/Q2/U10/V10 diagnosed at "
        "export by Monin-Obukhov similarity from the skin and the lowest full "
        "level (never the lowest level itself)"
    ),
}



#: wrfout global attribute naming the water fields' convention (GI-5).
WATER_CONVENTION_ATTR = "ARWEN_WATER_CONVENTION"
#: WRF's QVAPOR, QCLOUD, QRAIN, QICE, QSNOW, QGRAUP and Q2 are DRY MIXING
#: RATIOS, and every WRF-shaped reader of the tape (the renderer's dewpoint
#: and relative humidity, the ABI reference columns) reads them so.  The
#: model carries specific humidity, so the tape converts with the model's
#: own boundary conversion (physics/native_batch).  Writing q under WRF's
#: names read 2 percent dry at q = 0.02, a 0.33 K dewpoint bias (GI-5,
#: audit 2026-10-05).
TAPE_WATER_CONVENTION = (
    "dry mixing ratio (kg per kg of dry air): QVAPOR..QGRAUP = q_x / (1 - q_v) "
    "and Q2 = q2 / (1 - q2) from the model's specific humidities"
)


def tape_water_fields(species, q2):
    """The tape's water species and Q2 as WRF's dry mixing ratios.

    ``species`` maps every water species to the model's specific
    humidity (kg per kg of moist air) and must carry ``"qv"``; ``q2`` is
    the 2 m specific humidity.  Returns ``(species, q2)`` converted.
    """
    from .physics.native_batch import mixing_ratio_from_specific_humidity

    q2 = np.asarray(q2, dtype=np.float64)
    return mixing_ratio_from_specific_humidity(species), q2 / (1.0 - q2)


def _require_tape_writer() -> None:
    """The tape writer, before the first frame is assembled.

    `WrfoutWriter` drives the netcdf-writer cdylib, and on a machine without
    it the engine refuses in its own words with a `cargo build` for a
    checkout no user of a wheel has, at exit 1. A door that is not there is
    exit 3 and the bundle that publishes it, and the check belongs before the
    regridding rather than after several minutes of it.

    An operator who has selected the engine's netCDF4 writer is left alone:
    that route is the engine's own stated workaround, and refusing it here
    would break a road somebody deliberately took.
    """

    import os

    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import WRFOUT_WRITER_ENV

    from .doors import missing_door_refusal

    if os.environ.get(WRFOUT_WRITER_ENV, "rust") != "rust":
        return
    try:
        reason = nc_writer_bridge.unavailable_reason()
    except Exception:  # noqa: BLE001 - an engine whose bridge names it otherwise
        return
    if reason is not None:
        raise missing_door_refusal("netcdf_writer", str(reason))
def _gaussian_to_regular(grid, target_lat, target_lon):
    """Bilinear sampling from the Gaussian grid at regular lat-lon centres.

    ``target_lat`` names the tape's rows and ``target_lon`` its columns in
    the tape's own order.  The sampling runs in the Rust render kernels
    (`arwen_global.render_kernels`); the NumPy version it replaced is the
    test oracle in ``tests/render_kernel_oracle.py``.  The returned callable
    takes the field and an optional ``post`` of ``"floor_zero"`` (the water
    species' clip at zero) or ``"exp"`` (surface pressure from its log).
    """
    from .render_kernels import GaussianRegridder

    return GaussianRegridder(
        grid.latitude_deg, grid.longitude_deg, target_lat, target_lon)


#: wrfout global attribute saying what the tape's W is.
VERTICAL_VELOCITY_ATTR = "ARWEN_VERTICAL_VELOCITY"
#: The tape's ``W`` description.  It is not the regional parent frames':
#: a frame's W is the flux through the coordinate surfaces alone, which is
#: what the regional engine was fed and measured with, and the tape adds
#: the wind's climb along the surfaces to it
#: (:func:`diagnose_ground_relative_w`).
TAPE_W_DESCRIPTION = (
    "vertical velocity of the air relative to the ground at layer "
    "interfaces: the flux through the hybrid coordinate surfaces plus the "
    "wind's climb along them (the rise or fall of the surfaces themselves "
    "at a fixed place is left out; it is zero at the ground)"
)
#: What the tape's ``W`` is, in full: the tape's global attribute
#: (:data:`VERTICAL_VELOCITY_ATTR`) and the export receipt carry these words.
TAPE_VERTICAL_VELOCITY_POLICY = (
    "W is the air's vertical velocity relative to the ground at layer "
    "interfaces, as two of the three parts of Dz/Dt on a hybrid coordinate "
    "surface. The first is the flux through the surface, "
    "-(eta-dot dp/d-eta) / (rho g), from the state's own hybrid-layer mass "
    "continuity: the regional parent frames' W, zero at the model top and "
    "at the surface. The second is the wind's climb along the surface, "
    "V . grad z at constant eta, with z the hydrostatic interface height "
    "(the tape's PH + PHB over g) and V the mean wind of the two layers "
    "the interface separates. At the ground W is the lowest layer's wind "
    "times the terrain slope, and zero over level ground. The third part, "
    "the rate at which the surface itself rises or sinks at a fixed place, "
    "is left out: one checkpoint does not determine it, and it is zero at "
    "the ground"
)
#: What a wrfout post-processor reads beside the state to derive its own
#: fields: tape name, the unit string on the tape, content.  WRF writes all
#: eight in every history file and its readers ask for them by name
#: (absolute vorticity needs F and the two map factors, a land or lake mask
#: needs the land-use planes, anything in the vertical needs W), so a tape
#: without them is read as an unstretched, unrotating, all-land grid or
#: refused.  The unit strings are the engine writer's for these names; the
#: page's table and the tape are both held to this one.
DERIVATION_FIELDS = (
    ("F", "s-1", "Coriolis parameter, 2 Omega sin(latitude), at the rotation "
                 "rate the model integrates with"),
    ("E", "s-1", "the cosine Coriolis term, 2 Omega cos(latitude)"),
    ("MAPFAC_MX", "", "map factor in x: the tape's DX over the distance "
                      "between neighbouring columns, 1 / cos(latitude)"),
    ("MAPFAC_MY", "", "map factor in y: the tape's DY over the distance "
                      "between neighbouring rows, 1 on a grid with as many "
                      "degrees per row as per column"),
    ("LU_INDEX", "", "land-use category of the nearest model column"),
    ("IVGTYP", "", "the same category as an integer, WRF's vegetation type"),
    ("XLAND", "", "1 where the nearest model column runs as land or ice, "
                  "2 where it runs as open water"),
    ("W", "m s-1", TAPE_W_DESCRIPTION),
)
#: The land-use table the categories index, as WRF's global attributes: WRF
#: attribute name, member of the surface state's category convention
#: (``statics.CategoryConvention``).  Written when the checkpoint names its
#: convention; a state that does not is written without them, because a
#: table this door guessed would be read as the one the model ran with.
#: ``ISLAKE`` is the one attribute that is not a member of the convention:
#: see :data:`TAPE_ISLAKE`.
LANDUSE_TABLE_ATTRS = (
    ("MMINLU", "landuse_dataset"), ("ISWATER", "water_category"),
    ("ISLAKE", None), ("ISICE", "ice_category"),
    ("ISURBAN", "urban_category"),
)
#: The tape's ``ISLAKE``.  The convention names the lake class of the
#: land-use set (21 in MODIS with lakes), but no column of the state
#: carries it: the statics build folds the lake class into water, so a
#: lake column runs, and is written, as the water category.  WRF writes -1
#: for a land-use field with no separate lake class, and a reader that
#: masks lakes with ``LU_INDEX == ISLAKE`` then finds none, which is what
#: the planes hold.  The convention's own number here would promise a
#: class that is on no tape.
TAPE_ISLAKE = -1
#: Variable attributes this door states itself.  The engine's writer fills a
#: variable's description from WRF's Registry, where W is "z-wind
#: component" and nothing says which parts of it a tape carries.
TAPE_VARIABLE_ATTRS = {"W": {"description": TAPE_W_DESCRIPTION}}
#: What the tape's ``SST`` is.  WRF writes SST in every history file and its
#: readers ask for it by name; a tape without it is refused by the reader
#: or drawn from another field.  The model's open water has no ocean model:
#: its skin is the water temperature, held at the analysis's or followed
#: along a dated series (``[surface_update]``), and an inland lake's skin
#: is that lake's water.  So SST is the skin temperature, ``TSK``, on every
#: point.  Over land and under sea ice that is what WRF's real.exe writes
#: when its input carries no SST of its own, as the GFS and GDAS tables of
#: WPS do not: it fills SST from TSK on every point, land and ice included.
#: A reader that wants open water only masks with ``XLAND`` or ``SEAICE``,
#: as it must on a WRF file.
TAPE_SST_POLICY = (
    "SST is the skin temperature (TSK) on every point. Over open water and "
    "lakes the skin is the water temperature the model ran with: the "
    "analysis's sea surface held for the run, or a dated series the run "
    "followed. Over land and under sea ice it is the land or ice skin, as "
    "WRF's real.exe fills SST from TSK when its input carries no SST of its "
    "own; mask with XLAND or SEAICE for open water only"
)
#: What ``SST`` is on a tape of a state that carries the slab ocean
#: (``[ocean] mode = "slab"``): the coupler's plane replaces the skin copy
#: (arwen_global.ocean_coupler.tape_fields), and the export receipt names
#: this policy instead of :data:`TAPE_SST_POLICY` when any of its tapes
#: carries the slab.
TAPE_SST_SLAB_POLICY = (
    "SST is the slab ocean's mixed-layer temperature on ocean columns and "
    "the skin temperature (TSK) elsewhere, regridded as TSK is, so a coast "
    "or ice-edge point carries a blend of the two; HML and ARWEN_QFLUX "
    "beside it are the mixed-layer depth and the heat-flux correction"
)
#: The engine's writer class as imported.  ``WrfoutWriter`` is the name the
#: export calls, and a test may rebind it to a stand-in that writes no file.
_ENGINE_TAPE_WRITER = WrfoutWriter

#: The tape's TITLE.  A world's tape names the world and its label in it
#: (:func:`world_tape_title`); an Earth tape keeps exactly these words.
TAPE_TITLE = "Arwen Global research model"

# -- other worlds (plan section 4.10) ----------------------------------------
#
# A tape of a world that is not Earth's carries the world on it: its name,
# its label (science or illustration), its identity and settings, the hash
# of its world statement, whether its coast is its own, and the sun's true
# position.  An Earth tape (no [world] table, or an all-Earth one) carries
# none of these and is the same bytes as before worlds existed: no line
# below runs for it.

#: The world's name as the user wrote it (``[world] label``).
WORLD_ATTR = "ARWEN_WORLD"
#: The tape's label: "science" or "illustration" (plan section 2).  A world
#: that is "not supported yet" is refused before it runs and has no tape.
WORLD_CLASS_ATTR = "ARWEN_WORLD_CLASS"
#: The limit that goes with the label, in words, for this tape's lead time.
WORLD_LIMIT_ATTR = "ARWEN_WORLD_LIMIT"
#: The world's identity: 64 hex, the sha256 of :data:`WORLD_SETTINGS_ATTR`.
WORLD_IDENTITY_ATTR = "ARWEN_WORLD_IDENTITY"
#: The world's canonical settings as compact JSON with sorted keys, the
#: exact bytes the identity hashes.
WORLD_SETTINGS_ATTR = "ARWEN_WORLD_SETTINGS"
#: The dotted names of every setting that differs from Earth's.
WORLD_CHANGED_ATTR = "ARWEN_WORLD_CHANGED"
#: The sha256 of the world statement the run recorded, or
#: :data:`WORLD_STATEMENT_UNRECORDED` when the run recorded none.
WORLD_STATEMENT_ATTR = "ARWEN_WORLD_STATEMENT_SHA256"
WORLD_STATEMENT_UNRECORDED = "not recorded"
#: "own" when the world's land mask is not Earth's (a flood, a waterworld,
#: an invented world): a picture must then draw the coast from this tape's
#: LANDMASK and never Earth's map lines.  "earth" otherwise.
WORLD_COAST_ATTR = "ARWEN_WORLD_COAST"
#: The sentence every result of a world carries (plan section 1).
WORLD_SENTENCE_ATTR = "ARWEN_WORLD_SENTENCE"
#: Where the sun stands at the tape's valid time, in degrees north and
#: east (-180 to 180), and how that was worked out.  The valid time keeps
#: the ordinary calendar on every world (plan 4.6), so on a world with
#: another day length these, not the clock, say where it is day.
SUBSOLAR_LAT_ATTR = "ARWEN_SUBSOLAR_LAT"
SUBSOLAR_LON_ATTR = "ARWEN_SUBSOLAR_LON"
SUN_POSITION_ATTR = "ARWEN_SUN_POSITION"
#: Every attribute a world's tape carries, in the order it is written.  The
#: two sub-solar coordinates are left out (and :data:`SUN_POSITION_ATTR`
#: says why) for an orbit whose timing this door does not have.
WORLD_TAPE_ATTRS = (
    WORLD_ATTR, WORLD_CLASS_ATTR, WORLD_LIMIT_ATTR, WORLD_IDENTITY_ATTR,
    WORLD_SETTINGS_ATTR, WORLD_CHANGED_ATTR, WORLD_STATEMENT_ATTR,
    WORLD_COAST_ATTR, WORLD_SENTENCE_ATTR, SUBSOLAR_LAT_ATTR,
    SUBSOLAR_LON_ATTR, SUN_POSITION_ATTR,
)
#: The obliquity the model's own sun uses (WRF ``radconst``, clock.py).
_EARTH_OBLIQUITY_DEG = 23.5
_EARTH_DAY_S = 86_400.0
#: The window in which Earth with its mountains lowered is science (plan
#: section 2.1): days 2 to 6, near the lowered ranges.
_SCIENCE_WINDOW_S = (2.0 * _EARTH_DAY_S, 6.0 * _EARTH_DAY_S)


def world_settings_json(world) -> str:
    """The world's canonical settings as the exact text its identity is the
    sha256 of (``arwen_global.world.spec.WorldSpec.identity_hash``)."""
    import json

    return json.dumps(world.identity, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def world_tape_class(world, seconds: float) -> tuple[str, str]:
    """The label a world's tape carries at ``seconds`` of lead, and its
    limit in words: plan section 2, read per tape.

    Earth with its mountains lowered (and nothing else changed) is science
    from day 2 to day 6 near the lowered ranges; every other world that runs
    is an illustration, with the limit section 2.2 prints for what it
    changes.  Package 3.5's limits table is the owner of these labels; this
    is the same rule in code until that table exists.
    """
    if world.is_earth:
        raise ValueError("Earth has no world label: its tapes carry none")
    lowered_only = (
        world.changes_terrain and not world.changes_sun
        and not world.changes_sea and not world.changes_air
        and not world.is_invented and world.terrain.scale <= 1.0)
    if lowered_only:
        if seconds < _SCIENCE_WINDOW_S[0]:
            return "illustration", (
                "days 0 to 2 of Earth with its mountains lowered are the air "
                "falling into the space where rock was")
        if seconds <= _SCIENCE_WINDOW_S[1]:
            return "science", (
                "days 2 to 6 of Earth with its mountains lowered, near the "
                "lowered ranges only; anywhere else this tape is an "
                "illustration, and a difference from the control is shown "
                "only where it exceeds the noise floor")
        return "illustration", (
            "after day 6 the change is not separable from chaos without the "
            "noise floor of a perturbed control")
    limits: list[str] = []
    if world.is_invented:
        limits.append(
            "sea temperature is Earth's zonal mean, land cover and soil are "
            "invented by stated rules and the starting soil wetness decides "
            "the continental interior for more than a month; Earth's air over "
            "this map, not this world's past")
    if world.sea.land == "none":
        limits.append(
            "the prescribed sea temperature largely decides the answer; no "
            "ice can grow or melt")
    elif world.sea.level_m != 0.0:
        limits.append(
            "new coastlines under today's weather; the ice sheets whose "
            "melting would raise the sea are still there")
    elif world.changes_sea and not world.is_invented:
        limits.append("the prescribed sea temperature largely decides the answer")
    if world.changes_terrain:
        limits.append(
            "taller mountains than Earth's: the bottom layer thins over them"
            if world.terrain.scale > 1.0 else
            "beyond day 6, or far from the ranges, not separable from chaos "
            "without the noise floor")
    if world.changes_rotation:
        limits.append(
            "an adjustment from a state balanced for Earth's rotation, not "
            "comparable with published equilibrium results")
    if world.sun.obliquity_deg != _EARTH_OBLIQUITY_DEG or world.sun.orbit != "earth":
        limits.append(
            "correct sun geometry; ozone and vegetation stay on Earth's "
            "calendar")
    if world.sun.brightness != 1.0 or world.changes_air:
        limits.append(
            "over a held ocean this is the land and atmosphere's fast "
            "adjustment; a single twin's difference map is mostly chaos")
    co2 = world.air.co2_ppm
    if not isinstance(co2, str) and co2 > 1120.0:
        limits.append("unvalidated above eight times pre-industrial CO2")
    return "illustration", "; ".join(limits) or "an illustration of an altered Earth"


def world_tape_title(world, world_class: str) -> str:
    """The tape's TITLE: :data:`TAPE_TITLE` on Earth, and the world's name
    and label after it on any other world."""
    if world.is_earth:
        return TAPE_TITLE
    return f"{TAPE_TITLE}, world {world.label} ({world_class})"


def world_model_label(world) -> str:
    """The model name a world's tape gives the renderer: the world's name
    after the model's, in a form the renderer prints.  The renderer drops a
    name longer than :data:`RENDERER_MODEL_LABEL_MAX` or holding a
    parenthesis and prints "WRF" in its place, so the tape's class is not
    in it (the class reaches every panel title through ``render``) and a
    long world name is shortened to the model's name alone."""
    for label in (f"{MODEL_LABEL}, world {world.label}", f"{MODEL_LABEL}, {world.label}"):
        if len(label) <= RENDERER_MODEL_LABEL_MAX:
            return label
    return MODEL_LABEL


def world_sentence(world, seconds: float) -> str:
    """Plan section 1's sentence for one tape: weather, not climate."""
    days = f"{seconds / _EARTH_DAY_S:.4g}"
    what = (f"of the invented world {world.label}" if world.is_invented
            else f"after Earth was changed into the world {world.label}")
    return (f"This is the weather of the first {days} days {what}, over a "
            "sea surface that does not respond. It is not this world's climate.")


def subsolar_point(world, clock, valid: datetime, seconds: float):
    """Where the world's sun stands overhead at a tape's valid time:
    ``(latitude_deg, longitude_deg, how)``, or ``(None, None, why)`` for an
    orbit whose timing this door does not have.

    Dated runs take the model's own sun (WRF ``radconst``, as
    :func:`arwen_global.clock.solar_declination_and_equation_of_time` and
    the radiation's cosine of the zenith angle): the sub-solar longitude is
    where the hour angle is zero, ``180 - (60 hour + equation of time) / 4``
    degrees.  An undated run takes the fixture's sun: model time zero at
    00 UTC on an equinox.  The world then changes two things.  Its
    obliquity scales the sine of the declination by sin(obliquity) over
    sin(23.5 degrees), the same position on the same orbit with another
    tilt; a perpetual equinox holds the declination at zero.  Its day length
    turns the sun around the planet faster or slower from the start on:
    ``360 x hours elapsed x (1 / day_length_h - 1 / 24)`` degrees further
    west, so at Earth's day length the longitude is Earth's exactly.
    """
    sun = world.sun
    if sun.orbit == "eccentric":
        return None, None, (
            "not written: the eccentric orbit's timing of the seasons is not "
            "this door's, and a position from Earth's orbit "
            "would be read as the world's")
    if clock is not None and clock.dated:
        from .clock import solar_declination_and_equation_of_time

        instant = valid.replace(tzinfo=timezone.utc)
        declination, equation_min = solar_declination_and_equation_of_time(instant)
        midnight = instant.replace(hour=0, minute=0, second=0, microsecond=0)
        hour = (instant - midnight).total_seconds() / 3600.0
        longitude = 180.0 - (60.0 * hour + equation_min) / 4.0
        how = ("the model's own sun (WRF radconst declination and equation of "
               "time on its 365-day orbit) at the tape's valid time")
    else:
        declination = 0.0
        longitude = 180.0 - 360.0 * (seconds % _EARTH_DAY_S) / _EARTH_DAY_S
        how = ("the undated fixture's sun: model time zero at 00 UTC on an "
               "equinox")
    if sun.orbit == "perpetual-equinox":
        declination = 0.0
        how += ", held at a perpetual equinox"
    elif sun.obliquity_deg != _EARTH_OBLIQUITY_DEG:
        declination = math.asin(
            math.sin(math.radians(sun.obliquity_deg)) * math.sin(declination)
            / math.sin(math.radians(_EARTH_OBLIQUITY_DEG)))
        how += f", tilted to {sun.obliquity_deg:g} degrees"
    if sun.day_length_h != 24.0:
        longitude -= 360.0 * (seconds / 3600.0) * (
            1.0 / sun.day_length_h - 1.0 / 24.0)
        how += (f", on a {sun.day_length_h:g}-hour day since the start; the "
                "valid time keeps Earth's calendar and no longer says where "
                "it is day")
    longitude = (longitude + 180.0) % 360.0 - 180.0
    return math.degrees(declination), longitude, how


def world_tape_attributes(world, *, clock, valid: datetime, seconds: float,
                          statement_sha256: str | None = None) -> dict[str, object]:
    """The global attributes a world's tape carries (:data:`WORLD_TAPE_ATTRS`),
    or an empty dict for Earth, whose tapes carry none."""
    if world.is_earth:
        return {}
    world_class, limit = world_tape_class(world, seconds)
    latitude, longitude, how = subsolar_point(world, clock, valid, seconds)
    attrs: dict[str, object] = {
        WORLD_ATTR: world.label,
        WORLD_CLASS_ATTR: world_class,
        WORLD_LIMIT_ATTR: limit,
        WORLD_IDENTITY_ATTR: world.identity_hash,
        WORLD_SETTINGS_ATTR: world_settings_json(world),
        WORLD_CHANGED_ATTR: ", ".join(world.changed_settings()),
        WORLD_STATEMENT_ATTR: statement_sha256 or WORLD_STATEMENT_UNRECORDED,
        WORLD_COAST_ATTR: "own" if world.changes_coast else "earth",
        WORLD_SENTENCE_ATTR: world_sentence(world, seconds),
    }
    if latitude is not None:
        attrs[SUBSOLAR_LAT_ATTR] = np.float32(latitude)
        attrs[SUBSOLAR_LON_ATTR] = np.float32(longitude)
    attrs[SUN_POSITION_ATTR] = how
    return attrs


def _recorded_statement_sha256(metadata, given: str | None,
                               provenance=None) -> str | None:
    """The sha256 of the world statement: the text the caller gave, else the
    one the cold start recorded in its provenance
    (``provenance["world_statement"]["sha256"]``, package 1.6), else the
    one the checkpoint's world record carries (``statement_sha256``, or the
    ``statement`` text itself), else None."""
    import hashlib

    if given is not None:
        return hashlib.sha256(given.encode("utf-8")).hexdigest()
    stated = provenance.get("world_statement") if isinstance(provenance, dict) else None
    if isinstance(stated, dict):
        digest = stated.get("sha256")
        if isinstance(digest, str) and digest:
            return digest
    record = metadata.get("world") if isinstance(metadata, dict) else None
    if not isinstance(record, dict):
        return None
    digest = record.get("statement_sha256")
    if isinstance(digest, str) and digest:
        return digest
    text = record.get("statement")
    if isinstance(text, str) and text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return None


def _refuse_another_worlds_checkpoint(world, metadata, checkpoint) -> None:
    """A checkpoint whose world record names another world is not written
    onto this world's tape.  A checkpoint with no record (every Earth one,
    and every one written before worlds) is left to the config hash, which
    already carries a non-Earth world's identity."""
    from .world import world_identity_of

    if not isinstance(metadata, dict) or "world" not in metadata:
        return
    stored = world_identity_of(metadata["world"])
    if stored != world.identity_hash:
        raise ValueError(
            f"checkpoint {checkpoint} was integrated in the world "
            f"{stored!r} and this config is the world "
            f"{world.identity_hash!r} ({world.label}); its tape would carry "
            "the wrong world's name, so it is not written")


def _state_variable_attributes(writer, ocean_fields=()) -> None:
    """Put :data:`TAPE_VARIABLE_ATTRS` on the tape's variables.

    A variable's attributes can be written only while the header is open,
    which on the engine's default container ends at the first data byte.
    The export therefore declares every variable when it opens the tape
    (the writer's ``field_schema``) and calls this before the first frame.
    A tape whose variable is not there to label, or refuses the label, is
    not written: W under WRF's own description would be read as the whole
    wind relative to the ground, the part this door leaves out included,
    with no error anywhere.  A stand-in writer that is not the engine's has
    no file to label.
    """
    if not isinstance(writer, _ENGINE_TAPE_WRITER):
        return
    from .ocean_coupler import TAPE_FIELD_ATTRS

    attrs = TAPE_FIELD_ATTRS
    if any(name not in TAPE_FIELD_ATTRS for name in ocean_fields):
        from .ocean_dyn.coupler import TAPE_FIELD_ATTRS as DYNAMIC_ATTRS

        attrs = {**TAPE_FIELD_ATTRS, **DYNAMIC_ATTRS}
    stated = {**TAPE_VARIABLE_ATTRS,
              **{name: attrs[name] for name in ocean_fields}}
    for name, attrs in stated.items():
        try:
            variable = writer.ds.variables[name]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"the render tape's {name} cannot be given this door's own "
                f"description ({attrs.get('description', '')!r}) because the "
                "installed engine's wrfout writer did not declare it before "
                "the first frame, and its own description for that name is "
                "another quantity; the tape is not written rather than "
                "written mislabelled"
            ) from exc
        for key, value in attrs.items():
            variable.setncattr(key, value)


def diagnose_ground_relative_w(model, transform, atmosphere):
    """The air's vertical velocity relative to the ground (m/s) on the
    model's interfaces, top to surface, on the Gaussian grid.

    w = Dz/Dt, and on a coordinate surface z = z(place, eta, time), so

        w = eta-dot dz/d-eta  +  V . grad z (at constant eta)  +  dz/dt (at
            constant eta).

    The first term is the regional parent frames' W
    (:func:`arwen_global.mapped_parent_export.diagnose_interface_w`, taken
    from there unchanged): the flux through the surface over rho g.  The
    second is the wind's climb along the surface, and it is what this
    function adds: the interface heights are the hydrostatic ones the tape
    writes as PHB (the surface geopotential plus R Tv ln(p below / p above)
    for every layer underneath), their gradient is taken spectrally, and
    the wind on an interface is the mean of the two layers it separates
    (the one layer it bounds at the top and at the ground).  At the ground
    z is the model's terrain, so W there is the lowest layer's wind times
    the terrain slope, the kinematic boundary condition, and the flux
    through the ground stays zero.

    The third term is NOT here.  It is the rate at which the coordinate
    surface itself rises or sinks at a fixed place, and one checkpoint does
    not determine it.  It is zero at the ground whatever the flow.
    """
    xp = transform.backend.xp
    through = diagnose_interface_w(model, transform, atmosphere)
    native = model.grid_state(
        atmosphere, only=("u", "v", "virtual_temperature", "ln_ratio"))
    thickness = model.gas_constant * native["virtual_temperature"] * native["ln_ratio"]
    phi_half = xp.empty_like(through)
    phi_half[-1] = model.surface_geopotential
    phi_half[:-1] = model.surface_geopotential + xp.flip(
        xp.cumsum(xp.flip(thickness, axis=0), axis=0), axis=0)
    east, north = transform.gradient(transform.forward(phi_half))
    del phi_half, thickness

    def on_interfaces(wind):
        out = xp.empty_like(through)
        out[0] = wind[0]
        out[-1] = wind[-1]
        out[1:-1] = 0.5 * (wind[:-1] + wind[1:])
        return out

    climb = on_interfaces(native["u"]) * east + on_interfaces(native["v"]) * north
    return through + climb / GRAVITY_M_S2


def _nearest_column(grid, target_lat, target_lon):
    """The nearest model column of every tape point, for a field that must
    not be blended (a category, a flag).

    ``target_lat`` and ``target_lon`` are the tape's rows and columns as
    :func:`_gaussian_to_regular` takes them.  The row is the Gaussian
    latitude nearest in latitude (the last row beyond either end), the
    column the nearest longitude around the ring; a point exactly midway
    takes the southern row and the eastern column.  This is a choice of
    index and no arithmetic on the field, so it is not a render kernel.
    The returned callable picks from a host array whose last two axes are
    the Gaussian grid's.
    """
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)      # south to north
    lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    rows = np.searchsorted(
        0.5 * (lat[:-1] + lat[1:]), np.asarray(target_lat, dtype=np.float64))
    spacing = 360.0 / lon.size
    columns = np.mod(np.floor(np.mod(
        np.asarray(target_lon, dtype=np.float64) - lon[0], 360.0) / spacing + 0.5
    ).astype(np.int64), lon.size)

    def pick(values) -> np.ndarray:
        return np.asarray(values)[..., rows[:, None], columns[None, :]]

    return pick


def grid_metric_planes(xlat, *, rotation_rate_s: float,
                       row_spacing_deg: float,
                       column_spacing_deg: float) -> dict[str, np.ndarray]:
    """F, E and the two map factors of a regular lat-lon tape, float32.

    F and E are 2 Omega sin(latitude) and 2 Omega cos(latitude) at
    ``rotation_rate_s``, the rate the model integrates with, so the tape's
    absolute vorticity is the model's.  A map factor is the grid distance
    the tape declares over the distance on the sphere, WRF's definition.
    The tape declares DX = DY = the spacing of its columns at the equator,
    so MAPFAC_MX is 1 / cos(latitude) and MAPFAC_MY is the column spacing
    over the row spacing in degrees: exactly 1 on the usual grid of twice
    as many columns as rows, which is what WRF's own global lat-lon grid
    carries.
    """
    lat = np.deg2rad(np.asarray(xlat, dtype=np.float64))
    twice_omega = 2.0 * float(rotation_rate_s)
    return {
        "F": (twice_omega * np.sin(lat)).astype(np.float32),
        "E": (twice_omega * np.cos(lat)).astype(np.float32),
        "MAPFAC_MX": (1.0 / np.cos(lat)).astype(np.float32),
        "MAPFAC_MY": np.full(
            lat.shape, float(column_spacing_deg) / float(row_spacing_deg),
            dtype=np.float32),
    }


def land_use_planes(surface, nearest) -> dict[str, np.ndarray]:
    """LU_INDEX, IVGTYP and XLAND from the surface state's own columns.

    ``nearest`` picks the nearest model column of every tape point from a
    Gaussian-grid array (device or host).  The category is the one the land
    surface ran with, never a blend of two.  XLAND is the model's own
    land/water rule (:func:`arwen_global.statics.water_columns`) on that
    same column: 2 where the column runs as open water, 1 where it runs as
    land or, at half a cover of sea ice or more, as ice, as WRF sets it.
    The water category sits on exactly the columns that rule names (the
    native runtime refuses a state where it does not), so LU_INDEX and
    XLAND agree at every tape point.  LANDMASK is the bilinear land
    fraction at one half and may differ from both along a coast and under
    sea ice.
    """
    category = np.rint(nearest(surface.landuse_category))
    water = water_columns(
        nearest(surface.land_fraction),
        sea_ice_fraction=nearest(surface.sea_ice_fraction))
    return {
        "LU_INDEX": category.astype(np.float32),
        "IVGTYP": category.astype(np.int32),   # integer in WRF's Registry
        "XLAND": np.where(water, 2.0, 1.0).astype(np.float32),
    }


def landuse_table(bundle) -> dict[str, object] | None:
    """The land-use table the state's categories index, as WRF's global
    attributes (:data:`LANDUSE_TABLE_ATTRS`), or None for a state whose
    physics metadata does not name one."""
    row = bundle.physics_state.metadata.get(SURFACE_STATICS_METADATA_KEY)
    if not isinstance(row, dict):
        return None
    if any(member is not None and member not in row
           for _attr, member in LANDUSE_TABLE_ATTRS):
        return None
    return {
        attr: (TAPE_ISLAKE if member is None
               else str(row[member]) if attr == "MMINLU" else int(row[member]))
        for attr, member in LANDUSE_TABLE_ATTRS
    }


def _stagger_x_periodic(field: np.ndarray) -> np.ndarray:
    nx = field.shape[-1]
    out = np.empty((*field.shape[:-1], nx + 1), dtype=field.dtype)
    out[..., 1:nx] = 0.5 * (field[..., :-1] + field[..., 1:])
    out[..., 0] = 0.5 * (field[..., -1] + field[..., 0])
    out[..., nx] = out[..., 0]
    return out


def _stagger_x_clamped(field: np.ndarray) -> np.ndarray:
    """Edge-clamped x stagger for a longitude subset (no wrap column)."""
    nx = field.shape[-1]
    out = np.empty((*field.shape[:-1], nx + 1), dtype=field.dtype)
    out[..., 1:nx] = 0.5 * (field[..., :-1] + field[..., 1:])
    out[..., 0] = field[..., 0]
    out[..., nx] = field[..., -1]
    return out


def _stagger_y(field: np.ndarray) -> np.ndarray:
    ny = field.shape[-2]
    out = np.empty((*field.shape[:-2], ny + 1, field.shape[-1]), dtype=field.dtype)
    out[..., 1:ny, :] = 0.5 * (field[..., :-1, :] + field[..., 1:, :])
    out[..., 0, :] = field[..., 0, :]
    out[..., ny, :] = field[..., -1, :]
    return out


#: Surface energy and land-state planes of the native suite, in WRF's
#: history names, exported when the checkpoint's physics state carries
#: them (a state without them, the reference suite or a cold start, writes
#: none; the receipt lists which tapes carry them).  LH is Noah's energy
#: flux on land and XLV * QFX on water, where the surface layer's is the
#: only one; GRDFLX keeps WRF's sign (into the surface from the soil).
SURFACE_ENERGY_FIELDS = (
    ("HFX", "hfx"), ("QFX", "qfx"), ("GRDFLX", "noah_grdflx"),
    ("SWDOWN", "swdown"), ("GLW", "glw"), ("OLR", "olr"),
    ("UST", "ust"), ("ZNT", "znt"), ("PBLH", "pblh"),
)
SURFACE_STATE_FIELDS = (
    ("ALBEDO", "albedo"), ("EMISS", "emissivity"), ("VEGFRA", "vegetation_fraction"),
    ("LAI", "leaf_area_index"),
)


def _surface_energy_planes(bundle, regrid) -> dict[str, np.ndarray]:
    arrays = bundle.physics_state.arrays
    if not all(name in arrays for _tape, name in SURFACE_ENERGY_FIELDS):
        return {}
    surface = bundle.surface
    planes = {tape: regrid(arrays[name]).astype(np.float32) for tape, name in SURFACE_ENERGY_FIELDS}
    for tape, name in SURFACE_STATE_FIELDS:
        value = getattr(surface, name, None)
        if value is not None:
            planes[tape] = regrid(value).astype(np.float32)
    if "noah_lh" in arrays:
        land = regrid(surface.land_fraction) >= 0.5
        lh_land = regrid(arrays["noah_lh"])
        lh_water = LATENT_HEAT_VAPORIZATION * regrid(arrays["qfx"])
        planes["LH"] = np.where(land, lh_land, lh_water).astype(np.float32)
    if "VEGFRA" in planes:
        planes["VEGFRA"] = (100.0 * planes["VEGFRA"]).astype(np.float32)   # percent, as WRF
    return planes


def parse_start_date(start_date: str | None) -> datetime | None:
    """``--start-date`` (``YYYY-MM-DD_HH:MM:SS``, UTC) as an aware instant,
    or None when it was not given."""
    if start_date is None or not str(start_date).strip():
        return None
    try:
        start = datetime.strptime(str(start_date).strip(), "%Y-%m-%d_%H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            "start-date must be YYYY-MM-DD_HH:MM:SS (the analysis valid time)"
        ) from exc
    return start.replace(tzinfo=timezone.utc)


def tape_start(clock, given: datetime | None) -> datetime:
    """The instant every tape's model time is offset from (naive UTC, the
    tape convention).

    A dated run (an analysis start, or a config that states
    ``start_time_utc``) IS its own start: ``given`` may be left out, and a
    ``given`` that disagrees is refused by name, because every tape's
    Times, START_DATE and file name, and every map drawn from it, would be
    stamped from an instant the forecast did not start at.  An undated run
    (the idealized fixture) has no instant to take, so ``given`` is then
    required and labels the tapes."""
    from .clock import ClockMismatchError, CONFIG_START_TIME, mismatch_sentence

    if clock is not None and clock.dated:
        if given is not None and given != clock.start_utc:
            name = (
                "the config's physics start_time_utc"
                if clock.source == CONFIG_START_TIME
                else "the analysis valid time"
            )
            raise ClockMismatchError(
                mismatch_sentence("--start-date", given, clock.start_utc, name)
                + "; leave --start-date out to stamp the tapes from the run's "
                "own clock"
            )
        return clock.start_utc.replace(tzinfo=None)
    if given is None:
        source = "unstated" if clock is None else clock.source
        raise ValueError(
            f"this run's clock is {source}: it starts from no analysis and "
            "states no physics start_time_utc, so its checkpoints carry no "
            "instant a tape's valid time could be taken from.  Pass "
            "--start-date to label the tapes"
        )
    return given.replace(tzinfo=None)


#: The six fields a tape synthesizes from the spectral state on the model's
#: own device; every other value on a tape is a copy of a state array the
#: checkpoint carries, regridded on the host.
SYNTHESIZED_FIELDS = ("theta", "log_surface_pressure", "u", "v", "qv", "w")


class _DeviceFields:
    """The synthesized fields of one state, made on the device when asked.

    The export's own road: each field is synthesized where the state lives
    and copied to the host by the regrid that reads it, in the order the
    frame asks for them, so no more than one synthesized field (two for
    the wind) is on the device at a time, as before the frame was split.
    """

    def __init__(self, tape_grid, atmosphere):
        self._grid = tape_grid
        self._atmosphere = atmosphere
        self._wind = None

    def __getitem__(self, name):
        grid, atmosphere = self._grid, self._atmosphere
        if name == "theta":
            return grid.transform.inverse(atmosphere.theta)
        if name == "log_surface_pressure":
            return grid.transform.inverse(atmosphere.log_surface_pressure)
        if name in ("u", "v"):
            if self._wind is None:
                self._wind = grid.vector.wind_from_vordiv(
                    atmosphere.vorticity, atmosphere.divergence)
            u_grid, v_grid = self._wind
            if name == "v":
                self._wind = None
            return u_grid if name == "u" else v_grid
        if name == "qv":
            return grid.transform.inverse(atmosphere.qv)
        if name == "w":
            return diagnose_ground_relative_w(grid.model, grid.transform, atmosphere)
        raise KeyError(name)


class TapeGrid:
    """What every tape of one export, or of one run, shares: the regular
    grid, its sampler and nearest-column pick, the metric planes, the
    terrain, the global attributes and the valid-time origin.

    A tape is then two halves.  :meth:`synthesize` is the device half: the
    six fields the spectral state has to be synthesized for
    (:data:`SYNTHESIZED_FIELDS`), on the device that holds the state, copied
    to the host.  :meth:`write` is the host half: the regrid of those and of
    every state array the checkpoint carries, the Rust column kernels, the
    frame and the netCDF write, all on host arrays.  ``export`` runs both
    halves in one thread from a checkpoint it reads; a run given ``--tapes``
    runs the device half on the model thread at an output time and the
    host half on a worker beside the forecast, from the same arrays its
    checkpoint is written from.  Both call the same functions on the same
    values, so a tape the run wrote and the export of the checkpoint the
    run wrote at that time are the same bytes.
    """

    def __init__(self, cfg, model, transform, *, nlat: int, nlon: int,
                 start_date: str | None = None,
                 bbox: tuple[float, float, float, float] | None = None,
                 world_statement: str | None = None):
        from . import render_kernels
        from .world import EARTH

        given = parse_start_date(start_date)
        render_kernels.library()
        self.cfg = cfg
        # A config whose world is not Earth's stamps it on every tape and
        # in the receipt (export_wrfout's docstring); an Earth config writes
        # exactly the tapes it wrote before worlds existed.
        self.world = getattr(cfg, "world", None) or EARTH
        self.world_statement = world_statement
        self.model = model
        self.transform = transform
        self.render_kernels = render_kernels
        self.clock = model.forecast_clock
        self.start = tape_start(self.clock, given)
        self.backend = backend = transform.backend
        self.vector = VorticityDivergenceOperator(transform)
        self.nlat, self.nlon, self.bbox = int(nlat), int(nlon), bbox

        lat, lon = regular_latlon_coordinates(nlat, nlon, include_poles=False)
        # The renderer's proven global frames run -180..180-dlon like the MPAS
        # bridge; roll the ring so the wrap column is not duplicated.
        lon_signed = np.where(lon >= 180.0, lon - 360.0, lon)
        order = np.argsort(lon_signed)
        lon_out = lon_signed[order]
        # Each tape column is sampled at its own unsigned longitude, in the
        # tape's rolled order, and a bbox window samples only its own rows and
        # columns: every value depends on its own row and column alone, so this
        # is the whole-ring sample rolled and cut, without the whole ring.
        lon_sample = lon[order]

        lat_sel = lon_sel = None
        if bbox is not None:
            lat_min, lat_max, lon_min, lon_max = (float(v) for v in bbox)
            if not (lat_min < lat_max and lon_min < lon_max):
                raise ValueError(
                    "bbox must be (lat_min, lat_max, lon_min, lon_max) with "
                    "min < max on both axes"
                )
            lat_sel = np.where((lat >= lat_min) & (lat <= lat_max))[0]
            lon_sel = np.where((lon_out >= lon_min) & (lon_out <= lon_max))[0]
            if lat_sel.size < 2 or lon_sel.size < 2:
                raise ValueError(
                    f"bbox {bbox} selects {lat_sel.size}x{lon_sel.size} points; "
                    "a render tape needs at least 2x2"
                )
            lat = lat[lat_sel]
            lon_out = lon_out[lon_sel]
            lon_sample = lon_sample[lon_sel]
        sampler = _gaussian_to_regular(transform.grid, lat, lon_sample)

        def regrid(values, post: str = "none") -> np.ndarray:
            return sampler(_host_array(backend, values), post)

        nearest_pick = _nearest_column(transform.grid, lat, lon_sample)

        def nearest(values) -> np.ndarray:
            return nearest_pick(_host_array(backend, values))

        self.regrid, self.nearest = regrid, nearest
        xlong, xlat = np.meshgrid(lon_out, lat)
        self.xlong, self.xlat = xlong, xlat
        self.ny_out, self.nx_out = xlat.shape
        # The same in every tape of this export: they are the grid's, not the
        # state's.
        self.metric = grid_metric_planes(
            xlat, rotation_rate_s=model.rotation_rate_s,
            row_spacing_deg=180.0 / nlat, column_spacing_deg=360.0 / nlon)
        self.stagger_x = _stagger_x_periodic if bbox is None else _stagger_x_clamped
        self.dx_m = (360.0 / nlon) * math.pi / 180.0 * _EARTH_RADIUS_RENDER_M
        self.a_half = np.asarray(cfg.a_half_pa, dtype=np.float64)
        self.b_half = np.asarray(cfg.b_half, dtype=np.float64)
        self.nz = self.a_half.size - 1
        self.phi_surface = regrid(model.surface_geopotential)
        self.terrain_m = self.phi_surface / GRAVITY_M_S2
        start = self.start
        self.global_attrs = {
            "START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
            "SIMULATION_START_DATE": start.strftime("%Y-%m-%d_%H:%M:%S"),
            "MAP_PROJ": np.int32(6),
            "MAP_PROJ_CHAR": "Cylindrical Equidistant",
            "POLE_LAT": np.float32(90.0),
            "POLE_LON": np.float32(0.0),
            "TRUELAT1": np.float32(0.0),
            "TRUELAT2": np.float32(0.0),
            "STAND_LON": np.float32(0.0),
            "CEN_LON": np.float32(float(lon_out.mean())),
            "CEN_LAT": np.float32(float(lat.mean()) if bbox is not None else 0.0),
            "GRID_ID": np.int32(1),
            "PARENT_ID": np.int32(0),
            "DT": np.float32(cfg.dt_s),
            MODEL_LABEL_ATTR: MODEL_LABEL,
        }

    def device_fields(self, atmosphere) -> _DeviceFields:
        """The synthesized fields of ``atmosphere``, made when read (the
        export's road)."""
        return _DeviceFields(self, atmosphere)

    def synthesize(self, atmosphere) -> dict[str, np.ndarray]:
        """The device half of a tape: the six synthesized fields of
        ``atmosphere`` on the host, made one at a time on the device and
        copied off it before the next is made.  The state is only read."""
        lazy = self.device_fields(atmosphere)
        to_numpy = self.backend.to_numpy
        out = {}
        for name in SYNTHESIZED_FIELDS:
            value = lazy[name]
            out[name] = np.asarray(to_numpy(value))
            del value
        return out

    def write(self, bundle, fields, seconds: float, outdir: str | Path, *,
              overwrite: bool = False, checkpoint=None, extra_planes=None,
              metadata=None):
        """The host half of a tape: one wrfout file for ``bundle`` at model
        time ``seconds`` into ``outdir``, the synthesized fields read from
        ``fields`` (:meth:`synthesize`'s dict, or :meth:`device_fields`).
        ``metadata`` is the checkpoint's, which a world's tape reads its
        statement hash from when neither the caller nor the cold start
        gave one (:func:`_recorded_statement_sha256`).  Returns
        ``(path, row)``, the row the export receipt lists."""
        seconds = float(seconds)
        if abs(seconds - round(seconds)) > 1.0e-6:
            raise ValueError(
                f"checkpoint time {seconds} s is not a whole second"
            )
        valid = self.start + timedelta(seconds=round(seconds))
        cfg, model = self.cfg, self.model
        regrid, nearest = self.regrid, self.nearest
        render_kernels = self.render_kernels
        a_half, b_half, nz = self.a_half, self.b_half, self.nz
        ny_out, nx_out = self.ny_out, self.nx_out
        atmosphere = bundle.atmosphere

        theta = regrid(fields["theta"])
        ps = regrid(fields["log_surface_pressure"], "exp")
        u = regrid(fields["u"])
        v = regrid(fields["v"])
        # Vapor is synthesized from its coefficients; the condensate
        # species are grid fields already.
        species = {
            name: regrid(
                fields["qv"] if name == "qv" else getattr(atmosphere, name),
                "floor_zero",
            )
            for name in WATER_SPECIES
        }

        # The tape's column: hybrid pressure, temperature and virtual
        # temperature, and the hydrostatic geopotential integrated up from
        # the terrain, all in the Rust render kernels.
        p_half, p_full = render_kernels.hybrid_pressure(a_half, b_half, ps)
        temperature, virtual = render_kernels.virtual_temperature(
            theta, species, p_full,
            reference_pressure_pa=REFERENCE_PRESSURE_PA, kappa=KAPPA)
        phi_half = render_kernels.hydrostatic(
            virtual, p_half, self.phi_surface, gas_constant=model.gas_constant)

        def up(field: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(field[::-1]).astype(np.float32)

        surface = bundle.surface
        screen, screen_source = _screen_level_fields(
            cfg, bundle, self.backend, regrid, temperature, species, u, v, p_half, p_full
        )
        # WRF's moisture names carry WRF's convention (GI-5); the virtual
        # temperature and the screen fallback above use the model's q.
        tape_species, tape_q2 = tape_water_fields(species, screen["q2"])
        rain = regrid(surface.accumulated_rain_kg_m2)
        snow = regrid(surface.accumulated_snow_kg_m2)
        graupel = regrid(surface.accumulated_graupel_kg_m2)
        convective = _convective_rain(bundle, regrid, (ny_out, nx_out))
        land = regrid(surface.land_fraction)
        energy = _surface_energy_planes(bundle, regrid)
        ocean = _slab_ocean_planes(bundle, regrid)
        land_use = land_use_planes(surface, nearest)
        skin = regrid(surface.temperature_k).astype(np.float32)
        table = landuse_table(bundle)
        extras = {}
        if extra_planes is not None:
            for name, value in (extra_planes(checkpoint, bundle, regrid) or {}).items():
                plane = np.asarray(value, dtype=np.float64)
                if plane.shape != (ny_out, nx_out):
                    plane = regrid(plane)
                extras[str(name)] = plane.astype(np.float32)
        frame = {
            "XLAT": self.xlat.astype(np.float32),
            "XLONG": self.xlong.astype(np.float32),
            "T": up(theta - _THETA_OFFSET_K),
            "P": np.zeros((nz, ny_out, nx_out), dtype=np.float32),
            "PB": up(p_full),
            "PH": np.zeros((nz + 1, ny_out, nx_out), dtype=np.float32),
            "PHB": up(phi_half),
            "QVAPOR": up(tape_species["qv"]),
            "QCLOUD": up(tape_species["qc"]),
            "QRAIN": up(tape_species["qr"]),
            "QICE": up(tape_species["qi"]),
            "QSNOW": up(tape_species["qs"]),
            "QGRAUP": up(tape_species["qg"]),
            "U": self.stagger_x(up(u)),
            "V": _stagger_y(up(v)),
            # Screen-level diagnostics: the physics suite's own 2 m / 10 m
            # fields when the checkpoint carries them, else the similarity
            # diagnostic from the skin and the lowest level; never the
            # lowest level itself (audit 2026-09-01, task 1b).  The source
            # is stamped on the tape and in the export receipt.
            "T2": screen["t2"].astype(np.float32),
            "Q2": tape_q2.astype(np.float32),
            "U10": screen["u10"].astype(np.float32),
            "V10": screen["v10"].astype(np.float32),
            "PSFC": ps.astype(np.float32),
            "TSK": skin,
            # The skin under WRF's SST name on every point (TAPE_SST_POLICY);
            # a slab state's own SST replaces it below, in this position
            # (TAPE_SST_SLAB_POLICY).
            "SST": skin.copy(),
            "HGT": self.terrain_m.astype(np.float32),
            "LANDMASK": (land >= 0.5).astype(np.float32),
            "SINALPHA": np.zeros((ny_out, nx_out), dtype=np.float32),
            "COSALPHA": np.ones((ny_out, nx_out), dtype=np.float32),
            # What a post-processor derives its own fields from
            # (DERIVATION_FIELDS).  W is the regional parent frames' W plus
            # the wind's climb along the coordinate surfaces, and its
            # description on the tape says so (TAPE_VARIABLE_ATTRS).
            **self.metric,
            **land_use,
            "W": up(regrid(fields["w"])),
            # WRF's split: RAINC is the cumulus scheme's accumulator, RAINNC
            # the microphysics accumulators.
            "RAINC": convective.astype(np.float32),
            "RAINNC": (rain + snow + graupel).astype(np.float32),
            "SNOWNC": snow.astype(np.float32),
            "GRAUPELNC": graupel.astype(np.float32),
            "TSLB": regrid(surface.soil_temperature_k)[:_SOIL_LAYERS].astype(np.float32),
            "SMOIS": regrid(surface.soil_water_fraction)[:_SOIL_LAYERS].astype(np.float32),
            **energy,
            **extras,
            # The seeded surface under WRF's names: SEAICE and ICEDEPTH from
            # the surface state, SNOW (kg/m2), SNOWH (m) and SNOWC from
            # Noah's store when the checkpoint carries it (zero planes on a
            # state without a native physics namespace).
            "SEAICE": regrid(surface.sea_ice_fraction).astype(np.float32),
            "ICEDEPTH": regrid(surface.sea_ice_thickness_m).astype(np.float32),
            **_snow_planes(bundle, regrid, (ny_out, nx_out)),
            # The slab ocean's SST, mixed-layer depth and heat-flux
            # correction: only on a state that carries the slab, so every
            # other tape is the one written before, byte for byte.
            **ocean,
        }

        _require_tape_writer()
        path = Path(outdir) / wrfout_filename(valid)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"render tape {path} exists; pass --overwrite to replace it"
            )
        tape_attrs = {
            **self.global_attrs, SURFACE_DIAGNOSTICS_ATTR: screen_source,
            # A time-mean file (arwen_global.longrun_ops) says so on
            # its tape; an instantaneous state adds nothing here.
            **time_mean_attrs(
                {"physics_metadata": bundle.physics_state.metadata}),
            WATER_CONVENTION_ATTR: TAPE_WATER_CONVENTION,
            VERTICAL_VELOCITY_ATTR: TAPE_VERTICAL_VELOCITY_POLICY,
            **{
                name: (value if isinstance(value, str) else np.int32(value))
                for name, value in (table or {}).items()
            },
        }
        title = TAPE_TITLE
        world_row = None
        world = self.world
        if not world.is_earth:
            # The world on the tape: its attributes after every Earth one,
            # its name and label in the TITLE and in the model name the
            # renderer prints on every map.
            world_attrs = world_tape_attributes(
                world, clock=self.clock, valid=valid,
                seconds=float(round(seconds)),
                statement_sha256=_recorded_statement_sha256(
                    metadata, self.world_statement,
                    getattr(model, "initial_provenance", None)))
            world_class = str(world_attrs[WORLD_CLASS_ATTR])
            title = world_tape_title(world, world_class)
            tape_attrs[MODEL_LABEL_ATTR] = world_model_label(world)
            tape_attrs.update(world_attrs)
            world_row = {
                "class": world_class,
                "statement_sha256": world_attrs[WORLD_STATEMENT_ATTR],
                "subsolar_lat_deg": (
                    float(world_attrs[SUBSOLAR_LAT_ATTR])
                    if SUBSOLAR_LAT_ATTR in world_attrs else None),
                "subsolar_lon_deg": (
                    float(world_attrs[SUBSOLAR_LON_ATTR])
                    if SUBSOLAR_LON_ATTR in world_attrs else None),
            }
        writer = WrfoutWriter(
            path,
            nx=nx_out,
            ny=ny_out,
            nz=nz,
            dx=self.dx_m,
            dy=self.dx_m,
            title=title,
            global_attrs=tape_attrs,
            soil_layers=_SOIL_LAYERS,
            # Every variable is declared as the tape is opened, so the
            # attributes this door states itself go on before the first
            # frame closes the header (_state_variable_attributes).
            field_schema=frame,
        )
        try:
            _state_variable_attributes(writer, sorted(ocean))
            writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), frame)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        row = {
            "tape": str(path),
            "checkpoint": str(checkpoint),
            "valid": valid.strftime("%Y-%m-%d_%H:%M:%S"),
            "surface_diagnostics": screen_source,
            "surface_energy_fields": sorted(energy),
            **({"slab_ocean_fields": sorted(ocean)} if ocean else {}),
            "extra_planes": sorted(extras),
            "landuse_table": table,
        }
        if world_row is not None:
            row["world"] = world_row
        return path, row

    def receipt(self, tapes) -> dict[str, object]:
        """The export receipt for ``tapes`` (the rows :meth:`write` returned)."""
        cfg, clock, start = self.cfg, self.clock, self.start
        receipt = {
            "name": cfg.name,
            "config_hash": cfg.config_hash,
            "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
            "physics_mode": cfg.physics_mode,
            "vertical": {"coordinate": cfg.vertical_coordinate, **cfg.vertical.describe()},
            "screen_level_sources": SURFACE_DIAGNOSTICS_SOURCES,
            # Where every tape's valid time came from (arwen_global.clock).
            "start_date": start.strftime("%Y-%m-%d_%H:%M:%S"),
            "start_date_source": (
                clock.source if clock is not None and clock.dated
                else "start-date argument"
            ),
            "water_convention": TAPE_WATER_CONVENTION,
            # What every tape carries beside the state for a post-processor,
            # and what its W is.
            "derivation_fields": [name for name, _units, _text in DERIVATION_FIELDS],
            "vertical_velocity": TAPE_VERTICAL_VELOCITY_POLICY,
            "sea_surface_temperature": (
                TAPE_SST_SLAB_POLICY
                if any(row.get("slab_ocean_fields") for row in tapes)
                else TAPE_SST_POLICY),
            "tapes": list(tapes),
        }
        world = self.world
        if not world.is_earth:
            # Only a world's receipt gains the key: an Earth receipt is the
            # one written before worlds existed.
            receipt["world"] = {
                **world.record(),
                "coast": "own" if world.changes_coast else "earth",
                "tape_attributes": list(WORLD_TAPE_ATTRS),
            }
        return receipt


def _host_array(backend, values) -> np.ndarray:
    """``values`` on the host: a host array as it is, a device array (or a
    pinned host-tier slot) through the backend's own copy."""
    if isinstance(values, np.ndarray):
        return values
    return backend.to_numpy(values)


def export_wrfout(
    cfg,
    checkpoints,
    outdir: str | Path,
    *,
    nlat: int,
    nlon: int,
    start_date: str | None = None,
    overwrite: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    extra_planes=None,
    world_statement: str | None = None,
) -> list[Path]:
    """Write one wrfout tape per checkpoint on a regular ``nlat`` x ``nlon``
    grid (or its ``bbox`` window).  ``extra_planes(checkpoint, bundle,
    regrid)`` may return further 2-D planes on the Gaussian grid or already
    regridded (a plane of the tape's shape is taken as it is) to store under
    their own names: the renderer draws any stored 2-D variable as a
    ``var:<name>`` product, which is how the surface-energy instrument's
    bias maps reach it.

    The tapes' valid times are the run's forecast clock plus each
    checkpoint's model time (:func:`tape_start`): ``start_date`` may be
    left out of a dated run and is refused when it disagrees.

    A config whose world is not Earth's (``cfg.world``) stamps the world on
    every tape (:func:`world_tape_attributes`), in its TITLE and in the
    model name the renderer prints, and records it in the export receipt.
    ``world_statement`` is the world statement's text when the caller has
    it; otherwise its hash is taken from the cold start's provenance or the
    checkpoint's world record (:func:`_recorded_statement_sha256`).  An
    Earth config writes exactly the tapes and the receipt it wrote before
    worlds existed."""
    from .world import EARTH

    world = getattr(cfg, "world", None) or EARTH
    parse_start_date(start_date)
    # The render kernels before anything else: a machine without them
    # refuses here, with the missing-door exit and the command that stages
    # them, rather than after the model is built.
    from . import render_kernels

    render_kernels.library()
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=Path(outdir))
    tape_grid = TapeGrid(cfg, model, transform, nlat=nlat, nlon=nlon,
                         start_date=start_date, bbox=bbox,
                         world_statement=world_statement)
    backend = transform.backend

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tapes: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        metadata, arrays = read_checkpoint(
            checkpoint,
            expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        if not world.is_earth:
            _refuse_another_worlds_checkpoint(world, metadata, checkpoint)
        bundle = state_from_checkpoint(metadata, arrays, backend)
        # A checkpoint integrated from another start than this config's
        # clock is refused by name rather than stamped with the wrong
        # valid times.
        adopt_checkpoint_clock(model, bundle, f"checkpoint {checkpoint}")
        model.enforce(bundle)
        path, row = tape_grid.write(
            bundle, tape_grid.device_fields(bundle.atmosphere),
            float(metadata["time_s"]), output, overwrite=overwrite,
            checkpoint=checkpoint, extra_planes=extra_planes,
            metadata=metadata,
        )
        written.append(path)
        tapes.append(row)
    write_receipt(output / EXPORT_RECEIPT_NAME, tape_grid.receipt(tapes))
    return written


def _slab_ocean_planes(bundle, regrid) -> dict[str, np.ndarray]:
    """SST, HML and ARWEN_QFLUX on the tape when the state carries the
    slab ocean (arwen_global.ocean_coupler.tape_fields), else nothing.

    SST is regridded from the mixed-layer temperature on the ocean columns
    and the skin elsewhere, so a coast's bilinear sample blends the two as
    TSK's does."""
    from .ocean_coupler import tape_fields

    fields = tape_fields(bundle.physics_state.arrays, bundle.surface.temperature_k, None)
    if fields and any(name.startswith("ocean_dyn_tape_") for name in bundle.physics_state.arrays):
        # The dynamic ocean's planes (SSH, currents, KPP depth, D20, OHC300),
        # present only under [ocean] mode = "dynamic".
        from .ocean_dyn.coupler import tape_fields as dynamic_tape_fields

        fields.update(dynamic_tape_fields(bundle.physics_state.arrays))
    return {name: regrid(value).astype(np.float32) for name, value in fields.items()}


def _snow_planes(bundle, regrid, shape):
    """Noah's snow store, depth and cover flag on the tape (WRF SNOW,
    SNOWH, SNOWC), zero where the physics namespace has none."""
    arrays = bundle.physics_state.arrays
    out = {}
    for tape_name, store in (("SNOW", "noah_snow"), ("SNOWH", "noah_snowh"), ("SNOWC", "noah_snowc")):
        value = arrays.get(store)
        if value is None:
            out[tape_name] = np.zeros(shape, dtype=np.float32)
        else:
            out[tape_name] = regrid(value).astype(np.float32)
    return out


def _convective_rain(bundle, regrid, shape):
    """Accumulated convective precipitation (WRF RAINC) for one tape.

    The cumulus scheme books it in the physics state under
    CONVECTIVE_RAIN_ACCUMULATOR (native runtime _cumulus_step); a state
    without the array - physics mode "none", or a suite with no cumulus
    scheme - exports zeros, WRF's own convention for a run without one.
    """
    value = bundle.physics_state.arrays.get(CONVECTIVE_RAIN_ACCUMULATOR)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(regrid(value), dtype=np.float64)


def _screen_level_fields(cfg, bundle, backend, regrid, temperature, species, u, v, p_half, p_full):
    """T2/Q2/U10/V10 for one tape and the label naming where they came from."""
    arrays = bundle.physics_state.arrays
    if all(name in arrays for name in ("t2", "q2", "u10", "v10")):
        source = str(bundle.physics_state.metadata.get(SOURCE_METADATA_KEY, "physics-state"))
        return {
            name: regrid(arrays[name]) for name in ("t2", "q2", "u10", "v10")
        }, source
    # Fallback: the checkpoint carries no screen-level state (physics mode
    # "none", or a tape written before the suites persisted them), so the
    # similarity diagnostic runs here on the regridded skin and lowest level.
    surface = bundle.surface
    land = regrid(surface.land_fraction)
    wetness = (
        regrid(surface.soil_water_fraction)[0]
        / cfg.reference_physics.soil_wetness_capacity
    )
    skin = regrid(surface.temperature_k)
    humidity = effective_surface_humidity(
        skin, p_half[-1], land, wetness, species["qv"][-1], np
    )
    out = similarity_surface_diagnostics(
        u_lowest=u[-1], v_lowest=v[-1], temperature_lowest=temperature[-1],
        qv_lowest=species["qv"][-1], p_full_lowest_pa=p_full[-1],
        p_surface_pa=p_half[-1], skin_temperature_k=skin,
        surface_humidity=humidity, roughness_m=regrid(surface.roughness_m), xp=np,
    )
    return {name: out[name] for name in ("t2", "q2", "u10", "v10")}, EXPORT_FALLBACK_SOURCE


__all__ = [
    "DERIVATION_FIELDS", "EXPORT_RECEIPT_NAME", "LANDUSE_TABLE_ATTRS",
    "SURFACE_ENERGY_FIELDS", "TAPE_ISLAKE", "TAPE_SST_POLICY",
    "TAPE_SST_SLAB_POLICY", "TAPE_VARIABLE_ATTRS",
    "TAPE_VERTICAL_VELOCITY_POLICY", "TAPE_WATER_CONVENTION",
    "SYNTHESIZED_FIELDS", "TAPE_TITLE", "TAPE_W_DESCRIPTION", "TapeGrid",
    "VERTICAL_VELOCITY_ATTR", "WATER_CONVENTION_ATTR", "WORLD_TAPE_ATTRS",
    "diagnose_ground_relative_w", "export_wrfout", "grid_metric_planes",
    "land_use_planes", "landuse_table", "parse_start_date", "subsolar_point",
    "tape_start", "tape_water_fields", "world_tape_attributes",
    "world_tape_class", "world_tape_title",
]
