"""The sea of a world: its temperature and its ice (``[world.sea]``).

The sea is held for the whole run in every world this package builds,
exactly as the Earth run holds it: the open-water skin the cold start
writes is the sea surface temperature the ocean keeps (no ocean model;
``analysis_initial.open_water_skin_temperature``).  A world changes only
WHICH temperature and WHICH ice the cold start writes.

Three temperature sources (``world.sea.temperature``):

- ``"analysis"``: the analysis's own open-water skin, as Earth's run takes
  it.  A column the world floods is filled by the search the cold start
  already has.
- ``"zonal-mean"``: on every open-water column, the mean of the analysis's
  own open water along its latitude row of the model grid.  A row with no
  analysis open water reads the nearest rows that have it, linear in
  latitude, and beyond the last such row the last value is held.
- the four published aquaplanet profiles of Neale and Hoskins (2000):
  ``"aqua-control"``, ``"aqua-peaked"``, ``"aqua-flat"`` and
  ``"aqua-qobs"``, each 27 C at the equator and 0 C at and poleward of
  60 degrees (:func:`aquaplanet_profile_c`).

Three ice rules (``world.sea.ice``):

- ``"analysis"``: the analysis's ice, as Earth's run takes it.
- ``"zonal"``: on the analysis's water points, the mean ice fraction of
  each latitude row; the thickness is the row's mean over its ice points;
  the skin on the water points of a row that carries ice is the row's mean
  water skin (the composite of ice and open water the analysis reports,
  averaged as the fraction is); the zonal pack starts bare of snow.
- ``"none"``: no sea ice anywhere.  The water an analysis pack covered is
  open water at no less than the freezing point of sea water
  (:data:`SEA_WATER_FREEZING_K`, the ice-bottom temperature of the frozen
  surface step).

The ice acts on the analysis FRAME, before the cold start seeds its surface
from it (:func:`edit_sea_ice_frame`), because the seeding, the frozen
column and the statics' ice class all read it from there.  The temperature
acts on the cold start's open-water skin on the model grid
(:func:`world_open_water_skin`), after the cold start has formed it from
the analysis's own water points, so a zonal rule is exact on the model's
own rows.  Either function returns its input object unchanged when the
world's rule is the analysis's, and neither is called on Earth.
"""
from __future__ import annotations

import math

import numpy as np

from .spec import SEA_ICE_RULES, SEA_TEMPERATURE_PROFILES, SEA_TEMPERATURE_SOURCES

SEA_SCHEMA = "arwen-global.world-sea/v1"

#: Kelvin at 0 C.
ZERO_C_K = 273.15
#: The equatorial temperature of every Neale and Hoskins profile, C.
AQUAPLANET_PEAK_C = 27.0
#: The latitude at and beyond which every profile is 0 C, degrees.
AQUAPLANET_EDGE_DEG = 60.0
#: The freezing point of sea water: the ice-bottom temperature of the
#: frozen-surface step (physics/frozen_surface.SEA_ICE_BOTTOM_K).  Open
#: water the world writes is never colder.
SEA_WATER_FREEZING_K = 271.36
#: The sea-ice fraction at and above which a column freezes over
#: (statics.SEA_ICE_THRESHOLD; held here so this module imports nothing
#: heavy and checked against it by a test).
SEA_ICE_THRESHOLD = 0.5
#: Below this a source point is the analysis's water (the cold start's own
#: open-water test, ``land_src < 0.5``).
ANALYSIS_WATER_BELOW = 0.5


def aquaplanet_profile_c(name: str, latitude_deg) -> np.ndarray:
    """One of the four sea-surface temperature profiles of Neale and
    Hoskins (2000), in degrees C, at ``latitude_deg``.

    With phi the latitude and |phi| < 60 degrees (pi/3), and 0 C at and
    beyond 60 degrees:

    - control: 27 (1 - sin^2(3 phi / 2))
    - peaked:  27 (1 - 3 |phi| / pi)
    - flat:    27 (1 - sin^4(3 phi / 2))
    - qobs:    the mean of control and flat,
               27 (1 - sin^2(3 phi / 2) / 2 - sin^4(3 phi / 2) / 2)
    """
    if name not in SEA_TEMPERATURE_PROFILES:
        raise ValueError(
            f"unknown sea-temperature profile {name!r}; the published "
            f"profiles are {list(SEA_TEMPERATURE_PROFILES)}")
    phi = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    inside = np.abs(phi) < math.pi / 3.0
    s2 = np.sin(1.5 * phi) ** 2
    if name == "aqua-control":
        shape = 1.0 - s2
    elif name == "aqua-peaked":
        shape = 1.0 - 3.0 * np.abs(phi) / math.pi
    elif name == "aqua-flat":
        shape = 1.0 - s2 * s2
    else:  # aqua-qobs
        shape = 1.0 - 0.5 * s2 - 0.5 * s2 * s2
    return np.where(inside, AQUAPLANET_PEAK_C * shape, 0.0)


def aquaplanet_profile_k(name: str, latitude_deg) -> np.ndarray:
    """:func:`aquaplanet_profile_c` in kelvin."""
    return aquaplanet_profile_c(name, latitude_deg) + ZERO_C_K


def _row_mean(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean of ``values`` over ``mask`` along each row (the last axis),
    and whether the row had any masked point.  Rows without one read 0."""
    count = mask.sum(axis=-1)
    total = np.where(mask, values, 0.0).sum(axis=-1)
    has = count > 0
    mean = np.zeros(total.shape, dtype=np.float64)
    np.divide(total, count, out=mean, where=has)
    return mean, has


def _fill_rows_by_latitude(values: np.ndarray, has: np.ndarray,
                           latitude_deg: np.ndarray) -> np.ndarray:
    """``values`` on the rows that ``has``; every other row linear in
    latitude between the nearest rows that have one, and beyond the last
    such row on either side that row's value held."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    if has.all():
        return values
    if not has.any():
        raise ValueError(
            "the analysis has no open water on any latitude row of the model "
            "grid, so there is no zonal-mean sea to take")
    order = np.argsort(lat[has])
    return np.where(has, values,
                    np.interp(lat, lat[has][order], values[has][order]))


def zonal_mean_sea_temperature(open_water_skin, analysis_open_water,
                               latitude_deg) -> tuple[np.ndarray, dict]:
    """The zonal-mean rule's sea temperature for each model row, kelvin.

    ``open_water_skin`` is the cold start's open-water skin on the model
    grid; ``analysis_open_water`` the columns that are the analysis's own
    open water (where the mean is taken).  Returns ``(row_values, record)``.
    A row's value is never below :data:`SEA_WATER_FREEZING_K`.
    """
    skin = np.asarray(open_water_skin, dtype=np.float64)
    mask = np.asarray(analysis_open_water, dtype=bool)
    if skin.shape != mask.shape or skin.ndim != 2:
        raise ValueError("the open-water skin and its mask must be one model-grid plane")
    mean, has = _row_mean(skin, mask)
    filled = _fill_rows_by_latitude(mean, has, latitude_deg)
    floored = filled < SEA_WATER_FREEZING_K
    values = np.maximum(filled, SEA_WATER_FREEZING_K)
    record = {
        "rows": int(has.size),
        "rows_with_analysis_open_water": int(has.sum()),
        "rows_filled_by_latitude": int((~has).sum()),
        "rows_raised_to_freezing": int(floored.sum()),
        "analysis_open_water_columns": int(mask.sum()),
    }
    return values, record


def sea_temperature_rows(sea, latitude_deg, *, open_water_skin=None,
                         analysis_open_water=None) -> tuple[np.ndarray, dict]:
    """The sea temperature of each model row under ``sea.temperature``
    (kelvin, one value per row), for a rule that is a function of latitude
    only: ``"zonal-mean"`` (which reads the analysis's open water) or a
    published profile.  ``"analysis"`` has no row values and is refused
    here."""
    rule = sea.temperature
    if rule not in SEA_TEMPERATURE_SOURCES:
        raise ValueError(f"unknown sea-temperature source {rule!r}")
    if rule == "analysis":
        raise ValueError("the analysis's sea is not a function of latitude")
    if rule == "zonal-mean":
        if open_water_skin is None or analysis_open_water is None:
            raise ValueError("the zonal-mean sea is taken from the analysis's open water")
        return zonal_mean_sea_temperature(open_water_skin, analysis_open_water, latitude_deg)
    values = aquaplanet_profile_k(rule, latitude_deg)
    return values, {"profile": rule, "reference": "Neale and Hoskins (2000)"}


def world_open_water_skin(sea, open_water_skin, *, open_water, analysis_open_water,
                          latitude_deg, sea_columns=None):
    """The world's sea on the cold start's open-water skin.

    ``open_water_skin`` is the plane ``open_water_skin_temperature``
    returned (model grid, kelvin); ``open_water`` the run's open-water
    columns (``statics.water_columns`` on the world's own land and ice);
    ``analysis_open_water`` the columns that are the analysis's own open
    water (the analysis water points regridded, more than half water, and
    not frozen over in the run); ``latitude_deg`` the model rows'
    latitudes.  ``sea_columns`` (default ``open_water``) are the columns
    that receive the rule: a caller passes ``open_water & ~lakes`` to leave
    inland lakes on their own skin.

    Returns ``(skin, record)``.  Under ``"analysis"`` the skin is the input
    OBJECT, untouched.  Otherwise every sea column of a row carries the
    row's value, so the zonal mean of the result over the sea columns IS
    the rule's value at that latitude; every other column keeps the input.
    """
    if sea.temperature == "analysis":
        return open_water_skin, {
            "schema": SEA_SCHEMA, "temperature": "analysis",
            "rule": "the analysis's own open-water skin, as Earth's run takes it",
        }
    skin = np.asarray(open_water_skin, dtype=np.float64)
    open_water = np.asarray(open_water, dtype=bool)
    target = open_water if sea_columns is None else (
        np.asarray(sea_columns, dtype=bool) & open_water)
    rows, detail = sea_temperature_rows(
        sea, latitude_deg, open_water_skin=skin,
        analysis_open_water=np.asarray(analysis_open_water, dtype=bool) & open_water)
    plane = np.broadcast_to(np.asarray(rows, dtype=np.float64)[:, None], skin.shape)
    out = np.where(target, plane, skin)
    change = np.where(target, out - skin, 0.0)
    record = {
        "schema": SEA_SCHEMA,
        "temperature": sea.temperature,
        "rule": (
            "every open-water sea column of a model row carries the row's "
            + ("mean over the analysis's own open water (rows without any "
               "read the nearest rows linear in latitude; never below "
               f"{SEA_WATER_FREEZING_K} K)"
               if sea.temperature == "zonal-mean"
               else f"value of the {sea.temperature} profile of Neale and "
                    "Hoskins (2000): 27 C at the equator, 0 C at and beyond 60 degrees")
        ),
        "sea_columns": int(target.sum()),
        "row_values_k": {"min": float(np.min(rows)), "max": float(np.max(rows))},
        "change_k": {
            "min": float(change.min()) if target.any() else 0.0,
            "max": float(change.max()) if target.any() else 0.0,
            "mean_abs": float(np.abs(change[target]).mean()) if target.any() else 0.0,
        },
        **detail,
    }
    return out, record


def zonal_mean_of_sea(skin, sea_columns) -> np.ndarray:
    """The zonal mean of ``skin`` over ``sea_columns`` along each row (NaN
    on a row with none): the quantity a world's sea is checked by."""
    mean, has = _row_mean(np.asarray(skin, dtype=np.float64),
                          np.asarray(sea_columns, dtype=bool))
    return np.where(has, mean, np.nan)


# -- the ice, on the analysis frame -----------------------------------------


class _WorldField:
    """A replaced plane of an edited frame, carrying the product name of
    the field it replaced (``.source``, read by the cold start's receipt)."""

    __slots__ = ("values", "source")

    def __init__(self, values, source="primary"):
        self.values = values
        self.source = source


class WorldFrame:
    """An analysis frame with some of its planes replaced by a world's
    edit.  ``fields`` is the frame's own map with the replaced planes
    swapped in; every other attribute (the grid, the levels, the times,
    the digests, ``snow_on_sea_ice``, ``fill``) is read from the frame it
    wraps, whatever its type.  ``world_edits`` holds each edit's record."""

    def __init__(self, base, fields: dict, world_edits: dict):
        self._base = base
        self.fields = fields
        self.world_edits = world_edits

    def __getattr__(self, name):
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)


def edited_frame(frame, replacements: dict, record_key: str, record: dict):
    """``frame`` with ``replacements`` (name -> array) in place of the named
    planes; every other field and attribute is the frame's own object.  The
    edit's record is kept under ``world_edits[record_key]``.  The input
    frame is never modified."""
    fields = dict(frame.fields)
    for name, values in replacements.items():
        old = fields.get(name)
        fields[name] = _WorldField(values, getattr(old, "source", "primary"))
    edits = dict(getattr(frame, "world_edits", None) or {})
    edits[record_key] = record
    return WorldFrame(frame, fields, edits)


def edit_sea_ice_frame(frame, sea):
    """The world's ice rule on the analysis ``frame`` (source grid), before
    the cold start seeds its surface.  Returns ``(frame, record)``; under
    ``ice = "analysis"`` the frame is the input object, untouched.

    Reads and writes ``sea_ice_fraction``, ``sea_ice_thickness`` and
    ``skin_temperature`` on the analysis's water points (``land_fraction``
    below one half), and under ``"zonal"`` the two snow planes on the water
    points that freeze.  Land points are never touched.
    """
    rule = sea.ice
    if rule not in SEA_ICE_RULES:
        raise ValueError(f"unknown sea-ice rule {rule!r}")
    if rule == "analysis":
        return frame, {"schema": SEA_SCHEMA, "ice": "analysis",
                       "rule": "the analysis's sea ice, as Earth's run takes it"}
    for name in ("land_fraction", "sea_ice_fraction", "sea_ice_thickness",
                 "skin_temperature"):
        if name not in frame.fields:
            raise ValueError(f"the world's ice rule needs the analysis field {name}")
    land = np.asarray(frame.fields["land_fraction"].values, dtype=np.float64)
    water = land < ANALYSIS_WATER_BELOW
    ice = np.clip(np.asarray(frame.fields["sea_ice_fraction"].values, dtype=np.float64), 0.0, 1.0)
    thickness = np.maximum(np.asarray(frame.fields["sea_ice_thickness"].values,
                                      dtype=np.float64), 0.0)
    skin = np.asarray(frame.fields["skin_temperature"].values, dtype=np.float64)
    if ice.shape != land.shape or skin.shape != land.shape or thickness.shape != land.shape:
        raise ValueError("the sea-ice, skin and land planes must share the analysis grid")
    before_frozen = int(np.count_nonzero(water & (ice >= SEA_ICE_THRESHOLD)))
    before_any = water & (ice > 0.0)
    replacements = {}
    if rule == "none":
        new_ice = np.where(water, 0.0, ice)
        new_thickness = np.where(water, 0.0, thickness)
        thawed = before_any
        new_skin = np.where(thawed, np.maximum(skin, SEA_WATER_FREEZING_K), skin)
        record = {
            "schema": SEA_SCHEMA, "ice": "none",
            "rule": ("no sea ice: every analysis water point's ice fraction and "
                     "thickness are 0, and the water an analysis pack covered "
                     f"is open water at no less than {SEA_WATER_FREEZING_K} K"),
            "analysis_points_with_ice": int(before_any.sum()),
            "analysis_points_frozen_over": before_frozen,
            "skin_raised_points": int(np.count_nonzero(new_skin != skin)),
        }
    else:  # zonal
        row_ice, has_water = _row_mean(ice, water)
        thick_mask = water & (ice > 0.0)
        row_thickness, _has_thick = _row_mean(thickness, thick_mask)
        row_skin, _ = _row_mean(skin, water)
        plane = lambda row: np.broadcast_to(row[:, None], land.shape)  # noqa: E731
        icy_row = has_water & (row_ice > 0.0)
        new_ice = np.where(water, plane(row_ice), ice)
        new_thickness = np.where(water, np.where(plane(icy_row), plane(row_thickness), 0.0),
                                 thickness)
        # The skin of a row that carries ice is the row's mean water skin,
        # the composite of the ice and open water the analysis reports; a
        # row whose zonal pack does not freeze it over is open water and
        # is never colder than the freezing point.
        row_frozen = row_ice >= SEA_ICE_THRESHOLD
        row_skin_w = np.where(row_frozen, row_skin,
                              np.maximum(row_skin, SEA_WATER_FREEZING_K))
        new_skin = np.where(water & plane(icy_row), plane(row_skin_w), skin)
        freezes = water & (new_ice >= SEA_ICE_THRESHOLD)
        for name in ("snow_water_equivalent", "snow_depth"):
            if name in frame.fields:
                snow = np.asarray(frame.fields[name].values, dtype=np.float64)
                if snow.shape == land.shape:
                    replacements[name] = np.where(freezes, 0.0, snow)
        record = {
            "schema": SEA_SCHEMA, "ice": "zonal",
            "rule": ("on the analysis's water points each latitude row carries "
                     "its mean ice fraction; thickness is the row's mean over "
                     "its ice points; the skin of a row with ice is its mean "
                     "water skin (open water never below "
                     f"{SEA_WATER_FREEZING_K} K); the zonal pack starts bare"),
            "analysis_points_frozen_over": before_frozen,
            "world_points_frozen_over": int(freezes.sum()),
            "rows_with_ice": int(icy_row.sum()),
            "largest_row_fraction": float(row_ice.max()) if row_ice.size else 0.0,
        }
    replacements.update({
        "sea_ice_fraction": new_ice,
        "sea_ice_thickness": new_thickness,
        "skin_temperature": new_skin,
    })
    return edited_frame(frame, replacements, "sea_ice", record), record


def analysis_open_water_columns(land_src, regrid, run_open_water) -> np.ndarray:
    """The model columns that are the analysis's own open water: more than
    half analysis water under the regrid's stencil (``land_src`` below one
    half) and open water in the run.  The zonal-mean rule averages over
    these and nothing else, so a column a world floods is never averaged
    into the sea it is given."""
    water_src = (np.asarray(land_src, dtype=np.float64) < ANALYSIS_WATER_BELOW)
    weight = regrid(water_src.astype(np.float64))
    return (weight > 0.5) & np.asarray(run_open_water, dtype=bool)


__all__ = [
    "ANALYSIS_WATER_BELOW", "AQUAPLANET_EDGE_DEG", "AQUAPLANET_PEAK_C",
    "SEA_SCHEMA", "SEA_WATER_FREEZING_K", "ZERO_C_K", "WorldFrame",
    "analysis_open_water_columns", "aquaplanet_profile_c", "aquaplanet_profile_k",
    "edit_sea_ice_frame", "edited_frame", "sea_temperature_rows",
    "world_open_water_skin", "zonal_mean_of_sea", "zonal_mean_sea_temperature",
]
