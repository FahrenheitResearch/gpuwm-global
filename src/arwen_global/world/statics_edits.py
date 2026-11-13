"""Edits of Earth's static surface fields for an edited-Earth world.

Two worlds change Earth's coast without inventing a surface:

- a **flood** (``[world.sea] level_m > 0``): every land column whose
  statics terrain ``HGT_M`` is at or below the new sea level becomes open
  sea.  The rule is a bathtub: a column is flooded by its own height, not
  by a path of water from the ocean, so a depression below the new level
  floods even where no channel joins it to the sea.  Columns that are
  water on Earth (ocean and lakes) are left as they are.
- a **waterworld** (``[world.sea] land = "none"``): every column is open
  sea.  Land, land ice and Earth's inland lakes all become ocean, and the
  statics terrain is zero everywhere (the world table refuses a waterworld
  that is not flat).

The edit is applied in memory to an Earth cache that :func:`arwen_global.
statics.read_cache` has just read and verified, inside :func:`arwen_global.
statics.load_statics`, so every caller that loads the statics (the analysis
cold start, the analytic cold start, the surface-series builder and the
surface-series update) sees the same edited mask.  No edited cache is ever
written to disk: there is never a file that looks like an Earth cache and
is not one.

What a converted column carries, field by field, is what an Earth ocean
column carries:

- ``LANDMASK`` 0, ``LU_INDEX`` the land-use set's water class, ``LANDUSEF``
  all of its share in the water class (not the lake class, so the lake rule
  of :func:`arwen_global.statics.lake_columns` calls it sea), ``SCT_DOM``
  and ``SCB_DOM`` the water soil.  These are the planes the existing
  mask-against-category check in :func:`arwen_global.statics.
  resolve_surface_statics` compares, and they agree by construction.
- ``GREENFRAC``, ``LAI12M``, ``ALBEDO12M``, ``SNOALB`` and ``SOILTEMP``
  (plane by plane) take the median of Earth's open-ocean columns on the
  same latitude row; a row with no open ocean takes the nearest row that
  has some.  Over the sea these fields drive nothing in the land surface
  (Noah skips water), so the rule's only job is to leave no land value
  standing on a sea column.
- ``HGT_M`` is kept on a flood (the new sea's floor, and what the rule was
  evaluated on, so a reader can recount it) and is zero on a waterworld.
  No caller reads the statics terrain for the dynamics: the model's
  terrain always comes from the frame.

The edit's record (the rule, the counts and the sha256 of every edited
array) is added to the cache document under :data:`EDIT_KEY`; the
document's own ``arrays`` rows and file hashes still describe the Earth
cache that was read, which is the file on disk.
"""
from __future__ import annotations

import hashlib

import numpy as np

#: The version of the edit record (and of the rule it records).
EDIT_SCHEMA = "arwen-global.world-statics-edit/v1"
#: The cache-document key the edit record is stored under.
EDIT_KEY = "world_statics_edit"

#: Continuous fields given an ocean column's value on a converted column.
OCEAN_LIKE_FIELDS = ("GREENFRAC", "LAI12M", "ALBEDO12M", "SNOALB", "SOILTEMP")


def edits_statics(world) -> bool:
    """True when ``world`` edits Earth's statics: a flood or a waterworld on
    an edited Earth.  An invented world's statics are its own (they are
    written from its map, never edited from Earth's) and Earth needs none."""
    if world is None or world.is_earth or world.is_invented:
        return False
    return world.sea.land == "none" or world.sea.level_m > 0.0


def rule_text(world) -> str:
    """One sentence naming the edit, for the receipt and the run door."""
    if world.sea.land == "none":
        return ("waterworld: every column is open sea (land, land ice and "
                "inland lakes become ocean), statics terrain zero")
    return (f"flood to {world.sea.level_m:g} m: every land column whose "
            f"statics terrain HGT_M is at or below {world.sea.level_m:g} m "
            "becomes open sea (by its own height, not by connection to the "
            "ocean); Earth's ocean and lakes are kept")


def _categories(provenance: dict) -> tuple[int, int]:
    try:
        landuse = provenance["landuse"]
        return int(landuse["iswater"]), int(landuse["islake"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "the statics cache document names no land-use water and lake "
            "classes (landuse.iswater, landuse.islake), so a world's sea "
            "cannot be written into it") from exc


def earth_masks(fields: dict, provenance: dict) -> dict[str, np.ndarray]:
    """Earth's land, ocean and lake columns as the cache names them.

    Land is ``LANDMASK > 0.5`` (the crate's geogrid rule); a water column is
    a lake where the lake class's share beats the water class's (the rule
    of :func:`arwen_global.statics.lake_columns`), and ocean otherwise."""
    iswater, islake = _categories(provenance)
    landmask = np.asarray(fields["LANDMASK"], dtype=np.float64)
    luf = np.asarray(fields["LANDUSEF"], dtype=np.float64)
    land = landmask > 0.5
    lake = ~land & (luf[islake - 1] > luf[iswater - 1])
    return {"land": land, "lake": lake, "ocean": ~land & ~lake}


def new_water_columns(fields: dict, provenance: dict, world) -> np.ndarray:
    """The columns ``world`` turns from land into sea (bool plane).

    A flood: land with ``HGT_M <= level_m`` (in float64).  A waterworld:
    every land column."""
    masks = earth_masks(fields, provenance)
    if world.sea.land == "none":
        return masks["land"].copy()
    height = np.asarray(fields["HGT_M"], dtype=np.float64)
    return masks["land"] & (height <= float(world.sea.level_m))


def _row_ocean_values(plane: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Per latitude row, the median of ``plane`` over that row's ocean
    columns; a row with none takes the nearest row with some (the lower
    row on a tie).  Returns a (nlat,) float64 column."""
    nlat = plane.shape[0]
    values = np.full(nlat, np.nan, dtype=np.float64)
    data = np.asarray(plane, dtype=np.float64)
    for j in range(nlat):
        row = data[j][ocean[j]]
        if row.size:
            values[j] = float(np.median(row))
    have = np.flatnonzero(np.isfinite(values))
    if have.size == 0:
        raise ValueError(
            "the statics cache has no open-ocean column to take a sea "
            "column's values from")
    for j in np.flatnonzero(~np.isfinite(values)):
        nearest = have[np.argmin(np.abs(have - j))]
        values[j] = values[nearest]
    return values


def _ocean_like(value: np.ndarray, ocean: np.ndarray, converted: np.ndarray) -> np.ndarray:
    """``value`` (``(..., nlat, nlon)``) with each converted column set to
    its row's ocean median, plane by plane, in the array's own dtype."""
    out = np.array(value, copy=True)
    planes = out.reshape(-1, *out.shape[-2:])
    for plane in planes:
        row = _row_ocean_values(plane, ocean)
        filled = np.broadcast_to(row[:, None], plane.shape).astype(plane.dtype)
        plane[converted] = filled[converted]
    return planes.reshape(out.shape)


def _array_sha256(array: np.ndarray) -> str:
    # The statics cache's own array hash (statics._array_hash), restated so
    # this module needs nothing private from statics.py.
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def edit_statics(fields: dict, provenance: dict, world) -> tuple[dict, dict]:
    """``(fields, document)`` with ``world``'s flood or waterworld applied.

    ``fields`` and ``provenance`` are what :func:`arwen_global.statics.
    read_cache` returned for Earth's cache; neither is modified.  Every
    array keeps its shape and dtype.  The returned document is a copy with
    the edit record under :data:`EDIT_KEY`."""
    if not edits_statics(world):
        raise ValueError(
            f"world {world.label!r} does not edit Earth's statics (only a "
            "flood or a waterworld on an edited Earth does)")
    from arwen_global.statics import WORLD_CACHE_KEY

    if WORLD_CACHE_KEY in provenance:
        raise ValueError(
            "the statics cache read for an edited Earth carries a world "
            f"record ({str(provenance[WORLD_CACHE_KEY])[:80]}): it is an "
            "invented world's cache, and only Earth's cache is ever edited")
    iswater, islake = _categories(provenance)
    masks = earth_masks(fields, provenance)
    new_water = new_water_columns(fields, provenance, world)
    converted = new_water.copy()
    lakes_to_sea = np.zeros_like(new_water)
    if world.sea.land == "none":
        lakes_to_sea = masks["lake"].copy()
        converted |= lakes_to_sea
    ocean = masks["ocean"]

    out = {name: np.array(value, copy=True) for name, value in fields.items()}
    out["LANDMASK"][converted] = 0
    out["LU_INDEX"][converted] = iswater
    luf = out["LANDUSEF"]
    luf[:, converted] = 0
    luf[iswater - 1][converted] = 1
    from ..statics import ISOILWATER

    out["SCT_DOM"][converted] = ISOILWATER
    out["SCB_DOM"][converted] = ISOILWATER
    for name in OCEAN_LIKE_FIELDS:
        out[name] = _ocean_like(out[name], ocean, converted)
    if world.sea.land == "none":
        out["HGT_M"][...] = 0
    for name, value in out.items():
        source = fields[name]
        if value.shape != np.shape(source) or value.dtype != np.asarray(source).dtype:
            raise AssertionError(f"edited statics {name} changed shape or dtype")

    edited = sorted(name for name in out
                    if not np.array_equal(out[name], fields[name]))
    record = {
        "schema": EDIT_SCHEMA,
        "world": world.identity_hash,
        "label": world.label,
        "rule": rule_text(world),
        "sea_level_m": float(world.sea.level_m),
        "land": world.sea.land,
        "columns": int(new_water.size),
        "earth_land_columns": int(masks["land"].sum()),
        "earth_lake_columns": int(masks["lake"].sum()),
        "new_water_columns": int(new_water.sum()),
        "lake_columns_to_sea": int(lakes_to_sea.sum()),
        "land_columns_after": int((out["LANDMASK"] > 0.5).sum()),
        "ocean_value_rule": (
            "GREENFRAC, LAI12M, ALBEDO12M, SNOALB and SOILTEMP on a converted "
            "column: the median over Earth's open-ocean columns of its "
            "latitude row, plane by plane (nearest row with ocean where a row "
            "has none)"),
        "edited_arrays": {name: _array_sha256(out[name]) for name in edited},
    }
    document = dict(provenance)
    document[EDIT_KEY] = record
    return out, document


__all__ = [
    "EDIT_KEY", "EDIT_SCHEMA", "OCEAN_LIKE_FIELDS", "earth_masks",
    "edit_statics", "edits_statics", "new_water_columns", "rule_text",
]
