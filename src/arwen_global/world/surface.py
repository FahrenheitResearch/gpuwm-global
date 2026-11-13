"""The invented-world surface: a map's land and sea as the eleven cached
static arrays on the model grid.

An invented world (Pangea, a painted map) has no WPS_GEOG archive behind
it.  Its surface is written straight onto the model's Gaussian grid from
the map in the world's folder (:mod:`arwen_global.world.folder`), as a
cache the statics reader already accepts: the same eleven arrays, the
same per-array and side-file hashes, the same land-use convention.  The
cache sits in the world's folder under a name that carries the world's
surface key, never under an Earth cache's name and never in the Earth
cache directory.

The rules are deliberately dull, and each one is stated in the cache's
side file and the build report:

- **Sea.**  Every map cell at or below sea level (0 m) that is connected
  to the largest body of such cells (by area) is ocean.  Every other cell
  is land: a basin below sea level that does not reach the open ocean is
  dry land, so the world has no inland lakes.  The same rule is applied
  again on the model grid, so no water column the model sees is cut off
  from the ocean.
- **Land and sea on the grid.**  Each model cell is sampled on a small
  sub-grid (2 to 6 points a side, more when the map is finer than the
  grid), each sample taking the nearest map cell.  The water fraction is
  the share of samples that are ocean; a cell is land when that share is
  below one half, the crate's own LANDMASK rule.
- **One land-cover class** on every land column: MODIS open shrubland
  (category 7) or barren (16), from ``[world.surface] land_cover``.
- **One soil**: STAS loam (category 6), top and bottom.
- **Vegetation, leaf area and albedos from the WRF tables by class**
  (VEGPARM.TBL, the MODIS section): green fraction SHDFAC, leaf area
  LAIMAX, background albedo ALBEDOMIN and snow albedo MAXALB, the same
  every month (an invented world has no observed seasonal cycle).
- **No land ice, no lakes, no urban land.**
- **Deep soil** (the cache's SOILTEMP, the sea-level value the resolver
  lapses to the model's terrain): Earth's zonal mean at sea level, from an
  Earth statics cache, ``SOILTEMP + 0.0065 * HGT_M`` averaged over the land
  columns of each row and interpolated in latitude to this grid's rows
  (rows with no land take the value interpolated between the nearest rows
  that have some).
- **Terrain** (HGT_M) is the map's height above sea level averaged over
  the land samples, carried for the record only: the model's terrain
  always comes from the frame (plan 4.4), never from the statics.

Over water every array takes the value the Earth build leaves there: the
water category 17, water soil 14, no vegetation, 8 % albedo, no snow
albedo, no deep-soil temperature.

Nothing here runs for Earth.  ``load_world_statics`` is the reader the
statics seam calls for a world with its own surface; ``build_world_surface``
is what builds the cache, from Python or from the command line::

    python -m arwen_global.world.surface <run config> [--earth-statics <npz>] [--overwrite]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .folder import (
    WorldFolderError,
    file_sha256,
    read_lock,
    read_map,
    record_surface,
    resolve_world_folder,
    writing,
)

SURFACE_SCHEMA = "arwen-global.world-surface/v1"
#: Bumped whenever a rule below changes what a build writes; it is part of
#: the surface key, so a cache built under older rules is never read.
SURFACE_RULES_VERSION = 1
#: The land-use set an invented world's categories index: MODIS with
#: lakes, as the Earth build's, so the resolver, Noah's tables and the
#: runtime's convention check treat both alike.
LANDUSE_ATTRS = {
    "mminlu": "MODIFIED_IGBP_MODIS_NOAH",
    "iswater": 17, "islake": 21, "isice": 15, "isurban": 13,
    "num_land_cat": 21,
}
#: The MODIS category each ``[world.surface] land_cover`` puts on land.
LAND_COVER_CATEGORY = {"open-shrubland": 7, "barren": 16}
#: STAS loam, top and bottom (SOILPARM.TBL).
LOAM_SOIL_CATEGORY = 6
WATER_SOIL_CATEGORY = 14
#: The ocean's albedo in the Earth build's ALBEDO12M (percent).
WATER_ALBEDO_PERCENT = 8.0
#: real.exe's deep-soil lapse (statics.TMN_LAPSE_K_PER_M), used here to
#: bring Earth's deep soil to sea level.
DEEP_SOIL_LAPSE_K_PER_M = 0.0065
SEA_LEVEL_M = 0.0
#: Sub-samples per model cell side: at least 2 (so a coast inside a cell
#: gives a fractional cell), at most 6 (so a fine map stays cheap).
SAMPLES_MIN, SAMPLES_MAX = 2, 6
CACHE_PREFIX = "world-"


# -- connected bodies of water ----------------------------------------------


def _is_global(lon_deg: np.ndarray) -> bool:
    """A longitude axis that closes the circle (its spacing times its
    length is 360 degrees, within half a step)."""
    lon = np.sort(np.mod(np.asarray(lon_deg, dtype=np.float64), 360.0))
    if lon.size < 4:
        return False
    step = float(np.median(np.diff(lon)))
    return abs(step * lon.size - 360.0) <= 0.5 * step


def connected_labels(candidate: np.ndarray, *, periodic: bool,
                     polar_rows: tuple[int, ...] = ()) -> np.ndarray:
    """Label the 4-connected bodies of ``candidate`` cells on a (row,
    column) grid.  ``periodic`` joins the first and last columns;
    ``polar_rows`` are rows that touch a pole, where a cell is also the
    neighbour of the cell half-way round.  Each cell's label is the
    smallest flat index in its body; a non-candidate cell is -1.

    Minimum-label propagation with pointer jumping: pure numpy, no
    connectivity library."""
    cand = np.asarray(candidate, dtype=bool)
    nlat, nlon = cand.shape
    big = np.int64(cand.size)
    labels = np.where(cand, np.arange(cand.size, dtype=np.int64).reshape(cand.shape), big)
    half = nlon // 2 if nlon % 2 == 0 else None
    while True:
        new = labels.copy()
        both = cand[1:] & cand[:-1]
        new[1:] = np.minimum(new[1:], np.where(both, labels[:-1], big))
        new[:-1] = np.minimum(new[:-1], np.where(both, labels[1:], big))
        if periodic:
            for shift in (1, -1):
                neighbour = np.roll(labels, shift, axis=1)
                joined = cand & np.roll(cand, shift, axis=1)
                new = np.minimum(new, np.where(joined, neighbour, big))
        else:
            both = cand[:, 1:] & cand[:, :-1]
            new[:, 1:] = np.minimum(new[:, 1:], np.where(both, labels[:, :-1], big))
            new[:, :-1] = np.minimum(new[:, :-1], np.where(both, labels[:, 1:], big))
        if half is not None:
            for row in polar_rows:
                across = np.roll(labels[row], half)
                joined = cand[row] & np.roll(cand[row], half)
                new[row] = np.minimum(new[row], np.where(joined, across, big))
        new = np.where(cand, new, big)
        flat = new.ravel()
        inside = flat < big
        flat[inside] = np.minimum(flat[inside], flat[flat[inside]])
        new = flat.reshape(cand.shape)
        if np.array_equal(new, labels):
            break
        labels = new
    return np.where(cand, labels, -1)


def largest_body(candidate: np.ndarray, latitude_deg: np.ndarray, *,
                 periodic: bool, polar_rows: tuple[int, ...] = ()) -> tuple[np.ndarray, dict]:
    """The cells of the largest connected body of ``candidate`` by area
    (cells weighted by the cosine of their latitude), and a report."""
    cand = np.asarray(candidate, dtype=bool)
    if not cand.any():
        return np.zeros_like(cand), {"bodies": 0, "kept_cells": 0, "dropped_cells": 0,
                                     "kept_area_share": 0.0}
    labels = connected_labels(cand, periodic=periodic, polar_rows=polar_rows)
    weight = np.broadcast_to(
        np.cos(np.deg2rad(np.asarray(latitude_deg, dtype=np.float64)))[:, None],
        cand.shape)
    picked = labels[cand]
    area = np.bincount(picked, weights=weight[cand], minlength=cand.size)
    bodies = np.unique(picked)
    biggest = int(bodies[np.argmax(area[bodies])])
    kept = labels == biggest
    total = float(weight[cand].sum())
    return kept, {
        "bodies": int(bodies.size),
        "kept_cells": int(kept.sum()),
        "dropped_cells": int(cand.sum() - kept.sum()),
        "kept_area_share": float(weight[kept].sum() / total) if total > 0.0 else 0.0,
    }


def _polar_rows(latitude_deg: np.ndarray) -> tuple[int, ...]:
    """The rows nearest each pole, when they lie within one row spacing of
    it (a map that stops short of a pole has no polar join)."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    step = float(np.median(np.abs(np.diff(lat))))
    rows = []
    for pole in (90.0, -90.0):
        row = int(np.argmin(np.abs(lat - pole)))
        if abs(abs(lat[row]) - 90.0) <= step:
            rows.append(row)
    return tuple(sorted(set(rows)))


def flood_from_largest_ocean(elevation_m, latitude_deg, longitude_deg, *,
                             sea_level_m: float = SEA_LEVEL_M) -> tuple[np.ndarray, dict]:
    """The ocean of a map: cells at or below ``sea_level_m`` connected to
    the largest such body.  Returns ``(ocean, report)``."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    below = elevation <= float(sea_level_m)
    ocean, report = largest_body(
        below, latitude_deg, periodic=_is_global(longitude_deg),
        polar_rows=_polar_rows(latitude_deg))
    report = {
        "rule": ("ocean = cells at or below sea level connected to the largest "
                 "such body by area; every other cell is land (no inland lakes)"),
        "sea_level_m": float(sea_level_m),
        "cells_at_or_below_sea_level": int(below.sum()),
        "ocean_cells": int(ocean.sum()),
        "basins_made_land_cells": int(below.sum() - ocean.sum()),
        "bodies_below_sea_level": report["bodies"],
    }
    return ocean, report


# -- the map onto the model grid -------------------------------------------


def _nearest_index(sorted_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(sorted_axis, values), 1, sorted_axis.size - 1)
    left = right - 1
    pick_right = np.abs(sorted_axis[right] - values) < np.abs(values - sorted_axis[left])
    return np.where(pick_right, right, left)


def samples_per_side(map_lat, map_lon, nlat: int, nlon: int) -> int:
    map_dlat = float(np.median(np.abs(np.diff(np.asarray(map_lat, dtype=np.float64)))))
    map_dlon = float(np.median(np.abs(np.diff(np.asarray(map_lon, dtype=np.float64)))))
    ratio = max((180.0 / nlat) / map_dlat, (360.0 / nlon) / map_dlon)
    return int(min(SAMPLES_MAX, max(SAMPLES_MIN, math.ceil(ratio))))


def grid_sample_indices(map_lat, map_lon, grid_lat_deg, grid_lon_deg, n: int):
    """``(rows, columns)``: for each model row the ``n`` map rows its
    sub-samples take, ``(nlat, n)``, and for each model column the ``n``
    map columns, ``(nlon, n)``.  Model cells are bounded half-way between
    rows (the poles at the ends) and half a column either side."""
    map_lat = np.asarray(map_lat, dtype=np.float64)
    map_lon = np.asarray(map_lon, dtype=np.float64)
    lat = np.asarray(grid_lat_deg, dtype=np.float64)
    lon = np.asarray(grid_lon_deg, dtype=np.float64)
    order = np.argsort(lat)
    ordered = lat[order]
    edges = np.concatenate([[-90.0], 0.5 * (ordered[1:] + ordered[:-1]), [90.0]])
    fraction = (np.arange(n) + 0.5) / n
    sample_sorted = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * fraction[None]
    sample_lat = np.empty_like(sample_sorted)
    sample_lat[order] = sample_sorted
    lat_order = np.argsort(map_lat)
    rows = lat_order[_nearest_index(map_lat[lat_order], sample_lat)]
    dlon = 360.0 / lon.size
    sample_lon = np.mod(lon[:, None] + dlon * (fraction[None] - 0.5), 360.0)
    wrapped = np.mod(map_lon, 360.0)
    lon_order = np.argsort(wrapped)
    axis = wrapped[lon_order]
    # Extend the axis one column either side so the nearest search wraps.
    extended = np.concatenate([[axis[-1] - 360.0], axis, [axis[0] + 360.0]])
    index = _nearest_index(extended, sample_lon)
    columns = lon_order[np.mod(index - 1, axis.size)]
    return rows, columns


def sample_to_grid(values, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """The mean of ``values`` (a map array) over each model cell's
    sub-samples, ``(nlat, nlon)``."""
    field = np.asarray(values, dtype=np.float64)
    picked = field[rows[:, :, None, None], columns[None, None, :, :]]
    return picked.mean(axis=(1, 3))


# -- the class tables ------------------------------------------------------


def class_parameters(land_cover: str) -> dict:
    """The VEGPARM.TBL row an invented world's land takes, by the
    ``land_cover`` name."""
    from arwen_global.core.noah import load_tables

    if land_cover not in LAND_COVER_CATEGORY:
        raise ValueError(
            f"world.surface.land_cover must be one of {sorted(LAND_COVER_CATEGORY)}, "
            f"got {land_cover!r}")
    category = LAND_COVER_CATEGORY[land_cover]
    tables = load_tables(mminlu=LANDUSE_ATTRS["mminlu"])
    k = category - 1
    return {
        "land_cover": land_cover,
        "landuse_category": category,
        "green_fraction": float(tables.shdtbl[k]),
        "leaf_area_index": float(tables.laimaxtbl[k]),
        "background_albedo_percent": float(tables.albedomintbl[k]) * 100.0,
        "snow_albedo_percent": float(tables.maxalb[k]),
    }


# -- deep soil --------------------------------------------------------------


def earth_zonal_deep_soil(earth_fields: dict, latitude_deg) -> tuple[np.ndarray, dict]:
    """Earth's deep-soil temperature at sea level, zonally averaged over
    land, on the rows ``latitude_deg``.

    ``earth_fields`` is an Earth statics cache (any truncation): the mean
    of ``SOILTEMP + 0.0065 * HGT_M`` over each of its rows' land columns,
    interpolated in latitude to the requested rows.  Rows with no land are
    interpolated across; beyond the last row with land the end value is
    held."""
    soil = np.asarray(earth_fields["SOILTEMP"], dtype=np.float64)
    height = np.asarray(earth_fields["HGT_M"], dtype=np.float64)
    land = np.asarray(earth_fields["LANDMASK"], dtype=np.float64) > 0.5
    earth_lat = np.asarray(earth_fields["latitude_deg"], dtype=np.float64)
    sea_level = soil + DEEP_SOIL_LAPSE_K_PER_M * height
    count = land.sum(axis=1)
    has = count > 0
    if not has.any():
        raise ValueError("the Earth statics cache has no land column to take deep soil from")
    zonal = np.where(has, np.where(land, sea_level, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    order = np.argsort(earth_lat[has])
    rows = np.asarray(latitude_deg, dtype=np.float64)
    profile = np.interp(rows, earth_lat[has][order], zonal[has][order])
    if not np.isfinite(profile).all():
        raise ValueError("the Earth deep-soil profile is not finite")
    return profile, {
        "rule": ("Earth's zonal mean of SOILTEMP + 0.0065 * HGT_M over land, "
                 "interpolated in latitude"),
        "earth_rows": int(earth_lat.size),
        "earth_rows_without_land": int((~has).sum()),
        "range_k": [float(profile.min()), float(profile.max())],
    }


# -- the arrays ---------------------------------------------------------------


def surface_fields(water_fraction, land_height_m, deep_soil_k, land_cover: str,
                   grid) -> tuple[dict[str, np.ndarray], dict]:
    """The eleven cached arrays (and the grid's latitudes and longitudes)
    for a world whose water fraction and land height on the model grid
    are given.  ``deep_soil_k`` is one sea-level value per model row."""
    from arwen_global.statics import CACHED_FIELDS

    water = np.clip(np.asarray(water_fraction, dtype=np.float64), 0.0, 1.0)
    shape = tuple(int(n) for n in grid.shape)
    if water.shape != shape:
        raise ValueError(f"water fraction is {water.shape}, the grid is {shape}")
    deep = np.asarray(deep_soil_k, dtype=np.float64)
    if deep.shape != (shape[0],) or not np.isfinite(deep).all():
        raise ValueError(f"deep soil must be {shape[0]} finite values, one per row")
    if deep.min() < 150.0 or deep.max() > 350.0:
        raise ValueError(
            f"deep soil spans {deep.min():.1f} to {deep.max():.1f} K; a sea-level "
            "deep-soil temperature in kelvin is expected")
    height = np.asarray(land_height_m, dtype=np.float64)
    if height.shape != shape or not np.isfinite(height).all():
        raise ValueError("land height must be a finite plane on the grid")
    table = class_parameters(land_cover)
    category = table["landuse_category"]
    land = water < 0.5                       # the crate's LANDMASK rule
    land_share = 1.0 - water
    iswater = LANDUSE_ATTRS["iswater"]
    landusef = np.zeros((LANDUSE_ATTRS["num_land_cat"], *shape), dtype=np.float64)
    landusef[category - 1] = land_share
    landusef[iswater - 1] = water
    months = 12

    def monthly(on_land: float, on_water: float) -> np.ndarray:
        plane = np.where(land, on_land, on_water)
        return np.broadcast_to(plane, (months, *shape))

    fields = {
        "HGT_M": np.where(land, np.maximum(height, 0.0), 0.0),
        "LANDMASK": land.astype(np.float64),
        "LU_INDEX": np.where(land, float(category), float(iswater)),
        "SCT_DOM": np.where(land, float(LOAM_SOIL_CATEGORY), float(WATER_SOIL_CATEGORY)),
        "SCB_DOM": np.where(land, float(LOAM_SOIL_CATEGORY), float(WATER_SOIL_CATEGORY)),
        "LANDUSEF": landusef,
        "GREENFRAC": monthly(table["green_fraction"], 0.0),
        "LAI12M": monthly(table["leaf_area_index"], 0.0),
        "ALBEDO12M": monthly(table["background_albedo_percent"], WATER_ALBEDO_PERCENT),
        "SNOALB": np.where(land, table["snow_albedo_percent"], 0.0),
        "SOILTEMP": np.where(land, deep[:, None], 0.0),
    }
    if set(fields) != set(CACHED_FIELDS):
        raise AssertionError("the invented surface must write exactly the cached fields")
    out = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in fields.items()}
    out["latitude_deg"] = np.ascontiguousarray(grid.latitude_deg, dtype=np.float64)
    out["longitude_deg"] = np.ascontiguousarray(grid.longitude_deg, dtype=np.float64)
    weight = np.broadcast_to(np.asarray(grid.quadrature_weights, dtype=np.float64)[:, None], shape)
    detail = {
        "classes": table,
        "soil_category": LOAM_SOIL_CATEGORY,
        "land_columns": int(land.sum()),
        "water_columns": int((~land).sum()),
        "coastal_columns": int(((water > 0.0) & (water < 1.0)).sum()),
        "land_area_share": float((weight * land_share).sum() / weight.sum()),
    }
    return out, detail


# -- the cache in the world folder ------------------------------------------


def surface_key(map_sha256: str, land_cover: str) -> str:
    """What the eleven arrays depend on besides the grid: the map's bytes,
    the land-cover class and the rules version.  Soil wetness, the sea and
    the sun do not touch these arrays, so two worlds that differ only in
    those share one surface."""
    payload = {
        "schema": SURFACE_SCHEMA, "rules": SURFACE_RULES_VERSION,
        "map_sha256": str(map_sha256), "land_cover": str(land_cover),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def surface_cache_path(folder: Path, grid, key: str) -> Path:
    """The cache file for this grid and surface key inside the folder.
    The name carries ``world-<key>`` where an Earth cache carries its
    dataset tokens, so it can never be mistaken for one."""
    from arwen_global.statics import cache_stem

    stem = cache_stem(grid, (f"{CACHE_PREFIX}{key[:12]}",))
    return Path(folder) / f"{stem}.npz"


def _require_invented(world) -> None:
    if world is None or not getattr(world, "is_invented", False):
        raise ValueError(
            "the invented-world surface is read only for a world with "
            "[world.surface] folder set")


def build_world_surface(world, grid, *, deep_soil_k, base=None, deep_soil_detail=None,
                        overwrite: bool = False) -> tuple[Path, Path, dict]:
    """Build the world's surface cache for ``grid`` from the map in its
    folder.  Returns ``(npz, sidecar, report)``."""
    from arwen_global.statics import (
        STATICS_SCHEMA, _array_hash, machine_class, write_cache,
    )

    _require_invented(world)
    started = time.perf_counter()
    folder = resolve_world_folder(world.surface.folder, base)
    elevation, map_lat, map_lon, map_row = read_map(folder)
    land_cover = world.surface.land_cover
    key = surface_key(map_row["sha256"], land_cover)
    ocean, flood = flood_from_largest_ocean(elevation, map_lat, map_lon)
    nlat, nlon = (int(n) for n in grid.shape)
    n = samples_per_side(map_lat, map_lon, nlat, nlon)
    rows, columns = grid_sample_indices(map_lat, map_lon, grid.latitude_deg,
                                        grid.longitude_deg, n)
    water = sample_to_grid(ocean, rows, columns)
    land_samples = sample_to_grid(~ocean, rows, columns)
    height_sum = sample_to_grid(np.where(ocean, 0.0, np.maximum(elevation, 0.0)), rows, columns)
    height = np.where(land_samples > 0.0, height_sum / np.maximum(land_samples, 1e-300), 0.0)
    # The flood again on the model grid: a water column cut off from the
    # largest ocean at this resolution becomes land.
    wet = water >= 0.5
    kept, grid_flood = largest_body(
        wet, grid.latitude_deg, periodic=True, polar_rows=_polar_rows(grid.latitude_deg))
    cut_off = wet & ~kept
    water = np.where(cut_off, 0.0, water)
    fields, detail = surface_fields(water, height, deep_soil_k, land_cover, grid)
    flood["model_grid_water_columns_made_land"] = int(cut_off.sum())
    flood["model_grid_water_bodies"] = grid_flood["bodies"]
    world_surface = {
        "schema": SURFACE_SCHEMA,
        "rules_version": SURFACE_RULES_VERSION,
        "key": key,
        "folder": folder.name,
        "map_sha256": map_row["sha256"],
        "map_source": map_row.get("source", {}),
        "land_cover": land_cover,
        "samples_per_cell_side": n,
        "flood": flood,
        "deep_soil": deep_soil_detail or {"rule": "given by the caller"},
        "rules": [
            "ocean: at or below sea level and connected to the largest such body; no inland lakes",
            "land where the cell's ocean share is below one half",
            f"one land-cover class: MODIS {detail['classes']['landuse_category']} ({land_cover})",
            f"one soil: STAS {LOAM_SOIL_CATEGORY} (loam), top and bottom",
            "green fraction SHDFAC, leaf area LAIMAX, background albedo ALBEDOMIN, "
            "snow albedo MAXALB from VEGPARM.TBL, every month",
            "no land ice, no lakes, no urban land",
            "deep soil: Earth's zonal mean at sea level",
            "HGT_M for the record only; the model's terrain comes from the frame",
        ],
        **detail,
    }
    provenance = {
        "schema": STATICS_SCHEMA,
        "truncation": int(grid.truncation),
        "nlat": nlat, "nlon": nlon,
        "grid": {
            "kind": "rows", "lat_first_deg": float(grid.latitude_deg[0]),
            "lat_last_deg": float(grid.latitude_deg[-1]),
            "lon0_deg": float(grid.longitude_deg[0]),
            "dlon_deg": 360.0 / nlon,
        },
        "geog_root": None,
        "geog_data_res": [f"{CACHE_PREFIX}{key[:12]}"],
        "datasets": None,
        "landuse": dict(LANDUSE_ATTRS),
        "halo": 0,
        "builder": {"library": "arwen_global.world.surface",
                    "rules_version": SURFACE_RULES_VERSION},
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": machine_class(),
        "wall_seconds": float(time.perf_counter() - started),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str,
                   "sha256": _array_hash(value)}
            for name, value in fields.items()
        },
        "coverage": {},
        "world_surface": world_surface,
    }
    npz_path = surface_cache_path(folder, grid, key)
    with writing(folder):
        lock = read_lock(folder)
        if lock["map"]["sha256"] != map_row["sha256"]:
            raise WorldFolderError(
                f"the map in {folder} changed while its surface was being built; build again")
        npz, sidecar = write_cache(npz_path, fields, provenance, overwrite=overwrite)
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        record_surface(folder, npz.stem, {
            "file": npz.name,
            "key": key,
            "truncation": int(grid.truncation), "nlat": nlat, "nlon": nlon,
            "land_cover": land_cover,
            "npz_sha256": document["npz_sha256"],
            "self_sha256": document["self_sha256"],
            "built_at_utc": provenance["built_at_utc"],
        })
    report = {
        "cache": npz.name, "key": key, "land_cover": land_cover,
        "samples_per_cell_side": n, "flood": flood,
        "deep_soil": world_surface["deep_soil"], **detail,
        "wall_seconds": float(time.perf_counter() - started),
    }
    return npz, sidecar, report


def load_world_statics(world, grid, *, base=None,
                       folder=None) -> tuple[dict[str, np.ndarray], dict]:
    """The world's surface cache for ``grid``, with every hash checked, or
    a refusal naming the remedy.  This is the reader the statics seam
    calls in place of the Earth cache for a world with its own surface;
    it never falls back to an Earth cache.  ``folder`` is the world's
    folder as the config already resolved it (``statics.world_folder_of``);
    without it the folder is resolved against ``base``."""
    from arwen_global.statics import read_cache

    _require_invented(world)
    folder = (Path(folder) if folder is not None
              else resolve_world_folder(world.surface.folder, base))
    if not folder.is_dir():
        raise FileNotFoundError(
            f"world folder {folder} (world.surface.folder = {world.surface.folder!r}) "
            "does not exist; a relative folder is read beside the run config")
    lock = read_lock(folder)
    map_path = folder / lock["map"]["file"]
    if not map_path.is_file() or file_sha256(map_path) != lock["map"]["sha256"]:
        raise WorldFolderError(
            f"world map {map_path} is missing or is not the map the lock file "
            "pins; write it again with write_map and rebuild the surface")
    key = surface_key(lock["map"]["sha256"], world.surface.land_cover)
    npz_path = surface_cache_path(folder, grid, key)
    row = lock.get("surfaces", {}).get(npz_path.stem)
    remedy = ("build it once for this grid:\n"
              "  python -m arwen_global.world.surface <this run's config>")
    if row is None:
        raise FileNotFoundError(
            f"world folder {folder} has no surface for this grid and land cover "
            f"({npz_path.name}, T{grid.truncation} {grid.nlat}x{grid.nlon}, "
            f"{world.surface.land_cover}); {remedy}")
    if row.get("key") != key or not npz_path.is_file() or file_sha256(npz_path) != row["npz_sha256"]:
        raise WorldFolderError(
            f"world surface {npz_path} is not the cache the lock file names; {remedy} "
            "with --overwrite")
    fields, document = read_cache(npz_path, grid)
    stated = document.get("world_surface", {})
    if stated.get("key") != key or document.get("self_sha256") != row["self_sha256"]:
        raise WorldFolderError(
            f"world surface {npz_path} was built for another map or class; {remedy} "
            "with --overwrite")
    # What the receipt records about the folder the run read (its name and
    # hashes, never its path); added after every check, in memory only.
    document["world_folder"] = {
        "folder": folder.name,
        "lock_sha256": lock["self_sha256"],
        "map_sha256": lock["map"]["sha256"],
        "map_source": lock["map"].get("source", {}),
    }
    return fields, document


def surface_sentence(world, grid, *, base=None, folder=None) -> str:
    """The one line a door prints about an invented world's surface
    (``folder`` as in :func:`load_world_statics`)."""
    _require_invented(world)
    folder = (Path(folder) if folder is not None
              else resolve_world_folder(world.surface.folder, base))
    try:
        lock = read_lock(folder)
        key = surface_key(lock["map"]["sha256"], world.surface.land_cover)
        npz = surface_cache_path(folder, grid, key)
        state = "present" if npz.stem in lock.get("surfaces", {}) else "MISSING"
    except (WorldFolderError, KeyError):
        npz, state = folder, "MISSING (no map)"
    return (f"statics invented ({world.surface.land_cover}, loam, from the map in "
            f"{folder.name}; cache {npz.name} {state})")


# -- the command line ---------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arwen_global.world.surface",
        description="Build an invented world's surface cache for a run config's grid.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--earth-statics", type=Path, default=None,
                        help="an Earth statics cache (.npz with its .json) to take the "
                             "zonal-mean deep soil from; default: this config's own "
                             "Earth cache")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    from arwen_global.config import load_config
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.statics import StaticsOptions, cache_paths, read_cache

    try:
        cfg = load_config(args.config)
        world = cfg.world
        _require_invented(world)
        grid = GaussianGrid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                                   dealias_factor=cfg.dealias_factor)
        if args.earth_statics is not None:
            earth_path = args.earth_statics
        else:
            earth = StaticsOptions(source="real", geog_data_res=cfg.statics.geog_data_res,
                                   cache_dir=cfg.statics.cache_dir)
            earth_path = cache_paths(earth, grid)[0]
        earth_fields, earth_document = read_cache(earth_path)
        if earth_document.get("world_surface") is not None:
            raise ValueError(f"{earth_path} is an invented world's surface, not Earth's")
        deep, deep_detail = earth_zonal_deep_soil(earth_fields, grid.latitude_deg)
        deep_detail["earth_cache_npz_sha256"] = earth_document.get("npz_sha256")
        npz, _sidecar, report = build_world_surface(
            world, grid, deep_soil_k=deep, deep_soil_detail=deep_detail,
            base=Path(args.config).resolve().parent, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"gpuwm-global world surface: {exc}", file=sys.stderr)
        return 1
    print(f"gpuwm-global world surface: T{grid.truncation} {grid.nlat}x{grid.nlon} -> {npz}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "LAND_COVER_CATEGORY", "LANDUSE_ATTRS", "LOAM_SOIL_CATEGORY", "SURFACE_RULES_VERSION",
    "SURFACE_SCHEMA", "build_world_surface", "class_parameters", "connected_labels",
    "earth_zonal_deep_soil", "flood_from_largest_ocean", "grid_sample_indices",
    "largest_body", "load_world_statics", "sample_to_grid", "samples_per_side",
    "surface_cache_path", "surface_fields", "surface_key", "surface_sentence",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
