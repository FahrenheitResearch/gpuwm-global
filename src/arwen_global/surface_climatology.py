"""The monthly sea-surface climatology on a run's Gaussian grid.

A run of months has no dated SST to follow: a series stops at its last
record, and holding the start's SST holds the start's season with it.  The
long-run ocean rules (hold, anomaly, climatology) need one more object, a
climatological SST and sea-ice fraction by month on the run's own grid.
This module makes it, in two steps that never need a credential:

* :func:`fetch` stages the source files once.  The sources are ROWS of
  ``data/surface_climatology_sources.json`` (file names, routes, byte count
  and SHA-256, variable and units), the way ``surface_series_sources.json``
  holds the dated series, so a new product is a new row.  Every file is
  verified by its byte count and SHA-256 before it is installed, and a
  staged file with other bytes is moved aside and fetched again.  The WOA18
  mixed-layer row is first looked for in the regional engine's ocean cache,
  where its twelve pinned months may already be.
* :func:`build_climatology` regrids the twelve months onto the Gaussian grid
  with the cold start's own open-water rule
  (:func:`arwen_global.analysis_initial.open_water_skin_temperature`: the
  bilinear weights renormalised over the source's own water points, and a
  column with none in its stencil reads the source water grown outward over
  the source land), on the statics ocean (the water columns that are not
  inland lakes), as the dated series does on the water mask.  The source's
  own water is its missing-value mask.  Every input is verified again
  before it is decoded.  The output is one
  ``.npz`` (:data:`CLIMATOLOGY_SCHEMA`) written byte-reproducibly, so its
  SHA-256 is the identity a run can carry.

What the file is and is not.  It is a CLIMATOLOGY: the 1991-2020 mean of
each calendar month, not an observation of any year.  It is defined on
ocean columns only: land and inland-lake columns hold NaN (a lake's skin
follows its own energy budget), so a reader that forgets the mask fails
rather than reading a number nobody measured.  The mixed-layer depth is
stored as regridded, not clamped: WOA18 holds depths from near zero to
nearly 2 km, and the bounds a slab ocean clamps it to are that ocean's
parameters, not the file's.  How a run uses it --
which rule, which damping -- is the ocean switch's business, not this
file's.

Time interpolation (:func:`month_weights`) is linear between mid-month
instants, mid-month being the midpoint of each real calendar month of the
year in question (leap Februaries included), and continuous in seconds, so
an SST written every step has no daily step in it.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

import numpy as np

CLIMATOLOGY_SCHEMA = "gpuwm.arwen-global-surface-climatology/v1"
SOURCES_SCHEMA = "gpuwm.arwen-global-surface-climatology-sources/v1"
SOURCES_TABLE = Path(__file__).parent / "data" / "surface_climatology_sources.json"

#: Where :func:`fetch` stages the source files, overriding the default
#: beside the case data (``<case data root>/arwen-global-climatology``).
CACHE_ENV = "GPUWM_GLOBAL_CLIMATOLOGY"
#: The regional engine's ocean cache variable; unset, the engine's cache is
#: ``<case data root>/ocean`` (gpuwm/ocean/mixed_layer/climatology.py).
ENGINE_OCEAN_CACHE_ENV = "GPUWM_OCEAN_CACHE"

#: The planes a climatology file can carry, in the order they are written.
PLANES = ("sst_k", "sea_ice_fraction", "mixed_layer_depth_m")
#: The planes every climatology file carries; the mixed layer is the slab
#: ocean's and is built only on request.
REQUIRED_PLANES = ("sst_k", "sea_ice_fraction")
MLD_ROW = "woa18-a5b7-mld"
#: The sea-ice concentration counted as ice-covered for an extent figure
#: (the NSIDC convention).
EXTENT_THRESHOLD = 0.15
CELSIUS_TO_KELVIN = 273.15
#: The fixed timestamp every member of a written file carries, so two builds
#: from the same inputs are the same bytes.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_REQUEST_TIMEOUT_S = 120.0
_ATTEMPTS = 3


class ClimatologyPinError(ValueError):
    """A source file whose bytes are not the bytes its row pins."""


class ClimatologyUnavailable(RuntimeError):
    """No staged copy and no route delivered the pinned file."""


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def load_sources(path=None) -> dict:
    table = json.loads(Path(path or SOURCES_TABLE).read_text(encoding="utf-8"))
    if table.get("schema") != SOURCES_SCHEMA:
        raise ValueError(f"surface-climatology source table schema is not {SOURCES_SCHEMA}")
    return table


def resolve_rows(table: dict, row_ids=None) -> list[dict]:
    """The named rows, or the table's default rows when none are named."""
    rows = {row["id"]: row for row in table["rows"]}
    if row_ids is None:
        return [row for row in table["rows"] if row.get("default")]
    unknown = [name for name in row_ids if name not in rows]
    if unknown:
        raise ValueError(f"no surface-climatology source {unknown[0]!r}; the table has "
                         + ", ".join(sorted(rows)))
    return [rows[name] for name in dict.fromkeys(row_ids)]


def row_ids_for(with_mld: bool, table=None) -> list[str]:
    ids = [row["id"] for row in resolve_rows(load_sources(table))]
    return ids + [MLD_ROW] if with_mld and MLD_ROW not in ids else ids


def download_bytes(rows) -> int:
    """What staging ``rows`` downloads at most (nothing already staged)."""
    return int(sum(spec["bytes"] for row in rows for spec in row["files"]))


def climatology_bytes(nlat: int, nlon: int, *, mixed_layer: bool) -> int:
    """The planes' payload of a built file: twelve float32 planes per field."""
    fields = 3 if mixed_layer else 2
    return int(fields * 12 * int(nlat) * int(nlon) * 4)


# ---------------------------------------------------------------------------
# where the files live, and their pins
# ---------------------------------------------------------------------------

def default_cache_dir() -> Path:
    raw = os.environ.get(CACHE_ENV, "").strip()
    if raw:
        return Path(raw)
    from gpuwm.case_data import case_data_root

    return case_data_root() / "arwen-global-climatology"


def engine_ocean_cache() -> Path:
    raw = os.environ.get(ENGINE_OCEAN_CACHE_ENV, "").strip()
    if raw:
        return Path(raw)
    from gpuwm.case_data import case_data_root

    return case_data_root() / "ocean"


def _staged_path(row: dict, spec: dict, cache) -> Path:
    return Path(cache or default_cache_dir()) / row["id"] / spec["name"]


def _candidates(row: dict, spec: dict, cache) -> list[Path]:
    paths = [_staged_path(row, spec, cache)]
    if row.get("engine_cache") == "ocean":
        paths.append(engine_ocean_cache() / spec["name"])
    return paths


def sha256_file(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin_problem(path, spec: dict) -> str | None:
    """None when ``path`` holds the pinned bytes, else what differs.  The
    byte count is read first, so a short file is refused without hashing."""
    size = Path(path).stat().st_size
    if size != int(spec["bytes"]):
        return f"{size} bytes, the pin is {spec['bytes']}"
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        return f"SHA-256 {digest}, the pin is {spec['sha256']}"
    return None


def verify(path, spec: dict) -> Path:
    problem = pin_problem(path, spec)
    if problem:
        raise ClimatologyPinError(f"{path}: {problem}; it is not {spec['name']} as the "
                                  "source row pins it, and it is not read")
    return Path(path)


def locate(row: dict, spec: dict, cache=None) -> Path | None:
    """The first staged copy holding the pinned bytes: this package's cache,
    then (for a row that names it) the regional engine's.  A copy with other
    bytes is passed over, never used."""
    for path in _candidates(row, spec, cache):
        if path.is_file() and pin_problem(path, spec) is None:
            return path
    return None


def _download(url: str, destination: Path) -> None:
    last = None
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT_S) as response, \
                    destination.open("wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            return
        except OSError as error:
            last = error
            destination.unlink(missing_ok=True)
            if attempt + 1 < _ATTEMPTS:
                time.sleep(5.0 * (attempt + 1))
    raise last


def fetch(row_ids=None, *, cache=None, source_dir=None, table=None,
          progress=print) -> list[dict]:
    """Stage every file of the rows, verified; return one record per file.

    ``source_dir`` copies from a local folder instead of the network (an
    offline site, a mirror already on the box).  A file already staged with
    the pinned bytes is not fetched again; one with other bytes in this
    package's cache is moved aside first.  The engine's cache is only read.
    """
    rows = resolve_rows(load_sources(table), row_ids)
    records = []
    for row in rows:
        for spec in row["files"]:
            found = locate(row, spec, cache)
            if found is not None:
                where = ("in the engine's cache" if found != _staged_path(row, spec, cache)
                         else "staged")
                progress(f"climatology fetch: {spec['name']} already {where} with the pinned "
                         f"bytes ({spec['bytes']:,} B) at {found}")
                records.append({"row": row["id"], "name": spec["name"], "path": str(found),
                                "bytes": spec["bytes"], "sha256": spec["sha256"],
                                "action": f"already {where}"})
                continue
            target = _staged_path(row, spec, cache)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                aside = target.with_name(f"{target.name}.invalid-{time.time_ns()}")
                target.rename(aside)
                progress(f"climatology fetch: {target} does not hold the pinned bytes; "
                         f"moved aside to {aside.name}")
            partial = target.with_name(target.name + ".part")
            started = time.perf_counter()
            if source_dir is not None:
                local = Path(source_dir) / spec["name"]
                if not local.is_file():
                    raise ClimatologyUnavailable(
                        f"{spec['name']} (row {row['id']}) is not in {source_dir}")
                shutil.copyfile(local, partial)
                routes, errors = [local.resolve().as_uri()], []
                route = routes[0]
            else:
                routes, errors, route = list(spec["urls"]), [], None
                for url in routes:
                    try:
                        _download(url, partial)
                    except OSError as error:
                        errors.append(f"{url}: {type(error).__name__}: {error}")
                        continue
                    problem = pin_problem(partial, spec)
                    if problem is None:
                        route = url
                        break
                    errors.append(f"{url}: {problem}")
                    partial.unlink(missing_ok=True)
                if route is None:
                    raise ClimatologyUnavailable(
                        f"{spec['name']} (row {row['id']}) is not staged at {target} and no "
                        "route delivered the pinned bytes: " + "; ".join(errors)
                        + ". Stage it from a folder that holds it with "
                        "`gpuwm-global climatology fetch --from DIR`.")
            try:
                verify(partial, spec)
            except ClimatologyPinError:
                partial.unlink(missing_ok=True)
                raise
            partial.replace(target)
            seconds = time.perf_counter() - started
            progress(f"climatology fetch: {spec['name']} verified and installed "
                     f"({spec['bytes']:,} B, SHA-256 {spec['sha256'][:12]}..., "
                     f"{seconds:.1f} s) at {target}")
            records.append({"row": row["id"], "name": spec["name"], "path": str(target),
                            "bytes": spec["bytes"], "sha256": spec["sha256"],
                            "action": "installed", "route": route,
                            "seconds": round(seconds, 2)})
    return records


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def _decode_file(path: Path, spec: dict):
    """``(values, latitude, longitude)`` through the engine's Rust NetCDF
    door; CF missing values decode as NaN.  ``values`` is ``(months, nlat,
    nlon)`` for a twelve-month file and ``(1, nlat, nlon)`` for one month."""
    from gpuwm.netcdf_bridge import open_dataset

    with open_dataset(path) as ds:
        names = (spec["variable"], spec["latitude"], spec["longitude"])
        missing = [name for name in names if name not in ds.variables]
        if missing:
            raise ValueError(f"{path} carries no {missing[0]!r}")
        values = np.asarray(ds.variables[spec["variable"]][...], dtype=np.float64)
        latitude = np.asarray(ds.variables[spec["latitude"]][...], dtype=np.float64)
        longitude = np.asarray(ds.variables[spec["longitude"]][...], dtype=np.float64)
        bounds = (np.asarray(ds.variables["climatology_bounds"][...], dtype=np.float64)
                  if "climatology_bounds" in ds.variables else None)
        time_units = (str(getattr(ds.variables["time"], "units", ""))
                      if "time" in ds.variables else "")
    values = values.reshape((-1, latitude.size, longitude.size))
    months = int(spec.get("months", 1))
    if values.shape[0] != months:
        raise ValueError(f"{path}: {spec['variable']} holds {values.shape[0]} planes, "
                         f"the row says {months}")
    if months == 12:
        _check_month_order(path, bounds, time_units)
    return values, latitude, longitude


def _check_month_order(path, bounds, units: str) -> None:
    """A twelve-plane file must run January to December: its climatology
    bounds start on the first of each month of one year, in order."""
    if bounds is None or not units.startswith("days since "):
        raise ValueError(f"{path}: no climatology_bounds in days, so the month "
                         "order of its twelve planes cannot be checked")
    origin = datetime.fromisoformat(units[len("days since "):].strip().replace(" ", "T"))
    starts = [origin + timedelta(days=float(day)) for day in bounds[:, 0]]
    if [(d.month, d.day) for d in starts] != [(m, 1) for m in range(1, 13)] or \
            len({d.year for d in starts}) != 1:
        raise ValueError(f"{path}: the twelve planes do not start on the first of "
                         "January .. December of one year: "
                         + ", ".join(d.date().isoformat() for d in starts))


def _source_planes(rows, cache) -> tuple[dict, list[dict]]:
    """Every file verified, THEN decoded: a file with other bytes is refused
    before a byte of it is read as data."""
    staged = []
    for row in rows:
        for spec in row["files"]:
            path = locate(row, spec, cache)
            if path is None:
                looked = ", ".join(str(p) for p in _candidates(row, spec, cache))
                existing = next((p for p in _candidates(row, spec, cache) if p.is_file()), None)
                if existing is not None:
                    verify(existing, spec)  # raises, naming what differs
                raise ClimatologyUnavailable(
                    f"{spec['name']} (row {row['id']}) is not staged; looked in {looked}. "
                    "Stage it with `gpuwm-global climatology fetch`"
                    + (" --with-mld" if row["id"] == MLD_ROW else "") + ".")
            staged.append((row, spec, verify(path, spec)))
    decoded = {}
    inputs = []
    for row, spec, path in staged:
        values, lat, lon = _decode_file(path, spec)
        plane = spec["plane"]
        entry = decoded.setdefault(plane, {"months": {}, "specs": {}})
        first = 1 if int(spec.get("months", 1)) == 12 else int(spec["month"])
        for offset in range(values.shape[0]):
            entry["months"][first + offset] = (values[offset], lat, lon)
            entry["specs"][first + offset] = spec
        inputs.append({"row": row["id"], "name": spec["name"], "bytes": int(spec["bytes"]),
                       "sha256": spec["sha256"]})
    for plane, entry in decoded.items():
        if sorted(entry["months"]) != list(range(1, 13)):
            raise ValueError(f"the rows give {plane} for months {sorted(entry['months'])}, "
                             "not all twelve")
    return decoded, inputs


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _source_water(plane: str, month: int, decoded: dict) -> np.ndarray:
    values, _lat, _lon = decoded[plane]["months"][month]
    spec = decoded[plane]["specs"][month]
    rule = spec.get("water", "finite")
    if rule == "finite":
        return np.isfinite(values)
    if rule == "open-range":
        low, high = spec["valid_open_range"]
        with np.errstate(invalid="ignore"):
            return np.isfinite(values) & (values > low) & (values < high)
    if rule in decoded:
        return _source_water(rule, month, decoded)
    raise ValueError(f"{spec['name']}: water rule {rule!r} names no plane this build has")


def _source_values(plane: str, month: int, decoded: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(values on the source water, source water)`` in the file's units
    converted to the plane's (degC to K for the SST)."""
    values, _lat, _lon = decoded[plane]["months"][month]
    spec = decoded[plane]["specs"][month]
    water = _source_water(plane, month, decoded)
    values = np.array(values, dtype=np.float64)
    if "fill_on_water" in spec:
        values = np.where(water & ~np.isfinite(values), float(spec["fill_on_water"]), values)
    units = spec["units"]
    if units == "degC":
        values = values + CELSIUS_TO_KELVIN
    elif units == "fraction":
        top = float(np.nanmax(np.where(water, values, np.nan)))
        if top > 1.0 + 1.0e-6:
            raise ValueError(f"{spec['name']}: {spec['variable']} reaches {top}, which is "
                             "not a fraction; the row's units say fraction")
    elif units != "m":
        raise ValueError(f"{spec['name']}: units {units!r} are not ones this build converts")
    if not np.isfinite(values[water]).all():
        raise ValueError(f"{spec['name']}: non-finite values on the source's water points")
    return np.where(water, values, 0.0), water


def _model_areas_km2(grid) -> np.ndarray:
    from .spectral.constants import EARTH_RADIUS_M

    radius_km = EARTH_RADIUS_M / 1000.0
    row = radius_km * radius_km * np.asarray(grid.quadrature_weights) * (2.0 * np.pi / grid.nlon)
    return np.broadcast_to(row[:, None], (grid.nlat, grid.nlon))


def _source_areas_km2(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    from .spectral.constants import EARTH_RADIUS_M

    radius_km = EARTH_RADIUS_M / 1000.0
    dlat = np.deg2rad(abs(float(latitude[1] - latitude[0])))
    dlon = np.deg2rad(abs(float(longitude[1] - longitude[0])))
    phi = np.deg2rad(latitude)
    band = np.sin(np.minimum(phi + dlat / 2, np.pi / 2)) - np.sin(np.maximum(phi - dlat / 2, -np.pi / 2))
    return np.broadcast_to((radius_km * radius_km * dlon * band)[:, None],
                           (latitude.size, longitude.size))


def _at_source(mask, latitude, longitude, grid) -> np.ndarray:
    """A model-grid mask read at each source point's nearest model column."""
    rows = np.abs(np.asarray(latitude)[:, None] - grid.latitude_deg[None, :]).argmin(axis=1)
    step = 360.0 / grid.nlon
    columns = np.rint(np.mod(np.asarray(longitude), 360.0) / step).astype(np.int64) % grid.nlon
    return np.asarray(mask, dtype=bool)[rows][:, columns]


def _interior(mask) -> np.ndarray:
    """The columns whose 3 x 3 neighbourhood (periodic in longitude) is all
    ``mask``: not on a coast."""
    mask = np.asarray(mask, dtype=bool)
    rows = np.arange(mask.shape[0])
    out = mask.copy()
    for drow in (-1, 0, 1):
        shifted = mask[np.clip(rows + drow, 0, mask.shape[0] - 1)]
        for dcol in (-1, 0, 1):
            out &= np.roll(shifted, dcol, axis=1)
    return out


def _mean(values, mask, area) -> float:
    weights = np.where(mask, area, 0.0)
    return round(float(np.sum(weights * np.where(mask, values, 0.0)) / np.sum(weights)), 4)


def _ice_figures(values, mask, area, latitude_deg) -> dict:
    """Extent at 0.15 and 0.5 and ice area, per hemisphere, in 10^6 km2."""
    lat = np.broadcast_to(np.asarray(latitude_deg)[:, None], np.shape(values))
    ice = np.where(mask, values, 0.0)
    out = {}
    for name, side in (("nh", lat > 0.0), ("sh", lat < 0.0)):
        for label, threshold in (("extent_015", EXTENT_THRESHOLD), ("extent_050", 0.5)):
            out[f"{name}_{label}"] = round(float(np.sum(
                np.where(side & mask & (ice >= threshold), area, 0.0))) / 1.0e6, 4)
        out[f"{name}_area"] = round(float(np.sum(np.where(side, ice * area, 0.0))) / 1.0e6, 4)
    return out


def regrid_water_plane(values, source_water, latitude, longitude, grid, water):
    """One source plane onto ``grid``'s ``water`` columns by the cold start's
    open-water rule; every other column NaN.  Returns ``(plane, record)``;
    the record adds ``source_stencil_all_water``, the columns whose bilinear
    stencil holds no source land."""
    from . import analysis_initial

    regrid = analysis_initial._global_regridder(latitude, longitude, grid)
    land = (~np.asarray(source_water, dtype=bool)).astype(np.float64)
    out, record = analysis_initial.open_water_skin_temperature(
        np.where(source_water, values, 0.0), land, regrid, water,
        latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg)
    out = np.where(water, out, np.nan)
    record = {key: record[key] for key in (
        "open_water_columns", "stencil_with_analysis_land", "searched_columns",
        "search_passes", "unreached_columns_kept_plain_regrid")}
    if record["unreached_columns_kept_plain_regrid"] or not np.isfinite(out[water]).all():
        raise ValueError("the open-water search left water columns without a source "
                         "value; the source has no water near them")
    record["source_stencil_all_water"] = regrid(1.0 - land) >= 1.0 - 1.0e-12
    return out, record


def _check(plane, values, source_water, lat, lon, built, record, grid, ocean, area,
           ice_now) -> dict:
    """One month of one plane taken twice: each grid over its own water, and
    both over the intersection of the two masks (model ocean whose source
    stencil is all water; source water whose nearest model column is one of
    those).  The SST comparison also leaves out coastal and ice-covered
    points, so what remains between the two intersection figures is the
    regrid."""
    source_area = _source_areas_km2(lat, lon)
    model = ocean & record["source_stencil_all_water"]
    if plane == "sst_k":
        source_ice, model_ice = ice_now
        model = model & _interior(ocean) & (model_ice < EXTENT_THRESHOLD)
        source = source_water & (source_ice < EXTENT_THRESHOLD) & _at_source(model, lat, lon, grid)
        return {"own": {"source": _mean(values, source_water, source_area),
                        "model": _mean(built, ocean, area)},
                "intersection": {"source": _mean(values, source, source_area),
                                 "model": _mean(built, model, area)},
                "searched_columns": record["searched_columns"]}
    source = source_water & _at_source(model, lat, lon, grid)
    if plane == "sea_ice_fraction":
        return {"own": {"source": _ice_figures(values, source_water, source_area, lat),
                        "model": _ice_figures(built, ocean, area, grid.latitude_deg)},
                "intersection": {"source": _ice_figures(values, source, source_area, lat),
                                 "model": _ice_figures(built, model, area, grid.latitude_deg)},
                "searched_columns": record["searched_columns"]}
    return {"own": {"source": _mean(values, source_water, source_area),
                    "model": _mean(built, ocean, area)},
            "intersection": {"source": _mean(values, source, source_area),
                             "model": _mean(built, model, area)},
            "searched_columns": record["searched_columns"]}


def build_climatology(grid, ocean, *, row_ids=None, cache=None, table=None,
                      progress=print) -> tuple[dict, dict]:
    """``(planes, provenance)`` on ``grid`` (a ``GaussianGrid``) over its
    ``ocean`` columns (bool, ``(nlat, nlon)``: the statics' water columns
    that are not inland lakes).

    ``planes`` maps each of :data:`PLANES` the rows supply to a float32
    ``(12, nlat, nlon)`` array, January first, NaN off the ocean (land and
    lakes: a lake's skin follows its own energy budget and gets no
    climatology).  The provenance carries every input's name, byte count and
    SHA-256, the rule, and per month the check figures of :func:`_check`:
    the water-mean SST, each hemisphere's ice extent at 0.15 and 0.5 and its
    ice area, and the mean mixed-layer depth, each on the source grid and on
    the model grid, over each grid's own water and over the two masks'
    intersection.  The two "own" figures differ by the coastlines and the
    regrid, the two "intersection" figures by the regrid alone.
    """
    rows = resolve_rows(load_sources(table), row_ids)
    ocean = np.asarray(ocean, dtype=bool)
    if ocean.shape != (grid.nlat, grid.nlon):
        raise ValueError(f"ocean mask {ocean.shape} is not the grid's {(grid.nlat, grid.nlon)}")
    decoded, inputs = _source_planes(rows, cache)
    missing = [plane for plane in REQUIRED_PLANES if plane not in decoded]
    if missing:
        raise ValueError(f"the rows {[row['id'] for row in rows]} give no {missing[0]}")
    area = _model_areas_km2(grid)
    planes, months = {}, [{"month": month} for month in range(1, 13)]
    # The ice first: the SST check leaves out the points either grid calls
    # ice-covered that month.
    order = ("sea_ice_fraction", "sst_k", "mixed_layer_depth_m")
    for plane in order:
        if plane not in decoded:
            continue
        started = time.perf_counter()
        out = np.empty((12, grid.nlat, grid.nlon), dtype=np.float32)
        for month in range(1, 13):
            values, source_water = _source_values(plane, month, decoded)
            _v, lat, lon = decoded[plane]["months"][month]
            built, record = regrid_water_plane(values, source_water, lat, lon, grid, ocean)
            if plane == "sea_ice_fraction":
                built = np.where(ocean, np.clip(built, 0.0, 1.0), np.nan)
            out[month - 1] = built.astype(np.float32)
            ice_now = None
            if plane == "sst_k":
                ice_now = (_source_values("sea_ice_fraction", month, decoded)[0],
                           planes["sea_ice_fraction"][month - 1].astype(np.float64))
            months[month - 1][plane] = _check(
                plane, values, source_water, lat, lon,
                out[month - 1].astype(np.float64), record, grid, ocean, area, ice_now)
        planes[plane] = out
        progress(f"climatology build: {plane} on T{grid.truncation} {grid.nlat}x{grid.nlon} "
                 f"({time.perf_counter() - started:.1f} s)")
    provenance = {
        "schema": CLIMATOLOGY_SCHEMA,
        "rows": [{"id": row["id"], "period": row.get("period"), "licence": row.get("licence"),
                  "citation": row.get("citation")} for row in rows],
        "inputs": inputs,
        "planes": [plane for plane in PLANES if plane in planes],
        "months": "planes 0..11 are January..December",
        "time_interpolation": ("linear between mid-month instants, the midpoint of each real "
                               "calendar month of the valid year (leap Februaries included), "
                               "continuous in seconds"),
        "regrid": ("analysis_initial.open_water_skin_temperature over the statics ocean (water "
                   "columns that are not inland lakes): bilinear weights renormalised over the "
                   "source's own water points, its missing-value mask; a stencil with none "
                   "reads the source water grown outward over the source land; Rust "
                   "cold-start regrid; sea ice clipped to [0, 1]; NaN on land and lakes; the "
                   "mixed-layer depth is not clamped"),
        "units": {"sst_k": "K", "sea_ice_fraction": "1", "mixed_layer_depth_m": "m"},
        "truncation": int(grid.truncation), "nlat": int(grid.nlat), "nlon": int(grid.nlon),
        "ocean_columns": int(np.count_nonzero(ocean)),
        "ocean_mask_sha256": hashlib.sha256(np.ascontiguousarray(ocean, dtype=np.uint8)
                                            .tobytes()).hexdigest(),
        "check": months,
        "check_rule": ("per month, each figure on the source grid and on the model grid, "
                       "over each grid's own water ('own') and over the intersection "
                       "('intersection': model ocean whose bilinear source stencil is all "
                       "source water, and the source water whose nearest model column is one "
                       "of those); the SST intersection also leaves out model columns next to "
                       "land or a lake and points with ice >= 0.15 on either grid; extents at "
                       "0.15 and 0.5 and ice areas in 10^6 km2 per hemisphere"),
        "claim": ("a 1991-2020 (SST, sea ice) or 2005-2017 (mixed layer) monthly mean on the "
                  "model grid; no year's observation, and no forecast"),
    }
    return planes, provenance


def write_climatology(path, planes: dict, *, ocean, latitude_deg, longitude_deg,
                      provenance: dict, lake=None) -> str:
    """Write a climatology file byte-reproducibly; return its SHA-256.

    ``ocean`` is the mask the planes are defined on; ``lake`` (optional)
    the inland-lake columns, carried so a reader can tell a lake from land.
    Members carry a fixed timestamp and Unix attributes, and the provenance
    is canonical JSON, so the same planes and provenance are the same bytes
    on any machine.  The file is written beside its target and moved into
    place, so a reader never sees half of one.
    """
    for plane in REQUIRED_PLANES:
        if plane not in planes:
            raise ValueError(f"a climatology file carries {plane}")
    shape = np.shape(planes["sst_k"])
    if len(shape) != 3 or shape[0] != 12:
        raise ValueError(f"climatology planes are (12, nlat, nlon), not {shape}")
    arrays = {plane: np.ascontiguousarray(planes[plane], dtype=np.float32)
              for plane in PLANES if plane in planes}
    for plane, value in arrays.items():
        if value.shape != shape:
            raise ValueError(f"{plane} is {value.shape}, the SST is {shape}")
    arrays["ocean"] = np.ascontiguousarray(ocean, dtype=np.uint8)
    arrays["lake"] = np.ascontiguousarray(
        np.zeros(shape[1:], dtype=bool) if lake is None else lake, dtype=np.uint8)
    arrays["latitude_deg"] = np.ascontiguousarray(latitude_deg, dtype=np.float64)
    arrays["longitude_deg"] = np.ascontiguousarray(longitude_deg, dtype=np.float64)
    arrays["provenance"] = np.array(json.dumps(provenance, sort_keys=True,
                                               separators=(",", ":")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=_ZIP_EPOCH)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_STORED
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    partial.replace(path)
    return sha256_file(path)


def statics_ocean(fields: dict, provenance: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(ocean, lake)`` from real statics: water where LANDMASK < 0.5 (the
    side of one half the runtime's land fraction is held to), lakes by the
    runtime's own rule (:func:`arwen_global.statics.lake_columns` on the land
    fraction and the lake class's LANDUSEF, as ``resolve_surface_statics``
    derives them), and the ocean the water that is not a lake."""
    from .statics import CategoryConvention, consistent_land_fraction, lake_columns

    convention = CategoryConvention.from_landuse_attrs(provenance["landuse"])
    luf = np.asarray(fields["LANDUSEF"], dtype=np.float64)
    landmask = np.asarray(fields["LANDMASK"], dtype=np.float64)
    water_fraction = luf[convention.water_category - 1] + luf[convention.lake_category - 1]
    land_fraction = consistent_land_fraction(1.0 - water_fraction, landmask > 0.5)
    lake_fraction = np.clip(luf[convention.lake_category - 1], 0.0, 1.0)
    water = landmask < 0.5
    lake = np.asarray(lake_columns(land_fraction, lake_fraction), dtype=bool) & water
    return water & ~lake, lake


def default_output(cfg, grid, *, mixed_layer: bool, cache=None) -> Path:
    tokens = cfg.statics.tokens()
    stem = (f"arwen-global-surface-climatology-T{int(grid.truncation)}-{int(grid.nlat)}x"
            f"{int(grid.nlon)}-{'+'.join(tokens)}{'-mld' if mixed_layer else ''}")
    return Path(cache or default_cache_dir()) / f"{stem}.npz"


def build_for_config(cfg, *, with_mld: bool = False, cache=None, out=None, table=None,
                     progress=print) -> dict:
    """``gpuwm-global climatology build CONFIG``: the config's grid and its
    real statics' ocean, the default rows (plus the WOA18 mixed layer with
    ``with_mld``), written to ``out`` or beside the staged sources."""
    from .spectral.grid import GaussianGrid
    from .statics import load_statics

    if cfg.statics.source != "real":
        raise ValueError(
            "a surface climatology is built on the real statics' water mask, and this "
            f"config's statics are {cfg.statics.source!r}; a synthetic planet has no "
            "geography for a climatology to sit on (a test writes one with "
            "surface_climatology.write_climatology)")
    grid = GaussianGrid.create(int(cfg.truncation), dealias_factor=float(cfg.dealias_factor))
    fields, statics_provenance = load_statics(cfg.statics, grid)
    ocean, lake = statics_ocean(fields, statics_provenance)
    started = time.perf_counter()
    planes, provenance = build_climatology(
        grid, ocean, row_ids=row_ids_for(with_mld, table), cache=cache, table=table,
        progress=progress)
    provenance["statics"] = {
        "rule": ("ocean: water (LANDMASK < 0.5, the side of one half the runtime's land "
                 "fraction is held to) that is not an inland lake by statics.lake_columns"),
        "npz_sha256": statics_provenance.get("npz_sha256"),
        "geog_data_res": cfg.statics.geog_data_res,
        "lake_columns": int(np.count_nonzero(lake)),
    }
    path = Path(out) if out else default_output(cfg, grid, mixed_layer=with_mld, cache=cache)
    digest = write_climatology(path, planes, ocean=ocean, lake=lake,
                               latitude_deg=grid.latitude_deg,
                               longitude_deg=grid.longitude_deg, provenance=provenance)
    seconds = time.perf_counter() - started
    for line in check_lines(provenance):
        progress("climatology build: " + line)
    size = path.stat().st_size
    progress(f"climatology build: {path} ({size:,} B, SHA-256 {digest}, {seconds:.1f} s)")
    return {"path": str(path), "sha256": digest, "bytes": size, "seconds": round(seconds, 2),
            "provenance": provenance}


def check_lines(provenance: dict) -> list[str]:
    """The provenance's check figures as one line per month: each pair is
    source / model; ext = extent, in 10^6 km2."""
    lines = ["check, each pair source / model; 'own' over each grid's own water, 'int' "
             "over the two masks' intersection (SST also off the coasts and the ice); "
             "extent (ext) and ice area in 10^6 km2"]

    def pair(entry, side, key=None, fmt=".3f"):
        values = entry[side]
        a, b = ((values["source"], values["model"]) if key is None
                else (values["source"][key], values["model"][key]))
        return f"{a:{fmt}}/{b:{fmt}}"

    for row in provenance["check"]:
        sst, ice = row["sst_k"], row["sea_ice_fraction"]
        parts = [f"month {row['month']:2d}",
                 f"SST own {pair(sst, 'own')} int {pair(sst, 'intersection')} K"]
        for hemi in ("nh", "sh"):
            parts.append(
                f"{hemi.upper()} ext.15 own {pair(ice, 'own', hemi + '_extent_015')} "
                f"int {pair(ice, 'intersection', hemi + '_extent_015')}, "
                f"ext.5 int {pair(ice, 'intersection', hemi + '_extent_050')}, "
                f"area int {pair(ice, 'intersection', hemi + '_area')}")
        if "mixed_layer_depth_m" in row:
            parts.append(f"MLD own {pair(row['mixed_layer_depth_m'], 'own', fmt='.1f')} m")
        lines.append("; ".join(parts))
    return lines


# ---------------------------------------------------------------------------
# read and interpolate
# ---------------------------------------------------------------------------

@dataclass
class SurfaceClimatology:
    path: Path
    sha256: str
    sst_k: np.ndarray
    sea_ice_fraction: np.ndarray
    mixed_layer_depth_m: np.ndarray | None
    ocean: np.ndarray
    lake: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    provenance: dict

    def check_grid(self, latitude_deg, longitude_deg) -> None:
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        if lat.shape != self.latitude_deg.shape or lon.shape != self.longitude_deg.shape \
                or not np.allclose(lat, self.latitude_deg, atol=1.0e-9) \
                or not np.allclose(lon, self.longitude_deg, atol=1.0e-9):
            raise ValueError(
                f"climatology {self.path} is on T{self.provenance.get('truncation')} "
                f"{self.latitude_deg.size}x{self.longitude_deg.size}, not this run's "
                f"{lat.size}x{lon.size} grid; build it for this config with "
                "`gpuwm-global climatology build`")

    def at(self, plane: str, when, xp=np):
        """``plane`` interpolated to ``when`` (see :func:`interpolate`)."""
        return interpolate(getattr(self, plane), when, xp=xp)


def load_climatology(path) -> SurfaceClimatology:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        if provenance.get("schema") != CLIMATOLOGY_SCHEMA:
            raise ValueError(f"{path} is not a {CLIMATOLOGY_SCHEMA} file")
        arrays = {name: np.array(data[name]) for name in data.files if name != "provenance"}
    ocean = arrays["ocean"].astype(bool)
    shape = (12, *ocean.shape)
    for plane in PLANES:
        if plane not in arrays:
            if plane in REQUIRED_PLANES:
                raise ValueError(f"{path} carries no {plane}")
            continue
        value = arrays[plane]
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"{path}: {plane} is {value.dtype} {value.shape}, "
                             f"not float32 {shape}")
        if not np.isfinite(value[:, ocean]).all():
            raise ValueError(f"{path}: {plane} is not finite on every ocean column")
    return SurfaceClimatology(
        path=path, sha256=sha256_file(path), sst_k=arrays["sst_k"],
        sea_ice_fraction=arrays["sea_ice_fraction"],
        mixed_layer_depth_m=arrays.get("mixed_layer_depth_m"), ocean=ocean,
        lake=arrays["lake"].astype(bool),
        latitude_deg=arrays["latitude_deg"], longitude_deg=arrays["longitude_deg"],
        provenance=provenance)


def _utc(when) -> datetime:
    from .surface_series import _utc as series_utc

    return series_utc(when)


def mid_month(year: int, month: int) -> datetime:
    """The midpoint of a real calendar month, UTC (Feb 2028: 14.5 days in)."""
    days = calendar.monthrange(int(year), int(month))[1]
    return datetime(int(year), int(month), 1, tzinfo=timezone.utc) + timedelta(days=days / 2.0)


def month_weights(when) -> tuple[int, int, float]:
    """``(i0, i1, w)``: the value at ``when`` is ``(1 - w) * plane[i0] + w *
    plane[i1]``, planes indexed 0..11 for January..December.  ``w`` is the
    fraction of the way, in seconds, from the mid-month instant at or before
    ``when`` to the next one, across a year end as well."""
    when = _utc(when)
    year, month = when.year, when.month
    here = mid_month(year, month)
    if when >= here:
        a = (year, month)
        b = (year + 1, 1) if month == 12 else (year, month + 1)
    else:
        a = (year - 1, 12) if month == 1 else (year, month - 1)
        b = (year, month)
    t0, t1 = mid_month(*a), mid_month(*b)
    w = (when - t0).total_seconds() / (t1 - t0).total_seconds()
    return a[1] - 1, b[1] - 1, float(w)


def interpolate(planes, when, xp=np):
    """The twelve-plane field at ``when``, in the planes' own precision, by
    the same expression the dated series interpolates its records with."""
    i0, i1, w = month_weights(when)
    a, b = xp.asarray(planes[i0]), xp.asarray(planes[i1])
    dtype = a.dtype.type
    return a * dtype(1.0 - w) + b * dtype(w)


__all__ = [
    "CLIMATOLOGY_SCHEMA", "ClimatologyPinError", "ClimatologyUnavailable",
    "SOURCES_SCHEMA", "SurfaceClimatology", "build_climatology", "build_for_config",
    "check_lines", "climatology_bytes", "download_bytes", "fetch", "interpolate",
    "load_climatology", "load_sources", "locate", "mid_month", "month_weights",
    "resolve_rows", "statics_ocean", "verify", "write_climatology",
]
