"""Dated surface forcing on the Gaussian grid: SST, sea ice and greenness.

Until this module the global model held the sea surface temperature and
the sea ice at their cold-start values for the whole run (the ocean has no
model here: Noah skips open water and the surface layer reads its skin as
the SST).  For a ten-day forecast that is the largest fixed error of the
lower boundary.  hex closed the same gap on the MPAS mesh with MPAS-A's
``sfc_update`` services (``hexcore.mpasphys.surface_update_v842``, from
``mpas_atmphys_update_surface.F``): a timed series of SST and sea-ice
records read at the record interval, and the monthly greenness and
background albedo re-resolved to the date every day.  This is the same
service on the Gaussian grid, as one table-driven step:

* :func:`build_surface_series` produces the series.  The sources are ROWS
  of ``data/surface_series_sources.json`` (a URL template, the GRIB2 index
  names of the four records, the mapping whose selectors decode them), so
  a new source is a new row, not a new code path.  Each record is fetched
  by byte range (four messages, a few MB, not the 480 MB product),
  decoded by the engine's Rust GRIB2 stack and regridded by the cold
  start's own Rust door with the cold start's own open-water rule
  (:func:`arwen_global.analysis_initial.open_water_skin_temperature`), so
  record zero of a ``gdas-analysis`` series is the cold-start SST.
* :class:`SurfaceSeriesUpdate` applies it every step, on the surface
  state's own arrays (device, or the pinned host tier when parked):

  - **SST.** Linear in time between records, written on every open-ocean
    column (water, not frozen, not an inland lake: a lake's skin follows
    its own energy budget).  MPAS writes the latest record at the record
    alarm; a daily step change of up to a kelvin is a shock a 300 s step
    does not need, so the record is interpolated (GFS interpolates its
    daily SST the same way).
  - **Sea ice.** At each record time the fraction and thickness are
    replaced.  A column that crosses the threshold changes its whole
    surface class through the cold start's own rulebook
    (:func:`arwen_global.statics.resolve_surface_statics` at the record's
    valid time, applied on the crossing columns only), as MPAS's
    ``physics_update_sst`` converts in place.  The runtime's per-band
    frozen, partial-pack and lake masks are then rebuilt.
  - **Greenness.** Once per model day the vegetation fraction, leaf area
    index and background albedo on land are the monthly climatologies of
    the statics cache at the date (WRF real's ``monthly_interp_to_date``;
    MPAS ``config_greeness_update = 24:00:00``).

Stated scope, not ported: MPAS's SST skin (``sstskin``) diurnal warm layer
and the deep-soil lag mean (``deepsoiltemp``); both matter over months.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np

SERIES_SCHEMA = "gpuwm.arwen-global-surface-series/v1"
SOURCES_SCHEMA = "gpuwm.arwen-global-surface-series-sources/v1"
SOURCES_TABLE = Path(__file__).parent / "data" / "surface_series_sources.json"
RECORD_FIELDS = ("skin_temperature", "land_fraction", "sea_ice_fraction",
                 "sea_ice_thickness")
#: Freezing point of sea water the frozen-surface column holds its ice
#: bottom at (physics/frozen_surface); a column that freezes over starts
#: its skin no warmer than this.
SEA_WATER_FREEZING_K = 271.35


def _utc(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    if len(text) == 10 and text.isdigit():  # YYYYMMDDHH
        return datetime.strptime(text, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    if len(text) == 13 and text[10] == "T":  # YYYY-MM-DDTHH
        text += ":00"
    out = datetime.fromisoformat(text)
    return out if out.tzinfo else out.replace(tzinfo=timezone.utc)


def load_sources(path=None) -> dict:
    table = json.loads(Path(path or SOURCES_TABLE).read_text(encoding="utf-8"))
    if table.get("schema") != SOURCES_SCHEMA:
        raise ValueError(f"surface-series source table schema is not {SOURCES_SCHEMA}")
    return table


def resolve_source(table: dict, source_id: str) -> dict:
    rows = {row["id"]: row for row in table["rows"]}
    if source_id not in rows:
        raise ValueError(f"no surface-series source {source_id!r}; the table has "
                         + ", ".join(sorted(rows)))
    row = rows[source_id]
    missing = [name for name in RECORD_FIELDS if name not in row.get("records", {})]
    if missing:
        raise ValueError(f"source row {source_id!r} names no index record for {missing}")
    return row


def record_url(row: dict, *, cycle: datetime, valid: datetime) -> str:
    lead = int(round((valid - cycle).total_seconds() / 3600.0))
    return row["url"].format(cycle=cycle, valid=valid, lead=lead)


def _http(url: str, *, byte_range=None, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        request = urllib.request.Request(url)
        if byte_range is not None:
            first, last = byte_range
            tail = "" if last is None else str(last)
            request.add_header("Range", f"bytes={first}-{tail}")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_record(row: dict, url: str, destination: Path) -> dict:
    """The four GRIB2 messages of one record, by byte range off the index."""
    index = _http(url + row.get("index_suffix", ".idx")).decode("utf-8").splitlines()
    entries = []
    for line in index:
        parts = line.split(":")
        if len(parts) >= 5:
            entries.append((int(parts[1]), f"{parts[3]}:{parts[4]}"))
    wanted = {}
    for field_name, record in row["records"].items():
        positions = [i for i, (_, name) in enumerate(entries) if name == record]
        if len(positions) != 1:
            raise ValueError(f"{url}: the index lists {len(positions)} messages "
                             f"{record!r} for {field_name}; exactly one is required")
        i = positions[0]
        first = entries[i][0]
        last = entries[i + 1][0] - 1 if i + 1 < len(entries) else None
        wanted[field_name] = (first, last)
    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        chunks = dict(zip(wanted, pool.map(
            lambda span: _http(url, byte_range=span), wanted.values())))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        for name in RECORD_FIELDS:
            out.write(chunks[name])
    payload = destination.read_bytes()
    return {"url": url, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _selectors(mapping_id: str) -> list:
    """The decode selectors of the four records, from the mapping authority
    the cold start decodes the same fields with (one source of truth)."""
    from .analysis_initial import resolve_analysis_mapping

    mapping = json.loads(Path(resolve_analysis_mapping(mapping_id)).read_text())
    select = []
    for name in RECORD_FIELDS:
        rows = mapping["fields"][name]["selectors"]
        if len(rows) != 1:
            raise ValueError(f"mapping {mapping_id} has {len(rows)} selectors for {name}")
        sel = rows[0]
        select.append({
            "key": name, "discipline": int(sel["discipline"]),
            "category": int(sel["category"]), "parameter": int(sel["parameter"]),
            "constituent_type": None, "aerosol_type": None,
            "level_type": int(sel["level_type"]), "levels": None,
        })
    return select


def decode_record(path: Path, mapping_id: str, workdir: Path):
    """``(fields, latitude, longitude, valid_time)`` from the Rust stack."""
    from gpuwm.grib2_stack import stack_grib2

    out = Path(tempfile.mkdtemp(prefix="stack-", dir=workdir)) / "decoded"
    stacked = {item.key: item for item in stack_grib2(path, _selectors(mapping_id), out)}
    reference = stacked[RECORD_FIELDS[0]]
    fields = {}
    for name in RECORD_FIELDS:
        item = stacked[name]
        if item.latitude.shape != reference.latitude.shape or \
                item.longitude.shape != reference.longitude.shape:
            raise ValueError(f"{path}: {name} is on another grid than the skin temperature")
        fields[name] = np.array(item.array[0], dtype=np.float64)
    return (fields, np.asarray(reference.latitude), np.asarray(reference.longitude),
            reference.valid_time)


@dataclass
class SeriesGrid:
    truncation: int
    nlat: int
    nlon: int
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray


def series_grid(cfg) -> SeriesGrid:
    from .spectral.grid import GaussianGrid

    grid = GaussianGrid.create(int(cfg.truncation), dealias_factor=float(cfg.dealias_factor))
    return SeriesGrid(int(cfg.truncation), int(grid.nlat), int(grid.nlon),
                      np.asarray(grid.latitude_deg, dtype=np.float64),
                      np.asarray(grid.longitude_deg, dtype=np.float64))


def build_surface_series(cfg, *, source: str, cycle, hours: float,
                         interval_hours: float, out, table=None, workdir=None,
                         progress=print) -> dict:
    """Fetch, decode and regrid one record every ``interval_hours`` from
    ``cycle`` through ``cycle + hours`` onto ``cfg``'s Gaussian grid."""
    from . import analysis_initial
    from .statics import load_statics

    row = resolve_source(load_sources(table), source)
    cycle = _utc(cycle)
    count = int(round(hours / interval_hours)) + 1
    valids = [cycle + timedelta(hours=interval_hours * k) for k in range(count)]
    grid = series_grid(cfg)
    statics_fields, _ = load_statics(cfg.statics, grid)
    water = np.asarray(statics_fields["LANDMASK"], dtype=np.float64) < 0.5
    out = Path(out)
    work = Path(workdir) if workdir else out.parent / (out.stem + ".work")
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def fetch(k):
        url = record_url(row, cycle=cycle, valid=valids[k])
        return fetch_record(row, url, work / f"record-{k:03d}.grib2")

    with ThreadPoolExecutor(max_workers=min(8, count)) as pool:
        fetched = list(pool.map(fetch, range(count)))
    fetch_s = time.perf_counter() - started

    def decode(k):
        return decode_record(work / f"record-{k:03d}.grib2", row["mapping"], work)

    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, count, 12)) as pool:
        decoded = list(pool.map(decode, range(count)))
    sst = np.empty((count, grid.nlat, grid.nlon), dtype=np.float32)
    ice = np.empty_like(sst)
    thickness = np.empty_like(sst)
    lat0, lon0 = decoded[0][1], decoded[0][2]
    regrid = analysis_initial._global_regridder(lat0, lon0, grid)

    def place(k):
        fields, lat, lon, valid_time = decoded[k]
        if lat.shape != lat0.shape or lon.shape != lon0.shape:
            raise ValueError(f"record {k} is on another source grid than record 0")
        vt = _utc(valid_time)
        if vt != valids[k]:
            raise ValueError(f"record {k} decodes valid {vt.isoformat()}, "
                             f"expected {valids[k].isoformat()}")
        skin, _ = analysis_initial.open_water_skin_temperature(
            fields["skin_temperature"], fields["land_fraction"], regrid, water,
            latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg)
        sst[k] = skin
        ice[k] = np.clip(regrid(fields["sea_ice_fraction"]), 0.0, 1.0)
        thickness[k] = np.maximum(regrid(fields["sea_ice_thickness"]), 0.0)
        return {"valid_utc": vt.isoformat().replace("+00:00", "Z"), **fetched[k],
                "open_water_sst_k_mean": round(float(skin[water].mean()), 4),
                "sea_ice_columns_ge_half": int(np.count_nonzero((ice[k] >= 0.5) & water))}

    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, count, 12)) as pool:
        records = list(pool.map(place, range(count)))
    provenance = {
        "schema": SERIES_SCHEMA, "source": row["id"], "source_kind": row["kind"],
        "cycle_utc": cycle.isoformat().replace("+00:00", "Z"),
        "interval_s": float(interval_hours) * 3600.0,
        "truncation": grid.truncation, "nlat": grid.nlat, "nlon": grid.nlon,
        "records": records, "fetch_s": round(fetch_s, 2),
        "build_s": round(time.perf_counter() - started, 2),
        "regrid": "analysis_initial.open_water_skin_temperature on the statics water mask, "
                  "Rust cold-start regrid; sea ice bilinear, clipped",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, sst_k=sst, sea_ice_fraction=ice, sea_ice_thickness_m=thickness,
             valid_utc=np.array([r["valid_utc"] for r in records]),
             provenance=np.array(json.dumps(provenance)))
    import shutil

    shutil.rmtree(work, ignore_errors=True)
    progress(f"surface-series: {count} records {records[0]['valid_utc']} .. "
             f"{records[-1]['valid_utc']} on T{grid.truncation} {grid.nlat}x{grid.nlon} "
             f"-> {out} ({provenance['build_s']} s, fetch {provenance['fetch_s']} s)")
    return provenance


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

class SurfaceSeriesUpdate:
    """Apply a dated surface series to a run's surface state every step."""

    def __init__(self, path, *, start_time, duration_s: float,
                 statics_fields=None, statics_provenance=None,
                 terrain_height_m=None, latitude_deg=None):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.sst = np.ascontiguousarray(data["sst_k"], dtype=np.float32)
            self.ice = np.ascontiguousarray(data["sea_ice_fraction"], dtype=np.float32)
            self.thickness = np.ascontiguousarray(data["sea_ice_thickness_m"],
                                                  dtype=np.float32)
            self.valid = [_utc(str(v)) for v in data["valid_utc"]]
            self.provenance = json.loads(str(data["provenance"]))
        if self.provenance.get("schema") != SERIES_SCHEMA:
            raise ValueError(f"{self.path} is not a {SERIES_SCHEMA} series")
        self.start = _utc(start_time)
        self.times_s = np.array([(v - self.start).total_seconds() for v in self.valid])
        end_s = float(duration_s)
        if len(self.times_s) < 2 or self.times_s[0] > 0.0 or self.times_s[-1] < end_s:
            raise ValueError(
                f"surface series {self.path} covers {self.valid[0].isoformat()} .. "
                f"{self.valid[-1].isoformat()}, not the run's {self.start.isoformat()} .. "
                f"{(self.start + timedelta(seconds=end_s)).isoformat()}: the SST past "
                "its last record would be invented")
        if np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("surface series record times must increase")
        self.statics_fields = statics_fields
        self.statics_provenance = statics_provenance
        self.terrain = terrain_height_m
        self.latitude_deg = latitude_deg
        self.greenness = statics_fields is not None
        self._ice_record = None
        self._device = {}
        self._green_day = None
        self._ocean = None
        self.counts = {"sst_steps": 0, "ice_records": 0, "ice_to_water": 0,
                       "water_to_ice": 0, "greenness_updates": 0}
        self.seconds = 0.0
        self.last = {}

    @staticmethod
    def _xp(array):
        """The device module when the run is on a card (a parked plane is
        staged to the card to be worked on), numpy otherwise."""
        from .spill import spilled

        try:
            import cupy as cp
        except ImportError:  # pragma: no cover
            return np
        return cp if spilled(array) else cp.get_array_module(array)

    @staticmethod
    def _get(xp, surface, name):
        from .spill import resident

        return resident(xp, getattr(surface, name))

    @staticmethod
    def _put(surface, name, value) -> None:
        """Write a whole plane back: in place on the card, or the parked
        slot replaced through the tier (spill.SpilledArray.store), as the
        water fixer replaces its reservoir."""
        from .spill import spilled

        target = getattr(surface, name)
        if spilled(target):
            target.store(value)
        else:
            target[...] = value

    def _record_on(self, xp, k):
        key = (xp.__name__, k)
        if key not in self._device:
            if len(self._device) >= 4:
                self._device.clear()
            self._device[key] = xp.asarray(self.sst[k])
        return self._device[key]

    def _open_ocean(self, surface, xp):
        from .statics import lake_columns, water_columns

        if self._ocean is None:
            land = self._get(xp, surface, "land_fraction")
            ice = self._get(xp, surface, "sea_ice_fraction")
            water = water_columns(land, xp, sea_ice_fraction=ice)
            lake = lake_columns(land, self._get(xp, surface, "lake_fraction"), xp,
                                sea_ice_fraction=ice)
            self._ocean = xp.ascontiguousarray(water & ~lake)
        return self._ocean

    def apply(self, state, *, model=None) -> None:
        started = time.perf_counter()
        surface = state.surface
        xp = self._xp(surface.temperature_k)
        t = float(state.time_s)
        k = int(np.searchsorted(self.times_s, t, side="right") - 1)
        k = min(max(k, 0), len(self.times_s) - 2)
        if self._ice_record != k:
            # Record zero at a cold start is the analysis the state was
            # seeded from; replacing it with itself is the identity.
            self._apply_ice(state, model, k, xp)
            self._ice_record = k
        if self.greenness:
            day = int(t // 86400.0)
            if self._green_day != day:
                self._apply_greenness(surface, xp, self.start + timedelta(seconds=t))
                self._green_day = day
        w = (t - self.times_s[k]) / (self.times_s[k + 1] - self.times_s[k])
        w = float(min(max(w, 0.0), 1.0))
        a, b = self._record_on(xp, k), self._record_on(xp, k + 1)
        skin = self._get(xp, surface, "temperature_k")
        sst = (a * np.float32(1.0 - w) + b * np.float32(w)).astype(skin.dtype, copy=False)
        ocean = self._open_ocean(surface, xp)
        self._put(surface, "temperature_k", xp.where(ocean, sst, skin))
        self.counts["sst_steps"] += 1
        self.last = {"time_s": t, "record": k, "weight": round(w, 6)}
        self.seconds += time.perf_counter() - started

    def _apply_greenness(self, surface, xp, when) -> None:
        from gpuwm.static.build import monthly_interp_to_date

        f = self.statics_fields
        valid = when.replace(tzinfo=None)
        planes = {
            "vegetation_fraction": np.clip(monthly_interp_to_date(
                np.asarray(f["GREENFRAC"], dtype=np.float64), valid), 0.0, 1.0),
            "leaf_area_index": np.clip(monthly_interp_to_date(
                np.asarray(f["LAI12M"], dtype=np.float64), valid), 0.0, None),
            "background_albedo": np.clip(monthly_interp_to_date(
                np.asarray(f["ALBEDO12M"], dtype=np.float64), valid) / 100.0, 0.0, 1.0),
        }
        land = self._get(xp, surface, "land_fraction") > 0.5
        for name, plane in planes.items():
            current = self._get(xp, surface, name)
            self._put(surface, name, xp.where(
                land, xp.asarray(plane.astype(current.dtype)), current))
        self.counts["greenness_updates"] += 1

    def _apply_ice(self, state, model, k, xp) -> None:
        from .statics import SEA_ICE_THRESHOLD

        surface = state.surface
        ice_now = self._get(xp, surface, "sea_ice_fraction")
        dtype = ice_now.dtype
        new_ice = xp.asarray(self.ice[k].astype(dtype))
        new_thick = xp.asarray(self.thickness[k].astype(dtype))
        water = self._get(xp, surface, "land_fraction") <= 0.5
        threshold = np.float32(SEA_ICE_THRESHOLD)
        old_frozen = xp.asarray(ice_now, dtype=xp.float32) >= threshold
        new_frozen = xp.asarray(new_ice, dtype=xp.float32) >= threshold
        freeze = water & new_frozen & ~old_frozen
        melt = water & old_frozen & ~new_frozen
        n_freeze = int(xp.count_nonzero(freeze))
        n_melt = int(xp.count_nonzero(melt))
        if (n_freeze or n_melt) and self.statics_fields is not None:
            self._convert(surface, xp, k, new_ice, freeze, melt)
        elif n_freeze or n_melt:
            # Without the statics there is no rulebook to re-class with;
            # the crossing columns keep their class and their old ice.
            new_ice = xp.where(freeze | melt, ice_now, new_ice)
            n_freeze = n_melt = 0
        self._put(surface, "sea_ice_fraction", xp.where(water, new_ice, ice_now))
        self._put(surface, "sea_ice_thickness_m", xp.where(
            water, new_thick, self._get(xp, surface, "sea_ice_thickness_m")))
        self.counts["ice_records"] += 1
        self.counts["water_to_ice"] += n_freeze
        self.counts["ice_to_water"] += n_melt
        if n_freeze or n_melt:
            self._ocean = None
            invalidate_runtime_masks(model)

    def _convert(self, surface, xp, k, new_ice, freeze, melt) -> None:
        from .statics import resolve_surface_statics
        from .surface_seeding import sea_ice_initial_soil_temperature

        def host(a):
            return a.get() if hasattr(a, "get") else np.asarray(a)

        skin = host(self._get(xp, surface, "temperature_k")).astype(np.float64)
        soil = host(self._get(xp, surface, "soil_temperature_k")).astype(np.float64)
        sst = self.sst[k].astype(np.float64)
        ice = host(new_ice).astype(np.float32)
        melt_h, freeze_h = host(melt), host(freeze)
        skin = np.where(melt_h, sst,
                        np.where(freeze_h, np.minimum(skin, SEA_WATER_FREEZING_K), skin))
        resolved, _ = resolve_surface_statics(
            self.statics_fields, self.statics_provenance,
            valid_time=self.valid[k].replace(tzinfo=None),
            latitude_deg=self.latitude_deg, terrain_height_m=self.terrain,
            soil_temperature_k=soil, skin_temperature_k=skin,
            sea_ice_fraction=ice, snow_water_kg_m2=np.zeros_like(ice))
        mask = xp.asarray(freeze_h | melt_h)
        for name in ("landuse_category", "soil_category_top", "soil_category_bottom",
                     "albedo", "emissivity", "roughness_m", "snow_albedo"):
            current = self._get(xp, surface, name)
            self._put(surface, name, xp.where(
                mask, xp.asarray(resolved[name].astype(current.dtype)), current))
        new_soil = sea_ice_initial_soil_temperature(
            soil, skin, ice.astype(np.float64), self.thickness[k].astype(np.float64),
            np.zeros_like(skin))
        new_soil = np.where(melt_h[None], sst[None], new_soil)
        current = self._get(xp, surface, "soil_temperature_k")
        self._put(surface, "soil_temperature_k", xp.where(
            mask[None], xp.asarray(new_soil.astype(current.dtype)), current))
        current = self._get(xp, surface, "temperature_k")
        self._put(surface, "temperature_k", xp.where(
            mask, xp.asarray(skin.astype(current.dtype)), current))

    def receipt(self) -> dict:
        return {
            "series": str(self.path), "source": self.provenance.get("source"),
            "source_kind": self.provenance.get("source_kind"),
            "cycle_utc": self.provenance.get("cycle_utc"),
            "records": len(self.valid), "interval_s": self.provenance.get("interval_s"),
            "sst": "linear in time between records on open-ocean columns "
                   "(water, not frozen, not an inland lake)",
            "sea_ice": "replaced at each record; crossing columns re-classed by the "
                       "cold-start rulebook",
            "greenness": ("vegetation, LAI and background albedo on land re-resolved daily "
                          "from the statics monthly climatology" if self.greenness
                          else "off (synthetic statics)"),
            **self.counts, "host_seconds": round(self.seconds, 3), "last": self.last,
        }


def invalidate_runtime_masks(model) -> int:
    """Drop the native runtime's per-band frozen, partial-pack and lake
    masks and its statics check, which it otherwise keeps for the run."""
    physics = getattr(model, "physics", None) if model is not None else None
    runtime = getattr(physics, "_runtime", None)
    bands = getattr(runtime, "_bands", None) or {}
    for cache in bands.values():
        cache.frozen = cache.frozen_count = None
        cache.partial = cache.partial_count = None
        cache.lake = cache.lake_count = None
        cache.statics_checked = False
    return len(bands)


__all__ = ["SERIES_SCHEMA", "SurfaceSeriesUpdate", "build_surface_series",
           "invalidate_runtime_masks", "load_sources", "resolve_source"]
