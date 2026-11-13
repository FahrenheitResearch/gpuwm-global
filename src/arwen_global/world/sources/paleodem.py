"""The Pangea map reader: the PALEOMAP paleo-elevation maps.

Source: Scotese, C. R. and Wright, N. M. (2018), PALEOMAP Paleodigital
Elevation Models (PaleoDEMS) for the Phanerozoic, Zenodo record 5460860,
licensed CC BY 4.0.  The record holds the maps at 1 degree as netCDF in one
zip of 9,302,291 bytes; that zip is the one this reader fetches.

What is inside the files (read from the archive, not from the record page):

* netCDF4 (HDF5), written by GMT 5.4.1 ``grdsample -I1d``;
* one elevation variable named ``z`` (long name ``z``, no unit; the record
  says metres), float32 with NaN as its fill value, on ``(lat, lon)``;
* ``lat`` -90 to 90 and ``lon`` -180 to 180, both every degree, gridline
  registered: 181 rows from south to north and 361 columns, the last column
  repeating the first at the date line;
* 109 maps from 0 to 540 Ma.  The record page says 117; the archive's own
  name says "Maps 1-88", and the half numbers (``Map43.5``) make up the
  difference with 88.  The age in a file name is the nominal one; the
  ``history`` attribute can name the age the map was reconstructed for
  (``Map49 ... 250Ma.nc`` was resampled from ``... 252Ma.nc``), and the
  reader carries both.  83 of the 109 differ that way, by up to 3.2 Ma.

Land is every point strictly above sea level (0 m, or the world's
``level_m``).  Measured on the 0 Ma map against this model's own T255 Earth
statics, the map sampled bilinearly at the model's points and shares
weighted by the cosine of latitude: the two land masks agree on 97.0 % of
the globe, and on 98.4 % between 60S and 60N.  The largest single part of
the rest (1.25 % of the globe) is Antarctica, where under Earth's ice sheet
the 0 Ma map lies below sea level (median -356 m): it carries the rock, not
the ice surface.

Nothing is bundled in the wheel.  ``fetch`` downloads the zip on request
into its own new, empty folder, checks the record's size and md5, and
unpacks only the ``.nc`` maps and the licence file.  The reader never runs
anything it downloaded; it opens the maps as data.

Command line (``python -m arwen_global.world.sources.paleodem``)::

    fetch FOLDER [--from-zip ZIP]       # FOLDER must be new or empty
    list FOLDER
    describe FOLDER --age 250           # prints which map was chosen
    compare FOLDER --age 0 --statics STATICS.npz
                                        # land mask against an Earth cache
    make-world FOLDER --age 250 --world pangea-250ma.world
                                        # the chosen map into a world folder

The credit line (``CREDIT``, ``PaleoMap.credit_line()``) travels with every
map read here so the world statement can print it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request
import zipfile

import numpy as np

RECORD_DOI = "10.5281/zenodo.5460860"
RECORD_URL = "https://zenodo.org/records/5460860"
ARCHIVE_NAME = "Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc.zip"
ARCHIVE_URL = f"https://zenodo.org/api/records/5460860/files/{ARCHIVE_NAME}/content"
#: Size and md5 as the record's file listing states them.
ARCHIVE_BYTES = 9_302_291
ARCHIVE_MD5 = "77147998623ab039d86ff3e0b5e40344"
LICENCE = "CC-BY-4.0"
CREDIT = (
    "Elevation map: Scotese, C. R. and Wright, N. M. (2018), PALEOMAP "
    "Paleodigital Elevation Models (PaleoDEMS) for the Phanerozoic, Zenodo, "
    f"https://doi.org/{RECORD_DOI}, licensed CC BY 4.0")

SOURCE_RECORD = "paleodem-source.json"
SOURCE_SCHEMA = "arwen-global.world.paleodem-source/v1"
MAPS_DIR = "maps"
LICENCE_MEMBER = "License.txt"
#: No member of the archive is near this; a larger one is not a map.
MAX_MEMBER_BYTES = 64 * 2**20

MAP_NAME = re.compile(
    r"^Map(?P<number>\d+(?:\.\d+)?)_PALEOMAP_1deg_(?P<period>.+)_"
    r"(?P<age>\d+(?:\.\d+)?)Ma\.nc$")
HISTORY_AGE = re.compile(r"_(?P<age>\d+(?:\.\d+)?)Ma\.nc")
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
#: Names taken as the elevation variable, in order of preference.
ELEVATION_NAMES = ("z", "elevation", "elev", "topo", "topography", "height",
                   "hgt", "band1")
METRE_UNITS = ("m", "meter", "meters", "metre", "metres")
#: A request older than the oldest map by more than this is refused.
AGE_REACH_MA = 5.0


class PaleoDEMError(ValueError):
    """A map that cannot be fetched, chosen or read, with the reason."""


@dataclass(frozen=True)
class MapEntry:
    """One map file, named the way the archive names them."""

    number: str
    period: str
    age_ma: float
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def title(self) -> str:
        return f"Map{self.number} {self.period.replace('_', ' ')}, {self.age_ma:g} Ma"


@dataclass(frozen=True, eq=False)
class PaleoMap:
    """An elevation map on a regular latitude-longitude grid.

    Rows run south to north; columns run east from ``lon_deg[0]`` in
    [-180, 180) and cover the globe once (a repeated date-line column is
    dropped on reading).  ``elevation_m`` is float64 metres above present
    sea level, negative under the sea.
    """

    name: str
    age_ma: float | None
    history_age_ma: float | None
    period: str
    variable: str
    units_note: str
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    elevation_m: np.ndarray
    sha256: str
    seam_repeated: bool
    seam_max_diff_m: float | None

    # -- the grid ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation_m.shape

    def area_weights(self) -> np.ndarray:
        """Each point's share of the globe's area (they sum to 1).

        A point owns the band half a spacing either side of it, clipped at
        the poles, so a gridline-registered pole row owns half a band.
        """
        lat = self.lat_deg
        half = abs(float(lat[1] - lat[0])) / 2.0
        north = np.radians(np.minimum(lat + half, 90.0))
        south = np.radians(np.maximum(lat - half, -90.0))
        row = np.sin(north) - np.sin(south)
        nlon = self.lon_deg.size
        weights = np.repeat((row / (row.sum() * nlon))[:, None], nlon, axis=1)
        return weights

    def land_mask(self, sea_level_m: float = 0.0) -> np.ndarray:
        """Land is every point strictly above ``sea_level_m``."""
        return self.elevation_m > float(sea_level_m)

    def land_share(self, sea_level_m: float = 0.0) -> float:
        """The share of the globe's area that is land."""
        return float(np.sum(self.area_weights() * self.land_mask(sea_level_m)))

    def elevation_range(self) -> tuple[float, float]:
        return float(self.elevation_m.min()), float(self.elevation_m.max())

    def grid_text(self) -> str:
        lat, lon = self.lat_deg, self.lon_deg
        text = (f"{lat.size} x {lon.size} points, {abs(lat[1] - lat[0]):g} x "
                f"{lon[1] - lon[0]:g} degrees; latitude {lat[0]:g} to {lat[-1]:g} "
                f"(rows south to north); longitude {lon[0]:g} to {lon[-1]:g} east")
        if self.seam_repeated:
            text += (f"; the file's repeated {lon[0] + 360.0:g} column was dropped "
                     f"(largest difference from the {lon[0]:g} column: "
                     f"{self.seam_max_diff_m:g} m)")
        return text

    # -- sampling ---------------------------------------------------------
    def sample(self, lat_deg, lon_deg) -> np.ndarray:
        """Bilinear elevation at the given points (any longitude convention).

        Latitudes beyond the first or last row take that row.  This is a
        point sampler for checks; averaging onto a model grid belongs to the
        terrain package.
        """
        lat, lon = np.broadcast_arrays(np.asarray(lat_deg, dtype=np.float64),
                                       np.asarray(lon_deg, dtype=np.float64))
        nlat, nlon = self.shape
        fy = np.interp(lat, self.lat_deg, np.arange(nlat, dtype=np.float64))
        y0 = np.clip(np.floor(fy).astype(np.int64), 0, nlat - 2)
        ty = fy - y0
        dlon = 360.0 / nlon
        fx = np.mod(lon - self.lon_deg[0], 360.0) / dlon
        fx0 = np.floor(fx)
        tx = fx - fx0
        x0 = fx0.astype(np.int64) % nlon
        x1 = (x0 + 1) % nlon
        z = self.elevation_m
        south = (1.0 - tx) * z[y0, x0] + tx * z[y0, x1]
        north = (1.0 - tx) * z[y0 + 1, x0] + tx * z[y0 + 1, x1]
        return (1.0 - ty) * south + ty * north

    # -- what travels to the world statement --------------------------------
    def credit_line(self) -> str:
        age = "" if self.age_ma is None else f", {self.age_ma:g} Ma"
        return f"{CREDIT}; map {self.name}{age}"

    def provenance(self) -> dict:
        """A record for a world folder or receipt: no host path in it."""
        return {
            "source": "PALEOMAP PaleoDEM (Scotese and Wright 2018)",
            "record_doi": RECORD_DOI,
            "licence": LICENCE,
            "credit": CREDIT,
            "map": self.name,
            "age_ma": self.age_ma,
            "history_age_ma": self.history_age_ma,
            "variable": self.variable,
            "sha256": self.sha256,
        }

    def describe(self, sea_level_m: float = 0.0) -> str:
        low, high = self.elevation_range()
        if self.age_ma is None:
            age = "age not in the file name"
        else:
            age = f"{self.age_ma:g} Ma (file name)"
            if self.history_age_ma is not None and self.history_age_ma != self.age_ma:
                age += f"; its history line names {self.history_age_ma:g} Ma"
        title = self.period.replace("_", " ") if self.period else self.name
        lines = [
            f"map:        {title}, {age}",
            f"file:       {self.name}  sha256 {self.sha256}",
            f"variable:   {self.variable}; units: {self.units_note}",
            f"grid:       {self.grid_text()}",
            f"elevation:  {low:g} to {high:g} m",
            (f"land share: {self.land_share(sea_level_m):.4f} of the globe's area "
             f"above {float(sea_level_m):g} m"),
            f"credit:     {self.credit_line()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch(folder, *, url: str = ARCHIVE_URL, md5: str | None = ARCHIVE_MD5,
          size: int | None = ARCHIVE_BYTES, timeout_s: float = 300.0,
          log=print) -> Path:
    """Download the map archive into ``folder``, which must be new or empty.

    The download is checked against the record's size and md5 before
    anything is unpacked.  Only the ``.nc`` maps and the licence file are
    unpacked, flat, into ``folder/maps``; GPlates caches and the rest stay in
    the zip.  ``folder/paleodem-source.json`` records where the bytes came
    from, the credit line and each map's sha256.
    """
    folder = Path(folder)
    if folder.exists():
        if not folder.is_dir():
            raise PaleoDEMError(f"{folder} is a file, not a folder")
        if any(folder.iterdir()):
            raise PaleoDEMError(
                f"{folder} is not empty: the maps are fetched only into their "
                "own new, empty folder")
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / ARCHIVE_NAME
    part = folder / (ARCHIVE_NAME + ".part")
    digest_md5 = hashlib.md5(usedforsecurity=False)
    digest_sha = hashlib.sha256()
    count = 0
    log(f"fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response, \
                part.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                count += len(chunk)
                if size is not None and count > size:
                    raise PaleoDEMError(
                        f"the download is larger than the record's {size} bytes")
                digest_md5.update(chunk)
                digest_sha.update(chunk)
                out.write(chunk)
        if size is not None and count != size:
            raise PaleoDEMError(
                f"the download is {count} bytes; the record says {size}")
        if md5 is not None and digest_md5.hexdigest() != md5:
            raise PaleoDEMError(
                f"the download's md5 is {digest_md5.hexdigest()}; the record "
                f"says {md5}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(archive)
    maps = _unpack(archive, folder / MAPS_DIR)
    record = {
        "schema": SOURCE_SCHEMA,
        "record_doi": RECORD_DOI,
        "record_url": RECORD_URL,
        "archive": ARCHIVE_NAME,
        "archive_bytes": count,
        "archive_md5": digest_md5.hexdigest(),
        "archive_sha256": digest_sha.hexdigest(),
        "licence": LICENCE,
        "credit": CREDIT,
        "maps": maps,
    }
    (folder / SOURCE_RECORD).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"fetched {count} bytes, md5 {digest_md5.hexdigest()} as the record "
        f"states; {len(maps)} maps in {folder / MAPS_DIR}")
    return folder


def _unpack(archive: Path, dest: Path) -> dict[str, str]:
    dest.mkdir()
    maps: dict[str, str] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = PurePosixPath(info.filename.replace("\\", "/")).name
            if not (base.endswith(".nc") or base == LICENCE_MEMBER):
                continue
            if base in ("", ".", "..") or base.startswith(".") or "\x00" in base:
                raise PaleoDEMError(f"refused archive member name {info.filename!r}")
            if base in seen:
                raise PaleoDEMError(f"the archive holds {base} twice")
            seen.add(base)
            if info.file_size > MAX_MEMBER_BYTES:
                raise PaleoDEMError(
                    f"{base} claims {info.file_size} bytes; no map is that large")
            digest = hashlib.sha256()
            total = 0
            with zf.open(info) as src, (dest / base).open("wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_MEMBER_BYTES:
                        raise PaleoDEMError(f"{base} unpacks past {MAX_MEMBER_BYTES} bytes")
                    digest.update(chunk)
                    out.write(chunk)
            if base.endswith(".nc"):
                maps[base] = digest.hexdigest()
    if not maps:
        raise PaleoDEMError(f"{archive.name} holds no .nc map")
    return dict(sorted(maps.items()))


# ---------------------------------------------------------------------------
# Choose
# ---------------------------------------------------------------------------
def parse_map_name(path) -> MapEntry | None:
    """The archive's file-name pattern, or None for any other file."""
    path = Path(path)
    match = MAP_NAME.match(path.name)
    if match is None:
        return None
    return MapEntry(number=match["number"], period=match["period"],
                    age_ma=float(match["age"]), path=path)


def list_maps(folder) -> list[MapEntry]:
    """Every map under ``folder`` (searched recursively), youngest first."""
    folder = Path(folder)
    if not folder.is_dir():
        raise PaleoDEMError(
            f"no map folder at {folder}: fetch one with `python -m "
            "arwen_global.world.sources.paleodem fetch <new empty folder>`")
    entries: dict[str, MapEntry] = {}
    for path in sorted(folder.rglob("*.nc")):
        entry = parse_map_name(path)
        if entry is None:
            continue
        if entry.name in entries:
            raise PaleoDEMError(
                f"{entry.name} is in {folder} twice; keep one copy")
        entries[entry.name] = entry
    if not entries:
        raise PaleoDEMError(f"no PALEOMAP map (MapNN_PALEOMAP_1deg_..._NNMa.nc) under {folder}")
    return sorted(entries.values(), key=lambda e: (e.age_ma, float(e.number)))


def choose_map(entries, age_ma: float) -> MapEntry:
    """The map nearest ``age_ma``; on a tie, the younger one."""
    entries = list(entries)
    if not entries:
        raise PaleoDEMError("there are no maps to choose from")
    age = float(age_ma)
    if not math.isfinite(age) or age < 0.0:
        raise PaleoDEMError(f"age_ma must be a finite age at or after 0 Ma, not {age_ma!r}")
    oldest = max(e.age_ma for e in entries)
    if age > oldest + AGE_REACH_MA:
        raise PaleoDEMError(
            f"no map near {age:g} Ma: the oldest map here is {oldest:g} Ma")
    return min(entries, key=lambda e: (abs(e.age_ma - age), e.age_ma))


def open_nearest(folder, age_ma: float, *, variable: str | None = None,
                 log=print) -> PaleoMap:
    """Choose the map nearest ``age_ma`` in ``folder``, say which, read it."""
    entry = choose_map(list_maps(folder), age_ma)
    away = abs(entry.age_ma - float(age_ma))
    log(f"chose {entry.name} for {float(age_ma):g} Ma: {entry.title}"
        + ("" if away == 0.0 else f", {away:g} Ma away (the nearest map)"))
    return read_map(entry, variable=variable)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def read_map(source, *, variable: str | None = None) -> PaleoMap:
    """Read one map file into a :class:`PaleoMap`.

    The elevation variable is ``variable`` when given; otherwise the
    two-dimensional variable on the file's latitude and longitude whose name
    is one of ``ELEVATION_NAMES``, or the only two-dimensional one.
    """
    import netCDF4  # the engine's dependency; only this path needs it

    entry = source if isinstance(source, MapEntry) else parse_map_name(source)
    path = Path(source.path if isinstance(source, MapEntry) else source)
    if not path.is_file():
        raise PaleoDEMError(f"no map file at {path}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with netCDF4.Dataset(path) as ds:
        lat_name = _coordinate(ds, LAT_NAMES, "latitude")
        lon_name = _coordinate(ds, LON_NAMES, "longitude")
        lat_dim = ds.variables[lat_name].dimensions[0]
        lon_dim = ds.variables[lon_name].dimensions[0]
        name = _elevation_variable(ds, lat_dim, lon_dim, variable)
        var = ds.variables[name]
        var.set_auto_mask(False)
        z = np.array(var[:], dtype=np.float64)
        if var.dimensions == (lon_dim, lat_dim):
            z = z.T
        units = getattr(var, "units", None)
        history = str(getattr(ds, "history", ""))
        lat = np.array(ds.variables[lat_name][:], dtype=np.float64)
        lon = np.array(ds.variables[lon_name][:], dtype=np.float64)
    z, units_note = _to_metres(z, units, name)
    found = HISTORY_AGE.search(history)
    return build_map(
        lat, lon, z, name=path.name,
        age_ma=None if entry is None else entry.age_ma,
        history_age_ma=None if found is None else float(found["age"]),
        period="" if entry is None else entry.period,
        variable=name, units_note=units_note, sha256=sha)


def _coordinate(ds, names, what: str) -> str:
    for candidate in ds.variables:
        var = ds.variables[candidate]
        if candidate.lower() in names and var.ndim == 1:
            return candidate
    raise PaleoDEMError(
        f"no one-dimensional {what} variable (looked for {', '.join(names)}); "
        f"the file has {', '.join(ds.variables) or 'no variables'}")


def _elevation_variable(ds, lat_dim: str, lon_dim: str, wanted: str | None) -> str:
    grid = {lat_dim, lon_dim}
    candidates = [name for name, var in ds.variables.items()
                  if var.ndim == 2 and set(var.dimensions) == grid]
    if wanted is not None:
        if wanted not in candidates:
            raise PaleoDEMError(
                f"variable {wanted!r} is not a 2-D field on ({lat_dim}, {lon_dim}); "
                f"the 2-D fields are: {', '.join(candidates) or 'none'}")
        return wanted
    for preferred in ELEVATION_NAMES:
        for name in candidates:
            if name.lower() == preferred:
                return name
    if len(candidates) == 1:
        return candidates[0]
    raise PaleoDEMError(
        "cannot tell which variable is the elevation: "
        + (f"the 2-D fields are {', '.join(candidates)}; name one with variable="
           if candidates else f"no 2-D field on ({lat_dim}, {lon_dim})"))


def _to_metres(z: np.ndarray, units, name: str) -> tuple[np.ndarray, str]:
    if units is None or not str(units).strip():
        return z, "metres (the file states no unit; the record describes metres)"
    text = str(units).strip().lower()
    if text in METRE_UNITS:
        return z, f"metres ({units!s})"
    if text in ("km", "kilometre", "kilometres", "kilometer", "kilometers"):
        return z * 1000.0, f"kilometres in the file ({units!s}), read as metres"
    raise PaleoDEMError(f"variable {name!r} is in {units!r}; an elevation in metres is needed")


def build_map(lat, lon, z, *, name: str, age_ma: float | None = None,
              history_age_ma: float | None = None, period: str = "",
              variable: str = "z", units_note: str = "metres",
              sha256: str = "") -> PaleoMap:
    """Put an elevation array into the reader's conventions, or refuse it.

    Rows are flipped to run south to north; a repeated date-line column is
    dropped; longitudes are wrapped into [-180, 180) and rolled to ascend.
    The grid must be regular, cover the globe once in longitude, and hold
    no missing value.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or z.shape != (lat.size, lon.size):
        raise PaleoDEMError(
            f"{name}: elevation {z.shape} does not sit on latitude {lat.shape} "
            f"x longitude {lon.shape}")
    if lat.size < 2 or lon.size < 4:
        raise PaleoDEMError(f"{name}: a {lat.size} x {lon.size} grid is too small")
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        z = z[::-1, :]
    dlat = np.diff(lat)
    if np.any(dlat <= 0) or not np.allclose(dlat, dlat[0], rtol=0, atol=1e-6) \
            or lat[0] < -90.0 - 1e-6 or lat[-1] > 90.0 + 1e-6:
        raise PaleoDEMError(f"{name}: latitude is not a regular grid inside -90..90")
    seam_repeated = False
    seam_diff = None
    if abs((lon[-1] - lon[0]) - 360.0) < 1e-6:
        seam_repeated = True
        seam_diff = float(np.nanmax(np.abs(z[:, -1] - z[:, 0])))
        lon = lon[:-1]
        z = z[:, :-1]
    dlon = np.diff(lon)
    if np.any(dlon <= 0) or not np.allclose(dlon, dlon[0], rtol=0, atol=1e-6) \
            or abs(dlon[0] * lon.size - 360.0) > 1e-4:
        raise PaleoDEMError(
            f"{name}: longitude is not a regular grid covering the globe once")
    bad = int(np.count_nonzero(~np.isfinite(z)))
    if bad:
        raise PaleoDEMError(f"{name}: {bad} points hold no elevation (NaN or infinite)")
    wrapped = np.mod(lon + 180.0, 360.0) - 180.0
    order = np.argsort(wrapped, kind="stable")
    lon = np.round(wrapped[order], 9)
    z = np.ascontiguousarray(z[:, order])
    lat = np.round(lat, 9)
    for array in (lat, lon, z):
        array.setflags(write=False)
    return PaleoMap(name=name, age_ma=age_ma, history_age_ma=history_age_ma,
                    period=period, variable=variable, units_note=units_note,
                    lat_deg=lat, lon_deg=lon, elevation_m=z, sha256=sha256,
                    seam_repeated=seam_repeated, seam_max_diff_m=seam_diff)


# ---------------------------------------------------------------------------
# Into a world folder
# ---------------------------------------------------------------------------
def write_world_map(paleo_map: PaleoMap, world_folder, *, overwrite: bool = False) -> Path:
    """Write ``paleo_map`` as an invented world's map, its provenance as the
    map's source row (so the credit line reaches the world statement).

    The world folder, its lock and its map checks belong to
    ``arwen_global.world.folder`` (package 2.4); this only hands it the map.
    """
    try:
        from arwen_global.world.folder import write_map
    except ImportError as exc:  # the surface package is not in this tree yet
        raise PaleoDEMError(
            "this tree has no arwen_global.world.folder (the invented-world "
            f"folder) to write the map into: {exc}") from None
    return write_map(world_folder, paleo_map.elevation_m, paleo_map.lat_deg,
                     paleo_map.lon_deg, source=paleo_map.provenance(),
                     overwrite=overwrite)


# ---------------------------------------------------------------------------
# Against Earth
# ---------------------------------------------------------------------------
def land_mask_agreement(paleo_map: PaleoMap, earth_landmask, lat_deg, lon_deg,
                        *, sea_level_m: float = 0.0) -> dict:
    """Compare a map's land with an Earth land mask on the Earth grid.

    The map is sampled bilinearly at the Earth points and land is elevation
    above ``sea_level_m``; Earth's land is ``earth_landmask > 0.5``.  Shares
    are weighted by the cosine of latitude.  ``agreement`` is the share of
    the globe where the two say the same thing.
    """
    mask = np.asarray(earth_landmask, dtype=np.float64) > 0.5
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    if lat.shape != mask.shape or lon.shape != mask.shape:
        raise PaleoDEMError(
            f"Earth mask {mask.shape} does not match its coordinates {lat.shape}")
    land = paleo_map.sample(lat, lon) > float(sea_level_m)
    weight = np.cos(np.radians(lat))
    weight = weight / weight.sum()

    def share(selection) -> float:
        return float(np.sum(weight * selection))

    return {
        "points": int(mask.size),
        "agreement": share(land == mask),
        "map_land_share": share(land),
        "earth_land_share": share(mask),
        "map_land_earth_water": share(land & ~mask),
        "map_water_earth_land": share(~land & mask),
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arwen_global.world.sources.paleodem",
        description="Fetch, list and read the PALEOMAP paleo-elevation maps "
                    "(Scotese and Wright 2018, CC BY 4.0).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch", help="download the 1-degree maps into a new, empty folder")
    p_fetch.add_argument("folder")
    p_fetch.add_argument("--from-zip", help="install a copy of the archive downloaded by hand")
    p_list = sub.add_parser("list", help="list the maps in a folder")
    p_list.add_argument("folder")
    p_desc = sub.add_parser("describe", help="choose the map nearest an age and describe it")
    p_desc.add_argument("folder")
    p_desc.add_argument("--age", type=float, required=True, help="millions of years ago")
    p_desc.add_argument("--variable")
    p_desc.add_argument("--sea-level-m", type=float, default=0.0)
    p_cmp = sub.add_parser("compare", help="land mask against an Earth statics cache")
    p_cmp.add_argument("folder")
    p_cmp.add_argument("--age", type=float, default=0.0)
    p_cmp.add_argument("--statics", required=True,
                       help="an Earth statics cache .npz (LANDMASK, latitude_deg, longitude_deg)")
    p_cmp.add_argument("--sea-level-m", type=float, default=0.0)
    p_world = sub.add_parser("make-world", help="write the map nearest an age into a world folder")
    p_world.add_argument("folder")
    p_world.add_argument("--age", type=float, required=True)
    p_world.add_argument("--world", required=True, help="the world folder, e.g. pangea-250ma.world")
    p_world.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            url = ARCHIVE_URL if not args.from_zip else Path(args.from_zip).resolve().as_uri()
            fetch(args.folder, url=url)
            print(CREDIT)
        elif args.command == "list":
            for entry in list_maps(args.folder):
                print(f"{entry.age_ma:7g} Ma  {entry.name}")
        elif args.command == "describe":
            print(open_nearest(args.folder, args.age, variable=args.variable)
                  .describe(args.sea_level_m))
        elif args.command == "compare":
            paleo_map = open_nearest(args.folder, args.age)
            with np.load(args.statics, allow_pickle=False) as cache:
                mask = np.asarray(cache["LANDMASK"])
                lat = np.asarray(cache["latitude_deg"])
                lon = np.asarray(cache["longitude_deg"])
            result = land_mask_agreement(paleo_map, mask, lat, lon,
                                         sea_level_m=args.sea_level_m)
            print(json.dumps(result, indent=2))
        elif args.command == "make-world":
            paleo_map = open_nearest(args.folder, args.age)
            target = write_world_map(paleo_map, args.world, overwrite=args.overwrite)
            print(f"wrote {target}")
            print(paleo_map.credit_line())
    except (ValueError, FileExistsError) as exc:  # PaleoDEMError, a world folder's refusal
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
