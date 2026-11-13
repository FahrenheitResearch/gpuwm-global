"""An invented world's folder: the user's map, its lock file and the
surface caches built from it.

A world with its own surface (``[world.surface] folder = "..."``) keeps
everything it is made of in one folder::

    pangea-250ma.world/
        world-map.npz            the map: elevation_m (nlat, nlon) on a
                                 latitude-longitude grid, with its
                                 latitude_deg and longitude_deg
        world-lock.json          the lock file: the map's sha256, where it
                                 came from, and one row per surface cache
                                 built from it
        arwen-global-statics-T255-384x768-world-<key>.npz   (and .json)
                                 the eleven cached arrays on one model grid

The lock file is what makes the folder one thing.  It pins the map's bytes:
a map edited after a surface was built no longer matches its row and every
read is refused until the surface is rebuilt.  It names every cache built
from that map, so a cache copied in from another folder or left over from
an older map is never read.  Writing the map drops every cache row, since
the caches were built from the map that was there before.

While a build writes the folder it holds ``.world-lock.writing``, created
exclusively; a second writer is refused, and a writer that crashed leaves
the file behind with a message saying how to clear it.

Nothing here reads or writes outside the folder, and nothing here runs for
Earth: a config without ``[world.surface]`` never names a folder.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FOLDER_SCHEMA = "arwen-global.world-folder/v1"
MAP_SCHEMA = "arwen-global.world-map/v1"
MAP_FILE = "world-map.npz"
LOCK_FILE = "world-lock.json"
WRITING_FILE = ".world-lock.writing"
#: The arrays a map file holds, and nothing else.
MAP_ARRAYS = ("elevation_m", "latitude_deg", "longitude_deg")
#: The highest and lowest elevations a map may carry (m).  Earth's surface
#: spans about -11,000 to +8,850 m; a map outside this range is a units
#: error (feet, kilometres) or a fill value, not a world.
MAP_ELEVATION_RANGE_M = (-12000.0, 12000.0)


class WorldFolderError(ValueError):
    """The folder is not a usable world folder; the message says why and
    what to do."""


def resolve_world_folder(folder: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    """The folder a config names, as an absolute path.

    ``~`` is expanded.  A relative folder is taken relative to ``base``
    (the directory of the config that names it, the rule
    ``[surface_update] series`` follows), or to the working directory when
    no base is given.
    """
    path = Path(str(folder)).expanduser()
    if not path.is_absolute():
        path = (Path(base) if base is not None else Path.cwd()) / path
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _now() -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


# -- the write guard --------------------------------------------------------


@contextmanager
def writing(folder: Path):
    """Hold the folder's write guard for the duration of a write."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    guard = folder / WRITING_FILE
    try:
        handle = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise WorldFolderError(
            f"world folder {folder} is being written by another build ({guard} "
            "exists).  If no build is running, one crashed while writing: "
            f"delete {guard} and build again") from None
    try:
        os.write(handle, f"{os.getpid()}\n".encode())
        os.close(handle)
        yield folder
    finally:
        try:
            guard.unlink()
        except FileNotFoundError:
            pass


# -- the lock file ----------------------------------------------------------


def lock_path(folder: Path) -> Path:
    return Path(folder) / LOCK_FILE


def read_lock(folder: Path) -> dict:
    """The folder's lock document, its self-hash verified."""
    path = lock_path(folder)
    if not path.is_file():
        raise WorldFolderError(
            f"world folder {folder} has no {LOCK_FILE}: it holds no map yet.  "
            "Put a map in it first (arwen_global.world.folder.write_map, or the "
            "Pangea map reader), then build its surface")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorldFolderError(f"{path} is not JSON: {exc}") from None
    if not isinstance(document, dict) or document.get("schema") != FOLDER_SCHEMA:
        raise WorldFolderError(f"{path} is not a {FOLDER_SCHEMA} document")
    stated = document.pop("self_sha256", None)
    if stated != hashlib.sha256(_canonical(document)).hexdigest():
        raise WorldFolderError(
            f"{path} self-hash mismatch: the lock file was edited by hand.  "
            "Write the map again to start the folder's lock over")
    document["self_sha256"] = stated
    return document


def _write_lock(folder: Path, document: dict) -> Path:
    path = lock_path(folder)
    body = dict(document)
    body.pop("self_sha256", None)
    body["schema"] = FOLDER_SCHEMA
    body["self_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
    return path


# -- the map ----------------------------------------------------------------


def _strictly_monotonic(values: np.ndarray) -> bool:
    step = np.diff(values)
    return bool(np.all(step > 0.0) or np.all(step < 0.0))


def check_map(elevation_m, latitude_deg, longitude_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three map arrays as float64, or a refusal naming what is wrong."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    if elevation.ndim != 2 or lat.ndim != 1 or lon.ndim != 1:
        raise WorldFolderError(
            "a world map is elevation_m (nlat, nlon) with 1-D latitude_deg and "
            f"longitude_deg; got {elevation.shape}, {lat.shape}, {lon.shape}")
    if elevation.shape != (lat.size, lon.size):
        raise WorldFolderError(
            f"world map elevation_m is {elevation.shape}, its axes are "
            f"({lat.size}, {lon.size})")
    if lat.size < 2 or lon.size < 4:
        raise WorldFolderError("a world map needs at least 2 rows and 4 columns")
    for name, value in (("elevation_m", elevation), ("latitude_deg", lat),
                        ("longitude_deg", lon)):
        if not np.isfinite(value).all():
            raise WorldFolderError(
                f"world map {name} holds non-finite values (a fill value or a "
                "masked cell); fill them before writing the map")
    if lat.min() < -90.0 or lat.max() > 90.0 or not _strictly_monotonic(lat):
        raise WorldFolderError(
            "world map latitude_deg must be strictly monotonic inside [-90, 90]")
    if not _strictly_monotonic(lon) or abs(lon[-1] - lon[0]) >= 360.0:
        raise WorldFolderError(
            "world map longitude_deg must be strictly monotonic and span less "
            "than 360 degrees (the first column is not repeated at the end)")
    lo, hi = MAP_ELEVATION_RANGE_M
    if elevation.min() < lo or elevation.max() > hi:
        raise WorldFolderError(
            f"world map elevation_m spans {elevation.min():g} to {elevation.max():g} "
            f"m, outside [{lo:g}, {hi:g}]: metres are expected (a map in feet or "
            "kilometres, or one carrying a fill value, is not a world)")
    return elevation, lat, lon


def write_map(folder: str | os.PathLike, elevation_m, latitude_deg, longitude_deg, *,
              source: dict | None = None, overwrite: bool = False) -> Path:
    """Write the world's map into ``folder`` and start its lock afresh.

    ``source`` is a JSON object saying where the map came from (for a
    reconstruction: the dataset, the age, the credit line the world
    statement carries).  Every surface cache row is dropped from the lock,
    because they were built from whatever map was there before.
    """
    folder = Path(folder)
    elevation, lat, lon = check_map(elevation_m, latitude_deg, longitude_deg)
    if source is not None:
        _canonical(source)  # refuse a source row that is not plain JSON
    target = folder / MAP_FILE
    with writing(folder):
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"world map {target} exists; pass overwrite=True to replace it "
                "(every surface built from the old map is dropped from the lock)")
        temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, elevation_m=elevation, latitude_deg=lat,
                                longitude_deg=lon)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        land = elevation > 0.0
        _write_lock(folder, {
            "folder": folder.name,
            "map": {
                "schema": MAP_SCHEMA,
                "file": MAP_FILE,
                "sha256": file_sha256(target),
                "shape": [int(lat.size), int(lon.size)],
                "latitude_deg": [float(lat[0]), float(lat[-1])],
                "longitude_deg": [float(lon[0]), float(lon[-1])],
                "elevation_m": [float(elevation.min()), float(elevation.max())],
                "cells_above_sea_level": int(land.sum()),
                "source": source or {},
                "written_at_utc": _now(),
            },
            "surfaces": {},
        })
    return target


def read_map(folder: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """``(elevation_m, latitude_deg, longitude_deg, map_row)``, the map's
    bytes checked against the lock."""
    folder = Path(folder)
    lock = read_lock(folder)
    row = lock["map"]
    path = folder / row["file"]
    if not path.is_file():
        raise WorldFolderError(f"world folder {folder} lost its map {path.name}")
    digest = file_sha256(path)
    if digest != row["sha256"]:
        raise WorldFolderError(
            f"world map {path} is not the map the lock file pins (sha256 "
            f"{digest[:12]}, locked {row['sha256'][:12]}): it was changed after "
            "the folder was written.  Write it again with write_map so the lock "
            "and every surface built from it start over")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if names != set(MAP_ARRAYS):
            raise WorldFolderError(
                f"world map {path} holds {sorted(names)}, expected {list(MAP_ARRAYS)}")
        arrays = [np.array(archive[name], dtype=np.float64) for name in MAP_ARRAYS]
    elevation, lat, lon = check_map(*arrays)
    return elevation, lat, lon, row


def record_surface(folder: Path, key: str, row: dict) -> Path:
    """Add (or replace) the lock row of one built surface cache.  The caller
    holds the write guard."""
    lock = read_lock(folder)
    lock = dict(lock)
    surfaces = dict(lock.get("surfaces", {}))
    surfaces[key] = row
    lock["surfaces"] = surfaces
    return _write_lock(folder, lock)


def folder_record(folder: str | os.PathLike) -> dict:
    """What a receipt records about the folder a run read: its name, the
    lock's self-hash, the map's hash and source and every surface row.  The
    folder's path is left out (a fact about one host)."""
    folder = Path(folder)
    lock = read_lock(folder)
    return {
        "folder": folder.name,
        "lock_sha256": lock["self_sha256"],
        "map_sha256": lock["map"]["sha256"],
        "map_source": lock["map"].get("source", {}),
        "surfaces": lock.get("surfaces", {}),
    }


def world_map_for(world, transform, *, folder=None, base=None):
    """The invented world's terrain and land for its cold start
    (``arwen_global.world.frame.WorldMap``): the map in its folder, its
    bytes checked against the lock, averaged onto ``transform``'s grid and
    spectrally filtered (package 2.5), then sampled on the regular frame
    the cold start's regridder reads.  ``folder`` is the folder as the
    config resolved it (``statics.world_folder_of``); without it the
    world's folder is resolved against ``base``.  The false-height report
    of the filter is printed, as plan section 4.4 asks."""
    from .frame import world_map_from_elevation

    if not getattr(world, "is_invented", False):
        raise ValueError(
            f"world {getattr(world, 'label', world)!r} has no map of its own")
    folder = (Path(folder) if folder is not None
              else resolve_world_folder(world.surface.folder, base))
    elevation, latitude, longitude, _ = read_map(folder)
    world_map, report = world_map_from_elevation(
        world, elevation, latitude, longitude, transform)
    print(f"world {world.label}: dynamics terrain from {folder.name}: {report}",
          flush=True)
    return world_map


__all__ = [
    "FOLDER_SCHEMA", "LOCK_FILE", "MAP_ARRAYS", "MAP_FILE", "MAP_SCHEMA",
    "WRITING_FILE", "WorldFolderError", "check_map", "file_sha256",
    "folder_record", "lock_path", "read_lock", "read_map", "record_surface",
    "resolve_world_folder", "world_map_for", "write_map", "writing",
]
