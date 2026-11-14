"""Sub-grid orography statistics on the Gaussian grid, for the orographic drag.

The KIM orographic gravity-wave drag (WRF ``gwd_opt = 1``, carried in
:mod:`arwen_global.core.terrain_drag`) reads ten static planes per column:
the standard deviation of the sub-grid terrain (VAR, m), its convexity
(CON), its asymmetry in four directions (OA1-OA4) and its effective length
in the same four (OL1-OL4).  WPS geogrid writes them into geo_em from the
WPS_GEOG dataset ``orogwd_10m``, which the engine's ``gpuwm fetch-geog``
stages by default (NCAR's ``orogwd_10m.tar.bz2``, pinned in
``gpuwm.geog_assets``).

WHY THIS MODULE AND NOT THE STATICS CRATE.  The engine builds these fields
through its Rust static-fields crate (``gpuwm.static.orographic``), and that
crate refuses the Gaussian grid: MEASURED 2026-11-11 on gpuwm 2.9.13, the
first sector of a T21 ring handed to ``build_orographic_fields`` came back
``orographic WPS fields need a projected grid``, because the crate's
orographic rows are written for the WPS map projections and the global
model's ``rows`` grid is none of them.  The land-surface statics this model
builds through the same crate are not affected.

So this module reads the dataset itself, and does exactly what
GEOGRID.TBL.ARW's ``default`` rows for these ten fields say: each value is
``average_4pt`` of the source (the mean of the four 10-arc-minute source
points around the column centre), and the rows are ``masked = water`` with
``fill_missing = 0``, so a column the land-surface statics call water
(LANDMASK < 0.5, lakes and ocean alike) carries 0 in every plane and the
drag is off there, as WRF has it.  The source grid is global and periodic
in longitude; the four points wrap across the date line.

STATED, NOT MEASURED: the statistics are those of 10-arc-minute (about
18 km) source cells, sampled at the column centre, exactly as geogrid
samples them for a regional grid of any spacing.  They are not
re-aggregated over the column's own area, so at T255 (52 km) VAR does not
include the terrain variance between the 18 km and 52 km scales.  That is
what a WRF user running ``gwd_opt = 1`` at 50 km with the default
``geog_data_res`` gets, and it is the prescription here.

The planes are cached once per truncation beside the land-surface statics
cache, with the same self-hashed sidecar (:func:`arwen_global.statics.
write_cache`), and the sidecar records the SHA-256 of every source tile and
of the land-surface cache whose LANDMASK masked them.

BUILT WHEN A RUN NEEDS IT (0.2.3).  The shipped T255 and T383 forecast days
turn the drag on, so a machine that built its land-surface statics under
0.2.2 or earlier, and a runner that skips ``statics`` once the land cache is
there, would otherwise meet a refusal on its first 0.2.3 run.
:func:`ensure_orographic_statistics` builds the cache at the door instead,
from the WPS_GEOG tree the land cache was built from, and when that tree
lacks ``orogwd_10m`` it stages the one dataset with the engine's own
``gpuwm fetch-geog`` (NCAR's 3.3 MB archive, SHA-256 pinned by the engine).
Anything it cannot do is one sentence naming the two commands, or the
option that turns the drag off, before the card is touched.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .statics import (STATICS_SCHEMA, StaticsOptions, _array_hash,
                      _file_hash, cache_paths as statics_cache_paths,
                      load_statics, read_cache, machine_class, write_cache)

#: The ten planes the KIM drag reads, in WRF's geo_em names and launch order.
SSO_FIELDS = ("VAR", "CON", "OA1", "OA2", "OA3", "OA4",
              "OL1", "OL2", "OL3", "OL4")
#: GEOGRID.TBL.ARW's ``default`` dataset for these rows.
SSO_DATASET = "orogwd_10m"
#: The sidecar's ``kind``: what tells this cache from the land-surface one.
SSO_KIND = "orographic-statistics"
#: The interpolation and mask GEOGRID.TBL.ARW gives every one of the rows.
SSO_INTERPOLATION = "average_4pt"
SSO_MASK = "water"


def _leaf(name: str) -> str:
    return name.lower()


def geog_root(options: StaticsOptions) -> Path:
    """The WPS_GEOG root: the statics' own choice, else the engine's search.

    Only the root: the land-surface datasets are the land-surface build's
    business, and this build reads one dataset of its own.
    """
    if options.geog_root:
        root = Path(options.geog_root)
        looked = [root]
    else:
        from gpuwm.rustwx_static import geog_root_candidates

        looked = list(geog_root_candidates())
        root = next((path for path in looked if path.is_dir()), None)
    if root is None or not root.is_dir():
        rendered = ", ".join(str(path) for path in looked)
        raise FileNotFoundError(
            f"no WPS_GEOG archive to read {SSO_DATASET} from (looked at "
            f"{rendered}); stage one with `gpuwm fetch-geog` or name it with "
            "--geog-root")
    return root


def cache_paths(options: StaticsOptions, grid) -> tuple[Path, Path]:
    """``(npz, sidecar)``: the land-surface cache's stem plus the dataset."""
    npz, _ = statics_cache_paths(options, grid)
    stem = npz.with_suffix("").name + f"-{SSO_DATASET}"
    return npz.with_name(f"{stem}.npz"), npz.with_name(f"{stem}.json")


def parse_index(path: Path) -> dict[str, str]:
    """A WPS_GEOG ``index`` file as ``key -> value`` strings."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip().lower()] = value.strip().strip('"')
    return out


def read_global_tile(directory: Path) -> tuple[np.ndarray, dict[str, str], str]:
    """One WPS_GEOG field stored as a single global tile.

    Returns ``(values, index, tile_sha256)``: float64 values on the source
    rows (row 0 at ``known_lat``), scaled by the index's ``scale_factor``.
    Anything but the layout ``orogwd_10m`` ships in (one regular lat-lon
    tile covering 360 degrees, two-byte words, no border) is refused by
    name rather than read some other way.
    """
    directory = Path(directory)
    index = parse_index(directory / "index")

    def number(key, default=None):
        if key not in index:
            if default is None:
                raise ValueError(f"{directory}/index has no {key}")
            return default
        return float(index[key])

    if index.get("projection") != "regular_ll":
        raise ValueError(f"{directory}: projection {index.get('projection')!r} "
                         "is not regular_ll")
    if index.get("type") != "continuous":
        raise ValueError(f"{directory}: type {index.get('type')!r} is not continuous")
    nx, ny = int(number("tile_x")), int(number("tile_y"))
    if int(number("wordsize")) != 2 or int(number("tile_bdr", 0.0)) != 0:
        raise ValueError(f"{directory}: only two-byte, border-free tiles are read")
    if abs(nx * number("dx") - 360.0) > 1.0e-3:
        raise ValueError(f"{directory}: {nx} x {number('dx')} deg is not a "
                         "global ring")
    if "missing_value" in index:
        raise ValueError(f"{directory}: a missing_value is not expected in "
                         f"{SSO_DATASET}")
    tile = directory / f"00001-{nx:05d}.00001-{ny:05d}"
    if not tile.is_file():
        raise FileNotFoundError(f"{tile} is missing: {SSO_DATASET} ships as "
                                "one global tile per field")
    signed = index.get("signed", "no").lower() == "yes"
    endian = "<" if index.get("endian", "big").lower() == "little" else ">"
    raw = np.fromfile(tile, dtype=f"{endian}{'i' if signed else 'u'}2")
    if raw.size != nx * ny:
        raise ValueError(f"{tile}: {raw.size} words, expected {nx * ny}")
    values = raw.reshape(ny, nx).astype(np.float64) * number("scale_factor", 1.0)
    return values, index, _file_hash(tile)


def sample_average_4pt(values: np.ndarray, index: dict[str, str],
                       latitude_deg, longitude_deg) -> np.ndarray:
    """geogrid's ``average_4pt`` at each (latitude, longitude) point.

    The mean of the 2 x 2 source points around the point, in one-based
    source coordinates ``x = (lon - known_lon) / dx + known_x`` (wrapped onto
    the ring) and ``y = (lat - known_lat) / dy + known_y``; a coordinate on
    a source node takes that node twice (floor and ceiling agree), and a row
    within half a source cell of the first or last row takes that row, as
    WPS clamps it.
    """
    ny, nx = values.shape
    dx, dy = float(index["dx"]), float(index["dy"])
    known_x, known_y = float(index.get("known_x", 1.0)), float(index.get("known_y", 1.0))
    known_lat, known_lon = float(index["known_lat"]), float(index["known_lon"])
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    x = np.mod((lon - known_lon) / dx, float(nx)) + known_x
    y = (lat - known_lat) / dy + known_y
    if np.any(y < known_y - 0.5) or np.any(y > known_y + ny - 1 + 0.5):
        raise ValueError("a target latitude lies outside the source rows")
    i0 = np.floor(x).astype(np.int64)
    i1 = np.ceil(x).astype(np.int64)
    j0 = np.clip(np.floor(y).astype(np.int64), 1, ny)
    j1 = np.clip(np.ceil(y).astype(np.int64), 1, ny)
    i0 = (i0 - 1) % nx
    i1 = (i1 - 1) % nx
    j0 -= 1
    j1 -= 1
    total = values[j0, i0] + values[j0, i1]
    total = total + values[j1, i0]
    total = total + values[j1, i1]
    return total / 4.0


def build_orographic_statistics(grid, options: StaticsOptions) -> tuple[dict, dict]:
    """The ten planes on ``grid`` and the provenance their sidecar carries.

    The water mask is the land-surface statics cache's LANDMASK for the same
    grid, so that cache must exist (``gpuwm-global statics CONFIG``).
    """
    started = time.perf_counter()
    root = geog_root(options)
    statics_fields, statics_document = load_statics(options, grid)
    water = np.asarray(statics_fields["LANDMASK"], dtype=np.float64) < 0.5
    lat2 = np.broadcast_to(np.asarray(grid.latitude_deg, np.float64)[:, None],
                           (grid.nlat, grid.nlon))
    lon2 = np.broadcast_to(np.asarray(grid.longitude_deg, np.float64)[None, :],
                           (grid.nlat, grid.nlon))
    out: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, object]] = {}
    for name in SSO_FIELDS:
        directory = Path(root) / SSO_DATASET / _leaf(name)
        if not (directory / "index").is_file():
            raise FileNotFoundError(
                f"the WPS_GEOG tree at {root} lacks {SSO_DATASET}/{_leaf(name)}, "
                f"which the orographic drag's {name} is built from; stage it "
                f"with `gpuwm fetch-geog --root {root} --datasets "
                f"{SSO_DATASET} --source {FETCH_SOURCE}` or unpack NCAR's "
                f"{SSO_DATASET}.tar.bz2 under that tree")
        values, index, tile_sha = read_global_tile(directory)
        plane = sample_average_4pt(values, index, lat2, lon2)
        plane = np.where(water, 0.0, plane)
        out[name] = np.ascontiguousarray(plane, dtype=np.float32)
        sources[name] = {
            "dataset": f"{SSO_DATASET}/{_leaf(name)}",
            "tile_sha256": tile_sha,
            "index_sha256": _file_hash(directory / "index"),
            "scale_factor": float(index.get("scale_factor", 1.0)),
        }
    out["latitude_deg"] = np.ascontiguousarray(grid.latitude_deg, dtype=np.float64)
    out["longitude_deg"] = np.ascontiguousarray(grid.longitude_deg, dtype=np.float64)
    provenance = {
        "schema": STATICS_SCHEMA,
        "kind": SSO_KIND,
        "truncation": int(grid.truncation),
        "nlat": int(grid.nlat), "nlon": int(grid.nlon),
        "geog_root": str(root),
        "dataset": SSO_DATASET,
        "interpolation": SSO_INTERPOLATION,
        "masked": SSO_MASK,
        "fill_missing": 0.0,
        "sources": sources,
        "water_mask": {
            "field": "LANDMASK < 0.5",
            "statics_cache_self_sha256": statics_document.get("self_sha256"),
            "water_columns": int(water.sum()),
        },
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "host": machine_class(),
        "wall_seconds": float(time.perf_counter() - started),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str,
                   "sha256": _array_hash(value)}
            for name, value in out.items()
        },
    }
    return out, provenance


def missing_cache_remedy(npz_path: Path) -> str:
    return (
        f"no sub-grid orography cache for this truncation at {npz_path}; "
        "orographic_drag = \"kim-gwdo\" reads VAR, CON, OA1-4 and OL1-4 on "
        "every column, and without them the drag would run on zeros, which "
        "turns it off without saying so.  Build it once for this truncation "
        "(after the land-surface statics, whose land mask it uses):\n"
        "  gpuwm-global statics <this run's config> --orographic")


def load_orographic_statistics(options: StaticsOptions, grid) -> tuple[dict, dict]:
    """The cached planes for ``grid`` (verified), or the refusal with remedy."""
    npz_path, _ = cache_paths(options, grid)
    if not npz_path.is_file():
        raise FileNotFoundError(missing_cache_remedy(npz_path))
    fields, document = read_cache(npz_path, grid)
    if document.get("kind") != SSO_KIND:
        raise ValueError(f"{npz_path} is not a {SSO_KIND} cache")
    missing = [name for name in SSO_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"{npz_path} lacks {', '.join(missing)}")
    return fields, document


#: Where ``gpuwm fetch-geog`` takes ``orogwd_10m`` from: NCAR only (the
#: engine's mirror does not carry it; the engine pins the archive's bytes).
FETCH_SOURCE = "ncar"
#: How long a run waits for another process's build of the same cache.
LOCK_WAIT_S = 900.0
#: How long the engine's fetch of the one 3.3 MB archive may take.
FETCH_TIMEOUT_S = 900.0


def fetch_command(root: Path) -> list[str]:
    """The engine's own fetch of the one dataset, into ``root``."""
    return [sys.executable, "-m", "gpuwm", "fetch-geog", "--root", str(root),
            "--datasets", SSO_DATASET, "--source", FETCH_SOURCE]


def fetch_dataset(root: Path) -> None:
    """Stage ``orogwd_10m`` under ``root`` with ``gpuwm fetch-geog``.

    A subprocess of the installed engine's command, so the archive's pinned
    size and SHA-256, the geog root's lock and the fetch manifest are the
    engine's, exactly as ``gpuwm fetch-geog`` typed by hand would leave them.
    """
    command = fetch_command(root)
    done = subprocess.run(command, capture_output=True, text=True,
                          timeout=FETCH_TIMEOUT_S, check=False)
    if done.returncode != 0:
        lines = (done.stderr or done.stdout or "").strip().splitlines()
        raise OSError(f"`gpuwm {' '.join(command[3:])}` exited {done.returncode}"
                      + (f": {lines[-1]}" if lines else ""))


def _land_cache_geog_root(options, grid) -> Path | None:
    """The WPS_GEOG root the land-surface cache for ``grid`` was built from."""
    _, sidecar = statics_cache_paths(options, grid)
    try:
        root = json.loads(sidecar.read_text(encoding="utf-8")).get("geog_root")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(root) if root else None


def _dataset_present(root: Path) -> bool:
    return all((Path(root) / SSO_DATASET / _leaf(name) / "index").is_file()
               for name in SSO_FIELDS)


def unbuildable_sentence(npz_path: Path, error: BaseException) -> str:
    """The one line a run that cannot build the cache stops with."""
    reason = " ".join(f"{type(error).__name__}: {error}".split())
    return (
        f"orographic_drag = \"kim-gwdo\" needs the sub-grid orography cache "
        f"{npz_path}, and it could not be built here ({reason}); stage the "
        f"data with `gpuwm fetch-geog --datasets {SSO_DATASET} --source "
        f"{FETCH_SOURCE}` (add --root to name your WPS_GEOG tree) and build it "
        "with `gpuwm-global statics CONFIG --orographic`, or run without the "
        "drag by setting orographic_drag = \"none\" in [physics] "
        "native_adapter_options")


def _wait_for_other_build(npz_path: Path, sidecar: Path, lock: Path,
                          say) -> None:
    say(f"gpuwm-global: another process is building {npz_path.name}; "
        "waiting for it")
    deadline = time.monotonic() + LOCK_WAIT_S
    while lock.exists() and time.monotonic() < deadline:
        time.sleep(1.0)
    if npz_path.is_file() and sidecar.is_file():
        return
    if lock.exists():
        raise FileNotFoundError(unbuildable_sentence(npz_path, TimeoutError(
            f"the build lock {lock} is still there after {LOCK_WAIT_S:.0f} s; "
            "remove it if no other run is building the cache")))
    raise FileNotFoundError(unbuildable_sentence(npz_path, RuntimeError(
        "the other process stopped without writing it")))


def _say(line: str) -> None:
    print(line, flush=True)


def ensure_orographic_statistics(options: StaticsOptions, grid, *, say=_say,
                                 fetch=fetch_dataset) -> Path | None:
    """Build the drag's cache for ``grid`` now if it is not on disk.

    Returns the cache it wrote, or None when there was nothing to do: the
    cache is already there, or the land-surface cache it is masked with is
    not (that cache's own refusal, which names ``gpuwm-global statics``, is
    the first thing such a run must hear).  The WPS_GEOG tree read is the
    config's ``geog_root``, else the tree the land-surface cache records it
    was built from, else the engine's search; when that tree lacks
    ``orogwd_10m`` the one dataset is staged into it by ``fetch`` (the
    engine's ``gpuwm fetch-geog``; None stages nothing).  One process
    builds at a time: the others wait for its lock, so the ranks of a
    multi-card run do not race on one file.  Any failure is
    :func:`unbuildable_sentence`, one line, raised as FileNotFoundError.
    """
    npz_path, sidecar = cache_paths(options, grid)
    if npz_path.is_file() and sidecar.is_file():
        return None
    land_npz, land_sidecar = statics_cache_paths(options, grid)
    if not (land_npz.is_file() and land_sidecar.is_file()):
        return None
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    lock = npz_path.with_name(f".{npz_path.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _wait_for_other_build(npz_path, sidecar, lock, say)
        return None
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        started = time.perf_counter()
        try:
            if options.geog_root:
                root = Path(options.geog_root)
            else:
                root = _land_cache_geog_root(options, grid)
                if root is None or not root.is_dir():
                    root = geog_root(options)
            if not _dataset_present(root):
                if fetch is None:
                    raise FileNotFoundError(
                        f"the WPS_GEOG tree at {root} has no {SSO_DATASET}")
                say(f"gpuwm-global: the orographic drag reads WPS_GEOG "
                    f"{SSO_DATASET}, which {root} lacks; staging it with "
                    f"`gpuwm fetch-geog --datasets {SSO_DATASET} --source "
                    f"{FETCH_SOURCE}` (3.3 MB, SHA-256 pinned by the engine)")
                fetch(root)
            built = dataclasses.replace(options, geog_root=str(root))
            fields, provenance = build_orographic_statistics(grid, built)
            write_cache(npz_path, fields, provenance)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            raise FileNotFoundError(unbuildable_sentence(npz_path, error)) from error
        say(f"gpuwm-global: built the orographic drag's sub-grid orography "
            f"for T{int(grid.truncation)} {int(grid.nlat)}x{int(grid.nlon)} "
            f"from {root} in {time.perf_counter() - started:.1f} s: {npz_path}")
        return npz_path
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def orographic_statistics_summary(fields: dict, document: dict) -> dict:
    """The receipt row: which cache, and what it holds over land."""
    var = np.asarray(fields["VAR"], dtype=np.float64)
    land = var > 0.0
    return {
        "kind": SSO_KIND,
        "dataset": document.get("dataset"),
        "cache_self_sha256": document.get("self_sha256"),
        "npz_sha256": document.get("npz_sha256"),
        "columns_with_statistics": int(land.sum()),
        "var_max_m": float(var.max()) if var.size else 0.0,
        "var_mean_over_those_m": (float(var[land].mean()) if land.any()
                                  else 0.0),
    }


__all__ = [
    "SSO_DATASET", "SSO_FIELDS", "SSO_KIND", "build_orographic_statistics",
    "cache_paths", "ensure_orographic_statistics", "fetch_command", "fetch_dataset",
    "load_orographic_statistics", "missing_cache_remedy",
    "orographic_statistics_summary", "read_global_tile", "sample_average_4pt",
    "unbuildable_sentence",
]
