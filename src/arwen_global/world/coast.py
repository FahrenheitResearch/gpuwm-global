"""A world's own coast, as the map folder the engine's renderer reads.

The engine's renderer (``rw_wrfbatch``, behind ``gpuwm-global render``)
draws its land fill and its coastline from a map folder of Natural Earth
shapefiles.  It takes that folder from ``RUSTWX_BASEMAP_DIR`` before any
other place, so a world whose land mask is not Earth's can be drawn with
its own coast by writing a folder in the same layout from the world's mask
and pointing the variable at it.  Nothing in the engine changes.

THE THREE RULES (plan section 4.10), each read from the renderer's source
(``rustwx-render/src/features.rs``) and each kept by :func:`write_map_folder`
and refused by :func:`check_map_folder`:

1. The folder holds ALL THREE sub-folders the renderer looks for:
   ``natural_earth_10m``, ``natural_earth_110m`` and ``us_counties_5m``.
   The renderer resolves each sub-folder on its own, first match winning
   over a list of roots that ends in the engine's own Earth assets, so a
   sub-folder missing here is taken from Earth's and Earth's coast,
   borders or counties are drawn on the world.
2. The ocean polygon is never empty.  When a folder's land AND ocean
   polygons are both empty the renderer falls through to the next
   resolution and then to a cartopy shapefile cache in the user's home
   directory, which is Earth's land.  The ocean here is always the whole
   globe, drawn first, with the world's land filled over it.
3. The first render of a world is checked on a zoomed picture, where the
   renderer draws its finest (10m) coastline.  A FOURTH trap was found
   while reading for this one: at that zoom, when every line layer the
   folder supplies is empty (a waterworld has no coast), the renderer
   falls back to the cartopy cache's Earth coastline.  So a folder whose
   world has no coast carries one zero-length line in its lakes layer,
   which keeps the layer list non-empty and draws nothing
   (:data:`EMPTY_LINEWORK_SENTINEL`).

THE OUTLINE.  The coast is the boundary between land and water cells of the
mask, drawn along the cell edges (half-way between grid points), so it lies
within half a grid cell of any coast the mask could have come from.  The
land fill is the same boundary as closed rings, all in one polygon record:
the renderer fills a polygon even-odd, so islands in lakes in continents
come out right whatever the order of the rings.  Edges that are not coast
are never drawn as coast: the map's own edges (a pole row, a cropped
tape's border) and the date line between two land cells.

Everything here is numpy and the standard library.  The shapefiles are
written by a small writer of the published ESRI layout (``.shp`` and
``.shx``); the renderer reads only those two.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "COAST_MANIFEST_NAME",
    "EMPTY_LINEWORK_SENTINEL",
    "MAP_SUBFOLDERS",
    "CoastOutline",
    "MapFolderError",
    "check_map_folder",
    "outline_land_mask",
    "read_tape_land_mask",
    "read_shapefile",
    "write_map_folder",
    "write_shapefile",
]

#: The three sub-folders the renderer resolves, each on its own (rule 1).
MAP_SUBFOLDERS = ("natural_earth_10m", "natural_earth_110m", "us_counties_5m")

#: The manifest written beside the shapefiles: where the mask came from,
#: what was drawn, and the hash of every file.
COAST_MANIFEST_NAME = "world-coast.json"

#: The line a folder carries in its lakes layer when its world has no
#: coast at all: two equal points, which keep the renderer's layer list
#: non-empty (so it never falls back to the cartopy cache's Earth coast at
#: regional zoom) and draw nothing.
EMPTY_LINEWORK_SENTINEL = ((0.0, 0.0), (0.0, 0.0))

#: Every file the renderer opens, per resolution tag, and its shape type.
#: Polygon files are 5, line files 3.  Files the world has nothing for are
#: written empty rather than left out: a file that does not open is read
#: as empty by the renderer too, but an empty file says it was meant.
_POLYLINE = 3
_POLYGON = 5

_SHP_HEADER = 100
_SHP_FILE_CODE = 9994
_SHP_VERSION = 1000


class MapFolderError(ValueError):
    """A map folder the renderer would draw Earth's geography from."""


# ---------------------------------------------------------------------------
# The outline.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoastOutline:
    """A land mask's outline in degrees.

    ``coast``: open or closed lines of (lon, lat) points, drawn as the
    coastline.  ``rings``: closed rings of (lon, lat) points, first point
    repeated last, whose even-odd fill is exactly the land cells.  Longitudes
    run -180..180.  ``land_cells`` and ``cells`` count the mask.
    """

    coast: tuple[tuple[tuple[float, float], ...], ...]
    rings: tuple[tuple[tuple[float, float], ...], ...]
    land_cells: int
    cells: int
    periodic: bool

    @property
    def land_fraction_cells(self) -> float:
        return self.land_cells / self.cells if self.cells else 0.0


def _cell_edges_lat(lat: np.ndarray) -> np.ndarray:
    mid = 0.5 * (lat[1:] + lat[:-1])
    first = lat[0] - (mid[0] - lat[0])
    last = lat[-1] + (lat[-1] - mid[-1])
    return np.clip(np.concatenate(([first], mid, [last])), -90.0, 90.0)


def _lon_pieces(lon: np.ndarray, periodic: bool):
    """Cell pieces along longitude: (edges, column of each piece).

    ``lon`` is sorted and inside [-180, 180).  On a periodic ring the cell
    that straddles the date line is cut in two at +-180, so every piece
    lies inside [-180, 180] and no outline ever crosses the line.
    """

    nx = lon.size
    if not periodic:
        mid = 0.5 * (lon[1:] + lon[:-1])
        edges = np.concatenate(([lon[0] - (mid[0] - lon[0])], mid,
                                [lon[-1] + (lon[-1] - mid[-1])]))
        return edges, np.arange(nx)
    wrapped = np.concatenate(([lon[-1] - 360.0], lon, [lon[0] + 360.0]))
    west = 0.5 * (wrapped[:-2] + wrapped[1:-1])      # west edge of each column
    east = 0.5 * (wrapped[1:-1] + wrapped[2:])       # east edge of each column
    pieces_w: list[float] = []
    pieces_e: list[float] = []
    columns: list[int] = []
    tail: list[tuple[float, float, int]] = []
    head: list[tuple[float, float, int]] = []
    for j in range(nx):
        w, e = float(west[j]), float(east[j])
        if w < -180.0:
            head.append((-180.0, e, j))
            tail.append((w + 360.0, 180.0, j))
        elif e > 180.0:
            head.append((-180.0, e - 360.0, j))
            tail.append((w, 180.0, j))
        else:
            pieces_w.append(w)
            pieces_e.append(e)
            columns.append(j)
    ordered = ([(w, e, j) for w, e, j in head]
               + list(zip(pieces_w, pieces_e, columns))
               + [(w, e, j) for w, e, j in tail])
    ordered.sort(key=lambda item: item[0])
    edges = np.array([ordered[0][0]] + [item[1] for item in ordered])
    return edges, np.array([item[2] for item in ordered], dtype=np.int64)


def _normalise(lat, lon, mask):
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    land = np.asarray(mask)
    if land.ndim != 2 or land.shape != (lat.size, lon.size):
        raise ValueError(
            f"a land mask of shape {land.shape} does not match {lat.size} "
            f"latitudes by {lon.size} longitudes")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("a land mask needs at least 2 x 2 points")
    if not np.all(np.isfinite(land)):
        raise ValueError("the land mask holds a value that is not finite")
    land = land >= 0.5
    order = np.argsort(lat)
    lat, land = lat[order], land[order]
    signed = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(signed, kind="stable")
    lon, land = signed[order], land[:, order]
    if np.any(np.diff(lat) <= 0.0) or np.any(np.diff(lon) <= 0.0):
        raise ValueError("the mask's latitudes or longitudes repeat")
    spacing = float(np.median(np.diff(lon)))
    periodic = abs((lon[-1] - lon[0]) + spacing - 360.0) <= 0.01 * spacing
    return lat, lon, land, periodic


def outline_land_mask(lat, lon, mask) -> CoastOutline:
    """The outline of ``mask`` (rows ``lat``, columns ``lon``, land >= 0.5).

    Latitudes in any order; longitudes in any convention (0..360 or
    -180..180).  The longitude axis is treated as a ring when it covers 360
    degrees at its own spacing, and as a cropped window otherwise.
    """

    lat, lon, land, periodic = _normalise(lat, lon, mask)
    y_edges = _cell_edges_lat(lat)
    x_edges, columns = _lon_pieces(lon, periodic)
    piece = land[:, columns]                      # (ny, m) cell pieces
    ny, m = piece.shape

    padded = np.zeros((ny + 2, m + 2), dtype=bool)
    padded[1:-1, 1:-1] = piece
    here = padded[1:-1, 1:-1]
    south = padded[:-2, 1:-1]
    north = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    # Wrap neighbours, for telling a date-line edge that is coast from one
    # that only cuts a continent in two.
    if periodic:
        wrap_west = piece[:, -1]
        wrap_east = piece[:, 0]
    else:
        wrap_west = wrap_east = None

    starts: list[tuple[int, int]] = []
    ends: list[tuple[int, int]] = []
    coastal: list[bool] = []

    def add(rows, cols, d_start, d_end, coast_flags):
        for (i, j), is_coast in zip(zip(rows.tolist(), cols.tolist()),
                                    coast_flags):
            starts.append((j + d_start[0], i + d_start[1]))
            ends.append((j + d_end[0], i + d_end[1]))
            coastal.append(bool(is_coast))

    # Vertex (x index, y index); land on the left of every directed edge.
    rows, cols = np.nonzero(here & ~south)          # bottom edge, eastward
    add(rows, cols, (0, 0), (1, 0), rows > 0)
    rows, cols = np.nonzero(here & ~north)          # top edge, westward
    add(rows, cols, (1, 1), (0, 1), rows < ny - 1)
    rows, cols = np.nonzero(here & ~east)           # east edge, northward
    flags = cols < m - 1
    if wrap_east is not None:
        flags = flags | ((cols == m - 1) & ~wrap_east[rows])
    add(rows, cols, (1, 0), (1, 1), flags)
    rows, cols = np.nonzero(here & ~west)           # west edge, southward
    flags = cols > 0
    if wrap_west is not None:
        flags = flags | ((cols == 0) & ~wrap_west[rows])
    add(rows, cols, (0, 1), (0, 0), flags)

    rings_idx = _link_rings(starts, ends, coastal)

    def points(vertices):
        return _split_long_segments(
            [(float(x_edges[v[0]]), float(y_edges[v[1]])) for v in vertices])

    rings = []
    coast = []
    for ring in rings_idx:
        vertices = [starts[e] for e in ring] + [starts[ring[0]]]
        rings.append(points(_drop_collinear(vertices, closed=True)))
        for run in _coast_runs(ring, coastal):
            line = [starts[e] for e in run] + [ends[run[-1]]]
            closed = line[0] == line[-1]
            coast.append(points(_drop_collinear(line, closed=closed)))
    return CoastOutline(coast=tuple(coast), rings=tuple(rings),
                        land_cells=int(land.sum()), cells=int(land.size),
                        periodic=bool(periodic))


def _link_rings(starts, ends, coastal) -> list[list[int]]:
    """Chain directed edges into closed rings, each edge used once.

    Where two land cells touch only at a corner, two rings meet at one
    vertex; the edge chosen there is the one turning most to the left,
    which keeps the two cells' rings apart.  Any choice gives the same
    even-odd fill; this one also gives coast lines that do not cross.
    """

    outgoing: dict[tuple[int, int], list[int]] = {}
    for index, vertex in enumerate(starts):
        outgoing.setdefault(vertex, []).append(index)
    used = [False] * len(starts)
    rings: list[list[int]] = []
    # Start each ring on a coast edge where one exists, so a closed coast
    # ring is not split at an arbitrary vertex.
    order = sorted(range(len(starts)), key=lambda e: (not coastal[e], starts[e]))
    for first in order:
        if used[first]:
            continue
        ring = [first]
        used[first] = True
        origin = starts[first]
        current = first
        while ends[current] != origin:
            vertex = ends[current]
            choices = [e for e in outgoing.get(vertex, ()) if not used[e]]
            if not choices:
                raise RuntimeError(
                    f"the outline does not close at vertex {vertex}; the "
                    "mask's edges are inconsistent")
            if len(choices) > 1:
                dx = vertex[0] - starts[current][0]
                dy = vertex[1] - starts[current][1]

                def turn(e, dx=dx, dy=dy, vertex=vertex):
                    ex = ends[e][0] - vertex[0]
                    ey = ends[e][1] - vertex[1]
                    return -(dx * ey - dy * ex)        # left turn sorts first
                choices.sort(key=turn)
            current = choices[0]
            used[current] = True
            ring.append(current)
        rings.append(ring)
    return rings


def _coast_runs(ring: list[int], coastal) -> list[list[int]]:
    flags = [coastal[e] for e in ring]
    if all(flags):
        return [list(ring)]
    if not any(flags):
        return []
    # Rotate so the ring starts just after a non-coast edge.
    cut = next(k for k in range(len(ring)) if not flags[k])
    ring = ring[cut + 1:] + ring[:cut + 1]
    flags = flags[cut + 1:] + flags[:cut + 1]
    runs: list[list[int]] = []
    current: list[int] = []
    for edge, is_coast in zip(ring, flags):
        if is_coast:
            current.append(edge)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


#: The longest stretch of longitude one segment may span.  The renderer
#: joins two points the SHORT way round the globe, so a straight run of
#: coast along a latitude circle wider than 180 degrees (a polar cap's edge,
#: a band continent) would be drawn the wrong way or not at all.  Merged
#: runs are cut back into pieces no wider than this.
MAX_SEGMENT_LON_DEG = 45.0


def _split_long_segments(points, max_dlon: float = MAX_SEGMENT_LON_DEG):
    if len(points) < 2:
        return tuple(points)
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        pieces = int(np.ceil(abs(x1 - x0) / max_dlon)) if x1 != x0 else 1
        for k in range(1, pieces):
            t = k / pieces
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        out.append((x1, y1))
    return tuple(out)


def _drop_collinear(vertices, *, closed: bool):
    """Remove vertices in the middle of a straight run of edges."""

    if len(vertices) <= 2:
        return list(vertices)
    if closed:
        core = list(vertices[:-1])
        keep = []
        n = len(core)
        for k in range(n):
            a, b, c = core[k - 1], core[k], core[(k + 1) % n]
            if (b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0]):
                keep.append(b)
        if not keep:
            keep = core[:1]
        return keep + [keep[0]]
    keep = [vertices[0]]
    for k in range(1, len(vertices) - 1):
        a, b, c = vertices[k - 1], vertices[k], vertices[k + 1]
        if (b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0]):
            keep.append(b)
    keep.append(vertices[-1])
    return keep


# ---------------------------------------------------------------------------
# Shapefiles: the published ESRI layout, .shp and .shx only.
# ---------------------------------------------------------------------------


def _bbox(points: Iterable[tuple[float, float]]):
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _header(code_length_words: int, shape_type: int, box) -> bytes:
    return (struct.pack(">7i", _SHP_FILE_CODE, 0, 0, 0, 0, 0, code_length_words)
            + struct.pack("<2i", _SHP_VERSION, shape_type)
            + struct.pack("<4d", *box)
            + struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))


def write_shapefile(stem: Path, shape_type: int,
                    records: Sequence[Sequence[Sequence[tuple[float, float]]]]
                    ) -> list[Path]:
    """Write ``stem.shp`` and ``stem.shx``.

    ``records``: one entry per shape, each a list of parts (a line's parts,
    or a polygon's rings), each a list of (x, y).  An empty ``records``
    writes a valid file with no shapes.  Returns the two paths.
    """

    if shape_type not in (_POLYLINE, _POLYGON):
        raise ValueError(f"shape type {shape_type} is not a line (3) or a polygon (5)")
    bodies: list[bytes] = []
    every_point: list[tuple[float, float]] = []
    for parts in records:
        parts = [list(part) for part in parts if len(part) >= 2]
        points = [p for part in parts for p in part]
        every_point.extend(points)
        box = _bbox(points)
        offsets, total = [], 0
        for part in parts:
            offsets.append(total)
            total += len(part)
        content = (struct.pack("<i", shape_type) + struct.pack("<4d", *box)
                   + struct.pack("<2i", len(parts), total)
                   + struct.pack(f"<{len(offsets)}i", *offsets)
                   + b"".join(struct.pack("<2d", float(x), float(y))
                              for x, y in points))
        bodies.append(content)
    box = _bbox(every_point)
    shp = bytearray()
    shx = bytearray()
    offset_words = _SHP_HEADER // 2
    for number, content in enumerate(bodies, start=1):
        words = len(content) // 2
        shp += struct.pack(">2i", number, words) + content
        shx += struct.pack(">2i", offset_words, words)
        offset_words += 4 + words
    shp_bytes = _header((_SHP_HEADER + len(shp)) // 2, shape_type, box) + bytes(shp)
    shx_bytes = _header((_SHP_HEADER + len(shx)) // 2, shape_type, box) + bytes(shx)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, data in ((".shp", shp_bytes), (".shx", shx_bytes)):
        path = stem.with_name(stem.name + suffix)
        path.write_bytes(data)
        written.append(path)
    return written


def read_shapefile(path: Path) -> tuple[int, list[list[list[tuple[float, float]]]]]:
    """(shape type, records) of a ``.shp`` written in the layout above.

    A reader for checking what this module writes, and for the folder
    check; it reads polylines and polygons and skips any other shape.
    """

    data = Path(path).read_bytes()
    if len(data) < _SHP_HEADER:
        raise MapFolderError(f"{path} is shorter than a shapefile header")
    code = struct.unpack(">i", data[:4])[0]
    if code != _SHP_FILE_CODE:
        raise MapFolderError(f"{path} is not a shapefile (file code {code})")
    length = struct.unpack(">i", data[24:28])[0] * 2
    shape_type = struct.unpack("<i", data[32:36])[0]
    records = []
    at = _SHP_HEADER
    end = min(length, len(data))
    while at + 8 <= end:
        _number, words = struct.unpack(">2i", data[at:at + 8])
        body = data[at + 8:at + 8 + words * 2]
        at += 8 + words * 2
        kind = struct.unpack("<i", body[:4])[0]
        if kind not in (_POLYLINE, _POLYGON):
            continue
        nparts, npoints = struct.unpack("<2i", body[36:44])
        offsets = list(struct.unpack(f"<{nparts}i", body[44:44 + 4 * nparts]))
        base = 44 + 4 * nparts
        flat = struct.unpack(f"<{2 * npoints}d", body[base:base + 16 * npoints])
        points = list(zip(flat[0::2], flat[1::2]))
        bounds = offsets + [npoints]
        records.append([points[bounds[k]:bounds[k + 1]] for k in range(nparts)])
    return shape_type, records


# ---------------------------------------------------------------------------
# The map folder.
# ---------------------------------------------------------------------------

#: The whole globe as one ring: the ocean the land is filled over (rule 2),
#: clockwise as a shapefile outer ring, its long sides cut into pieces the
#: renderer joins the right way round.
_GLOBE_RING = _split_long_segments(((-180.0, -90.0), (-180.0, 90.0), (180.0, 90.0),
                                    (180.0, -90.0), (-180.0, -90.0)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_map_folder(outline: CoastOutline, folder: Path, *,
                     source: dict | None = None) -> Path:
    """Write the map folder the renderer reads for ``outline``.

    The folder gets the three sub-folders of rule 1, every file the
    renderer opens in them, an ocean of the whole globe (rule 2), the
    world's land as one even-odd polygon (also as the 110m country layer
    the world view fills land from), its coast as lines, empty borders,
    states and counties, and :data:`EMPTY_LINEWORK_SENTINEL` when there is
    no coast at all.  ``source`` is recorded in the manifest as given.
    Returns ``folder``.
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Polygon rings: shapefile outer rings run clockwise; the outline's run
    # with land on the left (anticlockwise), so each is reversed.  The
    # renderer fills even-odd and does not care; a reader that does gets
    # outer rings and holes the right way round.
    land = [[list(reversed(ring)) for ring in outline.rings]] if outline.rings else []
    coast = [[list(line)] for line in outline.coast]
    lakes = [] if coast else [[list(EMPTY_LINEWORK_SENTINEL)]]
    files: dict[str, str] = {}
    for tag in ("10m", "110m"):
        sub = folder / f"natural_earth_{tag}"
        plan = [
            (f"ne_{tag}_ocean", _POLYGON, [[list(_GLOBE_RING)]]),
            (f"ne_{tag}_land", _POLYGON, land),
            (f"ne_{tag}_coastline", _POLYLINE, coast),
            (f"ne_{tag}_lakes", _POLYLINE, lakes),
            (f"ne_{tag}_admin_0_boundary_lines_land", _POLYLINE, []),
            (f"ne_{tag}_admin_1_states_provinces_lines", _POLYLINE, []),
        ]
        if tag == "110m":
            plan.append((f"ne_{tag}_admin_0_countries", _POLYGON, land))
        for stem, kind, records in plan:
            for path in write_shapefile(sub / stem, kind, records):
                files[path.relative_to(folder).as_posix()] = _sha256(path)
    for path in write_shapefile(folder / "us_counties_5m" / "cb_2023_us_county_5m",
                                _POLYGON, []):
        files[path.relative_to(folder).as_posix()] = _sha256(path)
    manifest = {
        "what": "a world's own coast for the engine's renderer "
                "(RUSTWX_BASEMAP_DIR); written by arwen_global.world.coast",
        "source": source or {},
        "land_cells": outline.land_cells,
        "cells": outline.cells,
        "periodic": outline.periodic,
        "rings": len(outline.rings),
        "coast_lines": len(outline.coast),
        "coast_points": sum(len(line) for line in outline.coast),
        "empty_linework_sentinel": not coast,
        "files": dict(sorted(files.items())),
    }
    (folder / COAST_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return folder


def check_map_folder(folder: Path) -> Path:
    """Refuse a map folder the renderer would draw Earth's geography from.

    Rule 1: each of :data:`MAP_SUBFOLDERS` is there.  Rule 2: the 10m
    ocean polygon holds at least one ring.  Also: each resolution carries a
    coastline file, so the renderer finds no reason to look elsewhere.
    Returns the folder, resolved.
    """

    folder = Path(folder)
    if not folder.is_dir():
        raise MapFolderError(f"the map folder {folder} does not exist")
    missing = [name for name in MAP_SUBFOLDERS if not (folder / name).is_dir()]
    if missing:
        raise MapFolderError(
            f"the map folder {folder} has no {', '.join(missing)}.  The "
            "renderer looks for each of " + ", ".join(MAP_SUBFOLDERS)
            + " on its own and takes a missing one from Earth's assets, so "
            "Earth's lines would be drawn on this world")
    for tag in ("10m", "110m"):
        coast = folder / f"natural_earth_{tag}" / f"ne_{tag}_coastline.shp"
        if not coast.is_file():
            raise MapFolderError(
                f"the map folder {folder} has no {coast.relative_to(folder).as_posix()}")
    ocean = folder / "natural_earth_10m" / "ne_10m_ocean.shp"
    rings = 0
    if ocean.is_file():
        _kind, records = read_shapefile(ocean)
        rings = sum(len(parts) for parts in records)
    if rings == 0:
        raise MapFolderError(
            f"the map folder {folder} has an empty ocean polygon "
            f"({ocean.relative_to(folder).as_posix()}).  With no ocean and no "
            "land the renderer falls back to a cartopy cache in the home "
            "directory, which is Earth's land")
    return folder.resolve()


# ---------------------------------------------------------------------------
# A tape's own mask.
# ---------------------------------------------------------------------------


def read_tape_land_mask(path: Path):
    """(lat, lon, mask, sha256 of the mask bytes) from a wrfout tape.

    Reads ``LANDMASK``, ``XLAT`` and ``XLONG`` of the first frame.  The
    tapes this package exports are on a regular latitude-longitude grid,
    which is checked: a row of XLAT must be constant and a column of XLONG
    must be constant.
    """

    from netCDF4 import Dataset

    with Dataset(os.fspath(path)) as tape:
        for name in ("LANDMASK", "XLAT", "XLONG"):
            if name not in tape.variables:
                raise ValueError(f"{path} carries no {name}; it is not a tape "
                                 "the coast can be read from")
        mask = np.asarray(tape.variables["LANDMASK"][...], dtype=np.float64)
        xlat = np.asarray(tape.variables["XLAT"][...], dtype=np.float64)
        xlong = np.asarray(tape.variables["XLONG"][...], dtype=np.float64)
    while mask.ndim > 2:
        mask, xlat, xlong = mask[0], xlat[0], xlong[0]
    lat = xlat[:, 0]
    lon = xlong[0, :]
    if (np.max(np.abs(xlat - lat[:, None])) > 1e-4
            or np.max(np.abs(xlong - lon[None, :])) > 1e-4):
        raise ValueError(f"{path} is not on a regular latitude-longitude grid")
    land = (mask >= 0.5)
    digest = hashlib.sha256(np.ascontiguousarray(land, dtype=np.uint8).tobytes()
                            + struct.pack("<2i", *land.shape)).hexdigest()
    return lat, lon, land, digest
