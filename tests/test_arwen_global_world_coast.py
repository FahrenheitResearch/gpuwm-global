"""A world's own coast for the engine's renderer (worlds plan, package 2.7).

Known answers on small masks, CPU only: the land fill of the outline is
exactly the land cells, the coast is exactly the land-water cell edges and
never a map edge or the date line across land, the map folder keeps the
three rules of the plan's section 4.10, and the render door changes nothing
on Earth.  The picture checks on a real tape and the agreement of Earth's
own mask with the reference coast are measured on a host with the renderer
staged; they need the engine's map assets and are not run here.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from arwen_global.world.coast import (
    COAST_MANIFEST_NAME,
    EMPTY_LINEWORK_SENTINEL,
    MAP_SUBFOLDERS,
    MapFolderError,
    check_map_folder,
    outline_land_mask,
    read_shapefile,
    write_map_folder,
    write_shapefile,
)

SMOKE = Path(__file__).resolve().parents[1] / "src" / "arwen_global" / "configs" / "arwen_global_moist_smoke.toml"


def _global_grid(ny: int, nx: int, *, offset: bool):
    """Cell centres of a global grid with no pole points.  ``offset``
    puts the columns half a cell off the date line (no cell straddles it);
    without it column 0 is centred on -180 and is cut in two there."""

    dlat = 180.0 / ny
    lat = -90.0 + dlat * (np.arange(ny) + 0.5)
    dlon = 360.0 / nx
    lon = dlon * (np.arange(nx) + (0.5 if offset else 0.0))     # 0..360
    return lat, lon


def _even_odd(rings, lon_pts, lat_pts):
    """Even-odd inside test of every (lon, lat) against all rings."""

    inside = np.zeros(lon_pts.shape, dtype=bool)
    for ring in rings:
        ring = np.asarray(ring, dtype=np.float64)
        x0, y0 = ring[:-1, 0], ring[:-1, 1]
        x1, y1 = ring[1:, 0], ring[1:, 1]
        for a, b, c, d in zip(x0, y0, x1, y1):
            if b == d:
                continue
            crosses = (b > lat_pts) != (d > lat_pts)
            x_at = a + (lat_pts - b) * (c - a) / (d - b)
            inside ^= crosses & (lon_pts < x_at)
    return inside


def _signed(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def _unit_edges(outline_lines, lat_edges, lon_edges):
    """Every cell edge a set of axis-aligned lines covers, as rounded
    midpoints in degrees."""

    found = set()
    for line in outline_lines:
        # Rejoin the pieces a long straight run was cut into.
        merged = [line[0]]
        for k in range(1, len(line) - 1):
            a, b, c = merged[-1], line[k], line[k + 1]
            if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
                merged.append(b)
        merged.append(line[-1])
        for (x0, y0), (x1, y1) in zip(merged[:-1], merged[1:]):
            if y0 == y1:
                lo, hi = sorted((x0, x1))
                cuts = [x for x in lon_edges if lo - 1e-9 <= x <= hi + 1e-9]
                for a, b in zip(cuts[:-1], cuts[1:]):
                    found.add((round(0.5 * (a + b), 6), round(y0, 6)))
            else:
                assert x0 == x1, "an outline segment that is not on a grid line"
                lo, hi = sorted((y0, y1))
                cuts = [y for y in lat_edges if lo - 1e-9 <= y <= hi + 1e-9]
                for a, b in zip(cuts[:-1], cuts[1:]):
                    found.add((round(x0, 6), round(0.5 * (a + b), 6)))
    return found


# -- the outline ------------------------------------------------------------


def test_one_land_cell_is_one_square_ring_and_one_closed_coast():
    lat, lon = _global_grid(6, 12, offset=True)
    mask = np.zeros((6, 12))
    mask[3, 4] = 1.0
    outline = outline_land_mask(lat, lon, mask)
    assert outline.land_cells == 1 and outline.cells == 72
    assert len(outline.rings) == 1 and len(outline.coast) == 1
    ring = outline.rings[0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    x0, x1 = _signed(lon[4]) - 15.0, _signed(lon[4]) + 15.0
    assert sorted({p for p in ring}) == sorted(
        {(x0, 0.0), (x1, 0.0), (x1, 30.0), (x0, 30.0)})
    assert outline.coast[0][0] == outline.coast[0][-1]
    # Land on the left: the ring runs anticlockwise (positive area).
    pts = np.asarray(ring)
    area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
    assert area == pytest.approx(30.0 * 30.0)


@pytest.mark.parametrize("offset", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fill_of_the_outline_is_exactly_the_land_cells(seed, offset):
    rng = np.random.default_rng(seed)
    lat, lon = _global_grid(10, 20, offset=offset)
    mask = (rng.random((10, 20)) < 0.45).astype(np.float32)
    mask[0, :] = 1.0                 # an Antarctica touching the pole row
    mask[:, 0] = 1.0                 # land on the prime meridian and
    mask[:, 9:11] = 1.0              # across the date line
    outline = outline_land_mask(lat, lon, mask)
    lon2, lat2 = np.meshgrid(_signed(lon), lat)
    # A centre ON the date line (offset False, column 0) is tested a hair
    # inside the map on both sides.
    for nudge in (1e-6, -1e-6):
        x = np.where(lon2 == -180.0, -180.0 + 1e-6, lon2) if nudge > 0 \
            else np.where(lon2 == -180.0, 180.0 - 1e-6, lon2)
        assert np.array_equal(_even_odd(outline.rings, x, lat2), mask >= 0.5)


@pytest.mark.parametrize("seed", [0, 5, 9])
def test_coast_is_exactly_the_land_water_edges_and_nothing_else(seed):
    rng = np.random.default_rng(seed)
    ny, nx = 9, 18
    lat, lon = _global_grid(ny, nx, offset=True)
    mask = rng.random((ny, nx)) < 0.5
    mask[0, :] = True
    mask[:3, 8] = mask[:3, 9] = True    # land on both sides of the date line
    mask[3:6, 8] = True                 # land on its west side only (170 E)
    mask[3:6, 9] = False
    outline = outline_land_mask(lat, lon, mask.astype(float))
    lon_s = _signed(lon)
    order = np.argsort(lon_s)
    lon_s, m = lon_s[order], mask[:, order]
    lat_edges = np.linspace(-90.0, 90.0, ny + 1)
    lon_edges = np.linspace(-180.0, 180.0, nx + 1)
    expected = set()
    for i in range(ny):
        for j in range(nx):
            k = (j + 1) % nx
            if m[i, j] != m[i, k]:
                x = lon_edges[j + 1]
                if x == 180.0:
                    # The date-line edge is drawn on the land side's own
                    # map edge: +180 for land west of it, -180 for land
                    # east of it.
                    x = 180.0 if m[i, j] else -180.0
                    expected.add((round(x, 6), round(lat[i], 6)))
                else:
                    expected.add((round(x, 6), round(lat[i], 6)))
            if i + 1 < ny and m[i, j] != m[i + 1, j]:
                expected.add((round(lon_s[j], 6), round(lat_edges[i + 1], 6)))
    got = _unit_edges(outline.coast, lat_edges, lon_edges)
    assert got == expected
    # Never along a pole row.
    assert not any(abs(y) == 90.0 for _x, y in got)


def test_a_continent_across_the_date_line_draws_no_line_on_it():
    lat, lon = _global_grid(12, 24, offset=False)   # column 0 centred on -180
    mask = np.zeros((12, 24))
    mask[4:8, 0] = 1.0
    mask[4:8, 1] = 1.0
    mask[4:8, 23] = 1.0
    outline = outline_land_mask(lat, lon, mask)
    for line in outline.coast:
        for (x0, _y0), (x1, _y1) in zip(line[:-1], line[1:]):
            assert not (abs(x0) == 180.0 and x0 == x1), \
                "a coast segment drawn along the date line through land"
    assert all(-180.0 <= x <= 180.0 for ring in outline.rings for x, _y in ring)


def test_a_cropped_tape_draws_no_coast_along_its_border():
    lat = np.linspace(20.0, 40.0, 9)
    lon = np.linspace(-100.0, -80.0, 9)
    mask = np.zeros((9, 9))
    mask[:, :4] = 1.0                       # land on the western border
    outline = outline_land_mask(lat, lon, mask)
    assert not outline.periodic
    xs = {round(x, 6) for line in outline.coast for x, _y in line}
    assert xs == {round(0.5 * (lon[3] + lon[4]), 6)}


def test_waterworld_and_all_land():
    lat, lon = _global_grid(6, 12, offset=True)
    water = outline_land_mask(lat, lon, np.zeros((6, 12)))
    assert water.rings == () and water.coast == () and water.land_cells == 0
    land = outline_land_mask(lat, lon, np.ones((6, 12)))
    assert land.coast == ()
    assert land.land_cells == 72 and len(land.rings) == 1


def test_no_segment_spans_more_than_45_degrees_of_longitude():
    """The renderer joins two points the short way round the globe, so a
    polar cap's edge drawn as one segment from -180 to 180 would vanish."""

    lat, lon = _global_grid(18, 36, offset=False)
    mask = np.zeros((18, 36))
    mask[16:, :] = 1.0                       # a polar cap
    mask[8, :] = 1.0                         # a band continent
    outline = outline_land_mask(lat, lon, mask)
    assert len(outline.coast) == 3           # the cap's edge, the band's two
    for line in (*outline.coast, *outline.rings):
        steps = np.abs(np.diff(np.asarray(line)[:, 0]))
        assert steps.max() <= 45.0 + 1e-9
    for line in outline.coast:
        xs = np.asarray(line)[:, 0]
        assert xs.min() == -180.0 and xs.max() == 180.0


def test_mask_shape_and_values_are_checked():
    lat, lon = _global_grid(6, 12, offset=True)
    with pytest.raises(ValueError, match="does not match"):
        outline_land_mask(lat, lon, np.zeros((12, 6)))
    bad = np.zeros((6, 12))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        outline_land_mask(lat, lon, bad)


# -- shapefiles and the map folder -----------------------------------------


def test_shapefile_round_trip(tmp_path):
    records = [[[(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]],
               [[(-10.0, 5.0), (10.0, 5.0)], [(3.0, 3.0), (4.0, 4.0)]]]
    shp, shx = write_shapefile(tmp_path / "lines", 3, records)
    assert shp.suffix == ".shp" and shx.suffix == ".shx"
    kind, back = read_shapefile(shp)
    assert kind == 3 and back == records
    # The .shx index: one 8-byte entry per record after the header.
    assert shx.stat().st_size == 100 + 8 * len(records)
    kind, empty = read_shapefile(write_shapefile(tmp_path / "none", 5, [])[0])
    assert kind == 5 and empty == []
    shapefile = pytest.importorskip("shapefile")
    with shapefile.Reader(shp=open(shp, "rb"), shx=open(shx, "rb")) as reader:
        shapes = reader.shapes()
        assert len(shapes) == 2
        assert [tuple(p) for p in shapes[1].points] == [
            (-10.0, 5.0), (10.0, 5.0), (3.0, 3.0), (4.0, 4.0)]
        assert list(shapes[1].parts) == [0, 2]


def test_map_folder_keeps_the_three_rules(tmp_path):
    lat, lon = _global_grid(12, 24, offset=True)
    mask = np.zeros((12, 24))
    mask[3:7, 5:11] = 1.0
    outline = outline_land_mask(lat, lon, mask)
    folder = write_map_folder(outline, tmp_path / "world-map",
                              source={"tape": "x"})
    assert check_map_folder(folder) == folder.resolve()
    for name in MAP_SUBFOLDERS:                                    # rule 1
        assert (folder / name).is_dir()
    for tag in ("10m", "110m"):
        _kind, ocean = read_shapefile(folder / f"natural_earth_{tag}" / f"ne_{tag}_ocean.shp")
        assert len(ocean) == 1 and len(ocean[0]) == 1               # rule 2
        _kind, coast = read_shapefile(folder / f"natural_earth_{tag}" / f"ne_{tag}_coastline.shp")
        assert len(coast) == len(outline.coast) >= 1
        for empty in ("admin_0_boundary_lines_land", "admin_1_states_provinces_lines", "lakes"):
            assert read_shapefile(folder / f"natural_earth_{tag}" / f"ne_{tag}_{empty}.shp")[1] == []
    _kind, countries = read_shapefile(folder / "natural_earth_110m" / "ne_110m_admin_0_countries.shp")
    _kind, land = read_shapefile(folder / "natural_earth_10m" / "ne_10m_land.shp")
    assert countries == land and len(land) == 1
    # Shapefile order: outer rings clockwise (negative shoelace area).
    pts = np.asarray(land[0][0])
    assert np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]) < 0
    assert read_shapefile(folder / "us_counties_5m" / "cb_2023_us_county_5m.shp")[1] == []
    manifest = json.loads((folder / COAST_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["land_cells"] == 24 and manifest["source"] == {"tape": "x"}
    assert manifest["empty_linework_sentinel"] is False
    assert len(manifest["files"]) == 2 * (6 + 7 + 1)


def test_a_world_with_no_coast_carries_the_sentinel_line(tmp_path):
    lat, lon = _global_grid(6, 12, offset=True)
    folder = write_map_folder(outline_land_mask(lat, lon, np.zeros((6, 12))),
                              tmp_path / "water")
    check_map_folder(folder)
    for tag in ("10m", "110m"):
        _kind, lakes = read_shapefile(folder / f"natural_earth_{tag}" / f"ne_{tag}_lakes.shp")
        assert lakes == [[list(EMPTY_LINEWORK_SENTINEL)]]
        assert read_shapefile(folder / f"natural_earth_{tag}" / f"ne_{tag}_land.shp")[1] == []


def test_check_map_folder_refuses_by_name(tmp_path):
    lat, lon = _global_grid(6, 12, offset=True)
    outline = outline_land_mask(lat, lon, np.eye(6, 12))
    folder = write_map_folder(outline, tmp_path / "m")
    (folder / "us_counties_5m" / "cb_2023_us_county_5m.shp").unlink()
    (folder / "us_counties_5m" / "cb_2023_us_county_5m.shx").unlink()
    (folder / "us_counties_5m").rmdir()
    with pytest.raises(MapFolderError, match="has no us_counties_5m"):
        check_map_folder(folder)
    folder = write_map_folder(outline, tmp_path / "n")
    write_shapefile(folder / "natural_earth_10m" / "ne_10m_ocean", 5, [])
    with pytest.raises(MapFolderError, match="empty ocean polygon"):
        check_map_folder(folder)
    with pytest.raises(MapFolderError, match="does not exist"):
        check_map_folder(tmp_path / "absent")


# -- a tape's own mask ------------------------------------------------------


def _write_tape(path: Path, lat, lon, mask, attrs=None):
    netCDF4 = pytest.importorskip("netCDF4")
    xlong, xlat = np.meshgrid(lon, lat)
    with netCDF4.Dataset(path, "w") as tape:
        if attrs:
            tape.setncatts(attrs)
        tape.createDimension("Time", 1)
        tape.createDimension("south_north", lat.size)
        tape.createDimension("west_east", lon.size)
        for name, value in (("XLAT", xlat), ("XLONG", xlong), ("LANDMASK", mask)):
            var = tape.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            var[0] = value
    return path


def test_read_tape_land_mask(tmp_path):
    from arwen_global.world.coast import read_tape_land_mask

    lat, lon = _global_grid(6, 12, offset=True)
    lon = _signed(lon)
    mask = np.zeros((6, 12), dtype=np.float32)
    mask[2, 3] = 1.0
    tape = _write_tape(tmp_path / "wrfout_d01_2019-01-15_00_00_00", lat, lon, mask)
    got_lat, got_lon, land, digest = read_tape_land_mask(tape)
    assert np.allclose(got_lat, lat) and np.allclose(got_lon, lon)
    assert np.array_equal(land, mask >= 0.5) and len(digest) == 64


# -- the render door --------------------------------------------------------


class _Engine:
    """Stands in for ``gpuwm.render``: records each call and the map
    folder variable the renderer would have been handed."""

    def __init__(self):
        self.calls = []

    def render_wrfouts_rust(self, tapes, **kwargs):
        self.calls.append((list(tapes), dict(kwargs),
                           os.environ.get("RUSTWX_BASEMAP_DIR")))
        return [], [], []


@pytest.fixture()
def engine(monkeypatch):
    import arwen_global.render_door as render_door

    fake = _Engine()
    module = types.ModuleType("gpuwm.render")
    module.render_wrfouts_rust = fake.render_wrfouts_rust
    monkeypatch.setitem(sys.modules, "gpuwm.render", module)
    monkeypatch.setattr(render_door, "_require_renderer", lambda: Path("rw_wrfbatch"))
    return fake


def _args(*argv):
    import argparse

    from arwen_global.render_door import add_render_arguments

    parser = argparse.ArgumentParser()
    add_render_arguments(parser)
    return parser.parse_args(list(argv))


def _tape(tmp_path, name="wrfout_d01_2019-01-15_00_00_00", attrs=None):
    lat, lon = _global_grid(6, 12, offset=True)
    mask = np.zeros((6, 12), dtype=np.float32)
    mask[2:4, 3:6] = 1.0
    return _write_tape(tmp_path / name, lat, _signed(lon), mask, attrs)


def test_render_door_reads_the_world_from_the_tapes_themselves(tmp_path, engine, monkeypatch):
    """Tapes rendered with no config: the coast and the title come from the
    attributes a world's tape carries (packages 1.4 and 2.8), and tapes of
    different classes are titled each with their own."""

    from arwen_global.render_door import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    world = {"ARWEN_WORLD": "test-planet", "ARWEN_WORLD_COAST": "own"}
    tapes = [_tape(tmp_path, f"wrfout_d01_2019-01-1{day}_00_00_00",
                   dict(world, ARWEN_WORLD_CLASS=klass))
             for day, klass in ((5, "illustration"), (6, "science"), (7, "science"))]
    out = tmp_path / "out"
    assert render(_args(*map(str, tapes), "--outdir", str(out))) == 0
    assert [call[0] for call in engine.calls] == [tapes[:1], tapes[1:]]
    titles = [json.loads(Path(call[1]["annotate"]).read_text(encoding="utf-8"))
              for call in engine.calls]
    assert titles == [{"title_suffix": "test-planet (illustration)"},
                      {"title_suffix": "test-planet (science near the lowered ranges only)"}]
    assert all(Path(call[2]) == (out / "world-map").resolve() for call in engine.calls)
    # A world that keeps Earth's coast says "earth": titled, Earth's map.
    engine.calls.clear()
    (tmp_path / "k").mkdir()
    keep = _tape(tmp_path / "k", attrs={"ARWEN_WORLD": "on-its-side",
                                        "ARWEN_WORLD_COAST": "earth"})
    assert render(_args(str(keep), "--outdir", str(tmp_path / "o2"))) == 0
    (_tapes, kwargs, basemap), = engine.calls
    assert basemap is None and json.loads(Path(kwargs["annotate"]).read_text(
        encoding="utf-8")) == {"title_suffix": "on-its-side"}


def test_render_door_on_earth_changes_nothing(tmp_path, engine, monkeypatch):
    from arwen_global.render_door import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    tape = _tape(tmp_path)
    out = tmp_path / "out"
    assert render(_args(str(tape), "--outdir", str(out))) == 0
    (tapes, kwargs, basemap), = engine.calls
    assert tapes == [tape]
    assert kwargs["annotate"] is None and basemap is None
    assert set(kwargs) == {"products", "timeidx", "outdir", "size", "annotate"}
    assert not (out / "world-map").exists() and not (out / "world-title.json").exists()
    # An all-Earth [world] table given as the config: still nothing.
    cfg = tmp_path / "earth.toml"
    cfg.write_text(SMOKE.read_text(encoding="utf-8")
                   + '\n[world]\nlabel = "earth"\n[world.sea]\nlevel_m = 0.0\n',
                   encoding="utf-8")
    assert render(_args(str(tape), "--outdir", str(out), "--config", str(cfg))) == 0
    _tapes, kwargs, basemap = engine.calls[-1]
    assert kwargs["annotate"] is None and basemap is None
    assert "RUSTWX_BASEMAP_DIR" not in os.environ


def test_render_door_draws_the_tapes_own_coast_and_restores_the_environment(
        tmp_path, engine, monkeypatch):
    from arwen_global.render_door import render

    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", "/earth/assets/basemap")
    tape = _tape(tmp_path)
    out = tmp_path / "out"
    assert render(_args(str(tape), "--outdir", str(out), "--own-coast",
                        "--title-suffix", "Ring Planet")) == 0
    (_tapes, kwargs, basemap), = engine.calls
    assert Path(basemap) == (out / "world-map").resolve()
    assert os.environ["RUSTWX_BASEMAP_DIR"] == "/earth/assets/basemap"
    check_map_folder(Path(basemap))
    manifest = json.loads((out / "world-map" / COAST_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["land_cells"] == 6 and manifest["source"]["tape"] == tape.name
    assert json.loads(Path(kwargs["annotate"]).read_text(encoding="utf-8")) == {
        "title_suffix": "Ring Planet"}


def test_render_door_takes_coast_and_title_from_a_world_config(tmp_path, engine, monkeypatch):
    from arwen_global.render_door import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    tape = _tape(tmp_path)
    cfg = tmp_path / "flood.toml"
    cfg.write_text(SMOKE.read_text(encoding="utf-8")
                   + '\n[world]\nlabel = "sea-plus-70"\n[world.sea]\nlevel_m = 70.0\n',
                   encoding="utf-8")
    out = tmp_path / "out"
    assert render(_args(str(tape), "--outdir", str(out), "--config", str(cfg))) == 0
    (_tapes, kwargs, basemap), = engine.calls
    assert basemap is not None and Path(basemap).name == "world-map"
    assert json.loads(Path(kwargs["annotate"]).read_text(encoding="utf-8")) == {
        "title_suffix": "sea-plus-70"}
    assert "RUSTWX_BASEMAP_DIR" not in os.environ


def test_render_door_takes_the_namespaces_go_and_run_plan_build(tmp_path, engine, monkeypatch):
    """`go` and `run-plan` call the door with namespaces of their own that
    carry none of the three world options."""

    from arwen_global.render_door import render

    monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
    tape = _tape(tmp_path)
    leg = types.SimpleNamespace(
        config=None, inputs=[tape], outdir=tmp_path / "pictures",
        start_date=None, products="2m_temperature", size="1600x1000",
        nlat=360, nlon=720, bbox=None, tapes_dir=None, keep_tapes=False,
        overwrite=False)
    assert render(leg) == 0
    (_tapes, kwargs, basemap), = engine.calls
    assert kwargs["annotate"] is None and basemap is None


def test_render_door_refuses_a_bad_map_folder_before_drawing(tmp_path, engine):
    from arwen_global.render_door import render

    tape = _tape(tmp_path)
    (tmp_path / "half" / "natural_earth_10m").mkdir(parents=True)
    with pytest.raises(MapFolderError, match="natural_earth_110m"):
        render(_args(str(tape), "--outdir", str(tmp_path / "o"),
                     "--map-folder", str(tmp_path / "half")))
    assert engine.calls == []
