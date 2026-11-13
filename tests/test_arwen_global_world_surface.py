"""The invented-world surface (plan package 2.4): the world folder, its lock
file, the eleven cached arrays written on the model grid, the table rules
and flooding from the largest ocean.

Every test here is CPU-only and runs on the tiny grids: a T21 Gaussian
grid for the arrays and the shipped T3 moist smoke config for the config
path.  The world's map is a one-degree rectangle continent drawn here.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from arwen_global import statics
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.world import parse_world
from arwen_global.world import folder as world_folder
from arwen_global.world import surface as world_surface
from arwen_global.world.folder import WorldFolderError

VALID_TIME = datetime(2026, 11, 12, 0)
ISWATER = 17

# The rectangle continent: 20S-40N, 30W-60E, 500 m high, with a basin
# 100 m below sea level in its middle (10N-20N, 10E-20E) that must stay
# land, and a small sea 200 m deep cut off from the ocean by land
# (30N-35N, 40E-50E) that must become land too.
SOUTH, NORTH, WEST, EAST = -20.0, 40.0, -30.0, 60.0


def _map():
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    continent = (la > SOUTH) & (la < NORTH) & (lo > WEST) & (lo < EAST)
    elevation = np.where(continent, 500.0, -4000.0)
    basin = (la > 10.0) & (la < 20.0) & (lo > 10.0) & (lo < 20.0)
    inland_sea = (la > 30.0) & (la < 35.0) & (lo > 40.0) & (lo < 50.0)
    elevation = np.where(basin, -100.0, elevation)
    elevation = np.where(inland_sea, -200.0, elevation)
    return elevation, lat, lon


def _world(folder: str, land_cover: str = "open-shrubland"):
    return parse_world({
        "label": "rectangle",
        "sea": {"temperature": "zonal-mean", "ice": "none"},
        "surface": {"folder": folder, "land_cover": land_cover, "soil_wetness": 0.3},
    })


def _deep(grid) -> np.ndarray:
    return 300.0 - 40.0 * np.sin(np.deg2rad(grid.latitude_deg)) ** 2


@pytest.fixture(scope="module")
def t21_grid() -> GaussianGrid:
    return GaussianGrid.create(21, dealias_factor=1.5)


@pytest.fixture()
def planet(tmp_path):
    folder = tmp_path / "rectangle.world"
    world_folder.write_map(folder, *_map(), source={"name": "rectangle test map"})
    return folder


def _column(grid, lat_deg, lon_deg):
    j = int(np.argmin(np.abs(grid.latitude_deg - lat_deg)))
    i = int(np.argmin(np.abs(((grid.longitude_deg - lon_deg + 180.0) % 360.0) - 180.0)))
    return j, i


# -- connected bodies --------------------------------------------------------


def test_bodies_join_across_the_date_line_and_the_pole():
    cand = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
    ], dtype=bool)
    plain = world_surface.connected_labels(cand, periodic=False)
    assert plain[2, 0] == plain[2, 1] and plain[2, 5] != plain[2, 0]
    wrapped = world_surface.connected_labels(cand, periodic=True)
    assert wrapped[2, 5] == wrapped[2, 0] == 12
    assert wrapped[0, 0] == 0 and wrapped[3, 3] == 21
    assert np.all(wrapped[~cand] == -1)
    # Row 3 touches a pole: column 3 meets column 0 half-way round.
    cand[3, 0] = True
    polar = world_surface.connected_labels(cand, periodic=True, polar_rows=(3,))
    assert polar[3, 0] == polar[3, 3]


def test_the_largest_body_is_chosen_by_area_not_by_cells():
    lat = np.array([80.0, 0.0])
    cand = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 0]], dtype=bool)
    kept, report = world_surface.largest_body(cand, lat, periodic=False)
    # three cells at 80 degrees weigh 0.52, two at the equator weigh 2
    assert kept[1, 5] and kept[1, 6] and not kept[0].any()
    assert report["bodies"] == 2 and report["dropped_cells"] == 3


def test_flooding_keeps_the_ocean_and_dries_every_cut_off_basin():
    elevation, lat, lon = _map()
    ocean, report = world_surface.flood_from_largest_ocean(elevation, lat, lon)
    basin = (np.abs(lat - 15.0) < 4.0)[:, None] & (np.abs(lon - 15.0) < 4.0)[None, :]
    inland = (np.abs(lat - 32.5) < 2.0)[:, None] & (np.abs(lon - 45.0) < 4.0)[None, :]
    assert not ocean[basin].any() and not ocean[inland].any()
    assert report["basins_made_land_cells"] == int(
        ((elevation > -4000.0) & (elevation <= 0.0)).sum()) == 10 * 10 + 5 * 10
    assert report["bodies_below_sea_level"] == 3
    assert np.array_equal(ocean, elevation == -4000.0)
    # A planet with no cell at or below sea level has no ocean at all.
    dry, dry_report = world_surface.flood_from_largest_ocean(np.full((4, 8), 10.0),
                                                              lat[::45], lon[::45])
    assert not dry.any() and dry_report["ocean_cells"] == 0


# -- the folder and its lock ---------------------------------------------------


def test_the_map_is_checked_and_pinned_by_the_lock(tmp_path):
    folder = tmp_path / "x.world"
    elevation, lat, lon = _map()
    with pytest.raises(WorldFolderError, match="metres are expected"):
        world_folder.write_map(folder, elevation * 1000.0, lat, lon)
    with pytest.raises(WorldFolderError, match="non-finite"):
        world_folder.write_map(folder, np.where(elevation > 0, np.nan, elevation), lat, lon)
    with pytest.raises(WorldFolderError, match="span less than 360"):
        world_folder.write_map(folder, elevation[:, :-1], lat,
                               np.linspace(0.0, 360.0, lon.size - 1))
    path = world_folder.write_map(folder, elevation, lat, lon, source={"credit": "test"})
    lock = world_folder.read_lock(folder)
    assert lock["map"]["sha256"] == world_folder.file_sha256(path)
    assert lock["map"]["source"] == {"credit": "test"} and lock["surfaces"] == {}
    with pytest.raises(FileExistsError, match="overwrite"):
        world_folder.write_map(folder, elevation, lat, lon)
    read = world_folder.read_map(folder)
    assert np.array_equal(read[0], elevation)
    # A map changed after the folder was written is refused by name.
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["elevation_m"] = arrays["elevation_m"] + 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(WorldFolderError, match="not the map the lock file pins"):
        world_folder.read_map(folder)
    # A hand-edited lock is refused.
    world_folder.write_map(folder, elevation, lat, lon, overwrite=True)
    document = json.loads(world_folder.lock_path(folder).read_text(encoding="utf-8"))
    document["map"]["sha256"] = "0" * 64
    world_folder.lock_path(folder).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(WorldFolderError, match="self-hash"):
        world_folder.read_lock(folder)


def test_a_second_writer_is_refused_while_the_guard_is_held(planet):
    with world_folder.writing(planet):
        with pytest.raises(WorldFolderError, match="being written by another build"):
            world_folder.write_map(planet, *_map(), overwrite=True)
    assert not (planet / world_folder.WRITING_FILE).exists()


def test_relative_folders_are_read_beside_the_config(tmp_path):
    assert world_folder.resolve_world_folder("p.world", tmp_path) == tmp_path / "p.world"
    assert world_folder.resolve_world_folder(str(tmp_path / "q")) == tmp_path / "q"


# -- the rectangle continent ---------------------------------------------------


def test_class_rows_come_from_vegparm():
    shrub = world_surface.class_parameters("open-shrubland")
    assert shrub == {
        "land_cover": "open-shrubland", "landuse_category": 7, "green_fraction": 0.70,
        "leaf_area_index": 2.60, "background_albedo_percent": pytest.approx(22.0),
        "snow_albedo_percent": 65.0,
    }
    barren = world_surface.class_parameters("barren")
    assert barren["landuse_category"] == 16 and barren["green_fraction"] == 0.01
    assert barren["leaf_area_index"] == 0.75
    assert barren["background_albedo_percent"] == pytest.approx(38.0)
    assert barren["snow_albedo_percent"] == 75.0


def test_a_rectangle_continent_loads_through_the_cache_reader_and_the_resolver(planet, t21_grid):
    world = _world(str(planet))
    npz, sidecar, report = world_surface.build_world_surface(
        world, t21_grid, deep_soil_k=_deep(t21_grid))
    # The cache sits in the world's folder under a world name, never an Earth one.
    assert npz.parent == planet and sidecar.parent == planet
    key = world_surface.surface_key(world_folder.read_lock(planet)["map"]["sha256"],
                                    "open-shrubland")
    assert npz.name == f"arwen-global-statics-T21-33x66-world-{key[:12]}.npz"
    assert report["flood"]["bodies_below_sea_level"] == 3
    # The plain cache reader accepts it with every hash checked.
    fields, document = statics.read_cache(npz, t21_grid)
    assert set(fields) == {*statics.CACHED_FIELDS, "latitude_deg", "longitude_deg"}
    assert document["world_surface"]["key"] == key
    assert statics.real_convention(document).landuse_dataset == "MODIFIED_IGBP_MODIS_NOAH"
    # ... and so does the world reader, which checks the lock too.
    again, read = world_surface.load_world_statics(world, t21_grid)
    for name in fields:
        assert np.array_equal(again[name], fields[name]), name
    assert read["world_folder"] == {
        "folder": "rectangle.world",
        "lock_sha256": world_folder.read_lock(planet)["self_sha256"],
        "map_sha256": world_folder.read_lock(planet)["map"]["sha256"],
        "map_source": {"name": "rectangle test map"},
    }

    shape = t21_grid.shape
    terrain = np.where(fields["LANDMASK"] > 0.5, 500.0, 0.0)
    resolved, detail = statics.resolve_surface_statics(
        fields, document, valid_time=VALID_TIME, latitude_deg=t21_grid.latitude_deg,
        terrain_height_m=terrain, soil_temperature_k=np.full((4, *shape), 290.0),
        skin_temperature_k=np.full(shape, 291.0))
    for name, value in resolved.items():
        assert value.shape == shape and np.isfinite(value).all(), name
    land = fields["LANDMASK"] > 0.5
    water = statics.water_columns(resolved["land_fraction"])
    assert np.array_equal(water, ~land)
    assert np.all(resolved["landuse_category"][land] == 7)
    assert np.all(resolved["landuse_category"][~land] == ISWATER)
    assert np.all(resolved["soil_category_top"][land] == 6)
    assert np.all(resolved["soil_category_bottom"][land] == 6)
    assert np.all(resolved["soil_category_top"][~land] == 14)
    assert np.all(resolved["lake_fraction"] == 0.0)
    assert np.allclose(resolved["vegetation_fraction"][land], 0.70, atol=1e-6)
    assert np.allclose(resolved["leaf_area_index"][land], 2.60, atol=1e-6)
    assert np.allclose(resolved["background_albedo"][land], 0.22, atol=1e-6)
    assert np.allclose(resolved["snow_albedo"][land], 0.65, atol=1e-6)
    deep = _deep(t21_grid)[:, None] - 0.0065 * terrain
    assert np.allclose(resolved["deep_soil_temperature_k"][land],
                       np.broadcast_to(deep, shape)[land], atol=1e-4)
    assert detail["ice_class_columns"] == 0

    # Known answers on the grid: the continent's middle and the basin are
    # land, the cut-off sea is land, the open ocean is water.
    for lat_deg, lon_deg, is_land in ((0.0, 0.0, True), (15.0, 15.0, True),
                                      (32.5, 45.0, True), (0.0, -150.0, False),
                                      (-60.0, 0.0, False)):
        j, i = _column(t21_grid, lat_deg, lon_deg)
        assert bool(land[j, i]) is is_land, (lat_deg, lon_deg)
    # The land share is the rectangle's area share, to the grid's coastline.
    expected = (np.sin(np.deg2rad(NORTH)) - np.sin(np.deg2rad(SOUTH))) / 2.0 * (
        (EAST - WEST) / 360.0)
    assert report["land_area_share"] == pytest.approx(expected, abs=0.01)
    assert report["coastal_columns"] > 0


def test_a_barren_world_takes_the_barren_rows(planet, t21_grid):
    world = _world(str(planet), "barren")
    world_surface.build_world_surface(world, t21_grid, deep_soil_k=_deep(t21_grid))
    fields, document = world_surface.load_world_statics(world, t21_grid)
    land = fields["LANDMASK"] > 0.5
    assert np.all(fields["LU_INDEX"][land] == 16)
    assert np.allclose(fields["ALBEDO12M"][:, land], 38.0)
    assert np.all(fields["ALBEDO12M"][:, ~land] == 8.0)
    assert document["world_surface"]["land_cover"] == "barren"


def test_the_world_reader_refuses_by_name(planet, t21_grid):
    world = _world(str(planet))
    with pytest.raises(FileNotFoundError, match="python -m arwen_global.world.surface"):
        world_surface.load_world_statics(world, t21_grid)
    npz, _, _ = world_surface.build_world_surface(world, t21_grid, deep_soil_k=_deep(t21_grid))
    with pytest.raises(FileExistsError, match="--overwrite"):
        world_surface.build_world_surface(world, t21_grid, deep_soil_k=_deep(t21_grid))
    # The open-shrubland cache is never read for a barren world.
    with pytest.raises(FileNotFoundError, match="barren"):
        world_surface.load_world_statics(_world(str(planet), "barren"), t21_grid)
    # Nor for another grid.
    with pytest.raises(FileNotFoundError, match="no surface for this grid"):
        world_surface.load_world_statics(world, GaussianGrid.create(15, dealias_factor=1.5))
    # A cache replaced behind the lock's back is refused.
    other = npz.with_name("elsewhere.npz")
    npz.rename(other)
    with np.load(other) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["LU_INDEX"] = arrays["LU_INDEX"] * 1.0
    arrays["LU_INDEX"][0, 0] = 7.0
    np.savez_compressed(npz, **arrays)
    with pytest.raises(WorldFolderError, match="not the cache the lock file names"):
        world_surface.load_world_statics(world, t21_grid)
    # Writing a new map drops every surface built from the old one.
    world_folder.write_map(planet, *_map(), overwrite=True)
    assert world_folder.read_lock(planet)["surfaces"] == {}
    with pytest.raises(FileNotFoundError, match="no surface"):
        world_surface.load_world_statics(world, t21_grid)
    # Earth is never read through here.
    with pytest.raises(ValueError, match="read only for a world"):
        world_surface.load_world_statics(parse_world(None), t21_grid)


def test_the_receipt_row_names_the_folder_not_its_path(planet, t21_grid):
    world = _world(str(planet))
    world_surface.build_world_surface(world, t21_grid, deep_soil_k=_deep(t21_grid))
    record = world_folder.folder_record(planet)
    assert record["folder"] == "rectangle.world"
    assert str(planet) not in json.dumps(record)
    assert len(record["surfaces"]) == 1 and record["map_source"] == {"name": "rectangle test map"}


# -- deep soil -----------------------------------------------------------------


def test_earth_zonal_deep_soil_is_taken_at_sea_level_over_land_only():
    earth = {
        "SOILTEMP": np.array([[250.0, 0.0, 0.0], [0.0, 0.0, 0.0], [290.0, 280.0, 0.0]]),
        "HGT_M": np.array([[1000.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1000.0, -50.0]]),
        "LANDMASK": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        "latitude_deg": np.array([60.0, 0.0, -60.0]),
    }
    profile, detail = world_surface.earth_zonal_deep_soil(earth, [80.0, 60.0, 0.0, -60.0])
    # row 60N: 250 + 6.5; row 60S: mean(290, 280 + 6.5); the equator has no
    # land and is interpolated; 80N is beyond the last land row and held.
    assert profile == pytest.approx([256.5, 256.5, (256.5 + 288.25) / 2.0, 288.25])
    assert detail["earth_rows_without_land"] == 1


# -- the config path -------------------------------------------------------------


def _config(tmp_path, world_folder_text: str, land_cover="open-shrubland") -> Path:
    text = Path(_shipped_configs() / "arwen_global_moist_smoke.toml").read_text(encoding="utf-8")
    text += ('\n[statics]\nsource = "real"\nvalid_time = "2026-11-12T00:00:00Z"\n'
             '\n[world]\nlabel = "rectangle"\n[world.sea]\ntemperature = "zonal-mean"\n'
             'ice = "none"\n[world.surface]\n'
             f'folder = "{world_folder_text}"\nland_cover = "{land_cover}"\n'
             'soil_wetness = 0.3\n')
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_command_line_builds_the_surface_a_config_names(tmp_path, t21_grid, capsys):
    from arwen_global.config import load_config

    folder = tmp_path / "rectangle.world"
    world_folder.write_map(folder, *_map())
    # An Earth cache to take the deep soil from: a written one, any grid.
    earth_fields, earth_provenance = world_surface.surface_fields(
        np.where(np.abs(t21_grid.latitude_deg)[:, None] < 30.0, 0.0, 1.0)
        * np.ones(t21_grid.shape), np.zeros(t21_grid.shape), _deep(t21_grid),
        "open-shrubland", t21_grid)
    provenance = {
        "schema": statics.STATICS_SCHEMA, "nlat": 33, "nlon": 66,
        "landuse": dict(world_surface.LANDUSE_ATTRS),
        "arrays": {name: {"shape": list(v.shape), "dtype": v.dtype.str,
                          "sha256": statics._array_hash(v)}
                   for name, v in earth_fields.items()},
    }
    earth_npz, _ = statics.write_cache(tmp_path / "earth" / "earth.npz", earth_fields, provenance)
    config = _config(tmp_path, "rectangle.world")   # relative: beside the config
    assert world_surface.main([str(config), "--earth-statics", str(earth_npz)]) == 0
    out = capsys.readouterr().out
    assert "world surface: T3" in out
    cfg = load_config(config)
    from arwen_global.spectral.grid import GaussianGrid as Grid

    grid = Grid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                       dealias_factor=cfg.dealias_factor)
    fields, document = world_surface.load_world_statics(cfg.world, grid, base=tmp_path)
    assert document["world_surface"]["deep_soil"]["earth_cache_npz_sha256"]
    # the deep soil is the Earth cache's land-row profile, interpolated
    assert np.isfinite(fields["SOILTEMP"]).all()
    # a second build without --overwrite is a refusal, exit 1
    assert world_surface.main([str(config), "--earth-statics", str(earth_npz)]) == 1
    assert "--overwrite" in capsys.readouterr().err


def test_the_statics_seam_reads_the_world_folder_once_it_carries_the_world(tmp_path):
    """Live once the statics seam (package 2.1) carries the world on the
    statics options; skipped before then."""
    if "world" not in statics.StaticsOptions.__dataclass_fields__:
        pytest.skip("the statics seam does not carry the world yet (package 2.1)")
    from arwen_global.config import load_config
    from arwen_global.spectral.grid import GaussianGrid as Grid

    folder = tmp_path / "rectangle.world"
    world_folder.write_map(folder, *_map())
    cfg = load_config(_config(tmp_path, "rectangle.world"))
    grid = Grid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                       dealias_factor=cfg.dealias_factor)
    with pytest.raises(FileNotFoundError, match="no surface for this grid"):
        statics.load_statics(cfg.statics, grid)
    world_surface.build_world_surface(cfg.world, grid, deep_soil_k=_deep(grid), base=tmp_path)
    fields, document = statics.load_statics(cfg.statics, grid)
    assert document["world_surface"]["folder"] == "rectangle.world"
    npz, _ = statics.cache_paths(cfg.statics, grid)
    assert npz.parent == folder and "-world-" in npz.name
    # The receipt's statics row carries the folder's hashes and the rules.
    row = statics.real_provenance(cfg.statics, npz, document, {})
    assert row["world_folder"]["folder"] == "rectangle.world"
    assert row["world_surface"]["land_cover"] == "open-shrubland"
    assert "statics invented" in statics.statics_sentence(cfg)
    # A synthetic-statics run of an invented world loads and hashes, and is
    # refused at the run door: the synthetic planet would ignore the map.
    text = _config(tmp_path, "rectangle.world").read_text(encoding="utf-8")
    synthetic = tmp_path / "synthetic.toml"
    synthetic.write_text(text.replace('source = "real"\nvalid_time = "2026-11-12T00:00:00Z"\n',
                                      'source = "synthetic"\n'), encoding="utf-8")
    loaded = load_config(synthetic)
    assert loaded.config_hash != cfg.config_hash
    with pytest.raises(ValueError, match='needs \\[statics\\] source = "real"'):
        statics.statics_sentence(loaded)
