"""The statics seam of the worlds track (package 2.1).

The world rides on the statics options, and ``load_statics`` (the one
function every caller reads the statics through) applies it: Earth's cache
as it is for Earth, Earth's cache flooded or drowned in memory for an
edited Earth, an invented world's own surface from its own folder for an invented
world.  Known answers on a small hand-written cache on the tiny CPU grid:
no Rust door, no archive, no card."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest

from arwen_global import statics
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.statics import StaticsOptions
from arwen_global.world import EARTH, parse_world
from arwen_global.world.statics_edits import (
    EDIT_KEY, edit_statics, edits_statics, new_water_columns,
)

ISWATER, ISLAKE, ISICE, ISURBAN = 17, 21, 15, 13
ISOILWATER = 14
VALID_TIME = "2026-08-30T18:00:00Z"
SMOKE = config_root() / "arwen_global_moist_smoke.toml"

FLOOD = parse_world({"label": "sea-plus-30", "sea": {"level_m": 30.0}})
WATERWORLD = parse_world({
    "label": "waterworld", "terrain": {"scale": 0.0},
    "sea": {"land": "none", "temperature": "zonal-mean", "ice": "none"}})
NO_MOUNTAINS = parse_world({"label": "flat-earth", "terrain": {"scale": 0.0}})
PANGEA = parse_world({
    "label": "pangea", "surface": {"folder": "pangea.world", "soil_wetness": 0.3},
    "sea": {"temperature": "zonal-mean", "ice": "none"}})


# ---------------------------------------------------------------------------
# a small Earth-like cache, written the way the statics door writes one
# ---------------------------------------------------------------------------

def _earth_fields(grid) -> dict[str, np.ndarray]:
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)[:, None] * np.ones((1, grid.nlon))
    lon = np.asarray(grid.longitude_deg, dtype=np.float64)[None, :] * np.ones((grid.nlat, 1))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    land = ((np.abs(lat) < 60.0) & (((lon > -140.0) & (lon < -40.0))
                                    | ((lon > 0.0) & (lon < 130.0))))
    ice = lat < -65.0                       # an ice sheet, on land
    land |= ice
    lake = ~land & (lat > 60.0) & (lon > 20.0) & (lon < 80.0)
    # Lowlands near the equator, highlands poleward, one depression.
    height = np.where(land, 4.0 * np.abs(lat) - 15.0, 0.0)
    height = np.where(ice, 2500.0, height)
    shape = land.shape
    luf = np.zeros((21, *shape))
    luf[6] = np.where(land & ~ice, 0.8, 0.0)              # class 7
    luf[9] = np.where(land & ~ice, 0.2, 0.0)              # class 10
    luf[ISICE - 1] = np.where(ice, 1.0, 0.0)
    luf[ISLAKE - 1] = np.where(lake, 0.9, 0.0)
    luf[ISWATER - 1] = np.where(lake, 0.1, np.where(land, 0.0, 1.0))
    category = np.where(ice, ISICE, np.where(land, 7, np.where(lake, ISLAKE, ISWATER)))
    soil = np.where(land & ~ice, 8, ISOILWATER)            # the sheets coded water
    months = np.arange(12, dtype=np.float64)[:, None, None]
    fields = {
        "HGT_M": height,
        "LANDMASK": land.astype(np.float64),
        "LU_INDEX": category.astype(np.float64),
        "SCT_DOM": soil.astype(np.float64),
        "SCB_DOM": soil.astype(np.float64),
        "LANDUSEF": luf,
        "GREENFRAC": np.where(land, 0.05 * (months + 1), 0.0),
        "LAI12M": np.where(land, 0.5 * (months + 1), 0.0),
        "ALBEDO12M": np.where(land, 15.0 + months, 8.0),
        "SNOALB": np.where(land, 60.0, 0.0),
        "SOILTEMP": np.where(land, 285.0 - 0.3 * np.abs(lat), 0.0),
    }
    out = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in fields.items()}
    out["latitude_deg"] = np.ascontiguousarray(grid.latitude_deg, dtype=np.float64)
    out["longitude_deg"] = np.ascontiguousarray(grid.longitude_deg, dtype=np.float64)
    return out


def _provenance(grid, fields) -> dict:
    return {
        "schema": statics.STATICS_SCHEMA,
        "truncation": int(grid.truncation), "nlat": int(grid.nlat), "nlon": int(grid.nlon),
        "geog_data_res": ["default"],
        "landuse": {"mminlu": "MODIFIED_IGBP_MODIS_NOAH", "iswater": ISWATER,
                    "islake": ISLAKE, "isice": ISICE, "isurban": ISURBAN},
        "arrays": {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                          "sha256": statics._array_hash(value)}
                   for name, value in fields.items()},
        "coverage": {},
    }


@pytest.fixture(scope="module")
def grid():
    return GaussianGrid.create(21, dealias_factor=1.5)


@pytest.fixture()
def earth_cache(grid, tmp_path):
    """``(options, npz)``: Earth's cache where Earth's options look for it."""
    options = StaticsOptions(source="real", cache_dir=str(tmp_path / "cache"))
    fields = _earth_fields(grid)
    npz, _ = statics.write_cache(statics.cache_paths(options, grid)[0], fields,
                                 _provenance(grid, fields))
    return options, npz


def _with(options, world, **extra):
    return dataclasses.replace(options, world=world, **extra)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(fields, provenance, grid):
    shape = (grid.nlat, grid.nlon)
    return statics.resolve_surface_statics(
        fields, provenance, valid_time=statics.parse_utc(VALID_TIME, "t"),
        latitude_deg=grid.latitude_deg, terrain_height_m=np.zeros(shape),
        soil_temperature_k=np.full((4, *shape), 290.0),
        skin_temperature_k=np.full(shape, 291.0))


# ---------------------------------------------------------------------------
# Earth, and worlds that keep Earth's surface
# ---------------------------------------------------------------------------

def test_earth_reads_the_cache_exactly_as_before(earth_cache, grid):
    options, npz = earth_cache
    assert options.world is EARTH and options.world_folder is None
    assert options.identity == {"source": "real", "geog_data_res": "default"}
    before = statics.read_cache(npz, grid)
    for world in (EARTH, NO_MOUNTAINS):
        fields, document = statics.load_statics(_with(options, world), grid)
        assert document == before[1]
        assert set(fields) == set(before[0])
        for name, value in fields.items():
            assert value.dtype == before[0][name].dtype
            assert np.array_equal(value, before[0][name]), name
        assert statics.cache_paths(_with(options, world), grid)[0] == npz
    # the world never enters the statics identity (it joins the config's)
    assert _with(options, FLOOD).identity == options.identity
    assert not edits_statics(EARTH) and not edits_statics(NO_MOUNTAINS)
    assert not edits_statics(PANGEA)
    assert edits_statics(FLOOD) and edits_statics(WATERWORLD)


# ---------------------------------------------------------------------------
# the flood
# ---------------------------------------------------------------------------

def test_flood_turns_exactly_the_land_at_or_below_the_sea_level_into_sea(earth_cache, grid):
    options, npz = earth_cache
    on_disk = _file_sha(npz)
    earth, earth_doc = statics.read_cache(npz, grid)
    fields, document = statics.load_statics(_with(options, FLOOD), grid)
    land = earth["LANDMASK"] > 0.5
    expected = land & (earth["HGT_M"].astype(np.float64) <= 30.0)
    # the known answer, counted independently of the edit
    assert 0 < int(expected.sum()) < int(land.sum())
    assert int(expected.sum()) == int((land & (np.asarray(earth["HGT_M"]) <= np.float32(30.0))).sum())
    record = document[EDIT_KEY]
    assert record["new_water_columns"] == int(expected.sum())
    assert record["earth_land_columns"] == int(land.sum())
    assert record["land_columns_after"] == int(land.sum() - expected.sum())
    assert record["lake_columns_to_sea"] == 0
    assert record["world"] == FLOOD.identity_hash and record["label"] == "sea-plus-30"
    after_land = fields["LANDMASK"] > 0.5
    assert np.array_equal(after_land, land & ~expected)
    assert np.array_equal(new_water_columns(earth, earth_doc, FLOOD), expected)
    # the new sea carries the sea's values, field by field
    assert np.all(fields["LU_INDEX"][expected] == ISWATER)
    assert np.all(fields["SCT_DOM"][expected] == ISOILWATER)
    assert np.all(fields["SCB_DOM"][expected] == ISOILWATER)
    assert np.all(fields["LANDUSEF"][ISWATER - 1][expected] == 1.0)
    assert np.all(fields["LANDUSEF"][:, expected].sum(axis=0) == 1.0)
    assert np.all(fields["GREENFRAC"][:, expected] == 0.0)
    assert np.all(fields["LAI12M"][:, expected] == 0.0)
    assert np.all(fields["ALBEDO12M"][:, expected] == 8.0)
    assert np.all(fields["SNOALB"][expected] == 0.0)
    assert np.all(fields["SOILTEMP"][expected] == 0.0)
    # the seabed keeps its height (the rule is recountable from the edit)
    assert np.array_equal(fields["HGT_M"], earth["HGT_M"])
    # every other column, and every array's shape and dtype, is Earth's
    for name, value in fields.items():
        assert value.shape == earth[name].shape and value.dtype == earth[name].dtype
        if value.ndim >= 2 and value.shape[-2:] == land.shape:
            assert np.array_equal(value[..., ~expected], earth[name][..., ~expected]), name
        else:
            assert np.array_equal(value, earth[name]), name
    assert sorted(record["edited_arrays"]) == sorted(
        n for n in fields if not np.array_equal(fields[n], earth[n]))
    # Earth's cache on disk is untouched and still describes the file
    assert _file_sha(npz) == on_disk
    assert document["npz_sha256"] == earth_doc["npz_sha256"]
    assert document["arrays"] == earth_doc["arrays"]
    assert EDIT_KEY not in earth_doc


def test_waterworld_is_sea_on_every_column(earth_cache, grid):
    options, npz = earth_cache
    earth, _ = statics.read_cache(npz, grid)
    fields, document = statics.load_statics(_with(options, WATERWORLD), grid)
    land = earth["LANDMASK"] > 0.5
    lake = ~land & (earth["LANDUSEF"][ISLAKE - 1] > earth["LANDUSEF"][ISWATER - 1])
    assert land.any() and lake.any()
    record = document[EDIT_KEY]
    assert record["new_water_columns"] == int(land.sum())
    assert record["lake_columns_to_sea"] == int(lake.sum())
    assert record["land_columns_after"] == 0
    assert np.all(fields["LANDMASK"] == 0.0)
    assert np.all(fields["LU_INDEX"] == ISWATER)
    assert np.all(fields["LANDUSEF"][ISWATER - 1] == 1.0)
    assert np.all(fields["LANDUSEF"][ISLAKE - 1] == 0.0)
    assert np.all(fields["HGT_M"] == 0.0)
    assert np.all(fields["ALBEDO12M"] == 8.0)
    assert statics.lake_columns(np.zeros(land.shape), fields["LANDUSEF"][ISLAKE - 1]).sum() == 0


@pytest.mark.parametrize("world", [FLOOD, WATERWORLD], ids=["flood", "waterworld"])
def test_edited_planes_pass_the_mask_against_category_check(earth_cache, grid, world):
    options, _ = earth_cache
    fields, document = statics.load_statics(_with(options, world), grid)
    resolved, detail = _resolve(fields, document, grid)    # the check raises on a mismatch
    water = fields["LANDMASK"] < 0.5
    assert np.array_equal(statics.water_columns(resolved["land_fraction"]), water)
    assert np.array_equal(resolved["landuse_category"] == ISWATER, water)
    assert np.array_equal(resolved["soil_category_top"] == ISOILWATER, water)
    assert detail["land_columns"] == document[EDIT_KEY]["land_columns_after"]
    # the receipt's statics row carries the edit; Earth's carries none
    row = statics.real_provenance(_with(options, world), Path("x.npz"), document, detail)
    assert row[EDIT_KEY] == document[EDIT_KEY]
    earth_fields, earth_doc = statics.load_statics(options, grid)
    _, earth_detail = _resolve(earth_fields, earth_doc, grid)
    earth_row = statics.real_provenance(options, Path("x.npz"), earth_doc, earth_detail)
    assert EDIT_KEY not in earth_row and statics.WORLD_CACHE_KEY not in earth_row


def test_the_mask_against_category_check_still_bites_on_a_bad_edit(earth_cache, grid):
    options, npz = earth_cache
    earth, _ = statics.read_cache(npz, grid)
    fields, document = statics.load_statics(_with(options, FLOOD), grid)
    flooded = np.argwhere((fields["LANDMASK"] < 0.5) & (earth["LANDMASK"] > 0.5))
    assert len(flooded)
    j, i = (int(v) for v in flooded[0])
    fields["LU_INDEX"][j, i] = 7     # a new sea column left with a land category
    with pytest.raises(ValueError, match="disagree between the static land mask"):
        _resolve(fields, document, grid)


def test_an_edited_earth_refuses_a_cache_that_carries_a_world(earth_cache, grid):
    options, npz = earth_cache
    fields, document = statics.read_cache(npz, grid)
    del document["self_sha256"], document["npz_sha256"]
    document[statics.WORLD_CACHE_KEY] = PANGEA.record()
    statics.write_cache(npz, fields, document, overwrite=True)
    with pytest.raises(ValueError, match="carries a world record"):
        statics.load_statics(_with(options, FLOOD), grid)
    with pytest.raises(ValueError, match="only Earth's cache is ever edited"):
        edit_statics(fields, document, FLOOD)
    with pytest.raises(ValueError, match="does not edit Earth's statics"):
        edit_statics(fields, document, NO_MOUNTAINS)


# ---------------------------------------------------------------------------
# an invented world: its own cache, never Earth's
# ---------------------------------------------------------------------------

def _invented_map():
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    elevation = np.where((np.abs(la) < 40.0) & (np.abs(lo) < 60.0), 500.0, -4000.0)
    return elevation, lat, lon


def test_an_invented_world_reads_only_its_own_folder(earth_cache, grid, tmp_path):
    """The seam hands an invented world to the surface built from its own
    map in its own folder (package 2.4, arwen_global.world.surface), and
    never to Earth's cache, present or not."""
    from arwen_global.world import folder as world_folder
    from arwen_global.world import surface as world_surface

    options, npz = earth_cache
    folder = tmp_path / "pangea.world"
    world_options = _with(options, PANGEA, world_folder=str(folder))
    # no folder yet: refused by name, Earth's cache present beside it unread
    with pytest.raises(FileNotFoundError, match="does not exist"):
        statics.load_statics(world_options, grid)
    world_folder.write_map(folder, *_invented_map(), source={"name": "test map"})
    with pytest.raises(FileNotFoundError, match="no surface for this grid"):
        statics.load_statics(world_options, grid)
    deep = 300.0 - 40.0 * np.sin(np.deg2rad(grid.latitude_deg)) ** 2
    world_surface.build_world_surface(PANGEA, grid, deep_soil_k=deep, base=tmp_path)
    fields, document = statics.load_statics(world_options, grid)
    path, sidecar = statics.cache_paths(world_options, grid)
    assert path.parent == folder and sidecar == path.with_suffix(".json")
    assert path != npz and "default" not in path.name
    assert document[statics.WORLD_FOLDER_KEY]["folder"] == "pangea.world"
    assert EDIT_KEY not in document
    row = statics.real_provenance(world_options, path, document, {})
    assert row[statics.WORLD_CACHE_KEY]["land_cover"] == "open-shrubland"
    assert row[statics.WORLD_FOLDER_KEY]["folder"] == "pangea.world"
    # the folder as written, when the config did not resolve one
    assert statics.world_folder_of(_with(options, PANGEA)) == Path("pangea.world")
    with pytest.raises(ValueError, match="no folder of its own"):
        statics.world_folder_of(_with(options, FLOOD))


def test_the_statics_door_never_builds_a_world_surface(grid):
    # the statics door builds Earth's cache only: never into a world's name
    with pytest.raises(ValueError, match=r"never built from\s+the WPS_GEOG archive"):
        statics.build_statics(grid, StaticsOptions(source="real", world=PANGEA))


# ---------------------------------------------------------------------------
# every caller reads through the seam
# ---------------------------------------------------------------------------

#: The four callers that load the statics (plan 4.4), and where.
CALLERS = ("analysis_initial", "initial_conditions", "runner", "surface_series")


def test_every_caller_loads_the_statics_through_the_seam():
    package = Path(statics.__file__).parent
    for name in CALLERS:
        text = (package / f"{name}.py").read_text(encoding="utf-8")
        assert re.search(r"load_statics\(cfg\.statics,", text), name
    # and no run module opens a statics cache itself, which would read
    # Earth's arrays past the world.  The one exception is the long-run
    # line's sub-grid orography cache (orographic_statics.py), which only
    # the orographic drag reads, and the drag is refused under a world that
    # changes its terrain or its coast (the test below).
    for path in package.glob("*.py"):
        if path.name not in ("statics.py", "orographic_statics.py"):
            assert "read_cache(" not in path.read_text(encoding="utf-8"), path


@pytest.mark.parametrize("world", [NO_MOUNTAINS, FLOOD], ids=["terrain", "coast"])
def test_the_orographic_drag_is_refused_under_a_world_of_its_own_terrain(world):
    from types import SimpleNamespace

    from arwen_global.physics.orographic_drag import attach_orographic_statistics

    adapter = SimpleNamespace(options=SimpleNamespace(orographic_drag_enabled=True))
    cfg = SimpleNamespace(statics=SimpleNamespace(source="real"), world=world)
    with pytest.raises(ValueError, match="sub-grid orography statistics are built "
                                         "from Earth's terrain"):
        attach_orographic_statistics(adapter, cfg)
    # with the drag off nothing is read and nothing is refused
    adapter.options.orographic_drag_enabled = False
    assert attach_orographic_statistics(adapter, cfg) is None


def _analytic_config(tmp_path, cache_dir, world_table: str) -> Path:
    text = SMOKE.read_text(encoding="utf-8")
    body = (f'\n[statics]\nsource = "real"\nvalid_time = "{VALID_TIME}"\n'
            f'cache_dir = "{cache_dir.as_posix()}"\n' + world_table)
    path = tmp_path / "run.toml"
    path.write_text(text + body, encoding="utf-8")
    return path


def _carrying(cfg):
    """The run config with its world on the statics options.  load_config
    does this itself once its hook is in (patches/w2-1-for-w0-1.patch); the
    replace is the identity on a config that already carries it."""
    return dataclasses.replace(cfg, statics=dataclasses.replace(cfg.statics, world=cfg.world))


WATERWORLD_TABLE = (
    '\n[world]\nlabel = "waterworld"\n[world.terrain]\nscale = 0.0\n'
    '[world.sea]\nland = "none"\ntemperature = "zonal-mean"\nice = "none"\n')


def test_a_cold_start_sees_the_edited_mask(tmp_path):
    from arwen_global.initial_conditions import analytic_initial_state
    from arwen_global.runner import build_transform

    cache_dir = tmp_path / "cache"
    earth_cfg = load_config(_analytic_config(tmp_path, cache_dir, ""))
    transform = build_transform(earth_cfg)
    grid = transform.grid
    fields = _earth_fields(grid)
    statics.write_cache(statics.cache_paths(earth_cfg.statics, grid)[0], fields,
                        _provenance(grid, fields))
    host = transform.backend.to_numpy

    state, _phi, provenance = analytic_initial_state(earth_cfg, transform)
    earth_land = host(state.surface.land_fraction) > 0.5
    assert earth_land.any()
    assert EDIT_KEY not in provenance["statics"]

    cfg = _carrying(load_config(_analytic_config(tmp_path, cache_dir, WATERWORLD_TABLE)))
    assert cfg.statics.world == cfg.world == WATERWORLD
    state, _phi, provenance = analytic_initial_state(cfg, transform)
    land_fraction = host(state.surface.land_fraction)
    assert np.all(land_fraction <= 0.5)
    assert np.all(host(state.surface.landuse_category) == ISWATER)
    assert provenance["statics"][EDIT_KEY]["new_water_columns"] == int(earth_land.sum())
    assert "edits it in memory: waterworld" in statics.statics_sentence(cfg)


def test_an_analysis_cold_start_sees_the_edited_mask(tmp_path):
    from arwen_global import coldstart_bridge
    from arwen_global.analysis_initial import analysis_initial_state
    from arwen_global.doors import DoorMissing
    from arwen_global.runner import build_transform
    from test_arwen_global_statics import _analysis_frame, _config

    try:
        coldstart_bridge.load()
    except DoorMissing as exc:
        pytest.skip(f"the analysis cold start needs its Rust door: {exc}")
    cache_dir = tmp_path / "cache"
    body = f'[statics]\ncache_dir = "{cache_dir.as_posix()}"\n'
    earth_cfg = load_config(_config(tmp_path, body, analysis=True))
    transform = build_transform(earth_cfg)
    grid = transform.grid
    fields = _earth_fields(grid)
    statics.write_cache(statics.cache_paths(earth_cfg.statics, grid)[0], fields,
                        _provenance(grid, fields))
    host = transform.backend.to_numpy
    state, _phi, provenance = analysis_initial_state(earth_cfg, transform, frame=_analysis_frame())
    earth_land = host(state.surface.land_fraction) > 0.5
    assert earth_land.any() and EDIT_KEY not in provenance["statics"]

    flood = '\n[world]\nlabel = "sea-plus-30"\n[world.sea]\nlevel_m = 30.0\n'
    cfg = _carrying(load_config(_config(tmp_path, body + flood, analysis=True)))
    state, _phi, provenance = analysis_initial_state(cfg, transform, frame=_analysis_frame())
    land = host(state.surface.land_fraction) > 0.5
    expected = earth_land & ~(np.asarray(fields["HGT_M"], dtype=np.float64) <= 30.0)
    assert np.array_equal(land, expected)
    assert int((earth_land & ~land).sum()) == provenance["statics"][EDIT_KEY]["new_water_columns"]
    categories = host(state.surface.landuse_category)
    assert np.all(categories[earth_land & ~land] == ISWATER)


def test_a_coast_world_on_the_synthetic_planet_is_refused_at_the_cold_start():
    options = StaticsOptions()
    assert statics.synthetic_provenance(_with(options, NO_MOUNTAINS))["source"] == "synthetic"
    for world in (FLOOD, WATERWORLD, PANGEA):
        with pytest.raises(ValueError, match="changes the coast"):
            statics.synthetic_provenance(_with(options, world))
