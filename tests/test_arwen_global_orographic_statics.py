"""The sub-grid orography the orographic drag reads, on the Gaussian grid.

``arwen_global.orographic_statics`` reads WPS_GEOG's ``orogwd_10m`` itself,
because the engine's static-fields crate refuses the Gaussian rows grid for
these rows (MEASURED on gpuwm 2.9.13: ``orographic WPS fields need a
projected grid``).  What it must do is what GEOGRID.TBL.ARW's rows say:
``average_4pt`` of the source, zero on water.  These tests write a small
WPS tile whose every value is known and check the answers by hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arwen_global.orographic_statics import (
    SSO_DATASET, SSO_FIELDS, SSO_KIND, build_orographic_statistics,
    cache_paths, load_orographic_statistics, read_global_tile,
    sample_average_4pt)
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.statics import (STATICS_SCHEMA, StaticsOptions, _array_hash,
                                  cache_paths as statics_cache_paths,
                                  write_cache)

#: An 8 x 4 global source ring: 45-degree cells, the first row centred at
#: 67.5 S, the first column at 22.5 E, as orogwd_10m's own index places its
#: first cell half a cell in from the corner.
NX, NY, STEP = 8, 4, 45.0


def _write_field(directory: Path, values: np.ndarray, *, signed: bool,
                 scale: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index").write_text(
        "type = continuous\n"
        f"signed = {'yes' if signed else 'no'}\n"
        "projection = regular_ll\n"
        f"dx = {STEP}\ndy = {STEP}\n"
        "known_x = 1.0\nknown_y = 1.0\n"
        "known_lat = -67.5\nknown_lon = 22.5\n"
        f"scale_factor = {scale}\n"
        "wordsize = 2\n"
        f"tile_x = {NX}\ntile_y = {NY}\ntile_z = 1\ntile_bdr=0\n"
        'units="whoknows"\ndescription="something"\n', encoding="utf-8")
    raw = np.rint(values / scale).astype(">i2" if signed else ">u2")
    raw.tofile(directory / f"00001-{NX:05d}.00001-{NY:05d}")


def _source() -> np.ndarray:
    j, i = np.mgrid[0:NY, 0:NX]
    return (10.0 * j + i).astype(np.float64)


def test_the_tile_reads_as_wps_writes_it(tmp_path):
    values = _source() - 12.0
    _write_field(tmp_path / "f", values, signed=True, scale=0.5)
    got, index, digest = read_global_tile(tmp_path / "f")
    assert np.array_equal(got, values)
    assert index["projection"] == "regular_ll"
    assert len(digest) == 64


def test_average_4pt_is_the_mean_of_the_four_surrounding_points(tmp_path):
    _write_field(tmp_path / "f", _source(), signed=False, scale=1.0)
    values, index, _ = read_global_tile(tmp_path / "f")
    lat = np.array([-45.0, -45.0, -67.5, -10.0, 67.5])
    lon = np.array([45.0, 0.0, 22.5, 112.5, 359.0])
    got = sample_average_4pt(values, index, lat, lon)
    # (-45, 45): between columns 0,1 and rows 0,1 -> (0 + 1 + 10 + 11) / 4
    # (-45, 0): across the date line, columns 7 and 0 -> (7 + 0 + 17 + 10) / 4
    # (-67.5, 22.5): on a source node, which counts it four times -> 0
    # (-10, 112.5): on a column node between rows 1 and 2 -> (12 + 12 + 22 + 22) / 4
    # (67.5, 359): on the last row (counted twice), columns 7 and 0 -> (37 + 30) / 2
    assert got.tolist() == [5.5, 8.5, 0.0, 17.0, 33.5]


def _grid():
    return GaussianGrid.create(3)


def _fake_land_cache(tmp_path, grid, land) -> StaticsOptions:
    options = StaticsOptions(source="real", geog_root=str(tmp_path / "geog"),
                             cache_dir=str(tmp_path / "cache"))
    fields = {
        "LANDMASK": land.astype(np.float32),
        "latitude_deg": np.ascontiguousarray(grid.latitude_deg, np.float64),
        "longitude_deg": np.ascontiguousarray(grid.longitude_deg, np.float64),
    }
    provenance = {
        "schema": STATICS_SCHEMA, "truncation": int(grid.truncation),
        "nlat": int(grid.nlat), "nlon": int(grid.nlon),
        "arrays": {name: {"shape": list(v.shape), "dtype": v.dtype.str,
                          "sha256": _array_hash(v)} for name, v in fields.items()},
    }
    npz, _ = statics_cache_paths(options, grid)
    write_cache(npz, fields, provenance)
    return options


def _fake_geog(tmp_path):
    for k, name in enumerate(SSO_FIELDS):
        signed = name.startswith("OA")
        values = _source() * (k + 1) * (-1.0 if signed else 1.0)
        _write_field(tmp_path / "geog" / SSO_DATASET / name.lower(), values,
                     signed=signed, scale=0.25)


def test_the_build_samples_every_plane_and_zeroes_the_water(tmp_path):
    grid = _grid()
    land = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    land[1:-1, ::2] = True
    options = _fake_land_cache(tmp_path, grid, land)
    _fake_geog(tmp_path)
    fields, provenance = build_orographic_statistics(grid, options)
    lat2 = np.broadcast_to(grid.latitude_deg[:, None], land.shape)
    lon2 = np.broadcast_to(grid.longitude_deg[None, :], land.shape)
    for k, name in enumerate(SSO_FIELDS):
        directory = tmp_path / "geog" / SSO_DATASET / name.lower()
        values, index, digest = read_global_tile(directory)
        want = np.where(land, sample_average_4pt(values, index, lat2, lon2), 0.0)
        assert fields[name].dtype == np.float32
        assert np.array_equal(fields[name], want.astype(np.float32)), name
        assert provenance["sources"][name]["tile_sha256"] == digest
    assert (fields["VAR"][~land] == 0.0).all() and (fields["VAR"][land] > 0.0).all()
    assert provenance["kind"] == SSO_KIND
    assert provenance["water_mask"]["water_columns"] == int((~land).sum())


def test_the_cache_round_trips_and_is_refused_when_missing(tmp_path):
    grid = _grid()
    land = np.ones((grid.nlat, grid.nlon), dtype=bool)
    options = _fake_land_cache(tmp_path, grid, land)
    with pytest.raises(FileNotFoundError, match="--orographic"):
        load_orographic_statistics(options, grid)
    _fake_geog(tmp_path)
    fields, provenance = build_orographic_statistics(grid, options)
    npz, sidecar = cache_paths(options, grid)
    assert npz.name.endswith(f"-{SSO_DATASET}.npz")
    write_cache(npz, fields, provenance)
    back, document = load_orographic_statistics(options, grid)
    for name in SSO_FIELDS:
        assert np.array_equal(back[name], fields[name])
    assert document["kind"] == SSO_KIND and document["self_sha256"]


def test_a_tree_without_the_dataset_is_refused_with_the_remedy(tmp_path):
    grid = _grid()
    options = _fake_land_cache(tmp_path, grid,
                               np.ones((grid.nlat, grid.nlon), dtype=bool))
    (tmp_path / "geog").mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError) as refused:
        build_orographic_statistics(grid, options)
    # The remedy names the one dataset, its source and the tree that lacks
    # it, as the run door's own sentence does.
    message = str(refused.value)
    assert "gpuwm fetch-geog --root " in message
    assert "--datasets orogwd_10m --source ncar" in message
    assert str(tmp_path / "geog") in message


# ---------------------------------------------------------------------------
# Built when a run needs it (0.2.3): the shipped forecast days turn the drag
# on, and a land cache built by an earlier release has no drag cache beside it.
# ---------------------------------------------------------------------------

def _ensure(options, grid, **kwargs):
    from arwen_global.orographic_statics import ensure_orographic_statistics

    said = []
    kwargs.setdefault("fetch", None)
    out = ensure_orographic_statistics(options, grid, say=said.append, **kwargs)
    return out, said


def test_a_missing_cache_is_built_from_the_tree_and_then_left_alone(tmp_path):
    grid = _grid()
    land = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    land[1:-1, ::2] = True
    options = _fake_land_cache(tmp_path, grid, land)
    _fake_geog(tmp_path)
    written, said = _ensure(options, grid)
    npz, _ = cache_paths(options, grid)
    assert written == npz and npz.is_file()
    assert any("built the orographic drag's sub-grid orography" in line for line in said)
    fields, _document = load_orographic_statistics(options, grid)
    want, _ = build_orographic_statistics(grid, options)
    for name in SSO_FIELDS:
        assert np.array_equal(fields[name], want[name]), name
    assert not list(npz.parent.glob(".*.lock"))
    again, said = _ensure(options, grid)
    assert again is None and said == []


def test_the_tree_the_land_cache_records_is_read_when_the_config_names_none(tmp_path):
    from dataclasses import replace

    grid = _grid()
    land = np.ones((grid.nlat, grid.nlon), dtype=np.float32)
    unnamed = StaticsOptions(source="real", cache_dir=str(tmp_path / "cache"))
    fields = {
        "LANDMASK": land,
        "latitude_deg": np.ascontiguousarray(grid.latitude_deg, np.float64),
        "longitude_deg": np.ascontiguousarray(grid.longitude_deg, np.float64),
    }
    provenance = {
        "schema": STATICS_SCHEMA, "truncation": int(grid.truncation),
        "nlat": int(grid.nlat), "nlon": int(grid.nlon),
        "geog_root": str(tmp_path / "geog"),
        "arrays": {name: {"shape": list(v.shape), "dtype": v.dtype.str,
                          "sha256": _array_hash(v)} for name, v in fields.items()},
    }
    land_npz, _ = statics_cache_paths(unnamed, grid)
    write_cache(land_npz, fields, provenance)
    _fake_geog(tmp_path)
    assert replace(unnamed, geog_root=None) == unnamed
    written, _said = _ensure(unnamed, grid)
    assert written is not None
    _fields, built = load_orographic_statistics(unnamed, grid)
    assert built["geog_root"] == str(tmp_path / "geog")


def test_a_tree_without_the_dataset_has_it_staged_by_the_engine_fetch(tmp_path):
    grid = _grid()
    options = _fake_land_cache(tmp_path, grid, np.ones((grid.nlat, grid.nlon), dtype=bool))
    (tmp_path / "geog").mkdir()
    fetched = []

    def fetch(root):
        fetched.append(Path(root))
        _fake_geog(tmp_path)

    written, said = _ensure(options, grid, fetch=fetch)
    assert fetched == [tmp_path / "geog"] and written is not None
    assert any("gpuwm fetch-geog --datasets orogwd_10m --source ncar" in line for line in said)


def test_what_cannot_be_built_is_one_sentence_naming_the_commands_and_the_switch(tmp_path):
    grid = _grid()
    options = _fake_land_cache(tmp_path, grid, np.ones((grid.nlat, grid.nlon), dtype=bool))
    (tmp_path / "geog").mkdir()

    def offline(root):
        raise OSError("`gpuwm fetch-geog ...` exited 1: no route to host")

    for fetch in (None, offline):
        with pytest.raises(FileNotFoundError) as caught:
            _ensure(options, grid, fetch=fetch)
        sentence = str(caught.value)
        assert "\n" not in sentence
        assert "gpuwm fetch-geog --datasets orogwd_10m --source ncar" in sentence
        assert "gpuwm-global statics CONFIG --orographic" in sentence
        assert 'orographic_drag = "none"' in sentence
        npz, _ = cache_paths(options, grid)
        assert not npz.exists() and not list(npz.parent.glob(".*.lock"))


def test_without_a_land_cache_the_land_refusal_speaks_first(tmp_path):
    grid = _grid()
    options = StaticsOptions(source="real", geog_root=str(tmp_path / "geog"),
                             cache_dir=str(tmp_path / "cache"))
    _fake_geog(tmp_path)
    assert _ensure(options, grid) == (None, [])
    npz, _ = cache_paths(options, grid)
    assert not npz.exists()


def test_a_build_another_process_holds_is_waited_for(tmp_path, monkeypatch):
    from arwen_global import orographic_statics

    grid = _grid()
    options = _fake_land_cache(tmp_path, grid, np.ones((grid.nlat, grid.nlon), dtype=bool))
    npz, _ = cache_paths(options, grid)
    lock = npz.with_name(f".{npz.name}.lock")
    lock.write_text("4242\n", encoding="utf-8")
    monkeypatch.setattr(orographic_statics, "LOCK_WAIT_S", 0.0)
    with pytest.raises(FileNotFoundError, match="build lock"):
        _ensure(options, grid)
    assert lock.exists()


def test_the_fetch_is_the_engines_own_command_for_the_one_dataset(tmp_path):
    import sys

    from arwen_global.orographic_statics import fetch_command

    command = fetch_command(tmp_path)
    assert command[:4] == [sys.executable, "-m", "gpuwm", "fetch-geog"]
    assert command[4:] == ["--root", str(tmp_path), "--datasets", "orogwd_10m",
                           "--source", "ncar"]
