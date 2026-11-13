"""The monthly sea-surface climatology: its time interpolation, its regrid,
its pins and its file.

What these hold, each against an answer worked out by hand:

* the mid-month interpolation: plane m exactly at mid-month m, the mean of
  two planes exactly halfway, February 2028's 29 days, and a 31 December to
  1 January step that interpolates December to January;
* the regrid keeps the area mean of an analytic monthly field on a T3 and a
  T21 Gaussian grid within bilinear interpolation's bound, and reads only
  the source's own water;
* a staged file with another byte count or another SHA-256 is refused
  before any of it is decoded, by build and by fetch;
* the WOA18 row is found in the regional engine's ocean cache when a file
  with the pinned bytes is there, and its pins are the engine's;
* a build from NetCDF files through the Rust door carries every input's
  SHA-256 and writes the same bytes twice.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from arwen_global import surface_climatology as sc
from arwen_global.spectral.grid import GaussianGrid

UTC = timezone.utc


# ---------------------------------------------------------------------------
# 1. mid-month interpolation
# ---------------------------------------------------------------------------

def _planes():
    rng = np.random.default_rng(20261112)
    return (280.0 + 10.0 * rng.random((12, 3, 4))).astype(np.float32)


def test_mid_month_is_the_midpoint_of_the_real_calendar_month():
    assert sc.mid_month(2027, 1) == datetime(2027, 1, 16, 12, tzinfo=UTC)
    assert sc.mid_month(2027, 2) == datetime(2027, 2, 15, 0, tzinfo=UTC)
    # February 2028 has 29 days, so its midpoint is 14.5 days in.
    assert sc.mid_month(2028, 2) == datetime(2028, 2, 15, 12, tzinfo=UTC)
    assert sc.mid_month(2027, 4) == datetime(2027, 4, 16, 0, tzinfo=UTC)


def test_at_mid_month_the_field_is_that_months_plane_exactly():
    planes = _planes()
    for month in range(1, 13):
        when = sc.mid_month(2027, month)
        i0, i1, w = sc.month_weights(when)
        assert (i0, w) == (month - 1, 0.0)
        assert np.array_equal(sc.interpolate(planes, when), planes[month - 1])


def test_halfway_between_two_mid_months_the_field_is_their_mean():
    planes = _planes()
    for year in (2027, 2028):
        for month in range(1, 13):
            t0 = sc.mid_month(year, month)
            t1 = sc.mid_month(year + (month == 12), month % 12 + 1)
            i0, i1, w = sc.month_weights(t0 + (t1 - t0) / 2)
            assert (i0, i1, w) == (month - 1, month % 12, 0.5)
            mean = (planes[i0] + planes[i1]) * np.float32(0.5)
            assert np.array_equal(sc.interpolate(planes, t0 + (t1 - t0) / 2), mean)


def test_february_2028_uses_its_29_days():
    # Mid-January to mid-February is 15.5 + 14.5 = 30 days in 2028 and
    # 15.5 + 14 = 29.5 days in 2027; one day after mid-January the weight
    # is 1/30 and 1/29.5.
    for year, span_days in ((2028, 30.0), (2027, 29.5)):
        when = sc.mid_month(year, 1) + timedelta(days=1)
        assert sc.month_weights(when) == (0, 1, 1.0 / span_days)
    # 29 February 2028 lies between mid-February and mid-March.
    i0, i1, w = sc.month_weights(datetime(2028, 2, 29, 12, tzinfo=UTC))
    assert (i0, i1) == (1, 2)
    assert w == pytest.approx(14.0 / (14.5 + 15.5), abs=0.0)


def test_a_year_end_interpolates_december_to_january():
    planes = _planes()
    when = datetime(2027, 12, 31, 18, tzinfo=UTC)
    i0, i1, w = sc.month_weights(when)
    # mid-December 2027 is 16 Dec 12Z, mid-January 2028 is 16 Jan 12Z.
    assert (i0, i1) == (11, 0)
    assert w == 15.25 / 31.0
    after = datetime(2028, 1, 1, 6, tzinfo=UTC)
    assert sc.month_weights(after)[:2] == (11, 0)
    assert sc.month_weights(after)[2] == 15.75 / 31.0
    expected = planes[11] * np.float32(1.0 - w) + planes[0] * np.float32(w)
    assert np.array_equal(sc.interpolate(planes, when), expected)
    # Continuous in seconds: no step at midnight.
    a = sc.month_weights(datetime(2027, 12, 31, 23, 59, 59, tzinfo=UTC))[2]
    b = sc.month_weights(datetime(2028, 1, 1, 0, 0, 1, tzinfo=UTC))[2]
    assert b - a == pytest.approx(2.0 / (31.0 * 86400.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. the regrid keeps the area mean
# ---------------------------------------------------------------------------

def _cell_centred(step):
    lat = np.arange(-90.0 + step / 2, 90.0, step)
    lon = np.arange(step / 2, 360.0, step)
    return lat, lon


@pytest.mark.parametrize("truncation", [3, 21])
def test_regrid_keeps_the_area_mean_of_an_analytic_monthly_field(truncation):
    """f = cos(lat) (1 + 0.1 m) from OISST's own 0.25 degree cell-centred
    ring onto T3 and T21, against the Gaussian-weighted mean of f evaluated
    on the target grid.  The regrid is bilinear, so the bound is bilinear's:
    in latitude f'' = -f, and linear interpolation over a step h errs by at
    most h^2/8 max|f''| = h^2/8 (1 + 0.1 m), which bounds the mean's error
    too (2.4e-6 at h = 0.25 degree; measured 1.44e-6 relative onto T3 and
    1.40e-6 onto T21, under half the bound in every month).
    The exact integral pi/4 (1 + 0.1 m) is the wrong reference: Gauss-Legendre
    quadrature of cos(lat) alone misses it by 2.0e-3 on the six rows of T3
    and by 1.4e-5 on the 33 rows of T21."""
    grid = GaussianGrid.create(truncation, dealias_factor=1.5)
    lat, lon = _cell_centred(0.25)
    shape = (lat.size, lon.size)
    base = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], shape)
    target = np.broadcast_to(np.cos(np.deg2rad(grid.latitude_deg))[:, None],
                             (grid.nlat, grid.nlon))
    water = np.ones((grid.nlat, grid.nlon), dtype=bool)
    source_water = np.ones(shape, dtype=bool)
    bound = np.deg2rad(0.25) ** 2 / 8.0
    for month in (0, 6, 11):
        scale = 1.0 + 0.1 * month
        out, record = sc.regrid_water_plane(base * scale, source_water, lat, lon, grid, water)
        assert record["searched_columns"] == 0
        error = abs(grid.global_mean(out) - grid.global_mean(target * scale))
        assert error <= bound * scale, (truncation, month, error, bound * scale)
        assert error > 0.0  # the comparison sees the regrid, not a copy


def test_regrid_reads_only_the_sources_water():
    """A water column next to the source's land takes the water value, never
    a mix with the land beside it; a column whose stencil is all land takes
    the water grown over that land."""
    grid = GaussianGrid.create(21, dealias_factor=1.5)
    lat, lon = _cell_centred(2.0)
    source_water = np.broadcast_to((lon < 180.0)[None, :], (lat.size, lon.size)).copy()
    field = np.where(source_water, 300.0, 400.0)
    water = np.ones((grid.nlat, grid.nlon), dtype=bool)
    out, record = sc.regrid_water_plane(field, source_water, lat, lon, grid, water)
    assert record["searched_columns"] > 0
    assert np.allclose(out, 300.0, rtol=0.0, atol=1.0e-9)
    land = np.zeros_like(water)
    land[:, :3] = True
    out, _ = sc.regrid_water_plane(field, source_water, lat, lon, grid, ~land)
    assert np.isnan(out[land]).all() and np.allclose(out[~land], 300.0, atol=1.0e-9)


# ---------------------------------------------------------------------------
# 3. and 4. pins, the engine's cache
# ---------------------------------------------------------------------------

def _table(tmp_path, files, *, engine_cache=False, row_id="test-row"):
    rows = [{"id": row_id, "default": True, "files": files}]
    if engine_cache:
        rows[0]["engine_cache"] = "ocean"
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"schema": sc.SOURCES_SCHEMA, "rows": rows}), encoding="utf-8")
    return path


def _spec(name, payload, **extra):
    return {"name": name, "plane": "sst_k", "months": 12, "units": "degC",
            "variable": "sst", "latitude": "lat", "longitude": "lon",
            "urls": ["https://example.invalid/" + name],
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), **extra}


def _no_decode(*_args, **_kwargs):
    raise AssertionError("a file was decoded before its pin was checked")


@pytest.mark.parametrize("damage", ["byte count", "sha256"])
def test_a_file_with_other_bytes_is_refused_before_it_is_decoded(tmp_path, monkeypatch, damage):
    payload = b"pinned climatology bytes" * 64
    table = _table(tmp_path, [_spec("sst.nc", payload)])
    staged = tmp_path / "cache" / "test-row" / "sst.nc"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(payload + b"x" if damage == "byte count"
                       else payload[:-1] + b"?")
    monkeypatch.setattr(sc, "_decode_file", _no_decode)
    grid = GaussianGrid.create(3, dealias_factor=1.5)
    with pytest.raises(sc.ClimatologyPinError, match="the pin is"):
        sc.build_climatology(grid, np.ones((grid.nlat, grid.nlon), bool),
                             cache=tmp_path / "cache", table=table, progress=lambda _: None)
    # fetch from a folder holding the same damaged bytes installs nothing
    source = tmp_path / "from"
    source.mkdir()
    (source / "sst.nc").write_bytes(staged.read_bytes())
    with pytest.raises(sc.ClimatologyPinError):
        sc.fetch(cache=tmp_path / "cache", source_dir=source, table=table,
                 progress=lambda _: None)
    assert not staged.exists() and not staged.with_name("sst.nc.part").exists()
    assert len(list(staged.parent.glob("sst.nc.invalid-*"))) == 1


def test_fetch_from_a_folder_installs_the_pinned_bytes_once(tmp_path):
    payload = b"pinned climatology bytes" * 64
    table = _table(tmp_path, [_spec("sst.nc", payload)])
    source = tmp_path / "from"
    source.mkdir()
    (source / "sst.nc").write_bytes(payload)
    first = sc.fetch(cache=tmp_path / "cache", source_dir=source, table=table,
                     progress=lambda _: None)
    assert [r["action"] for r in first] == ["installed"]
    assert (tmp_path / "cache" / "test-row" / "sst.nc").read_bytes() == payload
    again = sc.fetch(cache=tmp_path / "cache", table=table, progress=lambda _: None)
    assert [r["action"] for r in again] == ["already staged"]


def test_the_fetch_door_installs_and_refuses_with_its_exit_codes(tmp_path, capsys):
    from arwen_global import cli

    payload = b"pinned climatology bytes" * 64
    table = _table(tmp_path, [_spec("sst.nc", payload)])
    source = tmp_path / "from"
    source.mkdir()
    (source / "sst.nc").write_bytes(payload[::-1])
    argv = ["climatology", "fetch", "--table", str(table), "--cache", str(tmp_path / "c"),
            "--from", str(source)]
    assert cli.main(argv) == 1
    assert "the pin is" in capsys.readouterr().err
    (source / "sst.nc").write_bytes(payload)
    assert cli.main(argv) == 0
    assert "1 files of test-row staged and verified" in capsys.readouterr().out


def test_the_mixed_layer_row_is_found_in_the_engines_ocean_cache(tmp_path, monkeypatch):
    good, other = b"woa month bytes" * 100, b"woa month BYTES" * 100
    spec = _spec("woa18_A5B7_M0201_04.nc", good, plane="mixed_layer_depth_m",
                 months=1, month=1, units="m", variable="M_an")
    table = _table(tmp_path, [spec], engine_cache=True, row_id="woa-test")
    engine = tmp_path / "engine-ocean"
    engine.mkdir()
    monkeypatch.setenv(sc.ENGINE_OCEAN_CACHE_ENV, str(engine))

    def no_network(*_args, **_kwargs):
        raise AssertionError("the network was reached for a file the engine has")

    monkeypatch.setattr(sc.urllib.request, "urlopen", no_network)
    row = sc.resolve_rows(sc.load_sources(table))[0]
    (engine / spec["name"]).write_bytes(other)
    assert sc.locate(row, row["files"][0], cache=tmp_path / "cache") is None
    (engine / spec["name"]).write_bytes(good)
    found = sc.locate(row, row["files"][0], cache=tmp_path / "cache")
    assert found == engine / spec["name"]
    records = sc.fetch(cache=tmp_path / "cache", table=table, progress=lambda _: None)
    assert records[0]["action"] == "already in the engine's cache"
    assert records[0]["path"] == str(engine / spec["name"])
    assert not (tmp_path / "cache" / "woa-test").exists()


def test_the_packaged_mixed_layer_pins_are_the_engines():
    climatology = pytest.importorskip("gpuwm.ocean.mixed_layer.climatology")
    row = next(r for r in sc.load_sources()["rows"] if r["id"] == sc.MLD_ROW)
    assert len(row["files"]) == 12
    for spec in row["files"]:
        month = spec["month"]
        assert spec["name"] == climatology.filename(month)
        assert (spec["bytes"], spec["sha256"]) == climatology.MONTH_PINS[month]
        assert spec["urls"] == [t.format(month=month) for t in climatology.URL_TEMPLATES]


def test_every_packaged_row_is_pinned_and_needs_no_credential():
    table = sc.load_sources()
    assert [row["id"] for row in sc.resolve_rows(table)] == ["oisst-v21-mon-ltm-1991-2020"]
    for row in table["rows"]:
        assert row["licence"] and row["citation"]
        for spec in row["files"]:
            assert spec["bytes"] > 0 and len(spec["sha256"]) == 64
            assert all(url.startswith("https://") and "@" not in url for url in spec["urls"])
    rung_one = sc.download_bytes(sc.resolve_rows(table))
    assert rung_one == 45945542 + 31003804
    assert sc.climatology_bytes(384, 768, mixed_layer=True) == 36 * 384 * 768 * 4


# ---------------------------------------------------------------------------
# 5. a whole build from NetCDF through the Rust door
# ---------------------------------------------------------------------------

def _write_oisst_like(directory: Path):
    """Two small files shaped like the OISST LTMs: 4-degree cell-centred
    ring, twelve months, a land block over 0-40E 0-40N with the CF missing
    value, ice only poleward of 70 degrees."""
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = _cell_centred(4.0)
    land = (lat[:, None] > 0.0) & (lat[:, None] < 40.0) & (lon[None, :] < 40.0)
    missing = np.float32(-9.96921e36)
    paths = {}
    for name, var in (("sst.nc", "sst"), ("icec.nc", "icec")):
        path = directory / name
        with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("nbnds", 2)
            ds.createDimension("lat", lat.size)
            ds.createDimension("lon", lon.size)
            ds.createVariable("lat", "f4", ("lat",))[:] = lat
            ds.createVariable("lon", "f4", ("lon",))[:] = lon
            t = ds.createVariable("time", "f8", ("time",))
            t.units = "days since 1800-01-01 00:00:00"
            starts = [(datetime(1991, m, 1) - datetime(1800, 1, 1)).days for m in range(1, 13)]
            t[:] = starts
            bounds = ds.createVariable("climatology_bounds", "f8", ("time", "nbnds"))
            bounds.units = t.units
            bounds[:] = np.stack([starts, np.array(starts) + 10957.0], axis=1)
            v = ds.createVariable(var, "f4", ("time", "lat", "lon"), fill_value=False)
            v.missing_value = missing
            values = np.empty((12, lat.size, lon.size), dtype=np.float32)
            for month in range(12):
                if var == "sst":
                    plane = np.full((lat.size, lon.size), 10.0 + month, dtype=np.float32)
                    plane[land] = missing
                else:
                    plane = np.where(np.abs(lat)[:, None] > 70.0, 0.8, missing)
                    plane = np.broadcast_to(plane, (lat.size, lon.size)).astype(np.float32)
                values[month] = plane
            v[:] = values
        paths[name] = path
    return paths


def test_a_build_carries_every_input_sha_and_writes_the_same_bytes_twice(tmp_path):
    source = tmp_path / "from"
    source.mkdir()
    paths = _write_oisst_like(source)
    files = []
    for name, var, plane, units, extra in (
            ("sst.nc", "sst", "sst_k", "degC", {"water": "finite"}),
            ("icec.nc", "icec", "sea_ice_fraction", "fraction",
             {"water": "sst_k", "fill_on_water": 0.0})):
        payload = paths[name].read_bytes()
        files.append({**_spec(name, payload), "variable": var, "plane": plane,
                      "units": units, **extra})
    table = _table(tmp_path, files)
    cache = tmp_path / "cache"
    sc.fetch(cache=cache, source_dir=source, table=table, progress=lambda _: None)
    grid = GaussianGrid.create(21, dealias_factor=1.5)
    water = np.ones((grid.nlat, grid.nlon), dtype=bool)
    water[: grid.nlat // 8, :] = False  # a southern continent on the model grid
    digests = []
    for attempt in range(2):
        planes, provenance = sc.build_climatology(grid, water, cache=cache, table=table,
                                                  progress=lambda _: None)
        out = tmp_path / f"build-{attempt}" / "climatology.npz"
        digests.append(sc.write_climatology(out, planes, ocean=water,
                                            latitude_deg=grid.latitude_deg,
                                            longitude_deg=grid.longitude_deg,
                                            provenance=provenance))
    assert digests[0] == digests[1]
    assert provenance["inputs"] == [
        {"row": "test-row", "name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"]}
        for f in files]
    loaded = sc.load_climatology(tmp_path / "build-1" / "climatology.npz")
    assert loaded.sha256 == digests[0] and loaded.mixed_layer_depth_m is None
    assert np.array_equal(loaded.ocean, water) and not loaded.lake.any()
    # Known answers: SST 10 + m degC on every water column, NaN on land.
    for month in range(12):
        sst = loaded.sst_k[month]
        assert np.allclose(sst[water], 283.15 + month, rtol=0.0, atol=2.0e-5)
        assert np.isnan(sst[~water]).all()
    ice = loaded.sea_ice_fraction
    lat = grid.latitude_deg
    assert np.allclose(ice[:, lat > 76.0, :], 0.8, atol=1.0e-6)
    band = np.abs(lat) < 60.0
    assert np.all(ice[:, band, :][:, water[band]] == 0.0)
    row = provenance["check"][0]
    for where in ("own", "intersection"):
        for side in ("source", "model"):
            assert row["sst_k"][where][side] == pytest.approx(283.15, abs=1e-4)
    # The ice cap: 0.8 on the source rows poleward of 70 degrees (72 N ..
    # 88 N), so the source extent is the cap 2 pi R^2 (1 - sin 70); on the
    # model grid the bilinear ramp from 0 at 68 N to 0.8 at 72 N crosses
    # 0.15 at 68.75 N, so the model extent is every Gaussian row at or
    # poleward of it, whole.
    from arwen_global.spectral.constants import EARTH_RADIUS_M

    own = row["sea_ice_fraction"]["own"]
    radius_km = EARTH_RADIUS_M / 1000.0
    cap = 2.0 * np.pi * radius_km ** 2 * (1.0 - np.sin(np.deg2rad(70.0))) / 1.0e6
    assert own["source"]["nh_extent_015"] == pytest.approx(cap, abs=1.0e-4)
    rows = sc._model_areas_km2(grid)[:, 0] * grid.nlon
    assert own["model"]["nh_extent_015"] == pytest.approx(
        rows[lat >= 68.75].sum() / 1.0e6, abs=1.0e-4)
    assert sc.check_lines(provenance)[1].startswith("month  1; SST own 283.150/283.150")
    loaded.check_grid(grid.latitude_deg, grid.longitude_deg)
    with pytest.raises(ValueError, match="not this run's"):
        other = GaussianGrid.create(3, dealias_factor=1.5)
        loaded.check_grid(other.latitude_deg, other.longitude_deg)


def test_the_build_door_refuses_a_synthetic_planet():
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root

    cfg = load_config(str(config_root() / "arwen_global_moist_smoke.toml"))
    assert cfg.statics.source == "synthetic"
    with pytest.raises(ValueError, match="real statics' water mask"):
        sc.build_for_config(cfg, progress=lambda _: None)
