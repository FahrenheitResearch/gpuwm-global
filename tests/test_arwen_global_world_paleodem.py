"""The Pangea map reader (worlds package 2.3), on small synthetic maps.

Every file here is written by the test in the archive's own shape (netCDF4,
``z`` on ``(lat, lon)``, gridline registered with the date line repeated,
NaN fill, a GMT ``history`` line), so the suite needs no network and no
download.  The real archive is read by the host check in the report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest

from arwen_global.world.sources import paleodem
from arwen_global.world.sources.paleodem import PaleoDEMError

netCDF4 = pytest.importorskip("netCDF4")

SRC = Path(__file__).resolve().parent.parent / "src" / "arwen_global"


def _write_map(path: Path, *, step: float = 10.0, field=None, names=("z",),
               units=None, history="grdsample Map49_PALEOMAP_Test_252Ma.nc -I1d",
               north_first=False, lon0=-180.0) -> Path:
    lat = np.arange(-90.0, 90.0 + step / 2, step)
    lon = np.arange(lon0, lon0 + 360.0 + step / 2, step)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    if field is None:
        z = np.where(lat2 > 0.0, 100.0, -100.0)
    else:
        z = field(lat2, lon2)
    if north_first:
        lat = lat[::-1]
        z = z[::-1, :]
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.history = history
        ds.createDimension("lon", lon.size)
        ds.createDimension("lat", lat.size)
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        for i, name in enumerate(names):
            var = ds.createVariable(name, "f4", ("lat", "lon"), fill_value=np.float32(np.nan))
            if units is not None:
                var.units = units
            var[:] = (z + i).astype(np.float32)
    return path


# -- reading ---------------------------------------------------------------
def test_the_archive_shape_reads_south_to_north_with_the_seam_dropped(tmp_path):
    path = _write_map(tmp_path / "Map49_PALEOMAP_1deg_Permo-Triassic Boundary_250Ma.nc")
    m = paleodem.read_map(path)
    assert m.shape == (19, 36)
    assert m.lat_deg[0] == -90.0 and m.lat_deg[-1] == 90.0
    assert m.lon_deg[0] == -180.0 and m.lon_deg[-1] == 170.0
    assert m.seam_repeated and m.seam_max_diff_m == 0.0
    assert m.variable == "z"
    assert "states no unit" in m.units_note
    assert m.age_ma == 250.0 and m.history_age_ma == 252.0
    assert m.period == "Permo-Triassic Boundary"
    assert m.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert m.elevation_range() == (-100.0, 100.0)


def test_land_share_is_the_closed_form_band_area(tmp_path):
    # Land on every row north of the equator; each point owns half a
    # spacing either side, so land covers sin(90) - sin(5) of the
    # northern band of area 1: a share of (1 - sin 5 deg) / 2.
    m = paleodem.read_map(_write_map(tmp_path / "a.nc"))
    expected = (1.0 - math.sin(math.radians(5.0))) / 2.0
    assert m.land_share() == pytest.approx(expected, abs=1e-12)
    assert float(m.area_weights().sum()) == pytest.approx(1.0, abs=1e-12)
    assert m.land_share(sea_level_m=200.0) == 0.0


def test_row_order_and_longitude_convention_do_not_change_the_map(tmp_path):
    field = lambda la, lo: 7.0 * la + np.cos(np.radians(lo)) * 50.0 + np.sin(np.radians(lo)) * 20.0
    base = paleodem.read_map(_write_map(tmp_path / "a.nc", field=field))
    flipped = paleodem.read_map(_write_map(tmp_path / "b.nc", field=field, north_first=True))
    east = paleodem.read_map(_write_map(tmp_path / "c.nc", field=field, lon0=0.0))
    for other in (flipped, east):
        np.testing.assert_array_equal(other.lat_deg, base.lat_deg)
        np.testing.assert_array_equal(other.lon_deg, base.lon_deg)
        np.testing.assert_allclose(other.elevation_m, base.elevation_m, rtol=0, atol=1e-3)


def test_the_elevation_variable_is_found_by_name_or_named(tmp_path):
    m = paleodem.read_map(_write_map(tmp_path / "a.nc", names=("mask", "elevation")))
    assert m.variable == "elevation"
    with pytest.raises(PaleoDEMError, match="cannot tell which variable.*a, b"):
        paleodem.read_map(_write_map(tmp_path / "b.nc", names=("a", "b")))
    picked = paleodem.read_map(tmp_path / "b.nc", variable="b")
    assert picked.variable == "b" and picked.elevation_range() == (-99.0, 101.0)
    with pytest.raises(PaleoDEMError, match="'c' is not a 2-D field"):
        paleodem.read_map(tmp_path / "b.nc", variable="c")


def test_units_are_metres_or_converted_or_refused(tmp_path):
    km = paleodem.read_map(_write_map(tmp_path / "a.nc", units="km"))
    assert km.elevation_range() == (-100000.0, 100000.0)
    assert paleodem.read_map(_write_map(tmp_path / "m.nc", units="m")).units_note == "metres (m)"
    with pytest.raises(PaleoDEMError, match="'feet'"):
        paleodem.read_map(_write_map(tmp_path / "b.nc", units="feet"))


def test_a_map_with_holes_is_refused(tmp_path):
    field = lambda la, lo: np.where((la == 0.0) & (lo == 0.0), np.nan, 1.0)
    with pytest.raises(PaleoDEMError, match="1 points hold no elevation"):
        paleodem.read_map(_write_map(tmp_path / "a.nc", field=field))


def test_sampling_is_bilinear_and_periodic(tmp_path):
    m = paleodem.read_map(_write_map(tmp_path / "a.nc", field=lambda la, lo: 3.0 * la + 2.0))
    lat = np.array([-87.3, -12.5, 0.0, 33.3, 89.9])
    np.testing.assert_allclose(m.sample(lat, 12.0), 3.0 * lat + 2.0, rtol=0, atol=1e-4)
    wave = paleodem.read_map(_write_map(tmp_path / "b.nc", field=lambda la, lo: np.cos(np.radians(lo)) * 100.0 + 0.0 * la))
    np.testing.assert_allclose(wave.sample(10.0, 190.0), wave.sample(10.0, -170.0), rtol=0, atol=1e-9)
    np.testing.assert_allclose(wave.sample(10.0, 365.0), wave.sample(10.0, 5.0), rtol=0, atol=1e-9)


def test_land_mask_agreement_against_an_earth_mask(tmp_path):
    m = paleodem.read_map(_write_map(tmp_path / "a.nc"))
    lat = np.array([-60.0, -20.0, 20.0, 60.0])
    lon = np.linspace(0.0, 350.0, 36)
    earth = np.repeat((lat > 0.0)[:, None], lon.size, axis=1).astype(np.float32)
    same = paleodem.land_mask_agreement(m, earth, lat, lon)
    assert same["agreement"] == pytest.approx(1.0)
    assert same["map_land_earth_water"] == 0.0 and same["map_water_earth_land"] == 0.0
    flipped = paleodem.land_mask_agreement(m, 1.0 - earth, lat, lon)
    assert flipped["agreement"] == pytest.approx(0.0)
    assert flipped["map_land_share"] == pytest.approx(0.5)


# -- choosing --------------------------------------------------------------
NAMES = (
    "Map01_PALEOMAP_1deg_Holocene_0Ma.nc",
    "Map43.5_PALEOMAP_1deg_Late_Triassic_205Ma.nc",
    "Map48_PALEOMAP_1deg_Middle_Triassic_245Ma.nc",
    "Map49_PALEOMAP_1deg_Permo-Triassic Boundary_250Ma.nc",
    "Map50_PALEOMAP_1deg_LatePermian_255Ma.nc",
)


def _folder_of_names(tmp_path: Path) -> Path:
    folder = tmp_path / "maps"
    (folder / "sub").mkdir(parents=True)
    for name in NAMES:
        _write_map(folder / "sub" / name)
    (folder / "sub" / "Map01_PALEOMAP_1deg_Holocene_0Ma.nc.band1.level0.gplates.cache").write_bytes(b"x")
    (folder / "notes.nc").write_bytes(b"")  # not a map name: ignored
    return folder


def test_maps_are_listed_by_age_and_the_nearest_is_chosen(tmp_path):
    entries = paleodem.list_maps(_folder_of_names(tmp_path))
    assert [e.age_ma for e in entries] == [0.0, 205.0, 245.0, 250.0, 255.0]
    assert entries[1].number == "43.5"
    assert paleodem.choose_map(entries, 250).name == NAMES[3]
    assert paleodem.choose_map(entries, 251).name == NAMES[3]
    assert paleodem.choose_map(entries, 252.5).name == NAMES[3]  # a tie goes younger
    assert paleodem.choose_map(entries, 3.0).name == NAMES[0]
    with pytest.raises(PaleoDEMError, match="oldest map here is 255 Ma"):
        paleodem.choose_map(entries, 261.0)
    with pytest.raises(PaleoDEMError, match="finite age"):
        paleodem.choose_map(entries, -1.0)


def test_open_nearest_says_which_map_it_chose(tmp_path):
    folder = _folder_of_names(tmp_path)
    said = []
    m = paleodem.open_nearest(folder, 248.0, log=said.append)
    assert m.name == NAMES[3]
    assert said == [f"chose {NAMES[3]} for 248 Ma: Map49 Permo-Triassic Boundary, "
                    "250 Ma, 2 Ma away (the nearest map)"]


def test_a_map_twice_and_an_empty_folder_are_refused(tmp_path):
    folder = _folder_of_names(tmp_path)
    _write_map(folder / NAMES[0])
    with pytest.raises(PaleoDEMError, match="twice"):
        paleodem.list_maps(folder)
    (tmp_path / "empty").mkdir()
    with pytest.raises(PaleoDEMError, match="no PALEOMAP map"):
        paleodem.list_maps(tmp_path / "empty")
    with pytest.raises(PaleoDEMError, match="no map folder"):
        paleodem.list_maps(tmp_path / "absent")


# -- fetching --------------------------------------------------------------
def _archive(tmp_path: Path, extra=()) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    zpath = tmp_path / "staging" / paleodem.ARCHIVE_NAME
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("Maps_v2/", "")
        for name in NAMES[:2]:
            zf.write(_write_map(src / name), f"Maps_v2/{name}")
        zf.writestr(f"Maps_v2/{NAMES[0]}.band1.level0.gplates.cache", b"cache")
        zf.writestr("Maps_v2/All_Maps.gpml", b"<gpml/>")
        zf.writestr("License.txt", b"CC BY 4.0")
        for name, data in extra:
            zf.writestr(name, data)
    return zpath


def _checks(zpath: Path) -> dict:
    data = zpath.read_bytes()
    return {"url": zpath.as_uri(), "md5": hashlib.md5(data).hexdigest(), "size": len(data)}


def test_fetch_unpacks_only_the_maps_into_its_own_empty_folder(tmp_path):
    zpath = _archive(tmp_path)
    folder = tmp_path / "pangea-maps"
    said = []
    paleodem.fetch(folder, log=said.append, **_checks(zpath))
    assert sorted(p.name for p in (folder / "maps").iterdir()) == sorted([*NAMES[:2], "License.txt"])
    record = json.loads((folder / paleodem.SOURCE_RECORD).read_text(encoding="utf-8"))
    assert record["record_doi"] == "10.5281/zenodo.5460860"
    assert record["licence"] == "CC-BY-4.0" and "Scotese" in record["credit"]
    assert set(record["maps"]) == set(NAMES[:2])
    for name, digest in record["maps"].items():
        assert digest == hashlib.sha256((folder / "maps" / name).read_bytes()).hexdigest()
    assert not list(folder.glob("*.part"))
    assert said[-1].endswith(f"2 maps in {folder / 'maps'}")
    assert [e.name for e in paleodem.list_maps(folder)] == [NAMES[0], NAMES[1]]
    with pytest.raises(PaleoDEMError, match="not empty"):
        paleodem.fetch(folder, log=said.append, **_checks(zpath))


def test_a_download_that_does_not_match_the_record_unpacks_nothing(tmp_path):
    zpath = _archive(tmp_path)
    checks = _checks(zpath)
    for bad in ({"md5": "0" * 32}, {"size": checks["size"] + 1}, {"size": checks["size"] - 1}):
        folder = tmp_path / f"f{len(list(tmp_path.iterdir()))}"
        with pytest.raises(PaleoDEMError, match="md5|bytes"):
            paleodem.fetch(folder, log=lambda _: None, **{**checks, **bad})
        assert list(folder.iterdir()) == []


def test_member_paths_cannot_leave_the_maps_folder(tmp_path):
    zpath = _archive(tmp_path, extra=[("../../evil_0Ma.nc", b"x"), ("/abs/Map99_PALEOMAP_1deg_X_1Ma.nc", b"y")])
    folder = tmp_path / "out" / "maps-here"
    paleodem.fetch(folder, log=lambda _: None, **_checks(zpath))
    assert (folder / "maps" / "evil_0Ma.nc").read_bytes() == b"x"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["maps-here"]
    assert not (tmp_path / "evil_0Ma.nc").exists()


# -- what travels on, and what does not ship -------------------------------
def test_the_credit_line_and_provenance_carry_no_host_path(tmp_path):
    m = paleodem.read_map(_write_map(tmp_path / NAMES[3]))
    assert "https://doi.org/10.5281/zenodo.5460860" in m.credit_line()
    assert "CC BY 4.0" in m.credit_line() and NAMES[3] in m.credit_line()
    record = m.provenance()
    assert record["map"] == NAMES[3] and record["sha256"] == m.sha256
    assert str(tmp_path) not in json.dumps(record)


def test_the_map_reaches_a_world_folder_with_its_provenance(tmp_path, monkeypatch):
    # The world folder is package 2.4's; this pins the call this reader makes
    # (its write_map signature), with a stand-in when that package is absent.
    import sys
    import types

    calls = []

    def write_map(folder, elevation_m, latitude_deg, longitude_deg, *, source=None, overwrite=False):
        calls.append((folder, elevation_m, latitude_deg, longitude_deg, source, overwrite))
        return Path(folder) / "world-map.npz"

    fake = types.ModuleType("arwen_global.world.folder")
    fake.write_map = write_map
    monkeypatch.setitem(sys.modules, "arwen_global.world.folder", fake)
    m = paleodem.read_map(_write_map(tmp_path / NAMES[3]))
    target = paleodem.write_world_map(m, tmp_path / "pangea-250ma.world")
    assert target == tmp_path / "pangea-250ma.world" / "world-map.npz"
    (folder, z, lat, lon, source, overwrite), = calls
    assert z is m.elevation_m and lat is m.lat_deg and lon is m.lon_deg
    assert source == m.provenance() and overwrite is False
    # what 2.4's map check asks: strictly monotonic axes, no repeated column
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0) and lon[-1] - lon[0] < 360.0


def test_no_map_ships_in_the_wheel():
    assert not [p for p in SRC.rglob("*") if p.suffix in (".nc", ".zip")]


def test_the_command_line_describes_the_chosen_map_and_refuses_by_name(tmp_path, capsys):
    folder = _folder_of_names(tmp_path)
    assert paleodem.main(["describe", str(folder), "--age", "250"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"chose {NAMES[3]} for 250 Ma")
    assert "variable:   z" in out and "land share: 0.4564" in out
    assert "its history line names 252 Ma" in out
    assert paleodem.main(["describe", str(folder), "--age", "400"]) == 1
    assert "refused: no map near 400 Ma" in capsys.readouterr().err
