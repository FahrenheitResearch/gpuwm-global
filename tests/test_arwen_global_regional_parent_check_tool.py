"""The instrument that says a regional run is fed by its parent measures zero
on a regional state that IS the parent, and finds a planted departure where
it was planted.

THE BREAKAGE THIS PREVENTS.  `tools/arwen_global_regional_parent_check.py` is
what the regional parent page's measured run rests on: bias and RMS of a
regional history file against a parent frame, by distance from the lateral
edge.  An instrument with a level off by one, a wind rotated the wrong way or
a zone mask shifted by a row still prints a table of plausible small numbers,
and "the boundary row tracks the parent" would then be a statement about the
instrument.  So the instrument is run on a case with a known answer: a
regional file sampled analytically from the same functions the parent frame
holds, on a different grid and different levels.

No device and no model: numpy and netCDF4.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")

from arwen_global import mapped_parent_export as mpe  # noqa: E402

TREE = Path(__file__).resolve().parents[1]
NY, NX = 124, 130
REGIONAL_PRESSURE = (90000.0, 70000.0, 50000.0)
PARENT_PRESSURE = (95000.0, 80000.0, 60000.0, 40000.0, 20000.0)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_regional_parent_check",
        TREE / "tools" / "arwen_global_regional_parent_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _temperature(pressure, lat, lon):
    return 250.0 + 20.0 * (np.log(pressure) - np.log(5.0e4)) + 0.3 * (lat - 35.0) + 0.2 * (lon + 84.0)


def _vapour(pressure, lat, lon):
    return 1.0e-3 * (1.0 + 0.1 * (lat - 35.0)) + 0.0 * pressure + 0.0 * lon


def _u(pressure, lat, lon):
    return 5.0 + 0.5 * (lon + 84.0) + 0.2 * (lat - 35.0) + 3.0 * (np.log(pressure) - np.log(5.0e4))


def _v(pressure, lat, lon):
    return -3.0 + 0.4 * (lat - 35.0) + 0.1 * (lon + 84.0) + 0.0 * pressure


def _parent_frame(path, tool, valid=datetime(2026, 11, 12)):
    lat = np.arange(30.0, 40.01, 0.5)
    lon = np.arange(-90.0, -77.99, 0.5)
    nlev, ny, nx = len(PARENT_PRESSURE), lat.size, lon.size
    lat3, lon3 = lat[None, :, None], lon[None, None, :]
    pressure = np.asarray(PARENT_PRESSURE)[:, None, None] * np.ones((nlev, ny, nx))
    temperature = _temperature(pressure, lat3, lon3)
    theta = temperature * (tool.REFERENCE_PRESSURE_PA / pressure) ** tool.KAPPA_PARENT
    plane = np.ones((ny, nx))
    fields = {
        "T": theta - 300.0, "PB": pressure,
        "QVAPOR": _vapour(pressure, lat3, lon3),
        "U_EARTH": _u(pressure, lat3, lon3), "V_EARTH": _v(pressure, lat3, lon3),
        "PHB": np.linspace(100.0, 12000.0, nlev + 1)[:, None, None]
               * tool.GRAVITY_M_S2 * np.ones((nlev + 1, ny, nx)),
        "W": np.zeros((nlev + 1, ny, nx)),
        "PSFC": 100000.0 * plane, "HGT": 100.0 * plane, "TSK": 285.0 * plane,
        "T2": 284.0 * plane, "Q2": 5.0e-3 * plane, "U10": plane, "V10": plane,
        "LANDMASK": plane, "SEAICE": 0.0 * plane, "SNOW": 0.0 * plane,
        "SNOWH": 0.0 * plane, "TSLB": 283.0 * np.ones((4, ny, nx)),
        "SMOIS": 0.3 * np.ones((4, ny, nx)),
    }
    for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        fields[name] = np.zeros((nlev, ny, nx))
    return mpe.write_parent_frame(path, valid, lat, lon, fields)


def _regional_file(path, tool, *, rotation_deg=0.0, warm_interior_k=0.0,
                   west_wind_m_s=0.0, valid="2026-11-12_00:00:00"):
    """A WRF-shaped history file whose state is the analytic parent state.

    The grid is regular in latitude and longitude, but the file's winds are
    written rotated by ``rotation_deg`` with the matching COSALPHA/SINALPHA,
    so reading them back to earth relative is part of what is tested."""
    from netCDF4 import Dataset

    dlat, dlon = 8.0 / (NY - 1), 10.0 / (NX - 1)
    lat1 = 31.0 + dlat * np.arange(NY)
    lon1 = -89.0 + dlon * np.arange(NX)
    lat2, lon2 = np.meshgrid(lat1, lon1, indexing="ij")
    nz = len(REGIONAL_PRESSURE)
    pressure = np.asarray(REGIONAL_PRESSURE)[:, None, None] * np.ones((nz, NY, NX))
    temperature = _temperature(pressure, lat2[None], lon2[None])
    distance = tool.edge_distance(NY, NX)
    temperature = temperature + warm_interior_k * (distance >= 5)[None]
    theta = temperature * (tool.REFERENCE_PRESSURE_PA / pressure) ** tool.KAPPA_REGIONAL
    alpha = np.deg2rad(rotation_deg)
    cosa, sina = np.cos(alpha), np.sin(alpha)

    def grid_wind(lat, lon, level_pressure):
        u_earth = _u(level_pressure, lat, lon)
        v_earth = _v(level_pressure, lat, lon)
        # the inverse of u_earth = u cos - v sin, v_earth = v cos + u sin
        return u_earth * cosa + v_earth * sina, v_earth * cosa - u_earth * sina

    lon_u = -89.0 + dlon * (np.arange(NX + 1) - 0.5)
    lat_v = 31.0 + dlat * (np.arange(NY + 1) - 0.5)
    u_stag = np.empty((nz, NY, NX + 1))
    v_stag = np.empty((nz, NY + 1, NX))
    for level, level_pressure in enumerate(REGIONAL_PRESSURE):
        lat_a, lon_a = np.meshgrid(lat1, lon_u, indexing="ij")
        u_stag[level] = grid_wind(lat_a, lon_a, level_pressure)[0]
        lat_b, lon_b = np.meshgrid(lat_v, lon1, indexing="ij")
        v_stag[level] = grid_wind(lat_b, lon_b, level_pressure)[1]
    u_stag[:, :, :20] += west_wind_m_s

    with Dataset(path, "w", format="NETCDF4_CLASSIC") as data:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", NX),
                           ("south_north", NY), ("bottom_top", nz),
                           ("west_east_stag", NX + 1), ("south_north_stag", NY + 1)):
            data.createDimension(name, size)
        times = data.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list(valid), dtype="S1")
        mass = ("Time", "bottom_top", "south_north", "west_east")
        plane = ("Time", "south_north", "west_east")
        for name, dims, value in (
            ("XLAT", plane, lat2), ("XLONG", plane, lon2),
            ("P", mass, np.zeros_like(pressure)), ("PB", mass, pressure),
            ("T", mass, theta - 300.0),
            ("QVAPOR", mass, _vapour(pressure, lat2[None], lon2[None])),
            ("U", ("Time", "bottom_top", "south_north", "west_east_stag"), u_stag),
            ("V", ("Time", "bottom_top", "south_north_stag", "west_east"), v_stag),
            ("COSALPHA", plane, np.full((NY, NX), cosa)),
            ("SINALPHA", plane, np.full((NY, NX), sina)),
            ("PSFC", plane, np.full((NY, NX), 100000.0)),
            ("HGT", plane, np.full((NY, NX), 100.0)),
            ("T2", plane, np.full((NY, NX), 284.0)),
        ):
            data.createVariable(name, "f8", dims)[0] = value
    return path


def test_a_regional_state_that_is_the_parent_measures_zero(tool, tmp_path):
    """Different grid, different levels, winds rotated 25 degrees in the
    file: every zone comes back at rounding, so the horizontal and vertical
    interpolation, the destaggering and the rotation are each right."""
    parent = _parent_frame(tmp_path / mpe.frame_filename(datetime(2026, 11, 12)), tool)
    regional = _regional_file(tmp_path / "wrfout_same", tool, rotation_deg=25.0)
    result = tool.track(str(regional), str(parent))
    assert result["regional_time"] == "2026-11-12_00:00:00"
    assert result["parent_time"] == "2026-11-12_00:00:00"
    assert result["excluded_below_parent_fraction"] == 0.0
    assert list(result["zones"]) == [
        "spec d=0", "relax d=1-4", "near d=5-19", "inner d>=60", "all interior d>=5"]
    assert result["zones"]["spec d=0"]["columns"] == 2 * NX + 2 * NY - 4
    assert result["zones"]["inner d>=60"]["columns"] == (NY - 120) * (NX - 120)
    for zone, row in result["zones"].items():
        # The parent frame is float32; a few 1e-4 is its rounding in theta.
        assert row["T K"]["rms"] < 2.0e-3, zone
        assert row["u m/s"]["rms"] < 1.0e-4, zone
        assert row["v m/s"]["rms"] < 1.0e-4, zone
        assert row["qv g/kg"]["rms"] < 1.0e-5, zone
        assert row["psfc hPa"]["rms"] < 1.0e-9, zone


def test_a_departure_is_found_in_the_zone_it_was_planted_in(tool, tmp_path):
    """One kelvin added from the fifth row inward: the specified and
    relaxation rows still read zero and the interior reads one."""
    parent = _parent_frame(tmp_path / mpe.frame_filename(datetime(2026, 11, 12)), tool)
    regional = _regional_file(tmp_path / "wrfout_warm", tool, warm_interior_k=1.0)
    zones = tool.track(str(regional), str(parent))["zones"]
    for zone in ("spec d=0", "relax d=1-4"):
        assert zones[zone]["T K"]["rms"] < 2.0e-3, zone
    for zone in ("near d=5-19", "inner d>=60", "all interior d>=5"):
        assert zones[zone]["T K"]["bias"] == pytest.approx(1.0, abs=2.0e-3), zone
        assert zones[zone]["T K"]["rms"] == pytest.approx(1.0, abs=2.0e-3), zone
        assert zones[zone]["u m/s"]["rms"] < 1.0e-4, zone


def test_a_regional_level_below_the_parents_lowest_is_left_out(tool, tmp_path):
    """Regional terrain lower than the parent's puts regional levels under
    the parent's lowest full level; they are counted, not extrapolated."""
    from netCDF4 import Dataset

    parent = _parent_frame(tmp_path / mpe.frame_filename(datetime(2026, 11, 12)), tool)
    regional = _regional_file(tmp_path / "wrfout_low", tool)
    with Dataset(regional, "a") as data:
        pressure = np.asarray(data["PB"][0])
        pressure[0] = 97000.0          # below the parent's 95000 Pa
        data["PB"][0] = pressure
    result = tool.track(str(regional), str(parent))
    assert result["excluded_below_parent_fraction"] == pytest.approx(1.0 / 3.0)


def test_a_regional_grid_outside_the_parent_window_is_refused(tool, tmp_path):
    from netCDF4 import Dataset

    parent = _parent_frame(tmp_path / mpe.frame_filename(datetime(2026, 11, 12)), tool)
    regional = _regional_file(tmp_path / "wrfout_out", tool)
    with Dataset(regional, "a") as data:
        data["XLONG"][0] = np.asarray(data["XLONG"][0]) + 5.0
    with pytest.raises(ValueError, match="leaves the parent frame's window"):
        tool.track(str(regional), str(parent))


def test_two_runs_differ_where_one_was_changed_and_nowhere_else(tool, tmp_path):
    """The fed-against-control comparison: identical files read zero in
    every zone, and a wind added along the west side shows in the west strip
    and not in the east one."""
    first = _regional_file(tmp_path / "wrfout_a", tool)
    same = _regional_file(tmp_path / "wrfout_b", tool)
    changed = _regional_file(tmp_path / "wrfout_c", tool, west_wind_m_s=4.0)
    zero = tool.run_difference(str(first), str(same))["zones"]
    for zone, row in zero.items():
        assert row["T K"] == 0.0 and row["u m/s"] == 0.0 and row["psfc hPa"] == 0.0, zone
    moved = tool.run_difference(str(changed), str(first))["zones"]
    assert moved["east strip"]["u m/s"] == 0.0
    assert moved["inner d>=60"]["u m/s"] == 0.0
    # 20 staggered faces carry +4, so 19 full mass points and one half.
    assert moved["west strip i<20"]["u m/s"] == pytest.approx(
        4.0 * np.sqrt((19 + 0.25) / 20), rel=1.0e-9)
    assert moved["west strip i<20"]["T K"] == 0.0
    # The mean wind into the domain on each side, 850 to 500 hPa, of the first file.
    assert moved["west strip i<20"]["inward wind m/s"] > 0.0
    assert moved["east strip"]["inward wind m/s"] < 0.0


def test_two_files_of_different_valid_times_are_not_compared(tool, tmp_path):
    first = _regional_file(tmp_path / "wrfout_a", tool)
    later = _regional_file(tmp_path / "wrfout_b", tool, valid="2026-11-12_03:00:00")
    with pytest.raises(ValueError, match="not of one valid time"):
        tool.run_difference(str(first), str(later))


def test_the_command_prints_the_table_and_writes_the_numbers(tool, tmp_path, capsys):
    import json

    parent = _parent_frame(tmp_path / mpe.frame_filename(datetime(2026, 11, 12)), tool)
    regional = _regional_file(tmp_path / "wrfout_same", tool)
    out = tmp_path / "track.json"
    assert tool.main(["track", str(regional), str(parent), "--label", "initial time",
                      "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("## initial time: regional 2026-11-12_00:00:00 vs parent")
    assert "| spec d=0 | " in printed and "| inner d>=60 | " in printed
    assert json.loads(out.read_text(encoding="utf-8"))["label"] == "initial time"
    assert tool.main(["diff", str(regional), str(regional)]) == 0
    assert "| west strip i<20 | " in capsys.readouterr().out
