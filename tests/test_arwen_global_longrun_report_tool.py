"""tools/arwen_global_longrun_report.py where the answers are known: a
daily long-term-mean file written here with a closed-form seasonal cycle
reads back that cycle (29 February as 28 February), the regular-grid area
mean of cos(lat) is pi/4 to the cells' accuracy, and the command line
reads a run folder's rows and its receipt's start into the gate table.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from arwen_global import longrun_gates as lg  # noqa: E402

TREE = Path(__file__).resolve().parents[1]
LEVELS = (1000.0, 850.0, 500.0, 200.0, 50.0, 10.0)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_longrun_report", TREE / "tools" / "arwen_global_longrun_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cycle(level, day_index):
    return 250.0 - level / 100.0 + 10.0 * math.sin(2.0 * math.pi * day_index / 365.0)


@pytest.fixture(scope="module")
def ltm_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ltm") / "air.day.ltm.nc"
    lat = np.arange(90.0, -90.1, -2.5)
    lon = np.arange(0.0, 360.0, 2.5)
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", 365)
        data.createDimension("level", len(LEVELS))
        data.createDimension("lat", lat.size)
        data.createDimension("lon", lon.size)
        data.createVariable("level", "f4", ("level",))[:] = LEVELS
        data.createVariable("lat", "f4", ("lat",))[:] = lat
        data.createVariable("lon", "f4", ("lon",))[:] = lon
        air = data.createVariable("air", "f4", ("time", "level", "lat", "lon"))
        values = np.empty((365, len(LEVELS), lat.size, lon.size), dtype=np.float32)
        for d in range(365):
            for k, level in enumerate(LEVELS):
                values[d, k] = _cycle(level, d)
        # The northern cap 30 K colder than the rest at 10 hPa.
        values[:, LEVELS.index(10.0), lat >= 60.0, :] -= 30.0
        air[:] = values
    return path


def test_the_long_term_mean_reads_back_its_daily_cycle(tool, ltm_file):
    ltm = tool.DailyLongTermMean(ltm_file)
    for day in (0, 100, 364):
        noon = dt.datetime(2027, 1, 1, 12, tzinfo=dt.timezone.utc) + dt.timedelta(days=day)
        assert ltm.value(lg.level_row_name(500), 0.0, noon) == pytest.approx(
            _cycle(500.0, day), abs=2e-4)
    # Between two noons, linear; 29 February reads 28 February.
    feb28 = dt.datetime(2028, 2, 28, 12, tzinfo=dt.timezone.utc)
    feb29 = dt.datetime(2028, 2, 29, 12, tzinfo=dt.timezone.utc)
    assert ltm.global_mean(850, feb29) == ltm.global_mean(850, feb28)
    assert ltm.value(lg.cap_row_name(10, "south"), 0.0, feb28) == pytest.approx(
        _cycle(10.0, 58), abs=2e-4)
    assert ltm.value(lg.cap_row_name(10, "north"), 0.0, feb28) == pytest.approx(
        _cycle(10.0, 58) - 30.0, abs=2e-4)
    assert ltm.value("not_a_row", 0.0, feb28) is None


def test_the_regular_grid_area_mean_of_cos_lat_is_pi_over_4(tool):
    lat = np.arange(90.0, -90.1, -0.25)
    field = np.repeat(np.cos(np.deg2rad(lat))[:, None], 8, axis=1)
    mean, caps = tool.regular_grid_rows(field, lat)
    assert mean == pytest.approx(math.pi / 4.0, rel=1e-5)
    # The cap's exact mean of cos(lat) over 60-90 deg (cells centred from
    # 60 deg, so the cap starts half a cell below it).
    edge = math.radians(60.0 - 0.125)
    exact = (math.pi / 4.0 - (edge / 2.0 + math.sin(2.0 * edge) / 4.0)) / (1.0 - math.sin(edge))
    assert caps["north"] == pytest.approx(exact, rel=1e-4)
    assert caps["south"] == pytest.approx(caps["north"], rel=1e-12)


def test_the_command_line_reads_a_run_folder_and_its_start(tool, ltm_file, tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for i in range(41):
        t = i * 6 * 3600.0
        rows.append({"step": i, "time_s": t, "minimum_temperature_k": 180.0,
                     "global_mean_surface_pressure_pa": 98570.0,
                     "global_mean_total_water_kg_m2": 709.6 + 1e-6 * i,
                     lg.level_row_name(850): 280.0})
    (run / "diagnostics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run / tool.RECEIPT_NAME).write_text(json.dumps(
        {"forecast_clock": {"start_utc": "2026-10-15T00:00:00Z"}}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert tool.main([str(run), "--ltm", str(ltm_file), "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["start_utc"] == "2026-10-15T00:00:00Z"
    assert report["gates"]["LR-1"]["total_water_relative_drift"] == pytest.approx(
        40e-6 / 709.6, rel=1e-6)
    assert report["gates"]["LR-5"]["climatology"] == ltm_file.name
    assert report["gates"]["LR-5"]["verdict"] == "not measured"
    assert "LR-9" in capsys.readouterr().out
