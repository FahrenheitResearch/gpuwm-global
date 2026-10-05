"""``gpuwm-global score``: the one command that scores a WOOF Global run
against ASOS stations and radiosondes through the Rust observation doors,
with the GFS and IFS analyses as secondary references.

No network, no card, no run: the doors are replaced by a recorder, the
radiosonde table is written in the door's own neutral format, and the
scores are read back against planted differences.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from arwen_global import forecast_scorecard as fs
from arwen_global import obs_scorecard as oc
from arwen_global.obs_table import TABLE_HEADER
from arwen_global.upper_air_scorecard import synthetic_grid

UTC = dt.timezone.utc


def _table(path: Path, rows: list[dict]) -> Path:
    """A ``gpuwm-obs.table.v2`` file in the shape ``rw_igra2 table`` writes."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(TABLE_HEADER)
        for r in rows:
            writer.writerow([
                "igra2", r["sid"], f"{r['lat']:.5f}", f"{r['lon']:.5f}", repr(float(r["z"])),
                repr(float(r["p"])), r["nominal"], r["var"], repr(float(r["value"])), "1",
                "sonde_level", r["nominal"], "", "", "",
            ])
    return path


def _sounding_rows(sid, lat, lon, nominal, level, z, t, u, v):
    base = {"sid": sid, "lat": lat, "lon": lon, "nominal": nominal, "p": level, "z": z}
    return [dict(base, var="temperature_k", value=t), dict(base, var="wind_u_m_s", value=u),
            dict(base, var="wind_v_m_s", value=v)]


def test_score_is_a_door_on_the_console_script():
    from arwen_global.cli import build_parser

    args = build_parser().parse_args(["score", "run-dir", "--out", "card", "--hours", "12", "24",
                                      "--reference", "ifs=a.grib2", "--families", "gfs"])
    assert args.command == "score"
    assert args.run_dir == Path("run-dir") and args.hours == [12.0, 24.0]
    assert args.reference == [("ifs", Path("a.grib2"))]


def test_a_reference_label_outside_the_table_is_refused_by_name():
    from arwen_global.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["score", "run-dir", "--out", "card", "--reference", "era5=x.grib"])


def test_the_reference_families_are_table_rows_that_name_their_fetch_and_file():
    cycle = dt.datetime(2026, 10, 1, 12, tzinfo=UTC)
    gfs, ifs = fs.REFERENCE_FAMILIES["gfs"], fs.REFERENCE_FAMILIES["ifs"]
    assert fs.reference_fetch_command(gfs, cycle, Path("d")) == [
        "fetch", "--source", "gdas", "--cycle", "2026-10-01T12", "--hours", "0", "--mode", "full-file",
        "--all-levels", "--out", "d"]
    assert fs.reference_fetch_command(ifs, cycle, Path("d")) == [
        "fetch", "--source", "ecmwf-open-data", "--cycle", "2026-10-01T12", "--hours", "0", "--mode",
        "full-file", "--out", "d"]
    assert fs.reference_path(gfs, cycle, Path("d")).name == "gdas.t12z.pgrb2.0p25.f000"
    assert fs.reference_path(ifs, cycle, Path("d")).name == "20261001120000-0h-oper-fc.grib2"


def test_the_station_record_comes_from_the_rw_asos_door_at_the_scored_instants(tmp_path, monkeypatch):
    calls = []

    def door(name, arguments):
        calls.append((name, list(arguments)))
        if arguments[0] == "networks":
            return {"networks": ["GB__ASOS", "IA_ASOS"]}
        return {"status": "READY"}

    monkeypatch.setattr(fs, "_door", door)
    valid = [dt.datetime(2026, 10, 1, h, tzinfo=UTC) for h in (6, 12, 18)] + [dt.datetime(2026, 10, 2, 0, tzinfo=UTC)]
    record = fs.asos_record(tmp_path, valid)
    assert record == tmp_path / "asos" / "record.json"
    assert [c[1][0] for c in calls] == ["networks", "stations", "fetch", "decode"]
    assert all(name == "rw_asos" for name, _ in calls)
    fetch = calls[2][1]
    assert fetch[fetch.index("--networks") + 1] == "GB__ASOS,IA_ASOS"
    assert fetch[fetch.index("--start") + 1] == "2026-10-01T05:00:00Z"
    assert fetch[fetch.index("--end") + 1] == "2026-10-02T01:00:00Z"
    decode = calls[3][1]
    assert decode[decode.index("--step-minutes") + 1] == "360"
    assert decode[decode.index("--start") + 1] == "2026-10-01T06:00:00Z"
    assert decode[decode.index("--end") + 1] == "2026-10-02T00:00:00Z"


def test_the_sounding_table_comes_from_the_rw_igra2_door_mandatory_levels(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fs, "_door", lambda name, arguments: calls.append((name, list(arguments))) or {})
    valid = [dt.datetime(2026, 10, 1, 12, tzinfo=UTC), dt.datetime(2026, 10, 2, 0, tzinfo=UTC)]
    table = fs.igra2_table(tmp_path, valid)
    assert table == tmp_path / "igra2" / "igra2.csv"
    assert [(n, a[0]) for n, a in calls] == [("rw_igra2", "fetch"), ("rw_igra2", "table")]
    assert "--mandatory-only" in calls[1][1]


def test_a_sounding_height_that_is_the_door_isa_stamp_is_refused_and_the_rest_kept(tmp_path):
    nominal = "2026-10-01T12:00:00Z"
    stamp = round(fs.isa_height_m(50_000.0), 1)
    assert stamp == 5574.4
    rows = (
        _sounding_rows("MEASURED", 36.7, 3.2, nominal, 50_000.0, 5890.0, 262.45, 0.64, 0.77)
        + _sounding_rows("MEASURED", 36.7, 3.2, nominal, 85_000.0, 1530.0, 290.0, -3.0, 4.0)
        + _sounding_rows("STAMPED", -20.0, 150.0, nominal, 50_000.0, stamp, 265.0, 10.0, 0.0)
        + _sounding_rows("OTHERHOUR", 10.0, 10.0, "2026-10-01T00:00:00Z", 50_000.0, 5800.0, 260.0, 1.0, 1.0)
    )
    table = _table(tmp_path / "igra2.csv", rows)
    records, counters = fs.soundings_from_table(table, dt.datetime(2026, 10, 1, 12, tzinfo=UTC))
    by_id = {r["station_id"]: r for r in records}
    assert sorted(by_id) == ["MEASURED", "STAMPED"]
    assert by_id["MEASURED"]["levels"]["50000"]["z"] == 5890.0
    assert by_id["MEASURED"]["levels"]["50000"]["t"] == 262.45
    assert by_id["MEASURED"]["levels"]["85000"]["wspd"] == pytest.approx(5.0, abs=1e-12)
    assert by_id["STAMPED"]["levels"]["50000"]["z"] is None
    assert by_id["STAMPED"]["levels"]["50000"]["t"] == 265.0
    assert counters["heights_refused_as_isa_stamp"] == 1


def test_planted_sounding_differences_read_back_through_the_door_table(tmp_path):
    grid = synthetic_grid(21)
    lf = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg)
    planted = {("z", 50_000.0): 12.0, ("t", 85_000.0): -1.5, ("t", 50_000.0): 0.8,
               ("u", 25_000.0): 3.0, ("v", 25_000.0): -4.0}
    nominal = "2026-10-01T12:00:00Z"
    rows = []
    for record in oc.synthetic_soundings(lf, 120, offsets=planted):
        for key, values in record["levels"].items():
            rows += _sounding_rows(record["station_id"], record["latitude"], record["longitude"], nominal,
                                   float(key), values["z"], values["t"], values["u"], values["v"])
    table = _table(tmp_path / "igra2.csv", rows)
    soundings, counters = fs.soundings_from_table(table, dt.datetime(2026, 10, 1, 12, tzinfo=UTC))
    assert counters["soundings"] == 120 and counters["heights_refused_as_isa_stamp"] == 0
    scores = oc.score_soundings({"woof": lf}, soundings)["scores"]["woof"]
    # The table carries five significant digits of position, so the read
    # differs from the plant by the field's change over 5e-6 degrees.
    assert scores["z500"]["global"]["bias"] == pytest.approx(-12.0, abs=1e-3)
    assert scores["t850"]["global"]["bias"] == pytest.approx(1.5, abs=1e-4)
    assert scores["t500"]["global"]["bias"] == pytest.approx(-0.8, abs=1e-4)
    assert scores["w250"]["global"]["rmsve"] == pytest.approx(5.0, abs=1e-4)
    assert scores["w850"]["global"]["rmsve"] == pytest.approx(0.0, abs=1e-4)


def test_a_lead_the_run_did_not_write_is_refused_naming_the_leads_it_did():
    start = dt.datetime(2026, 10, 1, tzinfo=UTC)
    rows = [{"lead_h": h, "valid": start + dt.timedelta(hours=h)} for h in (0.0, 6.0, 12.0)]
    assert [r["lead_h"] for r in fs.select_checkpoints(rows, None)] == [6.0, 12.0]
    assert [r["lead_h"] for r in fs.select_checkpoints(rows, [12])] == [12.0]
    with pytest.raises(ValueError, match="no checkpoint at lead 24 h; the run wrote 0, 6, 12 h"):
        fs.select_checkpoints(rows, [24])


def test_the_isa_height_inverts_the_table_isa_pressure():
    from arwen_global.obs_table import isa_pressure_pa

    for pressure in (100_000.0, 85_000.0, 50_000.0, 25_000.0, 10_000.0, 1_000.0):
        assert isa_pressure_pa(fs.isa_height_m(pressure)) == pytest.approx(pressure, rel=1e-12)


def test_every_reference_family_mapping_resolves_on_this_install():
    pytest.importorskip("gpuwm")
    from arwen_global.analysis_initial import resolve_analysis_mapping

    for family in fs.REFERENCE_FAMILIES.values():
        assert resolve_analysis_mapping(family.mapping).is_file(), family.label


def _analysis_frame(drop=()):
    lat = np.linspace(90.0, -90.0, 181)
    lon = np.arange(0.0, 360.0, 1.0)
    sf = oc.synthetic_surface_fields(lat, lon)
    lf = oc.synthetic_level_fields(lat, lon)
    levels = sorted(lf.levels_pa)
    fields = {
        "air_temperature_2m": sf.t2_k, "specific_humidity_2m": sf.q2_kg_kg,
        "eastward_wind_10m": sf.u10_m_s, "northward_wind_10m": sf.v10_m_s,
        "surface_pressure": sf.surface_pressure_pa, "terrain_height": sf.terrain_m,
        "land_fraction": np.ones_like(sf.t2_k),
        "geopotential_height": np.stack([lf.fields["z"][p] for p in levels]),
        "air_temperature": np.stack([lf.fields["t"][p] for p in levels]),
        "eastward_wind": np.stack([lf.fields["u"][p] for p in levels]),
        "northward_wind": np.stack([lf.fields["v"][p] for p in levels]),
    }
    for name in drop:
        fields.pop(name)
    when = dt.datetime(2026, 10, 1, 12)
    frame = oc._FakeFrame(lat, lon, fields, vertical_values=levels, valid_time=when, source_cycle=when)
    return frame, sf, lf


def test_each_analysis_is_scored_alone_against_the_same_reports():
    pytest.importorskip("gpuwm")
    frame, sf, lf = _analysis_frame()
    seam = "2026-10-01T12:00:00"
    stations = oc.synthetic_stations(sf, 150)
    observations = oc.synthetic_observations(sf, stations, seam, {"temperature_2m": 1.5, "dewpoint_2m": -2.0})
    soundings = oc.synthetic_soundings(lf, 80, offsets={("t", 85_000.0): -1.0, ("z", 50_000.0): 12.0})
    fit = fs.reference_fit("ifs", frame, "ifs.grib2", observations, seam, soundings)
    surface = fit["surface_fit"]["scores"]["ifs"]
    assert surface["temperature_2m"]["n"] == 150
    assert surface["temperature_2m"]["bias"] == pytest.approx(-1.5, abs=1e-9)
    assert surface["dewpoint_2m"]["bias"] == pytest.approx(2.0, abs=1e-9)
    upper = fit["upper_air_fit"]["scores"]["ifs"]
    assert upper["t850"]["global"]["bias"] == pytest.approx(1.0, abs=1e-9)
    assert upper["z500"]["global"]["bias"] == pytest.approx(-12.0, abs=1e-9)


def test_an_analysis_lacking_a_surface_field_is_refused_by_name_and_still_read_aloft():
    pytest.importorskip("gpuwm")
    frame, sf, lf = _analysis_frame(drop=("specific_humidity_2m",))
    seam = "2026-10-01T12:00:00"
    observations = oc.synthetic_observations(sf, oc.synthetic_stations(sf, 20), seam)
    fit = fs.reference_fit("gfs", frame, "gfs.grib2", observations, seam, oc.synthetic_soundings(lf, 10))
    assert "surface_fit" not in fit
    assert "specific_humidity_2m" in fit["surface_fit_refused"]
    assert fit["upper_air_fit"]["scores"]["gfs"]["t850"]["global"]["n"] == 10


def test_a_sounding_every_analysis_rejects_is_dropped_and_one_analysis_alone_cannot_drop_it():
    grid = synthetic_grid(21)
    a = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg)
    b = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg, offsets={("z", 50_000.0): 150.0})
    soundings = oc.synthetic_soundings(a, 6)
    bad = soundings[0]["levels"]["50000"]
    bad["z"] += 160.0          # 160 m from a, 10 m from b: b keeps it
    worse = soundings[1]["levels"]["85000"]
    worse["t"] += 7.0          # beyond 6 K from both
    windy = soundings[2]["levels"]["25000"]
    windy["u"] += 30.0         # beyond 25 m/s vector from both
    kept, record = fs.screen_soundings(soundings, {"a": a, "b": b})
    assert record["applied"] is True
    dropped = {(d["station_id"], d["level_pa"], d["field"]) for d in record["dropped"]}
    assert dropped == {(soundings[1]["station_id"], 85_000.0, "t"), (soundings[2]["station_id"], 25_000.0, "wind")}
    assert kept[0]["levels"]["50000"]["z"] == bad["z"]           # b agrees with it: kept
    assert kept[1]["levels"]["85000"]["t"] is None and kept[1]["levels"]["85000"]["z"] is not None
    assert kept[2]["levels"]["25000"]["u"] is None and kept[2]["levels"]["25000"]["wspd"] is None
    assert soundings[1]["levels"]["85000"]["t"] is not None      # the caller's records are not touched
    alone, record_alone = fs.screen_soundings(soundings, {"a": a})
    assert alone[0]["levels"]["50000"]["z"] is None
    unscreened, none = fs.screen_soundings(soundings, {})
    assert none["applied"] is False and unscreened is soundings


def test_only_the_values_a_row_scores_are_screened():
    grid = synthetic_grid(21)
    a = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg)
    soundings = oc.synthetic_soundings(a, 2)
    soundings[0]["levels"]["25000"]["z"] += 200.0    # no row reads a 250 hPa height
    soundings[1]["levels"]["50000"]["z"] += 200.0    # the 500 hPa height row does
    kept, record = fs.screen_soundings(soundings, {"a": a})
    assert kept[0]["levels"]["25000"]["z"] == soundings[0]["levels"]["25000"]["z"]
    assert kept[1]["levels"]["50000"]["z"] is None
    assert [(d["level_pa"], d["field"]) for d in record["dropped"]] == [(50_000.0, "z")]


def test_an_analysis_that_cannot_be_fetched_is_a_named_missing_row_not_a_lost_scorecard(tmp_path):
    valid = [dt.datetime(2026, 10, 1, h, tzinfo=UTC) for h in (6, 12)] + [dt.datetime(2026, 10, 2, 0, tzinfo=UTC)]
    asked = []

    def fetch(family, cycle, root):
        asked.append((family.label, cycle.hour))
        if family.label == "ifs" and cycle.day == 2:
            raise RuntimeError("ifs analysis for 2026-10-02T00:00:00Z: `gpuwm fetch ...` exited 1: 404")
        return root / family.label / f"{cycle:%Y%m%d%H}"

    held = {"gfs": {"2026-10-01T12:00:00Z": Path("held.grib2")}}
    failures = fs.gather_references(held, ("gfs", "ifs"), valid, tmp_path, progress=lambda *_: None, fetch=fetch)
    assert asked == [("gfs", 6), ("gfs", 0), ("ifs", 12), ("ifs", 0)]
    assert held["gfs"]["2026-10-01T12:00:00Z"] == Path("held.grib2")
    assert sorted(held["ifs"]) == ["2026-10-01T12:00:00Z"]
    assert failures == [{"family": "ifs", "valid_time": "2026-10-02T00:00:00Z",
                         "reason": "ifs analysis for 2026-10-02T00:00:00Z: `gpuwm fetch ...` exited 1: 404"}]
    lines = fs.scorecard_lines({"label": "w", "truncation": 255, "start_time_utc": "s", "leads": [],
                                "reference_failures": failures})
    assert lines[-1].startswith("No ifs analysis at 2026-10-02T00:00:00Z: ")


def test_a_lead_screened_by_fewer_analyses_than_its_rule_expects_says_so():
    """The screen can only use the analyses that arrived, so the model's
    sounding rows depend on fetch success; the lead's record and the text
    scorecard name it instead of leaving it silent."""
    grid = synthetic_grid(21)
    a = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg)
    soundings = oc.synthetic_soundings(a, 3)
    valid = dt.datetime(2026, 10, 1, 12, tzinfo=UTC)
    expected = fs.expected_references(("gfs", "ifs"), valid)
    assert expected == ["gfs", "ifs"]
    assert fs.expected_references(("gfs", "ifs"), valid.replace(hour=6)) == ["gfs"]
    _kept, record = fs.screen_lead(soundings, {"gfs": a}, expected)
    assert record["degraded"] is True and record["missing_references"] == ["ifs"]
    _kept, whole = fs.screen_lead(soundings, {"gfs": a, "ifs": a}, expected)
    assert whole["degraded"] is False
    row = {"n": 1, "bias": 0.0, "rmse": 0.0}
    lead = {"lead_h": 12.0, "analyses": {}, "upper_air": None,
            "surface": {"scores": {"w": {v: dict(row) for v in (
                "temperature_2m", "dewpoint_2m", "wind_speed_10m", "mslp")}}},
            "screen": {k: record[k] for k in
                       ("expected_references", "missing_references", "degraded")}}
    lines = fs.scorecard_lines({"label": "w", "truncation": 255, "start_time_utc": "s",
                                "leads": [lead], "reference_failures": []})
    assert lines[-1].startswith("Screen degraded at lead 12 h")
    assert "lacked ifs" in lines[-1] and "gfs alone" in lines[-1]


def test_an_analysis_floor_row_is_not_read_on_soundings_it_screened_itself():
    grid = synthetic_grid(21)
    a = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg)
    b = oc.synthetic_level_fields(grid.latitude_deg, grid.longitude_deg, offsets={("z", 50_000.0): 150.0})
    soundings = oc.synthetic_soundings(a, 4)
    soundings[0]["levels"]["50000"]["z"] += 160.0   # beyond the bar from a, 10 m from b
    floor_a = fs.floor_soundings(soundings, {"a": a, "b": b}, "a")
    assert floor_a[0]["levels"]["50000"]["z"] == soundings[0]["levels"]["50000"]["z"]
    floor_b_alone = fs.floor_soundings(soundings, {"a": a}, "a")
    assert floor_b_alone is soundings


def test_the_reference_fetch_runs_the_engine_this_python_imports(monkeypatch, tmp_path):
    """A ``gpuwm`` on PATH can belong to another interpreter with no engine
    (a rented box's /usr/local/bin/gpuwm ran the system Python and every
    reference fetch died with ModuleNotFoundError).  With no script beside
    this interpreter, the fetch runs this interpreter's own engine."""
    import importlib.util
    import shutil
    import sys

    from arwen_global import forecast_scorecard as fs

    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/gpuwm")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "gpuwm" else real(name, *a))
    assert fs._gpuwm_command() == [str(tmp_path / "python"), "-m", "gpuwm"]


def test_a_hung_reference_fetch_is_a_named_missing_row(monkeypatch, tmp_path):
    """The fetch route has no timeout of its own; a hung download held the
    whole score.  A fetch past the bar is now a named failure."""
    import subprocess

    from arwen_global import forecast_scorecard as fs

    def hang(command, **kwargs):
        assert kwargs.get("timeout") == fs.REFERENCE_FETCH_TIMEOUT_S
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(fs.subprocess, "run", hang)
    monkeypatch.setattr(fs, "_gpuwm_command", lambda: ["gpuwm"])
    by_family = {}
    valid = fs._utc(dt.datetime(2026, 9, 1, 0, tzinfo=dt.timezone.utc))
    failures = fs.gather_references(by_family, ["gfs"], [valid], tmp_path, progress=lambda *_: None)
    assert len(failures) == 1 and "did not finish" in failures[0]["reason"]


def test_a_reference_that_does_not_decode_is_a_named_missing_row(tmp_path):
    """A truncated cached analysis aborted the scorecard; it is now a
    failure row and the file is moved aside so the next score refetches."""
    from arwen_global import forecast_scorecard as fs

    bad = tmp_path / "gdas.t00z.pgrb2.0p25.f000"
    bad.write_bytes(b"GRIB-cut-short")
    files = {"2026-09-01_00:00:00": bad}
    failures = []

    def broken(mapping, path):
        raise ValueError("truncated message")

    got = fs.decode_or_fail("gfs", None, files, "2026-09-01_00:00:00", failures,
                            progress=lambda *_: None, decode=broken)
    assert got is None and files == {}
    assert "did not decode" in failures[0]["reason"]
    assert not bad.exists() and (tmp_path / (bad.name + ".undecodable")).exists()
