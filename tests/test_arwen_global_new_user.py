"""What a new user met on public 0.3.0, held fixed one breakage at a time.

Each test names the defect it holds, as the fresh-user test of 0.3.0
reproduced it (a clean container, an empty home, a second account sharing
the install):

* a second account's fetch died with ``PermissionError`` on a lock folder
  the first account owned, and nothing named the remedy;
* ``doctor`` gave a second account ten door gaps and never named the door
  folder or its override;
* ``gpuwm-global --version`` exited 2;
* ``status.json`` sent a reader to a failure receipt that was never
  written;
* the scorer's model row was another product's name;
* every map said ``Δx 55.589 km``, the tape's spacing, whatever the model
  grid, and nothing on the map named the grid.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


# ------------------------------------------------------------ fetch locks

def test_the_lock_variable_is_the_engines_own():
    from gpuwm import fetch_guard

    from arwen_global import accounts

    assert accounts.LOCK_ROOT_ENV == fetch_guard.LOCK_ROOT_ENV


def test_the_console_script_locks_in_this_accounts_folder(tmp_path, monkeypatch):
    """Unset: this account's own ~/.gpuwm/locks, which the engine's lock_root
    then answers.  Set by the operator: theirs, untouched."""

    from gpuwm import fetch_guard

    from arwen_global import accounts

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv(accounts.LOCK_ROOT_ENV, raising=False)
    monkeypatch.setattr(accounts, "_BOUND_BY", None)
    root = accounts.bind_account_lock_root()
    assert root == tmp_path / ".gpuwm" / "locks"
    assert fetch_guard.lock_root() == root
    assert accounts.lock_root_binding() == (root, "this program")

    chosen = tmp_path / "shared-locks"
    monkeypatch.setenv(accounts.LOCK_ROOT_ENV, str(chosen))
    monkeypatch.setattr(accounts, "_BOUND_BY", None)
    assert accounts.bind_account_lock_root() == chosen
    assert os.environ[accounts.LOCK_ROOT_ENV] == str(chosen)
    assert accounts.lock_root_binding() == (chosen, "operator")


def test_main_binds_the_lock_folder_before_any_command(tmp_path):
    """The binding is in `main`, so `fetch-analysis`, `score` and every
    engine fetch made inside this process see it."""

    env = {key: value for key, value in os.environ.items()
           if key != "GPUWM_FETCH_LOCK_ROOT"}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    code = ("import os, sys\n"
            "from arwen_global import cli\n"
            "cli.main(['configs'])\n"
            "sys.stderr.write('LOCKS=' + os.environ['GPUWM_FETCH_LOCK_ROOT'])\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr
    assert f"LOCKS={tmp_path / '.gpuwm' / 'locks'}" in done.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can write anywhere")
def test_another_accounts_engine_lock_folder_is_named_with_its_remedy(
        tmp_path, monkeypatch):
    """The state that stopped `gpuwm fetch-geog` for the second account:
    the engine's shared folder exists and this account cannot write it."""

    import tempfile

    from arwen_global import accounts, doctor

    base = tmp_path / "tmp"
    locks = base / "gpuwm" / "locks"
    locks.mkdir(parents=True)
    locks.chmod(0o555)
    try:
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(base))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv(accounts.LOCK_ROOT_ENV, raising=False)
        monkeypatch.setattr(accounts, "_BOUND_BY", None)
        assert accounts.engine_lock_root_state() == ("blocked", locks)

        report = doctor.Report()
        doctor._account_section(report)
        rows = {row.label: row for row in report.sections[0][1]}
        engine = rows["engine fetch locks"]
        assert engine.verdict == "gap"
        text = " ".join(engine.detail)
        assert "gpuwm fetch-geog" in text and "gpuwm fetch" in text
        # measured 2026-11-22: fetch-bridges stages into the account's own
        # versioned folder and ran as the second account without the variable
        assert "fetch-bridges" not in text
        assert f"export GPUWM_FETCH_LOCK_ROOT={tmp_path / 'home' / '.gpuwm' / 'locks'}" in text
        assert rows["fetch locks"].verdict == "ok"
    finally:
        locks.chmod(0o755)


def test_doctor_names_this_accounts_folders_and_their_overrides(tmp_path, monkeypatch):
    from arwen_global import accounts, doctor

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("ARWEN_GLOBAL_DOOR_DIR", raising=False)
    monkeypatch.delenv(accounts.LOCK_ROOT_ENV, raising=False)
    monkeypatch.setattr(accounts, "_BOUND_BY", None)
    report = doctor.Report()
    doctor._account_section(report)
    title, rows = report.sections[0]
    assert title == "this account"
    by = {row.label: row for row in rows}
    assert by["door folder"].finding == str(tmp_path / ".gpuwm" / "global-doors")
    assert "ARWEN_GLOBAL_DOOR_DIR=DIR" in " ".join(by["door folder"].detail)
    assert by["fetch locks"].finding == str(tmp_path / ".gpuwm" / "locks")
    assert "gpuwm fetch-bridges" in " ".join(by["engine bridges"].detail)

    shared = tmp_path / "shared-doors"
    monkeypatch.setenv("ARWEN_GLOBAL_DOOR_DIR", str(shared))
    report = doctor.Report()
    doctor._account_section(report)
    by = {row.label: row for row in report.sections[0][1]}
    assert by["door folder"].finding == str(shared)
    assert "named by ARWEN_GLOBAL_DOOR_DIR" in by["door folder"].detail[0]


def test_a_missing_door_row_names_the_folder_it_looked_in(tmp_path, monkeypatch):
    from arwen_global import doctor
    from arwen_global.doors import COMPANION_BUNDLE, ENGINE_BUNDLE

    companion = tmp_path / "doors"
    assert str(companion) in doctor._looked_in(COMPANION_BUNDLE, companion)
    assert "ARWEN_GLOBAL_DOOR_DIR" in doctor._looked_in(COMPANION_BUNDLE, companion)
    assert "this account's own" in doctor._looked_in(ENGINE_BUNDLE, companion)


# ---------------------------------------------------------------- version

def test_version_prints_this_release_and_the_engine():
    from importlib.metadata import version

    done = subprocess.run([sys.executable, "-m", "arwen_global", "--version"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    line = done.stdout.strip()
    assert line.startswith(f"gpuwm-global {version('gpuwm-global')} "), line
    assert f"engine gpuwm {version('gpuwm')}" in line


# ----------------------------------------------------------- status.json

def test_a_run_that_wrote_no_receipt_does_not_point_at_one():
    from arwen_global.cli import _run_failure_sentence

    missing = FileNotFoundError("analysis GRIB x does not exist; fetch it with y")
    sentence = _run_failure_sentence(missing, receipt_written=False)
    assert "receipt is beside" not in sentence
    assert "stopped before it wrote a receipt" in sentence
    assert "fetch it with y" in sentence
    assert "failure receipt is beside" in _run_failure_sentence(
        RuntimeError("x"), receipt_written=True)


# ------------------------------------------------------------- the scorer

def test_the_scorer_row_is_this_models_name():
    from arwen_global.forecast_scorecard import default_label

    assert default_label(255) == "arwen-global-t255"
    assert default_label(1023) == "arwen-global-t1023"


def test_the_scorer_prints_each_door_call_as_it_runs(monkeypatch, tmp_path):
    import datetime as dt

    from arwen_global import forecast_scorecard as fs

    calls = []

    def fake(name, arguments):
        calls.append((name, arguments[0]))
        return {"networks": ["IA_ASOS", "NE_ASOS"]} if arguments[0] == "networks" else {}

    import arwen_global.obs_streams as streams
    monkeypatch.setattr(streams, "_run_door", fake)
    said = []
    valid = [dt.datetime(2026, 9, 1, 3), dt.datetime(2026, 9, 1, 6)]
    fs.asos_record(tmp_path, valid, progress=said.append)
    assert [c[1] for c in calls] == ["networks", "stations", "fetch", "decode"]
    assert any("rw_asos stations ..." in line for line in said)
    assert any("rw_asos stations done in" in line for line in said)
    assert any("2 networks" in line for line in said)


# ------------------------------------------------------------ map labels

def _tape(path: Path, attrs: dict, nx: int = 720) -> Path:
    netCDF4 = pytest.importorskip("netCDF4")
    lon = -180.0 + (360.0 / nx) * (np.arange(nx) + 0.5)
    lat = np.linspace(-89.75, 89.75, 4)
    xlong, xlat = np.meshgrid(lon, lat)
    with netCDF4.Dataset(path, "w") as tape:
        tape.setncatts(attrs)
        tape.createDimension("Time", 1)
        tape.createDimension("south_north", lat.size)
        tape.createDimension("west_east", lon.size)
        for name, value in (("XLAT", xlat), ("XLONG", xlong)):
            var = tape.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            var[0] = value
    return path


def test_the_map_source_line_names_the_model_grid(tmp_path):
    from arwen_global import __version__
    from arwen_global.render_door import map_source_label
    from arwen_global.wrfout_export import GRID_SPACING_ATTR, TRUNCATION_ATTR

    t255 = _tape(tmp_path / "a.nc", {TRUNCATION_ATTR: np.int32(255),
                                     GRID_SPACING_ATTR: np.float32(52.12)})
    t63 = _tape(tmp_path / "b.nc", {TRUNCATION_ATTR: np.int32(63),
                                    GRID_SPACING_ATTR: np.float32(208.5)})
    old = _tape(tmp_path / "c.nc", {})
    plain = f"ArWen Global {__version__}"
    assert map_source_label([t255]) == f"{plain}, T255 52.1 km grid, drawn at 0.5°"
    assert map_source_label([t63]).startswith(f"{plain}, T63 208.5 km grid")
    assert map_source_label([old]) == plain
    assert map_source_label([t255, t63]) == plain
    assert map_source_label() == plain


def test_the_tape_states_the_model_grid_it_came_from():
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.wrfout_export import model_grid_spacing_km

    nlat, nlon = GaussianGrid.shape_for(255, dealias_factor=1.5)
    grid = type("G", (), {"nlon": nlon, "radius_m": 6371220.0})()
    assert nlon == 768
    assert round(model_grid_spacing_km(grid), 1) == 52.1


# ------------------------------------------------------------------ NOTICE

def test_notice_carries_the_copernicus_attribution():
    text = (Path(__file__).resolve().parents[1] / "NOTICE").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "contain modified Copernicus Climate Change Service information" in flat
    assert "Neither the European Commission nor ECMWF is responsible" in flat


# ---------------------------------------------------------------- statics

def test_statics_on_a_cache_that_is_already_built_is_a_no_op(tmp_path, monkeypatch, capsys):
    """0.3.0 exited 1 when the cache existed, so a reader who ran the
    README's `go` and then its stage-by-stage route failed at stage two."""

    from arwen_global import cli, statics
    import arwen_global.physics.orographic_drag as drag

    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"built")
    cache.with_suffix(".json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(statics, "cache_paths",
                        lambda options, grid: (cache, cache.with_suffix(".json")))
    monkeypatch.setattr(statics, "cache_problem", lambda path, grid: None)
    monkeypatch.setattr(drag, "prepare_orographic_statistics", lambda *a, **k: None)
    args = cli.build_parser().parse_args(
        ["statics", "arwen_global_gdas_t63_native_sl_si_24h"])
    assert cli._statics(args) == 0
    assert "nothing to do" in capsys.readouterr().out

    monkeypatch.setattr(statics, "cache_problem",
                        lambda path, grid: "statics sidecar self-hash mismatch")
    with pytest.raises(FileExistsError, match="does not verify.*--overwrite"):
        cli._statics(args)


# ------------------------------------------------------------ README links

def test_the_readme_links_resolve_off_github_too():
    """PyPI renders the README and does not resolve its relative links or
    images; every link and image names its full address (anchors within the
    page excepted)."""
    import re

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    relative = [t for t in targets if not t.startswith(("https://", "#"))]
    assert relative == []
