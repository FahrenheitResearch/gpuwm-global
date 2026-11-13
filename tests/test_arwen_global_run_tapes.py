"""``run --tapes``: the run writes the wrfout tape of every checkpoint it
writes, from its own state and beside the forecast.

THE BREAKAGE THESE GATES PREVENT.  A tape the run wrote is meant to be the
file ``export`` writes from that time's checkpoint, and the forecast is
meant not to notice it was asked.  A tape built from a state the export
never sees (another clock, an unrounded value, a field taken before the
step finished), or a synthesis that wrote into the state it read, would
put a map in front of a reader that no export reproduces, or move the
forecast itself.  These gates hold the run's tapes to the export's bytes,
the forecast's checkpoints to a run without tapes, a restart with tapes to
the straight run, and a failing tape to a printed line rather than a dead
forecast.  On the smoke configuration (CPU); the same comparisons on the
shipped T255 and T383 days on the card are measurements the CHANGELOG
reports, not gates.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_arwen_global_analysis_initial import CONFIG, _analysis_cfg  # noqa: E402
from test_arwen_global_forecast_clock import analysis_at  # noqa: E402,F401


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _same_checkpoint(left, right):
    a, b = _arrays(left), _arrays(right)
    assert sorted(a) == sorted(b)
    for name in a:
        assert a[name].dtype == b[name].dtype, name
        assert a[name].tobytes() == b[name].tobytes(), name


@pytest.fixture
def runs(tmp_path, analysis_at):
    pytest.importorskip("netCDF4")
    from arwen_global.runner import TapeRequest, run

    analysis_at(datetime(2026, 8, 30, 18))
    cfg = _analysis_cfg(tmp_path)
    plain = tmp_path / "plain"
    taped = tmp_path / "taped"
    run(cfg, plain)
    receipt = run(cfg, taped, tapes=TapeRequest(tmp_path / "tapes", nlat=8, nlon=16))
    return cfg, plain, taped, tmp_path / "tapes", receipt


def test_the_run_writes_the_tapes_export_writes(runs, tmp_path):
    from arwen_global.wrfout_export import EXPORT_RECEIPT_NAME, export_wrfout

    cfg, _plain, taped, tapes, receipt = runs
    checkpoints = sorted(taped.glob("arwen_global_step*.npz"))
    assert len(checkpoints) == 3
    exported = export_wrfout(cfg, checkpoints, tmp_path / "exported", nlat=8, nlon=16)
    written = sorted(tapes.glob("wrfout_*"))
    assert [p.name for p in written] == [p.name for p in exported]
    for mine, theirs in zip(written, exported):
        assert mine.read_bytes() == theirs.read_bytes(), mine.name
    # No temporary name is left behind for a folder watcher to pick up: the
    # run's folder holds what the export's holds (the writer's own
    # bookkeeping file included), nothing more.
    assert sorted(p.name for p in tapes.iterdir()) == sorted(
        p.name for p in (tmp_path / "exported").iterdir())
    assert EXPORT_RECEIPT_NAME in {p.name for p in tapes.iterdir()}
    block = receipt["tapes"]
    assert block["failures"] == []
    assert [Path(p).name for p in block["written"]] == [p.name for p in exported]
    rows = json.loads((tapes / EXPORT_RECEIPT_NAME).read_text(encoding="utf-8"))["tapes"]
    assert [Path(row["checkpoint"]).name for row in rows] == [p.name for p in checkpoints]
    assert [row["step"] for row in rows] == [0, 2, 4]


def test_the_forecast_does_not_see_its_tapes(runs):
    _cfg, plain, taped, _tapes, _receipt = runs
    names = sorted(p.name for p in plain.glob("arwen_global_step*.npz"))
    assert names == sorted(p.name for p in taped.glob("arwen_global_step*.npz"))
    for name in names:
        _same_checkpoint(plain / name, taped / name)


def test_a_restart_with_tapes_continues_the_straight_run(runs, tmp_path):
    from arwen_global.runner import TapeRequest, run

    cfg, _plain, taped, tapes, _receipt = runs
    first = tmp_path / "seg1"
    second = tmp_path / "seg2"
    seg_tapes = tmp_path / "seg_tapes"
    run(cfg, first, until_s=20.0, tapes=TapeRequest(seg_tapes, nlat=8, nlon=16))
    run(cfg, second, restart=first / "arwen_global_step00000002.npz",
        tapes=TapeRequest(seg_tapes, nlat=8, nlon=16))
    _same_checkpoint(second / "arwen_global_step00000004.npz",
                     taped / "arwen_global_step00000004.npz")
    straight = sorted(p for p in tapes.glob("wrfout_*"))
    segmented = sorted(p for p in seg_tapes.glob("wrfout_*"))
    assert [p.name for p in segmented] == [p.name for p in straight]
    for mine, theirs in zip(segmented, straight):
        assert mine.read_bytes() == theirs.read_bytes(), mine.name


def test_a_failing_tape_is_a_line_not_a_dead_forecast(tmp_path, analysis_at, monkeypatch, capsys):
    pytest.importorskip("netCDF4")
    from arwen_global import wrfout_export
    from arwen_global.runner import TapeRequest, run

    analysis_at(datetime(2026, 8, 30, 18))
    cfg = _analysis_cfg(tmp_path)

    def broken(self, *args, **kwargs):
        raise OSError("disk full (test)")

    monkeypatch.setattr(wrfout_export.TapeGrid, "write", broken)
    receipt = run(cfg, tmp_path / "run", tapes=TapeRequest(tmp_path / "tapes", nlat=8, nlon=16))
    # The run returned its receipt: the forecast went to its end.
    assert len(receipt["checkpoints"]) == 3
    failures = receipt["tapes"]["failures"]
    assert [row["step"] for row in failures] == [0, 2, 4]
    assert all(row["error_message"] == "disk full (test)" for row in failures)
    assert "tape failed step=2: OSError: disk full (test)" in capsys.readouterr().out
    assert not list((tmp_path / "tapes").glob("wrfout_*"))


def test_a_parked_or_multi_card_state_writes_no_tapes():
    from types import SimpleNamespace

    from arwen_global.runner import tape_refusal

    resident = SimpleNamespace(pipeline=SimpleNamespace(exchange=None))
    assert tape_refusal(None, resident) is None
    parked = SimpleNamespace(host_tier=object(), spill_slices=("tracers",),
                             pipeline=SimpleNamespace(exchange=None))
    assert "pinned host tier" in tape_refusal(None, parked)
    two = SimpleNamespace(pipeline=SimpleNamespace(exchange=SimpleNamespace(world=2)))
    assert "more than one card" in tape_refusal(None, two)


def test_the_cli_carries_the_tape_flags():
    from arwen_global.cli import _tape_request, build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "arwen_global_moist_smoke", "--tapes", "t", "--tape-nlat", "384",
                              "--tape-nlon", "768"])
    request = _tape_request(args)
    assert (request.outdir, request.nlat, request.nlon) == (Path("t"), 384, 768)
    assert _tape_request(parser.parse_args(["run", "arwen_global_moist_smoke"])) is None


def test_an_undated_run_writes_no_tapes_and_still_runs(tmp_path, capsys):
    """An idealized run has no instant to stamp a tape with: the run says so
    once, writes none, and integrates as it would have."""
    pytest.importorskip("netCDF4")
    from arwen_global.config import load_config
    from arwen_global.runner import TapeRequest, run

    cfg = load_config(CONFIG)
    receipt = run(cfg, tmp_path / "run", tapes=TapeRequest(tmp_path / "tapes", nlat=8, nlon=16))
    assert len(receipt["checkpoints"]) == 3
    assert receipt["tapes"]["refused"]
    assert "tapes off:" in capsys.readouterr().out
    assert not list((tmp_path / "tapes").glob("wrfout_*"))


def test_a_tape_that_falls_behind_keeps_its_own_checkpoints_arrays(tmp_path, analysis_at, monkeypatch):
    """A tape slower than the forecast holds the model at the next output
    time, before that checkpoint's copy: a checkpoint writer that copies
    every checkpoint into one host buffer must not overwrite the arrays a
    tape is still reading.  Here the writer hands the tape its own copy and
    zeroes it at the next checkpoint, as such a buffer would, and every
    tape is slowed past the forecast's two steps."""
    pytest.importorskip("netCDF4")
    import time

    from arwen_global import runner
    from arwen_global.runner import TapeRequest, run
    from arwen_global.wrfout_export import export_wrfout

    analysis_at(datetime(2026, 8, 30, 18))
    cfg = _analysis_cfg(tmp_path)
    rebuild = runner.state_from_checkpoint

    def slow(*args, **kwargs):
        time.sleep(1.0)
        return rebuild(*args, **kwargs)

    submit = runner._CheckpointWriter.submit
    handed = []

    def one_buffer(self, *args, **kwargs):
        self.join()
        for arrays in handed:
            for value in arrays.values():
                value[...] = 0
        out = submit(self, *args, **kwargs)
        self.last_arrays = {name: np.array(value, copy=True)
                            for name, value in self.last_arrays.items()}
        handed.append(self.last_arrays)
        return out

    with monkeypatch.context() as patch:
        patch.setattr(runner, "state_from_checkpoint", slow)
        patch.setattr(runner._CheckpointWriter, "submit", one_buffer)
        run(cfg, tmp_path / "run", tapes=TapeRequest(tmp_path / "tapes", nlat=8, nlon=16))
    checkpoints = sorted((tmp_path / "run").glob("arwen_global_step*.npz"))
    # Step 0 is written before the loop, outside the writer: two copies.
    assert len(handed) == len(checkpoints) - 1 == 2
    exported = export_wrfout(cfg, checkpoints, tmp_path / "exported", nlat=8, nlon=16)
    written = sorted((tmp_path / "tapes").glob("wrfout_*"))
    assert [p.name for p in written] == [p.name for p in exported]
    for mine, theirs in zip(written, exported):
        assert mine.read_bytes() == theirs.read_bytes(), mine.name
