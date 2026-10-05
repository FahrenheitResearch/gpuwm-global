"""A forecast stopped inside the model still delivers the hours it reached.

GP-2 (audit 2026-10-05): a column whose surface reservoir ran dry stopped a
five-day T255 forecast at hour 117.3 with "native physics water closure
exceeds the explicit surface reservoir", and `go` delivered no picture,
although every checkpoint up to the stop was on disk.  The stop is still a
failure; the pictures of the hours before it are drawn first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from arwen_global import cli, go_door, render_door
from arwen_global.configs_dir import config_root

pytestmark = pytest.mark.cpu_only

_STOP = "native physics water closure exceeds the explicit surface reservoir"


def _args(tmp_path: Path, **extra) -> argparse.Namespace:
    values = dict(
        config=config_root() / "arwen_global_t255_quickstart.toml",
        outdir=tmp_path / "run", start_date="2026-09-01_00:00:00",
        products=None, geog_root=None, no_statics=True, no_render=False,
        overwrite=True,
    )
    values.update(extra)
    return argparse.Namespace(**values)


def _stopping_run(steps):
    def fake_run(leg):
        out = Path(leg.outdir)
        out.mkdir(parents=True, exist_ok=True)
        for step in steps:
            (out / f"arwen_global_step{step:06d}.npz").write_bytes(b"x")
        raise FloatingPointError(_STOP)
    return fake_run


def test_a_stopped_forecast_draws_the_hours_it_reached(tmp_path, monkeypatch):
    drawn = []
    monkeypatch.setattr(cli, "_run", _stopping_run((0, 12, 24)))
    monkeypatch.setattr(render_door, "render",
                        lambda leg: drawn.append(list(leg.inputs)) or 0)
    code = go_door.go(_args(tmp_path))
    assert code == go_door.EXIT_PARTIAL
    assert [p.name for p in drawn[0]] == [
        "arwen_global_step000000.npz", "arwen_global_step000012.npz",
        "arwen_global_step000024.npz"]
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["state"] == "failed"
    assert "partial" in status["reason"] and _STOP in status["reason"]


def test_a_stop_with_nothing_to_draw_fails_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run", _stopping_run((0,)))
    monkeypatch.setattr(render_door, "render",
                        lambda leg: pytest.fail("nothing past the start to draw"))
    with pytest.raises(FloatingPointError, match="surface reservoir"):
        go_door.go(_args(tmp_path))


def test_no_render_keeps_the_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run", _stopping_run((0, 12)))
    monkeypatch.setattr(render_door, "render",
                        lambda leg: pytest.fail("--no-render draws nothing"))
    with pytest.raises(FloatingPointError):
        go_door.go(_args(tmp_path, no_render=True))


class _FakeRunner:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.stage = "forecast"
        self.warnings = []
        self.committed = 0

    def warn(self, scope, message):
        self.warnings.append((scope, message))

    def commit_new_checkpoints(self):
        self.committed += 1
        return []


def test_the_plan_route_draws_the_hours_a_stopped_forecast_reached(
        tmp_path, monkeypatch):
    from arwen_global import runplan

    for step in (0, 12):
        (tmp_path / f"arwen_global_step{step:06d}.npz").write_bytes(b"x")
    drawn = []

    def fake_render_stage(plan, config_path, *, runner):
        runner.stage = "render"
        drawn.append(sorted(p.name for p in runner.run_dir.glob("*.npz")))

    monkeypatch.setattr(runplan, "_render_stage", fake_render_stage)
    runner = _FakeRunner(tmp_path)
    runplan._render_reached(None, None, FloatingPointError(_STOP),
                            runner=runner)
    assert drawn == [["arwen_global_step000000.npz",
                      "arwen_global_step000012.npz"]]
    assert runner.committed == 1
    assert any(_STOP in message for _scope, message in runner.warnings)
    # The failure the caller reports next is the forecast's own.
    assert runner.stage == "forecast"
