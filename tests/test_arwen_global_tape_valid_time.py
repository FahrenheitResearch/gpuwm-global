"""The render tape's valid time is the run's forecast clock (finding GI-8).

THE BREAKAGE THESE GATES PREVENT.  ``export``, ``render`` and ``go`` took the
tapes' valid time from a free ``--start-date`` string with nothing to check
it against: a wrong value mis-stamped every wrfout's Times, START_DATE and
file name, and every map drawn from them, silently; ``go`` refused to render
at all without one; and the quickstart told a reader to type the same
instant a second time.  A dated run (an analysis start, or a config that
states start_time_utc) now stamps its tapes from its own clock, a typed
value that disagrees is refused by name, and only an idealized run with no
date needs the argument.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tomllib

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_arwen_global_analysis_initial import _analysis_cfg  # noqa: E402
from test_arwen_global_forecast_clock import analysis_at, dated_run  # noqa: E402,F401


def test_the_tapes_take_their_valid_time_from_the_run(dated_run, tmp_path):
    """No --start-date: the tapes are stamped from the analysis valid time.
    A --start-date that disagrees is refused by name instead of
    mis-stamping every wrfout and every map drawn from it."""
    pytest.importorskip("netCDF4")
    from arwen_global.clock import ClockMismatchError
    from arwen_global.wrfout_export import EXPORT_RECEIPT_NAME, export_wrfout

    cfg, outdir = dated_run
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    tapes = tmp_path / "tapes"
    written = export_wrfout(cfg, checkpoints, tapes, nlat=8, nlon=16)
    assert len(written) == len(checkpoints)
    assert "2026-08-30" in written[0].name and "18" in written[0].name
    receipt = json.loads((tapes / EXPORT_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["start_date"] == "2026-08-30_18:00:00"
    assert receipt["start_date_source"] == "analysis-valid-time"
    assert receipt["tapes"][1]["valid"] == "2026-08-30_18:00:20"

    # The same instant typed is accepted; another one is refused.
    export_wrfout(cfg, checkpoints[:1], tmp_path / "same", nlat=8, nlon=16,
                  start_date="2026-08-30_18:00:00")
    with pytest.raises(ClockMismatchError, match="--start-date 2026-08-30T12:00:00Z"):
        export_wrfout(cfg, checkpoints[:1], tmp_path / "wrong", nlat=8, nlon=16,
                      start_date="2026-08-30_12:00:00")


def test_a_checkpoint_from_another_start_is_not_exported(dated_run, analysis_at, tmp_path):
    """The same config pointed at a later cycle cannot stamp a checkpoint
    the earlier analysis started: every valid time would be off by the gap."""
    from arwen_global.clock import ClockMismatchError
    from arwen_global.wrfout_export import export_wrfout

    cfg, outdir = dated_run
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    analysis_at(datetime(2026, 8, 31, 0))
    with pytest.raises(ClockMismatchError, match="integrated from 2026-08-30T18:00:00Z"):
        export_wrfout(cfg, checkpoints[:1], tmp_path / "other", nlat=8, nlon=16)


def test_an_undated_run_needs_a_start_date_to_label_its_tapes(tmp_path):
    pytest.importorskip("netCDF4")
    from arwen_global.runner import run
    from arwen_global.wrfout_export import export_wrfout

    cfg = load_config(config_root() / "arwen_global_moist_smoke.toml")
    outdir = tmp_path / "run"
    run(cfg, outdir)
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))[:1]
    with pytest.raises(ValueError, match="idealized-fixture.*Pass --start-date"):
        export_wrfout(cfg, checkpoints, tmp_path / "tapes", nlat=8, nlon=16)
    written = export_wrfout(cfg, checkpoints, tmp_path / "tapes", nlat=8, nlon=16,
                            start_date="2026-08-30_18:00:00")
    assert written


def test_go_and_the_run_plan_render_a_dated_run_without_a_typed_date(tmp_path):
    from arwen_global.clock import undated_render_reason
    from arwen_global.runplan import _render_skip_reason

    dated = _analysis_cfg(tmp_path)
    undated = load_config(config_root() / "arwen_global_moist_smoke.toml")
    assert undated_render_reason(dated) is None
    assert "idealized fixture" in undated_render_reason(undated)
    assert _render_skip_reason({}, dated) is None
    assert "start_date" in _render_skip_reason({}, undated)
    assert _render_skip_reason({"start_date": "2026-08-30_18:00:00"}, undated) is None


def test_the_quickstart_route_types_no_date():
    """The documented route needs no date beyond the cycle it fetches."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "ARWEN_GLOBAL_QUICKSTART.md"
    text = doc.read_text(encoding="utf-8")
    route = text.split("## Rehearsing the chain without a card")[0]
    blocks = route.split("```bash")[1:]
    assert blocks
    for block in blocks:
        assert "--start-date" not in block.split("```")[0], block
    assert "Exactly two spellings travel together" in text
    quickstart = config_root() / "arwen_global_t255_quickstart.toml"
    raw = tomllib.loads(quickstart.read_text(encoding="utf-8"))
    assert raw["physics"]["mode"] == "reference"
    assert "--start-date" not in quickstart.read_text(encoding="utf-8")
