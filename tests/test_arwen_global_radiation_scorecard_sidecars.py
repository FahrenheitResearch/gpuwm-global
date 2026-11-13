"""The radiation scorecard reads a long run's folder without its sidecars.

A long run writes ``<checkpoint>.longrun.npz`` beside each checkpoint; the
checkpoint glob matched it and the reader refused the whole run on it
("checkpoint ... has no metadata"), seen on the albedo ten-day run.
"""
from __future__ import annotations

from arwen_global import radiation_scorecard as rs
from arwen_global.longrun_ops import SIDECAR_SUFFIX


def test_the_checkpoint_list_leaves_out_the_long_run_sidecars(tmp_path):
    names = [
        "arwen_global_step00000144.npz",
        "arwen_global_step00000072.npz",
        "arwen_global_step00000072" + SIDECAR_SUFFIX,
        "arwen_global_step00000144" + SIDECAR_SUFFIX,
        "arwen_global_daily_2026-11-11.npz",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    got = [path.name for path in rs.run_checkpoints(tmp_path)]
    assert got == ["arwen_global_step00000072.npz", "arwen_global_step00000144.npz"]
