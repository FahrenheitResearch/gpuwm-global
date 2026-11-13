"""The in-situ ledger's cadences in seconds of forecast time, and the
stratospheric noise reading (experimental, off by default).

The ledger's cadences (spectra, flush and energy every N steps) count
STEPS, so a run at a longer step samples less of its forecast.  Each may be
given in seconds instead; unset, the step counts and the ledger stand as
they were.  The stratospheric reading keeps the run's largest top-decile
share of kinetic energy between 10 and 100 hPa from the spectra the ledger
already samples; it reports in the receipt's in-situ block and never moves
the run or its status.  Neither enters the config hash.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.insitu.options import (
    InsituOptions,
    SECONDS_CADENCES,
    insitu_options_from_table,
)
from arwen_global.runner import run

SMOKE = _shipped_configs() / "arwen_global_moist_smoke.toml"


def _write(tmp_path: Path, text: str, name: str = "cfg.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _smoke_text(dt_s: float = 10.0, duration_s: float = 40.0,
                output_interval_s: float = 20.0, **tables) -> str:
    """The moist smoke on the semi-Lagrangian core at the given step and
    times, with extra TOML lines appended to the named tables."""
    text = SMOKE.read_text(encoding="utf-8").replace(
        'integrator = "ssprk3"', 'integrator = "sl_si"'
    ).replace("\nweight = 0.5\n", "\nweight = 1.0\n").replace(
        "dt_s = 10.0\nduration_s = 40.0\noutput_interval_s = 20.0\n",
        f"dt_s = {dt_s!r}\nduration_s = {duration_s!r}\n"
        f"output_interval_s = {output_interval_s!r}\n",
    )
    time_and_tables = tables
    for table, lines in time_and_tables.items():
        if f"[{table}]" in text:
            text = text.replace(f"[{table}]\n", f"[{table}]\n{lines}\n", 1)
        else:
            text = text.rstrip("\n") + f"\n\n[{table}]\n{lines}\n"
    return text


# -- the in-situ cadences in seconds -----------------------------------------


def test_a_cadence_in_seconds_is_counted_in_steps_and_refused_between_steps():
    options = InsituOptions(spectra_every_s=3000.0, flush_every_s=1800.0)
    assert options.cadence_steps("spectra_every", 300.0) == 10
    assert options.cadence_steps("spectra_every", 600.0) == 5
    assert options.cadence_steps("flush_every", 450.0) == 4
    assert options.cadence_steps("energy_every", 450.0) == 50  # still in steps
    resolved = options.resolved_for_step(600.0)
    assert (resolved.spectra_every, resolved.flush_every, resolved.energy_every) == (5, 3, 50)
    with pytest.raises(ValueError, match="not a whole number of 450 s steps"):
        options.cadence_steps("spectra_every", 450.0)
    with pytest.raises(ValueError, match="positive number of seconds"):
        InsituOptions(spectra_every_s=0.0)
    with pytest.raises(ValueError, match="positive number of seconds"):
        InsituOptions(energy_every_s=True)
    unset = InsituOptions()
    assert unset.resolved_for_step(450.0) is unset
    assert set(SECONDS_CADENCES) == {"flush_every", "spectra_every", "energy_every"}
    # The ledger header and the receipt's in-situ block keep their keys.
    assert not any(key.endswith("_s") for key in unset.identity)
    assert options.identity["spectra_every_s"] == 3000.0
    with pytest.raises(ValueError, match="both in steps"):
        insitu_options_from_table({"spectra_every": 10, "spectra_every_s": 3000.0})


def test_a_cadence_between_steps_is_refused_at_load_and_stays_out_of_the_hash(tmp_path):
    base = load_config(_write(tmp_path, _smoke_text(), "base.toml"))
    seconds = load_config(_write(
        tmp_path, _smoke_text(insitu="spectra_every_s = 40.0"), "seconds.toml"))
    assert seconds.config_hash == base.config_hash
    assert seconds.insitu.spectra_every_s == 40.0
    with pytest.raises(ValueError, match="not a whole number of 10 s steps"):
        load_config(_write(tmp_path, _smoke_text(insitu="spectra_every_s = 25.0"), "bad.toml"))


def _ledger(outdir: Path) -> list[dict]:
    with (outdir / "insitu.ndjson").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _stable(rows: list[dict]) -> list[dict]:
    """The ledger's rows without the header (it records the options as
    given) and without wall-clock fields."""
    kept = []
    for row in rows:
        if row.get("kind") == "header":
            continue
        kept.append(json.loads(json.dumps(row, sort_keys=True), object_hook=lambda d: {
            k: v for k, v in d.items() if "wall" not in k
        }))
    return kept


def test_seconds_at_the_step_they_name_write_the_ledger_the_step_counts_write(tmp_path):
    """At the step the counts were set at, the cadences in seconds sample,
    flush and trip at exactly the steps the counts do: every ledger row but
    the header (which records the options as given) is the same."""
    cfg = load_config(_write(tmp_path, _smoke_text(), "base.toml"))
    cfg = replace(cfg, duration_s=120.0, output_interval_s=120.0)
    counts = replace(cfg, insitu=InsituOptions(spectra_every=3, flush_every=4, energy_every=5))
    seconds = replace(cfg, insitu=InsituOptions(
        spectra_every_s=3 * cfg.dt_s, flush_every_s=4 * cfg.dt_s, energy_every_s=5 * cfg.dt_s))
    a = run(counts, tmp_path / "counts")
    b = run(seconds, tmp_path / "seconds")
    rows_a, rows_b = _ledger(tmp_path / "counts"), _ledger(tmp_path / "seconds")
    assert _stable(rows_a) == _stable(rows_b)
    assert [r["step"] for r in rows_a if r.get("kind") == "spectra"] == [3, 6, 9, 12]
    assert a["insitu"]["trip_count"] == b["insitu"]["trip_count"]
    assert a["insitu"]["spectra_rows"] == b["insitu"]["spectra_rows"] == 4
    header = rows_b[0]
    assert header["options"]["spectra_every_s"] == 3 * cfg.dt_s
    assert header["options"]["spectra_every"] == 3
    assert "spectra_every_s" not in rows_a[0]["options"]


def test_one_cadence_in_seconds_samples_the_same_forecast_times_at_two_steps(tmp_path):
    """A 450 / 600 s pair scaled to the smoke's 15 / 20 s:
    a cadence of 60 s is 4 steps at one and 3 at the other, and the spectra
    land at the same forecast times."""
    times = {}
    for dt in (15.0, 20.0):
        cfg = load_config(_write(
            tmp_path, _smoke_text(dt, 240.0, 240.0,
                                  insitu="spectra_every_s = 60.0\nflush_every_s = 60.0"),
            f"dt{dt:g}.toml"))
        run(cfg, tmp_path / f"dt{dt:g}")
        rows = _ledger(tmp_path / f"dt{dt:g}")
        times[dt] = [r["time_s"] for r in rows if r.get("kind") == "spectra"]
    assert times[15.0] == times[20.0] == [60.0, 120.0, 180.0, 240.0]


def _arrays_equal(a: Path, b: Path) -> None:
    meta_a, arrays_a = read_checkpoint(a)
    meta_b, arrays_b = read_checkpoint(b)
    assert meta_a["config_hash"] == meta_b["config_hash"]
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    assert sorted(arrays_a) == sorted(arrays_b)
    for name in arrays_a:
        assert arrays_a[name].dtype == arrays_b[name].dtype, name
        assert np.array_equal(arrays_a[name], arrays_b[name]), name




# -- the stratospheric noise reading (experimental, off by default) ----------


def _sample(step, time_s, top, total):
    return {
        "kind": "spectra", "step": step, "time_s": time_s,
        "rot_top_decile_by_level": [0.5 * t for t in top],
        "div_top_decile_by_level": [0.5 * t for t in top],
        "rot_total_by_level": [0.5 * t for t in total],
        "div_total_by_level": [0.5 * t for t in total],
    }


def test_the_stratospheric_gate_reads_the_top_decile_share_on_the_band_levels_only():
    from types import SimpleNamespace

    from arwen_global.insitu.stratosphere import (
        StratosphericNoiseGate,
        stratospheric_levels,
    )

    # Four levels on the reference column (A only, B = 0): 5, 30, 80, 500 hPa.
    halves = [100.0 * p for p in (1.0, 25.0, 36.0, 177.7, 1407.0)]
    vertical = SimpleNamespace(a_half_pa=halves, b_half=[0.0] * 5)
    levels = stratospheric_levels(vertical, (10.0, 100.0))
    assert [k for k, _ in levels] == [1, 2]
    gate = StratosphericNoiseGate(vertical, 1.0e-3, (10.0, 100.0))
    # Level 0 (5 hPa) and level 3 (500 hPa) are noisy and lie outside the band.
    gate.evaluate(_sample(10, 3000.0, [9.0, 1.0e-4, 2.0e-4, 9.0], [10.0, 1.0, 1.0, 10.0]))
    gate.evaluate(_sample(20, 6000.0, [9.0, 5.0e-4, 2.0e-3, 9.0], [10.0, 1.0, 1.0, 10.0]))
    gate.evaluate(_sample(30, 9000.0, [9.0, 1.0e-4, 3.0e-4, 9.0], [10.0, 1.0, 1.0, 10.0]))
    row = gate.summary()
    assert row["levels"] == [1, 2] and row["samples"] == 3
    assert row["value"] == pytest.approx(2.0e-3)
    assert row["at"]["step"] == 20 and row["at"]["level"] == 2
    assert row["first_over_limit"]["time_s"] == 6000.0
    assert row["last"]["share"] == pytest.approx(3.0e-4)
    assert row["passed"] is False
    quiet = StratosphericNoiseGate(vertical, 1.0e-2, (10.0, 100.0))
    quiet.evaluate(_sample(20, 6000.0, [9.0, 5.0e-4, 2.0e-3, 9.0], [10.0, 1.0, 1.0, 10.0]))
    assert quiet.summary()["passed"] is True and quiet.summary()["first_over_limit"] is None


def test_the_stratospheric_gate_is_off_unless_set_and_never_moves_the_run(tmp_path):
    base = load_config(_write(tmp_path, _smoke_text(), "base.toml"))
    assert base.insitu.stratospheric_noise_limit is None
    assert "stratospheric_noise_limit" not in base.insitu.identity
    with pytest.raises(ValueError, match="set stratospheric_noise_limit too"):
        insitu_options_from_table({"stratospheric_noise_band_hpa": [10.0, 100.0]})
    with pytest.raises(ValueError, match="two ascending"):
        insitu_options_from_table({"stratospheric_noise_limit": 1e-4,
                                   "stratospheric_noise_band_hpa": [100.0, 10.0]})
    with pytest.raises(ValueError, match="positive number"):
        InsituOptions(stratospheric_noise_limit=-1.0)
    # The smoke's four levels: a band wide enough to hold some of them.
    on = load_config(_write(tmp_path, _smoke_text(insitu=(
        "spectra_every = 2\nstratospheric_noise_limit = 1.0e-30\n"
        "stratospheric_noise_band_hpa = [1.0, 1100.0]")), "on.toml"))
    assert on.config_hash == base.config_hash
    off = replace(on, insitu=replace(on.insitu, stratospheric_noise_limit=None))
    receipt_on = run(on, tmp_path / "on")
    receipt_off = run(off, tmp_path / "off")
    row = receipt_on["insitu"]["stratospheric_noise"]
    assert row["samples"] == 2 and row["levels"] and row["passed"] is False
    assert "stratospheric_noise" not in receipt_off["insitu"]
    # A diagnostic: the run passes, and its forecast is the run without it.
    assert receipt_on["status"] == receipt_off["status"] == "pass"
    _arrays_equal(tmp_path / "on" / "arwen_global_step00000004.npz",
                  tmp_path / "off" / "arwen_global_step00000004.npz")


def test_the_replay_tool_reads_a_written_ledger_as_the_live_gate_did(tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "arwen_global_strat_noise_replay.py"
    spec = importlib.util.spec_from_file_location("_arwen_global_strat_noise_replay", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    config = _write(tmp_path, _smoke_text(insitu=(
        "spectra_every = 1\nstratospheric_noise_limit = 1.0e-3\n"
        "stratospheric_noise_band_hpa = [1.0, 1100.0]")), "on.toml")
    live = run(load_config(config), tmp_path / "on")["insitu"]["stratospheric_noise"]
    report = tool.replay(config, [tmp_path / "on" / "insitu.ndjson"], 1.0e-3, (1.0, 1100.0))
    replayed = report["ledgers"][str(tmp_path / "on" / "insitu.ndjson")]
    for key in ("value", "at", "last", "first_over_limit", "passed", "samples", "levels"):
        assert replayed[key] == live[key], key
