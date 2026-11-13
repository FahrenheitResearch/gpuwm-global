"""A run the card runs out of memory under carries on, or stops cleanly.

The breakage these hold: MEASURED 2026-11-12 on an RTX 5070 Ti (16 GB),
the one-band T383 L40 semi-Lagrangian plan, each segment a new process
started from a checkpoint: with five 222 MiB CUDA contexts of
another program opening after the run's door had read the card, one
segment in eleven died with CuPy's ``OutOfMemoryError`` on the 810 MiB
semi-Lagrangian gather stack (the card had 798 MiB free), and a fresh
process from the same checkpoint then got through.  The run now gives its
memory back, reads the card again and carries on from its newest
checkpoint on the next plan priced below the one that ran out
(``runner.run``, ``sizing.fallback_plan``), or stops with a message that
names the cause, the free memory, what else holds the card and the flags
to resume with (``device_memory.DeviceMemoryExhausted``).

What is pinned here is the CPU-provable half, with the failure forced: the
error is recognised, the allocator hook retries once and records what the
card showed, the door names the card's other tenants and leaves them a
margin, the lighter plan is the door's own price, and a run that falls back
writes the checkpoints the uninterrupted run writes, byte for byte.  What
it does on a card is measured on the card.
"""
from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import numpy as np
import pytest

from arwen_global import device_memory, runner, sizing
from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.dynamics import MoistHybridModel
from arwen_global.receipt import check_receipt

GIB = 2 ** 30
MIB = 2 ** 20
SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
T383_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_sl_si_24h.toml")


@pytest.fixture(autouse=True)
def _no_limit_in_the_shell(monkeypatch):
    monkeypatch.delenv(device_memory.POOL_LIMIT_ENV, raising=False)


class OutOfMemoryError(MemoryError):
    """Named as CuPy names its own (``cupy.cuda.memory.OutOfMemoryError``)."""


class CUDADriverError(RuntimeError):
    """Named as CuPy's driver error is; its text carries the driver's code."""


CUDADriverError.__module__ = "cupy_backends.cuda.api.driver"


# -- recognising the error ----------------------------------------------------


def test_the_pool_and_the_driver_out_of_memory_are_recognised():
    assert device_memory.is_out_of_memory(OutOfMemoryError(
        "Out of memory allocating 849,346,560 bytes (allocated so far: "
        "13,836,252,672 bytes, limit set to: 14,719,153,664 bytes)."))
    assert device_memory.is_out_of_memory(
        CUDADriverError("CUDA_ERROR_OUT_OF_MEMORY: out of memory"))
    # Raised from inside something else: the chain is followed.
    try:
        try:
            raise OutOfMemoryError("pool")
        except MemoryError as inner:
            raise ValueError("a step failed") from inner
    except ValueError as outer:
        assert device_memory.is_out_of_memory(outer)


def test_a_host_memory_error_and_other_cuda_errors_are_not():
    assert not device_memory.is_out_of_memory(MemoryError("host"))
    assert not device_memory.is_out_of_memory(
        CUDADriverError("CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access"))
    assert not device_memory.is_out_of_memory(ValueError("out of memory"))
    # The package's own verdict after a fallback was weighed is final.
    assert not device_memory.is_out_of_memory(
        device_memory.DeviceMemoryExhausted("the card ran out of memory"))


# -- the allocator hook -------------------------------------------------------


class _Block:
    def __init__(self, size):
        self.size = size


class _RefusingPool:
    """A pool whose card refuses ``refusals`` requests, each until the cached
    blocks are handed back; ``free_all_blocks`` counts the hand-backs."""

    def __init__(self, refusals=1, cached=3 * GIB):
        self.refusals = refusals
        self.live = 10 * GIB
        self.held = self.live + cached
        self.released = 0

    def malloc(self, size):
        if self.refusals > 0:
            self.refusals -= 1
            raise OutOfMemoryError(f"Out of memory allocating {size:,} bytes")
        self.live += size
        self.held = max(self.held, self.live)
        return _Block(size)

    def free_all_blocks(self):
        self.released += 1
        self.held = self.live

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.held

    def get_limit(self):
        return 0


def test_a_refusal_gives_the_cached_blocks_back_and_is_retried_once():
    pool = _RefusingPool(refusals=1)
    tracker = device_memory.DevicePeakTracker(pool)
    block = tracker(849_346_560)
    assert block.size == 849_346_560
    assert pool.released == 1
    assert tracker.refusals == 1
    row = tracker.last_refusal
    assert row["request_bytes"] == 849_346_560
    assert row["pool_total_bytes"] == 13 * GIB
    assert row["pool_used_bytes"] == 10 * GIB
    assert row["retried"] is True
    receipt = tracker.receipt()
    assert receipt["device_refusals"] == 1
    assert receipt["last_refusal"]["request_bytes"] == 849_346_560


def test_a_card_that_refuses_again_raises_the_card_error():
    pool = _RefusingPool(refusals=2)
    tracker = device_memory.DevicePeakTracker(pool)
    with pytest.raises(OutOfMemoryError):
        tracker(849_346_560)
    assert pool.released == 1
    assert tracker.last_refusal["request_bytes"] == 849_346_560


def test_an_error_that_is_not_the_card_is_raised_untouched():
    class _Broken(_RefusingPool):
        def malloc(self, size):
            raise ValueError("not memory")

    pool = _Broken()
    tracker = device_memory.DevicePeakTracker(pool)
    with pytest.raises(ValueError, match="not memory"):
        tracker(1024)
    assert pool.released == 0 and tracker.refusals == 0


# -- what else holds the card -------------------------------------------------


def _smi(cards, apps):
    """A ``subprocess.run`` stand-in answering the two nvidia-smi queries."""

    def run(argv, **_kwargs):
        query = next(a for a in argv if a.startswith("--query"))
        rows = cards if query.startswith("--query-gpu") else apps
        return types.SimpleNamespace(returncode=0, stdout="\n".join(rows) + "\n")

    return run


#: A card with another program's process and its five workers on it before a
#: forecast started (a headless 16 GB card; a display server is a graphics
#: process and is not in the compute list).
_SHARED_CARD = ["0, GPU-aaaa, Disabled"]
_SHARED_APPS = [
    "1100, /opt/env/bin/python3.12, 280, GPU-aaaa",
    *[f"{pid}, python3, 222, GPU-aaaa" for pid in (1101, 1102, 1103,
                                                   1104, 1105)],
]


def test_the_other_processes_on_the_card_are_read_and_this_one_is_not():
    seen = device_memory.card_neighbours(
        environ={}, run=_smi(_SHARED_CARD, _SHARED_APPS + ["4242, me, 300, GPU-aaaa"]),
        own_pid=4242)
    assert seen["card"] == "GPU-aaaa"
    assert seen["display_active"] is False
    assert [p["pid"] for p in seen["processes"]] == [
        1100, 1101, 1102, 1103, 1104, 1105]
    assert seen["other_bytes"] == (280 + 5 * 222) * MIB
    assert seen["own_bytes"] == 300 * MIB
    phrase = device_memory.neighbours_phrase(seen)
    assert "6 other processes hold 1,390 MiB on this card" in phrase
    assert "pid 1101 python3 222 MiB" in phrase


def test_another_cards_processes_are_not_this_cards():
    cards = ["0, GPU-aaaa, Disabled", "1, GPU-bbbb, Enabled"]
    apps = ["11, python3, 222, GPU-bbbb"]
    seen = device_memory.card_neighbours(
        environ={"CUDA_VISIBLE_DEVICES": "GPU-aaaa"}, run=_smi(cards, apps),
        own_pid=1)
    assert seen["processes"] == [] and seen["other_bytes"] == 0
    # With two cards and no way to tell CUDA's device 0 without a context,
    # nothing is attributed rather than the wrong card.
    assert device_memory.card_neighbours(
        environ={}, run=_smi(cards, apps), own_pid=1) is None
    shown = device_memory.card_neighbours(
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        run=_smi(cards, apps), own_pid=1)
    assert shown["display_active"] is True
    assert shown["other_bytes"] == 222 * MIB


def test_a_missing_nvidia_smi_reads_as_not_read():
    def run(argv, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert device_memory.card_neighbours(environ={}, run=run) is None
    assert "could not be read" in device_memory.neighbours_phrase(None)


# -- the door's margin --------------------------------------------------------


def _gate(free_mib, neighbours):
    cfg = load_config(T383_DAY)
    return sizing.run_memory_gate(
        cfg, probe={"free_bytes": free_mib * MIB}, neighbours=neighbours)


def test_an_empty_headless_card_keeps_the_one_band_plan():
    alone = {"card": "GPU-aaaa", "display_active": False, "processes": [],
             "other_bytes": 0, "own_bytes": None}
    gate = _gate(15_000, alone)
    assert gate["latitude_bands"] == 1
    assert gate["shared_card_margin_bytes"] == 0


def test_other_tenants_on_the_card_are_left_a_margin_and_named():
    # 15,000 MiB free fits the one-band plan's 14,608 MiB, and does not once
    # half a GiB is left to the card's other tenants.
    shared = {"card": "GPU-aaaa", "display_active": False,
              "processes": [{"pid": 7, "name": "python3", "used_bytes": 222 * MIB}],
              "other_bytes": 222 * MIB, "own_bytes": None}
    gate = _gate(15_000, shared)
    assert gate["latitude_bands"] == 2 and gate["plan"].spill_slices
    assert gate["shared_card_margin_bytes"] == device_memory.SHARED_CARD_MARGIN_BYTES
    assert "1 other process holds 222 MiB on this card" in gate["verdict"]
    assert not gate["refuse"]


def test_a_card_that_drives_a_display_is_left_the_margin():
    display = {"card": "GPU-aaaa", "display_active": True, "processes": [],
               "other_bytes": 0, "own_bytes": None}
    gate = _gate(15_000, display)
    assert gate["latitude_bands"] == 2
    assert "drives a display" in gate["verdict"]


def test_the_margin_never_turns_into_a_refusal():
    shared = {"card": "GPU-aaaa", "display_active": True, "processes": [],
              "other_bytes": 0, "own_bytes": None}
    lightest = min(card for _b, _s, card in sizing.lighter_plans(
        load_config(T383_DAY), 1, ()))
    gate = _gate(lightest // MIB + 64, shared)
    assert not gate["refuse"]
    assert gate["shared_card_margin_bytes"] == 0
    assert "chosen against the whole free figure" in gate["verdict"]


# -- the lighter plan ---------------------------------------------------------


def test_the_plan_after_one_band_is_the_one_the_door_prices_lower():
    cfg = load_config(T383_DAY)
    lighter = sizing.lighter_plans(cfg, 1, ())
    one_band = sizing.plan_run_memory(
        dataclasses.replace(cfg, latitude_bands=1, host_spill="off"),
        1 << 62).card_bytes
    assert lighter and all(card < one_band for _b, _s, card in lighter)
    # Two bands with nothing parked prices ABOVE one band at T383 L40, so it
    # is not a lighter plan; two bands with the host tier is the first.
    assert lighter[0][:2] == (2, True)
    plan = sizing.fallback_plan(cfg, 14_400 * MIB, 1, (), why="after test")
    assert plan.bands == 2 and plan.spill_slices
    assert plan.bands_chosen_by == "fallback"
    assert plan.reason.startswith("after test; ")
    assert plan.pool_limit_bytes
    # Nothing priced below the lightest plan fits a card smaller than it.
    assert sizing.fallback_plan(cfg, 4 * GIB, 1, ()) is None
    assert sizing.fallback_plan(cfg, None, 1, ()) is None


# -- the run: carry on, or stop cleanly ---------------------------------------


@pytest.fixture
def forced_out_of_memory(monkeypatch):
    """Make MoistHybridModel.step raise CuPy's out-of-memory error once, on
    the step that would produce ``at``, and let the numpy smoke run take the
    fallback path a cupy run takes (the fallback needs a card to read, so
    the card's readings are stood in)."""

    calls = {"armed": None, "raised": 0}
    original = MoistHybridModel.step

    def step(self, state, dt_s):
        if calls["armed"] is not None and int(state.step) + 1 == calls["armed"]:
            calls["armed"] = None
            calls["raised"] += 1
            raise OutOfMemoryError(
                "Out of memory allocating 849,346,560 bytes (allocated so far: "
                "13,836,252,672 bytes, limit set to: 14,719,153,664 bytes).")
        return original(self, state, dt_s)

    monkeypatch.setattr(MoistHybridModel, "step", step)
    monkeypatch.setattr(
        runner, "_may_fall_back",
        lambda cfg, attempt, error: (not attempt.get("loop_done")
                                     and device_memory.is_out_of_memory(error)))
    monkeypatch.setattr(sizing, "_free_device_bytes", lambda: 14_400 * MIB)
    monkeypatch.setattr(device_memory, "card_neighbours", lambda: {
        "card": "GPU-aaaa", "display_active": False,
        "processes": [{"pid": 7, "name": "python3", "used_bytes": 222 * MIB}],
        "other_bytes": 222 * MIB, "own_bytes": None})
    return calls


def _smoke():
    """The smoke run on eight latitude rows, so that two bands of four rows
    (the band floor) are a plan its grid allows."""
    return dataclasses.replace(load_config(SMOKE), nlat=8, nlon=16)


def _two_band_plan(cfg, *_args, **_kwargs):
    plan = sizing.plan_run_memory(
        dataclasses.replace(cfg, latitude_bands=2, host_spill="off"), None)
    return dataclasses.replace(plan, bands_chosen_by="fallback",
                               spill_chosen_by="fallback")


def _assert_same_checkpoints(a, b, steps):
    for step in steps:
        name = f"arwen_global_step{step:08d}.npz"
        meta_a, arrays_a = read_checkpoint(a / name)
        meta_b, arrays_b = read_checkpoint(b / name)
        assert meta_a["run_trackers"] == meta_b["run_trackers"], step
        assert arrays_a.keys() == arrays_b.keys(), step
        for key in arrays_a:
            assert arrays_a[key].dtype == arrays_b[key].dtype, (step, key)
            assert np.array_equal(arrays_a[key], arrays_b[key]), (step, key)
            assert arrays_a[key].tobytes() == arrays_b[key].tobytes(), (step, key)


def test_a_run_that_runs_out_carries_on_on_a_lighter_plan_to_the_same_bits(
        tmp_path, forced_out_of_memory, monkeypatch):
    cfg = _smoke()
    reference = tmp_path / "reference"
    runner.run(cfg, reference)

    monkeypatch.setattr(sizing, "fallback_plan", _two_band_plan)
    said = []
    forced_out_of_memory["armed"] = 3
    out = tmp_path / "fell-back"
    result = runner.run(cfg, out, notice=said.append)

    assert forced_out_of_memory["raised"] == 1
    assert result["status"] == "pass"
    _assert_same_checkpoints(reference, out, (0, 2, 4))
    receipt = check_receipt(out / "arwen-global-receipt.json")
    (row,) = receipt["recovery"]
    assert row["last_completed_step"] == 2
    assert row["resumed_from_step"] == 2
    assert row["from_plan"]["latitude_bands"] == 1
    assert row["to_plan"]["latitude_bands"] == 2
    assert row["to_plan"]["bands_chosen_by"] == "fallback"
    assert receipt["latitude_bands"]["latitude_bands"] == 2
    assert receipt["latitude_bands"]["chosen_by"] == "fallback"
    # The segment started where the first attempt did, and the whole run's
    # checkpoints are listed once each in the directory.
    assert receipt["segment_start_diagnostics"]["step"] == 0
    assert sorted(p.name for p in out.glob("arwen_global_step*.npz")) == [
        f"arwen_global_step{s:08d}.npz" for s in (0, 2, 4)]
    # The receipt lists the dead attempt's checkpoints before the ones the
    # attempt that carried on wrote.
    assert [Path(p).name for p in receipt["checkpoints"]] == [
        f"arwen_global_step{s:08d}.npz" for s in (0, 2, 4)]
    assert _diagnostics_steps(out) == _diagnostics_steps(reference)
    (sentence,) = said
    assert "the card ran out of memory after step 2" in sentence
    assert "Carrying on from step 2 on 2 latitude bands" in sentence
    assert "1 other process holds 222 MiB" in sentence


def _diagnostics_steps(outdir):
    import json

    return [json.loads(line)["step"] for line in
            (outdir / runner.DIAGNOSTICS_NAME).read_text().splitlines()]


def test_a_death_between_a_row_and_its_checkpoint_leaves_one_row_a_step(
        tmp_path, forced_out_of_memory, monkeypatch):
    cfg = _smoke()
    reference = tmp_path / "reference"
    runner.run(cfg, reference)
    monkeypatch.setattr(sizing, "fallback_plan", _two_band_plan)
    original = runner._CheckpointWriter.submit
    raised = []

    def submit(self, path, state, trackers, **kwargs):
        # The step's diagnostics row is on disk; its checkpoint's copy off
        # the card is where the card refuses.
        if int(state.step) == 2 and not raised:
            raised.append(2)
            raise OutOfMemoryError("Out of memory allocating 849,346,560 bytes")
        return original(self, path, state, trackers, **kwargs)

    monkeypatch.setattr(runner._CheckpointWriter, "submit", submit)
    out = tmp_path / "fell-back"
    result = runner.run(cfg, out, notice=lambda _s: None)
    assert raised == [2]
    assert result["status"] == "pass"
    (row,) = check_receipt(out / "arwen-global-receipt.json")["recovery"]
    assert row["resumed_from_step"] == 0
    _assert_same_checkpoints(reference, out, (0, 2, 4))
    assert _diagnostics_steps(out) == _diagnostics_steps(reference)


def test_a_restarted_segment_that_runs_out_carries_on_to_the_same_bits(
        tmp_path, forced_out_of_memory, monkeypatch):
    cfg = _smoke()
    reference = tmp_path / "reference"
    runner.run(cfg, reference)
    monkeypatch.setattr(sizing, "fallback_plan", _two_band_plan)
    forced_out_of_memory["armed"] = 4
    out = tmp_path / "segment"
    result = runner.run(
        cfg, out, restart=reference / "arwen_global_step00000002.npz",
        notice=lambda _s: None)
    assert result["status"] == "pass"
    _assert_same_checkpoints(reference, out, (4,))
    (row,) = check_receipt(out / "arwen-global-receipt.json")["recovery"]
    # Nothing was written in the segment before the death, so it carried on
    # from the checkpoint it started from.
    assert row["resumed_from_step"] == 2
    assert row["original_restart"].endswith("arwen_global_step00000002.npz")


def test_a_run_no_lighter_plan_fits_stops_with_what_to_do(
        tmp_path, forced_out_of_memory, monkeypatch):
    cfg = _smoke()
    monkeypatch.setattr(sizing, "fallback_plan", lambda *a, **k: None)
    monkeypatch.setattr(sizing, "lighter_plans",
                        lambda *a, **k: [(2, True, 10 * GIB)])
    forced_out_of_memory["armed"] = 3
    out = tmp_path / "stopped"
    with pytest.raises(device_memory.DeviceMemoryExhausted) as caught:
        runner.run(cfg, out, until_s=40.0, notice=lambda _s: None)
    message = str(caught.value)
    assert "the card ran out of memory after step 2" in message
    assert "14.06 GiB free" in message
    assert "1 other process holds 222 MiB on this card" in message
    assert "no lighter plan fits it" in message
    assert "stopped cleanly" in message
    assert f"--restart {out / 'arwen_global_step00000002.npz'}" in message
    assert "--latitude-bands 2 --host-spill on --until-s 40" in message
    # The failure receipt of the attempt that died is on disk.
    receipt = check_receipt(out / "arwen-global-receipt.json")
    assert receipt["status"] == "error"
    assert receipt["error_type"] == "OutOfMemoryError"


def test_a_user_pool_limit_is_read_from_the_pool_only_when_the_user_set_it():
    class Pool:
        def __init__(self, limit):
            self.limit = limit

        def get_limit(self):
            return self.limit

    env = {"CUPY_GPU_MEMORY_LIMIT": "50%"}
    assert runner._user_pool_limit_bytes(Pool(8 * GIB), env) == 8 * GIB
    # Not set, CuPy's "no limit", or no pool: nothing to weigh.
    assert runner._user_pool_limit_bytes(Pool(8 * GIB), {}) is None
    assert runner._user_pool_limit_bytes(Pool(0), env) is None
    assert runner._user_pool_limit_bytes(None, env) is None


def test_a_refusal_under_the_users_limit_names_the_limit_not_the_card():
    refusal = {"request_bytes": 101 * MIB, "card_free_bytes": 9_284 * MIB,
               "pool_total_bytes": 6 * GIB - 50 * MIB,
               "pool_used_bytes": 6 * GIB - 60 * MIB}
    words = runner._refusal_words(refusal, "", 6 * GIB)
    assert words.startswith(
        "the pool limit CUPY_GPU_MEMORY_LIMIT sets (6.00 GiB) refused 101 MiB")
    # A request that fit under the limit was the card's refusal.
    assert runner._refusal_words(dict(refusal, request_bytes=10 * MIB), "",
                                 6 * GIB).startswith("the card refused 10 MiB")
    assert runner._refusal_words(refusal, "").startswith("the card refused")
    # The error's own full stop does not run into the sentence's semicolon.
    assert runner._refusal_words(None, "OutOfMemoryError: bytes).") == \
        "the CUDA call failed with OutOfMemoryError: bytes)"


def test_a_run_under_a_user_limit_no_plan_fits_stops_naming_the_limit(
        tmp_path, forced_out_of_memory, monkeypatch):
    cfg = _smoke()
    # The card has 14 GiB free, but the user's 6 GiB pool limit is what
    # binds: the 10 GiB plan is not tried, and the remedy is the limit.
    monkeypatch.setattr(sizing, "lighter_plans",
                        lambda *a, **k: [(2, True, 10 * GIB)])
    monkeypatch.setattr(runner, "_user_pool_limit_bytes",
                        lambda pool, environ=None: 6 * GIB)
    forced_out_of_memory["armed"] = 3
    out = tmp_path / "stopped"
    with pytest.raises(device_memory.DeviceMemoryExhausted) as caught:
        runner.run(cfg, out, until_s=40.0, notice=lambda _s: None)
    message = str(caught.value)
    assert "CUPY_GPU_MEMORY_LIMIT holds this run's pool to 6.00 GiB" in message
    assert "no lighter plan fits under that limit" in message
    assert "Raise or unset CUPY_GPU_MEMORY_LIMIT and resume with" in message
    assert "Free the card" not in message


def test_a_run_under_a_user_limit_carries_on_on_a_plan_under_it(
        tmp_path, forced_out_of_memory, monkeypatch):
    cfg = _smoke()
    weighed = []

    def fallback(cfg, free, *args, **kwargs):
        weighed.append(free)
        return _two_band_plan(cfg)

    monkeypatch.setattr(sizing, "fallback_plan", fallback)
    monkeypatch.setattr(runner, "_user_pool_limit_bytes",
                        lambda pool, environ=None: 12 * GIB)
    forced_out_of_memory["armed"] = 3
    said = []
    result = runner.run(cfg, tmp_path / "out", notice=said.append)
    assert result["status"] == "pass"
    # Weighed against the limit plus the out-of-pool term, not the card.
    assert weighed == [12 * GIB + sizing.OUT_OF_POOL_BYTES]
    (row,) = check_receipt(
        tmp_path / "out" / "arwen-global-receipt.json")["recovery"]
    assert row["user_pool_limit_bytes"] == 12 * GIB
    assert "CUPY_GPU_MEMORY_LIMIT holds this run's pool to 12.00 GiB" in said[0]


def test_the_door_exits_at_the_device_code_for_a_clean_stop():
    from arwen_global import cli

    assert cli.exit_code_for(device_memory.DeviceMemoryExhausted("x")) == \
        cli.EXIT_DEVICE


def test_a_numpy_run_does_not_fall_back(tmp_path, monkeypatch):
    cfg = _smoke()
    attempt = {}
    assert not runner._may_fall_back(cfg, attempt, OutOfMemoryError("x"))
    cupy_cfg = dataclasses.replace(cfg, backend="cupy")
    assert runner._may_fall_back(cupy_cfg, attempt, OutOfMemoryError("x"))
    assert not runner._may_fall_back(cupy_cfg, {"loop_done": True},
                                     OutOfMemoryError("x"))
    assert not runner._may_fall_back(cupy_cfg, attempt, MemoryError("host"))
