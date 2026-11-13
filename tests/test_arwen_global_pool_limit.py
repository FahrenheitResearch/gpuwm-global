"""The run holds its own pool to the figure the door priced its plan at.

The breakage these hold: MEASURED 2026-11-12 on an RTX 5070 Ti, the
one-band T383 L40 day held 14.89 GiB of pool around 12.10 GiB live and
took 15,624 MiB of a 16 GB card, because a CuPy pool asks the card for a
new block whenever no cached block serves a request and so grows into
whatever the card has free.  The pool is now held to the plan's pool
figure (``RunMemoryPlan.pool_limit_bytes``,
``DevicePeakTracker.hold_pool_to``).  These tests pin the CPU-provable
half on a stand-in pool: whose limit stands, that the run's own limit
yields instead of ending a forecast, and that a card that is really out
of memory is not papered over.  What the limit does to a card is measured
on the card.
"""
from __future__ import annotations

import dataclasses

import pytest

from arwen_global import device_memory, sizing
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.runner import hold_pool_to_plan

GIB = 2 ** 30
MIB = 2 ** 20
T383_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_sl_si_24h.toml")
T255_DAY = str(_shipped_configs() / "arwen_global_gdas_t255_native_sl_si_24h.toml")
T255_EULERIAN_DAY = str(_shipped_configs() / "arwen_global_gdas_t255_native_imex_24h.toml")


@pytest.fixture(autouse=True)
def _no_limit_in_the_shell(monkeypatch):
    """A CUPY_GPU_MEMORY_LIMIT in the shell that runs the suite is not a
    limit any test here means: a test that wants the user's limit sets it.
    MEASURED 2026-11-12 on an RTX 5070 Ti: with the variable set in the shell
    (as a run on a shared card may have it), four tests that read the
    environment through ``hold_pool_to_plan`` found ``set_by == "user"`` and
    failed."""

    monkeypatch.delenv(device_memory.POOL_LIMIT_ENV, raising=False)


class OutOfMemoryError(MemoryError):
    """Named as CuPy names its own, which is how the hook knows it."""


class _Block:
    def __init__(self, pool, size):
        self.pool = pool
        self.size = size

    def free(self):
        self.pool.live -= self.size


class _LimitedPool:
    """A pool with CuPy's limit contract: ``set_limit(size=)``,
    ``get_limit()``, and an allocation that would take the held total past
    the limit fails after the cached blocks have been handed back."""

    def __init__(self, card_bytes=None):
        self.live = 0
        self.held = 0
        self.limit = 0
        self.card = card_bytes

    def set_limit(self, size=None, fraction=None):
        self.limit = int(size or 0)

    def get_limit(self):
        return self.limit

    def malloc(self, size):
        # No cached block serves the request: the cached blocks go back
        # (CuPy's free_all_blocks before it gives up) and the card is asked.
        want = self.live + size
        if self.card is not None and want > self.card:
            raise OutOfMemoryError("the card is out of memory")
        if self.limit and want > self.limit:
            self.held = self.live
            raise OutOfMemoryError("the limit")
        self.live = want
        self.held = max(self.held, self.live)
        return _Block(self, size)

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.held


class _UnlimitedPool:
    """A pool that takes no limit: the driver's async pool, the slab."""

    def __init__(self):
        self.live = 0

    def malloc(self, size):
        self.live += size
        return _Block(self, size)

    def used_bytes(self):
        return self.live

    def total_bytes(self):
        return self.live


def test_the_run_holds_a_default_pool_to_the_plan_figure():
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    row = tracker.hold_pool_to(12 * GIB, environ={})
    assert pool.get_limit() == 12 * GIB
    assert row["set_by"] == "sizer" and row["limit_bytes"] == 12 * GIB
    assert row["lifts"] == 0
    receipt = tracker.receipt()["pool_limit"]
    assert receipt["limit_bytes_at_end"] == 12 * GIB
    assert "pool held to 12.00 GiB" in device_memory.device_peak_sentence(
        device_memory.device_memory_receipt(
            tracker, "cupy", sizing_model_peak_bytes=None))


@pytest.mark.parametrize("value", ["13g", "14495514624", "0"])
def test_a_limit_the_user_set_in_the_environment_stands(value):
    # "0" too: it is CuPy's own spelling of "no limit", and a user who
    # spelled it has said the pool is theirs.
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    row = tracker.hold_pool_to(
        12 * GIB, environ={device_memory.POOL_LIMIT_ENV: value})
    assert pool.get_limit() == 0
    assert row["set_by"] == "user"
    assert device_memory.POOL_LIMIT_ENV + "=" + value in row["reason"]
    assert device_memory.pool_limit_phrase(tracker.receipt()) == ""


def test_an_empty_environment_value_is_not_a_user_limit():
    assert device_memory.environment_pool_limit(
        {device_memory.POOL_LIMIT_ENV: "  "}) is None
    assert device_memory.environment_pool_limit({}) is None


def test_a_limit_already_on_the_pool_stands():
    pool = _LimitedPool()
    pool.set_limit(size=9 * GIB)
    tracker = device_memory.DevicePeakTracker(pool)
    row = tracker.hold_pool_to(12 * GIB, environ={})
    assert pool.get_limit() == 9 * GIB
    assert row["set_by"] == "caller" and row["limit_bytes"] == 9 * GIB
    tracker.release_pool_limit()
    assert pool.get_limit() == 9 * GIB


def test_a_pool_that_takes_no_limit_is_left_alone():
    tracker = device_memory.DevicePeakTracker(_UnlimitedPool())
    row = tracker.hold_pool_to(12 * GIB, environ={})
    assert row["set_by"] == "nobody" and row["limit_bytes"] is None
    assert "takes no limit" in row["reason"]


def test_a_pool_that_already_holds_more_is_held_to_what_it_holds():
    # A launcher's reservation on a shared card: grown before the model
    # started, the run's own to spend, and not to be handed back.
    pool = _LimitedPool()
    pool.held = 14 * GIB
    tracker = device_memory.DevicePeakTracker(pool)
    row = tracker.hold_pool_to(12 * GIB, environ={})
    assert pool.get_limit() == 14 * GIB
    assert row["set_by"] == "sizer" and row["limit_bytes"] == 14 * GIB
    assert row["asked_bytes"] == 12 * GIB
    assert "already held more" in row["reason"]


def test_a_plan_with_no_pool_figure_sets_none():
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    assert tracker.hold_pool_to(None, environ={})["set_by"] == "nobody"
    assert pool.get_limit() == 0


def test_the_runs_own_limit_yields_to_an_allocation_that_does_not_fit():
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    tracker.hold_pool_to(4 * GIB, environ={})
    a = tracker(3 * GIB)
    b = tracker(2 * GIB)        # 5 GiB live does not fit under 4
    assert pool.used_bytes() == 5 * GIB
    row = tracker.receipt()["pool_limit"]
    assert row["lifts"] == 1
    # Raised past what the allocation needed by one lift, not to no limit.
    assert pool.get_limit() == 5 * GIB + device_memory.POOL_LIMIT_LIFT_BYTES
    assert row["limit_bytes"] == 4 * GIB
    assert row["limit_bytes_at_end"] == pool.get_limit()
    assert row["lifted_bytes"] == pool.get_limit() - 4 * GIB
    # The lift fell before the second allocation was counted.
    assert row["last_lift_at_allocation"] == 1
    assert tracker.peak_used_bytes == 5 * GIB and tracker.allocations == 2
    sentence = device_memory.pool_limit_phrase(tracker.receipt())
    assert "pool held to 4.00 GiB, raised 1 time to 5.03 GiB" in sentence
    a.free()
    b.free()


def test_a_card_that_is_out_of_memory_is_not_a_limit_to_raise():
    pool = _LimitedPool(card_bytes=6 * GIB)
    tracker = device_memory.DevicePeakTracker(pool)
    tracker.hold_pool_to(8 * GIB, environ={})
    tracker(5 * GIB)
    with pytest.raises(OutOfMemoryError, match="the card"):
        tracker(2 * GIB)        # under the limit, over the card
    assert tracker.receipt()["pool_limit"]["lifts"] == 0
    assert pool.get_limit() == 8 * GIB


def test_a_limit_somebody_else_moved_is_not_the_runs_to_raise():
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    tracker.hold_pool_to(4 * GIB, environ={})
    pool.set_limit(size=3 * GIB)
    with pytest.raises(OutOfMemoryError):
        tracker(5 * GIB)
    assert pool.get_limit() == 3 * GIB


def test_any_other_error_passes_through_untouched():
    class Broken(_LimitedPool):
        def malloc(self, size):
            raise ValueError("not a memory error")

    tracker = device_memory.DevicePeakTracker(Broken())
    tracker.hold_pool_to(4 * GIB, environ={})
    with pytest.raises(ValueError):
        tracker(GIB)


def test_the_pool_is_left_as_it_was_found():
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    tracker.hold_pool_to(4 * GIB, environ={})
    tracker(5 * GIB)            # one lift
    tracker.release_pool_limit()
    assert pool.get_limit() == 0
    # The receipt still says what the run was held to.
    row = tracker.receipt()["pool_limit"]
    assert row["limit_bytes"] == 4 * GIB and row["lifts"] == 1


def test_a_run_no_door_priced_is_not_held():
    assert hold_pool_to_plan(None, None) is None
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    row = hold_pool_to_plan(tracker, None)
    assert row["set_by"] == "nobody" and pool.get_limit() == 0
    assert "no door priced this run" in row["reason"]


class _Allocator:
    def __init__(self, name):
        self.name = name

    def receipt(self):
        return {"allocator": self.name}


@pytest.mark.parametrize("name", ["async", "slab"])
def test_a_run_under_another_allocator_is_not_held(name):
    # The driver's async pool and the slab bound themselves their own way
    # and are priced at their own runs; the run asks neither for a limit.
    pool = _LimitedPool()
    tracker = device_memory.DevicePeakTracker(pool)
    plan = sizing.RunMemoryPlan(
        bands=1, spill_slices=(), live_peak_bytes=8 * GIB,
        card_bytes=10 * GIB, spilled_bytes=0, budget_bytes=None,
        free_bytes=None, bands_chosen_by="sizer", spill_chosen_by="sizer",
        reason="", pool_limit_bytes=9 * GIB)
    row = hold_pool_to_plan(tracker, plan, _Allocator(name))
    assert row["set_by"] == "nobody" and pool.get_limit() == 0
    assert f"the {name} allocator takes no limit" in row["reason"]
    held = hold_pool_to_plan(tracker, plan, _Allocator("default"))
    assert held["set_by"] == "sizer" and pool.get_limit() == 9 * GIB
    tracker.release_pool_limit()


def test_the_async_pools_no_limit_reading_is_not_somebodys_limit():
    # MEASURED 2026-11-12 (T383, four bands, --device-allocator async): the
    # driver's pool reads 2**64 - 1 from get_limit() when nothing limited
    # it, and the receipt called that a limit the pool already carried.
    class AsyncLike(_LimitedPool):
        def get_limit(self):
            return self.limit or 2**64 - 1

    pool = AsyncLike()
    tracker = device_memory.DevicePeakTracker(pool)
    plan = sizing.RunMemoryPlan(
        bands=4, spill_slices=(), live_peak_bytes=7 * GIB,
        card_bytes=9 * GIB, spilled_bytes=0, budget_bytes=None,
        free_bytes=None, bands_chosen_by="sizer", spill_chosen_by="sizer",
        reason="", pool_limit_bytes=8 * GIB)
    row = hold_pool_to_plan(tracker, plan, _Allocator("async"))
    assert row["set_by"] == "nobody" and row["limit_bytes"] is None
    assert "the async allocator takes no limit" in row["reason"]
    assert device_memory.pool_limit_phrase(tracker.receipt()) == ""


def test_a_pool_that_refuses_the_limit_is_left_alone():
    class Refuses(_LimitedPool):
        def set_limit(self, size=None, fraction=None):
            raise NotImplementedError("no limit here")

    pool = Refuses()
    tracker = device_memory.DevicePeakTracker(pool)
    row = tracker.hold_pool_to(4 * GIB, environ={})
    assert row["set_by"] == "nobody" and "did not take the limit" in row["reason"]
    block = tracker(5 * GIB)     # and nothing is lifted, because nothing was set
    assert tracker.receipt()["pool_limit"].get("lifts") is None
    block.free()


# --- the plan carries the figure, and the door's line says it -------------


def _estimate(config, **changes):
    cfg = dataclasses.replace(load_config(config), **changes)
    return cfg, dataclasses.replace(
        sizing.estimate_global_memory(cfg), backend="cupy")


def test_a_plan_the_model_priced_is_held_to_the_pool_part_of_its_price():
    # The Eulerian T255 day: no run of its plan is on record, so the model
    # prices it (the semi-Lagrangian T255 day is priced at its held runs).
    cfg, estimate = _estimate(T255_EULERIAN_DAY)
    plan = sizing.plan_run_memory(cfg, 15 * GIB, estimate)
    assert plan.fits
    assert sizing.measured_card_row(estimate, plan.bands, plan.spill_slices) is None
    assert plan.pool_limit_bytes == plan.card_bytes - sizing.OUT_OF_POOL_BYTES
    assert plan.receipt()["pool_limit_bytes"] == plan.pool_limit_bytes


def test_a_run_with_no_card_to_price_against_carries_no_pool_figure():
    cfg = load_config(T255_DAY)
    assert sizing.plan_run_memory(cfg, None).pool_limit_bytes is None
    cfg, estimate = _estimate(T255_DAY)
    assert sizing.plan_run_memory(cfg, None, estimate).pool_limit_bytes is None
    hopeless = sizing.plan_run_memory(cfg, GIB, estimate)
    assert not hopeless.fits and hopeless.pool_limit_bytes is None


def test_the_doors_line_says_what_the_pool_is_held_to(monkeypatch):
    monkeypatch.delenv(device_memory.POOL_LIMIT_ENV, raising=False)
    cfg, _estimate_ = _estimate(T255_DAY)
    cfg = dataclasses.replace(cfg, backend="cupy")
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": 15 * GIB})
    plan = gate["plan"]
    assert plan.fits and plan.pool_limit_bytes
    assert (f"the pool is held to {plan.pool_limit_bytes / GIB:.2f} GiB"
            in gate["verdict"])
    assert "the limit is raised if the run needs more" in gate["verdict"]


def test_the_doors_line_says_when_the_limit_is_the_users(monkeypatch):
    monkeypatch.setenv(device_memory.POOL_LIMIT_ENV, "13g")
    cfg, _estimate_ = _estimate(T255_DAY)
    cfg = dataclasses.replace(cfg, backend="cupy")
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": 15 * GIB})
    assert "CUPY_GPU_MEMORY_LIMIT=13g is set" in gate["verdict"]
    assert "the pool is held to" not in gate["verdict"]


def test_an_allocator_that_takes_no_limit_is_not_promised_one(monkeypatch):
    monkeypatch.delenv(device_memory.POOL_LIMIT_ENV, raising=False)
    cfg, _estimate_ = _estimate(T255_DAY, device_allocator="async")
    cfg = dataclasses.replace(cfg, backend="cupy")
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": 15 * GIB})
    assert "the pool is held to" not in gate["verdict"]


# --- the one-band T383 plan on the 16 GB card, held to its limit ----------
#
# MEASURED 2026-11-12 on the RTX 5070 Ti at the 5,000-column chunk: with the
# pool growing into the card the one-band day took 15,624 MiB and the door
# asked 14.93 GiB free for it.  Held to a limit at its live peak the plan's
# pool stays where its allocations need it, and the door prices the plan
# at what the held run took.


def _t383_5000():
    cfg = load_config(T383_DAY)
    options = dict(cfg.native_adapter_options, radiation_column_chunk=5_000)
    cfg = dataclasses.replace(cfg, native_adapter_options=options)
    return cfg, dataclasses.replace(
        sizing.estimate_global_memory(cfg), backend="cupy")


def _one_band_row():
    rows = [row for row in sizing.COMPLETED_PLAN_PEAKS
            if row["truncation"] == 383 and row["bands"] == 1
            and not row["spill_slices"]
            and row["radiation_column_chunk"] == 5_000]
    assert len(rows) == 1
    return rows[0]


def test_the_one_band_t383_plan_is_priced_at_the_run_held_to_its_limit():
    row = _one_band_row()
    cfg, estimate = _t383_5000()
    plan = sizing.plan_run_memory(cfg, int(15.6 * GIB), estimate)
    assert plan.fits and plan.bands == 1 and plan.spill_slices == ()
    assert plan.card_bytes == row["card_bytes_measured"]
    assert plan.pool_limit_bytes == row["pool_limit_bytes"]
    # The limit starts at the plan's live peak, and the card figure is the
    # held run's: well under what the door asked before the limit.
    assert row["pool_limit_bytes"] <= row["peak_used_bytes"]
    assert row["pool_limit_end_bytes"] >= row["held_bytes"]
    assert plan.card_bytes < int(14.93 * GIB) - GIB // 2


def test_one_band_is_taken_from_the_held_runs_card_figure_up(monkeypatch):
    monkeypatch.delenv(device_memory.POOL_LIMIT_ENV, raising=False)
    row = _one_band_row()
    cfg, estimate = _t383_5000()
    card = int(row["card_bytes_measured"])
    assert sizing.plan_run_memory(cfg, card, estimate).bands == 1
    below = sizing.plan_run_memory(cfg, card - MIB, estimate)
    assert below.fits and below.bands == 2
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": card})
    assert (f"the pool is held to {row['pool_limit_bytes'] / GIB:.2f} GiB (the "
            "limit the completed run of exactly this plan started from and "
            f"raised to {row['pool_limit_end_bytes'] / GIB:.2f} GiB)"
            in gate["verdict"])
    assert (f"priced at the {card / GIB:.2f} GiB of card a completed run of "
            "exactly this plan took" in gate["verdict"])
