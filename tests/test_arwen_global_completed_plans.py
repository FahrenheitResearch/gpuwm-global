"""A completed run of a lighter plan bounds a heavier plan's live figure
from above (sizing.COMPLETED_PLAN_PEAKS, sizing.completed_plan_ceiling).

The breakage these hold: MEASURED 2026-09-07, the model read 15.08 GiB
live for T383 L40 at two bands with every slice parked and refused the
34.7 km day on a 16 GB card that had just completed it at 12.41 GiB live
(RTX 5070 Ti, the 24 h day run); and read 22.81 GiB for T533 L40 at
one band with every slice parked, at a pool ratio borrowed from another
class, where the 25 km day completed on the RTX 5090 at 21.12 GiB live
(the 25 km day run).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global import sizing
from arwen_global.config import load_config


GIB = 2 ** 30
T383_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_sl_si_24h.toml")
T533_DAY = str(_shipped_configs() / "arwen_global_gdas_t533_native_sl_si_24h.toml")
T383_EULERIAN_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_imex_24h.toml")
ALL = ("physics", "surface", "tracers")


def _at_chunk(config, chunk):
    """A shipped config at another radiation chunk.  The shipped T383 day
    carries 5,000 columns since 2026-11-12; the rows below that were
    measured at its former 12,500 are held at that chunk here."""

    cfg = load_config(config)
    options = dict(cfg.native_adapter_options, radiation_column_chunk=int(chunk))
    return dataclasses.replace(cfg, native_adapter_options=options)


def _plan(config, free_gib):
    cfg = load_config(config) if isinstance(config, str) else config
    estimate = sizing.estimate_global_memory(cfg)
    estimate = dataclasses.replace(estimate, backend="cupy")
    return sizing.plan_run_memory(cfg, int(free_gib * GIB), estimate)


def test_the_34_7_km_day_is_admitted_on_the_16_gb_card_at_the_plan_that_completed():
    plan = _plan(_at_chunk(T383_DAY, 12_500), 15.28)
    assert plan.fits
    assert plan.bands == 2 and plan.spill_slices == ALL
    assert plan.live_peak_bytes == 13_322_662_912
    assert plan.card_bytes <= int(15.28 * GIB)
    assert plan.ceiling and "34.7 km day" in plan.ceiling
    assert "a completed run of this shape held" in plan.reason
    assert plan.receipt()["prediction_margin"] == 1.0
    assert plan.receipt()["live_peak_ceiling"] == plan.ceiling


def test_the_25_km_day_is_admitted_on_the_32_gb_card_at_the_plan_that_completed():
    plan = _plan(T533_DAY, 30.9)
    assert plan.fits
    assert plan.bands == 1 and plan.spill_slices == ALL
    # This tree's own day: 23.23 GiB live where the model reads 22.81, and
    # a row at exactly the plan is the figure, above or below the model.
    assert plan.live_peak_bytes == 24_940_350_464
    assert plan.ceiling and "25 km day" in plan.ceiling
    assert plan.receipt()["prediction_margin"] == 1.0
    # The pool ratio is the plan's own class, measured on this tree's
    # runs of it (x1.2102), not the largest of every class.
    assert plan.fragmentation[0] == pytest.approx(30_183_100_928 / 24_940_278_784)


def test_a_ceiling_never_raises_the_model_and_never_crosses_a_core():
    # T533 at 32 bands with every slice parked: the merged tree's own
    # ten-step completion at that count is the lightest applicable row, and
    # the model, which now carries what the banded step holds whole, reads
    # above it; the ceiling can only lower the model to what a run took.
    cfg = load_config(T533_DAY)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    sizes = sizing._spill_census_estimate(cfg, estimate)
    model = sizing.banded_device_peak_bytes(estimate, 32, spilled_bytes=sum(sizes.values()))
    row = sizing.completed_plan_ceiling(estimate, 32, ALL)
    assert row is not None and int(row["peak_used_bytes"]) == 13_838_927_872
    assert model >= int(row["peak_used_bytes"])
    # The Eulerian core has no row: it holds neither the second time level
    # nor the gather, so the semi-Lagrangian rows do not speak for it.
    eulerian = dataclasses.replace(sizing.estimate_global_memory(load_config(T383_EULERIAN_DAY)), backend="cupy")
    assert eulerian.trajectory_bytes == 0
    assert sizing.completed_plan_ceiling(eulerian, 8, ALL) is None


@pytest.mark.parametrize("bands, slices, chunk, expected", [
    (1, ALL, 5_000, 24_940_350_464),           # the one-band day IS one band's figure
    (2, ALL, 5_000, 21_152_070_656),           # the two-band probe is the smaller bound at two
    (4, ALL, 5_000, 21_152_070_656),           # and the lightest applicable row bounds every count above
    (32, ALL, 5_000, 13_838_927_872),          # until a row measured at a higher count is lighter
    (2, ("physics",), 5_000, None),            # a plan that parks less is not bounded
    (2, ALL, 12_500, None),                    # a larger radiation chunk is not bounded by a smaller one's run
    (1, ("physics", "surface"), 5_000, None),  # the surface alone is not every slice
])
def test_a_row_applies_at_fewer_or_equal_bands_a_subset_of_the_slices_and_a_chunk_at_or_above(
        bands, slices, chunk, expected):
    estimate = SimpleNamespace(truncation=533, nlev=40, precision="float32",
                               radiation_column_chunk=chunk, trajectory_bytes=1)
    row = sizing.completed_plan_ceiling(estimate, bands, slices)
    if expected is None:
        assert row is None
    else:
        assert row is not None and int(row["peak_used_bytes"]) == expected


def test_every_row_is_a_completed_run_with_its_plan_and_its_pool_ratio_on_record():
    for row in sizing.COMPLETED_PLAN_PEAKS:
        assert row["core"] in ("semilag", "eulerian")
        assert int(row["bands"]) >= 1 and set(row["spill_slices"]) <= set(sizing.SPILL_SLICES)
        assert int(row["held_bytes"]) >= int(row["peak_used_bytes"]) > 0
        assert row["measured_on"] and row["device"] and "status pass" in row["run"] or "completions" in row["run"]


def test_an_exact_row_is_the_figure_even_when_the_model_reads_under_it():
    cfg = load_config(T533_DAY)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    sizes = sizing._spill_census_estimate(cfg, estimate)
    model = sizing.banded_device_peak_bytes(estimate, 1, spilled_bytes=sum(sizes.values()))
    row = sizing.completed_plan_ceiling(estimate, 1, ALL)
    assert sizing.completed_plan_is_exact(estimate, 1, ALL, row)
    assert model < int(row["peak_used_bytes"]) == 24_940_350_464
    plan = sizing.plan_run_memory(cfg, int(30.9 * GIB), estimate)
    assert plan.live_peak_bytes == 24_940_350_464 and plan.bands == 1


def test_the_sixteen_band_t383_plan_is_priced_at_its_own_rows_not_a_larger_shapes_ratio():
    # MEASURED 2026-09-07 on the RTX 5070 Ti: T383 L40 at sixteen bands with
    # every slice parked completes at 9.73 GiB live and 11.87 held
    # (x1.2201), while the door priced it at 18.30 GiB of card by charging
    # the T533 sixteen-band row's x1.4149 on the two-band day's 12.41 GiB
    # live, and refused rank 1 of a two-card T383 run for it.
    cfg = dataclasses.replace(_at_chunk(T383_DAY, 12_500), latitude_bands=16)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    ratio, phrase = sizing.pool_fragmentation_for(True, True, False, bands=16, truncation=383)
    assert abs(ratio - 12_746_315_776 / 10_446_974_464) < 1e-9
    assert "exactly T383 and 16 bands" in phrase
    larger, _ = sizing.pool_fragmentation_for(True, True, False, bands=16, truncation=533)
    assert larger > 1.4
    row = sizing.completed_plan_ceiling(estimate, 16, ALL)
    assert sizing.completed_plan_is_exact(estimate, 16, ALL, row)
    assert int(row["peak_used_bytes"]) == 10_446_974_464
    plan = sizing.plan_run_memory(cfg, int(15.28 * GIB), estimate)
    assert plan.fits and plan.bands == 16 and plan.spill_slices == ALL
    assert plan.live_peak_bytes == 10_446_974_464
    assert plan.card_bytes <= int(15.28 * GIB)
    assert plan.card_bytes < int(13.0 * GIB)


# --- the 34.7 km day on a 16 GB card that carries a desktop ---------------
#
# MEASURED 2026-11-11 on the RTX 5070 Ti, 24 h days of this shape at the
# 5,000-column chunk: one band with nothing parked took 15,624 MiB of card,
# two bands with every slice parked 12,486 and four bands 10,582, every
# plan byte-identical to the one-band day.  Before those rows the door
# priced both banded plans at 18.31 GiB of card and refused every T383 plan
# below 14.36 GiB free, so a card with a desktop on it could not run T383.
# MEASURED 2026-11-12 with the pool held to a limit: one band 13,880 and
# 14,608 MiB over two days, priced at the heavier, and two bands 10,946
# over 3 h and 11,090 over the day, priced at the day.

MIB = 2 ** 20


def _t383_16gb():
    """The shipped 34.7 km day, which carries the 5,000-column radiation
    chunk every 16 GB plan above was measured at."""

    cfg = load_config(T383_DAY)
    assert cfg.native_adapter_options["radiation_column_chunk"] == 5_000
    return cfg


def _plan_16gb(free_gib, **changes):
    cfg = dataclasses.replace(_t383_16gb(), **changes)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    assert estimate.radiation_column_chunk == 5_000
    return cfg, estimate, sizing.plan_run_memory(cfg, int(free_gib * GIB), estimate)


@pytest.mark.parametrize("free_gib, bands, slices, card_mib", [
    (15.6, 1, (), None),          # the card to itself: the fastest plan
    (14.3, 1, (), None),          # 1.3 GiB taken: one band held to its
                                  # pool limit still fits
    (14.0, 2, ALL, 11_090),       # a desktop's 1.6 GiB taken
    (13.0, 2, ALL, 11_090),       # 2.6 GiB taken
    (12.0, 2, ALL, 11_090),
    (11.0, 2, ALL, 11_090),       # 4.6 GiB taken
    (10.5, 4, ALL, 10_582),
    (10.4, 4, ALL, 10_582),
])
def test_a_card_with_less_free_takes_the_measured_low_memory_plan(free_gib, bands, slices, card_mib):
    _cfg, _estimate, plan = _plan_16gb(free_gib)
    assert plan.fits and plan.bands == bands and plan.spill_slices == slices
    assert plan.bands_chosen_by == "sizer"
    assert plan.card_bytes <= int(free_gib * GIB)
    if card_mib is not None:
        # Priced at what the run of exactly this plan took, not at a ratio
        # borrowed from T533.
        assert plan.card_bytes == card_mib * MIB
        assert "1 band with nothing parked prices at" in plan.reason
        assert "percent slower" in plan.reason


def test_the_two_band_plan_says_what_it_costs():
    _cfg, _estimate, plan = _plan_16gb(13.0)
    assert "2.85 s a step against 2.22 s for one band" in plan.reason
    assert "28 percent slower" in plan.reason


def test_more_free_never_buys_a_heavier_plan():
    previous = None
    for tenth in range(95, 161):
        _cfg, _estimate, plan = _plan_16gb(tenth / 10)
        if not plan.fits:
            assert previous is None, f"refused at {tenth / 10} GiB after a fit below it"
            continue
        key = (plan.bands, len(plan.spill_slices))
        if previous is not None:
            assert key <= previous, (tenth / 10, key, previous)
        previous = key
    assert previous == (1, 0)


def test_below_the_smallest_default_plan_the_refusal_names_the_async_plan():
    cfg = _t383_16gb()
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": int(9.6 * GIB)})
    assert gate["refuse"] is True
    sentence = sizing.refusal_sentence(gate["estimate"], gate["free_bytes"],
                                       config_path=T383_DAY, plan=gate["plan"], cfg=cfg)
    assert "--device-allocator async" in sentence
    assert "4 latitude bands with every slice parked" in sentence
    # Priced at its 24 h day (9,772 MiB), not at its 3 h run (9,196).
    assert "9.54 GiB of card" in sentence


def test_below_every_measured_plan_the_refusal_says_how_much_more_is_needed():
    cfg = _t383_16gb()
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": int(8.0 * GIB)})
    assert gate["refuse"] is True
    sentence = sizing.refusal_sentence(gate["estimate"], gate["free_bytes"],
                                       config_path=T383_DAY, plan=gate["plan"], cfg=cfg)
    assert "The smallest plan measured at this shape" in sentence
    assert "--device-allocator async" in sentence
    assert "1.54 GiB more free" in sentence


@pytest.mark.parametrize("changes, flag", [
    ({"host_spill": "off"}, "--host-spill on"),
    ({"latitude_bands": 1}, "--latitude-bands 2"),
])
def test_a_pinned_resident_plan_that_does_not_fit_is_refused_with_the_plan_that_does(changes, flag):
    cfg = dataclasses.replace(_t383_16gb(), **changes)
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": int(13.0 * GIB)})
    assert gate["refuse"] is True
    assert flag in gate["verdict"]
    sentence = sizing.refusal_sentence(gate["estimate"], gate["free_bytes"],
                                       config_path=T383_DAY, plan=gate["plan"], cfg=cfg)
    assert flag in sentence and "2 latitude bands with every slice parked" in sentence
    assert "28 percent slower" in sentence


def test_the_async_allocator_is_priced_at_its_own_runs():
    _cfg, _estimate, plan = _plan_16gb(9.6, device_allocator="async")
    assert plan.fits and plan.bands == 4 and plan.spill_slices == ALL
    assert plan.card_bytes == 9_772 * MIB
    _cfg, _estimate, plan = _plan_16gb(13.0, device_allocator="async")
    assert plan.fits and plan.bands == 2 and plan.card_bytes == 11_370 * MIB


def test_the_day_at_the_former_chunk_is_pointed_at_the_measured_chunk():
    # A larger radiation chunk is a larger run: the 5,000-column plans
    # bound nothing at 12,500 (the shipped day's chunk before 2026-11-12),
    # and a card the 12,500-column day does not fit is told the one line
    # that fits it.
    cfg = _at_chunk(T383_DAY, 12_500)
    shipped = _plan(cfg, 13.0)
    assert shipped.card_bytes not in (11_090 * MIB, 10_582 * MIB, 10_476 * MIB)
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": int(12.5 * GIB)})
    assert gate["refuse"] is True
    assert "radiation_column_chunk = 5000" in gate["verdict"]
    assert "2 latitude bands with every slice parked" in gate["verdict"]


T255_DAY = str(_shipped_configs() / "arwen_global_gdas_t255_native_sl_si_24h.toml")


def _t255_row():
    rows = [row for row in sizing.COMPLETED_PLAN_PEAKS
            if row["truncation"] == 255 and row["bands"] == 1
            and not row["spill_slices"]]
    assert len(rows) == 1
    return rows[0]


def test_the_t255_day_is_priced_at_its_held_runs_not_at_the_model():
    # MEASURED: 0.1.7's own runs of this plan took 8,438 to 9,374 MiB of the
    # card while the door priced it at 11.09 GiB from its model, so a card
    # with 9 to 11 GiB free sent a run that fits at one band to bands.
    row = _t255_row()
    cfg = load_config(T255_DAY)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    assert estimate.radiation_column_chunk == row["radiation_column_chunk"]
    card = int(row["card_bytes_measured"])
    assert card < int(11.09 * GIB) - GIB
    plan = sizing.plan_run_memory(cfg, card, estimate)
    assert plan.fits and plan.bands == 1 and plan.spill_slices == ()
    assert plan.card_bytes == card
    assert plan.live_peak_bytes == row["peak_used_bytes"]
    assert plan.pool_limit_bytes == row["pool_limit_bytes"]
    # One MiB less and the plan is not one band with nothing parked.
    below = sizing.plan_run_memory(cfg, card - MIB, estimate)
    assert (below.bands, below.spill_slices) != (1, ())
    gate = sizing.run_memory_gate(cfg, probe={"free_bytes": card})
    assert gate["refuse"] is False
    assert (f"priced at the {card / GIB:.2f} GiB of card a completed run of "
            "exactly this plan took" in gate["verdict"])


def test_the_t255_row_is_a_held_run_and_the_heaviest_one():
    row = _t255_row()
    assert row["pool_limit_bytes"] <= row["peak_used_bytes"]
    assert row["pool_limit_end_bytes"] >= row["held_bytes"] >= row["peak_used_bytes"]
    assert row["card_bytes_measured"] > row["held_bytes"]
    assert "status pass" in row["run"] and row["step_s"] > 0


def test_the_shipped_day_on_an_empty_16_gb_card_runs_one_band_with_no_flag():
    # The 16 GB card reads 15,880 MiB free with nothing else on it
    # (tests/test_arwen_global_host_spill.py).  At its former 12,500-column
    # chunk the shipped day was sent to a banded plan there; at 5,000 it
    # takes the full-speed plan.
    plan = _plan(T383_DAY, 15_880 / 1024)
    assert plan.fits and plan.bands == 1 and plan.spill_slices == ()
    before = _plan(_at_chunk(T383_DAY, 12_500), 15_880 / 1024)
    assert before.bands != 1 or before.spill_slices != ()
