"""T799 and T1023 on the 96 GB card: admitted at the plans a run of exactly
that plan measured, refused everywhere nothing has weighed them.

The breakage these hold: 0.2.4's door refused every truncation above T383
("above the fitted domain") and its plan refusal priced T1023 at 99.94 GiB
live and 129.46 GiB of card at one band, so a 96 GB card could only be
walked past by environment variable.  MEASURED 2026-11-15 on an RTX PRO
6000 Blackwell (97,887 MiB): the one-band T1023 run on the hires/memory
tree with semilag.gather_batch = 1 and the one-band T799 run both ran their
ten steps (sizing.COMPLETED_PLAN_PEAKS).  The rows lift the refusal for
those exact plans only; a card that cannot hold the measured figure, or a
config at a gather batch above the measured one, is refused as before.
"""
from __future__ import annotations

import dataclasses

from arwen_global import sizing
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs


GIB = 2 ** 30
T799_WALL = str(_shipped_configs() / "arwen_global_gdas_t799_native_sl_si_wall.toml")
T383_DAY = str(_shipped_configs() / "arwen_global_gdas_t383_native_sl_si_24h.toml")
#: The door's free reading of the empty RTX PRO 6000 (run log, 2026-11-15).
FREE_96 = int(94.37 * GIB)


def _t1023(gather_batch: int = 1):
    cfg = load_config(T799_WALL)
    return dataclasses.replace(
        cfg, truncation=1023,
        semilag=dataclasses.replace(cfg.semilag, gather_batch=int(gather_batch)))


def _gate(cfg, free_bytes):
    return sizing.run_memory_gate(cfg, probe={"free_bytes": int(free_bytes)},
                                  neighbours={})


def _row(truncation):
    rows = [row for row in sizing.COMPLETED_PLAN_PEAKS
            if int(row["truncation"]) == truncation]
    assert len(rows) == 1
    return rows[0]


def test_t1023_is_admitted_at_one_band_on_the_96_gb_card_at_its_measured_plan():
    gate = _gate(_t1023(), FREE_96)
    row = _row(1023)
    assert not gate["refuse"] and gate["unfitted"] is None
    plan = gate["plan"]
    assert plan.bands == 1 and plan.spill_slices == ()
    assert plan.live_peak_bytes == int(row["peak_used_bytes"])
    # The measured card figure plus the kept cuFFT plans (speed/fft-plans),
    # which the row's run (2026-11-15) did not keep.
    assert plan.card_bytes == (int(row["card_bytes_measured"])
                               + sizing.fft_plan_cache_bytes(int(row["truncation"]))) <= FREE_96
    assert "completed run of exactly this plan" in gate["verdict"]


def test_the_shipped_t1023_day_carries_the_gather_batch_the_door_admits():
    # The T1023 config carries semilag.gather_batch = 1 (no bit changes: the
    # batch only groups the same gathers); at the shipped batch of 8 the door
    # refuses this grid on the 96 GB card (test below).
    cfg = load_config(str(_shipped_configs() / "arwen_global_gdas_t1023_native_sl_si_24h.toml"))
    assert cfg.truncation == 1023 and cfg.semilag.gather_batch == 1
    assert cfg.semilag.extrapolation == "settls_departure"
    assert cfg.semilag.trajectory_solver == "newton_ls"
    gate = _gate(cfg, FREE_96)
    assert not gate["refuse"] and gate["unfitted"] is None
    assert gate["plan"].bands == 1 and gate["plan"].spill_slices == ()


def test_t799_is_admitted_at_one_band_on_the_96_gb_card_at_its_measured_plan():
    gate = _gate(load_config(T799_WALL), FREE_96)
    row = _row(799)
    assert not gate["refuse"] and gate["unfitted"] is None
    plan = gate["plan"]
    assert plan.bands == 1 and plan.spill_slices == ()
    # The measured card figure plus the kept cuFFT plans (speed/fft-plans),
    # which the row's run (2026-11-15) did not keep.
    assert plan.card_bytes == (int(row["card_bytes_measured"])
                               + sizing.fft_plan_cache_bytes(int(row["truncation"]))) <= FREE_96


def test_t1023_at_the_shipped_gather_batch_is_not_priced_by_the_batch_one_run():
    # Eight stacked fields are the 5,760 MiB blocks that ran the one-band
    # pool out of the card; the batch-one run says nothing about them.
    gate = _gate(_t1023(gather_batch=8), FREE_96)
    assert gate["refuse"] and gate["unfitted"]
    # ... and the refusal names the measured batch that is the way past it.
    assert "gather_batch = 1" in gate["unfitted"]
    # Where even that plan's card figure does not fit, it is not offered.
    small = _gate(_t1023(gather_batch=8), int(30.9 * GIB))
    assert small["refuse"] and "gather_batch = 1" not in str(small["unfitted"])


def test_t1023_and_t799_are_still_refused_where_nothing_has_weighed_them():
    for cfg in (_t1023(), load_config(T799_WALL)):
        gate = _gate(cfg, int(30.9 * GIB))
        assert gate["refuse"]


def test_the_rows_are_one_band_nothing_parked_rows_of_the_96_gb_card():
    for truncation in (799, 1023):
        row = _row(truncation)
        assert row["core"] == "semilag" and row["bands"] == 1
        assert row["spill_slices"] == ()
        assert int(row["held_bytes"]) >= int(row["peak_used_bytes"])
        assert int(row["card_bytes_measured"]) >= int(row["held_bytes"])
        assert "RTX PRO 6000" in row["device"]


def test_the_16_gb_t383_plan_is_unchanged():
    cfg = load_config(T383_DAY)
    estimate = dataclasses.replace(sizing.estimate_global_memory(cfg), backend="cupy")
    plan = sizing.plan_run_memory(cfg, int(15.6 * GIB), estimate)
    assert plan.fits and plan.bands == 1 and plan.spill_slices == ()
    assert plan.card_bytes == 14_608 * 1024 * 1024
