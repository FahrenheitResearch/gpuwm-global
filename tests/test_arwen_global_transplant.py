"""The scheme transplant between exact twins (arwen_global.transplant).

Every claim is on the real shipped step with the native suite behind the
level-5 fakes on the numpy backend (the T21 baroclinic configuration),
and every one is a byte-for-byte equality of every checkpointed array:

TX-1  self-substitution: a twin receiving its identical twin's cumulus
      and microphysics output equals the twin, and the donor's recording
      moves nothing in the donor
TX-2  known answer, cumulus: a run whose own scheme heats nothing,
      receiving the output of a twin whose scheme heats, equals a run
      whose own scheme heats -- and the receiver's scheme never runs
TX-3  known answer, microphysics: the same with the Morrison slot
TX-4  the copy-where-equal rule where float32 subtraction rounds
TX-5  refusals: a receiver ahead of its donor, a model receiving from
      itself, an unknown stage, a model hooked twice
TX-6  detached, a model runs the suite as before
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arwen_global import transplant as tx
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.physics.native_suite import ArwenCudaColumnSuite
from arwen_global.runner import build_model_and_cold_state
from test_arwen_global_level5_native import _fake_modules, _options

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")
STEPS = 3


def _modules(*, heating=0.0, warming=0.0):
    """Column-local fakes whose output depends on the column.

    The cumulus fake heats every level at ``heating`` K/s scaled by the
    column's lowest-level temperature and moistens nothing; the Morrison
    fake warms the lowest level by ``warming`` K and turns a fixed
    fraction of the cloud water into rain water, so a microphysics record
    carries changes too large for ``entry + (exit - entry)`` to round
    back to ``exit``."""
    def cumulus(result, atmosphere):
        t = np.asarray(atmosphere["temperature"], np.float32)
        result.rthcuten[...] = np.float32(heating) * (t[:1] / np.float32(250.0))
        result.rainc[...] = np.float32(heating * 10.0)

    modules = _fake_modules(cumulus=cumulus)

    def launch_morrison(theta, qv, qc, qr, *args, **kwargs):
        theta[0] += np.float32(warming)
        moved = qc * np.float32(0.37)
        qc -= moved
        qr += moved

    modules["arwen_global.core.morrison"].launch_morrison = launch_morrison
    return modules


def _model(**fakes):
    cfg = replace(load_config(CONFIG), host_spill="off")
    model, state = build_model_and_cold_state(cfg)
    model.physics = ArwenCudaColumnSuite(_options(), array_module=np, modules=_modules(**fakes))
    return cfg, model, state


def _step(cfg, pairs, steps=STEPS):
    """Step the (model, state) pairs in the order given, in lockstep."""
    states = [s for _, s in pairs]
    for _ in range(steps):
        for k, (model, _) in enumerate(pairs):
            states[k], _ = model.step(states[k], cfg.dt_s)
    return states


def _same(model_a, a, model_b, b):
    da, db = tx.array_digests(model_a, a), tx.array_digests(model_b, b)
    assert sorted(da) == sorted(db)
    return [name for name in da if da[name] != db[name]]


# ------------------------------------------------------------------ TX-1


def test_tx_1_self_substitution_is_bit_exact():
    cfg, plain, s0 = _model(heating=2.0e-4, warming=0.125)
    _, a, sa = _model(heating=2.0e-4, warming=0.125)
    _, b, sb = _model(heating=2.0e-4, warming=0.125)
    donor = tx.Donor(a, tx.STAGES, name="A")
    receiver = tx.Receiver(b, donor, name="B")
    plain_end, a_end, b_end = _step(cfg, [(plain, s0), (a, sa), (b, sb)])
    assert _same(a, a_end, b, b_end) == []
    # Recording moves nothing in the donor.
    assert _same(plain, plain_end, a, a_end) == []
    calls = donor.recorded["cumulus"]
    assert calls == donor.recorded["microphysics"] == 2 * STEPS
    assert receiver.ledger["cumulus"].calls == calls
    assert receiver.ledger["microphysics"].calls == calls
    # Identical entries everywhere: every microphysics cell was copied.
    assert receiver.ledger["microphysics"].incremented_cells == 0
    assert receiver.ledger["microphysics"].copied_cells > 0
    # The receiver's own cumulus scheme was never built.
    assert b.physics._runtime._cumulus is None
    assert not receiver.queue


# ------------------------------------------------------------------ TX-2


def test_tx_2_cumulus_known_answer():
    """A cold scheme receiving a heating scheme's output is the heating run."""
    cfg, heating, sh = _model(heating=2.0e-4)
    _, reference, sr = _model(heating=2.0e-4)
    _, cold, sc = _model(heating=0.0)
    _, cold_plain, sp = _model(heating=0.0)
    donor = tx.Donor(heating, ("cumulus",), name="heating")
    tx.Receiver(cold, donor, name="cold")
    _, ref_end, cold_end, plain_end = _step(
        cfg, [(heating, sh), (reference, sr), (cold, sc), (cold_plain, sp)])
    assert _same(reference, ref_end, cold, cold_end) == []
    assert _same(cold_plain, plain_end, cold, cold_end) != []
    assert cold.physics._runtime._cumulus is None


# ------------------------------------------------------------------ TX-3


def test_tx_3_microphysics_known_answer():
    cfg, warm, sw = _model(warming=0.25)
    _, reference, sr = _model(warming=0.25)
    _, cold, sc = _model(warming=0.0)
    _, cold_plain, sp = _model(warming=0.0)
    donor = tx.Donor(warm, ("microphysics",), name="warm")
    receiver = tx.Receiver(cold, donor, name="cold")
    _, ref_end, cold_end, plain_end = _step(
        cfg, [(warm, sw), (reference, sr), (cold, sc), (cold_plain, sp)])
    assert _same(reference, ref_end, cold, cold_end) == []
    assert _same(cold_plain, plain_end, cold, cold_end) != []
    assert receiver.ledger["microphysics"].calls == 2 * STEPS


# ------------------------------------------------------------------ TX-4


def test_tx_4_copy_where_equal():
    entry = np.array([1.0, 1.0, 0.5, -0.0], np.float32)
    exit_ = np.array([3.0e-8, 2.0, 0.25, 0.0], np.float32)
    # float32 rounds this one: the copy is what keeps a twin identical.
    assert entry[0] + (exit_[0] - entry[0]) != exit_[0]
    receiver = np.array([1.0, 1.5, 0.5, 0.0], np.float32)
    value, equal = tx.transplanted_value(np, receiver, entry, exit_)
    assert value.dtype == np.float32
    assert equal.tolist() == [True, False, True, False]  # -0.0 and 0.0 differ in bits
    assert value[0] == exit_[0]
    assert value[1] == np.float32(1.5) + (exit_[1] - entry[1])
    assert value[2] == exit_[2]
    assert value[3] == np.float32(0.0) + (exit_[3] - entry[3])


# ------------------------------------------------------------------ TX-5


def test_tx_5_refusals():
    cfg, a, sa = _model(heating=1.0e-4)
    _, b, sb = _model(heating=1.0e-4)
    donor = tx.Donor(a, ("cumulus",), name="A")
    tx.Receiver(b, donor, name="B")
    with pytest.raises(RuntimeError, match="must step first"):
        b.step(sb, cfg.dt_s)
    with pytest.raises(ValueError, match="itself"):
        tx.Receiver(a, donor)
    with pytest.raises(ValueError, match="stages"):
        tx.Donor(b, ("radiation",))
    with pytest.raises(ValueError, match="already carries"):
        tx.Receiver(b, donor)
    _, c, _ = _model()
    with pytest.raises(ValueError, match="does not record"):
        tx.Receiver(c, donor, ("microphysics",))


# ------------------------------------------------------------------ TX-6


def test_tx_6_detached_runs_as_before():
    cfg, a, sa = _model(heating=2.0e-4, warming=0.125)
    _, b, sb = _model(heating=0.0)
    _, c, sc = _model(heating=0.0)
    donor = tx.Donor(a, tx.STAGES)
    receiver = tx.Receiver(b, donor)
    receiver.detach()
    donor.detach()
    _, b_end, c_end = _step(cfg, [(a, sa), (b, sb), (c, sc)])
    assert _same(b, b_end, c, c_end) == []
    assert "_cumulus_result" not in b.physics._runtime.__dict__


# ------------------------------------------------------------------ TX-7


def test_tx_7_the_four_run_driver_end_to_end():
    """The experiment driver on the fakes: the twins start from the same
    bits, the self-substitution gate passes at every report hour, and the
    table's numbers are what its definitions say."""
    cfg = replace(load_config(CONFIG), host_spill="off")
    hours = 3 * cfg.dt_s / 3600.0
    plan = tx.ExperimentPlan(
        config=CONFIG, hours=hours, report_hours=(2 * cfg.dt_s / 3600.0, hours),
        stages=("cumulus",),
    )
    report = tx.run_experiment(
        plan, cfg=cfg,
        physics=lambda: ArwenCudaColumnSuite(
            _options(), array_module=np, modules=_modules(heating=2.0e-4, warming=0.125)),
    )
    assert report["start_identity"] == {
        "Cs_equals_C": True, "Cc_equals_C": True, "Ps_equals_P": True}
    gate = report["self_substitution_gate"]
    assert gate["pass"] and len(gate["rows"]) == 2
    assert all(r["arrays"] > 100 for r in gate["rows"])
    assert report["donor_records"] == {"C": {"cumulus": 6}, "P": {"cumulus": 6}}
    assert report["ledgers"]["Ps"]["cumulus"]["calls"] == 6
    row = report["rows"][-1]
    e = row["P_minus_C"]["energy_j_m2"]
    assert e > 0.0
    assert row["share_removed"] == 1.0 - row["Ps_minus_C"]["energy_j_m2"] / e
    assert row["share_alone"] == row["Cs_minus_C"]["energy_j_m2"] / e
    # The cumulus fake's heating depends on the column, so taking C's
    # output changes P's run, and P's output changes C's.
    assert row["P_minus_Ps"]["energy_j_m2"] > 0.0
    assert row["Cs_minus_C"]["energy_j_m2"] > 0.0
