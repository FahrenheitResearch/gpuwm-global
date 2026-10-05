"""The vapor positivity fixer pays a hole from beside it, and says so.

THE BREAKAGE THIS PREVENTS.  Water vapor is the one water species carried
in the spectral basis, so its grid synthesis rings below zero at sharp
gradients, mostly in the dry upper levels.  The fixer clipped the negative
lobes and paid the created water by scaling the column's positive vapor
down in proportion, which takes most of the payment from the levels that
hold most of the vapor -- the moist boundary layer -- and puts it in the
dry lobe aloft: a net upward moisture transfer inside the column every
step, at four clamps a step, that no budget named.  The fix keeps the
column integral exact, pays each hole first from its two adjacent layers
(the column at large pays only what they cannot hold), and carries a
per-level ledger of what the fill added and removed into every step's
metrics, every diagnostics record and the receipt.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.dynamics import MoistHybridModel, sum_fixer_ledgers
from arwen_global.runner import PositivityFixerLedger


class _Recorder:
    def __init__(self):
        self.calls = []

    def add_band(self, *args):
        self.calls.append(args)


def _fill(q, dp):
    """The band filler on numpy, without a model: one band, every row."""
    fake = SimpleNamespace(
        transform=SimpleNamespace(backend=SimpleNamespace(xp=np)),
        _borrow_from_neighbours=MoistHybridModel._borrow_from_neighbours,
    )
    accumulators = {name: _Recorder() for name in (
        "created", "unfillable", "neighbour", "rescale", "levels")}
    filled = MoistHybridModel._fill_column_holes_band(
        fake, q, dp, accumulators, slice(0, q.shape[1]))
    return filled, accumulators


def _column(values):
    return np.asarray(values, np.float64)[:, None, None]


def test_a_dry_lobe_is_paid_by_its_neighbour_not_by_the_boundary_layer():
    # Top first: a negative lobe at level 1 under a dry level 0, over a
    # moist boundary layer.
    q = _column([2.0e-6, -1.0e-6, 4.0e-6, 5.0e-4, 1.0e-2, 1.5e-2])
    dp = np.full_like(q, 1000.0)
    filled, acc = _fill(q, dp)
    assert float(filled.min()) >= 0.0
    # The column integral is exact.
    assert float(np.sum(filled * dp)) == pytest.approx(float(np.sum(q * dp)), rel=1e-14)
    # The lobe is paid by the level above it; the boundary layer and every
    # level the payment did not need are bit-identical.
    np.testing.assert_array_equal(filled[2:], q[2:])
    assert filled[1, 0, 0] == 0.0
    assert filled[0, 0, 0] == pytest.approx(1.0e-6, rel=1e-12)
    # The ledger names where the water went and where it came from.
    (_rows, levels), = acc["levels"].calls
    gain, loss = levels[0][:, 0], levels[1][:, 0]
    assert gain[1] == pytest.approx(1.0e-6 * 1000.0) and loss[0] == pytest.approx(1.0e-6 * 1000.0)
    assert np.all(gain[2:] == 0.0) and np.all(loss[2:] == 0.0)
    (_rows, neighbour), = acc["neighbour"].calls
    assert float(neighbour[0, 0]) == pytest.approx(1.0e-6 * 1000.0)


def test_what_the_neighbours_cannot_hold_the_column_pays_in_proportion():
    q = _column([0.0, -3.0e-5, 1.0e-6, 1.0e-3, 1.0e-2])
    dp = np.full_like(q, 1000.0)
    filled, acc = _fill(q, dp)
    assert float(filled.min()) >= 0.0
    assert float(np.sum(filled * dp)) == pytest.approx(float(np.sum(q * dp)), rel=1e-12)
    # Level 2 gave all it had; the remaining 2.9e-5 came from the rest of
    # the column in proportion.
    assert filled[2, 0, 0] == 0.0
    remaining = 2.9e-5
    positive = 1.0e-3 + 1.0e-2
    np.testing.assert_allclose(
        filled[3:, 0, 0], q[3:, 0, 0] * (positive - remaining) / positive, rtol=1e-12)
    (rescale,), = acc["rescale"].calls
    assert float(rescale[0, 0]) == pytest.approx(3.0e-5 / (1.0e-6 + positive), rel=1e-12)


def test_a_net_negative_column_keeps_its_plain_clip():
    q = _column([-1.0e-4, -1.0e-4, -1.0e-4])
    dp = np.full_like(q, 1000.0)
    filled, acc = _fill(q, dp)
    np.testing.assert_array_equal(filled, np.zeros_like(q))
    (_rows, unfillable), = acc["unfillable"].calls
    assert float(unfillable[0, 0]) == pytest.approx(-3.0e-4 * 1000.0)


def test_the_repair_carries_the_ledger(tmp_path):
    from test_arwen_global_grid_tracers import _t21_model

    _cfg, model, state = _t21_model(tmp_path, qv_surface=0.006)
    transform = model.transform
    nlat, nlon = transform.grid.shape
    g = model.grid_state(state.atmosphere, only=("qv", "dp"))
    qv = np.array(g["qv"], np.float64)
    qv[2, nlat // 2, nlon // 2] = -8.0 * qv[2, nlat // 2, nlon // 2]
    state.atmosphere.qv = transform.project(transform.forward(qv))
    _repaired, _negative, _tracer, fixer = model._repair_positivity(state)
    ledger = fixer["ledger"]
    assert len(ledger["level_gain_kg_m2"]) == model.nlev
    assert len(ledger["level_loss_kg_m2"]) == model.nlev
    gain = sum(ledger["level_gain_kg_m2"])
    loss = sum(ledger["level_loss_kg_m2"])
    assert gain > 0.0
    # Fillable columns close: what the fill took is what it added.
    assert fixer["unfillable_kg_m2"] == 0.0
    assert loss == pytest.approx(gain, rel=1e-9)
    assert gain == pytest.approx(fixer["water_kg_m2"], rel=1e-9)
    assert 0.0 < ledger["neighbour_kg_m2"] <= gain * (1 + 1e-9)


def test_the_run_ledger_names_the_vertical_transfer():
    ledger = PositivityFixerLedger()
    step = {
        "positivity_fixer_water_kg_m2": 2.0,
        "positivity_fixer_unfillable_kg_m2": 0.0,
        "positivity_fixer_neighbour_kg_m2": 0.5,
        # Top first: the fill added 2 at the top and took 0.5 and 1.5 below.
        "positivity_fixer_level_gain_kg_m2": [2.0, 0.0, 0.0],
        "positivity_fixer_level_loss_kg_m2": [0.0, 0.5, 1.5],
    }
    ledger.add(step)
    ledger.add(step)
    record = ledger.record()
    assert record["steps"] == 2
    assert record["clip_created_kg_m2"] == 4.0
    assert record["paid_by_adjacent_layer_kg_m2"] == 1.0
    assert record["level_net_kg_m2"] == [4.0, -1.0, -3.0]
    # Up across the interface below the top: 4; below level 1: 3; the
    # bottom interface closes at zero.
    assert record["upward_transfer_kg_m2"] == [4.0, 3.0, 0.0]
    assert record["maximum_upward_transfer_kg_m2"] == 4.0
    assert record["vertically_moved_kg_m2"] == 4.0


def test_the_four_clamps_sum_and_the_receipt_and_diagnostics_carry_it():
    a = {"level_gain_kg_m2": [1.0, 0.0], "level_loss_kg_m2": [0.0, 1.0], "neighbour_kg_m2": 1.0}
    assert sum_fixer_ledgers(None, a, a) == {
        "level_gain_kg_m2": [2.0, 0.0], "level_loss_kg_m2": [0.0, 2.0],
        "neighbour_kg_m2": 2.0}
    assert sum_fixer_ledgers(None, None) is None
    from arwen_global import runner

    source = inspect.getsource(runner)
    assert source.count('"positivity_fixer_ledger": fixer_ledger.record()') == 2
    assert 'diag["positivity_fixer_ledger"] = fixer_ledger.record()' in source
