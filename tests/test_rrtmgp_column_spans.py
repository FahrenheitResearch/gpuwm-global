"""The radiation driver's two same-bits savings, on the CPU: how a solver
call is cut into column spans, and which columns the shortwave loop runs.

Both are pure functions of a column count and a daylight mask, so they are
held here exactly, without a card.  The byte equality of the solvers across
spans is a device fact and is held in ``test_rrtmgp_column_spans_cuda.py``;
the byte equality of whole forecasts is a run on the card (CHANGELOG).
"""

from __future__ import annotations

import numpy as np
import pytest

from arwen_global.core import rrtmgp


@pytest.mark.parametrize("ncol, span", [
    (1, 2240), (2239, 2240), (2240, 2240), (2241, 2240), (12_500, 2240),
    (294_912, 2240), (663_552, 4096), (5_000, 1), (7, 3), (12_500, 12_500),
    (12_500, 50_000)])
def test_spans_cover_every_column_once_in_order_and_are_balanced(ncol, span):
    spans = rrtmgp._rte_column_spans(ncol, span)
    assert spans[0][0] == 0 and spans[-1][1] == ncol
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0
    widths = [c1 - c0 for c0, c1 in spans]
    assert max(widths) <= span
    assert max(widths) - min(widths) <= 1
    assert len(spans) == -(-ncol // span)


def test_a_span_at_or_above_the_column_count_is_the_one_launch_set():
    assert rrtmgp._rte_column_spans(12_500, 12_500) == ((0, 12_500),)
    assert rrtmgp._rte_column_spans(5_000, 10**9) == ((0, 5_000),)


@pytest.mark.parametrize("sms", [70, 128])   # RTX 5070 Ti, RTX 4090
def test_the_default_span_lands_both_solvers_on_the_folding_tile(
        monkeypatch, sms):
    monkeypatch.setattr(rrtmgp, "_rte_sm_count", lambda: sms)
    monkeypatch.delenv("GPUWM_RTE_COLUMN_SPAN", raising=False)
    for ncol in (294_912, 663_552, 12_500, 5_000):
        for c0, c1 in rrtmgp._rte_column_spans(ncol):
            for ngpt in (256, 224):
                assert rrtmgp._rte_gpt_tile(
                    None, c1 - c0, ngpt, fold=True) == 32, (ncol, c1 - c0)
    # The shipped chunks on their own, as one launch, did not.
    assert rrtmgp._rte_gpt_tile(None, 12_500, 256, fold=True) < 32
    assert rrtmgp._rte_gpt_tile(None, 5_000, 224, fold=True) < 32


def test_the_measurement_override_sets_the_span(monkeypatch):
    monkeypatch.setenv("GPUWM_RTE_COLUMN_SPAN", "1000")
    assert len(rrtmgp._rte_column_spans(12_500)) == 13


def _daylight(ny=6, nx=8):
    # Half of every row lit, as on a Gaussian grid at an equinox, plus a
    # lit polar row and a dark one.
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:, : nx // 2] = True
    mask[0, :] = True
    mask[-1, :] = False
    return mask


def test_the_shortwave_runs_the_sunlit_columns_in_ascending_order():
    mask = _daylight()
    chunks, lit_only = rrtmgp._shortwave_column_chunks(mask, 5, xp=np)
    assert lit_only
    joined = np.concatenate(chunks)
    np.testing.assert_array_equal(joined, np.flatnonzero(mask.reshape(-1)))
    assert [c.size for c in chunks][:-1] == [5] * (len(chunks) - 1)
    assert 1 <= chunks[-1].size <= 5


def test_an_all_lit_call_keeps_the_slices_it_always_had():
    mask = np.ones((4, 6), dtype=bool)
    chunks, lit_only = rrtmgp._shortwave_column_chunks(mask, 10, xp=np)
    assert not lit_only
    assert chunks == rrtmgp._column_slices(24, 10)


def test_an_all_dark_call_runs_no_shortwave_chunk():
    mask = np.zeros((4, 6), dtype=bool)
    chunks, lit_only = rrtmgp._shortwave_column_chunks(mask, 10, xp=np)
    assert lit_only and chunks == ()


def test_the_override_puts_the_dark_columns_back(monkeypatch):
    monkeypatch.setenv(rrtmgp.DARK_COLUMNS_ENV, "solve")
    chunks, lit_only = rrtmgp._shortwave_column_chunks(_daylight(), 5, xp=np)
    assert not lit_only
    assert chunks == rrtmgp._column_slices(48, 5)


def test_absent_species_answer_an_index_list_with_zeros():
    zeros = rrtmgp._ZeroColumns((48, 3), dtype=np.float32, xp=np)
    out = zeros[np.array([2, 5, 7])]
    assert out.shape == (3, 3) and out.dtype == np.float32 and not out.any()
    assert zeros[slice(4, 9)].shape == (5, 3)
