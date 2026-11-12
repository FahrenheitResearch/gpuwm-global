"""Both RTE solvers give the same bytes cut into column spans as in one
launch set, on the card.

The spans exist to move the solvers onto their folding path (see
``rrtmgp._rte_column_spans``); this holds the claim that the move changes no
bit, over the column counts where the tile, the path or the ragged tail
changes: one column, either side of a span, the shipped chunks, and widths
forced to odd values.  Inputs are synthetic but physical in range: positive
optical depths, single-scattering albedos below one, asymmetry in [0, 1),
sources and fluxes of the size a column carries.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import rrtmgp  # noqa: E402

NLAY = 54


def _lw_inputs(rng, ncol, ngpt=256):
    shape = (ncol, NLAY, ngpt)
    tau = rng.gamma(0.6, 0.4, size=shape).astype(np.float32)
    lay = rng.uniform(0.01, 2.0, size=shape).astype(np.float32)
    lev = rng.uniform(0.01, 2.0, size=(ncol, NLAY + 1, ngpt)).astype(np.float32)
    sfc = rng.uniform(0.5, 3.0, size=(ncol, ngpt)).astype(np.float32)
    emis = rng.uniform(0.85, 1.0, size=(ncol, ngpt)).astype(np.float32)
    return tuple(cp.asarray(a) for a in (tau, lay, lev, sfc, emis))


def _sw_inputs(rng, ncol, ngpt=224):
    shape = (ncol, NLAY, ngpt)
    tau = rng.gamma(0.6, 0.4, size=shape).astype(np.float32)
    ssa = rng.uniform(0.0, 0.999, size=shape).astype(np.float32)
    g = rng.uniform(0.0, 0.9, size=shape).astype(np.float32)
    mu = rng.uniform(0.02, 1.0, size=(ncol,)).astype(np.float32)
    alb = rng.uniform(0.05, 0.9, size=(ncol, ngpt)).astype(np.float32)
    inc = rng.uniform(0.0, 20.0, size=(ncol, ngpt)).astype(np.float32)
    return tuple(cp.asarray(a) for a in (tau, ssa, g, mu, alb, inc))


def _same_bytes(a, b):
    return cp.asnumpy(a).tobytes() == cp.asnumpy(b).tobytes()


CASES = [(1, None), (2047, None), (2048, None), (2049, None),
         (5_000, None), (12_500, None), (2049, 1000), (777, 7), (300, 1)]


@pytest.mark.parametrize("ncol, span", CASES)
def test_longwave_spans_give_the_one_launch_bytes(ncol, span):
    rng = np.random.default_rng(ncol)
    tau, lay, lev, sfc, emis = _lw_inputs(rng, ncol)
    one = rrtmgp._lw_rte(tau, lay, lev, sfc, emis, top_at_1=False,
                         out=None, incident_out=None, column_span=ncol)
    cut = rrtmgp._lw_rte(tau, lay, lev, sfc, emis, top_at_1=False,
                         out=None, incident_out=None, column_span=span)
    assert _same_bytes(one.flux_up, cut.flux_up)
    assert _same_bytes(one.flux_dn, cut.flux_dn)


@pytest.mark.parametrize("ncol, span", CASES)
def test_shortwave_spans_give_the_one_launch_bytes(ncol, span):
    rng = np.random.default_rng(10_000 + ncol)
    tau, ssa, g, mu, alb, inc = _sw_inputs(rng, ncol)
    one = rrtmgp._sw_rte(tau, ssa, g, mu, alb, alb, inc, top_at_1=False,
                         out=None, column_span=ncol)
    cut = rrtmgp._sw_rte(tau, ssa, g, mu, alb, alb, inc, top_at_1=False,
                         out=None, column_span=span)
    for name in ("flux_up", "flux_dn", "flux_dir"):
        assert _same_bytes(getattr(one, name), getattr(cut, name)), name


def test_a_column_solved_among_others_gives_its_own_bytes():
    # The shortwave loop now hands the solver the sunlit columns only, so a
    # column's neighbours in a launch change; its bytes must not.
    rng = np.random.default_rng(7)
    tau, ssa, g, mu, alb, inc = _sw_inputs(rng, 4_000)
    whole = rrtmgp._sw_rte(tau, ssa, g, mu, alb, alb, inc, top_at_1=False,
                           out=None)
    pick = cp.asarray(np.flatnonzero(rng.uniform(size=4_000) < 0.5))
    part = rrtmgp._sw_rte(tau[pick], ssa[pick], g[pick], mu[pick], alb[pick],
                          alb[pick], inc[pick], top_at_1=False, out=None)
    assert _same_bytes(whole.flux_up[pick], part.flux_up)
    assert _same_bytes(whole.flux_dn[pick], part.flux_dn)
