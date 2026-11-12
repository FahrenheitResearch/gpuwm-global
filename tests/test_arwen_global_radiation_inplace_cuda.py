"""The radiation's cloud finalize writes over the gas cubes it read, on the device.

Memory only.  The workspace-less radiation path took a second set of
(columns, layers, g-points) cubes for the finalized optics beside the gas
cubes it had just computed, and they were the step's live peak: MEASURED
2026-11-12 at T383 L40 on an RTX 5070 Ti with the 5,000-column chunk, the
shortwave's three finalize cubes (0.68 GiB) stood beside its two gas cubes
(0.45) and the last longwave chunk's Planck sources (0.37), 12.10 GiB live
on a radiation step against 11.71 on every other.  The finalize kernels
are one thread per element and read the gas value into a local before
they write the element, so the gas cubes are their own outputs.

These rows hold that the in-place call returns, bit for bit, what the
call with fresh outputs returns, with and without a subcolumn mask, and
that it took no new cube for the longwave and one (the asymmetry) for the
shortwave.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu

from arwen_global.core import rrtmgp  # noqa: E402


def _inputs(kind, masked, seed):
    tables = rrtmgp.load_gas_tables(kind)
    rng = np.random.default_rng(seed)
    ncol, nlay = 53, 17
    cube = (ncol, nlay, tables.ngpt)
    band = (ncol, nlay, tables.nband)
    dtype = np.float32
    tau_gas = rng.gamma(0.7, 2.0, cube).astype(dtype)
    ssa_gas = rng.uniform(0.0, 1.0, cube).astype(dtype)
    # Clear layers and thick ones, and a g-point with no gas at all, so the
    # floors inside the shortwave kernel are exercised.
    tau_gas[::7, ::3, ::5] = 0.0
    cloud = rrtmgp.CloudOpticsResult(
        cp.asarray((rng.gamma(0.5, 8.0, band)
                    * (rng.uniform(size=band) > 0.4)).astype(dtype)),
        cp.asarray(rng.uniform(0.0, 1.0, band).astype(dtype)),
        cp.asarray(rng.uniform(0.0, 0.95, band).astype(dtype)))
    mask = cp.asarray(rng.uniform(size=cube) > 0.5) if masked else None
    return tables, tau_gas, ssa_gas, cloud, mask


def _gas(kind, tau_gas, ssa_gas):
    return rrtmgp.GasOpticsResult(
        tau=cp.asarray(tau_gas),
        ssa=cp.asarray(ssa_gas) if kind == "sw" else None)


def _bits(array):
    return cp.asnumpy(array).tobytes()


@pytest.mark.parametrize("kind", ["lw", "sw"])
@pytest.mark.parametrize("masked", [True, False])
def test_the_finalize_over_its_own_gas_cubes_is_the_finalize(kind, masked):
    tables, tau_gas, ssa_gas, cloud, mask = _inputs(kind, masked, seed=20261112)
    fresh = rrtmgp._finalize_cloud_optics(
        tables, _gas(kind, tau_gas, ssa_gas), cloud, cloud_mask=mask)
    gas = _gas(kind, tau_gas, ssa_gas)
    out = (gas.tau,) if kind == "lw" else (gas.tau, gas.ssa, None)
    inplace = rrtmgp._finalize_cloud_optics(
        tables, gas, cloud, cloud_mask=mask, out=out)

    assert _bits(inplace.tau) == _bits(fresh.tau)
    # The sum is not the gas value it was read from: the comparison above
    # is of two finalized cubes, not of an untouched input with itself.
    assert _bits(fresh.tau) != tau_gas.tobytes()
    assert inplace.tau.data.ptr == gas.tau.data.ptr
    if kind == "sw":
        assert _bits(inplace.ssa) == _bits(fresh.ssa)
        assert _bits(inplace.g) == _bits(fresh.g)
        assert inplace.ssa.data.ptr == gas.ssa.data.ptr
        # One new cube, the asymmetry, where the call took three.
        assert inplace.g.data.ptr not in (gas.tau.data.ptr, gas.ssa.data.ptr)
