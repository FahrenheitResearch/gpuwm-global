"""Grell-Freitas reports the rain its own tendencies take out of the column.

The deep arm's output routine (gf.cu gfd_cup_output_ens_3d, WRF 4.7.1
``module_cu_gf_deep.F:3318-3348``) takes the downdraft's evaporation from
the detrained cloud water first and from the rain for the remainder, and
accumulates ``pre -= xmb * remainder``.  WRF then closes with
``PRE = -PRE + XMB*pwtot``, which flips that sign: the reported rain is
``xmb * (pwtot + remainder)``, while the applied rqvcuten/rqccuten remove
only ``xmb * (pwtot - remainder)``.  This package closes with
``pre + xmb*pwtot`` (row GF-CU-10, audit GP-1).

The breakage this prevents, named: the land surface is forced with RAINCV,
so every convecting land column received water the atmosphere never lost,
and the surface reservoir paid for it.  On the WRF oracle capture the
reported rain was 1.27 times the water removed over the 84 raining columns;
on the T255 GDAS forecast the excess was 1.8 times globally and 2.3 times
over land.  The tolerance below is the metric gap only: the kernel takes its
layer masses from its own interpolated cup pressures and this check takes
them from the model interfaces, which leaves 0.35 percent overall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import gf  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools" / "gf_wrf461_oracle"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

GRAVITY = 9.81


def _launch(module, fixture):
    from gf_field_lists import (DRV_IN_LEV, DRV_IN_SCA, DRV_ISCA_FIELDS,
                                DRV_LEV_FIELDS, DRV_SCA_FIELDS)
    from gpuwm.verify.gf_oracle import GF_NZ

    gl, gs, n = fixture.levels, fixture.surface, fixture.ncol
    lvin = np.zeros((n, len(DRV_IN_LEV), GF_NZ), dtype=np.float32)
    scin = np.zeros((n, len(DRV_IN_SCA)), dtype=np.float32)
    iin = np.zeros((n, 3), dtype=np.int32)
    for j, name in enumerate(DRV_IN_LEV):
        lvin[:, j, :] = gl[name]
    for j, name in enumerate(DRV_IN_SCA):
        scin[:, j] = gs[name].astype(np.float32)
    iin[:, 0] = gs["kpbl"].astype(np.int32)
    iin[:, 1] = gs["ishallow"].astype(np.int32)
    iin[:, 2] = gs["ichoice"].astype(np.int32)
    d = [cp.asarray(np.ascontiguousarray(a)) for a in (lvin, scin, iin)]
    d_lev = cp.zeros((n, len(DRV_LEV_FIELDS), GF_NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    d_ws = cp.empty(gf.gf_workspace_floats(GF_NZ, n), dtype=cp.float32)
    module.get_function("gf_gfdrv_stage")(
        ((n + 63) // 64,), (64,),
        (*d, d_lev, d_sca, d_isc, d_ws, np.int32(1), np.int32(n),
         np.int32(GF_NZ)))
    cp.cuda.Stream.null.synchronize()
    lev, sca = cp.asnumpy(d_lev), cp.asnumpy(d_sca)
    levels = {name: lev[:, j, :].astype(np.float64)
              for j, name in enumerate(DRV_LEV_FIELDS)}
    scalars = {name: sca[:, j].astype(np.float64)
               for j, name in enumerate(DRV_SCA_FIELDS)}
    return levels, scalars


@pytest.fixture(scope="module")
def fixture():
    pytest.importorskip("gpuwm.verify.gf_oracle")
    from gpuwm.verify.gf_oracle import load_gf_oracle

    return load_gf_oracle()


def _budget(fixture, module):
    levels, scalars = _launch(module, fixture)
    p8w = fixture.levels["p8w"].astype(np.float64)
    dp = np.zeros_like(p8w)
    dp[:, :-1] = p8w[:, :-1] - p8w[:, 1:]
    water = levels["rqvcuten"] + levels["rqccuten"] + levels["rqicuten"]
    # the top level carries no cumulus tendency, so its missing upper
    # interface costs nothing
    assert np.all(water[:, -1] == 0.0)
    dt = fixture.surface["dt"].astype(np.float64)
    removed = -(water * dp).sum(axis=1) / GRAVITY * dt
    return scalars["raincv"], removed


@pytest.mark.gpu
@pytest.mark.parametrize("coarse", [False, True], ids=["wrf-kernel", "coarse-column"])
def test_the_reported_rain_is_the_water_the_tendencies_remove(fixture, coarse):
    module = gf._gf_module(40, resolved_convergence_closure=coarse)
    rain, removed = _budget(fixture, module)
    raining = rain > 0.0
    assert int(raining.sum()) >= 50
    total = rain[raining].sum() / removed[raining].sum()
    assert total == pytest.approx(1.0, abs=0.01), total
    per_column = rain[raining] / removed[raining]
    assert per_column.max() < 1.06, per_column.max()
    assert per_column.min() > 0.99, per_column.min()
