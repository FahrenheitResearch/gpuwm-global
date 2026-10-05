"""YSU's thermal-enhanced Richardson sweep runs only where pblflg is set.

WRF ``bl_ysu.F90:703-728`` guards the thermal-enhanced re-diagnosis with
``if(pblflg(i))`` at :704, :711 and :720, and :684-698 can only LOWER the
flag.  A column whose first PBL-top guess sat below ``zq(i,2)`` is therefore
held in the local-K regime for the whole step.  The carried kernel and its
float64 mirror ran that sweep unguarded and recomputed the flag from its
result, so the thermal excess could promote such a column into full
non-local YSU (audit GP-4, engine fix 4d523b793, rows YSU-CU-1 and
NPREF-6).

The breakage this prevents, named: at convective onset under a residual
nocturnal cap, a column WRF keeps shallow (kpbl 1, hpbl 95.4 m, background
exchange 0.01 m2/s) came back kpbl 4, hpbl 431 m with countergradient
transport and an entrainment-boosted exchange coefficient of 22 m2/s, which
is a different boundary-layer onset time on exactly the columns that set
2 m temperature and dewpoint.  The shipped oracle has no column with
kpbl == 1, so no parity gate reaches this; the column below is the engine's
own measured case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arwen_global.core.npref import np_ysu_column

_KERNEL = (Path(__file__).resolve().parents[1] / "src" / "arwen_global"
           / "core" / "kernels" / "ysu.cu")


def _grid(nz=36, ztop=6000.0):
    z_ifc = np.linspace(0.0, ztop, nz + 1)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    p_ifc = 100000.0 * np.exp(-z_ifc / 8200.0)
    p = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    exner = (p / 100000.0) ** (287.0 / 1004.5)
    return z, dz, p, p_ifc, exner


def _onset_column():
    """Weakly negative br, a small positive hfx, a residual cap on the
    first two levels: the first guess puts the PBL top at 95.4 m, below
    zq(i,2) = 166.7 m."""
    nz = 36
    z, dz, p, p_ifc, exner = _grid(nz)
    theta = np.empty(nz)
    theta[0] = 300.0
    for k in range(1, nz):
        theta[k] = theta[k - 1] + (0.1 if k <= 2 else 3.0)
    u = np.full(nz, 8.0)
    v = np.zeros(nz)
    zero = np.zeros(nz)
    return dict(
        u=u, v=v, theta=theta, qv=np.full(nz, 0.008), qc=zero.copy(),
        qi=zero.copy(), p=p, p_interface=p_ifc, exner=exner, dz=dz,
        psfc=100000.0, znt=0.10, ust=0.30, hfx=60.0, qfx=2.0e-5,
        wspd=8.0, br=-0.001,
        psim=np.log(max(z[0] / 0.10, 1.01)),
        psih=np.log(max(z[0] / 0.01, 1.01)), xland=1.0,
        u10=8.0, v10=0.0, dt=60.0,
        # WRF's theta-li extension (:732-751) carries no pblflg test and
        # can revive a column legitimately; off, it isolates the guard.
        ysu_topdown_pblmix=0,
    )


def test_the_mirror_keeps_a_column_below_the_first_interface_local():
    args = _onset_column()
    out = np_ysu_column(**args)
    assert out["kpbl"] == 1
    assert out["hpbl"] == pytest.approx(95.4339, abs=1.0e-3)
    assert out["hpbl"] < args["dz"][0]
    assert out["delta"] == 0.0
    assert np.asarray(out["exch_h"]).max() == pytest.approx(0.01, abs=1e-9)


def test_the_kernel_spells_the_guard():
    """The device statement of the same rule, checked where no card is."""
    code = [line.strip() for line in _KERNEL.read_text("utf-8").splitlines()]
    code = [line for line in code if line and not line.startswith("//")]
    start = code.index("if (sfcflg && sflux > 0.0f) {")
    end = code.index("} else pblflg = false;", start)
    block = code[start + 1:end]
    call = next(k for k, line in enumerate(block)
                if line.startswith("ysu_diagnose("))
    assert block[call - 1] == "if (pblflg) {"
    joined = " ".join(block)
    assert "pblflg = true" not in joined
    assert "pblflg = kpbl > 1" not in joined
