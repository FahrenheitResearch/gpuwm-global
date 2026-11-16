"""Grell-Freitas experiments 7 and 8 keep land deep convection on at T383.

Experiments 7 and 8 run WRF's convective-memory formulas at csum 39 and 20,
which lower the land entrainment rate to 1e-5.  The kernel used to feed that
rate to the scale-aware factor too: radius = 0.2 / 1e-5 = 20 km, so
frh = 3.14 r^2 / dx^2 capped at 0.9 for any dx below 37.4 km, sig fell to
(1 - 0.9)^2 = 0.01 = sig_thresh, every land column with cloud-base RH >= 0.97
exited with ierr 231 and every other land column had its mass flux times 0.01
(land deep convection off at T383 and finer).  No reference driver runs the
formula at csum > 0 (WRF resets cactiv to 0 each step; CCPP comments the csum
term out), so the radius, frh and sig now come from the csum-0 rate and the
reduced rate acts on the plume only.

The breakage this prevents, named: land sig at T383 dx under experiment 7 is
again 0.01 instead of WRF's csum-0 value (about 0.958).  The probe runs the
shipped kernel's gf_deep_stage on synthetic columns and reads the stage's
own sig capture, which is written before any exit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import gf  # noqa: E402

_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "gf_wrf461_oracle"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from gf_field_lists import IN_LEV, IN_SCA, ISCA_FIELDS, LEV_FIELDS, SCA_FIELDS  # noqa: E402

NZ = 40
_N_IN_LEV, _N_IN_SCA = len(IN_LEV), len(IN_SCA)
_INS_XLAND, _INS_DX = IN_SCA.index("xlandi"), IN_SCA.index("dx")
_SCA_ENTR, _SCA_SIG = SCA_FIELDS.index("entr_rate"), SCA_FIELDS.index("sig")
_SCA_SIG_THRESH = SCA_FIELDS.index("sig_thresh")
_N_SCA, _N_ISCA, _N_LEV = len(SCA_FIELDS), len(ISCA_FIELDS), len(LEV_FIELDS)
#: T383 on its 1152-longitude Gaussian grid, at the equator; T255 and T127
DX_T383 = 2.0 * np.pi * 6.371e6 / 1152.0
DX = {"T127": 2.0 * np.pi * 6.371e6 / 384.0, "T255": 2.0 * np.pi * 6.371e6 / 768.0,
      "T383": DX_T383, "T1023": 2.0 * np.pi * 6.371e6 / 3072.0}


def _wrf_csum0_sig(dx: float) -> float:
    """module_cu_gf_deep.F:458-471 at csum 0, float32 as the kernel does it."""
    f = np.float32
    radius = f(0.2) / f(7.0e-5)
    frh = min(f(1.0), f(f(f(f(3.14) * radius) * radius) / f(dx)) / f(dx))
    frh = min(frh, f(0.9))
    return float(f(f(1.0) - frh) * f(f(1.0) - frh))


def _columns(xland: list[float], dx: list[float]):
    n = len(xland)
    p = np.linspace(100000.0, 10000.0, NZ, dtype=np.float32)
    z = (-7000.0 * np.log(p / 101325.0)).astype(np.float32)
    t = np.maximum(300.0 - 0.0065 * z, 200.0).astype(np.float32)
    q = (0.018 * np.exp(-z / 2500.0)).astype(np.float32)
    rho = (p / (287.0 * t)).astype(np.float32)
    lev = np.zeros((n, _N_IN_LEV, NZ), np.float32)
    for c in range(n):
        lev[c] = np.stack([z, t, q, t, q, p, np.zeros_like(p), np.zeros_like(p), rho,
                           np.zeros_like(p)])
    sca = np.zeros((n, _N_IN_SCA), np.float32)
    sca[:, 1] = 1000.0     # psur, hPa
    sca[:, 2] = 150.0      # hfx
    sca[:, 3] = 5.0e-5     # qfx
    sca[:, _INS_XLAND] = xland
    sca[:, _INS_DX] = dx
    sca[:, 6] = 150.0      # ccn
    sca[:, 7] = 600.0      # dtime
    kpbl = np.full((n,), 5, np.int32)
    return lev, sca, kpbl


def _stage(deep_experiment: int, xland: list[float], dx: list[float]):
    module = gf._gf_module(NZ, deep_experiment=deep_experiment)
    lev, sca, kpbl = _columns(xland, dx)
    n = len(xland)
    d_lev = cp.zeros((n, _N_LEV, NZ), cp.float32)
    d_sca = cp.zeros((n, _N_SCA), cp.float32)
    d_isca = cp.zeros((n, _N_ISCA), cp.int32)
    d_ws = cp.zeros(gf.gf_workspace_floats(NZ, n), cp.float32)
    fn = module.get_function("gf_deep_stage")
    fn(((n + 63) // 64,), (64,), (cp.asarray(lev), cp.asarray(sca), cp.asarray(kpbl),
                                  d_lev, d_sca, d_isca, d_ws, np.int32(n), np.int32(NZ)))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(d_sca)


@pytest.mark.gpu
def test_land_sig_under_experiment_7_and_8_is_the_csum0_value_at_every_resolution():
    names = list(DX)
    land = [1.0] * len(names)
    water = [2.0] * len(names)
    dx = [DX[k] for k in names]
    base_land = _stage(0, land, dx)
    base_water = _stage(0, water, dx)
    for experiment in (7, 8):
        on_land = _stage(experiment, land, dx)
        on_water = _stage(experiment, water, dx)
        for j, name in enumerate(names):
            want = _wrf_csum0_sig(dx[j])
            assert on_land[j, _SCA_SIG] == pytest.approx(want, rel=1e-6), (experiment, name)
            assert on_land[j, _SCA_SIG] == base_land[j, _SCA_SIG], (experiment, name)
            assert on_water[j, _SCA_SIG] == base_water[j, _SCA_SIG], (experiment, name)
            assert on_land[j, _SCA_SIG] > on_land[j, _SCA_SIG_THRESH], (experiment, name)
            # the plume keeps the reduced land rate: 7e-5 - 3e-6 min(20, csum) = 1e-5
            assert on_land[j, _SCA_ENTR] == pytest.approx(1.0e-5, rel=1e-5), (experiment, name)
            assert on_water[j, _SCA_ENTR] == pytest.approx(7.0e-5, rel=1e-5), (experiment, name)
    # the T383 value named in the docstring
    j = names.index("T383")
    assert _stage(7, land, dx)[j, _SCA_SIG] == pytest.approx(0.958, abs=1e-3)


@pytest.mark.gpu
def test_experiment_0_stage_is_unchanged_on_land_and_water():
    names = list(DX)
    dx = [DX[k] for k in names] * 2
    xland = [1.0] * len(names) + [2.0] * len(names)
    out = _stage(0, xland, dx)
    for j in range(len(dx)):
        assert out[j, _SCA_SIG] == pytest.approx(_wrf_csum0_sig(dx[j]), rel=1e-6)
        assert out[j, _SCA_ENTR] == pytest.approx(7.0e-5, rel=1e-5)


def test_experiments_7_and_8_take_the_radius_from_the_csum0_rate_only():
    source = (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")
    block = source[source.index("#if GF_DEEP_EXPERIMENT == 7 || GF_DEEP_EXPERIMENT == 8\n    // The scale-aware"):]
    block = block[:block.index("#endif")]
    assert "float radius = FDIV(K_P2, K_ENTR_BASE);" in block
    assert "#else\n    float radius = FDIV(K_P2, entr_rate);\n" in block
