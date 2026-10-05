"""Grell-Freitas's undilute buoyancy integral keeps its cloud-base layer.

WRF 4.7.1 ``module_cu_gf_deep.F:3024`` (cup_up_aa0) is
``IF(K.LT.KBCON(I))GO TO 100``: the ``k == kbcon`` layer is inside the
accumulation and its term reads ``dby(kbcon-1)``.  The carried kernel wrote
``k <= kbcon`` and dropped that layer, so aa0, aa1 and xaa0 were each one
layer short in the deep and the shallow arm (audit GP-3, engine fix
ab874b32e).

The breakage this prevents, named: a column whose only positive buoyancy
below the sampled levels sits at ``k == kbcon`` returns an integral of zero,
and the deep arm turns ``aa1 == 0`` into ``ierr = 17``, switching convection
off in that column; elsewhere the quasi-equilibrium closure member and the
cloud efficiency read one layer short.  The committed 216-column oracle
capture is byte-identical either way, so no parity gate catches it; this
probe compiles the shipped kernel source with one wrapper kernel and calls
the device routine directly.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core.kernels import module_source  # noqa: E402

#: GF_KMAX + 9 at the shipped compile, and the column stride GFWS_LANES.
_KP = 40 + 9
_LANES = 64

_PROBE = r"""
extern "C" __global__ void gf_probe_aa0(const float *buf, int kbcon,
                                        int ktop, int ktf, float *out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const size_t n = (size_t)GF_KP * (size_t)GFWS_LANES;
    GfCol z{(float *)buf}, zu{(float *)buf + n}, dby{(float *)buf + 2 * n},
          gam{(float *)buf + 3 * n}, tc{(float *)buf + 4 * n};
    out[0] = gfd_cup_up_aa0(z, zu, dby, gam, tc, kbcon, ktop, 0, ktf);
}
"""


@pytest.fixture(scope="module")
def probe():
    src = module_source("gf") + _PROBE
    mod = cp.RawModule(code=src, options=("-std=c++17",))
    mod.compile()
    return mod.get_function("gf_probe_aa0")


def _column(dby_levels: dict[int, float]):
    z = np.zeros(_KP, np.float32)
    zu = np.zeros(_KP, np.float32)
    dby = np.zeros(_KP, np.float32)
    gam = np.zeros(_KP, np.float32)
    tc = np.zeros(_KP, np.float32)
    for k in range(1, _KP):
        z[k] = 500.0 * k
        zu[k] = 1.0
        gam[k] = 0.5
        tc[k] = 280.0
    for k, value in dby_levels.items():
        dby[k] = value
    return z, zu, dby, gam, tc


def _launch(probe, arrays, kbcon, ktop, ktf):
    strided = np.zeros((len(arrays), _KP, _LANES), np.float32)
    for i, a in enumerate(arrays):
        strided[i, :, 0] = a
    buf = cp.asarray(strided.reshape(-1))
    out = cp.zeros(1, cp.float32)
    probe((1,), (1,), (buf, np.int32(kbcon), np.int32(ktop), np.int32(ktf), out))
    cp.cuda.Stream.null.synchronize()
    return float(out.get()[0])


def _wrf_aa0(arrays, kbcon, ktop, ktf):
    """module_cu_gf_deep.F:3020-3032 in float64, the Fortran indices kept."""
    z, zu, dby, gam, tc = (np.asarray(a, np.float64) for a in arrays)
    aa0 = 0.0
    for k in range(2, ktf + 1):
        if k < kbcon or k > ktop:
            continue
        dz = z[k] - z[k - 1]
        da = zu[k] * dz * (9.81 / (1004.0 * tc[k])) * dby[k - 1] / (1.0 + gam[k])
        aa0 += max(0.0, da)
    return aa0


@pytest.mark.gpu
def test_the_cloud_base_layer_contributes_its_buoyancy(probe):
    kbcon, ktop, ktf = 6, 20, 39
    # the only positive buoyancy is the one the k == kbcon term reads
    arrays = _column({kbcon - 1: 2.0})
    got = _launch(probe, arrays, kbcon, ktop, ktf)
    want = _wrf_aa0(arrays, kbcon, ktop, ktf)
    assert want > 0.0
    assert got == pytest.approx(want, rel=1e-6), (got, want)


@pytest.mark.gpu
def test_layers_strictly_below_cloud_base_stay_out(probe):
    kbcon, ktop, ktf = 6, 20, 39
    # buoyancy read only by k == kbcon - 1, which WRF skips
    arrays = _column({kbcon - 2: 2.0})
    assert _launch(probe, arrays, kbcon, ktop, ktf) == 0.0
    assert _wrf_aa0(arrays, kbcon, ktop, ktf) == 0.0


@pytest.mark.gpu
def test_a_full_profile_matches_the_reference(probe):
    kbcon, ktop, ktf = 5, 18, 39
    rng = np.random.default_rng(7)
    levels = {k: float(rng.uniform(-1.0, 3.0)) for k in range(1, 30)}
    arrays = _column(levels)
    got = _launch(probe, arrays, kbcon, ktop, ktf)
    want = _wrf_aa0(arrays, kbcon, ktop, ktf)
    assert got == pytest.approx(want, rel=2e-6), (got, want)
