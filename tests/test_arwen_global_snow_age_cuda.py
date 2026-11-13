"""The CUDA Noah kernel ages a carried snowpack to the ALCALC answer.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU).  Three
grassland columns under 100 kg/m2 of snow, at night, with no
precipitation, are stepped 24 times at dt = 3600 s, carrying every field
the kernel writes from one call to the next, SNOTIME included, as the
runtime now carries ``noah_snotime``.  After k calls the age is k dt to the
bit (every k dt is a float32 integer), and the albedo the kernel writes is
ALCALC's, ``alb + sncovr (snoalb1 0.94 ** ((k dt / 86400) ** 0.58) - alb)``
with ``snoalb1 = snoalb + lvcoef (0.85 - snoalb)``, evaluated from the
kernel's own ``albbck`` and ``snowc``, to float32.  The float64 mirror
stepped the same way agrees.  The CPU half of this check, and the carry
through the runtime and the checkpoint, are in
``test_arwen_global_snow_age.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.local_gpu import no_local_gpu

from test_arwen_global_snow_age import DZS, _cold_snowy_column, alcalc_albedo


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip("GPUWM_NO_LOCAL_GPU is set: no device for the Noah kernel")
    return cp


SNOW_ALBEDOS = (0.75, 0.6, 0.82)
DT = 3600.0
CALLS = 24


def test_the_kernel_ages_a_carried_pack_to_the_alcalc_answer(cupy):
    cp = cupy
    from arwen_global.core import noah
    from arwen_global.core.npref import np_noah_column

    params = noah.pack_params(noah.load_tables())
    lvcoef = float(params.gen[noah.GEN["lvcoef"]])
    columns = [
        {**_cold_snowy_column(noah, params), "snoalb": value} for value in SNOW_ALBEDOS
    ]
    n = len(columns)

    def plane(name):
        return cp.asarray(
            np.array([[float(col.get(name, 0.0)) for col in columns]], np.float32)
        )

    dev = {name: plane(name) for name in noah._F2D}
    for name in noah._F3D:
        if name == "smcrel":
            dev[name] = cp.zeros((4, 1, n), cp.float32)
        else:
            dev[name] = cp.asarray(
                np.stack([np.asarray(col[name], np.float32) for col in columns], axis=1)[:, None, :]
            )
    dev["ivgtyp"] = cp.asarray(np.array([[col["ivgtyp"] for col in columns]], np.int32))
    dev["isltyp"] = cp.asarray(np.array([[col["isltyp"] for col in columns]], np.int32))
    dev["ebal"] = cp.zeros((1, n), cp.int32)

    mirrors = [dict(col) for col in columns]
    for k in range(1, CALLS + 1):
        noah.launch_noah(dev, params, DT, DZS, itimestep=k + 1)
        age = cp.asnumpy(dev["snotime"])[0]
        albedo = cp.asnumpy(dev["albedo"])[0]
        albbck = cp.asnumpy(dev["albbck"])[0]
        snowc = cp.asnumpy(dev["snowc"])[0]
        snow = cp.asnumpy(dev["snow"])[0]
        assert np.all(snow > 0.0), k
        assert np.all(age == np.float32(k * DT)), (k, age)
        for i, value in enumerate(SNOW_ALBEDOS):
            expected = alcalc_albedo(
                float(albbck[i]), value, lvcoef, float(snowc[i]), k * DT
            )
            assert float(albedo[i]) == pytest.approx(expected, rel=2.0e-6), (k, i)
            out = np_noah_column(mirrors[i], params, DT, DZS)
            assert out["snotime"] == k * DT
            assert float(albedo[i]) == pytest.approx(out["albedo"], abs=2.0e-5), (k, i)
            mirrors[i] = {name: out[name] for name in mirrors[i]}
    # One day: the pack's albedo has fallen by the factor 0.94 from fresh.
    for i, value in enumerate(SNOW_ALBEDOS):
        snoalb1 = value + lvcoef * (0.85 - value)
        assert float(albedo[i]) == pytest.approx(
            float(albbck[i]) + float(snowc[i]) * (0.94 * snoalb1 - float(albbck[i])),
            rel=2.0e-6,
        )
