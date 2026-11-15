"""The CUDA Noah kernel's snow_fresh_albedo and snow_thaw_albedo against the float64 mirror.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU).  Packs aged
0, 1, 5 and 20 days, cold (skin 260 K) and at the melting point (273.16 K),
no snowfall: the default launch is bit for bit the launch that spells the
defaults, and every rule's albedo and age match the mirror.  CPU half:
``test_arwen_global_snow_fresh_albedo.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_arwen_global_snow_age import DZS, _cold_snowy_column
from test_arwen_global_snow_age_refresh_cuda import _device, cupy  # noqa: F401

DT = 3600.0
AGES = (0.0, 86400.0, 5 * 86400.0, 20 * 86400.0)
SKINS = (260.0, 273.16)


@pytest.mark.parametrize("rule", [
    {}, {"snow_fresh_albedo": "masked"}, {"snow_thaw_albedo": True},
    {"snow_fresh_albedo": "masked", "snow_thaw_albedo": True}])
def test_the_kernel_albedo_rules_match_the_mirror(cupy, rule):  # noqa: F811
    cp = cupy
    from arwen_global.core import noah
    from arwen_global.core.npref import np_noah_column

    params = noah.pack_params(noah.load_tables())
    base = _cold_snowy_column(noah, params)
    columns = [{**base, "snotime": a, "tsk": t, "rainbl": 0.0} for a in AGES for t in SKINS]
    dflt = _device(cp, noah, columns)
    noah.launch_noah(dflt, params, DT, DZS, itimestep=2, frpcpn=True,
                     snow_age_refresh="amount")
    spelled = _device(cp, noah, columns)
    noah.launch_noah(spelled, params, DT, DZS, itimestep=2, frpcpn=True,
                     snow_age_refresh="amount", snow_fresh_albedo="livneh",
                     snow_thaw_albedo=False)
    for name in (*noah._F2D, *noah._F3D):
        assert np.array_equal(cp.asnumpy(dflt[name]), cp.asnumpy(spelled[name])), name
    dev = _device(cp, noah, columns)
    noah.launch_noah(dev, params, DT, DZS, itimestep=2, frpcpn=True,
                     snow_age_refresh="amount", **rule)
    albedo = cp.asnumpy(dev["albedo"])[0]
    age = cp.asnumpy(dev["snotime"])[0]
    for i, c in enumerate(columns):
        out = np_noah_column(c, params, DT, DZS, frpcpn=True, snow_age_refresh="amount", **rule)
        assert float(age[i]) == pytest.approx(out["snotime"], rel=1.0e-6, abs=1.0e-3), i
        assert float(albedo[i]) == pytest.approx(out["albedo"], abs=2.0e-5), i
