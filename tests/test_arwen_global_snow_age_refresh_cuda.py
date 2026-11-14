"""The CUDA Noah kernel's snow-age refresh rules against the float64 mirror.

Device only (skipped without CuPy or under GPUWM_NO_LOCAL_GPU).  A day-old
grassland pack under 0.05, 0.4 and 2 kg/m2 of snow in one call: WRF's rule
(the default launch) zeroes the age on every column; the amount rule keeps
``(age + dt)(1 - dsnow / 1 kg/m2)`` below 1 kg/m2 and zeroes it above, as the
mirror does.  CPU half: ``test_arwen_global_snow_age_refresh.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.local_gpu import no_local_gpu

from test_arwen_global_snow_age import DZS, _cold_snowy_column


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if no_local_gpu():
        pytest.skip("GPUWM_NO_LOCAL_GPU is set: no device for the Noah kernel")
    return cp


DT = 3600.0
SNOWS = (0.05, 0.4, 2.0)


def _device(cp, noah, columns):
    n = len(columns)
    dev = {name: cp.asarray(np.array([[float(c.get(name, 0.0)) for c in columns]],
                                     np.float32)) for name in noah._F2D}
    for name in noah._F3D:
        if name == "smcrel":
            dev[name] = cp.zeros((4, 1, n), cp.float32)
        else:
            dev[name] = cp.asarray(np.stack(
                [np.asarray(c[name], np.float32) for c in columns], axis=1)[:, None, :])
    dev["ivgtyp"] = cp.asarray(np.array([[c["ivgtyp"] for c in columns]], np.int32))
    dev["isltyp"] = cp.asarray(np.array([[c["isltyp"] for c in columns]], np.int32))
    dev["ebal"] = cp.zeros((1, n), cp.int32)
    return dev


def test_the_kernel_amount_rule_matches_the_mirror_and_wrfs_rule_is_unchanged(cupy):
    cp = cupy
    from arwen_global.core import noah
    from arwen_global.core.npref import np_noah_column

    params = noah.pack_params(noah.load_tables())
    base = _cold_snowy_column(noah, params)
    columns = [{**base, "snotime": 86400.0, "rainbl": s, "sr": 1.0} for s in SNOWS]

    wrf = _device(cp, noah, columns)
    noah.launch_noah(wrf, params, DT, DZS, itimestep=2, frpcpn=True)
    spelled = _device(cp, noah, columns)
    noah.launch_noah(spelled, params, DT, DZS, itimestep=2, frpcpn=True,
                     snow_age_refresh="any")
    for name in (*noah._F2D, *noah._F3D):
        assert np.array_equal(cp.asnumpy(wrf[name]), cp.asnumpy(spelled[name])), name
    assert np.all(cp.asnumpy(wrf["snotime"]) == 0.0)

    amount = _device(cp, noah, columns)
    noah.launch_noah(amount, params, DT, DZS, itimestep=2, frpcpn=True,
                     snow_age_refresh="amount")
    age = cp.asnumpy(amount["snotime"])[0]
    albedo = cp.asnumpy(amount["albedo"])[0]
    for i, s in enumerate(SNOWS):
        out = np_noah_column(columns[i], params, DT, DZS, frpcpn=True,
                             snow_age_refresh="amount")
        assert float(age[i]) == pytest.approx(out["snotime"], rel=1.0e-6, abs=1.0e-3), i
        assert float(age[i]) == pytest.approx((86400.0 + DT) * max(1.0 - s, 0.0),
                                              rel=1.0e-6, abs=1.0e-3), i
        assert float(albedo[i]) == pytest.approx(out["albedo"], abs=2.0e-5), i
    assert np.array_equal(cp.asnumpy(wrf["snow"]), cp.asnumpy(amount["snow"]))
    assert float(albedo[0]) < float(cp.asnumpy(wrf["albedo"])[0, 0]) - 0.03


def test_the_kernel_starts_a_pack_on_bare_ground_young_under_the_amount_rule(cupy):
    """A melted-out column still carrying a 30-day age: no snow, 0.2 and
    2 kg/m2.  The amount rule zeroes the age of bare ground (Noah-MP SNOW_AGE,
    TAUSS = 0 when SNEQV <= 0), so the trace starts a pack dt * 0.8 old;
    WRF's rule leaves the dry column's age alone and zeroes it on snowfall."""
    cp = cupy
    from arwen_global.core import noah
    from arwen_global.core.npref import np_noah_column

    params = noah.pack_params(noah.load_tables())
    stale = 30 * 86400.0
    bare = {**_cold_snowy_column(noah, params), "snow": 0.0, "snowh": 0.0,
            "snowc": 0.0, "albedo": 0.2, "snotime": stale}
    falls = (0.0, 0.2, 2.0)
    columns = [{**bare, "rainbl": s, "sr": 1.0} for s in falls]

    wrf = _device(cp, noah, columns)
    noah.launch_noah(wrf, params, DT, DZS, itimestep=2, frpcpn=True)
    assert cp.asnumpy(wrf["snotime"])[0].tolist() == [stale, 0.0, 0.0]

    amount = _device(cp, noah, columns)
    noah.launch_noah(amount, params, DT, DZS, itimestep=2, frpcpn=True,
                     snow_age_refresh="amount")
    age = cp.asnumpy(amount["snotime"])[0]
    expected = (0.0, DT * (1.0 - 0.2), 0.0)
    for i, column in enumerate(columns):
        out = np_noah_column(column, params, DT, DZS, frpcpn=True,
                             snow_age_refresh="amount")
        assert out["snotime"] == pytest.approx(expected[i], rel=1.0e-12, abs=1.0e-9), i
        assert float(age[i]) == pytest.approx(expected[i], rel=1.0e-6, abs=1.0e-3), i
    # the dry column: only its carried age differs between the rules
    for name in noah._F2D:
        if name != "snotime":
            assert np.array_equal(cp.asnumpy(wrf[name])[:, 0],
                                  cp.asnumpy(amount[name])[:, 0]), name
