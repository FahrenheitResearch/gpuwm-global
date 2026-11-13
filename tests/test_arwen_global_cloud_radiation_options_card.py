"""The inhomogeneity factor in the RRTMGP driver, on the card.

``RRTMGPRadiation(cloud_path_scale=...)`` multiplies the in-cloud liquid and
ice paths both solvers radiate (Tiedtke 1996, Mon. Wea. Rev. 124, 745;
Cahalan et al. 1994, J. Atmos. Sci. 51, 2434).  This file holds:

* the factor spelled 1.0 is the driver it always was, bit for bit;
* the factor 0.7 is exactly the recall the cause table measured it with
  (``tools/arwen_global_radiation_recalls.py`` on exp/attribution scaled
  the paths handed to the cloud optics), here on both solvers, bit for bit;
* a clear column is untouched, and a sunlit cloudy column reflects less;
* the driver refuses a factor it cannot mean.

The CPU half is ``test_arwen_global_cloud_radiation_options.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import rrtmgp  # noqa: E402

NZ, NY, NX = 20, 2, 2
OUTPUTS = ("rthratenlw", "rthratensw", "swupt", "swdnt", "olr", "swdown",
           "gsw", "glw")


def _case():
    p_half = np.linspace(101325.0, 100.0, NZ + 1, dtype=np.float64)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    temperature = np.maximum(288.0 * (p_full / 101325.0) ** 0.19, 210.0)
    qv = 0.012 * (p_full / 101325.0) ** 3

    def cube(profile):
        return cp.asarray(np.broadcast_to(
            np.asarray(profile, np.float32)[:, None, None], (NZ, NY, NX)).copy())

    zeros = cp.zeros((NZ, NY, NX), cp.float32)
    qc, qi, qs = zeros.copy(), zeros.copy(), zeros.copy()
    # (0, 0): low liquid cloud; (0, 1): liquid under ice; (1, 0): ice only;
    # (1, 1): clear.
    low = np.flatnonzero((p_full < 92000.0) & (p_full > 80000.0))
    high = np.flatnonzero((p_full < 35000.0) & (p_full > 25000.0))
    qc[low, 0, 0] = 4.0e-4
    qc[low, 0, 1] = 2.0e-4
    qi[high, 0, 1] = 6.0e-5
    qi[high, 1, 0] = 6.0e-5
    qs[high, 1, 0] = 2.0e-5
    atmosphere = {
        "pressure": cube(p_full),
        "p_interface": cp.asarray(np.broadcast_to(
            p_half.astype(np.float32)[:, None, None], (NZ + 1, NY, NX)).copy()),
        "temperature": cube(temperature),
        "exner": cube((p_full / 1.0e5) ** (287.04 / 1004.6)),
        "qv": cube(qv),
        "qc": qc, "qi": qi,
    }
    plane = lambda v: cp.full((NY, NX), v, cp.float32)  # noqa: E731
    fields = {"tsk": plane(290.0), "emiss": plane(0.98), "albedo": plane(0.08)}
    state = SimpleNamespace(
        p_top=100.0, elapsed_seconds=0.0,
        qc=qc, qr=zeros.copy(), qi=qi, qs=qs,
        nc=zeros + cp.float32(1.0e8), nr=zeros.copy(),
        ni=zeros + cp.float32(1.0e5), ns=zeros + cp.float32(1.0e4),
        effc=zeros + cp.float32(9.0), effr=zeros + cp.float32(25.0),
        effi=zeros + cp.float32(30.0), effs=zeros + cp.float32(150.0),
        physics=SimpleNamespace(microphysics_updates=1))
    cfg = SimpleNamespace(
        mp_physics=10, bl_pbl_physics=1, icloud_bl=0,
        wrf_rrtmg_compatibility="none", dt=300.0, clock_dt=300.0,
        radt=50.0, radt_minutes=50.0)
    return atmosphere, fields, state, cfg


def _driver(**kwargs):
    # Noon at the Greenwich meridian near the June solstice: every column lit.
    lat = cp.asarray(np.array([[20.0, 20.0], [25.0, 25.0]], np.float32))
    lon = cp.asarray(np.array([[0.0, 0.5], [0.0, 0.5]], np.float32))
    return rrtmgp.RRTMGPRadiation(
        start_time=datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc),
        latitude_deg=lat, longitude_deg=lon, **kwargs)


def _run(driver):
    atmosphere, fields, state, cfg = _case()
    result = driver(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    return {name: cp.asnumpy(cp.asarray(getattr(result, name)))
            for name in OUTPUTS}


def test_the_factor_spelled_one_is_the_driver_it_always_was():
    base = _run(_driver())
    spelled = _run(_driver(cloud_path_scale=1.0))
    for name in OUTPUTS:
        assert np.array_equal(base[name], spelled[name]), name


def test_the_factor_is_the_recall_the_cause_table_measured(monkeypatch):
    scaled = _run(_driver(cloud_path_scale=0.7))
    original = rrtmgp._cloud_optics

    def recall(tables, clwp, ciwp, reliq, dgice, *, out):
        return original(tables, clwp * np.float32(0.7), ciwp * np.float32(0.7),
                        reliq, dgice, out=out)

    monkeypatch.setattr(rrtmgp, "_cloud_optics", recall)
    patched = _run(_driver())
    for name in OUTPUTS:
        assert np.array_equal(scaled[name], patched[name]), name


def test_clear_columns_stay_and_cloudy_columns_reflect_less():
    base = _run(_driver())
    scaled = _run(_driver(cloud_path_scale=0.7))
    # (1, 1) is clear: nothing moves there.
    for name in OUTPUTS:
        a, b = base[name], scaled[name]
        assert np.array_equal(a[..., 1, 1], b[..., 1, 1]), name
    for j, i in ((0, 0), (0, 1), (1, 0)):
        assert scaled["swupt"][j, i] < base["swupt"][j, i], (j, i)
        assert scaled["gsw"][j, i] > base["gsw"][j, i], (j, i)


@pytest.mark.parametrize("value", [0.0, 1.5, -0.7, True, "0.7"])
def test_the_driver_refuses_a_factor_it_cannot_mean(value):
    with pytest.raises(ValueError, match="cloud_path_scale"):
        _driver(cloud_path_scale=value)
