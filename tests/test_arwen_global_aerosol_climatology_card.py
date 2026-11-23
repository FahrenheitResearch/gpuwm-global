"""WRF aer_opt = 1 aerosol in the RRTMGP shortwave, on the card.

* "none" spelled out is the driver it always was, bit for bit;
* the aerosol finalize with a zero aerosol term is the original finalize,
  bit for bit, and with a real term it is the float64 combination;
* the device optics are the WRF-loop reference column for column (deserts,
  ocean, the date line, beyond the last data latitude, above the model top);
* on, the longwave does not move, every lit column's surface sunshine falls,
  and the falls are largest under the dust;
* the fused shortwave solver (the default) carries the aerosol, bit for bit
  the finalize path's answer.

The CPU half is test_arwen_global_aerosol_climatology.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import rrtmgp  # noqa: E402
from arwen_global.physics import aerosol_climatology as aer  # noqa: E402

NZ = 30
OUTPUTS = ("rthratenlw", "rthratensw", "swupt", "swdnt", "olr", "swdown", "gsw", "glw")
# (lat, lon): Sahara, Arabia, open Pacific, N Mexico, the date line, beyond 89N.
POINTS = [(22.0, 10.0), (22.0, 45.0), (2.0, 210.0), (26.0, 255.0), (-30.0, 179.9), (89.6, 20.0)]
WHEN = datetime(1976, 3, 20, 12, 0, tzinfo=timezone.utc)


def _case(cloudy=True):
    nx = len(POINTS)
    p_half = np.linspace(101325.0, 100.0, NZ + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    temperature = np.maximum(300.0 * (p_full / 101325.0) ** 0.19, 210.0)
    qv = 0.010 * (p_full / 101325.0) ** 3

    def cube(profile):
        return cp.asarray(np.broadcast_to(
            np.asarray(profile, np.float32)[:, None, None], (NZ, 1, nx)).copy())

    zeros = cp.zeros((NZ, 1, nx), cp.float32)
    qc = zeros.copy()
    if cloudy:
        low = np.flatnonzero((p_full < 90000.0) & (p_full > 80000.0))
        qc[low, 0, 1] = 3.0e-4
    atmosphere = {
        "pressure": cube(p_full),
        "p_interface": cp.asarray(np.broadcast_to(
            p_half.astype(np.float32)[:, None, None], (NZ + 1, 1, nx)).copy()),
        "temperature": cube(temperature),
        "exner": cube((p_full / 1.0e5) ** (287.04 / 1004.6)),
        "qv": cube(qv), "qc": qc, "qi": zeros.copy(),
    }
    plane = lambda v: cp.full((1, nx), v, cp.float32)  # noqa: E731
    fields = {"tsk": plane(300.0), "emiss": plane(0.95), "albedo": plane(0.3)}
    state = SimpleNamespace(
        p_top=100.0, elapsed_seconds=0.0, qc=qc, qr=zeros.copy(), qi=zeros.copy(),
        qs=zeros.copy(), nc=zeros + cp.float32(1.0e8), nr=zeros.copy(),
        ni=zeros + cp.float32(1.0e5), ns=zeros + cp.float32(1.0e4),
        effc=zeros + cp.float32(9.0), effr=zeros + cp.float32(25.0),
        effi=zeros + cp.float32(30.0), effs=zeros + cp.float32(150.0),
        physics=SimpleNamespace(microphysics_updates=1))
    cfg = SimpleNamespace(
        mp_physics=10, bl_pbl_physics=1, icloud_bl=0,
        wrf_rrtmg_compatibility="none", dt=600.0, clock_dt=600.0,
        radt=50.0, radt_minutes=50.0)
    return atmosphere, fields, state, cfg


def _driver(**kwargs):
    lat = cp.asarray(np.array([[p[0] for p in POINTS]], np.float32))
    lon = cp.asarray(np.array([[p[1] for p in POINTS]], np.float32))
    return rrtmgp.RRTMGPRadiation(start_time=WHEN, latitude_deg=lat, longitude_deg=lon, **kwargs)


def _run(driver, cloudy=True):
    atmosphere, fields, state, cfg = _case(cloudy)
    result = driver(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    return {name: cp.asnumpy(cp.asarray(getattr(result, name))) for name in OUTPUTS}


def test_none_spelled_out_is_the_driver_it_always_was():
    base = _run(_driver())
    spelled = _run(_driver(aerosol_climatology="none"))
    for name in OUTPUTS:
        assert np.array_equal(base[name], spelled[name]), name


def test_the_driver_refuses_a_name_it_does_not_know():
    with pytest.raises(ValueError, match="aerosol_climatology"):
        _driver(aerosol_climatology="tegen")


def test_a_zero_aerosol_term_is_the_original_finalize_bit_for_bit():
    tables = rrtmgp.load_gas_tables("sw")
    rng = np.random.default_rng(7)
    ncol, nlay = 37, 25
    shape, band = (ncol, nlay, tables.ngpt), (ncol, nlay, tables.nband)
    gas = rrtmgp.GasOpticsResult(
        cp.asarray(rng.gamma(0.5, 0.3, shape).astype(np.float32)),
        cp.asarray(rng.uniform(0, 1, shape).astype(np.float32)), None, None)
    cloud = rrtmgp.CloudOpticsResult(
        cp.asarray(rng.gamma(0.5, 5, band).astype(np.float32)),
        cp.asarray(rng.uniform(0.5, 1, band).astype(np.float32)),
        cp.asarray(rng.uniform(0.7, 0.9, band).astype(np.float32)))
    mask = cp.asarray(rng.uniform(0, 1, shape) < 0.5)
    zero = tuple(cp.zeros(band, cp.float32) for _ in range(3))
    a = rrtmgp._finalize_cloud_optics(tables, gas, cloud, cloud_mask=mask)
    b = aer.finalize_cloud_aerosol_sw(tables, gas, cloud, zero, cloud_mask=mask)
    for x, y in ((a.tau, b.tau), (a.ssa, b.ssa), (a.g, b.g)):
        assert cp.array_equal(x, y)
    # A real aerosol term: the float64 combination then delta-scaling.
    term = (cp.asarray(rng.gamma(0.5, 0.05, band).astype(np.float32)),
            cp.asarray(rng.uniform(0.6, 1, band).astype(np.float32)),
            cp.asarray(rng.uniform(0.5, 0.8, band).astype(np.float32)))
    c = aer.finalize_cloud_aerosol_sw(tables, gas, cloud, term, cloud_mask=mask)
    bands = tables.gpoint_bands
    tg, wg = (cp.asnumpy(v).astype(np.float64) for v in (gas.tau, gas.ssa))
    tc, wc, gc = (cp.asnumpy(v).astype(np.float64)[:, :, bands] for v in (cloud.tau, cloud.ssa, cloud.g))
    tc = tc * cp.asnumpy(mask)
    ta, wa, ga = (cp.asnumpy(v).astype(np.float64)[:, :, bands] for v in term)
    tau = tg + ta + tc
    sca = tg * wg + ta * wa + tc * wc
    w = sca / tau
    g = (ta * wa * ga + tc * wc * gc) / sca
    f = g * g
    np.testing.assert_allclose(cp.asnumpy(c.tau), (1 - w * f) * tau, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(c.ssa), (w - w * f) / (1 - w * f), rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(c.g), (g - f) / (1 - f), rtol=2e-5, atol=2e-6)


def test_the_device_optics_are_the_wrf_loop_reference():
    clim = aer.load_climatology()
    drv = aer.TegenAerosol(
        cp.asarray([p[0] for p in POINTS], cp.float32), cp.asarray([p[1] for p in POINTS], cp.float32))
    drv.set_date(WHEN)
    p_half = np.linspace(101325.0, 100.0, NZ + 1)
    rng = np.random.default_rng(3)
    ncol = len(POINTS)
    plev = np.stack([p_half * rng.uniform(0.85, 1.0) for _ in range(ncol)])
    plev[:, -1] = 100.0
    play = 0.5 * (plev[:, :-1] + plev[:, 1:])
    nmod = NZ - 3  # the top three layers stand for the appended above-model column
    tau, ssa, g, a550 = drv.band_optics(
        cp.asarray(play, cp.float32), cp.asarray(plev, cp.float32), slice(0, ncol), nmod, aod550=True)
    tau, ssa, g, a550 = (cp.asnumpy(v) for v in (tau, ssa, g, a550))
    for c, (la, lo) in enumerate(POINTS):
        aod, rt, rs, rg = aer.column_reference(
            clim, WHEN, la, lo, play[c].astype(np.float32).astype(np.float64),
            plev[c].astype(np.float32).astype(np.float64), nlay_model=nmod)
        np.testing.assert_allclose(a550[c], aod.sum(1), rtol=2e-5, atol=1e-9)
        np.testing.assert_allclose(tau[c], rt, rtol=2e-5, atol=1e-9)
        lit = rt > 1e-7
        np.testing.assert_allclose(ssa[c][lit], rs[lit], rtol=1e-5)
        np.testing.assert_allclose(g[c][lit], rg[lit], rtol=1e-5)
        assert np.all(a550[c, nmod:] == 0)
    col = a550.sum(1)
    assert col[0] > 0.15 and col[1] > 0.2 and col[2] < 0.12, col  # Sahara, Arabia; Pacific


def test_the_device_optics_are_wrfs_own_numbers():
    from test_arwen_global_aerosol_climatology import (
        ORACLE_SCALE, ORACLE_WHEN, oracle_columns, oracle_profile)

    cols = oracle_columns()
    drv = aer.TegenAerosol(cp.asarray([c["lat"] for c in cols], cp.float32),
                           cp.asarray([c["lon"] % 360.0 for c in cols], cp.float32))
    drv.set_date(ORACLE_WHEN)
    prof = [oracle_profile(s) for s in ORACLE_SCALE]
    play = cp.asarray(np.stack([p for p, _ in prof]))
    plev = cp.asarray(np.stack([pf for _, pf in prof]))
    tau, ssa, g, a550 = (cp.asnumpy(v) for v in drv.band_optics(
        play, plev, slice(0, len(cols)), play.shape[1], aod550=True))
    rows = list(aer.RRTMGP_TO_RRTMG_ROW)
    for i, c in enumerate(cols):
        band = c["band"][:, rows, :]
        np.testing.assert_allclose(a550[i], c["aod"].sum(1), rtol=3e-5, atol=1e-12)
        np.testing.assert_allclose(tau[i], band[..., 0], rtol=3e-5, atol=1e-12)
        lit = band[..., 0] > 0
        np.testing.assert_allclose(ssa[i][lit], band[..., 1][lit], rtol=1e-5)
        np.testing.assert_allclose(g[i][lit], band[..., 2][lit], rtol=1e-5)


def test_on_the_longwave_stays_and_the_surface_sunshine_falls():
    off = _run(_driver(), cloudy=False)
    on = _run(_driver(aerosol_climatology="tegen-wrf"), cloudy=False)
    for name in ("rthratenlw", "olr", "glw"):
        assert np.array_equal(off[name], on[name]), name
    # 1976-03-20 12Z: the Sahara (10E) and Arabia (45E) are lit, the Pacific,
    # Mexico and the date line are in the night.
    lit = off["swdown"] > 50.0
    assert lit[0, 0] and lit[0, 1]
    drop = off["swdown"] - on["swdown"]
    assert np.all(drop[lit] > 0)
    dark = off["swdown"] == 0
    assert dark.sum() >= 3 and np.all(on["swdown"][dark] == 0)
    # A clear subtropical sky under Tegen's March dust (AOD 0.23 at the
    # Sahara point, 0.34 in Arabia) near noon: some tens of W/m2 at the
    # surface, more under the thicker Arabian column at a lower sun.
    assert 10.0 < drop[0, 0] < 120.0, drop
    assert 10.0 < drop[0, 1] < 120.0, drop
    # The aerosol absorbs: the atmosphere takes more sunshine, not only the reflection.
    atm_off = (off["swdnt"] - off["swupt"]) - off["gsw"]
    atm_on = (on["swdnt"] - on["swupt"]) - on["gsw"]
    assert np.all((atm_on - atm_off)[lit] > 0)


def test_on_with_cloud_runs_and_moves_only_the_shortwave():
    off = _run(_driver(), cloudy=True)
    on = _run(_driver(aerosol_climatology="tegen-wrf"), cloudy=True)
    for name in ("rthratenlw", "olr", "glw"):
        assert np.array_equal(off[name], on[name]), name
    assert not np.array_equal(off["rthratensw"], on["rthratensw"])


@pytest.mark.parametrize("cloudy", [False, True])
def test_the_fused_solver_carries_the_aerosol_bit_for_bit(cloudy, monkeypatch):
    """The fused shortwave solver (the default) with the aerosol on is the
    finalize path with the aerosol on, every output bit for bit; and the
    aerosol moves it (when the solver had no aerosol term, on was off)."""
    monkeypatch.setenv(rrtmgp.FUSED_SOLVER_ENV, "1")
    fused = _run(_driver(aerosol_climatology="tegen-wrf"), cloudy=cloudy)
    fused_off = _run(_driver(), cloudy=cloudy)
    monkeypatch.setenv(rrtmgp.FUSED_SOLVER_ENV, "lw")
    finalize = _run(_driver(aerosol_climatology="tegen-wrf"), cloudy=cloudy)
    for name in OUTPUTS:
        assert np.array_equal(fused[name], finalize[name]), name
    assert not np.array_equal(fused["swdown"], fused_off["swdown"])
    assert np.array_equal(fused["rthratenlw"], fused_off["rthratenlw"])
