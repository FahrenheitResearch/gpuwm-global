"""WRF aer_opt = 1 (Tegen et al. 1997) aerosol climatology: the CPU half.

* the shipped file is the receipt's bytes, WRF's four text files unchanged,
  and its optical depths are the climatology's (global mean, deserts, ocean);
* WRF's time weights (aer_time_int), horizontal weights (interp_vec) and
  vertical rule (aer_p_int) as written in WRF;
* RRTMG's ECMWF optics on RRTMGP's band order;
* the native option: off by default, out of the identity when off, refused
  when misspelled.

The card half is test_arwen_global_aerosol_climatology_card.py.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from arwen_global.physics import aerosol_climatology as aer


def test_the_shipped_file_is_the_receipts():
    r = aer.receipt()
    assert hashlib.sha256(aer.DATA_FILE.read_bytes()).hexdigest() == r["npz_sha256"]
    assert r["licence"]["redistribution"] == "allowed"
    clim = aer.load_climatology()
    assert clim.aod_per_pa.shape == (6, 12, 46, 12, 72)
    assert np.all(clim.aod_per_pa >= 0) and np.all(np.isfinite(clim.aod_per_pa))
    assert clim.plev_hpa[0] == np.float32(20.0) and clim.plev_hpa[-1] == np.float32(959.0)
    assert clim.lon[0] == np.float32(-180.0) and clim.lon[-1] == np.float32(175.0)
    assert clim.lat[0] == np.float32(-89.0) and clim.lat[-1] == np.float32(89.0)


def _column_aod(clim, month):
    pin = clim.plev_hpa.astype(np.float64)
    edges = np.concatenate([[0.0], 0.5 * (pin[1:] + pin[:-1]), [1013.25]]) * 100.0
    return np.einsum("sjkl,k->sjl", clim.aod_per_pa[:, month].astype(np.float64), np.diff(edges))


def test_the_optical_depths_are_the_climatologys():
    clim = aer.load_climatology()
    w = np.cos(np.radians(clim.lat.astype(np.float64)))[:, None] * np.ones((1, 72))
    annual = np.mean([_column_aod(clim, m).sum(0) for m in range(12)], axis=0)
    gmean = float((annual * w).sum() / w.sum())
    # Tegen et al. 1997 and the ECMWF regridding WRF ships: about 0.09 at 550 nm.
    assert 0.07 < gmean < 0.12, gmean
    march = _column_aod(clim, 2)
    lat, lon = clim.lat, clim.lon

    def at(la, lo, species=None):
        j, i = int(np.argmin(abs(lat - la))), int(np.argmin(abs(lon - lo)))
        return float(march[:, j, i].sum() if species is None else march[species, j, i])

    assert at(22, 10) > 0.2 and at(22, 45) > 0.25          # Sahara, Arabia
    assert at(22, 10, species=2) > 0.6 * at(22, 10)        # mostly dust there
    assert at(2, -150) < 0.1 and at(-6, -60) < 0.06        # open Pacific, Amazon wet season
    assert float(march[5].max()) == 0.0                    # no volcanic stratosphere in it


@pytest.mark.parametrize("when, months, weights", [
    # 1976 is a leap year: 03-15 is day 75 and julian 74, so intjulian is 75,
    # which IS March's mid-month day.
    (datetime(1976, 3, 15), (3, 4), (1.0, 0.0)),
    (datetime(2019, 4, 15), (4, 5), (1.0, 0.0)),
    (datetime(2019, 1, 1), (12, 1), ((16 - 1.0) / 31, (1.0 + 365 - 350) / 31)),
    (datetime(2019, 12, 31, 12), (12, 1), ((16 + 365 - 365.5) / 31, (365.5 - 350) / 31)),
])
def test_wrf_time_weights(when, months, weights):
    nm, np_, f1, f2 = aer.time_weights(when)
    assert (nm + 1, np_ + 1) == months
    assert f1 == pytest.approx(weights[0], abs=2e-6)
    assert f2 == pytest.approx(weights[1], abs=2e-6)
    assert f1 + f2 == pytest.approx(1.0, abs=1e-6)


def test_wrf_interp_vec():
    clim = aer.load_climatology()
    assert aer.interp_vec(clim.lon, -177.5, True) == (0, 1, 0.5, 0.5)
    l1, l2, w1, w2 = aer.interp_vec(clim.lon, 177.5, True)
    assert (l1, l2) == (71, 0) and w2 == pytest.approx(0.5)
    l1, l2, w1, w2 = aer.interp_vec(clim.lat, 0.0, True)
    assert (clim.lat[l1], clim.lat[l2]) == (-2.0, 2.0) and w2 == pytest.approx(0.5)
    # WRF passes periodic for latitude too: beyond 89 it wraps to -89.
    l1, l2, w1, w2 = aer.interp_vec(clim.lat, 89.5, True)
    assert (l1, l2) == (45, 0) and w2 == pytest.approx(0.5 / 182.0)
    assert aer.wrf_longitude(200.0) == -160.0 and aer.wrf_longitude(180.0) == 180.0


def test_the_vertical_rule_and_the_optics_combination():
    clim = aer.load_climatology()
    when = datetime(1976, 3, 15, tzinfo=timezone.utc)  # March exactly
    # A column whose layer midpoints sit exactly on three of the data levels,
    # one above the top level and one below the bottom.
    play = np.array([1000.0, 787.0, 470.0, 45.0, 10.0]) * 100.0
    plev = np.array([1010.0, 900.0, 600.0, 300.0, 30.0, 1.0]) * 100.0
    lat, lon = -2.0, 0.0  # on a data point: no horizontal weighting
    aod, tau, ssa, g = aer.column_reference(clim, when, lat, lon, play, plev)
    j, i = int(np.argmin(abs(clim.lat - lat))), int(np.argmin(abs(clim.lon - lon)))
    a = clim.aod_per_pa[:, 2, j, :, i].astype(np.float64)
    dp = plev[:-1] - plev[1:]
    k = {p: int(np.flatnonzero(clim.plev_hpa == np.float32(p))[0]) for p in (959, 787, 470, 45, 20)}
    np.testing.assert_allclose(aod[0], a[:, k[959]] * dp[0], rtol=1e-6)   # below the bottom level
    np.testing.assert_allclose(aod[1], a[:, k[787]] * dp[1], rtol=1e-6)
    np.testing.assert_allclose(aod[2], a[:, k[470]] * dp[2], rtol=1e-6)
    np.testing.assert_allclose(aod[3], a[:, k[45]] * dp[3], rtol=1e-6)
    np.testing.assert_allclose(aod[4], a[:, k[20]] * 10.0 / 20.0 * dp[4], rtol=1e-6)  # A(1) p / p(1)
    # RRTMG iaer = 6 for band 10 (RRTMGP index 10, 16000-22650 cm-1, RRTMG row 9).
    r, w, gg = aer.RSRTAUA[9], aer.RSRPIZA[9], aer.RSRASYA[9]
    t = (r * aod[2]).sum()
    assert tau[2, 10] == pytest.approx(t, rel=1e-12)
    assert ssa[2, 10] == pytest.approx((r * aod[2] * w).sum() / t, rel=1e-12)
    assert g[2, 10] == pytest.approx((r * aod[2] * w * gg).sum() / (r * aod[2] * w).sum(), rel=1e-12)
    # Above-model layers carry none.
    aod2, tau2, _, _ = aer.column_reference(clim, when, lat, lon, play, plev, nlay_model=3)
    assert np.all(aod2[3:] == 0) and np.all(tau2[3:] == 0)


ORACLE = Path(__file__).parent / "data" / "oracles" / "aerosol_tegen_wrf" / "wrfcol.out"
ORACLE_WHEN = datetime(1976, 3, 20, 12, tzinfo=timezone.utc)
ORACLE_SCALE = (0.97, 1.0, 0.85, 1.0)
ORACLE_NLAY = 30


def oracle_columns():
    """WRF's own numbers (PROVENANCE.md): per column (lat, lon, aod (30, 6),
    tau/ssa/g (30, 14) in RRTMG band order)."""
    cols, cur = [], None
    for line in ORACLE.read_text().splitlines():
        t = line.split()
        if t[0] == "column":
            cur = {"lat": float(t[3]), "lon": float(t[5]), "aod": [], "band": []}
            cols.append(cur)
        elif t[0] == "layer":
            cur["aod"].append([float(x) for x in t[4:10]])
            cur["band"].append([])
        elif t[0] == "band":
            cur["band"][-1].append([float(x) for x in t[2:5]])
    for c in cols:
        c["aod"] = np.array(c["aod"])
        c["band"] = np.array(c["band"])  # (30, 14, 3)
    return cols


def oracle_profile(scale):
    """The driver's interfaces and layers, in its float32 arithmetic."""
    f32 = np.float32
    k = np.arange(ORACLE_NLAY + 1, dtype=f32)
    phalf = f32(101325.0) + (f32(100.0) - f32(101325.0)) * k / f32(ORACLE_NLAY)
    pf = (phalf * f32(scale)).astype(f32)
    pf[-1] = f32(100.0)
    p = (f32(0.5) * (pf[:-1] + pf[1:])).astype(f32)
    return p, pf


def test_the_reference_is_wrfs_own_numbers():
    clim = aer.load_climatology()
    cols = oracle_columns()
    assert len(cols) == 4
    rows = list(aer.RRTMGP_TO_RRTMG_ROW)
    for c, scale in zip(cols, ORACLE_SCALE):
        p, pf = oracle_profile(scale)
        aod, tau, ssa, g = aer.column_reference(
            clim, ORACLE_WHEN, c["lat"], c["lon"] % 360.0, p.astype(np.float64), pf.astype(np.float64))
        np.testing.assert_allclose(aod, c["aod"], rtol=2e-5, atol=1e-12)
        band = c["band"][:, rows, :]  # RRTMG order -> RRTMGP order
        np.testing.assert_allclose(tau, band[..., 0], rtol=2e-5, atol=1e-12)
        lit = band[..., 0] > 0
        np.testing.assert_allclose(ssa[lit], band[..., 1][lit], rtol=1e-6)
        np.testing.assert_allclose(g[lit], band[..., 2][lit], rtol=1e-6)
    # And the columns are the climatology's: dust over the Sahara, little over the sea.
    assert cols[0]["aod"].sum() > 0.2 and cols[0]["aod"][:, 2].sum() > 0.1
    assert cols[1]["aod"].sum() < 0.1


def test_rrtmgp_band_map():
    rows = aer.RRTMGP_TO_RRTMG_ROW
    assert sorted(rows) == list(range(14))
    for b, (lo, hi) in enumerate(aer.RRTMGP_SW_BAND_LIMITS_CM1):
        rlo, rhi = aer.RRTMG_SW_BAND_LIMITS_CM1[rows[b]]
        centre = 0.5 * (lo + hi)
        assert rlo <= centre <= rhi, (b, lo, hi, rlo, rhi)


def test_the_shipped_sw_k_distribution_has_the_assumed_bands():
    netcdf = pytest.importorskip("netCDF4")
    try:
        from arwen_global.core import rrtmgp
        path = rrtmgp._table("rrtmgp-gas-sw-g224.nc")
    except Exception as exc:  # companion data absent on this host
        pytest.skip(f"no RRTMGP tables: {exc}")
    with netcdf.Dataset(path) as nc:
        lims = np.asarray(nc["bnd_limits_wavenumber"][:], np.float64)
    np.testing.assert_allclose(lims, np.asarray(aer.RRTMGP_SW_BAND_LIMITS_CM1, np.float64), atol=0.5)


def _options(**extra):
    from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
    from arwen_global.physics.native_options import NativePhysicsOptions

    return NativePhysicsOptions.from_mapping({
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": "2024-05-21T00:00:00Z", **extra})


def test_the_native_option_is_off_by_default_and_out_of_the_identity():
    from arwen_global.physics import native_options

    assert native_options.AEROSOL_CLIMATOLOGIES == aer.AEROSOL_CLIMATOLOGIES
    base = _options()
    assert base.radiation_aerosol == "none"
    assert "radiation_aerosol" not in base.identity
    assert _options(radiation_aerosol="none").identity == base.identity
    on = _options(radiation_aerosol="tegen-wrf")
    assert on.identity["radiation_aerosol"] == "tegen-wrf"
    for bad in ("tegen", "TEGEN-WRF", 1, True):
        with pytest.raises(ValueError, match="radiation_aerosol must be"):
            _options(radiation_aerosol=bad)


def test_the_runtime_hands_the_driver_nothing_when_off():
    from arwen_global.physics.native_runtime import NativePhysicsRuntime

    off = NativePhysicsRuntime.__new__(NativePhysicsRuntime)
    off.options = _options()
    assert off._aerosol_kwargs() == {}
    on = NativePhysicsRuntime.__new__(NativePhysicsRuntime)
    on.options = _options(radiation_aerosol="tegen-wrf")
    assert on._aerosol_kwargs() == {"aerosol_climatology": "tegen-wrf"}
