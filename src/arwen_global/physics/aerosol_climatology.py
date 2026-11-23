# ======================================================================
# THIRD-PARTY NOTICE.  The per-type spectral optical properties below
# (RSRTAUA, RSRPIZA, RSRASYA) are transcribed from WRF v4.6.1
# phys/module_ra_rrtmg_sw.F, subroutine swaerpr (J.-J. Morcrette, ECMWF,
# 2003, reformatted by M. J. Iacono, AER, 2006).  That routine is AER's
# work and WRF preserves AER's own notice over it:
#
#     Copyright 2002-2008, Atmospheric & Environmental Research, Inc. (AER).
#     This software may be used, copied, or redistributed as long as it is
#     not sold and this copyright notice is reproduced on each copy made.
#     This model is provided as is without any express or implied warranties.
#                           (http://www.rtweb.aer.com/)
#
# ArWen takes this material under AER's own current grant instead: BSD
# 3-Clause, "Copyright (c) 2020, Atmospheric and Environmental Research",
# published by AER at github.com/AER-RC/RRTMG_SW.  Text in
# licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt.
#
# The climatology data file (data/aerosol-tegen-wrf.npz) is WRF's
# run/aerosol*.formatted, which WRF ships in the public domain (WRF
# LICENSE.txt); see data/aerosol-tegen-wrf-receipt.json.
# ======================================================================
"""WRF ``aer_opt = 1``: the Tegen et al. (1997) monthly aerosol climatology
in the RRTMGP shortwave.

``[physics] native_adapter_options = { ..., radiation_aerosol = "tegen-wrf" }``
gives the shortwave the aerosol WRF's RRTMG reads under ``aer_opt = 1``.
"none" (the default) is the radiation this package always ran: gas plus
cloud, no aerosol (WRF ``aer_opt = 0``), bit for bit.

What WRF does, and what this module does the same way:

* **Data.**  ``aerosol_in`` (module_physics_init.F) reads six types of
  aerosol optical depth at 550 nm per pascal on 12 pressure levels, 46
  latitudes, 72 longitudes and 12 months (Ryan Torn's regridding of the
  ECMWF/Tegen climatology: organic carbon, sea salt, dust, black carbon,
  sulfate, stratospheric).  The file this package ships IS those four text
  files, read in WRF's loop order (``data/aerosol-tegen-wrf.npz``; the
  receipt names their SHA-256s, equal to WRF v4.6.1's ``run/``).
* **Horizontal.**  ``interp_vec`` bilinear weights on the 5-degree
  longitudes (periodic) and the 46 latitudes (WRF passes ``periodic =
  .true.`` for latitude too; kept).
* **Time.**  ``aer_time_int`` (module_radiation_driver.F): linear between
  the mid-month days 16, 45, 75, ... 350 of a 365-day year, cycled.
* **Vertical.**  ``aer_p_int``: linear in pressure between the two
  bracketing levels, ``A(1) p / p(1)`` above the top level, the bottom
  level's value below it, times the layer's pressure thickness in pascals.
  The above-model layers the RRTMGP driver appends carry none (WRF's own
  extra top layer has ``ecaer = 0``).
* **Optics.**  RRTMG SW ``iaer = 6`` (module_ra_rrtmg_sw.F, ``spcvmc``
  inputs): per band, tau = sum r_s aod_s, ssa = sum r_s aod_s w_s / tau,
  g = sum r_s aod_s w_s g_s / (tau ssa), with ECMWF's six-type tables
  (continental, maritime, desert, urban, volcanic, stratospheric
  background) taken by WRF in file order.  RRTMGP's 14 shortwave bands are
  RRTMG's 14 bands with the 820-2680 cm-1 band moved first; the band map
  below is checked against the shipped k-distribution's limits.
* **Combination.**  Added to gas and cloud before the total is
  delta-scaled, as ``spcvmc`` combines Rayleigh, gas, aerosol and cloud
  into its total-sky ``ztauo/zomco/zgco`` and delta-scales it.  The cloud
  term keeps its McICA mask; aerosol is in every subcolumn.

Longwave: none, as in WRF (RRTMG LW reads aerosol only from WRF-Chem).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

#: The options ``radiation_aerosol`` accepts.
AEROSOL_CLIMATOLOGIES = ("none", "tegen-wrf")

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "aerosol-tegen-wrf.npz"
RECEIPT_FILE = Path(__file__).resolve().parent.parent / "data" / "aerosol-tegen-wrf-receipt.json"

#: WRF aer_time_int's mid-month days (date_oz), 1-based day of a 365-day year.
DATE_OZ = (16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350)

# RRTMG SW ECMWF aerosol optics (swaerpr), rows = RRTMG bands 16..29 in
# RRTMG's order (3.846-3.077 um first, 12.195-3.846 um last), columns = the
# six types.  rsrtaua: band optical depth over the 0.55 um optical depth;
# rsrpiza: single-scattering albedo; rsrasya: asymmetry parameter.
_T1 = (0.10849, 0.66699, 0.65255, 0.11600, 0.06529, 0.04468)
_T3 = (0.20543, 0.84642, 0.84958, 0.21673, 0.28270, 0.10915)
_T8 = (0.52838, 0.93285, 0.93449, 0.53078, 0.67148, 0.46608)
_T10 = (1.69446, 1.11855, 1.09212, 1.72145, 1.03858, 1.12044)
RSRTAUA = np.array([_T1, _T1, _T3, _T3, _T3, _T3, _T3, _T8, _T8,
                    _T10, _T10, _T10, _T10, _T1], np.float64)
_W1 = (.5230504, .7868518, .8531531, .4048149, .8748231, .2355667)
_W3 = (.8287144, .9949396, .9279543, .6765051, .9467578, .9955938)
_W8 = (.8970131, .9984940, .9245594, .7768385, .9532763, .9999999)
_W10 = (.9148907, .9956173, .7504584, .8131335, .9401905, .9999999)
RSRPIZA = np.array([_W1, _W1, _W3, _W3, _W3, _W3, _W3, _W8, _W8,
                    _W10, _W10, _W10, _W10, _W1], np.float64)
_G1 = (0.700610, 0.818871, 0.702399, 0.689886, .4629866, .1907639)
_G3 = (0.636342, 0.802467, 0.691305, 0.627497, .6105750, .4760794)
_G8 = (0.668431, 0.788530, 0.698682, 0.657422, .6735182, .6519706)
_G10 = (0.729019, 0.803129, 0.784592, 0.712208, .7008249, .7270548)
RSRASYA = np.array([_G1, _G1, _G3, _G3, _G3, _G3, _G3, _G8, _G8,
                    _G10, _G10, _G10, _G10, _G1], np.float64)

#: RRTMG SW band limits (cm-1) in RRTMG's order (bands 16..29).
RRTMG_SW_BAND_LIMITS_CM1 = (
    (2600, 3250), (3250, 4000), (4000, 4650), (4650, 5150), (5150, 6150),
    (6150, 7700), (7700, 8050), (8050, 12850), (12850, 16000),
    (16000, 22650), (22650, 29000), (29000, 38000), (38000, 50000),
    (820, 2600))
#: RRTMGP SW band b reads RRTMG table row RRTMGP_TO_RRTMG_ROW[b].
RRTMGP_TO_RRTMG_ROW = (13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
#: RRTMGP v1.9 SW band limits (cm-1), what the row map above assumes.
RRTMGP_SW_BAND_LIMITS_CM1 = (
    (820, 2680), (2680, 3250), (3250, 4000), (4000, 4650), (4650, 5150),
    (5150, 6150), (6150, 7700), (7700, 8050), (8050, 12850),
    (12850, 16000), (16000, 22650), (22650, 29000), (29000, 38000),
    (38000, 50000))

NSPECIES, NBAND = 6, 14


def band_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tau ratio, ssa, g), each (14 RRTMGP bands, 6 types), float64."""
    rows = list(RRTMGP_TO_RRTMG_ROW)
    return RSRTAUA[rows], RSRPIZA[rows], RSRASYA[rows]


@dataclass(frozen=True)
class TegenClimatology:
    aod_per_pa: np.ndarray  # (6 species, 12 months, 46 lat, 12 lev, 72 lon) float32
    plev_hpa: np.ndarray    # (12,) top-down
    lat: np.ndarray         # (46,) south to north
    lon: np.ndarray         # (72,) -180 .. 175


@lru_cache(maxsize=1)
def load_climatology(path: str | None = None) -> TegenClimatology:
    with np.load(path or DATA_FILE) as z:
        clim = TegenClimatology(
            aod_per_pa=np.ascontiguousarray(z["aod_per_pa"], np.float32),
            plev_hpa=np.ascontiguousarray(z["plev_hpa"], np.float32),
            lat=np.ascontiguousarray(z["lat"], np.float32),
            lon=np.ascontiguousarray(z["lon"], np.float32))
    if clim.aod_per_pa.shape != (NSPECIES, 12, 46, 12, 72):
        raise ValueError(f"aerosol climatology has shape {clim.aod_per_pa.shape}")
    if not (np.all(np.diff(clim.plev_hpa) > 0) and np.all(np.diff(clim.lat) > 0)
            and np.all(np.diff(clim.lon) > 0)):
        raise ValueError("aerosol climatology axes must increase")
    return clim


def receipt() -> dict:
    return json.loads(RECEIPT_FILE.read_text(encoding="utf-8"))


def time_weights(valid_time: datetime) -> tuple[int, int, float, float]:
    """WRF ``aer_time_int``: (nm, np, fact1, fact2), months 0-based.

    ``julian`` is WRF's 0-based fractional day of the year at 0Z 1 Jan.
    The arithmetic is WRF's REAL (float32), spelled as it is there.
    """
    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0)
    julian = np.float32((valid_time.timetuple().tm_yday - 1) + hour / 24.0)
    f32 = np.float32
    intjulian = f32(julian + f32(1.0))
    ijul = int(intjulian)
    intjulian = f32(intjulian - f32(ijul))
    ijul = ijul % 365
    if ijul == 0:
        ijul = 365
    intjulian = f32(intjulian + f32(ijul))
    np1 = 1
    for m in range(1, 13):
        if DATE_OZ[m - 1] > intjulian:
            np1 = m
            break
    cdayozp = f32(DATE_OZ[np1 - 1])
    if np1 > 1:
        cdayozm = f32(DATE_OZ[np1 - 2])
        np_, nm = np1, np1 - 1
    else:
        cdayozm = f32(DATE_OZ[11])
        np_, nm = np1, 12
    days = f32(365.0)
    if np1 == 1:
        deltat = f32(cdayozp + days - cdayozm)
        if intjulian > cdayozp:
            fact1 = f32((cdayozp + days - intjulian) / deltat)
            fact2 = f32((intjulian - cdayozm) / deltat)
        else:
            fact1 = f32((cdayozp - intjulian) / deltat)
            fact2 = f32((intjulian + days - cdayozm) / deltat)
    else:
        deltat = f32(cdayozp - cdayozm)
        fact1 = f32((cdayozp - intjulian) / deltat)
        fact2 = f32((intjulian - cdayozm) / deltat)
    return nm - 1, np_ - 1, float(fact1), float(fact2)


def date_table(clim: TegenClimatology, valid_time: datetime) -> np.ndarray:
    """(6, 46, 12, 72) float32: WRF aerodt, aerodm(nm) fact1 + aerodm(np) fact2."""
    nm, np_, fact1, fact2 = time_weights(valid_time)
    a = clim.aod_per_pa
    return np.ascontiguousarray(
        a[:, nm] * np.float32(fact1) + a[:, np_] * np.float32(fact2), np.float32)


def interp_vec(locvec: np.ndarray, locwant: float, periodic: bool):
    """WRF module_physics_init.F ``interp_vec``, 0-based indices, float32."""
    f32 = np.float32
    locwant = f32(locwant)
    n = locvec.size
    for k in range(n - 1):
        if locvec[k] <= locwant < locvec[k + 1]:
            loc1, loc2, v1, v2 = k, k + 1, locvec[k], locvec[k + 1]
            break
    else:
        if periodic:
            if locwant < locvec[0]:
                loc1, loc2, v1, v2 = n - 1, 0, f32(locvec[-1] - f32(360.0)), locvec[0]
            else:
                loc1, loc2, v1, v2 = n - 1, 0, locvec[-1], f32(locvec[0] + f32(360.0))
        else:
            if locwant < locvec[0]:
                loc1 = loc2 = 0
                v1 = v2 = locvec[0]
            else:
                loc1 = loc2 = n - 1
                v1 = v2 = locvec[-1]
    w2 = f32(f32(locwant - v1) / f32(v2 - v1))
    return loc1, loc2, float(f32(1.0) - w2), float(w2)


def wrf_longitude(lon_deg: float) -> float:
    """Model longitude (0..360) as WRF's XLONG (-180..180]."""
    lon = float(lon_deg) % 360.0
    return lon - 360.0 if lon > 180.0 else lon


def column_reference(clim: TegenClimatology, valid_time: datetime, lat: float,
                     lon: float, play_pa: np.ndarray, plev_pa: np.ndarray,
                     nlay_model: int | None = None):
    """One column the way WRF computes it, written as WRF's loops are.

    ``play_pa`` (nlay,) and ``plev_pa`` (nlay + 1,) bottom-to-top.  Returns
    (aod (nlay, 6) per type at 0.55 um, tau, ssa, g each (nlay, 14 RRTMGP
    bands)), float64.  The test reference for the device kernel.
    """
    table = date_table(clim, valid_time).astype(np.float64)
    j1, j2, wj1, wj2 = interp_vec(clim.lat, lat, True)
    i1, i2, wi1, wi2 = interp_vec(clim.lon, wrf_longitude(lon), True)
    prof = (wi1 * (wj1 * table[:, j1, :, i1] + wj2 * table[:, j2, :, i1])
            + wi2 * (wj1 * table[:, j1, :, i2] + wj2 * table[:, j2, :, i2]))  # (6, 12)
    pin = clim.plev_hpa.astype(np.float64)
    nlay = play_pa.size
    nmod = nlay if nlay_model is None else nlay_model
    aod = np.zeros((nlay, NSPECIES))
    for k in range(nmod):
        pmid = float(play_pa[k]) * 0.01
        dpm = float(plev_pa[k]) - float(plev_pa[k + 1])
        if pmid < pin[0]:
            val = prof[:, 0] * pmid / pin[0]
        elif pmid > pin[-1]:
            val = prof[:, -1]
        else:
            ku = max(kk for kk in range(pin.size - 1) if pin[kk] <= pmid)
            dpu, dpl = pmid - pin[ku], pin[ku + 1] - pmid
            val = (prof[:, ku] * dpl + prof[:, ku + 1] * dpu) / (dpl + dpu)
        aod[k] = val * dpm
    r, w, g = band_tables()
    tau = aod @ r.T
    ws = aod @ (r * w).T
    gws = aod @ (r * w * g).T
    asym = np.where(ws != 0, gws / np.where(ws != 0, ws, 1), 0.0)
    ssa = np.where(tau != 0, ws / np.where(tau != 0, tau, 1), 0.0)
    return aod, tau, ssa, asym


_KERNEL_SOURCE = r"""
extern "C" __device__ void tegen_interp_vec(
    const float* v, int n, float want, int periodic,
    int* l1, int* l2, float* w1, float* w2) {
  int a = -1, b = -1; float v1 = 0.0f, v2 = 0.0f;
  for (int k = 0; k < n - 1; ++k) {
    if (v[k] <= want && v[k + 1] > want) { a = k; b = k + 1; v1 = v[k]; v2 = v[k + 1]; break; }
  }
  if (a < 0) {
    if (periodic) {
      a = n - 1; b = 0;
      if (want < v[0]) { v1 = __fsub_rn(v[n - 1], 360.0f); v2 = v[0]; }
      else { v1 = v[n - 1]; v2 = __fadd_rn(v[0], 360.0f); }
    } else {
      if (want < v[0]) { a = b = 0; v1 = v2 = v[0]; }
      else { a = b = n - 1; v1 = v2 = v[n - 1]; }
    }
  }
  const float ww2 = __fdiv_rn(__fsub_rn(want, v1), __fsub_rn(v2, v1));
  *l1 = a; *l2 = b; *w2 = ww2; *w1 = __fsub_rn(1.0f, ww2);
}

// One thread per (column, layer).  table: (6, 46, 12, 72) AOD per Pa at the
// call's date.  Out: band tau/ssa/g (ncol, nlay, 14); optional aod550
// (ncol, nlay), the six types summed.
extern "C" __global__ void tegen_band_optics(
    const float* play, const float* plev, int ncol, int nlay, int nlay_model,
    const float* col_lat, const float* col_lon,
    const float* table, const float* alat, const float* alon, const float* pin,
    const double* rtau, const double* rssa, const double* rasy,
    float* tau, float* ssa, float* asym, float* aod550, int want_aod) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= ncol * nlay) return;
  const int col = idx / nlay;
  const int lay = idx - col * nlay;
  double aod[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (lay < nlay_model) {
    float lon = col_lon[col];
    lon = fmodf(lon, 360.0f); if (lon < 0.0f) lon += 360.0f;
    if (lon > 180.0f) lon -= 360.0f;
    int j1, j2, i1, i2; float wj1, wj2, wi1, wi2;
    tegen_interp_vec(alat, 46, col_lat[col], 1, &j1, &j2, &wj1, &wj2);
    tegen_interp_vec(alon, 72, lon, 1, &i1, &i2, &wi1, &wi2);
    const float pmid = play[idx] * 0.01f;
    const float dpm = plev[col * (nlay + 1) + lay] - plev[col * (nlay + 1) + lay + 1];
    int ku = 0, mode = 1;
    if (pmid < pin[0]) mode = 0;
    else if (pmid > pin[11]) mode = 2;
    else { for (int kk = 0; kk < 11; ++kk) if (pin[kk] <= pmid) ku = kk; }
    for (int s = 0; s < 6; ++s) {
#define TG(K) (wi1 * (wj1 * table[((s * 46 + j1) * 12 + (K)) * 72 + i1]    \
                      + wj2 * table[((s * 46 + j2) * 12 + (K)) * 72 + i1]) \
             + wi2 * (wj1 * table[((s * 46 + j1) * 12 + (K)) * 72 + i2]    \
                      + wj2 * table[((s * 46 + j2) * 12 + (K)) * 72 + i2]))
      float val;
      if (mode == 0) {
        val = TG(0) * pmid / pin[0];
      } else if (mode == 2) {
        val = TG(11);
      } else {
        const float dpu = pmid - pin[ku];
        const float dpl = pin[ku + 1] - pmid;
        val = (TG(ku) * dpl + TG(ku + 1) * dpu) / (dpl + dpu);
      }
#undef TG
      aod[s] = (double)(val * dpm);
    }
  }
  if (want_aod) aod550[idx] = (float)(aod[0] + aod[1] + aod[2] + aod[3] + aod[4] + aod[5]);
  for (int b = 0; b < 14; ++b) {
    double t = 0.0, w = 0.0, g = 0.0;
    for (int s = 0; s < 6; ++s) {
      const double x = rtau[b * 6 + s] * aod[s];
      t += x;
      w += x * rssa[b * 6 + s];
      g += x * rssa[b * 6 + s] * rasy[b * 6 + s];
    }
    if (w != 0.0) g = g / w;
    if (t != 0.0) w = w / t;
    const int o = idx * 14 + b;
    tau[o] = (float)t; ssa[o] = (float)w; asym[o] = (float)g;
  }
}

// rrtmgp_finalize_cloud_sw with a band aerosol term: gas + aerosol + cloud
// (McICA-masked) combined, then the total delta-scaled, as RRTMG spcvmc's
// total sky.  Only reached with aerosol on; the aerosol-off driver runs the
// original kernel.
extern "C" __global__ void rrtmgp_finalize_cloud_aer_sw(
    const float* tau_gas, const float* ssa_gas,
    const float* tau_cloud, const float* ssa_cloud, const float* g_cloud,
    const float* tau_aer, const float* ssa_aer, const float* g_aer,
    const int* gpoint_bands, const unsigned char* cloud_mask,
    float* tau, float* ssa, float* asym,
    int n, int ngpt, int nband, int have_mask) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  const int gpt = idx % ngpt;
  const int cell = idx / ngpt;
  const int band_idx = cell * nband + gpoint_bands[gpt];
  float tc = tau_cloud[band_idx];
  if (have_mask) tc = tc * (float)cloud_mask[idx];
  const float wc = ssa_cloud[band_idx];
  const float gc = g_cloud[band_idx];
  const float ta = tau_aer[band_idx];
  const float wa = ssa_aer[band_idx];
  const float ga = g_aer[band_idx];
  const float floor = 3.0f * 1.17549435e-38f;
  const float total_tau = __fadd_rn(__fadd_rn(tau_gas[idx], ta), tc);
  const float gas_scatter = __fmul_rn(tau_gas[idx], ssa_gas[idx]);
  const float aer_scatter = __fmul_rn(ta, wa);
  const float cloud_scatter = __fmul_rn(tc, wc);
  const float scatter = __fadd_rn(__fadd_rn(gas_scatter, aer_scatter), cloud_scatter);
  const float total_ssa = __fdiv_rn(scatter, fmaxf(floor, total_tau));
  const float gscatter = __fadd_rn(__fmul_rn(aer_scatter, ga), __fmul_rn(cloud_scatter, gc));
  const float total_g = __fdiv_rn(gscatter, fmaxf(floor, scatter));
  const float f = total_g * total_g;
  const float wf = total_ssa * f;
  tau[idx] = (1.0f - wf) * total_tau;
  ssa[idx] = (total_ssa - wf) / fmaxf(floor, 1.0f - wf);
  asym[idx] = (total_g - f) / fmaxf(floor, 1.0f - f);
}
"""


@lru_cache(maxsize=None)
def _module_on(device: int):
    import cupy as cp

    with cp.cuda.Device(device):
        return cp.RawModule(code=_KERNEL_SOURCE, options=("-std=c++17",))


def _kernel(name: str):
    import cupy as cp

    return _module_on(int(cp.cuda.Device().id)).get_function(name)


class TegenAerosol:
    """The driver-side state: the climatology on the card, the call's date
    table, and the columns' latitudes and longitudes."""

    def __init__(self, latitude_deg, longitude_deg, clim: TegenClimatology | None = None):
        import cupy as cp

        self.clim = clim or load_climatology()
        self._lat = cp.ascontiguousarray(cp.asarray(latitude_deg, cp.float32).reshape(-1))
        self._lon = cp.ascontiguousarray(cp.asarray(longitude_deg, cp.float32).reshape(-1))
        self._alat = cp.asarray(self.clim.lat)
        self._alon = cp.asarray(self.clim.lon)
        self._pin = cp.asarray(self.clim.plev_hpa)
        r, w, g = band_tables()
        self._r = cp.asarray(np.ascontiguousarray(r))
        self._w = cp.asarray(np.ascontiguousarray(w))
        self._g = cp.asarray(np.ascontiguousarray(g))
        self._table = None
        self.date = None

    def set_date(self, valid_time: datetime) -> None:
        import cupy as cp

        nm, np_, f1, f2 = time_weights(valid_time)
        self._table = cp.asarray(date_table(self.clim, valid_time))
        self.date = {"months": [nm + 1, np_ + 1], "weights": [f1, f2]}

    def band_optics(self, play, plev, columns, nlay_model: int, *, aod550=False):
        """Band (tau, ssa, g), each (ncol, nlay, 14) float32, for the chunk's
        ``columns`` (a slice or an index array of the driver's packed
        columns); layers at and above ``nlay_model`` get none."""
        import cupy as cp

        if self._table is None:
            raise RuntimeError("the aerosol date table is not set for this call")
        play = cp.ascontiguousarray(cp.asarray(play, cp.float32))
        plev = cp.ascontiguousarray(cp.asarray(plev, cp.float32))
        ncol, nlay = int(play.shape[0]), int(play.shape[1])
        if plev.shape != (ncol, nlay + 1):
            raise ValueError(f"plev must have shape {(ncol, nlay + 1)}, got {plev.shape}")
        lat = cp.ascontiguousarray(self._lat[columns])
        lon = cp.ascontiguousarray(self._lon[columns])
        if lat.size != ncol:
            raise ValueError("aerosol columns do not match the chunk")
        tau = cp.empty((ncol, nlay, NBAND), cp.float32)
        ssa = cp.empty_like(tau)
        asym = cp.empty_like(tau)
        a550 = cp.empty((ncol, nlay) if aod550 else (1,), cp.float32)
        n = ncol * nlay
        threads = 128
        _kernel("tegen_band_optics")(
            ((n + threads - 1) // threads,), (threads,),
            (play, plev, np.int32(ncol), np.int32(nlay), np.int32(nlay_model),
             lat, lon, self._table, self._alat, self._alon, self._pin,
             self._r, self._w, self._g, tau, ssa, asym,
             a550, np.int32(1 if aod550 else 0)))
        return (tau, ssa, asym, a550) if aod550 else (tau, ssa, asym)


def finalize_cloud_aerosol_sw(tables, gas, cloud, aerosol, cloud_mask=None, *, out=None):
    """``rrtmgp._finalize_cloud_optics`` for the shortwave with the band
    aerosol term ``aerosol = (tau, ssa, g)`` added before the delta-scaling."""
    import cupy as cp
    from arwen_global.core import rrtmgp

    tau_gas = cp.ascontiguousarray(cp.asarray(gas.tau, dtype=np.float32))
    if tau_gas.ndim != 3 or tau_gas.shape[2] != tables.ngpt or tables.kind != "sw":
        raise ValueError("aerosol optics need the shortwave gas optics")
    if gas.g is not None:
        raise ValueError("fused SW optics require the zero gas-g sentinel")
    band_shape = (*tau_gas.shape[:2], tables.nband)
    if tables.nband != NBAND:
        raise ValueError(f"the aerosol band map is for {NBAND} SW bands, got {tables.nband}")
    prof = rrtmgp._device_profile
    tau_cloud = prof(cloud.tau, band_shape, "cloud.tau")
    ssa_cloud = prof(cloud.ssa, band_shape, "cloud.ssa")
    g_cloud = prof(cloud.g, band_shape, "cloud.g")
    tau_aer = prof(aerosol[0], band_shape, "aerosol.tau")
    ssa_aer = prof(aerosol[1], band_shape, "aerosol.ssa")
    g_aer = prof(aerosol[2], band_shape, "aerosol.g")
    if cloud_mask is None:
        mask, have_mask = tau_gas, False
    else:
        mask = cp.ascontiguousarray(cp.asarray(cloud_mask, dtype=cp.bool_))
        if mask.shape != tau_gas.shape:
            raise ValueError(f"cloud_mask must have shape {tau_gas.shape}, got {mask.shape}")
        have_mask = True
    ssa_gas = prof(gas.ssa, tau_gas.shape, "gas.ssa")
    if out is None:
        tau, ssa, asym = (cp.empty_like(tau_gas) for _ in range(3))
    else:
        tau, ssa, asym = (
            rrtmgp._workspace_output(value, tau_gas.shape, name)
            for value, name in zip(out, ("optics_tau", "optics_ssa", "optics_g")))
    n = tau_gas.size
    threads = 256
    _kernel("rrtmgp_finalize_cloud_aer_sw")(
        ((n + threads - 1) // threads,), (threads,),
        (tau_gas, ssa_gas, tau_cloud, ssa_cloud, g_cloud, tau_aer, ssa_aer, g_aer,
         tables.to_device().gpoint_bands, mask, tau, ssa, asym, np.int32(n),
         np.int32(tables.ngpt), np.int32(tables.nband), np.int32(have_mask)))
    return rrtmgp.GasOpticsResult(tau, ssa, asym, gas.col_dry)


__all__ = [
    "AEROSOL_CLIMATOLOGIES", "DATA_FILE", "RRTMGP_SW_BAND_LIMITS_CM1",
    "RRTMGP_TO_RRTMG_ROW", "TegenAerosol", "TegenClimatology", "band_tables",
    "column_reference", "date_table", "finalize_cloud_aerosol_sw",
    "interp_vec", "load_climatology", "receipt", "time_weights",
]
