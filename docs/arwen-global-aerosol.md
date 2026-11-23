# Aerosol in the shortwave: `radiation_aerosol`

```toml
[physics]
native_adapter_options = { ..., radiation_aerosol = "tegen-wrf" }
```

| value | what the shortwave sees | basis |
|---|---|---|
| `"none"` (default) | gas and cloud, no aerosol | WRF `aer_opt = 0`; what this package always ran, bit for bit |
| `"tegen-wrf"` | gas, cloud and a monthly six-type aerosol climatology | WRF `aer_opt = 1`: Tegen et al. (1997) as WRF ships it, through RRTMG's ECMWF aerosol optics |

The longwave takes no aerosol under either value, as in WRF.

## What "tegen-wrf" does

The code is `arwen_global/physics/aerosol_climatology.py`; the driver hook is
`RRTMGPRadiation(aerosol_climatology=...)` in `core/rrtmgp.py`
(`docs/CARRIED-PHYSICS-DIVERGENCE.md`, row `RRTMGP-30`).

- **Data.** `arwen_global/data/aerosol-tegen-wrf.npz` is WRF v4.6.1's
  `run/aerosol.formatted`, `aerosol_lat.formatted`, `aerosol_lon.formatted`
  and `aerosol_plev.formatted`, values unchanged: optical depth at 550 nm
  per pascal for organic carbon, sea salt, dust, black carbon, sulfate and
  stratospheric aerosol, on 12 pressure levels (20-959 hPa), 46 latitudes,
  72 longitudes and 12 months. WRF is in the public domain; the receipt
  beside the file (`aerosol-tegen-wrf-receipt.json`) records the source
  hashes, the licence and the conversion. Its annual global mean column
  optical depth at 550 nm is 0.088; in March it is 0.25 over the Sahara,
  0.34 over Arabia and 0.08 over the open equatorial Pacific.
- **Interpolation.** As WRF: `interp_vec` bilinear in latitude and
  longitude, `aer_time_int` linear between mid-month days of a 365-day
  year, `aer_p_int` linear in pressure inside the 12 levels (`A(1) p/p(1)`
  above, the bottom value below) times the layer's pressure thickness. The
  layers the driver adds above the model top carry none.
- **Optics.** RRTMG SW `iaer = 6`: per band, the six types' optical depths
  times ECMWF's band-to-550 nm ratios, single-scattering albedos and
  asymmetry parameters (`swaerpr`), WRF's type order. RRTMGP's 14 bands are
  RRTMG's 14 with the 820-2680 cm-1 band first; the map is pinned against
  the shipped k-distribution's limits.
- **Combination.** Aerosol, gas and the McICA-masked cloud are summed and
  the total delta-scaled, as RRTMG's `spcvmc` forms its total sky. Under
  `"none"` the original kernel runs, so nothing moves.
- **Cost.** One kernel per shortwave chunk on the sunlit columns, and one
  (6, 46, 12, 72) float32 table uploaded per radiation call.

## Why

In every round-10 spring run the model's all-sky surface sunshine over
subtropical land equalled ERA5's clear-sky value, 30-42 W/m2 above ERA5's
all-sky value, and the model had no aerosol at all.

## Measurements

T127 L40, dt 600 s, from ERA5 1976-03-01 00Z for 10 days, round 10's
long-run recipe with the anomaly ocean, an RTX 4090, tree `f122835`
(this option). Every radiation call was doubled by a clouds-off call on the
same state (condensate zeroed, nothing else), whose result the model did not
use, so the clouds-off flux is the model's own clear sky on its own air.
ERA5 is the same days' 3-hourly fluxes at 1.5 deg. Land and ocean 15-30N
are cos-latitude means; days 03-01 to 03-09.

**The cause.** With no aerosol the model's clear sky over subtropical land is
too bright from the first day, when its air is still ERA5's:

| land 15-30N, surface SW down (W/m2), model minus ERA5 | 03-01 | 03-01..03-09 |
|---|---|---|
| clouds off against ERA5 clear sky, `"none"` | +22.4 | +25.2 |
| clouds off against ERA5 clear sky, `"tegen-wrf"` | +5.5 | +7.9 |
| all sky against ERA5 all sky, `"none"` | | +26.2 (03-02..03-09: +31) |
| all sky against ERA5 all sky, `"tegen-wrf"` | | +10.6 (03-02..03-09: +15) |

- The same pair from ERA5 2019-03-01 says the same: land 15-30N clear sky
  +24.9 to +7.5 W/m2 (day one +22.8 to +6.0), all sky +25.9 to +9.9.
- Ocean 15-30N clear sky: +15.0 to +4.6 W/m2 (2019: +14.2 to +3.8).
- Points, clear sky, 9-day mean: Sahara (22N 10E) +23.6 to +3.7, Arabia
  (22N 45E) +39.5 to +15.3, Thar (26N 72E) +24.0 to +5.5, northern Mexico
  (26N 105W) +20.9 to +16.0 W/m2 (Tegen has little aerosol there in March,
  0.10).
- The aerosol takes about two thirds of the clear-sky excess and half of the
  all-sky excess. What is left of the all-sky excess (+15 W/m2) is cloud;
  what is left of the clear-sky excess grows from +5.5 to +8.4 W/m2 over the
  nine days as the model's air dries (round 10's 1-2 g/kg).
- Land 15-30N 2 m temperature, 03-02..03-10: -1.26 K to -1.57 K against
  ERA5 (this early-March window is already cold there), sensible heat
  +15.5 to +9.6 W/m2 too large.
- Plots (Solar's runner, Rust backend), 1976-03-10 18Z, both arms: surface
  temperature and dewpoint, CONUS and global. No grid-scale noise; the
  synoptic patterns differ by day 10 as weather does.

**Forecasts.** The shipped T255 SL-SI config (`arwen_global_gdas_t255_native_sl_si_24h.toml`)
to 48 h from GDAS 2026-09-18 00Z and 2026-11-04 12Z, scored against the GFS
analysis at 1.5 deg (cos-latitude; T850 and MSLP where the GFS terrain is
under 1000 m), an RTX 4090, mean of the two starts, `"none"` to
`"tegen-wrf"` (starts better):

| row | 24 h | 48 h |
|---|---|---|
| NH Z500 RMSE (m) | 9.017 to 8.958 (2/2) | 16.625 to 16.619 (1/2) |
| SH Z500 RMSE (m) | 10.607 to 10.552 (2/2) | 21.925 to 21.938 (1/2) |
| tropical Z500 bias (m) | -1.310 to -0.875 (2/2) | -1.718 to -0.957 (2/2) |
| tropical Z500 RMSE, bias removed (m) | 4.411 to 4.500 (0/2) | 6.254 to 6.402 (0/2) |
| T850 RMSE NH / SH / tropics (K) | 0.9833 to 0.9837 / 0.9720 to 0.9745 / 1.0065 to 1.0036 | 1.3650 to 1.3661 / 1.4717 to 1.4788 / 1.1650 to 1.1641 |
| MSLP RMSE NH / SH / tropics (hPa) | 1.054 to 1.060 / 1.235 to 1.236 / 0.683 to 0.687 | 1.907 to 1.915 / 2.417 to 2.429 / 0.953 to 0.963 |
| 250 hPa vector wind RMSE NH / SH / tropics (m/s) | 3.989 to 3.992 / 4.587 to 4.578 / 4.094 to 4.098 | 6.197 to 6.208 / 7.222 to 7.211 / 5.366 to 5.356 |

Level overall. The absorbing aerosol warms the tropical column a little: the
tropical Z500 bias closes by 0.4-0.8 m in both starts, while the tropical
Z500 RMSE with the bias removed is worse in both (by 0.09 m at 24 h and
0.15 m at 48 h on the mean). Every other row
moves by hundredths (Z500 0.01-0.06 m, T850 under 0.01 K, MSLP under
0.015 hPa), with no sign the two starts agree on. Step wall over the 576
steps: 395 and 430 s with the option against 411 and 419 s without, so no
cost is measurable.

**Default.** Off. The sunshine fix is measured in two years and the
forecasts are level, but the error it corrects matters most in long runs,
where less subtropical land heating also feeds the 15-30N temperature and
the jet; it becomes the default once a 75-day pair (1976 and 2019, round 10's
recipe) measures that.
