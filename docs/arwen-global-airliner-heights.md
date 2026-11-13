# Airliner heights: the broadcast altitude pair as an upper-air observation (experimental)

Every ADS-B transponder broadcasts two altitudes. The barometric one is
pressure altitude: the static pressure the aircraft measures, named as the
height of that pressure in the ICAO standard atmosphere, always on the
1013.25 hPa datum. The geometric one is the aircraft's GNSS height above
the WGS84 ellipsoid. Together they say how high one pressure surface is at
one place and time, which is the quantity an analysis calls the geopotential
height at a pressure level. Open archives publish every trace their
receivers heard, every day, with no account.

This page is the observation operator, the tool that runs it, and what one
day of data measured against a reference model and radiosondes. Nothing is
assimilated and nothing in the package changed: the tool lives in `tools/`
and imports only numpy (matplotlib for the pictures).

## The operator

For each report:

1. **Pressure** from pressure altitude by the exact ICAO standard-atmosphere
   inversion, layer by layer to 32 km (`isa_pressure_pa`). FL350 is
   238.4 hPa, FL390 196.8 hPa.
2. **Height above mean sea level**: the GNSS height minus the EGM96 geoid
   undulation at the aircraft (a 15 arc-minute grid read bilinearly; the
   undulation runs from -107 m to +85 m, larger than the signal).
3. **Geopotential height** through WGS84 normal gravity and the latitude's
   effective radius:
   `Z = g(phi) / g0 * R(phi) * H / (R(phi) + H)`, with `g0 = 9.80665 m s-2`,
   Somigliana's `g(phi)` and `R(phi) = a / (1 + f + m - 2 f sin^2 phi)`.
   At 11 km this is 58 m larger at the pole than at the equator.

The model side reads geopotential height and temperature on isobaric levels:
bilinear in the horizontal; in the vertical, temperature linear in ln p
between the two levels that bracket the report, with the hypsometric
residual against the upper level's own height spread linearly in ln p, so
both levels are met exactly; linear in time between hourly products.

`tests/test_arwen_global_airliner_heights_tool.py` holds all of it against
known answers on the CPU: published flight-level pressures and the
inversion round trip to 30 km; the 1976 US Standard Atmosphere pair and
WGS84 free-air gravity integrated numerically, at three latitudes and three
heights, to 0.1 m; an aircraft in a standard atmosphere with a planted
geoid reads its pressure altitude back to 0.01 m; the vertical read is exact
for isothermal and log-linear layers and within 0.5 m for the standard
troposphere on 50 hPa levels; GRIB2 messages built in the test with simple
packing and with complex packing and first- and second-order spatial
differencing decode to the packing quantum; every rejection cause is counted
on a planted trace, and a split tar reads across the split. The GRIB2
reader was also compared with ecCodes on one GFS 0.25 degree product: the
same values to float32 rounding (largest difference 0.0005 m in height).

## The tool

```bash
# one day of the archive: both split parts of one daily release, in order
python tools/arwen_global_airliner_heights.py extract \
  --parts v2026.11.10-planes-readsb-prod-0.tar.aa v2026.11.10-planes-readsb-prod-0.tar.ab \
  --day 2026-11-10 --out airliner/rows-20261110.npz

# the geoid grid the PROJ data CDN publishes (needs tifffile and imagecodecs)
python tools/arwen_global_airliner_heights.py geoid --tif us_nga_egm96_15.tif --out egm96_15.npz

# the operator against hourly reference products, a longer-lead forecast
# valid at the same hours, and IGRA2 station files
python tools/arwen_global_airliner_heights.py score \
  --rows airliner/rows-20261110.npz --geoid egm96_15.npz --day 2026-11-10 \
  --reference REFERENCE_GRIB2... --forecast FORECAST_GRIB2... --sondes IGRA2_DIR \
  --cache-dir airliner/cache --out airliner/score
```

`extract` streams the tar without unpacking it and keeps one report per
aircraft per 60 s at or above 5,000 ft pressure altitude, from the
aircraft's own ADS-B transmissions (`adsb_icao`) with both altitudes
present. It keeps the 24-bit address as a number, the type designator and
the ADS-B quality fields, and no registration, owner or flight number. Its
JSON record counts every point it dropped by cause. `score` writes
`airliner-heights-stats.json`, `airliner-coverage-map.png` and
`airliner-error-statistics.png`.

Reference products need geopotential height and temperature on isobaric
levels in GRIB2 (grid 3.0, product 4.0, data 5.0, 5.2 or 5.3, no bitmap),
which is what the GFS 0.25 degree `pgrb2` files carry; the products are
given in increasing valid time, one hour apart.

## One day measured: 2026-11-10

Inputs: adsb.lol's `globe_history_2026` release `v2026.11.10-planes-readsb-prod-0`
(3.5 GiB, 75,360 aircraft traces, 102.9 million trace points); GFS 0.25
degree heights and temperatures on 16 levels from 850 to 100 hPa, hourly as
the reference (the analyses at 00, 06, 12 and 18Z and the 1 to 5 h
forecasts between them, to 2026-11-11 00Z) and the forecast from
2026-11-08 00Z at 48 to 72 h as the background; IGRA2 station files for
2026; EGM96. On a shared 24-core CPU host, no card, the extraction took
6 min and the score 7 min with the 50 products already decoded into the
cache (decoding them takes about 25 s each the first time).

**Volume.** 6.64 million reports kept from 42,666 aircraft; 6.17 million
from 39,415 aircraft pass the score's checks (150 to 850 hPa, bracketed by
two reference hours, ADS-B version 1 or 2, not a 100 ft altitude encoder,
within 300 m of the median). Coverage is North America, Europe, the Gulf,
India, east and south-east Asia, Australia, New Zealand, Brazil and South
Africa; the oceans and most of Africa are empty (the map).

**Single reports against the reference** (aircraft minus reference, m):

| reports | n | mean | std | robust std |
|---|---|---|---|---|
| all used, 150 to 850 hPa | 6,173,437 | +9.5 | 14.3 | 12.5 |
| level flight, at or above 300 hPa | 2,903,410 | +10.0 | 13.5 | 11.5 |
| 700 to 850 hPa | 905,960 | +3.7 | 13.7 | 12.0 |
| ADS-B version 0, at or above 350 hPa (left out) | 149,946 | +6.0 | 24.4 | 17.6 |

**Airframe bias, out of sample.** Each airframe's mean over its level
cruise reports at or above 350 hPa before 18Z, applied to its reports after
18Z, which the fit never saw:

| | n | mean | std | robust std |
|---|---|---|---|---|
| after 18Z, raw | 783,933 | +9.6 | 13.1 | 11.5 |
| after 18Z, airframe bias removed | 783,933 | -0.5 | 8.8 | 6.3 |
| the 20,822 fitted biases | | +10.7 | 12.7 | 10.8 |

87 percent of the post-18Z cruise reports came from an airframe with ten or
more fitted reports. The airframe means of the two halves of the day agree
with correlation 0.83 (12,967 airframes). A flight fitted before 18Z and
scored minutes after it is the easy case, so the split was also run with a
six-hour gap (`--split-hour 12 --gap-hours 6`: fitted before 12Z, scored
after 18Z): -0.5 m with std 8.9 m (robust 6.7 m) over 641,613 reports, 72
percent of them from a fitted airframe, and 4.1 m for super-observations of
five or more aircraft. The bias holds across six hours.

**Super-observations** (one value per 1-degree cell, 10 percent pressure
layer and hour: each aircraft's mean, then the mean over aircraft), airframe
bias removed, after 18Z: std 5.0 m with three or more aircraft (26,344),
3.9 m with five or more (13,116), 3.1 m with ten or more (3,393).

**Residuals of different aircraft against separation** (hour means, bias
removed, after 18Z): correlation 0.10 to 0.15 within 50 km, 0.07 at 100 to
200 km, 0.05 at 200 to 800 km, 0.01 beyond 1,600 km. The common part this
leaves for the satellite height error and the reference's own error
together is about 2 m (0.05 of a 74 m2 variance); one day cannot separate
the two.

**The information test.** Aircraft minus background against reference
minus background, where the background is the 48 to 72 h forecast: slope
1.02 and correlation 0.93 for single reports with the airframe bias removed
(783,933; the reference moved 21.9 m from the background as a standard
deviation); slope 1.02 and correlation 0.99 for super-observations of five
or more aircraft; slope 1.03 and correlation 0.79 for single raw reports. A
report would have moved a background of this quality in the direction and
by the amount the reference did.

**The height reference frame.** For 8,057 airframes whose cruise crossed
20 m or more of geoid undulation, the residual's slope against the
undulation is near 0 for 6,819 (heights above the ellipsoid, as ADS-B
version 1 and 2 define them) and near -1 for 53 (heights above the geoid
broadcast as geometric). The second group needs a flag per airframe before
any use.

**Radiosondes.** 1,158 collocations at 200, 250 and 300 hPa within 150 km
and one hour of a 00Z or 12Z sounding and 15 hPa of the level. Against the
reference, the aircraft near a sounding read +10.3 m (raw, no airframe
correction) and the sondes +8.7 m; aircraft minus sonde through the
reference is +1.6 m with std 14.3 m (-1.5 m at 200 hPa, +2.5 m at 250 hPa,
+4.6 m at 300 hPa). So most of the aircraft's +10 m offset against the
reference is shared by the sondes on this day; the airframes' fleet mean
is a few metres at most against sondes. The aircraft and sonde
departures are uncorrelated (0.003), so at the analysis hours the
reference's error at these places is small against either instrument's.
With each airframe's bias removed (fitted before 18Z, which includes these
00Z and 12Z reports) the aircraft near a sounding read -0.5 m with std
3.9 m against the reference, while the sondes keep +8.6 m with std 11.3 m:
at these levels the sondes scatter about the reference three times as much
as the corrected aircraft do. Their correlation is 0.06.

**Layer-mean temperature from climbs and descents.** Two reports of one
aircraft in a single climb or descent, the second the first to reach 0.18
in ln p (about 100 hPa at 550 hPa) within 20 min, give the thickness of the
layer between them and so its mean virtual temperature; an airframe's
constant bias cancels. 1,661,969 layers from 30,156 aircraft: aircraft
minus reference +0.11 K with std 1.37 K (robust 1.14 K), the same from 150
to 850 hPa (+0.33 K at the bottom, -0.31 K at the top). Super-observations
(1-degree cell, hour, 10 percent layer) of five or more aircraft: std 0.60 K
(77,751). Against the 48 to 72 h background: slope 0.97 and correlation
0.54 single, slope 0.99 and correlation 0.81 for the super-observations;
the background's layer temperatures were already within 0.9 K of the
reference as a standard deviation, so there was less to correct than in
height. This is the vertical-difference method of de Haan's 2026 preprint,
which reports 0.5 to 0.75 K for multi-aircraft 500 m layers against ECMWF
from a commercial feed; here it runs on the open archive.

**By hour.** The fleet's raw cruise mean moves between +7.8 m and +12.1 m
through the day. Which region flies at which hour changes with it, so this
is not a reading of a local-time GNSS error.

## Limits

- One day, one archive, one reference model. The airframe biases are fitted
  against the reference, so every bias-removed figure is relative to it:
  anchoring a fleet mean on a model makes the stream depend on that model.
- Coverage is where receivers are: land and coasts, where sondes already
  are. The gain is in time (hourly, against twice a day) and density.
- The residual map after 18Z has regional structure (Alaska 10 to 15 m
  low against the reference, the western United States a few metres
  high). One day cannot say whether it is the reference or a shared
  satellite-height error.
- No height variable exists in the observation table, so no stream entry,
  super-observation writer or assimilation is built. Those are the next
  steps, and the cycle would gain from them once it runs away from GDAS.
- The reference did not use these heights as far as is known; this was not
  checked.

## Data

Contains information from ADSB.lol (https://www.adsb.lol), made available
under the Open Database License (ODbL) 1.0. The statistics and pictures
the tool writes are produced works; a database of extracted rows that is
published carries the ODbL's share-alike terms. Geoid: EGM96 (NGA), as the
PROJ data CDN publishes it (`us_nga_egm96_15.tif`). Radiosondes: IGRA2
(NOAA NCEI). Reference and background: NCEP GFS from NOAA's open-data
archive.
