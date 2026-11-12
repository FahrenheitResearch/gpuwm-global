gpuwm-global 0.1.6

Arwen Global 0.1.6 starts the global model from ERA5, for any hour from 1940
to about six days ago, with no key. Until now the only start was a GDAS
analysis, and the GDAS archive the engine reads begins at 2021-03-22 12Z.
It needs `gpuwm>=2.9.13,<2.10` as 0.1.5 did, and the Rust doors are 0.1.5's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- `gpuwm-global fetch-analysis --source era5 --cycle YYYY-MM-DDTHH` fetches
  one whole-globe ERA5 hour from Google's public ARCO store through the
  engine's own ARCO retrieval, with ERA5's snow density beside it: about
  550 MB transferred and 2.3 GB on disk, 39 to 55 s an hour for the hours
  fetched for these notes. It prints the
  `[initial]` table that starts any analysis configuration from that hour:

  ```toml
  [initial]
  mode = "analysis"
  analysis_grib = "data/era5-analysis/2019052000/era5-combined.nc"
  analysis_mapping = "era5-arco-global"
  surface_water_kg_m2 = 500.0
  ```

  A native-physics configuration also needs its `start_time_utc` deleted,
  so the clock follows the ERA5 hour. `docs/ARWEN_GLOBAL_ERA5.md` has the
  route.
- `era5-arco-global` is this package's own NetCDF mapping. The decode is the
  engine's; `arwen_global.era5_analysis` completes what ERA5 lacks against
  GDAS, and the receipt records every rule under `era5_completion`.
- `tools/arwen_global_era5_check.py` checks a start against ERA5 itself:
  `read` reads the hour a second way with zarr-python, `state` writes the
  hour-0 state from a CPU cold start, `hour0` compares them with code that
  imports neither this package nor the engine, and `tapes` compares render
  tapes with ERA5 of their valid times by surface class.

## What ERA5 lacks and what stands in

- **Sea-ice thickness** is not in ERA5. Every sea-ice point starts at 3 m,
  WRF's `seaice_thickness_default`. That is too thick for first-year
  Antarctic ice (about 1 m), and a frozen lake ERA5 carries in its sea-ice
  cover gets 3 m too.
- **Snow depth** is not in ERA5. It is ERA5's snow water divided by ERA5's
  snow density, which the fetch asks for separately.
- **Snow on sea ice**: ERA5 has none, so the pack starts bare.
- **Soil over the sea**: ERA5 writes values there. They are set missing, and
  the cold start's water fill applies.
- **Water temperature**: ERA5's skin on a part-land point is a mix of land
  and water, so an ice-free water point takes ERA5's SST instead.
- **The coast ERA5's sea fields leave out** takes the mean of the sea values
  beside it.
- **Pressure levels** come back in hPa. The cold start now reads the unit
  each mapping declares and refuses any unit but Pa and hPa. Every earlier
  mapping declared Pa, so no earlier run changes.
- **Not read**: cloud, rain and vertical velocity start at zero, as with
  GDAS.

## Measured

Hour 0 against ERA5 read independently, on the model's grid and levels
(`tools/arwen_global_era5_check.py hour0`):

| | T255, 2019-05-20 00Z | T383, 2019-05-20 00Z | T255, 2019-01-15 00Z |
|---|---|---|---|
| temperature RMS by layer, 1 hPa to the ground | 0.01 to 0.16 K | 0.002 to 0.09 K | 0.008 to 0.18 K |
| surface pressure on the model terrain, bias / RMS | 0.004 / 0.13 hPa | 0.0015 / 0.09 hPa | 0.003 / 0.13 hPa |
| Z500, bias / RMS | -1.6 / 2.5 m | -1.7 / 2.5 m | -1.6 / 2.5 m |
| precipitable water, bias / RMS | -0.004 / 0.46 kg m-2 | -0.003 / 0.23 kg m-2 | -0.001 / 0.41 kg m-2 |
| frozen water columns with snow | 0 of 34,140 | 0 of 76,799 | 0 of 29,384 |

Runs under the native suite, gpuwm 2.9.13, every receipt `pass` with every
gate passed and no non-finite value on any tape:

| start | run | wall | card | mass / water drift | at the last lead, against ERA5 |
|---|---|---|---|---|---|
| 2019-05-20 00Z, T255 | 24 h, RTX 5070 Ti 16 GB | 5 min 34 s | 10.79 GiB held | 4.1e-7 / 4.4e-9 | Z500 -4.3 / 7.9 m, T2 -0.14 / 1.5 K |
| 2019-05-20 00Z, T383 | 24 h, RTX 4090, 16 bands, host tier on | 29 min 50 s | 13,090 MiB | 5.1e-7 / 1.0e-8 | Z500 -4.0 / 7.7 m, T2 -0.15 / 1.5 K |
| 2019-01-15 00Z, T255 | 5 days, RTX 4090 | 20 min 38 s | 11,844 MiB | 8.2e-8 / 9.0e-9 | Z500 -1.5 / 45.7 m, T2 -0.20 / 2.9 K |

Over the five January days the global biases stay flat (surface pressure
within 0.04 hPa, T2 within 0.2 K, precipitable water +0.4 to +0.6 kg m-2)
while the RMS grows as a forecast's error does. The bands and the host tier
of the T383 run change memory and not arithmetic. On a 16 GB card with
nothing else on it, nothing has to be set.

## Known limits

- **ERA5's snow is not capped.** ERA5 holds 10 m of water on glaciers, and
  on land the present-day statics call tundra (mostly the Canadian Arctic
  and the margins of Greenland) Noah runs that pack as snow-covered land.
  On the land that starts above 1000 kg m-2 without being land ice, the
  skin ran 4.9 K colder than ERA5's at 24 h in May (4.6 K at T383), and 2.8
  to 6.7 K warmer at every day from 24 h to 120 h in January, where the
  model's own northern land ice ran 3.2 to 4.8 K warm too. From 48 h to
  120 h that land's skin bias rose every day, from +2.8 to +6.7 K, and its
  2 m bias from -0.3 to +2.8 K, while the land ice's fell back. Nothing
  longer than five days was run, so whether it levels off is not known. No
  cap is applied. WRF's Noah-MP caps snow water at 2000 kg m-2 when it
  starts; no capped run was made.
- **The ice edge.** Open water beside the pack keeps ERA5's skin where the
  ERA5 point has any ice. On the January start 160 open-water columns began
  more than 1 K from ERA5's SST, 15 of them more than 5 K.
- **The T383 start tripped one in-situ wire.** `spectral_top_decile_growth`
  recorded six trips between hours 5 and 9 at levels 13 and 14, peaking at
  2.88 against 2.1. The wire never refuses, and the band held level from
  hour 10. The T255 ERA5 starts and three GDAS T383 starts on record tripped
  none. With one date of each, it is not known whether ERA5 or the case
  makes the difference.
- **Not measured.** No ERA5 start was compared with a GDAS start of the same
  valid time, because the GDAS route begins at 2021-03-22 12Z. Nothing
  longer than five days was run, or longer than 24 h at T383.
- **The statics are present-day** whatever the date: a 1950 start gets
  2020s MODIS land use. The monthly climatologies do follow the valid date.
- **ERA5T.** The last few months of the store are ECMWF's preliminary
  release, which may still be revised. The fetch says so.
- **Your own CDS downloads** usually lack sea-ice cover, surface geopotential
  and snow density, and carry relative humidity, so they cannot start this
  model as downloaded. The ARCO route carries all of them and needs no key.

## Fixed

- The cold start reads its analysis levels in the unit the mapping declares,
  where it used to take them as Pa unread. An hPa source would have put
  every model level below a "1000 Pa" bottom level with no refusal.
- A NetCDF mapping's soil layers reach Noah at their depths. They used to be
  refused for want of a GRIB level type.

## Unchanged

- The engine range, `gpuwm>=2.9.13,<2.10`, and the 48 pinned seam files.
  `tools/measure_boundary.py` reads 240 symbols across 71 modules with no
  gap.
- The ten Rust doors. `gpuwm-global fetch-doors` stages the bundle 0.1.3
  published.
- The GDAS start's rules. Every shipped mapping declares Pa, which the units
  guard leaves as it is, and the seeding keeps its GDAS rule for snow on sea
  ice; only a frame that says it carries no snow on its pack, as ERA5's
  does, takes the land-only rule.
