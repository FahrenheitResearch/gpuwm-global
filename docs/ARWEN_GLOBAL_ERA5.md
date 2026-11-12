# Arwen Global from ERA5: a start on any past hour

The global model cold-starts from one analysis. Until now that was a GDAS
0.25-degree analysis only, and the GDAS archive the engine's fetch route
reads begins at 2021-03-22 12Z and holds the four synoptic hours. ERA5
covers every hour from 1940 to about six days ago, so a run can start on a
date in another season, on a case from the past, or at 03Z. This page is
the route and what it does and does not carry.

## The route

```bash
gpuwm-global fetch-analysis --source era5 --cycle 2019-05-20T00
cp "$(gpuwm-global configs --paths | grep t255_quickstart)" era5_t255.toml
# edit era5_t255.toml: replace its [initial] table with the one the fetch printed
gpuwm-global run era5_t255.toml --outdir out/arwen-global-era5
gpuwm-global export era5_t255.toml \
  out/arwen-global-era5/arwen_global_step*.npz --outdir out/arwen-global-era5-tapes
```

The `[initial]` table the fetch prints, for this hour:

```toml
[initial]
mode = "analysis"
analysis_grib = "data/era5-analysis/2019052000/era5-combined.nc"
analysis_mapping = "era5-arco-global"
surface_water_kg_m2 = 500.0
```

- **No key.** The data come from Google's public ARCO ERA5 store
  (`gcp-public-data-arco-era5`), read by the engine's own ARCO retrieval
  (`gpuwm.era5_arco`) through the `rw_zarr` program in the engine's bridge
  bundle. No CDS account is used or needed.
- **What lands.** `data/era5-analysis/YYYYMMDDHH/era5-combined.nc` (the
  engine's file, with its `era5-arco-acquisition.json` receipt) and
  `era5-snow-density.nc` beside it. About 550 MB is transferred and 2.3 GB
  sits on disk per hour (float64 NetCDF). Each hour gets its own directory,
  because the engine's retrieval refuses to write a second hour over a
  verified first one.
- **The configuration.** Any analysis configuration starts from ERA5 when
  its `[initial]` names the ERA5 file and `analysis_mapping =
  "era5-arco-global"`; the T255 quickstart (`arwen_global_t255_quickstart`)
  is the one measured below. No ERA5 experiment ships, because `[initial]`
  names one hour's file and a shipped copy would name an hour nobody
  fetched. `era5-snow-density.nc` must stay beside the analysis. No date is
  typed anywhere else: the forecast clock, the sun and every tape's valid
  time are the analysis's own valid time. The export reads the analysis
  again, so keep the file until the tapes are written.
- **A native-physics configuration** states its own date,
  `start_time_utc` inside `native_adapter_options`, and the run is refused
  when that date is not the analysis's. Delete `start_time_utc` from the
  copy as well as replacing `[initial]`; the clock then follows the ERA5
  hour. The second start measured below is one.
- **ERA5T.** The last few months of the store are ERA5T, ECMWF's preliminary
  release, which may still be revised. The fetch says so when the hour is
  one of them.

`arwen_global.era5_analysis` is the code. The mapping is this package's own,
`src/arwen_global/data/package-mappings/rw-wps-era5-arco-global-analysis-netcdf.mapping.json`:
no engine release carries an ERA5 global mapping, and the decode itself is
the engine's.

## What already worked for a past date: GDAS

`gpuwm-global fetch-analysis --cycle YYYY-MM-DDTHH` (the default
`--source gdas`) fetches any cycle the engine's route can reach:

- NOMADS for the last ten days, then the AWS Open Data archive
  `noaa-gfs-bdp-pds`;
- **back to 2021-03-22 12Z**, the first cycle filed under the archive's
  `atmos/` folder, which is the only layout the engine looks in;
- **at 00, 06, 12 and 18Z only.**

For an earlier cycle the engine's readiness probe answers "waiting" rather
than refusing (measured for 2021-03-22 06Z). The archive does hold some older
0.25-degree analyses in the older folder layout, but not every day: there is
none for 2019-05-20 00Z or 2019-05-21 00Z. ERA5 is the input for those
dates.

## What ERA5 lacks against GDAS, and what stands in

The decode gives the cold start the same names it reads from GDAS. Each rule
below is applied by `era5_analysis` after the decode and is recorded in the
run receipt under `initial.provenance.era5_completion`.

| Cold-start field | ERA5 source (bridge name) | Rule |
|---|---|---|
| pressure levels | 37 levels, 1 to 1000 hPa (`level`) | read in hPa and converted to Pa (see below) |
| temperature, winds | `T`, `U`, `V` | as analysed |
| humidity | specific humidity `SPFH` (kg/kg) | as analysed; the same convention GDAS uses |
| surface pressure | `PSFC` (Pa) | as analysed |
| terrain | surface geopotential `SOILGEO` | divided by 9.80665 |
| skin temperature | `SKINTEMP`, and `SST` on water | as analysed on land and on ice; **an ice-free water point takes ERA5's SST** (see below) |
| land fraction | land-sea mask `LANDSEA` | land at 0.5 and above; the real statics' land fraction replaces it on the model grid |
| soil temperature, soil water | `ST000007` ... `ST100289`, `SM000007` ... `SM100289` | **set missing on water points**, then the cold start's water fill applies; the 0-7, 7-28, 28-100 and 100-289 cm layers reach Noah's 0-10, 10-40, 40-100 and 100-200 cm by depth |
| sea-ice fraction | `SEAICE`, missing on land and on some coast | **0 on land; a coast water point without a value takes the mean of the sea values beside it** (see below) |
| sea-ice thickness | **not in ERA5** | **3.0 m on every sea-ice point** (WRF's `seaice_thickness_default`) |
| snow water | `SNOW_EC` (m of water) | times 1000, to kg m-2 |
| snow depth | **not in ERA5**; ERA5 has snow density | **snow water divided by ERA5's snow density** (the fetch's second request) |
| snow on sea ice | none in ERA5 | the pack starts bare: snow is averaged over land points only and no frozen water column takes any (see below) |
| cloud, rain, vertical velocity | not read | zero, as with GDAS |

Five of these are traps that would have given a wrong start with no error:

- **Pressure units.** The engine returns ERA5's levels in hPa and GDAS's in
  Pa, and the cold start took them as Pa without checking. It now reads the
  unit each mapping declares, refuses any unit but Pa and hPa, and refuses a
  ladder whose deepest level reads above 500 hPa. Every earlier mapping
  declared Pa, so no earlier run changes.
- **Soil over the sea.** ERA5 writes soil values over water (soil water near
  0, down to -0.013). The engine's `landmask_water` policy would mask them,
  but gpuwm 2.9.13 applies it to GRIB only and ignores it on NetCDF, so the
  ERA5 step masks them itself. Unmasked, every coastal land column would mix
  ocean "soil" into its regrid.
- **Water temperature.** The cold start reads the skin temperature of the
  analysis's water points as the water temperature it holds for the whole
  run. GDAS's land mask is 0 or 1, so its skin on a water point is water.
  ERA5's mask is a fraction and its skin is the mean over the point's land
  and water, so a point that is three tenths land carries three tenths
  ground in its skin. Read as water, that started 1,747 open-water columns
  more than 1 K away from ERA5's own SST on the start below, 14 of them
  more than 5 K, and the sea as a whole 0.17 K cold (ERA5's skin is the
  cool skin, not the water under it). An ice-free ERA5 water point now
  takes ERA5's SST; a point with ice keeps the skin, which is the mix of ice
  and water the frozen column reads.
- **The coast the sea fields leave out.** ERA5 gives no sea-ice cover and no
  SST on land, and its sea mask is not its land-sea mask at 0.5: on the
  start below 2,900 points with land fraction 0.19 to 0.50 have neither.
  They are coast, not lakes (mean lake cover 0.07; ERA5's real lakes carry a
  sea-ice value). Each takes the mean of the sea values beside it. ERA5's
  lake ice depth is defined at every point, lake or not, so it is not read
  there: read as "frozen", it froze 147 coast points beside open sea (west
  Svalbard). It answers only for a water point no sea value reaches within
  four cells, and there were none.
- **Snow on the pack.** ERA5 has no snow on its sea-ice points and 10 m of
  water on the glacier beside them. The cold start's GDAS rule averages
  snow over land and sea-ice points together, which put the glacier's snow
  on 2,336 sea-ice columns, up to 8,178 kg m-2 (31 m deep), and thinned the
  coastal land's snow with the bare ice. From ERA5 snow is averaged over
  land points only and every frozen water column starts bare.

Stated limits of the stand-ins:

- **3 m of sea ice** is too thick for first-year Antarctic ice (about 1 m),
  which slows heat conduction through the pack.
- **A frozen lake gets 3 m too.** ERA5 carries lake ice in its sea-ice
  cover (Great Bear Lake reads 1.0 on the start below, with 1.2 m of lake
  ice), and the engine's file has no lake cover to tell a lake from the
  sea, so a frozen lake starts with the sea's 3 m.
- **Open water beside the pack** keeps ERA5's skin where the ERA5 point has
  any ice, so an open-water column at the ice edge can start below the
  freezing point of sea water: 103 columns were still more than 1 K from
  ERA5's SST on the May start below, the largest 4.0 K, and on the January
  start, whose ice edge is longer, 160, 15 of them more than 5 K and the
  largest 7.1 K.
- **Soil water is ERA5's volume fraction, unscaled.** ERA5's soil model has
  its own soil types (saturation up to 0.77 on peat); Noah's are different
  and nothing rescales between them, which is what WRF does with the same
  fields.
- **ERA5's snow is not capped.** ERA5 holds 10 m of water on glaciers and
  deep packs on some high-latitude and mountain land; GDAS holds far less.
  Most land that starts above 1000 kg m-2 without being land ice in the
  statics is ERA5 glacier that the present-day land use calls tundra: on
  2019-01-15, 348 of its 458 tape columns are tundra, most of them in the
  Canadian Arctic and on the margins of Greenland, with a median of
  3,150 kg m-2. Noah runs those columns as snow-covered land. Over the
  native starts below their skin ran 4.9 K colder than ERA5's at hour 24
  in May and 2.8 to 6.7 K warmer at every day from hour 24 to hour 120 in
  January, with their 2 m temperature 0.3 K colder to 2.8 K warmer. The
  model's own land ice north of 55N runs 3.2 to 4.8 K warm over the same
  days of the same run, and the terrain
  explains 0.2 K of it. From hour 48 to hour 120 the deep-snow land's skin
  bias rose every day, from +2.8 to +6.7 K, and its 2 m bias from -0.3 to
  +2.8 K, while the land ice's fell back from +4.8 to +3.2 K. Nothing
  longer than five days was run, so whether it levels off is not known.
  No cap is applied. WRF's Noah-MP caps snow water at 2000 kg m-2 when it
  starts; on these columns that would still leave about 7 m of snow at
  ERA5's density, and no capped run was made to show it would change the
  skin. This model runs Noah and keeps ERA5's value.
- **The statics are present-day** whatever the date: a 1950 start gets
  2020s MODIS land use. The monthly climatologies do follow the valid date.

## Measured starts

Four starts, all before the GDAS archive begins, with gpuwm 2.9.13. The
first two start from ERA5 2019-05-20 00Z on one RTX 5070 Ti 16 GB with a
`[statics] cache_dir` naming an existing T255 cache, from the same file and
the same cold start. The third is the same hour at T383 and the fourth a
five-day winter run from 2019-01-15 00Z, both on an RTX 4090 24 GB shared
with other jobs.

- **Fetch:** 36.5 s wall for the analysis and the snow density together.
- **Cold start alone**, run once on the CPU with no steps: 58 s.
- **Completion counts:**
  - 348,976 land points had no sea-ice value;
  - 2,900 coast water points had none either, every one was filled from the
    sea beside it, 1,270 of them with some ice, and no water point was out
    of reach;
  - 555,737 ice-free water points took ERA5's SST, 0.16 K above their skin
    on the mean and more than 1 K away from it on 6,987;
  - every one of 180,436 snowy points had a density;
  - 781 frozen water columns of 34,140 had land snow in their stencil and
    start bare.
- **Soil water floor:** 1,201 land-ice cells were set saturated and 1,997
  soil cells were set to their dry limit.

Hour 0 was checked against ERA5 read a second way, with zarr-python straight
from the store, sampled and interpolated by code that is not this
package's. The engine's file holds the store's values bit for bit. The
check is `tools/arwen_global_era5_check.py`, and these three lines rerun it
for any ERA5 configuration (`read` needs `pip install zarr fsspec aiohttp`,
which this package does not depend on; `state` runs the cold start once on
the CPU, with no step and no card):

```bash
python tools/arwen_global_era5_check.py read 2019-05-20T00 era5-2019052000.npz
python tools/arwen_global_era5_check.py state era5_t255.toml state-2019052000.npz
python tools/arwen_global_era5_check.py hour0 state-2019052000.npz era5-2019052000.npz
```

`hour0` imports neither this package nor the engine: the sampling onto the
model's columns, the interpolation in ln p onto its levels, the move of
ERA5's surface pressure onto the model terrain, the hydrostatic heights and
the depth overlap of the soil layers are written again in the tool. On the
model's own grid and levels:

| at hour 0, model state minus ERA5 | bias | RMS |
|---|---|---|
| temperature, 1 hPa to the ground, by layer | under 0.0002 K | 0.01 to 0.16 K |
| specific humidity, 700 hPa to the ground | 0.0003 g/kg | 0.15 g/kg |
| wind components, 700 hPa to the ground | under 0.001 m/s | 0.3 m/s |
| surface pressure, ERA5's moved to the model terrain | 0.004 hPa | 0.13 hPa |
| terrain height (T255 truncation of ERA5's) | -0.01 m | 41 m |
| precipitable water, against ERA5's own column total | -0.004 kg m-2 | 0.46 kg m-2 |
| open-water skin, against ERA5's SST | 0.0001 K | 0.007 K |
| sea-ice fraction, away from the coast | 0 | 4e-9 |
| snow water and depth, interior land | 0 | 1e-5 kg m-2 |
| soil temperature, four Noah layers, interior snow-free land, land ice left out | 0 | 2e-5 K |
| soil water, Noah layers 1, 2 and 4, interior snow-free land, land ice left out | 0 | under 0.0005 |

Soil water in Noah's third layer differs by 0.012 RMS, all of it where ERA5
holds zero or slightly negative values there, under deserts: the soil water
floor above lifts them to the soil's dry limit. Land ice is left out of the
soil rows because the cold start does not take ERA5's soil there: the floor
sets it saturated, and a land-ice column's top layer starts at its skin.

### The T255 quickstart, reference physics, 24 h

`arwen_global_t255_quickstart` with the `[initial]` table above.

- **Run:** 3 min 51 s wall from start to receipt, the decode and the cold
  start included; the receipt's own `wall_seconds` is 183 s. Peak host
  memory was 4.2 GB.
- **Card:** 4.36 GiB live pool peak and 5.26 GiB held, measured at the
  allocator. The door's sizing model predicted 8.89 GiB, because it prices
  the reference suite as the native one (it says so as it starts), and it
  refuses the run unless 11.09 GiB of the card is free.
- **Receipt:** status `pass`, every gate passed. Mass drift 2.3e-7 and total
  water drift 6.7e-9 relative, against ceilings of 1e-4 and 1e-3.

Three render tapes were exported (360 x 720, hours 0, 12 and 24). The tables
compare them with ERA5 itself: area-weighted global means, extremes, and the
bias and RMS of the tape against ERA5 sampled onto the tape grid. Z500 and
precipitable water are computed from the tape's columns.

**Hour 0, against ERA5 at 2019-05-20 00Z:**

| | ERA5 mean (min, max) | tape mean (min, max) | bias | RMS |
|---|---|---|---|---|
| surface pressure, hPa | 985.78 (496.8, 1037.4) | 985.76 (517.7, 1047.7) | -0.03 | 5.4 |
| 500 hPa height, m | 5678.6 (4736, 5928) | 5677.0 (4734, 5925) | -1.6 | 2.3 |
| 2 m temperature, K | 288.28 (200.1, 313.7) | 288.23 (205.1, 309.4) | -0.05 | 1.0 |
| precipitable water, kg m-2 | 25.30 (0.11, 71.6) | 25.26 (0.08, 69.7) | -0.04 | 0.59 |

The surface pressure RMS and extremes are the model's terrain, which is the
spectral T255 truncation of ERA5's: surface pressure moves with it, over
mountains and wherever the truncated terrain dips below the real one. The
global mean, which is the mass, agrees to 0.03 hPa. The 2 m temperature on a
tape is the model's own screen diagnostic, not ERA5's 2 m field.

**Hour 24, against ERA5 at 2019-05-21 00Z:**

| | ERA5 mean | tape mean | bias | RMS |
|---|---|---|---|---|
| surface pressure, hPa | 985.76 | 985.76 | -0.01 | 5.6 |
| 500 hPa height, m | 5679.4 | 5669.1 | -10.3 | 14.3 |
| 2 m temperature, K | 288.33 | 288.93 | +0.60 | 3.9 |
| precipitable water, kg m-2 | 25.38 | 28.00 | +2.62 | 4.3 |

No tape carries a non-finite value. Over the day the 500 hPa height runs
about 10 m low and the column moistens by about 10 %; 1.25 kg m-2 of
precipitation fell on the global mean.

### The native suite, 24 h

`arwen_global_gdas_t255_native_sl_si_24h` with the same `[initial]` table
and `start_time_utc` deleted. This is the suite that reads the soil, the
snow and the sea ice the cold start sets.

- **Run:** 5 min 34 s wall from start to receipt; the receipt's own
  `wall_seconds` is 293 s. Peak host memory was 4.2 GB.
- **Card:** 8.63 GiB live pool peak and 10.79 GiB held, measured at the
  allocator, against 8.89 GiB predicted.
- **Receipt:** status `pass`, every gate passed. Mass drift 4.1e-7 and total
  water drift 4.4e-9 relative.

Hour 0 on its tape is the table above, to every digit shown.

**Hour 24, against ERA5 at 2019-05-21 00Z:**

| | ERA5 mean | tape mean | bias | RMS |
|---|---|---|---|---|
| surface pressure, hPa | 985.76 | 985.76 | -0.01 | 5.4 |
| 500 hPa height, m | 5679.4 | 5675.2 | -4.3 | 7.9 |
| 2 m temperature, K | 288.33 | 288.19 | -0.14 | 1.5 |
| precipitable water, kg m-2 | 25.38 | 25.97 | +0.59 | 2.5 |

No tape carries a non-finite value; 2.17 kg m-2 of precipitation fell on the
global mean. The tape's skin temperature against ERA5's at hour 24, by the
tape's own surface classes:

| surface | bias, K | RMS, K |
|---|---|---|
| Antarctic land ice | +1.4 | 3.6 |
| northern land ice | -0.9 | 3.2 |
| Arctic sea ice | +1.0 | 1.6 |
| Antarctic sea ice | -1.2 | 2.5 |
| snow-free land | -0.3 | 2.4 |
| snowy land that is not land ice | -2.2 | 4.1 |
| of it, land starting above 1000 kg m-2 of snow | -4.9 | 7.3 |
| open water | +0.18 | 0.38 |

Open water sits 0.18 K above ERA5's skin because it holds ERA5's SST, from
which it differs by 0.01 K. Snow on the ice sheets did not change, and the
snowy land lost 5.6 kg m-2 on the mean over the day.

### T383, native suite, 24 h

`arwen_global_gdas_t383_native_sl_si_24h` with the same `[initial]` table
and `start_time_utc` deleted, run as `gpuwm-global run CONFIG --outdir OUT
--latitude-bands 16 --host-spill on` so that it fit beside the other jobs
on the shared card. Bands and the host tier change memory, not arithmetic:
a banded, spilled run reproduces the resident run's checkpoints byte for
byte and shares its config hash. On a 16 GB card with nothing else on it
nothing has to be set.

- **Hour 0**, `tools/arwen_global_era5_check.py hour0` on the T383 state:
  temperature 0.002 to 0.09 K RMS by layer from 1 hPa to the ground,
  humidity 0.08 g/kg RMS below 700 hPa, winds 0.13 m/s, surface pressure
  0.0015 hPa bias and 0.09 hPa RMS on the model terrain (the T383 terrain is
  20 m RMS from ERA5's against 41 m at T255), Z500 -1.7 / 2.5 m,
  precipitable water -0.003 / 0.23 kg m-2; open-water skin 0.00004 / 0.010 K
  against SST, sea ice, snow and soil equal to rounding, no snow on any of
  the frozen water columns. The cold start took 3 min 26 s on the CPU and
  7.3 GB of host memory.
- **Run:** 29 min 50 s wall from start to receipt (288 steps, 16 bands, the
  host tier on, a shared RTX 4090), the receipt's own `wall_seconds`
  1,725 s; five tapes (576 x 1152 model grid, 360 x 720 tapes) exported in
  3 min 8 s.
- **Card:** 8.09 GiB live pool peak and 12.23 GiB held; 13,090 MiB on the
  card by `nvidia-smi`.
- **Receipt:** status `pass`, every gate passed. Mass drift 5.1e-7 and
  total water drift 1.0e-8 relative.
- **One in-situ wire tripped**, `spectral_top_decile_growth`, six times
  between steps 60 and 110 (hours 5 to 9), at levels 13 and 14, peaking at
  2.88 against its 2.1 threshold. The wire records and never refuses. The
  top-decile kinetic energy of level 14 grew tenfold over those hours and
  then held between 0.0087 and 0.0145 to the end of the day, without a
  trip. The two ERA5 starts at T255 on the same hour tripped nothing, and
  neither did three GDAS starts at T383 from 2026-10-15 00Z (two of 24 h,
  one of ten days); with one date of each it is not known whether ERA5 or
  the case makes the difference.

**Hour 24, against ERA5 at 2019-05-21 00Z:**

| | ERA5 mean | tape mean | bias | RMS |
|---|---|---|---|---|
| surface pressure, hPa | 985.77 | 985.77 | +0.00 | 3.3 |
| 500 hPa height, m | 5679.4 | 5675.5 | -4.0 | 7.7 |
| 2 m temperature, K | 288.33 | 288.18 | -0.15 | 1.5 |
| precipitable water, kg m-2 | 25.38 | 25.90 | +0.52 | 2.4 |

No tape carries a non-finite value; 2.28 kg m-2 of precipitation fell on
the global mean. By the tape's surface classes at hour 24 the skin runs
+0.18 K on open water, -0.12 K on snow-free land, -2.1 K on snowy land that
is not land ice and -4.6 K on the part of it that started above
1000 kg m-2, the T255 run's figures within 0.3 K.

### Five days from a winter hour, native suite, T255

`arwen_global_gdas_t255_native_sl_si_24h` from ERA5 2019-01-15 00Z, with
`start_time_utc` deleted, `duration_s = 432000.0` and `output_interval_s =
86400.0`: deep northern-hemisphere snow and the polar night, the case the
24 h May runs could not speak for. The verifying hours were fetched through
the same door (`fetch-analysis --source era5 --cycle 2019-01-16T00` and so
on) and the tapes compared with `tools/arwen_global_era5_check.py tapes`.

- **Hour 0**, `tools/arwen_global_era5_check.py hour0` on the state: the
  May figures again (temperature 0.008 to 0.18 K RMS by layer, surface
  pressure 0.003 / 0.13 hPa on the model terrain, Z500 -1.6 / 2.5 m,
  precipitable water -0.001 / 0.41 kg m-2, snow equal to rounding), and
  none of the 29,384 frozen water columns carries snow.
- **Run:** 20 min 38 s wall from start to receipt (1,440 steps), the
  receipt's own `wall_seconds` 1,212 s; six tapes exported in 56 s.
- **Card:** 8.63 GiB live pool peak and 10.98 GiB held; 11,844 MiB on the
  card by `nvidia-smi`.
- **Receipt:** status `pass`, every gate passed. Mass drift 8.2e-8 and
  total water drift 9.0e-9 relative over the five days; no in-situ wire
  tripped.
- No tape carries a non-finite value. 14.35 kg m-2 of precipitation fell on
  the global mean in the five days; the wettest column took 1,426 kg m-2.

Against ERA5 of each valid time, bias / RMS:

| lead | surface pressure, hPa | Z500, m | 2 m temperature, K | precipitable water, kg m-2 |
|---|---|---|---|---|
| 0 h | -0.03 / 5.5 | -1.6 / 2.3 | -0.11 / 1.1 | -0.03 / 0.53 |
| 24 h | -0.03 / 5.6 | -3.8 / 8.7 | -0.08 / 1.7 | +0.56 / 2.3 |
| 48 h | -0.04 / 5.7 | -2.9 / 14.9 | -0.16 / 1.9 | +0.49 / 3.3 |
| 72 h | -0.03 / 6.0 | -2.1 / 24.2 | -0.20 / 2.1 | +0.46 / 4.3 |
| 96 h | -0.04 / 6.4 | -0.8 / 35.5 | -0.19 / 2.6 | +0.36 / 5.2 |
| 120 h | -0.01 / 7.0 | -1.5 / 45.7 | -0.20 / 2.9 | +0.48 / 6.2 |

The RMS grows as a forecast's error grows; the biases do not. By the
start tape's surface classes, skin / 2 m temperature bias in K:

| lead | snow-free land | snowy land, not land ice (23,502 columns) | of it, above 1000 kg m-2 (458) | land ice north of 55N | open water |
|---|---|---|---|---|---|
| 0 h | +0.05 / -0.44 | +0.11 / -0.38 | +0.62 / -3.60 | +0.42 / +0.67 | +0.19 / +0.02 |
| 24 h | -0.62 / -0.88 | +0.32 / +0.01 | +6.28 / +1.88 | +4.41 / +2.43 | +0.17 / +0.14 |
| 48 h | -0.57 / -0.99 | +0.49 / +0.29 | +2.81 / -0.34 | +4.81 / +2.81 | +0.16 / +0.04 |
| 72 h | -0.67 / -1.15 | +0.50 / +0.34 | +4.67 / +1.07 | +4.07 / +2.56 | +0.17 / +0.01 |
| 96 h | -0.89 / -1.44 | +1.32 / +1.04 | +6.50 / +1.95 | +3.28 / +1.37 | +0.17 / +0.06 |
| 120 h | -1.01 / -1.57 | +1.51 / +1.17 | +6.65 / +2.77 | +3.19 / +1.50 | +0.18 / +0.03 |

At hour 0 the 2 m temperature is the model's screen diagnostic over the
analysed surface, which is why the deep-snow class starts 3.6 K below
ERA5's own 2 m field while its skin is within 0.6 K. Over the five days the
deep-snow land runs warm, as the land ice beside it does; from 48 h on its
skin and 2 m biases rise every day while the land ice's fall back, and
whether they level off after day five was not measured. The snowy land as
a whole gained 4.6 kg m-2 of snow on the mean. The classes are the tape's own, so a coastal column can read
sea ice and a glacier's snow at once (the sea-ice rows are left out of the
table for that reason); the model's own pack starts bare on its own grid.

These are starts on two dates: they show the route works, the fields are
sane and five days stay inside every gate, and they are not a skill score.

**Not measured:**

- a run longer than five days, or longer than 24 h at T383;
- a comparison against a GDAS start of the same valid time: the GDAS route
  reaches back to 2021-03-22 12Z only, and every start above is earlier.

## Your own CDS downloads

A runner that already downloads ERA5 from the CDS for a regional model
usually asks for relative humidity, not specific humidity, and leaves out
sea-ice cover, surface geopotential and snow density. Files like that
cannot start this model as downloaded: there would be no ice, no terrain
and no snow depth, and the humidity would have to be rebuilt from RH. The
ARCO route above carries all of them except ice thickness, which no ERA5
copy has, and it needs no key.
