# The demo: a 52 km global forecast day, from the installed package

Every picture in this folder came out of `gpuwm-global` installed as a wheel
into a clean virtual environment, run on one RTX 5070 Ti (16 GB), and drawn by
the Rust renderer `rw_wrfbatch`. Nothing here was drawn by a plotting library.

## The run

| | |
|---|---|
| Case | the public GDAS 2026-09-01 00Z analysis, one whole-globe 0.25 degree object |
| Configuration | `arwen_global_gdas_t255_native_sl_si_24h`, shipped inside the package |
| Grid | T255, a 384 x 768 Gaussian grid, 52.1 km at the equator, 40 hybrid levels to 1 hPa |
| Core | the shipped default: two-time-level semi-Lagrangian semi-implicit, 300 s, six-point quintic gather, order 16 hyperdiffusion at 720 s, off-centring 0.55 |
| Physics | the native CUDA suite: RRTMGP radiation, Morrison microphysics, Grell-Freitas convection, YSU boundary layer, Noah land, MM5 surface layer |
| Steps | 288 whole steps for 86,400 s, a checkpoint every 36 steps |
| Card | RTX 5070 Ti, 16 GB, 15.28 GiB free at the door |
| What the sizer chose | one latitude band, nothing parked on the host ("nothing parked: the run fits the card without the tier"); no memory flag was set |
| Device peak | 8.70 GiB measured at the allocator, against 8.62 GiB predicted before anything was allocated |
| Wall | 300.4 s for the whole command, analysis decode and statics load included |
| Status | `pass`, all twelve gates |
| Receipt | `config_hash 8d488805…`, `pins_hash d82dc8ae…`, `self_sha256 64b9c3b7…` |
| Measured | 2026-09-07 |

## The gates it passed

Each is a ceiling; the value the run read is beside it.

| Gate | Limit | Read |
|---|---|---|
| total water relative drift | 1e-3 | 1.85e-9 |
| mass relative drift | 1e-4 | 1.13e-6 |
| transform round-trip relative Linf | 1e-6 | 4.00e-7 |
| transform Parseval relative error | 5e-7 | 1.12e-7 |
| semi-Lagrangian Lipschitz number | 0.75 | 0.239 |
| semi-Lagrangian trajectory convergence (cells) | 1e-2 | 2.58e-4 |
| tracer mass fixer, max step relative | 0.25 | 0.119 |
| tracer mass fixer, water relative | 1e-4 | 3.73e-5 |
| physics water repair, kg/m2 a step | 1.05e-2 | 1.39e-4 |
| mass fixer, max step log offset | 2e-4 | 7.14e-6 |
| water fixer, max step relative | 1e-5 | 3.30e-7 |
| positivity fixer, max step relative | 5e-3 | 1.20e-5 |

Global mean total water over the day, read at every output step: 710.659958,
710.659943, 710.659941, 710.659931, 710.659939, 710.659949, 710.659956 and
710.659945 kg/m2. Maximum wind stayed between 91.0 and 94.6 m/s.

## The pictures

Six pictures are in this folder; all 72 are in the evidence gallery under
`2026-09-07-global-package/`, laid out
`<run stamp>/<domain>/<product>/<valid-day>/`, which is the layout the render
door writes and nothing here rearranged.

Eight products were asked for and all eight drew at every one of the nine
forecast hours: `mslp_10m_winds`, `500mb_height_winds`, `2m_temperature`,
`2m_dewpoint`, `850mb_temperature_height_winds`, `250mb_height_winds`,
`precipitable_water`, `total_qpf`. The domain label the renderer stamps is
`d01-55.589km`, which is the export grid's own spacing (361 x 720 regular
lat/lon), not the model's 52.1 km Gaussian spacing.

### pwat-f024.png, precipitable water at 24 hours

The intertropical convergence zone runs unbroken across all three ocean
basins. The monsoon is over the Bay of Bengal, the South China Sea and the
Philippine Sea at the top of the scale. Mid-latitude moisture plumes reach out
of the tropics towards both the Pacific Northwest and western Europe. The
subtropical highs, the Sahara, the Arabian interior, the Australian interior
and the whole Antarctic are dry, which is where they should be at the start of
September.

### z500-f024.png, 500 hPa geopotential height and wind at 24 hours

Late austral winter: the Southern Hemisphere jet is a single unbroken ribbon
around the globe with cores above 70 m/s, and the Northern Hemisphere jet is
weaker, split and further north, as it is at the end of northern summer.
Closed lows sit over eastern Siberia and the North Pacific; 594 dam ridges
close over both the Atlantic and the Pacific subtropics.

### mslp-f024.png, mean sea level pressure and 10 m wind at 24 hours

The Southern Ocean storm belt carries lows at 949.9 hPa south of Australia and
961.5 hPa in the South Pacific. Tropical cyclones are resolved as closed lows
with wind maxima in the west Pacific and the Indian Ocean, at the scale a
52 km grid can carry them. The subtropical highs close at 1030.6 and
1033.5 hPa.

### t2m-f024.png, 2 m temperature at 24 hours

Northern continents at the end of their summer; the Sahara, the Arabian
peninsula and the Australian interior at the top of the scale; the Antarctic
plateau at the bottom. The Andes, the Himalaya, the Ethiopian highlands and
the Greenland ice sheet are all drawn by their own orography, which is the
static surface fields the `statics` door built doing their work.

### t850-f024.png, 850 hPa temperature, height and wind at 24 hours

The lower-troposphere temperature field, where the mid-latitude fronts the
mean sea level pressure chart implies are visible as thermal gradients rather
than as pressure contours.

### mslp-f000.png, mean sea level pressure at the analysis

**This one is here because it is different, not because it is pretty.** The
analysis is the model's spectral fit to a 0.25 degree GRIB field on its first
step, and the small-scale ringing around steep orography (the Himalaya, the
Andes, the Antarctic coast) is what a truncated spherical-harmonic
representation of a sharp field looks like. Compare it with `mslp-f024.png`
from the same run: the hyperdiffusion the core carries, order 16 at a 720 s
e-folding time, has removed it by the end of the first day, and the fields the
model produces after spin-up are the smooth ones. What effective resolution
this model actually has, as opposed to what its grid spacing says, is the
subject of `docs/arwen-global-effective-resolution.md`.

## The second demo: an analysis cycle, and the package's own picture door

The forecast pictures above were exported by this package and drawn through
the engine's render command. `analysis-t2m-f003.png` is the first picture
`gpuwm-global render` has ever drawn: the door passed one keyword the engine's
renderer does not declare, so every draw died at the call after exporting its
tapes. It draws now, and eight pictures came out of that one command with no
failures and nothing skipped.

### The run

| | |
|---|---|
| Case | the GDAS 2026-09-01 00Z analysis on disk, cycled forward against the public surface stream (`--stream iem-asos`) |
| Command | `gpuwm-global da fresh`, three hourly windows to 03Z, from the installed wheel |
| Filter | the shipped default: the deterministic successive correction, one member |
| Card | RTX 5070 Ti, 16 GB, shared with another job |
| Wall | 596.5 s for the whole command; 76.1 s a cycle on average, 89.6 s at worst, 0.025 of the hour each cycle covers |
| Status | `pass`, three of three cycles engineering-complete |
| Third analysis | 20,713 rows assimilated, 2,302 withheld for the gate of record |
| Fits at that analysis | station pressure 403.2 Pa o-minus-b against 113.2 o-minus-a on 4,519 rows; temperature 1.946 against 1.816 K on 4,717; wind 2.157 against 2.047 and 2.156 against 2.085 m/s |
| Dewpoint | 3.383 against 3.411, worse after the analysis than before it, printed as a reading: `--moisture-update` is off by default, so dewpoint rows are scored and not analysed |
| Receipt | `self_sha256 e1a61fbb…`, `pins_hash d82dc8ae…`, the same arithmetic identity the forecast demo's receipt carries |
| Measured | 2026-09-07 |

### analysis-t2m-f003.png, 2 m temperature at the third analysis

The state the cycle handed back, three hours after the analysis it started
from. The picture reads as the field should for 03Z: the Sahara, Arabia and
the Australian interior at the top of the scale, the Antarctic plateau at the
bottom, the terminator's diurnal signature across Asia, and the Andes, the
Himalaya, the Greenland ice sheet and the Ethiopian highlands drawn by their
own orography.

### The reproducibility reading nobody asked for

The mean sea level pressure picture this run drew at its step 0 is
**byte-identical** to `mslp-f000.png` above, `md5 b772960e…`. The two came
from different runs on different days through different code: one exported by
the forecast demo and drawn by the engine's render command, the other exported
by `gpuwm-global export` from the assimilation door's initial state and drawn
by `gpuwm-global render`. Same input state, same bytes out.

All eight analysis pictures, the DA receipt and the third cycle's full report
are in the evidence gallery under
`2026-09-07-global-package/analysis-quickstart/`.

## How to reproduce it

```bash
gpuwm fetch --source gdas --cycle 2026-09-01T00 --hours 0 \
  --mode full-file --out cases/baseline-2026090100
gpuwm-global statics arwen_global_gdas_t255_native_sl_si_24h --geog-root ~/WPS_GEOG
gpuwm-global run arwen_global_gdas_t255_native_sl_si_24h --outdir out/t255-day
gpuwm-global render out/t255-day --outdir out/pictures \
  --start-date 2026-09-01_00:00:00 \
  --products mslp_10m_winds,500mb_height_winds,2m_temperature,2m_dewpoint,850mb_temperature_height_winds,250mb_height_winds,precipitable_water,total_qpf
```

The run that made these pictures used the same configuration, taken from the
installed package's own configs directory, and its checkpoints were exported
and drawn in two steps rather than one. Its receipt is quoted above; the
export receipt carries the same `config_hash` and `pins_hash`, which is how a
tape is tied to the run that made it.

The analysis cycle above:

```bash
gpuwm-global da fresh arwen_global_gdas_t255_native_sl_si_24h \
  --outdir out/fresh --stream iem-asos \
  --analysis-grib cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000 \
  --analysis-cycle 2026-09-01T00:00:00Z --until-utc 2026-09-01T03:00:00Z \
  --forecast-hours 6
gpuwm-global export out/fresh/fresh-config.toml \
  out/fresh/arwen_global_analysis_step00000036.npz \
  --outdir out/tapes --nlat 361 --nlon 720 --start-date 2026-09-01_00:00:00
gpuwm-global render out/tapes/wrfout_d01_2026-09-01_03_00_00 \
  --outdir out/pictures --start-date 2026-09-01_00:00:00 \
  --products mslp_10m_winds,2m_temperature,precipitable_water,500mb_height_winds
```

Without `--analysis-grib` and `--analysis-cycle` the door fetches the newest
published GDAS cycle instead and cycles to the newest observation hour, which
is what the README's quickstart does.
