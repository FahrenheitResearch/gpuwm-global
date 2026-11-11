# Arwen Global as the parent of a regional run

`gpuwm-global export-mapped-parent` writes a global forecast in the form the
regional engine reads as initial and lateral boundary data. The feed is one
way: the regional run takes its first state and its boundary values from the
global forecast and returns nothing to it.

The regional engine (`gpuwm`) already builds a run from any model it can read
through `gpuwm prep --source mapped`: one NetCDF file per valid time on a
regular lat-lon grid, plus a mapping document (which variable is which field,
in which units) and a composition document (where terrain and soil come from).
This command writes all of that from checkpoints. Nothing changes on the
regional side, and the regional engine does the horizontal and vertical
interpolation, the adjustment to its own terrain, and the boundary file.

It is a different door from the two beside it. `export` writes render tapes
for pictures (staggered winds, no coordinate variables). `export-parent` and
the `make-regional-target`, `translate-regional-frame`, `make-parent-series`
chain write hash-bound arrays for a regional state held in Python; no shipped
regional command reads them.

## Run it

Export one frame per checkpoint, in time order, over a window that covers the
regional domain with a margin of a few degrees:

```
gpuwm-global export-mapped-parent run/arwen-global-run-config.toml run/arwen_global_step00000288.npz run/arwen_global_step00000324.npz run/arwen_global_step00000360.npz --outdir parent --nlat 576 --nlon 1152 --bbox 20 56 -130 -60
```

Give it the config copy the run left beside its checkpoints
(`arwen-global-run-config.toml`), as for `export`. A native-physics run is
exported on the device it ran on.

What the export itself holds you to:

- `--nlat` and `--nlon` name the regular grid the window is cut from. Give the
  run's own Gaussian grid (576 and 1152 at T383). The default, 360 by 720, is
  coarser than a T383 run.
- The window runs west to east inside -180..180. One that crosses the date
  line is refused. Leaving `--bbox` out writes the whole grid of cell
  centres, which has no point at either pole; that has been exercised only
  at test size, never through a regional preparation, and its memory at T383
  has not been measured.
- The whole series is checked from the checkpoint headers before the first
  frame is written: time order, frames that already exist (`--overwrite`
  replaces them), and parent frames in the folder that are not part of this
  export, which are refused with or without `--overwrite`. One folder holds
  one export.
- Peak memory on the card was 10.1 GiB for nine frames of the window below
  and 10.9 GiB for seventeen. Allow for the longer series.

The output folder then holds:

| file | what it is |
|---|---|
| `arwen_global_parent_YYYY-MM-DD_HH_MM_SS.nc` | one frame per checkpoint |
| `arwen-global-parent.mapping.json` | the mapping, with this model's level count and constants |
| `arwen-global-parent.composition.json` | terrain from the frames themselves, soil as Noah's four layers |
| `arwen-global-parent.provenance.json` | what the parent is and how each field was made |
| `arwen-global-mapped-parent-receipt.json` | every frame's SHA-256, valid time and checkpoint, and the next command |

The receipt's `regional_prep_command` is the regional preparation command for
these frames, complete except for the four arguments that belong to the
regional run: `--experiment-config`, `--wps-namelist`, `--geog-root` and
`--output-root`. Every frame appears twice in it, once as `--input` and once
as the terrain supplement. Its paths are absolute, so it runs from the
regional run's folder whatever `--outdir` was typed. Then `gpuwm sim` runs
the prepared folder.

Rules the regional preparation enforces on the series:

- the first frame's valid time is the regional start time;
- the frames are uniformly spaced, and the spacing is the boundary interval
  (the mapping accepts whole multiples, so every second frame also works);
- the last frame is at or after the end of the regional run;
- the window contains the whole regional domain.

A single frame is a valid initial state and the receipt says it is not a
boundary series.

## The regional side

Three commands of the regional engine follow the export. They are its own,
and `gpuwm domain --help`, `gpuwm prep --help` and `gpuwm sim --help` are the
reference for them.

```
gpuwm domain --source mapped --cadence 3 --point 41.0,-79.0 --root-dx 3 --point-extent-km 1350 --hours 24 --cycle 2026-11-12T00 --card "5070 Ti" --name fed --out fed.toml
```

`--cycle` is the first frame's valid time and `--cadence` the frame spacing in
hours. The command writes `fed.toml` and `fed.namelist.wps` and prints the
lat-lon window the domain needs; the `--bbox` of the export must contain it.
Edit the TOML now if at all, because the preparation binds its hash: the
emitted file asks for an hourly restart checkpoint, and `restart_interval_s =
0.0` turns that off.

Two defaults of the regional engine reach outside the frames. Add both lines
to the `[shared]` table in the same edit if the global forecast is to be the
only source of the run's state:

```
lake_state_source = "forcing"
sf_ocean_physics = 0
```

- Over a domain with large lakes the engine asks a lake surface analysis
  (NOAA GLSEA) for water temperature and ice when it starts, and keeps the
  parent's values only when the request fails. `lake_state_source =
  "forcing"` makes no request. Without it the log line `lake
  initialization:` says which happened, `forcing_fallback` meaning the
  parent's values were kept.
- A run of 48 h or longer switches on the engine's ocean mixed-layer model
  unless the TOML says otherwise, and starts it from a monthly climatology of
  mixed-layer depth (NOAA WOA18). The sea surface then no longer stays at the
  parent's value, and the forecast differs from the fixed-sea run from the
  first hour. `sf_ocean_physics = 0` keeps the parent's sea surface
  temperature, as a shorter run does by default. The log line `OCEAN
  INITIALIZATION FALLBACK` marks a run that took the climatology.

The preparation command is the receipt's `regional_prep_command` with the
regional run's own arguments appended:

```
--experiment-config fed.toml --wps-namelist fed.namelist.wps --geog-root WPS_GEOG --preprocess-backend cpu --output-root prepared
```

It needs no device. It writes the initial state, one boundary record per
interval and its own receipt, `prepared/proof.json`, which lists the forcing
times it read, the boundary times it wrote and the SHA-256 of the mapping. The
static fields of the regional grid (terrain, land use, soil and vegetation
categories, albedo, and the deep soil temperature under the lowest soil
layer) come from the geography tree named by `--geog-root`, never from the
frames. The deep soil temperature is the one of these the global model also
carries and the frames do not: in the measured run below the geography
tree's value, a reanalysis climatology, was 2.4 K warmer on average than the
global model's own, and the preparation nudged the lowest soil layer toward
it by 0.16 K RMS. The engine's `soil_texture_downscale = false` under
`[ingest]` turns that adjustment off; it was not tried here.

```
gpuwm sim prepared --experiment-config fed.toml --wps-namelist fed.namelist.wps --outdir run
```

This is the only step of the three that needs the card. Its `report.json`
repeats the mapping and manifest hashes it ran from under `input`, so a
finished run can be traced to the frames that fed it.

## The file contract

Regular lat-lon cell centres, latitude south to north, longitude west to east
in -180..180. Model levels bottom to top. `Time` is seconds since 1970-01-01
00:00:00. Every variable is float32 and carries a `units` attribute; the
regional decoder compares unit strings exactly.

| variable | dimensions | units | content |
|---|---|---|---|
| `T` | level | K | potential temperature minus 300 |
| `PB` | level | Pa | full pressure at layer midpoints |
| `PHB` | interface | m2 s-2 | hydrostatic geopotential at layer interfaces |
| `QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP` | level | kg kg-1 | mixing ratios per kg of dry air |
| `U_EARTH`, `V_EARTH` | level | m s-1 | earth-relative wind at mass points |
| `W` | interface | m s-1 | the air's vertical velocity through the model's coordinate surfaces; see below |
| `PSFC` | surface | Pa | surface pressure |
| `HGT` | surface | m | the model's own terrain, identical in every frame |
| `TSK` | surface | K | skin temperature; over water, the sea surface temperature |
| `T2`, `Q2`, `U10`, `V10` | surface | K, kg kg-1, m s-1 | screen-level fields |
| `LANDMASK`, `SEAICE` | surface | 1 | land mask (fraction at least 0.5) and ice fraction |
| `SNOW`, `SNOWH` | surface | kg m-2, m | snow water equivalent and depth |
| `TSLB`, `SMOIS` | soil | K, m3 m-3 | four Noah layers, 0-10-40-100-200 cm |

The mapping declares this model's constants so the regional side reads the
columns as they were written: height is `PHB` over 9.80616, and temperature
comes back from potential temperature with kappa 287/1004 and a reference
pressure of 1000 hPa.

The winds are sampled from the model's own wind synthesis at mass points. A
render tape's staggered `U` and `V`, averaged back, differ from them by up to
about 4 m/s at fronts.

`W` is the mass flux across each hybrid interface divided by density and
gravity, positive up. It is zero at the model top and at the surface by
construction and leaves out the motion of the coordinate surfaces
themselves, so over sloping ground it is not the velocity relative to the
ground: uniform flow up a slope has `W` equal to zero at every interface.
The regional run starts its own vertical velocity from it: in the measured
run below the two fields correlate at 0.97 to 0.99 at the initial time, and
the regional value at the ground starts at zero. A render tape's `W` is
this array plus the wind's climb along the coordinate surfaces
(`docs/ARWEN_GLOBAL.md`); the frames carry it without the climb, as the
measured run below was fed.

## What is not carried

- Number concentrations. The regional microphysics starts them from the mass
  fields.
- Land use, soil categories, vegetation and terrain for the regional grid. The
  regional run takes them from its own geography data; the global model's
  35 km versions are not used. `HGT` in the frames is the parent's own terrain
  and is read only to place the parent's columns in height.
- Deep soil temperature. The regional run takes its own from its geography
  data (see 'The regional side').
- Accumulated precipitation and the surface energy fields. They are on the
  render tapes.
- Anything back from the regional run. There is no two-way coupling.

The frames are a bilinear sample of the Gaussian grid, so skin, soil and
snow values blend land and sea along coasts; the regional preparation sorts
them by its own land mask and takes the nearest usable value where the two
masks disagree.

The first frame of a run started from an analysis carries screen-level fields
diagnosed at export, because the model has not yet stepped its surface layer;
each frame's `ARWEN_SURFACE_DIAGNOSTICS` attribute says which it carries.
Starting the regional run from a later forecast hour avoids it.

## A measured run

One run, so the figures have conditions: a T383 forecast with 40 levels and
the native physics suite, started from an analysis valid 2026-11-11 00 UTC,
feeding a 3 km regional run that starts at forecast hour 24 and runs 48 h.
Both ran on one RTX 5070 Ti 16 GB. The regional TOML carried the two lines
of 'The regional side', so the frames and the geography tree were its only
inputs.

| step | what | measured |
|---|---|---|
| export | 17 frames, forecast hours 24 to 72 every 3 h, window 20 to 56 N and 130 to 60 W of the 576 x 1152 grid (115 x 225 points, 52 MB a frame) | 219 s; 10.9 GiB peak on the card, 9.1 GiB host memory |
| preparation | Lambert, 450 x 360 points at 3 km, 49 levels, top at 50 hPa, 17 forcing times | 59 s on the CPU, six workers; 10.6 GiB host memory |
| forecast | 48 h, 7215 adaptive steps, hourly history, Morrison, YSU, Noah, RTE+RRTMGP, no cumulus scheme | 1620 s; 8.8 GiB peak on the card |

Whether the boundary is fed was measured with
`tools/arwen_global_regional_parent_check.py`, which interpolates a parent
frame to the regional mass points on its own (bilinear, then linear in ln p)
and reports the RMS difference by distance from the lateral edge: the
specified row, the four relaxation rows, the next fifteen, and the interior
at least sixty points in. Three numbers per cell: temperature in K, eastward
wind in m/s, surface pressure in hPa.

| regional time | against the parent frame of | specified row | relaxation rows | interior |
|---|---|---|---|---|
| hour 0 | hour 0 | 0.17, 0.28, 0.03 | 0.17, 0.28, 0.03 | 0.17, 0.30, 0.03 |
| hour 12 | the same time | 0.15, 0.28, 0.02 | 0.28, 0.77, 0.23 | 0.63, 1.69, 0.36 |
| hour 24 | the same time | 0.15, 0.27, 0.03 | 0.30, 0.97, 0.34 | 0.63, 1.64, 0.35 |
| hour 48 | the same time | 0.17, 0.28, 0.03 | 0.28, 0.76, 0.20 | 0.55, 1.50, 0.24 |
| hour 12 | hour 0 | 3.57, 9.23, 4.94 | 3.59, 9.12, 4.87 | 3.75, 7.18, 6.48 |
| hour 24 | hour 0 | 5.00, 12.23, 8.45 | 5.02, 12.14, 8.31 | 6.37, 10.41, 8.60 |
| hour 48 | hour 0 | 7.33, 16.93, 8.07 | 7.32, 16.92, 8.08 | 8.43, 15.36, 5.90 |

The first row is the floor of the comparison: at the initial time every zone
differs from the parent by the same small amount, as two different
interpolations of one frame would. The specified row
stays on that floor against the parent frame of the same valid time at all
sixteen updates, while the interior moves away from the parent and the same
row ends 7 K and 17 m/s from the frame the run started on. One hour after an
update the specified row is on the floor against the two bracketing frames
interpolated linearly in time (0.17 K, 0.29 m/s) and off it against either
frame alone (0.45 K and 0.90 K), so the boundary moves between updates as
well.

The control was the first 24 h again, prepared from nine copies of the first
frame, so its boundary never moves. The two runs are identical at the
initial time (every difference zero). After 24 h they differ by 4.9 K and
9.5 m/s RMS over the domain. Over the first hours the difference is largest
where air enters: at hour 3 the 20-point strip along the west side, where
the 850 to 500 hPa wind blows into the domain at 9 m/s, differs by 1.6 K and
4.4 m/s, the strip along the east side, where it leaves at 13 m/s, by 0.4 K
and 1.5 m/s, and the interior by 0.3 K and 0.9 m/s.

The regional run stays a continuation of its parent. At least twenty points
from the edge its 500 hPa height correlates with the parent frame of the
same time at 0.994 or better through 48 h (RMS 7 to 9 m), and its lowest
sea-level pressure follows the parent's: 999.1 against 998.9 hPa at hour 12,
992.9 against 993.4 at hour 24, 993.8 against 993.9 at hour 48, each within
140 km of the parent's position.

### What that run did not get right

**Cloud and precipitation at the boundary.** Temperature, wind, vapour,
column mass and geopotential reach the boundary row as the parent has them.
The five condensate species are written into the regional engine's boundary
tables too, but the engine does not put those tables on the row as they are.
For each boundary interval it serves the smaller of the two bracketing
frames at each point, taken again as the minimum over 25 km of edge either
way, and holds it for the interval. The rule exists so that a convective
cell caught in one frame does not pour into the domain from a fixed point,
and it has no switch. With frames 3 h apart little of a moving cloud field
is in both frames at one place: over this run the boundary row held 36 % of
the parent's condensate (34 % of its cloud water, 24 % of its rain, 49 % of
its ice, 41 % of its snow). Inflow air arrives short of cloud and the
regional model re-forms it within about six points. Frames an hour apart
would let more through; the run that made these was checkpointed every 3 h.

**The outflow side.** The step from 35 km to 3 km is taken in one go, and
where the regional run's own convection reached the outflow edge it stood
in the boundary zone. Precipitation averaged along the east side over the
first 24 h, in five-point bins from the edge inward, in mm:

| points from the edge | 0-4 | 5-9 | 10-14 | 15-19 | 20-24 | 25-29 | 30-34 | 35-39 | 40-44 | 45-49 | 50-54 | 55-59 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| the parent | 4.1 | 3.4 | 2.8 | 2.5 | 2.4 | 2.7 | 3.0 | 3.2 | 3.3 | 3.4 | 4.1 | 5.1 |
| regional, five-point boundary zone | 2.6 | 8.3 | 8.9 | 8.3 | 5.9 | 4.9 | 4.6 | 2.3 | 1.2 | 1.3 | 1.8 | 3.3 |
| regional, ten-point boundary zone | 0.8 | 7.5 | 9.2 | 6.8 | 4.8 | 3.0 | 1.0 | 0.9 | 1.2 | 1.9 | 3.0 | 5.1 |

About three times the parent's rain in points 5 to 19, and less than half
of it from 40 to 50 points in. Vertical velocity reached 13 m/s three and four
points in from that edge, against 2.4 m/s at most on the specified row. A
boundary zone twice as wide (`spec_bdy_width = 10`, `relax_zone = 9`,
`spec_exp = 0.33`) left the excess where it was and moved the dry band
closer to the edge, so it is not a remedy; the cyclone was the same in
both and the interior differed by 0.2 K RMS. Precipitation within about 50
points of the outflow edge of such a run is not a forecast. At least 50
points from every edge the 24 h total averaged 7.6 mm against the parent's
7.0 mm, with a pattern correlation of 0.79.

In the second day a band of showers also stood along the inflow (west)
edge, over Lake Michigan: the 48 h total in points 5 to 14 from that edge
averaged 14 and 16 mm against the parent's 7 and 8 mm.

## Measure a run of your own

```
python tools/arwen_global_regional_parent_check.py track run/wrfout/wrfout_d01_2026-11-12_12_00_00 parent/arwen_global_parent_2026-11-12_12_00_00.nc
python tools/arwen_global_regional_parent_check.py diff run/wrfout/wrfout_d01_2026-11-12_12_00_00 control/wrfout/wrfout_d01_2026-11-12_12_00_00
```

`track` prints bias and RMS of temperature, both wind components, vapour and
surface pressure for each zone; `diff` prints the RMS difference of two runs
at one valid time by zone and by side, with the mean wind into the domain on
each side. Both need numpy and netCDF4 and no device. The tool's own test
runs it on a regional state built from a parent frame, where the answer is
zero, and on one with a departure planted in a known zone.
