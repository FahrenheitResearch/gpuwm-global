# ArWen Global

A hydrostatic global spectral weather model that runs a whole forecast day on
one consumer graphics card, with its own ensemble data assimilation.

ArWen Global integrates the moist primitive equations on a Gaussian grid in
spherical-harmonic space, on a two-time-level semi-Lagrangian semi-implicit
core at a 300 s step, with a full GPU physics suite (RRTMGP radiation, Morrison
microphysics, Grell-Freitas or New Tiedtke convection, YSU boundary layer, Noah
land, MM5 surface layer) on a 52 to 25 km grid at 40 hybrid levels. It
cold-starts from a single public GDAS analysis, writes render-ready `wrfout`
tapes that the same Rust renderer draws as every other ArWen product, and can
run its own analysis cycle: a 32-member LETKF under the forecast control, fed
by surface observations, radiosondes, buoys, satellite motion vectors and radio
occultations through Rust decoders, with infrared and microwave radiance
operators beside them. At 52 km a 24 hour global forecast takes about four
minutes on a 16 GB card, and at 25 km about sixteen minutes on a 32 GB one.
Every number on this page was measured; each one carries the card and the date.

**This is an experimental research model.** Its configuration surface can move
between releases and its output is not a supported product. The claims it does
not make are listed in [docs/ARWEN_GLOBAL_FULL.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_FULL.md).

---

## Install

```bash
pip install "gpuwm-global[gpu-cu13]"     # a CUDA 13 runtime
pip install "gpuwm-global[gpu-cu12]"     # a CUDA 12 runtime
```

One extra per CUDA major, because no pip environment marker can tell them
apart. `gpu-cu13` pulls `cupy-cuda13x[ctk]`, `gpu-cu12` pulls
`cupy-cuda12x[ctk]`, and the bare `gpu` alias is `gpu-cu13`, the stack the
card figures on this page came off. Either line also resolves the engine
`gpuwm>=3.0.5,<3.1`, its data companion `gpuwm-data` beside it,
numpy, scipy and netCDF4. Against the published `gpuwm 3.0.5`,
`tools/measure_boundary.py` (in the source tree: the repository and the
sdist carry `tools/` and `tests/`, the wheel does not) reads 244 symbols
across 76 modules with no gap (Linux, with CuPy, 2026-11-23). Without either extra the package installs and
every CPU door runs; the forecast needs the card. It installs in one
environment beside `gpuwm-hex` 0.6, which needs the same engine line.

The rows dated 2026-11 up to 2026-11-15 on this page were taken on the
published engines `gpuwm 2.9.12` and `2.9.13`. The older forecast and assimilation figures
were taken on 2026-09-06 and 2026-09-07 in the engine checkout this model was
developed and graded in, whose own version string reads 2.6.4. That physics is what `arwen_global.core` carries, which is why
an install on the 2.8 engine integrates the same bytes those runs did, apart
from the engine fixes this package has taken since (listed under PULL in
[docs/CARRIED-PHYSICS-DIVERGENCE.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/CARRIED-PHYSICS-DIVERGENCE.md)).

The Rust binaries the model calls at run time arrive as release assets,
verified against the size and SHA-256 pins inside the wheel before they are
used, never from a local `cargo` build:

```bash
gpuwm fetch-bridges        # the engine's own bundle
gpuwm-global fetch-doors   # the doors the engine's bundle does not carry: observation binaries and this package's own libraries
gpuwm-global doctor        # what is installed, staged and missing
```

Of the eight observation doors, `rw_asos`, `rw_igra2`, `rw_ndbc`,
`rw_amv`, `rw_gnssro`, `rw_wis2` and `rw_goes` arrive in the engine's own
bundle on 3.0.x, so `fetch-doors` stages the other one.  It also stages the
two libraries this package builds from its own `rust/` workspace: `rw_global_coldstart` (the cold start's analysis regrid
and ln p remap) and `global_render_kernels` (the render tapes' regrid and
column and the regional translation).  Neither has a NumPy fallback, so an
install whose bundle predates them builds them from the source
(`cargo build --release` in `rust/`) and names each library in its
variable, as `doctor` says.

`doctor` exits 1 when a command this page or
[docs/CLI-REFERENCE.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/CLI-REFERENCE.md) documents cannot run, and every
line that says no names the command it stops and the remedy. Two engine
symbols are absent from every published engine through 3.0.5:
`preflight.measured_free_vram_bytes` and `surface_bias.interpolate_to_tape`.
Neither stops a documented command (the first stops the standalone
card-pricing check that no subcommand is wired to, the second the
surface-energy scorecard's regrid onto the tape), so `doctor` prints both as
optional notes and a correct install exits 0: the three lines above, then
`doctor`, exited 0 against `gpuwm 3.0.5` on Linux and on Windows (MEASURED 2026-11-22). CI runs the same three lines on every push and every
publish and fails on any gap. Calling either symbol still refuses by name.
The published 0.1.2 graded both as gaps and exited 1 on every correct
install; that is fixed from 0.1.3. Its "this account" section prints the
folders this account stages into and locks in, and the variable that moves
each one; on a machine shared by several accounts, read
[Several accounts on one machine](#several-accounts-on-one-machine) first.
`gpuwm-global --version` prints the release, the engine it runs on and the
Python.

`gpuwm-global` needs `gpuwm>=3.0.5,<3.1`. The floor is the engine this
release's test suite and forecast proofs ran on. The ceiling is not
decoration: the engine's own trees are not supersets of each other across
minors, and this package imports 244 symbols across that boundary.
`python tools/measure_boundary.py`, run from the source tree, regenerates the measurement rather than
trusting a table, and names the symbols it could not check on a host without
a CUDA runtime instead of counting them clean. The physics used in the
observation-scored runs is INSIDE this package (`arwen_global.core`: the
radiation, cumulus, surface-layer, boundary-layer, land-surface and
microphysics schemes, their kernels, the CUDA loader, the float64 mirror used
by kernel scorecards to compare GPU arithmetic (code verification, not an
observation score), and the Noah tables). The engine files the carried code
still reaches are pinned by path, size and SHA-256 in
`arwen_global/data/engine-seam.json`, and `gpuwm-global doctor` re-hashes them
and prints an `engine seam` section. The pins are taken at `gpuwm 3.0.5`,
where all 49 files are proven: 7 of them moved against 3.0.3, none in a way
that moves a number on the run path, and each was re-pinned at 3.0.5;
`gpuwm/core/constants.py`, whose CUDA defines open every carried kernel, did
not move. On any other engine, a file whose bytes moved is a `doctor` gap
that names it and the engine to install. gpuwm 3.0.3 removed
`gpuwm/da/letkf.py`, and the eight names the ensemble filter took from it are
carried in `arwen_global/da/letkf_core.py`, verbatim from 2.9.13; the one
engine file they still reach, the batched eigensolver
(`gpuwm/core/jacobi_eigh.py`), is pinned in its place.

In the source tree, the two nodes of `tests/test_arwen_global_engine_seam.py` that compare
hashes run against the pinned version and skip, naming both versions, on any
other; `doctor` reports a file whose bytes moved as a gap, by name, and the
version ceiling is the refusal. Where the carried physics differs from the
engine's own copy of the same file, and whether a future engine change to
that code should be taken, refused or offered back, is one row per
difference in
[docs/CARRIED-PHYSICS-DIVERGENCE.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/CARRIED-PHYSICS-DIVERGENCE.md), held
to the installed engine by `tests/test_engine_divergence.py`, which is
baselined on 2.8.5 and skips its two comparisons, naming both versions, on
any other.

---

## Quick start: a forecast day

One route, from an empty machine to drawn maps of a 24 hour T255 forecast
from the public GDAS 2026-09-01 00Z analysis. The same route, with the
reason for each step, is
[docs/ARWEN_GLOBAL_QUICKSTART.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_QUICKSTART.md).

```bash
gpuwm-global fetch-analysis --cycle 2026-09-01T00 --out cases/baseline-2026090100
gpuwm fetch-geog
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

`go` builds the static surface fields, integrates the forecast, exports
render-ready tapes and draws the product pictures into `out/day/pictures`,
naming the stage it is in while the detail goes to the run's log.

**What the first day needs** (measured 2026-11-23 from a clean install with an empty home, on an RTX 5070 Ti, 16 GB, taken through the machine's card queue while the host ran other work):

| | |
|---|---|
| Card | the door prices the day at 8.96 GiB of card and refuses before it allocates anything if the card has less free; the run held 5.63 GiB live, its pool 6.80 GiB |
| Host memory | about 18 GB while the statics are built (17.2 GiB peak resident), about 4.5 GB during the forecast |
| Disk | about 47 GB in the account's home: the geography archive 33.6 GB, the day's nine checkpoints 12.3 GB, the engine's bridges 0.47 GB, the analysis 0.47 GB, the pictures 0.06 GB; the install itself 3.3 GB, and pip keeps a 2.0 GB cache in the home of the account that installs; `score` adds 2.9 GB |
| Time | the install 70 s, `fetch-bridges` 12 s, `fetch-analysis` 48 s, `fetch-geog` 45 s, `go` 8.1 min the first time (the statics 2.8 min, the forecast 4.1 min by its receipt, then the tapes and pictures) |
| Network | about 5.5 GB: the install 2.04 GB (the package, the engine, CuPy and its CUDA libraries), the geography 2.75 GB, the analysis 0.48 GB, the bridges 0.19 GB and the doors 13 MB; `score` fetches 2.9 GB more |

Sixty-three experiments ship inside the package and resolve by bare name from
any directory (`gpuwm-global configs` lists them); a path that exists always
wins over a shipped name, so an edited copy runs.

### The same day, one stage at a time

**1. Fetch the analysis.** One whole-globe GDAS object, about 470 MB, no
credentials.

```bash
gpuwm-global fetch-analysis --cycle 2026-09-01T00 --out cases/baseline-2026090100
```

It lands `cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000`, the file
every shipped `..._native_sl_si_24h` day from T63 to T1023 reads, and prints
the commands that follow. It is the engine's whole-globe f000 fetch with its two non-default
flags bound (`gpuwm fetch --source gdas --hours 0 --mode full-file`): the
f000 alone is the assimilation cycle's own estimate of the atmosphere, and
the default GDAS transport is an area crop, which the initializer refuses
because its longitude ring does not close.

**2. Stage the geography and build the surface statics.** The archive once
per machine, the statics once per truncation.

```bash
gpuwm fetch-geog
gpuwm-global statics arwen_global_gdas_t255_native_sl_si_24h
```

`gpuwm fetch-geog` stages the WPS_GEOG archive (sixteen datasets,
a 2.75 GB download, 33.6 GB on disk) into `~/.local/share/gpuwm/WPS_GEOG`, which is where `statics`
looks. `GPUWM_WPS_GEOG` points `statics` and `go` at an archive already on
the machine; `gpuwm fetch-geog` does not read it and stages a copy of its
own, so skip that line when an archive is already there. It takes no
global config: run it bare (`--config` with a global day exits 2).
`statics` reads the archive and writes its cache under the account's own
home (`~/.local/share/gpuwm/arwen-global-statics`), so on a shared machine
each account builds its own, and it needs about 18 GB of host memory at
T255. The T255 and T383 forecast days also run the kim-gwdo
orographic drag (from 0.2.3), whose smaller cache, built from the archive's
`orogwd_10m`, this command builds too
([docs/ARWEN_GLOBAL.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL.md),
"Orographic gravity-wave drag").

**3. Run the day.** Nothing about memory is set: the door prices the card,
chooses the latitude band count and decides which state slices to park on the
host, and refuses before it allocates anything if the card cannot hold the run.

```bash
gpuwm-global run arwen_global_gdas_t255_native_sl_si_24h --outdir out/t255-day
```

288 whole steps at 300 s, a checkpoint every three hours (1.46 GB each after the first, 12.3 GB for the day),
`diagnostics.jsonl` and a self-hashed receipt beside them. The console line
every 36 steps prints the largest wind and `total_water`, the global mean of
every water store the run books (the atmosphere, about 27 kg/m², plus the
soil and the 500 kg/m² surface reservoir), which should hold flat.
`gpuwm-global check-receipt out/t255-day/arwen-global-receipt.json`
validates the receipt; `--restart` continues from a checkpoint.

**4. Draw it.**

```bash
gpuwm-global render out/t255-day \
  --outdir out/pictures \
  --products 2m_temperature,mslp_10m_winds,500mb_height_winds
```

`render` takes checkpoints, a whole run directory or tapes that are already
exported; it writes the intermediate tapes itself and removes them unless
`--keep-tapes` is given. No experiment is named on that line because `run` and
`go` leave a byte copy of the config beside their checkpoints, so a run
directory says what made it; `--config` states it for inputs that are not in
one. The valid times come from the analysis, so no date is typed either.
The tapes are a regular 0.5° grid; each map's title gives that spacing
("Δx 55.589 km") and its source line names the model grid it was drawn from
("T255 52.1 km grid"). `gpuwm render TAPE --list-products` prints the
catalogue and says which entries a given tape can actually fill. The tapes
carry `MAP_PROJ = 6`, so global frames go out through the ordinary layout,
`<outdir>/<domain>/<product>/<valid-day>/`. Weather fields are drawn by the
Rust renderer `rw_wrfbatch` and by nothing else. `gpuwm-global export`
writes the tapes without drawing them.

**5. Score it against observations** (optional).

```bash
gpuwm-global score out/t255-day --out out/score
```

ASOS stations through `rw_asos` and radiosondes through `rw_igra2` are the
rows of record; the GFS and IFS analyses are secondary references. On a first
run it takes about 25 to 27 minutes (1,613 s measured 2026-11-22, 1,490 s on 2026-11-23) and fetches about 2.9 GB of station and sounding records and GFS and IFS analyses (2.89 GB measured 2026-11-23). Most of the time is two steps, the station table (9 minutes) and the year's radiosonde archive (12 minutes); it prints
each door call as it goes. `--asos-record`, `--igra2-table` and `--reference`
reuse what an earlier run fetched.

**No card?** `arwen_global_moist_smoke` is a T3 four-step numpy run that goes
through the identical run, export and render sequence in seconds and produces a
real global frame. It rehearses the plumbing, not the forecast.

### Several accounts on one machine

Each account keeps its own staged files under its own home. On a Linux host
shared by several accounts and one install:

- **Fetch locks.** The engine's downloads (`gpuwm fetch-geog`, `gpuwm fetch`
  and the fetches inside `gpuwm-global`) take a lock file in one folder,
  `/tmp/gpuwm/locks` by default, which belongs to the first account that
  fetched; every other account's download then stops with
  `PermissionError`. `gpuwm-global` sets `GPUWM_FETCH_LOCK_ROOT` to the
  account's own `~/.gpuwm/locks` for itself and the engine calls it makes.
  `gpuwm fetch-geog` and `gpuwm fetch`, run from a shell, need it exported;
  `doctor` says when.
- **Doors.** `gpuwm-global fetch-doors` stages into `~/.gpuwm/global-doors`;
  `ARWEN_GLOBAL_DOOR_DIR` points every account at one copy instead. The
  engine's bridges stage under `~/.gpuwm/bridges` and the engine has no
  shared-folder override yet, so each account runs `gpuwm fetch-bridges`
  (a 0.19 GB download, 0.47 GB on disk).
- **Geography.** `GPUWM_WPS_GEOG` points every account's `statics` and
  `go` at one archive. `gpuwm fetch-geog` does not read it: an account that
  runs it stages its own 32 GB copy, so the accounts below skip it.

The account that stages the shared copies, once, into a folder every
account can read (here `/srv/arwen-global`, made by the machine's
administrator and writable by that account):

```bash
export ARWEN_GLOBAL_DOOR_DIR=/srv/arwen-global/doors
gpuwm-global fetch-doors
gpuwm fetch-geog --root /srv/arwen-global/WPS_GEOG
```

Every other account:

```bash
export GPUWM_FETCH_LOCK_ROOT=$HOME/.gpuwm/locks
export ARWEN_GLOBAL_DOOR_DIR=/srv/arwen-global/doors
export GPUWM_WPS_GEOG=/srv/arwen-global/WPS_GEOG
gpuwm fetch-bridges
gpuwm-global doctor
gpuwm-global fetch-analysis --cycle 2026-09-01T00 --out cases/baseline-2026090100
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

`doctor`'s "this account" section prints every folder this account uses and
which variable chose it. Two accounts that write into one shared output
folder point `GPUWM_FETCH_LOCK_ROOT` at one shared, group-writable folder
instead, so their fetches still lock each other out.

---

## A fresh global analysis

After the quick start, from the same directory, on a 16 GB card:

```bash
gpuwm-global da fresh arwen_global_gdas_t255_native_sl_si_24h \
  --outdir out/fresh --stream iem-asos --filter successive-correction \
  --analysis-cycle 2026-09-01T00:00:00Z --until-utc 2026-09-01T03:00:00Z
```

`da fresh` fetches a GDAS analysis, derives its run configuration from the
base experiment named on the line, builds the ensemble if the output
directory has none, cycles hourly to the last observation hour, and hands
back the analysis checkpoint. At least one observation source is required and
named: `--stream` fetches a stream for every window, `--obs` reads a table
already on disk, and both repeat. The last two flags fix the window to three
hourly analyses from the GDAS 2026-09-01 00Z analysis, so the line gives the
same answer whenever it runs; without them `da fresh` takes the newest GDAS
analysis and cycles to the newest observation hour.

`--filter successive-correction` is the deterministic analysis with a single
member. Without `--filter`, `da fresh` runs the ensemble filter, `letkf`,
with 32 members at T127 under the T255 control (`gpuwm-global statics
arwen_global_gdas_t127_native_sl_si_24h` builds the members' statics
first), and that needs a card larger than 16 GB.

**The card.** `da fresh` is not priced before it starts, as `run` and `go`
are. On an RTX 5070 Ti, 16 GB, the line above passed: three of three
cycles, 829 s for the whole command (the first cycle 583 s), 11,755 MiB of
card at its peak, and its receipt's `forecast_command` is the `da forecast`
line below (measured 2026-11-23 from a fresh account). On the same card the
ensemble filter ran out of card in the control's first hour, with 16 members
on the window above and with 32 on the newest analysis (15.7 GB allocated
both times, measured 2026-11-23), and public 0.3.0 does the same with 32;
this release ends there with one sentence naming the card and the remedy,
at exit 4. The ensemble figures further down were measured on an RTX 5090,
32 GB.

The receipt `out/fresh/da-receipt.json` names the analysis checkpoint it handed
back and carries `forecast_command`, the exact line that forecasts from it:

```bash
gpuwm-global da forecast out/fresh/fresh-config.toml \
  --analysis out/fresh/arwen_global_analysis_step00000036.npz \
  --outdir out/fresh/forecast
```

The forecast runs to the derived configuration's duration, which is the cycle
span plus `--forecast-hours` (24 by default); `--until-s` stops it earlier. The
legs are separately reachable: `da init` builds the ensemble, `da cycle` runs
the hourly windows over named streams, `da analyze` analyses one instant. Every
analysis writes a report with the observation counts, the quality-control
rejections by name, the o-minus-b and o-minus-a fits by stream and variable,
and the Desroziers diagnostics.

`gpuwm-global obs streams` prints the ten observation streams with the Rust
door each one decodes through, whether that decoder is built, and its measured
latency behind real time; `gpuwm-global obs fetch` writes a stream's table to
disk for `--obs`.

**Two measured assimilation arms.** The first used six hourly LETKF cycles
from the GDAS 2026-08-31 18Z analysis, 32 T127 members under a T255 control,
and METAR, IGRA2, NDBC and GOES derived motion winds. It offered 879,488 rows
over six windows; METAR surface-pressure o-minus-b grew from 115 to 158 Pa
while o-minus-a held at 76 to 82 Pa (measured 2026-09-06).

A later arm started from GDAS 2026-09-01 18Z and cycled to 2026-09-02 00Z.
Against the stations, its handed-back analysis reads 2 m temperature
+0.45 / 2.10 K against the same-instant GDAS analysis's -0.35 / 2.35,
2 m dewpoint +0.16 / 2.60 against -2.53 / 4.10, and mean sea-level pressure
-1.05 / 2.44 hPa against -1.68 / 2.79 (bias / RMSE). Its 10 m wind RMSE is
worse: 2.05 against 1.91 m/s, on the same 1,591 stations. These reports
come from the METAR/ASOS record the cycle assimilated, so the analysis scores
measure fit to assimilated data, not independent forecast accuracy. It is
also worse than same-instant GDAS at the soundings (measured 2026-09-06).

The later arm's subsequent 24 h forecasts improve 2 m temperature and dewpoint
relative to the GDAS cold start but lose on sea-level pressure, 10 m wind,
500 hPa height at 12 h, and every sounding temperature and wind row. The
[DA evidence](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_DA.md#the-first-real-arm-and-what-ships)
separates the two analysis times and the forecast scores. The system remains
**selectable, not default**; these case results do not establish general skill.

The analysis runs on the card. The analysis proper (the step less the members'
forecast hour and the control re-integration) costs 26.4 s on an RTX 5090,
against 185.8 s on the host path, and the whole hourly cycle costs 390.5 s, so
a cycle keeps up with the hour it covers by a factor of eight
(MEASURED 2026-09-07).

**The quickstart itself, run from the installed wheel.** Three hourly cycles
from the GDAS 2026-09-01 00Z analysis over the public surface stream on the
shipped deterministic filter, on an RTX 5070 Ti shared with another job:
status `pass`, three of three cycles engineering-complete, 596.5 s for the
whole command, 76.1 s a cycle on average and 89.6 s at worst, which is 0.025
of the hour each cycle covers. At the third analysis 20,713 rows were
assimilated and 2,302 withheld from it for the gate of record. Station
pressure read 403.2 Pa o-minus-b against 113.2 o-minus-a on 4,519 rows,
temperature 1.946 against 1.816 K on 4,717, and the wind components 2.157
against 2.047 and 2.156 against 2.085 m/s. Dewpoint read 3.383 against 3.411,
worse after the analysis than before it, and the door printed that as a
reading rather than hiding it: `--moisture-update` is off by default, so
dewpoint rows are scored and not analysed (MEASURED 2026-09-07).

---

## What it runs on

A 24 hour forecast, 288 steps at 300 s on the shipped semi-Lagrangian core, the
full native physics suite, 40 hybrid levels, float32. Wall clock is end to end
from each run's own receipt. "Shared" means another process held the card, so
the row is a capability row rather than a timing.

Every row was measured on the date in its last column. Rows dated 2026-09
were measured in the engine checkout this model was developed and graded in,
whose version string reads 2.6.4; rows dated 2026-11 on the published
`gpuwm 2.9.12` and `2.9.13`, with this release's physics (CAM ozone, the top
drag, the dated surface series).

| Truncation | Gaussian grid | Equator spacing | Card | A forecast day | Device peak | Measured |
|---|---|---|---|---|---|---|
| T63 | 96 x 192 | 208.5 km | RTX 5090, 32 GB, alone | **1.8 min** (108.0 s; 0.363 s a step), `arwen_global_gdas_t63_native_sl_si_24h` (experimental, [not a forecast](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-coarse-grids.md)) with no flag | 2.50 GiB live, 3,508 MiB on the card | 2026-11-12 |
| T127 | 192 x 384 | 104.2 km | RTX 5090, 32 GB, alone | **2.9 min** (171.8 s; 0.514 s a step), `arwen_global_gdas_t127_native_sl_si_24h` (experimental, not a forecast) with no flag | 3.32 GiB live, 5,180 MiB on the card | 2026-11-12 |
| T255 | 384 x 768 | 52.1 km | RTX 5070 Ti, 16 GB, alone | **4.28 min** (256.7 s) | 8.70 GiB | 2026-09-06 |
| T255 | 384 x 768 | 52.1 km | RTX 5070 Ti, bare from the door | 245.8 and 246.0 s | 8.70 GiB | 2026-09-07 |
| T255 | 384 x 768 | 52.1 km | RTX 5090, shared | 186.8 and 228.9 s | 8.70 GiB | 2026-09-07 |
| T255 | 384 x 768 | 52.1 km | RTX 5070 Ti, 16 GB, alone | **4.04 min** (242.4 s; 0.747 s a step), the shipped day with no flag, from the GDAS 2026-09-01 00Z analysis; 0.1.6 on the same card the same day 4.83 min (0.914 s a step), byte-identical | 6.70 GiB live, 9,028 MiB on the card (0.1.6: 8.63 GiB, 11,582 MiB) | 2026-11-12 |
| T255 | 384 x 768 | 52.1 km | RTX 5070 Ti, 16 GB, the card alone, the host running other work | 0.4.0, the shipped day through `run`: 0.824 and 0.595 s a step in two runs; public 0.3.0 alternating with them on the same card, 0.901 and 0.721 s; the same numbers | 5.63 GiB live | 2026-11-23 |
| T255 L80 | 384 x 768 | 52.1 km | RTX 5070 Ti, 16 GB, alone | **6.50 min** (390.2 s; 1.313 s a step), `arwen_global_gdas_t255_l80_satmedmf_scinocca_24h` (opt-in) through `go` with no flag, from the GDAS 2026-09-01 00Z analysis: one band, nothing parked, the pool held from 12.49 GiB and raised to 13.96 | 12.49 GiB live, 15,018 MiB on the card; a second day from an installed wheel 15,466 MiB, the pool raised to 14.05 | 2026-11-23 |
| T255 L80 | 384 x 768 | 52.1 km | RTX 5070 Ti, 16 GB, alone | **5.94 min** (356.5 s; 1.193 s a step), `arwen_global_gdas_t255_l80_ysu_scinocca_24h` (opt-in, the YSU twin) through `go` with no flag, the same analysis and plan | 9.54 GiB live, 12,730 MiB on the card | 2026-11-23 |
| T383 | 576 x 1152 | 34.7 km | RTX 5090, shared | **5.63 min** (337.9 s) | 15.86 GiB | 2026-09-07 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **11.8 min** (2.46 s a step; 12.4 min with the cold start), one band, nothing parked on the host | 12.10 GiB live, 15,624 MiB on the card | 2026-11-07 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone, 10 days | 108.9 min for 240 h (2.24 s a step) | 12.10 GiB live, 15,212 MiB on the card | 2026-11-06 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **10.7 and 10.8 min** (2.220 and 2.250 s a step over two days), one band, nothing parked, the pool held from 11.50 GiB and raised to 13.86 and 13.15 | 11.95 GiB live, 14,608 and 13,880 MiB on the card | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **13.7 min** (2.852 s a step), two bands, every slice parked, the pool held from 9.25 GiB and raised to 10.43 | 8.99 GiB live, 11,090 MiB on the card | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, 2,270 MiB held by another process | **13.7 min** (2.863 s a step; 14.8 min with the cold start), chosen by the door at 13.02 GiB free with no flag: two bands, every slice parked | 9.52 GiB live, 12,486 MiB on the card | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, 4,318 MiB held by another process | **14.1 min** (2.941 s a step; 15.1 min with the cold start), chosen by the door at 11.02 GiB free with no flag: four bands, every slice parked (on the door's figures of 2026-11-11; it now takes two bands there) | 7.29 GiB live, 11,342 MiB on the card (the pool grows into the room it has) | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **14.1 min** (2.933 s a step; 15.1 min with the cold start), `--latitude-bands 4 --host-spill on`, the pool held under 10.0 GiB | 7.29 GiB live, 10,582 MiB on the card | 2026-11-11 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **14.9 min** (3.098 s a step), `--latitude-bands 4 --host-spill on --device-allocator async` | 7.23 GiB live, 9,676 MiB on the card | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **9.6 min** (578.5 s; 1.805 s a step), the shipped day with no flag, from the GDAS 2026-09-01 00Z analysis: one band, nothing parked, chosen by the door at 15.24 GiB free, the pool held from 11.50 GiB and raised to 13.66; 0.1.6 on the same card the same day 15.3 min (2.981 s a step) at two bands with every slice parked, byte-identical | 11.94 GiB live, 14,392 MiB on the card (0.1.6: 12.00 GiB, 15,142 MiB) | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 5070 Ti, 16 GB, alone | **10.2 min** (610.7 s; 1.907 s a step), `arwen_global_gdas_t383_native_24h`, the config of record on the default core, with no flag, from the GDAS 2026-08-30 18Z analysis: one band, nothing parked (the day on the chunk change's own tree, before the stored checkpoints joined it); over 3 h 1.832 s a step where 0.1.7, at the 12,500-column radiation chunk it carried, took 2.535 at two bands with every slice parked, byte-identical | 11.84 GiB live, 14,316 MiB on the card | 2026-11-12 |
| T383 | 576 x 1152 | 34.7 km | RTX 4090, 24 GB, shared | **15.1 min** (904.5 s; 3.07 s a step), `arwen_global_gdas_t383_native_sl_si_24h` with `--latitude-bands 2 --host-spill on` (the run's share of the card was 12.5 GiB), from the GDAS 2026-09-01 00Z analysis, run from a clean install | 9.24 GiB live, the pool raised to 10.26 GiB; 2.28 GiB parked in pinned host memory | 2026-11-22 on `gpuwm 3.0.5` |
| T533 | 801 x 1602 | 25 km | RTX 5090, 32 GB, alone | **15.9 min** (953 s; 2.98 s a step) | 23.23 GiB live, 28.11 held | 2026-09-07 |
| T533 | 801 x 1602 | 25 km | RTX 5070 Ti, 16 GB, alone | **34.0 min** (6.67 s a step) with `--streaming on --device-allocator async`: four bands, every slice parked. Only with those two flags: with none, this release's door refuses the day on a 16 GB card (its lightest plan asks 20.13 GiB of card); with them it prices 15.22 GiB at 15.24 GiB free and admits it | 12.81 GiB live, 15,756 MiB on the card | 2026-11-06 on `gpuwm 2.8.9`; admitted by the door and its first hour rerun on `2.9.13`, 2026-11-07; priced by this release's door 2026-11-22 |
| T799 | 1200 x 2400 | 17 km | 16, 24 or 32 GB | **refused at the door**, before a byte is allocated, with the figure that refuses it | priced 45.96 GiB of card at its lightest plan (64.50 GiB at one band, the card a completed run of that plan took) | priced 2026-11-22 |

At the same truncation on the same tree, the selectable Eulerian core
`imex_ssp3` takes 1,099.3 s for the T255 day at a 60 s step and 1,050.2 s at
its shipped 90 s step on the RTX 5070 Ti (MEASURED 2026-09-06 and 2026-09-07).
A 300 s semi-Lagrangian step costs about one Eulerian step, so the day is 288
steps instead of 960.

**What this release's door asks for.** Each shipped forecast day, priced by
the memory gate `gpuwm-global run` applies before it allocates anything, in
GiB of card, at three free figures: 15.24 GiB (an empty 16 GB RTX 5070 Ti as
the door read it), 20.64 GiB (a 24 GB RTX 4090 beside other tenants as the
door read it) and 31.0 GiB (stated, for a 32 GB card). The plan the door
takes is beside each figure (PRICED 2026-11-23 on the 0.4.0 tree, every shipped config). The
80-level pair is priced at the card the heavier of its two measured
SA-TKE-EDMF days took (15,018 and 15,466 MiB: where a held pool ends is set
by how its blocks were cut), so the YSU twin is asked for more than it
holds. Only the one-band 80-level plan has been measured. A banded plan at
80 levels, chosen with `--latitude-bands` or `[memory]`, is priced by the
memory model outside the level counts it was fitted on (20, 40 and 48) and
is not a measured figure; the door itself takes one band for these days at
every free figure in the table.

| Experiment | 15.24 GiB free | 20.64 GiB free | 31.0 GiB free |
|---|---|---|---|
| `arwen_global_gdas_t63_native_sl_si_24h` | 4.78, one band | 4.78, one band | 4.78, one band |
| `arwen_global_gdas_t127_native_sl_si_24h` | 5.98, one band | 5.98, one band | 5.98, one band |
| `arwen_global_gdas_t255_native_sl_si_24h` | 8.96, one band | 8.96, one band | 8.96, one band |
| `arwen_global_gdas_t383_native_sl_si_24h` | 14.27, one band | 14.27, one band | 14.27, one band |
| `arwen_global_gdas_t533_native_sl_si_24h` | refused: its lightest plan asks 20.13 | 20.13, 32 bands, three slices parked | 28.92, one band, three slices parked |
| `arwen_global_gdas_t799_native_sl_si_wall` | refused: 45.96 | refused: 45.96 | refused: 45.96 |
| `arwen_global_gdas_t1023_native_sl_si_24h` | refused: 75.16 | refused: 75.16 | refused: 75.16 |
| `arwen_global_gdas_t255_l80_satmedmf_scinocca_24h` (opt-in) | 15.10, one band | 15.10, one band | 15.10, one band |
| `arwen_global_gdas_t255_l80_ysu_scinocca_24h` (opt-in) | 15.10, one band | 15.10, one band | 15.10, one band |

At 94.0 GiB free (stated, for a 96 GB card) the door admits T799 at 64.50 GiB
and T1023 at 85.93 GiB, each at one band and each priced at the card a
completed run of exactly that plan took. The other shipped T255 configs ask
8.96 GiB on the semi-Lagrangian core at 40 levels, 13.03 GiB at 48 levels
(`arwen_global_gdas_t255_jet48_24h`), and 10.78 GiB on the Eulerian `imex`
core. `gpuwm-global run-plan PLAN.json --estimate` prints the same figures
for any config against the card it runs on.

**The card is priced, not guessed.** `run` estimates the device peak, weighs it
against free VRAM, chooses the latitude band count and the host tier, and
refuses a plan that will not fit, naming which allocation dies first. Across
seventeen legs with a card reading, the gate was on the refusing side every
time, the tightest by 0.047 GiB (MEASURED 2026-09-06). Where a completed run of
a lighter plan bounds a heavier one, the door prices at that measurement
instead of at its model, and the receipt names the run that bounded it.

**A 16 GB card with a desktop on it.** A run holds its CuPy pool to a limit,
so the pool no longer grows into whatever the card has free: T383 at one band
took 15,624 MiB of a 16 GB card over a day before the limit and 13,880 and
14,608 MiB over two days held to it (the pool held from 11.50 GiB and raised
to 13.15 and 13.86 GiB: where it ends is set by how its blocks were cut, not
by the forecast), byte-identical, 2.250 and 2.220 s a step against 2.235 for
the tree before. The door prices it at the heavier day and asks 14.27 GiB free
for it at the 5,000-column radiation chunk, where it asked 14.93. With less free it
takes two latitude bands with every slice parked in pinned host memory from
10.83 GiB free, or four from 10.33, priced at what those plans' own runs took,
and its line says what the plan costs against one band and what the pool is
held to. The banded rows above are those plans: 28 and 32 percent slower than
one band on the same card, 2.27 GiB in pinned host memory, and about 2 GiB
more host RAM at the peak (7.4 GiB against 5.3). Below 10.2 GiB free the door
refuses and names `--device-allocator async`, whose four-band day took 9,772
and 9,676 MiB of the card over two 24 h days at 3.087 and 3.098 s a step,
byte-identical to the one-band day; the door prices it at the heavier
(MEASURED 2026-11-12). A `CUPY_GPU_MEMORY_LIMIT` set in the environment
stands instead of the run's own limit. The shipped T383 semi-Lagrangian day
carries the 5,000-column radiation chunk these rows were measured at; at the
12,500 it carried before, the step peaked 0.99 GiB higher live and took 2.42
to 2.45 s against 2.28 to 2.29 over 3 h, byte-identical (MEASURED
2026-11-12), and a config that still says 12,500 is pointed at
`radiation_column_chunk = 5000` by the door's refusal. From 0.1.8 so does
`arwen_global_gdas_t383_native_24h`, the config of record on the default
core, and the door runs it at one band on an empty 16 GB card (the row
above).

**A 12 GB card.** No 12 GB card was measured; the 16 GB card stood in for
one, with another process holding all but what a 12 GB card leaves free.
From the door with no flag, 24 h T383 days from the GDAS 2026-11-11 00Z
analysis: at 11.59 and 11.05 GiB free it chose two bands with every slice
parked, 10,880 and 10,972 MiB of the card at 2.858 and 2.871 s a step (13.7
and 13.8 min a forecast day); at 10.52 GiB free it chose four, 10,322 MiB at
2.961 s (14.2 min). Every array of all nine checkpoints of each day is
byte-identical to the one-band day (MEASURED 2026-11-12, RTX 5070 Ti). The
minutes are that card's: a 12 GB card's own step was not measured.

**Banding does not move a bit.** The physics suite runs a latitude band at a
time, and the ten-step T255 gate of record is byte-identical between one band
and eight: 309 of 309 checkpoint arrays, on both cards (MEASURED 2026-09-07).
The host tier is the same: 442 of 442 checkpoint arrays byte-identical with the
tier off against on (MEASURED 2026-09-07). Over a T383 day it holds as well:
two and four bands with every slice parked reproduce the one-band day's 133
checkpoint arrays byte for byte at step 288 (MEASURED 2026-11-11 and
2026-11-12, RTX 5070 Ti). The banded transforms are exact too:
93 of 93 FFT cases bit-exact at 0 ULP across T255, T383 and T533 crossed with
band counts 2 to 32, and 892 of 892 waist comparisons bit-exact on each card
(MEASURED 2026-09-06).

**Opt-in physics (0.4.0).** The default day above is unchanged. Two shipped
days run the 80-level grid with the new schemes, by name, from the quick
start's analysis: in its `go` line, put
`arwen_global_gdas_t255_l80_satmedmf_scinocca_24h` where the default day is
named (and a new `--outdir`), or
`arwen_global_gdas_t255_l80_ysu_scinocca_24h`, its twin with YSU. In any
native config, `[vertical] coordinate = "ifs_class"` gives 80 levels to a
1 hPa lid; in `native_adapter_options`, `pbl = "satmedmf"` is the SA-TKE-EDMF
boundary layer in single precision (`"satmedmf_fp64"` the engine's double
precision column), `nonorographic_gwd = "scinocca"` the non-orographic
gravity-wave drag, `microphysics = "tempo"` TEMPO, and `radiation_aerosol =
"tegen-wrf"` WRF's aerosol climatology. None is a default and no skill is
claimed for them; the sources and licences are in NOTICE. Two things to know
before choosing an 80-level day: its rain is not yet right (run 5 days, it
fell in regular north-south streaks about 1 degree apart over the tropical
east Pacific, and light rain covered most of the subtropical oceans where the
default day has none; the CHANGELOG's known limits), and its nine checkpoints
take 22 to 23 GB of disk, against 12.3 GB for the default day.

---

## The doors

One console script, 64 commands. `gpuwm-global <command> --help` prints every
option; [docs/CLI-REFERENCE.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/CLI-REFERENCE.md) is the same surface on
one page, generated from the parser.

| Command | What it does |
|---|---|
| `go` | statics, forecast, tapes and pictures in one command |
| `run` | run a configured experiment: the forecast, its gates and its receipt |
| `statics` | build and cache the surface static fields for a truncation |
| `surface-series` | build a dated SST, sea-ice and greenness series on a config's grid, from GDAS analyses or one GFS cycle's own forecast, for `run --surface-series` |
| `climatology fetch`, `climatology build` | stage the public monthly SST and sea-ice climatology (NOAA OISST v2.1, 1991-2020; optionally the WOA18 mixed-layer depth), each file verified by size and SHA-256, and build it onto a config's grid as one byte-reproducible file |
| `ocean build-grid`, `ocean build-init`, `ocean init-from-state`, `ocean run` | the experimental dynamic ocean (`[ocean] mode = "dynamic"`, off by default): build its grid from ETOPO 2022 and its start from NOAA GODAS and WOA23, and spin it up ocean-only under the GODAS fluxes ([docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md)) |
| `export` | render-ready global lat-lon `wrfout` tapes from checkpoints (`export-wrfout` is an accepted alias of the same door) |
| `render` | draw the products from a finished run through the Rust renderer |
| `configs` | list the experiments that ship inside the package |
| `world list`, `world new`, `world describe`, `world twin`, `world perturb`, `world diff` | other worlds: the shipped presets, a world config written from an Earth one, what a world changes before any card time, and a twin with its own noise floor (the control, a control perturbed at rounding level, and world-minus-Earth tapes that show a difference only where it beats the two controls' and print the uniform pressure offset they remove) |
| `doctor`, `fetch-doors` | what is installed, staged and missing; and stage what is missing |
| `fetch-analysis` | fetch the one whole-globe GDAS object a cold start needs, or with `--source era5` one ERA5 hour from 1940 on ([ERA5 start](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_ERA5.md)), and print the run it feeds |
| `score` | score a finished run against ASOS stations and radiosondes through `rw_asos` and `rw_igra2`, with the GFS and IFS analyses as secondary references, one row per lead, at any truncation |
| `assimilate` | assimilate point observations into one checkpoint |
| `cycle` | forecast and assimilate in one process, an analysis every interval |
| `da init`, `da cycle`, `da analyze`, `da fresh`, `da forecast`, `da static-covariance`, `da localisation` | the ensemble door: build, cycle, analyse one instant, do all of it from nothing, forecast from the analysis, estimate the hybrid's static covariance from lagged forecast pairs, derive the localisation radii from the members |
| `obs fetch`, `obs subscribe`, `obs hours`, `obs summary`, `obs streams`, `obs anchors` | the observation tables: fetch them, subscribe to the live feed, see which hours are covered, what each stream carries and where it is anchored |
| `microwave` | the ATMS leg: fetch, decode, thin and score microwave radiances |
| `abi-score`, `abi-reference`, `abi-fast-model` | the GOES ABI infrared operator, its CRTM reference leg and its trained fast model |
| `export-mapped-parent` | regional parent frames from checkpoints: model-level NetCDF on a regular lat-lon grid with the mapping and composition documents `gpuwm prep --source mapped` reads, so the forecast is the initial and lateral boundary data of a regional run |
| `export-parent`, `make-regional-target`, `translate-regional-frame`, `make-parent-series` (and their `inspect-` twins) | the one-way bridge that hands a limited-area model its lateral boundaries |
| `pins`, `physics-manifest`, `transform-check` | the arithmetic identity, the admitted physics adapters, the transform controls |
| `inspect`, `check-receipt`, `check-migration`, `check-native-evidence`, `check-native-candidate` | validate a checkpoint, a receipt, a migration, a device-qualification record |
| `migrate-level4-checkpoint` | carry an older checkpoint onto the current state contract |
| `native-qualify` | run the target-device qualification battery for the native physics suite |
| `run-plan` | the machine seam: discover sources, products and physics, probe the machine, resolve and estimate a plan, then execute it and leave a durable event stream |
| `sources` | every source this model initializes from or scores against, and which authority table answered its mapping: the engine's or this package's carried copy |

`run`, `go`, `render` and `run-plan` write `status.json` into their output directory,
rewritten atomically at each stage and at each step, and an append-only log
beside it: these are the four long-running doors, and a workspace driving one
of them reads its stage, its progress and its ending from that file rather
than by scraping a growing console. The shorter commands print their answer
and are read by watching stdout. Exit codes are `0` success, `1` a refusal
that names its reason, `2` an argument error, `3` a Rust door missing or
failing its pin, `4` the device declining admission, and, from `go` only,
`5` a forecast that stopped before its end (a refusal inside the model)
after the hours it reached were drawn into `pictures/`. Every finished run leaves
a self-hashed receipt carrying the configuration identity, the arithmetic
pins, the library versions the numbers rode on, the engine version and the
staged binary digests, so a run can be compared to a configuration without
opening a checkpoint.

### Driving it from a program

A desktop application, a terminal workspace or a CI job drives this model
through the same versioned contract the engine publishes, so a client written
for `gpuwm` drives `gpuwm-global` by changing the module it spawns.

```bash
gpuwm-global sources --json                          # gpuwm.run-plan.sources.v1
gpuwm-global run-plan --catalog                      # gpuwm.run-plan.catalog.v1
gpuwm-global run-plan --physics-profiles             # gpuwm.run-plan.physics-profiles.v1
gpuwm-global run-plan --probe --no-readiness         # gpuwm.run-plan.probe.v1
gpuwm-global run-plan plan.json --resolve            # gpuwm.run-plan.resolved.v1
gpuwm-global run-plan plan.json --estimate           # gpuwm.run-plan.estimate.v1
gpuwm-global run-plan plan.json                      # execute; events on stdout
```

A plan is a `gpuwm.run-plan.v1` envelope over an experiment TOML, and
`config.path` takes a shipped experiment's bare name. A run writes
`run-manifest.json`, `run-progress.json` and `events.jsonl` beside the
`status.json` above, all three in the engine's own schemas, so a client
reattaches to a run of this model with the reader it already has.
`python -P -m arwen_global.tui_worker --job-dir DIR -- <args>` is the same
durable handshake as `gpuwm.tui_worker` for a detached job.

[docs/ARWEN_GLOBAL_CLIENT.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_CLIENT.md) is the whole
contract, including what the engine offers that this package does not and why.

---

## What it looks like

These came out of the installed package on an RTX 5070 Ti: a 24 hour forecast
from the public GDAS 2026-09-01 00Z analysis at T255, valid 2026-09-02 00Z,
exported to `wrfout` tapes and drawn by the Rust renderer. Eight products at
nine forecast hours, 72 pictures; four of them here, the rest with their
captions in [docs/demo/CAPTIONS.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/demo/CAPTIONS.md).

| | |
|---|---|
| ![Precipitable water at 24 hours](https://raw.githubusercontent.com/FahrenheitResearch/gpuwm-global/v0.4.0/docs/demo/pwat-f024.png) | ![500 hPa height and wind at 24 hours](https://raw.githubusercontent.com/FahrenheitResearch/gpuwm-global/v0.4.0/docs/demo/z500-f024.png) |
| **Precipitable water.** The intertropical convergence zone, the monsoon over the Bay of Bengal and the South China Sea, mid-latitude moisture plumes, and the dry subtropical highs and Antarctic. | **500 hPa height and wind.** The austral winter jet runs unbroken around the Southern Ocean at over 70 m/s; the northern jet is weaker and split, with closed lows over Siberia and the North Pacific and 594 dam ridges over both ocean basins. |
| ![Mean sea level pressure and 10 m wind at 24 hours](https://raw.githubusercontent.com/FahrenheitResearch/gpuwm-global/v0.4.0/docs/demo/mslp-f024.png) | ![2 m temperature at 24 hours](https://raw.githubusercontent.com/FahrenheitResearch/gpuwm-global/v0.4.0/docs/demo/t2m-f024.png) |
| **Mean sea level pressure and 10 m wind.** The Southern Ocean storm belt carries lows at 949.9 and 961.5 hPa; tropical cyclones sit in the west Pacific and the Indian Ocean; the subtropical highs close at 1030 hPa. | **2 m temperature.** Late northern summer over the continents, the Sahara and Arabia at their hottest, the Antarctic plateau at its coldest, and the Andes and Himalaya drawn by their own orography. |

And the analysis cycle draws the same way. This is the state the three-cycle
quickstart above handed back, 2 m temperature valid 2026-09-01 03Z, exported
and drawn by `gpuwm-global` in one command:

![2 m temperature at the third analysis](https://raw.githubusercontent.com/FahrenheitResearch/gpuwm-global/v0.4.0/docs/demo/analysis-t2m-f003.png)

The mean sea level pressure picture that same run rendered at its own step 0 is
byte-identical to the forecast demo's, `md5 b772960e…`, from a different run
on a different day through a different export and a different render call
(MEASURED 2026-09-07).

---

## Other worlds

Experimental: a research tool, not a forecast product. The same model runs
Earth changed: no mountains, another tilt, a brighter or dimmer sun, other
CO2, a higher sea, no land, or a continent from a paleo-elevation map. A world is a `[world]` table at the end of an ordinary
config, and an all-Earth table changes nothing: the config hashes the same and
ten steps write the same bytes.

```bash
gpuwm-global world list
gpuwm-global world new flat-earth \
    --from arwen_global_gdas_t255_native_sl_si_24h --out flat-earth.toml
gpuwm-global world describe flat-earth.toml
```

Every world is checked on the CPU before any card time, labelled science,
illustration or not supported yet, and named on its receipt, checkpoints and
tapes. The sea is held for the whole run, so every result says: *This is the
weather of the first N days after Earth was changed, over a sea surface that
does not respond. It is not this world's climate.*
[docs/ARWEN_GLOBAL_WORLDS.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_WORLDS.md) is the whole
surface, its limits, what was measured, and the credits for the published
maps and profiles it uses.

---

## Where things come from

- **The engine.** `gpuwm` supplies the GRIB2 source decoder, the observation
  front door, the static-field builder, the LETKF filter core, the `wrfout`
  writer, the physics tables and the renderer. This package supplies the
  spectral core, the global dynamics, the global physics registry, the global
  data assimilation and the doors.
- **Every data path is Rust.** GRIB decode, NetCDF read and write, observation
  decoding, regridding and rendering run in Rust binaries the engine stages;
  Python is the command line and the CUDA driver code. The binaries arrive as
  release assets and are verified by size, SHA-256 and a contract marker before
  they are used.
- **The observation doors** the assimilation calls are `rw_asos` (surface
  networks), `rw_igra2` (radiosondes), `rw_ndbc` (buoys), `rw_amv` (satellite
  motion vectors), `rw_gnssro` (radio occultation), `rw_wis2` (the WMO
  information system), `rw_goes` (ABI) and `rw_atms` (microwave).
- **The render kernels** (`global_render_kernels`, built from this
  repository's `rust/global-render-kernels`) regrid the Gaussian grid onto
  every render tape, integrate the tape's hydrostatic column and translate
  parent exports onto regional targets. They ship in the same companion
  bundle as the observation doors.
- **Statics** come from the public geography archive through the engine, cached
  once per truncation.
- **The monthly sea-surface climatology** is NOAA OISST v2.1's 1991-2020
  monthly long-term mean of SST and sea-ice concentration from NOAA PSL
  (45,945,542 and 31,003,804 bytes), and, with `--with-mld`, the NOAA WOA18
  monthly mixed-layer depth (twelve files, 85,127,108 bytes, found in the
  regional engine's ocean cache when it has them). No route needs a
  credential. The rows, with their byte counts, SHA-256 pins, licences and
  citations, are `arwen_global/data/surface_climatology_sources.json`.
  `gpuwm-global climatology build` puts them on a config's Gaussian grid with
  the cold start's open-water rule over the statics ocean, NaN on land and
  inland lakes, and writes the same bytes from the same inputs; the file's
  provenance gives, for every month, the mean SST and each hemisphere's ice
  extent and area on the source grid and on the model grid, so what the
  regrid and the coastline change is stated in the file. It is a
  climatology, not any year's observation.

---

## Documentation

| Page | What it is |
|---|---|
| [ARWEN_GLOBAL_QUICKSTART.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_QUICKSTART.md) | the route above, with the reasoning behind each flag |
| [ARWEN_GLOBAL.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL.md) | what ships, the resolution rungs, and where the envelope ends |
| [ARWEN_GLOBAL_FULL.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_FULL.md) | the model: dynamics, hybrid coordinate, physics, receipts, and the non-claims |
| [ARWEN_GLOBAL_DA.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_DA.md) | the assimilation door, its gates and its grade |
| [ARWEN_GLOBAL_LEVEL5.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_LEVEL5.md) | the native CUDA physics suite and its device-admission battery |
| [ARWEN_GLOBAL_REGIONAL_PARENT.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_REGIONAL_PARENT.md) | the global forecast as initial and lateral boundary data of a regional run: the command, the file contract, what is and is not carried |
| [ARWEN_GLOBAL_FORCING.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_FORCING.md) | experimental: the forcing operator, a run pulled toward a stored record of large scales, and the pull read as model error |
| [LEVEL3_GLOBAL_SPECTRAL_DYCORE.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/LEVEL3_GLOBAL_SPECTRAL_DYCORE.md) | the spectral dynamical core |
| [arwen-global-effective-resolution.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-effective-resolution.md) | what the model resolves, as against what its grid spacing says |
| [arwen-global-coarse-grids.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-coarse-grids.md) | the T63 and T127 days (experimental): what they cost on a card and what they are for |
| [arwen-global-scm-cases.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-scm-cases.md) | the physics suite on seven published single-column cases, scored against their LES ranges (experimental) |
| [arwen-global-twin-scan.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-twin-scan.md) | exact-twin starts and the energy-norm reading of their differences (experimental) |
| [arwen-global-ensemble-da.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-ensemble-da.md) | the ensemble filter, its localisation and its inflation |
| [arwen-global-members.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-members.md) | experimental: forecast members from one analysis, their replay recipes and exact twins |
| [arwen-global-transplant.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-transplant.md) | experimental: one physics scheme's output carried between twins, and how much of their difference it carried |
| [arwen-global-composition-replay.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-composition-replay.md) | experimental: extra transported fields (tags, plumes, ozone) carried over a finished run's checkpoints |
| [arwen-global-radiation-between.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-radiation-between.md) | experimental: the shortwave between two radiation calls following the sun |
| [arwen-global-observations.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-observations.md) | the observation streams, their decoders and their error models |
| [arwen-global-abi-operator.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-abi-operator.md) | the GOES ABI clear-sky infrared operator |
| [arwen-global-microwave-operator.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-microwave-operator.md) | the ATMS clear-sky over-ocean operator |
| [arwen-global-airliner-heights.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/arwen-global-airliner-heights.md) | experimental: the ADS-B altitude pair as an upper-air height observation, and one day measured |
| [ARWEN_GLOBAL_WORLDS.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_WORLDS.md) | other worlds: the `[world]` table, the presets, the checks before any card time, the twins and their noise floor, what is refused, and the credits |
| [ARWEN_GLOBAL_CLIENT.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/ARWEN_GLOBAL_CLIENT.md) | driving this model from a program: the versioned documents, the plan envelope, the durable run files, and what is not offered |
| [CLI-REFERENCE.md](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/docs/CLI-REFERENCE.md) | every command and every option, generated from the parser |

---

## Licence

Apache-2.0. See [LICENSE](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/LICENSE) and [NOTICE](https://github.com/FahrenheitResearch/gpuwm-global/blob/v0.4.0/NOTICE).
