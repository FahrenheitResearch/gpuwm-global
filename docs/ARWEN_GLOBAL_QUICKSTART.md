# ArWen Global quickstart: a 24 h T255 global forecast from one analysis

The README's quick start, one step at a time, with the reason for each
flag. It is the same route, the same experiment and the same analysis:
`arwen_global_gdas_t255_native_sl_si_24h`, the full native physics suite on
the semi-Lagrangian core, cold-started from the public GDAS 2026-09-01 00Z
analysis. Every command is a door that already ships; this page is the
route through them, not a new tool.

ArWen Global is an **experimental research model**. It is reached through
`gpuwm-global`, its configuration surface can move between releases, and its
output is not a supported product. What ships and where the envelope ends
are in [ARWEN_GLOBAL.md](ARWEN_GLOBAL.md); what the model is, and the list
of things it explicitly does not claim, are in
[ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md).

## What you get

A 24 hour global forecast at T255 -- a 384 x 768 Gaussian grid, 52.1 km at
the equator, 40 surface-stretched hybrid layers to 100 Pa -- cold-started
from a single GDAS 0.25 degree analysis, nine checkpoints three hours apart,
and the product maps drawn by the same Rust renderer that draws every other
ArWen product.

## What you need

- **A CUDA card the door admits.** The door prices this day at 8.96 GiB
  of card and refuses before it allocates anything when the card has less
  free; a 16 GB card holds it. Measured (RTX 5070 Ti, 2026-11-23): 5.63
  GiB live at the run's peak, its pool 6.80 GiB. There is a no-card
  rehearsal at the bottom of this page.
- **About 18 GB of host memory** while the statics are built (17.2 GiB
  peak resident, measured); the forecast itself took about 4.5 GB.
- **About 47 GB of disk** in your home: the geography archive 33.6 GB, the
  day's nine checkpoints 12.3 GB, the engine's bridges 0.47 GB and the
  analysis 0.47 GB, with 2.9 GB more if you score the run. The install
  takes 3.3 GB, and pip keeps a 2.0 GB download cache.
- **About 5.5 GB of download**: the install 2.04 GB, the geography
  2.75 GB, the analysis 0.48 GB and the bridges 0.19 GB.
- **No credentials.** GDAS, the geography mirror and the release assets
  are public.

## 0. Install and stage the doors

```bash
pip install "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global doctor
```

`gpu-cu12` instead of `gpu-cu13` on a CUDA 12 driver. `doctor` exits 0 when
every documented command can run here and names, for each gap, the command
it stops and the remedy. Its "this account" section prints the folders
this account uses; on a machine shared by several accounts, read the
README's "Several accounts on one machine" first.

## 1. Fetch the analysis

```bash
gpuwm-global fetch-analysis --cycle 2026-09-01T00 --out cases/baseline-2026090100
```

This lands `cases/baseline-2026090100/gdas.t00z.pgrb2.0p25.f000` with a
`SHA256SUMS`, a series TSV and a `fetch-manifest.json` recording which
endpoint served each object, and it prints the commands that follow. The
shipped T255 day reads exactly that path, relative to the directory you run
from. Where the bytes come from is table data
(`gpuwm/authorities/rw-wps-fetch-routes.v1.json`), not a code path: GDAS
declares an ordered ladder of NOMADS and the AWS Open Data archive
`noaa-gfs-bdp-pds` (which the route reads back to 2021-03-22 12Z), and the
router HEAD-probes the archive per object and lets it serve whenever it
already has the bytes.

It is the engine's GDAS fetch with two flags bound that are not that door's
defaults, and both matter (`gpuwm fetch --source gdas --hours 0 --mode
full-file` spells the same fetch by hand):

- `--hours 0` takes the f000 analysis alone. GDAS is the one source that
  accepts it, because its f000 is the assimilation cycle's own estimate of
  the atmosphere rather than a forecast field -- which is exactly what a
  global cold start wants. There are no lateral boundaries to feed, so one
  object is the whole input.
- `--mode full-file` takes the whole-globe object. The default GDAS
  transport is the NOMADS grib-filter **area crop**, and a regional crop
  cannot initialize a global model: the initializer measures the longitude
  ring and refuses one that does not close (`analysis longitude ring covers
  N degrees, not the globe`). That refusal is deliberate and stays.

## 2. Stage the geography and build the statics

```bash
gpuwm fetch-geog
gpuwm-global statics arwen_global_gdas_t255_native_sl_si_24h
```

The land surface, the soil, the vegetation and the orography come from the
WPS_GEOG archive. `gpuwm fetch-geog` stages its sixteen datasets (a 2.75 GB download, 33.6 GB on disk)
into `~/.local/share/gpuwm/WPS_GEOG` once per machine, and `statics` builds
the truncation's cache from it (about 18 GB of host memory at T255) once
per truncation. `GPUWM_WPS_GEOG` points `statics` and `go` at an archive
already on the machine; `gpuwm fetch-geog` does not read it and stages a
copy of its own, so skip that line when an archive is already there. `go` builds a missing cache itself; `run` refuses without one and
names this command.

## 3. Size it against your card

There is no separate step: `gpuwm-global run` prices the card before it
allocates anything and refuses a plan that will not fit. The refusal is
an itemized device estimate and a verdict, the Legendre tables, the
spectral state, the grid fields and the step's working set weighed against
free VRAM less a 0.5 GiB other-process margin, and it names which
allocation dies first and the largest `[grid] truncation` that does fit.
Nothing is written and no device memory is taken when it refuses, so the
cheapest way to size a configuration is to start it.

`gpuwm-global run-plan PLAN.json --estimate` prints the same figures without
starting a run, as one JSON document, and a plan's `config.path` takes a
shipped experiment's bare name. The engine's own `gpuwm check` does NOT
price a global config on a published engine: measured on the Windows desktop
2026-09-10 against `gpuwm 2.7.0` and 2026-09-29 against `gpuwm 2.8.0`, it
refuses with `unknown table(s)` naming
this model's tables, because its sizing route knows the regional ones only.
`gpuwm-global configs --paths` prints the full path of every shipped
experiment, which is the file to copy and edit.

One number worth reading off either report: the Legendre tables are **cubic in
truncation and carry no vertical levels at all**. Trimming `[vertical]`
cannot make a tables-bound configuration fit. They are also built in
float64 on the *host* whatever the device precision is, so the host figure
does not halve with `precision = "float32"` -- but the build streams one
order band at a time, so at T533 that host peak is 0.21 GiB rather than
the 9.4 GiB the dense float64 squares once needed.

## 4. Run the forecast

```bash
gpuwm-global run arwen_global_gdas_t255_native_sl_si_24h --outdir out/t255-day
```

The shipped configuration reads the file step 1 wrote and decodes it through
the packaged authority mapping named by its bare id,
`analysis_mapping = "gdas-global"`. A bare id is asked of the engine's
authority table first and of this package's carried copies second, in both
the spellings those tables use, and must match exactly one document in
whichever answers, so it works from an installed wheel as well as from a
checkout. `gpuwm-global doctor` says which table answered each row. Adding a
source here is a mapping document, not code.

dt is 300 s on the semi-Lagrangian core, the shipped default, and the run is
86400 s, so it takes 288 whole steps and writes a checkpoint every 36 of them:
nine files, `out/t255-day/arwen_global_step00000000.npz` through
`...step00000288.npz` (1.46 GB each after the first, 12.3 GB for the day), plus `diagnostics.jsonl` and a
self-hashed `arwen-global-receipt.json`. The run prints the largest wind and
`total_water`, the global mean of every water store it books (the
atmosphere's share is about 27 kg/m²; the rest is soil water and the
500 kg/m² surface reservoir), as it goes. `gpuwm-global check-receipt`
validates the receipt, `gpuwm-global run --help` has the rest, and
`--restart` continues from a checkpoint.

## 5. Draw it

```bash
gpuwm-global render out/t255-day \
  --outdir out/pictures \
  --products 2m_temperature,mslp_10m_winds,500mb_height_winds
```

`render` exports one `wrfout_d01_<valid time>` NetCDF tape per checkpoint
on a regular lat/lon grid (360 x 720 by default), draws them, and removes
the tapes unless `--keep-tapes` is given. Every tape's valid time is the
run's forecast clock plus the checkpoint's model time, and the forecast
clock is the analysis frame's own valid time (the cycle you fetched), so
no date is typed. The same instant dates the physics: the sun the
radiation runs on is the analysis's sun.

Nothing about the drawing is global-specific. The tapes carry `MAP_PROJ = 6`
(cylindrical equidistant) and the production Rust renderer already draws
unrotated global frames, so the pictures land in the ordinary layout,
`out/pictures/<domain>/<product>/<valid-day>/`. Each map's title gives the
tape's spacing (`Δx 55.589 km` on the 0.5 degree tape, whatever the
truncation) and its source line names the model grid it was drawn from
(`T255 52.1 km grid, drawn at 0.5°`). `gpuwm render TAPE --list-products`
prints the catalogue for a tape and says which entries the tape can
actually fill; `--products all` draws every renderable one.

`gpuwm-global export` writes the tapes alone, and `run --tapes DIR` writes
them beside the forecast as each checkpoint lands, byte for byte the file
`export` writes from it. Beside the state, each tape carries what a reader
of WRF history files derives its own fields from: `F`, `E`, the two map
factors, the land-use planes and `W`. `docs/ARWEN_GLOBAL.md` ('What a tape
carries for a post-processor') says what each one is, and which part of the
vertical velocity `W` leaves out.

## All of it in one command

```bash
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

`go` runs steps 2 (the statics; the archive must already be staged), 4 and 5
and leaves the pictures in `out/day/pictures`. Its exit code is `5` when the
forecast stopped before its end, after the hours it reached were drawn.

## Moving to a current cycle

The shipped day names its analysis in two places, and both must name the
cycle you fetch:

1. `[initial] analysis_grib` (`cases/baseline-2026090100/gdas.t00z...`), the
   file `fetch-analysis --out` writes;
2. `start_time_utc` in the physics options, which the run checks against
   the analysis and refuses by name when the two disagree.

Copy the config (`gpuwm-global configs --paths` prints where it is), point
`analysis_grib` at the new file and delete `start_time_utc` so the clock
follows the analysis; `fetch-analysis` prints that `[initial]` block for any
file no shipped experiment reads. The forecast clock is read off the
analysis frame, so the solar geometry, the run receipt and every tape's
valid time follow the cycle you fetched.

GDAS publishes on the 00/06/12/18 UTC grid and lags its cycle by about seven
hours. An older cycle still fetches: anything past the operational window
goes straight to the AWS archive rung. That rung reaches back to
2021-03-22 12Z, the first cycle in the archive's `atmos/` layout; for an
earlier cycle the readiness probe answers "waiting" rather than refusing.
A start before that date, or at an hour between the four, comes from ERA5:
[ARWEN_GLOBAL_ERA5.md](ARWEN_GLOBAL_ERA5.md).

## Rehearsing the chain without a card

The three model doors are backend-agnostic.
`arwen_global_moist_smoke` is a T3 four-step numpy run.
It goes through the identical `run` -> `export` -> `render` sequence in
seconds and produces a real global frame, so a machine with no CUDA card can
prove the route before renting one:

```bash
gpuwm-global run arwen_global_moist_smoke --outdir out/smoke
gpuwm-global render out/smoke --outdir out/smoke-png \
  --products 2m_temperature --nlat 90 --nlon 180 \
  --start-date 2026-08-30_18:00:00
```

It is a rehearsal of the plumbing, not of the forecast: a four-step T3
atmosphere has nothing in it to look at, and an idealized run has no date of
its own, so `--start-date` names one.

## What is measured, and what is not

**Measured.** Every command on this page was run as printed from the
installed wheel in a clean container with an empty home, as a first account
and as a second account sharing the install (2026-11-22, on an RTX 4090 shared with other processes; the second account's step-288 checkpoint matched the first account's in all 135 state arrays). The `run` ->
`export` -> `render` chain is also exercised end to end at smoke truncation
on the CPU backend, against the real renderer binary, by
`tests/test_arwen_global_quickstart.py`, which also pins the shipped day's
own arithmetic: whole step count, output cadence landing on steps, the
analysis path the fetch writes, and the mapping resolving to exactly one
authority.

**Not measured here.** Forecast skill. `gpuwm-global score` scores a run
against stations and radiosondes; this page quotes no score.

## Where to go next

- [ARWEN_GLOBAL.md](ARWEN_GLOBAL.md) -- what ships, the resolution rungs,
  and the envelope: the reference suite's three-day limit, the native
  suite's admission evidence, and what may be said about effective
  resolution.
- [ARWEN_GLOBAL_FULL.md](ARWEN_GLOBAL_FULL.md) -- the model: dynamics,
  hybrid coordinate, reference physics, receipts, and the non-claims.
- [ARWEN_GLOBAL_LEVEL5.md](ARWEN_GLOBAL_LEVEL5.md) -- the native Arwen CUDA
  physics suite and its target-device admission battery.
- `arwen_global_t255_quickstart` -- the same grid and clock with the
  reference physics suite instead of the native one, on the GDAS
  2026-08-30 18Z analysis (`gpuwm-global fetch-analysis --cycle
  2026-08-30T18` writes the file it reads).
- `gpuwm-global assimilate` -- minutes-fresh point observations into a
  checkpoint, with an o-minus-b/o-minus-a gate of record.
- `gpuwm-global cycle` -- the forecast and the assimilation in one process:
  an analysis every interval on the resident state, each written as that
  hour's analysis checkpoint, then the forecast from the last one.
- `gpuwm-global --help` -- the rest of the research surface
  (pins, checkpoint and export inspection, Level-4 migration, the one-way
  regional parent bridge, the native qualification battery).
