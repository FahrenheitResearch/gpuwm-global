gpuwm-global 0.2.0

Arwen Global 0.2.0 adds four **experimental** pieces for runs longer than
a forecast: a long-run mode for runs of months, two switches for the
sunlight the model reflects too much of, a coupled mixed-layer ocean with
sea ice, and other worlds. Each is off by default and runs only when a
config names it. Each is labelled experimental because none has yet been
measured over the length of run it is for, and each has known faults,
listed below. A shipped forecast that names none of them keeps its config
hash, and its atmosphere is byte-identical to 0.1.9's. One checkpoint
array differs: the open-water surface reservoir, which nothing in the
physics reads (see Measured). A one-card run that runs out of card memory
now carries on on a lighter plan. It needs `gpuwm>=2.9.13,<2.10`, as
0.1.9 did, and the Rust doors are 0.1.9's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

Experimental, each off by default:

- **Long runs.** A run of months needs a lower boundary for months, so the
  sea-surface temperature and the sea ice follow a rule the config names:
  `[ocean] mode = "anomaly"` (the starting anomaly carried on a 1991-2020
  OISST climatology and damped), `"climatology"`, `"series"` or `"hold"`.
  With it come the vegetation's monthly march (`[land] greenness`),
  checkpoint retention, monthly and daily mean files and a daily summary
  (`[output] keep_checkpoints`, `mean_state`, `daily_summary`),
  `[radiation] ghg_year = "valid"` for a run that crosses a new year,
  orographic gravity-wave drag (`orographic_drag = "kim-gwdo"` in
  `native_adapter_options`, with `gpuwm-global statics CONFIG
  --orographic`), Noah's snow age carried between land calls
  (`snow_ageing = true`), and gates LR-0 to LR-10 read by
  `tools/arwen_global_longrun_report.py`. A long-run config sets, in
  `[physics] native_adapter_options`, `snow_ageing = true,
  morrison_droplet_number = "land_ocean", radiation_liquid_radius =
  "wrf_relcalc"`. To set one up: `gpuwm-global climatology fetch`,
  `gpuwm-global climatology build CONFIG`, `gpuwm-global surface-series
  CONFIG --source gdas-analysis --cycle YYYYMMDDHH --anomaly-reference
  --out reference.npz`, then the segments as `gpuwm-global run CONFIG
  --restart ... --until-s ...`, or `tools/arwen_global_longrun_chain.py
  start RUN_DIR --config CONFIG --lease-days 5` from the source
  distribution, which runs them as a chain of leases you can resume.
  `docs/ARWEN_GLOBAL_LONG_RUNS.md` covers all of it.
- **The albedo switches.** Two native options:
  `morrison_droplet_number = "land_ocean"` (300 cloud droplets per cm3
  over land and 100 over ocean, MPAS's values, in place of WRF's single
  250) and `radiation_liquid_radius = "wrf_relcalc"` (the liquid radius
  WRF's own RRTMG uses for Morrison cloud). Add both to `[physics]
  native_adapter_options`. A third, `radiation_cloud_inhomogeneity =
  "tiedtke1996"`, ships off and is not recommended: it cost 2 m
  temperature skill. `docs/albedo-fixes.md`.
- **A coupled ocean.** `[ocean] mode = "slab"`: a mixed layer whose depth
  follows the WOA18 climatology, with a heat-flux correction (Q-flux),
  restoring toward the climatology, and a sea-ice mass budget (`[sea_ice]
  mode = "thermodynamic"`). It is advanced before every step from the
  fluxes the physics held. To switch it on: stage and build the
  climatology with the mixed-layer depth (`gpuwm-global climatology fetch
  --with-mld`, then `gpuwm-global climatology build CONFIG --with-mld
  --out clim.npz`); run a calibration with `[ocean] mode =
  "climatology"`, `climatology` naming that file, `anomaly_reference =
  "start"` (or the seven-day reference series the long-run page builds)
  and `qflux_calibration = true`, which banks the fluxes only after
  `qflux_calibration_spinup_days` (2 by default, so run it longer or set
  it lower); make the Q-flux file from its last checkpoint
  (`gpuwm-global climatology qflux CHECKPOINT --out qflux.npz`); and run
  the slab with `[ocean] mode = "slab"`, the same `climatology`,
  `slab_qflux` naming the Q-flux file and `[sea_ice] mode =
  "thermodynamic"`, under the same physics options as the calibration.
  Write both paths in full: a relative path is read from the config's
  own folder. Tapes of a
  slab run carry `SST`, `HML` and `ARWEN_QFLUX`.
  `docs/ARWEN_GLOBAL_OCEAN.md`.
- **Other worlds.** A `[world]` table at the end of a config lowers or
  removes the mountains, tilts the axis, changes the sun, the day or the
  orbit, raises the sea, sets CO2, methane and nitrous oxide, or reads an
  invented continent from a map. `gpuwm-global world list` shows the
  fourteen presets, `gpuwm-global world new` writes a world config from an
  Earth one, and `gpuwm-global world twin` runs it beside its Earth
  control and a perturbed control. Two worlds ship as experiments:
  `arwen_global_world_t255_no_mountains` and
  `arwen_global_world_t255_on_its_side` (fifty-nine experiments shipped with 0.2.0). Each
  world is labelled science, illustration or not supported yet, and is
  named on its receipt, checkpoints and tapes.
  `docs/ARWEN_GLOBAL_WORLDS.md`.

On by default:

- **A run that runs out of card memory carries on.** When a one-card run
  raises an out-of-memory error mid-run, it gives its memory back, reads
  the card again and carries on from its newest checkpoint, in the same
  process, on the next plan the door prices lower (more latitude bands or
  the host tier). The receipt carries a `recovery` row. When no lighter
  plan fits, the run stops with one line (exit 4) that names the cause,
  the free memory, the card's other processes and the `--restart ...
  --latitude-bands ... --host-spill ...` to resume with. Under a pool
  limit set with `CUPY_GPU_MEMORY_LIMIT`, the lighter plans are weighed
  against that limit too, and a run that no plan fits under stops at once
  and says to raise or unset it. The door also leaves 512 MiB for other
  processes it sees on the card. Neither changes a forecast bit.
- **Open water is not a finite store.** On ocean, lake and sea-ice columns
  the water the physics moves goes to a signed account,
  `ocean_exchange_kg_m2`, and the surface reservoir there keeps its value.
  Before, a run of months was refused once an ocean column's reservoir
  ran out. No native scheme reads the reservoir, so the atmosphere is the
  same. A watch refuses a run whose open-water column evaporates more than
  60 kg/m2 in a day. `open_water_exchange = false` in `[physics]
  native_adapter_options` keeps the earlier bookkeeping.
- **Three fixes** from the long-run work: the vegetation march reads the
  model day's start, so a run restarted inside a day takes the straight
  run's vegetation; a column that changes class at an ice record leaves
  the frozen-surface step; and a numpy series run no longer writes the
  next step's skin into a checkpoint.
- A dated run longer than 384 h with no `[ocean]` table is refused with a
  message that names the modes. It used to hold the SST without saying
  so.

`CHANGELOG.md` has each change with its own measurements.

## Measured

**Forecasts unchanged with every new option off.** On the RTX 5090s of one
host, the shipped `arwen_global_gdas_t255_native_sl_si_24h` and
`arwen_global_gdas_t383_native_sl_si_24h` were run to 3 h by this release
and by 0.1.7 (whose arrays 0.1.8 and 0.1.9 write unchanged), 2026-11-13,
and compared with `tools/arwen_global_checkpoint_identity.py`:

- With no option named, the config hash is 0.1.7's at both truncations,
  and the step-0 checkpoint's 43 arrays are byte-identical. At step 36,
  132 of the 133 arrays are byte-identical. The one that differs is
  `surface_water_kg_m2`, the open-water surface reservoir (largest
  difference 52.9 kg/m2 at T255 and 85.0 at T383). It differs because on
  open-water columns the exchange account now takes the water the physics
  moves, and the reservoir there keeps its value. No native scheme reads
  that reservoir, so the forecast is the same: winds, temperatures,
  moisture, cloud, precipitation, the surface and the soil are
  byte-identical. The checkpoint also carries the account's two arrays,
  `ocean_exchange_kg_m2` and `ocean_evaporation_day_kg_m2`.
  `tests/test_arwen_global_open_water_exchange_cuda.py` holds the same on
  a card: one physics half-step of the native suite with the account and
  without it, and every prognostic, tracer, surface plane and physics
  array but the reservoir and the account's two is byte-identical.
- With `open_water_exchange = false`, the step-0 and step-36 checkpoints
  are byte-identical to 0.1.7's at both truncations, 43 and 133 arrays,
  archive hash included.
- 0.1.7's step-0 checkpoint of the T255 day, continued by 0.2.0 with
  `open_water_exchange = false`, ends at step 36 byte-identical to
  0.1.7's straight run, all 133 arrays. With no option named it ends as
  above: only the reservoir differs.
- On one RTX 5070 Ti 16 GB, the shipped T383 config to 3 h with
  `open_water_exchange = false`, this release's tree and 0.1.7 run in
  turn on the same card (four runs and three), 2026-11-13: steps 0 and
  36 byte-identical to 0.1.7's in every run, archive hash included, on
  0.1.7's plan (one band, nothing parked), at 1.817 to 1.821 s a step
  against 1.818 to 1.829 s, the card at 13,714 to 14,254 MiB at its
  peak against 13,920 to 14,212 MiB.
- Every shipped config hashes as in 0.1.8 and 0.1.9.

**The albedo switches.**

- Over three 48 h starts in November, July and January, the pair cut the
  sunlight reflected in excess of CERES-EBAF by 57 to 59 %, moved the
  outgoing longwave by under 0.5 W/m2 and left the 500 hPa height better
  in all six case-leads.
- Over ten days at T255 from GDAS 2026-11-10 18Z, in anomaly mode on an
  RTX 4090, beside a control: the planetary albedo 0.343 against 0.392
  (CERES-EBAF's November mean is 0.303), the net at the top of the
  atmosphere -4.85 against -21.33 W/m2, the 2 m temperature within
  0.03 K of the control on every day, every receipt gate passed.
- Three 90-day runs of the atmosphere at T255 from GDAS 2026-08-01 00Z, in
  5-day leases on RTX 5090s, were **in progress** when this release was
  cut (2026-11-13 05:39 UTC): the pair with the GDAS SST series to day
  60, the pair in anomaly mode to day 45, and a control in anomaly mode
  to day 55. Lease by lease, the planetary albedo read 0.316 to 0.331
  with the pair and 0.360 to 0.380 in the control, and the net at the
  top of the atmosphere -15.6 to -6.3 W/m2 against -29.2 to -22.1. The
  control's albedo rose from 0.360 to 0.380 over its 55 days; the pair's
  stayed between 0.316 and 0.331. Every lease passed all twelve receipt
  gates. These are partial runs, not a 90-day result.

**The coupled ocean.** On one RTX 5090, ten days of the shipped T255
native semi-Lagrangian config without the albedo pair, from GDAS
2026-11-10 18Z, after a ten-day calibration from the same start (so the
Q-flux is in sample), beside a twin with the SST prescribed:

- Against the climatology, the open-ocean SST moved -0.018 K in the area
  mean and 0.224 K rms; against the analysis, -0.024 K in the mean. The
  largest change on one column was 5.46 K.
- Without the Q-flux, the open mixed layers lost about 19 W/m2 to the
  atmosphere, and the SST moved -0.067 K in the mean and 0.46 K rms.
- The energy books agreed to 2.8e-3 J/m2 on every ocean column after ten
  days, and no number was NaN.
- The northern ice area grew 0.93 million km2 against the climatology's
  0.76. The southern grew 0.23 while the climatology's shrank by 1.04.
- Restarted from +6, +12 and +24 h, the run ends identical to the straight
  run, archive hash included. A step costs 1.1 % more than without the
  ocean, and the card peak is the same.
- On this release's tree: one coupled T255 day with the albedo pair, snow
  ageing and `--tapes`, restarted at +12 h, ended with its +18 h and
  +24 h checkpoints identical to the straight day's (143 of 143 arrays)
  and both later tapes byte-identical.
- Three coupled runs at T255 from GDAS 2026-08-01 00Z, in 5-day leases,
  were **in progress** at the cut (2026-11-13 05:39 UTC): the pair to day
  20 of 90, a control without the pair to day 10 of 90, and the pair with
  365-day restoring to day 10 of 180. With the pair, lease by lease, the
  planetary albedo read 0.314 to 0.324 and the net at the top of the
  atmosphere -12.4 to -15.1 W/m2, against 0.358 to 0.366 and -27.4 to
  -28.1 without it. In the 90-day run with the pair, from day 5 to day
  20, the mean SST went from 293.03 to 292.94 K, the northern ice area
  from 3.7 to 2.9 million km2 and the southern from 13.8 to 14.6, and the
  ocean's energy books agreed to within 4.3e-3 J/m2. Every lease passed
  all twelve receipt gates. These are partial runs, not a 90-day result.
- The 90-day runs of the atmosphere and of the coupled ocean run on the
  development line this release merges, where snow ageing was on by
  default; a config that repeats them sets `snow_ageing = true`.

**Other worlds.** On one RTX 5090 at T127 (the T255 forecast-day config
cut to truncation 127): no mountains for 10 days, on its side for 5, the
waterworld, a 250-million-year-old continent and a 12-hour day for 20
days each, the sea 70 m higher for 10, and one step each at 190 to 1,120
ppm of CO2. Every run passed every gate, and the global-mean surface
pressure moved by at most 1.2e-6 of itself over a run. The world with no
mountains stayed distinguishable from a perturbed Earth at 500 hPa until
about hour 180. The two shipped worlds ran their T255 day in 368 and
364 s, against 376 s for the plain day. On this release's tree the world
with no mountains ran its T127 day and passed in 83 s.

**The long-run day on this release's tree.** One T255 day of a long-run
config (anomaly mode, the orographic drag, the albedo pair, snow ageing,
monthly means and retention) passed every gate in 261 s on one RTX
5090.

**Running out of memory.** Measured on 0.1.7 with this change alone, at
T383 on one RTX 5070 Ti 16 GB, 2026-11-13: with another process taking
2 GiB of the card after the start, three segments each ran out of memory,
carried on from their newest checkpoint on two bands with the host tier
on, and finished; with five other processes' CUDA contexts opening after
the start, which on 0.1.7 killed one segment in eleven, nine segments
finished (one after a fallback); with another process taking every byte
the card freed, the run stopped with the one-line message and exit 4.
All 74 checkpoints those runs wrote are byte-identical files to the
reference run's at the same step. With the error forced in the radiation,
in a checkpoint's copy off the card and twice in a row, and under a
`CUPY_GPU_MEMORY_LIMIT` of 12 GiB, each run carried on and finished, and
the 15 checkpoints they wrote are byte-identical to the reference's.
Under a 6 GiB limit, below every plan, the run first rebuilt on three
lighter plans in 81 s that the limit refused, and told the user to free
a card nothing else held; with the fix this release carries it stops
(exit 4) after 23 s and names the limit. On this release's tree, on the
same card, the shipped T383 config to 3 h: under a 6 GiB limit it
stopped (exit 4) after 16 s with that line; under a 12 GiB limit it
carried on from step 0 on two bands with the host tier, and with
`open_water_exchange = false` its step-36 checkpoint is byte-identical to
0.1.7's straight run, archive hash included.
`tests/test_arwen_global_oom_fallback.py` runs on this release's tree.

## Upgrading

- `pip install --upgrade "gpuwm-global[gpu-cu13]"`, then `gpuwm-global
  fetch-doors` and `gpuwm-global doctor`, as before. The door bundle is
  the one 0.1.3 published, and a bundle already staged stays valid.
- **No shipped config hash moves.** A checkpoint written by 0.1.7, 0.1.8
  or 0.1.9 resumes under 0.2.0 with the same config. It has no exchange
  account, so the run gives it a zero account, and the receipt's
  `open_water_exchange.filled` says so.
- **A native checkpoint now carries** `ocean_exchange_kg_m2` and
  `ocean_evaporation_day_kg_m2`, and `noah_snotime` when `snow_ageing =
  true`. A run with `open_water_exchange = false` refuses a checkpoint
  that carries an account. To compare a 0.2.0 forecast with an earlier
  one array for array, set `open_water_exchange = false` in `[physics]
  native_adapter_options`.
- **The experimental options join the config hash when they are on**
  (`snow_ageing = true`, the albedo pair, `orographic_drag`, the long-run
  ocean rules by the SHA-256 of their climatology and reference files,
  the slab by its Q-flux file's). Retention, means and the daily summary
  never join it. A checkpoint written with one of them on is refused by a
  config without it, and the other way round. A world has its own
  identity, and an all-Earth `[world]` table changes nothing.
- **Snow ageing is off by default.** A long-run config sets `snow_ageing
  = true`. A config written for the long-run work before this release,
  when the option was on by default, has to name it to reproduce those
  runs.
- **A Q-flux file belongs to its physics.** A slab run with the albedo
  pair refuses a file made without it, and a slab run refuses a restart
  under another ocean.

## Known limits

- **None of the four has been measured over its full length.** The
  longest finished runs with the long-run mode, the pair or the ocean are
  ten days, and the longest world runs are twenty days at T127. The 90-day
  runs above were still running at the cut.
- **The model still reflects too much sunlight with the pair on**: about
  14 W/m2 more than CERES-EBAF over the ten-day run, all of it from ocean
  cloud between 60 S and 30 N. The net at the top of the atmosphere is
  still negative, in the ten-day run and in every lease of the 90-day
  runs so far.
- **The slab is a mixed layer with a heat-flux correction, not a
  circulating ocean.** It has no currents, no upwelling and no
  entrainment, and its depth is a climatology. The Q-flux puts back the
  climatological effect of all of them, but not their anomalies, so an
  El Nino is damped rather than grown or ended. A Q-flux is only as good
  as the length, season and physics of the calibration it came from; the
  coupled 90-day runs' Q-flux rests on about five calibrated days a
  month.
- **The sea-ice area does not follow the climatology's.** In the ten-day
  coupled run the southern pack grew where the climatology's shrank, and
  the ocean page's gate OC-4 fails for the southern ice. The ice carries
  no snow, does not move and does not ridge.
- **Gates still open on long runs.** LR-2 (the open-water water budget)
  reads "not measured", LR-4 flags an energy residual of about 5 W/m2 in
  every run, paired, control and coupled alike, and LR-10 has no
  threshold. LR-4 asks for 1 W/m2 or less before a slab ocean, so the
  coupled runs carry that imbalance.
- **A world refuses** cycling, the assimilation door, the regional parent
  exports, the scorecards, and the orographic drag when the world changes
  the terrain or the coast. Most worlds have run only at T127.
- **The open-water watch's 60 kg/m2 a day** is an engineering value, not a
  measured one.
- **The open-water reservoir in a shipped forecast's checkpoint** is not
  the earlier releases' (see Measured). Nothing reads it; a tool that
  compares it between releases sees it move. `open_water_exchange =
  false` keeps the earlier bookkeeping.
- **The out-of-memory fallback** covers a one-card run on the card. It
  does not cover a spread run or a run whose lightest plan does not fit,
  and the steps since the newest checkpoint are run again.

## Unchanged

- The engine range, `gpuwm>=2.9.13,<2.10`. Two more engine files are
  pinned, 50 in all, and `tools/measure_boundary.py` reads 242 symbols
  across 72 modules with no gap.
- The ten Rust doors. `gpuwm-global fetch-doors` stages the bundle 0.1.3
  published.
- Every shipped config's hash, and every byte of a shipped forecast's
  atmosphere.
