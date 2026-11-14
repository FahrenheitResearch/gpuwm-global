# Changelog

## 0.2.3

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.2.2 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 242
symbols across 72 modules with no gap, and all 50 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.2.0.  Two shipped configs change on purpose: the T255 and T383
forecast days (`arwen_global_gdas_t255_native_sl_si_24h`,
`arwen_global_gdas_t383_native_sl_si_24h`) turn the kim-gwdo orographic drag
on, and the T383 day calls radiation every 3000 s instead of 2100 s.  Every
other config 0.2.2 shipped keeps its 0.2.2 hash, and one config is new.
Everything else here is off by default: opt-in switches for long and coupled
runs, and two experimental pieces (a dynamic ocean, storm switches on New
Tiedtke).  With `orographic_drag = "none"` the T255 day writes 0.2.2's
checkpoints byte for byte (Measured, below).

### Changed (defaults)

- The kim-gwdo orographic gravity-wave drag (WRF `gwd_opt = 1`,
  `orographic_drag = "kim-gwdo"`) is on in the T255 and T383 forecast days.
  Over the forecast scorecard's 16 starts (2026-09-14 to 11-08, 120 h, T255,
  0.2.1 tree on RTX 5090s, GFS analysis as truth), against the same day
  without it: NH Z500 RMSE at 120 h 52.0 against 54.3 m (better on 14 of 16
  starts), NH MSLP 5.11 against 5.33 hPa (16 of 16), NH 250 hPa wind 0.40
  m/s lower (14 of 16), NH T850 0.08 K lower (15 of 16), about 3.6 h of NH
  Z500 lead at 96-120 h.  The southern hemisphere is neutral; SH MSLP is
  0.08 hPa worse at 120 h (5 of 16 better).  It was measured at T255 only:
  its skill at T383 and over a season is not measured.  `orographic_drag =
  "none"` in a copy of the T255 day gives 0.2.2's arithmetic and hash.  The
  T383 day also changed its radiation interval (below), so a copy of it
  needs `orographic_drag = "none"` and `radiation_interval_s = 2100.0` to
  get 0.2.2's hash back (the drag off alone gives neither release's); its
  checkpoints were not compared with 0.2.2's.
- The drag reads a second statics cache (VAR, CON, OA1-4 and OL1-4 from
  WPS_GEOG `orogwd_10m`, masked with the land cache's land mask), which a
  machine set up under 0.2.2 does not have.  It is now built for you:
  `gpuwm-global statics CONFIG` builds it after the land cache, and `run`
  (so `go` and `run-plan` too) builds it before it prices the card when it
  is missing, from the WPS_GEOG tree the land cache records.  When that
  tree has no `orogwd_10m`, the run stages the one dataset with the
  engine's own `gpuwm fetch-geog --datasets orogwd_10m --source ncar` (3.3
  MB, SHA-256 pinned by the engine).  One process builds at a time behind a
  lock file, so the ranks of a multi-card run do not race.  What cannot be
  built is one sentence, before the first step, naming `gpuwm fetch-geog
  --datasets orogwd_10m --source ncar`, `gpuwm-global statics CONFIG
  --orographic` and `orographic_drag = "none"`.  Neither host this release was checked on had `orogwd_10m` in its WPS_GEOG
  tree; from an empty tree the run door staged it and built the cache in
  2.5 s at T255 and 3.9 s at T383 (fetch included), then ran (Measured).
- The T383 day calls radiation every 3000 s (2100 s before).  Over eight
  120 h T383 starts (2026-09-29 to 11-08) scored against the GFS analysis,
  no metric was worse by more than three times the round-off twin spread
  (0 of 65 cells; one cell better, T850 NH at 96 h), and the run stepped
  1.067x faster on an RTX 5090 (1.056 against 1.126 s a step, the paired
  median over the eight starts) and 1.17x in one 24 h run on an RTX 5070 Ti
  (1.70 against 1.99 s a step).  The other T383 configs keep 2100 s.

### Added (opt-in)

- `arwen_global_gdas_t255_native_sl_si_24h_dayone`: the T255 forecast day
  with New Tiedtke (`cumulus = "ntiedtke"`) in place of Grell-Freitas, and
  the drag, which the day itself now carries.  Measured on all 16 of the
  scorecard's starts, 120 h runs on RTX 5090s, GFS analysis as truth,
  against 0.2.2 as shipped.  "k/16" counts the starts where the set is
  better; the scorecard's bootstrap 95% interval excludes zero for every
  difference quoted here unless it says otherwise:
  - NH Z500 RMSE 10.30 to 8.60 m at 24 h (16/16) and 54.35 to 50.46 m at
    120 h (15/16). NH Z500 ACC at 120 h goes from 0.835 to 0.852 (14/16).
    kim-gwdo alone gives 9.45 m and 0.845. Read off the control's ACC curve
    on the 14 starts verified to 240 h, the set is worth 7.3 h of NH Z500
    lead at 120 h (kim-gwdo alone 3.7 h): ACC 0.8 at about 137 h against
    the control's 130 h, GFS's 150 h and IFS's 166 h.
  - T850 RMSE is lower in the NH and the tropics at 24, 72 and 120 h
    (14-16/16). 250 hPa wind is lower in the NH and the tropics (14-16/16).
    NH MSLP RMSE goes from 5.33 to 4.89 hPa at 120 h (13/16).
  - SH Z500 RMSE is better through 72 h (14-15/16). At 120 h it is 61.62 to
    59.84 m on 10 of 16 starts, which is not significant.
  - Cost: tropical Z500. RMSE goes from 5.02 to 5.68 m at 24 h (0/16) and
    from 11.24 to 14.20 m at 120 h (1/16). Most of it is a mean offset: the
    tropical Z500 bias goes from +0.07 to -8.83 m at 120 h. Tropical Z500
    ACC at 120 h drops from 0.664 to 0.644 (7/16, not significant). On 6
    starts New Tiedtke alone gives -7.96 m, so New Tiedtke causes it. On
    those 6 the tropical MSLP bias is no lower (-0.67 against -0.77 hPa),
    so the offset is in the 1000-500 hPa thickness. It is not traced
    further, so the config stays opt-in.
  - New Tiedtke alone, on 6 starts (09-14, 09-18, 09-21 12Z, 09-25, 10-21,
    10-28): NH Z500 RMSE 8.67 m at 24 h and 44.93 m at 120 h, against 7.97
    and 43.64 m for the set on the same starts.
  - A 0.2.2 control on 2026-09-14, rerun on the same card type as the arms,
    reproduces the scorecard's control to 0.02 m at 24 h (NH Z500 RMSE
    8.008 against 8.026 m) and 0.41 m at 120 h. The fields are not bit for
    bit the same.
- `[repair] mass_fixer_target = "dry"`, `energy_fixer = true` and
  `vapour_fixer = true`: the dry mass, the dry total energy and the vapour
  column held across the dynamics for long runs
  (`docs/ARWEN_GLOBAL_LONG_RUNS.md`, "The conservation switches a long run
  sets").  All three are off by default and join the config hash only
  when set; a run that names none of them writes 0.2.1's checkpoints,
  receipts and diagnostics byte for byte (the shipped smoke and a T21
  semi-Lagrangian variant over 1,500 m terrain, CPU; and 36 T127 steps of
  the full native suite on an RTX 5090).  Sixty T127 days
  with them: LR-4's residual 0.24 / 0.43 W/m2 over days 0-30 / 30-60
  (4.06 / 4.50 without), the dry-mass range 0.26 Pa (35.1 without); ten
  T255 days: -0.29 W/m2 (5.44 without) and 0.21 Pa (10.2 without).  The
  energy the dynamics lost is 28-40% of the troposphere's cold drift
  against GDAS over 60 days, not all of it.  Under the dry
  target the receipt's mass gate
  defaults to 1e-5 and a tighter one is refused (the dry row carries each
  step's half-step E - P at an output instant), and a branch from a run
  that held the other quantity opens a new mass epoch.  Two five-day T255
  forecasts with them scored neutral against GDAS (NH Z500 RMSE at 120 h
  52.70 against 53.10 m and 43.94 against 44.33 m); the 850 hPa warm bias
  at 120 h rose from +0.16 to +0.31 K and from +0.37 to +0.55 K.  They stay off for forecasts.
- The long-run recipe (`docs/ARWEN_GLOBAL_LONG_RUNS.md`, "The native
  options a long run sets") now sets the three Grell-Freitas shallow-arm
  switches of 0.2.2 (`gf_shallow_buoyancy_from_cloud_base`,
  `gf_shallow_blqe_to_cloud_base`, `gf_shallow_depth_min_m = 500`) and the
  three conservation switches above, for long and coupled runs; the
  package ships no long-run config, so no shipped hash moves.  In the
  90-day atmosphere-only arm from GDAS 2026-08-01 (anomaly SST, kim-gwdo,
  the albedo pair) the shallow switches took the excess of reflected
  sunlight over CERES from +15.0 / +15.9 / +14.2 to +3.5 / +5.4 / +4.5
  W/m2 in August / September / October, and in the coupled slab arm the
  open-water SST against GDAS at day 59 read -0.07 K against -0.30 K, on a
  Q-flux calibrated without them.  They stay off for forecasts: over the
  16-start scorecard they cost tropical T850 +0.04 to +0.08 K and 250 hPa
  wind +0.22 to +0.24 m/s at 72-120 h, and in the southern summer they vent
  the Peru deck (-28.8 against +8.5 W/m2 of excess in a 30-day December
  twin).  The two sets were measured apart; a long run with both on has
  not been run, and neither has a Q-flux calibration with the shallow
  switches on.  `docs/trade-shallow.md` has the 90-day arm, the December
  twin, the scorecard's tropical cost and the coupled arm.
- Ocean: an off-by-default mixed-layer cooling response for the prescribed
  sea surfaces (`[ocean] mixed_layer_response = true` under `mode =
  "hold"`, `"anomaly"`, `"climatology"` or `"series"`;
  docs/ARWEN_GLOBAL_OCEAN_RESPONSE.md).  A held sea cannot cool under days
  of dry offshore wind, and in the T383 50-day runs its evaporation reached
  51-55 kg m-2 a day off Tampico, in the northern Red Sea and in the Gulf of
  Tehuantepec, close to the 60 kg m-2 ceiling that stops a run.  The
  response carries one departure D from the prescribed field on open ocean,
  driven by the net surface heat flux less the column's own running-mean
  flux, into a fixed 40 m layer, relaxed back over 10 days and held within
  5 K.  With the key absent or false no config hash, identity, receipt,
  checkpoint array or metadata moves; with it on, two float64 planes ride in
  the checkpoint and a restart is bit for bit.
  `--branch-from-other-config` may switch it on at a checkpoint written
  without it, under the same rule; a branch onto the response over another
  rule is refused, as a branch onto the slab is.
  `tests/test_arwen_global_ocean_response.py`.  Measured on
  the T383 50-day path at the defaults: the Tampico norther days 55.05,
  52.10, 49.74 kg m-2 become 52.83, 46.71, 42.61, and the Tehuantepec gap
  jet day 50.54 becomes 48.14; global open-water departure mean within
  0.02 K of zero, its 1st to 99th percentiles within 0.5 K; no measurable
  cost in memory or step time on the 16 GB card.  It lowers the margin to
  the ceiling, it does not remove the risk.
- Noah: `snow_age_refresh = "amount"` (native option, needs `snow_ageing =
  true`, off by default and in the config hash only when set).  WRF's rule
  (`"any"`, the default) makes a snowpack fresh on any call that snows, and
  on this model a trace of snow reaches most snow-covered land on most land
  calls: in the 365-day T127 run 79% of the snow columns were under 0.01
  day old on 2027-02-28, and NH land absorbed 10-20 W/m2 less sunlight than
  ERA5 from February to April.  `"amount"` keeps 1 - min(dsnow / 1 kg m-2,
  1) of the age (Noah-MP's refresh) and a column with no snow carries no
  age.  In two 60-day T127 winter twins it lowered the NH land (20-70N)
  albedo by 0.024 / 0.025 (late-winter twin, February / March) and 0.027 /
  0.043 (second twin, January / February), about 30% of the late-winter
  twin's excess over ERA5, and warmed NH land 2 m temperature in February by +0.58 K and
  +1.72 K; two twins agree in sign, they are not an ensemble.  One 120 h
  T255 forecast was neutral.  With the default rule a T127 day is
  byte-identical to 0.2.1 (143 of 143 arrays).  Divergence rows `NOAH-CU-3`,
  `NOAH-PY-5` and `NPREF-18`; `docs/ARWEN_GLOBAL_LEVEL5.md`, "Snow age".

### Added (experimental)

- The dynamic ocean, `[ocean] mode = "dynamic"` (off by default), and the
  `ocean` command (`build-grid`, `build-init`, `init-from-state`, `run`)
  that builds its grid from ETOPO 2022 and its start from NOAA GODAS and
  WOA23: a 360 x 209 x 50 ocean from 78S to 65N with KPP mixing, coupled
  every hour, with the slab and its thermodynamic ice north of 65N and the
  slab's ice above it.  Off, it changes nothing (55 config
  hashes and a T127 6 h run's 317 arrays as v0.2.1).  A coupled restart is
  byte-identical to the straight run (177 of 177 arrays).  A 30-day coupled
  T255 run passed every gate.  It fit the 16 GB RTX 5070 Ti at T383 (card
  peak 14,990 of 16,303 MiB) at +13.7% a step (+11.6% at T255), measured on
  the T383 day as 0.2.1 shipped it (radiation every 2100 s, no drag); it was
  not re-measured on 0.2.3's T383 day.  Open, and
  why it is experimental: the equatorial undercurrent at 140W falls from
  0.73 to 0.26 m/s in 30 days against GODAS's 0.76 to 0.63; the Kelvin
  (+6.6%) and Rossby (-10.9%) wave speeds and wind-driven mixed-layer
  deepening (27% too shallow) fail their gates; the remap's effect on the
  SST, ice and stress the atmosphere sees is not measured.  A run is
  refused when `dt_s` does not divide 3600 s or it asks for more than one
  card.  `docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md`.
- New Tiedtke: five storm-intensity switches, off by default, in the
  config hash only when set away from their defaults under `cumulus =
  "ntiedtke"`: `ntiedtke_momentum_transport` (WRF's `lmfdudv`),
  `ntiedtke_adjustment_time_factor`, `ntiedtke_entrainment_factor`,
  `ntiedtke_trigger_entrainment_factor` and `ntiedtke_downdraft_factor`.
  There is no recommended value.  New Tiedtke barely dilutes the plume in a
  moist cyclone core (Katrina at T383: rain within 150 km of the centre a
  third of ERA5's from hour 18, 95% of it from the scheme).  Twice the
  organised entrainment deepened Katrina from 977 to 952 hPa (ERA5 933),
  but put 9 cells over 500 mm in a global 72 h twin (0 without; GFS 1) and
  moved Sandy's landfall-time track error from 352 to 744 km; three times
  gave 75 such cells.  At the defaults a 2 h T383 checkpoint is
  byte-identical to v0.2.1's.  Divergence rows `NTIEDTKE-PY-5` and
  `NTIEDTKE-CU-2`.

### Fixed

- The sizer matched its completed-plan, measured-run and floor rows on the
  truncation alone, so a T383 run on a linear 384x768 grid was priced at
  the 14.27 GiB of card a quadratic T383 run took (its own allocator read
  6.75 GiB, RTX 5090).  Rows now match on (truncation, nlat, nlon), and a
  row without a shape is the truncation's quadratic grid, so every shipped
  config prices as before.  The upper-air and observation scorecards
  cached the rebuilt surface geopotential under the analysis and nlat x
  nlon only, so a T255 run and a 384x768 T383 run read each other's
  terrain; the cache name now carries the truncation
  (`tests/test_arwen_global_grid_shape_keys.py`).

### Changed

- The carried-physics divergence ledger has 947 rows (912 at 0.2.2): the
  storm switches' `NTIEDTKE-PY-5` and `NTIEDTKE-CU-2`, the snow-age rows,
  and every row of the five carried files 0.2.3 changed re-read, all
  measured against the 2.8.5 baseline the rows are keyed to.  The 2.9.13
  re-baseline is still owed, and the two comparison nodes of
  `tests/test_engine_divergence.py` skip as before.

### Measured

- On an RTX 4090 (24 GB, shared with other processes), one card lease,
  2026-11-14, GDAS 2026-09-01 00Z, the shipped T255 day for 24 h, each tree
  over gpuwm 2.9.13; checkpoints compared with
  `tools/arwen_global_checkpoint_identity.py`:
  - 0.2.3 with `orographic_drag = "none"` (config hash c7a000d28c206ace,
    0.2.2's) against 0.2.2: all nine checkpoints, steps 0 to 288,
    byte-identical (44 and 135 arrays, `self_sha256` equal);
  - 0.2.3 as shipped (hash 7b140980a546263e), from a copy whose only
    change was `[statics] geog_root` pointing at an empty folder (paths are
    outside the identity): the door staged `orogwd_10m` with `gpuwm
    fetch-geog` and built the T255 drag cache (97,190 land columns) in 2.5 s,
    then ran; against 0.2.2 its step 0 is equal in all 44 arrays and steps
    144 and 288 differ in 119 of 135; allocator live peak 6.714 GiB against
    6.703 GiB without the drag;
  - a restart of it from its step 144 wrote steps 180 to 288 byte-identical
    to its straight run (135 arrays);
  - all four receipts pass.
- On the 16 GB RTX 5070 Ti, alone, the shipped T383 day (drag on,
  radiation every 3000 s) for 24 h from the same analysis, from an empty
  WPS_GEOG tree as above (the door staged `orogwd_10m` and built the T383
  cache, 218,598 land columns, in 3.9 s), against the same day with
  `orographic_drag = "none"`, two runs each, the second with the CuPy
  kernel cache warm (the first compiled every kernel):
  - warm: 1.722 s a step with the drag against 1.701 without (+1.2 %);
  - card peak (process, sampled every 250 ms): 14,368 and 14,434 MiB with
    the drag, 14,826 and 14,246 MiB without, of 16,303 MiB; allocator live
    peak 11.877 GiB with, 11.951 GiB without;
  - all four receipts pass.
- Config hashes, every shipped config loaded under 0.2.2 and 0.2.3: 53
  identical, the T255 and T383 days changed (c7a000d2 to 7b140980 and
  a30d9e52 to 0b7d0643), the day-one config new (a8ea1d16), the same four
  fixtures that do not load as run configs.  Without the drag the T255
  day's 0.2.3 hash is c7a000d2 again.
- The CPU tier (`pytest -m "not gpu"`) from the installed wheel: 3506
  passed, 86 skipped, 0 failed (Python 3.12, numpy 2.5.4, gpuwm 2.9.13,
  cards hidden); without CuPy installed 3506 passed and 100 skipped.  The
  CLI reference, the 151 doc examples, line endings, the engine boundary
  and the 50 seam files check clean.  `python -m build` and `twine check
  --strict` pass, and `gpuwm-global doctor` from the installed wheel finds
  no gap.

## 0.2.2

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.2.1 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 242
symbols across 72 modules with no gap, and all 50 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.2.0.  Every shipped config keeps its 0.2.1 hash, and a forecast
that names no new option and holds the sea or runs an ocean rule (not
`[ocean] mode = "slab"`) writes 0.2.1's checkpoints byte for byte
(Measured, below, on a held sea; the ocean rules' code is 0.2.1's).  Two
changes need no option: a coupled slab run takes the under-ice share and
the cubic tendency from its first step (Fixed, below), and a data
assimilation cycle whose analysed surface pressure lands between 109,663
and 115,000 Pa now applies the analysis where 0.2.1 kept the background.
The coupled slab ocean no longer grows the southern
summer pack, and its heat-flux correction no longer steps at mid-month.
Three experimental switches on the Grell-Freitas shallow arm, off by
default, vent the trade-wind cloud sheet.  A held-sea run can be moved onto
an ocean rule from one of its checkpoints, `gpuwm-global score` decodes its
reference analyses again, and the data assimilation and the worlds
preflight use the radiation's 1,150 hPa ceiling.

### Fixed

- Coupled ocean (`[ocean] mode = "slab"` only): under ice, a cooling
  heat-flux correction Q is now carried only in the share (1 - A) / (1 - C_i)
  where the slab's ice fraction A is above the climatology's C_i, and in
  full otherwise.  Under the southern summer pack the calibrated flux is
  mostly the shortwave the climatology's open leads absorbed; charged in
  full to a pack whose leads were closed, it was a sink of -78 W/m2 in
  December and -87 W/m2 in January on the cells where the model and the
  climatology both held ice, and the new ice it formed closed the leads
  further.  The 180-day coupled run, resumed from its 30 November
  checkpoint with the share (RTX 5090), against the same run without it,
  1 December to 28 January.  That run is on the long-run line, not this
  tree: the fix (`9aaac7d`) was cherry-picked onto it, so its
  `ocean_coupler.py`, `ocean_slab.py` and the other ocean modules are
  this release's byte for byte, but its atmosphere is the long-run
  line's:
  - southern ice volume 7.21 to 2.52 thousand km3, against 7.24 to 11.04;
  - southern 15 % ice area -53 % (8.47 to 3.94 million km2), against
    -11 %; the climatology's fell 53 % (13.22 to 6.24);
  - monthly-mean Q on the shared cells -30 W/m2 in December and -42 in
    January, against -78 and -87;
  - northern ice area at 28 January 13.52 million km2, against 13.59;
  - largest column residual of the energy books 1.05e-2 J/m2, against
    1.09e-2; open-ocean SST against the climatology -0.471 K, against
    -0.472.
  The share has been measured over December and January only; under the
  northern summer pack (July and August, Q -51 to -61 W/m2) it is not yet
  measured.  `docs/ARWEN_GLOBAL_OCEAN.md` has the derivation.
- Coupled ocean: the climatology's tendencies dC/dt now come from a
  shape-preserving cubic (PCHIP) through the monthly planes at their
  mid-month instants, and each step books the curve's change over the
  step, so a column driven by the calibrated flux still reaches each
  month's plane at mid-month.  The piecewise-linear slope it replaces
  stepped at every mid-month instant: in the same run the ocean mean of Q
  jumped by 12.07 W/m2 on 16 December and 12.16 W/m2 on 16 January (21 to
  27 W/m2 at the August to November instants); with the cubic, 0.04 and
  0.08 W/m2, and no 6-hourly change exceeded 0.18 W/m2.
- `gpuwm-global score`: the scorecard's default reference decoder was a
  name only `score_forecast` imported, so the first reference analysis
  raised `NameError: name 'decode_reference' is not defined` and the whole
  scorecard was lost (measured with the 0.2.0 wheel on a T255 72 h run; it
  stopped at lead 24 h).  The decoder is imported where it is used
  (`tests/test_forecast_scorecard.py`).
- Data assimilation and worlds: the radiation's pressure ceiling is
  115,000 Pa since 0.2.1, but the ensemble's member reading, the
  analysed-state bound, the twin's displaced-start refusal
  (`da/analysis.py` `RADIATION_SURFACE_PRESSURE_CEILING_PA`) and the worlds
  preflight row (`surface_pressure_ceiling_pa`) still used the gas tables'
  109,663 Pa, so they named or refused states the radiation steps.  All of
  them now use 115,000 Pa, and the planet row says air thicker than 1.15
  bar.  `tests/test_rrtmgp_pressure_ceiling.py` holds the three numbers
  equal.  Nothing changes for a state under 109,663 Pa; the 2026-09 DA
  deaths the DA page cites were measured against the old ceiling and have
  not been re-run.

### Added

- Three experimental switches on the Grell-Freitas shallow arm, in
  `native_adapter_options`, all off by default and read only under
  `cumulus = "gf"`; each joins the config hash only when set:
  - `gf_shallow_buoyancy_from_cloud_base`: the cloud-top test integrates
    the plume's buoyancy from cloud base, as GSL's GFL and MPAS v8.4.2 do,
    instead of from the source level as WRF v4.6.1 does
    (`module_cu_gf_sh.F:530`);
  - `gf_shallow_blqe_to_cloud_base`: the boundary-layer quasi-equilibrium
    closure sums the PBL's MSE tendency over 1..kbcon, as GFL does, instead
    of 1..kpbl;
  - `gf_shallow_depth_min_m`: a shallow cloud less than this many metres
    deep exits, the deep arm's ierr 6 test; this one is the package's own.
  - Cause: on a probe of the T255 control from GDAS 2026-08-01 00Z, WRF's
    shallow arm exited on ierr 5 in 67, 47 and 66 % of the open-water
    columns of the SE Pacific, NE Pacific and S Indian trade boxes; under
    the cloud-base integral 94 to 97 % of those columns convect.
  - 10-day T255 twins from GDAS 2026-08-01 00Z, exact 00Z-00Z means from
    the daily checkpoints over 2 to 11 August, against CERES EBAF 4.2.1
    August, all three set (500 m) against the control: excess reflected
    shortwave global +3.7 against +14.9 W/m2, ocean 30S-30N +7.9 against
    +30.1, SE Pacific trades +4.9 against +57.4, Peru stratocumulus -40.3
    against -32.1, net TOA -5.4 against -15.0.  Z500 RMSE 7.93 / 25.14 m
    at 24 / 72 h against 8.12 / 24.90; T850 at 72 h 1.899 against 1.872 K;
    rain against GPCP, 48-72 h, RMSE 10.63 against 10.98 mm/day.  The
    cloud-base integral alone collapses the stratocumulus decks (Peru cover
    0.64 to 0.23); the depth minimum keeps them (0.58).  Open: NE Pacific
    trades +43.3 W/m2, 30N-60N land 5.5 W/m2 darker than the control, and
    one August case.  `docs/trade-shallow.md` has the cause, the probe
    (`tools/arwen_global_shallow_probe.py`) and the twins.  The twins ran
    the 90-day anomaly arm's config (the albedo pair on), not a shipped
    config.  With the three off, a 24 h run on the branch matched the
    control's step-288 checkpoint in all 135 state arrays.
  - A 30-day December twin on the same config, from GDAS 2025-12-16 00Z,
    all three set against none, monthly means against CERES December and
    January: the trades vent as in August (ocean 30S-30N +3.3 against
    +26.0 W/m2, SE Pacific -0.9 against +48.7, global +5.8 against +16.3,
    30-day rain field RMSE against GPCP 2.94 against 3.30 mm/day), but the
    Peru deck goes from +8.5 to -28.8 and the Australian deck from +24.5
    to -15.8: in the southern summer the depth minimum does not keep the
    decks.  The page has the table.
  - Divergence ledger rows `GF-CU-13` (new) and `GF-PY-3`, `GF-PY-4`,
    `GF-CU-3` (carried side moved).
- `--branch-from-other-config` may now change the ocean rule between hold,
  anomaly and climatology.  The rules are closed forms of model time built
  from the cold state, so the branch's first step evaluates what a
  straight run under the new rule evaluates at that instant; the receipt's
  `branch.surface_boundary` names the parent's boundary and the branch's,
  and the branch's checkpoints carry the branch's, so its own restarts
  continue it.  A boundary with prognostic arrays (the slab, a
  calibration) is still refused.  On the 50-day T383 run of 2026-11-12
  (held sea), branched at day 25 onto `mode = "anomaly"`,
  `anomaly_reference = "start"`: the SST off Tampico read 24.36 C against
  the held 26.25 C, and the largest 24 h open-water evaporation fell from
  55.05 to 46.61 kg/m2 (day 26-27: 52.10 to 44.61).
  `tests/test_arwen_global_ocean_rules.py` holds the branch and its
  restart bit for bit.

### Changed

- The carried-physics divergence ledger has rows for the 0.2.1 radiation
  ceiling (`RRTMGP-29`, `GAS-CU-2`, `NPREF-17`) and the shallow-arm
  switches, measured against the 2.8.5 baseline the rows are keyed to (912
  rows).  The ledger has not been re-baselined on 2.9.13, the engine the
  declared range installs: against it the tree measures 1,449 hunks, 526
  carrying a row forward, 923 new and 386 stale, and the two comparison
  nodes of `tests/test_engine_divergence.py` skip as they did in 0.2.0
  and 0.2.1.  `docs/CARRIED-PHYSICS-DIVERGENCE.md` says where the rows
  stand.

### Measured

- On an RTX 4090 (24 GB), the shipped
  `arwen_global_gdas_t255_native_sl_si_24h` from its GDAS 2026-09-01 00Z
  analysis, run 24 h by the 0.2.1 tree and by this tree in one card
  lease, checkpoints compared with
  `tools/arwen_global_checkpoint_identity.py`, 2026-11-14.  The config
  hash is 0.2.1's (c7a000d28c206ace).  A cold start wrote its nine
  checkpoints, steps 0 to 288 every 3 h, byte-identical to 0.2.1's (44
  and 135 arrays, archive hash included).  A restart from 0.2.1's 12 h
  checkpoint (step 144) to 24 h wrote steps 180 to 288 byte-identical to
  0.2.1's straight run.  All three receipts pass.  This config runs
  Grell-Freitas with the three shallow switches off and the sea held, so
  it covers the defaults; the slab ocean and the ocean-rule branch are
  not on its path.
- The under-ice share and the cubic tendency with the ocean off: the
  shipped `arwen_global_gdas_t255_native_sl_si_6h` dated to GDAS
  2026-09-01 00Z, on 0.2.0 and on the fix branch (0.2.0 plus `9aaac7d`;
  RTX 5090, 2026-11-14), wrote steps 0, 36 and 72 identical array for
  array and in `self_sha256`.  On this tree the held-sea run above stands
  for it.
- Every shipped config hashes as in 0.2.1: the same 55 hashes, and the
  same four fixtures that do not load as configs.
- The CPU tier (`pytest -m "not gpu"`) from the installed tree passed:
  3434 passed, 86 skipped, 0 failed (Python 3.12, numpy 2.5.4, gpuwm
  2.9.13, cards hidden).  Without CuPy installed the same tree gave 3434
  passed and 100 skipped.

## 0.2.1

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.2.0 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 242
symbols across 72 modules with no gap, and all 50 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.2.0.  Every shipped config keeps its 0.2.0 hash, and a forecast
that names no new option writes 0.2.0's checkpoints byte for byte
(Measured, below).  One experimental option, off by default, takes the
jet ripples out of T383 runs and costs northern-hemisphere skill there.
A run, a restarted segment or an export no longer repeats the Legendre
analysis table's solves when an earlier process on the same machine has
already made that table.  A run whose air gets deeper than the radiation's
gas tables (1,096.6 hPa) no longer stops (Fixed, below).

### Fixed

- Radiation: a model layer deeper than the RRTMGP gas tables' 1,096.6 hPa
  is now looked up at the tables' deepest pressure instead of stopping the
  run.  A T127 year run stopped at day 193 on a 1,097 hPa layer on the
  Caspian shore, where the truncated terrain dips to -101 m; resumed under
  this fix it ran through that step.  Layers up to 1,150 hPa are accepted
  and deeper ones are still refused.  Only the gas-optics lookup moves
  (the UFS/CCPP RRTMGP interface bounds the pressure the same way); runs
  that stay inside the tables are bit-for-bit unchanged: a 12 h winter
  T255 run from GDAS 2025-12-16 00Z wrote five checkpoints, 589 arrays,
  byte-identical to 0.2.0's (RTX 5090).
  `tests/test_rrtmgp_pressure_ceiling.py` and
  `tests/test_rrtmgp_pressure_ceiling_cuda.py` hold both halves.

### Measured

- On an RTX 5070 Ti (16 GB), the shipped
  `arwen_global_gdas_t255_native_sl_si_24h` from its GDAS 2026-09-01 00Z
  analysis, this tree and 0.2.0 in one card lease, checkpoints compared
  with `tools/arwen_global_checkpoint_identity.py`, 2026-11-13.  The
  config hash is 0.2.0's (c7a000d28c206ace).  A cold start to 3 h wrote
  step-0 and step-36 checkpoints byte-identical to 0.2.0's (44 and 135
  arrays, archive hash included), both when this tree built the Legendre
  table and wrote it to the cache and when it read the table back.  A
  restart to 6 h wrote a step-72 checkpoint byte-identical to 0.2.0's
  restart (135 arrays, archive hash included): from 0.2.0's step-36
  checkpoint with the cache read and with it off, and from this tree's
  own step 36.  The cache file the card run wrote is byte-identical to
  the one a CPU-only build on the same host wrote.
- Every shipped config hashes as in 0.2.0: the same 55 hashes, and the
  same four fixtures that do not load as configs.
- Start-up.  On that card with the host otherwise quiet (two leases,
  2026-11-13), the table took 0.29 to 0.36 s to build under 0.2.0 and
  0.19 to 0.23 s to read, and a restarted segment reached its first step
  in 9.10 to 9.32 s against 9.19 to 9.60 s under 0.2.0.  On a quiet host
  the saving is about a tenth of a second, inside the run-to-run spread.
  Most of the rest is the cold state rebuilt from the analysis (5.7 to
  5.8 s), which this release does not change.  On a busy host the build
  is what stalls.  On the same host's CPU, with other jobs loading it
  (load average up to 33), building the table in a fresh process took
  0.20 to 20.3 s at T255 and 0.87 to 158.7 s at T383 over three tries
  each; reading it took 0.09 to 0.48 s and 0.24 to 1.50 s.
- The CPU tier (`pytest -m "not gpu"`) passed: 3420 passed, 86 skipped,
  0 failed, with the cache off (the suite's default).  With the cache on,
  the suite passed on this tree before the radiation fix was merged
  (3417 passed, 86 skipped); the fix does not touch the transform.

### Added

- `[semilag] extrapolation = "settls_residual_departure"` (experimental,
  off by default; it enters the config hash when set).  It reads the
  extrapolated nonlinear residual at the departure point and keeps the
  shipped trajectory: the residual half of `settls_departure`.
  - It removes the jet ripples.  At T383 and 300 s from the GDAS
    2026-11-12 12Z analysis, the RMS of the divergent wind above degree
    280 near 240 hPa, inside winds over 50 m/s, was 0.26 m/s at 96 h,
    against 1.21 m/s with the shipped `settls` (0.30 m/s in the analysis),
    on an RTX 5090.
  - It costs skill at T383.  Over four 48 h forecasts at 300 s (GDAS
    2026-11-09 and 2026-11-10, 12Z and 18Z), the northern-hemisphere
    500 hPa height RMSE was 21.20 m, against 20.38 m with `settls`; it was
    higher in all four.
  - At T255 it is neutral.  On two 48 h forecasts (2026-11-10 12Z and
    18Z) every score moved by less than 0.08 m or 0.04 m/s.
  - So the default stays `settls`.  The option is for runs where clean
    jets matter more than that skill, for example pictures.  It runs on
    one card at any latitude-band count; a run on several cards refuses
    it by name.  `tests/test_arwen_global_semilag_settls_departure.py`
    reads it off the step: the departure points are the shipped ones bit
    for bit, and the gathered bundle carries (dt/2)(N^n - N^{n-1}) more
    than the shipped one.

### Changed

- `[semilag] extrapolation = "settls_departure"` (0.1.8, experimental)
  now runs on one card at any latitude-band count, as the new option
  does.  It used to be refused on the banded step, so the T383 configs
  sized to latitude bands on a 12 or 16 GB card could not take it.  On
  the CPU, two and four bands give the one-band result with and without
  `level_rate_limiter` (`tests/test_arwen_global_semilag_bands.py`); on a
  card, the residual form on four bands wrote the one-band run's 6 h
  checkpoint, all 135 arrays, byte for byte.
- The Legendre analysis table is kept on disk per machine and reused
  (`arwen_global/spectral/legendre_cache.py`).  Building it means one small
  linear solve per order, and those solves are the only BLAS work in the
  table build.  A multi-threaded BLAS on a busy host stalls on them:
  measured on a 24-core host with other jobs running, the T255 solves took
  10.9 s where they take 0.12 s on an idle host, and the T383 ones 10.7 s
  where they take 0.6 s.  Every run, restarted segment and export paid
  that again.  Now the first process writes the table, and later ones on
  the same machine read it.
  - The file holds exactly the arrays the build hands to the card.  A run
    that reads it holds the same bits as a run that builds the table
    (Measured, above).
  - A file is used only when everything the solves depend on matches:
    the grid, the band, the dtype, the Gaussian nodes and weights, the
    source of the three modules involved, the numpy version and its BLAS
    build, the machine and CPU model, and the BLAS environment variables.
    Anything else, or a torn or unreadable file, is a miss: the table is
    built as in 0.2.0 and the file is written again.
  - Where: `~/.gpuwm/cache/arwen-global-legendre`, or the folder named by
    `ARWEN_GLOBAL_LEGENDRE_CACHE`; set it to `off` to turn the cache off.
    One file per truncation, grid, band and dtype: 54 MiB at T255 and
    176 MiB at T383 in float32.  Truncations below 127, tables over 1 GiB,
    a transform split over several cards and a streaming transform are
    never cached.  The folder can be deleted at any time.
  - `tests/test_global_spectral_legendre_cache.py` holds the cached table
    to the built one bit for bit in float32 and float64, along with an
    analysis and a synthesis through it, and checks that a torn, corrupted
    or foreign file and an unwritable folder each fall back to the build.
    The test suite runs with the cache off.

## 0.2.0

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.1.9 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 242
symbols across 72 modules with no gap, and all 50 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.1.9.  Every shipped config keeps its 0.1.8 and 0.1.9 hash.
Noah's snow age can now be carried (`snow_ageing = true`, off by default,
which a long run sets).  The one array a shipped forecast writes
differently is the open-water reservoir `surface_water_kg_m2`, which no
native scheme reads (Measured together, below).  The long-run keys, the
albedo options, the coupled ocean and other worlds are experimental and
off unless a config names them.  A one-card run that runs out of card
memory carries on from its newest checkpoint on a lighter plan (A run
that runs out of memory, below).

### Measured together

- On the RTX 5090s of one host, the shipped
  `arwen_global_gdas_t255_native_sl_si_24h` and
  `arwen_global_gdas_t383_native_sl_si_24h` to 3 h, this tree against
  0.1.7 (which 0.1.8 and 0.1.9 match array for array), checkpoints
  compared with `tools/arwen_global_checkpoint_identity.py`, 2026-11-13.
  With no
  option named, the config hash is 0.1.7's at both truncations
  (c7a000d28c206ace and a30d9e5284289755), and the step-0 checkpoint's
  43 arrays are byte-identical.  At step 36, 132 of the 133 arrays 0.1.7
  writes are byte-identical.  The one that differs is the surface
  reservoir `surface_water_kg_m2` (largest difference 52.9 kg/m2 at T255
  and 85.0 at T383).  It differs because on open-water columns the
  exchange account now takes the water the physics moves, and the
  reservoir there keeps its value.  No native scheme reads that
  reservoir, so the atmosphere, the surface and the soil are
  byte-identical.  The checkpoint also carries the account and the
  open-water evaporation sum (`ocean_exchange_kg_m2`,
  `ocean_evaporation_day_kg_m2`).  `tests/test_arwen_global_open_water_exchange_cuda.py`
  holds the same on a card for one physics half-step of the native suite.
  With `open_water_exchange = false` the step-0 and step-36 checkpoints
  are byte-identical at both truncations, 43 and 133 arrays, archive hash
  included.  Every shipped config hashes as in 0.1.8 and 0.1.9.
- 0.1.7's step-0 checkpoint of the T255 day continued by this tree to
  step 36 with `open_water_exchange = false` ends byte-identical to
  0.1.7's straight run, all 133 arrays.  With no option named, the run is
  given a zero account, its receipt says so, nothing is reported missing,
  and it ends as above: one array differs, the reservoir.
- The experimental pieces together, on one RTX 5090 each, 2026-11-13: one
  T255 day of a long-run config (anomaly mode, the orographic drag, the
  albedo pair, snow ageing, monthly means and retention) passed in
  261 s; one coupled T255 day with the slab, the albedo pair, snow
  ageing and `--tapes`, restarted at +12 h, ended with the +18 h and +24 h
  checkpoints identical to the straight day's (143 of 143 arrays) and
  both later tapes byte-identical; a world with no mountains at T127 ran
  its day and passed in 83 s.
- Experimental, each off unless a config names it: the long-run line's
  `[ocean]`, `[land]` and `[output]` long-run keys, `[radiation] ghg_year =
  "valid"` and the orographic drag; the three albedo options; the coupled
  ocean (`[ocean] mode = "slab"`); and other worlds (`[world]`).

### Added

- Three native options for the sunlight the model reflects too much of,
  each off by default and outside the config identity at its default, so
  a config that names none of them keeps its hash and its arithmetic
  (`docs/albedo-fixes.md`):
  `morrison_droplet_number = "land_ocean"`, 300 cloud droplets per cm3
  over land and 100 over ocean and sea ice in place of WRF's single 250,
  a declared departure (`docs/morrison-droplet-number.md`);
  `radiation_liquid_radius = "wrf_relcalc"`, the liquid radius WRF 4.7.1's
  RRTMG radiates for Morrison cloud, 14 um over sea and ice and 8 um over
  warm snow-free land; and `radiation_cloud_inhomogeneity = "tiedtke1996"`,
  the in-cloud liquid and ice paths times 0.7 in both solvers, a declared
  departure. The 0.7 enters the carried driver as `cloud_path_scale`,
  ledger row `RRTMGP-28`.
- A long run sets the first two (`docs/ARWEN_GLOBAL_LONG_RUNS.md`, "The
  cloud options a long run sets"). Over three 48 h starts in November, July
  and January they cut the reflected excess over CERES-EBAF by 57 to 59%,
  moved OLR by under 0.5 W/m2 and left Z500 better in all six case-leads;
  the 0.7 factor stays off because it worsened 2 m land temperature at
  24 h by 0.05 to 0.09 K in every case. `tools/arwen_global_albedo_twins.py`
  reads the twins. Over the ten-day anomaly run from GDAS 2026-11-10 18Z,
  against a control beside it, the pair took the top-of-atmosphere net
  from -21.33 to -4.85 W/m2 and the planetary albedo from 0.392 to 0.343
  (CERES-EBAF November 0.303), with every receipt gate passed.
- `gpuwm-global climatology fetch` and `gpuwm-global climatology build`
  make a monthly sea-surface climatology on a config's Gaussian grid from
  public data that needs no credential.  The sources are rows of
  `arwen_global/data/surface_climatology_sources.json`: NOAA OISST v2.1's
  1991-2020 monthly long-term mean of SST and sea-ice concentration
  (two files, 76,949,346 bytes), and, with `--with-mld`, the NOAA WOA18
  monthly mixed-layer depth (twelve files, 85,127,108 bytes, pinned as the
  regional engine pins them and read from its ocean cache when it holds
  them).  Every file is checked against its row's byte count and SHA-256
  when it is installed and again before it is decoded, and a file with other
  bytes is refused.  The build regrids each month with the cold start's
  open-water rule over the statics ocean (NaN on land and on inland lakes,
  whose skin follows its own energy budget; the mixed-layer depth is not
  clamped), writes one `.npz` that is the same bytes from the same inputs,
  and records in it, for every month, the area-mean SST, each hemisphere's
  ice extent at 0.15 and 0.5 and its ice area, and the mean mixed-layer
  depth, on the source grid and on the model grid, over each grid's own
  water and over the intersection of the two masks.  The file is a
  climatology, not any year's observation.
  Measured at T255 and T383 from the pinned inputs: 43,131,694 and
  96,957,814 bytes, each built in under 30 s on a host CPU, and two T255
  builds gave the same SHA-256.  On the intersection (model ocean whose
  source stencil is all water, and for the SST also off the coasts and the
  ice) the model's mean SST is 0.030 to 0.041 K below OISST's at T255 and
  0.015 to 0.020 K below at T383 in every month, and per hemisphere the ice
  extent at 0.15 is within 0.8 %, the extent at 0.5 within 1.6 % and the ice
  area within 0.4 % of OISST's.  Over each grid's own water the figures
  differ more, and the difference is the coastline: the southern extent at
  0.15 is 1.9 % to 4.7 % below OISST's (12 % below at 0.5 in February),
  because about 0.48 million km2 of OISST water that is ice-covered in every
  month lies under columns the T255 statics call land, almost all of it
  between 63 and 79 S.  The built mixed-layer depth is under 10 m on 1.3 %
  of the ocean column-months, the smallest 3e-12 m.
- Long-run rows in `diagnostics.jsonl`, on every output step: area means
  on the model's Gaussian quadrature of the radiation (the held top and
  surface planes and the flux-times-dt accumulators), the sampled sensible
  and latent heat fluxes, accumulated and frozen precipitation, the moist
  total energy, precipitable water, the dry-mass surface pressure
  (ps - g W), global-mean temperature at 850, 500, 200, 50 and 10 hPa,
  polar-cap (60-90 deg) temperature at 10 and 50 hPa, zonal-mean wind at
  10 hPa at 60 N and 60 S, the land reservoir minimum, land-mean soil
  water, NH snow-covered area, the Nino-3.4 mean SST and the sea-ice
  extent and area by hemisphere.  A row whose inputs a state does not carry
  is left out, and a pressure level above the model top is written as
  null.  The physics' per-call flat means (`mean_outgoing_longwave_w_m2`
  and the rest) are unchanged.  No arithmetic of the forecast changed: the
  rows read the state and write nothing back.
- `arwen_global.longrun_gates` and `tools/arwen_global_longrun_report.py`:
  gates LR-0 to LR-9 of a long run, read from one run folder or a chain of
  segment folders, per 30-day window and over the whole run, with the
  temperature trends' 95% intervals widened for lag-one autocorrelation.
  The temperature gates are gated against another run or a table of
  analysis means by date, which the tool's `analysis-table` mode writes
  from isobaric analysis GRIBs through the cold start's mapping; a daily
  long-term-mean file is reported beside them and never gated.  Every segment of a run longer than 384 h carries
  the same table in its receipt as `longrun_gates`, outside the receipt's
  `status`.  All thresholds but the restart and boundary identities are
  engineering proposals, not standards.  A required gate passes only on
  all of its evidence: LR-2 reads "not measured" until a run writes the
  open-water E - P row its account identity needs, which no package writes
  yet, even where the land reservoir minimum holds.
  `tests/test_arwen_global_longrun_gates.py` holds the known answers (the
  area mean of cos(lat) against pi/4, a planted 0.2 K per 30 days drift
  read back to 1e-9, ps - g W on the smoke state, the Nino-3.4 mean of an
  SST of 300 + latitude, an energy budget that closes and a planted leak
  read back, an interval widened by the effective sample size under
  lag-one correlated noise, a seasonal TOA swing reported per window while
  its year passes); `tests/test_arwen_global_longrun_report_tool.py` holds
  the tool's long-term-mean reader to a cycle written for it.  Read over the `diagnostics.jsonl` of the 10-day T383 GDAS
  2026-10-15 forecast, the report gives that run receipt's mass and water
  drifts, 4.359e-07 and 1.309e-08; that run predates these rows, so the
  rest of its table reads "not measured".
- Gate LR-10 and two rows on every diagnostics row,
  `global_mean_relative_angular_momentum_kg_s` and
  `global_mean_axial_angular_momentum_kg_s`; the gate reports the planet's
  totals per day and their drift per 30-day window, with no threshold yet.
- The daily summary gains the surface stress along the 10 m wind,
  sea-level pressure (mean and minimum), 850 hPa relative vorticity (mean
  and the day's cyclonic maximum), the sensible and latent heat fluxes,
  the liquid and ice water paths, cloud cover, and the interval means of
  the top-of-atmosphere and surface radiation, made on the host from the
  checkpoint arrays (`docs/ARWEN_GLOBAL_LONG_RUNS.md`, "Daily summary").
  Before retention deletes a checkpoint it writes its header, every
  array's SHA-256 included, to `arwen_global_pruned_checkpoints.jsonl`,
  and every receipt carries a `software_stack` block.  None of this adds an
  array to the state or changes what the model integrates
  (`tests/test_arwen_global_longrun_ops.py`).
- `tools/arwen_global_longrun_chain.py` (source distribution) runs one long
  forecast as a chain of restarted leases of a few model days each, so a
  shared card is held one lease at a time: it resumes from the newest whole
  checkpoint after a stop or a crash, refuses a second copy of itself,
  stops when a receipt or a required long-run gate fails, prunes across
  the leases by the config's retention rule, and writes a `STATUS` line per
  lease with the lease's planetary albedo and net top-of-atmosphere flux.
  `tests/test_arwen_global_longrun_chain.py`.

- The ocean rule is a config choice: `[ocean] mode = "hold"`, `"anomaly"`,
  `"climatology"` or `"series"`, with a matching `[sea_ice] mode` (`match`
  by default).  Anomaly mode carries the starting SST anomaly on the
  evolving OISST 1991-2020 climatology: its global open-ocean mean
  persisted, the rest smoothed to T42 and damped at 90 days poleward of
  30 degrees and persisted within 10 degrees of the equator in the
  tropical Pacific; the ice anomaly damped at 60 days; no ice over water at
  or above T_f + 2 K; new ice 1.0 m thick.  Climatology mode removes the
  anomaly with a 3-day ramp.  Both are closed forms in time from the
  climatology file and the cold state, so they read no surface record
  later than the start, and lakes are never written.  `slab` and the
  `thermodynamic` sea ice are the experimental coupled ocean below.  The
  tables enter the config identity only where they change arithmetic, the
  climatology and the reference series by content SHA-256; `mode = "hold"`
  and the series keep every hash.  The receipt's `surface_boundary` block
  and per-output rows in `diagnostics.jsonl` say which rule produced the
  boundary, with its starting anomaly and the LR-8 rows
  `boundary_max_abs_sst_departure_k` and
  `boundary_ice_record_mismatch_columns`.  A checkpoint carries the rule's
  identity block and a restart under another rule, climatology or
  starting anomaly is refused before its arrays are used.
  `docs/ARWEN_GLOBAL_LONG_RUNS.md`; `tests/test_arwen_global_ocean_rules.py`
  (the closed forms at known points, the floor, the ice clip and warm-water
  rule, the T42 projection of a degree-10 and a degree-50 harmonic, lakes,
  identity, the long-run refusal, and a smoke run restarted at a
  mid-month instant inside an ice record, bit for bit).
- `gpuwm-global surface-series ... --anomaly-reference` builds the anomaly
  reference: the 28 six-hourly analyses of the 7 days before `--cycle`,
  read with the analysis sea ice as land in the open-water rule.
- `[land] greenness = "daily" | "start"`: the vegetation fraction, leaf
  area index and background albedo march with their monthly climatology
  once a model day, by default under a rule or a series.
- `[output]`: `keep_checkpoints = "newest:N"` or `"newest:N+month-ends"`
  prunes during the segment; `mean_state = "monthly"`, `"daily"` or both
  writes time-mean files in the checkpoint layout, refused as a restart and
  labelled `ARWEN_TIME_MEAN` when exported, with the open windows' float64
  sums in a sidecar so a restart writes the straight run's means;
  `daily_summary = true` writes a daily file of 2 m temperature mean,
  minimum and maximum, humidity, wind, outgoing longwave, skin, sea ice,
  snow cover and the day's precipitation.  A run that states these or runs
  longer than 384 h prints a disk estimate scaled from the measured T383
  checkpoint and is refused when the estimate exceeds the free space less
  `disk_reserve_gb` (20).  Outside the identity.
  `tests/test_arwen_global_longrun_ops.py`.
- Orographic gravity-wave drag, off by default: the native adapter option
  `orographic_drag = "kim-gwdo"` runs WRF's `gwd_opt = 1`, the KIM
  orographic drag with flow blocking, carried from the engine's
  terrain-drag launcher and kernel (gpuwm 2.9.13) and cut to that one
  option, and adds it to YSU's momentum tendencies from the column state YSU
  read, where WRF's PBL driver adds it.  Before it the model had no
  gravity-wave drag of any kind; non-orographic drag is still absent.  The
  option joins the config identity only when on, so every config hash and
  checkpoint of 0.1.5 is unchanged.
- `gpuwm-global statics CONFIG --orographic` builds the ten sub-grid
  orography planes the drag reads (VAR, CON, OA1-4, OL1-4) from WPS_GEOG's
  `orogwd_10m`, as GEOGRID.TBL.ARW's rows do (`average_4pt`, 0 on the
  land-surface statics' water columns), into a self-hashed cache beside the
  land-surface one.  The engine's static-fields crate refuses the Gaussian
  grid for these rows (`orographic WPS fields need a projected grid`, gpuwm
  2.9.13), so the package reads the dataset itself.  A drag-on run without
  the cache is refused before the card is touched, and the cache's identity
  travels in the physics state, so a restart that would read different
  statistics is refused.
- The known answer: the package's float32 numpy transcription of the kernel
  lands on every output word of WRF v4.6.1's own `gwdo` (the engine's oracle:
  149 columns, two vertical grids, six grid lengths, copied with its hashes
  into `tests/data/oracles/gwdo_kim`) without a card; on a card the carried
  kernel does too under the engine's strict numerics, and it is the engine's
  kernel word for word in either numerics mode.  Measured once at T255 (GDAS
  2026-10-15 00Z, 72 h, RTX 4090): both arms stable, the drag slowing the
  30-60N lowest-level zonal-mean wind from 0.98 to 0.84 m/s and the 60N
  50 hPa zonal-mean wind from 10.74 to 10.35 m/s, 712 s against 696 s at
  the same 8.64 GiB peak, and a restart at 24 h bit-identical to the
  straight run at 48 h (`docs/ARWEN_GLOBAL.md`).
- `[radiation] ghg_year = "valid"`: the native radiation re-selects CO2,
  CH4 and N2O at every call from the call's valid year, so a run that
  crosses 31 December radiates the next year's NOAA GML row from its first
  call at or after 1 January 00:00 UTC. The default, `"start"`, is the
  start year for the whole run, as before: a config that sets nothing, or
  sets `"start"`, keeps its config hash, and `"valid"` joins it. Inside one
  calendar year the two rules are bit-identical. The key is refused under
  the reference suite, whose gray radiation carries no gas by year, and is
  spelled only in `[radiation]`. The carried driver gains one public
  method for it, `RRTMGPRadiation.select_trace_gases`, filed as
  `RRTMGP-27` in `docs/CARRIED-PHYSICS-DIVERGENCE.md` and offered to the
  engine line.
- The receipt's `radiation_greenhouse_gases` record names its rule
  (`ghg_year`), carries `by_valid_year` under `"valid"` (one row per
  calendar year the run touches, with the instant it takes effect), and,
  under either rule, a `held_past_table` line when a year read is past the
  table's last published year. A run dated 2026 now says in that line that
  it radiates 2025's means; the values it radiates are unchanged.
- `docs/ARWEN_GLOBAL_LEVEL5.md` states what the radiation is forced with
  by date: the gases and their year rule, WRF's 365-day solar orbit and
  its one-day leap-year slip, the 1370 W/m2 solar constant, the CAM ozone
  climatology, and no aerosols. `NOTICE` names the NOAA GML means and the
  terms their files state.

- Other worlds, experimental: the same forecast run on Earth changed. A
  `[world]` table at the end of an analysis-start config lowers or removes
  the mountains (`[world.terrain]`, the surface pressure moved through the
  analysis's own air, Earth's amount of air kept by default), tilts the axis,
  dims or brightens the sun, shortens the day or gives it another orbit
  (`[world.sun]`), raises the sea, takes the land away or sets the sea's
  temperature and ice by rule (`[world.sea]`), sets CO2, methane and nitrous
  oxide (`[world.air]`, the last two new options of the native radiation
  beside CO2), or reads an invented continent from a world folder built from
  a map (`[world.surface]`; the PALEOMAP reader fetches the published maps
  on request and ships none of them).  A world has its own config identity,
  is checked on the CPU against the frame it starts from before any card
  time (refused by name when its terrain is too tall, its start pressure is
  over the radiation's ceiling or its warmed land skin is over 355 K), is
  labelled science, illustration or not supported yet from the rows of
  `data/world-limits.json`, and is named on its receipt, checkpoints and
  tapes.  A restart under another world, the regional parent exports,
  cycling, the assimilation door and the scorecards refuse a world run by
  name, and so does the orographic drag under a world that changes its
  terrain or its coast, because its sub-grid statistics are Earth's.  `gpuwm-global world` (`list`, `new`, `describe`, `twin`, `perturb`,
  `diff`) lists the fourteen presets, writes a world config from an Earth
  one, prints the world statement and runs a world beside its Earth control
  and a perturbed control that sets the noise floor.  Two worlds ship as
  experiments: `arwen_global_world_t255_no_mountains` and
  `arwen_global_world_t255_on_its_side`, each the T255 forecast-day config
  with one `[world]` table appended.  `docs/ARWEN_GLOBAL_WORLDS.md` is the
  whole surface, its limits and its credits; NOTICE credits the PALEOMAP maps
  and the Aqua-Planet Experiment profiles.  The `tests/test_arwen_global_world_*.py`
  and `tests/test_arwen_global_worlds_*.py` files hold each part on the CPU.
- With no `[world]` table, or an all-Earth one, nothing moves.  Measured on
  the worlds line on its own: every shipped config hashed as in 0.1.7 with
  and without an all-Earth table appended, and ten steps of the moist smoke config on the CPU write the
  same checkpoint bytes as 0.1.7 with and without it.  On the card, 12
  steps of `arwen_global_gdas_t255_native_sl_si_24h` from its GDAS
  2026-09-01 00Z analysis (`duration_s` and `output_interval_s` set to
  3600) on one RTX 5090, 0.1.7 and this tree back to back, and this tree
  again with an all-Earth `[world]` table appended: the step-0 and step-12
  checkpoints of all three are byte-identical
  (`tools/arwen_global_checkpoint_identity.py`, 43 and 133 arrays, archive
  hash and config hash included).
- The two shipped worlds each ran their forecast day from the same analysis
  on one RTX 5090, beside the plain experiment (2026-11-12, numbers from the
  receipts and `diagnostics.jsonl`).  All three passed every gate, in 368 s
  (no mountains), 364 s (on its side) and 376 s (the plain experiment).  No
  mountains: the lowest surface pressure at step 0 is 92,401 Pa against the
  plain experiment's 51,354 Pa; the global-mean surface pressure is
  98,585.56 Pa against 98,584.67 Pa (Earth's amount of air, to 0.9 Pa), and
  it drifts by 4.4e-7 over the day.  On its side (90 degree tilt): at 24 h
  the mean land skin temperature is 4.48 K warmer than the plain
  experiment's north of 60 N and 2.82 K warmer from 30 to 60 N, and 1.48 K
  and 2.14 K colder from 0 to 30 S and 30 to 60 S (land columns, land
  fraction over one half).  One day is a smoke of each world, not a result
  about it.
- The demo worlds ran at T127 on one RTX 5090 (2026-11-12 and 13; the T255
  forecast-day config cut to truncation 127 at `dt_s = 600`, from GDAS
  2026-09-01 00Z and 2026-11-10 18Z): no mountains for 10 days, on its side
  for 5, the waterworld, a PALEOMAP 250 Ma continent at two soil wetnesses
  and a 12-hour day for 20 each, the sea 70 m higher for 10, and one step
  each at 190, 280, 560 and 1120 ppm CO2.  Every run finished and passed
  every gate; the global-mean surface pressure moved by at most 1.2e-6 over
  a run.  No mountains beside Earth and a perturbed Earth: the start's
  3-hour surface-pressure change over hours 12 to 24 is 1.10 times Earth's,
  and the 500 hPa height difference over 20 to 80 N stays above the
  perturbed control's (ratio 5.1) until about hour 180.  CO2 at 560 and
  1120 ppm adds 2.88 and 5.86 W m-2 of instantaneous net downward flux at
  the top against 280 ppm.  The table, the labels and what each world
  showed are on `docs/ARWEN_GLOBAL_WORLDS.md` under "The demo worlds at
  T127".  The shipped on-its-side config's comment now gives the sun's
  latitude for its own start date (about 19 N on 2026-09-01) as well as
  for a mid-November one.

- Experimental: `[ocean] mode = "slab"`, the coupled ocean of
  `docs/ARWEN_GLOBAL_OCEAN.md`, is admitted by the parser with its keys
  (`slab_qflux`, `slab_depth`, `slab_depth_bounds_m`,
  `slab_relaxation_days`, `slab_target`), with `[sea_ice] mode =
  "thermodynamic"` (which `match` resolves to under the slab, new ice
  0.5 m), and so is `qflux_calibration` under `mode = "climatology"`, the
  run its heat-flux correction is made from.  The slab is a provider of the
  boundary service that advances the mixed layer and the ice mass before
  every step from the fluxes the physics held, writes the ice at each
  record and the mixed-layer temperature as the open-ocean skin; its seven
  float64 arrays ride in the physics state, so a checkpoint carries them and
  a restart resumes them, and its identity block refuses a restart under
  another ocean.  The calibration deposits the same flux by month into two
  arrays while the climatology rule writes the surface.  On the reference
  suite the flux is a net surface flux plane the suite holds only when one
  of them is on.  The render tape gains `SST`, `HML` and `ARWEN_QFLUX` only
  when the checkpoint carries the slab.  `gpuwm-global climatology qflux`
  is the command that writes the correction file from calibration
  checkpoints.  The mixed layer and its heat-flux correction
  (`ocean_slab.py`, `ocean_qflux.py`) and the zero-layer ice mass budget
  (`sea_ice_thermo.py`) are the arithmetic the coupler calls.  Under ice,
  the Q-flux leaves out the climatology's SST slope, because the mixed
  layer there sits at the freezing point.  In the first coupled card run
  that heat went to the pack instead, and the southern ice shrank at twice
  the climatology's rate.  Measured on one RTX 5090: a ten-day calibration
  and a ten-day slab run of the shipped T255 native semi-Lagrangian config
  from GDAS 2026-11-10 18Z (the Q-flux is in sample).  Against the
  climatology, the open-ocean SST moved -0.018 K in the area mean and
  0.224 K rms.  The largest change on one column was 5.46 K, over the
  page's 5 K bound.  The northern ice area grew 0.93 million km2 against
  the climatology's 0.76; the southern grew 0.23 while the climatology's
  shrank by 1.04, so gate OC-4 holds for the SST and not for the southern
  ice.  On every ocean column the energy books agreed to 2.8e-3 J/m2 after
  ten days.  The run restarted from +6, +12 and +24 h ends identical to
  the straight run array for array and in `self_sha256`.  A step cost
  1.1 % more than the calibration run's, and the measured device peak was
  the same within 0.04 GiB.  With
  no `[ocean]` table and under every rule, nothing changes: measured on a
  host CPU, the T3 smoke with no table and under the anomaly, climatology
  and hold rules wrote the same 12 checkpoint files byte for byte as the
  commit before, and the same three tapes; on an RTX 5090, the T255 native
  semi-Lagrangian config from GDAS 2026-09-01 00Z with no `[ocean]` table,
  run for 1 h and for 6 h (12 and 72 steps of 300 s), wrote every
  checkpoint (steps 0, 6, 12 and 0, 36, 72) identical array for array and
  in `self_sha256` to the tree without the ocean packages, read with
  `tools/arwen_global_checkpoint_identity.py`.
  `tests/test_arwen_global_ocean_coupling.py` holds the coupler's known
  answers, the parser's refusals and the slab and calibration smokes
  restarted bit for bit in float64 and in float32;
  `tests/test_arwen_global_ocean_slab.py`,
  `tests/test_arwen_global_ocean_qflux.py` and
  `tests/test_arwen_global_sea_ice_thermo.py` hold the parts' known
  answers.

### Changed

- With native physics, open water is no longer a finite store of water.
  Every physics call books the column water it moves against the explicit
  surface reservoir, and the run refuses when a column's reservoir falls
  below zero; an ocean column whose evaporation exceeds its rain empties
  the 500 kg/m2 most shipped configurations seed in 100 days at 5 mm a day
  of E - P.  On ocean, lake and sea-ice columns the movement, and the
  global water fixer's correction, now go to a signed float64 account,
  `ocean_exchange_kg_m2` in the physics namespace, and the reservoir there
  keeps its value; land is unchanged.  Every water ledger counts the
  account beside the runoff outflow account, so the conserved total is
  held water plus outflow plus the account.  No native scheme reads the
  reservoir, so the atmosphere is the same: one native physics half-step on
  a card from the native T3 smoke cold state, with the account and on the
  finite reservoir, has every prognostic array byte-identical
  (`tests/test_arwen_global_open_water_exchange_cuda.py`, and on the CPU
  with stand-in kernels `tests/test_arwen_global_open_water_exchange.py`).
  The reference suite caps one call's evaporation by its reservoir, keeps
  it, and is not a configuration for long runs.
- In place of the reservoir refusal on open water, the native suite sums
  each open-water column's evaporation over 24 h blocks of model time and
  refuses the run, naming the column, when one passes the native option
  `open_water_evaporation_ceiling_kg_m2_day` (default 60, a provisional
  engineering value, not a measured one);
  `open_water_evaporation_refusal = false` reports without refusing.
  Neither option enters the config identity.
- The native option `open_water_exchange = false` turns the account off
  and keeps the finite reservoir on open water, the bookkeeping of 0.1.8;
  the runaway watch runs only with the account.  It is outside the config
  identity, and a run with it off refuses a checkpoint that carries an
  account (`tests/test_arwen_global_open_water_exchange.py`).
- New readings: the diagnostic `global_mean_ocean_exchange_kg_m2`, the
  in-situ ledger row `water_ocean_exchange_kg_m2`, the receipt row
  `open_water_exchange`, the step diagnostics
  `maximum_open_water_evaporation_day_kg_m2`,
  `maximum_open_water_evaporation_last_day_kg_m2` and
  `open_water_evaporation_exceeds_ceiling`, and a reservoir term of
  `python -m arwen_global.water_budget`.  The account moves no config
  hash, and a checkpoint written without it resumes with a zero account
  and a receipt that says so.
  `docs/ARWEN_GLOBAL_FULL.md`, "Open water: the exchange account".
- The native option `snow_ageing` (default `false`, below) joins the
  config identity when it is on, so a config that sets it hashes apart
  from the same config without it, and a checkpoint written without the
  age is refused by it rather than resumed with other physics.  Off, a
  config keeps its 0.1.8 hash, arithmetic and arrays (no age plane is
  made), and a restart of a checkpoint without the plane reports nothing
  missing; only a run that carries the age names the plane as absent.

- A dated run from an analysis on real statics longer than 384 h with no
  `[ocean]` table and no series is refused before it starts, with a message
  naming the modes; `[ocean] mode = "hold"` runs it with the SST held.  It
  used to hold the SST for the whole run without saying so.  Idealized and
  undated runs are not refused.
- The lower boundary is one service with two kinds of record provider, the
  dated series and the rules (`surface_boundary.py`, `ocean_rules.py`), and
  the greenness march is its own service (`land_climatology.py`).  The
  series' arithmetic is unchanged: the reference-physics smoke with a
  2-record series ends in the state 0.1.5 ends in, every array of its last
  checkpoint byte-identical.

### Fixed

- Noah's snow age (WRF SNOTIME) can be carried between land calls and
  checkpointed as `noah_snotime`, under the native option
  `snow_ageing = true`, which the long-run recipe sets
  (`docs/ARWEN_GLOBAL_LONG_RUNS.md`).  By default, as in every earlier
  release, it is rebuilt as zeros on every land call, so every snowpack
  keeps its fresh-snow albedo for the whole run (0.999 of it after any
  number of days at a 60 s land interval; by Noah's formula a pack ten
  days old has 0.79 of it).  On, it changes the numbers of every native
  run with snow.
  `docs/ARWEN_GLOBAL_LEVEL5.md`, "Snow age";
  `tests/test_arwen_global_snow_age.py` and
  `tests/test_arwen_global_snow_age_cuda.py`.
- The greenness march reads the model day's start, not the instant of the
  first step it runs on.  A run started at 06, 12 or 18 UTC and restarted
  inside a model day took another UTC day's vegetation than the straight
  run (on a field of ten times the month number from 2026-04-03 18 UTC:
  26.452 restarted at 12 h against 26.129 straight).  Restarted series runs
  from such starts change; straight runs whose steps land on each day
  boundary do not.
- A column that changes class at an ice record now leaves the native
  runtime's frozen-surface step.  The masks the runtime keeps for the run
  were meant to be dropped after every re-class, but the drop looked for
  the runtime on the physics bridge instead of on its column suite, found
  none and dropped nothing, so a melted column was still integrated as ice
  and a frozen-over one as water; a series run with crossings had the same
  fault.  Measured on a T255 anomaly-mode run from GDAS 2026-11-10 18Z
  before the fix: a column off Antarctica melted at the first daily record
  went from 273 K to 201 K in four days, and 211 open-ocean columns were
  more than 1e-3 K off the rule at day 5.
- On the numpy backend a series run's checkpoints held the next step's
  skin: the series wrote the SST into the state's own array while the
  checkpoint writer was still hashing it, so such an archive was refused by
  its own hash check on read.  A boundary plane on the host is now
  replaced, never written in place.  The forecast's state was not affected
  (the smoke's last checkpoint is the same bytes before and after); card
  runs copy every array to the host before writing and were not affected.

### A run that runs out of memory

A run that falls back carries on with a different band count and host
tier.  Neither is in the config identity or changes the arithmetic, so
no forecast bit moves, and a run that never runs out of memory runs as
0.1.8 ran.  The card measurements below were taken on 0.1.7 with this
change alone; this release runs the change's tests,
`tests/test_arwen_global_oom_fallback.py`, on its own tree.

- Measured: T383 L40 on the semi-Lagrangian core (the shipped
  `arwen_global_gdas_t383_native_sl_si_24h` with a 50-day duration and
  3-hourly checkpoints), from a GDAS 2026-11-12 00Z
  analysis, on one RTX 5070 Ti 16 GB, 2026-11-13, each segment a new
  process restarting from a checkpoint:
  - The reference: one process from the analysis to 72 h with nothing
    else on the card, one band with nothing parked, 864 steps at 1.854 s
    a step.
  - Nothing going wrong: 9 to 18 h from the reference's 9 h checkpoint,
    0.1.7 then this release on the same card: 1.786 and 1.785 s a step,
    241.8 and 241.5 s of run wall, both one band, the card at 1,104 and
    1,108 MiB free at its tightest.
  - Another process taking 2 GiB of the card 45 s after the start, three
    segments from 36 h: in each the card refused the semi-Lagrangian
    gather's 810 MiB (after steps 473, 439 and 439, with 177, 319 and
    441 MiB free), and each run carried on from its newest checkpoint on
    two bands with the host tier on (priced at 10.83 GiB against the
    13.21 GiB the card then had free) and finished its segment, at 2.454,
    2.480 and 2.565 s a step.
  - Five bare CUDA contexts of other processes (222 MiB each) opening
    45 s after the start, which on 0.1.7 killed one segment in eleven:
    nine segments of 9 h (one of 12 h) from checkpoints between 27 and
    54 h.  In one the card refused the gather's 810 MiB after step 420
    with 481 MiB free; it carried on from step 396 on two bands with the
    host tier on (priced at 10.83 GiB against 14.10 GiB free) and
    finished at 2.561 s a step.  The other eight finished on one band
    without needing the fallback, the card at 5 to 418 MiB free at their
    tightest.
  - Another process taking every byte the card frees, up to 6 GiB, from
    45 s after the start: the card refused 202 MiB with 103 MiB free, no
    lighter plan fitted the 9.02 GiB left (the lightest prices at
    10.23 GiB), and the run stopped 53 s after its start with exit 4 and
    the line naming the other process, the free memory and the
    `--restart ... --latitude-bands 2 --host-spill on --until-s ...` to
    resume with.
  - A chain from the analysis to 72 h in eight 9 h segments, each a new
    process from the last one's newest checkpoint, with another process
    taking 1 GiB of the card 45 s after each start: all eight finished on
    one band without needing the fallback, at 1.786 to 1.822 s a step,
    the card at 18 to 658 MiB free at each segment's tightest.
  - Bytes: all 74 checkpoints the runs above wrote, the 11 written after
    a fallback and the chain's 25 among them, are byte-identical files
    to the reference's at the same step.
- The same shape on the same card, 2026-11-13, a chain of segments from
  the analysis, each a new process, with the out-of-memory error forced
  where the gather is not the one asking:
  - In the radiation, after step 48 of a cold start: carried on from
    step 36 on two bands with the host tier on.
  - In the device-to-host copy of the 12 h checkpoint, after that step's
    diagnostics row was written: carried on from the 9 h checkpoint the
    segment started from, and the diagnostics file holds one row a step.
  - Twice in a row, in the radiation after step 230 on one band and
    after step 223 on the plan it fell to: carried on on two bands with
    the tier, then on four bands with the tier (priced at 10.33 GiB).
  - With `CUPY_GPU_MEMORY_LIMIT` set to 12 GiB, below the one-band
    plan's pool: the limit refused a 101 MiB allocation at the segment's
    first step with 3,172 MiB of the card free, and the run carried on on
    two bands with the tier and finished.
  - Another process taking 2 GiB of the card 45 s after the start: the
    card refused the gather's 810 MiB after step 417 with 339 MiB free,
    and the run carried on from step 396 on two bands with the tier.
  - Bytes: all 15 checkpoints these runs wrote are byte-identical files
    to the reference's at the same step.
  - With `CUPY_GPU_MEMORY_LIMIT` set to 6 GiB, below every plan: before
    the change below, the run rebuilt on two, four and eight bands, each
    plan weighed against the card's 15.2 GiB and each refused by the
    limit, and stopped (exit 4) after 81 s telling the user to free the
    card of what else held it, with nothing else on it.  After it, the
    same segment stopped (exit 4) after 23 s with no rebuild, the line
    saying the 6.00 GiB limit had refused 101 MiB with 9,380 MiB of the
    card free and to raise or unset the limit; and under the 12 GiB limit
    a segment again carried on on two bands with the tier, its line
    naming the limit, and its checkpoint is byte-identical to the
    reference's.

- A run the card runs out of memory under no longer dies if a lighter
  plan fits.  When a one-card run on the card raises CuPy's
  `OutOfMemoryError` (or a CUDA library's out-of-memory error) mid-run,
  it gives its memory back, reads the card again, and carries on from
  its newest checkpoint in the same process on the next plan the door
  prices lower (more latitude bands and/or the pinned host tier; at
  T383 L40 after one band that is two bands with the tier on, the plan
  `--latitude-bands 2 --host-spill on` names).  It prints one line
  naming what the card refused, with how much free, and what else holds
  the card, and the receipt carries a `recovery` row and lists the
  checkpoints written before the fallback beside those written after
  it.  The time lost is the steps since that checkpoint and the
  rebuild, and the rest of the segment runs at the lighter plan's
  speed.  When no lighter plan fits, the run stops with a one-line
  message (exit 4) that names the cause, the free memory, the other
  processes on the card and the exact `--restart ... --latitude-bands
  ... --host-spill ...` to resume with.  Before this, the out-of-memory
  error ended the run as a traceback, and a fresh process from the same
  checkpoint got through: measured 2026-11-12 on the RTX 5070 Ti, with five 222 MiB
  CUDA contexts of another program opening after the door had read
  15.24 GiB free, one T383 segment in eleven died that way on 0.1.7.
- The door names the card's other tenants and leaves them 512 MiB.
  When `nvidia-smi` shows other processes holding memory on the run's
  card, or the card drives a display, the plan is chosen against the
  free memory less 512 MiB, and the door's line lists the processes.
  A plan that fits only without the margin is still started; the margin
  never refuses a run.  On an empty headless card nothing changes (the
  T383 day still takes one band at 15.24 GiB free).  The figure: with
  512 MiB held by another process from before the start, the one-band
  T383 run's tightest moment left 150 MiB of the card free (2026-11-12,
  RTX 5070 Ti), so the margin is room for one more tenant that size, not
  for five; the fallback above covers more.
- A run whose pool limit the user set (`CUPY_GPU_MEMORY_LIMIT`) weighs
  the plan it falls back to against that limit as well as the card, says
  that it was the limit that refused the allocation when it was, and,
  when no plan fits under the limit, stops at once with exit 4 saying to
  raise or unset it.
- The allocator hook records every allocation the card refuses (the
  request, what the pool held and used, the card's free bytes) in the
  receipt's `device_memory` row, and tries the allocation once more
  after handing the pool's cached blocks back.

### Engine seam

- Two engine files are pinned for the carried launcher's compile route,
  `gpuwm/nvrtc_ptx_cache.py` and `gpuwm/kernel_compile_notice.py`: 50 seam
  files at gpuwm 2.9.13.  `tools/measure_boundary.py` reads 242 symbols
  across 72 modules with no gap.

## 0.1.9

0.1.8 was tagged but never reached PyPI: its publish workflow's test job
failed on GitHub's ubuntu runner, so nothing was uploaded. 0.1.9 is that
release with the failing gate corrected, and carries everything listed
under 0.1.8 below. Needs `gpuwm>=2.9.13,<2.10` and
`gpuwm-data>=2.9.13,<2.10`, as 0.1.8 did; proven against `gpuwm 2.9.13`.
The companion bundle carries ten Rust doors, unchanged from 0.1.7. No
forecast bit moves and no code path changes: every config hash is 0.1.8's,
and a run that names neither `[transforms]` switch is 0.1.8's run.

- The experimental `[transforms]` switch `divergence_only_mass_flux` agrees
  with the shipped step to rounding on a CPU, not byte for byte everywhere.
  Its ten-step numpy checkpoint is byte-identical to the shipped one on the
  hosts measured, but not on GitHub's hosted runners: the publish job's
  ubuntu runner and three of the four push-test jobs on 0.1.8's tree (one
  ubuntu, both Windows) failed it. There the BLAS returns different bits
  for the narrower Legendre call, 19 or 20 of the checkpoint's 52 arrays
  moved, and the float64 operator test found the divergence-only analysis
  unequal to the full one. `tests/test_arwen_global_transform_census.py` now holds
  that switch to what is guaranteed: every float array of the ten-step
  checkpoint within 1e-9 of its own largest value and every other array
  equal, and the operator within 1e-13 of the full analysis in float64
  (1e-6 in float32, as before). Emulated by moving every element of every
  divergence-only analysis by up to one ulp, the checkpoint moved the same
  20 arrays by at most 7.0e-11 of their largest value. `skip_unread_planes`
  keeps its byte-for-byte gate: it held on every CPU measured, GitHub's
  hosted runners included. The card measurements under 0.1.8 stand: the T255 day
  with both switches on wrote every array byte-identical on an RTX 5070 Ti
  and on an RTX 5090. `docs/ARWEN_GLOBAL_LEVEL5.md` and 0.1.8's notes say
  the same.

## 0.1.8

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.1.7 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 240
symbols across 71 modules with no gap, and all 48 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.1.7.  No forecast bit moves: a run that names none of the new
flags, tables and options writes 0.1.7's checkpoints array for array,
archive hash included, and the tapes, the stored checkpoints, the pinned
snapshot buffer and the T255 price change when and how bytes are written,
not what they are.  Every config hash is 0.1.7's except
`arwen_global_gdas_t383_native_24h`'s, whose radiation chunk moved; the T63
and T127 configs are new.  Everything marked experimental below is off
unless it is named.

### Measured together

- On one RTX 5070 Ti 16 GB with nothing else on it, the shipped configs
  with no edit, from the analyses they name, this tree on 2026-11-13 and
  0.1.7 on the same card on 2026-11-12 (wall from the receipt, card from
  nvidia-smi every 250 ms against the run's own process):
  - `arwen_global_gdas_t383_native_24h` to 3 h: the door chose one band
    with nothing parked at 15.24 GiB free; 1.832 s a step, 74.0 s, 14,072
    MiB of card, 11.50 GiB live, the pool held from 11.50 GiB and raised 7
    times to 13.34.  0.1.7's door ran it at two bands with every slice
    parked (2.27 GiB in host memory): 2.535 s a step, 152.2 s, 12,984 MiB.
  - `arwen_global_gdas_t255_native_sl_si_24h`, 24 h: one band both; 0.737
    s a step, 218.2 s, 9,150 MiB, 6.70 GiB live, the pool held from 6.70
    GiB and raised 4 times to 8.53, against 0.736 s, 238.7 s and 9,110
    MiB for 0.1.7.  The same day with `--tapes`: 0.741 s a step, 219.8 s,
    9,154 MiB; nine tapes, the worker's half 17.8 s and the synthesis on
    the model thread 0.9 s over the day; all nine checkpoints
    byte-identical to the day without the flag, archive hash included,
    and `SST` equal to `TSK` bit for bit on the tapes.
- On RTX 5090s, the shipped `arwen_global_gdas_t255_native_sl_si_24h` and
  `arwen_global_gdas_t383_native_sl_si_24h` to 3 h, this tree and 0.1.7:
  every array of the step-0 and step-36 checkpoints is byte-identical,
  archive hash included (43 and 133 arrays at each truncation).  The files
  differ because this tree stores their members: 1,452,926,392 bytes at
  T255 step 36 against 639,811,011 deflated, and 3,268,930,432 against
  1,437,283,140 at T383.  A T255 day with `--tapes` against the day
  without: all nine checkpoints byte-equal file for file, and all nine
  tapes byte-identical to `export` of the plain day's checkpoints, which
  took 87.0 s as its own process.  That day's receipt wall was 332.2 s
  against 254.4 s without the flag, its tape worker's half 154.6 s over
  nine tapes, on a host running other jobs at a load of 20 to 30: the
  worker shares the host's processors with everything else on them.

### Added

- `gpuwm-global run --tapes DIR [--tape-nlat N --tape-nlon M]` writes the
  wrfout tape of every checkpoint the run writes (step 0 of a cold start
  included; a restart's own starting checkpoint is the earlier run's) into
  `DIR`, from the run's own state, so a forecast no longer has to stop for
  an `export` process to have the card.  The tape is split in two halves
  that `export` now runs too (`wrfout_export.TapeGrid`): the six fields
  synthesized from the spectral state, made on the model thread one at a
  time and copied off the device, and everything else (the regrid, the
  render kernels, the frame, the netCDF write) on one worker thread beside
  the forecast, from the same host arrays the checkpoint of that time is
  written from.  Both are the same functions on the same values as
  `export` of that checkpoint.  A tape that fails prints `tape failed
  step=N: ...` and is listed in the receipt's `tapes` block, and the
  forecast goes on (its checkpoint is on disk for `export`); a run whose
  state is parked in the pinned host tier or spread over cards prints
  `tapes off: ...` and writes none.  Each tape is published under its
  final name only when whole.  The flags are outside the config identity.
  At most one tape is in flight: a tape slower than the forecast holds the
  model at the next output time, before that checkpoint is copied off the
  card, so a later checkpoint never overwrites the arrays a tape is still
  reading.  `tests/test_arwen_global_run_tapes.py` holds on the smoke
  configuration (CPU): every tape the run writes is byte-identical to
  `export` of the same checkpoint, also when every tape is slower than the
  forecast and the checkpoint writer reuses one host buffer; the
  checkpoints of a run with tapes are byte-identical to those of a run
  without; a restart with tapes reproduces the straight run's last
  checkpoint and tape byte for byte; and a failing tape leaves a finished
  forecast.
  Measured on the card on 2026-11-12: the shipped
  `arwen_global_gdas_t255_native_sl_si_24h` and
  `arwen_global_gdas_t383_native_sl_si_24h` days from their GDAS
  2026-09-01 00Z analysis, every run on an RTX 5090 32 GB (one run per
  card, the machine's processors shared with other jobs), checkpoints
  compared with `tools/arwen_global_checkpoint_identity.py` and tapes byte
  for byte, at the default 360 x 720 grid:
  - Without `--tapes`, every checkpoint of this tree is byte-identical to
    0.1.7's, archive hash included: nine at T255 and five at T383.  With
    `--tapes`, every checkpoint is byte-identical to the run without.
  - Every tape the run wrote, nine at T255 and five at T383, is
    byte-identical to `export` of the same checkpoints.  `export` as its
    own process took 191.8 s for the T255 day's nine and 152.9 s for the
    T383 day's five (the shell's wall around the command); the runs that
    wrote them took 4.0 and 28.0 s more than the run without at T255 and
    5.0 s more at T383 (the receipt's wall), so `--tapes` saved 164 to
    188 s of the T255 day and 148 s of the T383 day against
    run-then-export.
  - A run stopped at 12 h (`--until-s 43200`) and restarted from its step
    144 checkpoint, both halves with `--tapes`, writes the straight run's
    checkpoints and tapes byte for byte, at T255 and at T383.
  - The card: the pool's live peak is 6.698 GiB at T255 with and without
    tapes, and 12.305 GiB at T383 against 12.303 without.  The pool held
    to the door's figure (10.35 GiB at T255, 11.50 GiB at T383) was
    raised by the run as often with tapes as without: never at T255 and
    seven times at T383.  The card's maximum read by `nvidia-smi` every
    250 ms was 10,339 MiB with tapes and 10,363 without at T255, and
    15,303 and 15,449 at T383.
  - The time: the synthesis on the model thread took 1.7 to 2.0 s of a
    T255 day (about 0.2 s a tape) and 2.1 s of a T383 day; the worker's
    half, beside the forecast, 38 s and 27 s.  The receipt's step time
    was 0.644 s without tapes and 0.669 and 0.725 s in two runs with them
    at T255, and 1.328 s against 1.331 s at T383; the T255 spread is the
    shared machine's as much as the tapes', and 0.1.7's own run beside
    them took 0.652 s.
  - A T255 day whose step-72 tape was held 90 s and whose step-108 tape
    raised: the forecast waited for the held tape at the next output
    time and went on, the failure printed `tape failed step=108` and is
    the receipt's one row under `failures`, every checkpoint is
    byte-identical to the run without tapes, and the other eight tapes
    to those of the run that wrote all nine.  Its receipt wall was 404.6
    s against 340.2 and 364.2 s for the two runs with nothing held.
- Every render tape carries `SST` (K), the name a reader of WRF files asks
  for a sea-surface map by. It is the skin temperature, `TSK`, on every
  point: over open water and lakes the water temperature the model ran
  with, and over land and sea ice the land or ice skin, which is what WRF's
  real.exe writes there when its input carries no SST of its own (WPS's GFS
  table carries none). Mask with `XLAND` or `SEAICE` for open water. The
  export receipt states the rule under `sea_surface_temperature`, and
  `tests/test_arwen_global_tape_sst.py` holds `SST` to `TSK` and to the
  model's own skin. No forecast bit moves; the tape gains one variable.
  On the card, the 3 h tape of `arwen_global_gdas_t255_native_sl_si_24h`
  carries `SST` equal to `TSK` bit for bit, and its other 61 variables are
  byte-identical to 0.1.7's tape of the same checkpoint.
- An experimental `[transforms]` table with two switches, both off by
  default, that stop the semi-Lagrangian step paying for transforms it does
  not use: `skip_unread_planes` (no grid vorticity or divergence synthesis)
  and `divergence_only_mass_flux` (the mass-flux analyses contract only the
  divergence half). Each is carried in the config identity when it is on,
  and a config that does not name the table keeps its hash.
  `tools/arwen_global_transform_census.py` counts what a step makes: on the
  numpy T21 40-level census an ordinary step goes from 2,573 level-planes
  to 2,253 in the same 61 contractions, and
  `tests/test_arwen_global_transform_census.py` pins both counts. On the
  card, `arwen_global_gdas_t255_native_sl_si_24h` with both switches on
  wrote every array of its nine checkpoints byte-identical to the day
  without them, on an RTX 5070 Ti and on an RTX 5090 (133 of 133 arrays at
  each of steps 36 to 288; the config hash differs, by design). No speed
  is claimed for it: the day ran 0.728 s a step against 0.736 on the RTX
  5070 Ti with the card to itself, and 0.676 and 0.751 against 0.642, 0.646 and
  0.688 on RTX 5090s shared with other jobs.
- Two coarse grids of the forecast day of record ship as experiments,
  EXPERIMENTAL and not forecast configurations:
  `arwen_global_gdas_t63_native_sl_si_24h` (96 x 192, 208.5 km at the
  equator) and `arwen_global_gdas_t127_native_sl_si_24h` (192 x 384, 104.2
  km).  Each is `arwen_global_gdas_t255_native_sl_si_24h` with its
  truncation and `dx_m` changed and nothing else, which
  `tests/test_arwen_global_coarse_grids.py` holds by key.  Fifty-seven
  experiments now ship.  Measured with no flag from the GDAS 2026-09-01 00Z
  analysis on one RTX 5090 with no other job on the card, 2026-11-12, every
  gate passed: the T63 day 108.0 s (0.363 s a step), 2.50 GiB live and
  3,508 MiB on the card; the T127 day 171.8 s (0.514 s a step), 3.32 GiB
  and 5,180 MiB.  Building their statics took 121 s and 17.8 GB of host
  memory at T63 and 134 s and 18.6 GB at T127.  Nobody has scored either
  against an analysis.  `docs/arwen-global-coarse-grids.md` says what they
  are for and what they are not.
- `tools/arwen_global_scm_cases.py`, EXPERIMENTAL, runs the native
  physics suite, as a forecast calls it, on published single-column cases:
  DYCOMS-II RF01 (transcribed from Stevens et al. 2005) and BOMEX, RICO,
  FIRE, ARM SGP, TRMM-LBA and GABLS1 read from the DEPHY-SCM community
  files, which `fetch` downloads at a pinned commit and checks against
  their sha256.  It lays each case on the shipped 40-level stack, steps it
  with the forecast's 300 s Strang split, applies the case's large-scale
  forcing between the two physics half-steps and the case's surface
  through the suite's `modules` door, and ends every run on a table of its
  readings against the LES ranges the cases' papers give.  A case runs in
  seconds on a card; `--option KEY=VALUE` scores a scheme change.  What
  the shipped suite does on them, as read on 0.1.7: Grell-Freitas rains
  0.34 mm/day in BOMEX, a case that does not rain, and BOMEX makes no
  cloud; RICO keeps 1.0 g/m2 of cloud water in its last four hours against
  the LES's 4.8 to 47.7; and RRTMGP cools the DYCOMS-II RF01 boundary layer
  by 108 W/m2 where the case's formula, on the same column, gives 84.
  `docs/arwen-global-scm-cases.md` has the method and every reading; the
  CPU known answers are in `tests/test_arwen_global_scm_cases.py`.
- `tools/arwen_global_twin_scan.py`, EXPERIMENTAL, writes exact-twin
  starts and reads their differences: `perturb` plants one seeded
  potential-temperature pattern, the same at every truncation, in a step-0
  checkpoint the restart door admits; `bitflip` writes the one-bit twin;
  `measure` reads arm minus control in a mass-weighted dry energy norm, by
  degree band, with a plus and minus pair's nonlinearity and two arms'
  correlation.  On the T63 day, six hours on one RTX 5090: the
  zero-amplitude restart is byte-identical to the control, archive hash
  included (133 of 133 arrays); the one-bit twin reads 0.019 m/s against
  0.119 m/s for a 0.1 K pattern.  `docs/arwen-global-twin-scan.md`;
  `tests/test_arwen_global_twin_scan_tool.py`.
- Experimental: one forcing operator, `arwen_global.forcing`: before every
  step it pulls the chosen spectral fields toward a stored record of large
  scales, `x <- x + a L (x_rec - x)` with `a = 1 - exp(-dt/tau)` on the
  degrees the low-pass keeps (1 to 42 by default), and keeps the pull per
  field, level, degree and hour in `forcing_ledger.json` as a measurement
  of model error.  `gpuwm-global run --forcing RECORD[:options]` turns it
  on; `tau_h=inf` observes without pulling.  `--write-forcing-record`
  writes a run's own record; `tools/arwen_global_forcing_record.py` cuts
  one from a finished run's checkpoints; `tools/arwen_global_forcing_table.py`
  reads a forced and a free twin against their reference and prints a
  ledger's table.  Off by default, outside the config identity, recorded in
  the receipt and in the checkpoints' `forcing_history`.
  `docs/ARWEN_GLOBAL_FORCING.md` has the operator, its tests and the one
  twin pair measured with it.
- Experimental: forecast members outside the assimilation path
  (`arwen_global.members`, driver `tools/arwen_global_members.py`).  A
  member is one analysis plus a weighted sum of the filter's own
  perturbation draws, stepped through one shared model; no draw is the
  control and one draw of weight 1 is the filter's member of the same
  index, array for array.  Each member carries a recipe of a few
  kilobytes (config hash, base-state digest, options, draws and weights,
  conservation targets, digests at recorded steps, software stack and
  card), and `replay` rebuilds the member from it and refuses any digest
  that differs.  One member's failed step marks that member dead and the
  others go on.  `MemberSet.branch` starts an exact twin from a member
  checkpoint with one deliberate change.  `docs/arwen-global-members.md`
  has the interface, the recipe format and what was measured.
- `gpuwm-global run --restart PARENT --branch-from-other-config`: a run
  may resume from a checkpoint written under another config hash, an
  exact twin that differs from its parent by a config change, only with
  this flag, and its receipt records both hashes under `branch`.  Without
  the flag such a restart is refused, as before.
- Experimental: `tools/arwen_global_airliner_heights.py` reads the two altitudes every
  ADS-B transponder broadcasts, pressure altitude and GNSS height, as one
  observation: the geopotential height of the pressure the aircraft flies
  at.  `extract` streams one day of an open trace archive (adsb.lol's
  daily releases, ODbL) into a row file with every rejected point counted
  by cause, keeping no registration, owner or flight number; `score` runs
  the operator (ICAO standard-atmosphere inversion, EGM96 geoid, WGS84
  normal gravity to geopotential height) against hourly reference products
  and a longer-lead forecast through a numpy GRIB2 reader of its own, and
  against IGRA2 radiosondes, reads the layer-mean temperature of climbs and
  descents from the same reports, and writes the statistics and the
  coverage map.  Nothing is assimilated and the package is unchanged:
  `docs/arwen-global-airliner-heights.md` has the operator, the one day
  measured (2026-11-10) and its limits.
- Experimental: the scheme transplant, `python -m arwen_global.transplant`.
  The native runtime's cumulus call is split at the scheme's output
  (`_cumulus_result`, `_apply_cumulus_result`) and its Morrison call at the
  launch (`_microphysics_kernel`, `_book_microphysics`); the default path
  calls the halves in sequence, the arithmetic it ran before.  A donor run
  records one scheme's output and a receiver applies it in place of its
  own, cumulus by its rates and microphysics by copy-where-equal, with the
  receiver keeping its own books.  The driver runs a control, a perturbed
  twin, each with the other's scheme output, and a self-substitution twin
  that must equal the control at every report hour.
  `docs/arwen-global-transplant.md` has the seams, the experiment and one
  T63 case on one RTX 4090: the self-substitution twin equal to the
  control at every report hour, and the control's cumulus output given to
  the perturbed twin removing none of their difference over 48 h.
- Experimental: the composition replay (`arwen_global.composition`,
  `tools/arwen_global_replay_tracers.py`).  Extra transported fields (a
  constant, latitude and pressure tags, an origin map, point releases, a
  field from a file) ride a finished run's checkpoints, read for their wind
  and surface pressure only, through the model's departure search,
  quasi-monotone gather and Bermejo-Conde fixer, and land in sidecars keyed
  by the weather's `self_sha256` and the spec's hash.  The weather
  checkpoints are read only.  `tools/arwen_global_ozone_field.py` puts the
  GDAS analysis's ozone on the model's grid and levels and checks its
  column against the analysis's total ozone;
  `tools/arwen_global_ozone_change_score.py` scores a carried column's
  change against the analysis's; `tools/arwen_global_tracer_plume_chart.py`
  maps a row's burden.  `docs/arwen-global-composition-replay.md` has the
  rows, the route and what was measured: sixteen rows over a five-day T255
  run kept every six hours in 448 s on one RTX 4090, the 21 weather files
  unchanged by hash, and the carried ozone column's 24 h change at a
  pattern correlation of 0.77 with the analysis's.
- Experimental: `radiation_between` in the native suite's
  `native_adapter_options`, what the shortwave does between two radiation
  calls.  `"hold"` (the default) applies the call's planes unchanged, as
  before, adds no array and stays out of the identity.  `"surface"` is
  WRF's `swint_opt = 1` fit of the surface downward shortwave, evaluated at
  the sun of every physics call; `"column"` also scales the shortwave
  heating and the top-of-atmosphere shortwave by the cosine ratio.  Both
  join the config identity.  On one T127 case over 48 h on one RTX 4090,
  against a run that calls radiation every step, `"surface"` and
  `"column"` took the land 2 m temperature RMS from 0.306 K to 0.272 and
  0.275 K on the first day and from 0.453 K to 0.434 and 0.432 K on the
  second.  `docs/arwen-global-radiation-between.md` has the
  modes, the tests and the measurement.

### Changed

- A run's checkpoints no longer hold the card while one thread deflates
  them.  `run` and `cycle` write a checkpoint's members stored
  (`np.savez`) instead of deflated, read a stored checkpoint back member
  by member at its offset in the file and verify its hashes on four
  threads, and copy the state off the card into one page-locked host
  buffer kept for the run instead of a fresh pageable array per
  checkpoint.  A run also prices its memory once, from the card reading
  the door already took, where it used to probe the card three times
  (the receipt's `sizing_model_device_peak_bytes` is that one figure).
  The new `[output]` table chooses: `checkpoint_compression = "none"`
  (the default) or `"deflate"`, and `snapshot_buffer = "pinned"` (the
  default) or `"pageable"`.  Neither is in the config hash, and neither
  moves a byte of any array, a per-array SHA-256 or `self_sha256`: a
  stored checkpoint resumes a deflated lineage and the other way round,
  and 0.1.7 reads a stored checkpoint.  The other commands that write a
  checkpoint (assimilation, migration, the in-situ ledger) write it
  deflated, as before.
  Measured on one RTX 5090 on 2026-11-12, the shipped
  `arwen_global_gdas_t255_native_sl_si_24h` and
  `arwen_global_gdas_t383_native_sl_si_24h` configs with the output
  interval set to one hour, three hours from their GDAS 2026-09-01 00Z
  analysis (36 steps, checkpoints at steps 0, 12, 24 and 36), 0.1.7 and
  this tree alternated on the same card, three samples each, on a host
  running other jobs:
  - Every array of the four checkpoints of a cold start is byte-equal to
    0.1.7's at both truncations, with the defaults and with `"deflate"` /
    `"pageable"` (43 arrays at step 0, 133 after;
    `tools/arwen_global_checkpoint_identity.py`, `self_sha256` equal).
    A restart from the step-12 checkpoint ends at a step-36 checkpoint
    byte-equal to the uninterrupted 0.1.7 run's whichever tree wrote the
    step-12 checkpoint and whichever restarted from it.  0.1.7's `export`
    of its own step-36 checkpoint, this tree's `export` of its own, and
    0.1.7's `export` of this tree's stored one write byte-identical
    `wrfout` files.
  - A cold start to 3 h took 154.7, 176.4 and 156.4 s at T255 under
    0.1.7 and 68.1, 58.8 and 67.2 s with the defaults; 317.1, 351.9 and
    318.3 s at T383 against 115.6, 107.1 and 122.4 s.  With `"deflate"`
    and `"pageable"` it took 151.7 s and 327.9 s.  A restart from 1 h to
    3 h took 121.9, 118.4 and 124.6 s at T255 against 60.6, 61.8 and
    85.4 s, and 244.3, 259.8 and 251.8 s at T383 against 129.2, 104.6
    and 112.1 s.
  - The end of a run waited for its last checkpoint 37.1 to 53.2 s at
    T255 and 83.0 to 96.0 s at T383 under 0.1.7 (six runs each), and 4.6
    to 32.9 s and 11.9 to 26.5 s with the defaults.  The model thread
    spent 62.5 to 64.0 s (T255) and 136.8 to 166.7 s (T383) of a cold
    start at its three output times waiting for the previous deflate and
    copying the state off the card, and 1.7 to 1.8 s and 3.4 to 4.5 s
    with the defaults.  The step-0 checkpoint of a cold start took 5.2
    to 5.8 s and 11.8 to 12.1 s to write, and 2.4 to 4.7 s and 4.6 to
    5.4 s stored.  The two predictions a run no longer makes, each with
    its own card probe, took 2.4 to 3.0 s together.
  - `read_checkpoint` of the step-36 checkpoint on the host took 6.6 s
    (T255) and 18.6 s (T383) deflated under 0.1.7, and 0.8 s and 1.9 s
    stored with this tree; `export` of it took 36.6 s and 66.2 s against
    28.9 s and 45.0 s.
  - The card is untouched: the receipts' peak used device bytes are
    7,191,504,896 at T255 and 13,210,467,328 at T383 in all fifteen runs
    of each truncation, either tree, with the same predicted peak.
  The price is disk.  A stored checkpoint is its arrays' own size: 1,452.9
  MB at T255 against 626.6 to 639.8 MB deflated, 3,268.9 MB at T383
  against 1,404.0 to 1,437.3 MB (2.3 times); a cold start's step-0
  checkpoint 596.4 MB against 48.6 MB and 1,341.9 MB against 108.0 MB.  A
  run that keeps its checkpoints, such as a 90-day T383 run keeping one
  every six hours (about 1.18 TB stored, 0.52 TB deflated), sets
  `checkpoint_compression = "deflate"`.  The pinned buffer is one
  checkpoint's size of page-locked host memory for the life of the run.
  `tests/test_arwen_global_checkpoint_stored.py` holds both containers to
  one record and one set of arrays, refuses a flipped byte, a truncated
  file and a member whose size disagrees with its header, reads a member
  past 4 GiB at its zip64 offset, resumes a stored run from a stored and
  from a deflated midpoint to the same arrays, and holds a run to one
  memory prediction.
- The door prices the T255 semi-Lagrangian day
  (`arwen_global_gdas_t255_native_sl_si_24h`) at what its held runs took
  rather than at its model: 9,178 MiB of card where it read 11.09 GiB, so
  a 16 GB card with 9 to 11 GiB free runs it at one band instead of in
  latitude bands. The pool starts held to the day's 6.70 GiB live peak and
  rises as allocations need. MEASURED 2026-11-12 on an RTX 5070 Ti, the
  card to itself: two held days took 9,178 and 9,056 MiB at 0.736 s a step,
  against 9,110 MiB and 0.736 s for 0.1.7's day, and every array of all
  nine checkpoints was byte-identical to 0.1.7's. The row and the door's
  choice at the measured figure are held by
  `tests/test_arwen_global_completed_plans.py`.
- `arwen_global_gdas_t383_native_24h`, the T383 config of record on the
  default core, carries the 5,000-column radiation chunk the T383
  semi-Lagrangian config moved to in 0.1.7, where it still carried 12,500.
  On an empty 16 GB card the door now runs it at one band with nothing
  parked. MEASURED 2026-11-12 on an RTX 5070 Ti with nothing else on it,
  3 h from its GDAS 2026-08-30 18Z analysis: at 12,500 the door took two
  latitude bands with every slice parked, 2.535 s a step and 12,984 MiB of
  card; at 5,000 one band, 1.831 s a step and 14,254 MiB, and the 24 h day
  1.907 s a step and 14,316 MiB, under the 14,608 MiB the door prices the
  plan at. Every array of the step-0 and step-36 checkpoints is
  byte-identical at the two chunks (133 of 133 at step 36), and on an RTX
  5090, where both chunks run at one band, every array of all nine
  checkpoints of the 24 h day is (1.578 s a step at 12,500 and 1.590 at
  5,000 there, with other jobs on the machine's other cards).
  The chunk is in the config hash (`6f32b3255085ce4e` where it was
  `d00cc42c95d1e452`), so a checkpoint this config wrote under 0.1.7 is
  refused on restart; restart it under a copy that says
  `radiation_column_chunk = 12500`. Measured on an RTX 5090: 0.1.7's step
  36 of this day is refused under the new config ("checkpoint config
  identity does not match this run"), and under a copy that says 12500
  this tree continues it to step 72 byte-identical to 0.1.7's straight day,
  archive hash included.

### Stepping options

No default moved: a config that sets none of the keys below runs 0.1.7's
forecast under 0.1.7's config hash.  Ten steps of three shipped configs
with no edit (`arwen_global_gdas_t255_native_sl_si_24h` at 300 s,
`arwen_global_gdas_t255_native_sl_si_24h_dt600` at 600 s and
`arwen_global_gdas_t255_native_imex_24h` at 90 s), this tree and 0.1.7 on
the same card model (RTX 5090), give byte-identical step-0 and step-10
checkpoints, archive hash included (`tools/arwen_global_checkpoint_identity.py`:
43 and 133 arrays on the semi-Lagrangian configs, 43 and 126 on the
Eulerian one).  The 600 s config with `[insitu] spectra_every_s = 3600.0`
and a `stratospheric_noise_limit` set is byte-identical to the same
config without them at step 10, under the same config hash.

- `[semilag] extrapolation = "settls_departure"` (experimental, off by
  default) reads the time extrapolation at the departure point, Hortal's
  form of SETTLS, where the shipped `"settls"` reads it at the arrival
  point.  `[semilag] level_rate_limiter = true` (experimental, off by
  default) keeps the time-n vertical rate wherever it changed sign between
  the two time levels.  Both enter the config hash when set; the departure
  form runs on the one-band resident step only and the banded step
  refuses it by name.  Measured with both keys set against the shipped
  scheme on `arwen_global_gdas_t255_native_sl_si_24h_dt600` run 48 h from
  four GDAS analyses (2026-01-15, 04-15, 07-15 and 10-15 00Z), scored by
  `python -m arwen_global.upper_air_scorecard` against the verifying GDAS
  analyses, on an RTX 5090.  Global four-case means, shipped then
  departure form:
  - 48 h: 500 hPa height RMSE 17.40 to 17.12 m, 850 hPa temperature RMSE
    1.582 to 1.576 K, 250 hPa vector wind RMSE 7.35 to 7.29 m/s, 850 hPa
    4.59 to 4.56 m/s, 700 hPa relative humidity RMSE 19.20 to 19.12
    percent, 500 hPa anomaly correlation 0.9816 to 0.9821; no case is
    worse on any of them except one of four on the 250 hPa wind.
  - 24 h: the means are equal or better except 850 hPa temperature RMSE,
    1.2236 to 1.2252 K, higher in all four cases; 500 hPa height RMSE is
    lower in the mean (9.298 to 9.280 m) and higher in three of the four
    cases.
  - The tropics at 48 h are slightly worse: 250 hPa vector wind RMSE 6.635
    to 6.662 m/s (all four cases), 700 hPa relative humidity RMSE 15.84 to
    15.90 percent, 500 hPa anomaly correlation 0.9378 to 0.9370.
  - The stratospheric reading below, largest top-decile share of kinetic
    energy between 10 and 100 hPa over the 48 h, is lower in all four
    cases: 2.42e-4 to 2.01e-4, 3.01e-4 to 2.55e-4, 1.09e-4 to 1.01e-4 and
    2.03e-4 to 1.45e-4.
  The form alone, without the limiter, was not scored, and nothing was
  measured at any other step or truncation.
  `tests/test_arwen_global_semilag_settls_departure.py` holds both keys to
  their contracts on the CPU.
- `[insitu] spectra_every_s`, `flush_every_s` and `energy_every_s` give the
  in-situ ledger's cadences in seconds of forecast time instead of steps,
  so a run at a longer step samples the same forecast times.  Each must be
  a whole number of steps or the config is refused at load, and a cadence
  given both ways is refused.  The 48 h runs above took 48 spectra samples
  each at `spectra_every_s = 3600.0` and a 600 s step.
- `[insitu] stratospheric_noise_limit` (experimental, off unless set)
  reports the run's largest top-decile share of kinetic energy on the
  levels between 10 and 100 hPa (`stratospheric_noise_band_hpa`), read from
  the spectra the ledger already samples, in the receipt's in-situ block
  and never in the run's status.  No limit has been calibrated for it.
  `tools/arwen_global_strat_noise_replay.py` replays written ledgers
  through the same code.  Nothing in `[insitu]` enters the config hash;
  `tests/test_arwen_global_insitu_cadence_seconds.py` holds the cadences and
  the reading to their contracts on the CPU.

## 0.1.7

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.1.6 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 240
symbols across 71 modules with no gap, and all 48 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.1.6.  No forecast bit moves: the radiation driver and the physics
suite's bookkeeping do the same arithmetic on fewer launches, fewer
columns and fewer transfers, and the sizer, the device-memory hook, the
pool the run and cycle commands hold, the radiation's cloud finalize and
the cumulus scheme's packing change memory only.  Every config hash is
0.1.6's except the two T383 semi-Lagrangian configs', whose radiation
chunk moved.

### Measured together

- The shipped configs with no edit, 24 h from their GDAS 2026-09-01 00Z
  analysis, on one RTX 5070 Ti 16 GB with nothing else on it, this tree
  and 0.1.6 back to back on 2026-11-12 (wall from the receipt, card from
  nvidia-smi every 250 ms against the run's own process):
  - `arwen_global_gdas_t383_native_sl_si_24h`: the door chose one band
    with nothing parked at 15.24 GiB free; 1.805 s a step, 578.5 s, 14,392
    MiB of card, 11.94 GiB live, the pool held from 11.50 GiB and raised
    7 times to 13.66.  0.1.6's door ran its 12,500-column copy at two
    bands with every slice parked (2.27 GiB in host memory): 2.981 s a
    step, 918.7 s, 15,142 MiB, 12.00 GiB live.
  - `arwen_global_gdas_t255_native_sl_si_24h`: one band both; 0.747 s a
    step, 242.4 s, 9,028 MiB, 6.70 GiB live (the pool never raised),
    against 0.914 s, 289.9 s, 11,582 MiB and 8.63 GiB.
  Every array of the nine T255 checkpoints is byte-identical to 0.1.6's,
  archive hash included (1,107 arrays).  The T383 days differ in their
  chunk and so in their config hash, and in nothing else: 575 of 575
  arrays over five checkpoints are byte-identical.  A 3 h T383 run is
  byte-identical, archive hash included, to 0.1.6 run at the 5,000-column
  chunk (133 of 133 at step 36).  Stopped at step 18 and restarted, both
  truncations end at step 36 identical to the straight run, archive hash
  included.  Four one-band T383 runs of this tree took 13,706 to 14,518
  MiB of the card, under the 14,608 MiB the door prices the plan at.
- A 0.1.6 checkpoint continued by this tree, same card, same day: the T255
  day's step 36 ends at step 72 identical to this tree's straight day,
  archive hash included.  The T383 day's step 72 is refused under the
  shipped config ("checkpoint config identity does not match this run",
  the chunk moved); under a copy that says `radiation_column_chunk =
  12500` the door took two bands with every slice parked and the run
  ended at step 144 with 133 of 133 arrays byte-identical to this tree's
  own day.

### Changed

- A radiation call is faster at the same bits.  The two RTE solvers launch
  over column spans of about SMs x 1024 / 32 columns (2,240 on a 70-SM
  card), so their g-point tile lands on 32 and they take the folding path
  whatever `radiation_column_chunk` is; the chunk had set the tile, and at
  12,500 and 5,000 columns it put both solvers on the partial-writing path.
  The shortwave loop runs the sunlit columns only; a dark column's
  shortwave was solved with the sun at the zenith and then zeroed, and
  keeps the zero it was given.  Measured on the RTX 5070 Ti with the
  profiler's device time, two calls a run, each run against the 0.1.6
  source on the same card in the same sitting:
  - T255, `radiation_column_chunk = 12500`, GDAS 2026-11-11 00Z, 21 steps:
    2.260 and 2.265 s a call before, 0.965 and 0.962 s after (the spans
    alone 1.327, the sunlit columns alone 1.661 and 1.701).  A step that
    fires radiation went from 2.96 to 1.67 s; an ordinary step did not
    move (700 against 695 ms, median of 18).
  - T383, `radiation_column_chunk = 5000` (the chunk the shipped T383
    semi-Lagrangian configs now carry), GDAS 2026-11-10 18Z, 15 steps:
    4.274 and 4.332 s a call before, 2.169 and 2.169 s after; a
    radiation step 5.89 to 3.75 s.  The run's own card
    reading peaked at 14,996 MiB against 15,054 before, under the same
    CuPy pool cap.
  Every one of the 134 arrays of the step-21 (T255) and step-15 (T383)
  checkpoints is byte-equal to the 0.1.6 run's, for both changes together
  and each alone, and a run restarted at step 11 (T255) or step 8 (T383)
  ends byte-equal to the straight run.  `GPUWM_RTE_COLUMN_SPAN` (columns)
  and `GPUWM_RRTMGP_DARK_COLUMNS=solve` are measurement overrides that put
  either part back, so the two can be interleaved in one session.
  `tests/test_rrtmgp_column_spans.py` holds the spans and the sunlit chunks
  to known answers on the CPU, and `tests/test_rrtmgp_column_spans_cuda.py`
  holds both solvers to byte equality against one launch on the card,
  over one column, either side of a span, the shipped chunks and forced
  odd widths.
- The native physics suite spends less of each call outside its schemes.
  The in-situ ledger's component capture marks each field with one kernel
  pass (the difference, the float64 zonal means, the max-abs and the held
  copy in one read) where it ran about ten launches and three and a half
  volume passes per field per mark; a call takes five column-water
  integrals where it took eight, because three were readings of a state
  another booking had just read; the layer thicknesses are built once per
  call, not five times; the closure's three readings cross to the host in
  one transfer; and the batch flips and converts each water species in
  one pass each way.  MEASURED on the RTX 5070 Ti, 2026-11-12, the shipped
  T255 native semi-Lagrangian config from its GDAS 2026-09-01 00Z
  analysis, 13 steps with ten profiled after two warm-up steps
  (`--profile-steps 10`), this change alone against 0.1.6 interleaved
  twice in one sitting: the median step 678.2 and 672.0 ms of device time
  before, 631.6 and 631.0 after; the first physics call 171.0 and 170.4 ms
  against 150.0 and 149.3, the second 160.9 and 160.5 against 138.8 and
  138.7; 12.98 MB in 40.1 transfers a step to the host against 5.90 MB in
  38.1.  The step-13 checkpoints are byte-identical, archive hash included.
  `tests/test_arwen_global_native_shared_readings.py` runs two suite calls
  of the moist smoke configuration and compares every array and
  diagnostic, `array_equal`, against the same calls with every reading
  taken fresh as 0.1.6 took it.  The ledger's capture means are float64
  sums in another order, so they can move in their last bits; the
  max-abs readings cannot.  `tests/test_arwen_global_insitu_fused_capture.py`
  (on a card) holds the fused mark to the chain it replaced, max-abs equal
  and means within a relative 1e-12, at the T21 and T255 shapes,
  whole-grid and in bands.

### 16 GB cards

- T383 (35 km) runs on a 16 GB card that also carries a desktop.  The door
  priced two and four bands with every slice parked at 18.31 GiB of card,
  borrowing the T533 two-band pool ratio, and refused every T383 plan below
  14.4 GiB free.  It now prices those plans at what their own runs took,
  so with less free than the one-band run needs (14.27 GiB at the
  5,000-column radiation chunk, below) it chooses two bands from 10.83 GiB
  free, four from 10.33 and eight from 10.23, with no flag set.  MEASURED
  on an RTX 5070 Ti, 24 h forecasts from the GDAS 2026-11-11 00Z analysis,
  before the pool was held (below):
  two bands 12,486 MiB of card and 13.7 min a forecast day (2.863 s a
  step), four bands 10,582 MiB and 14.1 min (2.933 s), against 15,624 MiB
  and 10.7 min (2.229 s) for one band with nothing parked.  Both
  reproduce the one-band day's checkpoint byte for byte, 133 of 133
  arrays at step 288.  The banded plans hold 2.27 GiB in pinned host
  memory and peak about 2 GiB higher in host RAM (7.4 GiB against 5.3).
- The door's line says what the plan it chose costs: what one band with
  nothing parked would need, and the measured step against the one-band
  step.  A refusal names the measured plan that fits the card, with its
  flags when the config pinned the band count or the host tier
  (`--latitude-bands 2`, `--host-spill on`), the allocator
  (`--device-allocator async`: four bands, 9,772 MiB, a 24 h day) or the
  radiation chunk (`radiation_column_chunk = 5000`, for a T383 day at
  12,500); below every measured plan it says how much more has to
  be free.
- MEASURED from the door with no memory flag, 24 h T383 days at the
  5,000-column chunk with another process holding part of the card, on
  the door's figures before the pool was held: 2,270 MiB held, the door
  read 13.02 GiB free and took two bands (12,486 MiB, 2.863 s a step);
  4,318 MiB held, it read 11.02 GiB free and took four bands (11,342 MiB,
  the pool growing into the room it had, 2.941 s a step).  Every
  conservation gate of both receipts passed, and step 288 of both is
  byte-identical to the one-band day.  At the prices below the door takes
  two bands at 11.02 GiB free.

### The full-speed T383 plan takes less of the card

- A run holds its CuPy pool to a limit from before its first device byte,
  so the pool no longer grows into whatever the card has free.  The limit
  is the pool figure of the plan the door chose: for a plan a completed
  run of exactly itself measured under a limit, the limit that run
  started from; otherwise the door's card figure less the out-of-pool
  term.  When an allocation finds no room under the limit the run raises
  it by what the allocation needs and 32 MiB, and the receipt's
  `device_memory.pool_limit` row counts the raises.  A
  `CUPY_GPU_MEMORY_LIMIT` set in the environment stands instead, and the
  run sets no limit of its own; so does a run under the `async` or `slab`
  allocator.  The door's line says what the pool is held to.
- MEASURED on an RTX 5070 Ti, the 24 h T383 day from the GDAS 2026-11-11
  00Z analysis at one band with nothing parked and the 5,000-column
  radiation chunk, held from 11.50 GiB: two days took 13,880 and 14,608
  MiB of card (the limit raised to 13.15 and 13.86 GiB, the last time at
  steps 17 and 135) against 15,624 MiB for the day of the tree before
  (15,056 with `CUPY_GPU_MEMORY_LIMIT` at 14.3 GiB).  Where a held pool ends is set by how its blocks were cut:
  every raise after the first steps is the semi-Lagrangian gather's 810
  MiB stack finding no whole free block while 9.33 GiB is live, and
  nine held 3 h runs ended at 13,642 to 14,384 MiB.  The door prices
  the plan at the heavier day, 14,608 MiB, so it asks 14.27 GiB free
  for it against 14.93.  Every array of all eight checkpoints of both
  days is byte-identical to the day before the change (1,064 of 1,064
  each).  The held days took 2.220 and 2.250 s a step against 2.235 for
  the tree before the same day on the same card (the 2.250 day ran with
  as little as 1.6 GiB of host RAM free); the 3 h runs 2.279 to 2.298
  held against 2.287 to 2.296 for the tree before, unheld.  Four more
  days, the tree before and this one alternated on the same card within
  the hour from the door with no flag (the tree before under
  `CUPY_GPU_MEMORY_LIMIT` at 14.3 GiB to keep it inside its share of the
  card; its unheld day of record ran 2.2287 s a step):
  this tree 14,272 and 14,204 MiB at 2.2195 and 2.2186 s a step, the tree
  before 14,960 and 14,898 MiB at 2.2349 and 2.2290 s, all nine
  checkpoints of every day byte-identical to the one-band day of record.
- The radiation's cloud finalize writes over the gas cubes it read and
  drops the longwave Planck sources at their last reader, and the cumulus
  scheme packs four tiles of columns a pass where it packed its whole
  131,072-column chunk.  Memory only: the live peak of a T383 step at the
  5,000-column chunk is 11.51 GiB over 3 h against 12.10 (the 24 h day:
  11.95 against 12.10), byte-identical.
  `tests/test_arwen_global_radiation_inplace_cuda.py` (on a card) holds
  the finalize to the bit.
- Two bands with every slice parked, re-measured held to a limit from
  9.25 GiB: the 24 h day 11,090 MiB of card (raised 8 times to 10.43 GiB)
  and 2.852 s a step, 3 h 10,946 MiB, against 12,058 to 12,486 MiB for
  the tree before; all eight checkpoints of the day byte-identical to
  the one-band day.  The door prices the plan at the day, so it takes two
  bands from 10.83 GiB free, and four from 10.33.
- The shipped T383 semi-Lagrangian configs
  (`arwen_global_gdas_t383_native_sl_si_24h` and `..._sl_si_wall`) carry
  the 5,000-column radiation chunk where they carried 12,500, as the T533
  configs already did, so the shipped day takes the full-speed plan on an
  empty 16 GB card with no edit; at 12,500 the door sent it to a banded
  plan there.  MEASURED on an RTX 5070 Ti, held runs of 3 h of the T383
  day from the GDAS 2026-11-11 00Z analysis, before the column spans
  above took the chunk out of the solvers' speed: 2,500 columns 11.51 GiB live
  and 2.277 s a step; 5,000 columns 11.51 GiB and 2.280 to 2.292 s (three
  runs); 7,500 columns 11.68 GiB and 2.331 s; 12,500 columns 12.49 GiB,
  2.421 to 2.448 s (three runs) and 14,532 to 14,952 MiB of card against
  13,642 to 13,870 for the two 5,000-column runs beside them.  Every
  checkpoint array was byte-identical at every chunk (133 of 133 at step
  36).  The chunk is in the config hash, so the two configs' hashes move:
  a checkpoint written under 12,500 restarts under a config that says
  `radiation_column_chunk = 12500`.  The other T383 configs keep 12,500:
  the Eulerian ones (the chunk was measured on the semi-Lagrangian core
  only) and `arwen_global_gdas_t383_native_24h`, which runs the
  semi-Lagrangian core by default and which the door, as in 0.1.6, sends
  to two bands with every slice parked on an empty 16 GB card.
- `tests/test_arwen_global_pool_limit.py` holds the limit's rules on the
  CPU: whose limit stands, that the run's own limit yields instead of
  ending a forecast, that a card really out of memory is not papered
  over, and that the door prices the one-band T383 plan at the held run.

### A 12 GB card

- MEASURED on an RTX 5070 Ti standing in for a 12 GB card (another
  process holding all but what such a card leaves free), 24 h T383 days
  from the GDAS 2026-11-11 00Z analysis from the door with no flag: at
  11.59 and 11.05 GiB free it chose two bands with every slice parked,
  10,880 and 10,972 MiB of card at 2.858 and 2.871 s a step; at 10.52
  GiB free it chose four, 10,322 MiB at 2.961 s.  Every array of all nine
  checkpoints of each day is byte-identical to the one-band day.  The
  step is that card's; a 12 GB card's own step was not measured.
- The four-band plan under `--device-allocator async` is priced at its
  24 h day, 9,772 MiB of card at 3.087 s a step; a 3 h run of it read
  9,196, and the driver's pool, which takes no limit, grew past that over
  the day.  The same day on this tree read 9,676 MiB at 3.098 s, all nine
  checkpoints byte-identical to the one-band day.  The refusal below every
  default-pool plan names it at 9.54 GiB of card.
- Under `--device-allocator async` the receipt's `device_memory.pool_limit`
  row says the pool carries no limit: the driver's pool reads 2**64 - 1
  from `get_limit()` when nothing limited it, and that reading is no
  limit.

## 0.1.6

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.1.5 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 240
symbols across 71 modules with no gap, and all 48 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.1.5.  The model changed only where an analysis is read: the cold
start reads the unit of its levels from the mapping (every shipped mapping
declares Pa, which it leaves as it is) and an ERA5 frame is completed
before it; the GDAS path keeps its rules.

### Added

- The global model cold-starts from ERA5.  `gpuwm-global fetch-analysis
  --source era5 --cycle YYYY-MM-DDTHH` fetches one whole-globe ERA5 hour
  (any hour from 1940 to about six days ago) from Google's public ARCO
  store with no key, through the engine's own ARCO retrieval, plus ERA5's
  snow density beside it, and prints the `[initial]` table that starts any
  analysis configuration from it, through this package's own NetCDF
  mapping (`era5-arco-global`, in `data/package-mappings/`, which the re-cut
  does not own).  No ERA5 experiment ships.  The decode is
  the engine's; `arwen_global.era5_analysis` completes what ERA5 lacks
  against GDAS by stated rules the receipt records under `era5_completion`:
  soil masked over water; sea ice 0 on land, and on a coast point ERA5's sea
  fields leave out the mean of the sea values beside it; ERA5's SST, not its
  tile-mean skin, as the water temperature of an ice-free water point;
  sea-ice thickness 3 m (WRF's default; ERA5 has none), on a frozen lake
  too; snow depth from ERA5's snow density; snow averaged over land points
  only, so the pack starts bare as ERA5 has it; ERA5's snow not capped.
  The GDAS archive the engine reads begins at 2021-03-22 12Z; this is the
  input for anything earlier.  `docs/ARWEN_GLOBAL_ERA5.md` has the field
  table, hour 0 checked against ERA5 read a second way, and four measured
  starts: T255 for 24 h under the reference physics and under the native
  suite, T383 for 24 h under the native suite, and T255 for five days from
  a January hour.
- `tools/arwen_global_era5_check.py` is the check the page cites: `read`
  reads one ERA5 hour straight from the ARCO store with zarr-python,
  `state` writes the hour-0 state of an ERA5 configuration from a CPU cold
  start, `hour0` compares the two with code that imports neither this
  package nor the engine, and `tapes` compares render tapes with ERA5 of
  their valid times by the start tape's surface classes.

### Fixed

- The cold start reads the analysis's pressure levels in the unit its
  mapping declares (`coordinates.vertical.units`, Pa or hPa) and refuses
  any other unit, or a ladder whose deepest level reads above 500 hPa.  It
  took them as Pa unread, so an hPa source would have put every model level
  below a "1000 Pa" bottom level with no refusal.  Every shipped mapping
  until now declared Pa, so no earlier run is changed.
- A NetCDF mapping's soil layers reach Noah at their depths, from the
  mapping's row in `data/soil-level-depths.json`; they used to be refused
  for want of a GRIB level type.

### Known limits

- ERA5 carries no sea-ice thickness: every sea-ice point starts at 3 m,
  WRF's default, which is too thick for first-year Antarctic ice, and a
  frozen lake ERA5 carries in its sea-ice cover gets the same 3 m.
- ERA5's snow is not capped.  Land the statics call tundra where ERA5 has
  glacier starts with metres of snow water; its skin ran 4.9 K cold at
  24 h in May and 2.8 to 6.7 K warm over five January days, rising every
  day from 48 h to 120 h, and nothing longer than five days was run, so
  whether it levels off is not known.  Open water at the ice edge keeps
  ERA5's skin: on the January start 15 columns began more than 5 K from
  ERA5's SST.
- The T383 start from ERA5 tripped the in-situ wire
  `spectral_top_decile_growth` six times between hours 5 and 9, recorded
  and not refused, and its receipt passes; the T255 ERA5 starts and the
  GDAS T383 starts on record tripped none.
- Measured: two dates, T255 and T383, up to five days.  Nothing was
  compared with a GDAS start of the same valid time, because the GDAS route
  begins at 2021-03-22 12Z.  The statics are present-day whatever the date.

## 0.1.5

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`, as 0.1.4 did;
proven against `gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 237
symbols across 69 modules with no gap, and all 48 engine seam files are
pinned at 2.9.13.  The companion bundle carries ten Rust doors, unchanged
from 0.1.4.  No model source changed: the source files that differ from
0.1.4 are the two exporters and the command line, so a forecast, its
checkpoints and its config hashes are 0.1.4's.

### Added

- `gpuwm-global export-mapped-parent` writes a forecast as the regional
  engine's mapped NetCDF source: one model-level frame per checkpoint on a
  regular lat-lon window, with the mapping, composition and provenance
  documents and a receipt that carries the `gpuwm prep --source mapped`
  command for them.  The regional engine then builds its initial state and
  its lateral boundary file from the global forecast with no change on its
  side.  The frames carry earth winds at mass points from the model's own
  wind synthesis, interface vertical velocity, the six water species as dry
  mixing ratios, and the skin, soil, snow and ice state; number
  concentrations, deep soil temperature and static geography are not
  carried.  The whole series is checked from the checkpoint headers before
  the first frame is written (time order, frames that already exist, parent
  frames in the folder that belong to another export), and every path in
  the receipt is absolute.  `W` is the air's velocity through the model's
  coordinate surfaces, zero at the top and at the ground, and is labelled
  so.
  `docs/ARWEN_GLOBAL_REGIONAL_PARENT.md` has the file contract, the two
  regional settings that keep the global forecast the only source of the
  run's state (`lake_state_source = "forcing"`, `sf_ocean_physics = 0`),
  and one measured run: a T383 forecast feeding a 3 km regional run for
  48 h on one 16 GB card, with the boundary row within 0.2 K RMS of the
  parent at every update while the interior departs from it.  The page
  also says what that run did not get right: the regional engine puts
  about a third of the parent's cloud and precipitation on the boundary
  row with frames 3 h apart, precipitation within about 50 points of the
  outflow edge is not a forecast, and the regional run's deep soil
  temperature came from its geography tree's climatology, 2.4 K warmer on
  average than the global model's own.
- `tools/arwen_global_regional_parent_check.py` measures whether a regional
  run is fed by its parent: a regional history file against a parent frame
  by distance from the lateral edge, and a fed run against a control whose
  boundary was held still.  No device.
- `gpuwm-global export` writes eight more variables on every render tape,
  the ones a reader of WRF history files asks for by name when it derives
  its own fields: `F` and `E` (the Coriolis terms at the rotation rate the
  model integrates with), `MAPFAC_MX` and `MAPFAC_MY` (the map factors of
  the lat-lon grid, 1 / cos(latitude) and the ratio of the column spacing
  to the row spacing), `LU_INDEX`, `IVGTYP` and `XLAND` (the land-use
  category and the land/water flag of the nearest model column, never
  blended), and `W`.  Without the map factors a reader differentiates a
  lat-lon tape as if it were unstretched, and without the land-use planes
  it reads every column as land.
  `W` is the air's vertical velocity relative to the ground in two of its
  three parts: the flux through the model's coordinate surfaces, which
  alone is the regional parent frames' `W`, plus the wind's climb along
  those surfaces, so at the ground it is the lowest layer's wind times the
  terrain slope.  The rise or fall of the surfaces themselves at a fixed
  place is left out, because one checkpoint does not determine it; it is
  zero at the ground.  On one 576 x 1152 tape of a T383 forecast the climb
  is 0.016 to 0.030 m/s RMS from the ground to 260 hPa, where the flux
  alone correlates with the tape's `W` at 0.82; the tape's `W` agrees with
  an estimate differenced from the tape's own heights and winds to
  0.0012 m/s RMS (correlation 0.9993); and the part left out is under 4 %
  of `W` up to 100 hPa and as large as `W` near 1 hPa.  The variable's
  `description` says what it carries in place of WRF's "z-wind component",
  the tape's `ARWEN_VERTICAL_VELOCITY` attribute carries the definition,
  and a tape that would be published without that description is not
  written.  The tape also names the land-use table the categories index
  (`MMINLU`, `ISWATER`, `ISICE`, `ISURBAN`) when the checkpoint names it,
  with `ISLAKE = -1`, WRF's value for a field with no separate lake class,
  because the model folds lakes into the water category.  The export
  receipt lists the eight under `derivation_fields`.  Every variable a
  0.1.4 tape carried is unchanged bit for bit.  `docs/ARWEN_GLOBAL.md` has
  the table and the measurements.

### Known limits

- The regional feed has one measured case, and it was not verified against
  observations.  Frames of the whole globe have been written only at test
  size, and a window across the date line is refused.
- `W` on a render tape is not the whole velocity relative to the ground
  above about 10 hPa, where the part it leaves out is as large as what it
  carries.  That part needs the state's tendency, which a checkpoint does
  not hold.
- The render catalogue the run plan reads was not measured again.  It is
  the 2026-09-10 measurement of a tape without `W`, and the nine products
  `gpuwm render` offers on a tape that carries `W` have not been drawn and
  checked on a global tape.

## 0.1.4

Needs `gpuwm>=2.9.13,<2.10` and `gpuwm-data>=2.9.13,<2.10`; proven against
`gpuwm 2.9.13`.  `tools/measure_boundary.py` reads 237 symbols across 69
modules with no gap, and all 48 engine seam files are pinned at 2.9.13.
The companion bundle carries ten Rust doors, unchanged from 0.1.3.

### Fixed

- Surface pressure holds in float32.  The dry 9-day Jablonowski-Williamson
  steady state at T255 drifted 1.506 hPa in 0.1.3 and now drifts 0.280 hPa
  (float64: 0.251).  The semi-Lagrangian step advects ln ps as an anomaly
  about a reference value, the gather sums its taps about the nearest
  stencil node, the theta analysis is taken about the reference column, and
  the mass fixer's surface-pressure plane and target are float64.  Nothing
  else moved to float64.  In a real 10-day T383 forecast the global-mean
  surface pressure holds to 0.001 hPa.
- The top model level no longer cools under the physics.  From GDAS
  2026-10-15 00Z the global-mean potential temperature of the top level
  (1.0-1.44 hPa) moved -6.07 K a day against the analysis over five days
  in 0.1.3 at T255 (-9.06 on day 1); it now moves +0.43 K a day at T255
  and +0.40 at T383, and over a 10-day T383 forecast the top level warms
  0.06 K a day.  The causes, each measured on a per-process budget of the
  top levels (`tools/arwen_global_top_budget.py`):
  - the air above the 1 hPa lid is radiated on log-spaced cap layers with
    a standard-atmosphere mesosphere, not one isothermal layer;
  - the radiation reads the CAM zonal-monthly ozone climatology WRF and
    MPAS-A ship, by date, latitude and pressure, instead of one global
    median profile (`ozone = "rfmip-median"` keeps the old profile);
  - the kinetic energy the top absorber removes is booked as heat;
  - a zonal-mean Rayleigh drag in the top levels is on by default in
    native-physics forecasts (5-day rate at the lid, cos^2 ramp to 5 hPa,
    as MPAS-A damps its top); `[sponge] mean_drag_base_pa = 0` turns it
    off.  The idealized cases (Jablonowski-Williamson, Held-Suarez) stay
    unforced, so their config hashes do not change.
- SST, sea ice and greenness are no longer frozen at their cold-start
  values.  `gpuwm-global surface-series` builds a dated series on the
  model grid from GDAS analyses (`gdas-analysis`) or from one GFS cycle's
  own forecast (`gfs-forecast`), decoded and regridded by the Rust doors
  the cold start uses; `run --surface-series` (or `[surface_update]
  series`) applies it: SST every step, sea ice at each record, greenness
  once a day.  Sources are rows of `surface_series_sources.json`.
- `--streaming on` no longer rebuilds the Legendre tables on the host on
  every transform call: they come from a pinned host cache built once
  (T533 sat more than 12 minutes before its first step).
- The host tier stages a band straight out of its pinned slot with one
  2-D async copy instead of a single-thread pageable gather on the main
  thread: 18.9 to 6.67 s a step at T533 on a 16 GB card, the T255
  checkpoint 134 of 134 arrays bit-identical.
- The CPU test tier finishes under pytest-xdist: BLAS and OpenMP threads
  are capped at cores per worker.

### Changed

- Runs on the gpuwm 2.9 engine.  2.9 compiles CuPy kernels in fast
  numerics by default; the fused-transport bitwise checks are now strict
  numerics checks (`GPUWM_NUMERICS=strict`).
- The CAM ozone default moves the config identity of every native-physics
  experiment, so their config hashes change in this release.

### 16 GB cards

- T383 (35 km) runs resident on a 16 GB card: one band, nothing parked on
  the host.  A 10-day forecast took 108.9 min (2.24 s a step), 12.10 GiB
  live, 15,212 MiB on the card (RTX 5070 Ti).  0.1.3 needed two bands with
  every slice parked and ran at 3.35 s a step.
- T533 (25 km) runs on a 16 GB card at four bands with every slice parked,
  `--streaming on --device-allocator async`: 34 min a forecast day,
  12.81 GiB live.  The door admits that plan from a measured row and
  refuses the default allocator, which runs out of memory at that plan,
  naming the two flags.

## 0.1.3

### Added

- `gpuwm-global score RUN_DIR --out DIR` scores a finished run of any
  truncation (T255, T383, T533) in one command.  The verification of record
  is observations: ASOS/METAR stations through `rw_asos` (2 m temperature,
  2 m dewpoint, 10 m wind speed, sea-level pressure) and IGRA2 radiosondes
  through `rw_igra2` (500 hPa height, 850 and 500 hPa temperature, 250 and
  850 hPa vector wind).  The GFS analysis (GDAS f000) and the IFS analysis
  (ECMWF open data, 0 h) are secondary references, scored on the run's grid
  and labelled as a distance from another model, not as skill.  Every decode
  is a Rust door; the doors and the analyses are fetched when not given.
  Before it, observation scores came from two module entrances fed by a
  Python IGRA2 extract, and T383 and T533 had none.

### Fixed

- The cold start's regrid of the analysis onto the Gaussian grid and its
  ln(p) remap onto model levels run in Rust: `rw_global_coldstart`, a
  library built from this package's own `rust/` workspace and carried in the
  companion bundle from this release.  Both ran in host NumPy on every
  analysis-initialised start, which the Python boundary forbids on the data
  path.  The Rust output is byte-identical to the NumPy it replaces, so no
  analysis-initialised answer moves.  There is no NumPy fallback: without
  the library a cold start refuses as a missing door, and `doctor` names it.
- The render tapes and the regional translation run in Rust.  Every tape
  `export`, `render` and `go` write was regridded from the Gaussian grid
  with NumPy and its hydrostatic column integrated in a Python loop per
  level, and `translate-regional-frame` interpolated each target column in
  ln p with its own Python call to np.interp.  Both now run in
  `global_render_kernels`, the second library built from this package's
  own `rust/` workspace and carried in the companion bundle from this
  release, with no NumPy fallback: without it those commands refuse as a
  missing door.  Measured on Linux (glibc, AVX2) the output is bit for bit
  what the NumPy wrote.  On an AVX-512 machine (measured on an EPYC 9554)
  NumPy's own exp, log and pow differ from the C library's in the last
  bit, so the float64 regional frames differ from the earlier NumPy ones by
  about 1 ulp there (the float32 tapes did not change); the library gives
  the same bits on both.  The Windows build has not been measured against
  the Windows NumPy.  The companion bundle now carries ten Rust doors, the
  cold-start library and the render kernels among them.
- The semi-Lagrangian tracer mass fixer no longer rescales a whole species by
  one factor.  The default `[semilag] tracer_fixer = "bermejo_conde"` is now
  the published Bermejo-Conde fixer in its IFS form: each point is corrected
  in proportion to the gap between its cubic and its trilinear value at the
  departure point, toward the trilinear value and never past it.  On the T255
  bare forecast day the uniform factor fell from 0.119 (graupel), 0.057
  (snow), 0.050 and 0.043 (their number moments) in the worst step to float32
  roundoff (2.4e-7), with every species still closed to 2.9e-7 of its mass per
  step.  Answers change: after 24 h the global graupel content is 10.7
  percent lower, accumulated graupel 8.9 percent lower, accumulated rain 1.3
  percent lower and cloud water 5.3 percent higher than on 0.1.2.  The old
  uniform form is selectable as `mass_proportional`.  Restarts written by
  0.1.2 under the semi-Lagrangian core stay inspectable and are not resumed,
  because the scalar transport pin moved to v4.
- `gpuwm-global doctor` exits 0 on a correct install against the published
  gpuwm 2.8.0.  It graded `preflight.measured_free_vram_bytes` and
  `surface_bias.interpolate_to_tape`, which that engine does not carry, as
  gaps and exited 1, though no documented command reaches either.  Both now
  print as optional notes; calling either still refuses by name.
- A Rust door the installed engine's own bundle declares is the engine's
  door.  `doctor` grades it against the engine's pins and names the
  engine's `fetch-bridges` as its remedy, and `fetch-doors` leaves it to the
  engine.  On an install whose engine bundle carries every door this model
  runs on, `doctor` graded the engine's `rw_asos` and `rw_goes` against this
  package's pins for a different build, reported the other six as not
  staged, and failed with eight gaps; there `fetch-doors` now says it has
  nothing to stage and succeeds.  The contract literal still has to be in
  the bytes, so an engine build older than the contract is still a gap.
- `fetch-doors` downloads from the repository the installed distribution's
  own metadata names, not from an address written into the source.
- An open-water column starts on a water temperature.  Open water holds its
  starting skin temperature for the whole forecast (there is no ocean model;
  an inland lake now follows its own heat budget, below), and the cold start
  took that skin from a plain bilinear regrid of the analysis skin
  temperature, which mixes in the analysis's land points wherever the run's
  land fraction calls a column water and the analysis calls part of its
  stencil land.  On the GDAS 2026-09-30 12Z analysis the north basin of Lake
  Turkana (4.45 N, 36.09 E on the T255 grid) started at 321.1 K, the
  afternoon desert around the lake, where the analysis's own lake point read
  300.8 K.  Held for the forecast, that column evaporated about 1.2e-3
  kg/m2/s, near 3,000 W/m2 of latent heat, day and night, and emptied its
  500 kg/m2 surface reservoir at hour 117.3 of a 120 h T255 forecast, which
  ended with "native physics water closure exceeds the explicit surface
  reservoir".  The water books were exact; the flux they booked was wrong.
  Open-water columns now take the analysis skin over the analysis's own
  water points only: the bilinear weights renormalised over the stencil's
  water points, and a stencil with none reads the analysis water values
  grown outward over its land, the rule metgrid applies to SST.  Land and
  sea-ice columns keep the plain regrid.  On that analysis the rule changed
  1,784 of the 164,018 open-water columns by more than 1 K, and the run
  receipt records it under `initial.provenance.open_water_skin`.
- The land surface never receives convective rain the atmosphere did not
  lose.  Grell-Freitas's deep arm reports more rain than its own tendencies
  take out of the column: 1.27 times over the raining columns of WRF
  v4.6.1's oracle fixture, and 1.77 times over the globe and 2.28 times over
  land in the first day of the T255 GDAS 2026-09-30 12Z forecast.  Noah was
  forced with the reported number, so its soil, canopy and runoff gained
  water that never left the atmosphere, and the surface reservoir paid for
  it: in a 240 h forecast from that analysis the reservoir of a convective
  column in Colombia fell to 124 kg/m2 as the scheme reported 660 kg/m2 of
  rain and removed 288, about 1.6 kg/m2/h, enough to stop a forecast of
  about two weeks with the same refusal as the lake.  The land bucket now
  takes the reported rain only up to the water the call removed, and `RAINC`
  carries that same number, so the maps and the water budget's evaporation
  read the convective rain that left the atmosphere rather than the scheme's
  larger report (the scheme reported 660 kg/m2 on that Colombian column and
  removed 288).  What a call reported beyond the removed water is kept as its
  diagnostic `cumulus_rain_withheld_kg_m2`, beside
  `cumulus_rain_reported_kg_m2` and `cumulus_water_removed_kg_m2`.  The
  atmosphere is unchanged.  On the GDAS 2026-09-25 12Z case the lowest
  reservoir at 120 h rises from 267 to 399 kg/m2; the global 2 m temperature
  and 500 hPa height errors against the GDAS analyses at 24, 72 and 120 h (a
  diagnostic, not the verification of record) change by less than 0.01 K
  and 0.5 m.
  Against observations on that case, the verification of record (one
  T255 case, 1 degree boxes), the four-day precipitation total over 60 S
  to 60 N falls from 1.43 to 1.06 times CMORPH's gauge-adjusted total, over
  land from 1.45 to 1.05 times the CPC gauge analysis, and over the
  contiguous United States from 1.59 to 1.17 times the MRMS multi-sensor
  total, and the equitable threat score at 10 and 25 mm rises against all
  three.  At about 2,700 to 2,950 surface stations (METAR, 12Z, 24 to
  120 h) the 2 m temperature error rises by 0.002 to 0.022 K, the 2 m
  dewpoint error moves by -0.034 to +0.195 K (the largest at 96 h, where
  the dry bias grows from 1.83 to 1.98 K) and the sea-level pressure error
  by +0.002 to +0.224 hPa; against about 470 to 540 radiosondes (12 to
  72 h) the 500 hPa height error falls by 0.02 to 0.18 m and the 850 hPa
  temperature error by 0.002 to 0.015 K.
- An inland lake's water temperature follows its own heat budget.  Open
  water held its starting skin temperature for the whole forecast.  On the
  ocean that is the analysis SST; on an inland lake it is the analysis's
  lake temperature, which is weakly observed.  GDAS carried Lake Tana (11.9
  N, 37.5 E) at 304.8 to 305.1 K at all eight cycles of 2026-09-29 and
  2026-09-30, 9.2 K above the 2 m air over it, and held for a 240 h T255
  forecast that column evaporated about 22 kg/m2 a day, several times
  published estimates of the lake's open-water evaporation (about 4 to 5 mm
  a day); the warm lakes were the columns whose surface reservoirs would
  next reach "native physics water closure exceeds the explicit surface
  reservoir", about 23 days out.  A lake column (WRF's LAKEMASK rule: open
  water whose lake share in the statics' land-use fractions beats its ocean
  share; 920 columns at T255, from the Black Sea, the Caspian and the Great
  Lakes to Tana and Turkana) now integrates its skin on the land cadence
  from the shortwave and longwave it absorbs, its own emission and the
  surface layer's sensible and latent fluxes, over the water heat capacity
  the state carries (a 10 m mixed layer), and is not cooled below 273.15 K.
  The ocean, land and sea ice are unchanged, and a run without lakes is bit
  for bit the previous runtime.  The statics carry the lake share as the new
  surface member `lake_fraction`: a checkpoint written before it reads with
  no lakes and records the absence, so it resumes and runs as it did, and a
  checkpoint written with it is refused by a reader from before it
  ("checkpoint has an unknown array namespace").  Against observations on
  the GDAS 2026-09-25 12Z case (T255, 120 h, one RTX PRO 4500, both arms
  with the two fixes above): against OISST v2.1 on the lake columns it
  covers, the lake surface temperature error at 120 h falls from 1.37 to
  1.11 K RMSE (bias +0.37 to -0.35 K), because the observed lakes cooled 1.0
  K over the five days and a held skin cannot follow;  Lake Superior 1.49 to
  0.66 K, Lake Huron 1.09 to 0.68 K, Lake Michigan 1.15 to 0.68 K, the Black
  Sea 1.22 to 0.99 K, the Caspian 1.64 to 1.39 K, while Lakes Erie and
  Ontario (1.15 to 1.51 K) and Ladoga (0.75 to 0.80 K) move the other way.
  Lake Tana's column cools from 305.3 to 299.6 K by 120 h and evaporates
  13.7 kg/m2 a day over the five days instead of 20.7.  At about 2,700 to
  2,960 surface stations (METAR, 12Z, 24 to 120 h) the 2 m temperature RMSE
  moves by -0.013 to +0.019 K, the 2 m dewpoint by -0.041 to +0.086 K and
  the sea-level pressure by -0.164 to +0.016 hPa; at the 210 to 220 of them
  within 100 km of a lake column the 2 m temperature moves by -0.025 to
  +0.171 K (the largest at 96 h) and the dewpoint by -0.031 to +0.067 K.
  Against 425 to 540 radiosondes (IGRA2, 12 to 72 h, the two arms rerun on
  one RTX 5090) the 500 hPa height RMSE moves by -0.085 to +0.059 m and the
  850 hPa temperature by -0.011 to +0.020 K.  The precipitation total
  against CMORPH (60 S to 60 N), the CPC gauges and MRMS moves by at most
  0.017 of its ratio to the observed total.
- A render tape names its model.  Every wrfout-shaped tape the export
  writes carries the global attribute `GPUWM_MODEL_LABEL`, so the engine's
  renderer draws the global model's name in each map's metadata row.  The
  renderer imported any such file under its generic WRF identity, so every
  global map said WRF.  An engine whose renderer does not read the
  attribute ignores it and draws as before.
- A configuration that omits `[time] dt_s` under the semi-Lagrangian core
  takes the core's 300 s step at every truncation, T533 included.  A
  truncation-scaled step (225 s at T533) was carried briefly as the remedy
  for the 0.75 trajectory norm gate's stop of a T533 forecast at 66.3 h; it
  is retired with that gate (below), and the T533 presets are back on
  300 s with their 1500 s radiation and 300 s land buckets, so their
  identities are the 0.1.2 ones apart from the dated greenhouse gases.
- The physics registry and the documentation keep verification, validation
  and device qualification apart.  An adapter contract that has passed its
  device battery is `device-qualified`, a statement about the device, not a
  validation against observations.  A contract spelled `validated` is still
  accepted as an alias and keeps its spelling, so its hash and every bridge
  or checkpoint identity bound to it are unchanged.
- Grell-Freitas reports the convective rain its own tendencies take out of
  the column.  The deep arm added the downdraft evaporation the detrained
  cloud water could not supply to the rain instead of subtracting it, a
  sign WRF 4.7.1 carries (`module_cu_gf_deep.F:3348`), so the reported rain
  was 1.27 times the water removed over the WRF oracle capture's raining
  columns and, on the T255 GDAS 2026-09-30 12Z day, 1.77 times globally and
  2.28 times over land; Noah was forced with the excess and the surface
  reservoir paid for it.  It is now 1.004 times on the oracle capture and
  1.03 globally and 1.02 over land on that day, the remainder being the
  model's specific-humidity water metric against the kernel's
  mixing-ratio one (the oracle capture reads 1.025 in the model's metric).
  Day-one convective rain falls from 1.570 to 0.913 kg/m2 globally and from
  1.727 to 0.764 kg/m2 over land; the land's top soil water ends the day
  1.2 percent drier, its evaporation 3.9 percent lower and its 2 m
  temperature 0.02 K warmer.
- Grell-Freitas's undilute buoyancy integral keeps its cloud-base layer
  (WRF `module_cu_gf_deep.F:3024`), so the deep and the shallow closures no
  longer read one layer short.  Deep convection ran in 0.08 percent more
  column calls on the T255 day and the day's convective rain moved by 0.09
  percent.
- YSU runs its thermal-enhanced Richardson sweep only where the
  boundary-layer flag is set (WRF `bl_ysu.F90:703-728`), so a column whose
  first boundary-layer top sat below the first interface stays in the local
  diffusion regime as WRF keeps it; on the engine's measured onset column
  the boundary layer reads 95 m, not 431 m.  Global and land means on the
  T255 day move by about 0.001 K in 2 m temperature and 0.3 m in
  boundary-layer height.
- The semi-Lagrangian core no longer stops a forecast on shear or
  rotation.  Its trajectory gate refused when dt times a norm of the flow
  Jacobian passed 0.75, and a norm cannot tell a fold from a rotation: it
  refused a T533 forecast at 66.3 h (norm 0.7511) and a six-iteration
  T799 run at 0.25 h.  The gate now refuses where the trajectory map
  actually folds, when the smallest det(I -+ (dt/2) J) over the grid falls
  below `[time] minimum_fold_determinant` (default 0.2), and the norms stay
  in the receipt as diagnostics.  A departure search that misses its
  convergence test is searched again at eight iterations before the step is
  refused, and the receipt counts those steps.  `maximum_lipschitz = 0.75`
  in an existing config is accepted and does nothing; any other value is
  refused, because it would no longer be read.  Every configuration keeps
  its identity, so existing checkpoints still restart, and a run the old
  gate admitted is bit-identical (T255, six hours: all 133 checkpoint
  arrays equal).  MEASURED 2026-10-05 on an RTX PRO 6000, GDAS 2026-09-25
  12Z at the shipped 300 s step: T533 now completes 120 h with every gate
  passing (smallest fold determinant 0.808, largest norm 0.772), where
  0.1.2 stops at 41.9 h on a norm of 0.757; T799 completes six hours
  (fold determinant 0.674, norm 1.085), where 0.1.2's three-iteration
  search is refused by its convergence test at the first step (the retry
  at eight iterations admits that run, not the fold gate).  On the merged
  0.1.3 tree the same T799 case runs the whole day at 300 s with the
  native suite (RTX PRO 6000, 7.2 s a step): every gate passes, the
  smallest fold determinant is 0.633, and every one of the 288 steps took
  the eight-iteration retry, so a truncation-scaled iteration count would
  save time there.  A dry T799 day (physics off) is refused by the fold
  gate at 2.9 h after its grid-scale noise trip: with no boundary-layer
  drag the flow really folds.
- Every physics menu row answers the engine's `urban_scheme_id` with 0.
  The gpuwm 2.8.5 menu rows carry that field, so a picker keyed on the
  engine's own rows took a KeyError on every row of this menu, and the
  suite's engine-measured menu test failed on the engine a fresh install
  resolves.  No shipped experiment selects an urban canopy scheme, so 0,
  the engine's own spelling of none, is true of every row.
- The forecast clock is the analysis valid time.  The native suite's solar
  geometry ran on the config's `start_time_utc` literal and the reference
  suite on a 00 UTC equinox fixture, so a config repointed at another
  analysis radiated on the wrong sun.  A run that starts from an analysis
  now takes its clock from the analysis itself, a stated literal that
  disagrees with it is refused naming the fix, the reference suite's
  zenith follows the same dated geometry as RRTMGP, checkpoints carry the
  clock and a restart from another start is refused, and a render tape's
  valid time comes from the run's clock (`--start-date` is optional and
  refused when it disagrees).  The 30 shipped GDAS native configs state the
  clock their analysis carries, so their identities do not move.  The run
  receipt's greenhouse-gas record is dated by the same clock: a config that
  follows its analysis and states no `start_time_utc`, as the clock's own
  refusal tells the reader to write it, integrated its whole forecast and
  then died writing the receipt.  Measured on a T255 48 h run from GDAS
  2026-10-02 00Z with no `start_time_utc`: before, all 576 steps ran and
  the run exited 1 at the receipt; after, it ends with status pass.
- RRTMGP runs with the dated present-day well-mixed gases.  It ran with
  369.55 ppm CO2 (the year-2000 value) and the RFMIP climatology's CH4 and
  N2O; CO2, CH4 and N2O now come from the run's valid year in a pinned
  NOAA GML table, and the receipt records what the radiation ran with.
- The native suite's stratospheric temperature floor is off by default.
  Default-on at 50 hPa it pulled the analysed polar-night stratosphere
  toward 195 K on a 30 minute timescale, warming the cold vortex the
  analysis carried.  The hour-83.4 top-sag death it was installed for does
  not reproduce on this tree: a 240 h T255 run with the floor off passes
  every gate, its top level is 189 K at its coldest at 240 h where the
  floored run holds 195 K, and the two runs are identical through hour 144.
  An arm that turns it on reaches only above 5 hPa, and every receipt books
  the floor's heating and the points it touched in a ledger.
- The spectral water vapour's per-step hole filler takes a negative value's
  deficit from its own column's neighbouring levels first, and the
  receipt carries a ledger of how much water it moved and how far.  The
  proportional refill it replaced moved water up and down the column with
  no record; measured on the T255 day its vertical transfer is far below
  the 2 m dewpoint dry bias, so the fixer is not that bias's cause.
- An analysis whose top lies below the model lid is extended upward on the
  standard atmosphere's shape from the analysed top value.  The IFS
  open-data start tops at 10 hPa under a 1 hPa lid and held the 10 hPa
  temperature into the lid, 20 to 40 K colder than the stratopause; the
  extension warms the levels at 1.2 to 7.55 hPa by 41.2 to 3.6 K.  A
  float32 run keeps its bits on every level below the analysis top.
- Render tapes and regional frames write water as WRF's dry mixing ratio.
  They wrote the model's specific humidity under WRF's QVAPOR..QGRAUP and
  Q2 names, 2 percent dry at q = 0.02 (a 0.33 K dewpoint bias in every
  WRF-shaped reader); the conversion is the model's own q / (1 - qv), the
  tape names its convention, and the regional dry column and specific
  volume use the mixing ratio.
- The assimilation's 2 m temperature operator is the model's own
  screen-level diagnostic, as its 10 m wind operator already was, instead
  of a 6.5 K/km lapse reduction of the lowest level that injected a cold
  bias under inversions.  The door's successive-correction gain for a 2 m
  report is the optimal-interpolation gain of that operator: it follows
  the lowest level by only h kelvin per kelvin (the rest of the 2 m
  temperature is the skin's), so the spread takes the innovation d/h under
  the gain g h^2, with h measured per report through the same diagnostic
  and floored at 0.25; the report records h.  Spread with unit
  sensitivity, as the lapse operator's door did, the first analysis of the
  smoke case removed 17 percent of the 2 m misfit; with the operator's
  gain it removes 20 percent.  The lapse operator's 58 percent came from
  reading a 2 m report as the lowest level itself, which is the inversion
  injection this fixes, so the cycling test's bar is now the OI one (0.85
  of the background misfit, measured 0.80) instead of the lapse
  operator's 0.6 or the 0.9 it had been loosened to.
- IFS soil layers are remapped onto Noah's layers by depth.  They were
  handed over by position, so the IFS 0 to 7 cm layer was read as Noah's 0
  to 10 cm and the 100 to 289 cm layer as the 100 to 200 cm one.  The GDAS
  layers already match Noah's and pass through bit for bit.
- The semi-Lagrangian departure level is clamped at the lid and ground
  interfaces, not at the outermost full levels, and the level rate divides
  by the geometric spacing of the full levels the gather reads.  Clamped at
  level 0 a parcel arriving there under descent read its own value, so the
  lid cooled under ascent and never warmed under descent; the arithmetic
  layer means made vertical advection 1.7 to 4 percent too slow.  The
  Eulerian cores' top and bottom faces take the one-sided gradient to their
  neighbour.  On the dry T255 day the level-0 theta drift goes from -12.6
  to +0.5 K a day.  The ln ps continuity's ride on the lowest level was
  measured at T63 and showed no orography error there.
- A forecast stopped by a refusal inside the model still delivers the
  pictures of the hours it reached: `go` renders the checkpoints the
  stopped forecast wrote and exits 5, and the run-plan `go` route renders
  them before it reports the stop.  A five-day forecast whose one column's
  surface reservoir ran dry used to deliver nothing.
- Multi-card runs split the work and carry it on the device link.  The
  physics runs only the latitude bands a card owns and the semi-Lagrangian
  step runs a band at a time behind a row halo, each card holding its own
  rows of the trajectory, so P cards return one card's bits; device
  exchanges ride NCCL when every rank can open it (the gpu extras now
  install the NCCL wheel CuPy's own does not), with a watchdog that fails a
  stuck collective by name and a running check that the ranks ran the same
  exchanges; `card_transport = "auto"` no longer refuses a box where NCCL
  imports; each rank is placed on its own card; and the halo goes to the
  two neighbours only.  Gate WIRE-1 grades the device link's probe on
  NCCL, because a collective's in-run time includes its wait on the
  slowest rank.  The in-box rate and the GPU 1-vs-P identity at T255 and
  T383 on a multi-card box are measured separately (see Measured).
- The render tape names the model WOOF Global.
- `da fresh` from an analytic native base (the level-5 smoke) dates the
  physics with `--start-utc`; it kept the base's own date and the cycle
  refused every other start.
- `abi-score` reuses an existing tape only when the tape was written for
  the same start; a disagreeing `--start-date` goes back through the
  export door's clock check.
- `score` fetches its reference analyses with the engine its own Python
  imports; a `gpuwm` on PATH belonging to another interpreter left a
  scorecard with no analysis rows.
- The longwave radiates the air above a 1 hPa model top.  WRF v4.6.1's
  RRTMG longwave adds nint(p_top / 4 hPa) buffer layers above the model
  top and, when that rounds to none, moves the top interface to zero
  pressure, so the air above the top is always radiated.  The carried
  driver kept the count and not the zero top, so under the global model's
  1 hPa lid the top layer received no downward longwave from above and
  cooled to space.  The driver now carries at least one longwave layer
  from the model top to the coefficient floor; a top of 2 hPa or more
  keeps WRF's count exactly, so no regional top moves.  Global-mean
  potential temperature drift of the top level on the T255 native suite,
  every gate passing: GDAS 2026-09-01 00Z 24 h, -59.4 to -10.6 K/day;
  GDAS 2026-10-02 00Z 48 h, -47.5 to -8.1 K/day (levels 1 to 3 move
  from -13.7/-6.6/-2.8 to -3.7/-2.3/-0.9 on the second case).  Rain
  against CPC gauges and MRMS on the 48 h case moves from 1.210 to 1.200
  and from 1.279 to 1.262 of the observed total, within noise.
- The cold-start and render libraries honour `RAYON_NUM_THREADS` and
  `OMP_NUM_THREADS`; they took every core of a shared box.
- Smaller: the tracer fixer's local share reaches the receipt, a failed
  run's receipt records the greenhouse gases it radiated with, the soil
  depth refusals name the breakage, and an empty `CUDA_VISIBLE_DEVICES` is
  left as the operator set it.

### Changed

- CI proves `doctor` on a correct install.  The test job (every push) and
  the publish job run the README's three install lines, `gpuwm
  fetch-bridges`, `gpuwm-global fetch-doors` and `gpuwm-global doctor`, and
  fail on any gap.  Nothing that decided the 0.1.2 cut ran doctor, which is
  how a doctor that exited 1 on every correct install shipped.  The suite
  holds the job's lines and the README's lines in step.
- The CPU suite runs in two tiers.  The quick tier,
  `-m "not gpu and not slow and not network"`, runs on every push.  The
  full tier, `-m "not gpu"`, runs nightly and on every publish.  Sixteen
  whole-model gates (the IMEX and semi-Lagrangian rest, energy and
  nonlinear runs, and the profiled runs), each 6 s or more on its own, move
  to the full tier by name in `tests/conftest.py`, each with its measured
  time; none is skipped.
- The engine seam is proven at `gpuwm 2.8.5`, the engine a fresh install
  resolves (GV-6).  The pins stood at 2.8.0, so on 2.8.5 `doctor` proved
  22 of 47 files, and the two seam hash nodes and the two carried-physics
  divergence comparisons skipped.  The 2.8.0-to-2.8.5 change was read hunk
  by hunk.  Sixteen symbols this package reaches in the 25 moved seam files
  changed definition, and none of them is reached by a shipped experiment:
  the native suite calls the carried scheme modules directly with a
  namespace configuration and never builds `RunConfig`, `DomainState` or
  the carried physics driver; its radiation, surface layer, land surface,
  boundary layer and microphysics slots each admit one scheme (RRTMGP,
  sfclay, Noah, YSU, Morrison) and the cumulus slot GF, New Tiedtke, the
  mass-flux scheme or none; and the reference suite reaches no engine
  module.  Per symbol: `RunConfig` flips `moist_cq` on and appends
  regional fields that default off; `DomainState` gains a device face-mass
  kernel and a host copy option, and `init_at_rest` an admission refusal,
  both reached only from two reference-state builders nothing calls, as is
  `update_diagnostics`, which gains an opt-in strict-arithmetic branch;
  `KainFritsch` gains an adaptive-step cumulus period and uncleared output
  buffers and is selected only at `cu_physics=1`;
  `launch_mynn_surface_layer` and `mynn_pbl_step` gain stochastic
  perturbation and scalar mixing (MYNN is not admitted);
  `NoahmpRuntimeParameters`, `noahmp_cold_start` and `noahmp_lsm_step`
  gain the urban arm (Noah-MP is not admitted); `RucRuntimeParameters`
  and `ruc_lsm_step` gain mosaic, lake and soil-property options (RUC is
  not admitted); `launch_shinhong` moves to a tiled global workspace
  (that boundary layer is not admitted); and `AnalyticClearSkyRadiation`,
  `DudhiaShortwaveRadiation` and `RRTMGLegacyRadiation` take the
  adaptive-step period and a device adapter, behind a radiation selector
  the native suite never reaches.  Every name the shipped path binds in
  those files is unchanged in its definition and in everything it calls
  inside its file.  The 2.8.5 package init also installs a process-wide
  strict-arithmetic compile hook when `GPUWM_WRF_EXACT=1` is set, which
  would reach the carried kernels too; nothing in this package sets it.
  The pins are re-taken at 2.8.5 with `tools/pin_engine_seam.py` (47 of
  47, every file byte-identical to the public `v2.8.5` tag), and the
  divergence baseline with `tools/fingerprint_engine_divergence.py
  --rewrite`: 850 hunks, 371 carried forward, 53 dropped and 479 read and
  classified in `docs/CARRIED-PHYSICS-DIVERGENCE.md`.  None of them moves
  a shipped number, because the shipped path runs the carried copy and
  reaches the engine's copy of a carried file only through
  `load_trace_climatology`, unchanged.  Newly owed there, not taken here:
  the engine's respelling of constant divisions for Blackwell cards (five
  kernel rows, the only pull that moves bits once taken), per-card and
  per-stream caches, a value-keyed land-surface table cache, and an FP32
  literal in the cold-start soil water.  On 2.8.5 the two hash nodes and
  the two divergence comparisons now run and pass; on any other engine
  they skip by name.

### Measured

- The README install, then `doctor`, against `gpuwm 2.8.5` from PyPI on a
  Linux CPU host with no card, Python 3.11: exit 0, with the two absent
  engine symbols printed as optional notes, every observation binary ok (eight), of which
  `fetch-doors` staged six and the engine's own bundle carried `rw_asos`
  and `rw_goes`, and the engine seam at 22 of 47 against the 2.8.0 pins.
  Measured 2026-10-05.  After the re-pin (Fixed above), the same `doctor`
  on the same engine reads `ok seam 47/47 files proven, pinned against
  gpuwm 2.8.5`, exit 0; Linux CPU host, Python 3.12, 2026-10-05.
- The CPU suite from the installed wheel against `gpuwm 2.8.5`, one
  process, on a 24-thread Linux CPU host: the quick tier 1,908 passed,
  82 skipped, none failed in 205.6 s; the full tier 1,930 passed, 82
  skipped, none failed in 609.8 s.  Before the menu fix the same
  `-m "not gpu"` selection on this engine failed one test, the menu
  test above.
  Measured 2026-10-05.

### Open

- The top model level still cools under the physics, about 8 to 11 K a
  day of potential temperature at level 0 on the measured cases (Fixed
  above), against under a kelvin a day for the dry dynamics alone.  The
  remainder has not been attributed.
- The dry nine-day steady-state test (Jablonowski-Williamson) drifts in
  surface pressure under the shipped float32 state: 1.51 hPa by day 9,
  past the 1 hPa bar from day 6, where the same run in float64 stays
  within 0.25 hPa.  Measured 2026-10-05; a precision fix is in progress.

## 0.1.2

### Fixed

- A configuration has the same identity on every machine.  The hybrid
  level weights are now computed with the C library's pow and exp one
  element at a time.  With numpy 2.5 on an AVX-512 Linux machine, numpy's
  vector power and exp rounded the last bit differently, so every shipped
  configuration hashed to a different identity there than on Windows or an
  older CPU.  Every recorded identity is unchanged.
- The door bundles carry a licence notice.  Each zip now holds
  `THIRD-PARTY-LICENSES.txt`, generated from the engine workspace's
  `Cargo.lock` for that platform: every crate statically linked into the
  binaries with its licence expression, and every distinct licence and notice
  text those crates ship.  The 0.1.1 zips carried the binaries alone, while
  the MIT, BSD, Apache-2.0, ISC and Unicode licences of the crates inside them
  each ask for their notice to travel with a binary copy.  `fetch-doors`
  stages the notice beside the doors, and `tools/build_door_bundle.py pin`
  refuses a bundle without one.
- The door binaries carry no build-machine paths.  They are built with
  `--remap-path-prefix` for the engine checkout, the Cargo home and the rustup
  home.  The 0.1.1 binaries embedded about 1,100 paths under the Linux build
  host's home directory and about 1,250 under the Windows one.
  `tools/build_door_bundle.py scan` reads every member with its NUL bytes
  stripped, and `pin` refuses a bundle it finds any such path in.
- The door bundles are built from public source.  `bridges.yml` builds them
  from the engine's public `v2.8.0` tag, the first public engine release to
  carry the `rw-atms` crate, the five global observation binaries and the
  `rw_asos` and `rw_goes` subcommands this package calls.  Through 0.1.1 they
  were built from a revision no public repository carried, so nobody could
  rebuild them.  `rw_atms` keeps its source-revision stamp in that build, and
  the release no longer exempts it.
- No shipped file names a private machine or a private case.  Six comment
  lines in the carried physics (`core/gf.py`, `kernels/gf.cu`,
  `kernels/ntiedtke.cu`, `kernels/ysu.cu`), the static covariance receipt and
  its npz twin, and the ABI fast-model table named the machines figures were
  measured on; eight docstring lines in `core/landuse.py`, `core/npref.py`,
  `core/physics.py` and `core/rrtmgp.py` named one private case.  They now
  name the card, the platform or "the reference configuration".  The carried
  files are rewritten by a rule in `tools/resync_from_owner.py`, not by hand,
  and `tests/test_no_provenance.py` asserts both over everything that ships.
  The three kernels change comment text only: their `source_sha256` and the
  kernel-set digest a receipt records move, and no compiled image does.
- `NOTICE` credits Recommendation ITU-R P.676-13, whose oxygen and water
  vapour line tables `arwen_global/microwave/absorption.py` carries, and says
  what of CRTM ships: a regression fitted to CRTM v3.1.1 optical depths, the
  digests of the coefficient files that reference read, and infrared land
  emissivities averaged from CRTM's IGBP table.  CRTM v3.1.1 is dedicated to
  the public domain under CC0 1.0.
- `tools/crtm_reference/`, the CRTM reference driver the ABI operator page,
  the `abi-reference` help and the fast-model table all name, ships.  It is
  this project's Fortran calling an installed CRTM, with no CRTM code in it.
- RRTMGP no longer reads the RFMIP clear-sky input file.  Every radiation
  construction opened `rfmip-clear-sky-inputs.nc` from the engine's
  `gpuwm-data` companion for 136 numbers: sixteen trace-gas global means and
  the median layer pressure and ozone profile.  The engine stops shipping
  that file at 2.8.0, which would have made the first radiation call refuse
  on that companion.  The carried driver now reads the engine's own derived
  table through `gpuwm.core.rrtmgp.load_trace_climatology`, pinned by
  SHA-256, and the RFMIP clear-sky oracle fetches its input through the
  engine's pinned upstream route.  The radiation input is unchanged: every
  value equals what the replaced code computed from the NetCDF, bit for bit.
- The Morrison microphysics kernel takes the four engine fixes of
  2026-09-12: vapor returned by the final condensate cleanup is stored
  instead of dropped, cloud freezing is evaluated in log space so an ordinary
  cloud slope no longer overflows and erases it, exceptional rain freezing
  stays within the joint donor budget, and an in-range number moment is no
  longer rebuilt during slope diagnosis.
- A land cell over the water soil category keeps its land use as its
  vegetation and takes silty clay loam, as real.exe matches it, instead of
  becoming mixed forest in the final reconciliation (engine fix of
  2026-09-27).  Glacier cells still take the ice soil.
- The `rw_asos` door row carries the engine's 2.8 `--abi` line (the v2
  surface record with the observation time and the one-minute route), so one
  binary built from the engine's commit answers both tables.
- Every global statics build on the 2.8 engine raised AttributeError on its
  first sector: the static builder asks the grid for a sampling handle the
  rows grid did not answer.  It does now.
- The statics cache sidecar recorded the build machine's hostname, and two
  probe tools did the same in their reports.  They record the kind of
  machine (system and architecture) instead.
- Six statics tests never ran against a published engine.  The suite's
  probe for the static builder's `rows` grid kind called the bridge with a
  signature and a spec the 2.8 engine does not accept, and read the error as
  "no rows kind".  The probe now sends this package's own rows spec, and the
  six run: the Gaussian-grid build through the crate and the statics door
  that builds the cache a run reads among them.

### Changed

- The engine range is `gpuwm>=2.8.0,<2.9` and `gpuwm-data>=2.8.0,<2.9`.
  The floor names a breakage: the trace-gas loader, the RFMIP fetch route,
  the v2 `rw_asos` line and the static builder's `rows` grid kind are all
  first in 2.8.0.
- The engine seam is re-pinned at the published `gpuwm 2.8.0`: 47 of 47
  files, read off the PyPI wheel and byte-identical in the Windows wheel,
  the pure wheel and the public `v2.8.0` tag.  Eighteen of the 46 files
  pinned at 2.7.3 moved by 2.8.0, and `gpuwm/core/rfmip_upstream.py` joins
  the table because the RFMIP oracle fetches through it.  `tools/measure_boundary.py` reads 234 symbols
  across 67 modules, with no gap among the 194 a CPU host can resolve.
- The door bundles are built from the engine's public `v2.8.0` tag
  (`0164ae0f2d70`), and `door-pins.json` pins them at release `v0.1.2`.  The
  Windows doors are built for `x86_64-pc-windows-gnu`, and
  `tools/build_door_bundle.py pack --triple` resolves that bundle's licence
  notice for the target it was built for and names that target in the
  notice.
- The run manifest carries a `physics` block, as the engine's does from
  2.8.0.
- The carried-physics divergence rows are re-baselined on the published
  `gpuwm 2.8.0`: 428 hunks, every one classified in
  `docs/CARRIED-PHYSICS-DIVERGENCE.md`.

### Measured

- The ten-step acceptance 0.1.1 owed, taken.  The T255 L40 native ten-step
  gate (semi-Lagrangian core, 300 s, the full native physics suite, GDAS
  2026-09-01 00Z, one latitude band) ran once from the model source tree at
  the revision `SOURCE.md` names and once from this package installed against
  published `gpuwm 2.7.3`, on one RTX 4090, from one config file.  Every
  member of every checkpoint is byte-identical: 42 arrays at step 0 and 132
  at steps 5 and 10 (306 arrays, 309 members with the three metadata blobs),
  and the three checkpoints' `self_sha256` agree.  Config hash `e76cca951a57`;
  kernel-set digest in the package receipt `c1fd4a73a864`, the new value the
  comment-only kernel edits above produce (0.1.1: `9e256633abc3`).
  Measured 2026-09-26.
- The trace-gas table swap changes no forecast number.  The same ten-step
  config ran before and after the swap on one RTX 4090, against `gpuwm
  2.7.3`, with RRTMGP in the physics suite.  All three checkpoint files
  are byte-identical, 309 members in all, including the surface and
  top-of-atmosphere radiative fluxes, the heating rates, OLR and 2 m
  temperature.  Measured 2026-09-26.
- On the published engine, from the installed wheel.  `gpuwm-global go` on
  the shipped T255 L40 native semi-Lagrangian semi-implicit experiment, cut
  to 6 h (72 steps of 300 s), from the GDAS analysis of 2026-09-01 00Z, on
  one RTX 5090 against `gpuwm 2.8.0`, `gpuwm-data 2.8.0` and
  `cupy-cuda13x` from PyPI: receipt pass, all eight physics modules from
  this package, the engine seam proven 47 of 47, maximum wind 91.536 m/s
  and total water 710.659936 kg/m2 at 6 h, 12 pictures.  Statics built
  fresh by the engine's static builder on the rows grid give the same
  numbers as the cached statics.  The CPU test selection from the installed
  wheel on Python 3.12 against the same engine: 1,906 passed, 79 skipped,
  none failed.  Measured 2026-09-29.

## 0.1.1

### New

- `docs/CARRIED-PHYSICS-DIVERGENCE.md`, which answers "the engine changed this
  file, does this package want the change?" by reading one page. The carried
  physics and the engine's copies of the same files move independently, and
  until now nothing in this repository separated a deliberate adaptation from a
  stale copy: `SOURCE.md` says which revision was cut and
  `arwen_global/data/engine-seam.json` pins the engine files the carried code
  still reaches, and neither says which differences are on purpose. The
  document carries 82 rows over the 19 carried files that differ, one section
  per file, each row giving the carried and engine line ranges, what the common
  ancestor says, the class of the difference, the commit on the side that
  moved, what changes numerically, and a decision: pull, refuse, offer or none.
  Three lists close it: what this package owes the engine, what it has that the
  engine line may want, and what it holds a position on. Measured against the
  published `gpuwm 2.7.3`; the rows on every file but the radiation driver, the
  physics driver, the physics inventory and the kernel loader read the same
  against any `2.7.x`.
- `arwen_global/data/engine-divergence.json`, the same 375 differences keyed by
  a fingerprint that survives a line shift: per carried file, a unified diff of
  the engine's copy against this one at zero context with the carve's rewiring
  applied to the engine side, then a SHA-256 over each hunk's engine-side lines
  and another over its carried-side lines. `tools/fingerprint_engine_divergence.py`
  computes it, documents the normalisation exactly, refuses to measure an
  engine outside the range this package declares, and records the version it
  did measure. All 35 carried files are measured, whatever the suffix, the
  four WRF parameter tables among them; a pair either side of which does not
  decode as text is compared byte for byte instead. The engine-side hash is
  what the engine line verifies a row by: read the row's engine line range out
  of the published engine, rewire, hash, compare.
- `tests/test_engine_divergence.py`, the gate. Against the installed engine it
  recomputes the fingerprints and fails, by file and by line, on a difference no
  row covers; by row name on a row whose difference is gone; on any row left
  unclassified; on an engine fix marked for pulling that the document's PULL
  list does not carry, and on a name that list carries which no row owes; on a
  row whose table in the document and whose entry in the JSON disagree about
  the file, the class, the decision or either line range; and on a carried file
  the measurement does not walk. Its two comparison nodes skip, naming both
  versions, when the installed engine is not the one the rows were measured
  against; the nodes that hold the rows, the document and the file list to each
  other need no engine. Every failure was driven before it shipped: one changed
  line in a carried module fails naming that module, one appended line in a
  parameter table fails naming the table, a deleted row fails naming the hunk
  it abandoned, a fabricated row fails naming itself, a table cell that
  disagrees with its rows fails naming the field, and a suffix filter put back
  into the walker fails naming the four tables it drops.

### Fixed

- The engine version the front page names. The index published `gpuwm 2.7.3`
  on 2026-09-12 and it is what `gpuwm>=2.7.0,<2.8` resolves, so `README.md`
  said the range resolved 2.7.2 while the new divergence document said 2.7.3,
  and a reader had two records of the same thing at two baselines. The page now
  names 2.7.3, carries the boundary reading measured against it (the same 231
  symbols across 65 modules, no gap among the 191 a host without a CUDA runtime
  resolves, Windows desktop 2026-09-12), and states what the engine seam does
  and does not cover. The pins have since moved to 2.7.3 as well, in the entry
  below this one, so the page, the document and the manifest all name the
  engine a fresh install resolves.

- The Noah FRZX fixture's hold on the mirror it imports. It compared the
  imported module's PATH with the path the source assertions read, so a suite
  run from an unpacked sdist against that sdist's own install -- which is how
  the release CPU selection is measured -- errored all four of its numerical
  nodes on two byte-identical copies of one file (a Linux CPU host, Python
  3.14.4, `gpuwm 2.7.3`, 2026-09-12: 1,865 passed, 83 skipped, 96 deselected,
  4 errors). The hold is now a SHA-256 on both sides and the message prints
  both paths and both digests, so the breakage it was written for -- a
  checkout with a different revision installed beside it, source assertions
  reading one tree while the numbers came from the other -- still fires, and
  a correct install stops being refused for its path.

- The engine seam, re-pinned against `gpuwm 2.7.3`. The pins hold one engine
  version at a time and were taken against 2.7.0; the index published 2.7.3 on
  2026-09-12 and it is what a fresh install of this release resolves, so
  `gpuwm-global doctor` printed eight files as moved and two nodes of
  `tests/test_arwen_global_engine_seam.py` failed on the only engine anybody
  gets from the index. The eight were read hunk by hunk against 2.7.0 before
  the pins moved: `config.py`, `core/state.py`, `core/preflight.py`,
  `physics_compat.py`, `physics_vertical_contract.py`, `core/microphysics.py`,
  `core/refl.py` and `core/rrtmg_legacy.py`. Of the 41 symbols this package
  imports out of them, 38 are byte-identical between the two engines and the
  three that moved change no number: `RunConfig` and `DomainState` moved in
  comment text with no field, default or validation touched, and
  `radiation_scheme_ids` narrowed a refusal, so a configuration that writes
  `ra_physics=4` beside `ra_lw_physics=4`/`ra_sw_physics=4` now resolves to the
  same `(4, 4)` pair instead of raising, while a contradicting pair is still
  refused. Reached-but-unchanged in behaviour, recorded so a later reader need
  not re-derive it: the scratch arena accepts a same-width dtype it used to
  reject and gained an optional argument nothing here passes; the microphysics
  ring-guard family is now derived from the engine's registry and grew ten
  scheme-native names behind a presence guard, on a path with no lateral
  boundary and so no door of this package enters; `core/refl.py` replaced four
  inline required-field literals with one published table whose rows are the
  same species; and four pinned files now run a registry-agreement check at
  import, which pulls `gpuwm/physics_registry.py` and
  `gpuwm/core/noahmp_kernel_sources.py` into the set of engine modules that
  execute when the carried physics is imported. `tools/measure_boundary.py`
  reads the same 231 symbols across 65 modules with no gap, and
  `tools/measure_engine_signatures.py` the same 28 signature rows, on 2.7.3 as
  on 2.7.2. The two nodes that compare hashes now skip, naming both versions,
  on any engine other than the pinned one, which is the rule the divergence
  gate already followed; the version ceiling stays the refusal. Three seam
  rationales quoted a line count against the engine and were re-measured
  against 2.7.3 (`config.py` 308 to 1,130, `core/state.py` 100 to 141,
  `core/rrtmg_legacy.py` 157 to 177); the first also claimed all of its
  differing lines sat inside `RunConfig`, and the measurement says 105 of
  1,130 do.

- The third-party notices this distribution owes. 0.1.0 shipped sixteen CUDA
  sources under `arwen_global/core/kernels/` and four WRF parameter tables
  verbatim, and carried no licence text for any of them: its NOTICE said
  "Nothing from that engine is copied into this distribution" and "This
  distribution ships none of that code and none of those tables", and
  `license-files` named `LICENSE` and `NOTICE` only. Five conditions went
  unperformed. The RTE+RRTMGP transcriptions (`rrtmgp_gas.cu`,
  `rrtmgp_cloud.cu`, `rrtmgp_rte.cu`, and their driver `core/rrtmgp.py` with
  the float64 mirror in `core/npref.py`) are BSD 3-Clause, whose clause 1 asks
  a source redistribution to retain the notice, the conditions and the
  disclaimer. `rrtmgp_mcica.cu` is a sixth file with the same prefix and a
  different owner: the McICA subcolumn generator the radiation is driven with
  is WRF's RRTMG generator, which is AER's work under its own BSD 3-Clause,
  and the same routine reaches `core/rrtmgp.py` and `core/npref.py` as well.
  `core/kernels/glibc_flt32.cuh` carries Arm's logf, expf, exp2f
  and powf under an MIT that asks for its copyright notice and its permission
  notice in every copy, and FDLIBM's expm1f and lgammaf reduction under a
  notice whose one condition is that it be preserved. UCAR asks that its
  notice travel with any copy of WRF, and the four tables are byte copies.
- 0.1.1 performs all five. `licenses/` holds the texts and ships in the sdist
  and, through `license-files = ["LICENSE", "NOTICE", "licenses/*"]`, in the
  wheel at `gpuwm_global-<version>.dist-info/licenses/licenses/`. The notices
  also sit beside the code: `core/kernels/LICENSE-third-party.txt` covers the
  device sources, which cannot take a comment without moving the
  `source_sha256` a run receipt records, and
  `data/noah_tables/LICENSE-WRF.txt` sits with the tables. `core/rrtmgp.py`
  and `core/npref.py` open with the notice for what they transcribe. NOTICE
  gains a section per work, each naming the carried file and the text that
  covers it. Both beside-the-code notices are carried by
  `tools/resync_from_owner.py`, so a re-cut cannot drop them, and the kernels'
  notice is narrowed by rule to the files this package actually ships rather
  than inheriting the source tree's whole-directory version.
- `rrtmgp_mcica.cu` is filed under the work that wrote it. Its own header has
  always said what it is, a transcription of `mcica_subcol_gen_sw` in WRF
  v4.6.1 `phys/module_ra_rrtmg_sw.F`, and WRF preserves AER's copyright notice
  over that module. The first pass at these notices filed it under RTE+RRTMGP
  on the strength of its filename prefix, so the distribution transcribed an
  AER work while shipping no AER text. NOTICE gains an RRTMG section naming
  the kernel, its host driver and its float64 mirror;
  `licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt` and
  `licenses/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt` ship with the other
  texts; the beside-the-code notice covers it in its own section; and both
  inline Python headers carry the AER notice beside the RTE+RRTMGP one. Three
  sentences that were false went with it: the notices said all six `rrtmgp_`
  files cite the upstream commit in their headers, when three do
  (`rrtmgp_validation.cu` is this project's own input validation and
  `rrtmgp_planck_common.cuh` its own in-kernel Planck derivation).
  `tests/test_licence_notices_ship.py` now assigns every carried device source
  to the section that covers it and checks the assignment against what the
  file's own header says it transcribes, so a file cannot be filed by its
  name again.
- Said rather than left silent: the gamma routines at the end of
  `core/kernels/glibc_flt32.cuh` (`gfk_gamma_product`, `gfk_gammaf_positive`
  and `gfk_tgamma`) are this project's own work under its Apache-2.0 licence
  and not a transcription of any C library. `NOTICE` and the notice beside
  the kernels say so, and nothing in that block needs a third-party notice.
  The code of that block is unchanged since 0.1.0; the comment text in
  `glibc_flt32.cuh` and in `gf.cu` that described it was rewritten to say
  whose work it is, so the bytes of both files differ from 0.1.0's and the
  assembled-source digests a run receipt records for the `gf` and `ntiedtke`
  modules differ from the ones 0.1.0 wrote, for that reason alone. The
  `noah` and `morrison` digests move because their code moved.
- Noah's frozen-ground infiltration limiter did not limit, so a frozen column
  soaked up everything that fell on it. WRF's REDPRM builds two quantities,
  `FRZFACT = (SMCMAX/SMCREF)*(0.412/0.468)` and `FRZX = FRZK*FRZFACT`, and
  SFLX passes the second one down. The dummy argument that receives it is
  merely spelled `FRZFACT` at every level below SFLX, and SRT names it back to
  `FRZX` before spending it as `ACRT = CVFRZ*FRZX/DICE`. The carried kernel and
  the carried float64 mirror had both followed the name instead of the
  argument and passed `FRZFACT`, so `ACRT` ran `1/FRZK` = 6.67 times large, the
  `CVFRZ` series in `FCR` saturated at one, and the limit was inert. Both pass
  `FRZX` now, and the carried kernel is byte-identical to the published
  engine's copy again, which carries the same repair. Measured against the
  unmodified WRF driver over the four switch fixtures the port is graded on,
  the surface runoff distance falls from 60,641,303 ULP to 2,812, and soil
  liquid water, relative soil moisture and soil moisture go from 6,508 / 4,729
  / 1,627 ULP to exactly bitwise (RTX 3080, 2026-09-12). On the float64 mirror,
  one wet loam column at 266 K under 5 mm of rain reads FCR 0.9653 before and
  0.0837 after, and surface runoff 4.356 mm before and 4.488 mm after.
- What this moves, and where. SRT reaches the limiter only where the column's
  soil ice clears its own `DICE > 1e-2` threshold, so for an early September
  case the change lives in Antarctica, Greenland, the high Arctic and high
  terrain. Soil moisture, soil liquid water, surface runoff and the surface
  fluxes that follow them move there; a column with no soil ice is unchanged
  bit for bit. Frozen-ground columns move under this repair, so a forecast
  that reaches them is not byte-identical to one 0.1.0 wrote. No shipped page
  quotes a ten-step identity hash; the ten-step comparison of this
  distribution against the model source tree at the revision it carries, the
  309-array acceptance 0.1.0 passed, was not taken for 0.1.1 and is owed as
  the next card job.
- `tests/test_arwen_global_noah_frzx.py` holds both ends. It reads the kernel's
  three call sites and the mirror's two out of the shipped source, reads back
  SRT's own declaration and its `ACRT` line, and runs the mirror on a frozen
  column with the whole SRT infiltration path recomputed by hand from the
  arguments SRT was called with, including the arm that forces the old value
  back in and states what it was worth.
- Morrison's sedimentation stage rebuilt two quantities WRF holds still. WRF
  v4.6.1 `module_mp_morr_two_moment.F` builds them once per level inside its
  column loop and then spends them, unchanged, in the sedimentation block that
  runs after the loop closes: the cloud droplet Stokes coefficient
  `ACN(K) = G*RHOW/(18.*MU(K))` at :1438, frozen above the warm branch's small
  snow and graupel melt at :1504 and :1511 and read by the fall speeds at
  :3440-3441; and the particle-size reference density
  `DUM = PRES(K)/(287.15*T3D(K))` at :3405. That second T3D is not a
  sedimentation-time value: the column loop closes at :3332, the sedimentation
  block runs at :3358-3667, and the microphysics tendencies are applied at
  :3710 below all of it, so nothing writes T3D between the melt and the apply
  and the density :3405 reads is the one the process section's own
  reconstruction used at :1558 and :2182. The carried kernel rebuilt both from
  the temperature it held at sedimentation time, which is the post-process one,
  and the carried float64 mirror did the same. The Stokes coefficient is the
  expensive half: `d ln(ACN)/dT` is -0.288 percent per kelvin at 278 K, so both
  cloud droplet fall speeds were wrong by about that much per kelvin of the
  step's own temperature change, on every cloudy level. The reference density
  was the cheap half, stale by the entry cleanup and melt alone, about 8e-4 K
  and 2.4e-6 relative in PGAM.
- The remedy is the one WRF's own structure names: the process stage publishes
  both level quantities and the sedimentation stage consumes them. It is not
  handing the sedimentation stage its current temperature, which would have
  moved PGAM about 2,400 times further from WRF than the error it removes, in
  the wrong direction. `morr_process_level` writes `G*MRHOW/(18*mu)` from the
  pre-melt viscosity it already computes and `pressure/(287.15*temperature)` at
  the moment of its own reconstruction; `morr_terminal_velocity` takes both and
  rebuilds neither, at the Courant pass and at the sedimentation pass alike.
  The mirror follows: `_np_morrison_apply_level` returns the pair and
  `_np_morrison_fall_speeds` requires it instead of falling back to the air
  density it was handed, which is the third value it used to produce.
- `morr_bound` builds PGAM's reference density from the current temperature.
  This is the engine's own repair, taken with the engine's text, so the two
  copies of `morr_bound` agree again. It reaches the process section's
  reconstruction and, at the final call, the cloud droplet effective radius the
  radiation reads.
- Measured against the unmodified WRF driver, over the 28 oracle columns and
  10,948 compared values the port is graded on, desktop, NVIDIA GeForce RTX
  3080, 2026-09-12: values disagreeing with WRF fall from 3,554 to 3,505. Cloud
  water 228 to 195, cloud ice 333 to 327, accumulated rain 10 to 4 of 28,
  per-call rain 22 to 21, frozen fraction 12 to 9. No field gets worse, and
  every field's worst ULP distance is unchanged, so the three pinned per-platform
  signatures the model source tree holds still hold. On the float64 mirror, one
  900 hPa 285 K cloudy column that warms 4.04 K in a 60 s step reads cloud
  droplet mass-weighted fall speed 0.044494 m/s before and 0.044986 after, and
  number-weighted 0.020406 before and 0.020636 after, both 1.1 percent.
- What this moves, and where. Cloud droplet sedimentation touches every column
  that holds cloud water, so unlike the frozen-ground repair this one is not
  confined to a climate band: it moves wherever cloud water exists and the
  microphysics changes the temperature, which is every precipitating column and
  most cloudy ones.
- `tests/test_arwen_global_morrison_sedimentation.py` holds both ends, and
  ships in this distribution's own suite where the CPU selection runs it. It
  reads out of the shipped kernel source that the sedimentation routine takes
  the two published quantities and rebuilds neither, and that the process
  stage publishes the Stokes coefficient above the melt and the reference
  density below it; and it runs the carried float64 mirror on a column whose
  temperature moves during the step, checking the published pair against
  WRF's own two formulae evaluated at WRF's own two temperatures, and the
  cloud fall speeds against WRF :3440-3441, with an arm that states what the
  sedimentation-time rebuild was worth. The same test is maintained in the
  model's own source tree.

## 0.1.0

The first cut of Arwen Global as a distribution of its own. Everything below
was previously reachable only from inside the engine's own checkout.

### New

- `gpuwm-global`, one console script with fifty-one commands: the forecast door, the
  statics builder, the assimilation door and its five ensemble legs, the render
  tape export, the regional parent bridge, the radiance operators and every
  inspection and validation leg. The engine's `gpuwm global` reached eleven of
  them; the rest were reachable only as a module invocation.
- The machine seam, so a program drives this model the way it drives the
  engine. `gpuwm-global run-plan` reads a `gpuwm.run-plan.v1` plan envelope and
  answers `--catalog`, `--sources`, `--physics-profiles`, `--probe`,
  `--resolve` and `--estimate` as one JSON document each; `gpuwm-global
  sources --json` publishes the same source registry on a door of its own. A
  run writes `run-manifest.json` before it allocates anything,
  `run-progress.json` while it works, and an appended `events.jsonl` whose
  sequence is monotonic; a directory a previous run owns is refused by name
  rather than having a second stream interleaved into its file.
  `python -P -m arwen_global.tui_worker --job-dir DIR -- <command>` spawns any
  command through the same handshake the engine's worker uses. Every schema id
  is the engine's, so a client written for one parses the other, and a
  document that cannot answer an engine field names the field and the reason.
  `docs/ARWEN_GLOBAL_CLIENT.md`.
- The spectral core ships inside the package as `arwen_global.spectral`. No
  published engine wheel carried it, so the model could not be installed at
  all before this cut.
- Semi-Lagrangian semi-implicit core at a 300 s step is the default at every
  truncation: six-point quintic gather, order 16 hyperdiffusion at a 720 s
  e-folding time, off-centring 0.55, Lipschitz gate 0.75. The Eulerian core
  `imex_ssp3` stays selectable by name at the step its own refusal admits with
  margin (90 s at T255, 60 s at T383, 40 s at T533).
- A selectable spectral eddy viscosity as the truncation drain, derived from
  the two-point closure theory of turbulence rather than tuned:
  `[diffusion] closure = "spectral_eddy_viscosity"` reads each level's own
  kinetic energy at the cutoff every step and drains at the eddy viscosity
  that energy implies (EDQNM plateau 0.267, cusp 9.21 at decay 3.03, eddy
  Prandtl 0.6 for the scalars, five tail degrees), applied as the exact
  exponential factor per degree and per level the hyperdiffusion is applied
  as. The exponential hyperdiffusion stays the DEFAULT at every truncation,
  and the closure's six fields join a configuration's identity only when the
  closure is selected, so every record config hash is the hash it was before
  this existed. `arwen_global_gdas_t255_native_closure_24h` is the record
  T255 experiment with its `[diffusion]` table changed and nothing else. The
  theory constants are options so a sweep can measure their sensitivity; at
  their defaults nothing in it is tuned. NOT GRADED ON A CARD: the closure
  arm has no card run of its own, so it ships selectable and unmeasured
  against observations, and the drain of record is the one the graded arms
  ran with. The backscatter term of the same closure is not built.
- Native CUDA physics suite: RRTMGP radiation, Morrison microphysics,
  Grell-Freitas and New Tiedtke convection, YSU boundary layer, Noah land
  surface, MM5 surface layer, run a latitude band at a time.
- The card is priced before anything is allocated: the door estimates the
  device peak, weighs it against free VRAM, chooses the latitude band count and
  the host tier, and refuses a plan that will not fit, naming which allocation
  dies first and what the largest truncation that does fit is.
- Ensemble data assimilation: a 32-member T127 LETKF under the T255 control,
  hourly windows, tapered spectral transfer, RTPS inflation, incremental
  analysis update, and a scorecard with four assessments. Eight observation
  doors, all Rust: surface networks, radiosondes, buoys, satellite motion
  vectors, radio occultation, the WMO information system, GOES ABI and ATMS.
  It ships selectable, not default.
- `gpuwm-global fetch-analysis` fetches the one whole-globe GDAS object a
  global cold start needs, with the two non-default flags bound, and prints
  the run command it feeds. The transport is the engine's Rust fetch route.
- The six source mappings this model reads ship inside the package, and one
  resolver answers every bare id and every mapping file name: the engine's
  authority table first, the carried copies second. A published engine
  carries none of the six, so before this every shipped GDAS experiment
  refused at its first door. `gpuwm-global doctor` prints which table
  answered each row and the SHA-256 of the file that answered it, and a
  mapping both tables carry with different bytes is refused by name rather
  than chosen between. A bare name is a key into those tables, never a file
  in the working directory; a spec that names a path (a directory part, an
  absolute path, or `./name`) opens that file. Every row the resolver answers
  is asked both ways the tables spell it, the file whose name ends at the id
  and the family glob, so one spelling of a row cannot read as a gap on a
  table that carries it.
- The decode scratch is translated rather than refused. This model's decoder
  was called with a `scratch_destination`, which names where several GB of
  float64 frames are staged; a published engine places the same directory by
  `GPUWM_COMPOSE_SCRATCH`. `arwen_global.mapped_source_compat` reads the
  environment, places the directory and restores it inside one lock held
  across the decode, so two decodes take turns instead of staging into each
  other, and each decode's receipt names which of the two placed the
  directory it used. A nested decode names this package's enclosing
  placement rather than crediting a caller who set nothing.
- Every product that decodes a source through the engine records how it was
  decoded, beside the mapping digest it already carried: which mechanism
  staged the decoder's multi-GB frame stream and what placed it, whether the
  installed engine's soil-only `preserve_mask` narrowing had to be adapted
  and for which records, and the engine version that did it. The cold start,
  the ATMS columns, the radiation reference cover, the surface-energy state
  product and the upper-air reference all keep the block. The surface-energy
  flux ladder runs no decode at all, since its records are
  product-definition-template 8 read through the engine's GRIB2 bridges, and
  records how its mapping document was read instead.
- A brightness-temperature pack (`gpuwm-obs.goes-bt.v1`, written by this
  package's `rw_goes bt`) reads on a published engine, whose own schema table
  stops one family short. The container parse is the engine's throughout.
- `gpuwm-global obs subscribe` opens the WMO information system feed through
  its Rust door, archiving every notification and payload with its integrity
  digest and reporting coverage per centre. `obs fetch` for that stream
  refuses and names it: no decoder writes the neutral table yet.
- The LETKF analysis runs on the card by default: batched eigendecomposition of
  the localised solve, point operators contracting a device-resident member
  stack. `--letkf-solve-path host` keeps the numpy reference for comparison.
- GOES ABI clear-sky infrared operator (bands 13 and 8 over water) and ATMS
  clear-sky over-ocean operator (channels 4 to 14), each a registered operator
  entry with an acceptance contract, with a CRTM reference leg and a trainable
  fast model.
- One-way regional parent bridge: a global run exports a parent series that
  drives a regional ArWen forecast, with a validation leg on every artefact.
- Fifty-five configured experiments ship inside the package, from a four-step
  T3 numpy smoke to the 25 km forecast day.
- `tools/build_cli_reference.py` writes the command reference from the parser
  itself and `--check` fails when the committed page falls behind it.
- `tools/measure_boundary.py` measures which engine symbols this package uses
  and whether the installed engine carries them, so the dependency range is
  re-measured at every engine bump rather than transcribed.
- `tools/check_doc_examples.py` checks every documented command line against
  the parser this package ships, and runs in CI beside the reference check.

- The completed ensemble assimilation system: the balance package, the hybrid
  covariance at beta 0.75 over a packaged static table, localisation and
  observation errors measured on the ensemble each window, and the radiance
  streams. `gpuwm-global da fresh` with nothing set runs 32 T127 members under
  the T255 control over every stream the day carries.
- `gpuwm-global da static-covariance` estimates the static covariance table,
  and `gpuwm-global da localisation` measures the localisation length on an
  ensemble store.
- The static covariance table ships inside the package (779,912 B), so
  `--static-covariance packaged` resolves without a source tree.
- Latitude-banded native physics: the suite runs a band at a time and the whole
  advanced bundle is byte-identical to the resident run at every band count.
  The door's plan prices band count and host tier together against the card's
  free bytes.
- The ABI and ATMS radiance tables ship inside the package: the fast model, the
  operator entries and both satellites' acceptance entries.

- The physics this model was graded with travels inside the package, as
  `arwen_global.core`: eleven engine modules, the float64 mirror the
  scorecards grade against, the NVRTC loader, twelve kernels and three headers
  (629,739 B) and the four Noah and land-use tables. Nine of the eleven
  modules differ from published `gpuwm 2.7.0`, and eight of the fifteen
  kernel files do: seven translation units and one header, measured on the
  Windows desktop 2026-09-10 with line endings normalised. `noah` and
  `morrison` match the published engine apart from the carve's import
  rewrites and are carried anyway, because their kernels are two of the eight
  and the loader binds its own directory. So a bare install now
  integrates the same bytes the grading tree did. The receipt's scheme
  identity and the cumulus refusal name those carried files by path, and
  the carve rule that produces the spelling is derived from the carve
  table rather than written out.
- `arwen_global/data/engine-seam.json` pins the 46 engine files the carried
  physics reaches and does not carry, by path, size and SHA-256 at the engine
  version they were measured against, and the suite fails when an engine
  module a carried file imports is in neither place. `gpuwm-global doctor` gains an `engine seam`
  section that hashes the installed engine's copies and reports each as proven
  or moved. Moved is a warning naming the file, and the section's summary
  row reads `note` while anything is unproven: the dependency ceiling is
  the refusal.
- `tools/pin_engine_seam.py` writes that manifest by hashing the installed
  engine, and `--check` compares without writing.
- `tools/resync_from_owner.py` cuts the carried core the same way it cuts the
  model, three ways, and its dry run merges into copies so a re-cut can be
  read before it happens. It gives a merged file the line endings the tree
  stores it with, in both directions.
- `.gitattributes` stops git converting line endings on any platform, and
  `tools/check_line_endings.py` refuses a change that rewrites them: a file
  written back under the other convention diffs whole, so the real change is
  removed and re-added with every other line and nobody reads it. The gate
  prints the CRLF and bare-LF counts on both sides of every flip in a
  revision range and walks the tree for a file that holds both at once.
- A run receipt records which physics module actually integrated, its origin
  and the SHA-256 of the file that was imported, with one digest over the
  kernel directory the loader bound. Two copies of several of these modules
  exist in one process, so a version number cannot answer it.
- A run receipt also carries the seam verdict: the engine version the pins
  were taken against, the engine version that resolved, the proven count and
  any file whose bytes moved. The module hashes cannot see the staying half
  of the physics, and one of those files reaches every carried kernel's
  assembled source.
- `gpuwm-data` is declared as a dependency of this package rather than
  inherited from the engine. Carrying the radiation driver made it an import
  requirement: the driver resolves a member of the companion wheel at module
  scope.

### Fixed

- Four `device_cache_key` imports reached beyond the package's own top level
  and made the installed wheel unimportable.
- The four authority-mapping lookups resolved a path relative to this package
  and named a directory that does not exist once the package left the engine
  tree, so every bare source id in every shipped configuration refused.
- One observation at the exact pole refused the whole analysis. Every surface
  row is evaluated through the lowest level's wind, and a lat-lon vector has no
  direction there; the public surface stream carries a station at latitude
  -90.0000, so the first cycle of a global run stopped. That row is now
  rejected by name, and nothing else is.
- A bare `render` named four products the renderer's catalogue does not carry,
  put every product in the skipped list, exited 0 and left an empty directory.
- The analysis quickstart taught two command lines the parser refuses.
- Thirty-nine written command lines were the console script's name with a
  module name after a dot, which no shell can run. They are `python -m
  arwen_global.<module>` now, and a gate holds the tree to it.
- Three published links named a repository that is not published.
- The stream fetch door reported an empty archive window as the exit code
  reserved for a missing Rust door, so a caller acted on it by restaging
  binaries that were already staged.
- The documented microwave command lines were reported as refused by the
  example checker, which reads one parser and could not see through a door
  that forwards to its own.
- The forecast command wrote no `status.json`. It is the longest-running
  command in the distribution and the only long-running one with no machine
  surface, so a workspace driving a day-long integration had a growing log to
  scrape and no answer to what stage it was in or whether it ended.
- A native forecast against a published engine was accepted, priced the card,
  allocated it, integrated its first steps and then refused its own argument
  vector inside the physics: the published surface layer takes no `vegfra`,
  the published radiation no `column_size_bounding`, and neither published
  cumulus constructor a `column_chunk`. Four refusals stood at the door
  because of it. The carve removes the cause; those four, the YSU
  mixing-length refusal beside them and every skip that cited any of the five
  are retired with it. The last row in that table was the mapped-source
  decoder's `scratch_destination`, which is a placement rather than physics
  and is now translated, so the table is empty: against a published 2.7.0
  there is no call this package makes that the installed engine refuses. The
  measurement stays reachable (`tools/measure_engine_signatures.py`) and a
  row added to the table is still refused at the door by name.
- Selecting `ysu_free_atmosphere_mixing_length = "fixed"` was refused on every
  published engine, because the mode is one more integer argument to the YSU
  kernel. The kernel travels with the package, so the option runs.
- The radiation and boundary-layer scorecards graded against whichever float64
  mirror the installed engine happened to carry, which is not the mirror of
  the kernels this package runs. They grade against the carried mirror.
- A run receipt recorded the configuration, the arithmetic pins, the physics
  identity and the machine, and nothing about the libraries. The spectral
  tables ride on a numpy routine whose bits moved between two numpy releases,
  so a receipt could not answer the first question two differing runs raise.
  It records the interpreter and every installed distribution the arithmetic
  rides on, under the self-hash.
- Two comment lines in the carried physics driver cited files in a
  development-tooling directory on the source tree. Neither path resolves in
  any install, and the source of this distribution ships with it. Both are
  rewritten by the carve rule that reproduces them, and the provenance gate
  now stops that directory name at anything that ships.
- Fourteen test modules set the never-open-the-local-device switch at import
  time. pytest imports every collected module before it runs anything, so one
  CPU-only file decided the device for the whole session: on a card host the
  device selection reported 8 failed and 4 errors, nine of those rows nothing
  but the leak, and under that noise a test that reached an engine module a
  published 2.7 does not carry rode the cut unmarked. The switch is set per
  test through `pytest.mark.cpu_only` and put back, the missing mark is on,
  and two gates read the suite's own source for either shape.
- The engine seam's stated scope read as an import closure. It is the direct
  engine imports of the carried physics plus four files, 46 rows; the closure
  is 109 modules, and the 63 the seam does not pin are reached only through
  runtimes this package's door cannot select. The module and the `doctor` row
  say so.
- A plan naming a render product the renderer does not carry resolved at exit
  0 with no warning, built the statics, integrated the whole forecast and died
  at the render stage naming neither the slug nor anything a client could act
  on. On a T255 day that is a forecast day spent to learn a spelling. Every
  token is checked against the renderer's own product list, its group keywords
  and its skip token, asked through the catalogue door rather than
  transcribed: an unknown slug is a warning in a query mode and a refusal on
  the route that starts work, the parameterized form is reported as unchecked
  rather than unknown, and a machine with no staged renderer says it could not
  ask and names what that costs. The same plan now fails in 1 s with no
  checkpoint written and no picture drawn, and the render stage's own words
  reach the machine channel instead of an exit code.
- A `go` plan with no start date resolved with `render` in its stage list and
  no warning, then finished at exit 0 with the stage silently skipped: a
  reviewer who approved it expecting imagery got a green run and an empty
  directory. Both conditions the stage checks have one spelling now, read by
  the stage and by the resolved document.
- Every `gpuwm.run-plan.resolved.v1` document stated a model top of 0.0 Pa for
  all 55 shipped experiments, because the snapshot read the surface end of the
  half-level ladder instead of its top. It reads the top, and a gate holds it
  against the TOML's own value and against the coordinate's own description on
  every shipped config that loads.
- The checkpoint count was taken from step zero on every plan and always added
  the cold state, while a restart begins at its checkpoint's step and writes no
  cold state. Two T21 restarts, from step 60 and from step 180, both promised
  five checkpoints and wrote three and one, in the resolved document, in the
  estimate's disk block and in the `expected_checkpoints` a client sizes its
  progress bar with. The step is read from the checkpoint's own metadata
  record, the cold state is counted only on a cold start, a restart whose
  checkpoint cannot be read reports no count rather than a wrong one, and the
  restart path joins the declared inputs so a missing one is refused before
  anything starts.
- `run-progress.json` named the second-to-last checkpoint at `status:
  complete` on every run by construction: the final checkpoint lands on disk
  when the writer is joined, after the last heartbeat update. The post-loop
  sweep writes the last committed path, the final model time and the final
  step through the same heartbeat the in-loop callback uses, and the client
  page states that contract.
- The physics-profile menu offered three choices no route can execute: three
  shipped configurations load only through the spectral core's own module
  entry. The document carries one row per shipped experiment naming the door
  it loads through and whether a plan can run it, measured by attempting each
  door's own load, and every profile repeats the subset a plan cannot execute.
- A shipped experiment's bare name resolved on `run` and `go` but not on
  `export`, `assimilate`, `da init`, `da analyze`, `da static-covariance`,
  `da localisation`, `abi-score` or `abi-reference --config`, each of which
  opened it as a file in the working directory. The quickstart's no-card
  rehearsal is three lines and the second is an `export`,
  so a reader with no card met `[Errno 2] No such file or directory:
  'arwen_global_moist_smoke'` on the first thing they could run. Every door
  that takes a config resolves the shipped name now, and a gate reads the
  parser rather than a list of door names, so a door added later is covered by
  existing.

### What a bare install runs

`pip install "gpuwm-global[gpu-cu13]"` on a CUDA 13 runtime, or
`[gpu-cu12]` on a CUDA 12 one. Measured on the Windows desktop 2026-09-10 in
three virtual environments created from scratch, one for each engine version
the declared range admits and the public index carries: `gpuwm 2.7.0`,
`2.7.1` and `2.7.2`, with `gpuwm-data` matched to each. The reading is the
same on all three: 231 symbols across the boundary with no gap the host could
see, the two absent symbols in the last row below, and 46 of 46 seam files
proven although the pins were taken against 2.7.0. A bare install resolves
2.7.2 today.

| Leg | Where it comes from | What it needs |
|---|---|---|
| the spectral core, the dynamics, the transforms, the T3 numpy smoke, tape export and the render call | inside this package | the wheels: `gpuwm>=2.7.0,<2.8`, `gpuwm-data` beside it, numpy, scipy, netCDF4 |
| the physics the model was graded with: radiation, cumulus, surface layer, boundary layer, land surface, microphysics, the land-use rulebook, twelve kernels and three headers, the float64 mirror, the CUDA loader and the Noah tables | inside this package, `arwen_global.core` | nothing on the engine's side; any `2.7.x` engine integrates the same bytes |
| a real forecast at T255 and above | needs a card | CuPy matched to the driver's CUDA major, an NVIDIA card with the free bytes the door prices before it allocates |
| GRIB2 source decode, the observation front door, the static-field builder, the `wrfout` writer, the regrid and the renderer | the engine's own bundle, inside the engine's platform wheel | `gpuwm fetch-bridges` where the wheel carries no binaries (the `py3-none-any` install) |
| the eight Rust observation binaries: `rw_asos`, `rw_igra2`, `rw_ndbc`, `rw_amv`, `rw_gnssro`, `rw_wis2`, `rw_goes`, `rw_atms` | this package's companion bundle, `gpuwm-global fetch-doors` | a published release, or `--from DIR` for a local build; every byte is verified against the SHA-256 pins inside the wheel before it is used |
| the six source mappings the shipped experiments name | inside this package, `arwen_global/data/authorities` | nothing; the engine is asked first for every row and its copy wins the day it publishes one |
| the LETKF filter core, its eigensolver kernel, the local-GPU switch, `gpuwm/core/constants.py` and the rest of the staying half | the engine, pinned by path, size and SHA-256 in `arwen_global/data/engine-seam.json` | nothing; a file whose bytes moved is a `note` naming the file, and the `gpuwm<2.8` ceiling is the refusal |
| `preflight.measured_free_vram_bytes` and `surface_bias.interpolate_to_tape` | absent from every published 2.7 (2.7.0, 2.7.1 and 2.7.2 read 2026-09-10) | an engine that carries them. They stop the standalone card-pricing check, which no command is wired to, and the surface-energy scorecard's regrid onto the tape. `gpuwm-global doctor` prints both and exits 1 |

The observation crates are this package's because no published engine bridge
bundle carries six of them at all, and carries the other two in an older
build without the subcommands this package calls. If the engine takes those
crates onto its own line, `arwen_global/data/door-pins.json` goes empty and
every binary resolves from `gpuwm fetch-bridges`.
