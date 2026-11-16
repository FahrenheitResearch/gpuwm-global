gpuwm-global 0.2.5

Arwen Global 0.2.5 does not reproduce 0.2.4's forecasts. It changes the
Grell-Freitas convection default to a setting that improved most scores
over 32 T255 forecasts, so every config that runs Grell-Freitas has a new
config hash (three keys bring back 0.2.4's convection settings and hash),
and from T383 up the shipped semi-Lagrangian configs use the
departure-point time extrapolation, which removes grid-scale stripes across
the jets. Five fixes change results under unchanged hashes: the convection
schemes now see the vertical motion over terrain, condensation no longer
makes water, convective rain is the water that actually left the column,
coastal land starts from the analysis's land, and lakes are no longer
treated as salt water. It also adds a T1023 forecast day for a 96 GB card
and a second pass for the slab ocean's Q-flux (with the corrected file
for the long-run recipe); a long run on a held sea now lets its land follow
the calendar, and an ERA5 start completed from a second analysis no longer
puts snow on its sea ice. Everything else is off unless a config turns it
on. It needs `gpuwm>=2.9.13,<2.10`, as 0.2.4 did, and the Rust doors are
0.2.4's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Changed defaults

- **A new default, D** (`gf_deep_experiment = 7`, `gf_deep_physics = 34`,
  `gf_momentum_transport = true`). It runs WRF's own convective-memory
  formulas at their saturated value instead of the zero this model passed,
  takes two pieces of NOAA's operational version of the scheme, and couples
  the deep clouds' momentum transport as WRF does. Over 16 autumn and 16
  summer 2026 T255 forecasts to 5 days, against the GFS analysis: 500 hPa
  height at day 1 is better in every region on 14 to 16 of 16 starts;
  tropical 250 hPa wind and sea-level pressure stay better at day 5 (13 of
  16 in both seasons); tropical rain matches GPCP better on every start
  (pattern correlation 0.43 to 0.51, error 10.9 to 7.8 mm/day); the
  tropical upper troposphere loses most of its cold bias; and no score is
  significantly worse beyond day 1. What it costs: the tropics' mean
  500 hPa height and sea-level pressure sit lower (-3.5 m against -0.8 m
  at day 5; the model's tropical surface pressure was already too low,
  and 0.2.4's warm 700-600 hPa layer hid it), southern 850 hPa temperature
  at day 1 is about 0.01 K worse on every start, by day 10 the tropical
  850 hPa layer is 0.3-0.6 K colder than with 0.2.4 on two of three summer
  runs, and the dry central Pacific is still about 1 mm/day too dry. In
  the runner's pictures it removes grid-scale speckle and small spurious
  tropical lows.
- **Keeping 0.2.4's convection:** add `gf_deep_experiment = 0,
  gf_deep_physics = 0, gf_momentum_transport = false` to the config's
  `native_adapter_options`. That gives back the config's 0.2.4 hash
  exactly, so a 0.2.4 checkpoint restarts.
- **The departure-point time extrapolation from T383 up.** The shipped
  semi-Lagrangian configs at T383, T533 and T799 now state `[semilag]
  extrapolation = "settls_departure"` (the T1023 day already did); T255 and
  the option's own default are unchanged. The shipped form painted grid-scale
  stripes across the fastest jets. Over four 120 h T383 forecasts the
  departure form cut that jet noise from about 1.0 to 0.2 m/s on every
  start and the tripwire fires from 9 to 0; tropical 500 hPa height was
  better on 4 of 4 at day 1 and day 5 and northern 500 hPa height on 3 of
  4 at day 5, which four starts cannot make significant; 850 hPa
  temperature costs are small (about 0.01 K), and one start was worse at
  day 5. They run on one card or several: on several, each card takes its
  neighbours' rows of the previous time level once a step, and two cards
  return the one-card run bit for bit. State `extrapolation = "settls"`
  (with the three Grell-Freitas keys above) to get back their 0.2.4 hash.

## Fixed

- **The convection schemes see the ascent over terrain.** They were handed
  the mass flux through the model's hybrid levels, which is zero at the
  ground whatever the flow, as their vertical velocity, so steady flow up a
  slope reached Grell-Freitas, New Tiedtke and the mass-flux scheme with no
  ascent at all. They now get the true pressure velocity (a 1 km ridge under
  a 10 m/s wind gives the expected w = u dh/dx at the ground). Transport is
  unchanged. Every config that runs a convection scheme computes different
  results under the same hash.
- **Grell-Freitas's new default keeps deep convection over land at T383
  and finer.** Its convective-memory setting also shrank a scale factor so
  far that, at T383 and finer, nearly every moist land column was rejected:
  in two T383 days Grell-Freitas made 5-6 % of the land rain, and a few
  grid columns rained about 340-385 mm in a day. The factor keeps WRF's own value
  now; Grell-Freitas makes about half of the land rain and the largest
  daily totals are 140-155 mm. Water columns are unchanged.
- **Condensation no longer makes water.** The native physics works in
  dry mixing ratio and converted back to specific humidity with the vapour
  it returned, so vapour that condensed came back as about q_v times
  itself of new water. Rain then no longer matched what the atmosphere
  lost: Grell-Freitas's convective rain was held below its own water, and
  rain on land pulled the surface reservoir down by about q_v times the rain.
  The conversion back now uses each layer's dry fraction as the physics
  received it, and the convective rain is the water the scheme's rates took
  out of the column on the model's own mass, so nothing is held back: on
  the T255 day the held-back share went from 2.83 percent of what
  Grell-Freitas reported to 0.0006 percent (round-off), and the lowest land
  reservoir from 494.0 to 496.9 kg/m2. A call whose rates and column loss
  disagree beyond round-off is refused. No config hash moves; every
  native-physics run computes slightly different water.

- **A long run on a held sea no longer freezes its land.** With
  `[ocean] mode = "hold"` the vegetation, leaf area and background albedo
  stayed at the start date, so a run from 1 March kept March vegetation
  under a May sun. A runner project prepared before the long-run recipe
  holds its sea this way. A dated run of more than 16 days now follows the
  monthly land climatology under `hold` too, by default; a forecast of 16
  days or less keeps the start, as before. This fix moves no config hash, and a
  held run checkpointed under 0.2.4 picks up the calendar on its next
  restart: from ERA5 1976-03-01, a 0.2.4 held checkpoint late in a 75-day
  run restarted under 0.2.5 and its northern mid-latitude vegetation
  fraction went from 0.236 (1 March) to 0.342. The weaker late-April jet
  seen with the fix on two starts comes from one run per arm, so it is not
  claimed. `[land] greenness = "start"` keeps the old behaviour; on such a
  run it now has its own config hash.
- **An ERA5 start completed from a second analysis kept its sea ice
  bare.** ERA5 has no snow on its sea ice, and the model knows it, but the
  merged frame built for `analysis_fill_grib` lost that, so the glacier and
  coast snow beside the ice could be spread onto it. Found in review; GDAS
  starts are unchanged.

- **Coastal land starts from the analysis's land.** At a cold start a
  coastal land column's soil and skin were regridded over land and sea
  points alike; on the T255 start the 3,999 such columns started with
  1-2 m soil about 1.6 K warmer on average (up to 15 K). They now read only
  the analysis's land points, or the nearest one. Inland columns and the
  sea are unchanged.
- **Lakes are not salt water.** The surface layer gave lake columns the
  sea's 0.98 salt factor on evaporation; it now knows them as lakes.
- On another world with its own sun, the light between radiation calls
  followed Earth's sun; it now follows the world's.

## New

- **T1023 on a 96 GB card** (`arwen_global_gdas_t1023_native_sl_si_24h`,
  about 13 km). On one RTX PRO 6000 Blackwell it ran its full day in
  9.57 s a step (46 minutes a model day), with a card peak of 89,350 MiB
  of 97,887, one latitude band and nothing parked on the host. It uses the
  departure-point time extrapolation (`settls_departure`): with the shipped
  `settls` the same run grew grid-scale stripes across the strong cold-air
  flow behind a NE Pacific low within 3 hours, and the departure form
  removed them through 24 h. From 9 h a field of grid-scale shower cells
  appears in that cold air; that is still open. It also smooths its
  model terrain (`[initial] orography_filter_half_degree = 350.0`): the
  unsmoothed GDAS terrain put a ripple on the 500 hPa height over the
  western mountains, and in 12 h runs the filter brought that roughness
  from 3.0-4.2 times GFS's to 0.96-1.22 times, at the cost of 270 m off the
  highest peaks in Nevada, Utah and Idaho. Setting it to `0.0` gives back
  the unfiltered terrain and the day's hash before this. From 0.2.5 the T533
  day and wall, the T799 wall and the T1023 day also run the orographic
  drag the T255 and T383 days already did (`orographic_drag =
  "kim-gwdo"`): at T799 over 12 h it lowered northern 500 hPa height error
  from 11.57 to 10.51 m and the jet box wind error from 5.22 to 4.60 kt,
  with no measurable time or memory cost. The run door now admits
  T799 and T1023 on a 96 GB card at the plans that were measured, and
  prices every other shape as 0.2.4 did (the T383 day on a 16 GB card:
  14.27 GiB of card in both).
- **Less memory per step, same bits.** The first physics half now works in
  place on a forecast, the step drops its input state once it is used, and
  a gather of one field is no longer copied. At T1023 that is what lets
  the day fit at one band.
- **A second pass for the slab Q-flux** (`gpuwm-global climatology qflux
  second-pass`). A six-day calibration books only its first week, and the
  slab ocean then drifts cold over months. The second pass corrects the
  file from two long slab runs that read it. From three starts it never
  saw, 59 days each, it kept the ocean much closer to its start than the
  first file: 0.076 against 0.242 K lost from 1 November, 0.073 against
  0.152 K from 1 May, and both held from 15 January. One pass does not
  remove it all (the May and November oceans still lack 3-4 W/m2 in their
  second month). `--skip-days N` leaves each run's first days out; it made
  no measurable difference here, so the recipe does not use it.

## New, off unless you turn it on

- **Newton departure points** (`[semilag] trajectory_solver = "newton"` or
  `"newton_ls"`). The shipped fixed-point search can stall in strong
  rotation or vertical shear; Newton solves the same equation with its own
  Jacobian, and `"newton_ls"` adds a line search so a point that overshoots
  once is not stuck. Card only. The T1023 day uses `"newton_ls"`: with the
  orographic drag on, the plain Newton refused that day's first step. At
  T383 with a 600 s step, newton_ls retried none of 288 steps on two
  starts, where the shipped search retried 251 and 285.
- **A terrain filter** (`[initial] orography_filter_half_degree = n0`,
  experimental). It smooths the model terrain spectrally, to half
  amplitude at total degree n0, before the surface pressure moves onto
  it. Off (`0.0`) by default, and off it leaves every config hash as it
  was. The semi-Lagrangian T533, T799 and T1023 days turn it on at 350
  (above).
- **A denser snow pack at a GDAS cold start**
  (`[initial] analysis_snow_density = "wrf-floor"`). GDAS writes a snow
  depth that makes most packs as light as fresh snow, which insulates the
  ground and starts northern land too cold. The floor starts such packs at
  WRF's 200 kg m-3. Over nine T255 120 h forecasts it moved northern land
  2 m temperature toward the analysis on every start, but improved its
  error only at day 5 (by 0.03 K), left 500 hPa height neutral, made
  northern sea-level pressure very slightly worse (better on only 1 of 9
  starts at 1 to 3 days), and in December it makes Canada too warm (+1.1 K
  at 72 h from one start). So it is not a default.

## For long runs

- **The recipe's second-pass Q-flux is attached to this release**
  (`qflux-t127-r5-sp1.npz`), with the 90-day T127 slab config that names
  it (`recipe-slab-t127.toml`). The file was calibrated with 0.2.4's
  Grell-Freitas, so the template states the three 0.2.4 values; without
  them the slab's guard refuses the file, as it refuses any calibration
  made with other physics. Build the climatology it names with
  `gpuwm-global climatology fetch --with-mld` and
  `gpuwm-global climatology build arwen_global_gdas_t127_native_sl_si_24h --with-mld --out climatology-t127-mld.npz`.
  On 0.2.5 the guard accepted the file for that template. The guard binds
  the file to the physics options, not to the code: the 0.2.5 fixes that
  change results under unchanged hashes (above) also change the arithmetic
  the file was calibrated with, which no run has measured yet. The first file
  and its template stay on the v0.2.4 release; under 0.2.5 that template
  needs the same three lines.

## Measured for this release

- **The release, on the 16 GB card** (RTX 5070 Ti, alone, 24 h, the
  shipped configs by name): the T255 day passed with no tripwire at 0.79 s
  a step and a card peak of 8,484 MiB; the T383 day passed with no
  tripwire at 1.65 s a step and a card peak of 12,808 MiB of 16,303.
  The T1023 day as it ships passed its full day on a 96 GB card with no
  tripwire and no retried step, at 8.58 s a step, a card peak of 88,777
  MiB of 97,887 and a host memory peak of 38.3 GiB.
- **Hashes.** Every config that does not run Grell-Freitas hashes as in
  0.2.4; every one that does is back to its 0.2.4 hash with the three keys
  above (and, from T383 up, `extrapolation = "settls"`). The results are not
  0.2.4's even then: the fixes above change them under the same hashes.
  Before those fixes and the convection change, the T255 day under 0.2.5
  wrote 0.2.4's nine checkpoints byte for byte, a restart from a 0.2.4
  checkpoint continued 0.2.4's run byte for byte, and the T383 day wrote
  0.2.4's five checkpoints byte for byte.
- **Less memory.** Before the convection change the T383 day's card peak
  fell from 14,220 to 12,964 MiB and the T255 day's from 9,110 to 7,806
  MiB against 0.2.4, with the same bits.
- **The second-pass Q-flux.** On 0.2.5 the slab's guard accepted
  `qflux-t127-r5-sp1.npz` for the attached `recipe-slab-t127.toml` (CPU
  check, no run).
- **Tests.** 3629 CPU tests pass from the installed wheel, with CuPy (95
  skipped) and without (112 skipped).

## Upgrading

- A config that runs Grell-Freitas (every shipped forecast config does)
  computes with the new default D and has a new hash. To keep 0.2.4's
  convection, or to restart a 0.2.4 checkpoint, add `gf_deep_experiment =
  0, gf_deep_physics = 0, gf_momentum_transport = false` to its
  `native_adapter_options`. A slab or dynamic-ocean run whose Q-flux was
  calibrated under 0.2.4 needs the same three keys, or the guard refuses
  the file.
- Every config that runs the native physics computes different results
  from 0.2.4 under its own hash (the convection, water, coastal and lake
  fixes above), with or without the three Grell-Freitas keys. A 0.2.4
  checkpoint whose hash still matches restarts, and continues with 0.2.5's
  arithmetic; it is not the run 0.2.4 would have continued.
- A held run of more than 16 days computes something new from 0.2.5 on:
  its land follows the calendar. Set `[land] greenness = "start"` to keep
  0.2.4's land; a 0.2.4 checkpoint of such a run then restarts only with
  `--branch-from-other-config`, because the stated `"start"` now has its
  own hash.
