# Long runs: the ocean rule, the land march and the long-run outputs

A forecast of months from one initial state needs a lower boundary for
months. The model has no ocean below the surface, so the sea-surface
temperature and the sea ice of a long run are prescribed by a rule the
config names, and the receipt says which rule produced every record. Nothing
here predicts the ocean; a quantity is either predicted by the model's
equations, prescribed by a stated rule, or held at its starting value, and
this page says which.

| class | after this change |
|---|---|
| predicted | atmosphere; Noah soil temperature and water, snowpack (with its age, under `snow_ageing = true`), canopy water; inland-lake skin; sea-ice column temperature |
| prescribed by a stated rule | ocean SST and ocean sea-ice fraction (`[ocean]`, `[sea_ice]`); vegetation fraction, leaf area index and background albedo on land (`[land] greenness`, monthly climatology by date); ozone (CAM monthly); CO2, CH4 and N2O (the start year's NOAA GML means, or each call's valid year under `[radiation] ghg_year = "valid"`); solar geometry |
| held at the start | deep-soil temperature; the thickness of ice that exists at the start (new ice takes `new_ice_thickness_m`); lake ice fraction (a lake never freezes or thaws); aerosols (none); the ocean, under `mode = "hold"` |

## The ocean switch

```toml
[ocean]
mode = "anomaly"            # hold | anomaly | climatology | series | slab (docs/ARWEN_GLOBAL_OCEAN.md)
climatology = "/abs/path/arwen-global-surface-climatology-T255-384x768-default.npz"
anomaly_reference = "mean-7d"          # start | mean-7d
anomaly_reference_series = "/abs/path/reference-2026111000.npz"
anomaly_damping_days = 90.0            # poleward of the tropical band
tropical_anomaly_damping_days = "none" # "none": persisted, undamped
tropical_band_deg = [10.0, 30.0]       # half-cosine ramp of the rate between them
tropical_persist_region = "pacific"    # pacific (120 E to 80 W) | global
large_scale_offset = "persist"         # persist | damp
climatology_ramp_days = 3.0            # climatology mode
anomaly_truncation = 42                # "none" keeps the grid field

[sea_ice]
mode = "match"                 # match | hold | anomaly | climatology | thermodynamic (slab only)
anomaly_damping_days = 60.0
record_interval_hours = 24.0
new_ice_thickness_m = 1.0
```

The climatology file is built once per grid with
`gpuwm-global climatology fetch` and
`gpuwm-global climatology build CONFIG` (NOAA OISST v2.1, 1991-2020 monthly
mean). The anomaly reference is the 28 six-hourly GDAS analyses of the 7
days before the start, built with
`gpuwm-global surface-series CONFIG --source gdas-analysis --cycle 2026111000 --anomaly-reference --out reference.npz`,
where `--cycle` is the run's start. Its open-water rule reads an analysis
point as water only where it is neither land nor ice, so an ice skin never
enters an open-water SST. `anomaly_reference = "start"` takes the anomaly
from the cold-start skin instead and needs no reference file.

Write both paths in full. A relative path is read from the config's own
folder, and the copy of the config a run keeps in its output folder would
then point somewhere else.

Any of hold, anomaly, climatology and series may also carry the
experimental, off-by-default mixed-layer cooling response
(`mixed_layer_response = true`): a departure from the prescribed SST that
answers the surface heat flux into a fixed mixed layer, so a held sea can
cool under days of dry offshore wind instead of evaporating at its first
rate (docs/ARWEN_GLOBAL_OCEAN_RESPONSE.md).

### The rules

Write t for the elapsed time, C_s(t) and C_i(t) for the climatological SST
and ice fraction (linear in time between mid-month values, real month
lengths, leap Februaries included), and T_f = 271.35 K.

- The starting anomaly is A_0 = S_0 - C_s(0) on columns that are open ocean
  at the start. S_0 is the reference mean or the cold-start skin, held at or
  above T_f: a colder open-water value is an ice skin or a blend of ice and
  water skins, not an SST. On the GDAS analyses of 2026-11-03 to 11-10 the
  reference read 252 K on open-water columns near 76 N, and the cold start's
  own skin read 241 K on columns beside the ice near 82 N. A_0 is 0
  on columns that start as ice, on lakes, and where the climatology or S_0
  has no value. Those columns take the climatology.
- A_0 = Abar + A'. Abar is the area mean of A_0 over those columns, on the
  grid's Gaussian quadrature. A' is the rest, projected onto spherical
  harmonics up to T`anomaly_truncation` and set back to zero where A_0 is
  zero.
- The damping rate is r = r_x s + r_t (1 - s), with r_x =
  1/`anomaly_damping_days`, r_t = 1/`tropical_anomaly_damping_days` (0 for
  "none"), and s rising as a half cosine from 0 at the band's inner
  latitude to 1 at its outer one. Under `"pacific"` the tropical rate holds
  between 120 E and 80 W, with a 10 degree half-cosine taper outside them,
  and r = r_x elsewhere.

| mode | open-ocean SST |
|---|---|
| `hold` | not written: the cold-start SST for the whole run |
| `anomaly` | max(T_f, C_s(t) + Abar g + A' exp(-r t)), with g = 1 (`persist`) or exp(-r_x t) (`damp`) |
| `climatology` | max(T_f, C_s(t) + (Abar + A') exp(-t / ramp)) |
| `series` | linear in time between the dated records of `[surface_update] series`, refused past the last record |

The sea ice is replaced at every record, t_k = k `record_interval_hours`, on
ocean columns (land fraction at most one half, not a lake, in the
climatology's ocean):

- the fraction is clip(C_i(t_k) + (I_0 - C_i(0)) exp(-t_k / tau), 0, 1),
  with tau = `[sea_ice] anomaly_damping_days` in anomaly mode and the
  climatology ramp in climatology mode;
- the fraction is 0 where the column's open-water SST is at or above
  T_f + 2 K;
- a column that becomes ice takes `new_ice_thickness_m`, and one that stays
  ice keeps its thickness.

A column that crosses one half is re-classed by the cold start's own
rulebook, as a series record does. `match` gives the ice the ocean's mode.

What a rule cannot do: none of them grows, ends or moves an El Nino. Anomaly
mode keeps a starting event at full strength in the tropical Pacific while
the climatology underneath advances with the seasons. The longer the run,
the more likely it holds an event that has really ended. Climatology mode
is a 1991-2020 ocean, cooler than recent years by the global offset Abar
that anomaly mode keeps. The rules' form is the one operational centres use
for forecasts of 16 days or less (GFS damps the observed anomaly with a 90
day e-folding time; the IFS persisted it). That gives the form of the rule,
not evidence that it works over months.

Lakes are never written by a rule: their skin follows its own energy budget
(a 10 m slab floored at 273.15 K), and their ice stays as it starts.

### Which runs need a table

A run with no `[ocean]` table keeps its earlier behaviour and its config
hash: the series when `[surface_update] series` is set, the held SST when it
is not. A dated run from an analysis on real statics that is longer than
384 h, with no table and no series, is refused before it starts, with a
message that names the modes. `mode = "hold"` states the hold and runs at
any length. Idealized and undated runs are not refused. `slab`, a mixed
layer with a heat-flux correction and the `thermodynamic` sea ice that goes
with it, and `qflux_calibration`, the climatology run its correction is made
from, have their own page, [ARWEN_GLOBAL_OCEAN.md](ARWEN_GLOBAL_OCEAN.md).

### Identity and restarts

- The tables enter the config identity only where they change arithmetic.
  `mode = "hold"` with held ice and `mode = "series"` leave every hash as it
  was. The climatology and the reference series enter by content SHA-256,
  never by path, so moving a file moves no hash.
- Every segment rebuilds the rule from its own cold state, so the starting
  anomaly of a restarted segment is the straight run's. The checkpoint
  carries the rule's identity block (`physics_metadata["surface_boundary"]`:
  the modes, their parameters, the climatology's SHA-256, the start and the
  SHA-256 of the starting anomaly). A restart whose block differs from the
  rebuilt one is refused before any of its arrays becomes the run's state.
- A held run can be moved onto a rule from one of its checkpoints:
  `gpuwm-global run RULE_CONFIG --restart RUN/arwen_global_stepN.npz
  --branch-from-other-config`. The rule is rebuilt from the run's cold
  start, so the branch's first step writes the SST a straight rule run
  writes at that instant; the receipt's `branch.surface_boundary` names the
  parent's boundary and the branch's, and the branch's checkpoints carry
  the branch's, so its own restarts continue it. The same works between
  hold, anomaly and climatology. A boundary with prognostic arrays (the
  slab, a calibration) cannot be branched onto. Measured on the 50-day T383
  run of 2026-11-12 (held sea): from day 25 under `mode = "anomaly"`,
  `anomaly_reference = "start"`, the SST off Tampico read 24.4 C against the
  held 26.25 C and the day's largest open-water evaporation fell from 55.0
  to 46.6 kg/m2 (day 26-27: 52.1 to 44.6).
- The rules are closed forms in time, an ice record re-applied after a
  restart is the identity, and the greenness march is anchored at the model
  day's start. So a restarted run reproduces the straight run bit for bit.
  `tests/test_arwen_global_ocean_rules.py` holds a reference-physics smoke run
  to that, from a start at 11:59:20 UTC, restarted at January's mid-month
  instant, inside an ice record.

### The receipt and the rows

- `surface_boundary` in the receipt: the modes, every parameter, the
  climatology's path, SHA-256 and inputs, and the starting anomaly. The
  anomaly is summarized as Abar, the Nino-3.4 mean of A_0, the tropical and
  extratropical means of A', the columns given a zero anomaly and why, and
  the root-mean-square and largest change of the first application. It also
  carries `future_surface_records: 0`.
- `surface_update` in the receipt: the counts of SST steps, ice records and
  crossings.
- At every output step, `diagnostics.jsonl` gains:
  - the climatology months and weight;
  - the damping factor at 0, 20 and 45 N (200 E);
  - the open-ocean mean SST, and its global, tropical and extratropical
    mean departure from the climatology;
  - the Nino-3.4 departure;
  - NH and SH ice extent and area at 0.5 (the model's ice) and at 0.15;
  - the largest |SST - rule| on open ocean
    (`boundary_max_abs_sst_departure_k`) and the ocean columns whose ice
    differs from the last record (`boundary_ice_record_mismatch_columns`).
    These last two are gate LR-8's rows.

## The greenness march

`[land] greenness = "daily"` sets the vegetation fraction, leaf area index
and background albedo on land to the statics' monthly climatologies once a
model day. `"start"` keeps them at the start. The default is daily under a
rule or a series and the start under `hold`, and the key enters the
identity only where it says something the mode would not do.

The date read is the model day's start. Until this change the march read
the instant of the first step it ran on, so a run started at 06, 12 or 18
UTC and restarted inside a model day took another UTC day's vegetation than
the straight run. On a field of ten times the month number, a run from
2026-04-03 18 UTC read 26.452 restarted at 12 h against 26.129 straight.
`tests/test_arwen_global_greenness_march.py` holds the march to
`monthly_interp_to_date` at the anchor and the 6 h and 12 h restarts to the
straight run.

## Long-run outputs

These change no arithmetic and are outside the identity.

```toml
[output]
keep_checkpoints = "newest:2+month-ends"   # all | newest:N | newest:N+month-ends
mean_state = "monthly"                     # none | monthly | daily | monthly+daily
daily_summary = true
disk_reserve_gb = 20.0
```

### Retention

After each checkpoint is handed to the writer, the durable checkpoints
beyond the newest N are deleted with their sidecars. Two kinds are kept:
under `+month-ends`, the last checkpoint of every calendar month, and the
checkpoint the segment restarted from. So at most N durable checkpoints and
the one being written are on disk during a segment. When something
outside the run exports a tape of every checkpoint and deletes the
checkpoint after it, leave retention at `all`.

Before a checkpoint is deleted, its header goes into
`arwen_global_pruned_checkpoints.jsonl` beside the checkpoints, one JSON
object per line: `checkpoint`, `bytes`, `step`, `time_s`, `valid_utc`,
`self_sha256`, `config_hash`, `pins_hash`, `schema`,
`physics_state_schema`, `run_trackers` and `array_sha256` (every array's
SHA-256 by name). A replay of any window can then be checked against the
state the run held there, byte for byte and array by array. A header that
cannot be read is recorded as `unreadable`, not skipped.

Every run receipt also carries `software_stack`: the package version and
where it was imported from, the interpreter and platform, the recorded
libraries, every CUDA user-space wheel (`nvidia-*`, `cuda-*`,
`gpuwm-kernels*`), and on the CuPy backend the card's name, compute
capability and memory, the CUDA runtime, driver-API and NVRTC versions and
the NVIDIA driver version from `nvidia-smi`. A field that cannot be read
says so.

### Mean state

At every output step, a float64 running sum is kept for each floating
atmosphere and surface array and for the physics namespace's 2-D planes, in
the calendar window of the step's valid time. When the next window starts,
the closed one is written as `arwen_global_mean_<YYYY-MM>.npz` or
`<YYYY-MM-DD>` in the checkpoint layout, with
`physics_metadata["time_mean"]` naming the window and its samples. Every
other array is the last instant's.

A mean file is refused as a restart. Its exported tape carries
`ARWEN_TIME_MEAN`: a field the exporter derives nonlinearly from these
means, such as a wind speed, is a function of means, not a mean of the
function. Export mean files into a folder of their own, because their
tapes are named by the last sample's valid time.

The open windows' sums ride beside each checkpoint in
`<checkpoint>.longrun.npz`, so a restarted run writes the straight run's
mean files (`tests/test_arwen_global_longrun_ops.py`). A window the run
ends inside is written with the samples it has.

### Daily summary

`arwen_global_daily_<YYYY-MM-DD>.npz` holds one UTC day:

- 2 m temperature mean, minimum and maximum over the output instants;
- 2 m humidity and 10 m wind means;
- outgoing longwave mean;
- skin temperature mean (the SST on open ocean), sea-ice fraction mean and
  snow cover mean;
- the day's precipitation: the surface buckets plus the cumulus
  accumulator, the last instant of the day less the last of the day before.

The planes below are made on the host from the same checkpoint arrays,
through a numpy synthesis of the run's own grid. Means are over the day's
output instants unless a plane says otherwise.

| plane | units | what it is |
|---|---|---|
| `surface_stress_eastward_mean_n_m2`, `surface_stress_northward_mean_n_m2` | N/m2 | the force on the surface, positive eastward and northward along the wind: rho u*^2 along the 10 m wind, rho = ps / (R_d T2 (1 + 0.608 q2)) |
| `sea_level_pressure_mean_pa`, `sea_level_pressure_min_pa` | Pa | the scorecard's reduction of ps over the terrain with T2 |
| `vorticity_850hpa_mean_per_s` | 1/s | relative vorticity at 850 hPa, linear in ln p between full levels |
| `vorticity_850hpa_cyclonic_max_per_s` | 1/s | the day's largest vorticity times the sign of latitude (cyclonic is positive in both hemispheres) |
| `pressure_850hpa_below_ground_fraction` | 1 | the share of instants at which 850 hPa lay below the lowest full level; there the vorticity is that level's |
| `sensible_heat_flux_mean_w_m2`, `latent_heat_flux_mean_w_m2` | W/m2 | HFX and L_v QFX, upward positive |
| `liquid_water_path_mean_kg_m2`, `ice_water_path_mean_kg_m2` | kg/m2 | int qc dp/g and int qi dp/g (cloud water only) |
| `toa_incoming_shortwave_w_m2`, `toa_reflected_shortwave_w_m2`, `toa_absorbed_shortwave_w_m2`, `toa_outgoing_longwave_w_m2`, `toa_net_w_m2` | W/m2 | exact interval means of the radiation accumulators |
| `surface_downward_shortwave_w_m2`, `surface_upward_shortwave_w_m2`, `surface_downward_longwave_w_m2`, `surface_upward_longwave_w_m2`, `surface_net_radiation_w_m2` | W/m2 | the same, at the surface; the net is downward positive |
| `cloud_cover_fraction` | 1 | the cloud-cover accumulator's interval mean |

The radiation planes cover the precipitation's interval: from the last
instant of the day before to the last instant of the day. A run's first day
has neither, and its metadata says so. Clear-sky fluxes are not written,
because the radiation holds no clear-sky planes and a second radiation
pass would be neither cheap nor outside the step.

Each file's `metadata` gives every plane's rule, the radiation interval
and `global_means`. That block holds the area mean of every plane, plus the
planetary albedo (reflected over incoming), the surface's net energy (net
radiation less H and LE) and the precipitation in mm/day. Z500, T850, U250
and V250 are not in the summary.

### The angular-momentum row

Every diagnostics row carries `global_mean_relative_angular_momentum_kg_s`
(int u a cos(phi) dp/g) and `global_mean_axial_angular_momentum_kg_s` (the
relative part plus int Omega a^2 cos^2(phi) dp/g). Both are global means
per unit area, with the insitu ledger's definitions. Gate LR-10 reports
them as the planet's totals in kg m2/s: each day's mean, the least-squares
drift of the daily means per day over the run and per 30-day window, and
each window's change. It has no threshold yet.

### The launch estimate and gate

A run that states any of these outputs, or runs longer than 384 h, prints
its disk and wall estimate and is refused when the disk estimate exceeds
the free space of its output folder less `disk_reserve_gb`. The disk
estimate is an upper bound built from these parts:

- checkpoint bytes scaled by grid area and levels from the one measured
  T383 L40 native checkpoint (1,522,236,583 B), about 0.68 GB at T255;
- mean files as large as a checkpoint;
- sidecars at twice a checkpoint per open window;
- the daily summary's planes in float32.

The wall estimate is the stepping time at a planning rate per model day,
4.3 min at T255 and 12 min at T383 (none at other truncations), with the
cold build and the export not counted. It is a planning figure, not a
bound: the 10-day T255 run below stepped at 3.8 min per model day, and
its two 5-day segments spent 1.2 and 3.1 min more on the cold build, the
diagnostics and the outputs.

The receipt's `long_run_outputs` records the estimate, the retention and
the products written.

## Launching a long run

```sh
gpuwm-global climatology fetch
gpuwm-global climatology build CONFIG
gpuwm-global surface-series CONFIG --source gdas-analysis --cycle 2026111000 --anomaly-reference --out reference.npz
gpuwm-global run CONFIG --outdir RUN --until-s 432000
gpuwm-global run CONFIG --outdir RUN2 --restart RUN/arwen_global_step00001440.npz --until-s 864000
```

The config is one with an `[ocean]` table as above, a `duration_s` of the
whole run, and an `[output]` table. Each segment is one
`--restart ... --until-s` call, and each segment rebuilds its cold state
from the analysis, so the analysis file must stay where the config names it
for the whole run. The step number of a restart file is `t / dt_s`.

`tools/arwen_global_longrun_chain.py`, in the source distribution, runs
these segments for you as a chain of leases (`start RUN_DIR --config
CONFIG --lease-days 5`), resumes it with the same command after a stop or
a crash, and appends one `STATUS` line per lease; its docstring lists the
commands.

### The native options a long run sets

A long run, and a coupled run (`docs/ARWEN_GLOBAL_OCEAN.md`), also sets
six native options that a forecast leaves off, in `[physics]
native_adapter_options`:

```toml
native_adapter_options = { ..., snow_ageing = true, morrison_droplet_number = "land_ocean", radiation_liquid_radius = "wrf_relcalc", gf_shallow_buoyancy_from_cloud_base = true, gf_shallow_blqe_to_cloud_base = true, gf_shallow_depth_min_m = 500 }
```

and the three conservation switches of the next section. The forecast
configs that ship leave all of these off except one: from 0.2.3 the T255 and
T383 forecast days carry `orographic_drag = "kim-gwdo"`, and a long run built
from one of them keeps it (the 90-day runs of `docs/trade-shallow.md` ran
with it).

The last three are the Grell-Freitas shallow-arm switches of
`docs/trade-shallow.md` (0.2.2, off by default). They vent the trade-wind
cloud sheet the model reflects too much sunlight from. In the 90-day
atmosphere-only arm from GDAS 2026-08-01 00Z (anomaly SST, kim-gwdo, the two
cloud-radiation options above) the global excess of reflected shortwave over
CERES read +3.5 / +5.4 / +4.5 W/m2 in August / September / October against
+15.0 / +15.9 / +14.2 without them, and the net at the top of the
atmosphere -3.8 / -0.2 / +4.0 against -13.9 / -8.5 / -4.6 W/m2. In the
coupled slab arm, on the Q-flux calibrated without them, the open-water heat
flux less the calibration's read +3.1 W/m2 over days 2-30 and -3.5 over days
30-60 against -8.8 and -12.4, and the open-water SST against GDAS at day 59
-0.07 K against -0.30 K; part of that gain is the extra sunlight acting on a
Q-flux tuned for less of it, and a calibration made with the switches on has
not been run.
They stay off for forecasts, because over the 16-start forecast scorecard
they cost tropical skill at 72-120 h (T850 +0.04 to +0.08 K, 250 hPa wind
+0.22 to +0.24 m/s); and in the southern summer the depth minimum does not
keep the Peru and Australian decks (a 30-day December twin: Peru -28.8
against +8.5 W/m2 of excess). The page has the tables.

The shallow-arm switches and the conservation switches below were each
measured on their own; a long run with both sets on has not been run.

`snow_ageing = true` carries Noah's snow age between land calls, so a pack
that lies for weeks darkens as Noah intends instead of keeping its
fresh-snow albedo (`docs/ARWEN_GLOBAL_LEVEL5.md`, "Snow age"). It is off by
default, so a shipped forecast keeps 0.1.8's arithmetic and hash; on, it
joins the config hash. The 90-day runs of the 0.2.0 release notes ran on
a tree where it was on by default, so a config that reproduces them sets
it.

The other two are cloud-radiation options of `docs/albedo-fixes.md`.

The first is a land and an ocean cloud droplet number (300 and 100 per
cm3, MPAS's values) in place of WRF's single 250; the second radiates
Morrison's liquid at the radius WRF's own RRTMG gives it. The third option,
the 0.7 in-cloud path factor, stays off. The package default of all three
is off, so a config that names none of them runs, and hashes, as it did.
Without them the model reflects too much sunlight for a run of months: the
first ten-day run below lost 21.7 W/m2 at the top of the atmosphere at a
planetary albedo of about 0.39, against CERES-EBAF's 0.30 for November.
Why these two and not the third is in `docs/albedo-fixes.md`, "The
long-run set"; what they did over ten days is below.

### The conservation switches a long run sets

Three `[repair]` keys hold what the dynamics does not conserve on its own.
Each is off by default, joins the config hash only when set, and leaves a
run that names none of them byte-identical to the release before them. A
long run and a coupled run set all three (0.2.3): with them LR-4's residual
read 0.24 and 0.43 W/m2 over days 0-30 and 30-60 of 60 T127 days and -0.29
W/m2 over 10 T255 days, under the 2 W/m2 at which the gate flags it, where
it read 4.1 to 5.4 W/m2 without (numbers below). A forecast leaves them off:
on two five-day T255 starts the skill was neutral, and the measurements are
of long runs.

```toml
[repair]
mass_fixer = true
water_fixer = true
positivity_repair = true
mass_fixer_target = "dry"
energy_fixer = true
vapour_fixer = true
```

- `mass_fixer_target = "dry"`: the mass fixer holds the global-mean dry
  surface pressure ps - g W instead of the total. The physics books water
  at a fixed surface pressure, so under the total target every kilogram
  per square metre the atmosphere's water gains or loses moved the dry
  mass by g Pa: LR-1 read 60 Pa over the T127 year run, all of it the
  swing of the planet's precipitable water. The receipt's mass-drift gate
  reads the row the fixer holds. That row is read after the second physics
  half, which books its half step's E - P at a fixed surface pressure, so
  at an output instant it sits g (E - P) dt/2 off the target until the next
  step re-anchors it: at most 1.33e-6 of the mass over 60 T127 days and
  1.23e-6 over 10 T255 days (the 6-hourly dry-mass rows of every lease's
  `diagnostics.jsonl`). Under the dry target `[gates]
  mass_relative_drift` therefore defaults to 1e-5, and a config that sets
  it tighter (the smoke's 1e-9, written for the total target) is refused
  by name; the long-run configs' 1e-4 is kept. A branch
  (`--branch-from-other-config`) from a run that held the other quantity
  starts a new mass epoch at the restart state, recorded in the receipt's
  `restart_targets`; without it the first step would take the whole
  difference out at once (49 Pa from day 180 of the year run).
- `energy_fixer = true`: the dry total energy int (cp T + K) dp/g +
  Phi_s ps/g the state carries into the dynamics is restored after the
  semi-Lagrangian step, the hyperdiffusion and the mass fixer by one
  uniform temperature increment (a degree-zero theta offset per level,
  weighted so the energy added equals the deficit). The in-situ ledger of
  the T127 year run put that bracket at -4.9 W/m2 in every 30-day window
  (the semi-Lagrangian ln ps gains mass, the mass fixer takes it back out,
  the ln ps smoothing moves mass along the terrain): LR-4's residual.
  Turning the ln ps smoothing off moves the loss to the mass fixer and
  leaves it (-3.6 against -3.8 W/m2 over ten days), so the remedy is
  global. It is not most of the troposphere's cold drift: over 60 T127
  days the fixer takes 28-40% of the drift against GDAS away at 850, 500
  and 200 hPa (numbers below); the rest is physical.
  The increment is exact in float64. In float32 each level's degree-zero
  coefficient rounds each step's addition to its own spacing (up to about
  8% of it at T255, dt 300 s) without a bias, because the coefficient moves
  by many spacings every step: the in-situ ledger's dynamics bracket reads
  -0.01 W/m2 (T127) and -0.03 W/m2 (T255) with the fixer on.
  Under the dry target the fixer's record (`energy_fixer_w_m2`, and the
  ledger's dynamics bracket with it) also carries the enthalpy of the
  mass the dry-mass fixer adds or removes to follow the physics' E - P,
  about cp T = 2.5e5 J per kg/m2 of precipitable-water change (0.3
  W/m2 for a 1 kg/m2 swing over ten days). Nothing in the ledger carries
  that energy, so restoring it keeps the budget closed, but the row is not
  the core's loss alone.
- `vapour_fixer = true`: the vapour column is restored across the same
  bracket by one uniform rescale of qv. Vapour rides the semi-Lagrangian
  core spectrally with no mass fixer; the adiabatic step created 1-2e-4
  kg/m2 of it a step, which the global water fixer then took off the
  surface stores, leaving the atmosphere 0.5-1.0 W/m2 of latent energy
  nothing supplied.

With the three set against the same days without (fx/energy-mass, GDAS
2026-08-01 00Z, slab ocean, float32):

- T127, 60 days (dt 600 s): LR-4's residual 0.24 against 4.06 W/m2 over
  days 0-30 and 0.43 against 4.50 over days 30-60; the dry-mass range
  0.26 against 35.1 Pa; the water the global water fixer takes off the
  surface stores 2.2e-5 against 1.9e-4 kg/m2 a step. Against what the
  GDAS analyses themselves did from 2026-08-01 to 09-30, the global-mean
  temperature at day 60 drifts -0.24 / -0.92 / -1.84 K at 850 / 500 / 200
  hPa with the switches and -0.40 / -1.27 / -2.82 K without.
- T255, 10 days (dt 300 s): LR-4's residual -0.29 against 5.44 W/m2
  (-0.44 over the first five days), the dry-mass range 0.21 against 10.2
  Pa, the water fixer's surface debit 1.3e-5 against 2.4e-4 kg/m2 a step.
- Five-day T255 forecast from GDAS 2026-09-01 00Z, scored against GDAS:
  NH Z500 error 22.36 / 52.70 m at 72 / 120 h against 22.46 / 53.10 m.
- A 12 h T383 run with all three switches has the same live peak as
  without (12.316 GiB on a 5090; pool held 13.99 against 14.48 GiB).
- Float64: under the dry target the mass fixer's per-step offset also
  carries the physics' E - P since the previous step: 1.1e-7 against
  3.1e-8 under the total target on the T21 semi-Lagrangian CPU case over
  1,500 m terrain, below the float64 per-step gate of 1e-6. A real-weather
  float64 run under the dry target has not been measured. Float32 runs
  read 2.8-3.5e-6 against their 2e-4.

## What has been measured

On an RTX 4090, T255 L40 with the whole native suite, from the GDAS
analysis of 2026-11-10 18 UTC, `open_water_evaporation_refusal = false`.
These are days, not months: no run of 30 days or more has been made, so
no gate that needs one (LR-5, the LR-6 cap bias, the LR-9 year) has been
evaluated.

**Restarts.** In anomaly, climatology and hold mode a 30 h run and its
restarts from +6 h, +12 h and +24 h (the last at the second ice record)
end on the same checkpoint, every one of its 136 arrays, the physics
metadata and the run trackers equal byte for byte. In series mode, with a
GDAS series of five 6-hourly records from 2026-11-10 18 UTC to 11-11
18 UTC, a 24 h run and its restarts from +6 h and +12 h do the same.

**Ten days in anomaly mode**, as two 5-day segments (the second restarted
from the first's last checkpoint, across an ice record):

- Both segments pass every gate of the receipt. Mass drift 5.26e-7, total
  water drift 4.05e-10 (the exchange account counted). The device peak was
  8.64 GiB. The receipts time `model.step` alone at 1,156 s and 1,146 s for
  each segment's 1,440 steps, 3.8 min a model day, with the card shared;
  each segment's whole process took 1,227 s and 1,333 s.
- The boundary was recomputed apart from the package (its own month
  weights, masks, damping rate and formulas; only the T42 projection uses
  the package's transform) and compared at 6 h, 1, 5, 9.75 and 10 days:
  the largest open-ocean |SST - rule| is 1.53e-5 K, half a float32 step at
  290 K, and every ocean column's ice fraction equals its record to
  float32. The receipt's LR-8 rows read the same.
- The starting anomaly, from the 28 analyses of 2026-11-03 18 UTC to
  11-10 12 UTC: Abar = +0.247 K, a Nino-3.4 mean of +2.67 K. The Nino-3.4
  departure read +2.68 K at day 10, the global open-ocean departure rose
  from +0.25 to +0.26 K as the climatology advanced. The first application
  changed the open-ocean SST by 0.73 K rms and 30.7 K at most, on an
  open-water column whose cold-start skin was an ice skin.
- The first ice record removed the analysed ice of 43 columns (42,600 km2,
  all between 52 and 80 N) where the 1991-2020 November SST is 273.6 to
  277.9 K, by the warm-water rule: a column that starts as ice takes the
  climatology, and ice is not kept over water at T_f + 2 K or warmer. Over
  the ten days 565 columns went from ice to water and 202 from water to
  ice.
- Daily global means: surface pressure 98,567.3 Pa on every day (within
  0.1 Pa); precipitable water 25.77 kg/m2 on day 1 falling to 24.52 on day
  10; precipitation 2.90 to 3.14 mm/day. The dry-mass range was 13.6 Pa,
  the land reservoir minimum 479 kg/m2, the largest 24 h open-water
  evaporation 34.7 mm/day, the lowest temperature 179.1 K. No number in
  the diagnostics, the daily files or the last checkpoint is NaN or
  infinite.
- Gate LR-2 reads "not measured" on this run. The land minimum holds, but
  the exchange account's identity (the account equals minus the open-water
  E - P, less the fixer and the clipping) is not checked: the run writes
  the account
  (`global_mean_ocean_exchange_kg_m2`, -3.01 kg/m2 at day 10) and no row
  yet writes the open-water E - P, the fixer's share on water or the
  clipping.
- **The top of the atmosphere loses 21.7 W/m2** over the ten days
  (-8.5 W/m2 on day 1, -22.5 to -24.2 W/m2 on each later day): 349.4 W/m2
  in, 137.2 reflected, 233.8 emitted. The reflected shortwave is the model's
  own, a planetary albedo of about 0.39 against the observed 0.29, and a
  rule for the ocean cannot correct it. The atmosphere's moist total
  energy fell at 7.5 W/m2 and LR-4's residual was 4.6 W/m2, above the 2
  W/m2 at which it is flagged.

**Ten days with the long-run set.** The same start, configuration and two
5-day segments, once with `morrison_droplet_number = "land_ocean"` and
`radiation_liquid_radius = "wrf_relcalc"` added and once without, both on
the tree that carries the options. The control is a later tree and Python
environment than the run above: the two agree at step 0 and part by small
amounts from hour 6 (net at the top over the ten days -21.33 against
-21.67 W/m2). Ten steps of the control are byte-identical to the long-run
line before the options (136 of 136 arrays, same config hash).

- **The top of the atmosphere loses 4.85 W/m2** over the ten days with
  the set, against 21.33 for the control (days 2 to 10: 5.89 against
  22.77): 349.4 W/m2 in, 119.8 reflected against 136.9, 234.5 emitted
  against 233.8. The planetary albedo is 0.343 against 0.392 (CERES-EBAF
  4.2.1, the 2000-2024 November mean: 0.303), and each day from day 2 reads
  0.345 to 0.353 against 0.396 to 0.402, with no trend. The excess
  reflection over CERES at this insolation falls from 31.1 to 13.9 W/m2.
- By band over hours 6 to 240, with the control and CERES: 60S-30S 0.378
  (0.458, 0.323), 30S-0 0.288 (0.339, 0.231), 0-30N 0.283 (0.315, 0.246),
  30N-60N 0.416 (0.447, 0.374). What remains is ocean cloud, as in the
  48 h twins of `docs/albedo-fixes.md`.
- Cloud cover (each day's four checkpoints) 0.72 to 0.74 from day 2,
  against 0.73 to 0.75; grid-mean liquid water path 95 to 99 g/m2 against
  113 to 117.
- The daily area-mean 2 m temperature is within 0.03 K of the control on
  every day, and precipitation reads 2.89 against 2.90 mm/day over the
  ten days. The 850 hPa global mean runs 0.1 to 0.3 K above the control
  from day 3 and rose 0.26 K over the ten days, against 0.01 K, read as
  the mean of day 1's 6-hourly values to the mean of day 10's. From the
  analysis instant (281.06 K) to hour 240 it reads +0.07 K with the set
  and -0.20 K for the control: the set holds 850 hPa near its starting
  value where the control cools.
- Both segments pass every gate of the receipt. Mass drift 3.7e-7 and
  8.9e-7, total water drift 3.0e-9 and 1.7e-9, dry-mass range 15.4 Pa, no
  number NaN or infinite. LR-4's residual is 4.76 W/m2 (control 4.74). The
  device peak at the allocator was 8.69 GiB in each segment (control 8.64).
  `model.step` took 1,099 s and 1,146 s for each segment's 1,440 steps,
  3.7 min a model day with the card shared (control 1,122 s and 1,123 s).

## Limits

- The reference physics suite caps evaporation by its finite surface
  reservoir and is not a configuration for long runs.
- The climatology is a 1991-2020 mean. The rules' damping times are
  declared engineering choices inside the published ranges, not fitted to
  this model.
- Lakes do not freeze or thaw, and an open lake evaporates at no less than
  0 C all winter.
- There is no ocean below the surface, no sea-ice growth or melt from the
  model's fluxes, and no aerosol.
- The solar constant is WRF's 1370 W/m2 times the eccentricity factor.
- Orographic gravity-wave drag is a native option, `orographic_drag =
  "kim-gwdo"`, off by default and on in the shipped T255 and T383 forecast
  days since 0.2.3; non-orographic drag is absent.  The only
  momentum sink near the model top otherwise is the zonal-mean drag of the
  top levels.
