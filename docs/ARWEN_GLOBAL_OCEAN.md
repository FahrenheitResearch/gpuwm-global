# The coupled ocean: a slab mixed layer with a heat-flux correction, and thermodynamic sea ice

**Status: experimental.** All three packages are in: the switch, the
coupling in the step, the physics-state arrays, the restart, the tape
fields and the `climatology qflux` command (package C, `ocean_coupler.py`
and the files listed under Files), the mixed layer and the
Q-flux file (package A, `ocean_slab.py`, `ocean_qflux.py`) and the ice mass
(package B, `sea_ice_thermo.py`). The parser admits `[ocean] mode =
"slab"`, `[sea_ice] mode = "thermodynamic"` and `qflux_calibration`. On a
card it has run for ten days from one start with the ocean on, after a ten-day
calibration from the same start, so its Q-flux is in sample. Gates OC-1,
OC-2 and OC-3 hold on the card. Gate OC-4 holds for the sea surface but
not for the ice: the open-ocean SST stays within its drift bounds in the
mean and the rms, its largest change on one column is over the 5 K bound,
and the southern pack grows while the climatology's shrinks (Gates). Every
number on this page is a constant the design chooses, arithmetic, or a
measurement named as one under Gates.

## What it is for

The rules of the long-run page (hold, anomaly, climatology, series) write
the sea surface from a closed form or a file. None of them answers the
model's own fluxes, so none of them can run past what the form or the file
says, and none can let the ocean and the atmosphere adjust to each other.
This page adds an ocean that does: one mixed layer per ocean column whose
temperature is the model's SST, kept near the observed climate by a
heat-flux correction ("Q-flux") made from the model's own prescribed-SST
run, an optional weak restoring toward a target SST as a safety, and sea
ice that grows and melts from the energy that reaches it.

It is off by default. With no `[ocean]` table, and under every existing
mode, no array, hash, receipt row or tape field changes (gate OC-1).

## The switch

```toml
[ocean]
mode = "slab"
climatology = "climatology-t255.npz"   # built with --with-mld when slab_depth = "woa18"
slab_qflux = "qflux-t255.npz"          # required; enters identity by SHA-256
slab_depth = "woa18"                   # "woa18" | metres, e.g. 50.0
slab_depth_bounds_m = [10.0, 200.0]
slab_relaxation_days = 365.0           # days | "none"
slab_target = "climatology"            # climatology | anomaly

[sea_ice]
mode = "match"                         # under slab: thermodynamic (the only choice)
new_ice_thickness_m = 0.5              # default 0.5 under thermodynamic
record_interval_hours = 24.0
```

The calibration run that makes the Q-flux file is a climatology run with
two more keys:

```toml
[ocean]
mode = "climatology"
climatology = "climatology-t255.npz"
anomaly_reference = "start"            # or "mean-7d" with anomaly_reference_series
climatology_ramp_days = 0.25
qflux_calibration = true
qflux_calibration_spinup_days = 2.0
```

Rules the parser enforces (and, where it says so, the launch):

- `slab` needs `slab_qflux`, a climatology, and a dated run on real statics
  or an analysis; at launch, the native physics suite or the reference
  suite (`physics.mode = "none"` computes no surface flux and is refused).
- The slab keys are read only under `slab`; a slab key under another mode
  is refused by name.
- `slab` goes with `[sea_ice] mode = "thermodynamic"` (`match` resolves to
  it), and `thermodynamic` goes only with `slab`. Prescribed ice under a
  prognostic SST is not offered: the ice and the water must answer the same
  energy.
- `qflux_calibration` goes only with `mode = "climatology"`.
- `slab_depth = "woa18"` needs a climatology that carries the
  `mixed_layer_depth_m` plane (`--with-mld` at build time); the launch
  refuses a climatology without it.
- Under `slab_target = "anomaly"` the anomaly keys of the long-run page
  (damping, reference series, truncation) apply to the target, which is the
  anomaly rule's SST built from the cold start as the rule builds it.
- The Q file is checked at launch against the run's grid, physics identity,
  climatology and the months the run reaches (`QFluxFile.check`), and a
  file whose bytes changed since the config was read is refused.

The climatology is the engine's own monthly OISST 1991-2020 file, built
by the existing climatology command from the pinned public rows (NOAA data,
public domain, credited in the sources table). A daily long-term-mean file
a user already has can become a new source row later; nothing here depends
on one, and no climatology is bundled in the package.

## The mixed layer

Per ocean column (inside the climatology's ocean, land fraction at most one
half, not an inland lake), with T the mixed-layer temperature:

    rho_w c_w h(t) dT/dt = F_open + Q + R

- rho_w c_w = 1026 x 3990 = 4.094e6 J m-3 K-1.
- **h(t)**: `"woa18"` is the climatology's WOA18 monthly mixed-layer depth,
  interpolated in time with the SST's own month weights (linear between
  mid-month instants) and clamped to `slab_depth_bounds_m` (default 10 to
  200 m); a gap on an ocean column takes the lower bound. A number is a
  fixed depth everywhere. Climatological depth is the default because a
  fixed depth gives a tropical ocean the seasonal swing of a high-latitude
  one (or the reverse).
- **F_open**: the net heat into open water over the interval, from the
  planes the physics held for it: `swdown (1 - albedo) + emissivity (glw -
  sigma T_skin^4) - hfx - L_v qfx`, with the albedo and emissivity the
  radiation saw. On a frozen-class column (fraction at least one half) it
  is the lead tile's: `(1 - A_s)` times the same balance with the lead
  fluxes the native suite already computes (`lead_hfx`, `lead_qfx`), the
  open-water albedo 0.08, emissivity 0.98 and the lead skin, where A_s is
  the fraction the atmosphere saw.
- **Q** and **R** below.
- **Step**: explicit, once per model step (300 s at T255). The radiative
  damping time rho_w c_w h / (4 eps sigma T^3) is about 7e6 s at h = 10 m
  (arithmetic), so the explicit step is far inside its stability limit.
- **Freezing**: T is never below T_f = 271.35 K (the rules' freezing point
  of sea water). Cooling past it is a deficit handed to the ice as new ice
  (below). While a column holds ice, T stays at T_f.
- **Start**: T is the cold-start skin on open-ocean columns, held at or
  above T_f; T_f on columns with any ice. Lakes keep their own 10 m skin
  slab and are never written.

## The heat-flux correction (Q-flux)

**Why it is needed.** A slab without Q drifts by the ocean heat transport
it lacks and by every stationary bias of the model's surface fluxes. The
long-run page measured one of those biases: the top of the atmosphere loses
21.7 W/m2 over ten days, most of it a planetary albedo of about 0.39 against
an observed 0.29. A slab that absorbed even a fraction of that unopposed
would cool by tenths of a kelvin a month at 50 m and by more at 10 m
(arithmetic: 10 W/m2 for 30 days into 4.09e7 J m-2 K-1 is 0.63 K).

**What Q is.** The heat the ocean must receive, besides the model's own
fluxes, for the slab and the ice to follow the observed climate. With C_s
and C_i the climatology's SST and ice fraction and F_cal(t) the net heat
the model's atmosphere puts into the ocean-and-ice system when the SST is
prescribed to C_s:

    Q(x, t) = rho_w c_w h(t) dC_s/dt - rho_i L_f h_ref dC_i/dt - F_cal(x, t)

An open column whose fluxes equal the calibration's therefore follows C_s
exactly (a column under ice, below); a column whose fluxes differ (because its SST differs, or the
weather does) departs from them, which is the point. Q absorbs the ocean
heat transport, the model's albedo and energy biases, the solar constant
choice and the absent aerosol, as far as they are stationary by season.
h_ref is the thickness the climatology's ice fraction is read with, 2.0 m
north and 1.0 m south of the equator: the climatology has no thickness, so
these are stated engineering choices near observed annual means, not fits.

The slopes dC_s/dt and dC_i/dt come from a shape-preserving cubic (PCHIP,
Fritsch and Carlson 1980) through the monthly planes at their mid-month
instants, and each step books the curve's change over that step. A column
driven by exactly F_cal therefore still reaches each month's plane at its
mid-month instant, and Q no longer steps there. The piecewise-linear slope
it replaced was constant between mid-month instants and stepped at each:
in the 180-day coupled run the ocean mean of Q jumped by 21 to 27 W/m2 at
the August to November instants and by 12 W/m2 at the December and January
ones. With the cubic, on the same days the jumps were 0.04 and 0.08 W/m2,
and no 6-hourly change exceeded 0.18 W/m2.

**Under ice.** While a column holds ice the mixed layer sits at the
freezing point and the ocean's heat goes to the ice, so on those columns
Q leaves out its first term, the heat that warms or cools the
climatology's open water; the ice term stays. Handed to the pack, that
term melts or grows ice at a rate the climatology's ice slope does not ask
for. The first ten-day coupled run on a card applied it under ice, and the
southern pack lost 2.17 million km2 against the climatology's 1.04. Most
of that loss came after the mid-November instant, where the slope of the
southern SST steepens (on the frozen columns south of 50S the term's
area mean went from 17 to 64 W/m2 across that instant). The rule has a
cost: a column whose climatology warms while the slab still holds ice
reaches open water at the freezing point, then warms at the climatology's
rate but behind it. One mixed layer per column cannot be both partly
frozen and above freezing, and this is a limit of the design.

**A cooling Q under a pack.** Where the slab holds ice, a negative Q is
carried only in the share (1 - A) / (1 - C_i) when the slab's fraction A is
above the climatology's C_i, and in full otherwise. The reason is how that
cooling arises. Under the southern summer pack, F_cal is mostly the
shortwave that the climatology's leads, and its partly frozen water-class
columns, absorbed at the open-water albedo. That heat is beyond what the
climatology's slopes store, so Q books it as heat the ocean carried away.
A pack more closed than the climatology's absorbs only its own open share
of that heat, so it is charged only that share. A column whose state and
fluxes are the calibration's is unchanged.

Charged in full, the sink grew the 180-day coupled run's southern pack
through the austral summer. On the cells where the model and the
climatology both held ice, the monthly-mean Q was -78 W/m2 in December and
-87 W/m2 in January. The slab's pack absorbed 5 W/m2 in its leads at the
end of December against the 45 to 56 W/m2 the calibration booked, because
its fraction was 0.96 against the climatology's 0.55. The new ice the
negative flux formed closed the leads further. Southern volume went from
7.24 to 11.04 thousand km3 between 1 December and 28 January.

The same run resumed with the share from its 30 November checkpoint
(RTX 5090, 2026-11-13/14) gave these figures. The run is on the long-run
line with the fix cherry-picked onto it: its ocean modules are 0.2.2's
byte for byte, and its atmosphere is that line's, not 0.2.2's.

- Monthly-mean Q on the shared cells was -30 W/m2 in December and -42 W/m2
  in January.
- The fraction on those cells was 0.78 and 0.72, against the climatology's
  0.63 and 0.53.
- Southern volume fell from 7.21 to 2.52 thousand km3.
- The 15 % ice area fell from 8.47 to 3.94 million km2 (-53 %). The
  climatology's area fell from 13.22 to 6.24 (-53 %).
- The northern ice area at 28 January was 13.52 million km2 against 13.59
  without the share, and the northern volume 11.98 against 12.70 thousand
  km3. Over those months the share hardly acts in the north: Q under the
  winter pack is positive.
- The energy books' largest column residual was 1.05e-2 J/m2, and the
  open-ocean SST against the climatology was -0.471 K against -0.472.

The CAM slab ocean's answer to the same problem (the CAM5 scientific guide,
"Slab Ocean Model") is different. It tapers every cooling Q to zero as the
mixed layer nears freezing, spreads the withheld heat over warm water, and
tunes a per-hemisphere term under ice. Here that would withhold 0.6 to
0.8 PW in December and January. It would also remove the -51 to -61 W/m2
under the northern summer pack, which follows the analysis to its
September minimum with its fraction as compact as the climatology's.

**How F_cal is made.** The standard way (Hansen and co-authors 1984, the
CESM slab-ocean lineage): a run of this model with the SST prescribed.
`[ocean] mode = "climatology"` with `qflux_calibration = true` runs the
coupler's flux arithmetic every step on every ocean column without writing
anything to the surface. The flux it books is the one a slab run would
consume: F_open plus, on frozen-class columns, A_s (M_top - G_up), the
top-melt surplus and the base conduction of the ice column (see sea ice).
After `qflux_calibration_spinup_days` (default 2, the atmosphere's
adjustment to the prescribed surface; the ramp of the climatology mode
should be short, 0.25 days in the example) each step adds `F dt` to the two
calendar-month bins whose mid-month instants bracket the step, with the
month weights, and the weights in seconds to a second array. Both arrays
ride in the physics state, so a restarted calibration accumulates what the
straight one does, and a calibration can be made of several runs.

**Storage.** `gpuwm-global climatology qflux` (a subcommand of the
climatology command) reads the last checkpoint of one or more
calibration runs and writes one `.npz`, schema
`gpuwm.arwen-global-slab-qflux/v1`, byte-reproducible:

| array | shape | meaning |
|---|---|---|
| `calibration_flux_w_m2` | (12, nlat, nlon) float64 | F_cal per month (sum / weight), NaN where a month has no weight on a column |
| `coverage_s` | (12, nlat, nlon) float64 | the month weights in seconds |
| `latitude_deg`, `longitude_deg` | grid | the grid it is valid on |
| `provenance` | JSON | schema, truncation, physics identity of the calibration runs, their config hashes, the climatology's SHA-256, h_ref, the spin-up, the smoothing truncation, package version |

The file stores F_cal, not Q: Q is formed at run time from the climatology's
slope and the run's own h(t), so changing the depth bounds needs no new
calibration. An optional smoothing truncation (none by default) projects
each month onto spherical harmonics over the ocean to damp the weather
noise of a short calibration.

**Refusals.** A slab run refuses a Q file on another grid, of another
physics identity (`identity_of_global_physics_options`), or made with
another climatology, and refuses at launch when the run reaches a month
whose coverage is zero on every ocean column, naming the months. The
receipt prints every month's coverage.

**Cost and noise.** A whole year of calibration at the long-run page's
planning rate of 4.3 min per model day at T255 is about 26 card hours
(arithmetic). The file can be made from shorter runs (any months they
reach are covered, with the month weights), so a first file can come from
ten days and later files from longer runs. A short calibration carries the
weather of its days in F_cal: tens of W/m2 locally over ten days. That is
the noise gate OC-4 sees, and it is stated with the measurement.

## Restoring

    R = rho_w c_w h(t) (T* - T) / tau_r

`slab_relaxation_days` (default 365, `"none"` turns it off) toward T*:
`slab_target = "climatology"` (C_s, the default, the safe target for runs
of indefinite length) or `"anomaly"` (the anomaly rule's SST, which keeps
the starting anomaly the rule keeps). It is a safety, not the mechanism:
at h = 50 m and 365 days it is a feedback of 6.5 W m-2 K-1 (arithmetic),
below the atmosphere's own surface-flux feedback of order 15 to 20, so it
bounds slow drift and leaves weekly and monthly anomalies mostly to the
fluxes. Under ice R is part of the ocean's heat into the ice base, so where
the target is open water it melts the pack slowly.

## Sea ice

One owner for the ice temperature, as today: the native suite's four-node
conduction column (`physics/frozen_surface.py`, snow over ice, base at
271.36 K, albedo 0.65, emissivity 0.98, WRF's composite with open water for
the atmosphere). It is not changed, so a run with the ocean off is
unchanged. What is added is the ice mass, per ocean column: the fraction A
and the cell-mean volume V (metres of ice per square metre of cell;
thickness V / A). The balance is Semtner's zero-layer one (1976,
doi:10.1175/1520-0485(1976)006<0379:AMFTTG>2.0.CO;2) with the area rules of
the regional engine's original thermodynamic ice; no source is transcribed.

Per step, with rho_i L_f = 917 x 3.337e5 = 3.06e8 J m-3:

- **Base.** G_up, the conduction the column draws from its base into its
  bottom node (from the column's own layers and node layout), grows ice.
  The ocean's net heat F_o = F_open + Q + R (the mixed layer being at T_f
  while ice is present) melts ice when positive and grows it when negative.
- **Top melt.** M_top, where the column's top node sits at the melting
  point, is the surface balance there that the column now discards
  (`frozen_surface.py` says the energy above the melting point is not
  carried). It melts ice. This closes that leak in slab runs; computed from
  the held planes after the column's step, it is an estimate of what the
  implicit step discarded, and that is stated as a limit.
- **Volume:** dV = dt [A (G_up - M_top) - F_o] / (rho_i L_f), plus the
  slab's freezing deficit / (rho_i L_f).
- **Area.** New ice from open-water cooling (a negative F_o, the freezing
  deficit) covers open area at `new_ice_thickness_m` (default 0.5 m under
  this mode); A is capped at 1. Melting lowers A as A sqrt(V_new / V_old).
  A volume that would go below zero is set to zero with A, and the energy
  beyond it warms the mixed layer.
- **Snow on ice: none carried.** The column keeps reading the snow depth the
  state holds, as today. Snowfall on the pack is a limit of this design.
- **Freezing point** 271.35 K for the mixed layer; the column's base stays
  at its own 271.36 K. The 0.01 K is left as it is so that nothing changes
  with the ocean off.
- **To the atmosphere.** The prognostic A and V / A reach the surface state
  at each ice record (`record_interval_hours`, default 24 h), through the
  same path the rules use: a column that crosses the ice threshold of one
  half is re-classed by the cold start's rulebook, and the native suite's
  masks are dropped. Between records the atmosphere sees the last record.
- **Start.** A is the analysed fraction, V is A times the analysed
  thickness (a column with ice and no thickness takes the new-ice
  thickness).
- No ice motion, no ridging, no lead opening by wind.

## Coupling: interval and order in the step

The coupling interval is the model step. The boundary service already runs
before every `model.step` (`runner.py`); the slab is a provider of that
service with one more method, `advance`, called first. In each step:

1. **Advance** the mixed layer and the ice mass from the last update to the
   state's time, with the fluxes the physics held over that interval (those
   of the previous step, evaluated at the skin it saw). At a cold start the
   interval is empty and nothing moves. Time-dependent inputs (h, the
   climatology's slope, F_cal, T*) are evaluated at the interval's start.
2. **Ice record**, when one is due: fraction and thickness written,
   crossings re-classed.
3. **Write** T as the skin of every open-ocean column (frozen-class columns
   keep the column's blended skin).
4. `model.step`.

The flux lag of one step is the lag the rule SST already has. Doing it in
the service, not inside the native suite's land call, keeps the native
runtime untouched (gate OC-1), lets the reference suite on the CPU run the
same code (the T3 smoke the ocean-rule tests use), and puts the coupling
beside the provider machinery that already handles parked planes, ice
records and class crossings. The native suite holds every plane F_open
needs in its physics state (`swdown`, `glw`, `hfx`, `qfx`, `lead_hfx`,
`lead_qfx`, `ice_hfx`, `ice_qfx`). The reference suite does not; when, and
only when, the slab or the calibration is on, it holds one plane of its own
net surface flux for the coupler to read
(`ocean_held_net_surface_flux_w_m2`: the slab's cold start puts the plane
in the state, and `physics/reference.py` writes it only where it finds it).
On the reference suite F_open is that plane on every ocean column, frozen
or not: the suite steps no ice column, so G_up and M_top are zero there.

Under the calibration the service's provider is the climatology rule, which
writes the surface as it always does, and the calibrating `SlabProvider` is
the service's coupler: its `advance` runs first in the step, books the flux
and writes nothing to the surface. The rows and the restart identity are
the rule's with the calibration's additions.

An ice record is written once. The metadata records the last record the
slab wrote, so a restart inside a record does not write it again (the
surface it restarts from already carries it); a restart at a record's first
instant writes it, as the straight run did at that instant.

## State, checkpoints and identity

Every prognostic array lives in the physics state, float64, so a checkpoint
carries it as `physics__<name>` and a restart resumes it bit for bit:

| name | meaning |
|---|---|
| `ocean_mixed_layer_temperature_k` | T (0 off the ocean columns) |
| `ocean_mixed_layer_depth_m` | h at the last update (for the tape) |
| `ocean_qflux_w_m2` | Q at the last update (for the tape and the rows) |
| `ocean_ice_fraction`, `ocean_ice_volume_m` | the prognostic ice |
| `ocean_energy_sources_j_m2` | cumulative sum of dt (F_open + Q + R + A (M_top - G_up)) |
| `ocean_energy_stored_j_m2` | cumulative sum of rho_w c_w h dT - rho_i L_f dV |
| `ocean_qflux_calibration_sum_j_m2`, `..._weight_s` | (12, nlat, nlon), calibration only |
| `ocean_held_net_surface_flux_w_m2` | the reference suite's net surface flux of the last step, slab or calibration on the reference suite only |

A calibration carries the two calibration arrays and not the seven slab
arrays. `physics_state.metadata["ocean_slab"]` holds the time of the last
update, the count of advances, the last ice record written and, in a
calibration, the count of deposits. The restart identity is the existing
`physics_metadata["surface_boundary"]` block with `ocean_mode = "slab"`,
`sea_ice_mode = "thermodynamic"`, the climatology's and the Q file's
SHA-256, the start and every slab and ice parameter (a calibration run's
block adds `qflux_calibration`); a restart under another block is refused
before any of its arrays is used, as for the rules. The config identity
gains the slab keys only when `mode = "slab"` or `qflux_calibration` is
set, so every existing config hash stays as it is. The Q file enters by
content SHA-256, never by path.

## Tape fields

Every tape carries `SST`, the skin temperature on every point. When the
slab is on, the slab's own `SST` takes its place and two fields join it
(with the slab off, the tape is the one written before, byte for byte):

| field | units | meaning |
|---|---|---|
| `SST` | K | T on ocean columns, the skin elsewhere |
| `HML` | m | h, 0 off the ocean (WRF's name for the mixed-layer depth) |
| `ARWEN_QFLUX` | W/m2 | Q |

`SEAICE` and `ICEDEPTH` are already on every tape from the surface state;
under the slab they carry the prognostic ice of the last record. The three
fields carry their units and a description on the tape, and the export
receipt's `sea_surface_temperature` row says which `SST` the tapes carry.

## Gates

Each gate says what has been run and what has not. The coupling tests are
in `tests/test_arwen_global_ocean_coupling.py` and run packages A's and
B's own functions; the parts' known answers are in
`tests/test_arwen_global_ocean_slab.py`,
`tests/test_arwen_global_ocean_qflux.py` and
`tests/test_arwen_global_sea_ice_thermo.py`.

- **OC-1, off is byte-identical.** Every shipped config's hash unchanged,
  the existing suite passing unchanged, and the T3 reference smoke with no
  `[ocean]` table and under each rule mode writing checkpoints and a tape
  equal byte for byte to the parent commit's. On the card, a 6 h T255
  native run with no table ending on the parent commit's checkpoint bytes.
  Measured on a host CPU (2026-11-12), the T3 smoke with no table and under
  the anomaly, climatology and hold rules (the rule tests' dated smoke),
  run once from this package and once from the design commit it is built
  on: all 12 checkpoints equal as files byte for byte, the diagnostics
  equal, and the three tapes of the no-table run equal byte for byte. The
  coupling tests hold the no-table smoke to carrying no ocean array, no
  held flux plane and no slab metadata, and no rule's identity payload to
  gaining a slab key. Measured on an RTX 5090 (2026-11-12), the T255
  native semi-Lagrangian config from GDAS 2026-09-01 00Z with no table, run
  for 6 h (72 steps of 300 s) and for 1 h (12 steps) once from the merged
  tree and once from the same tree without the three packages: every
  checkpoint (steps 0, 36, 72 and 0, 6, 12) identical array for array and
  in `self_sha256`, read with `tools/arwen_global_checkpoint_identity.py`.
  The 1 h run repeated on 2026-11-13, before and after the under-ice rule
  and the checkpoint reader's float64 slab arrays: identical at steps 0, 6
  and 12, config hash `500bba26...` on both. Repeated for the under-ice
  share and the cubic tendency (2026-11-14, RTX 5090): the shipped T255
  native semi-Lagrangian 6 h config dated to GDAS 2026-09-01 00Z, no table,
  on 0.2.0 and on the fixed tree. Steps 0, 36 and 72 were identical array
  for array and in `self_sha256`, config hash `bb21f857...` on both.
- **OC-2, restart is exact with the ocean on.** The T3 smoke under `slab`
  with a synthetic climatology and Q file, and under the calibration,
  straight and restarted at a step inside an ice record and across a
  mid-month instant: every checkpoint array, the physics metadata and the
  trackers equal byte for byte. On the card, as the long-run page did: a
  30 h T255 run restarted from +6, +12 and +24 h. The two smokes are
  coupling tests (from 11:59:20 UTC on 16 January, restarted at step 4 of
  8, on January's mid-month instant, inside the ice record that began at
  30 s); a restart under another slab depth, and a plain climatology run
  from a calibration's checkpoint, are refused. Measured on an RTX 5090
  (2026-11-13), the ten-day slab run of OC-4 restarted from its +6, +12
  and +24 h checkpoints (the last on the second ice record) to +30 h: every
  later checkpoint (+12, +18, +24 and +30 h) identical array for array and
  in `self_sha256`, read with `tools/arwen_global_checkpoint_identity.py`.
  The first card restart, before the checkpoint reader kept these arrays
  in float64, was not exact. A float32 run read the slab's arrays in its own
  precision, so every ocean column moved one step after the restart: the
  mixed-layer temperature by up to 1.5e-5 K and the energy books by up to
  0.03 J/m2. The restart smokes now run in float32 as well as float64.
- **OC-3, the energy budget closes.** On every ocean column, at every
  output step, |stored - sources| at most 1e-9 of the column's summed
  |increments| plus 1e-3 J/m2 (float64). The F_open the slab took equals
  the surface balance recomputed from the checkpoint's own planes to float32
  rounding. A known-answer test: a column driven by exactly F_cal follows
  C_s to 1e-9 K at the mid-month instants. The coupling tests hold the books
  to the first bound on a constant flux, restoring, freezing into new ice,
  growth under ice, a complete melt and the slab smoke, and a column driven
  by exactly F_cal on a flat climatology to 1e-11 K. The mid-month known
  answer on a sloped climatology is
  `test_a_column_driven_by_exactly_f_cal_follows_the_climatology` in the
  slab tests. Measured on an RTX 5090 (2026-11-13), the ten-day slab run of
  OC-4: at days 1, 5 and 10 no ocean column of the 194,234 has
  |stored - sources| above 1e-9 |sources| + 1e-3 J/m2. That is a stricter
  test than the bound, because |sources| is never more than the summed
  |increments|. The largest |stored - sources| on any column is 2.8e-3 J/m2 at
  day 10, and the global totals (1.963e21 J) agree to 1.9e8 J. On a run
  with a checkpoint every step (steps 1 to 5 of the same start, on the tree
  before the under-ice rule), the F_open the slab took equals the balance
  recomputed from each checkpoint's own planes to 7e-13 W/m2 on every
  open column. That balance differs from the radiation's own surface
  planes (gsw + glw - lwupb - hfx - L_v qfx) by 0.05 W/m2 in the area
  mean and at most 0.19 W/m2. The shortwave parts agree to 1e-4 W/m2.
- **OC-4, drift over 10 days.** T255 native from the GDAS analysis of
  2026-11-10 18 UTC, `slab_relaxation_days = "none"`, a Q file calibrated on
  the same ten days (in sample, and said so). Bounds, chosen here: the
  open-ocean area mean of T - C_s changes by at most 0.15 K from day 0 to
  day 10; the area rms of that change is at most 0.6 K; its largest value
  at most 5 K on columns that kept their class (the others counted); each
  hemisphere's ice area changes by no more than 0.5 million km2 more or
  less than the climatology's change over the same days; no NaN anywhere.
  The arithmetic behind the rms bound: 10 W/m2 of unresolved flux for ten
  days moves a 10 m layer 0.21 K and a 50 m layer 0.04 K. An out-of-sample
  check (another start, or a calibration of other days) follows as soon as
  a second calibration exists. Measured on an RTX 5090 (2026-11-13), the
  shipped T255 native semi-Lagrangian config dated to that analysis, with
  a climatology built `--with-mld`, `slab_depth = "woa18"` in [10, 200] m
  and the Q file of a ten-day calibration from the same start with two
  days of spin-up (its coverage: October 0.17 days, November 7.45,
  December 0.37). Day 0 to day 10: the mean change of T - C_s is
  -0.018 K, within the bound. The rms is 0.224 K, within the bound. The
  largest change on a column that kept its class is 5.46 K, over the 5 K
  bound, and 2,615 columns changed class. The northern ice area grew
  0.93 million km2 against the climatology's 0.76, within the bound. The
  southern ice area grew 0.23 million km2 while the climatology's shrank
  by 1.04, which is outside the bound. The southern growth is on columns
  that held ice at the start; new ice on open water adds 0.001 million
  km2. No
  array held a NaN. The prescribed-SST twin of the same start (the
  climatology rule, which the calibration matched array for array at days
  5 and 10 apart from its two sums) puts the slab's open-ocean SST
  0.228 K above it at day 1 and 0.216 K above it at day 10, because the
  slab starts from the analysis, which is 0.22 K warmer than the
  climatology over the open ocean, and keeps most of that anomaly. The
  same run with F_cal set to zero (Q is then the climatology's slope
  alone; run on the tree before the under-ice rule) drifts more: a mean
  change of -0.067 K, an rms of 0.462 K, and 6.38 K on one column. Over
  ten days its mixed layers lost 1.6e7 J/m2 in the open-ocean mean
  (19 W/m2), against a gain of 8e5 J/m2 with F_cal on the same tree. This
  is the bias Q removes. The atmosphere's planetary albedo of 0.39 to 0.40 and a
  top-of-atmosphere net of -21 to -24 W/m2 are the same in the twin and
  in the slab run. Cost on one card (on the tree before the under-ice
  rule), against the calibration run of the same start (the climatology
  rule with the coupler's flux bookkeeping): 0.4024 s against 0.3981 s per
  step (1.1 % more), and a device peak at the allocator of 6.73 GiB
  against 6.77 GiB.
- **OC-5, the parts.** Known answers on the host: a constant flux with fixed
  h moves T linearly; restoring alone decays as (1 - dt/tau)^n; freezing,
  basal growth, top melt, the square-root area rule and a complete melt
  each give their closed form; the month deposit and its read-back are
  exact; the parser refuses every combination listed under the switch.
  The coupler's share is in the coupling tests: the linear march, the
  decay, the freezing hold and its ice, the melt that warms the mixed layer
  with what is left, the calibration's month deposit with its spin-up, an
  ice record written once, the tape fields, and the parser's refusals.

## Files

**A, the mixed layer, the Q-flux and the restoring.**
`src/arwen_global/ocean_slab.py` (`mixed_layer_depth`,
`climatology_tendency`, `open_water_flux`, `qflux`, `restoring_flux`,
`slab_step`, `initial_temperature`, `SlabParameters`) and
`src/arwen_global/ocean_qflux.py` (`deposit`, `months_needed`,
`build_qflux`, `load_qflux`, `QFluxFile.at`, `QFluxFile.check`, `main`);
tests `tests/test_arwen_global_ocean_slab.py` and
`tests/test_arwen_global_ocean_qflux.py`.

**B, sea ice.** `src/arwen_global/sea_ice_thermo.py`
(`ice_reference_thickness`, `initial_ice`, `thickness`,
`basal_conduction`, `top_melt_surplus`, `ice_step` returning `IceStep`);
tests `tests/test_arwen_global_sea_ice_thermo.py`.

**C, the coupling.** `src/arwen_global/ocean_coupler.py` (`SlabProvider`
with `advance`, `tape_fields`, the array names and the gates OC-1 to OC-4),
with the switch, checkpoint, restart, tape and command parts in
`boundary_options.py`, `runner.py`, `surface_boundary.py`,
`wrfout_export.py`, `cli.py` (the `climatology qflux` subcommand calling
`ocean_qflux.main`) and `physics/reference.py` (the held net-flux plane,
only when on); tests `tests/test_arwen_global_ocean_coupling.py` and its
`_cuda` companion.

All arithmetic in A and B is float64 on the array module handed in (numpy
or cupy) and reads no run state; C chooses the columns, the planes and the
time.

## Limits

- No ocean below the mixed layer: no currents, no upwelling, no
  entrainment; the mixed-layer depth is a climatology, not a forecast. The
  Q-flux puts back the climatological effect of all of them, not their
  anomalies, so an El Nino is damped by the surface-flux feedback rather
  than grown or ended by the ocean.
- Q is as good as its calibration: its length, its season and its physics.
  A physics change needs a new file (the identity check refuses the old
  one).
- No snow on the pack, no ice motion, no ridging; ice fraction reaches the
  atmosphere at records, not every step.
- Lakes keep their own skin slab and do not freeze, as on the long-run
  page.
- The top-melt surplus is estimated from the held planes after the ice
  column's own implicit step.
- The ice area does not follow the climatology's, even with Q. In the
  ten-day card run the southern pack grew on the columns that held it,
  while the climatology's pack shrank (OC-4). The area rules are not
  symmetric: a net cooling of the ocean under a pack adds area at the
  new-ice thickness, a melt takes area away only as the square root of
  the volume, and Q's ice term pays for the climatology's area change at
  h_ref. A cooling Q under a pack more closed than the climatology's is now
  carried only in its open share (under ice, above), which stopped the
  southern summer pack's growth; the area rules themselves are unchanged.
  The share has been measured over December and January only; under the
  northern summer pack (July and August, where Q is -51 to -61 W/m2) it is
  not yet measured.
- A month's F_cal comes from whatever days of that month a calibration
  reached. A ten-day calibration in mid-November leaves October and
  December with under half a day each (0.17 and 0.37 days), and after the
  mid-month instant Q mixes in a December value made from those 0.37
  days.
