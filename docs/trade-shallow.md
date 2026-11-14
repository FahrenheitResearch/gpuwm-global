# Trade cumulus: the shallow arm that would not run

**Short version:** over the trade-wind oceans the model's remaining excess of
reflected sunlight is a near-saturated sheet of low cloud under the trade
inversion. It stays because Grell-Freitas shallow convection exits in most trade
columns. The exit comes from one sum in WRF v4.6.1's shallow cloud-top test,
which both later Grell-Freitas lines write differently. Fixing that sum vents the
trades. Fixing it alone also vents the stratocumulus decks away, for two
reasons:
- **The closure.** A second WRF line, the shallow closure's boundary-layer
  member under a capped boundary layer, hands the deck plumes 3-5 times the
  trade mass flux. GFL corrects that line too.
- **The plume.** Any shallow plume is too much for a deck one or two levels
  deep, and nothing in the reference schemes keeps one out.

Three opt-in options carry the two reference corrections and a cloud-depth
minimum of this package's own, which reuses the deep arm's ierr 6 test. All are
off by default, and with them off the kernel is WRF's word for word.

A 30-day December twin vents the trades as the August twins do, but it darkens
the Peru and Australian decks: in that season the depth minimum does not keep
the southern decks apart from the trades.

## Measured: 10-day T255 twins

**Setup:**
- All arms start from GDAS 2026-08-01 00Z.
- The control is the 90-day anomaly+fix arm cut to 10 days, with the albedo
  pair on. The run reproducing it on this tree with the options off matches
  135 of 135 arrays at step 288.
- The radiation and rain are exact 00Z-00Z means of the accumulators in the
  daily checkpoints, 2026-08-02 00Z to 08-11 00Z, the same window for every
  arm. They are not the daily summaries: those follow the output cadence, which
  differs between the arms.
- CERES EBAF 4.2.1 is the August 2000-2024 mean.

**Reflected shortwave in excess of CERES** (W/m2 at each region's insolation):

| Region | CERES albedo | control | cb | cbqe | **cbqed** |
|---|---|---|---|---|---|
| Global | 0.278 | +14.9 | +0.4 | +0.8 | **+3.7** |
| Ocean 30S-30N | 0.220 | +30.1 | +1.3 | +1.7 | **+7.9** |
| Trades SE Pacific | 0.163 | +57.4 | -2.7 | -2.5 | **+4.9** |
| Trades NE Pacific | 0.179 | +62.3 | +32.0 | +34.8 | **+43.3** |
| Trades S Indian | 0.215 | +54.0 | -3.2 | -2.2 | **-0.1** |
| Sc Peru | 0.342 | -32.1 | -72.2 | -72.7 | **-40.3** |
| Sc Namibia | 0.338 | -31.5 | -66.1 | -66.5 | **-37.7** |
| Sc California | 0.298 | -70.9 | -75.3 | -76.1 | **-72.3** |
| 30N-60N | 0.288 | +2.2 | -6.6 | -6.2 | **-4.3** |
| ocean 30N-60N | 0.276 | +9.4 | -1.3 | -1.2 | **+1.9** |
| land 30N-60N | 0.301 | -5.3 | -12.1 | -11.4 | **-10.8** |
| Southern Ocean 40-60S | 0.401 | +11.3 | +9.5 | +9.4 | **+9.5** |
| Net TOA, global (W/m2) | | -15.0 | -2.5 | -2.9 | **-5.4** |

The arms are cumulative: **cb** adds `gf_shallow_buoyancy_from_cloud_base`,
**cbqe** adds `gf_shallow_blqe_to_cloud_base` to that, and **cbqed** adds
`gf_shallow_depth_min_m = 500` to that. The Peru deck's cover is 0.64 in the
control, 0.23 under cb and 0.58 under cbqed.

**Forecast and rain:**

| | control | cb | cbqe | cbqed |
|---|---|---|---|---|
| Z500 RMSE 24 / 72 h (m) | 8.12 / 24.90 | 7.89 / 25.19 | 7.90 / 25.34 | 7.93 / 25.14 |
| T850 RMSE 72 h (K) | 1.872 | 1.919 | 1.916 | 1.899 |
| T2m RMSE 24 / 72 h (K) | 1.504 / 1.960 | 1.511 / 1.957 | 1.512 / 1.962 | 1.514 / 1.965 |
| Rain against GPCP, 48-72 h day, RMSE 60S-60N (mm/day) | 10.98 | 10.66 | 10.39 | 10.63 |
| Rain, 9-day mean field, RMSE / correlation 60S-60N | 5.06 / 0.594 | 5.34 / 0.563 | 4.99 / 0.593 | 4.90 / 0.609 |
| Z500 RMSE, 10-day mean (m) | 52.5 | 54.2 | 53.9 | 52.9 |

**What this says:**
- **The cause is the WRF sum.** With it corrected alone (cb), the excess over
  the tropical oceans goes from +30 to +1 W/m2, and the SE Pacific and S Indian
  trades come to within 3 W/m2 of CERES. The stratocumulus decks collapse,
  though: Peru's cover falls from 0.64 to 0.23, and both African and South
  American decks lose 35-40 W/m2.
- **The BLQE correction does not move the radiation.** It cuts the deck mass
  flux by about 3 times, but that is not enough to keep a thin deck. It
  improves the rain (the 9-day field RMSE goes from 5.34 to 4.99).
- **The depth minimum keeps the decks and most of the trade gain.** Under cbqed
  the global excess is +3.7 (it was +14.9), the tropical ocean +7.9 (it was
  +30.1), SE Pacific +4.9 and S Indian -0.1. Peru and Namibia stay within 6-8
  W/m2 of the control, against 35-40 under cb.
- **No material harm at 72 h.** Z500 is 0.25 m (1%) worse, T850 0.03 K and T2m
  0.005 K, while the 48-72 h rain is better. Over the ten days the Z500 mean is
  within the spread the earlier twins showed (52.9 against 52.5).

**What is still open:**
- **NE Pacific trades:** the excess stays at +43 W/m2. Shallow convection runs
  in 85% of the columns there already, so another process carries what
  remains.
- **Midlatitude land:** 30N-60N land loses 5.5 W/m2 more reflection, which takes
  the band to -4.3 against CERES. Over summer midlatitude land the cloud-base
  integral more than doubles the shallow arm's activity (5.5% to 12.5% of
  columns, four times the mean mass flux), and the depth minimum barely touches
  it.
- **The decks:** Peru and Namibia are 6-8 W/m2 darker than the control even
  with the deck columns' own plumes turned off. A probe on cbqed's own day-5
  state finds the Peru deck intact (low cloud in 84% of its columns, at 963-928
  hPa) and the shallow arm running in 2% of deck columns (Namibia 0.5%,
  California none). The residual is therefore not local venting. It comes from
  the deck's surroundings, or from the edges where the deck breaks into trade
  cumulus. The decks' own bias (too shallow, too dark) is unchanged.
- **Two cases:** August (10 days, above) and December (30 days, below). No
  longer run has been reported.

## Measured: a 30-day December twin

**Setup:**
- Two T255 arms run from GDAS 2025-12-16 00Z to 2026-01-15 00Z on the same
  config as the August twins: the 90-day anomaly+fix config with the albedo
  pair on and `[ocean] mode = "anomaly"`. It is not a shipped config.
- **on** sets all three options (`gf_shallow_depth_min_m = 500`), as the cbqed
  arm above. **off** sets none of them.
- The radiation, temperature and rain are the runs' monthly mean states (16-31
  December and 1-14 January), weighted by days. CERES EBAF 4.2.1 is the
  2000-2024 mean of each calendar month. The rain is compared with GPCP daily
  over the same 30 days.

**Reflected shortwave in excess of CERES** (W/m2 at each region's insolation):

| Region | CERES albedo | off | **on** |
|---|---|---|---|
| Global | 0.303 | +16.3 | **+5.8** |
| Ocean 30S-30N | 0.220 | +26.0 | **+3.3** |
| Trades SE Pacific | 0.170 | +48.7 | **-0.9** |
| Trades NE Pacific | 0.219 | +33.4 | **+12.5** |
| Trades S Indian | 0.191 | +40.0 | **+7.6** |
| Warm pool | 0.270 | +15.0 | **+12.4** |
| Sc Peru | 0.246 | +8.5 | **-28.8** |
| Sc Namibia | 0.238 | +46.4 | **+22.0** |
| Sc California | 0.293 | -5.7 | **-12.1** |
| Sc Australia (25-35S, 95-105E) | 0.240 | +24.5 | **-15.8** |
| 30N-60N | 0.407 | +5.6 | **+3.2** |
| ocean 30N-60N | 0.389 | +7.8 | **+2.7** |
| land 30N-60N | 0.426 | +3.9 | **+4.2** |
| 30S-60S | 0.314 | +25.4 | **+18.4** |
| Southern Ocean 40-60S | 0.352 | +30.8 | **+27.6** |
| Net TOA, global (W/m2) | | -2.6 | **+7.2** |

**Rain and drift:**

| | off | on |
|---|---|---|
| Rain, 30-day mean field against GPCP, RMSE / correlation 60S-60N | 3.30 / 0.627 | 2.94 / 0.666 |
| Rain, daily against GPCP, RMSE 60S-60N (mm/day) | 10.62 | 10.07 |
| Global mean T2m (K) | 285.78 | 285.91 |
| Drift, first day to last 10 days: T850 / T200 (K) | +0.25 / -1.83 | +0.08 / -1.87 |
| Drift, first day to last 10 days: precipitable water (kg/m2) | -0.96 | -0.75 |

**What this says:**
- **The trades vent in December too.** The SE Pacific goes from +48.7 to
  -0.9 W/m2 and the S Indian from +40.0 to +7.6. The tropical ocean goes from
  +26.0 to +3.3 and the globe from +16.3 to +5.8. The rain is closer to GPCP,
  and the 850 hPa warming over the month is a third of the off arm's.
- **The depth minimum does not keep the southern decks in this season.** Peru
  goes from +8.5 to -28.8 W/m2 and Australia from +24.5 to -15.8. A probe on
  the on arm's last state (2026-01-15, 8 firings over one day) finds the
  shallow arm running in 42% of the Peru deck's columns under cbqed, against
  79% without the depth minimum and 2% under WRF's arm. Namibia is at 0.5% and
  California at 4%. The 500 m minimum that separated the decks from the trades
  in August does not separate them in the southern summer.
- **What stays open:** the NE Pacific trades (+12.5 W/m2), Namibia (+22.0), the
  Southern Ocean (+27.6), and the decks above. The options stay off by default.

## What the shallow arm does in WRF v4.6.1

`module_cu_gf_sh.F` (carried in `core/kernels/gf.cu`, `gfd_shallow_column`)
lifts a parcel from the level of highest moist static energy, `k22`, and finds
its cloud base `kbcon` with `cup_kbcon`. It then decides the cloud top from the
vertically integrated buoyancy of the plume:

```
dbyt(k) = dbyt(k-1) + dbyo(k)*dz        k = k22+1 .. ktop      (:530)
ki      = maxloc(dbyt)                                          (:532)
ktop    = min(ktop, ki+1)
if (ktop < kbcon+1) ierr = 5                                    (:547)
```

Below `kbcon` the plume is unsaturated, so `dbyo = hco - h*` is negative by
`cup_kbcon`'s own definition of cloud base. WRF's sum starts at the source
level, so this sub-cloud deficit, which the parcel has already crossed, cancels
the positive area of the cloud layer. A plume that is buoyant from cloud base to
the inversion is rejected whenever its sub-cloud layer is deeper than its cloud
layer is buoyant. `dbyt` then never rises above the zeros below `k22`, `maxloc`
returns level 1, and the column exits on code 5.

The two later Grell-Freitas lines both start at cloud base:

- **GSL GFL** (`module_cu_gfl_sh.F90`, `cu_gfl_sh_run`, as in RRFS MPAS):
  `if(k.ge.kbcon(i)) dbyt(i,k)=dbyt(i,k-1)+dbyo(i,k)*dz`.
- **MPAS v8.4.2** (`module_cu_gf.mpas.F`, `CUP_gf_sh`): the plume is held at
  `hkb` up to `kbcon`, and `ktop` is its first level above `kbcon` with
  `dbyo < 0`.

## What the probe saw

`tools/arwen_global_shallow_probe.py` restarts the twin control (T255, GDAS
2026-08-01 00Z) at day 1. Every 3 hours, for 8 firings, it hands the model's own
cumulus inputs to the shipped kernel and to each option, and runs the shallow
stage with its capture lanes on the columns of seven regime boxes. Its
shipped-kernel call equals the model's own call to 0.0 in both the heating and
the moisture rates.

Shallow-arm activity (share of a box's open-water columns over the 8 firings),
with the arm's median cloud-base mass flux in kg m-2 s-1 where it runs:

| Box | WRF (shipped) | cloud-base integral (cb) | cb + BLQE to cloud base (cbqe) | cbqe + 500 m depth minimum (cbqed) |
|---|---|---|---|---|
| Trades SE Pacific (0-20S, 150-100W) | 18% (ierr 5: 67%), 0.039 | 81%, 0.077 | 81%, 0.058 | 71%, 0.058 |
| Trades NE Pacific (10-30N, 180-140W) | 45% (ierr 5: 47%), 0.031 | 90%, 0.044 | 90%, 0.040 | 85%, 0.039 |
| Trades S Indian (10-30S, 60-100E) | 16% (ierr 5: 66%), 0.057 | 79%, 0.138 | 79%, 0.099 | 74%, 0.100 |
| Sc Peru (10-20S, 90-80W) | 0.03% | 27%, 0.219 | 27%, 0.069 | 0.6% |
| Sc Namibia (10-20S, 0-10E) | 0.08% | 11%, 0.172 | 11%, 0.050 | 0.7% |
| Sc California (20-30N, 130-120W) | 2.6%, 0.017 | 11%, 0.042 | 11%, 0.021 | 0.9% |
| Warm pool (10S-10N, 130-170E) | 43%, 0.029 | 83%, 0.039 | 83%, 0.036 | 79%, 0.035 |
| Ocean 30S-30N (whole grid) | 29% | 72% | 72% | 64% |

- **The exit really is the sum.** In 99% of the trade columns that WRF's arm
  rejects on ierr 5, the plume is positively buoyant somewhere above cloud base.
  Under the cloud-base integral, 94-97% of them convect.
- **Where the arm then puts the water:** in the SE Pacific it detrains at the
  inversion (median ktop 834 hPa, against a cloud layer at 925-864 hPa). It dries
  the layer below 900 hPa by 5.3 g/kg/day, against 1.0 under WRF's arm, and
  moistens 900-800 hPa by 2.5 g/kg/day. In the S Indian box it vents above the
  inversion (+0.6 g/kg/day at 800-700 hPa).
- **Why the decks need the second sum.** Under the cloud-base integral the
  stratocumulus plume is two levels deep. Its cloud work function is 0.6-1 J/kg,
  against 5-8 in the trades, yet the BLQE member (blqe / (hc - he) at kbcon)
  hands it 0.17-0.22 kg m-2 s-1. That is 3-5 times the trade rate, because
  `kpbl` sits two levels above cloud base there, the WRF sum runs to `kpbl`, and
  `hc - he` is small in a well-mixed layer. Summed to `kbcon`, the deck mass flux
  falls about 3 times (Peru 0.069) while the trades lose 10-30%.
- **Why the decks need the depth minimum as well.** Even a third of that mass
  flux vents a deck that is only 300 m deep within a day. Through the twins'
  first day, Peru's cover falls from 0.43 to 0.12 under cbqe, exactly as under
  cb. The cloud depth from kbcon to ktop separates the two regimes cleanly:
  250-490 m (interquartile) under the decks, 770-1330 m in the trades and the
  warm pool. 500 m, the deep arm's own minimum for its mid-level clouds, turns
  the deck columns off and keeps 85-95% of the trade activity.

## The options

| Option (`native_adapter_options`) | Define | Reference | What moves |
|---|---|---|---|
| `gf_shallow_buoyancy_from_cloud_base` | `GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE` | GFL; MPAS v8.4.2 | the `dbyt` sum starts at `kbcon` |
| `gf_shallow_blqe_to_cloud_base` | `GF_SHALLOW_BLQE_TO_CLOUD_BASE` | GFL (MPAS v8.4.2 sums to `kbcon-1`) | the BLQE closure sums the PBL's MSE tendency over `1..kbcon`, not `1..kpbl` |
| `gf_shallow_depth_min_m` | `GF_SHALLOW_DEPTH_MIN_M` | none: this package's own, the deep arm's ierr 6 test (`module_cu_gf_deep.F:949`) on the shallow cloud | a shallow cloud less than this many metres from cloud base to cloud top exits (0 = no test) |

Each option is read only when `cumulus = "gf"`. Each joins the config identity
only when it is set (true, or a depth above 0), so every earlier checkpoint keeps
its hash. With all of them off, a 24-hour run on this tree reproduces the twin control's step-288
checkpoint array for array (135 of 135). The CPU rows are in
`tests/test_arwen_global_gf_shallow_cloud_base.py`.

The GFL shallow scheme differs from WRF's in other places too: entrainment
`1e-3` flat in place of `9e-5 (2.3 - 2 RH)`, detrainment `0.75` of
entrainment, `cap_maxs` 175, the first-guess `ktop` taken from `kpbl`, and a
`c1d` condensate detrainment below `ktop`. Its driver also calls the shallow arm
with `ichoice = 3` (the BLQE member alone), switches it off under MYNN, and
switches it off wherever deep convection did not run, so GSL leaves trade
cumulus to MYNN-EDMF. None of these was taken: the probe names the two sums
above as the ones that decide whether the trade and stratocumulus columns
convect. A fourth piece, GFL's cloud-top condensate detrainment (half of
WRF's), was built and run. Over the first day it stayed within about 1 W/m2 of
the cloud-base arm in every region, so it was taken back out.

## The boundary layer

YSU (with `ysu_topdown_pblmix = 1`) puts the trade PBL top inside the sheet:
`kpbl` is at a median of 907 hPa in the SE Pacific box, with the cloud layer at
925-864 hPa and the inversion at about 835 hPa. Its bulk Richardson number,
including the top-down extension on liquid-water potential temperature, sees a
conditionally unstable cloud layer, as it should. It does not mix that layer to
the inversion, and in WRF that job belongs to the shallow scheme. With the
shallow arm silent, nothing mixes the cloud layer and it saturates. YSU itself
follows WRF here, and nothing was changed in it.

## Reproducing

```
# the options (native_adapter_options), the cbqed arm
gf_shallow_buoyancy_from_cloud_base = true, gf_shallow_blqe_to_cloud_base = true, gf_shallow_depth_min_m = 500

# the probe, on any checkpoint of a Grell-Freitas run (restart admitted across configs)
PROBE_OUT=probe.npz PROBE_EVERY_S=10800 PROBE_MAX_FIRINGS=8 \
  python tools/arwen_global_shallow_probe.py run CONFIG --outdir DIR \
  --restart CHECKPOINT --branch-from-other-config --until-s T
```

The probe writes the shallow arm's activity, mass flux, cloud top and column
moisture, condensate and heat rates for every kernel variant in its `VARIANTS`
table. For the columns of the regime boxes it also writes the shallow stage's
capture lanes: the exit code at every stage, `kbcon`, `dbyo`, `dbyt` and the
closure members.

## Longer runs and the forecast scorecard

**90 days** (atmosphere only, the V1a anomaly+fix setup from GDAS 2026-08-01 00Z, all three switches). Compared with V1a, which has the albedo pair alone; over days 2-10 the two trees differ by 0.1 W/m2 globally.

| | switches on | V1a | CERES |
|---|---|---|---|
| global albedo, Aug / Sep / Oct | 0.288 / 0.295 / 0.304 | 0.323 / 0.326 / 0.332 | 0.278 / 0.279 / 0.291 |
| excess reflected SW over CERES (W/m2) | +3.5 / +5.4 / +4.5 | +15.0 / +15.9 / +14.2 | |
| net TOA (W/m2) | -3.8 / -0.2 / +4.0 | -13.9 / -8.5 / -4.6 | |

- **Regions, 90-day excess over CERES:**
  - ocean 30S-30N +3.5 (+23.4 in V1a);
  - trades SE Pacific / NE Pacific / S Indian: +4.5 / +21.9 / -8.3 (+47.6 / +40.9 / +38.5);
  - 30N-60N +3.0 (+10.8);
  - Southern Ocean 40-60S +17.0 (+19.1);
  - Peru / Namibia / California decks: -59.5 / -37.2 / -37.6 (-55.9 / -40.0 / -16.5).
- **Unchanged:** the temperature drift (T850 283.2 to 281.1 K in both), precipitable water (27.8 to 23.5-23.7 kg/m2) and the LR-4 energy residual (5.6-5.9 W/m2).

**30 days from a northern-winter start** (GDAS 2025-12-16 00Z, on against off):
- **Better:**
  - global excess +5.8 against +16.3 W/m2, and ocean 30S-30N +3.3 against +26.0;
  - SE Pacific trades -0.9 against +48.7, and Namibia +22 against +46;
  - rain against GPCP: 30-day field RMSE 2.94 against 3.30 mm/day.
- **Worse:** the Peru deck, -29 against +8.5. In austral summer the model's Peru plumes are 660-825 m deep, above the 500 m minimum, so the depth test does not keep the deck in that season.
- **Little changed:** the Southern Ocean summer, +27.6 against +30.8.

**Forecast skill** (16 starts, 120 h, against the GFS and IFS analyses):
- **Extratropics:** neutral.
- **Tropics:** slightly worse. T850 is +0.04 to +0.08 K at 72-120 h, and 250 hPa wind +0.22 to +0.24 m/s.
- **No subset helps:** in the 10-day twins all three subsets move tropical T850 within 0.03 K of each other, so the cost comes from running the shallow arm at all, not from either guard.

**Verdict for 0.2.3:** ship the three together, off by default in the forecast configs and on in the long-run and coupled configs. There the 10 W/m2 radiation gain decides the outcome, and the tropical 72-120 h skill matters less.

**Coupled slab** (the r019 coupled-pair setup, with the same start and the same pair-physics Q-flux F_cal, plus the three switches, at day 59 against the old arm at the same day):

| | switches on | old coupled arm |
|---|---|---|
| open-water SST minus GDAS (K), global / 10S-10N | -0.07 / -0.23 | -0.30 / -0.60 |
| open-water flux minus F_cal (W/m2), days 2-30 / 30-60 | +3.1 / -3.5 | -8.8 / -12.4 |
| net TOA (W/m2), days 30-60 | -0.7 | -9.2 |

- **Effect:** the switches cut the slab's cold drift by about two thirds over the first two months.
- **Caveat:** the F_cal is the pair calibration reused as-is, so part of the gain is extra sunlight acting on a Q tuned for less of it.
