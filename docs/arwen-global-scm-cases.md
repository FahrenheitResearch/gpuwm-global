# Arwen Global: the native suite on published single-column cases

`tools/arwen_global_scm_cases.py` runs the model's own physics suite (the
`arwen-native` suite: RRTMGP, sfclay, YSU, Grell-Freitas, Morrison) on
published single-column cases and scores its cloud, rain, surface fluxes and
radiation against the ranges the cases' large-eddy simulation (LES)
intercomparisons published.  It is the single-column bench of the scheme
work: a case runs in seconds on a card, so a scheme change can be scored on
every case before it is run in a forecast.

Nothing in the package changes.  The bench drives
`ArwenCudaColumnSuite`, the object a forecast calls, through the same
`modules` door the CPU tests use for their fakes.

## What one case run is

- **The column.** The case's initial profiles laid on the shipped 40-level
  stack (`arwen_global_gdas_t255_native_sl_si_24h.toml`) at the case's
  surface pressure, heights from the coordinate's own hydrostatic integral.
  Above a case's top level the column joins the US standard atmosphere
  through an offset that decays with a 3 km e-folding height; vapour decays
  from its top value with a 2.5 km e-folding height and never exceeds 30
  percent of ice saturation, so the joined column holds no cloud of its own.
- **The step.** The forecast's 300 s step and its Strang split: the suite
  over 150 s, the case's large-scale forcing over 300 s, the suite over
  150 s, with the shipped options (radiation every 3000 s, cumulus
  Grell-Freitas, `dx_m` 52,100 m).
- **The forcing**, in place of the dycore: subsidence (first-order upwind in
  height on theta and every water species), the prescribed horizontal
  advection of temperature and water, the geostrophic (Coriolis) term
  integrated exactly, wind nudging where the case asks for it, and the
  subsidence's pressure velocity on the half levels (which Grell-Freitas
  reads).
- **The surface.** The case's sea-surface or skin temperature.  Where the
  case prescribes its surface fluxes (and friction velocity or roughness),
  the shipped sfclay runs and the case's values are written over its result
  on every call.
- **Two arms.** `spec` is the case as published: its radiation is the
  case's (a prescribed tendency, a formula in liquid water path, or none),
  and RRTMGP still runs with its heating dropped, so its fluxes are read on
  the model's own cloud.  `native-radiation` keeps RRTMGP's heating and
  drops the case's radiation; it is offered only where a case states its
  radiation apart from its advection.
- **The sun.** In the `spec` arm of a case whose radiation is prescribed,
  `--sun noon` moves RRTMGP's clock so that the middle of the scoring
  window is local solar noon.  The heating is not applied in that arm, so
  only the readings move: the model's reflected sunlight for the cloud the
  model made.
- **Two identical columns.** Every case runs as two identical columns of
  one batch; each result records the largest difference between them
  (`column_spread_theta_k`, `column_spread_qc`), which must be zero.
  New Tiedtke takes each column's grid length from the batch's latitude
  rings, so under `cumulus=ntiedtke` the case runs on two rings of the T255
  grid's length (768 columns) and spacing (180/384 degrees), centred on the
  case latitude, and is read at the first column.
- **Rain** is the microphysics' surface accumulator plus the cumulus
  scheme's `rainc`, each also reported on its own.

## The cases and where they come from

| Case | Regime | Forcing read from | Length, scoring window |
|---|---|---|---|
| `dycoms-rf01` | nocturnal marine stratocumulus | Stevens et al. 2005 (Mon. Wea. Rev. 133, 1443), transcribed in the tool | 4 h, hours 3 to 4 |
| `bomex` | trade-wind shallow cumulus | DEPHY-SCM `BOMEX/REF` | 6 h, hours 3 to 6 |
| `rico` | precipitating shallow cumulus | DEPHY-SCM `RICO/SHORT` | 24 h, hours 20 to 24 |
| `fire` | diurnal marine stratocumulus (FIRE I, EUROCS) | DEPHY-SCM `FIRE/REF` | 48 h, hours 24 to 48 |
| `armcu` | diurnal shallow cumulus over land (ARM SGP) | DEPHY-SCM `ARMCU/REF` | 14.5 h, last hour |
| `lba` | shallow-to-deep transition over land (TRMM-LBA) | DEPHY-SCM `LBA/REF` | 6 h, hours 5 to 6 |
| `gabls1` | stable boundary layer | DEPHY-SCM `GABLS1/REF` | 9 h, hours 8 to 9 |

The DEPHY-SCM files are the community single-column format
(https://github.com/GdR-DEPHY/DEPHY-SCM); the tool reads each case's
`*_DEF_driver.nc`, and each result records the file's sha256.  DYCOMS-II
RF01 is not in DEPHY-SCM.

Stated departures, each recorded in the result's `departures` list:

- The land cases (`armcu`, `lba`, `gabls1`) run as water columns to YSU and
  the cumulus scheme, because their fluxes are prescribed and Noah would
  overwrite them; the surface layer runs its land branch with the case's
  roughness.
- A case that prescribes its fluxes and no skin temperature (`armcu`,
  `lba`) gets a skin diagnosed on every call from the prescribed sensible
  heat flux through the surface layer's own heat exchange coefficient,
  within 20 K of the lowest level, so the stability sfclay and YSU read
  agrees with the flux they are handed.
- Advection of liquid-water potential temperature and of total water is
  applied to theta and to vapour.
- DYCOMS-II RF01's radiation formula reads the inversion density from the
  column at the diagnosed 8 g/kg total-water isoline, and its flux is held
  constant above 1500 m.

## First readings (2026-11-12, one RTX 4090, the shipped suite)

Each case is one run of the shipped options, `spec` arm unless named;
"window" is the case's scoring window, "run" the whole run after its first
hour.  Cloud cover is the radiation's total cloud cover, which the suite
holds between its 3000 s radiation calls.  Every pair of identical columns
came back identical (spread 0 in all 39 results).  A case runs in 2 to 59 s
of wall time after the kernels are compiled (`arwen-massflux-v1`, which is
array code, 20 to 255 s).

The readings were taken again on 0.1.7, on the same card model, with the
six case files downloaded by `fetch` (below): every one of the 39 results
is bit-equal to the first readings in every per-step reading and every
stored profile, so every number on this page is 0.1.7's.

| Case | Reading | Model | Published LES range | Verdict |
|---|---|---|---|---|
| DYCOMS-II RF01 | liquid water path, hours 3 to 4 | 70.4 g/m2 | about 5 to 65, mean about 35 | above |
| | the same, `native-radiation` | 45.6 g/m2 | as above | inside |
| | cloud fraction | 0.999 | about 0.2 to 1.0; observed 1.0 | inside |
| | boundary-layer longwave cooling below 1.5 km, hours 3 to 4 | RRTMGP -108 W/m2, the case formula on the same column -84 W/m2 (`native-radiation`: -106 and -83) | | |
| | top-of-atmosphere albedo with the sun at local noon (`--sun noon`) | 0.45 | | |
| BOMEX | total cloud cover, hours 3 to 6 | 0 | 0.08 to 0.17, mean 0.13 | below |
| | liquid water path | 0 | about 3 to 13 | below |
| | surface rain (a non-precipitating case) | 0.34 mm/day, all Grell-Freitas | | |
| RICO | cloud cover, hours 20 to 24 | 0.21 | 0.09 to 0.38, mean 0.19 | inside |
| | the same over the run | 0.83 (1.0 from hour 1.75 to hour 20.8) | | |
| | liquid water path, hours 20 to 24 | 1.0 g/m2 (run: 33) | 4.8 to 47.7, mean 19.4 | below |
| | rain water path | 0.00001 g/m2 | 0.1 to 20.6, mean 3.8 | below |
| | surface rain | 2.09 mm/day, all Grell-Freitas | 0 to 1.08, median 0.26 | above |
| | sensible, latent heat flux | 10.0, 151 W/m2 | 5.9 to 14.5; 104 to 189 | inside, inside |
| | cloud base; inversion | 781 m; 1881 m | 120 to 751; 1919 to 2645 | above; below |
| | top-of-atmosphere albedo, sun at noon at hour 10 | 0.52 (cover 1.0, 98 g/m2) | | |
| GABLS1 | boundary-layer height, hours 8 to 9 | 268 m (YSU's) | about 160 to 200 | above |
| | surface wind turning | 32.4 degrees | 36, bar about 34 to 38 | below |
| FIRE | liquid water path, hours 24 to 48 | 160 g/m2 (from 40 by day to 227 by night) | not read | |
| | cloud cover; sunlit top-of-atmosphere albedo | 1.0; 0.55 | not read | |
| ARM SGP | liquid water path, last hour; over the run | 67.8; 27.6 g/m2 | not read | |
| | cloud cover: hours 0 to 9; from hour 10 | 0; 1.0 | not read | |
| TRMM-LBA | surface rain, hours 1 to 2; hours 3 to 6 | 7 to 9; 0.2 to 0.7 mm/day, all Grell-Freitas | not read | |
| | liquid water path at hour 6; cloud top | 421 g/m2; 1.9 km | not read | |

Where the ranges come from, read for this page:

- BOMEX: Siebesma et al. 2003 (J. Atmos. Sci. 60, 1201-1219), p. 1204 for
  the cloud cover (the models' 3 h means run from 0.08 to 0.17, ensemble
  mean 0.13) and Fig. 2b for the liquid water path, read from the figure.
- RICO: vanZanten et al. 2011 (JAMES 3, doi:10.1029/2011MS000056), Fig. 3
  (the full ensemble's minimum, four-hour mean and maximum over hours 20 to
  24) and Table 3 (the thirteen simulations' surface precipitation, 0.0 to
  31.2 W/m2, converted at 2.5e6 J/kg).
- DYCOMS-II RF01: the Stevens et al. 2005 ensemble as Pressel et al. 2017
  (JAMES 9, doi:10.1002/2016MS000778) replot it in Figs. 2 and 3, read from
  the figures; the observed cover is from their p. 1350.
- GABLS1: Holtslag et al. 2012 (the ECMWF GABLS workshop overview), p. 13
  and Fig. 2b, read from the figure.
- FIRE, ARM SGP and TRMM-LBA: no published range was read for this page.

What the readings say, as far as one column of each case can:

- **The trade-cumulus regime has no partial cloud.**  BOMEX makes no cloud
  water at all at 40 levels and rains from Grell-Freitas in a case that does
  not rain; RICO goes overcast (cover 1.0, up to 103 g/m2 on one to three
  levels) for 19 of its 24 hours and reflects 52 percent of the noon sun
  there, where the LES cover is about a fifth.  Grid-scale saturation in
  Morrison and the radiation's cloud fraction on it give all or nothing.
- **Shallow convection rains, and the warm cloud does not.**  Grell-Freitas
  rains 2.1 mm/day in RICO's last four hours (twice the wettest LES) while
  Morrison, with its droplet number fixed at 250 per cm3, makes almost no
  rain water from 30 to 100 g/m2 of cloud.
- **Stratocumulus holds.**  DYCOMS-II RF01 keeps a full deck; its liquid
  water path is inside the LES range when RRTMGP drives it and above it
  under the case's own radiation, and RRTMGP cools the boundary layer by
  27 percent more than the formula fitted to the observations.
- **The stable boundary layer is too deep**, by about half, as the GABLS
  overview reports for operational schemes.

### The same cases under the other cumulus settings

Window means, `spec` arm, `--option cumulus=...`; liquid water path in
g/m2, cover, surface rain in mm/day.

| Case | Grell-Freitas (shipped) | none | `arwen-massflux-v1` (`own`) | New Tiedtke |
|---|---|---|---|---|
| BOMEX | 0, 0, 0.34 | 21.7, 1.0, 0 | 0, 0, 0.45 | 0, 0, 0.32 |
| RICO | 1.0, 0.21, 2.09 | 269, 1.0, 0.002 | 0, 0.21, 0 | 0, 0, 0.70 |
| TRMM-LBA | 338, 1.0, 0.39 | 340, 1.0, 0 | 0, 0, 15.6 | 209, 1.0, 4.14 |
| ARM SGP | 67.8, 1.0, 0 | identical to Grell-Freitas | 0, 0, 0 | 0, 0, 0 |
| FIRE | 160, 1.0, 0.03 | 171, 1.0, 0.005 | 168, 1.0, 0.01 | 111, 1.0, 0.18 |
| DYCOMS-II RF01 | 70.4, 1.0, 0 | identical to Grell-Freitas | identical | 240, 1.0, 0.02 |
| GABLS1 | no cloud under any of the four | | | |

Grell-Freitas never acts on the ARM column (every stored profile of every
step equals the no-cumulus run's), and neither it nor `arwen-massflux-v1`
acts on the stratocumulus or the stable case (the same equality); New
Tiedtke does act on DYCOMS-II RF01 and more than triples its liquid water
path.  Without a cumulus scheme every trade-wind case goes overcast at the
grid scale; `arwen-massflux-v1` and New Tiedtke clear the trade-wind cases
entirely, and `arwen-massflux-v1` rains the LBA column at 15.6 mm/day.  In RICO
New Tiedtke rains 0.70 mm/day, inside the LES range; Grell-Freitas rains
twice the wettest LES and the other two almost nothing.  None of the four
gives the partial cover the LES give.

## Running it

The bench is a research tool and is EXPERIMENTAL: its cases, departures
and reference rows may change between releases.  It runs from a checkout
of this repository on a host with a CUDA card, in an environment with the
package installed and `netCDF4` beside it (`pip install netCDF4`); a case
takes well under 1 GB of the card.

Download the six DEPHY-SCM case files once.  `fetch` takes them at the
commit the readings on this page were taken on and refuses a file whose
sha256 is not the pinned one:

    python tools/arwen_global_scm_cases.py fetch --out dephy

Run every case in both arms.  The run prints one line of readings per case
as it finishes and ends on the score table, each reading against its
published range:

    python tools/arwen_global_scm_cases.py run --case all --dephy-root dephy \
        --arm spec --arm native-radiation --out scm-results

`--case` repeats (`--case rico --case bomex`).  `--dephy-root` also reads
a checkout of the DEPHY-SCM repository itself (`BOMEX/REF/...`).  The
table is printed again, from the result files, by

    python tools/arwen_global_scm_cases.py score scm-results/*.json

and `summary` prints one row of readings per result, the window and the
whole run beside each other.

`--option KEY=VALUE` overrides one native physics option (for example
`--option cumulus=ntiedtke`) and `--label` names the set in the result file
names, so a scheme change is scored by running the bench twice and reading
the two `score` tables side by side.

The known answers are in `tests/test_arwen_global_scm_cases.py` (CPU): the
DEPHY reader at hand-computed heights, the upwind subsidence on a linear
profile, the exact geostrophic turn, the saturation split, the DYCOMS-II
formula's fluxes, the forcing alone adding exactly its tendency, the
suite seeing the case's prescribed fluxes through the substitution, the
`spec` arm dropping RRTMGP's heating and the `native-radiation` arm keeping
it, the score table, and `fetch` keeping only the pinned bytes.
