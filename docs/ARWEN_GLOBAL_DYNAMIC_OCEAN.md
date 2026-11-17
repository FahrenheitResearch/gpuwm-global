# ArWen Global: the dynamic ocean (stage 1, experimental, off by default)

`[ocean] mode = "dynamic"` couples the atmosphere to a three-dimensional ocean
with currents: a hydrostatic Boussinesq primitive-equation model on z\*
levels, Arakawa C grid, split-explicit free surface, KPP mixed layer, GM eddy
transport (Ferrari et al. 2010), the Wright (1997) equation of state and
two-band shortwave penetration.  It runs on the card beside the atmosphere,
in the same CuPy, and couples once an hour.  Without the table nothing of it
exists in a run: no array, key, hash or receipt row (every shipped config's
hash is the one v0.2.1 wrote).

This page states what was measured.  Where a gate failed it says so.

## The ocean

| | |
|---|---|
| grid | 360 x 209 x 50: 1 deg in longitude; 1/3 deg in latitude within 10 deg of the equator, a geometric ramp to 1 deg at 30 deg, 1 deg poleward; 78S to 65N |
| levels | 22 of 10 m to 220 m, tanh-stretched to 5500 m; partial bottom cells |
| sea floor | ETOPO 2022 60 s (bed for depth, surface elevation so ice shelves are land); one connected ocean flood-filled from the equatorial Pacific; Caspian and Black Sea left to the slab; hand-opened straits: Gibraltar, Bab-el-Mandeb, Hormuz, Lombok, Ombai, Timor, Makassar, Florida, Gulf of California |
| step | 3600 s (the coupling interval), 36 barotropic substeps |
| north of 65N, closed seas | the slab ocean and its thermodynamic ice, exactly as `mode = "slab"`; a 60-65N sponge toward WOA23 stands in for the Nordic Seas |
| fresh water | virtual salt flux (P - E on open water, brine from ice growth), global mean removed; SSS restoring to WOA23 at 50 m / 300 days; rivers: the land's runoff goes, mass for mass, to the nearest ocean column at once (no delay, no river network) |
| sea ice | the slab's thermodynamic ice on the atmosphere grid; under ice the ocean gives it a basal heat flux and the frazil it makes |

## Turning it on

```toml
[ocean]
mode = "dynamic"
climatology = "climatology-t255-mld.npz"   # as for the slab
slab_qflux = "qflux-t255.npz"              # the slab's, for the columns the ocean does not cover
dynamic_grid = "ocean-grid-v1.npz"
dynamic_initial = "ocean-init-2026080100.npz"
# dynamic_sss_restoring_m_per_day = 0.1667   # "none" turns it off
# dynamic_gm_kappa_m2_s = 800.0

[sea_ice]
mode = "match"            # thermodynamic, as for the slab
```

The two files are built on a Linux host:

```
gpuwm-global ocean build-grid --etopo-bed ETOPO_2022_v1_60s_N90W180_bed.nc --etopo-surface ETOPO_2022_v1_60s_N90W180_surface.nc --out ocean-grid-v1.npz
gpuwm-global ocean build-init --grid ocean-grid-v1.npz --valid 2026070200 --godas-dir godas --woa-dir woa23 --out ocean-init-2026070200.npz
gpuwm-global ocean run --grid ocean-grid-v1.npz --init ocean-init-2026070200.npz --godas-dir godas --start 2026070200 --days 30 --out spinup
gpuwm-global ocean init-from-state --grid ocean-grid-v1.npz --base-init ocean-init-2026080100.npz --state spinup/state-000720.npz --out ocean-init-spun-2026080100.npz
```

`build-init` reads NOAA GODAS pentads (`https://ftp.cpc.ncep.noaa.gov/godas/pentad/YYYY/godas.P.YYYYMMDD.grb`,
no login) and the WOA23 1991-2020 monthly and annual temperature and salinity
(NCEI).  GODAS's depth-mean velocity is dropped: kept, it drained the Amazon
shelf in two days on this grid's sea floor.  The ocean-only `run` spins the
model's own barotropic flow up under the GODAS pentad fluxes; its state is
the coupled run's start.

## Measured

Unless stated, on a shared RTX 4090 and a Linux CPU, 2026-11-14.

**Parts.**  Equation of state within 0.011 kg m-3 of the UNESCO EOS-80
one-atmosphere values.  The conservative remap closes to 0 and 1e-16 relative
at T127, T255 and T383.  The GRIB1 reader agrees with eccodes exactly.  The
grid file is byte-reproducible.  CPU and GPU agree to 1e-10 after two forced
steps.  An ocean restart is bitwise.

**Mixed layer.**  Free convection (1e-7 m2 s-3 into N2 = 1e-5) deepens within
12 % of the entrainment law.  Wind-driven deepening (Kato-Phillips, u\* =
0.01 m/s, N2 = 1e-4) is 27 % shallow at 30 h on the model's 10 m levels
(10 % at 1-2 m levels): the 15 % gate FAILS at the model's resolution.

**Idealised** (on the card).  An ocean at rest over the real sea floor stays
at rest within 2.8 mm/s after 30 days (gate 5 mm/s).  Heat, salt and volume
are conserved to 2.0e-15, 2.9e-15 and 1.9e-16 relative over 1000
random-forcing steps.  A wind-driven gyre in a flat box carries the Sverdrup
transport within 2.1 % at mid-basin (gate 10 %; -20 % at 45N and -54 % at
49N, next to the northern wall).  A front slumps with the eddy transport on,
at no volume change.  The equatorial Kelvin and Rossby waves pass their
gates as `tools/ocean_dyn_gates.py` measures them since 0.2.4 (RTX 5090, 60
days): the Kelvin crest fitted over days 10-40 runs +2.4 % against the
first baroclinic mode's c1 = 2.046 m/s (gate 3 %), and the first Rossby
crest +3.4 % against the n = 1 dispersion relation for the bump the test
launches (gate 5 %); the amplitude falls from 4.0 to 2.75 K in 60 days.
The model is the one that failed the earlier form of the gate (Kelvin
+6.6 %, Rossby -10.9 %): that form fitted days 5-25, when the crest still
overlaps the 8-degree source bump (every window starting at day 10 or later
reads +1.7 to +2.7 %), and held the Rossby crest to c1/3, the long-wave
limit, which a bump only three deformation radii wide does not reach (its
exact n = 1 crest moves at -13.8 % of c1/3 over days 10-40).

**Spin-up from GODAS** (2026-07-02 to 08-01, GODAS fluxes), against GODAS on
2026-08-01: Nino3.4 SST +0.23 K, tropical (10S-10N) SST bias +0.27 K (rms
0.79), global +0.75 K (rms 1.71); D20 rms 9.2 m in 10S-10N; Equatorial
Undercurrent at 140W 0.73 m/s at 95 m (GODAS 0.76 at 85 m); Drake 111 Sv,
Indonesian throughflow 11 Sv.  Heat, salt and volume books close to 4e-16.

**Off is byte-identical.**  The 55 shipped run configs hash as under v0.2.1, and
a T127 6-hour run of the shipped native config without an `[ocean]` table
writes the same checkpoints as v0.2.1, all 317 arrays byte for byte, on the
same card.

**Coupled restart.**  The T3 smoke under the dynamic ocean, restarted inside a
coupling interval, equals the straight run byte for byte in float64 and
float32.  On the card at T255 (RTX 4090, GDAS 2026-08-01, 24 steps), runs
restarted at 2700 s (inside a coupling hour) and at 3600 s (on it) write the
straight run's 177 checkpoint arrays byte for byte.  That card test first
found a restart that was not bitwise (the first ice record was skipped at a
cold start and applied at a restart); it is fixed and the smoke test now
covers it.

## Coupled at T255

From GDAS 2026-08-01 00Z (T255 L40, the native suite with orographic drag,
5-day leases on an RTX 5090), against the slab twin of the same start (the
slab carries the long-run Q-flux correction; the dynamic ocean has none) and
OISST v2.1, open water only:

| daily mean | SST bias 60S-60N (dynamic / slab) | 10S-10N | 30N-60N | Nino3.4 (dynamic / slab / OISST) | EUC at 140W (model / GODAS) |
|---|---|---|---|---|---|
| 08-01 | -0.55 / -0.56 K | -0.37 / -0.46 | -0.79 / -0.76 | 29.23 / 29.13 / 29.50 C | 0.73 / 0.76 m/s |
| 08-05 | -0.57 / -0.54 | -0.34 / -0.46 | | 29.25 / 28.99 / 29.48 | 0.81 |
| 08-10 | -0.66 / -0.58 | -0.40 / -0.55 | -1.19 / -0.68 | 29.19 / 28.78 / 29.46 | 0.74 / 0.71 (08-13) |
| 08-15 | -0.71 / -0.62 | -0.40 / -0.62 | -1.54 / -0.68 | 29.20 / 28.62 / 29.43 | 0.67 |
| 08-20 | -0.75 / -0.63 | -0.42 / -0.61 | -1.57 / -0.76 | 29.21 / 28.50 / 29.32 | 0.50 / 0.57 (08-23) |
| 08-25 | -0.80 / -0.68 | -0.49 / -0.69 | -1.78 / -1.02 | 29.16 / 28.36 / 29.36 | 0.34 |
| 08-30 | -0.82 / -0.69 | -0.55 / -0.78 | -1.77 / -1.04 | 29.13 / 28.26 / 29.42 | 0.26 / 0.63 (09-02) |

RMS (60S-60N) on 08-30: dynamic 1.31 K, slab 1.13 K.  D20 on the equatorial
Pacific on day 30: 148 m west and 123 m east (GODAS 09-02: 146 and 131).

Both arms start about 0.55 K below OISST because the model starts from the
GDAS skin, which sits below OISST.  Over the run the dynamic ocean holds the
tropics and Nino3.4 better than the slab (Nino3.4 within 0.3 K of OISST
against the slab's 0.4-1.2 K), and cools the northern mid-latitudes faster
(about 0.8 K below the slab by day 25): it has no Q-flux, so it takes the
atmosphere's open-water deficit in full.  The undercurrent at 140W weakens
as GODAS's does over August, but far faster: 0.26 m/s on day 30 against
GODAS's 0.63, a stage-1 failure to look into.  Heat, salt and volume books
close to rounding every hour; the remap books a residual of -2.7e20 J over
the 30 days (about -0.3 W m-2 over the ocean).  Every lease passed the long-run gates (12 of 12).  This
run used the code before two restart fixes (the first ice record, see
Coupled restart, and a restart identity that named the ocean's files by
path) and before river routing.

## Cost and memory

The ocean step: 0.13-0.25 s per ocean hour on the shared RTX 4090, 0.50 s
per coupling (fluxes, remap and step) at T255 on an RTX 5090.  Held on the
card: 310 MiB (state and static grid) plus 11 float64 planes on the
atmosphere grid; the step's transient peak is 0.77 GiB above that and is
freed before the atmosphere steps.

| one model day, GDAS 2026-09-01 00Z | step | allocator peak (live / pool) | card peak (nvidia-smi) |
|---|---|---|---|
| T383 on the RTX 5070 Ti 16 GB, no ocean coupling | 1.802 s | 11.95 / 13.45 GiB | 14,218 MiB |
| T383 on the RTX 5070 Ti 16 GB, dynamic ocean | 2.048 s (+13.7 %) | 12.39 / 14.20 GiB | 14,990 MiB of 16,303 |
| T255 on an RTX 5090, slab ocean (2026-08-01, days 0-5) | 0.660 s | | |
| T255 on an RTX 5090, dynamic ocean (same start and days) | 0.736 s (+11.6 %) | 7.12 GiB live | |

T383 keeps its one-band plan with the ocean on and fits the 16 GB card with
1.3 GB to spare.  The cost is well above the design's estimate (under 1 %)
and is not yet broken down; the T383 arm without coupling also leaves out the
slab's own services, so part of its 13.7 % is the slab's.  The T383 pair
uses a stand-in climatology and no Q-flux (memory and cost only), and ran
the T383 day as 0.2.1 shipped it (radiation every 2100 s, no orographic
drag); 0.2.3's T383 day (radiation every 3000 s, drag on) has not been run
with the dynamic ocean.

## Limits (stage 1)

- Rivers arrive at once at the nearest ocean column: no river network, no
  delay.  The 30-day coupled run below predates the routing and ran without it.
- No Redi diffusion, Langmuir mixing, tidal mixing, chlorophyll shortwave or
  diurnal warm layer (stage 2).  No Arctic ocean, no moving ice (stage 3).
- The tape carries SSH, the top cell's current, the KPP depth, D20 and the
  0-300 m heat content beside the SST, refreshed hourly.
- A free coupled ocean inherits the atmosphere's surface-flux bias (8-14
  W m-2 less into open water in long runs, measured with the slab).

## Fixed in 0.2.4

- **The stress the ocean receives is the stress the atmosphere loses.** The
  coupler handed the ocean 1.2 u\*², about 8.5 % more momentum than YSU
  removes from the lowest level.  It now hands over YSU's own sink,
  rho u\*² (|V1|/wspd)² at the actual air density (`ysu.cu` 252 and
  687-689), from a plane the surface layer writes after each call
  (`pbl_surface_stress_n_m2`).  Only a dynamic-ocean run carries that plane;
  every other run's namespace, checkpoints and arithmetic are unchanged.  The
  daily stress product reports the same quantity.  Measured on the EUC at
  140W: at most 0.001 m/s over days 16-20 of a paired restart.
- **Stress, ice cover and the start's SST match take the plain overlap
  mean.**  The conservative remap's spread residual, right for heat and fresh
  water, is a uniform shift on an intensive field: at T255 the owned columns
  cover 3.4445e14 m² against the wet ocean's 3.4717e14 m², so the spread
  moved every wet cell by about -0.78 % of the field's global mean.  Heat,
  fresh water and the ice-growth salt keep the spread, and it is still
  booked.  In a 15-day coupled pair from 2026-08-01 the global open-water SST
  bias against OISST moved by less than 0.005 K by day 10; the tropics
  (10S-10N) read 0.01-0.04 K colder from day 1, because the old spread had
  warmed the start's SST match by about 0.09 K at 29 C and hidden that much
  of the coupled run's tropical cold bias.
- **The wave gates** (Measured, above): the model is unchanged; the gate's
  fitting window and its Rossby reference were wrong.

## Open problems (as shipped in 0.2.4)

The dynamic ocean ships experimental and off by default because these are
not solved:

- **The equatorial undercurrent still weakens** in a coupled run from
  2026-08-01: at 140W it falls from 0.73 to 0.26 m/s over 30 days (0.28
  with 0.2.4's coupler and `sfclay_ocean_roughness = "coare35"`), against
  GODAS's 0.76 to 0.63.  The ocean alone, from the same state, reproduces
  it under the coupled run's own daily stress (0.31 m/s at day 30) and holds
  0.61 under GODAS's, so the cause is the atmosphere's equatorial zonal
  stress, not the coupling.  Against a matched reference (the same ocean
  forced by GFS-analysis stress), six coupled starts from 2026-08-01 to
  08-11 (with the coupler fix and COARE on) held 0.85-1.00 of the
  reference's EUC at days 5 and 10 (means 0.97 and 0.93), and the two that reached day 20 held
  0.88 and 0.92; that is short of the eight starts at day 20 the test asked
  for.  ArWen makes the mid-August west-Pacific westerly burst only when it
  starts 2-8 days before it, weaker than the analysis (area-mean peak +3.1
  to +3.8 m/s against +4.9), and every start is 0.5-1.2 m/s too easterly
  there over days 1-5.  The six starts share one El Nino month and one
  burst.
- **Wind-driven mixed-layer deepening** is 27 % too shallow against the
  Kato-Phillips case (15 % gate) on the model's 10 m levels; the
  heat-conserving mixed depth is 10.9 % shallow, inside the gate, and
  convective deepening passes.  The likely fix (LMD94 appendix D's enhanced
  diffusivity at the interface below hbl) is not attempted.
- **The 30-day coupled run predates the 0.2.4 fixes**, so it shows
  stability, not those fixes.
- **`sfclay_ocean_roughness = "coare35"` joins the physics identity,** so a
  slab Q-flux calibrated under WRF's Charnock (every Q-flux file so far) is
  refused with it on; a coupled run with COARE needs a Q-flux calibrated
  with it.
- **SST against OISST** on day 30 (2026-08-30): 60S-60N -0.82 K against the
  slab's -0.69 K, and 30N-60N -1.77 K against -1.04 K; 10S-10N -0.55 K
  against -0.78 K, and Nino-3.4 29.13 C against the slab's 28.26 C (OISST
  29.42 C).
- A run is refused before its first step when `dt_s` does not divide 3600 s
  (the coupler takes exactly one hour per coupling) or when it asks for more
  than one card.
- Cost: +11.6 % a step at T255 and +13.7 % at T383; the T383 day as 0.2.1
  shipped it (radiation every 2100 s, no drag) fit the 16 GB RTX 5070 Ti at
  a card peak of 14,990 of 16,303 MiB.  0.2.3's and 0.2.4's T383 days were
  not measured with the dynamic ocean.
