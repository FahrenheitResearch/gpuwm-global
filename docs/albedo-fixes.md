# The albedo fixes: three opt-in options and the key set that turns them on

Arwen Global's ten-day runs reflected too much sunlight: planetary albedo
about 0.39 to 0.40 against CERES-EBAF's 0.30 for November, a reflected
excess of 32.5 to 34 W/m2 of global mean. The cause table (radiation-only
recalls on the feat/long-runs ten-day T255 run, branch `exp/attribution`)
put all of it in cloud, almost all of it in liquid cloud over the oceans
between 60S and 30N. The clear sky was within +0.6 to +0.8 W/m2 of CERES.

Three options address it. Each is off by default, and off is bit-identical
to the package before it: none of them joins the config identity at its
default, so checkpoint hashes do not move.

| Option (in `[physics] native_adapter_options`) | Off (default) | On | Kind |
|---|---|---|---|
| `morrison_droplet_number` | `"fixed"`: WRF's 250 cm-3 everywhere | `"land_ocean"`: 300 cm-3 over land, 100 cm-3 over ocean and sea ice (MPAS-Thompson's landmask values) | declared departure from WRF's Morrison; see `docs/morrison-droplet-number.md` |
| `radiation_liquid_radius` | `"scheme"`: the radiation reads Morrison's own `effc` | `"wrf_relcalc"`: the radiation reads WRF's `relcalc` radius, 14 um over ocean and sea ice, 8 um over warm snow-free land | **what WRF itself does** for Morrison |
| `radiation_cloud_inhomogeneity` | `"none"`: homogeneous cloudy subcolumns | `"tiedtke1996"`: in-cloud liquid and ice paths times 0.7 in both solvers | declared departure from WRF's RRTMG |

## The key sets

The long-run set, which a long run on this line sets
(`docs/ARWEN_GLOBAL_LONG_RUNS.md`), is the first two:

```toml
native_adapter_options = { ..., morrison_droplet_number = "land_ocean", radiation_liquid_radius = "wrf_relcalc" }
```

All three together close more of the excess and cost forecast skill (see
"The long-run set" below):

```toml
native_adapter_options = { ..., morrison_droplet_number = "land_ocean", radiation_liquid_radius = "wrf_relcalc", radiation_cloud_inhomogeneity = "tiedtke1996" }
```

The three keys combine without double counting. Under `wrf_relcalc` the
radiation no longer reads Morrison's radius. The droplet number then acts
only through the microphysics: rain formation (Khairoutdinov-Kogan
autoconversion goes as N^-1.79), droplet sedimentation and the water the
cloud keeps. Its Twomey effect on the radiated radius is not counted on top
of WRF's radius.

## `radiation_liquid_radius = "wrf_relcalc"`: WRF's own coupling

WRF v4.7.1 hands RRTMG the microphysics' own droplet radius (`has_reqc = 1`)
only for Thompson, Thompson-aerosol, the RCON scheme, NSSL 2-moment,
WSM3/5/6/7, WDM5/6/7, NU-WRF 4-ice, Jensen-Ishmael and P3
(`phys/module_physics_init.F:1000-1024`). Morrison (`mp_physics = 10`) is not
in that list. So WRF's RRTMG radiates Morrison liquid at `relcalc`'s radius
(`phys/module_ra_rrtmg_lw.F:14542-14601`; called from
`module_ra_rrtmg_sw.F:11104-11107` and `module_ra_rrtmg_lw.F:12569-12572`).
That is Kiehl's CAM3 rule:

1. 8 um, ramped to 14 um as the layer cools from 273.16 K to 253.16 K.
2. Ramped to 14 um under snow, reaching it at 0.1 m of snow water.
3. Ramped to 14 um by `1 - landm`.
4. Ramped to 14 um by the sea-ice fraction.

The inputs are `landm = 2 - XLAND`, `icefrac = XICE` and
`snowh = 0.001 * SNOW` (`module_ra_rrtmg_sw.F:11020-11023`). Here they come
from the runtime's own `xland` plane (1 + (1 - land_fraction), exactly 1 on
sea-ice columns), the analysed sea-ice fraction and Noah's snow water
equivalent. The ice keeps Morrison's own sizes. WRF's `reicalc` ice sizes
would make the cloud brighter: +3.0 to +3.9 W/m2 in the cause table.

This package had radiated Morrison's `effc` since the port. That radius is
built from WRF's continental 250 cm-3: its liquid-path-weighted median is
9.1 to 9.2 um, and 42% of liquid cells sit below RRTMGP's 2.5 um table
floor. The option makes the coupling the one WRF runs.

The first radiation call of a cold start still sizes its cloud from the
moments, as before the option. The driver reads the scheme radii only once
Morrison has run, and that gate is unchanged.

Code: `physics/cloud_radiation_options.py` (`wrf_relcalc`), and
`NativePhysicsRuntime._radiation_state`, which hands the driver a state whose
`effc` is the relcalc plane. Morrison's persistent `effc` is not touched.

## `radiation_cloud_inhomogeneity = "tiedtke1996"`: the plane-parallel bias

A cloud whose water varies inside the grid box reflects less than a uniform
cloud with the same mean water, because albedo is concave in optical depth.
Cahalan et al. (1994, J. Atmos. Sci. 51, 2434) measured an effective
thickness of about 0.7 of the mean for marine stratocumulus. Tiedtke (1996,
Mon. Wea. Rev. 124, 745) put 0.7 into the ECMWF model, on liquid and ice,
shortwave and longwave. Shonk and Hogan (2008, J. Climate 21, 2352) describe
it that way.

This package's McICA draws cloudy subcolumns from the overlap and fills each
one with homogeneous cloud, so nothing here already carries the effect.
WRF's RRTMG has no such factor, which makes this a declared departure. The
factor is the published 0.7 and nothing else: the option is a mode, not a
number.

Two caveats from the literature:

- Later work finds the right factor varies with cloud type and is closer to 1
  for optically thin cloud. Shonk and Hogan's Tripleclouds and the
  fractional-standard-deviation McICA treatments replace the single factor.
  0.7 is the operational precedent.
- The 0.7 was chosen for marine stratocumulus; applying it to ice is
  Tiedtke's choice.

Code: `RRTMGPRadiation(cloud_path_scale=...)` in `core/rrtmgp.py`, divergence
ledger row `RRTMGP-28` (decision **offer**).

## The long-run set: three seasons of 48 h twins

Twenty 48 h runs at T255 L40 from GDAS 2026-11-09 18Z, 2026-07-20 18Z and
2026-01-15 18Z, each option arm against its control from the same start,
read over hours 24 to 48 against CERES-EBAF 4.2.1's 2000-2024 mean of the
same calendar month and against the GDAS analyses 24 h and 48 h on
(`tools/arwen_global_albedo_twins.py`, subcommands `ceres`, `score` and
`table`). The runs are deterministic: the November control rerun matched
the earlier control in every flux and score. "Pair" is the long-run set.

| | Nov off | Nov pair | Nov all | Jul off | Jul pair | Jul all | Jan off | Jan pair | Jan all |
|---|---|---|---|---|---|---|---|---|---|
| Albedo (CERES 0.3029 / 0.2837 / 0.2998) | 0.3868 | 0.3387 | 0.3156 | 0.3532 | 0.3131 | 0.2922 | 0.3817 | 0.3334 | 0.3109 |
| Reflected excess over CERES (W/m2) | 29.2 | 12.5 | 4.4 | 23.0 | 9.7 | 2.8 | 28.9 | 11.9 | 3.9 |
| OLR (W/m2) | 235.76 | 236.10 | 239.99 | 242.53 | 242.85 | 246.58 | 234.37 | 234.80 | 238.61 |
| Net at top (W/m2) | -21.95 | -5.55 | -1.37 | -28.72 | -15.81 | -12.62 | -16.03 | +0.58 | +4.72 |
| Z500 RMSE 24 h / 48 h (m) | 9.530 / 18.370 | 9.468 / 18.368 | 9.531 / 18.452 | 8.344 / 14.450 | 8.245 / 14.331 | 8.293 / 14.402 | 11.087 / 20.544 | 11.013 / 20.492 | 11.029 / 20.452 |
| 2 m land temperature RMSE 24 h (K) | 2.324 | 2.323 | 2.377 | 2.285 | 2.295 | 2.353 | 2.812 | 2.815 | 2.901 |

Why the pair and not all three:

- The pair cuts the excess by 57 to 59% in every season, mostly over the
  oceans, and moves OLR by only +0.33 to +0.44 W/m2. Z500 is better in all
  six case-leads, and 2 m land temperature moves by no more than 0.012 K
  either way.
- Adding the 0.7 factor makes 2 m land temperature at 24 h worse by 0.053,
  0.068 and 0.088 K, above the 0.03 K the attribution plan allowed, in every
  case. It raises OLR by about 4 W/m2 and worsens W250 at 48 h in all three
  cases. Being uniform, it overshoots where the excess is already gone
  (July 30N-60N: 0.279 against CERES 0.298).
- About 0.03 of albedo, 9.7 to 12.5 W/m2, remains with the pair, in ocean
  cloud between 60S and 30N. Cloud cover is 0.66 to 0.71 there; no
  radiation option reaches cloud amount.

Over ten days on the long-run line (anomaly ocean, GDAS 2026-11-10 18Z,
against a control beside it) the pair held: planetary albedo 0.343 against
0.392, net at the top -4.85 against -21.33 W/m2, the daily albedo flat at
0.345 to 0.353 from day 2. `docs/ARWEN_GLOBAL_LONG_RUNS.md`, "What has been
measured", has the day-by-day reading.

## What was not built, and why

- **Open-water albedo 0.06** (cause table: -1.5 W/m2). Not built. The model's
  clear sky is already within +0.6 to +0.8 W/m2 of CERES clear sky, about the
  size of the definitional gap between the two (Loeb et al. 2020). 0.08 is
  WRF's water value. There is no measured surface excess to fix.
- **Cloud fraction** (all-or-nothing would add +4.5 W/m2; random overlap would
  add +0.5). It points the wrong way. The remaining excess sits with cloud
  amount (cover 0.73 to 0.74, 0.77 to 0.80 over sea). A sub-grid condensation
  or cloud-fraction scheme for this grid is a research project and is not
  built here.
- **Ice above the 180 um table bound** (clip against carry: 0.03 W/m2). The
  ice path above the bound is real (39%) but optically thin.
- **Reverting the 2026-09-04 ice coupling** (+3.3 to +4.3 W/m2). It is the
  physically correct area-conserving merge. Nothing shows it wrong.
- **WRF's ice sizes** (`reicalc`, +3.0 to +3.9). They make the cloud brighter.

## Measurements

All on an RTX 4090 (Linux), T255 L40 native sl_si,
from GDAS 2026-11-09 18Z, run from the shipped
`arwen_global_gdas_t255_native_24h.toml` with only the start, length and
option keys changed. The case and the off arm are the droplet-number
stage's. Off is bit-identical to `rel/0.1.6`, and that stage's off arm is
reused as the control.

**Off is bit-identical.** Ten steps with every option off on this branch
against `rel/0.1.6`: `tools/arwen_global_checkpoint_identity.py` reads
IDENTICAL at step 0 (43 of 43 arrays) and step 10 (133 of 133 arrays), with
the same `config_hash` and the same `self_sha256` (a54bde8c… at step 10).
With all three options on, the same ten steps differ, as they must.

**The 48 h runs** (hours 24 to 48, from the runs' own radiation time
integrals; `radiation_scorecard --window-hours 24 48`):

| | Off | Droplet only | relcalc + 0.7 | All three |
|---|---|---|---|---|
| Reflected SW at top, global (W/m2) | 134.85 | 128.31 | 112.10 | **110.04** |
| ... ocean | 143.07 | 133.81 | 113.20 | 110.29 |
| ... land | 114.61 | 114.77 | 109.40 | 109.40 |
| ... 60S-30S | 201.86 | 188.72 | 155.82 | 152.48 |
| ... 30S-0 | 153.54 | 143.90 | 124.57 | 120.65 |
| ... 0-30N | 111.26 | 106.74 | 94.37 | 92.80 |
| Albedo, global (CERES-EBAF Nov: 0.3029) | 0.3868 | 0.3680 | 0.3215 | **0.3156** |
| Reflected excess over CERES at this insolation (W/m2) | 29.2 | 22.7 | 6.5 | **4.4** |
| OLR, global (W/m2) | 235.76 | 236.00 | 239.79 | 239.99 |
| Net at top, global (W/m2) | -21.95 | -15.65 | -3.24 | **-1.37** |
| Total cloud cover, global | 0.7175 | 0.7166 | 0.7078 | 0.7077 |
| Grid-mean liquid water path, g/m2 (mean of the 24-48 h checkpoints) | 110.0 | 95.8 | 107.6 | 94.8 |
| Precipitation, global (mm/day) | 2.867 | 2.892 | 2.904 | 2.923 |

The "droplet only" column is the droplet stage's on arm. The other two
columns are this branch's.

**Skill against the verifying GDAS analyses** (global RMSE, the
`upper_air_scorecard.score_pair` path of `gpuwm-global score`):

| | Off | All three | relcalc + 0.7 |
|---|---|---|---|
| Z500, 24 h (m) | 9.530 | 9.531 | 9.527 |
| Z500, 48 h (m) | 18.370 | 18.452 | 18.441 |
| T850, 24 h (K) | 1.2416 | 1.2260 | 1.2366 |
| T850, 48 h (K) | 1.5222 | 1.5085 | 1.5005 |
| RH700, 24 h (%) | 12.849 | 12.820 | 12.855 |
| RH700, 48 h (%) | 18.674 | 18.703 | 18.652 |

This is one case of 48 h, so it is not a skill claim:

- T850 improves at both leads.
- Z500 at 48 h is 0.08 m worse (0.4%). That is inside the 0.3 m bar the
  droplet stage used, but it is a worsening.
- Station scoring did not run: the IEM ASOS fetch failed at the droplet
  stage and was not retried here.

**Reading.**

- WRF's radius and the 0.7 factor carry most of the change: -22.7 W/m2 of
  reflected sunlight.
- The droplet number adds -2.1 W/m2 on top, through the water. It takes the
  liquid water path from 107.6 to 94.8 g/m2, since its radius no longer
  reaches the radiation.
- The 0.7 factor on the longwave raises OLR by about 4 W/m2. CERES OLR was
  not fetched here, so that change has not been compared with an
  observation.
- Everything here is one 48 h November forecast. The 90-day run is what
  will say whether the clouds respond and give some of it back.
