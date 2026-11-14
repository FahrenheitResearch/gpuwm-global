# The mixed-layer cooling response of a prescribed sea surface

**Status: experimental, off by default.** `[ocean] mixed_layer_response =
true` lets a prescribed sea surface (`mode = "hold"`, `"anomaly"`,
`"climatology"` or `"series"`) cool under fluxes it would otherwise ignore.
With the key absent or `false` nothing of this page is built: no config
hash, restart identity, receipt row, checkpoint array or metadata moves, and
a run writes what it wrote before, byte for byte.

## What it is for

A prescribed sea surface does not answer the atmosphere. Under days of
strong, dry offshore wind its evaporation keeps the rate of the first hour
for as long as the wind blows, where a real mixed layer would cool by
several kelvin and cut the flux. In the T383 50-day runs from GDAS
2026-11-12 00Z under `mode = "hold"` (a 24 h forecast config run to 50
days) the largest 24 h open-water evaporation reached:

| where | day | 24 h evaporation, kg m-2 | situation |
|---|---|---|---|
| western Gulf of Mexico off Tampico, 23.26N 97.50W | 25-26 | 54.94 | a norther: T2 4.7 C over a sea held at 26.25 C, 21 m/s |
| northern Red Sea, 25.16N 37.19E (5070 Ti path) | 21-22 | 55.40 | easterly outflow of Arabian air, q2 5 g/kg, 20 m/s |
| Gulf of Tehuantepec, 15.8N 95.0W | 37-38 | 51.10 | a Tehuano gap jet, 24 m/s, over a sea held at 29.3 C |

The run stops at 60 kg m-2 a day on one column (the open-water evaporation
ceiling). Three places on two weather paths came within 10 %.

## The switch

```toml
[ocean]
mode = "hold"                        # or anomaly, climatology, series
mixed_layer_response = true          # default false
mixed_layer_depth_m = 40.0           # h, fixed
mixed_layer_relaxation_days = 10.0   # tau_r, back to the prescribed field
mixed_layer_max_departure_k = 5.0    # D_max, |D| <= D_max
mixed_layer_reference_days = 30.0    # the running mean that defines a normal flux
```

The parser refuses the four numeric keys without `mixed_layer_response =
true`, refuses the response under `mode = "slab"` (the slab already answers
the fluxes) and with `qflux_calibration` (the calibration records the
climatology rule's own sea surface), and the launch refuses it under
`physics.mode = "none"`, which computes no surface flux. Every key joins
the config identity while the response is on.

## The arithmetic

On every open-ocean column the boundary writes (water, not frozen, not an
inland lake, defined by the prescribed field), the response carries one
prognostic departure D from the prescribed field P, and the skin written is
`max(T_f, P + D)`, T_f = 271.35 K. Over each model step dt:

    F_ref <- F_ref + (clip(F, -1000, +1200) - F_ref) * min(1, dt / min(t - t0, tau_m))
    D     <- clip(D + dt * ((F - clip(F_ref, -600, +300)) / (rho_w c_w h) - D / tau_r), -D_max, +D_max)

* F is the net heat into open water over the interval the physics held,
  positive down: absorbed shortwave, absorbed longwave less emission at the
  skin the step saw, less sensible and latent heat (the slab's
  `open_water_flux`, on the native suite's held planes; on the reference
  suite, its held net surface flux). rho_w c_w = 1026 x 3990 J m-3 K-1.
* F_ref is the column's normal flux: the plain mean of the limited flux
  since the response began (t0, the cold start or a branch) until tau_m =
  `mixed_layer_reference_days` have passed, an exponential mean over tau_m
  after.
* A column whose flux does not change is never moved: F_ref equals F from
  the first step, so F - F_ref is zero. Over a long run the mean of F -
  F_ref is zero, so D is a departure from the prescribed field that decays
  back to it, not a drift. A diurnal cycle of 400 W m-2 moves a 40 m layer
  by about 0.05 K and averages to zero (test below).
* The two limits keep a run that starts or branches inside an extreme event
  from taking the event for normal. An instantaneous flux enters the mean at
  no more than 1000 W m-2 of loss (normal open-water nights reach 600 to 900)
  or 1200 W m-2 of gain (above any clear-sky noon). The reference is held
  within -600 to +300 W m-2, the range of monthly net surface heat flux over
  open water from western boundary currents in January to the equatorial cold
  tongue. These are constants of the design, not keys.
* A column that is land, ice or a lake at a step, or that the prescribed
  field does not define, holds D = 0 and keeps its F_ref. Sea ice is the
  prescribed field's: the response writes no ice.

D and F_ref are two float64 (nlat, nlon) planes in the physics state
(`ocean_response_sst_departure_k`, `ocean_response_reference_flux_w_m2`),
10.6 MB at T383, so a checkpoint carries them and a restart resumes them bit
for bit. The physics metadata carries `ocean_response` (`started_s`,
`last_update_s`, `advances`), and the restart identity
(`physics_metadata["surface_boundary"]`) carries the parameters, so a
checkpoint written with the response restarts only with it, at the same
parameters.

## Branching onto it

`gpuwm-global run CONFIG --restart CHECKPOINT --branch-from-other-config`
may switch the response on at a checkpoint written under the same boundary
without it (a held-sea checkpoint, for a `hold` config with the response).
The response starts there with D = 0 and F_ref = 0, and the receipt's
`branch.surface_boundary` names both boundaries. The branch's own
checkpoints carry the response, so they restart without the flag. A branch
that also changes the rule (hold to anomaly, say) is refused.

Starting in the middle of an event, the plain mean begins with the event's
flux, limited: the reference is -600 W m-2 there, so the response cools
from the first step with F + 600 W m-2, a little less than a run that had
the response on for weeks before the event (F_ref near the column's normal,
-100 to -300 W m-2).

## Rows

Each output step's row carries `ocean_response_mean_departure_k`,
`ocean_response_rms_departure_k`, `ocean_response_min_departure_k`,
`ocean_response_max_departure_k`, `ocean_response_area_cooled_1k_fraction`
(the share of the open-ocean area with D below -1 K) and
`ocean_response_mean_reference_flux_w_m2`, area means on the open-ocean
columns, beside the prescribed rule's own rows.

## Measured

On the T383 L40 50-day chain's day-25 checkpoint (GDAS 2026-11-12 00Z,
`mode = "hold"`, the runner's config, one RTX 5090), branched onto the
response at the defaults and run to day 39, beside the same code with the
response off, which reproduces the chain's own path (its watch 54.94,
51.94, 49.67 ... 51.10, 44.37 kg m-2, the chain's numbers to the digit).
Largest 24 h open-water evaporation in each box, kg m-2 (the per-column
estimate from the 3-hourly checkpoints):

| day | W Gulf (Tampico), off | on | change |
|---|---|---|---|
| 25-26 | 55.05 | 52.83 | -4 % |
| 26-27 | 52.10 | 46.71 | -10 % |
| 27-28 | 49.74 | 42.61 | -14 % |
| 28-29 | 43.10 | 36.15 | -16 % |
| 29-30 | 47.97 | 41.03 | -14 % |

The norther column cooled 0.9 K by day 26 and 1.6 to 2.2 K by days 27-29
(coldest of the run -2.76 K off Veracruz, day 30.6). The run's own watch,
largest column anywhere, days 25-39: 54.94 off, 52.71 on. After about ten
days the two arms follow different weather, so later days are not a
paired comparison; the Gulf of Tehuantepec gap jet of day 37-38 (51.10 off)
did not form at strength on the response arm. It was measured instead from
the response-off path's day-34 checkpoint, both arms on one 16 GB RTX 5070
Ti, to day 38:

| day 37-38 | off | on | change |
|---|---|---|---|
| Gulf of Tehuantepec, 16.08N 95.31W | 50.54 | 48.14 | -5 % |
| W Gulf of Mexico (Veracruz) | 44.37 | 43.14 | -3 % |
| the run's own watch, largest column | 50.54 | 48.11 | -5 % |

The jet column cooled only 0.72 K: gap winds blew on days 34-36 too, so
F_ref there had already reached -820 W m-2 (used at the -600 limit). The
northern Red Sea day (55.40, day 21-22) is on the 5070 Ti's own 50-day path,
of which no checkpoint was kept; on this path the northern Red Sea stayed
below 22 kg m-2 in every arm, its departures within -0.3 K.

The response takes 4 to 16 % off a hotspot day at the defaults, 5 to 8 %
for each kelvin of cooling, and a 40 m layer under 1,500 to 2,000 W m-2 of
loss cools about 1 K a day. It lowers the margin to the ceiling; it does not
remove the ceiling's risk.

Open-water departure over the whole globe, area weighted: mean +0.001 K
(day 26) to -0.020 K (day 38), rms 0.05 K (day 26) rising to 0.17 K (day
39), 1st/99th percentiles -0.45 / +0.49 K at day 39; cooled by more than
1 K: 0.02 to 0.07 % of the open ocean, by more than 2 K: at most 0.01 %;
warmest +0.91 K (see Limits).

Cost: two float64 planes (10.6 MB at T383). On the 16 GB RTX 5070 Ti (the
runner's T383 config, 12 h cold from GDAS): allocator peak 11.87 GiB on
against 11.85 GiB off, wall 268 s on against 271 s off (and 271 s for 0.2.1);
with the response off the 12 h run's diagnostics and in-situ rows are
byte-identical to 0.2.1 and the config hash is unchanged. On the RTX 5090,
1.000 s a step on against 0.994 s off (days 25-27); on the 5070 Ti over
the four Tehuantepec days, 2,126 s on against 2,109 s off and an allocator
peak of 11.97 GiB on against 11.95 GiB off. A 12 h T383 run on the 5070 Ti
restarted from its own 6 h checkpoint ends with a checkpoint equal to the
straight run's in all 138 arrays, the two response planes among them.

## Limits

- One fixed depth everywhere. A shallow mixed layer over a sharp
  thermocline (the Gulf of Tehuantepec, 20 m over 0.25 K/m) cools faster
  than 40 m allows; the deep winter layers of the Red Sea and the Gulf
  Stream slower. Wind-driven deepening, entrainment of cold water and
  upwelling (the larger part of a Tehuano's 4-8 K) are not modelled.
- After an event F_ref is a little below the column's normal for about
  tau_m, so the sea recovers with a small warm overshoot (about +0.3 to
  +0.7 K at most after a three-day -2000 W m-2 event at the defaults on the
  toy column). On the T383 branch above it was larger than the cooling: in
  the Pacific west of southern Baja California (24.2N 115.0W) the branch
  began inside an offshore-wind event (turbulent loss to 1,180 W m-2), F_ref
  took the event as normal (-800 W m-2 on day 27), the column cooled only
  0.35 K, and over the calm days that followed it warmed to +0.85 K by day
  38 (+0.91 K on day 39) while F_ref crept back (-293 W m-2). A branch started inside an event
  carries this for about tau_m; a run with the response on from its cold
  start has weeks of ordinary flux in F_ref before such an event.
- The response does not move the ice, and an open column cooled to T_f stays
  open water at T_f.
- It is not a coupled ocean. For a mixed layer that carries the whole
  surface energy budget, use `mode = "slab"` (docs/ARWEN_GLOBAL_OCEAN.md).

## Files

- `src/arwen_global/ocean_response.py`: the arithmetic and the provider
  that wraps a prescribed one on the boundary service.
- `src/arwen_global/boundary_options.py`: the keys, the identity and the
  receipt.
- `src/arwen_global/runner.py`: the provider's construction, the branch.
- `tests/test_arwen_global_ocean_response.py`: the parser, the arithmetic on
  known answers, the boundary on a toy planet, and the T3 reference smoke
  straight and restarted (float64 and float32, hold and anomaly) and
  branched.
