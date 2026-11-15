gpuwm-global 0.2.4

Arwen Global 0.2.4 changes no default. Every config 0.2.3 shipped keeps its
config hash, and the T255 forecast day writes 0.2.3's checkpoints byte for
byte. The fixes are to the experimental dynamic ocean: the stress it
receives, how its grid remap treats stress and ice, and one of its tests.
The rest is new options, all off unless a config turns them on: an
open-water roughness from COARE 3.5, two snow albedo options, and two
experimental switches on Grell-Freitas's deep convection. It needs
`gpuwm>=2.9.13,<2.10`, as 0.2.3 did, and the Rust doors are 0.2.3's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Fixed in the dynamic ocean (experimental, off by default)

- **The ocean receives the stress the atmosphere loses.** The coupler gave
  the ocean 1.2 u\*^2, about 8.5 % more momentum than the boundary layer
  took out of the wind. It now gives it the boundary layer's own surface
  drag at the real air density. Only a dynamic-ocean run carries the new
  stress plane; nothing else changes. On its own it moved the equatorial
  undercurrent by at most 0.001 m/s.
- **Stress and sea ice are remapped without the heat-flux fix-up.** The
  remap between the ocean grid and the atmosphere's grid spreads a small
  leftover over the whole ocean so that heat and fresh water add up. On
  stress, ice cover and the starting sea surface temperature that leftover
  is a uniform shift, about 0.78 % of the field's global mean on every
  ocean cell at T255. Those three now take the plain overlap mean; heat and
  fresh water keep the fix-up. In a 15-day coupled pair the global sea
  surface temperature against OISST changed by less than 0.005 K at day 10;
  the tropics read 0.01-0.04 K colder, because the old shift had warmed the
  start by about 0.09 K and hidden that much of the coupled run's cold
  tropics.
- **The Kelvin and Rossby wave test was wrong, not the ocean.** It fitted
  the Kelvin wave while it still overlapped its source, and held the
  Rossby wave to a speed that a disturbance of that width cannot reach.
  Corrected, the same model passes: Kelvin +2.4 % (gate 3 %), Rossby
  +3.4 % (gate 5 %), over 60 days.
- **The undercurrent over several starts.** Against the same ocean driven
  by the analysis winds, six coupled starts in August 2026 kept 0.85 to 1.00
  of its equatorial undercurrent at days 5 and 10 (0.97 and 0.93 on
  average), and the two that ran to
  day 20 kept 0.88 and 0.92 (with the stress fix and the COARE option on).
  The test asked for eight starts at day 20, so it is not passed. The
  single August 1 start still falls to 0.28 m/s by day 30. The model makes
  the mid-August west-Pacific westerly burst only from 2 to 8 days ahead,
  and weaker than the analysis. The dynamic ocean stays experimental:
  `docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md` lists what is still open.

## New, off unless you turn it on

- **COARE 3.5 open-water roughness** (`sfclay_ocean_roughness =
  "coare35"`). WRF's surface layer puts more drag on the open ocean than
  COARE at trade-wind speeds. The option cuts that drag by about 14 %, but
  the winds speed up 2-5 % in reply, so the stress on the equatorial ocean
  falls by only 2-6 %. Forecast scores with it were not measured. It
  changes the physics identity, so a slab ocean's Q-flux made without it is
  refused with it on.
- **Fresh snow at the observed brightness** (`snow_fresh_albedo =
  "masked"`). WRF makes fresh snow brighter than the satellite maximum it
  starts from, most of all in forests, so snowy land reflects too much
  sunlight. The option keeps the satellite maximum as the fresh value. In
  10- to 15-day T127 tests from February, March and April 2026 it closed 57
  to 81 % of the northern land albedo excess left against ERA5, and northern
  land got 0.3 to 0.9 K warmer at 2 m. It is not a recommended default:
  from a November start it closed only about 20 %, and in January it made
  wooded tundra, Canada and the needleleaf forests darker than ERA5 (wooded
  tundra 0.557 against 0.646). One start a month, no skill scores.
- **Melting snow ages faster** (`snow_thaw_albedo = true`), on the curve WRF
  carries commented out. Its effect is small (-0.004 to -0.006 albedo).

## Experimental, not recommended

- **Where Grell-Freitas's deep plume peaks** (`gf_deep_experiment` = 2, 3
  or 4; 0 is WRF's). WRF's placement keeps tropical cloud tops low, which is
  why the upper tropical troposphere cools by about 0.7 K at 300 hPa in a
  day. Moving the peak lifts the tops and removes that 24 h cold bias, and
  lowers NH 500 hPa height error at 24 h on 8 of 8 starts, but that gain is
  the mean height lifted, not a better pattern: with the bias removed the
  error gets slightly worse. It doubles the
  scheme's tropical rain, warms 700-500 hPa by 0.3-0.5 K, makes tropical
  850 hPa temperature and 250 hPa wind worse on every start, and by day 3
  the cold bias has turned warm.
- **Convective momentum transport for Grell-Freitas**
  (`gf_momentum_transport = true`), as WRF couples it. Tropical 250 hPa
  wind improved slightly (3.79 to 3.71 m/s) and 500 hPa height got worse
  on 4 of 4 starts.

## For long runs

- `docs/ARWEN_GLOBAL_LONG_RUNS.md` now explains the year run's -17 W/m2
  energy imbalance: it is the imbalance of the physics the Q-flux was
  calibrated with, not an ocean drift. Twelve calibrations with the 0.2.3
  long-run recipe read -0.55 W/m2 on average against -11.32 for the year
  run's physics, inside the gate's [-4, 6]. A long run with the recipe
  needs a Q-flux calibrated with the recipe; the page says how one is made.
- **The recipe's Q-flux is attached to this release**
  (`qflux-t127-recipe-r5.npz`), with the 90-day T127 slab config it was
  calibrated for (`recipe-slab-t127.toml`). Build the climatology it names
  with `gpuwm-global climatology fetch --with-mld` and
  `gpuwm-global climatology build arwen_global_gdas_t127_native_sl_si_24h --with-mld --out climatology-t127-mld.npz`:
  on 0.2.4 that rebuilt the climatology byte for byte, and the slab's guard
  accepted the Q-flux for `recipe-slab-t127.toml`. A config with other native options is
  refused and needs its own calibration.

## Measured for this release

- **Unchanged where nothing was turned on.** On an RTX 5070 Ti, the T255
  forecast day under 0.2.4 wrote 0.2.3's nine checkpoints byte for byte,
  and 0.2.4 restarted from a 0.2.3 checkpoint continued 0.2.3's run byte for
  byte. The T383 day wrote 0.2.3's five checkpoints byte for byte. Every
  shipped config hashes as in 0.2.3.
- **T383 on the 16 GB card, with the defaults.** 1.72 s a step and a card
  peak of 14,384 MiB of 16,303 (0.2.3 on the same card in the same lease:
  1.74 s and 14,152 MiB).
- **The recipe Q-flux.** On 0.2.4 the documented tool rebuilt the
  climatology the file names byte for byte, and the slab's guard accepted
  the file for `recipe-slab-t127.toml` (CPU check, no run).
- **The dynamic ocean restarts bit for bit with the new stress.** A T255
  coupled run restarted inside a coupling hour, and on one, wrote the
  straight run's checkpoint byte for byte (RTX 4090).
- **Tests.** 3520 CPU tests pass from the installed wheel, with CuPy (90
  skipped) and without (104 skipped).

## Upgrading

- Nothing changes for a config that names no new option, and a 0.2.3
  checkpoint restarts under 0.2.4 with the same config (the T255 day,
  restarted from 0.2.3's step 144, continued 0.2.3's run byte for byte).
- A dynamic-ocean run restarted from a 0.2.3 checkpoint continues with the
  0.2.4 coupler (the stress and remap fixes), so it is not the run 0.2.3
  would have continued.
