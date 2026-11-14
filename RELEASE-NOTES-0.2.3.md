gpuwm-global 0.2.3

Arwen Global 0.2.3 changes two defaults and adds options. The T255 and T383
forecast days now run the kim-gwdo orographic gravity-wave drag, which
improved northern-hemisphere forecasts on 14 to 16 of 16 scorecard starts.
The T383 day calls its radiation less often, which made it faster at no
measured cost in skill. Everything else is off unless a config turns it on:
switches for long and coupled runs, a mixed-layer response for the
prescribed sea, a snow-age fix, an opt-in "day-one" config, and two
experimental pieces (a dynamic ocean and storm switches for New Tiedtke).
Every other shipped config keeps its 0.2.2 hash. It needs
`gpuwm>=2.9.13,<2.10`, as 0.2.2 did, and the Rust doors are 0.2.2's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Changed by default

- **Orographic drag on in the forecast days.**
  `arwen_global_gdas_t255_native_sl_si_24h` and
  `arwen_global_gdas_t383_native_sl_si_24h` now carry
  `orographic_drag = "kim-gwdo"` (WRF's `gwd_opt = 1`). On the forecast
  scorecard's 16 starts (2026-09-14 to 11-08, 120 h forecasts at T255,
  verified against the GFS analysis), against the same day without it:
  - NH 500 hPa height error at 120 h: 52.0 m against 54.3 m, better on 14
    of 16 starts;
  - NH sea-level pressure error at 120 h: 5.11 against 5.33 hPa, better on
    16 of 16;
  - NH 850 hPa temperature 0.08 K and 250 hPa wind 0.40 m/s better at 120 h;
  - about 3.6 hours of NH 500 hPa forecast lead at 96-120 h;
  - the southern hemisphere is neutral; its sea-level pressure is 0.08 hPa
    worse at 120 h.

  It was measured at T255 only. Its skill at T383, and over a season, has
  not been measured. Set `orographic_drag = "none"` in a copy of the T255
  day to get 0.2.2's forecast back: on an RTX 4090 that copy wrote 0.2.2's
  nine checkpoints byte for byte. A copy of the T383 day needs
  `radiation_interval_s = 2100.0` as well to get 0.2.2's config hash back
  (its checkpoints were not compared with 0.2.2's).
- **The drag's statics are built for you.** The drag reads ten sub-grid
  terrain fields from WPS_GEOG's `orogwd_10m`, cached once per resolution
  beside the surface statics. A machine set up under 0.2.2 does not have
  that cache, and many WPS_GEOG folders do not have `orogwd_10m` (neither
  of the two machines this release was checked on had it). So:
  - `gpuwm-global statics CONFIG` builds the drag's cache after the surface
    fields;
  - `gpuwm-global run` (and `go`) builds it before the forecast starts when
    it is missing, from the WPS_GEOG folder the surface fields were built
    from;
  - when that folder has no `orogwd_10m`, the run downloads it into the
    folder with the engine's own `gpuwm fetch-geog --datasets orogwd_10m
    --source ncar` (3.3 MB, checked against the engine's pinned SHA-256).

  On both test machines the first run downloaded it and built the cache in
  2.5 s (T255) and 3.9 s (T383) before the forecast started. If it cannot
  download or build it (no network, a folder it cannot write), the run
  stops before its first step with one sentence that names the two commands
  to run by hand, and the option that turns the drag off. A runner that
  calls `statics` once and then `run` needs no change.
- **The T383 day calls radiation every 3000 s** instead of 2100 s. Over
  eight 120 h T383 forecasts (2026-09-29 to 11-08) no score was worse by more
  than three times the round-off spread between two identical runs. It
  stepped 1.067x faster on an RTX 5090, and 1.17x faster in one 24 h run on
  an RTX 5070 Ti. On the 16 GB RTX 5070 Ti the 0.2.3 T383 day (drag on,
  radiation every 3000 s) ran at 1.72 s a step, 1.2 % slower than without
  the drag, with a card peak of 14,434 MiB of 16,303. The other T383
  configs keep 2100 s.

## New, off unless you turn it on

- **The day-one config,** `arwen_global_gdas_t255_native_sl_si_24h_dayone`:
  the T255 day (with its drag) and New Tiedtke convection in place of
  Grell-Freitas. On all 16 scorecard starts against 0.2.2 it takes NH
  500 hPa height error from 10.30 to 8.60 m at 24 h (16 of 16 starts) and
  from 54.35 to 50.46 m at 120 h (15 of 16), worth about 7.3 hours of NH
  lead at 120 h, read off the control's 500 hPa ACC curve on the 14 starts
  verified to 240 h (the drag alone, read the same way, 3.7 hours; the 3.6
  hours above is the 16-start figure at 96-120 h). Tropical temperature
  and wind improve too. The cost is tropical 500 hPa height: an offset of -8.8 m at 120 h,
  which New Tiedtke causes and which is not traced yet. That is why it is
  not the default.
- **Conservation switches for long runs:** `[repair] mass_fixer_target =
  "dry"`, `energy_fixer = true`, `vapour_fixer = true`. Over 60 T127 days
  they cut the energy budget's residual (gate LR-4) from 4.1-4.5 to
  0.2-0.4 W/m2 and the dry-mass drift from 35 Pa to 0.26 Pa. Two five-day
  T255 forecasts scored the same with them. They stay off for forecasts.
- **The long-run recipe** (`docs/ARWEN_GLOBAL_LONG_RUNS.md`) now turns on
  those three switches and 0.2.2's three Grell-Freitas shallow-cloud
  switches. Over 90 days the shallow switches cut the excess sunlight the
  model reflects, against CERES, from about 15 W/m2 to about 4.5 W/m2. In a
  coupled run they cut the ocean's cold drift by about two thirds by day 59.
  They stay off for forecasts because they cost a little tropical skill at
  3 to 5 days. The two sets were measured separately; a long run with both
  has not been run yet.
- **A mixed-layer response for the prescribed sea** (`[ocean]
  mixed_layer_response = true`). A held sea surface cannot cool under days
  of dry offshore wind, so evaporation off Tampico, in the Red Sea and in
  the Gulf of Tehuantepec reached 51-55 kg/m2 a day in 50-day T383 runs,
  near the 60 kg/m2 limit that stops a run. The response lets the sea
  surface cool by up to 5 K: it took 4-16 % off those days (for example
  55.05 to 52.83 and 49.74 to 42.61 kg/m2). It lowers the risk; it does not
  remove it.
- **Snow age by snowfall amount** (`snow_age_refresh = "amount"`, with
  `snow_ageing = true`). WRF's rule makes any snowpack fresh whenever a
  trace of snow falls, which on this model is most of the time, so snow
  stays too bright. The new rule removed about 30 % of the snow's albedo
  excess against ERA5 in a late-winter test, and warmed NH land by 0.6 and
  1.7 K in February in two 60-day tests (two tests, not an ensemble).

## Experimental

- **A dynamic ocean** (`[ocean] mode = "dynamic"`, and the new
  `gpuwm-global ocean` command that builds its grid and starting state). It
  runs, restarts bit for bit and fit the 16 GB RTX 5070 Ti at T383 (14,990
  of 16,303 MiB) at 14 % more time a step, measured on the T383 day as
  0.2.1 shipped it (radiation every 2100 s, no drag), not on 0.2.3's. It is
  not ready: the equatorial
  undercurrent collapses within 30 days, two idealised wave tests and the
  wind-driven mixing test fail their gates, and the effect of its grid
  remap on the sea surface the atmosphere sees is not measured.
  `docs/ARWEN_GLOBAL_DYNAMIC_OCEAN.md` lists what is open.
- **Storm switches for New Tiedtke** (five `ntiedtke_*` options). New
  Tiedtke keeps hurricanes too weak because it barely dilutes the rising air
  in a moist storm core. Stronger dilution deepens Katrina (977 to 952 hPa
  against ERA5's 933) but puts unrealistic rain cells on coasts and pulls
  Sandy's track off. No setting is recommended.

## Fixed

- The memory sizer and the scorecards' terrain cache no longer mistake a
  grid of another shape at the same truncation for the shipped one. No
  shipped config is affected.

## Measured for this release

- **Unchanged where nothing was turned on.** With the drag set to `"none"`,
  a 24 h T255 day under 0.2.3 wrote 0.2.2's nine checkpoints byte for byte
  (RTX 4090). Every other shipped config hashes as in 0.2.2.
- **The drag's first run, from nothing.** From a WPS_GEOG folder without
  `orogwd_10m`, the T255 and T383 days downloaded it, built their caches
  and ran; a restart of the T255 day reproduced its straight run bit for
  bit.
- **T383 on the 16 GB card.** With the drag 1.72 s a step and a card peak of
  14,368 to 14,434 MiB; without it 1.70 s and 14,246 to 14,826 MiB (two
  runs each, RTX 5070 Ti, 16,303 MiB; the step times are from runs with
  the kernels already compiled, and the first run, which compiled them,
  took 1.90 s a step).
- **Tests.** 3506 CPU tests pass from the installed wheel, with CuPy (86
  skipped) and without (100 skipped).

## Upgrading

- A T255 or T383 forecast day under 0.2.3 is a different forecast from
  0.2.2's (the drag; at T383 also the radiation interval), and its config
  hash differs, so a 0.2.2 checkpoint of those days does not restart under
  the 0.2.3 config. Restart it under a copy with `orographic_drag = "none"`
  (T255; at T383 also `radiation_interval_s = 2100.0`), or finish it on
  0.2.2.
- The first run of a forecast day on a machine without the drag's cache
  builds it, and may download `orogwd_10m` (3.3 MB) into your WPS_GEOG
  folder.
- Nothing else changes for a config that names no new option.
