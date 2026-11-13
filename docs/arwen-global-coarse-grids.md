# Arwen Global: the T63 and T127 coarse grids (experimental)

Two shipped experiments run the forecast day of record at a quarter and at
half of its resolution:

| Experiment | Grid (lat x lon) | Equatorial spacing |
|---|---|---|
| `arwen_global_gdas_t63_native_sl_si_24h` | 96 x 192 | 208.5 km |
| `arwen_global_gdas_t127_native_sl_si_24h` | 192 x 384 | 104.2 km |
| `arwen_global_gdas_t255_native_sl_si_24h` (the day of record, for comparison) | 384 x 768 | 52.1 km |

Each is `arwen_global_gdas_t255_native_sl_si_24h` with two things changed:
the truncation, and `dx_m`, the grid length the surface layer and the
Grell-Freitas closure read, set to the coarse grid's equatorial spacing.
The step (300 s), the semi-Lagrangian core and its drain, the full native
physics suite with every option, the 40 levels, the gates and the case
(the GDAS 2026-09-01 00Z analysis, at the same relative path) are the T255
config's.  `tests/test_arwen_global_coarse_grids.py` holds them to that:
any other difference from the T255 config fails it by key.

They are EXPERIMENTAL.  Nobody has scored them against an analysis, and
they are not forecast configurations.

## Running one

The same three steps as the T255 day in the README, with the coarse
experiment's name.  The analysis is the one the T255 day reads:

```bash
gpuwm fetch --source gdas --cycle 2026-09-01T00 --hours 0 \
  --mode full-file --out cases/baseline-2026090100
gpuwm-global statics arwen_global_gdas_t63_native_sl_si_24h --geog-root ~/WPS_GEOG
gpuwm-global run arwen_global_gdas_t63_native_sl_si_24h --outdir out/t63-day
```

The statics are built once per truncation, on the CPU.  Building them took
121 s and 17.8 GB of host memory at T63, and 134 s and 18.6 GB at T127
(one build at a time, measured 2026-11-12).

## What a day costs, measured

The shipped experiment by its bare name, no flag, 24 h from the GDAS
2026-09-01 00Z analysis, on one RTX 5090 (32 GB) with no other job on the
card, 2026-11-12.  Wall clock and the step time are from the run's own
receipt; "on the card" is the run's process in nvidia-smi every 250 ms;
"live" is the receipt's peak of the CuPy pool's live bytes.  Every gate of
both receipts passed.

| Experiment | A forecast day | A step | Live peak | On the card |
|---|---|---|---|---|
| T63 | 108.0 s | 0.363 s | 2.50 GiB | 3,508 MiB |
| T127 | 171.8 s | 0.514 s | 3.32 GiB | 5,180 MiB |

At these grids the card is far from full, so the step time is not
proportional to the number of grid points: T127 has four times T63's
points and took 1.4 times its step.

## What they are for

- **Checking a change of the model end to end, cheaply.**  Every scheme
  the T255 day runs, runs here, through the same doors, gates and
  checkpoints, in two or three minutes of a card; a change that breaks a
  gate, a restart or a receipt usually shows here before it costs a T255
  day.
- **Small cards.**  A T63 day needs under 4 GB of card and a T127 day
  under 6 GB.
- **Many runs.**  Twin, ensemble-sized and sensitivity experiments where
  the count of runs matters more than the resolution (the twin-scan tool,
  `docs/arwen-global-twin-scan.md`, was proven on the T63 day).

What they are not for: forecasts.  The physics options are the T255
config's, chosen and checked at 52 km, and nobody has checked them at 104
or 208 km; a coarse day has not been scored against an analysis; and one
day of each is all that has been run on these configurations.  A coarse
day's weather is the model's at that grid, not a cheaper copy of the T255
forecast.
