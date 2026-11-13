# Arwen Global composition replay: extra transported fields over a finished run (experimental)

Module `src/arwen_global/composition.py`; drivers
`tools/arwen_global_replay_tracers.py`, `tools/arwen_global_ozone_field.py`,
`tools/arwen_global_ozone_change_score.py` and
`tools/arwen_global_tracer_plume_chart.py`; tests
`tests/test_arwen_global_composition.py`.

This is an experiment tool. A forecast's checkpoints hold its wind and its
surface pressure at every time it kept. The replay reads those and nothing
else, and carries a table of extra fields (tags, an origin map, a point
release, a field read from a file such as an ozone analysis) through the
same semi-Lagrangian machinery the model's own grid tracers ride: the
departure-point search, the quasi-monotone cubic gather and the
Bermejo-Conde mass fixer. The physics is not run again and the weather is
not integrated again. A field nobody thought of before the forecast was
issued does not cost the forecast again, and it cannot move a byte of the
weather.

## What it is and is not

It is an offline transport. The model advects its own species with the wind
of every step; a replay has the wind only at the times the run kept a
checkpoint, and between two of them it uses the straight line in time
between the two winds (and the two layer thicknesses). That is the
stored-field approach of offline chemistry-transport models, and its error
grows with the checkpoint interval. The receipt states the interval every
sidecar was made from, and an interval that is not a whole number of
substeps is refused.

The rows have no boundary-layer mixing, no convective transport and no wet
or dry removal: a replay row is carried by the resolved wind alone. That is
a fair first answer above the boundary layer and outside deep convection
(stratospheric ozone, a volcanic plume at cruising height, an origin map in
the extratropical free troposphere) and wrong inside them. Every receipt
says so under `processes`.

Nothing in the module is imported by the forecast loop and nothing in it
writes a weather checkpoint. A run made with `gpuwm-global run` is the
0.1.7 run.

## Rows

| Kind | What it carries |
|---|---|
| `constant` | a uniform mixing ratio (`value`, default 1); with the fixer on it measures the air-mass drift of the replayed flow |
| `band` | one inside a latitude and pressure box at the first checkpoint, zero outside: where air came from |
| `origin` | three rows `<name>_x`, `_y`, `_z`, the unit-sphere position of each point at the first checkpoint, stored as `(1 + c) / 2` |
| `release` | zero at the start; `mass_kg` emitted over the air of a cylinder (`lon_deg`, `lat_deg`, `radius_km`, between `p_top_pa` and `p_bottom_pa`) from `start_s` for `duration_s`; a float64 ledger holds the emitted mass beside the carried mass |
| `field` | a `(nlev, nlat, nlon)` array from an `.npz` on the model grid (`path`, `array`, `scale`), floored at zero |

A spec is a TOML file with one `[composition]` table:

```toml
[composition]
schema = "gpuwm.arwen-global-composition/v1"
substep_s = 300.0
fixer = "bermejo_conde"

[[composition.tracer]]
name = "plume"
kind = "release"
lon_deg = 15.0
lat_deg = 37.75
radius_km = 60.0
p_bottom_pa = 40000.0
p_top_pa = 20000.0
start_s = 0.0
duration_s = 21600.0
mass_kg = 1.0e9
```

## The route

```bash
# carry the spec's rows through a finished run's checkpoints
python tools/arwen_global_replay_tracers.py --config RUN.toml --spec SPEC.toml \
  --out replay --sha256-before-after run/arwen_global_step*.npz

# ozone from a GDAS 0.25 degree analysis on the model's grid and levels, for a field row
python tools/arwen_global_ozone_field.py --grib gdas.t18z.pgrb2.0p25.f000 \
  --config RUN.toml --checkpoint run/arwen_global_step00000000.npz --out ozone-t0.npz

# the carried column's change against the analysis's own change
python tools/arwen_global_ozone_change_score.py --config RUN.toml \
  --start ozone-t0.npz --verify ozone-t24.npz \
  --sidecar replay/arwen_global_step00000288.composition.npz \
  --checkpoint run/arwen_global_step00000288.npz

# column-burden maps of one row
python tools/arwen_global_tracer_plume_chart.py --config RUN.toml --replay replay \
  --weather run --row plume --steps 72,288,576 --out plume.png
```

The replay writes one sidecar per weather checkpoint
(`<checkpoint>.composition.npz`) and `arwen-global-composition-receipt.json`.
A sidecar is keyed by the weather checkpoint's `self_sha256` and the spec's
own hash, so it belongs to exactly one weather and one spec; an edited
sidecar is refused. `--resume SIDECAR` continues a replay from one of its
sidecars and writes the sidecars an unbroken replay writes. A config whose
hash is not the run's but carries the same grid and vertical coordinate is
refused unless `--accept-geometry-config` is given, and the receipt then
records both hashes. `--sha256-before-after` hashes every weather checkpoint
file before and after the replay and writes both lists into the receipt.

The ozone driver decodes the analysis's ozone mixing ratio and its
total-ozone record through the package's mapped engine door, regrids every
level onto the run's grid with the cold start's regridder and interpolates
in ln p onto the model's levels. It also reports the model-level column
against the analysis's total ozone, in Dobson units, globally, poleward of
20 degrees and in the tropics. The score compares the CHANGE of the column,
forecast minus start against analysis minus start, so a fixed difference
between the two columns cancels and persistence scores a correlation of
zero.

## What is tested

`tests/test_arwen_global_composition.py`, on the CPU:

- a replay leaves every weather checkpoint byte-identical;
- the Bermejo-Conde fixer holds every row's mass and books a release;
- a constant row stays constant with the fixer off;
- two segments (a replay resumed from its own sidecar) write the sidecars
  one segment writes;
- a field row at rest comes back unchanged;
- the spec file round-trips and keys the result;
- the refusals, and an edited sidecar refused;
- the ozone driver interpolates linearly in ln p.

## Measured on a card

Measured 2026-11-12 on one RTX 4090. The weather was a five-day T255 L40
forecast of the native suite on the semi-Lagrangian core at 300 s from the
GDAS 2026-11-10 18Z analysis, kept every six hours (21 checkpoints, steps 0
to 1440). The spec carried sixteen rows at 300 s substeps with the
Bermejo-Conde fixer: the constant, the origin map (three rows), seven
latitude and pressure bands, four point releases of 1e9 kg each (six or
twelve hours of emission between 30 and 400 hPa, at four volcanoes'
coordinates) and the GDAS ozone of the start as a field row.

| Measurement | Result |
|---|---|
| replay of the twenty six-hour intervals, sixteen rows | 448 s in the receipt, 7 min 30 s for the whole process |
| the 21 weather checkpoint files, SHA-256 before and after (`--sha256-before-after`) | all 21 equal |
| largest per-substep displacement | 0.64 of a grid cell |
| relative mass drift over five days, the constant row | 1.2e-6 |
| relative mass drift over five days, largest of the sixteen rows | 2.3e-5 (the southern stratospheric band) |
| each release's carried mass against its emitted ledger at five days | within 2.5e-6 |

The ozone driver against the analysis's own total ozone, on the start
(2026-11-10 18Z), model-level column against `TOZNE`:

| region | model column (DU) | analysis (DU) | relative bias | RMS relative difference |
|---|---:|---:|---:|---:|
| global | 286.1 | 285.1 | +0.38% | 0.91% |
| poleward of 20 degrees | 298.5 | 297.1 | +0.46% | 1.03% |
| tropics | 262.6 | 262.0 | +0.20% | 0.63% |

The carried ozone column's change against the analysis's change
(`tools/arwen_global_ozone_change_score.py`), the analysis at the later
time put on the grid by the same driver:

| lead | pattern correlation, global / poleward of 20 / tropics | RMSE over the RMS of the analysis change, global / tropics |
|---|---|---|
| 24 h (2026-11-11 18Z) | 0.77 / 0.78 / 0.36 | 0.75 / 1.42 |
| 30 h (2026-11-12 00Z) | 0.74 / 0.75 / 0.30 | 0.78 / 1.40 |

Outside the tropics the resolved wind alone carries about three quarters
of the column's change pattern over a day. In the tropics, where the
change is small (an RMS of 4.3 DU at 24 h) and convection and chemistry
matter, the replay's change is further from the analysis's than no change
at all would be. This is one case of one start.

## Limits

- Offline: the wind between two kept checkpoints is a straight line in
  time, so a run kept every six hours gives a six-hourly wind.
- No boundary-layer mixing, convection, chemistry or removal.
- Cupy configs replay on the card; the replay was measured at T255 only.
- The score tool builds the run's grid from the config, so a native-suite
  config needs a card to load; a copy of the config without its
  `[physics]` table gives the same grid on the CPU.
