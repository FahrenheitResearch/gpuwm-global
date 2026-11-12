gpuwm-global 0.1.7

Arwen Global 0.1.7 makes a forecast day faster and lighter on the card with
every forecast bit unchanged. On a 16 GB RTX 5070 Ti the shipped T383 day now
runs at full speed with nothing set, in 9.6 minutes where 0.1.6 took 15.3, and
the shipped T255 day takes 9,028 MiB of the card where it took 11,582. It
needs `gpuwm>=2.9.13,<2.10` as 0.1.6 did, and the Rust doors are 0.1.6's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- **Faster radiation.** The two RTE solvers launch over spans of about
  2,240 columns on a 70-SM card, so they take their fast path whatever the
  radiation chunk is, and the shortwave is solved for the sunlit columns
  only. A radiation call on the RTX 5070 Ti takes 0.96 s at T255 where it
  took 2.26, and 2.17 s at T383 where it took 4.30.
- **Faster physics bookkeeping.** The in-situ ledger marks each field in
  one kernel pass, a physics call takes five column-water integrals where
  it took eight and builds the layer thicknesses once, and the closure's
  three numbers cross to the host in one transfer. With this change alone
  an ordinary T255 step on the RTX 5070 Ti takes 631 ms where it took 672
  to 678 (the median of ten profiled steps, two runs each), and sends
  5.9 MB to the host where it sent 13.0.
- **T383 at full speed on a 16 GB card.** A run now holds its CuPy pool to
  the figure of the plan the door chose and raises it only when an
  allocation finds no room, instead of letting the pool grow into the
  whole card. The radiation's cloud finalize writes over the arrays it
  read, and the cumulus scheme packs four tiles of columns a pass. The
  shipped T383 semi-Lagrangian configs moved to the 5,000-column radiation
  chunk. Together the shipped T383 day gets the one-band plan on an empty
  16 GB card with no edit, where 0.1.6's door sent it to two latitude
  bands with every slice parked in host memory.
- **The door prices T383 at what its runs took.** One band with nothing
  parked from 14.27 GiB free, two bands with every slice parked from 10.83,
  four from 10.33. Its line says what the plan it chose costs against one
  band, and what the pool is held to.

## Measured

The shipped configs with no edit, from their GDAS 2026-09-01 00Z analysis,
on one RTX 5070 Ti 16 GB with nothing else on it, this release's tree and
0.1.6's run back to back on the same card on 2026-11-12. Wall is the
receipt's; the card figure is the run's own process as nvidia-smi reports
it, sampled every 250 ms.

| 24 h day | | plan the door chose | a step | wall | on the card | live peak |
|---|---|---|---|---|---|---|
| T383 `arwen_global_gdas_t383_native_sl_si_24h` | 0.1.6 | two bands, every slice parked (2.27 GiB in host memory) | 2.981 s | 918.7 s (15.3 min) | 15,142 MiB | 12.00 GiB |
| | **0.1.7** | **one band, nothing parked** | **1.805 s** | **578.5 s (9.6 min)** | **14,392 MiB** | **11.94 GiB** |
| T255 `arwen_global_gdas_t255_native_sl_si_24h` | 0.1.6 | one band | 0.914 s | 289.9 s (4.8 min) | 11,582 MiB | 8.63 GiB |
| | **0.1.7** | **one band** | **0.747 s** | **242.4 s (4.0 min)** | **9,028 MiB** | **6.70 GiB** |

- **Same bits.** Every array of every checkpoint of the two T255 days is
  byte-identical, archive hash included (9 checkpoints, 1,107 arrays).
  The two T383 days differ in their radiation chunk, so their config hash
  differs and nothing else: 575 of 575 arrays over 5 checkpoints are
  byte-identical. A 3 h T383 run of 0.1.7 is identical to one of 0.1.6 run
  at the 5,000-column chunk, archive hash included (133 of 133 arrays at
  step 36).
- **Restart.** Stopped at step 18 and restarted, both truncations end at
  step 36 identical to the straight run, archive hash included.
- **The pool.** The T383 day's pool started held at 11.50 GiB and was
  raised 7 times to 13.66 GiB; the run took 14,392 MiB of the card, under
  the 14,608 MiB the door prices the plan at. The T255 day's pool was never
  raised. Host memory peaked at 5.3 GiB (T383) and 3.9 GiB (T255).

Every receipt passed. `CHANGELOG.md` has the measurements behind each
change on its own.

## Upgrading

- `pip install --upgrade "gpuwm-global[gpu-cu13]"`, then `gpuwm-global
  fetch-doors` and `gpuwm-global doctor` as before. The door bundle is the
  one 0.1.3 published; a bundle already staged stays valid.
- **A T383 run in flight from 0.1.6.** The two shipped T383
  semi-Lagrangian configs (`arwen_global_gdas_t383_native_sl_si_24h` and
  `arwen_global_gdas_t383_native_sl_si_wall`) now say
  `radiation_column_chunk = 5000` where they said 12500. The chunk is part
  of the config hash, so a checkpoint written by 0.1.6 under either config
  continues only under a config that says `radiation_column_chunk = 12500`:
  keep 0.1.6's copy of the file (or that one line) for that run. Measured:
  0.1.6's step-72 checkpoint of the shipped T383 day is refused under the
  new config ("checkpoint config identity does not match this run"); under
  a copy that says 12500, 0.1.7 continued it to step 144 at two latitude
  bands with every slice parked, the plan the door chooses for 12,500 on
  an empty 16 GB card (2.44 s a step), and every array at step 144 is
  byte-identical to 0.1.7's own day. A run started under 0.1.7 uses the
  new configs as they are.
- **Everything else continues.** Every other config hash is 0.1.6's.
  0.1.6's step-36 checkpoint of the shipped T255 day, continued by 0.1.7,
  ends at step 72 identical to 0.1.7's straight run, archive hash
  included.
- **If you set `CUPY_GPU_MEMORY_LIMIT`,** your limit stands and the run sets
  none of its own, so the pool is not held to the plan's figure. Unset it
  to get the held pool. Runs under `--device-allocator async` or `slab` are
  not held either.
- **If you choose the plan yourself** (`--latitude-bands`, `--host-spill`),
  the run takes it. Left alone, the door now takes one band for the
  shipped T383 day from 14.27 GiB free and two bands from 10.83, so a
  free-memory ladder of your own written for 0.1.6 sends some runs to a
  slower plan than the door would choose.

## Known limits

- **Where a held pool ends varies from run to run.** It is set by how the
  pool's blocks were cut, not by the forecast. Over the four one-band T383
  runs made for these notes the run took 13,706 to 14,518 MiB of the card,
  and the runs of the change's own measurements 13,642 to 14,608. The door
  prices the plan at 14,608 MiB, the most any run of it took. A run that
  needs more on a card without the room ends with the card's out-of-memory
  error, not the door's refusal.
- **No T383 run longer than a day** was made on this release's tree. The
  ten-day T383 row on the README is from 2026-11-06.
- **The four- and eight-band T383 prices** are from runs of 2026-11-11,
  before the pool was held. The one- and two-band prices are this
  release's.
- **The other T383 configs keep 12,500.** `arwen_global_gdas_t383_native_24h`
  runs the semi-Lagrangian core by default and, as in 0.1.6, the door sends
  it to two bands with every slice parked on an empty 16 GB card; a copy
  that says `radiation_column_chunk = 5000` gets the one-band plan (and a
  config hash of its own). The Eulerian T383 configs keep 12,500 too; the
  chunk was measured on the semi-Lagrangian core only.
- **The door still prices the T255 day at 11.09 GiB of card**, as 0.1.6
  did, although the day now takes 9,028 MiB. With less than that free it
  chooses latitude bands for a run that would fit at one.
- **The card reading at start and end.** The door reads the card from a
  short-lived process of its own, and the receipt does again at the end:
  each holds about 222 MiB of the card for under a second.

## Unchanged

- The engine range, `gpuwm>=2.9.13,<2.10`, and the 48 pinned seam files.
  `tools/measure_boundary.py` reads 240 symbols across 71 modules with no
  gap.
- The ten Rust doors. `gpuwm-global fetch-doors` stages the bundle 0.1.3
  published.
- Every forecast bit, and every config hash but the two T383
  semi-Lagrangian configs'.
