gpuwm-global 0.1.8

Arwen Global 0.1.8 runs the T383 config of record at full speed on a 16 GB
card, writes the wrfout tapes from inside the run, and stops a run waiting
on its own checkpoints. On an RTX 5070 Ti,
`arwen_global_gdas_t383_native_24h` now takes 1.832 s a step where 0.1.7
took 2.535, with every array unchanged. Everything else new is
experimental and off unless named. It needs `gpuwm>=2.9.13,<2.10` as 0.1.7
did, and the Rust doors are 0.1.7's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- **The T383 config of record at full speed on a 16 GB card.**
  `arwen_global_gdas_t383_native_24h`, which runs the default
  semi-Lagrangian core, carries the 5,000-column radiation chunk the T383
  semi-Lagrangian configs moved to in 0.1.7, where it still carried
  12,500. On an empty 16 GB card the door now runs it at one band with
  nothing parked, where it sent it to two latitude bands with every slice
  parked in host memory. Its config hash moves; see Upgrading.
- **Tapes written by the run.** `gpuwm-global run --tapes DIR` writes the
  wrfout tape of every checkpoint the run writes into `DIR`, from the
  run's own state, on a worker thread beside the forecast, so a day no
  longer stops for an `export` process to have the card. Every tape is
  byte-identical to `export` of the same checkpoint, and the checkpoints
  are byte-identical to those of a run without the flag. A tape that fails
  is listed in the receipt and the forecast goes on. `--tape-nlat` and
  `--tape-nlon` set the grid; the flags are outside the config identity.
- **Checkpoints that do not hold the run.** `run` and `cycle` write a
  checkpoint's members stored instead of deflated, read a stored
  checkpoint member by member and check its hashes on four threads, and
  copy the state off the card into one page-locked buffer kept for the
  run. A run also prices its memory once, from the card reading the door
  already took, where it probed the card three times. No array, per-array
  hash or archive hash moves. The new `[output]` table chooses:
  `checkpoint_compression = "none"` (the default) or `"deflate"`, and
  `snapshot_buffer = "pinned"` (the default) or `"pageable"`; neither is
  in the config hash.
- **`SST` on every tape.** The skin temperature, `TSK`, under the name a
  reader of WRF files asks for a sea-surface map by: the water temperature
  the model ran with over open water and lakes, the land or ice skin
  elsewhere, which is what WRF's real.exe writes when its input carries no
  SST of its own. Mask with `XLAND` or `SEAICE` for open water.
- **The T255 day priced at what its runs took.** The door prices
  `arwen_global_gdas_t255_native_sl_si_24h` at 9,178 MiB of card where it
  read 11.09 GiB, so a 16 GB card with 9 to 11 GiB free runs it at one
  band instead of in latitude bands, and the pool starts held to the
  day's 6.70 GiB live peak.

Experimental, each off unless named. A run that names none of them is
0.1.7's run, array for array:

- **Two coarse grids**, `arwen_global_gdas_t63_native_sl_si_24h` and
  `arwen_global_gdas_t127_native_sl_si_24h`: the T255 day of record with
  its truncation and grid spacing changed and nothing else. 57 experiments
  now ship. Neither has been scored against an analysis.
- **Single-column cases**, `tools/arwen_global_scm_cases.py`: the native
  physics suite, as a forecast calls it, on DYCOMS-II RF01 and six
  DEPHY-SCM community cases, each run ending on its readings against the
  ranges the cases' LES papers give.
- **Exact twins**, `tools/arwen_global_twin_scan.py`: seeded and one-bit
  twin starts, and an energy-norm reading of their difference.
- **Six experiment tools**: the forcing operator (`run --forcing`),
  forecast members outside the assimilation path, airliner heights from
  ADS-B, the scheme transplant, the composition replay of extra tracers,
  and `radiation_between` for the shortwave between two radiation calls.
  With them, `run --restart PARENT --branch-from-other-config` resumes a
  checkpoint written under another config hash, an exact twin, only when
  asked.
- **Stepping options**: `[semilag] extrapolation = "settls_departure"`, the
  departure-point form of SETTLS, and `[semilag] level_rate_limiter`;
  in-situ cadences in seconds of forecast time (`spectra_every_s`,
  `flush_every_s`, `energy_every_s`); and a stratospheric noise reading
  (`stratospheric_noise_limit`).
- **Two transform-diet switches** in a `[transforms]` table,
  `skip_unread_planes` and `divergence_only_mass_flux`, that stop the
  semi-Lagrangian step making planes it never reads. On the cards
  measured every array is unchanged with them on; on a CPU
  `divergence_only_mass_flux` agrees to rounding (0.1.9's notes); no
  speed is claimed.

`CHANGELOG.md` has each change with its own measurements.

## Measured

On one RTX 5070 Ti 16 GB with nothing else on it, the shipped configs with
no edit from the analyses they name, 2026-11-12 and 13. Wall is the
receipt's; the card figure is the run's own process as nvidia-smi reports
it, sampled every 250 ms. The 0.1.7 rows were run on the same card the same
evening.

| | | plan the door chose | a step | wall | on the card |
|---|---|---|---|---|---|
| T383 `arwen_global_gdas_t383_native_24h`, 3 h | 0.1.7 | two bands, every slice parked (2.27 GiB in host memory) | 2.535 s | 152.2 s | 12,984 MiB |
| | **0.1.8** | **one band, nothing parked** | **1.832 s** | **74.0 s** | **14,072 MiB** |
| T255 `arwen_global_gdas_t255_native_sl_si_24h`, 24 h | 0.1.7 | one band | 0.736 s | 238.7 s | 9,110 MiB |
| | **0.1.8** | **one band** | **0.737 s** | **218.2 s** | **9,150 MiB** |
| | **0.1.8, `--tapes`** | **one band** | **0.741 s** | **219.8 s** | **9,154 MiB** |

- **The T383 day.** A 24 h day of `arwen_global_gdas_t383_native_24h` at
  the new chunk on the same card, run on the chunk change's own tree
  before the stored checkpoints joined it: 1.907 s a step, 610.7 s (10.2
  min), 14,316 MiB, under the 14,608 MiB the door prices the plan at. Every
  array of the step-0 and step-36 checkpoints is byte-identical at the
  two chunks (133 of 133 at step 36); on an RTX 5090, where both chunks
  run at one band, every array of all nine checkpoints of the day is.
- **The tapes.** The `--tapes` day wrote its nine tapes in 1.6 s more
  wall than the day without, and its nine checkpoints are byte-identical
  to that day's, archive hash included. Its tapes carry `SST` equal to
  `TSK` bit for bit. `export` of a day's nine checkpoints as its own
  process took 87.0 s on an RTX 5090.
- **The checkpoints.** On an RTX 5090, three-hour runs with hourly
  checkpoints, three each: a cold start took 58.8 to 68.1 s at T255 where
  0.1.7 took 154.7 to 176.4 s, and 107.1 to 122.4 s at T383 (the
  semi-Lagrangian day) where it took 317.1 to 351.9 s.
- **Same bits.** On RTX 5090s, every array of the step-0 and step-36
  checkpoints of the shipped T255 and T383 semi-Lagrangian days to 3 h is
  byte-identical to 0.1.7's, archive hash included (43 and 133 arrays at
  each truncation). The files are larger because their members are
  stored; see Upgrading.

Every receipt passed.

## Upgrading

- `pip install --upgrade "gpuwm-global[gpu-cu13]"`, then `gpuwm-global
  fetch-doors` and `gpuwm-global doctor` as before. The door bundle is the
  one 0.1.3 published; a bundle already staged stays valid.
- **One config hash moves.** `arwen_global_gdas_t383_native_24h` says
  `radiation_column_chunk = 5000` where it said 12500, and the chunk is in
  the hash (`6f32b3255085ce4e` where it was `d00cc42c95d1e452`). A
  checkpoint it wrote under 0.1.7 is refused on restart under the new
  file ("checkpoint config identity does not match this run"); continue
  it under a copy that says `radiation_column_chunk = 12500`. Measured on
  an RTX 5090: under such a copy 0.1.8 continued 0.1.7's step 36 to step
  72 byte-identical to 0.1.7's straight day, archive hash included. Every
  other shipped config keeps its 0.1.7 hash; the two coarse grids are new.
- **Checkpoints go both ways.** 0.1.8 continues a checkpoint 0.1.7 wrote,
  and 0.1.7 reads and continues one 0.1.8 wrote stored. Measured on an
  RTX 5090: a restart from step 12 ends at a step-36 checkpoint
  byte-identical to the uninterrupted 0.1.7 run's whichever release wrote
  step 12 and whichever continued it, and 0.1.7's `export` reads a
  stored checkpoint.
- **Stored checkpoints are 2.3 times larger.** 1,452.9 MB at T255 where a
  deflated one is 626.6 to 639.8 MB, and 3,268.9 MB at T383 where it is
  1,404.0 to 1,437.3 MB. A run that keeps its checkpoints sets `[output]
  checkpoint_compression = "deflate"`. The pinned buffer is one
  checkpoint's size of page-locked host memory for the life of the run;
  `snapshot_buffer = "pageable"` gives it back. Assimilation, migration
  and the in-situ ledger write their checkpoints deflated, as before.
- **The T255 door.** With 9 to 11 GiB free, the T255 day now runs at one
  band where 0.1.7 chose latitude bands. A free-memory ladder of your own
  written for 0.1.7 sends it to a slower plan than the door would choose.

## Known limits

- **Tapes need the state on one card.** A run parked in the pinned host
  tier or spread over cards prints `tapes off: ...` and writes none. So
  does an idealized run with no date, such as `arwen_global_moist_smoke`:
  `run` takes no `--start-date`, and `export --start-date` writes its
  tapes from the checkpoints.
- **The tapes share the host's processors.** Their regrid and netCDF
  write run on a worker thread beside the forecast. On a quiet host the
  T255 day's nine took 17.8 s of that thread and 1.6 s of wall; on an RTX
  5090 host running other jobs at a load of 20 to 30 they took 154.6 s of
  it, and the day's wall went from 254.4 to 332.2 s, about what `export`
  would have cost.
- **The departure form was scored on one config.** Four 48 h cases of
  `arwen_global_gdas_t255_native_sl_si_24h_dt600`, with the level-rate
  limiter on: better at 48 h in the global means, 850 hPa temperature
  worse at 24 h in all four, and the tropics slightly worse at 48 h. The
  form alone, other steps and other truncations were not scored.
- **The coarse grids and the single-column cases are research tools.**
  Neither coarse grid has been scored against an analysis; the
  single-column readings show where the shipped suite disagrees with the
  cases' LES.
- **Where a held pool ends varies from run to run**, as in 0.1.7. The
  door prices the one-band T383 plan at 14,608 MiB, the most any run of it
  took; the runs of `arwen_global_gdas_t383_native_24h` measured here took
  14,072 to 14,316 MiB.
- **No T383 run longer than a day** was made on this release's tree.

## Unchanged

- The engine range, `gpuwm>=2.9.13,<2.10`, and the 48 pinned seam files.
  `tools/measure_boundary.py` reads 240 symbols across 71 modules with no
  gap.
- The ten Rust doors. `gpuwm-global fetch-doors` stages the bundle 0.1.3
  published.
- Every forecast bit, and every config hash but
  `arwen_global_gdas_t383_native_24h`'s.
