# gpuwm-global 0.1.0

Arwen Global as a distribution of its own: a hydrostatic global spectral
weather model with ensemble data assimilation, one console script, and the
physics it was graded with inside the package.

```bash
pip install "gpuwm-global[gpu-cu13]"
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- One console script, `gpuwm-global`, with 48 commands. Forecast, statics,
  assimilation and its five ensemble legs, render tape export, the regional
  parent bridge, the radiance operators, and every inspection and validation
  leg.
- The physics this model was graded with travels inside the package, as
  `arwen_global.core`: the radiation, cumulus, surface-layer, boundary-layer,
  land-surface and microphysics schemes, the land-use rulebook, twelve kernels
  and three headers, the CUDA loader that binds them, the float64 mirror the
  scorecards grade against, and the Noah and land-use tables. The schemes this
  model executes and that differ from a published 2.7.0 engine are carried;
  `noah` and `morrison` match that engine apart from the carve's import
  rewrites and travel anyway, because their kernels differ and the CUDA
  loader binds its own directory. A bare install on any 2.7.x engine
  integrates the same bytes the grading tree did.
- `arwen_global/data/engine-seam.json` pins the engine files the carried
  physics reaches and does not carry, by path, size and SHA-256.
  `gpuwm-global doctor` re-hashes them: 46 of 46 proven against each of
  `gpuwm 2.7.0`, `2.7.1` and `2.7.2`, on the Windows desktop 2026-09-10, in
  three virtual environments created from scratch. A file whose bytes moved is
  reported by name as unproven, and the `gpuwm<2.8` ceiling is the refusal.
  Every run receipt carries the same verdict beside the module digests.
- The six source mappings the shipped experiments name ship inside the
  package. One resolver answers every bare id and every mapping file name,
  asking the engine's authority table first and the carried copies second, so
  the engine's row wins the day it publishes one. `gpuwm-global doctor` prints
  which table answered each row; a mapping both tables carry with different
  bytes is refused by name with both digests rather than chosen between.
- The decode scratch is translated rather than refused. This model names the
  directory where a decode stages several GB of float64 frames; a published
  engine places the same directory by `GPUWM_COMPOSE_SCRATCH`. The read, the
  placement and the restore are one critical section held across the decode,
  so two decodes take turns, and each receipt names which of the two placed
  the directory it used.
- A machine seam for programs. `gpuwm-global run-plan` runs a plan envelope and
  answers `--catalog`, `--sources`, `--physics-profiles`, `--probe`,
  `--resolve` and `--estimate` as JSON; `gpuwm-global sources --json` is the
  source registry on its own command; `python -P -m arwen_global.tui_worker`
  spawns any command through the engine's worker handshake. A run writes
  `run-manifest.json`, `run-progress.json` and a monotonic `events.jsonl`, and
  refuses a directory another run already owns. The schema ids are the
  engine's, and a document that cannot answer an engine field names the field
  and the reason. See `docs/ARWEN_GLOBAL_CLIENT.md`.
- The spectral core ships inside the package. No published engine wheel carried
  it, so the model could not be installed before this cut.
- Semi-Lagrangian semi-implicit core at a 300 s step, the default at every
  truncation. The Eulerian core is selectable by name at the step its own
  refusal admits with margin.
- A selectable spectral eddy viscosity as the truncation drain, derived from
  closure theory rather than tuned: every level reads its own kinetic energy
  at the cutoff each step and drains at the eddy viscosity that energy
  implies. The exponential hyperdiffusion stays the default, the closure's
  fields join a configuration's identity only when it is selected, so no
  record hash moves, and `arwen_global_gdas_t255_native_closure_24h` is the
  record T255 experiment with its `[diffusion]` table changed and nothing
  else. It has no card run of its own: it ships selectable and ungraded.
- Native CUDA physics: RRTMGP radiation, Morrison microphysics, Grell-Freitas
  and New Tiedtke convection, YSU boundary layer, Noah land, MM5 surface layer,
  run a latitude band at a time.
- The card is priced before anything is allocated. The door chooses the
  latitude band count and the host tier, and refuses a plan that will not fit,
  naming which allocation dies first.
- `gpuwm-global fetch-analysis` fetches the one whole-globe GDAS object a cold
  start needs, with the two non-default flags bound, and prints the run
  command it feeds.
- `gpuwm-global obs subscribe` opens the WMO information system feed and
  archives what arrives with its digest and per centre coverage.
- 32-member T127 LETKF under the T255 control, hourly windows, eight Rust
  observation doors, the GOES ABI and ATMS radiance operators among them. It
  ships selectable, not default.
- The LETKF analysis runs on the card by default; the numpy reference path
  stays selectable for comparison.
- 55 configured experiments in the package, from a four-step numpy smoke to the
  25 km forecast day.
- A command reference generated from the parser, and a boundary measurement
  tool that re-measures the engine dependency at every bump and names the
  symbols it could not check rather than counting them clean.

## Fixed

- A native forecast against a published engine was accepted, priced the card,
  allocated it, integrated its first steps and then refused its own argument
  vector inside the physics: the published surface layer takes no `vegfra`,
  the published radiation no size bounding, neither published cumulus
  constructor a column chunk. The carve removes the cause, and every refusal,
  doctor row and skip that cited one of them is retired with it. Against a
  published 2.7.0 there is now no call this package makes that the installed
  engine refuses.
- Selecting the fixed free-atmosphere mixing length was refused on every
  published engine, because the mode is one more argument to the YSU kernel.
  The kernel travels with the package, so the option runs.
- The radiation and boundary-layer scorecards graded against whichever float64
  mirror the installed engine happened to carry, which is not the mirror of
  the kernels this package runs. They grade against the carried mirror.
- Every shipped GDAS experiment refused at its first door, because no
  published engine carries any of the six source mappings the model names.
- Four imports reached beyond the package's own top level and made the
  installed wheel unimportable.
- One observation at the exact pole refused the whole analysis. It is rejected
  by name now, and nothing else is.
- A bare render call named products the renderer's catalogue does not carry and
  drew nothing while reporting success.
- The forecast command wrote no `status.json`, so a workspace driving a
  day-long integration had a growing log to scrape and no answer to what
  stage it was in.
- A run receipt recorded the configuration, the arithmetic pins and the
  machine, and nothing about the libraries the numbers rode on. It records the
  interpreter and every installed distribution under its self-hash.
- A plan naming a render product the renderer does not carry resolved clean,
  built the statics, integrated the whole forecast and died at the render
  stage naming neither the slug nor anything to act on. Every token is checked
  against the renderer's own product list at review, and the render stage's
  own words reach the machine channel.
- A plan with no start date resolved with a render stage it would silently
  skip, and finished green with an empty directory.
- Every resolved plan document stated a model top of 0.0 Pa, reading the
  surface end of the half-level ladder instead of its top.
- The checkpoint count ignored a restart: two runs that wrote three and one
  checkpoint both promised five, in the resolved document, in the estimate and
  in the number a client sizes its progress bar with.
- `run-progress.json` named the second-to-last checkpoint at `status:
  complete` on every run, because the final one lands when the writer is
  joined.
- The physics-profile menu offered three experiments no plan route can
  execute. Every shipped experiment now carries the door it loads through.
- A shipped experiment's bare name resolved on `run` and `go` but not on
  `export`, `assimilate`, `da init`, `da analyze`, `da static-covariance`,
  `da localisation`, `abi-score` or `abi-reference --config`. The quickstart's
  no-card rehearsal is three lines and the second is an `export`, so a reader
  with no card met a missing-file error on the first thing they could run.

## What it runs

24 hour forecasts, 288 steps at 300 s, the full native physics suite, 40
levels, float32, end to end from each run's own receipt.

| Truncation | Spacing | Card | A forecast day | Measured |
|---|---|---|---|---|
| T255 | 52.1 km | RTX 5070 Ti, 16 GB | 4.28 min | 2026-09-06 |
| T383 | 34.7 km | RTX 5090, shared | 5.63 min | 2026-09-07 |
| T533 | 25 km | RTX 5090, 32 GB | 15.9 min | 2026-09-07 |

T799 at 17 km is refused at the door, with the figure that refuses it.

Banding a run does not move a bit: the ten-step T255 gate is byte-identical
between one latitude band and eight, 309 of 309 checkpoint arrays, on both
cards; the host tier is byte-identical too, 442 of 442 (measured 2026-09-07).

Without a card the package installs and every CPU door runs, including the T3
numpy smoke that goes through the same run, export and render sequence and
draws a real global frame. `CHANGELOG.md` carries the whole table of what a
bare install runs, what needs a card, and what stays on the engine's side.

## Requires

Python 3.11 or later, `gpuwm>=2.7.0,<2.8` with `gpuwm-data` beside it, and a
CUDA card for a real forecast. That range resolves `gpuwm 2.7.2` from the
public index today; the physics the model integrates is inside this package,
so any `2.7.x` engine runs the same bytes. `gpu-cu13` and `gpu-cu12` are the two CUDA
majors. The Rust binaries the model calls arrive as release assets, verified
against the pins inside the wheel before they are used, never from a local
build.

Apache-2.0.
