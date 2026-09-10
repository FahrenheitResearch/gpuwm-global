# Changelog

## 0.1.0

The first cut of Arwen Global as a distribution of its own. Everything below
was previously reachable only from inside the engine's own checkout.

### New

- `gpuwm-global`, one console script with 48 commands: the forecast door, the
  statics builder, the assimilation door and its five ensemble legs, the render
  tape export, the regional parent bridge, the radiance operators and every
  inspection and validation leg. The engine's `gpuwm global` reached eleven of
  them; the rest were reachable only as a module invocation.
- The machine seam, so a program drives this model the way it drives the
  engine. `gpuwm-global run-plan` reads a `gpuwm.run-plan.v1` plan envelope and
  answers `--catalog`, `--sources`, `--physics-profiles`, `--probe`,
  `--resolve` and `--estimate` as one JSON document each; `gpuwm-global
  sources --json` publishes the same source registry on a door of its own. A
  run writes `run-manifest.json` before it allocates anything,
  `run-progress.json` while it works, and an appended `events.jsonl` whose
  sequence is monotonic; a directory a previous run owns is refused by name
  rather than having a second stream interleaved into its file.
  `python -P -m arwen_global.tui_worker --job-dir DIR -- <command>` spawns any
  command through the same handshake the engine's worker uses. Every schema id
  is the engine's, so a client written for one parses the other, and a
  document that cannot answer an engine field names the field and the reason.
  `docs/ARWEN_GLOBAL_CLIENT.md`.
- The spectral core ships inside the package as `arwen_global.spectral`. No
  published engine wheel carried it, so the model could not be installed at
  all before this cut.
- Semi-Lagrangian semi-implicit core at a 300 s step is the default at every
  truncation: six-point quintic gather, order 16 hyperdiffusion at a 720 s
  e-folding time, off-centring 0.55, Lipschitz gate 0.75. The Eulerian core
  `imex_ssp3` stays selectable by name at the step its own refusal admits with
  margin (90 s at T255, 60 s at T383, 40 s at T533).
- A selectable spectral eddy viscosity as the truncation drain, derived from
  the two-point closure theory of turbulence rather than tuned:
  `[diffusion] closure = "spectral_eddy_viscosity"` reads each level's own
  kinetic energy at the cutoff every step and drains at the eddy viscosity
  that energy implies (EDQNM plateau 0.267, cusp 9.21 at decay 3.03, eddy
  Prandtl 0.6 for the scalars, five tail degrees), applied as the exact
  exponential factor per degree and per level the hyperdiffusion is applied
  as. The exponential hyperdiffusion stays the DEFAULT at every truncation,
  and the closure's six fields join a configuration's identity only when the
  closure is selected, so every record config hash is the hash it was before
  this existed. `arwen_global_gdas_t255_native_closure_24h` is the record
  T255 experiment with its `[diffusion]` table changed and nothing else. The
  theory constants are options so a sweep can measure their sensitivity; at
  their defaults nothing in it is tuned. NOT GRADED ON A CARD: the closure
  arm has no card run of its own, so it ships selectable and unmeasured
  against observations, and the drain of record is the one the graded arms
  ran with. The backscatter term of the same closure is not built.
- Native CUDA physics suite: RRTMGP radiation, Morrison microphysics,
  Grell-Freitas and New Tiedtke convection, YSU boundary layer, Noah land
  surface, MM5 surface layer, run a latitude band at a time.
- The card is priced before anything is allocated: the door estimates the
  device peak, weighs it against free VRAM, chooses the latitude band count and
  the host tier, and refuses a plan that will not fit, naming which allocation
  dies first and what the largest truncation that does fit is.
- Ensemble data assimilation: a 32-member T127 LETKF under the T255 control,
  hourly windows, tapered spectral transfer, RTPS inflation, incremental
  analysis update, and a scorecard with four assessments. Eight observation
  doors, all Rust: surface networks, radiosondes, buoys, satellite motion
  vectors, radio occultation, the WMO information system, GOES ABI and ATMS.
  It ships selectable, not default.
- `gpuwm-global fetch-analysis` fetches the one whole-globe GDAS object a
  global cold start needs, with the two non-default flags bound, and prints
  the run command it feeds. The transport is the engine's Rust fetch route.
- The six source mappings this model reads ship inside the package, and one
  resolver answers every bare id and every mapping file name: the engine's
  authority table first, the carried copies second. A published engine
  carries none of the six, so before this every shipped GDAS experiment
  refused at its first door. `gpuwm-global doctor` prints which table
  answered each row and the SHA-256 of the file that answered it, and a
  mapping both tables carry with different bytes is refused by name rather
  than chosen between. A bare name is a key into those tables, never a file
  in the working directory; a spec that names a path (a directory part, an
  absolute path, or `./name`) opens that file. Every row the resolver answers
  is asked both ways the tables spell it, the file whose name ends at the id
  and the family glob, so one spelling of a row cannot read as a gap on a
  table that carries it.
- The decode scratch is translated rather than refused. This model's decoder
  was called with a `scratch_destination`, which names where several GB of
  float64 frames are staged; a published engine places the same directory by
  `GPUWM_COMPOSE_SCRATCH`. `arwen_global.mapped_source_compat` reads the
  environment, places the directory and restores it inside one lock held
  across the decode, so two decodes take turns instead of staging into each
  other, and each decode's receipt names which of the two placed the
  directory it used. A nested decode names this package's enclosing
  placement rather than crediting a caller who set nothing.
- Every product that decodes a source through the engine records how it was
  decoded, beside the mapping digest it already carried: which mechanism
  staged the decoder's multi-GB frame stream and what placed it, whether the
  installed engine's soil-only `preserve_mask` narrowing had to be adapted
  and for which records, and the engine version that did it. The cold start,
  the ATMS columns, the radiation reference cover, the surface-energy state
  product and the upper-air reference all keep the block. The surface-energy
  flux ladder runs no decode at all, since its records are
  product-definition-template 8 read through the engine's GRIB2 bridges, and
  records how its mapping document was read instead.
- A brightness-temperature pack (`gpuwm-obs.goes-bt.v1`, written by this
  package's `rw_goes bt`) reads on a published engine, whose own schema table
  stops one family short. The container parse is the engine's throughout.
- `gpuwm-global obs subscribe` opens the WMO information system feed through
  its Rust door, archiving every notification and payload with its integrity
  digest and reporting coverage per centre. `obs fetch` for that stream
  refuses and names it: no decoder writes the neutral table yet.
- The LETKF analysis runs on the card by default: batched eigendecomposition of
  the localised solve, point operators contracting a device-resident member
  stack. `--letkf-solve-path host` keeps the numpy reference for comparison.
- GOES ABI clear-sky infrared operator (bands 13 and 8 over water) and ATMS
  clear-sky over-ocean operator (channels 4 to 14), each a registered operator
  entry with an acceptance contract, with a CRTM reference leg and a trainable
  fast model.
- One-way regional parent bridge: a global run exports a parent series that
  drives a regional ArWen forecast, with a validation leg on every artefact.
- 55 configured experiments ship inside the package, from a four-step T3 numpy
  smoke to the 25 km forecast day.
- `tools/build_cli_reference.py` writes the command reference from the parser
  itself and `--check` fails when the committed page falls behind it.
- `tools/measure_boundary.py` measures which engine symbols this package uses
  and whether the installed engine carries them, so the dependency range is
  re-measured at every engine bump rather than transcribed.
- `tools/check_doc_examples.py` checks every documented command line against
  the parser this package ships, and runs in CI beside the reference check.

- The completed ensemble assimilation system: the balance package, the hybrid
  covariance at beta 0.75 over a packaged static table, localisation and
  observation errors measured on the ensemble each window, and the radiance
  streams. `gpuwm-global da fresh` with nothing set runs 32 T127 members under
  the T255 control over every stream the day carries.
- `gpuwm-global da static-covariance` estimates the static covariance table,
  and `gpuwm-global da localisation` measures the localisation length on an
  ensemble store.
- The static covariance table ships inside the package (779,912 B), so
  `--static-covariance packaged` resolves without a source tree.
- Latitude-banded native physics: the suite runs a band at a time and the whole
  advanced bundle is byte-identical to the resident run at every band count.
  The door's plan prices band count and host tier together against the card's
  free bytes.
- The ABI and ATMS radiance tables ship inside the package: the fast model, the
  operator entries and both satellites' acceptance entries.

- The physics this model was graded with travels inside the package, as
  `arwen_global.core`: eleven engine modules, the float64 mirror the
  scorecards grade against, the NVRTC loader, twelve kernels and three headers
  (629,739 B) and the four Noah and land-use tables. Nine of the eleven
  modules differ from published `gpuwm 2.7.0`, and eight of the fifteen
  kernel files do: seven translation units and one header, measured on the
  Windows desktop 2026-09-10 with line endings normalised. `noah` and
  `morrison` match the published engine apart from the carve's import
  rewrites and are carried anyway, because their kernels are two of the eight
  and the loader binds its own directory. So a bare install now
  integrates the same bytes the grading tree did. The receipt's scheme
  identity and the cumulus refusal name those carried files by path, and
  the carve rule that produces the spelling is derived from the carve
  table rather than written out.
- `arwen_global/data/engine-seam.json` pins the 46 engine files the carried
  physics reaches and does not carry, by path, size and SHA-256 at the engine
  version they were measured against, and the suite fails when an engine
  module a carried file imports is in neither place. `gpuwm-global doctor` gains an `engine seam`
  section that hashes the installed engine's copies and reports each as proven
  or moved. Moved is a warning naming the file, and the section's summary
  row reads `note` while anything is unproven: the dependency ceiling is
  the refusal.
- `tools/pin_engine_seam.py` writes that manifest by hashing the installed
  engine, and `--check` compares without writing.
- `tools/resync_from_owner.py` cuts the carried core the same way it cuts the
  model, three ways, and its dry run merges into copies so a re-cut can be
  read before it happens. It gives a merged file the line endings the tree
  stores it with, in both directions.
- `.gitattributes` stops git converting line endings on any platform, and
  `tools/check_line_endings.py` refuses a change that rewrites them: a file
  written back under the other convention diffs whole, so the real change is
  removed and re-added with every other line and nobody reads it. The gate
  prints the CRLF and bare-LF counts on both sides of every flip in a
  revision range and walks the tree for a file that holds both at once.
- A run receipt records which physics module actually integrated, its origin
  and the SHA-256 of the file that was imported, with one digest over the
  kernel directory the loader bound. Two copies of several of these modules
  exist in one process, so a version number cannot answer it.
- A run receipt also carries the seam verdict: the engine version the pins
  were taken against, the engine version that resolved, the proven count and
  any file whose bytes moved. The module hashes cannot see the staying half
  of the physics, and one of those files reaches every carried kernel's
  assembled source.
- `gpuwm-data` is declared as a dependency of this package rather than
  inherited from the engine. Carrying the radiation driver made it an import
  requirement: the driver resolves a member of the companion wheel at module
  scope.

### Fixed

- Four `device_cache_key` imports reached beyond the package's own top level
  and made the installed wheel unimportable.
- The four authority-mapping lookups resolved a path relative to this package
  and named a directory that does not exist once the package left the engine
  tree, so every bare source id in every shipped configuration refused.
- One observation at the exact pole refused the whole analysis. Every surface
  row is evaluated through the lowest level's wind, and a lat-lon vector has no
  direction there; the public surface stream carries a station at latitude
  -90.0000, so the first cycle of a global run stopped. That row is now
  rejected by name, and nothing else is.
- A bare `render` named four products the renderer's catalogue does not carry,
  put every product in the skipped list, exited 0 and left an empty directory.
- The analysis quickstart taught two command lines the parser refuses.
- Thirty-nine written command lines were the console script's name with a
  module name after a dot, which no shell can run. They are `python -m
  arwen_global.<module>` now, and a gate holds the tree to it.
- Three published links named a repository that is not published.
- The stream fetch door reported an empty archive window as the exit code
  reserved for a missing Rust door, so a caller acted on it by restaging
  binaries that were already staged.
- The documented microwave command lines were reported as refused by the
  example checker, which reads one parser and could not see through a door
  that forwards to its own.
- The forecast command wrote no `status.json`. It is the longest-running
  command in the distribution and the only long-running one with no machine
  surface, so a workspace driving a day-long integration had a growing log to
  scrape and no answer to what stage it was in or whether it ended.
- A native forecast against a published engine was accepted, priced the card,
  allocated it, integrated its first steps and then refused its own argument
  vector inside the physics: the published surface layer takes no `vegfra`,
  the published radiation no `column_size_bounding`, and neither published
  cumulus constructor a `column_chunk`. Four refusals stood at the door
  because of it. The carve removes the cause; those four, the YSU
  mixing-length refusal beside them and every skip that cited any of the five
  are retired with it. The last row in that table was the mapped-source
  decoder's `scratch_destination`, which is a placement rather than physics
  and is now translated, so the table is empty: against a published 2.7.0
  there is no call this package makes that the installed engine refuses. The
  measurement stays reachable (`tools/measure_engine_signatures.py`) and a
  row added to the table is still refused at the door by name.
- Selecting `ysu_free_atmosphere_mixing_length = "fixed"` was refused on every
  published engine, because the mode is one more integer argument to the YSU
  kernel. The kernel travels with the package, so the option runs.
- The radiation and boundary-layer scorecards graded against whichever float64
  mirror the installed engine happened to carry, which is not the mirror of
  the kernels this package runs. They grade against the carried mirror.
- A run receipt recorded the configuration, the arithmetic pins, the physics
  identity and the machine, and nothing about the libraries. The spectral
  tables ride on a numpy routine whose bits moved between two numpy releases,
  so a receipt could not answer the first question two differing runs raise.
  It records the interpreter and every installed distribution the arithmetic
  rides on, under the self-hash.
- Two comment lines in the carried physics driver cited files in a
  development-tooling directory on the source tree. Neither path resolves in
  any install, and the source of this distribution ships with it. Both are
  rewritten by the carve rule that reproduces them, and the provenance gate
  now stops that directory name at anything that ships.
- Fourteen test modules set the never-open-the-local-device switch at import
  time. pytest imports every collected module before it runs anything, so one
  CPU-only file decided the device for the whole session: on a card host the
  device selection reported 8 failed and 4 errors, nine of those rows nothing
  but the leak, and under that noise a test that reached an engine module a
  published 2.7 does not carry rode the cut unmarked. The switch is set per
  test through `pytest.mark.cpu_only` and put back, the missing mark is on,
  and two gates read the suite's own source for either shape.
- The engine seam's stated scope read as an import closure. It is the direct
  engine imports of the carried physics plus four files, 46 rows; the closure
  is 109 modules, and the 63 the seam does not pin are reached only through
  runtimes this package's door cannot select. The module and the `doctor` row
  say so.
- A plan naming a render product the renderer does not carry resolved at exit
  0 with no warning, built the statics, integrated the whole forecast and died
  at the render stage naming neither the slug nor anything a client could act
  on. On a T255 day that is a forecast day spent to learn a spelling. Every
  token is checked against the renderer's own product list, its group keywords
  and its skip token, asked through the catalogue door rather than
  transcribed: an unknown slug is a warning in a query mode and a refusal on
  the route that starts work, the parameterized form is reported as unchecked
  rather than unknown, and a machine with no staged renderer says it could not
  ask and names what that costs. The same plan now fails in 1 s with no
  checkpoint written and no picture drawn, and the render stage's own words
  reach the machine channel instead of an exit code.
- A `go` plan with no start date resolved with `render` in its stage list and
  no warning, then finished at exit 0 with the stage silently skipped: a
  reviewer who approved it expecting imagery got a green run and an empty
  directory. Both conditions the stage checks have one spelling now, read by
  the stage and by the resolved document.
- Every `gpuwm.run-plan.resolved.v1` document stated a model top of 0.0 Pa for
  all 55 shipped experiments, because the snapshot read the surface end of the
  half-level ladder instead of its top. It reads the top, and a gate holds it
  against the TOML's own value and against the coordinate's own description on
  every shipped config that loads.
- The checkpoint count was taken from step zero on every plan and always added
  the cold state, while a restart begins at its checkpoint's step and writes no
  cold state. Two T21 restarts, from step 60 and from step 180, both promised
  five checkpoints and wrote three and one, in the resolved document, in the
  estimate's disk block and in the `expected_checkpoints` a client sizes its
  progress bar with. The step is read from the checkpoint's own metadata
  record, the cold state is counted only on a cold start, a restart whose
  checkpoint cannot be read reports no count rather than a wrong one, and the
  restart path joins the declared inputs so a missing one is refused before
  anything starts.
- `run-progress.json` named the second-to-last checkpoint at `status:
  complete` on every run by construction: the final checkpoint lands on disk
  when the writer is joined, after the last heartbeat update. The post-loop
  sweep writes the last committed path, the final model time and the final
  step through the same heartbeat the in-loop callback uses, and the client
  page states that contract.
- The physics-profile menu offered three choices no route can execute: three
  shipped configurations load only through the spectral core's own module
  entry. The document carries one row per shipped experiment naming the door
  it loads through and whether a plan can run it, measured by attempting each
  door's own load, and every profile repeats the subset a plan cannot execute.
- A shipped experiment's bare name resolved on `run` and `go` but not on
  `export`, `assimilate`, `da init`, `da analyze`, `da static-covariance`,
  `da localisation`, `abi-score` or `abi-reference --config`, each of which
  opened it as a file in the working directory. The quickstart's no-card
  rehearsal is three lines and the second is an `export`,
  so a reader with no card met `[Errno 2] No such file or directory:
  'arwen_global_moist_smoke'` on the first thing they could run. Every door
  that takes a config resolves the shipped name now, and a gate reads the
  parser rather than a list of door names, so a door added later is covered by
  existing.

### What a bare install runs

`pip install "gpuwm-global[gpu-cu13]"` on a CUDA 13 runtime, or
`[gpu-cu12]` on a CUDA 12 one. Measured on the Windows desktop 2026-09-10 in
three virtual environments created from scratch, one for each engine version
the declared range admits and the public index carries: `gpuwm 2.7.0`,
`2.7.1` and `2.7.2`, with `gpuwm-data` matched to each. The reading is the
same on all three: 231 symbols across the boundary with no gap the host could
see, the two absent symbols in the last row below, and 46 of 46 seam files
proven although the pins were taken against 2.7.0. A bare install resolves
2.7.2 today.

| Leg | Where it comes from | What it needs |
|---|---|---|
| the spectral core, the dynamics, the transforms, the T3 numpy smoke, tape export and the render call | inside this package | the wheels: `gpuwm>=2.7.0,<2.8`, `gpuwm-data` beside it, numpy, scipy, netCDF4 |
| the physics the model was graded with: radiation, cumulus, surface layer, boundary layer, land surface, microphysics, the land-use rulebook, twelve kernels and three headers, the float64 mirror, the CUDA loader and the Noah tables | inside this package, `arwen_global.core` | nothing on the engine's side; any `2.7.x` engine integrates the same bytes |
| a real forecast at T255 and above | needs a card | CuPy matched to the driver's CUDA major, an NVIDIA card with the free bytes the door prices before it allocates |
| GRIB2 source decode, the observation front door, the static-field builder, the `wrfout` writer, the regrid and the renderer | the engine's own bundle, inside the engine's platform wheel | `gpuwm fetch-bridges` where the wheel carries no binaries (the `py3-none-any` install) |
| the eight Rust observation binaries: `rw_asos`, `rw_igra2`, `rw_ndbc`, `rw_amv`, `rw_gnssro`, `rw_wis2`, `rw_goes`, `rw_atms` | this package's companion bundle, `gpuwm-global fetch-doors` | a published release, or `--from DIR` for a local build; every byte is verified against the SHA-256 pins inside the wheel before it is used |
| the six source mappings the shipped experiments name | inside this package, `arwen_global/data/authorities` | nothing; the engine is asked first for every row and its copy wins the day it publishes one |
| the LETKF filter core, its eigensolver kernel, the local-GPU switch, `gpuwm/core/constants.py` and the rest of the staying half | the engine, pinned by path, size and SHA-256 in `arwen_global/data/engine-seam.json` | nothing; a file whose bytes moved is a `note` naming the file, and the `gpuwm<2.8` ceiling is the refusal |
| `preflight.measured_free_vram_bytes` and `surface_bias.interpolate_to_tape` | absent from every published 2.7 (2.7.0, 2.7.1 and 2.7.2 read 2026-09-10) | an engine that carries them. They stop the standalone card-pricing check, which no command is wired to, and the surface-energy scorecard's regrid onto the tape. `gpuwm-global doctor` prints both and exits 1 |

The observation crates are this package's because no published engine bridge
bundle carries six of them at all, and carries the other two in an older
build without the subcommands this package calls. If the engine takes those
crates onto its own line, `arwen_global/data/door-pins.json` goes empty and
every binary resolves from `gpuwm fetch-bridges`.
