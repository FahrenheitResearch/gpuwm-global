# The carried physics, and where it differs from the engine's

`src/arwen_global/core/` carries the physics this model was graded with: eleven
Python modules, the float64 mirror the scorecards grade against, the CUDA
loader, fifteen kernel sources and the four WRF parameter tables. Every one of
them exists on the engine as well, and the two lines move independently. That
is not an accident to be tidied away; it is the arrangement. What it needs is
a file that says, for each place the two differ, what differs, why, and what
should happen the next time one side changes that code.

This is that file. `SOURCE.md` says which revision of the model's own source
tree the package was cut from. `src/arwen_global/data/engine-seam.json` pins
the engine files the carried code still reaches. Neither of those says which
differences are on purpose, which is what this one is for.

## How to use it

**The engine changed a file this package carries.** Find the file's section
below. Find the row whose *engine lines* cover the change. Do what the
**Decision** column says:

| Decision | What it means |
|---|---|
| **pull** | The engine fixed something this package has not taken. Take it, on the model's own source tree first, and re-cut. |
| **refuse** | This package holds a position on that code. Re-apply the engine's change around it, or decline it, for the reason in the row. Never overwrite. |
| **offer** | This package fixed or added something the engine does not have. Nothing to take; the engine line may want it. |
| **none** | Nothing is owed either way. The row exists so a reader does not have to re-derive that. |

The **Class** column says why the difference exists, which is what makes the
decision readable. It is the same vocabulary in
`src/arwen_global/data/engine-divergence.json`, where every row carries it:

| Class | What it means |
|---|---|
| **no-behaviour** | Comments, docstrings, names, formatting. Nothing numeric moves. |
| **adaptation** | This package needed it to run under its own driver: latitude banding, column chunks, band-local geometry, receipts. A future engine change to the same lines is taken with the adaptation re-applied on top of it. |
| **deliberate** | A physics position held on purpose, with the commit that says why. A future engine change to the same lines is refused unless it addresses the same reason. |
| **engine-fix** | The engine fixed something after the common ancestor and this line has not taken it. |
| **global-fix** | This line fixed or added something the engine does not have. A defect both lines carried and only this one repaired is a `global-fix`, because what the row describes is the repair and the work it implies is an offer. |
| **defect** | Unintentional on this side and still present. No row carries it, and one that did would be work owed rather than a description of a difference. |
| **unknown** | What the fingerprint tool writes for a hunk nobody has read yet. The gate fails on it, so it never ships. |

The **Commit** column names the commit that MOVED the lines, on the side the
**Ancestor** column says moved. `engine` there means the published engine's
own line; `owner` means the model's own source tree, the one `SOURCE.md`
records a revision of and out of which `src/arwen_global/core/` is cut. Where a row names a second commit it is a
companion that settled the same question, and where that companion does not
touch the file the cell says so, so `git show <hash> -- <file>` on the named
side always produces a diff.

The carve's own rewiring, the import and prose rules that make the engine's
modules import as this package's, is not a class because it is not a
difference: the fingerprint applies those rules to the engine side before it
diffs, so a hunk that is nothing but rewiring produces no hunk and no row.

**The change is outside every row.** Then it is new, and it needs a row. Read
it, add it here, run `python tools/fingerprint_engine_divergence.py --rewrite`,
and fill in the class and the decision it writes as `unknown`.
`tests/test_engine_divergence.py` fails until you do, by file and by line.

**A row describes a difference that is no longer there.** The same gate fails,
by row name. One side moved under the document; say so here before the row
goes.

Line numbers go stale the first time either side inserts a line above them, so
they are not the identity of a row. The identity is a FINGERPRINT: per carried
file, a unified diff of the engine's copy against this one with zero context
lines and the carve's own rewiring applied to the engine side, then a SHA-256
over each hunk's engine-side lines and another over its carried-side lines. All
35 carried files are measured that way, whatever their suffix; a pair either
side of which does not decode as text has no lines to diff, so it is compared
byte for byte instead and a difference there is one whole-file hunk.
`src/arwen_global/data/engine-divergence.json` carries all 358 of them with the
class and decision of the row they belong to, and the engine line keys its own
change lists by the engine-side hash, so the same difference has the same name
on both sides. `tools/fingerprint_engine_divergence.py` documents the
normalisation exactly and is what recomputes it.

## What this was measured against

* **Engine:** the published `gpuwm 2.7.3`, dated 2026-09-12, which is what
  `pip` resolves inside the range this package declares (`gpuwm>=2.7.0,<2.8`).
  2.7.0, 2.7.1 and 2.7.2 are byte-identical to it in every carried file except
  `core/rrtmgp.py`, `core/physics.py`, `core/physics_inventory.py` and the
  kernel loader, which 2.7.3 moved. Rows on the other files therefore read the
  same against any `2.7.x`.
* **This package:** the model source revision `a2a674a2757c2dfc70fc9a2b12b4412ba3b18fcd`
  of 2026-09-12, which is what `SOURCE.md` records and what `pyproject.toml`
  states beside the version.
* **The common ancestor:** the revision the two lines share, which is what the
  **Ancestor** column is read against. A hunk whose carried text equals the
  ancestor and whose engine text does not is the engine moving alone; the
  reverse is this package moving alone; "neither" is both sides having moved.

**Standing note.** The next engine minor is expected to move several of these
files. That is the workflow this file exists for and not an emergency: install
it, run the fingerprint tool with `--rewrite`, and the rows whose hunks are
unchanged carry their classification forward while the ones that moved come
back as `unknown` and as failures naming themselves. Re-read those, re-check
the rows the same commit touched, and update the three lists at the end. The gate's two comparison nodes run
against the engine version the rows name and skip, naming both versions, on
any other, because four of these files moved between versions that are all
inside the declared range and a failure there would be saying nothing about
this package. The nodes that hold the rows and this page to each other need
no engine and always run.

## What does not differ at all

Ten carried files and the four parameter tables are byte-identical to the
engine's once the carve's rewiring is applied, so they have no rows and an
engine change to any of them can be taken as it stands. That is a measurement
and not an assertion: the fingerprint tool reads every one of the 35 carried
files, the four `.TBL` parameter tables among them, and a file with no row is
a file the tool found nothing in rather than a file it skipped. Change one
byte of any of them, on either side, and the gate fails naming the file and
the lines, exactly as it does for a file that has rows. The list:
`core/noah.py`, `core/kernels/noah.cu`, `core/kernels/ntiedtke.cu`,
`core/kernels/ysu_validation.cu`, `core/kernels/rrtmgp_gas.cu`,
`core/kernels/rrtmgp_cloud.cu`, `core/kernels/rrtmgp_mcica.cu`,
`core/kernels/rrtmgp_validation.cu`, `core/kernels/common.cuh`,
`core/kernels/rrtmgp_planck_common.cuh`, and
`data/noah_tables/{GENPARM,LANDUSE,SOILPARM,VEGPARM}.TBL` with their
`PROVENANCE.md` and `LICENSE-WRF.txt`.

Two of those were not identical a day ago and are worth naming, because
somebody reading an older note will look for them here. The land surface
scheme's kernel carried a substitution in its frozen-ground infiltration
limiter, which the engine fixed on 2026-09-04 and this line took on 2026-09-12
in owner commit `4b74fc7cf`; the carried kernel is byte-identical to the
engine's again. And the microphysics bounding routine rebuilt a reference
density from the wrong temperature, which the engine fixed in the same window
and this line took on 2026-09-12 in owner commit `bef189b1d`. Both are closed.
The rest of that second commit is not closed and is in `MORRISON-CU-1` below,
because it repairs two defects the engine still carries.

## The rows

The sections follow the order of the carve's own table, which is the eleven
modules, then the float64 mirror, then the loader, then the kernels, then the
notice that travels with them.

### `core/rrtmgp.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `RRTMGP-1` | 9-9 | 9-9 | equal to engine | no-behaviour | package cd407ae, 2026-09-12 | nothing | **none** |
| `RRTMGP-2` | 18-34 | after 17 | not present | global-fix | package 78cad51, 2026-09-12 | nothing | **offer** |
| `RRTMGP-3` | after 61, 1974-1974, 2084-2084, 3194-3194, 5037-5037 | 44-45, 1582-1582, 1692-1693, 2779-2780, 4338-4339 | equal to carried | engine-fix | engine 86bd301f5, 2026-09-03 | nothing to any flux | **pull** |
| `RRTMGP-4` | after 85, after 96 | 70-101, 113-118 | equal to carried | engine-fix | engine ee3d2bac0 and 8279bdeeb, both 2026-09-10 | nothing | **pull** |
| `RRTMGP-5` | after 120, after 188, after 190, after 314, 458-458, 467-467, 2160-2181, 2349-2352, 2356-2356, 3568-3570 | 143-180, 249-258, 261-262, 387-413, 540-541, 550-550, 1767-1780, 1943-2015, 2019-2019, 3085-3104 | equal to carried | engine-fix | engine 4a1ed7f3c, 2026-09-10 | nothing at this package's settings | **none** |
| `RRTMGP-6` | 393-443 | 492-501 | equal to carried | engine-fix | engine 6a0713146, 2026-09-05, and 4a1ed7f3c, 2026-09-10 | nothing | **pull in part** |
| `RRTMGP-7` | 390-391 | 489-490 | equal to carried | no-behaviour | engine 71f2b7cd1, 2026-09-10 | nothing | **none** |
| `RRTMGP-8` | after 452 | 511-534 | equal to carried | engine-fix | engine 9d7e57f27, 2026-09-10 | nothing, and it raises at import | **refuse** |
| `RRTMGP-9` | 1708-1715, 1758-1777, after 1966, after 3179 | 1329-1331, 1374-1374, 1564-1574, 2756-2764 | equal to carried | engine-fix | engine a679e477b, 2026-09-05 | nothing for this package | **pull** |
| `RRTMGP-10` | 1079-1088, 1094-1540, 2113-2115, 2189-2190, 2224-2224, 2226-2227, 3131-3139, 3150-3154 | 1156-1156, after 1161, 1722-1722, after 1787, 1821-1821, 1823-1823, after 2715, after 2725 | equal to engine | global-fix | owner 4bf9f3e94, eaa150a42 and 873d73d3b, 2026-09-04; f6999e233, 2026-09-05; 6464f501b, 2026-09-06 | large for the microphysics this model runs | **offer** |
| `RRTMGP-11` | 2270-2276 | 1866-1870 | equal to engine | deliberate | owner eaa150a42, 2026-09-04 | nothing relative to the engine | **refuse** |
| `RRTMGP-12` | 2388-2392, 2405-2430, 2432-2433 | after 2050, after 2062, 2064-2065 | equal to engine | global-fix | owner 984c1cc61, 2026-09-04 | large where transport outruns the microphysics call | **offer** |
| `RRTMGP-13` | 2435-2439, 2441-2455 | 2067-2068, 2070-2076 | equal to engine | global-fix | owner 4bf9f3e94 and 9cfc20654, both 2026-09-04 | large for the frozen size | **offer** |
| `RRTMGP-14` | after 3159, after 3165, 3403-3404, 3407-3407, 3793-3795, 3925-3927 | 2731-2732, 2739-2741, 2989-2992, 2995-2995, 3246-3250, 3380-3382 | equal to carried | engine-fix | engine 267900003, 2026-09-04 | no flux moves; one refusal boundary tightens by a layer | **pull** |
| `RRTMGP-15` | 3419-3434 | 3007-3012 | equal to engine | deliberate | owner 42c37383c, 2026-08-31, restructured by eb34dfa19, 2026-09-05 | at most 1.77 percent of a 1 to 2 hPa layer's emission, where it binds | **offer** |
| `RRTMGP-16` | 52-52, 692-697, 701-701, 2936-2964, after 2980, 3009-3013, 3021-3022, after 3023, 3029-3033, 3040-3049, 3439-3440, 3446-3451, 3453-3512, 3527-3537, 3549-3550, after 3566, 3583-3586, 3596-3597, 3606-3609, after 3621, 3624-3700, 3709-3709, 3711-3780, 4047-4143 | after 34, after 774, 778-778, after 2556, 2573-2573, 2602-2612, after 2619, 2621-2622, 2628-2628, after 2634, 3017-3017, 3023-3024, 3026-3033, 3048-3049, 3061-3065, 3082-3083, 3117-3118, 3128-3129, 3138-3144, 3157-3158, 3161-3186, 3195-3209, 3211-3233, after 3503 | equal to engine | adaptation | owner eb34dfa19, 2026-09-05 | nothing | **none** |
| `RRTMGP-17` | 4257-4276, 4278-4299, 4304-4306, 4308-4308, 4314-4341, 4344-4345 | after 3616, 3618-3619, 3624-3625, 3627-3627, 3633-3647, 3650-3651 | equal to engine | adaptation | owner eb34dfa19, 2026-09-05 | nothing | **none** |
| `RRTMGP-18` | 4024-4025, 4354-4359 | 3479-3482, 3660-3660 | equal to engine | global-fix | owner 4bf9f3e94, 2026-09-04 | nothing inside a scheme | **offer** |
| `RRTMGP-19` | 5148-5155 | after 4449 | equal to engine | no-behaviour | owner 4bf9f3e94, 9cfc20654 and 984c1cc61, 2026-09-04; 6464f501b, 2026-09-06 | nothing | **none** |

**`RRTMGP-1`.** Both copies now open with the same third-party notice. One word inside it differs, because this copy cites the reference commit from its docstring and the engine's cites it from its header. Take an engine rewording of the notice whenever it is convenient.

**`RRTMGP-2`.** The McICA subcolumn generator this radiation is driven with is WRF's RRTMG generator, which is AER's work under AER's own grant, not rte-rrtmgp's. The engine's copy of this file files all six rrtmgp_ sources under RTE+RRTMGP on the strength of the filename prefix, so it transcribes an AER work while naming no AER text. The paragraph added here is the correction, and it applies to the engine's copy unchanged.

**`RRTMGP-3`.** The shipped HDF5 is not thread safe, so every netCDF reader in the process takes one lock. This copy opens all four coefficient files outside it and carries no import of that lock's helper at all. The engine's record is three of eight concurrent runs killed at nine frames of fourteen. Two conditions ride with the pull: that lock's helper is an engine module this package does not carry, so it becomes a new entry in the engine seam manifest, and it exists at the floor this package declares, so the pin does not move.

**`RRTMGP-4`.** The engine declared, in the module that opens them, the five NetCDF members radiation loads, and made the opener refuse a filename outside the set. This copy opens exactly those five, so the check can never fire on today's load path; what it buys is that a sixth member cannot be added to the load path without joining a set something checks.

**`RRTMGP-5`.** The engine turned one microphysics selector from a refusal into a forecast whose radii come from its own particle size distribution. This model radiates one microphysics scheme and its own door admits that scheme by name and refuses every other value, so the branch would be dead code here carrying a dependency on an engine module the seam does not pin. Take it only if this package ever admits a second microphysics scheme, and then take it whole: the constants, the bands arm, the table row, the docstring paragraph, the paths branch and the driver branch are one change, and a partial take publishes a scheme name the radii dispatch cannot serve.

**`RRTMGP-6`.** The deliberate-exclusion table, which the engine emptied in two steps. One of its two entries should go and the other should stay, which is why the decision reads pull in part. The entry for the P3 selector is a long refusal record that the coupling table three hundred lines above already contradicts: that table carries a row for the same selector on all three copies, and only a selector MISSING from it reaches the exclusion table, so the entry is unreachable and should be deleted. The other entry must stay, because the engine removed it only when it added the radii path this package declines above; delete it here and a scheme this package genuinely cannot radiate stops getting the sentence that says why and starts getting the generic add-a-row message, which is the one instruction that is wrong for a deliberate exclusion. So the table ends with one entry here, not zero.

**`RRTMGP-7`.** A comment citing the engine's other named-refusal table by its old name. The module it names is not carried, so either spelling is correct for some engine version and neither is checkable from here.

**`RRTMGP-8`.** A module-scope check that holds this module's two scheme tables equal to the engine registry's rows. Two reasons to refuse it, either sufficient. It imports a name the pinned engine does not define, and the call is at module scope while the global model constructs this radiation on every run, so a carried copy would make every forecast die at import. And on a newer engine it would fail by design, because by the two rows above these tables deliberately lack one row and deliberately keep one exclusion. Revisit only if the engine pin moves forward and this package publishes a registry of its own to be held against.

**`RRTMGP-9`.** The engine moved the gas names and the override validator into a shared module and additionally refuses an override naming a gas the SELECTED coefficient tables do not carry. The only override this package ever passes names a gas both tables carry, so the added refusal never fires; what it converts is a silently ignored override into a refusal. Like the lock above it adds an engine module to the seam manifest.

**`RRTMGP-10`.** Above the cloud tables' upper size bound the engine clips the particle size and leaves the path alone, so a 500 micrometre snow layer is radiated as 180 micrometre ice at 2.8 times its extinction, with no record that it happened. This copy scales the in-cloud path by bound over size and sets the size to the bound, which is the extinction the geometric limit gives, and it counts what it bounded. On the control checkpoints 74 percent of ice cells carrying 91 percent of the ice path sat above the bound. The bounds are read off the loaded tables rather than written as literals, and the record carries cloud-fraction-weighted shares beside the in-cloud ones because the in-cloud share overstates the radiative weight: cells at zero cloud fraction held 18 percent of the in-cloud ice path and none of the radiation. The silent clip is a defect on both lines and the record is how anyone finds out it is binding.

**`RRTMGP-11`.** The couplings for the explicit-radius schemes and for P3 pass the clip treatment on purpose, so above the upper bound they clip at the full path exactly as the engine does; only the counting is new. Their arithmetic is pinned to the older radiation driver's own size cap and snow discount, which already bound the oversize that driver's way, so the geometric carry on top discounted the snow twice and moved four pinned seams. Refuse a future engine change that made these two branches carry instead of clip, unless it also retires those fixtures. The carry belongs to the branch this model runs, which is its own coupling and not a transcription.

**`RRTMGP-12`.** The microphysics kernel writes a 25 micrometre sentinel into a species' radius where that species had no mass at its last update. Transport between two microphysics calls puts real mass in those cells, so at radiation time they carried the sentinel beside mass and radiated at 25 micrometres clipped to the table top, for a droplet population near 5 micrometres. Device counters on an early arm read 52 to 54 percent of the liquid cells in that state with about a fifth of the in-cloud liquid path. Such a cell now takes the radius its own moments give. The engine runs the same kernel with the same sentinel and the same gap between the two calls, so the defect is there unchanged.

**`RRTMGP-13`.** Two changes to the size at which cloud ice and snow enter the ice table. The merge weight moves from number to the mass-weighted harmonic mean, which is the size whose cross-section at the summed mass equals the summed cross-section; number weighting put the merged size at the cloud-ice radius wherever crystals outnumber flakes, which is nearly everywhere, and snow is four fifths of the frozen condensate on the control. Then each species enters at the solid-ice effective diameter its area per unit mass implies, because the table's size axis is defined on solid ice: a 100 kilogram per cubic metre snowflake read at its own diameter received one ninth of its extinction. The two densities are the microphysics scheme's own, the ones its effective radius is defined with, so this is a transcription correction and not a tuning choice.

**`RRTMGP-14`.** The engine composes the longwave and shortwave selectors independently. Five of the six sites are inert with both spectra on: both chunk loops run over every column, both flux pairs are allocated as before, and every flux and heating rate is bit-identical. The sixth moves a refusal even with both on: the above-model layer bound becomes the larger of the two spectra's counts instead of the longwave's alone, which at this package's default model top tightens the level ceiling by one. Take the tightening with it rather than keeping the old bound: the larger count is what the shortwave above-model column actually needs, so the old bound was under-counting, and the only door that can reach a level count that high is the explicit hybrid ladder. This is the one engine change that lands on lines this package rewrote, so the gating has to be re-applied over the chunked loops rather than merged onto them; refusing it buys a conflict on these lines at every future re-cut.

**`RRTMGP-15`.** A 1 hPa model top reaches polar-night air below the k-distribution's own temperature domain, and the alternative to bounding is a refusal in the middle of a forecast. Layer and interface temperatures in the band between the garbage threshold and the table floor are fed to the tables at the floor; the prognostic state is untouched and below the garbage threshold still refuses. The trigger was measured at 159.27 K on the top interface at hour 4.3 of a run. Nothing changes above the floor, which is every column a lower model top reaches, so the engine can take it at no cost; whether it wants a floor at all is its own call rather than a defect on its side.

**`RRTMGP-16`.** The driver packs its inputs and prepares its clouds one column chunk at a time, the fused validator ORs every chunk into one flag word read once, and the record's path sums reduce once over the whole grid from the chunks' terms rather than as a sum of per-chunk sums, because the latter is not the whole-grid sum's bits in a record every checkpoint carries. The one route by which chunking could have moved bits is the subcolumn generator, and it does not: it seeds per column from that column's own bottom four pressures, never from a position in the packed array, so a chunk boundary cannot change a draw. What moves is the device peak, which is why this package carries the driver at all: the whole-grid preparation was the model's peak, and a ten-step probe at the largest truncation died inside the path routine at 19.9 GiB. This is the row that makes a future engine change to the driver body conflict. Anything the engine does between packing and the solver loops has to be re-expressed per chunk here rather than merged.

**`RRTMGP-17`.** The heating rates form per chunk straight into their output arrays. The per-cell expression is unchanged and a column lands at the same index the whole-grid reshape produced; what goes away is the whole-grid difference, net-flux and convergence temporaries. A future engine edit to the flux-to-tendency mapping has to be applied inside the chunk loop.

**`RRTMGP-18`.** Four optional fields appear on the result: the upward and downward shortwave at the top of the column, the upward longwave at the surface as the longwave solver formed it from skin temperature and band emissivities, and the column cloud cover under the maximum-random overlap the subcolumn generator samples. Nothing moves inside a scheme; what moves is what can be read out of one, and before this no top-of-atmosphere or surface radiation budget could be formed from what the result carried. The offer carries a condition: the four fields are declared on the carried physics driver's result type and the engine's ends one field earlier, so the radiation module and the result declaration travel together or neither does.

**`RRTMGP-19`.** The exported-name list gained the public names of the rows above it. It follows them and is not a decision of its own.

### `core/gf.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `GF-PY-1` | 5-6 | 5-12 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing by itself | **refuse** |
| `GF-PY-2` | after 81 | 88-96 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing on the shipped path | **refuse** |
| `GF-PY-3` | 87-92, 94-128, 134-157 | 102-102, 104-108, 114-114 | equal to engine | adaptation | owner fca73b086, 2026-09-04; 957059111, 2090ac84a, e30f9fda6, c4a95bc61 and 58fd47304, 2026-09-05 | nothing at the shipped defaults | **refuse** |
| `GF-PY-4` | 211-231, 267-270, 272-308, 353-354, 356-356, after 362, 365-365, 369-369, 374-393, 399-403, 407-407, 413-419, 422-433, 435-486, 489-496, 502-519, 526-528 | after 167, 203-203, after 204, 249-250, 252-252, 259-259, 262-262, 266-266, 271-289, 295-301, 305-305, 311-323, 326-332, 334-340, 343-351, 357-358, 365-366 | equal to engine | adaptation | owner a74805618, 2026-09-01 | nothing | **offer** |

**`GF-PY-1`.** The docstring of the cumulus driver. On the engine it dates and describes the engine's own gamma replacement; here it describes the gamma this package runs. No number moves. Refused with GF-CU-1, whose position it restates.

**`GF-PY-2`.** Three per-column scalar slots at the engine's driver entry point that let a reference capture's shape factor be pinned in place of the computed one; the engine's shipped path zeroes all three. They exist so the engine can audit its own gamma replacement against a capture. This package computes its shape factor with the gamma it was graded with and has nothing to audit that way, so the slots are refused with GF-CU-9, the kernel half of the same hatch.

**`GF-PY-3`.** The per-column closure reading's names, and the plumbing that compiles the kernel with integer defines for the two coarse-column switches. Both switches default off, so a default run compiles the engine's own text. An engine change to the module dispatcher lands on the wider signature and has to be re-applied rather than copied over.

**`GF-PY-4`.** The seam packs and allocates one column chunk at a time. One thread still owns one complete column, the kernel never reads across columns, and the per-launch slices are contiguous views indexed exactly as before, so a column's answer does not depend on which chunk carried it. The engine already tiles the LAUNCH; what this adds is an outer loop that also bounds the ALLOCATION, and an absent driver lane stops materialising a full-batch zero array. Measured on the largest global batch, 1,283,202 columns by 40 levels on a 32 GB card, the whole-batch input block was 3.07 GB and the output block 3.28 GB and the output allocation failed with 32.2 GB already live; at the shipped chunk the two blocks are 0.65 GB. The engine's seam still builds the whole batch in one block and would hit the same wall at the same size, so it is worth offering back.

### `core/ntiedtke.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NTIEDTKE-PY-1` | 1278-1278, 1281-1293, 1494-1495 | 1278-1278, after 1280, after 1451 | equal to engine | adaptation | owner 92e58c4ff, 2026-09-02 | nothing at the shipped default | **none** |
| `NTIEDTKE-PY-2` | 1352-1366, 1392-1393 | 1339-1339, 1365-1368 | equal to engine | global-fix | owner 3d22c51cf, 2026-09-02 | nothing today, on either line | **offer** |
| `NTIEDTKE-PY-3` | 1452-1469 | 1427-1427 | equal to engine | global-fix | owner 92e58c4ff, 2026-09-02 | real when the lane is bound, nothing when it is not | **offer** |

**`NTIEDTKE-PY-1`.** The cumulus workspace is bounded by one column-chunk option, whichever scheme fills the slot. With no cap the tile width is the engine's exactly, from constants equal on both sides. With a cap the domain is walked in more and narrower chunks; every stage is per column, so tendencies do not see the partition, except one launch scalar that is an OR over the chunk's columns, and the pipeline re-checks that hoist's precondition at the partition's own scope and refuses rather than passing a wrong scalar. A future engine change to the constructor signature or to the chunk walk needs these two regions re-applied, not dropped.

**`NTIEDTKE-PY-2`.** The engine keys its one reusable pipeline on the column and level counts and reads the step and the closure flag only at construction, where the step becomes the kernels' reciprocal step and the mass-flux caps and the closure becomes a kernel scalar. This copy keys on all four. Neither carrier changes its step or its closure between calls, so the wider key never forces a rebuild the narrow key would not have; what it removes is a latent defect the engine still carries, where a carrier that varied its step would silently get the previous call's pipeline and form every tendency with the previous step, with nothing refusing.

**`NTIEDTKE-PY-3`.** The engine fills the spacing array with one scalar for every column; this copy reads the driver's per-column spacing when it is present and falls back to the identical scalar fill otherwise. The spacing reaches the kernel at exactly one site, the scale factors, which multiply the deep closure's adjustment time and divide the shallow closure's mass flux. On a Gaussian grid the square root of the cell area at 60 degrees is about 0.707 of the equatorial value, so for a 50 km equatorial spacing the deep factor goes 1.665 to 1.470, about 13 percent on the adjustment time. Keep it here: the native suite runs on a grid whose zonal spacing shrinks with the cosine of latitude and this closure is scale aware. Offer it: the engine's own driver already publishes that lane and its other cumulus scheme already reads it, so this is the one scheme of the two that ignores a lane the engine already has.

### `core/sfclay.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `SFCLAY-PY-1` | 7-8, after 35, 104-107, 120-123, 138-138, 145-146, 153-154, after 169, 175-177 | 7-9, 37-37, 98-101, 114-115, 128-128, 135-136, 143-146, 162-162, 168-170 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing to the 33 fields this model reads | **pull** |
| `SFCLAY-PY-2` | 72-73, 78-86, 116-116, 125-125, 127-128, 130-130, 148-148, 158-158, 179-180 | 74-74, 79-80, 110-110, 117-117, after 118, 120-120, 138-138, 150-150, 172-172 | equal to engine | deliberate | owner c3f2de9b7, corrected by 47ea9ae89, both 2026-09-04 | nothing at the default | **refuse** |

**`SFCLAY-PY-1`.** WRF's registry carries the momentum friction velocity as unconditional restart state and the surface-layer routine writes it on every column; the port never published it and the engine added it after the fork. It adds a 34th field to the output set, to the result type, to the allocator and to both signatures. Pull it as ONE change across this module, its kernel and the physics inventory: the launcher passes result pointers in the output set's order, so inserting the name at index 2 in the inventory without moving the kernel signature would write each field into the next one's slot all the way down, with nothing raising. The merged signature is thirteen read-only inputs, the vegetation fraction last, then eight inout fields.

**`SFCLAY-PY-2`.** A vegetation-weighted thermal-roughness closure joins the land options as a fourth value, with the vegetation fraction as a new input. It is off by default, and at the three original options the new pointer is never dereferenced, so a default run is bit for bit the engine's. It exists for a measured bias: the default form holds one scalar thermal roughness near 4e-4 m under every canopy, and the ladder that produced it put the 18Z skin temperature 2.66 K above the reference over land with the pattern of the canopy, croplands 6.1 K high, forests 3.6 K high, bare and shrub 2.4 K low. The engine's option validator still admits only the three original values, so an engine edit dropped in unchanged would delete the door. Refuse means re-apply over the fourth branch. The pair is complete here, kernel and mirror, so it is offerable to the engine as a new option rather than as an edit to an existing branch.

### `core/ysu.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `YSU-PY-1` | 33-33, 90-91, 93-99, 150-151, 189-189 | after 32, 89-89, after 90, after 140, after 177 | equal to engine | deliberate | owner 36cb7cd43, 2026-09-05; 8e6bb6aed, the same day, set the default and does not touch this file | nothing at the shipped default | **refuse** |

**`YSU-PY-1`.** The launcher half of the free-atmosphere mixing-length switch: a name resolved through a two-entry table that refuses anything else, passed to the kernel as an integer. The default resolves to the flag the kernel line does not execute. Nothing else moves: validation, output allocation, tiling and workspace pricing are the engine's.

### `core/ysu_contract.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `YSU-CONTRACT-1` | 1-58 | absent | not present | deliberate | owner 36cb7cd43 and 8e6bb6aed, both 2026-09-05 | nothing at the default | **refuse** |

**`YSU-CONTRACT-1`.** A module the engine has never had: a two-entry name-to-flag table, the fixed mode's 30 m constant, and a refusal for any other name. The default name is WRF's own thickness-scaled rule, bit for bit. It exists because on a 40-level global stack whose jet-level layers are about 1500 m thick that rule reads a 150 m asymptotic length where a 300 m regional layer reads 30 m, and the energy ledger found the resulting diffusivity draining the 100 to 400 km kinetic energy at 237 hPa at 0.7 to 1.2 per day. Nothing to pull; keep both modes.

### `core/landuse.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `LANDUSE-1` | 288-297 | after 287 | equal to engine | global-fix | owner 50ac55898, 2026-09-05 | large on ice sheets and glaciers | **offer** |

**`LANDUSE-1`.** Two lines that put a land cell of the ice vegetation class on the ice soil class before the land mask and soil reconciliation, which otherwise converts any land column on the water soil category to mixed forest on silty clay loam. Measured on the desktop with both copies of the same entry point on the same input, a land cell of the ice class on the water soil texture under 120 kg per square metre of snow: the engine gives albedo 0.2968, roughness 0.20 m, emissivity 0.93 and moisture availability 0.60; this copy gives 0.70, 0.001 m, 0.95 and 0.95. Second effect: the carried land-surface driver skips ice-class columns, so after the fix the column leaves that scheme entirely and is run by the frozen surface path instead of being integrated as a forest. On the global statics at this model's working truncation, 30,233 Antarctic columns were affected. Two caveats for the engine, and they are why this is an offer rather than a defect on its side: WRF itself has no land-ice soil rule, so this is a physics choice rather than a closer transcription; and the rule fires on every ice-class land cell, so an engine that wants the fix without the width should condition it on the soil reading the water category.

### `core/physics_inventory.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `INVENTORY-1` | after 83, 88-88 | 83-84, 89-90 | equal to carried | engine-fix | engine d4ec65c0a, 2026-09-04 | nothing in this package | **pull** |
| `INVENTORY-2` | after 113 | 116-131 | equal to carried | engine-fix | engine 8156dd58f and d4ec65c0a, both 2026-09-04 | nothing in this package | **none** |
| `INVENTORY-3` | 182-182 | 200-203 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing in this package today | **none** |
| `INVENTORY-4` | 23-23, after 218, 237-269, 277-296 | after 22, 240-291, 310-319, 327-327 | equal to carried | engine-fix | engine 9d7e57f27 and ecfadd2a9, both 2026-09-10 | nothing in this package | **refuse** |

**`INVENTORY-1`.** A one-conjunct correction to the predicate that says whether a boundary-layer scheme retains a raw output dictionary. It claimed one scheme retains a dictionary it never creates, which made the engine's memory estimator price buffers that do not exist. The scheme is not in this package's admitted set and the sole carried caller is inside the path a run with that scheme never enters. Taking it costs one conjunct and needs no new import. Do not take the commit whole: its other half edits the driver to merge the relocation buffers of the row below.

**`INVENTORY-2`.** Three names for recoupling held tendencies after terrain and base state are transplanted under a moving grid. This model integrates one fixed global grid that does not move, so there is nothing for them to attach to and the names alone would be inert; the behaviour lives in the driver and the restart path, and neither of those is carried. A note for a future re-cut: the engine did not add the three names to this module's export list, which is byte-identical on both sides, so a pull inherits that asymmetry.

**`INVENTORY-3`.** The momentum friction velocity enters the surface-layer output set at index 2. The name is not a number: it is the allocation key, the result type's keyword set, and the POSITIONAL tail of the kernel argument tuple. Inserting it here alone shifts 32 device pointers by one slot against a kernel signature with no such parameter, though in practice it would stop earlier on an unexpected keyword. This package's surface-layer set is internally coherent without the field and the engine's is coherent with it, so neither side is half done. The rule to carry forward: this name can never be pulled on its own. It moves as a set with the surface-layer module, its kernel and the mirror's tuple, and the trigger to do so is this model adopting the engine's two-dimensional mixing or needing the field on output or across a restart.

**`INVENTORY-4`.** The specified-zone ring guard's slot pricing, rewritten to derive from a registry's consumer rows. On the engine it fixes a real under-pricing for a nested run under one microphysics selector. Here the function has no consumer at all: it is exported and called from nowhere, it prices a nested-domain mechanism, and this model has no nest and runs one microphysics scheme. The rewritten body reads a registry name the pinned engine does not define and needs rows in a registry JSON this package does not ship, so taking it converts a dead but correct function into a dead function that raises if anyone ever calls it. The import sits inside the function body, so a re-cut that took this hunk would break nothing at import time and would be easy to miss, which is exactly why it is written down. Revisit if the engine pin moves and this package gains a nested-domain path.

### `core/physics.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `PHYSICS-1` | 40-40, after 52, 1910-1911, after 2677, after 2690, 3046-3050, 3414-3414 | 40-42, 55-55, 1904-1920, 2701-2722, 2736-2740, 3081-3092, 3456-3462 | equal to carried | engine-fix | engine 1467ad9d1 and 8156dd58f, both 2026-09-04 | nothing on a fixed grid | **pull** |
| `PHYSICS-2` | 688-688 | 691-691 | equal to carried | engine-fix | engine db9f5d5bf, 2026-09-05 | nothing here today | **pull** |
| `PHYSICS-3` | 1164-1176 | after 1166 | equal to engine | global-fix | owner 4bf9f3e94, 2026-09-04 | nothing inside a scheme | **offer** |
| `PHYSICS-4` | after 1344, 1356-1356, 1363-1363, 1370-1370, 1377-1384, 1394-1395, 1879-1879, 3661-3662, 4251-4251, 4584-4586 | 1335-1335, 1347-1347, 1354-1355, 1362-1365, 1372-1374, 1384-1384, 1872-1873, 3709-3710, 4299-4303, 4682-4684 | equal to carried | engine-fix | engine d4ec65c0a, 2026-09-04 | nothing reachable from this package's doors | **pull** |
| `PHYSICS-5` | after 1762, after 1773, 4744-4744, 4793-4793, after 5352, after 5445 | 1752-1753, 1765-1766, 4843-4843, 4892-4892, 5444-5445, 5539-5541 | equal to carried | engine-fix | engine 007731bbe and 4d46b0aa0, both 2026-09-05 | nothing when no adapter is attached | **pull** |
| `PHYSICS-6` | 1784-1784, 1791-1791, 5048-5048 | 1777-1777, 1784-1784, 5145-5145 | equal to carried | no-behaviour | engine aeddfc47f, 2026-09-10 | nothing | **none** |
| `PHYSICS-7` | after 1980, 4309-4314, 4414-4416 | 1990-2003, 4361-4404, 4504-4514 | equal to carried | engine-fix | engine 399d95a86 and d7c5a9eca, both 2026-09-02 | zero on a fixed step, by construction | **pull** |
| `PHYSICS-8` | 2914-2929, 5051-5054, 5088-5097 | 2964-2964, 5148-5152, after 5185 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing reachable here | **refuse** |
| `PHYSICS-9` | after 4740, 5217-5217, after 5245, 5258-5258, after 5259 | 4839-4839, 5305-5305, 5334-5334, 5347-5348, 5350-5350 | equal to carried | engine-fix | engine da2e1dd8f, 2026-09-04 | nothing reachable here, and a real wrong answer inside the carried text | **pull** |
| `PHYSICS-10` | 4821-4868 | 4920-4922 | equal to carried | engine-fix | engine 267900003, 2026-09-04 | nothing here, and it would silently change whose physics runs | **refuse** |
| `PHYSICS-11` | 4902-4903 | 4956-4997 | equal to carried | engine-fix | engine b61739b71, 2026-09-10 | nothing in this package | **none** |
| `PHYSICS-12` | 4921-4921, 4923-4926 | 5015-5015, 5017-5023 | equal to carried | engine-fix | engine aeddfc47f, 2026-09-10 | nothing in this package | **none** |

**`PHYSICS-1`.** Two engine commits that are one change to read: the held boundary-layer forcing is allocated once at construction and written into, instead of being left unbound and rebound to the producer's temporary arrays each call, and the raw rates are stored beside the coupled ones so a moved or relocated grid can rebuild the held tendencies on the new mass field without running a scheme or advancing a cadence. On a resident single-domain run the values are identical either way; the difference is identity, not contents, and what it removes is a tile gather or a restart left pointing at the previous call's storage. This model integrates a fixed global grid, so no number it produces moves, but the carried driver is otherwise missing a capability it is written as if it has. The pull is NOT self-contained: it needs the two names of INVENTORY-2. Take the two files together or neither.

**`PHYSICS-2`.** A cadence declared as a decimal that is not a binary fraction becomes a ratio with a denominator near two to the 54th, and this copy then refuses it as not a whole number of model steps even though it is commensurate. The engine reads the decimal the user declared. Reached only from a driver method this package never calls; this package's own radiation cadence goes through the float helper, spelled identically on both copies. So the fix moves no number here and closes a latent refusal inside carried code.

**`PHYSICS-3`.** Four optional fields on the radiation result type. THIS IS THE ONE DIVERGENCE IN THIS FILE THAT THIS MODEL'S OWN DOOR REACHES: the carried radiation module fills them and the radiation scorecard forms its top-of-atmosphere and surface budget from them, so a re-cut that took the engine's result type would raise on the first radiation call. Offer it to the engine, which has the same gap. And refuse in the other direction: if a future engine change renames or drops these four, the carried copy keeps them and the re-cut conflict is the correct outcome. Carriers ADDED to this type should be taken.

**`PHYSICS-4`.** Three changes for one boundary-layer scheme this package does not admit: its vertical momentum becomes a real field of the state dataclass so it survives allocation and serialisation and a step that skips the scheme, the setup refusal that pinned its cadence to zero is removed, and one rate is dropped from the coupling when the state supplies a zero ice placeholder. Taken at a re-cut it is free and stops the carried driver refusing a configuration the engine now supports.

**`PHYSICS-5`.** A root ozone climatology can be attached and evaluated onto the model pressures at every due radiation step, so radiation sees a time-varying ozone column instead of the scheme's own default. Inert without an adapter, which is the only state the carried signature can produce, and the engine module the attachment imports exists at the pinned engine. It keeps the carried driver from being a version of the entry point that quietly cannot accept an argument the engine's callers pass.

**`PHYSICS-6`.** Three literal scheme numbers become the constant the engine already exported for them. It imports cleanly against the pinned engine and changes no value, so a re-cut may take it silently and nothing needs deciding. It is recorded because a reader who finds a new constant name after a re-cut should not have to work out where it came from.

**`PHYSICS-7`.** The radiation and cumulus due predicates gain an override, because under a live time step both of their inputs stop meaning what they say: the step counter is reconstructed as elapsed time over the configured step, which is not a step count once steps differ in size, and the cadence in steps is re-derived from the momentary step each call, so the phase condition fires irregularly. The engine's measurement was radiation at a 373 s interval against a 358 s target, and 660 s with the cadence frozen. This model integrates at a fixed step and does not call the method, so this is inert either way; the pull keeps the carried driver correct for anyone driving it with a live clock, and the two attributes are self-contained.

**`PHYSICS-8`.** The driver half of the momentum friction velocity: the kernel writes it, so the host stops forming it after the call and stops allocating it for the one path that consumes it. It cannot be taken alone. This package's output set has no entry for the field, so deleting the allocation leaves the name unbound at both the post-call correction and the branch that seeds it; and taking the whole change means taking the engine's surface-layer module and kernel, which have no vegetation fraction and would silently drop the graded closure of SFCLAY-PY-2. This package's doors admit neither the mixing option nor a configuration with the boundary layer off, so the field is unreachable: the pull costs a measured physics choice and buys nothing. The correct resolution is the offer in SFCLAY-PY-2; if the engine takes the vegetation fraction, this comes across with it in one step.

**`PHYSICS-9`.** The run's land-use dataset reaches the parameter-table readers instead of three copies of one hardwired default. A case whose statics use the other convention therefore receives the wrong vegetation rows here with nothing raised: wrong roughness, albedo, leaf area and stomatal resistance for every land column. It is only not a defect because nothing reaches it: this model's own runtime already passes its statics convention into the table reader, so its tables match its land-use field regardless. The keyword defaults to the same string, so no existing caller changes and the two copies stop disagreeing about something with a right answer.

**`PHYSICS-10`.** The engine replaced the inline radiation selector ladder with a call to a shared factory. That factory's branch for this model's radiation imports the ENGINE's radiation class: the one with no cloud-particle size bounding, no column chunking and none of PHYSICS-3's carriers. Pulling three lines would route this package's radiation construction straight back to the physics the carve exists to replace, and silently, because the import succeeds and the object is a working adapter. If the independent composition is ever wanted here it has to be written against the carried radiation module. This is the clearest case in the package of a change that is right on the engine and wrong to carry.

**`PHYSICS-11`.** The engine re-checks, at the driver, a scheme pairing its configuration doors already refuse. Two facts make it a no-op here: this driver is not this model's physics path, and the pairing cannot be built even in principle, because the built-in adapter admits one boundary-layer scheme by name and refuses every other value including none. The underlying property is still true of the carried cumulus kernel, so if a future door ever makes the boundary layer optional beside that cumulus scheme, this refusal is the one to bring across.

**`PHYSICS-12`.** The engine narrowed a refusal: a dry boundary-layer run now reaches the driver for four schemes, and only the fifth keeps a moist requirement, for the saturated stability it forms. This model is moist on every path it ships and does not use this driver at all. Recorded so that a re-cut taking it is not mistaken for a loosened gate on this side.

### `core/morrison.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `MORRISON-PY-1` | 42-43, 57-60, 135-144, 168-168, 174-176, 210-211, 252-254 | 42-42, after 55, after 129, after 152, 158-158, after 191, 232-232 | equal to engine | global-fix | owner bef189b1d, 2026-09-12 | nothing by itself; it carries the two quantities of MORRISON-CU-1 | **offer** |

**`MORRISON-PY-1`.** The driver half of the sedimentation repair: two scratch volumes, allocated as named slots so a stepping run creates nothing per step, carrying the two per-level quantities the process stage publishes to the sedimentation stage. At this model's working truncation with 40 levels that is two 45 MiB volumes, and both arms of the card comparison reported the same device peak, 8.80 GiB, so they did not move the card requirement.

### `core/npref.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NPREF-1` | 1-37 | after 0 | not present | global-fix | package cd407ae, 2026-09-12 | nothing | **offer** |
| `NPREF-2` | 1045-1046, 1051-1056, 1075-1083, 1123-1132, 1136-1136, after 1137, after 1139, 1387-1389, 1409-1412, 1945-1945, 2047-2048, 2096-2097, 2135-2135 | 1008-1008, after 1012, 1031-1034, 1074-1075, 1079-1079, 1081-1081, 1084-1084, after 1331, after 1350, 1883-1883, after 1984, 2032-2032, after 2069 | equal to engine | global-fix | owner bef189b1d, 2026-09-12 | the mirror half of MORRISON-CU-1 | **offer** |
| `NPREF-3` | 4320-4320, 4489-4491, after 4502, after 4766, 4806-4806, 4819-4821, 4828-4828, 4857-4857 | 4254-4254, 4423-4425, 4437-4437, 4685-4690, 4730-4730, 4743-4745, 4752-4754, 4777-4778 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing in any field this mirror already returns | **none** |
| `NPREF-4` | 4736-4749, 4752-4753, 4758-4759, 4837-4844, 4859-4859, 4866-4866 | 4671-4671, after 4673, after 4677, 4763-4764, after 4779, 4786-4786 | equal to engine | deliberate | owner c3f2de9b7, corrected by 47ea9ae89, both 2026-09-04 | nothing at the three original options | **refuse** |
| `NPREF-5` | 6365-6366, 6368-6372, 6408-6410, 6752-6753, 6756-6757 | 6285-6285, after 6286, after 6321, after 6678, after 6680 | equal to engine | deliberate | owner 36cb7cd43, 2026-09-05 | nothing at the shipped default | **refuse** |
| `NPREF-6` | 6459-6460, 6481-6481, 6522-6523 | 6370-6377, 6398-6398, 6439-6455 | equal to carried | engine-fix | engine 4d523b793, 2026-09-04 | real, on columns that straddle the first interface | **pull** |
| `NPREF-7` | 6547-6554, 6557-6569 | 6479-6496, after 6498 | neither | global-fix | owner 304177d6f, 2026-09-04; engine 4d523b793, same day | identical where the index is above one; the height differs in one corner | **offer** |
| `NPREF-8` | 6580-6581 | after 6508 | equal to engine | global-fix | owner 304177d6f, 2026-09-04 | narrow but not empty | **offer** |
| `NPREF-9` | 7435-7441, 7443-7444, 7450-7466, 7468-7471, 7485-7485, 7491-7496, 7499-7499, 7503-7503, 7518-7521, 7534-7535, 7537-7541, 7557-7557, 7559-7559 | after 7357, 7359-7359, after 7364, 7366-7368, 7382-7382, 7388-7391, 7394-7394, 7398-7398, 7413-7415, after 7427, 7429-7435, 7451-7451, 7453-7453 | equal to engine | global-fix | owner 50e109983, 2026-09-04 | up to 3.35 W/m2 of spurious absorption removed | **offer** |
| `NPREF-10` | 7746-7746, 7751-7890, 7897-7900, 7914-7930, 7932-7933, 7965-7967, 7987-7989, 8020-8024, 8038-8048, 8050-8051, after 8052, 8054-8071 | 7640-7640, after 7644, 7651-7651, 7665-7666, after 7667, 7699-7700, 7720-7722, after 7752, after 7765, 7767-7768, 7770-7771, 7773-7777 | equal to engine | global-fix | owner 4bf9f3e94, 9cfc20654 and 984c1cc61, 2026-09-04; eaa150a42, 2026-09-04; f6999e233, 2026-09-05 | the mirror half of RRTMGP-10, RRTMGP-12 and RRTMGP-13 | **offer** |
| `NPREF-11` | after 9331, 9333-9333, 9418-9419, 9432-9433, 9435-9437, 9439-9442, 9444-9444, 9472-9472 | 9038-9043, 9045-9045, 9130-9133, 9146-9149, 9151-9154, 9156-9159, 9161-9161, 9189-9189 | equal to carried | engine-fix | engine ab874b32e, 2026-09-04 | 1.23 percent on the shallow tendencies at a 90 s step | **pull** |

**`NPREF-1`.** The third-party notice for what this file transcribes. Its RTE+RRTMGP half matches the engine's own header for the same work; its AER half is the correction of RRTMGP-2, in the mirror rather than the driver, and applies to the engine's copy unchanged.

**`NPREF-2`.** The float64 mirror is the instrument the scorecards grade against, so it moves with the kernel or it is a flawed instrument. Here it stops rebuilding the two per-level quantities and requires them instead of falling back, which is how it produced a third value distinct from both the kernel's and WRF's.

**`NPREF-3`.** The engine adds one output, the momentum friction velocity, computed as the same relaxation as the friction velocity but on the raw wind speed, without the convective and subgrid additions, without the wind speed's own floor and without the land floor. Everything else is identical arithmetic on both sides. Do not pull the mirror hunks alone: this package's kernel contains no occurrence of the field and this model's boundary layer never reads it, so a mirror returning a field the kernel it mirrors does not write is exactly the failure a float64 mirror exists to prevent. One loose end worth recording: this file still contains a turbulence-energy routine that ACCEPTS the field and falls back to zero when it is absent, so the carried tree can consume it and cannot produce one. If this package ever takes the engine's surface-layer kernel, these hunks ride with that kernel change in the same commit.

**`NPREF-4`.** The mirror half of SFCLAY-PY-2. The correction is the half that made it survive: with the momentum roughness inside the roughness Reynolds number, an unvegetated column at a roughness near 1 m and a friction velocity above 1 m/s put the roughness ratio near 100, decoupled the skin from the air, and the first day-long arm died on a non-finite wind inside its first hour. A future engine change to the option ladder is refused if it would remove the fourth option or fold it into a WRF branch; a change confined to the first three lands cleanly on the branch above and can be taken.

**`NPREF-5`.** The mirror half of YSU-CU-4. At the default this is bit-identical to the engine. Selecting the fixed mode replaces the thickness-scaled asymptotic length with a constant, which changes the squared length and so the local diffusivity above the boundary layer: on the 40-level global stack, whose layers across the jet are about 1500 m thick, WRF's rule reads a 150 m length where a 300 m layer reads 30 m, about 25 times the momentum diffusivity for the same shear and Richardson number.

**`NPREF-6`.** WRF wraps the whole thermal-enhanced Richardson sweep in the boundary-layer flag, and nothing in the block above can raise that flag, so WRF has no path from the local-diffusivity regime into the convective one inside a step. This copy runs the sweep unconditionally and then recomputes the flag, so a column whose first-guess top sat below the first interface can be promoted into the convective regime by the thermal excess and receive countergradient transport and the entrainment block WRF would withhold for the whole step; the reverse also happens. This copy also applies the interface clamp inside the diagnosis, so a clamped index becomes the starting index of the liquid-water scan, where WRF interpolates and clamps once, after the scan. Two conditions on the pull. It must ride with the kernel, YSU-CU-1, or the mirror grades a kernel it no longer matches. And it must be applied on top of NPREF-7's restructured block rather than by taking the engine's whole diff for that commit, because the second half of that commit collides with this package's fix from the same day. Expect the boundary-layer scorecards to move, and re-read NPREF-5's grade afterwards, because it was measured under the unguarded sweep.

**`NPREF-7`.** The one hunk in this file where both lines moved. Both treat the two statements at the end of WRF's scan as the independent statements they are, and that half is now identical. They differ on what happens to the boundary-layer height. The scan can set the flag true on its last visited level without ever moving the index off one, and the engine then interpolates with an index of zero, reading the element before the start of the column, which in Fortran is undefined and in the mirror is the top model level: the height comes back somewhere between the column top and the first level before the flag is cleared, and that value is what the caller reads and what the stable enhancement's own test reads next. This copy skips the interpolation on that path entirely. This form is a strict superset of the engine's: it contains the engine's independent clearing statement verbatim and adds the guard that refuses the undefined read. The project rule is to implement defined behaviour where WRF is undefined rather than reproduce the read, so nothing is pulled here, and the same read is still live on the engine in both its mirror and its kernel.

**`NPREF-8`.** WRF clears the same flag after the stable enhancement's own sweep when that sweep returns an index of one. This copy added the statement; the engine has no equivalent. The enhancement runs only under a stable surface and a low boundary-layer top, and the flag can still be true there when the liquid-water scan revived it, so on the engine such a column can carry the flag true into the entrainment block with an index of one, where the entrainment level is minus one and indexes the top of the column. Offer it with NPREF-7: the two statements are the same rule at the scheme's two sweeps and the engine took only the first.

**`NPREF-9`.** The floor under the two-stream diffusivity moves from ten thousand times the float32 epsilon to ten thousand times the float64 epsilon, which is the reference's own value at the reference's working precision and only keeps the quantity strictly positive; the higher floor is an absorption the medium does not have in any layer whose true value is below it, which is the conservative-scattering bands. Measured: a conservative layer of optical depth 10 absorbed 0.50 W/m2 of 680 incident and one of depth 30 absorbed 3.35, and through the shipped cloud optics an overcast liquid cloud lost 0.24 to 2.27 W/m2 of reflected flux. The literal alone is not enough: the equations as written are unstable in float32 at small products, 6.9 W/m2 of spurious absorption at optical depth 2, so the layer arithmetic is rearranged through the exponential-minus-one function and an algebraic identity so that every term is of the order of the quantity itself. The function also gains a dtype argument so the mirror can be evaluated in float32 to reproduce the device kernel. Offer it AS A PAIR with RTE-CU-1: the engine's mirror deliberately sets its floor from the float32 epsilon so that it matches its own kernel, and both then disagree with the reference, so an engine that took the kernel alone would have a mirror that no longer mirrors it.

**`NPREF-10`.** About 330 carried lines against 145 on the engine, which makes this the largest single block of divergence in the file. It is the mirror of the three cloud-optics rows: the counted size bounding with its geometric carry, the area-conserving ice and snow merge, and the sentinel reconstruction, plus the column cloud cover the engine does not have. The arguments that select the old behaviour exist so an instrument can state the before beside the after; they are not device options, and their defaults are the corrected coupling, so a bare call gets the fixed behaviour. A future engine change anywhere inside this block will need re-expressing rather than merging.

**`NPREF-11`.** WRF's shallow cumulus arm re-sets its adjustment time to exactly 2400 s, discarding the rounding applied earlier, and every feedback tendency then divides by the unrounded value while the closure and advection arithmetic above it keeps the rounded one. This copy divides all of the shallow feedback tendencies by the rounded value. Whenever the step does not divide 2400 exactly the two disagree by the rounding ratio; at common steps they are identical. The case for pulling is unusual: this package carries no kernel for this scheme, this model runs no such scheme, and the mirror function reaches the engine for its own table through an import the carve deliberately leaves pointing at the engine. So the only kernel this function can ever be checked against is the published engine's, which has carried the fix since before the pinned floor. That makes the carried mirror wrong against the only kernel it mirrors, which is the definition of a flawed instrument. The pull is bookkeeping, costs nothing because no run reaches this code, and removes a trap for anyone who calls it.

### `core/kernels/__init__.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `LOADER-1` | after 96 | 97-112 | equal to carried | engine-fix | engine 322df3fa6 and 94f4aa331, both 2026-09-10 | nothing in this package | **refuse** |

**`LOADER-1`.** The loader gained a branch that fires only for a module name with one prefix, sending a standalone land-surface translation unit through a composition factory instead of the plain compile route. The carried loader cannot be handed such a name, and that needs spelling out, because this package DOES run that scheme: it imports the step function from an engine module, maps the selector to it and calls it. That scheme compiles on the engine's side of the seam. All thirteen engine modules that build its device code import the engine's loader, not this one, so such a name never reaches the carried one. The carried loader's own inputs say the same from the other end: the carve carries fifteen kernel files and none is one of that scheme's units, the kernel directory has no fallback to the engine's, and the three carried call sites name carried kernels only. Refuse on unreachability alone: the branch is dead code here whichever engine version is installed. On symbol availability, recorded so nobody re-derives it as a reason, the helper the branch calls is absent from the floor of this package's declared range and present from 2.7.3 up, so on an install resolving today's engine the import would succeed and nothing would raise. Reopen only if the carve ever carries one of that scheme's kernels, and then the engine's translation-unit composition has to come with it, not just this branch.

### `core/kernels/gf.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `GF-CU-1` | 14-14, 42-47, 49-54, 56-61, 629-635, 3030-3032, 3040-3046 | 14-15, 43-49, 51-57, 59-95, 576-583, 2813-2822, 2830-2833 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | taking the engine's would move the deep cloud-base mass flux by up to 7.3 percent | **refuse** |
| `GF-CU-2` | 1246-1246 | 1183-1183 | equal to carried | engine-fix | engine ab874b32e, 2026-09-04 | moves the deep and the shallow closure at once | **pull** |
| `GF-CU-3` | 98-184 | after 131 | equal to engine | deliberate | owner bc62dcfdd and 4939f1c13, 2026-09-04; 957059111, 2026-09-05 | nothing at the shipped defaults | **refuse** |
| `GF-CU-4` | 1199-1210, 2239-2239, 2304-2317 | 1147-1147, after 2134, after 2198 | equal to engine | deliberate | owner 957059111, 2026-09-05 | nothing with the define off | **refuse** |
| `GF-CU-5` | 2344-2344, 2375-2390, 2480-2480, 2948-2948, 3002-3003, 3016-3016 | after 2224, after 2254, 2344-2344, after 2733, 2787-2787, after 2799 | equal to engine | deliberate | owner bc62dcfdd and 4939f1c13, both 2026-09-04; 55aa34fb4, 2026-09-04 | nothing at the default | **refuse** |
| `GF-CU-6` | 2900-2920, 3933-3936, 3939-3939, 3948-3948, 3965-3966, 4169-4169, 4171-4172, 4198-4200 | after 2706, after 3719, 3722-3722, after 3730, after 3746, after 3927, 3929-3929, 3954-3955 | equal to engine | deliberate | owner 957059111, 2026-09-05 | nothing with the define off, re-derived rather than assumed | **refuse** |
| `GF-CU-7` | 1487-1488, 1511-1513, 1516-1517, 2777-2833, 2840-2840 | 1422-1422, after 1443, after 1445, after 2640, 2647-2647 | equal to engine | deliberate | owner 2090ac84a, e30f9fda6, c4a95bc61 and 58fd47304, all 2026-09-05 | nothing with the define off, re-derived rather than assumed | **refuse** |
| `GF-CU-8` | 1341-1341, 1346-1346, 1390-1390, 1495-1495, 1710-1730, 1743-1744, 1751-1759, 2773-2773, 2867-2867, 3014-3014, 3992-4010, 4016-4030, 4184-4185, 4194-4194, 4300-4315, 4322-4325 | 1278-1278, after 1282, after 1325, after 1428, after 1637, after 1649, after 1655, 2637-2637, 2674-2674, 2798-2798, after 3781, after 3786, 3941-3941, 3950-3950, after 4054, after 4060 | equal to engine | adaptation | owner fca73b086, 2026-09-04; 957059111 and 58fd47304, 2026-09-05 | nothing, verified by reading every new access | **offer** |
| `GF-CU-9` | after 3976, after 3978, after 4059, 4164-4164, 4190-4190 | 3757-3765, 3768-3768, 3816-3818, 3923-3923, 3946-3946 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing on the shipped path | **refuse** |

**`GF-CU-1`.** The comment block and the library probe of the cumulus kernel, on the side of the gamma the kernel calls. The shape factor of the mass-flux profile is a ratio of three gamma functions, and the closure's difference of two nearly equal quantities turns one unit in the last place of it into up to 7.3 percent of the deep cloud-base mass flux, so which gamma the kernel calls is a physics choice. The gamma this package carries is its own work and the one the global model was graded with; the engine line's correctly rounded variant moves that shape factor on 68.17 percent of reachable tuning values (the engine line's own host-compiled enumeration of 2026-09-04, cited in full under LIBM-3) and is a numerics choice to grade against observations, not to pull. The number-moving definition is the shared header, LIBM-3.

**`GF-CU-2`.** One character. The undilute buoyancy integral skips only layers strictly below cloud base in the reference, and this copy drops the cloud-base layer itself. The engine measured its test column's integral at 0.0 before and 0.0348961316 after, identical to the float64 reference. The routine is called three times in the deep column and three times in the shallow one, so both arms move: the deep quasi-equilibrium member and the cloud efficiency, and through them the cloud-base mass flux; and the shallow scheme's own rejection test, its efficiency and its closure weight, hence its mass flux. It is a transcription error and not a choice: the sibling routine in the same file transcribes the opposite test character for character, and the float64 mirror already reads it the corrected way. The committed oracle capture is byte-identical either way, so no parity gate catches it.

**`GF-CU-3`.** The two compile-time defines for the coarse-column deep arm, and the comment that states what each one does. Both default to zero, so a default run compiles the engine's own text. They exist because at 50 km the grid scale cannot carry deep convection: the observed failure was a warm-pool column with 1,200 J/kg of available energy and seventeen saturated levels raining 84 mm/h through the microphysics while the cumulus scheme booked 0.06 mm/h. A future engine change to the code these defines guard is re-applied around them, never instead of them.

**`GF-CU-4`.** With the define off both guards are inside the conditional block, so the kernel takes the reference's exit exactly as the engine does and the counter is only exported. With it set, a level the downdraft's profile reaches with no mass takes the environment's moisture, enthalpy and momentum, evaporates nothing, contributes no buoyancy, and the sweep continues below it. The reason is the same as the row above: on a column the grid resolves none of, exiting the whole scheme because the downdraft profile collapsed within a level or two of its detrainment height hands deep convection to a grid scale that cannot carry it. The updraft's own zero-denominator exit is untouched on both sides.

**`GF-CU-5`.** With the define off, the override is a disjunction of two zero macros, so the flag stays zero and the downdraft entrainment reduces to the engine's expression; the export chain is write-only. With it set, a column the reference rejects for a downdraft that is not negatively buoyant, or for one whose moisture equation exits on a zero denominator, keeps its deep updraft, its rain and its closure with the downdraft forced off, and the overridden exit is exported per column.

**`GF-CU-6`.** With the define off the threshold this adds is initialised to zero and written only inside the conditional block, so the comparison against the positive per-level heating cap never fires, the shallow call passes a literal zero and is excluded by its own test anyway, and the two new out-parameters are read back only into void casts and the census. With the define set, the 300.01 K/day cap becomes the larger of itself and the column's own profile peak scaled by the latent heat of the moisture the grid is converging into it, so a column whose request stays under the reference bound is untouched and one above it is let through up to that latent heat. The bound is tuned for grids that resolve part of the convection; on a 52 km column the scheme is the only sink. The negative check's signature grew three parameters and both of its call sites moved with it, so any engine change to that routine or to either call arrives as a conflict and has to be re-applied on the wider signature.

**`GF-CU-7`.** The floor is declared zero OUTSIDE the conditional block and computed only inside it, and the mass flux is non-negative by construction, so with the define off the comparison never binds. With it set, a convecting column's cloud-base mass flux is floored at a share of the Kuo moisture-convergence member, partitioned by a critical humidity of 0.9 measured on the control's own storm hour rather than inherited. It is applied after the diurnal-cycle term, so a column that term silenced can convect on the floor alone: 1,750 of 3,063 floored columns at hour 12, 1,298 of them silenced ones. The reason is that the sixteen-member mean gives the moisture-convergence members a quarter of the weight, a hedge that is wrong where the grid carries none of the converging moisture itself. The output routine gained two parameters and its call site carries both, so an engine change there is a three-way merge.

**`GF-CU-8`.** A per-column reading of what the closure asked for and what it applied, built because the storm columns of the control left the scheme through the downdraft exits or under the heating cap and no number said which. Every new slot is written and never read back into a term, the new slots are appended after the existing ones so no index moves, and the three reads that were introduced are each shown inert with both defines off in the rows above. All arithmetic in the kernel goes through the round-to-nearest intrinsics, so the added stack structure cannot perturb results through contraction either. This is most of why this scheme's driver sits at 45.7 percent line similarity with the engine's; the file is not rebuilt physics. The engine may want the same instrument. Three device-function signatures grew, so any engine change to those routines has to be re-applied on the wider signature.

**`GF-CU-9`.** The kernel half of GF-PY-2: three per-column override slots for the shape factor and the two call sites that pass zero here where the engine passes the override. The engine opened the hatch so its own gamma replacement stays auditable against a reference capture. This package keeps the gamma it was graded with, so the hatch is refused with GF-CU-1 and GF-PY-2.

### `core/kernels/sfclay.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `SFCLAY-CU-1` | 10-13, 236-237, after 262, after 509, 544-544 | 10-13, 219-220, 246-246, 472-478, 513-514 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing to any existing output | **pull** |
| `SFCLAY-CU-2` | 15-31 | after 14 | equal to engine | no-behaviour | owner 9360aca3e, 2026-09-04 | nothing, comment only | **none** |
| `SFCLAY-CU-3` | 472-492, 495-495, 500-500 | 456-456, after 458, after 462 | equal to engine | deliberate | owner c3f2de9b7, corrected by 47ea9ae89, both 2026-09-04 | nothing at the three original options | **refuse** |

**`SFCLAY-CU-1`.** The kernel half of SFCLAY-PY-1: one inout pointer, three lines that read the previous value, form the raw wind speed and store the relaxation, and nothing else. Verified by reading the engine kernel: neither new local appears anywhere else. It is the friction velocity relaxation without the convective correction, without the wind speed's floor and without the land floor, which is why it is a separate field rather than a diagnostic. Inert for this model today, and the reason to take it is that it removes a permanent three-file drift at zero numeric cost.

**`SFCLAY-CU-2`.** Seventeen comment lines recording this kernel's column-by-column grade against hash-verified reference sources. Kept because it is the standing reason to REFUSE a future patch from either line that adds the moisture-flux floor, the negative sensible-flux floor or the dissipative heating term: all three are commented out in both reference files, so their absence here is the transcription and adding them would be bit-exact to a scheme the reference does not run.

**`SFCLAY-CU-3`.** The kernel half of SFCLAY-PY-2. The engine's line here is still the two-way conditional, which would silently run the fourth option as the third, so any engine change to that block has to be re-applied over the new branch.

### `core/kernels/ysu.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `YSU-CU-1` | 299-302 | 298-316 | equal to carried | engine-fix | engine 4d523b793, 2026-09-04 | measured on the engine's own mirror for one such column: index 4 to 1, height 431.15 to 95.43 m, countergradient 10.23 to 0, peak heat exchange 22.0035 to 0.0100 | **pull** |
| `YSU-CU-2` | 331-352 | 345-360 | neither | global-fix | owner 304177d6f, 2026-09-04; engine 4d523b793, same day | identical wherever the index is above one | **offer** |
| `YSU-CU-3` | 365-367 | 373-374 | equal to engine | no-behaviour | owner 304177d6f, 2026-09-04 | nothing; the statement is provably unreachable on both sides | **none** |
| `YSU-CU-4` | 166-167, 569-575, 578-578 | 166-166, after 575, after 577 | equal to engine | deliberate | owner 36cb7cd43, 2026-09-05 | nothing at the shipped default; about 25 times the free-atmosphere diffusivity when selected | **refuse** |

**`YSU-CU-1`.** The kernel half of NPREF-6. Dropping the re-clamp is separately bit-neutral given the guard, which the engine proved by hashing all fifteen returned fields over 4,000 randomised columns. Pull it as three parts: this hunk, the mirror's diagnosis helper gaining the clamp switch, and the mirror's own guard.

**`YSU-CU-2`.** The kernel half of NPREF-7. The engine's commit asserts the read is unreachable with its two rules in place. It is not: reaching it needs the flag true with an index of one on entry, and entering the scan with a false flag and an index of one is ordinary, because a stable-surface column takes the branch that clears the flag and the first-interface clamp is unchanged on both lines; the scan then raises the flag on any unstable level and the pending index update is never consumed if that level is the last one visited. Neither engine rule closes it: the guard of YSU-CU-1 applies only inside the surface-convective branch a stable column never enters, and the statement split changes what happens after the read, not whether it happens. Reproduced here for a stable-surface column whose liquid-water deficit sits 0.02 K below the surface value at the last visited level: the index read 1 with the top-down term on against 8 with it off, in the mirror and on a card; with the guard both read 8 and every output was bit-identical between the two settings and finite. Offer the kernel hunk with the mirror.

**`YSU-CU-3`.** A clearing statement after the stable re-diagnosis, citing its reference line. Reaching that block needs a stable surface, which means the branch above set the flag false, so only the liquid-water scan can revive it; if it does, the block above either clears the flag or recomputes the height and clears it whenever the height fell below the first interface. So arriving there with the flag still true implies both an index above one and a height at or above the first interface, and the second of those is exactly what keeps the branch from being entered. Keep it: it costs nothing, it cites its reference line, and it stops being dead the moment the block above it changes.

**`YSU-CU-4`.** One added line and the flag that controls it. At the shipped default the line never executes and the arithmetic is bit for bit the engine's. The mode is selectable and reported as a workaround rather than made the default, because its 24 h grade split: the northern 250 km energy ratio at 250 hPa improved 0.67 to 0.85 with a better speed bias, while the northern 250 hPa vector wind error rose 4.41 to 4.56 m/s and the grid-limit ratio 0.038 to 0.053. Refuse means re-apply, not drop: an engine change to the asymptotic length goes into the flag-zero path, which is meant to be the reference bit for bit, and the flag-one line is left alone unless the engine change addresses the same layer-thickness scaling, in which case this option is what is being replaced and should be retired rather than kept beside it. Note that the extra kernel argument is a binary-interface break in both directions: the published launcher would pass the level count into the flag and shift every later argument by one.

### `core/kernels/morrison.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `MORRISON-CU-1` | 157-162, 164-165, 173-173, after 201, 241-241, 249-252, 268-271, 800-800, 805-806, 818-819, 830-831, 848-848, after 858, 864-865, after 909, 946-947, 1013-1013, 1019-1019, 1025-1026, 1049-1050, 1070-1070, 1073-1073, 1077-1077, 1081-1081, 1085-1085, 1089-1089, 1112-1113, 1122-1123 | after 156, 158-158, 166-167, 196-197, after 236, after 243, after 258, 787-787, 792-792, 804-804, 815-815, 832-832, 843-843, 849-849, 894-894, after 930, after 995, after 1000, after 1005, 1028-1029, 1049-1049, 1052-1052, 1056-1056, 1060-1060, 1064-1064, 1068-1068, 1091-1092, 1101-1101 | equal to engine | global-fix | owner bef189b1d, 2026-09-12 | about 1.1 percent on both cloud droplet fall speeds on a column that warms 4 K in a step | **offer** |

**`MORRISON-CU-1`.** WRF builds two quantities once per level inside its column loop and then spends them, unchanged, in the sedimentation block that runs after the loop closes and before the tendency apply: the cloud droplet Stokes coefficient, frozen above the warm branch's small melt, and the particle-size reference density. Nothing writes the temperature between the melt and the apply, so the density the sedimentation block reads is the one the process section's own reconstruction used. Both copies of this kernel rebuilt both quantities from the temperature they held at sedimentation time, which is the post-process one. The Stokes coefficient is the expensive half, minus 0.288 percent per kelvin at 278 K on every cloudy level; the density was stale by the entry cleanup and melt alone, about 8e-4 K and 2.4e-6 relative. The remedy is the one WRF's own structure names: the process stage publishes both and the sedimentation stage consumes them. It is NOT handing the sedimentation stage its current temperature, which would have moved the size distribution about 2,400 times further from WRF than the error it removes, in the wrong direction. Measured against the unmodified reference driver over the 28 oracle columns and 10,948 compared values, desktop, NVIDIA GeForce RTX 3080, 2026-09-12: values disagreeing with the reference fall from 3,554 to 3,505, no field gets worse, and every field's worst distance in units of the last place is unchanged. The engine carries the identical rebuild from the original port commit and is owed the offer. Not confined to a climate band: cloud droplet sedimentation touches every column that holds cloud water.

### `core/kernels/rrtmgp_rte.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `RTE-CU-1` | 355-370, 427-435, 448-448, 450-454 | 355-355, 412-414, after 426, 428-434 | equal to engine | global-fix | owner 50e109983, 2026-09-04 | up to 3.35 W/m2 of spurious absorption removed | **offer** |

**`RTE-CU-1`.** The device half of NPREF-9, and the only difference in this file: the engine's copy is the common ancestor byte for byte at 2.7.2, at 2.7.3 and through its next branch, with zero engine commits since the fork. The rearranged form agrees with the as-written form to 1e-8 relative in float64, matches float64 to 1e-3 W/m2 in every flux, and absorbs under 1e-4 W/m2 on a conservative cloud at optical depths 2, 10 and 30. Settled with no card, on the float64 mirror and the same arithmetic evaluated in float32. Offer it as a pair with the mirror.

### `core/kernels/glibc_flt32.cuh`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `LIBM-1` | after 0 | 1-41 | equal to carried | engine-fix | engine c0ffc53b5 and c706892ac, both 2026-09-04 | nothing runs differently | **refuse** |
| `LIBM-2` | 192-209 | 233-251 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing on its own | **refuse** |
| `LIBM-3` | 298-634, 637-650 | 340-517, 520-553 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | taking the engine's would move the shape factor on 68.17 percent of reachable tuning values | **refuse** |

**`LIBM-1`.** The engine put the third-party notice for this header inline. This package does not, and that is deliberate rather than an omission: the bytes of these device sources ARE the identity a run receipt records, because the loader assembles each module's compile string out of this directory and the receipt pins its digest, so a notice comment prepended to one of them would move that digest without moving a compiled image and every published receipt would stop matching the code that wrote it. The obligation is performed instead by `core/kernels/LICENSE-third-party.txt`, which sits beside the file, is named explicitly in the package data, and ships in the wheel and the sdist; the complete texts are in `licenses/` and the root `NOTICE` carries a section per work. Refuse the inline form; take any engine change to the notice's CONTENT into the file beside the code, which is what the gamma removal did: the engine put the record of it inside this header and this package put it in `core/kernels/LICENSE-third-party.txt` and the root `NOTICE`. This is the only row left in this file, because the two below it are taken.

**`LIBM-2`.** Three routines beneath the gamma block, carrying the reference library's exp2f, expm1f and the positive arm of its lgammaf, whose only caller is that block. The engine deleted them when it replaced its gamma; this package keeps its gamma, so it keeps them, and the Arm and FDLIBM notices the kernels notice performs cover them.

**`LIBM-3`.** The gamma itself: gfk_gamma_product, gfk_gammaf_positive and gfk_tgamma, this project's own work, against the engine's correctly rounded variant. The gamma this package carries is the one the global model was graded with; the engine's variant moves the cumulus mass-flux shape factor on 68.17 percent of all 53,687,093 reachable float32 tuning values and the deep mass flux by up to 7.3 percent on converged columns, and taking it is a numerics choice to grade against observations, not a fix to pull. Those two figures are the engine line's own measurements, recorded with its commit c0ffc53b5 on 2026-09-04: the shape-factor count from the header compiled unmodified as host C++ (gcc 13.3.0, `-ffp-contract=off`, x86-64 SSE2, no card) and enumerated over the whole reachable set rather than sampled, the mass-flux bound from the shipped kernel compiled the same way over the engine's 216-column WRF capture; that record names the compiler and the date and no host. Nothing in this package has re-measured either. The other cumulus scheme is untouched either way: it uses only the logarithm, exponential and power surface.

### `core/kernels/LICENSE-third-party.txt`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `KERNEL-NOTICE-1` | 15-30, 35-42, 49-57, 130-151, 167-171, 177-177, 179-191, 210-221 | 15-25, 30-36, 43-59, 132-155, 171-172, 178-179, 181-182, 201-207 | not present | adaptation | package cd407ae and 78cad51, both 2026-09-12 | nothing | **none** |

**`KERNEL-NOTICE-1`.** The notice beside the device sources, narrowed by rule to the files this package actually ships rather than inherited whole. The carve's own scope rules do the narrowing, so the remaining differences are the ones the rules do not cover: the reason the notice sits beside the code rather than inside it is this package's receipt digests and not the source tree's frozen-digest suites, the scope paragraph names one header instead of a list of files this package does not have, the McICA generator is filed under its own work, and the closing scope section says whose work the gamma routines are.
## PULL

Engine fixes this package should take. Each one lands on the model's own
source tree first and reaches the package by re-cut, because a file edited
only on the package side is not what a re-cut reproduces.

**Already taken, listed so nobody re-opens them.**

* The land surface scheme's frozen-ground infiltration limiter received the
  soil factor where it should have received that factor times the frozen
  coefficient, so the limiter did not limit. Engine fix 2026-09-04; taken here
  in owner commit `4b74fc7cf`, 2026-09-12, at the kernel's three call sites and
  the float64 mirror's two, and re-cut into 0.1.1. Against the unmodified
  reference driver the surface runoff distance fell from 60,641,303 units in
  the last place to 2,812 and three soil fields went to exactly bitwise.
* The microphysics bounding routine built its reference density from the entry
  temperature rather than the current one. Engine fix 2026-09-04; taken here in
  owner commit `bef189b1d`, 2026-09-12, with the engine's own text.
* The RTE+RRTMGP notice in the radiation driver and the mirror, and the WRF
  notice beside the parameter tables. Engine work 2026-09-04; performed here in
  package commits `cd407ae` and `78cad51`, 2026-09-12, together with the
  licence texts, the notice beside the kernels and a root `NOTICE` with a
  section per work.
**Open.** In rough order of what they cost to take:

1. `GF-CU-2`, one character, which moves the deep and the shallow cumulus
   closure at once and which no parity gate catches.
2. `NPREF-6` with `YSU-CU-1`, WRF's guard on the thermal-enhanced sweep, as
   three parts and applied on top of `NPREF-7` rather than as the engine's whole
   diff. Re-read `NPREF-5`'s grade afterwards.
3. `SFCLAY-PY-1` with `SFCLAY-CU-1` and the inventory half named in
   `INVENTORY-3`, as one three-file change or not at all.
4. `NPREF-11`, the shallow cumulus tendency divisor, which is bookkeeping and
   closes a flawed instrument.
5. `RRTMGP-3` and `RRTMGP-9`, each of which adds one engine module to the seam
   manifest; `RRTMGP-4`; `RRTMGP-14`, which has to be re-applied over the
   chunked loops; `RRTMGP-6`, as a targeted deletion of one entry and not a
   copy of the engine's now empty table.
6. `PHYSICS-1` with `INVENTORY-1` and `INVENTORY-2`'s names, together or
   neither; then `PHYSICS-2`, `PHYSICS-4`, `PHYSICS-5`, `PHYSICS-7` and
   `PHYSICS-9`, none of which moves a number this model produces and all of
   which close a wrong answer or a false refusal inside carried code.

Rows owed: `GF-CU-2`, `NPREF-6`, `YSU-CU-1`, `SFCLAY-PY-1`, `SFCLAY-CU-1`,
`NPREF-11`, `RRTMGP-3`, `RRTMGP-4`, `RRTMGP-6`, `RRTMGP-9`, `RRTMGP-14`,
`PHYSICS-1`, `INVENTORY-1`, `PHYSICS-2`, `PHYSICS-4`, `PHYSICS-5`,
`PHYSICS-7`, `PHYSICS-9`. That is the whole of what is owed; every other row
in this document is an offer, a refusal or a no-op.

## OFFER

Fixes and capability on this side that the engine line does not have. Nothing
here is owed to this package; the list is what to send.

* `MORRISON-CU-1` with `MORRISON-PY-1` and `NPREF-2`. Two defects the engine
  carries identically from the original port commit: the sedimentation stage
  rebuilt the Stokes coefficient and the particle-size reference density from a
  temperature the reference never reads there. Owner commit `bef189b1d`,
  2026-09-12.
* `RTE-CU-1` with `NPREF-9`, as a pair. The shortwave two-stream floor and the
  float32-safe rearrangement. Owner commit `50e109983`, 2026-09-04.
* `RRTMGP-10`, `RRTMGP-12` and `RRTMGP-13` with `NPREF-10`, and `RRTMGP-18`
  with `PHYSICS-3`, which travel together because the four carriers are
  declared on the driver's result type. The counted size bounding, the
  area-conserving ice and snow merge, the no-mass sentinel reconstruction and
  the broadband carriers. Owner commits `4bf9f3e94`, `9cfc20654`, `984c1cc61`
  and `eaa150a42`, 2026-09-04; `873d73d3b`, 2026-09-04; `f6999e233`,
  2026-09-05; `6464f501b`, 2026-09-06.
* `YSU-CU-2` with `NPREF-7`, and `NPREF-8`. The boundary-layer flag rule at both
  of the scheme's sweeps, without the read below the start of the column that
  the engine's own commit believes unreachable and that this line reproduced.
  Owner commit `304177d6f`, 2026-09-04.
* `LANDUSE-1`. A land cell of the ice class keeps it on the ice soil, with the
  two caveats in the row. Owner commit `50ac55898`, 2026-09-05.
* `NTIEDTKE-PY-2` and `NTIEDTKE-PY-3`. The cumulus pipeline cache key, which
  closes a latent defect the engine still has, and the per-column grid spacing,
  which the engine's own driver already publishes and its other cumulus scheme
  already reads. Owner commits `3d22c51cf` and `92e58c4ff`, 2026-09-02.
* `GF-PY-4`. The chunked cumulus seam, which the engine would want at the same
  batch size its own seam cannot allocate. Owner commit `a74805618`,
  2026-09-01.
* `GF-CU-8`. The per-column closure reading, if the engine wants the same
  instrument. Owner commits `fca73b086`, 2026-09-04, and `957059111` and
  `58fd47304`, 2026-09-05.
* `RRTMGP-15`, the k-distribution temperature floor, which is inert wherever
  air stays above it.
* `SFCLAY-PY-2` with `SFCLAY-CU-3` and `NPREF-4`, as a new option rather than
  an edit to an existing branch. Taking it is also how the engine's momentum
  friction velocity comes across here in one step, which is the whole of
  `PHYSICS-8`'s refusal.
* `RRTMGP-2` and `NPREF-1`. The McICA subcolumn generator is AER's work, not
  rte-rrtmgp's, and the engine's notices file it by its filename prefix.

## REFUSE

Positions this package holds. A future engine change to these lines is
re-applied around them or declined, never dropped in over them.

**Physics chosen on measurement.**

* `SFCLAY-PY-2`, `SFCLAY-CU-3`, `NPREF-4`: the vegetation-weighted thermal
  roughness, off by default, for an 18Z skin temperature 2.66 K above the
  reference over land with the pattern of the canopy. The engine's validator
  still admits only the three original options, so an unchanged engine edit
  would delete the door.
* `YSU-PY-1`, `YSU-CU-4`, `NPREF-5`, `YSU-CONTRACT-1`: the fixed
  free-atmosphere mixing length, off by default, for a diffusivity that on a
  40-level stack drained the 100 to 400 km kinetic energy at 237 hPa at 0.7 to
  1.2 per day. Refuse means re-apply into the default path, which is meant to
  be the reference bit for bit.
* `GF-CU-3`, `GF-CU-4`, `GF-CU-5`, `GF-CU-6`, `GF-CU-7`: the coarse-column deep
  cumulus arm behind two compile-time defines, both off by default, for a
  warm-pool column raining 84 mm/h through the microphysics while the cumulus
  scheme booked 0.06 mm/h.
* `LIBM-3`, `LIBM-2`, `GF-CU-1`, `GF-CU-9`, `GF-PY-1`, `GF-PY-2`: the gamma
  this package carries is its own work and the one the global model was
  graded with; the engine line's correctly rounded variant moves the cumulus
  mass-flux shape factor and is a numerics choice to grade against
  observations, not to pull. The three routines beneath it, the driver
  docstring and the override slots go with that decision.
* `RRTMGP-11`: the explicit-radius and P3 cloud couplings clip at the table
  bound on purpose, because their arithmetic is pinned to fixtures that already
  bound the oversize the older driver's way.
* `LIBM-1`: the third-party notice for the device header sits beside the code
  and not inside it, because these bytes are the identity a run receipt records.

**Shapes this package needs to run.**

* `GF-PY-3` and the chunk and closure plumbing it carries: an engine change to
  the module dispatcher lands on a wider signature.

**Engine changes that are right there and wrong here.**

* `PHYSICS-10`: the shared radiation factory builds the ENGINE's radiation,
  which is the physics the carve exists to replace, and the import succeeds so
  nothing would raise.
* `PHYSICS-8`: the driver half of the momentum friction velocity cannot be
  taken without the engine's surface-layer kernel, which would silently drop
  the graded closure above.
* `RRTMGP-8`: a module-scope registry check that would make every forecast die
  at import.
* `INVENTORY-4`: a rewritten pricing function with no consumer here that reads
  a registry this package does not ship.
* `LOADER-1`: a composition branch for a kernel name that cannot reach this
  loader.
