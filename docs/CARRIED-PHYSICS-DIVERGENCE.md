# The carried physics, and where it differs from the engine's

`src/arwen_global/core/` carries the physics this model was graded with: eleven
Python modules, the float64 mirror used by kernel scorecards to compare GPU
arithmetic (code verification), the CUDA
loader, fifteen kernel sources and the four WRF parameter tables. Every one of
them exists on the engine as well, and the two lines move independently. That
is not an accident to be tidied away; it is the arrangement. What it needs is
a file that says, for each place the two differ, what differs, why, and what
should happen the next time one side changes that code.

This is that file. `SOURCE.md` says which revision of the model's own source
tree the package was cut from. It does not say which differences are on
purpose, which is what this one is for.

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
`src/arwen_global/data/engine-divergence.json` carries all 852 of them with the
class and decision of the row they belong to, and the engine line keys its own
change lists by the engine-side hash, so the same difference has the same name
on both sides. `tools/fingerprint_engine_divergence.py` documents the
normalisation exactly and is what recomputes it.

## What this was measured against

* **Engine:** the published `gpuwm 2.8.5` from PyPI, which is what the
  declared range `gpuwm>=2.8.0,<2.9` resolves on 2026-10-05. Its carried
  files, read off the manylinux wheel, are byte-identical to the public
  `v2.8.5` tag (commit `3e2d2734a96a`), and the two comparisons in
  `tests/test_engine_divergence.py` run against 2.8.5 and skip, naming both
  versions, on any other. The re-baseline from 2.8.0 (2026-10-05) carried
  371 hunks forward unchanged, dropped 53 whose engine text moved, and read
  479 new ones. A new hunk that sits on the carried lines of an existing
  row, or carries that row's subject, keeps the row's name, class and
  decision, with a sentence added where the engine's movement matters to
  the decision; the rest are new rows, every one of them the engine moving
  alone on code this package runs from its own copy. The **Commit** cell
  of a 2.8.5 row names the public tag whose tree first holds the hunk's
  engine text (`v2.8.1` `d94f9ea`, `v2.8.2` `4c27749`, `v2.8.5` `3e2d273`;
  no hunk's text first appears at `v2.8.3` or `v2.8.4`), because each
  public tag is one snapshot commit. None of these hunks
  moves a number this package produces: the shipped experiments run the
  carried copy, and the engine's copy of a carried file is reached only by
  `load_trace_climatology`, unchanged since 2.8.0. Of what the re-baseline
  decides to pull, only the constant-division respelling (`SFCLAY-CU-4`
  and its four siblings) moves bits once it is taken, and then on
  Blackwell cards only. The longwave lid fix then added two hunks, measured on the same engine, filed as `RRTMGP-26`. The 2.8.0 baseline
  before it is in the git history of this file; it read 361 hunks forward
  from 2.7.3, dropped 14 and read 66 new ones.
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
any other. In the unified distribution the engine and this package come from
one commit, so the rows are re-baselined whenever the engine moves under them
and the two nodes run. The nodes that hold the rows and this page to each
other need no engine and always run.

## What does not differ at all

Four carried files and the four parameter tables are byte-identical to the
engine's once the carve's rewiring is applied, so they have no rows and an
engine change to any of them can be taken as it stands. That is a measurement
and not an assertion: the fingerprint tool reads every one of the 35 carried
files, the four `.TBL` parameter tables among them, and a file with no row is
a file the tool found nothing in rather than a file it skipped. Change one
byte of any of them, on either side, and the gate fails naming the file and
the lines, exactly as it does for a file that has rows. The list:
`core/kernels/ysu_validation.cu`, `core/kernels/rrtmgp_validation.cu`,
`core/kernels/common.cuh`, `core/kernels/rrtmgp_planck_common.cuh`, and
`data/noah_tables/{GENPARM,LANDUSE,SOILPARM,VEGPARM}.TBL` with their
`PROVENANCE.md` and `LICENSE-WRF.txt`. At 2.8.0 the list also held
`core/noah.py`, `core/kernels/noah.cu`, `core/kernels/ntiedtke.cu`,
`core/kernels/rrtmgp_gas.cu`, `core/kernels/rrtmgp_cloud.cu` and
`core/kernels/rrtmgp_mcica.cu`; the engine moved all six between 2.8.1 and
2.8.5, so each has a section below now.

Two of the 2.8.0 list were not identical a day before that and are worth naming, because
somebody reading an older note will look for them here. The land surface
scheme's kernel carried a substitution in its frozen-ground infiltration
limiter, which the engine fixed on 2026-09-04 and this line took on 2026-09-12
in owner commit `4b74fc7cf`; the carried kernel was byte-identical to the
engine's again at 2.8.0. And the microphysics bounding routine rebuilt a reference
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
| `RRTMGP-3` | after 61, 1986-1986, 2096-2096, 5051-5051 | 46-47, 1825-1825, 1970-1971, 4851-4851 | equal to carried | engine-fix | engine 86bd301f5, 2026-09-03 | nothing to any flux | **pull** |
| `RRTMGP-4` | 75-75, after 85, after 96 | 62-63, 74-105, 117-122 | equal to carried | engine-fix | engine ee3d2bac0 and 8279bdeeb, both 2026-09-10 | nothing | **pull** |
| `RRTMGP-5` | after 120, after 188, after 190, after 314, 458-458, 467-467, 2172-2193, 2361-2364, 2368-2368, 3579-3581 | 147-184, 253-262, 265-266, 391-417, 544-545, 554-554, 2045-2058, 2221-2293, 2297-2297, 3490-3509 | equal to carried | engine-fix | engine 4a1ed7f3c, 2026-09-10 | nothing at this package's settings | **none** |
| `RRTMGP-6` | 393-443 | 496-505 | equal to carried | engine-fix | engine 6a0713146, 2026-09-05, and 4a1ed7f3c, 2026-09-10 | nothing | **pull in part** |
| `RRTMGP-7` | 390-391 | 493-494 | equal to carried | no-behaviour | engine 71f2b7cd1, 2026-09-10 | nothing | **none** |
| `RRTMGP-8` | after 452 | 515-538 | equal to carried | engine-fix | engine 9d7e57f27, 2026-09-10 | nothing, and it raises at import | **refuse** |
| `RRTMGP-9` | 1720-1727, 1770-1789, after 3193, after 5176 | 1558-1560, 1603-1603, 3167-3175, 4970-4970 | equal to carried | engine-fix | engine a679e477b, 2026-09-05 | nothing for this package | **pull** |
| `RRTMGP-10` | 1091-1100, 1106-1552, 2125-2127, 2201-2202, 2236-2236, 2238-2239, 3145-3153, 3164-3168 | 1363-1363, after 1368, 2000-2000, after 2065, 2099-2099, 2101-2101, after 3126, after 3136 | equal to engine | global-fix | owner 4bf9f3e94, eaa150a42 and 873d73d3b, 2026-09-04; f6999e233, 2026-09-05; 6464f501b, 2026-09-06 | large for the microphysics this model runs | **offer** |
| `RRTMGP-11` | 2282-2288 | 2144-2148 | equal to engine | deliberate | owner eaa150a42, 2026-09-04 | nothing relative to the engine | **refuse** |
| `RRTMGP-12` | 2400-2404, 2417-2442, 2444-2445 | after 2328, after 2340, 2342-2343 | equal to engine | global-fix | owner 984c1cc61, 2026-09-04 | large where transport outruns the microphysics call | **offer** |
| `RRTMGP-13` | 2447-2451, 2453-2467 | 2345-2346, 2348-2354 | equal to engine | global-fix | owner 4bf9f3e94 and 9cfc20654, both 2026-09-04 | large for the frozen size | **offer** |
| `RRTMGP-14` | after 3173, after 3179, 3414-3415, 3418-3418, 3804-3806, 3936-3940 | 3142-3143, 3150-3152, 3394-3397, 3400-3400, 3689-3693, 3819-3853 | equal to carried | engine-fix | engine 267900003, 2026-09-04 | no flux moves; one refusal boundary tightens by a layer | **pull** |
| `RRTMGP-15` | 3430-3445 | 3412-3417 | equal to engine | deliberate | owner 42c37383c, 2026-08-31, restructured by eb34dfa19, 2026-09-05 | at most 1.77 percent of a 1 to 2 hPa layer's emission, where it binds | **offer** |
| `RRTMGP-16` | 52-52, 704-709, 713-713, 2948-2976, after 2992, 3021-3025, 3033-3034, after 3035, 3041-3045, 3052-3061, 3450-3451, 3457-3462, 3464-3523, 3538-3548, 3560-3561, after 3577, 3594-3597, 3607-3608, 3617-3620, after 3632, 3635-3711, 3720-3720, 3722-3791, 4058-4154 | after 34, after 784, 788-788, after 2965, 2982-2982, 3011-3021, after 3028, 3030-3031, 3037-3037, after 3043, 3422-3422, 3428-3429, 3431-3438, 3453-3454, 3466-3470, 3487-3488, 3522-3523, 3533-3534, 3543-3549, 3562-3563, 3566-3629, 3638-3652, 3654-3676, after 3992 | equal to engine | adaptation | owner eb34dfa19, 2026-09-05 | nothing | **none** |
| `RRTMGP-17` | 4268-4287, 4289-4310, 4315-4317, 4319-4319, 4325-4352, 4355-4356 | after 4105, 4107-4108, 4113-4114, 4116-4116, 4122-4136, 4139-4140 | equal to engine | adaptation | owner eb34dfa19, 2026-09-05 | nothing | **none** |
| `RRTMGP-18` | 4035-4036, 4365-4370 | 3962-3971, 4149-4149 | equal to engine | global-fix | owner 4bf9f3e94, 2026-09-04 | nothing inside a scheme | **offer** |
| `RRTMGP-19` | 5168-5175 | after 4968 | equal to engine | no-behaviour | owner 4bf9f3e94, 9cfc20654 and 984c1cc61, 2026-09-04; 6464f501b, 2026-09-06 | nothing | **none** |
| `RRTMGP-20` | after 2089, 3136-3137, 3208-3209, 5097-5097, 5166-5166, after 5177 | 1929-1963, 3118-3119, after 3189, 4897-4897, 4966-4967, 4972-4972 | neither | adaptation | engine 3ade28599, 2026-09-25; package 2026-09-29, on the move to the 2.8 engine | nothing: every value bit-identical | **none** |
| `RRTMGP-21` | after 71, after 2551 | 58-58, 2439-2567 | equal to carried | engine-fix | engine ed582aa45 and 74cd41bfa, both 2026-09-27 | nothing: this model's microphysics has no coupling in the radius table | **none** |
| `RRTMGP-23` | after 55, after 56, 610-613, 624-624, 627-627, 635-635, 639-639, 641-641, 646-646, 731-731, 741-741, 763-763, 898-898, 916-916, after 1021, 1914-1918, 1931-1932, 1964-1975, 2581-2581, 4695-4695 | 38-38, 40-40, 697-698, 709-709, 712-715, 723-724, 728-728, 730-733, 738-739, 806-806, 816-819, 841-842, 1094-1094, 1112-1114, 1250-1293, 1728-1730, 1743-1744, 1776-1814, 2597-2597, 4477-4477 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing numeric | **pull** |
| `RRTMGP-24` | 843-844, after 866, 876-882, after 894, after 959, after 1562, 1600-1608, 1610-1610, 1614-1614, 1629-1629, 1657-1657, 2652-2652, 3305-3305, 3817-3817, 3865-3865, 3896-3899, 3948-3949, 3972-3977, 3979-3979, 3993-3993, 4005-4010, 4019-4025, 4027-4027, 4456-4456, 4493-4500, 4526-4537, 4709-4730, 4732-4742, 4750-4753, 4757-4770, 4774-4781, 4788-4788, 4840-4849, 4852-4852, 4886-4886, 4929-4936, 4939-4939 | 922-923, 946-954, 964-1004, 1017-1090, 1158-1187, 1379-1399, 1437-1446, 1448-1448, 1452-1452, 1467-1467, 1495-1495, 2668-2670, 3285-3285, 3704-3704, 3752-3752, 3783-3784, 3861-3866, 3889-3891, 3893-3893, 3907-3907, 3919-3921, 3930-3952, 3954-3954, 4235-4235, 4272-4280, 4306-4319, 4491-4516, 4518-4532, after 4539, 4543-4543, after 4546, 4553-4553, 4605-4624, 4627-4634, 4668-4669, 4712-4729, 4732-4739 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing: the engine records every output word unchanged | **none** |
| `RRTMGP-25` | 3915-3916, 3918-3919 | 3800-3800, 3802-3802 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing at a fixed step | **none** |
| `RRTMGP-26` | 664-681, 693-693 | 757-762, 774-774 | equal to engine | global-fix | package 9df6bf6 and 9d4ea0f, 2026-10-05 | level-0 potential temperature drift -59.4 to -10.6 K/day (T255, GDAS 2026-09-01 00Z); binds only below a 2 hPa top | **offer** |

**`RRTMGP-1`.** Both copies now open with the same third-party notice. One word inside it differs, because this copy cites the reference commit from its docstring and the engine's cites it from its header. Take an engine rewording of the notice whenever it is convenient.

**`RRTMGP-2`.** The McICA subcolumn generator this radiation is driven with is WRF's RRTMG generator, which is AER's work under AER's own grant, not rte-rrtmgp's. The engine's copy of this file files all six rrtmgp_ sources under RTE+RRTMGP on the strength of the filename prefix, so it transcribes an AER work while naming no AER text. The paragraph added here is the correction, and it applies to the engine's copy unchanged.

**`RRTMGP-3`.** The shipped HDF5 is not thread safe, so every netCDF reader in the process takes one lock. This copy opens all four coefficient files outside it and carries no import of that lock's helper at all. The engine's record is three of eight concurrent runs killed at nine frames of fourteen. One condition rides with the pull: that lock's helper is an engine module this package does not carry, and it exists at the floor this package declares.

**`RRTMGP-4`.** The engine declared, in the module that opens them, the five NetCDF members radiation loads, and made the opener refuse a filename outside the set. This copy opens exactly those five, so the check can never fire on today's load path; what it buys is that a sixth member cannot be added to the load path without joining a set something checks.

**`RRTMGP-5`.** The engine turned one microphysics selector from a refusal into a forecast whose radii come from its own particle size distribution. This model radiates one microphysics scheme and its own door admits that scheme by name and refuses every other value, so the branch would be dead code here carrying a dependency on an engine module the seam does not pin. Take it only if this package ever admits a second microphysics scheme, and then take it whole: the constants, the bands arm, the table row, the docstring paragraph, the paths branch and the driver branch are one change, and a partial take publishes a scheme name the radii dispatch cannot serve.

**`RRTMGP-6`.** The deliberate-exclusion table, which the engine emptied in two steps. One of its two entries should go and the other should stay, which is why the decision reads pull in part. The entry for the P3 selector is a long refusal record that the coupling table three hundred lines above already contradicts: that table carries a row for the same selector on all three copies, and only a selector MISSING from it reaches the exclusion table, so the entry is unreachable and should be deleted. The other entry must stay, because the engine removed it only when it added the radii path this package declines above; delete it here and a scheme this package genuinely cannot radiate stops getting the sentence that says why and starts getting the generic add-a-row message, which is the one instruction that is wrong for a deliberate exclusion. So the table ends with one entry here, not zero.

**`RRTMGP-7`.** A comment citing the engine's other named-refusal table by its old name. The module it names is not carried, so either spelling is correct for some engine version and neither is checkable from here.

**`RRTMGP-8`.** A module-scope check that holds this module's two scheme tables equal to the engine registry's rows. Two reasons to refuse it, either sufficient. It imports a name the pinned engine does not define, and the call is at module scope while the global model constructs this radiation on every run, so a carried copy would make every forecast die at import. And on a newer engine it would fail by design, because by the two rows above these tables deliberately lack one row and deliberately keep one exclusion. Revisit only if the engine pin moves forward and this package publishes a registry of its own to be held against.

**`RRTMGP-9`.** The engine moved the gas names and the override validator into a shared module and additionally refuses an override naming a gas the SELECTED coefficient tables do not carry. The only override this package ever passes names a gas both tables carry, so the added refusal never fires; what it converts is a silently ignored override into a refusal. Like the lock above it reaches one more engine module, which exists at the floor this package declares. The engine also added a reader for one gas table's temperature span beside the names, for its initial-state perturbation; it rides with the move. At 2.8.1 to 2.8.5 the engine moved that reader beside the cloud tables' upload, so its hunk is counted under `RRTMGP-23` now and four remain here.

**`RRTMGP-10`.** Above the cloud tables' upper size bound the engine clips the particle size and leaves the path alone, so a 500 micrometre snow layer is radiated as 180 micrometre ice at 2.8 times its extinction, with no record that it happened. This copy scales the in-cloud path by bound over size and sets the size to the bound, which is the extinction the geometric limit gives, and it counts what it bounded. On the control checkpoints 74 percent of ice cells carrying 91 percent of the ice path sat above the bound. The bounds are read off the loaded tables rather than written as literals, and the record carries cloud-fraction-weighted shares beside the in-cloud ones because the in-cloud share overstates the radiative weight: cells at zero cloud fraction held 18 percent of the in-cloud ice path and none of the radiation. The silent clip is a defect on both lines and the record is how anyone finds out it is binding.

**`RRTMGP-11`.** The couplings for the explicit-radius schemes and for P3 pass the clip treatment on purpose, so above the upper bound they clip at the full path exactly as the engine does; only the counting is new. Their arithmetic is pinned to the older radiation driver's own size cap and snow discount, which already bound the oversize that driver's way, so the geometric carry on top discounted the snow twice and moved four pinned seams. Refuse a future engine change that made these two branches carry instead of clip, unless it also retires those fixtures. The carry belongs to the branch this model runs, which is its own coupling and not a transcription.

**`RRTMGP-12`.** The microphysics kernel writes a 25 micrometre sentinel into a species' radius where that species had no mass at its last update. Transport between two microphysics calls puts real mass in those cells, so at radiation time they carried the sentinel beside mass and radiated at 25 micrometres clipped to the table top, for a droplet population near 5 micrometres. Device counters on an early arm read 52 to 54 percent of the liquid cells in that state with about a fifth of the in-cloud liquid path. Such a cell now takes the radius its own moments give. The engine runs the same kernel with the same sentinel and the same gap between the two calls, so the defect is there unchanged.

**`RRTMGP-13`.** Two changes to the size at which cloud ice and snow enter the ice table. The merge weight moves from number to the mass-weighted harmonic mean, which is the size whose cross-section at the summed mass equals the summed cross-section; number weighting put the merged size at the cloud-ice radius wherever crystals outnumber flakes, which is nearly everywhere, and snow is four fifths of the frozen condensate on the control. Then each species enters at the solid-ice effective diameter its area per unit mass implies, because the table's size axis is defined on solid ice: a 100 kilogram per cubic metre snowflake read at its own diameter received one ninth of its extinction. The two densities are the microphysics scheme's own, the ones its effective radius is defined with, so this is a transcription correction and not a tuning choice.

**`RRTMGP-14`.** The engine composes the longwave and shortwave selectors independently. Five of the six sites are inert with both spectra on: both chunk loops run over every column, both flux pairs are allocated as before, and every flux and heating rate is bit-identical. The sixth moves a refusal even with both on: the above-model layer bound becomes the larger of the two spectra's counts instead of the longwave's alone, which at this package's default model top tightens the level ceiling by one. Take the tightening with it rather than keeping the old bound: the larger count is what the shortwave above-model column actually needs, so the old bound was under-counting, and the only door that can reach a level count that high is the explicit hybrid ladder. This is the one engine change that lands on lines this package rewrote, so the gating has to be re-applied over the chunked loops rather than merged onto them; refusing it buys a conflict on these lines at every future re-cut. At 2.8.1 the engine also stopped computing a dark column's shortwave (`RRTMGP-24`) on the lines this row gates, so the re-application covers that too.

**`RRTMGP-15`.** A 1 hPa model top reaches polar-night air below the k-distribution's own temperature domain, and the alternative to bounding is a refusal in the middle of a forecast. Layer and interface temperatures in the band between the garbage threshold and the table floor are fed to the tables at the floor; the prognostic state is untouched and below the garbage threshold still refuses. The trigger was measured at 159.27 K on the top interface at hour 4.3 of a run. Nothing changes above the floor, which is every column a lower model top reaches, so the engine can take it at no cost; whether it wants a floor at all is its own call rather than a defect on its side.

**`RRTMGP-16`.** The driver packs its inputs and prepares its clouds one column chunk at a time, the fused validator ORs every chunk into one flag word read once, and the record's path sums reduce once over the whole grid from the chunks' terms rather than as a sum of per-chunk sums, because the latter is not the whole-grid sum's bits in a record every checkpoint carries. The one route by which chunking could have moved bits is the subcolumn generator, and it does not: it seeds per column from that column's own bottom four pressures, never from a position in the packed array, so a chunk boundary cannot change a draw. What moves is the device peak, which is why this package carries the driver at all: the whole-grid preparation was the model's peak, and a ten-step probe at the largest truncation died inside the path routine at 19.9 GiB. This is the row that makes a future engine change to the driver body conflict. Anything the engine does between packing and the solver loops has to be re-expressed per chunk here rather than merged. At the published 2.8.0 the engine hunk this row covers also carries a guard that skips the uniform-top reduction inside a CUDA graph capture (engine e5102999c); this model never captures a step as a graph, so the guard cannot fire here.

**`RRTMGP-17`.** The heating rates form per chunk straight into their output arrays. The per-cell expression is unchanged and a column lands at the same index the whole-grid reshape produced; what goes away is the whole-grid difference, net-flux and convergence temporaries. A future engine edit to the flux-to-tendency mapping has to be applied inside the chunk loop.

**`RRTMGP-18`.** Four optional fields appear on the result: the upward and downward shortwave at the top of the column, the upward longwave at the surface as the longwave solver formed it from skin temperature and band emissivities, and the column cloud cover under the maximum-random overlap the subcolumn generator samples. Nothing moves inside a scheme; what moves is what can be read out of one, and before this no top-of-atmosphere or surface radiation budget could be formed from what the result carried. The offer carries a condition: the four fields are declared on the carried physics driver's result type and the engine's ends one field earlier, so the radiation module and the result declaration travel together or neither does. At 2.8.1 the engine's result gained the surface direct and diffuse shortwave beside these fields, on the same lines.

**`RRTMGP-19`.** The exported-name list gained the public names of the rows above it. It follows them and is not a decision of its own.

**`RRTMGP-20`.** The carried driver reads its trace-gas global means and ozone profile through the engine's table loader, `gpuwm.core.rrtmgp.load_trace_climatology`, and the RFMIP clear-sky oracle fetches its input file through the engine's pinned route, `gpuwm.core.rfmip_upstream.fetch_rfmip`, instead of opening it from the companion. Until this package moved onto the 2.8 engine it shipped its own copy of the 136-number table and a loader for it, because the engines it could install beside still opened the NetCDF; the engine made the same change at 2.8.0, so the copy went. What remains is shape: the engine defines and exports the loader and this copy imports it, and the oracle's docstring says since when the file is not shipped. Every value is the one the ten-step before/after run read on 2026-09-26, pinned by the same SHA-256.

**`RRTMGP-21`.** The engine radiates a cloudy layer whose liquid radius is the microphysics' no-cloud background at WRF's cloudy-layer radius (10.5 um over water, 7.5 um over land, and the Kristjansson-Mitchell table for ice), after the boundary-layer cloud merge, for the schemes in its radius table. This model radiates one microphysics scheme, which is not in that table, and runs no boundary layer that supplies subgrid cloud, so neither branch can fire here. The engine's radius block also sits inside the region `RRTMGP-16` restructured per chunk, and that hunk is counted under `RRTMGP-16`; take it only with a scheme that has a radius coupling, and then re-express it per chunk.

**`RRTMGP-23`.** The engine keys every table radiation uploads (the gas and cloud tables, the solar source, the trace-gas and longwave climatology rows, the above-model rows, the subcolumn jump tables) and every named scratch buffer by device and by stream, and fills each one once under a lock (`cached_ready`, `_upload_once`). Its record: slabs of one domain stepping on their own streams each uploaded a copy at the first radiation call, the last to finish replaced the others in the cache, and a slab holding a replaced copy kept launching kernels that read it. This copy keys several of these by device already, none by stream and none under a lock, so the race is latent here while one process steps one band at a time on a card, and live the moment bands share a process across streams or cards. Nothing numeric moves. One condition rides with the pull: the helpers live in `gpuwm.core.device_cache`, which the engine first ships at 2.8.1, above this package's 2.8.0 floor, so taking it raises the floor or carries that module. Take it with `LOADER-3`. The temperature-span reader `RRTMGP-9` names now sits inside one of this row's hunks, beside the cloud tables' upload.

**`RRTMGP-24`.** The host half of the engine's radiation speed work: a dark column's shortwave is never computed rather than computed and zeroed, the dry-air column amount forms inside the gas kernel, the clear above-model tails and the profile appends form in one launch each, model fluxes are stored straight into their final views, the subcolumn jump table is stored bit-major, and the two-stream solvers fold a buffered row of g-points per column with a tile chosen without querying the card. Each host change pairs with a kernel change (`GAS-CU-1`, `CLOUD-CU-1`, `MCICA-CU-1`, `RTE-CU-3`), and the engine records every one as producing the same output words. This driver runs per column chunk (`RRTMGP-16`), so the host halves would have to be re-expressed per chunk rather than merged. Speed only, not owed; take a host half and its kernel half together or neither. The one-word comment row `RRTMGP-22` sat in the tile function this rewrite replaced, so it is part of this row now.

**`RRTMGP-25`.** The half-interval hour offset of the shortwave geometry reads `_physics_period_seconds(minutes, cfg)` instead of `_physics_interval_seconds(minutes, _model_clock_dt(cfg))`. The new helper returns the old expression unless the configuration sets `use_adaptive_time_step`, which the namespace this package's native suite hands radiation does not carry, so every value is the old one. It needs the helper `PHYSICS-19` adds to the driver; take it only with an adaptive step.

**`RRTMGP-26`.** WRF's longwave puts `nint(p_top*.01/4)` 4 hPa layers above the model top (module_ra_rrtmg_lw.F:11565,12998-13001), and below a 2 hPa top that rounds to none; WRF then moves the model's own top interface to zero pressure (`plev(ncol,nlayers+1) = 0.00`, :12344), so the air above the top is still radiated, merged into the top model layer. Both copies transcribed the count and not the moved interface, so a 1 hPa top took no longwave layer at all: the top model layer (1.0 to 1.4 hPa on the default 40-level stack) saw no downward longwave from the air above it and cooled to space. This copy keeps the model's top interface, because its heating is formed on the model's own layer thicknesses, and gives the longwave the shortwave's count when the 4 hPa rule gives fewer: one layer from the model top to the coefficient floor, the shape the shortwave already uses. Measured on the T255 native day from GDAS 2026-09-01 00Z, the level-0 potential temperature drift went from -59.4 to -10.6 K a day. It binds only for a model top under 2 hPa, which no regional top reaches, so the engine's regional runs are unaffected; its global line, and any regional run with such a top, has the same gap. With this row the longwave count is never below the shortwave's, so the bound `RRTMGP-14` would take (the larger of the two spectra's counts) equals the longwave count here.

### `core/gf.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `GF-PY-1` | 5-6 | 5-12 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing by itself | **refuse** |
| `GF-PY-2` | after 81 | 88-96 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing on the shipped path | **refuse** |
| `GF-PY-3` | 87-92, 94-128, 134-157 | 102-102, 104-111, 117-117 | equal to engine | adaptation | owner fca73b086, 2026-09-04; 957059111, 2090ac84a, e30f9fda6, c4a95bc61 and 58fd47304, 2026-09-05 | nothing at the shipped defaults | **refuse** |
| `GF-PY-4` | 209-229, 267-270, 272-308, 352-356, after 362, 365-365, 369-369, 374-393, 399-403, 407-407, 413-419, 422-433, 435-486, 489-496, 502-521, 526-528 | after 267, 305-305, after 306, 350-365, 372-372, 375-375, 379-379, 384-402, 408-418, 422-422, 428-441, 444-450, 452-458, 461-469, 475-479, 484-485 | equal to engine | adaptation | owner a74805618, 2026-09-01 | nothing | **offer** |
| `GF-PY-5` | 338-338 | 336-336 | equal to carried | no-behaviour | engine 9a2f350e7, 2026-09-14 | nothing, docstring only | **none** |
| `GF-PY-6` | 345-345, 347-348 | 343-344, 346-346 | equal to carried | engine-fix | engine v2.8.5 (3e2d273), 2026-10-03 | nothing at this package's settings | **none** |
| `GF-PY-7` | 189-197 | 149-256 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing: the engine records every output word unchanged | **none** |

**`GF-PY-1`.** The docstring of the cumulus driver. On the engine it dates and describes the engine's own gamma replacement; here it describes the gamma this package runs. No number moves. Refused with GF-CU-1, whose position it restates.

**`GF-PY-2`.** Three per-column scalar slots at the engine's driver entry point that let a reference capture's shape factor be pinned in place of the computed one; the engine's shipped path zeroes all three. They exist so the engine can audit its own gamma replacement against a capture. This package computes its shape factor with the gamma it was graded with and has nothing to audit that way, so the slots are refused with GF-CU-9, the kernel half of the same hatch.

**`GF-PY-3`.** The per-column closure reading's names, and the plumbing that compiles the kernel with integer defines for the two coarse-column switches. Both switches default off, so a default run compiles the engine's own text. An engine change to the module dispatcher lands on the wider signature and has to be re-applied rather than copied over. At 2.8.5 the engine's module dispatcher also took a stochastic-perturbation argument (`GF-PY-6`) on the signature this row widens.

**`GF-PY-4`.** The seam packs and allocates one column chunk at a time. One thread still owns one complete column, the kernel never reads across columns, and the per-launch slices are contiguous views indexed exactly as before, so a column's answer does not depend on which chunk carried it. The engine already tiles the LAUNCH; what this adds is an outer loop that also bounds the ALLOCATION, and an absent driver lane stops materialising a full-batch zero array. Measured on the largest global batch, 1,283,202 columns by 40 levels on a 32 GB card, the whole-batch input block was 3.07 GB and the output block 3.28 GB and the output allocation failed with 32.2 GB already live; at the shipped chunk the two blocks are 0.65 GB. The engine's seam still builds the whole batch in one block and would hit the same wall at the same size, so it is worth offering back. Between 2.8.1 and 2.8.5 the engine's own seam took the stochastic pattern lanes, empty rather than zeroed output blocks and a view-based packing on these lines; it still builds the whole batch in one block, so the offer stands.

**`GF-PY-5`.** One docstring word from the engine's 2.7.5 word sweep. Nothing to decide.

**`GF-PY-6`.** The engine's stochastically perturbed parameterisation for this scheme: four pattern channels clipped to [-1, 1] and a specialised kernel compiled through `gpuwm.core.spp_kernel_sources`. This package runs no stochastic physics and its door has no option that would reach it.

**`GF-PY-7`.** The tile now assumes eight resident blocks per SM because the driver kernel carries launch bounds (`GF-CU-12`), measured by the engine at 10.85 ms against 11.53 per call on an 82-SM card with every output word unchanged; and a provenance type lets the engine's driver read this scheme's fresh outputs without a validation copy. Speed only. The carried cumulus scheme next door sizes its own tile from these constants, so the tile constant cannot be taken without the launch bound, or without checking the other scheme's workspace.

### `core/ntiedtke.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NTIEDTKE-PY-1` | 1278-1278, 1281-1293, 1494-1495 | 1395-1395, after 1397, after 1564 | equal to engine | adaptation | owner 92e58c4ff, 2026-09-02 | nothing at the shipped default | **none** |
| `NTIEDTKE-PY-2` | after 1087, 1352-1366, 1392-1393 | 1110-1124, 1456-1456, 1482-1489 | equal to engine | global-fix | owner 3d22c51cf, 2026-09-02; engine 5a6053b7a, 2026-09-27 | nothing, on either line | **none** |
| `NTIEDTKE-PY-3` | 1452-1480, after 1496, 1515-1518 | 1549-1551, 1566-1568, after 1588 | equal to engine | global-fix | owner 92e58c4ff, 2026-09-02 | real when the lane is bound, nothing when it is not | **offer** |
| `NTIEDTKE-PY-4` | after 62, after 225, 243-243, 250-250, 581-602, 604-607, after 668, after 723, after 754, 757-757, 764-767, 846-846, 857-857, 863-863, 870-870, 879-879, 886-886, 894-894, 901-901, 908-908, 917-917, 924-924, 932-932, 938-938, 946-946, 953-953, 959-959, 965-965, 971-971, 978-978, 987-987, 995-995, 1001-1002, 1004-1004, after 1062, 1093-1093, after 1104, after 1105, after 1109, after 1117, 1151-1151, 1169-1170, after 1172, after 1174, 1184-1188, after 1241, 1441-1441, 1444-1444, after 1497, after 1523 | 63-74, 238-239, 257-257, 264-267, 598-606, 608-613, 675-676, 732-733, 765-767, 770-772, 779-786, 865-865, 876-876, 882-882, 889-889, 898-898, 905-905, 913-913, 920-920, 927-927, 936-936, 943-943, 951-951, 957-957, 965-965, 972-972, 978-978, 984-984, 990-990, 997-997, 1006-1006, 1014-1014, 1020-1021, 1023-1023, 1082-1084, 1130-1133, 1145-1152, 1154-1154, 1159-1159, 1168-1169, 1203-1223, after 1240, 1243-1262, 1265-1271, after 1280, 1334-1358, 1537-1538, 1541-1541, 1570-1571, 1594-1594 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing at this package's settings | **none** |

**`NTIEDTKE-PY-1`.** The cumulus workspace is bounded by one column-chunk option, whichever scheme fills the slot. With no cap the tile width is the engine's exactly, from constants equal on both sides. With a cap the domain is walked in more and narrower chunks; every stage is per column, so tendencies do not see the partition, except one launch scalar that is an OR over the chunk's columns, and the pipeline re-checks that hoist's precondition at the partition's own scope and refuses rather than passing a wrong scalar. A future engine change to the constructor signature or to the chunk walk needs these two regions re-applied, not dropped.

**`NTIEDTKE-PY-2`.** The engine keyed its one reusable pipeline on the column and level counts and read the step only at construction, so a carrier that varied its step silently got the previous call's step. This copy keys on the step and the closure flag as well, so a changed step builds a new pipeline. The engine closed the same defect its own way on 2026-09-27: it re-times the reused pipeline to each call's step. Both lines are now correct and nothing is owed either way, which is why the offer is withdrawn.

**`NTIEDTKE-PY-3`.** The engine fills the spacing array with one scalar for every column; this copy reads the driver's per-column spacing when it is present and falls back to the identical scalar fill otherwise. The spacing reaches the kernel at exactly one site, the scale factors, which multiply the deep closure's adjustment time and divide the shallow closure's mass flux. On a Gaussian grid the square root of the cell area at 60 degrees is about 0.707 of the equatorial value, so for a 50 km equatorial spacing the deep factor goes 1.665 to 1.470, about 13 percent on the adjustment time. Keep it here: the native suite runs on a grid whose zonal spacing shrinks with the cosine of latitude and this closure is scale aware. Offer it: the engine's own driver already publishes that lane and its other cumulus scheme already reads it, so this is the one scheme of the two that ignores a lane the engine already has. At 2.8.1 the engine fills its scalar spacing once for every chunk (`NTIEDTKE-PY-4`); this copy still hands each column its own.

**`NTIEDTKE-PY-4`.** The engine's speed work on this scheme's driver: the per-chunk validation flag is reduced on the device into a mask the kernels read as a guard instead of a host synchronisation per chunk (`NTIEDTKE-CU-1` is the kernel half), eleven stages run as one launch above a column threshold through `gpuwm.core.ntiedtke_fused`, argument bindings are cached per stage, and the tile is capped at 512 columns per SM instead of borrowing the other cumulus scheme's constants. The fused path is an engine module this package does not carry and has not graded. This model's default cumulus scheme is the other one; take this only with a measurement on this package's own columns.

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
| `YSU-PY-1` | 33-33, 91-91, 93-99, 150-151, 189-189 | after 32, 98-102, after 103, 171-212, after 250 | equal to engine | deliberate | owner 36cb7cd43, 2026-09-05; 8e6bb6aed, the same day, set the default and does not touch this file | nothing at the shipped default | **refuse** |
| `YSU-PY-2` | after 85, after 105, 162-162, 191-191 | 85-92, 110-126, 223-224, 252-253 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing at this package's settings | **none** |

**`YSU-PY-1`.** The launcher half of the free-atmosphere mixing-length switch: a name resolved through a two-entry table that refuses anything else, passed to the kernel as an integer. The default resolves to the flag the kernel line does not execute. Nothing else moves: validation, output allocation, tiling and workspace pricing are the engine's. At 2.8.2 the engine's urban and topographic-wind arguments (`YSU-PY-2`) joined the same signature.

**`YSU-PY-2`.** Argument checks and the choice between three kernel entries for the engine's urban canopy arm (`sf_urban_physics` 2 or 3) and its sub-grid terrain wind (`topo_wind`). This package runs neither; with neither, the engine launches the same `ysu_column` entry as before. `YSU-CU-5` is the kernel half. These lines sit on the signature `YSU-PY-1` widened, so a later take re-applies that switch beside them.

### `core/ysu_contract.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `YSU-CONTRACT-1` | 1-58 | absent | not present | deliberate | owner 36cb7cd43 and 8e6bb6aed, both 2026-09-05 | nothing at the default | **refuse** |

**`YSU-CONTRACT-1`.** A module the engine has never had: a two-entry name-to-flag table, the fixed mode's 30 m constant, and a refusal for any other name. The default name is WRF's own thickness-scaled rule, bit for bit. It exists because on a 40-level global stack whose jet-level layers are about 1500 m thick that rule reads a 150 m asymptotic length where a 300 m regional layer reads 30 m, and the energy ledger found the resulting diffusivity draining the 100 to 400 km kinetic energy at 237 hPa at 0.7 to 1.2 per day. Nothing to pull; keep both modes.

### `core/landuse.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `LANDUSE-1` | 316-326 | after 315 | equal to engine | global-fix | owner 50ac55898, 2026-09-05; comment updated 2026-09-29 for the soil match taken from engine b69ac2b52 | large on ice sheets and glaciers | **offer** |
| `LANDUSE-2` | after 384, after 434 | 374-374, 425-431 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing at this package's settings | **none** |

**`LANDUSE-1`.** Two lines that put a land cell of the ice vegetation class on the ice soil class after the soil match and before the land mask and soil reconciliation. Before the engine's soil match (taken here, see PULL) the reconciliation converted any land column on the water soil category to mixed forest on silty clay loam; since the match, such a column keeps its land use and takes silty clay loam, and these two lines still put a glacier on the ice soil instead. The measurement below was taken before the match. Measured on the desktop with both copies of the same entry point on the same input, a land cell of the ice class on the water soil texture under 120 kg per square metre of snow: the engine gives albedo 0.2968, roughness 0.20 m, emissivity 0.93 and moisture availability 0.60; this copy gives 0.70, 0.001 m, 0.95 and 0.95. Second effect: the carried land-surface driver skips ice-class columns, so after the fix the column leaves that scheme entirely and is run by the frozen surface path instead of being integrated as a forest. On the global statics at this model's working truncation, 30,233 Antarctic columns were affected. Two caveats for the engine, and they are why this is an offer rather than a defect on its side: WRF itself has no land-ice soil rule, so this is a physics choice rather than a closer transcription; and the rule fires on every ice-class land cell, so an engine that wants the fix without the width should condition it on the soil reading the water category.

**`LANDUSE-2`.** Under an urban model the engine keeps WRF's eleven Local Climate Zone categories and stops refusing them as land use the vegetation table cannot index, because the land surface then runs those columns as natural. This package runs no urban model.

### `core/physics_inventory.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `INVENTORY-1` | after 79, after 83, 88-88 | 80-130, 135-138, 143-143 | equal to carried | engine-fix | engine d4ec65c0a, 2026-09-04; 25cc6013a, 2026-09-28 | nothing in this package | **pull** |
| `INVENTORY-2` | after 113 | 169-188 | equal to carried | engine-fix | engine 8156dd58f and d4ec65c0a, both 2026-09-04 | nothing in this package | **none** |
| `INVENTORY-3` | 182-182 | 257-260 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing in this package today | **none** |
| `INVENTORY-4` | 22-23, after 218, 237-269, 277-296 | 22-23, 416-499, 518-527, 535-535 | equal to carried | engine-fix | engine 9d7e57f27 and ecfadd2a9, both 2026-09-10 | nothing in this package | **refuse** |
| `INVENTORY-5` | after 214, 361-361 | 293-411, 617-624 | equal to carried | engine-fix | engine 2e3e315d5, 2026-09-28 | nothing in this package | **none** |
| `INVENTORY-6` | 339-339, after 359, 364-364, after 371 | 578-583, 604-615, 627-630, 638-699 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing in this package | **none** |

**`INVENTORY-1`.** A one-conjunct correction to the predicate that says whether a boundary-layer scheme retains a raw output dictionary. It claimed one scheme retains a dictionary it never creates, which made the engine's memory estimator price buffers that do not exist. The scheme is not in this package's admitted set and the sole carried caller is inside the path a run with that scheme never enters. Taking it costs one conjunct and needs no new import. Do not take the commit whole: its other half edits the driver to merge the relocation buffers of the row below. At the published 2.8.0 the engine narrowed the predicate again, to the one scheme that creates the dictionary (`YSU_PBL_SCHEME`, engine 25cc6013a); this model runs that scheme, so the carried predicate answers the same for every run it admits, and the pull is the constant plus one comparison.

**`INVENTORY-2`.** Three names for recoupling held tendencies after terrain and base state are transplanted under a moving grid. This model integrates one fixed global grid that does not move, so there is nothing for them to attach to and the names alone would be inert; the behaviour lives in the driver and the restart path, and neither of those is carried. A note for a future re-cut: the engine did not add the three names to this module's export list, which is byte-identical on both sides, so a pull inherits that asymmetry.

**`INVENTORY-3`.** The momentum friction velocity enters the surface-layer output set at index 2. The name is not a number: it is the allocation key, the result type's keyword set, and the POSITIONAL tail of the kernel argument tuple. Inserting it here alone shifts 32 device pointers by one slot against a kernel signature with no such parameter, though in practice it would stop earlier on an unexpected keyword. This package's surface-layer set is internally coherent without the field and the engine's is coherent with it, so neither side is half done. The rule to carry forward: this name can never be pulled on its own. It moves as a set with the surface-layer module, its kernel and the mirror's tuple, and the trigger to do so is this model adopting the engine's two-dimensional mixing or needing the field on output or across a restart.

**`INVENTORY-4`.** The specified-zone ring guard's slot pricing, rewritten to derive from a registry's consumer rows. On the engine it fixes a real under-pricing for a nested run under one microphysics selector. Here the function has no consumer at all: it is exported and called from nowhere, it prices a nested-domain mechanism, and this model has no nest and runs one microphysics scheme. The rewritten body reads a registry name the pinned engine does not define and needs rows in a registry JSON this package does not ship, so taking it converts a dead but correct function into a dead function that raises if anyone ever calls it. The import sits inside the function body, so a re-cut that took this hunk would break nothing at import time and would be easy to miss, which is exactly why it is written down. Revisit if the engine pin moves and this package gains a nested-domain path.

**`INVENTORY-5`.** The engine names the Eta surface layer's and MYJ's persistent fields in tuples so its VRAM estimate prices the arrays the physics allocates. This package admits neither scheme and has no estimate that reads these tuples. Its driver half is `PHYSICS-14`.

**`INVENTORY-6`.** Field inventories and memory-pricing constants for schemes this package does not admit (the UW boundary layer, the `shinhong` boundary layer's workspace, the reflectivity producers, the terrain drag), the exported-name list that follows them, a comment on a boundary-layer tile setting, and the scheme dispatch table, which the engine moved here out of the driver (`PHYSICS-20` is the other half of that move). Nothing here is read by a door of this package.

At 2.8.1 to 2.8.5 the engine added inventories next to `INVENTORY-1`, `INVENTORY-2`, `INVENTORY-4` and `INVENTORY-5` (`INVENTORY-6`), so those rows' engine lines grew; what each describes is unchanged.

### `core/physics.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `PHYSICS-1` | 40-42, after 52, 1910-1911, after 2677, after 2690, 3046-3050, 3414-3414 | 40-46, 60-62, 1975-1991, 2862-2901, 2915-2919, 3267-3286, 3859-3865 | equal to carried | engine-fix | engine 1467ad9d1 and 8156dd58f, both 2026-09-04 | nothing on a fixed grid | **pull** |
| `PHYSICS-2` | 688-688 | 660-660 | equal to carried | engine-fix | engine db9f5d5bf, 2026-09-05 | nothing here today | **pull** |
| `PHYSICS-3` | 1164-1176 | 1141-1149 | equal to engine | global-fix | owner 4bf9f3e94, 2026-09-04 | nothing inside a scheme | **offer** |
| `PHYSICS-4` | after 1344, 1356-1356, 1363-1363, 1370-1370, 1377-1384, 1394-1395, 1879-1879, 3661-3662, 4251-4251, 4562-4630 | 1318-1318, 1330-1330, 1337-1338, 1345-1348, 1355-1357, 1367-1367, 1943-1944, 4240-4241, 4848-4853, 5320-5322 | equal to carried | engine-fix | engine d4ec65c0a, 2026-09-04 | nothing reachable from this package's doors | **pull** |
| `PHYSICS-5` | after 1762, after 1773, 4744-4744, 4793-4793, after 5352, after 5445 | 1793-1796, 1820-1836, 5538-5542, 5591-5591, 6212-6213, 6384-6386 | equal to carried | engine-fix | engine 007731bbe and 4d46b0aa0, both 2026-09-05 | nothing when no adapter is attached | **pull** |
| `PHYSICS-6` | 1784-1784, 1791-1791, 5048-5048 | 1847-1847, 1854-1854, 5874-5874 | equal to carried | no-behaviour | engine aeddfc47f, 2026-09-10 | nothing | **none** |
| `PHYSICS-7` | after 1980, 4309-4314, 4414-4416 | 2067-2083, 4939-4982, 5164-5174 | equal to carried | engine-fix | engine 399d95a86 and d7c5a9eca, both 2026-09-02 | zero on a fixed step, by construction | **pull** |
| `PHYSICS-8` | 2914-2929, 5051-5054, 5088-5097 | 3149-3149, 5877-5881, after 5918 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing reachable here | **refuse** |
| `PHYSICS-9` | after 4740, 5217-5217, after 5245, 5258-5258, after 5259 | 5534-5534, 6059-6059, 6088-6088, 6116-6117, 6119-6119 | equal to carried | engine-fix | engine da2e1dd8f, 2026-09-04 | nothing reachable here, and a real wrong answer inside the carried text | **pull** |
| `PHYSICS-10` | 4821-4868 | 5629-5631 | equal to carried | engine-fix | engine 267900003, 2026-09-04 | nothing here, and it would silently change whose physics runs | **refuse** |
| `PHYSICS-11` | 4902-4903 | 5665-5706 | equal to carried | engine-fix | engine b61739b71, 2026-09-10 | nothing in this package | **none** |
| `PHYSICS-12` | 4921-4921, 4923-4926 | 5724-5724, 5726-5733 | equal to carried | engine-fix | engine aeddfc47f, 2026-09-10 | nothing in this package | **none** |
| `PHYSICS-13` | 159-162, 567-567, 2001-2001, 2016-2016, 4013-4013, 4063-4063, 4082-4082, 4637-4637 | 171-173, 532-532, 2104-2104, 2119-2119, 4610-4610, 4660-4660, 4679-4679, 5329-5329 | equal to carried | no-behaviour | engine 9a2f350e7, 2026-09-14 | nothing, comments only | **none** |
| `PHYSICS-14` | after 51, 5070-5073, 5075-5083, 5171-5173 | 57-58, 5897-5913, after 5914, 5992-6015 | equal to carried | engine-fix | engine 2e3e315d5, 2026-09-28 | nothing in this package | **none** |
| `PHYSICS-15` | after 186, 222-224, 2140-2141, after 2143 | 198-204, 240-240, 2247-2257, 2260-2265 | equal to carried | engine-fix | engine 5a6053b7a, 2026-09-27 | nothing in this package | **none** |
| `PHYSICS-16` | after 421, 914-915, 923-923, 3836-3836, 3900-3900, 3928-3928, 3961-3962, 3964-3966, 3979-3981 | 377-397, 889-892, 900-900, 4415-4421, 4485-4486, 4514-4515, 4548-4549, 4551-4556, 4569-4578 | equal to carried | engine-fix | engine 758c353d3, 2026-09-18, and fd86b13f2, 2026-09-27 | nothing in this package | **none** |
| `PHYSICS-17` | 3234-3236, after 3246, after 3252 | 3625-3632, 3643-3644, 3651-3652 | equal to carried | engine-fix | engine 9bcf423a4, 2026-09-14 | nothing in this package | **none** |
| `PHYSICS-18` | after 4340 | 5011-5039 | equal to carried | engine-fix | engine 7f684beb7, 2026-09-18 | nothing on this model's path; closes a false refusal inside carried code | **pull** |
| `PHYSICS-19` | after 672, 716-716, after 717, after 1695, after 1747, after 1765, after 1794, after 1952, 2038-2039, after 2319, after 2321, after 2381, after 2582, after 2631, after 2846, after 2880, after 3025, after 3052, 3099-3100, 3125-3126, after 3149, 3155-3157, 3188-3188, 3231-3231, after 3304, after 3341, after 3367, 3383-3383, 3442-3442, after 3444, after 3449, after 3453, after 3656, after 4281, after 4286, 4319-4319, after 4357, after 4379, after 4383, after 4384, after 4732, after 4795, 4979-4981, after 4997, after 5017, after 5247, 5291-5293, 5391-5391 | 638-644, 688-689, 691-692, 1712-1717, 1770-1777, 1800-1811, 1858-1858, 2033-2038, 2141-2146, 2442-2448, 2451-2463, 2524-2528, 2736-2753, 2803-2815, 3076-3078, 3113-3115, 3246-3246, 3289-3325, 3372-3390, 3416-3426, 3450-3518, 3524-3536, 3567-3572, 3615-3622, 3705-3712, 3750-3761, 3788-3799, 3815-3827, 3893-3893, 3896-3898, 3904-3907, 3912-3915, 4121-4235, 4884-4905, 4911-4916, 4987-4989, 5057-5077, 5100-5110, 5115-5124, 5126-5134, 5425-5525, 5594-5603, 5786-5790, 5807-5821, 5842-5843, 6091-6105, 6151-6152, 6252-6329 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, v2.8.2 (4c27749), 2026-10-02, and v2.8.5 (3e2d273), 2026-10-03 | nothing in this package | **none** |
| `PHYSICS-20` | after 45, 61-61, after 108, 327-332, 359-413, 470-480, after 1422, after 1490, after 1549, 2411-2416, 2481-2482, 3114-3114, 3406-3406, 3467-3468, 3645-3645 | 50-50, 71-72, 120-120, after 342, after 368, after 445, 1395-1400, 1469-1485, 1545-1565, 2558-2569, 2634-2635, 3404-3405, 3850-3851, 3929-3931, 4108-4109 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, v2.8.2 (4c27749), 2026-10-02, and v2.8.5 (3e2d273), 2026-10-03 | nothing in this package | **none** |
| `PHYSICS-21` | 208-208, 287-287, 290-291, 1857-1858 | 226-226, 303-303, 306-307, 1921-1922 | equal to carried | no-behaviour | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing, comment only | **none** |

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

**`PHYSICS-13`.** Comment and docstring words from the engine's 2.7.5 word sweep, plus one sentence in the physics-constant note that names the engine's own module where this copy names the module it is carried from. Nothing to decide.

**`PHYSICS-14`.** The driver half of `INVENTORY-5`: the MYJ and Eta surface fields are allocated from the priced tuples instead of literal name lists. Same fields, same cold starts. This model runs neither scheme.

**`PHYSICS-15`.** The WSM6-family SR check follows the live step's minor-loop count under an adaptive clock. This model runs one microphysics scheme, not of that family, at a fixed step, and the carried driver is not its physics path (`PHYSICS-11`).

**`PHYSICS-16`.** Three SASE boundary-layer changes: the closure's switches fall back to the defaults the run configuration declares instead of literals (the literal fallback dropped the additive dissipation channel, which ships on, on any configuration object without the field), and SASE runs on nested domains with the nest's boundary rows masked like a specified domain's. This package does not admit SASE. At 2.8.1 the engine moved the switch fallback into a helper function, which is this row's first hunk now.

**`PHYSICS-17`.** Noah-MP options that reach no code are admitted with one warning and skipped at the driver instead of refused a second time. This model runs Noah, not Noah-MP.

**`PHYSICS-18`.** A driver whose first step is not the first model step, a data-assimilation leg built fresh on an analysis, runs the radiation producer when a carrier the land surface consumes is still unsourced, instead of refusing at the first surface call. The carried driver is not this model's physics path, so nothing it produces moves; the pull closes a false refusal inside carried code, like `PHYSICS-2`.

**`PHYSICS-19`.** The options the engine's driver gained between 2.8.0 and 2.8.5: the urban canopy models, the UW boundary layer, stochastically perturbed parameterisations, the lake model, the Noah mosaic, slope radiation and topographic shading, the sub-grid terrain wind and orographic drag, scalar boundary-layer mixing, the RUC soil-property lineage, the adaptive time step's periods and deadlines, and the cold starts they need. This package never constructs the carried driver: no door and no test calls `initialize_physics`, because the native suite calls the scheme modules directly, so every one of these hunks is dead code here whichever way it is decided. Take any of them only with the option it serves.

**`PHYSICS-20`.** Structure rather than options: the output-field, dispatch and split flux-diagnostic tables move out of the driver into engine modules (`gpuwm.io.history_layout` and the engine's inventory, `INVENTORY-6`), the coupling of boundary-layer, cumulus and radiation tendencies to the faces becomes one launch through `gpuwm.core.tendency_coupling`, the driver imports the engine's new cumulus-clock, land-forcing, surface-diagnostic, surface humidity and boundary-layer scratch modules, the cumulus result of the engine's own scheme skips a validation copy, and the boundary-layer coupler is handed the atmosphere. Same standing as `PHYSICS-19`: the driver is never constructed here. Taking any of it would import engine modules the seam does not pin.

**`PHYSICS-21`.** Comment line citations into the engine's WSM6 and WDM6 kernels, which moved. Nothing to decide.

At 2.8.1 to 2.8.5 the engine's new driver options and structure (`PHYSICS-19`, `PHYSICS-20`) landed next to `PHYSICS-1`, `PHYSICS-3`, `PHYSICS-4`, `PHYSICS-5`, `PHYSICS-7` and `PHYSICS-14`, so those rows' engine lines grew; what each describes is unchanged.

### `core/noah.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NOAH-PY-1` | after 378, 392-392 | 381-385, 399-399 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing on a shipped path | **pull** |
| `NOAH-PY-2` | 439-441, after 444, 448-461 | 446-447, 451-459, 463-476 | equal to carried | engine-fix | engine v2.8.5 (3e2d273), 2026-10-03 | nothing numeric | **pull** |
| `NOAH-PY-3` | 15-15, 469-469, after 477, 511-511 | 15-17, 538-539, 548-555, 589-598 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing at this package's settings | **none** |
| `NOAH-PY-4` | after 462 | 478-531 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing in this package | **none** |

**`NOAH-PY-1`.** WRF's cold-start soil initialisation compares a stored FP32 soil temperature against the FP32 literal 273.149. This copy compares against the binary64 273.149, and the FP32 word nearest it is 273.148987, below it, so a level stored at exactly that word goes down the frozen branch where WRF takes the warm copy. The engine compares against `float(np.float32(273.149))`. Reached here only through `sh2o_init`, which no door calls and one test does, so no shipped number moves; the pull is one line and removes a wrong answer from carried code.

**`NOAH-PY-2`.** The cache of uploaded land-surface tables was keyed by the identity of the parameter object and held a reference to it. Every run in one process loads its own parameter object, so every ensemble member or assimilation cycle left one more copy of the same tables on the card; the engine keys the cache by the values it uploads. Nothing numeric moves. The engine's version fills the cache through `gpuwm.core.device_cache`, so it carries the same floor condition as `RRTMGP-23`.

**`NOAH-PY-3`.** The launcher takes the urban model's hand-over and chooses the kernel entry compiled with it; without one it launches the same `noah_column` as before. The docstring follows. This package runs no urban model. `NOAH-CU-2` is the kernel half.

**`NOAH-PY-4`.** A device version of the cold-start liquid-water initialisation, compiled from an engine-only unit (`LOADER-4`). This package runs no cold-start soil initialisation from any door.

### `core/morrison.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `MORRISON-PY-1` | 42-43, 57-60, 135-144, 168-168, 174-176, 210-218, 252-254 | 63-63, after 76, after 150, after 173, 179-179, 213-217, 251-251 | equal to engine | global-fix | owner bef189b1d, 2026-09-12 | nothing by itself; it carries the two quantities of MORRISON-CU-1 | **offer** |
| `MORRISON-PY-2` | 4-4, after 29, 172-172 | 4-4, 30-32, 177-177 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing: the engine records every output word unchanged | **none** |
| `MORRISON-PY-3` | after 33 | 37-54 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing in this package | **none** |

**`MORRISON-PY-1`.** The driver half of the sedimentation repair: two scratch volumes, allocated as named slots so a stepping run creates nothing per step, carrying the two per-level quantities the process stage publishes to the sedimentation stage. At this model's working truncation with 40 levels that is two 45 MiB volumes, and both arms of the card comparison reported the same device peak, 8.80 GiB, so they did not move the card requirement. At 2.8.1 the engine's `apply` preparation (`MORRISON-PY-3`) replaced the neighbouring lines.

**`MORRISON-PY-2`.** The launcher half of `MORRISON-CU-3`: the sedimentation block grows to one warp per hydrometeor category, and the docstring says so.

**`MORRISON-PY-3`.** `apply` forms potential temperature, the Exner function and layer depth in one elementwise kernel that reproduces the separate array operations' roundings. This package calls `launch_morrison` and never `apply`.

### `core/npref.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NPREF-1` | 1-37 | after 0 | not present | global-fix | package cd407ae, 2026-09-12 | nothing | **offer** |
| `NPREF-2` | 1045-1046, 1051-1056, 1075-1083, 1123-1132, 1136-1136, after 1137, after 1139, 1387-1389, 1409-1412, 1945-1945, 2047-2048, 2096-2097, 2135-2135 | 1049-1049, after 1053, 1072-1075, 1115-1116, 1120-1120, 1122-1122, 1125-1125, after 1372, after 1391, 1924-1924, after 2025, 2073-2073, after 2110 | equal to engine | global-fix | owner bef189b1d, 2026-09-12 | the mirror half of MORRISON-CU-1 | **offer** |
| `NPREF-3` | 4320-4320, 4489-4491, after 4502, after 4766, 4806-4806, 4819-4821, 4828-4828, 4857-4857 | 4394-4394, 4563-4565, 4577-4577, 4825-4830, 4870-4870, 4883-4885, 4892-4894, 4917-4918 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing in any field this mirror already returns | **none** |
| `NPREF-4` | 4736-4749, 4752-4753, 4758-4759, 4837-4844, 4859-4859, 4866-4866 | 4811-4811, after 4813, after 4817, 4903-4904, after 4919, 4926-4926 | equal to engine | deliberate | owner c3f2de9b7, corrected by 47ea9ae89, both 2026-09-04 | nothing at the three original options | **refuse** |
| `NPREF-5` | 6365-6366, 6368-6372, 6408-6410, 6773-6774, 6777-6778 | 6425-6425, after 6426, after 6461, after 6818, after 6820 | equal to engine | deliberate | owner 36cb7cd43, 2026-09-05 | nothing at the shipped default | **refuse** |
| `NPREF-7` | 6568-6575, 6578-6590 | 6619-6636, after 6638 | neither | global-fix | owner 304177d6f, 2026-09-04; engine 4d523b793, same day | identical where the index is above one; the height differs in one corner | **offer** |
| `NPREF-8` | 6601-6602 | after 6648 | equal to engine | global-fix | owner 304177d6f, 2026-09-04 | narrow but not empty | **offer** |
| `NPREF-9` | 7456-7462, 7464-7465, 7471-7487, 7489-7492, 7506-7506, 7512-7517, 7520-7520, 7524-7524, 7539-7542, 7555-7556, 7558-7562, 7578-7578, 7580-7580 | after 7497, 7499-7499, after 7504, 7506-7508, 7522-7522, 7528-7531, 7534-7534, 7538-7538, 7553-7555, after 7567, 7569-7575, 7591-7591, 7593-7593 | equal to engine | global-fix | owner 50e109983, 2026-09-04 | up to 3.35 W/m2 of spurious absorption removed | **offer** |
| `NPREF-10` | 7767-7767, 7772-7911, 7918-7921, 7935-7951, 7953-7954, 7986-7988, 8008-8010, 8041-8045, 8059-8069, 8071-8072, after 8073, 8075-8092 | 7780-7780, after 7784, 7791-7791, 7805-7806, after 7807, 7839-7840, 7860-7862, after 7892, after 7905, 7907-7908, 7910-7911, 7913-7917 | equal to engine | global-fix | owner 4bf9f3e94, 9cfc20654 and 984c1cc61, 2026-09-04; eaa150a42, 2026-09-04; f6999e233, 2026-09-05 | the mirror half of RRTMGP-10, RRTMGP-12 and RRTMGP-13 | **offer** |
| `NPREF-11` | after 9352, 9354-9354, 9439-9440, 9453-9454, 9456-9458, 9460-9463, 9465-9465, 9493-9493 | 9178-9183, 9185-9185, 9270-9273, 9286-9289, 9291-9294, 9296-9299, 9301-9301, 9329-9329 | equal to carried | engine-fix | engine ab874b32e, 2026-09-04 | 1.23 percent on the shallow tendencies at a 90 s step | **pull** |
| `NPREF-12` | 3257-3257, 3309-3309, 3421-3421, 3442-3442, 3454-3454 | 3303-3303, 3355-3355, 3467-3467, 3488-3488, 3500-3500 | equal to carried | no-behaviour | engine 9a2f350e7, 2026-09-14 | nothing, comments only | **none** |
| `NPREF-13` | after 2300 | 2276-2346 | not present | engine-fix | engine 72f18213d, 2026-09-16 | nothing in this package | **none** |
| `NPREF-14` | 609-611, after 669 | 572-576, 635-673 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing in this package | **none** |
| `NPREF-15` | 3506-3507, 3513-3513, 3530-3530, 3532-3532, 3541-3541, 3544-3544, 3549-3549, 3552-3552, 3560-3560, 3563-3563, 3616-3616, 3618-3619, 3623-3623, after 3637, 3639-3639, 3641-3641, 3750-3751, 3801-3801, 3809-3809 | 3552-3553, 3559-3559, 3576-3576, 3578-3581, 3590-3594, 3597-3600, 3605-3605, 3608-3610, 3618-3622, 3625-3628, 3681-3682, 3684-3685, 3689-3690, 3705-3705, 3707-3707, 3709-3709, 3818-3819, 3869-3872, 3880-3883 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing in this package | **none** |

**`NPREF-1`.** The third-party notice for what this file transcribes. Its RTE+RRTMGP half matches the engine's own header for the same work; its AER half is the correction of RRTMGP-2, in the mirror rather than the driver, and applies to the engine's copy unchanged.

**`NPREF-2`.** The float64 mirror is the instrument the scorecards grade against, so it moves with the kernel or it is a flawed instrument. Here it stops rebuilding the two per-level quantities and requires them instead of falling back, which is how it produced a third value distinct from both the kernel's and WRF's.

**`NPREF-3`.** The engine adds one output, the momentum friction velocity, computed as the same relaxation as the friction velocity but on the raw wind speed, without the convective and subgrid additions, without the wind speed's own floor and without the land floor. Everything else is identical arithmetic on both sides. Do not pull the mirror hunks alone: this package's kernel contains no occurrence of the field and this model's boundary layer never reads it, so a mirror returning a field the kernel it mirrors does not write is exactly the failure a float64 mirror exists to prevent. One loose end worth recording: this file still contains a turbulence-energy routine that ACCEPTS the field and falls back to zero when it is absent, so the carried tree can consume it and cannot produce one. If this package ever takes the engine's surface-layer kernel, these hunks ride with that kernel change in the same commit.

**`NPREF-4`.** The mirror half of SFCLAY-PY-2. The correction is the half that made it survive: with the momentum roughness inside the roughness Reynolds number, an unvegetated column at a roughness near 1 m and a friction velocity above 1 m/s put the roughness ratio near 100, decoupled the skin from the air, and the first day-long arm died on a non-finite wind inside its first hour. A future engine change to the option ladder is refused if it would remove the fourth option or fold it into a WRF branch; a change confined to the first three lands cleanly on the branch above and can be taken.

**`NPREF-5`.** The mirror half of YSU-CU-4. At the default this is bit-identical to the engine. Selecting the fixed mode replaces the thickness-scaled asymptotic length with a constant, which changes the squared length and so the local diffusivity above the boundary layer: on the 40-level global stack, whose layers across the jet are about 1500 m thick, WRF's rule reads a 150 m length where a 300 m layer reads 30 m, about 25 times the momentum diffusivity for the same shear and Richardson number.

**`NPREF-7`.** The one hunk in this file where both lines moved. Both treat the two statements at the end of WRF's scan as the independent statements they are, and that half is now identical. They differ on what happens to the boundary-layer height. The scan can set the flag true on its last visited level without ever moving the index off one, and the engine then interpolates with an index of zero, reading the element before the start of the column, which in Fortran is undefined and in the mirror is the top model level: the height comes back somewhere between the column top and the first level before the flag is cleared, and that value is what the caller reads and what the stable enhancement's own test reads next. This copy skips the interpolation on that path entirely. This form is a strict superset of the engine's: it contains the engine's independent clearing statement verbatim and adds the guard that refuses the undefined read. The project rule is to implement defined behaviour where WRF is undefined rather than reproduce the read, so nothing is pulled here, and the same read is still live on the engine in both its mirror and its kernel.

**`NPREF-8`.** WRF clears the same flag after the stable enhancement's own sweep when that sweep returns an index of one. This copy added the statement; the engine has no equivalent. The enhancement runs only under a stable surface and a low boundary-layer top, and the flag can still be true there when the liquid-water scan revived it, so on the engine such a column can carry the flag true into the entrainment block with an index of one, where the entrainment level is minus one and indexes the top of the column. Offer it with NPREF-7: the two statements are the same rule at the scheme's two sweeps and the engine took only the first.

**`NPREF-9`.** The floor under the two-stream diffusivity moves from ten thousand times the float32 epsilon to ten thousand times the float64 epsilon, which is the reference's own value at the reference's working precision and only keeps the quantity strictly positive; the higher floor is an absorption the medium does not have in any layer whose true value is below it, which is the conservative-scattering bands. Measured: a conservative layer of optical depth 10 absorbed 0.50 W/m2 of 680 incident and one of depth 30 absorbed 3.35, and through the shipped cloud optics an overcast liquid cloud lost 0.24 to 2.27 W/m2 of reflected flux. The literal alone is not enough: the equations as written are unstable in float32 at small products, 6.9 W/m2 of spurious absorption at optical depth 2, so the layer arithmetic is rearranged through the exponential-minus-one function and an algebraic identity so that every term is of the order of the quantity itself. The function also gains a dtype argument so the mirror can be evaluated in float32 to reproduce the device kernel. Offer it AS A PAIR with RTE-CU-1: the engine's mirror deliberately sets its floor from the float32 epsilon so that it matches its own kernel, and both then disagree with the reference, so an engine that took the kernel alone would have a mirror that no longer mirrors it.

**`NPREF-10`.** About 330 carried lines against 145 on the engine, which makes this the largest single block of divergence in the file. It is the mirror of the three cloud-optics rows: the counted size bounding with its geometric carry, the area-conserving ice and snow merge, and the sentinel reconstruction, plus the column cloud cover the engine does not have. The arguments that select the old behaviour exist so an instrument can state the before beside the after; they are not device options, and their defaults are the corrected coupling, so a bare call gets the fixed behaviour. A future engine change anywhere inside this block will need re-expressing rather than merging.

**`NPREF-11`.** WRF's shallow cumulus arm re-sets its adjustment time to exactly 2400 s, discarding the rounding applied earlier, and every feedback tendency then divides by the unrounded value while the closure and advection arithmetic above it keeps the rounded one. This copy divides all of the shallow feedback tendencies by the rounded value. Whenever the step does not divide 2400 exactly the two disagree by the rounding ratio; at common steps they are identical. The case for pulling is unusual: this package carries no kernel for this scheme, this model runs no such scheme, and the mirror function reaches the engine for its own table through an import the carve deliberately leaves pointing at the engine. So the only kernel this function can ever be checked against is the published engine's, which has carried the fix since before the pinned floor. That makes the carried mirror wrong against the only kernel it mirrors, which is the definition of a flawed instrument. The pull is bookkeeping, costs nothing because no run reaches this code, and removes a trap for anyone who calls it.

**`NPREF-12`.** Comment words from the engine's 2.7.5 word sweep. Nothing to decide.

**`NPREF-13`.** A float32 CPU oracle for the engine's dycore vertical-velocity diagnosis. This package carries no dycore, so there is nothing for it to check here.

**`NPREF-14`.** WRF advects w at the model-top level too, with horizontal fluxes extrapolated from the last two mass levels and a one-sided lid term; this mirror gives that level zero tendency. It is the float64 mirror of the regional dycore, which this package does not carry and no door calls, so the standing is `NPREF-13`'s.

**`NPREF-15`.** Three more regional dycore mirror changes: the radiative open boundary takes WRF's staggered map factors, the normal-direction diffusion term is excluded on outer open and specified rows as WRF does, and the vertical-velocity damping starts at `w_crit_cfl` under implicit vertical advection, with the default unchanged. Same standing as `NPREF-14`.

### `core/kernels/__init__.py`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `LOADER-1` | after 96 | 182-197 | equal to carried | engine-fix | engine 322df3fa6 and 94f4aa331, both 2026-09-10 | nothing in this package | **refuse** |
| `LOADER-2` | 99-99, 131-131, after 171 | 205-205, 237-237, 279-290 | equal to carried | engine-fix | engine 3845bcab9, 2026-09-24 | nothing numeric | **none** |
| `LOADER-3` | after 2, 106-106, 160-160, 166-166 | 3-3, 212-212, 267-267, 273-273 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing numeric | **pull** |
| `LOADER-4` | after 39, after 61, after 65 | 41-47, 70-74, 79-107 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing in this package | **none** |
| `LOADER-5` | 72-72, 74-74, 78-78, 84-85, 88-91, 94-94, after 97, 146-146, 156-157 | 114-114, 116-116, 120-120, 126-133, 136-144, 147-179, 199-203, 252-252, 262-264 | equal to carried | engine-fix | engine v2.8.2 (4c27749), 2026-10-02 | nothing in this package | **none** |

**`LOADER-1`.** The loader gained a branch that fires only for a module name with one prefix, sending a standalone land-surface translation unit through a composition factory instead of the plain compile route. The carried loader cannot be handed such a name, and that needs spelling out, because this package DOES run that scheme: it imports the step function from an engine module, maps the selector to it and calls it. That scheme compiles on the engine's side of the seam. All thirteen engine modules that build its device code import the engine's loader, not this one, so such a name never reaches the carried one. The carried loader's own inputs say the same from the other end: the carve carries fifteen kernel files and none is one of that scheme's units, the kernel directory has no fallback to the engine's, and the three carried call sites name carried kernels only. Refuse on unreachability alone: the branch is dead code here whichever engine version is installed. On symbol availability, recorded so nobody re-derives it as a reason, the helper the branch calls is absent from the floor of this package's declared range and present from 2.7.3 up, so on an install resolving today's engine the import would succeed and nothing would raise. Reopen only if the carve ever carries one of that scheme's kernels, and then the engine's translation-unit composition has to come with it, not just this branch.

**`LOADER-2`.** The engine's loader publishes a real compile, as opposed to a cache load, to a watching run as progress. Nothing numeric moves and the carried loader compiles the same images. It needs the engine's compile-notice module, present from 2.8.0; take it when this package wants the same progress line.

**`LOADER-3`.** The four loader caches move from a process-wide `lru_cache` to the engine's per-device `cuda_cache`. A kernel wrapper is bound to the card it was first resolved on, so a process-wide cache hands a second card the first card's function. Latent here for the reason `RRTMGP-23` gives, and it carries the same condition: `gpuwm.core.device_cache` first ships at 2.8.1.

**`LOADER-4`.** Header-table entries for the engine's urban, lake, mosaic, UW boundary layer, real-data initialisation and cold-start units. None of these names reaches the carried loader, and an entry for an absent name composes nothing, so unlike `LOADER-1` there is nothing to refuse either.

**`LOADER-5`.** A `kernel_dir` argument through the composition functions, so the engine's division census can assemble a unit from a tree other than the imported one; `module_options` with a separate `--fmad=false` compile site for the lake kernel; and the opt-in strict-arithmetic hook, which under a `GPUWM_WRF_EXACT` environment variable adds headers to two engine units. None of it reaches a name this loader compiles, and the hook imports `gpuwm.wrf_exact`, which the 2.8.0 floor does not ship.

### `core/kernels/gf.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `GF-CU-1` | 14-14, 42-47, 49-54, 56-61, 629-635, 3038-3040, 3048-3054 | 14-15, 43-49, 51-57, 59-95, 576-583, 2859-2868, 2876-2879 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | taking the engine's would move the deep cloud-base mass flux by up to 7.3 percent | **refuse** |
| `GF-CU-3` | 98-184 | after 131 | equal to engine | deliberate | owner bc62dcfdd and 4939f1c13, 2026-09-04; 957059111, 2026-09-05 | nothing at the shipped defaults | **refuse** |
| `GF-CU-4` | 1199-1210, 2247-2247, 2312-2325 | 1151-1151, after 2180, after 2244 | equal to engine | deliberate | owner 957059111, 2026-09-05 | nothing with the define off | **refuse** |
| `GF-CU-5` | 2352-2352, 2383-2398, 2488-2488, 2956-2956, 3010-3011, 3024-3024 | after 2270, after 2300, 2390-2390, after 2779, 2833-2833, after 2845 | equal to engine | deliberate | owner bc62dcfdd and 4939f1c13, both 2026-09-04; 55aa34fb4, 2026-09-04 | nothing at the default | **refuse** |
| `GF-CU-6` | 2908-2928, 3941-3944, 3947-3947, 3956-3956, 3973-3974, 4177-4177, 4179-4180, 4206-4208 | after 2752, after 3765, 3768-3768, after 3776, after 3792, after 3981, 3983-3983, 4008-4009 | equal to engine | deliberate | owner 957059111, 2026-09-05 | nothing with the define off, re-derived rather than assumed | **refuse** |
| `GF-CU-7` | 1487-1488, 1511-1513, 1516-1517, 2785-2841, 2848-2848 | 1426-1426, after 1447, after 1449, after 2686, 2693-2693 | equal to engine | deliberate | owner 2090ac84a, e30f9fda6, c4a95bc61 and 58fd47304, all 2026-09-05 | nothing with the define off, re-derived rather than assumed | **refuse** |
| `GF-CU-8` | 1341-1341, 1346-1346, 1390-1390, 1495-1495, 1718-1738, 1751-1752, 1759-1767, 2781-2781, 2875-2875, 3022-3022, 4000-4018, 4024-4038, 4192-4193, 4202-4202, 4308-4323, 4330-4333 | 1282-1282, after 1286, after 1329, after 1432, after 1641, after 1653, after 1659, 2683-2683, 2720-2720, 2844-2844, after 3827, after 3832, 3995-3995, 4004-4004, after 4108, after 4114 | equal to engine | adaptation | owner fca73b086, 2026-09-04; 957059111 and 58fd47304, 2026-09-05 | nothing, verified by reading every new access | **offer** |
| `GF-CU-9` | after 3984, after 3986, after 4067, 4172-4172, 4198-4198 | 3803-3811, 3814-3814, 3870-3872, 3977-3977, 4000-4000 | equal to carried | deliberate | owner c9a5f2c88, 2026-09-12, keeps the block; engine c0ffc53b5, 2026-09-04, replaced its own copy | nothing on the shipped path | **refuse** |
| `GF-CU-10` | 1544-1552 | 1476-1476 | equal to engine | global-fix | package, 2026-10-05 | the reported convective rain falls to the water the tendencies remove: 1.27 to 1.004 times it over the oracle capture's raining columns | **offer** |
| `GF-CU-11` | 674-674, 676-676, 682-682, 697-697, 699-704, 718-730, 1883-1884, 1893-1893, 1903-1903, after 1963 | 622-622, 624-624, 630-630, 645-645, 647-654, 668-682, 1775-1792, 1801-1801, 1811-1811, 1872-1897 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing at the engine's settings, by its record | **none** |
| `GF-CU-12` | 4042-4042 | 3836-3844 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing: the engine records every output word unchanged | **none** |

**`GF-CU-1`.** The comment block and the library probe of the cumulus kernel, on the side of the gamma the kernel calls. The shape factor of the mass-flux profile is a ratio of three gamma functions, and the closure's difference of two nearly equal quantities turns one unit in the last place of it into up to 7.3 percent of the deep cloud-base mass flux, so which gamma the kernel calls is a physics choice. The gamma this package carries is its own work and the one the global model was graded with; the engine line's correctly rounded variant moves that shape factor on 68.17 percent of reachable tuning values (the engine line's own host-compiled enumeration of 2026-09-04, cited in full under LIBM-3) and is a numerics choice to grade against observations, not to pull. The number-moving definition is the shared header, LIBM-3.

**`GF-CU-3`.** The two compile-time defines for the coarse-column deep arm, and the comment that states what each one does. Both default to zero, so a default run compiles the engine's own text. They exist because at 50 km the grid scale cannot carry deep convection: the observed failure was a warm-pool column with 1,200 J/kg of available energy and seventeen saturated levels raining 84 mm/h through the microphysics while the cumulus scheme booked 0.06 mm/h. A future engine change to the code these defines guard is re-applied around them, never instead of them.

**`GF-CU-4`.** With the define off both guards are inside the conditional block, so the kernel takes the reference's exit exactly as the engine does and the counter is only exported. With it set, a level the downdraft's profile reaches with no mass takes the environment's moisture, enthalpy and momentum, evaporates nothing, contributes no buoyancy, and the sweep continues below it. The reason is the same as the row above: on a column the grid resolves none of, exiting the whole scheme because the downdraft profile collapsed within a level or two of its detrainment height hands deep convection to a grid scale that cannot carry it. The updraft's own zero-denominator exit is untouched on both sides.

**`GF-CU-5`.** With the define off, the override is a disjunction of two zero macros, so the flag stays zero and the downdraft entrainment reduces to the engine's expression; the export chain is write-only. With it set, a column the reference rejects for a downdraft that is not negatively buoyant, or for one whose moisture equation exits on a zero denominator, keeps its deep updraft, its rain and its closure with the downdraft forced off, and the overridden exit is exported per column.

**`GF-CU-6`.** With the define off the threshold this adds is initialised to zero and written only inside the conditional block, so the comparison against the positive per-level heating cap never fires, the shallow call passes a literal zero and is excluded by its own test anyway, and the two new out-parameters are read back only into void casts and the census. With the define set, the 300.01 K/day cap becomes the larger of itself and the column's own profile peak scaled by the latent heat of the moisture the grid is converging into it, so a column whose request stays under the reference bound is untouched and one above it is let through up to that latent heat. The bound is tuned for grids that resolve part of the convection; on a 52 km column the scheme is the only sink. The negative check's signature grew three parameters and both of its call sites moved with it, so any engine change to that routine or to either call arrives as a conflict and has to be re-applied on the wider signature.

**`GF-CU-7`.** The floor is declared zero OUTSIDE the conditional block and computed only inside it, and the mass flux is non-negative by construction, so with the define off the comparison never binds. With it set, a convecting column's cloud-base mass flux is floored at a share of the Kuo moisture-convergence member, partitioned by a critical humidity of 0.9 measured on the control's own storm hour rather than inherited. It is applied after the diurnal-cycle term, so a column that term silenced can convect on the floor alone: 1,750 of 3,063 floored columns at hour 12, 1,298 of them silenced ones. The reason is that the sixteen-member mean gives the moisture-convergence members a quarter of the weight, a hedge that is wrong where the grid carries none of the converging moisture itself. The output routine gained two parameters and its call site carries both, so an engine change there is a three-way merge.

**`GF-CU-8`.** A per-column reading of what the closure asked for and what it applied, built because the storm columns of the control left the scheme through the downdraft exits or under the heating cap and no number said which. Every new slot is written and never read back into a term, the new slots are appended after the existing ones so no index moves, and the three reads that were introduced are each shown inert with both defines off in the rows above. All arithmetic in the kernel goes through the round-to-nearest intrinsics, so the added stack structure cannot perturb results through contraction either. This is most of why this scheme's driver sits at 45.7 percent line similarity with the engine's; the file is not rebuilt physics. The engine may want the same instrument. Three device-function signatures grew, so any engine change to those routines has to be re-applied on the wider signature.

**`GF-CU-9`.** The kernel half of GF-PY-2: three per-column override slots for the shape factor and the two call sites that pass zero here where the engine passes the override. The engine opened the hatch so its own gamma replacement stays auditable against a reference capture. This package keeps the gamma it was graded with, so the hatch is refused with GF-CU-1 and GF-PY-2.

**`GF-CU-10`.** One sign. The deep arm's output routine takes the downdraft's evaporation from the detrained cloud water first and from the rain for the remainder, and accumulates minus that remainder into the rain. The reference then closes with the rain set to minus the accumulator plus the updraft's condensate, which reports the condensate PLUS the remainder, while the moisture and cloud-water tendencies it applies remove the condensate MINUS the remainder. So the scheme reported rain its own tendencies never took out of the column: 1.27 times the removed water over the 84 raining columns of the WRF oracle capture, about 1.8 times globally and 2.3 times over land on a T255 GDAS forecast, and the land surface was forced with the difference while the surface reservoir paid for it. This copy adds the accumulator instead, so the rain is the condensate less the remainder: over the oracle capture the reported rain is 1.004 times the removed water, the remaining 0.4 percent being the gap between the kernel's interpolated layer masses and the model's. The tendencies, the closure and the set of convecting columns are unchanged on that capture. This is a defect the reference itself carries (WRF 4.7.1 `module_cu_gf_deep.F:3348`), so it is a deliberate divergence from the reference and not a transcription question; the regional engine's copy carries the same line and is where to send it.

**`GF-CU-11`.** A production column the trigger rejects writes zero outputs and returns, and the environment above the trigger's reach is formed only for columns that pass. The engine records its outputs as unchanged. Here it would have to be proved again against this package's coarse-column arm (`GF-CU-3` to `GF-CU-7`), which changes what later stages do with a flagged column; not owed, and not taken without that check.

**`GF-CU-12`.** `__launch_bounds__(64, 8)` on the driver kernel, guarded so the host oracle still includes the source. The kernel half of `GF-PY-7`; take the two together.

### `core/kernels/ntiedtke.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NTIEDTKE-CU-1` | 123-123, 128-128, after 236, 243-243, after 454, 461-461, after 534, 541-541, after 875, 882-882, 989-995, after 1044, after 1222, 1229-1229, after 1301, 1308-1308, after 1485, 1492-1492, after 1725, 1732-1732, after 1733, after 2175, 2182-2182, after 2281, 2288-2288, after 2437, 2444-2444, after 2551, 2558-2558, after 2739, 2746-2746, after 3035, 3042-3042, after 3126, 3133-3133, after 3263, 3270-3270, after 3347, 3354-3354, after 3465, 3472-3472, after 3588, 3595-3595, after 3692, 3699-3699, after 3787, 3794-3794 | 123-123, 128-129, 238-238, 245-245, 457-457, 464-464, 538-538, 545-545, 880-880, 887-887, 994-1008, 1058-1058, 1237-1237, 1244-1244, 1317-1317, 1324-1324, 1502-1502, 1509-1509, 1743-1743, 1750-1750, 1752-1753, 2196-2196, 2203-2203, 2303-2303, 2310-2310, 2460-2460, 2467-2467, 2575-2575, 2582-2582, 2764-2764, 2771-2771, 3061-3061, 3068-3068, 3153-3153, 3160-3160, 3291-3291, 3298-3298, 3376-3376, 3383-3383, 3495-3495, 3502-3502, 3619-3619, 3626-3626, 3724-3724, 3731-3731, 3820-3820, 3827-3827 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing at this package's settings | **none** |

**`NTIEDTKE-CU-1`.** The kernel half of `NTIEDTKE-PY-4`: each stage kernel takes the device-side validation mask and returns early on a chunk already found invalid, the one stage that reads the flag reads it from the mask, and a per-trial scratch clear that inactive trials cannot observe moves to the final reset. Take it with its driver half or not at all.

### `core/kernels/sfclay.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `SFCLAY-CU-1` | 10-13, 236-237, after 262, after 509, 544-544 | 10-13, 219-220, 246-246, 472-478, 513-514 | equal to carried | engine-fix | engine 3669990e8, 2026-09-04 | nothing to any existing output | **pull** |
| `SFCLAY-CU-2` | 15-31 | after 14 | equal to engine | no-behaviour | owner 9360aca3e, 2026-09-04 | nothing, comment only | **none** |
| `SFCLAY-CU-3` | 472-492, 495-495, 500-500 | 456-456, after 458, after 462 | equal to engine | deliberate | owner c3f2de9b7, corrected by 47ea9ae89, both 2026-09-04 | nothing at the three original options | **refuse** |
| `SFCLAY-CU-4` | 123-124, 135-136, 273-273, 275-275, 293-293, 299-299, 397-397, 399-399, 401-401, 429-429, 436-436, 513-513, 519-519, 522-523 | 106-107, 118-119, 257-257, 259-259, 277-277, 283-283, 381-381, 383-383, 385-385, 413-413, 420-420, 482-482, 488-488, 491-492 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | Blackwell cards only: at most one unit in the last place per constant division | **pull** |

**`SFCLAY-CU-1`.** The kernel half of SFCLAY-PY-1: one inout pointer, three lines that read the previous value, form the raw wind speed and store the relaxation, and nothing else. Verified by reading the engine kernel: neither new local appears anywhere else. It is the friction velocity relaxation without the convective correction, without the wind speed's floor and without the land floor, which is why it is a separate field rather than a diagnostic. Inert for this model today, and the reason to take it is that it removes a permanent three-file drift at zero numeric cost.

**`SFCLAY-CU-2`.** Seventeen comment lines recording this kernel's column-by-column grade against hash-verified reference sources. Kept because it is the standing reason to REFUSE a future patch from either line that adds the moisture-flux floor, the negative sensible-flux floor or the dissipative heating term: all three are commented out in both reference files, so their absence here is the transcription and adding them would be bit-exact to a scheme the reference does not run.

**`SFCLAY-CU-3`.** The kernel half of SFCLAY-PY-2. The engine's line here is still the two-way conditional, which would silently run the fourth option as the third, so any engine change to that block has to be re-applied over the new branch. At 2.8.1 the engine respelled the constant division inside this block (`SFCLAY-CU-4`); that part is a pull and rides with that row.

**`SFCLAY-CU-4`.** The engine respelled every float division by a compile-time constant as `__fdiv_rn(x, C)`. Its record (the division census, engine A146): NVRTC targeting compute_100 and compute_120, under the `-ftz=true` CuPy appends to every RawModule, replaces `x / C` with a multiply by the rounded reciprocal, which is not IEEE division and puts about a third of `x / 3.0f` quotients one unit in the last place off; compute_90 and older keep `div.rn`, and `__fdiv_rn` reaches PTX as `div.rn` on every architecture. The carried loader compiles through the same `cp.RawModule` call, so on a Blackwell card these kernels divide by reciprocal multiply today and agree with an older card only where the reciprocal rounds right. Taking it moves bits on Blackwell only, by at most one unit in the last place per affected quotient before propagation, and makes the same words come out on every architecture. Pull, with the kernel digests and every bit-identity anchor re-recorded on a Blackwell card when it lands. The same respelling is `YSU-CU-6`, `NOAH-CU-1`, `MORRISON-CU-2` and `RTE-CU-2`, and the hunks under `SFCLAY-CU-3`, `YSU-CU-5`, `NOAH-CU-2` and `MORRISON-CU-1` hold a few more of these divisions.

### `core/kernels/ysu.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `YSU-CU-2` | 346-367 | 392-407 | neither | global-fix | owner 304177d6f, 2026-09-04; engine 4d523b793, same day | identical wherever the index is above one | **offer** |
| `YSU-CU-3` | 380-382 | 420-421 | equal to engine | no-behaviour | owner 304177d6f, 2026-09-04 | nothing; the statement is provably unreachable on both sides | **none** |
| `YSU-CU-4` | 166-168, 584-590, 593-593 | 213-214, after 622, after 624 | equal to engine | deliberate | owner 36cb7cd43, 2026-09-05 | nothing at the shipped default; about 25 times the free-atmosphere diffusivity when selected | **refuse** |
| `YSU-CU-5` | 151-152, 204-204, 214-214, 616-616, 618-618, 633-634, after 636, 654-654, 656-656, 671-672, 676-676, 690-690, 693-693, 707-708, after 710, 719-719, 728-728, after 729, after 744 | 151-199, 250-250, 260-260, 647-654, 656-664, 679-688, 691-699, 717-723, 725-733, 748-757, 761-773, 787-865, 868-876, 890-899, 902-913, 922-929, 938-944, 946-946, 962-1039 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing at this package's settings | **none** |
| `YSU-CU-6` | 241-242, 256-256, 282-282, 286-286, 288-288, 393-393, 402-402, 441-441, 445-445, 456-456, 458-458, 461-461, 578-580 | 287-288, 302-302, 328-328, 332-332, 334-334, 432-432, 441-441, 480-480, 484-484, 495-495, 497-497, 500-500, 617-619 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | Blackwell cards only: at most one unit in the last place per constant division | **pull** |

**`YSU-CU-2`.** The kernel half of NPREF-7. The engine's commit asserts the read is unreachable with its two rules in place. It is not: reaching it needs the flag true with an index of one on entry, and entering the scan with a false flag and an index of one is ordinary, because a stable-surface column takes the branch that clears the flag and the first-interface clamp is unchanged on both lines; the scan then raises the flag on any unstable level and the pending index update is never consumed if that level is the last one visited. Neither engine rule closes it: WRF's guard on the thermal-enhanced sweep (pulled 2026-10-05) applies only inside the surface-convective branch a stable column never enters, and the statement split changes what happens after the read, not whether it happens. Reproduced here for a stable-surface column whose liquid-water deficit sits 0.02 K below the surface value at the last visited level: the index read 1 with the top-down term on against 8 with it off, in the mirror and on a card; with the guard both read 8 and every output was bit-identical between the two settings and finite. Offer the kernel hunk with the mirror.

**`YSU-CU-3`.** A clearing statement after the stable re-diagnosis, citing its reference line. Reaching that block needs a stable surface, which means the branch above set the flag false, so only the liquid-water scan can revive it; if it does, the block above either clears the flag or recomputes the height and clears it whenever the height fell below the first interface. So arriving there with the flag still true implies both an index above one and a height at or above the first interface, and the second of those is exactly what keeps the branch from being entered. Keep it: it costs nothing, it cites its reference line, and it stops being dead the moment the block above it changes.

**`YSU-CU-4`.** One added line and the flag that controls it. At the shipped default the line never executes and the arithmetic is bit for bit the engine's. The mode is selectable and reported as a workaround rather than made the default, because its 24 h grade split: the northern 250 km energy ratio at 250 hPa improved 0.67 to 0.85 with a better speed bias, while the northern 250 hPa vector wind error rose 4.41 to 4.56 m/s and the grid-limit ratio 0.038 to 0.053. Refuse means re-apply, not drop: an engine change to the asymptotic length goes into the flag-zero path, which is meant to be the reference bit for bit, and the flag-one line is left alone unless the engine change addresses the same layer-thickness scaling, in which case this option is what is being replaced and should be retired rather than kept beside it. Note that the extra kernel argument is a binary-interface break in both directions: the published launcher would pass the level count into the flag and shift every later argument by one. At 2.8.2 the kernel signature this flag sits on also took the urban and topographic-wind arms (`YSU-CU-5`).

**`YSU-CU-5`.** The column body becomes a template whose urban arm (WRF's `flag_bep`, with the engine's declared rural-drag divergence) and topographic-wind arm sit under `if constexpr`, with two new entry points; the default `ysu_column` is compiled from the statements it had. This package runs neither arm. Two of these hunks also hold division respellings that belong to `YSU-CU-6`, which is how they would be taken.

**`YSU-CU-6`.** The same respelling as `SFCLAY-CU-4`, in this kernel, and the same decision for the same reason. One hunk carries the urban arm's right-hand side beside its division; take the division and leave the arm.

### `core/kernels/noah.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `NOAH-CU-1` | 98-98, 169-169, 214-214, 230-230, 285-285, 422-422, 839-839, 850-850, 857-857 | 101-101, 172-172, 217-217, 233-233, 288-288, 425-425, 842-842, 853-853, 860-860 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | Blackwell cards only: at most one unit in the last place per constant division | **pull** |
| `NOAH-CU-2` | 12-13, 866-866, 933-1527 | 12-16, 869-1605, 1672-1785 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing at this package's settings | **none** |

**`NOAH-CU-1`.** The same respelling as `SFCLAY-CU-4`, in this kernel, and the same decision for the same reason. The divisions inside the column body itself sit in `NOAH-CU-2`'s hunk.

**`NOAH-CU-2`.** The column body becomes a template with the urban hand-over under `if constexpr`, so the default `noah_column` compiles from the statements it had; the engine measured a runtime-flag version moving default forecasts through changed multiply-add contraction on an RTX 5090. This package runs no urban model. Because the whole body moved into the template, this row's hunk also holds the body's division respellings; take those under `NOAH-CU-1`.

### `core/kernels/morrison.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `MORRISON-CU-1` | 158-163, 165-166, 174-174, after 202, 366-366, 374-377, 393-396, 890-940, 944-944, after 954, 960-961, after 1005, 1042-1043, 1109-1109, 1115-1115, 1121-1122, 1145-1146, 1164-1188, 1208-1209, 1218-1219 | after 157, 159-159, 167-168, 197-198, after 361, after 368, after 383, after 876, 880-880, 892-892, 898-898, 961-961, after 997, after 1062, after 1067, after 1072, 1095-1096, 1117-1131, 1151-1152, 1161-1161 | equal to engine | global-fix | owner bef189b1d, 2026-09-12 | about 1.1 percent on both cloud droplet fall speeds on a column that warms 4 K in a step | **offer** |
| `MORRISON-CU-2` | 112-112, 133-133, 138-139, 152-152, 178-179, 242-242, 245-245, 414-416, 477-477, 580-581, 660-660, 694-694, 719-719, 1088-1088 | 112-112, 133-133, 138-139, 152-152, 172-173, 238-238, 241-241, 401-403, 464-464, 567-568, 647-647, 681-681, 706-706, 1042-1042 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | Blackwell cards only: at most one unit in the last place per constant division | **pull** |
| `MORRISON-CU-3` | 4-4, 946-946, after 952, after 968, after 978, 1160-1161, after 1268 | 4-4, 882-882, 889-889, 906-908, 919-933, 1110-1114, 1211-1222 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing: the engine records every output word unchanged | **none** |

**`MORRISON-CU-1`.** WRF builds two quantities once per level inside its column loop and then spends them, unchanged, in the sedimentation block that runs after the loop closes and before the tendency apply: the cloud droplet Stokes coefficient, frozen above the warm branch's small melt, and the particle-size reference density. Nothing writes the temperature between the melt and the apply, so the density the sedimentation block reads is the one the process section's own reconstruction used. Both copies of this kernel rebuilt both quantities from the temperature they held at sedimentation time, which is the post-process one. The Stokes coefficient is the expensive half, minus 0.288 percent per kelvin at 278 K on every cloudy level; the density was stale by the entry cleanup and melt alone, about 8e-4 K and 2.4e-6 relative. The remedy is the one WRF's own structure names: the process stage publishes both and the sedimentation stage consumes them. It is NOT handing the sedimentation stage its current temperature, which would have moved the size distribution about 2,400 times further from WRF than the error it removes, in the wrong direction. Measured against the unmodified reference driver over the 28 oracle columns and 10,948 compared values, desktop, NVIDIA GeForce RTX 3080, 2026-09-12: values disagreeing with the reference fall from 3,554 to 3,505, no field gets worse, and every field's worst distance in units of the last place is unchanged. The engine carries the identical rebuild from the original port commit and is owed the offer. Not confined to a climate band: cloud droplet sedimentation touches every column that holds cloud water. At 2.8.1 the engine restructured the sedimentation stage these quantities feed (`MORRISON-CU-3`) and respelled one division on a line this row also touches (`MORRISON-CU-2`), so the offer is now made into the per-category structure.

**`MORRISON-CU-2`.** The same respelling as `SFCLAY-CU-4`, in this kernel, and the same decision for the same reason.

**`MORRISON-CU-3`.** Sedimentation becomes one warp per hydrometeor category, the substep count is one maximum reduced across the five categories in shared memory (a chain of `fmaxf`, exactly associative), and terminal velocities are computed once instead of twice. The finalize stage gains a fast path for a cell with no hydrometeors and positive vapor; read statement by statement, every statement it skips adds zero or writes the value it writes. The engine records the outputs word-identical. It lands on the lines where `MORRISON-CU-1`'s published level quantities are consumed, so a take re-applies that offer inside the per-category structure.

### `core/kernels/rrtmgp_rte.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `RTE-CU-1` | 355-370, 427-436, 448-448, 450-454 | 354-354, 416-419, after 430, 432-438 | equal to engine | global-fix | owner 50e109983, 2026-09-04 | up to 3.35 W/m2 of spurious absorption removed | **offer** |
| `RTE-CU-2` | 168-168 | 167-167 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | Blackwell cards only: at most one unit in the last place per constant division | **pull** |
| `RTE-CU-3` | 59-77, 80-94, 134-134, after 135, 171-171, 250-250, 276-276, 289-289, 372-373, 375-375, 379-379, after 380, 457-458, 461-461, 468-469, 471-475, 478-478, 481-481, 483-483, 485-485, 493-495, 498-498, 500-500, 502-502, 510-511, 513-516, 519-519, 522-522, 524-524, 526-526, 533-535, 538-538, 540-540, 542-542 | 59-68, 71-90, 130-130, 132-134, 170-170, 249-249, 275-275, 288-288, 356-359, 361-361, 365-365, 367-369, 441-442, 445-447, 454-455, 457-462, 465-465, 468-468, 470-470, 472-472, 480-483, 486-486, 488-488, 490-490, 498-499, 501-505, 508-508, 511-511, 513-513, 515-515, 522-525, 528-528, 530-530, 532-532 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing: the engine records every output word unchanged | **none** |

**`RTE-CU-1`.** The device half of NPREF-9, and the only difference in this file: the engine's copy is the common ancestor byte for byte at 2.7.2, at 2.7.3 and through its next branch, with zero engine commits since the fork. The rearranged form agrees with the as-written form to 1e-8 relative in float64, matches float64 to 1e-3 W/m2 in every flux, and absorbs under 1e-4 W/m2 on a conservative cloud at optical depths 2, 10 and 30. Settled with no card, on the float64 mirror and the same arithmetic evaluated in float32. Offer it as a pair with the mirror. At 2.8.2 the engine packed these coefficients into local vectors (`RTE-CU-3`) on the same lines.

**`RTE-CU-2`.** The same respelling as `SFCLAY-CU-4`, in this kernel, and the same decision for the same reason.

**`RTE-CU-3`.** The two-stream solvers buffer sixteen flux values per column and fold each row in ascending g-point order with the same fused-multiply-add chain, several columns share a block, and the shortwave adding passes keep their coefficients in packed local vectors. The kernel half of `RRTMGP-24`'s solver change. The lines of `RTE-CU-1` sit inside this restructure, so a take re-applies that offer on top of it.

### `core/kernels/rrtmgp_gas.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `GAS-CU-1` | after 78, after 79, 400-412 | 79-87, 89-289, 610-624 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01, and v2.8.2 (4c27749), 2026-10-02 | nothing: the engine records every output word unchanged | **none** |

**`GAS-CU-1`.** The gas optics kernel runs four cells per block, one warp each, computes each minor-gas scaling and each flavour weight once per cell in shared memory, forms the dry-air column amount itself, and stores coalesced. The engine records every word unchanged. The kernel half of `RRTMGP-24`'s gas change.

### `core/kernels/rrtmgp_cloud.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `CLOUD-CU-1` | after 132 | 133-303 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing: a copy and a positive-zero fill | **none** |

**`CLOUD-CU-1`.** A copy-and-fill kernel for the clear above-model layers of the five cloud fields, the kernel half of `RRTMGP-24`'s one-launch tails.

### `core/kernels/rrtmgp_mcica.cu`

| Row | Carried lines | Engine lines | Ancestor | Class | Commit | Numeric effect | Decision |
|---|---|---|---|---|---|---|---|
| `MCICA-CU-1` | 42-42, 54-54, 76-76, 78-78, 82-82, after 89, after 109, after 113, after 114, 118-119, 125-129, 134-135, 137-137, 144-144, after 145, 152-152 | 42-42, 54-62, 84-85, 87-87, 91-91, 99-108, 129-130, 135-135, 137-145, 149-150, 156-158, 163-164, 166-171, 178-178, 180-188, 195-195 | equal to carried | engine-fix | engine v2.8.1 (d94f9ea), 2026-10-01 | nothing: the same mask, by reading and by the engine's record | **none** |

**`MCICA-CU-1`.** The seed is computed once per column into shared memory, the jump table is read bit-major, the modular products use a precomputed-reciprocal reduction, an all-clear column writes its zero mask and leaves, and a clear layer consumes its draw and moves on. Two of these were read here rather than taken on the record: on an aliased residue the walk now advances only the two multiply-with-carry components and takes the other two from their exact jumps, which is the state the full walk reached; and after a clear layer the next layer takes a fresh draw whichever way the clear layer's cumulative value was kept, because every draw is below one. The draw's multiply-add and the rescaling multiply are pinned to the rounding the compiler already chose. The kernel half of `RRTMGP-24`'s jump-table layout.

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
| `KERNEL-NOTICE-1` | 15-30, 34-42, 49-60, after 71, 109-110, 130-151, 167-171, 177-177, 179-191, 206-206, 210-221 | 15-26, 30-38, 45-67, 79-103, 141-142, 162-185, 201-202, 208-209, 211-212, 227-308, 312-365 | not present | adaptation | package cd407ae and 78cad51, both 2026-09-12 | nothing | **none** |

**`KERNEL-NOTICE-1`.** The notice beside the device sources, narrowed by rule to the files this package actually ships rather than inherited whole. The carve's own scope rules do the narrowing, so the remaining differences are the ones the rules do not cover: the reason the notice sits beside the code rather than inside it is this package's receipt digests and not the source tree's frozen-digest suites, the scope paragraph names one header instead of a list of files this package does not have, the McICA generator is filed under its own work, and the closing scope section says whose work the gamma routines are. At 2.8.1 and 2.8.5 the engine's notice also grew sections for work this package does not ship (the trigonometric cores, CORE-MATH, the log-gamma coefficients and the binary64 boundary-layer library); the narrowing rule leaves them out.
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
* The four microphysics kernel fixes the engine made on 2026-09-12 after this
  package's cut: vapor returned by the final condensate cleanup is stored
  instead of dropped (`35077a4c1`); cloud freezing is kept across the slope
  and exponential ranges by evaluating the gamma moments in log space, where
  the direct sixth power overflowed at ordinary cloud slopes and erased the
  freezing (`3f111e162`); exceptional rain freezing stays within the joint
  donor budget (`c621693fa`); and an in-range number moment is not rebuilt
  during slope diagnosis (`c03dcfb59`). Taken here 2026-09-29 with the
  engine's own text on the move to the 2.8 engine; `MORRISON-CU-1` is again
  the only difference in that kernel.
* The land-use rulebook's soil match: a land cell over the water soil
  category keeps its land use as its vegetation and takes silty clay loam, as
  real.exe matches it, instead of becoming mixed forest in the final pass.
  Engine fix `b69ac2b52`, 2026-09-27; taken here 2026-09-29 with the engine's
  own text. It reaches this model through the global statics door.
* The cumulus scheme's undilute buoyancy integral dropped the cloud-base
  layer itself, where the reference skips only the layers strictly below it
  (`module_cu_gf_deep.F:3024`, `IF(K.LT.KBCON(I))`), so the deep and the
  shallow closures each read one layer short. Engine fix `ab874b32e`,
  2026-09-04; taken here 2026-10-05 with the engine's own text, which retires
  the row that tabled it. The committed oracle capture is byte-identical
  either way; a device probe compiled from the shipped source now reads the
  reference's integral on a column whose only buoyancy sits at cloud base.
* The boundary layer's thermal-enhanced Richardson sweep ran unguarded and
  then recomputed the boundary-layer flag from its result, where the reference
  wraps the whole sweep in that flag (`bl_ysu.F90:703-728`) and nothing above
  it can raise the flag, so a column whose first-guess top sat below the first
  interface was promoted into the convective regime. Engine fix `4d523b793`,
  2026-09-04; taken here 2026-10-05 with the engine's own text in the kernel
  and in the mirror (the diagnosis helper's clamp switch and the guard),
  applied on top of this package's own clearing block, which retires the two
  rows that tabled it. On the engine's measured onset column the kernel and
  the mirror now read index 1, height 95.43 m, countergradient 0 and a peak
  heat exchange of 0.01 m2/s, where both read index 4, 431.15 m and 22.0 m2/s
  before. `NPREF-5`'s grade was measured under the unguarded sweep and should
  be re-read.
**Open.** In rough order of what they cost to take:

1. `SFCLAY-PY-1` with `SFCLAY-CU-1` and the inventory half named in
   `INVENTORY-3`, as one three-file change or not at all.
2. `NPREF-11`, the shallow cumulus tendency divisor, which is bookkeeping and
   closes a flawed instrument.
3. `RRTMGP-3` and `RRTMGP-9`, each of which reaches one more engine module; `RRTMGP-4`; `RRTMGP-14`, which has to be re-applied over the
   chunked loops; `RRTMGP-6`, as a targeted deletion of one entry and not a
   copy of the engine's now empty table.
4. `PHYSICS-1` with `INVENTORY-1` and `INVENTORY-2`'s names, together or
   neither; then `PHYSICS-2`, `PHYSICS-4`, `PHYSICS-5`, `PHYSICS-7` and
   `PHYSICS-9` and `PHYSICS-18`, none of which moves a number this model
   produces and all of which close a wrong answer or a false refusal inside
   carried code.
5. `NOAH-PY-1`, one line in the cold-start soil liquid water, reached by no
   door.
6. `RRTMGP-23` with `LOADER-3`, and `NOAH-PY-2`: per-card and per-stream
   caches, and a value-keyed table cache. Nothing numeric; they matter when
   one process steps bands on more than one card or stream, or runs many
   members. All three need `gpuwm.core.device_cache`, which the engine
   first ships at 2.8.1, so they raise the floor to 2.8.1 or carry it.
7. `SFCLAY-CU-4`, `YSU-CU-6`, `NOAH-CU-1`, `MORRISON-CU-2` and `RTE-CU-2`,
   the constant-division respelling, as one change across the five kernels:
   the only pull here that moves bits, on Blackwell cards, by a unit in the
   last place per affected quotient, so it lands with the kernel digests and
   the bit-identity anchors re-recorded on a Blackwell card.

Rows owed: `SFCLAY-PY-1`, `SFCLAY-CU-1`,
`NPREF-11`, `RRTMGP-3`, `RRTMGP-4`, `RRTMGP-6`, `RRTMGP-9`, `RRTMGP-14`,
`PHYSICS-1`, `INVENTORY-1`, `PHYSICS-2`, `PHYSICS-4`, `PHYSICS-5`,
`PHYSICS-7`, `PHYSICS-9`, `PHYSICS-18`, `NOAH-PY-1`, `RRTMGP-23`,
`LOADER-3`, `NOAH-PY-2`, `SFCLAY-CU-4`, `YSU-CU-6`, `NOAH-CU-1`,
`MORRISON-CU-2`, `RTE-CU-2`. That is the whole of what is owed; every other row
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
* `NTIEDTKE-PY-3`. The per-column grid spacing, which the engine's own driver
  already publishes and its other cumulus scheme already reads. Owner commit
  `92e58c4ff`, 2026-09-02. (`NTIEDTKE-PY-2`, the pipeline cache key, is no
  longer offered: the engine closed the same defect its own way on
  2026-09-27.)
* `GF-PY-4`. The chunked cumulus seam, which the engine would want at the same
  batch size its own seam cannot allocate. Owner commit `a74805618`,
  2026-09-01.
* `GF-CU-8`. The per-column closure reading, if the engine wants the same
  instrument. Owner commits `fca73b086`, 2026-09-04, and `957059111` and
  `58fd47304`, 2026-09-05.
* `GF-CU-10`. The cumulus rain sign: the reported rain is the updraft's
  condensate less the downdraft evaporation the cloud water could not supply,
  where the reference and the engine add that remainder instead. The engine's
  regional runs with this scheme over-report their convective rain the same
  way. Package commit, 2026-10-05.
* `RRTMGP-15`, the k-distribution temperature floor, which is inert wherever
  air stays above it.
* `RRTMGP-26`, one longwave layer above a model top under 2 hPa, where
  WRF's 4 hPa rule rounds to none and the engine transcribes the count
  without WRF's moved top interface. Package commits `9df6bf6` and
  `9d4ea0f`, 2026-10-05.
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
