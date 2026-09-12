# gpuwm-global 0.1.1

Three fixes. One in the carried Noah land surface scheme: frozen
ground stopped limiting infiltration because the wrong one of WRF's two
frozen-ground quantities reached the limiter, so every column with soil ice in
it moves and every column without soil ice is unchanged bit for bit. One in the
carried Morrison microphysics: the sedimentation stage rebuilt two quantities
WRF holds still across it, so the cloud droplet fall speeds were wrong by about
0.29 percent per kelvin of each step's own temperature change, on every cloudy
level. And the third-party licence notices the physics this package carries has
always owed, which 0.1.0 shipped without.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- `docs/CARRIED-PHYSICS-DIVERGENCE.md`. The physics this package carries and
  the engine's copies of the same files move independently, and nothing here
  said which of the differences were on purpose. The document carries 82 rows
  over the 19 carried files that differ, each with the carried and engine line
  ranges, the common ancestor's state, the class, the commit on the side that
  moved, what changes numerically, and a decision: pull, refuse, offer or none.
  It closes with what this package owes the engine, what it has that the engine
  line may want, and what it holds a position on. Measured against the
  published `gpuwm 2.7.3`.
- `arwen_global/data/engine-divergence.json` and
  `tools/fingerprint_engine_divergence.py`. The same 375 differences keyed by a
  SHA-256 of each hunk's text on both sides, so a row survives a line shift and
  the engine line can verify the same difference from the row's engine line
  range. All 35 carried files are measured, the four WRF parameter tables among
  them.
- `tests/test_engine_divergence.py`. Against the installed engine it fails by
  file and line on a difference no row covers, and by row name on a row whose
  difference is gone. Without an engine it still holds the document's tables
  and the shipped rows to each other, and holds the measurement to the carve's
  own file list.

## Fixed

- The licence notices for what the carried physics transcribes. 0.1.0 shipped
  the CUDA kernels and four WRF parameter tables and carried no licence text
  for any of them. RTE+RRTMGP is BSD 3-Clause and clause 1 asks a source
  redistribution to retain the notice, the conditions and the disclaimer;
  AER's RRTMG McICA generator, which is what `core/kernels/rrtmgp_mcica.cu`
  transcribes whatever its filename suggests, is BSD 3-Clause under its own
  copyright; Arm's libm cores in `core/kernels/glibc_flt32.cuh` are MIT, which
  asks for its copyright and permission notice in every copy; FDLIBM's expm1f
  and lgammaf reduction in the same file carry a notice whose one condition is
  that it be preserved; UCAR asks that its notice travel with any copy of WRF.
  0.1.1 performs all five.
- Where they are. `licenses/` holds the texts, in the sdist and in the wheel
  at `gpuwm_global-0.1.1.dist-info/licenses/licenses/`. Beside the code:
  `core/kernels/LICENSE-third-party.txt` with the kernels and
  `data/noah_tables/LICENSE-WRF.txt` with the tables. `core/rrtmgp.py` and
  `core/npref.py` open with the notice for what they transcribe. NOTICE gains
  a section per work, each naming the carried file and the text that covers
  it.
- Said rather than left silent: the gamma routines at the end of
  `core/kernels/glibc_flt32.cuh` (`gfk_gamma_product`, `gfk_gammaf_positive`
  and `gfk_tgamma`) are this project's own work under its Apache-2.0 licence
  and not a transcription of any C library. `NOTICE` and the notice beside
  the kernels say so, and nothing in that block needs a third-party notice.
  The code is unchanged since 0.1.0; the comment text around it in
  `glibc_flt32.cuh` and `gf.cu` was rewritten to say whose work it is, so the
  assembled-source digests a run receipt records for the `gf` and `ntiedtke`
  modules differ from 0.1.0's for that reason alone.
- Noah's frozen-ground infiltration limiter did not limit. WRF's REDPRM builds
  two quantities, `FRZFACT = (SMCMAX/SMCREF)*(0.412/0.468)` and
  `FRZX = FRZK*FRZFACT`, and SFLX passes the second one down. The dummy
  argument that receives it is merely spelled `FRZFACT` at every level below
  SFLX, and SRT names it back to `FRZX` before spending it as
  `ACRT = CVFRZ*FRZX/DICE`. The carried CUDA kernel and the carried float64
  mirror had both followed the name instead of the argument and passed
  `FRZFACT`, so `ACRT` ran `1/FRZK` = 6.67 times large, the `CVFRZ` series in
  `FCR` saturated at one, and a frozen column infiltrated everything that fell
  on it. Both pass `FRZX` now, and the carried kernel is byte-identical to the
  published engine's copy again, which carries the same repair.
- The two ends had made the same substitution, so no self-consistency gate
  could see it: the mirror agreed with the kernel and both disagreed with WRF.
  The fix is graded against WRF's own driver and against the mirror recomputed
  by hand, not against either of them alone.
- Morrison's sedimentation stage rebuilt two quantities WRF holds still. WRF
  builds the cloud droplet Stokes coefficient `ACN(K) = G*RHOW/(18.*MU(K))`
  once per level at `module_mp_morr_two_moment.F:1438`, above the warm branch's
  small-particle melt, and the particle-size reference density
  `PRES(K)/(287.15*T3D(K))` at :3405; its sedimentation block runs after the
  column loop closes and before the tendency apply at :3710, so both are level
  values, not sedimentation-time ones. The carried kernel and the carried
  float64 mirror rebuilt both from the post-process temperature. The Stokes
  coefficient is the expensive half at -0.288 percent per kelvin, so both cloud
  droplet fall speeds were wrong by about that much per kelvin of the step's own
  temperature change, on every cloudy level.
- The process stage now publishes both and the sedimentation stage consumes
  them, which is what WRF's structure says. Handing the sedimentation stage its
  current temperature instead, the obvious-looking repair, would have moved PGAM
  about 2,400 times further from WRF than the error it removes.
- `morr_bound` builds PGAM's reference density from the current temperature
  rather than the density frozen at entry. This is the engine's own repair,
  taken with the engine's text, and it reaches the cloud droplet effective
  radius the radiation reads.

## What it changes

Measured against the unmodified WRF driver over the four switch fixtures the
port is graded on, RTX 3080, 2026-09-12:

| Field | Before | After |
|---|---|---|
| Surface runoff | 60,641,303 ULP | 2,812 ULP |
| Soil liquid water | 6,508 ULP | bitwise |
| Relative soil moisture | 4,729 ULP | bitwise |
| Soil moisture | 1,627 ULP | bitwise |

On the float64 mirror, one wet loam column at 266 K under 5 mm of rain:

| Quantity | Before | After |
|---|---|---|
| `ACRT` | 6.790 | 1.019 |
| `FCR` | 0.9653 | 0.0837 |
| Surface runoff | 4.356 mm | 4.488 mm |
| Top layer soil moisture | 0.41849 | 0.41717 |

SRT reaches the limiter only where the column's soil ice clears its own
`DICE > 1e-2` threshold, so in an early September case the change lives in
Antarctica, Greenland, the high Arctic and high terrain. A column with no soil
ice is unchanged bit for bit. Frozen-ground columns move under this repair,
so a forecast that reaches them is not byte-identical to one 0.1.0 wrote. No
shipped page quotes a ten-step identity hash; the ten-step comparison of this
distribution against the model source tree at the revision it carries, the
309-array acceptance 0.1.0 passed, was not taken for 0.1.1 and is owed as the
next card job.

Morrison, measured against the unmodified WRF driver over the 28 oracle columns
and 10,948 compared values, RTX 3080, 2026-09-12:

| | Before | After |
|---|---|---|
| Values disagreeing with WRF | 3,554 | 3,505 |
| Cloud water | 228 | 195 |
| Cloud ice | 333 | 327 |
| Accumulated rain (of 28) | 10 | 4 |
| Per-call rain (of 28) | 22 | 21 |
| Frozen fraction (of 28) | 12 | 9 |

No field is worse and every field's worst ULP distance is unchanged. On the
float64 mirror, one 900 hPa 285 K cloudy column that warms 4.04 K in a 60 s
step:

| Quantity | Before | After |
|---|---|---|
| Cloud droplet mass-weighted fall speed | 0.044494 m/s | 0.044986 m/s |
| Cloud droplet number-weighted fall speed | 0.020406 m/s | 0.020636 m/s |

Cloud droplet sedimentation touches every column that holds cloud water, so
this change is not confined to a climate band the way the frozen-ground repair
is: it moves wherever cloud water exists and the microphysics moves the
temperature.

- The engine record on the front page. The index published `gpuwm 2.7.3` on
  2026-09-12 and it is what the declared range resolves, so the page said the
  range resolved 2.7.2 while the new document said 2.7.3. It now names 2.7.3,
  records the boundary reading taken against it, and says what the engine seam
  does and does not cover.
- The engine seam, re-pinned against `gpuwm 2.7.3`. The pins were taken
  against 2.7.0, so on the engine the index actually serves `doctor` printed
  eight files as moved and two seam test nodes failed. The eight were read hunk
  by hunk first: of the 41 symbols this package imports out of them, 38 are
  byte-identical between the two engines, and the three that moved change no
  number. `RunConfig` and `DomainState` moved in comment text with no field,
  default or validation touched. `radiation_scheme_ids` narrowed a refusal, so
  a configuration that writes `ra_physics=4` beside
  `ra_lw_physics=4`/`ra_sw_physics=4` now resolves to the same `(4, 4)` pair
  instead of raising, while a contradicting pair is still refused. The boundary
  reading and the signature reading are the same on 2.7.3 as on 2.7.2: 231
  symbols across 65 modules with no gap, 28 signature rows. The two nodes that
  compare hashes now skip, naming both versions, on any other published 2.7,
  which is the rule the divergence gate already followed.

## Requires

Unchanged from 0.1.0: Python 3.11 or later, `gpuwm>=2.7.0,<2.8` with
`gpuwm-data` beside it, and a CUDA card for a real forecast. `gpu-cu13` and
`gpu-cu12` are the two CUDA majors.

Apache-2.0, with the third-party notices in `NOTICE` and the texts they point
at in `licenses/`.
