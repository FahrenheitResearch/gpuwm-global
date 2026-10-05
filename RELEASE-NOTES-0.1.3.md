# gpuwm-global 0.1.3

WOOF Global 0.1.3 fixes how the model handles water. On 0.1.2 the default T255
forecast from a GDAS analysis could stop near hour 117 with a water-closure
refusal, and it rained about 1.7 times what the gauges measured. This release
fixes both. It also fixes the forecast clock and the top-level cooling, and
moves the last NumPy data-path steps into Rust. It needs
`gpuwm>=2.8.0,<2.9` and is proven against `gpuwm 2.8.5`.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Scores against observations

T255, 48 h from GDAS 2026-10-02 00Z, on one RTX 5090. Rain is averaged over
1 degree boxes and given as the model total divided by the observed total.

| Rain, model / observed | Before (0.1.2 build, audit base arm) | 0.1.3 |
|---|---|---|
| CPC gauges, land 60S to 60N, 24 to 48 h | 1.72 | 1.20 |
| MRMS 24 h QPE, CONUS, 12 to 36 h | 1.80 | 1.26 |

The 0.1.3 column scores the release tree with every fix in it. Against CPC
its 1 mm equitable threat score is 0.358. Without the lid fix the same tree
scored 1.21 and 1.28, and its 25 mm frequency bias against CPC was 1.51,
down from 2.43.

| Top-level potential temperature drift (K/day) | Before the lid fix | 0.1.3 |
|---|---|---|
| GDAS 2026-09-01 00Z, 24 h | -59.4 | -10.6 |
| GDAS 2026-10-02 00Z, 48 h | -47.5 | -8.1 |

A 240 h T255 forecast from GDAS 2026-09-01 00Z now passes every gate. On
0.1.2 a 120 h T255 forecast from GDAS 2026-09-30 12Z stopped at hour 117.

## Fixed

- **Open water starts at a water temperature.** The cold start took the open
  water skin from a plain regrid that mixed in nearby land. On the GDAS
  2026-09-30 12Z analysis that made one lake column start at 321 K. That
  column evaporated near 3,000 W/m2 and emptied its surface reservoir at hour
  117. Open water now takes the analysis's own water points, the rule metgrid
  applies to SST.
- **Inland lakes follow their own heat budget.** Before, they held their
  starting temperature for the whole run.
- **Convective rain matches the water that left the atmosphere.**
  Grell-Freitas added downdraft evaporation to the rain instead of
  subtracting it. Its reported rain also no longer feeds the land surface
  beyond the water the call actually removed.
- **The forecast clock is the analysis valid time.** Solar geometry and the
  dated greenhouse gases follow the analysis. A stated `start_time_utc` that
  disagrees with the analysis is refused. A config that omits it now runs to
  the end: before this fix it integrated the whole forecast and then failed
  while writing the receipt.
- **The longwave radiates the air above a 1 hPa model top**, as WRF's RRTMG
  does. A top of 2 hPa or more is unchanged.
- **RRTMGP runs with dated NOAA greenhouse gases.** It had run with a fixed
  CO2 of 369.55 ppm.
- **The stratospheric temperature floor is off by default.** The failure it
  guarded against does not reproduce.
- Grell-Freitas keeps its cloud-base layer in the undilute buoyancy integral.
  YSU runs its thermal-enhanced Richardson sweep only where `pblflg` is set.
- Semi-Lagrangian core:
  - Vertical advection uses geometric full-level spacing.
  - The departure level is read correctly at the lid.
  - The fold gate stops a forecast only on a real fold, not on shear. T533
    at the shipped 300 s step completes 120 h.
- The tracer mass fixer is Bermejo-Conde in its IFS form. It no longer
  rescales a whole species by one factor.
- An analysis whose top lies below the model lid is extended upward instead
  of held constant.
- IFS soil layers are remapped onto Noah's layers by depth.
- Assimilation:
  - The 2 m temperature operator is the model's own surface layer.
  - The 2 m temperature gain is the optimal-interpolation gain of that
    operator's sensitivity.
- Render tapes write water as WRF's dry mixing ratio, take their valid time
  from the run's clock, and name the model WOOF Global.
- A forecast stopped by a refusal still delivers the maps for the hours it
  reached.
- Multi-card runs split the physics and the semi-Lagrangian work across
  cards and use NCCL on the device.
- `gpuwm-global doctor` exits 0 on a correct install.

## Added

- `gpuwm-global score RUN_DIR --out DIR` scores a finished run of any
  truncation against ASOS/METAR stations and IGRA2 radiosondes through the
  Rust decoders. GFS and IFS analyses are scored only as secondary
  references.

## Changed

- The cold-start regrid, the render-tape regrid and the regional
  translation run in Rust: `rw_global_coldstart` and
  `global_render_kernels`, built from this package's own `rust/` workspace.
  Both give the same output as the NumPy they replace, bit for bit,
  measured on Linux with AVX2.
- The engine seam and the carried-physics divergence baseline are proven at
  `gpuwm 2.8.5`, the engine a fresh install resolves.

## Door bundles

Attached to this release and pinned inside the wheel:

- `arwen-global-doors-v0.1.3-linux-x86_64.zip`
- `arwen-global-doors-v0.1.3-win-x86_64.zip` (built for
  `x86_64-pc-windows-gnu`)

The eight observation doors are the v0.1.2 binaries, built from engine
commit `0164ae0f2d70` (tag `v2.8.0`). The two new libraries are built from
this package's `rust/` workspace.

The full list of changes and measurements is in `CHANGELOG.md`.
