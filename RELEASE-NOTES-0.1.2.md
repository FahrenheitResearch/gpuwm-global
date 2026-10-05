# gpuwm-global 0.1.2

Arwen Global on the published ArWen engine 2.8.0. The package now needs
`gpuwm>=2.8.0,<2.9`, the door bundles are built from the engine's public
`v2.8.0` tag, and the carried Morrison microphysics and land-use rules take
the engine's fixes.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Fixed

- Global statics builds on the 2.8 engine. The rows grid answers the
  sampling handle the 2.8 static builder asks for; without it every statics
  build stopped on its first sector.
- RRTMGP reads its trace-gas and ozone table through the engine's loader,
  because the 2.8.0 data companion no longer ships the RFMIP input file. The
  radiation input is unchanged, bit for bit. The RFMIP oracle fetches its
  input through the engine's pinned route.
- Morrison microphysics takes the engine's four kernel fixes: vapor from the
  final cleanup is kept, cloud freezing no longer overflows, rain freezing
  stays within its donor budget, and an in-range number moment is not
  rebuilt.
- A land cell over water soil keeps its land use and takes silty clay loam,
  as real.exe does. Glacier cells still take the ice soil.
- The `rw_asos` door row carries the engine's 2.8 `--abi` line (the v2
  surface record).
- The door bundles carry `THIRD-PARTY-LICENSES.txt` and no build-machine
  paths.
- No shipped file names a machine or a private case. The statics cache
  sidecar records the kind of machine, not its hostname.
- `NOTICE` credits ITU-R P.676-13 and states what of CRTM ships, and the CRTM
  reference driver the docs name now ships in `tools/crtm_reference/`.

## Changed

- Engine range `gpuwm>=2.8.0,<2.9` and `gpuwm-data>=2.8.0,<2.9`.
- Engine seam re-pinned at `gpuwm 2.8.0`: 47 of 47 files, identical in the
  PyPI wheels and the public `v2.8.0` tag.
- The run manifest carries a `physics` block, as the engine's does.
- Carried-physics divergence rows re-baselined on `gpuwm 2.8.0`: 428 hunks,
  every one classified.

## Door bundles

Attached to this release and pinned inside the wheel:

- `arwen-global-doors-v0.1.2-linux-x86_64.zip`
- `arwen-global-doors-v0.1.2-win-x86_64.zip` (built for
  `x86_64-pc-windows-gnu`)

Both are built from engine commit `0164ae0f2d70` (tag `v2.8.0`).

## Found after release

These hold for the published 0.1.2. Each is stated with the measurement
that found it; the release that changes it says so in its own notes.

- `gpuwm-global doctor` exits 1 on a correct install against every published
  2.8 engine. It grades `preflight.measured_free_vram_bytes` and
  `surface_bias.interpolate_to_tape`, which no published engine carries and
  no documented command reaches, as gaps. Every command still runs; only the
  exit code is wrong. Fixed in 0.1.3.
- `gpuwm 2.8.5`, published inside this release's range, is what a fresh
  install resolves (2026-10-05). Against it the physics menu rows lack the
  engine's new `urban_scheme_id` field (fixed in 0.1.3), and the engine seam
  proves 22 of its 47 pinned files, not the 47 these notes state for 2.8.0.
- The default T255 forecast puts more rain on the ground than the model
  removes from the air. Grell-Freitas reports its downdraft evaporation as
  rain instead of subtracting it, an open-water column takes its skin
  temperature from a regrid that mixes in land points, and every lake holds
  its analysis skin temperature for the whole run. From the GDAS analysis of
  2026-09-30 12Z the reported convective rain was 1.77 times the water the
  scheme removed (2.28 times over land), four-day totals were 1.43 times
  CMORPH, 1.45 times CPC and 1.59 times MRMS, and the 120 hour run stopped
  at hour 117.3 on the surface-reservoir refusal from a lake that started at
  321.1 K (measured 2026-10-01).
- The radiation clock of the native physics is the configuration's
  `start_time_utc`, which each shipped native GDAS experiment fixes at the
  analysis it was built on (2026-09-01 00Z, or 2026-08-30 18Z for a few) and
  which `da fresh` does not change, so a forecast from a newer analysis runs
  its solar geometry on that date and hour. Nothing compares it with the
  analysis valid time.
