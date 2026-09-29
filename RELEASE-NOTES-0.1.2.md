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
