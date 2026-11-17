gpuwm-global 0.3.0 (ArWen Global)

ArWen Global 0.3.0 runs on the gpuwm 3.0 engine line. It needs
`gpuwm>=3.0.3,<3.1`, so it installs in one environment beside gpuwm-hex 0.6
and the gpuwm 3.0.3 they both use. gpuwm 3.0.3
removed the engine module the ensemble filter borrowed its arithmetic from;
that arithmetic now ships inside this package, unchanged. The name is
spelled ArWen Global in the README, the documentation, the command-line
help and on its maps; the package, import name, command and config names do
not change. No config changes, and the Rust doors are 0.2.5's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Changed

- **The engine pin moves to gpuwm 3.0.3.** On gpuwm 3.0.3, 0.2.5 cannot
  start: the engine no longer has `gpuwm.da.letkf`, and every command,
  `doctor` and `--help` included, fails at import. The eight names the
  filter took from it are carried in `arwen_global/da/letkf_core.py`,
  verbatim from gpuwm 2.9.13.
- **The engine files the carried physics reaches are re-pinned at 3.0.3**
  (12 of 49 moved; the CUDA defines every carried kernel starts with did not).
- **Maps say ArWen Global and this package's version** in their source
  line, instead of the engine's "WOOF" and the engine's version.

## Measured

- **Forecasts change, for two engine reasons only.** On the shipped T255
  day (GDAS 2026-09-01 00Z, RTX 5070 Ti) 0.3.0 on gpuwm 3.0.3 runs as
  fast as 0.2.5 on gpuwm 2.9.13 (0.896 against 0.892 s a step, 6.22 GiB
  live on both) and its start differs in two ways, both from the engine:
  the land use and soil at 180 degrees, where gpuwm 3.0 repairs bad
  pixels in the WPS_GEOG land-use data, and a one-ulp change in the
  monthly greenness, leaf area and albedo, which gpuwm 3.0 now blends in
  single precision as WRF does. With those two put back, 0.3.0 reproduces
  0.2.5 bit for bit over the whole day.
- Every shipped config keeps its 0.2.5 hash.
