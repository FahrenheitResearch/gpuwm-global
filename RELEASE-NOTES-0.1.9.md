gpuwm-global 0.1.9

Arwen Global 0.1.9 is 0.1.8 as it was meant to reach PyPI. 0.1.8 was
tagged, but its publish workflow's test job failed on GitHub's ubuntu
runner and nothing was uploaded, so `pip install gpuwm-global` stayed at
0.1.7. Everything new in 0.1.8 is in 0.1.9: read
[RELEASE-NOTES-0.1.8.md](RELEASE-NOTES-0.1.8.md) for what it is and what it
measured. It needs `gpuwm>=2.9.13,<2.10` as 0.1.7 and 0.1.8 did, and the
Rust doors are 0.1.7's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## What changed from 0.1.8

One gate and the sentences that relied on it. No forecast bit moves and no
code path changes; every config hash is 0.1.8's.

- **`divergence_only_mass_flux` agrees to rounding on a CPU.** The
  experimental `[transforms]` switch narrows the Legendre call the
  mass-flux analysis makes, and whether a narrower call returns the same
  bits is the BLAS library's choice on the CPU it runs on. On the hosts
  measured it does, and the 0.1.8 test asserted that; on GitHub's hosted
  runners (ubuntu and Windows) it does not, and 19 or 20 of the 52 arrays
  of the ten-step numpy checkpoint moved. The test now holds the switch to rounding: every
  float array within 1e-9 of its own largest value, every other array
  equal. A one-ulp emulation of the difference moved the checkpoint by at
  most 7.0e-11 of an array's largest value. `skip_unread_planes` is still
  held byte for byte, and on the cards measured both switches together
  still write every array of the T255 day unchanged.

`CHANGELOG.md` has the change with its measurements.

## Upgrading

From 0.1.7: `pip install --upgrade "gpuwm-global[gpu-cu13]"`. The doors are
unchanged (the v0.1.3 bundle), so no re-fetch is needed.
