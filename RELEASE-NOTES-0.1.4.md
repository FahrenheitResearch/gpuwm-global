gpuwm-global 0.1.4

Arwen Global 0.1.4 runs on the gpuwm 2.9 engine and fixes the surface
pressure drift, the top-level cooling and the frozen sea surface. T383 now
runs resident on a 16 GB card, and T533 runs there too with two flags. It
needs `gpuwm>=2.9.13,<2.10`.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- Runs on gpuwm 2.9.13.
- Dated SST, sea ice and greenness. `gpuwm-global surface-series` builds a
  series on the model grid from GDAS analyses or from one GFS cycle's own
  forecast, and `run --surface-series` applies it: SST every step, sea ice
  at each record, greenness daily.
- The radiation reads the CAM zonal-monthly ozone climatology by date,
  latitude and pressure. `ozone = "rfmip-median"` keeps the old profile.
- T383 (35 km) on a 16 GB card, resident, one band: a 10-day forecast in
  109 minutes on an RTX 5070 Ti.
- T533 (25 km) on a 16 GB card with `--streaming on --device-allocator async`.

## Fixed

- Surface pressure drift in float32: the dry 9-day Jablonowski-Williamson
  test drifts 0.28 hPa, down from 1.51.
- The top level no longer cools 6 to 9 K a day: log-spaced radiation layers
  above the lid, CAM ozone, the absorber's kinetic energy booked as heat, and
  a zonal-mean top drag on by default in native-physics forecasts
  (`[sponge] mean_drag_base_pa = 0` turns it off).
- SST and sea ice no longer stay frozen at their cold-start values.
- `--streaming on` no longer rebuilds the Legendre tables on every transform.
- The host tier stages bands with one async copy instead of a single-thread
  gather: T533 on a 16 GB card went from 18.9 to 6.7 s a step.
- The CPU test tier finishes under pytest-xdist.

Config hashes of the native-physics experiments change in this release,
because the ozone default changed.
