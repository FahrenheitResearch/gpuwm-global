# Morrison droplet number: one value, or a land and an ocean value

`physics.native_adapter_options.morrison_droplet_number` chooses how many
cloud droplets Morrison assumes.

| Value | What it does |
|---|---|
| `"fixed"` (default) | WRF's own arm. Every call resets the droplet number to `NDCNST = 250` per cm3 (`module_mp_morr_two_moment.F:271-276`, reset at `:1493-1495`). Results are bit-identical to the WRF port. |
| `"land_ocean"` | A **declared departure from WRF's Morrison**. Each column gets its own droplet number: `morrison_droplet_number_land_cm3` (default 300) on a land column, `morrison_droplet_number_ocean_cm3` (default 100) on every other column. |

```toml
[physics]
native_adapter_options = { ..., morrison_droplet_number = "land_ocean" }
```

The two values can be set alongside the mode
(`morrison_droplet_number_land_cm3 = 300.0`,
`morrison_droplet_number_ocean_cm3 = 100.0`). Each must be between 1 and
2000 per cm3. Under `"fixed"` the two values are never read, and they stay
out of the configuration identity. A checkpoint written before the option
existed therefore keeps its hash.

## Why the number matters

WRF's 250 per cm3 is a continental number. Over the ocean, marine air holds
far fewer cloud condensation nuclei. Morrison uses the droplet number in three
places, and all three move together.

1. **Droplet size, and so cloud brightness.** At a fixed amount of water,
   fewer droplets are larger, and RRTMGP's liquid optics read the larger
   effective radius (`effc`). With a fixed shape, `effc` scales as
   `N^(-1/3)`. Morrison also narrows the spectrum as the number falls (its
   dispersion relation at `F:1558`). The shape parameter is about 8.3 at 100
   per cm3 and 4.8 at 250, so the rise is smaller than the bare cube root.
   At fixed water, 100 against 250 per cm3 gives `effc` a factor of 1.29,
   not 1.36 (`tests/test_arwen_global_morrison_droplet_number.py`).
2. **Drizzle.** Khairoutdinov-Kogan autoconversion goes as `N^-1.79`
   (`morrison.cu:413-414`). At 100 against 250 per cm3, rain forms 5.16
   times faster from the same cloud water. The cloud keeps less water,
   which is a second, separate effect on how bright it is.
3. **Cloud-droplet fall speed.** Morrison sediments cloud droplets at their
   Stokes speed (`F:3440-3441`), and larger droplets fall faster. This
   matters most for fog and for cloud at the lowest level.

Below 233 K the droplets freeze homogeneously. The ice number they make is
the droplet number, so the option also touches the size of ice formed that
way.

## Where the values come from

The values are taken from the sources below, not tuned to an albedo.

- **MPAS-Thompson**, NCAR's global model running WRF physics: 300 per cm3
  where `landmask = 1` and 100 per cm3 where it is 0
  (`mpas_atmphys_init_microphysics.F`, `init_thompson_clouddroplets_forMPAS`).
  These are the defaults here.
- **WRF's own Thompson scheme** (`module_mp_thompson.F`, WRF 4.7.1,
  lines 80-88) gives the same guidance beside its single `Nt_c = 100.E6`:
  "roughly 100 per cc for Maritime cases and 300 per cc for Continental".
- **CCPP (GFS) Thompson** uses 50 per cm3 over ocean and 150 per cm3 over land
  (`Nt_c_o`, `Nt_c_l`, chosen by `lsm == 1`). That is a second published
  pair, and it can be set through the two keys.

## Which columns count as land

A column is land where the model calls it land: `land_fraction` above one
half. That is `xland < 1.5` on the ice-free plane, the same split the surface
layer, Noah and Grell-Freitas use. The rest of the decision follows from it.

- **Coast cells** take whichever side of that split they fall on. The
  published prescriptions are binary too.
- **Sea ice** takes the ocean value. The air over the pack is marine air.
  Both MPAS (`landmask`) and CCPP (`lsm == 1`) key on a land mask that does
  not include ice.
- **Lakes** follow `land_fraction`.

## What it measured

This is one case: a T255 L40 native forecast from the GDAS 2026-11-09 18Z
analysis, 48 h, with the option off and on and everything else equal. It ran
on an RTX 4090 on 2026-11-12. The readings are hours 24 to 48 from the run's
own radiation time integrals (`arwen_global.radiation_scorecard
--window-hours 24 48`).

| Reading | Off | On | Change |
|---|---|---|---|
| Reflected shortwave at the top, global (W/m2) | 134.85 | 128.31 | -6.54 |
| ... over ocean | 143.07 | 133.81 | -9.26 |
| ... over land | 114.61 | 114.77 | +0.16 |
| ... 60S-30S | 201.86 | 188.72 | -13.14 |
| ... 30S-0 | 153.54 | 143.90 | -9.64 |
| Planetary albedo, global | 0.3868 | 0.3680 | -0.0187 |
| Outgoing longwave, global (W/m2) | 235.76 | 236.00 | +0.24 |
| Net at the top, global (W/m2) | -21.95 | -15.65 | +6.30 |
| Total cloud cover, global | 0.7175 | 0.7166 | -0.0009 |
| Grid-mean liquid water path (g/m2, mean of the 24-48 h checkpoints) | 110.0 | 95.8 | -13% |
| Morrison `effc` median where liquid (um) | 3.96 | 4.57 | +15% |
| Precipitation, global (mm/day) | 2.867 | 2.892 | +0.026 |

Against the GDAS analyses at 24 h and 48 h, the global RMSE moves by
-0.013 m and -0.007 m for 500 hPa height, and by -0.014 K and +0.0003 K for
850 hPa temperature.

Cloud cover did not move. The brightness fell through two channels: larger
droplets, and less liquid water from the faster drizzle. CERES-EBAF 4.2.1
puts the November global albedo at 0.303. This one change closes about a
fifth of the gap from this case to that value. It is not the whole of the
albedo error.

## How the identity is kept

The option is built as a separate kernel. `MORR_DROPLET_PLANE` compiles a
version of `morrison.cu` whose process and finalize stages read a
per-column plane. `launch_morrison` uses that build only when it is handed a
plane, and the runtime hands it one only under `"land_ocean"`. Without the
build flag, both resets read WRF's literal 250 per cm3. The plane build
handed 250 per cm3 everywhere is bit-identical to the WRF build
(`tests/test_arwen_global_morrison_droplet_number_card.py`). In the divergence ledger the change is
`MORRISON-CU-4`, `MORRISON-PY-4` and `NPREF-16`.
