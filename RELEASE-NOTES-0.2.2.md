gpuwm-global 0.2.2

Arwen Global 0.2.2 is 0.2.1 with two fixes to the coupled slab ocean,
three experimental switches for shallow convection, a way to move a
held-sea run onto an ocean rule from one of its checkpoints, and a fix
for `gpuwm-global score`. The data assimilation and the worlds preflight
now accept the deeper air the radiation has accepted since 0.2.1. Every
shipped config keeps its hash, and a forecast that names no new option is
0.2.1's forecast, checkpoint for checkpoint, unless it runs the coupled
slab ocean (`[ocean] mode = "slab"`), which takes both ocean fixes with no
option. A data assimilation cycle whose analysed surface pressure lands
between 1,096.6 and 1,150 hPa now applies the analysis where 0.2.1 kept
the background. It needs
`gpuwm>=2.9.13,<2.10`, as 0.2.1 did, and the Rust doors are 0.2.1's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## Fixed

- **The coupled slab ocean no longer grows the southern summer sea ice.**
  This is `[ocean] mode = "slab"`; the default ocean holds the sea surface
  and is not touched. Under the ice, the heat-flux correction charged the
  pack with the heat the calibration's open leads had taken in, even where
  the model's pack was far more closed than the climatology's. In a
  180-day coupled run that sink was -78 to -90 W/m2, and the southern ice
  volume went from 7.24 to 11.04 thousand km3 between 1 December and
  28 January, while the real summer pack melts. A cooling correction under
  ice is now charged only in the pack's open share. The same run, resumed
  from its 30 November checkpoint with the fix, took the southern volume
  from 7.21 to 2.52 thousand km3. Its 15 % ice area fell 53 %, as the
  climatology's did (without the fix it fell 11 %). The northern ice area
  at 28 January moved by 0.07 million km2. That run is the long-run
  line's, with this release's ocean code but that line's atmosphere.
- **The coupled slab ocean's correction no longer jumps at mid-month.**
  The climatology's monthly tendency now comes from a smooth curve through
  the monthly planes instead of straight lines between them. In the same
  run the ocean mean of the correction jumped by 12 W/m2 at the 16 December
  and 16 January instants; now it moves by 0.04 and 0.08 W/m2 there.
- **`gpuwm-global score` works again.** In 0.2.0 and 0.2.1 it stopped at
  the first reference analysis with `NameError: name 'decode_reference' is
  not defined`, so no scorecard was written.
- **The data assimilation and the worlds preflight use the radiation's
  ceiling.** Since 0.2.1 the radiation accepts layers down to 1,150 hPa
  (its gas tables end at 1,096.6 hPa, and a layer between the two is looked
  up at the tables' deepest pressure). The ensemble's member check, the
  analysed-state bound, the twin's start check and the worlds preflight
  still refused anything past 1,096.6 hPa. They now use 1,150 hPa. Nothing
  changes for a state under 1,096.6 hPa.

## New

- **Three shallow-convection switches** (experimental, all off by default).
  Over the trade-wind oceans the model reflects too much sunlight, from a
  sheet of low cloud under the trade inversion that shallow convection
  should break up. The Grell-Freitas shallow arm exits in most of those
  columns because of one sum in WRF v4.6.1 that the later Grell-Freitas
  lines (GSL's GFL, MPAS v8.4.2) write differently. In
  `native_adapter_options`:
  - `gf_shallow_buoyancy_from_cloud_base = true` sums the plume's buoyancy
    from cloud base, as GFL and MPAS do;
  - `gf_shallow_blqe_to_cloud_base = true` sums the boundary-layer closure
    over the layer below cloud base, as GFL does;
  - `gf_shallow_depth_min_m = 500` turns off shallow clouds less than
    500 m deep, which keeps the stratocumulus decks.
  - Measured: 10-day T255 runs from GDAS 2026-08-01 00Z against CERES,
    with all three set, on the albedo-pair config of the 90-day runs. The global excess of reflected sunlight went from
    +14.9 to +3.7 W/m2, the tropical ocean's from +30.1 to +7.9, and net
    top-of-atmosphere from -15.0 to -5.4. Z500 RMSE was 25.14 m at 72 h
    against 24.90, and the 48-72 h rain against GPCP improved (RMSE 10.63
    against 10.98 mm/day). Still open: the NE Pacific trades (+43 W/m2),
    summer midlatitude land (5.5 W/m2 less reflection than before), and
    this is one August case. `docs/trade-shallow.md` has the details.
  - Measured: a 30-day December run on the same config, from GDAS
    2025-12-16 00Z, with all three set against none. The trades vent as
    in August (the tropical ocean's excess +3.3 against +26.0 W/m2, the
    global +5.8 against +16.3), but the Peru stratocumulus deck darkens
    from +8.5 to -28.8 W/m2: in the southern summer the depth minimum
    does not keep the decks.
  - With the switches off the kernel is WRF's, word for word. Setting one
    changes the config hash.
- **A held-sea run can move onto an ocean rule from one of its
  checkpoints**: `gpuwm-global run RULE_CONFIG --restart
  RUN/arwen_global_stepN.npz --branch-from-other-config`, between hold,
  anomaly and climatology. The branch's first step writes the sea surface
  a straight run under the rule writes at that instant, and its
  checkpoints carry the new rule. The slab and the calibration cannot be
  branched onto (the parent does not carry their arrays). On the 50-day
  T383 run of 2026-11-12, branched at day 25 onto the anomaly rule, the
  sea off Tampico read 24.4 C against the held 26.25 C, and the day's
  largest open-water evaporation fell from 55.0 to 46.6 kg/m2.

## Measured

- **Defaults unchanged.** On an RTX 4090, the shipped T255
  semi-Lagrangian config run 24 h from a cold start wrote all nine
  checkpoints byte-identical to 0.2.1's, every array and the archive
  hash, and a restart from 0.2.1's 12 h checkpoint wrote 0.2.1's
  checkpoints to 24 h. That config holds the sea, so the slab ocean is not
  on its path. Every shipped config keeps its 0.2.1 hash.
- The CPU test suite passed: 3434 passed, 86 skipped, 0 failed.

`CHANGELOG.md` has the details.

## Upgrading

From 0.2.1: `pip install --upgrade "gpuwm-global[gpu-cu13]"`. The doors
are unchanged (the v0.1.3 bundle), so no re-fetch is needed. The config
hashes are 0.2.1's, so a 0.2.1 checkpoint restarts under 0.2.2. Unless it
is a coupled slab run, it continues as it would have under 0.2.1. A
coupled slab run restarted under 0.2.2 takes the new under-ice rule and
the smooth tendency from its next step on.
