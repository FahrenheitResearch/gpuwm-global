gpuwm-global 0.1.5

Arwen Global 0.1.5 hands a global forecast to the regional engine: one
command writes it as the initial and lateral boundary data of a regional
run, with nothing changed on the regional side. Render tapes also carry the
eight variables a reader of WRF history files asks for by name. The model
itself is 0.1.4's, and it needs `gpuwm>=2.9.13,<2.10` as 0.1.4 did.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- `gpuwm-global export-mapped-parent` writes a forecast as the regional
  engine's mapped NetCDF source: one model-level frame per checkpoint on a
  regular lat-lon window, with the mapping, composition and provenance
  documents and a receipt that carries the `gpuwm prep --source mapped`
  command for them. `docs/ARWEN_GLOBAL_REGIONAL_PARENT.md` has the commands,
  the file contract and what is not carried.
- One measured run of that feed: a T383 forecast feeding a 3 km, 450 x 360
  regional run (gpuwm 2.9.13) for 48 h on one RTX 5070 Ti 16 GB. Seventeen
  frames exported in 219 s, the preparation took 59 s on the CPU and the
  forecast 1620 s. The boundary row stays within 0.2 K RMS of the parent
  frame at all sixteen updates while the interior moves 0.5 to 0.6 K and
  1.3 to 1.7 m/s away from it, and at least 20 points from the edge the
  500 hPa height correlates with the parent at 0.994 or better through
  48 h.
- `tools/arwen_global_regional_parent_check.py` measures whether a regional
  run is fed by its parent, by distance from the lateral edge. No device.
- `gpuwm-global export` writes eight more variables on every render tape:
  `F`, `E`, `MAPFAC_MX`, `MAPFAC_MY`, `LU_INDEX`, `IVGTYP`, `XLAND` and `W`.
  Without the map factors a reader differentiates a lat-lon tape as if it
  were unstretched, and without the land-use planes it reads every column
  as land. `docs/ARWEN_GLOBAL.md` has the table.
- `W` on a tape is the air's vertical velocity relative to the ground in
  two of its three parts: the flux through the model's coordinate surfaces
  plus the wind's climb along them. At the ground it is the lowest layer's
  wind times the terrain slope. The variable's `description` and the tape's
  `ARWEN_VERTICAL_VELOCITY` attribute say what it carries.
- The tape names the land-use table its categories index (`MMINLU`,
  `ISWATER`, `ISICE`, `ISURBAN`). `ISLAKE` is -1, because the model folds
  lakes into the water category.

## What the regional run did not get right

- Cloud and precipitation at the boundary. With frames 3 h apart the
  regional engine put 36 % of the parent's condensate on the boundary row.
  Inflow air arrives short of cloud and the regional model re-forms it
  within about six points.
- The outflow side. Precipitation within about 50 points of the outflow
  edge is not a forecast: about three times the parent's rain 5 to 19
  points in, and less than half of it 40 to 50 points in. A boundary zone
  twice as wide did not remove it.
- Two defaults of the regional engine reach outside the frames: a lake
  surface analysis at start-up and, from 48 h, an ocean mixed-layer model
  started from a climatology. `lake_state_source = "forcing"` and
  `sf_ocean_physics = 0` in the regional TOML turn both off.
- Number concentrations, deep soil temperature and the regional grid's
  static geography are not carried. The regional run takes its deep soil
  temperature from its geography tree, a reanalysis climatology: in the
  measured run it was 2.4 K warmer on average than the global model's own,
  and the preparation moved the lowest soil layer toward it by 0.16 K RMS.
- It is one case, and it was not verified against observations. Frames of
  the whole globe have been written only at test size, and a window across
  the date line is refused.

## Known limits of the tape's `W`

- It leaves out the rise or fall of the coordinate surfaces at a fixed
  place, which one checkpoint does not determine. On the one T383 tape
  measured that part is zero at the ground and under 4 % of `W` up to
  100 hPa, 30 % at 10 hPa, and as large as `W` near 1 hPa. Above about
  10 hPa `W` is not the velocity relative to the ground.
- On that tape `W` agrees with an estimate differenced from the tape's own
  heights and winds to 0.0012 m/s RMS (correlation 0.9993) between the
  ground and 120 hPa.
- The regional parent frames keep the flux through the coordinate surfaces
  alone, zero at the ground, as the measured run was fed.
- `F` and `E` use the rotation rate the model integrates with,
  7.292e-5 rad/s. WRF's own constant is 7.2921e-5.
- With `W` on the tape, `gpuwm render` of engine 2.9.13 offers nine more
  products. None has been drawn and checked on a global tape, and the
  render catalogue was not measured again.

## Unchanged

- The engine range, `gpuwm>=2.9.13,<2.10`, and the 48 pinned seam files.
- The ten Rust doors. `gpuwm-global fetch-doors` stages the bundle 0.1.3
  published.
- The model. The source files that differ from 0.1.4 are the two exporters
  and the command line, so config hashes do not change, and every variable
  a 0.1.4 tape carried is the same bit for bit.
