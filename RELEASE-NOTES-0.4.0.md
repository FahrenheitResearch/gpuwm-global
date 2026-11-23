gpuwm-global 0.4.0 (ArWen Global)

ArWen Global 0.4.0 needs `gpuwm>=3.0.5,<3.1`. New physics ships opt-in;
no default changes.

The README's install and quick start, from an empty machine to the drawn
maps of a 24 hour T255 forecast:

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global fetch-analysis --cycle 2026-09-01T00 --out cases/baseline-2026090100
gpuwm fetch-geog
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- `arwen_global_gdas_t255_l80_satmedmf_scinocca_24h`: the T255 day on 80
  levels with the SA-TKE-EDMF boundary layer and the Scinocca drag.
- `arwen_global_gdas_t255_l80_ysu_scinocca_24h`: its twin with YSU.
- `[vertical] coordinate = "ifs_class"`: 80 levels to a 1 hPa lid.
- `pbl = "satmedmf"` (single precision) and `pbl = "satmedmf_fp64"`.
- `nonorographic_gwd = "scinocca"`: non-orographic gravity-wave drag.
- `microphysics = "tempo"`: TEMPO.
- `radiation_aerosol = "tegen-wrf"`: WRF's aerosol climatology.
- Stochastic physics (SPPT, SKEB), `run --member` and `--ensemble-seed`.
- `radiation_stride_lat`, `radiation_stride_lon`, a second background
  hyperdiffusion and three Grell-Freitas option bits, off unless set.
- Kept cuFFT plans, fewer host syncs, batched radiation, a faster
  Grell-Freitas kernel and fewer exchange copies, with the same numbers.
- `gpuwm-global --version`.

## Changed

- The engine floor is gpuwm 3.0.5, and the engine files the carried physics
  reaches are re-pinned there.
- `doctor` reports an engine file that moved under the carried physics as a
  gap, with the engine version to install.
- `doctor` prints the folders this account uses and the variable that moves
  each one.
- The README has one quick start and a section on several accounts sharing
  one machine.

## Fixed

- A second account on one machine can fetch: `gpuwm-global` locks its
  downloads in the account's own `~/.gpuwm/locks`.
- `fetch-analysis` names the shipped day that reads the fetched file and the
  commands that follow.
- `statics` over a cache that verifies exits 0.
- The two shipped `_6h` days read the 24 h day's analysis, and a missing
  analysis names the command that fetches it.
- `score` labels its model row `arwen-global-t<N>` and prints each step.
- Map source lines name the model grid; the console line says
  `total_water`.
- README links resolve on PyPI.
- The scorecard and `score` rebuild the surface with the run's orography
  filter.
- `go` hands the forecast's cached card memory back before it renders.
- `render --bbox` with already exported tapes is refused by name.
- A card that runs out of memory in a command the door does not price
  (`da fresh` among them) ends with a message naming the card, the free
  memory and the remedy, not a traceback.
- The statics refusal names the config that builds the missing statics.
- Documented figures (card, memory, disk, time, download) match what the
  program does.
