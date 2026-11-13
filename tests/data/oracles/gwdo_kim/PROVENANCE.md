# gwd_opt = 1 (KIM orographic drag) WRF v4.6.1 column oracle

Copied byte for byte from the regional engine's own oracle,
`tests/data/oracles/gwdo_kim/` of gpuwm 2.9.13's source tree, where
`tools/gwdo_kim_wrf461_oracle/build.sh` and `compare_gpu.py --write-fixture`
wrote it on 2026-10-07.  The engine's sector-sample file is not copied: it
grades the device atan2f on its own, and the carried kernel's sector words
are graded here through the columns that sit on a sector boundary.

- WRF source: github.com/wrf-model/WRF tag v4.6.1, commit
  `d66e442fccc04111067e29274c9f9eaccc3cef28`, byte-unmodified:
  `phys/physics_mmm/bl_gwdo.F90` (sha256
  `c294ef2eb9da1ee9ce6594b97adea7280eb18760e0c5576ab7149c35bb7c3c3b`) and
  `phys/module_bl_gwdo.F` (sha256
  `d782ddcda578a1fde0cd74e6526bf14f27cb0fb48113c227ad971717e6ca90cc`), both
  byte-identical to WRF v4.7.1's, called as `module_pbl_driver.F:2136-2151`
  calls them, with WRF's own `module_model_constants`.
- Compiler: GNU Fortran 13.3.0 on Ubuntu 24.04 (glibc 2.39), `-O0
  -ffp-contract=off`; WRF's stock `-O2 -ftree-vectorize -funroll-loops`
  build writes the same bytes.

| file | sha256 | content |
|---|---|---|
| `gwdo-kim-inputs-L50.npz` | `a30bba9ae5737bd0e4f76e405bac314dc3fa5726dc8a306473574544eba517f4` | float32 inputs, 50 levels |
| `gwdo-kim-inputs-L90.npz` | `7599931644f46b9e8f5b52eed99f3486773692d7335a1fcb71a39b05a03fee49` | float32 inputs, 90 levels |
| `gwdo-kim-wrf461-L50.npz` | `64aafab05aa280cd2c870fae9acc977061a98c42456179fe85c1405986f32598` | WRF's outputs, 50 levels |
| `gwdo-kim-wrf461-L90.npz` | `8a5033c4259b8fc929b6ee342de12779c401a2dac773d9e44d98efc105f3a66d` | WRF's outputs, 90 levels |

149 columns on two WRF-like eta grids (50 and 90 levels, p_top 50 hPa) in 12
regimes (convective, stable_night, neutral_windy, blocking, high_terrain,
ocean, cold_snow, tropical_moist, jet_shear, critical_level, sector_boundary
and 17 edge cases), each run through one `gwdo` call at six (dx, dt) cases:
1 km/6 s, 3 km/18 s, 9 km/54 s, 12 km/72 s, 30 km/180 s and 50 km/300 s.
Outputs: RUBLTEN, RVBLTEN, DTAUX3D, DTAUY3D (case, level, column) and
DUSFCG, DVSFCG (case, column).

`tests/test_arwen_global_orographic_drag.py` reads these and checks the
hashes above.
