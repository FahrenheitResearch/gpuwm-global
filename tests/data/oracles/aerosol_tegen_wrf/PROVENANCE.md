# WRF's own aer_opt = 1 numbers on four columns

`wrfcol.out` is what WRF's routines compute for four test columns, the oracle
`tests/test_arwen_global_aerosol_climatology.py` and its card half hold
`arwen_global.physics.aerosol_climatology` to.

- `wrfaer.F90` is cut verbatim from WRF v4.6.1 (github.com/wrf-model/WRF, tag
  v4.6.1): `interp_vec` from `phys/module_physics_init.F`, `aer_time_int` and
  `aer_p_int` from `phys/module_radiation_driver.F` (lines 5107-5377), and the
  body of `swaerpr` from `phys/module_ra_rrtmg_sw.F` (lines 5045-5130, its
  `_rb` kind spelled `_8`). Only `wrf_error_fatal` is a stub.
- `wrfcol.F90` is the driver: `aerosol_in`'s read loops and its horizontal
  step for one point, then `aer_time_int`, `aer_p_int`, and the `iaer = 6`
  combination loop of `rrtmg_sw` (copied), on four columns: (22N, 10E),
  (2N, 150W), (26N, 105W) and (30S, 179.9E), 30 layers from interfaces
  101325 + (100 - 101325)(k - 1)/30 Pa times 0.97, 1.0, 0.85 and 1.0 with
  the top interface at 100 Pa, at 1976-03-20 12Z (julday 80, julian 79.5).
- Built with GNU Fortran 15.2.0 (`gfortran -O0 -ffree-line-length-none
  -fno-range-check wrfaer.F90 wrfcol.F90`) on Linux x86-64 and run on WRF 4.7.1's
  `run/aerosol*.formatted`, which are byte-identical to v4.6.1's (hashes in
  `src/arwen_global/data/aerosol-tegen-wrf-receipt.json`), on 2026-11-16.
  `sha256(wrfcol.out) = fe34d2207d35b577b8221f77fe366997517ef4cc8dd5f3cf40aaa6c8c7ccc480`.

Format: per column a `column` line (latitude, WRF longitude, column total
AOD), then per layer (bottom-up) a `layer` line (layer pressure Pa, layer
thickness Pa, the six types' 550 nm optical depths) followed by 14 `band`
lines in RRTMG's band order (tau, ssa, g).

WRF is in the public domain (licenses/LICENSE-WRF-public-domain.txt);
`swaerpr` is AER's (licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt).
