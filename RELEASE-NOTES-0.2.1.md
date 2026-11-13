gpuwm-global 0.2.1

Arwen Global 0.2.1 is 0.2.0 with one new experimental option, a faster
start for every process, and a fix for long winter runs. The option takes
the jet ripples out of T383 runs and costs some northern-hemisphere skill
there, so it is off by default.
The faster start comes from keeping the Legendre analysis table on disk,
so a restarted segment or an export does not rebuild it. A forecast that
names no new option is 0.2.0's forecast, checkpoint for checkpoint, and
every shipped config keeps its hash. It needs `gpuwm>=2.9.13,<2.10`, as
0.2.0 did, and the Rust doors are 0.2.0's.

```bash
pip install --upgrade "gpuwm-global[gpu-cu13]"
gpuwm fetch-bridges
gpuwm-global fetch-doors
gpuwm-global go arwen_global_gdas_t255_native_sl_si_24h --outdir out/day
```

## New

- **`[semilag] extrapolation = "settls_residual_departure"`**
  (experimental, off by default). The shipped semi-Lagrangian step
  (`"settls"`) extrapolates the nonlinear residual in time at the arrival
  point. At T383 and 300 s that puts divergent waves of 4 to 6 grid
  lengths across the fastest jets. This option reads the extrapolated
  residual at the departure point instead and keeps the shipped
  trajectory.
  - It removes the ripples. From the GDAS 2026-11-12 12Z analysis at
    T383, the ripple measure (the RMS of the divergent wind above degree
    280 near 240 hPa, inside winds over 50 m/s) was 0.26 m/s at 96 h with
    the option and 1.21 m/s without it. The analysis itself has 0.30 m/s.
  - It costs skill at T383. Over four 48 h forecasts (GDAS 2026-11-09 and
    2026-11-10, 12Z and 18Z), the northern-hemisphere 500 hPa height RMSE
    was 21.20 m with the option and 20.38 m without it, worse in all four.
  - At T255 it is neutral: on two 48 h forecasts every score moved by
    less than 0.08 m or 0.04 m/s.
  - Use it where clean jets matter more than that skill, for example for
    pictures. Setting it changes the config hash. It runs on one card at
    any latitude-band count.

## Changed

- **A faster start for runs, restarted segments and exports.** The
  Legendre analysis table is built with one small linear solve per order,
  and on a busy host a multi-threaded BLAS can stall on those solves for
  10 s or more where they normally take a fraction of a second. The table
  is now written once per machine to `~/.gpuwm/cache/arwen-global-legendre`
  (54 MiB at T255, 176 MiB at T383) and read by every later process. A
  table read from the file holds exactly the bits a built one does, and
  any mismatch rebuilds it. `ARWEN_GLOBAL_LEGENDRE_CACHE` names another
  folder, or `off` turns it off. The folder can be deleted at any time.
- `extrapolation = "settls_departure"` (experimental, from 0.1.8) now runs
  on latitude bands on one card too, so the T383 configs sized to bands on
  a 12 or 16 GB card can use it.

## Fixed

- **A run no longer stops when its air gets deeper than the radiation's
  tables.** The RRTMGP gas tables end at 1,096.6 hPa. A T127 year run
  stopped at day 193 on a 1,097 hPa layer on the Caspian shore, where the
  truncated terrain dips below sea level under a winter high. Such a layer
  is now looked up at the tables' deepest pressure, as the UFS/CCPP RRTMGP
  interface does; layers up to 1,150 hPa are accepted. Runs that stay
  inside the tables are bit-for-bit unchanged.

## Measured

- **Defaults unchanged.** On an RTX 5070 Ti, the shipped T255
  semi-Lagrangian config run 3 h from a cold start, and restarted to 6 h,
  wrote checkpoints byte-identical to 0.2.0's, every array and the
  archive hash: with the Legendre table built, read from the cache, or
  the cache off, and with 0.2.0's own checkpoint as the restart. Every
  shipped config keeps its 0.2.0 hash.
- **Start-up.** On a quiet host the cache saves about a tenth of a
  second per process, inside the run-to-run spread (0.29 to 0.36 s to
  build the T255 table, 0.19 to 0.23 s to read it). On a busy host it
  saves much more: with other jobs loading the same host, building the
  table took up to 20.3 s at T255 and 158.7 s at T383, and reading it at
  most 0.48 s and 1.50 s. Most of
  a restart's remaining start-up (about 6 of 9 s at T255) is the cold
  state rebuilt from the analysis, which this release does not change.
- The CPU test suite passed: 3420 passed, 86 skipped, 0 failed.

`CHANGELOG.md` has the details.

## Upgrading

From 0.2.0: `pip install --upgrade "gpuwm-global[gpu-cu13]"`. The doors
are unchanged (the v0.1.3 bundle), so no re-fetch is needed. The config
hashes are 0.2.0's, so a 0.2.0 checkpoint restarts under 0.2.1 and
continues as it would have under 0.2.0. The first run after the upgrade
writes the Legendre table file (54 MiB at T255, 176 MiB at T383); set
`ARWEN_GLOBAL_LEGENDRE_CACHE=off` to keep it off the disk.
