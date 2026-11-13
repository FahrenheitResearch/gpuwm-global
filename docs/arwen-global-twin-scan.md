# Arwen Global: exact-twin amplitude scans (experimental)

The tool on this page is a research instrument and is EXPERIMENTAL: its
subcommands, its norm and its output may change between releases.

Two runs of this model that start from byte-identical checkpoints stay
byte-identical, on one card, for as long as they run. A twin experiment
uses that: change one thing in the start, run both, and every difference
between the two is caused by the change. This page describes the
instrument that writes such starts and reads the differences, and the
measurements taken with it.

## The instrument

`tools/arwen_global_twin_scan.py` has three subcommands. None of them
changes a configuration or a line of the model.

- `perturb BASE OUT --config CFG --amplitude-k A` copies a step-0
  checkpoint and adds `A` times one fixed pattern to its potential
  temperature. The pattern is seeded spherical-harmonic coefficients on
  degrees 1 to 42 (no global mean), white in degree and order, with a
  Gaussian profile in ln p centred on 500 hPa (width 0.5 in ln p) at a
  reference surface pressure of 1000 hPa. `A = 1` is a grid RMS of 1 K on
  the level of largest weight. The coefficients are drawn only up to degree
  42, so a T63, a T127 and a T255 start receive the same physical
  perturbation, bit for bit. `A = 0` writes arrays byte-identical to the
  input. The checkpoint keeps its own metadata with the changed array's
  hash and the self hash re-signed, so the restart door admits it under
  the config hash of the run that wrote it. A checkpoint past step 0 is
  refused.
- `bitflip BASE OUT` writes the one-bit twin: the lowest mantissa bit of
  the real part of one potential-temperature coefficient flipped (level 20,
  degree 10, order 3 by default). Its distance from the control is the
  noise floor of every twin comparison.
- `measure --config CFG --control DIR --arm NAME=DIR ...` reads, at every
  checkpoint step a control and an arm share, the difference arm minus
  control from the spectral coefficients: the dry total-energy norm
  (kinetic, `cp/Tr` times the potential-temperature variance and `R Tr`
  times the ln ps variance over two, `Tr = 300 K`), mass-weighted over the
  levels at 1000 hPa, and its parts as RMS wind, potential temperature,
  surface pressure and vapour. `--pair` adds the relative nonlinearity
  `||d+ + d-|| / ||d+ - d-||` of a plus and minus pair, `--truncate N`
  compares runs of different truncations on the degrees they share, and
  `--compare` adds the energy-norm correlation and amplitude ratio of two
  arms' differences, overall and in degree bands 1-10, 11-21, 22-42 and
  43-63. A second `--seed` gives an independent pattern, whose response
  against the first's is the null a correlation is read against.

`tests/test_arwen_global_twin_scan_tool.py` checks it on the T3 smoke
configuration on the CPU: a zero-amplitude start is byte-identical to its
input and a restart from it lands on the cold run's last checkpoint byte
for byte; the pattern has unit grid RMS by the package's own synthesis; a
planted amplitude reads back exactly; a plus and minus pair at step 0 reads
a nonlinearity of zero; the one-bit twin differs in one element by one unit
in the last place.

## On a card: the T63 day (2026-11-12)

The proof that the instrument does what this page says, on the shipped
`arwen_global_gdas_t63_native_sl_si_24h` (experimental, see
`docs/arwen-global-coarse-grids.md`), six hours of each run, every run on
the same RTX 5090.  A control ran from the GDAS 2026-09-01 00Z analysis;
`perturb` and `bitflip` wrote four starts from its step-0 checkpoint, and
each restarted to the same six hours:

    gpuwm-global run arwen_global_gdas_t63_native_sl_si_24h --outdir ctl --until-s 21600
    python tools/arwen_global_twin_scan.py perturb ctl/arwen_global_step00000000.npz \
        p010.npz --config t63.toml --amplitude-k 0.1 --verify
    gpuwm-global run arwen_global_gdas_t63_native_sl_si_24h --outdir run-p010 \
        --restart p010.npz --until-s 21600

(`t63.toml` is the shipped config file, whose path `gpuwm-global configs
--paths` prints.)  What `measure` read, arm minus control:

| Arm | 3 h: sqrt of the energy, RMS theta | 6 h: the same |
|---|---|---|
| amplitude 0 | 0 exactly (identical) | 0 exactly (identical) |
| the one-bit twin | 0.0064 m/s, 0.0040 K | 0.0192 m/s, 0.0120 K |
| +0.1 K | 0.1150 m/s, 0.0511 K | 0.1187 m/s, 0.0571 K |
| -0.1 K | 0.1150 m/s, 0.0511 K | 0.1188 m/s, 0.0576 K |
| the +-0.1 K pair's relative nonlinearity | 0.097 | 0.234 |

The zero-amplitude restart's step-72 checkpoint is byte-identical to the
control's, archive hash included (133 of 133 arrays,
`tools/arwen_global_checkpoint_identity.py`), so a restart from a start
the tool wrote is the control's own run.  The one-bit twin grows from
nothing to about a sixth of the 0.1 K response in six hours; that is the
floor a smaller perturbation is read against at this grid.  This is a
proof of the instrument on one day, not a predictability study.
