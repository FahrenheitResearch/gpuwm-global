# Arwen Global forcing: pull a run toward a record, and read the pull (experimental)

The forcing operator pulls a run toward a stored record of large scales
before every step and keeps the pull as a measurement. A run that does not
ask for it takes none of its code. This page covers the operator, its
options, what it writes, and the one twin pair measured with it. It is
spectral nudging. The method is old and this page claims nothing new
about it.

## What it does

Before each step, the chosen spectral fields move part of the way toward
the record, on the chosen degrees only:

    x <- x + a(k) L (x_rec(t) - x),     a = 1 - exp(-dt / tau)

- `L` weights each total degree. The weight is one at and below
  `full_degree`, zero at and above `zero_degree`, and a raised cosine in
  between. The defaults are 42 and 43, so degrees 1 to 42 are pulled in
  full: wavelengths longer than about 950 km.
- `tau` is the relaxation time (default six hours). `tau_h=inf` observes
  only. The operator then reads the departure from the record every step
  and changes nothing.
- `k` is an optional weight for each level, top first.
- `x_rec(t)` is the record, interpolated linearly in model time, plus an
  optional offset.
- Pulled by default: vorticity, divergence and potential temperature.
  Vapour and ln ps can be named. Degree 0 of ln ps is never pulled,
  because the mass fixer owns the global mean of surface pressure.

The pull depends only on the state, the model time and the record, so a
restart in the middle of a pulled run continues it exactly. The config
identity does not change, so a free run and a pulled run from one start
share one checkpoint lineage and differ only by the pull. A pulled state's
checkpoints say so under `physics_metadata["forcing_history"]`: the
record's digest, the options, and the step the pull began. The receipt
carries a `forcing` block, and a `forcing_record` block when the run wrote
a record. The operator refuses a run on more than one card and a run with
a host tier that parks the spectral state.

## The route

```bash
# a run that writes a record of its own low degrees, hourly
gpuwm-global run teacher.toml --outdir out/teacher \
  --write-forcing-record "out/teacher-record.npz:degree=42;every_h=1;dtype=complex64"
# or cut a record afterwards from a finished run's checkpoints
python tools/arwen_global_forcing_record.py out/teacher/arwen_global_step*.npz \
  --out out/teacher-cut.npz --degree 42
# the same start, free (observing only) and pulled
gpuwm-global run student.toml --outdir out/free --forcing "out/teacher-record.npz:tau_h=inf"
gpuwm-global run student.toml --outdir out/forced --forcing "out/teacher-record.npz:tau_h=6"
python tools/arwen_global_forcing_table.py --reference out/teacher \
  --free out/free --forced out/forced --ledger out/forced
```

`--forcing` options (`key=value;...`): `tau_h` or `tau_s`, `full_degree`,
`zero_degree`, `fields`, `start_h`/`end_h`, `offset_h`, `level_weights`,
`window_h`. `--write-forcing-record` options: `degree`, `every_h`,
`fields`, `dtype`.

A record from a finer run can pull a coarser one, as long as both have the
same level count and the record holds the pulled degrees. Coefficients are
in the state's own normalisation at every truncation.

## What it writes

`forcing_ledger.json` sits beside the receipt. It holds, per pulled field
and level, and for the horizontal wind derived from vorticity and
divergence:

- `departure_rms`: the time RMS of `L (record - model)` before the pull,
  as a grid mean on the sphere.
- `pull_rms_per_day`: the time RMS of the change the operator made, as a
  tendency per day.
- `time_mean_pull_rms_per_day`: the RMS of the pull's time mean. This is
  the systematic part. Once the run is held near the record, it is what
  the model needed added to stay there, which makes it the estimate of
  model error on the pulled scales.
- `global_mean_pull_per_day`, for potential temperature, vapour and ln ps:
  the degree-0 part.
- `by_degree`: the departure and pull power for each degree, summed over
  levels.
- `windows`: the same readings for each window of model time (default one
  hour).

`forcing_pull.npz` holds the time-mean pull as spectral coefficients, so
it can be mapped afterwards.

Python callers use `arwen_global.forcing.ForcingOperator(record, options,
transform=..., dt_s=..., nlev=...).apply(state, model)`, the same call the
runner makes. Any loop that holds a state, a model and a step can make it.

## What is tested

`tests/test_arwen_global_forcing.py`, on the T3 CPU smoke configuration:

- One pull on a planted departure moves exactly the planted coefficients
  by `1 - exp(-dt/tau)`. Degree 0 of ln ps does not move. The ledger reads
  the planted size back for each field and level.
- A run pulled toward its own record, taken every step, is the free run
  bit for bit. So is a run that only observes.
- A run pulled toward a record carrying a planted anomaly closes on it
  window by window. The observing twin reads the anomaly back as its
  departure.
- A restart in the middle of the pull continues it exactly, metadata
  included.
- `tools/arwen_global_forcing_record.py` and the in-run writer give one
  digest for the same snapshots.
- The table tool reads a known answer.

## The one twin pair measured

One T63 case on one RTX 4090, all six runs back to back on the card, from the ERA5
2019-01-15 00Z start, 40 levels, 300 s step, 36 hours (432 steps).

- The record: a teacher run that calls radiation every step
  (`radiation_interval_s = 300`), writing degrees up to 42 hourly as
  complex64.
- The students: the same config at the shipped 3000 s radiation interval,
  one observing (`tau_h=inf`, the free twin) and one pulled
  (`tau_h=6`, the forced twin).

So the model error being read is the error of calling radiation every
ten steps instead of every step, seen on degrees 1 to 42.

**Default results unchanged.** Ten steps of the same config from this
tree with neither flag, and from 0.1.7, give checkpoints with one
`self_sha256` (`tools/arwen_global_checkpoint_identity.py`: 133 arrays
compared, none differing). The observing run over the same ten steps gives
that same checkpoint too. All six runs were taken again on this tree on
2026-11-12 on the same card model, and the tables below came back the same
to the digits printed.

**The pull closes the gap.** `tools/arwen_global_forcing_table.py` gives
`r = 1 - E_forced / E_free`, the share of the free twin's departure
energy from the teacher that the pull removed, on degrees 1 to 42 and 43
to 63:

| hour | vorticity | divergence | theta | vapour | ln ps | wind |
|---:|---|---|---|---|---|---|
| 6 | 0.32 / 0.11 | 0.24 / 0.12 | -0.13 / 0.09 | 0.17 / 0.07 | -2.02 / 0.16 | 0.26 / 0.11 |
| 12 | 0.48 / 0.10 | 0.51 / 0.16 | 0.36 / 0.16 | 0.19 / 0.13 | 0.01 / 0.15 | 0.49 / 0.14 |
| 24 | 0.76 / 0.34 | 0.79 / 0.43 | 0.73 / 0.35 | 0.32 / 0.23 | 0.64 / 0.46 | 0.79 / 0.41 |
| 36 | 0.86 / 0.53 | 0.86 / 0.49 | 0.82 / 0.43 | 0.37 / 0.28 | 0.78 / 0.49 | 0.87 / 0.51 |

Vapour and ln ps were not pulled; they follow the pulled fields. In the
first six hours the pulled run's ln ps and theta on the large scales sat
further from the teacher than the free run's did (negative r), and from
twelve hours on every field is closer. Degrees 43 to 63, which are not
pulled, close by about half by 36 hours.

**The pull, read as model error.** The forced twin's ledger, at the model
levels nearest 850, 500, 250 and 100 hPa:

| level (hPa) | theta departure (K) | theta mean pull (K/day) | wind departure (m/s) | wind mean pull (m/s per day) |
|---:|---:|---:|---:|---:|
| 861.5 | 0.075 | 0.119 | 0.111 | 0.174 |
| 474.0 | 0.030 | 0.051 | 0.052 | 0.080 |
| 238.7 | 0.031 | 0.055 | 0.080 | 0.141 |
| 98.8 | 0.028 | 0.029 | 0.048 | 0.048 |

The systematic pull is largest at the lowest level shown, about 0.12 K per
day of potential temperature on degrees 1 to 42. The observing twin over
the same 36 hours read departures of 0.114, 0.061, 0.063 and 0.085 K at
those levels with no pull. These are 36 hours of one case; they say what
the ledger reads, not what the radiation interval costs in general.
