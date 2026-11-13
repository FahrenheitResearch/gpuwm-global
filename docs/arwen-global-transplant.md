# Arwen Global scheme transplant: how much of a difference one scheme carried (experimental)

Module `src/arwen_global/transplant.py`; seams in
`src/arwen_global/physics/native_runtime.py`; tests
`tests/test_arwen_global_transplant.py`.

This is an experiment tool. It answers one question about two runs of the
same configuration: how much of the difference between them did one physics
scheme carry? It does that by running a twin in which that scheme
contributes nothing of its own. A run made with `gpuwm-global run` never
meets it: the transplant is attached by hand to models built in one process,
and a model it is not attached to runs the native suite as before.

## What it does

Two runs of the same configuration step side by side in one process. At
every physics call the DONOR records what one scheme of the native suite put
out, and the RECEIVER applies the donor's output in place of computing its
own. The receiver's scheme never runs, so whatever difference is left
between donor and receiver travelled through the dynamics and the other
schemes.

Two schemes can be transplanted.

- `cumulus`: the native runtime splits its cumulus call at the scheme's
  output. `_cumulus_result` returns what the scheme decided (the theta,
  vapour, cloud-water and cloud-ice rates, the momentum pair when the scheme
  carries one, RAINCV and its diagnostics); `_apply_cumulus_result`
  integrates it. The receiver applies the donor's RATES to its own column,
  `x += dt * rate`. Where the two columns are bit-equal this is the donor's
  own operation on the donor's own operands, so an identical twin stays
  identical with no copy rule.
- `microphysics`: the Morrison call is split into `_microphysics_kernel`
  (the launch, which updates the batch in place) and `_book_microphysics`.
  The donor records every array the launch touches before and after. The
  receiver copies the donor's exit value where its entry equals the
  donor's bit for bit and adds the donor's change `exit - entry` elsewhere,
  because float32 does not give `e + (x - e) == x`. A water species or
  number concentration the added change would leave negative is set to zero
  and counted.

The rule for everything persistent: the scheme's output is the donor's, the
books are the receiver's. The receiver credits its own surface reservoir
with the column water the call removed as measured in its own column, caps
the cumulus rain at the water the rates removed from its own column, and
keeps its own accumulators and call counters. The effective radii and `sr`
are the donor's, because the radiation reads them as the microphysics'
output.

The donor steps first. Every record carries its band, model time and step
length, and a receiver whose call does not match is refused by name. A
model is a donor or a receiver, not both, and cannot receive from itself.

On the default path the runtime calls the two halves in sequence, which is
the arithmetic it ran before the split. A run with no transplant attached is
the 0.1.7 run (measured below).

## The four-run experiment

```bash
python -m arwen_global.transplant RUN_CONFIG.toml --hours 48 \
  --report-hours 6 12 24 36 48 --stages cumulus --out cumulus.json
```

From one analysis decode it builds five models of `RUN_CONFIG.toml` (the
native suite, `physics.mode = "arwen-native"`) and steps them in lockstep:

| Run | What it is |
|---|---|
| `C` | the control |
| `P` | the control plus one seeded perturbation (`--seed`, `--amplitude-scale`) |
| `Ps` | `P` receiving `C`'s scheme output: the scheme carries none of `P`'s own difference |
| `Cs` | `C` receiving `P`'s scheme output: the difference the scheme's output alone carries into an unperturbed start |
| `Cc` | a second `C` receiving `C`'s output: the self-substitution |

`Cc` must equal `C` at every report hour, every checkpointed array compared
by its hash; the command exits non-zero if it does not. At each report hour
the JSON gives the dry total energy of each difference (global area mean,
J/m2) and two shares of `E(P - C)`:

- `share_removed = 1 - E(Ps - C) / E(P - C)`: the fraction of the
  difference that disappears when `P`'s scheme output is `C`'s;
- `share_alone = E(Cs - C) / E(P - C)`: how far `C` moves with `P`'s
  scheme output and nothing else of `P`'s.

It also gives both readings in a box of `--box-radius-deg` around the
largest surface-pressure difference poleward of 20 degrees, and the
receivers' ledgers (calls, copied and incremented cells, cells left negative
and their sum). `--self-only` runs `C` and `Cc` alone. The experiment writes
its JSON and no checkpoint or run receipt.

## What is tested

`tests/test_arwen_global_transplant.py` runs the real shipped step with the
native suite behind the level-5 fakes on the numpy backend (the T21
baroclinic configuration), and every claim is a byte-for-byte equality of
every checkpointed array:

- a twin receiving its identical twin's cumulus and microphysics output
  equals the twin, and recording moves nothing in the donor;
- a run whose own cumulus scheme heats nothing, receiving the output of a
  twin whose scheme heats, equals the heating run, and its own scheme never
  runs; the same for the Morrison slot;
- the copy-where-equal rule where float32 subtraction rounds;
- the refusals (a receiver ahead of its donor, a model receiving from
  itself, an unknown stage, a model hooked twice);
- a detached model runs the suite as before;
- the four-run driver end to end.

## Measured on a card

Measured 2026-11-12 on one RTX 4090 at T63 L40: the native suite on the
semi-Lagrangian core at 300 s from the GDAS 2026-11-10 18Z analysis, one
perturbation drawn by the filter's own route (`--seed 20261112`, amplitude
scale 1, degrees up to 31).

| hour | E(P - C) (J/m2) | cumulus `share_removed` | cumulus `share_alone` | microphysics `share_removed` | microphysics `share_alone` |
|---:|---:|---:|---:|---:|---:|
| 6 | 52,998 | +0.005 | 0.022 | -0.052 | 0.077 |
| 12 | 60,325 | +0.006 | 0.051 | -0.091 | 0.183 |
| 24 | 82,819 | -0.005 | 0.097 | -0.226 | 0.396 |
| 36 | 117,235 | -0.018 | 0.144 | | |
| 48 | 158,934 | -0.012 | 0.210 | | |

The cumulus run went 48 hours (576 steps, 848 s for the five models) and
the microphysics run 24 hours (288 steps, 273 s). In both, `Cc` equalled
`C` at every report hour, all 133 checkpointed arrays by hash, and the
three receivers equalled their twins before the first step.

Read plainly: giving `P` the control's cumulus output removed none of the
pair's difference (the share stays within 0.02 of zero), while `C` with
`P`'s cumulus output alone reached a fifth of the pair's difference energy
by 48 hours. The difference this perturbation made did not travel through
the cumulus scheme's own decisions; that scheme's output is nonetheless
enough to seed a difference of its own. The Morrison transplant made the
pair further apart, not closer (a negative `share_removed`), and its
receivers set 55 to 58 million cell values to zero over the day where the
donor's change would have left them negative: the donor's microphysics
change does not fit a column that differs from the donor's, so that slot's
shares say more about the transplant than about the scheme.

A run with no transplant attached is the 0.1.7 run: the native suite's
default path calls the split halves in sequence. Twelve steps of the
shipped `arwen_global_gdas_t255_native_sl_si_24h` (Grell-Freitas cumulus
and Morrison microphysics both on) from 0.1.7 and from this tree, on one
RTX 5090, wrote step-12 checkpoints whose 133 arrays are bit-identical
(`tools/arwen_global_checkpoint_identity.py`).

## Limits

- One process holds all five models, so the card has to hold five states of
  the configuration; the four-run experiment was measured at T63 only.
- The shares are of one seeded perturbation in one case. They say what the
  scheme carried in that pair, not in general.
- Only the cumulus and Morrison microphysics slots have seams. The PBL,
  surface layer, land and radiation do not.
- A transplanted cumulus output can leave a receiver's water negative where
  its column differs from the donor's; the ledger counts those cells and
  the suite's own positivity repair clips them, as it clips any scheme's.
