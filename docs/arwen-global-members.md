# Arwen Global forecast members: the member engine and its replay recipes (experimental)

Module `src/arwen_global/members.py`; driver `tools/arwen_global_members.py`;
tests `tests/test_arwen_global_members.py`.

This page describes forecast members started from one analysis outside the
assimilation path, the recipe of a few kilobytes that names each one, the
replay that reproduces a member from its recipe, and exact twins branched
from a member checkpoint. Numbers appear only where they were measured, and
section 6 says where.

## 1. What a member is

A member is the base state (the cold start of one analysis at the member
truncation) plus a weighted sum of perturbation draws, stepped through one
shared model on one card. Draw `k` is `da.perturbations.draw_perturbation`
from the stream `member_rng(seed, k, "initial")`, after the one integer the
filter's own construction takes from that stream first. So member `k` of a
set started here is member `k` of `GlobalEnsemble.from_state` with the same
options, array for array, and any member can be drawn again alone.

- No draws: the control.
- One draw of weight 1: an ordinary member.
- Several draws with other weights: a blend, the form an aimed state takes.
  A draw of weight 0 is recorded and never summed, so it changes no byte.

The members share the model, its transform tables, its statics and the
physics workspace, and each steps under its own conservation targets and its
own semi-Lagrangian second time level, exactly as the filter's ensemble
steps (`GlobalEnsemble.step_all`).

## 2. The interface

```python
from arwen_global.da import EnsembleOptions, ensemble_config
from arwen_global.members import MemberSet, replay, read_recipe
from arwen_global.runner import build_model_and_cold_state, build_transform

options = EnsembleOptions(members=3, truncation=127, seed=20260906)
cfg = ensemble_config(run_cfg, options)             # the run config cut to T127
transform = build_transform(cfg)
model, base = build_model_and_cold_state(cfg, transform)
members = MemberSet.start(cfg, model, transform, base, options,
                          [[], 0, 1, 2])            # control and draws 0, 1, 2
members.advance(288, mark_every=36)                 # 24 h at 300 s, digests every 3 h
members.finish_recipes()
members.write("out/members")                        # checkpoints and recipes
```

- `MemberSet.start(cfg, model, transform, base, options, specs)`: one member
  per entry of `specs`, each a whole number `k` (draw `k` at weight 1) or a
  list of draws `[{"member": k, "weight": w}, ...]` (empty for the control).
- `MemberSet.advance(steps, mark_every=n, observer=f)`: steps every living
  member; every `n` steps and after the last it records each member's digests
  in its recipe; `observer(member_set)` runs after every step.
- `MemberSet.step_all(dt)`: one step. A member whose step raises a
  non-finite or gate error is marked dead with its error and the others go
  on; a dead member keeps the state that entered the failed step. Any other
  exception propagates, and so does the error when every member is dead.
- `MemberSet.write(outdir)`: each member's checkpoint in the run format
  (zero run trackers) and its recipe beside it as `*.recipe.json`.
- `MemberSet.branch(...)`: an exact twin, section 4.
- `replay(recipe, build)`: section 5.

Not supported: a config with a dated surface series (the run door applies it
between steps and a member set does not, so the set refuses it), and members
whose sea-ice or land planes differ (the shared model refuses them, as the
filter does).

## 3. The recipe

A recipe is JSON, schema `gpuwm.arwen-global-member-recipe/v1`. It holds no
array. A member's recipe after a 24 h run with digests every 3 h measured
3.6 to 3.7 KB; a twin's carries its parent's inline and measured 5.4 KB.

| Key | What it holds |
|---|---|
| `config` | the config's name, `config_hash`, truncation, step, integrator, semi-implicit scheme, physics mode, and the SHA-256 of the TOML it was cut from |
| `base` | step, time and `arrays_sha256` of the base state before any draw (for a twin, of the parent's state at the branch step) |
| `perturbation` | the perturbation family and stream, every `EnsembleOptions` value, the draws and weights in summation order, the seed integer each draw's stream gave |
| `targets` | the global-mean surface pressure and total water the member's fixers hold |
| `marks` | per recorded step: `arrays_sha256` (the table of per-array hashes of every checkpointed array, second time level included) and `checkpoint_sha256` (the `self_sha256` of the member's checkpoint at that step) |
| `branch` | null, or the parent recipe, the parent checkpoint's hash, both config hashes, whether they differ, whether the opt-in was given, and the change in words |
| `stack` | package, engine, numpy, CuPy and CUDA versions, the backend and the card's name |
| `died` | null, or the error that ended the member |

`arrays_sha256` does not depend on the config hash, the run trackers or the
code path, so a member and a run-door twin can be compared on it.
`checkpoint_sha256` is the hash of the file `write` puts on disk.

## 4. Exact twins and the branch opt-in

`MemberSet.branch(child_cfg, model, transform, parent_checkpoint,
parent_recipe, draws=..., change=..., allow_config_branch=False)` starts a
one-member set from a member checkpoint: the same state, its own second time
level and the conservation targets its recipe records, plus the deliberate
change: draws added at the branch step, a different config, or both. The
parent's recipe must mark the checkpoint's step and hash.

A child whose config hash differs from the parent checkpoint's is refused
unless `allow_config_branch=True`, and the recipe records that the opt-in was
given. The run door has the same rule: `gpuwm-global run CONFIG --restart
PARENT.npz --branch-from-other-config` admits a parent written under another
config hash and the receipt records both hashes under `branch`. Without the
flag such a restart is refused as before, and a receipt without the flag
carries no `branch` row.

## 5. Replay

`replay(recipe, build)` rebuilds the member a recipe names and refuses
(`ReplayMismatch`) unless:

- the config `build` supplies hashes to the recipe's `config_hash`;
- the base state's `arrays_sha256` is the recipe's, before the first step;
- the member's conservation targets are the recipe's;
- every marked step's two digests are the recipe's.

`build(config_record)` returns the config, a fresh model, its transform and
the base state. For a twin, `replay` replays the parent to the branch step,
writes it, checks the branch state's digest and replays the twin from there.
The driver's `replay` command does this in a fresh process:

```
python tools/arwen_global_members.py replay RUN_CONFIG.toml MEMBER.recipe.json --outdir out/replay
```

The config hash includes the analysis file's path, so a recipe replays where
the config it names hashes the same. A recipe names its card and stack
because identity between two cards is not shown.

## 6. What is measured

On the T3 smoke configuration (`tests/test_arwen_global_members.py`, CPU):
started members equal the filter's members array for array; a member set's
step equals the filter's step under the Eulerian and the semi-Lagrangian
cores; a recipe replays its member alone in a fresh model to the recorded
digests and a changed weight or base digest is refused; a zero-weight draw is
the control byte for byte; a blend is the weighted sum of its draws; one
member's death leaves the others byte-identical to a set where nobody died;
an unchanged twin continues its parent bit for bit; a twin under another
config is refused without the opt-in, recorded with it, and replays; the run
door's restart under another config is refused unless the opt-in is given.

On a card, measured 2026-11-12 on one RTX 5070 Ti 16 GB (CuPy 14.2.0,
CUDA 13.2, gpuwm 2.9.13). The run config was
the shipped `arwen_global_gdas_t255_native_sl_si_24h` (native suite, the
semi-Lagrangian core at 300 s) started from the GDAS analysis of
2026-10-15 00Z, cut to T127 by `ensemble_config`, with the default
perturbation options:

```
python tools/arwen_global_members.py start RUN_CONFIG.toml --truncation 127 --members 3 --control --hours 24 --mark-every 36 --write-at 144 --outdir out/t127
python tools/arwen_global_members.py replay RUN_CONFIG.toml out/t127/m001_*.recipe.json --outdir out/replay
python tools/arwen_global_members.py branch RUN_CONFIG.toml out/t127/step000144/m002_*.npz out/t127/step000144/m002_*.recipe.json --hours 12 --outdir out/twin
python tools/arwen_global_members.py branch RUN_CONFIG.toml out/t127/step000144/m002_*.npz out/t127/step000144/m002_*.recipe.json --hours 12 --draw 7:0.1 --outdir out/kick
python tools/arwen_global_members.py replay RUN_CONFIG.toml out/kick/*.recipe.json --outdir out/kick-replay
```

| Measurement | Result |
|---|---|
| control and three members, T127, 24 h (288 steps each) | all four finished; none died |
| member `m001` replayed alone in a fresh process from its 3.7 KB recipe | every marked digest met; step 288 `arrays_sha256` `a56fe70d806a5481...` and checkpoint `6d1567bcff067cb3...` in both |
| unchanged twin of `m002` branched at step 144, run to step 288 | `arrays_sha256` `376c5c37a44f993f...`, the parent's own at step 288 |
| twin of `m002` with draw 7 at weight 0.1 added at step 144, its 5.4 KB recipe replayed in a fresh process (parent to step 144, branch, step 288) | every digest met |
| wall per member-step in the four-member set, mean over 1,148 member-steps after each member's first | 0.286 s (median 0.184 s; the mean carries the radiation steps) |
| the same for one member alone (the replay), mean over 287 steps | 0.239 s (median 0.184 s) |
| a member-day at T127 in the set, from the mean step | 82 s; from the steps' wall including the digest marks, 86 s |
| a member-day at T127 alone, the replay's whole process from start to exit | 80 s |
| card held, four-member set | CuPy pool 7.95 GB, device 8.47 GB used at the end |
| card held, one member alone | CuPy pool 6.91 GB, device 7.40 GB used at the end |
| one member's arrays at T127, measured | 292 MB (physics namespace 144 MB, grid tracers 118 MB) |

So on this card the shared model saves memory and not time: four members
used 1.07 GB more of the card than one, and a member step in the set took
longer on average than a member stepped alone. Running members as separate processes on one card was
not measured. The departure of the members from the
control at 24 h, as the norm of the difference of the spectral coefficients
over the norm of the control's, was
0.84e-4 to 1.30e-4 in log surface pressure, 1.3e-3 to 1.5e-3 in potential
temperature and 0.65 to 0.69 in vorticity.

The same five commands were taken again on the 0.1.7 line on 2026-11-12, on
one RTX 5090 at T255 (`--truncation 255 --hours 6 --mark-every 12
--write-at 36`, branches of 3 h from step 36) from the shipped config's
own GDAS 2026-09-01 00Z analysis: the control and three members all
finished; `m001` replayed alone met every digest; the unchanged twin of
`m002` ended on the parent's own step-72 `arrays_sha256`; the twin with
draw 7 at weight 0.1 replayed from its recipe to the same digest. The card
read 15,571 MiB used at its peak over the five commands; a member-day at
T255 in the set of four, from the mean member step, was 167 s, and a step
of one member alone (the twin) took 0.61 s on average (median 0.52 s).

Default runs are unchanged: the run door's twelve-step forecast of the
shipped `arwen_global_gdas_t255_native_sl_si_24h` from its GDAS 2026-09-01
00Z analysis, run from 0.1.7 and from this tree on one RTX 5090 on
2026-11-12, wrote step-12 checkpoints whose 133 arrays are bit-identical
(`tools/arwen_global_checkpoint_identity.py`) and whose `self_sha256` is
the same (`4236a15dcf6f8a34...`).

## 7. Limits

- Members step one after another through the shared model; they are not
  batched into one step.
- The control here is the base state through the member path (positivity
  repair and enforcement before its first step). Whether it equals the run
  door's cold start of the same config byte for byte is not measured.
- A dead member's recipe records the error; the set does not retry it.
- No door: the member engine is a library and a driver under `tools/`.
