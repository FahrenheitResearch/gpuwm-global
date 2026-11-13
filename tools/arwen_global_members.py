"""Start forecast members from one analysis, replay one from its recipe,
or branch an exact twin from a member checkpoint.

``python tools/arwen_global_members.py start CONFIG --truncation 127 --members 3 --hours 24 --outdir D``
    The config is cut to ``--truncation`` the way the ensemble cuts it
    (``da.ensemble.ensemble_config``); its cold start is the base state.
    One member per draw ``0 .. N-1`` (plus the control with
    ``--control``) steps through one shared model; each member's
    checkpoint at the end and its recipe (``*.recipe.json``) are written,
    with ``members-run.json``: the wall per member-step, the cost of a
    member-day it implies, the card's pool peak and the stack.

``python tools/arwen_global_members.py replay CONFIG RECIPE --outdir D``
    Re-runs the one member a recipe names in a fresh process and refuses
    unless every digest the recipe marks is met (``replay.json``).

``python tools/arwen_global_members.py branch CONFIG PARENT.npz PARENT.recipe.json --hours H --outdir D [--draw K:W ...] [--allow-config-branch]``
    An exact twin from a member checkpoint: with no ``--draw`` and the
    parent's config it must continue the parent bit for bit.

CONFIG is the run config the members were cut from; the recipe names the
cut config's hash and refuses any other.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from arwen_global.config import load_config
from arwen_global.da.ensemble import ensemble_config, recut_config
from arwen_global.da.options import EnsembleOptions
from arwen_global.members import (
    MemberSet,
    read_recipe,
    recipe_sha256,
    replay,
    software_stack,
    write_recipe,
)
from arwen_global.runner import build_model_and_cold_state, build_transform


def _sha_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pool_peak(backend) -> dict:
    if getattr(backend, "name", "numpy") != "cupy":
        return {}
    pool = backend.xp.get_default_memory_pool()
    free, total = backend.xp.cuda.runtime.memGetInfo()
    return {"pool_total_bytes": int(pool.total_bytes()), "pool_used_bytes": int(pool.used_bytes()),
            "device_used_bytes": int(total - free)}


def _builder(config_path):
    base_cfg = load_config(config_path)

    def build(record):
        cfg = recut_config(base_cfg, int(record["truncation"]), dt_s=float(record["dt_s"]), name=record["name"])
        transform = build_transform(cfg)
        model, cold = build_model_and_cold_state(cfg, transform)
        return cfg, model, transform, cold

    return build


def cmd_start(args) -> None:
    base_cfg = load_config(args.config)
    options = EnsembleOptions(members=max(3, args.members), truncation=args.truncation, seed=args.seed)
    cfg = ensemble_config(base_cfg, options)
    steps = int(round(args.hours * 3600.0 / cfg.dt_s))
    t0 = time.perf_counter()
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    t_build = time.perf_counter() - t0
    specs = ([[]] if args.control else []) + list(range(args.members))
    t1 = time.perf_counter()
    members = MemberSet.start(cfg, model, transform, cold, options, specs,
                              config_source_sha256=_sha_file(args.config))
    t_start = time.perf_counter() - t1

    def progress(member_set, timing):
        step = member_set.ensemble.step
        if step % max(1, steps // 8) == 0 or step == steps:
            print(f"step {step}/{steps} wall {timing.wall_s:.2f}s alive {len(member_set.alive())}", flush=True)

    t2 = time.perf_counter()
    t_write = 0.0
    done = 0
    for stop in sorted(s for s in set(args.write_at or ()) if 0 < s < steps) + [steps]:
        members.advance(stop - done, mark_every=args.mark_every, progress=progress)
        done = stop
        if stop < steps:
            tw = time.perf_counter()
            members.finish_recipes()
            members.write(Path(args.outdir, f"step{stop:06d}"))
            t_write += time.perf_counter() - tw
    t_run = time.perf_counter() - t2 - t_write
    members.finish_recipes()
    paths = members.write(args.outdir)
    seconds = members.member_seconds()
    per = seconds["mean_per_member_step_s"]
    steps_per_day = 86400.0 / float(cfg.dt_s)
    report = {
        "config": cfg.name, "config_hash": cfg.config_hash, "truncation": int(cfg.truncation),
        "dt_s": float(cfg.dt_s), "steps": steps, "members": members.labels,
        "dead": {members.labels[k]: v for k, v in members.dead.items()},
        "wall_build_s": t_build, "wall_draw_s": t_start, "wall_steps_s": t_run,
        "member_step": seconds,
        "member_day_s_from_mean_step": None if per is None else per * steps_per_day,
        "member_day_s_from_wall": t_run / max(1, len(members.alive())) * steps_per_day / max(1, steps),
        "resident": members.ensemble.resident_bytes(),
        "memory": _pool_peak(transform.backend),
        "checkpoints": [p.name for p in paths],
        "recipes": {members.labels[k]: {"sha256": recipe_sha256(r), "bytes": len(json.dumps(r, indent=1, sort_keys=True))}
                    for k, r in enumerate(members.recipes)},
        "final_marks": {members.labels[k]: r["marks"][-1] if r["marks"] else None for k, r in enumerate(members.recipes)},
        "stack": software_stack(transform.backend),
    }
    Path(args.outdir, "members-run.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("member_day_s_from_mean_step", "member_day_s_from_wall", "dead", "final_marks")}, indent=1))


def cmd_replay(args) -> None:
    recipe = read_recipe(args.recipe)
    t0 = time.perf_counter()
    out = replay(recipe, _builder(args.config))
    wall = time.perf_counter() - t0
    mark = out.recipes[0]["marks"][-1]
    want = recipe["marks"][-1]
    result = {"recipe": Path(args.recipe).name, "recipe_sha256": recipe_sha256(recipe), "wall_s": wall,
              "replayed_final": mark, "recipe_final": want,
              "identical": mark["arrays_sha256"] == want["arrays_sha256"]
              and mark["checkpoint_sha256"] == want["checkpoint_sha256"],
              "member_step": out.member_seconds(),
              "stack": software_stack(out.ensemble.transform.backend),
              "memory": _pool_peak(out.ensemble.transform.backend)}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir, "replay.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))


def cmd_branch(args) -> None:
    parent = read_recipe(args.parent_recipe)
    build = _builder(args.config)
    record = dict(parent["config"])
    if args.dt_s is not None:
        record["dt_s"] = args.dt_s
        record["name"] = record["name"] + f"-dt{int(args.dt_s)}"
    cfg, model, transform, _cold = build(record)
    draws = []
    for text in args.draw or ():
        k, _, w = text.partition(":")
        draws.append({"member": int(k), "weight": float(w or 1.0)})
    twin = MemberSet.branch(cfg, model, transform, args.parent_checkpoint, parent, draws=draws,
                            change=args.change, allow_config_branch=args.allow_config_branch)
    steps = int(round(args.hours * 3600.0 / cfg.dt_s))
    twin.advance(steps, mark_every=args.mark_every)
    twin.finish_recipes()
    [path] = twin.write(args.outdir)
    result = {"checkpoint": path.name, "final": twin.recipes[0]["marks"][-1],
              "recipe_sha256": recipe_sha256(twin.recipes[0]), "member_step": twin.member_seconds(),
              "branch": {k: v for k, v in twin.recipes[0]["branch"].items() if k != "parent"}}
    Path(args.outdir, "branch.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start")
    s.add_argument("config")
    s.add_argument("--truncation", type=int, default=127)
    s.add_argument("--members", type=int, default=3)
    s.add_argument("--seed", type=int, default=EnsembleOptions.seed)
    s.add_argument("--control", action="store_true")
    s.add_argument("--hours", type=float, default=24.0)
    s.add_argument("--mark-every", type=int, default=0)
    s.add_argument("--write-at", type=int, action="append",
                   help="also write every member's checkpoint and recipe at this step (a branch point)")
    s.add_argument("--outdir", required=True)
    s.set_defaults(func=cmd_start)
    r = sub.add_parser("replay")
    r.add_argument("config")
    r.add_argument("recipe")
    r.add_argument("--outdir", required=True)
    r.set_defaults(func=cmd_replay)
    b = sub.add_parser("branch")
    b.add_argument("config")
    b.add_argument("parent_checkpoint")
    b.add_argument("parent_recipe")
    b.add_argument("--draw", action="append")
    b.add_argument("--change", default="")
    b.add_argument("--dt-s", type=float, default=None)
    b.add_argument("--allow-config-branch", action="store_true")
    b.add_argument("--hours", type=float, default=1.0)
    b.add_argument("--mark-every", type=int, default=0)
    b.add_argument("--outdir", required=True)
    b.set_defaults(func=cmd_branch)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
