"""The member engine on the T3 smoke configuration (numpy, reference
physics).

What is held: members started here equal the filter's ensemble members
with the same options, array for array; stepping a member set equals the
filter's ensemble step under both cores; a recipe replays its member alone
in a fresh model to the recorded digests and refuses a changed recipe; a
zero-weight draw is the control byte for byte; one member's death leaves
the others byte-identical to a set where nobody died; a twin branched from
a member checkpoint with no change continues its parent bit for bit; a
branch under another config is refused without the opt-in and recorded
with it, and its recipe replays; the run door's restart under another
config is refused unless ``branch_config`` is given, and the receipt
records the opt-in only then.
"""
from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.constants import SPECTRAL_FIELDS
from arwen_global.da import EnsembleOptions, GlobalEnsemble, ensemble_config
from arwen_global.da.ensemble import recut_config
from arwen_global.members import (
    RECIPE_SCHEMA,
    MemberSet,
    ReplayMismatch,
    read_recipe,
    replay,
)
from arwen_global.runner import build_model_and_cold_state, build_transform, run

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _cut(integrator=None):
    cfg = load_config(CONFIG)
    if integrator is not None:
        cfg = dataclasses.replace(cfg, integrator=integrator)
    options = EnsembleOptions(members=3, truncation=cfg.truncation, seed=11, additive_inflation_fraction=0.0)
    return cfg, options, ensemble_config(cfg, options)


def _fresh(ecfg):
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    return model, transform, cold


def _builder(base_cfg):
    def build(record):
        cfg = recut_config(base_cfg, int(record["truncation"]), dt_s=float(record["dt_s"]), name=record["name"])
        model, transform, cold = _fresh(cfg)
        return cfg, model, transform, cold
    return build


def _arrays(state, to_numpy):
    out = {name: np.asarray(to_numpy(getattr(state.atmosphere, name))) for name in SPECTRAL_FIELDS}
    for name, value in state.atmosphere.grid_tracers().items():
        if value is not None:
            out[f"grid__{name}"] = np.asarray(to_numpy(value))
    for name, value in state.physics_state.arrays.items():
        out[f"physics__{name}"] = np.asarray(to_numpy(value))
    return out


def _same(a, b, to_numpy):
    x, y = _arrays(a, to_numpy), _arrays(b, to_numpy)
    assert x.keys() == y.keys()
    for name in x:
        assert np.array_equal(x[name], y[name]), name
        assert x[name].tobytes() == y[name].tobytes(), name


def test_started_members_are_the_filters_members_bit_for_bit():
    _cfg, options, ecfg = _cut()
    model, transform, cold = _fresh(ecfg)
    filt = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    model2, transform2, cold2 = _fresh(ecfg)
    mine = MemberSet.start(ecfg, model2, transform2, cold2, options, range(3))
    assert mine.ensemble.seeds == filt.seeds
    for k in range(3):
        _same(mine.ensemble.members[k], filt.members[k], transform.backend.to_numpy)
        assert mine.ensemble.targets[k] == filt.targets[k]
        assert mine.recipes[k]["schema"] == RECIPE_SCHEMA
        assert mine.recipes[k]["perturbation"]["draws"] == [{"member": k, "weight": 1.0}]


@pytest.mark.parametrize("integrator", [None, "sl_si"])
def test_stepping_a_member_set_is_the_filters_step(integrator):
    _cfg, options, ecfg = _cut(integrator)
    model, transform, cold = _fresh(ecfg)
    filt = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    model2, transform2, cold2 = _fresh(ecfg)
    mine = MemberSet.start(ecfg, model2, transform2, cold2, options, range(3))
    for _ in range(3):
        filt.step_all(ecfg.dt_s)
        mine.step_all(ecfg.dt_s)
    for k in range(3):
        _same(mine.ensemble.members[k], filt.members[k], transform.backend.to_numpy)


@pytest.mark.parametrize("integrator", [None, "sl_si"])
def test_a_recipe_replays_its_member_alone_and_refuses_a_changed_recipe(tmp_path, integrator):
    cfg, options, ecfg = _cut(integrator)
    model, transform, cold = _fresh(ecfg)
    members = MemberSet.start(ecfg, model, transform, cold, options, [[], 0, 1, 2])
    members.advance(4, mark_every=2)
    members.finish_recipes()
    members.write(tmp_path)
    recipe = read_recipe(next(tmp_path.glob("m002_*.recipe.json")))
    assert [m["step"] for m in recipe["marks"]] == [2, 4]
    assert len(json.dumps(recipe)) < 8192, "a recipe is a few kilobytes"
    out = replay(recipe, _builder(cfg))
    assert out.recipes[0]["marks"][-1]["arrays_sha256"] == recipe["marks"][-1]["arrays_sha256"]
    assert out.recipes[0]["marks"][-1]["checkpoint_sha256"] == recipe["marks"][-1]["checkpoint_sha256"]
    # The written checkpoint is the one the recipe names.
    [ckpt] = tmp_path.glob("m002_*.npz")
    meta, _ = read_checkpoint(ckpt)
    assert meta["self_sha256"] == recipe["marks"][-1]["checkpoint_sha256"]
    # The control replays too.
    control = read_recipe(next(tmp_path.glob("control_*.recipe.json")))
    replay(control, _builder(cfg))
    # Any change is refused.
    changed = copy.deepcopy(recipe)
    changed["perturbation"]["draws"][0]["weight"] = 1.0000001
    with pytest.raises(ReplayMismatch):
        replay(changed, _builder(cfg))
    other_base = copy.deepcopy(recipe)
    other_base["base"]["arrays_sha256"] = "0" * 64
    with pytest.raises(ReplayMismatch, match="base state"):
        replay(other_base, _builder(cfg))


def test_a_zero_weight_draw_is_the_control_byte_for_byte():
    _cfg, options, ecfg = _cut("sl_si")
    model, transform, cold = _fresh(ecfg)
    members = MemberSet.start(ecfg, model, transform, cold, options,
                              [[], [{"member": 0, "weight": 0.0}], [{"member": 1, "weight": 0.0}, {"member": 2, "weight": 0.0}]])
    members.advance(2, mark_every=1)
    to_numpy = transform.backend.to_numpy
    _same(members.ensemble.members[0], members.ensemble.members[1], to_numpy)
    _same(members.ensemble.members[0], members.ensemble.members[2], to_numpy)
    assert members.recipes[0]["marks"] == members.recipes[1]["marks"] == members.recipes[2]["marks"]


def test_a_blend_is_the_weighted_sum_of_its_draws():
    _cfg, options, ecfg = _cut()
    model, transform, cold = _fresh(ecfg)
    members = MemberSet.start(ecfg, model, transform, cold, options,
                              [[], [{"member": 0, "weight": 0.5}, {"member": 1, "weight": -0.25}]])
    from arwen_global.members import draw_increment

    to_numpy = transform.backend.to_numpy
    i0, _, _ = draw_increment(model, transform, cold.atmosphere, options, 0)
    model.release_syntheses()
    i1, _, _ = draw_increment(model, transform, cold.atmosphere, options, 1)
    blend = members.ensemble.members[1].atmosphere
    control = members.ensemble.members[0].atmosphere
    expect = np.asarray(to_numpy(i0["theta"] * 0.5 + i1["theta"] * -0.25))
    got = np.asarray(to_numpy(blend.theta)) - np.asarray(to_numpy(control.theta))
    assert np.allclose(got, expect, rtol=0.0, atol=1.0e-9)
    assert members.labels == ["control", "blend001"]


def test_one_members_death_leaves_the_others_byte_identical(monkeypatch):
    _cfg, options, ecfg = _cut("sl_si")
    model, transform, cold = _fresh(ecfg)
    clean = MemberSet.start(ecfg, model, transform, cold, options, range(3))
    clean.advance(3)
    model2, transform2, cold2 = _fresh(ecfg)
    hurt = MemberSet.start(ecfg, model2, transform2, cold2, options, range(3))
    real_step = model2.step
    calls = {"n": 0}

    def dying_step(state, dt_s):
        calls["n"] += 1
        if calls["n"] == 5:  # member 1's second step
            raise FloatingPointError("native physics produced non-finite theta")
        return real_step(state, dt_s)

    monkeypatch.setattr(model2, "step", dying_step)
    hurt.advance(3)
    assert set(hurt.dead) == {1} and "non-finite theta" in hurt.dead[1]
    to_numpy = transform.backend.to_numpy
    for k in (0, 2):
        _same(hurt.ensemble.members[k], clean.ensemble.members[k], to_numpy)
    assert hurt.ensemble.members[1].step == 1
    hurt.finish_recipes()
    assert hurt.recipes[1]["died"].startswith("step 2:")


def test_an_unchanged_twin_continues_its_parent_bit_for_bit(tmp_path):
    cfg, options, ecfg = _cut("sl_si")
    model, transform, cold = _fresh(ecfg)
    parent = MemberSet.start(ecfg, model, transform, cold, options, [1])
    parent.advance(2)
    [ckpt] = parent.write(tmp_path / "at2")
    parent_recipe = read_recipe(next((tmp_path / "at2").glob("*.recipe.json")))
    parent.advance(3)
    model2, transform2, _ = _fresh(ecfg)
    twin = MemberSet.branch(ecfg, model2, transform2, ckpt, parent_recipe, change="none")
    twin.advance(3)
    assert twin.recipes[0]["marks"][-1] == parent.recipes[0]["marks"][-1]
    _same(twin.ensemble.members[0], parent.ensemble.members[0], transform.backend.to_numpy)
    assert twin.recipes[0]["branch"]["config_differs"] is False
    # A twin with a draw differs and replays from its recipe.
    model3, transform3, _ = _fresh(ecfg)
    kicked = MemberSet.branch(ecfg, model3, transform3, ckpt, parent_recipe,
                              draws=[{"member": 7, "weight": 0.1}], change="draw 7 at 0.1")
    kicked.advance(3)
    assert kicked.recipes[0]["marks"][-1]["arrays_sha256"] != parent.recipes[0]["marks"][-1]["arrays_sha256"]
    out = replay(kicked.recipes[0], _builder(cfg))
    assert out.recipes[0]["marks"][-1] == kicked.recipes[0]["marks"][-1]


def test_a_branch_under_another_config_is_an_explicit_opt_in(tmp_path):
    cfg, options, ecfg = _cut("sl_si")
    model, transform, cold = _fresh(ecfg)
    parent = MemberSet.start(ecfg, model, transform, cold, options, [0])
    parent.advance(2)
    [ckpt] = parent.write(tmp_path)
    parent_recipe = read_recipe(next(tmp_path.glob("*.recipe.json")))
    child_cfg = recut_config(cfg, ecfg.truncation, dt_s=ecfg.dt_s / 2.0, name=ecfg.name + "-half")
    assert child_cfg.config_hash != ecfg.config_hash
    model2, transform2, _ = _fresh(child_cfg)
    with pytest.raises(ValueError, match="explicit opt-in"):
        MemberSet.branch(child_cfg, model2, transform2, ckpt, parent_recipe)
    twin = MemberSet.branch(child_cfg, model2, transform2, ckpt, parent_recipe,
                            allow_config_branch=True, change="half the step")
    branch = twin.recipes[0]["branch"]
    assert branch["config_differs"] is True and branch["allow_config_branch"] is True
    assert branch["parent_config_hash"] == ecfg.config_hash
    twin.advance(2)
    out = replay(twin.recipes[0], _builder(cfg))
    assert out.recipes[0]["marks"][-1] == twin.recipes[0]["marks"][-1]


def test_the_run_door_branches_under_another_config_only_on_the_opt_in(tmp_path):
    cfg = load_config(CONFIG)
    run(cfg, tmp_path / "parent")
    midpoint = tmp_path / "parent" / "arwen_global_step00000002.npz"
    child = dataclasses.replace(cfg, name=cfg.name + "-twin")
    assert child.config_hash != cfg.config_hash
    with pytest.raises(ValueError, match="config identity"):
        run(child, tmp_path / "refused", restart=midpoint)
    with pytest.raises(ValueError, match="--restart"):
        run(child, tmp_path / "nothing", branch_config=True)
    result = run(child, tmp_path / "twin", restart=midpoint, branch_config=True)
    assert result["branch"]["parent_config_hash"] == cfg.config_hash
    assert result["branch"]["config_hash"] == child.config_hash
    assert result["branch"]["config_differs"] is True
    # A change of name moves no arithmetic: the twin's arrays are the parent's.
    _, a = read_checkpoint(tmp_path / "parent" / "arwen_global_step00000004.npz")
    _, b = read_checkpoint(tmp_path / "twin" / "arwen_global_step00000004.npz")
    for name in a:
        assert np.array_equal(a[name], b[name]), name
    plain = run(cfg, tmp_path / "plain", restart=midpoint)
    assert "branch" not in plain
