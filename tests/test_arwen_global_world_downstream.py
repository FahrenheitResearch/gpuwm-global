"""The world downstream of the [world] table (other-worlds plan, package 1.2).

Three promises, each a gate here:

1. World off, the cold state is byte-identical.  An Earth config (no table,
   or an all-Earth one) runs no world code: the cold-start hook never calls
   the frame editor, no world record is stamped on the physics metadata,
   and every array, the terrain and the provenance are those of a config
   without a table.
2. A restart under another world is refused with the named reason (a
   ``WorldMismatchError`` whose message starts "world mismatch" and names
   both worlds), before the config-hash comparison that would refuse it
   without saying why.
3. An old checkpoint reads as Earth: a checkpoint (or receipt) with no world
   record is Earth's, so an Earth run resumes from every checkpoint written
   before worlds existed and a world run refuses them.

The ten-step bit gate against the 0.1.6 line itself is
``test_arwen_global_world_spec.test_ten_steps_with_an_all_earth_world_are_the_bytes_of_ten_without``
plus the cross-tree run named in the package report.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.analysis_initial import analysis_initial_state
from arwen_global.checkpoint import (
    WORLD_RECORD_KEY,
    WorldMismatchError,
    bundle_arrays,
    check_checkpoint_world,
    checkpoint_world_identity,
    checkpoint_world_record,
    read_checkpoint,
    read_checkpoint_header,
)
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.receipt import receipt_world_identity, world_receipt_rows
from arwen_global.runner import (
    adopt_checkpoint_clock,
    build_model_and_cold_state,
    build_transform,
    frame_world_reason,
    run,
)
from arwen_global.world import EARTH, parse_world

from test_arwen_global_analysis_initial import _analysis_cfg, _synthetic_frame

SMOKE = config_root() / "arwen_global_moist_smoke.toml"
EDITOR = "arwen_global.world.frame_edits"

ALL_EARTH = (
    '[world]\nlabel = "earth"\n'
    "[world.terrain]\nscale = 1.0\nair_mass = \"conserve\"\n"
    "[world.sun]\nobliquity_deg = 23.5\nbrightness = 1.0\nday_length_h = 24.0\n"
    "[world.sea]\nlevel_m = 0.0\nland = \"keep\"\ntemperature = \"analysis\"\n"
    "ice = \"analysis\"\n"
    "[world.air]\nco2_ppm = \"dated\"\n"
)
ON_ITS_SIDE = '[world]\nlabel = "on-its-side"\n[world.sun]\nobliquity_deg = 90.0\n'
BRIGHT = '[world]\nlabel = "bright"\n[world.sun]\nbrightness = 1.1\n'
FLAT = '[world]\nlabel = "flat"\n[world.terrain]\nscale = 0.0\n'


def _smoke_cfg(tmp_path: Path, extra: str, tag: str):
    path = tmp_path / f"smoke-{tag}.toml"
    path.write_text(SMOKE.read_text(encoding="utf-8") + "\n" + extra,
                    encoding="utf-8")
    return load_config(path)


def _world_analysis_cfg(tmp_path: Path, extra: str, tag: str):
    """The analysis-mode smoke config of the cold-start gates, with a world."""
    base = _analysis_cfg(tmp_path)
    world = parse_world(None) if not extra else load_config(
        _write(tmp_path / f"w-{tag}.toml",
               SMOKE.read_text(encoding="utf-8") + "\n" + extra)).world
    return replace(base, world=world)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class _Statement:
    """Package 1.6's statement, held to one that refuses nothing."""

    sha256 = "0" * 64
    report = SimpleNamespace(raise_if_refused=lambda: None)

    def record(self):
        return {"text": "test statement"}


#: Sibling packages the hooks in this package's files call (2.2's frame
#: start, 1.3's sun, 1.6's statement).  These gates are about the plumbing,
#: so each is held to a pass-through here: a sun world on the reference
#: suite is refused by 1.3 by name, which is 1.3's gate, not this one.
SIBLINGS = {
    "arwen_global.world.start": SimpleNamespace(
        world_start_frame=lambda world, frame: (frame, None)),
    "arwen_global.world.radiation": SimpleNamespace(
        attach_sun=lambda physics, world: physics),
    "arwen_global.world.statement": SimpleNamespace(
        describe_config=lambda cfg, *, frame, levels_pa: _Statement()),
    "arwen_global.world.folder": SimpleNamespace(world_map_for=lambda world, *_, **__: None),
    "arwen_global.world.frame": SimpleNamespace(
        invented_world_frame=lambda world, frame, world_map, **_: frame),
}


def _hold_siblings(patch) -> None:
    for name, stub in SIBLINGS.items():
        patch.setitem(sys.modules, name, stub)
    # Package 3.3's air refusal lives in an existing module (native_options),
    # so it is held by attribute: these gates run sun worlds, never air ones.
    from arwen_global.physics import native_options

    patch.setattr(native_options, "world_air_physics_refusal",
                  lambda world, physics_mode: None, raising=False)


@pytest.fixture(autouse=True)
def _siblings_held(monkeypatch):
    _hold_siblings(monkeypatch)


def _refusing_editor(monkeypatch):
    """An editor that fails the gate if anything calls it."""
    def edit_frame(*_args, **_kwargs):
        raise AssertionError("an Earth or sun-only world called the frame editor")

    monkeypatch.setitem(sys.modules, EDITOR, SimpleNamespace(edit_frame=edit_frame))


def _cold_arrays(cfg, transform):
    state, phi, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame())
    to_numpy = transform.backend.to_numpy
    return (bundle_arrays(state, to_numpy), np.asarray(to_numpy(phi)),
            provenance, dict(state.physics_state.metadata))


def _assert_same_bits(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].dtype == right[name].dtype, name
        assert left[name].tobytes() == right[name].tobytes(), name


# -- 1. the cold-start hook ------------------------------------------------


def test_an_earth_cold_start_runs_no_world_code_and_its_bits_are_unchanged(
    tmp_path, monkeypatch,
):
    _refusing_editor(monkeypatch)
    # Every sibling module is made unimportable: an Earth start that
    # reached any world code would raise.
    for name in (EDITOR, *SIBLINGS):
        monkeypatch.setitem(sys.modules, name, None)
    bare = _analysis_cfg(tmp_path)
    earth = _world_analysis_cfg(tmp_path, ALL_EARTH, "earth")
    assert bare.world.is_earth and earth.world.is_earth
    assert bare.config_hash == earth.config_hash
    transform = build_transform(bare)
    arrays_a, phi_a, prov_a, meta_a = _cold_arrays(bare, transform)
    arrays_b, phi_b, prov_b, meta_b = _cold_arrays(earth, transform)
    _assert_same_bits(arrays_a, arrays_b)
    assert phi_a.tobytes() == phi_b.tobytes()
    assert prov_a == prov_b and "world" not in prov_a
    assert meta_a == meta_b and WORLD_RECORD_KEY not in meta_a


def test_the_hook_hands_the_decoded_frame_to_the_editor_before_any_field_is_read(
    tmp_path, monkeypatch,
):
    calls = []
    order = []

    def describe_config(cfg, *, frame, levels_pa):
        order.append("statement")
        return _Statement()

    def world_start_frame(world, frame):
        order.append("start")
        return frame, None

    monkeypatch.setitem(sys.modules, "arwen_global.world.statement",
                        SimpleNamespace(describe_config=describe_config))
    monkeypatch.setitem(sys.modules, "arwen_global.world.start",
                        SimpleNamespace(world_start_frame=world_start_frame))

    def edit_frame(frame, world, *, levels_pa, grid):
        order.append("terrain")
        calls.append((world.label, np.array(levels_pa), grid))
        frame.fields["terrain_height"].values = np.zeros_like(
            frame.fields["terrain_height"].values)
        frame.fields["surface_pressure"].values = np.full_like(
            frame.fields["surface_pressure"].values, 101000.0)
        return frame, {"rule": "test-editor", "lowered_columns": 7}

    monkeypatch.setitem(sys.modules, EDITOR, SimpleNamespace(edit_frame=edit_frame))
    cfg = _world_analysis_cfg(tmp_path, FLAT, "flat")
    transform = build_transform(cfg)
    state, phi, provenance = analysis_initial_state(
        cfg, transform, frame=_synthetic_frame())
    assert len(calls) == 1
    # The preflight sees the frame before the edit; the start sees it after.
    assert order == ["statement", "terrain", "start"]
    label, levels_pa, grid = calls[0]
    assert label == "flat" and grid is transform.grid
    # In Pa and ascending top to bottom, as the cold start reads them.
    assert np.all(np.diff(levels_pa) > 0.0) and levels_pa[-1] == 100000.0
    # The dynamics' terrain is the edited frame's: flat to rounding.
    assert np.max(np.abs(transform.backend.to_numpy(phi))) < 1.0e-6
    world = provenance["world"]
    assert world["record"] == cfg.world.record()
    assert world["frame_edit"] == {"rule": "test-editor", "lowered_columns": 7}
    # The cold start never stamps the record itself; the model builder does.
    assert WORLD_RECORD_KEY not in state.physics_state.metadata


def test_a_frame_world_in_a_build_without_the_editor_is_refused_by_name(
    tmp_path, monkeypatch,
):
    monkeypatch.setitem(sys.modules, EDITOR, None)
    cfg = _world_analysis_cfg(tmp_path, FLAT, "flat")
    with pytest.raises(ValueError, match="carries no frame editor"):
        analysis_initial_state(cfg, build_transform(cfg), frame=_synthetic_frame())


def test_a_sun_world_leaves_the_frame_as_decoded(tmp_path, monkeypatch):
    _refusing_editor(monkeypatch)
    side = _world_analysis_cfg(tmp_path, ON_ITS_SIDE, "side")
    assert frame_world_reason(side.world) is None
    transform = build_transform(side)
    arrays_w, phi_w, prov_w, _ = _cold_arrays(side, transform)
    arrays_e, phi_e, prov_e, _ = _cold_arrays(_analysis_cfg(tmp_path), transform)
    _assert_same_bits(arrays_w, arrays_e)
    assert phi_w.tobytes() == phi_e.tobytes()
    assert prov_w.pop("world")["frame_edit"].startswith("none:")
    assert prov_w.pop("world_sea") == {"frame": None, "open_water": None}
    assert prov_w.pop("world_statement") == {"sha256": "0" * 64,
                                             "text": "test statement"}
    assert prov_w == prov_e


def test_an_earth_model_build_reaches_no_world_module(tmp_path, monkeypatch):
    for name in (EDITOR, *SIBLINGS):
        monkeypatch.setitem(sys.modules, name, None)
    cfg = _smoke_cfg(tmp_path, ALL_EARTH, "earth")
    model, cold = build_model_and_cold_state(cfg)
    assert model.world_spec.is_earth
    assert WORLD_RECORD_KEY not in cold.physics_state.metadata


def test_a_day_length_world_spins_the_dynamics_from_the_planet_clock(
    tmp_path, monkeypatch,
):
    """Package 3.1's patch: the rotation rate comes from world.clock's one
    function; an idealized start of another rotation is not refused."""
    from arwen_global.dynamics import EARTH_ROTATION_RATE_S

    seen = []
    planet = SimpleNamespace(rotation_rate_s=2.0 * EARTH_ROTATION_RATE_S)

    def from_world(world):
        seen.append(world.label)
        return planet

    monkeypatch.setitem(sys.modules, "arwen_global.world.clock", SimpleNamespace(
        PlanetClock=SimpleNamespace(from_world=from_world)))
    cfg = _smoke_cfg(tmp_path, '[world]\nlabel = "fast"\n[world.sun]\n'
                     "day_length_h = 12.0\n", "fast")
    model, _cold = build_model_and_cold_state(cfg)
    assert seen == ["fast"] and model.rotation_rate_s == planet.rotation_rate_s
    earth, _ = build_model_and_cold_state(_smoke_cfg(tmp_path, "", "earth"))
    assert earth.rotation_rate_s == EARTH_ROTATION_RATE_S


def test_a_frame_world_on_a_start_with_no_frame_is_refused(tmp_path):
    cfg = _smoke_cfg(tmp_path, FLAT, "flat")
    assert cfg.initial_mode != "analysis"
    with pytest.raises(ValueError, match="world 'flat' changes its terrain.*no frame"):
        build_model_and_cold_state(cfg)


# -- 2. the record in checkpoints and receipts ------------------------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One Earth and one on-its-side smoke run (four steps, checkpoints at
    steps 0, 2 and 4), shared by the gates below."""
    root = tmp_path_factory.mktemp("world-runs")
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        _hold_siblings(patch)
        for tag, extra in (("earth", ""), ("side", ON_ITS_SIDE), ("bright", BRIGHT)):
            cfg = _smoke_cfg(root, extra, tag)
            receipt = run(cfg, root / f"run-{tag}")
            assert receipt["status"] == "pass", tag
            out[tag] = SimpleNamespace(cfg=cfg, receipt=receipt, dir=root / f"run-{tag}")
    out["root"] = root
    return out


def _step(run_dir: Path, step: int) -> Path:
    return run_dir / f"arwen_global_step{step:08d}.npz"


def test_every_checkpoint_and_the_receipt_of_a_world_run_carry_its_record(runs):
    side = runs["side"]
    record = side.cfg.world.record()
    for step in (0, 2, 4):
        header = read_checkpoint_header(_step(side.dir, step))
        assert header["physics_metadata"][WORLD_RECORD_KEY] == record, step
        assert checkpoint_world_identity(header) == side.cfg.world.identity_hash
    assert side.receipt["world"] == record
    assert receipt_world_identity(side.receipt) == side.cfg.world.identity_hash
    # The world joins the config hash; the label does not.
    assert side.receipt["config_hash"] != runs["earth"].receipt["config_hash"]


def test_an_earth_run_writes_no_world_record_anywhere(runs):
    earth = runs["earth"]
    for step in (0, 2, 4):
        header = read_checkpoint_header(_step(earth.dir, step))
        assert WORLD_RECORD_KEY not in header["physics_metadata"], step
        assert checkpoint_world_record(header) is None
    assert "world" not in earth.receipt
    assert "world" not in earth.receipt["initial"]["provenance"]
    assert world_receipt_rows(EARTH) == {} and world_receipt_rows(None) == {}


# -- 3. the cross-world restart refusal --------------------------------------


@pytest.mark.parametrize("source,target,written,running", [
    ("side", "earth", "world 'on-its-side'", "Earth"),
    ("earth", "side", "Earth", "world 'on-its-side'"),
    ("side", "bright", "world 'on-its-side'", "world 'bright'"),
])
def test_a_restart_under_another_world_is_refused_with_the_named_reason(
    runs, source, target, written, running,
):
    checkpoint = _step(runs[source].dir, 2)
    outdir = runs["root"] / f"restart-{source}-under-{target}"
    with pytest.raises(WorldMismatchError) as caught:
        run(runs[target].cfg, outdir, restart=checkpoint)
    message = str(caught.value)
    assert message.startswith("world mismatch: checkpoint ")
    assert f"integrated on {written}" in message
    assert f"this config runs {running}" in message
    # Refused before anything was built or written.
    assert not outdir.exists() or not any(outdir.iterdir())


def test_a_restart_under_the_same_world_continues_the_run_bit_for_bit(runs):
    side = runs["side"]
    outdir = runs["root"] / "restart-side-under-side"
    receipt = run(side.cfg, outdir, restart=_step(side.dir, 2))
    assert receipt["status"] == "pass" and receipt["world"] == side.cfg.world.record()
    resumed_meta, resumed_arrays = read_checkpoint(
        _step(outdir, 4), expected_world=side.cfg.world)
    straight_meta, straight_arrays = read_checkpoint(_step(side.dir, 4))
    _assert_same_bits(resumed_arrays, straight_arrays)
    assert resumed_meta["physics_metadata"] == straight_meta["physics_metadata"]


def test_the_label_is_a_name_and_never_refuses_a_restart(runs):
    side = runs["side"]
    renamed = replace(side.cfg.world, label="sideways")
    header = read_checkpoint_header(_step(side.dir, 2))
    check_checkpoint_world(header["physics_metadata"], renamed, "checkpoint x")


def test_every_door_that_adopts_a_checkpoint_refuses_another_world(runs):
    """A cycle, a tape export or a parent export rebuilds a model from its
    config and adopts the checkpoint through adopt_checkpoint_clock."""
    side = runs["side"]
    metadata, arrays = read_checkpoint(_step(side.dir, 2))
    state = SimpleNamespace(
        physics_state=SimpleNamespace(metadata=dict(metadata["physics_metadata"])))
    earth_model = SimpleNamespace(world_spec=EARTH, forecast_clock=None)
    with pytest.raises(WorldMismatchError, match="world mismatch: tape source"):
        adopt_checkpoint_clock(earth_model, state, "tape source")
    side_model = SimpleNamespace(world_spec=side.cfg.world, forecast_clock=None)
    adopt_checkpoint_clock(side_model, state, "tape source")


# -- 4. an old checkpoint reads as Earth ---------------------------------------


def test_an_old_checkpoint_and_an_old_receipt_read_as_earth(runs):
    earth = runs["earth"]
    path = _step(earth.dir, 2)
    # A checkpoint with no world record is exactly what every checkpoint
    # before worlds was: the Earth run's bytes are those of the 0.1.6 line
    # (the ten-step gate), and it carries no record.
    header = read_checkpoint_header(path)
    assert checkpoint_world_identity(header) == "none"
    assert checkpoint_world_identity(header["physics_metadata"]) == "none"
    assert checkpoint_world_identity({}) == "none"
    # An Earth reader takes it; a world reader refuses it by name.
    read_checkpoint(path, expected_world=EARTH)
    with pytest.raises(WorldMismatchError, match="integrated on Earth"):
        read_checkpoint(path, expected_world=runs["side"].cfg.world)
    assert receipt_world_identity({"schema": "x"}) == "none"
    assert receipt_world_identity(None) == "none"
