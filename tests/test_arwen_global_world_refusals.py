"""Refusals downstream (worlds plan, package 2.9).

The doors that assume Earth -- the regional parent export, cycling and
assimilation, and the scorecards -- refuse a run on another world by name,
before any work, and say why.  One test per door asserts the reason's
text.  Each door is also shown to pass an Earth input unchanged, since an
Earth run carries no world record.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.world import EARTH, parse_world
from arwen_global.world.refusals import (
    WHY,
    WorldRefusedError,
    describe_world,
    refuse_on_world,
    stored_world_record,
    world_refusal,
)

SMOKE = config_root() / "arwen_global_moist_smoke.toml"
FLAT = '[world]\nlabel = "flat-earth"\n\n[world.terrain]\nscale = 0.0\n'
WATER = ('[world]\nlabel = "waterworld"\n\n[world.terrain]\nscale = 0.0\n\n'
         '[world.sea]\nland = "none"\ntemperature = "zonal-mean"\n')
ALL_EARTH = '[world]\nlabel = "earth"\n\n[world.terrain]\nscale = 1.0\n\n[world.sun]\nobliquity_deg = 23.5\n'


def _config(tmp_path: Path, extra: str = "", name: str = "w.toml") -> Path:
    path = tmp_path / name
    path.write_text(SMOKE.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    return path


def _cfg(tmp_path: Path, extra: str = "", name: str = "w.toml"):
    return load_config(_config(tmp_path, extra, name))


def _flat_world():
    import tomllib

    return parse_world(tomllib.loads(FLAT)["world"])


def _fake_checkpoint(path: Path, *, world_record=None, step: int = 0,
                     time_s: float = 0.0) -> Path:
    """A file :func:`checkpoint.read_checkpoint_header` reads: the metadata
    member alone, the world record (if any) in the physics metadata where
    the run writes it."""
    physics = {} if world_record is None else {"world": world_record}
    meta = {"step": step, "time_s": time_s, "physics_metadata": physics}
    np.savez(path, __metadata__=np.array(json.dumps(meta)))
    return path


def _run_dir(path: Path, receipt: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "arwen-global-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _expect(door: str, family: str, world, source: str) -> str:
    return f"{door} is refused on {world}, read from {source}: {WHY[family]}"


# -- the reasons ------------------------------------------------------------


def test_the_three_reasons_are_the_planned_ones_word_for_word():
    assert set(WHY) == {"regional", "assimilation", "scorecard"}
    assert WHY["regional"].startswith(
        "the regional engine has no [world] table: it takes terrain, coast "
        "and land use from Earth's geography and runs Earth's sun, rotation "
        "and greenhouse gases")
    assert WHY["assimilation"].startswith(
        "assimilation draws the state toward observations of Earth")
    assert WHY["assimilation"].endswith(
        "Cycling and assimilation stay refused on every world")
    assert WHY["scorecard"].startswith(
        "a scorecard scores a forecast of Earth against Earth's observations "
        "or analyses, and this run is not a forecast of Earth")


def test_earth_is_never_refused_and_reads_nothing_more():
    for subject in (None, EARTH, {}, {"config": {"truncation": 3}},
                    {"physics_metadata": {}}, {"world": EARTH.record()}):
        assert world_refusal(subject, door="d", family="scorecard", source="s") is None
        refuse_on_world(subject, door="d", family="scorecard", source="s")
    assert stored_world_record({"world": EARTH.record()}) is None


def test_an_all_earth_table_is_earth_at_every_door(tmp_path):
    cfg = _cfg(tmp_path, ALL_EARTH)
    assert cfg.world.is_earth
    for family in WHY:
        refuse_on_world(cfg.world, door="d", family=family, source="s")


def test_a_world_is_described_by_label_identity_and_changed_settings():
    world = _flat_world()
    text = describe_world(world)
    assert text == f"world 'flat-earth' (world identity {world.token}, changing terrain.scale)"
    # The same world read back from its stored record.
    assert describe_world({"world": world.record()}) == text
    assert describe_world({"physics_metadata": {"world": world.record()}}) == text


def test_a_receipt_written_before_the_world_record_is_read_from_its_config_identity(tmp_path):
    cfg = _cfg(tmp_path, FLAT)
    receipt = {"config": cfg.config_identity}
    record = stored_world_record(receipt)
    assert record is not None and record["identity"] == cfg.world.identity_hash
    assert describe_world(receipt) == (
        f"an unlabelled world (world identity {cfg.world.token}, changing terrain.scale)")


def test_an_unknown_family_is_a_programming_error():
    with pytest.raises(ValueError, match="unknown refusal family"):
        world_refusal(_flat_world(), door="d", family="other", source="s")


# -- the regional parent export ----------------------------------------------


def test_export_mapped_parent_refuses_a_world_config_before_building(tmp_path):
    from arwen_global.mapped_parent_export import REGIONAL_DOOR, export_mapped_parent

    cfg = _cfg(tmp_path, FLAT)
    with pytest.raises(WorldRefusedError) as caught:
        export_mapped_parent(cfg, [tmp_path / "absent.npz"], tmp_path / "out", nlat=4, nlon=8)
    assert str(caught.value) == _expect(
        REGIONAL_DOOR, "regional", describe_world(cfg.world), f"config {cfg.name!r}")
    assert not (tmp_path / "out").exists()


def test_export_mapped_parent_refuses_a_world_checkpoint_with_nothing_written(tmp_path):
    from datetime import datetime

    from arwen_global.mapped_parent_export import REGIONAL_DOOR, plan_parent_frames

    world = _flat_world()
    earth = _fake_checkpoint(tmp_path / "earth.npz", step=0, time_s=0.0)
    other = _fake_checkpoint(tmp_path / "world.npz", world_record=world.record(),
                             step=1, time_s=3600.0)
    out = tmp_path / "out"
    out.mkdir()
    # Earth's checkpoint plans as before.
    plan = plan_parent_frames([earth], datetime(2026, 11, 10, 18), out)
    assert len(plan) == 1
    with pytest.raises(WorldRefusedError) as caught:
        plan_parent_frames([earth, other], datetime(2026, 11, 10, 18), out)
    assert str(caught.value) == _expect(
        REGIONAL_DOOR, "regional", describe_world(world), f"checkpoint {other}")
    assert list(out.iterdir()) == []


def test_export_mapped_parent_refuses_a_changed_coast(tmp_path):
    from arwen_global.mapped_parent_export import export_mapped_parent

    cfg = _cfg(tmp_path, WATER)
    assert cfg.world.changes_coast
    with pytest.raises(WorldRefusedError, match="takes terrain, coast and land use from Earth's geography"):
        export_mapped_parent(cfg, [tmp_path / "absent.npz"], tmp_path / "out", nlat=4, nlon=8)


def test_the_regional_bridge_parent_export_refuses_a_world(tmp_path):
    from arwen_global.export import PARENT_EXPORT_DOOR, export_parent

    cfg = _cfg(tmp_path, FLAT)
    with pytest.raises(WorldRefusedError) as caught:
        export_parent(cfg, tmp_path / "absent.npz", tmp_path / "out.npz", nlat=4, nlon=8)
    assert str(caught.value) == _expect(
        PARENT_EXPORT_DOOR, "regional", describe_world(cfg.world), f"config {cfg.name!r}")
    world = _flat_world()
    other = _fake_checkpoint(tmp_path / "world.npz", world_record=world.record())
    with pytest.raises(WorldRefusedError) as caught:
        export_parent(_cfg(tmp_path), other, tmp_path / "out.npz", nlat=4, nlon=8)
    assert str(caught.value) == _expect(
        PARENT_EXPORT_DOOR, "regional", describe_world(world), f"checkpoint {other}")


# -- cycling and assimilation -------------------------------------------------


def test_cycle_refuses_a_world_config(tmp_path):
    from arwen_global.cycle import CYCLE_DOOR, cycle

    cfg = _cfg(tmp_path, FLAT)
    with pytest.raises(WorldRefusedError) as caught:
        cycle(cfg, tmp_path / "out", obs_locations=["x"], cycles=1)
    assert str(caught.value) == _expect(
        CYCLE_DOOR, "assimilation", describe_world(cfg.world), f"config {cfg.name!r}")
    assert not (tmp_path / "out").exists()


def test_cycle_refuses_a_restart_from_a_world_checkpoint(tmp_path):
    from arwen_global.cycle import CYCLE_DOOR, cycle

    cfg = _cfg(tmp_path)  # Earth
    world = _flat_world()
    restart = _fake_checkpoint(tmp_path / "w.npz", world_record=world.record())
    with pytest.raises(WorldRefusedError) as caught:
        cycle(cfg, tmp_path / "out", obs_locations=["x"], cycles=1, restart=restart)
    assert str(caught.value) == _expect(
        CYCLE_DOOR, "assimilation", describe_world(world), f"checkpoint {restart}")


@pytest.mark.parametrize("leg", ["init", "analyze", "cycle", "localisation", "forecast"])
def test_every_da_leg_refuses_a_world_config(leg, tmp_path):
    from arwen_global import da_door

    cfg = _cfg(tmp_path, FLAT)
    out = tmp_path / "out"
    calls = {
        "init": lambda: da_door.init(cfg, out),
        "analyze": lambda: da_door.analyze(cfg, tmp_path / "bg.npz", ["x"], out),
        "cycle": lambda: da_door.cycle(cfg, out, cycles=1),
        "localisation": lambda: da_door.localisation(cfg, tmp_path / "ens", out),
        "forecast": lambda: da_door.forecast(cfg, out, analysis=tmp_path / "an.npz"),
    }
    with pytest.raises(WorldRefusedError) as caught:
        calls[leg]()
    assert str(caught.value) == _expect(
        da_door.da_door_name(leg), "assimilation", describe_world(cfg.world),
        f"config {cfg.name!r}")
    assert not out.exists()


def test_da_static_covariance_refuses_a_world_config(tmp_path):
    from arwen_global.da_static import STATIC_COVARIANCE_DOOR, estimate_table

    cfg = _cfg(tmp_path, FLAT)
    with pytest.raises(WorldRefusedError) as caught:
        estimate_table(cfg, [], tmp_path / "out")
    assert str(caught.value) == _expect(
        STATIC_COVARIANCE_DOOR, "assimilation", describe_world(cfg.world),
        f"config {cfg.name!r}")


def test_da_fresh_refuses_a_world_base_config_before_fetching(tmp_path):
    from arwen_global import da_door

    base = _config(tmp_path, FLAT, "base.toml")
    with pytest.raises(WorldRefusedError) as caught:
        da_door.fresh(base, tmp_path / "out")
    assert str(caught.value) == _expect(
        da_door.da_door_name("fresh"), "assimilation", describe_world(_flat_world()),
        "base config base.toml")
    assert not (tmp_path / "out").exists()


def test_assimilate_file_door_refuses_a_world_config(tmp_path):
    from arwen_global.assimilate import ASSIMILATE_DOOR, assimilate

    cfg = _cfg(tmp_path, FLAT)
    with pytest.raises(WorldRefusedError) as caught:
        assimilate(cfg, tmp_path / "bg.npz", ["x"], tmp_path / "out")
    assert str(caught.value) == _expect(
        ASSIMILATE_DOOR, "assimilation", describe_world(cfg.world), f"config {cfg.name!r}")


def test_assimilate_file_door_refuses_a_world_background(tmp_path):
    from arwen_global.assimilate import ASSIMILATE_DOOR, assimilate

    cfg = _cfg(tmp_path)  # Earth
    world = _flat_world()
    background = _fake_checkpoint(tmp_path / "bg.npz", world_record=world.record())
    with pytest.raises(WorldRefusedError) as caught:
        assimilate(cfg, background, ["x"], tmp_path / "out")
    assert str(caught.value) == _expect(
        ASSIMILATE_DOOR, "assimilation", describe_world(world), f"checkpoint {background}")


def test_the_one_analysis_function_refuses_a_resident_world_state(tmp_path):
    """``analyse`` is the one place every increment is formed (the cycle's
    resident path, the file door, ``da analyze``)."""
    from arwen_global.assimilate import ASSIMILATE_DOOR, analyse

    cfg = _cfg(tmp_path)  # Earth
    world = _flat_world()
    state = SimpleNamespace(physics_state=SimpleNamespace(metadata={"world": world.record()}))
    with pytest.raises(WorldRefusedError) as caught:
        analyse(cfg, None, None, state, [], sources=[], background={})
    assert str(caught.value) == _expect(
        ASSIMILATE_DOOR, "assimilation", describe_world(world),
        "the background state's physics metadata")


# -- the scorecards ---------------------------------------------------------


def _world_receipt(tmp_path: Path) -> tuple[Path, object]:
    world = _flat_world()
    run_dir = _run_dir(tmp_path / "run", {
        "name": "flat", "config": {"truncation": 3}, "world": world.record()})
    return run_dir, world


def test_score_refuses_a_world_run(tmp_path):
    from arwen_global.forecast_scorecard import SCORE_DOOR, score_forecast

    run_dir, world = _world_receipt(tmp_path)
    with pytest.raises(WorldRefusedError) as caught:
        score_forecast(run_dir, tmp_path / "card", fetch_references=False)
    assert str(caught.value) == _expect(
        SCORE_DOOR, "scorecard", describe_world(world),
        f"run receipt {run_dir / 'arwen-global-receipt.json'}")


def test_the_observation_scorecard_refuses_a_world_run(tmp_path):
    from arwen_global.obs_scorecard import OBS_SCORECARD_DOOR, model_surface_fields

    run_dir, world = _world_receipt(tmp_path)
    with pytest.raises(WorldRefusedError) as caught:
        model_surface_fields(run_dir, run_dir / "x.npz", surface_geopotential=np.zeros(1))
    assert str(caught.value) == _expect(
        OBS_SCORECARD_DOOR, "scorecard", describe_world(world),
        f"run receipt {run_dir / 'arwen-global-receipt.json'}")


def test_the_upper_air_scorecard_refuses_a_world_run(tmp_path):
    from arwen_global.upper_air_scorecard import UPPER_AIR_DOOR, measure_run

    run_dir, world = _world_receipt(tmp_path)
    with pytest.raises(WorldRefusedError) as caught:
        measure_run(run_dir, [], cache_dir=tmp_path / "cache")
    assert str(caught.value) == _expect(
        UPPER_AIR_DOOR, "scorecard", describe_world(world),
        f"run receipt {run_dir / 'arwen-global-receipt.json'}")
    assert not (tmp_path / "cache").exists()


def test_the_radiation_scorecard_refuses_a_world_run(tmp_path):
    from arwen_global.radiation_scorecard import RADIATION_SCORECARD_DOOR, main

    run_dir, world = _world_receipt(tmp_path)
    with pytest.raises(WorldRefusedError) as caught:
        main(["--run-dir", str(run_dir)])
    assert str(caught.value) == _expect(
        RADIATION_SCORECARD_DOOR, "scorecard", describe_world(world),
        f"run receipt in {run_dir}")


def test_a_scorecard_refuses_a_world_receipt_from_before_the_world_record(tmp_path):
    """A receipt that carries the world only in its config identity."""
    from arwen_global.upper_air_scorecard import read_receipt

    cfg = _cfg(tmp_path, WATER)
    run_dir = _run_dir(tmp_path / "run", {"name": "w", "config": cfg.config_identity})
    with pytest.raises(WorldRefusedError, match=(r"an unlabelled world \(world identity [0-9a-f]{12}, "
                                                  r"changing terrain\.scale, sea\.land, sea\.temperature\)")):
        read_receipt(run_dir)


def test_a_scorecard_reads_an_earth_receipt_unchanged(tmp_path):
    from arwen_global.upper_air_scorecard import read_receipt

    cfg = _cfg(tmp_path, ALL_EARTH)
    receipt = {"name": "e", "config": cfg.config_identity, "config_hash": cfg.config_hash}
    run_dir = _run_dir(tmp_path / "run", receipt)
    assert "world" not in cfg.config_identity
    assert read_receipt(run_dir) == json.loads(json.dumps(receipt))
