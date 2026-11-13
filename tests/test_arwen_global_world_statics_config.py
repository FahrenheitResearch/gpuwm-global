"""load_config carries the world on the statics options (package 2.1's
hook in config.py): Earth leaves them untouched, a world rides on them, an
invented world's folder is resolved against the config file, and a world
that changes the coast still loads on the synthetic planet (its cold start
refuses it, statics.synthetic_provenance)."""
from __future__ import annotations

from pathlib import Path

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.statics import StaticsOptions
from arwen_global.world import EARTH

SMOKE = config_root() / "arwen_global_moist_smoke.toml"
REAL = '\n[statics]\nsource = "real"\nvalid_time = "2026-08-30T18:00:00Z"\n'
EARTH_TABLE = '\n[world]\nlabel = "earth"\n[world.sea]\nlevel_m = 0.0\n'
FLOOD = '\n[world]\nlabel = "sea-plus-70"\n[world.sea]\nlevel_m = 70.0\n'
PANGEA = ('\n[world]\nlabel = "pangea"\n[world.surface]\nfolder = "maps/pangea.world"\n'
          'soil_wetness = 0.3\n[world.sea]\ntemperature = "zonal-mean"\nice = "none"\n')


def _cfg(tmp_path, body: str):
    path = tmp_path / "run.toml"
    path.write_text(SMOKE.read_text(encoding="utf-8") + body, encoding="utf-8")
    return load_config(path)


def test_earth_leaves_the_statics_options_as_they_were(tmp_path):
    for body in (REAL, REAL + EARTH_TABLE):
        cfg = _cfg(tmp_path, body)
        assert cfg.statics.world is EARTH and cfg.statics.world_folder is None
        assert cfg.statics == StaticsOptions(
            source="real", valid_time="2026-08-30T18:00:00Z", declared=True)


def test_a_world_rides_on_the_statics_options(tmp_path):
    cfg = _cfg(tmp_path, REAL + FLOOD)
    assert cfg.statics.world == cfg.world and cfg.world.label == "sea-plus-70"
    assert cfg.statics.world_folder is None
    assert cfg.config_identity["statics"] == _cfg(tmp_path, REAL).config_identity["statics"]
    invented = _cfg(tmp_path, REAL + PANGEA)
    assert Path(invented.statics.world_folder) == tmp_path / "maps" / "pangea.world"


def test_a_coast_world_on_the_synthetic_planet_loads_and_its_cold_start_refuses(tmp_path):
    from arwen_global.statics import synthetic_provenance

    cfg = _cfg(tmp_path, FLOOD)
    assert cfg.statics.source == "synthetic" and cfg.statics.world == cfg.world
    with pytest.raises(ValueError, match="changes the coast"):
        synthetic_provenance(cfg.statics)
