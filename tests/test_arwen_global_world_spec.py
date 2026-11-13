"""The ``[world]`` table and the world object (worlds plan, package 0.1).

The contract every other part of the worlds track codes against: the key
names are frozen here, every range is refused by name, and the identity
rule holds -- a config with no ``[world]`` table and one whose every
setting is Earth's hash the same and step the same bytes, and any other
world moves the hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.world import (
    EARTH,
    EARTH_IDENTITY,
    NOT_SUPPORTED_YET,
    LAUNCHER_RESERVED_KEYS,
    WORLD_KEYS,
    WorldSpec,
    parse_world,
    world_identity_of,
)

#: Every shipped config that is on Earth.  The world templates
#: (``arwen_global_world_*``, package 1.7) carry their own [world] table and
#: are held to it by tests/test_arwen_global_world_templates.py.
SHIPPED = sorted(path for path in config_root().glob("*.toml")
                 if not path.stem.startswith("arwen_global_world_"))
SMOKE = config_root() / "arwen_global_moist_smoke.toml"

#: Every setting written out at Earth's value.
ALL_EARTH = """
[world]
label = "earth"

[world.terrain]
scale = 1.0
air_mass = "conserve"

[world.sun]
obliquity_deg = 23.5
brightness = 1.0
day_length_h = 24.0
orbit = "earth"

[world.sea]
level_m = 0.0
land = "keep"
temperature = "analysis"
ice = "analysis"

[world.air]
co2_ppm = "dated"
ch4_ppb = "dated"
n2o_ppb = "dated"
"""

#: The plan's section 4.1 example, key for key.
PLAN_EXAMPLE = """
[world]
label = "flat-earth"

[world.terrain]
scale = 0.0
air_mass = "conserve"

[world.sun]
obliquity_deg = 23.5
brightness = 1.0
day_length_h = 24.0

[world.sea]
level_m = 0.0
land = "keep"
temperature = "analysis"
ice = "analysis"

[world.air]
co2_ppm = "dated"
"""


def _with(tmp_path: Path, base: Path, extra: str, name: str = "w.toml") -> Path:
    path = tmp_path / name
    path.write_text(base.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    return path


def _world(text: str) -> WorldSpec:
    import tomllib

    return parse_world(tomllib.loads(text).get("world"))


# -- the frozen key names ---------------------------------------------------


def test_the_key_names_are_frozen():
    """Renaming a key breaks every template and preset that writes one."""
    assert {k: set(v) for k, v in WORLD_KEYS.items()} == {
        "": {"label", "terrain", "sun", "sea", "air", "surface"},
        "terrain": {"scale", "air_mass", "cap_m", "box"},
        "terrain.box": {"south_deg", "north_deg", "west_deg", "east_deg", "taper_deg"},
        "sun": {"obliquity_deg", "brightness", "day_length_h", "orbit",
                "eccentricity", "perihelion_deg"},
        "sea": {"level_m", "land", "temperature", "ice"},
        "air": {"co2_ppm", "ch4_ppb", "n2o_ppb"},
        "surface": {"folder", "land_cover", "soil_wetness"},
    }


def test_no_world_key_uses_a_name_a_launcher_sets_by_pattern():
    assert LAUNCHER_RESERVED_KEYS == {
        "name", "duration_s", "output_interval_s", "analysis_grib",
        "start_time_utc", "analysis_mapping", "radiation_column_chunk",
    }
    for keys in WORLD_KEYS.values():
        assert not keys & LAUNCHER_RESERVED_KEYS
    for path in WORLD_KEYS:
        prefix = "[world]" if not path else f"[world.{path}]"
        with pytest.raises(ValueError, match="set by pattern by a launcher"):
            if path == "terrain.box":
                _world("[world]\nlabel='x'\n[world.terrain]\nscale=0.5\n"
                       "[world.terrain.box]\nname = 'x'\n")
            else:
                _world(f"{prefix}\nname = 'x'\n")


def test_the_plan_example_parses_to_a_world_with_no_mountains():
    world = _world(PLAN_EXAMPLE)
    assert world.label == "flat-earth"
    assert world.terrain.scale == 0.0 and world.terrain.air_mass == "conserve"
    assert world.changed_settings() == ["terrain.scale"]
    assert world.changes_terrain and not world.changes_sun and not world.changes_coast


# -- the identity rule ------------------------------------------------------


@pytest.mark.parametrize("config", SHIPPED, ids=lambda p: p.stem)
def test_every_shipped_config_hashes_the_same_with_an_all_earth_world(config, tmp_path):
    extras = (ALL_EARTH, "[world]\n", '[world]\nlabel = "earth"\n',
              # The air-mass rule acts only where the terrain moves.
              '[world]\n[world.terrain]\nair_mass = "fill"\n')
    try:
        bare = load_config(config)
    except ValueError as refused:
        # A shipped example that names an adapter this install does not
        # register refuses alone; it must refuse the same with a world.
        for extra in extras:
            with pytest.raises(ValueError) as again:
                load_config(_with(tmp_path, config, extra))
            assert str(again.value) == str(refused)
        return
    assert bare.world == EARTH
    for extra in extras:
        dressed = load_config(_with(tmp_path, config, extra))
        assert dressed.world.is_earth
        assert dressed.world.identity_hash == EARTH_IDENTITY
        assert dressed.config_identity == bare.config_identity
        assert dressed.config_hash == bare.config_hash


#: One change of each setting; every one must move the hash, and no two
#: may share one.
ONE_CHANGE = {
    "terrain.scale": '[world.terrain]\nscale = 0.0\n',
    "terrain.scale+fill": '[world.terrain]\nscale = 0.0\nair_mass = "fill"\n',
    "terrain.cap_m": '[world.terrain]\ncap_m = 1500.0\n',
    "terrain.box": ('[world.terrain]\nscale = 0.0\n[world.terrain.box]\n'
                    'south_deg = 20.0\nnorth_deg = 50.0\nwest_deg = 60.0\n'
                    'east_deg = 110.0\ntaper_deg = 5.0\n'),
    "sun.obliquity_deg": '[world.sun]\nobliquity_deg = 90.0\n',
    "sun.brightness": '[world.sun]\nbrightness = 1.02\n',
    "sun.day_length_h": '[world.sun]\nday_length_h = 12.0\n',
    "sun.orbit": '[world.sun]\norbit = "perpetual-equinox"\n',
    "sun.eccentric": '[world.sun]\norbit = "eccentric"\neccentricity = 0.1\n',
    "sea.level_m": '[world.sea]\nlevel_m = 70.0\n',
    "sea.land": '[world.terrain]\nscale = 0.0\n[world.sea]\nland = "none"\n',
    "sea.temperature": '[world.sea]\ntemperature = "zonal-mean"\n',
    "sea.temperature.profile": '[world.sea]\ntemperature = "aqua-qobs"\n',
    "sea.ice": '[world.sea]\nice = "none"\n',
    "air.co2_ppm": '[world.air]\nco2_ppm = 560.0\n',
    "air.ch4_ppb": '[world.air]\nch4_ppb = 700.0\n',
    "air.n2o_ppb": '[world.air]\nn2o_ppb = 270.0\n',
    "surface.folder": ('[world.sea]\ntemperature = "zonal-mean"\nice = "none"\n'
                       '[world.surface]\nfolder = "pangea-250ma.world"\n'
                       'soil_wetness = 0.3\n'),
    "surface.soil_wetness": ('[world.sea]\ntemperature = "zonal-mean"\nice = "none"\n'
                             '[world.surface]\nfolder = "pangea-250ma.world"\n'
                             'soil_wetness = 0.6\n'),
    "surface.land_cover": ('[world.sea]\ntemperature = "zonal-mean"\nice = "none"\n'
                           '[world.surface]\nfolder = "pangea-250ma.world"\n'
                           'soil_wetness = 0.3\nland_cover = "barren"\n'),
}


def test_changing_one_setting_moves_the_hash(tmp_path):
    base = load_config(SMOKE).config_hash
    seen = {base: "earth"}
    for key, text in ONE_CHANGE.items():
        cfg = load_config(_with(tmp_path, SMOKE, '[world]\nlabel = "w"\n' + text,
                                name=f"{key}.toml"))
        assert not cfg.world.is_earth, key
        assert cfg.config_identity["world"] == cfg.world.identity, key
        digest = cfg.config_hash
        assert digest not in seen, f"{key} shares a hash with {seen.get(digest)}"
        seen[digest] = key
        assert len(cfg.world.identity_hash) == 64 and len(cfg.world.token) == 12


def test_the_label_is_a_name_and_joins_no_identity(tmp_path):
    a = load_config(_with(tmp_path, SMOKE, '[world]\nlabel = "a"\n[world.sun]\n'
                          'obliquity_deg = 90.0\n', "a.toml"))
    b = load_config(_with(tmp_path, SMOKE, '[world]\nlabel = "b"\n[world.sun]\n'
                          'obliquity_deg = 90.0\n', "b.toml"))
    assert a.world.label != b.world.label
    assert a.config_hash == b.config_hash
    assert a.world.record()["label"] == "a"


def test_an_invented_world_joins_by_its_folder_name_not_its_path():
    text = ('[world]\nlabel = "p"\n[world.sea]\ntemperature = "zonal-mean"\n'
            'ice = "none"\n[world.surface]\nsoil_wetness = 0.3\nfolder = ')
    here = _world(text + '"pangea-250ma.world"\n')
    there = _world(text + '"/data/worlds/pangea-250ma.world/"\n')
    windows = _world(text + "'C:\\worlds\\pangea-250ma.world'\n")
    assert here.identity_hash == there.identity_hash == windows.identity_hash
    assert here.identity["surface"]["folder"] == "pangea-250ma.world"
    assert here.is_invented and here.changes_coast


def test_a_record_written_before_worlds_reads_as_earth():
    assert world_identity_of(None) == EARTH_IDENTITY
    assert world_identity_of({}) == EARTH_IDENTITY
    assert world_identity_of({"config_hash": "abc"}) == EARTH_IDENTITY
    assert world_identity_of(EARTH.record()) == EARTH_IDENTITY
    flat = _world(PLAN_EXAMPLE)
    assert world_identity_of(flat.record()) == flat.identity_hash
    assert world_identity_of({"world": flat.record()}) == flat.identity_hash
    assert world_identity_of(flat.identity_hash) == flat.identity_hash


def test_the_toml_a_world_writes_parses_back_to_it():
    import tomllib

    for key, text in ONE_CHANGE.items():
        world = _world('[world]\nlabel = "w"\n' + text)
        again = parse_world(tomllib.loads(world.to_toml())["world"])
        assert again == world, key
    assert parse_world(tomllib.loads(EARTH.to_toml())["world"]) == EARTH


def test_earth_helpers_are_exact():
    assert EARTH == WorldSpec() and EARTH.is_earth
    assert EARTH.identity is None and EARTH.token == "none"
    assert EARTH.sun.rotation_factor == 1.0
    assert EARTH.changed_settings() == []
    assert _world('[world]\nlabel = "d"\n[world.sun]\nday_length_h = 12.0\n'
                  ).sun.rotation_factor == 2.0


# -- refusals ---------------------------------------------------------------


REFUSED = [
    ('[world]\nlabel = "x"\nplanet = 1\n', "unknown keys in \\[world\\]"),
    ('[world.terrain]\nheight = 1\n', "unknown keys in \\[world.terrain\\]"),
    ('[world]\nlabel = "x"\ngravity_m_s2 = 3.7\n', "gravity is not supported yet"),
    ('[world]\nlabel = "x"\nradius_km = 3390.0\n', "radius is not supported yet"),
    ('[world]\nlabel = "x"\nyear_length_d = 687.0\n', "year length is not supported yet"),
    ('[world]\nlabel = "x"\n[world.air]\ncomposition = "co2"\n',
     "air composition is not supported yet"),
    ('[world]\nlabel = "x"\n[world.air]\nsurface_pressure_pa = 600.0\n',
     "amount of air is not supported yet"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = -0.1\n', "world.terrain.scale must lie in"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = 2.5\n', "world.terrain.scale must lie in"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = true\n', "must be a number"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = nan\n', "must be finite"),
    ('[world]\nlabel = "x"\n[world.terrain]\nair_mass = "keep"\n', "air_mass must be one of"),
    ('[world]\nlabel = "x"\n[world.terrain]\ncap_m = -5.0\n', "cap_m must lie in"),
    ('[world]\nlabel = "x"\n[world.terrain.box]\nsouth_deg = 0.0\nnorth_deg = 10.0\n'
     'west_deg = 0.0\neast_deg = 10.0\n', "this table edits nothing"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = 0.0\n[world.terrain.box]\n'
     'south_deg = 10.0\nnorth_deg = 0.0\nwest_deg = 0.0\neast_deg = 10.0\n',
     "north_deg must be north of south_deg"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = 0.0\n[world.terrain.box]\n'
     'south_deg = 0.0\nnorth_deg = 10.0\nwest_deg = 0.0\n', "east_deg is required"),
    ('[world]\nlabel = "x"\n[world.sun]\nobliquity_deg = 181.0\n', "obliquity_deg must lie in"),
    ('[world]\nlabel = "x"\n[world.sun]\nbrightness = 0.0\n', "brightness must lie in"),
    ('[world]\nlabel = "x"\n[world.sun]\nday_length_h = 6.0\n', "stability bound of the rotation term"),
    ('[world]\nlabel = "x"\n[world.sun]\nday_length_h = 5000.0\n', "slowly rotating"),
    ('[world]\nlabel = "x"\n[world.sun]\norbit = "tidal"\n', "orbit must be one of"),
    ('[world]\nlabel = "x"\n[world.sun]\neccentricity = 0.1\n', "read only with orbit"),
    ('[world]\nlabel = "x"\n[world.sun]\norbit = "eccentric"\n', "eccentricity is required"),
    ('[world]\nlabel = "x"\n[world.sun]\norbit = "eccentric"\neccentricity = 0.5\n',
     "eccentricity must lie in"),
    ('[world]\nlabel = "x"\n[world.sun]\norbit = "perpetual-equinox"\nobliquity_deg = 10.0\n',
     "no meaning under orbit"),
    ('[world]\nlabel = "x"\n[world.sea]\nlevel_m = -120.0\n', "below 0 is not supported yet"),
    ('[world]\nlabel = "x"\n[world.sea]\nlevel_m = 500.0\n', "level_m must lie in"),
    ('[world]\nlabel = "x"\n[world.sea]\nland = "all"\n', "land must be one of"),
    ('[world]\nlabel = "x"\n[world.sea]\ntemperature = "warm"\n', "temperature must be one of"),
    ('[world]\nlabel = "x"\n[world.sea]\nice = "thick"\n', "ice must be one of"),
    ('[world]\nlabel = "x"\n[world.sea]\nland = "none"\n', "a waterworld .* is flat"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = 0.0\n[world.sea]\nland = "none"\n'
     'level_m = 10.0\n', "no coast to move"),
    ('[world]\nlabel = "x"\n[world.air]\nco2_ppm = 0.0\n', "co2_ppm must lie in"),
    ('[world]\nlabel = "x"\n[world.air]\nco2_ppm = "today"\n', "co2_ppm must be 'dated'"),
    ('[world]\nlabel = "x"\n[world.surface]\nsoil_wetness = 0.3\n', "read only with world.surface.folder"),
    ('[world]\nlabel = "x"\n[world.surface]\nfolder = "p.world"\n', "soil_wetness is required"),
    ('[world]\nlabel = "x"\n[world.surface]\nfolder = "p.world"\nsoil_wetness = 0.3\n',
     "sea has no analysis to take"),
    ('[world]\nlabel = "x"\n[world.terrain]\nscale = 0.0\n[world.sea]\ntemperature = '
     '"zonal-mean"\nice = "none"\n[world.surface]\nfolder = "p.world"\nsoil_wetness = 0.3\n',
     "terrain comes from its own map"),
    ('[world]\nlabel = "x"\n[world.sea]\nlevel_m = 5.0\ntemperature = "zonal-mean"\n'
     'ice = "none"\n[world.surface]\nfolder = "p.world"\nsoil_wetness = 0.3\n',
     "coast comes from its own map"),
    ('[world]\nlabel = "x"\n[world.sea]\ntemperature = "zonal-mean"\nice = "none"\n'
     '[world.surface]\nfolder = "p.world"\nsoil_wetness = 1.5\n', "soil_wetness must lie in"),
    ('[world.terrain]\nscale = 0.0\n', "world.label is required"),
    ('[world]\nlabel = "earth"\n[world.terrain]\nscale = 0.0\n', "kept for Earth"),
    ('[world]\nlabel = "Flat Earth"\n', "world.label must be"),
    ('[world]\nlabel = ""\n', "world.label must be"),
    ('world = 3\n', "must be a TOML table"),
    ('[world]\nterrain = 3\n', "must be a TOML table"),
]


@pytest.mark.parametrize("text,message", REFUSED)
def test_each_refusal_names_its_key(text, message):
    with pytest.raises(ValueError, match=message):
        _world(text)


def test_every_not_supported_key_is_refused_with_its_reason():
    for dotted, reason in NOT_SUPPORTED_YET.items():
        if "." in dotted:
            part, key = dotted.split(".")
            text = f'[world]\nlabel = "x"\n[world.{part}]\n{key} = 1.0\n'
        else:
            text = f'[world]\nlabel = "x"\n{dotted} = 1.0\n'
        with pytest.raises(ValueError) as caught:
            _world(text)
        assert reason in str(caught.value)


def test_a_world_and_the_physics_option_may_not_both_set_co2(tmp_path):
    native = config_root() / "arwen_global_level5_native_smoke.toml"
    text = native.read_text(encoding="utf-8")
    marker = "native_adapter_options = { "
    assert text.count(marker) == 1
    both = tmp_path / "both.toml"
    both.write_text(
        text.replace(marker, marker + "trace_co2_ppm = 400.0, ")
        + '\n[world]\nlabel = "c"\n[world.air]\nco2_ppm = 560.0\n',
        encoding="utf-8")
    with pytest.raises(ValueError, match="name it in one place"):
        load_config(both)
    # Either one alone is a config.
    assert load_config(_with(tmp_path, native, '[world]\nlabel = "c"\n[world.air]\n'
                             'co2_ppm = 560.0\n')).world.air.co2_ppm == 560.0


# -- ten steps, world off ---------------------------------------------------


def _ten_step_arrays(tmp_path: Path, extra: str, tag: str):
    from arwen_global.runner import run

    text = SMOKE.read_text(encoding="utf-8")
    text = text.replace("duration_s = 40.0", "duration_s = 100.0")
    text = text.replace("output_interval_s = 20.0", "output_interval_s = 100.0")
    path = tmp_path / f"ten-{tag}.toml"
    path.write_text(text + "\n" + extra, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.duration_s == 10 * cfg.dt_s
    outdir = tmp_path / f"run-{tag}"
    result = run(cfg, outdir)
    assert result["status"] == "pass"
    final = outdir / "arwen_global_step00000010.npz"
    return cfg, read_checkpoint(final), final.read_bytes()


def test_ten_steps_with_an_all_earth_world_are_the_bytes_of_ten_without(tmp_path):
    cfg_a, (meta_a, arrays_a), raw_a = _ten_step_arrays(tmp_path, "", "bare")
    cfg_b, (meta_b, arrays_b), raw_b = _ten_step_arrays(tmp_path, ALL_EARTH, "earth")
    assert cfg_a.config_hash == cfg_b.config_hash
    assert meta_a == meta_b
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert arrays_a[name].dtype == arrays_b[name].dtype, name
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    assert hashlib.sha256(raw_a).hexdigest() == hashlib.sha256(raw_b).hexdigest()
