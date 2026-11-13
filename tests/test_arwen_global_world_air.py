"""A world's air (worlds plan, package 3.3): methane and nitrous oxide
options beside CO2, and the [world.air] table written onto them.

What these tests hold, on the CPU:

* the two new options (``trace_ch4_ppb``, ``trace_n2o_ppb``) left at None
  are absent from the normalized options and from the identity, so every
  config and checkpoint written before they existed keeps its bytes;
* a number overrides its gas alone in what RRTMGP is handed and in the
  receipt's record, joins the identity, and is refused by name outside
  (0, 1e5) ppb;
* ``world_air_options`` leaves an Earth run's options exactly as they are,
  writes a world's numbers onto the three options, keeps a "dated" gas on
  the options' own value, and refuses a gas named in both places;
* a changed air under a suite that radiates no trace gas is refused.

The forcing check against the published RFMIP reference fluxes needs the
card: ``tests/test_arwen_global_world_air_device.py``.
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.physics.native_options import (
    GAS_OPTIONS,
    WORLD_AIR_OPTIONS,
    NativePhysicsOptions,
    world_air_options,
    world_air_physics_refusal,
)
from arwen_global.world import EARTH, parse_world

NATIVE = config_root() / "arwen_global_level5_native_smoke.toml"
ACK = "device-pending-arwen-native-physics-v1"


def _options(**extra) -> NativePhysicsOptions:
    return NativePhysicsOptions.from_mapping({
        "acknowledgement": ACK,
        "start_time_utc": "2026-09-01T00:00:00Z",
        **extra,
    })


def _world(text: str):
    return parse_world(tomllib.loads(text).get("world"))


def _with(tmp_path: Path, extra: str, name: str = "w.toml") -> Path:
    path = tmp_path / name
    path.write_text(NATIVE.read_text(encoding="utf-8") + "\n" + extra,
                    encoding="utf-8")
    return path


# -- the options, world off -------------------------------------------------


def test_unset_methane_and_nitrous_oxide_leave_no_trace():
    options = _options()
    for name in ("trace_ch4_ppb", "trace_n2o_ppb"):
        assert getattr(options, name) is None
        assert name not in options.normalized
        assert name not in options.identity
    # CO2 carried its None into every hash since it was an option; it stays.
    assert options.identity["trace_co2_ppm"] is None
    assert options.normalized["trace_co2_ppm"] is None
    # The normalized options still rebuild the same options.
    assert NativePhysicsOptions.from_mapping(options.normalized) == options


def test_the_gas_table_names_each_option_once():
    assert {name for name, _ in GAS_OPTIONS.values()} == set(WORLD_AIR_OPTIONS.values())
    assert set(GAS_OPTIONS) == {"co2", "ch4", "n2o"}
    for name in WORLD_AIR_OPTIONS.values():
        assert name in NativePhysicsOptions.__dataclass_fields__


def test_methane_alone_overrides_methane():
    gases = _options(trace_ch4_ppb=808.249).greenhouse_gas_overrides()
    assert math.isclose(gases["ch4"], 808.249e-9, rel_tol=1e-15)
    # The others stay on the dated table (2026 reads 2025).
    assert math.isclose(gases["co2"], 425.64e-6, rel_tol=1e-12)
    assert math.isclose(gases["n2o"], 338.85e-9, rel_tol=1e-12)


def test_nitrous_oxide_alone_overrides_nitrous_oxide():
    gases = _options(trace_n2o_ppb=273.021).greenhouse_gas_overrides()
    assert math.isclose(gases["n2o"], 273.021e-9, rel_tol=1e-15)
    assert math.isclose(gases["ch4"], 1935.94e-9, rel_tol=1e-12)
    assert math.isclose(gases["co2"], 425.64e-6, rel_tol=1e-12)


def test_all_three_together_and_the_record_names_them():
    options = _options(trace_co2_ppm=284.317, trace_ch4_ppb=808.249,
                       trace_n2o_ppb=273.021)
    gases = options.greenhouse_gas_overrides()
    assert math.isclose(gases["co2"], 284.317e-6, rel_tol=1e-15)
    assert math.isclose(gases["ch4"], 808.249e-9, rel_tol=1e-15)
    assert math.isclose(gases["n2o"], 273.021e-9, rel_tol=1e-15)
    record = options.greenhouse_gas_record()
    assert record["overridden"] == ["ch4", "co2", "n2o"]
    assert record["mole_fraction"] == gases
    # The year the table would have been read at is still recorded.
    assert record["table_year"] == {"co2": 2025, "ch4": 2025, "n2o": 2025}


def test_a_set_gas_joins_the_identity_and_moves_it():
    base = _options().identity
    ch4 = _options(trace_ch4_ppb=808.249).identity
    n2o = _options(trace_n2o_ppb=273.021).identity
    assert ch4["trace_ch4_ppb"] == 808.249 and "trace_n2o_ppb" not in ch4
    assert n2o["trace_n2o_ppb"] == 273.021 and "trace_ch4_ppb" not in n2o
    assert ch4 != base and n2o != base and ch4 != n2o
    assert _options(trace_ch4_ppb=808.249).normalized["trace_ch4_ppb"] == 808.249


@pytest.mark.parametrize("name", ["trace_ch4_ppb", "trace_n2o_ppb"])
@pytest.mark.parametrize("bad", [0.0, -1.0, 1.0e5, 2.0e5, float("nan"),
                                 float("inf"), True, "800"])
def test_out_of_range_methane_and_nitrous_oxide_are_refused_by_name(name, bad):
    with pytest.raises(ValueError, match=name):
        _options(**{name: bad})


def test_the_bounds_are_open_and_reach_the_world_tables():
    from arwen_global.world.spec import TRACE_RANGE_PPB

    assert TRACE_RANGE_PPB == (0.0, 1.0e5)
    assert _options(trace_ch4_ppb=1.0e-3).trace_ch4_ppb == 1.0e-3
    assert _options(trace_n2o_ppb=99_999.0).trace_n2o_ppb == 99_999.0
    # Inside RRTMGP's own open bound of 1e-2 mole fraction.
    assert 99_999.0e-9 < 1.0e-2


# -- the world's air onto the options ---------------------------------------


def test_earth_and_an_earth_air_leave_the_options_exactly_as_they_are():
    raw = {"acknowledgement": ACK, "trace_co2_ppm": 400.0}
    for world in (None, EARTH, _world('[world]\nlabel = "e"\n[world.air]\n'
                                      'co2_ppm = "dated"\n'),
                  _world('[world]\nlabel = "tilt"\n[world.sun]\n'
                         'obliquity_deg = 90.0\n')):
        out = world_air_options(raw, world)
        assert out == raw and out is not raw


def test_a_world_writes_its_numbers_onto_the_three_options():
    world = _world('[world]\nlabel = "air"\n[world.air]\nco2_ppm = 1137.27\n'
                   'ch4_ppb = 808.249\nn2o_ppb = 273.021\n')
    out = world_air_options({"acknowledgement": ACK}, world)
    assert out["trace_co2_ppm"] == 1137.27
    assert out["trace_ch4_ppb"] == 808.249
    assert out["trace_n2o_ppb"] == 273.021
    gases = NativePhysicsOptions.from_mapping(
        {**out, "start_time_utc": "2026-11-12T00:00:00Z"}).greenhouse_gas_overrides()
    assert math.isclose(gases["co2"], 1137.27e-6, rel_tol=1e-15)
    assert math.isclose(gases["ch4"], 808.249e-9, rel_tol=1e-15)
    assert math.isclose(gases["n2o"], 273.021e-9, rel_tol=1e-15)


def test_a_dated_gas_keeps_the_options_own_value():
    world = _world('[world]\nlabel = "m"\n[world.air]\nch4_ppb = 3600.0\n')
    out = world_air_options({"acknowledgement": ACK, "trace_co2_ppm": 400.0,
                             "trace_n2o_ppb": 300.0}, world)
    assert out["trace_ch4_ppb"] == 3600.0
    assert out["trace_co2_ppm"] == 400.0
    assert out["trace_n2o_ppb"] == 300.0


@pytest.mark.parametrize("key,name,label", [
    ("co2_ppm", "trace_co2_ppm", "CO2"),
    ("ch4_ppb", "trace_ch4_ppb", "CH4"),
    ("n2o_ppb", "trace_n2o_ppb", "N2O"),
])
def test_a_gas_named_in_both_places_is_refused(key, name, label):
    world = _world(f'[world]\nlabel = "x"\n[world.air]\n{key} = 500.0\n')
    with pytest.raises(ValueError, match="name it in one place") as caught:
        world_air_options({"acknowledgement": ACK, name: 400.0}, world)
    assert f"world.air.{key}" in str(caught.value)
    assert label in str(caught.value)


def test_a_changed_air_is_refused_where_nothing_radiates_a_trace_gas():
    world = _world('[world]\nlabel = "no-methane"\n[world.air]\nch4_ppb = 1.0\n')
    for mode in ("reference", "none"):
        reason = world_air_physics_refusal(world, mode)
        assert reason is not None
        assert "no-methane" in reason and "world.air.ch4_ppb" in reason
        assert "arwen-native" in reason
    assert world_air_physics_refusal(world, "arwen-native") is None
    assert world_air_physics_refusal(EARTH, "reference") is None
    assert world_air_physics_refusal(None, "none") is None
    tilt = _world('[world]\nlabel = "tilt"\n[world.sun]\nobliquity_deg = 60.0\n')
    assert world_air_physics_refusal(tilt, "reference") is None


# -- through the config reader ----------------------------------------------


def test_a_world_air_moves_the_hash_once_and_never_the_stored_options(tmp_path):
    bare = load_config(NATIVE)
    methane = load_config(_with(tmp_path, '[world]\nlabel = "m"\n[world.air]\n'
                                          'ch4_ppb = 808.249\n'))
    # The stored options are the config's, untouched by the world ...
    assert methane.native_adapter_options == bare.native_adapter_options
    assert "trace_ch4_ppb" not in bare.native_adapter_options
    # ... the hash moves through the world's identity ...
    assert methane.config_hash != bare.config_hash
    assert "world" in methane.config_identity and "world" not in bare.config_identity
    assert (methane.config_identity["native_adapter_options"]
            == bare.config_identity["native_adapter_options"])
    # ... and the physics is built with the world's methane.
    built = world_air_options(methane.native_adapter_options, methane.world)
    assert built["trace_ch4_ppb"] == 808.249
    gases = NativePhysicsOptions.from_mapping(built).greenhouse_gas_overrides()
    assert math.isclose(gases["ch4"], 808.249e-9, rel_tol=1e-15)
    # Earth builds what it always built.
    assert world_air_options(bare.native_adapter_options, bare.world) == dict(
        bare.native_adapter_options)


def test_an_option_set_in_the_config_joins_the_native_identity(tmp_path):
    text = NATIVE.read_text(encoding="utf-8")
    marker = "native_adapter_options = { "
    assert text.count(marker) == 1
    path = tmp_path / "n2o.toml"
    path.write_text(text.replace(marker, marker + "trace_n2o_ppb = 273.021, "),
                    encoding="utf-8")
    cfg = load_config(path)
    bare = load_config(NATIVE)
    assert cfg.native_adapter_options["trace_n2o_ppb"] == 273.021
    assert cfg.config_identity["native_adapter_options"]["trace_n2o_ppb"] == 273.021
    assert cfg.config_hash != bare.config_hash


def test_every_shipped_native_config_keeps_its_stored_options():
    """No shipped config sets the new options, so none of their stored
    options or identities gains a key (the hash gate itself is the
    cross-tree run against rel/0.1.6)."""
    for path in sorted(config_root().glob("*.toml")):
        try:
            cfg = load_config(path)
        except Exception:  # noqa: BLE001 - some shipped configs refuse on their own
            continue
        if cfg.physics_mode != "arwen-native":
            continue
        assert "trace_ch4_ppb" not in cfg.native_adapter_options, path.name
        assert "trace_n2o_ppb" not in cfg.native_adapter_options, path.name
        identity = cfg.config_identity["native_adapter_options"]
        assert "trace_ch4_ppb" not in identity and "trace_n2o_ppb" not in identity
