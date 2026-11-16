"""The opt-in deep-arm switches of Grell-Freitas: gf_deep_physics,
gf_deep_experiment and gf_momentum_transport.

CPU rows only: each option's default (the 0.2.5 default D: 7, 34, true), its validation and its identity rule
(0.2.4's values, 0 and false, join neither the identity nor the config hash, so a
config that states them keeps its 0.2.4 hash); the define the seam
compiles with; the kernel guards that keep the default source WRF v4.6.1's;
and the runtime handing the option to the seam.  No kernel is compiled.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.core import gf

SMOKE = Path(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))


def _options():
    return {
        "acknowledgement": "device-pending-arwen-native-physics-v1",
        "start_time_utc": "2024-05-21T00:00:00Z",
        "radiation": "rrtmgp", "radiation_interval_s": 10.0,
        "radiation_column_chunk": 512, "radiation_validation_mode": "fused",
        "surface_layer": "sfclay", "sfclay_option": 1,
        "land_surface": "noah", "land_surface_interval_s": 10.0,
        "pbl": "ysu", "ysu_topdown_pblmix": 1,
        "microphysics": "morrison", "morr_rimed_ice": 1, "dx_m": 50_000.0,
    }


def _smoke_with(tmp_path, name, setting):
    text = SMOKE.read_text(encoding="utf-8")
    assert "native_adapter_options = { " in text
    path = tmp_path / f"{name}.toml"
    path.write_text(text.replace("native_adapter_options = { ",
                                 f"native_adapter_options = {{ {setting}, "),
                    encoding="utf-8")
    return path


def _gf_source() -> str:
    return (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")


# --- gf_deep_physics ------------------------------------------------------

def test_deep_physics_defaults_to_34_and_0_the_0_2_4_arithmetic_leaves_the_identity():
    from arwen_global.physics.native_options import NativePhysicsOptions

    default = NativePhysicsOptions.from_mapping(_options())
    assert default.gf_deep_physics == 34
    assert default.identity["gf_deep_physics"] == 34
    spelt = NativePhysicsOptions.from_mapping({**_options(), "gf_deep_physics": 34})
    assert spelt.identity == default.identity
    wrf = NativePhysicsOptions.from_mapping({**_options(), "gf_deep_physics": 0})
    assert "gf_deep_physics" not in wrf.identity
    assert {k: v for k, v in default.identity.items() if k != "gf_deep_physics"} == wrf.identity
    for scheme in ("ntiedtke", "own", "none"):
        other = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme})
        assert "gf_deep_physics" not in other.identity, scheme


@pytest.mark.parametrize("good", [0, 1, 2, 4, 8, 15, 32, 34, 35, 47])
def test_deep_physics_accepts_every_set_of_the_bits_the_kernel_has(good):
    from arwen_global.physics.native_options import NativePhysicsOptions

    options = NativePhysicsOptions.from_mapping({**_options(), "gf_deep_physics": good})
    assert options.gf_deep_physics == good


@pytest.mark.parametrize("bad", [16, 17, 48, 64, 255, -1, -34, True, 1.5, "34"])
def test_deep_physics_refuses_bit_16_bits_above_32_and_non_integers(bad):
    from arwen_global.physics.native_options import NativePhysicsOptions

    with pytest.raises(ValueError, match="gf_deep_physics must be an integer bit set"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_deep_physics": bad})


def test_deep_physics_34_hashes_as_the_default_and_0_moves_the_hash(tmp_path):
    from arwen_global.config import load_config

    default = load_config(SMOKE)
    zero = load_config(_smoke_with(tmp_path, "zero", "gf_deep_physics = 0"))
    on = load_config(_smoke_with(tmp_path, "on", "gf_deep_physics = 34"))
    assert on.config_hash == default.config_hash
    assert "gf_deep_physics" not in zero.config_identity["native_adapter_options"]
    assert default.config_identity["native_adapter_options"]["gf_deep_physics"] == 34
    assert zero.config_hash != default.config_hash
    assert zero.native_adapter_options["gf_deep_physics"] == 0


def test_deep_physics_compiles_its_define_only_when_set():
    assert gf._gf_defines(40) == ()
    assert gf._gf_defines(40, deep_physics=0) == ()
    assert gf._gf_defines(40, deep_physics=34) == (("GF_DEEP_PHYS", 34),)
    assert gf._gf_defines(40, deep_physics=2) == (("GF_DEEP_PHYS", 2),)
    assert gf._gf_defines(40, deep_experiment=2, deep_physics=34) == (
        ("GF_DEEP_EXPERIMENT", 2), ("GF_DEEP_PHYS", 34))
    assert gf.GrellFreitas().deep_physics == 0
    assert gf.GrellFreitas(deep_physics=34).deep_physics == 34
    from arwen_global.core.kernels import module_source_int_defines

    assert "#define GF_DEEP_PHYS 34\n" in module_source_int_defines("gf", (("GF_DEEP_PHYS", 34),))
    from arwen_global.physics import native_runtime

    assert 'extra["deep_physics"] = int(self.options.gf_deep_physics)' in Path(
        native_runtime.__file__).read_text(encoding="utf-8")


def test_deep_physics_bits_sit_behind_guards_that_default_to_off():
    source = _gf_source()
    assert "#ifndef GF_DEEP_PHYS\n#define GF_DEEP_PHYS 0\n#endif" in source
    for bit in (1, 2, 4, 8, 32):
        assert source.count(f"#if (GF_DEEP_PHYS & {bit})") == 1, bit
    # bit 16 is refused by validation because the kernel has no such block
    assert "GF_DEEP_PHYS & 16" not in source
    # bit 32: the moisture-convergence members without WRF's divisor, and the
    # WRF spelling still the #else branch
    block = source[source.index("#if (GF_DEEP_PHYS & 32)"):]
    block = block[:block.index("#endif")]
    assert "const float v = mconv;" in block
    assert "*mconv_den_out = K_ONE;" in block
    assert "const float v = FDIV(mconv, den);" in block.split("#else", 1)[1]
    # bit 2: the cloud-depth beta of CCPP get_zu_zd_pdf_fim
    block = source[source.index("#if (GF_DEEP_PHYS & 2)"):]
    assert "beta = FADD(1.3f, FSUB(K_ONE, FDIV(FSUB(p[kb_adj], p[kt]), 1200.0f)));" in block[:block.index("#endif")]
    # bit 1: c1d = clwdet * up_massdetr(k-1), clwdet = 0.1
    block = source[source.index("#if (GF_DEEP_PHYS & 1)"):]
    assert "c1d[k] = FMUL(0.1f, upmd[k - 1]);" in block[:block.index("#endif")]


# --- gf_deep_experiment and gf_momentum_transport (0.2.4, untested until now)

def test_deep_experiment_and_momentum_transport_default_to_d_and_are_validated():
    from arwen_global.physics.native_options import NativePhysicsOptions

    default = NativePhysicsOptions.from_mapping(_options())
    assert default.gf_deep_experiment == 7 and default.gf_momentum_transport is True
    assert default.identity["gf_deep_experiment"] == 7
    assert default.identity["gf_momentum_transport"] is True
    wrf = NativePhysicsOptions.from_mapping(
        {**_options(), "gf_deep_experiment": 0, "gf_momentum_transport": False})
    assert "gf_deep_experiment" not in wrf.identity
    assert "gf_momentum_transport" not in wrf.identity
    for value in (2, 3, 4, 5, 6, 7, 8):
        on = NativePhysicsOptions.from_mapping({**_options(), "gf_deep_experiment": value})
        assert on.identity["gf_deep_experiment"] == value
        assert {k: v for k, v in on.identity.items() if k != "gf_deep_experiment"} == {
            k: v for k, v in default.identity.items() if k != "gf_deep_experiment"}
    for bad in (1, 9, -1, True):
        with pytest.raises(ValueError, match="gf_deep_experiment must be 0 or 2 to 8"):
            NativePhysicsOptions.from_mapping({**_options(), "gf_deep_experiment": bad})
    with pytest.raises(ValueError, match="gf_momentum_transport must be true or false"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_momentum_transport": 1})
    for scheme in ("ntiedtke", "own", "none"):
        other = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme})
        assert "gf_deep_experiment" not in other.identity, scheme
        assert "gf_momentum_transport" not in other.identity, scheme


def test_deep_experiment_compiles_its_define_only_when_set():
    assert gf._gf_defines(40, deep_experiment=0) == ()
    assert gf._gf_defines(40, deep_experiment=2) == (("GF_DEEP_EXPERIMENT", 2),)
    assert gf._gf_defines(40, deep_experiment=5) == (("GF_DEEP_EXPERIMENT", 5),)
    assert gf.GrellFreitas(deep_experiment=3).deep_experiment == 3
    # 5: the GEOS GF2020 peak (Freitas et al. 2021), its own guarded block
    assert "GF_DEEP_EXPERIMENT == 5" in _gf_source()
    # 6: lev_start at the cap of WRF's own min(0.9, 0.4 + 0.013 csum)
    assert gf._gf_defines(40, deep_experiment=6) == (("GF_DEEP_EXPERIMENT", 6),)
    assert "#elif GF_DEEP_EXPERIMENT == 6\n        lev_start = K_P9;" in _gf_source()
    assert "#ifndef GF_DEEP_EXPERIMENT\n#define GF_DEEP_EXPERIMENT 0\n#endif" in _gf_source()
    from arwen_global.physics import native_runtime

    runtime = Path(native_runtime.__file__).read_text(encoding="utf-8")
    assert 'extra["deep_experiment"] = int(self.options.gf_deep_experiment)' in runtime
    assert 'extra["momentum_transport"] = True' in runtime
