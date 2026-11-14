"""The shallow arm's cloud-base buoyancy integral (gf_shallow_buoyancy_from_cloud_base).

CPU rows: the option's default, its identity rule and its validation; the
define the seam compiles with; the kernel guard sitting on the one sum it
changes; and the runtime handing the switch to the seam.  The device
evidence is the shallow probe (tools/arwen_global_shallow_probe.py) and the
10-day T255 twins in docs/trade-shallow.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arwen_global.core import gf


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


def test_the_option_is_off_by_default_and_joins_the_identity_only_when_on():
    from arwen_global.physics.native_options import NativePhysicsOptions

    wrf = NativePhysicsOptions.from_mapping(_options())
    assert wrf.gf_shallow_buoyancy_from_cloud_base is False
    assert "gf_shallow_buoyancy_from_cloud_base" not in wrf.identity
    on = NativePhysicsOptions.from_mapping({**_options(), "gf_shallow_buoyancy_from_cloud_base": True})
    assert on.identity["gf_shallow_buoyancy_from_cloud_base"] is True
    assert {k: v for k, v in on.identity.items()
            if k != "gf_shallow_buoyancy_from_cloud_base"} == wrf.identity
    for scheme in ("ntiedtke", "own", "none"):
        other = NativePhysicsOptions.from_mapping(
            {**_options(), "cumulus": scheme, "gf_shallow_buoyancy_from_cloud_base": True})
        assert "gf_shallow_buoyancy_from_cloud_base" not in other.identity, scheme
    with pytest.raises(ValueError, match="gf_shallow_buoyancy_from_cloud_base must be true or false"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_shallow_buoyancy_from_cloud_base": 1})


def test_the_seam_compiles_the_define_only_when_asked():
    assert gf._gf_defines(40) == ()
    assert gf._gf_defines(40, shallow_buoyancy_from_cloud_base=True) == (
        ("GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE", 1),)
    assert gf._gf_defines(40, resolved_convergence_closure=True, shallow_buoyancy_from_cloud_base=True) == (
        ("GF_RESOLVED_CONVERGENCE_CLOSURE", 1), ("GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE", 1))
    assert gf.GrellFreitas().shallow_buoyancy_from_cloud_base is False
    assert gf.GrellFreitas(shallow_buoyancy_from_cloud_base=True).shallow_buoyancy_from_cloud_base is True
    from arwen_global.core.kernels import module_source_int_defines

    assert "#define GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE 1\n" in module_source_int_defines(
        "gf", (("GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE", 1),))


def test_the_guard_sits_on_the_shallow_sum_alone():
    source = (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")
    assert source.count("#if GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE") == 1
    block = source[source.index("#if GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE"):]
    block = block[:block.index("ki = gfd_maxloc(dbyt, 1, nz);")]
    assert "if (k >= kbcon)" in block
    assert "dbyt[k] = FADD(dbyt[k - 1], FMUL(dbyo[k], dz));" in block
    # the guarded sum is the shallow column's (gfd_shallow_column), not the deep arm's
    shallow = source.index("__device__ void gfd_shallow_column(")
    assert source.index("#if GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE") > shallow
    assert "#define GF_SHALLOW_BUOYANCY_FROM_CLOUD_BASE 0" in source


def test_the_native_runtime_hands_the_seam_the_switch():
    from arwen_global.physics import native_runtime

    source = Path(native_runtime.__file__).read_text(encoding="utf-8")
    assert 'extra["shallow_buoyancy_from_cloud_base"] = True' in source
    assert "self.options.gf_shallow_buoyancy_from_cloud_base" in source


def test_the_blqe_option_is_off_by_default_and_moves_only_the_closure_sum_bound():
    from arwen_global.physics.native_options import NativePhysicsOptions

    wrf = NativePhysicsOptions.from_mapping(_options())
    assert wrf.gf_shallow_blqe_to_cloud_base is False
    assert "gf_shallow_blqe_to_cloud_base" not in wrf.identity
    on = NativePhysicsOptions.from_mapping({**_options(), "gf_shallow_blqe_to_cloud_base": True})
    assert on.identity["gf_shallow_blqe_to_cloud_base"] is True
    with pytest.raises(ValueError, match="gf_shallow_blqe_to_cloud_base must be true or false"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_shallow_blqe_to_cloud_base": 1})
    assert gf._gf_defines(40, shallow_blqe_to_cloud_base=True) == (("GF_SHALLOW_BLQE_TO_CLOUD_BASE", 1),)
    source = (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")
    assert source.count("#if GF_SHALLOW_BLQE_TO_CLOUD_BASE") == 1
    block = source[source.index("#if GF_SHALLOW_BLQE_TO_CLOUD_BASE"):]
    assert block.startswith("#if GF_SHALLOW_BLQE_TO_CLOUD_BASE\n        const int kblqe = kbcon;\n#else\n"
                            "        const int kblqe = kpbl;\n#endif\n        for (int k = 1; k <= kblqe; k++) {")
    from arwen_global.physics import native_runtime

    assert 'extra["shallow_blqe_to_cloud_base"] = True' in Path(
        native_runtime.__file__).read_text(encoding="utf-8")


def test_the_depth_minimum_is_off_at_zero_and_reuses_the_deep_arms_ierr_6_test():
    from arwen_global.physics.native_options import NativePhysicsOptions

    wrf = NativePhysicsOptions.from_mapping(_options())
    assert wrf.gf_shallow_depth_min_m == 0
    assert "gf_shallow_depth_min_m" not in wrf.identity
    on = NativePhysicsOptions.from_mapping({**_options(), "gf_shallow_depth_min_m": 500})
    assert on.identity["gf_shallow_depth_min_m"] == 500
    for bad in (-1, 1.5, True, 6000):
        with pytest.raises(ValueError, match="gf_shallow_depth_min_m must be a whole number"):
            NativePhysicsOptions.from_mapping({**_options(), "gf_shallow_depth_min_m": bad})
    assert gf._gf_defines(40, shallow_depth_min_m=500) == (("GF_SHALLOW_DEPTH_MIN_M", 500),)
    assert gf._gf_defines(40, shallow_depth_min_m=0) == ()
    with pytest.raises(ValueError, match="not negative"):
        gf.GrellFreitas(shallow_depth_min_m=-5)
    source = (Path(gf.__file__).parent / "kernels" / "gf.cu").read_text(encoding="utf-8")
    assert source.count("#if GF_SHALLOW_DEPTH_MIN_M") == 1
    block = source[source.index("#if GF_SHALLOW_DEPTH_MIN_M"):]
    block = block[:block.index("#endif")]
    assert "FADD(-zo_cup[kbcon], zo_cup[ktop]) < (float)GF_SHALLOW_DEPTH_MIN_M" in block
    assert "ierr = 6;" in block and "if (ierr == 0" in block
    # the deep arm's own test, which this one copies
    assert "if (FADD(-zo_cup[kbcon], zo_cup[ktop]) < depth_min) ierr = 6;" in source
    from arwen_global.physics import native_runtime

    assert 'extra["shallow_depth_min_m"] = self.options.gf_shallow_depth_min_m' in Path(
        native_runtime.__file__).read_text(encoding="utf-8")
