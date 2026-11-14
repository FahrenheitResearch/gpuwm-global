"""New Tiedtke's storm-intensity switches: off is WRF, and off keeps every hash.

``ntiedtke_momentum_transport`` (WRF's lmfdudv) and
``ntiedtke_adjustment_time_factor`` (a factor on the deep closure's
adjustment time) are read only under ``cumulus = "ntiedtke"``.  At their
defaults they leave the identity, so a New Tiedtke or Grell-Freitas
checkpoint keeps its hash; set, they are different trajectories and join.
The momentum switch skips exactly the four lmfdudv stages of the walk and
nothing else; the factor reaches the closure kernel as a named argument.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.core import ntiedtke as nt
from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.physics.native_suite import ArwenCudaColumnSuite
from conftest import requires_engine_module, requires_gpu
from test_arwen_global_level5_native import _fake_modules, _options

NATIVE_SMOKE_CONFIG = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")
KERNEL = Path(nt.__file__).parent / "kernels" / "ntiedtke.cu"
FACTORS = ("ntiedtke_adjustment_time_factor", "ntiedtke_entrainment_factor",
           "ntiedtke_downdraft_factor")
SWITCHES = ("ntiedtke_momentum_transport",) + FACTORS


def _nt(**overrides):
    return {**_options(), "cumulus": "ntiedtke", **overrides}


def test_defaults_are_wrf_and_leave_the_identity():
    options = NativePhysicsOptions.from_mapping(_nt())
    assert options.ntiedtke_momentum_transport is True
    for name in FACTORS:
        assert getattr(options, name) == 1.0
    # The build payload carries every switch; the identity carries none.
    for name in SWITCHES:
        assert name in options.normalized
        assert name not in options.identity
    spelled = NativePhysicsOptions.from_mapping(
        _nt(ntiedtke_momentum_transport=True, **{name: 1.0 for name in FACTORS}))
    assert spelled.identity == options.identity


def test_a_set_switch_joins_only_under_new_tiedtke():
    base = NativePhysicsOptions.from_mapping(_nt()).identity
    variants = [NativePhysicsOptions.from_mapping(_nt(ntiedtke_momentum_transport=False)).identity]
    assert variants[0]["ntiedtke_momentum_transport"] is False
    for name in FACTORS:
        ident = NativePhysicsOptions.from_mapping(_nt(**{name: 2.0})).identity
        assert ident[name] == 2.0
        assert all(other not in ident for other in FACTORS if other != name)
        variants.append(ident)
    assert len({repr(sorted(x.items())) for x in [base, *variants]}) == 1 + len(variants)
    for scheme in ("gf", "none"):
        plain = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme}).identity
        flagged = NativePhysicsOptions.from_mapping({
            **_options(), "cumulus": scheme, "ntiedtke_momentum_transport": False,
            **{name: 2.0 for name in FACTORS}}).identity
        assert flagged == plain
        for name in SWITCHES:
            assert name not in plain


@pytest.mark.parametrize("bad", [0, 1, "false", None])
def test_the_momentum_switch_is_a_bool(bad):
    with pytest.raises(ValueError, match="ntiedtke_momentum_transport must be true or false"):
        NativePhysicsOptions.from_mapping(_nt(ntiedtke_momentum_transport=bad))


@pytest.mark.parametrize("name", FACTORS)
@pytest.mark.parametrize("bad", [0.0, 0.05, 25.0, float("nan"), float("inf"), True, "2"])
def test_each_factor_is_a_bounded_number(name, bad):
    with pytest.raises(ValueError, match=f"{name} must be a number"):
        NativePhysicsOptions.from_mapping(_nt(**{name: bad}))


def test_hashes_hold_at_default_and_split_when_set(tmp_path):
    base = open(NATIVE_SMOKE_CONFIG, encoding="utf-8").read()
    marker = "dx_m = 50000.0 }"
    assert marker in base
    gf = replace(load_config(NATIVE_SMOKE_CONFIG), semi_implicit_scheme="external").config_hash
    hashes = {}
    for label, extra in (
        ("nt", ', cumulus = "ntiedtke" }'),
        ("nt_spelled", ', cumulus = "ntiedtke", ntiedtke_momentum_transport = true,'
                       ' ntiedtke_adjustment_time_factor = 1.0 }'),
        ("nocmt", ', cumulus = "ntiedtke", ntiedtke_momentum_transport = false }'),
        ("tau3", ', cumulus = "ntiedtke", ntiedtke_adjustment_time_factor = 3.0 }'),
        ("entr2", ', cumulus = "ntiedtke", ntiedtke_entrainment_factor = 2.0 }'),
        ("dd05", ', cumulus = "ntiedtke", ntiedtke_downdraft_factor = 0.5 }'),
        ("gf_flagged", ', ntiedtke_momentum_transport = false,'
                       ' ntiedtke_adjustment_time_factor = 3.0,'
                       ' ntiedtke_entrainment_factor = 2.0, ntiedtke_downdraft_factor = 0.5 }'),
    ):
        path = tmp_path / f"{label}.toml"
        path.write_text(base.replace(marker, "dx_m = 50000.0" + extra), encoding="utf-8")
        hashes[label] = replace(load_config(path), semi_implicit_scheme="external").config_hash
    assert hashes["nt_spelled"] == hashes["nt"]
    assert hashes["gf_flagged"] == gf
    arms = ("nt", "nocmt", "tau3", "entr2", "dd05")
    assert len({gf, *(hashes[a] for a in arms)}) == 1 + len(arms)


def test_the_suite_hands_every_switch_to_the_scheme():
    set_ = {"ntiedtke_adjustment_time_factor": 2.5, "ntiedtke_entrainment_factor": 1.5,
            "ntiedtke_downdraft_factor": 0.5}
    suite = ArwenCudaColumnSuite(
        _nt(ntiedtke_momentum_transport=False, **set_), array_module=np, modules=_fake_modules())
    cfg = suite._native_cfg(300.0)
    assert cfg.ntiedtke_momentum_transport is False
    for name, value in set_.items():
        assert getattr(cfg, name) == value
    default = ArwenCudaColumnSuite(_nt(), array_module=np, modules=_fake_modules())._native_cfg(300.0)
    assert default.ntiedtke_momentum_transport is True
    for name in FACTORS:
        assert getattr(default, name) == 1.0


class _Walk:
    """Records the assembler's calls in order, launching nothing."""

    def __init__(self, momentum_transport):
        self.momentum_transport = momentum_transport
        self.calls = []

    def zero_run_head(self):
        self.calls.append("zero_run_head")

    def snapshot_forcing(self):
        self.calls.append("snapshot_forcing")

    def snapshot_momentum(self):
        self.calls.append("snapshot_momentum")

    def reduce_llo3(self):
        self.calls.append("reduce_llo3")

    def run_stage(self, stage):
        self.calls.append(stage)


def test_momentum_off_skips_exactly_the_four_lmfdudv_stages():
    on, off = _Walk(True), _Walk(False)
    nt._nt_walk(on)
    nt._nt_walk(off)
    assert [c for c in on.calls if c.startswith("ntiedtke_")] == list(nt.NT_CALL_ORDER)
    assert "snapshot_momentum" in on.calls
    skipped = [c for c in on.calls if c not in off.calls]
    assert set(skipped) == set(nt.NT_MOMENTUM_STAGES) | {"snapshot_momentum"}
    assert [c for c in on.calls if c not in skipped] == off.calls
    # The four are a contiguous tail of cumastrn, after the tendencies and
    # before the driver-level conversion, so nothing downstream of them
    # inside cumastrn is left reading an unwritten array.
    order = list(nt.NT_CALL_ORDER)
    idx = sorted(order.index(s) for s in nt.NT_MOMENTUM_STAGES)
    assert idx == list(range(idx[0], idx[0] + 4))
    assert order[idx[-1] + 1] == "ntiedtke_post_conversion"


def _kernel_params(name):
    text = KERNEL.read_text(encoding="utf-8")
    m = re.search(r'extern "C" __global__ void ' + name + r"\((.*?)\)\s*\{", text, re.S)
    assert m, name
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return tuple(p.split()[-1].lstrip("*") for p in body.split(",") if p.strip())


@pytest.mark.parametrize("stage,factor", [
    ("ntiedtke_closure", "ztau_factor"),
    ("ntiedtke_cutypen", "ztrig_factor"),
    ("ntiedtke_cuascn", "zentr_factor"),
    ("ntiedtke_cudlfsn", "zdd_factor"),
])
def test_each_signature_names_its_factor_where_the_kernel_takes_it(stage, factor):
    assert _kernel_params(stage) == nt.NT_STAGE_SIGNATURE[stage]
    assert factor in nt.NT_STAGE_SIGNATURE[stage]


def test_every_factor_at_one_takes_the_untouched_arm():
    text = KERNEL.read_text(encoding="utf-8")
    assert "if (ztau_factor != 1.0f) ztau = __fmul_rn(ztau, ztau_factor);" in text
    assert text.count("? __fmul_rn(1.75e-3f, entr_factor) : 1.75e-3f;") == 1
    assert text.count("? __fmul_rn(1.75e-3f, zentr_factor) : 1.75e-3f;") == 1
    assert "? __fmul_rn(NT_CMFDEPS, zdd_factor) : NT_CMFDEPS;" in text
    # No bare organised-entrainment constant is left outside the two selects.
    assert text.count("1.75e-3f") == 4


@pytest.mark.gpu
@requires_gpu
@requires_engine_module("gpuwm.verify.harness", "04")
def test_the_real_scheme_default_is_bitwise_and_momentum_off_zeroes_the_pair():
    cp = pytest.importorskip("cupy")
    from gpuwm.verify.harness.convection_closure import measure_component
    from gpuwm.verify.harness.subjects import DeviceHostedSubject
    from test_arwen_global_level5_native import CONFIG
    from test_arwen_global_native_ntiedtke import _recording_ntiedtke

    def battery(**switches):
        recorded = []
        options = _nt(radiation_interval_s=1800.0, land_surface_interval_s=1800.0, **switches)
        suite = DeviceHostedSubject(ArwenCudaColumnSuite(
            options, modules={"arwen_global.core.ntiedtke": _recording_ntiedtke(cp, recorded)}))
        measure_component(CONFIG, physics=suite, steps=2)
        return recorded

    base = battery()
    spelled = battery(ntiedtke_momentum_transport=True, ntiedtke_adjustment_time_factor=1.0,
                      ntiedtke_entrainment_factor=1.0, ntiedtke_downdraft_factor=1.0,
                      ntiedtke_trigger_entrainment_factor=1.0)
    for a, b in zip(base, spelled):
        for name in a:
            assert np.array_equal(a[name], b[name]), name
    off = battery(ntiedtke_momentum_transport=False)
    assert any(np.any(call["rucuten"] != 0.0) for call in base)
    for call in off:
        assert np.all(call["rucuten"] == 0.0) and np.all(call["rvcuten"] == 0.0)
        for name, value in call.items():
            assert np.all(np.isfinite(value)), name
    for switch in ({"ntiedtke_adjustment_time_factor": 3.0},
                   {"ntiedtke_entrainment_factor": 2.0},
                   {"ntiedtke_downdraft_factor": 0.5}):
        moved = battery(**switch)
        for call in moved:
            for name, value in call.items():
                assert np.all(np.isfinite(value)), (switch, name)
        assert any(not np.array_equal(a["rthcuten"], b["rthcuten"])
                   for a, b in zip(base, moved)), switch


def test_the_trigger_factor_follows_unless_set_and_joins_only_when_set():
    options = NativePhysicsOptions.from_mapping(_nt(ntiedtke_entrainment_factor=3.0))
    assert options.ntiedtke_trigger_entrainment_factor is None
    assert "ntiedtke_trigger_entrainment_factor" not in options.normalized
    assert "ntiedtke_trigger_entrainment_factor" not in options.identity
    split = NativePhysicsOptions.from_mapping(
        _nt(ntiedtke_entrainment_factor=3.0, ntiedtke_trigger_entrainment_factor=1.0))
    assert split.identity["ntiedtke_trigger_entrainment_factor"] == 1.0
    assert split.identity != options.identity
    # The normalized options rebuild the same options (None comes back).
    assert NativePhysicsOptions.from_mapping(options.normalized) == options
    assert NativePhysicsOptions.from_mapping(split.normalized) == split
    for scheme in ("gf", "none"):
        plain = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme}).identity
        flagged = NativePhysicsOptions.from_mapping(
            {**_options(), "cumulus": scheme, "ntiedtke_trigger_entrainment_factor": 2.0}).identity
        assert flagged == plain
    for bad in (0.0, 9.0, float("nan"), True, "1"):
        with pytest.raises(ValueError, match="ntiedtke_trigger_entrainment_factor must be unset or a number"):
            NativePhysicsOptions.from_mapping(_nt(ntiedtke_trigger_entrainment_factor=bad))
    cfg = ArwenCudaColumnSuite(
        _nt(ntiedtke_entrainment_factor=3.0, ntiedtke_trigger_entrainment_factor=1.0),
        array_module=np, modules=_fake_modules())._native_cfg(300.0)
    assert cfg.ntiedtke_entrainment_factor == 3.0 and cfg.ntiedtke_trigger_entrainment_factor == 1.0
    assert ArwenCudaColumnSuite(_nt(), array_module=np, modules=_fake_modules())._native_cfg(
        300.0).ntiedtke_trigger_entrainment_factor is None


def test_the_trigger_parcel_reads_its_own_factor_and_the_updraft_its_own():
    assert "ztrig_factor" in nt.NT_STAGE_SIGNATURE["ntiedtke_cutypen"]
    assert "zentr_factor" not in nt.NT_STAGE_SIGNATURE["ntiedtke_cutypen"]
    assert "zentr_factor" in nt.NT_STAGE_SIGNATURE["ntiedtke_cuascn"]
    text = KERNEL.read_text(encoding="utf-8")
    assert text.count("nt_ascent_step(jk, 0, klev, ncol, i, false, ztrig_factor,") == 1
    assert text.count("nt_ascent_step(jk, levels, klev, ncol, i, true, ztrig_factor,") == 1
