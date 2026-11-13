"""The orographic gravity-wave drag (WRF gwd_opt = 1) against WRF's own words.

WHAT IS HELD HERE.  The global model had no gravity-wave drag of any kind
before ``orographic_drag = "kim-gwdo"``.  The scheme is the KIM orographic
drag with flow blocking, carried from the engine (``core/terrain_drag.py``
and ``core/kernels/terrain_drag.cu``), and three things are graded:

* the known answer: WRF v4.6.1's ``SUBROUTINE gwdo`` itself, byte-unmodified,
  run by gfortran on 149 columns in 12 regimes on two vertical grids at six
  grid lengths (the engine's oracle, copied with its provenance into
  ``tests/data/oracles/gwdo_kim``).  The package's float32 numpy
  transcription of the kernel lands on every one of WRF's output words
  without a card; on a card, under the engine's strict numerics, the
  carried kernel does too, and in either numerics mode it is the engine's
  own kernel word for word;
* the seam: the native suite adds the drag to YSU's momentum rates from the
  column state YSU read, where WRF adds it, and only with the option on;
* the bookkeeping: the option is off by default and joins the identity only
  when on, the statistics' identity travels in the physics state so a
  restart that would read other statistics is refused, and a run with the
  option on and no statistics is refused by name.

The sub-grid orography statistics are graded in
``tests/test_arwen_global_orographic_statics.py``.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from arwen_global.constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.physics.orographic_drag import (
    METADATA_KEY, OrographicStatistics, SSO_FIELDS, apply_orographic_drag,
    reference_launch_gwdo)

FIXTURE = Path(__file__).resolve().parent / "data" / "oracles" / "gwdo_kim"
GRIDS = ("L50", "L90")
#: The engine oracle's own record (its PROVENANCE.md), copied unchanged.
FIXTURE_SHA256 = {
    "gwdo-kim-inputs-L50.npz": "a30bba9ae5737bd0e4f76e405bac314dc3fa5726dc8a306473574544eba517f4",
    "gwdo-kim-inputs-L90.npz": "7599931644f46b9e8f5b52eed99f3486773692d7335a1fcb71a39b05a03fee49",
    "gwdo-kim-wrf461-L50.npz": "64aafab05aa280cd2c870fae9acc977061a98c42456179fe85c1405986f32598",
    "gwdo-kim-wrf461-L90.npz": "8a5033c4259b8fc929b6ee342de12779c401a2dac773d9e44d98efc105f3a66d",
}
OUTPUTS = ("rublten", "rvblten", "dtaux3d", "dtauy3d", "dusfcg", "dvsfcg")


def _load(grid):
    with np.load(FIXTURE / f"gwdo-kim-inputs-{grid}.npz") as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(FIXTURE / f"gwdo-kim-wrf461-{grid}.npz") as f:
        wrf = {k: f[k] for k in f.files}
    return inputs, wrf


def _run(launch, inputs, xp=np):
    """``launch`` once per (dx, dt) case, laid out as the fixture."""
    nz, ncol = inputs["u"].shape

    def lev(a):
        return xp.asarray(np.ascontiguousarray(a, np.float32).reshape(a.shape[0], 1, ncol))

    def row(a):
        return xp.asarray(np.ascontiguousarray(a, np.float32).reshape(1, ncol))

    def host(a):
        return np.asarray(a.get() if hasattr(a, "get") else a)

    atm = {"u": lev(inputs["u"]), "v": lev(inputs["v"]),
           "temperature": lev(inputs["t"]), "qv": lev(inputs["qv"]),
           "pressure": lev(inputs["p"]), "exner": lev(inputs["pi"]),
           "p_interface": lev(inputs["p_interface"]), "z": lev(inputs["z"]),
           "dz": lev(inputs["dz"])}
    out = {k: [] for k in OUTPUTS}
    for dx, dt in zip(inputs["dx"], inputs["dt"]):
        du = lev(inputs["rublten0"]).copy()
        dv = lev(inputs["rvblten0"]).copy()
        diag = {}
        launch(atm, du, dv, var=row(inputs["var"]), con=row(inputs["oc1"]),
               oa=[row(inputs["oa"][m]) for m in range(4)],
               ol=[row(inputs["ol"][m]) for m in range(4)],
               sina=row(inputs["sina"]), cosa=row(inputs["cosa"]),
               dx=float(dx), dt=float(dt), diagnostics=diag)
        out["rublten"].append(host(du)[:, 0, :])
        out["rvblten"].append(host(dv)[:, 0, :])
        out["dtaux3d"].append(host(diag["DTAUX3D"])[:, 0, :])
        out["dtauy3d"].append(host(diag["DTAUY3D"])[:, 0, :])
        out["dusfcg"].append(host(diag["DUSFCG"])[0])
        out["dvsfcg"].append(host(diag["DVSFCG"])[0])
    return {k: np.ascontiguousarray(np.stack(v)) for k, v in out.items()}


def _missed_words(got, wrf):
    return {name: int((np.asarray(got[name], np.float32).view(np.uint32)
                       != np.asarray(wrf[name], np.float32).view(np.uint32)).sum())
            for name in OUTPUTS}


def test_the_fixture_is_the_engine_oracle_unchanged():
    for name, digest in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
    for grid, nz in (("L50", 50), ("L90", 90)):
        inputs, wrf = _load(grid)
        assert inputs["u"].shape == (nz, 149)
        assert inputs["dx"].tolist() == [1000.0, 3000.0, 9000.0, 12000.0,
                                         30000.0, 50000.0]
        assert len(set(map(str, inputs["regime"]))) == 12
        for name in OUTPUTS:
            assert np.all(np.isfinite(wrf[name])), name


def test_the_oracle_is_not_vacuous():
    """WRF drags most columns, and at the 50 km case as hard as at 1 km."""
    for grid in GRIDS:
        inputs, wrf = _load(grid)
        moved = (wrf["rublten"].view(np.uint32)
                 != inputs["rublten0"][None].view(np.uint32)).any(axis=1)
        # every case drags more than half the columns (the rest are the
        # calm, convective, ocean and edge columns the scheme switches off)
        assert moved.sum(axis=1).min() > 75, moved.sum(axis=1)
        # the 50 km / 300 s case, this model's grid length: a real stress
        assert float(np.abs(wrf["dusfcg"][5]).max()) > 0.1


@pytest.mark.parametrize("grid", GRIDS)
def test_the_numpy_transcription_lands_on_wrf_s_words(grid):
    """The known answer, without a card.

    On Linux numpy's float32 power is glibc's powf, the words the kernel's
    gfk_pow reproduces, so every output word must be WRF's (measured on the
    test host: 0 of every field differs, both grids, all six grid lengths).
    Elsewhere the platform's powf may round differently in its last place,
    and the answer is held to WRF within a relative 1e-4 of each field's
    largest value instead.
    """
    inputs, wrf = _load(grid)
    got = _run(reference_launch_gwdo, inputs)
    if sys.platform.startswith("linux"):
        assert _missed_words(got, wrf) == {name: 0 for name in OUTPUTS}
    else:
        for name in OUTPUTS:
            scale = float(np.abs(wrf[name]).max())
            diff = float(np.abs(got[name].astype(np.float64) - wrf[name]).max())
            assert diff <= 1.0e-4 * scale, (name, diff, scale)


def test_the_transcription_rejects_a_changed_sector_rule():
    """The grade can fail: the double atan2 the engine once used misses."""
    from arwen_global.physics import orographic_drag as od

    inputs, wrf = _load("L50")
    saved = od._atan2f
    try:
        od._atan2f = lambda y, x: np.arctan2(
            np.asarray(y, np.float64), np.asarray(x, np.float64)).astype(np.float32)
        got = _run(reference_launch_gwdo, inputs)
    finally:
        od._atan2f = saved
    sel = np.asarray(inputs["regime"]) == "sector_boundary"
    missed = (got["rublten"].view(np.uint32) != wrf["rublten"].view(np.uint32))
    assert missed[..., sel].any() and not missed[..., ~sel].any()


@pytest.mark.parametrize("grid", GRIDS)
def test_the_carried_kernel_lands_on_wrf_s_words_on_a_card(grid):
    """Word for word under the engine's strict numerics.

    The engine's compile boundary (``gpuwm.numerics``, default ``fast``
    since 2.9) drops the kernel's ``-fmad=false`` request, so NVRTC fuses
    multiply-adds, and the words move: MEASURED on an RTX 4090 (gpuwm 2.9.13,
    cupy 14.2, NVRTC 13.5) in fast mode the L90 grid differed from WRF in
    11,306 of 80,460 RUBLTEN words, up to 0.17 m s-2 where a fused rounding
    flips one of the scheme's branches, and the engine's own kernel differed
    by exactly the same words.  Under ``GPUWM_NUMERICS=strict`` both grids
    land on every word.  One process runs one mode, so this runs in a
    process that selected strict and says so otherwise.
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDevice()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no usable CUDA device: {exc}")
    numerics = pytest.importorskip("gpuwm.numerics")
    if numerics.mode() != "strict":
        pytest.skip("the engine's numerics are fast (multiply-adds fused at "
                    "the compile boundary); run with GPUWM_NUMERICS=strict "
                    "to grade against WRF's words")
    from arwen_global.core.terrain_drag import launch_gwdo

    inputs, wrf = _load(grid)
    got = _run(launch_gwdo, inputs, xp=cp)
    assert _missed_words(got, wrf) == {name: 0 for name in OUTPUTS}


@pytest.mark.parametrize("grid", GRIDS)
def test_the_carried_kernel_is_the_engine_kernel_word_for_word(grid):
    """In whichever numerics mode the process runs: both compile through the
    same boundary, so the carried copy must be the engine's words."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDevice()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no usable CUDA device: {exc}")
    engine = pytest.importorskip("gpuwm.core.terrain_drag")
    from arwen_global.core.terrain_drag import launch_gwdo

    inputs, _wrf = _load(grid)
    ours = _run(launch_gwdo, inputs, xp=cp)
    theirs = _run(engine.launch_gwdo, inputs, xp=cp)
    assert _missed_words(ours, theirs) == {name: 0 for name in OUTPUTS}


def test_the_carried_module_assembles_from_the_libm_header_and_its_kernel():
    from arwen_global.core import terrain_drag

    source = terrain_drag.module_source()
    kernels = Path(terrain_drag.__file__).parent / "kernels"
    assert source.endswith((kernels / "glibc_flt32.cuh").read_text(encoding="utf-8")
                           + (kernels / "terrain_drag.cu").read_text(encoding="utf-8"))
    assert "void gwdo_column(" in source
    for absent in ("void gwdo_gsl_column(", "void topo_wind_static(",
                   "void terrain_pbl_top("):
        assert absent not in source, absent
    assert terrain_drag.MODULE_OPTIONS == ("-std=c++17", "-fmad=false", "--ftz=false")


# ---------------------------------------------------------------------------
# options and identity
# ---------------------------------------------------------------------------

def _raw(**extra):
    return {"acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT, **extra}


def test_the_drag_is_off_by_default_and_out_of_the_identity():
    options = NativePhysicsOptions.from_mapping(_raw())
    assert options.orographic_drag == "none"
    assert not options.orographic_drag_enabled
    assert "orographic_drag" not in options.identity
    spelled = NativePhysicsOptions.from_mapping(_raw(orographic_drag="none"))
    assert spelled.identity == options.identity


def test_the_drag_joins_the_identity_only_when_on():
    on = NativePhysicsOptions.from_mapping(_raw(orographic_drag="kim-gwdo"))
    assert on.orographic_drag_enabled
    assert on.identity["orographic_drag"] == "kim-gwdo"
    with pytest.raises(ValueError, match="orographic_drag must be"):
        NativePhysicsOptions.from_mapping(_raw(orographic_drag="gsl"))


# ---------------------------------------------------------------------------
# the seam into the native suite (numpy, fake schemes)
# ---------------------------------------------------------------------------

def _native_helpers():
    import test_arwen_global_level5_native as native

    return native


def _statistics(shape, *, var=400.0, sha="a" * 64):
    nlat, nlon = shape
    planes = {name: np.zeros(shape, np.float32) for name in SSO_FIELDS}
    planes["VAR"][...] = np.float32(var)
    planes["CON"][...] = np.float32(2.0)
    for m in range(1, 5):
        planes[f"OA{m}"][...] = np.float32(0.3)
        planes[f"OL{m}"][...] = np.float32(0.4)
    # one column with no statistics: the drag must leave it alone
    planes["VAR"][0, 0] = np.float32(0.0)
    return OrographicStatistics(planes=planes, identity={
        "kind": "orographic-statistics", "dataset": "orogwd_10m",
        "self_sha256": sha})


def _windy(exchange, speed=20.0):
    u = np.full_like(np.asarray(exchange.u), speed)
    v = np.full_like(np.asarray(exchange.v), 0.25 * speed)
    return replace(exchange, u=u, v=v)


def _suite(drag, statistics=None, modules=None):
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite

    native = _native_helpers()
    options = native._options()
    if drag:
        options["orographic_drag"] = "kim-gwdo"
    modules = dict(native._fake_modules() if modules is None else modules)
    modules["arwen_global.core.terrain_drag"] = type(
        "TerrainDrag", (), {"launch_gwdo": staticmethod(reference_launch_gwdo)})
    suite = ArwenCudaColumnSuite(options, array_module=np, modules=modules)
    suite.orographic_statistics = statistics
    return suite


def test_the_suite_adds_the_drag_to_ysu_s_rates_from_the_state_ysu_read():
    from arwen_global.physics.native_batch import NativeColumnBatch
    from arwen_global.physics.native_runtime import NativePhysicsRuntime
    from arwen_global.physics.orographic_drag import column_atmosphere

    native = _native_helpers()
    exchange = _windy(native._exchange())
    shape = tuple(np.asarray(exchange.u).shape[1:])
    statistics = _statistics(shape)
    off = _suite(False).step(exchange)
    on = _suite(True, statistics).step(exchange)

    # what the drag alone does to the column the PBL step was handed: the
    # fake YSU returns zero rates, so the whole difference is the drag
    batch = NativeColumnBatch.from_exchange(exchange, np)
    du = np.zeros_like(batch.arrays["u"])
    dv = np.zeros_like(batch.arrays["v"])
    reference_launch_gwdo(
        column_atmosphere(batch, NativePhysicsRuntime._layer_thickness(batch)), du, dv,
        var=statistics.planes["VAR"], con=statistics.planes["CON"],
        oa=[statistics.planes[f"OA{m}"] for m in range(1, 5)],
        ol=[statistics.planes[f"OL{m}"] for m in range(1, 5)],
        sina=np.zeros(shape, np.float32), cosa=np.ones(shape, np.float32),
        dx=50_000.0, dt=float(exchange.dt_s))
    assert np.abs(du).max() > 0.0, "the fixture column state drags nowhere"
    step = np.float32(exchange.dt_s)
    # exchange arrays are top-first; the batch is bottom-first
    want_u = (np.asarray(off.u, np.float32)[::-1] + step * du)[::-1]
    want_v = (np.asarray(off.v, np.float32)[::-1] + step * dv)[::-1]
    assert np.array_equal(np.asarray(on.u, np.float32), want_u)
    assert np.array_equal(np.asarray(on.v, np.float32), want_v)
    # drag opposes the low-level wind and the statistics-free column is left alone
    assert (du[:, 0, 0] == 0.0).all() and (dv[:, 0, 0] == 0.0).all()
    assert du[0][np.abs(du[0]) > 0].max() < 0.0
    # everything but the wind is the drag-off result
    for name in ("theta", "qv", "qc"):
        assert np.array_equal(getattr(on, name), getattr(off, name)), name
    assert on.physics_state.metadata[METADATA_KEY] == statistics.identity
    assert METADATA_KEY not in off.physics_state.metadata


def test_the_drag_on_with_no_statistics_is_refused_by_name():
    native = _native_helpers()
    exchange = native._exchange()
    with pytest.raises(ValueError, match="no sub-grid orography statistics"):
        _suite(True, None).step(exchange)


def test_a_restart_that_would_read_other_statistics_is_refused():
    native = _native_helpers()
    exchange = _windy(native._exchange())
    shape = tuple(np.asarray(exchange.u).shape[1:])
    first = _suite(True, _statistics(shape)).step(exchange)
    resumed = replace(
        exchange, time_s=exchange.time_s + exchange.dt_s,
        physics_state=first.physics_state, surface=first.surface)
    # the same statistics continue
    _suite(True, _statistics(shape)).step(resumed)
    with pytest.raises(ValueError, match="not the ones this run started with"):
        _suite(True, _statistics(shape, sha="b" * 64)).step(resumed)


def test_a_band_takes_its_own_rows_of_the_statistics():
    planes = {name: np.arange(6 * 8, dtype=np.float32).reshape(6, 8) + i
              for i, name in enumerate(SSO_FIELDS)}
    statistics = OrographicStatistics(planes=planes, identity={})
    band = statistics.band((2, 4), np, (2, 8))
    assert np.array_equal(band["VAR"], planes["VAR"][2:4])
    assert np.array_equal(band["OL4"], planes["OL4"][2:4])
    with pytest.raises(ValueError, match="another grid"):
        statistics.band((0, 3), np, (2, 8))


def test_apply_refuses_a_rate_it_cannot_add_into_in_place():
    from types import SimpleNamespace

    batch = SimpleNamespace(xp=np, band=None, dt_s=60.0, arrays={})
    rate = np.zeros((4, 3, 2), np.float64)
    with pytest.raises(TypeError, match="C-contiguous float32"):
        apply_orographic_drag(_statistics((3, 2)), batch,
                              {"du": rate, "dv": rate}, launch=None, dz=None,
                              dx_m=50_000.0, metadata={})
