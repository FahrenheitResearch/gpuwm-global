"""Noah's snow age is carried between land calls, checkpointed and restarted.

Noah's ALCALC (WRF ``module_sf_noahlsm.F``; ``core/kernels/noah.cu:155-170``)
decays the snow albedo with the pack's age since its last snowfall::

    snoalb1 = snoalb + lvcoef (0.85 - snoalb)
    snotime = 0                      when snow falls this step
    snotime = snotime + dt           otherwise, and then
    snoalb2 = snoalb1 0.94 ** ((snotime / 86400) ** 0.58)
    albedo  = min(alb + sncovr (max(snoalb2, alb) - alb), max(snoalb2, alb))

The kernel reads SNOTIME and writes it back, and WRF carries it as state.
This model handed the kernel a fresh array of zeros on every land call, so
a snowpack was one land interval old forever and sat at its fresh-snow
albedo for the whole run.  Under the native option ``snow_ageing = true``
(off by default, set by a long run) the age lives in the persistent native
namespace as ``noah_snotime``.

On the CPU this file checks three things.  The runtime hands each land call
the age the previous call wrote.  A checkpoint carries the age and a
restart from it continues bit for bit.  An archive written before the age
was carried resumes at age zero, and the reader says so.  The known answer
of the ageing itself is checked on the float64 mirror here and on the CUDA
kernel in ``test_arwen_global_snow_age_cuda.py``.
"""
from __future__ import annotations

from arwen_global.configs_dir import config_root as _shipped_configs
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from arwen_global.checkpoint import (
    LATER_CARRIED_PHYSICS_ARRAYS,
    _canonical,
    read_checkpoint,
    state_from_checkpoint,
    write_checkpoint,
)
from arwen_global.config import load_config
from arwen_global.physics.native_state import NOAH_SNOW_AGE
from arwen_global.physics.native_suite import ArwenCudaColumnSuite
from arwen_global.runner import build_model_and_cold_state

from test_arwen_global_level5_native import _advance, _fake_modules, _options


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
#: The land interval of these tests; every call below is a due land call.
LAND_DT = 10.0


def _ageing_modules(seen):
    """The fakes of the native tests, with a Noah that ages its snow.

    It applies ALCALC's age rule to the columns the kernel integrates
    (land, not sea ice), and only where there is snow, as the kernel only
    calls ALCALC there: zero where snow falls this step, plus the elapsed
    step otherwise.  It hands the aged plane back as a NEW array rather
    than writing into the one it was given, so the runtime only keeps the
    age if it copies the kernel's output into the namespace itself.
    ``seen`` records the age each call was handed and the columns it aged.
    """
    modules = _fake_modules()
    stock = modules["arwen_global.core.noah"]

    def launch_noah(dev, params, dt, dzs, **kwargs):
        integrated = (np.asarray(dev["xland"]) < 1.5) & (np.asarray(dev["xice"]) < 0.5)
        snowy = integrated & (np.asarray(dev["snow"]) > 0.0)
        seen.append(SimpleNamespace(
            snotime=np.array(dev["snotime"], copy=True), snowy=snowy,
        ))
        snowing = (np.asarray(dev["rainbl"]) > 0.0) & (np.asarray(dev["sr"]) > 0.5)
        aged = np.where(
            snowy, np.where(snowing, np.float32(0.0), dev["snotime"] + np.float32(dt)),
            dev["snotime"],
        ).astype(np.float32)
        dev["snotime"] = aged

    modules["arwen_global.core.noah"] = SimpleNamespace(
        _F2D=stock._F2D, load_tables=stock.load_tables,
        pack_params=stock.pack_params, launch_noah=launch_noah,
    )
    return modules


def _snowy_start():
    """The smoke planet's exchange with 40 kg/m2 of snow on every column."""
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    shape = state.surface.land_fraction.shape
    arrays = state.physics_state.arrays
    arrays["noah_snow"] = np.full(shape, 40.0, np.float32)
    arrays["noah_snowh"] = np.full(shape, 0.2, np.float32)
    arrays["noah_snowc"] = np.ones(shape, np.float32)
    exchange = model._physics_exchange(state, 5.0)
    land = np.asarray(exchange.surface.land_fraction) >= 0.5
    assert land.any() and (~land).any(), "the smoke planet needs land and water"
    return cfg, model, state, exchange, land


def _suite(seen):
    return ArwenCudaColumnSuite(
        {**_options(), "land_surface_interval_s": LAND_DT, "snow_ageing": True},
        array_module=np, modules=_ageing_modules(seen),
    )


def _checkpoint(tmp_path, cfg, model, state, result, name="step.npz"):
    bundle = replace(state, surface=result.surface, physics_state=result.physics_state)
    backend = model.transform.backend
    return write_checkpoint(
        tmp_path / name, bundle, config_hash=cfg.config_hash,
        to_numpy=backend.to_numpy, semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )


def _read(cfg, path):
    return read_checkpoint(
        path, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )


def test_the_snow_age_is_created_and_carried_across_land_calls():
    cfg, model, state, exchange, land = _snowy_start()
    assert NOAH_SNOW_AGE not in exchange.physics_state.arrays
    seen = []
    suite = _suite(seen)
    t0 = float(exchange.time_s)
    first = suite.step(exchange)
    age = first.physics_state.arrays[NOAH_SNOW_AGE]
    assert age.dtype == np.float32 and age.shape == land.shape
    second = suite.step(_advance(exchange, first, t0 + LAND_DT))
    third = suite.step(_advance(exchange, second, t0 + 2.0 * LAND_DT))
    # The first call starts from fresh snow; each later call is handed the
    # age the call before it wrote, which is one more land step on every
    # snow-covered land column.  Without the carry every call saw zero.
    snowy = seen[0].snowy
    assert snowy.any() and (~snowy).any()
    for k, call in enumerate(seen):
        assert np.array_equal(call.snowy, snowy)
        assert np.all(call.snotime[snowy] == np.float32(k * LAND_DT)), k
    after = third.physics_state.arrays[NOAH_SNOW_AGE]
    assert np.all(after[snowy] == np.float32(3.0 * LAND_DT))
    # Open water is never integrated by Noah, so its age stays at the seed.
    assert not np.any(after[~snowy])


def test_with_ageing_off_no_age_plane_is_made_and_every_call_sees_zero():
    # Off is the arithmetic and the namespace of 0.1.8: the kernel is
    # handed zeros on every call and the state carries no age plane, so a
    # checkpoint has the arrays an 0.1.8 checkpoint has.
    cfg, model, state, exchange, land = _snowy_start()
    seen = []
    suite = ArwenCudaColumnSuite(
        {**_options(), "land_surface_interval_s": LAND_DT, "snow_ageing": False},
        array_module=np, modules=_ageing_modules(seen),
    )
    t0 = float(exchange.time_s)
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, t0 + LAND_DT))
    assert NOAH_SNOW_AGE not in first.physics_state.arrays
    assert NOAH_SNOW_AGE not in second.physics_state.arrays
    assert len(seen) == 2 and seen[0].snowy.any()
    for call in seen:
        assert not np.any(call.snotime)


def test_snowfall_resets_the_carried_age():
    cfg, model, state, exchange, land = _snowy_start()
    seen = []
    suite = _suite(seen)
    t0 = float(exchange.time_s)
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, t0 + LAND_DT))
    # Snow falls into the land bucket before the third call.
    snowing = _advance(exchange, second, t0 + 2.0 * LAND_DT)
    arrays = snowing.physics_state.arrays
    arrays["land_rainbl"] = np.full_like(arrays["land_rainbl"], 0.5)
    arrays["land_snowbl"] = np.full_like(arrays["land_snowbl"], 0.5)
    third = suite.step(snowing)
    snowy = seen[2].snowy
    assert np.all(seen[2].snotime[snowy] == np.float32(2.0 * LAND_DT))
    assert not np.any(third.physics_state.arrays[NOAH_SNOW_AGE])


def test_a_restart_from_a_checkpoint_continues_the_snow_age_bit_for_bit(tmp_path):
    cfg, model, state, exchange, land = _snowy_start()
    seen = []
    suite = _suite(seen)
    t0 = float(exchange.time_s)
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, t0 + LAND_DT))
    path = _checkpoint(tmp_path, cfg, model, state, second)
    metadata, arrays = _read(cfg, path)
    assert f"physics__{NOAH_SNOW_AGE}" in arrays
    assert metadata["absent_carried_physics_arrays"] == []
    back = state_from_checkpoint(metadata, arrays, model.transform.backend)
    assert np.array_equal(
        np.asarray(back.physics_state.arrays[NOAH_SNOW_AGE], np.float32),
        second.physics_state.arrays[NOAH_SNOW_AGE],
    )
    # The straight run takes its third call; a fresh runtime takes the same
    # call from the checkpointed namespace and surface.
    straight = suite.step(_advance(exchange, second, t0 + 2.0 * LAND_DT))
    restarted_seen = []
    restarted = _suite(restarted_seen).step(
        replace(
            _advance(exchange, second, t0 + 2.0 * LAND_DT),
            surface=back.surface, physics_state=back.physics_state,
        )
    )
    assert np.array_equal(restarted_seen[0].snotime, seen[2].snotime)
    assert np.any(seen[2].snotime)
    assert set(restarted.physics_state.arrays) == set(straight.physics_state.arrays)
    for name, value in straight.physics_state.arrays.items():
        other = restarted.physics_state.arrays[name]
        assert np.asarray(other).dtype == np.asarray(value).dtype, name
        assert np.array_equal(np.asarray(other), np.asarray(value)), name
    assert restarted.physics_state.metadata == straight.physics_state.metadata


def _without_the_age(path, older):
    """The same archive as a tree that did not carry the age would write it."""
    with np.load(path) as stored:
        kept = {name: stored[name] for name in stored.files if name != "__metadata__"}
        metadata = json.loads(str(stored["__metadata__"]))
    kept.pop(f"physics__{NOAH_SNOW_AGE}")
    metadata["arrays"].pop(f"physics__{NOAH_SNOW_AGE}")
    metadata.pop("self_sha256")
    metadata["self_sha256"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    np.savez_compressed(
        older, __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)), **kept
    )
    return older


def test_an_archive_without_the_snow_age_resumes_fresh_and_says_so(tmp_path, capsys):
    assert LATER_CARRIED_PHYSICS_ARRAYS == {NOAH_SNOW_AGE: "last_land_time_s"}
    cfg, model, state, exchange, land = _snowy_start()
    seen = []
    suite = _suite(seen)
    t0 = float(exchange.time_s)
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, t0 + LAND_DT))
    older = _without_the_age(
        _checkpoint(tmp_path, cfg, model, state, second), tmp_path / "older.npz"
    )
    metadata, arrays = _read(cfg, older)
    assert metadata["absent_carried_physics_arrays"] == [f"physics__{NOAH_SNOW_AGE}"]
    capsys.readouterr()
    # A reader that does not carry the age (the default, an export, an
    # inspection) is missing nothing and says nothing.
    state_from_checkpoint(metadata, arrays, model.transform.backend)
    assert capsys.readouterr().err == ""
    back = state_from_checkpoint(
        metadata, arrays, model.transform.backend,
        carried_physics_arrays=frozenset({NOAH_SNOW_AGE}))
    said = capsys.readouterr().err
    assert f"physics__{NOAH_SNOW_AGE}" in said
    assert "fresh-snow albedo" in said
    assert NOAH_SNOW_AGE not in back.physics_state.arrays
    restarted_seen = []
    _suite(restarted_seen).step(
        replace(
            _advance(exchange, second, t0 + 2.0 * LAND_DT),
            surface=back.surface, physics_state=back.physics_state,
        )
    )
    assert not np.any(restarted_seen[0].snotime)


def test_an_archive_from_before_any_land_call_is_not_reported(tmp_path, capsys):
    # A cold state's namespace has no age because no land call has run in
    # its lineage, and the resume seeds what the uninterrupted run seeds.
    cfg, model, state, exchange, land = _snowy_start()
    backend = model.transform.backend
    path = write_checkpoint(
        tmp_path / "cold.npz", state, config_hash=cfg.config_hash,
        to_numpy=backend.to_numpy, semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    metadata, arrays = _read(cfg, path)
    assert f"physics__{NOAH_SNOW_AGE}" not in arrays
    assert metadata["absent_carried_physics_arrays"] == []
    capsys.readouterr()
    state_from_checkpoint(metadata, arrays, backend)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# the known answer, on the float64 mirror
# ---------------------------------------------------------------------------

SIGMA = 5.67e-8
DZS = (0.1, 0.3, 0.6, 1.0)
SOILTYP = 6          # loam
VEGTYP = 10          # grassland


def _cold_snowy_column(noah, params):
    """A grassland column under 100 kg/m2 of snow at night, 255 K air, no
    precipitation: the pack stays and ages, and nothing falls to reset it."""
    sfctmp, sfcprs, dz8w1 = 255.0, 98000.0, 60.0
    psfc = sfcprs * (1.0 + 9.81 * 0.5 * dz8w1 / (287.0 * sfctmp))
    es = 611.2 * np.exp(21.87 * (sfctmp - 273.15) / (sfctmp - 7.66))
    qgh = 0.622 * es / (sfcprs - 0.378 * es)
    row = params.soil[SOILTYP - 1]
    smcmax = row[noah.SOIL_COLS.index("smcmax")]
    tslb = np.array([262.0, 265.0, 268.0, 271.0])
    smc = np.full(4, 0.6 * smcmax)
    return dict(
        psfc=psfc, sfcprs=sfcprs, sfctmp=sfctmp,
        qv1=0.8 * qgh, qgh=qgh, dz8w1=dz8w1,
        glw=0.75 * SIGMA * sfctmp ** 4, swdown=0.0,
        rainbl=0.0, sr=0.0, chs=0.005, cqs2=0.005, chs2=0.005, rib=0.0,
        ivgtyp=VEGTYP, isltyp=SOILTYP, vegfra=40.0, shdmin=10.0,
        shdmax=90.0, tmn=272.0, xland=1.0, xice=0.0, snoalb=0.75,
        embck=0.95, tsk=258.0, canwat=0.0, snow=100.0, snowh=0.4,
        snowc=1.0,
        smois=np.asarray(smc, np.float64),
        tslb=np.asarray(tslb, np.float64),
        sh2o=np.asarray(noah.sh2o_init(smc, tslb, SOILTYP, params), np.float64),
        albedo=0.7, albbck=0.2, emiss=0.95, z0=0.1, znt=0.1,
        snotime=0.0, lai=2.0, sfcrunoff=0.0, udrunoff=0.0, acsnow=0.0,
        acsnom=0.0, snopcx=0.0, potevp=0.0, hfx=0.0, qfx=0.0, lh=0.0,
        grdflx=0.0, qsfc=0.0)


def alcalc_albedo(alb, snoalb, lvcoef, sncovr, snotime_s):
    """ALCALC's albedo for a pack ``snotime_s`` old (no snowfall this step)."""
    snoalb1 = snoalb + lvcoef * (0.85 - snoalb)
    snoalb2 = max(snoalb1 * 0.94 ** ((snotime_s / 86400.0) ** 0.58), alb)
    return min(alb + sncovr * (snoalb2 - alb), snoalb2)


def test_the_mirror_ages_a_carried_pack_to_the_alcalc_answer():
    from arwen_global.core import noah
    from arwen_global.core.npref import np_noah_column

    params = noah.pack_params(noah.load_tables())
    lvcoef = float(params.gen[noah.GEN["lvcoef"]])
    assert lvcoef == 0.5     # GENPARM.TBL LVCOEF_DATA
    dt = 3600.0
    col = _cold_snowy_column(noah, params)
    for k in range(1, 25):
        out = np_noah_column(col, params, dt, DZS)
        assert out["skip"] == 0 and out["snow"] > 0.0
        # The age is the time since the pack's last snowfall: k steps.
        assert out["snotime"] == k * dt
        assert out["albedo"] == pytest.approx(
            alcalc_albedo(out["albbck"], col["snoalb"], lvcoef, out["snowc"], k * dt),
            rel=1.0e-12,
        )
        col = {name: out[name] for name in col}
    # One day old: the fresh pack's albedo times 0.94 (the cover is 0.98
    # at most, so the cell's albedo is that blend with the ground's).
    snoalb1 = 0.75 + 0.5 * (0.85 - 0.75)
    assert out["snotime"] == 86400.0
    assert out["albedo"] == pytest.approx(
        out["albbck"] + out["snowc"] * (0.94 * snoalb1 - out["albbck"]), rel=1.0e-12
    )
    # Against the pre-carry behaviour, the same pack handed age zero on
    # every call reads one step old: brighter by the ageing it lost.
    fresh = np_noah_column({**col, "snotime": 0.0}, params, dt, DZS)
    assert fresh["snotime"] == dt
    assert fresh["albedo"] - out["albedo"] > 0.03


def test_snow_ageing_is_off_by_default_and_on_joins_the_identity():
    """Off is the default and the zero-age arithmetic of 0.1.8 and earlier,
    so a shipped config keeps its hash: the Grell-Freitas native smoke
    under the external scheme hashed 367431fd... before the option existed,
    and the shipped semi-Lagrangian T255 day c7a000d2... (0.1.7's).  On is a
    new trajectory and joins the identity, so a checkpoint written under the
    zero-age arithmetic is refused rather than resumed with other
    physics."""
    from arwen_global.runner import carried_physics_arrays

    native = load_config(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))
    assert native.native_adapter_options.get("snow_ageing", False) is False
    assert "snow_ageing" not in native.config_identity["native_adapter_options"]
    assert replace(native, semi_implicit_scheme="external").config_hash == (
        "367431fd86f9b0d7d1338d9fbcaf97ff62cada4cb4e249c2ccc658789cde1836")
    day = load_config(str(
        _shipped_configs() / "arwen_global_gdas_t255_native_sl_si_24h.toml"))
    assert day.config_hash == (
        "c7a000d28c206ace5f88312d92320f4c30e72f7f0eb6d6108587022d935b4bbf")
    assert carried_physics_arrays(native) == frozenset()
    on = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_ageing": True})
    assert on.config_identity["native_adapter_options"]["snow_ageing"] is True
    assert on.config_hash != native.config_hash
    assert carried_physics_arrays(on) == frozenset({NOAH_SNOW_AGE})
    off = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_ageing": False})
    assert off.config_hash == native.config_hash
