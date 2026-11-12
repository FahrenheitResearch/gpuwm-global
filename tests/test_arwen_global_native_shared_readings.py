"""The native suite's per-call shared readings give the same bits.

A physics call took eight column-water integrals, three of them of a
state another one had just read, and built the layer thicknesses five
times from inputs no step writes.  ``NativePhysicsRuntime`` now hands the
readings on within one call (``_water_before`` / ``_water_after``,
``_dz``) and the suite's closure reads its three numbers in one transfer.
These tests run the tiny CPU configuration (the moist smoke config) with
the fake scheme modules of ``test_arwen_global_level5_native`` and compare
every array and diagnostic of two suite calls against the same calls with
every reading taken fresh, as rel/0.1.6 took them: ``array_equal``, not a
tolerance.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arwen_global.constants import NUMBER_MOMENTS, WATER_SPECIES
from arwen_global.physics import native_runtime as nr
from arwen_global.physics.native_batch import NativeColumnBatch
from arwen_global.physics.native_suite import ArwenCudaColumnSuite

from test_arwen_global_level5_native import _exchange, _fake_modules, _options


def _convection(result, atmosphere):
    # Rates that move water in every lane the cumulus booking measures.
    result.rthcuten[0] = 1.0e-3
    result.rqvcuten[0] = -2.0e-6
    result.rqccuten[1] = 5.0e-7
    result.rqicuten[2] = 2.0e-7
    result.rainc[...] = np.float32(1.0e-4)


def _ysu_moistening(modules):
    """The fake YSU with a vapour and cloud tendency, so the PBL booking's
    before and after differ (the stock fake returns zeros)."""
    stock = modules["arwen_global.core.ysu"].launch_ysu

    def launch_ysu(u, v, theta, qv, qc, qi, *args, **kwargs):
        out = stock(u, v, theta, qv, qc, qi, *args, **kwargs)
        out["dqv"][0] = np.float32(3.0e-7)
        out["dqc"][1] = np.float32(-1.0e-9)
        out["dtheta"][0] = np.float32(2.0e-4)
        return out

    modules["arwen_global.core.ysu"].launch_ysu = launch_ysu
    return modules


def _modules():
    return _ysu_moistening(_fake_modules(
        precipitation=(0.4, 0.1, 0.05), cumulus=_convection))


def _fresh_readings(monkeypatch):
    """Every reading taken where rel/0.1.6 took it: no hand-on, no held
    thicknesses, the offered ledger reading ignored."""
    monkeypatch.setattr(
        nr.NativePhysicsRuntime, "_water_before",
        lambda self, batch, reuse=True: batch.atmospheric_water_kg_m2())
    monkeypatch.setattr(
        nr.NativePhysicsRuntime, "_water_after",
        lambda self, batch, hand_on=True: batch.atmospheric_water_kg_m2())
    monkeypatch.setattr(
        nr.NativePhysicsRuntime, "_dz",
        lambda self, batch: nr.NativePhysicsRuntime._layer_thickness(batch))


def _two_calls():
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_modules())
    first = suite.step(exchange)
    second = suite.step(replace(
        exchange,
        time_s=float(exchange.time_s) + float(exchange.dt_s),
        u=first.u, v=first.v, theta=first.theta,
        **{name: getattr(first, name) for name in (*WATER_SPECIES, *NUMBER_MOMENTS)},
        surface=first.surface,
        physics_state=first.physics_state,
    ))
    return first, second


def _flatten(result):
    out = {}
    for name in ("u", "v", "theta", *WATER_SPECIES, *NUMBER_MOMENTS):
        out[name] = np.asarray(getattr(result, name))
    for name, value in result.surface.arrays().items():
        out[f"surface.{name}"] = np.asarray(value)
    for name, value in result.physics_state.arrays.items():
        out[f"physics.{name}"] = np.asarray(value)
    for name, value in result.planes.items():
        out[f"plane.{name}"] = np.asarray(value)
    return out


def test_shared_readings_give_the_bits_of_fresh_readings(monkeypatch):
    shared = _two_calls()
    with monkeypatch.context() as patch:
        _fresh_readings(patch)
        fresh = _two_calls()
    for call, (a, b) in enumerate(zip(shared, fresh)):
        arrays_a, arrays_b = _flatten(a), _flatten(b)
        assert arrays_a.keys() == arrays_b.keys()
        for name in arrays_a:
            assert arrays_a[name].dtype == arrays_b[name].dtype, (call, name)
            assert np.array_equal(arrays_a[name], arrays_b[name], equal_nan=True), (call, name)
        assert a.diagnostics == b.diagnostics, call
        assert a.physics_state.metadata == b.physics_state.metadata, call
    # The fakes moved water in the PBL, cumulus and microphysics bookings,
    # so the hand-ons were exercised on changing states, not on a still one.
    assert shared[0].diagnostics["maximum_native_water_residual_kg_m2"] >= 0.0
    assert float(np.max(np.abs(
        np.asarray(shared[0].qv) - np.asarray(_exchange().qv)))) > 0.0


def test_a_call_takes_five_column_water_integrals_and_one_thickness(monkeypatch):
    counts = {"water": 0, "dz": 0}
    water = NativeColumnBatch.atmospheric_water_kg_m2
    thickness = nr.NativePhysicsRuntime._layer_thickness

    def counted_water(self):
        counts["water"] += 1
        return water(self)

    def counted_thickness(batch):
        counts["dz"] += 1
        return thickness(batch)

    monkeypatch.setattr(NativeColumnBatch, "atmospheric_water_kg_m2", counted_water)
    monkeypatch.setattr(nr.NativePhysicsRuntime, "_layer_thickness",
                        staticmethod(counted_thickness))
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_modules())
    suite.step(_exchange())
    # Ledger before (handed to the PBL booking), PBL after (handed to the
    # cumulus booking), cumulus after (handed to the microphysics),
    # microphysics after, ledger after: five, where there were eight.
    assert counts["water"] == 5
    assert counts["dz"] == 1
    runtime = suite._runtime
    assert runtime._call_dz is None and runtime._held_water is None
    assert runtime._offered_water is None and not runtime._in_call


def test_other_cumulus_schemes_read_their_before_fresh(monkeypatch):
    counts = {"water": 0}
    water = NativeColumnBatch.atmospheric_water_kg_m2

    def counted_water(self):
        counts["water"] += 1
        return water(self)

    monkeypatch.setattr(NativeColumnBatch, "atmospheric_water_kg_m2", counted_water)
    options = {**_options(), "cumulus": "ntiedtke"}
    suite = ArwenCudaColumnSuite(options, array_module=np, modules=_modules())
    suite.step(_exchange())
    # New Tiedtke is handed the batch's own water arrays, so its before
    # is not taken from the PBL booking.
    assert counts["water"] == 6


def test_a_reading_offered_for_another_batch_is_not_used():
    runtime = nr.NativePhysicsRuntime.__new__(nr.NativePhysicsRuntime)
    runtime._offered_water = (object(), "stale")
    runtime._held_water = None
    runtime._call_dz = None
    runtime._in_call = False

    class Probe(Exception):
        pass

    def stop(batch, cfg, persistent):
        assert runtime._held_water is None
        assert runtime._in_call
        raise Probe

    runtime._run = stop
    with pytest.raises(Probe):
        runtime.run(object(), None)
    assert runtime._offered_water is None and not runtime._in_call


def flip_and_conversion_bits(xp, shape):
    """Inbound: ``ascontiguousarray(float32(x)[::-1]) / (1 - qv)``;
    outbound: ``ascontiguousarray((r / (1 + r_v))[::-1])``, the chains
    rel/0.1.6 ran, against the one-pass forms, ``array_equal``.  Shared
    with the device half in test_arwen_global_insitu_fused_capture."""
    from arwen_global.physics.native_batch import _flipped_quotient

    rng = np.random.default_rng(3)
    for dtype in (np.float32, np.float64):
        qv = xp.asarray((rng.random(shape) * 3.0e-2).astype(np.float32))
        dry = 1.0 - qv
        source = xp.asarray((rng.random(shape) * 4.0e-3).astype(dtype))
        source[0, 0, :3] = 0.0
        chain = xp.ascontiguousarray(
            xp.ascontiguousarray(xp.asarray(source, dtype=xp.float32)[::-1]) / dry)
        fused = _flipped_quotient(xp, source, dry)
        assert fused.dtype == chain.dtype == xp.float32
        assert fused.flags.c_contiguous
        assert bool(xp.array_equal(fused, chain)), dtype
        moist = 1.0 + qv
        chain_out = xp.ascontiguousarray((source.astype(xp.float32) / moist)[::-1])
        out = xp.empty(shape, dtype=xp.float32)
        xp.divide(source.astype(xp.float32)[::-1], moist[::-1], out=out)
        assert bool(xp.array_equal(out, chain_out)), dtype


@pytest.mark.parametrize("shape", [(8, 32, 64), (40, 384, 768)], ids=["t21", "t255"])
def test_the_one_pass_flip_and_conversion_are_the_chain_bits(shape):
    flip_and_conversion_bits(np, shape)


def test_the_batch_returns_the_chain_bits():
    exchange = _exchange()
    batch = NativeColumnBatch.from_exchange(exchange, np)
    arrays = {name: value.copy() for name, value in batch.arrays.items()}
    moist = 1.0 + np.asarray(arrays["qv"], dtype=np.float32)
    returned = batch.global_prognostics(consume=True)
    for name in WATER_SPECIES:
        chain = np.ascontiguousarray(
            (np.asarray(arrays[name], dtype=np.float32) / moist)[::-1])
        assert returned[name].flags.c_contiguous
        assert np.array_equal(returned[name], chain), name
    # Inbound, against the specification the module docstring states.
    qv = np.ascontiguousarray(np.asarray(exchange.qv, dtype=np.float32)[::-1])
    for name in WATER_SPECIES:
        source = np.ascontiguousarray(
            np.asarray(getattr(exchange, name), dtype=np.float32)[::-1])
        assert np.array_equal(arrays[name], source / (1.0 - qv)), name


def test_the_overdraft_refusal_still_fires_from_the_one_read():
    def drown(result, atmosphere):
        # A cumulus call that moistens the column by far more than the
        # reservoir holds: the closure must refuse it before returning.
        result.rqvcuten[...] = np.float32(1.0)

    exchange = _exchange()
    surface = exchange.surface.copy()
    surface.water_kg_m2[...] = np.float32(0.0)
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(cumulus=drown))
    with pytest.raises(FloatingPointError, match="exceeds the explicit surface reservoir"):
        suite.step(replace(exchange, surface=surface))
