"""The open-water exchange account: open water is not a finite store.

Every physics call books the column water it moves against the explicit
surface reservoir, and the run refuses when a column's reservoir reaches
zero.  On an ocean, a lake or the sea under its ice that reservoir is
bookkeeping, not water a column can run out of: 500 kg/m2 is 100 days at
5 mm a day of E - P, so a run of a few months would be refused on every
column whose evaporation exceeds its rain.  With native physics those
columns book the movement to a signed float64 account
(``water.OCEAN_EXCHANGE_NAMES``) that every water ledger counts, and
nothing the atmosphere reads changes: no native kernel reads the
reservoir.  The reference suite caps one call's evaporation by its
reservoir and keeps it.

Known answers, on the tiny CPU configuration (T3, dt 10 s), the native
suite driven with fake kernels that put a known surface flux into the
atmosphere:

1. E - P = +3 kg/m2 on an open-water column: the reservoir is unchanged,
   the account reads -3 and the ledger closes (exactly in the booking
   itself; to the float32 planes through the native closure).
2. The same on a land column: the reservoir is debited bit for bit as
   without the account.
3. The native suite's atmosphere with the account is the one without it,
   bit for bit (the 4-step native run on a card is
   ``test_arwen_global_open_water_exchange_cuda.py``).
4. The global water fixer's correction lands in the account on open water,
   so its "requires more surface water" refusal cannot fire there.
5. A column evaporating past the 24 h ceiling is refused by name, one
   just below it is not, and with the refusal off the watch only reports.
6. A checkpoint written without the account (the 0.1.5 bookkeeping)
   restarts, is given a zero account, and the receipt says so; a float32
   run carries the account in float64 and restarts bit for bit.
"""
from __future__ import annotations

from arwen_global.configs_dir import config_root as _shipped_configs
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.constants import WATER_SPECIES
from arwen_global.runner import build_model_and_cold_state, run
from arwen_global.state import PhysicsState
from arwen_global.water import (
    OCEAN_EXCHANGE_NAME,
    OPEN_WATER_EVAPORATION_NAME,
    book_open_water_exchange,
    exchange_columns,
    total_water_column,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
ACCOUNT = "physics__" + OCEAN_EXCHANGE_NAME
RESERVOIR = "surface__surface_water_kg_m2"


def _two_columns(reservoir_kg_m2):
    """One open-water column and one land column, float32 like the model."""
    return SimpleNamespace(
        water_kg_m2=np.full((1, 2), reservoir_kg_m2, np.float32),
        land_fraction=np.array([[0.0, 1.0]], np.float32),
        soil_water_fraction=np.zeros((4, 1, 2), np.float32),
    )


def test_e_minus_p_on_open_water_moves_to_the_account_and_land_keeps_its_debit():
    surface = _two_columns(500.0)
    arrays = {OCEAN_EXCHANGE_NAME: np.zeros((1, 2), np.float64)}
    bundle = SimpleNamespace(surface=surface, physics_state=PhysicsState(arrays=arrays))
    atmosphere = np.array([[20.0, 20.0]])
    before = total_water_column(bundle, None, np, atmospheric=atmosphere)["total"]

    # The call: both columns evaporate 3 kg/m2 more than they receive; the
    # physics books it against the reservoir as it always has.
    reservoir_before = surface.water_kg_m2.copy()
    surface.water_kg_m2 = surface.water_kg_m2 - np.float32(3.0)
    atmosphere = atmosphere + 3.0
    moved = book_open_water_exchange(surface, arrays, reservoir_before, np)

    assert surface.water_kg_m2.tolist() == [[500.0, 497.0]]
    assert arrays[OCEAN_EXCHANGE_NAME].tolist() == [[-3.0, 0.0]]
    assert moved.tolist() == [[-3.0, 0.0]]
    after = total_water_column(bundle, None, np, atmospheric=atmosphere)["total"]
    assert np.array_equal(after, before)
    assert surface.water_kg_m2.dtype == np.float32
    assert arrays[OCEAN_EXCHANGE_NAME].dtype == np.float64


def test_a_namespace_without_the_account_keeps_the_finite_reservoir():
    surface = _two_columns(500.0)
    arrays = {}
    reservoir_before = surface.water_kg_m2.copy()
    surface.water_kg_m2 = surface.water_kg_m2 - np.float32(3.0)
    assert book_open_water_exchange(surface, arrays, reservoir_before, np) is None
    assert surface.water_kg_m2.tolist() == [[497.0, 497.0]]
    assert arrays == {}


def test_the_account_columns_are_the_open_water_rule_without_the_ice_override():
    # Ocean, lake and the sea under its ice (land fraction below one half
    # by the xland rule) are open water; land and land ice are not.  The
    # sea-ice fraction does not enter: the set is fixed by the statics.
    land = np.array([[0.0, 0.3, 0.5, 0.7, 1.0]], np.float32)
    assert exchange_columns(land, np).tolist() == [[True, True, True, False, False]]


def test_native_closure_books_open_water_evaporation_to_the_account():
    from test_arwen_global_level5_native import (
        _exchange, _options, _surface_flux_modules,
    )
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite

    # E - P = +3 kg/m2 on every column: sfclay reports a uniform moisture
    # flux and the fake YSU puts exactly qfx * dt of vapor into the lowest
    # layer (ysu.cu:617), nothing else moves water.
    qfx = 0.6

    def call(with_account):
        exchange = _exchange()
        if with_account:
            exchange.physics_state.arrays[OCEAN_EXCHANGE_NAME] = np.zeros_like(
                np.asarray(exchange.surface.water_kg_m2))
        suite = ArwenCudaColumnSuite(
            _options(), array_module=np, modules=_surface_flux_modules(qfx))
        return exchange, suite.step(exchange)

    exchange, booked = call(True)
    _, finite = call(False)
    flux = qfx * float(exchange.dt_s)
    assert flux == pytest.approx(3.0)
    water = exchange_columns(np.asarray(exchange.surface.land_fraction), np)
    assert water.any() and (~water).any()

    dp = np.asarray(exchange.dp, np.float64)

    def column(result):
        return (sum(np.asarray(getattr(result, name), np.float64)
                    for name in WATER_SPECIES) * dp / 9.80665).sum(axis=0)

    # The fake YSU adds qfx * dt in the kernel's dry mixing ratio; the
    # model's specific-humidity metric, the one every ledger prices, reads
    # that as r / (1 + r) per layer, 0.53 percent less on this column.  The
    # account carries the measured gain, which is what the ledger closes on.
    gained = column(booked) - column(exchange)
    assert np.allclose(gained, flux, rtol=1.0e-2, atol=0.0)

    # 3. The atmosphere is the one the finite-reservoir closure returns.
    for name in ("u", "v", "theta", *WATER_SPECIES):
        assert np.array_equal(getattr(booked, name), getattr(finite, name)), name

    reservoir0 = np.asarray(exchange.surface.water_kg_m2, np.float32)
    reservoir_booked = np.asarray(booked.surface.water_kg_m2)
    reservoir_finite = np.asarray(finite.surface.water_kg_m2)
    account = np.asarray(booked.physics_state.arrays[OCEAN_EXCHANGE_NAME], np.float64)
    # 1. Open water: the reservoir is the one the column entered with, and
    # the account carries -(E - P), the column's gain as the returned state
    # reads it (to the float32 round trip of the batch, 1.5e-4 here, inside
    # the 1e-3 the reservoir booking test of the native suite allows).
    assert np.array_equal(reservoir_booked[water], reservoir0[water])
    assert np.allclose(account[water], -gained[water], rtol=0.0, atol=1.0e-3)
    assert np.allclose(account[water], -3.0, rtol=1.0e-2, atol=0.0)
    # 2. Land: today's debit, bit for bit, and nothing in the account.
    assert np.array_equal(reservoir_booked[~water], reservoir_finite[~water])
    assert np.allclose(
        reservoir_booked[~water].astype(np.float64) - reservoir0[~water],
        -gained[~water], rtol=0.0, atol=1.0e-3)
    assert np.all(account[~water] == 0.0)
    # The ledger closes: reservoir + account on open water is the finite
    # closure's reservoir, to the float32 rounding of the two planes.
    assert np.allclose(
        reservoir_booked[water].astype(np.float64) + account[water],
        reservoir_finite[water].astype(np.float64), rtol=0.0, atol=1.0e-4)
    assert (booked.diagnostics["maximum_local_water_repair_kg_m2"]
            == finite.diagnostics["maximum_local_water_repair_kg_m2"])
    assert OCEAN_EXCHANGE_NAME not in finite.physics_state.arrays


NATIVE_SMOKE = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")


def test_native_physics_books_the_account_and_the_reference_suite_keeps_its_reservoir():
    from arwen_global.runner import books_open_water_exchange

    assert books_open_water_exchange(load_config(NATIVE_SMOKE)) is True
    assert books_open_water_exchange(load_config(CONFIG)) is False


def test_open_water_exchange_false_keeps_the_finite_reservoir_and_the_hash(tmp_path):
    from arwen_global.runner import books_open_water_exchange

    text = Path(NATIVE_SMOKE).read_text(encoding="utf-8")
    line = next(row for row in text.splitlines()
                if row.startswith("native_adapter_options = {"))
    assert line.rstrip().endswith("}")
    off = tmp_path / "native-smoke-finite-reservoir.toml"
    off.write_text(
        text.replace(line, line.rstrip()[:-1].rstrip()
                     + ", open_water_exchange = false }"),
        encoding="utf-8")
    on_cfg, off_cfg = load_config(NATIVE_SMOKE), load_config(off)
    assert off_cfg.native_adapter_options["open_water_exchange"] is False
    assert books_open_water_exchange(off_cfg) is False
    assert books_open_water_exchange(on_cfg) is True
    # Bookkeeping only: the config identity is the one the account gives.
    assert off_cfg.config_hash == on_cfg.config_hash


def test_a_reference_run_keeps_the_finite_reservoir_bookkeeping(tmp_path):
    result = run(load_config(CONFIG), tmp_path / "reference")
    assert result["status"] == "pass"
    _, arrays = read_checkpoint(tmp_path / "reference" / "arwen_global_step00000004.npz")
    assert ACCOUNT not in arrays
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["open_water_exchange"] == {
        "enabled": False, "account": None, "filled": None}


def _booking(monkeypatch):
    """The forecast loop as it runs native physics, on the reference smoke
    configuration: the model books open water to the account.  The
    reference suite itself keeps debiting its reservoir, so on these runs
    the account carries the water fixer's corrections on open water."""
    import arwen_global.runner as runner

    monkeypatch.setattr(runner, "books_open_water_exchange", lambda cfg: True)


def test_the_water_fixer_books_open_water_to_the_account():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    water = exchange_columns(np.asarray(state.surface.land_fraction), np)
    assert water.any() and (~water).any()
    # Empty reservoirs on open water and 100 kg/m2 on land, and a target
    # 50 kg/m2 below the current global mean: the correction is -50 on
    # every column, which the finite reservoir cannot pay on open water.
    state.surface.water_kg_m2 = np.where(water, 0.0, 100.0).astype(
        np.asarray(state.surface.water_kg_m2).dtype)
    finite = state.copy()
    current = model._global_mean_total_water(finite)
    model.set_conservation_targets(None, current - 50.0)
    with pytest.raises(FloatingPointError, match="requires more surface water"):
        model._fix_total_water(finite)

    model.open_water_exchange = True
    assert model.ensure_exchange_account(state) is not None
    assert np.asarray(state.physics_state.arrays[OCEAN_EXCHANGE_NAME]).dtype == np.float64
    fixed, correction = model._fix_total_water(state)
    assert correction == pytest.approx(-50.0, abs=1.0e-9)
    reservoir = np.asarray(fixed.surface.water_kg_m2)
    account = np.asarray(fixed.physics_state.arrays[OCEAN_EXCHANGE_NAME])
    assert np.all(reservoir[water] == 0.0)
    assert np.allclose(account[water], correction, rtol=0.0, atol=1.0e-12)
    assert np.allclose(reservoir[~water], 100.0 + correction, rtol=0.0, atol=1.0e-12)
    assert np.all(account[~water] == 0.0)
    # The fixer held the total: the global mean is the target.
    assert model._global_mean_total_water(fixed) == pytest.approx(
        current - 50.0, abs=1.0e-9)
    # The caller's namespace was not written.
    assert np.all(np.asarray(state.physics_state.arrays[OCEAN_EXCHANGE_NAME]) == 0.0)


def test_a_run_without_the_account_refuses_a_state_that_carries_one():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    assert model.open_water_exchange is False
    assert model.ensure_exchange_account(state) is None
    state.physics_state.arrays[OCEAN_EXCHANGE_NAME] = np.zeros_like(
        np.asarray(state.surface.water_kg_m2))
    with pytest.raises(ValueError, match="keeps the finite surface reservoir"):
        model.step(state, cfg.dt_s)


def test_a_checkpoint_without_the_account_restarts_with_a_zero_one_and_says_so(
        tmp_path, monkeypatch):
    cfg = load_config(CONFIG)
    run(cfg, tmp_path / "old")
    midpoint = tmp_path / "old" / "arwen_global_step00000002.npz"
    _, start = read_checkpoint(midpoint)
    assert ACCOUNT not in start
    _booking(monkeypatch)
    result = run(cfg, tmp_path / "resumed", restart=midpoint)
    assert result["status"] == "pass"
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["open_water_exchange"]["enabled"] is True
    filled = receipt["open_water_exchange"]["filled"]
    assert filled.startswith(f"the state carried no {OCEAN_EXCHANGE_NAME}")
    assert "at step 2" in filled

    _, resumed = read_checkpoint(tmp_path / "resumed" / "arwen_global_step00000004.npz")
    _, straight = read_checkpoint(tmp_path / "old" / "arwen_global_step00000004.npz")
    assert resumed[ACCOUNT].dtype == np.float64
    # The atmosphere is the straight run's, bit for bit.
    for name in straight:
        if name.startswith("atmosphere__"):
            assert np.array_equal(resumed[name], straight[name]), name
    # Open water's surface water is reservoir + account, the straight run's
    # reservoir to the last bits of the fixer's float64 global mean.
    water = exchange_columns(resumed["surface__land_fraction"], np)
    assert np.allclose(
        resumed[RESERVOIR][water] + resumed[ACCOUNT][water],
        straight[RESERVOIR][water], rtol=0.0, atol=1.0e-9)
    assert np.all(resumed[ACCOUNT][~water] == 0.0)
    assert np.allclose(
        resumed[RESERVOIR][~water], straight[RESERVOIR][~water], rtol=0.0, atol=1.0e-9)


def test_the_runaway_watch_reports_a_day_of_open_water_evaporation_without_refusing():
    from test_arwen_global_level5_native import (
        _advance, _exchange, _options, _surface_flux_modules,
    )
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite

    def suite(qfx):
        return ArwenCudaColumnSuite(
            {**_options(), "open_water_evaporation_refusal": False},
            array_module=np, modules=_surface_flux_modules(qfx))

    exchange = _exchange()
    exchange.physics_state.arrays[OCEAN_EXCHANGE_NAME] = np.zeros_like(
        np.asarray(exchange.surface.water_kg_m2), dtype=np.float64)
    water = exchange_columns(np.asarray(exchange.surface.land_fraction), np)
    dt = float(exchange.dt_s)

    # qfx = 0.6 kg/m2/s for 5 s: 3 kg/m2 on every open-water column.
    first = suite(0.6).step(exchange)
    plane = np.asarray(first.physics_state.arrays[OPEN_WATER_EVAPORATION_NAME])
    assert plane.dtype == np.float64
    assert np.allclose(plane[water], np.float32(0.6) * dt, rtol=1.0e-7, atol=0.0)
    assert np.all(plane[~water] == 0.0)
    assert first.diagnostics["maximum_open_water_evaporation_day_kg_m2"] == pytest.approx(3.0, rel=1.0e-6)
    assert first.diagnostics["maximum_open_water_evaporation_last_day_kg_m2"] == 0.0
    assert first.diagnostics["open_water_evaporation_exceeds_ceiling"] == 0.0

    # The same 24 h block adds; the next block closes the day and starts again.
    t0 = float(exchange.time_s)
    second = suite(0.6).step(_advance(exchange, first, t0 + 600.0))
    assert second.diagnostics["maximum_open_water_evaporation_day_kg_m2"] == pytest.approx(6.0, rel=1.0e-6)
    # 13 kg/m2/s for 5 s is 65 kg/m2 in the new block: above the 60 kg/m2
    # ceiling, reported and, with the refusal off, not refused.
    third = suite(13.0).step(_advance(exchange, second, t0 + 86400.0))
    assert third.diagnostics["maximum_open_water_evaporation_last_day_kg_m2"] == pytest.approx(6.0, rel=1.0e-6)
    assert third.diagnostics["maximum_open_water_evaporation_day_kg_m2"] == pytest.approx(65.0, rel=1.0e-6)
    assert third.diagnostics["open_water_evaporation_exceeds_ceiling"] == 1.0
    assert third.physics_state.metadata["open_water_evaporation_block"] == int((t0 + 86400.0) // 86400.0)


def test_a_float32_run_carries_a_float64_account_and_restarts_bit_for_bit(
        tmp_path, monkeypatch):
    _booking(monkeypatch)
    cfg = replace(load_config(CONFIG), precision="float32")
    run(cfg, tmp_path / "full")
    midpoint = tmp_path / "full" / "arwen_global_step00000002.npz"
    run(cfg, tmp_path / "resumed", restart=midpoint)
    _, a = read_checkpoint(tmp_path / "full" / "arwen_global_step00000004.npz")
    _, b = read_checkpoint(tmp_path / "resumed" / "arwen_global_step00000004.npz")
    assert a[ACCOUNT].dtype == np.float64
    assert a[RESERVOIR].dtype == np.float32
    _, zero = read_checkpoint(tmp_path / "full" / "arwen_global_step00000000.npz")
    assert np.all(zero[ACCOUNT] == 0.0)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].dtype == b[name].dtype, name
        assert np.array_equal(a[name], b[name]), name


@pytest.mark.parametrize("qfx, refused", [(13.0, True), (11.8, False)])
def test_a_column_evaporating_past_the_ceiling_is_refused_by_name(qfx, refused):
    """65 kg/m2 in one 24 h block against the 60 kg/m2 default is refused,
    naming the column; 59 kg/m2 is not."""
    from test_arwen_global_level5_native import (
        _exchange, _options, _surface_flux_modules,
    )
    from arwen_global.physics.native_options import NativePhysicsOptions
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite

    defaults = NativePhysicsOptions.from_mapping(_options())
    assert defaults.open_water_evaporation_ceiling_kg_m2_day == 60.0
    assert defaults.open_water_evaporation_refusal is True
    exchange = _exchange()
    exchange.physics_state.arrays[OCEAN_EXCHANGE_NAME] = np.zeros_like(
        np.asarray(exchange.surface.water_kg_m2), dtype=np.float64)
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_surface_flux_modules(qfx))
    if refused:
        with pytest.raises(FloatingPointError, match=(
                r"open-water column evaporated more than the runaway ceiling: "
                r"column \d+ of row \d+ of this band \(-?\d+\.\d+ deg latitude")):
            suite.step(exchange)
    else:
        result = suite.step(exchange)
        assert result.diagnostics["maximum_open_water_evaporation_day_kg_m2"] == pytest.approx(
            59.0, rel=1.0e-6)
        assert result.diagnostics["open_water_evaporation_exceeds_ceiling"] == 0.0


def test_the_watch_settings_are_validated_and_stay_out_of_the_identity():
    from test_arwen_global_level5_native import _options
    from arwen_global.physics.native_options import NativePhysicsOptions

    def options(**overrides):
        return NativePhysicsOptions.from_mapping({**_options(), **overrides})

    base = options()
    quiet = options(
        open_water_evaporation_ceiling_kg_m2_day=120.0,
        open_water_evaporation_refusal=False)
    assert quiet.identity == base.identity
    assert "open_water_evaporation_ceiling_kg_m2_day" not in base.identity
    assert "open_water_evaporation_refusal" not in base.identity
    for bad in (0.0, -1.0, float("inf"), True):
        with pytest.raises(ValueError, match="open_water_evaporation_ceiling_kg_m2_day"):
            options(open_water_evaporation_ceiling_kg_m2_day=bad)
    with pytest.raises(ValueError, match="open_water_evaporation_refusal"):
        options(open_water_evaporation_refusal=1)
    assert base.open_water_exchange is True
    assert options(open_water_exchange=False).identity == base.identity
    assert "open_water_exchange" not in base.identity
    with pytest.raises(ValueError, match="open_water_exchange"):
        options(open_water_exchange=0)
