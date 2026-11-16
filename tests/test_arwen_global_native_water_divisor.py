"""The bridge's outbound humidity conversion conserves water across a call.

Audit 2026-11-15 (fix/water-divisor): the inbound conversion fixes each
layer's dry fraction at d = 1 - q_v_in (r_x = q_x / d), but the outbound one
divided by the POST-kernel 1 + r_v.  The physics call holds dp fixed, so
the layer's dry-air mass d dp / g is fixed through it; pricing the returned
species against a dry mass that moves with the vapour made a pure
vapour-to-condensate change (which conserves sum_x r_x) raise the model's
ledger sum_x q_x dp / g by about q_v times the condensed amount.  On one
moist column that was +0.151 kg/m2 on 10.04 kg/m2 condensed (1.50%), and
after fallout the atmosphere lost 9.913 kg/m2 against a kernel fallout of
10.044: the rain books no longer matched the atmosphere's loss.  The
batch now holds d (``NativeColumnBatch.dry_fraction``) and both the ledger
and the outbound conversion use q_x = r_x d.

The breakage these tests prevent, named: Grell-Freitas's RAINC capped
1.5-3% below its own removed water, and Morrison rain on land debiting the
surface reservoir by q_v times the rain on every raining land column.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arwen_global.constants import GRAVITY_M_S2, WATER_SPECIES
from arwen_global.physics.native_batch import NativeColumnBatch

from test_arwen_global_level5_native import _exchange


def _moist_exchange():
    """The cold exchange with one tropical column planted at (0, 0):
    q_v from 0.019 at the lowest level decaying upward, and no condensate."""
    exchange = _exchange()
    qv = np.array(exchange.qv, copy=True)
    nlev = qv.shape[0]
    # Model order is top-to-bottom: the last level is the lowest.
    profile = 0.019 * np.exp(-np.arange(nlev)[::-1] / 4.0)
    qv[:, 0, 0] = profile.astype(qv.dtype)
    zeros = {name: np.zeros_like(getattr(exchange, name))
             for name in WATER_SPECIES if name != "qv"}
    return replace(exchange, qv=qv, **zeros)


def _model_ledger(species, dp):
    """The dycore's ledger (water.atmospheric_water_column) summed over
    levels, in float64: sum_x q_x dp / g."""
    total = sum(np.asarray(species[name], np.float64) for name in WATER_SPECIES)
    return (total * np.asarray(dp, np.float64) / GRAVITY_M_S2).sum(axis=0)


def _column(ledger):
    return float(np.asarray(ledger)[0, 0])


def _old_ledger(batch):
    """The pre-fix ledger, sum_x r_x / (1 + r_v) dp / g, for contrast."""
    moist = 1.0 + batch.arrays["qv"].astype(np.float64)
    total = sum(batch.arrays[name].astype(np.float64) / moist
                for name in WATER_SPECIES)
    return (total * batch.arrays["dp"].astype(np.float64) / GRAVITY_M_S2).sum(axis=0)


def _condense(batch, fraction):
    """Move ``fraction`` of the column's vapour to cloud water, in mixing
    ratio (what every WRF-lineage kernel conserves); returns the amount
    moved in the model's metric, sum c d dp / g, in float64."""
    qv = batch.arrays["qv"]
    moved = (qv[:, 0, 0] * np.float32(fraction)).astype(np.float32)
    # The amount actually moved after float32 rounding of both species.
    new_qv = (qv[:, 0, 0] - moved).astype(np.float32)
    moved_exact = qv[:, 0, 0].astype(np.float64) - new_qv.astype(np.float64)
    qv[:, 0, 0] = new_qv
    batch.arrays["qc"][:, 0, 0] = (
        batch.arrays["qc"][:, 0, 0].astype(np.float64) + moved_exact
    ).astype(np.float32)
    dry = batch.dry_fraction[:, 0, 0].astype(np.float64)
    dp = batch.arrays["dp"][:, 0, 0].astype(np.float64)
    return float((moved_exact * dry * dp / GRAVITY_M_S2).sum())


def test_pure_condensation_conserves_the_ledger_and_fallout_is_the_loss():
    exchange = _moist_exchange()
    batch = NativeColumnBatch.from_exchange(exchange, np)
    # The inbound reading is the model's own ledger on the exchange.
    model_before = _column(_model_ledger(
        {name: getattr(exchange, name) for name in WATER_SPECIES}, exchange.dp))
    before = _column(batch.atmospheric_water_kg_m2())
    assert before == pytest.approx(model_before, rel=2.0e-7)
    old_before = _column(_old_ledger(batch))

    condensed = _condense(batch, 0.5)
    assert condensed > 5.0      # a real burst, kg/m2
    after_condensation = _column(batch.atmospheric_water_kg_m2())
    # Mixing-ratio water conserved -> the model's ledger conserved, to the
    # float32 rounding of the moved cloud water (one ulp of qc per level).
    assert abs(after_condensation - before) <= 1.0e-6 * condensed
    # The divisor the bug used gains about q_v times the condensed amount.
    old_gain = _column(_old_ledger(batch)) - old_before
    assert old_gain > 5.0e-3 * condensed

    # Fallout: all the cloud water leaves the column.
    qc = batch.arrays["qc"][:, 0, 0].astype(np.float64)
    dry = batch.dry_fraction[:, 0, 0].astype(np.float64)
    dp = batch.arrays["dp"][:, 0, 0].astype(np.float64)
    fallout = float((qc * dry * dp / GRAVITY_M_S2).sum())
    assert fallout == pytest.approx(condensed, rel=1.0e-6)
    batch.arrays["qc"][:, 0, 0] = 0.0
    after_fallout = _column(batch.atmospheric_water_kg_m2())
    assert before - after_fallout == pytest.approx(fallout, rel=1.0e-12, abs=1.0e-9)

    # What the model receives prices the same: the returned specific
    # humidities on the exchange's dp, up to the float32 return cast.
    back = batch.global_prognostics()
    model_after = _column(_model_ledger(back, exchange.dp))
    assert model_after == pytest.approx(after_fallout, rel=2.0e-7)
    assert model_before - model_after == pytest.approx(fallout, rel=1.0e-5)


def test_the_round_trip_returns_the_model_humidity_when_nothing_moves():
    exchange = _moist_exchange()
    batch = NativeColumnBatch.from_exchange(exchange, np)
    np.testing.assert_array_equal(
        batch.dry_fraction,
        (np.float32(1.0) - np.ascontiguousarray(
            np.asarray(exchange.qv, np.float32)[::-1])))
    back = batch.global_prognostics(consume=True)
    assert batch.dry_fraction is None
    for name in WATER_SPECIES:
        source = np.asarray(getattr(exchange, name), np.float32)
        np.testing.assert_allclose(back[name], source, rtol=2.4e-7, atol=0.0)


def test_a_batch_built_from_mixing_ratios_takes_its_own_dry_fraction():
    batch = NativeColumnBatch.from_exchange(_moist_exchange(), np)
    rebuilt = NativeColumnBatch(
        xp=np, arrays=dict(batch.arrays), surface=batch.surface,
        physics_state=batch.physics_state, time_s=batch.time_s,
        dt_s=batch.dt_s)
    np.testing.assert_allclose(
        rebuilt.dry_fraction, batch.dry_fraction, rtol=2.4e-7, atol=0.0)


def test_cumulus_rain_is_the_rates_water_and_a_disagreement_is_refused(monkeypatch):
    # Grell-Freitas priced its RAINCV on the moist layer mass, so capped at
    # the measured loss it withheld 1.56 percent of its rain at T255.  The
    # rain is now the column integral of the applied rates on the model's
    # mass, and the runtime refuses a call where that integral and the
    # measured column loss part by more than the float32 rounding of the
    # update (a water route the rates do not carry).
    from arwen_global.constants import CONVECTIVE_RAIN_ACCUMULATOR
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite
    from test_arwen_global_level5_native import _fake_modules, _options

    exchange = _moist_exchange()

    def convection(result, atmosphere):
        result.rqvcuten[0] = -2.0e-6
        result.rainc[...] = 1.0     # the scheme's own number, ignored

    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(cumulus=convection))
    result = suite.step(exchange)
    d = result.diagnostics
    rainc = np.asarray(result.physics_state.arrays[CONVECTIVE_RAIN_ACCUMULATOR], np.float64)
    # Round-off only: the float32 rounding of the updated vapour is 2^-24
    # of the column water (about 1.5e-6 kg/m2 on these columns), against
    # a 1e-3 relative disagreement the guard refuses below.
    assert float(rainc.mean()) == pytest.approx(d["cumulus_water_removed_kg_m2"], abs=2.0e-6)
    assert 0.0 <= d["cumulus_rain_withheld_kg_m2"] < 2.0e-6
    assert 0.0 <= d["cumulus_rain_closure_kg_m2"] < 2.0e-6

    integral = NativeColumnBatch.column_integral_kg_m2
    monkeypatch.setattr(
        NativeColumnBatch, "column_integral_kg_m2",
        lambda self, increment: 1.001 * integral(self, increment))
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(cumulus=convection))
    with pytest.raises(FloatingPointError, match="cumulus rain"):
        suite.step(exchange)
