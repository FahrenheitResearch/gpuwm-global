"""The open-water exchange account on the card: the native atmosphere is unchanged.

The CPU half (``test_arwen_global_open_water_exchange.py``) drives the
native suite with stand-in kernels.  This is the real five-scheme suite on
the native T3 smoke configuration's cold state, one physics half-step of
5 s through the model's own band loop, taken once with the open-water
exchange account in the namespace (what native physics books) and once
without it (the finite reservoir, the 0.1.5 bookkeeping).  Every
prognostic and tracer the call returns, every surface plane but the
reservoir and every namespace array but the account and its runaway watch
are byte-identical; on open water the booked call's reservoir keeps its
value and reservoir + account is the finite call's reservoir.

Why one half-step and not a forecast: the native T3 smoke configuration
does not complete its first step on a card, under this tree or under 0.1.5
(RRTMGP refuses a skin of 363.5 K against its 355 K bound, the same
reading under both), so a whole run cannot be compared.
"""
from __future__ import annotations

from arwen_global.configs_dir import config_root as _shipped_configs

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.config import load_config  # noqa: E402
from arwen_global.constants import PROGNOSTIC_FIELDS  # noqa: E402
from arwen_global.runner import build_model_and_cold_state  # noqa: E402
from arwen_global.state import SURFACE_ARRAY_NAMES  # noqa: E402
from arwen_global.water import (  # noqa: E402
    OCEAN_EXCHANGE_NAME,
    OPEN_WATER_EVAPORATION_NAME,
    exchange_columns,
)

NATIVE_SMOKE = str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")


def _host(value):
    return np.asarray(cp.asnumpy(value) if isinstance(value, cp.ndarray) else value)


def test_a_native_physics_call_with_the_account_has_the_atmosphere_of_the_finite_reservoir():
    cfg = load_config(NATIVE_SMOKE)
    outputs = {}
    for booked in (True, False):
        model, state = build_model_and_cold_state(cfg)
        assert model.open_water_exchange is True
        model.open_water_exchange = booked
        model.ensure_exchange_account(state)
        assert (OCEAN_EXCHANGE_NAME in state.physics_state.arrays) is booked
        reservoir0 = _host(state.surface.water_kg_m2).copy()
        land = _host(state.surface.land_fraction).copy()
        after, _diagnostics = model._apply_physics_half(state, 0.5 * cfg.dt_s)
        outputs[booked] = {
            "fields": {
                name: _host(getattr(after.atmosphere, name))
                for name in PROGNOSTIC_FIELDS
            },
            "surface": {member: _host(getattr(after.surface, member))
                        for member in SURFACE_ARRAY_NAMES},
            "physics": {name: _host(value)
                        for name, value in after.physics_state.arrays.items()},
            "reservoir0": reservoir0,
            "land": land,
        }
    a, b = outputs[True], outputs[False]
    for name in b["fields"]:
        assert np.array_equal(a["fields"][name], b["fields"][name]), name
    differ = sorted(member for member in SURFACE_ARRAY_NAMES
                    if not np.array_equal(a["surface"][member], b["surface"][member]))
    assert differ in ([], ["water_kg_m2"])
    assert set(a["physics"]) - set(b["physics"]) == {
        OCEAN_EXCHANGE_NAME, OPEN_WATER_EVAPORATION_NAME}
    for name in b["physics"]:
        assert np.array_equal(a["physics"][name], b["physics"][name]), name
    water = exchange_columns(a["land"], np)
    assert water.any() and (~water).any()
    reservoir = a["surface"]["water_kg_m2"]
    account = a["physics"][OCEAN_EXCHANGE_NAME]
    assert account.dtype == np.float64
    assert np.array_equal(reservoir[water], a["reservoir0"][water])
    assert np.array_equal(reservoir[~water], b["surface"]["water_kg_m2"][~water])
    assert np.all(account[~water] == 0.0)
    # The call moved open-water surface water, so the comparison has teeth.
    assert np.any(account[water] != 0.0)
    assert np.allclose(
        reservoir[water].astype(np.float64) + account[water],
        b["surface"]["water_kg_m2"][water].astype(np.float64), rtol=0.0, atol=1.0e-4)
