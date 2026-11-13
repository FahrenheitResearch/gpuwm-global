"""The long-run rows (arwen_global.longrun_gates) on the card: the same
cold state on the cupy backend gives the numpy backend's rows.

Its own module because the suite marks every test of a module that opens
a device (tests/conftest.py), and the CPU known answers of
test_arwen_global_longrun_gates.py must stay in the CPU tiers.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from arwen_global import longrun_gates as lg
from arwen_global.configs_dir import config_root as _shipped_configs

BANDED = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")


@pytest.mark.gpu
def test_the_long_run_rows_on_the_card_are_the_host_rows():
    """The same cold state on the cupy backend: every long-run row is the
    numpy backend's to roundoff (the device reductions sum in another
    order), and none goes missing."""
    pytest.importorskip("cupy")
    from arwen_global.config import load_config
    from arwen_global.runner import build_model_and_cold_state

    cfg = load_config(BANDED)
    readings = []
    for backend in ("numpy", "cupy"):
        model, cold = build_model_and_cold_state(replace(cfg, backend=backend))
        readings.append(model.diagnostics(cold))
    host, card = readings
    names = [k for k in host
             if k.startswith(("global_mean_", "polar_cap_", "zonal_mean_u_",
                              "minimum_land_", "land_mean_", "nino34_"))]
    assert lg.level_row_name(10) in names and "land_mean_soil_water_kg_m2" in names
    assert set(names) <= set(card)
    for name in names:
        if host[name] is None:
            assert card[name] is None, name
            continue
        assert card[name] == pytest.approx(host[name], rel=1.0e-11, abs=1.0e-9), name
