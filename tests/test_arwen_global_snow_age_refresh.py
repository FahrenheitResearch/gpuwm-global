"""How much snow it takes to make a pack new: the native option ``snow_age_refresh``.

WRF's ALCALC sets the snow age to zero on any call that snows, of any
amount.  On this model a trace of snow reaches most snow-covered land on
most land calls, so the age never grew: in the 365-day T127 run with
``snow_ageing = true`` 79 % of the snow-covered land columns Noah integrates
on 2027-02-28 were under 0.01 day old (90th percentile 0.35 day), and the
packs sat at their fresh-snow albedo all winter
(surface albedo 0.65 against ERA5's 0.45 over Siberia in February).
``snow_age_refresh = "amount"`` is Noah-MP's snowfall-amount refresh
(SNOW_AGE, after BATS) on WRF's Livneh age: the age grows by the call's
interval and a snowfall of dsnow kg/m2 then keeps ``1 - min(dsnow / SWEMX, 1)``
of it, SWEMX = 1 kg/m2, and bare ground carries no age.

CPU checks here, on the float64 mirror and the runtime; the CUDA kernel's
answer is checked in ``test_arwen_global_snow_age_refresh_cuda.py``.
"""
from __future__ import annotations

from arwen_global.configs_dir import config_root as _shipped_configs
from dataclasses import replace

import numpy as np
import pytest

from arwen_global.config import load_config
from arwen_global.physics.native_suite import ArwenCudaColumnSuite

from test_arwen_global_level5_native import _options
from test_arwen_global_snow_age import (
    DZS, LAND_DT, _ageing_modules, _cold_snowy_column, _snowy_start, alcalc_albedo,
)


def _aged(noah, params, dt, calls, **kwargs):
    from arwen_global.core.npref import np_noah_column

    col = _cold_snowy_column(noah, params)
    for _ in range(calls):
        out = np_noah_column(col, params, dt, DZS, **kwargs)
        col = {name: out[name] for name in col}
    return col


def _params():
    from arwen_global.core import noah

    return noah, noah.pack_params(noah.load_tables())


def test_the_amount_rule_ages_a_dry_pack_exactly_as_wrf_does():
    # With no snowfall the two rules are the same arithmetic: age += dt.
    noah, params = _params()
    wrf = _aged(noah, params, 3600.0, 24)
    amount = _aged(noah, params, 3600.0, 24, snow_age_refresh="amount")
    assert wrf["snotime"] == amount["snotime"] == 86400.0
    for name in wrf:
        assert np.array_equal(np.asarray(wrf[name]), np.asarray(amount[name])), name


@pytest.mark.parametrize("dsnow", [0.001, 0.05, 0.4])
def test_a_trace_of_snow_renews_only_its_share_of_the_pack(dsnow):
    """WRF's rule makes a day-old pack new under even 0.001 kg/m2 of snow;
    the amount rule keeps (age + dt)(1 - dsnow / 1 kg/m2) of its age."""
    from arwen_global.core.npref import np_noah_column

    noah, params = _params()
    lvcoef = float(params.gen[noah.GEN["lvcoef"]])
    dt = 3600.0
    col = _aged(noah, params, dt, 24, snow_age_refresh="amount")
    snowing = {**col, "rainbl": dsnow, "sr": 1.0}
    wrf = np_noah_column(snowing, params, dt, DZS, frpcpn=True)
    assert wrf["snotime"] == 0.0
    out = np_noah_column(snowing, params, dt, DZS, frpcpn=True, snow_age_refresh="amount")
    age = (86400.0 + dt) * (1.0 - dsnow / noah.SNOW_AGE_SWEMX_KG_M2)
    assert out["snotime"] == pytest.approx(age, rel=1.0e-12)
    assert out["albedo"] == pytest.approx(
        alcalc_albedo(out["albbck"], col["snoalb"], lvcoef, out["snowc"], age), rel=1.0e-12)
    assert wrf["albedo"] - out["albedo"] > 0.03
    # The snow itself lands on the pack the same way under both rules.
    assert out["snow"] == wrf["snow"]


@pytest.mark.parametrize("dsnow", [1.0, 3.0])
def test_a_snowfall_of_swemx_or_more_makes_the_pack_new_under_both_rules(dsnow):
    from arwen_global.core.npref import np_noah_column

    noah, params = _params()
    col = _aged(noah, params, 3600.0, 24, snow_age_refresh="amount")
    snowing = {**col, "rainbl": dsnow, "sr": 1.0}
    wrf = np_noah_column(snowing, params, 3600.0, DZS, frpcpn=True)
    out = np_noah_column(snowing, params, 3600.0, DZS, frpcpn=True, snow_age_refresh="amount")
    assert wrf["snotime"] == out["snotime"] == 0.0
    assert wrf["albedo"] == out["albedo"]


STALE_AGE_S = 30 * 86400.0


def _bare_with_stale_age(noah, params):
    """The cold grassland column after its pack melted out, still carrying
    the 30-day age the pack had when it went (ALCALC runs only under snow,
    so nothing else clears it)."""
    return {**_cold_snowy_column(noah, params), "snow": 0.0, "snowh": 0.0,
            "snowc": 0.0, "albedo": 0.2, "snotime": STALE_AGE_S}


@pytest.mark.parametrize("dsnow", [0.2, 0.001])
def test_a_pack_that_forms_on_bare_ground_starts_young_under_the_amount_rule(dsnow):
    """Noah-MP's SNOW_AGE sets the age to zero where there is no snow, so a
    trace on bare ground starts a pack dt (1 - dsnow / SWEMX) old, never with
    the melted pack's age.  WRF's rule zeroes it on the snowfall anyway."""
    from arwen_global.core.npref import np_noah_column

    noah, params = _params()
    dt = 3600.0
    snowing = {**_bare_with_stale_age(noah, params), "rainbl": dsnow, "sr": 1.0}
    wrf = np_noah_column(snowing, params, dt, DZS, frpcpn=True)
    assert wrf["snotime"] == 0.0
    out = np_noah_column(snowing, params, dt, DZS, frpcpn=True, snow_age_refresh="amount")
    assert out["snow"] > 0.0
    assert out["snotime"] == pytest.approx(dt * (1.0 - dsnow / noah.SNOW_AGE_SWEMX_KG_M2),
                                           rel=1.0e-12)


def test_bare_ground_carries_no_age_under_the_amount_rule_and_wrfs_is_untouched():
    from arwen_global.core.npref import np_noah_column

    noah, params = _params()
    bare = _bare_with_stale_age(noah, params)
    wrf = np_noah_column(bare, params, 3600.0, DZS)
    # WRF never reads or writes the age of a snow-free column: carried as is.
    assert wrf["snotime"] == STALE_AGE_S
    out = np_noah_column(bare, params, 3600.0, DZS, snow_age_refresh="amount")
    assert out["snotime"] == 0.0
    for name in bare:
        if name != "snotime":
            assert np.array_equal(np.asarray(out[name]), np.asarray(wrf[name])), name


def test_freezing_rain_does_not_renew_the_pack_under_either_rule():
    # ALCALC renews only when it snows (SNOWNG); freezing rain (FRZGRA)
    # adds to the pack but leaves its age.
    from arwen_global.core.npref import np_noah_column

    noah, params = _params()
    col = _aged(noah, params, 3600.0, 24, snow_age_refresh="amount")
    rain = {**col, "rainbl": 2.0, "sr": 0.0}
    for kwargs in ({}, {"snow_age_refresh": "amount"}):
        out = np_noah_column(rain, params, 3600.0, DZS, frpcpn=True, **kwargs)
        assert out["snotime"] == 86400.0 + 3600.0


def test_the_launcher_refuses_an_unknown_rule_before_touching_the_device():
    noah, params = _params()
    with pytest.raises(ValueError, match="snow_age_refresh"):
        noah.launch_noah({}, params, 600.0, DZS, snow_age_refresh="fresh")
    with pytest.raises(ValueError, match="swemx"):
        noah.launch_noah({}, params, 600.0, DZS, snow_age_refresh="amount",
                         snow_age_swemx_kg_m2=0.0)


def test_snow_age_refresh_is_wrf_by_default_and_amount_joins_the_identity():
    from arwen_global.physics.native_options import NativePhysicsOptions

    native = load_config(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))
    assert "snow_age_refresh" not in native.config_identity["native_adapter_options"]
    ageing = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_ageing": True})
    spelled = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_ageing": True, "snow_age_refresh": "any"})
    assert spelled.config_hash == ageing.config_hash
    amount = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_ageing": True, "snow_age_refresh": "amount"})
    assert amount.config_identity["native_adapter_options"]["snow_age_refresh"] == "amount"
    assert amount.config_hash != ageing.config_hash
    with pytest.raises(ValueError, match="needs snow_ageing"):
        NativePhysicsOptions.from_mapping({**_options(), "snow_age_refresh": "amount"})
    with pytest.raises(ValueError, match="snow_age_refresh must be"):
        NativePhysicsOptions.from_mapping(
            {**_options(), "snow_ageing": True, "snow_age_refresh": "trace"})


def test_the_runtime_hands_the_kernel_the_rule_only_when_it_is_not_wrfs():
    cfg, model, state, exchange, land = _snowy_start()
    calls = []

    def modules_for():
        modules = _ageing_modules([])
        inner = modules["arwen_global.core.noah"].launch_noah

        def launch_noah(dev, params, dt, dzs, **kwargs):
            calls.append(dict(kwargs))
            inner(dev, params, dt, dzs, **kwargs)

        modules["arwen_global.core.noah"].launch_noah = launch_noah
        return modules

    for rule in ("any", "amount"):
        suite = ArwenCudaColumnSuite(
            {**_options(), "land_surface_interval_s": LAND_DT, "snow_ageing": True,
             "snow_age_refresh": rule},
            array_module=np, modules=modules_for(),
        )
        suite.step(exchange)
    assert len(calls) == 2
    assert "snow_age_refresh" not in calls[0]
    assert calls[1]["snow_age_refresh"] == "amount"
