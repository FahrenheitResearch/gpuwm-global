"""The fresh-snow albedo of a snow-covered cell: the native option ``snow_fresh_albedo``.

WRF's ALCALC lifts the maximum snow albedo half way to 0.85 for fresh snow
(SNOALB1 = snoalb + LVCOEF (0.85 - snoalb)).  snoalb is the MODIS maximum
snow albedo, the albedo of the whole cell under deep fresh snow, canopy
included, so a fresh evergreen needleleaf cell sat 0.2 above the observed
maximum.  ``"masked"`` keeps snoalb as the fresh value and ages only the
exposed snow, fexp = (snoalb - alb) / (0.85 - alb) of the cell, on Livneh's
curve from 0.85; the rest of the cell keeps the snow-free albedo.

CPU checks on the float64 mirror, the launcher and the options.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs

from test_arwen_global_level5_native import _options
from test_arwen_global_snow_age import DZS, _cold_snowy_column


def _params():
    from arwen_global.core import noah

    return noah, noah.pack_params(noah.load_tables())


def _aged(noah, params, dt, calls, **kwargs):
    from arwen_global.core.npref import np_noah_column

    col = _cold_snowy_column(noah, params)
    for _ in range(calls):
        out = np_noah_column(col, params, dt, DZS, **kwargs)
        col = {name: out[name] for name in col}
    return col, out


def test_livneh_is_the_default_arithmetic():
    noah, params = _params()
    a, oa = _aged(noah, params, 3600.0, 24)
    b, ob = _aged(noah, params, 3600.0, 24, snow_fresh_albedo="livneh")
    for name in a:
        assert np.array_equal(np.asarray(a[name]), np.asarray(b[name])), name
    assert oa["albedo"] == ob["albedo"]


def test_masked_fresh_value_is_the_maximum_snow_albedo_and_only_exposed_snow_ages():
    from arwen_global.core.npref import _noah_alcalc

    alb, snoalb = 0.12, 0.39
    a0, _, _ = _noah_alcalc(alb, snoalb, 0.95, 1.0, 600.0, True, 0.0, 0.5, fresh="masked")
    assert a0 == pytest.approx(snoalb, abs=1e-12)
    l0, _, _ = _noah_alcalc(alb, snoalb, 0.95, 1.0, 600.0, True, 0.0, 0.5)
    assert l0 == pytest.approx(snoalb + 0.5 * (0.85 - snoalb), abs=1e-12)
    age = 20 * 86400.0
    a20, _, t = _noah_alcalc(alb, snoalb, 0.95, 1.0, 0.0, False, age, 0.5, fresh="masked")
    assert t == age
    dec = 0.94 ** ((age / 86400.0) ** 0.58)
    fexp = (snoalb - alb) / (0.85 - alb)
    assert a20 == pytest.approx(alb + fexp * (0.85 * dec - alb), abs=1e-12)
    assert alb < a20 < snoalb


def test_masked_changes_the_albedo_not_the_age_or_the_snow_on_the_call():
    noah, params = _params()
    a, oa = _aged(noah, params, 3600.0, 1)
    m, om = _aged(noah, params, 3600.0, 1, snow_fresh_albedo="masked")
    assert om["snotime"] == oa["snotime"]
    assert om["snow"] == oa["snow"]
    assert om["albedo"] < oa["albedo"]


def test_the_launcher_refuses_an_unknown_rule_before_touching_the_device():
    noah, params = _params()
    with pytest.raises(ValueError, match="snow_fresh_albedo"):
        noah.launch_noah({}, params, 600.0, DZS, snow_fresh_albedo="bright")


def test_livneh_is_the_default_and_masked_joins_the_identity():
    from arwen_global.physics.native_options import NativePhysicsOptions

    native = load_config(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))
    assert "snow_fresh_albedo" not in native.config_identity["native_adapter_options"]
    spelled = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_fresh_albedo": "livneh"})
    assert spelled.config_hash == native.config_hash
    masked = replace(native, native_adapter_options={
        **native.native_adapter_options, "snow_fresh_albedo": "masked"})
    assert masked.config_identity["native_adapter_options"]["snow_fresh_albedo"] == "masked"
    assert masked.config_hash != native.config_hash
    with pytest.raises(ValueError, match="snow_fresh_albedo must be"):
        NativePhysicsOptions.from_mapping({**_options(), "snow_fresh_albedo": "bright"})


def test_the_runtime_hands_the_kernel_the_rule_only_when_it_is_not_wrfs():
    from types import SimpleNamespace

    rt = SimpleNamespace(options=SimpleNamespace(snow_age_refresh="any", snow_fresh_albedo="livneh"))
    from arwen_global.physics import native_runtime as nr

    cls = [v for v in vars(nr).values() if isinstance(v, type) and hasattr(v, "_snow_age_refresh_kwargs")][0]
    assert cls._snow_age_refresh_kwargs(rt) == {}
    rt.options.snow_fresh_albedo = "masked"
    assert cls._snow_age_refresh_kwargs(rt) == {"snow_fresh_albedo": "masked"}


@pytest.mark.parametrize("fresh", ["livneh", "masked"])
def test_thaw_ages_a_melting_pack_on_the_ablation_curve_and_leaves_a_cold_one(fresh):
    from arwen_global.core.npref import _noah_alcalc

    alb, snoalb, age = 0.2, 0.7, 5 * 86400.0
    kw = dict(fresh=fresh)
    cold_off, _, _ = _noah_alcalc(alb, snoalb, 0.95, 1.0, 0.0, False, age, 0.5, **kw)
    cold_on, _, _ = _noah_alcalc(alb, snoalb, 0.95, 1.0, 0.0, False, age, 0.5,
                                 thaw=True, tsnow=260.0, **kw)
    assert cold_on == cold_off
    warm, _, _ = _noah_alcalc(alb, snoalb, 0.95, 1.0, 0.0, False, age, 0.5,
                              thaw=True, tsnow=273.16, **kw)
    dec = 0.82 ** (5.0 ** 0.46)
    if fresh == "livneh":
        want = (snoalb + 0.5 * (0.85 - snoalb)) * dec
    else:
        want = alb + (snoalb - alb) / (0.85 - alb) * (0.85 * dec - alb)
    assert warm == pytest.approx(want, abs=1e-12)
    assert warm < cold_off
