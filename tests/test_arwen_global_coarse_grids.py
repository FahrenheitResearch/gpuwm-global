"""The T63 and T127 coarse grids are the T255 config of record, re-gridded.

``arwen_global_gdas_t63_native_sl_si_24h`` and
``arwen_global_gdas_t127_native_sl_si_24h`` promise in their comments that
they differ from ``arwen_global_gdas_t255_native_sl_si_24h`` in the grid
and in nothing a reader has to diff.  These tests hold them to it: a
change to the T255 config that is not carried to the coarse grids, or an
edit to a coarse grid that is not a grid edit, fails here by key.

One declared exception: from 0.2.3 the T255 forecast day turns the
orographic drag on (``orographic_drag = "kim-gwdo"``), measured on the T255
forecast scorecard; the coarse grids are experiments, not forecasts, and
keep 0.2.2's physics and hash.
"""
from __future__ import annotations

import math
import tomllib

import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root, resolve_config

T255 = "arwen_global_gdas_t255_native_sl_si_24h"
#: The keys the T255 record carries and the coarse grids do not (0.2.3).
RECORD_ONLY = {"physics.native_adapter_options.orographic_drag"}
COARSE = {
    "arwen_global_gdas_t63_native_sl_si_24h": 63,
    "arwen_global_gdas_t127_native_sl_si_24h": 127,
}
EARTH_RADIUS_M = 6_371_220.0


def _toml(name: str) -> dict:
    with (config_root() / f"{name}.toml").open("rb") as stream:
        return tomllib.load(stream)


def _flatten(table: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in table.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _equatorial_dx_m(truncation: int) -> float:
    # dealias 1.5 quadrature: 3 (T + 1) longitudes (768 at T255, 192 at T63),
    # the spacing rounded to 100 m as the T255 config rounds 52,122 m.
    nlon = 3 * (truncation + 1)
    return round(2.0 * math.pi * EARTH_RADIUS_M / nlon / 100.0) * 100.0


@pytest.mark.parametrize("name,truncation", sorted(COARSE.items()))
def test_a_coarse_grid_differs_from_the_t255_record_only_in_its_grid(name, truncation):
    record = _flatten(_toml(T255))
    coarse = _flatten(_toml(name))
    assert set(coarse) == set(record) - RECORD_ONLY, sorted(set(coarse) ^ set(record))
    assert record["physics.native_adapter_options.orographic_drag"] == "kim-gwdo"
    differing = {key for key in coarse if record[key] != coarse[key]}
    assert differing == {"arwen_global.name", "grid.truncation",
                         "physics.native_adapter_options.dx_m"}, sorted(differing)
    assert coarse["grid.truncation"] == truncation
    assert coarse["arwen_global.name"] == name.replace("_", "-")
    assert coarse["physics.native_adapter_options.dx_m"] == _equatorial_dx_m(truncation)


def test_the_t255_record_reads_its_own_dx_by_the_same_rule():
    record = _flatten(_toml(T255))
    assert record["physics.native_adapter_options.dx_m"] == _equatorial_dx_m(255)


@pytest.mark.parametrize("name,truncation", sorted(COARSE.items()))
def test_a_coarse_grid_resolves_by_bare_name_and_loads(name, truncation):
    path = resolve_config(name)
    assert path == config_root() / f"{name}.toml"
    config = load_config(path)
    assert config.truncation == truncation
    assert config.integrator == "sl_si"
    assert config.backend == "cupy" and config.precision == "float32"
    assert config.dt_s == 300.0 and config.duration_s == 86400.0
    record = load_config(config_root() / f"{T255}.toml")
    # The same column, the same case, the same physics: only the grid moved.
    assert config.a_half_pa == record.a_half_pa
    assert config.analysis_grib == record.analysis_grib


@pytest.mark.parametrize("name", sorted(COARSE))
def test_a_coarse_grid_says_it_is_experimental_and_not_a_forecast(name):
    text = (config_root() / f"{name}.toml").read_text(encoding="utf-8")
    assert "EXPERIMENTAL" in text
    assert "not a forecast configuration" in text
