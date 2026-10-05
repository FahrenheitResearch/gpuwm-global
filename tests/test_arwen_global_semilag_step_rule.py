"""The omitted semi-Lagrangian step is 300 s at every truncation.

A truncation-scaled step (225 s at T533) was the remedy for the 0.75
trajectory norm gate's refusal of a T533 forecast at 66.3 h.  That gate is
retired (DYC-2: the fold gate refuses folds, not shear), and the guard
installed for it is retired with it: under the fold gate the T533 300 s
forecast from GDAS 2026-09-25 12Z completes 120 h with every gate passing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arwen_global.config import DEFAULT_SEMILAG_STEP_S, load_config
from arwen_global.configs_dir import config_root

pytestmark = pytest.mark.cpu_only


MINIMAL = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "step-rule"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"

[grid]
truncation = {truncation}

[time]
duration_s = 86400.0
output_interval_s = 3600.0
"""


@pytest.mark.parametrize("truncation", [63, 255, 383, 533, 799])
def test_an_omitted_step_is_the_cores_300_s_at_every_truncation(
        tmp_path, truncation):
    path = tmp_path / "config.toml"
    path.write_text(MINIMAL.format(truncation=truncation), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.integrator == "sl_si"
    assert cfg.dt_s == DEFAULT_SEMILAG_STEP_S == 300.0


@pytest.mark.parametrize("name", [
    "arwen_global_gdas_t533_native_24h",
    "arwen_global_gdas_t533_native_sl_si_24h",
    "arwen_global_gdas_t533_native_sl_si_wall",
])
def test_the_t533_presets_run_the_300_s_step_and_its_buckets(name):
    cfg = load_config(config_root() / f"{name}.toml")
    assert cfg.dt_s == 300.0
    assert cfg.native_adapter_options["land_surface_interval_s"] == 300.0
    assert cfg.native_adapter_options["radiation_interval_s"] == 1500.0
