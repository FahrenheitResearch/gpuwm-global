"""`[initial] analysis_snow_density = "wrf-floor"`: a snow pack the analysis
starts lighter than WRF real.exe's derived density (SNOWH = SNOW * 0.005,
200 kg m-3) starts at that depth instead; a denser pack and the default
are untouched, and the key joins the config identity only when set."""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global import surface_seeding as seeding
from arwen_global.analysis_initial import _global_regridder
from arwen_global.config import load_config
from arwen_global.spectral.grid import GaussianGrid

from test_arwen_global_analysis_initial import _analysis_cfg


def _seed(frame, **kw):
    grid = GaussianGrid.create(21)
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)
    land = regrid(frame.fields["land_fraction"].values)
    return seeding.seed_surface_from_analysis(frame, regrid, target_land_fraction=land, **kw)


def test_the_gdas_ten_to_one_pack_starts_at_the_wrf_derived_depth():
    # GDAS writes depth = 10 x SWE: 60 kg/m2 over 0.60 m, 100 kg/m3.
    frame = seeding.synthetic_analysis(swe_kg_m2=60.0, snow_depth_m=0.60)
    plain = _seed(frame)
    floored = _seed(frame, snow_density_floor=True)
    snowy = plain.snow_cover > 0.5
    assert snowy.any()
    # Default: the analysis depth, bit for bit, and the receipt says nothing new.
    assert "density_floor" not in plain.provenance["seeded_from_analysis"]["snow_depth_m"]
    # Floor: half the depth (200 kg/m3), the same water, the same cover.
    assert np.allclose(floored.snow_depth_m[snowy],
                       floored.snow_water_kg_m2[snowy] / seeding.DERIVED_SNOW_DENSITY_KG_M3)
    assert np.array_equal(floored.snow_water_kg_m2, plain.snow_water_kg_m2)
    assert np.array_equal(floored.snow_cover, plain.snow_cover)
    assert np.all(floored.snow_depth_m <= plain.snow_depth_m)
    record = floored.provenance["seeded_from_analysis"]["snow_depth_m"]["density_floor"]
    assert record["floor_kg_m3"] == 200.0 and record["floored_source_cells"] > 0


def test_a_pack_denser_than_the_floor_keeps_its_analysis_depth():
    # 60 kg/m2 over 0.20 m is 300 kg/m3, already denser than 200.
    frame = seeding.synthetic_analysis(swe_kg_m2=60.0, snow_depth_m=0.20)
    plain = _seed(frame)
    floored = _seed(frame, snow_density_floor=True)
    assert np.array_equal(floored.snow_depth_m, plain.snow_depth_m)
    assert floored.provenance["seeded_from_analysis"]["snow_depth_m"]["density_floor"][
        "floored_source_cells"] == 0


def test_the_key_joins_the_identity_only_when_set_and_is_checked(tmp_path):
    cfg = _analysis_cfg(tmp_path)
    assert cfg.analysis_snow_density == "analysis"
    assert "analysis_snow_density" not in cfg.config_identity
    text = (tmp_path / "analysis.toml").read_text(encoding="utf-8")
    for value, joins in (("analysis", False), ("wrf-floor", True)):
        path = tmp_path / f"{value}.toml"
        path.write_text(text.replace(
            "[initial]\n", f"[initial]\nanalysis_snow_density = \"{value}\"\n", 1), encoding="utf-8")
        other = load_config(path)
        assert other.analysis_snow_density == value
        assert ("analysis_snow_density" in other.config_identity) is joins
        assert (other.config_hash != cfg.config_hash) is joins
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("[initial]\n", "[initial]\nanalysis_snow_density = \"era5\"\n", 1),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="analysis_snow_density must be"):
        load_config(bad)


def test_the_resolved_plan_names_the_floor_only_when_it_is_on(tmp_path):
    # The plan document a reviewer reads before a run must say the cold
    # start is changed; a default plan document keeps its bytes.
    from arwen_global.runplan import _config_snapshot

    cfg = _analysis_cfg(tmp_path)
    assert "analysis_snow_density" not in _config_snapshot(cfg, tmp_path / "analysis.toml")["initial"]
    text = (tmp_path / "analysis.toml").read_text(encoding="utf-8")
    path = tmp_path / "floor.toml"
    path.write_text(text.replace(
        "[initial]\n", "[initial]\nanalysis_snow_density = \"wrf-floor\"\n", 1), encoding="utf-8")
    snap = _config_snapshot(load_config(path), path)
    assert snap["initial"]["analysis_snow_density"] == "wrf-floor"
