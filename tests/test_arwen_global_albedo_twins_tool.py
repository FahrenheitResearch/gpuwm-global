"""The albedo twins reader: CERES month means, the 2 m terrain carry and the
table's control differences, on planted inputs (CPU, no files from a run)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_albedo_twins_tool", TREE / "tools" / "arwen_global_albedo_twins.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ceres_reference_reads_planted_albedo(tool):
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    rsdt = np.full((3, lat.size, lon.size), 340.0)
    # Albedo 0.2 in the southern hemisphere, 0.4 in the northern.
    alb = np.where(lat[:, None] < 0, 0.2, 0.4) * np.ones((1, lon.size))
    fields = {"rsdt": rsdt, "rsut": rsdt * alb[None], "rlut": np.full_like(rsdt, 240.0)}
    ref = tool.ceres_reference(fields, lat, lon, {}, 11)
    assert ref["global"]["albedo"] == pytest.approx(0.3, abs=1e-12)
    assert ref["global"]["net"] == pytest.approx(340.0 - 102.0 - 240.0, abs=1e-9)
    assert ref["bands"]["60S-30S"]["albedo"] == pytest.approx(0.2, abs=1e-12)
    assert ref["bands"]["60N-90N"]["albedo"] == pytest.approx(0.4, abs=1e-12)
    assert set(ref["bands"]) == {"90S-60S", "60S-30S", "30S-0", "0-30N", "30N-60N", "60N-90N"}


def test_lapse_adjust_carries_reference_to_model_terrain(tool):
    # A reference point 1000 m lower than the model's terrain is 6.5 K warmer there.
    assert tool.lapse_adjust(290.0, 0.0, 1000.0) == pytest.approx(283.5)
    assert tool.lapse_adjust(290.0, 500.0, 500.0) == pytest.approx(290.0)


def test_masked_stats_weights_and_masks(tool):
    diff = np.array([[1.0, -1.0], [3.0, np.nan]])
    w = np.array([[1.0, 1.0], [2.0, 2.0]])
    s = tool.masked_stats(diff, w, np.ones_like(diff, bool))
    assert s["n"] == 3
    assert s["bias"] == pytest.approx((1 - 1 + 6) / 4.0)
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 1 + 18) / 4.0))
    assert tool.masked_stats(diff, w, np.zeros_like(diff, bool))["rmse"] is None


def _rad(up, olr):
    regions = ("global", "land", "ocean", "90S-60S", "60S-30S", "30S-0", "0-30N", "30N-60N", "60N-90N")
    flux = lambda v: {r: float(v) for r in regions}  # noqa: E731
    return {"window": {"hours": [24.0, 48.0], "fluxes": {
        "sw_up_top": flux(up), "sw_down_top": flux(340.0), "lw_up_top": flux(olr),
        "net_top": flux(340.0 - up - olr), "total_cloud_cover": flux(0.7), "sw_net_surface": flux(160.0)}},
        "checkpoints": [], "regions": list(regions)}


def test_table_differences_against_control(tool, tmp_path):
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    rsdt = np.full((1, lat.size, lon.size), 340.0)
    ref = tool.ceres_reference({"rsdt": rsdt, "rsut": rsdt * 0.3}, lat, lon, {}, 11)
    (tmp_path / "ceres.json").write_text(__import__("json").dumps(ref))
    for arm, up in (("off", 136.0), ("on", 112.2)):
        (tmp_path / f"rad-{arm}.json").write_text(__import__("json").dumps(_rad(up, 236.0)))
    manifest = {"cases": {"c": {"ceres": str(tmp_path / "ceres.json"), "control": "off", "arms": {
        a: {"radiation": str(tmp_path / f"rad-{a}.json")} for a in ("off", "on")}}}}
    table = tool.build_table(manifest)
    on = table["cases"]["c"]["arms"]["on"]
    assert on["albedo_global"] == pytest.approx(0.33)
    assert on["albedo_minus_ceres_global"] == pytest.approx(0.03)
    assert on["excess_rsut_vs_ceres_w_m2"] == pytest.approx(10.2)
    assert on["delta"]["rsut_global"] == pytest.approx(-23.8)
    assert table["cases"]["c"]["arms"]["off"]["delta"]["albedo_global"] == 0.0
    summary = table["summary"]["on"]
    assert summary["cases"] == ["c"]
    assert summary["rsut_global"]["delta_mean"] == pytest.approx(-23.8)
    # The band error falls (0.4 -> 0.33 against 0.3 everywhere): no case worse.
    assert summary["band_albedo_abs_error_mean"]["cases_worse"] == 0
    md = tool.markdown(table)
    assert "| on |" in md and "Mean over cases" in md
