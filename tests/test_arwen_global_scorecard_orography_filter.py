"""The scorecards rebuild the surface geopotential the run integrated with,
including the run's ``[initial] orography_filter_half_degree``.

Before this fix ``surface_geopotential_for_run`` rebuilt the terrain
unfiltered for every run.  The shipped T533, T799 and T1023 configs filter
at half degree 350, so their model 500 hPa heights were integrated up from
the wrong surface and every height score carried the filter's residual as
model error (MEASURED 2026-11-22 on GDAS 2026-11-10 00Z: 26.8 m RMS over the
globe and 34.8 m over 20N-90N at T533 on the scorecard's own grid; see
``surface_geopotential_from_frame``).  An unfiltered run's rebuild, and its
cache file name, are unchanged byte for byte.
"""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global import upper_air_scorecard as sc
from arwen_global.spectral.orography_filter import orography_filter_factor
from arwen_global.spectral.transform import SphericalHarmonicTransform

TRUNCATION = 21
HALF_DEGREE = 8.0


@pytest.fixture(scope="module")
def transform():
    return SphericalHarmonicTransform.create(TRUNCATION, backend="numpy", precision="float64")


def _terrain_frame():
    lat_src = np.arange(90.0, -90.001, -1.0)
    lon_src = np.arange(0.0, 360.0, 1.0)
    lon_s, lat_s = np.meshgrid(lon_src, lat_src)
    lat_r, lon_r = np.radians(lat_s), np.radians(lon_s)
    # Planetary shape plus small-scale ridges, so the filter at degree 8 has
    # something to remove and keeps the large scales.
    terrain = (800.0 * (1.0 + np.sin(lat_r) * np.sin(lon_r))
               + 600.0 * np.cos(lat_r) ** 2 * np.cos(9.0 * lon_r) * np.sin(7.0 * lat_r) ** 2
               + 300.0 * np.cos(lat_r) * np.cos(15.0 * lon_r))
    levels = np.asarray(sc.LEVELS_PA)
    stack = np.zeros((levels.size, *terrain.shape))
    return sc.SyntheticFrame(lat_src, lon_src, levels, {
        "geopotential_height": stack, "air_temperature": stack + 250.0,
        "specific_humidity": stack + 0.001, "eastward_wind": stack, "northward_wind": stack,
        "surface_pressure": np.full(terrain.shape, 101_000.0), "terrain_height": terrain,
    })


def _receipt(transform, half_degree):
    config = {"truncation": TRUNCATION, "dealias_factor": 1.5}
    if half_degree:
        config["orography_filter_half_degree"] = half_degree
    return {
        "config": config,
        "transform": {"truncation": TRUNCATION, "nlat": int(transform.grid.nlat),
                      "nlon": int(transform.grid.nlon), "radius_m": 6_371_220.0},
        "initial": {"provenance": {"mode": "analysis", "mapping": "m.json",
                                   "input_sha256": {"start.grib2": "cd" * 32}}},
    }


def test_off_is_the_unfiltered_rebuild_bit_for_bit(transform):
    frame = _terrain_frame()
    from arwen_global.analysis_initial import _global_regridder
    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    phi_src = sc.GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    old = np.asarray(transform.inverse(transform.forward(phi_src)), dtype=np.float64)
    assert np.array_equal(sc.surface_geopotential_from_frame(frame, transform), old)
    assert np.array_equal(sc.surface_geopotential_from_frame(frame, transform, 0.0), old)


def test_the_filtered_rebuild_carries_the_filter_factor_degree_by_degree(transform):
    frame = _terrain_frame()
    plain = sc.surface_geopotential_from_frame(frame, transform)
    filtered = sc.surface_geopotential_from_frame(frame, transform, HALF_DEGREE)
    c_plain = np.asarray(transform.forward(plain))
    c_filt = np.asarray(transform.forward(filtered))
    factor = orography_filter_factor(TRUNCATION, HALF_DEGREE)
    scale = float(np.max(np.abs(c_plain)))
    assert np.max(np.abs(c_filt - c_plain * factor[:, None])) < 1.0e-9 * scale
    # The filter removed something real, and kept the global mean.
    assert float(np.sqrt(np.mean((filtered - plain) ** 2))) > 10.0 * sc.GRAVITY_M_S2 * 0.1
    assert abs(c_filt[0, 0] - c_plain[0, 0]) < 1.0e-12 * scale


def test_the_run_rebuild_reads_the_filter_from_the_receipt(transform, tmp_path, monkeypatch):
    """The regression: a filtered run's rebuild is the filtered terrain."""
    frame = _terrain_frame()
    start = tmp_path / "start.grib2"
    start.write_bytes(b"x")
    monkeypatch.setattr(sc, "_sha256_file", lambda path: "cd" * 32)
    monkeypatch.setattr(sc, "decode_reference", lambda mapping, path: (frame, {"stub": True}))
    receipt = _receipt(transform, HALF_DEGREE)
    phi, provenance = sc.surface_geopotential_for_run(receipt, transform, start, tmp_path / "m.json",
                                                      cache_dir=tmp_path / "cache")
    expected = sc.surface_geopotential_from_frame(frame, transform, HALF_DEGREE)
    assert np.array_equal(phi, expected)
    assert provenance["orography_filter_half_degree"] == HALF_DEGREE
    assert "orography filter at half degree 8" in provenance["construction"]
    # The cache holds the filtered field and is read back as such.
    again, prov2 = sc.surface_geopotential_for_run(receipt, transform, start, tmp_path / "m.json",
                                                   cache_dir=tmp_path / "cache")
    assert np.array_equal(again, expected) and prov2["cache"].endswith("-orogf8.npz")
    # An unfiltered receipt from the same analysis does not read that cache.
    plain_receipt = _receipt(transform, 0.0)
    plain, prov3 = sc.surface_geopotential_for_run(plain_receipt, transform, start, tmp_path / "m.json",
                                                   cache_dir=tmp_path / "cache")
    assert np.array_equal(plain, sc.surface_geopotential_from_frame(frame, transform))
    assert not np.array_equal(plain, expected)
    assert "orogf" not in prov3["cache"]


def test_cache_names_split_filtered_and_unfiltered_and_keep_the_old_name(transform):
    digest = "ab" * 32
    plain = sc.surface_geopotential_cache_name(digest, _receipt(transform, 0.0))
    filt = sc.surface_geopotential_cache_name(digest, _receipt(transform, 350.0))
    shape = f"{int(transform.grid.nlat)}x{int(transform.grid.nlon)}"
    assert plain == f"phi-surface-{digest[:16]}-T{TRUNCATION}-{shape}.npz"
    assert filt == f"phi-surface-{digest[:16]}-T{TRUNCATION}-{shape}-orogf350.npz"


def test_the_observation_scorer_reads_the_runs_own_surface_through_its_run_door(
        transform, tmp_path, monkeypatch):
    """The door `gpuwm-global score` takes (obs_scorecard._phi_for, from the
    run directory): a filtered run is scored on its filtered surface and an
    unfiltered run of the same analysis on the unfiltered one, both when the
    rebuild is made and when the cache the upper-air scorecard shares is read."""
    from pathlib import Path

    from arwen_global import obs_scorecard as obs

    frame = _terrain_frame()
    start = tmp_path / "start.grib2"
    start.write_bytes(b"x")
    for module in (sc, obs):
        monkeypatch.setattr(module, "_sha256_file", lambda path: "cd" * 32)
    monkeypatch.setattr(sc, "decode_reference", lambda mapping, path: (frame, {"stub": True}))
    receipts = {"filtered": _receipt(transform, HALF_DEGREE), "plain": _receipt(transform, 0.0)}
    monkeypatch.setattr(obs, "read_receipt", lambda run_dir, door=None: receipts[Path(run_dir).name])
    monkeypatch.setattr(obs, "resolve_mapping", lambda spec, receipt: tmp_path / "m.json")
    monkeypatch.setattr(obs, "build_transform_from_receipt", lambda receipt: transform)
    filtered = sc.surface_geopotential_from_frame(frame, transform, HALF_DEGREE)
    plain = sc.surface_geopotential_from_frame(frame, transform)
    assert not np.array_equal(filtered, plain)
    cache = tmp_path / "cache"
    for attempt in ("rebuilt", "read from the cache"):
        phi, provenance, _mapping = obs._phi_for(tmp_path / "filtered", None, cache, start)
        assert np.array_equal(phi, filtered), attempt
        assert provenance["orography_filter_half_degree"] == HALF_DEGREE
        phi, provenance, _mapping = obs._phi_for(tmp_path / "plain", None, cache, start)
        assert np.array_equal(phi, plain), attempt
    assert len(list(cache.glob("*.npz"))) == 2
