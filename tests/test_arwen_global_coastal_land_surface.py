"""A coastal land column's soil and skin come from the analysis's own land.

The breakage (audit 2026-11-15, T255 from GDAS 2026-09-01 00Z): the cold
start filled the GRIB soil over water with the skin (the sea surface
temperature there) and the soil water with 0.25, then regridded plainly,
so 4,040 land columns with water in their stencil started with the sea
blended into their soil (1-2 m soil mean +1.60 K, range -7.9..+15.1 K) and
77 land columns with no analysis land in their stencil started with soil =
SST and soil water 0.25.  WPS reads these over the land-sea mask
(METGRID.TBL.ARW: ST*/SM* masked=water, interp_mask=LANDSEA(0); SKINTEMP
masked=both); ``land_only_regrid`` is that rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global.analysis_initial import (
    _global_regridder,
    _TargetGrid,
    land_only_regrid,
)

#: A target column between four 0.25-degree analysis points.
TARGET_LAT, TARGET_LON = 40.125, 10.125
LAND_SOIL_K, SEA_SKIN_K = 285.0, 300.0


def _source(land_points, *, land_everywhere=False):
    """A global 0.25-degree analysis: water at SEA_SKIN_K with the soil
    masked (NaN) except on ``land_points``, where the soil is LAND_SOIL_K
    and the soil water 0.30."""
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(1440) * 0.25
    land = np.full((lat.size, lon.size), 1.0 if land_everywhere else 0.0)
    for la, lo in land_points:
        land[int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))] = 1.0
    skin = np.where(land >= 0.5, LAND_SOIL_K, SEA_SKIN_K)
    soil_t = np.where(land >= 0.5, LAND_SOIL_K, np.nan)[None] * np.ones((4, 1, 1))
    soil_w = np.where(land >= 0.5, 0.30, np.nan)[None] * np.ones((4, 1, 1))
    return lat, lon, land, skin, soil_t, soil_w


def _regrid(lat, lon):
    return _global_regridder(
        lat, lon, _TargetGrid(np.asarray([TARGET_LAT]), np.asarray([TARGET_LON])))


def _axes(lat, lon):
    return {"source_latitude_deg": lat, "source_longitude_deg": lon,
            "target_latitude_deg": np.asarray([TARGET_LAT]),
            "target_longitude_deg": np.asarray([TARGET_LON])}


def _old_rule(regrid, soil_t, soil_w, skin):
    """analysis_initial.py:1407-1415 at rel/0.2.5 e9e202a."""
    t = regrid(np.where(np.isfinite(soil_t), soil_t, skin[None]))
    w = np.clip(regrid(np.where(np.isfinite(soil_w), soil_w, 0.25)), 0.0, 1.0)
    return t, w


def test_a_coastal_stencil_blended_the_sea_into_the_soil_and_now_does_not():
    # Two of the four stencil points are land, two are sea.
    lat, lon, land, skin, soil_t, soil_w = _source(
        [(40.25, 10.0), (40.25, 10.25)])
    regrid = _regrid(lat, lon)
    target = np.ones((1, 1), dtype=bool)
    old_t, old_w = _old_rule(regrid, soil_t, soil_w, skin)
    # The defect: half sea surface temperature, half soil.
    assert old_t[0, 0, 0] == pytest.approx(0.5 * (LAND_SOIL_K + SEA_SKIN_K), abs=1e-6)
    assert old_w[0, 0, 0] == pytest.approx(0.275, abs=1e-6)
    new_t, weight, searched, unreached = land_only_regrid(soil_t, land, regrid, target, **_axes(lat, lon))
    new_w, *_ = land_only_regrid(soil_w, land, regrid, target, **_axes(lat, lon))
    new_skin, *_ = land_only_regrid(skin, land, regrid, target, **_axes(lat, lon))
    assert np.allclose(new_t, LAND_SOIL_K, rtol=0.0, atol=1e-9)
    assert np.allclose(new_w, 0.30, rtol=0.0, atol=1e-12)
    assert new_skin[0, 0] == pytest.approx(LAND_SOIL_K, abs=1e-9)
    assert weight[0, 0, 0] == pytest.approx(0.5, abs=1e-9)
    assert not searched.any() and not unreached.any()


def test_a_land_column_with_no_land_in_its_stencil_reads_the_nearest_land():
    # The nearest analysis land is two cells north of the stencil, a colder
    # island farther away: the old rule started this land column with
    # soil = SST and soil water 0.25.
    lat, lon, land, skin, soil_t, soil_w = _source([(40.75, 10.0), (42.0, 10.0)])
    far = (int(np.argmin(np.abs(lat - 42.0))), int(np.argmin(np.abs(lon - 10.0))))
    soil_t[:, far[0], far[1]] = 270.0
    soil_w[:, far[0], far[1]] = 0.10
    regrid = _regrid(lat, lon)
    target = np.ones((1, 1), dtype=bool)
    old_t, old_w = _old_rule(regrid, soil_t, soil_w, skin)
    assert np.allclose(old_t, SEA_SKIN_K) and np.allclose(old_w, 0.25)
    new_t, weight, searched, unreached = land_only_regrid(soil_t, land, regrid, target, **_axes(lat, lon))
    new_w, *_ = land_only_regrid(soil_w, land, regrid, target, **_axes(lat, lon))
    assert (weight == 0.0).all()
    assert searched.all() and not unreached.any()
    assert np.allclose(new_t, LAND_SOIL_K, rtol=0.0, atol=1e-9)
    assert np.allclose(new_w, 0.30, rtol=0.0, atol=1e-12)


def test_water_columns_and_a_world_without_land_are_left_to_the_caller():
    lat, lon, land, skin, soil_t, soil_w = _source([(40.25, 10.0)])
    regrid = _regrid(lat, lon)
    water, *_ = land_only_regrid(soil_t, land, regrid, np.zeros((1, 1), dtype=bool), **_axes(lat, lon))
    assert np.isnan(water).all()
    # No analysis land anywhere: nothing to read, the caller keeps its fill.
    lat, lon, land, skin, soil_t, soil_w = _source([])
    plane, _w, searched, unreached = land_only_regrid(
        soil_t, land, regrid, np.ones((1, 1), dtype=bool), **_axes(lat, lon))
    assert np.isnan(plane).all() and searched.all() and unreached.all()


def test_a_finite_soil_value_on_analysis_water_is_not_land():
    # GDAS TSOIL is finite over sea ice: the land is the land_fraction
    # split, not the bitmap.
    lat, lon, land, skin, soil_t, soil_w = _source(
        [(40.25, 10.0), (40.25, 10.25)])
    soil_t = np.where(np.isfinite(soil_t), soil_t, 271.0)
    new_t, *_ = land_only_regrid(soil_t, land, _regrid(lat, lon), np.ones((1, 1), dtype=bool),
                                **_axes(lat, lon))
    assert np.allclose(new_t, LAND_SOIL_K, rtol=0.0, atol=1e-9)
