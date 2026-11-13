"""Terrain for the dynamics of an invented world (arwen_global.world.terrain).

The gates, each on the tiny CPU configuration:

*   grid averaging keeps the map's area mean to rounding, maps a constant
    to the same constant, and reads a grid-registered map (its repeated
    360-degree column, its pole rows) and a cell-centred one alike;
*   the filter never moves degree 0, ``kind="none"`` is the plain
    truncation, and on a sharp-edged continent the default filter leaves
    shallower Gibbs pits over water and on land than the plain truncation;
*   the frame sampler reproduces the transform's own synthesis on the
    Gaussian grid and the package's own point evaluator on a regular one;
*   the rest test: an atmosphere at rest over flat ground stays at rest to
    rounding, and over the 2,000 m ring mountain the largest and the rms
    wind of the first six hours stay under the bounds measured for them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.constants import GRAVITY_M_S2  # noqa: E402
from arwen_global.dynamics import MoistHybridModel  # noqa: E402
from arwen_global.semi_implicit import VerticalModeSemiImplicit  # noqa: E402
from arwen_global.semilag import SEMILAG_INTEGRATORS, SemiLagrangianOptions  # noqa: E402
from arwen_global.spectral.sampling import sample_scalar  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402
from arwen_global.world import parse_world  # noqa: E402
from arwen_global.world.terrain import (  # noqa: E402
    DEFAULT_FILTER, NO_FILTER, RING_HEIGHT_M, TerrainFilter, average_to_grid,
    average_to_latlon, dynamics_terrain, dynamics_terrain_for_world,
    frame_planes, land_height, map_area_mean,
    rest_surface_pressure, rest_temperature, rest_test, resting_state,
    ring_mountain, sample_on_latlon,
)

from test_arwen_global_vertical_modes import _surface  # noqa: E402


def _transform(truncation=21):
    return SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64")


def _paleomap_axes(step=4.0):
    """A grid-registered map like the PALEOMAP files: -180..180 with the
    first column repeated, -90..90 with both pole rows."""
    lat = np.arange(-90.0, 90.0 + 0.5 * step, step)
    lon = np.arange(-180.0, 180.0 + 0.5 * step, step)
    return lat, lon


def _continent(lat, lon, height=2000.0, depth=-4000.0):
    """One sharp-edged rectangular continent in a deep sea."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    inside = (np.abs(la - 10.0) <= 40.0) & (np.abs(lo - 20.0) <= 50.0)
    return np.where(inside, height, depth)


# ---------------------------------------------------------- averaging

def test_grid_averaging_keeps_the_map_area_mean():
    transform = _transform(42)
    rng = np.random.default_rng(7)
    lat, lon = _paleomap_axes(3.0)
    values = rng.normal(500.0, 300.0, (lat.size, lon.size))
    values[:, -1] = values[:, 0]          # the repeated column is one place
    averaged = average_to_grid(values, lat, lon, transform.grid)
    target = map_area_mean(values, lat, lon)
    assert transform.grid.global_mean(averaged) == pytest.approx(
        target, rel=1e-12, abs=1e-9)


def test_grid_averaging_maps_a_constant_to_itself_from_either_registration():
    transform = _transform(21)
    lat, lon = _paleomap_axes(2.0)
    flat = np.full((lat.size, lon.size), 123.5)
    out = average_to_grid(flat, lat, lon, transform.grid)
    assert np.max(np.abs(out - 123.5)) < 1e-10
    # Cell-centred, descending latitudes, longitudes from 0: the same.
    clat = np.arange(89.5, -90.0, -1.0)
    clon = np.arange(0.5, 360.0, 1.0)
    out = average_to_grid(np.full((clat.size, clon.size), -7.0), clat, clon,
                          transform.grid)
    assert np.max(np.abs(out + 7.0)) < 1e-10


def test_grid_averaging_refuses_a_partial_map():
    transform = _transform(21)
    lat = np.arange(-60.0, 60.5, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    with pytest.raises(ValueError, match="reach both poles"):
        average_to_grid(np.zeros((lat.size, lon.size)), lat, lon,
                        transform.grid)
    lat = np.arange(-90.0, 90.5, 1.0)
    lon = np.arange(0.0, 180.0, 1.0)
    with pytest.raises(ValueError, match="full ring"):
        average_to_grid(np.zeros((lat.size, lon.size)), lat, lon,
                        transform.grid)
    lon = np.arange(0.0, 360.0, 1.0)
    bad = np.zeros((lat.size, lon.size))
    bad[3, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        average_to_grid(bad, lat, lon, transform.grid)


# ------------------------------------------------------------- filter

def test_the_filter_never_moves_degree_zero_and_none_is_the_truncation():
    for terrain_filter in (DEFAULT_FILTER, NO_FILTER,
                           TerrainFilter(cutoff_fraction=0.3, order=1)):
        factors = terrain_filter.factors(255)
        assert factors[0] == 1.0
        assert np.all(np.diff(factors) <= 0.0)
    assert np.all(NO_FILTER.factors(42) == 1.0)
    transform = _transform(21)
    lat, lon = _paleomap_axes(2.0)
    elevation = _continent(lat, lon)
    sharp = dynamics_terrain(elevation, lat, lon, transform,
                             terrain_filter=NO_FILTER)
    plain = transform.inverse(transform.forward(sharp.averaged_m))
    assert np.max(np.abs(sharp.height_m - plain)) < 1e-9


def test_the_filter_refuses_settings_by_name():
    with pytest.raises(ValueError, match="kind"):
        TerrainFilter(kind="lanczos")
    with pytest.raises(ValueError, match="cutoff_fraction"):
        TerrainFilter(cutoff_fraction=0.0)
    with pytest.raises(ValueError, match="order"):
        TerrainFilter(order=2.5)


def test_the_default_filter_leaves_less_false_height_than_a_sharp_cut():
    """A 2,000 m rectangle continent in a 4,000 m deep sea, at T42.  The
    sea floor never reaches the dynamics; the mean height is the map's
    own; the default filter's largest false height over open water is
    below the sharp truncation's."""
    transform = _transform(42)
    lat, lon = _paleomap_axes(1.0)
    elevation = _continent(lat, lon)
    sharp = dynamics_terrain(elevation, lat, lon, transform,
                             terrain_filter=NO_FILTER)
    smooth = dynamics_terrain(elevation, lat, lon, transform)
    for built in (sharp, smooth):
        report = built.report
        assert report.averaged_mean_m == pytest.approx(report.map_mean_m,
                                                       rel=1e-12)
        assert report.model_mean_m == pytest.approx(report.map_mean_m,
                                                     rel=1e-9)
        assert report.map_peak_m == 2000.0
        assert float(np.min(built.averaged_m)) >= 0.0
        assert 0.0 < report.land_share < 1.0
        assert report.open_water_cells > 0
        assert "largest false height over open water" in report.text()
    # The Gibbs pits, over water and on land, are what the filter is for;
    # the largest height over water sits beside the coast, where a cliff
    # the grid cannot hold spreads into the first sea cells under either.
    assert smooth.report.lowest_over_water_m > sharp.report.lowest_over_water_m
    assert smooth.report.lowest_over_water_m < 0.0
    assert smooth.report.model_peak_m < sharp.report.model_peak_m
    assert smooth.report.model_peak_m > 0.9 * smooth.report.map_peak_m


def test_the_world_entry_reads_its_sea_level_and_refuses_earth():
    transform = _transform(21)
    lat, lon = _paleomap_axes(2.0)
    elevation = _continent(lat, lon)
    with pytest.raises(ValueError, match="Earth"):
        dynamics_terrain_for_world(parse_world(None), elevation, lat, lon,
                                   transform)
    risen = parse_world({"label": "flood", "sea": {"level_m": 100.0}})
    built = dynamics_terrain_for_world(risen, elevation, lat, lon, transform)
    assert built.report.map_peak_m == 1900.0


# -------------------------------------------------------------- frame

def test_the_frame_sampler_matches_the_transform_and_the_point_evaluator():
    transform = _transform(21)
    lat, lon = _paleomap_axes(2.0)
    built = dynamics_terrain(_continent(lat, lon), lat, lon, transform)
    grid = transform.grid
    on_grid = built.on_latlon(grid.latitude_deg, grid.longitude_deg)
    assert np.max(np.abs(on_grid - built.height_m)) < 1e-9
    flat = np.linspace(-90.0, 90.0, 37)
    ring = np.arange(-180.0, 180.0, 5.0)
    frame = sample_on_latlon(built.coefficients, 21, flat, ring)
    lon2, lat2 = np.meshgrid(ring, flat)
    reference = sample_scalar(transform, built.coefficients, lat2, lon2)
    assert np.max(np.abs(frame - reference)) < 1e-9
    # A ring coarser than the truncation is the field's own values there,
    # with nothing folded.
    coarse = np.arange(0.0, 360.0, 10.0)
    lon2, lat2 = np.meshgrid(coarse, flat)
    reference = sample_scalar(transform, built.coefficients, lat2, lon2)
    frame = sample_on_latlon(built.coefficients, 21, flat, coarse)
    assert np.max(np.abs(frame - reference)) < 1e-9
    with pytest.raises(ValueError, match="regular ring"):
        sample_on_latlon(built.coefficients, 21, flat, np.arange(0.0, 180.0, 10.0))


def test_the_frame_planes_are_regular_pole_to_pole_and_keep_the_land():
    transform = _transform(21)
    lat, lon = _paleomap_axes(2.0)
    elevation = _continent(lat, lon)
    built = dynamics_terrain(elevation, lat, lon, transform)
    f_lat, f_lon, terrain, land = frame_planes(built, elevation, lat, lon,
                                               step_deg=1.0)
    assert f_lat[0] == -90.0 and f_lat[-1] == 90.0 and f_lat.size == 181
    assert f_lon[0] == 0.0 and f_lon.size == 360
    assert terrain.shape == land.shape == (181, 360)
    assert float(land.min()) == 0.0
    assert float(land.max()) == pytest.approx(1.0, abs=1e-12)
    # The land share is the map's own, to rounding.
    _, indicator = land_height(elevation)
    assert map_area_mean(land, f_lat, f_lon) == pytest.approx(
        map_area_mean(indicator, lat, lon), rel=1e-12)
    with pytest.raises(ValueError, match="divide"):
        frame_planes(built, elevation, lat, lon, step_deg=0.7)
    with pytest.raises(ValueError, match="repeats"):
        average_to_latlon(indicator, lat, lon, lat, lon)


# ---------------------------------------------------------- rest test

def _model(transform, vertical, phis):
    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=phis, physics=None,
        diffusion=None,
        semi_implicit=VerticalModeSemiImplicit(off_centring_weight=0.55),
        integrator=SEMILAG_INTEGRATORS[0], mass_fixer=False,
        water_fixer=False, positivity_repair=False, maximum_cfl=1.0e9,
        sponge_base_pa=0.0, semilag=SemiLagrangianOptions(),
    )


def test_the_resting_profile_is_the_closed_form():
    assert rest_temperature(1.0e5) == pytest.approx(288.15)
    assert rest_temperature(100.0) == pytest.approx(216.65)
    # 2,000 m of the standard atmosphere is 79,495 Pa under 101,325 Pa at
    # sea level; the same ratio under this profile's 100,000 Pa is 78,455.
    assert float(rest_surface_pressure(2000.0)) == pytest.approx(
        79495.0 * 1.0e5 / 101325.0, abs=5.0)
    with pytest.raises(ValueError, match="tropopause"):
        rest_surface_pressure(np.array([12000.0]))


def test_rest_over_flat_ground_stays_at_rest():
    transform = _transform(21)
    vertical = HybridCoordinate.surface_stretched(20, 100.0)
    phis = np.zeros(transform.grid.shape)
    bundle, phis_model = resting_state(transform, vertical, phis,
                                       _surface(transform))
    model = _model(transform, vertical, phis_model)
    result = rest_test(model, bundle, dt_s=1800.0, days=0.25,
                       samples_per_day=8)
    assert result["largest_m_s"] < 1.0e-9


#: MEASURED 2026-11-12 (a shared 24-core Linux CPU host, numpy float64;
#: the test takes 31 s there, so it runs in the full tier): at rest over the ring
#: mountain, T42, 20 surface-stretched levels, sl_si at 1,200 s, no
#: diffusion, no fixers, sampled every hour.  Largest wind over the first
#: six hours 3.31 m/s (hour 5, top level, over the ring's centre), rms
#: 0.264 m/s at hour 6; over the whole first day 4.47 m/s (hour 7) with
#: the rms levelling at 0.30-0.32 m/s from hour 7 to 24, so the residual
#: saturates rather than grows.  The gates are those six-hour numbers with
#: half again of headroom.
RING_T42_SIX_HOUR_LIMIT_M_S = 5.0
RING_T42_SIX_HOUR_RMS_LIMIT_M_S = 0.4


@pytest.mark.slow
def test_rest_over_the_ring_mountain_stays_under_its_measured_bound():
    transform = _transform(42)
    vertical = HybridCoordinate.surface_stretched(20, 100.0)
    height = ring_mountain(transform.grid)
    # The centre sits on the equator, between two Gaussian rows.
    assert 0.95 * RING_HEIGHT_M < float(np.max(height)) <= RING_HEIGHT_M
    bundle, phis_model = resting_state(
        transform, vertical, GRAVITY_M_S2 * height, _surface(transform))
    model = _model(transform, vertical, phis_model)
    samples = []
    result = rest_test(model, bundle, dt_s=1200.0, days=0.25,
                       samples_per_day=24, progress=samples.append)
    day = result["per_day"][0]
    print(f"ring T42 six hours: {day['max_wind_m_s']:.4e} m/s at level "
          f"{day['level']} {day['lat_deg']:.1f}N {day['lon_deg']:.1f}E, "
          f"rms {samples[-1]['rms_wind_m_s']:.4e}")
    assert len(samples) == 6
    assert math.isfinite(day["max_wind_m_s"])
    assert day["max_wind_m_s"] < RING_T42_SIX_HOUR_LIMIT_M_S
    assert samples[-1]["rms_wind_m_s"] < RING_T42_SIX_HOUR_RMS_LIMIT_M_S
