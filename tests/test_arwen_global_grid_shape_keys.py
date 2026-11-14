"""Measured rows and caches are keyed on the grid's shape, not its truncation.

A truncation does not fix the grid: ``[grid] dealias_factor = 1.0`` puts
T383 on 384 x 768, the shape of shipped T255, where shipped T383 is
576 x 1152.  Two places keyed on the truncation alone or on the shape alone
(the linear-grid study, MEASURED 2026-11-14 on an RTX 5090):

* the sizer matched its completed-plan, measured-run and floor rows on the
  truncation, so T383 on 384 x 768 was priced at the 14.27 GiB of card a
  quadratic T383 run took while its own allocator read 6.75 GiB;
* the scorecards cached the rebuilt surface geopotential under the analysis
  and nlat x nlon, so a T255 run and a T383 run on 384 x 768 from the same
  analysis read each other's terrain.

Neither changes a shipped config's result: every shipped config runs on its
truncation's quadratic grid, where a row without a shape is that grid.
"""
from __future__ import annotations

from arwen_global import sizing
from arwen_global.upper_air_scorecard import surface_geopotential_cache_name


def test_a_row_without_a_shape_is_its_truncations_quadratic_grid():
    assert sizing.quadratic_grid_shape(383) == (576, 1152)
    assert sizing.same_grid({"truncation": 383}, 383)
    assert sizing.same_grid({"truncation": 383, "nlat": 576, "nlon": 1152}, 383)
    assert not sizing.same_grid({"truncation": 383, "nlat": 576, "nlon": 1152}, 383, (384, 768))
    assert sizing.same_grid({"truncation": 383, "nlat": 384, "nlon": 768}, 383, (384, 768))
    assert not sizing.same_grid({"truncation": 255, "nlat": 384, "nlon": 768}, 383, (384, 768))


def test_no_quadratic_row_prices_another_shape_of_the_same_truncation():
    linear = {"truncation": 383, "nlat": 384, "nlon": 768}
    assert sizing.grid_shape_of(linear) == (384, 768)
    for rows in (sizing.COMPLETED_PLAN_PEAKS, sizing.MEASURED_RUNS, sizing.MEASURED_FLOORS):
        for row in rows:
            if int(row["truncation"]) == 383:
                assert not sizing.same_grid(row, 383, (384, 768)), row


def test_the_terrain_cache_is_keyed_on_the_truncation_as_well_as_the_shape():
    digest = "ab" * 32
    t255 = {"transform": {"truncation": 255, "nlat": 384, "nlon": 768}, "config": {}}
    tl383 = {"transform": {"truncation": 383, "nlat": 384, "nlon": 768}, "config": {}}
    a, b = (surface_geopotential_cache_name(digest, r) for r in (t255, tl383))
    assert a != b and "T255-384x768" in a and "T383-384x768" in b
    assert a.startswith(f"phi-surface-{digest[:16]}-") and a.endswith(".npz")
