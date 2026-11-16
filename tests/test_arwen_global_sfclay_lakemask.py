"""The surface layer is handed WRF's LAKEMASK: an inland lake is not sea.

WRF reduces the saturation vapour pressure over water by the sea-salt
factor 0.98 only where xland > 1.5 and LAKEMASK = 0 (module_sf_sfclay.F:459,
sf_sfclayrev.F90:283; the kernel's sfclay.cu:272).  The native runtime
handed sfclay lakemask = 0 on every column (audit 2026-11-15), so every
inland lake evaporated as if it were salt water.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from arwen_global.physics.native_suite import ArwenCudaColumnSuite
from arwen_global.statics import lake_columns, water_columns

from test_arwen_global_level5_native import (  # noqa: E402
    _exchange, _fake_modules, _options)


def test_sfclay_is_handed_the_lake_columns_as_its_lakemask():
    exchange = _exchange()
    land = np.asarray(exchange.surface.land_fraction, np.float32)
    ice = exchange.surface.sea_ice_fraction
    water = water_columns(land, sea_ice_fraction=ice)
    assert water.sum() >= 2
    # Two open-water columns become inland lakes (all of their water is
    # lake); every other water column stays sea.
    rows, cols = np.nonzero(water)
    lake_fraction = np.zeros_like(land)
    lake_fraction[rows[:2], cols[:2]] = 1.0 - land[rows[:2], cols[:2]]
    exchange = replace(exchange, surface=replace(exchange.surface, lake_fraction=lake_fraction))
    lakes = lake_columns(land, lake_fraction, sea_ice_fraction=ice)
    assert lakes.sum() == 2

    modules = _fake_modules()
    stock = modules["arwen_global.core.sfclay"].sfclay
    seen = []

    def sfclay(*args, **kwargs):
        seen.append(np.array(kwargs["lakemask"], copy=True))
        return stock(*args, **kwargs)

    modules["arwen_global.core.sfclay"].sfclay = sfclay
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    suite.step(exchange)
    assert seen
    mask = seen[0]
    assert mask.dtype == np.float32
    np.testing.assert_array_equal(mask, lakes.astype(np.float32))
    # Nonzero on the lakes, zero on the sea and the land.
    assert (mask[lakes] == 1.0).all()
    assert (mask[~lakes] == 0.0).all()
    assert (water & ~lakes).any()
