"""A world's sun on the card (worlds plan, package 1.3): at Earth's values
the world radiation's two sun methods return the model radiation's device
arrays bit for bit, on a T255-sized grid.  Seconds of card.  The CPU half
of the package's test is ``tests/test_arwen_global_world_sun.py``."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from arwen_global.world.radiation import world_radiation_class
from arwen_global.world.sun import WorldSun

cp = pytest.importorskip("cupy")


def _card_present() -> bool:
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - no driver, no card
        return False


pytestmark = pytest.mark.skipif(not _card_present(), reason="the world sun's card check needs a card")

INSTANTS = (
    datetime(2026, 11, 12, 0, 0),
    datetime(2026, 11, 12, 13, 47, 30),
    datetime(2024, 2, 29, 18, 0),
    datetime(2025, 6, 21, 12, 0),
    datetime(2026, 9, 1, 4, 20, 10, 250000),
)


def _t255_grid():
    nlat, nlon = 384, 768
    lat = (np.arange(nlat) + 0.5) * (180.0 / nlat) - 90.0
    lon = np.arange(nlon) * (360.0 / nlon)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def test_at_earths_values_the_world_radiation_matches_the_model_radiation_on_the_card():
    from arwen_global.core.rrtmgp import RRTMGPRadiation

    lat, lon = _t255_grid()
    start = INSTANTS[0]
    parent = RRTMGPRadiation(start_time=start, latitude_deg=lat, longitude_deg=lon)
    child = world_radiation_class(RRTMGPRadiation)(
        start_time=start, latitude_deg=lat, longitude_deg=lon, world_sun=WorldSun())
    for when in INSTANTS:
        for offset in (0.0, 1500.0):
            want = parent._cosine_zenith(when, hour_offset_seconds=offset)
            got = child._cosine_zenith(when, hour_offset_seconds=offset)
            assert isinstance(got, cp.ndarray)
            assert got.dtype == want.dtype and got.shape == want.shape
            assert bool(cp.all(got.view(cp.uint32) == want.view(cp.uint32))), (when, offset)
        assert child._solar_constant(when) == parent._solar_constant(when)


def test_a_tilted_brighter_sun_on_the_card_keeps_a_quarter_of_its_constant():
    from arwen_global.core.rrtmgp import RRTMGPRadiation

    lat, lon = _t255_grid()
    start = INSTANTS[0]
    child = world_radiation_class(RRTMGPRadiation)(
        start_time=start, latitude_deg=lat, longitude_deg=lon,
        world_sun=WorldSun(obliquity_deg=90.0, brightness=1.05))
    solar = child._solar_constant(start)
    assert solar == RRTMGPRadiation._solar_constant(start) * 1.05
    mu = cp.asnumpy(child._cosine_zenith(start)).astype(np.float64)
    weight = np.cos(np.deg2rad(lat))
    mean = float(np.sum(np.maximum(mu, 0.0) * weight) / np.sum(weight))
    assert mean == pytest.approx(0.25, rel=1e-3)
    # The sun stands near 52 S in mid-November on a planet on its side.
    row = np.unravel_index(int(np.argmax(mu)), mu.shape)[0]
    assert abs(lat[row, 0] - WorldSun(obliquity_deg=90.0).subsolar_point(start)[0]) < 0.5
