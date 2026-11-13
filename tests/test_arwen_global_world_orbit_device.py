"""More orbits on the card (worlds plan, package 3.4): the world radiation
under the perpetual-equinox and an eccentric orbit, on a T255-sized grid,
gives the CPU's closed-form sun.  Seconds of card.  The CPU half of the
package's test is ``tests/test_arwen_global_world_orbit.py``; this one needs
package 1.3's ``world/sun.py`` with this package's patch, and skips by name
without it."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from arwen_global.world import parse_world
from arwen_global.world.orbit import Orbit

cp = pytest.importorskip("cupy")


def _card_present() -> bool:
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - no driver, no card
        return False


pytestmark = pytest.mark.skipif(not _card_present(), reason="the orbit card check needs a card")

INSTANTS = (
    datetime(2026, 1, 3, 0, 0),
    datetime(2026, 3, 21, 12, 0),
    datetime(2026, 6, 21, 6, 30),
    datetime(2026, 11, 12, 13, 47, 30),
)


def _world_sun_module():
    sun = pytest.importorskip(
        "arwen_global.world.sun", reason="package 1.3's world/sun.py is not on this tree")
    if not hasattr(sun.WorldSun, "equation_of_time_min"):
        pytest.skip("world/sun.py still refuses the orbits: package 3.4's patch "
                    "(w3-4-for-w1-3.patch) is not applied")
    return sun


def _world(text: str):
    import tomllib

    return parse_world(tomllib.loads(text)["world"])


def _t255_grid():
    nlat, nlon = 384, 768
    lat = (np.arange(nlat) + 0.5) * (180.0 / nlat) - 90.0
    lon = np.arange(nlon) * (360.0 / nlon)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def _radiation(sun, lat, lon):
    from arwen_global.core.rrtmgp import RRTMGPRadiation
    from arwen_global.world.radiation import world_radiation_class

    return world_radiation_class(RRTMGPRadiation)(
        start_time=INSTANTS[0], latitude_deg=lat, longitude_deg=lon, world_sun=sun)


def test_an_eccentric_orbit_on_the_card_is_the_cpus_closed_form():
    from arwen_global.core.rrtmgp import DTYPE

    sun_module = _world_sun_module()
    world = _world('[world]\nlabel = "e"\n[world.sun]\norbit = "eccentric"\n'
                   'eccentricity = 0.1\nperihelion_deg = 0.0\nobliquity_deg = 23.5\n')
    sun = sun_module.WorldSun.from_world(world)
    assert sun.orbit == Orbit.eccentric(0.1, 0.0, obliquity_deg=23.5)
    lat, lon = _t255_grid()
    radiation = _radiation(sun, lat, lon)
    for when in INSTANTS:
        got = radiation._cosine_zenith(when, hour_offset_seconds=1500.0)
        want = sun.cosine_zenith(cp.asnumpy(radiation.latitude_deg),
                                 cp.asnumpy(radiation.longitude_deg),
                                 when, hour_offset_seconds=1500.0, xp=np, dtype=DTYPE)
        # The plane's type is the one Earth's sun gives on the same grid: the
        # equation of time is a NumPy float64 in both, and it decides the
        # precision of the hour-angle plane it is added into.
        earth = type(radiation).__mro__[1]._cosine_zenith(
            radiation, when, hour_offset_seconds=1500.0)
        assert isinstance(got, cp.ndarray) and got.dtype == earth.dtype == want.dtype
        assert float(cp.max(cp.abs(got - cp.asarray(want)))) <= 2.0e-6, when
        assert radiation._solar_constant(when) == sun.solar_constant(0.0, when)
    # Closest approach at julian 80 (22 March 00 UTC): 1370 / 0.9**2.
    assert radiation._solar_constant(datetime(2026, 3, 22)) == pytest.approx(1370.0 / 0.81,
                                                                           rel=1e-12)


def test_the_perpetual_equinox_on_the_card_keeps_the_sun_over_the_equator():
    sun_module = _world_sun_module()
    world = _world('[world]\nlabel = "aqua"\n[world.sun]\norbit = "perpetual-equinox"\n')
    sun = sun_module.WorldSun.from_world(world)
    lat, lon = _t255_grid()
    radiation = _radiation(sun, lat, lon)
    weight = np.cos(np.deg2rad(lat))
    for when in INSTANTS:
        assert radiation._solar_constant(when) == 1370.0
        mu = cp.asnumpy(radiation._cosine_zenith(when)).astype(np.float64)
        mean = float(np.sum(np.maximum(mu, 0.0) * weight) / np.sum(weight))
        assert mean == pytest.approx(0.25, rel=1e-3)
        row, col = np.unravel_index(int(np.argmax(mu)), mu.shape)
        assert abs(lat[row, 0]) < 0.5
        sub_lat, sub_lon = sun.subsolar_point(when)
        assert sub_lat == 0.0
        assert abs(((lon[0, col] - sub_lon) + 180.0) % 360.0 - 180.0) < 0.5
