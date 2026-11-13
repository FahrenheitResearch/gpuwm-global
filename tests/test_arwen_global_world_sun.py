"""A world's sun: tilt and brightness (worlds plan, package 1.3).

On the CPU, with NumPy standing in for CuPy (the radiation's own CPU
stand-in convention), so the radiation's two sun methods and the world's
overrides of them run here exactly as written:

* at Earth's values the override returns the parent's arrays bit for bit;
* global-mean sunlight is a quarter of the solar constant, at any tilt;
* the annual mean at a pole is the solar constant times sin(tilt) over pi;
* the parent's two method names, their signatures and their calls from
  ``__call__`` are pinned, so an upstream rename cannot leave the override
  silently unused;
* a sun setting another package builds is refused by name, and so is a
  world sun on a physics with no place for it.

The same bit-for-bit comparison on the card is
``tests/test_arwen_global_world_sun_device.py``.
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone
import inspect
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from arwen_global.core import rrtmgp
from arwen_global.core.rrtmgp import DTYPE, RRTMGPRadiation
from arwen_global.world import EARTH, parse_world
from arwen_global.world.radiation import (
    SUN_METHODS, attach_sun, world_radiation_class)
from arwen_global.world.sun import NOT_BUILT_HERE, WorldSun

UTC = timezone.utc

#: Instants that cross every branch of WRF's clock: before and after the
#: julian-80 equinox, a leap day, the year's last hour, sub-second time.
INSTANTS = (
    datetime(2026, 11, 12, 0, 0),
    datetime(2026, 11, 12, 13, 47, 30),
    datetime(2019, 1, 15, 6, 0),
    datetime(2024, 2, 29, 18, 0),
    datetime(2024, 3, 21, 0, 0),
    datetime(2025, 6, 21, 12, 0),
    datetime(2025, 12, 31, 23, 0),
    datetime(2026, 9, 1, 4, 20, 10, 250000),
)
OFFSETS = (0.0, 1500.0, 450.0)


@pytest.fixture
def numpy_as_cupy(monkeypatch):
    """The radiation imports its array module inside each method; NumPy
    answers for it here."""
    monkeypatch.setitem(sys.modules, "cupy", np)


def _grid(nlat: int = 32, nlon: int = 64):
    """A small regular grid of cell centres in the radiation's float32."""
    lat = (np.arange(nlat) + 0.5) * (180.0 / nlat) - 90.0
    lon = np.arange(nlon) * (360.0 / nlon)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2.astype(DTYPE), lon2.astype(DTYPE)


def _bare(cls, *, world_sun=None, grid=None):
    """An instance carrying only what the two sun methods read (no gas
    tables, no device)."""
    radiation = object.__new__(cls)
    radiation.latitude_deg, radiation.longitude_deg = grid or _grid()
    if world_sun is not None:
        radiation.world_sun = world_sun
    return radiation


def _world(text: str):
    import tomllib

    return parse_world(tomllib.loads(text)["world"])


# --------------------------------------------------------- Earth, bit for bit


def test_at_earths_values_the_override_returns_the_parents_arrays_bit_for_bit(
        numpy_as_cupy):
    parent = _bare(RRTMGPRadiation)
    child = _bare(world_radiation_class(RRTMGPRadiation), world_sun=WorldSun())
    for when in INSTANTS:
        for offset in OFFSETS:
            want = parent._cosine_zenith(when, hour_offset_seconds=offset)
            got = child._cosine_zenith(when, hour_offset_seconds=offset)
            assert got.dtype == want.dtype
            assert got.shape == want.shape
            assert got.tobytes() == want.tobytes(), (when, offset)
        want_s = parent._solar_constant(when)
        got_s = child._solar_constant(when)
        assert type(got_s) is type(want_s)
        assert got_s == want_s


def test_the_sun_module_alone_reproduces_the_parent_at_earths_values(numpy_as_cupy):
    """The geometry in world/sun.py is the parent's on its own terms, so a
    later package (3.1, 3.4) that edits it is held to the same planes."""
    parent = _bare(RRTMGPRadiation)
    sun = WorldSun()
    for when in INSTANTS:
        want = parent._cosine_zenith(when, hour_offset_seconds=1500.0)
        got = sun.cosine_zenith(parent.latitude_deg, parent.longitude_deg, when,
                                hour_offset_seconds=1500.0, dtype=DTYPE)
        assert got.tobytes() == want.tobytes()


def test_tilt_and_brightness_change_the_sun_and_nothing_else(numpy_as_cupy):
    when = datetime(2026, 11, 12, 0, 0)
    parent = _bare(RRTMGPRadiation)
    tilted = _bare(world_radiation_class(RRTMGPRadiation),
                   world_sun=WorldSun(obliquity_deg=90.0, brightness=1.05))
    assert tilted._solar_constant(when) == parent._solar_constant(when) * 1.05
    assert not np.array_equal(tilted._cosine_zenith(when), parent._cosine_zenith(when))
    # The equinox: every tilt puts the sun over the equator.
    equinox_julian = 80.0
    for tilt in (0.0, 23.5, 60.0, 90.0):
        assert float(WorldSun(obliquity_deg=tilt).declination_rad(equinox_julian)) == 0.0
    # No tilt: no seasons.
    for when in INSTANTS:
        lat, _ = WorldSun(obliquity_deg=0.0).subsolar_point(when)
        assert lat == 0.0


def test_the_subsolar_point_is_where_the_sun_is_overhead(numpy_as_cupy):
    """Mid-November with the planet on its side: the sun stands over about
    52 S (plan, demo 2), and the cosine of the zenith angle there is 1."""
    when = datetime(2026, 11, 12, 0, 0)
    sun = WorldSun(obliquity_deg=90.0)
    lat, lon = sun.subsolar_point(when)
    assert -53.0 < lat < -51.0
    mu = sun.cosine_zenith(np.array([lat]), np.array([lon]), when)
    assert mu[0] == pytest.approx(1.0, abs=1e-12)
    # Earth's November sun, for comparison: about 17.8 S.
    earth_lat, _ = WorldSun().subsolar_point(when)
    assert -18.5 < earth_lat < -17.0


# ------------------------------------------------------- the closed forms


def _area_mean(values, lat_deg):
    weight = np.cos(np.deg2rad(lat_deg))
    return float(np.sum(values * weight) / np.sum(weight))


@pytest.mark.parametrize("tilt", [0.0, 23.5, 45.0, 90.0, 120.0])
@pytest.mark.parametrize("brightness", [0.9, 1.0, 1.1])
def test_global_mean_sunlight_is_a_quarter_of_the_solar_constant(
        numpy_as_cupy, tilt, brightness):
    """Half the sphere is lit and the lit half's mean cosine is a half:
    the global mean of S max(mu, 0) is S/4 whatever the tilt.  Read through
    the world radiation's own two methods on a 0.25-degree grid."""
    lat, lon = _grid(720, 1440)
    radiation = _bare(world_radiation_class(RRTMGPRadiation),
                      world_sun=WorldSun(obliquity_deg=tilt, brightness=brightness),
                      grid=(lat.astype(np.float64), lon.astype(np.float64)))
    for when in (datetime(2026, 11, 12, 0), datetime(2025, 6, 21, 9),
                 datetime(2024, 3, 21, 17, 30)):
        solar = radiation._solar_constant(when)
        mu = radiation._cosine_zenith(when)
        mean = _area_mean(solar * np.maximum(mu, 0.0), lat)
        assert mean / solar == pytest.approx(0.25, rel=2e-4), when
        assert solar == pytest.approx(RRTMGPRadiation._solar_constant(when) * brightness,
                                      rel=1e-15)


@pytest.mark.parametrize("tilt", [10.0, 23.5, 45.0, 60.0, 90.0])
def test_the_annual_mean_at_the_pole_is_the_solar_constant_times_sin_tilt_over_pi(tilt):
    """At a pole the cosine of the zenith angle is the sine of the
    declination, sin(tilt) sin(L) on WRF's orbit, and the year's mean of
    its positive half is sin(tilt)/pi.  Hourly over one 365-day year, at a
    fixed solar constant (the model's own varies 3.4 % over its orbit and is
    not what this closed form is about)."""
    sun = WorldSun(obliquity_deg=tilt, brightness=1.0)
    solar = 1370.0
    start = datetime(2025, 1, 1)
    hours = [start + timedelta(hours=h) for h in range(365 * 24)]
    pole = np.array([90.0])
    lon = np.array([0.0])
    north = np.array([sun.insolation(pole, lon, t, solar)[0] for t in hours])
    south = np.array([sun.insolation(-pole, lon, t, solar)[0] for t in hours])
    want = solar * math.sin(math.radians(tilt)) / math.pi
    assert north.mean() == pytest.approx(want, rel=2e-3)
    assert south.mean() == pytest.approx(want, rel=2e-3)


def test_tilt_and_its_supplement_give_the_same_sun():
    for when in INSTANTS:
        a = WorldSun(obliquity_deg=60.0).subsolar_point(when)
        b = WorldSun(obliquity_deg=120.0).subsolar_point(when)
        assert a == pytest.approx(b, abs=1e-9)


# ---------------------------------------------- the parent's methods, pinned


def test_the_two_parent_sun_methods_are_pinned():
    assert SUN_METHODS == ("_solar_constant", "_cosine_zenith")
    assert isinstance(RRTMGPRadiation.__dict__["_solar_constant"], staticmethod)
    assert list(inspect.signature(RRTMGPRadiation._solar_constant).parameters) == [
        "valid_time"]
    signature = inspect.signature(RRTMGPRadiation._cosine_zenith)
    assert list(signature.parameters) == ["self", "valid_time", "hour_offset_seconds"]
    assert signature.parameters["hour_offset_seconds"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["hour_offset_seconds"].default == 0.0
    call = inspect.getsource(RRTMGPRadiation.__call__)
    assert call.count("self._cosine_zenith(") == 1
    assert call.count("self._solar_constant(") == 1


def test_the_parent_holds_the_sun_in_those_two_methods_only():
    """Earth's tilt appears once in the radiation module, inside the
    cosine of the zenith angle: no second sun the override would miss."""
    source = inspect.getsource(rrtmgp)
    assert source.count("23.5") == 1
    assert "np.sin(23.5 * degrad)" in inspect.getsource(RRTMGPRadiation._cosine_zenith)
    assert source.count("1370.0") == 1
    assert "1370.0" in inspect.getsource(RRTMGPRadiation._solar_constant)


def test_the_world_radiation_is_a_dataclass_subclass_with_one_more_field():
    cls = world_radiation_class(RRTMGPRadiation)
    assert cls is world_radiation_class(RRTMGPRadiation)
    assert issubclass(cls, RRTMGPRadiation)
    parent_fields = [f.name for f in fields(RRTMGPRadiation)]
    assert [f.name for f in fields(cls)] == parent_fields + ["world_sun"]
    with pytest.raises(TypeError, match="needs world_sun"):
        cls(start_time=datetime(2026, 11, 12), latitude_deg=[0.0], longitude_deg=[0.0])

    class NoSun:
        pass

    with pytest.raises(TypeError, match="has no _solar_constant"):
        world_radiation_class(NoSun)


# -------------------------------------------------------------- refusals


def test_every_sun_setting_is_built():
    # The day length (package 3.1, arwen_global.world.clock) and the two
    # other orbits (package 3.4, arwen_global.world.orbit; its tests are
    # tests/test_arwen_global_world_orbit.py) are built, so nothing of
    # [world.sun] is refused here any more.
    assert NOT_BUILT_HERE == {}
    orbit = _world('[world]\nlabel = "eq"\n[world.sun]\norbit = "perpetual-equinox"\n')
    assert WorldSun.from_world(orbit).orbit.kind == "perpetual-equinox"
    both = _world('[world]\nlabel = "eq-fast"\n[world.sun]\n'
                  'orbit = "perpetual-equinox"\nday_length_h = 12.0\n')
    sun = WorldSun.from_world(both)
    assert sun.orbit.kind == "perpetual-equinox" and sun.day_length_h == 12.0


def test_a_day_length_builds_a_sun_that_turns_from_the_start(numpy_as_cupy):
    """Package 3.1: a 12-hour world's sun stands where Earth's does at the
    start instant, and is half a turn further round after six hours."""
    day = _world('[world]\nlabel = "fast"\n[world.sun]\nday_length_h = 12.0\n')
    sun = WorldSun.from_world(day)
    assert sun.day_length_h == 12.0 and not sun.is_earth
    start = datetime(2026, 11, 12, 0, 0)
    child = _bare(world_radiation_class(RRTMGPRadiation), world_sun=sun)
    child.start_time = start
    parent = _bare(RRTMGPRadiation)
    # At the start instant (offset 0) the two suns agree to the bit.
    assert (child._cosine_zenith(start).tobytes()
            == parent._cosine_zenith(start).tobytes())
    # Twelve hours on, the 12-hour sun has made a whole turn and Earth's
    # half of one: its sub-solar longitude is Earth's plus 180.
    later = start + timedelta(hours=12)
    earth_lat, earth_lon = WorldSun().subsolar_point(later)
    lat, lon = sun.subsolar_point(later, start_time=start)
    assert lat == earth_lat
    assert abs(((lon - earth_lon) % 360.0) - 180.0) < 1e-9
    # And the radiation's brightest column is at that longitude.
    mu = child._cosine_zenith(later)
    row, col = np.unravel_index(np.argmax(mu), mu.shape)
    gap = abs(((float(child.longitude_deg[row, col]) - lon + 180.0) % 360.0) - 180.0)
    assert gap <= 360.0 / child.longitude_deg.shape[1]
    # Without a start instant a day of another length is refused by name.
    with pytest.raises(ValueError, match="12-hour day.*start instant"):
        sun.cosine_zenith(np.zeros(1), np.zeros(1), later)


def test_a_sun_out_of_range_is_refused():
    with pytest.raises(ValueError, match="obliquity_deg"):
        WorldSun(obliquity_deg=-1.0)
    with pytest.raises(ValueError, match="brightness"):
        WorldSun(brightness=0.0)


TILTED = '[world]\nlabel = "on-its-side"\n[world.sun]\nobliquity_deg = 90.0\n'


def test_an_earth_world_leaves_the_physics_untouched():
    marker = object()
    assert attach_sun(marker, EARTH) is marker
    assert attach_sun(None, EARTH) is None
    flat = _world('[world]\nlabel = "flat"\n[world.terrain]\nscale = 0.0\n')
    assert attach_sun(marker, flat) is marker


def test_a_world_sun_with_no_physics_or_the_reference_physics_is_refused():
    world = _world(TILTED)
    with pytest.raises(ValueError, match="runs no physics"):
        attach_sun(None, world)
    with pytest.raises(ValueError, match="only the native physics"):
        attach_sun(SimpleNamespace(), world)


def _native_bridge():
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.physics import NativeArwenPhysicsBridge

    cfg = load_config(config_root() / "arwen_global_level5_native_smoke.toml")
    return NativeArwenPhysicsBridge(cfg.native_adapter_name,
                                    dict(cfg.native_adapter_options))


def test_a_world_sun_reaches_the_runtime_the_suite_will_run():
    from arwen_global.physics.native_runtime import NativePhysicsRuntime
    from arwen_global.physics.native_suite import ArwenCudaColumnSuite

    bridge = _native_bridge()
    suite = bridge.adapter
    assert suite._runtime is None
    assert attach_sun(bridge, _world(TILTED)) is bridge
    runtime = suite._runtime
    assert isinstance(runtime, NativePhysicsRuntime)
    assert runtime.world_sun == WorldSun(obliquity_deg=90.0, brightness=1.0)
    assert runtime.options is suite.options
    # The suite keeps the runtime it finds: it builds one only when it has
    # none.  If that ever changes, the sun above would be dropped.
    step = inspect.getsource(ArwenCudaColumnSuite.step)
    assert "if self._runtime is None:" in step
    assert NativePhysicsRuntime.world_sun is None
    # An Earth bridge never gets a runtime early or a sun.
    earth = _native_bridge()
    attach_sun(earth, EARTH)
    assert earth.adapter._runtime is None


def test_a_sun_cannot_be_attached_after_the_radiation_was_built():
    bridge = _native_bridge()
    attach_sun(bridge, _world(TILTED))
    bridge.adapter._runtime._band  # the first band cache, as a step makes it
    with pytest.raises(RuntimeError, match="before the physics first runs"):
        attach_sun(bridge, _world(TILTED))


# ------------------------------------- the runtime's one branch, on the CPU


class _Built(Exception):
    pass


def _fake_radiation_module():
    from dataclasses import dataclass, field

    @dataclass
    class FakeRadiation:
        start_time: object
        latitude_deg: object
        longitude_deg: object
        column_chunk: int = 0
        validation_mode: str = "fused"
        trace_gas_overrides: object = None
        column_size_bounding: bool = False
        ozone_climatology: str = ""
        seen: dict = field(default_factory=dict, init=False)

        def __post_init__(self):
            pass

        @staticmethod
        def _solar_constant(valid_time):
            return 1370.0

        def _cosine_zenith(self, valid_time, *, hour_offset_seconds=0.0):
            return None

        def __call__(self, **kwargs):
            raise _Built

    # An empty module name: the provenance recorder skips it, so this fake
    # never enters the process-wide record of resolved physics modules.
    module = ModuleType("")
    module.RRTMGPRadiation = FakeRadiation
    return module


def _radiation_once(world_sun):
    from arwen_global.physics.native_runtime import NativePhysicsRuntime

    module = _fake_radiation_module()
    options = SimpleNamespace(
        radiation_interval_s=10.0, start_time=datetime(2026, 11, 12, tzinfo=UTC),
        radiation_column_chunk=512, radiation_validation_mode="fused",
        greenhouse_gas_overrides=lambda valid=None: None, ozone="cam-zonal-monthly",
        radiation_cloud_inhomogeneity="none", radiation_liquid_radius="scheme",
        ghg_year="start", radiation_between="hold")
    runtime = NativePhysicsRuntime(options, modules={"arwen_global.core.rrtmgp": module})
    runtime.world_sun = world_sun
    plane = np.zeros((2, 3))
    batch = SimpleNamespace(
        xp=np, time_s=0.0, band=None,
        arrays={name: plane for name in (
            "latitude_deg", "longitude_deg", "p_full", "p_half", "temperature",
            "exner", "qv", "qc", "qi")},
        surface=SimpleNamespace(temperature_k=plane, emissivity=plane, albedo=plane))
    persistent = SimpleNamespace(metadata={"last_radiation_bucket": -1},
                                 fake_radiation_state=lambda: None)
    with pytest.raises(_Built):
        runtime._radiation_step(batch, persistent, cfg=None)
    return module.RRTMGPRadiation, runtime._radiation


def test_the_runtime_builds_the_models_radiation_when_no_world_sun_is_set():
    parent, built = _radiation_once(None)
    assert type(built) is parent
    assert not hasattr(built, "world_sun")


def test_the_runtime_builds_the_world_radiation_under_a_world_sun():
    sun = WorldSun(obliquity_deg=90.0, brightness=1.05)
    parent, built = _radiation_once(sun)
    assert type(built) is world_radiation_class(parent)
    assert built.world_sun is sun
    assert built.ozone_climatology == "cam-zonal-monthly"
    assert built.column_size_bounding is True
