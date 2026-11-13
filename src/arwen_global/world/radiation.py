"""The radiation under a world's sun.

The model's radiation (``arwen_global.core.rrtmgp.RRTMGPRadiation``) holds
the sun in two methods, ``_solar_constant`` and ``_cosine_zenith``, and its
``__call__`` reads the sun through nothing else.  A world's radiation is a
subclass that overrides exactly those two from :class:`~.sun.WorldSun`;
every kernel, table and pin under ``core/`` is untouched.  The subclass is
built only when the world changes the sun (the one branch where the native
runtime builds its radiation), so an Earth run never imports this module.

The subclass is made from whichever radiation class the runtime resolved
(it may be a substituted module's), so a substitution is honoured rather
than bypassed.  ``tests/test_arwen_global_world_sun.py`` pins the two parent
method names, their signatures and that ``__call__`` calls both: a rename
upstream would otherwise leave the override silently unused.

:func:`attach_sun` is what the run door calls on the physics it built: it
hands the native physics the world's sun, and refuses a world sun on a
physics that has no place for it (the reference physics' own sun, or no
physics at all), before any card time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .sun import WorldSun

#: The two methods of the parent that hold the sun.  A test pins them.
SUN_METHODS = ("_solar_constant", "_cosine_zenith")


@lru_cache(maxsize=None)
def world_radiation_class(parent: type) -> type:
    """The radiation class ``parent`` under a world's sun: a dataclass
    subclass with one more field, ``world_sun`` (a :class:`WorldSun`)."""
    for name in SUN_METHODS:
        if not callable(getattr(parent, name, None)):
            raise TypeError(
                f"{parent.__qualname__} has no {name}; the world sun overrides "
                f"{' and '.join(SUN_METHODS)} and cannot act on this radiation")

    @dataclass
    class WorldSunRadiation(parent):
        world_sun: WorldSun | None = None

        def __post_init__(self) -> None:
            if not isinstance(self.world_sun, WorldSun):
                raise TypeError(
                    "a world radiation needs world_sun, a WorldSun; got "
                    f"{type(self.world_sun).__name__}")
            super().__post_init__()

        def _solar_constant(self, valid_time) -> float:
            return self.world_sun.solar_constant(
                super()._solar_constant(valid_time), valid_time)

        def _cosine_zenith(self, valid_time, *, hour_offset_seconds=0.0):
            import cupy as cp
            from arwen_global.core.rrtmgp import DTYPE

            # The forecast's start instant: a day of another length turns
            # the sun from where Earth's stood then (world/clock.py).  The
            # radiation's own start_time is the forecast clock's start,
            # held across restarts by the checkpoint clock check.
            return self.world_sun.cosine_zenith(
                self.latitude_deg, self.longitude_deg, valid_time,
                hour_offset_seconds=hour_offset_seconds, xp=cp, dtype=DTYPE,
                start_time=getattr(self, "start_time", None))

    WorldSunRadiation.__name__ = f"WorldSun{parent.__name__}"
    WorldSunRadiation.__qualname__ = WorldSunRadiation.__name__
    return WorldSunRadiation


def attach_sun(physics, world):
    """Hand the world's sun to the physics ``build_physics`` made, and return
    the physics.  A world that keeps Earth's sun returns it untouched.

    The sun goes on the native runtime's ``world_sun`` slot, and the runtime
    builds the world radiation in place of the model's on its first
    radiation call.  Refused by name: a world sun with no
    physics, with the reference physics (its sun is its own and is not wired
    to a world), or with a sun setting this package does not build."""
    if world is None or world.is_earth or not world.changes_sun:
        return physics
    sun = WorldSun.from_world(world)
    if sun.is_earth:
        # Only settings refused above differ; nothing left to change.
        return physics
    if physics is None:
        raise ValueError(
            f"world {world.label!r} changes the sun, and this config runs no "
            "physics, so nothing would read it; set physics_mode = "
            "\"arwen-native\" or leave [world.sun] at Earth's values")
    suite = getattr(physics, "adapter", None)
    if suite is None or not hasattr(suite, "_runtime"):
        raise ValueError(
            f"world {world.label!r} changes the sun, and only the native "
            "physics (physics_mode = \"arwen-native\") runs a world's sun; "
            f"{type(physics).__name__} keeps its own and would radiate under "
            "Earth's")
    runtime = suite._runtime
    if runtime is None:
        # The suite builds its runtime on its first step only when it has
        # none (native_suite.ArwenCudaColumnSuite.step, pinned by a test),
        # so the one built here, with the suite's own arguments, is the one
        # that runs.  Built now, not at the first step, so the sun is in
        # place before any radiation is constructed and no suite edit is
        # needed to carry it.
        from ..physics.native_runtime import NativePhysicsRuntime

        runtime = suite._runtime = NativePhysicsRuntime(
            suite.options, modules=suite._modules)
    if runtime._bands:
        raise RuntimeError(
            "the world sun must be attached before the physics first runs: "
            "this runtime has already built its radiation under Earth's sun")
    runtime.world_sun = sun
    return physics


__all__ = ["SUN_METHODS", "attach_sun", "world_radiation_class"]
