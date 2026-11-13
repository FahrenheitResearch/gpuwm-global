"""A world's sun: its tilt and its brightness.

The model's sun is WRF v4.6.1's ``radconst``/``calc_coszen`` geometry, carried
in ``arwen_global.core.rrtmgp.RRTMGPRadiation`` as two methods: the solar
constant (1370 W m-2 times WRF's eccentricity factor on its fixed 365-day
orbit) and the cosine of the solar zenith angle.  A world changes two
numbers in them and nothing else:

* ``obliquity_deg`` replaces the 23.5 degrees inside the declination,
  ``asin(sin(obliquity) sin(solar longitude))``.  The orbit, its 365-day
  year, the equation of time and the hour angle stay WRF's.
* ``brightness`` multiplies the model's own solar constant, eccentricity
  factor included.

The arithmetic below is the parent's, operation for operation and type for
type (NumPy float64 scalars for the orbit, the radiation's float32 for the
planes), so at Earth's values it returns the parent's numbers bit for bit;
``tests/test_arwen_global_world_sun.py`` holds it to that on the CPU, and
its card test makes the same comparison on the device.

The other two orbits of ``[world.sun]`` (package 3.4) come from
``arwen_global.world.orbit``: under ``orbit = "perpetual-equinox"`` or
``"eccentric"`` the declination, the equation of time and the factor on
the solar constant are that orbit's closed forms in place of WRF's series,
on the same 365-day year and julian clock, and the solar constant is the
model's 1370 W m-2 at mean distance times the orbit's ``(a / r)**2`` and
the brightness.  Under ``orbit = "earth"`` none of that code runs.

* ``day_length_h`` turns the hour angle faster or slower from the
  forecast's start instant (``arwen_global.world.clock``, package 3.1): at
  the start the sun stands where Earth's does, and ``elapsed`` seconds later
  it has turned ``PlanetClock.extra_turn_deg(elapsed)`` further round.  The
  declination, the equation of time and the solar constant stay on the
  calendar's dates.  The dynamics' rotation rate follows the same day from
  the same module, so the sun and the Coriolis term cannot disagree.

Ozone and the land's vegetation calendar stay on Earth's dates under any
sun.

An obliquity above 90 degrees is accepted as the parser accepts it.  In this
model it gives the same sun as 180 degrees minus it: the declination only
reads ``sin(obliquity)``, and the model's solar day is a fixed 24 hours, so
the reversed spin relative to the orbit has nothing to act on.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np

from .orbit import MEAN_DISTANCE_SOLAR_CONSTANT_W_M2, Orbit, orbit_of

#: Earth's tilt as the model's sun holds it (WRF ``radconst``).
EARTH_OBLIQUITY_DEG = 23.5
#: The model's sun at full strength.
EARTH_BRIGHTNESS = 1.0
#: The calendar's solar day, in hours.
EARTH_DAY_LENGTH_H = 24.0

#: Settings of ``[world.sun]`` this package does not build, with the reason
#: and the package that will.  Refused before any card time.  Empty since
#: the day length (package 3.1) and the orbits (package 3.4) were built.
NOT_BUILT_HERE: dict[str, str] = {}


def wrf_clock(valid_time: datetime) -> tuple[float, float]:
    """WRF's clock at ``valid_time``: the hour of the UTC day and the
    zero-based fractional julian day, with the parent's arithmetic
    (``RRTMGPRadiation._cosine_zenith``)."""
    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0
            + valid_time.microsecond / 3.6e9)
    julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
    return hour, julian


def wrf_solar_longitude_deg(julian: float) -> float:
    """WRF ``radconst``'s orbital longitude of the sun, in degrees from the
    March equinox it places at julian 80, on a 365-day year."""
    dpd = 360.0 / 365.0
    if julian >= 80.0:
        return dpd * (julian - 80.0)
    return dpd * (julian + 285.0)


def wrf_equation_of_time_min(julian: float):
    """WRF ``radconst``'s equation of time in minutes (a NumPy float64, as
    the parent computes it: its type decides the precision of the hour
    angle plane it is added into)."""
    da = 2.0 * np.pi * (julian - 1.0) / 365.0
    return 229.18 * (
        0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
        - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))


@dataclass(frozen=True)
class WorldSun:
    """The sun of one world: WRF's orbit with ``obliquity_deg`` and
    ``brightness``, or with ``orbit`` set, one of the orbits of
    ``arwen_global.world.orbit`` (whose tilt must be ``obliquity_deg``).
    Built from a parsed world with :meth:`from_world`."""

    obliquity_deg: float = EARTH_OBLIQUITY_DEG
    brightness: float = EARTH_BRIGHTNESS
    orbit: Orbit | None = None
    day_length_h: float = EARTH_DAY_LENGTH_H

    def __post_init__(self) -> None:
        from .clock import PlanetClock

        object.__setattr__(
            self, "day_length_h", PlanetClock(self.day_length_h).day_length_h)
        if self.orbit is not None:
            if not isinstance(self.orbit, Orbit):
                raise TypeError(
                    f"a world sun's orbit is an Orbit or None, got {type(self.orbit).__name__}")
            if float(self.obliquity_deg) != self.orbit.obliquity_deg:
                raise ValueError(
                    f"the sun's obliquity_deg {self.obliquity_deg!r} is not its "
                    f"orbit's {self.orbit.obliquity_deg!r}")
        obliquity = float(self.obliquity_deg)
        brightness = float(self.brightness)
        if not math.isfinite(obliquity) or not 0.0 <= obliquity <= 180.0:
            raise ValueError(
                f"the sun's obliquity_deg must be 0 to 180, got {self.obliquity_deg!r}")
        if not math.isfinite(brightness) or brightness <= 0.0:
            raise ValueError(
                f"the sun's brightness must be a positive number, got {self.brightness!r}")
        object.__setattr__(self, "obliquity_deg", obliquity)
        object.__setattr__(self, "brightness", brightness)

    @classmethod
    def from_world(cls, world) -> "WorldSun":
        """The sun of a parsed world (``arwen_global.world.WorldSpec``).
        Refuses by name a sun setting this package does not build."""
        sun = world.sun
        return cls(obliquity_deg=sun.obliquity_deg, brightness=sun.brightness,
                   orbit=orbit_of(sun), day_length_h=sun.day_length_h)

    @property
    def is_earth(self) -> bool:
        return (self.obliquity_deg == EARTH_OBLIQUITY_DEG
                and self.brightness == EARTH_BRIGHTNESS
                and self.orbit is None
                and self.day_length_h == EARTH_DAY_LENGTH_H)

    def record(self) -> dict[str, object]:
        """What a receipt or a tape may store about this sun."""
        if self.orbit is not None:
            return {
                "geometry": "world-orbit",
                "obliquity_deg": self.obliquity_deg,
                "brightness": self.brightness,
                "day_length_h": self.day_length_h,
                **self.orbit.record(self.brightness),
            }
        return {
            "geometry": "wrf-radconst",
            "obliquity_deg": self.obliquity_deg,
            "brightness": self.brightness,
            "day_length_h": self.day_length_h,
        }

    def _extra_minutes(self, valid_time: datetime, start_time, offset_s: float) -> float:
        """The hour angle this world's day adds to WRF's at ``valid_time``
        (plus ``offset_s``, WRF's mid-interval shift), in minutes of WRF's
        ``solar_minutes`` (4 per degree).  Exactly 0.0 for a 24-hour day,
        which then never reads ``start_time``."""
        if self.day_length_h == EARTH_DAY_LENGTH_H:
            return 0.0
        if start_time is None:
            raise ValueError(
                f"a {self.day_length_h:g}-hour day turns the sun from the "
                "forecast's start instant, and no start instant was given")
        from .clock import PlanetClock

        elapsed = (valid_time - start_time).total_seconds() + float(offset_s)
        return 4.0 * PlanetClock(self.day_length_h).extra_turn_deg(elapsed)

    # -- the solar constant -------------------------------------------------

    def solar_constant(self, model_value: float, valid_time: datetime | None = None):
        """The top-of-atmosphere irradiance: the model's own solar constant
        at this instant (``RRTMGPRadiation._solar_constant``) times the
        brightness.  At brightness 1 it is ``model_value`` itself.

        Under another orbit WRF's eccentricity factor in ``model_value`` is
        not this orbit's, so the value is the model's 1370 W m-2 at mean
        distance times the orbit's ``(a / r)**2`` at ``valid_time`` (which is
        then required) times the brightness."""
        if self.orbit is not None:
            if valid_time is None:
                raise TypeError(
                    f"the solar constant on the {self.orbit.kind} orbit needs "
                    "the valid time: the distance changes through the year")
            _, julian = wrf_clock(valid_time)
            return (np.float64(MEAN_DISTANCE_SOLAR_CONSTANT_W_M2)
                    * self.orbit.distance_factor(julian) * self.brightness)
        if self.brightness == EARTH_BRIGHTNESS:
            return model_value
        return model_value * self.brightness

    # -- the geometry ---------------------------------------------------------

    def equation_of_time_min(self, julian: float):
        """The equation of time in minutes: WRF's series on WRF's orbit, or
        this world's orbit's closed form (a NumPy float64 either way)."""
        if self.orbit is not None:
            return self.orbit.equation_of_time_min(julian)
        return wrf_equation_of_time_min(julian)

    def declination_rad(self, julian: float):
        """The sun's declination on WRF's orbit with this tilt, or on this
        world's orbit, in radians (a NumPy float64, as the parent computes
        it)."""
        if self.orbit is not None:
            return self.orbit.declination_rad(julian)
        degrad = np.pi / 180.0
        return np.arcsin(
            np.sin(self.obliquity_deg * degrad)
            * np.sin(wrf_solar_longitude_deg(julian) * degrad))

    def cosine_zenith(self, latitude_deg, longitude_deg, valid_time: datetime, *,
                      hour_offset_seconds: float = 0.0, xp=np, dtype=None,
                      start_time: datetime | None = None):
        """The cosine of the solar zenith angle on the given planes.

        Operation for operation ``RRTMGPRadiation._cosine_zenith`` with the
        declination taken from :meth:`declination_rad`: ``hour_offset_seconds``
        shifts the hour angle only (WRF's mid-interval call), ``xp`` is the
        array module of the planes (CuPy in the radiation, NumPy on the CPU)
        and ``dtype`` the type the two declination factors are cast to
        (the radiation's float32; float64 when None).  ``start_time`` is the
        forecast's start instant, read only when the day is not 24 hours."""
        dtype = np.float64 if dtype is None else dtype
        hour, julian = wrf_clock(valid_time)
        declination = self.declination_rad(julian)
        equation = self.equation_of_time_min(julian)
        solar_minutes = 60.0 * (hour + hour_offset_seconds / 3600.0)
        if self.day_length_h != EARTH_DAY_LENGTH_H:
            solar_minutes = solar_minutes + self._extra_minutes(
                valid_time, start_time, hour_offset_seconds)
        solar_minutes = solar_minutes + equation + 4.0 * longitude_deg
        hour_angle = xp.deg2rad(solar_minutes / 4.0 - 180.0)
        latitude = xp.deg2rad(latitude_deg)
        mu = (xp.sin(latitude) * dtype(np.sin(declination))
              + xp.cos(latitude) * dtype(np.cos(declination))
              * xp.cos(hour_angle))
        return xp.clip(mu, dtype(-1.0), dtype(1.0))

    def subsolar_point(self, valid_time: datetime,
                       start_time: datetime | None = None) -> tuple[float, float]:
        """The latitude and longitude, in degrees, where the sun stands
        overhead at ``valid_time`` (longitude in [-180, 180)).  The point
        where this sun's hour angle is zero (``start_time`` as in
        :meth:`cosine_zenith`)."""
        hour, julian = wrf_clock(valid_time)
        latitude = math.degrees(float(self.declination_rad(julian)))
        equation = float(self.equation_of_time_min(julian))
        extra = self._extra_minutes(valid_time, start_time, 0.0)
        longitude = 180.0 - (60.0 * hour + extra + equation) / 4.0
        longitude = (longitude + 180.0) % 360.0 - 180.0
        return latitude, longitude

    def insolation(self, latitude_deg, longitude_deg, valid_time: datetime,
                   model_solar_constant: float, *, xp=np,
                   start_time: datetime | None = None):
        """Top-of-atmosphere sunlight in W m-2 on the given planes: the
        solar constant times the cosine of the zenith angle where the sun
        is up, zero where it is down (float64, for checks)."""
        mu = self.cosine_zenith(latitude_deg, longitude_deg, valid_time, xp=xp,
                                start_time=start_time)
        return self.solar_constant(model_solar_constant, valid_time) * xp.maximum(mu, 0.0)


__all__ = [
    "EARTH_BRIGHTNESS", "EARTH_DAY_LENGTH_H", "EARTH_OBLIQUITY_DEG",
    "NOT_BUILT_HERE", "WorldSun",
    "wrf_clock", "wrf_equation_of_time_min", "wrf_solar_longitude_deg",
]
