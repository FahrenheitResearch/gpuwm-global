"""A world's day: the sun's hour angle and the planet's rotation, from one number.

``[world.sun] day_length_h`` is the SOLAR day, noon to noon, in hours.  Two
things in the model follow it and must never disagree (plan 4.6):

* the hour angle of the sun, which turns once per solar day relative to the
  ground, so it advances ``2 pi / day`` radians per second;
* the rotation rate handed to the dynamics (the Coriolis parameter
  ``2 Omega sin(latitude)``), which is the spin relative to the stars: the
  solar rate plus the rate at which the sun moves along the orbit.

Both come from :func:`planet_rates`.  The orbital rate is the model's own:
``EARTH_ROTATION_RATE_S - 2 pi / 86400``, the difference between the
dynamics' Earth spin (7.292e-5 s-1) and its 24-hour solar day, so a 24-hour
day returns ``EARTH_ROTATION_RATE_S`` itself and any other day keeps the
same year.  This is not ``24 / day_length_h`` times Earth's spin
(``Sun.rotation_factor``): for a 12-hour day the two differ by 0.14 percent,
for a 2,400-hour day by 27 percent, because a slow spinner's solar day is
set mostly by its orbit.  The solar day is what the user typed, so the spin
is derived from it and the sun stays where the dynamics' rotation puts it.

THE CLOCK.  Time labels do not change (plan 4.6): a tape's valid time is the
start instant plus elapsed seconds in the ordinary calendar, so the
declination, the equation of time, the ozone and vegetation calendars and
the solar constant stay on Earth's dates.  What changes is the hour angle:
at the forecast's start instant every world's sun stands where Earth's
does, and from then on it turns ``360 x hours elapsed x (1 / day_length_h -
1 / 24)`` degrees further round than Earth's (westward when the day is
shorter).  :meth:`PlanetClock.extra_turn_deg` is that angle; the radiation
adds it to WRF's hour angle and the tapes subtract it from Earth's sub-solar
longitude, so the two agree by construction.

THE RADIATION INTERVAL.  Radiation is called every ``radiation_interval_s``.
The shipped T255 configs call it every 3,000 s, 12.5 degrees of sun
movement per call on Earth and 25 on a 12-hour day.
:meth:`PlanetClock.radiation_interval_advice` gives the interval that
restores the config's own degrees per call, rounded down to a whole number
of steps; it never asks for a longer one.

LOCAL TIME.  Two instruments bin by local solar time ``UTC + longitude /
15`` on a 24-hour day (:mod:`arwen_global.diurnal_phase`,
:mod:`arwen_global.trigger_diagnostic`).  On a world with another day
length that hour is not the sun's, so both refuse such a run by name
(:func:`refuse_local_time`) rather than print a phase of the wrong day.

Nothing here runs for Earth: a world whose day is 24 hours never builds a
:class:`PlanetClock`, and every reader below returns 24 for a record or a
receipt written before worlds existed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..spectral.constants import EARTH_ROTATION_RATE_S
from .spec import EARTH_DAY_LENGTH_H

#: Seconds in Earth's solar day, the day every calendar label counts.
EARTH_DAY_S = 86400.0
#: The sun's motion along the model's orbit, in rad/s: the dynamics' Earth
#: spin less its 24-hour solar rate (about 1.98e-7, a 367-day year).
ORBITAL_RATE_S = EARTH_ROTATION_RATE_S - 2.0 * math.pi / EARTH_DAY_S

#: The two instruments that bin by local solar time on a 24-hour day.
LOCAL_TIME_INSTRUMENTS = ("diurnal_phase", "trigger_diagnostic")


class LocalTimeRefused(ValueError):
    """A local-solar-time instrument was pointed at a world whose day is not
    24 hours."""


def planet_rates(day_length_h: float) -> tuple[float, float]:
    """``(solar_rate, rotation_rate)`` in rad/s for a solar day of
    ``day_length_h`` hours: how fast the sun's hour angle turns, and the
    spin the dynamics' Coriolis term uses.  THE one function both follow.

    A 24-hour day returns ``(2 pi / 86400, EARTH_ROTATION_RATE_S)`` exactly,
    so Earth's Coriolis parameter is the model's own to the bit."""
    day = float(day_length_h)
    if not math.isfinite(day) or day <= 0.0:
        raise ValueError(f"a day length must be a positive number of hours, got {day_length_h!r}")
    if day == EARTH_DAY_LENGTH_H:
        return 2.0 * math.pi / EARTH_DAY_S, EARTH_ROTATION_RATE_S
    solar = 2.0 * math.pi / (day * 3600.0)
    return solar, solar + ORBITAL_RATE_S


@dataclass(frozen=True)
class PlanetClock:
    """The day of one world.  Built only for a world whose day is not 24
    hours (:meth:`from_world` returns None otherwise)."""

    day_length_h: float = EARTH_DAY_LENGTH_H

    def __post_init__(self) -> None:
        planet_rates(self.day_length_h)  # refuses a non-positive day
        object.__setattr__(self, "day_length_h", float(self.day_length_h))

    @classmethod
    def from_world(cls, world) -> "PlanetClock | None":
        """The clock of a parsed world, or None when its day is Earth's
        (including Earth itself), so no world code runs."""
        if world is None or world.is_earth or not world.changes_rotation:
            return None
        return cls(day_length_h=world.sun.day_length_h)

    @property
    def is_earth(self) -> bool:
        return self.day_length_h == EARTH_DAY_LENGTH_H

    @property
    def day_s(self) -> float:
        return self.day_length_h * 3600.0

    @property
    def solar_rate_s(self) -> float:
        return planet_rates(self.day_length_h)[0]

    @property
    def rotation_rate_s(self) -> float:
        """The spin handed to the dynamics (``MoistHybridModel.rotation_rate_s``)."""
        return planet_rates(self.day_length_h)[1]

    @property
    def sidereal_day_h(self) -> float:
        return 2.0 * math.pi / self.rotation_rate_s / 3600.0

    # -- the hour angle ------------------------------------------------------

    def extra_turn_deg(self, elapsed_s: float) -> float:
        """How many degrees further round this world's sun has turned than
        Earth's, ``elapsed_s`` seconds after the forecast's start instant,
        reduced to [0, 360).  Added to WRF's hour angle by the radiation;
        subtracted from Earth's sub-solar longitude by the tapes.  Zero for
        a 24-hour day."""
        if self.is_earth:
            return 0.0
        solar = planet_rates(self.day_length_h)[0]
        earth = 2.0 * math.pi / EARTH_DAY_S
        turn = math.degrees((solar - earth) * float(elapsed_s))
        return turn % 360.0

    def hour_angle_deg(self, earth_hour_angle_deg, elapsed_s: float):
        """This world's hour angle from Earth's at the same valid time
        (scalar or array; the array's own type is kept)."""
        return earth_hour_angle_deg + self.extra_turn_deg(elapsed_s)

    def subsolar_longitude_deg(self, earth_longitude_deg: float, elapsed_s: float) -> float:
        """Where this world's sun is overhead, in [-180, 180), from where
        Earth's is at the same valid time."""
        longitude = float(earth_longitude_deg) - self.extra_turn_deg(elapsed_s)
        return (longitude + 180.0) % 360.0 - 180.0

    def local_solar_hour(self, earth_local_hour, elapsed_s: float):
        """The local hour on this world's own clock (0 to ``day_length_h``,
        noon at half the day) from Earth's local solar hour at the same
        place and valid time."""
        earth_angle = (earth_local_hour / 24.0) * 360.0
        angle = (earth_angle + self.extra_turn_deg(elapsed_s)) % 360.0
        return angle / 360.0 * self.day_length_h

    # -- the radiation interval ----------------------------------------------

    def radiation_interval_advice(self, interval_s: float, dt_s: float) -> dict[str, object] | None:
        """The radiation interval that moves this world's sun no further per
        call than ``interval_s`` moves Earth's, rounded down to whole steps
        of ``dt_s`` (one step at least), or None when the configured
        interval already does (a day of 24 hours or longer) or is already
        a single step."""
        interval = float(interval_s)
        dt = float(dt_s)
        if interval <= 0.0 or dt <= 0.0:
            raise ValueError("the radiation interval and the step must be positive")
        earth_deg = 360.0 * interval / EARTH_DAY_S
        world_deg = 360.0 * interval / self.day_s
        if world_deg <= earth_deg:
            return None
        target = interval * self.day_length_h / EARTH_DAY_LENGTH_H
        steps = max(1, int(math.floor(target / dt + 1e-9)))
        recommended = steps * dt
        if recommended >= interval:
            # Already one step: nothing shorter to recommend.
            return None
        return {
            "radiation_interval_s": interval,
            "degrees_per_call": world_deg,
            "earth_degrees_per_call": earth_deg,
            "recommended_radiation_interval_s": recommended,
            "recommended_degrees_per_call": 360.0 * recommended / self.day_s,
            "sentence": (
                f"on a {self.day_length_h:g}-hour day radiation every {interval:g} s "
                f"moves the sun {world_deg:.1f} degrees per call against Earth's "
                f"{earth_deg:.1f}; radiation_interval_s = {recommended:g} "
                f"({steps} steps of {dt:g} s) restores {360.0 * recommended / self.day_s:.1f}"),
        }

    def record(self) -> dict[str, object]:
        """What a receipt may store about this clock."""
        return {
            "day_length_h": self.day_length_h,
            "rotation_rate_s": self.rotation_rate_s,
            "sidereal_day_h": self.sidereal_day_h,
            "orbital_rate_s": ORBITAL_RATE_S,
            "rule": ("solar day as typed; spin = 2 pi / day + the model's orbital "
                     "rate; the sun turns from Earth's position at the start instant"),
        }


def rotation_rate_of(world) -> float:
    """The spin the dynamics runs under ``world`` (Earth's constant for Earth
    and for any world whose day is 24 hours)."""
    clock = PlanetClock.from_world(world)
    return EARTH_ROTATION_RATE_S if clock is None else clock.rotation_rate_s


# -- the two local-time instruments ----------------------------------------


def record_day_length_h(record) -> float:
    """The day length a world record states (``WorldSpec.record()``, as a
    checkpoint, a receipt or a tape stores it); 24 for None, for Earth's
    record and for anything written before worlds existed."""
    if not isinstance(record, dict):
        return EARTH_DAY_LENGTH_H
    settings = record.get("settings")
    if not isinstance(settings, dict):
        return EARTH_DAY_LENGTH_H
    sun = settings.get("sun")
    if not isinstance(sun, dict) or sun.get("day_length_h") is None:
        return EARTH_DAY_LENGTH_H
    return float(sun["day_length_h"])


def receipt_day_length_h(receipt) -> float:
    """The day length of the world a run receipt names: its ``world`` record,
    else the world identity in its ``config`` block (``config_identity``
    carries ``world`` only for a non-Earth world, which is also what the
    instruments' config-shaped stand-in receipt holds); 24 when it names
    none."""
    if not isinstance(receipt, dict):
        return EARTH_DAY_LENGTH_H
    if isinstance(receipt.get("world"), dict):
        return record_day_length_h(receipt["world"])
    config = receipt.get("config")
    identity = config.get("world") if isinstance(config, dict) else None
    if isinstance(identity, dict):
        return record_day_length_h({"settings": identity})
    return EARTH_DAY_LENGTH_H


def checkpoint_day_length_h(path) -> float:
    """The day length of the world a checkpoint was integrated under, read
    from its metadata alone (``physics_metadata["world"]``); 24 for a
    checkpoint of Earth or one written before worlds existed."""
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            return EARTH_DAY_LENGTH_H
        metadata = json.loads(str(archive["__metadata__"].item()))
    return metadata_day_length_h(metadata)


def metadata_day_length_h(metadata) -> float:
    """The day length in a checkpoint's metadata dict (see above)."""
    if not isinstance(metadata, dict):
        return EARTH_DAY_LENGTH_H
    physics = metadata.get("physics_metadata")
    if not isinstance(physics, dict):
        return EARTH_DAY_LENGTH_H
    return record_day_length_h(physics.get("world"))


def refuse_local_time(instrument: str, day_length_h: float, where: str) -> None:
    """Refuse, by name, a local-solar-time instrument on a run whose day is
    not 24 hours.  Returns quietly for a 24-hour day."""
    if float(day_length_h) == EARTH_DAY_LENGTH_H:
        return
    raise LocalTimeRefused(
        f"{instrument} refused: {where} was integrated on a world whose day is "
        f"{float(day_length_h):g} hours, and this instrument bins by local solar "
        "time on a 24-hour day (UTC + longitude / 15), which on that world is "
        "not where the sun is; its phases and its noon window would describe "
        "a day the run never had (arwen_global.world.clock)")


__all__ = [
    "EARTH_DAY_S", "LOCAL_TIME_INSTRUMENTS", "LocalTimeRefused", "ORBITAL_RATE_S",
    "PlanetClock", "checkpoint_day_length_h", "metadata_day_length_h", "planet_rates", "receipt_day_length_h",
    "record_day_length_h", "refuse_local_time", "rotation_rate_of",
]
