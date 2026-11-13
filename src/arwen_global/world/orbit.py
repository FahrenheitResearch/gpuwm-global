"""More orbits for a world's sun: the perpetual equinox and an eccentric orbit.

The model's own sun (``orbit = "earth"``) is WRF v4.6.1's ``radconst``: a
365-day year with the March equinox at zero-based julian 80, a declination
that advances uniformly in time, Spencer's equation of time and Paltridge
and Platt's eccentricity factor on a 1370 W m-2 solar constant.  This module
adds the two other orbits the ``[world.sun]`` table names, in closed form,
on the same 365-day year and the same julian clock:

``perpetual-equinox``
    The sun stays over the equator all year (declination 0), at the orbit's
    mean distance (no eccentricity factor) and with no equation of time:
    the sun crosses each meridian at local noon by the clock, every day.
    This is the published aquaplanet's sun (Neale and Hoskins 2000): a
    circular orbit with no tilt, keeping the diurnal cycle.

``eccentric``
    A Kepler ellipse of eccentricity ``e``.  The year is still 365 days of
    the model's clock and the MEAN sun is still WRF's: the mean longitude
    ``L = M + perihelion`` is WRF's uniform solar longitude,
    ``360 (julian - 80) / 365`` degrees, so a date keeps its meaning against
    the model's calendar and a circular orbit (``e = 0``) moves the sun
    exactly as WRF does.  What the ellipse changes is how far ahead of or
    behind the mean sun the true sun runs (by up to about ``2 e`` radians,
    so the equinoxes and solstices move by up to about ``2 e`` years over
    ``2 pi``; Earth's March equinox lands near julian 78, as it does on
    Earth) and how far away it is.  The mean anomaly is
    ``M = 2 pi (julian - 80) / 365 - perihelion``; Kepler's
    equation ``M = E - e sin E`` is solved by Newton's method to rounding;
    the true anomaly gives the sun's longitude along its apparent path,
    ``lambda = nu + perihelion``; the declination is
    ``asin(sin(tilt) sin(lambda))`` and the distance factor
    ``(a / r)**2 = 1 / (1 - e cos E)**2``.  The equation of time is the
    closed form ``4 (L - alpha)`` minutes, the mean sun's longitude
    ``L = M + perihelion`` less the true sun's right ascension
    ``alpha = atan2(cos(tilt) sin(lambda), cos(lambda))``, so both of its
    parts (the ellipse and the tilt) come from this world's own numbers.

``perihelion_deg`` is where the sun stands on its apparent yearly path at
closest approach, measured from the March equinox: 0 puts closest approach
at the March equinox, 90 at the June solstice, 270 at the December
solstice.  Earth today is about 283 (closest approach in early January).
The paleoclimate convention's longitude of perihelion (Berger 1978, about
103 degrees for Earth today) is this number less 180.

The model's 1370 W m-2 is read as the flux at the orbit's mean distance
(the semi-major axis), times the world's brightness.  An eccentric orbit is
labelled by its flux at closest approach, ``1370 b / (1 - e)**2``, beside
its flux farthest out, ``1370 b / (1 + e)**2``, and its year mean,
``1370 b / sqrt(1 - e**2)`` (the time mean of ``(a / r)**2`` over a Kepler
orbit).

Every function here takes the radiation's zero-based fractional julian day
(``arwen_global.world.sun.wrf_clock``) and returns NumPy float64 scalars, the
types the radiation's own sun hands its planes.  At Earth's eccentricity
and perihelion the eccentric orbit agrees with WRF's sun to the tolerances
``tests/test_arwen_global_world_orbit.py`` states; it is not bit for bit,
because WRF's sun is a set of fitted series and this one is the ellipse.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np

#: WRF ``radconst``: the March equinox at zero-based julian 80 of a 365-day
#: year (``RRTMGPRadiation._cosine_zenith``).  Here it is where the MEAN sun
#: crosses the equinox; the true sun of an eccentric orbit runs ahead of or
#: behind it.
EQUINOX_JULIAN = 80.0
YEAR_DAYS = 365.0
#: The model's solar constant before WRF's eccentricity factor
#: (``RRTMGPRadiation._solar_constant`` is ``1370.0 * eccfac``); read here as
#: the flux at the orbit's mean distance.  A test pins it to the parent.
MEAN_DISTANCE_SOLAR_CONSTANT_W_M2 = 1370.0

#: The orbits this module builds ("earth" is the model's own WRF sun).
ORBIT_KINDS = ("perpetual-equinox", "eccentric")

#: Earth's orbit today, for comparisons only: the J2000 eccentricity and
#: the sun's apparent longitude at perihelion (the heliocentric longitude of
#: perihelion, 102.9373 degrees at J2000, plus 180).
EARTH_ECCENTRICITY = 0.0167086
EARTH_PERIHELION_DEG = 282.9373

_TWO_PI = 2.0 * math.pi
_KEPLER_TOLERANCE_RAD = 1.0e-15
_KEPLER_MAX_ITERATIONS = 50


class OrbitState(NamedTuple):
    """Where the sun is at one instant of an orbit."""

    mean_anomaly_rad: float
    eccentric_anomaly_rad: float
    true_anomaly_rad: float
    solar_longitude_deg: float      # along the apparent path, from the March equinox, [0, 360)
    distance_factor: float          # (a / r)**2: the flux over its mean-distance value
    declination_rad: np.float64
    equation_of_time_min: np.float64


def solve_kepler(mean_anomaly_rad: float, eccentricity: float) -> float:
    """The eccentric anomaly ``E`` with ``E - e sin E = M``, by Newton's
    method from ``M + e sin M``, to rounding; ``E`` is returned in the same
    turn as ``M`` reduced to [-pi, pi]."""
    e = float(eccentricity)
    m = math.remainder(float(mean_anomaly_rad), _TWO_PI)
    if e == 0.0:
        return m
    big_e = m + e * math.sin(m)
    for _ in range(_KEPLER_MAX_ITERATIONS):
        step = (big_e - e * math.sin(big_e) - m) / (1.0 - e * math.cos(big_e))
        big_e -= step
        if abs(step) <= _KEPLER_TOLERANCE_RAD:
            break
    return big_e


def _mean_anomaly_of_true(true_anomaly_rad: float, e: float) -> float:
    eccentric = math.atan2(math.sqrt(1.0 - e * e) * math.sin(true_anomaly_rad),
                           e + math.cos(true_anomaly_rad))
    return eccentric - e * math.sin(eccentric)


@dataclass(frozen=True)
class Orbit:
    """One of the two orbits of :data:`ORBIT_KINDS`, with the world's tilt.

    Build it with :meth:`perpetual_equinox`, :meth:`eccentric` or
    :func:`orbit_of` (from a parsed world).  Ranges are the parser's; the
    checks here are for a direct caller."""

    kind: str
    obliquity_deg: float = 0.0
    eccentricity: float = 0.0
    perihelion_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ORBIT_KINDS:
            raise ValueError(f"an orbit is one of {ORBIT_KINDS}, got {self.kind!r}")
        tilt, e, peri = (float(self.obliquity_deg), float(self.eccentricity),
                         float(self.perihelion_deg))
        if self.kind == "perpetual-equinox" and (tilt, e, peri) != (0.0, 0.0, 0.0):
            raise ValueError(
                "the perpetual-equinox orbit has no tilt, no eccentricity and no "
                "perihelion: the sun stays over the equator at the mean distance")
        if not math.isfinite(tilt) or not 0.0 <= tilt <= 180.0:
            raise ValueError(f"an orbit's obliquity_deg must be 0 to 180, got {tilt!r}")
        if not math.isfinite(e) or not 0.0 <= e < 1.0:
            raise ValueError(f"an orbit's eccentricity must be 0 to below 1, got {e!r}")
        if not math.isfinite(peri) or not 0.0 <= peri < 360.0:
            raise ValueError(f"an orbit's perihelion_deg must be 0 to below 360, got {peri!r}")
        object.__setattr__(self, "obliquity_deg", tilt)
        object.__setattr__(self, "eccentricity", e)
        object.__setattr__(self, "perihelion_deg", peri)

    @classmethod
    def perpetual_equinox(cls) -> "Orbit":
        return cls("perpetual-equinox")

    @classmethod
    def eccentric(cls, eccentricity: float, perihelion_deg: float = 0.0, *,
                  obliquity_deg: float = 23.5) -> "Orbit":
        return cls("eccentric", obliquity_deg=obliquity_deg,
                   eccentricity=eccentricity, perihelion_deg=perihelion_deg)

    # -- the year -------------------------------------------------------------

    @staticmethod
    def mean_longitude_rad(julian: float) -> float:
        """WRF's uniform solar longitude, the mean sun of every orbit here."""
        return _TWO_PI * (float(julian) - EQUINOX_JULIAN) / YEAR_DAYS

    def mean_anomaly_rad(self, julian: float) -> float:
        return self.mean_longitude_rad(julian) - math.radians(self.perihelion_deg)

    def julian_of_longitude(self, longitude_deg: float) -> float:
        """The zero-based julian day, in [0, 365), when the true sun stands
        at ``longitude_deg`` from the March equinox (0: the March equinox,
        90: the June solstice, 180: the September equinox)."""
        e = self.eccentricity
        nu = math.radians(float(longitude_deg) - self.perihelion_deg)
        mean_longitude = _mean_anomaly_of_true(nu, e) + math.radians(self.perihelion_deg)
        return (EQUINOX_JULIAN + YEAR_DAYS * mean_longitude / _TWO_PI) % YEAR_DAYS

    def closest_approach_julian(self) -> float:
        """The zero-based julian day, in [0, 365), of closest approach
        (where the mean anomaly is a whole turn).  For the perpetual
        equinox, whose distance never changes, 0."""
        if self.kind == "perpetual-equinox":
            return 0.0
        return (EQUINOX_JULIAN + YEAR_DAYS * self.perihelion_deg / 360.0) % YEAR_DAYS

    def state(self, julian: float) -> OrbitState:
        """The sun's place at zero-based fractional julian day ``julian``."""
        zero = np.float64(0.0)
        if self.kind == "perpetual-equinox":
            # Exact zeros: no tilt, no ellipse, so no declination and no
            # equation of time (the general path would leave rounding).
            m = math.remainder(_TWO_PI * (float(julian) - EQUINOX_JULIAN) / YEAR_DAYS,
                               _TWO_PI)
            return OrbitState(m, m, m, math.degrees(m) % 360.0, 1.0, zero, zero)
        e = self.eccentricity
        m = self.mean_anomaly_rad(julian)
        big_e = solve_kepler(m, e)
        nu = math.atan2(math.sqrt(1.0 - e * e) * math.sin(big_e), math.cos(big_e) - e)
        peri = math.radians(self.perihelion_deg)
        lam = nu + peri
        tilt = np.deg2rad(np.float64(self.obliquity_deg))
        declination = np.arcsin(np.sin(tilt) * np.sin(np.float64(lam)))
        alpha = math.atan2(math.cos(float(tilt)) * math.sin(lam), math.cos(lam))
        equation = np.float64(4.0 * math.degrees(math.remainder(m + peri - alpha, _TWO_PI)))
        distance = 1.0 / (1.0 - e * math.cos(big_e)) ** 2
        return OrbitState(m, big_e, nu, math.degrees(lam) % 360.0, distance,
                          declination, equation)

    def declination_rad(self, julian: float) -> np.float64:
        return self.state(julian).declination_rad

    def equation_of_time_min(self, julian: float) -> np.float64:
        return self.state(julian).equation_of_time_min

    def distance_factor(self, julian: float) -> float:
        """``(a / r)**2``: the flux at this instant over the mean-distance flux."""
        return self.state(julian).distance_factor

    # -- the label --------------------------------------------------------------

    @property
    def closest_approach_factor(self) -> float:
        return 1.0 / (1.0 - self.eccentricity) ** 2

    @property
    def farthest_factor(self) -> float:
        return 1.0 / (1.0 + self.eccentricity) ** 2

    @property
    def year_mean_factor(self) -> float:
        """The time mean of ``(a / r)**2`` over one Kepler orbit."""
        return 1.0 / math.sqrt(1.0 - self.eccentricity ** 2)

    def fluxes_w_m2(self, brightness: float = 1.0) -> dict[str, float]:
        """Top-of-atmosphere flux facing the sun: at closest approach,
        farthest out and the year's mean, for a world of ``brightness``."""
        base = MEAN_DISTANCE_SOLAR_CONSTANT_W_M2 * float(brightness)
        return {
            "closest_approach_w_m2": base * self.closest_approach_factor,
            "farthest_w_m2": base * self.farthest_factor,
            "year_mean_w_m2": base * self.year_mean_factor,
        }

    def describe(self, brightness: float = 1.0) -> str:
        """One line for the world statement; an eccentric orbit leads with
        its flux at closest approach."""
        if self.kind == "perpetual-equinox":
            flux = MEAN_DISTANCE_SOLAR_CONSTANT_W_M2 * float(brightness)
            return (f"perpetual equinox: the sun stays over the equator at "
                    f"{flux:.0f} W m-2 all year, no seasons, no equation of time")
        f = self.fluxes_w_m2(brightness)
        return (f"eccentric orbit: {f['closest_approach_w_m2']:.0f} W m-2 at closest "
                f"approach (day {self.closest_approach_julian() + 1.0:.0f} of 365, sun "
                f"at {self.perihelion_deg:g} deg from the March equinox), "
                f"{f['farthest_w_m2']:.0f} W m-2 farthest out, "
                f"{f['year_mean_w_m2']:.0f} W m-2 year mean; e = {self.eccentricity:g}")

    def record(self, brightness: float = 1.0) -> dict[str, object]:
        """What a receipt, a tape or the world statement stores about it."""
        out: dict[str, object] = {"orbit": self.kind, "year_days": YEAR_DAYS,
                                  "equinox_julian": EQUINOX_JULIAN}
        if self.kind == "eccentric":
            out.update({
                "eccentricity": self.eccentricity,
                "perihelion_deg": self.perihelion_deg,
                "closest_approach_julian": self.closest_approach_julian(),
            })
        out.update({k: round(v, 3) for k, v in self.fluxes_w_m2(brightness).items()})
        return out


def orbit_of(world_or_sun) -> Orbit | None:
    """The orbit of a parsed world (``WorldSpec``) or its ``world.sun``;
    ``None`` for the model's own (``orbit = "earth"``)."""
    sun = getattr(world_or_sun, "sun", world_or_sun)
    if sun.orbit == "earth":
        return None
    if sun.orbit == "perpetual-equinox":
        return Orbit.perpetual_equinox()
    if sun.orbit == "eccentric":
        return Orbit.eccentric(sun.eccentricity, sun.perihelion_deg or 0.0,
                               obliquity_deg=sun.obliquity_deg)
    raise ValueError(f"world.sun.orbit = {sun.orbit!r} is not an orbit this model builds")


__all__ = [
    "EARTH_ECCENTRICITY", "EARTH_PERIHELION_DEG", "EQUINOX_JULIAN",
    "MEAN_DISTANCE_SOLAR_CONSTANT_W_M2", "ORBIT_KINDS", "Orbit", "OrbitState",
    "YEAR_DAYS", "orbit_of", "solve_kepler",
]
