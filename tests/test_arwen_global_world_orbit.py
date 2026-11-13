"""More orbits for a world's sun (worlds plan, package 3.4).

On the CPU, closed forms only:

* Kepler's equation is solved to rounding at every eccentricity the table
  accepts; closest approach is where the sun's longitude is
  ``perihelion_deg`` and the flux there is ``1 / (1 - e)**2`` of the mean;
  the year is 365 days and the March equinox stays at julian 80;
* the time mean of ``(a / r)**2`` is ``1 / sqrt(1 - e**2)``, and the annual
  mean sunlight at the pole is ``S sin(tilt) / (pi sqrt(1 - e**2))`` at any
  perihelion;
* the perpetual equinox has no declination, no equation of time and no
  change of distance, so a day's mean sunlight at latitude ``phi`` is
  ``S cos(phi) / pi``;
* at Earth's eccentricity and perihelion the eccentric orbit agrees with
  the model's own WRF sun (``RRTMGPRadiation``) to the tolerances below,
  which are the largest differences measured over a whole year with a
  margin; the orbit is the ellipse, WRF's sun a set of fitted series, so
  the agreement is a tolerance and not bit for bit.

The tests of the last section need ``arwen_global.world.sun`` with the
orbits wired in (package 1.3's module with this package's patch); they skip
by name where that module is absent or still refuses the orbits.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import sys

import numpy as np
import pytest

from arwen_global.world import EARTH, parse_world
from arwen_global.world.orbit import (
    EARTH_ECCENTRICITY, EARTH_PERIHELION_DEG, EQUINOX_JULIAN,
    MEAN_DISTANCE_SOLAR_CONSTANT_W_M2, YEAR_DAYS, Orbit, orbit_of, solve_kepler)

S0 = MEAN_DISTANCE_SOLAR_CONSTANT_W_M2

#: Tolerances of the eccentric orbit at Earth's values (e 0.0167086, sun at
#: 282.9373 deg at closest approach, tilt 23.5) against the model's WRF sun
#: over 2026, each the largest difference this test measures, rounded up
#: (the measured values are in the comments): the solar constant (relative),
#: the declination (degrees), the equation of time (minutes), the cosine of
#: the zenith angle on a 32 x 64 grid every 6 hours, a day's mean
#: top-of-atmosphere sunlight by latitude every fifth day (W m-2), and the
#: mean of those days by latitude (W m-2).  The declination gap is WRF's: its
#: sun moves uniformly from julian 80, so it reaches the September equinox
#: about three days early; against Spencer's (1971) fit of Earth's
#: declination the ellipse is 0.49 degrees off and WRF's sun 1.25.  The
#: day-mean gap follows from it at high latitude near the equinoxes, where a
#: day's sunlight moves 24 W m-2 per degree of declination.
EARTH_TOLERANCE = {
    "solar_constant_rel": 1.0e-3,    # measured 8.54e-4
    "declination_deg": 0.8,          # measured 0.755
    "equation_of_time_min": 1.4,     # measured 1.265
    "cosine_zenith": 1.5e-2,         # measured 0.0138
    "daily_mean_w_m2": 20.0,         # measured 17.87
    "annual_mean_w_m2": 5.0,         # measured 4.58
}


def _world(text: str):
    import tomllib

    return parse_world(tomllib.loads(text)["world"])


def _earth_orbit() -> Orbit:
    return Orbit.eccentric(EARTH_ECCENTRICITY, EARTH_PERIHELION_DEG, obliquity_deg=23.5)


def _year_julians(per_day: int):
    return (np.arange(int(YEAR_DAYS) * per_day) + 0.5) / per_day


# ------------------------------------------------------------ Kepler, closed forms


@pytest.mark.parametrize("e", [0.0, 0.0167086, 0.1, 0.2, 0.3])
def test_keplers_equation_is_solved_to_rounding(e):
    for m in np.linspace(-math.pi, math.pi, 2001):
        big_e = solve_kepler(float(m), e)
        assert abs(big_e - e * math.sin(big_e) - m) <= 4.0e-15


@pytest.mark.parametrize("e, peri", [(0.0167086, 282.9373), (0.1, 0.0), (0.3, 200.0),
                                     (0.05, 90.0)])
def test_closest_approach_is_at_the_perihelion_longitude_with_its_flux(e, peri):
    orbit = Orbit.eccentric(e, peri, obliquity_deg=30.0)
    near = orbit.closest_approach_julian()
    at = orbit.state(near)
    assert math.remainder(at.solar_longitude_deg - peri, 360.0) == pytest.approx(0.0, abs=1e-9)
    assert at.distance_factor == pytest.approx(1.0 / (1.0 - e) ** 2, rel=1e-12)
    far = orbit.state(near + YEAR_DAYS / 2.0)
    assert far.distance_factor == pytest.approx(1.0 / (1.0 + e) ** 2, rel=1e-12)
    assert math.remainder(far.solar_longitude_deg - peri - 180.0, 360.0) == pytest.approx(
        0.0, abs=1e-9)
    # The flux is largest at closest approach and smallest half a year on.
    flux = [orbit.distance_factor(j) for j in _year_julians(4)]
    assert max(flux) <= at.distance_factor * (1.0 + 1e-12)
    assert min(flux) >= far.distance_factor * (1.0 - 1e-12)


@pytest.mark.parametrize("e, peri", [(0.0167086, 282.9373), (0.3, 45.0), (0.0, 0.0),
                                     (0.0, 123.0)])
def test_the_year_is_365_days_and_the_mean_sun_is_wrfs(e, peri):
    orbit = Orbit.eccentric(e, peri, obliquity_deg=23.5)
    dpd = 360.0 / 365.0
    for j in (0.0, 33.3, 79.5, 80.0, 200.7, 364.9):
        state = orbit.state(j)
        mean_longitude = math.degrees(state.mean_anomaly_rad) + peri
        wrf = dpd * (j - 80.0) if j >= 80.0 else dpd * (j + 285.0)
        assert math.remainder(mean_longitude - wrf, 360.0) == pytest.approx(0.0, abs=1e-9)
    # The true sun at the equinoxes and solstices, found in closed form.
    for lam in (0.0, 90.0, 180.0, 270.0):
        j = orbit.julian_of_longitude(lam)
        assert math.remainder(orbit.state(j).solar_longitude_deg - lam, 360.0) == pytest.approx(
            0.0, abs=1e-8)
    if e == 0.0:
        # A circle: the equinox is WRF's, julian 80, with no declination.
        assert orbit.julian_of_longitude(0.0) == pytest.approx(EQUINOX_JULIAN, abs=1e-9)
        assert abs(float(orbit.state(EQUINOX_JULIAN).declination_rad)) <= 1e-15
    for j in (0.0, 33.3, 80.0, 200.7, 364.9):
        a, b = orbit.state(j), orbit.state(j + YEAR_DAYS)
        assert b.distance_factor == pytest.approx(a.distance_factor, rel=1e-12)
        assert float(b.declination_rad) == pytest.approx(float(a.declination_rad), abs=1e-12)
        assert float(b.equation_of_time_min) == pytest.approx(
            float(a.equation_of_time_min), abs=1e-9)


@pytest.mark.parametrize("e", [0.0167086, 0.1, 0.3])
def test_the_year_mean_flux_factor_is_one_over_root_one_minus_e_squared(e):
    orbit = Orbit.eccentric(e, 123.0)
    mean = np.mean([orbit.distance_factor(j) for j in _year_julians(96)])
    assert mean == pytest.approx(1.0 / math.sqrt(1.0 - e * e), rel=1e-10)
    assert orbit.year_mean_factor == 1.0 / math.sqrt(1.0 - e * e)


@pytest.mark.parametrize("e, peri, tilt", [(0.0167086, 282.9373, 23.5), (0.1, 37.0, 23.5),
                                           (0.3, 200.0, 60.0), (0.2, 90.0, 90.0)])
def test_the_annual_mean_sunlight_at_the_pole_is_the_closed_form(e, peri, tilt):
    """At the north pole the sun circles at an elevation equal to its
    declination, so a day's mean sunlight is ``S (a/r)**2 sin(dec)`` while
    it is up.  Over a Kepler year that averages to
    ``S sin(tilt) / (pi sqrt(1 - e**2))``, whatever the perihelion."""
    orbit = Orbit.eccentric(e, peri, obliquity_deg=tilt)
    samples = [orbit.state(j) for j in _year_julians(96)]
    mean = np.mean([s.distance_factor * max(math.sin(float(s.declination_rad)), 0.0)
                    for s in samples]) * S0
    want = S0 * math.sin(math.radians(tilt)) / (math.pi * math.sqrt(1.0 - e * e))
    assert mean == pytest.approx(want, rel=1e-5)


def test_earths_orbit_gives_earths_long_northern_summer_half_year():
    """With Earth's eccentricity and perihelion the sun takes longer over
    the half of its path from the March to the September equinox: 186.4
    days on Earth (2020: 20 March 03:50 to 22 September 13:31 UTC)."""
    orbit = _earth_orbit()
    e = orbit.eccentricity
    peri = math.radians(orbit.perihelion_deg)

    def mean_anomaly_of_longitude(lam):
        nu = lam - peri
        big_e = math.atan2(math.sqrt(1 - e * e) * math.sin(nu), e + math.cos(nu))
        return big_e - e * math.sin(big_e)

    turn = (mean_anomaly_of_longitude(math.pi) - mean_anomaly_of_longitude(0.0)) % (2 * math.pi)
    days = YEAR_DAYS * turn / (2 * math.pi)
    assert days == pytest.approx(186.4, abs=0.15)
    march, september = orbit.julian_of_longitude(0.0), orbit.julian_of_longitude(180.0)
    assert september - march == pytest.approx(days, abs=1e-9)
    # With WRF's mean sun the true equinoxes land within a day of 2026's:
    # 20 March 14:46 UTC (julian 78.615) and 23 September 00:05 UTC (julian
    # 265.004).  WRF's own sun puts them at julian 80 and 262.5.
    assert march == pytest.approx(78.615, abs=1.0)
    assert september == pytest.approx(265.004, abs=1.0)
    # Closest approach early in January (2026: 3 January 17 UTC, julian 2.7).
    assert orbit.closest_approach_julian() == pytest.approx(2.7, abs=1.5)


def test_a_circular_orbit_moves_the_sun_on_wrfs_uniform_longitude():
    orbit = Orbit.eccentric(0.0, 0.0, obliquity_deg=23.5)
    dpd = 360.0 / 365.0
    for j in (0.0, 12.25, 79.99, 80.0, 180.5, 364.75):
        wrf = dpd * (j - 80.0) if j >= 80.0 else dpd * (j + 285.0)
        assert math.remainder(orbit.state(j).solar_longitude_deg - wrf, 360.0) == pytest.approx(
            0.0, abs=1e-10)
        assert orbit.distance_factor(j) == 1.0


# ------------------------------------------------------------ perpetual equinox


def test_the_perpetual_equinox_has_no_season_no_equation_of_time_no_distance_change():
    orbit = Orbit.perpetual_equinox()
    for j in _year_julians(3):
        state = orbit.state(float(j))
        assert state.declination_rad == 0.0
        assert state.equation_of_time_min == 0.0
        assert state.distance_factor == 1.0
        assert type(state.declination_rad) is np.float64
    assert orbit.fluxes_w_m2(1.0) == {
        "closest_approach_w_m2": S0, "farthest_w_m2": S0, "year_mean_w_m2": S0}
    with pytest.raises(ValueError, match="no tilt"):
        Orbit("perpetual-equinox", obliquity_deg=23.5)


def test_the_perpetual_equinox_day_mean_is_s_cos_latitude_over_pi():
    """WRF's hour angle with no declination and no equation of time: a
    day's mean sunlight at latitude phi is S cos(phi) / pi on every day."""
    orbit = Orbit.perpetual_equinox()
    lat = np.deg2rad(np.array([-80.0, -45.0, -10.0, 0.0, 30.0, 60.0, 89.0]))
    lon = 37.5
    steps = 2880
    for day in (0, 80, 172, 300):
        total = np.zeros_like(lat)
        for k in range(steps):
            hour = 24.0 * (k + 0.5) / steps
            julian = day + hour / 24.0
            state = orbit.state(julian)
            minutes = 60.0 * hour + state.equation_of_time_min + 4.0 * lon
            h = np.deg2rad(minutes / 4.0 - 180.0)
            mu = (np.sin(lat) * np.sin(state.declination_rad)
                  + np.cos(lat) * np.cos(state.declination_rad) * np.cos(h))
            total += S0 * state.distance_factor * np.maximum(mu, 0.0)
        np.testing.assert_allclose(total / steps, S0 * np.cos(lat) / np.pi,
                                   rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------ the label


def test_an_eccentric_orbit_is_labelled_by_its_closest_approach_flux():
    orbit = Orbit.eccentric(0.1, 0.0)
    fluxes = orbit.fluxes_w_m2(1.0)
    assert fluxes["closest_approach_w_m2"] == pytest.approx(1370.0 / 0.81)
    assert fluxes["farthest_w_m2"] == pytest.approx(1370.0 / 1.21)
    line = orbit.describe(1.0)
    assert line.startswith("eccentric orbit: 1691 W m-2 at closest approach")
    assert "1132 W m-2 farthest out" in line
    # Closest approach at the March equinox: day 81 of the year.
    assert orbit.closest_approach_julian() == pytest.approx(EQUINOX_JULIAN, abs=1e-9)
    record = orbit.record(1.2)
    assert record["orbit"] == "eccentric"
    assert record["closest_approach_w_m2"] == round(1.2 * 1370.0 / 0.81, 3)
    assert "perpetual equinox" in Orbit.perpetual_equinox().describe()


def test_the_parsed_world_gives_its_orbit():
    assert orbit_of(EARTH) is None
    flat = _world('[world]\nlabel = "eq"\n[world.sun]\norbit = "perpetual-equinox"\n')
    assert orbit_of(flat) == Orbit.perpetual_equinox()
    ecc = _world('[world]\nlabel = "e"\n[world.sun]\norbit = "eccentric"\n'
                 'eccentricity = 0.2\nperihelion_deg = 270.0\nobliquity_deg = 10.0\n')
    assert orbit_of(ecc) == Orbit.eccentric(0.2, 270.0, obliquity_deg=10.0)
    no_peri = _world('[world]\nlabel = "e"\n[world.sun]\norbit = "eccentric"\n'
                     'eccentricity = 0.05\n')
    assert orbit_of(no_peri).perihelion_deg == 0.0


def test_out_of_range_orbits_are_refused():
    with pytest.raises(ValueError, match="one of"):
        Orbit("earth")
    with pytest.raises(ValueError, match="eccentricity"):
        Orbit.eccentric(1.0)
    with pytest.raises(ValueError, match="perihelion_deg"):
        Orbit.eccentric(0.1, 360.0)


# ------------------------------------------------------------ against the model's sun


@pytest.fixture
def numpy_as_cupy(monkeypatch):
    """The radiation imports its array module inside each method; NumPy
    answers for it here (package 1.3's CPU convention)."""
    monkeypatch.setitem(sys.modules, "cupy", np)


def _grid(nlat: int = 32, nlon: int = 64):
    from arwen_global.core.rrtmgp import DTYPE

    lat = (np.arange(nlat) + 0.5) * (180.0 / nlat) - 90.0
    lon = np.arange(nlon) * (360.0 / nlon)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2.astype(DTYPE), lon2.astype(DTYPE)


def _wrf_clock(when: datetime):
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    return hour, when.timetuple().tm_yday - 1.0 + hour / 24.0


def _wrf_declination_deg(julian: float) -> float:
    """WRF ``radconst``'s declination (``RRTMGPRadiation._cosine_zenith``)."""
    dpd = 360.0 / 365.0
    lam = dpd * (julian - 80.0) if julian >= 80.0 else dpd * (julian + 285.0)
    return math.degrees(math.asin(math.sin(math.radians(23.5)) * math.sin(math.radians(lam))))


def _spencer_declination_deg(julian: float) -> float:
    """Spencer's (1971) Fourier series for Earth's declination, the source
    of WRF's own equation of time and eccentricity factor (WRF's declination
    is instead a uniform sun from julian 80).  Day angle from the zero-based
    julian day."""
    g = 2.0 * math.pi * julian / 365.0
    return math.degrees(
        0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))


def _wrf_equation_of_time_min(julian: float) -> float:
    da = 2.0 * math.pi * (julian - 1.0) / 365.0
    return 229.18 * (0.000075 + 0.001868 * math.cos(da) - 0.032077 * math.sin(da)
                     - 0.014615 * math.cos(2.0 * da) - 0.04089 * math.sin(2.0 * da))


def _orbit_cosine_zenith(orbit: Orbit, lat, lon, when: datetime, dtype):
    """The parent's arithmetic with the orbit's declination and equation of
    time (what ``WorldSun.cosine_zenith`` does with an orbit)."""
    hour, julian = _wrf_clock(when)
    state = orbit.state(julian)
    minutes = 60.0 * hour + state.equation_of_time_min + 4.0 * lon
    h = np.deg2rad(minutes / 4.0 - 180.0)
    phi = np.deg2rad(lat)
    mu = (np.sin(phi) * dtype(np.sin(state.declination_rad))
          + np.cos(phi) * dtype(np.cos(state.declination_rad)) * np.cos(h))
    return np.clip(mu, dtype(-1.0), dtype(1.0))


def measure_earth_agreement(year: int = 2026) -> dict[str, float]:
    """The largest differences over ``year`` between the eccentric orbit at
    Earth's values and the model's WRF sun.  Needs NumPy standing in for
    CuPy (the ``numpy_as_cupy`` fixture)."""
    from arwen_global.core.rrtmgp import DTYPE, RRTMGPRadiation

    orbit = _earth_orbit()
    parent = object.__new__(RRTMGPRadiation)
    parent.latitude_deg, parent.longitude_deg = _grid()
    start = datetime(year, 1, 1)
    out = dict.fromkeys(EARTH_TOLERANCE, 0.0)
    out["declination_vs_spencer_orbit_deg"] = 0.0
    out["declination_vs_spencer_wrf_deg"] = 0.0
    for hours in range(0, 365 * 24):
        when = start + timedelta(hours=hours)
        _, julian = _wrf_clock(when)
        state = orbit.state(julian)
        spencer = _spencer_declination_deg(julian)
        out["declination_vs_spencer_orbit_deg"] = max(
            out["declination_vs_spencer_orbit_deg"],
            abs(math.degrees(float(state.declination_rad)) - spencer))
        out["declination_vs_spencer_wrf_deg"] = max(
            out["declination_vs_spencer_wrf_deg"], abs(_wrf_declination_deg(julian) - spencer))
        want_s = float(RRTMGPRadiation._solar_constant(when))
        got_s = S0 * state.distance_factor
        out["solar_constant_rel"] = max(out["solar_constant_rel"], abs(got_s / want_s - 1.0))
        out["declination_deg"] = max(out["declination_deg"], abs(
            math.degrees(float(state.declination_rad)) - _wrf_declination_deg(julian)))
        out["equation_of_time_min"] = max(out["equation_of_time_min"], abs(
            float(state.equation_of_time_min) - _wrf_equation_of_time_min(julian)))
        if hours % 6 == 0:
            want = parent._cosine_zenith(when)
            got = _orbit_cosine_zenith(orbit, parent.latitude_deg, parent.longitude_deg,
                                       when, DTYPE)
            out["cosine_zenith"] = max(out["cosine_zenith"],
                                       float(np.max(np.abs(got - want))))
    # A day's mean sunlight by latitude, every fifth day, from 48 half-hour
    # planes on one meridian of each latitude row.
    lat = parent.latitude_deg[:, :1]
    lon = parent.longitude_deg[:, :1]
    column = object.__new__(RRTMGPRadiation)
    column.latitude_deg, column.longitude_deg = lat, lon
    year_want = np.zeros(lat.shape)
    year_got = np.zeros(lat.shape)
    days = range(0, 365, 5)
    for day in days:
        want = np.zeros(lat.shape)
        got = np.zeros(lat.shape)
        for k in range(48):
            when = start + timedelta(days=day, minutes=30 * k + 15)
            _, julian = _wrf_clock(when)
            want += (float(RRTMGPRadiation._solar_constant(when))
                     * np.maximum(column._cosine_zenith(when), 0.0))
            got += (S0 * orbit.distance_factor(julian)
                    * np.maximum(_orbit_cosine_zenith(orbit, lat, lon, when, DTYPE), 0.0))
        out["daily_mean_w_m2"] = max(out["daily_mean_w_m2"],
                                     float(np.max(np.abs(got - want))) / 48.0)
        year_want += want / 48.0
        year_got += got / 48.0
    out["annual_mean_w_m2"] = float(np.max(np.abs(year_got - year_want))) / len(days)
    # The rows are symmetric about the equator: north minus south of the
    # year's mean, largest over the rows.
    out["annual_mean_north_minus_south_orbit_w_m2"] = float(
        np.max(np.abs(year_got - year_got[::-1]))) / len(days)
    out["annual_mean_north_minus_south_wrf_w_m2"] = float(
        np.max(np.abs(year_want - year_want[::-1]))) / len(days)
    return out


def test_the_model_solar_constant_is_1370_at_mean_distance():
    """The parent's solar constant is 1370 W m-2 times WRF's eccentricity
    factor; the orbits read 1370 as the flux at mean distance."""
    from arwen_global.core.rrtmgp import RRTMGPRadiation

    for when in (datetime(2026, 1, 3), datetime(2026, 7, 4, 12), datetime(2024, 2, 29, 6)):
        hour, julian = _wrf_clock(when)
        da = 2.0 * np.pi * julian / 365.0
        eccfac = (1.000110 + 0.034221 * np.cos(da) + 0.001280 * np.sin(da)
                  + 0.000719 * np.cos(2.0 * da) + 0.000077 * np.sin(2.0 * da))
        assert RRTMGPRadiation._solar_constant(when) / eccfac == pytest.approx(S0, rel=1e-15)


def test_the_eccentric_orbit_at_earths_values_agrees_with_the_models_sun(numpy_as_cupy):
    measured = measure_earth_agreement()
    print("earth agreement:", measured)
    for key, limit in EARTH_TOLERANCE.items():
        assert measured[key] <= limit, (key, measured[key], limit)
    # Where the two differ in declination, the ellipse is the nearer to
    # Spencer's fit of Earth's declination: the gap is WRF's uniform sun.
    assert (measured["declination_vs_spencer_orbit_deg"]
            < measured["declination_vs_spencer_wrf_deg"])
    # The year's mean sunlight: a Kepler orbit gives both hemispheres the
    # same (the sun is nearest where it moves fastest), WRF's uniform sun with
    # its varying distance does not; that is the annual-mean gap above.
    assert measured["annual_mean_north_minus_south_orbit_w_m2"] < 0.2
    assert (measured["annual_mean_north_minus_south_wrf_w_m2"]
            > 10.0 * measured["annual_mean_north_minus_south_orbit_w_m2"])


# ------------------------------------------------------------ wired into the world sun


def _world_sun_module():
    sun = pytest.importorskip(
        "arwen_global.world.sun", reason="package 1.3's world/sun.py is not on this tree")
    if "orbit" in getattr(sun, "NOT_BUILT_HERE", {}) or not hasattr(sun.WorldSun, "equation_of_time_min"):
        pytest.skip("world/sun.py still refuses the orbits: package 3.4's patch "
                    "(w3-4-for-w1-3.patch) is not applied")
    return sun


def test_the_world_sun_runs_the_parsed_orbit(numpy_as_cupy):
    sun_module = _world_sun_module()
    from arwen_global.core.rrtmgp import DTYPE, RRTMGPRadiation
    from arwen_global.world.radiation import world_radiation_class

    ecc = _world('[world]\nlabel = "e"\n[world.sun]\norbit = "eccentric"\n'
                 'eccentricity = 0.1\nperihelion_deg = 0.0\nbrightness = 1.1\n')
    sun = sun_module.WorldSun.from_world(ecc)
    assert sun.orbit == Orbit.eccentric(0.1, 0.0)
    assert not sun.is_earth
    record = sun.record()
    assert record["geometry"] == "world-orbit"
    assert record["closest_approach_w_m2"] == round(1.1 * 1370.0 / 0.81, 3)

    cls = world_radiation_class(RRTMGPRadiation)
    radiation = object.__new__(cls)
    radiation.latitude_deg, radiation.longitude_deg = _grid()
    radiation.world_sun = sun
    for when in (datetime(2026, 3, 21, 0), datetime(2026, 9, 1, 6), datetime(2026, 11, 12)):
        _, julian = _wrf_clock(when)
        solar = radiation._solar_constant(when)
        assert solar == pytest.approx(1370.0 * 1.1 * sun.orbit.distance_factor(julian),
                                      rel=1e-15)
        want = _orbit_cosine_zenith(sun.orbit, radiation.latitude_deg,
                                    radiation.longitude_deg, when, DTYPE)
        got = radiation._cosine_zenith(when)
        assert got.dtype == want.dtype
        assert got.tobytes() == want.tobytes()
    # Closest approach at the March equinox (julian 80): 1691 W m-2 times 1.1.
    assert radiation._solar_constant(datetime(2026, 3, 22, 0)) == pytest.approx(
        1.1 * 1370.0 / 0.81, rel=1e-12)
    with pytest.raises(TypeError, match="needs the valid time"):
        sun.solar_constant(1370.0)


def test_the_world_sun_on_the_perpetual_equinox(numpy_as_cupy):
    sun_module = _world_sun_module()
    eq = _world('[world]\nlabel = "aqua"\n[world.sun]\norbit = "perpetual-equinox"\n')
    sun = sun_module.WorldSun.from_world(eq)
    assert sun.obliquity_deg == 0.0
    for hours in range(0, 365 * 24, 37):
        when = datetime(2026, 1, 1) + timedelta(hours=hours)
        lat, lon = sun.subsolar_point(when)
        assert lat == 0.0
        # No equation of time: the sun is over the meridian where it is
        # noon by the clock.
        hour = when.hour + when.minute / 60.0
        assert math.remainder(lon - (180.0 - 15.0 * hour), 360.0) == pytest.approx(0.0, abs=1e-9)
        assert sun.solar_constant(999.0, when) == 1370.0
