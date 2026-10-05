"""Present-day well-mixed greenhouse gases for the native radiation, by year.

THE BREAKAGE THIS PREVENTS.  Until 2026-10-05 the native suite handed RRTMGP
a fixed 369.55 ppm CO2 override on every run, a value from about the year
2000 that sits roughly 56 ppm below the air of 2026, and the methane and
nitrous oxide it radiated were the RFMIP experiment-zero values of 2014
(1831.47 ppb and 326.99 ppb).  A forecast of the present atmosphere radiated
a quarter-century-old CO2 column: order 0.8 W/m2 too little longwave forcing
at the tropopause and too little stratospheric cooling, on every default run.

WHAT IT DOES.  :func:`present_day_greenhouse_gases` returns the CO2, CH4 and
N2O mole fractions for the run's valid year from the NOAA Global Monitoring
Laboratory globally averaged marine surface annual means, pinned below.  The
year selected is the latest published year at or before the valid year (the
current year's mean is published only after it ends), and a year before a
table's first entry holds that first entry.  The runtime makes no network
access.  The halocarbons and the other minor gases keep the RFMIP
climatology the radiation loads (their radiative weight at forecast range is
negligible), and ozone keeps its climatological profile.

SOURCE.  NOAA GML Carbon Cycle Greenhouse Gases group, "Trends in
globally-averaged CO2, CH4 and N2O determined from NOAA Global Monitoring
Laboratory measurements", https://doi.org/10.15138/9N0H-ZH07,
files ``co2_annmean_gl.txt``, ``ch4_annmean_gl.txt`` and ``n2o_annmean_gl.txt``
under https://gml.noaa.gov/webdata/ccgg/trends/<gas>/, fetched 2026-10-05
(file creation 2026-09-05).  The last year of each file is provisional and
may be revised by NOAA; re-pinning is editing these three tables and
:data:`GREENHOUSE_GAS_SOURCE` together.
"""

from __future__ import annotations

from datetime import date, datetime
from types import MappingProxyType

#: The name every receipt carries for the tables below.
GREENHOUSE_GAS_SOURCE = (
    "NOAA GML global annual mean CO2/CH4/N2O (doi:10.15138/9N0H-ZH07), "
    "fetched 2026-10-05, file creation 2026-09-05"
)

#: CO2 dry-air mole fraction, ppm.  The same file the carried RRTMGP's own
#: dated table pins (core.rrtmgp._NOAA_GML_CO2_ANNUAL_PPM, the 2026-07-05
#: release), value for value: two dated tables that disagree drift apart,
#: and this one overrides that one on every native run.
CO2_PPM = MappingProxyType({
    1979: 336.85, 1980: 338.91, 1981: 340.11, 1982: 340.85, 1983: 342.53,
    1984: 344.07, 1985: 345.54, 1986: 346.97, 1987: 348.68, 1988: 351.16,
    1989: 352.79, 1990: 354.06, 1991: 355.40, 1992: 356.09, 1993: 356.84,
    1994: 358.33, 1995: 360.18, 1996: 361.93, 1997: 363.04, 1998: 365.70,
    1999: 367.80, 2000: 368.96, 2001: 370.57, 2002: 372.58, 2003: 375.14,
    2004: 376.95, 2005: 378.98, 2006: 381.15, 2007: 382.90, 2008: 385.02,
    2009: 386.50, 2010: 388.75, 2011: 390.62, 2012: 392.65, 2013: 395.40,
    2014: 397.34, 2015: 399.65, 2016: 403.07, 2017: 405.22, 2018: 407.61,
    2019: 410.07, 2020: 412.44, 2021: 414.70, 2022: 417.08, 2023: 419.35,
    2024: 422.79, 2025: 425.64,
})

#: CH4 dry-air mole fraction, ppb.
CH4_PPB = MappingProxyType({
    1984: 1644.84, 1985: 1657.30, 1986: 1670.09, 1987: 1682.71,
    1988: 1693.19, 1989: 1704.55, 1990: 1714.43, 1991: 1724.79,
    1992: 1735.47, 1993: 1736.48, 1994: 1742.08, 1995: 1748.88,
    1996: 1751.30, 1997: 1754.51, 1998: 1765.57, 1999: 1772.30,
    2000: 1773.22, 2001: 1771.29, 2002: 1772.68, 2003: 1777.33,
    2004: 1777.13, 2005: 1774.23, 2006: 1774.93, 2007: 1781.31,
    2008: 1787.01, 2009: 1793.51, 2010: 1798.93, 2011: 1803.03,
    2012: 1808.07, 2013: 1813.42, 2014: 1822.58, 2015: 1834.19,
    2016: 1843.18, 2017: 1849.62, 2018: 1857.33, 2019: 1866.59,
    2020: 1878.72, 2021: 1894.79, 2022: 1910.95, 2023: 1921.40,
    2024: 1929.41, 2025: 1935.94,
})

#: N2O dry-air mole fraction, ppb.
N2O_PPB = MappingProxyType({
    2001: 316.36, 2002: 316.94, 2003: 317.63, 2004: 318.25, 2005: 318.91,
    2006: 319.82, 2007: 320.44, 2008: 321.50, 2009: 322.27, 2010: 323.18,
    2011: 324.21, 2012: 325.05, 2013: 325.94, 2014: 327.09, 2015: 328.17,
    2016: 328.95, 2017: 329.74, 2018: 330.90, 2019: 331.88, 2020: 333.00,
    2021: 334.27, 2022: 335.61, 2023: 336.68, 2024: 337.71, 2025: 338.85,
})

_TABLES = (("co2", CO2_PPM, 1.0e-6), ("ch4", CH4_PPB, 1.0e-9),
           ("n2o", N2O_PPB, 1.0e-9))


def _year_of(table, year: int) -> int:
    earlier = [entry for entry in table if entry <= year]
    return max(earlier) if earlier else min(table)


def present_day_greenhouse_gases(valid: date | datetime) -> dict[str, float]:
    """``{"co2", "ch4", "n2o"}`` mole fractions for ``valid``'s year."""
    if not isinstance(valid, (date, datetime)):
        raise TypeError("greenhouse-gas selection needs a date or datetime")
    return {
        gas: float(table[_year_of(table, valid.year)]) * scale
        for gas, table, scale in _TABLES
    }


def greenhouse_gas_record(valid: date | datetime,
                          overrides: dict[str, float] | None = None
                          ) -> dict[str, object]:
    """What a receipt says the radiation ran with: the year each table was
    read at, the mole fractions after any case override, and the source."""
    selected = present_day_greenhouse_gases(valid)
    record: dict[str, object] = {
        "source": GREENHOUSE_GAS_SOURCE,
        "valid_year": int(valid.year),
        "table_year": {
            gas: _year_of(table, valid.year) for gas, table, _ in _TABLES
        },
        "mole_fraction": dict(selected),
        "overridden": sorted(overrides or {}),
    }
    if overrides:
        record["mole_fraction"].update(overrides)
    return record


__all__ = [
    "CH4_PPB", "CO2_PPM", "GREENHOUSE_GAS_SOURCE", "N2O_PPB",
    "greenhouse_gas_record", "present_day_greenhouse_gases",
]
