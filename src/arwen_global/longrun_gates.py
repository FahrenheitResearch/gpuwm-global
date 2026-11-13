"""Long-run diagnostics and gates LR-0 to LR-10: rows written, gates read.

A run of a month or more is judged on drift, and a drift is read from the
series of rows the run appends to ``diagnostics.jsonl`` at every output
interval.  This module holds both halves:

* the ROWS (:func:`atmosphere_band_zonal_means`, :func:`atmosphere_rows`,
  :func:`surface_rows`), which ``MoistHybridModel.diagnostics`` writes on
  every output step.  Every global mean here is an AREA mean on the
  model's own Gaussian quadrature (``GaussianGrid.global_mean``: half the
  weight-sum of the zonal means), the quadrature the mass and water rows
  have always used.  The physics' own per-call radiation readings
  (``mean_outgoing_longwave_w_m2`` and the rest) are flat means over the
  grid's points, which over-count the polar rows; they are kept unchanged
  because receipts compare them, and the ``area_mean_*`` rows sit beside
  them under their own names.
* the GATES (:func:`evaluate`), which read a run's rows (one folder, or a
  chain of segment folders in order) and judge each 30-day window and the
  whole run.  ``tools/arwen_global_longrun_report.py`` prints the table;
  the runner writes the same table into the receipt of every segment of a
  run longer than 384 h (``longrun_gates``), outside the receipt's
  ``status``: the thresholds marked engineering below are proposals to be
  revisited once a long run has measured the model, not standards.

What each gate reads, and where the number comes from (long-run design,
section 11, revision 2):

LR-0 restart   a straight and a segmented terminal checkpoint, bitwise
               (self hashes, then array by array); a separate test, not
               the segmented-only long arms.
LR-1 mass      range of ``global_mean_dry_surface_pressure_pa`` (global
               mean ps minus g times the global-mean atmospheric water).
               The mass fixer holds TOTAL mass, so dry mass carries the
               swing of global water; reported, 30 Pa over the run.
LR-2 reservoir minimum of the explicit surface reservoir over land
               columns (land fraction >= 0.5) >= 50 kg/m2; and, when the
               open-water exchange account's rows exist, the identity
               account = -int(E-P)_water - fixer - clipping to 1e-6
               relative, with the largest 24 h water-column evaporation
               beside a provisional 60 mm/day ceiling (reported, not
               enforced).  This module does not write those rows.
LR-3 water     per window: area-mean precipitation from the accumulated
               precipitation (difference of two rows, mm/day), in
               [2.5, 3.3]; the vapour column (precipitable water), in
               [20, 30] kg/m2; global E - P as the atmospheric water
               tendency and E as P plus it.  Warn.
LR-4 energy    the implied residual TOA net - surface net radiation + H +
               LE + Lf S - dE/dt, W/m2, MEASURED AND STATED per window
               with every term.  Radiation is the exact interval mean of
               the namespace's flux-times-dt accumulators; H and LE are the
               held surface-layer fluxes SAMPLED at the output instants and
               averaged in time (trapezoid), so they are an estimate; S is
               the frozen precipitation from its accumulator; E is the
               moist total energy of the budgets ledger (insitu/budgets.py).
               The residual holds everything these terms do not book.
               Above 2 W/m2 flagged; 1 W/m2 is the slab ocean's
               prerequisite.
LR-5 T drift   global-mean T at 850/500/200/50/10 hPa minus a GATING
               reference at the same time (another run's rows, arm B
               against arm A, or a table taken from the analyses of the
               run's dates), ordinary least-squares slope from day 30 over
               at least 60 days, per 30 days, with a 95% interval widened
               for lag-one autocorrelation (effective sample size
               n (1 - r1) / (1 + r1)).  |slope| < 0.1 K at 850-200 hPa,
               < 0.5 K at 10 hPa; 50 hPa reported.  A climatology (the
               report tool reads a daily long-term-mean file) is fitted
               and reported beside it, never gated.
LR-6 strat     polar-cap (60-90 deg) mean T at 10 and 50 hPa against the
               gating reference, |window-mean bias| <= 10 K (proposed, to
               be set after the first long runs); minimum temperature
               anywhere below 165 K (5 K above the radiation's 160 K input
               clamp) refuses, and is never by itself a pass; zonal-mean u
               at 10 hPa at 60 N and 60 S reported.
LR-7 land      land-mean soil water (2 m column) change per 30 days; NH
               snow-covered area.  Reported.
LR-8 boundary  maximum |SST - provider| on open ocean <= 1e-3 K and ice
               equal to the record, from the boundary service's rows.
               This module does not write them.
LR-9 TOA       area-mean TOA net and OLR per window, exact interval means,
               reported; gated only on a full-year mean (TOA net within
               1 +- 5 W/m2), since insolation swings about 16 W/m2
               absorbed over the year.
LR-10 AAM      the atmosphere's axial angular momentum (relative int u a
               cos(phi) dp/g plus planetary int Omega a^2 cos^2(phi) dp/g,
               the insitu ledger's definitions) as the planet's total in
               kg m2/s: each day's mean over its output instants, the
               least-squares drift of the daily means per day over the run
               and per window.  Reported, no threshold yet.

Interpolation to pressure levels is linear in ln p between the bracketing
full levels of each column.  Below a column's lowest full level the
temperature is extrapolated down the 6.5 K/km standard lapse rate (T
(p / p_low) ** (R Gamma / g)) and the wind is held at the lowest level's;
a level above the top full level of any column is not reported (the row
is null), never invented.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from .constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EARTH_RADIUS_M,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPORIZATION,
    LIQUID_WATER_DENSITY,
    CONVECTIVE_RAIN_ACCUMULATOR,
)

#: The pressure levels the drift gates read, hPa.
PRESSURE_LEVELS_HPA = (850, 500, 200, 50, 10)
#: The stratospheric level of LR-6's wind, and its polar-cap levels.
STRATOSPHERE_LEVEL_HPA = 10
POLAR_CAP_LEVELS_HPA = (10, 50)
#: Standard-atmosphere lapse rate for the below-ground extrapolation, K/m.
STANDARD_LAPSE_RATE_K_M = 0.0065
#: A column at or above this land fraction is land for the land rows, and
#: below it water for the ocean and ice rows.
LAND_FRACTION_THRESHOLD = 0.5
#: Sea-ice extent counts the area of water cells with at least this ice
#: fraction (the NSIDC/OISST convention).
ICE_EXTENT_THRESHOLD = 0.15
#: Nino-3.4: 5S-5N, 170W-120W.
NINO34_LATITUDE_DEG = (-5.0, 5.0)
NINO34_LONGITUDE_DEG = (190.0, 240.0)
#: Polar caps of LR-6.
POLAR_CAP_EDGE_DEG = 60.0

#: The namespace's radiation accumulators (J/m2, flux times dt summed over
#: every physics call) and the row each one's area mean is written under.
RADIATION_ACCUMULATOR_ROWS = (
    ("acc_sw_down_top_j_m2", "area_mean_acc_sw_down_top_j_m2"),
    ("acc_sw_up_top_j_m2", "area_mean_acc_sw_up_top_j_m2"),
    ("acc_lw_up_top_j_m2", "area_mean_acc_lw_up_top_j_m2"),
    ("acc_sw_down_surface_j_m2", "area_mean_acc_sw_down_surface_j_m2"),
    ("acc_sw_up_surface_j_m2", "area_mean_acc_sw_up_surface_j_m2"),
    ("acc_lw_down_surface_j_m2", "area_mean_acc_lw_down_surface_j_m2"),
    ("acc_lw_up_surface_j_m2", "area_mean_acc_lw_up_surface_j_m2"),
)

#: Rows other packages write and the gates read when present.  LR-2: the
#: open-water exchange account and the three terms its identity holds it
#: to (accumulated E - P over water, the water fixer's corrections on
#: water columns, clipping), all global means in kg/m2, and the largest
#: 24 h water-column evaporation (mm/day).  LR-8: the boundary service's
#: daily departure and ice mismatch.
EXCHANGE_ACCOUNT_ROW = "global_mean_ocean_exchange_kg_m2"
WATER_NET_EVAPORATION_ROW = "global_mean_water_net_evaporation_kg_m2"
WATER_FIXER_ON_WATER_ROW = "global_mean_water_fixer_on_water_kg_m2"
WATER_CLIPPING_ON_WATER_ROW = "global_mean_water_clipping_on_water_kg_m2"
WATER_EVAPORATION_24H_ROW = "maximum_water_column_evaporation_24h_mm_day"
#: Where the native suite writes its 24 h open-water evaporation watch:
#: each physics half-call's diagnostics under ``step_metrics``, in kg/m2
#: over a 24 h block of model time (so mm/day), for the running block and
#: the last complete one.  The complete block's value is held for a whole
#: block, so output rows at most 24 h apart see every complete block.
EVAPORATION_WATCH_HALVES = ("first_half_physics", "second_half_physics")
EVAPORATION_WATCH_ROWS = ("maximum_open_water_evaporation_day_kg_m2",
                          "maximum_open_water_evaporation_last_day_kg_m2")
BOUNDARY_SST_DEPARTURE_ROW = "boundary_max_abs_sst_departure_k"
BOUNDARY_ICE_MISMATCH_ROW = "boundary_ice_record_mismatch_columns"

WINDOW_DAYS = 30.0
SECONDS_PER_DAY = 86400.0
#: Runs longer than this carry the gate table in every segment's receipt
#: (the design's long-run regime begins past the 384 h series horizon).
LONG_RUN_THRESHOLD_S = 384.0 * 3600.0

# Thresholds (long-run design section 11).  All but LR-0 and LR-8 are
# engineering proposals, to be revisited once a long run has measured the
# model.
LR1_RANGE_PA = 30.0
LR2_LAND_MINIMUM_KG_M2 = 50.0
LR2_LEDGER_RELATIVE = 1.0e-6
LR2_EVAPORATION_CEILING_MM_DAY = 60.0
LR3_PRECIPITATION_MM_DAY = (2.5, 3.3)
LR3_PRECIPITABLE_WATER_KG_M2 = (20.0, 30.0)
LR4_RESIDUAL_FLAG_W_M2 = 2.0
LR4_RESIDUAL_SLAB_W_M2 = 1.0
LR5_SLOPE_K_PER_30D = {850: 0.1, 500: 0.1, 200: 0.1, 50: None, 10: 0.5}
LR5_FIT_START_DAY = 30.0
LR5_MINIMUM_FIT_DAYS = 60.0
LR6_MINIMUM_TEMPERATURE_K = 165.0
LR6_CAP_BIAS_PROPOSED_K = 10.0
LR8_SST_DEPARTURE_K = 1.0e-3
LR9_TOA_NET_W_M2 = (1.0 - 5.0, 1.0 + 5.0)
LR9_ANNUAL_DAYS = 365.0


# --------------------------------------------------------------------- rows


def level_row_name(level_hpa: int) -> str:
    return f"global_mean_temperature_{int(level_hpa)}hpa_k"


def cap_row_name(level_hpa: int, hemisphere: str) -> str:
    return f"polar_cap_mean_temperature_{int(level_hpa)}hpa_{hemisphere}_k"


def cell_areas_m2(grid) -> np.ndarray:
    """Area of one cell on each Gaussian row, m2 (the row's quadrature
    weight's share of the sphere, divided among its longitudes)."""
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    return 4.0 * math.pi * float(grid.radius_m) ** 2 * (0.5 * weights) / grid.nlon


def area_mean(grid, plane, mask=None) -> float | None:
    """Area mean of a ``(nlat, nlon)`` plane on the grid's quadrature,
    over ``mask`` (a 0..1 weight plane) when one is given; None when the
    mask covers nothing."""
    arr = np.asarray(plane, dtype=np.float64)
    if mask is None:
        return float(grid.global_mean(arr))
    weight = np.asarray(mask, dtype=np.float64)
    covered = grid.global_mean(weight)
    if covered <= 0.0:
        return None
    return float(grid.global_mean(arr * weight) / covered)


def box_mask(grid, latitude_deg, longitude_deg) -> np.ndarray:
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    lon = np.mod(np.asarray(grid.longitude_deg, dtype=np.float64), 360.0)
    rows = (lat >= latitude_deg[0]) & (lat <= latitude_deg[1])
    cols = (lon >= longitude_deg[0]) & (lon <= longitude_deg[1])
    return (rows[:, None] & cols[None, :]).astype(np.float64)


def interpolate_to_pressure(xp, field, p_full, target_pa: float, *,
                            extrapolate: str):
    """``field`` (levels, rows, lon) at pressure ``target_pa`` per column.

    Linear in ln p between the bracketing full levels.  ``extrapolate``
    says what a column whose lowest full level lies above the target does:
    ``"lapse"`` (temperature, down the standard lapse rate from the lowest
    level) or ``"hold"`` (the lowest level's value).  Returns the plane
    and a boolean plane, True where the target lies above the column's TOP
    full level (those columns hold the top value; the caller decides
    whether the level may be reported at all).
    """
    # Index 0 is the model top (smallest pressure), as the model stores it.
    # Level by level and gather by gather: nothing here is a volume, so
    # the cost on the card is a handful of planes (the diagnostics step is
    # the widest grid state a run builds, and a T383 run has no room for a
    # float64 copy of a volume beside it).
    nlev = int(p_full.shape[0])
    above = xp.zeros(p_full.shape[1:], dtype=xp.int64)
    for k in range(nlev):
        above += p_full[k] < target_pa  # levels lying above the target
    k = xp.clip(above, 1, nlev - 1)[None]

    def gather(volume, index):
        return xp.take_along_axis(volume, index, axis=0)[0].astype(xp.float64)

    p_a, p_b = gather(p_full, k - 1), gather(p_full, k)
    f_a, f_b = gather(field, k - 1), gather(field, k)
    weight = (math.log(target_pa) - xp.log(p_a)) / (xp.log(p_b) - xp.log(p_a))
    value = f_a + weight * (f_b - f_a)
    lowest = field[nlev - 1].astype(xp.float64)
    if extrapolate == "lapse":
        exponent = DRY_AIR_GAS_CONSTANT * STANDARD_LAPSE_RATE_K_M / GRAVITY_M_S2
        below = lowest * (target_pa / p_full[nlev - 1].astype(xp.float64)) ** exponent
    elif extrapolate == "hold":
        below = lowest
    else:
        raise ValueError(f"extrapolate must be 'lapse' or 'hold', got {extrapolate!r}")
    value = xp.where(above >= nlev, below, value)
    over_top = above <= 0
    value = xp.where(over_top, field[0].astype(xp.float64), value)
    return value, over_top


#: The zonal means :func:`atmosphere_band_zonal_means` returns, in order.
#: ``over_top_<level>`` is the fraction of a row's columns the level lies
#: above.
ATMOSPHERE_ZONAL_NAMES = (
    "moist_total_energy", "precipitable_water",
    *(f"temperature_{level}" for level in PRESSURE_LEVELS_HPA),
    *(f"over_top_{level}" for level in PRESSURE_LEVELS_HPA),
    "u_stratosphere",
    # LR-10: int u dp/g and int dp/g, the two column integrals the axial
    # angular momentum is made of (the row multiplies them by a cos(phi)
    # and Omega a^2 cos^2(phi) per row).
    "u_column", "column_mass",
)
#: The angular-momentum rows (LR-10): global means per unit area, the
#: insitu ledger's names and definitions (insitu/budgets.py), kg/s.
RELATIVE_ANGULAR_MOMENTUM_ROW = "global_mean_relative_angular_momentum_kg_s"
AXIAL_ANGULAR_MOMENTUM_ROW = "global_mean_axial_angular_momentum_kg_s"


def atmosphere_band_zonal_means(xp, g, surface_geopotential):
    """One latitude band's contribution to the atmosphere rows: the zonal
    mean, per row, of every column quantity they need, as one
    ``(len(ATMOSPHERE_ZONAL_NAMES), rows)`` float64 array.

    ``g`` is the band's grid state (``MoistHybridModel.grid_band``):
    temperature, the wind, the water species, ``ps``, ``dp`` and
    ``p_full``; ``surface_geopotential`` the band's rows of the terrain
    geopotential.  Every quantity is column-local and every zonal mean
    reads one row, so a band's entries are the whole call's entries
    there, whatever the band count; only this vector per row leaves the
    band.
    """
    temperature = g["temperature"]
    u = g["u"]
    v = g["v"]
    # The budgets ledger's moist total energy (insitu/budgets.py): the
    # Kasahara invariant int (cp T + K) dp/g + Phi_s ps/g, plus the latent
    # energy of vapour, less the fusion energy of the frozen species.
    # Summed level by level into float64 planes (no volume temporaries).
    shape = temperature.shape[1:]
    enthalpy = xp.zeros(shape, dtype=xp.float64)
    vapour = xp.zeros(shape, dtype=xp.float64)
    frozen = xp.zeros(shape, dtype=xp.float64)
    u_column = xp.zeros(shape, dtype=xp.float64)
    column_mass = xp.zeros(shape, dtype=xp.float64)
    for k in range(int(temperature.shape[0])):
        dp_g = g["dp"][k].astype(xp.float64) / GRAVITY_M_S2
        enthalpy += (DRY_AIR_CP * temperature[k].astype(xp.float64)
                     + 0.5 * (u[k].astype(xp.float64) ** 2
                              + v[k].astype(xp.float64) ** 2)) * dp_g
        u_column += u[k].astype(xp.float64) * dp_g
        column_mass += dp_g
        vapour += g["qv"][k].astype(xp.float64) * dp_g
        frozen += (g["qi"][k].astype(xp.float64) + g["qs"][k].astype(xp.float64)
                   + g["qg"][k].astype(xp.float64)) * dp_g
    energy = (
        enthalpy
        + xp.asarray(surface_geopotential).astype(xp.float64)
        * xp.asarray(g["ps"]).astype(xp.float64) / GRAVITY_M_S2
        + LATENT_HEAT_VAPORIZATION * vapour
        - LATENT_HEAT_FUSION * frozen
    )
    planes = {"moist_total_energy": energy, "precipitable_water": vapour}
    for level in PRESSURE_LEVELS_HPA:
        plane, over_top = interpolate_to_pressure(
            xp, temperature, g["p_full"], 100.0 * level, extrapolate="lapse")
        planes[f"temperature_{level}"] = plane
        planes[f"over_top_{level}"] = over_top.astype(xp.float64)
    planes["u_stratosphere"], _ = interpolate_to_pressure(
        xp, u, g["p_full"], 100.0 * STRATOSPHERE_LEVEL_HPA, extrapolate="hold")
    planes["u_column"] = u_column
    planes["column_mass"] = column_mass
    return xp.stack([
        xp.mean(planes[name], axis=-1, dtype=xp.float64)
        for name in ATMOSPHERE_ZONAL_NAMES
    ])


def global_mean_of_zonal(grid, zonal) -> float:
    """``GaussianGrid.global_mean`` from the zonal means it would form."""
    return float(0.5 * np.sum(np.asarray(zonal, dtype=np.float64)
                              * grid.quadrature_weights))


def zonal_area_mean(grid, zonal, low_deg: float, high_deg: float) -> float:
    """Area mean over the Gaussian rows in ``[low_deg, high_deg]``."""
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    weights = np.where((lat >= low_deg) & (lat <= high_deg),
                       grid.quadrature_weights, 0.0)
    return float(np.sum(np.asarray(zonal, dtype=np.float64) * weights) / weights.sum())


def zonal_at_latitude(grid, zonal, latitude_deg: float) -> float:
    """A zonal-mean profile at one latitude, linear in latitude between
    the two Gaussian rows that bracket it."""
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    order = np.argsort(lat)
    return float(np.interp(float(latitude_deg), lat[order],
                           np.asarray(zonal, dtype=np.float64)[order]))


def angular_momentum_rows(grid, u_column, column_mass, *,
                          rotation_rate_s: float = EARTH_ROTATION_RATE_S) -> dict[str, float]:
    """The axial angular momentum of the atmosphere from the zonal means of
    int u dp/g and int dp/g per Gaussian row: global means per unit area
    (kg/s), relative int u a cos(phi) dp/g and axial (relative plus the
    planetary int Omega a^2 cos^2(phi) dp/g), the insitu ledger's terms.
    Times 4 pi a^2 they are the planet's totals in kg m2/s."""
    radius = float(grid.radius_m)
    cos_lat = np.asarray(grid.cos_lat, dtype=np.float64)
    relative = global_mean_of_zonal(
        grid, np.asarray(u_column, dtype=np.float64) * radius * cos_lat)
    planetary = global_mean_of_zonal(
        grid, np.asarray(column_mass, dtype=np.float64)
        * float(rotation_rate_s) * radius * radius * cos_lat * cos_lat)
    return {RELATIVE_ANGULAR_MOMENTUM_ROW: relative,
            AXIAL_ANGULAR_MOMENTUM_ROW: relative + planetary}


def atmosphere_rows(grid, zonal, global_mean_ps_pa: float,
                    global_mean_atmospheric_water_kg_m2: float, *,
                    rotation_rate_s: float = EARTH_ROTATION_RATE_S) -> dict[str, object]:
    """The atmosphere's long-run rows from the whole globe's zonal means
    (``(len(ATMOSPHERE_ZONAL_NAMES), nlat)``, grid order)."""
    z = {name: np.asarray(zonal[i], dtype=np.float64)
         for i, name in enumerate(ATMOSPHERE_ZONAL_NAMES)}
    rows: dict[str, object] = {
        # Dry mass as a surface pressure: what the column weighs without
        # its water, ps - g W, global mean (LR-1).
        "global_mean_dry_surface_pressure_pa": float(
            global_mean_ps_pa - GRAVITY_M_S2 * global_mean_atmospheric_water_kg_m2),
        "global_mean_moist_total_energy_j_m2": global_mean_of_zonal(
            grid, z["moist_total_energy"]),
        "global_mean_precipitable_water_kg_m2": global_mean_of_zonal(
            grid, z["precipitable_water"]),
    }
    for level in PRESSURE_LEVELS_HPA:
        resolved = not bool(np.any(z[f"over_top_{level}"] > 0.0))
        rows[level_row_name(level)] = (
            global_mean_of_zonal(grid, z[f"temperature_{level}"]) if resolved else None)
    for level in POLAR_CAP_LEVELS_HPA:
        resolved = not bool(np.any(z[f"over_top_{level}"] > 0.0))
        for name, low, high in (("north", POLAR_CAP_EDGE_DEG, 90.0),
                                ("south", -90.0, -POLAR_CAP_EDGE_DEG)):
            rows[cap_row_name(level, name)] = (
                zonal_area_mean(grid, z[f"temperature_{level}"], low, high)
                if resolved else None)
    level = STRATOSPHERE_LEVEL_HPA
    resolved = not bool(np.any(z[f"over_top_{level}"] > 0.0))
    for name, lat in (("60n", POLAR_CAP_EDGE_DEG), ("60s", -POLAR_CAP_EDGE_DEG)):
        rows[f"zonal_mean_u_{level}hpa_{name}_m_s"] = (
            zonal_at_latitude(grid, z["u_stratosphere"], lat) if resolved else None)
    rows.update(angular_momentum_rows(grid, z["u_column"], z["column_mass"],
                                      rotation_rate_s=rotation_rate_s))
    return rows


def surface_rows(grid, surface: dict[str, np.ndarray],
                 physics: dict[str, np.ndarray],
                 physics_metadata: dict | None) -> dict[str, object]:
    """The surface's and the physics namespace's long-run rows.

    ``surface`` and ``physics`` are host planes by name (the bundle's
    surface members and physics-state arrays); a row whose inputs the
    state does not carry is left out, never zero-filled.
    """
    rows: dict[str, object] = {}
    land_fraction = np.asarray(surface["land_fraction"], dtype=np.float64)
    land = (land_fraction >= LAND_FRACTION_THRESHOLD)
    water = ~land
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    north = np.repeat((lat > 0.0)[:, None], grid.nlon, axis=1)
    south = ~north
    areas = np.repeat(cell_areas_m2(grid)[:, None], grid.nlon, axis=1)

    reservoir = np.asarray(surface["water_kg_m2"], dtype=np.float64)
    if bool(land.any()):
        rows["minimum_land_surface_water_kg_m2"] = float(reservoir[land].min())
    soil = surface.get("soil_water_fraction")
    if soil is not None:
        from .water import SOIL_LAYER_THICKNESS_M

        thickness = np.asarray(SOIL_LAYER_THICKNESS_M, dtype=np.float64)[:, None, None]
        column = (np.maximum(np.asarray(soil, dtype=np.float64), 0.0)
                  * thickness * LIQUID_WATER_DENSITY).sum(axis=0)
        mean = area_mean(grid, column, land_fraction)
        if mean is not None:
            rows["land_mean_soil_water_kg_m2"] = mean
    snow_cover = physics.get("noah_snowc")
    if snow_cover is not None:
        cover = np.clip(np.asarray(snow_cover, dtype=np.float64), 0.0, 1.0)
        rows["nh_snow_covered_area_km2"] = float(
            (cover * land_fraction * areas)[north].sum() / 1.0e6)

    sst = surface.get("sea_surface_temperature_k")
    if sst is None:
        sst = surface["temperature_k"]
    box = box_mask(grid, NINO34_LATITUDE_DEG, NINO34_LONGITUDE_DEG) * water
    nino = area_mean(grid, sst, box)
    if nino is not None:
        rows["nino34_mean_sst_k"] = nino
    ice = surface.get("sea_ice_fraction")
    if ice is not None:
        ice = np.clip(np.asarray(ice, dtype=np.float64), 0.0, 1.0)
        extent = water & (ice >= ICE_EXTENT_THRESHOLD)
        for name, half in (("nh", north), ("sh", south)):
            rows[f"{name}_sea_ice_extent_km2"] = float(areas[extent & half].sum() / 1.0e6)
            rows[f"{name}_sea_ice_area_km2"] = float(
                (ice * areas)[water & half].sum() / 1.0e6)

    # Precipitation: the surface's microphysics buckets plus the cumulus
    # accumulator, kg/m2 since the cold start (equal to mm).
    buckets = [surface.get(name) for name in (
        "accumulated_rain_kg_m2", "accumulated_snow_kg_m2",
        "accumulated_graupel_kg_m2")]
    if all(b is not None for b in buckets):
        total = sum(np.asarray(b, dtype=np.float64) for b in buckets)
        frozen = np.asarray(buckets[1], dtype=np.float64) + np.asarray(
            buckets[2], dtype=np.float64)
        convective = physics.get(CONVECTIVE_RAIN_ACCUMULATOR)
        if convective is not None:
            total = total + np.asarray(convective, dtype=np.float64)
        rows["area_mean_accumulated_precipitation_kg_m2"] = area_mean(grid, total)
        rows["area_mean_accumulated_frozen_precipitation_kg_m2"] = area_mean(grid, frozen)

    # Radiation: the held planes at this instant, and the accumulators.
    held = {name: physics.get(name) for name in (
        "olr", "swdnt", "swupt", "gsw", "glw", "lwupb")}
    if held["olr"] is not None:
        rows["area_mean_outgoing_longwave_w_m2"] = area_mean(grid, held["olr"])
    if held["swdnt"] is not None and held["swupt"] is not None:
        rows["area_mean_toa_downward_shortwave_w_m2"] = area_mean(grid, held["swdnt"])
        rows["area_mean_toa_upward_shortwave_w_m2"] = area_mean(grid, held["swupt"])
        if held["olr"] is not None:
            rows["area_mean_toa_net_w_m2"] = area_mean(
                grid, np.asarray(held["swdnt"], dtype=np.float64)
                - np.asarray(held["swupt"], dtype=np.float64)
                - np.asarray(held["olr"], dtype=np.float64))
    if all(held[name] is not None for name in ("gsw", "glw", "lwupb")):
        rows["area_mean_surface_net_radiation_w_m2"] = area_mean(
            grid, np.asarray(held["gsw"], dtype=np.float64)
            + np.asarray(held["glw"], dtype=np.float64)
            - np.asarray(held["lwupb"], dtype=np.float64))
    if physics.get("hfx") is not None:
        rows["area_mean_sensible_heat_flux_w_m2"] = area_mean(grid, physics["hfx"])
    if physics.get("qfx") is not None:
        rows["area_mean_latent_heat_flux_w_m2"] = LATENT_HEAT_VAPORIZATION * area_mean(
            grid, physics["qfx"])
    accumulated = [(row, physics.get(name)) for name, row in RADIATION_ACCUMULATOR_ROWS]
    if all(value is not None for _row, value in accumulated):
        for row, value in accumulated:
            rows[row] = area_mean(grid, value)
        rows["radiation_accumulated_s"] = float(
            (physics_metadata or {}).get("radiation_accumulated_s", 0.0))
    return rows


# -------------------------------------------------------------------- gates


def read_rows(folders) -> list[dict]:
    """The diagnostics rows of a run: one folder, or a chain of segment
    folders in order.  A restart's opening row repeats the previous
    segment's last step; the first row of each step is kept."""
    rows: list[dict] = []
    seen: set[int] = set()
    for folder in folders:
        path = Path(folder)
        if path.is_dir():
            path = path / "diagnostics.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                step = int(row["step"])
                if step in seen:
                    continue
                seen.add(step)
                rows.append(row)
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def _value(row, name):
    value = row.get(name)
    return None if value is None else float(value)


def _series(rows, name):
    pairs = [(float(r["time_s"]), _value(r, name)) for r in rows]
    return [(t, v) for t, v in pairs if v is not None]


def evaporation_watch(row) -> float | None:
    """The largest 24 h water-column evaporation a row carries, mm/day:
    its own row when written, else the native suite's open-water watch."""
    value = _value(row, WATER_EVAPORATION_24H_ROW)
    if value is not None:
        return value
    metrics = row.get("step_metrics") or {}
    found = []
    for half in EVAPORATION_WATCH_HALVES:
        physics = (metrics.get(half) or {}).get("physics") or {}
        found += [float(physics[name]) for name in EVAPORATION_WATCH_ROWS
                  if physics.get(name) is not None]
    return max(found) if found else None


def windows(rows, window_days: float = WINDOW_DAYS) -> list[tuple[float, float, list]]:
    """``(start_day, end_day, rows)`` per window, rows on both edges
    included so a difference spans the whole window; the last window may
    be shorter than ``window_days``."""
    if not rows:
        return []
    t0 = float(rows[0]["time_s"])
    end = (float(rows[-1]["time_s"]) - t0) / SECONDS_PER_DAY
    out = []
    start = 0.0
    while start < end - 1.0e-9:
        stop = min(start + window_days, end)
        inside = [r for r in rows
                  if start - 1.0e-9 <= (float(r["time_s"]) - t0) / SECONDS_PER_DAY
                  <= stop + 1.0e-9]
        if len(inside) >= 2:
            out.append((start, stop, inside))
        start += window_days
    return out


def relative_drift(rows, name) -> float | None:
    series = _series(rows, name)
    if len(series) < 2:
        return None
    first, last = series[0][1], series[-1][1]
    return abs(last - first) / max(abs(first), 1.0e-30)


def _difference_rate(rows, name, *, per_s=None):
    """(last - first) / elapsed of an accumulated row; elapsed in model
    seconds unless ``per_s`` names the row holding the accumulated seconds."""
    a, b = rows[0], rows[-1]
    va, vb = _value(a, name), _value(b, name)
    # The cold start's row carries no radiation accumulator: no physics call
    # has run, and the accumulators and their seconds begin at zero.
    at_cold_start = (per_s is not None and va is None and float(a["time_s"]) == 0.0
                     and _value(a, per_s) is None)
    if at_cold_start:
        va = 0.0
    if va is None or vb is None:
        return None
    if per_s is None:
        seconds = float(b["time_s"]) - float(a["time_s"])
    else:
        sa, sb = (0.0 if at_cold_start else _value(a, per_s)), _value(b, per_s)
        if sa is None or sb is None:
            return None
        seconds = sb - sa
    if seconds <= 0.0:
        return None
    return (vb - va) / seconds


def _time_mean(rows, name):
    """Trapezoid time mean of a sampled row over the window."""
    series = _series(rows, name)
    if len(series) < 2:
        return None
    t = np.array([s[0] for s in series])
    v = np.array([s[1] for s in series])
    return float(np.trapezoid(v, t) / (t[-1] - t[0]))


def energy_budget(rows) -> dict[str, float] | None:
    """The terms of LR-4 over one window, W/m2."""
    def rad(name):
        return _difference_rate(rows, name, per_s="radiation_accumulated_s")

    sw_down_top = rad("area_mean_acc_sw_down_top_j_m2")
    sw_up_top = rad("area_mean_acc_sw_up_top_j_m2")
    lw_up_top = rad("area_mean_acc_lw_up_top_j_m2")
    sfc = [rad(f"area_mean_acc_{n}_surface_j_m2")
           for n in ("sw_down", "sw_up", "lw_down", "lw_up")]
    sensible = _time_mean(rows, "area_mean_sensible_heat_flux_w_m2")
    latent = _time_mean(rows, "area_mean_latent_heat_flux_w_m2")
    frozen = _difference_rate(rows, "area_mean_accumulated_frozen_precipitation_kg_m2")
    tendency = _difference_rate(rows, "global_mean_moist_total_energy_j_m2")
    needed = (sw_down_top, sw_up_top, lw_up_top, *sfc, sensible, latent, frozen, tendency)
    if any(term is None for term in needed):
        return None
    toa = sw_down_top - sw_up_top - lw_up_top
    surface = sfc[0] - sfc[1] + sfc[2] - sfc[3]
    snow = LATENT_HEAT_FUSION * frozen
    residual = toa - surface + sensible + latent + snow - tendency
    return {
        "toa_net_w_m2": toa,
        "outgoing_longwave_w_m2": lw_up_top,
        "surface_net_radiation_w_m2": surface,
        "sensible_heat_flux_w_m2": sensible,
        "latent_heat_flux_w_m2": latent,
        "frozen_precipitation_latent_w_m2": snow,
        "energy_tendency_w_m2": tendency,
        "residual_w_m2": residual,
    }


def _t_quantile_975(dof: float) -> float:
    """Student t 97.5% point (Cornish-Fisher expansion about the normal,
    within 1e-3 of the table for dof >= 3)."""
    z = 1.959963984540054
    if dof <= 0:
        return math.inf
    n = float(dof)
    return (z + (z**3 + z) / (4 * n) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * n**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * n**3))


def trend(days, values) -> dict[str, float] | None:
    """OLS slope per 30 days with a 95% interval, widened for lag-one
    autocorrelation of the residuals (effective n = n (1-r1)/(1+r1))."""
    x = np.asarray(days, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 3:
        return None
    xm = x - x.mean()
    sxx = float(np.sum(xm * xm))
    if sxx <= 0.0:
        return None
    slope = float(np.sum(xm * (y - y.mean())) / sxx)
    residual = y - (y.mean() + slope * xm)
    ss = float(np.sum(residual * residual))
    r1 = 0.0
    if ss > 0.0 and n > 3:
        r1 = float(np.sum(residual[1:] * residual[:-1]) / ss)
    r1 = min(max(r1, 0.0), 0.99)
    n_eff = n * (1.0 - r1) / (1.0 + r1)

    def half_width(dof):
        if dof <= 0.0:
            return math.inf
        stderr = math.sqrt(ss / dof / sxx) if ss > 0.0 else 0.0
        return _t_quantile_975(dof) * stderr

    half = half_width(n_eff - 2.0)
    return {
        "slope_per_30d": slope * WINDOW_DAYS,
        "ci95_half_width_per_30d": half * WINDOW_DAYS,
        # The interval independent residuals would give, for comparison.
        "ols_ci95_half_width_per_30d": half_width(n - 2.0) * WINDOW_DAYS,
        "samples": int(n),
        "effective_samples": float(n_eff),
        "lag1_autocorrelation": float(r1),
    }


def _gate(name, kind, threshold, verdict, **fields):
    return {"name": name, "kind": kind, "threshold": threshold,
            "verdict": verdict, **fields}


def _within(value, bounds):
    return bounds[0] <= value <= bounds[1]


def _when(start_utc, row):
    if start_utc is None:
        return None
    return start_utc + dt.timedelta(seconds=float(row["time_s"]))


class RunReference:
    """Another run's rows as a reference (arm B against arm A): the
    reference row of the same name, linear in model time between its
    rows, None outside them."""

    def __init__(self, rows, label: str = "reference run"):
        self.label = label
        self._rows = rows

    def value(self, name, time_s, when=None):
        series = _series(self._rows, name)
        if not series:
            return None
        t = np.array([s[0] for s in series])
        v = np.array([s[1] for s in series])
        if time_s < t[0] - 1.0e-6 or time_s > t[-1] + 1.0e-6:
            return None
        return float(np.interp(float(time_s), t, v))


class TableReference:
    """A table of reference values by valid time (``{"valid_utc": ISO,
    <row name>: value, ...}`` records, e.g. the same global means taken
    from the analyses of the run's dates), linear in time between
    records, None outside them."""

    def __init__(self, records, label: str = "reference table"):
        self.label = label
        parsed = []
        for record in records:
            when = dt.datetime.fromisoformat(str(record["valid_utc"]).replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            parsed.append((when.timestamp(), record))
        parsed.sort(key=lambda item: item[0])
        self._records = parsed

    def value(self, name, time_s, when=None):
        if when is None:
            return None
        pairs = [(t, float(r[name])) for t, r in self._records if r.get(name) is not None]
        if not pairs:
            return None
        t = np.array([p[0] for p in pairs])
        v = np.array([p[1] for p in pairs])
        x = when.timestamp()
        if x < t[0] - 1.0 or x > t[-1] + 1.0:
            return None
        return float(np.interp(x, t, v))


def _bias_series(rows, name, reference, start_utc, from_day=0.0):
    """(day, model - reference) for every row past ``from_day`` where both
    exist."""
    if reference is None:
        return [], []
    t0 = float(rows[0]["time_s"])
    xs, ys = [], []
    for r in rows:
        day = (float(r["time_s"]) - t0) / SECONDS_PER_DAY
        value = _value(r, name)
        if value is None or day < from_day - 1.0e-9:
            continue
        ref = reference.value(name, float(r["time_s"]), _when(start_utc, r))
        if ref is None:
            continue
        xs.append(day)
        ys.append(value - float(ref))
    return xs, ys


def _drift_fits(rows, reference, start_utc):
    fits = {}
    for level in PRESSURE_LEVELS_HPA:
        xs, ys = _bias_series(rows, level_row_name(level), reference, start_utc,
                              from_day=LR5_FIT_START_DAY)
        span = (xs[-1] - xs[0]) if xs else 0.0
        fit = trend(xs, ys) if span >= LR5_MINIMUM_FIT_DAYS - 1.0e-9 else None
        if fit is not None:
            limit = LR5_SLOPE_K_PER_30D.get(level)
            fit.update(limit_per_30d=limit, fit_days=span,
                       mean_bias_k=float(np.mean(ys)),
                       passed=None if limit is None else abs(fit["slope_per_30d"]) < limit)
        fits[f"{level}hpa"] = fit
    return fits


def daily_means(rows, name, start_utc=None) -> list[dict[str, object]]:
    """The mean of a sampled row over each day's output instants: by UTC
    date when the run's start is known, else by whole model day from the
    first row.  ``[{"day", "samples", "mean", "mean_day"}]`` in time order,
    ``mean_day`` the mean of the samples' days since the first row."""
    if not rows:
        return []
    t0 = float(rows[0]["time_s"])
    groups: dict[object, list[float]] = {}
    times: dict[object, list[float]] = {}
    for r in rows:
        value = _value(r, name)
        if value is None:
            continue
        when = _when(start_utc, r)
        key = (when.strftime("%Y-%m-%d") if when is not None
               else int(math.floor((float(r["time_s"]) - t0) / SECONDS_PER_DAY + 1.0e-9)))
        groups.setdefault(key, []).append(value)
        times.setdefault(key, []).append((float(r["time_s"]) - t0) / SECONDS_PER_DAY)
    return [{"day": key, "samples": len(values), "mean": float(np.mean(values)),
             "mean_day": float(np.mean(times[key]))}
            for key, values in groups.items()]


def _slope_per_day(days, values) -> float | None:
    x = np.asarray(days, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return None
    xm = x - x.mean()
    sxx = float(np.sum(xm * xm))
    return None if sxx <= 0.0 else float(np.sum(xm * (y - y.mean())) / sxx)


def angular_momentum_gate(rows, spans, start_utc=None) -> dict[str, object]:
    """LR-10: the atmosphere's axial angular momentum per day and its
    drift, reported.  The rows are global means per unit area (kg/s); the
    table states the planet's totals (times 4 pi a^2, kg m2/s), each day's
    mean over its output instants, the least-squares drift of the daily
    means per day over the run and per window, and the change of each
    window's first instant to its last.  No threshold: the model has no
    discrete angular-momentum identity (audit 2026-09-01, DN-8), the
    relative part moves with the mountain and friction torques as it should,
    and a drift bound is to be set after a long run has measured it."""
    area = 4.0 * math.pi * EARTH_RADIUS_M ** 2
    names = {"axial": AXIAL_ANGULAR_MOMENTUM_ROW, "relative": RELATIVE_ANGULAR_MOMENTUM_ROW}
    t0 = float(rows[0]["time_s"])

    def series(subset, name):
        return [((float(t) - t0) / SECONDS_PER_DAY, v * area) for t, v in _series(subset, name)]

    if not _series(rows, AXIAL_ANGULAR_MOMENTUM_ROW):
        return _gate("angular momentum", "report", "reported", "not measured",
                     note=f"row {AXIAL_ANGULAR_MOMENTUM_ROW} is absent")
    days = {key: daily_means(rows, name, start_utc) for key, name in names.items()}
    table = [{"day": a["day"], "samples": a["samples"], "mean_day": a["mean_day"],
              "axial_kg_m2_s": a["mean"] * area,
              "relative_kg_m2_s": r["mean"] * area}
             for a, r in zip(days["axial"], days["relative"])]
    positions = [d["mean_day"] for d in table]

    def drift(values):
        return _slope_per_day(positions, values)

    axial = [d["axial_kg_m2_s"] for d in table]
    relative = [d["relative_kg_m2_s"] for d in table]
    axial_drift = drift(axial)
    windows_out = []
    for a, b, sub in spans:
        entry = {"start_day": a, "end_day": b}
        for key, name in names.items():
            s = series(sub, name)
            entry[f"{key}_change_kg_m2_s"] = (s[-1][1] - s[0][1]) if len(s) >= 2 else None
            entry[f"{key}_drift_kg_m2_s_per_day"] = _slope_per_day(
                [p[0] for p in s], [p[1] for p in s])
        windows_out.append(entry)
    mean_axial = float(np.mean(axial))
    return _gate(
        "angular momentum", "report", "reported (no threshold yet)", "report",
        days=table,
        axial_mean_kg_m2_s=mean_axial,
        axial_drift_kg_m2_s_per_day=axial_drift,
        axial_relative_drift_per_day=(None if axial_drift is None
                                      else axial_drift / max(abs(mean_axial), 1.0e-30)),
        relative_mean_kg_m2_s=float(np.mean(relative)),
        relative_drift_kg_m2_s_per_day=drift(relative),
        windows=windows_out,
        note=("axial = relative + planetary (int Omega a^2 cos^2(phi) dp/g, which "
              "moves with the mass fixer's global mass); totals are global means "
              "per unit area times 4 pi a^2; the daily drift is the least-squares "
              "slope of the daily means against their mean sample time"))


def evaluate(rows, *, start_utc: dt.datetime | None = None, reference=None,
             climatology=None, restart: dict | None = None,
             window_days: float = WINDOW_DAYS) -> dict[str, object]:
    """The gate table of a run's rows.

    ``reference`` is what LR-5 and LR-6 are GATED against and
    ``climatology`` what they are REPORTED against: each an object with
    ``value(row_name, time_s, when)`` returning the reference value of a
    diagnostics row (K) or None -- :class:`RunReference` (arm A for arm B),
    :class:`TableReference` (the analyses of the run's dates), or the
    report tool's daily long-term mean.  ``start_utc`` is the instant model
    time zero stands for (a reference by date needs it); ``restart`` the
    LR-0 comparison (:func:`restart_comparison`).
    """
    if not rows:
        raise ValueError("no diagnostics rows to judge")
    t0 = float(rows[0]["time_s"])
    run_days = (float(rows[-1]["time_s"]) - t0) / SECONDS_PER_DAY
    spans = windows(rows, window_days)
    gates: dict[str, object] = {}

    # LR-0
    threshold = "bitwise (a separate straight/segmented test, not the segmented arms)"
    if restart is None:
        gates["LR-0"] = _gate("restart", "required", threshold, "not measured",
                              note="no straight/segmented checkpoint pair given")
    else:
        gates["LR-0"] = _gate("restart", "required", threshold,
                              "pass" if restart["bit_identical"] else "fail", **restart)

    # LR-1
    def mass_range(subset):
        values = [v for _t, v in _series(subset, "global_mean_dry_surface_pressure_pa")]
        return (max(values) - min(values)) if values else None

    run_range = mass_range(rows)
    gates["LR-1"] = _gate(
        "mass", "report", f"dry-mass range <= {LR1_RANGE_PA:g} Pa over the run",
        "not measured" if run_range is None else
        ("pass" if run_range <= LR1_RANGE_PA else "fail"),
        run_range_pa=run_range,
        windows=[{"start_day": a, "end_day": b, "range_pa": mass_range(sub)}
                 for a, b, sub in spans],
        mass_relative_drift=relative_drift(rows, "global_mean_surface_pressure_pa"),
        total_water_relative_drift=relative_drift(rows, "global_mean_total_water_kg_m2"),
    )

    # LR-2
    parts: dict[str, object] = {}
    land = [v for _t, v in _series(rows, "minimum_land_surface_water_kg_m2")]
    if land:
        parts["land_minimum_kg_m2"] = min(land)
        parts["land_passed"] = min(land) >= LR2_LAND_MINIMUM_KG_M2
    ledger = []
    for r in rows:
        account = _value(r, EXCHANGE_ACCOUNT_ROW)
        net = _value(r, WATER_NET_EVAPORATION_ROW)
        if account is None or net is None:
            continue
        fixer = _value(r, WATER_FIXER_ON_WATER_ROW) or 0.0
        clipping = _value(r, WATER_CLIPPING_ON_WATER_ROW) or 0.0
        scale = max(abs(net), abs(fixer), abs(clipping), abs(account))
        if scale > 0.0:
            ledger.append(abs(account + net + fixer + clipping) / scale)
    if ledger:
        parts["ledger_maximum_relative"] = max(ledger)
        parts["ledger_passed"] = max(ledger) <= LR2_LEDGER_RELATIVE
    else:
        absent = [name for name in (EXCHANGE_ACCOUNT_ROW, WATER_NET_EVAPORATION_ROW)
                  if all(_value(r, name) is None for r in rows)]
        parts["ledger"] = ("not measured: " + (
            f"row {absent[0]} is absent" if len(absent) == 1 else
            f"rows {' and '.join(absent)} are absent" if absent else
            f"no row carries both {EXCHANGE_ACCOUNT_ROW} and "
            f"{WATER_NET_EVAPORATION_ROW}"))
    evaporation = [v for v in map(evaporation_watch, rows) if v is not None]
    if evaporation:
        parts["maximum_water_column_evaporation_24h_mm_day"] = max(evaporation)
        parts["evaporation_ceiling_mm_day"] = LR2_EVAPORATION_CEILING_MM_DAY
        parts["evaporation_ceiling_note"] = "provisional, reported and not enforced"
    # A required gate passes only on all of its evidence: a land minimum
    # that holds says nothing about the account's identity, so with the
    # ledger unmeasured the gate is "not measured" (a failing part still
    # fails it).
    verdicts = [parts[k] for k in ("land_passed", "ledger_passed") if k in parts]
    gates["LR-2"] = _gate(
        "reservoir", "required",
        f"land minimum >= {LR2_LAND_MINIMUM_KG_M2:g} kg/m2; account = "
        f"-int(E-P)_water - fixer - clipping to {LR2_LEDGER_RELATIVE:g} relative",
        "fail" if not all(verdicts) else
        ("pass" if len(verdicts) == 2 else "not measured"),
        **parts)

    # LR-3
    rows3 = []
    for a, b, sub in spans:
        rate = _difference_rate(sub, "area_mean_accumulated_precipitation_kg_m2")
        net = _difference_rate(sub, "global_mean_atmospheric_water_kg_m2")
        p = None if rate is None else rate * SECONDS_PER_DAY
        e_minus_p = None if net is None else net * SECONDS_PER_DAY
        rows3.append({
            "start_day": a, "end_day": b,
            "precipitation_mm_day": p,
            "precipitable_water_kg_m2": _time_mean(sub, "global_mean_precipitable_water_kg_m2"),
            "evaporation_minus_precipitation_mm_day": e_minus_p,
            "evaporation_mm_day": None if p is None or e_minus_p is None else p + e_minus_p,
        })
    measured = [w for w in rows3 if w["precipitation_mm_day"] is not None
                and w["precipitable_water_kg_m2"] is not None]
    gates["LR-3"] = _gate(
        "water cycle", "warn",
        f"P in {list(LR3_PRECIPITATION_MM_DAY)} mm/day; PW in "
        f"{list(LR3_PRECIPITABLE_WATER_KG_M2)} kg/m2",
        "not measured" if not measured else (
            "pass" if all(_within(w["precipitation_mm_day"], LR3_PRECIPITATION_MM_DAY)
                          and _within(w["precipitable_water_kg_m2"],
                                      LR3_PRECIPITABLE_WATER_KG_M2)
                          for w in measured) else "warn"),
        windows=rows3,
        note=("E - P is the global atmospheric water tendency (advection sums to "
              "zero over the globe); not a leak detector: 9 W/m2 of latent heat "
              "is 0.31 mm/day"))

    # LR-4 (the budgets also give LR-9)
    budgets = [{"start_day": a, "end_day": b, **(energy_budget(sub) or {}),
                # The surface-flux budget's water half, beside its energy.
                "evaporation_minus_precipitation_mm_day":
                    water["evaporation_minus_precipitation_mm_day"]}
               for (a, b, sub), water in zip(spans, rows3)]
    measured = [w for w in budgets if "residual_w_m2" in w]
    complete = bool(budgets) and len(measured) == len(budgets)
    gates["LR-4"] = _gate(
        "energy", "measured",
        f"measured and stated per window; |residual| > {LR4_RESIDUAL_FLAG_W_M2:g} W/m2 "
        f"flagged; <= {LR4_RESIDUAL_SLAB_W_M2:g} W/m2 required before a slab ocean",
        "not measured" if not complete else (
            "pass" if all(abs(w["residual_w_m2"]) <= LR4_RESIDUAL_FLAG_W_M2
                          for w in measured) else "flag"),
        slab_prerequisite_met=(None if not complete else all(
            abs(w["residual_w_m2"]) <= LR4_RESIDUAL_SLAB_W_M2 for w in measured)),
        windows=budgets,
        note=("residual = TOA net - surface net radiation + H + LE + Lf S - dE/dt; "
              "radiation is the exact interval mean of the accumulators, H and LE "
              "the held fluxes sampled at the output instants"))

    # LR-5
    fits = _drift_fits(rows, reference, start_utc) if reference is not None else {}
    climate_fits = (_drift_fits(rows, climatology, start_utc)
                    if climatology is not None else {})
    verdicts = [f["passed"] for f in fits.values() if f and f["passed"] is not None]
    note = None
    if reference is None:
        note = "no gating reference given (a reference run or the analyses of the run's dates)"
    elif not verdicts:
        note = (f"needs at least {LR5_MINIMUM_FIT_DAYS:g} days of rows past day "
                f"{LR5_FIT_START_DAY:g} with a reference value")
    gates["LR-5"] = _gate(
        "temperature drift", "required",
        "|slope| < 0.1 K/30 d at 850-200 hPa, < 0.5 K/30 d at 10 hPa, "
        f"fitted from day {LR5_FIT_START_DAY:g} over >= {LR5_MINIMUM_FIT_DAYS:g} days",
        "not measured" if not verdicts else ("pass" if all(verdicts) else "fail"),
        reference=None if reference is None else getattr(reference, "label", "given"),
        slopes=fits,
        climatology=None if climatology is None else getattr(climatology, "label", "given"),
        climatology_slopes=climate_fits,
        **({"note": note} if note else {}))

    # LR-6
    minimum = [v for _t, v in _series(rows, "minimum_temperature_k")]
    lowest = min(minimum) if minimum else None

    def cap_biases(ref):
        out = {}
        if ref is None:
            return out
        for level in POLAR_CAP_LEVELS_HPA:
            for hemisphere in ("north", "south"):
                name = cap_row_name(level, hemisphere)
                per_window = []
                for a, b, sub in spans:
                    _xs, ys = _bias_series(sub, name, ref, start_utc)
                    per_window.append({"start_day": a, "end_day": b,
                                       "mean_bias_k": float(np.mean(ys)) if ys else None})
                out[f"{level}hpa_{hemisphere}"] = per_window
        return out

    gated = cap_biases(reference)
    measured_bias = [w["mean_bias_k"] for series in gated.values() for w in series
                     if w["mean_bias_k"] is not None]
    if lowest is not None and lowest < LR6_MINIMUM_TEMPERATURE_K:
        verdict6 = "fail"
    elif not measured_bias:
        verdict6 = "not measured"
    else:
        verdict6 = ("pass" if all(abs(b) <= LR6_CAP_BIAS_PROPOSED_K for b in measured_bias)
                    else "fail")
    gates["LR-6"] = _gate(
        "stratosphere", "required",
        f"|polar-cap bias| at 10 and 50 hPa <= {LR6_CAP_BIAS_PROPOSED_K:g} K per window "
        f"(proposed, to be set after V1); minimum T < {LR6_MINIMUM_TEMPERATURE_K:g} K "
        "refuses (a refusal only, never a pass)",
        verdict6,
        minimum_temperature_k=lowest,
        cap_bias=gated or None,
        climatology_cap_bias=cap_biases(climatology) or None,
        windows=[{
            "start_day": a, "end_day": b,
            "minimum_temperature_k": min(
                (v for _t, v in _series(sub, "minimum_temperature_k")), default=None),
            "zonal_mean_u_10hpa_60n_m_s": _time_mean(sub, "zonal_mean_u_10hpa_60n_m_s"),
            "zonal_mean_u_10hpa_60s_m_s": _time_mean(sub, "zonal_mean_u_10hpa_60s_m_s"),
        } for a, b, sub in spans])

    # LR-7
    def soil_change(sub):
        rate = _difference_rate(sub, "land_mean_soil_water_kg_m2")
        return None if rate is None else rate * SECONDS_PER_DAY * WINDOW_DAYS

    gates["LR-7"] = _gate(
        "land", "report", "reported", "report",
        windows=[{
            "start_day": a, "end_day": b,
            "soil_water_change_kg_m2_per_30d": soil_change(sub),
            "nh_snow_covered_area_km2": _time_mean(sub, "nh_snow_covered_area_km2"),
        } for a, b, sub in spans])

    # LR-8
    departure = [v for _t, v in _series(rows, BOUNDARY_SST_DEPARTURE_ROW)]
    mismatch = [v for _t, v in _series(rows, BOUNDARY_ICE_MISMATCH_ROW)]
    threshold = f"max |SST - provider| <= {LR8_SST_DEPARTURE_K:g} K; ice equals the record"
    if departure or mismatch:
        ok = ((not departure or max(departure) <= LR8_SST_DEPARTURE_K)
              and (not mismatch or max(mismatch) == 0.0))
        gates["LR-8"] = _gate(
            "boundary", "required", threshold, "pass" if ok else "fail",
            maximum_sst_departure_k=max(departure) if departure else None,
            maximum_ice_mismatch_columns=max(mismatch) if mismatch else None)
    else:
        gates["LR-8"] = _gate(
            "boundary", "required", threshold, "not measured",
            note=f"rows {BOUNDARY_SST_DEPARTURE_ROW} and {BOUNDARY_ICE_MISMATCH_ROW} are absent")

    # LR-9: reported per window; gated on a full-year mean only.
    annual = None
    if run_days >= LR9_ANNUAL_DAYS - 1.0e-9:
        last = float(rows[-1]["time_s"])
        year = [r for r in rows
                if float(r["time_s"]) >= last - LR9_ANNUAL_DAYS * SECONDS_PER_DAY - 1.0e-6]
        annual = energy_budget(year)
    if annual is None:
        verdict9 = "report"
    else:
        verdict9 = "pass" if _within(annual["toa_net_w_m2"], LR9_TOA_NET_W_M2) else "fail"
    gates["LR-9"] = _gate(
        "TOA", "annual",
        f"full-year mean TOA net in {list(LR9_TOA_NET_W_M2)} W/m2; windows reported",
        verdict9,
        annual_toa_net_w_m2=None if annual is None else annual["toa_net_w_m2"],
        annual_outgoing_longwave_w_m2=(None if annual is None
                                       else annual["outgoing_longwave_w_m2"]),
        windows=[{"start_day": w["start_day"], "end_day": w["end_day"],
                  "toa_net_w_m2": w.get("toa_net_w_m2"),
                  "outgoing_longwave_w_m2": w.get("outgoing_longwave_w_m2")}
                 for w in budgets],
        note=("a 30-day window is not held to an annual target: insolation swings "
              "about 16 W/m2 absorbed over the year with the Earth-Sun distance"))

    gates["LR-10"] = angular_momentum_gate(rows, spans, start_utc)

    return _finite({
        "schema": "gpuwm.arwen-global-longrun-gates/v1",
        "rows": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "days": run_days,
        "window_days": float(window_days),
        "start_utc": None if start_utc is None else start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gates": gates,
        "note": ("the thresholds other than LR-0 and LR-8 are engineering proposals "
                 "(long-run design section 11), not literature standards; the "
                 "gates are outside the receipt's status"),
    })


def _finite(value):
    """The table with every non-finite float as None: receipts and the
    diagnostics file are strict JSON."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def restart_comparison(straight, segmented) -> dict[str, object]:
    """LR-0: two checkpoints of one step, by their self hashes and then
    array by array (bytes, never a tolerance)."""
    import hashlib

    def load(path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
            arrays = {name: archive[name] for name in archive.files
                      if name != "__metadata__"}
        return metadata, arrays

    meta_a, arrays_a = load(straight)
    meta_b, arrays_b = load(segmented)
    differing = []
    for name in sorted(set(arrays_a) | set(arrays_b)):
        a, b = arrays_a.get(name), arrays_b.get(name)
        if a is None or b is None or a.dtype != b.dtype or a.shape != b.shape:
            differing.append(name)
            continue
        if hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).digest() != \
                hashlib.sha256(np.ascontiguousarray(b).view(np.uint8)).digest():
            differing.append(name)
    same_header = meta_a.get("self_sha256") == meta_b.get("self_sha256")
    return {
        "straight": str(straight), "segmented": str(segmented),
        "step": [meta_a.get("step"), meta_b.get("step")],
        "arrays": len(set(arrays_a) | set(arrays_b)),
        "differing_arrays": differing,
        "self_sha256_equal": bool(same_header),
        "bit_identical": bool(same_header and not differing),
    }


def receipt_block(diagnostics_path, start_utc=None) -> dict[str, object]:
    """The runner's ``longrun_gates`` receipt block: the table over every
    row the run's folder holds so far, no reference, no restart pair."""
    try:
        return evaluate(read_rows([diagnostics_path]), start_utc=start_utc)
    except Exception as exc:  # noqa: BLE001 - an instrument never fails the run
        return {"error_type": type(exc).__name__, "error_message": str(exc)}


__all__ = [
    "ATMOSPHERE_ZONAL_NAMES", "AXIAL_ANGULAR_MOMENTUM_ROW", "PRESSURE_LEVELS_HPA",
    "RELATIVE_ANGULAR_MOMENTUM_ROW", "angular_momentum_gate", "angular_momentum_rows",
    "area_mean", "daily_means",
    "atmosphere_band_zonal_means", "atmosphere_rows", "cell_areas_m2",
    "energy_budget", "evaluate", "evaporation_watch", "global_mean_of_zonal",
    "interpolate_to_pressure",
    "RunReference", "TableReference", "cap_row_name", "level_row_name",
    "read_rows", "receipt_block", "relative_drift", "restart_comparison",
    "surface_rows", "trend", "windows", "zonal_area_mean", "zonal_at_latitude",
]
