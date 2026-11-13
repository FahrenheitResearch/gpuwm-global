"""Long-run rows and gates LR-0 to LR-10 (arwen_global.longrun_gates), on
cases whose answers are known in closed form.

* The area mean of cos(lat) on a Gaussian grid is pi/4 to the quadrature's
  own accuracy, and converges with the grid; the flat mean over the
  grid's points, which the physics' per-call radiation readings use, does
  not.
* Interpolation to pressure is exact for a field linear in ln p, follows
  the standard lapse rate below the ground and flags a level above the
  model top.
* A drift of 0.2 K per 30 days planted on a seasonal reference is read
  back as 0.2 to 1e-9 and fails LR-5; 0.05 passes; a climatology is
  fitted and never gated, and a run too short for the 60-day fit is not
  measured.  LR-6 gates the polar-cap bias and refuses on the minimum
  temperature, which is never a pass by itself.
* The dry-mass row is ps - g W on the smoke model's own state, the band
  count does not move a long-run row, and the Nino-3.4 box mean of an SST
  of 300 + latitude is 300.
* The energy budget closes to zero on fluxes that balance and reads a
  planted leak back; precipitation is read from its accumulator.
* A segmented run's repeated opening row is dropped, the restart gate
  compares bytes, and a run longer than 384 h carries the table in its
  receipt.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from arwen_global import longrun_gates as lg
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.constants import GRAVITY_M_S2, LATENT_HEAT_FUSION
from arwen_global.spectral.grid import GaussianGrid

SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
BANDED = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")


# ------------------------------------------------------------ area weights


def test_area_mean_of_cos_lat_is_pi_over_4_and_the_flat_mean_is_not():
    # T3: six Gaussian latitudes.  The area mean is the Gauss-weighted sum
    # to float64; it differs from pi/4 by the quadrature's own error on
    # sqrt(1 - mu^2), which is not a polynomial (1.95e-3 relative), and the
    # error shrinks with every refinement; the flat mean over the points
    # differs from both by far more.
    errors = []
    for truncation in (3, 21, 63):
        grid = GaussianGrid.create(truncation)
        plane = np.repeat(grid.cos_lat[:, None], grid.nlon, axis=1)
        mean = lg.area_mean(grid, plane)
        assert mean == pytest.approx(
            float(np.sum(grid.quadrature_weights * grid.cos_lat) / 2.0), rel=1e-15)
        errors.append(abs(mean - math.pi / 4.0) / (math.pi / 4.0))
        flat = float(plane.mean())
        assert abs(flat - math.pi / 4.0) / (math.pi / 4.0) > 10.0 * errors[-1]
    assert GaussianGrid.create(3).nlat == 6
    assert errors[0] == pytest.approx(1.95e-3, rel=0.01)
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1.0e-4


def test_global_mean_from_zonal_means_is_the_grid_global_mean():
    grid = GaussianGrid.create(21)
    plane = np.random.default_rng(3).standard_normal(grid.shape)
    assert lg.global_mean_of_zonal(grid, plane.mean(axis=-1)) == pytest.approx(
        grid.global_mean(plane), rel=0.0, abs=1.0e-15)
    # The cell areas sum to the sphere.
    assert lg.cell_areas_m2(grid).sum() * grid.nlon == pytest.approx(
        4.0 * math.pi * grid.radius_m ** 2, rel=1.0e-12)


# ----------------------------------------------------------- interpolation


def test_interpolation_to_pressure_is_exact_in_ln_p_and_lapses_below_ground():
    p = np.array([500.0, 5000.0, 30000.0, 60000.0, 90000.0])[:, None, None]
    p = np.broadcast_to(p, (5, 2, 3)).copy()
    p[:, 1, :] *= 0.8  # a high-ground row: lowest full level at 720 hPa
    field = 200.0 + 10.0 * np.log(p)
    value, over = lg.interpolate_to_pressure(np, field, p, 50000.0, extrapolate="hold")
    assert np.allclose(value, 200.0 + 10.0 * math.log(50000.0), rtol=0, atol=1e-10)
    assert not over.any()
    t, _ = lg.interpolate_to_pressure(np, field, p, 85000.0, extrapolate="lapse")
    lowest = field[-1, 1, 0]
    expected = lowest * (85000.0 / 72000.0) ** (
        lg.DRY_AIR_GAS_CONSTANT * lg.STANDARD_LAPSE_RATE_K_M / GRAVITY_M_S2)
    assert np.allclose(t[1], expected, rtol=1e-14)
    assert np.allclose(t[0], 200.0 + 10.0 * math.log(85000.0), atol=1e-10)
    _, over = lg.interpolate_to_pressure(np, field, p, 300.0, extrapolate="hold")
    assert over.all()


def test_a_level_above_the_model_top_is_null_never_invented():
    grid = GaussianGrid.create(3)
    zonal = np.zeros((len(lg.ATMOSPHERE_ZONAL_NAMES), grid.nlat))
    zonal[lg.ATMOSPHERE_ZONAL_NAMES.index("temperature_10")] = 230.0
    zonal[lg.ATMOSPHERE_ZONAL_NAMES.index("over_top_10")][0] = 0.5
    rows = lg.atmosphere_rows(grid, zonal, 1.0e5, 25.0)
    assert rows[lg.level_row_name(10)] is None
    assert rows["polar_cap_mean_temperature_10hpa_north_k"] is None
    assert rows["zonal_mean_u_10hpa_60s_m_s"] is None
    assert rows[lg.level_row_name(850)] == 0.0
    assert rows["global_mean_dry_surface_pressure_pa"] == pytest.approx(
        1.0e5 - GRAVITY_M_S2 * 25.0, rel=0, abs=1e-9)


# ----------------------------------------------------------------- LR-5


class _Seasonal:
    """A reference by date: a seasonal cycle per level, fixed polar caps."""

    label = "synthetic seasonal reference"

    def __init__(self, cap_k=200.0):
        self.cap_k = cap_k

    def global_mean(self, level, when):
        day = (when - dt.datetime(when.year, 1, 1, tzinfo=when.tzinfo)).total_seconds() / 86400.0
        return 250.0 - level / 100.0 + 8.0 * math.sin(2.0 * math.pi * day / 365.0)

    def value(self, name, time_s, when):
        for level in lg.PRESSURE_LEVELS_HPA:
            if name == lg.level_row_name(level):
                return self.global_mean(level, when)
            if name in (lg.cap_row_name(level, "north"), lg.cap_row_name(level, "south")):
                return self.cap_k
        return None


def _drifting_rows(slope_per_30d, days=90, every_h=6, cap_k=204.0, minimum_k=180.0):
    start = dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
    reference = _Seasonal()
    rows = []
    for i in range(int(days * 24 / every_h) + 1):
        time_s = i * every_h * 3600.0
        when = start + dt.timedelta(seconds=time_s)
        row = {"step": i, "time_s": time_s, "minimum_temperature_k": minimum_k}
        for level in lg.PRESSURE_LEVELS_HPA:
            row[lg.level_row_name(level)] = (
                reference.global_mean(level, when) + 1.5
                + slope_per_30d * (time_s / 86400.0) / 30.0)
        for level in lg.POLAR_CAP_LEVELS_HPA:
            for hemisphere in ("north", "south"):
                row[lg.cap_row_name(level, hemisphere)] = cap_k
        rows.append(row)
    return rows, start, reference


@pytest.mark.parametrize("slope, verdict", ((0.2, "fail"), (0.05, "pass")))
def test_lr5_reads_back_a_planted_drift_on_a_seasonal_reference(slope, verdict):
    rows, start, reference = _drifting_rows(slope)
    report = lg.evaluate(rows, start_utc=start, reference=reference)
    gate = report["gates"]["LR-5"]
    for level in lg.PRESSURE_LEVELS_HPA:
        fit = gate["slopes"][f"{level}hpa"]
        assert fit["slope_per_30d"] == pytest.approx(slope, rel=0, abs=1.0e-9)
        assert fit["mean_bias_k"] == pytest.approx(1.5 + slope * 2.0, abs=1e-9)
        # From day 30 to day 90: 60 days of 6-hourly rows, both ends included.
        assert fit["samples"] == 241 and fit["fit_days"] == pytest.approx(60.0)
        assert fit["ci95_half_width_per_30d"] < 1.0e-9
    assert gate["slopes"]["50hpa"]["passed"] is None  # reported, no limit
    assert gate["verdict"] == verdict
    json.dumps(report, allow_nan=False)


def test_lr5_is_not_gated_on_a_climatology_or_on_a_short_run():
    rows, start, reference = _drifting_rows(0.2)
    # The same reference handed over as a climatology is fitted and
    # reported, never gated.
    reported = lg.evaluate(rows, start_utc=start, climatology=reference)["gates"]["LR-5"]
    assert reported["verdict"] == "not measured"
    assert reported["climatology_slopes"]["850hpa"]["slope_per_30d"] == pytest.approx(
        0.2, abs=1e-9)
    assert lg.evaluate(rows)["gates"]["LR-5"]["verdict"] == "not measured"
    # 80 days: 50 days past day 30, under the 60 the gate needs.
    short, start, reference = _drifting_rows(0.2, days=80)
    gate = lg.evaluate(short, start_utc=start, reference=reference)["gates"]["LR-5"]
    assert gate["verdict"] == "not measured" and "60" in gate["note"]


def test_arm_b_against_arm_a_reads_the_difference_of_the_two_runs():
    arm_a, _start, _ref = _drifting_rows(0.0)
    arm_b, _start, _ref = _drifting_rows(0.3)
    gate = lg.evaluate(arm_b, reference=lg.RunReference(arm_a, "arm A"))["gates"]["LR-5"]
    assert gate["slopes"]["500hpa"]["slope_per_30d"] == pytest.approx(0.3, abs=1e-9)
    assert gate["verdict"] == "fail" and gate["reference"] == "arm A"


def test_lr6_gates_the_cap_bias_and_refuses_on_minimum_temperature_only():
    rows, start, reference = _drifting_rows(0.0, cap_k=204.0)
    gates = lg.evaluate(rows, start_utc=start, reference=reference)["gates"]
    assert gates["LR-6"]["verdict"] == "pass"
    assert gates["LR-6"]["cap_bias"]["50hpa_south"][0]["mean_bias_k"] == pytest.approx(4.0)
    rows, start, reference = _drifting_rows(0.0, cap_k=212.0)
    assert lg.evaluate(rows, start_utc=start, reference=reference)[
        "gates"]["LR-6"]["verdict"] == "fail"
    # A minimum above 165 K with no reference is not a pass; below it, a
    # refusal whatever the caps read.
    assert lg.evaluate(rows)["gates"]["LR-6"]["verdict"] == "not measured"
    # A cap 15 K colder than its reference fails with the minimum above 165 K.
    colder, start, reference = _drifting_rows(0.0, cap_k=185.0, minimum_k=170.0)
    assert lg.evaluate(colder, start_utc=start, reference=reference)[
        "gates"]["LR-6"]["verdict"] == "fail"
    cold, start, reference = _drifting_rows(0.0, cap_k=204.0, minimum_k=164.0)
    assert lg.evaluate(cold, start_utc=start, reference=reference)[
        "gates"]["LR-6"]["verdict"] == "fail"


def _noisy_drift(slope_per_30d, r1, seed, days=90.0, every_h=6.0, sigma=0.15):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, days + 1e-9, every_h / 24.0)
    noise = np.empty_like(t)
    noise[0] = rng.normal(0.0, sigma)
    for i in range(1, t.size):
        noise[i] = r1 * noise[i - 1] + math.sqrt(1.0 - r1 * r1) * rng.normal(0.0, sigma)
    return t, slope_per_30d * t / 30.0 + noise


def test_white_noise_drift_lies_in_its_interval_and_fails_lr5():
    t, bias = _noisy_drift(0.2, 0.0, seed=11)
    keep = t >= 30.0
    fit = lg.trend(t[keep], bias[keep])
    assert abs(fit["slope_per_30d"] - 0.2) < fit["ci95_half_width_per_30d"]
    assert abs(fit["lag1_autocorrelation"]) < 0.2
    start = dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
    reference = _Seasonal()
    for slope, verdict in ((0.2, "fail"), (0.05, "pass")):
        t, bias = _noisy_drift(slope, 0.0, seed=12)
        rows = []
        for i, day in enumerate(t):
            when = start + dt.timedelta(days=float(day))
            row = {"step": i, "time_s": float(day) * 86400.0}
            for level in lg.PRESSURE_LEVELS_HPA:
                row[lg.level_row_name(level)] = (
                    reference.global_mean(level, when) + 0.02 * bias[i])
            # 850 hPa carries the drift at full amplitude.
            row[lg.level_row_name(850)] = reference.global_mean(850, when) + bias[i]
            rows.append(row)
        gate = lg.evaluate(rows, start_utc=start, reference=reference)["gates"]["LR-5"]
        assert gate["verdict"] == verdict, (slope, gate["slopes"]["850hpa"])


def test_ar1_noise_widens_the_interval_by_the_effective_sample_size():
    t, bias = _noisy_drift(0.1, 0.8, seed=5)
    fit = lg.trend(t, bias)
    n, n_eff = fit["samples"], fit["effective_samples"]
    assert fit["lag1_autocorrelation"] == pytest.approx(0.8, abs=0.1)
    assert n_eff == pytest.approx(
        n * (1.0 - fit["lag1_autocorrelation"]) / (1.0 + fit["lag1_autocorrelation"]))
    factor = (math.sqrt((n - 2.0) / (n_eff - 2.0))
              * lg._t_quantile_975(n_eff - 2.0) / lg._t_quantile_975(n - 2.0))
    assert fit["ci95_half_width_per_30d"] == pytest.approx(
        factor * fit["ols_ci95_half_width_per_30d"], rel=1e-12)
    assert factor > 2.5


# -------------------------------------------------- the model's own rows


@pytest.fixture(scope="module")
def smoke():
    from arwen_global.config import load_config
    from arwen_global.runner import build_model_and_cold_state

    cfg = load_config(SMOKE)
    model, cold = build_model_and_cold_state(cfg)
    return cfg, model, cold


def test_dry_mass_row_is_ps_minus_g_times_the_column_water(smoke):
    from arwen_global.constants import WATER_SPECIES

    _cfg, model, cold = smoke
    diag = model.diagnostics(cold)
    grid = model.transform.grid
    g = model.grid_state(cold.atmosphere, only=("ps", "dp", *WATER_SPECIES))
    ps = np.asarray(g["ps"], dtype=np.float64)
    water = sum(np.asarray(g[n], dtype=np.float64) for n in WATER_SPECIES)
    column = (water * np.asarray(g["dp"], dtype=np.float64)).sum(axis=0) / GRAVITY_M_S2
    expected = grid.global_mean(ps) - GRAVITY_M_S2 * grid.global_mean(column)
    assert diag["global_mean_dry_surface_pressure_pa"] == pytest.approx(
        expected, rel=1.0e-13)
    assert 0.0 < diag["global_mean_surface_pressure_pa"] - expected < 1.0e3
    vapour = (np.asarray(g["qv"], dtype=np.float64)
              * np.asarray(g["dp"], dtype=np.float64)).sum(axis=0) / GRAVITY_M_S2
    assert diag["global_mean_precipitable_water_kg_m2"] == pytest.approx(
        grid.global_mean(vapour), rel=1.0e-13)
    json.dumps(diag, allow_nan=False)


def test_level_temperature_row_is_the_column_interpolation_area_mean(smoke):
    _cfg, model, cold = smoke
    diag = model.diagnostics(cold)
    grid = model.transform.grid
    g = model.grid_state(cold.atmosphere, only=("temperature", "p_full"))
    t = np.asarray(g["temperature"], dtype=np.float64)
    p = np.asarray(g["p_full"], dtype=np.float64)
    plane = np.empty(grid.shape)
    for j in range(grid.nlat):
        for i in range(grid.nlon):
            plane[j, i] = np.interp(math.log(50000.0), np.log(p[:, j, i]), t[:, j, i])
    assert diag[lg.level_row_name(500)] == pytest.approx(
        grid.global_mean(plane), rel=1.0e-12)


def test_the_band_count_does_not_move_a_long_run_row():
    from arwen_global.config import load_config
    from arwen_global.runner import build_model_and_cold_state

    cfg = load_config(BANDED)
    readings = []
    for bands in (1, 4):
        model, cold = build_model_and_cold_state(replace(cfg, latitude_bands=bands))
        readings.append(model.diagnostics(cold))
    names = [k for k in readings[0]
             if k.startswith(("global_mean_", "polar_cap_", "zonal_mean_u_"))]
    assert lg.level_row_name(850) in names
    assert {k: readings[0][k] for k in names} == {k: readings[1][k] for k in names}


def test_nino34_box_mean_of_300_plus_latitude_is_300():
    grid = GaussianGrid.create(63)
    lat = np.repeat(grid.latitude_deg[:, None], grid.nlon, axis=1)
    surface = {
        "land_fraction": np.zeros(grid.shape),
        "water_kg_m2": np.full(grid.shape, 500.0),
        "temperature_k": 300.0 + lat,
        # Ice of fraction 0.9 over the whole northern ocean, and 0.1 (under
        # the 15% extent threshold) over the southern.
        "sea_ice_fraction": np.where(lat > 0.0, 0.9, 0.1),
    }
    rows = lg.surface_rows(grid, surface, {}, None)
    assert abs(rows["nino34_mean_sst_k"] - 300.0) < 1.0e-12
    # On an ocean planet the northern extent is the hemisphere, 2 pi a^2
    # (the Gaussian weights of a hemisphere sum to one), its ice area 0.9
    # of it; the southern ice counts in area and not in extent; and no land
    # row is invented on a planet without land.
    hemisphere = 2.0 * math.pi * grid.radius_m ** 2 / 1e6
    assert rows["nh_sea_ice_extent_km2"] == pytest.approx(hemisphere, rel=1e-12)
    assert rows["nh_sea_ice_area_km2"] == pytest.approx(0.9 * hemisphere, rel=1e-12)
    assert rows["sh_sea_ice_extent_km2"] == 0.0
    assert rows["sh_sea_ice_area_km2"] == pytest.approx(0.1 * hemisphere, rel=1e-12)
    assert "minimum_land_surface_water_kg_m2" not in rows


# ------------------------------------------------------ budgets and LR-3


def _budget_rows(leak_w_m2=0.0, days=60, every_h=6):
    """Area-mean rows of a planet whose fluxes balance: TOA net +1, net
    surface radiation 100, H 20, LE 79, frozen precipitation 0.01 mm/day
    (Lf S = 0.0387 W/m2), and the atmosphere's energy gaining the sum plus
    a planted leak.  Precipitation 2.7 mm/day, PW 25 kg/m2."""
    snow_rate = 0.01 / 86400.0
    gain = 1.0 - 100.0 + 20.0 + 79.0 + LATENT_HEAT_FUSION * snow_rate + leak_w_m2
    rows = []
    for i in range(int(days * 24 / every_h) + 1):
        t = i * every_h * 3600.0
        rows.append({
            "step": i, "time_s": t,
            "radiation_accumulated_s": t,
            "area_mean_acc_sw_down_top_j_m2": 340.0 * t,
            "area_mean_acc_sw_up_top_j_m2": 99.0 * t,
            "area_mean_acc_lw_up_top_j_m2": 240.0 * t,
            "area_mean_acc_sw_down_surface_j_m2": 185.0 * t,
            "area_mean_acc_sw_up_surface_j_m2": 25.0 * t,
            "area_mean_acc_lw_down_surface_j_m2": 340.0 * t,
            "area_mean_acc_lw_up_surface_j_m2": 400.0 * t,
            "area_mean_sensible_heat_flux_w_m2": 20.0 + 5.0 * math.sin(i),
            "area_mean_latent_heat_flux_w_m2": 79.0,
            "area_mean_accumulated_frozen_precipitation_kg_m2": snow_rate * t,
            "area_mean_accumulated_precipitation_kg_m2": 2.7 * t / 86400.0,
            "global_mean_precipitable_water_kg_m2": 25.0,
            # E - P of +0.1 mm/day: the atmosphere moistens.
            "global_mean_atmospheric_water_kg_m2": 25.1 + 0.1 * t / 86400.0,
            "global_mean_moist_total_energy_j_m2": 2.5e9 + gain * t,
            "global_mean_dry_surface_pressure_pa": 98325.0 + 5.0 * math.sin(t / 8.64e5),
            "global_mean_surface_pressure_pa": 98570.0,
            "global_mean_total_water_kg_m2": 709.6,
            "minimum_land_surface_water_kg_m2": 480.0,
            "minimum_temperature_k": 181.0,
        })
    return rows


def test_energy_residual_closes_on_balanced_fluxes_and_reads_a_planted_leak():
    balanced = lg.evaluate(_budget_rows())["gates"]
    for window in balanced["LR-4"]["windows"]:
        # H carries a sampled wiggle whose trapezoid mean is not exactly 20.
        sampled = window["sensible_heat_flux_w_m2"] - 20.0
        assert window["residual_w_m2"] == pytest.approx(sampled, abs=1.0e-6)
        assert window["toa_net_w_m2"] == pytest.approx(1.0, abs=1e-9)
        assert window["surface_net_radiation_w_m2"] == pytest.approx(100.0, abs=1e-9)
    assert balanced["LR-4"]["verdict"] == "pass"
    # Sixty days: TOA net and OLR are reported per window, not gated.
    assert balanced["LR-9"]["verdict"] == "report"
    assert balanced["LR-9"]["windows"][1]["toa_net_w_m2"] == pytest.approx(1.0, abs=1e-9)
    leaking = lg.evaluate(_budget_rows(leak_w_m2=-9.3))["gates"]
    for window, base in zip(leaking["LR-4"]["windows"], balanced["LR-4"]["windows"]):
        assert window["residual_w_m2"] - base["residual_w_m2"] == pytest.approx(9.3, abs=1e-6)
    assert leaking["LR-4"]["verdict"] == "flag"
    assert leaking["LR-4"]["slab_prerequisite_met"] is False


def test_a_cold_start_row_without_accumulators_still_opens_the_first_window():
    """The model's step-0 row carries no radiation accumulator and no flux
    (no physics call has run); the accumulators begin at zero, so the first
    window's radiation terms are the same as with explicit zeros."""
    rows = _budget_rows(days=30)
    cold = {k: v for k, v in rows[0].items()
            if not k.startswith(("area_mean_acc_", "area_mean_sensible",
                                 "area_mean_latent"))
            and k != "radiation_accumulated_s"}
    gates = lg.evaluate([cold] + rows[1:])["gates"]
    window = gates["LR-4"]["windows"][0]
    assert window["toa_net_w_m2"] == pytest.approx(1.0, abs=1e-9)
    assert window["surface_net_radiation_w_m2"] == pytest.approx(100.0, abs=1e-9)
    assert window["latent_heat_flux_w_m2"] == pytest.approx(79.0, abs=1e-12)
    assert gates["LR-9"]["windows"][0]["outgoing_longwave_w_m2"] == pytest.approx(
        240.0, abs=1e-9)
    # A row at a later instant that lacks an accumulator is not read as zero.
    late = [dict(r) for r in rows[4:]]
    del late[0]["area_mean_acc_sw_down_top_j_m2"]
    assert lg.energy_budget(late) is None


def test_lr2_reads_the_native_suites_24h_evaporation_watch():
    """The largest 24 h open-water evaporation is the native suite's watch,
    written in each physics half-call's diagnostics: the running block and
    the last complete block, kg/m2 per 24 h block = mm/day."""
    def row(step, first, second):
        halves = {
            "first_half_physics": {"physics": dict(zip(lg.EVAPORATION_WATCH_ROWS, first))},
            "second_half_physics": {"physics": dict(zip(lg.EVAPORATION_WATCH_ROWS, second))},
        }
        return {"step": step, "time_s": step * 21600.0, "step_metrics": halves,
                "minimum_land_surface_water_kg_m2": 480.0}

    rows = [row(0, (0.0, 0.0), (0.0, 0.0)), row(1, (12.5, 0.0), (12.6, 0.0)),
            row(2, (24.9, 0.0), (0.04, 34.7)), row(3, (8.0, 34.7), (8.1, 34.7))]
    assert [lg.evaporation_watch(r) for r in rows] == [0.0, 12.6, 34.7, 34.7]
    lr2 = lg.evaluate(rows)["gates"]["LR-2"]
    assert lr2["maximum_water_column_evaporation_24h_mm_day"] == 34.7
    # A row of its own wins over the watch.
    assert lg.evaporation_watch(dict(rows[1], **{lg.WATER_EVAPORATION_24H_ROW: 3.0})) == 3.0


def test_the_rows_the_model_writes_are_the_rows_the_gates_read():
    """surface_rows on a namespace whose fluxes are fixed planes, at two
    instants a day apart: the gates read back each plane's AREA mean as
    the interval flux, under the names the writer used."""
    grid = GaussianGrid.create(21)
    coslat = np.repeat(grid.cos_lat[:, None], grid.nlon, axis=1)
    fluxes = {
        "swdnt": 1361.0 / 4.0 * 1.2 * coslat, "swupt": 100.0 + 0.0 * coslat,
        "olr": 200.0 + 50.0 * coslat, "swdown": 200.0 * coslat,
        "gsw": 160.0 * coslat, "glw": 330.0 + 0.0 * coslat,
        "lwupb": 390.0 + 0.0 * coslat,
    }
    from arwen_global.physics.native_state import RADIATION_ACCUMULATORS

    surface = {
        "land_fraction": np.zeros(grid.shape), "water_kg_m2": np.full(grid.shape, 500.0),
        "temperature_k": np.full(grid.shape, 290.0),
        "accumulated_rain_kg_m2": np.zeros(grid.shape),
        "accumulated_snow_kg_m2": np.zeros(grid.shape),
        "accumulated_graupel_kg_m2": np.zeros(grid.shape),
    }
    rows = []
    for step, seconds in ((0, 0.0), (1, 86400.0)):
        planes = dict(fluxes, swupb=fluxes["swdown"] - fluxes["gsw"])
        physics = {acc: seconds * planes[name]
                   for acc, name in RADIATION_ACCUMULATORS.items()
                   if name in planes}
        physics.update(fluxes)  # the held planes, read at the instant
        physics.update(hfx=np.full(grid.shape, 20.0), qfx=np.full(grid.shape, 3.0e-5))
        surface["accumulated_rain_kg_m2"] = np.full(grid.shape, 2.9 * seconds / 86400.0)
        row = {"step": step, "time_s": seconds,
               "global_mean_moist_total_energy_j_m2": 2.5e9}
        row.update(lg.surface_rows(grid, surface, physics,
                                   {"radiation_accumulated_s": seconds}))
        rows.append(row)
    budget = lg.energy_budget(rows)
    mean = lambda name: lg.area_mean(grid, fluxes[name])  # noqa: E731
    assert budget["toa_net_w_m2"] == pytest.approx(
        mean("swdnt") - mean("swupt") - mean("olr"), rel=1e-12)
    assert budget["outgoing_longwave_w_m2"] == pytest.approx(mean("olr"), rel=1e-12)
    assert budget["surface_net_radiation_w_m2"] == pytest.approx(
        mean("gsw") + mean("glw") - mean("lwupb"), rel=1e-12)
    assert budget["latent_heat_flux_w_m2"] == pytest.approx(75.0, rel=1e-12)
    assert rows[1]["area_mean_toa_net_w_m2"] == pytest.approx(budget["toa_net_w_m2"])
    gates = lg.evaluate(rows)["gates"]
    assert gates["LR-3"]["windows"][0]["precipitation_mm_day"] == pytest.approx(2.9)
    assert gates["LR-9"]["windows"][0]["outgoing_longwave_w_m2"] == pytest.approx(
        mean("olr"), rel=1e-12)


def test_water_cycle_mass_and_reservoir_gates_read_their_rows():
    gates = lg.evaluate(_budget_rows())["gates"]
    assert [w["start_day"] for w in gates["LR-3"]["windows"]] == [0.0, 30.0]
    for window in gates["LR-3"]["windows"]:
        assert window["precipitation_mm_day"] == pytest.approx(2.7, rel=1e-12)
        assert window["precipitable_water_kg_m2"] == pytest.approx(25.0)
        assert window["evaporation_minus_precipitation_mm_day"] == pytest.approx(0.1, rel=1e-9)
        assert window["evaporation_mm_day"] == pytest.approx(2.8, rel=1e-9)
    assert gates["LR-3"]["verdict"] == "pass"
    assert gates["LR-1"]["run_range_pa"] == pytest.approx(10.0, abs=0.01)
    assert gates["LR-1"]["verdict"] == "pass"
    assert gates["LR-2"]["land_minimum_kg_m2"] == 480.0
    assert gates["LR-2"]["land_passed"] is True
    # The land half holds, but the account's identity is unmeasured, so the
    # required gate is not passed on half its evidence.
    assert gates["LR-2"]["verdict"] == "not measured"
    assert gates["LR-2"]["ledger"] == (
        f"not measured: rows {lg.EXCHANGE_ACCOUNT_ROW} and "
        f"{lg.WATER_NET_EVAPORATION_ROW} are absent")
    # The account alone (what the native suite writes today) names only the
    # row that is missing; a land minimum below 50 kg/m2 still fails.
    rows = _budget_rows()
    for r in rows:
        r[lg.EXCHANGE_ACCOUNT_ROW] = -1.0
    lr2 = lg.evaluate(rows)["gates"]["LR-2"]
    assert lr2["verdict"] == "not measured"
    assert lr2["ledger"] == f"not measured: row {lg.WATER_NET_EVAPORATION_ROW} is absent"
    rows[-1]["minimum_land_surface_water_kg_m2"] = 49.0
    assert lg.evaluate(rows)["gates"]["LR-2"]["verdict"] == "fail"
    assert gates["LR-8"]["verdict"] == "not measured"
    assert gates["LR-0"]["verdict"] == "not measured"


def test_lr9_a_seasonal_swing_is_reported_per_window_and_the_year_passes():
    """Absorbed shortwave swinging 16 W/m2 peak to peak over the year on a
    +1 W/m2 annual mean: windows reach +-8 W/m2 off the annual target and
    are reported, not failed; the annual mean passes."""
    rows = _budget_rows(days=365, every_h=24)
    year_s = 365.0 * 86400.0
    for r in rows:
        t = r["time_s"]
        # int of 8 sin(2 pi t / year) dt, added to the absorbed shortwave.
        swing = 8.0 * year_s / (2.0 * math.pi) * (1.0 - math.cos(2.0 * math.pi * t / year_s))
        r["area_mean_acc_sw_down_top_j_m2"] += swing
    gate = lg.evaluate(rows)["gates"]["LR-9"]
    nets = [w["toa_net_w_m2"] for w in gate["windows"]]
    assert max(nets) > 6.0 and min(nets) < -4.0  # outside the annual band
    assert gate["verdict"] == "pass"
    assert gate["annual_toa_net_w_m2"] == pytest.approx(1.0, abs=1e-6)


def test_lr9_is_gated_on_a_full_year_mean_only():
    year = lg.evaluate(_budget_rows(days=366, every_h=24))["gates"]["LR-9"]
    assert year["verdict"] == "pass"
    assert year["annual_toa_net_w_m2"] == pytest.approx(1.0, abs=1e-9)
    assert len(year["windows"]) == 13


def test_rows_another_package_writes_are_judged_when_present():
    rows = _budget_rows(days=10)
    for r in rows:
        day = r["time_s"] / 86400.0
        # account = -int(E-P)_water - fixer - clipping
        r[lg.WATER_NET_EVAPORATION_ROW] = 3.0 * day
        r[lg.WATER_FIXER_ON_WATER_ROW] = -0.02 * day
        r[lg.WATER_CLIPPING_ON_WATER_ROW] = 0.001 * day
        r[lg.EXCHANGE_ACCOUNT_ROW] = -3.0 * day + 0.02 * day - 0.001 * day
        r[lg.WATER_EVAPORATION_24H_ROW] = 14.0
        r[lg.BOUNDARY_SST_DEPARTURE_ROW] = 2.0e-4
        r[lg.BOUNDARY_ICE_MISMATCH_ROW] = 0.0
    gates = lg.evaluate(rows)["gates"]
    assert gates["LR-2"]["ledger_maximum_relative"] < 1.0e-15
    assert gates["LR-2"]["verdict"] == "pass"
    assert gates["LR-2"]["maximum_water_column_evaporation_24h_mm_day"] == 14.0
    rows[-1][lg.EXCHANGE_ACCOUNT_ROW] *= 1.0 + 1.0e-5
    assert lg.evaluate(rows)["gates"]["LR-2"]["verdict"] == "fail"
    assert gates["LR-8"]["verdict"] == "pass"
    rows[-1][lg.BOUNDARY_ICE_MISMATCH_ROW] = 1.0
    assert lg.evaluate(rows)["gates"]["LR-8"]["verdict"] == "fail"


# ------------------------------------------------- rows on disk, LR-0


def test_a_segment_chain_drops_the_repeated_opening_row(tmp_path):
    rows = _budget_rows(days=2)
    first, second = tmp_path / "day1", tmp_path / "day2"
    first.mkdir()
    second.mkdir()
    (first / "diagnostics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows[:5]), encoding="utf-8")
    (second / "diagnostics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows[4:]), encoding="utf-8")
    chained = lg.read_rows([first, second])
    assert [r["step"] for r in chained] == [r["step"] for r in rows]
    assert lg.relative_drift(chained, "global_mean_surface_pressure_pa") == 0.0


def test_restart_gate_compares_bytes(tmp_path):
    def write(path, value):
        arrays = {"theta": np.array([1.0, value])}
        meta = {"self_sha256": "x" if value == 2.0 else "y", "step": 4}
        np.savez(path, __metadata__=np.asarray(json.dumps(meta)), **arrays)

    write(tmp_path / "a.npz", 2.0)
    write(tmp_path / "b.npz", 2.0)
    write(tmp_path / "c.npz", np.nextafter(2.0, 3.0))
    same = lg.restart_comparison(tmp_path / "a.npz", tmp_path / "b.npz")
    assert same["bit_identical"] and same["differing_arrays"] == []
    moved = lg.restart_comparison(tmp_path / "a.npz", tmp_path / "c.npz")
    assert not moved["bit_identical"] and moved["differing_arrays"] == ["theta"]
    gate = lg.evaluate(_budget_rows(days=1), restart=moved)["gates"]["LR-0"]
    assert gate["verdict"] == "fail"


def test_a_run_longer_than_384_h_carries_the_table_in_its_receipt(tmp_path, monkeypatch):
    """The smoke run (40 s) with the long-run threshold lowered under it:
    the receipt carries the table, strict JSON, outside ``status``."""
    from arwen_global import runner
    from arwen_global.config import load_config

    cfg = load_config(SMOKE)
    short = runner.run(cfg, tmp_path / "short")
    assert "longrun_gates" not in short
    monkeypatch.setattr(runner, "LONG_RUN_THRESHOLD_S", 0.0)
    receipt = runner.run(cfg, tmp_path / "long")
    table = receipt["longrun_gates"]
    assert table["schema"] == "gpuwm.arwen-global-longrun-gates/v1"
    assert table["rows"] == 3 and table["last_step"] == 4
    assert set(table["gates"]) == {f"LR-{i}" for i in range(11)}
    assert table["gates"]["LR-10"]["verdict"] == "report"
    assert receipt["status"] == short["status"]
    rows = lg.read_rows([tmp_path / "long"])
    assert lg.level_row_name(850) in rows[0]
    assert "global_mean_moist_total_energy_j_m2" in rows[-1]


# ----------------------------------------------------------------- LR-10


def test_angular_momentum_of_solid_body_rotation_is_two_thirds_of_u0_a_m():
    """u = U0 cos(phi) over columns of equal mass M: the relative part per
    unit area is U0 a M <cos^2> and the planetary part Omega a^2 M <cos^2>,
    with <cos^2> = 2/3 exactly on the Gaussian quadrature (1 - mu^2 is a
    polynomial); the band's zonal means feed the row unchanged."""
    from arwen_global.constants import EARTH_ROTATION_RATE_S

    grid = GaussianGrid.create(21)
    nlev, u0 = 4, 15.0
    shape = (nlev, *grid.shape)
    dp = np.full(shape, 25000.0)
    cos = np.broadcast_to(grid.cos_lat[None, :, None], shape)
    band = {
        "temperature": np.full(shape, 250.0), "u": u0 * cos, "v": np.zeros(shape),
        "qv": np.zeros(shape), "qi": np.zeros(shape), "qs": np.zeros(shape),
        "qg": np.zeros(shape), "dp": dp, "ps": np.full(grid.shape, 1.0e5),
        "p_full": np.cumsum(dp, axis=0) - 12500.0,
    }
    zonal = lg.atmosphere_band_zonal_means(np, band, np.zeros(grid.shape))
    rows = lg.atmosphere_rows(grid, zonal, 1.0e5, 0.0)
    mass = 1.0e5 / GRAVITY_M_S2
    a = grid.radius_m
    relative = u0 * a * mass * 2.0 / 3.0
    planetary = EARTH_ROTATION_RATE_S * a * a * mass * 2.0 / 3.0
    assert rows[lg.RELATIVE_ANGULAR_MOMENTUM_ROW] == pytest.approx(relative, rel=1e-12)
    assert rows[lg.AXIAL_ANGULAR_MOMENTUM_ROW] == pytest.approx(
        relative + planetary, rel=1e-12)
    # A row whose rotation rate is named reads it.
    still = lg.atmosphere_rows(grid, zonal, 1.0e5, 0.0, rotation_rate_s=0.0)
    assert still[lg.AXIAL_ANGULAR_MOMENTUM_ROW] == pytest.approx(relative, rel=1e-12)


def test_lr10_reads_daily_means_and_a_planted_drift():
    """Six-hourly rows over 12 days: the axial row drifts by a planted 1e-9
    of itself per day and the relative row carries a diurnal swing that
    each day's mean cancels; LR-10 reads the days, the drift and the
    window changes back, as planet totals (times 4 pi a^2)."""
    from arwen_global.constants import EARTH_RADIUS_M

    start = dt.datetime(2026, 11, 10, 18, tzinfo=dt.timezone.utc)
    area = 4.0 * math.pi * EARTH_RADIUS_M ** 2
    axial0, relative0 = 1.0e33 / area, 1.5e26 / area
    rows = []
    for i in range(12 * 4 + 1):
        t = i * 21600.0
        day = t / 86400.0
        rows.append({
            "step": i, "time_s": t,
            lg.AXIAL_ANGULAR_MOMENTUM_ROW: axial0 * (1.0 + 1.0e-9 * day),
            lg.RELATIVE_ANGULAR_MOMENTUM_ROW: relative0 * (
                1.0 + 0.1 * math.sin(2.0 * math.pi * (i % 4) / 4.0)),
        })
    gate = lg.evaluate(rows, start_utc=start)["gates"]["LR-10"]
    assert gate["verdict"] == "report"
    days = gate["days"]
    assert days[0]["day"] == "2026-11-10" and days[0]["samples"] == 1
    assert days[0]["axial_kg_m2_s"] == pytest.approx(1.0e33, rel=1e-12)
    assert days[0]["relative_kg_m2_s"] == pytest.approx(1.5e26, rel=1e-12)
    assert days[1]["mean_day"] == pytest.approx(0.625, rel=1e-12)
    assert [d["samples"] for d in days[1:-1]] == [4] * 11
    for d in days[1:-1]:
        assert d["relative_kg_m2_s"] == pytest.approx(1.5e26, rel=1e-12)
    assert gate["axial_drift_kg_m2_s_per_day"] == pytest.approx(1.0e24, rel=1e-6)
    assert gate["axial_relative_drift_per_day"] == pytest.approx(1.0e-9, rel=1e-5)
    assert gate["relative_drift_kg_m2_s_per_day"] == pytest.approx(0.0, abs=1.0e17)
    window = gate["windows"][0]
    assert window["axial_change_kg_m2_s"] == pytest.approx(12.0e24, rel=1e-6)
    assert window["axial_drift_kg_m2_s_per_day"] == pytest.approx(1.0e24, rel=1e-6)
    # Without the row the gate says so; without a start the days are counted.
    assert lg.evaluate([{"step": 0, "time_s": 0.0}, {"step": 1, "time_s": 1.0}])[
        "gates"]["LR-10"]["verdict"] == "not measured"
    assert lg.evaluate(rows)["gates"]["LR-10"]["days"][1]["day"] == 1
