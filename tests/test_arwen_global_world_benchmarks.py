"""The two CPU benchmarks other worlds are judged against measure what they
say, on cases with a known answer.

THE BREAKAGE THIS PREVENTS.  `tools/arwen_global_world_benchmarks.py` gives
two numbers the other-worlds track rests on: the wind a resting atmosphere
picks up over Earth's terrain (the floor a no-mountain twin's difference
map sits on) and whether a steady state stays steady at 0.5 to 8 times
Earth's rotation (which sets the day lengths a world may ask for).  A rest
state that was not hydrostatic, a regridder that slid the terrain, a ladder
that built the state at one rate and integrated it at another, or a builder
that changed the published Jablonowski-Williamson case under every shipped
baroclinic config would each still print plausible small numbers.  So:

*   the resting profile matches a numerical hydrostatic integration, and is
    continuous at its tropopause;
*   the regridder is exact on a field bilinear interpolation must reproduce,
    and refuses a field that does not close the globe;
*   with no terrain the resting atmosphere stays at rest to float64
    rounding through the real core, and a mountain makes it move;
*   the Jablonowski-Williamson builder at its defaults is the published case
    bit for bit, provenance included, and the scaled-jet rung differs from
    it by the jet's own squared term only;
*   the ladder's model integrates at the rung's rate, and a fast rung runs.

CPU only: numpy, the tiny T10 configuration derived from the shipped T85
case.  No device, no network.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from arwen_global.constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
)
from arwen_global.testcases import (
    JW_U0,
    REST_LAPSE_K_M,
    REST_SEA_LEVEL_PRESSURE_PA,
    REST_SEA_LEVEL_TEMPERATURE_K,
    REST_TROPOPAUSE_K,
    baroclinic_wave_initial_state,
    jablonowski_williamson_fields,
    rest_atmosphere_surface_pressure,
    rest_atmosphere_temperature,
    terrain_on_model_grid,
)

TREE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_world_benchmarks",
        TREE / "tools" / "arwen_global_world_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_cfg(tool, tmp_path_factory, config_root):
    from arwen_global.config import load_config

    text = (Path(config_root) / tool.DEFAULT_CONFIG).read_text(encoding="utf-8")
    path = tmp_path_factory.mktemp("worlds-bench") / "tiny.toml"
    path.write_text(tool.derived_config_text(
        text, backend="numpy", precision="float64", truncation=10),
        encoding="utf-8")
    return load_config(path)


def _quiet(*_args, **_kwargs):
    return None


# ------------------------------------------------------------ the profile

def test_rest_profile_is_the_hydrostatic_integral_of_its_own_temperature():
    """ln p(z) by RK4 through dz = -(R T(p) / g) d ln p agrees with the
    closed form to 1e-8 at the heights the plan names and above the
    tropopause."""
    heights = np.array([0.0, 500.0, 2000.0, 4500.0, 8848.0, 12000.0, 20000.0])
    closed = rest_atmosphere_surface_pressure(heights)

    def slope(ln_p):
        return -GRAVITY_M_S2 / (
            DRY_AIR_GAS_CONSTANT * float(rest_atmosphere_temperature(math.exp(ln_p))))

    ln_p = math.log(REST_SEA_LEVEL_PRESSURE_PA)
    z = 0.0
    dz = 5.0
    numeric = []
    for target in heights:
        while z < target - 1.0e-9:
            h = min(dz, target - z)
            k1 = slope(ln_p)
            k2 = slope(ln_p + 0.5 * h * k1)
            k3 = slope(ln_p + 0.5 * h * k2)
            k4 = slope(ln_p + h * k3)
            ln_p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            z += h
        numeric.append(math.exp(ln_p))
    # 1e-8, not tighter: RK4 steps across the tropopause kink lose an order.
    np.testing.assert_allclose(closed, numeric, rtol=1.0e-8)
    assert closed[0] == REST_SEA_LEVEL_PRESSURE_PA


def test_rest_profile_is_continuous_at_its_tropopause():
    z_trop = (REST_SEA_LEVEL_TEMPERATURE_K - REST_TROPOPAUSE_K) / REST_LAPSE_K_M
    p_trop = float(rest_atmosphere_surface_pressure(z_trop))
    assert float(rest_atmosphere_temperature(p_trop)) == pytest.approx(
        REST_TROPOPAUSE_K, abs=1.0e-9)
    below = float(rest_atmosphere_surface_pressure(z_trop - 1.0e-6))
    above = float(rest_atmosphere_surface_pressure(z_trop + 1.0e-6))
    assert abs(below - above) < 1.0e-4
    assert float(rest_atmosphere_temperature(100.0)) == REST_TROPOPAUSE_K


# ------------------------------------------------------------ the regrid

class _Grid:
    def __init__(self, lat, lon):
        self.latitude_deg = np.asarray(lat, dtype=np.float64)
        self.longitude_deg = np.asarray(lon, dtype=np.float64)


def test_terrain_regrid_is_exact_where_bilinear_must_be():
    lat = np.linspace(90.0, -90.0, 37)          # descending, as ERA5 stores it
    lon = np.arange(0.0, 360.0, 5.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    field = 1000.0 + 7.0 * la + 0.0 * lo        # linear in latitude
    target = _Grid([-88.6, -10.3, 0.0, 33.3, 89.1],
                   [0.0, 2.5, 181.2, 357.5, 359.9])
    out = terrain_on_model_grid(lat, lon, field, target)
    expect = 1000.0 + 7.0 * target.latitude_deg[:, None] + 0.0 * target.longitude_deg[None]
    np.testing.assert_allclose(out, expect, rtol=0.0, atol=1.0e-9)
    # Across the dateline: halfway between 355 and 0 is the mean of the two.
    ring = np.zeros_like(field)
    ring[:, -1] = 100.0
    out = terrain_on_model_grid(lat, lon, ring, _Grid([0.0], [357.5]))
    assert out[0, 0] == pytest.approx(50.0, abs=1.0e-12)


def test_terrain_regrid_refuses_a_field_that_does_not_close_the_globe():
    lat = np.linspace(-90.0, 90.0, 19)
    lon = np.arange(0.0, 180.0, 10.0)
    with pytest.raises(ValueError, match="close the globe"):
        terrain_on_model_grid(lat, lon, np.zeros((19, 18)), _Grid([0.0], [0.0]))


# ------------------------------------------------- the rest benchmark

def test_flat_rest_state_stays_at_rest_to_rounding(tool, tiny_cfg):
    model, bundle = tool.rest_case(tiny_cfg, None, 0.0)
    run = tool.integrate(model, bundle, dt_s=float(tiny_cfg.dt_s), steps=24,
                         sample_every=12, progress=_quiet)
    assert run["completed"], run["refusal"]
    last = run["rows"][-1]
    assert last["max_wind_m_s"] < 1.0e-9
    assert last["max_ps_change_pa"] < 1.0e-7


def test_a_mountain_makes_the_resting_atmosphere_move(tool, tiny_cfg):
    """A 2,000 m Gaussian mountain: the start is level and still, the core's
    pressure-gradient error over the slope makes a wind that is larger than
    rounding and far from a blow-up."""

    def mountain(grid):
        lat = np.deg2rad(np.asarray(grid.latitude_deg))[:, None]
        lon = np.deg2rad(np.asarray(grid.longitude_deg))[None, :]
        cos_r = np.sin(lat) * math.sin(math.radians(30.0)) + np.cos(lat) * math.cos(
            math.radians(30.0)) * np.cos(lon - math.pi)
        r = np.arccos(np.clip(cos_r, -1.0, 1.0))
        return 2000.0 * np.exp(-(r / 0.35) ** 2)

    model, bundle = tool.rest_case(tiny_cfg, mountain, 1.0)
    assert float(np.max(model.surface_geopotential)) / GRAVITY_M_S2 > 1000.0
    run = tool.integrate(model, bundle, dt_s=float(tiny_cfg.dt_s), steps=24,
                         sample_every=12, progress=_quiet)
    assert run["completed"], run["refusal"]
    assert run["rows"][0]["max_wind_m_s"] == 0.0
    assert 1.0e-6 < run["rows"][-1]["max_wind_m_s"] < 20.0
    days = tool.per_day(run["rows"], "max_wind_m_s", float(tiny_cfg.dt_s))
    assert [row["day"] for row in days] == [1]


# ----------------------------------------------------- the ladder

def test_published_case_is_unchanged_by_the_ladder_keywords(tiny_cfg):
    from arwen_global.runner import build_transform

    transform = build_transform(tiny_cfg)
    plain = baroclinic_wave_initial_state(tiny_cfg, transform)
    spelled = baroclinic_wave_initial_state(
        tiny_cfg, transform, rotation_rate_s=EARTH_ROTATION_RATE_S, u0_m_s=JW_U0)
    assert plain[2] == spelled[2]
    assert set(plain[2]) == {"mode", "case", "perturbation_m_s", "surface_pressure_pa"}
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure"):
        a = np.asarray(getattr(plain[0].atmosphere, name))
        b = np.asarray(getattr(spelled[0].atmosphere, name))
        assert a.tobytes() == b.tobytes(), name
    assert np.asarray(plain[1]).tobytes() == np.asarray(spelled[1]).tobytes()


@pytest.mark.parametrize("factor", [0.5, 2.0, 8.0])
def test_scaled_jet_rung_keeps_earths_temperature_but_the_jet_term(factor):
    """At rate N Omega and jet U0 / N the rotation term of the temperature
    is Earth's exactly; what differs is the jet's own squared term."""
    eta = np.linspace(0.05, 1.0, 12)[:, None]
    lat = np.deg2rad(np.linspace(-89.0, 89.0, 31))[None, :]
    radius = 6371220.0
    _, t_earth, _ = jablonowski_williamson_fields(eta, lat, radius_m=radius)
    _, t_rung, _ = jablonowski_williamson_fields(
        eta, lat, radius_m=radius, rotation_rate_s=factor * EARTH_ROTATION_RATE_S,
        u0_m_s=JW_U0 / factor)
    eta_v = (eta - 0.252) * (math.pi / 2.0)
    first = -2.0 * np.sin(lat) ** 6 * (np.cos(lat) ** 2 + 1.0 / 3.0) + 10.0 / 63.0
    shape = 0.75 * eta * math.pi / DRY_AIR_GAS_CONSTANT * np.sin(eta_v) * np.sqrt(
        np.cos(eta_v)) * first * 2.0 * np.cos(eta_v) ** 1.5
    expected = shape * ((JW_U0 / factor) ** 2 - JW_U0 ** 2)
    np.testing.assert_allclose(t_rung - t_earth, expected, rtol=1.0e-9, atol=1.0e-9)


def test_ladder_model_integrates_at_the_rungs_rate(tool, tiny_cfg):
    earth, _, rate1, jet1 = tool.ladder_case(tiny_cfg, 1.0)
    fast, bundle, rate4, jet4 = tool.ladder_case(tiny_cfg, 4.0)
    assert rate1 == EARTH_ROTATION_RATE_S and jet1 == JW_U0
    assert rate4 == 4.0 * EARTH_ROTATION_RATE_S and jet4 == JW_U0 / 4.0
    assert fast.rotation_rate_s == rate4
    assert np.asarray(fast.coriolis).tobytes() == np.asarray(4.0 * earth.coriolis).tobytes()
    run = tool.integrate(fast, bundle, dt_s=float(tiny_cfg.dt_s), steps=12,
                         sample_every=6, progress=_quiet)
    assert run["completed"], run["refusal"]
    assert run["rows"][-1]["finite"]


def test_steadiness_rule_reads_against_the_earth_rung(tool):
    def run(factor, drift, completed=True):
        return {"rotation_factor": factor, "completed": completed,
                "rows": [{"rms_ps_change_pa": 0.0}, {"rms_ps_change_pa": drift}]}

    runs = [run(1.0, 3.0), run(2.0, 29.0), run(4.0, 31.0), run(8.0, 1.0, False),
            run(0.5, 9.0)]
    tool.steadiness(runs)
    assert [r["steady"] for r in runs] == [True, True, False, False, True]
    assert runs[0]["steady_limit_pa"] == 30.0
    quiet = [run(1.0, 0.1), run(2.0, 9.9)]
    tool.steadiness(quiet)
    assert quiet[1]["steady_limit_pa"] == tool.STEADY_FLOOR_PA and quiet[1]["steady"]
    tool.steadiness([run(2.0, 1.0)])  # no Earth rung: no verdict, no error


def test_derived_config_refuses_a_key_it_cannot_place(tool):
    with pytest.raises(ValueError, match="truncation"):
        tool.derived_config_text('backend = "cupy"\n', truncation=10)
    assert tool.derived_config_text(
        '[a]\nbackend = "cupy"\n[grid]\ntruncation = 85\n',
        backend="numpy", truncation=10) == '[a]\nbackend = "numpy"\n[grid]\ntruncation = 10\n'


def test_diffusion_choices_edit_only_the_diffusion_table(tool, config_root, tmp_path):
    from arwen_global.config import load_config

    text = (Path(config_root) / tool.DEFAULT_CONFIG).read_text(encoding="utf-8")
    assert tool.diffusion_config_text(text, "shipped") == text
    off = tool.diffusion_config_text(text, "off")
    quiet = tool.diffusion_config_text(text, "no-pressure")
    # Only the [diffusion] table moves: [semi_implicit] and [insitu] keep
    # their own 'enabled' lines.
    assert off.count("enabled = false") == text.count("enabled = false") + 1
    assert "pressure_strength = 0.0" in quiet
    with pytest.raises(ValueError, match="diffusion must be"):
        tool.diffusion_config_text(text, "less")
    for name, edited in (("off", off), ("quiet", quiet)):
        (tmp_path / f"{name}.toml").write_text(edited, encoding="utf-8")
    assert not load_config(tmp_path / "off.toml").diffusion_enabled
    cfg = load_config(tmp_path / "quiet.toml")
    assert cfg.diffusion_enabled and cfg.pressure_diffusion_strength == 0.0


def test_eddy_spectrum_finds_the_wave_it_was_given(tool):
    class Grid:
        quadrature_weights = np.full(8, 0.25)

    lon = np.arange(64) * (2.0 * math.pi / 64)
    for wavenumber, top in ((5, 0.0), (20, 1.0)):
        field = np.broadcast_to(np.sin(wavenumber * lon), (3, 8, 64))
        out = tool.eddy_spectrum(field, Grid(), truncation=21)
        assert out["eddy_peak_wavenumber"] == wavenumber
        assert out["eddy_top_third_share"] == pytest.approx(top, abs=1.0e-12)
    assert tool.eddy_spectrum(np.zeros((1, 8, 64)), Grid(), truncation=21) == {
        "eddy_peak_wavenumber": 0, "eddy_top_third_share": 0.0}
