"""The invented-world frame (plan package 2.6): Earth's zonal-mean air
raised onto an invented world's terrain by the pressure mover, land skin
and soil from that air, typed soil wetness, no snow and no land ice, the
deep soil from Earth's zonal mean at sea level.

The gates: the mover against the closed form of a constant-lapse
atmosphere and bit for bit at zero height; each rule of the frame on a
small invented planet (one rectangular continent with a mountain); the
step-0 surface pressure of the cold state built from that frame equals the
mover's output; and a 24-hour full-physics run on that planet finishes with
no refusal.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from arwen_global.constants import GRAVITY_M_S2
from arwen_global.world import parse_world
from arwen_global.world.frame import (
    DRY_AIR_GAS_CONSTANT, LAPSE_K_PER_M, NOAH_ZBOT_M, SEA_WATER_FREEZING_K,
    WorldMap, aquaplanet_sea_temperature_k, area_mean, deep_soil_plane,
    earth_zonal_mean_air, invented_world_frame, move_surface_pressure,
    sea_level_deep_soil_by_latitude, soil_water_from_wetness,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_arwen_global_analysis_initial import (  # noqa: E402
    _analysis_cfg, _synthetic_frame,
)

R_OVER_G = DRY_AIR_GAS_CONSTANT / GRAVITY_M_S2

INVENTED = {
    "label": "test-continent",
    "sea": {"temperature": "zonal-mean", "ice": "none"},
    "surface": {"folder": "test-continent.world", "soil_wetness": 0.5},
}


def _world(**changes):
    table = json.loads(json.dumps(INVENTED))
    for dotted, value in changes.items():
        part, key = dotted.split(".")
        table.setdefault(part, {})[key] = value
    return parse_world(table)


def _continent_map(nlat=37, nlon=72, mountain_m=2000.0):
    """The test suite's invented planet: one rectangular continent from 20 S
    to 50 N and 0 to 120 E on a 5-degree pole-to-pole grid, a smooth
    mountain of ``mountain_m`` near its middle, sea everywhere else."""
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    land = ((lat2 >= -20.0) & (lat2 <= 50.0) & (lon2 >= 0.0) & (lon2 <= 120.0))
    mountain = mountain_m * np.exp(
        -(((lat2 - 20.0) / 12.0) ** 2) - (((lon2 - 60.0) / 18.0) ** 2))
    terrain = np.where(land, 50.0 + mountain, 0.0)
    return WorldMap(lat, lon, terrain, land.astype(np.float64))


def _constant_lapse_levels(p0=100000.0, t0=290.0, lapse=0.0065):
    levels = np.asarray([5000.0, 10000.0, 20000.0, 30000.0, 50000.0,
                         70000.0, 85000.0, 92500.0, 100000.0])
    temperature = t0 * (levels / p0) ** (R_OVER_G * lapse)
    return levels, temperature


# ---------------------------------------------------------------------------
# The pressure mover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("height_m", [500.0, 2000.0, 4500.0])
def test_the_mover_matches_the_closed_form_of_a_constant_lapse_atmosphere(height_m):
    for lapse in (0.0065, 0.004, 0.009):
        levels, temperature = _constant_lapse_levels(lapse=lapse)
        closed = 100000.0 * (1.0 - lapse * height_m / 290.0) ** (1.0 / (R_OVER_G * lapse))
        raised = move_surface_pressure(levels, temperature[:, None], 100000.0, 0.0,
                                       np.asarray([height_m]))
        assert abs(float(raised[0]) - closed) < 1e-6
        # And down again, from the raised surface to sea level.
        lowered = move_surface_pressure(levels, temperature[:, None], closed, height_m,
                                        np.asarray([0.0]))
        assert abs(float(lowered[0]) - 100000.0) < 1e-6


def test_the_mover_returns_its_input_bit_for_bit_where_the_height_is_unchanged():
    levels, temperature = _constant_lapse_levels()
    p_from = np.asarray([101325.0 + 1e-7, 99000.123456789, 70000.0])
    z = np.asarray([0.0, 312.5, 3000.0])
    moved = move_surface_pressure(levels, temperature[:, None], p_from, z, z)
    assert np.array_equal(moved, p_from)


def test_the_mover_raises_and_lowers_back_through_an_isothermal_layer():
    levels = np.asarray([10000.0, 50000.0, 85000.0, 100000.0])
    temperature = np.asarray([[220.0], [250.0], [270.0], [270.0]])  # isothermal at the bottom
    up = move_surface_pressure(levels, temperature, 100000.0, 0.0, np.asarray([800.0]))
    assert abs(float(up[0]) - 100000.0 * np.exp(-800.0 / (R_OVER_G * 270.0))) < 1e-6
    back = move_surface_pressure(levels, temperature, up, 800.0, np.asarray([0.0]))
    assert abs(float(back[0]) - 100000.0) < 1e-6


def test_the_mover_refuses_a_surface_above_the_analysis_top():
    levels, temperature = _constant_lapse_levels()
    with pytest.raises(ValueError, match="above the analysis top level"):
        move_surface_pressure(levels, temperature[:, None], 100000.0, 0.0,
                              np.asarray([40000.0]))


# ---------------------------------------------------------------------------
# Earth's zonal-mean air
# ---------------------------------------------------------------------------


def test_zonal_mean_air_of_a_zonal_earth_with_no_terrain_is_that_earth():
    frame = _synthetic_frame(ridge_sigma_deg=1e-3)
    frame.fields["terrain_height"].values = np.zeros_like(frame.fields["terrain_height"].values)
    frame.fields["surface_pressure"].values = np.full_like(
        frame.fields["surface_pressure"].values, 101000.0)
    column = earth_zonal_mean_air(frame, frame.vertical_values)
    order = np.argsort(frame.latitude)
    t = np.asarray(frame.fields["air_temperature"].values)[:, order, 0]
    assert np.allclose(column.temperature_k, t, rtol=0, atol=1e-10)
    assert np.allclose(column.sea_level_pressure_pa, 101000.0, rtol=0, atol=1e-9)
    assert column.below_ground_filled == 0


def test_a_level_below_the_ground_everywhere_continues_at_six_and_a_half_k_per_km():
    frame = _synthetic_frame()
    # Put one row's whole ring on a 3,000 m plateau at 70 kPa.
    row = 5
    frame.fields["surface_pressure"].values[row, :] = 70000.0
    frame.fields["terrain_height"].values[row, :] = 3000.0
    column = earth_zonal_mean_air(frame, frame.vertical_values)
    levels = np.asarray(frame.vertical_values)
    ascending_row = int(np.where(np.argsort(frame.latitude) == row)[0][0])
    t = column.temperature_k[:, ascending_row]
    q = column.specific_humidity[:, ascending_row]
    exponent = R_OVER_G * LAPSE_K_PER_M
    # 85 and 100 kPa are below that row's ground: filled from 70 kPa.
    assert t[4] == pytest.approx(t[3] * (85000.0 / 70000.0) ** exponent, rel=1e-14)
    assert t[5] == pytest.approx(t[4] * (100000.0 / 85000.0) ** exponent, rel=1e-14)
    assert q[4] == q[3] and q[5] == q[3]
    assert column.below_ground_filled >= 2


# ---------------------------------------------------------------------------
# The frame's rules on the invented planet
# ---------------------------------------------------------------------------


def test_only_an_invented_world_gets_an_invented_frame():
    with pytest.raises(ValueError, match="only for a world with its own surface"):
        invented_world_frame(parse_world({"label": "flat", "terrain": {"scale": 0.0}}),
                             _synthetic_frame(), _continent_map())
    with pytest.raises(ValueError, match="only for a world with its own surface"):
        invented_world_frame(parse_world(None), _synthetic_frame(), _continent_map())


def test_conserve_keeps_earths_amount_of_air_and_fill_keeps_sea_level_pressure():
    earth = _synthetic_frame()
    world_map = _continent_map()
    earth_mean = area_mean(earth.fields["surface_pressure"].values[::-1], earth.latitude[::-1])

    conserve = invented_world_frame(_world(), earth, world_map)
    record = conserve.world_frame["air_mass"]
    assert record["rule"] == "conserve"
    assert area_mean(conserve.fields["surface_pressure"].values, world_map.latitude_deg) == (
        pytest.approx(earth_mean, rel=1e-12))
    # The frame carries the mover's output and nothing else touched it.
    assert conserve.fields["surface_pressure"].values is conserve.mover_surface_pressure_pa

    fill = invented_world_frame(_world(**{"terrain.air_mass": "fill"}), earth, world_map)
    assert fill.world_frame["air_mass"]["factor"] == 1.0
    column = earth_zonal_mean_air(earth, earth.vertical_values)
    p0 = column.at(world_map.latitude_deg)[4]
    sea = world_map.terrain_m == 0.0
    ps = fill.fields["surface_pressure"].values
    # Over the sea the mover returns the row's sea-level pressure bit for bit.
    assert np.array_equal(ps[sea], np.broadcast_to(p0[:, None], ps.shape)[sea])
    # Over the mountain the air is thinner by the column's own hydrostatics.
    peak = np.unravel_index(np.argmax(world_map.terrain_m), ps.shape)
    assert ps[peak] < 0.8 * p0[peak[0]]
    assert np.allclose(conserve.fields["surface_pressure"].values,
                       ps * record["factor"], rtol=1e-15, atol=0)


def test_the_air_is_earths_zonal_mean_on_every_column():
    frame = invented_world_frame(_world(), _synthetic_frame(), _continent_map())
    for name in ("air_temperature", "specific_humidity", "eastward_wind", "northward_wind"):
        values = frame.fields[name].values
        assert np.all(values == values[..., :1])


def test_land_skin_and_soil_come_from_the_air_and_the_deep_soil():
    earth = _synthetic_frame()
    world_map = _continent_map()
    frame = invented_world_frame(_world(), earth, world_map)
    land = world_map.land_fraction >= 0.5
    levels = np.asarray(earth.vertical_values)
    ps = frame.fields["surface_pressure"].values
    temperature = frame.fields["air_temperature"].values
    skin = frame.fields["skin_temperature"].values
    # The skin of every land column is its air interpolated to its surface
    # pressure at the layer's own lapse (here every surface lies between
    # 85 and 100 kPa or below the bottom level).
    for j, i in zip(*np.nonzero(land)):
        p = ps[j, i]
        k = int(np.searchsorted(levels, p))
        if k >= levels.size:
            expected = temperature[-1, j, i] * (p / levels[-1]) ** (R_OVER_G * LAPSE_K_PER_M)
        else:
            c = np.log(temperature[k - 1, j, i] / temperature[k, j, i]) / np.log(
                levels[k - 1] / levels[k])
            expected = temperature[k, j, i] * (p / levels[k]) ** c
        assert skin[j, i] == pytest.approx(expected, rel=1e-13)
    # The soil runs from the skin toward the deep soil at 8 m.
    lat_p, profile = sea_level_deep_soil_by_latitude(
        earth.latitude, earth.fields["soil_temperature"].values[-1],
        earth.fields["land_fraction"].values,
        terrain_m=earth.fields["terrain_height"].values)
    deep = deep_soil_plane(lat_p, profile, world_map.latitude_deg, world_map.terrain_m)
    soil = frame.fields["soil_temperature"].values
    for layer, mid in enumerate((0.05, 0.25, 0.7, 1.5)):
        expected = skin + (deep - skin) * (mid / NOAH_ZBOT_M)
        assert np.allclose(soil[layer][land], expected[land], rtol=0, atol=1e-10)
        assert np.all(np.isnan(soil[layer][~land]))
    assert "bottom soil layer" in frame.world_frame["deep_soil"]["source"]


def test_earths_statics_soiltemp_sets_the_deep_soil_when_handed_in():
    """With Earth's real-statics SOILTEMP (already at sea level) the soil's
    bottom anchor is its zonal mean over land, less the world's lapse: on a
    uniform 285 K climatology the 8 m anchor is 285 K - 0.0065 K/m * terrain."""
    earth = _synthetic_frame()
    world_map = _continent_map()
    lat = np.linspace(90.0, -90.0, 19)
    soiltemp = np.full((19, 36), 285.0)
    landmask = np.ones((19, 36))
    frame = invented_world_frame(_world(), earth, world_map,
                                 earth_deep_soil=(lat, soiltemp, landmask))
    assert "SOILTEMP" in frame.world_frame["deep_soil"]["source"]
    land = world_map.land_fraction >= 0.5
    skin = frame.fields["skin_temperature"].values
    deep = 285.0 - LAPSE_K_PER_M * world_map.terrain_m
    bottom = frame.fields["soil_temperature"].values[3]
    assert np.allclose(bottom[land], (skin + (deep - skin) * (1.5 / NOAH_ZBOT_M))[land],
                       rtol=0, atol=1e-10)


def test_a_level_ladder_not_in_pa_is_refused():
    earth = _synthetic_frame()
    with pytest.raises(ValueError, match="not a pressure in Pa"):
        invented_world_frame(_world(), earth, _continent_map(),
                             levels_pa=np.asarray(earth.vertical_values) / 100.0)


def test_deep_soil_is_earths_zonal_mean_at_sea_level_less_the_worlds_terrain():
    lat = np.asarray([90.0, 45.0, 0.0, -45.0, -90.0])
    soiltemp = np.asarray([[250.0, 250.0, 250.0, 250.0],
                           [280.0, 284.0, 999.0, 999.0],
                           [300.0, 300.0, 300.0, 300.0],
                           [999.0, 999.0, 999.0, 999.0],   # no land in this row
                           [240.0, 240.0, 240.0, 240.0]])
    land = np.asarray([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0],
                       [1, 1, 1, 1]], dtype=float)
    lat_p, profile = sea_level_deep_soil_by_latitude(lat, soiltemp, land)
    assert list(lat_p) == [-90.0, -45.0, 0.0, 45.0, 90.0]
    assert list(profile) == [240.0, 270.0, 300.0, 282.0, 250.0]
    terrain = np.full((5, 2), 1000.0)
    plane = deep_soil_plane(lat_p, profile, lat, terrain)
    assert np.allclose(plane[:, 0], [250.0 - 6.5, 282.0 - 6.5, 293.5, 263.5, 233.5])
    assert np.array_equal(deep_soil_plane(lat_p, profile, lat, terrain, at_sea_level=True)[:, 1],
                          [250.0, 282.0, 300.0, 270.0, 240.0])


def test_typed_soil_wetness_is_a_fraction_of_saturation_of_loam():
    water, record = soil_water_from_wetness(0.5)
    assert water == pytest.approx(0.5 * 0.439) and record["maxsmc"] == 0.439
    assert soil_water_from_wetness(0.0)[0] == 0.066   # loam's dry limit
    assert soil_water_from_wetness(1.0)[0] == 0.439
    frame = invented_world_frame(_world(**{"surface.soil_wetness": 0.8}),
                                 _synthetic_frame(), _continent_map())
    moisture = frame.fields["volumetric_soil_moisture"].values
    land = _continent_map().land_fraction >= 0.5
    assert np.all(moisture[:, land] == pytest.approx(0.8 * 0.439))
    assert np.all(np.isnan(moisture[:, ~land]))


def test_no_snow_no_land_ice_and_the_sea_follows_the_world():
    earth = _synthetic_frame()
    world_map = _continent_map()
    land = world_map.land_fraction >= 0.5
    frame = invented_world_frame(_world(), earth, world_map)
    for name in ("snow_water_equivalent", "snow_depth", "sea_ice_fraction", "sea_ice_thickness"):
        assert np.all(frame.fields[name].values == 0.0)
    skin = frame.fields["skin_temperature"].values
    assert skin[~land].min() >= SEA_WATER_FREEZING_K
    # Zonal: every sea column of a row carries the same water.
    for j in range(skin.shape[0]):
        row = skin[j][~land[j]]
        assert np.all(row == row[0])

    zonal_ice = invented_world_frame(_world(**{"sea.ice": "zonal"}), earth, world_map)
    ice = zonal_ice.fields["sea_ice_fraction"].values
    assert ice.max() > 0.5 and np.all(ice[land] == 0.0)
    assert np.all(zonal_ice.fields["snow_water_equivalent"].values == 0.0)

    aqua = invented_world_frame(_world(**{"sea.temperature": "aqua-control"}), earth, world_map)
    expected = aquaplanet_sea_temperature_k("aqua-control", world_map.latitude_deg)
    assert np.allclose(aqua.fields["skin_temperature"].values[~land],
                       np.broadcast_to(expected[:, None], skin.shape)[~land])


@pytest.mark.parametrize("profile", ["aqua-control", "aqua-peaked", "aqua-flat", "aqua-qobs"])
def test_the_published_aquaplanet_profiles_are_27_c_at_the_equator_and_0_c_poleward_of_60(profile):
    values = aquaplanet_sea_temperature_k(profile, np.asarray([0.0, 60.0, -60.0, 75.0, -90.0]))
    assert values[0] == pytest.approx(273.15 + 27.0)
    assert np.allclose(values[1:], 273.15)


# ---------------------------------------------------------------------------
# Step 0 and a day of full physics
# ---------------------------------------------------------------------------


def test_the_cold_state_surface_pressure_is_the_movers_output(tmp_path):
    """The cold start reads the frame as an analysis: its step-0 surface
    pressure is the mover's output, moved only by the cold start's own
    hypsometric step onto the spectrally resolved terrain, and nothing
    else."""
    from arwen_global.analysis_initial import (
        _global_regridder, analysis_initial_state, surface_virtual_temperature,
    )
    from arwen_global.lnps_reference import LNPS_REFERENCE
    from arwen_global.runner import build_transform

    cfg = _analysis_cfg(tmp_path)
    transform = build_transform(cfg)
    frame = invented_world_frame(_world(), _synthetic_frame(), _continent_map())
    state, phi_model, provenance = analysis_initial_state(cfg, transform, frame=frame)

    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    ps_src = regrid(frame.mover_surface_pressure_pa)
    phi_src = GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    host = transform.backend.to_numpy
    levels = np.asarray(frame.vertical_values, dtype=np.float64)
    tv = surface_virtual_temperature(
        regrid(frame.fields["air_temperature"].values),
        np.clip(regrid(frame.fields["specific_humidity"].values), 0.0, None),
        np.log(levels), ps_src)
    expected = ps_src * np.exp((phi_src - host(phi_model)) / (DRY_AIR_GAS_CONSTANT * tv))
    truncated = np.exp(host(transform.inverse(transform.forward(
        np.log(expected) - LNPS_REFERENCE))) + LNPS_REFERENCE)
    ps0 = np.exp(host(transform.inverse(state.atmosphere.log_surface_pressure)))
    assert np.allclose(ps0, truncated, rtol=1e-12, atol=0)
    # The land the cold start built is the world's continent, snow-free.
    land = host(state.surface.land_fraction) > 0.5
    assert land.any() and (~land).any()
    assert np.all(host(state.physics_state.arrays["noah_snow"]) == 0.0)
    assert np.all(host(state.surface.sea_ice_fraction) == 0.0)
    assert provenance["valid_time"] == str(frame.valid_time)


def test_a_day_of_full_physics_on_the_invented_planet_finishes_with_no_refusal(
        tmp_path, monkeypatch):
    """24 model hours of the reference suite (radiation, surface fluxes,
    turbulence, convection, microphysics) on the small invented planet,
    through the runner, from the invented-world frame.  (The native suite's
    day on the same planet runs on a card: tools-free, see the package
    report.)"""
    from arwen_global import analysis_initial
    from arwen_global.config import load_config
    from arwen_global.runner import run
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.world import folder as world_folder
    from arwen_global.world import surface as world_surface

    real = analysis_initial.analysis_initial_state
    world = _world()

    # The world folder a user would make: the test continent as a map (sea
    # 4 km deep), and the surface built from it on the run's grid.  The
    # cold start's own world hook reads the map back (world_map_for) and
    # builds the invented frame from the test's Earth frame.
    continent = _continent_map()
    world_folder.write_map(
        tmp_path / "test-continent.world",
        np.where(continent.land_fraction > 0.5, continent.terrain_m, -4000.0),
        continent.latitude_deg, continent.longitude_deg,
        source={"name": "test continent"})

    def through_the_world(cfg, transform, frame=None, statics=None, **kwargs):
        return real(cfg, transform, frame=_synthetic_frame(), statics=statics, **kwargs)

    monkeypatch.setattr(analysis_initial, "analysis_initial_state", through_the_world)
    base = _analysis_cfg(tmp_path, statics=(
        '[statics]\nsource = "real"\n'))
    text = (tmp_path / "analysis.toml").read_text(encoding="utf-8")
    text = text.replace("dt_s = 10.0", "dt_s = 300.0", 1)
    text = text.replace("duration_s = 40.0", "duration_s = 86400.0", 1)
    text = text.replace("output_interval_s = 20.0", "output_interval_s = 43200.0", 1)
    text += "\n" + world.to_toml()
    path = tmp_path / "invented.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    grid = GaussianGrid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                               dealias_factor=cfg.dealias_factor)
    world_surface.build_world_surface(
        cfg.world, grid, base=tmp_path,
        deep_soil_k=300.0 - 40.0 * np.sin(np.deg2rad(grid.latitude_deg)) ** 2)
    assert cfg.world.is_invented and base.world.is_earth
    assert cfg.duration_s == 86400.0
    outdir = tmp_path / "run"
    run(cfg, outdir)
    receipt = json.loads((outdir / "arwen-global-receipt.json").read_text(encoding="utf-8"))
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    assert checkpoints and checkpoints[-1].name.endswith(f"{288:08d}.npz")
    with np.load(checkpoints[-1], allow_pickle=False) as data:
        for name in data.files:
            if data[name].dtype.kind == "f":
                assert np.isfinite(data[name]).all(), name
    assert receipt["status"] in ("pass", "fail"), receipt.get("status")
    failed = sorted(name for name, row in receipt.get("gates", {}).items()
                    if isinstance(row, dict) and not row.get("passed", True))
    print("invented-planet day: status", receipt["status"], "failed gates", failed)
