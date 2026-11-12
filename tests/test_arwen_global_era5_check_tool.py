"""The instrument that checks an ERA5 start against ERA5 measures zero on a
state that IS ERA5, and finds a planted departure where it was planted.

THE BREAKAGE THIS PREVENTS.  `tools/arwen_global_era5_check.py` is what the
hour-0 table and the multi-day surface rows of docs/ARWEN_GLOBAL_ERA5.md rest
on.  An instrument that samples a row off, interpolates in the wrong
direction, hands soil layers over by position or counts a class from the
wrong tape still prints a table of plausible small numbers, and "hour 0
agrees with ERA5" would then be a statement about the instrument.  So it is
run on cases with a known answer: model columns placed on ERA5 points and
model levels on ERA5 levels, where the sampling and the interpolation are
exact, with offsets planted in one band, one field and one surface class.

No device, no model and no network: numpy, and netCDF4 for the tapes.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[1]
LEVELS_HPA = np.array([1.0, 10.0, 100.0, 300.0, 500.0, 700.0, 850.0, 1000.0])


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_era5_check", TREE / "tools" / "arwen_global_era5_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _era5_grid(step=10.0, lon0=0.0):
    lat = np.arange(90.0, -90.0 - 1e-9, -step)
    lon = lon0 + np.arange(int(round(360.0 / step))) * step
    return lat, lon


def _era5_hour(lat, lon):
    """An ERA5 hour as `read` writes it, from smooth functions."""

    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ny, nx = la.shape
    nlev = LEVELS_HPA.size
    lnp = np.log(LEVELS_HPA * 100.0)[:, None, None]
    land = (np.cos(np.deg2rad(lo)) > 0.3).astype(np.float64)
    ice = np.where(land >= 0.5, np.nan, np.where(np.abs(la) > 70, 1.0, 0.0))
    sst = np.where(land >= 0.5, np.nan, np.where(np.abs(la) > 70, np.nan, 300.0 - 0.3 * np.abs(la)))
    snow = np.where((land >= 0.5) & (la > 40), 0.2, 0.0)          # m of water
    era5 = {
        "latitude": lat, "longitude": lon, "level": LEVELS_HPA,
        "surface_pressure": 101000.0 - 10.0 * np.abs(la),
        "geopotential_at_surface": 9.80665 * (500.0 + 300.0 * np.sin(np.deg2rad(lo)) * np.cos(np.deg2rad(la))),
        "land_sea_mask": land, "skin_temperature": 290.0 - 0.4 * np.abs(la),
        "sea_surface_temperature": sst, "sea_ice_cover": ice,
        "snow_depth": snow, "snow_density": np.full((ny, nx), 250.0),
        "2m_temperature": 288.0 - 0.4 * np.abs(la),
        "total_column_water_vapour": 30.0 - 0.3 * np.abs(la),
        "temperature": 200.0 + 8.0 * lnp + 0.1 * la[None],
        "specific_humidity": 1e-3 * np.exp(lnp - lnp[-1]) + 0.0 * la[None],
        "u_component_of_wind": 10.0 + 0.0 * lnp + 0.2 * la[None],
        "v_component_of_wind": -2.0 + 0.0 * lnp + 0.05 * lo[None],
        "geopotential": 9.80665 * (16000.0 - 1500.0 * (lnp - lnp[0])) + 0.0 * la[None],
    }
    for n in range(4):
        era5[f"soil_temperature_level_{n + 1}"] = 280.0 + n + 0.0 * la
        era5[f"volumetric_soil_water_layer_{n + 1}"] = 0.1 * (n + 1) + 0.0 * la
    assert nlev == era5["temperature"].shape[0]
    return era5


def _model_state(tool, era5, rows, cols):
    """A model state that is ERA5 on a subset of ERA5's points and levels."""

    pick = lambda f: np.asarray(f)[..., rows, :][..., cols]  # noqa: E731
    lat, lon = era5["latitude"][rows], era5["longitude"][cols]
    ps = pick(era5["surface_pressure"])
    # Model full levels on ERA5's levels above the ground; half levels
    # between them, the surface at the bottom.
    p_full = (LEVELS_HPA[:, None, None] * 100.0) * np.ones((1, lat.size, lon.size))
    p_full[-1] = np.minimum(p_full[-1], ps - 1.0)
    p_half = np.concatenate([np.full((1,) + ps.shape, 50.0),
                             0.5 * (p_full[:-1] + p_full[1:]), ps[None]], axis=0)
    land = pick(era5["land_sea_mask"])
    overlap = tool.soil_overlap_weights()
    stack_t = np.stack([pick(era5[f"soil_temperature_level_{n + 1}"]) for n in range(4)])
    stack_w = np.stack([pick(era5[f"volumetric_soil_water_layer_{n + 1}"]) for n in range(4)])
    state = {
        "lat": lat, "lon": lon, "ps": ps, "p_full": p_full, "p_half": p_half,
        "T": pick(era5["temperature"]), "q": pick(era5["specific_humidity"]),
        "u": pick(era5["u_component_of_wind"]), "v": pick(era5["v_component_of_wind"]),
        "phi": pick(era5["geopotential_at_surface"]),
        "sfc_land_fraction": land,
        "sfc_sea_ice_fraction": np.nan_to_num(pick(era5["sea_ice_cover"])),
        "sfc_temperature_k": np.where(land >= 0.5, pick(era5["skin_temperature"]),
                                      np.nan_to_num(pick(era5["sea_surface_temperature"]),
                                                    nan=271.0)),
        "sfc_soil_temperature_k": np.einsum("ks,syx->kyx", overlap, stack_t),
        "sfc_soil_water_fraction": np.einsum("ks,syx->kyx", overlap, stack_w),
        "sfc_landuse_category": np.where(land >= 0.5, 10.0, 17.0),
        "phys_noah_snow": np.where(land >= 0.5, pick(era5["snow_depth"]) * 1000.0, 0.0),
        "phys_noah_snowh": np.where(land >= 0.5, pick(era5["snow_depth"]) * 1000.0 / 250.0, 0.0),
    }
    return state


def _rows(result, section):
    return result["rows"][section]


def test_soil_overlap_is_by_depth(tool):
    w = tool.soil_overlap_weights()
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w[0], [0.7, 0.3, 0.0, 0.0])
    # Noah's 10-40 cm: 18 cm of ERA5's 7-28 cm layer and 12 cm of its 28-100.
    assert np.allclose(w[1], [0.0, 0.6, 0.4, 0.0])
    assert np.allclose(w[2], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(w[3], [0.0, 0.0, 0.0, 1.0])


def test_sampler_closes_the_ring_and_reads_rows_north_to_south(tool):
    lat, lon = _era5_grid(10.0, lon0=-10.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    field = 3.0 * la + 0.0 * lo
    # Off-grid points, one in the seam between the last column and the first.
    sample = tool.Sampler(lat, lon, np.array([45.0, -33.0]), np.array([355.0, 5.0, 123.0]))
    got = sample(field)
    assert np.allclose(got, 3.0 * np.array([45.0, -33.0])[:, None])
    ring = np.cos(np.deg2rad(lo))
    sample = tool.Sampler(lat, lon, np.array([0.0]), np.array([355.0]))
    expected = 0.5 * (np.cos(np.deg2rad(350.0)) + np.cos(np.deg2rad(360.0)))
    assert np.isclose(sample(ring)[0, 0], expected)
    with pytest.raises(ValueError):
        tool.Sampler(lat[::-1], lon, np.array([0.0]), np.array([0.0]))


def test_hour0_reads_zero_on_a_state_that_is_era5(tool):
    lat, lon = _era5_grid(10.0)
    era5 = _era5_hour(lat, lon)
    rows, cols = np.arange(1, lat.size - 1), np.arange(lon.size)
    result = tool.compare_hour0(_model_state(tool, era5, rows, cols), era5)
    assert result["orientation"]["as_is"] == pytest.approx(1.0)
    assert result["orientation"]["shifted_180"] < 0.0
    for section in ("atmosphere on model levels", "soil, interior snow-free land"):
        for label, stats in _rows(result, section).items():
            if stats is None:
                continue
            assert abs(stats["bias"]) < 1e-9, (label, stats)
            assert stats["rms"] < 1e-9, (label, stats)
    terrain = _rows(result, "terrain and surface pressure")
    assert terrain["terrain height, m"]["rms"] < 1e-9
    assert terrain["surface pressure, ERA5's as it is, hPa"]["rms"] < 1e-9
    surface = _rows(result, "surface")
    assert surface["snow water, interior land, kg m-2"]["rms"] < 1e-9
    assert surface["snow depth, interior land, m"]["rms"] < 1e-9
    assert surface["sea-ice fraction (stencil all sea)"]["rms"] < 1e-9
    assert result["open_water_columns_off_era5_sst"]["more_than_1_k"] == 0


def test_hour0_finds_each_planted_departure_where_it_was_planted(tool):
    lat, lon = _era5_grid(10.0)
    era5 = _era5_hour(lat, lon)
    rows, cols = np.arange(1, lat.size - 1), np.arange(lon.size)
    state = _model_state(tool, era5, rows, cols)
    band = (state["p_full"] >= 30000.0) & (state["p_full"] < 70000.0)
    state["T"] = np.where(band, state["T"] + 1.5, state["T"])
    state["u"] = state["u"] - 0.25
    # Soil layers handed over by position: Noah layer 1 reads ERA5 layer 1.
    state["sfc_soil_water_fraction"][0] = np.asarray(
        era5["volumetric_soil_water_layer_1"])[rows][:, cols]
    # The unit trap: snow water in metres.
    state["phys_noah_snow"] = state["phys_noah_snow"] / 1000.0
    result = tool.compare_hour0(state, era5)
    atmosphere = _rows(result, "atmosphere on model levels")
    assert atmosphere["T 300-700 hPa, K"]["bias"] == pytest.approx(1.5)
    for label in ("T 1-10 hPa, K", "T 10-100 hPa, K", "T 100-300 hPa, K", "T 700 hPa to the ground, K"):
        if atmosphere[label] is not None:
            assert abs(atmosphere[label]["bias"]) < 1e-9, label
    for label, stats in atmosphere.items():
        if label.startswith("u ") and stats is not None:
            assert stats["bias"] == pytest.approx(-0.25), label
    soil = _rows(result, "soil, interior snow-free land")
    # 0.1 against 0.7 * 0.1 + 0.3 * 0.2 = 0.13
    assert soil["soil water, Noah layer 1, m3 m-3"]["bias"] == pytest.approx(-0.03)
    assert abs(soil["soil water, Noah layer 2, m3 m-3"]["bias"]) < 1e-9
    snow = _rows(result, "surface")["snow water, interior land, kg m-2"]
    assert snow["bias"] < 0.0
    assert snow["max_abs"] == pytest.approx(200.0 - 0.2)


def test_hour0_refuses_era5_levels_bottom_first(tool):
    lat, lon = _era5_grid(10.0)
    era5 = _era5_hour(lat, lon)
    state = _model_state(tool, era5, np.arange(1, lat.size - 1), np.arange(lon.size))
    era5["level"] = era5["level"][::-1]
    with pytest.raises(ValueError, match="top first"):
        tool.compare_hour0(state, era5)


# ------------------------------------------------------------------ tapes

def _write_era5_nc(path, lat, lon, valid, t2_offset=0.0):
    netCDF4 = pytest.importorskip("netCDF4")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", 1)
        d.createDimension("level", LEVELS_HPA.size)
        d.createDimension("latitude", lat.size)
        d.createDimension("longitude", lon.size)
        t = d.createVariable("time", "f8", ("time",))
        t.units = "hours since 1900-01-01 00:00:00"
        t[:] = [(valid - dt.datetime(1900, 1, 1)).total_seconds() / 3600.0]
        d.createVariable("level", "f8", ("level",))[:] = LEVELS_HPA[::-1]
        d.createVariable("latitude", "f8", ("latitude",))[:] = lat
        d.createVariable("longitude", "f8", ("longitude",))[:] = lon
        plane = ("time", "latitude", "longitude")
        cube = ("time", "level", "latitude", "longitude")
        d.createVariable("PSFC", "f4", plane)[:] = (100000.0 + 0.0 * la)[None]
        d.createVariable("T2", "f4", plane)[:] = (280.0 - 0.3 * np.abs(la) + t2_offset)[None]
        d.createVariable("SKINTEMP", "f4", plane)[:] = (281.0 - 0.3 * np.abs(la))[None]
        z = 9.80665 * (5500.0 + 0.0 * la)
        d.createVariable("Z", "f4", cube)[:] = np.broadcast_to(z, (LEVELS_HPA.size,) + la.shape)[None]
        d.createVariable("SPFH", "f4", cube)[:] = np.zeros((1, LEVELS_HPA.size) + la.shape)


def _write_tape(path, lat, lon, valid, *, snow, t2, tsk, seaice, xland, lu):
    netCDF4 = pytest.importorskip("netCDF4")
    ny, nx = lat.size, lon.size
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    nz = 3
    with netCDF4.Dataset(path, "w") as d:
        d.ISICE = 15
        d.createDimension("Time", 1)
        d.createDimension("DateStrLen", 19)
        d.createDimension("bottom_top", nz)
        d.createDimension("bottom_top_stag", nz + 1)
        d.createDimension("south_north", ny)
        d.createDimension("west_east", nx)
        times = d.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = netCDF4.stringtochar(np.array([valid.strftime("%Y-%m-%d_%H:%M:%S")], "S19"))[0]
        plane = ("Time", "south_north", "west_east")
        d.createVariable("XLAT", "f4", plane)[:] = la[None]
        d.createVariable("XLONG", "f4", plane)[:] = lo[None]
        d.createVariable("PSFC", "f4", plane)[:] = np.full((1, ny, nx), 100000.0)
        pb = np.array([90000.0, 50000.0, 20000.0])[:, None, None] * np.ones((nz, ny, nx))
        d.createVariable("PB", "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = pb[None]
        phb = 9.80665 * np.array([0.0, 2000.0, 9000.0, 14000.0])[:, None, None] * np.ones((nz + 1, ny, nx))
        d.createVariable("PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))[:] = phb[None]
        d.createVariable("QVAPOR", "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = np.zeros((1, nz, ny, nx))
        for name, values in (("T2", t2), ("TSK", tsk), ("SNOW", snow), ("SEAICE", seaice),
                             ("XLAND", xland), ("LU_INDEX", lu)):
            d.createVariable(name, "f4", plane)[:] = np.asarray(values, dtype=np.float32)[None]


def test_tapes_hold_the_start_classes_and_find_the_planted_cold_snow(tool, tmp_path):
    pytest.importorskip("netCDF4")
    lat, lon = _era5_grid(10.0, lon0=-10.0)
    tape_lat, tape_lon = lat[1:-1], np.arange(0.0, 360.0, 10.0)
    la, lo = np.meshgrid(tape_lat, tape_lon, indexing="ij")
    start = dt.datetime(2019, 1, 15)
    later = start + dt.timedelta(hours=24)
    land = np.cos(np.deg2rad(lo)) > 0.3
    deep = land & (la > 40)
    ice_sheet = land & (la < -60)
    snow0 = np.where(deep, 1500.0, 0.0) + np.where(ice_sheet, 10000.0, 0.0)
    lu = np.where(ice_sheet, 15.0, np.where(land, 10.0, 17.0))
    xland = np.where(land, 1.0, 2.0)
    t2_era = 280.0 - 0.3 * np.abs(la)
    tsk_era = 281.0 - 0.3 * np.abs(la)
    seaice = np.zeros_like(la)
    paths = []
    for valid, cold, snow in ((start, 0.0, snow0),
                              # The snow melts away by hour 24; the class is the start's.
                              (later, -4.0, np.where(deep, 0.0, snow0))):
        # No colons in the name: Windows refuses them, and the tool reads
        # the valid time from the tape's Times, never from its name.
        tape = tmp_path / f"wrfout_d01_{valid:%Y-%m-%d_%H}"
        era = tmp_path / f"era5-{valid:%Y%m%d%H}.nc"
        _write_era5_nc(era, lat, lon, valid)
        _write_tape(tape, tape_lat, tape_lon, valid, snow=snow,
                    t2=np.where(deep, t2_era + cold, t2_era), tsk=tsk_era,
                    seaice=seaice, xland=xland, lu=lu)
        paths.append((str(tape), str(era)))
    # Given in reverse order: the earliest valid time is still the start.
    result = tool.compare_tapes(paths[::-1])
    first, second = result["tapes"]
    assert first["lead_h"] == 0.0 and second["lead_h"] == 24.0
    assert first["nonfinite"] == 0 and second["nonfinite"] == 0
    deep_label = "  of it, start > 1000 kg m-2"
    snowy_label = "snowy land that is not land ice (start > 10 kg m-2)"
    assert first["by_class"][deep_label]["t2"]["bias"] == pytest.approx(0.0, abs=1e-4)
    assert second["by_class"][deep_label]["t2"]["bias"] == pytest.approx(-4.0, abs=1e-4)
    assert second["by_class"][snowy_label]["t2"]["bias"] == pytest.approx(-4.0, abs=1e-4)
    assert second["by_class"][deep_label]["snow_change_kg_m2"] == pytest.approx(-1500.0)
    assert second["by_class"]["snow-free land"]["t2"]["bias"] == pytest.approx(0.0, abs=1e-4)
    assert second["by_class"]["land ice, south of 60S"]["n"] == int(ice_sheet.sum())
    assert first["global"]["500 hPa height, m"]["rms"] < 1e-3
    # A tape paired with ERA5 of another hour is refused.
    with pytest.raises(ValueError, match="its own hour"):
        tool.compare_tapes([(paths[0][0], paths[1][1])])
