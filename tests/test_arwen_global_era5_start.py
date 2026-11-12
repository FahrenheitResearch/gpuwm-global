"""ERA5 as a global cold start: the completion, the units guard, the soil
depths, the mapping, the fetch door and the shipped experiment.

Every number here is a known answer on a plane small enough to read: a
coast point ERA5's sea fields leave out reads the sea beside it, enclosed
water with 0.2 m of ice is frozen and 0.2 m thick, an ice-free water point's
skin is ERA5's SST, 300 kg m-2 of snow at 250 kg m-3 is 1.2 m deep, an hPa
ladder is the same initial state as the Pa ladder it scales to.  No test
fetches anything or needs a card.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from arwen_global import analysis_fetch, era5_analysis
from arwen_global.analysis_initial import (
    PACKAGE_OWN_MAPPINGS_DIR,
    analysis_initial_state,
    analysis_levels_pa,
    resolve_analysis_mapping,
    resolve_analysis_mapping_row,
    soil_layer_bounds_m,
)
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.runner import build_transform

ERA5_MAPPING_FILE = "rw-wps-era5-arco-global-analysis-netcdf.mapping.json"
VALID = dt.datetime(2019, 5, 20, 0)
ERA5_LEVELS_HPA = np.asarray([
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000], dtype=np.float64)


def _era5_mapping() -> Path:
    return PACKAGE_OWN_MAPPINGS_DIR / ERA5_MAPPING_FILE


# ------------------------------------------------------------- the mapping

def test_the_bare_id_resolves_to_this_packages_own_mapping_only():
    row = resolve_analysis_mapping_row(era5_analysis.ERA5_GLOBAL_MAPPING_ID)
    assert row.path == _era5_mapping().resolve()
    assert row.origin == "package"
    document = json.loads(row.path.read_text(encoding="utf-8"))
    assert document["name"] == era5_analysis.ERA5_GLOBAL_MAPPING_NAME
    assert document["format"] == "netcdf"
    assert document["coordinates"]["vertical"]["units"] == "hPa"
    assert era5_analysis.is_era5_mapping(row.path)
    assert not era5_analysis.is_era5_mapping(resolve_analysis_mapping("gdas-global"))
    # The decoded fields the cold start reads, from the bridge's names; the
    # four the completion adds are not in the mapping.
    selectors = {name: [s["name"][0] for s in field["selectors"]]
                 for name, field in document["fields"].items()}
    assert selectors["specific_humidity"] == ["SPFH"]
    assert selectors["terrain_height"] == ["SOILGEO"]
    assert selectors["soil_temperature"] == ["ST000007", "ST007028", "ST028100", "ST100289"]
    assert document["fields"]["terrain_height"]["units"]["scale"] == pytest.approx(1 / 9.80665)
    assert document["fields"]["snow_water_equivalent"]["units"]["scale"] == 1000.0
    assert "sea_ice_thickness" not in document["fields"]
    assert "snow_depth" not in document["fields"]


def test_the_era5_soil_layers_reach_noah_at_their_depths():
    expected = ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89))
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        assert soil_layer_bounds_m(_era5_mapping(), name) == expected


def test_a_netcdf_soil_stack_without_a_depth_row_is_refused(tmp_path):
    document = json.loads(_era5_mapping().read_text(encoding="utf-8"))
    document["name"] = "a-netcdf-source-with-no-depth-row"
    path = tmp_path / "m.mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="carries no depths"):
        soil_layer_bounds_m(path, "soil_temperature")


# ------------------------------------------------------------ units guard

def test_hpa_levels_are_read_in_pa_and_pa_levels_are_unchanged(tmp_path):
    levels, units = analysis_levels_pa(ERA5_LEVELS_HPA, _era5_mapping())
    assert units == "hPa"
    assert levels[0] == 100.0 and levels[-1] == 100000.0
    gdas = resolve_analysis_mapping("gdas-global")
    same, units = analysis_levels_pa(ERA5_LEVELS_HPA * 100.0, gdas)
    assert units == "Pa" and np.array_equal(same, ERA5_LEVELS_HPA * 100.0)


def test_an_unknown_unit_and_a_ladder_in_the_wrong_unit_are_refused(tmp_path):
    document = json.loads(_era5_mapping().read_text(encoding="utf-8"))
    document["coordinates"]["vertical"]["units"] = "mbar"
    path = tmp_path / "mbar.mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="will not guess"):
        analysis_levels_pa(ERA5_LEVELS_HPA, path)
    # hPa values through a mapping that declares Pa: the silent failure.
    with pytest.raises(ValueError, match="not in that unit"):
        analysis_levels_pa(ERA5_LEVELS_HPA, resolve_analysis_mapping("gdas-global"))


# ------------------------------------------------------------ the completion

def _planes():
    """A 5 x 4 analysis, the longitude a ring.  Row 0 land; row 1 sea (ice on
    the left); row 2 land with two coast points ERA5's sea fields leave out
    (land fraction 0.4 and 0.3, no sea-ice value, no SST); row 3 land; row 4
    land around two points of enclosed water no sea value reaches, the left
    one with 0.2 m of lake ice."""

    nan = np.nan
    land = np.asarray([[1.0, 1.0, 0.9, 0.6],
                       [0.0, 0.0, 0.2, 0.0],
                       [0.4, 1.0, 0.3, 1.0],
                       [1.0, 1.0, 1.0, 1.0],
                       [1.0, 0.0, 0.0, 1.0]])
    soil = np.stack([np.full((5, 4), 280.0 + k) for k in range(4)])
    water = np.stack([np.full((5, 4), 0.3 - 0.01 * k) for k in range(4)])
    water[:, 1, :] = -0.005           # ERA5's "soil water" over the sea
    ice = np.asarray([[nan, nan, nan, nan],
                      [0.6, 0.0, 0.0, 0.0],
                      [nan, nan, nan, nan],
                      [nan, nan, nan, nan],
                      [nan, nan, nan, nan]])
    sst = np.asarray([[nan, nan, nan, nan],
                      [271.46, 285.0, 286.0, 287.0],
                      [nan, nan, nan, nan],
                      [nan, nan, nan, nan],
                      [nan, nan, nan, nan]])
    # The skin is the mean over a point's tiles: ice and water on the pack,
    # 0.3 K of cool skin on the open sea, ground on the two coast points.
    skin = np.asarray([[290.0, 291.0, 292.0, 293.0],
                       [262.0, 284.7, 285.7, 286.7],
                       [296.0, 300.0, 299.0, 301.0],
                       [302.0, 302.0, 302.0, 302.0],
                       [303.0, 268.0, 277.0, 303.0]])
    lake_ice = np.asarray([[0.1] * 4,           # FLake answers everywhere,
                           [0.1] * 4,           # lake or not
                           [0.5, 0.0, 0.5, 0.0],
                           [0.0] * 4,
                           [0.0, 0.2, 0.0, 0.0]])
    swe = np.asarray([[300.0, 0.0, 10000.0, 50.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
    density = np.full((5, 4), 100.0)
    density[0] = [250.0, 100.0, 400.0, nan]
    fields = {"land_fraction": land, "soil_temperature": soil,
              "volumetric_soil_moisture": water, "sea_ice_fraction": ice,
              "skin_temperature": skin, "snow_water_equivalent": swe}
    return fields, lake_ice, density, sst


def test_the_completion_has_known_answers():
    fields, lake_ice, density, sst = _planes()
    before = {name: np.array(values, copy=True) for name, values in fields.items()}
    out, record = era5_analysis.complete_era5_fields(
        fields, lake_ice_depth_m=lake_ice, snow_density_kg_m3=density,
        sea_surface_temperature_k=sst)

    # Inputs untouched.
    for name, values in before.items():
        assert np.array_equal(values, fields[name], equal_nan=True), name
    land = before["land_fraction"] >= 0.5
    # Soil: missing exactly on water, untouched on land.
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        assert np.isnan(out[name][:, ~land]).all()
        assert np.array_equal(out[name][:, land], before[name][:, land])
    assert record["soil_over_water"]["values_masked"] == 2 * 4 * int((~land).sum())
    # Sea ice: no holes; land 0; sea as analysed.
    ice = out["sea_ice_fraction"]
    assert np.isfinite(ice).all()
    assert np.array_equal(ice[0], [0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(ice[1], [0.6, 0.0, 0.0, 0.0])
    assert np.array_equal(ice[3], [0.0, 0.0, 0.0, 0.0])
    # The two coast points read the sea beside them, across the ring: the
    # left one has the pack among its three sea neighbours (0.0, 0.6, 0.0)
    # and the other has open sea only.  FLake's 0.5 m of "lake ice" on both
    # answers for neither.
    assert ice[2, 0] == pytest.approx(0.2)
    assert ice[2, 2] == 0.0
    assert np.array_equal(ice[2, [1, 3]], [0.0, 0.0])
    # Enclosed water, which no sea value reaches: FLake's ice depth answers.
    assert np.array_equal(ice[4], [0.0, 1.0, 0.0, 0.0])
    # Thickness: 3 m stand-in on sea ice (the coast point with ice too), the
    # enclosed water's own ice depth, 0 elsewhere.
    thickness = out["sea_ice_thickness"]
    assert thickness[1, 0] == era5_analysis.SEA_ICE_THICKNESS_STANDIN_M == 3.0
    assert thickness[2, 0] == 3.0
    assert thickness[4, 1] == pytest.approx(0.2)
    assert np.count_nonzero(thickness) == 3
    sea = record["sea_ice_fraction"]
    assert sea["missing_on_land"] == 12
    assert sea["coast_points_without_a_sea_value"] == 4
    assert sea["coast_points_filled_from_the_sea_beside"] == 2
    assert sea["coast_points_filled_with_ice"] == 1
    assert sea["enclosed_water_points"] == 2
    assert sea["frozen_enclosed_water_points"] == 1
    assert sea["sea_ice_points"] == 2
    assert record["sea_ice_thickness"]["stand_in"] is True
    # Skin: ERA5's SST on the ice-free sea; the mean of the open sea beside
    # it on the ice-free coast point, not that point's 299 K of ground; the
    # skin itself on the pack, on the coast point with ice, on the enclosed
    # water and on land.
    skin = out["skin_temperature"]
    assert np.array_equal(skin[1], [262.0, 285.0, 286.0, 287.0])
    assert skin[2, 2] == pytest.approx((285.0 + 286.0 + 287.0) / 3.0)
    assert skin[2, 0] == 296.0
    assert np.array_equal(skin[4], before["skin_temperature"][4])
    assert np.array_equal(skin[land], before["skin_temperature"][land])
    water_t = record["sea_surface_temperature"]
    assert water_t["water_points_taking_sst"] == 4
    assert water_t["water_points_keeping_skin"] == 4
    assert water_t["ice_free_water_points_no_sst_reaches"] == 1
    assert water_t["sst_minus_skin_k"]["points_changed_by_more_than_1_k"] == 1
    assert water_t["sst_minus_skin_k"]["range"] == pytest.approx([-13.0, 0.3])
    # Snow depth: water / density; no density, no depth (counted).
    depth = out["snow_depth"]
    assert depth[0, 0] == pytest.approx(300.0 / 250.0)
    assert depth[0, 2] == pytest.approx(10000.0 / 400.0)
    assert depth[0, 3] == 0.0
    assert np.count_nonzero(depth) == 2
    assert record["snow_depth"]["snowy_points"] == 3
    assert record["snow_depth"]["snowy_points_without_density"] == 1
    # No cap: ERA5's glacier water is taken as analysed.
    assert out["snow_water_equivalent"][0, 2] == 10000.0


def test_a_masked_snow_water_point_stays_masked_in_the_depth():
    fields, lake_ice, density, sst = _planes()
    fields["snow_water_equivalent"] = fields["snow_water_equivalent"].copy()
    fields["snow_water_equivalent"][1, 3] = np.nan
    out, _ = era5_analysis.complete_era5_fields(
        fields, lake_ice_depth_m=lake_ice, snow_density_kg_m3=density,
        sea_surface_temperature_k=sst)
    assert np.isnan(out["snow_depth"][1, 3])
    assert np.isfinite(np.delete(out["snow_depth"].ravel(), 7)).all()


def test_a_frame_without_snow_on_its_ice_starts_the_pack_bare():
    """The seeding's ERA5 arm on a plane with a glacier beside the pack: 10 m
    of water on the land, none on the ice.  Read the GDAS way the pack takes
    the glacier's snow; read as ERA5 is, the pack is bare and the land's
    snow is not diluted by the bare ice beside it."""

    from arwen_global.analysis_initial import _global_regridder
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.surface_seeding import seed_surface_from_analysis, synthetic_analysis

    assert era5_analysis.Era5Frame.snow_on_sea_ice is False
    grid = GaussianGrid.create(21)
    # A Gaussian row that sits past the middle of a one-degree analysis cell:
    # with land up to the cell's south edge and ice from its north edge, the
    # row reads under half glacier and over half pack, a frozen water column.
    rows = np.asarray(grid.latitude_deg, dtype=np.float64)
    row = next(r for r in rows if 55.0 < r < 80.0 and 0.55 < r - np.floor(r) < 0.95)
    south = float(np.floor(row))
    frame = synthetic_analysis(
        ice_edge_deg=south + 1.0, snow_line_deg=50.0, land_band=(30.0, south),
        swe_kg_m2=10000.0, snow_depth_m=33.0)
    # ERA5's shape: no bitmap on the snow planes, 0 on water and on ice.
    for name in ("snow_water_equivalent", "snow_depth"):
        frame.fields[name].values = np.nan_to_num(frame.fields[name].values)
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)
    target_land = regrid(frame.fields["land_fraction"].values)
    as_gdas = seed_surface_from_analysis(frame, regrid, target_land_fraction=target_land)
    as_era5 = seed_surface_from_analysis(
        frame, regrid, target_land_fraction=target_land, snow_on_sea_ice=False)
    pack = (target_land <= 0.5) & (as_era5.sea_ice_fraction >= 0.5)
    edge = pack & (target_land > 0.0)
    assert edge.any()
    # The defect: the glacier's share of the stencil, as snow on the pack.
    assert np.allclose(as_gdas.snow_water_kg_m2[edge], 10000.0 * target_land[edge])
    assert as_gdas.snow_water_kg_m2[edge].min() > 500.0
    assert as_era5.snow_water_kg_m2[pack].max() == 0.0
    assert as_era5.snow_depth_m[pack].max() == 0.0
    assert as_era5.snow_cover[pack].max() == 0.0
    # Land keeps its own snow either way.
    inland = (target_land == 1.0) & (rows[:, None] > 52.0)
    assert inland.any()
    assert np.allclose(as_era5.snow_water_kg_m2[inland], 10000.0)
    record = as_era5.provenance["snow_on_sea_ice"]
    assert record["frozen_water_columns"] == int(pack.sum())
    assert record["columns_cleared_of_land_snow"] > 0
    assert "snow_on_sea_ice" not in as_gdas.provenance
    assert np.array_equal(as_gdas.sea_ice_fraction, as_era5.sea_ice_fraction)


# ----------------------------------------------- the supplement files

def _write_bridge_plane(path: Path, variable: str, values, latitude, longitude,
                        when=VALID, units="1", more=()):
    """One surface plane as the bridge writes it; ``more`` is further
    ``(variable, values, units)`` planes of the same file."""

    from netCDF4 import Dataset

    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("latitude", latitude.size)
        dataset.createDimension("longitude", longitude.size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "seconds since 1970-01-01 00:00:00"
        time[:] = [(when - dt.datetime(1970, 1, 1)).total_seconds()]
        dataset.createVariable("latitude", "f8", ("latitude",))[:] = latitude
        dataset.createVariable("longitude", "f8", ("longitude",))[:] = longitude
        for name, plane, unit in ((variable, values, units), *more):
            written = dataset.createVariable(
                name, "f8", ("time", "latitude", "longitude"), fill_value=np.nan)
            written.units = unit
            written[0] = plane


def _sst_of(frame) -> np.ndarray:
    """ERA5's SST for a synthetic frame: the skin on its sea points, missing
    where its sea-ice cover is."""

    return np.where(np.isfinite(frame.fields["sea_ice_fraction"].values),
                    frame.fields["skin_temperature"].values, np.nan)


@dataclasses.dataclass
class _Field:
    values: np.ndarray


@dataclasses.dataclass
class _Frame:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_values: np.ndarray
    fields: dict
    mapping_sha256: str = "test-mapping"
    input_sha256: str = "test-input"
    source_cycle: dt.datetime = VALID
    valid_time: dt.datetime = VALID


def _era5_frame(nlat=37, nlon=72):
    """An ERA5-shaped decoded frame: 37 levels in hPa, ascending; soil
    values over the sea; sea ice missing on land; no thickness, no depth;
    the longitude ring starting one column west of 0 as the bridge writes it."""

    latitude = np.linspace(90.0, -90.0, nlat)
    step = 360.0 / nlon
    longitude = -step + np.arange(nlon) * step
    levels = ERA5_LEVELS_HPA
    lat2 = np.deg2rad(latitude)[:, None] * np.ones((1, nlon))
    lon2 = np.deg2rad(longitude)[None, :] * np.ones((nlat, 1))
    p = (levels * 100.0)[:, None, None]
    shape3 = (levels.size, nlat, nlon)
    temperature = np.maximum(
        200.0, 288.0 * (p / 100000.0) ** 0.19) - 25.0 * np.sin(lat2)[None] ** 2
    humidity = 0.012 * (p / 100000.0) ** 3 * np.cos(lat2)[None] ** 2 * np.ones(shape3)
    u = 15.0 * np.cos(lat2)[None] * np.ones(shape3)
    v = np.zeros(shape3)
    envelope = np.exp(-((np.rad2deg(lat2) - 30.0) / 15.0) ** 2)
    terrain = 1500.0 * envelope * (1.0 + 0.3 * np.cos(lon2))
    ps = 101000.0 * np.exp(-terrain / 8000.0)
    land = (terrain > 100.0).astype(np.float64)
    skin = 288.0 - 30.0 * np.sin(lat2) ** 2
    soil_t = np.stack([skin - 1.0 - k for k in range(4)])
    soil_m = np.stack([np.where(land > 0.5, 0.3, -0.004) for _ in range(4)])
    lat_deg = np.rad2deg(lat2)
    ice = np.where(land >= 0.5, np.nan, np.where(np.abs(lat_deg) >= 70.0, 0.9, 0.0))
    swe = np.where((land >= 0.5) & (lat_deg >= 38.0), 60.0, 0.0)
    fields = {
        "air_temperature": temperature, "specific_humidity": humidity,
        "eastward_wind": u, "northward_wind": v, "surface_pressure": ps,
        "terrain_height": terrain, "skin_temperature": skin,
        "land_fraction": land, "soil_temperature": soil_t,
        "volumetric_soil_moisture": soil_m, "sea_ice_fraction": ice,
        "snow_water_equivalent": swe,
    }
    frame = _Frame(latitude, longitude, "pressure", levels,
                   {name: _Field(values) for name, values in fields.items()})
    return frame


def _write_supplements(directory: Path, frame, *, when=VALID):
    shape = (frame.latitude.size, frame.longitude.size)
    analysis = directory / "era5-combined.nc"
    _write_bridge_plane(analysis, "LAKE_ICE_DEPTH", np.full(shape, np.nan),
                        frame.latitude, frame.longitude, when=when, units="m",
                        more=(("SST", _sst_of(frame), "K"),))
    _write_bridge_plane(directory / era5_analysis.SNOW_DENSITY_FILE,
                        "SNOW_DENSITY", np.full(shape, 240.0),
                        frame.latitude, frame.longitude, when=when,
                        units="kg m-3")
    return analysis


def test_complete_era5_frame_reads_the_supplements_beside_the_analysis(tmp_path):
    pytest.importorskip("netCDF4")
    frame = _era5_frame()
    analysis = _write_supplements(tmp_path, frame)
    completed, record = era5_analysis.complete_era5_frame(frame, analysis)
    depth = completed.fields["snow_depth"].values
    swe = frame.fields["snow_water_equivalent"].values
    assert np.allclose(depth[swe > 0], 60.0 / 240.0)
    assert completed.valid_time == VALID
    assert completed.vertical_values is frame.vertical_values
    assert record["snow_density_file"].endswith(era5_analysis.SNOW_DENSITY_FILE)


def test_a_supplement_of_another_hour_or_none_at_all_is_refused(tmp_path):
    pytest.importorskip("netCDF4")
    frame = _era5_frame()
    analysis = _write_supplements(tmp_path, frame, when=VALID + dt.timedelta(hours=1))
    with pytest.raises(ValueError, match="same hour"):
        era5_analysis.complete_era5_frame(frame, analysis)
    (tmp_path / era5_analysis.SNOW_DENSITY_FILE).unlink()
    _write_bridge_plane(analysis, "LAKE_ICE_DEPTH",
                        np.zeros((frame.latitude.size, frame.longitude.size)),
                        frame.latitude, frame.longitude,
                        more=(("SST", _sst_of(frame), "K"),))
    with pytest.raises(FileNotFoundError, match="fetch-analysis --source era5"):
        era5_analysis.complete_era5_frame(frame, analysis)


def test_a_supplement_on_another_grid_is_refused(tmp_path):
    pytest.importorskip("netCDF4")
    frame = _era5_frame()
    analysis = _write_supplements(tmp_path, frame)
    shifted = dataclasses.replace(frame, longitude=frame.longitude + 5.0)
    with pytest.raises(ValueError, match="not on the decoded analysis grid"):
        era5_analysis.complete_era5_frame(shifted, analysis)


# ------------------------------------------- through the cold start itself

def _smoke_cfg(tmp_path, mapping, grib):
    """An analysis-mode config on the smoke grid with the synthetic planet;
    ``mapping`` is a bare id (str) or a file (Path)."""

    spelled = mapping.as_posix() if isinstance(mapping, Path) else mapping
    text = (_shipped_configs() / "arwen_global_moist_smoke.toml").read_text(encoding="utf-8")
    replaced = text.replace(
        "[initial]\n",
        "[initial]\nmode = \"analysis\"\n"
        f"analysis_grib = \"{Path(grib).as_posix()}\"\n"
        f"analysis_mapping = \"{spelled}\"\n",
        1)
    kept = [line for line in replaced.splitlines()
            if not any(line.strip().startswith(key) for key in (
                "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
                "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
                "zonal_wavenumber", "terrain_amplitude_m"))]
    path = tmp_path / "analysis.toml"
    path.write_text("\n".join(kept) + "\n[statics]\nsource = \"synthetic\"\n",
                    encoding="utf-8")
    return load_config(path)


def _grid_atmosphere(transform, state):
    inverse = lambda c: np.asarray(  # noqa: E731
        transform.backend.to_numpy(transform.inverse(c)), dtype=np.float64)
    atmosphere = state.atmosphere
    return {name: inverse(getattr(atmosphere, name))
            for name in ("theta", "qv", "log_surface_pressure", "vorticity",
                         "divergence")}


def test_an_hpa_ladder_is_the_same_initial_state_as_its_pa_ladder(tmp_path):
    """The guard's known answer: the frame in hPa through a mapping that
    declares hPa, and the same frame scaled to Pa through one that declares
    Pa, start the same atmosphere."""

    gdas = resolve_analysis_mapping("gdas-global")
    document = json.loads(gdas.read_text(encoding="utf-8"))
    document["coordinates"]["vertical"]["units"] = "hPa"
    hpa_mapping = tmp_path / "gdas-in-hpa.mapping.json"
    hpa_mapping.write_text(json.dumps(document), encoding="utf-8")

    frame = _era5_frame()
    completed, _ = era5_analysis.complete_era5_fields(
        {name: field.values for name, field in frame.fields.items()},
        lake_ice_depth_m=np.zeros_like(frame.fields["land_fraction"].values),
        snow_density_kg_m3=np.full_like(frame.fields["land_fraction"].values, 250.0),
        sea_surface_temperature_k=_sst_of(frame))
    in_hpa = dataclasses.replace(
        frame, fields={name: _Field(values) for name, values in completed.items()})
    in_pa = dataclasses.replace(in_hpa, vertical_values=ERA5_LEVELS_HPA * 100.0)

    cfg_pa = _smoke_cfg(tmp_path, gdas, "unused.grib")
    transform = build_transform(cfg_pa)
    state_pa, _, prov_pa = analysis_initial_state(cfg_pa, transform, frame=in_pa)
    cfg_hpa = dataclasses.replace(cfg_pa, analysis_mapping=str(hpa_mapping))
    state_hpa, _, prov_hpa = analysis_initial_state(cfg_hpa, transform, frame=in_hpa)
    assert prov_pa["analysis_vertical_units"] == "Pa"
    assert prov_hpa["analysis_vertical_units"] == "hPa"
    pa, hpa = _grid_atmosphere(transform, state_pa), _grid_atmosphere(transform, state_hpa)
    for name in pa:
        assert np.array_equal(pa[name], hpa[name]), name
    ps = np.exp(pa["log_surface_pressure"])
    assert 50000.0 < ps.min() and ps.max() < 106000.0


def test_the_cold_start_completes_a_decoded_era5_frame(tmp_path, monkeypatch):
    """The decode branch: an ERA5 mapping's frame is completed before any
    field is read, and the receipt says how."""

    pytest.importorskip("netCDF4")
    from types import SimpleNamespace

    import arwen_global.mapped_source_compat as compat

    frame = _era5_frame()
    analysis = _write_supplements(tmp_path, frame)
    calls = []

    def decode(mapping, inputs, scratch_destination=None):
        calls.append((Path(mapping).name, [Path(p).name for p in inputs]))
        return SimpleNamespace(frames=[frame], receipt={"decoded": "synthetic"})

    monkeypatch.setattr(compat, "decode_through_engine", decode)
    cfg = _smoke_cfg(tmp_path, era5_analysis.ERA5_GLOBAL_MAPPING_ID, analysis)
    transform = build_transform(cfg)
    state, _phi, provenance = analysis_initial_state(cfg, transform)
    assert calls == [(ERA5_MAPPING_FILE, ["era5-combined.nc"])]
    record = provenance["era5_completion"]
    assert record["schema"] == era5_analysis.COMPLETION_SCHEMA
    assert record["sea_ice_thickness"]["stand_in_m"] == 3.0
    assert provenance["analysis_vertical_units"] == "hPa"
    assert provenance["analysis_top"]["analysis_top_pa"] == 100.0
    assert provenance["soil_layers"]["soil_temperature"]["source_bounds_m"][0] == [0.0, 0.07]
    grid = _grid_atmosphere(transform, state)
    for name, values in grid.items():
        assert np.isfinite(values).all(), name
    host = transform.backend.to_numpy
    for name in ("sea_ice_fraction", "sea_ice_thickness_m", "soil_water_fraction",
                 "soil_temperature_k"):
        assert np.isfinite(host(getattr(state.surface, name))).all(), name
    # The sea's -0.004 "soil water" never reaches a column: the water fill
    # (0.25) replaced it before the regrid.
    assert float(host(state.surface.soil_water_fraction).min()) >= 0.0
    # Sea ice exists only where the analysis froze the sea, 3 m thick there.
    thickness = host(state.surface.sea_ice_thickness_m)
    assert float(thickness.max()) == pytest.approx(3.0)


def _write_bridge_file(path: Path, frame) -> None:
    """``frame`` as the engine's ARCO bridge writes it: 1000 hPa first,
    geopotential in m2 s-2, snow in metres of water, and the bridge's
    variable names."""

    from netCDF4 import Dataset

    g = 9.80665  # the mapping's scale, not the model's gravity
    fields = {name: np.asarray(field.values, dtype=np.float64)
              for name, field in frame.fields.items()}
    shape = fields["land_fraction"].shape
    levels = np.asarray(frame.vertical_values)[::-1]
    with Dataset(path, "w") as dataset:
        for name, size in (("time", 1), ("level", levels.size),
                           ("latitude", shape[0]), ("longitude", shape[1])):
            dataset.createDimension(name, size)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "seconds since 1970-01-01 00:00:00"
        time[:] = [(frame.valid_time - dt.datetime(1970, 1, 1)).total_seconds()]
        for name, values, units in (("level", levels, "hPa"),
                                    ("latitude", frame.latitude, "degrees_north"),
                                    ("longitude", frame.longitude, "degrees_east")):
            variable = dataset.createVariable(name, "f8", (name,))
            variable.units = units
            variable[:] = values
        thickness = 287.0 * 250.0 / g * np.log(100000.0 / (levels * 100.0))
        upper = {"T": (fields["air_temperature"], "K"),
                 "SPFH": (fields["specific_humidity"], "kg kg-1"),
                 "U": (fields["eastward_wind"], "m s-1"),
                 "V": (fields["northward_wind"], "m s-1"),
                 "Z": (g * thickness[::-1, None, None] * np.ones(fields["air_temperature"].shape),
                       "m2 s-2")}
        for name, (values, units) in upper.items():
            variable = dataset.createVariable(
                name, "f8", ("time", "level", "latitude", "longitude"))
            variable.units = units
            variable[0] = values[::-1]
        surface = {
            "PSFC": (fields["surface_pressure"], "Pa"),
            "SOILGEO": (g * fields["terrain_height"], "m2 s-2"),
            "SKINTEMP": (fields["skin_temperature"], "K"),
            "T2": (fields["skin_temperature"] - 1.0, "K"),
            "Q2": (np.full(shape, 8.0e-3), "kg kg-1"),
            "U10": (np.full(shape, 3.0), "m s-1"),
            "V10": (np.full(shape, -1.0), "m s-1"),
            "LANDSEA": (fields["land_fraction"], "1"),
            "SEAICE": (fields["sea_ice_fraction"], "1"),
            "SNOW_EC": (fields["snow_water_equivalent"] / 1000.0, "m"),
            "LAKE_ICE_DEPTH": (np.full(shape, np.nan), "m"),
            "SST": (_sst_of(frame), "K"),
        }
        for k, suffix in enumerate(("000007", "007028", "028100", "100289")):
            surface[f"ST{suffix}"] = (fields["soil_temperature"][k], "K")
            surface[f"SM{suffix}"] = (fields["volumetric_soil_moisture"][k], "m3 m-3")
        for name, (values, units) in surface.items():
            variable = dataset.createVariable(
                name, "f8", ("time", "latitude", "longitude"), fill_value=np.nan)
            variable.units = units
            variable[0] = values


def test_the_engine_decodes_a_bridge_shaped_file_to_the_injected_state(
        tmp_path, monkeypatch):
    """The real decode through the shipped mapping, end to end: a small file
    in the bridge's layout (1000 hPa first, hPa levels, geopotential, metres
    of snow water) starts the same atmosphere and surface as the frame it
    was written from.  This is where the hPa ladder, the level order and the
    mapping's scales are all read by the engine rather than by this suite."""

    pytest.importorskip("netCDF4")
    pytest.importorskip("gpuwm.mapped_source")
    from types import SimpleNamespace

    import arwen_global.mapped_source_compat as compat

    frame = _era5_frame()
    analysis = tmp_path / "era5-combined.nc"
    _write_bridge_file(analysis, frame)
    _write_bridge_plane(tmp_path / era5_analysis.SNOW_DENSITY_FILE, "SNOW_DENSITY",
                        np.full(frame.fields["land_fraction"].values.shape, 240.0),
                        frame.latitude, frame.longitude, units="kg m-3")
    cfg = _smoke_cfg(tmp_path, era5_analysis.ERA5_GLOBAL_MAPPING_ID, analysis)
    transform = build_transform(cfg)
    real, _, real_prov = analysis_initial_state(
        cfg, transform, scratch_destination=tmp_path / "scratch")

    monkeypatch.setattr(compat, "decode_through_engine",
                        lambda *a, **k: SimpleNamespace(frames=[frame], receipt={}))
    injected, _, injected_prov = analysis_initial_state(cfg, transform)
    assert real_prov["analysis_vertical_units"] == "hPa"
    assert real_prov["era5_completion"] == injected_prov["era5_completion"]
    a, b = _grid_atmosphere(transform, real), _grid_atmosphere(transform, injected)
    for name in a:
        assert np.allclose(a[name], b[name], rtol=1e-6, atol=1e-9), name
    host = transform.backend.to_numpy
    for name in ("sea_ice_fraction", "sea_ice_thickness_m", "soil_temperature_k",
                 "soil_water_fraction", "temperature_k"):
        assert np.allclose(host(getattr(real.surface, name)),
                           host(getattr(injected.surface, name)),
                           rtol=1e-6, atol=1e-9), name


# ------------------------------------------------------------ the fetch door

def _parser():
    parser = argparse.ArgumentParser()
    analysis_fetch.add_fetch_analysis_arguments(parser)
    return parser


def test_the_door_takes_era5_and_refuses_it_without_an_hour():
    args = _parser().parse_args(["--source", "era5"])
    with pytest.raises(SystemExit, match="needs --cycle"):
        analysis_fetch.fetch_analysis_main(args)
    args = _parser().parse_args([])
    assert args.source == "gdas" and args.out is None


def test_the_era5_door_lands_a_dated_directory_and_prints_the_run(
        tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_fetch(out, cycle, *, progress=print):
        seen["out"], seen["cycle"] = Path(out), cycle
        return {"analysis": Path(out) / "era5-combined.nc",
                "snow_density": Path(out) / era5_analysis.SNOW_DENSITY_FILE,
                "cycle": cycle, "provisional_times": []}

    monkeypatch.setattr(era5_analysis, "fetch_era5_analysis", fake_fetch)
    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args(["--source", "era5", "--cycle", "2019-05-20T00"])
    assert analysis_fetch.fetch_analysis_main(args) == 0
    assert seen["cycle"] == VALID
    assert seen["out"] == Path("data/era5-analysis/2019052000")
    printed = capsys.readouterr().out
    assert f"copy {era5_analysis.ERA5_GLOBAL_QUICKSTART_CONFIG}" in printed
    assert "gpuwm-global run CONFIG" in printed
    assert f'analysis_mapping = "{era5_analysis.ERA5_GLOBAL_MAPPING_ID}"' in printed
    assert 'analysis_grib = "data/era5-analysis/2019052000/era5-combined.nc"' in printed


def test_an_hour_before_era5_is_refused_before_anything_is_fetched(tmp_path):
    pytest.importorskip("gpuwm.era5_arco")
    with pytest.raises(ValueError, match="ERA5 begins"):
        era5_analysis.fetch_era5_analysis(tmp_path, dt.datetime(1939, 12, 31, 23))
    assert not any(tmp_path.iterdir())


# ------------------------------------------------- the shipped experiment

def test_the_quickstart_with_the_printed_initial_table_is_an_era5_config(tmp_path):
    """The epilogue's [initial] table, pasted over the quickstart's, is a
    configuration that loads and names the ERA5 mapping, and nothing else
    about the quickstart moves."""

    base = _shipped_configs() / f"{era5_analysis.ERA5_GLOBAL_QUICKSTART_CONFIG}.toml"
    analysis = Path("data/era5-analysis/2019052000/era5-combined.nc")
    lines = "\n".join(era5_analysis.era5_fetch_epilogue(analysis=analysis))
    table = "\n".join(line[len("  #   "):] for line in lines.splitlines()
                      if line.startswith("  #   "))
    assert table.startswith("[initial]\n")
    text = base.read_text(encoding="utf-8")
    head, rest = text.split("[initial]\n", 1)
    replaced = head + table + "\n" + rest[rest.index("\n\n") + 1:]
    assert replaced.count("[initial]") == 1
    path = tmp_path / "era5_t255.toml"
    path.write_text(replaced, encoding="utf-8")
    era5 = load_config(path)
    gdas = load_config(base)
    assert era5.initial_mode == "analysis"
    assert era5.analysis_mapping == era5_analysis.ERA5_GLOBAL_MAPPING_ID
    assert resolve_analysis_mapping(era5.analysis_mapping).name == ERA5_MAPPING_FILE
    assert Path(era5.analysis_grib) == analysis
    assert gdas.analysis_mapping == "gdas-global"
    for field in ("truncation", "dealias_factor", "dt_s", "duration_s",
                  "output_interval_s", "integrator", "a_half_pa", "b_half",
                  "physics_mode", "backend", "precision", "diffusion_order",
                  "diffusion_efold_s", "surface_water_kg_m2"):
        assert getattr(era5, field) == getattr(gdas, field), field
