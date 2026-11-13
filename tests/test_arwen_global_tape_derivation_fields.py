"""The render tape carries what a wrfout post-processor derives its fields from.

THE BREAKAGE THESE GATES PREVENT.  A reader of WRF history files asks for
eight variables by name beside the state: F and E, the two map factors,
LU_INDEX, IVGTYP and XLAND, and W.  A tape without them is not refused; it
is read as something else.  Without the map factors a lat-lon grid is
differentiated as if a degree of longitude were as long at 80 degrees as at
the equator, so vorticity is wrong by a factor that grows to the pole and
nothing says so.  Without the land-use planes every column is land.  And a
W that is only the flux through the model's coordinate surfaces is read as
the wind relative to the ground, which over a slope it is not: uniform flow
up a mountain has no flux through a surface that follows the ground.

So each of the eight is pinned here against an answer known without the
exporter: the Coriolis terms and the map factors from the latitude, the
land-use planes from the nearest model column found by distance, W from
steady flow over a prescribed mountain, where the answer at the ground is
the wind times the slope, and against the parent frame of the same
checkpoint, which keeps the flux alone.  The variables a 0.1.4 tape
carried are listed by name, so one that goes missing fails here too.

None of them needs a device.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_netcdf_writer, requires_render_kernels  # noqa: E402

from arwen_global import mapped_parent_export as mpe
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root

START = "2026-08-30_18:00:00"
EIGHT = ("F", "E", "MAPFAC_MX", "MAPFAC_MY", "LU_INDEX", "IVGTYP", "XLAND", "W")
#: Every variable a 0.1.4 tape of this configuration carried.
TAPE_014 = frozenset({
    "Times", "XTIME", "ITIMESTEP", "XLAT", "XLONG", "T", "P", "PB", "PH", "PHB",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "U", "V",
    "T2", "Q2", "U10", "V10", "PSFC", "TSK", "HGT", "LANDMASK", "SINALPHA",
    "COSALPHA", "RAINC", "RAINNC", "SNOWNC", "GRAUPELNC", "TSLB", "SMOIS",
    "SEAICE", "ICEDEPTH", "SNOW", "SNOWH", "SNOWC",
})


# ---------------------------------------------------------------------------
# the planes on their own, where the answer is arithmetic
# ---------------------------------------------------------------------------

def test_the_coriolis_terms_and_the_map_factors_have_known_answers():
    from arwen_global.wrfout_export import grid_metric_planes

    omega = 7.292e-5
    xlat = np.array([[-60.0, -30.0], [0.0, 30.0], [60.0, 89.84375]])
    planes = grid_metric_planes(
        xlat, rotation_rate_s=omega, row_spacing_deg=0.3125, column_spacing_deg=0.3125)
    assert all(plane.dtype == np.float32 and plane.shape == xlat.shape
               for plane in planes.values())
    # sin 30 = cos 60 = 1/2: F there is Omega, and so is E at 60.
    np.testing.assert_allclose(planes["F"][1, 1], omega, rtol=1.0e-7)
    np.testing.assert_allclose(planes["F"][0, 1], -omega, rtol=1.0e-7)
    np.testing.assert_allclose(planes["E"][2, 0], omega, rtol=1.0e-7)
    assert planes["F"][1, 0] == 0.0
    np.testing.assert_allclose(planes["E"][1, 0], 2.0 * omega, rtol=1.0e-7)
    # 1 / cos 60 = 2; one at the equator; 366.7 on the last row of a
    # 576-row grid, where a degree of longitude is that much shorter.
    np.testing.assert_allclose(planes["MAPFAC_MX"][2, 0], 2.0, rtol=1.0e-7)
    assert planes["MAPFAC_MX"][1, 0] == 1.0
    np.testing.assert_allclose(planes["MAPFAC_MX"][2, 1], 366.69345, rtol=1.0e-6)
    np.testing.assert_array_equal(planes["MAPFAC_MY"], np.float32(1.0))
    # Rows 15 degrees apart under columns 10 degrees apart: the tape's DY is
    # its DX, two thirds of the distance between rows.
    oblong = grid_metric_planes(
        xlat, rotation_rate_s=omega, row_spacing_deg=15.0, column_spacing_deg=10.0)
    np.testing.assert_array_equal(oblong["MAPFAC_MY"], np.float32(10.0 / 15.0))
    # The rotation rate is the model's, whatever it is: a planet that does
    # not turn has no Coriolis term on its tape.
    still = grid_metric_planes(
        xlat, rotation_rate_s=0.0, row_spacing_deg=1.0, column_spacing_deg=1.0)
    np.testing.assert_array_equal(still["F"], 0.0)
    np.testing.assert_array_equal(still["E"], 0.0)


def test_a_category_comes_from_the_nearest_column_and_is_never_blended():
    """Four Gaussian rows and four longitudes, every column its own
    category.  A tape point takes the category of the column nearest to it,
    the last row beyond either end of the Gaussian latitudes, the nearest
    longitude around the ring; no point takes a value no column has."""
    from arwen_global.wrfout_export import _nearest_column, land_use_planes

    grid = SimpleNamespace(
        latitude_deg=np.array([-60.0, -20.0, 20.0, 60.0]),
        longitude_deg=np.array([0.0, 90.0, 180.0, 270.0]))
    category = np.arange(1.0, 17.0).reshape(4, 4)        # row r, column c: 4 r + c + 1
    # rows: beyond the south end, just south of a midpoint, just north of
    # it, beyond the north end.  columns (unsigned, as the exporter samples):
    # 350 wraps to the column at 0, 44 is nearer 0, 46 nearer 90, 200 nearer 180.
    lat = np.array([-85.0, -1.0, 1.0, 85.0])
    lon = np.array([350.0, 44.0, 46.0, 200.0])
    pick = _nearest_column(grid, lat, lon)
    np.testing.assert_array_equal(pick(category), np.array([
        [1.0, 1.0, 2.0, 3.0],
        [5.0, 5.0, 6.0, 7.0],
        [9.0, 9.0, 10.0, 11.0],
        [13.0, 13.0, 14.0, 15.0],
    ]))
    # A leading axis rides along (a soil stack, a level stack).
    assert pick(np.stack([category, category + 100.0])).shape == (2, 4, 4)
    # Exactly midway between two rows the southern one is taken, and
    # exactly midway between two columns the eastern one.
    midway = _nearest_column(grid, np.array([0.0]), np.array([45.0, 315.0]))
    np.testing.assert_array_equal(midway(category), np.array([[6.0, 5.0]]))

    # The land-use planes.  Row 1 is open water, row 2 land; the first
    # column of row 1 is under ice and carries the ice category, as the
    # statics leave it.  Stored as floats that are not quite integers, as a
    # float32 carrier can be.
    land_fraction = np.array([[0.0] * 4, [0.0] * 4, [1.0] * 4, [0.7] * 4])
    sea_ice = np.zeros((4, 4))
    sea_ice[1, 0] = 0.8
    sea_ice[1, 1] = 0.3                                  # below one half: still water
    stored = np.where(land_fraction > 0.5, 7.0, 17.0)
    stored[1, 0] = 15.0
    surface = SimpleNamespace(
        landuse_category=stored + 1.0e-4, land_fraction=land_fraction,
        sea_ice_fraction=sea_ice)
    planes = land_use_planes(surface, pick)
    assert planes["LU_INDEX"].dtype == np.float32
    assert planes["IVGTYP"].dtype == np.int32
    assert planes["XLAND"].dtype == np.float32
    np.testing.assert_array_equal(planes["LU_INDEX"], np.array([
        [17.0] * 4, [15.0, 15.0, 17.0, 17.0], [7.0] * 4, [7.0] * 4]))
    np.testing.assert_array_equal(planes["IVGTYP"], planes["LU_INDEX"].astype(np.int32))
    # 2 on open water, 1 on land and on the ice column.
    np.testing.assert_array_equal(planes["XLAND"], np.array([
        [2.0] * 4, [1.0, 1.0, 2.0, 2.0], [1.0] * 4, [1.0] * 4]))


def test_the_land_use_table_is_the_states_own_or_absent():
    from arwen_global.statics import SURFACE_STATICS_METADATA_KEY, SYNTHETIC_CONVENTION
    from arwen_global.wrfout_export import LANDUSE_TABLE_ATTRS, landuse_table

    def bundle(metadata):
        return SimpleNamespace(physics_state=SimpleNamespace(metadata=metadata))

    row = SYNTHETIC_CONVENTION.as_metadata("synthetic")
    # ISLAKE is -1 whatever the convention's lake class (21 here): the
    # statics build folds that class into water, no column carries it, and
    # -1 is what WRF writes for a field with no separate lake class.
    assert row["lake_category"] == 21
    assert landuse_table(bundle({SURFACE_STATICS_METADATA_KEY: row})) == {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": -1,
        "ISICE": 15, "ISURBAN": 13}
    # Another land-use set is written as itself, not as the one above.
    usgs = {**row, "landuse_dataset": "USGS", "water_category": 16, "ice_category": 24}
    table = landuse_table(bundle({SURFACE_STATICS_METADATA_KEY: usgs}))
    assert (table["MMINLU"], table["ISWATER"], table["ISICE"]) == ("USGS", 16, 24)
    assert table["ISLAKE"] == -1
    # A row that names no lake class at all still gets its table: the
    # tape's ISLAKE does not come from it.
    no_lake = {key: value for key, value in row.items() if key != "lake_category"}
    assert landuse_table(bundle({SURFACE_STATICS_METADATA_KEY: no_lake}))["ISLAKE"] == -1
    # A state that names no convention gets no table: nothing is guessed.
    assert landuse_table(bundle({})) is None
    partial = {key: value for key, value in row.items() if key != "water_category"}
    assert landuse_table(bundle({SURFACE_STATICS_METADATA_KEY: partial})) is None
    assert [attr for attr, _member in LANDUSE_TABLE_ATTRS] == [
        "MMINLU", "ISWATER", "ISLAKE", "ISICE", "ISURBAN"]


def test_the_tape_and_the_parent_frame_each_say_what_their_own_w_is():
    """Two arrays, two descriptions.  A parent frame's W is the flux through
    the coordinate surfaces alone, which is what the regional engine was fed
    and measured with, and its words stay as they were.  The tape's W adds
    the wind's climb along the surfaces, and says that and what it still
    leaves out."""
    from arwen_global import wrfout_export as we

    (frame_text,) = [text for name, _units, text in mpe.INTERFACE_FIELDS if name == "W"]
    assert "through the hybrid coordinate surfaces" in frame_text
    assert "not the velocity relative to the ground" in frame_text
    assert we.TAPE_W_DESCRIPTION != frame_text
    assert "relative to the ground" in we.TAPE_W_DESCRIPTION
    assert "not the velocity relative to the ground" not in we.TAPE_W_DESCRIPTION
    assert "is left out" in we.TAPE_W_DESCRIPTION
    assert we.TAPE_VARIABLE_ATTRS == {"W": {"description": we.TAPE_W_DESCRIPTION}}
    # The tape starts from the frames' own diagnosis and does not replace it.
    assert we.diagnose_interface_w is mpe.diagnose_interface_w
    policy = mpe.parent_provenance(
        run_name="r", config_hash="0", start="s", grid="g", nlev=3)["vertical_velocity_policy"]
    assert policy == mpe.VERTICAL_VELOCITY_POLICY
    assert "B (dps/dt + V . grad ps)" in policy
    for phrase in ("relative to the ground", "the regional parent frames' W",
                   "V . grad z at constant eta", "is left out",
                   "zero at the ground"):
        assert phrase in we.TAPE_VERTICAL_VELOCITY_POLICY, phrase
    assert [name for name, _units, _text in we.DERIVATION_FIELDS] == list(EIGHT)


def test_w_over_a_prescribed_mountain_is_the_wind_times_the_slope():
    """The known answer.  The atmosphere turns as a solid body, each layer
    at its own rate, u = U_k cos(latitude), over a mountain of height
    H cos(latitude) cos(longitude), under a surface pressure that is the
    same everywhere.  The flow has no divergence and the layers have the
    same mass everywhere, so nothing crosses a coordinate surface: the
    parent frames' W is zero on every interface.  The air still climbs the
    mountain.  At the ground

        w = u dh/dx = -U H cos(latitude) sin(longitude) / a

    with the lowest layer's U, and that is all of it there, because the
    ground does not move.  Each layer is warmer over the mountain by an
    amount A_k cos(latitude) cos(longitude), so the surfaces above tilt
    more than the ground does: an interface stands higher than the ground
    by the hydrostatic thickness underneath it, R Tv ln(p below / p above)
    summed over the layers below, and its slope is known in closed form
    from the layers' warm anomalies.  W there is the mean wind of the two
    layers it separates times that slope."""
    import math
    from dataclasses import replace

    from arwen_global import wrfout_export as we
    from arwen_global.constants import GRAVITY_M_S2
    from arwen_global.lnps_reference import (
        LNPS_REFERENCE, lnps_from_anomaly_coefficients)
    from arwen_global.runner import build_model_and_cold_state, build_transform
    from arwen_global.spectral.vector import VorticityDivergenceOperator

    cfg = load_config(config_root() / "arwen_global_moist_smoke.toml")
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    grid = transform.grid
    lat = np.deg2rad(np.asarray(grid.latitude_deg, dtype=np.float64))[:, None]
    lon = np.deg2rad(np.asarray(grid.longitude_deg, dtype=np.float64))[None, :]
    plane = np.cos(lat) * np.cos(lon)
    nlev = cold.atmosphere.theta.shape[0]
    shape = (nlev, *plane.shape)
    height_m = 1500.0
    turn = np.linspace(35.0, 8.0, nlev)                  # U_k, top to surface
    base = np.linspace(430.0, 290.0, nlev)               # theta, K
    warm = np.linspace(6.0, 2.0, nlev)                   # its anomaly, K

    def state(turning, warming, mountain_m):
        u = turning[:, None, None] * np.cos(lat)[None] * np.ones(shape)
        vorticity, divergence = VorticityDivergenceOperator(
            transform).vordiv_from_wind(u, np.zeros(shape))
        theta = base[:, None, None] + warming[:, None, None] * plane[None]
        zeros = np.zeros(shape)
        model.surface_geopotential = np.asarray(GRAVITY_M_S2 * mountain_m * plane)
        return replace(
            cold.atmosphere, vorticity=vorticity, divergence=divergence,
            theta=transform.forward(theta),
            log_surface_pressure=lnps_from_anomaly_coefficients(
                transform, transform.forward(
                    np.full(plane.shape, math.log(1.0e5) - LNPS_REFERENCE))),
            qv=transform.forward(zeros), qc=zeros.copy(), qr=zeros.copy(),
            qi=zeros.copy(), qs=zeros.copy(), qg=zeros.copy())

    atmosphere = state(turn, warm, height_m)
    through = np.asarray(mpe.diagnose_interface_w(model, transform, atmosphere))
    w = np.asarray(we.diagnose_ground_relative_w(model, transform, atmosphere))
    assert w.shape == (nlev + 1, *plane.shape)
    # Nothing crosses a coordinate surface: the frames' W has no climb in it.
    np.testing.assert_allclose(through, 0.0, atol=1.0e-12)

    # The ground: the lowest layer's wind times the terrain slope.
    radius = float(grid.radius_m)
    pattern = -np.cos(lat) * np.sin(lon) / radius        # w per (m/s of U) per m
    ground = turn[-1] * height_m * pattern
    assert np.abs(ground).max() > 1.0e-3                 # millimetres a second, not zero
    np.testing.assert_allclose(w[-1], ground, rtol=1.0e-9, atol=1.0e-12)

    # Above it: the tilt of each interface from the layers underneath,
    # summed here one layer at a time from the state's own columns.
    native = model.grid_state(atmosphere, only=("virtual_temperature", "ln_ratio"))
    virtual = np.asarray(native["virtual_temperature"], dtype=np.float64)
    ln_ratio = np.asarray(native["ln_ratio"], dtype=np.float64)
    j, i0, i1 = plane.shape[0] // 2, 0, plane.shape[1] // 2
    tilt_m = height_m                                    # amplitude of z on cos cos
    for k in range(nlev, -1, -1):
        if k < nlev:
            thickness = model.gas_constant * virtual[k] * ln_ratio[k] / GRAVITY_M_S2
            amplitude = (thickness[j, i0] - thickness[j, i1]) / (plane[j, i0] - plane[j, i1])
            # The layer's thickness is a constant plus that amplitude on
            # the mountain's own pattern, and nothing else.
            np.testing.assert_allclose(
                thickness - thickness.mean(), amplitude * (plane - plane.mean()),
                atol=1.0e-9 * abs(amplitude) + 1.0e-9)
            assert amplitude > 0.0                       # warm over the mountain: thicker
            tilt_m += amplitude
        wind = (turn[0] if k == 0 else turn[-1] if k == nlev
                else 0.5 * (turn[k - 1] + turn[k]))
        np.testing.assert_allclose(
            w[k], wind * tilt_m * pattern, rtol=1.0e-9, atol=1.0e-12, err_msg=f"interface {k}")
    assert tilt_m > 1.2 * height_m                       # the top tilts more than the ground

    # Level ground under still layers of the same warmth everywhere: the
    # wind blows along surfaces that do not tilt, and W is zero.
    level = state(turn, np.zeros(nlev), 0.0)
    np.testing.assert_allclose(
        np.asarray(we.diagnose_ground_relative_w(model, transform, level)),
        0.0, atol=1.0e-12)
    # And air at rest climbs nothing, whatever is under it.
    rest = state(np.zeros(nlev), warm, height_m)
    np.testing.assert_allclose(
        np.asarray(we.diagnose_ground_relative_w(model, transform, rest)),
        0.0, atol=1.0e-12)


# ---------------------------------------------------------------------------
# a real (tiny, CPU) run, exported
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """The moist smoke configuration run once and exported as tapes on an
    18 x 36 grid, as tapes on a 12 x 36 grid, and as parent frames."""
    from arwen_global.runner import run
    from arwen_global.wrfout_export import export_wrfout

    root = tmp_path_factory.mktemp("derivation-fields")
    cfg = load_config(config_root() / "arwen_global_moist_smoke.toml")
    run(cfg, root / "run")
    checkpoints = sorted((root / "run").glob("arwen_global_step*.npz"))
    assert len(checkpoints) >= 3
    tapes = export_wrfout(
        cfg, checkpoints, root / "tapes", nlat=18, nlon=36, start_date=START)
    oblong = export_wrfout(
        cfg, checkpoints[-1:], root / "oblong", nlat=12, nlon=36, start_date=START)
    parent = mpe.export_mapped_parent(
        cfg, checkpoints, root / "parent", nlat=18, nlon=36, start_date=START)
    return SimpleNamespace(
        cfg=cfg, checkpoints=checkpoints, root=root, tapes=tapes,
        oblong=oblong, parent=parent)


def _model_state(cfg, checkpoint):
    """The model, its transform and one checkpoint's state, as the exporter
    builds them."""
    from arwen_global.checkpoint import read_checkpoint, state_from_checkpoint
    from arwen_global.runner import (
        adopt_checkpoint_clock, build_model_and_cold_state, build_transform)

    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(cfg, transform)
    metadata, arrays = read_checkpoint(
        checkpoint, expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator)
    bundle = state_from_checkpoint(metadata, arrays, transform.backend)
    adopt_checkpoint_clock(model, bundle, f"checkpoint {checkpoint}")
    model.enforce(bundle)
    return model, transform, bundle


def _nearest_row_and_column(grid, xlat, xlong):
    """Row and column of the Gaussian column nearest each tape point: the
    row nearest in latitude and the column nearest in longitude around the
    ring, each found by measuring the distance to every one, and whether
    each is the only nearest.  Independent of the exporter's index
    arithmetic."""
    glat = np.asarray(grid.latitude_deg, dtype=np.float64)
    glon = np.asarray(grid.longitude_deg, dtype=np.float64)
    apart_lat = np.abs(np.asarray(xlat, dtype=np.float64)[..., None] - glat)
    turn = np.mod(np.asarray(xlong, dtype=np.float64)[..., None] - glon, 360.0)
    apart_lon = np.minimum(turn, 360.0 - turn)
    unique = np.ones(np.shape(xlat), dtype=bool)
    for apart in (apart_lat, apart_lon):
        ordered = np.sort(apart, axis=-1)
        unique &= ordered[..., 1] - ordered[..., 0] > 1.0e-9
    return np.argmin(apart_lat, axis=-1), np.argmin(apart_lon, axis=-1), unique


@requires_netcdf_writer
@requires_render_kernels
def test_every_tape_carries_the_eight_fields_beside_everything_it_carried(smoke):
    from netCDF4 import Dataset

    from arwen_global import wrfout_export as we

    expected = {
        "F": ("south_north", "west_east"), "E": ("south_north", "west_east"),
        "MAPFAC_MX": ("south_north", "west_east"),
        "MAPFAC_MY": ("south_north", "west_east"),
        "LU_INDEX": ("south_north", "west_east"),
        "IVGTYP": ("south_north", "west_east"),
        "XLAND": ("south_north", "west_east"),
        "W": ("bottom_top_stag", "south_north", "west_east"),
    }
    assert len(smoke.tapes) == len(smoke.checkpoints)
    for tape in smoke.tapes:
        with Dataset(tape) as ds:
            # Nothing a 0.1.4 tape carried is gone, and the eight and SST
            # are all that is new.
            assert set(ds.variables) == TAPE_014 | set(EIGHT) | {"SST"}
            for name, units, _text in we.DERIVATION_FIELDS:
                variable = ds.variables[name]
                assert variable.dimensions == ("Time", *expected[name]), name
                assert variable.getncattr("units") == units, name
                assert variable.dtype == (np.int32 if name == "IVGTYP" else np.float32), name
                assert np.all(np.isfinite(np.asarray(variable[0]))), name
            assert ds.variables["W"].getncattr("stagger") == "Z"
            # One plane per layer interface, the ground and the top included.
            assert ds.variables["W"].shape[1] == len(smoke.cfg.a_half_pa)
    receipt = json.loads(
        (smoke.root / "tapes" / we.EXPORT_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["derivation_fields"] == list(EIGHT)
    assert receipt["vertical_velocity"] == we.TAPE_VERTICAL_VELOCITY_POLICY


@requires_netcdf_writer
@requires_render_kernels
def test_the_tapes_coriolis_terms_and_map_factors_are_its_latitudes(smoke):
    from netCDF4 import Dataset

    from arwen_global.runner import build_model_and_cold_state, build_transform

    model, _cold = build_model_and_cold_state(smoke.cfg, build_transform(smoke.cfg))
    omega = float(model.rotation_rate_s)
    assert omega == 7.292e-5                             # the rate the model integrates with
    first = None
    for tape in smoke.tapes:
        with Dataset(tape) as ds:
            planes = {name: np.asarray(ds[name][0]) for name in
                      ("XLAT", "F", "E", "MAPFAC_MX", "MAPFAC_MY")}
        lat = np.deg2rad(planes["XLAT"].astype(np.float64))
        # The tape's rows are -85, -75, ..., 85.
        np.testing.assert_allclose(
            planes["XLAT"][:, 0], -85.0 + 10.0 * np.arange(18), rtol=0, atol=1.0e-5)
        np.testing.assert_allclose(planes["F"], 2.0 * omega * np.sin(lat), rtol=2.0e-7)
        np.testing.assert_allclose(planes["E"], 2.0 * omega * np.cos(lat), rtol=2.0e-7)
        np.testing.assert_allclose(planes["MAPFAC_MX"] * np.cos(lat), 1.0, rtol=2.0e-7)
        np.testing.assert_array_equal(planes["MAPFAC_MY"], np.float32(1.0))
        # Antisymmetric and symmetric about the equator, constant along a row.
        np.testing.assert_array_equal(planes["F"], -planes["F"][::-1])
        np.testing.assert_array_equal(planes["E"], planes["E"][::-1])
        assert np.all(planes["F"] == planes["F"][:, :1])
        # The grid's, not the state's: the same in every tape.
        if first is None:
            first = planes
        for name, plane in planes.items():
            np.testing.assert_array_equal(plane, first[name], err_msg=name)


@requires_netcdf_writer
@requires_render_kernels
def test_the_map_factor_in_y_is_the_declared_spacing_over_the_real_one(smoke):
    """Twelve rows under thirty-six columns: rows 15 degrees apart, columns
    10.  The tape declares DY = DX, the 10 degrees at the equator, so the
    y map factor is two thirds and DY over it is the distance between rows."""
    from netCDF4 import Dataset

    (tape,) = smoke.oblong
    with Dataset(tape) as ds:
        dx, dy = float(ds.getncattr("DX")), float(ds.getncattr("DY"))
        my = np.asarray(ds["MAPFAC_MY"][0])
        mx = np.asarray(ds["MAPFAC_MX"][0])
        lat = np.asarray(ds["XLAT"][0], dtype=np.float64)
    assert dx == dy
    np.testing.assert_array_equal(my, np.float32(10.0 / 15.0))
    radius = dx / np.deg2rad(10.0)
    np.testing.assert_allclose(dy / my, radius * np.deg2rad(15.0), rtol=1.0e-6)
    np.testing.assert_allclose(
        dx / mx, radius * np.deg2rad(10.0) * np.cos(np.deg2rad(lat)), rtol=1.0e-6)


@requires_netcdf_writer
@requires_render_kernels
def test_the_tapes_land_use_is_the_nearest_columns_and_xland_agrees_with_it(smoke):
    from netCDF4 import Dataset

    from arwen_global.statics import SURFACE_STATICS_METADATA_KEY

    for checkpoint, tape in zip(smoke.checkpoints, smoke.tapes):
        _model, transform, bundle = _model_state(smoke.cfg, checkpoint)
        category = np.asarray(transform.backend.to_numpy(bundle.surface.landuse_category))
        fraction = np.asarray(transform.backend.to_numpy(bundle.surface.land_fraction))
        # The synthetic planet of this configuration has both: one land
        # class, and water on the polar rows and three columns beside them.
        assert set(np.unique(category)) == {7.0, 17.0}
        row = bundle.physics_state.metadata[SURFACE_STATICS_METADATA_KEY]
        with Dataset(tape) as ds:
            xlat, xlong = np.asarray(ds["XLAT"][0]), np.asarray(ds["XLONG"][0])
            lu = np.asarray(ds["LU_INDEX"][0])
            ivgtyp = np.asarray(ds["IVGTYP"][0])
            xland = np.asarray(ds["XLAND"][0])
            table = {name: ds.getncattr(name) for name in
                     ("MMINLU", "ISWATER", "ISLAKE", "ISICE", "ISURBAN")}
        rows, columns, unique = _nearest_row_and_column(transform.grid, xlat, xlong)
        assert unique.all()                              # no tape point sits on a tie
        np.testing.assert_array_equal(lu, category[rows, columns])
        assert set(np.unique(lu)) == {7.0, 17.0}         # both classes reach the tape
        np.testing.assert_array_equal(ivgtyp, lu.astype(np.int32))
        # The table the categories index travels with them.  ISLAKE is -1,
        # WRF's value for a field with no separate lake class: the state
        # names one (21) and no column of it, or of the tape, carries it.
        assert table == {
            "MMINLU": row["landuse_dataset"], "ISWATER": row["water_category"],
            "ISLAKE": -1, "ISICE": row["ice_category"],
            "ISURBAN": row["urban_category"]}
        assert table["MMINLU"] == "MODIFIED_IGBP_MODIS_NOAH" and table["ISWATER"] == 17
        assert row["lake_category"] == 21 and not np.any(lu == row["lake_category"])
        # XLAND is 2 exactly where the category is the table's water class,
        # and exactly where the nearest column's land fraction is at most
        # one half (this planet has no ice).
        np.testing.assert_array_equal(xland, np.where(lu == table["ISWATER"], 2.0, 1.0))
        np.testing.assert_array_equal(
            xland, np.where(fraction[rows, columns] <= 0.5, 2.0, 1.0))
        assert 0 < int((xland == 2.0).sum()) < xland.size


@requires_netcdf_writer
@requires_render_kernels
def test_the_tapes_w_is_the_frames_flux_plus_the_climb_and_the_frame_keeps_the_flux(smoke):
    from netCDF4 import Dataset

    from render_kernel_oracle import gaussian_to_regular

    from arwen_global import wrfout_export as we
    from arwen_global.constants import GRAVITY_M_S2

    largest = []
    for checkpoint, tape, row in zip(smoke.checkpoints, smoke.tapes, smoke.parent["frames"]):
        with Dataset(tape) as ds, Dataset(row["frame"]) as frame:
            variable = ds.variables["W"]
            w = np.asarray(variable[0])
            flux = np.asarray(frame["W"][0])
            frame_text = frame.variables["W"].getncattr("long_name")
            assert variable.getncattr("description") == we.TAPE_W_DESCRIPTION
            assert variable.getncattr("units") == "m s-1"
            assert ds.getncattr(we.VERTICAL_VELOCITY_ATTR) == we.TAPE_VERTICAL_VELOCITY_POLICY
            xlat = np.asarray(ds["XLAT"][0, :, 0], dtype=np.float64)
            xlong = np.asarray(ds["XLONG"][0, 0, :], dtype=np.float64)
        model, transform, bundle = _model_state(smoke.cfg, checkpoint)
        to_host = transform.backend.to_numpy
        regrid = gaussian_to_regular(
            transform.grid.latitude_deg, transform.grid.longitude_deg,
            xlat, np.mod(xlong, 360.0))

        # THE PARENT FRAME IS UNTOUCHED: still the flux through the
        # coordinate surfaces alone, zero on the ground and at the model
        # top whatever the flow, under its own words.
        through = np.asarray(to_host(
            mpe.diagnose_interface_w(model, transform, bundle.atmosphere)))
        np.testing.assert_allclose(
            flux, regrid(through)[::-1].astype(np.float32), rtol=1.0e-6, atol=1.0e-12)
        np.testing.assert_array_equal(flux[0], 0.0)
        np.testing.assert_array_equal(flux[-1], 0.0)
        assert "not the velocity relative to the ground" in frame_text
        assert frame_text != we.TAPE_W_DESCRIPTION

        # The tape is the model's ground-relative diagnosis, sampled
        # bilinearly and written bottom to top: nothing else happens to it.
        native = np.asarray(to_host(
            we.diagnose_ground_relative_w(model, transform, bundle.atmosphere)))
        np.testing.assert_allclose(
            w, regrid(native)[::-1].astype(np.float32), rtol=1.0e-6, atol=1.0e-12)

        # At the ground that is the lowest layer's wind times the terrain
        # slope, taken here from the state without the exporter's function:
        # this planet has a 25 m mountain and a 2 m/s wind over it.
        state = model.grid_state(bundle.atmosphere, only=("u", "v"))
        east, north = transform.gradient(transform.forward(model.surface_geopotential))
        climb = np.asarray(to_host(
            state["u"][-1] * east + state["v"][-1] * north)) / GRAVITY_M_S2
        assert np.abs(climb).max() > 1.0e-6
        np.testing.assert_allclose(
            w[0], regrid(climb).astype(np.float32), rtol=1.0e-5, atol=1.0e-10)
        # So the tape and the frame differ, by the climb and by nothing at
        # all where the ground is level and the surfaces above it are too.
        assert np.abs(w - flux).max() > 1.0e-6
        largest.append(float(np.abs(flux).max()))
    # The flow starts almost at rest and the flux through the surfaces
    # grows from it.
    assert largest[-1] > 1.0e-6 and largest[-1] > largest[0]


@requires_netcdf_writer
@requires_render_kernels
def test_a_window_carries_the_whole_grids_eight_fields_cut(smoke, tmp_path):
    from netCDF4 import Dataset

    from arwen_global.wrfout_export import export_wrfout

    (window,) = export_wrfout(
        smoke.cfg, smoke.checkpoints[-1:], tmp_path / "window", nlat=18, nlon=36,
        start_date=START, bbox=(-45.0, 60.0, -120.0, 40.0))
    with Dataset(window) as cut, Dataset(smoke.tapes[-1]) as whole:
        lat = np.asarray(whole["XLAT"][0, :, 0])
        lon = np.asarray(whole["XLONG"][0, 0, :])
        rows = np.where((lat >= -45.0) & (lat <= 60.0))[0]
        columns = np.where((lon >= -120.0) & (lon <= 40.0))[0]
        assert rows.size == 11 and columns.size == 17
        for name in EIGHT:
            np.testing.assert_array_equal(
                np.asarray(cut[name][0]),
                np.asarray(whole[name][0])[..., rows[:, None], columns[None, :]],
                err_msg=name)
        assert cut.variables["W"].getncattr("description") == \
            whole.variables["W"].getncattr("description")


@requires_netcdf_writer
@requires_render_kernels
def test_the_netcdf4_writer_carries_the_same_fields_and_the_same_label(
        smoke, tmp_path, monkeypatch):
    """The engine has two tape writers and an operator may select the
    second.  W's description is put on at the same moment on both, so the
    selected container does not decide what the tape says W is."""
    from netCDF4 import Dataset

    from gpuwm.io.wrfout import WRFOUT_WRITER_ENV, resolve_wrfout_engine

    from arwen_global import wrfout_export as we

    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")
    assert resolve_wrfout_engine(None) == "python"
    (tape,) = we.export_wrfout(
        smoke.cfg, smoke.checkpoints[-1:], tmp_path / "netcdf4", nlat=18, nlon=36,
        start_date=START)
    with Dataset(tape) as other, Dataset(smoke.tapes[-1]) as default:
        assert other.variables["W"].getncattr("description") == we.TAPE_W_DESCRIPTION
        for name in EIGHT:
            np.testing.assert_array_equal(
                np.asarray(other[name][0]), np.asarray(default[name][0]), err_msg=name)
            assert other.variables[name].dtype == default.variables[name].dtype, name


@requires_netcdf_writer
@requires_render_kernels
def test_a_tape_whose_w_cannot_be_labelled_is_not_written(smoke, tmp_path, monkeypatch):
    """A variable's attributes can be written only before the first frame,
    so the export declares every variable as it opens the tape and labels W
    then.  An engine writer that does not declare them would publish W as
    WRF's "z-wind component" with no error anywhere; here it is one that
    drops the declaration, and the tape is refused before a frame is
    written."""
    from gpuwm.io.wrfout import WrfoutWriter

    from arwen_global import wrfout_export as we

    class Undeclaring(WrfoutWriter):
        def __init__(self, path, *, field_schema=None, **kwargs):
            super().__init__(path, **kwargs)

    monkeypatch.setattr(we, "WrfoutWriter", Undeclaring)
    outdir = tmp_path / "unlabelled"
    with pytest.raises(RuntimeError, match="not written rather than written mislabelled"):
        we.export_wrfout(
            smoke.cfg, smoke.checkpoints[-1:], outdir, nlat=18, nlon=36,
            start_date=START)
    assert not list(outdir.glob("wrfout_d01_*"))
    assert not (outdir / we.EXPORT_RECEIPT_NAME).exists()


def test_the_page_states_the_eight_fields_the_tape_carries():
    """The page is where a reader learns what the tape's W is and what the
    land-use planes index.  A field added to the writer and not to the
    page, or a unit changed in one place, is caught here."""
    from arwen_global import wrfout_export as we

    page = Path(__file__).resolve().parents[1] / "docs" / "ARWEN_GLOBAL.md"
    if not page.is_file():
        pytest.skip("the docs are not beside this test")
    text = page.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    for name, units, _text in we.DERIVATION_FIELDS:
        (row,) = [line for line in rows if f"`{name}`" in line.split("|")[1]]
        assert (units or "none") in row.split("|")[3], (name, units, row)
    for phrase in (
            "the air's vertical velocity relative to the ground",
            "wind's climb along the surface",
            "This part is left out",
            "`ISLAKE` is -1",
            f"`{we.VERTICAL_VELOCITY_ATTR}`",
            "`MMINLU`", "`derivation_fields`"):
        assert phrase in " ".join(text.split()), phrase
