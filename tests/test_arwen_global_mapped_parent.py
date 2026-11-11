"""Regional parent frames: the global model as a mapped NetCDF source.

THE BREAKAGE THESE GATES PREVENT.  The regional engine builds initial and
lateral boundary data from any model it can read through ``gpuwm prep
--source mapped``, and its decoder is exact about what it reads: 1-D CF
coordinates, a unit string per variable compared character for character,
winds on mass points, levels bottom to top.  A render tape meets none of the
first three, and a frame that is merely close decodes into a plausible wrong
atmosphere (theta read with another kappa, heights divided by another
gravity) rather than into a refusal.  These tests pin the file contract, the
documents that describe it, and the one check that matters most: the
regional engine's own inspector materializes the frames this door writes.

None of them needs a device.
"""
from __future__ import annotations

from calendar import timegm
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")

from arwen_global import mapped_parent_export as mpe  # noqa: E402
from arwen_global.config import load_config  # noqa: E402
from arwen_global.configs_dir import config_root  # noqa: E402
from arwen_global.constants import GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA  # noqa: E402


def _synthetic_fields(nlev=5, ny=6, nx=7):
    rng = np.random.default_rng(7)
    shape = (nlev, ny, nx)
    pressure = np.linspace(95000.0, 2000.0, nlev)[:, None, None] * np.ones(shape)
    height = np.linspace(100.0, 30000.0, nlev + 1)[:, None, None] * np.ones((nlev + 1, ny, nx))
    fields = {name: rng.uniform(0.0, 1.0e-3, shape).astype(np.float32)
              for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")}
    fields.update({
        "T": rng.uniform(-20.0, 200.0, shape), "PB": pressure,
        "U_EARTH": rng.normal(0.0, 10.0, shape), "V_EARTH": rng.normal(0.0, 10.0, shape),
        "PHB": height * GRAVITY_M_S2, "W": rng.normal(0.0, 0.1, (nlev + 1, ny, nx)),
        "PSFC": np.full((ny, nx), 100000.0), "HGT": np.full((ny, nx), 100.0),
        "TSK": np.full((ny, nx), 285.0), "T2": np.full((ny, nx), 284.0),
        "Q2": np.full((ny, nx), 5.0e-3), "U10": np.ones((ny, nx)), "V10": np.ones((ny, nx)),
        "LANDMASK": np.ones((ny, nx)), "SEAICE": np.zeros((ny, nx)),
        "SNOW": np.zeros((ny, nx)), "SNOWH": np.zeros((ny, nx)),
        "TSLB": np.full((4, ny, nx), 283.0), "SMOIS": np.full((4, ny, nx), 0.3),
    })
    lat = np.linspace(30.0, 40.0, ny)
    lon = np.linspace(-90.0, -78.0, nx)
    return lat, lon, fields


def test_the_mapping_speaks_this_models_constants_and_the_frames_unit_strings():
    """Heights are geopotential over THIS model's gravity and temperature is
    theta under THIS model's kappa; the packaged profile's values (9.80665
    and 2/7) would shift every height by 5e-5 and every temperature by up to
    a few tenths of a kelvin without a refusal anywhere."""
    mapping = mpe.parent_mapping(40, boundary_interval_s=10800)
    vertical = mapping["coordinates"]["vertical"]
    assert vertical["kind"] == "model_level"
    assert vertical["levels"] == list(range(1, 41))
    assert vertical["interface_levels"] == list(range(1, 42))
    fields = mapping["fields"]
    assert fields["interface_height"]["units"]["scale"] == 1.0 / GRAVITY_M_S2
    assert fields["potential_temperature"]["units"]["offset"] == 300.0
    (theta,) = [row for row in mapping["derivations"]
                if row["name"] == "temperature-from-theta"]
    assert theta["kappa"] == KAPPA
    assert theta["reference_pressure_pa"] == REFERENCE_PRESSURE_PA
    assert fields["eastward_wind"]["selectors"][0]["name"] == ["U_EARTH"]
    assert fields["northward_wind"]["selectors"][0]["name"] == ["V_EARTH"]
    assert mapping["target"]["boundary_interval_seconds"] == 10800
    assert mapping["target"]["accept_boundary_interval_multiples"] is True

    # Every variable the mapping selects is one the frame writes, with the
    # same unit string: the decoder compares the two as strings.
    units = {name: unit for group in (mpe.MASS_FIELDS, mpe.INTERFACE_FIELDS,
                                      mpe.SURFACE_FIELDS, mpe.SOIL_FIELDS)
             for name, unit, _text in group}
    selected = set()
    for canonical, field in fields.items():
        for selector in field.get("selectors", ()):
            (name,) = selector["name"]
            selected.add(name)
            assert field["units"]["source"] == units[name], canonical
    assert selected == set(mpe.FRAME_VARIABLES)
    # Every field the target requires is provided, directly or derived.
    for row in mapping["target"]["required_fields"]:
        assert row["name"] in fields, row["name"]
    # The number moments are not carried, and the mapping does not name them.
    assert not [name for name in fields if name.endswith("_number")]


def test_the_composition_takes_terrain_from_the_frames_and_soil_one_to_one():
    composition = mpe.parent_composition()
    terrain = composition["supplements"]["terrain_height"]
    assert terrain["data_role"] == mpe.SUPPLEMENT_ROLE
    assert terrain["provenance_role"] == mpe.PROVENANCE_ROLE
    assert terrain["require_invariant_across_time"] is True
    layers = composition["soil_layers"]
    assert [(row["top"], row["bottom"]) for row in layers["source_layers"]] == [
        (0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]
    assert layers["target_layers"] == [
        {"top": row["top"], "bottom": row["bottom"]} for row in layers["source_layers"]]


def test_a_frame_carries_the_file_contract(tmp_path):
    from netCDF4 import Dataset

    lat, lon, fields = _synthetic_fields()
    valid = datetime(2026, 11, 12, 3)
    path = mpe.write_parent_frame(
        tmp_path / mpe.frame_filename(valid), valid, lat, lon, fields,
        attrs={"START_DATE": "2026-11-11_00:00:00"})
    assert path.name == "arwen_global_parent_2026-11-12_03_00_00.nc"
    with Dataset(path) as data:
        assert data.data_model == "NETCDF4_CLASSIC"
        assert {name: len(dim) for name, dim in data.dimensions.items()} == {
            "Time": 1, "bottom_top": 5, "bottom_top_stag": 6, "lat": 6,
            "lon": 7, "soil_layers_stag": 4}
        assert data["lat"].units == "degrees_north"
        assert data["lon"].units == "degrees_east"
        assert data["Time"].units == "seconds since 1970-01-01 00:00:00"
        assert data["Time"].calendar == "standard"
        assert float(data["Time"][0]) == timegm(valid.timetuple())
        assert list(data["bottom_top"][:]) == [1, 2, 3, 4, 5]
        assert list(data["bottom_top_stag"][:]) == [1, 2, 3, 4, 5, 6]
        assert list(data["soil_layers_stag"][:]) == [0, 10, 40, 100]
        assert data["soil_layers_stag"].units == "cm"
        assert data["U_EARTH"].dimensions == ("Time", "bottom_top", "lat", "lon")
        assert data["PHB"].dimensions == ("Time", "bottom_top_stag", "lat", "lon")
        assert data["TSLB"].dimensions == ("Time", "soil_layers_stag", "lat", "lon")
        # No variable is left without a unit: the tape's empty LANDMASK and
        # SEAICE units are what the intake refused.
        for name in mpe.FRAME_VARIABLES:
            assert data[name].units, name
        assert data["LANDMASK"].units == "1" and data["SEAICE"].units == "1"
        np.testing.assert_array_equal(
            np.asarray(data["PB"][0]), fields["PB"].astype(np.float32))
    assert not list(tmp_path.glob(".*partial*"))


@pytest.mark.parametrize("damage, match", [
    (lambda lat, lon, f: f["T"].__setitem__((0, 0, 0), np.nan), "T holds a non-finite"),
    (lambda lat, lon, f: f.__setitem__("PB", f["PB"][::-1].copy()), "PB must decrease upward"),
    (lambda lat, lon, f: f.__setitem__("PHB", f["PHB"][::-1].copy()), "PHB must increase upward"),
    (lambda lat, lon, f: f.pop("W"), r"missing \['W'\]"),
    (lambda lat, lon, f: f.__setitem__("U", f["U_EARTH"]), r"undeclared variables \['U'\]"),
    (lambda lat, lon, f: f.__setitem__("PSFC", f["PSFC"][:-1]), "PSFC has shape"),
    (lambda lat, lon, f: lat.__setitem__(slice(None), lat[::-1].copy()), "south to north"),
])
def test_a_frame_the_intake_would_misread_is_refused(tmp_path, damage, match):
    """A top-down column or a NaN decodes somewhere downstream as weather."""
    lat, lon, fields = _synthetic_fields()
    damage(lat, lon, fields)
    target = tmp_path / "frame.nc"
    with pytest.raises(ValueError, match=match):
        mpe.write_parent_frame(target, datetime(2026, 11, 12), lat, lon, fields)
    assert not target.exists() and not list(tmp_path.glob(".*partial*"))


@pytest.fixture(scope="module")
def smoke_export(tmp_path_factory):
    """A real (tiny, CPU) run of this model, exported both ways."""
    from arwen_global.runner import run
    from arwen_global.wrfout_export import export_wrfout

    root = tmp_path_factory.mktemp("mapped-parent")
    cfg = load_config(config_root() / "arwen_global_moist_smoke.toml")
    run(cfg, root / "run")
    checkpoints = sorted((root / "run").glob("arwen_global_step*.npz"))
    assert len(checkpoints) >= 3
    receipt = mpe.export_mapped_parent(
        cfg, checkpoints, root / "parent", nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00")
    tapes = export_wrfout(
        cfg, checkpoints, root / "tapes", nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00")
    return cfg, checkpoints, root, receipt, tapes


def test_the_export_writes_frames_documents_and_a_receipt(smoke_export):
    from arwen_global.receipt import check_receipt

    cfg, checkpoints, root, receipt, _tapes = smoke_export
    parent = root / "parent"
    frames = sorted(parent.glob(f"{mpe.FRAME_PREFIX}*.nc"))
    assert len(frames) == len(checkpoints)
    assert frames[0].name == "arwen_global_parent_2026-08-30_18_00_00.nc"
    for name in (mpe.MAPPING_NAME, mpe.COMPOSITION_NAME, mpe.PROVENANCE_NAME,
                 mpe.RECEIPT_NAME):
        assert (parent / name).is_file(), name
    assert check_receipt(parent / mpe.RECEIPT_NAME)["self_sha256"] == receipt["self_sha256"]
    assert receipt["config_hash"] == cfg.config_hash
    assert receipt["boundary_interval_seconds"] == 20
    assert receipt["lateral_boundary_series"] == "uniform"
    assert [row["valid"] for row in receipt["frames"]][:2] == [
        "2026-08-30_18:00:00", "2026-08-30_18:00:20"]
    mapping = json.loads((parent / mpe.MAPPING_NAME).read_text(encoding="utf-8"))
    assert mapping == mpe.parent_mapping(4, boundary_interval_s=20)
    assert mapping["coordinates"]["vertical"]["levels"] == [1, 2, 3, 4]
    # The command a reader runs next names every frame twice (as input and as
    # the in-band terrain supplement) and the provenance under its role.
    command = receipt["regional_prep_command"]
    assert command[:6] == ["gpuwm", "prep", "--source", "mapped", "--source-format", "netcdf"]
    assert command.count("--input") == len(frames)
    assert command.count("--supplement") == len(frames)
    assert f"{mpe.SUPPLEMENT_ROLE}={frames[0]}" in command
    assert f"{mpe.PROVENANCE_ROLE}={parent / mpe.PROVENANCE_NAME}" in command
    # Every frame's recorded hash is the file's.
    for row in receipt["frames"]:
        assert mpe._file_sha256(Path(row["frame"])) == row["sha256"]


def test_a_frame_is_the_tapes_state_with_mass_point_winds(smoke_export):
    """The frame and the render tape are two views of one checkpoint: the
    thermodynamic columns agree exactly, the terrain is the same in every
    frame, and the frame's winds are the model's mass-point winds, which the
    tape's staggered ones are the average of."""
    from netCDF4 import Dataset

    _cfg, _checkpoints, root, receipt, tapes = smoke_export
    terrain = None
    for row, tape in zip(receipt["frames"], tapes):
        with Dataset(row["frame"]) as frame, Dataset(tape) as wrf:
            for name in ("T", "PB", "PHB", "QVAPOR", "PSFC", "HGT", "TSK", "T2", "Q2"):
                np.testing.assert_array_equal(
                    np.asarray(frame[name][0]), np.asarray(wrf[name][0]), err_msg=name)
            np.testing.assert_array_equal(
                np.asarray(frame["lat"][:], dtype=np.float32), np.asarray(wrf["XLAT"][0, :, 0]))
            np.testing.assert_array_equal(
                np.asarray(frame["lon"][:], dtype=np.float32), np.asarray(wrf["XLONG"][0, 0, :]))
            u = np.asarray(frame["U_EARTH"][0], dtype=np.float64)
            v = np.asarray(frame["V_EARTH"][0], dtype=np.float64)
            staggered_u = np.asarray(wrf["U"][0], dtype=np.float64)
            staggered_v = np.asarray(wrf["V"][0], dtype=np.float64)
            # Interior U faces are the mean of the two mass points beside them.
            np.testing.assert_allclose(
                staggered_u[..., 1:-1], 0.5 * (u[..., :-1] + u[..., 1:]), atol=1.0e-5)
            np.testing.assert_allclose(
                staggered_v[:, 1:-1, :], 0.5 * (v[:, :-1, :] + v[:, 1:, :]), atol=1.0e-5)
            assert np.all(np.diff(np.asarray(frame["PB"][0]), axis=0) < 0.0)
            w = np.asarray(frame["W"][0])
            assert w.shape[0] == u.shape[0] + 1 and np.all(np.isfinite(w))
            hgt = np.asarray(frame["HGT"][0])
            if terrain is None:
                terrain = hgt
            np.testing.assert_array_equal(hgt, terrain)
            # The lowest interface is the terrain.
            np.testing.assert_allclose(
                np.asarray(frame["PHB"][0, 0]) / GRAVITY_M_S2, hgt, atol=1.0e-2)


def test_a_window_is_the_whole_grid_cut(smoke_export, tmp_path):
    from netCDF4 import Dataset

    cfg, checkpoints, _root, receipt, _tapes = smoke_export
    window = mpe.export_mapped_parent(
        cfg, checkpoints[:1], tmp_path / "window", nlat=8, nlon=16,
        bbox=(-40.0, 40.0, -100.0, 20.0), start_date="2026-08-30_18:00:00")
    assert window["boundary_interval_seconds"] is None
    assert window["lateral_boundary_series"].startswith("not a boundary series")
    with Dataset(window["frames"][0]["frame"]) as cut, \
            Dataset(receipt["frames"][0]["frame"]) as whole:
        lat = np.asarray(whole["lat"][:])
        lon = np.asarray(whole["lon"][:])
        rows = np.where((lat >= -40.0) & (lat <= 40.0))[0]
        cols = np.where((lon >= -100.0) & (lon <= 20.0))[0]
        np.testing.assert_array_equal(np.asarray(cut["lat"][:]), lat[rows])
        np.testing.assert_array_equal(np.asarray(cut["lon"][:]), lon[cols])
        for name in ("T", "U_EARTH", "W", "PSFC", "TSLB"):
            np.testing.assert_array_equal(
                np.asarray(cut[name][0]),
                np.asarray(whole[name][0])[..., rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1],
                err_msg=name)


def _folder_state(folder: Path) -> dict[str, str]:
    return {path.name: mpe._file_sha256(path)
            for path in sorted(folder.iterdir()) if path.is_file()}


def test_an_existing_frame_is_not_replaced_without_overwrite(smoke_export):
    cfg, checkpoints, root, _receipt, _tapes = smoke_export
    before = _folder_state(root / "parent")
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        mpe.export_mapped_parent(
            cfg, checkpoints, root / "parent", nlat=8, nlon=16,
            start_date="2026-08-30_18:00:00")
    assert _folder_state(root / "parent") == before


def test_checkpoints_out_of_time_order_are_refused_before_a_frame_is_written(
        smoke_export, tmp_path):
    """The order check used to sit inside the loop: the first frame of a
    reversed series was written, the refusal came at the second, and the
    correct retry into the same folder then refused that leftover frame."""
    cfg, checkpoints, _root, _receipt, _tapes = smoke_export
    target = tmp_path / "reversed"
    with pytest.raises(ValueError, match="in time order"):
        mpe.export_mapped_parent(
            cfg, checkpoints[::-1], target, nlat=8, nlon=16,
            start_date="2026-08-30_18:00:00")
    assert not list(target.glob(f"{mpe.FRAME_PREFIX}*.nc"))
    assert not (target / mpe.RECEIPT_NAME).exists()
    retry = mpe.export_mapped_parent(
        cfg, checkpoints, target, nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00")
    assert len(retry["frames"]) == len(checkpoints)


def test_a_clash_on_the_last_frame_is_refused_before_the_first_is_written(
        smoke_export, tmp_path):
    """On the card each frame is seconds of device time; a clash on the ninth
    was found after the eight before it had been regridded and written."""
    cfg, checkpoints, _root, receipt, _tapes = smoke_export
    target = tmp_path / "clash"
    target.mkdir()
    last = Path(receipt["frames"][-1]["frame"])
    (target / last.name).write_bytes(last.read_bytes())
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        mpe.export_mapped_parent(
            cfg, checkpoints, target, nlat=8, nlon=16,
            start_date="2026-08-30_18:00:00")
    assert sorted(path.name for path in target.iterdir()) == [last.name]


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_folder_holding_frames_of_another_series_is_refused(
        smoke_export, tmp_path, overwrite):
    """One checkpoint exported with --overwrite into a folder of three left
    three frames on disk beside a receipt and documents naming one, and the
    next step lists the folder."""
    cfg, checkpoints, _root, receipt, _tapes = smoke_export
    target = tmp_path / "stale"
    target.mkdir()
    for row in receipt["frames"]:
        source = Path(row["frame"])
        (target / source.name).write_bytes(source.read_bytes())
    before = _folder_state(target)
    with pytest.raises(FileExistsError, match="this export does not write"):
        mpe.export_mapped_parent(
            cfg, checkpoints[:1], target, nlat=8, nlon=16,
            start_date="2026-08-30_18:00:00", overwrite=overwrite)
    assert _folder_state(target) == before


def test_overwrite_replaces_the_frames_and_the_receipt_of_the_same_series(
        smoke_export, tmp_path):
    from arwen_global.receipt import check_receipt

    cfg, checkpoints, _root, receipt, _tapes = smoke_export
    target = tmp_path / "again"
    first = mpe.export_mapped_parent(
        cfg, checkpoints, target, nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00")
    damaged = Path(first["frames"][1]["frame"])
    damaged.write_bytes(b"not a frame")
    (target / mpe.RECEIPT_NAME).write_text("{}", encoding="utf-8")
    second = mpe.export_mapped_parent(
        cfg, checkpoints, target, nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00", overwrite=True)
    from netCDF4 import Dataset

    for row, original in zip(second["frames"], receipt["frames"]):
        assert mpe._file_sha256(Path(row["frame"])) == row["sha256"]
        with Dataset(row["frame"]) as new, Dataset(original["frame"]) as old:
            for name in ("T", "U_EARTH", "PSFC"):
                np.testing.assert_array_equal(
                    np.asarray(new[name][0]), np.asarray(old[name][0]), err_msg=name)
    assert check_receipt(target / mpe.RECEIPT_NAME)["self_sha256"] == second["self_sha256"]
    assert len(list(target.glob(f"{mpe.FRAME_PREFIX}*.nc"))) == len(checkpoints)


def test_a_relative_outdir_leaves_absolute_paths_in_the_command(
        smoke_export, tmp_path, monkeypatch):
    """The page's own example passes a relative --outdir, and the command in
    the receipt is run from the regional run's folder."""
    cfg, checkpoints, _root, _receipt, _tapes = smoke_export
    monkeypatch.chdir(tmp_path)
    relative = mpe.export_mapped_parent(
        cfg, checkpoints[:2], "rel/parent", nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00")
    command = relative["regional_prep_command"]
    paths = [item.split("=", 1)[-1] for item in command
             if item.endswith((".nc", ".json"))]
    assert len(paths) == 2 + 2 + 4
    for item in paths:
        assert Path(item).is_absolute(), item
    for row in relative["frames"]:
        assert Path(row["frame"]).is_absolute() and Path(row["frame"]).is_file()
        assert Path(row["checkpoint"]).is_absolute()
    for row in relative["documents"].values():
        assert Path(row["path"]).is_absolute() and Path(row["path"]).is_file()
    assert mpe.regional_prep_command(["a.nc"], "rel")[7] == str(
        Path(tmp_path / "rel" / mpe.MAPPING_NAME))


def test_w_is_the_flux_through_the_coordinate_surfaces_with_a_known_answer():
    """W is the mass flux across the hybrid interfaces over rho g, not the
    velocity relative to the ground.  Three layers, B rising 0, 0, 1/2, 1
    from the top: a convergence of D Pa/s confined to the lowest layer
    raises the surface pressure by D, half of which the lowest layer takes
    up itself, so D/2 crosses the interface above it, upward, and nothing
    crosses the other three.  A state with no mass-flux divergence has
    W = 0 on every interface whatever its wind and whatever the slope under
    it, which is why the frame says W is not the velocity over the ground."""
    from types import SimpleNamespace

    from arwen_global.spectral.backend import get_backend
    from arwen_global.vertical import HybridCoordinate

    backend = get_backend("numpy")
    vertical = HybridCoordinate(
        np.array([100.0, 20000.0, 10000.0, 0.0]), np.array([0.0, 0.0, 0.5, 1.0]))
    shape = (3, 2, 3)
    p_full = np.array([5000.0, 40000.0, 90000.0])[:, None, None] * np.ones(shape)
    virtual = np.array([220.0, 250.0, 285.0])[:, None, None] * np.ones(shape)
    gas_constant = 287.0
    native = {"dp": np.ones(shape), "u": np.full(shape, 30.0), "v": np.full(shape, -10.0),
              "p_full": p_full, "virtual_temperature": virtual}
    convergence = 0.02                                   # Pa/s, lowest layer only
    divergence = np.zeros(shape)
    divergence[2] = -convergence

    def model_with(div_mass):
        return SimpleNamespace(
            gas_constant=gas_constant, vertical=vertical,
            grid_state=lambda atmosphere: native,
            _mass_flux_divergence=lambda dp, u, v: (None, div_mass))

    transform = SimpleNamespace(backend=backend)
    w = np.asarray(mpe.diagnose_interface_w(model_with(divergence), transform, None))
    assert w.shape == (4, 2, 3)
    rho = p_full / (gas_constant * virtual)
    expected = 0.5 * convergence / (0.5 * (rho[1] + rho[2]) * GRAVITY_M_S2)
    np.testing.assert_array_equal(w[0], 0.0)
    np.testing.assert_allclose(w[1], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(w[2], expected, rtol=1.0e-12)
    assert np.all(w[2] > 0.0)                            # up, top-to-surface order
    np.testing.assert_array_equal(w[3], 0.0)
    still = np.asarray(mpe.diagnose_interface_w(
        model_with(np.zeros(shape)), transform, None))
    np.testing.assert_array_equal(still, 0.0)
    # The frame and the provenance note say what it is.
    (text,) = [text for name, _units, text in mpe.INTERFACE_FIELDS if name == "W"]
    assert "through the hybrid coordinate surfaces" in text
    assert "not the velocity relative to the ground" in text
    policy = mpe.parent_provenance(
        run_name="r", config_hash="0", start="s", grid="g", nlev=3)["vertical_velocity_policy"]
    assert "B (dps/dt + V . grad ps)" in policy


def test_the_regional_engines_inspector_materializes_the_frames(smoke_export):
    """The check that the two packages agree: the regional engine's own
    mapped-source inspector decodes every frame through the mapping this
    door wrote, with no missing value in any canonical field."""
    mapped_source = pytest.importorskip("gpuwm.mapped_source")
    if not hasattr(mapped_source, "inspect_mapped_source"):
        pytest.skip("this engine has no mapped-source inspector")

    _cfg, checkpoints, root, receipt, _tapes = smoke_export
    frames = [row["frame"] for row in receipt["frames"]]
    report = mapped_source.inspect_mapped_source(
        root / "parent" / mpe.MAPPING_NAME, frames)
    text = json.dumps(report)
    assert report["status"].startswith("CANONICAL_FRAMES_MATERIALIZED"), report["status"]
    assert '"verdict": "PASS"' in text
    assert f'"frame_count": {len(checkpoints)}' in text


def test_the_regional_engine_loads_the_composition_and_parses_the_command(smoke_export):
    """The composition, the two role names and the assembled command were
    checked only against this module's own constants; one real run was the
    only thing that had put them through the engine.  Here the engine's own
    composition loader validates the document against the mapping this door
    wrote, and the engine's own argument parser takes the receipt's command
    with the four arguments that belong to the regional run."""
    composition = pytest.importorskip("gpuwm.mapped_composition")
    engine_cli = pytest.importorskip("gpuwm.cli")
    if not hasattr(composition, "load_composition"):
        pytest.skip("this engine has no composition loader")

    _cfg, _checkpoints, root, receipt, _tapes = smoke_export
    parent = root / "parent"
    contract = composition.load_composition(
        parent / mpe.COMPOSITION_NAME, parent / mpe.MAPPING_NAME)
    terrain = contract["supplements"]["terrain_height"]
    assert terrain["data_role"] == mpe.SUPPLEMENT_ROLE
    assert terrain["provenance_role"] == mpe.PROVENANCE_ROLE

    command = receipt["regional_prep_command"]
    assert command[0] == "gpuwm"
    parsed = engine_cli.build_parser().parse_args(command[1:] + [
        "--experiment-config", "regional.toml", "--wps-namelist", "regional.namelist.wps",
        "--geog-root", "geog", "--output-root", "prepared"])
    frames = [Path(row["frame"]) for row in receipt["frames"]]
    assert parsed.command == "prep" and parsed.source == "mapped"
    assert parsed.source_format == "netcdf"
    assert [Path(item) for item in parsed.mapped_inputs] == frames
    assert parsed.supplement == [f"{mpe.SUPPLEMENT_ROLE}={frame}" for frame in frames]
    assert parsed.provenance == [
        f"{mpe.PROVENANCE_ROLE}={parent / mpe.PROVENANCE_NAME}"]
    assert Path(parsed.mapping) == parent / mpe.MAPPING_NAME
    assert Path(parsed.composition) == parent / mpe.COMPOSITION_NAME


def test_the_command_reaches_the_exporter(monkeypatch, tmp_path, capsys):
    from arwen_global import cli

    seen = {}

    def record(cfg, checkpoints, outdir, **keywords):
        seen.update(keywords, checkpoints=list(checkpoints), outdir=outdir)
        return {"frames": [{"frame": "f.nc", "valid": "2026-11-12_00:00:00"}],
                "boundary_interval_seconds": None,
                "documents": {"mapping": {"path": "m.json", "sha256": "0"}},
                "regional_prep_command": ["gpuwm", "prep"]}

    monkeypatch.setattr(mpe, "export_mapped_parent", record)
    code = cli.main([
        "export-mapped-parent", "arwen_global_moist_smoke", "a.npz", "b.npz",
        "--outdir", str(tmp_path), "--nlat", "576", "--nlon", "1152",
        "--bbox", "20", "56", "-130", "-60"])
    assert code == 0
    assert seen["nlat"] == 576 and seen["nlon"] == 1152
    assert seen["bbox"] == (20.0, 56.0, -130.0, -60.0)
    assert [path.name for path in seen["checkpoints"]] == ["a.npz", "b.npz"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["frames"] == ["f.nc"]
    assert printed["regional_prep_command"] == ["gpuwm", "prep"]


def test_the_boundary_interval_is_the_frames_own_spacing_or_nothing():
    """The mapping's base cadence is read off the frames.  A series with a
    gap has no cadence, and the receipt must not invent one: the regional
    preparation refuses an uneven series, and a receipt that named an
    interval would send the reader there with a number that is not true."""
    start = datetime(2026, 11, 12)

    def at(*hours):
        return [start + timedelta(hours=hour) for hour in hours]

    assert mpe._uniform_interval(at(0, 3, 6, 9)) == 10800
    assert mpe._uniform_interval(at(0, 6)) == 21600
    assert mpe._uniform_interval(at(0, 3, 9)) is None
    assert mpe._uniform_interval(at(0)) is None
    assert mpe._uniform_interval([]) is None


def test_every_second_checkpoint_is_a_series_at_twice_the_spacing(smoke_export, tmp_path):
    cfg, checkpoints, _root, _receipt, _tapes = smoke_export
    sparse = mpe.export_mapped_parent(
        cfg, checkpoints[::2], tmp_path / "sparse", nlat=8, nlon=16,
        start_date="2026-08-30_18:00:00")
    assert [row["forecast_seconds"] for row in sparse["frames"]] == [0.0, 40.0]
    assert sparse["boundary_interval_seconds"] == 40
    assert sparse["lateral_boundary_series"] == "uniform"
    mapping = json.loads(
        (tmp_path / "sparse" / mpe.MAPPING_NAME).read_text(encoding="utf-8"))
    assert mapping["target"]["boundary_interval_seconds"] == 40


@pytest.mark.parametrize("bbox, match", [
    ((40.0, -40.0, -100.0, 20.0), "min < max on both axes"),
    ((-40.0, 40.0, 20.0, -100.0), "min < max on both axes"),
    ((0.0, 1.0, 0.0, 1.0), "needs at least 2x2"),
])
def test_a_window_that_cannot_hold_a_frame_is_refused(smoke_export, tmp_path, bbox, match):
    """A window typed in the wrong order, or one narrower than the grid
    spacing, would otherwise surface as an empty file the regional intake
    refuses for a reason that names neither the window nor this command."""
    cfg, checkpoints, _root, _receipt, _tapes = smoke_export
    with pytest.raises(ValueError, match=match):
        mpe.export_mapped_parent(
            cfg, checkpoints[:1], tmp_path / "bad-window", nlat=8, nlon=16,
            bbox=bbox, start_date="2026-08-30_18:00:00")
    assert not list((tmp_path / "bad-window").glob(f"{mpe.FRAME_PREFIX}*.nc"))


def test_a_mapping_for_no_column_or_no_cadence_is_refused():
    with pytest.raises(ValueError, match="at least two model levels"):
        mpe.parent_mapping(1)
    with pytest.raises(ValueError, match="positive number of seconds"):
        mpe.parent_mapping(40, boundary_interval_s=0)


def test_the_page_states_the_file_contract_the_writer_keeps():
    """The page is where a reader on the regional side learns the variable
    names and unit strings, and the decoder compares both as strings.  A
    variable added to the writer and not to the page, or a unit changed in
    one place, is caught here and not by a refused preparation."""
    page = Path(__file__).resolve().parents[1] / "docs" / "ARWEN_GLOBAL_REGIONAL_PARENT.md"
    if not page.is_file():
        pytest.skip("the docs are not beside this test")
    text = page.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    for group in (mpe.MASS_FIELDS, mpe.INTERFACE_FIELDS, mpe.SURFACE_FIELDS, mpe.SOIL_FIELDS):
        for name, units, _text in group:
            (row,) = [line for line in rows if f"`{name}`" in line.split("|")[1]]
            assert units in row.split("|")[3], (name, units, row)
    for name in (mpe.MAPPING_NAME, mpe.COMPOSITION_NAME, mpe.PROVENANCE_NAME,
                 mpe.RECEIPT_NAME):
        assert f"`{name}`" in text, name
