"""The render tape and the regional translation run in Rust, bit for bit.

THE BREAKAGE THIS PREVENTS (audit GI-3, the Python boundary law).  Every
render tape the default `go` and `render` paths write was regridded from
the Gaussian grid with NumPy on the host and its hydrostatic column
integrated in a Python per-level loop, and the regional translation
interpolated every target column in a Python loop of np.interp calls,
millions of them for a kilometre-scale target.  Those now run in the
`global_render_kernels` library, and these tests hold it to the NumPy it
replaced (`tests/render_kernel_oracle.py`) byte for byte, kernel by kernel
and then through a whole tape export and a whole regional translation.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_render_kernels

import render_kernel_oracle as oracle

from arwen_global import render_kernels
from arwen_global import doors

RNG = np.random.default_rng(20261005)


def _same_bits(actual, expected) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    a = np.ascontiguousarray(actual, dtype=np.float64).view(np.uint64)
    b = np.ascontiguousarray(expected, dtype=np.float64).view(np.uint64)
    differ = np.count_nonzero(a != b)
    assert differ == 0, (
        f"{differ} of {a.size} values differ; largest difference "
        f"{float(np.nanmax(np.abs(actual - expected))):.3e}")


# ---------------------------------------------------- without the library

def test_the_render_kernels_are_a_door_of_this_package_with_their_contract():
    door = doors.door_by_name(render_kernels.DOOR)
    assert door.library
    assert door.bundle == doors.COMPANION_BUNDLE
    assert doors.built_from_this_repository(door)
    assert door.marker == render_kernels.CONTRACT
    assert (Path(__file__).resolve().parents[1] / door.crate / "Cargo.toml").is_file()
    assert {"export", "render", "go"} <= set(door.used_by)


def test_a_machine_without_the_library_refuses_with_the_missing_door_exit(monkeypatch):
    """No NumPy fallback: a missing library is the missing-door refusal."""

    monkeypatch.setattr(render_kernels, "_LIBRARY", None)
    monkeypatch.setattr(doors, "find_door", lambda name: None)
    monkeypatch.setattr(doors, "search_path", lambda name: (Path("nowhere"),))
    with pytest.raises(doors.DoorMissing) as refusal:
        render_kernels.library()
    assert render_kernels.DOOR in str(refusal.value)
    assert "fetch-doors" in str(refusal.value)


def test_the_default_paths_carry_no_numpy_regrid_or_column_loop():
    """The NumPy arithmetic is gone from the three default-path modules."""

    root = Path(__file__).resolve().parents[1] / "src" / "arwen_global"
    tape = (root / "wrfout_export.py").read_text(encoding="utf-8")
    interpolation = (root / "regional" / "interpolation.py").read_text(encoding="utf-8")
    translate = (root / "regional" / "translate.py").read_text(encoding="utf-8")
    assert "np.interp(" not in tape + interpolation + translate
    assert "for k in range(nz - 1, -1, -1)" not in tape
    assert "for k in range(nlev - 1, -1, -1)" not in translate
    assert "for column in range(" not in interpolation
    assert "np.searchsorted(" not in interpolation


# ------------------------------------------------------- kernel by kernel

def _gaussian_latitudes(nlat):
    nodes, _ = np.polynomial.legendre.leggauss(nlat)
    return np.rad2deg(np.arcsin(nodes))          # south to north, as the grid


def _field(*shape, negative=False):
    values = RNG.normal(size=shape) * 10.0
    if negative:
        values[..., ::5, ::3] = -np.abs(values[..., ::5, ::3])
        values[..., 1::7, 2::5] = -0.0
        values[..., 2::9, ::4] = 0.0
    return values


@requires_render_kernels
@pytest.mark.parametrize("post", ["none", "floor_zero", "exp"])
def test_gaussian_regrid_matches_the_oracle(post):
    nlat, nlon = 48, 96
    lat_src = _gaussian_latitudes(nlat)
    lon_src = np.arange(nlon) * (360.0 / nlon)
    from arwen_global.spectral.sampling import regular_latlon_coordinates

    lat, lon = regular_latlon_coordinates(37, 72, include_poles=False)
    # Edge longitudes: just west of the first node, exactly on nodes, one
    # period out, negative.
    lon = np.concatenate([lon, [-1.0e-20, 0.0, 360.0, -3.75, 359.9999999999]])
    field = _field(3, nlat, nlon, negative=(post == "floor_zero"))
    if post == "exp":
        field = 11.5 + 0.01 * field
    got = render_kernels.GaussianRegridder(lat_src, lon_src, lat, lon)(field, post)
    want = oracle.OracleSeam.GaussianRegridder(lat_src, lon_src, lat, lon)(field, post)
    _same_bits(got, want)
    if post == "floor_zero":
        assert not np.signbit(got).any()


@requires_render_kernels
def test_a_rolled_window_is_the_whole_ring_regridded_then_rolled_and_cut():
    """The old tape regridded the whole ring and then rolled and cut it;
    the kernel samples the window's own rows and columns."""

    nlat, nlon = 40, 80
    lat_src = _gaussian_latitudes(nlat)
    lon_src = np.arange(nlon) * (360.0 / nlon)
    from arwen_global.spectral.sampling import regular_latlon_coordinates

    lat, lon = regular_latlon_coordinates(30, 64, include_poles=False)
    lon_signed = np.where(lon >= 180.0, lon - 360.0, lon)
    order = np.argsort(lon_signed)
    lat_sel = np.where((lat >= -20.0) & (lat <= 45.0))[0]
    lon_sel = np.where((lon_signed[order] >= -130.0) & (lon_signed[order] <= 10.0))[0]
    field = _field(2, nlat, nlon)
    whole = oracle.gaussian_to_regular(lat_src, lon_src, lat, lon)(field)
    want = whole[..., :, order][..., lat_sel, :][..., lon_sel]
    got = render_kernels.GaussianRegridder(
        lat_src, lon_src, lat[lat_sel], lon[order][lon_sel])(field)
    _same_bits(got, want)


@requires_render_kernels
@pytest.mark.parametrize("descending", [False, True])
def test_periodic_bilinear_matches_the_oracle(descending):
    lat = np.linspace(-87.5, 87.5, 36)
    if descending:
        lat = lat[::-1]
    lon = np.arange(72) * 5.0
    field = _field(4, lat.size, lon.size).astype(np.float32)
    target_lat = RNG.uniform(-87.5, 87.5, size=(9, 11))
    target_lon = RNG.uniform(-400.0, 400.0, size=(9, 11))
    target_lat[0, :3] = [lat.min(), lat.max(), 0.0]
    target_lon[0, :3] = [0.0, 355.0, -1.0e-20]
    got = render_kernels.periodic_bilinear(
        lat, float(lon[0]), 5.0, lon.size, field, target_lat, target_lon)
    want = oracle.periodic_bilinear(lat, lon, field, target_lat, target_lon)
    _same_bits(got, want)


def _columns(nsrc=12, ntgt=15, ny=5, nx=7):
    base = np.sort(RNG.uniform(1_000.0, 100_000.0, size=(nsrc, ny, nx)), axis=0)
    values = _field(nsrc, ny, nx)
    target = np.sort(RNG.uniform(500.0, 104_000.0, size=(ntgt, ny, nx)), axis=0)
    target[0, 0, 0] = base[3, 0, 0]          # an exact node hit
    target[-1, 0, 1] = base[-1, 0, 1]        # exactly the bottom level
    return base, values, target


@requires_render_kernels
@pytest.mark.parametrize("continued", [False, True])
def test_log_pressure_interpolation_matches_the_oracle(continued):
    base, values, target = _columns()
    bottom = _field(*target.shape) if continued else None
    got = render_kernels.log_pressure_interpolate(base, values, target, bottom)
    want = oracle.log_pressure_interpolate(base, values, target, bottom)
    _same_bits(got, want)


@requires_render_kernels
def test_standard_lapse_theta_matches_the_oracle():
    base, _values, target = _columns()
    theta = 280.0 + 40.0 * RNG.random(base.shape)
    kwargs = dict(reference_pressure_pa=100_000.0, kappa=287.0 / 1004.0,
                  gas_constant=287.0, gravity=9.80665, lapse_rate=6.5e-3)
    got = render_kernels.standard_lapse_theta_below(base, theta, target, **kwargs)
    want = oracle.standard_lapse_theta_below(base, theta, target, **kwargs)
    _same_bits(got, want)


@requires_render_kernels
def test_the_tape_column_matches_the_oracle():
    nz, ny, nx = 9, 6, 8
    a_half = np.concatenate([[100.0], np.linspace(2_000.0, 0.0, nz)])
    b_half = np.concatenate([[0.0], np.linspace(0.0, 1.0, nz)])
    ps = RNG.uniform(60_000.0, 104_000.0, size=(ny, nx))
    theta = 280.0 + 60.0 * RNG.random((nz, ny, nx))
    species = {name: np.maximum(RNG.normal(size=(nz, ny, nx)) * 1.0e-3, 0.0)
               for name in ("qv", "qc", "qr", "qi", "qs", "qg")}
    phis = RNG.uniform(0.0, 30_000.0, size=(ny, nx))
    kappa, r = 0.2857142857142857, 287.04
    p_half, p_full = render_kernels.hybrid_pressure(a_half, b_half, ps)
    t, tv = render_kernels.virtual_temperature(
        theta, species, p_full, reference_pressure_pa=100_000.0, kappa=kappa)
    phi = render_kernels.hydrostatic(tv, p_half, phis, gas_constant=r)
    want = oracle.tape_column(theta, species, ps, phis, a_half, b_half,
                              reference_pressure_pa=100_000.0, kappa=kappa,
                              gas_constant=r)
    for got_value, want_value in zip((p_half, p_full, t, tv, phi), want):
        _same_bits(got_value, want_value)


@requires_render_kernels
def test_full_level_geopotential_matches_the_oracle():
    nz, ny, nx = 7, 4, 5
    p_half = np.sort(RNG.uniform(1_000.0, 101_000.0, size=(nz + 1, ny, nx)), axis=0)
    tv = 200.0 + 100.0 * RNG.random((nz, ny, nx))
    phis = RNG.uniform(0.0, 20_000.0, size=(ny, nx))
    half, full = render_kernels.hydrostatic(tv, p_half, phis, gas_constant=287.0,
                                            full_levels=True)
    want_half, want_full = oracle.hydrostatic_half(tv, p_half, phis, gas_constant=287.0)
    _same_bits(half, want_half)
    _same_bits(full, want_full)


@requires_render_kernels
def test_the_library_refuses_a_malformed_grid_in_its_own_words():
    with pytest.raises(render_kernels.RenderKernelError, match="increase strictly"):
        render_kernels.GaussianRegridder(
            [10.0, 5.0, 0.0], [0.0, 120.0, 240.0], [1.0], [1.0])(np.zeros((3, 3)))


# ------------------------------------------------ whole paths, end to end

def _with_oracle(monkeypatch):
    for name in ("GaussianRegridder", "library", "periodic_bilinear",
                 "log_pressure_interpolate", "standard_lapse_theta_below",
                 "hybrid_pressure", "virtual_temperature", "hydrostatic"):
        monkeypatch.setattr(render_kernels, name, getattr(oracle.OracleSeam, name))


class _CapturingWriter:
    frames: list = []

    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self.kwargs = kwargs

    def write_frame(self, valid, frame):
        type(self).frames.append((valid, {k: np.array(v) for k, v in frame.items()}))
        self.path.write_bytes(b"captured")

    def close(self):
        pass

    def abort(self):
        pass


def _export(cfg, checkpoints, outdir, monkeypatch, **kwargs):
    from arwen_global import wrfout_export

    _CapturingWriter.frames = []
    monkeypatch.setattr(wrfout_export, "WrfoutWriter", _CapturingWriter)
    monkeypatch.setattr(wrfout_export, "_require_tape_writer", lambda: None)
    wrfout_export.export_wrfout(cfg, checkpoints, outdir,
                                start_date="2026-10-05_00:00:00", **kwargs)
    return list(_CapturingWriter.frames)


@requires_render_kernels
@pytest.mark.parametrize("bbox", [None, (-30.0, 50.0, -120.0, 40.0)])
def test_a_whole_tape_export_is_bit_identical_to_the_numpy_it_replaced(
        tmp_path, monkeypatch, bbox):
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.runner import run

    cfg = load_config(str(config_root() / "arwen_global_moist_smoke.toml"))
    receipt = run(cfg, tmp_path / "run")
    checkpoints = [Path(p) for p in receipt["checkpoints"]]
    rust = _export(cfg, checkpoints, tmp_path / "rust", monkeypatch,
                   nlat=18, nlon=36, bbox=bbox)
    with monkeypatch.context() as patch:
        _with_oracle(patch)
        numpy_frames = _export(cfg, checkpoints, tmp_path / "numpy", monkeypatch,
                               nlat=18, nlon=36, bbox=bbox)
    assert len(rust) == len(numpy_frames) == len(checkpoints)
    for (valid, got), (valid_numpy, want) in zip(rust, numpy_frames):
        assert valid == valid_numpy
        assert sorted(got) == sorted(want)
        for name in want:
            assert got[name].dtype == want[name].dtype, name
            assert got[name].tobytes() == want[name].tobytes(), name


@requires_render_kernels
def test_a_whole_regional_translation_is_bit_identical_to_the_numpy_it_replaced(
        tmp_path, monkeypatch):
    from test_arwen_global_level5_regional import CONFIG, _target_arrays

    from arwen_global.config import load_config
    from arwen_global.export import export_parent
    from arwen_global.regional.artifact import read_regional_frame, write_regional_target
    from arwen_global.regional.translate import translate_parent_to_regional_frame
    from arwen_global.runner import run

    cfg = load_config(CONFIG)
    run(cfg, tmp_path / "global")
    target = tmp_path / "target.npz"
    write_regional_target(target, _target_arrays(cfg), name="unit-target",
                          grid_id="unit-grid")
    parent = tmp_path / "parent.npz"
    export_parent(cfg, tmp_path / "global" / "arwen_global_step00000002.npz",
                  parent, nlat=17, nlon=36)
    translate_parent_to_regional_frame(parent, target, tmp_path / "rust.npz")
    with monkeypatch.context() as patch:
        _with_oracle(patch)
        translate_parent_to_regional_frame(parent, target, tmp_path / "numpy.npz")
    _, got = read_regional_frame(tmp_path / "rust.npz")
    _, want = read_regional_frame(tmp_path / "numpy.npz")
    assert sorted(got) == sorted(want)
    for name in want:
        assert np.asarray(got[name]).tobytes() == np.asarray(want[name]).tobytes(), name
