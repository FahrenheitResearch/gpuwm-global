"""The cold start's regrid and remap run in the Rust door, bit for bit.

Audit GI-3 (2026-10-05): every analysis-initialised start regridded the
analysis onto the Gaussian grid and remapped it onto model levels in host
NumPy, which the Python boundary forbids on the data path.  Both now run in
``rust/rw-global-coldstart`` behind :mod:`arwen_global.coldstart_bridge`,
and the NumPy expressions survive only in ``tests/coldstart_oracle.py``.

What these hold:

* the door's output is BYTE-identical to the NumPy expressions it replaced,
  on the shapes a real cold start hands it (a 0.25-degree GDAS-like ring
  onto a Gaussian grid, 31 isobaric levels onto hybrid full levels), on a
  latitude-ascending ring that starts at -180, and with NaN in the field;
* the thread count moves no bit;
* the default path has no NumPy body left: with the library unreachable the
  cold start refuses with the door's exit-3 refusal instead of regridding
  in Python.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
import sys

import numpy as np
import pytest

from arwen_global import coldstart_bridge
from arwen_global.analysis_initial import _global_regridder, _to_model_levels
from arwen_global.doors import DoorMissing, door_by_name
from arwen_global.spectral.grid import GaussianGrid

_ORACLE_PATH = Path(__file__).with_name("coldstart_oracle.py")
_spec = importlib.util.spec_from_file_location("coldstart_oracle", _ORACLE_PATH)
oracle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oracle
_spec.loader.exec_module(oracle)


def _same_bits(rust: np.ndarray, numpy: np.ndarray) -> None:
    assert rust.dtype == np.float64 and numpy.dtype == np.float64
    assert rust.shape == numpy.shape
    nan_r, nan_n = np.isnan(rust), np.isnan(numpy)
    assert np.array_equal(nan_r, nan_n), "the two disagree on which values are NaN"
    finite = ~nan_r
    differ = int(np.count_nonzero(
        rust[finite].view(np.uint64) != numpy[finite].view(np.uint64)))
    assert differ == 0, f"{differ} of {finite.sum()} values differ in their bits"


class _Grid:
    def __init__(self, latitude_deg, longitude_deg):
        self.latitude_deg = np.asarray(latitude_deg, dtype=np.float64)
        self.longitude_deg = np.asarray(longitude_deg, dtype=np.float64)


def _quarter_degree_ring():
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(1440) * 0.25
    return lat, lon


def _field(rng, shape, dtype=np.float64):
    lat_axis = np.linspace(-1.0, 1.0, shape[-2])[:, None]
    lon_axis = np.linspace(0.0, 2.0 * np.pi, shape[-1])[None, :]
    smooth = 250.0 + 40.0 * np.cos(np.pi * lat_axis) * np.sin(3.0 * lon_axis)
    noise = rng.standard_normal(shape)
    return (smooth + noise).astype(dtype)


def test_regrid_is_byte_identical_on_a_gdas_ring_onto_a_gaussian_grid():
    rng = np.random.default_rng(20261005)
    lat, lon = _quarter_degree_ring()
    grid = GaussianGrid.create(255)
    rust = _global_regridder(lat, lon, grid)
    numpy = oracle.numpy_global_regridder(lat, lon, grid)
    for values in (
        _field(rng, (3, lat.size, lon.size)),              # a level stack
        _field(rng, (lat.size, lon.size), np.float32),     # a GRIB float32 plane
    ):
        _same_bits(rust(values), numpy(values))


def test_regrid_is_byte_identical_on_an_ascending_ring_from_minus_180_with_nan():
    rng = np.random.default_rng(7)
    lat = np.linspace(-90.0, 90.0, 181)
    lon = -180.0 + np.arange(360) * 1.0
    target = _Grid(
        np.concatenate(([-90.0], np.linspace(-89.3, 89.3, 57), [90.0])),
        # wraps, negatives, a value on the seam, and past 360
        np.concatenate((np.linspace(-200.0, 520.0, 113), [-180.0, 180.0, 0.0])))
    values = _field(rng, (2, lat.size, lon.size))
    values[0, 90, 17] = np.nan
    values[1, 0, :] = np.nan
    rust = _global_regridder(lat, lon, target)(values)
    numpy = oracle.numpy_global_regridder(lat, lon, target)(values)
    _same_bits(rust, numpy)


def _hybrid_columns(rng, nlev, nlat, nlon):
    """ln(p) of model full levels over surface pressures from 50 to 106 kPa,
    so the deepest columns sit below the bottom (1000 hPa) source level."""
    ps = rng.uniform(50_000.0, 106_000.0, size=(nlat, nlon))
    eta = (np.arange(nlev) + 0.5) / nlev
    a = 2_000.0 * (1.0 - eta)
    b = eta ** 1.5
    return np.log(a[:, None, None] + b[:, None, None] * ps[None]), ps


def test_remap_is_byte_identical_on_isobaric_to_hybrid_columns():
    rng = np.random.default_rng(31)
    levels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250,
                       300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
                       850, 900, 925, 950, 975, 1000], dtype=np.float64) * 100.0
    ln_source = np.log(levels)
    nlat, nlon = 48, 96
    ln_target, ps = _hybrid_columns(rng, 64, nlat, nlon)
    values = _field(rng, (levels.size, nlat, nlon))
    values[4, 3, 5] = np.nan
    for extrapolate in (False, True):
        _same_bits(
            _to_model_levels(values, ln_source, ln_target, extrapolate_below=extrapolate),
            oracle.numpy_to_model_levels(values, ln_source, ln_target,
                                         extrapolate_below=extrapolate))
    # surface_virtual_temperature's shape: one target level, ln(ps)[None]
    ln_ps = np.log(ps)[None]
    _same_bits(
        _to_model_levels(values, ln_source, ln_ps, extrapolate_below=True),
        oracle.numpy_to_model_levels(values, ln_source, ln_ps, extrapolate_below=True))
    # a float32 field promotes exactly as NumPy promoted it
    single = values.astype(np.float32)
    _same_bits(_to_model_levels(single, ln_source, ln_target),
               oracle.numpy_to_model_levels(single, ln_source, ln_target))


def test_thread_count_moves_no_bit():
    rng = np.random.default_rng(3)
    lat, lon = _quarter_degree_ring()
    grid = GaussianGrid.create(127)
    values = _field(rng, (4, lat.size, lon.size))
    args = (lat, lon, grid.latitude_deg, grid.longitude_deg)
    one = coldstart_bridge.regrid(values, *args, threads=1)
    many = coldstart_bridge.regrid(values, *args, threads=7)
    _same_bits(one, many)
    ln_source = np.log(np.linspace(100.0, 100_000.0, 31))
    ln_target, _ = _hybrid_columns(rng, 64, 96, 192)
    column = _field(rng, (31, 96, 192))
    _same_bits(coldstart_bridge.remap(column, ln_source, ln_target, threads=1),
               coldstart_bridge.remap(column, ln_source, ln_target, threads=7))


def test_the_default_cold_start_has_no_numpy_body(monkeypatch, tmp_path):
    """With the door unreachable the cold start refuses (exit 3, the bundle
    that stages it) instead of quietly regridding in Python."""

    monkeypatch.setattr(coldstart_bridge, "_LIBRARY", None)
    monkeypatch.setattr(coldstart_bridge, "library_candidates",
                        lambda: (tmp_path / "absent.so",))
    monkeypatch.delenv(door_by_name("rw_global_coldstart").env_var, raising=False)
    lat, lon = _quarter_degree_ring()
    with pytest.raises(DoorMissing, match="rw_global_coldstart"):
        _global_regridder(lat, lon, GaussianGrid.create(63))
    with pytest.raises(DoorMissing, match="rw_global_coldstart"):
        _to_model_levels(np.zeros((2, 3)), np.log([1.0e4, 1.0e5]), np.zeros((4, 3)))


def test_the_door_carries_its_contract_literal():
    door = door_by_name("rw_global_coldstart")
    assert door.library and door.marker == coldstart_bridge.ABI_MARKER
    assert door.marker in coldstart_bridge.resolved_path().read_bytes()


def test_nothing_under_src_imports_the_oracle():
    src = Path(coldstart_bridge.__file__).resolve().parent
    pattern = re.compile(r"^\s*(?:from\s+\S*coldstart_oracle|import\s+\S*coldstart_oracle)", re.M)
    for path in src.rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
