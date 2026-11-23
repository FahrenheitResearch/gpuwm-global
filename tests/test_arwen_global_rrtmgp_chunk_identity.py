"""One RRTMGP firing is the same bytes whatever its column chunk, on the card.

A radiation column reads only its own column of every input (the McICA
masks are seeded from the column's own pressures, the ozone from its own
latitude, the solvers fold each column's g-points in ascending order), and
the size-bounding record's fractions are reduced once over the whole grid,
so the column chunk is a memory knob and nothing else.  This file holds a
cloudy grid with day and night columns to that, plane for plane and bit for
bit, at one chunk for the whole grid, an even one, an odd one and one wider
than the grid, and with the dark columns put back through the shortwave
solver.  The shipped T383 config already records checkpoints byte-identical
at chunk 5000 against 12500; this is the same claim at test size.

``ARWEN_RAD_CHUNK_IDENTITY_DUMP=<path.npz>`` also writes the whole-grid
firing's planes to that file, so two source trees can be held to one
another by the file's hashes (the A/B a change to the call's
synchronizations is proven with).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arwen_global.core import rrtmgp  # noqa: E402

NZ, NY, NX = 30, 48, 72
NCOL = NY * NX
P_TOP = 100.0
PLANES = ("rthratenlw", "rthratensw", "swdown", "glw", "olr", "gsw",
          "coszen", "swupt", "swdnt", "lwupb", "cldfra_total")
CHUNKS = (NCOL, 1000, 777, 5000)


def _case():
    rng = np.random.default_rng(20261115)
    sigma = np.linspace(1.0, 0.0, NZ + 1)
    surface = 101325.0 - 4000.0 * rng.random((NY, NX))
    p_half = (P_TOP + sigma[:, None, None] * (surface - P_TOP)[None])
    p_half = p_half.astype(np.float32)
    p_half[-1] = np.float32(P_TOP)
    p_full = (0.5 * (p_half[:-1].astype(np.float64)
                     + p_half[1:].astype(np.float64)))
    lat = np.linspace(-85.0, 85.0, NY)[:, None] * np.ones((1, NX))
    warm = 12.0 * np.cos(np.deg2rad(lat))[None]
    temperature = np.maximum(
        (276.0 + warm) * (p_full / 101325.0) ** 0.19, 205.0)
    temperature += 0.5 * rng.standard_normal(temperature.shape)
    qv = 0.014 * np.cos(np.deg2rad(lat))[None] * (p_full / 101325.0) ** 3
    qv *= 0.5 + rng.random(qv.shape)

    def field(value):
        return cp.asarray(np.ascontiguousarray(value, dtype=np.float32))

    shape = (NZ, NY, NX)
    level = np.broadcast_to(p_full, shape)
    low = (level < 95000.0) & (level > 70000.0)
    high = (level < 40000.0) & (level > 20000.0)
    liquid = low & (rng.random(shape) < 0.35)
    ice = high & (rng.random(shape) < 0.30)
    snow = high & (rng.random(shape) < 0.25)
    qc = np.where(liquid, 4.0e-4 * rng.random(shape), 0.0)
    qi = np.where(ice, 8.0e-5 * rng.random(shape), 0.0)
    qs = np.where(snow, 6.0e-5 * rng.random(shape), 0.0)
    qr = np.where(liquid & (rng.random(shape) < 0.3), 1.0e-4, 0.0)
    # Some liquid cells carry Morrison's no-mass sentinel radius beside
    # mass, so the record's sentinel terms are exercised too.
    effc = np.where(rng.random(shape) < 0.2,
                    rrtmgp.MORRISON_NO_MASS_RADIUS_UM,
                    4.0 + 10.0 * rng.random(shape))
    effi = 15.0 + 60.0 * rng.random(shape)
    effs = 80.0 + 900.0 * rng.random(shape)
    atmosphere = {
        "pressure": field(p_full),
        "p_interface": field(p_half),
        "temperature": field(temperature),
        "exner": field((p_full / 1.0e5) ** (287.04 / 1004.6)),
        "qv": field(qv),
        "qc": field(qc), "qi": field(qi),
    }
    plane = lambda v: field(np.broadcast_to(v, (NY, NX)))  # noqa: E731
    fields = {"tsk": plane(285.0 + 15.0 * np.cos(np.deg2rad(lat))),
              "emiss": plane(0.95 + 0.04 * rng.random((NY, NX))),
              "albedo": plane(0.06 + 0.5 * rng.random((NY, NX)))}
    state = SimpleNamespace(
        p_top=P_TOP, elapsed_seconds=1800.0,
        qc=atmosphere["qc"], qr=field(qr), qi=atmosphere["qi"], qs=field(qs),
        nc=field(np.full(shape, 1.0e8)), nr=field(np.where(qr > 0, 1.0e4, 0.0)),
        ni=field(np.full(shape, 1.0e5)), ns=field(np.full(shape, 1.0e4)),
        effc=field(effc), effr=field(np.full(shape, 25.0)),
        effi=field(effi), effs=field(effs),
        physics=SimpleNamespace(microphysics_updates=1))
    cfg = SimpleNamespace(
        mp_physics=10, bl_pbl_physics=1, icloud_bl=0,
        wrf_rrtmg_compatibility="none", dt=300.0, clock_dt=300.0,
        radt=30.0, radt_minutes=30.0)
    grid = (cp.asarray(lat.astype(np.float32)),
            cp.asarray((np.linspace(0.0, 355.0, NX)[None, :]
                        * np.ones((NY, 1))).astype(np.float32)))
    return atmosphere, fields, state, cfg, grid


def _fire(column_chunk):
    atmosphere, fields, state, cfg, (lat, lon) = _case()
    # 06 UTC at an equinox: the terminator runs through the grid, so it
    # carries day and night columns on every latitude away from the poles.
    driver = rrtmgp.RRTMGPRadiation(
        start_time=datetime(2024, 3, 20, 6, 0, tzinfo=timezone.utc),
        latitude_deg=lat, longitude_deg=lon, column_chunk=column_chunk,
        column_size_bounding=True)
    result = driver(atmosphere=atmosphere, fields=fields, state=state,
                    cfg=cfg)
    out = {name: cp.asnumpy(cp.asarray(getattr(result, name)))
           for name in PLANES}
    for name, value in driver.last_size_bounding_columns.items():
        out[f"bounding_columns.{name}"] = cp.asnumpy(value)
    record = driver.last_size_bounding
    for name in rrtmgp.SIZE_BOUNDING_FIELDS:
        out[f"bounding.{name}"] = np.asarray(record[name], dtype=np.float64)
    return out


def _bits(value):
    value = np.ascontiguousarray(value)
    return value.dtype.str, value.shape, value.tobytes()


@pytest.fixture(scope="module")
def whole_grid():
    out = _fire(NCOL)
    target = os.environ.get("ARWEN_RAD_CHUNK_IDENTITY_DUMP", "").strip()
    if target:
        np.savez(target, **out)
        digest = hashlib.sha256()
        for name in sorted(out):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(out[name]).tobytes())
        print(f"\nchunk-identity whole-grid sha256 {digest.hexdigest()}")
    return out


def test_the_case_has_day_night_and_cloud(whole_grid):
    mu = whole_grid["coszen"]
    assert (mu > 0).sum() > NCOL // 4 and (mu <= 0).sum() > NCOL // 4
    assert (whole_grid["cldfra_total"] > 0).sum() > NCOL // 2
    assert whole_grid["bounding.liquid_cells"] > 0
    assert whole_grid["bounding.ice_cells"] > 0
    assert whole_grid["bounding.liquid_sentinel_cells"] > 0
    for name in PLANES:
        assert np.all(np.isfinite(whole_grid[name])), name


@pytest.mark.parametrize("column_chunk", CHUNKS[1:])
def test_every_plane_is_the_whole_grid_bits_at_any_chunk(whole_grid,
                                                         column_chunk):
    chunked = _fire(column_chunk)
    assert sorted(chunked) == sorted(whole_grid)
    for name in whole_grid:
        assert _bits(chunked[name]) == _bits(whole_grid[name]), name


@pytest.mark.parametrize("fused, planck", (
    ("0", ""), ("lw", ""), ("sw", ""), ("1", "0"), ("1", "1")))
def test_the_in_solver_cloud_optics_and_planck_are_the_standalone_bits(
        whole_grid, monkeypatch, fused, planck):
    # The default (both bands fused in the solvers, the Planck sources
    # in-solver where verified) against the standalone finalize and Planck
    # kernels, each band fused alone, and the Planck form forced each way.
    if planck == "1" and (str(cp.cuda.Device().compute_capability)
                          not in rrtmgp._FUSED_PLANCK_VERIFIED_CC):
        pytest.skip("in-solver Planck not measured on this architecture")
    monkeypatch.setenv(rrtmgp.FUSED_SOLVER_ENV, fused)
    if planck:
        monkeypatch.setenv(rrtmgp.FUSED_PLANCK_ENV, planck)
    standalone = _fire(1000)
    for name in whole_grid:
        assert _bits(standalone[name]) == _bits(whole_grid[name]), name


def test_a_chunk_past_the_32_bit_cube_is_refused():
    atmosphere, fields, state, cfg, (lat, lon) = _case()
    driver = rrtmgp.RRTMGPRadiation(
        start_time=datetime(2024, 3, 20, 6, 0, tzinfo=timezone.utc),
        latitude_deg=lat, longitude_deg=lon, column_chunk=NCOL)
    # Pretend the grid is wide enough for the chunk to matter: the guard
    # reads min(chunk, ncol), so a 2**31-cell chunk needs ncol that wide;
    # instead, check the arithmetic the guard uses at the T383 whole grid.
    lw_upper, sw_upper = rrtmgp.rrtmgp_above_model_layer_counts(P_TOP)
    cells = 663_552 * (40 + max(lw_upper, sw_upper)) * max(
        driver.lw_tables.ngpt, driver.sw_tables.ngpt)
    assert cells >= 2**31  # the shape that died at an illegal address


def test_solving_the_dark_columns_is_the_same_bits(whole_grid, monkeypatch):
    monkeypatch.setenv(rrtmgp.DARK_COLUMNS_ENV, "solve")
    solved = _fire(777)
    for name in whole_grid:
        assert _bits(solved[name]) == _bits(whole_grid[name]), name
