"""The opt-in Newton solver of the departure-point equation.

The fixed-point pass contracts by the spectral radius of (dt/2) J, which
the fold gate does not bound (a rotation or the jet's vertical-shear
coupling keeps det(I + (dt/2) J) at or above one whatever its size).  The
Newton form solves the same equation through the same gather, so where
both converge they agree, and it converges where the fixed-point pass
does not.  The default stays the fixed-point pass and stays out of the
identity, so every shipped hash holds.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arwen_global.semilag.options import SemiLagrangianOptions

cp = pytest.importorskip("cupy")


def _has_device():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - no driver
        return False


device = pytest.mark.skipif(not _has_device(), reason="needs a CUDA device")


def test_default_solver_is_fixed_point_and_absent_from_the_identity():
    default = SemiLagrangianOptions()
    assert default.trajectory_solver == "fixed_point"
    assert "trajectory_solver" not in default.identity
    newton = SemiLagrangianOptions(trajectory_solver="newton")
    assert newton.identity["trajectory_solver"] == "newton"
    with pytest.raises(ValueError):
        SemiLagrangianOptions(trajectory_solver="anderson")


def _hard_wind(tables, grid, dtype, nlev=20):
    from arwen_global.semilag.trajectory import CartesianWind, cartesian_wind

    lat = cp.asarray(grid.lat_rad)[None, :, None]
    lon = cp.asarray(grid.lon_rad)[None, None, :]
    kk = cp.arange(nlev, dtype=dtype)[:, None, None]
    u = (60 * cp.cos(lat) ** 2 * (1 + 0.1 * kk) + 40 * cp.sin(24 * lon)
         * cp.cos(24 * lat) * cp.cos(lat)).astype(dtype)
    v = (40 * cp.cos(24 * lon) * cp.sin(24 * lat) * cp.cos(lat)).astype(dtype)
    s = (2e-3 * cp.sin(16 * lon) * cp.cos(lat)
         * cp.cos(kk * 0.3)).astype(dtype)
    vx, vy, vz = cartesian_wind(u, v, tables)
    return CartesianWind(cp.ascontiguousarray(vx), cp.ascontiguousarray(vy),
                         cp.ascontiguousarray(vz), cp.ascontiguousarray(s))


def _distance(a, b, tables):
    dl = 0.5 * (a.xi - b.xi) * tables.dlam
    dp = 0.5 * (a.phi - b.phi)
    hav = cp.sin(dp) ** 2 + cp.cos(a.phi) * cp.cos(b.phi) * cp.sin(dl) ** 2
    return 2 * tables.radius_m * cp.arcsin(cp.sqrt(cp.clip(hav, 0, 1)))


@device
def test_newton_converges_where_the_fixed_point_pass_does_not():
    """MEASURED on the GPU box (T255, float64, dt = 1800 s, fold
    determinant 0.374, Lipschitz 1.06): the fixed-point pass left 1570 m
    at three iterations and 43 m at eight; Newton left 1.25 m at three
    and 0 at four, against a 60-iteration Newton reference that the
    60-iteration fixed-point pass reproduces to roundoff."""
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import departure_points

    grid = GaussianGrid.create(255)
    tables = SphericalGridTables.create(grid, xp=cp, dtype=np.float64)
    wind = _hard_wind(tables, grid, np.float64)
    dt = 1800.0
    fixed, _ = departure_points(wind, tables, dt, iterations=60)
    newton60, _ = departure_points(wind, tables, dt, iterations=60,
                                   solver="newton")
    assert float(cp.max(_distance(fixed, newton60, tables))) < 1e-3
    fixed8, d8 = departure_points(wind, tables, dt, iterations=8)
    newton4, d4 = departure_points(wind, tables, dt, iterations=4,
                                   solver="newton")
    assert d8.move_max_cells > 1e-3
    assert d4.move_max_cells < 1e-3
    assert float(cp.max(_distance(newton4, newton60, tables))) < 1.0
    assert float(cp.max(_distance(fixed8, newton60, tables))) > 10.0


@device
def test_newton_band_reads_the_whole_grids_bits():
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import CartesianWind, departure_points

    grid = GaussianGrid.create(255)
    tables = SphericalGridTables.create(grid, xp=cp, dtype=np.float32)
    wind = _hard_wind(tables, grid, np.float32)
    whole, _ = departure_points(wind, tables, 600.0, iterations=3,
                                solver="newton")
    nlat = grid.nlat
    for a0, a1, halo in ((0, 40, 12), (40, 100, 12), (200, nlat, 12)):
        s0, s1 = max(0, a0 - halo), min(nlat, a1 + halo)
        window = tables.window(slice(s0, s1), slice(a0, a1))
        held = CartesianWind(*(cp.ascontiguousarray(x[:, s0:s1])
                               for x in wind.arrays()))
        band, _ = departure_points(held, tables, 600.0, iterations=3,
                                   window=window, solver="newton")
        for name in ("xi", "phi", "level"):
            assert bool(cp.array_equal(getattr(band, name),
                                       getattr(whole, name)[:, a0:a1])), name


@device
@pytest.mark.parametrize("alpha", [0.0, math.pi / 4.0, math.pi / 2.0])
def test_newton_matches_the_fixed_point_pass_on_solid_body(alpha):
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.semilag.cases import (
        CASE_PERIOD_S, _mesh, solid_body_wind)
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import (
        CartesianWind, cartesian_wind, departure_points)

    grid = GaussianGrid.create(21)
    tables = SphericalGridTables.create(grid, xp=cp, dtype=np.float64)
    lam, phi = _mesh(grid)
    u0 = 2.0 * math.pi * grid.radius_m / CASE_PERIOD_S
    u2, v2 = solid_body_wind(lam, phi, u0=u0, alpha=alpha)
    u = cp.asarray(np.repeat(u2[None], 4, 0))
    v = cp.asarray(np.repeat(v2[None], 4, 0))
    vx, vy, vz = cartesian_wind(u, v, tables)
    wind = CartesianWind(cp.ascontiguousarray(vx), cp.ascontiguousarray(vy),
                         cp.ascontiguousarray(vz), cp.zeros_like(vx))
    dt = CASE_PERIOD_S / 256.0
    fixed, _ = departure_points(wind, tables, dt, iterations=30)
    newton, diag = departure_points(wind, tables, dt, iterations=3,
                                    solver="newton")
    assert diag.move_max_cells < 1e-6
    assert float(cp.max(_distance(fixed, newton, tables))) < 1e-3
