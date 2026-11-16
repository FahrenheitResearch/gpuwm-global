"""The globalized Newton departure-point search (trajectory_solver = "newton_ls").

The plain Newton form linearizes a piecewise-trilinear interpolant whose
Jacobian jumps at every cell face; past (dt/2)|J| of about one a full step
from one cell lands in the next and the point cycles, and its damping only
ever halves while its gate reads the undamped step.  The line-search form
takes the Newton direction and shortens it until the residual falls, and
its gate reads the Newton step at the point it returns.  These gates say:
it is opt-in and named in the identity; it solves the same equation (it
agrees with a long fixed-point reference wherever that converges); it
converges where both shipped solvers stall; the move it reports bounds
its real error; and a band reads the whole grid's bits.
"""
from __future__ import annotations

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


def test_newton_ls_is_opt_in_and_named_in_the_identity():
    default = SemiLagrangianOptions()
    assert "trajectory_solver" not in default.identity
    ls = SemiLagrangianOptions(trajectory_solver="newton_ls")
    assert ls.identity["trajectory_solver"] == "newton_ls"
    newton = SemiLagrangianOptions(trajectory_solver="newton")
    assert ls.identity != newton.identity


def _wind(tables, grid, dtype, *, wave, amp_uv, amp_s, nlev=20):
    from arwen_global.semilag.trajectory import CartesianWind, cartesian_wind

    lat = cp.asarray(grid.lat_rad)[None, :, None]
    lon = cp.asarray(grid.lon_rad)[None, None, :]
    kk = cp.arange(nlev, dtype=dtype)[:, None, None]
    u = (60 * cp.cos(lat) ** 2 * (1 + 0.1 * kk) + amp_uv * cp.sin(wave * lon)
         * cp.cos(wave * lat) * cp.cos(lat)).astype(dtype)
    v = (amp_uv * cp.cos(wave * lon) * cp.sin(wave * lat)
         * cp.cos(lat)).astype(dtype)
    s = (amp_s * cp.sin((2 * wave // 3) * lon) * cp.cos(lat)
         * cp.cos(kk * 0.3)).astype(dtype)
    vx, vy, vz = cartesian_wind(u, v, tables)
    return CartesianWind(cp.ascontiguousarray(vx), cp.ascontiguousarray(vy),
                         cp.ascontiguousarray(vz), cp.ascontiguousarray(s))


def _distance(a, b, tables):
    dl = 0.5 * (a.xi - b.xi) * tables.dlam
    dp = 0.5 * (a.phi - b.phi)
    hav = cp.sin(dp) ** 2 + cp.cos(a.phi) * cp.cos(b.phi) * cp.sin(dl) ** 2
    return 2 * tables.radius_m * cp.arcsin(cp.sqrt(cp.clip(hav, 0, 1)))


def _cells(tables):
    from arwen_global.semilag.trajectory import meridional_grid_length_m

    return cp.asarray(meridional_grid_length_m(tables))[None, :, None]


@device
def test_newton_ls_solves_the_fixed_point_equation():
    """Where the fixed-point pass converges (60 iterations) the line-search
    Newton form at four lands on the same departure points."""
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import departure_points

    grid = GaussianGrid.create(255)
    tables = SphericalGridTables.create(grid, xp=cp, dtype=np.float64)
    wind = _wind(tables, grid, np.float64, wave=24, amp_uv=40.0, amp_s=2e-3)
    dt = 1800.0
    fixed60, f60 = departure_points(wind, tables, dt, iterations=60)
    assert f60.move_max_cells < 1e-6
    ls4, d4 = departure_points(wind, tables, dt, iterations=4,
                               solver="newton_ls")
    assert d4.move_max_cells < 1e-3
    # the search stops a point once its residual is under 1e-4 cells
    # (5.2 m at T255): MEASURED 5.97 m from the 60-iteration pass at most
    err = _distance(ls4, fixed60, tables) / _cells(tables)
    assert float(cp.max(err)) < 2e-4
    assert float(cp.max(cp.abs(ls4.level - fixed60.level))) < 1e-3


#: (wave number, horizontal amplitude m/s, level-rate amplitude 1/s, dt s).
#: MEASURED on the RTX 5070 Ti, float64 T255 L20 (moves in meridional cells
#: at 8 iterations for the shipped solvers and 4 for newton_ls):
#:   24/40/2e-3/1800  fold 0.374: fixed 0.00247, newton 3e-14, ls 1.2e-4
#:   24/40/2e-3/2200  fold 0.267: fixed 0.016,   newton 3e-14, ls 1.2e-4
#:   120/30/4e-3/1000 fold 0.263: fixed 0.0064,  newton 0.00464, ls 1.3e-4
#:   24/40/2e-3/2700  fold 0.147: fixed 0.0925,  newton 0.00797, ls 1.2e-4
#:   120/30/4e-3/1200 fold 0.173: fixed 0.0167,  newton 0.00809, ls 1.9e-4
#: and newton_ls 4 sits on its 30-iteration reference in every one.  The
#: third is the shipped Newton form's stall inside the fold gate: a
#: grid-scale flow at an admitted determinant, unconverged by 0.002 cells
#: after eight passes because one overshoot halved its damping for good.
#: Where the map folds (wave 60 at 1800 s, determinant -0.09) no solver
#: converges, and the fold gate refuses that step.
_HARD = (
    (24, 40.0, 2e-3, 1800.0),
    (24, 40.0, 2e-3, 2200.0),
    (120, 30.0, 4e-3, 1000.0),
    (24, 40.0, 2e-3, 2700.0),
    (120, 30.0, 4e-3, 1200.0),
)


@device
@pytest.mark.parametrize("wave,amp_uv,amp_s,dt", _HARD)
def test_newton_ls_converges_where_the_shipped_solvers_stall(
        wave, amp_uv, amp_s, dt):
    """Grid-scale flows at long steps.  Wherever a 30-iteration
    line-search reference converged, eight line-search iterations converge
    under the gate, and the move the search reports bounds its real
    distance from that reference."""
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import departure_points, lipschitz

    grid = GaussianGrid.create(255)
    tables = SphericalGridTables.create(grid, xp=cp, dtype=np.float64)
    wind = _wind(tables, grid, np.float64, wave=wave, amp_uv=amp_uv,
                 amp_s=amp_s)
    cells = _cells(tables)
    lip = lipschitz(wind, tables, dt)
    fixed8, df8 = departure_points(wind, tables, dt, iterations=8)
    newton8, dn8 = departure_points(wind, tables, dt, iterations=8,
                                    solver="newton")
    ref, dref = departure_points(wind, tables, dt, iterations=30,
                                 solver="newton_ls")
    ls4, d4 = departure_points(wind, tables, dt, iterations=4,
                               solver="newton_ls")
    ls8, d8 = departure_points(wind, tables, dt, iterations=8,
                               solver="newton_ls")
    err = {name: float(cp.max(_distance(st, ref, tables) / cells))
           for name, st in (("fixed8", fixed8), ("newton8", newton8),
                            ("ls4", ls4), ("ls8", ls8))}
    print(f"\nHARD wave={wave} amp={amp_uv} s={amp_s} dt={dt} "
          f"fold={lip.fold_determinant:.3f} lip={lip.lipschitz:.3f} | "
          f"moves fixed8 {df8.move_max_cells:.3g} newton8 "
          f"{dn8.move_max_cells:.3g} ls4 {d4.move_max_cells:.3g} ls8 "
          f"{d8.move_max_cells:.3g} ref30 {dref.move_max_cells:.3g} | "
          f"errors vs ref " + " ".join(f"{k} {v:.3g}" for k, v in err.items()))
    assert dref.move_max_cells < 1e-3
    assert d4.move_max_cells < 1e-3
    assert err["ls4"] < 1e-3
    # the reported move bounds the real error
    assert err["ls4"] <= 2.0 * d4.move_max_cells + 1e-4
    # and the shipped solvers at twice the passes are ten times further out
    if (wave, dt) != (24, 1800.0) and (wave, dt) != (24, 2200.0):
        assert dn8.move_max_cells > 10.0 * d4.move_max_cells
    assert df8.move_max_cells > 10.0 * d4.move_max_cells


@device
def test_newton_ls_band_reads_the_whole_grids_bits():
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import CartesianWind, departure_points

    grid = GaussianGrid.create(255)
    tables = SphericalGridTables.create(grid, xp=cp, dtype=np.float32)
    wind = _wind(tables, grid, np.float32, wave=24, amp_uv=40.0, amp_s=2e-3)
    whole, wd = departure_points(wind, tables, 600.0, iterations=4,
                                 solver="newton_ls")
    nlat = grid.nlat
    for a0, a1, halo in ((0, 40, 12), (40, 100, 12), (200, nlat, 12)):
        s0, s1 = max(0, a0 - halo), min(nlat, a1 + halo)
        window = tables.window(slice(s0, s1), slice(a0, a1))
        held = CartesianWind(*(cp.ascontiguousarray(x[:, s0:s1])
                               for x in wind.arrays()))
        band, _ = departure_points(held, tables, 600.0, iterations=4,
                                   window=window, solver="newton_ls")
        for name in ("xi", "phi", "level"):
            assert bool(cp.array_equal(getattr(band, name),
                                       getattr(whole, name)[:, a0:a1])), name
