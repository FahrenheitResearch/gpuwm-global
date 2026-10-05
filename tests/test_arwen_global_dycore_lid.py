"""The model lid and ground in both dynamical cores (DYC-1, DYC-3).

DYC-1.  The level-0 rate is half the first interior interface rate, so a
parcel arriving at level 0 under descent departs from the half layer
between the lid interface (index -1/2) and level 0.  The semi-Lagrangian
search clamped that departure to level 0 and the gather read the parcel's
own value, while an ascending parcel read the colder level below: a
first-order cooling of the lid, -0.5 d (theta_0 - theta_1) per step for a
zero-mean displacement d, with theta_0 - theta_1 = 232 K on the default
40-level stack.  The Eulerian core had the same one-sided defect through a
zero boundary-layer gradient.  These tests fail on that arithmetic and
pass on the interface clamp, the bounded half-layer read and the
one-sided boundary gradient.

DYC-3.  The level-index rate divided omega by the arithmetic mean layer
thickness, where the gather's full levels sit at geometric layer means.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.constants import KAPPA, REFERENCE_PRESSURE_PA  # noqa: E402
from arwen_global.dynamics import MoistHybridModel  # noqa: E402
from arwen_global.semi_implicit import BarotropicSemiImplicit  # noqa: E402
from arwen_global.semilag.interpolate import Stencil, gather, zero_stencil  # noqa: E402
from arwen_global.semilag.tables import SphericalGridTables  # noqa: E402
from arwen_global.semilag.trajectory import (  # noqa: E402
    CartesianWind,
    departure_points,
    level_rate_from_mass_flux,
)
from arwen_global.spectral.grid import GaussianGrid  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402

PS = 101325.0


def _tables(truncation=21):
    grid = GaussianGrid.create(truncation)
    return grid, SphericalGridTables.create(grid, xp=np, dtype=np.float64)


def _column_stencil(tables, grid, level_of_k):
    """The identity stencil with each level's departure index replaced."""
    nlev = len(level_of_k)
    base = zero_stencil(tables, nlev, xp=np, dtype=np.float64)
    level = np.broadcast_to(
        np.asarray(level_of_k, dtype=np.float64)[:, None, None],
        (nlev, grid.nlat, grid.nlon),
    ).copy()
    return Stencil(xi=np.asarray(base.xi).copy(), phi=np.asarray(base.phi).copy(),
                   level=level, tables=tables)


def _layered(grid, profile):
    nlev = len(profile)
    return np.broadcast_to(
        np.asarray(profile, dtype=np.float64)[:, None, None],
        (nlev, grid.nlat, grid.nlon),
    ).copy()


def _default_theta():
    """theta on the default 40-level stack, isothermal 220 K above 100 hPa
    and a 6.5 K/km-like profile below: the lid pair differs by hundreds of
    kelvin, as in the forecast state."""
    vertical = HybridCoordinate.surface_stretched(40, 100.0)
    p_half = vertical.a_half_pa + vertical.b_half * PS
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    temperature = np.maximum(
        220.0, 288.15 * (p_full / PS) ** (287.0 * 0.0065 / 9.80665)
    )
    return p_full, temperature * (REFERENCE_PRESSURE_PA / p_full) ** KAPPA


# ---------------------------------------------------------------- DYC-1 SL

def test_departure_level_is_clamped_at_the_boundary_interfaces():
    grid, tables = _tables()
    nlev = 6
    shape = (nlev, grid.nlat, grid.nlon)
    zero = np.zeros(shape)
    dt = 300.0
    for rate, top, bottom in (
        (1.0e-3, -0.3, nlev - 1 - 0.3),     # descent: lid departs above level 0
        (-1.0e-3, 0.3, nlev - 1 + 0.3),     # ascent: ground departs below the last
        (4.0e-3, -0.5, nlev - 1 - 1.2),     # beyond the lid interface: clamped there
        (-4.0e-3, 1.2, nlev - 0.5),
    ):
        wind = CartesianWind(zero, zero, zero, np.full(shape, rate))
        stencil, _ = departure_points(wind, tables, dt, iterations=3)
        level = np.asarray(stencil.level)
        assert np.allclose(level[0], top, atol=1e-12), (rate, level[0, 0, 0])
        assert np.allclose(level[-1], bottom, atol=1e-12), (rate, level[-1, 0, 0])


def test_gather_reads_the_half_layer_outside_the_outermost_levels():
    grid, tables = _tables()
    nlev = 6
    profile = 10.0 + 3.0 * np.arange(nlev)
    field = _layered(grid, profile)
    levels = [-0.3, 1.0, 2.0, 3.0, 4.0, nlev - 1 + 0.4]
    out = gather(field, _column_stencil(tables, grid, levels), monotone=False)
    # a linear column is read exactly, including in the two half layers
    assert np.allclose(out[0], 10.0 - 0.9, atol=1e-12)
    assert np.allclose(out[-1], 10.0 + 3.0 * (nlev - 1 + 0.4), atol=1e-12)
    # beyond the interface the read is the interface value, never further
    far = gather(field, _column_stencil(tables, grid, [-3.0, 1, 2, 3, 4, 9.0]),
                 monotone=False)
    assert np.allclose(far[0], 10.0 - 1.5, atol=1e-12)
    assert np.allclose(far[-1], 10.0 + 3.0 * (nlev - 0.5), atol=1e-12)


def test_half_layer_read_is_bounded_by_the_interface_reconstruction():
    """An oscillating column: the one-sided cubic at -1/2 is 5.0, far past
    anything the column's end slope implies; the read is held to
    [interface reconstruction, outermost value]."""
    grid, tables = _tables()
    nlev = 6
    profile = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    field = _layered(grid, profile)
    out = gather(field, _column_stencil(tables, grid, [-0.5, 1, 2, 3, 4, 5.5]),
                 monotone=False)
    top_face = profile[0] + 0.5 * (profile[0] - profile[1])
    bottom_face = profile[-1] + 0.5 * (profile[-1] - profile[-2])
    assert top_face - 1e-12 <= out[0].min() and out[0].max() <= profile[0] + 1e-12
    assert profile[-1] - 1e-12 <= out[-1].min() and out[-1].max() <= bottom_face + 1e-12
    # the limited gather keeps its cell box
    limited = gather(field, _column_stencil(tables, grid, [-0.5, 1, 2, 3, 4, 5.5]),
                     monotone=True)
    assert limited[0].min() >= 0.0 - 1e-12 and limited[0].max() <= 1.0 + 1e-12


@pytest.mark.parametrize("d", [0.001, 0.01, 0.03])
def test_lid_read_is_symmetric_under_zero_mean_vertical_motion(d):
    """The audit's lid asymmetry, on the default stack: level 0 displaced by
    +d and -d in equal halves.  The clamp read -0.5 d (theta_0 - theta_1)
    on the mean (-0.124 K per step at d = 0.001); the interface clamp reads
    the column's curvature only, second order in d."""
    grid, tables = _tables()
    _, theta = _default_theta()
    nlev = theta.shape[0]
    field = _layered(grid, theta)
    rest = list(range(1, nlev))
    up = gather(field, _column_stencil(tables, grid, [+d, *rest]), monotone=False)
    down = gather(field, _column_stencil(tables, grid, [-d, *rest]), monotone=False)
    mean_change = 0.5 * (float(up[0, 0, 0]) + float(down[0, 0, 0])) - theta[0]
    clamped = -0.5 * d * (theta[0] - theta[1])
    assert theta[0] - theta[1] > 100.0
    assert abs(mean_change) < 0.05 * abs(clamped), (mean_change, clamped)


# ---------------------------------------------------------- DYC-1 Eulerian

def _eulerian(nlev):
    transform = SphericalHarmonicTransform.create(5, backend="numpy",
                                                  precision="float64")
    vertical = HybridCoordinate.surface_stretched(nlev, 100.0)
    model = MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=np.zeros(transform.grid.shape), physics=None,
        rotation_rate_s=0.0, diffusion=None,
        semi_implicit=BarotropicSemiImplicit(enabled=False),
        mass_fixer=False, water_fixer=False, positivity_repair=False,
        maximum_cfl=1.0e9,
    )
    pressure = vertical.pressure(np.full(transform.grid.shape, PS),
                                 transform.backend)
    return transform, model, pressure


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_eulerian_boundary_faces_use_the_one_sided_gradient(sign):
    """The top layer's bottom face under descent and the bottom layer's top
    face under ascent are the linear interpolation between the two layers,
    so the top layer warms under descent instead of carrying its own value
    out (the zero-gradient face gave exactly no tendency)."""
    nlev = 40
    transform, model, pressure = _eulerian(nlev)
    p_half, p_full, dp = pressure["p_half"], pressure["p_full"], pressure["dp"]
    _, theta_col = _default_theta()
    theta = np.broadcast_to(theta_col[:, None, None], p_full.shape).copy()
    omega = np.zeros((nlev + 1, *transform.grid.shape))
    omega[1:nlev] = sign * 0.01
    div = model._vertical_scalar_flux_divergence(theta, omega, p_full, p_half)
    if sign > 0:  # descent through interface 1: the top layer
        k, j, n = 0, 1, 1
    else:         # ascent through interface nlev-1: the bottom layer
        k, j, n = nlev - 1, nlev - 1, nlev - 2
    face = theta[k] + (theta[n] - theta[k]) / (p_full[n] - p_full[k]) * (
        p_half[j] - p_full[k])
    flux = omega[j] * face
    want = flux if k == 0 else -flux
    np.testing.assert_allclose(div[k], want, rtol=1e-12)
    # advective tendency of the top layer: -omega_1 (face_1 - theta_0) / dp_0
    if sign > 0:
        tendency = -(div[0] - theta[0] * omega[1]) / dp[0]
        assert float(tendency.min()) > 0.0


# ------------------------------------------------------------------- DYC-3

@pytest.mark.parametrize("builder,nlev", [
    ("surface_stretched", 40), ("pressure_blend", 20), ("jet_refined", 48),
])
def test_level_rate_uses_the_geometric_full_level_spacing(builder, nlev):
    """A uniform interface mass flux of 1 Pa/s moves a parcel one full-level
    spacing in (p_full[k] - p_full[k-1]) seconds, so the interface rate is
    its reciprocal.  The arithmetic mean thickness was 1.7 to 4.1 percent
    off at the worst interface of each shipped stack."""
    vertical = getattr(HybridCoordinate, builder)(nlev, 100.0)
    p_half = vertical.a_half_pa + vertical.b_half * PS
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    omega = np.ones(nlev + 1)
    omega[0] = omega[-1] = 0.0
    rate = level_rate_from_mass_flux(omega, p_full=p_full)
    s_half = np.zeros(nlev + 1)
    s_half[1:nlev] = 1.0 / np.diff(p_full)
    np.testing.assert_allclose(rate, 0.5 * (s_half[:-1] + s_half[1:]),
                               rtol=1e-13)
    arithmetic = 0.5 * (np.diff(p_half)[:-1] + np.diff(p_half)[1:])
    worst = float(np.max(np.abs(np.diff(p_full) / arithmetic - 1.0)))
    assert worst > 0.015  # the defect this pins was real on every stack


def test_level_rate_refuses_a_positional_thickness():
    nlev = 6
    dp = np.full(nlev, 2500.0)
    with pytest.raises(TypeError):
        level_rate_from_mass_flux(np.zeros(nlev + 1), dp)
    assert math.isfinite(float(level_rate_from_mass_flux(
        np.zeros(nlev + 1), p_full=np.cumsum(dp))[0]))
