"""The trajectory fold gate of the semi-Lagrangian core.

The gate reads ``det(I -+ (dt/2) J)`` of the flow Jacobian, which reaches
zero exactly where the trajectory map folds.  It replaced a ceiling on dt
times a NORM of the Jacobian, which cannot tell a fold from a rotation or
a shear and refused a T533 forecast at 66.3 h on a flow the map did not
fold.  These tests pin both halves: what does not fold is admitted, what
folds is refused by name.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from arwen_global.spectral.grid import GaussianGrid
from arwen_global.semilag.cases import _mesh, solid_body_wind
from arwen_global.semilag.options import SemiLagrangianOptions
from arwen_global.semilag.tables import SphericalGridTables
from arwen_global.semilag.trajectory import (
    DEFAULT_MINIMUM_FOLD_DETERMINANT,
    MAXIMUM_TRAJECTORY_ITERATIONS,
    CartesianWind,
    LipschitzDiagnostics,
    cartesian_wind,
    converged_departure_points,
    departure_points,
    lipschitz,
    refuse_trajectory_fold,
)

pytestmark = pytest.mark.cpu_only


def _wind(tables, u, v, rate=None):
    vx, vy, vz = cartesian_wind(u, v, tables, xp=np, dtype=np.float64)
    return CartesianWind(
        np.ascontiguousarray(vx), np.ascontiguousarray(vy),
        np.ascontiguousarray(vz),
        np.zeros_like(u) if rate is None else rate,
    )


def _audit_shear(truncation=127, nlev=6):
    """The audit's CPU case (DYC-2): a pure zonal shear u(y), v = 0."""
    grid = GaussianGrid.create(truncation)
    tables = SphericalGridTables.create(grid, xp=np, dtype=np.float64)
    nlat, nlon = tables.shape
    lat = tables.lat_ext_host[2:nlat + 2]
    y = tables.radius_m * lat
    u1 = (50.0 * np.sin(2.0 * np.pi * y / 2.6e6)
          * np.exp(-((lat / np.deg2rad(60.0)) ** 8)))
    u = np.broadcast_to(u1[None, :, None], (nlev, nlat, nlon)).copy()
    return tables, _wind(tables, u, np.zeros_like(u)), u1, y


def test_the_audit_shear_flow_is_not_refused():
    """DYC-2's demonstration, now admitted.

    A pure shear has a strictly triangular Jacobian, so its fold
    determinant is one wherever it is measured, and the departure search
    converges.  The retired norm gate read 0.861 here and refused it.
    """
    tables, wind, u1, y = _audit_shear()
    dt = 7200.0
    assert float(np.max(np.abs(np.gradient(u1, y)))) == pytest.approx(
        1.2e-4, rel=0.05)
    diag = lipschitz(wind, tables, dt)
    # The norm the retired gate read is still measured, and is past its
    # old ceiling: this is the state that was refused.
    assert diag.lipschitz == pytest.approx(0.861, abs=0.005)
    assert diag.lipschitz > 0.75
    # The fold determinant sees no fold anywhere: one in the plane, and
    # within 0.5 percent of it on the sphere, where the zonal flow's own
    # curvature term u tan(phi) / a meets the shear.
    assert diag.fold_determinant == pytest.approx(1.0, abs=0.005)
    refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)
    _, trajectory = departure_points(wind, tables, dt, iterations=3)
    assert trajectory.move_max_cells < 0.01


@pytest.mark.parametrize("alpha", [0.0, math.pi / 2.0])
def test_a_rigid_rotation_never_folds(alpha):
    """Rotation: the norm reads the rotation rate and refused it, the
    determinant reads at least one on every point, poles included."""
    grid = GaussianGrid.create(31)
    tables = SphericalGridTables.create(grid, xp=np, dtype=np.float64)
    lam, phi = _mesh(grid)
    u0 = 200.0
    u2, v2 = solid_body_wind(lam, phi, u0=u0, alpha=alpha)
    u = np.repeat(u2[None], 4, axis=0)
    v = np.repeat(v2[None], 4, axis=0)
    wind = _wind(tables, u, v)
    omega = u0 / grid.radius_m
    dt = 1.6 / omega  # a norm of 1.6: far past the retired ceiling
    diag = lipschitz(wind, tables, dt)
    assert diag.lipschitz > 1.5
    # The tangent Jacobian is omega sin(phi') times a rotation (phi' the
    # latitude about the rotation axis), so the determinant is
    # 1 + (omega sin(phi') dt / 2)**2 and its minimum, one, sits on the
    # rotation's own equator.
    assert diag.fold_determinant == pytest.approx(1.0, abs=1e-4)
    refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)


def _zonal_compression(amplitude, truncation=31, nlev=4):
    """u = A sin(lambda) cos(phi): du/dx = A cos(lambda) / a, a flow that
    converges on one meridian and diverges on the opposite one."""
    grid = GaussianGrid.create(truncation)
    tables = SphericalGridTables.create(grid, xp=np, dtype=np.float64)
    lam, phi = _mesh(grid)
    u2 = amplitude * np.sin(lam) * np.cos(phi)
    u = np.repeat(u2[None], nlev, axis=0)
    return grid, tables, _wind(tables, u, np.zeros_like(u))


def test_the_fold_determinant_is_the_closed_form_of_a_compression():
    grid, tables, wind = _zonal_compression(100.0)
    rate = 100.0 / grid.radius_m  # max |du/dx| on the equator
    dt = 1.0 / rate  # (dt/2) |du/dx| = 0.5 at the extremes
    diag = lipschitz(wind, tables, dt)
    # arrival side 1 - (dt/2) du/dx, departure side 1 + (dt/2) du/dx: both
    # reach 0.5 at their own meridian, and the finite difference on 64
    # longitudes reads du/dx low by about sin(dlam)/dlam
    assert diag.fold_determinant_arrival == pytest.approx(0.5, abs=0.01)
    assert diag.fold_determinant_departure == pytest.approx(0.5, abs=0.01)
    refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)


def test_a_folding_flow_is_refused_by_name():
    grid, tables, wind = _zonal_compression(100.0)
    rate = 100.0 / grid.radius_m
    dt = 3.0 / rate  # (dt/2) |du/dx| = 1.5: past the fold on both sides
    diag = lipschitz(wind, tables, dt)
    assert diag.fold_determinant < 0.0
    with pytest.raises(ValueError, match="trajectory map folds") as caught:
        refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)
    assert "share a departure point" in str(caught.value)
    # and the margin refuses short of the fold itself
    dt = 1.7 / rate  # determinant 0.15, under the 0.2 floor
    with pytest.raises(ValueError, match="trajectory map folds"):
        refuse_trajectory_fold(lipschitz(wind, tables, dt),
                               DEFAULT_MINIMUM_FOLD_DETERMINANT)


def test_the_arrival_side_reads_the_extrapolated_wind():
    """The search reads V_ex at the arrival point; a fold there is a fold
    of the map it solves, though the current wind is at rest."""
    grid, tables, compression = _zonal_compression(100.0)
    still = CartesianWind(*[a * 0.0 for a in compression.arrays()])
    rate = 100.0 / grid.radius_m
    diag = lipschitz(still, tables, 3.0 / rate, extrapolated=compression)
    assert diag.fold_determinant_departure == 1.0
    assert diag.fold_determinant_arrival < 0.0
    assert diag.fold_side == "arrival"
    with pytest.raises(ValueError, match="arrival"):
        refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)
    with pytest.raises(ValueError, match="extrapolated wind"):
        lipschitz(still, tables, 300.0,
                  extrapolated=CartesianWind(*[a[:2] for a in compression.arrays()]))


def test_the_fold_determinant_does_not_depend_on_its_chunking():
    grid = GaussianGrid.create(21)
    tables = SphericalGridTables.create(grid, xp=np, dtype=np.float64)
    nlev = 9
    lam = np.asarray(grid.lon_rad)[None, None, :]
    phi = np.asarray(grid.lat_rad)[None, :, None]
    k = np.arange(nlev)[:, None, None] / (nlev - 1)
    shape = (nlev, grid.nlat, grid.nlon)
    u = np.broadcast_to(60.0 * np.sin(lam) * np.cos(phi) * (0.3 + k), shape).copy()
    v = np.broadcast_to(20.0 * np.sin(2.0 * lam) * np.cos(phi) * (1.0 - k), shape).copy()
    rate = np.broadcast_to(4e-4 * np.sin(math.pi * k) * np.cos(lam), shape).copy()
    wind = _wind(tables, u, v, rate)
    whole = lipschitz(wind, tables, 3600.0, level_chunk=nlev)
    assert whole.fold_determinant < 1.0
    for chunk in (1, 2, 4):
        part = lipschitz(wind, tables, 3600.0, level_chunk=chunk)
        assert part.fold_determinant_arrival == pytest.approx(
            whole.fold_determinant_arrival, rel=1e-12)
        assert part.fold_determinant_departure == pytest.approx(
            whole.fold_determinant_departure, rel=1e-12)


def test_the_fold_determinant_needs_no_reference_length():
    """A level-to-metres length is a diagonal similarity of J and cannot
    reach a determinant; the norms move with it, the gate does not."""
    grid = GaussianGrid.create(21)
    tables = SphericalGridTables.create(grid, xp=np, dtype=np.float64)
    nlev = 6
    lam = np.asarray(grid.lon_rad)[None, None, :]
    phi = np.asarray(grid.lat_rad)[None, :, None]
    k = np.arange(nlev)[:, None, None] / (nlev - 1)
    shape = (nlev, grid.nlat, grid.nlon)
    u = np.broadcast_to(40.0 * np.cos(phi) * (0.2 + k), shape).copy()
    rate = np.broadcast_to(3e-4 * np.sin(math.pi * k) * np.cos(lam), shape).copy()
    wind = _wind(tables, u, np.zeros_like(u), rate)
    a = lipschitz(wind, tables, 1800.0)
    b = lipschitz(wind, tables, 1800.0, reference_length_m=7.0e5)
    assert a.lipschitz_mixed != pytest.approx(b.lipschitz_mixed, rel=1e-3)
    assert a.fold_determinant == pytest.approx(b.fold_determinant, rel=1e-12)


def test_the_gate_floor_is_validated():
    diag = LipschitzDiagnostics(
        dt_s=300.0, reference_length_m=52000.0, lipschitz=2.0,
        lipschitz_frobenius=2.0, lipschitz_horizontal=2.0,
        jacobian_spectral_s=0.01, jacobian_frobenius_s=0.01,
        jacobian_horizontal_s=0.01,
        fold_determinant_arrival=0.9, fold_determinant_departure=0.95,
    )
    # a norm of 2.0 is not a fold
    refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)
    refuse_trajectory_fold(diag, 0.0)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
            refuse_trajectory_fold(diag, bad)
    nan = LipschitzDiagnostics(
        dt_s=300.0, reference_length_m=52000.0, lipschitz=0.1,
        lipschitz_frobenius=0.1, lipschitz_horizontal=0.1,
        jacobian_spectral_s=0.0, jacobian_frobenius_s=0.0,
        jacobian_horizontal_s=0.0,
        fold_determinant_arrival=float("nan"),
    )
    with pytest.raises(ValueError, match="folds"):
        refuse_trajectory_fold(nan, DEFAULT_MINIMUM_FOLD_DETERMINANT)


def test_the_receipt_carries_the_fold_and_the_norms():
    tables, wind, _, _ = _audit_shear(truncation=21, nlev=4)
    row = lipschitz(wind, tables, 300.0).as_dict()
    for key in ("semilag_fold_determinant", "semilag_fold_determinant_arrival",
                "semilag_fold_determinant_departure", "semilag_lipschitz",
                "semilag_lipschitz_balanced"):
        assert key in row


def test_the_model_default_is_the_gate_default():
    from arwen_global.config import DEFAULT_MINIMUM_FOLD_DETERMINANT as CONFIG
    from arwen_global.dynamics import MoistHybridModel

    assert CONFIG == DEFAULT_MINIMUM_FOLD_DETERMINANT
    field = MoistHybridModel.__dataclass_fields__["minimum_fold_determinant"]
    assert field.default == DEFAULT_MINIMUM_FOLD_DETERMINANT
    assert not hasattr(MoistHybridModel, "maximum_lipschitz")


def test_an_unconverged_search_is_retried_at_the_most_iterations():
    """A flow the fold gate admits whose three-pass search does not
    converge is searched again at eight passes before it is refused."""
    assert MAXIMUM_TRAJECTORY_ITERATIONS == 8
    with pytest.raises(ValueError, match="1..8"):
        SemiLagrangianOptions(trajectory_iterations=9)
    tables, wind, _, _ = _audit_shear(truncation=42, nlev=4)
    dt = 4.0e4  # a norm of 4.4, a fold determinant of 0.88
    diag = lipschitz(wind, tables, dt)
    refuse_trajectory_fold(diag, DEFAULT_MINIMUM_FOLD_DETERMINANT)
    # the shipped [semilag] trajectory_convergence_cells: three passes
    # leave 0.142 of a cell, eight leave 0.0055
    limit = SemiLagrangianOptions().trajectory_convergence_cells
    _, three = departure_points(wind, tables, dt, iterations=3)
    _, eight = departure_points(wind, tables, dt, iterations=8)
    assert three.move_max_cells > limit > eight.move_max_cells
    stencil, used = converged_departure_points(
        wind, tables, dt, iterations=3, limit_cells=limit)
    assert used.iterations == 8
    assert used.move_max_cells == eight.move_max_cells
    # a converged search never reaches the retry
    _, plain = converged_departure_points(
        wind, tables, 300.0, iterations=3, limit_cells=0.01)
    assert plain.iterations == 3
    # and a search eight passes cannot converge is still refused by name
    with pytest.raises(ValueError, match="did not converge"):
        converged_departure_points(
            wind, tables, dt, iterations=3,
            limit_cells=0.1 * eight.move_max_cells)


_CONFIG = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "fold-gate"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"
[grid]
truncation = 10
[time]
dt_s = 300.0
duration_s = 3000.0
integrator = "{integrator}"
{time_extra}
[vertical]
coordinate = "pressure_blend"
nlev = 8
[physics]
mode = "none"
"""


def _load(tmp_path, integrator="sl_si", time_extra=""):
    from arwen_global.config import load_config

    path = Path(tmp_path) / "fold.toml"
    path.write_text(_CONFIG.format(integrator=integrator, time_extra=time_extra),
                    encoding="utf-8")
    return load_config(path)


def test_the_config_reads_the_fold_floor(tmp_path):
    cfg = _load(tmp_path, time_extra="minimum_fold_determinant = 0.1")
    assert cfg.minimum_fold_determinant == 0.1
    assert cfg.config_identity["minimum_fold_determinant"] == 0.1
    for bad in ("-0.1", "1.0"):
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
            _load(tmp_path, time_extra=f"minimum_fold_determinant = {bad}")
    with pytest.raises(ValueError, match="minimum_fold_determinant is read only"):
        _load(tmp_path, integrator="imex_ssp3",
              time_extra="minimum_fold_determinant = 0.1")


def test_the_retired_key_is_a_no_op_at_its_old_default_and_refused_elsewhere(tmp_path):
    plain = _load(tmp_path)
    old = _load(tmp_path, time_extra="maximum_lipschitz = 0.75")
    assert old.config_hash == plain.config_hash
    with pytest.raises(ValueError, match="no longer read"):
        _load(tmp_path, time_extra="maximum_lipschitz = 0.6")
    with pytest.raises(ValueError, match="maximum_lipschitz is read only"):
        _load(tmp_path, integrator="imex_ssp3", time_extra="maximum_lipschitz = 0.75")


def test_a_default_config_keeps_its_hash(tmp_path):
    """The fold gate moves no config identity: the reference preset's hash
    is the one computed at 7548983, before the fold gate.  The two native
    presets' identities moved with the dated greenhouse gases (GP-8: the
    native options' trace_co2_ppm defaults to None, the dated table), not
    with the gate, and again with the stratospheric floor's retirement from
    the default (GP-5); their digests below are the merged tree's."""
    from arwen_global.config import load_config

    root = Path(__file__).resolve().parents[1] / "src" / "arwen_global" / "configs"
    for name, digest in (
        ("arwen_global_t255_quickstart.toml",
         "ce5629409342af640456f621304ae4d09b91f1dbc0171198d45b8757fba8c3fb"),
        ("arwen_global_gdas_t533_native_24h.toml",
         "f6eccb32f359d22e26a31bf90d944b93b729f1fc1f5d3722b077885b65878454"),
        ("arwen_global_gdas_t255_native_sl_si_24h.toml",
         "2fe9a5683d3c5038a56c560f3a82b99b88aa804ac1f50b296374f34b7ff32c2d"),
    ):
        assert load_config(root / name).config_hash == digest, name
    identity = _load(tmp_path).config_identity
    assert identity["maximum_lipschitz"] == 0.75
    assert "minimum_fold_determinant" not in identity
    imex = _load(tmp_path, integrator="imex_ssp3").config_identity
    assert "maximum_lipschitz" not in imex
    assert "minimum_fold_determinant" not in imex


def test_the_receipt_counts_retried_steps_and_keeps_the_smallest_fold():
    from arwen_global.checkpoint import normalize_trackers
    from arwen_global.runner import _update_trackers, fresh_supplementary

    trackers = normalize_trackers()
    supplementary = fresh_supplementary()
    assert supplementary["minimum_semilag_fold_determinant"] == 1.0
    assert supplementary["semilag_trajectory_retried_steps"] == 0.0
    base = {
        "spectral_cfl": 0.0, "mass_fixer_log_offset": 0.0,
        "global_water_fixer_kg_m2": 0.0,
        "maximum_repaired_negative_mixing_ratio": 0.0,
        "maximum_repaired_negative_number_per_kg": 0.0,
        "semi_implicit_max_divergence_increment_s1": 0.0,
    }
    for fold, retried in ((0.9, 0.0), (0.6, 1.0), (0.8, 1.0), (0.95, 0.0)):
        _update_trackers(trackers, supplementary, {
            **base, "semilag_fold_determinant": fold,
            "semilag_trajectory_retried": retried,
        })
    assert supplementary["minimum_semilag_fold_determinant"] == 0.6
    assert supplementary["semilag_trajectory_retried_steps"] == 2.0
