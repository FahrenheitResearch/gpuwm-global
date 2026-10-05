"""Departure points on the sphere, and the diagnostics that gate them.

The trajectory is solved in geocentric Cartesian coordinates.  There is no
rotation matrix, no local basis and therefore no polar special case: the
arrival point is a unit vector, the wind is three Cartesian components of a
tangent field, the trial departure point is the chord from the arrival point
projected back onto the sphere, and the vertical coordinate is the continuous
full-level index moved by the level-index rate.

The iteration, for arrival point ``r_A`` and level index ``k_A``::

    r_D^(m+1) = normalize( r_A - (dt/2) [ V_ex(r_A) + V^n(r_D^(m)) ] / a )
    k_D^(m+1) = k_A - (dt/2) [ s_ex(r_A,k_A) + s^n(r_D^(m), k_D^(m)) ]
    r_D^(0)   = normalize( r_A - dt V^n(r_A) / a )
    k_D^(0)   = k_A - dt s^n(r_A, k_A)

``V_ex`` and ``s_ex`` are the extrapolated fields, read at the arrival point
and so needing no interpolation at all; a caller with no previous time level
(the first step of a fresh run) passes none and gets ``V^n`` at the arrival
point in their place, which is the non-extrapolated two-time-level start-up.

Two diagnostics ship with the solver because the two ways a semi-Lagrangian
trajectory goes wrong are both silent:

*   :func:`convergence` measures the last iteration's move.  An unconverged
    trajectory reads as a Rossby phase error and mislocated cyclones and
    produces no other symptom, so the iteration count is measured rather
    than trusted.
*   :func:`lipschitz` measures the flow deformation, and
    :func:`refuse_trajectory_fold` gates on the one part of it that folds
    the trajectory map: ``det(I -+ (dt/2) J)``.  At zero two arrival points
    share a departure point and the interpolation samples a folded field.
    The model does not blow up when this happens.  It silently mislocates.
    The deformation NORMS ride beside it in every receipt and are not
    gated, because a norm reads vorticity and shear, which never fold.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .interpolate import Stencil, _is_cupy
from .tables import BandWindow, HaloEscape, SphericalGridTables


@dataclass(frozen=True)
class CartesianWind:
    """The advecting flow, in the form the trajectory reads it.

    ``vx, vy, vz`` are the geocentric Cartesian components of the horizontal
    wind in m/s at full levels; ``level_rate`` is the vertical velocity in
    continuous level indices per second, also at full levels.
    """

    vx: Any
    vy: Any
    vz: Any
    level_rate: Any

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.vx.shape)  # type: ignore[return-value]

    def arrays(self) -> tuple:
        return (self.vx, self.vy, self.vz, self.level_rate)


def _geometry(tables: SphericalGridTables, nlev: int, *, xp, dtype,
              rows: slice | None = None):
    """The grid's trigonometry, shaped to broadcast over a volume.

    ``rows`` cuts the latitude rows of a band out of the WHOLE grid's
    sines and cosines, so a band reads the very values the whole-grid call
    reads at those rows rather than a second evaluation of them.
    """
    nlat, nlon = tables.shape
    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)
    lon = xp.asarray(np.arange(nlon) * tables.dlam, dtype=dtype)
    sin_lat = xp.sin(lat)[None, :, None]
    cos_lat = xp.cos(lat)[None, :, None]
    if rows is not None:
        sin_lat = sin_lat[:, rows]
        cos_lat = cos_lat[:, rows]
    sin_lon = xp.sin(lon)[None, None, :]
    cos_lon = xp.cos(lon)[None, None, :]
    return sin_lat, cos_lat, sin_lon, cos_lon


def cartesian_wind(u, v, tables: SphericalGridTables, *, xp=None, dtype=None,
                   rows: slice | None = None):
    """Local east-north wind as geocentric Cartesian components.

    ``rows`` names the latitude rows ``u`` and ``v`` hold when they are a
    band rather than the whole grid.
    """
    xp = tables.xp if xp is None else xp
    dtype = tables.dtype if dtype is None else dtype
    nlev = int(u.shape[0])
    sph, cph, sla, cla = _geometry(tables, nlev, xp=xp, dtype=dtype,
                                   rows=rows)
    vx = -u * sla - v * sph * cla
    vy = u * cla - v * sph * sla
    vz = v * cph + xp.zeros_like(u)
    return vx, vy, vz


def local_wind(vx, vy, vz, tables: SphericalGridTables, *, xp=None, dtype=None):
    """Geocentric Cartesian components back to the local east-north wind."""
    xp = tables.xp if xp is None else xp
    dtype = tables.dtype if dtype is None else dtype
    nlev = int(vx.shape[0])
    sph, cph, sla, cla = _geometry(tables, nlev, xp=xp, dtype=dtype)
    u = -vx * sla + vy * cla
    v = -vx * sph * cla - vy * sph * sla + vz * cph
    return u, v


def level_rate_from_mass_flux(omega_half, *, p_full, xp=None):
    """The vertical rate in level indices per second, at full levels.

    ``omega_half`` is the eta-coordinate vertical mass flux ``etadot dp/deta``
    in Pa/s at the ``nlev+1`` interfaces, zero at BOTH boundaries, which is
    what :func:`arwen_global.vertical.continuity` produces.  Dividing
    it by the pressure distance per unit level index gives the rate in level
    indices, and the zero boundary values mean no trajectory leaves the
    model through the lid or the ground by construction of the interior flow.

    The distance per unit index at interior interface ``k`` is
    ``p_full[k] - p_full[k-1]``: the gather interpolates in the continuous
    level index with full level ``k`` at index ``k``, and the full levels
    sit at the GEOMETRIC layer means ``sqrt(p_half[k] p_half[k+1])``
    (vertical.HybridCoordinate.pressure).  Until DYC-3 this divided by the
    arithmetic mean thickness ``(dp[k-1] + dp[k]) / 2``, which is the right
    distance only for full levels at arithmetic layer centres: vertical
    displacements were 1.7 percent too short at the worst interface of the
    default 40-level stack (2.1 hPa), 4.1 percent on the 20-level
    pressure_blend stack and up to 3.8 percent too LONG at 118.7 hPa on
    jet_refined.  ``p_full`` is keyword-only so a caller still handing in
    layer thicknesses fails loudly instead of being read as pressures.
    """
    xp = np if xp is None else xp
    nlev = int(p_full.shape[0])
    if int(omega_half.shape[0]) != nlev + 1:
        raise ValueError(
            f"omega_half must have {nlev + 1} interfaces for {nlev} layers, "
            f"got {int(omega_half.shape[0])}"
        )
    s_half = xp.zeros_like(omega_half)
    distance = p_full[1:] - p_full[:-1]
    s_half[1:nlev] = omega_half[1:nlev] / distance
    return 0.5 * (s_half[:nlev] + s_half[1:nlev + 1])


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    iterations: int
    dt_s: float
    #: largest last-iteration move, metres
    move_max_m: float
    #: the same, in local meridional grid lengths (the reference length that
    #: does not collapse at the pole the way the zonal one does)
    move_max_cells: float
    #: largest last-iteration move of the vertical coordinate, level indices
    move_max_levels: float
    #: largest horizontal displacement of the finished trajectory, metres
    displacement_max_m: float
    displacement_max_cells: float
    displacement_max_levels: float

    def as_dict(self) -> dict:
        return {
            "semilag_trajectory_iterations": self.iterations,
            "semilag_trajectory_dt_s": self.dt_s,
            "semilag_trajectory_move_max_m": self.move_max_m,
            "semilag_trajectory_move_max_cells": self.move_max_cells,
            "semilag_trajectory_move_max_levels": self.move_max_levels,
            "semilag_displacement_max_m": self.displacement_max_m,
            "semilag_displacement_max_cells": self.displacement_max_cells,
            "semilag_displacement_max_levels": self.displacement_max_levels,
        }


def meridional_grid_length_m(tables: SphericalGridTables) -> np.ndarray:
    """``a * dphi`` per latitude row, from the pole-extended table.

    The convergence gate divides by this rather than by the local ZONAL
    spacing.  The zonal spacing collapses to 326 m on the outermost T255 ring
    and 75 m at T533, so a threshold expressed in local zonal cells would
    refuse a converged trajectory over 0.26 percent of the grid and pass a
    diverged one everywhere else.  The meridional spacing of a Gaussian grid
    is uniform to a few percent, and at T255 it is 52.1 km, the same length
    the equatorial zonal cell has.
    """
    lat_ext = tables.lat_ext_host
    nlat = tables.nlat
    upper = lat_ext[3:nlat + 3]
    lower = lat_ext[1:nlat + 1]
    return tables.radius_m * 0.5 * (upper - lower)


def departure_points(
    wind: CartesianWind,
    tables: SphericalGridTables,
    dt_s: float,
    *,
    iterations: int = 3,
    extrapolated: CartesianWind | None = None,
    window: BandWindow | None = None,
    defer_escape: bool = False,
) -> tuple[Stencil, TrajectoryDiagnostics]:
    """Solve for the departure point of every arrival point of the grid.

    With a ``window`` the arrival points are the band's rows only: ``wind``
    holds the band and its halo (``window.source``) and ``extrapolated``
    the band's own rows, because the trajectory reads the extrapolated
    flow at the arrival point alone.  A search that reads a row the band
    does not hold is refused by name (:class:`HaloEscape`); with
    ``defer_escape`` the window's flag is left for the caller to read once
    (tables.BandWindow.raise_if_escaped).
    """
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    shape = wind.shape
    nlev, nrows, nlon = shape
    nlat = tables.nlat
    held = tables.nlat if window is None else window.source_rows
    if (nrows, nlon) != (held, tables.nlon):
        raise ValueError(
            f"the wind is {(nrows, nlon)} where {(held, tables.nlon)} rows "
            f"were expected on a {tables.shape} grid"
        )
    if nlev < 4:
        raise ValueError(f"nlev >= 4 is required, got {nlev}")
    if int(iterations) < 1:
        raise ValueError(
            "the departure-point search needs at least one iteration; zero "
            "would ship the explicit first guess, which is first order in dt "
            "and is not a trajectory"
        )
    out_shape = shape if window is None else window.arrival_shape(nlev)
    ex = wind if extrapolated is None else extrapolated
    if window is not None and extrapolated is None:
        ex = CartesianWind(*(
            xp.ascontiguousarray(field[:, window.local])
            for field in wind.arrays()
        ))
    for name, array, want in (("wind", wind, shape),
                              ("extrapolated", ex, out_shape)):
        for field in array.arrays():
            if tuple(int(s) for s in field.shape) != want:
                raise ValueError(f"{name} fields must all be {want}")
            if np.dtype(field.dtype) != dtype:
                raise ValueError(
                    f"{name} is {np.dtype(field.dtype)} where the grid tables "
                    f"were built for {dtype}"
                )

    if _is_cupy(xp):
        from ._cuda import get_kernel

        band = "_band" if window is not None else ""
        name = (f"sl_departure{band}_f32" if dtype == np.float32
                else f"sl_departure{band}_f64")
        kernel = get_kernel(name)
        xi = xp.empty(out_shape, dtype=dtype)
        phi = xp.empty(out_shape, dtype=dtype)
        lev = xp.empty(out_shape, dtype=dtype)
        move_m = xp.empty(out_shape, dtype=dtype)
        move_k = xp.empty(out_shape, dtype=dtype)
        npoints = out_shape[0] * out_shape[1] * out_shape[2]
        threads = 256
        blocks = (npoints + threads - 1) // threads
        rowoff = tables.rowoff if window is None else window.rowoff
        arguments = (
            xp.ascontiguousarray(wind.vx), xp.ascontiguousarray(wind.vy),
            xp.ascontiguousarray(wind.vz),
            xp.ascontiguousarray(wind.level_rate),
            xp.ascontiguousarray(ex.vx), xp.ascontiguousarray(ex.vy),
            xp.ascontiguousarray(ex.vz),
            xp.ascontiguousarray(ex.level_rate),
            tables.lat_ext, tables.mlut, rowoff, tables.rowshift,
            xi, phi, lev, move_m, move_k,
            np.int32(int(iterations)), scalar(dt_s),
            scalar(tables.radius_m), scalar(tables.dlam),
            np.int32(tables.lookup_bins), np.int32(nlev), np.int32(nlat),
            np.int32(nlon), scalar(tables.inv_dlam),
            scalar(tables.lut_scale),
        )
        if window is not None:
            if not defer_escape:
                window.clear_escape()
            arguments = arguments + (
                np.int32(window.source_rows), np.int32(window.arrival_rows),
                np.int32(window.arrival.start), np.int32(window.lead),
                window.escape,
            )
        kernel((blocks,), (threads,), arguments)
        if window is not None and not defer_escape:
            window.raise_if_escaped("departure-point search")
    else:
        xi, phi, lev, move_m, move_k = _departure_numpy(
            wind, ex, tables, dt_s, iterations=int(iterations), window=window
        )

    stencil = Stencil(xi=xi, phi=phi, level=lev, tables=tables, window=window)
    diagnostics = _trajectory_diagnostics(
        stencil, move_m, move_k, tables, dt_s, int(iterations)
    )
    return stencil, diagnostics


def _trajectory_diagnostics(
    stencil: Stencil, move_m, move_k, tables: SphericalGridTables,
    dt_s: float, iterations: int,
) -> TrajectoryDiagnostics:
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    nlev, nrows, nlon = stencil.shape
    nlat = tables.nlat
    rows = (slice(0, nlat) if stencil.window is None
            else stencil.window.arrival)
    cell = xp.asarray(meridional_grid_length_m(tables), dtype=dtype)
    cell = cell[None, :, None][:, rows]

    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2],
                     dtype=dtype)[None, :, None][:, rows]
    index = xp.arange(nlon, dtype=dtype)[None, None, :]
    lev0 = xp.arange(nlev, dtype=dtype)[:, None, None]
    # Great-circle distance between arrival and departure, through the CHORD.
    # The arccos of the dot product loses half its digits as the angle goes to
    # zero, and reads 0.095 m of displacement for a wind that is exactly zero
    # at T21 in float64, which would put a floor under every convergence
    # measurement this diagnostic is here to make.
    dlam = dtype.type(tables.dlam)
    dphi = dtype.type(0.5) * (stencil.phi - lat)
    dlon = dtype.type(0.5) * (stencil.xi - index) * dlam
    hav = (xp.sin(dphi) ** 2
           + xp.cos(lat) * xp.cos(stencil.phi) * xp.sin(dlon) ** 2)
    dist = (dtype.type(2.0 * tables.radius_m)
            * xp.arcsin(xp.sqrt(xp.clip(hav, dtype.type(0.0),
                                        dtype.type(1.0)))))

    return TrajectoryDiagnostics(
        iterations=int(iterations),
        dt_s=float(dt_s),
        move_max_m=float(xp.max(move_m)),
        move_max_cells=float(xp.max(move_m / cell)),
        move_max_levels=float(xp.max(move_k)),
        displacement_max_m=float(xp.max(dist)),
        displacement_max_cells=float(xp.max(dist / cell)),
        displacement_max_levels=float(xp.max(xp.abs(stencil.level - lev0))),
    )


def _trilinear_numpy(fields, lam, phi, lev, tables: SphericalGridTables,
                     window: BandWindow | None = None):
    """The trilinear gather the search uses, as the numpy specification."""
    nlev = int(fields[0].shape[0])
    nlat, nlon = tables.shape
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    lat_ext = tables.lat_ext_host.astype(dtype)
    rowoff = tables.rowoff_host if window is None else window.rowoff_host
    rowshift = tables.rowshift_host
    half_pi = scalar(np.pi / 2.0)

    p = np.clip(phi, -half_pi, half_pi)
    xk = np.clip(lev, scalar(0.0), scalar(nlev - 1))
    xi = lam * scalar(tables.inv_dlam)
    fi = np.floor(xi)
    ax = (xi - fi).astype(dtype)
    c0 = fi.astype(np.int64) % nlon
    c1 = (c0 + 1) % nlon
    d0 = (c0 + nlon // 2) % nlon
    d1 = (c1 + nlon // 2) % nlon

    m0 = np.clip(
        np.searchsorted(lat_ext.astype(np.float64), p.astype(np.float64),
                        side="right") - 1,
        1, nlat + 1,
    )
    la = lat_ext[m0]
    lb = lat_ext[m0 + 1]
    ay = ((p - la) / (lb - la)).astype(dtype)
    if window is not None and bool(
            (rowoff[m0] < 0).any() or (rowoff[m0 + 1] < 0).any()):
        raise HaloEscape(window, "departure-point search")
    r0 = rowoff[m0] // nlon
    r1 = rowoff[m0 + 1] // nlon
    s0 = rowshift[m0] > 0
    s1 = rowshift[m0 + 1] > 0
    i00 = r0 * nlon + np.where(s0, d0, c0)
    i01 = r0 * nlon + np.where(s0, d1, c1)
    i10 = r1 * nlon + np.where(s1, d0, c0)
    i11 = r1 * nlon + np.where(s1, d1, c1)

    k0 = np.clip(np.floor(xk).astype(np.int64), 0, nlev - 2)
    az = (xk - k0).astype(dtype)
    plane = (nlat if window is None else window.source_rows) * nlon
    bx, by, bz = 1.0 - ax, 1.0 - ay, 1.0 - az

    out = []
    for field in fields:
        flat = np.ascontiguousarray(field).reshape(-1)
        base0 = k0 * plane
        base1 = base0 + plane
        t0 = (by * (bx * flat[base0 + i00] + ax * flat[base0 + i01])
              + ay * (bx * flat[base0 + i10] + ax * flat[base0 + i11]))
        t1 = (by * (bx * flat[base1 + i00] + ax * flat[base1 + i01])
              + ay * (bx * flat[base1 + i10] + ax * flat[base1 + i11]))
        out.append((bz * t0 + az * t1).astype(dtype))
    return out


def _departure_numpy(wind, ex, tables, dt_s, *, iterations, window=None):
    nlev = int(wind.shape[0])
    nlat, nlon = tables.shape
    rows = slice(0, nlat) if window is None else window.arrival
    # The arrival rows' own values of the advecting flow: the whole field
    # on the whole grid, the band's rows of the held rows in a window.
    own = (slice(None) if window is None else window.local)
    shape = (nlev, rows.stop - rows.start, nlon)
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    lat = tables.lat_ext_host[2:nlat + 2].astype(dtype)
    lon = (np.arange(nlon) * tables.dlam).astype(dtype)
    sph = np.sin(lat)[None, :, None][:, rows]
    cph = np.cos(lat)[None, :, None][:, rows]
    sla = np.sin(lon)[None, None, :]
    cla = np.cos(lon)[None, None, :]
    ax = np.broadcast_to(cph * cla, shape).astype(dtype)
    ay = np.broadcast_to(cph * sla, shape).astype(dtype)
    az = np.broadcast_to(np.broadcast_to(sph, (1, shape[1], 1)) + 0.0 * cla,
                         shape).astype(dtype)
    ak = np.broadcast_to(np.arange(nlev, dtype=dtype)[:, None, None],
                         shape).astype(dtype)

    radius = scalar(tables.radius_m)
    inv_r = scalar(1.0) / radius
    dt = scalar(dt_s)
    hdt = scalar(0.5) * dt

    qx = ax - dt * wind.vx[:, own] * inv_r
    qy = ay - dt * wind.vy[:, own] * inv_r
    qz = az - dt * wind.vz[:, own] * inv_r
    norm = np.sqrt(qx * qx + qy * qy + qz * qz)
    qx, qy, qz = qx / norm, qy / norm, qz / norm
    # Clamped at the lid and ground interfaces, index -1/2 and nlev - 1/2
    # (DYC-1); the kernel's sl_departure_body is the same statement.
    klo, khi = scalar(-0.5), scalar(nlev - 0.5)
    qk = np.clip(ak - dt * wind.level_rate[:, own], klo, khi)

    move_m = np.zeros(shape, dtype=dtype)
    move_k = np.zeros(shape, dtype=dtype)
    for _ in range(iterations):
        lam = np.arctan2(qy, qx)
        phi = np.arctan2(qz, np.sqrt(qx * qx + qy * qy))
        w = _trilinear_numpy(
            [wind.vx, wind.vy, wind.vz, wind.level_rate], lam, phi, qk, tables,
            window,
        )
        px = ax - hdt * (ex.vx + w[0]) * inv_r
        py = ay - hdt * (ex.vy + w[1]) * inv_r
        pz = az - hdt * (ex.vz + w[2]) * inv_r
        norm = np.sqrt(px * px + py * py + pz * pz)
        px, py, pz = px / norm, py / norm, pz / norm
        pk = np.clip(ak - hdt * (ex.level_rate + w[3]), klo, khi)
        chord = np.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
        move_m = radius * 2.0 * np.arcsin(
            np.clip(0.5 * chord, scalar(0.0), scalar(1.0)))
        move_k = np.abs(pk - qk)
        qx, qy, qz, qk = px, py, pz, pk

    xi = (np.arctan2(qy, qx) * scalar(tables.inv_dlam)).astype(dtype)
    phi = np.arctan2(qz, np.sqrt(qx * qx + qy * qy)).astype(dtype)
    return xi, phi, qk.astype(dtype), move_m.astype(dtype), move_k.astype(dtype)


# ---------------------------------------------------------------------------
# The Lipschitz diagnostic.


def _sym3_max_eigenvalue(xp, a00, a01, a02, a11, a12, a22):
    """Largest eigenvalue of a symmetric 3x3, in closed form.

    Eleven million singular value decompositions a step is not a diagnostic,
    it is a second model.  The largest singular value of J is the square root
    of the largest eigenvalue of J^T J, and a symmetric 3x3 has a closed-form
    spectrum through the trigonometric solution of its characteristic cubic.

    The matrix is SCALED to unit trace before the cubic is solved and the
    eigenvalue scaled back afterwards.  The unscaled form divides by the cube
    of a deviation that is 1e-6 on a real atmospheric Jacobian in float32,
    and 1e-18 underflows to exactly zero there, so a point whose deviation
    vanished returned 0/0 and one NaN made the whole reduction NaN.  That is
    a diagnostic reporting "not a number" for a flow with a perfectly good
    deformation, which is worse than reporting nothing.
    """
    trace = a00 + a11 + a22
    # A RELATIVE floor, not an absolute one.  The matrix below is scaled to
    # unit trace, so its deviation p is O(1) unless the point is isotropic,
    # and eps is the smallest deviation that means anything there.  An
    # absolute floor of 1e-30 cubes to 1e-90 and underflows to exactly zero
    # in float32, which turned an isotropic point into 0/0 and made one NaN
    # poison the whole reduction.
    eps = float(np.finfo(np.dtype(xp.asarray(trace).dtype)).eps)
    tiny = xp.asarray(eps, dtype=trace.dtype)
    scale = xp.maximum(trace, tiny)
    inv = 1.0 / scale
    a00, a01, a02 = a00 * inv, a01 * inv, a02 * inv
    a11, a12, a22 = a11 * inv, a12 * inv, a22 * inv
    q = (a00 + a11 + a22) / 3.0
    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q
    p2 = (b00 * b00 + b11 * b11 + b22 * b22
          + 2.0 * (a01 * a01 + a02 * a02 + a12 * a12)) / 6.0
    p = xp.sqrt(xp.maximum(p2, 0.0))
    safe = xp.maximum(p, tiny)
    det = (b00 * (b11 * b22 - a12 * a12)
           - a01 * (a01 * b22 - a12 * a02)
           + a02 * (a01 * a12 - b11 * a02))
    r = xp.clip(det / (2.0 * safe * safe * safe), -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    return scale * (q + 2.0 * p * xp.cos(phi))


def _sym2_max_eigenvalue(xp, a00, a01, a11):
    half = 0.5 * (a00 + a11)
    rad = xp.sqrt(xp.maximum(0.25 * (a00 - a11) ** 2 + a01 * a01, 0.0))
    return half + rad


def _reflected_rows(xp, field, nlon):
    """``field`` with one reflected row added at each pole."""
    top = xp.roll(field[:, :1, :], nlon // 2, axis=2)
    bot = xp.roll(field[:, -1:, :], nlon // 2, axis=2)
    return xp.concatenate([top, field, bot], axis=1)


def _band_rows_with_neighbours(xp, field, window: BandWindow, nlon):
    """The band's rows of a held-rows ``field`` with one neighbour row on
    each side: the next row inside the held rows, or at a pole the
    reflected row :func:`_reflected_rows` adds there."""
    nlat = window.tables.nlat
    a0, a1 = window.arrival.start, window.arrival.stop
    lo = window.lead
    hi = lo + window.arrival_rows
    if a0 == 0:
        top = xp.roll(field[:, lo:lo + 1, :], nlon // 2, axis=2)
    else:
        if lo < 1:
            raise HaloEscape(window, "Lipschitz diagnostic")
        top = field[:, lo - 1:lo, :]
    if a1 == nlat:
        bot = xp.roll(field[:, hi - 1:hi, :], nlon // 2, axis=2)
    else:
        if hi + 1 > field.shape[1]:
            raise HaloEscape(window, "Lipschitz diagnostic")
        bot = field[:, hi:hi + 1, :]
    return xp.concatenate([top, field[:, lo:hi, :], bot], axis=1)


def _d_dphi(xp, field, lat_ext, nlon, window: BandWindow | None = None):
    """``d(field)/dphi`` on the non-uniform latitude nodes, poles included.

    The two virtual rows come from the pole reflection, which is exact for a
    Cartesian component of a tangent field and for any scalar, so the
    outermost rings get the same centred three-point formula as everything
    else instead of a one-sided difference that would read as shear.

    With a ``window``, ``field`` holds the band and its halo and the
    derivative is returned on the band's rows, each read from the same
    three rows the whole-grid formula reads there.
    """
    if window is None:
        nlat = int(field.shape[1])
        first = 0
        ext = _reflected_rows(xp, field, nlon)
    else:
        nlat = window.arrival_rows
        first = window.arrival.start
        ext = _band_rows_with_neighbours(xp, field, window, nlon)
    x0 = lat_ext[first + 1:first + nlat + 1][None, :, None]
    x1 = lat_ext[first + 2:first + nlat + 2][None, :, None]
    x2 = lat_ext[first + 3:first + nlat + 3][None, :, None]
    f0 = ext[:, :-2, :]
    f1 = ext[:, 1:-1, :]
    f2 = ext[:, 2:, :]
    return (f0 * (x1 - x2) / ((x0 - x1) * (x0 - x2))
            + f1 * (2.0 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2))
            + f2 * (x1 - x0) / ((x2 - x0) * (x2 - x1)))


def _own(xp, field, rows: slice):
    """``field``'s band rows as a contiguous slab, or ``field`` itself on
    the whole grid (the slab rule of :func:`arwen_global.dynamics.band_view`).
    """
    if rows == slice(None):
        return field
    return xp.ascontiguousarray(field[:, rows])


def _d_dlam(xp, field, dlam):
    return (xp.roll(field, -1, axis=2) - xp.roll(field, 1, axis=2)) / (2.0 * dlam)


def _d_dk(xp, field):
    out = xp.empty_like(field)
    out[1:-1] = 0.5 * (field[2:] - field[:-2])
    out[0] = field[1] - field[0]
    out[-1] = field[-1] - field[-2]
    return out


@dataclass(frozen=True)
class LipschitzDiagnostics:
    """The flow deformation: the fold the gate reads and the norms beside it.

    ``fold_determinant`` is the GATED number.  The search solves
    ``r_D = r_A - h [V_ex(r_A) + V(r_D)]`` with ``h = dt/2``, so the
    departure point's derivative with respect to the arrival point is
    ``(I + h J_D)^-1 (I - h J_ex)``: the map folds (two arrival points
    share a departure point) exactly where ``det(I - h J_ex)`` reaches
    zero, and the implicit equation for ``r_D`` loses its unique solution
    where ``det(I + h J)`` does.  ``fold_determinant`` is the smaller of
    the two minima over the grid, each from the full 3x3 Jacobian of the
    flow in the coordinates the trajectory travels in (no reference
    length reaches a determinant: a change of the length that converts a
    level into metres is a diagonal similarity and leaves it unchanged).

    The norms below are the RETIRED gate, kept in every receipt as
    diagnostics.  A norm cannot tell a fold from a rotation: for a pure
    shear ``u(y)`` the spectral norm is ``|du/dy|`` and for solid rotation
    half the vorticity, while ``det(I +- h J)`` is exactly one for the
    shear and ``1 + (h zeta / 2)**2 >= 1`` for the rotation, and neither
    flow ever folds.  MEASURED 2026-10-05 (the audit's T127
    shear case, max ``|du/dy|`` 1.2e-4 per second, dt = 7200 s): the
    balanced norm read 0.861 and refused, the fold determinant is 1 in
    the audit's planar finite-difference form and 0.9956 from this
    module's own lipschitz() on the sphere, and the three-iteration search
    converges to 2.6e-4 cells.
    A T533 GDAS forecast at the shipped 300 s step was refused at 66.3 h
    on a balanced norm of 0.7511 (horizontal 0.4402, vertical 0.5134).

    ``lipschitz`` is the largest of the THREE norms that no choice of
    units can change: the spectral norm of the 2x2 HORIZONTAL block of
    the flow Jacobian, the vertical rate's own derivative in level
    indices, and the spectral norm of the BALANCED 3x3, which is the
    whole Jacobian with the length that converts a model level into
    metres chosen per point so the two coupling blocks have equal
    magnitude.  Below one in all of them the map cannot fold; above one
    it MAY, and the fold determinant is what says whether it does.

    The third exists because the first two are not the only invariant
    folds.  The coupling entries are ``dV/dk`` and ``grad(level rate)``,
    and each of them alone moves with the length chosen for a level, but
    their PRODUCT does not: a matrix ``[[0, a], [b, 0]]`` has both
    diagonal blocks zero and eigenvalues ``+-sqrt(ab)``, so a gate that
    read only the blocks would read zero on a fold that is there.
    Balancing is the closed-form scaling that equalizes the two coupling
    blocks.  It is not exactly the minimizer of the norm -- MEASURED on
    2,000 random matrices of this shape, it sits up to 10.8 percent above
    the minimum over all scalings -- but it needs no search, it is an
    upper bound on the spectral radius for the same reason any norm is,
    and every submatrix of it (the horizontal block and the vertical
    entry included) is bounded by it, which is why one number gates.
    MEASURED 2026-09-06, two hours of a real T127 L40 GDAS run at
    dt = 300 s, physics off: the balanced norm reads 0.1086 where the
    horizontal fold reads 0.0984, the vertical 0.0957 and the invariant
    spectral radius of the same matrix 0.0957, so the coupling adds 10
    percent to the gated number on a real flow and the mixed-unit norm
    at the equatorial spacing reads 0.4341, four times the gate, on the
    same states.  The verdict does not move where it has been measured:
    the whole T127 ladder from 100 to 900 s passes on the new number, at
    0.0827 to 0.7397.  The T255 rung at dt = 600 s, which read 0.6945 of
    0.75 on the old number, has not been rerun on this one and could sit
    near the limit; that rerun is the named follow-up.

    The mixed-unit 3x3 norms are reported beside it and are NOT gated,
    because their value depends on a length chosen to convert one model
    level into metres and no such length is a property of the flow.
    MEASURED 2026-09-06 on a real T255 L40 forecast day (the 16 GB host, GDAS
    analysis, dt = 100 s, physics off): the mixed 3x3 spectral norm reads
    3.114e-3 per second where the horizontal block reads 1.033e-3, a
    factor of 3.01, and the difference is the vertical rate's HORIZONTAL
    gradient multiplied by the 52.1 km equatorial spacing.  On the
    outermost T255 ring the zonal spacing is 326 m, so that one length
    overweights the polar rows by 160x, and gating on the mixed norm
    would have refused the 300 s step this integrator exists to take on
    the strength of two grid rings that are 0.26 percent of the planet.
    The horizontal block's own 1.033e-3 per second sits beside the
    1.0970e-3 the departure-point lane measured on the same class of
    state with no vertical motion in it at all, which is the cross-check
    that says the two instruments agree about the flow and disagree only
    about what a level index is worth.
    """

    dt_s: float
    reference_length_m: float
    #: dt * max over the grid of max(horizontal, vertical, balanced):
    #: the retired gate's number, reported and not gated.
    lipschitz: float
    #: dt * max Frobenius norm of the mixed-unit 3x3, reported beside it
    #: in every receipt so a future disagreement about which norm was
    #: quoted cannot be had silently
    lipschitz_frobenius: float
    #: dt * max spectral norm of the 2x2 horizontal block
    lipschitz_horizontal: float
    #: the mixed-unit 3x3 norms, per second, without the time step
    jacobian_spectral_s: float
    jacobian_frobenius_s: float
    jacobian_horizontal_s: float
    #: dt * max |d(level rate)/d(level index)|: the vertical fold
    lipschitz_vertical: float = 0.0
    jacobian_vertical_s: float = 0.0
    #: dt * the mixed-unit 3x3 spectral norm, reported and not gated
    lipschitz_mixed: float = 0.0
    #: dt * the BALANCED 3x3 spectral norm: the same matrix with the
    #: level-to-metre length chosen per point so the two coupling blocks
    #: have equal magnitude, a closed-form scaling that leaves a number no
    #: choice of length can change.  It dominates
    #: both scale-free folds (each is a submatrix of it) and it is the one
    #: the gate reads.
    lipschitz_balanced: float = 0.0
    jacobian_balanced_s: float = 0.0
    #: min over the grid of det(I - (dt/2) J_ex), J_ex the Jacobian of the
    #: wind the search reads at the ARRIVAL point (the extrapolated one
    #: when there is a previous time level): zero is a fold
    fold_determinant_arrival: float = 1.0
    #: min over the grid of det(I + (dt/2) J), J the Jacobian of the wind
    #: the search reads at the DEPARTURE point: zero is a search with no
    #: unique solution
    fold_determinant_departure: float = 1.0

    @property
    def fold_determinant(self) -> float:
        """The gated number: the smaller of the two fold determinants."""
        return min(self.fold_determinant_arrival,
                   self.fold_determinant_departure)

    @property
    def fold_side(self) -> str:
        return ("arrival"
                if self.fold_determinant_arrival
                <= self.fold_determinant_departure else "departure")

    @property
    def binding_direction(self) -> str:
        if (self.lipschitz_balanced
                > 1.000001 * max(self.lipschitz_horizontal,
                                 self.lipschitz_vertical)):
            return "coupling"
        return ("vertical"
                if self.lipschitz_vertical > self.lipschitz_horizontal
                else "horizontal")

    def as_dict(self) -> dict:
        return {
            "semilag_lipschitz": self.lipschitz,
            "semilag_lipschitz_frobenius": self.lipschitz_frobenius,
            "semilag_lipschitz_horizontal": self.lipschitz_horizontal,
            "semilag_lipschitz_vertical": self.lipschitz_vertical,
            "semilag_lipschitz_balanced": self.lipschitz_balanced,
            "semilag_lipschitz_mixed": self.lipschitz_mixed,
            "semilag_jacobian_balanced_s": self.jacobian_balanced_s,
            "semilag_jacobian_spectral_s": self.jacobian_spectral_s,
            "semilag_jacobian_frobenius_s": self.jacobian_frobenius_s,
            "semilag_jacobian_horizontal_s": self.jacobian_horizontal_s,
            "semilag_jacobian_vertical_s": self.jacobian_vertical_s,
            "semilag_lipschitz_reference_length_m": self.reference_length_m,
            "semilag_lipschitz_dt_s": self.dt_s,
            "semilag_fold_determinant": self.fold_determinant,
            "semilag_fold_determinant_arrival": self.fold_determinant_arrival,
            "semilag_fold_determinant_departure":
                self.fold_determinant_departure,
        }


_NORM_KERNEL_CACHE: dict = {}

#: The largest singular value of a 3x3, written once as the square root of
#: the top eigenvalue of its symmetric product, in the closed form the
#: fused kernel calls twice: once for the mixed-unit matrix it reports and
#: once for the balanced one it gates on.
_NORM_PREAMBLE = """
template <typename T>
__device__ __forceinline__ T sl_sym3_top(T a00, T a01, T a02,
                                         T a11, T a12, T a22)
{
    const T eps = (T)(sizeof(T) == 4 ? 1.1920929e-7 : 2.220446049250313e-16);
    const T trace = a00 + a11 + a22;
    const T scale = max(trace, eps);
    const T inv = (T)1 / scale;
    a00 *= inv; a11 *= inv; a22 *= inv;
    a01 *= inv; a02 *= inv; a12 *= inv;
    const T q = (a00 + a11 + a22) / (T)3;
    const T b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    const T p2 = (b00*b00 + b11*b11 + b22*b22
                  + (T)2 * (a01*a01 + a02*a02 + a12*a12)) / (T)6;
    const T pp = sqrt(max(p2, (T)0));
    const T safe = max(pp, eps);
    const T det = b00 * (b11*b22 - a12*a12)
                - a01 * (a01*b22 - a12*a02)
                + a02 * (a01*a12 - b11*a02);
    T r = det / ((T)2 * safe * safe * safe);
    r = min(max(r, (T)-1), (T)1);
    const T ang = acos(r) / (T)3;
    return sqrt(max(scale * (q + (T)2 * pp * cos(ang)), (T)0));
}
"""


def _norm_kernel(xp):
    """One launch for the projection and both closed-form spectra.

    Written as an ElementwiseKernel on the tree's standing convention that a
    fused device path replaces a numpy expression that stays in place as the
    specification; :func:`_norms_reference` below is that specification and
    the two are graded against each other by the unit tests.

    The reason it exists is measured: the array-operation form ran about a
    hundred grid-sized elementwise passes and cost 18.6 ms at T255 and
    147.2 ms at T533 on the 5070 Ti, against 7.7 ms and 29.8 ms for the
    fourteen-field tricubic gather it is a diagnostic on.  A gate that costs
    five times the arithmetic it gates is not a gate, it is a second model.
    """
    key = getattr(xp, "__name__", "numpy")
    kernel = _NORM_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "T dvxdx, T dvydx, T dvzdx, T dvxdy, T dvydy, T dvzdy, "
            "T dvxdk, T dvydk, T dvzdk, T dsdx, T dsdy, T dsdk, "
            "T sph, T cph, T sla, T cla, T lref",
            "T spectral, T frobenius, T horizontal, T balanced",
            """
            const T exx = -sla, exy = cla, exz = (T)0;
            const T eyx = -sph * cla, eyy = -sph * sla, eyz = cph;
            const T j00 = exx * dvxdx + exy * dvydx + exz * dvzdx;
            const T j01 = exx * dvxdy + exy * dvydy + exz * dvzdy;
            const T j02 = (exx * dvxdk + exy * dvydk + exz * dvzdk) / lref;
            const T j10 = eyx * dvxdx + eyy * dvydx + eyz * dvzdx;
            const T j11 = eyx * dvxdy + eyy * dvydy + eyz * dvzdy;
            const T j12 = (eyx * dvxdk + eyy * dvydk + eyz * dvzdk) / lref;
            const T j20 = lref * dsdx;
            const T j21 = lref * dsdy;
            const T j22 = dsdk;
            frobenius = sqrt(j00*j00 + j01*j01 + j02*j02
                           + j10*j10 + j11*j11 + j12*j12
                           + j20*j20 + j21*j21 + j22*j22);
            const T h00 = j00*j00 + j10*j10;
            const T h11 = j01*j01 + j11*j11;
            const T h01 = j00*j01 + j10*j11;
            const T half = (T)0.5 * (h00 + h11);
            const T rad = sqrt(max((T)0.25 * (h00 - h11) * (h00 - h11)
                                   + h01 * h01, (T)0));
            horizontal = sqrt(max(half + rad, (T)0));
            spectral = sl_sym3_top<T>(h00 + j20*j20,
                                      j00*j01 + j10*j11 + j20*j21,
                                      j00*j02 + j10*j12 + j20*j22,
                                      h11 + j21*j21,
                                      j01*j02 + j11*j12 + j21*j22,
                                      j02*j02 + j12*j12 + j22*j22);
            /* The BALANCED matrix: the same Jacobian with the one length
               that converts a model level into metres chosen per point so
               the two coupling blocks have equal magnitude, which is the
               scaling that equalizes them.  Its entries are
               sqrt(|dV/dk| |grad s|) whatever length the caller passed, so
               this number is the coupling's own invariant and the gate can
               read it.  Both blocks go to zero when either does, which is
               the block-triangular case whose fold IS max(horizontal,
               vertical). */
            const T e0 = j02 * lref, e1 = j12 * lref;
            const T d0 = dsdx, d1 = dsdy;
            const T emag = sqrt(e0*e0 + e1*e1);
            const T dmag = sqrt(d0*d0 + d1*d1);
            const T gmean = sqrt(emag * dmag);
            const T se = emag > (T)0 ? gmean / emag : (T)0;
            const T sd = dmag > (T)0 ? gmean / dmag : (T)0;
            const T c02 = e0 * se, c12 = e1 * se;
            const T c20 = d0 * sd, c21 = d1 * sd;
            balanced = sl_sym3_top<T>(h00 + c20*c20,
                                      j00*j01 + j10*j11 + c20*c21,
                                      j00*c02 + j10*c12 + c20*j22,
                                      h11 + c21*c21,
                                      j01*c02 + j11*c12 + c21*j22,
                                      c02*c02 + c12*c12 + j22*j22);
            """,
            "arwen_semilag_lipschitz_norms",
            preamble=_NORM_PREAMBLE,
        )
        _NORM_KERNEL_CACHE[key] = kernel
    return kernel


def _norms_reference(xp, dv, ds, sph, cph, sla, cla, lref):
    """The numpy specification of :func:`_norm_kernel`."""
    ex = (-sla, cla, xp.zeros_like(sla))
    ey = (-sph * cla, -sph * sla, cph)

    def project(basis, vec):
        return basis[0] * vec[0] + basis[1] * vec[1] + basis[2] * vec[2]

    dvdx, dvdy, dvdk = dv
    dsdx, dsdy, dsdk = ds
    j00 = project(ex, dvdx)
    j01 = project(ex, dvdy)
    j02 = project(ex, dvdk) / lref
    j10 = project(ey, dvdx)
    j11 = project(ey, dvdy)
    j12 = project(ey, dvdk) / lref
    j20 = lref * dsdx
    j21 = lref * dsdy
    j22 = dsdk
    frob = xp.sqrt(j00 * j00 + j01 * j01 + j02 * j02
                   + j10 * j10 + j11 * j11 + j12 * j12
                   + j20 * j20 + j21 * j21 + j22 * j22)
    h00 = j00 * j00 + j10 * j10
    h11 = j01 * j01 + j11 * j11
    h01 = j00 * j01 + j10 * j11
    horizontal = xp.sqrt(xp.maximum(
        _sym2_max_eigenvalue(xp, h00, h01, h11), 0.0))
    m00 = h00 + j20 * j20
    m11 = h11 + j21 * j21
    m22 = j02 * j02 + j12 * j12 + j22 * j22
    m01 = j00 * j01 + j10 * j11 + j20 * j21
    m02 = j00 * j02 + j10 * j12 + j20 * j22
    m12 = j01 * j02 + j11 * j12 + j21 * j22
    spectral = xp.sqrt(xp.maximum(
        _sym3_max_eigenvalue(xp, m00, m01, m02, m11, m12, m22), 0.0))
    # The balanced matrix: see the kernel's comment.  The coupling blocks
    # both become sqrt(|dV/dk| |grad s|), which no choice of length can
    # change, and both vanish when either does.
    e0, e1 = j02 * lref, j12 * lref
    d0, d1 = dsdx, dsdy
    emag = xp.sqrt(e0 * e0 + e1 * e1)
    dmag = xp.sqrt(d0 * d0 + d1 * d1)
    gmean = xp.sqrt(emag * dmag)
    se = xp.where(emag > 0.0, gmean / xp.where(emag > 0.0, emag, 1.0), 0.0)
    sd = xp.where(dmag > 0.0, gmean / xp.where(dmag > 0.0, dmag, 1.0), 0.0)
    c02, c12 = e0 * se, e1 * se
    c20, c21 = d0 * sd, d1 * sd
    balanced = xp.sqrt(xp.maximum(_sym3_max_eigenvalue(
        xp,
        h00 + c20 * c20,
        j00 * j01 + j10 * j11 + c20 * c21,
        j00 * c02 + j10 * c12 + c20 * j22,
        h11 + c21 * c21,
        j01 * c02 + j11 * c12 + c21 * j22,
        c02 * c02 + c12 * c12 + j22 * j22,
    ), 0.0))
    return spectral, frob, horizontal, balanced


_FOLD_KERNEL_CACHE: dict = {}


def _fold_kernel(xp):
    """``det(I + c J)`` of the 3x3 flow Jacobian, one launch per chunk.

    The fused form of :func:`_fold_reference`, graded against it by the
    unit tests on the tree's convention for a device path that replaces a
    numpy expression.  ``c`` is ``+dt/2`` for the departure side of the
    search and ``-dt/2`` for the arrival side.  No reference length enters:
    the coupling entries appear in the determinant only as the product
    ``dV/dk . grad(level rate)``, which has the units of a rate squared.
    """
    key = getattr(xp, "__name__", "numpy")
    kernel = _FOLD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "T dvxdx, T dvydx, T dvzdx, T dvxdy, T dvydy, T dvzdy, "
            "T dvxdk, T dvydk, T dvzdk, T dsdx, T dsdy, T dsdk, "
            "T sph, T cph, T sla, T cla, T c",
            "T det",
            """
            const T exx = -sla, exy = cla;
            const T eyx = -sph * cla, eyy = -sph * sla, eyz = cph;
            const T m00 = (T)1 + c * (exx * dvxdx + exy * dvydx);
            const T m01 = c * (exx * dvxdy + exy * dvydy);
            const T m02 = c * (exx * dvxdk + exy * dvydk);
            const T m10 = c * (eyx * dvxdx + eyy * dvydx + eyz * dvzdx);
            const T m11 = (T)1 + c * (eyx * dvxdy + eyy * dvydy + eyz * dvzdy);
            const T m12 = c * (eyx * dvxdk + eyy * dvydk + eyz * dvzdk);
            const T m20 = c * dsdx;
            const T m21 = c * dsdy;
            const T m22 = (T)1 + c * dsdk;
            det = m00 * (m11 * m22 - m12 * m21)
                - m01 * (m10 * m22 - m12 * m20)
                + m02 * (m10 * m21 - m11 * m20);
            """,
            "arwen_semilag_fold_determinant",
        )
        _FOLD_KERNEL_CACHE[key] = kernel
    return kernel


def _fold_reference(xp, dv, ds, sph, cph, sla, cla, c):
    """The numpy specification of :func:`_fold_kernel`."""
    ex = (-sla, cla, xp.zeros_like(sla))
    ey = (-sph * cla, -sph * sla, cph)

    def project(basis, vec):
        return basis[0] * vec[0] + basis[1] * vec[1] + basis[2] * vec[2]

    dvdx, dvdy, dvdk = dv
    dsdx, dsdy, dsdk = ds
    m00 = 1.0 + c * project(ex, dvdx)
    m01 = c * project(ex, dvdy)
    m02 = c * project(ex, dvdk)
    m10 = c * project(ey, dvdx)
    m11 = 1.0 + c * project(ey, dvdy)
    m12 = c * project(ey, dvdk)
    m20 = c * dsdx
    m21 = c * dsdy
    m22 = 1.0 + c * dsdk
    return (m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20))


#: Levels per pass of the Lipschitz diagnostic.  The full-volume form held
#: about twenty-five grid arrays at once and ran the 5070 Ti out of memory at
#: T533 L40 while the timing harness held the fourteen advected fields
#: (MEASURED 2026-09-06: OutOfMemoryError on a 205,312,512 B request with
#: 16,246,371,840 B already allocated).  Eight levels is 41 MiB an array
#: there, and the diagnostic is a reduction, so nothing but the running
#: maxima crosses a chunk boundary.
LIPSCHITZ_LEVEL_CHUNK = 8


def lipschitz(
    wind: CartesianWind,
    tables: SphericalGridTables,
    dt_s: float,
    *,
    extrapolated: CartesianWind | None = None,
    reference_length_m: float | None = None,
    level_chunk: int = LIPSCHITZ_LEVEL_CHUNK,
    fused_norms: bool = True,
    window: BandWindow | None = None,
    extrapolated_window: BandWindow | None = None,
) -> LipschitzDiagnostics:
    """The flow deformation the trajectory has to stay inside.

    Returns the fold determinants the gate reads
    (:attr:`LipschitzDiagnostics.fold_determinant`) and the norms the
    receipt reports beside them.  ``extrapolated`` is the wind the search
    reads at the ARRIVAL point (``departure_points`` takes the same
    argument); None means the arrival point reads ``wind`` as well, which
    is the first step of a run and the ``extrapolation = "none"`` arm.

    The Jacobian is built from the CARTESIAN components of the wind and then
    projected onto the local tangent basis.  Building it from the local
    east-north components instead would put a false 0.037 per second into the
    outermost T255 ring for a uniform flow past the pole, because the local
    basis rotates through 2*pi across 326 m of zonal spacing there, and the
    diagnostic would refuse every step for a deformation that does not exist.
    A Cartesian component of a tangent field is a plain smooth scalar on the
    sphere and its zonal derivative vanishes at the pole with the spacing.

    The vertical is the level index.  To put its entries in the same units as
    the horizontal ones the mixed terms of the NORMS are scaled by a single
    REFERENCE length, by default the equatorial zonal spacing
    ``a*2*pi/nlon``, so no entry inherits the polar collapse of the local
    spacing.  The fold determinants need no such length.
    """
    xp = tables.xp
    dtype = np.dtype(tables.dtype)
    nlev, nrows, nlon = wind.shape
    nlat = tables.nlat
    held = nlat if window is None else window.source_rows
    if (nrows, nlon) != (held, tables.nlon):
        raise ValueError(
            f"the wind is {(nrows, nlon)} where {(held, tables.nlon)} rows "
            f"were expected on a {tables.shape} grid")
    # The rows the diagnostic is taken on: the whole grid, or the band.
    rows = slice(0, nlat) if window is None else window.arrival
    if extrapolated is not None:
        if window is None:
            if tuple(extrapolated.shape) != tuple(wind.shape):
                raise ValueError(
                    f"the extrapolated wind is {extrapolated.shape} where the "
                    f"wind is {wind.shape}"
                )
        else:
            # A band's extrapolated wind holds its own rows and one row
            # either side (the meridional derivative's neighbours), not
            # the band's halo: extrapolated_window says which.
            if (extrapolated_window is None
                    or extrapolated_window.arrival != window.arrival):
                raise ValueError(
                    "a band's extrapolated wind needs the window of the rows "
                    "it holds, computing the band's own rows"
                )
            want = extrapolated_window.source_shape(nlev)
            if tuple(extrapolated.shape) != tuple(want):
                raise ValueError(
                    f"the band's extrapolated wind is {extrapolated.shape} "
                    f"where {want} was expected"
                )
    lref = (tables.equatorial_spacing_m() if reference_length_m is None
            else float(reference_length_m))
    if not math.isfinite(lref) or lref <= 0.0:
        raise ValueError("reference_length_m must be finite and positive")
    chunk = max(1, int(level_chunk))

    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)[None, :, None]
    lon = xp.asarray(np.arange(nlon) * tables.dlam, dtype=dtype)[None, None, :]
    lat_ext = xp.asarray(tables.lat_ext_host, dtype=dtype)
    sph, cph = xp.sin(lat)[:, rows], xp.cos(lat)[:, rows]
    sla, cla = xp.sin(lon), xp.cos(lon)
    radius = float(tables.radius_m)

    # 1/(a cos phi) is unbounded at a pole in isolation; the zonal derivative
    # of a smooth Cartesian component vanishes there at the same rate, so the
    # product is bounded.  The floor keeps the intermediate finite on a grid
    # whose outermost node sits exactly on a pole, which a Gaussian grid never
    # does but a caller-supplied grid might.
    inv_x = 1.0 / (radius * xp.maximum(cph, dtype.type(1e-12)))
    inv_y = dtype.type(1.0 / radius)

    if fused_norms and not hasattr(xp, "ElementwiseKernel"):
        fused_norms = False
    half = 0.5 * float(dt_s)

    def derivatives(flow, start, stop, win):
        lo = max(start - 1, 0)
        hi = min(stop + 1, nlev)
        take = slice(start - lo, start - lo + (stop - start))
        comps = (flow.vx, flow.vy, flow.vz)
        rate = flow.level_rate
        own = slice(None) if win is None else win.local
        dvdx = [inv_x * _d_dlam(xp, _own(xp, c[start:stop], own),
                                tables.dlam) for c in comps]
        dvdy = [inv_y * _d_dphi(xp, c[start:stop], lat_ext, nlon, win)
                for c in comps]
        dvdk = [_d_dk(xp, _own(xp, c[lo:hi], own))[take] for c in comps]
        ds = (inv_x * _d_dlam(xp, _own(xp, rate[start:stop], own),
                              tables.dlam),
              inv_y * _d_dphi(xp, rate[start:stop], lat_ext, nlon, win),
              _d_dk(xp, _own(xp, rate[lo:hi], own))[take])
        return dvdx, dvdy, dvdk, ds

    def fold(dvdx, dvdy, dvdk, ds, c):
        c = dtype.type(c)
        if fused_norms:
            det = _fold_kernel(xp)(
                dvdx[0], dvdx[1], dvdx[2],
                dvdy[0], dvdy[1], dvdy[2],
                dvdk[0], dvdk[1], dvdk[2],
                ds[0], ds[1], ds[2],
                sph, cph, sla, cla, c,
            )
        else:
            det = _fold_reference(
                xp, (dvdx, dvdy, dvdk), ds, sph, cph, sla, cla, c)
        out = float(xp.min(det))
        del det
        return out

    sm = fm = hm = vm = bm = 0.0
    arrival = departure = 1.0
    for start in range(0, nlev, chunk):
        stop = min(start + chunk, nlev)
        dvdx, dvdy, dvdk, ds = derivatives(wind, start, stop, window)
        # The vertical fold's own number, in the level indices the
        # trajectory travels in: no reference length reaches it.
        vertical_rate_derivative = ds[2]

        if fused_norms:
            spectral, frob, horizontal, balanced = _norm_kernel(xp)(
                dvdx[0], dvdx[1], dvdx[2],
                dvdy[0], dvdy[1], dvdy[2],
                dvdk[0], dvdk[1], dvdk[2],
                ds[0], ds[1], ds[2],
                sph, cph, sla, cla, dtype.type(lref),
            )
        else:
            spectral, frob, horizontal, balanced = _norms_reference(
                xp, (dvdx, dvdy, dvdk), ds, sph, cph, sla, cla, lref)
        sm = max(sm, float(xp.max(spectral)))
        fm = max(fm, float(xp.max(frob)))
        hm = max(hm, float(xp.max(horizontal)))
        vm = max(vm, float(xp.max(xp.abs(vertical_rate_derivative))))
        bm = max(bm, float(xp.max(balanced)))
        del spectral, frob, horizontal, balanced, vertical_rate_derivative

        # The departure side reads the current wind wherever the search
        # puts the departure point, so the minimum over every grid point
        # bounds it whichever point that is.
        departure = min(departure, fold(dvdx, dvdy, dvdk, ds, +half))
        if extrapolated is None:
            arrival = min(arrival, fold(dvdx, dvdy, dvdk, ds, -half))
            del dvdx, dvdy, dvdk, ds
        else:
            # One chunk's derivatives at a time, so the peak is the one
            # the norms alone had (LIPSCHITZ_LEVEL_CHUNK says why it
            # matters at T533 on a 16 GB card).
            del dvdx, dvdy, dvdk, ds
            dvdx, dvdy, dvdk, ds = derivatives(
                extrapolated, start, stop,
                None if window is None else extrapolated_window)
            arrival = min(arrival, fold(dvdx, dvdy, dvdk, ds, -half))
            del dvdx, dvdy, dvdk, ds

    dt = float(dt_s)
    gated = max(hm, vm, bm)
    return LipschitzDiagnostics(
        dt_s=dt,
        reference_length_m=lref,
        lipschitz=dt * gated,
        lipschitz_frobenius=dt * fm,
        lipschitz_horizontal=dt * hm,
        jacobian_spectral_s=sm,
        jacobian_frobenius_s=fm,
        jacobian_horizontal_s=hm,
        lipschitz_vertical=dt * vm,
        jacobian_vertical_s=vm,
        lipschitz_mixed=dt * sm,
        lipschitz_balanced=dt * bm,
        jacobian_balanced_s=bm,
        fold_determinant_arrival=arrival,
        fold_determinant_departure=departure,
    )


def convergence(diagnostics: TrajectoryDiagnostics, limit_cells: float) -> None:
    """Refuse a trajectory whose last iteration still moved.

    Named breakage: an unconverged trajectory reads as a Rossby phase error
    and mislocated cyclones, and it produces no other symptom at all.  The
    move is measured every step rather than the iteration count being
    trusted, because the count that converges on an hour-0 state is not the
    count that converges on a day-2 jet.
    """
    if diagnostics.move_max_cells > float(limit_cells):
        raise ValueError(
            "the departure-point search did not converge: the last of "
            f"{diagnostics.iterations} iterations still moved a point by "
            f"{diagnostics.move_max_m:.3f} m, which is "
            f"{diagnostics.move_max_cells:.4f} of the local meridional grid "
            f"length against a limit of {float(limit_cells):.4f}; an "
            "unconverged trajectory mislocates the flow and shows no other "
            "symptom"
        )


#: The most iterations the departure-point search takes: the ceiling of
#: ``[semilag] trajectory_iterations`` (options.py) and the count a step
#: whose search missed the convergence test is retried at.
MAXIMUM_TRAJECTORY_ITERATIONS = 8


def converged_departure_points(
    wind: CartesianWind,
    tables: SphericalGridTables,
    dt_s: float,
    *,
    iterations: int,
    limit_cells: float,
    extrapolated: CartesianWind | None = None,
    window: BandWindow | None = None,
    defer_escape: bool = False,
) -> tuple[Stencil, TrajectoryDiagnostics]:
    """The departure points, converged or refused.

    The search runs at the configured count.  Where its last iteration
    still moved a point by more than ``limit_cells``, the step is searched
    again from the start at :data:`MAXIMUM_TRAJECTORY_ITERATIONS` before
    :func:`convergence` refuses it: the fixed-point iteration contracts by
    ``(dt/2) |J|`` per pass, so a step the fold gate admits but three
    passes leave unconverged is usually converged by eight, and a refusal
    there would stop a forecast the scheme can integrate.  A step that
    converges at the configured count never reaches the retry, so its bits
    are the plain search's.
    """
    stencil, diagnostics = departure_points(
        wind, tables, dt_s, iterations=int(iterations),
        extrapolated=extrapolated, window=window, defer_escape=defer_escape,
    )
    if (diagnostics.move_max_cells > float(limit_cells)
            and int(iterations) < MAXIMUM_TRAJECTORY_ITERATIONS):
        del stencil
        stencil, diagnostics = departure_points(
            wind, tables, dt_s, iterations=MAXIMUM_TRAJECTORY_ITERATIONS,
            extrapolated=extrapolated, window=window,
            defer_escape=defer_escape,
        )
    convergence(diagnostics, limit_cells)
    return stencil, diagnostics


#: [time] minimum_fold_determinant default: the floor under
#: :attr:`LipschitzDiagnostics.fold_determinant`.  Zero is the fold itself;
#: the margin above it covers the linearization (the determinant is read
#: from grid-point Jacobians, the search from interpolated winds).  0.2 sits
#: under the smallest determinant the retired norm gate could admit on the
#: wind it read: that gate held ``dt ||J|| <= 0.75`` in the balanced norm,
#: so every eigenvalue of ``(dt/2) J`` lay within 0.375 of zero, every
#: eigenvalue of ``I +- (dt/2) J`` within 0.375 of one, and the
#: determinant was at least ``0.625**3 = 0.244``.  On the current wind the
#: change therefore only removes refusals, the ones a norm makes on
#: rotation and shear.  The arrival side reads the EXTRAPOLATED wind
#: ``2 V(t) - V(t - dt)``, which the norm gate never read; a fold there is
#: a fold of the map the search actually solves and is refused too.
DEFAULT_MINIMUM_FOLD_DETERMINANT = 0.2


def refuse_trajectory_fold(
    diagnostics: LipschitzDiagnostics, minimum: float
) -> None:
    """Refuse a time step whose trajectory map folds.

    Named breakage: where ``det(I - (dt/2) J_ex)`` reaches zero two
    arrival points share a departure point and the interpolation samples
    a folded field; where ``det(I + (dt/2) J)`` reaches zero the search's
    equation has no unique solution.  Neither blows up.  Both mislocate
    the flow silently, which is why the step refuses instead.  Rotation
    and shear do not fold (their determinants are at least one) and are
    not refused, whatever their vorticity.
    """
    floor = float(minimum)
    if not (0.0 <= floor < 1.0):
        raise ValueError(
            "the minimum fold determinant must lie in [0, 1): zero is the "
            "fold itself, and one would refuse every flow with any "
            "divergence or strain at all"
        )
    value = diagnostics.fold_determinant
    if not math.isfinite(value) or value < floor:
        raise ValueError(
            f"the trajectory map folds: min det(I - (dt/2) J) over the grid "
            f"reads {diagnostics.fold_determinant_arrival:.4f} on the "
            f"arrival side and det(I + (dt/2) J) "
            f"{diagnostics.fold_determinant_departure:.4f} on the departure "
            f"side at dt = {diagnostics.dt_s:g} s, against a floor of "
            f"{floor:.4f} (binding: {diagnostics.fold_side}); at zero two "
            "arrival points share a departure point and the interpolation "
            "samples a folded field, which mislocates the flow silently.  "
            f"The deformation norms, reported and not gated, read "
            f"{diagnostics.lipschitz:.4f} (horizontal "
            f"{diagnostics.lipschitz_horizontal:.4f}, vertical "
            f"{diagnostics.lipschitz_vertical:.4f}, balanced "
            f"{diagnostics.lipschitz_balanced:.4f}).  A shorter [time] dt_s "
            "moves the determinant toward one"
        )
