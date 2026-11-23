"""Radiation on a reduced grid: RRTMGP on every ``s``-th row and column,
its planes brought back to the model grid.

An operational practice (the IFS has computed its full radiation on a
coarser grid than the dynamics for decades and interpolated the results
back), here in its simplest form: the radiation call sees the columns at
rows ``0, sy, 2 sy, ...`` and longitudes ``0, sx, 2 sx, ...`` of the
physics batch (``radiation_stride_lat`` / ``radiation_stride_lon``,
:class:`~arwen_global.physics.native_options.NativePhysicsOptions`), so a
2 x 2 stride solves a quarter of the columns.  Default 1 x 1 is the full
grid; the option is then out of the adapter identity and no arithmetic
changes.

How each plane comes back (bilinear in row and longitude index, periodic
in longitude, the last reduced row held to the band's edge):

* Longwave heating, surface downward longwave, OLR and the total cloud
  fraction: interpolated as they are.
* Shortwave heating, surface downward shortwave and the two top-of-
  atmosphere shortwave planes: interpolated as a ratio to the reduced
  column's cosine of the solar zenith angle over the sunlit reduced
  columns only, then multiplied by the model column's own cosine (the
  driver's ``_cosine_zenith`` at the same interval-midpoint sun), so a
  dark model column is dark and the incident top flux is exact.
* Net surface shortwave: the interpolated downward shortwave times one
  minus the model column's own albedo (the solver's surface condition:
  the same albedo for the direct and diffuse beams, so ``gsw = swdown (1
  - albedo)`` exactly on a full-grid call).
* Surface upward longwave: the model column's own grey-body emission and
  reflection, ``emiss sigma tsk^4 + (1 - emiss) glw``.
* The size-bounding per-column sums: nearest reduced column.

Clouds, humidity and temperature are sampled, not averaged, so a reduced
column radiates the model column it sits on.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

F = np.float32


def validate_strides(lat: int, lon: int) -> None:
    for name, value in (("radiation_stride_lat", lat),
                        ("radiation_stride_lon", lon)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 8:
            raise ValueError(f"{name} must be a whole number 1..8, got {value!r}")


class FullRadiationGrid:
    """The default: the radiation sees the model grid, nothing is copied."""

    reduced = False

    @staticmethod
    def sub(value):
        return value


FULL_GRID = FullRadiationGrid()


class ReducedRadiationGrid:
    """Index maps and weights between one batch's (ny, nx) grid and its
    (ceil(ny/sy), nx/sx) radiation grid."""

    reduced = True

    def __init__(self, xp, ny: int, nx: int, sy: int, sx: int):
        if nx % sx:
            raise ValueError(
                f"radiation_stride_lon {sx} must divide the {nx} longitudes")
        self.xp, self.ny, self.nx, self.sy, self.sx = xp, ny, nx, sy, sx
        nyr = (ny + sy - 1) // sy
        nxr = nx // sx
        self.nyr, self.nxr = nyr, nxr
        j = np.arange(ny)
        j0 = np.minimum(j // sy, nyr - 1)
        j1 = np.minimum(j0 + 1, nyr - 1)
        g = np.where(j1 > j0, (j - j0 * sy) / float(sy), 0.0)
        i = np.arange(nx)
        i0 = i // sx
        i1 = (i0 + 1) % nxr
        f = (i - i0 * sx) / float(sx)
        self.j0 = xp.asarray(j0[:, None])
        self.j1 = xp.asarray(j1[:, None])
        self.i0 = xp.asarray(i0[None, :])
        self.i1 = xp.asarray(i1[None, :])
        g = g[:, None]
        f = f[None, :]
        self.w00 = xp.asarray(((1 - g) * (1 - f)).astype(F))
        self.w01 = xp.asarray(((1 - g) * f).astype(F))
        self.w10 = xp.asarray((g * (1 - f)).astype(F))
        self.w11 = xp.asarray((g * f).astype(F))

    def is_plane(self, value) -> bool:
        shape = getattr(value, "shape", None)
        return (shape is not None and len(shape) >= 2
                and tuple(shape[-2:]) == (self.ny, self.nx))

    def sub(self, value):
        """The reduced grid's columns of a (..., ny, nx) array; anything
        else unchanged."""
        if not self.is_plane(value):
            return value
        return self.xp.ascontiguousarray(value[..., ::self.sy, ::self.sx])

    def sub_state(self, state):
        """A copy of the radiation state namespace with every model-grid
        array reduced (nested namespaces and scalars as they are)."""
        return SimpleNamespace(**{
            name: self.sub(value) for name, value in vars(state).items()})

    def linear(self, reduced):
        """Bilinear back to the model grid: (..., nyr, nxr) -> (..., ny, nx)."""
        a = reduced
        out = self.w00 * a[..., self.j0, self.i0]
        out += self.w01 * a[..., self.j0, self.i1]
        out += self.w10 * a[..., self.j1, self.i0]
        out += self.w11 * a[..., self.j1, self.i1]
        return self.xp.ascontiguousarray(out.astype(F, copy=False))

    def nearest(self, reduced):
        """(..., nyr * nxr) or (..., nyr, nxr) -> (..., ny, nx), nearest
        reduced column at or before."""
        a = self.xp.asarray(reduced)
        if a.ndim == 1:
            a = a.reshape(self.nyr, self.nxr)
        return self.xp.ascontiguousarray(a[..., self.j0, self.i0])


def full_grid_result(grid, result, *, coszen_full, albedo, emissivity,
                     skin_temperature, stefan_boltzmann):
    """The reduced call's RadiationResult on the model grid (see the module
    notes), as a namespace with the same plane names."""
    xp = grid.xp
    mu_r = xp.asarray(result.coszen, dtype=F)
    lit_r = mu_r > F(0.0)
    inverse = xp.where(lit_r, F(1.0) / xp.where(lit_r, mu_r, F(1.0)), F(0.0))
    weight = grid.linear(lit_r.astype(F))
    mu = xp.asarray(coszen_full, dtype=F)
    scale = xp.where((mu > F(0.0)) & (weight > F(0.0)),
                     mu / xp.where(weight > F(0.0), weight, F(1.0)), F(0.0))

    def shortwave(plane):
        return xp.ascontiguousarray(
            (grid.linear(plane * inverse) * scale).astype(F, copy=False))

    swdown = shortwave(result.swdown)
    glw = grid.linear(result.glw)
    emissivity = xp.asarray(emissivity, dtype=F)
    tsk = xp.asarray(skin_temperature, dtype=F)
    cldfra = getattr(result, "cldfra_total", None)
    return SimpleNamespace(
        rthratenlw=grid.linear(result.rthratenlw),
        rthratensw=shortwave(result.rthratensw),
        swdown=swdown,
        glw=glw,
        olr=grid.linear(result.olr),
        gsw=xp.ascontiguousarray(
            swdown * (F(1.0) - xp.asarray(albedo, dtype=F))),
        coszen=xp.ascontiguousarray(mu),
        swupt=shortwave(result.swupt),
        swdnt=shortwave(result.swdnt),
        lwupb=xp.ascontiguousarray(
            emissivity * F(stefan_boltzmann) * tsk ** 4
            + (F(1.0) - emissivity) * glw),
        cldfra_total=None if cldfra is None else grid.linear(cldfra),
    )


__all__ = ["FULL_GRID", "ReducedRadiationGrid", "full_grid_result",
           "validate_strides"]
