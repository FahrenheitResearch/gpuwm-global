"""The NumPy cold-start regrid and remap, kept as the test oracle.

These are the exact expressions the cold start ran on every
analysis-initialised start through WOOF Global 0.1.2 (``analysis_initial``
at 7548983, ``_global_regridder`` and ``_to_model_levels``), copied
verbatim.  The default path now runs them in the Rust cold-start door
(``rust/rw-global-coldstart``); these copies exist only so the suite can
hold the door byte-identical to them.  Nothing under ``src/`` imports this
module, and nothing should.
"""
from __future__ import annotations

import numpy as np


def numpy_global_regridder(latitude: np.ndarray, longitude: np.ndarray, grid):
    """Periodic bilinear interpolation weights from a regular global
    lat-lon grid onto the transform's Gaussian grid."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    flip = lat[0] > lat[-1]
    if flip:
        lat = lat[::-1]
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("analysis grid must be two-dimensional")
    if np.max(np.abs(dlat - dlat[0])) > 1.0e-6 or np.max(np.abs(dlon - dlon[0])) > 1.0e-6:
        raise ValueError("analysis grid must be regular in latitude and longitude")
    span = float(dlon[0]) * lon.size
    if abs(span - 360.0) > 1.0e-3:
        raise ValueError(
            f"analysis longitude ring covers {span:.4f} degrees, not the "
            "globe; a regional subset cannot initialize the global model"
        )
    target_lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    target_lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    if target_lat[0] < lat[0] - 1.0e-9 or target_lat[-1] > lat[-1] + 1.0e-9:
        raise ValueError(
            "analysis latitudes do not cover the Gaussian grid; the source "
            "must reach both poles"
        )
    fy = np.clip((target_lat - lat[0]) / dlat[0], 0.0, lat.size - 1.0)
    y0 = np.minimum(fy.astype(np.int64), lat.size - 2)
    wy = fy - y0
    fx = np.mod(target_lon - lon[0], 360.0) / dlon[0]
    x0 = np.mod(fx.astype(np.int64), lon.size)
    wx = fx - np.floor(fx)
    x1 = np.mod(x0 + 1, lon.size)

    def regrid(values: np.ndarray) -> np.ndarray:
        field = np.asarray(values, dtype=np.float64)
        if flip:
            field = field[..., ::-1, :]
        yl = y0[:, None]
        yu = (y0 + 1)[:, None]
        a = field[..., yl, x0[None, :]]
        b = field[..., yl, x1[None, :]]
        c = field[..., yu, x0[None, :]]
        d = field[..., yu, x1[None, :]]
        wyc = wy[:, None]
        wxc = wx[None, :]
        return (
            (1.0 - wyc) * ((1.0 - wxc) * a + wxc * b)
            + wyc * ((1.0 - wxc) * c + wxc * d)
        )

    return regrid


def numpy_to_model_levels(values, ln_source, ln_target, *, extrapolate_below=False):
    """Linear-in-ln(p) interpolation from source pressure levels to model
    full levels.  Above the top source level the value is held; below the
    bottom the value is held unless ``extrapolate_below`` continues the
    bottom layer's ln(p) gradient (temperature's lapse continuation)."""
    upper = np.clip(np.searchsorted(ln_source, ln_target), 1, ln_source.size - 1)
    lower = upper - 1
    weight = (ln_target - ln_source[lower]) / (ln_source[upper] - ln_source[lower])
    weight = np.maximum(weight, 0.0)
    if not extrapolate_below:
        weight = np.minimum(weight, 1.0)
    low = np.take_along_axis(values, lower, axis=0)
    high = np.take_along_axis(values, upper, axis=0)
    return low * (1.0 - weight) + high * weight
