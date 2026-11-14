"""Exact first-order conservative remapping between the atmosphere's
Gaussian grid and the ocean grid.

Both grids are separable in longitude and latitude, so every overlap area is
the product of two 1-D overlaps (design 5): in sin(latitude) and in
longitude (periodic).  The atmosphere's cells are bounded in sin(latitude)
by the running sum of its Gauss weights, so each cell's area is exactly
R^2 dlambda w_j, the weight the model itself averages with.

* ``to_ocean(f)``: the overlap-weighted mean of ``f`` over the OWNED
  atmosphere columns (the atmosphere's ocean columns at least half covered
  by wet ocean cells), with any wet ocean cell that overlaps no owned
  column taking its nearest owned column; then the global difference
  between sum_a A_a f_a over the owned columns and sum_o A_o f_o over the
  wet cells is spread evenly per unit area and returned, so the ocean gets
  exactly what the atmosphere exchanged.
* ``to_atmosphere(g)``: the overlap-weighted mean of ``g`` over the wet
  ocean cells of each atmosphere column (defined where the column overlaps
  any wet cell), with the coverage (wet overlap / column area).
"""
from __future__ import annotations

import numpy as np


def _overlap_1d(a_edges, b_edges, period=None):
    """(na, nb) overlap lengths of intervals a and b (both ascending)."""
    a0, a1 = a_edges[:-1, None], a_edges[1:, None]
    out = np.zeros((a_edges.size - 1, b_edges.size - 1))
    shifts = [0.0] if period is None else [-period, 0.0, period]
    for s in shifts:
        b0, b1 = b_edges[None, :-1] + s, b_edges[None, 1:] + s
        out += np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0, None)
    return out


def gaussian_edges_mu(sin_lat_or_lat_deg, weights):
    """sin(latitude) edges of Gaussian cells in the grid's own row order
    (returned as (lower, upper) per row)."""
    w = np.asarray(weights, dtype=np.float64)
    lat = np.asarray(sin_lat_or_lat_deg, dtype=np.float64)
    order = np.argsort(lat)
    edges = np.concatenate([[-1.0], -1.0 + np.cumsum(w[order])])
    edges[-1] = 1.0
    lower = np.empty_like(w)
    upper = np.empty_like(w)
    lower[order] = edges[:-1]
    upper[order] = edges[1:]
    return lower, upper


class Remap:
    def __init__(self, ocean_grid, *, atm_lat_deg, atm_weights, atm_nlon: int, atm_ocean,
                 radius_m: float):
        from .grid import RADIUS

        og = ocean_grid
        self.og = og
        lat_a = np.asarray(atm_lat_deg, dtype=np.float64)
        lo, hi = gaussian_edges_mu(lat_a, atm_weights)
        mu_o = np.sin(np.deg2rad(og.lat_edges_deg))
        # (ny_o, nlat_a) overlap in mu, rows in the atmosphere's own order.
        a0, a1 = mu_o[:-1, None], mu_o[1:, None]
        self.wy = np.clip(np.minimum(a1, hi[None]) - np.maximum(a0, lo[None]), 0.0, None)
        dl = 2.0 * np.pi / int(atm_nlon)
        lam_a = (np.arange(int(atm_nlon) + 1) - 0.5) * dl
        lam_o = np.deg2rad(og.lon_edges_deg)
        self.wx = _overlap_1d(lam_o, lam_a, period=2.0 * np.pi)   # (nx_o, nlon_a)
        self.R2 = float(radius_m) ** 2
        self.area_a = (self.R2 * dl * (hi - lo))[:, None] * np.ones((1, int(atm_nlon)))
        self.area_o = np.broadcast_to(og.area, (og.ny, og.nx)).astype(np.float64)
        self.wet_o = og.wet.astype(np.float64)
        atm_ocean = np.asarray(atm_ocean, dtype=bool)
        # Wet coverage of each atmosphere column.
        wet_overlap = self.wy.T @ self.wet_o @ self.wx * self.R2
        self.coverage = np.clip(wet_overlap / self.area_a, 0.0, 1.0)
        self.owned = atm_ocean & (self.coverage >= 0.5)
        own = self.owned.astype(np.float64)
        den = self.wy @ own @ self.wx.T
        need = og.wet & (den <= 0.0)
        self.fill_ocean = np.nonzero(need.ravel())[0]
        self.fill_atm = np.zeros(0, dtype=np.int64)
        if self.fill_ocean.size:
            from scipy.spatial import cKDTree

            def xyz(lat, lon):
                la, lo_ = np.deg2rad(lat), np.deg2rad(lon)
                return np.stack([np.cos(la) * np.cos(lo_), np.cos(la) * np.sin(lo_), np.sin(la)], -1)

            lon_a = np.rad2deg(np.arange(int(atm_nlon)) * dl)
            LA, LO = np.meshgrid(lat_a, lon_a, indexing="ij")
            idx = np.nonzero(self.owned.ravel())[0]
            tree = cKDTree(xyz(LA.ravel()[idx], LO.ravel()[idx]))
            LT, LN = np.meshgrid(og.lat_t, og.lon_t, indexing="ij")
            _, near = tree.query(xyz(LT.ravel()[self.fill_ocean], LN.ravel()[self.fill_ocean]))
            self.fill_atm = idx[near]
        self.den_o = np.where(den > 0.0, den, 1.0)
        self.has_o = den > 0.0
        den_a = self.wy.T @ self.wet_o @ self.wx
        self.den_a = np.where(den_a > 0.0, den_a, 1.0)
        self.has_a = den_a > 0.0
        self.wet_area_total = float((self.area_o * self.wet_o).sum())
        self._dev = {}

    def stats(self) -> dict:
        return {"owned_columns": int(self.owned.sum()),
                "partly_covered_unowned": int(((self.coverage > 0) & ~self.owned).sum()),
                "ocean_cells_filled_from_nearest": int(self.fill_ocean.size),
                "owned_area_m2": float((self.area_a * self.owned).sum()),
                "wet_area_m2": self.wet_area_total}

    def on(self, xp):
        key = xp.__name__
        if key not in self._dev:
            names = ("wy", "wx", "area_a", "area_o", "wet_o", "owned", "den_o", "has_o", "den_a",
                     "has_a", "fill_ocean", "fill_atm", "coverage")
            self._dev[key] = {n: xp.asarray(getattr(self, n)) for n in names}
        return self._dev[key]

    def to_ocean(self, f, xp):
        """``(g, residual)``: g on the ocean grid (0 off the wet cells) and
        the spread global residual (units of f times m2)."""
        d = self.on(xp)
        f = xp.asarray(f, dtype=xp.float64)
        own = d["owned"]
        fm = xp.where(own, f, 0.0)
        num = d["wy"] @ fm @ d["wx"].T
        g = xp.where(d["has_o"], num / d["den_o"], 0.0)
        if self.fill_ocean.size:
            flat = g.reshape(-1)
            flat[d["fill_ocean"]] = fm.reshape(-1)[d["fill_atm"]]
            g = flat.reshape(g.shape)
        g = g * d["wet_o"]
        total_a = xp.sum(fm * d["area_a"])
        total_o = xp.sum(g * d["area_o"])
        residual = total_a - total_o
        g = g + d["wet_o"] * (residual / self.wet_area_total)
        return g, residual

    def to_atmosphere(self, g, xp):
        d = self.on(xp)
        g = xp.asarray(g, dtype=xp.float64) * d["wet_o"]
        num = d["wy"].T @ g @ d["wx"]
        return xp.where(d["has_a"], num / d["den_a"], 0.0)


__all__ = ["Remap", "gaussian_edges_mu"]
