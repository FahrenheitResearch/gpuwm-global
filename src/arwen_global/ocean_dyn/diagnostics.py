"""The dynamic ocean's rows (design 9): Nino indices, the Equatorial
Undercurrent at 140W, D20, heat content, transports, books."""
from __future__ import annotations

import numpy as np

from .grid import CP0, RHO0
from .model import BOOKS


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


#: Sections for the transports (Sv): name, (kind, longitude or latitude,
#: range); "x" is a meridional section of U faces at the given longitude
#: (eastward positive), "y" a zonal section of V faces (northward positive).
SECTIONS = (
    ("drake", "x", 292.0, (-67.0, -55.0)),
    ("itf_116e", "y", -8.5, (114.0, 126.0)),   # a zonal line south of the passages: northward +
    ("florida_27n", "y", 26.5, (279.0, 282.0)),
    ("gibraltar", "x", 354.0, (35.0, 37.0)),
)


def area_mean(grid, plane, mask):
    a = np.broadcast_to(grid.area, plane.shape)
    w = (a * mask).sum()
    return float((plane * a * mask).sum() / w) if w > 0 else float("nan")


def d20(theta, grid):
    """Depth (m) of the 20 deg C isotherm (NaN where the column never crosses
    it from above)."""
    z = grid.z_t_nom
    th = np.where(grid.tmask, theta, -99.0)
    warm = th >= 20.0
    out = np.full(theta.shape[1:], np.nan)
    for k in range(grid.nz - 1):
        cross = warm[k] & ~warm[k + 1] & np.isnan(out) & grid.tmask[k + 1]
        t = (th[k] - 20.0) / np.where(cross, th[k] - th[k + 1], 1.0)
        out = np.where(cross, z[k] + t * (z[k + 1] - z[k]), out)
    return out


def rows(grid, state, *, sst_ref_c=None) -> dict:
    th = _host(state["theta"])
    u = _host(state["u"])
    v = _host(state["v"])
    eta = _host(state["eta"])
    b = _host(state["books"])
    lat, lon = grid.lat_t, grid.lon_t
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    wet = grid.wet
    sst = th[0]
    out = {}
    for name, (lo0, lo1, la0, la1) in {"nino34": (190, 240, -5, 5), "nino3": (210, 270, -5, 5),
                                      "nino4": (160, 210, -5, 5)}.items():
        m = wet & (LON >= lo0) & (LON <= lo1) & (LAT >= la0) & (LAT <= la1)
        out[f"ocean_{name}_sst_c"] = area_mean(grid, sst, m)
    out["ocean_mean_sst_c"] = area_mean(grid, sst, wet)
    if sst_ref_c is not None:
        ok = wet & np.isfinite(sst_ref_c)
        diff = np.where(ok, sst - sst_ref_c, 0.0)
        out["ocean_sst_bias_c"] = area_mean(grid, diff, ok)
        out["ocean_sst_rms_c"] = float(np.sqrt(area_mean(grid, diff ** 2, ok)))
    # EUC at 140W (220E): the largest eastward speed within 2S-2N, 0-400 m
    i = int(np.argmin(np.abs(grid.lon_edges_deg[1:] - 220.0)))
    js = np.nonzero(np.abs(lat) <= 2.0)[0]
    ks = np.nonzero(grid.z_t_nom <= 400.0)[0]
    sec = u[np.ix_(ks, js, [i])][..., 0]
    if sec.size:
        k_, j_ = np.unravel_index(np.argmax(sec), sec.shape)
        out["ocean_euc_140w_max_m_s"] = float(sec.max())
        out["ocean_euc_140w_depth_m"] = float(grid.z_t_nom[ks[k_]])
        out["ocean_surface_u_140w_eq_m_s"] = float(u[0, js, i].mean())
    else:  # a grid with no row within 2 deg of the equator (tests)
        out["ocean_euc_140w_max_m_s"] = out["ocean_euc_140w_depth_m"] = float("nan")
        out["ocean_surface_u_140w_eq_m_s"] = float("nan")
    dd = d20(th, grid)
    for name, (lo0, lo1) in {"west": (130, 180), "east": (210, 270)}.items():
        m = wet & np.isfinite(dd) & (LON >= lo0) & (LON <= lo1) & (np.abs(LAT) <= 5)
        out[f"ocean_d20_eq_pacific_{name}_m"] = area_mean(grid, np.nan_to_num(dd), m)
    s = 1.0 + eta / np.where(wet, grid.H, 1.0)
    h = grid.dzt * s[None]
    top = np.cumsum(h, axis=0) <= 300.0 + 1e-6
    out["ocean_ohc300_mean_j_m2"] = area_mean(grid, (RHO0 * CP0 * th * h * top).sum(axis=0), wet)
    out["ocean_mean_ssh_m"] = area_mean(grid, eta, wet)
    out["ocean_max_speed_m_s"] = float(max(np.abs(u).max(), np.abs(v).max()))
    out["ocean_mean_hbl_m"] = area_mean(grid, _host(state["hbl"]).astype(np.float64), wet)
    # transports
    su = 0.5 * (s + np.roll(s, -1, 1))
    sv = s.copy()
    sv[:-1] = 0.5 * (s[:-1] + s[1:])
    for name, kind, where, rng in SECTIONS:
        if kind == "x":
            i = int(np.argmin(np.abs(grid.lon_edges_deg[1:] - where)))
            js = np.nonzero((lat >= rng[0]) & (lat <= rng[1]))[0]
            tr = (u[:, js, i] * grid.dzu[:, js, i] * su[js, i] * grid.dyu[js, 0]).sum()
        else:
            j = int(np.argmin(np.abs(grid.lat_edges_deg[1:] - where)))
            is_ = np.nonzero((lon >= rng[0]) & (lon <= rng[1]))[0]
            tr = (v[:, j, is_] * grid.dzv[:, j, is_] * sv[j, is_] * grid.dxv[j, 0]).sum()
        out[f"ocean_transport_{name}_sv"] = float(tr) / 1e6
    # books
    ii = BOOKS.index
    s3 = 1.0 + eta / np.where(wet, grid.H, 1.0)
    vol = grid.dzt * s3[None] * grid.area
    heat = float((vol * th).sum()) * RHO0 * CP0
    salt = float((vol * _host(state["salt"])).sum())
    out["ocean_heat_book_residual_rel"] = (heat - b[ii("heat_content0_j")] - b[ii("heat_sources_j")]) / heat
    out["ocean_salt_book_residual_rel"] = (salt - b[ii("salt_content0")] - b[ii("salt_sources")]) / salt
    out["ocean_volume_book_residual_rel"] = (float(vol.sum()) - b[ii("volume0_m3")]) / float(vol.sum())
    out["ocean_frazil_j"] = float(b[ii("frazil_j")])
    out["ocean_steps"] = int(b[ii("steps")])
    # A row the grid cannot define (no such section, no equatorial row) is
    # None: the run's rows are strict JSON.
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in out.items()}


__all__ = ["SECTIONS", "d20", "rows"]
