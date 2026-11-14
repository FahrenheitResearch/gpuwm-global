"""The dynamic ocean's parts and idealised gates (design 11, L0 and L1).

    python tools/ocean_dyn_gates.py --xp numpy|cupy --grid ocean-grid-v1.npz \
        --out results.json [gate ...]

Gates: eos tridiag remap kpp-kato kpp-convective rest cons kelvin gyre restart
parity slump.  Each prints and records what it measured; a gate fails only
on its stated threshold.  Run on a Linux host (CPU) or through the pool
runner (GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from arwen_global.ocean_dyn import eos
from arwen_global.ocean_dyn.column import thomas
from arwen_global.ocean_dyn.grid import GRAVITY, RHO0, CP0, OceanGrid, idealized, level_edges, row_edges
from arwen_global.ocean_dyn.model import OceanModel, OceanParams, BOOKS, vertical_modes


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def gate_eos(xp, args):
    refs = ((35.0, 5.0, 1027.67547), (35.0, 25.0, 1023.34306), (0.0, 5.0, 999.96675))
    rows = [{"S": s, "T": t, "rho": float(eos.density(t, s, 0.0)), "eos80": r,
             "diff": float(eos.density(t, s, 0.0) - r)} for s, t, r in refs]
    worst = max(abs(r["diff"]) for r in rows)
    # analytic derivatives against centred differences
    t, s, p = 10.0, 35.0, 2.0e7
    _, dt, ds = eos.density_derivatives(t, s, p)
    e = 1e-4
    fdt = (eos.density(t + e, s, p) - eos.density(t - e, s, p)) / (2 * e)
    fds = (eos.density(t, s + e, p) - eos.density(t, s - e, p)) / (2 * e)
    derr = max(abs(dt - fdt) / abs(fdt), abs(ds - fds) / abs(fds))
    return {"pass": worst < 0.02 and derr < 1e-7, "worst_vs_eos80_kg_m3": worst,
            "derivative_rel_err": derr, "rows": rows,
            "note": "Wright (1997) full range against the UNESCO EOS-80 one-atmosphere "
                    "check values (theta = t at p = 0); threshold 0.02 kg m-3"}


def gate_tridiag(xp, args):
    rng = np.random.default_rng(1)
    n, m = 50, 200
    a = rng.uniform(-1, 0, (n, m))
    c = rng.uniform(-1, 0, (n, m))
    b = 2.5 + rng.uniform(0, 1, (n, m))
    d = rng.normal(size=(n, m))
    x = host(thomas(xp.asarray(a), xp.asarray(b), xp.asarray(c), xp.asarray(d), xp))
    worst = 0.0
    for j in range(m):
        A = np.diag(b[:, j]) + np.diag(a[1:, j], -1) + np.diag(c[:-1, j], 1)
        ref = np.linalg.solve(A, d[:, j])
        worst = max(worst, float(np.abs(x[:, j] - ref).max() / np.abs(ref).max()))
    return {"pass": worst < 1e-12, "max_rel_err": worst}


def gate_remap(xp, args):
    from arwen_global.ocean_dyn.remap import Remap
    from arwen_global.spectral.grid import GaussianGrid

    og = OceanGrid.load(args.grid)
    out = {}
    ok = True
    rng = np.random.default_rng(2)
    for T in (127, 255, 383):
        gg = GaussianGrid.create(T)
        lat = np.rad2deg(gg.lat_rad)
        # an atmosphere ocean mask: everywhere the ocean grid has any wet
        # coverage, plus random extra columns (masks that disagree).
        tmp = Remap(og, atm_lat_deg=lat, atm_weights=gg.quadrature_weights, atm_nlon=gg.nlon,
                    atm_ocean=np.ones((gg.nlat, gg.nlon), bool), radius_m=gg.radius_m)
        ocean = (tmp.coverage > 0.3) | (rng.uniform(size=tmp.coverage.shape) < 0.01)
        rm = Remap(og, atm_lat_deg=lat, atm_weights=gg.quadrature_weights, atm_nlon=gg.nlon,
                   atm_ocean=ocean, radius_m=gg.radius_m)
        f = rng.normal(size=(gg.nlat, gg.nlon)) * 100.0 + 50.0
        g, res = rm.to_ocean(xp.asarray(f), xp)
        g = host(g)
        res = float(res)
        total_a = float((f * rm.area_a * rm.owned).sum())
        total_o = float((g * rm.area_o * rm.wet_o).sum())
        rel = abs(total_a - total_o) / abs(total_a)
        # constant field maps to the constant (ocean -> atmosphere)
        back = host(rm.to_atmosphere(xp.asarray(np.full((og.ny, og.nx), 7.0)), xp))
        const_err = float(np.abs(np.where(rm.has_a, back - 7.0, 0.0)).max())
        stats = rm.stats()
        out[f"T{T}"] = {"conservation_rel": rel, "residual_spread_rel": abs(res) / abs(total_a),
                        "constant_roundtrip_err": const_err, **stats}
        ok &= rel < 1e-12 and const_err < 1e-12
    return {"pass": bool(ok), **out}


def _column_grid(nz_levels=None, depth=300.0, lat_center=0.0):
    z = level_edges() if nz_levels is None else nz_levels
    z = z[z <= depth + 1e-9]
    lat = [lat_center - 0.5, lat_center + 0.5]
    lon = np.arange(5) * 1.0
    return idealized(lat_edges_deg=lat, lon_edges_deg=lon, z_edges_m=z,
                     depth_m=np.full((1, 4), z[-1]), periodic=True, substeps=4)


def _mld_from_n2(m, st, n2_threshold):
    xp = m.xp
    th, sa = st["theta"], st["salt"]
    rhop, rt, rs = m._density(th, sa, st["rho_ref"])
    h = m.g["dzt"]
    n2 = host(m._n2(th, sa, rt, rs, h))[:, 0, 0]
    zw = np.cumsum(m.grid.dzt[:, 0, 0])
    k = int(np.argmax(n2 > n2_threshold))
    if k == 0:
        return float(zw[0])
    # interpolate between interfaces k-1 and k
    a, b = n2[k - 1], n2[k]
    t = (n2_threshold - a) / (b - a) if b != a else 0.0
    return float(zw[k - 1] + t * (zw[k] - zw[k - 1]))


def gate_kpp_kato(xp, args):
    """Kato-Phillips: u* = 0.01 m/s into N^2 = 1e-4 s-2, no rotation; h(30 h)
    against 1.05 u* sqrt(t/N)."""
    g = _column_grid(depth=300.0)
    p = OceanParams(gm=False, sponge=False, kappa_h=0.0, viscosity=False, sss_piston_m_per_day=None)
    m = OceanModel(g, p, xp)
    alpha = 2.0e-4
    N2 = 1.0e-4
    z = np.broadcast_to(g.z_t_nom[:, None, None], (g.nz, 1, 4))
    _, rt, _ = eos.density_derivatives(15.0, 35.0, RHO0 * GRAVITY * z)
    dTdz = N2 / (GRAVITY * (-rt / RHO0))
    T = 15.0 - np.cumsum(np.concatenate([np.zeros((1, 1, 4)), dTdz[1:] * np.diff(z, axis=0)]), axis=0)
    st = m.new_state(T, np.full_like(T, 35.0))
    ustar = 0.01
    fc = {"taux": xp.full((1, 4), RHO0 * ustar ** 2), "tauy": xp.zeros((1, 4))}
    rows = []
    for n in range(30):
        st, _ = m.step(st, fc)
        if (n + 1) % 6 == 0:
            t = (n + 1) * 3600.0
            rows.append({"hours": n + 1, "hbl": float(host(st["hbl"])[0, 0]),
                         "mld_n2": _mld_from_n2(m, st, 0.5 * N2),
                         "kato_phillips": 1.05 * ustar * math.sqrt(t / math.sqrt(N2))})
    last = rows[-1]
    err = (last["mld_n2"] - last["kato_phillips"]) / last["kato_phillips"]
    return {"pass": abs(err) <= 0.15, "rel_err_30h": err, "rows": rows,
            "note": "mixed-layer depth from N^2 > N0^2/2 on the model's own 10 m levels"}


def gate_kpp_convective(xp, args):
    """Free convection: B0 = 1e-7 m2 s-3 out of the surface into N^2 = 1e-5,
    no wind: h^2 = 2 (1 + 2 A) B0 t / N^2 with A = 0.2."""
    g = _column_grid(depth=400.0)
    p = OceanParams(gm=False, sponge=False, kappa_h=0.0, viscosity=False, sss_piston_m_per_day=None)
    m = OceanModel(g, p, xp)
    N2 = 1.0e-5
    z = np.broadcast_to(g.z_t_nom[:, None, None], (g.nz, 1, 4))
    _, rt, _ = eos.density_derivatives(15.0, 35.0, RHO0 * GRAVITY * z)
    alpha = -rt / RHO0
    dTdz = N2 / (GRAVITY * alpha)
    T = 15.0 - np.cumsum(np.concatenate([np.zeros((1, 1, 4)), dTdz[1:] * np.diff(z, axis=0)]), axis=0)
    st = m.new_state(T, np.full_like(T, 35.0))
    B0 = 1.0e-7
    q = -B0 * RHO0 * CP0 / (GRAVITY * float(alpha[0, 0, 0]))
    fc = {"q_ns": xp.full((1, 4), q), "taux": xp.full((1, 4), 1.0e-6)}
    rows = []
    for n in range(48):
        st, _ = m.step(st, fc)
        if (n + 1) % 12 == 0:
            t = (n + 1) * 3600.0
            rows.append({"hours": n + 1, "hbl": float(host(st["hbl"])[0, 0]),
                         "mld_n2": _mld_from_n2(m, st, 0.5 * N2),
                         "encroachment_A0.2": math.sqrt(2 * 1.4 * B0 * t / N2)})
    last = rows[-1]
    err = (last["mld_n2"] - last["encroachment_A0.2"]) / last["encroachment_A0.2"]
    return {"pass": abs(err) <= 0.15, "rel_err_48h": err, "q_w_m2": q, "rows": rows}


def _stratified(g, xp_unused=None):
    z = g.z_t_nom
    T = 2.0 + 23.0 * np.exp(-z / 400.0)
    S = 34.7 + 0.3 * np.exp(-z / 300.0)
    T3 = np.broadcast_to(T[:, None, None], (g.nz, g.ny, g.nx)).copy()
    S3 = np.broadcast_to(S[:, None, None], (g.nz, g.ny, g.nx)).copy()
    return T3, S3


def gate_rest(xp, args):
    g = OceanGrid.load(args.grid)
    m = OceanModel(g, OceanParams(sss_piston_m_per_day=None), xp)
    T, S = _stratified(g)
    st = m.new_state(T, S)
    days = args.days or 30
    t0 = time.time()
    worst = []
    for n in range(int(days * 24)):
        st, _ = m.step(st, {})
        if (n + 1) % 24 == 0:
            worst.append(float(max(abs(host(st["u"])).max(), abs(host(st["v"])).max())))
    return {"pass": max(worst) <= 5e-3, "days": days, "max_speed_m_s": max(worst),
            "daily_max": worst[-5:], "seconds_per_step": (time.time() - t0) / (days * 24),
            "note": "rho' is taken at the nominal level pressure against a fixed per-level "
                    "reference, so horizontally uniform T and S give rho' = 0 exactly: "
                    "what remains is the free surface and rounding"}


def _books(m, st):
    b = host(st["books"])
    heat, salt, vol = m.contents(st)
    ii = BOOKS.index
    return {"heat_rel": (heat - b[ii("heat_content0_j")] - b[ii("heat_sources_j")]) / heat,
            "salt_rel": (salt - b[ii("salt_content0")] - b[ii("salt_sources")]) / salt,
            "volume_rel": (vol - b[ii("volume0_m3")]) / vol}


def gate_cons(xp, args):
    g = OceanGrid.load(args.grid)
    m = OceanModel(g, OceanParams(sss_piston_m_per_day=None, sponge=False), xp)
    T, S = _stratified(g)
    st = m.new_state(T, S)
    rng = np.random.default_rng(3)
    steps = args.steps or 1000
    shape = (g.ny, g.nx)
    worst = {"heat_rel": 0.0, "salt_rel": 0.0, "volume_rel": 0.0}
    frazil = 0.0
    t0 = time.time()
    for n in range(steps):
        fc = {"q_ns": xp.asarray(rng.normal(-30, 150, shape)),
              "q_sw": xp.asarray(np.clip(rng.normal(150, 100, shape), 0, None)),
              "fw": xp.asarray(rng.normal(0, 5e-5, shape)),
              "taux": xp.asarray(rng.normal(0, 0.1, shape)),
              "tauy": xp.asarray(rng.normal(0, 0.1, shape))}
        st, d = m.step(st, fc)
        if (n + 1) % 50 == 0 or n == steps - 1:
            bk = _books(m, st)
            for k in worst:
                worst[k] = max(worst[k], abs(bk[k]))
    b = host(st["books"])
    ok = worst["heat_rel"] < 1e-12 and worst["salt_rel"] < 1e-12 and worst["volume_rel"] < 1e-12
    finite = bool(np.isfinite(host(st["theta"])).all() and np.isfinite(host(st["u"])).all())
    return {"pass": bool(ok and finite), "steps": steps, **{f"worst_{k}": v for k, v in worst.items()},
            "finite": finite, "frazil_j": float(b[BOOKS.index("frazil_j")]),
            "max_speed": float(max(abs(host(st["u"])).max(), abs(host(st["v"])).max())),
            "seconds_per_step": (time.time() - t0) / steps,
            "note": "random fluxes each step (heat N(-30,150) W m-2, SW, fresh water, stress "
                    "N(0,0.1) N m-2); relative book residuals of the whole ocean"}


def gate_kelvin(xp, args):
    """Aquaplanet ocean 78S-65N flat 4000 m; a first-mode thermocline bump on
    the equator; measure the eastward Kelvin and westward Rossby speeds."""
    lat = row_edges()
    lon = np.arange(361, dtype=float)
    z = level_edges()
    z = z[z <= 4000.0 + 1e-6]
    if z[-1] < 4000.0:
        z = np.append(z, 4000.0)
    depth = np.full((lat.size - 1, 360), 4000.0)
    g = idealized(lat_edges_deg=lat, lon_edges_deg=lon, z_edges_m=z, depth_m=depth)
    p = OceanParams(gm=False, sponge=False, kappa_h=0.0, sss_piston_m_per_day=None, kpp=False)
    m = OceanModel(g, p, xp)
    zc = g.z_t_nom
    # Two-layer-equivalent stratification: a sharp thermocline at 150 m.
    T0 = 8.0 + 9.0 * np.tanh((150.0 - zc) / 25.0) + 4.0
    # N^2 at the interfaces from the model's own equation of state (both
    # neighbours at the interface's reference pressure).
    zw = g.z_edges_m[1:-1]
    pw = RHO0 * GRAVITY * zw
    rho_up = eos.density(T0[:-1], 35.0, pw)
    rho_dn = eos.density(T0[1:], 35.0, pw)
    n2 = -(GRAVITY / RHO0) * (rho_up - rho_dn) / np.diff(zc)
    n2 = np.maximum(n2, 1e-9)
    c1, c2 = vertical_modes(g, n2, 2)
    # displacement bump: T(z - zeta), zeta Gaussian (lon 180, lat 0)
    LON, LAT = np.meshgrid(g.lon_t, g.lat_t)
    amp = 10.0 * np.exp(-((LON - 180.0) / 8.0) ** 2 - (LAT / 3.0) ** 2)
    zc3 = zc[:, None, None]
    T = 8.0 + 9.0 * np.tanh((150.0 + amp[None] - zc3) / 25.0) + 4.0
    st = m.new_state(T, np.full_like(T, 35.0))
    k150 = int(np.argmin(np.abs(zc - 150.0)))
    j0 = np.argsort(np.abs(g.lat_t))[:2]
    track = []
    days = args.days or 60
    for n in range(int(days * 24)):
        st, _ = m.step(st, {})
        if (n + 1) % 24 == 0:
            th = host(st["theta"])[k150]
            anom = th[j0].mean(axis=0) - (8.0 + 9.0 * np.tanh(0.0) + 4.0)
            # Rossby: first meridional mode peaks off the equator (~3-4 deg)
            jr = np.nonzero((np.abs(g.lat_t) > 2.0) & (np.abs(g.lat_t) < 5.0))[0]
            ranom = th[jr].mean(axis=0) - 12.0
            east = np.where(g.lon_t >= 180.0, anom, 0.0)
            west = np.where(g.lon_t <= 180.0, ranom, 0.0)

            def peak(series):
                # sub-grid peak by a parabola through the maximum and its neighbours
                i = int(np.argmax(series))
                a, b, c = series[i - 1], series[i], series[(i + 1) % series.size]
                den = a - 2 * b + c
                off = 0.5 * (a - c) / den if den != 0 else 0.0
                return float(g.lon_t[i] + np.clip(off, -0.5, 0.5)), float(b)

            kl, ka = peak(east)
            rl, ra = peak(west)
            track.append({"day": (n + 1) / 24.0, "kelvin_lon": kl, "kelvin_amp": ka,
                          "rossby_lon": rl, "rossby_amp": ra})
    R = 6.37122e6
    def speed(key, d0, d1, sign):
        a = [t for t in track if d0 <= t["day"] <= d1]
        x = np.array([t[key] for t in a]); tt = np.array([t["day"] for t in a])
        slope = np.polyfit(tt, x, 1)[0]  # deg/day
        return sign * slope * math.pi / 180.0 * R / 86400.0
    ck = speed("kelvin_lon", 5, min(25, days), 1.0)
    cr = -speed("rossby_lon", 10, min(40, days), 1.0)
    errk = (ck - c1) / c1
    errr = (cr - c1 / 3.0) / (c1 / 3.0)
    return {"pass": abs(errk) <= 0.03 and abs(errr) <= 0.05, "c1_modes_m_s": c1, "c2_m_s": c2,
            "kelvin_speed_m_s": ck, "kelvin_rel_err": errk, "rossby_speed_m_s": cr,
            "rossby_rel_err": errr, "amplitude_day1": track[0]["kelvin_amp"],
            "amplitude_last": track[-1]["kelvin_amp"], "days": days, "track": track[::5]}


def gate_gyre(xp, args):
    lat = np.arange(15.0, 55.0 + 1e-9, 1.0)
    lon = np.arange(0.0, 61.0, 1.0)
    z = np.array([0.0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    depth = np.full((lat.size - 1, lon.size - 1), 4000.0)
    g = idealized(lat_edges_deg=lat, lon_edges_deg=lon, z_edges_m=z, depth_m=depth, periodic=False)
    p = OceanParams(gm=False, sponge=False, kappa_h=0.0, sss_piston_m_per_day=None)
    m = OceanModel(g, p, xp)
    T = np.full((g.nz, g.ny, g.nx), 10.0)
    st = m.new_state(T, np.full_like(T, 35.0))
    L = 40.0
    tau = 0.1 * np.cos(np.pi * (g.lat_t - 15.0) / L)
    taux = np.broadcast_to(tau[:, None], (g.ny, g.nx)).copy()
    fc = {"taux": xp.asarray(taux), "tauy": xp.zeros((g.ny, g.nx))}
    days = args.days or 120
    for n in range(int(days * 24)):
        st, _ = m.step(st, fc)
    V = host(st["vbt"])  # depth-integrated, at V points (north faces)
    R = 6.37122e6
    phi_v = np.deg2rad(g.lat_edges_deg[1:])
    beta = 2 * 7.292e-5 * np.cos(phi_v) / R
    # curl tau at V latitude: -d tau_x / dy
    curl = 0.1 * (np.pi / (np.deg2rad(L) * R)) * np.sin(np.pi * (g.lat_edges_deg[1:] - 15.0) / L)
    sverdrup = curl / (RHO0 * beta)
    rows = []
    for j in range(5, g.ny - 6, 4):
        interior = V[j, 20:50].mean()
        rows.append({"lat": float(g.lat_edges_deg[j + 1]), "model_m2_s": float(interior),
                     "sverdrup_m2_s": float(sverdrup[j]),
                     "rel_err": float((interior - sverdrup[j]) / sverdrup[j])})
    jm = int(np.argmin(np.abs(g.lat_edges_deg[1:] - 35.0)))
    mid = {"lat": float(g.lat_edges_deg[jm + 1]), "model": float(V[jm, 20:50].mean()),
           "sverdrup": float(sverdrup[jm])}
    err = (mid["model"] - mid["sverdrup"]) / mid["sverdrup"]
    return {"pass": abs(err) <= 0.10, "mid_basin": mid, "rel_err": err, "rows": rows, "days": days,
            "max_speed": float(abs(host(st["u"])).max())}


def gate_restart(xp, args):
    g = OceanGrid.load(args.grid)
    m = OceanModel(g, OceanParams(), xp)
    T, S = _stratified(g)
    rng = np.random.default_rng(4)
    shape = (g.ny, g.nx)
    forcings = [{"q_ns": xp.asarray(rng.normal(-30, 150, shape)),
                 "taux": xp.asarray(rng.normal(0, 0.1, shape)),
                 "tauy": xp.asarray(rng.normal(0, 0.1, shape)),
                 "fw": xp.asarray(rng.normal(0, 5e-5, shape))} for _ in range(6)]
    a = m.new_state(T, S)
    for fc in forcings:
        a, _ = m.step(a, fc)
    b = m.new_state(T, S)
    for fc in forcings[:3]:
        b, _ = m.step(b, fc)
    import io
    buf = io.BytesIO()
    np.savez(buf, **{k: host(v) for k, v in b.items()})
    buf.seek(0)
    loaded = np.load(buf)
    m2 = OceanModel(g, OceanParams(), xp)
    c = {k: xp.asarray(loaded[k]) for k in loaded.files}
    for fc in forcings[3:]:
        c, _ = m2.step(c, fc)
    diffs = {k: int((host(a[k]) != host(c[k])).sum()) for k in a}
    return {"pass": all(v == 0 for v in diffs.values()), "differing_values": diffs}


def gate_parity(xp, args):
    """GPU against the CPU reference, two forced steps on the real grid."""
    import cupy as cp

    g = OceanGrid.load(args.grid)
    T, S = _stratified(g)
    rng = np.random.default_rng(5)
    shape = (g.ny, g.nx)
    fcs = [{"q_ns": rng.normal(-30, 150, shape), "q_sw": np.clip(rng.normal(150, 100, shape), 0, None),
            "taux": rng.normal(0, 0.1, shape), "tauy": rng.normal(0, 0.1, shape),
            "fw": rng.normal(0, 5e-5, shape)} for _ in range(2)]
    out = {}
    states = {}
    for mod in (np, cp):
        m = OceanModel(g, OceanParams(), mod)
        st = m.new_state(T, S)
        for fc in fcs:
            st, _ = m.step(st, {k: mod.asarray(v) for k, v in fc.items()})
        states[mod.__name__] = {k: host(v) for k, v in st.items()}
    for k in ("theta", "salt", "u", "v", "eta"):
        a, b = states["numpy"][k], states["cupy"][k]
        scale = max(float(np.abs(a).max()), 1e-30)
        out[k] = float(np.abs(a - b).max() / scale)
    return {"pass": out["theta"] < 1e-6 and out["u"] < 1e-3, "max_rel_diff": out,
            "note": "momentum is float32 and the two libraries round sums differently"}


def gate_slump(xp, args):
    """GM: tilted isopycnals in a closed flat channel flatten; volume unchanged."""
    lat = np.arange(30.0, 41.0, 1.0)
    lon = np.arange(0.0, 21.0, 1.0)
    z = level_edges()
    z = z[z <= 1000.0 + 1e-9]
    depth = np.full((lat.size - 1, lon.size - 1), z[-1])
    g = idealized(lat_edges_deg=lat, lon_edges_deg=lon, z_edges_m=z, depth_m=depth, periodic=False)
    zc = g.z_t_nom[:, None, None]
    x = np.clip((g.lon_t[None, None, :] - 10.0) / 2.5, -1.0, 1.0)
    T = 10.0 + 8.0 * np.tanh((300.0 + 150.0 * x - zc) / 100.0) + 0 * g.lat_t[None, :, None]
    k = int(np.argmin(np.abs(g.z_t_nom - 300.0)))

    def tilt(s):
        th = host(s["theta"])
        return float(th[k, :, 13].mean() - th[k, :, 6].mean())

    out = {}
    for gm in (True, False):
        p = OceanParams(sponge=False, kappa_h=0.0, sss_piston_m_per_day=None, gm=gm)
        m = OceanModel(g, p, xp)
        st = m.new_state(T, np.full_like(T, 35.0))
        t0 = tilt(st)
        vol0 = m.contents(st)[2]
        for n in range(int((args.days or 60) * 24)):
            st, _ = m.step(st, {})
        out["gm" if gm else "no_gm"] = {"tilt_start_K": t0, "tilt_end_K": tilt(st),
                                        "volume_rel_change": (m.contents(st)[2] - vol0) / vol0}
    a, b = out["gm"], out["no_gm"]
    return {"pass": abs(a["tilt_end_K"]) < abs(b["tilt_end_K"])
            and abs(a["volume_rel_change"]) < 1e-12, **out,
            "note": "a 5-deg-wide front across a closed 20-deg channel; T difference across "
                    "the front at 300 m with GM on and off (the resolved flow slumps it too)"}


GATES = {"eos": gate_eos, "tridiag": gate_tridiag, "remap": gate_remap, "kpp-kato": gate_kpp_kato,
         "kpp-convective": gate_kpp_convective, "rest": gate_rest, "cons": gate_cons,
         "kelvin": gate_kelvin, "gyre": gate_gyre, "restart": gate_restart, "parity": gate_parity,
         "slump": gate_slump}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="*", default=["eos", "tridiag", "remap"])
    ap.add_argument("--xp", default="numpy")
    ap.add_argument("--grid", default="ocean-grid-v1.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args(argv)
    if args.xp == "cupy":
        import cupy as xp
    else:
        xp = np
    results = {}
    for name in args.gates:
        t0 = time.time()
        try:
            r = GATES[name](xp, args)
        except Exception as exc:  # report and continue
            import traceback
            r = {"pass": False, "error": repr(exc), "traceback": traceback.format_exc()}
        r["wall_s"] = round(time.time() - t0, 2)
        r["xp"] = args.xp
        results[name] = r
        print(json.dumps({name: r}, default=float), flush=True)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1, default=float)
    return 0 if all(r.get("pass") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
