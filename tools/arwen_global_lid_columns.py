#!/usr/bin/env python3
"""Model-top radiation against a resolved upper atmosphere, on RFMIP columns.

Each RFMIP site (experiment 0, present day) is put on the model's own
40-level grid below the 1 hPa lid (T, H2O, O3 interpolated in ln p from the
site's 60 layers).  Above the lid the columns are:

* REF  -- the site's own atmosphere from 1 hPa to the coefficient floor on
          ``--ref-layers`` log-spaced layers (temperature, H2O, O3 from the
          site, which reaches 0.01 hPa);
* SHIP -- the shipped above-model treatment (``_extend_above_model_profile``:
          one LW layer 1 hPa -> floor on WRF's climatology shape, one SW
          layer isothermal at the top interface), ozone above the lid from
          the site or from the shipped median-RFMIP climatology;
* LIVE -- the driver's own above-model path as shipped in the installed
          tree (``--live``): whatever ``_extend_above_model_profile`` builds.

LW and SW are solved with the carried RRTMGP gas optics and RTE solvers
(the same calls the driver makes) and the heating of the top model layers
is reported as the RFMIP profile-weighted mean, K/day of T and of theta.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import cupy as cp
from netCDF4 import Dataset

from arwen_global.core import rrtmgp as R
from arwen_global.vertical import HybridCoordinate
from gpuwm.core.rfmip_upstream import fetch_rfmip
from gpuwm.core.rrtmgp import load_trace_climatology

G, CP_AIR, RD = 9.80665, 1004.5, 287.04
KAPPA = RD / CP_AIR
DT = R.DTYPE


def interp_lnp(p_src, v_src, p_dst):
    out = np.empty(p_dst.shape)
    for i in range(p_src.shape[0]):
        o = np.argsort(p_src[i])
        out[i] = np.interp(np.log(p_dst[i]), np.log(p_src[i][o]), v_src[i][o])
    return out


def load_sites():
    src = fetch_rfmip("rfmip-clear-sky-inputs.nc")
    with Dataset(src) as nc:
        nc.set_auto_mask(False)
        d = {k: np.asarray(nc[k][:], np.float64) for k in (
            "pres_layer", "pres_level", "profile_weight", "surface_emissivity",
            "surface_albedo", "solar_zenith_angle", "total_solar_irradiance",
            "lat")}
        for k in ("temp_layer", "temp_level", "water_vapor", "ozone",
                  "surface_temperature"):
            d[k] = np.asarray(nc[k][0], np.float64)
        gm = {}
        for gas, name in R._RFMIP_GAS_NAMES.items():
            v = nc[name + "_GM"]
            scale = float(getattr(v, "units", "1").replace(" ", ""))
            gm[gas] = float(np.asarray(v[0], np.float64) * scale)
    d["gm"] = gm
    return d


def build_model_grid(sites, nlev=40, p_top=100.0):
    vert = HybridCoordinate.surface_stretched(nlev, p_top)
    a = np.asarray(vert.a_half_pa, np.float64)
    b = np.asarray(vert.b_half, np.float64)
    ps = sites["pres_level"].max(axis=1)
    plev = a[None, :] + b[None, :] * ps[:, None]
    if plev[0, 0] < plev[0, -1]:
        plev = plev[:, ::-1]
    plev = np.ascontiguousarray(plev)                       # bottom first
    play = np.sqrt(plev[:, :-1] * plev[:, 1:])
    return play, plev


def site_fields(sites, play):
    pl = sites["pres_layer"]
    t = interp_lnp(pl, sites["temp_layer"], play)
    q = interp_lnp(pl, sites["water_vapor"], play)
    o = interp_lnp(pl, sites["ozone"], play)
    return t, q, o


def vmr_array(tables, gm, q, o):
    v = np.zeros((*q.shape, tables.ngas + 1))
    v[..., tables.gas_index["h2o"]] = q
    v[..., tables.gas_index["o3"]] = o
    for gas, val in gm.items():
        if gas in tables.gas_index:
            v[..., tables.gas_index[gas]] = val
    return v


def _d(x):
    return cp.asarray(np.ascontiguousarray(x), dtype=DT)


def solve(play, plev, tlay, tlev, q, o, sites, kind):
    tables = R.load_gas_tables(kind)
    vmr = vmr_array(tables, sites["gm"], q, o)
    dplay, dplev, dtlay, dvmr = _d(play), _d(plev), _d(tlay), _d(vmr)
    opt = R.gas_optics(tables, dplay, dplev, dtlay, dvmr)
    ncol = play.shape[0]
    if kind == "lw":
        src = R.planck_sources(tables, dplay, dplev, dtlay, _d(tlev),
                               _d(sites["surface_temperature"]), dvmr)
        emis = cp.broadcast_to(_d(sites["surface_emissivity"])[:, None],
                               (ncol, tables.nband))
        emis = R._expand_band_to_gpoint(emis, tables, "emis")
        f = R.lw_rte(opt.tau, src.lay_source, src.lev_source,
                     src.sfc_source, emis, top_at_1=False)
        up, dn = f.flux_up, f.flux_dn
    else:
        tau, ssa, g = R.delta_scale(opt.tau, opt.ssa, opt.g)
        mu_raw = cp.cos(_d(sites["solar_zenith_angle"]) * DT(np.pi / 180))
        day = mu_raw > 0
        mu = cp.where(day, mu_raw, DT(1.0))
        alb = cp.ascontiguousarray(cp.broadcast_to(
            _d(sites["surface_albedo"])[:, None], (ncol, tables.ngpt)))
        inc = R._normalized_solar_incident(tables.solar_source,
                                           sites["total_solar_irradiance"])
        f = R.sw_rte(tau, ssa, g, mu, alb, alb, inc, top_at_1=False)
        up = cp.where(day[:, None], f.flux_up, DT(0))
        dn = cp.where(day[:, None], f.flux_dn, DT(0))
    up = cp.asnumpy(up).astype(np.float64)
    dn = cp.asnumpy(dn).astype(np.float64)
    net = dn - up
    heat = G / CP_AIR * (net[:, 1:] - net[:, :-1]) / (plev[:, :-1] - plev[:, 1:])
    return heat * 86400.0


def tlev_of(play, plev, tlay):
    return cp.asnumpy(R._interface_temperatures(
        _d(play), _d(plev), _d(tlay))).astype(np.float64)


def ref_column(sites, play_m, plev_m, nref):
    floor = R.RRTMGP_TOA_PRESSURE_PA * 1.0001
    top = plev_m[:, -1]
    edges = np.exp(np.linspace(np.log(top), np.log(np.full_like(top, floor)),
                               nref + 1, axis=1)) if False else np.exp(
        np.log(top)[:, None] + np.linspace(0.0, 1.0, nref + 1)[None, :]
        * (np.log(floor) - np.log(top))[:, None])
    plev = np.concatenate([plev_m, edges[:, 1:]], axis=1)
    play = np.concatenate([play_m, np.sqrt(edges[:, :-1] * edges[:, 1:])], axis=1)
    t, q, o = site_fields(sites, play)
    return play, plev, t, q, o


def extended_column(sites, play_m, plev_m, t_m, q_m, o_m, kind, ozone_mode):
    tlev_m = tlev_of(play_m, plev_m, t_m)
    prof = R._extend_above_model_profile(
        play_m, plev_m, t_m, tlev_m, q_m, p_top=float(plev_m[0, -1]),
        kind=kind, xp=cp, validate_top=False)
    play = cp.asnumpy(prof.play).astype(np.float64)
    plev = cp.asnumpy(prof.plev).astype(np.float64)
    tlay = cp.asnumpy(prof.tlay).astype(np.float64)
    tlev = cp.asnumpy(prof.tlev).astype(np.float64)
    q = cp.asnumpy(prof.qv).astype(np.float64)
    nup = prof.upper_nlay
    up_play = play[:, play_m.shape[1]:]
    if ozone_mode == "climatology":
        clim = load_trace_climatology()
        pc, oc = np.asarray(clim.pressure_layer_pa), np.asarray(clim.ozone_vmr)
        order = np.argsort(pc)
        o_up = np.interp(np.log(up_play), np.log(pc[order]), oc[order])
        o_mod = np.interp(np.log(play_m), np.log(pc[order]), oc[order])
    else:
        _, _, o_up = site_fields(sites, up_play)
        o_mod = o_m
    o = np.concatenate([o_mod, o_up], axis=1)
    return play, plev, tlay, tlev, q, o, nup


def report(name, heat_lw, heat_sw, w, play_m, nmodel, levels=4):
    rows = {}
    for k in range(levels):
        idx = nmodel - 1 - k
        lw = float(np.sum(w * heat_lw[:, idx]) / np.sum(w))
        sw = float(np.sum(w * heat_sw[:, idx]) / np.sum(w))
        exner = float(np.sum(w * (play_m[:, idx] / 1e5) ** KAPPA) / np.sum(w))
        rows[f"lev{k}"] = {"p_hpa": float(np.median(play_m[:, idx])) / 100,
                           "lw_T": lw, "sw_T": sw, "net_T": lw + sw,
                           "net_theta": (lw + sw) / exner}
    print(name)
    for k, r in rows.items():
        print(f"  {k} {r['p_hpa']:6.2f} hPa  LW {r['lw_T']:7.2f}  SW {r['sw_T']:7.2f}"
              f"  net {r['net_T']:6.2f} K/d T  {r['net_theta']:7.2f} K/d theta", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-layers", type=int, default=24)
    ap.add_argument("--out", default="lid-columns.json")
    args = ap.parse_args()
    sites = load_sites()
    w = sites["profile_weight"]
    play_m, plev_m = build_model_grid(sites)
    nmodel = play_m.shape[1]
    t_m, q_m, o_m = site_fields(sites, play_m)
    out = {"model_top_interfaces_pa": plev_m[0, -5:][::-1].tolist()}
    for nref in (args.ref_layers, 2 * args.ref_layers):
        play, plev, t, q, o = ref_column(sites, play_m, plev_m, nref)
        tlev = tlev_of(play, plev, t)
        lw = solve(play, plev, t, tlev, q, o, sites, "lw")
        sw = solve(play, plev, t, tlev, q, o, sites, "sw")
        out[f"ref{nref}"] = report(
            f"REF: site atmosphere above 1 hPa on {nref} layers", lw, sw, w,
            play_m, nmodel)
    for mode in ("site", "climatology"):
        res = {}
        for kind in ("lw", "sw"):
            play, plev, tlay, tlev, q, o, nup = extended_column(
                sites, play_m, plev_m, t_m, q_m, o_m, kind, mode)
            res[kind] = solve(play, plev, tlay, tlev, q, o, sites, kind)
            res[kind + "_n"] = nup
        out[f"live_{mode}"] = report(
            f"LIVE above-model path (LW {res['lw_n']} / SW {res['sw_n']} "
            f"layers), ozone: {mode}", res["lw"], res["sw"], w, play_m, nmodel)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
