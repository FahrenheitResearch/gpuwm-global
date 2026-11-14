#!/usr/bin/env python3
"""Grell-Freitas shallow-arm probe on a restarted state.

    PROBE_OUT=probe.npz PROBE_EVERY_S=10800 PROBE_MAX_FIRINGS=8 \\
        python tools/arwen_global_shallow_probe.py run CONFIG --outdir DIR \\
        --restart CHECKPOINT --branch-from-other-config --until-s T

Every argument after the script name goes to ``gpuwm-global``.  At a
probe firing (the first cumulus call at or after each PROBE_EVERY_S
boundary, per latitude band) the model's own call runs untouched; then the
same column inputs (``NativePhysicsRuntime._cumulus_inputs``) are handed to
two fresh Grell-Freitas adapters, the shipped kernel and the kernel with
``shallow_buoyancy_from_cloud_base``, and, for the columns of the regime
boxes below, to the shallow stage alone (``gf_shallow_stage``) with its
capture lanes, so each box column carries the shallow arm's exit code at
every stage, its cloud base, its integrated buoyancy and the tendencies it
returns under both kernels.

Writes ``$PROBE_OUT`` with ``.fNN`` per firing: whole-grid planes
``<variant>__xmb_shallow``, ``__ktop_shallow``, ``__kbcon_shallow``,
``__rainc``, the column integrals of the moisture, condensate and heat
rates (``__colq``, ``__colqc`` kg m-2 s-1, ``__colh`` W m-2) and the
masks; and for the box columns (``box__*``, columns along the last axis)
the stage captures and the column state (t, qv, qc, p, z, kpbl, pblh, hfx,
qfx).  Read-only: nothing the probe computes reaches the run.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time

import numpy as np

from arwen_global.physics import native_runtime as _nr

OUT = os.environ.get("PROBE_OUT", "shallow-probe.npz")
EVERY_S = float(os.environ.get("PROBE_EVERY_S", "0") or 0.0)
MAX_FIRINGS = int(os.environ.get("PROBE_MAX_FIRINGS", "1") or 1)
VARIANTS = {"wrf": (False, False, False, 0), "cb": (True, False, False, 0),
            "cbqe": (True, False, True, 0), "cbqed": (True, False, True, 500)}

# Regime boxes (lat0, lat1, lon0, lon1 in degrees east), open water only.
BOXES = {
    "trades_se_pac": (-20.0, 0.0, 210.0, 260.0),
    "trades_ne_pac": (10.0, 30.0, 180.0, 220.0),
    "trades_s_ind": (-30.0, -10.0, 60.0, 100.0),
    "sc_peru": (-20.0, -10.0, 270.0, 280.0),
    "sc_namibia": (-20.0, -10.0, 0.0, 10.0),
    "sc_calif": (20.0, 30.0, 230.0, 240.0),
    "warm_pool": (-10.0, 10.0, 130.0, 170.0),
}

SH_LEV_KEEP = ("dbyo", "dbyt", "hco", "heso_cup", "heo_cup", "po_cup0",
               "zo_cup", "zuo", "qrco", "outt", "outq", "outqc")
STATE: dict = {"firings": {}, "schedule": {}}


def _host(v):
    if hasattr(v, "get") and not isinstance(v, (np.ndarray, dict)):
        return np.asarray(v.get())
    return np.asarray(v)


def due_firing(schedule, key, time_s):
    entry = schedule.get(key)
    if entry is None:
        schedule[key] = {"t0": float(time_s), "n": 1}
        return 0
    if EVERY_S <= 0 or entry["n"] >= MAX_FIRINGS:
        return None
    if float(time_s) >= entry["t0"] + entry["n"] * EVERY_S:
        entry["n"] += 1
        return entry["n"] - 1
    return None


class _Readings:
    """Extend the seam's closure readings with the shallow arm's, for the
    probe's own calls only."""

    def __init__(self, gf):
        self.gf = gf

    def __enter__(self):
        self.saved = (self.gf.CLOSURE_READING_SCALARS, self.gf.CLOSURE_READING_INTEGERS)
        self.gf.CLOSURE_READING_SCALARS = self.saved[0] + ("xmb_shallow", "prets", "cutens")
        self.gf.CLOSURE_READING_INTEGERS = self.saved[1] + (
            "ktop_shallow", "kbcon_shallow", "k22_shallow")

    def __exit__(self, *exc):
        self.gf.CLOSURE_READING_SCALARS, self.gf.CLOSURE_READING_INTEGERS = self.saved
        return False


def _shallow_stage(gf, xp, atm, fields, state, driver, cfg, cols, flag, top=False, qe=False, dmin=0):
    """gf_shallow_stage on the flat column indices ``cols`` with the inputs
    gf.cu's driver builds for its shallow arm (:4109-4160)."""
    from gpuwm.core.kf import _model_clock_dt

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "gf_wrf461_oracle"))
    import gf_field_lists as fl

    nz, ny, nx = state.p.shape
    f32 = xp.float32

    def col(a):
        return xp.ascontiguousarray(a.reshape(nz, ny * nx)[:, cols].T.astype(f32))

    def sca(a):
        return xp.ascontiguousarray(a.reshape(ny * nx)[cols].astype(f32))

    dt = f32(_model_clock_dt(cfg))
    t = col(atm["temperature"])
    q = xp.maximum(col(atm["qv"]), f32(1.0e-8))
    pi = col(atm["exner"])
    dz = col(atm["dz"])
    rthbl = col(driver.gf_rthblten)
    rqvbl = col(driver.gf_rqvblten)
    ter = xp.maximum(sca(state.ht), f32(0.0))
    zo = xp.empty_like(dz)
    zo[:, 0] = ter + f32(0.5) * dz[:, 0]
    for k in range(1, nz):
        zo[:, k] = zo[:, k - 1] + f32(0.5) * (dz[:, k - 1] + dz[:, k])
    lanes = {"zo": zo, "t2d": t, "q2d": q, "tshall": t + rthbl * pi * dt,
             "qshall": q + rqvbl * dt, "po": col(atm["pressure"]) * f32(0.01),
             "dhdt": (f32(1004.6) * rthbl * pi + f32(2.5e6) * rqvbl).astype(f32),
             "rhoi": col(atm["rho"])}
    n = int(cols.size)
    lvin = xp.stack([lanes[name] for name in fl.SH_IN_LEV], axis=1)
    p8w = col(atm["p_interface"][:-1])
    scin = xp.stack([ter, p8w[:, 0] * f32(0.01), sca(fields["hfx"]), sca(fields["qfx"]),
                     sca(fields["xland"]), xp.full((n,), dt, f32), xp.zeros((n,), f32)],
                    axis=1)
    iin = xp.ascontiguousarray(fields["kpbl"].reshape(ny * nx)[cols].astype(xp.int32))
    lvin = xp.ascontiguousarray(lvin)
    scin = xp.ascontiguousarray(scin)
    mod = gf._gf_module(nz, shallow_buoyancy_from_cloud_base=flag,
                        shallow_blqe_to_cloud_base=qe,
                        shallow_depth_min_m=dmin)
    fn = mod.get_function("gf_shallow_stage")
    lev = xp.zeros((n, len(fl.SH_LEV_FIELDS), nz), f32)
    sc = xp.zeros((n, len(fl.SH_SCA_FIELDS)), f32)
    isc = xp.zeros((n, len(fl.SH_ISCA_FIELDS)), xp.int32)
    tile = 4096
    ws = xp.empty(gf.gf_workspace_floats(nz, tile), dtype=f32)
    for t0 in range(0, n, tile):
        t1 = min(t0 + tile, n)
        fn((((t1 - t0) + gf.GF_BLOCK - 1) // gf.GF_BLOCK,), (gf.GF_BLOCK,),
           (lvin[t0:t1], scin[t0:t1], iin[t0:t1], lev[t0:t1], sc[t0:t1], isc[t0:t1], ws,
            np.int32(0), np.int32(t1 - t0), np.int32(nz)))
    out = {}
    for name in SH_LEV_KEEP:
        out[f"lev__{name}"] = lev[:, fl.SH_LEV_FIELDS.index(name), :].T
    for j, name in enumerate(fl.SH_SCA_FIELDS):
        out[f"sca__{name}"] = sc[:, j]
    for j, name in enumerate(fl.SH_ISCA_FIELDS):
        out[f"isc__{name}"] = isc[:, j]
    return out


def _probe(runtime, batch, persistent, cfg, entry, pbl_out, own, firing):
    from arwen_global.core import gf

    xp = batch.xp
    t0 = time.time()
    atm, fields, state, driver = runtime._cumulus_inputs(batch, persistent, entry, pbl_out)
    nz, ny, nx = state.p.shape
    dp = (atm["p_interface"][:-1] - atm["p_interface"][1:]) / xp.float32(9.80665)
    lat = batch.arrays["latitude_deg"]
    lon = xp.mod(batch.arrays["longitude_deg"], 360.0)
    s = batch.surface
    water = (xp.asarray(s.land_fraction) < 0.5)
    if s.sea_ice_fraction is not None:
        water &= xp.asarray(s.sea_ice_fraction) <= 0
    band = {"lat": lat, "lon": lon, "water": water.astype(xp.float32)}
    model = runtime._cumulus
    flat_masks = {}
    for name, (a0, a1, o0, o1) in BOXES.items():
        m = water & (lat >= a0) & (lat < a1) & (lon >= o0) & (lon < o1)
        flat_masks[name] = xp.nonzero(m.reshape(-1))[0]
    cols = xp.concatenate([flat_masks[k] for k in BOXES]) if flat_masks else None
    box_id = xp.concatenate([xp.full(flat_masks[k].shape, i, xp.int32)
                             for i, k in enumerate(BOXES)])
    check = {}
    for tag, (flag, top, qe, dmin) in VARIANTS.items():
        g = gf.GrellFreitas(
            column_chunk=model.column_chunk,
            updraft_only_when_downdraft_dry=model.updraft_only_when_downdraft_dry,
            resolved_convergence_closure=model.resolved_convergence_closure,
            resolved_convergence_floor_percent=model.resolved_convergence_floor_percent,
            shallow_buoyancy_from_cloud_base=flag,
            shallow_blqe_to_cloud_base=qe, shallow_depth_min_m=dmin)
        g.bind_driver(driver)
        with _Readings(gf):
            r = g(atmosphere=atm, fields=fields, state=state, cfg=cfg)
        d = g.last_column_diagnostics
        for key in ("xmb_shallow", "prets", "ktop_shallow", "kbcon_shallow", "k22_shallow",
                    "ierr_deep"):
            band[f"{tag}__{key}"] = d[f"gf_{key}"].astype(xp.float32)
        band[f"{tag}__rainc"] = r.rainc
        rqc = r.rqccuten if r.rqicuten is None else r.rqccuten + r.rqicuten
        band[f"{tag}__colq"] = xp.sum(r.rqvcuten * dp, axis=0)
        band[f"{tag}__colqc"] = xp.sum(rqc * dp, axis=0)
        band[f"{tag}__colh"] = xp.sum(r.rthcuten * atm["exner"] * dp, axis=0) * xp.float32(1004.6)
        if tag == "wrf":
            check = {"rthcuten": float(xp.max(xp.abs(r.rthcuten - own.rthcuten))),
                     "rqvcuten": float(xp.max(xp.abs(r.rqvcuten - own.rqvcuten)))}
        if cols is not None and cols.size:
            box = {f"box__{tag}__rqv": r.rqvcuten.reshape(nz, -1)[:, cols],
                   f"box__{tag}__rqc": rqc.reshape(nz, -1)[:, cols],
                   f"box__{tag}__rth": r.rthcuten.reshape(nz, -1)[:, cols]}
            st = _shallow_stage(gf, xp, atm, fields, state, driver, cfg, cols, flag, top, qe, dmin)
            box.update({f"box__{tag}__{k}": v for k, v in st.items()})
            band.update(box)
    if cols is not None and cols.size:
        flat3 = lambda a: a.reshape(nz, -1)[:, cols]
        band.update({
            "box__id": box_id, "box__lat": lat.reshape(-1)[cols], "box__lon": lon.reshape(-1)[cols],
            "box__t": flat3(atm["temperature"]), "box__qv": flat3(atm["qv"]),
            "box__qc": flat3(batch.arrays["qc"]), "box__p": flat3(atm["pressure"]),
            "box__dz": flat3(atm["dz"]),
            "box__kpbl": fields["kpbl"].reshape(-1)[cols],
            "box__pblh": fields["pblh"].reshape(-1)[cols],
            "box__hfx": fields["hfx"].reshape(-1)[cols], "box__qfx": fields["qfx"].reshape(-1)[cols],
            "box__rthbl": flat3(driver.gf_rthblten), "box__rqvbl": flat3(driver.gf_rqvblten),
        })
    rec = STATE["firings"].setdefault(firing, {"bands": [], "meta": []})
    rec["bands"].append({k: _host(v) for k, v in band.items()})
    rec["meta"].append({"band": None if batch.band is None else list(batch.band),
                        "time_s": float(batch.time_s), "wrf_minus_model_max_abs": check,
                        "wall_s": time.time() - t0})
    print(f"[probe] firing {firing} band {batch.band} t={float(batch.time_s):.0f} s "
          f"check {check} {time.time() - t0:.1f} s", file=sys.stderr, flush=True)


_orig = _nr.NativePhysicsRuntime._cumulus_result


def _hooked(self, batch, persistent, cfg, entry, pbl_out):
    result = _orig(self, batch, persistent, cfg, entry, pbl_out)
    if self.options.cumulus_scheme == "gf":
        key = None if batch.band is None else tuple(batch.band)
        firing = due_firing(STATE["schedule"], key, batch.time_s)
        if firing is not None:
            _probe(self, batch, persistent, cfg, entry, pbl_out, result, firing)
    return result


def _dump():
    for firing, rec in sorted(STATE["firings"].items()):
        bands = rec["bands"]
        out = {}
        for k in bands[0]:
            if k.startswith("box__"):
                axis = -1
            else:
                axis = 0
            out[k] = np.concatenate([b[k] for b in bands], axis=axis) if len(bands) > 1 else bands[0][k]
        meta = {"schema": "arwen-global.shallow-probe/v1", "firing": firing,
                "bands": rec["meta"], "argv": sys.argv[1:], "boxes": BOXES,
                "variants": VARIANTS}
        out["meta"] = np.array(json.dumps(meta))
        stem = OUT[:-4] if OUT.endswith(".npz") else OUT
        path = f"{stem}.f{firing:02d}.npz"
        np.savez_compressed(path, **out)
        print(f"[probe] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    from arwen_global.cli import main

    _nr.NativePhysicsRuntime._cumulus_result = _hooked
    atexit.register(_dump)
    sys.exit(main(sys.argv[1:]))
