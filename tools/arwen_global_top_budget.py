#!/usr/bin/env python3
"""Per-process heating budget of the top model levels, read-only.

Runs ``gpuwm-global run ...`` (every argument is passed through) with three
read-only hooks and writes ``$TOPBUDGET_OUT`` (JSON) at exit:

* physics components: the native suite's own component marks
  (``NativePhysicsRuntime._observe``: rrtmgp, sfclay, noah, ysu, cumulus,
  morrison), booked as the area-weighted global-mean theta change of the
  top ``TOPBUDGET_LEVELS`` levels of the physics batch;
* radiation split: the held longwave and shortwave theta rates times the
  call's dt, the same levels;
* dynamics operators: the step's energy-ledger marks (physics_first,
  positivity_first, semilag_transport, semilag_implicit, diffusion,
  mass_fixer, physics_second, positivity_second), booked as the change of
  the spectral (0,0) theta coefficient of the top levels, scaled to the
  grid global mean once at the first mark.

Nothing here writes a prognostic array.  Accumulators stay on the device;
one host read at exit.  Levels are reported TOP FIRST (index 0 = model top).
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time

import numpy as np

NTOP = int(os.environ.get("TOPBUDGET_LEVELS", "6"))
OUT = os.environ.get("TOPBUDGET_OUT", "top-budget.json")

from arwen_global.physics import native_runtime as _nr  # noqa: E402
from arwen_global.insitu import ledger as _led  # noqa: E402

PHYS: dict = {"acc": {}, "order": [], "prev": None, "wsum": None, "calls": 0}
#: Zonal-mean accumulators per process, keyed by the batch's latitude rows
#: (one key per latitude band of a banded run): {key: {name: (NTOP, nrow)}}.
ZON: dict = {"acc": {}, "lat": {}, "prev": {}}
DYN: dict = {"acc": {}, "order": [], "prev": None, "scale": None,
             "series": [], "steps": 0, "scale_check": []}
META: dict = {"started": time.time()}


def _zon_add(key, name, value, xp):
    slot = ZON["acc"].setdefault(key, {})
    if name not in slot:
        slot[name] = xp.zeros(value.shape, dtype=xp.float64)
    slot[name] += value


def _phys_add(name, value, xp):
    if name not in PHYS["acc"]:
        PHYS["acc"][name] = xp.zeros((NTOP,), dtype=xp.float64)
        PHYS["order"].append(name)
    PHYS["acc"][name] += value


def _band_mean(xp, top_first, lat_plane):
    """Sum over rows of cos(lat) * zonal mean, (NTOP,) float64."""
    w = xp.cos(xp.deg2rad(lat_plane[:, 0].astype(xp.float64)))
    zonal = xp.mean(top_first, axis=-1, dtype=xp.float64)
    return xp.sum(zonal * w[None, :], axis=-1), xp.sum(w)


_orig_observe = _nr.NativePhysicsRuntime._observe


def _observe(self, name, batch):
    _orig_observe(self, name, batch)
    xp = batch.xp
    theta = batch.arrays["theta"]           # bottom-to-top
    top = theta[-NTOP:][::-1]
    lat_plane = batch.arrays["latitude_deg"]
    val, wsum = _band_mean(xp, top, lat_plane)
    zon = xp.mean(top, axis=-1, dtype=xp.float64)        # (NTOP, nrow)
    key = (int(lat_plane.shape[0]), float(lat_plane[0, 0]))
    if key not in ZON["lat"]:
        ZON["lat"][key] = lat_plane[:, 0].astype(xp.float64)
    if name != "start":
        _zon_add(key, name, zon - ZON["prev"][key], xp)
    ZON["prev"][key] = zon
    if name == "start":
        PHYS["wsum"] = wsum if PHYS["wsum"] is None else PHYS["wsum"] + wsum
        PHYS["calls"] += 1
    else:
        _phys_add(name, val - PHYS["prev"], xp)
    PHYS["prev"] = val


_nr.NativePhysicsRuntime._observe = _observe

_orig_rad = _nr.NativePhysicsRuntime._radiation_step


def _radiation_step(self, batch, persistent, cfg):
    due = _orig_rad(self, batch, persistent, cfg)
    xp = batch.xp
    lat = batch.arrays["latitude_deg"]
    dt = float(batch.dt_s)
    for key, label in (("rad_rthratenlw", "rad_lw"), ("rad_rthratensw", "rad_sw")):
        rate = persistent.arrays[key][-NTOP:][::-1]
        val, _ = _band_mean(xp, rate, lat)
        _phys_add(label, dt * val, xp)
        key = (int(lat.shape[0]), float(lat[0, 0]))
        _zon_add(key, label, dt * xp.mean(rate, axis=-1, dtype=xp.float64), xp)
    return due


_nr.NativePhysicsRuntime._radiation_step = _radiation_step


def _top_mean(ledger, atmosphere):
    xp = ledger.xp
    coef = atmosphere.theta[:NTOP, 0, 0]
    coef = xp.real(coef).astype(xp.float64)
    if DYN["scale"] is None:
        grid = ledger.model.grid_state(atmosphere, only=("theta",))["theta"]
        mean = _led.global_mean_device(xp, grid[:NTOP], ledger.energy.weights) \
            if hasattr(_led, "global_mean_device") else None
        if mean is None:
            from arwen_global.insitu.budgets import global_mean_device
            mean = global_mean_device(xp, grid[:NTOP], ledger.energy.weights)
        DYN["scale"] = mean / coef
        DYN["scale_grid0"] = mean
    return coef * DYN["scale"]


def _begin(self, atmosphere):
    value = _top_mean(self, atmosphere)
    DYN["prev"] = value
    DYN["series"].append(value)
    DYN["steps"] += 1
    self.energy.active = False
    return True


def _mark(self, name, atmosphere):
    value = _top_mean(self, atmosphere)
    xp = self.xp
    if name not in DYN["acc"]:
        DYN["acc"][name] = xp.zeros((NTOP,), dtype=xp.float64)
        DYN["order"].append(name)
    DYN["acc"][name] += value - DYN["prev"]
    DYN["prev"] = value
    DYN["last"] = value
    DYN["ledger"] = self
    DYN["last_atm"] = atmosphere


from arwen_global import dynamics as _dyn  # noqa: E402

SPONGE: dict = {"ke": None, "theta": None, "calls": 0, "skipped": 0}
_orig_sponge = _dyn.MoistHybridModel._top_sponge


def _top_sponge(self, u, v, p_full, p_half, dt_s):
    out_u, out_v = _orig_sponge(self, u, v, p_full, p_half, dt_s)
    xp = self.transform.backend.xp
    nlat = int(self.transform.grid.quadrature_weights.shape[0])
    if u.shape[-2] != nlat:
        SPONGE["skipped"] += 1
        return out_u, out_v
    w = xp.asarray(self.transform.grid.quadrature_weights, dtype=xp.float64)
    top = slice(0, NTOP)
    def ke(a, b):
        a = a[top].astype(xp.float64); b = b[top].astype(xp.float64)
        return 0.5 * (a * a + b * b)
    dke = ke(u, v) - ke(out_u, out_v)                 # J/kg removed, >= 0
    exner = (p_full[top].astype(xp.float64) / 1.0e5) ** (287.04 / 1004.5)
    dtheta = dke / (1004.5 * exner)
    def gm(f):
        return 0.5 * xp.sum(xp.mean(f, axis=-1) * w[None, :], axis=-1)
    a, b = gm(dke), gm(dtheta)
    SPONGE["ke"] = a if SPONGE["ke"] is None else SPONGE["ke"] + a
    SPONGE["theta"] = b if SPONGE["theta"] is None else SPONGE["theta"] + b
    SPONGE["calls"] += 1
    return out_u, out_v


_dyn.MoistHybridModel._top_sponge = _top_sponge

_led.InsituLedger.begin_energy_step = _begin
_led.InsituLedger.mark_energy = _mark


def _host(value):
    try:
        import cupy as cp
        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return np.asarray(value)


def _dump():
    out = {"schema": "arwen-global.top-budget/v1", "levels_top_first": NTOP,
           "argv": sys.argv[1:], "wall_s": time.time() - META["started"]}
    steps = DYN["steps"]
    out["steps"] = steps
    calls = PHYS["calls"]
    out["physics_calls_band_starts"] = calls
    if PHYS["wsum"] is not None and steps:
        wglobe = float(_host(PHYS["wsum"])) / (2.0 * steps)
        out["physics_weight_globe"] = wglobe
        out["physics_theta_change_k"] = {
            name: (_host(PHYS["acc"][name]) / wglobe).tolist()
            for name in PHYS["order"]}
    if ZON["acc"]:
        lats, cols = [], {}
        for key in sorted(ZON["acc"], key=lambda k: float(_host(ZON["lat"][k])[0])):
            lats.append(_host(ZON["lat"][key]))
            for name, arr in ZON["acc"][key].items():
                cols.setdefault(name, []).append(_host(arr))
        lat_all = np.concatenate(lats)
        order = np.argsort(lat_all)
        # whole-run change per Gaussian row (both Strang halves summed)
        out["zonal_lat_deg"] = lat_all[order].tolist()
        out["zonal_physics_theta_change_k"] = {
            name: np.concatenate(parts, axis=-1)[:, order].tolist()
            for name, parts in cols.items()}
    if DYN["acc"]:
        out["dynamics_theta_change_k"] = {
            name: _host(DYN["acc"][name]).tolist() for name in DYN["order"]}
    if DYN["series"]:
        series = np.stack([_host(v) for v in DYN["series"]])
        out["start_of_step_theta_k"] = series.tolist()
        if "last" in DYN:
            out["end_theta_k"] = _host(DYN["last"]).tolist()
    if SPONGE["ke"] is not None:
        out["sponge_calls"] = SPONGE["calls"]
        out["sponge_skipped_banded_calls"] = SPONGE["skipped"]
        out["sponge_ke_removed_j_kg"] = _host(SPONGE["ke"]).tolist()
        # the absorber + top drag kinetic-energy removal as theta; it is
        # BOOKED as heat by the model unless [sponge] heat_booking = false
        out["sponge_ke_removed_theta_k"] = _host(SPONGE["theta"]).tolist()
    if DYN.get("ledger") is not None:
        led = DYN["ledger"]
        xp = led.xp
        from arwen_global.insitu.budgets import global_mean_device
        grid = led.model.grid_state(DYN["last_atm"], only=("theta",))["theta"]
        out["scale_check_end_grid_mean"] = _host(
            global_mean_device(xp, grid[:NTOP], led.energy.weights)).tolist()
        vert = led.model.vertical
        a = np.asarray(vert.a_half_pa, dtype=np.float64)
        b = np.asarray(vert.b_half, dtype=np.float64)
        p = a + b * 1.0e5
        out["p_half_ref_pa_top"] = p[:NTOP + 1].tolist()
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[top-budget] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    from arwen_global.cli import main
    try:
        code = main(sys.argv[1:])
    finally:
        _dump()
    sys.exit(code)
