"""Device cost of one stochastic physics call at a given truncation.

    python tools/arwen_global_stochastic_cost.py --truncation 383 --nlev 40

Times, on the card, everything the stochastic physics adds to one physics
call of a one-band run: the pattern advance and the SPPT synthesis
(``begin``), the four input copies and the SPPT/SKEB band work
(``perturb_band``), and the SKEB spectral forcing (``finish``).  The
physics suite itself is not run: the "result" is the input plus a fixed
random increment, which costs the stochastic code exactly what a real
result does.  A step makes two such calls (Strang halves).
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncation", type=int, default=383)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    import cupy as cp

    from arwen_global.bands import BandPipeline
    from arwen_global.spectral.transform import SphericalHarmonicTransform
    from arwen_global.state import PhysicsState
    from arwen_global.stochastic import StochasticConfig, StochasticPhysics

    transform = SphericalHarmonicTransform.create(
        args.truncation, backend="cupy", precision="float32")
    nlat, nlon = transform.grid.shape
    nlev = args.nlev
    rng = cp.random.default_rng(1)
    shape = (nlev, nlat, nlon)
    p_full = cp.broadcast_to(cp.linspace(1000.0, 100000.0, nlev, dtype=cp.float32)[:, None, None],
                             shape).copy()
    exner = (p_full / 1.0e5) ** 0.2857
    geopotential = cp.broadcast_to(
        cp.linspace(3.0e5, 100.0, nlev, dtype=cp.float32)[:, None, None], shape).copy()
    exchange = SimpleNamespace(
        u=rng.standard_normal(shape, dtype=cp.float32) * 10,
        v=rng.standard_normal(shape, dtype=cp.float32) * 10,
        theta=300 + rng.standard_normal(shape, dtype=cp.float32),
        qv=cp.full(shape, 5.0e-3, dtype=cp.float32),
        exner=exner, p_full=p_full, geopotential=geopotential)
    increment = {name: rng.standard_normal(shape, dtype=cp.float32) * scale
                 for name, scale in (("u", 1e-2), ("v", 1e-2), ("theta", 1e-2), ("qv", 1e-6))}
    model = SimpleNamespace(
        nlev=nlev, surface_geopotential=cp.zeros((nlat, nlon), dtype=cp.float32),
        pipeline=BandPipeline(nlat, 1))
    sp = StochasticPhysics(StochasticConfig(sppt=True, skeb=True), transform)
    zeta = transform.zeros(nlev)
    theta_spec = transform.zeros(nlev)
    bundle = SimpleNamespace(time_s=0.0, physics_state=PhysicsState())
    rows = slice(0, nlat)
    parts = {"begin": [], "perturb_band": [], "finish": [], "call": []}
    for k in range(args.warmup + args.calls):
        bundle.time_s = 300.0 * (k // 2)  # two calls per step share a clock
        events = [cp.cuda.Event() for _ in range(4)]
        events[0].record()
        ctx = sp.begin(model, bundle, 150.0, 300.0)
        events[1].record()
        inputs = {name: cp.array(getattr(exchange, name), copy=True)
                  for name in ("u", "v", "theta", "qv")}
        result = SimpleNamespace(**{name: getattr(exchange, name) + increment[name]
                                    for name in ("u", "v", "theta", "qv")})
        sp.perturb_band(ctx, model, rows, inputs, result, exchange, dt_s=150.0)
        events[2].record()
        z, t, _record = sp.finish(ctx, zeta, theta_spec, dt_s=150.0)
        events[3].record()
        events[3].synchronize()
        bundle.physics_state = PhysicsState(arrays=dict(ctx.state))
        if k >= args.warmup:
            parts["begin"].append(cp.cuda.get_elapsed_time(events[0], events[1]))
            parts["perturb_band"].append(cp.cuda.get_elapsed_time(events[1], events[2]))
            parts["finish"].append(cp.cuda.get_elapsed_time(events[2], events[3]))
            parts["call"].append(cp.cuda.get_elapsed_time(events[0], events[3]))
        del result, inputs, z, t
    # The "result" arithmetic (input + increment) is the harness's, not the
    # stochastic code's: timed alone and taken off perturb_band.
    harness = []
    for _ in range(args.calls):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        _r = [getattr(exchange, n) + increment[n] for n in ("u", "v", "theta", "qv")]
        b.record()
        b.synchronize()
        harness.append(cp.cuda.get_elapsed_time(a, b))
        del _r
    # The fused kernel against its specification (the array expressions),
    # on one call, and the specification's own time.
    check, unfused = {}, []
    for fused in (True, False):
        ctx = sp.begin(model, bundle, 150.0, 300.0)
        inputs = {name: cp.array(getattr(exchange, name), copy=True)
                  for name in ("u", "v", "theta", "qv")}
        result = SimpleNamespace(**{name: getattr(exchange, name) + increment[name]
                                    for name in ("u", "v", "theta", "qv")})
        sp.perturb_band(ctx, model, rows, inputs, result, exchange, dt_s=150.0, fused=fused)
        check[fused] = (result, ctx.sink_rows.complete().copy(), ctx.taper_rows.complete().copy(),
                        int(cp.sum(cp.stack(ctx.reverted))))
    for _ in range(10):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        ctx = sp.begin(model, bundle, 150.0, 300.0)
        inputs = {name: cp.array(getattr(exchange, name), copy=True)
                  for name in ("u", "v", "theta", "qv")}
        result = SimpleNamespace(**{name: getattr(exchange, name) + increment[name]
                                    for name in ("u", "v", "theta", "qv")})
        a.record()
        sp.perturb_band(ctx, model, rows, inputs, result, exchange, dt_s=150.0, fused=False)
        b.record()
        b.synchronize()
        unfused.append(cp.cuda.get_elapsed_time(a, b))
    fused_r, spec_r = check[True][0], check[False][0]
    diff = {name: float(cp.max(cp.abs(getattr(fused_r, name) - getattr(spec_r, name))))
            for name in ("u", "v", "theta", "qv")}
    diff["sink_rows_rel"] = float(cp.max(cp.abs(check[True][1] - check[False][1]))
                                  / cp.max(cp.abs(check[False][1])))
    diff["taper_rows"] = float(cp.max(cp.abs(check[True][2] - check[False][2])))
    diff["reverted_fused_vs_spec"] = [check[True][3], check[False][3]]
    out = {name: float(np.median(v)) for name, v in parts.items()}
    out["fused_vs_specification_max_abs"] = diff
    out["unfused_perturb_band_ms"] = float(np.median(unfused))
    out["harness_result_ms"] = float(np.median(harness))
    out["stochastic_per_call_ms"] = out["call"] - out["harness_result_ms"]
    out["stochastic_per_step_ms"] = 2 * out["stochastic_per_call_ms"]
    out.update(truncation=args.truncation, nlev=nlev, grid=[nlat, nlon],
               device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               peak_pool_mib=cp.get_default_memory_pool().total_bytes() / 2**20,
               note="median over calls, CUDA events, host waits included; one band")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
