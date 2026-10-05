"""One rank of a semi-Lagrangian band-identity run, hashed in memory.

Builds a config's model and cold state, takes ``steps`` steps at
``bands`` latitude bands (one card, or one rank of ``world`` cards over
TCP), and prints one JSON line: the sha256 of every prognostic,
surface, physics and trajectory array (the trajectory assembled across
the cards exactly as the checkpoint assembles it), every scalar metric of
every step, the steps' own wall seconds and the device live peak; with
SEMILAG_BAND_PROFILE set, the mean per-step device and host milliseconds
of the step's sections as well.  Two
runs that print the same ``arrays`` and ``metrics`` returned the same
bits.  No checkpoint is written, so a node with little disk can run it.

Usage:
  python tools/semilag_band_identity_run.py CONFIG STEPS BANDS \\
      [WORLD RANK ADDR0,ADDR1,...]
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time

import numpy as np


def _digest(array) -> str:
    host = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(
        host.dtype.str.encode() + str(host.shape).encode() + host.tobytes()
    ).hexdigest()[:20]


def _scalars(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, str)):
            out[prefix + key] = value
        elif isinstance(value, (int, float)):
            out[prefix + key] = float(value)
        elif isinstance(value, dict):
            out.update(_scalars(value, prefix + key + "."))
    return out


def main(argv) -> int:
    from arwen_global import cards
    from arwen_global.config import load_config
    from arwen_global.device_memory import start_device_peak_tracking
    from arwen_global.runner import build_model_and_cold_state, build_transform
    from arwen_global.spill import resident

    config, steps, bands = argv[1], int(argv[2]), int(argv[3])
    world = int(argv[4]) if len(argv) > 4 else 1
    rank = int(argv[5]) if len(argv) > 5 else 0
    addresses = tuple(argv[6].split(",")) if len(argv) > 6 else ()
    cfg = load_config(config)
    cfg = dataclasses.replace(cfg, latitude_bands=bands)
    tracker = start_device_peak_tracking(cfg.backend)
    transform = build_transform(cfg)
    session = None
    if world > 1:
        transport = cards.TcpCards(rank, world, addresses)
        session = cards.CardSession(transport, transform.grid.nlat, bands,
                                    weights=tuple([1.0] * world))
    model, bundle = build_model_and_cold_state(
        cfg, transform, card_session=session)
    xp = transform.backend.xp
    to_numpy = transform.backend.to_numpy
    profiler = None
    if os.environ.get("SEMILAG_BAND_PROFILE") and steps > 2:
        # The step's own sections (profile.StepProfiler): which share of a
        # step is the band work a second card divides and which is the
        # replicated spectral work it does not.  Same bits either way.
        from arwen_global.profile import StepProfiler, attach_profiler

        profiler = StepProfiler(transform.backend, steps=steps - 2, warmup=2)
        attach_profiler(model, profiler)
    trace = []
    walls = []
    for _ in range(steps):
        xp.cuda.Device().synchronize() if hasattr(xp, "cuda") else None
        if profiler is not None:
            profiler.begin_step(bundle.step + 1)
        started = time.perf_counter()
        bundle, metrics = model.step(bundle, cfg.dt_s)
        if profiler is not None:
            profiler.end_step()
        xp.cuda.Device().synchronize() if hasattr(xp, "cuda") else None
        walls.append(time.perf_counter() - started)
        trace.append(_scalars(metrics))
    arrays = {}
    atmosphere = bundle.atmosphere
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure",
                 "qv"):
        arrays["atmosphere." + name] = _digest(to_numpy(getattr(atmosphere, name)))
    for name, value in atmosphere.grid_tracers().items():
        arrays["tracer." + name] = _digest(to_numpy(resident(xp, value)))
    for name, value in bundle.surface.arrays().items():
        arrays["surface." + name] = _digest(to_numpy(resident(xp, value)))
    for name, value in sorted(bundle.physics_state.arrays.items()):
        arrays["physics." + name] = _digest(to_numpy(resident(xp, value)))
    trajectory = model.trajectory_state()
    if trajectory is not None:
        host = {name: np.array(to_numpy(value), copy=True)
                for name, value in trajectory.arrays().items()}
        if session is not None:
            host = cards.gather_named_arrays(
                session, host, transform.grid.nlat, transform.grid.nlon)
        for name, value in host.items():
            arrays["trajectory." + name] = _digest(value)
    report = {
        "config": config, "steps": steps, "bands": bands, "world": world,
        "rank": rank,
        "local_rows": list(model.pipeline.local_rows()),
        "halo_rows": list(getattr(model, "semilag_halo_rows", (None, None))),
        "arrays": arrays,
        "metrics": trace,
        "step_wall_s": walls,
        "device_peak_used_bytes": (
            None if tracker is None else int(tracker.peak_used_bytes)),
        "profile": (None if profiler is None else [
            row for row in profiler.summary() if row["depth"] <= 2]),
    }
    if session is not None:
        session.transport.close()
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
