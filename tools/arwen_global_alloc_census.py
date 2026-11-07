"""Who holds the device bytes at a run's peak?  (the allocation table)

``python tools/arwen_global_alloc_census.py --json census.json -- run CONFIG [run args...]``

Runs the real ``gpuwm-global`` command in-process with a CuPy memory hook
that records every live pool allocation by the first two ``arwen_global``
frames that requested it.  Each time the pool's live bytes reach a new
maximum the live set is snapshotted, grouped by call site, so the table
written at the end is what was live AT the peak, not a sum of everything
ever allocated.  The hook adds host work per allocation (a frame walk); the
run's own timings are therefore not timings of record (a profiler runs on
a separate run).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time

GIB = 2 ** 30


def _site(depth: int = 3) -> str:
    frame = sys._getframe(3)
    parts = []
    while frame is not None and len(parts) < depth:
        name = frame.f_code.co_filename
        if "arwen_global" in name and "alloc_census" not in name:
            short = name.split("arwen_global/", 1)[-1]
            parts.append(f"{short}:{frame.f_lineno}:{frame.f_code.co_name}")
        frame = frame.f_back
    return " < ".join(parts) if parts else "<outside arwen_global>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--min-mib", type=float, default=1.0,
                        help="snapshot only allocations above this size")
    parser.add_argument("--admit", action="store_true",
                        help="MEASUREMENT ONLY: let the door start a plan its pricing "
                             "refuses, to measure the plan the pricing has no row for")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    import cupy as cp
    from cupy.cuda import memory_hook

    live: dict[int, tuple[int, str]] = {}
    state = {"used": 0, "peak": 0, "snap": {}, "snap_t": 0.0,
             "snap_count": 0, "mallocs": 0, "t0": time.time(),
             "phase_peaks": []}
    min_bytes = int(args.min_mib * 2 ** 20)

    class Census(memory_hook.MemoryHook):
        name = "AllocCensus"

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
            site = _site(args.depth) if mem_size >= min_bytes else "<small>"
            live[mem_ptr] = (mem_size, site)
            state["used"] += mem_size
            state["mallocs"] += 1
            if state["used"] > state["peak"] * 1.002 + 2 ** 20:
                state["peak"] = state["used"]
                agg = collections.defaultdict(lambda: [0, 0])
                for sz, s in live.values():
                    agg[s][0] += sz
                    agg[s][1] += 1
                state["snap"] = dict(agg)
                state["snap_t"] = time.time() - state["t0"]
                state["snap_count"] += 1
                pool = cp.get_default_memory_pool()
                state["pool_total_at_snap"] = int(pool.total_bytes())

        def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
            rec = live.pop(mem_ptr, None)
            if rec is not None:
                state["used"] -= rec[0]

    if args.admit:
        from arwen_global import sizing

        gate = sizing.run_memory_gate

        def admitted(*a, **k):
            out = gate(*a, **k)
            if out.get("refuse"):
                print("census: --admit: the door would refuse: " + str(out.get("verdict"))[:400])
            out["refuse"] = False
            out["unfitted"] = None
            return out

        sizing.run_memory_gate = admitted
    from arwen_global.cli import main as run_main
    rc = 1
    try:
        with Census():
            rc = run_main(rest)
    finally:
        rows = sorted(state["snap"].items(), key=lambda kv: -kv[1][0])
        out = {
            "argv": rest, "rc": rc,
            "peak_live_bytes": state["peak"],
            "peak_live_gib": round(state["peak"] / GIB, 3),
            "pool_total_at_peak_gib": round(state.get("pool_total_at_snap", 0) / GIB, 3),
            "peak_at_s": round(state["snap_t"], 1),
            "mallocs": state["mallocs"],
            "table": [{"site": s, "gib": round(b / GIB, 4), "count": c}
                      for s, (b, c) in rows],
        }
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"census: peak live {out['peak_live_gib']} GiB "
              f"(pool held {out['pool_total_at_peak_gib']}) at {out['peak_at_s']} s")
        for row in out["table"][:40]:
            print(f"  {row['gib']:8.3f} GiB x{row['count']:<5d} {row['site']}")
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
