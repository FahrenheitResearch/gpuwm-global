"""One rank of the WOOF Global multi-card wire probe: placement, rate, bytes.

Launch one process per rank (the box script does), each with the same
``--world`` and ``--addresses`` and its own ``--rank``.  Each rank:

1. places itself on its own card exactly as the run door does
   (:func:`arwen_global.cards.place_rank`) BEFORE CuPy is imported;
2. opens the transport the run would open (``--transport auto`` takes NCCL
   when every rank can, the TCP mesh otherwise) and all-gathers the card
   identities, so two ranks on one card are refused by UUID;
3. gathers a latitude-row buffer of ``--mb`` megabytes the way the Fourier
   waist is gathered, exchanges a deep halo with its two neighbours, and
   all-gathers a partial sum added in rank order, and checks every byte
   against the buffer one card would hold;
4. times ``--repeats`` row gathers and halo exchanges with the stream
   synchronised, and writes the per-rank rate and the transport's own
   receipt to ``--json``.

Exit status 0 when every check passes on this rank.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--addresses", required=True)
    ap.add_argument("--transport", default="auto", choices=("auto", "tcp", "nccl"))
    ap.add_argument("--mb", type=float, default=256.0,
                    help="size of the whole latitude-row buffer, MiB")
    ap.add_argument("--nlat", type=int, default=768)
    ap.add_argument("--halo", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)

    from arwen_global import cards

    addresses = [a.strip() for a in args.addresses.split(",") if a.strip()]
    placement = cards.place_rank(args.rank, args.world, addresses, "cupy")
    import cupy as cp

    transport = cards.open_transport(
        args.rank, args.world, addresses, prefer=args.transport, device=True)
    devices = cards.check_card_placement(transport, cards.card_identity(cp))
    session = cards.CardSession(
        transport, args.nlat, 4 * args.world, weights=(1.0,) * args.world)
    cols = max(1, int(args.mb * 2**20 / 8 / args.nlat))
    whole = (cp.arange(args.nlat * cols, dtype=cp.float64).reshape(args.nlat, cols)
             * 1.000000119 + 0.25)
    first, last = session.local_rows()
    checks = {}

    rows = cp.full_like(whole, cp.nan)
    rows[first:last] = whole[first:last]
    session.gather_rows(cp, rows, 0, name="rows")
    checks["row_gather_is_the_one_card_buffer"] = bool(
        cp.array_equal(rows.view(cp.uint64), whole.view(cp.uint64)))

    halo = cp.full_like(whole, cp.nan)
    halo[first:last] = whole[first:last]
    session.exchange_halo(cp, halo, 0, args.halo, name="halo")
    lo, hi = max(0, first - args.halo), min(args.nlat, last + args.halo)
    checks["halo_fills_exactly_the_window"] = bool(
        cp.array_equal(halo[lo:hi], whole[lo:hi])
        and bool(cp.isnan(halo[:lo]).all()) and bool(cp.isnan(halo[hi:]).all()))

    parts_host = [cp.full((4096,), 0.1 * (r + 1) + 1e-17 * r, dtype=cp.float64)
                  for r in range(args.world)]
    total = session.gather_partials(cp, parts_host[args.rank], name="partial")
    want = cards.sum_in_rank_order(parts_host)
    checks["partial_sum_is_the_rank_order_sum"] = bool(
        cp.array_equal(total.view(cp.uint64), want.view(cp.uint64)))

    stream = cp.cuda.get_current_stream()
    received_rows = (args.nlat - (last - first)) * cols * 8
    stream.synchronize()
    started = time.perf_counter()
    for _ in range(args.repeats):
        session.gather_rows(cp, rows, 0, name="rows")
    stream.synchronize()
    gather_s = (time.perf_counter() - started) / args.repeats
    halo_bytes = ((first > 0) + (last < args.nlat)) * args.halo * cols * 8
    started = time.perf_counter()
    for _ in range(args.repeats):
        session.exchange_halo(cp, halo, 0, args.halo, name="halo")
    stream.synchronize()
    halo_s = (time.perf_counter() - started) / args.repeats

    receipt = session.receipt()
    out = {
        "rank": args.rank, "world": args.world,
        "transport": transport.name,
        "transport_reason": getattr(transport, "reason", ""),
        "placement": placement, "devices": devices,
        "buffer_mib": round(args.nlat * cols * 8 / 2**20, 2),
        "rows_owned": [first, last],
        "row_gather_s": gather_s,
        "row_gather_received_gb_s": received_rows / gather_s / 1e9,
        "halo_s": halo_s,
        "halo_received_gb_s": (halo_bytes / halo_s / 1e9) if halo_bytes else None,
        "checks": checks,
        "wire": receipt.get("wire"),
    }
    session.close()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2, sort_keys=True, default=str),
                         encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "rank", "transport", "row_gather_received_gb_s", "halo_received_gb_s",
        "checks")}, default=str), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
