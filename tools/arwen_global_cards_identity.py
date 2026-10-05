"""Launch one WOOF Global configuration on 1 and on P cards and compare.

The multi-card instrument for gates BIT-5 (P cards return one card's
checkpoint bit for bit) and MG-4 (each rank runs on its own card), with
the transport each run took and the rate it achieved.  Orchestration
only: every rank is the shipped ``python -m arwen_global run`` door, each
in its own process with its own output directory, rendezvousing over
loopback in rank order, so the placement the door does (the rank's local
rank picks its card) is exactly what is exercised.

For every card count it reports, per rank: the card the receipt says the
rank ran on (name, PCI bus, UUID), the transport that ran and why, the
wire ledger's achieved rate and bytes per step, the step wall, and
whether the rank's final checkpoint is the one-card checkpoint array for
array and by ``self_sha256``, with any gate the rank's receipt failed.
Exit status 0 when every rank of every card count finished, its
checkpoint matches, and no two ranks of a run share a card UUID.

usage:
  python tools/arwen_global_cards_identity.py --config C.toml --cards 1,2,4,8 \
      --until-s 3600 --out OUT [--latitude-bands 16] [--transport auto] \
      [--base-port 29610] [--python python3] [--json OUT/summary.json]

On a CPU (``backend = "numpy"``) it is the loopback proof of the exchange
code; on a CUDA box it is the placement, bandwidth and identity proof.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arwen_global_checkpoint_identity import compare  # noqa: E402


def _launch(args, cards: int, outroot: Path) -> list[subprocess.Popen]:
    ports = [args.base_port + 10 * cards + r for r in range(cards)]
    addresses = ",".join(f"127.0.0.1:{p}" for p in ports)
    procs = []
    for rank in range(cards):
        outdir = outroot / f"cards{cards}" / f"rank{rank}"
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [args.python, "-m", "arwen_global", "run",
               str(args.config), "--outdir", str(outdir),
               "--overwrite", "--until-s", str(args.until_s)]
        if args.latitude_bands:
            cmd += ["--latitude-bands", str(args.latitude_bands)]
        if cards > 1:
            cmd += ["--cards", str(cards), "--card-rank", str(rank),
                    "--card-addresses", addresses,
                    "--card-transport", args.transport]
        env = dict(os.environ)
        log = open(outdir / "run.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
    return procs


def _final_checkpoint(outdir: Path) -> Path | None:
    found = sorted(outdir.glob("arwen_global_step*.npz"))
    return found[-1] if found else None


def _receipt(outdir: Path) -> dict:
    for path in sorted(outdir.glob("arwen-global-receipt.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _cards_block(receipt: dict) -> dict:
    block = receipt.get("cards")
    return block if isinstance(block, dict) else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cards", default="1,2")
    ap.add_argument("--until-s", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--latitude-bands", type=int, default=0)
    ap.add_argument("--transport", default="auto", choices=("auto", "tcp", "nccl"))
    ap.add_argument("--base-port", type=int, default=29610)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    counts = [int(c) for c in str(args.cards).split(",") if c.strip()]
    if counts[0] != 1:
        counts = [1] + counts
    summary: dict = {"config": str(args.config), "until_s": args.until_s,
                     "runs": {}, "ok": True}
    reference = None
    for cards in counts:
        started = time.perf_counter()
        procs = _launch(args, cards, args.out)
        codes = []
        for proc in procs:
            try:
                codes.append(proc.wait(timeout=args.timeout_s))
            except subprocess.TimeoutExpired:
                proc.kill()
                codes.append("timeout")
        wall = time.perf_counter() - started
        ranks = []
        uuids = []
        for rank in range(cards):
            outdir = args.out / f"cards{cards}" / f"rank{rank}"
            final = _final_checkpoint(outdir)
            receipt = _receipt(outdir)
            block = _cards_block(receipt)
            wire = block.get("wire") or {}
            devices = block.get("card_devices") or []
            mine = devices[rank] if rank < len(devices) else None
            if mine and mine.get("uuid"):
                uuids.append(mine["uuid"])
            row = {
                "rank": rank, "exit": codes[rank],
                "checkpoint": None if final is None else str(final),
                "device": mine,
                "placement": block.get("card_placement"),
                "transport": block.get("transport"),
                "transport_reason": block.get("transport_reason"),
                "achieved_gb_s": wire.get("achieved_gb_s"),
                "bytes_per_step": wire.get("bytes_per_step"),
                "exposed_ms_per_step": wire.get("exposed_ms_per_step"),
                "step_s": (receipt.get("step_wall") or {}).get("step_s"),
                "status": receipt.get("status"),
                "failed_gates": sorted(
                    name for name, gate in (receipt.get("gates") or {}).items()
                    if isinstance(gate, dict) and gate.get("passed") is False),
            }
            if cards == 1:
                reference = final
            if final is not None and reference is not None and cards > 1:
                result = compare(reference, final)
                row["identical_to_one_card"] = result.get("verdict") == "IDENTICAL"
                row["arrays_identical"] = (
                    not result.get("differing") and not result.get("inventory_mismatch"))
                row["arrays_compared"] = result.get("arrays_compared")
                row["self_sha256"] = (result.get("b") or {}).get("self_sha256")
                row["differing_arrays"] = [d["array"] for d in result.get("differing", [])][:8]
            elif cards > 1:
                row["identical_to_one_card"] = False
            ranks.append(row)
        distinct = len(set(uuids)) == len(uuids)
        # The verdict is the identity and the placement.  A rank that ran
        # to the end and failed only a gate (WIRE-1's 2 GB/s floor on a
        # loopback or LAN link is the usual one) still wrote the checkpoint
        # being compared, so its gates are reported, not folded in; a rank
        # with no receipt or no final checkpoint fails the run.
        finished = all(r["status"] in ("pass", "fail") and r["checkpoint"]
                       for r in ranks)
        ok = finished and distinct and all(
            r.get("identical_to_one_card", True) for r in ranks)
        summary["runs"][str(cards)] = {
            "wall_s": round(wall, 2), "ok": ok,
            "distinct_cards": distinct if uuids else None, "ranks": ranks}
        summary["ok"] = summary["ok"] and ok
        print(f"cards={cards} wall={wall:.1f}s ok={ok} exits={codes} "
              f"transport={[r['transport'] for r in ranks]} "
              f"gb_s={[r['achieved_gb_s'] for r in ranks]} "
              f"identical={[r.get('identical_to_one_card') for r in ranks]} "
              f"failed_gates={sorted({g for r in ranks for g in r['failed_gates']})} "
              f"devices={[(r['device'] or {}).get('pci_bus_id') for r in ranks]}",
              flush=True)
    text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
