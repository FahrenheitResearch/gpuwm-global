"""Carry extra transported fields through a finished run's checkpoints.

The driver for :func:`arwen_global.composition.replay`.  It reads the
run's weather checkpoints (only their wind and surface pressure), carries
the rows a composition spec names through that flow, and writes one
sidecar per checkpoint plus a receipt into ``--out``.  The weather
checkpoints are opened read only; the physics is not run.

usage:
  python tools/arwen_global_replay_tracers.py --config RUN.toml \\
      --spec SPEC.toml --out DIR CHECKPOINT.npz [CHECKPOINT.npz ...]
      [--resume SIDECAR] [--accept-geometry-config] [--sha256-before-after]

``--config`` is the run's own config.  A config that carries the run's
grid and vertical coordinate under another hash (a run written by a tree
whose config tables this one does not read, for one) is refused unless
``--accept-geometry-config`` is given, and the receipt then records both
hashes.  ``--sha256-before-after`` hashes every weather checkpoint file
before and after the replay and writes both lists into the receipt, so the
claim that the weather did not move is a measurement in the same file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    from arwen_global.composition import RECEIPT_NAME, load_spec, replay
    from arwen_global.config import load_config

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--accept-geometry-config", action="store_true")
    parser.add_argument("--sha256-before-after", action="store_true")
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    spec = load_spec(args.spec)
    paths = [Path(item) for item in args.checkpoints]
    before = ({path.name: _file_sha256(path) for path in paths}
              if args.sha256_before_after else None)

    def progress(record: dict) -> None:
        worst = max(record["fixer_relative_max"].values(), default=0.0)
        print(f"{record['to_checkpoint']}: {record['substeps']} substeps, "
              f"displacement <= {record['displacement_max_cells']:.2f} cells, "
              f"fixer <= {worst:.3e}", flush=True)

    receipt = replay(
        cfg, paths, spec, args.out, resume=args.resume,
        accept_geometry_config=args.accept_geometry_config,
        progress=progress,
    )
    if before is not None:
        after = {path.name: _file_sha256(path) for path in paths}
        receipt["weather_file_sha256"] = {
            "before": before, "after": after,
            "identical": before == after,
        }
        target = Path(args.out) / RECEIPT_NAME
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True)
                          + "\n", encoding="utf-8")
        if before != after:
            print("weather checkpoint files CHANGED during the replay",
                  file=sys.stderr)
            return 1
    drift = receipt["ledger"]["relative_mass_drift"]
    print(json.dumps({
        "sidecars": len(receipt["sidecars"]),
        "wall_seconds": round(receipt["wall_seconds"], 3),
        "relative_mass_drift": drift,
        "emitted_kg": receipt["ledger"]["final"]["emitted_kg"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
