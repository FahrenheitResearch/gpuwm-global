"""Cut a forcing record from a finished run's checkpoints.

``gpuwm-global run --forcing RECORD`` pulls a run toward a record of the
low degrees of the spectral fields (arwen_global.forcing).  A run can write
one of itself as it goes (``--write-forcing-record``); this cuts the same
thing afterwards from checkpoints already on disk, so a replay toward an old
run does not need that run again.  Each checkpoint is verified by its own
digest and ordered by its model time; the record names the checkpoints it
came from and their config hash.  The record's times are the checkpoints'
times, so the pull interpolates between output times: a run checkpointed
daily gives a daily record.

usage:
  python tools/arwen_global_forcing_record.py RUN/arwen_global_step*.npz \
      --out RECORD.npz [--degree 42] [--fields vorticity,divergence,theta] \
      [--dtype complex64]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    from arwen_global.constants import SPECTRAL_FIELDS
    from arwen_global.forcing import record_from_checkpoints

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=42)
    parser.add_argument("--fields", default=",".join(SPECTRAL_FIELDS))
    parser.add_argument("--dtype", choices=("complex128", "complex64"), default="complex128")
    args = parser.parse_args(argv)
    record = record_from_checkpoints(
        args.checkpoints, degree=args.degree,
        fields=tuple(filter(None, args.fields.split(","))), dtype=args.dtype)
    path = record.write(args.out)
    print(json.dumps({"path": str(path), **record.identity()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
