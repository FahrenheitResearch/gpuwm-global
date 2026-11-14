"""``gpuwm-global ocean ...``: the dynamic ocean's files and its ocean-only run.

* ``build-grid --etopo-bed BED [--etopo-surface SURF] --out ocean-grid-v1.npz``
* ``build-init --grid G --valid YYYYMMDDHH --godas-dir D --woa-dir W --out F``
* ``init-from-state --grid G --base-init F --state S --out F2``: an initial
  file from an ocean-only run's state (a spun-up start), the targets kept
* ``run --grid G --init F --godas-dir D --start YYYYMMDDHH --days N --out DIR``
  (the ocean-only driver, forced by GODAS pentad fluxes)

All of them run on a Linux host; ``run`` on a card (``--xp cupy``) or the CPU.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

HELP = "the dynamic ocean: build its grid and initial state, run it ocean-only"


def add_arguments(parser) -> None:
    legs = parser.add_subparsers(dest="ocean_command", required=True)
    grid = legs.add_parser("build-grid", help="the stage-1 ocean grid from ETOPO 2022 60 s")
    grid.add_argument("--etopo-bed", type=Path, required=True)
    grid.add_argument("--etopo-surface", type=Path, default=None)
    grid.add_argument("--out", type=Path, required=True)
    init = legs.add_parser("build-init", help="an initial state from GODAS pentads and WOA23")
    init.add_argument("--grid", type=Path, required=True)
    init.add_argument("--valid", required=True, help="YYYYMMDDHH")
    init.add_argument("--godas-dir", type=Path, required=True)
    init.add_argument("--woa-dir", type=Path, required=True)
    init.add_argument("--barotropic", choices=("remove", "godas"), default="remove")
    init.add_argument("--out", type=Path, required=True)
    fs = legs.add_parser("init-from-state", help="an initial file from an ocean-only run's state")
    fs.add_argument("--grid", type=Path, required=True)
    fs.add_argument("--base-init", type=Path, required=True)
    fs.add_argument("--state", type=Path, required=True)
    fs.add_argument("--out", type=Path, required=True)
    run = legs.add_parser("run", help="the ocean-only run forced by GODAS pentad fluxes")
    run.add_argument("--grid", type=Path, required=True)
    run.add_argument("--init", type=Path, required=True)
    run.add_argument("--godas-dir", type=Path, required=True)
    run.add_argument("--start", required=True, help="YYYYMMDDHH")
    run.add_argument("--days", type=float, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--xp", choices=("cupy", "numpy"), default="cupy")
    run.add_argument("--checkpoint-days", type=float, default=5.0)
    run.add_argument("--sst-restore-days", type=float, default=None,
                     help="restore the top cell toward GODAS's own SST (the restored arm)")


def main(args) -> int:
    from .grid import OceanGrid, build_grid

    if args.ocean_command == "build-grid":
        grid, report = build_grid(args.etopo_bed, args.etopo_surface)
        report["sha256"] = grid.write(args.out)
        print(json.dumps(report, indent=1, default=str))
        return 0
    if args.ocean_command == "build-init":
        from .init import build_init

        grid = OceanGrid.load(args.grid)
        out = build_init(grid, valid=datetime.strptime(args.valid, "%Y%m%d%H"),
                         godas_dir=args.godas_dir, woa_dir=args.woa_dir, out_path=args.out,
                         barotropic=args.barotropic)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if args.ocean_command == "init-from-state":
        from .init import init_from_state

        out = init_from_state(OceanGrid.load(args.grid), base_init=args.base_init,
                              state_path=args.state, out_path=args.out)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if args.ocean_command == "run":
        from .driver import main as run_main

        return run_main(["--grid", str(args.grid), "--init", str(args.init), "--godas-dir",
                         str(args.godas_dir), "--start", args.start, "--days", str(args.days),
                         "--out", str(args.out), "--xp", args.xp, "--checkpoint-days",
                         str(args.checkpoint_days)]
                        + ([] if args.sst_restore_days is None
                           else ["--sst-restore-days", str(args.sst_restore_days)]))
    raise ValueError(args.ocean_command)


__all__ = ["HELP", "add_arguments", "main"]
