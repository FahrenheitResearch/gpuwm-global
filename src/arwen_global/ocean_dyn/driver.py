"""The ocean-only driver (design L2, ``gpuwm-global ocean run``): the dynamic
ocean forced by the GODAS pentads' surface fluxes, linear in time between
pentad centres, with an optional SST-restoring arm.

Resumable: a state checkpoint (``state-<step>.npz``, every array of the
state, bitwise) is written every ``checkpoint_days``; a run started again
in the same folder continues from the newest one.  Rows go to
``rows.jsonl`` once a model day.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import numpy as np

from .diagnostics import rows as ocean_rows
from .grid import OceanGrid
from .init import OceanInit, Targets, bracket, godas_forcing, pentad_center
from .model import OceanModel, OceanParams, STATE_DTYPES


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class GodasForcing:
    def __init__(self, grid, godas_dir, xp):
        self.grid = grid
        self.dir = Path(godas_dir)
        self.dates = sorted(datetime.strptime(p.name.split(".")[2], "%Y%m%d")
                            for p in self.dir.glob("godas.P.*.grb"))
        self.xp = xp
        self.cache = {}

    def _pentad(self, d):
        if d not in self.cache:
            if len(self.cache) > 3:
                self.cache.pop(sorted(self.cache)[0])
            f = godas_forcing([(self.dir / f"godas.P.{d:%Y%m%d}.grb", 1.0)], self.grid)
            self.cache[d] = {k: self.xp.asarray(v) for k, v in f.items()}
        return self.cache[d]

    def at(self, when: datetime) -> dict:
        d0, d1, w = bracket(self.dates, when)
        a, b = self._pentad(d0), self._pentad(d1)
        return {k: (1.0 - w) * a[k] + w * b[k] for k in a}


def save_state(path: Path, state: dict, meta: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: _host(v) for k, v in state.items()}, meta_json=json.dumps(meta))
    tmp.replace(path)


def load_state(path: Path, xp) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        state = {k: xp.asarray(d[k]) for k in STATE_DTYPES}
    return state, meta


def run(*, grid_path, init_path, godas_dir, start: datetime, days: float, out_dir, xp,
        params: OceanParams | None = None, checkpoint_days: float = 5.0, log=print,
        sst_restore_days: float | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = OceanGrid.load(grid_path)
    init = OceanInit(init_path)
    params = params or OceanParams()
    if sst_restore_days:
        from dataclasses import replace

        params = replace(params, sst_restoring_days=float(sst_restore_days))
    targets = Targets(init, start, xp)
    model = OceanModel(grid, params, xp, targets=targets.as_model_targets())
    forcing = GodasForcing(grid, godas_dir, xp)
    ckpts = sorted(out.glob("state-*.npz"))
    if ckpts:
        state, meta = load_state(ckpts[-1], xp)
        step0 = int(meta["step"])
        log(f"ocean run: resuming from {ckpts[-1].name} (step {step0})")
    else:
        a = init.arrays
        state = model.new_state(a["theta"], a["salt"], a["u"], a["v"], a["eta"])
        step0 = 0
        (out / "run.json").write_text(json.dumps({
            "grid_sha256": grid.sha256, "init_sha256": init.sha256, "start": start.isoformat(),
            "days": days, "params": params.identity(), "godas_dir": str(godas_dir)}, indent=1))
    dt = params.dt
    nsteps = int(round(days * 86400.0 / dt))
    per_day = int(round(86400.0 / dt))
    every = int(round(checkpoint_days * per_day))
    t_wall = time.time()
    with (out / "rows.jsonl").open("a") as rows_out:
        for n in range(step0, nsteps):
            when = start + timedelta(seconds=(n + 0.5) * dt)
            fc = forcing.at(when)
            fc["time_s"] = (n + 0.5) * dt
            # the SST-restoring arm restores toward GODAS's own surface temperature
            sst_godas = fc.pop("godas_sst_c")
            if params.sst_restoring_days:
                fc["sst_target_c"] = sst_godas
            state, diag = model.step(state, fc)
            if (n + 1) % per_day == 0:
                r = ocean_rows(grid, state)
                r["day"] = (n + 1) / per_day
                r["valid"] = (start + timedelta(seconds=(n + 1) * dt)).isoformat()
                r["wall_s_per_day"] = (time.time() - t_wall)
                t_wall = time.time()
                rows_out.write(json.dumps(r) + "\n")
                rows_out.flush()
                log(f"day {r['day']:.0f}: nino3.4 {r['ocean_nino34_sst_c'] or float('nan'):.2f} C, EUC140W "
                    f"{r['ocean_euc_140w_max_m_s']:.2f} m/s @ {r['ocean_euc_140w_depth_m']:.0f} m, "
                    f"max speed {r['ocean_max_speed_m_s']:.2f}, heat book {r['ocean_heat_book_residual_rel']:.1e}")
                if r["ocean_mean_sst_c"] is None or not np.isfinite(r["ocean_mean_sst_c"]):
                    raise FloatingPointError(f"the ocean went non-finite on day {r['day']}")
            if (n + 1) % every == 0 or n + 1 == nsteps:
                save_state(out / f"state-{n + 1:06d}.npz", state, {"step": n + 1,
                           "valid": (start + timedelta(seconds=(n + 1) * dt)).isoformat()})
                for old in sorted(out.glob("state-*.npz"))[:-2]:
                    old.unlink()
    return {"steps": nsteps, "out": str(out)}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="gpuwm-global ocean run")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--godas-dir", required=True)
    ap.add_argument("--start", required=True, help="YYYYMMDDHH")
    ap.add_argument("--days", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xp", default="cupy")
    ap.add_argument("--sst-restore-days", type=float, default=None)
    ap.add_argument("--checkpoint-days", type=float, default=5.0)
    args = ap.parse_args(argv)
    if args.xp == "cupy":
        import cupy as xp
    else:
        xp = np
    start = datetime.strptime(args.start, "%Y%m%d%H")
    run(grid_path=args.grid, init_path=args.init, godas_dir=args.godas_dir, start=start,
        days=args.days, out_dir=args.out, xp=xp, checkpoint_days=args.checkpoint_days,
        sst_restore_days=args.sst_restore_days, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
