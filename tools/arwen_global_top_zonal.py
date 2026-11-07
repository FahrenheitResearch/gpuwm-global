#!/usr/bin/env python3
"""Zonal-mean top-level theta and its change between checkpoints (host).

    python tools/arwen_global_top_zonal.py RUN_DIR [--levels 4] [--out JSON]

Synthesises the top ``--levels`` theta levels of every checkpoint in
RUN_DIR with the package's own transform on the numpy backend (float64)
and writes, per checkpoint, the global mean and the zonal mean in 10-degree
bands, plus the gridded top-level theta change first -> last as .npy for
the map.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from arwen_global.spectral.transform import SphericalHarmonicTransform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    files = sorted(run.glob("arwen_global_step*.npz"),
                   key=lambda p: int(re.findall(r"(\d+)", p.name)[-1]))
    first = np.load(files[0])
    theta0 = first["atmosphere__theta"][: args.levels]
    t = theta0.shape[-1] - 1
    nlat, nlon = first["atmosphere__qc"].shape[-2:]
    tr = SphericalHarmonicTransform.create(t, nlat=nlat, nlon=nlon,
                                           backend="numpy", precision="float64")
    lat = np.degrees(np.arcsin(np.asarray(tr.grid.sin_lat, np.float64)))
    w = np.asarray(tr.grid.quadrature_weights, np.float64)
    edges = np.arange(-90, 91, 10)
    rows = []
    grids = []
    for f in files:
        z = np.load(f)
        step = int(re.findall(r"(\d+)", f.name)[-1])
        g = np.asarray(tr.inverse(z["atmosphere__theta"][: args.levels]
                                  .astype(np.complex128)), np.float64)
        zonal = g.mean(axis=-1)                       # (lev, nlat)
        gm = 0.5 * np.sum(zonal * w[None, :], axis=-1)
        bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (lat >= lo) & (lat < hi)
            bands.append((np.sum(zonal[:, m] * w[m], axis=-1) / np.sum(w[m])).tolist())
        rows.append({"step": step, "global_mean": gm.tolist(), "bands": bands})
        grids.append(g[0])
    out = {"run": str(run), "levels": args.levels,
           "band_edges_deg": edges.tolist(), "rows": rows}
    np.save(run / "top_theta_change_lev0.npy", (grids[-1] - grids[0]).astype(np.float32))
    np.save(run / "top_theta_lat.npy", lat.astype(np.float32))
    path = Path(args.out) if args.out else run / "top-zonal.json"
    path.write_text(json.dumps(out, indent=1))
    a, b = rows[0], rows[-1]
    print(f"{run}: steps {a['step']} -> {b['step']}")
    print("global mean lev0..: start", [round(v, 2) for v in a["global_mean"]],
          "end", [round(v, 2) for v in b["global_mean"]])
    print("lev0 zonal change by 10-deg band (S->N):")
    print(" ".join(f"{bb[0]-aa[0]:7.1f}" for aa, bb in zip(a["bands"], b["bands"])))


if __name__ == "__main__":
    main()
