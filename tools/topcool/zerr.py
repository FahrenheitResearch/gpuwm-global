#!/usr/bin/env python3
"""zerr.py ANL_ZONAL.json DAY0_DATE STEPS_PER_DAY RUN... : model minus analysis
zonal-mean theta at levels 0-2 in 10-degree bands at every daily checkpoint;
area-weighted RMS over bands of the error CHANGE since day 0 (the drift)."""
import json, sys, re, datetime as dt
from pathlib import Path
import numpy as np
from arwen_global.spectral.transform import SphericalHarmonicTransform
anl = json.load(open(sys.argv[1])); d0 = dt.datetime.strptime(sys.argv[2], "%Y%m%d"); spd = int(sys.argv[3])
L = 3; tr = None
edges = np.arange(-90, 91, 10); centers = 0.5 * (edges[:-1] + edges[1:]); bw = np.cos(np.deg2rad(centers))
res = {}
for run in sys.argv[4:]:
    files = sorted(Path(run).glob("arwen_global_step*.npz"), key=lambda p: int(re.findall(r"(\d+)", p.name)[-1]))
    rows = []
    for f in files:
        step = int(re.findall(r"(\d+)", f.name)[-1])
        if step % spd: continue
        z = np.load(f)
        if tr is None:
            t = z["atmosphere__theta"].shape[-1] - 1; nlat, nlon = z["atmosphere__qc"].shape[-2:]
            tr = SphericalHarmonicTransform.create(t, nlat=nlat, nlon=nlon, backend="numpy", precision="float64")
            lat = np.degrees(np.arcsin(np.asarray(tr.grid.sin_lat, np.float64))); w = np.asarray(tr.grid.quadrature_weights, np.float64)
        g = np.asarray(tr.inverse(z["atmosphere__theta"][:L].astype(np.complex128)), np.float64).mean(-1)
        bands = np.array([[(g[k, (lat >= lo) & (lat < lo + 10)] * w[(lat >= lo) & (lat < lo + 10)]).sum() / w[(lat >= lo) & (lat < lo + 10)].sum() for lo in edges[:-1]] for k in range(L)])
        date = (d0 + dt.timedelta(days=step // spd)).strftime("%Y%m%d")
        a = np.array(anl["dates"][date])
        rows.append({"day": step // spd, "err": (bands - a).tolist()})
    e0 = np.array(rows[0]["err"])
    out = []
    for r in rows:
        de = np.array(r["err"]) - e0
        rms = np.sqrt((de ** 2 * bw).sum(-1) / bw.sum())
        out.append({"day": r["day"], "drift_rms_k": rms.round(2).tolist(), "drift_by_band_lev0": np.round(de[0], 1).tolist()})
        print(run, "day", r["day"], "band-RMS error change lev0-2:", rms.round(1), "| lev0 S->N:", " ".join(f"{x:5.0f}" for x in de[0]))
    res[run] = out
json.dump(res, open("zerr-out.json", "w"), indent=1)
