#!/usr/bin/env python3
"""anlzon.py DATE... : analysis zonal-mean theta at the model level-0/1 mid
pressures (1.2017, 1.7353 hPa) in 10-degree bands, from tmp-DATE.grb
(GDAS f000 TMP at 0.4/0.7/1/2/3/5 hPa).  Writes anl-zonal.json."""
import json, subprocess, sys
import numpy as np
P = {"0.4": 0.4, "0.7": 0.7, "1": 1.0, "2": 2.0, "3": 3.0, "5": 5.0}
PM = [1.201684076246857, 1.7352854241687516, 2.5058295793826804]
edges = np.arange(-90, 91, 10)
out = {"pmid_hpa": PM, "band_edges_deg": edges.tolist(), "dates": {}}
for date in sys.argv[1:]:
    prof = []
    for lab, typ in (("40", "isobaricInPa"), ("70", "isobaricInPa"), ("1", "isobaricInhPa"),
                     ("2", "isobaricInhPa"), ("3", "isobaricInhPa"), ("5", "isobaricInhPa")):
        txt = subprocess.run(["grib_get_data", "-w", f"level={lab},typeOfLevel={typ}", f"tmp-{date}.grb"],
                             capture_output=True, text=True).stdout.split("\n")[1:]
        a = np.array([l.split() for l in txt if l.strip()], dtype=float)
        lat = a[:, 0]; v = a[:, 2]
        ul = np.unique(lat)
        zm = np.array([v[lat == x].mean() for x in ul])
        prof.append(zm)
    prof = np.array(prof)          # (6 levels, nlat)
    plev = np.array([0.4, 0.7, 1.0, 2.0, 3.0, 5.0])
    bands = []
    for k, pm in enumerate(PM):
        t = np.array([np.interp(np.log(pm), np.log(plev), prof[:, j]) for j in range(prof.shape[1])])
        th = t * (1000.0 / pm) ** (287.04 / 1004.5)
        w = np.cos(np.deg2rad(ul))
        row = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ul >= lo) & (ul < hi) if hi < 90 else (ul >= lo) & (ul <= hi)
            row.append(float((th[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float(th[m].mean()))
        bands.append(row)
    out["dates"][date] = bands
json.dump(out, open("anl-zonal.json", "w"), indent=1)
for d, b in out["dates"].items():
    print(d, " ".join(f"{x:7.1f}" for x in b[0]))
