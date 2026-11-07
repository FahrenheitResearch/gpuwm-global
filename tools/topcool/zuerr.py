#!/usr/bin/env python3
"""zuerr.py DAY0_DATE STEPS_PER_DAY RUN... : zonal-mean zonal wind of the top
5 levels against the GDAS analysis (ugrd-DATE.grb at 1/2/3/5/10 hPa, ln-p
interpolated), area-weighted RMS over 10-degree bands, per daily checkpoint."""
import json, sys, re, subprocess, datetime as dt
from pathlib import Path
import numpy as np
from arwen_global.spectral.transform import SphericalHarmonicTransform
d0 = dt.datetime.strptime(sys.argv[1], "%Y%m%d"); spd = int(sys.argv[2])
L = 5
PH = np.array([100.0, 144.4044619105262, 208.52648619668622, 301.12155033325286, 434.83295445536834, 627.9181880909184]) / 100
PM = np.sqrt(PH[:-1] * PH[1:])
edges = np.arange(-90, 91, 10); bw = np.cos(np.deg2rad(0.5 * (edges[:-1] + edges[1:])))
_anl = {}
def anl_bands(date):
    if date in _anl: return _anl[date]
    prof = []
    for lev in ("1", "2", "3", "5", "10"):
        txt = subprocess.run(["grib_get_data", "-w", f"level={lev},typeOfLevel=isobaricInhPa", f"anl/ugrd-{date}.grb"],
                             capture_output=True, text=True).stdout.split("\n")[1:]
        a = np.array([l.split() for l in txt if l.strip()], dtype=float)
        lat, v = a[:, 0], a[:, 2]; ul = np.unique(lat)
        prof.append(np.array([v[lat == x].mean() for x in ul]))
    prof = np.array(prof); plev = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    out = []
    for pm in PM:
        u = np.array([np.interp(np.log(pm), np.log(plev), prof[:, j]) for j in range(prof.shape[1])])
        w = np.cos(np.deg2rad(ul))
        out.append([float((u[(ul >= lo) & (ul < lo + 10)] * w[(ul >= lo) & (ul < lo + 10)]).sum() / max(w[(ul >= lo) & (ul < lo + 10)].sum(), 1e-12)) if lo < 80 else float(u[(ul >= lo)].mean()) for lo in edges[:-1]])
    _anl[date] = np.array(out)
    return _anl[date]
tr = None; res = {}
for run in sys.argv[3:]:
    files = sorted(Path(run).glob("arwen_global_step*.npz"), key=lambda p: int(re.findall(r"(\d+)", p.name)[-1]))
    res[run] = []
    for f in files:
        step = int(re.findall(r"(\d+)", f.name)[-1])
        if step % spd: continue
        z = np.load(f)
        if tr is None:
            t = z["atmosphere__vorticity"].shape[-1] - 1; nlat, nlon = z["atmosphere__qc"].shape[-2:]
            tr = SphericalHarmonicTransform.create(t, nlat=nlat, nlon=nlon, backend="numpy", precision="float64")
            lat = np.degrees(np.arcsin(np.asarray(tr.grid.sin_lat, np.float64))); w = np.asarray(tr.grid.quadrature_weights, np.float64)
        vor = z["atmosphere__vorticity"][:L].astype(np.complex128).copy()
        vor[..., :, 1:] = 0.0                      # zonal mean only
        psi = tr.inverse_laplacian(vor)
        u = -np.asarray(tr.inverse_meridional_derivative(psi), np.float64).mean(-1) / tr.grid.radius_m
        mb = np.array([[(u[k, (lat >= lo) & (lat < lo + 10)] * w[(lat >= lo) & (lat < lo + 10)]).sum() / w[(lat >= lo) & (lat < lo + 10)].sum() for lo in edges[:-1]] for k in range(L)])
        date = (d0 + dt.timedelta(days=step // spd)).strftime("%Y%m%d")
        err = mb - anl_bands(date)
        rms = np.sqrt((err ** 2 * bw).sum(-1) / bw.sum())
        res[run].append({"day": step // spd, "u_rms_err_ms": rms.round(2).tolist(), "lev0_model": mb[0].round(1).tolist(), "lev0_anl": anl_bands(date)[0].round(1).tolist()})
        print(run, "day", step // spd, "zonal-mean u RMS err lev0-4 (m/s):", rms.round(1))
    print(run, "lev0 model S->N:", " ".join(f"{x:5.0f}" for x in res[run][-1]["lev0_model"]))
    print(run, "lev0 anl   S->N:", " ".join(f"{x:5.0f}" for x in res[run][-1]["lev0_anl"]))
json.dump(res, open("zuerr-out.json", "w"), indent=1)
