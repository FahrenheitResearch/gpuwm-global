#!/usr/bin/env python3
"""charts.py OUTDIR LABEL=DRIFTJSON:ZERRKEY ... : analysis charts (not weather fields).
Inputs: drift-<run>.json (drift.py), zerr-out.json, zuerr-out.json in cwd."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
out = sys.argv[1]
runs = [a.split("=") for a in sys.argv[2:]]          # label=run
zerr = json.load(open("zerr-out.json")); zu = json.load(open("zuerr-out.json"))
cent = np.arange(-85, 90, 10)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for lab, run in runs:
    d = json.load(open(f"drift-{run}.json"))
    days = [r["day"] for r in d["days"]]
    e = np.array([r["model_minus_analysis_k"][0] for r in d["days"]]); e -= e[0]
    ax[0].plot(days, e, marker="o", label=f"{lab} ({d['drift_vs_analysis_k_per_day'][0]:+.2f} K/day)")
    if run in zerr:
        ax[1].plot(cent, zerr[run][-1]["drift_by_band_lev0"], marker=".", label=f"{lab} day {zerr[run][-1]['day']}")
    if run in zu:
        ax[2].plot([r["day"] for r in zu[run]], [r["u_rms_err_ms"][0] for r in zu[run]], marker="o", label=lab)
ax[0].axhspan(-5, 5, color="0.9", zorder=0, label="+-1 K/day over 5 days")
ax[0].set(xlabel="forecast day", ylabel="K", title="Top level (1.0-1.44 hPa): global-mean theta,\nmodel minus GDAS, change since t=0")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set(xlabel="latitude (10-degree band centre)", ylabel="K", title="Top level: zonal-mean theta error change\n(model minus GDAS) at the last day")
ax[2].set(xlabel="forecast day", ylabel="m/s", title="Top level: zonal-mean u RMS error\nagainst GDAS (10-degree bands)")
for a in ax: a.legend(fontsize=7); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{out}/top-level-drift.png", dpi=130)
print("wrote", f"{out}/top-level-drift.png")
