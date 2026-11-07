#!/usr/bin/env python3
"""drift.py BUDGET.json ANL_FLDMEAN.txt [steps_per_day]

Top-level drift table from a top-budget probe: global-mean theta of the top
levels per day, the analysis at the same valid time (GDAS global-mean T at
0.4/0.7/1/2/3/5 hPa, ln-p interpolated to each model level's mid pressure),
model minus analysis, and the per-process daily budget.
"""
import json, sys, math
import numpy as np

b = json.load(open(sys.argv[1]))
anl = [l.split() for l in open(sys.argv[2]) if l.strip()]
spd = int(sys.argv[3]) if len(sys.argv) > 3 else 288
plev_hpa = np.array([0.4, 0.7, 1.0, 2.0, 3.0, 5.0])
ph = np.array(b["p_half_ref_pa_top"]) / 100.0
pmid = np.sqrt(ph[:-1] * ph[1:])
series = np.array(b["start_of_step_theta_k"] + [b["end_theta_k"]])
nlev_show = 3
out = {"levels_top_first_mid_hpa": pmid[:nlev_show].tolist(), "days": []}
for d in range(0, len(series) // spd + 1):
    i = min(d * spd, len(series) - 1)
    if d * spd > len(series) - 1:
        break
    row = {"day": d, "model_theta_k": series[i, :nlev_show].tolist()}
    if d < len(anl):
        t = np.array([float(x) for x in anl[d][1:7]])
        ta = np.interp(np.log(pmid[:nlev_show]), np.log(plev_hpa), t)
        th = ta * (1000.0 / pmid[:nlev_show]) ** (287.04 / 1004.5)
        row["analysis_date"] = anl[d][0]
        row["analysis_theta_k"] = th.tolist()
        row["model_minus_analysis_k"] = (series[i, :nlev_show] - th).tolist()
    out["days"].append(row)
days = out["days"]
nd = len(days) - 1
m0 = np.array(days[0]["model_theta_k"]); mN = np.array(days[nd]["model_theta_k"])
out["model_drift_k_per_day"] = ((mN - m0) / nd).tolist()
if "model_minus_analysis_k" in days[nd]:
    e0 = np.array(days[0]["model_minus_analysis_k"]); eN = np.array(days[nd]["model_minus_analysis_k"])
    out["drift_vs_analysis_k_per_day"] = ((eN - e0) / nd).tolist()
    out["analysis_change_k_per_day"] = ((np.array(days[nd]["analysis_theta_k"]) - np.array(days[0]["analysis_theta_k"])) / nd).tolist()
budget = {}
for k in ("physics_theta_change_k", "dynamics_theta_change_k"):
    for n, v in b.get(k, {}).items():
        if any(abs(x) > 1e-6 for x in v[:nlev_show]):
            budget[f"{k.split('_')[0]}:{n}"] = (np.array(v[:nlev_show]) / nd).round(3).tolist()
for k in ("sponge_ke_removed_theta_k",):
    if k in b:
        budget[k] = (np.array(b[k][:nlev_show]) / nd).round(3).tolist()
out["budget_k_per_day"] = budget
print(json.dumps(out, indent=1))
