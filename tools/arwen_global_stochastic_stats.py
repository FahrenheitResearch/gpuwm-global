"""Measure the stochastic patterns against their settings (JSON on stdout).

    python tools/arwen_global_stochastic_stats.py --truncation 383 --draws 120

For every SPPT scale and the SKEB streamfunction: the point variance, the
distance at which the meridional correlation falls to exp(-1/2) (the
requested correlation length), and the lag-one correlation of the AR(1)
coefficients at the model step (the requested decorrelation time).  Plus
the point standard deviation of the summed SPPT pattern and of its bounded
value.  Numpy, CPU only.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from arwen_global.spectral.transform import SphericalHarmonicTransform
from arwen_global.stochastic import StochasticConfig, StochasticPhysics


def correlation_length(fields, lat, radius):
    band = np.abs(np.rad2deg(lat)) < 60
    rows = np.flatnonzero(band)
    dlat = float(np.mean(np.abs(np.diff(lat[rows]))))
    target = math.exp(-0.5)
    previous = (0.0, 1.0)
    for k in range(1, len(rows) // 2):
        a = fields[:, rows[:-k]]
        b = fields[:, rows[:-k] + k]
        corr = float(np.mean(a * b) / math.sqrt(np.mean(a * a) * np.mean(b * b)))
        distance = k * dlat * radius
        if corr < target:
            x0, y0 = previous
            return x0 + (target - y0) * (distance - x0) / (corr - y0)
        previous = (distance, corr)
    return float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncation", type=int, default=383)
    parser.add_argument("--draws", type=int, default=120)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--dt", type=float, default=300.0)
    args = parser.parse_args(argv)
    transform = SphericalHarmonicTransform.create(
        args.truncation, backend="numpy", precision="float64")
    radius = transform.grid.radius_m
    lat = transform.grid.lat_rad
    base = StochasticConfig(sppt=True, skeb=True)
    out = {"truncation": args.truncation, "draws": args.draws, "processes": []}
    sums, bounded = [], []
    per_process = None
    for member in range(args.draws):
        sp = StochasticPhysics(StochasticConfig(sppt=True, skeb=True, member=member), transform)
        coeff, clock = sp._empty_state()
        coeff, clock, _ = sp.advance(coeff, clock, 0.0)
        if per_process is None:
            per_process = [[] for _ in sp.processes]
        for i in range(len(sp.processes)):
            per_process[i].append(np.asarray(transform.inverse(sp.spectral(coeff, i)), np.float32))
        total = np.asarray(transform.inverse(sp.sppt_spectral(coeff)))
        sums.append(total.astype(np.float32))
        bounded.append(np.clip(total, -1, 1).astype(np.float32))
    sp = StochasticPhysics(base, transform)
    # AR(1): pooled over modes and members, the correlation of each
    # process's coefficients after one step and after one decorrelation
    # time of model steps (expected exp(-1)).
    taus = [p.time_s for p in sp.processes]
    lag_steps = [max(1, round(t / args.dt)) for t in taus]
    pairs_one = [[[], []] for _ in taus]
    pairs_tau = [[[], []] for _ in taus]
    for member in range(args.members):
        run = StochasticPhysics(StochasticConfig(sppt=True, skeb=True, member=10_000 + member), transform)
        coeff, clock = run._empty_state()
        coeff, clock, _ = run.advance(coeff, clock, 0.0)
        start = coeff.copy()
        for k in range(1, max(lag_steps) + 1):
            coeff, clock, _ = run.advance(coeff, clock, k * args.dt)
            for i, n in enumerate(lag_steps):
                if k == 1:
                    pairs_one[i][0].append(start[i].ravel()); pairs_one[i][1].append(coeff[i].ravel())
                if k == n:
                    pairs_tau[i][0].append(start[i].ravel()); pairs_tau[i][1].append(coeff[i].ravel())

    def corr(pair):
        a, b = np.concatenate(pair[0]), np.concatenate(pair[1])
        return float(np.sum(a * b) / math.sqrt(np.sum(a * a) * np.sum(b * b)))

    for i, p in enumerate(sp.processes):
        f = np.stack(per_process[i])
        measured_length = correlation_length(f, lat, radius)
        lag1, lagtau = corr(pairs_one[i]), corr(pairs_tau[i])
        out["processes"].append({
            "name": p.name, "degree": p.degree,
            "length_requested_km": p.length_m / 1e3,
            "length_measured_km": measured_length / 1e3,
            "point_variance": float(np.mean(f.astype(np.float64) ** 2)),
            "time_requested_h": p.time_s / 3600.0,
            "lag1_measured": lag1, "lag1_expected": math.exp(-args.dt / p.time_s),
            "lag_tau_steps": lag_steps[i],
            "lag_tau_measured": lagtau, "lag_tau_expected": math.exp(-lag_steps[i] * args.dt / p.time_s),
            "time_measured_h": (-lag_steps[i] * args.dt / math.log(lagtau)) / 3600.0 if 0 < lagtau < 1 else None,
        })
    s = np.stack(sums).astype(np.float64)
    out["sppt_sum_std"] = float(np.std(s))
    out["sppt_sum_std_expected"] = math.sqrt(sum(a * a for a, _l, _t in base.sppt_scales))
    out["sppt_bounded_std"] = float(np.std(np.stack(bounded).astype(np.float64)))
    out["sppt_fraction_clipped"] = float(np.mean(np.abs(s) > 1))
    out["skeb_curl_rms_per_m"] = sp.curl_rms
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
