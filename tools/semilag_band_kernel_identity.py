"""Device identity of the semi-Lagrangian band kernels.

Two questions, both answered on the card:

1. Does a latitude band read through its row halo return the whole-grid
   kernel's bits at every row it computes?  (departure search, cubic
   limited gather with its clip deficit, quintic gather, parallel
   transport, Lipschitz maxima)
2. Are the whole-grid kernels themselves unchanged?  Prints a sha256 per
   whole-grid output so a run of this script against an older tree can be
   compared line for line.

Usage: python tools/semilag_band_kernel_identity.py [truncation] [bands] [whole]
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


def main(argv) -> int:
    import cupy as cp

    from arwen_global.spectral.transform import SphericalHarmonicTransform
    from arwen_global.semilag.tables import SphericalGridTables
    from arwen_global.semilag.trajectory import (
        CartesianWind, cartesian_wind, departure_points, lipschitz,
    )
    from arwen_global.semilag.interpolate import gather_batch
    from arwen_global.semilag.vectors import transport_to_arrival

    truncation = int(argv[1]) if len(argv) > 1 else 255
    bands = int(argv[2]) if len(argv) > 2 else 8
    transform = SphericalHarmonicTransform.create(
        truncation, backend="cupy", precision="float32")
    grid = transform.grid
    nlat, nlon = grid.shape
    nlev = 40
    dtype = np.float32
    tables = SphericalGridTables.create(grid, xp=cp, dtype=dtype)
    lat = cp.asarray(grid.lat_rad, dtype=cp.float64)[None, :, None]
    lon = cp.asarray(grid.lon_rad, dtype=cp.float64)[None, None, :]
    level = cp.arange(nlev, dtype=cp.float64)[:, None, None] / nlev
    u = (60.0 * cp.cos(lat) * (0.3 + level) + 15.0 * cp.sin(5.0 * lon)
         * cp.cos(lat) ** 2).astype(dtype)
    v = (25.0 * cp.sin(3.0 * lon + 2.0 * level) * cp.cos(lat) ** 2
         ).astype(dtype)
    s = (2.0e-4 * cp.cos(2.0 * lon) * cp.cos(lat) * (level - 0.5)).astype(dtype)
    vx, vy, vz = cartesian_wind(u, v, tables)
    wind = CartesianWind(vx, vy, vz, s)
    ex = CartesianWind(*(1.01 * a for a in wind.arrays()))
    dt = 300.0
    rng = cp.random.default_rng(3)
    fields = [rng.standard_normal((nlev, nlat, nlon), dtype=cp.float32)
              for _ in range(3)]
    positive = [cp.maximum(f, 0.0) for f in fields]

    def digest(a) -> str:
        h = cp.asnumpy(a)
        return hashlib.sha256(h.tobytes()).hexdigest()[:16]

    stencil, diag = departure_points(wind, tables, dt, extrapolated=ex)
    cubic = gather_batch(positive, stencil, monotone="quasi_monotone",
                         deficit=True, batch=8)
    quintic = gather_batch(fields, stencil, monotone=False, order=6, batch=8)
    au, av = transport_to_arrival(*quintic, stencil, tables)
    whole_lipschitz = lipschitz(wind, tables, dt)
    report = {
        "truncation": truncation, "bands": bands,
        "whole": {
            "xi": digest(stencil.xi), "phi": digest(stencil.phi),
            "level": digest(stencil.level),
            "cubic_qm": [digest(v) for v, _c in cubic],
            "cubic_deficit": [digest(c) for _v, c in cubic],
            "quintic": [digest(v) for v in quintic],
            "transport": [digest(au), digest(av)],
            "lipschitz_balanced_s": whole_lipschitz.jacobian_balanced_s,
        },
        "bands_equal": [],
    }
    if len(argv) > 3 and argv[3] == "whole":
        # The whole-grid digests alone, for a tree without band windows.
        print(json.dumps(report))
        return 0
    from arwen_global.semilag.halo import default_halo_rows
    from arwen_global.bands import BandPipeline

    halo = default_halo_rows(nlat, dt, grid.radius_m)
    pipeline = BandPipeline(nlat, bands)
    folded = 0.0
    for rows in pipeline.slices():
        held, _l, _t = pipeline.halo(rows, halo)
        window = tables.window(held, rows)
        a0, a1 = rows.start, rows.stop
        cut = lambda f: cp.ascontiguousarray(f[:, held])  # noqa: E731
        band_wind = CartesianWind(*(cut(a) for a in wind.arrays()))
        band_ex = CartesianWind(*(cp.ascontiguousarray(a[:, rows])
                                  for a in ex.arrays()))
        bs, bd = departure_points(band_wind, tables, dt,
                                  extrapolated=band_ex, window=window)
        bc = gather_batch([cut(f) for f in positive], bs,
                          monotone="quasi_monotone", deficit=True, batch=8)
        bq = gather_batch([cut(f) for f in fields], bs, monotone=False,
                          order=6, batch=8)
        bu, bv = transport_to_arrival(*bq, bs, tables)
        bl = lipschitz(band_wind, tables, dt, window=window)
        folded = max(folded, bl.jacobian_balanced_s)
        same = all([
            bool(cp.array_equal(bs.xi, stencil.xi[:, rows])),
            bool(cp.array_equal(bs.phi, stencil.phi[:, rows])),
            bool(cp.array_equal(bs.level, stencil.level[:, rows])),
            *(bool(cp.array_equal(x, y[:, rows])) and
              bool(cp.array_equal(c, d[:, rows]))
              for (x, c), (y, d) in zip(bc, cubic)),
            *(bool(cp.array_equal(x, y[:, rows])) for x, y in zip(bq, quintic)),
            bool(cp.array_equal(bu, au[:, rows])),
            bool(cp.array_equal(bv, av[:, rows])),
        ])
        report["bands_equal"].append([a0, a1, held.start, held.stop, same])
    report["lipschitz_folded_equal"] = (
        folded == whole_lipschitz.jacobian_balanced_s)
    report["all_bands_equal"] = all(row[-1] for row in report["bands_equal"])
    print(json.dumps(report))
    return 0 if report["all_bands_equal"] and report["lipschitz_folded_equal"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
