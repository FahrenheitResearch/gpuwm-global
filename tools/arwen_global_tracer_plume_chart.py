"""Maps of a replayed row's column burden, one panel per chosen sidecar.

Reads the sidecars a composition replay wrote (``*.composition.npz``) and
the weather checkpoints beside them (for the layer thickness and the land
outline), and draws the vertically integrated burden of one row on the
model's Gaussian grid.  It also prints, per panel, the row's float64 mass,
the burden's maximum, its mass-weighted centre and the area above a tenth
of a gram per square metre, so the figure's numbers are in the log beside
it.

usage:
  python tools/arwen_global_tracer_plume_chart.py --config RUN.toml \\
      --replay DIR --weather DIR --row NAME --steps 72,288,576,864,1440 \\
      --out FIGURE.png [--accept-geometry-config] [--title TEXT]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from arwen_global.composition import FlowReader, read_sidecar, sidecar_path
    from arwen_global.config import load_config
    from arwen_global.constants import GRAVITY_M_S2

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--row", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--accept-geometry-config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    reader = FlowReader(cfg)
    to_numpy = reader.transform.backend.to_numpy
    grid = reader.transform.grid
    lat = np.degrees(np.asarray(grid.lat_rad, dtype=np.float64))
    nlon = int(grid.nlon)
    lon = np.arange(nlon) * 360.0 / nlon
    area = (4.0 * math.pi * float(grid.radius_m) ** 2
            * np.asarray(grid.quadrature_weights, dtype=np.float64)
            / (2.0 * nlon))[:, None]
    steps = [int(item) for item in args.steps.split(",")]
    weather = Path(args.weather)
    panels = []
    for step in steps:
        checkpoint = weather / f"arwen_global_step{step:08d}.npz"
        meta, rows = read_sidecar(sidecar_path(args.replay, checkpoint))
        if meta["weather"]["checkpoint"] != checkpoint.name:
            raise SystemExit(f"sidecar for {checkpoint.name} names another checkpoint")
        flow = reader.frame(checkpoint)
        dp = np.asarray(to_numpy(flow.dp), dtype=np.float64)
        burden = np.sum(np.asarray(rows[args.row], dtype=np.float64) * dp,
                        axis=0) / GRAVITY_M_S2 * 1.0e3  # g m-2
        total = float(np.sum(burden * area)) / 1.0e3
        with np.load(checkpoint, allow_pickle=False) as archive:
            land = np.array(archive["surface__land_fraction"])
        x = np.cos(np.radians(lat))[:, None] * np.cos(np.radians(lon))[None]
        y = np.cos(np.radians(lat))[:, None] * np.sin(np.radians(lon))[None]
        z = np.broadcast_to(np.sin(np.radians(lat))[:, None], burden.shape)
        w = burden * area
        cx, cy, cz = (float(np.sum(c * w) / max(np.sum(w), 1e-300)) for c in (x, y, z))
        centre = (math.degrees(math.atan2(cy, cx)) % 360.0,
                  math.degrees(math.atan2(cz, math.hypot(cx, cy))))
        above = float(np.sum(area[:, 0][:, None] * (burden > 0.1)) / 1.0e6)
        record = {
            "step": step, "hours": meta["weather"]["time_s"] / 3600.0,
            "row_mass_kg": total,
            "ledger_mass_kg": meta["ledger"]["mass_kg"][args.row],
            "emitted_kg": meta["ledger"]["emitted_kg"].get(args.row),
            "burden_max_g_m2": float(burden.max()),
            "centre_lon_lat": [round(centre[0], 2), round(centre[1], 2)],
            "area_above_0p1_g_m2_km2": above,
        }
        print(json.dumps(record), flush=True)
        panels.append((record, burden, land))

    count = len(panels)
    cols = 2 if count > 1 else 1
    rowsn = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(7.2 * cols, 3.9 * rowsn),
                             squeeze=False, constrained_layout=True)
    vmax = max(float(panel[1].max()) for panel in panels)
    norm = LogNorm(vmin=max(vmax * 1.0e-4, 1.0e-6), vmax=vmax)
    shift = nlon // 2
    lon_c = np.concatenate([lon[shift:] - 360.0, lon[:shift]])
    image = None
    for index, (record, burden, land) in enumerate(panels):
        ax = axes[index // cols][index % cols]
        field = np.concatenate([burden[:, shift:], burden[:, :shift]], axis=1)
        mask = np.concatenate([land[:, shift:], land[:, :shift]], axis=1)
        masked = np.ma.masked_less(field, norm.vmin)
        image = ax.pcolormesh(lon_c, lat, masked, norm=norm, cmap="magma_r",
                              shading="nearest", rasterized=True)
        ax.contour(lon_c, lat, mask, levels=[0.5], colors="0.45",
                   linewidths=0.4)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.tick_params(labelsize=7)
        ax.set_title(f"+{record['hours']:.0f} h   max {record['burden_max_g_m2']:.3g} g m$^{{-2}}$",
                     fontsize=9, loc="left")
    for index in range(count, rowsn * cols):
        axes[index // cols][index % cols].axis("off")
    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01)
    bar.set_label(f"{args.row} column burden (g m$^{{-2}}$)", fontsize=8)
    if args.title:
        fig.suptitle(args.title, fontsize=10)
    fig.savefig(args.out, dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
