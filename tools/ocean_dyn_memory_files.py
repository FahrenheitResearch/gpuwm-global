"""Stand-in slab files for MEASURING the dynamic ocean's memory and cost at a
truncation that has no calibrated slab (T383): a flat climatology and a
zero Q-flux file on the config's own grid, matched to its physics identity.

    python tools/ocean_dyn_memory_files.py CONFIG.toml --out DIR

The dynamic ocean needs the slab's files for the columns its grid does not
cover; the values in these do not change a single allocation, so a memory and
cost measurement may use them.  A forecast may not: their climate is not a
climate.  Both files say so in their provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    from arwen_global.config import load_config
    from arwen_global.ocean_coupler import physics_identity
    from arwen_global.ocean_qflux import QFLUX_SCHEMA, _write_npz
    from arwen_global.spectral.grid import GaussianGrid
    from arwen_global.surface_climatology import load_climatology, write_climatology

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    grid = GaussianGrid.create(int(cfg.truncation), nlat=cfg.nlat, nlon=cfg.nlon,
                               dealias_factor=float(cfg.dealias_factor))
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    args.out.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(grid.latitude_deg)[None, :, None]
    sst = 273.15 + np.clip(28.0 * np.cos(np.radians(lat)) ** 2, -1.8, 30.0) + np.zeros((12, nlat, nlon))
    ice = np.where(np.abs(lat) > 70.0, 0.9, 0.0) + np.zeros((12, nlat, nlon))
    clim_path = args.out / f"stand-in-climatology-t{cfg.truncation}.npz"
    write_climatology(clim_path, {"sst_k": sst, "sea_ice_fraction": ice},
                      ocean=np.ones((nlat, nlon), dtype=bool),
                      latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
                      provenance={"schema": "gpuwm.arwen-global-surface-climatology/v1",
                                  "stand_in": "memory and cost measurement only; not a climate"})
    clim = load_climatology(clim_path)
    provenance = {"schema": QFLUX_SCHEMA, "physics_identity": physics_identity(cfg),
                  "climatology_sha256": clim.sha256,
                  "stand_in": "zero Q-flux for a memory and cost measurement only"}
    q_path = args.out / f"stand-in-qflux-t{cfg.truncation}.npz"
    _write_npz(q_path, {
        "calibration_flux_w_m2": np.zeros((12, nlat, nlon)),
        "coverage_s": np.ones((12, nlat, nlon)),
        "latitude_deg": np.asarray(grid.latitude_deg, dtype=np.float64),
        "longitude_deg": np.asarray(grid.longitude_deg, dtype=np.float64),
        "provenance": np.asarray(json.dumps(provenance, default=str)),
    })
    print(json.dumps({"climatology": str(clim_path), "qflux": str(q_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
