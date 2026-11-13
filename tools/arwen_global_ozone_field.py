"""Ozone from a GDAS analysis on the model's grid and levels, for a replay row.

The 0.25-degree GDAS pgrb2 analysis carries the ozone mixing ratio (O3MR)
on its isobaric levels and a total-ozone record (TOZNE).  This driver
decodes both through the package's mapped engine door
(``mapped_source_compat.decode_through_engine``) with a mapping it builds
from the microwave-column mapping the package already carries (same grid,
same 41 isobaric levels, the two ozone records in place of the
temperature and moisture ones), regrids every level onto the run's
Gaussian grid with the cold start's own regridder, and interpolates in
ln p onto the model's full levels as the weather checkpoint at the start of
the replay holds them.  The result is an ``.npz`` a ``kind = "field"``
composition row reads (array ``ozone``, kg/kg).

It also takes the plumbing gate: the model's total column, summed over its
own layers, against the analysis's TOZNE regridded to the same grid, in
Dobson units.  The isobaric levels below the ground are the analysis's own
extrapolation and the model's levels never reach them; levels above the
analysis's top level take the top level's value.

usage:
  python tools/arwen_global_ozone_field.py --grib GDAS.f000 --config RUN.toml \\
      --checkpoint arwen_global_step00000000.npz --out OZONE.npz \\
      [--accept-geometry-config]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import tempfile

import numpy as np

#: Kilograms of ozone per square metre in one Dobson unit.
KG_PER_M2_PER_DU = 2.1415e-5

#: O3MR is an NCEP local parameter (192), so its selector binds the whole
#: Section 1 table authority, as the engine requires of a local code.
_O3MR = {"format": "grib2", "discipline": 0, "category": 14,
         "parameter": 192, "level_type": 100, "center": 7, "subcenter": 0,
         "master_table_version": 2, "local_table_version": 1}
_TOZNE = {"format": "grib2", "discipline": 0, "category": 14,
          "parameter": 0, "level_type": 200, "level_value": 0, "pdt": 0,
          "center": 7, "subcenter": 0, "master_table_version": 2,
          "local_table_version": 1}


def ozone_mapping() -> dict:
    """The microwave-column mapping with the two ozone records in it."""
    from arwen_global.microwave.columns import mapping_path

    base = json.loads(Path(mapping_path()).read_text(encoding="utf-8"))
    mapping = copy.deepcopy(base)
    mapping["name"] = "ncep-gdas-pgrb2-0p25-grib2-ozone"
    fields = {
        "ozone_mixing_ratio": {
            "selectors": [_O3MR],
            "units": {"source": "kg kg-1", "target": "kg kg-1"},
            "source_axes": ["vertical", "y", "x"],
            "target_axes": ["vertical", "y", "x"],
            "location": "mass", "staggering": "none",
            "missing": {"kind": "reject"},
        },
        "total_ozone": {
            "selectors": [_TOZNE],
            "units": {"source": "DU", "target": "DU"},
            "source_axes": ["y", "x"], "target_axes": ["y", "x"],
            "location": "surface", "staggering": "none",
            "missing": {"kind": "reject"},
        },
    }
    # Every record the base mapping's target requires stays in it; the two
    # ozone records are added beside them.
    mapping["fields"] = {**base["fields"], **fields}
    return mapping


def _vertical_interpolate(source_pa, values, target_pa):
    """Linear in ln p from fixed isobaric levels onto model full levels.

    ``source_pa`` is one-dimensional (the same levels in every column), so
    the bracket is one ``searchsorted`` over the flattened targets.
    """
    order = np.argsort(source_pa)
    lp = np.log(np.asarray(source_pa, dtype=np.float64)[order])
    data = np.asarray(values, dtype=np.float64)[order]
    flat = np.log(np.clip(np.asarray(target_pa, dtype=np.float64),
                          math.exp(lp[0]), math.exp(lp[-1]))).ravel()
    upper = np.clip(np.searchsorted(lp, flat), 1, lp.size - 1)
    lower = upper - 1
    weight = (flat - lp[lower]) / (lp[upper] - lp[lower])
    nlev, nlat, nlon = target_pa.shape
    columns = data.reshape(lp.size, -1)
    index = np.arange(nlat * nlon)
    index = np.tile(index, nlev)
    out = ((1.0 - weight) * columns[lower, index]
           + weight * columns[upper, index])
    return out.reshape(nlev, nlat, nlon)


def main(argv: list[str] | None = None) -> int:
    from arwen_global.analysis_initial import _global_regridder
    from arwen_global.composition import FlowReader
    from arwen_global.config import load_config
    from arwen_global.constants import GRAVITY_M_S2
    from arwen_global.checkpoint import read_checkpoint_header
    from arwen_global.mapped_source_compat import decode_through_engine

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grib", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--accept-geometry-config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    header = read_checkpoint_header(args.checkpoint)
    if str(header.get("config_hash")) != str(cfg.config_hash) and not args.accept_geometry_config:
        raise SystemExit(
            "the checkpoint's config hash is not the config's; pass the run's "
            "own config or --accept-geometry-config")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        mapping_file = Path(scratch) / "rw-wps-gdas-pgrb2-0p25-ozone.mapping.json"
        mapping_file.write_text(json.dumps(ozone_mapping(), indent=1),
                                encoding="utf-8")
        decoded = decode_through_engine(mapping_file, [Path(args.grib)],
                                        scratch_destination=scratch)
    frame = decoded.frames[0]
    source_pa = np.asarray(frame.vertical_values, dtype=np.float64)
    ozone_src = np.clip(np.asarray(frame.fields["ozone_mixing_ratio"].values,
                                   dtype=np.float64), 0.0, None)
    tozne_src = np.asarray(frame.fields["total_ozone"].values, dtype=np.float64)

    reader = FlowReader(cfg)
    reader.check_geometry(args.checkpoint)
    grid = reader.transform.grid
    regrid = _global_regridder(frame.latitude, frame.longitude, grid)
    on_grid = np.stack([np.asarray(regrid(level), dtype=np.float64)
                        for level in ozone_src])
    tozne = np.asarray(regrid(tozne_src), dtype=np.float64)
    flow = reader.frame(args.checkpoint)
    to_numpy = reader.transform.backend.to_numpy
    p_full = np.asarray(to_numpy(flow.p_full), dtype=np.float64)
    dp = np.asarray(to_numpy(flow.dp), dtype=np.float64)
    ozone = np.maximum(_vertical_interpolate(source_pa, on_grid, p_full), 0.0)

    column_du = np.sum(ozone * dp, axis=0) / GRAVITY_M_S2 / KG_PER_M2_PER_DU
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)[:, None] * np.ones_like(column_du)
    lat = np.degrees(np.asarray(grid.lat_rad, dtype=np.float64))[:, None] * np.ones_like(column_du)

    def bands(mask):
        w = weights * mask
        ratio = column_du / np.maximum(tozne, 1.0e-6)
        mean_model = float(np.sum(column_du * w) / np.sum(w))
        mean_tozne = float(np.sum(tozne * w) / np.sum(w))
        return {
            "model_column_du": mean_model,
            "analysis_tozne_du": mean_tozne,
            "relative_bias": mean_model / mean_tozne - 1.0,
            "rms_relative_difference": float(np.sqrt(np.sum((ratio - 1.0) ** 2 * w) / np.sum(w))),
            "max_abs_relative_difference": float(np.max(np.abs(ratio - 1.0)[mask > 0])),
        }

    gate = {
        "global": bands(np.ones_like(column_du)),
        "poleward_of_20": bands((np.abs(lat) >= 20.0).astype(float)),
        "tropics": bands((np.abs(lat) < 20.0).astype(float)),
    }
    metadata = {
        "schema": "gpuwm.arwen-global-ozone-field/v1",
        "grib": Path(args.grib).name,
        "input_sha256": dict(frame.input_sha256),
        "valid_time": frame.valid_time.isoformat(),
        "source_levels_pa": source_pa.tolist(),
        "weather_checkpoint": Path(args.checkpoint).name,
        "weather_self_sha256": header.get("self_sha256"),
        "gate_column_against_tozne": gate,
        "gate_pass_3_percent": abs(gate["global"]["relative_bias"]) <= 0.03,
    }
    np.savez(out, ozone=ozone, total_ozone_analysis_du=tozne,
             column_model_du=column_du,
             __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata["gate_column_against_tozne"], indent=2))
    print("source levels:", len(source_pa), "from", source_pa.min(), "to", source_pa.max(), "Pa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
