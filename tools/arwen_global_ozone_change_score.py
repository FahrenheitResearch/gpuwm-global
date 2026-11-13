"""Score a replayed ozone row's change of total column against the analysis's.

The row starts from the analysis's own ozone on the model's levels
(``tools/arwen_global_ozone_field.py``) and is carried by the replayed flow
alone.  The test is on the CHANGE of the column, forecast minus start
against analysis minus start, so a fixed difference between the analysis's
isobaric ozone and the model-level column cancels, and persistence scores a
correlation of zero by construction:

* ``pattern_correlation``: area-weighted, centred, of the forecast change
  against the analysis change;
* ``rmse_over_rms_change``: the RMS of the forecast change's error, as a
  fraction of the RMS of the analysis change (persistence reads 1).

Both are reported poleward of 20 degrees, in the tropics and globally.  The
start and verifying columns are both the analysis's ozone integrated over
the model's layers at their own times, by the same driver.

usage:
  python tools/arwen_global_ozone_change_score.py --config RUN.toml \\
      --start OZONE_T0.npz --verify OZONE_T.npz --sidecar SIDE_T.composition.npz \\
      --checkpoint WEATHER_T.npz [--row ozone] [--accept-geometry-config] [--out R.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KG_PER_M2_PER_DU = 2.1415e-5


def scores(forecast_change, analysis_change, weights, mask) -> dict:
    w = weights * mask
    total = np.sum(w)
    f = forecast_change - np.sum(forecast_change * w) / total
    a = analysis_change - np.sum(analysis_change * w) / total
    correlation = float(np.sum(f * a * w) / np.sqrt(np.sum(f * f * w) * np.sum(a * a * w)))
    rms_change = float(np.sqrt(np.sum(analysis_change ** 2 * w) / total))
    rmse = float(np.sqrt(np.sum((forecast_change - analysis_change) ** 2 * w) / total))
    return {
        "pattern_correlation": correlation,
        "rms_analysis_change_du": rms_change,
        "rmse_forecast_change_du": rmse,
        "rmse_over_rms_change": rmse / rms_change,
    }


def main(argv: list[str] | None = None) -> int:
    from arwen_global.composition import FlowReader, read_sidecar
    from arwen_global.config import load_config
    from arwen_global.constants import GRAVITY_M_S2

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--verify", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--row", default="ozone")
    parser.add_argument("--out")
    parser.add_argument("--accept-geometry-config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    meta, rows = read_sidecar(args.sidecar)
    if meta["weather"]["checkpoint"] != Path(args.checkpoint).name:
        raise SystemExit("the sidecar belongs to another weather checkpoint")
    reader = FlowReader(cfg)
    flow = reader.frame(args.checkpoint)
    dp = np.asarray(reader.transform.backend.to_numpy(flow.dp), dtype=np.float64)
    forecast = (np.sum(np.asarray(rows[args.row], dtype=np.float64) * dp, axis=0)
                / GRAVITY_M_S2 / KG_PER_M2_PER_DU)
    with np.load(args.start) as start, np.load(args.verify) as verify:
        start_column = np.array(start["column_model_du"])
        verify_column = np.array(verify["column_model_du"])
    grid = reader.transform.grid
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)[:, None] * np.ones_like(forecast)
    lat = np.degrees(np.asarray(grid.lat_rad, dtype=np.float64))[:, None] * np.ones_like(forecast)
    forecast_change = forecast - start_column
    analysis_change = verify_column - start_column
    result = {
        "hours": meta["weather"]["time_s"] / 3600.0,
        "sidecar": Path(args.sidecar).name,
        "poleward_of_20": scores(forecast_change, analysis_change, weights,
                                 (np.abs(lat) >= 20.0).astype(float)),
        "tropics": scores(forecast_change, analysis_change, weights,
                          (np.abs(lat) < 20.0).astype(float)),
        "global": scores(forecast_change, analysis_change, weights,
                         np.ones_like(forecast)),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
