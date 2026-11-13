"""Replay written in-situ ledgers through the stratospheric noise gate.

``python tools/arwen_global_strat_noise_replay.py CONFIG LEDGER [LEDGER ...]
--limit 1e-4 [--band-hpa 10,100] [--json out.json]``

Each ``insitu.ndjson`` is read as written (its spectra rows carry the
per-level top-decile and total kinetic energy) and fed, sample by sample,
to ``arwen_global.insitu.stratosphere.StratosphericNoiseGate``, the code a
run with ``[insitu] stratospheric_noise_limit`` set evaluates live.  CONFIG
supplies the vertical coordinate the band's levels are chosen on; it must
be the configuration the ledgers were written under (its level count is
checked against each ledger's header).  Nothing is recomputed from a state:
no transform, no model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def replay(config: Path, ledgers: list[Path], limit: float,
           band_hpa=(10.0, 100.0)) -> dict:
    from arwen_global.config import load_config
    from arwen_global.insitu.stratosphere import StratosphericNoiseGate

    cfg = load_config(config)
    vertical = SimpleNamespace(a_half_pa=cfg.a_half_pa, b_half=cfg.b_half)
    nlev = len(cfg.a_half_pa) - 1
    results = {}
    for path in ledgers:
        gate = StratosphericNoiseGate(vertical, limit, band_hpa)
        header = None
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "header":
                    header = row
                    if int(row.get("nlev", nlev)) != nlev:
                        raise SystemExit(
                            f"{path}: written with {row.get('nlev')} levels, "
                            f"{config} has {nlev}"
                        )
                elif row.get("kind") == "spectra":
                    gate.evaluate(row)
        summary = gate.summary()
        summary["dt_s"] = None if header is None else header.get("dt_s")
        results[str(path)] = summary
    return {"config": str(config), "limit": float(limit),
            "band_hpa": list(band_hpa), "ledgers": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("ledgers", type=Path, nargs="+")
    parser.add_argument("--limit", type=float, required=True)
    parser.add_argument("--band-hpa", default="10,100")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    band = tuple(float(v) for v in args.band_hpa.split(","))
    report = replay(args.config, args.ledgers, args.limit, band)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    for path, row in report["ledgers"].items():
        at = row["at"] or {}
        if row["value"] is None:
            print(f"{Path(path).name}: no spectra sample on a level in the band")
            continue
        print(f"{Path(path).name}: dt {row['dt_s']} s, {row['samples']} samples, "
              f"max share {row['value']:.3g} at {at.get('time_s', 0) / 3600:.1f} h "
              f"level {at.get('level')}, limit {row['limit']:g}: "
              + ("pass" if row["passed"] else "TRIPS"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
