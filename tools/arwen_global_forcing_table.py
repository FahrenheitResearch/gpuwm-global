"""The forced-against-free table of a forcing twin pair, and the pull table
of a forced run's ledger.

A forcing twin is two runs from one start, one free and one pulled toward a
stored record of large scales (``gpuwm-global run --forcing``), set against
the run the record was cut from (the reference).  For every checkpoint time
the two runs share, this reads each run's departure from the reference by
field and by band of total degree and reports

    r = 1 - E_forced / E_free

per band: the fraction of the free run's departure the pull removed.  The
pulled band (degrees up to the record's cutoff) says how hard the pull
held; the bands above it say how much of the small scales the model made
again on its own when its large scales were held.  The comparison is over
the degrees all three carry (a T255 reference beside T127 runs is read on
degrees 0..127); ``--offset-s`` aligns a reference that started earlier.

``--ledger RUN_DIR`` prints the forced run's own pull table from its
``forcing_ledger.json``: per field at the levels nearest the requested
pressures, the departure from the record, the pull's RMS and the RMS of its
time mean (the systematic part, the model-error estimate), as tendencies
per day.

usage:
  python tools/arwen_global_forcing_table.py --reference REF_DIR \
      --free FREE_DIR --forced FORCED_DIR [--offset-s 0] \
      [--bands 1-42,43-84,85-127] [--ledger FORCED_DIR] \
      [--levels-hpa 850,500,250] [--out-json T.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

FIELDS = ("vorticity", "divergence", "theta", "qv", "log_surface_pressure")
EARTH_RADIUS_M = 6371220.0


def _checkpoints(run_dir: Path) -> dict[float, Path]:
    from arwen_global.checkpoint import read_checkpoint_header

    out: dict[float, Path] = {}
    for path in sorted(Path(run_dir).glob("arwen_global_step*.npz")):
        meta = read_checkpoint_header(path)
        out[float(meta["time_s"])] = path
    return out


def _spectral(path: Path, k: int) -> dict[str, np.ndarray]:
    from arwen_global.checkpoint import read_checkpoint

    _meta, arrays = read_checkpoint(path)
    return {name: np.asarray(arrays[f"atmosphere__{name}"][..., :k, :k], dtype=np.complex128)
            for name in FIELDS}


def _truncation(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        return int(archive["atmosphere__theta"].shape[-1]) - 1


def _degree_power(c: np.ndarray) -> np.ndarray:
    """Per-degree power ``(..., n)``: the sum over n is the grid mean square."""
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0) / (4.0 * math.pi)
    return np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=-1)


def _wind_factor(k: int) -> np.ndarray:
    n = np.arange(k, dtype=np.float64)
    f = np.zeros(k)
    f[1:] = EARTH_RADIUS_M ** 2 / (n[1:] * (n[1:] + 1.0))
    return f


def _band_powers(diff: dict[str, np.ndarray], bands) -> dict[str, list[float]]:
    """Level-mean power per band for each field and for the wind."""
    out: dict[str, list[float]] = {}
    powers = {}
    for name, c in diff.items():
        p = _degree_power(c)
        powers[name] = p if p.ndim == 1 else p.mean(axis=0)
    k = next(iter(diff.values())).shape[-1]
    powers["wind"] = (powers["vorticity"] + powers["divergence"]) * _wind_factor(k)
    for name, p in powers.items():
        out[name] = [float(p[lo:hi + 1].sum()) for lo, hi in bands]
    return out


def _parse_bands(text: str) -> list[tuple[int, int]]:
    bands = []
    for part in text.split(","):
        lo, hi = part.split("-")
        bands.append((int(lo), int(hi)))
    return bands


def twin_table(reference: Path, free: Path, forced: Path, *, offset_s: float, bands) -> dict:
    ref_ck, free_ck, forced_ck = _checkpoints(reference), _checkpoints(free), _checkpoints(forced)
    k = 1 + min(_truncation(next(iter(ref_ck.values()))), _truncation(next(iter(free_ck.values()))))
    bands = [(lo, min(hi, k - 1)) for lo, hi in bands if lo <= k - 1]
    rows = []
    for t in sorted(set(free_ck) & set(forced_ck)):
        ref_path = ref_ck.get(t + offset_s)
        if ref_path is None:
            continue
        ref = _spectral(ref_path, k)
        a = _spectral(free_ck[t], k)
        b = _spectral(forced_ck[t], k)
        e_free = _band_powers({n: a[n] - ref[n] for n in FIELDS}, bands)
        e_forced = _band_powers({n: b[n] - ref[n] for n in FIELDS}, bands)
        fields = {}
        for name in e_free:
            fields[name] = {
                "free_rms": [math.sqrt(v) for v in e_free[name]],
                "forced_rms": [math.sqrt(v) for v in e_forced[name]],
                "r": [None if f == 0.0 else 1.0 - g / f for f, g in zip(e_free[name], e_forced[name])],
            }
        rows.append({"time_s": t, "hours": t / 3600.0, "fields": fields})
    return {"reference": str(reference), "free": str(free), "forced": str(forced),
            "offset_s": offset_s, "compared_degrees": k - 1,
            "bands": [list(b) for b in bands],
            "conventions": "rms = sqrt of the level-mean grid mean square of (run - reference) in the band; "
                           "r = 1 - E_forced / E_free; wind from vorticity and divergence in m/s",
            "rows": rows}


def ledger_table(run_dir: Path, levels_hpa) -> dict:
    ledger = json.loads((Path(run_dir) / "forcing_ledger.json").read_text(encoding="utf-8"))
    out = {"options": ledger["options"], "record": ledger["record"]["sha256"],
           "steps": ledger["steps"], "observe_only": ledger["observe_only"], "fields": {}}
    for name, block in ledger["fields"].items():
        levels = block["levels"]
        picked = []
        if levels and "reference_pressure_hpa" in levels[0]:
            for want in levels_hpa:
                row = min(levels, key=lambda r: abs(r["reference_pressure_hpa"] - want))
                if row not in picked:
                    picked.append(row)
        else:
            picked = levels
        out["fields"][name] = {"units": block["units"], "levels": picked}
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3g}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--free", type=Path)
    parser.add_argument("--forced", type=Path)
    parser.add_argument("--offset-s", type=float, default=0.0)
    parser.add_argument("--bands", default="1-42,43-84,85-127")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--levels-hpa", default="850,500,250")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)
    result: dict[str, object] = {}
    if args.reference and args.free and args.forced:
        table = twin_table(args.reference, args.free, args.forced, offset_s=args.offset_s,
                           bands=_parse_bands(args.bands))
        result["twin"] = table
        bands = ", ".join(f"{lo}-{hi}" for lo, hi in table["bands"])
        print(f"forced against free, departure from the reference by band ({bands}); r = 1 - E_forced/E_free")
        for row in table["rows"]:
            for name, f in row["fields"].items():
                print(f"  {row['hours']:6.1f} h {name:22s} free {' '.join(_fmt(v) for v in f['free_rms'])}"
                      f" | forced {' '.join(_fmt(v) for v in f['forced_rms'])}"
                      f" | r {' '.join(_fmt(v) for v in f['r'])}")
    if args.ledger is not None:
        table = ledger_table(args.ledger, [float(v) for v in args.levels_hpa.split(",")])
        result["ledger"] = table
        print(f"pull ledger of {args.ledger} ({table['steps']} steps, observe_only={table['observe_only']})")
        for name, block in table["fields"].items():
            for row in block["levels"]:
                label = row.get("reference_pressure_hpa", row.get("level"))
                extra = row.get("global_mean_pull_per_day")
                print(f"  {name:22s} {str(label):>9s}  departure {_fmt(row['departure_rms'])}"
                      f"  pull/day {_fmt(row['pull_rms_per_day'])}"
                      f"  mean pull/day {_fmt(row.get('time_mean_pull_rms_per_day'))}"
                      + ("" if extra is None else f"  global-mean pull/day {_fmt(extra)}")
                      + f"  [{block['units']}]")
    if not result:
        parser.error("give --reference/--free/--forced, or --ledger, or both")
    if args.out_json is not None:
        args.out_json.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
