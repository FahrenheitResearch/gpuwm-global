#!/usr/bin/env python3
"""Read 48 h twin runs of the albedo options against CERES and the analyses.

Three subcommands, each reading files only (no card, no network):

``ceres --dir DIR --month M --out JSON``
    The CERES-EBAF obs4MIPs monthly climatology for one calendar month
    (every year the files hold): global and latitude-band albedo, reflected
    and incoming shortwave, and outgoing longwave when an ``rlut_*.nc`` file
    sits beside the shortwave files.  Bands are the radiation scorecard's.

``score --run-dir RUN --analysis GRIB [--analysis GRIB ...] --out JSON``
    One run against verifying analyses at the leads whose valid time an
    analysis matches: the upper-air rows of record
    (``upper_air_scorecard.score_pair``: z500, t850, w250, w850, rh700) and
    2 m temperature against the analysis's own 2 m temperature on the
    model grid, plus precipitation between the 24 h and 48 h checkpoints.
    The 2 m comparison carries the analysis to the model's terrain with
    the standard-atmosphere lapse rate 6.5 K/km; the adjustment is the same
    for every arm of a case, so arm-minus-control differences do not
    depend on it.

``table --manifest JSON --out-json JSON --out-md MD``
    Assembles per-arm radiation scorecards (``python -m
    arwen_global.radiation_scorecard --window-hours 24 48``) and ``score``
    outputs into one table per case, with each arm's change from the
    control and its distance from CERES.  The manifest is
    ``{"cases": {NAME: {"ceres": PATH, "control": ARM,
    "arms": {ARM: {"radiation": PATH, "score": PATH}}}}}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: The radiation scorecard's latitude bands (south edge, north edge).
BANDS = ((-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90))
LAPSE_K_PER_M = 0.0065
REGIONS = ("global", "land", "ocean", "90S-60S", "60S-30S", "30S-0", "0-30N", "30N-60N", "60N-90N")


def band_name(lo: float, hi: float) -> str:
    def f(v):
        return "0" if v == 0 else (f"{abs(v)}S" if v < 0 else f"{v}N")
    return f"{f(lo)}-{f(hi)}"


# --------------------------------------------------------------------------
# CERES
# --------------------------------------------------------------------------


def ceres_month(folder: str | Path, month: int) -> dict:
    from netCDF4 import Dataset, num2date

    folder = Path(folder)
    fields, meta = {}, {}
    lat = lon = None
    for var in ("rsut", "rsdt", "rsutcs", "rlut"):
        paths = sorted(folder.glob(f"{var}_*.nc"))
        if not paths:
            continue
        with Dataset(paths[-1]) as nc:
            t = nc.variables["time"]
            dates = num2date(t[:], t.units, getattr(t, "calendar", "standard"))
            pick = [i for i, d in enumerate(dates) if d.month == month]
            years = sorted({dates[i].year for i in pick})
            fields[var] = np.asarray(nc.variables[var][pick, :, :], dtype=np.float64)
            lat = np.asarray(nc.variables["lat"][:], np.float64)
            lon = np.asarray(nc.variables["lon"][:], np.float64)
            meta[var] = {"file": paths[-1].name, "years": [years[0], years[-1]], "n_months": len(pick)}
    if "rsut" not in fields or "rsdt" not in fields:
        raise SystemExit(f"{folder}: needs rsut_*.nc and rsdt_*.nc")
    return ceres_reference(fields, lat, lon, meta, month)


def ceres_reference(fields: dict, lat: np.ndarray, lon: np.ndarray, meta: dict, month: int) -> dict:
    """Area-weighted (cos latitude) climatology of the picked months."""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    w = w / w.sum()
    clim = {k: v.mean(axis=0) for k, v in fields.items()}
    yearly = {k: (v * w[None]).sum(axis=(1, 2)) for k, v in fields.items()}
    out = {"month": month, "meta": meta,
           "global": {k: float((clim[k] * w).sum()) for k in clim}, "bands": {}}
    g = out["global"]
    g["albedo"] = g["rsut"] / g["rsdt"]
    alb_years = yearly["rsut"] / yearly["rsdt"]
    out["global_interannual_std"] = {"albedo": float(np.std(alb_years, ddof=1)) if alb_years.size > 1 else None}
    if "rlut" in clim:
        g["net"] = g["rsdt"] - g["rsut"] - g["rlut"]
    lat2 = lat[:, None] * np.ones((1, lon.size))
    for lo, hi in BANDS:
        rows = (lat2 >= lo) & (lat2 < hi) if hi < 90 else (lat2 >= lo)
        s = float((w * clim["rsdt"] * rows).sum())
        area = float((w * rows).sum())
        entry = {"rsdt": s / area, "rsut": float((w * clim["rsut"] * rows).sum()) / area,
                 "albedo": float((w * clim["rsut"] * rows).sum()) / s if s > 0 else None}
        if "rlut" in clim:
            entry["rlut"] = float((w * clim["rlut"] * rows).sum()) / area
        out["bands"][band_name(lo, hi)] = entry
    return out


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------


def gaussian_weights(nlat: int, nlon: int) -> np.ndarray:
    _, w = np.polynomial.legendre.leggauss(nlat)
    return np.repeat(w[:, None], nlon, axis=1)


def lapse_adjust(t_ref_k, z_ref_m, z_model_m, lapse=LAPSE_K_PER_M):
    """The reference 2 m temperature carried to the model's terrain."""
    return np.asarray(t_ref_k) + lapse * (np.asarray(z_ref_m) - np.asarray(z_model_m))


def masked_stats(diff: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict:
    m = mask & np.isfinite(diff)
    ww = weights * m
    s = ww.sum()
    if s <= 0:
        return {"rmse": None, "bias": None, "n": 0}
    d = np.where(m, diff, 0.0)
    return {"rmse": float(np.sqrt((ww * d * d).sum() / s)), "bias": float((ww * d).sum() / s), "n": int(m.sum())}


def t2_regions(lat_deg: np.ndarray, land: np.ndarray) -> dict:
    lat2 = np.repeat(np.asarray(lat_deg)[:, None], land.shape[1], axis=1)
    return {"global": np.ones_like(land, bool), "land": land, "ocean": ~land,
            "nh_extratropics_land": land & (lat2 >= 20.0), "tropics_land": land & (np.abs(lat2) < 20.0),
            "sh_extratropics_land": land & (lat2 <= -20.0)}


def precip_window(run_dir: Path, rows: list[dict], t0_h: float, t1_h: float) -> dict:
    pick = {r["lead_h"]: r for r in rows}
    if t0_h not in pick or t1_h not in pick:
        return {"status": "INCOMPLETE", "reason": f"no checkpoint at {t0_h} or {t1_h} h"}

    def load(path):
        with np.load(path, allow_pickle=False) as a:
            return {k: np.array(a[k]) for k in ("physics__rainnc", "physics__rainc", "surface__land_fraction")}

    a, b = load(pick[t0_h]["path"]), load(pick[t1_h]["path"])
    w = gaussian_weights(*a["physics__rainnc"].shape)
    land = b["surface__land_fraction"] > 0.5
    scale = 24.0 / (t1_h - t0_h)
    grid = b["physics__rainnc"] - a["physics__rainnc"]
    conv = b["physics__rainc"] - a["physics__rainc"]
    out = {"hours": [t0_h, t1_h]}
    for name, m in (("global", np.ones_like(land)), ("land", land), ("ocean", ~land)):
        mean = lambda f: float((f * w * m).sum() / (w * m).sum())  # noqa: E731
        out[name] = {"total_mm_day": mean(grid + conv) * scale, "grid_scale_mm_day": mean(grid) * scale,
                     "cumulus_mm_day": mean(conv) * scale}
    return out


def score_run(run_dir: Path, analyses: list[str], cache_dir: Path) -> dict:
    from arwen_global import forecast_scorecard as fs
    from arwen_global.analysis_initial import resolve_analysis_mapping
    from arwen_global.obs_scorecard import (_phi_for, _regrid_to, model_surface_fields,
                                            product_surface_fields)
    from arwen_global.upper_air_scorecard import (ModelReader, decode_reference,
                                                  reference_fields, score_pair)

    mapping = resolve_analysis_mapping(fs.REFERENCE_FAMILIES["gfs"].mapping)
    frames = {}
    for path in analyses:
        frame, decode = decode_reference(mapping, path)
        frames[fs._stamp(frame.valid_time)] = (path, frame, decode)
    receipt, rows = fs.run_checkpoints(run_dir)
    phi, _prov, _m = _phi_for(run_dir, None, cache_dir, None)
    reader = ModelReader(receipt, phi)
    grid = reader.transform.grid
    result = {"run_dir": str(run_dir), "config_hash": receipt.get("config_hash"),
              "references": {k: v[0] for k, v in frames.items()}, "leads": {}}
    for row in rows:
        stamp = fs._stamp(row["valid"])
        if stamp not in frames or row["lead_h"] <= 0:
            continue
        path, frame, decode = frames[stamp]
        plf = reader.sample(row["path"])
        ref = reference_fields(frame, grid, path=str(path), decode=decode)
        entry = {"valid": stamp, "reference": path, "scores": score_pair(plf, ref)}
        model = model_surface_fields(run_dir, row["path"], surface_geopotential=phi)
        prod = product_surface_fields(frame, path=str(path))
        lat, lon = model.latitude_deg, model.longitude_deg
        t_ref = _regrid_to(prod, "t2_k", lat, lon)
        z_ref = _regrid_to(prod, "terrain_m", lat, lon)
        diff = model.t2_k - lapse_adjust(t_ref, z_ref, model.terrain_m)
        w = gaussian_weights(lat.size, lon.size)
        entry["t2m"] = {name: masked_stats(diff, w, m) for name, m in t2_regions(lat, model.land).items()}
        entry["t2m_method"] = ("model physics__t2 minus the analysis 2 m temperature, bilinear to the model "
                               "grid, carried to the model terrain at 6.5 K/km; Gaussian-weighted; land is "
                               "the model's land_fraction >= 0.5")
        result["leads"][f"{row['lead_h']:g}"] = entry
    result["precip_24_48h"] = precip_window(run_dir, rows, 24.0, 48.0)
    return result


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------


def _flux(rad: dict, name: str, region: str):
    return rad["window"]["fluxes"][name][region]


def arm_row(rad: dict, score: dict | None) -> dict:
    row = {}
    for region in REGIONS:
        up, dn = _flux(rad, "sw_up_top", region), _flux(rad, "sw_down_top", region)
        row[f"albedo_{region}"] = up / dn if dn else None
    row["rsut_global"] = _flux(rad, "sw_up_top", "global")
    row["rsdt_global"] = _flux(rad, "sw_down_top", "global")
    row["olr_global"] = _flux(rad, "lw_up_top", "global")
    row["net_toa_global"] = _flux(rad, "net_top", "global")
    row["cloud_cover_global"] = _flux(rad, "total_cloud_cover", "global")
    row["sw_net_surface_global"] = _flux(rad, "sw_net_surface", "global")
    t0, t1 = (h * 3600.0 for h in rad["window"]["hours"])
    keep = [c for c in rad["checkpoints"] if t0 - 1 <= c["time_s"] <= t1 + 1 and c.get("cloud_optics")]
    if keep:
        row["lwp_g_m2"] = float(np.mean([c["cloud_optics"]["grid_mean_liquid_water_path_g_m2"] for c in keep]))
        row["iwp_snow_g_m2"] = float(np.mean([c["cloud_optics"]["grid_mean_ice_plus_snow_water_path_g_m2"] for c in keep]))
    if score:
        pr = score.get("precip_24_48h", {})
        row["precip_mm_day"] = pr.get("global", {}).get("total_mm_day")
        for lead, block in score["leads"].items():
            for target in ("z500", "t850", "rh700", "w250"):
                regions = block["scores"][target]["regions"]
                # Scalars carry rmse; winds carry the vector rmse (rmsve).
                row[f"{target}_rmse_{lead}h"] = regions["global"].get("rmse", regions["global"].get("rmsve"))
                for r in ("nh_extratropics", "sh_extratropics", "tropics"):
                    row[f"{target}_rmse_{lead}h_{r}"] = regions[r].get("rmse", regions[r].get("rmsve"))
            row[f"t2m_rmse_{lead}h_land"] = block["t2m"]["land"]["rmse"]
            row[f"t2m_bias_{lead}h_land"] = block["t2m"]["land"]["bias"]
            row[f"t2m_rmse_{lead}h_global"] = block["t2m"]["global"]["rmse"]
    return row


def build_table(manifest: dict) -> dict:
    out = {"cases": {}}
    for case, spec in manifest["cases"].items():
        ceres = json.loads(Path(spec["ceres"]).read_text())
        arms = {}
        for arm, files in spec["arms"].items():
            rad = json.loads(Path(files["radiation"]).read_text())
            score = json.loads(Path(files["score"]).read_text()) if files.get("score") else None
            arms[arm] = arm_row(rad, score)
        control = arms[spec["control"]]
        for arm, row in arms.items():
            row["albedo_minus_ceres_global"] = row["albedo_global"] - ceres["global"]["albedo"]
            row["excess_rsut_vs_ceres_w_m2"] = row["albedo_global"] * row["rsdt_global"] - ceres["global"]["albedo"] * row["rsdt_global"]
            band_err = [row[f"albedo_{b}"] - ceres["bands"][b]["albedo"] for b in ceres["bands"]
                        if row.get(f"albedo_{b}") is not None and ceres["bands"][b]["albedo"] is not None]
            row["band_albedo_abs_error_mean"] = float(np.mean(np.abs(band_err)))
            row["delta"] = {k: (v - control[k]) for k, v in row.items()
                            if isinstance(v, float) and isinstance(control.get(k), float)}
        out["cases"][case] = {"ceres": ceres, "control": spec["control"], "arms": arms}
    out["summary"] = summarize(out["cases"])
    return out


#: Rows the cross-case summary reads, with the sense in which a rise is worse.
SUMMARY_KEYS = (("albedo_minus_ceres_global", None), ("band_albedo_abs_error_mean", "up"),
                ("rsut_global", None), ("olr_global", None), ("net_toa_global", None),
                ("cloud_cover_global", None), ("lwp_g_m2", None), ("precip_mm_day", None),
                ("z500_rmse_24h", "up"), ("z500_rmse_48h", "up"), ("t850_rmse_24h", "up"),
                ("t850_rmse_48h", "up"), ("w250_rmse_48h", "up"), ("rh700_rmse_48h", "up"),
                ("t2m_rmse_24h_land", "up"), ("t2m_rmse_48h_land", "up"),
                ("t2m_rmse_24h_global", "up"), ("t2m_rmse_48h_global", "up"))


def summarize(cases: dict) -> dict:
    """Per arm, over the cases that ran it: the mean of each row's value and
    of its change from the control, and for error rows the number of cases
    in which the arm was worse than the control."""
    arms = []
    for block in cases.values():
        arms += [a for a in block["arms"] if a not in arms]
    out = {}
    for arm in arms:
        rows = [block["arms"][arm] for block in cases.values() if arm in block["arms"]]
        entry = {"cases": [c for c, b in cases.items() if arm in b["arms"]]}
        for key, worse in SUMMARY_KEYS:
            vals = [r[key] for r in rows if isinstance(r.get(key), float)]
            deltas = [r["delta"][key] for r in rows if key in r.get("delta", {})]
            if not vals:
                continue
            entry[key] = {"mean": float(np.mean(vals)), "delta_mean": float(np.mean(deltas)) if deltas else None,
                          "deltas": deltas}
            if worse == "up":
                entry[key]["cases_worse"] = int(sum(d > 0 for d in deltas))
        out[arm] = entry
    return out


def markdown(table: dict) -> str:
    lines = []
    cols = [("albedo_global", "albedo", 4), ("albedo_minus_ceres_global", "alb-CERES", 4),
            ("band_albedo_abs_error_mean", "band |err|", 4), ("rsut_global", "RSUT", 2),
            ("olr_global", "OLR", 2), ("net_toa_global", "net TOA", 2),
            ("cloud_cover_global", "cover", 4), ("lwp_g_m2", "LWP", 1), ("precip_mm_day", "precip", 3),
            ("z500_rmse_24h", "Z500 24h", 3), ("z500_rmse_48h", "Z500 48h", 3),
            ("t850_rmse_24h", "T850 24h", 4), ("t850_rmse_48h", "T850 48h", 4),
            ("t2m_rmse_24h_land", "T2m land 24h", 3), ("t2m_rmse_48h_land", "T2m land 48h", 3)]
    for case, block in table["cases"].items():
        c = block["ceres"]["global"]
        lines.append(f"### {case} (CERES month {block['ceres']['month']}: albedo {c['albedo']:.4f})")
        lines.append("")
        lines.append("| arm | " + " | ".join(h for _, h, _ in cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for arm, row in block["arms"].items():
            cells = []
            for key, _h, nd in cols:
                v = row.get(key)
                cells.append("" if v is None else f"{v:.{nd}f}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    summary = table.get("summary") or {}
    if summary:
        keys = [("albedo_minus_ceres_global", "alb-CERES", 4), ("rsut_global", "dRSUT", 2),
                ("olr_global", "dOLR", 2), ("net_toa_global", "dnet", 2), ("lwp_g_m2", "dLWP", 1),
                ("z500_rmse_24h", "dZ500 24h", 3), ("z500_rmse_48h", "dZ500 48h", 3),
                ("t850_rmse_48h", "dT850 48h", 4), ("w250_rmse_48h", "dW250 48h", 3),
                ("t2m_rmse_24h_land", "dT2m land 24h", 3), ("t2m_rmse_48h_land", "dT2m land 48h", 3)]
        lines.append("### Mean over cases (change from control; worse/cases for errors)")
        lines.append("")
        lines.append("| arm | cases | " + " | ".join(h for _, h, _ in keys) + " |")
        lines.append("|---" * (len(keys) + 2) + "|")
        for arm, entry in summary.items():
            cells = []
            for key, _h, nd in keys:
                e = entry.get(key)
                if e is None:
                    cells.append("")
                    continue
                v = e["mean"] if key == "albedo_minus_ceres_global" else e["delta_mean"]
                text = "" if v is None else f"{v:+.{nd}f}"
                if "cases_worse" in e:
                    text += f" ({e['cases_worse']}/{len(e['deltas'])})"
                cells.append(text)
            lines.append(f"| {arm} | {len(entry['cases'])} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ceres")
    c.add_argument("--dir", required=True)
    c.add_argument("--month", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--analysis", action="append", required=True)
    s.add_argument("--cache-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("table")
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--out-json", type=Path, required=True)
    t.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "ceres":
        ref = ceres_month(args.dir, args.month)
        args.out.write_text(json.dumps(ref, indent=1))
        print(json.dumps(ref["global"], indent=1))
    elif args.cmd == "score":
        res = score_run(args.run_dir, args.analysis, args.cache_dir)
        args.out.write_text(json.dumps(res, indent=1, default=float))
        for lead, block in res["leads"].items():
            print(lead, {k: round(v["regions"]["global"].get("rmse", float("nan")), 4)
                         for k, v in block["scores"].items() if "rmse" in v["regions"]["global"]},
                  "t2m land", block["t2m"]["land"])
        print("precip", res["precip_24_48h"].get("global"))
    else:
        table = build_table(json.loads(args.manifest.read_text()))
        args.out_json.write_text(json.dumps(table, indent=1))
        md = markdown(table)
        args.out_md.write_text(md)
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
