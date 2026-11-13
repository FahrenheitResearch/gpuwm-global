"""Print the long-run gate table (LR-0 to LR-10) of an Arwen Global run.

::

    python tools/arwen_global_longrun_report.py RUN_DIR [RUN_DIR ...]
        [--reference-run REF_DIR [REF_DIR ...] | --reference-table REF.jsonl]
        [--ltm air.day.ltm.1991-2020.nc] [--start-utc 2026-08-01T00:00:00Z]
        [--restart-pair STRAIGHT.npz SEGMENTED.npz] [--window-days 30]
        [--json report.json]
    python tools/arwen_global_longrun_report.py analysis-table GRIB [GRIB ...]
        --out REF.jsonl [--mapping gdas-global]

RUN_DIR is a run's output folder, or several segment folders in run order;
each holds the ``diagnostics.jsonl`` the runner appends at every output
interval.  The rows are read, a restart's repeated opening row dropped, and
:func:`arwen_global.longrun_gates.evaluate` judges every 30-day window and
the whole run.  Nothing here runs the model or touches a card.

The temperature gates (LR-5, LR-6) are GATED against one reference:

* ``--reference-run``: another run's folders (arm B against arm A); its
  rows are matched by model time;
* ``--reference-table``: a JSON-lines table of ``{"valid_utc": ...,
  "<row name>": value}`` records, e.g. the same global and polar-cap means
  taken from the analyses of the run's dates.  ``analysis-table`` writes
  one from isobaric analysis GRIBs (the GDAS files an observed-boundary
  arm's series already fetched), decoded through the engine with the
  mapping the cold start uses, each level's mean an area mean over the
  analysis grid's cells; below the ground the analysis carries its
  producer's extrapolation.

``--ltm`` names a daily long-term-mean air temperature file in the NOAA
PSL layout (``air(time=365, level, lat, lon)``, levels in hPa, a regular
latitude-longitude grid; the NCEP/NCAR reanalysis 1991-2020 file is one).
It is a CLIMATOLOGY: the bias and trend against it are reported beside
the gate, never gated.  Its global means are area means over its cells
(a cell's weight is the sine difference of its latitude edges, the polar
half-cells included); 29 February reads 28 February; a row's valid date is
the run's start (the receipt's ``forecast_clock``, or ``--start-utc``)
plus its model time.  Below the ground the file carries the reanalysis's
own extrapolation and the model rows the standard lapse rate, so the 850
hPa bias over high terrain is a comparison of two extrapolations.

``--restart-pair`` is LR-0: the terminal checkpoint of a straight run and
of the same run in segments, compared by their self hashes and byte for
byte.

Exit status: 0 when no required gate failed, 1 when one did (a gate that
was not measured is not a failure; the table says which were not).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from arwen_global import longrun_gates as lg

RECEIPT_NAME = "arwen-global-receipt.json"


class DailyLongTermMean:
    """A daily climatology of air temperature on pressure levels.

    ``global_mean(level_hpa, when)`` and ``cap_mean(level_hpa, hemisphere,
    when)`` return the file's area mean at the date, linear in time between
    the daily means (each taken at its day's noon), periodic over the
    365-day year.
    """

    def __init__(self, path, levels=lg.PRESSURE_LEVELS_HPA):
        from netCDF4 import Dataset

        self.label = Path(path).name
        with Dataset(str(path)) as data:
            data.set_auto_maskandscale(True)
            file_levels = np.asarray(data.variables["level"][:], dtype=np.float64)
            lat = np.asarray(data.variables["lat"][:], dtype=np.float64)
            air = data.variables["air"]
            if air.shape[0] != 365:
                raise ValueError(
                    f"{path}: expected 365 daily means, found {air.shape[0]}")
            self._global: dict[int, np.ndarray] = {}
            self._caps: dict[tuple[int, str], np.ndarray] = {}
            weights = self.cell_weights(lat)
            for level in levels:
                index = np.flatnonzero(np.isclose(file_levels, float(level)))
                if index.size != 1:
                    raise ValueError(f"{path}: no {level} hPa level")
                field = np.ma.filled(air[:, int(index[0]), :, :].astype(np.float64),
                                     np.nan)
                if not np.isfinite(field).all():
                    raise ValueError(f"{path}: missing values at {level} hPa")
                zonal = field.mean(axis=-1)                      # (365, nlat)
                self._global[int(level)] = zonal @ weights / weights.sum()
                for name, rows in (("north", lat >= lg.POLAR_CAP_EDGE_DEG),
                                   ("south", lat <= -lg.POLAR_CAP_EDGE_DEG)):
                    w = np.where(rows, weights, 0.0)
                    self._caps[(int(level), name)] = zonal @ w / w.sum()

    @staticmethod
    def cell_weights(lat_deg: np.ndarray) -> np.ndarray:
        """Area weight of each latitude row of a regular grid whose rows
        sit on cell centres, with half-cells at the poles."""
        lat = np.asarray(lat_deg, dtype=np.float64)
        step = abs(float(lat[1] - lat[0]))
        upper = np.minimum(lat + step / 2.0, 90.0)
        lower = np.maximum(lat - step / 2.0, -90.0)
        return np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower))

    @staticmethod
    def day_position(when: dt.datetime) -> float:
        """Days since 1 January 00 UTC on a 365-day year (29 February
        reads 28 February), minus half a day: the position on the axis of
        daily means taken at noon."""
        start = dt.datetime(when.year, 1, 1, tzinfo=when.tzinfo)
        days = (when - start).total_seconds() / 86400.0
        leap = when.year % 4 == 0 and (when.year % 100 != 0 or when.year % 400 == 0)
        if leap and days >= 59.0:
            days = days - 1.0 if days >= 60.0 else 58.0 + (days - 59.0)
        return days - 0.5

    @staticmethod
    def _at(series: np.ndarray, when: dt.datetime) -> float:
        x = DailyLongTermMean.day_position(when) % 365.0
        lo = int(math.floor(x)) % 365
        hi = (lo + 1) % 365
        frac = x - math.floor(x)
        return float((1.0 - frac) * series[lo] + frac * series[hi])

    def global_mean(self, level_hpa: int, when: dt.datetime) -> float:
        return self._at(self._global[int(level_hpa)], when)

    def cap_mean(self, level_hpa: int, hemisphere: str, when: dt.datetime) -> float:
        return self._at(self._caps[(int(level_hpa), hemisphere)], when)

    def value(self, name: str, time_s: float, when: dt.datetime | None):
        """The reference protocol of :func:`arwen_global.longrun_gates.evaluate`."""
        if when is None:
            return None
        for level in lg.PRESSURE_LEVELS_HPA:
            if name == lg.level_row_name(level):
                return self.global_mean(level, when)
            for hemisphere in ("north", "south"):
                if (name == lg.cap_row_name(level, hemisphere)
                        and (int(level), hemisphere) in self._caps):
                    return self.cap_mean(level, hemisphere, when)
        return None


def regular_grid_rows(field, lat_deg):
    """Global and polar-cap means of a regular latitude-longitude field
    (..., nlat, nlon), each latitude row weighted by its cell area."""
    weights = DailyLongTermMean.cell_weights(lat_deg)
    zonal = np.asarray(field, dtype=np.float64).mean(axis=-1)
    lat = np.asarray(lat_deg, dtype=np.float64)
    caps = {}
    for name, rows in (("north", lat >= lg.POLAR_CAP_EDGE_DEG),
                       ("south", lat <= -lg.POLAR_CAP_EDGE_DEG)):
        w = np.where(rows, weights, 0.0)
        caps[name] = zonal @ w / w.sum()
    return zonal @ weights / weights.sum(), caps


def analysis_record(grib, mapping="gdas-global") -> dict[str, object]:
    """One table record from one isobaric analysis GRIB: its valid time and
    the global-mean and polar-cap temperature rows at the gate levels,
    decoded through the engine with the mapping the cold start uses.
    Below the ground the analysis carries its producer's own
    extrapolation."""
    from arwen_global.analysis_initial import resolve_analysis_mapping
    from arwen_global.mapped_source_compat import decode_through_engine

    decoded = decode_through_engine(resolve_analysis_mapping(mapping), [Path(grib)])
    (frame,) = decoded.frames
    levels_hpa = np.asarray(frame.vertical_values, dtype=np.float64) / 100.0
    lat = np.asarray(frame.latitude, dtype=np.float64)
    if lat.ndim == 2:
        lat = lat[:, 0]
    temperature = np.asarray(frame.fields["air_temperature"].values, dtype=np.float64)
    when = frame.valid_time
    if isinstance(when, dt.datetime):
        when = (when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc))
        when = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    record: dict[str, object] = {"valid_utc": str(when), "source": Path(grib).name}
    for level in lg.PRESSURE_LEVELS_HPA:
        index = np.flatnonzero(np.isclose(levels_hpa, float(level)))
        if index.size != 1:
            continue
        mean, caps = regular_grid_rows(temperature[int(index[0])], lat)
        record[lg.level_row_name(level)] = float(mean)
        if level in lg.POLAR_CAP_LEVELS_HPA:
            for hemisphere, value in caps.items():
                record[lg.cap_row_name(level, hemisphere)] = float(value)
    return record


def parse_utc(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def start_from_receipts(folders) -> dt.datetime | None:
    for folder in folders:
        path = Path(folder) / RECEIPT_NAME
        if not path.is_file():
            continue
        clock = json.loads(path.read_text(encoding="utf-8")).get("forecast_clock") or {}
        if clock.get("start_utc"):
            return parse_utc(clock["start_utc"])
    return None


def _fmt(value, spec=".4g"):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _fit_line(label, level, fit):
    if fit is None:
        return f"{label} {level}: not fitted (too few days past day 30)"
    return (f"{label} {level}: slope {fit['slope_per_30d']:+.4f} "
            f"+- {_fmt(fit['ci95_half_width_per_30d'])} K/30 d over "
            f"{fit['fit_days']:.1f} d (limit {_fmt(fit['limit_per_30d'])}, "
            f"n {fit['samples']}, n_eff {fit['effective_samples']:.1f}), "
            f"mean bias {fit['mean_bias_k']:+.3f} K")


def summary_lines(report: dict) -> list[str]:
    gates = report["gates"]
    lr5 = gates["LR-5"]
    lines = [
        f"rows {report['rows']} (steps {report['first_step']}-{report['last_step']}), "
        f"{report['days']:.3f} days, windows of {report['window_days']:g} days, "
        f"start {report['start_utc'] or '-'}",
        f"gating reference {lr5['reference'] or '-'}; climatology {lr5['climatology'] or '-'}",
        f"mass_relative_drift {_fmt(gates['LR-1']['mass_relative_drift'], '.3e')}  "
        f"total_water_relative_drift "
        f"{_fmt(gates['LR-1']['total_water_relative_drift'], '.3e')}",
        "",
        f"{'gate':<6} {'name':<18} {'kind':<9} {'verdict':<13} threshold",
    ]
    for key, gate in gates.items():
        lines.append(f"{key:<6} {gate['name']:<18} {gate['kind']:<9} "
                     f"{gate['verdict']:<13} {gate['threshold']}")
    lines.append("")
    lines.append(f"LR-1 dry-mass range over the run: "
                 f"{_fmt(gates['LR-1']['run_range_pa'])} Pa")
    lr2 = gates["LR-2"]
    lines.append(f"LR-2 land reservoir minimum: {_fmt(lr2.get('land_minimum_kg_m2'))} kg/m2; "
                 f"ledger: {_fmt(lr2.get('ledger_maximum_relative', lr2.get('ledger')))}; "
                 f"largest 24 h water-column evaporation "
                 f"{_fmt(lr2.get('maximum_water_column_evaporation_24h_mm_day'))} mm/day")
    for w in gates["LR-3"]["windows"]:
        lines.append(f"LR-3 days {w['start_day']:.1f}-{w['end_day']:.1f}: "
                     f"P {_fmt(w['precipitation_mm_day'])} mm/day, "
                     f"E-P {_fmt(w['evaporation_minus_precipitation_mm_day'])} mm/day, "
                     f"E {_fmt(w['evaporation_mm_day'])} mm/day, "
                     f"PW {_fmt(w['precipitable_water_kg_m2'])} kg/m2")
    for w in gates["LR-4"]["windows"]:
        lines.append(
            f"LR-4 days {w['start_day']:.1f}-{w['end_day']:.1f}: "
            f"TOA {_fmt(w.get('toa_net_w_m2'))} sfc-rad {_fmt(w.get('surface_net_radiation_w_m2'))} "
            f"H {_fmt(w.get('sensible_heat_flux_w_m2'))} LE {_fmt(w.get('latent_heat_flux_w_m2'))} "
            f"LfS {_fmt(w.get('frozen_precipitation_latent_w_m2'))} "
            f"dE/dt {_fmt(w.get('energy_tendency_w_m2'))} "
            f"residual {_fmt(w.get('residual_w_m2'))} W/m2")
    for level, fit in (lr5.get("slopes") or {}).items():
        lines.append(_fit_line("LR-5", level, fit))
    for level, fit in (lr5.get("climatology_slopes") or {}).items():
        lines.append(_fit_line("LR-5 climatology (reported)", level, fit))
    lr6 = gates["LR-6"]
    lines.append(f"LR-6 minimum T {_fmt(lr6['minimum_temperature_k'])} K")
    for label, biases in (("LR-6 cap bias", lr6.get("cap_bias")),
                          ("LR-6 climatology cap bias (reported)",
                           lr6.get("climatology_cap_bias"))):
        for name, series in (biases or {}).items():
            lines.append(f"{label} {name}: " + ", ".join(
                f"days {w['start_day']:.0f}-{w['end_day']:.0f} "
                f"{_fmt(w['mean_bias_k'], '+.2f')} K"
                for w in series))
    for w in lr6["windows"]:
        lines.append(f"LR-6 days {w['start_day']:.1f}-{w['end_day']:.1f}: "
                     f"u10 60N {_fmt(w['zonal_mean_u_10hpa_60n_m_s'])} "
                     f"60S {_fmt(w['zonal_mean_u_10hpa_60s_m_s'])} m/s")
    for w in gates["LR-7"]["windows"]:
        lines.append(f"LR-7 days {w['start_day']:.1f}-{w['end_day']:.1f}: "
                     f"soil water {_fmt(w['soil_water_change_kg_m2_per_30d'])} kg/m2 per 30 d, "
                     f"NH snow {_fmt(w['nh_snow_covered_area_km2'])} km2")
    for w in gates["LR-9"]["windows"]:
        lines.append(f"LR-9 days {w['start_day']:.1f}-{w['end_day']:.1f}: "
                     f"TOA net {_fmt(w['toa_net_w_m2'])} "
                     f"OLR {_fmt(w['outgoing_longwave_w_m2'])} W/m2")
    lr10 = gates.get("LR-10") or {}
    if lr10.get("verdict") == "report":
        lines.append(
            f"LR-10 axial angular momentum {_fmt(lr10['axial_mean_kg_m2_s'])} kg m2/s, "
            f"drift {_fmt(lr10['axial_drift_kg_m2_s_per_day'])} per day "
            f"({_fmt(lr10['axial_relative_drift_per_day'], '.3e')} relative); "
            f"relative part {_fmt(lr10['relative_mean_kg_m2_s'])}, drift "
            f"{_fmt(lr10['relative_drift_kg_m2_s_per_day'])} per day")
    for key, gate in gates.items():
        if gate.get("note"):
            lines.append(f"{key}: {gate['note']}")
    lines.append(report["note"])
    return lines


def analysis_table_main(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="arwen_global_longrun_report.py analysis-table",
        description="Write the gating reference table from isobaric analyses.")
    parser.add_argument("gribs", nargs="+", help="one analysis GRIB per valid time")
    parser.add_argument("--mapping", default="gdas-global",
                        help="the analysis mapping, as [initial] analysis_mapping names it")
    parser.add_argument("--out", required=True, help="the JSON-lines table to write")
    args = parser.parse_args(argv)
    with open(args.out, "w", encoding="utf-8", newline="\n") as stream:
        for grib in args.gribs:
            record = analysis_record(grib, args.mapping)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            print(f"{record['valid_utc']} {record['source']}: " + ", ".join(
                f"{level} hPa {record.get(lg.level_row_name(level), float('nan')):.3f} K"
                for level in lg.PRESSURE_LEVELS_HPA))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "analysis-table":
        return analysis_table_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run folder(s) in run order")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reference-run", nargs="+", default=None,
                        help="the gating reference: another run's folder(s)")
    source.add_argument("--reference-table", default=None,
                        help="the gating reference: a JSON-lines table by valid_utc")
    parser.add_argument("--ltm", default=None,
                        help="a daily long-term-mean air temperature netCDF (reported)")
    parser.add_argument("--start-utc", default=None,
                        help="model time zero, when no receipt carries it")
    parser.add_argument("--restart-pair", nargs=2, default=None,
                        metavar=("STRAIGHT", "SEGMENTED"))
    parser.add_argument("--window-days", type=float, default=lg.WINDOW_DAYS)
    parser.add_argument("--json", default=None, help="write the table here")
    args = parser.parse_args(argv)

    rows = lg.read_rows(args.runs)
    start = parse_utc(args.start_utc) if args.start_utc else start_from_receipts(args.runs)
    reference = None
    if args.reference_run:
        reference = lg.RunReference(lg.read_rows(args.reference_run),
                                    label="run " + " ".join(args.reference_run))
    elif args.reference_table:
        text = Path(args.reference_table).read_text(encoding="utf-8")
        reference = lg.TableReference(
            [json.loads(line) for line in text.splitlines() if line.strip()],
            label=Path(args.reference_table).name)
    climatology = DailyLongTermMean(args.ltm) if args.ltm else None
    if (climatology is not None or args.reference_table) and start is None:
        parser.error("a reference by date needs the run's start: no receipt "
                     "carries it; give --start-utc")
    restart = (lg.restart_comparison(*args.restart_pair)
               if args.restart_pair else None)
    report = lg.evaluate(rows, start_utc=start, reference=reference,
                         climatology=climatology, restart=restart,
                         window_days=args.window_days)
    report["runs"] = [str(Path(p)) for p in args.runs]
    for line in summary_lines(report):
        print(line)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    failed = [key for key, gate in report["gates"].items()
              if gate["kind"] == "required" and gate["verdict"] == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
