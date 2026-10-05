"""``gpuwm-global score``: one command that scores a WOOF Global forecast.

What it does
------------
Given the output directory of ``gpuwm-global run`` (any truncation: T255,
T383 and T533 read the same way, because every number below is taken on
the run's own Gaussian grid from the run's own receipt), it writes one
scorecard with two kinds of rows, each row keyed by the forecast lead:

* **Observations, the verification of record.**
  ASOS/METAR stations through the ``rw_asos`` door (``networks``,
  ``stations``, ``fetch``, ``decode``: the ``gpuwm-obs.asos-surface.v2``
  record), scored by :func:`arwen_global.obs_scorecard.
  score_surface_stations` for 2 m temperature, 2 m dewpoint, 10 m wind
  speed and mean sea level pressure.  Radiosondes through the ``rw_igra2``
  door (``fetch`` and ``table --mandatory-only``: the neutral
  ``gpuwm-obs.table.v2`` table), scored by :func:`arwen_global.
  obs_scorecard.score_soundings` for 500 hPa height, 850 and 500 hPa
  temperature and the 250 and 850 hPa vector wind.

* **Analyses, secondary references.**  The GFS analysis (the GDAS cycle's
  f000) and the IFS analysis (the ECMWF open-data 0 h field), each decoded
  through the Rust mapped-source door with its own mapping and scored on
  the run's grid by :func:`arwen_global.upper_air_scorecard.score_pair`
  (z500, t850, w250, w850, rh700; four regions).  An analysis is another
  model's best estimate, not a measurement: these rows say how far WOOF
  Global sits from the two operational centres' starting points, and they
  are labelled as references, never as skill.  Each analysis is also
  scored ALONE against the same station and sounding reports (its
  ``surface_fit`` and ``upper_air_fit``): the floor of instrument and
  representativeness error the model's observation rows sit on.  On the
  2026-10-01 00Z case the GFS analysis read a 2 m dewpoint bias of -1.49 K
  against 3,106 stations and the IFS analysis -0.01 K against 3,242, which
  is what tells a reader how much of a model's dry bias it was handed.

Every decode on the data path is Rust: the two observation doors and the
mapped-source GRIB decode.  This module orders the calls, groups the
neutral table's rows into soundings and writes the record.

The reference families are TABLE ROWS (:data:`REFERENCE_FAMILIES`): a
label, the mapping its files decode with, the engine fetch route that
brings one cycle's analysis, the file name the route lands and the cycle
hours the centre publishes an analysis at.  Adding a third centre is one
more row, never a new code path.

Radiosonde heights and the ISA stamp
------------------------------------
``rw_igra2 table`` writes every level's geopotential height in the
``elevation_m`` column.  Where the archive gave a level no height the door
stamps the ICAO standard-atmosphere height of the level's pressure and
counts it (``levels_without_height_isa_used``); that stamp is a placeholder
for the vertical position of a temperature or wind, not a measured height,
and scoring it as a 500 hPa height reads the model against the standard
atmosphere.  Measured on the 2026-10-01 to 2026-10-04 table: 78 of 2,403
500 hPa levels carry it (5574.4 m, the stamp to one decimal), against
integer heights everywhere else.  :func:`soundings_from_table` therefore
refuses the height of a level whose value sits within
:data:`ISA_STAMP_TOLERANCE_M` of the stamp and keeps its temperature and
wind; the refusals are counted in the record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ISA_STAMP_TOLERANCE_M",
    "REFERENCE_FAMILIES",
    "ReferenceFamily",
    "SCHEMA",
    "add_score_arguments",
    "isa_height_m",
    "reference_fetch_command",
    "run_checkpoints",
    "score_forecast",
    "score_main",
    "scorecard_lines",
    "soundings_from_table",
]

SCHEMA = "gpuwm.woof-global-forecast-scorecard/v1"

#: A level's height within this of the ISA height of its pressure is the
#: door's ISA stamp, not a measurement.  The stamp is written to 0.1 m, so
#: it sits at most 0.05 m from the unrounded ISA height; IGRA2 heights are
#: whole metres, so a measured height can only fall inside 0.1 m of it when
#: the ISA height itself is within 0.1 m of an integer (true of 200, 300
#: and 925 hPa, none of which is a scored height).
ISA_STAMP_TOLERANCE_M = 0.1


@dataclass(frozen=True)
class ReferenceFamily:
    """One analysis centre as table data.

    ``mapping`` is what :func:`arwen_global.analysis_initial.
    resolve_analysis_mapping` resolves (a bare source id or mapping file
    name); ``fetch_arguments`` follow ``gpuwm fetch --source <fetch_source>
    --cycle <cycle>`` and must land exactly one analysis named
    ``file_pattern`` (strftime of the cycle) in the ``--out`` directory.
    """

    label: str
    subject: str
    mapping: str
    fetch_source: str
    fetch_arguments: tuple[str, ...]
    file_pattern: str
    cycle_hours: tuple[int, ...]


REFERENCE_FAMILIES: dict[str, ReferenceFamily] = {
    "gfs": ReferenceFamily(
        label="gfs",
        subject="the GFS analysis: the GDAS cycle's f000 on the 0.25 degree grid",
        mapping="gdas-global",
        fetch_source="gdas",
        fetch_arguments=("--hours", "0", "--mode", "full-file", "--all-levels"),
        file_pattern="gdas.t%Hz.pgrb2.0p25.f000",
        cycle_hours=(0, 6, 12, 18),
    ),
    "ifs": ReferenceFamily(
        label="ifs",
        subject="the IFS analysis: the ECMWF open-data oper cycle's 0 h field on the 0.25 degree grid",
        # By file name: the engine's ecmwf-open-data row is the regional
        # forecast mapping, and the bare id answers both tables with
        # different bytes (MappingTablesDisagree); this package's global
        # row is the one that decodes the whole globe on isobaric levels.
        mapping="rw-wps-ecmwf-open-data-global-forecast-grib2.mapping.json",
        fetch_source="ecmwf-open-data",
        fetch_arguments=("--hours", "0", "--mode", "full-file"),
        file_pattern="%Y%m%d%H0000-0h-oper-fc.grib2",
        cycle_hours=(0, 12),
    ),
}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _utc(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _stamp(moment: dt.datetime) -> str:
    return _utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_checkpoints(run_dir: Path) -> tuple[dict, list[dict]]:
    """The run receipt and every checkpoint with its valid time and lead,
    from the checkpoint metadata alone (no arrays are read)."""
    from .checkpoint import read_checkpoint_header
    from .upper_air_scorecard import read_receipt, receipt_start_utc

    receipt = read_receipt(Path(run_dir))
    start = _utc(receipt_start_utc(receipt))
    rows = []
    for path in sorted(Path(run_dir).glob("arwen_global_step*.npz")):
        meta = read_checkpoint_header(path)
        valid = start + dt.timedelta(seconds=float(meta["time_s"]))
        rows.append({
            "path": path, "step": int(meta["step"]), "time_s": float(meta["time_s"]),
            "valid": valid, "lead_h": float(meta["time_s"]) / 3600.0,
        })
    if not rows:
        raise FileNotFoundError(
            f"{run_dir}: no arwen_global_step*.npz checkpoints; score reads the "
            "output directory of `gpuwm-global run`"
        )
    return receipt, rows


def _lead_text(row: dict) -> str:
    return f"{row['lead_h']:g}"


def select_checkpoints(rows: list[dict], hours: list[float] | None) -> list[dict]:
    """``hours`` given: exactly those leads, each refused by name when the
    run wrote no checkpoint there.  Not given: every lead after the start."""
    if hours is None:
        chosen = [r for r in rows if r["lead_h"] > 0.0]
        if not chosen:
            raise ValueError("the run wrote no checkpoint after its start; nothing to score")
        return chosen
    by_lead = {round(r["lead_h"], 6): r for r in rows}
    missing = [h for h in hours if round(float(h), 6) not in by_lead]
    if missing:
        # Hoisted out of the f-string: a quote of the same kind nested inside
        # an f-string is Python 3.12 syntax and 3.11 is this package's floor.
        asked = ", ".join(f"{h:g} h" for h in missing)
        written = ", ".join(_lead_text(r) for r in rows)
        raise ValueError(f"no checkpoint at lead {asked}; the run wrote {written} h")
    return [by_lead[round(float(h), 6)] for h in hours]


# --------------------------------------------------------------------------
# observations through the Rust doors
# --------------------------------------------------------------------------


def _door(name: str, arguments: list[str]) -> dict:
    from .obs_streams import _run_door

    return _run_door(name, arguments)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _stride_minutes(valid_times: list[dt.datetime]) -> int:
    """The decode stride that lands on every scored instant from the first."""
    first = valid_times[0]
    offsets = [int(round((v - first).total_seconds() / 60.0)) for v in valid_times]
    if any(abs((v - first).total_seconds() / 60.0 - o) > 1.0e-6 for v, o in zip(valid_times, offsets)):
        raise ValueError("scored valid times must fall on whole minutes for the station decode")
    stride = 0
    for o in offsets[1:]:
        stride = math.gcd(stride, o)
    return stride or 60


def asos_record(obs_dir: Path, valid_times: list[dt.datetime]) -> Path:
    """The ``gpuwm-obs.asos-surface.v2`` record covering ``valid_times``,
    through ``rw_asos``: every ASOS network the archive lists, the station
    table frozen from them, one fetch of the window and one decode at the
    scored instants."""
    out = Path(obs_dir) / "asos"
    out.mkdir(parents=True, exist_ok=True)
    first, last = valid_times[0], valid_times[-1]
    stride = _stride_minutes(valid_times)
    networks_record = _door("rw_asos", ["networks", "--out", str(out / "networks.json")])
    networks = ",".join(networks_record["networks"])
    _door("rw_asos", ["stations", "--networks", networks, "--out", str(out / "stations.json")])
    fetched = _door("rw_asos", [
        "fetch", "--networks", networks,
        "--start", _stamp(first - dt.timedelta(hours=1)),
        "--end", _stamp(last + dt.timedelta(hours=1)),
        "--out", str(out / "obs.csv"),
    ])
    _write_json(out / "fetch.json", fetched)
    decoded = _door("rw_asos", [
        "decode", "--obs", str(out / "obs.csv"), "--stations", str(out / "stations.json"),
        "--start", _stamp(first), "--end", _stamp(last), "--step-minutes", str(stride),
        "--out", str(out / "record.json"),
    ])
    _write_json(out / "decode.json", decoded)
    return out / "record.json"


def igra2_table(obs_dir: Path, valid_times: list[dt.datetime]) -> Path:
    """The radiosonde mandatory levels of the window as a neutral table,
    through ``rw_igra2`` (the year-to-date archive, every station)."""
    out = Path(obs_dir) / "igra2"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _door("rw_igra2", ["fetch", "--out", str(raw)])
    table = out / "igra2.csv"
    _door("rw_igra2", [
        "table", "--zips", str(raw), "--start", _stamp(valid_times[0]), "--end", _stamp(valid_times[-1]),
        "--mandatory-only", "--fetch-record", str(raw / "fetch.json"), "--out", str(table),
    ])
    return table


def isa_height_m(pressure_pa: float) -> float:
    """The ICAO standard-atmosphere height of a pressure, the inverse of
    :func:`arwen_global.obs_table.isa_pressure_pa` with its constants."""
    from .obs_table import (
        ISA_EXPONENT,
        ISA_LAPSE_K_M,
        ISA_SEA_LEVEL_K,
        ISA_SEA_LEVEL_PA,
        ISA_STRATOSPHERE_SCALE_M,
        ISA_TROPOPAUSE_M,
        ISA_TROPOPAUSE_PA,
    )

    p = float(pressure_pa)
    if p >= ISA_TROPOPAUSE_PA:
        return ISA_SEA_LEVEL_K / ISA_LAPSE_K_M * (1.0 - (p / ISA_SEA_LEVEL_PA) ** (1.0 / ISA_EXPONENT))
    return ISA_TROPOPAUSE_M + ISA_STRATOSPHERE_SCALE_M * math.log(ISA_TROPOPAUSE_PA / p)


def soundings_from_table(path: Path, valid: dt.datetime, *, levels_pa=None) -> tuple[list[dict], dict]:
    """The soundings filed under the nominal instant ``valid`` in a
    ``rw_igra2`` neutral table, in the record shape
    :func:`arwen_global.obs_scorecard.score_soundings` reads:
    ``{station_id, nominal, latitude, longitude, levels: {"50000": {z, t,
    u, v, wspd}}}``.  A level's height is refused when it is the door's ISA
    stamp (see the module text); counters say how many."""
    from .obs_scorecard import UPPER_LEVELS_PA
    from .obs_table import decode_neutral_csv

    levels = tuple(float(v) for v in (levels_pa or UPPER_LEVELS_PA))
    wanted = _utc(valid)
    _source, rows, counters = decode_neutral_csv(Path(path).read_text(encoding="utf-8"))
    sites: dict[str, dict] = {}
    isa_refused = 0
    for row in rows:
        nominal = row.nominal_time or row.valid_time
        if nominal is None or _utc(nominal) != wanted or row.level_pa is None:
            continue
        level = next((v for v in levels if abs(row.level_pa - v) < 0.5), None)
        if level is None:
            continue
        site = sites.setdefault(row.station_id, {
            "station_id": row.station_id, "nominal": _stamp(wanted),
            "latitude": row.latitude_deg, "longitude": row.longitude_deg,
            "levels": {}, "_height": {},
        })
        entry = site["levels"].setdefault(str(int(level)), {"z": None, "t": None, "u": None, "v": None, "wspd": None})
        if str(int(level)) not in site["_height"]:
            height = float(row.elevation_m)
            if abs(height - isa_height_m(level)) <= ISA_STAMP_TOLERANCE_M:
                isa_refused += 1
                height = None
            site["_height"][str(int(level))] = height
            entry["z"] = height
        if row.variable == "temperature_k":
            entry["t"] = float(row.value)
        elif row.variable == "wind_u_m_s":
            entry["u"] = float(row.value)
        elif row.variable == "wind_v_m_s":
            entry["v"] = float(row.value)
    records = []
    for sid in sorted(sites):
        site = sites[sid]
        site.pop("_height")
        for entry in site["levels"].values():
            if entry["u"] is not None and entry["v"] is not None:
                entry["wspd"] = math.hypot(entry["u"], entry["v"])
        records.append(site)
    return records, {
        "table": str(path), "table_counters": counters, "soundings": len(records),
        "heights_refused_as_isa_stamp": isa_refused,
        "isa_stamp_rule": f"|height - ISA height of the level| <= {ISA_STAMP_TOLERANCE_M} m",
    }


# --------------------------------------------------------------------------
# the analyses
# --------------------------------------------------------------------------


def _gpuwm_command() -> list[str]:
    """The engine's console script beside this interpreter, else this
    interpreter running the engine it imports, else a ``gpuwm`` on PATH.

    The engine this process imports comes before PATH: a ``gpuwm`` on PATH
    can belong to another interpreter that has no engine at all (measured
    on a rented box: /usr/local/bin/gpuwm runs the system Python and every
    reference fetch died with ModuleNotFoundError, so a T383 score ran with
    no analysis rows and an unscreened radiosonde set)."""
    beside = Path(sys.executable).parent / ("gpuwm.exe" if os.name == "nt" else "gpuwm")
    if beside.exists():
        return [str(beside)]
    import importlib.util

    if importlib.util.find_spec("gpuwm") is not None:
        return [sys.executable, "-m", "gpuwm"]
    found = shutil.which("gpuwm")
    if found:
        return [found]
    raise FileNotFoundError(
        "the engine's `gpuwm` command is not installed beside this Python or on PATH; "
        "score fetches the reference analyses through `gpuwm fetch` (pass --reference "
        "LABEL=FILE for analyses already on disk, or --no-fetch-references)"
    )


def reference_fetch_command(family: ReferenceFamily, cycle: dt.datetime, out_dir: Path) -> list[str]:
    """The ``gpuwm fetch`` argument list that lands ``family``'s analysis
    for ``cycle`` in ``out_dir`` (the executable is prepended by the caller)."""
    return ["fetch", "--source", family.fetch_source, "--cycle", _utc(cycle).strftime("%Y-%m-%dT%H"),
            *family.fetch_arguments, "--out", str(out_dir)]


def reference_path(family: ReferenceFamily, cycle: dt.datetime, out_dir: Path) -> Path:
    return Path(out_dir) / _utc(cycle).strftime(family.file_pattern)


#: How long one reference fetch may take before it is a named missing row.
#: THE BREAKAGE: the fetch route has no timeout of its own, so one hung
#: download held the whole score (and the observation rows of record that
#: need no reference at all) for as long as the socket stayed open.  A
#: cold 0.25-degree GDAS analysis (about 480 MB) lands in two to four
#: minutes on the rented boxes; the bar is several times that.
REFERENCE_FETCH_TIMEOUT_S = 1800.0


def fetch_reference(family: ReferenceFamily, cycle: dt.datetime, root: Path) -> Path:
    """One centre's analysis valid at ``cycle``: reused when already on
    disk under ``root``, else fetched through the engine's fetch route."""
    out_dir = Path(root) / family.label / _utc(cycle).strftime("%Y%m%d%H")
    path = reference_path(family, cycle, out_dir)
    if path.is_file():
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    command = _gpuwm_command() + reference_fetch_command(family, cycle, out_dir)
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                timeout=REFERENCE_FETCH_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{family.label} analysis for {_stamp(cycle)}: `gpuwm {' '.join(command[1:])}` "
            f"did not finish in {REFERENCE_FETCH_TIMEOUT_S:g} s"
        ) from exc
    (out_dir / "fetch.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0 or not path.is_file():
        tail = [line for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
        raise RuntimeError(
            f"{family.label} analysis for {_stamp(cycle)}: `gpuwm {' '.join(command[1:])}` "
            f"{'exited ' + str(result.returncode) if result.returncode else 'landed no ' + path.name}"
            f"{': ' + tail[-1] if tail else ''}"
        )
    return path


#: A sounding value that departs from EVERY reference analysis present at
#: its instant by more than this is a gross error and is dropped for every
#: row.  THE BREAKAGE: one sonde 173 m and 7.6 K off both analyses
#: (2026-10-02 00Z, a site whose every mandatory level disagrees) raised
#: the IFS analysis's own 500 hPa height rmse against 532 soundings from
#: 10.9 m to 13.2 m, so one bad report set a fifth of the row.  Sized on
#: that instant against the IFS analysis: 500 hPa height departures have a
#: median of 5.9 m and a 99th percentile of 41 m, 850 hPa temperature 0.37 K
#: and 3.3 K; the bars sit at about twice the 99th percentile, so they
#: remove reports no analysis believes and keep the ordinary tail.  The
#: model under test is never consulted.
SOUNDING_GROSS_DEPARTURE: dict[str, float] = {"z": 80.0, "t": 6.0, "wind": 25.0}


def screen_soundings(soundings: list[dict], references: dict) -> tuple[list[dict], dict]:
    """Drop each sounding value every reference analysis puts beyond
    :data:`SOUNDING_GROSS_DEPARTURE` (``references``: label to the
    analysis's :class:`arwen_global.obs_scorecard.LevelFields`).  Only the
    values a row scores are screened (``obs_scorecard.UPPER_TARGETS``: the
    500 hPa height, never a 250 hPa height whose ordinary departures are
    several times larger and which no row reads).  A value no analysis can
    be sampled at is kept.  Returns the screened copies and a record naming
    every value dropped and the departures that dropped it."""
    import copy

    from .obs_scorecard import UPPER_TARGETS

    scored = {(kind, float(level)) for _name, kind, level in UPPER_TARGETS}

    record: dict = {"thresholds": dict(SOUNDING_GROSS_DEPARTURE), "references": sorted(references),
                    "dropped": [], "rule": "dropped when beyond the bar from every analysis present"}
    if not references or not soundings:
        record["applied"] = False
        return soundings, record
    record["applied"] = True
    lats = np.array([s["latitude"] for s in soundings], dtype=np.float64)
    lons = np.array([s["longitude"] for s in soundings], dtype=np.float64)
    samples = {}
    for name, lf in references.items():
        sampler = lf.sampler()
        samples[name] = {(f, level): sampler.sample(lf.fields[f][level], lats, lons)
                         for f in ("z", "t", "u", "v") for level in lf.levels_pa}
    out = copy.deepcopy(soundings)
    for k, site in enumerate(out):
        for key, values in site["levels"].items():
            if values is None:
                continue
            level = float(key)
            checks = [("z", ("z",)), ("t", ("t",)), ("wind", ("u", "v"))]
            for name, members in checks:
                if (name, level) not in scored or any(values.get(m) is None for m in members):
                    continue
                departures = {}
                for ref, table in samples.items():
                    if any((m, level) not in table for m in members):
                        continue
                    analysed = [float(table[(m, level)][k]) for m in members]
                    if not all(math.isfinite(v) for v in analysed):
                        continue
                    departures[ref] = math.hypot(*[values[m] - v for m, v in zip(members, analysed)])
                if departures and all(d > SOUNDING_GROSS_DEPARTURE[name] for d in departures.values()):
                    for m in members:
                        values[m] = None
                    if name == "wind":
                        values["wspd"] = None
                    record["dropped"].append({"station_id": site["station_id"], "level_pa": level,
                                              "field": name, "departures": departures})
    return out, record


def expected_references(families, valid) -> list[str]:
    """The reference families whose cycle hours carry ``valid``: the set the
    gross-error screen's rule expects at that lead, whether or not each one
    could be fetched."""
    hour, minute = _utc(valid).hour, _utc(valid).minute
    return sorted(name for name in families
                  if name in REFERENCE_FAMILIES and minute == 0
                  and hour in REFERENCE_FAMILIES[name].cycle_hours)


def screen_lead(soundings: list[dict], fields: dict, expected) -> tuple[list[dict], dict]:
    """:func:`screen_soundings` at one lead, its record naming the
    references the rule expects there, the ones present, and ``degraded``
    when an expected one is missing.

    THE BREAKAGE THE RECORD PREVENTS: the screen can only use the analyses
    that arrived, so a failed fetch screened the model's soundings by fewer
    analyses (a value one analysis alone puts beyond the bar is then
    dropped) and the model's Z500, T and wind rows moved with fetch success
    while the scorecard said nothing.  The scores stay what the present
    analyses screen; the record and the text scorecard say when that is
    fewer than the rule expects."""
    screened, record = screen_soundings(soundings, fields)
    expected = sorted(expected)
    missing = sorted(set(expected) - set(fields))
    record["expected_references"] = expected
    record["missing_references"] = missing
    record["degraded"] = bool(missing)
    return screened, record


def floor_soundings(soundings: list[dict], fields: dict, name: str) -> list[dict]:
    """The soundings ``name``'s floor row is read on: screened by the other
    analyses present, never by ``name`` itself.  An analysis that screens
    the values it is then scored on removes exactly the values it fits
    worst, which biases its floor low by construction (one Z500 value at
    one site moved the IFS fit from 13.21 m to 10.89 m on the case of
    record).  With no other analysis present the soundings are unscreened."""
    others = {other: value for other, value in fields.items() if other != name}
    if not others:
        return soundings
    screened, _record = screen_soundings(soundings, others)
    return screened


def decode_or_fail(name: str, mapping, files: dict, stamp: str, failures: list, *,
                   progress=print, decode=None):
    """One reference analysis decoded, or a named missing row.

    A fetched file that does not decode (a download cut short, a cache
    left by an interrupted fetch) used to abort the whole scorecard; it is
    now a failure like an unfetchable analysis, and the file is renamed
    aside so the next score fetches it again instead of reusing it."""
    decode = decode or decode_reference
    path = Path(files[stamp])
    try:
        return decode(mapping, path)
    except Exception as reason:  # noqa: BLE001 - any decode failure is a missing row
        aside = path.with_name(path.name + ".undecodable")
        try:
            path.replace(aside)
        except OSError:
            aside = None
        del files[stamp]
        failures.append({"family": name, "valid_time": stamp,
                         "reason": f"{path.name} did not decode: {reason}"
                                   + (f" (moved aside to {aside.name})" if aside else "")})
        progress(f"score: {name} analysis {stamp} NOT decoded: {reason}")
        return None


def gather_references(by_family: dict, families, valid_times, root: Path, *, progress=print,
                      fetch=None) -> list[dict]:
    """Fetch each family's analysis at every scored instant on one of its
    cycle hours that ``by_family`` does not already hold, into ``by_family``.

    An analysis that cannot be fetched is a missing SECONDARY row, never a
    reason to withhold the observation rows of record: it is returned as a
    failure naming the family, the instant and the fetch route's own last
    line, printed, and written into the scorecard and its text, so a
    scorecard without an IFS row says why it has none."""
    fetch = fetch or fetch_reference
    failures = []
    for family_label in families:
        family = REFERENCE_FAMILIES[family_label]
        for valid in valid_times:
            if _utc(valid).hour not in family.cycle_hours or _utc(valid).minute != 0:
                continue
            if _stamp(valid) in by_family.get(family_label, {}):
                continue
            progress(f"score: {family_label} analysis {_stamp(valid)} through gpuwm fetch")
            try:
                by_family.setdefault(family_label, {})[_stamp(valid)] = fetch(family, valid, root)
            except (OSError, RuntimeError) as reason:
                failures.append({"family": family_label, "valid_time": _stamp(valid), "reason": str(reason)})
                progress(f"score: {family_label} analysis {_stamp(valid)} NOT fetched: {reason}")
    return failures


def reference_fit(name: str, frame, path: str, observations, seam: str, soundings: list[dict]) -> dict:
    """The analysis's own fit to the same observations, on its own grid:
    the floor the model's observation rows are read against (instrument
    error plus what a 0.25 degree field cannot represent at a point).

    Each analysis is scored alone, through the same two instruments as the
    model; each row carries its own count.  The surface station set is the
    model's own and no reference changes it.  The SOUNDING set is screened
    for gross errors by the analyses present at the lead
    (:func:`screen_lead`), so which references a lead had changes which
    sounding values the model is scored on: the lead's screen record names
    the references its rule expects, the ones it used, and ``degraded``
    when one was missing, and the text scorecard prints a degraded lead.
    ``soundings`` here are screened by the OTHER analyses only
    (:func:`floor_soundings`), so an analysis's floor row is not read on
    soundings it screened itself.  A product that lacks a field one
    instrument needs is recorded as refused, by the instrument's own
    message, and the other instrument still runs."""
    from .obs_scorecard import (
        product_level_fields,
        product_surface_fields,
        score_soundings,
        score_surface_stations,
    )

    out: dict = {}
    try:
        fields = product_surface_fields(frame, path=path)
        out["surface_fit"] = score_surface_stations({name: fields}, observations, seam)
    except (KeyError, ValueError) as refusal:
        out["surface_fit_refused"] = str(refusal)
    if soundings:
        try:
            out["upper_air_fit"] = score_soundings({name: product_level_fields(frame, path=path)}, soundings)
        except (KeyError, ValueError) as refusal:
            out["upper_air_fit_refused"] = str(refusal)
    return out


def _parse_reference(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition("=")
    if not sep or label not in REFERENCE_FAMILIES or not path:
        raise argparse.ArgumentTypeError(
            f"--reference wants LABEL=FILE with LABEL one of {', '.join(REFERENCE_FAMILIES)}; got {spec!r}"
        )
    return label, Path(path)


# --------------------------------------------------------------------------
# the scorecard
# --------------------------------------------------------------------------


def score_forecast(
    run_dir: Path, out_dir: Path, *, hours: list[float] | None = None, label: str | None = None,
    asos: Path | None = None, igra2: Path | None = None, references: list[tuple[str, Path]] | None = None,
    fetch_references: bool = True, families: tuple[str, ...] = tuple(REFERENCE_FAMILIES),
    initial_analysis: Path | None = None, progress=print,
) -> dict:
    """Score one run; see the module text.  ``asos`` and ``igra2`` name an
    existing door record and table (else the doors fetch them into
    ``out_dir/obs``); ``references`` name analysis files already on disk
    (else, with ``fetch_references``, each family in ``families`` is
    fetched for every scored instant at one of its cycle hours)."""
    from .analysis_initial import resolve_analysis_mapping
    from .obs_scorecard import (
        LevelFields,
        _phi_for,
        model_surface_fields,
        product_level_fields,
        score_soundings,
        score_surface_stations,
    )
    from .upper_air_scorecard import ModelReader, decode_reference, reference_fields, score_pair

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt, rows = run_checkpoints(run_dir)
    chosen = select_checkpoints(rows, hours)
    truncation = int(receipt["config"]["truncation"])
    label = label or f"woof-global-t{truncation}"
    valid_times = [r["valid"] for r in chosen]

    progress(f"score: {run_dir} T{truncation}, leads {', '.join(_lead_text(r) for r in chosen)} h")
    phi, phi_provenance, _mapping = _phi_for(run_dir, None, out_dir / "cache", initial_analysis)

    if asos is None:
        progress("score: ASOS through rw_asos (networks, stations, fetch, decode)")
        asos = asos_record(out_dir / "obs", valid_times)
    if igra2 is None:
        progress("score: radiosondes through rw_igra2 (fetch, table --mandatory-only)")
        igra2 = igra2_table(out_dir / "obs", valid_times)

    by_family: dict[str, dict[str, Path]] = {}
    for family_label, path in references or []:
        frame, _decode = decode_reference(resolve_analysis_mapping(REFERENCE_FAMILIES[family_label].mapping), path)
        by_family.setdefault(family_label, {})[_stamp(frame.valid_time)] = Path(path)
    reference_failures: list[dict] = []
    if fetch_references:
        reference_failures = gather_references(by_family, families, valid_times, out_dir / "references",
                                               progress=progress)

    from gpuwm.obs.sources import AsosSurfaceSource

    asos_source = AsosSurfaceSource(asos)
    reader = ModelReader(receipt, phi)
    grid = reader.transform.grid
    mappings = {name: resolve_analysis_mapping(REFERENCE_FAMILIES[name].mapping) for name in by_family}

    leads = []
    for row in chosen:
        stamp = _stamp(row["valid"])
        seam = _utc(row["valid"]).strftime("%Y-%m-%dT%H:%M:%S")
        progress(f"score: lead {row['lead_h']:g} h valid {stamp}")
        entry: dict = {"lead_h": row["lead_h"], "valid_time": stamp, "checkpoint": str(row["path"]),
                       "step": row["step"]}
        observations = asos_source.observations([seam])
        surface = score_surface_stations(
            {label: model_surface_fields(run_dir, row["path"], surface_geopotential=phi)},
            observations, seam)
        entry["surface"] = surface
        plf = reader.sample(row["path"])
        raw_soundings, sounding_counters = soundings_from_table(igra2, row["valid"])
        frames = {}
        for name, files in sorted(by_family.items()):
            if stamp in files:
                decoded = decode_or_fail(name, mappings[name], files, stamp, reference_failures,
                                         progress=progress)
                if decoded is not None:
                    frames[name] = (files[stamp], *decoded)
        level_fields = {name: product_level_fields(frame, path=str(path))
                        for name, (path, frame, _d) in frames.items()}
        expected = expected_references(
            set(families if fetch_references else ()) | set(by_family), row["valid"])
        soundings, screen = screen_lead(raw_soundings, level_fields, expected)
        sounding_counters["gross_error_screen"] = screen
        entry["screen"] = {key: screen[key] for key in
                           ("expected_references", "missing_references", "degraded")}
        if soundings:
            upper = score_soundings({label: LevelFields.from_pressure_level_fields(plf)}, soundings)
            upper["soundings_record"] = sounding_counters
            entry["upper_air"] = upper
        else:
            entry["upper_air"] = None
            entry["upper_air_absent"] = f"the radiosonde table files no sounding under {stamp}"
        entry["analyses"] = {}
        for name, (path, frame, decode) in frames.items():
            ref = reference_fields(frame, grid, path=str(path), decode=decode)
            block = {
                "subject": REFERENCE_FAMILIES[name].subject, "file": str(path),
                "mapping": str(mappings[name]), "reference": ref.source,
                "scores": score_pair(plf, ref),
            }
            block.update(reference_fit(name, frame, str(path), observations, seam,
                                       floor_soundings(raw_soundings, level_fields, name)))
            entry["analyses"][name] = block
        leads.append(entry)

    payload = {
        "schema": SCHEMA,
        "label": label,
        "run_dir": str(run_dir),
        "truncation": truncation,
        "grid": {"nlat": int(receipt["transform"]["nlat"]), "nlon": int(receipt["transform"]["nlon"])},
        "config_hash": receipt.get("config_hash"),
        "integrator": receipt.get("config", {}).get("integrator"),
        "start_time_utc": _stamp(rows[0]["valid"] - dt.timedelta(seconds=rows[0]["time_s"])),
        "surface_geopotential": phi_provenance,
        "observations": {"asos_record": str(asos), "igra2_table": str(igra2),
                         "doors": ["rw_asos", "rw_igra2"]},
        "references": {name: {"subject": REFERENCE_FAMILIES[name].subject, "mapping": str(mappings[name]),
                              "files": {k: str(v) for k, v in sorted(files.items())}}
                       for name, files in sorted(by_family.items())},
        "reading": ("observation rows are the verification of record (model minus observation); "
                    "analysis rows are model minus the named centre's analysis on the run's grid, "
                    "a distance from another model, not skill; each analysis's surface_fit and "
                    "upper_air_fit are that analysis scored alone against the same reports, the "
                    "floor of instrument and representativeness error the model rows sit on"),
        "reference_failures": reference_failures,
        "leads": leads,
    }
    _write_json(out_dir / "scorecard.json", payload)
    text = "\n".join(scorecard_lines(payload)) + "\n"
    (out_dir / "scorecard.txt").write_text(text, encoding="utf-8")
    return payload


def _fmt(value, width: int = 8, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f"{'-':>{width}s}"
    return f"{value:{width}.{digits}f}"


def scorecard_lines(payload: dict) -> list[str]:
    """The scorecard as plain text: one block per row kind, one row per lead."""
    label = payload["label"]
    lines = [f"WOOF Global forecast scorecard: {label}, T{payload['truncation']}, "
             f"start {payload['start_time_utc']}", ""]
    def surface_row(lead_h, who, scores):
        n = max(int(scores[v]["n"]) for v in scores)
        cells = [f"{_fmt(scores[v]['bias'], 7)}/{_fmt(scores[v]['rmse'], 7)}"
                 for v in ("temperature_2m", "dewpoint_2m", "wind_speed_10m", "mslp")]
        return f"{lead_h:6g} {who:18s} {n:5d} " + " ".join(cells)

    def sounding_row(lead_h, who, scores):
        g = {name: scores[name]["global"] for name in ("z500", "t850", "t500", "w250", "w850")}
        return (f"{lead_h:6g} {who:18s} {g['z500']['n']:5d} "
                f"{_fmt(g['z500']['bias'], 7)}/{_fmt(g['z500']['rmse'], 7)} "
                f"{_fmt(g['t850']['bias'], 7)}/{_fmt(g['t850']['rmse'], 7)} "
                f"{_fmt(g['t500']['bias'], 7)}/{_fmt(g['t500']['rmse'], 7)} "
                f"{_fmt(g['w250']['rmsve'], 7)} {_fmt(g['w850']['rmsve'], 7)}")

    lines.append("Surface stations (ASOS/METAR via rw_asos), minus observation: bias / rmse "
                 "(analysis rows: that analysis's own fit to the same reports, its own station set)")
    lines.append(f"{'lead h':>6s} {'row':18s} {'n':>5s} {'T2 K':>15s} {'Td2 K':>15s} {'WS10 m/s':>15s} {'MSLP hPa':>15s}")
    for lead in payload["leads"]:
        lines.append(surface_row(lead["lead_h"], label, lead["surface"]["scores"][label]))
        for name, block in sorted(lead.get("analyses", {}).items()):
            if block.get("surface_fit"):
                lines.append(surface_row(lead["lead_h"], f"{name} analysis", block["surface_fit"]["scores"][name]))
    lines.append("")
    lines.append("Radiosondes (IGRA2 via rw_igra2), global, minus observation: bias / rmse; winds: rmsve")
    lines.append(f"{'lead h':>6s} {'row':18s} {'sites':>5s} {'Z500 m':>15s} {'T850 K':>15s} {'T500 K':>15s} {'W250':>7s} {'W850':>7s}")
    for lead in payload["leads"]:
        upper = lead.get("upper_air")
        if not upper:
            continue
        lines.append(sounding_row(lead["lead_h"], label, upper["scores"][label]))
        for name, block in sorted(lead.get("analyses", {}).items()):
            if block.get("upper_air_fit"):
                lines.append(sounding_row(lead["lead_h"], f"{name} analysis", block["upper_air_fit"]["scores"][name]))
    for name in payload.get("references", {}):
        lines.append("")
        lines.append(f"Secondary reference: model minus the {name.upper()} analysis on the run's grid, "
                     "global: rmse (z500 m, t850 K, rh700 %), rmsve (w250, w850 m/s), z500 anomaly correlation")
        lines.append(f"{'lead h':>6s} {'Z500':>7s} {'T850':>7s} {'W250':>7s} {'W850':>7s} {'RH700':>7s} {'Z500 AC':>8s}")
        for lead in payload["leads"]:
            block = lead["analyses"].get(name)
            if not block:
                continue
            s = block["scores"]
            g = {k: s[k]["regions"]["global"] for k in s}
            lines.append(
                f"{lead['lead_h']:6g} {_fmt(g['z500']['rmse'], 7)} {_fmt(g['t850']['rmse'], 7)} "
                f"{_fmt(g['w250']['rmsve'], 7)} {_fmt(g['w850']['rmsve'], 7)} {_fmt(g['rh700']['rmse'], 7)} "
                f"{_fmt(g['z500'].get('anomaly_correlation'), 8, 4)}")
    for failure in payload.get("reference_failures", []):
        lines.append("")
        lines.append(f"No {failure['family']} analysis at {failure['valid_time']}: {failure['reason']}")
    for lead in payload["leads"]:
        screen = lead.get("screen") or {}
        if screen.get("degraded"):
            lines.append("")
            used = sorted(set(screen["expected_references"]) - set(screen["missing_references"]))
            lines.append(
                f"Screen degraded at lead {lead['lead_h']:g} h: the gross-error screen expects "
                f"{', '.join(screen['expected_references'])} and lacked "
                f"{', '.join(screen['missing_references'])}, so the model's radiosonde rows there "
                f"were screened by {', '.join(used) or 'no analysis'} alone")
    return lines


# --------------------------------------------------------------------------
# the door
# --------------------------------------------------------------------------


def add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", type=Path, help="the output directory of `gpuwm-global run` (any truncation)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory the scorecard (scorecard.json, scorecard.txt), the door records and "
                             "the fetched analyses land in")
    parser.add_argument("--hours", type=float, nargs="+", default=None,
                        help="the leads to score, in hours; default every checkpoint after the start")
    parser.add_argument("--label", default=None, help="the model's row label (default woof-global-t<N>)")
    parser.add_argument("--asos-record", type=Path, default=None,
                        help="a `rw_asos decode` record (gpuwm-obs.asos-surface.v2) covering the leads; "
                             "default: fetch and decode one through rw_asos")
    parser.add_argument("--igra2-table", type=Path, default=None,
                        help="a `rw_igra2 table` neutral table covering the leads; default: fetch and "
                             "decode one through rw_igra2")
    parser.add_argument("--reference", type=_parse_reference, action="append", default=[],
                        metavar="LABEL=FILE",
                        help=f"an analysis already on disk, LABEL one of {', '.join(REFERENCE_FAMILIES)}; "
                             "repeat per file")
    parser.add_argument("--families", default=",".join(REFERENCE_FAMILIES),
                        help="the reference centres to fetch analyses for (comma-separated)")
    parser.add_argument("--no-fetch-references", action="store_true",
                        help="score only the --reference files given; fetch no analysis")
    parser.add_argument("--initial-analysis", type=Path, default=None,
                        help="the run's initial analysis file when the receipt's recorded path is not on "
                             "this machine (its SHA-256 must be the one the receipt recorded)")


def score_main(args: argparse.Namespace) -> int:
    families = tuple(name.strip() for name in str(args.families).split(",") if name.strip())
    unknown = [name for name in families if name not in REFERENCE_FAMILIES]
    if unknown:
        raise SystemExit(f"--families: {', '.join(unknown)} is not a reference family; the table holds "
                         f"{', '.join(REFERENCE_FAMILIES)}")
    payload = score_forecast(
        args.run_dir, args.out, hours=args.hours, label=args.label, asos=args.asos_record,
        igra2=args.igra2_table, references=list(args.reference),
        fetch_references=not args.no_fetch_references, families=families,
        initial_analysis=args.initial_analysis,
    )
    print("\n".join(scorecard_lines(payload)))
    print(f"score: wrote {Path(args.out) / 'scorecard.json'} and {Path(args.out) / 'scorecard.txt'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="arwen_global.forecast_scorecard")
    add_score_arguments(parser)
    sys.exit(score_main(parser.parse_args()))
