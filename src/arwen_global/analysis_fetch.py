"""``gpuwm-global fetch-analysis``: the one object a global cold start needs.

The route a reader takes is fetch, run, export, render, and the FIRST of the
four used to belong to a different program: `gpuwm fetch --source gdas
--hours 0 --mode full-file`, with two flags that are not that door's defaults
and both of which decide whether the run works at all.  The engine's own
acquisition door prints what comes next after a whole-globe GDAS fetch --
`GDAS_GLOBAL_ANALYSIS_MAPPING_ID` and `gdas_fetch_epilogue` in
`gpuwm.fetch` -- and a PUBLISHED engine carries neither, so on a plain `pip
install gpuwm gpuwm-global` the first command of the route ended in a full
stop that says this ArWen has no regional GDAS route, and nothing at all
about the model the reader installed.

So the route's first step is a door of this package's own.  It is the same
transport underneath -- the engine's Rust fetch route, through
`arwen_global.da_streams.fetch_analysis`, which is what `gpuwm-global da
fresh` has always used -- with the two flags bound rather than typed, and
the next step printed.

THE ENGINE IS ASKED FIRST for both names.  When a published engine carries
`GDAS_GLOBAL_ANALYSIS_MAPPING_ID` this module returns the engine's value,
and the `--print-next-step` text is the engine's `gdas_fetch_epilogue` when
the engine has one.  What this module never does is print the engine's
regional full stop as though it were this model's answer.

``--source era5`` is the second input.  The GDAS archive the engine's route
reads begins at 2021-03-22 12Z and holds the four synoptic hours only; ERA5
covers every hour from 1940 to about six days ago, from Google's public ARCO
store with no key.  The fetch and what the cold start adds to ERA5 are
`arwen_global.era5_analysis`.

WHY THE NEXT STEP NAMES A SHIPPED EXPERIMENT BY NAME.  A repository-relative
config path is typed by someone who has the repository; a reader who ran `pip
install` has the experiments as package data, and a bare name resolves to
them (`arwen_global.configs_dir`).  The name is the spelling that works in
both places.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

__all__ = [
    "GDAS_GLOBAL_ANALYSIS_MAPPING_ID",
    "GDAS_GLOBAL_QUICKSTART_CONFIG",
    "GDAS_GLOBAL_QUICKSTART_CYCLE",
    "GDAS_GLOBAL_QUICKSTART_DOC",
    "GDAS_GLOBAL_QUICKSTART_OUT",
    "add_fetch_analysis_arguments",
    "analysis_mapping_id",
    "fetch_analysis_main",
    "gdas_fetch_epilogue",
    "missing_analysis_remedy",
    "shipped_configs_reading",
]

#: The source id the shipped GDAS experiments name.  The engine's value when
#: a published engine carries it, this copy otherwise; the two are held
#: together by tests/test_arwen_global_quickstart.py.
GDAS_GLOBAL_ANALYSIS_MAPPING_ID = "gdas-global"

#: The shipped experiment the next step names, by the spelling that resolves
#: from an install as well as from a checkout.  It is the README's forecast
#: day: one quick start, one config, one cycle (0.3.0 had two, the README's
#: native-physics day and the quickstart page's reference-physics
#: 2026-08-30 18Z run).
GDAS_GLOBAL_QUICKSTART_CONFIG = "arwen_global_gdas_t255_native_sl_si_24h"

#: The cycle and folder that config reads, spelled once for the hint and
#: the pages (tests/test_arwen_global_quickstart.py holds them to the
#: config's own ``[initial] analysis_grib``).
GDAS_GLOBAL_QUICKSTART_CYCLE = "2026-09-01T00"
GDAS_GLOBAL_QUICKSTART_OUT = "cases/baseline-2026090100"

#: The page that runs the whole route, in this repository.
GDAS_GLOBAL_QUICKSTART_DOC = "docs/ARWEN_GLOBAL_QUICKSTART.md"

#: The object a whole-globe GDAS fetch lands, by the transport's own format
#: string (`gpuwm.fetch._fetch_gfs_fullfile_locked` names each object
#: `f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"`).
ANALYSIS_OBJECT = "gdas.t{hour:02d}z.pgrb2.0p25.f000"


def analysis_mapping_id() -> str:
    """The GDAS global analysis source id: the engine's, or the carried one."""

    try:
        from gpuwm import fetch as engine_fetch
    except Exception:  # pragma: no cover - a broken engine install
        return GDAS_GLOBAL_ANALYSIS_MAPPING_ID
    value = getattr(engine_fetch, "GDAS_GLOBAL_ANALYSIS_MAPPING_ID", None)
    return str(value) if value else GDAS_GLOBAL_ANALYSIS_MAPPING_ID


def gdas_fetch_epilogue(*, out, cycle: dt.datetime,
                        whole_globe_analysis: bool) -> list[str]:
    """What a completed GDAS fetch says about what comes next.

    The engine's own, when a published engine carries it; this text
    otherwise.  Returned as lines rather than printed, so the hint and the
    shipped quickstart config cannot drift apart unnoticed --
    tests/test_arwen_global_quickstart.py holds the ``[initial]`` keys below
    against the ones that config really carries.

    A CROPPED fetch gets no next step, and that is not an oversight.  The
    default GDAS transport is a NOMADS area crop, and a crop cannot
    initialize a global model: `analysis_initial` measures the longitude ring
    and refuses one that does not close.  Printing `gpuwm-global run` there
    would send a reader to a refusal, which is worse than a stop.
    """

    try:
        from gpuwm import fetch as engine_fetch
    except Exception:  # pragma: no cover - a broken engine install
        engine_fetch = None
    engine_epilogue = getattr(engine_fetch, "gdas_fetch_epilogue", None)
    if engine_epilogue is not None:
        return list(engine_epilogue(
            out=Path(out), cycle=cycle,
            whole_globe_analysis=whole_globe_analysis))

    lines: list[str] = []
    if whole_globe_analysis:
        # as_posix, because this is printed INSIDE a TOML basic string a
        # reader pastes: a Windows `data\gdas-analysis\...` carries `\g` and
        # `\.`, which TOML rejects as invalid escapes.
        analysis = (Path(out) / ANALYSIS_OBJECT.format(
            hour=cycle.hour)).as_posix()
        readers = shipped_configs_reading(analysis)
        if readers:
            name = (GDAS_GLOBAL_QUICKSTART_CONFIG
                    if GDAS_GLOBAL_QUICKSTART_CONFIG in readers else readers[0])
            lines.append(
                f"fetch-analysis: next: the shipped forecast day {name} "
                "reads this file:")
            lines.extend(_remaining_route(name))
            if len(readers) > 1:
                lines.append(
                    f"  # {len(readers) - 1} other shipped experiment(s) read "
                    "it too; `gpuwm-global configs --paths` lists every one.")
        else:
            lines.append(
                "fetch-analysis: next: no shipped experiment reads this file, "
                "so the run's config is yours.  Copy one "
                "(`gpuwm-global configs --paths` prints where they are) and "
                "set the key that is this fetch's output:")
            lines.append(
                "  # CONFIG is yours, and one key of it is this fetch's "
                "output:\n"
                "  #   [initial]\n"
                "  #   mode = \"analysis\"\n"
                f"  #   analysis_grib = \"{analysis}\"\n"
                f"  #   analysis_mapping = \"{analysis_mapping_id()}\"\n"
                "  # and delete start_time_utc from its physics options (or "
                "set it to this cycle):\n"
                "  # a run refuses an analysis whose instant disagrees.")
            lines.extend(_remaining_route("CONFIG.toml"))
        lines.append(
            f"  # the README's day is {GDAS_GLOBAL_QUICKSTART_CONFIG} on the "
            f"GDAS {GDAS_GLOBAL_QUICKSTART_CYCLE}Z analysis in "
            f"{GDAS_GLOBAL_QUICKSTART_OUT};")
        lines.append(
            "  # the route with the reason for each step is "
            f"{GDAS_GLOBAL_QUICKSTART_DOC}.")
    else:
        lines.append(
            "fetch-analysis: this object is a NOMADS AREA CROP and cannot "
            "cold-start a global model: the initializer measures the "
            "longitude ring and refuses one that does not close.  Re-fetch "
            "without --area for a whole-globe analysis.")
    return lines


def shipped_configs_reading(analysis: str | Path) -> list[str]:
    """The shipped experiments whose ``[initial] analysis_grib`` is this
    file, by bare name, sorted.  Compared as written relative paths, which
    is how a shipped config spells it and how a reader types ``--out``."""

    import tomllib

    from .configs_dir import config_root

    def norm(value) -> str:
        text = Path(value).as_posix()
        while text.startswith("./"):
            text = text[2:]
        return text

    wanted = norm(analysis)
    names: list[str] = []
    for path in sorted(config_root().glob("*.toml")):
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        grib = (raw.get("initial") or {}).get("analysis_grib")
        if isinstance(grib, str) and norm(grib) == wanted:
            names.append(path.stem)
    return names


def _geog_staged() -> bool:
    """Whether a WPS_GEOG archive the statics door would read is here."""

    try:
        from gpuwm.rustwx_static import geog_root_candidates
    except Exception:  # pragma: no cover - a broken engine install
        return False
    return any(Path(path).is_dir() for path in geog_root_candidates())


def _remaining_route(config: str) -> list[str]:
    """The commands between this fetch and a drawn map, in order."""

    lines: list[str] = []
    if not _geog_staged():
        lines.append(
            "  gpuwm fetch-geog                     # once per machine: the "
            "WPS_GEOG archive (2.7 GB download, 32 GB on disk)")
    lines.append(
        f"  gpuwm-global statics {config}   # once per truncation")
    lines.append(f"  gpuwm-global run {config} --outdir out/day")
    lines.append(
        "  gpuwm-global render out/day --outdir out/pictures "
        "--products 2m_temperature,mslp_10m_winds,500mb_height_winds")
    lines.append(
        f"  # or statics, run and render in one: gpuwm-global go {config} "
        "--outdir out/day")
    return lines


def missing_analysis_remedy(grib: str | Path, start_time_utc: str | None) -> str:
    """The sentence a run prints when its analysis file is not there yet.

    It names the fetch that writes exactly that file.  The cycle comes from
    the config's ``start_time_utc`` when it states one (the shipped native
    days do), else from the hour in the file's own name with the date left
    to the reader.
    """

    import re

    grib = Path(grib)
    out = grib.parent.as_posix() or "."
    cycle = None
    if start_time_utc:
        try:
            when = dt.datetime.fromisoformat(
                str(start_time_utc).replace("Z", "+00:00"))
            cycle = f"{when:%Y-%m-%dT%H}"
        except ValueError:
            cycle = None
    if cycle is None:
        match = re.match(r"gdas\.t(\d\d)z\.pgrb2", grib.name)
        cycle = f"YYYY-MM-DDT{match.group(1)}" if match else "YYYY-MM-DDTHH"
    return (f"analysis GRIB {grib} does not exist; fetch it with "
            f"`gpuwm-global fetch-analysis --cycle {cycle} --out {out}`, "
            "which writes exactly that file")


# ------------------------------------------------------------------- the door

#: Where each source lands when --out is not given.  An ERA5 hour gets a
#: directory of its own, because the engine's ARCO retrieval refuses to
#: write a second hour over a verified first one.
DEFAULT_OUT = {"gdas": Path("data/gdas-analysis"),
               "era5": Path("data/era5-analysis")}


def add_fetch_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source", default="gdas", choices=("gdas", "era5"),
        help="gdas: the NCEP GDAS 0.25-degree analysis (00/06/12/18Z, the "
             "archive the engine reads reaches back to 2021-03-22 12Z); era5: "
             "ECMWF ERA5 from Google's public ARCO store, any hour from 1940 "
             "to about six days ago, no key")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="directory the analysis object and its manifest land in "
             "(default data/gdas-analysis, or data/era5-analysis/YYYYMMDDHH "
             "with --source era5)")
    parser.add_argument(
        "--cycle", default=None, metavar="YYYY-MM-DDTHH",
        help="the analysis time to fetch; without it, the newest published "
             "GDAS cycle (required with --source era5)")
    parser.add_argument(
        "--engine", default="auto", choices=("auto", "rust", "python"),
        help="which transport the engine's GDAS fetch route uses")
    parser.add_argument(
        "--quiet", action="store_true",
        help="do not print the transport's own progress lines")


def _parse_cycle(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "")
    for shape in ("%Y-%m-%dT%H", "%Y-%m-%d_%H", "%Y-%m-%dT%H:%M",
                  "%Y%m%d%H"):
        try:
            return dt.datetime.strptime(text, shape)
        except ValueError:
            continue
    raise ValueError(
        f"--cycle {value!r} is not a cycle instant; spell it "
        "YYYY-MM-DDTHH (GDAS runs at 00, 06, 12 and 18Z)")


def fetch_analysis_main(args: argparse.Namespace) -> int:
    """Fetch one whole-globe analysis and say what comes next."""

    cycle = _parse_cycle(getattr(args, "cycle", None))
    progress = (lambda *_a, **_k: None) if getattr(args, "quiet", False) else print
    source = getattr(args, "source", "gdas") or "gdas"
    out = getattr(args, "out", None)
    if source == "era5":
        from .era5_analysis import era5_fetch_epilogue, fetch_era5_analysis

        if cycle is None:
            raise SystemExit(
                "fetch-analysis --source era5 needs --cycle YYYY-MM-DDTHH: "
                "ERA5 is a reanalysis of any past hour, so there is no "
                "'newest' to default to")
        out = Path(out) if out is not None else (
            DEFAULT_OUT["era5"] / f"{cycle:%Y%m%d%H}")
        record = fetch_era5_analysis(out, cycle, progress=progress)
        print(f"fetch-analysis: {record['analysis']}")
        print(f"fetch-analysis: {record['snow_density']}")
        print(f"fetch-analysis: ERA5 valid {record['cycle']:%Y-%m-%dT%H}Z, "
              "mapping era5-arco-global")
        if record.get("provisional_times"):
            print("fetch-analysis: this hour is ERA5T (preliminary); ECMWF "
                  "may still revise it")
        for line in era5_fetch_epilogue(analysis=record["analysis"]):
            print(line)
        return 0

    from .da_streams import fetch_analysis

    record = fetch_analysis(
        Path(out) if out is not None else DEFAULT_OUT["gdas"],
        source="gdas", cycle=cycle,
        engine=getattr(args, "engine", "auto"), progress=progress)
    print(f"fetch-analysis: {record.path}")
    print(f"fetch-analysis: {record.bytes:,} bytes, sha256 {record.sha256}")
    print(f"fetch-analysis: cycle {record.cycle_utc}, mapping {record.mapping}")
    if record.manifest:
        print(f"fetch-analysis: manifest {record.manifest}")
    fetched = dt.datetime.fromisoformat(record.cycle_utc.replace("Z", "+00:00"))
    for line in gdas_fetch_epilogue(
            out=Path(out) if out is not None else DEFAULT_OUT["gdas"],
            cycle=fetched.replace(tzinfo=None),
            whole_globe_analysis=True):
        print(line)
    return 0
