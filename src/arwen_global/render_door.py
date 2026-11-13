"""``gpuwm-global render``: checkpoints to pictures in one command.

Before this door a picture took two commands and a hop between two
distributions: export render-ready tapes with ``export``, find them, then
drive the engine's ``render``.  ``wrfout_export`` says so in its own
docstring.  A capability that needs a reader to know the intermediate format
is a capability with no door, so this is the door.

THE RENDER PATH IS THE ENGINE'S RUST RENDERER AND NOTHING ELSE.  Every
weather-field picture comes out of ``rw_wrfbatch`` through
``gpuwm.render.render_wrfouts_rust``.  There is no matplotlib fallback here
and there will not be one: the engine's own ``--engine matplotlib`` escape
exists for a different problem and drawing a weather field with it is not
something this package offers.  When the renderer is not staged, this door
refuses and names the binary and the bundle it comes from.

The tapes are an intermediate, so by default they are written under the
output directory and removed when the pictures are drawn.  ``--keep-tapes``
keeps them, and ``--tapes-dir`` puts them somewhere a reader chooses; either
way the export receipt travels with them.

A WORLD'S COAST AND NAME (worlds plan, package 2.7).  A run whose
``[world]`` changes the land mask, or whose tapes say their coast is their
own (``ARWEN_WORLD_COAST = "own"``), is drawn with the coast of its own
tapes: the land mask of the first tape is outlined into a map folder
(:mod:`arwen_global.world.coast`) under ``<outdir>/world-map``, and the
renderer is pointed at it through ``RUSTWX_BASEMAP_DIR`` for this draw only.
Any world that is not Earth also has its label, and the tape's science or
illustration class when the tape carries one, appended to every panel
title.  ``--own-coast``, ``--map-folder`` and ``--title-suffix`` ask for the
same by hand.  On Earth the renderer is called exactly as before, in one
call with no annotation, and its environment is untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil

from .configs_dir import config_argument
from .status import StatusWriter

__all__ = ["add_render_arguments", "render"]

#: The checkpoint files a run directory holds, in time order.
CHECKPOINT_GLOB = "arwen_global_step*.npz"

#: What a bare `--products` means.  `all` renders the renderer's full catalog,
#: which for a global tape is minutes rather than seconds, so the default is
#: four surface fields a reader looks at first.
#:
#: THESE ARE THE RENDERER'S OWN SLUGS, and they were read off its catalog for a
#: global tape rather than invented (`gpuwm render TAPE --list-products`).  The
#: first spelling of this default named `t2,wind10,precip,olr`, none of which is
#: a slug the catalog carries, so a bare `gpuwm-global render` drew nothing at
#: all and said so only in the skipped list.  A default that renders nothing is
#: a door that does not exist.  MEASURED 2026-09-07 against a T255 tape: these
#: four report `renderable`, while `simulated_ir_satellite` (the outgoing
#: longwave a fifth entry would want) reports `missing-fields` because the tape
#: does not store it, so it is deliberately not here.
DEFAULT_PRODUCTS = "2m_temperature,mslp_10m_winds,10m_wind_speed_and_direction,total_qpf"


def add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=config_argument, default=None,
        help="the experiment TOML the checkpoints were run under, or the name "
             "of a shipped experiment.  Optional: a run directory written by "
             "`run` or `go` carries a copy of its own config, and that copy is "
             "used when this flag is absent")
    parser.add_argument(
        "inputs", nargs="+", type=Path, metavar="CHECKPOINT_OR_RUNDIR",
        help="checkpoints in time order, run directories (every "
             f"{CHECKPOINT_GLOB} inside, sorted), or wrfout tapes that are "
             "already exported")
    parser.add_argument(
        "--outdir", type=Path, required=True,
        help="where the pictures go, laid out <outdir>/<domain>/<product>/<valid-day>/")
    parser.add_argument(
        "--start-date", default=None,
        help="analysis valid time as YYYY-MM-DD_HH:MM:SS.  Optional: a run "
             "that starts from an analysis (or states start_time_utc) stamps "
             "its tapes from its own forecast clock, and a value that "
             "disagrees with that clock is refused; required only for an "
             "idealized run with no date")
    parser.add_argument(
        "--products", default=DEFAULT_PRODUCTS, metavar="LIST",
        help=f"comma-separated products, or 'all' (default: {DEFAULT_PRODUCTS})")
    parser.add_argument(
        "--size", default="1600x1000", metavar="WxH",
        help="picture size in pixels (default 1600x1000)")
    parser.add_argument(
        "--nlat", type=int, default=360,
        help="latitude points of the export grid (default 360)")
    parser.add_argument(
        "--nlon", type=int, default=720,
        help="longitude points of the export grid (default 720)")
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="crop to a lat/lon window (degrees, lon in -180..180)")
    parser.add_argument(
        "--tapes-dir", type=Path, default=None,
        help="where the intermediate tapes are written (default: a directory "
             "under --outdir that is removed when the pictures are drawn)")
    parser.add_argument(
        "--keep-tapes", action="store_true",
        help="keep the exported tapes and their receipt")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace tapes and pictures that already exist")
    parser.add_argument(
        "--own-coast", action="store_true",
        help="draw the coast of the tapes' own land mask instead of Earth's "
             "map (automatic when the config's [world] changes the coast)")
    parser.add_argument(
        "--map-folder", type=Path, default=None, metavar="DIR",
        help="draw land, coast and borders from this map folder (the "
             "natural_earth_10m, natural_earth_110m and us_counties_5m "
             "layout the renderer reads) instead of Earth's")
    parser.add_argument(
        "--title-suffix", default=None, metavar="TEXT",
        help="append TEXT to every panel title (default for a world that is "
             "not Earth: its label, and the tape's science or illustration "
             "class when the tape carries one)")


#: Where a world's own map folder is written, under --outdir.
WORLD_MAP_DIRNAME = "world-map"

#: The panel annotation file the title suffix travels in, under --outdir.
#: A draw whose tapes carry different classes writes one per class,
#: ``world-title-1.json`` and on.
WORLD_ANNOTATION_NAME = "world-title.json"

#: The global attributes a world's tape carries (worlds packages 1.4 and
#: 2.8) that this door reads: the label (1.4 writes ARWEN_WORLD, 2.8
#: ARWEN_WORLD_LABEL), the tape's class for its lead time, and whether the
#: coast is the world's own ("own") or Earth's ("earth").  An Earth tape
#: carries none of them.
TAPE_WORLD_ATTRS = ("ARWEN_WORLD", "ARWEN_WORLD_LABEL", "ARWEN_WORLD_CLASS",
                    "ARWEN_WORLD_COAST")

#: How a tape's class reads in a panel title.  A tape is "science" only on
#: days 2 to 6 of Earth with its mountains lowered, and then only near the
#: lowered ranges (``wrfout_export.world_tape_class``); a whole-globe panel
#: titled plain "science" would claim the rest of the map too.
CLASS_TITLES = {"science": "science near the lowered ranges only"}


def _tape_world(path: Path) -> dict[str, str]:
    """The world attributes of one tape, or {} for an Earth tape (or a file
    netCDF4 cannot open here; the renderer then says what is wrong)."""

    try:
        from netCDF4 import Dataset
    except ImportError:          # pragma: no cover - netCDF4 is a dependency
        return {}
    try:
        with Dataset(os.fspath(path)) as tape:
            names = set(tape.ncattrs())
            return {name: str(tape.getncattr(name))
                    for name in TAPE_WORLD_ATTRS if name in names}
    except OSError:
        return {}


def _world_draw_settings(args: argparse.Namespace, world, tapes, outdir: Path,
                         status) -> tuple[Path | None, list]:
    """(map folder, [(tapes, annotation file)]) for this draw.

    On Earth: (None, [(all tapes, None)]), which is the one call the door
    has always made.

    The map folder is the flag's, or one written from the first tape's own
    land mask when the world changes the coast (by the config's [world] or
    by the tapes' own ARWEN_WORLD_COAST) or ``--own-coast`` asks.  Every
    tape of the draw must carry the same mask: one folder is handed to the
    renderer for all of them.  The title suffix is the flag's, or the
    world's label with the tape's class; consecutive tapes with the same
    suffix are drawn in one call.
    """

    from .world.coast import (
        check_map_folder, outline_land_mask, read_tape_land_mask,
        write_map_folder)

    attrs = [_tape_world(tape) for tape in tapes]
    # `go` and `run-plan` call this door with namespaces of their own that
    # predate these three options; absent, each is its default.
    map_folder = getattr(args, "map_folder", None)
    own_coast = (getattr(args, "own_coast", False)
                 or (world is not None and world.changes_coast)
                 or any(row.get("ARWEN_WORLD_COAST") == "own" for row in attrs))
    folder = None
    if map_folder is not None:
        folder = check_map_folder(Path(map_folder))
    elif own_coast:
        lat, lon, land, digest = read_tape_land_mask(tapes[0])
        for other in tapes[1:]:
            if read_tape_land_mask(other)[3] != digest:
                raise ValueError(
                    f"{other} carries a land mask that differs from "
                    f"{tapes[0]}'s; one draw takes one coast, so render "
                    "tapes of different coasts separately")
        outline = outline_land_mask(lat, lon, land)
        folder = check_map_folder(write_map_folder(
            outline, outdir / WORLD_MAP_DIRNAME,
            source={"tape": Path(tapes[0]).name, "land_mask_sha256": digest,
                    "world_label": None if world is None else world.label,
                    "world_identity": None if world is None
                    else world.identity_hash}))
        status.note(
            f"own coast: {len(outline.coast)} coast lines, "
            f"{outline.land_cells} land cells of {outline.cells}, map folder "
            f"{folder}")

    explicit = getattr(args, "title_suffix", None)
    config_label = None
    if world is not None and not world.is_earth:
        # A world left with the default label would otherwise be titled
        # "earth"; its token says which world it is instead.
        config_label = (world.label if world.label != "earth"
                        else f"world {world.token}")
    suffixes = []
    for row in attrs:
        if explicit:
            suffixes.append(explicit)
            continue
        label = row.get("ARWEN_WORLD") or row.get("ARWEN_WORLD_LABEL") or config_label
        klass = row.get("ARWEN_WORLD_CLASS")
        klass = CLASS_TITLES.get(klass, klass)
        suffixes.append(None if not label
                        else f"{label} ({klass})" if klass else label)

    groups: list[tuple[list, Path | None]] = []
    files: dict[str, Path] = {}
    distinct = list(dict.fromkeys(s for s in suffixes if s))
    for tape, suffix in zip(tapes, suffixes):
        annotation = None
        if suffix:
            annotation = files.get(suffix)
            if annotation is None:
                outdir.mkdir(parents=True, exist_ok=True)
                name = (WORLD_ANNOTATION_NAME if len(distinct) == 1 else
                        f"world-title-{distinct.index(suffix) + 1}.json")
                annotation = outdir / name
                annotation.write_text(
                    json.dumps({"title_suffix": suffix}) + "\n", encoding="utf-8")
                files[suffix] = annotation
        if groups and groups[-1][1] == annotation:
            groups[-1][0].append(tape)
        else:
            groups.append(([tape], annotation))
    return folder, groups


@contextlib.contextmanager
def _map_folder_environment(folder: Path | None):
    """``RUSTWX_BASEMAP_DIR`` set to ``folder`` for the draw, then restored.

    The engine copies this process's environment into the renderer's and
    leaves an explicit ``RUSTWX_BASEMAP_DIR`` alone, and the renderer reads
    that variable before any other map root.  ``None`` touches nothing.
    """

    if folder is None:
        yield
        return
    before = os.environ.get("RUSTWX_BASEMAP_DIR")
    os.environ["RUSTWX_BASEMAP_DIR"] = str(folder)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("RUSTWX_BASEMAP_DIR", None)
        else:
            os.environ["RUSTWX_BASEMAP_DIR"] = before


def _sweep_render_scratch(outdir: Path, status) -> None:
    """Remove the renderer's scratch root when it is empty.

    The engine gives each render its own store under `<outdir>.render-scratch`
    and removes the store when the file's render finishes; on some filesystems
    the empty directory survives, and what a reader then sees beside their
    pictures is a directory named "scratch" that nothing cleaned up.

    Only ever removed when it is COMPLETELY EMPTY -- no entries at all, not
    merely no files.  Two reasons, and the second is measured: a concurrent
    render may own a store in there and taking it would kill that run's
    output; and on Windows a store path inside a deep output directory
    exceeds MAX_PATH, at which point `Path.is_file()` answers False for a
    file that exists.  A sweep that trusted that answer would call a
    populated store empty, so the test is the one question a long path
    cannot corrupt: does the directory have any entry.
    """

    scratch = Path(str(outdir) + ".render-scratch")
    if not scratch.is_dir():
        return
    try:
        entries = any(scratch.iterdir())
    except OSError:
        return
    if entries:
        status.note(
            f"left {scratch}: the renderer's own scratch stores are still "
            "there.  They are the engine's to remove and it did not; nothing "
            "here deletes a directory it did not create the contents of")
        return
    try:
        shutil.rmtree(scratch)
    except OSError:
        pass


def _config_for(args: argparse.Namespace) -> Path:
    """The config to export under: the flag, or the run directory's own copy.

    A reader who has just run a forecast should not have to re-state which
    experiment produced it.  `run` and `go` leave a byte copy of the config in
    the run directory precisely so this question has an answer; when neither
    the flag nor the copy is there, the refusal says which run directory was
    looked in rather than printing "config is required".
    """

    from .cli import read_run_sidecar

    if args.config is not None:
        return Path(args.config)
    looked = []
    for entry in args.inputs:
        path = Path(entry)
        directory = path if path.is_dir() else path.parent
        looked.append(str(directory))
        found = read_run_sidecar(directory)
        if found is not None:
            return found
    raise ValueError(
        "no --config was given and none of the inputs sit in a run directory "
        "that carries its own config copy.\n"
        "Looked in: " + ", ".join(looked) + "\n"
        "Pass --config with the experiment TOML (or a shipped experiment "
        "name), or render a run directory written by `run` or `go`, which "
        "leaves its config beside the checkpoints.")


def _expand(inputs) -> tuple[list[Path], list[Path]]:
    """Split the positional inputs into (checkpoints, already-exported tapes)."""

    checkpoints: list[Path] = []
    tapes: list[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            found = sorted(path.glob(CHECKPOINT_GLOB))
            if not found:
                raise FileNotFoundError(
                    f"{path} is a directory with no {CHECKPOINT_GLOB} in it; "
                    "point at a run directory, a checkpoint, or an exported tape")
            checkpoints.extend(found)
        elif path.suffix == ".npz":
            checkpoints.append(path)
        elif path.suffix in (".nc", ""):
            tapes.append(path)
        else:
            raise ValueError(
                f"{path} is neither a checkpoint (.npz), a run directory, nor a "
                "wrfout tape")
    return checkpoints, tapes


def _require_renderer() -> Path:
    """The renderer, or a refusal naming the binary and its bundle."""

    from .doors import find_door

    found = find_door("rw_wrfbatch")
    if found is None:
        raise RuntimeError(
            "rw_wrfbatch is not staged, so there is nothing that can draw a "
            "weather field here.  It ships in the engine's bridge bundle: run "
            "`gpuwm fetch-bridges`.  This door has no matplotlib fallback.")
    return found


def _parse_size(spec: str) -> tuple[int, int]:
    try:
        width, height = spec.lower().split("x", 1)
        return int(width), int(height)
    except ValueError:
        raise ValueError(
            f"--size {spec!r} is not WxH, for example 1600x1000") from None


def render(args: argparse.Namespace) -> int:
    from .config import load_config
    from .wrfout_export import EXPORT_RECEIPT_NAME, export_wrfout

    outdir = Path(args.outdir)
    status = StatusWriter(outdir, "render", stages=("export", "draw"))
    if status.log_path is not None:
        print(f"log: {status.log_path}")
    try:
        _require_renderer()
        checkpoints, tapes = _expand(args.inputs)
        size = _parse_size(args.size)
        world = None

        if checkpoints:
            tapes_dir = Path(args.tapes_dir) if args.tapes_dir else outdir / "tapes"
            status.stage("export", step_count=len(checkpoints))
            cfg = load_config(_config_for(args))
            world = cfg.world
            written = export_wrfout(
                cfg, checkpoints, tapes_dir,
                nlat=args.nlat, nlon=args.nlon,
                start_date=args.start_date,
                overwrite=args.overwrite,
                bbox=tuple(args.bbox) if args.bbox else None,
            )
            status.note(f"{len(written)} tapes under {tapes_dir}")
            tapes = [*tapes, *written]
        else:
            tapes_dir = None
            if args.config is not None:
                world = load_config(Path(args.config)).world

        if not tapes:
            raise ValueError("nothing to render: no checkpoints and no tapes")

        map_folder, groups = _world_draw_settings(
            args, world, tapes, outdir, status)

        status.stage("draw", step_count=len(tapes))
        from gpuwm.render import render_wrfouts_rust

        # Every keyword on this call exists on the engine's renderer, and a
        # test holds it there: `series` was passed here from the day the door
        # was written and NO engine has ever declared it, so every draw died
        # at the call with a TypeError after the tapes had been exported
        # (measured 2026-09-07 against 2.7.0 and against the tree the model is
        # developed on; both signatures agree and neither has it).
        #
        # On Earth there is one group, every tape, and its `annotate` is None,
        # which is the engine's own default and adds no argument to the
        # renderer's command line: the one call this door has always made.
        # A world's tapes of different classes are drawn group by group, each
        # with its own title; the renderer draws one file per invocation
        # either way.
        pictures, failures, skipped = [], [], []
        with _map_folder_environment(map_folder):
            for group, annotation in groups:
                drawn, failed, passed_over = render_wrfouts_rust(
                    group, products=args.products, timeidx=None,
                    outdir=outdir, size=size, annotate=annotation)
                pictures.extend(drawn)
                failures.extend(failed)
                skipped.extend(passed_over)
        status.note(f"{len(pictures)} pictures, {len(failures)} failures")

        if tapes_dir is not None and not args.keep_tapes:
            receipt = tapes_dir / EXPORT_RECEIPT_NAME
            keep = receipt.read_text(encoding="utf-8") if receipt.is_file() else None
            shutil.rmtree(tapes_dir, ignore_errors=True)
            if keep is not None:
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / EXPORT_RECEIPT_NAME).write_text(keep, encoding="utf-8")
            status.note("intermediate tapes removed; --keep-tapes keeps them")

        _sweep_render_scratch(outdir, status)

        print(json.dumps({
            "pictures": [str(path) for path in pictures],
            "failures": list(failures),
            "skipped": [list(row) for row in skipped],
            "outdir": str(outdir),
        }, indent=2, sort_keys=True))
        if failures:
            status.failed(f"{len(failures)} product(s) did not draw")
            return 1
        status.done(f"{len(pictures)} pictures")
        return 0
    except Exception as exc:
        status.failed(f"{type(exc).__name__}: {exc}")
        raise
