"""Build this package's door bundle and the pins the wheel carries.

Two steps, run in this order by the release workflow and reproducible by
hand from a checkout of the engine's source tree:

``pack``
    On each target platform, after ``cargo build --release --locked
    --offline`` in the engine's ``tools/rustwx`` workspace, collect every
    door of ``arwen_global.doors.doors_from_bundle("gpuwm-global")`` into one
    zip named for the release and the platform.  The archive is
    deterministic: binaries in the declared door order, fixed member
    timestamps, no directory entries, so two packs of the same bytes produce
    the same archive.

``pin``
    On one machine, from the bundles ``pack`` produced, compute the size and
    SHA-256 of every bundle and of every binary inside it and write them into
    ``src/arwen_global/data/door-pins.json`` BEFORE the wheel is built.

Nothing here invents a hash: every number written comes from hashing bytes
that exist on this disk at the moment it runs.  ``pin`` refuses a bundle
whose members are not exactly the door set this package expects for that
platform, so a half-built bundle cannot be pinned into a wheel.

``pin`` also refuses a STALE bundle, and does it two ways, because hashing
binds a wheel to exact bytes and says nothing about which source produced
them.  The engine's own cut nearly shipped bundles predating its source tip
once, with every check passing because every check hashed what it was handed.

*   Every door built from the engine's workspace embeds
    ``GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`` at build time.
    ``pin --source-rev COMMIT`` -- a required argument, so no cut can skip it
    -- extracts the stamp from every member and refuses a bundle whose stamp
    is absent, unparseable, ambiguous, or names any other commit.  String
    extraction, never execution: it works on the other platform's binaries
    and on a runner with no GPU.

*   Every door also carries the contract literal its ``Door`` row declares,
    which is what makes "this is the door the Python half was written
    against" a property of the bytes rather than of a filename.

One door is exempt from the first check and says so by name rather than
silently: ``rw_atms`` declares its source-revision stamp as a ``pub static``
that nothing in the crate reads, so the linker drops it and the built binary
carries no stamp at all.  ``--allow-unstamped rw_atms`` is how a cut accepts
that, and it exists to be deleted: the fix is one line in the engine's
``crates/rw-atms/src/main.rs``, delivered with this release's patch series.
Until it lands the exemption is named on the command line of every cut, so
nobody can mistake it for a door that was checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"


def _load_door_table():
    """Load the door table from source, without importing the package.

    ``import arwen_global.doors`` would run the package's ``__init__``, which
    pulls in the model and therefore the engine.  This tool runs on a build
    runner that has a Rust toolchain and no gpuwm install, and it needs one
    table: the doors, their filenames, their contract literals.  Loading the
    module by path keeps the table the single source it already is without
    making the bundle build depend on the whole import graph.
    """

    import importlib.util

    path = SOURCE_ROOT / "arwen_global" / "doors.py"
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_doors_table", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


door_table = _load_door_table()

#: Fixed member timestamp so two packs of identical bytes produce identical
#: archives (a zip stores an mtime per member).
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

#: The byte marker every door built from the engine's workspace embeds.
SOURCE_REV_MARKER = b"GPUWM_BRIDGE_SOURCE_REV="
_SOURCE_REV_LENGTH = 40
_STAMP = re.compile(
    SOURCE_REV_MARKER + b"([0-9a-f]{%d})" % _SOURCE_REV_LENGTH)

PINS_PATH = REPO_ROOT / "src" / "arwen_global" / "data" / "door-pins.json"


def _companion_doors():
    return door_table.doors_from_bundle(door_table.COMPANION_BUNDLE)


def _locate(name: str, search: list[Path]) -> Path:
    for directory in search:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"build_door_bundle: {name} is in none of the search directories: "
        + ", ".join(str(d) for d in search))


def pack(release: str, platform: str, search: list[Path],
         out_dir: Path) -> Path:
    if platform not in door_table.SUPPORTED_PLATFORMS:
        raise SystemExit(
            f"build_door_bundle: unknown platform {platform!r}; known: "
            + ", ".join(door_table.SUPPORTED_PLATFORMS))
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / door_table.bundle_filename(release, platform)
    sources = [
        (door_table.artifact_filename(door.name, platform),
         _locate(door_table.artifact_filename(door.name, platform), search))
        for door in _companion_doors()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, source in sources:
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 0o755 in the high half of external_attr: the unix mode a zip
            # can carry.  Staging chmods anyway, but an operator who unzips
            # by hand should get executables.
            info.external_attr = (0o100755 << 16)
            zf.writestr(info, source.read_bytes())
    print(f"build_door_bundle: packed {archive} "
          f"({archive.stat().st_size:,} B) from {len(sources)} doors")
    for name, source in sources:
        print(f"  {name} <- {source}")
    return archive


def _platform_of(archive: Path, release: str) -> str:
    for platform in door_table.SUPPORTED_PLATFORMS:
        if archive.name == door_table.bundle_filename(release, platform):
            return platform
    raise SystemExit(
        f"build_door_bundle: {archive.name} is not a bundle name for release "
        f"{release}; expected one of "
        + ", ".join(door_table.bundle_filename(release, p)
                    for p in door_table.SUPPORTED_PLATFORMS))


def _verify_source_revision(payload: bytes, expected: str, label: str) -> None:
    found = sorted({match.decode("ascii") for match in _STAMP.findall(payload)})
    if not found:
        raise SystemExit(
            f"build_door_bundle: {label} carries no "
            f"{SOURCE_REV_MARKER.decode()}<commit> stamp, so nothing in the "
            "bytes says which source produced them; rebuild it from the "
            "engine checkout with GPUWM_BRIDGE_SOURCE_REV set, or name it in "
            "--allow-unstamped if the crate is known not to keep its stamp")
    if len(found) > 1:
        raise SystemExit(
            f"build_door_bundle: {label} carries more than one source stamp "
            f"({', '.join(found)}); refusing to pin a bundle whose provenance "
            "is ambiguous")
    if found[0] != expected:
        raise SystemExit(
            f"build_door_bundle: {label} was built from {found[0]} and this "
            f"cut declares {expected}; refusing to pin a stale door")


def _pin_bundle(archive: Path, release: str, source_rev: str,
                unstamped: set[str]) -> tuple[str, dict]:
    platform = _platform_of(archive, release)
    expected = [(door, door_table.artifact_filename(door.name, platform))
                for door in _companion_doors()]
    with zipfile.ZipFile(archive) as zf:
        held = set(zf.namelist())
        missing = [name for _, name in expected if name not in held]
        if missing:
            raise SystemExit(
                f"build_door_bundle: {archive.name} is missing "
                f"{', '.join(missing)}; refusing to pin a partial bundle")
        extra = sorted(held - {name for _, name in expected})
        if extra:
            raise SystemExit(
                f"build_door_bundle: {archive.name} carries members this "
                f"package does not publish ({', '.join(extra)}); refusing to "
                "pin a bundle whose contents are not the door set")
        binaries = []
        for door, name in expected:
            payload = zf.read(name)
            if door.name in unstamped:
                print(f"  {name}: source stamp NOT CHECKED "
                      f"(--allow-unstamped {door.name})")
            else:
                _verify_source_revision(
                    payload, source_rev, f"{archive.name}: {name}")
            if door.marker is not None and door.marker not in payload:
                raise SystemExit(
                    f"build_door_bundle: {archive.name}: {name} does not "
                    f"carry the contract literal "
                    f"{door.marker.decode(errors='replace')!r}, so it is not "
                    "the door this package's Python half was written against; "
                    f"rebuild it from {door.crate}")
            binaries.append({
                "artifact": door.name,
                "filename": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    record = {
        "bundle": {
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": door_table.sha256_file(archive),
        },
        "binaries": binaries,
    }
    return platform, record


def pin(release: str, archives: list[Path], out: Path, source_rev: str,
        unstamped: set[str], engine_rev: str | None) -> Path:
    platforms: dict[str, dict] = {}
    for archive in archives:
        platform, record = _pin_bundle(archive, release, source_rev, unstamped)
        if platform in platforms:
            raise SystemExit(f"build_door_bundle: two bundles for {platform}")
        platforms[platform] = record
    existing = json.loads(out.read_text(encoding="utf-8"))
    document = {
        "schema": door_table.DOOR_PINS_SCHEMA,
        "release": release,
        "engine_source_rev": engine_rev or source_rev,
        "note": existing.get("note"),
        "platforms": platforms,
    }
    if unstamped:
        document["unstamped"] = sorted(unstamped)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"build_door_bundle: wrote {out} for "
          f"{', '.join(sorted(platforms))} at release {release}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_door_bundle",
        description="pack and pin the door bundle this package publishes")
    sub = parser.add_subparsers(dest="command", required=True)

    packer = sub.add_parser("pack", help="build one platform's bundle")
    packer.add_argument("--release", required=True,
                        help="the release tag, e.g. v0.1.0")
    packer.add_argument("--platform", required=True,
                        choices=door_table.SUPPORTED_PLATFORMS)
    packer.add_argument("--search", required=True, nargs="+", type=Path,
                        help="directories holding the built doors")
    packer.add_argument("--out", required=True, type=Path)

    pinner = sub.add_parser("pin", help="write the pins from packed bundles")
    pinner.add_argument("--release", required=True)
    pinner.add_argument("--source-rev", required=True,
                        help="the 40-hex engine commit every door was built "
                             "from; a bundle whose stamps disagree is refused")
    pinner.add_argument("--engine-rev", default=None,
                        help="recorded in the pins as the engine revision the "
                             "crates came from (defaults to --source-rev)")
    pinner.add_argument("--allow-unstamped", action="append", default=[],
                        metavar="DOOR",
                        help="doors whose build is known not to keep the "
                             "source-revision stamp; each one is printed as "
                             "NOT CHECKED")
    pinner.add_argument("--out", type=Path, default=PINS_PATH)
    pinner.add_argument("archives", nargs="+", type=Path)

    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(args.release, args.platform, list(args.search), args.out)
        return 0
    unstamped = set(args.allow_unstamped or ())
    unknown = sorted(unstamped - {d.name for d in _companion_doors()})
    if unknown:
        raise SystemExit(
            f"build_door_bundle: --allow-unstamped names doors this package "
            f"does not publish: {', '.join(unknown)}")
    pin(args.release, list(args.archives), args.out, args.source_rev,
        unstamped, args.engine_rev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
