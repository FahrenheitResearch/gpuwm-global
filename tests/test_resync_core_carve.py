"""The carried engine core, and the tool that cuts it.

THE BREAKAGE THESE TESTS PREVENT.  `src/arwen_global/core/` is the engine's
physics, cut in because the published engine's copy of it is a different piece
of physics.  Three things about that cut are silent when they go wrong:

* A KERNEL whose bytes are not the source tree's, put through the one rule
  that is allowed to change them.  The loader hands the assembled source
  string to nvrtc AND to the manifest that digests it, so a rewritten COMMENT
  moves the `source_sha256` every published receipt was written under and
  nothing raises.  The first run of the carve rewrote eight such comments
  across five translation units, adding 48 bytes.  The one thing that IS
  rewritten is stated, tabled and measured: the two words this project does
  not publish, and one citation of the tool a comment was written with, all
  five lines of it, 17 bytes across three files.  Everything else is byte for
  byte, and this file is what says which is which.
* A LOADER that reads somebody else's directory.  `_KDIR` is
  `Path(__file__).parent`, which is the whole mechanism: if the carried
  modules resolved the engine's loader instead, they would compile the
  engine's kernels while every file here said otherwise.
* A MANIFEST NAMESPACE that moved.  `MODULE_KEY_ROOT` is a string, not an
  import path, and the rewiring's regex matches it.  Rewriting it would orphan
  every kernel pin this model has published.

The re-cut itself is checked against the source tree when there is one, and
skipped by name when there is not.  The tree's location is deliberately not
recorded in this repository: it is given in `GPUWM_GLOBAL_SOURCE_WORKTREE`.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "resync_from_owner.py"
PACKAGE = REPO / "src" / "arwen_global"


def _tool():
    spec = importlib.util.spec_from_file_location("resync_from_owner", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resync = _tool()


def _recorded_revision() -> str:
    text = (REPO / "SOURCE.md").read_text(encoding="utf-8")
    return re.search(r"^- Revision:\s*`([0-9a-f]{7,40})`", text, re.M).group(1)


SOURCE_WORKTREE = os.environ.get("GPUWM_GLOBAL_SOURCE_WORKTREE", "").strip()

#: The re-cut can only be exercised where the model's source tree is.  It is
#: private and its path is not published, so a run without it skips by name
#: rather than passing on a check it did not make.
needs_the_source_tree = pytest.mark.skipif(
    not (SOURCE_WORKTREE and (Path(SOURCE_WORKTREE) / ".git").exists()),
    reason="GPUWM_GLOBAL_SOURCE_WORKTREE is not set to the model's source "
           "tree, so the re-cut cannot be run here")


# --------------------------------------------------------------- the table

def test_every_carried_module_the_table_names_is_here():
    for name in resync.CORE_MODULES:
        assert (PACKAGE / "core" / f"{name}.py").is_file(), name
    assert (PACKAGE / "core" / "npref.py").is_file()
    assert (PACKAGE / "core" / "kernels" / "__init__.py").is_file()


def test_every_kernel_a_carried_module_compiles_is_carried():
    """The loader has no fallback, so an omission is a FileNotFoundError.

    The names are read off the carried sources rather than restated: a module
    that starts compiling a new translation unit must drag it into the table,
    and this is what says so.
    """

    wanted: set[str] = set()
    for path in sorted((PACKAGE / "core").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for call in re.finditer(
                r"(?:get_kernel|load_module)(?:_int_defines)?\(\s*\"([a-z0-9_]+)\"",
                text):
            wanted.add(call.group(1) + ".cu")
    assert wanted, "no kernel names were found in the carried modules at all"
    carried = {p.name for p in (PACKAGE / "core" / "kernels").iterdir()}
    assert wanted <= carried, sorted(wanted - carried)


def test_the_headers_the_loader_prepends_are_carried():
    kernels = PACKAGE / "core" / "kernels"
    sys.path.insert(0, str(PACKAGE.parent))
    try:
        from arwen_global.core.kernels import EXTRA_HEADERS
    finally:
        sys.path.pop(0)
    assert (kernels / "common.cuh").is_file()
    for name in (p.name for p in kernels.glob("*.cu")):
        for header in EXTRA_HEADERS.get(name[:-3], ()):
            assert (kernels / header).is_file(), (name, header)


# -------------------------------------------------------------- the wiring

def test_no_carried_module_still_imports_a_carried_name_from_the_engine():
    engine_names = "|".join(
        list(resync.CORE_MODULES) + ["kernels"])
    pattern = re.compile(rf"gpuwm\.core\.({engine_names})\b|gpuwm\.verify\.npref")
    for path in sorted((PACKAGE / "core").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            # The manifest namespace is the one sanctioned occurrence.
            assert "MODULE_KEY_ROOT" in text.splitlines()[line - 1], (
                f"{path.name}:{line} reaches the engine's {match.group(0)}")


def test_the_manifest_namespace_did_not_move():
    """A carried loader keys its images where every receipt already looks.

    `certify/kernel_manifest.record_module` files a second, differing image
    under a deterministic suffix rather than replacing the first, so the
    engine's loader and this one coexist under one root without either losing
    an entry.  Moving the root would orphan the pins instead.
    """

    text = (PACKAGE / "core" / "kernels" / "__init__.py").read_text(encoding="utf-8")
    assert 'MODULE_KEY_ROOT = "gpuwm.core.kernels"' in text


def test_the_kernel_sources_carry_no_rewiring():
    """Byte-for-byte, comments included: the string is hashed, not read."""

    for path in sorted((PACKAGE / "core" / "kernels").iterdir()):
        if path.suffix not in (".cu", ".cuh"):
            continue
        assert "arwen_global" not in path.read_text(encoding="utf-8"), path.name


def test_the_verbatim_suffixes_are_the_device_sources():
    assert resync.VERBATIM_SUFFIXES == {".cu", ".cuh"}


# ------------------------------------------------------------- the re-cut

@needs_the_source_tree
def test_the_carried_core_re_cuts_cleanly_from_the_recorded_revision():
    """A dry run merges every carried file into a copy and reports the verdict.

    This is the claim the carve table makes -- that what is here is what the
    source tree has, mergeable file by file -- turned into a command whose
    output a reader can check.  It writes nothing.
    """

    revision = _recorded_revision()
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--worktree", SOURCE_WORKTREE,
         "--to", revision, "--unit", "core", "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "NEED A DECISION" not in out, out
    assert "0 added" in out, out
    for name in resync.CORE_MODULES:
        assert f"ok  core/{name}.py" in out, (name, out)
    assert "ok  core/npref.py" in out, out
    assert "ok  core/kernels/__init__.py" in out, out
    for kernel in resync.CORE_KERNELS:
        assert f"ok  core/kernels/{kernel}" in out, (kernel, out)


@needs_the_source_tree
def test_every_carried_kernel_is_the_source_tree_s_under_the_stated_rule():
    """The source tree's bytes, with the prose rule applied and nothing else.

    Not "byte for byte" any more, and the difference is the point: three of
    the fifteen device sources carried text that cannot ship on a public
    index, so `tools/resync_from_owner.py` rewrites those five lines when it
    cuts them.  Reproducing the carried file therefore means applying the
    same rule to the source tree's blob, which is what this does.  A hand
    edit would fail here, because a hand edit is not a rule.
    """

    revision = _recorded_revision()
    changed = []
    for kernel in resync.CORE_KERNELS:
        blob = subprocess.run(
            ["git", "show", f"{revision}:gpuwm/core/kernels/{kernel}"],
            cwd=SOURCE_WORKTREE, capture_output=True).stdout
        assert blob, kernel
        source = blob.decode("utf-8")
        carved = source
        for pattern, replacement in resync.CORE_PROSE:
            carved = re.sub(pattern, replacement, carved)
        for name, old, new in resync.DEVICE_EXACT:
            if name == kernel:
                carved = carved.replace(old, new)
        here = (PACKAGE / "core" / "kernels" / kernel).read_bytes()
        assert carved.encode("utf-8") == here, kernel
        if carved != source:
            changed.append((kernel, len(carved.encode("utf-8")) - len(blob)))
    # NAMED, so that a rule quietly growing a sixth site is a red row rather
    # than a number nobody looks at.
    assert sorted(changed) == [("gf.cu", -2), ("glibc_flt32.cuh", -11),
                               ("ntiedtke.cu", -4)], sorted(changed)


def test_no_carried_device_source_carries_what_cannot_ship():
    """The post-condition, on the tree, without the source tree.

    `tools/resync_from_owner.py` raises during a cut when a device source
    still carries one of these; this is the same question asked of what is
    actually here, so a file edited by hand after the cut cannot pass.
    """

    for path in sorted((PACKAGE / "core" / "kernels").iterdir()):
        if path.suffix not in (".cu", ".cuh"):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, what in resync.DEVICE_FORBIDDEN:
            match = re.search(pattern, text)
            assert match is None, (
                f"{path.name}:{text[:match.start()].count(chr(10)) + 1} "
                f"carries {what} ({match.group(0)!r})")


def test_the_device_rule_is_the_word_table_plus_one_named_string():
    """A rule that grew would move kernel digests with no measurement.

    The word table is shared with the carried Python, so it is checked by
    what it produces rather than by its length; the exact-string table is
    the one that can silently gain a row, and it is asserted here.
    """

    assert len(resync.DEVICE_EXACT) == 1
    name, old, new = resync.DEVICE_EXACT[0]
    assert name == "glibc_flt32.cuh"
    assert len(old) - len(new) == 11
    assert resync.VERBATIM_SUFFIXES == {".cu", ".cuh"}


@needs_the_source_tree
def test_the_existing_rows_still_merge_three_ways():
    """The core table is an addition; the model and the suite are unchanged.

    Run against the revision BEFORE the one recorded, which is a real re-cut
    with real changed files, so this exercises the merge rather than a no-op.
    """

    text = (REPO / "SOURCE.md").read_text(encoding="utf-8")
    previous = re.search(r"^- Previous:\s*`([0-9a-f]{7,40})`", text, re.M).group(1)
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--worktree", SOURCE_WORKTREE,
         "--from", previous, "--to", _recorded_revision(),
         "--unit", "model", "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    assert "the model:" in proc.stdout, proc.stdout
    assert "changed" in proc.stdout


def test_no_shipped_module_names_a_carried_file_at_the_source_tree_s_path():
    """The path a receipt prints is the file that ran.

    THE BREAKAGE THIS PREVENTS, measured on the tree 2026-09-10 from an
    install of the built wheel.  `physics/builtin_adapters._contract` writes
    the cumulus `scheme_identity` into every run receipt and
    `physics/native_options.validate` repeats it in the refusal a user reads;
    both named `gpuwm/core/kernels/gf.cu` and `gpuwm/core/kernels/ntiedtke.cu`,
    which exist in an install and are NOT the files that ran, and
    `gpuwm/arwen_global/physics/arwen_massflux.py`, which exists in no install
    at all.  The import rewiring moves DOTTED names; a slash-spelled path is
    prose to it, so nothing here saw them.

    The table is the carve's own, so a path naming a file that STAYS on the
    engine -- `gpuwm/core/dycore.py`, `gpuwm/core/kernels/thompson.cu`,
    `gpuwm/core/kernels/p3.cu` -- is not a hit: it still names where that file
    is.  The suite is out of scope for the same reason the rule is: `tests/`
    names source paths as data.
    """

    hits: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(PACKAGE).as_posix()
        for pattern, dest in resync.SLASH_PATHS:
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count(chr(10)) + 1
                hits.append(f"{rel}:{line}: {match.group(0)} "
                            f"(this carve takes it to {dest})")
        for pattern, what in resync.CARRIED_PATH_FORBIDDEN:
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count(chr(10)) + 1
                hits.append(f"{rel}:{line}: {what}")
    assert hits == [], (
        "a shipped module names a file this package carries at the path the "
        "source tree keeps it under, so a reader following the path lands on "
        "bytes that did not run:" + chr(10) + "  "
        + (chr(10) + "  ").join(hits))


@needs_the_source_tree
def test_every_carried_python_file_is_the_source_tree_s_under_the_stated_rules():
    """The kernels' test, for the half of the carve written in Python.

    The dry run above says the carried files MERGE cleanly, which with one
    revision on both sides is a merge of a file into itself: it cannot see a
    carried file whose bytes are not what the rules produce.  This runs the
    extraction the tool runs and compares bytes, which is the claim the carve
    actually makes.  A hand edit fails here, because a hand edit is not a
    rule.
    """

    revision = _recorded_revision()
    tmp = Path(tempfile.mkdtemp(prefix="carve-check-"))
    try:
        resync.extract(Path(SOURCE_WORKTREE), revision, tmp,
                       resync.CORE_CARVE, None,
                       resync.CORE_REWIRE + resync.CORE_PROSE,
                       resync.CORE_PRESERVE, resync.SLASH_PATHS)
        produced = tmp / "arwen_global"
        differ = []
        for path in sorted(produced.rglob("*.py")):
            rel = path.relative_to(produced).as_posix()
            here = PACKAGE / rel
            if not here.is_file():
                differ.append(rel + " (absent here)")
            elif here.read_bytes() != path.read_bytes():
                differ.append(rel + " (" + str(len(here.read_bytes()))
                              + " bytes here, " + str(len(path.read_bytes()))
                              + " from the rules)")
        assert differ == [], differ
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_slash_rule_is_derived_from_the_carve_and_not_a_written_list():
    """A hand-written list drifts from the table it is supposed to follow."""

    sources = {src for src, _ in resync.CARVE + resync.CORE_CARVE}
    assert len(resync.SLASH_PATHS) == len(sources)
    for pattern, dest in resync.SLASH_PATHS:
        assert dest.startswith("arwen_global"), dest
    # The three the carve does NOT take, named so that a prefix rule cannot
    # arrive here quietly: each of these still lives where it is written.
    for stray in ("gpuwm/core/dycore.py", "gpuwm/core/kernels/thompson.cu",
                  "gpuwm/core/kernels/p3.cu"):
        assert not any(re.match(pattern, stray)
                       for pattern, _ in resync.SLASH_PATHS), stray
