"""The README's install section, held to the installed engine.

THE BREAKAGE THIS PREVENTS: the install block said `fetch-doors` stages
"the eight observation binaries only this package publishes" while the
published 2.8 engines carry `rw_asos` and `rw_goes` in their own bundle, so
on every 2.8 install `fetch-doors` staged six and `doctor` named the engine
as the publisher of the other two.  A reader counting what arrived against
the page saw two doors missing that were not missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_CLAIM = re.compile(
    r"Of\s+the\s+(\w+)\s+observation\s+doors,\s+(.+?)\s+arrives?\s+in\s+the\s+"
    r"engine's\s+own\s+bundle\s+on\s+(\d+\.\d+)\.x,\s+so\s+`fetch-doors`\s+"
    r"stages\s+the\s+other\s+(\w+)\.",
    re.DOTALL)


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _observation_doors(text: str) -> list[str]:
    """The doors the README's "observation doors" bullet lists."""

    match = re.search(r"\*\*The observation doors\*\*(.+?)\n-", text, re.DOTALL)
    assert match, "the README no longer lists the observation doors"
    return re.findall(r"`(rw_[a-z0-9]+)`", match.group(1))


def test_the_install_section_says_which_doors_the_engine_bundle_carries():
    from importlib.metadata import PackageNotFoundError, version as dist_version

    from arwen_global import doors

    text = _readme()
    claim = _CLAIM.search(text)
    assert claim, "the install section does not say which doors the engine carries"
    total, named, series, rest = claim.groups()
    observation = _observation_doors(text)
    assert _WORDS[total] == len(observation), (total, observation)

    try:
        version = dist_version("gpuwm")
    except PackageNotFoundError:
        version = ""
    if not version.startswith(series + "."):
        pytest.skip(f"the README states the {series}.x engines and the "
                    f"installed engine is {version or 'unreadable'}")
    from_engine = {name for name in observation
                   if doors.publisher(name) == doors.ENGINE_BUNDLE}
    assert set(re.findall(r"`(rw_[a-z0-9]+)`", named)) == from_engine
    assert _WORDS[rest] == len(observation) - len(from_engine)
    staged = {door.name for door in doors.companion_doors()
              if not door.built_here}
    assert staged == set(observation) - from_engine, sorted(staged)
    # The libraries this package builds from its own rust/ workspace are
    # staged by the same command, and the install section names each one.
    for door in doors.companion_doors():
        if door.built_here:
            assert f"`{door.name}`" in text, door.name


def test_the_fetch_doors_line_claims_no_door_the_engine_publishes():
    line = next(line for line in _readme().splitlines()
                if line.startswith("gpuwm-global fetch-doors"))
    assert "only this package publishes" not in line, line
