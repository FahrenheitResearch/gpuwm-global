"""The private names the provenance gates refuse, read at run time and never shipped.

THE BREAKAGE THIS PREVENTS (15 Nov 2026): the gates that keep private names
out of the published artefacts (`tests/test_no_provenance.py`,
`tools/provenance_sweep.py`, `tools/check_case_token_leakage.py`) and the
carve that rewrites them out of carried files (`tools/resync_from_owner.py`)
spelled those names themselves, split across string literals so that their
own rules would not see them.  Every one of those files ships in the source
package, so the published sdist carried the list the gates exist to keep
out.  A scan of the built artefacts that joins split literals reads them
whole, and so does anyone who reads the file.

So the names live in a JSON file that is never committed and never packaged,
and every consumer reads it through `load()`:

  * the path in the environment variable `ARWEN_GLOBAL_PRIVATE_NAMES`, or
  * `.private-names.json` at the repository root (ignored by git and
    excluded from the sdist by MANIFEST.in).

When neither is there, `load()` returns None.  The gates then run every rule
that needs no private name and skip, by name and with this module in the
reason, the rules that do; the carve refuses to cut, because a cut without
its identity rules would carry the names into the package.

The file's keys (every one optional; a missing key is an empty rule):

  tool_names        names of tools a file might have been written with
  tooling_dir       the directory one of those tools keeps working notes in
  personal_names    personal names, matched case-sensitively
  machine_pattern   a regular expression for the private machines' hostnames
  machine_examples  hostnames the machine rule must catch (its self-test)
  carried_host      the short hostname carried physics comments name
  carried_host_alias  the long form of the same machine
  case_tokens       private case names, matched case-insensitively
  private_folder_patterns  regular expressions for private working folders
  private_folder_examples  folder names those patterns must catch (self-test)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV = "ARWEN_GLOBAL_PRIVATE_NAMES"
LOCAL = Path(__file__).resolve().parents[1] / ".private-names.json"

#: The reason a skipped rule gives, so a reader of a test run knows how to
#: turn it on.
SKIP_REASON = (f"the private names are not on this machine (set {ENV} to the "
               f"private names file, or put it at {LOCAL.name} in the "
               "repository root); see tools/private_names.py")


def path() -> Path | None:
    """The private names file this run would read, or None."""

    named = os.environ.get(ENV, "").strip()
    if named:
        return Path(named).expanduser()
    return LOCAL if LOCAL.is_file() else None


def load() -> dict[str, Any] | None:
    """The private names, or None when no file is on this machine.

    A file that is named but unreadable is an error rather than a skip: a
    gate that quietly turned off because of a typo in a path is the failure
    this module exists to make visible.
    """

    where = path()
    if where is None:
        return None
    data = json.loads(where.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{where}: the private names file is a JSON object")
    return data


def strings(data: dict[str, Any] | None, key: str) -> tuple[str, ...]:
    """One list-valued key as a tuple of non-empty strings (empty when absent)."""

    if not data:
        return ()
    value = data.get(key) or ()
    if isinstance(value, str):
        value = (value,)
    return tuple(str(item) for item in value if str(item))


def string(data: dict[str, Any] | None, key: str) -> str:
    """One string-valued key ("" when absent)."""

    if not data:
        return ""
    return str(data.get(key) or "")
