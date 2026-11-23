"""The folders each account on a machine uses, set and named in one place.

Two accounts sharing one install (one virtual environment, one machine) used
to collide on the first day.  Two things were the cause, and both are named
here so a reader of ``doctor`` sees which folder is whose:

* **The fetch lock folder.**  Every engine download takes a lock file under
  one folder.  The engine's default is ``<temp>/gpuwm/locks``
  (``gpuwm.fetch_guard.lock_root``), shared by every account on the machine.
  The first account to fetch creates it with that account's permissions,
  and every other account's fetch then dies with ``PermissionError`` on a
  lock file it cannot create.  The engine reads ``GPUWM_FETCH_LOCK_ROOT``
  before its default, so this package sets that variable, for its own
  process and the engine calls inside it, to a folder in the account's own
  home: :func:`account_lock_root`.  An operator who set the variable keeps
  theirs.  Locks then serialise fetches within one account, which is what
  they are for; two accounts that write the SAME output folder (a shared,
  group-writable cases folder) set ``GPUWM_FETCH_LOCK_ROOT`` to one shared,
  group-writable folder instead, and lock each other out as before.

  The engine's own downloads run from a shell (``gpuwm fetch-geog`` and
  ``gpuwm fetch``; ``gpuwm fetch-bridges`` stages into the account's own
  versioned folder and was measured unaffected, 2026-11-22) do not run
  through this package, so on a machine where another account owns the
  engine's default folder they need the variable exported in the shell.  :func:`engine_lock_root_state` says when that is
  the case, and ``doctor`` reports it as a gap naming those commands.

* **The door folder.**  The doors this package publishes stage into
  ``~/.gpuwm/global-doors`` by default, a folder per account
  (:func:`arwen_global.doors.companion_door_dir`); ``ARWEN_GLOBAL_DOOR_DIR``
  points every account at one shared copy instead.  The engine's own bridges
  stage under ``~/.gpuwm/bridges`` and the engine has no shared-folder
  override for them, so each account stages its own with
  ``gpuwm fetch-bridges``.

This module never changes a variable the operator set, and every folder it
chooses is printed by ``doctor``.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

__all__ = [
    "LOCK_ROOT_ENV",
    "account_lock_root",
    "bind_account_lock_root",
    "engine_default_lock_root",
    "engine_lock_root_state",
    "lock_root_binding",
]

#: The engine's own variable (``gpuwm.fetch_guard.LOCK_ROOT_ENV``), spelled
#: here so this module imports nothing from the engine at start-up;
#: tests/test_arwen_global_accounts.py holds the two spellings together.
LOCK_ROOT_ENV = "GPUWM_FETCH_LOCK_ROOT"

#: What :func:`bind_account_lock_root` did in this process: ``"operator"``
#: when the variable was already set, ``"this program"`` when it set it,
#: ``None`` before it ran.
_BOUND_BY: str | None = None


def account_lock_root() -> Path:
    """This account's own lock folder: ``~/.gpuwm/locks``.

    Beside the engine's ``~/.gpuwm/bridges`` and this package's
    ``~/.gpuwm/global-doors``, in the one place every account can write.
    """

    return Path.home() / ".gpuwm" / "locks"


def engine_default_lock_root() -> Path:
    """The folder the engine locks in when nothing names another one.

    The same rule as ``gpuwm.fetch_guard.lock_root`` with the variable
    unset: ``%PROGRAMDATA%`` on Windows, the system temporary folder
    elsewhere, then ``gpuwm/locks``.
    """

    if os.name == "nt":
        base = Path(os.environ.get("PROGRAMDATA", tempfile.gettempdir()))
    else:
        base = Path(tempfile.gettempdir())
    return base / "gpuwm" / "locks"


def bind_account_lock_root() -> Path:
    """Point this process's engine fetches at this account's lock folder.

    Called once at the top of the console script, beside
    :func:`arwen_global.doors.bind_companion_doors`.  It sets
    ``GPUWM_FETCH_LOCK_ROOT`` only when the operator has not, and returns
    the folder the engine will lock in either way.
    """

    global _BOUND_BY
    current = os.environ.get(LOCK_ROOT_ENV)
    if current:
        if _BOUND_BY is None:
            _BOUND_BY = "operator"
        return Path(current).expanduser()
    root = account_lock_root()
    os.environ[LOCK_ROOT_ENV] = str(root)
    _BOUND_BY = "this program"
    return root


def lock_root_binding() -> tuple[Path, str]:
    """``(folder, who set it)`` for this process, binding it if needed."""

    root = bind_account_lock_root()
    return root, _BOUND_BY or "this program"


def _writable_dir(path: Path) -> bool:
    """Whether this account can create a file in ``path`` (or, when it does
    not exist yet, in the nearest folder above it that does)."""

    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return False
        probe = parent
    return probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)


def engine_lock_root_state() -> tuple[str, Path]:
    """Whether the engine's OWN commands can take their locks here.

    ``("free", path)`` when the default folder is absent or writable by
    this account, ``("blocked", path)`` when it exists and this account
    cannot create a lock in it, which is the state that stops
    ``gpuwm fetch-geog`` and ``gpuwm fetch`` with ``PermissionError``
    until ``GPUWM_FETCH_LOCK_ROOT`` is exported.
    """

    path = engine_default_lock_root()
    return ("free" if _writable_dir(path) else "blocked"), path
