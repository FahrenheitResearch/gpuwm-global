# Where this package's model source comes from

`src/arwen_global/` is not written here. It is cut from the model's own source
tree, where the science work happens, and re-cut from a newer revision when
something lands there. This file records which revision the package currently
carries, so that "is that fix in the package?" is a question with an answer
rather than a diff somebody has to take by eye.

## The revision this package carries

- Revision: `857cb277c712370d14704eefd989fb4979a9e960`
- Dated: 2026-09-09T20:43:47-07:00
- Previous: `f875866d433cc25488cafaaf69f74c80d7e6e9da`, 2026-09-07T08:48:48-07:00, 4 commits earlier

The same revision is stated in `pyproject.toml` beside the version, so a
reader who has only an installed wheel's metadata can still name the source it
was built from.

The source tree is not public and its location is not recorded here: it is
given to the re-cut script on the command line, or in
`GPUWM_GLOBAL_SOURCE_WORKTREE`.

## What is cut, and where it lands

| Source path | Lands as |
|---|---|
| `gpuwm/arwen_global/` | `src/arwen_global/` |
| `gpuwm/global_spectral/` | `src/arwen_global/spectral/` |
| `gpuwm/core/global_physics_registry.py` | `src/arwen_global/physics/registry.py` |
| `gpuwm/static/rows.py` | `src/arwen_global/statics_rows.py` |
| `gpuwm/data/arwen_global/` | `src/arwen_global/data/` |

The experiments under `src/arwen_global/configs/` and the suite under `tests/`
came with the first cut and are maintained here; they are not re-cut, because
both have diverged on purpose. The suite runs against an installed engine
wheel, which the source tree's copy of it does not.

## How a re-cut is done

    python tools/resync_from_owner.py --worktree <dir> --to <revision>

It reads the revision above as the merge base, extracts both revisions, puts
BOTH through the same import and command rewiring, and three-way merges each
file against the copy in this repository. Fixes made here survive; changes
made there arrive; anything both sides touched comes back with conflict
markers in the file, which is the one case a person has to read.

**Why a merge and not a copy.** This package carries fixes the source tree
does not have: the engine-boundary seam, the door table, the refusal exit
codes, four resolvers that stopped assuming the model lives inside another
distribution. A re-copy throws every one of them away silently. The first
re-cut, to the revision above, had 46 files changed and 12 added upstream
against 40 files carrying fixes here; by eye that is how a fix lands in one
tree and not the other. It found four defects that would otherwise have
shipped, and they are listed in the commit that took it.

After a re-cut: rebuild the companion door bundle at the same revision, run
the suite against a freshly built wheel on both platforms, update this file
and the comment in `pyproject.toml`, and commit. The script writes neither
the record nor the commit, on purpose: a recorded revision should mean the
suite passed on it, not that a script copied it.
