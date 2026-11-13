"""The member engine: forecast members from one analysis, outside the
assimilation path, each described by a replay recipe of a few kilobytes.

A member is the base state (one analysis at the member truncation) plus a
weighted sum of perturbation draws, stepped through the shared model of
:class:`~arwen_global.da.ensemble.GlobalEnsemble`.  A draw is named by
its stream alone: draw ``k`` is :func:`~arwen_global.da.perturbations.
draw_perturbation` from ``member_rng(options.seed, k, "initial")`` after
the one integer :meth:`GlobalEnsemble.from_state` takes from that stream
first, so member ``k`` of a set started here is member ``k`` of the
filter's ensemble with the same options, and it can be redrawn alone.  A
recipe with no draws is the control; one draw of weight 1 is an ordinary
member; several draws with other weights are a blend, the form an aimed
state takes.

What a recipe holds (:data:`RECIPE_SCHEMA`, JSON): the identity of the
config the member ran under, the digest of the base state before any draw,
the perturbation options, the draws and weights, the conservation targets
the member's fixers held, the steps it ran, the digest of its state at the
end (and at any step it recorded), and the software stack and card it ran
on.  It holds no array.  :func:`replay` rebuilds the member from a config
and a recipe and refuses unless the base digest matches before the first
step and the end digest matches after the last.

Exact twins.  :meth:`MemberSet.branch` starts a one-member set from a
member checkpoint: the same state, its own second time level and its
recipe's conservation targets, plus one deliberate change (draws added at
the branch step, a different config, or both).  A child whose config hash
differs from the parent checkpoint's is refused unless the caller passes
``allow_config_branch=True``, and the recipe records that the opt-in was
given (the run door's ``--branch-from-other-config`` is the same rule).
The child's recipe carries its parent's recipe inline, so replaying the
twin replays the parent to the branch step first.

Failure isolation.  :meth:`MemberSet.step_all` steps each living member
under its own conservation targets and second time level, exactly as
:meth:`GlobalEnsemble.step_all` does; a member whose step raises is
marked dead with its error and the others go on.  A dead member's state
is the one that entered the failed step.

Not here: the incremental analysis update (a forecast member has none),
a dated surface series (refused: the run door applies it between steps
and a member set does not yet), and members whose sea-ice or land planes
differ (refused by the shared model, see ``SHARED_SURFACE_BREAKAGE``).
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checkpoint import (
    _array_hash,
    bundle_arrays,
    checkpoint_metadata,
    read_checkpoint,
    state_from_checkpoint,
    trajectory_from_checkpoint,
)
from .constants import SPECTRAL_FIELDS
from .da.ensemble import ENSEMBLE_SCHEMA, GlobalEnsemble, MemberTargets, StepTiming, member_checkpoint_path
from .da.options import EnsembleOptions
from .da.perturbations import draw_perturbation, member_rng, perturbed_state

#: The recipe format.
RECIPE_SCHEMA = "gpuwm.arwen-global-member-recipe/v1"
#: The stream purpose an initial draw comes from (the filter's own).
DRAW_PURPOSE = "initial"
#: Recipe file name of member ``label`` beside its checkpoint.
RECIPE_SUFFIX = ".recipe.json"


class ReplayMismatch(ValueError):
    """A replay reached a digest that is not the recipe's."""


# -- digests ---------------------------------------------------------------

def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def arrays_digest(arrays: dict) -> str:
    """SHA-256 of the canonical table of per-array hashes (the hashes a
    checkpoint carries): the identity of a state's arrays, independent of
    the config hash, the run trackers and the code path that produced
    them, so a member and a run-door twin compare on it."""
    table = {name: _array_hash(np.asarray(array)) for name, array in sorted(arrays.items())}
    return hashlib.sha256(_canonical(table)).hexdigest()


def state_identity(state, cfg, to_numpy, trajectory=None) -> dict:
    """The two digests of ``state``: ``arrays_sha256`` (:func:`arrays_digest`
    of every checkpointed array, the second time level included) and
    ``checkpoint_sha256`` (the ``self_sha256`` the member's checkpoint
    under ``cfg`` with zero run trackers carries; the file
    :meth:`GlobalEnsemble.write` writes for it has this hash)."""
    arrays = bundle_arrays(state, to_numpy, trajectory)
    metadata = checkpoint_metadata(
        arrays, step=int(state.step), time_s=float(state.time_s),
        physics_state_schema=state.physics_state.schema,
        physics_metadata=state.physics_state.metadata,
        config_hash=cfg.config_hash, trackers=None,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    return {
        "step": int(state.step),
        "time_s": float(state.time_s),
        "arrays_sha256": arrays_digest(arrays),
        "checkpoint_sha256": metadata["self_sha256"],
    }


def base_digest(base, to_numpy) -> str:
    """The digest of the base state before any draw (no second level)."""
    return arrays_digest(bundle_arrays(base, to_numpy, None))


def software_stack(backend) -> dict:
    """The versions and the card a member ran on: a recipe names them,
    because identity across cards and stacks is not shown."""
    from importlib import metadata as md

    def _version(name):
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001 - an absent distribution is recorded as absent
            return None

    row = {
        "gpuwm-global": _version("gpuwm-global"),
        "gpuwm": _version("gpuwm"),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "backend": getattr(backend, "name", "numpy"),
    }
    if row["backend"] == "cupy":
        cp = backend.xp
        row["cupy"] = cp.__version__
        try:
            row["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
            row["cuda_driver"] = int(cp.cuda.runtime.driverGetVersion())
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = props.get("name", b"")
            row["device"] = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:  # noqa: BLE001 - the card's name is a record, not a gate
            pass
    return row


# -- the config a recipe names ---------------------------------------------

def config_record(cfg) -> dict:
    return {
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "truncation": int(cfg.truncation),
        "dt_s": float(cfg.dt_s),
        "integrator": str(cfg.integrator),
        "semi_implicit_scheme": str(cfg.semi_implicit_scheme),
        "physics_mode": str(cfg.physics_mode),
    }


def _refuse_unsupported(cfg) -> None:
    if getattr(cfg, "surface_series", None):
        raise ValueError(
            "the member engine does not apply a dated surface series between "
            "steps (the run door does); run this config through the run door"
        )


# -- draws -----------------------------------------------------------------

def normalize_draws(draws) -> list[dict]:
    """``[{"member": k, "weight": w}, ...]`` checked and in the given
    order (the sum is taken in that order, so the order is identity)."""
    out = []
    for row in draws or ():
        if isinstance(row, int) and not isinstance(row, bool):
            row = {"member": row, "weight": 1.0}
        member = row["member"]
        if isinstance(member, bool) or int(member) != member or member < 0:
            raise ValueError("a draw's member must be a nonnegative whole number")
        weight = float(row.get("weight", 1.0))
        if not math.isfinite(weight):
            raise ValueError("a draw's weight must be finite")
        out.append({"member": int(member), "weight": weight})
    return out


def draw_increment(model, transform, base_atmosphere, options: EnsembleOptions, member: int):
    """Draw ``member``'s perturbation exactly as
    :meth:`GlobalEnsemble.from_state` draws it (the stream's first integer
    is the seed the manifest records, the draw follows): ``(increments,
    record, seed_k)``."""
    rng = member_rng(options.seed, int(member), DRAW_PURPOSE)
    seed_k = int(rng.integers(0, 2 ** 31 - 1))
    increments, record = draw_perturbation(model, transform, base_atmosphere, options, rng)
    return increments, record, seed_k


def combined_increment(model, transform, base_atmosphere, options, draws):
    """The weighted sum of ``draws`` in their order, or None for no draw.
    One draw of weight 1 is that draw's increments unscaled (bit for bit
    the filter's member)."""
    draws = normalize_draws(draws)
    # A draw of weight zero adds nothing, exactly: it is recorded in the
    # recipe and never summed (a product with 0.0 can carry a negative
    # zero that a sum would write into a zero coefficient's sign bit).
    live = [row for row in draws if row["weight"] != 0.0]
    if not live:
        return None, []
    draws = live
    total = None
    records = []
    for index, row in enumerate(draws):
        if index:
            # Between draws, as the filter's construction releases the
            # memo between members; never before the first, so a single
            # draw meets the model exactly as the filter's member does.
            model.release_syntheses()
        inc, _record, seed_k = draw_increment(model, transform, base_atmosphere, options, row["member"])
        records.append({"member": row["member"], "seed": seed_k})
        if len(draws) == 1 and row["weight"] == 1.0:
            return inc, records
        scaled = {name: inc[name] * row["weight"] for name in SPECTRAL_FIELDS}
        total = scaled if total is None else {name: total[name] + scaled[name] for name in SPECTRAL_FIELDS}
    return total, records


# -- the member set ----------------------------------------------------------

@dataclass
class MemberSet:
    """Forecast members sharing one model, each with its recipe.

    ``ensemble`` is a :class:`GlobalEnsemble` (its members, targets and
    second time levels are the set's); ``recipes`` one dict per member;
    ``dead`` maps a member's position to the error its step raised."""

    ensemble: GlobalEnsemble
    recipes: list[dict]
    labels: list[str]
    dead: dict[int, str] = field(default_factory=dict)
    #: Recorded digests per member: ``{position: [identity, ...]}``.
    marks: dict[int, list] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.ensemble.size

    @property
    def cfg(self):
        return self.ensemble.cfg

    # -- construction -------------------------------------------------------

    @classmethod
    def start(cls, cfg, model, transform, base, options: EnsembleOptions, specs, *,
              config_source_sha256: str | None = None, labels=None) -> "MemberSet":
        """One member per entry of ``specs`` around ``base`` (the analysis
        at the member truncation, before any draw).  An entry is a list of
        draws (:func:`normalize_draws`; an empty list is the control) or a
        whole number ``k``, which is draw ``k`` at weight 1.  The set of
        ``specs = range(N)`` equals ``GlobalEnsemble.from_state`` with
        ``options.members = N`` member for member, bit for bit."""
        _refuse_unsupported(cfg)
        if int(transform.truncation) != int(options.truncation):
            raise ValueError(
                f"the member transform is T{transform.truncation} but the options say "
                f"T{options.truncation}"
            )
        backend = transform.backend
        model.enforce(base)
        base_sha = base_digest(base, backend.to_numpy)
        members, targets, seeds, recipes, names = [], [], [], [], []
        for position, spec in enumerate(specs):
            draws = normalize_draws([spec] if isinstance(spec, int) and not isinstance(spec, bool) else spec)
            increments, records = combined_increment(model, transform, base.atmosphere, options, draws)
            state = perturbed_state(model, transform, base, increments or {})
            state.atmosphere.time_s = base.atmosphere.time_s
            state.atmosphere.step = base.atmosphere.step
            target = MemberTargets(
                mass_pa=model.initialize_mass_target(state.atmosphere),
                total_water_kg_m2=model.initialize_water_target(state),
            )
            model.release_syntheses()
            members.append(state)
            targets.append(target)
            seeds.append(records[0]["seed"] if records else -1)
            label = (labels[position] if labels is not None else
                     ("control" if not draws else
                      (f"m{draws[0]['member']:03d}" if len(draws) == 1 and draws[0]["weight"] == 1.0
                       else f"blend{position:03d}")))
            names.append(label)
            recipes.append({
                "schema": RECIPE_SCHEMA,
                "label": label,
                "config": {**config_record(cfg), "source_sha256": config_source_sha256},
                "base": {"step": int(base.step), "time_s": float(base.time_s), "arrays_sha256": base_sha},
                "perturbation": {
                    "family": "arwen_global.da.perturbations.draw_perturbation",
                    "stream": f"member_rng(seed, member, {DRAW_PURPOSE!r}), one integer taken first",
                    "options": options.identity(),
                    "draws": draws,
                    "draw_seeds": [r["seed"] for r in records],
                },
                "targets": {"mass_pa": target.mass_pa, "total_water_kg_m2": target.total_water_kg_m2},
                "branch": None,
                "marks": [],
            })
        ensemble = GlobalEnsemble(
            cfg=cfg, model=model, transform=transform, options=options,
            members=members, targets=targets, seeds=seeds,
            provenance={"schema": ENSEMBLE_SCHEMA, "made_by": "MemberSet.start",
                        "base_step": int(base.step), "base_time_s": float(base.time_s),
                        "options": options.identity(), "base_arrays_sha256": base_sha},
        )
        ensemble._check_shared_surface()
        return cls(ensemble=ensemble, recipes=recipes, labels=names)

    @classmethod
    def branch(cls, child_cfg, model, transform, parent_checkpoint, parent_recipe: dict, *,
               draws=(), options: EnsembleOptions | None = None, change: str = "",
               allow_config_branch: bool = False, config_source_sha256: str | None = None,
               label: str | None = None) -> "MemberSet":
        """An exact twin of the member ``parent_recipe`` describes, started
        from its checkpoint ``parent_checkpoint`` (written by
        :meth:`write`): the same state and second time level, the parent's
        conservation targets, plus ``draws`` (drawn around the branch
        state with ``options``, default the parent's) and/or the config
        change ``child_cfg`` carries.  ``change`` says in words what
        differs.  A child config hash that is not the parent checkpoint's
        is refused unless ``allow_config_branch``; the recipe records the
        opt-in."""
        _refuse_unsupported(child_cfg)
        backend = transform.backend
        header, arrays = read_checkpoint(
            parent_checkpoint, expected_config_hash=None,
            semi_implicit_scheme=child_cfg.semi_implicit_scheme, integrator=child_cfg.integrator,
        )
        parent_hash = header["config_hash"]
        differs = parent_hash != child_cfg.config_hash
        if differs and not allow_config_branch:
            raise ValueError(
                "the child config's hash differs from the parent checkpoint's; a branch "
                "under another config is an explicit opt-in (allow_config_branch=True, "
                "the run door's --branch-from-other-config), recorded in the recipe"
            )
        recorded = {int(m["step"]): m for m in parent_recipe.get("marks", [])}
        mark = recorded.get(int(header["step"]))
        if mark is None or mark["checkpoint_sha256"] != header["self_sha256"]:
            raise ValueError(
                f"the parent recipe does not record checkpoint {header['self_sha256'][:12]} at step "
                f"{header['step']}; branch from a checkpoint its recipe marks"
            )
        state = state_from_checkpoint(header, arrays, backend)
        trajectory = trajectory_from_checkpoint(header, arrays, backend)
        model.enforce(state)
        options = options or options_from_recipe(parent_recipe)
        draws = normalize_draws(draws)
        branch_sha = arrays_digest(bundle_arrays(state, backend.to_numpy, trajectory))
        records = []
        if draws:
            increments, records = combined_increment(model, transform, state.atmosphere, options, draws)
            new = perturbed_state(model, transform, state, increments)
            new.atmosphere.time_s = state.atmosphere.time_s
            new.atmosphere.step = state.atmosphere.step
            state = new
            model.release_syntheses()
        targets = parent_recipe["targets"]
        target = MemberTargets(mass_pa=float(targets["mass_pa"]),
                               total_water_kg_m2=float(targets["total_water_kg_m2"]))
        name = label or f"{parent_recipe['label']}-twin"
        recipe = {
            "schema": RECIPE_SCHEMA,
            "label": name,
            "config": {**config_record(child_cfg), "source_sha256": config_source_sha256},
            "base": {"step": int(header["step"]), "time_s": float(header["time_s"]), "arrays_sha256": branch_sha},
            "perturbation": {
                "family": "arwen_global.da.perturbations.draw_perturbation",
                "stream": f"member_rng(seed, member, {DRAW_PURPOSE!r}), one integer taken first; drawn around the branch state",
                "options": options.identity(),
                "draws": draws,
                "draw_seeds": [r["seed"] for r in records],
            },
            "targets": dict(targets),
            "branch": {
                "parent": parent_recipe,
                "parent_checkpoint_sha256": header["self_sha256"],
                "parent_config_hash": parent_hash,
                "config_differs": bool(differs),
                "allow_config_branch": bool(allow_config_branch),
                "change": str(change),
            },
            "marks": [],
        }
        ensemble = GlobalEnsemble(
            cfg=child_cfg, model=model, transform=transform, options=options,
            members=[state], targets=[target], seeds=[records[0]["seed"] if records else -1],
            provenance={"schema": ENSEMBLE_SCHEMA, "made_by": "MemberSet.branch"},
            trajectories=[trajectory],
        )
        return cls(ensemble=ensemble, recipes=[recipe], labels=[name])

    # -- integration --------------------------------------------------------

    def alive(self) -> list[int]:
        return [k for k in range(self.size) if k not in self.dead]

    def step_all(self, dt_s: float) -> StepTiming:
        """Advance every living member one step, in member order, as
        :meth:`GlobalEnsemble.step_all` does (own conservation targets,
        own second time level, syntheses released between members).  A
        member whose step raises is marked dead and the rest go on."""
        ens = self.ensemble
        if ens.pending_increments is not None:
            raise ValueError("a member set carries no incremental analysis update")
        model = ens.model
        per_member = []
        ens._sync()
        t0 = time.perf_counter()
        for k in self.alive():
            state = ens.members[k]
            target = ens.targets[k]
            model.set_conservation_targets(target.mass_pa, target.total_water_kg_m2)
            model.set_trajectory_state(ens.trajectory_of(k))
            t_member = time.perf_counter()
            try:
                new_state, _metrics = model.step(state, float(dt_s))
            except (FloatingPointError, ValueError, RuntimeError) as exc:
                # A non-finite prognostic, a refused gate: this member's
                # death is recorded and the others go on.  Anything else
                # (a programming error) propagates.
                self.dead[k] = f"step {int(state.step) + 1}: {type(exc).__name__}: {exc}"
                model.set_trajectory_state(None)
                model.release_syntheses()
                if not self.alive():
                    raise
                continue
            ens.store_trajectory(k, model.trajectory_state())
            model.set_trajectory_state(None)
            model.release_syntheses()
            ens._sync()
            per_member.append(time.perf_counter() - t_member)
            ens.members[k] = new_state
        timing = StepTiming(members=len(per_member), wall_s=time.perf_counter() - t0, per_member_s=per_member)
        ens.timings.append(timing)
        return timing

    def advance(self, steps: int, dt_s: float | None = None, *, mark_every: int = 0,
                observer=None, progress=None) -> list[StepTiming]:
        """``steps`` steps of ``dt_s`` (default the config's); every
        ``mark_every`` steps (and after the last) each living member's
        digests are recorded in its recipe.  ``observer(member_set)`` runs
        after every step."""
        dt_s = float(self.cfg.dt_s if dt_s is None else dt_s)
        timings = []
        for i in range(int(steps)):
            timings.append(self.step_all(dt_s))
            last = i == int(steps) - 1
            if last or (mark_every and (i + 1) % int(mark_every) == 0):
                self.mark()
            if observer is not None:
                observer(self)
            if progress is not None:
                progress(self, timings[-1])
        return timings

    def mark(self) -> None:
        """Record each living member's digests at its current step."""
        ens = self.ensemble
        to_numpy = ens.transform.backend.to_numpy
        for k in self.alive():
            identity = state_identity(ens.members[k], self.cfg, to_numpy, ens.trajectory_of(k))
            marks = self.recipes[k]["marks"]
            if not marks or marks[-1]["step"] != identity["step"]:
                marks.append(identity)

    # -- output -------------------------------------------------------------

    def write(self, outdir, *, which=None) -> list[Path]:
        """Each member's checkpoint (the run format, zero trackers) and its
        recipe beside it; returns the checkpoint paths.  The checkpoint's
        ``self_sha256`` is the recipe's last mark's ``checkpoint_sha256``."""
        from .checkpoint import write_checkpoint

        ens = self.ensemble
        output = Path(outdir)
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for k in (self.alive() if which is None else which):
            state = ens.members[k]
            path = member_checkpoint_path(output, k, state.step)
            path = path.with_name(f"{self.labels[k]}_{path.name}")
            write_checkpoint(
                path, state, config_hash=self.cfg.config_hash,
                to_numpy=ens.transform.backend.to_numpy, trackers=None,
                semi_implicit_scheme=self.cfg.semi_implicit_scheme,
                integrator=self.cfg.integrator, trajectory=ens.trajectory_of(k),
            )
            write_recipe(path.with_name(path.stem + RECIPE_SUFFIX), self.recipes[k])
            paths.append(path)
        return paths

    def finish_recipes(self, stack: dict | None = None) -> None:
        """Stamp the software stack and any death on every recipe."""
        stack = stack or software_stack(self.ensemble.transform.backend)
        for k, recipe in enumerate(self.recipes):
            recipe["stack"] = stack
            recipe["died"] = self.dead.get(k)

    def member_seconds(self) -> dict:
        """Wall per member-step over the set's steps so far (excluding the
        first step of each member, which carries the one-time builds)."""
        rows = [t.per_member_s for t in self.ensemble.timings]
        flat = [s for row in rows[1:] for s in row]
        first = [s for s in (rows[0] if rows else [])]
        return {
            "steps": len(rows),
            "first_step_per_member_s": first,
            "mean_per_member_step_s": float(np.mean(flat)) if flat else None,
            "median_per_member_step_s": float(np.median(flat)) if flat else None,
            "member_step_samples": len(flat),
        }


# -- recipes ---------------------------------------------------------------

def write_recipe(path, recipe: dict) -> Path:
    target = Path(path)
    target.write_text(json.dumps(recipe, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


def read_recipe(path) -> dict:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError(f"{path} is not a member recipe ({RECIPE_SCHEMA})")
    return recipe


def recipe_sha256(recipe: dict) -> str:
    return hashlib.sha256(_canonical(recipe)).hexdigest()


def options_from_recipe(recipe: dict) -> EnsembleOptions:
    identity = dict(recipe["perturbation"]["options"])
    return EnsembleOptions(**identity)


def replay(recipe: dict, build, *, steps: int | None = None, observer=None) -> MemberSet:
    """Re-run the member ``recipe`` describes and refuse on any digest that
    differs.  ``build(config_record) -> (cfg, model, transform, base)``
    supplies the config the recipe names, a fresh model and the base
    state at the member truncation (for a branched recipe, the parent's
    config first).  ``steps`` stops early (at a marked step or not; the
    marks passed are checked).  Returns the replayed one-member set."""
    if recipe.get("branch"):
        parent = recipe["branch"]["parent"]
        branch_step = int(recipe["base"]["step"])
        parent_set = replay(parent, build, steps=branch_step - int(parent["base"]["step"]))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            parent_set.recipes[0]["marks"] = [
                m for m in parent_set.recipes[0]["marks"] if m["step"] == branch_step]
            [path] = parent_set.write(tmp)
            parent_mark = {"label": parent["label"], "marks": parent_set.recipes[0]["marks"]}
            # The parent's model goes before the child's is built: one
            # model on the card at a time.
            del parent_set
            cfg, model, transform, _base = build(recipe["config"])
            _check_config(cfg, recipe)
            twin = MemberSet.branch(
                cfg, model, transform, path, {**parent, **parent_mark},
                draws=recipe["perturbation"]["draws"], options=options_from_recipe(recipe),
                change=recipe["branch"]["change"],
                allow_config_branch=bool(recipe["branch"]["allow_config_branch"]),
                label=recipe["label"],
            )
        if twin.recipes[0]["base"]["arrays_sha256"] != recipe["base"]["arrays_sha256"]:
            raise ReplayMismatch("the replayed parent's state at the branch step is not the recipe's")
        out = twin
    else:
        cfg, model, transform, base = build(recipe["config"])
        _check_config(cfg, recipe)
        out = MemberSet.start(cfg, model, transform, base, options_from_recipe(recipe),
                              [recipe["perturbation"]["draws"]], labels=[recipe["label"]],
                              config_source_sha256=recipe["config"].get("source_sha256"))
        if out.recipes[0]["base"]["arrays_sha256"] != recipe["base"]["arrays_sha256"]:
            raise ReplayMismatch(
                "the base state is not the recipe's: base arrays digest "
                f"{out.recipes[0]['base']['arrays_sha256'][:12]} against {recipe['base']['arrays_sha256'][:12]}")
        if out.recipes[0]["targets"] != recipe["targets"]:
            raise ReplayMismatch("the member's conservation targets are not the recipe's")
    marks = {int(m["step"]): m for m in recipe.get("marks", [])}
    start_step = int(recipe["base"]["step"])
    end_step = max(marks) if marks else start_step
    total = (end_step - start_step) if steps is None else int(steps)
    to_numpy = out.ensemble.transform.backend.to_numpy

    def check(member_set):
        step = member_set.ensemble.members[0].step
        if member_set.dead:
            raise ReplayMismatch(f"the replayed member died: {member_set.dead[0]}")
        if int(step) in marks:
            got = state_identity(member_set.ensemble.members[0], member_set.cfg, to_numpy,
                                 member_set.ensemble.trajectory_of(0))
            want = marks[int(step)]
            if got["arrays_sha256"] != want["arrays_sha256"] or got["checkpoint_sha256"] != want["checkpoint_sha256"]:
                raise ReplayMismatch(
                    f"step {step}: arrays {got['arrays_sha256'][:12]} against the recipe's "
                    f"{want['arrays_sha256'][:12]}")
        if observer is not None:
            observer(member_set)

    out.advance(total, observer=check)
    return out


def _check_config(cfg, recipe) -> None:
    if cfg.config_hash != recipe["config"]["config_hash"]:
        raise ReplayMismatch(
            f"the config supplied hashes {cfg.config_hash[:12]}, the recipe names "
            f"{recipe['config']['config_hash'][:12]}")


__all__ = [
    "DRAW_PURPOSE",
    "MemberSet",
    "RECIPE_SCHEMA",
    "ReplayMismatch",
    "arrays_digest",
    "base_digest",
    "combined_increment",
    "draw_increment",
    "normalize_draws",
    "options_from_recipe",
    "read_recipe",
    "recipe_sha256",
    "replay",
    "software_stack",
    "state_identity",
    "write_recipe",
]
