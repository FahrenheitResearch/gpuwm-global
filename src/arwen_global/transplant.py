"""The scheme transplant: one physics scheme's output carried between twins.

What it is.  Two runs of the same configuration step side by side in one
process.  At every physics call the DONOR run records what one scheme of
the native suite put out, and the RECEIVER run applies the donor's output
in place of computing its own.  That scheme then contributes nothing of
its own to the difference between the two runs: whatever difference is
left travelled through the dynamics and the other schemes.  Compared with
the untouched pair, the gap is that scheme's share.

The two seams (:mod:`arwen_global.physics.native_runtime`):

``cumulus``
    ``_cumulus_result`` returns the scheme's output for one call (the
    theta, vapor, cloud-water and cloud-ice rates, the momentum pair when
    the scheme carries one, its RAINCV and its own diagnostics) before
    anything is applied; ``_apply_cumulus_result`` integrates it.  The
    donor records a copy of the output; the receiver's ``_cumulus_result``
    hands the copy back and its own scheme never runs.  The receiver then
    applies the donor's RATES to its own column with its own arithmetic:
    ``x += dt * rate``.  Where the two columns are bit-equal this is the
    donor's own operation on the donor's own operands, so a receiver
    whose donor is an identical twin stays identical bit for bit; no
    copy-where-equal rule is needed on this seam.

``microphysics``
    ``_microphysics_kernel`` is the Morrison launch alone, which updates
    the batch and the namespace in place and returns nothing.  The donor
    records every array it touches before and after; the receiver applies
    the copy-where-equal rule (:func:`transplanted_value`): where
    the receiver's entry value equals the donor's bit for bit the donor's
    exit value is copied, elsewhere the donor's change
    ``exit - entry`` is added.  The per-call outputs (``rainncv``,
    ``snowncv``, ``graupelncv``, ``sr``, the effective radii) are the
    donor's exit values.  A water species or number concentration the
    added change would leave negative is set to zero and counted.

THE PERSISTENT-STATE RULE.  The scheme's per-call output comes from the
donor; everything the runtime books from that output is the receiver's
own and follows the receiver's column: the integration into its state,
the surface reservoir credited with the column water the call removed as
measured in the receiver's own metric (so the receiver's local water
closure holds whatever the donor did, and the zero a clipped species is
set to is paid for by that reservoir, column by column, not by the global
fixer), the cumulus rain cap at the water the donor's rates removed from
the RECEIVER's column, the accumulators (RAINC, RAINNC, the land
buckets), and the call counters.  The two dynamics forcing lanes the
cumulus schemes read from the previous step
(``native_runtime._measure_dynamics_forcing``) stay the receiver's own and
are kept measured, but no scheme reads them in the receiver while the
transplant is on, so they need no pin.  Effective radii and ``sr`` are the
donor's, because the radiation reads them as the microphysics' output.

Order.  The donor steps first; its records wait in a queue per receiver,
and every record carries the call's band, time and step length, which the
receiver's call must match or it is refused by name.  One donor may feed
several receivers.

What this is not.  Nothing here touches a run made with ``gpuwm-global
run``: the transplant is attached by hand to models built in one process
(:func:`run_experiment`), and a model it is not attached to runs the suite
exactly as before.  The experiment writes its own JSON; it writes no
checkpoint and no run receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCHEMA = "gpuwm.arwen-global-transplant/v1"
STAGES = ("cumulus", "microphysics")

# The cumulus result fields a record carries (native_runtime
# CUMULUS_RATE_FIELDS, CUMULUS_MOMENTUM_FIELDS and RAINCV).
_CUMULUS_ARRAYS = (
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten",
    "rainc",
)
# What the Morrison launch updates in place (native_runtime
# _microphysics_kernel's argument list).
MICROPHYSICS_BATCH_ARRAYS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
)
#: Of those, the ones that may not go below zero.
MICROPHYSICS_NONNEGATIVE = (
    "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng",
)
#: Namespace accumulators: entry plus the donor's change.
MICROPHYSICS_ACCUMULATORS = ("rainnc", "snownc", "graupelnc")
#: Per-call outputs the kernel overwrites: the donor's exit value.
MICROPHYSICS_PER_CALL = (
    "rainncv", "snowncv", "graupelncv", "sr", "effc", "effr", "effi", "effs",
)
_CUMULUS_WATER_TARGETS = ("qv", "qc", "qi")


def suite_runtime(model):
    """The native runtime under ``model``'s physics, built if not yet.

    The suite builds its runtime lazily on its first step; building it
    here is the same construction (``native_suite.ArwenCudaColumnSuite
    .step``), so attaching before the first step changes nothing."""
    from .physics.native_runtime import NativePhysicsRuntime
    from .physics.native_suite import ArwenCudaColumnSuite

    physics = model.physics
    suite = getattr(physics, "adapter", physics)
    if not isinstance(suite, ArwenCudaColumnSuite):
        raise TypeError(
            "the scheme transplant works on the native column suite "
            "(physics.mode = 'arwen-native'); this model's physics is "
            f"{type(suite).__name__}"
        )
    if suite._runtime is None:
        suite._runtime = NativePhysicsRuntime(suite.options, modules=suite._modules)
    return suite._runtime


def _key(batch) -> tuple:
    band = None if batch.band is None else (int(batch.band[0]), int(batch.band[1]))
    return (band, float(batch.time_s), float(batch.dt_s))


@dataclass
class _Record:
    stage: str
    key: tuple
    payload: object


@dataclass
class StageLedger:
    """What the receiver's side of one stage did, summed over calls."""

    calls: int = 0
    #: Microphysics: cells whose entry equalled the donor's bit for bit
    #: (exit copied) and cells that received the donor's change.
    copied_cells: int = 0
    incremented_cells: int = 0
    #: Cells the transplanted output left negative (cumulus: after the
    #: integration, counted only where the entry was not negative) or
    #: that were set to zero (microphysics), and the mixing ratio (or
    #: number) summed over them, before the model's positivity repair.
    negative_cells: int = 0
    negative_sum: float = 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls, "copied_cells": self.copied_cells,
            "incremented_cells": self.incremented_cells,
            "negative_cells": self.negative_cells,
            "negative_sum": self.negative_sum,
        }


class Donor:
    """Records one model's scheme outputs for its receivers."""

    def __init__(self, model, stages=("cumulus",), *, name: str = "donor"):
        self.name = name
        self.stages = _check_stages(stages)
        self.runtime = suite_runtime(model)
        self.queues: list[deque] = []
        self.recorded = {stage: 0 for stage in self.stages}
        rt = self.runtime
        if "cumulus" in self.stages:
            original = type(rt)._cumulus_result

            def cumulus_result(batch, persistent, cfg, entry, pbl_out):
                result = original(rt, batch, persistent, cfg, entry, pbl_out)
                self._push("cumulus", batch, _copy_cumulus(batch.xp, result))
                return result

            rt._cumulus_result = cumulus_result
        if "microphysics" in self.stages:
            original_mp = type(rt)._microphysics_kernel

            def microphysics_kernel(batch, persistent):
                entry = _microphysics_snapshot(batch, persistent)
                original_mp(rt, batch, persistent)
                self._push("microphysics", batch,
                           (entry, _microphysics_snapshot(batch, persistent)))

            rt._microphysics_kernel = microphysics_kernel

    def _push(self, stage, batch, payload) -> None:
        self.recorded[stage] += 1
        record = _Record(stage, _key(batch), payload)
        for queue in self.queues:
            queue.append(record)

    def detach(self) -> None:
        for name in ("_cumulus_result", "_microphysics_kernel"):
            self.runtime.__dict__.pop(name, None)


class Receiver:
    """Applies a donor's scheme outputs to one model in place of its own."""

    def __init__(self, model, donor: Donor, stages=None, *, name: str = "receiver"):
        self.name = name
        self.stages = donor.stages if stages is None else _check_stages(stages)
        missing = [s for s in self.stages if s not in donor.stages]
        if missing:
            raise ValueError(f"the donor does not record {missing}")
        self.runtime = suite_runtime(model)
        if self.runtime is donor.runtime:
            raise ValueError("a model cannot receive from itself")
        for name in ("_cumulus_result", "_microphysics_kernel"):
            if name in self.runtime.__dict__:
                raise ValueError(
                    "this model already carries a transplant hook; one "
                    "model is a donor or a receiver, not both"
                )
        self.queue: deque = deque()
        donor.queues.append(self.queue)
        self.donor = donor
        self.ledger = {stage: StageLedger() for stage in self.stages}
        rt = self.runtime
        if "cumulus" in self.stages:
            apply_original = type(rt)._apply_cumulus_result

            def cumulus_result(batch, persistent, cfg, entry, pbl_out):
                return self._pop("cumulus", batch)

            def apply_cumulus_result(batch, persistent, result, **kw):
                xp = batch.xp
                before = {n: batch.arrays[n] < 0 for n in _CUMULUS_WATER_TARGETS}
                apply_original(rt, batch, persistent, result, **kw)
                ledger = self.ledger["cumulus"]
                ledger.calls += 1
                for n in _CUMULUS_WATER_TARGETS:
                    fresh = (batch.arrays[n] < 0) & ~before[n]
                    ledger.negative_cells += int(xp.count_nonzero(fresh))
                    ledger.negative_sum += float(xp.sum(
                        xp.where(fresh, batch.arrays[n], 0.0), dtype=xp.float64))

            rt._cumulus_result = cumulus_result
            rt._apply_cumulus_result = apply_cumulus_result
        if "microphysics" in self.stages:

            def microphysics_kernel(batch, persistent):
                entry, exit_ = self._pop("microphysics", batch)
                _apply_microphysics(batch, persistent, entry, exit_,
                                    self.ledger["microphysics"])

            rt._microphysics_kernel = microphysics_kernel

    def _pop(self, stage, batch):
        key = _key(batch)
        if not self.queue:
            raise RuntimeError(
                f"transplant {self.name}: the {stage} call at band {key[0]}, "
                f"time {key[1]:g} s has no donor record; the donor "
                f"{self.donor.name!r} must step first"
            )
        # The donor's records are in call order; a stage the receiver does
        # not take is skipped.
        while self.queue and self.queue[0].stage not in self.stages:
            self.queue.popleft()
        record = self.queue.popleft()
        if record.stage != stage or record.key != key:
            raise RuntimeError(
                f"transplant {self.name}: the receiver's {stage} call "
                f"(band, time, dt) = {key} meets the donor's {record.stage} "
                f"record {record.key}; the two runs are not stepping the "
                "same calls (same configuration, same bands, same clock)"
            )
        return record.payload

    def detach(self) -> None:
        for name in ("_cumulus_result", "_apply_cumulus_result", "_microphysics_kernel"):
            self.runtime.__dict__.pop(name, None)
        if self.queue in self.donor.queues:
            self.donor.queues.remove(self.queue)


def _check_stages(stages) -> tuple[str, ...]:
    stages = tuple(dict.fromkeys(str(s) for s in stages))
    unknown = [s for s in stages if s not in STAGES]
    if unknown or not stages:
        raise ValueError(f"transplant stages are {STAGES}, got {stages}")
    return stages


def _copy_cumulus(xp, result):
    out = SimpleNamespace()
    for name in _CUMULUS_ARRAYS:
        value = getattr(result, name, None)
        if value is not None:
            setattr(out, name, xp.array(value, copy=True))
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics is not None:
        out.diagnostics = dict(diagnostics)
    return out


def _microphysics_snapshot(batch, persistent) -> dict:
    xp = batch.xp
    snap = {n: xp.array(batch.arrays[n], copy=True) for n in MICROPHYSICS_BATCH_ARRAYS}
    f = persistent.arrays
    for n in (*MICROPHYSICS_ACCUMULATORS, *MICROPHYSICS_PER_CALL):
        snap[n] = xp.array(f[n], copy=True)
    return snap


def transplanted_value(xp, receiver_entry, donor_entry, donor_exit):
    """The copy-where-equal rule for an in-place scheme, one array.

    Where the receiver's entry equals the donor's bit for bit (compared
    as raw bits, so a NaN or a signed zero is not mistaken) the donor's
    exit is the result; elsewhere the donor's change is added.  Returns
    ``(value, equal_mask)``.  In float32 ``e + (x - e)`` is ``x`` only when
    ``x`` and ``e`` are within a factor of two, so the copy is what makes
    an identical twin stay identical."""
    a = xp.ascontiguousarray(receiver_entry)
    b = xp.ascontiguousarray(donor_entry)
    equal = a.view(_uint_of(a.dtype)) == b.view(_uint_of(b.dtype))
    value = xp.where(equal, donor_exit, receiver_entry + (donor_exit - donor_entry))
    return value.astype(receiver_entry.dtype, copy=False), equal


def _uint_of(dtype):
    return {4: np.uint32, 8: np.uint64, 2: np.uint16}[np.dtype(dtype).itemsize]


def _apply_microphysics(batch, persistent, entry, exit_, ledger: StageLedger) -> None:
    xp = batch.xp
    f = persistent.arrays
    ledger.calls += 1
    for name in MICROPHYSICS_BATCH_ARRAYS:
        value, equal = transplanted_value(xp, batch.arrays[name], entry[name], exit_[name])
        ledger.copied_cells += int(xp.count_nonzero(equal))
        ledger.incremented_cells += int(equal.size - xp.count_nonzero(equal))
        if name in MICROPHYSICS_NONNEGATIVE:
            negative = value < 0
            count = int(xp.count_nonzero(negative))
            if count:
                ledger.negative_cells += count
                ledger.negative_sum += float(xp.sum(
                    xp.where(negative, value, 0.0), dtype=xp.float64))
                value = xp.where(negative, xp.zeros((), dtype=value.dtype), value)
        batch.arrays[name][...] = value
    for name in MICROPHYSICS_ACCUMULATORS:
        value, _ = transplanted_value(xp, f[name], entry[name], exit_[name])
        f[name][...] = value
    for name in MICROPHYSICS_PER_CALL:
        f[name][...] = exit_[name]


# ---------------------------------------------------------------- experiment


def array_digests(model, state) -> dict[str, str]:
    """Every checkpointed array's SHA-256 (the checkpoint's own set,
    the semi-Lagrangian second time level included)."""
    from .checkpoint import bundle_arrays

    trajectory = model.trajectory_state() if model.semi_lagrangian else None
    out = {}
    for name, value in bundle_arrays(
        state, model.transform.backend.to_numpy, trajectory=trajectory,
    ).items():
        array = np.ascontiguousarray(np.asarray(value))
        out[name] = hashlib.sha256(
            array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
        ).hexdigest()
    return out


_CP_DRY = 1004.64
_R_DRY = 287.04
_T_REF = 270.0
_PS_REF = 1.0e5


def difference_fields(model, state) -> dict[str, np.ndarray]:
    """The grid fields the difference measures read, float64 on the host."""
    to_numpy = model.transform.backend.to_numpy
    g = model.grid_state(state.atmosphere, only=("u", "v", "temperature", "ps", "p_half"))
    out = {
        name: np.asarray(to_numpy(g[name]), dtype=np.float64)
        for name in ("u", "v", "temperature", "ps")
    }
    p_half = np.asarray(to_numpy(g["p_half"]), dtype=np.float64)
    out["dp"] = np.diff(p_half, axis=0) if p_half.shape[0] == out["u"].shape[0] + 1 else None
    surface = state.surface
    precip = np.zeros_like(out["ps"])
    for member in ("accumulated_rain_kg_m2", "accumulated_snow_kg_m2", "accumulated_graupel_kg_m2"):
        value = getattr(surface, member, None)
        if value is not None:
            precip = precip + np.asarray(to_numpy(value), dtype=np.float64)
    rainc = state.physics_state.arrays.get("rainc")
    convective = np.zeros_like(precip) if rainc is None else np.asarray(to_numpy(rainc), dtype=np.float64)
    out["precip_total_kg_m2"] = precip + convective
    out["precip_convective_kg_m2"] = convective
    return out


def difference_measures(a: dict, b: dict, weights: np.ndarray, box=None) -> dict[str, float]:
    """``b - a`` as global numbers: the dry total-energy norm (J/m2,
    the moist-free form of Ehrendorfer's), surface-pressure and precip
    RMS, and the sums the projections need."""
    w = weights / np.sum(weights)
    du, dv, dt = b["u"] - a["u"], b["v"] - a["v"], b["temperature"] - a["temperature"]
    dps = b["ps"] - a["ps"]
    dp = a["dp"] if a.get("dp") is not None else None
    if dp is None:
        lev = np.ones(du.shape[0])[:, None, None] / du.shape[0] * _PS_REF / 9.80665
        mass = np.broadcast_to(lev, du.shape)
    else:
        mass = dp / 9.80665
    kinetic = 0.5 * np.sum((du * du + dv * dv) * mass, axis=0)
    thermal = 0.5 * _CP_DRY / _T_REF * np.sum(dt * dt * mass, axis=0)
    surface = 0.5 * _R_DRY * _T_REF * (dps / _PS_REF) ** 2 * _PS_REF / 9.80665
    energy = kinetic + thermal + surface
    dpr = b["precip_total_kg_m2"] - a["precip_total_kg_m2"]
    out = {
        "energy_j_m2": float(np.sum(w * energy)),
        "kinetic_j_m2": float(np.sum(w * kinetic)),
        "ps_rms_hpa": float(np.sqrt(np.sum(w * dps * dps)) / 100.0),
        "precip_rms_mm": float(np.sqrt(np.sum(w * dpr * dpr))),
        "ps_max_abs_hpa": float(np.max(np.abs(dps)) / 100.0),
    }
    if box is not None:
        rows, cols = box
        out["box_ps_mean_hpa"] = float(np.mean(dps[rows, cols]) / 100.0)
        out["box_precip_mean_mm"] = float(np.mean(dpr[rows, cols]))
    return out


def energy_inner(a: dict, b: dict, c: dict, weights: np.ndarray) -> float:
    """The total-energy inner product of ``b - a`` with ``c - a``."""
    w = weights / np.sum(weights)
    mass = a["dp"] / 9.80665
    x = {k: b[k] - a[k] for k in ("u", "v", "temperature", "ps")}
    y = {k: c[k] - a[k] for k in ("u", "v", "temperature", "ps")}
    kin = 0.5 * np.sum((x["u"] * y["u"] + x["v"] * y["v"]) * mass, axis=0)
    th = 0.5 * _CP_DRY / _T_REF * np.sum(x["temperature"] * y["temperature"] * mass, axis=0)
    sf = 0.5 * _R_DRY * _T_REF * (x["ps"] / _PS_REF) * (y["ps"] / _PS_REF) * _PS_REF / 9.80665
    return float(np.sum(w * (kin + th + sf)))


@dataclass
class ExperimentPlan:
    config: str
    hours: float
    report_hours: tuple[float, ...]
    stages: tuple[str, ...] = ("cumulus",)
    seed: int = 20261112
    amplitude_scale: float = 1.0
    perturbation: str = "ensemble"
    self_only: bool = False
    box_radius_deg: float = 10.0
    extras: dict = field(default_factory=dict)


def _cached_initial_state(cfg):
    """Build every model of the experiment from ONE decode of the analysis.

    ``runner.build_model_and_cold_state`` reads the analysis through
    ``analysis_initial.analysis_initial_state``; this returns a function
    that hands every later call a copy of the first call's state, so five
    models cost one decode and start from the same bits."""
    import contextlib

    from . import analysis_initial

    held = {}
    original = analysis_initial.analysis_initial_state

    def cached(cfg_, transform, **kwargs):
        if "value" not in held:
            held["value"] = original(cfg_, transform, **kwargs)
        cold, phis, provenance = held["value"]
        return cold.copy(), phis, provenance

    @contextlib.contextmanager
    def patched():
        analysis_initial.analysis_initial_state = cached
        try:
            yield
        finally:
            analysis_initial.analysis_initial_state = original

    return patched()


def _perturb(model, state, plan: ExperimentPlan):
    from .da.options import EnsembleOptions
    from .da.perturbations import draw_perturbation, member_rng, perturbed_state

    options = EnsembleOptions(truncation=int(model.transform.truncation))
    increments, record = draw_perturbation(
        model, model.transform, state.atmosphere, options,
        member_rng(plan.seed, 1, "scheme-transplant"),
        amplitude_scale=plan.amplitude_scale,
    )
    return perturbed_state(model, model.transform, state, increments), {
        "route": "da.perturbations.draw_perturbation with the ensemble's default options",
        "seed": plan.seed, "member": 1, "purpose": "scheme-transplant",
        "amplitude_scale": plan.amplitude_scale,
        "max_degree": record.get("max_degree"),
    }


def run_experiment(plan: ExperimentPlan, *, progress=None, physics=None, cfg=None) -> dict:
    """The four-run experiment with its bit-exact control.

    Runs, all from one analysis decode, stepped in lockstep:

    * ``C``  the control;
    * ``P``  the control plus one seeded perturbation;
    * ``Ps`` P with C's scheme output (``stages``): the scheme can carry
      none of P's own difference;
    * ``Cs`` C with P's scheme output: the difference the scheme's output
      alone carries into an unperturbed start;
    * ``Cc`` a second C receiving C's output: the self-substitution, which
      must equal C bit for bit at every report hour.

    ``self_only`` runs C and Cc only (the gate).
    """
    from .config import load_config
    from .runner import build_model_and_cold_state

    say = progress or (lambda _m: None)
    cfg = cfg if cfg is not None else load_config(plan.config)
    steps_total = int(round(plan.hours * 3600.0 / cfg.dt_s))
    report_steps = sorted({int(round(h * 3600.0 / cfg.dt_s)) for h in plan.report_hours if h > 0})
    report_steps = [s for s in report_steps if s <= steps_total] or [steps_total]
    names = ("C", "Cc") if plan.self_only else ("C", "P", "Ps", "Cs", "Cc")
    t0 = time.time()
    models, states = {}, {}
    with _cached_initial_state(cfg):
        for name in names:
            models[name], states[name] = build_model_and_cold_state(cfg)
            if physics is not None:
                models[name].physics = physics()
    say(f"built {len(names)} models in {time.time() - t0:.1f} s")
    perturbation = None
    start_identity = {}
    if "P" in names:
        # P and Ps take the same seeded draw, each through its own model
        # (the repair and enforcement are the model's), and must start
        # from the same bits; C, Cs and Cc start from the analysis.
        states["P"], perturbation = _perturb(models["P"], states["P"], plan)
        states["Ps"], _ = _perturb(models["Ps"], states["Ps"], plan)
        for a, b in (("C", "Cs"), ("C", "Cc"), ("P", "Ps")):
            da = array_digests(models[a], states[a])
            db = array_digests(models[b], states[b])
            start_identity[f"{b}_equals_{a}"] = all(db.get(k) == v for k, v in da.items())
        if not all(start_identity.values()):
            raise RuntimeError(f"the twins do not start from the same bits: {start_identity}")
    donors = {"C": Donor(models["C"], plan.stages, name="C")}
    if "P" in names:
        donors["P"] = Donor(models["P"], plan.stages, name="P")
    receivers = {"Cc": Receiver(models["Cc"], donors["C"], name="Cc")}
    if "P" in names:
        receivers["Ps"] = Receiver(models["Ps"], donors["C"], name="Ps")
        receivers["Cs"] = Receiver(models["Cs"], donors["P"], name="Cs")
    order = [n for n in names if n in donors] + [n for n in names if n not in donors]
    weights = None
    rows = []
    gate = []
    t_step = time.time()
    for step in range(1, steps_total + 1):
        for name in order:
            states[name], _metrics = models[name].step(states[name], cfg.dt_s)
        if step in report_steps:
            hour = step * cfg.dt_s / 3600.0
            dc = array_digests(models["C"], states["C"])
            dcc = array_digests(models["Cc"], states["Cc"])
            differing = sorted(k for k in dc if dcc.get(k) != dc[k])
            gate.append({"hour": hour, "arrays": len(dc), "differing_arrays": differing})
            if "P" in names:
                fields = {n: difference_fields(models[n], states[n]) for n in ("C", "P", "Ps", "Cs")}
                if weights is None:
                    grid = models["C"].transform.grid
                    weights = _cell_weights(grid)
                box = _target_box(fields["C"], fields["P"], models["C"], plan.box_radius_deg)
                row = {
                    "hour": hour,
                    "target_box": box["record"],
                    "P_minus_C": difference_measures(fields["C"], fields["P"], weights, box["index"]),
                    "Ps_minus_C": difference_measures(fields["C"], fields["Ps"], weights, box["index"]),
                    "Cs_minus_C": difference_measures(fields["C"], fields["Cs"], weights, box["index"]),
                    "P_minus_Ps": difference_measures(fields["Ps"], fields["P"], weights, box["index"]),
                }
                e_pc = row["P_minus_C"]["energy_j_m2"]
                row["projection_Ps_on_P"] = (
                    energy_inner(fields["C"], fields["Ps"], fields["P"], weights) / e_pc if e_pc > 0 else None)
                row["projection_Cs_on_P"] = (
                    energy_inner(fields["C"], fields["Cs"], fields["P"], weights) / e_pc if e_pc > 0 else None)
                rows.append(row)
            say(f"step {step}/{steps_total} hour {hour:g}: Cc vs C differing arrays {len(differing)}; "
                f"{time.time() - t_step:.1f} s stepping")
    wall = time.time() - t0
    report = {
        "schema": SCHEMA,
        "config": str(plan.config),
        "config_hash": cfg.config_hash,
        "truncation": int(cfg.truncation),
        "dt_s": float(cfg.dt_s),
        "steps": steps_total,
        "stages": list(plan.stages),
        "runs": list(names),
        "perturbation": perturbation,
        "start_identity": start_identity,
        "self_substitution_gate": {
            "definition": "Cc (C receiving C's own scheme output) against C, every checkpointed "
                          "array by SHA-256 at every report hour",
            "rows": gate,
            "pass": all(not g["differing_arrays"] for g in gate),
        },
        "rows": rows,
        "ledgers": {name: {s: l.as_dict() for s, l in r.ledger.items()} for name, r in receivers.items()},
        "donor_records": {name: dict(d.recorded) for name, d in donors.items()},
        "wall_s": wall,
        "definitions": {
            "share_removed": "1 - E(Ps - C) / E(P - C): the fraction of P's difference energy that "
                             "disappears when P's scheme output is C's",
            "share_alone": "E(Cs - C) / E(P - C): the difference energy C's start reaches with P's "
                           "scheme output alone",
            "energy": "dry total energy of the difference, global area mean, J/m2: "
                      "0.5 (du2 + dv2) + cp/(2 Tr) dT2 summed with dp/g, plus 0.5 R Tr (dps/pr)2 pr/g; "
                      "Tr 270 K, pr 1000 hPa",
            "target_box": "the cells within box_radius_deg of the largest |ps(P) - ps(C)| poleward of "
                          "20 degrees at that hour",
        },
    }
    for row in rows:
        e = row["P_minus_C"]["energy_j_m2"]
        row["share_removed"] = None if e <= 0 else 1.0 - row["Ps_minus_C"]["energy_j_m2"] / e
        row["share_alone"] = None if e <= 0 else row["Cs_minus_C"]["energy_j_m2"] / e
        bp = row["P_minus_C"].get("box_ps_mean_hpa")
        if bp:
            row["box_ps_share_removed"] = 1.0 - row["Ps_minus_C"]["box_ps_mean_hpa"] / bp
            row["box_ps_share_alone"] = row["Cs_minus_C"]["box_ps_mean_hpa"] / bp
    for r in receivers.values():
        r.detach()
    for d in donors.values():
        d.detach()
    return report


def _cell_weights(grid) -> np.ndarray:
    """The Gaussian quadrature weight of every cell (the model's own
    global mean, ``GaussianGrid.global_mean``)."""
    nlat, nlon = grid.shape
    col = np.asarray(grid.quadrature_weights, dtype=np.float64)
    return np.broadcast_to(col[:, None], (nlat, nlon)).copy()


def _target_box(c: dict, p: dict, model, radius_deg: float) -> dict:
    grid = model.transform.grid
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    nlat, nlon = grid.shape
    lon = np.mod(np.asarray(grid.longitude_deg, dtype=np.float64), 360.0)
    dps = np.abs(p["ps"] - c["ps"])
    mask = np.abs(lat)[:, None] >= 20.0
    j, i = np.unravel_index(int(np.argmax(np.where(mask, dps, -1.0))), dps.shape)
    dlat = np.abs(lat[:, None] - lat[j])
    dlon = np.abs(((lon[None, :] - lon[i]) + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(lat[j]))
    inside = (dlat ** 2 + dlon ** 2) <= radius_deg ** 2
    rows, cols = np.nonzero(inside)
    return {
        "index": (rows, cols),
        "record": {"centre_latitude_deg": float(lat[j]), "centre_longitude_deg": float(lon[i]),
                   "cells": int(rows.size), "radius_deg": radius_deg},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arwen_global.transplant",
        description="The scheme transplant between exact twins: the four-run experiment "
                    "with its bit-exact self-substitution control.",
    )
    parser.add_argument("config")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--report-hours", type=float, nargs="+", default=None)
    parser.add_argument("--stages", nargs="+", default=["cumulus"], choices=list(STAGES))
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--self-only", action="store_true",
                        help="the gate alone: C and its self-substitution twin")
    parser.add_argument("--box-radius-deg", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    plan = ExperimentPlan(
        config=args.config, hours=args.hours,
        report_hours=tuple(args.report_hours or (args.hours,)),
        stages=tuple(args.stages), seed=args.seed, amplitude_scale=args.amplitude_scale,
        self_only=args.self_only, box_radius_deg=args.box_radius_deg,
    )
    report = run_experiment(plan, progress=lambda m: print(m, flush=True))
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate_pass": report["self_substitution_gate"]["pass"],
        "rows": [{k: r.get(k) for k in ("hour", "share_removed", "share_alone")} for r in report["rows"]],
    }, indent=2))
    return 0 if report["self_substitution_gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
