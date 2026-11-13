"""How many spherical transforms does one model step make, and where?

``python tools/arwen_global_transform_census.py CONFIG OUTDIR [--steps N] [--warmup W] [--json PATH] [run args...]``

A profiled step that writes output also runs the output diagnostics, a
full grid view of the finished state; to count ordinary steps, profile a
window that holds no output step.

Runs ``gpuwm-global run CONFIG`` in-process with the step profiler on and
counts, for each profiled step and each profiler section:

* every Legendre contraction (``SphericalHarmonicTransform._batched_contract``)
  and the level-planes it carried (the GEMM's M is that count);
* every ``numpy.fft.rfft`` / ``irfft`` call and its planes (numpy backend
  only; on cupy the FFT columns read zero and the contraction columns are
  still exact);
* every stacked synthesis request (``MoistHybridModel._synthesis_stack``)
  and how many of its planes a SUBSET memo could have served: a request
  that missed the memo although every one of its source arrays had already
  been synthesized, in some earlier stack, since the memo was last
  released.  That column is the upper bound of what serving a subset of a
  cached stack can save on this configuration.

The counts depend on the fields, the levels and the options, not on the
truncation, so a T21 numpy run counts what a T255 card step makes.  The
hooks change no arithmetic: the wrapped calls are the originals.

``census_run`` is the function the CI guard calls
(tests/test_arwen_global_transform_census.py).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys

COLUMNS = (
    "legendre_calls", "legendre_planes", "rfft_calls", "rfft_planes",
    "irfft_calls", "irfft_planes", "stack_requests", "stack_planes_missed",
    "stack_planes_subset_servable",
)


def _where(profilers):
    for profiler in profilers:
        stack = getattr(profiler, "_stack", None)
        if (getattr(profiler, "active", False) and stack and stack[0] == "step"
                and getattr(profiler, "_step", None) is not None
                and len(stack) > 1):
            return ".".join(stack), int(profiler._step)
    return None


def _lead(shape) -> int:
    count = 1
    for extent in shape[:-2]:
        count *= int(extent)
    return count


@contextlib.contextmanager
def counting():
    """Install the hooks; yield the record list; take them out again."""
    import numpy as np

    from arwen_global import dynamics as D
    from arwen_global import profile as P
    from arwen_global.spectral import transform as T

    records: list[tuple] = []
    profilers: list = []
    produced: dict[int, object] = {}

    init = P.StepProfiler.__init__
    contract = T.SphericalHarmonicTransform._batched_contract
    synthesis = D.MoistHybridModel._synthesis_stack
    release = D.MoistHybridModel.release_syntheses
    rfft, irfft = np.fft.rfft, np.fft.irfft

    def init_hook(self, *args, **kwargs):
        init(self, *args, **kwargs)
        profilers.append(self)

    def contract_hook(self, values, table, *, transposed=False):
        where = _where(profilers)
        if where:
            records.append(("legendre", *where, _lead(values.shape)))
        return contract(self, values, table, transposed=transposed)

    def fft_hook(kind, original):
        def hook(a, *args, **kwargs):
            where = _where(profilers)
            if where:
                records.append((kind, *where, _lead(np.shape(a))))
            return original(a, *args, **kwargs)
        return hook

    def synthesis_hook(self, state, spectral_names):
        where = _where(profilers)
        sources = [getattr(state, name) for name in spectral_names]
        memo = self._memo
        hits = memo.hits if memo is not None else 0
        payload = synthesis(self, state, spectral_names)
        hit = memo is not None and memo.hits > hits
        if where:
            planes = len(spectral_names) * int(self.nlev)
            missed = 0 if hit else planes
            servable = 0
            if not hit and all(id(a) in produced for a in sources):
                servable = planes
            records.append(("stack", *where, planes, missed, servable))
        for array in sources:
            # Hold the array, so its id cannot be recycled under the entry.
            produced[id(array)] = array
        return payload

    def release_hook(self):
        produced.clear()
        return release(self)

    P.StepProfiler.__init__ = init_hook
    T.SphericalHarmonicTransform._batched_contract = contract_hook
    D.MoistHybridModel._synthesis_stack = synthesis_hook
    D.MoistHybridModel.release_syntheses = release_hook
    np.fft.rfft = fft_hook("rfft", rfft)
    np.fft.irfft = fft_hook("irfft", irfft)
    try:
        yield records
    finally:
        P.StepProfiler.__init__ = init
        T.SphericalHarmonicTransform._batched_contract = contract
        D.MoistHybridModel._synthesis_stack = synthesis
        D.MoistHybridModel.release_syntheses = release
        np.fft.rfft, np.fft.irfft = rfft, irfft


def tabulate(records) -> dict[int, dict[str, dict[str, int]]]:
    """``{step: {section: {column: count}}}`` plus a ``TOTAL`` row per step."""
    steps: dict[int, dict[str, dict[str, int]]] = {}
    for record in records:
        kind, section, step = record[0], record[1], record[2]
        rows = steps.setdefault(step, collections.OrderedDict())
        row = rows.setdefault(section, dict.fromkeys(COLUMNS, 0))
        if kind == "stack":
            planes, missed, servable = record[3:]
            row["stack_requests"] += 1
            row["stack_planes_missed"] += missed
            row["stack_planes_subset_servable"] += servable
        else:
            row[f"{kind}_calls"] += 1
            row[f"{kind}_planes"] += record[3]
    for rows in steps.values():
        total = dict.fromkeys(COLUMNS, 0)
        for row in rows.values():
            for column in COLUMNS:
                total[column] += row[column]
        rows["TOTAL"] = total
    return steps


def census_run(config, outdir, *, steps: int = 2, warmup: int = 4,
               extra_args=()) -> dict[int, dict[str, dict[str, int]]]:
    """Run ``config`` with the census hooks and return :func:`tabulate`."""
    from arwen_global.cli import main

    with counting() as records:
        code = main([
            "run", str(config), "--outdir", str(outdir), "--overwrite",
            "--profile-steps", str(int(steps)),
            "--profile-warmup", str(int(warmup)), *extra_args,
        ])
    if code not in (0, None):
        raise RuntimeError(f"the census run exited {code}")
    return tabulate(records)


def render(table) -> str:
    lines = []
    head = ("%-36s %5s %8s | %5s %7s | %5s %7s | %5s %7s %7s"
            % ("section", "Lcall", "Lplanes", "rfft", "planes", "irfft",
               "planes", "stack", "missed", "subset"))
    for step, rows in sorted(table.items()):
        lines.append(f"== step {step}")
        lines.append(head)
        for section, row in rows.items():
            lines.append(
                "%-36s %5d %8d | %5d %7d | %5d %7d | %5d %7d %7d" % (
                    section, row["legendre_calls"], row["legendre_planes"],
                    row["rfft_calls"], row["rfft_planes"], row["irfft_calls"],
                    row["irfft_planes"], row["stack_requests"],
                    row["stack_planes_missed"],
                    row["stack_planes_subset_servable"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("outdir")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--json", default=None)
    # Anything else is handed to `gpuwm-global run` as it stands.
    args, rest = parser.parse_known_args(argv)
    rest = [item for item in rest if item != "--"]
    table = census_run(args.config, args.outdir, steps=args.steps,
                       warmup=args.warmup, extra_args=rest)
    print(render(table))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump({str(k): v for k, v in table.items()}, stream, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
