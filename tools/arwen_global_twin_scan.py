"""Exact-twin perturbation scans: plant one pattern in a start, read the response.

Three subcommands, none of which touches a run's source or configuration:

``perturb BASE.npz OUT.npz --config CFG --amplitude-k A``
    Write a copy of a step-0 checkpoint whose potential temperature carries
    ``A`` times one fixed pattern.  The pattern is a seeded field of
    spherical-harmonic coefficients on degrees 1 to ``--max-degree`` (no
    global mean), white in (n, m), with a Gaussian profile in ln p centred
    on ``--centre-hpa`` (width ``--width-ln-p``) at the reference surface
    pressure 1000 hPa, scaled so that ``A = 1`` is a grid RMS of 1 K on the
    level of largest weight.  The coefficients are drawn in (n, m) order up
    to ``--max-degree`` only, so the same seed and degree give the same
    field bit for bit at every truncation at or above that degree: a T63,
    a T127 and a T255 start receive one physical perturbation.
    ``A = 0`` writes arrays byte-identical to the input.  Every other array
    is copied unchanged; the checkpoint's own metadata is kept, with the
    one array hash and the self hash recomputed, so the file restarts
    under the config hash of the run that wrote it.

``bitflip BASE.npz OUT.npz``
    The one-bit twin: the lowest mantissa bit of the real part of one
    potential-temperature coefficient (``--level``, ``--n``, ``--m``)
    flipped, nothing else changed.  Its distance from the control is the
    noise floor every twin difference is read against.

``measure --config CFG --control DIR --arm NAME=DIR ...``
    For every checkpoint step the control and an arm share, the difference
    arm minus control, from the spectral coefficients alone:

    * the dry total-energy norm (J/kg), mass-weighted over the levels at
      the reference surface pressure:
      KE = a^2 / (8 pi) sum mult (|dzeta|^2 + |dD|^2) / (n (n + 1)),
      PE = (cp / Tr) / (8 pi) sum mult |dtheta|^2,
      SE = R Tr / (8 pi) sum mult |dlnps|^2, Tr = 300 K;
    * its square root (m/s), and RMS wind, theta, surface-pressure
      (1000 hPa times d ln ps) and vapour (g/kg) differences;
    * the same energy in three degree bands (1-21, 22-63, 64 and up).

    ``--truncate N`` truncates both states to degree N first (exact), so
    runs at different truncations are compared on the degrees they share.
    ``--pair NAME=PLUS,MINUS`` adds, per step, the relative nonlinearity
    ||d+ + d-|| / ||d+ - d-|| in the energy norm.  ``--compare A=B`` adds
    the energy-norm correlation and amplitude ratio of two arms'
    differences (A's response against B's, each measured from its own
    control when ``--control-of NAME=DIR`` names one), and the correlation
    in degree bands 1-10, 11-21, 22-42, 43-63 and above.  A second
    ``--seed`` gives an independent pattern: its response against the
    first's is the null a correlation is read against.

The coefficient convention is the package's orthonormal one: the grid mean
square of a scalar is sum mult |c_nm|^2 / (4 pi), mult 1 at m = 0 and 2
above (arwen_global.da.analysis.spectral_power_by_degree).

    python tools/arwen_global_twin_scan.py perturb run/arwen_global_step00000000.npz \
        p010.npz --config cfg.toml --amplitude-k 0.1
    python tools/arwen_global_twin_scan.py measure --config cfg.toml \
        --control ctl --arm p010=run-p010 --arm m010=run-m010 \
        --pair a010=p010,m010 --json scan.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

#: The default pattern: seed, top degree and vertical profile.
DEFAULT_SEED = 20261112
DEFAULT_MAX_DEGREE = 42
DEFAULT_CENTRE_HPA = 500.0
DEFAULT_WIDTH_LN_P = 0.5
REFERENCE_PS_PA = 100_000.0
REFERENCE_TEMPERATURE_K = 300.0
#: Degree bands of the energy report: large scale, the rest of T63, above T63.
BANDS = ((1, 21), (22, 63), (64, 10_000))
SPECTRAL = ("vorticity", "divergence", "theta", "log_surface_pressure", "qv")
STEP_RE = re.compile(r"_step(\d{8})\.npz$")


# ---------------------------------------------------------------------------
# checkpoint plumbing
# ---------------------------------------------------------------------------

def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def read_raw(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {name: np.array(archive[name], copy=True)
                  for name in archive.files if name != "__metadata__"}
    return metadata, arrays


def write_with(path: Path, metadata: dict, arrays: dict[str, np.ndarray],
               changed: tuple[str, ...]) -> Path:
    """The input's metadata with the changed arrays re-hashed and the self
    hash re-signed, written in the checkpoint writer's own layout."""
    metadata = json.loads(json.dumps(metadata))
    metadata.pop("self_sha256", None)
    for name in changed:
        metadata["arrays"][name]["sha256"] = _array_hash(arrays[name])
    metadata["self_sha256"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(json.dumps(metadata, sort_keys=True,
                                               allow_nan=False)),
            **arrays)
    os.replace(temporary, target)
    return target


def _refuse_unless_start(metadata: dict, arrays: dict) -> None:
    if any(name.startswith("trajectory__") for name in arrays):
        raise SystemExit(
            "refused: this checkpoint carries a second time level "
            "(trajectory__ arrays), which a perturbed state would not match; "
            "perturb the run's step-0 checkpoint")
    if int(metadata.get("step", -1)) != 0:
        raise SystemExit("refused: only a step-0 checkpoint is perturbed")


# ---------------------------------------------------------------------------
# the pattern
# ---------------------------------------------------------------------------

def reference_levels(cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(p_full_pa, dp_pa)`` of the config's levels at 1000 hPa."""
    vertical = cfg.vertical
    p_half = vertical.a_half_pa + vertical.b_half * REFERENCE_PS_PA
    return 0.5 * (p_half[1:] + p_half[:-1]), np.diff(p_half)


def vertical_profile(p_full_pa: np.ndarray, centre_hpa: float,
                     width_ln_p: float) -> np.ndarray:
    x = np.log(p_full_pa / (centre_hpa * 100.0)) / width_ln_p
    w = np.exp(-x * x)
    return w / w.max()


def horizontal_pattern(truncation: int, seed: int, max_degree: int) -> np.ndarray:
    """``(T+1, T+1)`` complex128: unit grid RMS, degrees 1..max_degree.

    Drawn in (n, m) order over the triangle n <= max_degree only, so the
    coefficients are the same numbers at every truncation >= max_degree."""
    if max_degree > truncation:
        raise SystemExit(
            f"refused: pattern degree {max_degree} is above this state's "
            f"truncation T{truncation}")
    rng = np.random.default_rng(int(seed))
    out = np.zeros((truncation + 1, truncation + 1), dtype=np.complex128)
    for n in range(1, max_degree + 1):
        for m in range(n + 1):
            re_part = rng.standard_normal()
            im_part = rng.standard_normal()
            out[n, m] = re_part + (0.0 if m == 0 else 1j * im_part)
    mult = np.where(np.arange(truncation + 1) == 0, 1.0, 2.0)
    mean_square = float(np.sum(mult * np.abs(out) ** 2) / (4.0 * math.pi))
    return out / math.sqrt(mean_square)


def theta_pattern(cfg, truncation: int, *, seed: int, max_degree: int,
                  centre_hpa: float, width_ln_p: float) -> np.ndarray:
    p_full, _ = reference_levels(cfg)
    w = vertical_profile(p_full, centre_hpa, width_ln_p)
    h = horizontal_pattern(truncation, seed, max_degree)
    return w[:, None, None] * h[None, :, :]


def _load_cfg(path):
    from arwen_global.config import load_config

    return load_config(str(path))


def cmd_perturb(args) -> dict:
    metadata, arrays = read_raw(Path(args.base))
    _refuse_unless_start(metadata, arrays)
    name = "atmosphere__theta"
    theta = arrays[name]
    truncation = theta.shape[-1] - 1
    cfg = _load_cfg(args.config)
    pattern = theta_pattern(cfg, truncation, seed=args.seed,
                            max_degree=args.max_degree,
                            centre_hpa=args.centre_hpa,
                            width_ln_p=args.width_ln_p)
    if pattern.shape != theta.shape:
        raise SystemExit(f"refused: pattern {pattern.shape} against theta "
                         f"{theta.shape}; is --config this run's config?")
    if args.amplitude_k != 0.0:
        arrays[name] = (theta.astype(np.complex128)
                        + float(args.amplitude_k) * pattern).astype(theta.dtype)
    write_with(Path(args.out), metadata, arrays, (name,))
    report = {"tool": "perturb", "base": str(args.base), "out": str(args.out),
              "amplitude_k": float(args.amplitude_k), "seed": int(args.seed),
              "max_degree": int(args.max_degree),
              "centre_hpa": float(args.centre_hpa),
              "width_ln_p": float(args.width_ln_p),
              "truncation": int(truncation),
              "pattern_sha256": _array_hash(pattern)}
    if args.verify:
        from arwen_global.checkpoint import read_checkpoint

        read_checkpoint(Path(args.out), expected_config_hash=cfg.config_hash,
                        semi_implicit_scheme=cfg.semi_implicit_scheme,
                        integrator=cfg.integrator)
        report["verified"] = True
    return report


def cmd_bitflip(args) -> dict:
    metadata, arrays = read_raw(Path(args.base))
    _refuse_unless_start(metadata, arrays)
    name = "atmosphere__theta"
    theta = arrays[name].copy()
    if theta.dtype.kind != "c":
        raise SystemExit("refused: theta is not complex coefficients")
    real = theta.real.copy()
    kind = {4: np.uint32, 8: np.uint64}[real.dtype.itemsize]
    view = real.view(kind)
    before = float(real[args.level, args.n, args.m])
    view[args.level, args.n, args.m] ^= kind(1)
    theta = (real + 1j * theta.imag).astype(theta.dtype)
    arrays[name] = theta
    write_with(Path(args.out), metadata, arrays, (name,))
    return {"tool": "bitflip", "base": str(args.base), "out": str(args.out),
            "index": [args.level, args.n, args.m], "before": before,
            "after": float(real[args.level, args.n, args.m])}


# ---------------------------------------------------------------------------
# measuring
# ---------------------------------------------------------------------------

def checkpoint_steps(directory: Path) -> dict[int, Path]:
    out = {}
    for path in sorted(Path(directory).glob("*_step*.npz")):
        match = STEP_RE.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def load_spectral(path: Path, truncation: int | None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        out = {}
        for name in SPECTRAL:
            c = np.asarray(archive[f"atmosphere__{name}"], dtype=np.complex128)
            if truncation is not None:
                if c.shape[-1] - 1 < truncation:
                    raise SystemExit(f"{path}: T{c.shape[-1] - 1} is below "
                                     f"--truncate {truncation}")
                c = c[..., : truncation + 1, : truncation + 1]
            out[name] = c
    return out


def _mult(t1: int) -> np.ndarray:
    return np.where(np.arange(t1) == 0, 1.0, 2.0)


def energy_terms(d: dict[str, np.ndarray], level_weight: np.ndarray,
                 band: tuple[int, int] | None = None) -> dict[str, float]:
    """Mass-weighted energy (J/kg) of one difference, optionally on a band."""
    from arwen_global.constants import (
        DRY_AIR_CP, DRY_AIR_GAS_CONSTANT, EARTH_RADIUS_M)

    t1 = d["theta"].shape[-1]
    n = np.arange(t1, dtype=np.float64)
    keep = np.ones(t1, dtype=bool)
    if band is not None:
        keep = (n >= band[0]) & (n <= band[1])
    mult = _mult(t1)[None, :] * keep[:, None]   # (n, m)
    inv = np.zeros(t1)
    inv[1:] = 1.0 / (n[1:] * (n[1:] + 1.0))

    def level_power(c):   # (lev, n, m) -> (lev,)
        return np.sum(mult[None] * (c.real ** 2 + c.imag ** 2), axis=(-2, -1))

    rot = d["vorticity"]
    div = d["divergence"]
    ke_lev = (EARTH_RADIUS_M ** 2 / (8.0 * math.pi)) * np.sum(
        mult[None] * inv[None, :, None]
        * (np.abs(rot) ** 2 + np.abs(div) ** 2), axis=(-2, -1))
    th_lev = level_power(d["theta"]) / (4.0 * math.pi)
    qv_lev = level_power(d["qv"]) / (4.0 * math.pi)
    ps = d["log_surface_pressure"]
    ps_ms = float(np.sum(mult * (ps.real ** 2 + ps.imag ** 2)) / (4.0 * math.pi))
    w = level_weight / level_weight.sum()
    ke = float(np.sum(w * ke_lev))
    th_ms = float(np.sum(w * th_lev))
    qv_ms = float(np.sum(w * qv_lev))
    pe = 0.5 * DRY_AIR_CP / REFERENCE_TEMPERATURE_K * th_ms
    se = 0.5 * DRY_AIR_GAS_CONSTANT * REFERENCE_TEMPERATURE_K * ps_ms
    total = ke + pe + se
    return {
        "energy_j_kg": total, "kinetic_j_kg": ke, "thermal_j_kg": pe,
        "surface_j_kg": se, "sqrt_energy_m_s": math.sqrt(total),
        "rms_wind_m_s": math.sqrt(2.0 * ke), "rms_theta_k": math.sqrt(th_ms),
        "rms_ps_hpa": 1000.0 * math.sqrt(ps_ms),
        "rms_qv_g_kg": 1000.0 * math.sqrt(qv_ms),
    }


def energy_inner(a: dict, b: dict, level_weight: np.ndarray,
                 band: tuple[int, int] | None = None) -> float:
    """Energy-norm inner product of two differences (J/kg), optionally on
    the degrees of one band."""
    from arwen_global.constants import (
        DRY_AIR_CP, DRY_AIR_GAS_CONSTANT, EARTH_RADIUS_M)

    t1 = a["theta"].shape[-1]
    n = np.arange(t1, dtype=np.float64)
    keep = np.ones(t1, dtype=bool)
    if band is not None:
        keep = (n >= band[0]) & (n <= band[1])
    mult = _mult(t1)[None, :] * keep[:, None]
    inv = np.zeros(t1)
    inv[1:] = 1.0 / (n[1:] * (n[1:] + 1.0))
    w = level_weight / level_weight.sum()

    def dot(x, y):
        return (x.real * y.real + x.imag * y.imag)

    ke = (EARTH_RADIUS_M ** 2 / (8.0 * math.pi)) * np.sum(
        mult[None] * inv[None, :, None]
        * (dot(a["vorticity"], b["vorticity"]) + dot(a["divergence"], b["divergence"])),
        axis=(-2, -1))
    th = np.sum(mult[None] * dot(a["theta"], b["theta"]), axis=(-2, -1)) / (4.0 * math.pi)
    ps = float(np.sum(mult * dot(a["log_surface_pressure"], b["log_surface_pressure"]))
               / (4.0 * math.pi))
    return float(np.sum(w * ke)) + 0.5 * DRY_AIR_CP / REFERENCE_TEMPERATURE_K * float(
        np.sum(w * th)) + 0.5 * DRY_AIR_GAS_CONSTANT * REFERENCE_TEMPERATURE_K * ps


#: Degree bands of the cross-arm correlation (the ensembles attack's 1-10, 11-21,
#: 22-42, then the rest of T63 and above it).
COMPARE_BANDS = ((1, 10), (11, 21), (22, 42), (43, 63), (64, 10_000))


def _band_correlations(a: dict, b: dict, level_weight: np.ndarray) -> dict:
    top = a["theta"].shape[-1] - 1
    out = {}
    for lo, hi in COMPARE_BANDS:
        if lo > top:
            continue
        aa = energy_inner(a, a, level_weight, (lo, hi))
        bb = energy_inner(b, b, level_weight, (lo, hi))
        ab = energy_inner(a, b, level_weight, (lo, hi))
        out[f"{lo}-{min(hi, top)}"] = (ab / math.sqrt(aa * bb)) if aa > 0 and bb > 0 else None
    return out


def _diff(a: dict, b: dict) -> dict:
    return {k: a[k] - b[k] for k in SPECTRAL}


def _named(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"{what} wants NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def cmd_measure(args) -> dict:
    cfg = _load_cfg(args.config)
    _, dp = reference_levels(cfg)
    arms = _named(args.arm, "--arm")
    control_of = _named(args.control_of, "--control-of")
    pairs = {k: tuple(v.split(",")) for k, v in _named(args.pair, "--pair").items()}
    compares = _named(args.compare, "--compare")
    controls = {"": checkpoint_steps(Path(args.control))}
    for name, directory in control_of.items():
        controls[name] = checkpoint_steps(Path(directory))
    arm_steps = {name: checkpoint_steps(Path(d)) for name, d in arms.items()}
    rows = []
    cache: dict = {}

    def state(path):
        key = (str(path), args.truncate)
        if key not in cache:
            cache[key] = load_spectral(path, args.truncate)
        return cache[key]

    steps = sorted(set(controls[""]).union(*[set(s) for s in arm_steps.values()]))
    for step in steps:
        diffs = {}
        for name, by_step in arm_steps.items():
            ctl = controls.get(name, controls[""])
            if step not in by_step or step not in ctl:
                continue
            diffs[name] = _diff(state(by_step[step]), state(ctl[step]))
            row = {"step": step, "arm": name}
            row.update(energy_terms(diffs[name], dp))
            for lo, hi in BANDS:
                if lo > diffs[name]["theta"].shape[-1] - 1:
                    continue
                row[f"energy_n{lo}_{min(hi, diffs[name]['theta'].shape[-1] - 1)}_j_kg"] = \
                    energy_terms(diffs[name], dp, (lo, hi))["energy_j_kg"]
            row["identical"] = all(not np.any(v) for v in diffs[name].values())
            rows.append(row)
        for name, (plus, minus) in pairs.items():
            if plus in diffs and minus in diffs:
                s = {k: diffs[plus][k] + diffs[minus][k] for k in SPECTRAL}
                r = {k: diffs[plus][k] - diffs[minus][k] for k in SPECTRAL}
                es = energy_terms(s, dp)["energy_j_kg"]
                er = energy_terms(r, dp)["energy_j_kg"]
                rows.append({"step": step, "pair": name,
                             "relative_nonlinearity": (math.sqrt(es / er) if er > 0 else None),
                             "odd_response_sqrt_energy_m_s": math.sqrt(er / 4.0)})
        for name, other in compares.items():
            if name in diffs and other in diffs:
                aa = energy_inner(diffs[name], diffs[name], dp)
                bb = energy_inner(diffs[other], diffs[other], dp)
                ab = energy_inner(diffs[name], diffs[other], dp)
                rows.append({"step": step, "compare": f"{name}~{other}",
                             "correlation": ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None,
                             "amplitude_ratio": math.sqrt(aa / bb) if bb > 0 else None,
                             "band_correlation": _band_correlations(
                                 diffs[name], diffs[other], dp)})
        cache.clear()   # one step's states at a time: a T255 state is ~0.2 GB
    return {"tool": "measure", "config": str(args.config),
            "truncate": args.truncate, "control": str(args.control),
            "arms": arms, "control_of": control_of, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("perturb", help="plant A times the pattern in a step-0 checkpoint")
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--amplitude-k", type=float, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-degree", type=int, default=DEFAULT_MAX_DEGREE)
    p.add_argument("--centre-hpa", type=float, default=DEFAULT_CENTRE_HPA)
    p.add_argument("--width-ln-p", type=float, default=DEFAULT_WIDTH_LN_P)
    p.add_argument("--verify", action="store_true",
                   help="read the output back through the package's checkpoint reader")
    b = sub.add_parser("bitflip", help="the one-bit twin of a step-0 checkpoint")
    b.add_argument("base", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--level", type=int, default=20)
    b.add_argument("--n", type=int, default=10)
    b.add_argument("--m", type=int, default=3)
    m = sub.add_parser("measure", help="energy-norm differences arm minus control")
    m.add_argument("--config", type=Path, required=True)
    m.add_argument("--control", type=Path, required=True)
    m.add_argument("--arm", action="append", default=[])
    m.add_argument("--control-of", action="append", default=[])
    m.add_argument("--pair", action="append", default=[])
    m.add_argument("--compare", action="append", default=[])
    m.add_argument("--truncate", type=int, default=None)
    for q in (p, b, m):
        q.add_argument("--json", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = {"perturb": cmd_perturb, "bitflip": cmd_bitflip,
              "measure": cmd_measure}[args.command](args)
    text = json.dumps(report, indent=1, sort_keys=True, allow_nan=False)
    if args.json is not None:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    if args.command != "measure" or args.json is None:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
