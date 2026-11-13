"""Atomic self-hashed Arwen Global receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .constants import RECEIPT_SCHEMA
from .pins import KNOWN_PINS_HASHES, PINS_HASH_BY_SCHEME


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


#: Distributions whose version changes a run's numbers, in the order a reader
#: cares about them.  A name that is not installed is simply absent.
RECORDED_LIBRARIES = ("gpuwm-global", "gpuwm", "gpuwm-data", "numpy", "scipy",
                      "cupy-cuda13x", "cupy-cuda12x", "cupy", "netCDF4")


def library_versions() -> dict[str, str]:
    """The installed versions of everything a run's arithmetic rides on.

    THE BREAKAGE THIS NAMES.  A receipt recorded the configuration, the
    arithmetic pins, the physics identity and the machine, and said nothing
    about the libraries.  Two runs of the same configuration on the same card
    can disagree, and this package already has a measured instance of exactly
    that: the Legendre quadrature this model's spectral tables ride on moved
    its bits between numpy 2.2.6 and 2.3.0, and a table's recorded digest
    reproduces under one and not the other.  A receipt that cannot say which
    numpy produced it cannot answer the first question anybody asks about two
    runs that differ, and the answer is not recoverable later: by the time the
    question is asked the environment has moved.

    The interpreter is recorded beside them for the same reason.
    """

    import sys
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {
        "python": ".".join(str(n) for n in sys.version_info[:3]),
    }
    for name in RECORDED_LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


#: Distribution name prefixes whose wheels carry the CUDA user-space a CuPy
#: run loads (runtime, NVRTC, cuBLAS, cuFFT, ...): recorded in the software
#: stack beside the libraries above.
CUDA_WHEEL_PREFIXES = ("nvidia-", "cuda-", "gpuwm-kernels")


def software_stack(backend: str) -> dict[str, object]:
    """Everything a later replay of a run's window has to reproduce, read
    once at the start of a run: the interpreter and platform, every
    recorded library and every CUDA user-space wheel by version, and on the
    CuPy backend the card (name, compute capability, memory), the CUDA
    runtime, driver-API and NVRTC versions CuPy reports and the NVIDIA
    driver version ``nvidia-smi`` reports.  Descriptive, in no identity;
    it never raises (a field it cannot read says so)."""
    import platform
    import sys

    stack: dict[str, object] = {"backend": str(backend)}
    try:
        from ._version import __version__ as package_version

        stack["arwen_global"] = str(package_version)
    except Exception as exc:  # noqa: BLE001
        stack["arwen_global"] = f"unreadable ({type(exc).__name__})"
    # The version above is the installed distribution's; a run from a source
    # tree on PYTHONPATH imports that tree instead, and says where it is
    # (the receipt's physics_modules hash the files that integrated).
    stack["arwen_global_imported_from"] = str(Path(__file__).resolve().parent)
    stack["python"] = sys.version.split()[0]
    stack["platform"] = {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine(),
                         "libc": " ".join(platform.libc_ver()).strip() or None}
    stack["libraries"] = library_versions()
    try:
        from importlib.metadata import distributions

        wheels = {}
        for dist in distributions():
            name = (dist.metadata.get("Name") or "").lower()
            if name.startswith(CUDA_WHEEL_PREFIXES):
                wheels[name] = dist.version
        stack["cuda_wheels"] = dict(sorted(wheels.items()))
    except Exception as exc:  # noqa: BLE001
        stack["cuda_wheels"] = f"unreadable ({type(exc).__name__}: {exc})"
    stack["device"] = _device_stack() if str(backend) == "cupy" else None
    return stack


def _device_stack() -> dict[str, object]:
    device: dict[str, object] = {}

    def read(key, getter):
        try:
            device[key] = getter()
        except Exception as exc:  # noqa: BLE001
            device[key] = f"unreadable ({type(exc).__name__}: {exc})"

    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        return {"cupy": f"not importable ({type(exc).__name__})"}

    def version(number):
        number = int(number)
        return f"{number // 1000}.{(number % 1000) // 10}"

    device["cupy"] = cp.__version__
    read("cuda_runtime", lambda: version(cp.cuda.runtime.runtimeGetVersion()))
    read("cuda_driver_api", lambda: version(cp.cuda.runtime.driverGetVersion()))
    read("nvrtc", lambda: ".".join(str(n) for n in cp.cuda.nvrtc.getVersion()))
    read("ordinal", lambda: int(cp.cuda.runtime.getDevice()))

    def card():
        props = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.runtime.getDevice()))
        name = props["name"]
        return {"name": name.decode() if isinstance(name, bytes) else str(name),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "total_memory_bytes": int(props["totalGlobalMem"]),
                "multiprocessors": int(props["multiProcessorCount"])}

    read("card", card)
    read("nvidia_driver", _nvidia_driver_version)
    return device


def _nvidia_driver_version() -> str:
    """The kernel driver's version string, from ``nvidia-smi`` (the CUDA
    calls above report only the driver API level)."""
    import os as _os
    import subprocess

    command = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible:
        command.insert(1, f"--id={visible}")
    out = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi printed no driver version")
    return lines[0]


def finalize_receipt(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["schema"] = RECEIPT_SCHEMA
    # Beside the pins, never inside them: `pins_hash` is the identity of the
    # ARITHMETIC and two runs under different numpys share it, which is the
    # point of it.  The libraries are what tells a reader why two runs that
    # share it might still differ.
    result.setdefault("libraries", library_versions())
    # WHOSE PHYSICS INTEGRATED.  The libraries above name the engine's
    # version, and this package carries most of the physics it runs, so the
    # engine's version does not answer the question.  Every physics module
    # this process actually resolved is recorded with its origin and the
    # SHA-256 of the file that was imported (and, for the kernel loader, one
    # digest over the device sources it binds).  A command that ran no
    # physics records an empty table rather than a claim about code that did
    # not execute.
    from .physics.provenance import integrated_physics_modules

    result.setdefault("physics_modules", integrated_physics_modules())
    result.setdefault("engine_seam", engine_seam_verdict())
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def engine_seam_verdict() -> dict[str, object]:
    """Whether the engine under the carried physics is the one it was pinned to.

    THE BREAKAGE THIS NAMES.  `physics_modules` records the SHA-256 of every
    carried module that integrated, which answers "whose physics ran" for the
    half of the physics this package carries.  It cannot see the other half.
    `gpuwm/core/constants.py` stays on the engine and supplies CUDA_DEFINES to
    the preamble of every carried kernel, so a run against a 2.8.x that moved
    it produces a receipt whose module hashes are identical to yesterday's
    while every compiled kernel's assembled source, its digest and its
    floating-point contraction have moved.  The same is true of the twelve
    engine modules the carried driver imports at module scope.

    So the seam verdict travels in the receipt beside the module hashes: the
    engine version the pins were taken against, the engine version that
    actually resolved, and every file whose bytes are not the pinned ones, by
    name.  A receipt is read long after the environment it was written in is
    gone, which is why this is recorded rather than left to the doctor.

    It never raises.  A receipt that failed to write because the seam manifest
    was unreadable would lose the run, so an unreadable seam is recorded AS
    unreadable.
    """

    verdict: dict[str, object] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            verdict["engine_resolved"] = version("gpuwm")
        except PackageNotFoundError:
            verdict["engine_resolved"] = "not installed"

        from .engine_seam import check_seam, load_manifest

        manifest = load_manifest()
        verdict["pinned_against"] = str(
            manifest.get("engine", {}).get("version", "unrecorded"))
        rows = check_seam()
        unproven = [row for row in rows if row.verdict != "proven"]
        verdict["files"] = len(rows)
        verdict["proven"] = len(rows) - len(unproven)
        verdict["unproven"] = {row.path: row.verdict for row in unproven}
    except Exception as exc:  # noqa: BLE001 - a receipt is never lost to this
        verdict["unreadable"] = f"{type(exc).__name__}: {exc}"
    return verdict


#: The receipt key a run on another world carries its record under.  The
#: same name as the checkpoint's physics-metadata key
#: (``checkpoint.WORLD_RECORD_KEY``) so one reader serves both.
RECEIPT_WORLD_KEY = "world"


def world_receipt_rows(world) -> dict[str, object]:
    """The receipt rows a run on ``world`` adds: ``{"world": record}`` for
    a world that is not Earth's, and NOTHING for Earth, so an Earth receipt
    has the key set it had before worlds existed.  ``world`` is a
    ``WorldSpec`` or None (a config built without one is Earth's)."""
    if world is None or world.is_earth:
        return {}
    return {RECEIPT_WORLD_KEY: world.record()}


def receipt_world_identity(payload) -> str:
    """``"none"`` for a receipt of an Earth run (including every receipt
    written before worlds existed), otherwise the world's identity hash."""
    from .world.spec import world_identity_of

    if not isinstance(payload, dict):
        return world_identity_of(None)
    return world_identity_of(payload.get(RECEIPT_WORLD_KEY))


def write_receipt(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = finalize_receipt(payload)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def check_receipt(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("Arwen Global receipt schema mismatch")
    # The self-hash only proves the file is intact, not that the numbers in it
    # were produced by this build's arithmetic.  Without the pin comparison a
    # receipt written under a superseded PIN_DOCUMENT validates green and its
    # gate values are read as evidence for the current one.  The build
    # integrates one arithmetic per [semi_implicit] scheme, each under its
    # own pin (the external proxy's is the pin of every receipt written
    # before the vertical-mode scheme existed), so any of those pins is
    # this build's.
    if payload.get("pins_hash") not in KNOWN_PINS_HASHES:
        raise ValueError(
            "Arwen Global receipt arithmetic pins mismatch: receipt "
            f"{payload.get('pins_hash')!r} is not a pin of this build "
            + "("
            + ", ".join(
                f"{scheme}: {digest}" for scheme, digest in PINS_HASH_BY_SCHEME.items()
            )
            + ")"
        )
    self_hash = payload.pop("self_sha256", None)
    expected = hashlib.sha256(canonical(payload)).hexdigest()
    if self_hash != expected:
        raise ValueError("Arwen Global receipt self-hash mismatch")
    payload["self_sha256"] = self_hash
    return payload


__all__ = [
    "RECEIPT_WORLD_KEY",
    "check_receipt",
    "finalize_receipt",
    "receipt_world_identity",
    "software_stack",
    "world_receipt_rows",
    "write_receipt",
]
