"""A per-machine cache of the Gram-solved Legendre analysis table.

THE COST THIS REMOVES.  Every process that builds a transform -- a run, a
restarted segment, every ``export`` -- solves one small dense system per
order to make the analysis table (``transform.analysis_band``).  Those
solves are the only BLAS work in the table build, and a multi-threaded
BLAS on a busy host collapses on them: MEASURED on a 24-core host with
other work running, the T255 solves took 10.9 s where they take 0.12 s
idle, and the T383 ones 10.7 s against 0.6 s (one-thread BLAS: 0.26 and
1.3 s).  Beside a plot pool that stall is the ordinary case, and it is
paid again by every segment and every export.

WHAT IS STORED.  Exactly the arrays the transform would hold: each band's
analysis rows cast to the backend's float dtype by the same ``asarray``
call the build makes, one stored (uncompressed) ``.npz`` per table.  A hit
hands those bytes to the transform in place of the solves, so a run with
a warm cache holds the bits a fresh build in the same process would.  The
basis and the derivative are elementwise recurrences with no BLAS in
them (0.04 to 0.2 s) and are always computed.

WHEN A FILE IS USED.  Only when every input of the solve is the one it was
made under: the truncation, latitudes, band and dtype; the Gaussian nodes
and weights, by digest; the bytes of this module, ``legendre.py`` and
``transform.py`` (any edit to the arithmetic retires every file); the
numpy version and its BLAS build; the machine and CPU model; and every
environment variable that picks a BLAS kernel or thread count.  A file
that does not match, does not open, or fails its zip checksums is a miss,
and the table is built as it always was and the file written again.  A
write goes to a temporary name and is renamed into place, so a reader
never sees half a file, and a failed write leaves the run as it was.

WHERE.  ``~/.gpuwm/cache/arwen-global-legendre``, or the directory named
by ``ARWEN_GLOBAL_LEGENDRE_CACHE``; ``off`` (or ``0``, ``false``, ``no``,
empty) turns it off.  One file per truncation, grid, band and dtype: 54
MiB at T255 and 176 MiB at T383 in float32.  Truncations below 127 and
tables above 1 GiB are never cached, nor is a transform that holds only
some orders (several cards) or none (streaming).  The folder can be
deleted at any time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

#: The environment variable naming the folder, or ``off``.
CACHE_ENV = "ARWEN_GLOBAL_LEGENDRE_CACHE"
#: Values of :data:`CACHE_ENV` that turn the cache off.
OFF_VALUES = frozenset({"", "off", "0", "false", "no"})
#: Smaller tables build in well under a second even on a busy host.
MIN_TRUNCATION = 127
#: A table above this many bytes is not written to the user's disk.
MAX_TABLE_BYTES = 1 << 30
#: Bumped whenever the file layout changes.
SCHEMA = "arwen-global-legendre-analysis/v1"
#: Environment variables that select a BLAS kernel, its thread count or its
#: reproducibility mode; each is part of the key.
BLAS_ENV = (
    "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR",
    "MKL_ENABLE_INSTRUCTIONS", "MKL_DEBUG_CPU_TYPE",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)

_HERE = Path(__file__).resolve().parent


def cache_dir() -> Path | None:
    """The cache folder, or ``None`` when the cache is off."""
    raw = os.environ.get(CACHE_ENV)
    if raw is None:
        return Path.home() / ".gpuwm" / "cache" / "arwen-global-legendre"
    if raw.strip().lower() in OFF_VALUES:
        return None
    return Path(raw).expanduser()


def _digest(*parts: bytes) -> str:
    sha = hashlib.sha256()
    for part in parts:
        sha.update(len(part).to_bytes(8, "little"))
        sha.update(part)
    return sha.hexdigest()


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or ""


def _blas_build() -> object:
    try:
        config = np.show_config(mode="dicts")
    except Exception:  # noqa: BLE001 - an older numpy prints only
        return None
    try:
        deps = config.get("Build Dependencies", {})
        return {name: deps.get(name) for name in ("blas", "lapack")}
    except Exception:  # noqa: BLE001
        return None


def _source_digest() -> str:
    names = ("legendre_cache.py", "legendre.py", "transform.py")
    return _digest(*((_HERE / name).read_bytes() for name in names))


def table_key(truncation: int, nlat: int, band: int, dtype,
              sin_lat, weights) -> dict:
    """Everything the analysis table's bits depend on, as a JSON object."""
    return {
        "schema": SCHEMA,
        "truncation": int(truncation),
        "nlat": int(nlat),
        "band": int(band),
        "dtype": np.dtype(dtype).str,
        "nodes": _digest(np.ascontiguousarray(sin_lat, dtype=np.float64).tobytes(),
                         np.ascontiguousarray(weights, dtype=np.float64).tobytes()),
        "source": _source_digest(),
        "numpy": np.__version__,
        "blas": _blas_build(),
        "machine": [platform.system(), platform.machine(), _cpu_model()],
        "env": {name: os.environ.get(name) for name in BLAS_ENV},
    }


class AnalysisTableCache:
    """One analysis table's file: :meth:`load` it or :meth:`store` it."""

    def __init__(self, path: Path, key: dict, bounds, nlat: int, dtype):
        self.path = path
        self.key = key
        self.key_text = json.dumps(key, sort_keys=True)
        self.bounds = [(int(a), int(b)) for a, b in bounds]
        self.truncation = int(key["truncation"])
        self.nlat = int(nlat)
        self.dtype = np.dtype(dtype)
        #: ``hit``, ``miss`` (built, then ``written`` or ``unwritten``).
        self.status = "miss"

    @classmethod
    def for_tables(cls, truncation: int, nlat: int, band: int, dtype,
                   sin_lat, weights, bounds) -> "AnalysisTableCache | None":
        """The cache file for this table, or ``None`` when not cached."""
        folder = cache_dir()
        if folder is None or int(truncation) < MIN_TRUNCATION:
            return None
        t1 = int(truncation) + 1
        elements = sum((b - a) * (t1 - a) * int(nlat) for a, b in bounds)
        if elements * np.dtype(dtype).itemsize > MAX_TABLE_BYTES:
            return None
        try:
            key = table_key(truncation, nlat, band, dtype, sin_lat, weights)
            tag = _digest(json.dumps(key, sort_keys=True).encode("utf-8"))[:20]
            name = (f"analysis-T{int(truncation)}-nlat{int(nlat)}-"
                    f"band{int(band)}-{np.dtype(dtype).name}-{tag}.npz")
            return cls(folder / name, key, bounds, nlat, dtype)
        except Exception:  # noqa: BLE001 - no key, no cache; the build runs
            return None

    def _expected_shape(self, m0: int, m1: int) -> tuple[int, int, int]:
        return (m1 - m0, self.nlat, self.truncation + 1 - m0)

    def load(self) -> list[np.ndarray] | None:
        """The band tables, in band order, or ``None`` on any mismatch."""
        if not self.path.is_file():
            return None
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                if str(archive["key"][()]) != self.key_text:
                    return None
                tables = []
                for index, (m0, m1) in enumerate(self.bounds):
                    table = archive[f"band_{index:04d}"]
                    if (table.dtype != self.dtype
                            or table.shape != self._expected_shape(m0, m1)):
                        return None
                    tables.append(table)
                if len(archive.files) != len(self.bounds) + 1:
                    return None
        except Exception:  # noqa: BLE001 - torn, foreign or unreadable: a miss
            return None
        self.status = "hit"
        return tables

    def store(self, tables: list[np.ndarray]) -> bool:
        """Write the tables; ``False`` (and nothing left behind) on failure."""
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {f"band_{i:04d}": t for i, t in enumerate(tables)}
            with open(tmp, "wb") as stream:
                np.savez(stream, key=np.array(self.key_text), **arrays)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 - a cache that cannot be written is no cache
            try:
                tmp.unlink()
            except OSError:
                pass
            self.status = "unwritten"
            return False
        self.status = "written"
        return True


__all__ = [
    "AnalysisTableCache", "CACHE_ENV", "MAX_TABLE_BYTES", "MIN_TRUNCATION",
    "cache_dir", "table_key",
]
