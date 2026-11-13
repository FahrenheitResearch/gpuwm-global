"""One forcing operator: pull the model toward a stored record of large
scales, and keep the pull as a measurement of model error.

Every step, before the step integrates, the chosen spectral fields move a
fraction of the way toward a stored record on the chosen degrees and
nowhere else,

    x <- x + a(k) L (x_rec(t) - x),        a = 1 - exp(-dt / tau)

with ``L`` a per-degree low-pass (weight one at and below ``full_degree``,
zero at and above ``zero_degree``, raised cosine between; the defaults 42
and 43 pull degrees 1 to 42, wavelengths longer than about 950 km), ``k``
an optional weight per level, ``tau`` the relaxation time and ``x_rec(t)``
the record interpolated linearly in time.  Degree 0 of ln ps is never
pulled: the mass fixer owns the global mean of surface pressure.

The pull is the measurement.  Once a run is held near the record, the
time-mean pull is what the model needed added to stay there: the model's
systematic tendency error against the record, on the pulled scales, per
field and level.  :class:`PullLedger` keeps it, with the departure from
the record (which an observe-only operator, ``tau = inf``, reports for a
free run without moving a bit), the instantaneous pull's root mean square
and the global-mean pull of the scalar fields, per level, per degree and
per window of model time.

The record is a :class:`ForcingRecord`: the low degrees of the spectral
fields at a list of model times, at most a few megabytes per time.  A run
writes one with :class:`ForcingRecordWriter`; :func:`record_from_checkpoints`
cuts one from a finished run's checkpoints.  Alignment is by model time
plus ``record_offset_s``.

One operator, one owner.  The runner's step loop calls
:meth:`ForcingOperator.apply` before each step when a run asks for it
(``gpuwm-global run --forcing``); any other loop that holds a state, a
model and a step (a cycle, an ensemble) calls the same method.  A run
without the option never constructs it, and its arithmetic is the shipped
one.  The pull is a pure function of the state, the model time and the
record, so a restart in the middle of a pulled run continues it exactly.
The pulled state's checkpoints carry the forcing under
``physics_metadata["forcing_history"]``; the config identity is unchanged,
so a free and a pulled run from one start share one checkpoint lineage
and differ only by the pull.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import EARTH_RADIUS_M, SPECTRAL_FIELDS
from .da.analysis import taper_weights
from .state import ArwenGlobalState

FORCING_RECORD_SCHEMA = "gpuwm.arwen-global-forcing-record/v1"
FORCING_LEDGER_SCHEMA = "gpuwm.arwen-global-forcing-ledger/v1"
FORCING_HISTORY_KEY = "forcing_history"
FORCING_LEDGER_NAME = "forcing_ledger.json"
FORCING_PULL_NAME = "forcing_pull.npz"
DEFAULT_FORCED_FIELDS = ("vorticity", "divergence", "theta")
#: Fields whose degree-0 coefficient is a global mean (theta, vapour, ln ps).
SCALAR_FIELDS = ("theta", "qv", "log_surface_pressure")
SURFACE_FIELD = "log_surface_pressure"
UNITS = {
    "vorticity": "s-1",
    "divergence": "s-1",
    "theta": "K",
    "qv": "kg kg-1",
    "log_surface_pressure": "1 (ln Pa)",
    "wind": "m s-1",
}
_DAY_S = 86400.0
_TIME_TOLERANCE_S = 1.0e-6


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcingOptions:
    """The pull's parameters.

    tau_s
        Relaxation time in seconds (six hours by default).  ``math.inf``
        is the observe-only operator: it measures the departure from the
        record every step and moves nothing.
    full_degree, zero_degree
        The low-pass: weight one at and below ``full_degree``, zero at and
        above ``zero_degree``, raised cosine between (defaults 42 and 43:
        degrees 1 to 42 pulled in full, nothing above).
    fields
        The spectral fields pulled (vorticity, divergence and theta by
        default; vapour and ln ps may be named).
    start_s, end_s
        The model-time window the pull acts in (default: the whole run).
    level_weights
        One factor per level, top first, multiplying ``a`` (default: one
        everywhere).
    record_offset_s
        The record time a model time reads: ``t_record = t + offset``.
    window_s
        The ledger's reporting window in model seconds (one hour).
    """

    tau_s: float = 6.0 * 3600.0
    full_degree: int = 42
    zero_degree: int = 43
    fields: tuple[str, ...] = DEFAULT_FORCED_FIELDS
    start_s: float = 0.0
    end_s: float | None = None
    level_weights: tuple[float, ...] | None = None
    record_offset_s: float = 0.0
    window_s: float = 3600.0

    def __post_init__(self) -> None:
        tau = float(self.tau_s)
        if math.isnan(tau) or tau <= 0.0:
            raise ValueError("forcing tau_s must be positive (math.inf observes without pulling)")
        if int(self.full_degree) < 0 or int(self.zero_degree) <= int(self.full_degree):
            raise ValueError("forcing degrees need 0 <= full_degree < zero_degree")
        fields = tuple(self.fields)
        unknown = [f for f in fields if f not in SPECTRAL_FIELDS]
        if unknown or not fields or len(set(fields)) != len(fields):
            raise ValueError(f"forcing fields must be distinct names from {SPECTRAL_FIELDS}, got {fields}")
        object.__setattr__(self, "fields", fields)
        if not math.isfinite(float(self.start_s)) or float(self.start_s) < 0.0:
            raise ValueError("forcing start_s must be finite and nonnegative")
        if self.end_s is not None and not float(self.end_s) > float(self.start_s):
            raise ValueError("forcing end_s must lie after start_s")
        if not math.isfinite(float(self.record_offset_s)):
            raise ValueError("forcing record_offset_s must be finite")
        if not math.isfinite(float(self.window_s)) or float(self.window_s) <= 0.0:
            raise ValueError("forcing window_s must be finite and positive")
        if self.level_weights is not None:
            weights = tuple(float(w) for w in self.level_weights)
            if not weights or any(not math.isfinite(w) or w < 0.0 or w > 1.0 for w in weights):
                raise ValueError("forcing level_weights must be numbers in [0, 1]")
            object.__setattr__(self, "level_weights", weights)

    @property
    def observe_only(self) -> bool:
        return math.isinf(float(self.tau_s))

    def alpha(self, dt_s: float) -> float:
        """The fraction of the departure applied per step of ``dt_s``."""
        if self.observe_only:
            return 0.0
        return float(-math.expm1(-float(dt_s) / float(self.tau_s)))

    def identity(self) -> dict[str, object]:
        return {
            "tau_s": "inf" if self.observe_only else float(self.tau_s),
            "full_degree": int(self.full_degree),
            "zero_degree": int(self.zero_degree),
            "fields": list(self.fields),
            "start_s": float(self.start_s),
            "end_s": None if self.end_s is None else float(self.end_s),
            "level_weights": None if self.level_weights is None else list(self.level_weights),
            "record_offset_s": float(self.record_offset_s),
            "window_s": float(self.window_s),
        }


def _hours(value: str) -> float:
    return float(value) * 3600.0


def _split_spec(text: str) -> tuple[str, dict[str, str]]:
    """``PATH`` or ``PATH:key=value;key=value``; a Windows drive letter is
    one character before the first colon."""
    head, _, options_text = str(text).partition(":")
    if len(head) == 1 and options_text and options_text[:1] in ("\\", "/"):
        rest, _, options_text = options_text.partition(":")
        head = f"{head}:{rest}"
    options: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in options_text.split(";"))):
        if "=" not in part:
            raise ValueError(f"forcing option {part!r} is not key=value")
        key, value = part.split("=", 1)
        options[key.strip()] = value.strip()
    if not head:
        raise ValueError("a forcing spelling needs a path before its options")
    return head, options


def parse_forcing_spec(text: str) -> tuple[Path, ForcingOptions]:
    """``RECORD[:key=value;...]`` with keys ``tau_h`` or ``tau_s`` (``inf``
    observes only), ``full_degree``, ``zero_degree``, ``fields`` (comma
    list), ``start_h``/``end_h`` (or ``_s``), ``offset_h``/``offset_s``,
    ``level_weights`` (comma list, top first), ``window_h``/``window_s``."""
    head, options = _split_spec(text)
    kwargs: dict[str, object] = {}
    known = {"tau_h", "tau_s", "full_degree", "zero_degree", "fields", "start_h", "start_s",
             "end_h", "end_s", "offset_h", "offset_s", "level_weights", "window_h", "window_s"}
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"unknown forcing option(s) {unknown}; known: {sorted(known)}")
    for key, value in options.items():
        if key in ("tau_h", "tau_s"):
            kwargs["tau_s"] = math.inf if value.lower() in ("inf", "observe") else (
                _hours(value) if key == "tau_h" else float(value))
        elif key in ("full_degree", "zero_degree"):
            kwargs[key] = int(value)
        elif key == "fields":
            kwargs["fields"] = tuple(filter(None, value.split(",")))
        elif key in ("start_h", "start_s"):
            kwargs["start_s"] = _hours(value) if key.endswith("_h") else float(value)
        elif key in ("end_h", "end_s"):
            kwargs["end_s"] = _hours(value) if key.endswith("_h") else float(value)
        elif key in ("offset_h", "offset_s"):
            kwargs["record_offset_s"] = _hours(value) if key.endswith("_h") else float(value)
        elif key == "level_weights":
            kwargs["level_weights"] = tuple(float(v) for v in filter(None, value.split(",")))
        elif key in ("window_h", "window_s"):
            kwargs["window_s"] = _hours(value) if key.endswith("_h") else float(value)
    return Path(head), ForcingOptions(**kwargs)


def parse_record_spec(text: str) -> tuple[Path, dict[str, object]]:
    """``OUT.npz[:degree=42;every_h=1;fields=...;dtype=complex64]`` for the
    record a run writes of itself."""
    head, options = _split_spec(text)
    known = {"degree", "every_h", "every_s", "fields", "dtype"}
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"unknown forcing-record option(s) {unknown}; known: {sorted(known)}")
    out: dict[str, object] = {}
    if "degree" in options:
        out["degree"] = int(options["degree"])
    if "every_h" in options:
        out["every_s"] = _hours(options["every_h"])
    if "every_s" in options:
        out["every_s"] = float(options["every_s"])
    if "fields" in options:
        out["fields"] = tuple(filter(None, options["fields"].split(",")))
    if "dtype" in options:
        out["dtype"] = options["dtype"]
    return Path(head), out


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def _digest(degree: int, times_s: np.ndarray, fields: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(FORCING_RECORD_SCHEMA.encode())
    h.update(f"|degree={int(degree)}|".encode())
    times = np.ascontiguousarray(times_s, dtype=np.float64)
    h.update(times.tobytes())
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        h.update(f"|{name}|{array.dtype.str}|{array.shape}|".encode())
        h.update(array.tobytes())
    return h.hexdigest()


@dataclass
class ForcingRecord:
    """The low degrees of spectral fields at a list of model times.

    ``fields[name]`` has shape ``(ntimes, nlev, degree+1, degree+1)``
    (``(ntimes, degree+1, degree+1)`` for ln ps), in the state's own
    ``(n, m)`` coefficient layout and normalisation; ``times_s`` is
    strictly increasing model time.  ``source`` is descriptive.
    """

    times_s: np.ndarray
    fields: dict[str, np.ndarray]
    degree: int
    source: dict[str, object]

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=np.float64)
        if self.times_s.ndim != 1 or self.times_s.size == 0:
            raise ValueError("a forcing record needs at least one time")
        if not np.all(np.isfinite(self.times_s)) or np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("forcing record times must be finite and strictly increasing")
        if not self.fields:
            raise ValueError("a forcing record needs at least one field")
        k = int(self.degree) + 1
        for name, array in self.fields.items():
            if name not in SPECTRAL_FIELDS:
                raise ValueError(f"forcing record field {name!r} is not a spectral field")
            if not np.iscomplexobj(array):
                raise ValueError(f"forcing record field {name} must be complex")
            want_ndim = 3 if name == SURFACE_FIELD else 4
            if array.ndim != want_ndim or array.shape[0] != self.times_s.size or array.shape[-2:] != (k, k):
                raise ValueError(
                    f"forcing record field {name} has shape {array.shape}; expected "
                    f"(ntimes{', nlev' if want_ndim == 4 else ''}, {k}, {k})")

    @property
    def sha256(self) -> str:
        return _digest(self.degree, self.times_s, self.fields)

    @property
    def nlev(self) -> int | None:
        for name, array in self.fields.items():
            if name != SURFACE_FIELD:
                return int(array.shape[1])
        return None

    def identity(self) -> dict[str, object]:
        return {
            "schema": FORCING_RECORD_SCHEMA,
            "sha256": self.sha256,
            "degree": int(self.degree),
            "fields": sorted(self.fields),
            "ntimes": int(self.times_s.size),
            "first_time_s": float(self.times_s[0]),
            "last_time_s": float(self.times_s[-1]),
            "nlev": self.nlev,
            "source": self.source,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": FORCING_RECORD_SCHEMA,
            "degree": int(self.degree),
            "fields": sorted(self.fields),
            "sha256": self.sha256,
            "source": self.source,
        }
        arrays = {f"field__{name}": np.ascontiguousarray(a) for name, a in self.fields.items()}
        tmp = path.with_name(path.name + ".partial")
        with tmp.open("wb") as stream:
            np.savez_compressed(
                stream, __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)),
                times_s=self.times_s, **arrays)
        tmp.replace(path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> "ForcingRecord":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"forcing record {path} does not exist")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
            if metadata.get("schema") != FORCING_RECORD_SCHEMA:
                raise ValueError(f"{path} is not a forcing record ({metadata.get('schema')!r})")
            fields = {name: np.array(archive[f"field__{name}"]) for name in metadata["fields"]}
            times = np.array(archive["times_s"], dtype=np.float64)
        record = cls(times_s=times, fields=fields, degree=int(metadata["degree"]),
                     source=dict(metadata.get("source") or {}))
        if record.sha256 != metadata.get("sha256"):
            raise ValueError(f"forcing record {path} does not match its own digest")
        return record

    def bracket(self, time_s: float) -> tuple[int, int, float] | None:
        """``(i0, i1, w)`` with the record at ``time_s`` equal to
        ``(1 - w) r[i0] + w r[i1]``, or None outside the record."""
        t = float(time_s)
        times = self.times_s
        if t < times[0] - _TIME_TOLERANCE_S or t > times[-1] + _TIME_TOLERANCE_S:
            return None
        i1 = int(np.searchsorted(times, t, side="left"))
        if i1 < times.size and abs(times[i1] - t) <= _TIME_TOLERANCE_S:
            return i1, i1, 0.0
        if i1 > 0 and abs(times[i1 - 1] - t) <= _TIME_TOLERANCE_S:
            return i1 - 1, i1 - 1, 0.0
        i0 = i1 - 1
        w = (t - times[i0]) / (times[i1] - times[i0])
        return i0, i1, float(w)

    def at(self, name: str, time_s: float, degree: int) -> np.ndarray | None:
        """The record's field at ``time_s`` on degrees ``0..degree``
        (exactly a stored snapshot at a stored time), or None outside."""
        b = self.bracket(time_s)
        if b is None:
            return None
        i0, i1, w = b
        k = int(degree) + 1
        a = self.fields[name]
        r0 = a[i0, ..., :k, :k]
        if i0 == i1 or w == 0.0:
            return np.array(r0)
        r1 = a[i1, ..., :k, :k]
        return (1.0 - w) * r0 + w * r1


class ForcingRecordWriter:
    """Snapshots of a running state's low degrees at a cadence."""

    def __init__(self, *, degree: int = 42, every_s: float = 3600.0,
                 fields: tuple[str, ...] = SPECTRAL_FIELDS, dtype: str = "complex128",
                 source: dict[str, object] | None = None) -> None:
        if int(degree) < 0:
            raise ValueError("forcing record degree must be nonnegative")
        if not math.isfinite(float(every_s)) or float(every_s) <= 0.0:
            raise ValueError("forcing record every_s must be finite and positive")
        unknown = [f for f in fields if f not in SPECTRAL_FIELDS]
        if unknown or not fields:
            raise ValueError(f"forcing record fields must be names from {SPECTRAL_FIELDS}")
        if np.dtype(dtype) not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError("forcing record dtype must be complex64 or complex128")
        self.degree = int(degree)
        self.every_s = float(every_s)
        self.fields = tuple(fields)
        self.dtype = np.dtype(dtype)
        self.source = dict(source or {})
        self._times: list[float] = []
        self._snaps: dict[str, list[np.ndarray]] = {name: [] for name in self.fields}

    def due(self, time_s: float) -> bool:
        q = float(time_s) / self.every_s
        if abs(q - round(q)) * self.every_s > _TIME_TOLERANCE_S:
            return False
        return not self._times or float(time_s) > self._times[-1] + _TIME_TOLERANCE_S

    def offer(self, state: ArwenGlobalState, to_numpy, truncation: int | None = None) -> bool:
        """Take a snapshot of ``state`` if its model time is on the cadence."""
        if not self.due(state.time_s):
            return False
        k = self.degree + 1
        for name in self.fields:
            coeff = getattr(state.atmosphere, name)
            if coeff.shape[-1] < k:
                raise ValueError(
                    f"forcing record degree {self.degree} exceeds the run's truncation "
                    f"{coeff.shape[-1] - 1}")
            # A copy: on the host backend the state's own array is a view.
            self._snaps[name].append(np.array(to_numpy(coeff[..., :k, :k]), dtype=self.dtype, copy=True))
        self._times.append(float(state.time_s))
        return True

    def record(self) -> ForcingRecord:
        if not self._times:
            raise ValueError("the forcing record writer holds no snapshot")
        return ForcingRecord(
            times_s=np.asarray(self._times),
            fields={name: np.stack(snaps) for name, snaps in self._snaps.items()},
            degree=self.degree, source=dict(self.source, every_s=self.every_s,
                                            dtype=self.dtype.name))

    def write(self, path: str | Path) -> dict[str, object]:
        record = self.record()
        written = record.write(path)
        return {"path": str(written), **record.identity()}


def record_from_checkpoints(paths, *, degree: int = 42, fields: tuple[str, ...] = SPECTRAL_FIELDS,
                            dtype: str = "complex128", source: dict | None = None) -> ForcingRecord:
    """A record cut from a finished run's checkpoints (each verified by
    its own digest; ordered by their model time)."""
    from .checkpoint import read_checkpoint

    writer = ForcingRecordWriter(degree=degree, every_s=1.0, fields=fields, dtype=dtype,
                                 source=dict(source or {}, cut_from="checkpoints"))
    rows = []
    for path in paths:
        metadata, arrays = read_checkpoint(path)
        rows.append((float(metadata["time_s"]), str(path), metadata["self_sha256"],
                     metadata.get("config_hash"), arrays))
    rows.sort(key=lambda row: row[0])
    k = int(degree) + 1
    for time_s, _p, _s, _c, arrays in rows:
        if writer._times and time_s <= writer._times[-1]:
            raise ValueError(f"two checkpoints at model time {time_s:g} s")
        for name in fields:
            coeff = arrays[f"atmosphere__{name}"]
            if coeff.shape[-1] < k:
                raise ValueError(f"degree {degree} exceeds the checkpoint's truncation")
            writer._snaps[name].append(np.array(coeff[..., :k, :k], dtype=writer.dtype, copy=True))
        writer._times.append(time_s)
    writer.source["checkpoints"] = [{"path": Path(p).name, "self_sha256": s} for _t, p, s, _c, _a in rows]
    hashes = sorted({c for *_x, c, _a in rows if c})
    writer.source["config_hash"] = hashes[0] if len(hashes) == 1 else hashes
    record = writer.record()
    record.source.pop("every_s", None)
    return record


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


class PullLedger:
    """Per field, level and degree: the departure from the record and the
    pull applied, accumulated on the device and read to the host once per
    window.

    Mean squares are grid means (``degree_power``'s convention: the sum
    over degrees is the mean square of the field on the sphere).
    """

    def __init__(self, *, fields, degree: int, dt_s: float, xp, to_numpy, nlev: int,
                 window_s: float, level_pressure_pa=None) -> None:
        self.fields = tuple(fields)
        self.degree = int(degree)
        self.dt_s = float(dt_s)
        self.xp = xp
        self.to_numpy = to_numpy
        self.nlev = int(nlev)
        self.window_s = float(window_s)
        self.level_pressure_pa = None if level_pressure_pa is None else [float(p) for p in level_pressure_pa]
        k = self.degree + 1
        self._mult = None
        self.steps = 0
        self.first_step: int | None = None
        self.last_step: int | None = None
        self.first_time_s: float | None = None
        self.last_time_s: float | None = None
        self.skipped_outside_window = 0
        self.skipped_uncovered = 0
        # Whole-segment sums (device): per-degree power of departure and
        # pull, per level; and the sum of the pull's coefficients.
        self._dep_power: dict[str, object] = {}
        self._pull_power: dict[str, object] = {}
        self._pull_sum: dict[str, object] = {}
        # Window sums (device) and the closed windows (host).
        self._win_dep: dict[str, object] = {}
        self._win_pull: dict[str, object] = {}
        self._win_steps = 0
        self._win_start: float | None = None
        self.windows: list[dict[str, object]] = []
        self._k = k

    def _power(self, coeff):
        """Per-degree power per level ``(..., n)`` of ``coeff (..., n, m)``."""
        xp = self.xp
        if self._mult is None:
            mult = np.where(np.arange(self._k) == 0, 1.0, 2.0) / (4.0 * math.pi)
            self._mult = xp.asarray(mult)
        p = coeff.real * coeff.real + coeff.imag * coeff.imag
        return xp.sum(p * self._mult.astype(p.dtype), axis=-1)

    def _window_due(self, time_s: float) -> bool:
        return self._win_start is not None and time_s >= self._win_start + self.window_s - _TIME_TOLERANCE_S

    def begin_step(self, step: int, time_s: float) -> None:
        if self._window_due(time_s):
            self._close_window(time_s)
        if self._win_start is None:
            self._win_start = float(time_s)
        if self.first_step is None:
            self.first_step = int(step)
            self.first_time_s = float(time_s)
        self.last_step = int(step)
        self.last_time_s = float(time_s)
        self.steps += 1
        self._win_steps += 1

    def add(self, name: str, departure, pull) -> None:
        dep_p = self._power(departure)
        pull_p = self._power(pull)
        for store, value in ((self._dep_power, dep_p), (self._pull_power, pull_p),
                             (self._win_dep, dep_p), (self._win_pull, pull_p)):
            store[name] = value if name not in store else store[name] + value
        self._pull_sum[name] = pull if name not in self._pull_sum else self._pull_sum[name] + pull

    def _close_window(self, end_s: float) -> None:
        if self._win_steps == 0:
            self._win_start = None
            return
        row: dict[str, object] = {"start_s": self._win_start, "end_s": float(end_s), "steps": self._win_steps,
                                  "fields": {}}
        host_dep = {n: np.asarray(self.to_numpy(v), dtype=np.float64) / self._win_steps
                    for n, v in self._win_dep.items()}
        host_pull = {n: np.asarray(self.to_numpy(v), dtype=np.float64) / self._win_steps
                     for n, v in self._win_pull.items()}
        for name in self.fields:
            if name not in host_dep:
                continue
            row["fields"][name] = {
                "departure_rms": _rms_levels(host_dep[name]),
                "pull_rms_per_day": [r * _DAY_S / self.dt_s for r in _rms_levels(host_pull[name])],
            }
        wind = _wind_levels(host_dep, host_pull, self.dt_s)
        if wind is not None:
            row["fields"]["wind"] = wind
        self.windows.append(row)
        self._win_dep, self._win_pull, self._win_steps = {}, {}, 0
        self._win_start = None

    def skip(self, reason: str) -> None:
        if reason == "window":
            self.skipped_outside_window += 1
        else:
            self.skipped_uncovered += 1

    def finish(self, end_s: float | None = None) -> None:
        if self._win_steps:
            self._close_window(float(end_s if end_s is not None else (self.last_time_s or 0.0) + self.dt_s))

    def mean_pull(self) -> dict[str, np.ndarray]:
        """The time-mean pull per step, coefficients on the pulled degrees."""
        if not self.steps:
            return {}
        return {n: np.asarray(self.to_numpy(v), dtype=np.complex128) / self.steps
                for n, v in self._pull_sum.items()}

    def summary(self) -> dict[str, object]:
        """The table: per field and level, the departure from the record,
        the pull's instantaneous RMS, the RMS of its time mean (the
        systematic part) and, for the scalar fields, its global mean; all
        pulls as tendencies per day."""
        if not self.steps:
            return {"steps": 0, "fields": {}}
        per_day = _DAY_S / self.dt_s
        dep = {n: np.asarray(self.to_numpy(v), dtype=np.float64) / self.steps for n, v in self._dep_power.items()}
        pull = {n: np.asarray(self.to_numpy(v), dtype=np.float64) / self.steps for n, v in self._pull_power.items()}
        mean = self.mean_pull()
        fields: dict[str, object] = {}
        for name in self.fields:
            if name not in dep:
                continue
            mean_power = _degree_power_levels(mean[name])
            levels = []
            nl = 1 if dep[name].ndim == 1 else dep[name].shape[0]
            dep2 = dep[name].reshape(nl, -1)
            pull2 = pull[name].reshape(nl, -1)
            mean2 = mean_power.reshape(nl, -1)
            mean_c = mean[name].reshape(nl, self._k, self._k)
            for lev in range(nl):
                row = {
                    "level": lev if name != SURFACE_FIELD else "surface",
                    "departure_rms": float(math.sqrt(max(dep2[lev].sum(), 0.0))),
                    "pull_rms_per_day": float(math.sqrt(max(pull2[lev].sum(), 0.0))) * per_day,
                    "time_mean_pull_rms_per_day": float(math.sqrt(max(mean2[lev].sum(), 0.0))) * per_day,
                }
                if name in SCALAR_FIELDS:
                    row["global_mean_pull_per_day"] = float(mean_c[lev, 0, 0].real / math.sqrt(4.0 * math.pi)) * per_day
                if self.level_pressure_pa is not None and name != SURFACE_FIELD:
                    row["reference_pressure_hpa"] = round(self.level_pressure_pa[lev] / 100.0, 3)
                levels.append(row)
            fields[name] = {
                "units": UNITS[name],
                "levels": levels,
                "by_degree": {
                    "departure_power": [float(v) for v in dep2.sum(axis=0)],
                    "pull_power_per_day2": [float(v) * per_day * per_day for v in pull2.sum(axis=0)],
                },
            }
        wind = _wind_levels(dep, pull, self.dt_s, mean=mean)
        if wind is not None:
            if self.level_pressure_pa is not None:
                for lev, row in enumerate(wind["levels"]):
                    row["reference_pressure_hpa"] = round(self.level_pressure_pa[lev] / 100.0, 3)
            fields["wind"] = {"units": UNITS["wind"], **wind}
        return {
            "steps": int(self.steps),
            "first_step": self.first_step,
            "last_step": self.last_step,
            "first_time_s": self.first_time_s,
            "last_time_s": self.last_time_s,
            "skipped_outside_window": int(self.skipped_outside_window),
            "skipped_outside_record": int(self.skipped_uncovered),
            "fields": fields,
        }


def _rms_levels(power) -> list[float]:
    p = np.asarray(power, dtype=np.float64)
    if p.ndim == 1:
        return [float(math.sqrt(max(p.sum(), 0.0)))]
    return [float(math.sqrt(max(v, 0.0))) for v in p.sum(axis=-1)]


def _degree_power_levels(coeff) -> np.ndarray:
    c = np.asarray(coeff)
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0) / (4.0 * math.pi)
    return np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=-1)


def _wind_factor(k: int) -> np.ndarray:
    n = np.arange(k, dtype=np.float64)
    f = np.zeros(k)
    f[1:] = EARTH_RADIUS_M ** 2 / (n[1:] * (n[1:] + 1.0))
    return f


def _wind_levels(dep, pull, dt_s, mean=None):
    """The horizontal wind's departure and pull per level from the
    vorticity and divergence powers: ``|u|^2`` per degree is
    ``a^2 / (n (n + 1))`` times ``|zeta|^2 + |delta|^2``."""
    if "vorticity" not in dep or "divergence" not in dep:
        return None
    k = dep["vorticity"].shape[-1]
    f = _wind_factor(k)
    per_day = _DAY_S / float(dt_s)
    d = (dep["vorticity"] + dep["divergence"]) * f
    p = (pull["vorticity"] + pull["divergence"]) * f
    out_levels = []
    mean_p = None
    if mean is not None:
        mean_p = (_degree_power_levels(mean["vorticity"]) + _degree_power_levels(mean["divergence"])) * f
    for lev in range(d.shape[0]):
        row = {
            "level": lev,
            "departure_rms": float(math.sqrt(max(d[lev].sum(), 0.0))),
            "pull_rms_per_day": float(math.sqrt(max(p[lev].sum(), 0.0))) * per_day,
        }
        if mean_p is not None:
            row["time_mean_pull_rms_per_day"] = float(math.sqrt(max(mean_p[lev].sum(), 0.0))) * per_day
        out_levels.append(row)
    if mean is None:
        return {"departure_rms": [r["departure_rms"] for r in out_levels],
                "pull_rms_per_day": [r["pull_rms_per_day"] for r in out_levels]}
    return {"levels": out_levels,
            "by_degree": {"departure_power": [float(v) for v in d.sum(axis=0)],
                          "pull_power_per_day2": [float(v) * per_day * per_day for v in p.sum(axis=0)]}}


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------


class ForcingOperator:
    """``x <- x + a L (x_rec(t) - x)`` before each step, with its ledger.

    ``transform`` supplies the backend and truncation; ``nlev`` the level
    count the record must match; ``level_pressure_pa`` (optional) labels
    the ledger's levels.
    """

    def __init__(self, record: ForcingRecord, options: ForcingOptions, *, transform, dt_s: float,
                 nlev: int, level_pressure_pa=None) -> None:
        self.record = record
        self.options = options
        self.backend = transform.backend
        self.truncation = int(transform.truncation)
        self.dt_s = float(dt_s)
        self.degree = min(int(options.zero_degree) - 1, self.truncation)
        if record.degree < self.degree:
            raise ValueError(
                f"the forcing record holds degrees up to {record.degree}; the pull "
                f"needs {self.degree} (zero_degree {options.zero_degree} at truncation {self.truncation})")
        missing = [f for f in options.fields if f not in record.fields]
        if missing:
            raise ValueError(f"the forcing record does not hold {missing}")
        if record.nlev is not None and int(record.nlev) != int(nlev):
            raise ValueError(f"the forcing record has {record.nlev} levels, the model {nlev}")
        if options.level_weights is not None and len(options.level_weights) != int(nlev):
            raise ValueError(f"forcing level_weights has {len(options.level_weights)} entries, the model {nlev} levels")
        self.nlev = int(nlev)
        self.alpha = options.alpha(self.dt_s)
        xp = self.backend.xp
        k = self.degree + 1
        weights = taper_weights(self.truncation, int(options.full_degree), int(options.zero_degree))[:k]
        n = np.arange(k)[:, None]
        m = np.arange(k)[None, :]
        self._weights_host = np.where(n >= m, weights[:, None], 0.0)
        self._weights = xp.asarray(self._weights_host)
        lw = np.ones(self.nlev) if options.level_weights is None else np.asarray(options.level_weights)
        self._level_weights = xp.asarray(lw)
        self.ledger = PullLedger(fields=options.fields, degree=self.degree, dt_s=self.dt_s, xp=xp,
                                 to_numpy=self.backend.to_numpy, nlev=self.nlev, window_s=options.window_s,
                                 level_pressure_pa=level_pressure_pa)
        self._history_entry = {
            "record_sha256": record.sha256,
            "options": options.identity(),
        }

    def in_window(self, time_s: float) -> bool:
        t = float(time_s)
        if t < float(self.options.start_s) - _TIME_TOLERANCE_S:
            return False
        if self.options.end_s is not None and t >= float(self.options.end_s) - _TIME_TOLERANCE_S:
            return False
        return True

    def _stamp(self, state: ArwenGlobalState, step: int) -> None:
        history = state.physics_state.metadata.get(FORCING_HISTORY_KEY)
        history = list(history) if isinstance(history, list) else []
        last = history[-1] if history else None
        if isinstance(last, dict) and last.get("record_sha256") == self._history_entry["record_sha256"] \
                and last.get("options") == self._history_entry["options"]:
            return
        history.append({**self._history_entry, "from_step": int(step)})
        state.physics_state.metadata[FORCING_HISTORY_KEY] = history

    def apply(self, state: ArwenGlobalState, model=None) -> ArwenGlobalState:
        """The state pulled toward the record at its own model time (the
        same object, untouched, outside the window, outside the record or
        when observing only)."""
        t = state.time_s
        if not self.in_window(t):
            self.ledger.skip("window")
            return state
        rt = t + float(self.options.record_offset_s)
        if self.record.bracket(rt) is None:
            self.ledger.skip("record")
            return state
        xp = self.backend.xp
        k = self.degree + 1
        self.ledger.begin_step(state.step, t)
        fields = list(state.atmosphere.fields())
        moved: list[str] = []
        for name in self.options.fields:
            index = SPECTRAL_FIELDS.index(name)
            x = fields[index]
            target = xp.asarray(self.record.at(name, rt, self.degree), dtype=x.dtype)
            low = x[..., :k, :k]
            departure = (target - low) * self._weights.astype(x.real.dtype)
            if name == SURFACE_FIELD:
                departure[..., 0, 0] = 0.0
                scale = self.alpha
            else:
                scale = self._level_weights.astype(x.real.dtype)[:, None, None] * self.alpha
            pull = departure * scale
            self.ledger.add(name, departure, pull)
            if self.alpha > 0.0 and bool(xp.any(pull != 0)):
                new = x.copy()
                new[..., :k, :k] = low + pull
                fields[index] = new
                moved.append(name)
        if not moved:
            return state
        new_state = ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state)
        if "qv" in moved and model is not None:
            new_state, _n, _t, _f = model._repair_positivity(new_state)
        if model is not None:
            model.release_syntheses()
        self._stamp(new_state, state.step)
        return new_state

    def finish(self, outdir: str | Path | None = None, end_s: float | None = None) -> dict[str, object]:
        """Close the ledger; with ``outdir``, write ``forcing_ledger.json``
        and ``forcing_pull.npz`` there.  Returns the receipt row."""
        self.ledger.finish(end_s)
        summary = self.ledger.summary()
        ledger = {
            "schema": FORCING_LEDGER_SCHEMA,
            "record": self.record.identity(),
            "options": self.options.identity(),
            "pulled_degree": int(self.degree),
            "truncation": int(self.truncation),
            "alpha_per_step": float(self.alpha),
            "dt_s": float(self.dt_s),
            "observe_only": bool(self.options.observe_only),
            "conventions": (
                "departure = L (record - model) before the pull, per level, on the pulled degrees; "
                "pull = the change the operator made, as a tendency per day; time_mean_pull is the "
                "RMS of the pull's time mean (the systematic part, the model-error estimate once the "
                "run is held near the record); RMS are grid means on the sphere; level 0 is the top; "
                "wind is the horizontal wind from vorticity and divergence"),
            **summary,
            "windows": self.ledger.windows,
        }
        row: dict[str, object] = {
            "record": self.record.identity(),
            "options": self.options.identity(),
            "pulled_degree": int(self.degree),
            "alpha_per_step": float(self.alpha),
            "observe_only": bool(self.options.observe_only),
            "steps": int(summary.get("steps", 0)),
        }
        if outdir is not None:
            outdir = Path(outdir)
            path = outdir / FORCING_LEDGER_NAME
            text = json.dumps(ledger, indent=1, sort_keys=True, allow_nan=False)
            path.write_text(text, encoding="utf-8")
            row["ledger"] = {"path": path.name, "sha256": hashlib.sha256(text.encode()).hexdigest()}
            mean = self.ledger.mean_pull()
            if mean:
                pull_path = outdir / FORCING_PULL_NAME
                with pull_path.open("wb") as stream:
                    np.savez_compressed(stream, **{f"mean_pull__{n}": v for n, v in mean.items()})
                row["time_mean_pull"] = {"path": pull_path.name,
                                         "note": "time-mean pull per step, spectral coefficients (n, m) "
                                                 "on the pulled degrees"}
        return row


def level_reference_pressure_pa(cfg) -> list[float] | None:
    """Full-level pressures at the vertical grid's reference surface
    pressure, top first (labels for the ledger), or None when the config
    does not state its grid."""
    try:
        layers = cfg.vertical.describe().get("layers")
        return [float(row["p_full_pa"]) for row in layers]
    except Exception:  # noqa: BLE001 - labels only
        return None


__all__ = [
    "DEFAULT_FORCED_FIELDS",
    "FORCING_HISTORY_KEY",
    "FORCING_LEDGER_NAME",
    "FORCING_LEDGER_SCHEMA",
    "FORCING_PULL_NAME",
    "FORCING_RECORD_SCHEMA",
    "ForcingOperator",
    "ForcingOptions",
    "ForcingRecord",
    "ForcingRecordWriter",
    "PullLedger",
    "level_reference_pressure_pa",
    "parse_forcing_spec",
    "parse_record_spec",
    "record_from_checkpoints",
]
