"""The true device peak of an Arwen Global run, measured at the allocator.

The number a run used to print as its "device peak" was the sizing
model's PREDICTION (:mod:`arwen_global.sizing`), and a probe that
sampled the CuPy pool at its own points read 10.41 GiB on a T533
forty-level native run whose pool reached 28.50 GiB live between those
points (2026-09-02, RTX 5090).  The prediction printed beside this
measurement is now the calibrated model's -- fitted to receipts written
by this very hook (``sizing.DEVICE_PEAK_CALIBRATION``) -- and it is
labelled as a prediction wherever it appears.  A peak can only be read where memory is
taken, so this hook sits in the allocator: it wraps the default pool's
``malloc`` and folds ``pool.used_bytes()`` after every successful
allocation into a running maximum.  Every byte the process puts on the
device through CuPy passes through here, including the Legendre tables
and the initial state, so the maximum is the run's true high-water mark
of live pool bytes.

What it measures: live bytes of the in-process CuPy default memory pool
(``used_bytes``), not the pool's held total (which includes cached free
blocks) and not the card's device-wide residency.  A numpy run installs
nothing and says so in its receipt.
"""
from __future__ import annotations

import time

GIB = 2**30

MECHANISM = "cupy-pool-malloc-hook"


def installed_pool(cp):
    """The pool whose live bytes the CURRENTLY installed allocator spends.

    ``cp.cuda.set_allocator(MemoryAsyncPool().malloc)`` leaves the default
    pool untouched, so a hook that reads ``get_default_memory_pool()``
    reports 0.00 GiB for a run that allocated tens of GiB through the
    driver pool -- the measurement reads zero exactly where the peak is
    the reason the pool was swapped.  The installed allocator is a bound
    ``malloc``, so its owner is the pool that holds the bytes.

    A run also installs the peak hook ON TOP of its allocator, and the
    hook is a callable OBJECT rather than a bound method, so it carries no
    ``__self__``.  MEASURED 2026-09-06 on an RTX 5090: with the async pool
    holding 16,000,000 live bytes under the hook, this returned the
    default ``MemoryPool`` reading 0.  Every caller mid-run -- the block
    release between ingests, the door's reusable-pool credit, the mass
    flux scheme's inter-chunk release -- was therefore still naming the
    default pool while the run spent through another.  A wrapper that
    publishes the pool it wraps is followed, so the chain resolves to the
    allocator that actually holds the bytes.
    """
    try:
        allocator = cp.cuda.get_allocator()
    except AttributeError:
        # A caller holding something that is not a full cupy module (the
        # sizing gate reads whatever `sys.modules` has, deliberately, so
        # that it never stands up a CUDA context to answer a question
        # about bytes) still gets the default pool rather than nothing.
        return cp.get_default_memory_pool()
    for _ in range(8):
        if allocator is None:
            break
        owner = getattr(allocator, "__self__", None)
        if owner is not None and hasattr(owner, "used_bytes"):
            return owner
        if hasattr(allocator, "used_bytes"):
            return allocator
        wrapped = getattr(allocator, "pool", None)
        if wrapped is not None and hasattr(wrapped, "used_bytes"):
            return wrapped
        allocator = getattr(allocator, "wrapped_allocator", None)
    return cp.get_default_memory_pool()


# The three allocators a run can spend its device bytes through.  Every
# one of them is bit-neutral: an address does not change a floating-point
# result, and the ten-step T255 checkpoint gate is run under each.
DEVICE_ALLOCATORS = ("default", "async", "slab")

# The out-of-pool tax: bytes the context, the driver and cuFFT hold on the
# card outside any pool.  MEASURED 2026-09-06, card held minus pool held:
# 0.42 / 0.44 / 0.37 / 0.73 GiB at T63 / T127 / T255 / T383.  The arena is
# sized under it so that taking the arena does not itself run the card out.
OUT_OF_POOL_RESERVE_BYTES = int(0.75 * GIB)


class PoolAllocatorChoice:
    """A CuPy pool selected by name, with the receipt row that says which.

    ``default`` leaves the process's allocator exactly as it was found
    (today's behaviour); ``async`` installs ``MemoryAsyncPool``, the
    driver's own ``cudaMallocAsync`` pool, which is what made a T533 step
    profile run at all on 2026-09-06.
    """

    def __init__(self, name: str, *, fallback_reason: str = ""):
        self.name = str(name)
        self.fallback_reason = str(fallback_reason)
        self._pool = None
        self._previous_allocator = None
        self._installed = False

    def install(self) -> "PoolAllocatorChoice":
        import cupy as cp

        if self.name == "default":
            self._pool = cp.get_default_memory_pool()
            return self
        if self.name != "async":
            raise ValueError(f"unknown pool allocator {self.name!r}")
        self._pool = cp.cuda.MemoryAsyncPool()
        self._previous_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(self._pool.malloc)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        import cupy as cp

        cp.cuda.set_allocator(self._previous_allocator)
        self._installed = False

    def receipt(self) -> dict[str, object]:
        row: dict[str, object] = {
            "allocator": self.name,
            "measures": (
                "the CuPy default memory pool, the process allocator left as "
                "found" if self.name == "default"
                else "the CUDA driver's cudaMallocAsync pool (MemoryAsyncPool)"
            ),
        }
        if self.fallback_reason:
            # The run asked for an allocator it could not have.  It is
            # recorded here rather than dropped, because a run that
            # silently spent through a different allocator than the one
            # its receipt names is a measurement defect.
            row["requested_allocator"] = "slab"
            row["fallback_reason"] = self.fallback_reason
        return row


def slab_arena_bytes(
    predicted_peak_bytes: int | None, free_bytes: int | None
) -> tuple[int, int | None, str]:
    """The slab's first segment, its ceiling, and the sentence that says why.

    The first segment is a SLICE of the door's prediction, not the whole
    of it: MEASURED 2026-09-06 on an RTX 5070 Ti, the prediction over-read
    a T85 reference run's live peak by 6.2x (3.57 GiB predicted, 0.572 GiB
    live), so an arena taken at the prediction held seven times what the
    run used and the fragmentation figure the receipt reports would have
    been a sizing error wearing a fragmentation's name.  The allocator
    extends to what the run asks for instead.

    ``(0, None, reason)`` means no arena can be taken at all and the run
    falls back to a pool, with ``reason`` printed in the receipt rather
    than dropped.
    """
    from .bands import INITIAL_FRACTION, MIN_SEGMENT_BYTES

    ceiling = None
    if free_bytes is not None and int(free_bytes) > 0:
        ceiling = int(free_bytes) - OUT_OF_POOL_RESERVE_BYTES
        if ceiling < MIN_SEGMENT_BYTES:
            return 0, None, (
                f"the card reports {int(free_bytes) / GIB:.2f} GiB free, which "
                f"leaves under {MIN_SEGMENT_BYTES / GIB:.2f} GiB once the "
                f"{OUT_OF_POOL_RESERVE_BYTES / GIB:.2f} GiB the context and "
                "cuFFT hold outside any pool is set aside, so no arena could "
                "be taken"
            )
    if predicted_peak_bytes is None or int(predicted_peak_bytes) <= 0:
        first = MIN_SEGMENT_BYTES
        reason = (
            f"{first / GIB:.2f} GiB first segment: no device-peak prediction "
            "was available for this configuration, so the arena starts at the "
            "minimum segment and extends to what the run asks for"
        )
    else:
        first = max(
            MIN_SEGMENT_BYTES, int(int(predicted_peak_bytes) * INITIAL_FRACTION)
        )
        reason = (
            f"{first / GIB:.2f} GiB first segment = {INITIAL_FRACTION:.3f} x the "
            f"door's {int(predicted_peak_bytes) / GIB:.2f} GiB device-peak "
            "prediction; the arena extends to what the run asks for"
        )
    if ceiling is not None:
        if first > ceiling:
            first = ceiling
            reason = (
                f"{first / GIB:.2f} GiB first segment = the card's "
                f"{int(free_bytes) / GIB:.2f} GiB free less the "
                f"{OUT_OF_POOL_RESERVE_BYTES / GIB:.2f} GiB out-of-pool reserve"
            )
        reason += (
            f", up to a {ceiling / GIB:.2f} GiB ceiling (the card's free VRAM "
            "less the out-of-pool reserve)"
        )
    return int(first), ceiling, reason


def _free_device_bytes(cp) -> int | None:
    try:
        free, _total = cp.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 - a card that will not report is not a run failure
        return None
    return int(free)


def select_device_allocator(
    name: str, backend_name: str, *, predicted_peak_bytes: int | None = None
):
    """Install the run's allocator and return it, or ``None`` for numpy."""
    key = str(name).strip().lower()
    if key not in DEVICE_ALLOCATORS:
        raise ValueError(
            f"device_allocator must be one of {DEVICE_ALLOCATORS}, got {name!r}"
        )
    if str(backend_name) != "cupy":
        return None
    import cupy as cp

    if key != "slab":
        return PoolAllocatorChoice(key).install()
    from .bands import SlabAllocator

    arena, ceiling, reason = slab_arena_bytes(
        predicted_peak_bytes, _free_device_bytes(cp)
    )
    if arena <= 0:
        return PoolAllocatorChoice("default", fallback_reason=reason).install()
    return SlabAllocator(
        arena, ceiling_bytes=ceiling, arena_reason=reason
    ).install()


#: The environment variable CuPy reads its default pool's limit from.  A
#: user who has set it has chosen the pool's ceiling (``0`` is CuPy's own
#: spelling of "no limit"), and the run then sets none of its own.
POOL_LIMIT_ENV = "CUPY_GPU_MEMORY_LIMIT"

#: How far the run raises its own pool limit when an allocation does not fit
#: under it.  The limit is the door's figure for the plan, and a figure can
#: read under the run (the model's held-out error is several percent; a
#: shape nobody has run under a limit has no measurement at all), so the
#: limit yields rather than end a forecast the card had room for: it is
#: raised by this much past what the allocation needs, the allocation is
#: retried, and the receipt counts the lift.  Small, because the lifted
#: level is where the pool stays: MEASURED 2026-11-12 at T383 L40 on an RTX
#: 5070 Ti, 3 h from a limit at the run's live peak (11.50 GiB), the pool
#: was lifted 8 times in 32 MiB steps to 13.15 GiB and the step took the
#: unlimited run's time (2.289 s against 2.287 to 2.296).
POOL_LIMIT_LIFT_BYTES = 32 * 2**20

#: CuPy rounds every pool request up to this many bytes.
_POOL_ROUND_BYTES = 512

#: A ``get_limit()`` reading at or above this is no limit: the driver's
#: async pool answers 2**64 - 1 when nothing limited it (hold_pool_to).
_NO_LIMIT_READING = 2**63


def environment_pool_limit(environ=None) -> str | None:
    """What the user set ``CUPY_GPU_MEMORY_LIMIT`` to, or ``None``.

    Stripped; an empty value is CuPy's "not set" and is ``None`` here."""

    import os

    value = (os.environ if environ is None else environ).get(POOL_LIMIT_ENV)
    value = None if value is None else str(value).strip()
    return value or None


#: The phrases a CUDA library puts in an out-of-memory error.  CuPy's pool
#: raises ``OutOfMemoryError``; a refusal outside the pool (a cuFFT plan's
#: work area, a driver call) arrives as ``CUDARuntimeError`` /
#: ``CUDADriverError`` / ``CuFFTError`` carrying one of these.  MEASURED
#: 2026-11-12 on an RTX 5090 standing in for a 16 GB card with other
#: processes on it: 11 of 12 deaths were the pool's ``OutOfMemoryError``
#: and one was ``CUDA_ERROR_OUT_OF_MEMORY`` from a driver call.
OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "cudaErrorMemoryAllocation",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUFFT_ALLOC_FAILED",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUSOLVER_STATUS_ALLOC_FAILED",
)


def is_out_of_memory(error) -> bool:
    """Whether ``error``, or an exception it was raised from, is the card
    running out of memory: CuPy's ``OutOfMemoryError`` by name, or a CUDA
    library error whose text says so (:data:`OUT_OF_MEMORY_MARKERS`).

    A host ``MemoryError`` is NOT one (the card fallback cannot help a host
    that is out of memory), and neither is :class:`DeviceMemoryExhausted`,
    which is this package's own verdict after a fallback was tried.
    """

    seen = 0
    while error is not None and seen < 8:
        if isinstance(error, DeviceMemoryExhausted):
            return False
        if type(error).__name__ == "OutOfMemoryError":
            return True
        kind = type(error).__name__
        if kind != "MemoryError" and (
                kind.startswith(("CUDA", "CuFFT", "CUFFT", "CuBLAS", "CUBLAS",
                                 "CuSolver", "CUSOLVER"))
                or "cuda" in type(error).__module__.lower()):
            text = str(error)
            if any(marker in text for marker in OUT_OF_MEMORY_MARKERS):
                return True
        error = error.__cause__ or error.__context__
        seen += 1
    return False


class DeviceMemoryExhausted(RuntimeError):
    """The card ran out of memory mid-run and no lighter plan fits it.

    Raised by :func:`arwen_global.runner.run` in place of the CUDA error
    once a fallback was weighed and none was possible; the message names
    the cause, the free memory seen, what else holds the card and the
    flags to resume with.  ``RuntimeError`` so the doors print it as one
    line, at the device exit code (``cli.EXIT_DEVICE``).
    """


def _card_free_bytes_now() -> int | None:
    """``memGetInfo``'s free bytes from inside a process that already holds
    a context (the allocator hook's), or ``None``."""

    import sys

    cupy = sys.modules.get("cupy")
    if cupy is None:
        return None
    try:
        free, _total = cupy.cuda.runtime.memGetInfo()
    except Exception:  # noqa: BLE001 - a card that will not report is reported as not reporting
        return None
    return int(free)


#: What the door leaves unplanned on a card that other processes share or
#: that drives a display, because such a card's other tenants can grow or
#: multiply after the door has read it.  MEASURED 2026-11-12 on an RTX
#: 5070 Ti (16 GB) at T383 L40, the one-band semi-Lagrangian plan: with
#: 512 MiB held by another process from before the start the run kept one
#: band and its tightest moment left 150 MiB of the card free, and with
#: five 222 MiB contexts opening after the start (1,110 MiB) one segment in
#: eleven died.  Half a GiB is room for another tenant that size to arrive
#: or grow after the start; it is not room for five, and the run's own
#: fallback (``runner.run``) is what covers a larger arrival.
SHARED_CARD_MARGIN_BYTES = 512 * 2**20


def _visible_card_row(rows, environ) -> dict | None:
    """The ``nvidia-smi`` row of the card this process's CUDA device 0 is,
    or ``None`` when that cannot be told without opening a context."""

    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        first = str(visible).split(",")[0].strip()
        if not first or first == "-1":
            return None
        if first.startswith(("GPU-", "MIG-")):
            for row in rows:
                if row["uuid"].startswith(first):
                    return row
            return None
        # A bare index counts in CUDA's order, which is nvidia-smi's only
        # when the order is the bus order or there is one card to order.
        if first.isdigit() and (
                len(rows) == 1
                or environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"):
            for row in rows:
                if row["index"] == first:
                    return row
        return None
    if len(rows) == 1:
        return rows[0]
    if environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID":
        for row in rows:
            if row["index"] == "0":
                return row
    return None


def card_neighbours(*, environ=None, run=None, own_pid=None) -> dict | None:
    """What OTHER processes hold on this run's card, and whether it drives
    a display, read through ``nvidia-smi`` (no CUDA context is opened).

    ``{"card": uuid, "display_active": bool | None, "processes": [{"pid",
    "name", "used_bytes"}], "other_bytes": int, "own_bytes": int | None}``,
    this process left out of ``processes`` and reported as ``own_bytes``;
    ``None`` when ``nvidia-smi`` is missing, fails, or the card cannot be
    told apart from the others on the machine.  ``run`` is the
    ``subprocess.run`` seam, for tests.
    """

    import os
    import shutil
    import subprocess

    environ = os.environ if environ is None else environ
    own_pid = os.getpid() if own_pid is None else int(own_pid)
    runner = subprocess.run if run is None else run
    smi = shutil.which("nvidia-smi") if run is None else "nvidia-smi"
    if not smi:
        return None

    def query(*args) -> list[list[str]] | None:
        try:
            done = runner([smi, *args, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=30)
        except Exception:  # noqa: BLE001 - no reading is reported as no reading
            return None
        if getattr(done, "returncode", 1) != 0:
            return None
        return [[field.strip() for field in line.split(",")]
                for line in str(done.stdout or "").splitlines() if line.strip()]

    cards = query("--query-gpu=index,uuid,display_active")
    if not cards:
        return None
    rows = [{"index": c[0], "uuid": c[1],
             "display_active": (None if len(c) < 3 else
                                {"enabled": True, "disabled": False}.get(
                                    c[2].lower()))}
            for c in cards if len(c) >= 2]
    card = _visible_card_row(rows, environ)
    if card is None:
        return None
    apps = query("--query-compute-apps=pid,process_name,used_memory,gpu_uuid")
    if apps is None:
        return None
    processes, own = [], None
    for app in apps:
        if len(app) < 4 or app[3] != card["uuid"]:
            continue
        try:
            pid, used = int(app[0]), int(app[2]) * 2**20
        except ValueError:
            continue
        if pid == own_pid:
            own = used
            continue
        processes.append({"pid": pid, "name": app[1], "used_bytes": used})
    return {
        "card": card["uuid"],
        "display_active": card["display_active"],
        "processes": processes,
        "other_bytes": sum(p["used_bytes"] for p in processes),
        "own_bytes": own,
    }


def neighbours_phrase(neighbours: dict | None) -> str:
    """``3 other processes hold 1,390 MiB on this card (pid 11 python3 222
    MiB, ...)``, or what could not be read."""

    if neighbours is None:
        return "what else holds the card could not be read (nvidia-smi)"
    procs = neighbours.get("processes") or []
    display = neighbours.get("display_active")
    tail = "; the card drives a display" if display else ""
    if not procs:
        return "no other process holds memory on this card" + tail
    listed = ", ".join(
        f"pid {p['pid']} {p['name']} {p['used_bytes'] >> 20:,} MiB"
        for p in procs[:8])
    more = "" if len(procs) <= 8 else f", and {len(procs) - 8} more"
    return (f"{len(procs)} other process{'' if len(procs) == 1 else 'es'} "
            f"hold{'s' if len(procs) == 1 else ''} "
            f"{int(neighbours['other_bytes']) >> 20:,} MiB on this card "
            f"({listed}{more}){tail}")


class DevicePeakTracker:
    """Allocator hook: running maximum of ``pool.used_bytes()`` after each
    successful allocation through ``pool.malloc``.

    ``pool`` needs ``malloc(size)`` and ``used_bytes()``; the CuPy default
    pool is the production one, and a stand-in with the same two methods
    lets the folding logic be proved on the CPU.
    """

    def __init__(self, pool):
        self._pool = pool
        self._malloc = pool.malloc
        self.peak_used_bytes = int(pool.used_bytes())
        self.peak_total_bytes = int(pool.total_bytes())
        self.held_at_peak_bytes = int(pool.total_bytes())
        self.allocations = 0
        self.hook_seconds = 0.0
        self._previous_allocator = None
        self._installed = False
        # The run's own limit on this pool (hold_pool_to): what it was set
        # to, who set it, and how often it had to yield.
        self._limit_row: dict[str, object] = {
            "set_by": "nobody",
            "limit_bytes": None,
            "reason": "no limit was asked of this pool",
        }
        self._own_limit_bytes = 0
        self._limit_found = None
        # The card refusing an allocation: how often, and the last time
        # what the pool and the card held (``refusal_row``), so a run that
        # falls back or stops can say what it saw.
        self.refusals = 0
        self.last_refusal: dict[str, object] | None = None

    def hold_pool_to(self, limit_bytes, *, reason: str = "",
                     environ=None) -> dict[str, object]:
        """Hold the pool to ``limit_bytes``: the run's own ceiling on what
        its pool may take from the card.

        WHY.  A CuPy pool keeps every block it has ever been given, and it
        asks the card for a new one whenever no cached block serves a
        request, so on a card with room it grows into the room.  Under a
        limit CuPy hands its cached blocks back and asks again before it
        gives up, which is the whole mechanism: the run holds what its
        allocation pattern needs and not what the card happened to have.

        THE USER'S LIMIT STANDS.  With ``CUPY_GPU_MEMORY_LIMIT`` set in the
        environment (``0`` included: that is CuPy's "no limit") or a limit
        already on the pool, nothing is changed and the row says whose
        limit it is.

        THE RUN'S OWN LIMIT YIELDS.  It is a figure priced before the run,
        so an allocation that does not fit under it raises it
        (:data:`POOL_LIMIT_LIFT_BYTES`) and is retried; the receipt counts
        the lifts.  A pool with no ``set_limit`` (the driver's async pool,
        the slab) is left alone.
        """

        row: dict[str, object] = {"asked_bytes": None if limit_bytes is None
                                  else int(limit_bytes)}
        asked = environment_pool_limit(environ)
        get_limit = getattr(self._pool, "get_limit", None)
        set_limit = getattr(self._pool, "set_limit", None)
        found = None
        if get_limit is not None:
            try:
                found = int(get_limit())
            except Exception:  # noqa: BLE001 - a pool that will not say has no limit to respect
                found = None
        if found is not None and found >= _NO_LIMIT_READING:
            # MEASURED 2026-11-12 on an RTX 5070 Ti (T383, four bands,
            # --device-allocator async): the driver's pool answers
            # get_limit() with 2**64 - 1 when nothing limited it, and the
            # receipt then said "the pool already carried a limit when the
            # run found it" of a pool nobody had limited.  That reading is
            # no limit, as 0 is on the default pool.
            found = None
        if asked is not None:
            row.update(set_by="user", limit_bytes=found, reason=(
                f"{POOL_LIMIT_ENV}={asked} is set in the environment, so the "
                "pool's limit is the user's and the run set none of its own"))
        elif found:
            row.update(set_by="caller", limit_bytes=found, reason=(
                "the pool already carried a limit when the run found it, so "
                "the run set none of its own"))
        elif limit_bytes is None or int(limit_bytes) <= 0:
            row.update(set_by="nobody", limit_bytes=None, reason=(
                reason or "the plan carries no pool figure to hold the pool to"))
        elif set_limit is None or get_limit is None:
            row.update(set_by="nobody", limit_bytes=None, reason=(
                f"{type(self._pool).__name__} takes no limit, so the pool is "
                "not held"))
        else:
            # A pool that already holds more than the figure (a launcher's
            # reservation on a shared card, grown before the model started)
            # is held to what it holds: the bytes are the run's own, and a
            # limit under them would make CuPy hand the reservation back.
            try:
                already = int(self._pool.total_bytes())
            except Exception:  # noqa: BLE001 - a pool that will not say holds nothing to respect
                already = 0
            limit_bytes = max(int(limit_bytes), already)
            try:
                set_limit(size=int(limit_bytes))
                held = int(get_limit()) == int(limit_bytes)
            except Exception:  # noqa: BLE001 - a pool that refuses a limit is left as found
                held = False
            if held:
                self._own_limit_bytes = int(limit_bytes)
                self._limit_found = 0 if found is None else int(found)
                row.update(set_by="sizer", limit_bytes=int(limit_bytes),
                           lifts=0, lifted_bytes=0, reason=(
                    reason or "the pool figure the door priced this plan at"))
                if already and int(limit_bytes) == already:
                    row["reason"] = (
                        "the pool already held more than the plan's pool "
                        "figure when the run found it, so it is held to "
                        "what it holds")
            else:
                row.update(set_by="nobody", limit_bytes=None, reason=(
                    f"{type(self._pool).__name__} did not take the limit, so "
                    "the pool is not held"))
        self._limit_row = row
        return dict(row)

    def _lift_pool_limit(self, size, error) -> bool:
        """Raise the run's own limit past an allocation that did not fit
        under it; ``False`` when the limit was not the reason."""

        if not self._own_limit_bytes:
            return False
        if type(error).__name__ != "OutOfMemoryError":
            return False
        try:
            limit = int(self._pool.get_limit())
            total = int(self._pool.total_bytes())
        except Exception:  # noqa: BLE001 - the allocation's own error stands
            return False
        if limit != self._own_limit_bytes:
            # Somebody else moved the limit since the run set it: it is no
            # longer the run's to raise.
            return False
        need = -(-int(size) // _POOL_ROUND_BYTES) * _POOL_ROUND_BYTES
        if total + need <= limit:
            # The request fits under the limit and still failed: the CARD is
            # out of memory, and raising a limit does not make more of it.
            return False
        raised = max(limit, total + need) + int(POOL_LIMIT_LIFT_BYTES)
        self._pool.set_limit(size=raised)
        self._own_limit_bytes = raised
        row = self._limit_row
        row["lifts"] = int(row.get("lifts", 0)) + 1
        row["lifted_bytes"] = raised - int(row["limit_bytes"])
        # WHEN the limit last yielded, in allocations since the hook went
        # in: a limit that stops yielding early in a run found its level,
        # and one still yielding at the end did not.
        row["last_lift_at_allocation"] = int(self.allocations)
        return True

    def _refused(self, size, error) -> bool:
        """The card refused ``size`` bytes: record what the pool and the
        card held, hand the pool's cached free blocks back once, and say
        whether the allocation should be tried again.

        CuPy's default pool already frees its cached blocks and retries
        before it raises (``MemoryPool._try_malloc``), so on that pool the
        retry here gains nothing and costs one call; the driver's async
        pool does not, and there it is the difference.  Either way the
        record is what a fallback or a clean stop reports.
        """

        if not is_out_of_memory(error):
            return False
        self.refusals += 1
        row: dict[str, object] = {
            "request_bytes": int(size),
            "allocation": int(self.allocations),
            "error": f"{type(error).__name__}: {error}"[:400],
        }
        for key, read in (("pool_total_bytes", "total_bytes"),
                          ("pool_used_bytes", "used_bytes"),
                          ("pool_limit_bytes", "get_limit")):
            try:
                row[key] = int(getattr(self._pool, read)())
            except Exception:  # noqa: BLE001 - a pool that will not say is recorded as not saying
                row[key] = None
        row["card_free_bytes"] = _card_free_bytes_now()
        self.last_refusal = row
        release = getattr(self._pool, "free_all_blocks", None)
        if release is None:
            return False
        try:
            release()
        except Exception:  # noqa: BLE001 - the allocation's own error stands
            return False
        return True

    def __call__(self, size):
        retried = False
        while True:
            try:
                memory = self._malloc(size)
                break
            except Exception as error:  # noqa: BLE001 - re-raised unless the run's own limit was the reason
                if self._lift_pool_limit(size, error):
                    continue
                if retried or not self._refused(size, error):
                    raise
                retried = True
                if self.last_refusal is not None:
                    self.last_refusal["retried"] = True
        started = time.perf_counter()
        used = self._pool.used_bytes()
        # The held total is folded beside the live total at every
        # allocation, so the receipt carries the fragmentation gap (held
        # over live) the memory gate MEM-2 reads.  Reading it only at the
        # end reports whatever the last free left behind, which at T383
        # was 4 GiB under the true maximum.
        total = self._pool.total_bytes()
        if used > self.peak_used_bytes:
            self.peak_used_bytes = int(used)
            self.held_at_peak_bytes = int(total)
        if total > self.peak_total_bytes:
            self.peak_total_bytes = int(total)
        self.allocations += 1
        self.hook_seconds += time.perf_counter() - started
        return memory

    @property
    def pool(self):
        """The allocator this hook wraps, so ``installed_pool`` resolves
        through the hook to the pool that actually holds the bytes."""
        return self._pool

    def install(self) -> "DevicePeakTracker":
        import cupy as cp

        if self._installed:
            raise RuntimeError("the device peak tracker is already installed")
        self._previous_allocator = cp.cuda.get_allocator()
        # The hook sits ON TOP of whatever allocator was installed, not
        # beside it: an external probe that wrapped the pool before the
        # run (a per-method tracer hooked the same way) keeps seeing
        # every allocation, and the tracker still folds the pool's live
        # bytes after each one.  Calling pool.malloc directly here read
        # a 0.00 GiB global peak on such a probe for a T533 run that
        # allocated tens of GiB through this hook.
        if self._previous_allocator is not None:
            self._malloc = self._previous_allocator
        cp.cuda.set_allocator(self)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        import cupy as cp

        cp.cuda.set_allocator(self._previous_allocator)
        self._malloc = self._pool.malloc
        self._installed = False
        self.release_pool_limit()

    def release_pool_limit(self) -> None:
        """Put the pool's limit back as the run found it, so the process's
        pool is left as it was found."""

        if not self._own_limit_bytes:
            return
        try:
            if int(self._pool.get_limit()) == self._own_limit_bytes:
                self._pool.set_limit(size=int(self._limit_found or 0))
        except Exception:  # noqa: BLE001 - a pool that is gone has no limit to restore
            pass
        self._own_limit_bytes = 0

    def pool_limit_receipt(self) -> dict[str, object]:
        """The receipt's ``device_memory.pool_limit`` row."""

        row = dict(self._limit_row)
        if row.get("set_by") == "sizer":
            row["limit_bytes_at_end"] = (
                int(row["limit_bytes"]) + int(row.get("lifted_bytes", 0)))
        return row

    def receipt(self) -> dict[str, object]:
        return {
            "backend": "cupy",
            "mechanism": MECHANISM,
            "pool": type(self._pool).__name__,
            "measures": (
                "maximum of the live bytes (used_bytes) of the CuPy pool "
                f"this run's allocator spends ({type(self._pool).__name__}) "
                "after each allocation, over the whole run: tables, initial "
                "state and every step"
            ),
            "peak_used_bytes": int(self.peak_used_bytes),
            "peak_used_gib": round(self.peak_used_bytes / GIB, 3),
            "used_bytes_at_end": int(self._pool.used_bytes()),
            "pool_total_bytes_at_end": int(self._pool.total_bytes()),
            # The fragmentation gap: what the allocator holds from the
            # device over what the run has live.  MEM-2 reads
            # held_over_live_peak.
            "peak_total_bytes": int(self.peak_total_bytes),
            "peak_total_gib": round(self.peak_total_bytes / GIB, 3),
            "held_at_live_peak_bytes": int(self.held_at_peak_bytes),
            "held_over_live_peak": (
                None if self.peak_used_bytes <= 0
                else round(self.peak_total_bytes / self.peak_used_bytes, 4)
            ),
            "allocations": int(self.allocations),
            "hook_seconds": float(self.hook_seconds),
            # The run's own ceiling on the pool, whose it was, and how
            # often it had to yield (hold_pool_to).
            "pool_limit": self.pool_limit_receipt(),
            # How often the CARD refused an allocation, and what the pool
            # and the card held the last time it did.
            "device_refusals": int(self.refusals),
            "last_refusal": (None if self.last_refusal is None
                             else dict(self.last_refusal)),
        }


def start_device_peak_tracking(backend_name: str) -> DevicePeakTracker | None:
    """Install the hook for a cupy run; ``None`` (nothing touched) otherwise."""
    if str(backend_name) != "cupy":
        return None
    import cupy as cp

    return DevicePeakTracker(installed_pool(cp)).install()


def disable_fft_plan_cache(backend_name: str):
    """Turn the CuPy cuFFT plan cache off for a cupy run; returns the restore.

    Memory only.  CuPy keeps up to 16 cuFFT plans per device and every
    cached plan keeps its work area allocated from the pool: measured
    1.15 GiB for the six-field rfft AND 1.15 GiB for its irfft at T533
    float32 (the work area equals the data; allocator probe,
    2026-09-02), so the step's distinct transform shapes left several
    GiB resident between calls, counted as live pool bytes.  With the
    cache off every call plans afresh and its work area lives only for
    the call.  cuFFT planning is deterministic -- the same parameters
    select the same kernels -- so a re-planned transform returns the
    cached plan's bits.  ``restore()`` puts the previous cache size back;
    a numpy run touches nothing and gets a no-op.
    """
    if str(backend_name) != "cupy":
        return lambda: None
    import cupy as cp

    cache = cp.fft.config.get_plan_cache()
    previous = cache.get_size()
    cache.set_size(0)

    def restore() -> None:
        cache.set_size(previous)

    return restore


def device_memory_receipt(
    tracker: DevicePeakTracker | None, backend_name: str, *,
    sizing_model_peak_bytes: int | None,
    allocator=None,
) -> dict[str, object]:
    """The receipt's ``device_memory`` row.

    ``sizing_model_peak_bytes`` is the pre-run PREDICTION the door printed
    before anything was allocated (the calibrated model of
    :mod:`arwen_global.sizing`); it rides along under its own name
    so a reader can weigh the model against the measurement, and it is
    never the number reported as the peak.
    """
    if tracker is None:
        row: dict[str, object] = {
            "backend": str(backend_name),
            "mechanism": "not-installed",
            "measures": (
                f"nothing: backend={backend_name!r} allocates no device "
                "memory, so no allocator hook was installed"
            ),
            "peak_used_bytes": None,
            "peak_used_gib": None,
        }
    else:
        row = tracker.receipt()
    row["sizing_model_device_peak_bytes"] = (
        None if sizing_model_peak_bytes is None else int(sizing_model_peak_bytes)
    )
    if allocator is not None:
        # Which allocator spent the bytes this row measures, and its own
        # arithmetic.  A peak has no meaning without it: the same run
        # under two allocators holds two different totals.
        row["allocator"] = allocator.receipt()
    row["sizing_model_note"] = (
        "prediction of arwen_global.sizing made before the run, not a "
        "measurement: the calibrated model of the allocator-measured pool "
        "peak (fitted to DEVICE_PEAK_CALIBRATION), labelled extrapolated by "
        "the door when the configuration is outside the measured domain"
    )
    return row


def device_peak_sentence(row: dict[str, object]) -> str:
    """One line for the door: the measured peak, then the prediction."""
    predicted = row.get("sizing_model_device_peak_bytes")
    predicted_text = (
        "" if predicted is None
        else f"; calibrated model predicted {predicted / GIB:.2f} GiB"
    )
    if row.get("peak_used_bytes") is None:
        return (
            f"memory: no device peak measured (backend={row.get('backend')!r}"
            f", allocator hook not installed){predicted_text}"
        )
    allocator = row.get("allocator") or {}
    named = allocator.get("allocator")
    gap = row.get("held_over_live_peak")
    gap_text = "" if gap is None else f", held/live {gap:.3f}"
    allocator_text = "" if not named else f" through the {named} allocator"
    return (
        f"memory: {row['peak_used_bytes'] / GIB:.2f} GiB device peak "
        f"measured at the allocator{allocator_text} "
        f"({row['allocations']} allocations{gap_text}, "
        f"hook {row['hook_seconds']:.2f} s){pool_limit_phrase(row)}"
        f"{predicted_text}"
    )


def pool_limit_phrase(row: dict[str, object]) -> str:
    """``; pool held to X GiB``: the limit the run kept its pool under and
    whether it had to raise it, or ``""`` when the run set none."""
    limit = row.get("pool_limit") or {}
    if limit.get("set_by") != "sizer" or not limit.get("limit_bytes"):
        return ""
    text = f"; pool held to {int(limit['limit_bytes']) / GIB:.2f} GiB"
    lifts = int(limit.get("lifts", 0) or 0)
    if lifts:
        text += (f", raised {lifts} time{'' if lifts == 1 else 's'} to "
                 f"{int(limit['limit_bytes_at_end']) / GIB:.2f} GiB because "
                 "the run needed more")
    return text


__all__ = [
    "DEVICE_ALLOCATORS",
    "DeviceMemoryExhausted",
    "DevicePeakTracker",
    "OUT_OF_MEMORY_MARKERS",
    "SHARED_CARD_MARGIN_BYTES",
    "card_neighbours",
    "is_out_of_memory",
    "neighbours_phrase",
    "MECHANISM",
    "OUT_OF_POOL_RESERVE_BYTES",
    "POOL_LIMIT_ENV",
    "POOL_LIMIT_LIFT_BYTES",
    "environment_pool_limit",
    "PoolAllocatorChoice",
    "installed_pool",
    "select_device_allocator",
    "slab_arena_bytes",
    "device_memory_receipt",
    "device_peak_sentence",
    "pool_limit_phrase",
    "disable_fft_plan_cache",
    "start_device_peak_tracking",
]
