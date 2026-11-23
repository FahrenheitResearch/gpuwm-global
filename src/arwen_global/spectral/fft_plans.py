"""Kept cuFFT plans whose work area lives only for the call.

Why this exists.  CuPy's own plan cache keeps every plan WITH its work
area, and at T533 float32 a six-field transform's work area was measured
at 1.15 GiB (the size of the data), so the step's distinct shapes left
several GiB resident between calls.  The run therefore turns CuPy's cache
off (``device_memory.disable_fft_plan_cache``) and, before this module,
every transform re-planned on every call: at T799 that was 388
``cuModuleLoadData`` a step and about one second of host time a step
(nsys, opspeed 2026-11-15).

What this does instead.  A plan is built once per (device, length, type,
batch) and kept, but it owns NO work area.  The work area is taken from
the run's pool when the transform is called, bound to the plan
(``cufftSetWorkArea``), and handed back when the call returns -- the same
bytes, for the same span, as a re-planned call took.  So the device
memory a run holds is what it held with the cache off; the only addition
is each kept cuFFT handle's own small device tables, which live outside
the pool and are bounded by ``max_plans`` (receipt ``fft_plans``; the door
charges them through ``sizing.OUT_OF_POOL_MEASURES``).

The pool hands a block back to the stream it was taken on and serves the
next request on that stream after the kernels already queued there, so
releasing the work area as soon as the transform is ENQUEUED is the
ordering every pool allocation in this package already relies on.

The bits do not move: cuFFT planning is deterministic (the same
parameters select the same kernels), the kept plan is the plan the call
would have built, and a work area is scratch whose contents never reach
the output.  The plan is entered through CuPy's documented plan context
(``with plan:``), so CuPy's own check of length, type and batch runs on
every call.  Where ``cufftSetWorkArea`` cannot be reached (no loaded cuFFT
library found), nothing is kept and every call plans as before.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading
from collections import OrderedDict

#: Kept plans per device (least recently used goes first).
DEFAULT_MAX_PLANS = 256

_LOCK = threading.Lock()
_ACTIVE: "FftPlanCache | None" = None


def _loaded_cufft():
    """``cufftSetWorkArea`` from the cuFFT library CuPy already loaded, or
    None.  The SAME library: a second copy would not know CuPy's handles."""
    from cupy.cuda import cufft  # noqa: F401 - loads the library

    lib = None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="utf-8") as maps:
                paths = {line.split()[-1] for line in maps
                         if "libcufft.so" in line and "/" in line}
        except OSError:
            paths = set()
        paths = sorted(p for p in paths if "libcufftw" not in p)
        if len(paths) == 1:
            lib = ctypes.CDLL(paths[0], mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    elif sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        for name in ("cufft64_12.dll", "cufft64_11.dll", "cufft64_13.dll"):
            if kernel32.GetModuleHandleW(name):
                lib = ctypes.WinDLL(name)
                break
    if lib is None:
        return None
    try:
        set_work_area = lib.cufftSetWorkArea
    except AttributeError:
        return None
    set_work_area.restype = ctypes.c_int
    set_work_area.argtypes = [ctypes.c_int, ctypes.c_void_p]
    return set_work_area


class FftPlanCache:
    """Kept cuFFT plans per device; work areas from the pool per call."""

    def __init__(self, cp, set_work_area, *,
                 max_plans: int = DEFAULT_MAX_PLANS) -> None:
        self._cp = cp
        self._set_work_area = set_work_area
        self.max_plans = max(1, int(max_plans))
        # key -> (plan, work_bytes)
        self._plans: dict[int, OrderedDict] = {}
        # A plan is built against this block and re-bound before every
        # call, so it never owns a work area of its own.
        self._parking: dict[int, object] = {}
        self.hits = 0
        self.built = 0
        self.evicted = 0
        self.largest_work_bytes = 0

    @classmethod
    def for_this_process(cls, cp, **kwargs) -> "FftPlanCache | None":
        set_work_area = _loaded_cufft()
        return None if set_work_area is None else cls(cp, set_work_area,
                                                      **kwargs)

    def _build(self, device: int, nx: int, fft_type: int, batch: int):
        cp = self._cp
        from cupy.cuda import cufft

        outer = cp.cuda.get_allocator()
        asked = {"bytes": 0}
        parking = self._parking.get(device)
        if parking is None:
            parking = self._parking[device] = outer(512)

        def allocate(size: int):
            asked["bytes"] = int(size)
            if int(size) == 0:
                return outer(0)
            return cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(parking.ptr, int(size), parking), 0)

        with cp.cuda.using_allocator(allocate):
            plan = cufft.Plan1d(int(nx), int(fft_type), int(batch))
        self.built += 1
        self.largest_work_bytes = max(self.largest_work_bytes, asked["bytes"])
        return plan, asked["bytes"]

    @contextlib.contextmanager
    def _call(self, plan, work_bytes: int):
        cp = self._cp
        work = cp.cuda.alloc(work_bytes) if work_bytes else None
        result = self._set_work_area(
            int(plan.handle), None if work is None else int(work.ptr))
        if result != 0:
            raise RuntimeError(f"cufftSetWorkArea returned {result}")
        try:
            with plan:
                yield plan
        finally:
            del work

    def kept(self, nx: int, fft_type: int, batch: int):
        device = int(self._cp.cuda.runtime.getDevice())
        key = (int(nx), int(fft_type), int(batch))
        with _LOCK:
            plans = self._plans.setdefault(device, OrderedDict())
            entry = plans.get(key)
            if entry is not None:
                plans.move_to_end(key)
                self.hits += 1
            else:
                entry = plans[key] = self._build(device, *key)
                while len(plans) > self.max_plans:
                    plans.popitem(last=False)
                    self.evicted += 1
        return self._call(*entry)

    def clear(self) -> None:
        with _LOCK:
            self._plans.clear()
            self._parking.clear()

    def receipt(self) -> dict[str, object]:
        return {
            "mechanism": ("kept cuFFT plans, work area from the pool per "
                          "call (arwen_global.spectral.fft_plans)"),
            "max_plans": int(self.max_plans),
            "kept_plans": int(sum(len(p) for p in self._plans.values())),
            "plans_built": int(self.built),
            "hits": int(self.hits),
            "evicted": int(self.evicted),
            "largest_work_bytes": int(self.largest_work_bytes),
        }


def activate(cache: FftPlanCache | None) -> FftPlanCache | None:
    """Install ``cache`` as the process's kept plans; returns the previous."""
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, cache
    return previous


def active() -> FftPlanCache | None:
    return _ACTIVE


def _fft_type(dtype, inverse: bool) -> int | None:
    from cupy.cuda import cufft
    import numpy as np

    kind = np.dtype(dtype)
    if inverse:
        return {np.dtype(np.complex64): cufft.CUFFT_C2R,
                np.dtype(np.complex128): cufft.CUFFT_Z2D}.get(kind)
    return {np.dtype(np.float32): cufft.CUFFT_R2C,
            np.dtype(np.float64): cufft.CUFFT_D2Z}.get(kind)


def kept(array, nx: int, *, inverse: bool = False):
    """Context for ONE ``rfft`` (``inverse=False``) or ``irfft`` along the
    last axis of ``array`` with transform length ``nx``: inside it CuPy
    runs the kept plan.  A numpy array, no active cache, an empty batch
    or an unexpected dtype gets a no-op context (CuPy plans as before)."""
    cache = _ACTIVE
    if cache is None or type(array).__module__.split(".")[0] != "cupy":
        return contextlib.nullcontext()
    last = int(array.shape[-1]) if array.ndim else 0
    if last == 0 or array.size == 0:
        return contextlib.nullcontext()
    fft_type = _fft_type(array.dtype, inverse)
    if fft_type is None:
        return contextlib.nullcontext()
    return cache.kept(int(nx), fft_type, int(array.size // last))


__all__ = ["DEFAULT_MAX_PLANS", "FftPlanCache", "activate", "active", "kept"]
