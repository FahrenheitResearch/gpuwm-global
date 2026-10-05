"""ctypes seam onto the Rust cold-start door (``rw_global_coldstart``).

The library is ``rust/rw-global-coldstart`` in this repository.  It carries
the two data-path operations of a cold start that used to run in host NumPy
on every analysis-initialised run:

* the periodic bilinear regrid from the analysis's regular latitude-longitude
  grid onto the spectral transform's Gaussian grid
  (:func:`regrid`, behind ``analysis_initial._global_regridder``), and
* the linear-in-ln(p) remap from isobaric levels onto model full levels
  (:func:`remap`, behind ``analysis_initial._to_model_levels``).

Both are byte-identical to the NumPy expressions they replace; the NumPy
forms survive only as the test oracle (``tests/coldstart_oracle.py``), so
the default path has no Python regrid at all.

THERE IS NO NUMPY FALLBACK, and the refusal names its breakage.  A missing
or stale library raises :class:`arwen_global.doors.DoorMissing` (exit 3,
"stage the bundle") rather than quietly regridding in NumPy: a cold start
that silently leaves the Rust data path is the Python-boundary defect this
door exists to close, and a run that did it could not say so in a receipt
anyone reads.

Why ctypes and not an extension module: the engine's library doors
(obs-regrid, static-fields, netcdf-writer) are all cdylibs behind ctypes,
one loading discipline, one staging path, one ABI-marker rule, and no build
per interpreter.  The wheel stays pure Python.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading

import numpy as np

__all__ = [
    "ABI_MARKER",
    "COLDSTART_ABI",
    "COLDSTART_DOOR",
    "ColdStartBridgeError",
    "library_candidates",
    "load",
    "regrid",
    "remap",
    "resolved_path",
]

#: The door row in :mod:`arwen_global.doors`.
COLDSTART_DOOR = "rw_global_coldstart"

#: The C ABI version this module speaks; a library answering anything else
#: is refused rather than called with signatures it does not have.
COLDSTART_ABI = 1

#: The contract literal (the door row's marker), compiled into the library
#: and returned by its ``arwen_global_coldstart_contract`` export.
ABI_MARKER = b"arwen-global-coldstart-abi-v1"

_F64P = ctypes.POINTER(ctypes.c_double)


class ColdStartBridgeError(RuntimeError):
    """The Rust cold-start door refused, in its own words."""


def library_candidates() -> tuple[Path, ...]:
    """Where the library is looked for, best first: the door ladder
    (:func:`arwen_global.doors.search_path`), whose last rung for a door
    built from this repository is the checkout's own ``rust/target/release``.
    """

    from .doors import search_path

    return tuple(search_path(COLDSTART_DOOR))


_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_PATH: Path | None = None


def _refusal(detail: str):
    from .doors import missing_door_refusal

    return missing_door_refusal(COLDSTART_DOOR, detail)


def load() -> ctypes.CDLL:
    """The loaded library, or the door refusal naming every rung searched."""

    global _LIBRARY, _PATH
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        from .doors import door_by_name

        override = os.environ.get(door_by_name(COLDSTART_DOOR).env_var)
        candidates = library_candidates()
        if override and not Path(override).is_file():
            raise _refusal(
                f"{door_by_name(COLDSTART_DOOR).env_var} names {override}, "
                "which is not a file")
        chosen = next((c for c in candidates if c.is_file()), None)
        if chosen is None:
            searched = "\n  ".join(str(c) for c in candidates)
            raise _refusal(
                "the cold start regrids the analysis onto the Gaussian grid "
                "and remaps it onto model levels through this library, and "
                "without it no analysis-initialised run can start; "
                f"searched:\n  {searched}")
        try:
            library = ctypes.CDLL(str(chosen))
        except OSError as error:
            raise _refusal(f"{chosen} would not load: {error}") from error
        try:
            version = library.arwen_global_coldstart_abi_version
        except AttributeError:
            raise _refusal(
                f"{chosen} exports no ABI probe; it is not the cold-start "
                "door") from None
        version.restype = ctypes.c_uint32
        version.argtypes = []
        answered = int(version())
        contract = getattr(library, "arwen_global_coldstart_contract", None)
        literal = b""
        if contract is not None:
            contract.restype = ctypes.c_char_p
            contract.argtypes = []
            literal = contract() or b""
        if answered != COLDSTART_ABI or literal != ABI_MARKER:
            raise _refusal(
                f"{chosen} speaks cold-start ABI {answered}, and this package "
                f"calls ABI {COLDSTART_ABI}; a stale build would be handed "
                "arguments in an order it does not read")
        library.arwen_global_coldstart_last_error.restype = ctypes.c_size_t
        library.arwen_global_coldstart_last_error.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t]
        library.arwen_global_coldstart_regrid.restype = ctypes.c_int32
        library.arwen_global_coldstart_regrid.argtypes = [
            _F64P, ctypes.c_size_t, _F64P, ctypes.c_size_t,
            _F64P, ctypes.c_size_t, _F64P, ctypes.c_size_t,
            _F64P, ctypes.c_size_t, _F64P, ctypes.c_size_t,
        ]
        library.arwen_global_coldstart_remap.restype = ctypes.c_int32
        library.arwen_global_coldstart_remap.argtypes = [
            _F64P, _F64P, ctypes.c_size_t, ctypes.c_size_t,
            _F64P, ctypes.c_size_t, ctypes.c_uint8, _F64P, ctypes.c_size_t,
        ]
        _LIBRARY = library
        _PATH = chosen
        return library


def resolved_path() -> Path:
    """The file the loaded library came from (loading it if needed)."""

    load()
    assert _PATH is not None
    return _PATH


def _last_error(library: ctypes.CDLL) -> str:
    size = int(library.arwen_global_coldstart_last_error(None, 0))
    buffer = ctypes.create_string_buffer(max(size, 1))
    library.arwen_global_coldstart_last_error(buffer, size)
    return buffer.raw[:size].decode("utf-8", errors="replace")


def _f64(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def _ptr(array: np.ndarray):
    return array.ctypes.data_as(_F64P)


def regrid(values, latitude, longitude, target_latitude,
           target_longitude, *, threads: int = 0) -> np.ndarray:
    """Bilinear regrid of ``values`` (``(..., nlat, nlon)``) onto the target
    grid, as ``(..., ntlat, ntlon)`` float64.

    The caller has already refused a grid that is not regular, not global
    or does not reach the poles (``analysis_initial._global_regridder``);
    the library refuses only what it cannot index.
    """

    library = load()
    lat = _f64(latitude).ravel()
    lon = _f64(longitude).ravel()
    tlat = _f64(target_latitude).ravel()
    tlon = _f64(target_longitude).ravel()
    field = _f64(values)
    if field.ndim < 2 or field.shape[-2:] != (lat.size, lon.size):
        raise ValueError(
            f"field shape {field.shape} does not end in the analysis grid "
            f"({lat.size}, {lon.size})")
    lead = field.shape[:-2]
    planes = int(np.prod(lead, dtype=np.int64)) if lead else 1
    out = np.empty(lead + (tlat.size, tlon.size), dtype=np.float64)
    code = library.arwen_global_coldstart_regrid(
        _ptr(lat), lat.size, _ptr(lon), lon.size,
        _ptr(tlat), tlat.size, _ptr(tlon), tlon.size,
        _ptr(field), planes, _ptr(out), int(threads))
    if code != 0:
        raise ColdStartBridgeError(
            f"the cold-start regrid refused: {_last_error(library)}")
    return out


def remap(values, ln_source, ln_target, *, extrapolate_below: bool = False,
          threads: int = 0) -> np.ndarray:
    """Linear-in-ln(p) remap of ``values`` (``(nsrc, ...)``) from ascending
    ``ln_source`` onto ``ln_target`` (``(ntgt, ...)``), as float64.

    The trailing shapes broadcast against each other the way
    ``numpy.take_along_axis`` broadcasts them, and the two arrays must have
    the same number of dimensions, as there.
    """

    library = load()
    source = _f64(ln_source)
    if source.ndim != 1:
        raise ValueError("ln_source must be one-dimensional")
    field = np.asarray(values)
    target = np.asarray(ln_target)
    if field.ndim != target.ndim:
        raise ValueError(
            "`values` and the target levels must have the same number of "
            f"dimensions ({field.ndim} and {target.ndim})")
    if field.shape[0] != source.size:
        raise ValueError(
            f"`values` carries {field.shape[0]} levels and ln_source "
            f"{source.size}")
    trailing = np.broadcast_shapes(field.shape[1:], target.shape[1:])
    field = _f64(np.broadcast_to(field, field.shape[:1] + trailing))
    target = _f64(np.broadcast_to(target, target.shape[:1] + trailing))
    points = int(np.prod(trailing, dtype=np.int64)) if trailing else 1
    out = np.empty(target.shape, dtype=np.float64)
    code = library.arwen_global_coldstart_remap(
        _ptr(field), _ptr(source), source.size, points,
        _ptr(target), target.shape[0], 1 if extrapolate_below else 0,
        _ptr(out), int(threads))
    if code != 0:
        raise ColdStartBridgeError(
            f"the cold-start remap refused: {_last_error(library)}")
    return out
