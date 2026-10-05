"""The render-tape and regional-translation kernels, in Rust.

WOOF Global writes its weather-field products through wrfout render tapes
and hands its parent exports to the regional model through translated
frames.  Both steps are data-path processing: the Gaussian grid is sampled
onto the tape's regular latitude-longitude grid, the tape's hydrostatic
column is integrated, and a parent export is interpolated horizontally and
in ln p onto a regional target.  Under the Python boundary those belong in
Rust, and this module is the seam to the library that does them,
``global_render_kernels`` (crate ``rust/global-render-kernels`` in this
repository, published in this package's own door bundle).

The NumPy versions these kernels replace survive only as the test oracle
(``tests/render_kernel_oracle.py``), and ``tests/test_render_kernels.py``
holds the library to them byte for byte.  The library evaluates the same
IEEE operations in the same order as the oracle and takes ``exp``, ``log``
and ``pow`` from the C library one element at a time.  That is the
oracle's own arithmetic wherever NumPy also takes them from the C library
(every x86-64 machine without AVX-512); on an AVX-512 machine NumPy's
vector ``exp``, ``log`` and ``pow`` round the last bit differently, and the
library is then the reproducible one of the two, as the hybrid level
weights already are.

THERE IS NO NUMPY FALLBACK.  A machine without the library refuses with
the missing-door exit (3) and the command that stages it.  A silent
fallback would put the Python data path back on exactly the machines
nobody is watching, which is the breakage the boundary exists to prevent.

Arrays cross as C-contiguous float64, positional C signatures, guarded by
an ABI version probe and the contract literal the door table pins.
"""
from __future__ import annotations

import ctypes
import threading

import numpy as np

__all__ = [
    "ABI_VERSION",
    "CONTRACT",
    "DOOR",
    "GaussianRegridder",
    "hybrid_pressure",
    "hydrostatic",
    "library",
    "log_pressure_interpolate",
    "periodic_bilinear",
    "standard_lapse_theta_below",
    "virtual_temperature",
]

#: The door table's name for the library.
DOOR = "global_render_kernels"
#: The seam's ABI version; the library answers it from `grk_abi_version`.
ABI_VERSION = 1
#: The contract literal the library carries and the door table pins.
CONTRACT = b"woof-global-render-kernels-v1"

#: Post-operation codes of `grk_gaussian_to_regular`.
_POST = {"none": 0, "floor_zero": 1, "exp": 2}

_F64P = ctypes.POINTER(ctypes.c_double)
_SIZE = ctypes.c_size_t
_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None


class RenderKernelError(RuntimeError):
    """The library refused a call; the message is its own."""


def _declare(lib: ctypes.CDLL) -> None:
    f64 = ctypes.c_double
    lib.grk_abi_version.restype = ctypes.c_uint32
    lib.grk_abi_version.argtypes = []
    lib.grk_contract.restype = ctypes.c_char_p
    lib.grk_contract.argtypes = []
    lib.grk_last_error.restype = _SIZE
    lib.grk_last_error.argtypes = [ctypes.c_char_p, _SIZE]
    signatures = {
        "grk_gaussian_to_regular": [
            _F64P, _SIZE, f64, _SIZE, _F64P, _SIZE, _F64P, _SIZE, _F64P,
            _SIZE, ctypes.c_uint32, _F64P],
        "grk_periodic_bilinear": [
            _F64P, _SIZE, f64, f64, _SIZE, _F64P, _SIZE, _F64P, _F64P,
            _SIZE, _F64P],
        "grk_log_pressure_interpolate": [
            _F64P, _F64P, _SIZE, _F64P, _SIZE, _SIZE, _F64P, _F64P],
        "grk_standard_lapse_theta_below": [
            _F64P, _F64P, _SIZE, _F64P, _SIZE, _SIZE, f64, f64, f64, f64,
            f64, _F64P],
        "grk_hybrid_pressure": [
            _F64P, _F64P, _SIZE, _F64P, _SIZE, _F64P, _F64P],
        "grk_virtual_temperature": [
            _F64P, _F64P, _F64P, _F64P, _F64P, _F64P, _F64P, _F64P, _SIZE,
            f64, f64, _F64P, _F64P],
        "grk_hydrostatic": [
            _F64P, _F64P, _SIZE, _F64P, _SIZE, f64, _F64P, _F64P],
    }
    for name, argtypes in signatures.items():
        function = getattr(lib, name)
        function.restype = ctypes.c_int32
        function.argtypes = argtypes


def library() -> ctypes.CDLL:
    """The loaded library, or the missing-door refusal.

    Loaded once per process.  A library that answers another ABI version or
    another contract is refused by name before any kernel is called, because
    a positional C signature of the wrong shape reads the wrong memory
    rather than failing.
    """

    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        from .doors import find_door, missing_door_refusal, search_path

        path = find_door(DOOR)
        if path is None:
            raise missing_door_refusal(
                DOOR, "searched: " + ", ".join(str(p) for p in search_path(DOOR)))
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise missing_door_refusal(DOOR, f"{path} does not load: {exc}") from exc
        _declare(lib)
        version = int(lib.grk_abi_version())
        if version != ABI_VERSION:
            raise missing_door_refusal(
                DOOR, f"{path} answers ABI version {version}; this package "
                f"calls version {ABI_VERSION}, and a positional signature of "
                "another shape reads the wrong memory")
        contract = lib.grk_contract() or b""
        if contract != CONTRACT:
            raise missing_door_refusal(
                DOOR, f"{path} carries the contract {contract!r}, not "
                f"{CONTRACT!r}")
        _LIBRARY = lib
        return lib


def _check(lib: ctypes.CDLL, code: int, what: str) -> None:
    if code == 0:
        return
    size = int(lib.grk_last_error(None, 0))
    buffer = ctypes.create_string_buffer(max(size, 1))
    lib.grk_last_error(buffer, size)
    raise RenderKernelError(f"{what}: {buffer.raw[:size].decode('utf-8', 'replace')}")


def _f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def _ptr(array: np.ndarray | None):
    if array is None:
        return None
    return array.ctypes.data_as(_F64P)


class GaussianRegridder:
    """Bilinear sampling of the Gaussian grid at regular-grid centres.

    ``target_latitude_deg`` names the output rows and
    ``target_longitude_deg`` the output columns, in output order: each
    output value depends only on its own row's latitude and its own
    column's longitude, so a rolled or windowed target is sampled directly
    rather than sampled whole and then cut.
    """

    def __init__(self, source_latitude_deg, source_longitude_deg,
                 target_latitude_deg, target_longitude_deg):
        self._src_lat = _f64(source_latitude_deg).reshape(-1)
        lon = np.asarray(source_longitude_deg, dtype=np.float64).reshape(-1)
        if lon.size < 1:
            raise ValueError("the Gaussian grid has no longitudes")
        self._lon0 = float(lon[0])
        self._nsx = int(lon.size)
        self._tgt_lat = _f64(target_latitude_deg).reshape(-1)
        self._tgt_lon = _f64(target_longitude_deg).reshape(-1)
        self.shape = (self._tgt_lat.size, self._tgt_lon.size)

    def __call__(self, values, post: str = "none") -> np.ndarray:
        field = _f64(values)
        nsy = self._src_lat.size
        if field.ndim < 2 or field.shape[-2:] != (nsy, self._nsx):
            raise ValueError(
                f"field shape {field.shape} does not end in the Gaussian grid "
                f"{nsy}x{self._nsx}")
        lead = field.shape[:-2]
        nlev = int(np.prod(lead, dtype=np.int64)) if lead else 1
        out = np.empty((*lead, *self.shape), dtype=np.float64)
        lib = library()
        _check(lib, lib.grk_gaussian_to_regular(
            _ptr(self._src_lat), nsy, self._lon0, self._nsx,
            _ptr(self._tgt_lat), self.shape[0], _ptr(self._tgt_lon), self.shape[1],
            _ptr(field), nlev, _POST[post], _ptr(out)), "Gaussian regrid")
        return out


def periodic_bilinear(source_latitude_deg, longitude0_deg: float,
                      spacing_deg: float, nlon: int, values,
                      target_latitude_deg, target_longitude_deg) -> np.ndarray:
    """Bilinear sampling of a regular periodic-longitude parent grid.

    The caller has refused every malformed grid and every target outside
    the parent's latitude span; the library repeats none of that.
    """

    lat = _f64(source_latitude_deg).reshape(-1)
    field = _f64(values)
    lead = field.shape[:-2]
    nlev = int(np.prod(lead, dtype=np.int64)) if lead else 1
    target_lat = _f64(target_latitude_deg)
    target_lon = _f64(target_longitude_deg)
    npts = target_lat.size
    out = np.empty((*lead, *target_lat.shape), dtype=np.float64)
    lib = library()
    _check(lib, lib.grk_periodic_bilinear(
        _ptr(lat), lat.size, float(longitude0_deg), float(spacing_deg), int(nlon),
        _ptr(field), nlev, _ptr(target_lat), _ptr(target_lon), npts, _ptr(out)),
        "periodic bilinear")
    return out


def log_pressure_interpolate(source_pressure, source_values, target_pressure,
                             bottom_values=None) -> np.ndarray:
    """Independent columns, linear in ln p, held outside the parent's span,
    with ``bottom_values`` used wherever a target lies below the parent's
    bottom level."""

    src_p = _f64(source_pressure)
    src_v = _f64(source_values)
    tgt_p = _f64(target_pressure)
    nsrc = src_p.shape[0]
    ntgt = tgt_p.shape[0]
    ncol = src_p.size // max(nsrc, 1)
    bottom = None if bottom_values is None else _f64(bottom_values)
    out = np.empty(tgt_p.shape, dtype=np.float64)
    lib = library()
    _check(lib, lib.grk_log_pressure_interpolate(
        _ptr(src_p), _ptr(src_v), nsrc, _ptr(tgt_p), ntgt, ncol, _ptr(bottom),
        _ptr(out)), "log-pressure interpolation")
    return out


def standard_lapse_theta_below(source_pressure, source_theta, target_pressure, *,
                               reference_pressure_pa: float, kappa: float,
                               gas_constant: float, gravity: float,
                               lapse_rate: float) -> np.ndarray:
    src_p = _f64(source_pressure)
    src_theta = _f64(source_theta)
    tgt_p = _f64(target_pressure)
    nsrc = src_p.shape[0]
    ntgt = tgt_p.shape[0]
    ncol = src_p.size // max(nsrc, 1)
    out = np.empty(tgt_p.shape, dtype=np.float64)
    lib = library()
    _check(lib, lib.grk_standard_lapse_theta_below(
        _ptr(src_p), _ptr(src_theta), nsrc, _ptr(tgt_p), ntgt, ncol,
        float(reference_pressure_pa), float(kappa), float(gas_constant),
        float(gravity), float(lapse_rate), _ptr(out)), "standard-lapse theta")
    return out


def hybrid_pressure(a_half, b_half, surface_pressure) -> tuple[np.ndarray, np.ndarray]:
    """``(p_half, p_full)``: ``a + b ps`` and the geometric mean of each layer."""

    a = _f64(a_half).reshape(-1)
    b = _f64(b_half).reshape(-1)
    ps = _f64(surface_pressure)
    nhalf = a.size
    p_half = np.empty((nhalf, *ps.shape), dtype=np.float64)
    p_full = np.empty((max(nhalf - 1, 0), *ps.shape), dtype=np.float64)
    lib = library()
    _check(lib, lib.grk_hybrid_pressure(
        _ptr(a), _ptr(b), nhalf, _ptr(ps), ps.size, _ptr(p_half), _ptr(p_full)),
        "hybrid pressure")
    return p_half, p_full


def virtual_temperature(theta, species, p_full, *, reference_pressure_pa: float,
                        kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """``(T, Tv)`` with ``Tv = T (1 + 0.61 qv - (qc + qr + qi + qs + qg))``;
    ``species`` maps the six water names to arrays of ``theta``'s shape."""

    th = _f64(theta)
    arrays = [_f64(species[name]) for name in ("qv", "qc", "qr", "qi", "qs", "qg")]
    p = _f64(p_full)
    for array in (*arrays, p):
        if array.shape != th.shape:
            raise ValueError("virtual temperature inputs differ in shape")
    temperature = np.empty(th.shape, dtype=np.float64)
    virtual = np.empty(th.shape, dtype=np.float64)
    lib = library()
    _check(lib, lib.grk_virtual_temperature(
        _ptr(th), *(_ptr(a) for a in arrays), _ptr(p), th.size,
        float(reference_pressure_pa), float(kappa), _ptr(temperature),
        _ptr(virtual)), "virtual temperature")
    return temperature, virtual


def hydrostatic(virtual_temperature_k, p_half, surface_geopotential, *,
                gas_constant: float, full_levels: bool = False):
    """Half-level geopotential integrated up from the surface, and the
    full-level geopotential too when ``full_levels`` is true."""

    tv = _f64(virtual_temperature_k)
    ph = _f64(p_half)
    phis = _f64(surface_geopotential)
    nz = tv.shape[0]
    if ph.shape != (nz + 1, *tv.shape[1:]) or phis.shape != tv.shape[1:]:
        raise ValueError("hydrostatic inputs differ in shape")
    half = np.empty(ph.shape, dtype=np.float64)
    full = np.empty(tv.shape, dtype=np.float64) if full_levels else None
    lib = library()
    _check(lib, lib.grk_hydrostatic(
        _ptr(tv), _ptr(ph), nz, _ptr(phis), phis.size, float(gas_constant),
        _ptr(half), _ptr(full)), "hydrostatic integration")
    return (half, full) if full_levels else half
