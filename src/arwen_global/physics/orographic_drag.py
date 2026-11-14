"""Orographic gravity-wave drag for the native suite (WRF ``gwd_opt = 1``).

The scheme is the KIM orographic gravity-wave drag with flow blocking (Choi
and Hong 2015; WRF ``phys/module_bl_gwdo.F`` calling
``phys/physics_mmm/bl_gwdo.F90``), carried from the engine as
:mod:`arwen_global.core.terrain_drag` and its kernel.  This module is the
seam between it and the native suite:

* :class:`OrographicStatistics` holds the ten sub-grid orography planes the
  scheme reads (:mod:`arwen_global.orographic_statics`) for the whole
  Gaussian grid, with the cache's identity, and hands each latitude band its
  rows on the band's device;
* :func:`apply_orographic_drag` adds the drag to YSU's momentum tendencies
  for one batch, where and when WRF adds it: after the PBL scheme, from the
  same column state the PBL scheme read, at the PBL cadence
  (``module_pbl_driver.F:2129-2185``);
* :func:`reference_launch_gwdo` is a float32 numpy transcription of the
  carried kernel, with the device launcher's signature, so the scheme can be
  checked against WRF's own words on a machine with no card
  (``tests/test_arwen_global_orographic_drag.py``).  The native suite runs
  the device kernel; nothing in a forecast calls the transcription.

WHAT IS PRESCRIBED HERE, and said so because none of it is measured:

* the grid length the scheme is handed (``dx`` in WRF: its effective
  length ``cleff`` and the blocking area's ``dx**2``) is ONE value for the
  whole globe, the options' ``dx_m``, which the shipped configurations set to
  the truncation's equatorial column spacing (52.1 km at T255).  A spectral
  model resolves the same scales everywhere on the sphere, and the stress
  the scheme returns is per unit area, so the column's own zonal width,
  which shrinks toward the poles, is not what it should see;
* the column heights are above the surface, formed from the hydrostatic
  layer depths the suite already uses (``z = 0.5 (z_w(k) + z_w(k+1))``, as
  WRF's ``phy_prep`` forms them); the scheme reads only height differences;
* the wind is earth-relative on the Gaussian grid, so the map rotation is
  zero (``sina = 0``, ``cosa = 1``);
* the device words are WRF's under the engine's strict numerics
  (``GPUWM_NUMERICS=strict``); in its default fast numerics the compile
  boundary drops the kernel's ``-fmad=false`` and multiply-adds are fused,
  the arithmetic the other carried kernels compile in (the carried loader
  asks for no ``-fmad=false``), so a branch of the scheme can flip on a
  rounding (tests/test_arwen_global_orographic_drag.py has the measurement);
* the scheme adds no heat: WRF's ``gwd_opt = 1`` returns no potential
  temperature tendency, so the kinetic energy it removes is not returned as
  dissipation heating (the energy advisory, which prices enthalpy, does not
  see it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

#: The option values: off, and WRF's gwd_opt = 1.
OROGRAPHIC_DRAG_SCHEMES = ("none", "kim-gwdo")
#: The planes the scheme reads, in the launcher's order.
from ..orographic_statics import SSO_FIELDS  # noqa: E402

#: physics_state.metadata key the cache identity travels under, so a restart
#: that would read different statistics is refused rather than continued.
METADATA_KEY = "orographic_drag_statistics"

# WRF's module_model_constants values the PBL driver hands the scheme, as the
# carried launcher spells them (float32 throughout).
_G = np.float32(9.81)
_R_D = np.float32(287.0)
_R_V = np.float32(461.6)
_CP = np.float32(np.float32(7.0) * _R_D / np.float32(2.0))
_EP_1 = np.float32(_R_V / _R_D - np.float32(1.0))
_PI = np.float32(3.141592653)
_F = np.float32


@dataclass
class OrographicStatistics:
    """The whole grid's sub-grid orography, on the host, and its identity."""

    planes: dict[str, np.ndarray]
    identity: dict[str, object]
    _bands: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_cache(cls, fields: Mapping[str, np.ndarray], document: dict):
        planes = {name: np.ascontiguousarray(fields[name], dtype=np.float32)
                  for name in SSO_FIELDS}
        identity = {
            "kind": document.get("kind"),
            "dataset": document.get("dataset"),
            "self_sha256": document.get("self_sha256"),
        }
        return cls(planes=planes, identity=identity)

    def band(self, band, xp, shape):
        """This band's rows of every plane, as ``xp`` float32 arrays."""
        key = (None if band is None else (int(band[0]), int(band[1])),
               getattr(xp, "__name__", str(xp)))
        cached = self._bands.get(key)
        if cached is None:
            rows = slice(None) if band is None else slice(int(band[0]), int(band[1]))
            cached = {name: xp.ascontiguousarray(xp.asarray(plane[rows],
                                                            dtype=xp.float32))
                      for name, plane in self.planes.items()}
            got = tuple(cached["VAR"].shape)
            if got != tuple(shape):
                raise ValueError(
                    f"the orographic statistics' rows {band} are {got}, the "
                    f"batch's columns are {tuple(shape)}: the cache was built "
                    "for another grid")
            self._bands[key] = cached
        return cached


def column_atmosphere(batch, dz) -> dict[str, object]:
    """The column state the scheme reads, from the batch before YSU applies.

    Bottom-first ``(nz, ny, nx)`` like every native-suite array: winds,
    temperature (theta times Exner), vapour mixing ratio, full and half-level
    hydrostatic pressures, Exner, the layer depths and interface heights
    above the surface.
    """
    xp = batch.xp
    arrays = batch.arrays
    depth = xp.asarray(dz, dtype=xp.float64)
    z_interface = xp.concatenate(
        [xp.zeros((1,) + tuple(depth.shape[1:]), dtype=xp.float64),
         xp.cumsum(depth, axis=0)], axis=0).astype(xp.float32)
    return {
        "u": arrays["u"], "v": arrays["v"],
        "temperature": xp.ascontiguousarray(arrays["theta"] * arrays["exner"]),
        "qv": arrays["qv"], "pressure": arrays["p_full"],
        "exner": arrays["exner"], "p_interface": arrays["p_half"],
        "z_interface": xp.ascontiguousarray(z_interface),
        "dz": dz,
    }


def stamp_identity(metadata: dict, statistics: OrographicStatistics) -> None:
    """Record the statistics in the namespace, or refuse a different set."""
    stated = metadata.get(METADATA_KEY)
    if stated is None:
        metadata[METADATA_KEY] = dict(statistics.identity)
        return
    if dict(stated) != dict(statistics.identity):
        raise ValueError(
            "the orographic drag's sub-grid statistics are not the ones this "
            f"run started with: the state carries {stated}, the cache now "
            f"reads {statistics.identity}.  A restart that read other "
            "statistics would not reproduce the straight run; rebuild the "
            "cache the run started from, or start the run again")


def apply_orographic_drag(statistics: OrographicStatistics | None, batch,
                          pbl_out, *, launch, dz, dx_m: float, metadata: dict):
    """Add the KIM drag to YSU's ``du``/``dv`` (in place) for one batch."""
    if statistics is None:
        raise ValueError(
            "orographic_drag = \"kim-gwdo\" and the native suite was handed no "
            "sub-grid orography statistics; the run door loads them from the "
            "statics cache (gpuwm-global statics CONFIG --orographic), and a "
            "suite built another way must be handed them before its first "
            "step")
    stamp_identity(metadata, statistics)
    xp = batch.xp
    du, dv = pbl_out["du"], pbl_out["dv"]
    for name, rate in (("du", du), ("dv", dv)):
        # The kernel adds into the rate in place, so it must be the array
        # the PBL step applies, laid out as the kernel indexes it.
        if rate.dtype != xp.float32 or not rate.flags.c_contiguous:
            raise TypeError(f"YSU's {name} must be C-contiguous float32 for "
                            "the orographic drag to add into it in place")
    shape = tuple(du.shape[1:])
    planes = statistics.band(batch.band, xp, shape)
    zero = xp.zeros(shape, dtype=xp.float32)
    one = xp.ones(shape, dtype=xp.float32)
    launch(column_atmosphere(batch, dz), du, dv,
           var=planes["VAR"], con=planes["CON"],
           oa=[planes[f"OA{m}"] for m in range(1, 5)],
           ol=[planes[f"OL{m}"] for m in range(1, 5)],
           sina=zero, cosa=one, dx=float(dx_m), dt=float(batch.dt_s))


def config_turns_the_drag_on(cfg) -> bool:
    """Whether ``cfg`` runs the native suite with the orographic drag on,
    read off the config's own options (no suite is built)."""
    if getattr(cfg, "physics_mode", None) != "arwen-native":
        return False
    options = getattr(cfg, "native_adapter_options", None) or {}
    return str(options.get("orographic_drag", "none")) != "none"


def prepare_orographic_statistics(cfg, grid=None, *, say=None):
    """Build the drag's sub-grid orography cache for ``cfg`` if it is missing.

    The run door calls this before it sizes the card, so a machine whose
    land-surface statics predate the drag (or a runner that skips
    ``statics`` once those exist) gets the cache built, and its WPS_GEOG
    dataset staged, rather than a refusal on its first run
    (:func:`arwen_global.orographic_statics.ensure_orographic_statistics`).
    A no-op for a config without the drag, a synthetic planet, or a world
    the drag is refused under (:func:`attach_orographic_statistics` says
    why).  Returns the cache written, or None.
    """
    if not config_turns_the_drag_on(cfg):
        return None
    if getattr(getattr(cfg, "statics", None), "source", None) != "real":
        return None
    world = getattr(cfg, "world", None)
    if world is not None and not world.is_earth and (
            world.changes_terrain or world.changes_coast or world.is_invented):
        return None
    from ..orographic_statics import ensure_orographic_statistics

    if grid is None:
        from ..spectral.grid import GaussianGrid

        grid = GaussianGrid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                                   dealias_factor=cfg.dealias_factor)
    if say is None:
        return ensure_orographic_statistics(cfg.statics, grid)
    return ensure_orographic_statistics(cfg.statics, grid, say=say)


def attach_orographic_statistics(adapter, cfg, grid=None):
    """Hand a native suite built from ``cfg`` its sub-grid orography.

    A no-op unless the suite's options turn the drag on.  With it on, the
    run needs real statics (the water mask and the grid come from them) and
    the cache ``gpuwm-global statics CONFIG --orographic`` writes.  A
    missing drag cache is built here (from the land cache's own WPS_GEOG
    tree, staging ``orogwd_10m`` when that tree lacks it) if the run door
    has not already built it; what cannot be built, or missing real
    statics, is refused here, before the first step.  Returns the
    statistics attached, or None.
    """
    options = getattr(adapter, "options", None)
    if options is None or not getattr(options, "orographic_drag_enabled", False):
        return None
    world = getattr(cfg, "world", None)
    if world is not None and not world.is_earth and (
            world.changes_terrain or world.changes_coast):
        # The statistics cache is built from Earth's 30-second terrain and
        # read past the world's statics seam (arwen_global.world): under a
        # world whose mountains or coast are its own, the drag would see
        # Earth's.
        raise ValueError(
            f"orographic_drag = \"kim-gwdo\" is refused under world "
            f"{world.label!r}, which changes its terrain or its coast: the "
            "sub-grid orography statistics are built from Earth's terrain "
            "and no world has its own; turn the drag off for this world")
    if getattr(cfg.statics, "source", None) != "real":
        raise ValueError(
            "orographic_drag = \"kim-gwdo\" needs [statics] source = \"real\": "
            "the sub-grid orography is built on the real planet's land mask, "
            "and a synthetic planet has neither terrain statistics nor that "
            "mask")
    from ..orographic_statics import (ensure_orographic_statistics,
                                      load_orographic_statistics,
                                      orographic_statistics_summary)

    if grid is None:
        from ..spectral.grid import GaussianGrid

        grid = GaussianGrid.create(cfg.truncation, nlat=cfg.nlat, nlon=cfg.nlon,
                                   dealias_factor=cfg.dealias_factor)
    ensure_orographic_statistics(cfg.statics, grid)
    fields, document = load_orographic_statistics(cfg.statics, grid)
    statistics = OrographicStatistics.from_cache(fields, document)
    summary = orographic_statistics_summary(fields, document)
    print(f"gpuwm-global: orographic drag kim-gwdo (WRF gwd_opt = 1) on "
          f"{summary['columns_with_statistics']} land columns, VAR up to "
          f"{summary['var_max_m']:.0f} m, dx {float(options.dx_m):.0f} m "
          f"(sub-grid statistics {document.get('dataset')}, cache "
          f"{str(document.get('self_sha256'))[:12]})", flush=True)
    adapter.orographic_statistics = statistics
    return statistics


# ---------------------------------------------------------------------------
# The float32 numpy transcription of kernels/terrain_drag.cu gwdo_column
# ---------------------------------------------------------------------------

_ATANF_TAB = np.array([
    0x3EED6338, 0x3F490FDA, 0x3F7B985E, 0x3FC90FDA,
    0x31AC3769, 0x33222168, 0x33140FB4, 0x33A22168,
    0x3EAAAAAB, 0xBE4CCCCD, 0x3E124925, 0xBDE38E38,
    0x3DBA2E6E, 0xBD9D8795, 0x3D886B35, 0xBD6EF16B,
    0x3D4BDA59, 0xBD15A221, 0x3C8569D7,
], dtype=np.uint32).view(np.float32)


def _u2f(word: int) -> np.float32:
    return np.array([word], dtype=np.uint32).view(np.float32)[0]


def _atanf(x: np.ndarray) -> np.ndarray:
    """td_atanf, elementwise, for finite float32 ``x``."""
    x = np.asarray(x, dtype=np.float32)
    hx = x.view(np.uint32)
    ix = hx & np.uint32(0x7FFFFFFF)
    negative = (hx >> np.uint32(31)) == 1
    ax = np.abs(x)
    one, two, half3 = _F(1.0), _F(2.0), _F(1.5)
    with np.errstate(all="ignore"):
        r0 = (two * ax - one) / (two + ax)
        r1 = (ax - one) / (ax + one)
        r2 = (ax - half3) / (one + half3 * ax)
        r3 = _F(-1.0) / ax
    small = ix < np.uint32(0x3EE00000)
    ident = np.full(x.shape, -1, dtype=np.int64)
    ident = np.where(small, -1,
                     np.where(ix < np.uint32(0x3F300000), 0,
                              np.where(ix < np.uint32(0x3F980000), 1,
                                       np.where(ix < np.uint32(0x401C0000), 2, 3))))
    xr = np.where(small, x, np.where(ident == 0, r0,
                                     np.where(ident == 1, r1,
                                              np.where(ident == 2, r2, r3))))
    xr = xr.astype(np.float32)
    t = _ATANF_TAB[8:]
    z = xr * xr
    w = z * z
    s1 = w * t[10]
    s1 = w * (t[8] + s1)
    s1 = w * (t[6] + s1)
    s1 = w * (t[4] + s1)
    s1 = w * (t[2] + s1)
    s1 = z * (t[0] + s1)
    s2 = w * t[9]
    s2 = w * (t[7] + s2)
    s2 = w * (t[5] + s2)
    s2 = w * (t[3] + s2)
    s2 = w * (t[1] + s2)
    s = s1 + s2
    plain = xr - xr * s
    safe = np.clip(ident, 0, 3)
    hi = _ATANF_TAB[safe]
    lo = _ATANF_TAB[4 + safe]
    r = hi - ((xr * s - lo) - xr)
    r = np.where(negative, -r, r)
    out = np.where(ident < 0, plain, r)
    tiny = ix < np.uint32(0x31000000)
    out = np.where(tiny, x, out)
    big = _ATANF_TAB[3] + _ATANF_TAB[7]
    out = np.where(ix >= np.uint32(0x4C000000),
                   np.where(negative, -big, big), out)
    return out.astype(np.float32)


def _atan2f(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """td_atan2f, elementwise, for the finite winds the scheme hands it."""
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    tiny = _F(1.0e-30)
    pi_o_2 = _u2f(0x3FC90FDB)
    pi = _u2f(0x40490FDB)
    pi_lo = _u2f(0xB3BBBD2E)
    hx = x.view(np.int32)
    hy = y.view(np.int32)
    ix = hx & np.int32(0x7FFFFFFF)
    iy = hy & np.int32(0x7FFFFFFF)
    if np.any(ix >= 0x7F800000) or np.any(iy >= 0x7F800000):
        raise ValueError("the transcription takes finite winds only")
    m = ((hy >> 31) & 1) | ((hx >> 30) & 2)
    k = (iy - ix) >> 23
    with np.errstate(all="ignore"):
        quotient = np.abs(y / x)
    quotient = np.where(np.isfinite(quotient), quotient, _F(0.0))
    z = _atanf(quotient)
    z = np.where(k > 60, pi_o_2 + _F(0.5) * pi_lo, z)
    z = np.where((hx < 0) & (k < -60), _F(0.0), z).astype(np.float32)
    general = np.where(m == 0, z, np.where(
        m == 1, -z, np.where(m == 2, pi - (z - pi_lo), (z - pi_lo) - pi)))
    y_zero = np.where(m <= 1, y, np.where(m == 2, pi + tiny, -pi - tiny))
    x_zero = np.where(hy < 0, -pi_o_2 - tiny, pi_o_2 + tiny)
    out = np.where(iy == 0, y_zero, np.where(ix == 0, x_zero, general))
    at_one = hx == 0x3F800000
    out = np.where(at_one, _atanf(y), out)
    return out.astype(np.float32)


def _nint(x: np.ndarray) -> np.ndarray:
    """Fortran NINT for x >= 0: round half away from zero, exactly."""
    whole = np.trunc(x)
    return (whole + ((x - whole) >= _F(0.5))).astype(np.int64)


def reference_launch_gwdo(atmosphere, du, dv, *, var, con, oa, ol, sina, cosa,
                          dx: float, dt: float, diagnostics=None):
    """gwdo_column on numpy, float32, statement for statement.

    Same signature and the same in-place contract as the carried
    :func:`arwen_global.core.terrain_drag.launch_gwdo`.  Every operation is
    one float32 operation in the kernel's order (numpy contracts nothing
    into a fused multiply-add); powers are numpy's float32 ``power``, which
    calls the platform's ``powf`` (glibc's on Linux, the words the kernel's
    ``gfk_pow`` reproduces), and ``atan2f`` is the kernel's own ``td_atan2f``
    transcribed.
    """
    F = np.float32
    nz, ny, nx = du.shape
    n = ny * nx

    def lev(a, rows=nz):
        return np.asarray(a, dtype=np.float32).reshape(rows, n)

    def col(a):
        return np.asarray(a, dtype=np.float32).reshape(n)

    u = lev(atmosphere["u"])
    v = lev(atmosphere["v"])
    t = lev(atmosphere["temperature"])
    qv = lev(atmosphere["qv"])
    p = lev(atmosphere["pressure"])
    pk = lev(atmosphere["exner"])
    prsi = lev(atmosphere["p_interface"], nz + 1)
    if "z" in atmosphere:
        zl = lev(atmosphere["z"])
    else:
        zi = lev(atmosphere["z_interface"], nz + 1)
        zl = F(0.5) * (zi[:-1] + zi[1:])
    varc = col(var)
    oc1 = col(con)
    oa4 = np.stack([col(plane) for plane in oa])
    ol4 = np.stack([col(plane) for plane in ol])
    ca = col(cosa)
    sa = col(sina)
    dxmeter = F(dx)
    deltim = F(dt)
    g_, cp_, rd_, fv_, pi_ = _G, _CP, _R_D, _EP_1, _PI
    ric, dw2min, rimin, bnv2min = F(0.25), F(1.0), F(-100.0), F(1.0e-5)
    efmin, efmax, gmax = F(0.0), F(10.0), F(1.0)
    veleps, frc, ce, cg = F(1.0), F(1.0), F(0.8), F(0.5)
    kpblmin = 2
    frmax, olmin, odmin, odmax = F(10.0), F(1.0e-5), F(0.1), F(10.0)
    kte = nz
    mdir = 8
    nwdir = np.array([6, 7, 5, 8, 2, 3, 1, 4])
    one = F(1.0)
    cols = np.arange(n)
    err = np.errstate(all="ignore")
    err.__enter__()
    try:
        # Fortran-indexed (k = 1..nz) accessors.
        def U1(k):
            return u[k - 1] * ca - v[k - 1] * sa

        def V1(k):
            return u[k - 1] * sa + v[k - 1] * ca

        def VTJ(k):
            return t[k - 1] * (one + fv_ * qv[k - 1])

        def VTK(k):
            return VTJ(k) / pk[k - 1]

        def RHO(k):
            return one / rd_ * p[k - 1] / VTJ(k)

        def PRSI(k):
            return prsi[k - 1]

        def PRSL(k):
            return p[k - 1]

        def DEL(k):
            return PRSI(k) - PRSI(k + 1)

        def ZL(k):
            return zl[k - 1]

        def at(array, k):
            """array[k - 1] per column, k an integer array."""
            return array[k - 1, cols]

        fdir = F(mdir) / (F(2.0) * pi_)
        delx = dely = dxmeter
        diag = np.sqrt(np.power(delx, F(2.0)) + np.power(dely, F(2.0)))
        dxy4 = np.array([delx, dely, diag, diag], dtype=np.float32)
        dxy4p = np.array([dxy4[1], dxy4[0], dxy4[3], dxy4[2]], dtype=np.float32)
        cleff = dxmeter

        zlowtop = F(2.0) * varc
        klowtop = np.zeros(n, dtype=np.int64)
        found = np.zeros(n, dtype=bool)
        for k in range(2, kte + 1):
            hit = (~found) & (zlowtop > F(0.0)) & ((ZL(k) - ZL(1)) >= zlowtop)
            klowtop = np.where(hit, k + 1, klowtop)
            found |= hit
        kpblmax = kte
        kbl = np.maximum(np.minimum(klowtop, kpblmax), kpblmin)
        delks = one / (PRSI(1) - at(prsi, kbl))
        delks1 = one / (PRSL(1) - at(p, kbl))
        ubar = np.zeros(n, np.float32)
        vbar = np.zeros(n, np.float32)
        rhobar = np.zeros(n, np.float32)
        for k in range(1, kpblmax + 1):
            m_ = k < kbl
            rcsks = DEL(k) * delks
            rdelks = DEL(k) * delks
            ubar = np.where(m_, ubar + rcsks * U1(k), ubar)
            vbar = np.where(m_, vbar + rcsks * V1(k), vbar)
            rhobar = np.where(m_, rhobar + rdelks * RHO(k), rhobar)

        wdir = _atan2f(ubar, vbar) + pi_
        idir = _nint(fdir * wdir) % mdir + 1
        nwd = nwdir[idir - 1]
        msec = (nwd - 1) % 4
        oa_ = (F(1.0) - F(2.0) * ((nwd - 1) // 4).astype(np.float32)) * oa4[msec, cols]
        oa_ = oa_.astype(np.float32)
        ol_ = ol4[msec, cols]
        ol4p = ol4[[1, 0, 3, 2]]
        olp = ol4p[msec, cols]
        od = olp / np.fmax(ol_, olmin)
        od = np.fmin(od, odmax)
        od = np.fmax(od, odmin)
        dxy = dxy4[msec]
        dxyp = dxy4p[msec]

        def TI(k):
            return F(2.0) / (t[k - 1] + t[k])

        def RDZ(k):
            return one / (ZL(k + 1) - ZL(k))

        def usqj_at(k):
            ti = TI(k)
            rdz = RDZ(k)
            tem1 = U1(k) - U1(k + 1)
            tem2 = V1(k) - V1(k + 1)
            dw2 = tem1 * tem1 + tem2 * tem2
            shr2 = np.fmax(dw2, dw2min) * rdz * rdz
            bvf2 = g_ * (g_ / cp_ + rdz * (VTJ(k + 1) - VTJ(k))) * ti
            return np.fmax(bvf2 / shr2, rimin)

        def bnv2_at(k):
            if k >= kte:
                return np.zeros(n, np.float32)
            rdz = RDZ(k)
            return F(2.0) * g_ * rdz * (VTK(k + 1) - VTK(k)) / (VTK(k + 1) + VTK(k))

        ulow = np.fmax(np.sqrt(ubar * ubar + vbar * vbar), one)
        rulow = one / ulow

        def velco_at(k):
            vc = F(0.5) * ((U1(k) + U1(k + 1)) * ubar + (V1(k) + V1(k + 1)) * vbar)
            vc = vc * rulow
            return np.where((vc < veleps) & (vc > F(0.0)), veleps, vc).astype(np.float32)

        ldrag = velco_at(1) <= F(0.0)
        for k in range(kpblmin, kpblmax + 1):
            if k < kte:
                ldrag = ldrag | ((k < kbl) & (velco_at(k) <= F(0.0)))
        wtkbj = (PRSL(1) - PRSL(2)) * delks1
        bnv2_1 = wtkbj * bnv2_at(1)
        for k in range(kpblmin, kpblmax + 1):
            if k < kte:
                rdelks = (PRSL(k) - PRSL(k + 1)) * delks1
                bnv2_1 = np.where(k < kbl, bnv2_1 + bnv2_at(k) * rdelks, bnv2_1)
        ldrag = ldrag | (bnv2_1 <= F(0.0))
        ldrag = ldrag | (ulow == one)
        ldrag = ldrag | (varc <= F(0.0))
        go = ~ldrag

        bnv = np.where(go, np.sqrt(bnv2_1), F(0.0)).astype(np.float32)
        fr = np.where(go, np.fmin(bnv * rulow * varc * od, frmax), F(0.0)).astype(np.float32)
        xn = np.where(go, ubar * rulow, F(0.0)).astype(np.float32)
        yn = np.where(go, vbar * rulow, F(0.0)).astype(np.float32)
        efact = np.power(oa_ + F(2.0), ce * fr / frc)
        efact = np.fmin(np.fmax(efact, efmin), efmax)
        coefm_on = np.power(one + ol_, oa_ + one)
        xlinv = coefm_on / cleff
        tem = fr * fr * oc1
        gfobnv = gmax * tem / ((tem + cg) * bnv)
        taub_on = xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact
        taub = np.where(go, taub_on, F(0.0)).astype(np.float32)
        coefm = np.where(go, coefm_on, F(0.0)).astype(np.float32)

        taup = np.zeros((kte + 1, n), np.float32)
        for k in range(1, kpblmax + 1):
            taup[k - 1] = np.where(k <= kbl, taub, taup[k - 1])
        icrilv = np.zeros(n, dtype=bool)
        brvf = np.zeros(n, np.float32)
        for k in range(kpblmin, kte):
            kp1 = k + 1
            above = k >= kbl
            usqj = usqj_at(k)
            velco = velco_at(k)
            icrilv = np.where(above, icrilv | (usqj < ric) | (velco <= F(0.0)), icrilv)
            brvf = np.where(above, np.sqrt(np.fmax(bnv2_at(k), bnv2min)), brvf).astype(np.float32)
            live = above & go & (~icrilv) & (taup[k - 1] > F(0.0))
            temv = one / velco
            tem1 = coefm / dxy * (RHO(kp1) + RHO(k)) * brvf * velco * F(0.5)
            hd = np.sqrt(taup[k - 1] / tem1)
            fro = brvf * hd * temv
            tem2 = np.sqrt(usqj)
            temx = one + tem2 * fro
            rim = usqj * (one - fro) / (temx * temx)
            temc = F(2.0) + one / tem2
            hd2 = velco * (F(2.0) * np.sqrt(temc) - temc) / brvf
            broken = tem1 * hd2 * hd2
            # (oa <= 0) || (kp1 >= kpblmin): kp1 >= 3 > kpblmin, always true.
            new = np.where(rim <= ric, broken, taup[k - 1])
            taup[kp1 - 1] = np.where(live, new, taup[kp1 - 1])

        # Flow blocking.
        kblk = np.zeros(n, dtype=np.int64)
        fbdpe = np.zeros(n, np.float32)
        zblk = np.zeros(n, np.float32)
        zl_kbl = at(zl, kbl)
        for k in range(kte, kpblmin - 1, -1):
            active = go & (kblk == 0) & (k <= kbl)
            pe = fbdpe + bnv2_at(k) * (zl_kbl - ZL(k)) * DEL(k) / g_ / RHO(k)
            fbdpe = np.where(active, pe, fbdpe).astype(np.float32)
            fbdke = F(0.5) * (np.power(U1(k), F(2.0)) + np.power(V1(k), F(2.0)))
            hit = active & (fbdpe >= fbdke)
            kk = np.minimum(k, kbl)
            kblk = np.where(hit, kk, kblk)
            zblk = np.where(hit, at(zl, np.maximum(kk, 1)) - ZL(1), zblk).astype(np.float32)
        blocked = kblk != 0
        fbdcd = np.fmax(F(2.0) - one / od, F(0.0))
        taufb = (F(0.5) * rhobar * coefm / (dxmeter * dxmeter) * fbdcd * dxyp * olp
                 * zblk * (ulow * ulow)).astype(np.float32)
        tautem = taufb / np.maximum(kblk - 1, 1).astype(np.float32)
        taup[0] = np.where(blocked, taup[0] + taufb, taup[0])
        for k in range(2, kte + 2):
            inside = blocked & (k <= kblk)
            taufb = np.where(inside, taufb - tautem, taufb).astype(np.float32)
            taup[k - 1] = np.where(inside, taup[k - 1] + taufb,
                                   np.where(blocked, taup[k - 1] + F(0.0), taup[k - 1]))

        def taud_at(k):
            return F(1.0) * (taup[k] - taup[k - 1]) * g_ / DEL(k)

        dtfac = np.ones(n, np.float32)
        for k in range(1, kpblmax):
            td = taud_at(k)
            use = (k <= kbl) & (td != F(0.0))
            dtfac = np.where(use, np.fmin(dtfac, np.abs(velco_at(k) / (deltim * td))),
                             dtfac).astype(np.float32)
        dusfc = np.zeros(n, np.float32)
        dvsfc = np.zeros(n, np.float32)
        du2 = du.reshape(nz, n)
        dv2 = dv.reshape(nz, n)
        dtx = np.zeros((nz, n), np.float32)
        dty = np.zeros((nz, n), np.float32)
        for k in range(1, kte + 1):
            td = taud_at(k) * dtfac
            dtaux = td * xn
            dtauy = td * yn
            dusfc = dusfc + dtaux * DEL(k)
            dvsfc = dvsfc + dtauy * DEL(k)
            du2[k - 1] = du2[k - 1] + dtaux * ca + dtauy * sa
            dv2[k - 1] = dv2[k - 1] - dtaux * sa + dtauy * ca
            dtx[k - 1] = dtaux * ca + dtauy * sa
            dty[k - 1] = -dtaux * sa + dtauy * ca
        dusfc = (F(-1.0) / g_) * dusfc
        dvsfc = (F(-1.0) / g_) * dvsfc
    finally:
        err.__exit__(None, None, None)
    if not (np.shares_memory(du2, du) and np.shares_memory(dv2, dv)):
        du[...] = du2.reshape(du.shape)
        dv[...] = dv2.reshape(dv.shape)
    if diagnostics is not None:
        diagnostics.update(
            DTAUX3D=dtx.reshape(nz, ny, nx), DTAUY3D=dty.reshape(nz, ny, nx),
            DUSFCG=(dusfc * ca + dvsfc * sa).reshape(ny, nx),
            DVSFCG=(-dusfc * sa + dvsfc * ca).reshape(ny, nx))


__all__ = [
    "METADATA_KEY", "OROGRAPHIC_DRAG_SCHEMES", "OrographicStatistics",
    "SSO_FIELDS", "apply_orographic_drag", "attach_orographic_statistics",
    "column_atmosphere", "config_turns_the_drag_on", "prepare_orographic_statistics",
    "reference_launch_gwdo", "stamp_identity",
]
