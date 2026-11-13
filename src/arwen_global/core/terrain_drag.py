"""Sub-grid orographic drag from WRF v4.7.1, on the device: gwd_opt = 1.

Carried from the engine's terrain-drag launcher (gpuwm 2.9.13), cut to the
one option this model runs:

* ``gwd_opt = 1`` -- the KIM orographic gravity-wave drag with flow blocking
  (Choi and Hong 2015; ``module_bl_gwdo.F`` -> ``bl_gwdo_run``), from VAR,
  CON, OA1-4 and OL1-4: :func:`launch_gwdo`.  WRF adds it to the PBL
  momentum tendencies after the PBL scheme, at the PBL cadence
  (``module_pbl_driver.F:2129-2185``), and so does this model's native suite
  (``arwen_global.physics.orographic_drag``).

The engine's module also carries ``topo_wind`` (YSU's surface-wind
correction over complex terrain), ``gwd_opt = 3`` (the GSL drag suite) and
the domain object that builds them from a regional domain's static fields.
None of them is carried: this package runs neither option, and its static
fields come from its own Gaussian-grid statics (``arwen_global.
orographic_statics``), not from a regional domain's.  What is here is the
engine's launcher and compile route for ``gwdo_column`` unchanged, and the
kernel itself (``kernels/terrain_drag.cu``) is the engine's ``gwdo_column``
word for word.  The engine holds the kernel to WRF's own Fortran by the
oracle in ``tools/gwdo_kim_wrf461_oracle``
(``tests/test_gwdo_kim_wrf461_parity.py``); this package holds its copy to
the same fixture (``tests/test_arwen_global_orographic_drag.py``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

#: WRF v4.7.1 gwd_opt values ported here.
GWD_OPTIONS = (0, 1)

#: No FMA contraction (gfortran contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "arwen_global.core.terrain_drag:terrain_drag"
_SOURCES = ("glibc_flt32.cuh", "terrain_drag.cu")
_BLOCK = 128
_KMAX = 128
#: A null device pointer for the kernels' optional diagnostic outputs.
_NULL = np.uintp(0)

#: WRF's module_model_constants values the PBL driver hands both schemes:
#: CP = 7*R_D/2, G, R_D, R_V, EP_1 = R_V/R_D - 1 (all float32 in WRF) and
#: the literal PI=3.141592653 of module_pbl_driver.F:2141 and :2177.
_G = np.float32(9.81)
_R_D = np.float32(287.0)
_R_V = np.float32(461.6)
_CP = np.float32(np.float32(7.0) * _R_D / np.float32(2.0))
_EP_1 = np.float32(_R_V / _R_D - np.float32(1.0))
_PI = np.float32(3.141592653)


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from arwen_global.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device.

    Not ``cupy.RawModule``: CuPy appends ``-ftz=true`` after the caller's
    options and NVRTC honours the last one, so ``--ftz=false`` would be
    dropped.  MEASURED on the RTX 5090 (sm_120) and RTX 4090 (sm_89),
    cupy 14.2: through RawModule a product of two normal floats with a
    subnormal result came back 0 on both cards, and on sm_120 ``x / 10.0f``
    came back as the reciprocal product (0.6936707 for (-23.063293 + 30) /
    10, where IEEE division and gfortran give 0.69367063).  Through
    ``compile_using_nvrtc`` both are IEEE on both cards.  The form-drag
    tendencies (``gwdo_gsl_column``) are subnormal well inside the column
    and topo_wind's taper divides by 10, so the loader is what holds the
    kernels to WRF.  ``-arch`` is left to CuPy (NVRTC 13 refuses it twice).
    """
    import cupy as cp
    from gpuwm import nvrtc_ptx_cache as compiler

    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "terrain_drag.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function(name)


def _f32(xp, array, shape, name):
    out = xp.ascontiguousarray(xp.asarray(array, dtype=xp.float32))
    if out.shape != tuple(shape):
        raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return out


def _stack4(cp, planes, shape, name):
    arrays = [_f32(cp, plane, shape, f"{name}{m + 1}")
              for m, plane in enumerate(planes)]
    return cp.ascontiguousarray(cp.stack(arrays))


def _column_inputs(cp, atmosphere, nz, ny, nx):
    names = ("u", "v", "temperature", "qv", "pressure", "exner")
    out = {name: _f32(cp, atmosphere[name], (nz, ny, nx), name)
           for name in names}
    out["p_interface"] = _f32(cp, atmosphere["p_interface"], (nz + 1, ny, nx),
                              "p_interface")
    if "z" in atmosphere:
        out["z"] = _f32(cp, atmosphere["z"], (nz, ny, nx), "z")
    else:
        # phy_prep (module_big_step_utilities_em.F:4893):
        # z = 0.5*(z_at_w(k) + z_at_w(k+1)), heights above sea level.
        zi = _f32(cp, atmosphere["z_interface"], (nz + 1, ny, nx),
                  "z_interface")
        out["z"] = cp.ascontiguousarray(
            cp.float32(0.5) * (zi[:-1] + zi[1:]))
    out["dz"] = _f32(cp, atmosphere["dz"], (nz, ny, nx), "dz")
    return out


def launch_gwdo(atmosphere: Mapping[str, object], du, dv, *, var, con, oa,
                ol, sina, cosa, dx: float, dt: float, diagnostics=None):
    """gwd_opt = 1: add WRF's KIM orographic drag to ``du``/``dv`` in place.

    ``atmosphere`` is the physics driver's phy_prep dict (mass-point winds,
    temperature, vapour, hydrostatic pressures, Exner, interface heights).
    ``oa``/``ol`` are the four directional planes each.  ``diagnostics``,
    when a dict, receives DTAUX3D, DTAUY3D, DUSFCG, DVSFCG.
    """
    import cupy as cp

    nz, ny, nx = du.shape
    if nz > _KMAX:
        raise ValueError(f"gwd_opt = 1 holds a column of at most {_KMAX} "
                         f"levels per thread, got {nz}")
    col = _column_inputs(cp, atmosphere, nz, ny, nx)
    shape2 = (ny, nx)
    oa4 = _stack4(cp, oa, shape2, "oa")
    ol4 = _stack4(cp, ol, shape2, "ol")
    dtx = dty = dsx = dsy = None
    if diagnostics is not None:
        dtx = cp.zeros((nz, ny, nx), dtype=cp.float32)
        dty = cp.zeros((nz, ny, nx), dtype=cp.float32)
        dsx = cp.zeros(shape2, dtype=cp.float32)
        dsy = cp.zeros(shape2, dtype=cp.float32)

    n = ny * nx
    _kernel("gwdo_column")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (col["u"], col["v"], col["temperature"], col["qv"], col["pressure"],
         col["p_interface"], col["exner"], col["z"],
         _f32(cp, var, shape2, "var"), _f32(cp, con, shape2, "con"),
         oa4, ol4, _f32(cp, sina, shape2, "sina"),
         _f32(cp, cosa, shape2, "cosa"), du, dv,
         dtx if dtx is not None else _NULL, dty if dty is not None else _NULL,
         dsx if dsx is not None else _NULL, dsy if dsy is not None else _NULL,
         np.float32(dx), np.float32(dt), _G, _CP, _R_D, _EP_1, _PI,
         np.int32(nz), np.int32(ny), np.int32(nx)))
    if diagnostics is not None:
        diagnostics.update(DTAUX3D=dtx, DTAUY3D=dty, DUSFCG=dsx, DVSFCG=dsy)

