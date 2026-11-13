"""Two opt-in couplings between the cloud and the radiation.

Both are off by default, and off is the call the native runtime always made.

``radiation_liquid_radius = "wrf_relcalc"``
    The liquid effective radius WRF's own RRTMG radiates for Morrison cloud.
    WRF v4.7.1 sets ``has_reqc`` (hand RRTMG the scheme's own radius) for
    Thompson, WSM/WDM, NSSL 2-moment, Jensen-Ishmael and P3 and NOT for
    Morrison (mp_physics = 10): ``phys/module_physics_init.F:1000-1024``.  So
    WRF radiates Morrison liquid at ``relcalc``'s prescription
    (``phys/module_ra_rrtmg_lw.F:14542-14601``, called at
    ``module_ra_rrtmg_sw.F:11104-11107`` and ``module_ra_rrtmg_lw.F:12569-12572``,
    Kiehl's CAM3 rule): 8 um over warm snow-free land, 14 um over ocean and
    sea ice, ramped to 14 um below freezing and under snow.  Its inputs are
    ``landfrac = landm = 2 - XLAND``, ``icefrac = XICE`` and
    ``snowh = 0.001 * SNOW`` (``module_ra_rrtmg_sw.F:11020-11023``).  This
    package has radiated Morrison's own ``effc`` instead; the option restores
    WRF's coupling for the liquid.  The ice keeps Morrison's own sizes, the
    area-conserving merge of 2026-09-04 (WRF's ``reicalc`` would make the ice
    brighter, measured +3.0 to +3.9 W/m2 of reflected sunlight).

``radiation_cloud_inhomogeneity = "tiedtke1996"``
    Liquid and ice in-cloud paths times 0.7 in both solvers: the effective
    thickness of an inhomogeneous cloud (Cahalan et al. 1994, J. Atmos. Sci.
    51, 2434, about 0.7 for marine stratocumulus), as the ECMWF model applied
    it to liquid and ice, shortwave and longwave (Tiedtke 1996, Mon. Wea.
    Rev. 124, 745; described so by Shonk and Hogan 2008, J. Climate 21,
    2352).  WRF's RRTMG and this package's McICA radiate a cloudy subcolumn
    as homogeneous, so nothing here already accounts for it.  A declared
    departure from WRF.
"""

from __future__ import annotations

import numpy as np

#: module_ra_rrtmg_lw.F:14582-14585 (WRF v4.7.1).
RELCALC_TMELT_K = 273.16
RELCALC_RLIQ_OCEAN_UM = 14.0
RELCALC_RLIQ_ICE_UM = 14.0
RELCALC_RLIQ_LAND_UM = 8.0

RADIATION_LIQUID_RADIUS_MODES = frozenset({"scheme", "wrf_relcalc"})

#: Mode -> factor on the in-cloud liquid and ice paths (None: unchanged).
CLOUD_INHOMOGENEITY_FACTORS = {"none": None, "tiedtke1996": 0.7}


def wrf_relcalc(temperature, landm, icefrac, snow_kg_m2, xp=np):
    """WRF ``relcalc`` (module_ra_rrtmg_lw.F:14586-14600), float32.

    ``temperature`` is the (nz, ny, nx) layer temperature in K; ``landm`` is
    WRF's ``2 - XLAND`` on (ny, nx); ``icefrac`` the sea-ice fraction;
    ``snow_kg_m2`` the snow water equivalent (WRF's ``SNOW``; ``snowh`` is
    0.001 of it in metres).  Returns the radius in microns, (nz, ny, nx).
    The four steps are WRF's, in WRF's order.
    """
    f32 = xp.float32
    one, zero = f32(1.0), f32(0.0)
    rl, ro, ri = (f32(RELCALC_RLIQ_LAND_UM), f32(RELCALC_RLIQ_OCEAN_UM),
                  f32(RELCALC_RLIQ_ICE_UM))
    t = xp.asarray(temperature, dtype=xp.float32)
    # Temperature-dependent value, continental air.
    rel = rl + (ro - rl) * xp.minimum(
        one, xp.maximum(zero, (f32(RELCALC_TMELT_K) - t) * f32(0.05)))
    # Snow depth over land.
    snowh = xp.asarray(snow_kg_m2, dtype=xp.float32) * f32(0.001)
    rel = rel + (ro - rel) * xp.minimum(one, xp.maximum(zero, snowh * f32(10.0)))[None]
    # Polluted land value ramped to the clean ocean value.
    land = xp.asarray(landm, dtype=xp.float32)
    rel = rel + (ro - rel) * xp.minimum(one, xp.maximum(zero, one - land))[None]
    # Ramped to the sea-ice value where there is ice.
    ice = xp.asarray(icefrac, dtype=xp.float32)
    rel = rel + (ri - rel) * xp.minimum(one, xp.maximum(zero, ice))[None]
    return xp.ascontiguousarray(rel.astype(xp.float32))


__all__ = [
    "CLOUD_INHOMOGENEITY_FACTORS",
    "RADIATION_LIQUID_RADIUS_MODES",
    "wrf_relcalc",
]
