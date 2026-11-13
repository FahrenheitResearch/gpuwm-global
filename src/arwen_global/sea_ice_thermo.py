"""Thermodynamic sea ice: the mass budget of the pack over a slab ocean.

Design: docs/ARWEN_GLOBAL_OCEAN.md.  The ice TEMPERATURE keeps its one
owner, the native suite's four-node conduction column
(arwen_global.physics.frozen_surface), which this module never steps.
What it adds is the ice MASS: per ocean column an ice fraction A and a
cell-mean ice volume V (m of ice per m2 of cell; thickness V/A), changed by
the energy that reaches them:

* top melt: the energy the column throws away when its top node sits at
  the melting point (:func:`top_melt_surplus`), rho_i L_f per m of ice;
* the base: the conduction the column draws up from its base at the
  freezing point (:func:`basal_conduction`) grows ice; the ocean's net heat
  under and beside the pack (F_open + Q + R, the mixed layer being held at
  the freezing point while ice is present) melts it when positive and
  grows it when negative;
* freeze-up: the deficit the slab step withheld at the freezing point
  forms new ice of thickness ``new_ice_thickness_m`` on the open part;
* lateral melt: on melting, A falls with V as A_new = A sqrt(V_new / V);
* when the whole volume melts, the energy left over warms the mixed layer.

Semtner (1976), doi:10.1175/1520-0485(1976)006<0379:AMFTTG>2.0.CO;2, for
the zero-layer balance, and the regional engine's original Apache-2.0
thermodynamic ice (gpuwm/ocean/seaice) for the area rules; no source is
transcribed.  No snow is carried on the pack here (the column reads the
snow depth the state holds), no ice moves, no ridging.  Arithmetic is
float64 on the array module handed in.
"""
from __future__ import annotations

from typing import NamedTuple

from .surface_series import SEA_WATER_FREEZING_K

RHO_ICE_KG_M3 = 917.0
LATENT_HEAT_FUSION_J_KG = 3.337e5
#: Energy to melt one metre of ice over one square metre (3.06e8 J m-3).
ICE_FUSION_J_M3 = RHO_ICE_KG_M3 * LATENT_HEAT_FUSION_J_KG
#: New ice forms at this thickness on open water (the default of
#: ``[sea_ice] new_ice_thickness_m`` under ``mode = "thermodynamic"``).
DEFAULT_NEW_ICE_THICKNESS_M = 0.5
#: The thickness the Q-flux's ice term reads the climatology's fraction
#: with, by hemisphere (the climatology has no thickness): engineering
#: choices near the observed annual means, stated, not fitted.
ICE_REFERENCE_THICKNESS_M = {"nh": 2.0, "sh": 1.0}
#: Fraction cap, and the volume below which a column holds no ice.
MAX_FRACTION = 1.0
MIN_VOLUME_M = 0.0
#: The freezing point of the mixed layer under ice; the column's base is
#: frozen_surface.SEA_ICE_BOTTOM_K (271.36 K), kept as it is so a run with
#: the ocean off is unchanged.
FREEZING_K = SEA_WATER_FREEZING_K
#: The melting point the column caps its nodes at
#: (physics/frozen_surface.py MELT_POINT_K; the physics package is
#: imported inside the functions that use it, as ocean_rules does).
MELT_POINT_K = 273.15


class IceStep(NamedTuple):
    """The result of :func:`ice_step`, float64 planes.

    ``mixed_layer_heat_j_m2`` is the energy handed back to the mixed layer
    (the surplus after the whole volume melted; >= 0).  ``stored_j_m2`` is
    the change of -rho_i L_f V this step, the ice half of the energy
    ledger."""

    fraction: object
    volume_m: object
    mixed_layer_heat_j_m2: object
    stored_j_m2: object


def _f64(a, xp):
    return xp.asarray(a, dtype=xp.float64)


def ice_reference_thickness(latitude_deg, nlon: int, xp):
    """The (nlat, nlon) plane of :data:`ICE_REFERENCE_THICKNESS_M` by the
    hemisphere of each row: the northern value on rows at or north of the
    equator (a Gaussian grid has no equator row), the southern value on
    rows south of it."""
    lat = _f64(latitude_deg, xp).reshape(-1)
    row = xp.where(
        lat >= 0.0, ICE_REFERENCE_THICKNESS_M["nh"], ICE_REFERENCE_THICKNESS_M["sh"]
    ).astype(xp.float64)
    return xp.broadcast_to(row[:, None], (lat.shape[0], int(nlon))).copy()


def initial_ice(fraction, thickness_m, ocean, new_ice_thickness_m: float, xp):
    """``(A, V)`` at a cold start on ``ocean`` columns: A the analysed
    fraction clipped to [0, 1], V = A times the analysed thickness, a
    column with ice and no thickness taking ``new_ice_thickness_m``; zeros
    off ``ocean``.  A fraction or thickness that is not finite (a masked
    analysis value) reads as no ice and no thickness."""
    a = _f64(fraction, xp)
    a = xp.where(xp.isfinite(a), a, 0.0)
    a = xp.clip(a, 0.0, MAX_FRACTION)
    h = _f64(thickness_m, xp)
    h = xp.where(xp.isfinite(h) & (h > 0.0), h, float(new_ice_thickness_m))
    ocean = xp.asarray(ocean, dtype=bool)
    a = xp.where(ocean, a, 0.0)
    v = xp.where(a > 0.0, a * h, 0.0)
    return a.astype(xp.float64), v.astype(xp.float64)


def thickness(fraction, volume_m, xp):
    """V / A where A > 0, else 0."""
    a = _f64(fraction, xp)
    v = _f64(volume_m, xp)
    ice = a > 0.0
    return xp.where(ice, v / xp.where(ice, a, 1.0), 0.0).astype(xp.float64)


def _column_layout(thickness_m, snow_depth_m, xp):
    """The column's node thicknesses and conductivities exactly as
    frozen_surface builds them (float32 there), widened to float64."""
    from .physics import frozen_surface

    dz = frozen_surface.sea_ice_layer_thickness(thickness_m, snow_depth_m, xp)
    conductivity, _ = frozen_surface.column_properties(dz, snow_depth_m, xp)
    return _f64(dz, xp), _f64(conductivity, xp)


def basal_conduction(*, layers_k, thickness_m, snow_depth_m, xp):
    """W/m2 per m2 of ice, positive upward: the conduction from the base
    at frozen_surface.SEA_ICE_BOTTOM_K into the column's bottom node, with
    the node layout and conductivities frozen_surface uses
    (sea_ice_layer_thickness).

    The conductance is the one :func:`frozen_surface.column_step` gives a
    sea-ice column's base (half the bottom node's thickness over its
    conductivity, no extra resistance), formed in float64 from the
    column's float32 layout: (T_b - T_3) k_3 / (dz_3 / 2)."""
    from .physics import frozen_surface

    dz, conductivity = _column_layout(thickness_m, snow_depth_m, xp)
    t = _f64(layers_k, xp)
    half = 0.5 * dz[3] / conductivity[3]
    return ((float(frozen_surface.SEA_ICE_BOTTOM_K) - t[3]) / half).astype(xp.float64)


def top_melt_surplus(*, layers_k, thickness_m, snow_depth_m, swdown, glw, ice_hfx,
                     ice_qfx, xp):
    """W/m2 per m2 of ice, >= 0: where the column's top node is at the
    melting point, the surface balance at the melting point (absorbed
    shortwave with the sea-ice albedo, absorbed longwave less emission at
    the melting point, less the ice tile's sensible and sublimation fluxes,
    less the conduction into the node below) that the column discarded;
    0 elsewhere.

    The top node is at the melting point where its float32 value equals
    frozen_surface.MELT_POINT_K as float32 (column_step caps every node
    there with a minimum, so a capped node holds exactly that value).  The
    balance is column_step's own surface forcing evaluated at the melting
    point, with the sea-ice albedo and emissivity the native suite gives
    a sea-ice column, the latent flux booked with the heat of sublimation,
    and the conduction to the second node through the two half nodes in
    series, as column_step forms it."""
    from .physics import frozen_surface

    dz, conductivity = _column_layout(thickness_m, snow_depth_m, xp)
    t = _f64(layers_k, xp)
    top32 = xp.asarray(layers_k, dtype=xp.float32)[0]
    at_melt = top32 >= xp.float32(frozen_surface.MELT_POINT_K)
    tm = float(xp.float32(frozen_surface.MELT_POINT_K))
    eps = float(frozen_surface.SEA_ICE_EMISSIVITY)
    sigma = float(frozen_surface.STEFAN_BOLTZMANN_W_M2_K4)
    half = 0.5 * dz / conductivity
    g01 = 1.0 / (half[0] + half[1])
    balance = (
        _f64(swdown, xp) * (1.0 - float(frozen_surface.SEA_ICE_ALBEDO))
        + eps * _f64(glw, xp)
        - eps * sigma * tm ** 4
        - _f64(ice_hfx, xp)
        - float(frozen_surface.LATENT_HEAT_SUBLIMATION_J_KG) * _f64(ice_qfx, xp)
        - g01 * (tm - t[1])
    )
    return xp.where(at_melt, xp.maximum(balance, 0.0), 0.0).astype(xp.float64)


def ice_step(*, fraction, volume_m, ocean_flux_w_m2, basal_conduction_w_m2,
             top_melt_w_m2, freeze_deficit_j_m2, dt_s: float,
             new_ice_thickness_m: float, xp) -> IceStep:
    """One step of the mass budget on every column (a column with no ice and
    no deficit is returned unchanged).

    Volume change, m: dt [A (G_up - M_top) - F_o] / (rho_i L_f) plus
    deficit / (rho_i L_f).  New ice from a negative F_o and from the deficit
    covers open area at ``new_ice_thickness_m`` (A capped at
    :data:`MAX_FRACTION`); melting lowers A by the square-root rule; a
    volume that would go below zero is set to zero with A, and the energy
    beyond it is returned as ``mixed_layer_heat_j_m2``.

    Precisely, per column, with E the energy (J/m2 of cell) the ice mass
    takes this step:

    * a column holds ice when its volume is above :data:`MIN_VOLUME_M`;
      only there do F_o, G_up and M_top act (on an ice-free column the
      slab step has already taken F_o, and what it could not take is the
      deficit), so E = dt [A (G_up - M_top) - F_o] + deficit on a column
      with ice and E = deficit on one without;
    * V_new = max(V + E / (rho_i L_f), 0), and the energy a complete melt
      did not need, max(-(rho_i L_f V + E), 0), is ``mixed_layer_heat_j_m2``;
    * the area: when V_new >= V, the open-water freezing (the deficit, plus
      dt max(-F_o, 0) on a column with ice) covers open area at
      ``new_ice_thickness_m``, A_new = min(A + freezing / (rho_i L_f h_new),
      1); when V_new < V, A_new = A sqrt(V_new / V); A_new is 0 where
      V_new is 0, and a column left with volume and no area (inconsistent
      input) takes min(V_new / h_new, 1);
    * ``stored_j_m2`` = -rho_i L_f (V_new - V), so that, to rounding,
      stored_j_m2 + mixed_layer_heat_j_m2 = -E, which is dt [F_o + A (M_top
      - G_up)] - deficit on a column with ice and -deficit on one without
      (the ice half of the ledger gate OC-3 closes).

    The input fraction of a column without ice is ignored (read as 0).
    """
    a = xp.clip(_f64(fraction, xp), 0.0, MAX_FRACTION)
    v = xp.maximum(_f64(volume_m, xp), MIN_VOLUME_M)
    f_o = _f64(ocean_flux_w_m2, xp)
    g_up = _f64(basal_conduction_w_m2, xp)
    m_top = _f64(top_melt_w_m2, xp)
    deficit = xp.maximum(_f64(freeze_deficit_j_m2, xp), 0.0)
    dt = float(dt_s)
    h_new = float(new_ice_thickness_m)
    if not h_new > 0.0:
        raise ValueError(f"new_ice_thickness_m must be positive, got {new_ice_thickness_m!r}")
    rho_lf = float(ICE_FUSION_J_M3)

    ice = v > MIN_VOLUME_M
    a = xp.where(ice, a, 0.0)
    energy = xp.where(ice, dt * (a * (g_up - m_top) - f_o), 0.0) + deficit
    held = rho_lf * v + energy
    heat = xp.maximum(-held, 0.0)
    v_new = xp.maximum(v + energy / rho_lf, MIN_VOLUME_M)
    v_new = xp.where(held > 0.0, v_new, 0.0)

    freezing = deficit + xp.where(ice, dt * xp.maximum(-f_o, 0.0), 0.0)
    grown = xp.minimum(a + freezing / (rho_lf * h_new), MAX_FRACTION)
    ratio = v_new / xp.where(ice, v, 1.0)
    shrunk = a * xp.sqrt(xp.clip(ratio, 0.0, 1.0))
    a_new = xp.where(v_new >= v, grown, shrunk)
    a_new = xp.where(v_new > 0.0, a_new, 0.0)
    a_new = xp.where((v_new > 0.0) & (a_new <= 0.0), xp.minimum(v_new / h_new, MAX_FRACTION), a_new)
    stored = -rho_lf * (v_new - v)
    return IceStep(
        fraction=a_new.astype(xp.float64),
        volume_m=v_new.astype(xp.float64),
        mixed_layer_heat_j_m2=heat.astype(xp.float64),
        stored_j_m2=stored.astype(xp.float64),
    )


__all__ = [
    "DEFAULT_NEW_ICE_THICKNESS_M", "FREEZING_K", "ICE_FUSION_J_M3",
    "ICE_REFERENCE_THICKNESS_M", "IceStep", "LATENT_HEAT_FUSION_J_KG", "MAX_FRACTION",
    "MELT_POINT_K", "MIN_VOLUME_M", "RHO_ICE_KG_M3", "basal_conduction", "ice_reference_thickness",
    "ice_step", "initial_ice", "thickness", "top_melt_surplus",
]
