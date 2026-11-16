"""The cumulus schemes are handed the pressure velocity omega = Dp/Dt,
not the continuity closure's eta-dot dp/d-eta.

``vertical.continuity`` returns the mass flux THROUGH the hybrid
interfaces, pinned to zero at the surface, so steady flow over a ridge
gives no ascent at all.  The physics exchange adds the interfaces' own
motion, B (dps/dt + ps V . grad ln ps) (Simmons and Burridge 1981), and
the transport keeps the flux.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2
from arwen_global.runner import build_model_and_cold_state
from arwen_global.spectral.backend import get_backend
from arwen_global.vertical import HybridCoordinate

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")


def _ridge(nlev=32):
    """Steady 10 m/s flow over a 1 km Witch-of-Agnesi ridge, 100 km
    half-width, isothermal 280 K, with every layer's mass flux dp*u
    uniform in x, so the flow is exactly steady and non-divergent."""
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    backend = get_backend("numpy", "float64")
    x = np.linspace(-500.0e3, 500.0e3, 2001)
    half_width, height, wind, t0, p0 = 100.0e3, 1000.0, 10.0, 280.0, 100_000.0
    h = height / (1.0 + (x / half_width) ** 2)
    dh_dx = -2.0 * height * x / half_width**2 / (1.0 + (x / half_width) ** 2) ** 2
    rt = DRY_AIR_GAS_CONSTANT * t0
    ps = p0 * np.exp(-GRAVITY_M_S2 * h / rt)
    dlnps_dx = -GRAVITY_M_S2 / rt * dh_dx
    a = vertical.a_half_pa[:, None]
    b = vertical.b_half[:, None]
    dp = np.diff(a + b * ps[None], axis=0)
    dp_far = np.diff(a + b * p0, axis=0)
    u = wind * dp_far / dp
    shape = (1, x.size)
    return dict(
        vertical=vertical, backend=backend, x=x, dh_dx=dh_dx, rt=rt,
        ps=ps.reshape(shape), dlnps_dx=dlnps_dx.reshape(shape),
        u=u.reshape((nlev, *shape)), v=np.zeros((nlev, *shape)),
        div_mass=np.zeros((nlev, *shape)),
    )


def test_ridge_exchange_omega_is_the_flow_up_the_slope():
    r = _ridge()
    vertical, backend = r["vertical"], r["backend"]
    ps_t = -np.sum(r["div_mass"], axis=0)
    flux, _closure = vertical.continuity(r["div_mass"], ps_t, backend)
    # The bug: the closure's flux sees no ascent anywhere.
    assert np.max(np.abs(flux)) < 1.0e-9
    omega = vertical.pressure_velocity(
        flux, ps_t, r["ps"], r["u"], r["v"], r["dlnps_dx"],
        np.zeros_like(r["dlnps_dx"]), backend,
    )
    assert omega.shape == flux.shape
    assert np.all(omega[0] == 0.0)  # pressure lid, B = 0
    rho_surface = r["ps"][0] / r["rt"]
    w_surface = -omega[-1, 0] / (rho_surface * GRAVITY_M_S2)
    expected = r["u"][-1, 0] * r["dh_dx"]
    # Exact for this column: w = u dh/dx at the ground.
    np.testing.assert_allclose(w_surface, expected, rtol=1.0e-10, atol=1.0e-12)
    # And within 15% of the upstream wind's U dh/dx (u speeds up by the
    # layer-thinning ratio over the crest), peak ascent about 0.065 m/s.
    upstream = 10.0 * r["dh_dx"]
    peak = np.argmax(upstream)
    assert 0.06 < w_surface[peak] < 0.08
    assert abs(w_surface[peak] / upstream[peak] - 1.0) < 0.15
    # Upslope ascent, downslope descent.
    assert np.all(w_surface[r["x"] < -1.0e3] > 0.0)
    assert np.all(w_surface[r["x"] > 1.0e3] < 0.0)


def _model():
    cfg = replace(load_config(CONFIG), latitude_bands=1, host_spill="off")
    model, state = build_model_and_cold_state(cfg)
    return cfg, model, state


def test_exchange_omega_adds_the_interface_motion_and_transport_keeps_the_flux():
    _cfg, model, state = _model()
    xp = model.transform.backend.xp
    sources = model.grid_sources(
        state.atmosphere, only=model._PHYSICS_GRID_KEYS
    )
    _spectral, div_mass, ps_t, flux = model._mass_flux_and_omega(sources)
    # Transport's flux is the continuity closure's, unchanged: pinned to
    # zero at the top and the surface.
    direct, _ = model.vertical.continuity(div_mass, ps_t, model.transform.backend)
    np.testing.assert_array_equal(np.asarray(flux), np.asarray(direct))
    assert np.all(np.asarray(flux[-1]) == 0.0)

    exchange, _fixer = model._physics_exchange_with_closure(state, 5.0)
    omega = np.asarray(exchange.omega_half_pa_s)
    g, _stack, _names = model.grid_band(sources, model.whole_rows)
    east, north = model.transform.gradient(state.atmosphere.log_surface_pressure)
    ps = np.asarray(g["p_half"][-1])
    adv = np.asarray(g["u"]) * np.asarray(east)[None] + np.asarray(g["v"]) * np.asarray(north)[None]
    adv_half = np.concatenate([adv[:1], 0.5 * (adv[:-1] + adv[1:]), adv[-1:]])
    b = np.asarray(model.vertical.b_half)[:, None, None]
    expected = np.asarray(flux) + b * (np.asarray(ps_t)[None] + ps[None] * adv_half)
    np.testing.assert_allclose(omega, expected, rtol=1.0e-10, atol=1.0e-9)
    # The surface omega is ps_t + V . grad ps, not zero.
    assert np.max(np.abs(omega[-1])) > 1.0e-4
    del xp
