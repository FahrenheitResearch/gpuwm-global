"""Analytic dynamical-core test cases with a published answer.

Two cases live here, and both exist because an integrator that only ever
runs against real data can be wrong in a way nothing measures.

*   The Jablonowski and Williamson 2006 baroclinic wave.  A steady,
    balanced, analytically specified state that a correct core keeps
    steady for days, plus the same state with a 1 m/s wind perturbation
    that grows into a wave train.  The steady arm measures whether the
    core is a fixed point of a state it did not construct; the perturbed
    arm measures whether it grows the instability at the right rate, and
    an integrator that damped the baroclinic growth would pass every
    conservation gate in the tree while forecasting nothing.

*   The Held and Suarez 1994 forcing.  Newtonian relaxation to a
    prescribed equilibrium temperature plus Rayleigh friction in the
    boundary layer, with no radiation, no moisture and no surface: the
    standard way to ask what a dynamical core's CLIMATE looks like after
    a thousand days, which is the timescale on which an over-dissipative
    interpolation shows up and a twenty-four hour forecast does not.

The forcing is written as a state-to-state operator rather than as a
physics suite.  It relaxes temperature and wind and touches nothing else,
so a suite's exchange, closure ledger and water reservoir would all be
machinery with nothing to do; the split is the one the case itself
specifies.
"""
from __future__ import annotations

import math

import numpy as np

from arwen_global.spectral.vector import VorticityDivergenceOperator

from .constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)
from .state import ArwenGlobalState, MoistHybridState, PhysicsState, SurfaceState
from .lnps_reference import LNPS_REFERENCE, lnps_from_anomaly_coefficients

# --------------------------------------------------------------------------
# Jablonowski and Williamson 2006.

#: Jet speed of the analytic zonal flow, m/s.
JW_U0 = 35.0
#: Reference surface temperature and lapse rate of the mean profile.
JW_T0 = 288.0
JW_LAPSE_K_M = 0.005
#: Stratospheric temperature increment and the level it starts at.
JW_DELTA_T = 4.8e5
JW_ETA_TOP = 0.2
#: Centre of the vertical jet profile.
JW_ETA_0 = 0.252
#: Uniform surface pressure of the case, Pa.
JW_SURFACE_PRESSURE_PA = 1.0e5
#: The perturbation: 1 m/s of zonal wind in a Gaussian of radius a/10
#: centred at 20 E, 40 N.
JW_PERTURBATION_M_S = 1.0
JW_PERTURBATION_RADIUS_FRACTION = 0.1
JW_PERTURBATION_LON_RAD = math.pi / 9.0
JW_PERTURBATION_LAT_RAD = 2.0 * math.pi / 9.0


def _jw_shape(sin_lat, cos_lat):
    """The two latitude shapes both the temperature and the geopotential
    of the analytic state are written in."""
    first = -2.0 * sin_lat ** 6 * (cos_lat ** 2 + 1.0 / 3.0) + 10.0 / 63.0
    second = (
        (8.0 / 5.0) * cos_lat ** 3 * (sin_lat ** 2 + 2.0 / 3.0) - math.pi / 4.0
    )
    return first, second


def jablonowski_williamson_fields(
    eta, lat_rad, *, radius_m: float, rotation_rate_s: float = EARTH_ROTATION_RATE_S,
    gas_constant: float = DRY_AIR_GAS_CONSTANT, u0_m_s: float = JW_U0,
):
    """``(u, temperature, surface_geopotential)`` of the steady state.

    ``eta`` is ``p / ps`` at the model's full levels, shaped to broadcast
    against ``lat_rad``.  Everything is float64 host arithmetic: the case
    is a specification and the model reads it once.

    ``u0_m_s`` is the jet speed.  The state is an exact steady solution of
    the primitive equations for any jet speed and any rotation rate; the
    published case is ``JW_U0`` at Earth's rate, which the defaults give
    bit for bit.
    """
    eta = np.asarray(eta, dtype=np.float64)
    lat = np.asarray(lat_rad, dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    first, second = _jw_shape(sin_lat, cos_lat)
    exponent = gas_constant * JW_LAPSE_K_M / GRAVITY_M_S2
    u0 = float(u0_m_s)

    eta_v = (eta - JW_ETA_0) * (math.pi / 2.0)
    cos_v = np.cos(eta_v)
    zonal = u0 * cos_v ** 1.5 * np.sin(2.0 * lat) ** 2

    mean_t = JW_T0 * eta ** exponent
    mean_t = np.where(
        eta < JW_ETA_TOP,
        mean_t + JW_DELTA_T * (JW_ETA_TOP - eta) ** 5,
        mean_t,
    )
    temperature = mean_t + 0.75 * (
        eta * math.pi * u0 / gas_constant
    ) * np.sin(eta_v) * np.sqrt(cos_v) * (
        first * 2.0 * u0 * cos_v ** 1.5
        + second * radius_m * rotation_rate_s
    )

    # The surface geopotential is the same expression at eta = 1, whose
    # mean part is zero: the case's orography IS the balance of its jet.
    surface_v = (1.0 - JW_ETA_0) * (math.pi / 2.0)
    cos_s = math.cos(surface_v)
    surface_geopotential = u0 * cos_s ** 1.5 * (
        first * u0 * cos_s ** 1.5 + second * radius_m * rotation_rate_s
    )
    return zonal, temperature, surface_geopotential


def jablonowski_williamson_perturbation(lat_rad, lon_rad, radius_m: float):
    """The 1 m/s zonal wind bump, as a function of position alone."""
    lat = np.asarray(lat_rad, dtype=np.float64)
    lon = np.asarray(lon_rad, dtype=np.float64)
    cos_r = (
        math.sin(JW_PERTURBATION_LAT_RAD) * np.sin(lat)
        + math.cos(JW_PERTURBATION_LAT_RAD) * np.cos(lat)
        * np.cos(lon - JW_PERTURBATION_LON_RAD)
    )
    great_circle = np.arccos(np.clip(cos_r, -1.0, 1.0))
    scale = JW_PERTURBATION_RADIUS_FRACTION
    return JW_PERTURBATION_M_S * np.exp(-((great_circle / scale) ** 2))


def baroclinic_wave_initial_state(cfg, transform, statics=None, *,
                                  rotation_rate_s: float = EARTH_ROTATION_RATE_S,
                                  u0_m_s: float = JW_U0):
    """The Jablonowski and Williamson 2006 state, in this model's own
    variables: ``(bundle, surface_geopotential, provenance)``.

    ``cfg.perturbation_amplitude`` scales the wind bump, so the STEADY
    arm of the gate is the same config with the amplitude at zero and
    therefore the same everything else.

    ``rotation_rate_s`` and ``u0_m_s`` build the same exact steady state
    for another planet's rotation and another jet (the rotation ladder of
    ``tools/arwen_global_world_benchmarks.py``).  At their defaults the
    state and the provenance are the published case's, bit for bit; a
    model that integrates the state must be given the same rotation rate.
    """
    backend = transform.backend
    xp = backend.xp
    grid = transform.grid
    lat = np.asarray(grid.lat_rad, dtype=np.float64)[:, None]
    lon = np.asarray(grid.lon_rad, dtype=np.float64)[None, :]
    ps = np.full(grid.shape, JW_SURFACE_PRESSURE_PA)
    pressure = cfg.vertical.pressure(
        np.asarray(ps, dtype=np.float64), _HostBackend()
    )
    eta = (pressure["p_full"] / JW_SURFACE_PRESSURE_PA)[:, :1, :1]
    zonal, temperature, surface_geopotential = jablonowski_williamson_fields(
        eta, lat[None], radius_m=grid.radius_m,
        rotation_rate_s=rotation_rate_s, u0_m_s=u0_m_s,
    )
    u = np.broadcast_to(zonal, pressure["p_full"].shape).copy()
    amplitude = float(cfg.perturbation_amplitude)
    if amplitude != 0.0:
        u = u + amplitude * jablonowski_williamson_perturbation(
            lat, lon, grid.radius_m
        )[None]
    temperature = np.broadcast_to(
        temperature, pressure["p_full"].shape
    ).copy()
    theta = temperature / (pressure["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
    surface_geopotential = np.broadcast_to(
        np.asarray(surface_geopotential).reshape(-1)[:, None], grid.shape
    ).copy()

    dtype = backend.float_dtype
    vector = VorticityDivergenceOperator(transform)
    vorticity, divergence = vector.vordiv_from_wind(
        backend.asarray(u, dtype=dtype),
        backend.asarray(np.zeros_like(u), dtype=dtype),
    )
    zeros_grid = xp.zeros(
        (cfg.vertical.nlev, *grid.shape), dtype=dtype
    )
    zeros_spectral = xp.zeros(
        (cfg.vertical.nlev, *transform.spectral_shape),
        dtype=backend.complex_dtype,
    )
    atmosphere = MoistHybridState(
        vorticity=vorticity,
        divergence=divergence,
        theta=transform.project(
            transform.forward(backend.asarray(theta, dtype=dtype))
        ),
        # About ln(p_ref), so the float32 analysis never rounds at 11.5
        # (arwen_global.lnps_reference).
        log_surface_pressure=lnps_from_anomaly_coefficients(
            transform, transform.project(transform.forward(backend.asarray(
                np.log(ps) - LNPS_REFERENCE, dtype=dtype)))
        ),
        qv=zeros_spectral,
        **{
            name: zeros_grid.copy()
            for name in ("qc", "qr", "qi", "qs", "qg",
                         "nc", "nr", "ni", "ns", "ng")
        },
    )
    surface = _dry_surface(transform, temperature[-1])
    provenance = {
        "mode": "baroclinic_wave",
        "case": "jablonowski-williamson-2006",
        "perturbation_m_s": amplitude,
        "surface_pressure_pa": JW_SURFACE_PRESSURE_PA,
    }
    # Only a state that is not the published case says so, so the
    # published case's provenance (and every checkpoint that carries it)
    # is unchanged.
    if float(rotation_rate_s) != EARTH_ROTATION_RATE_S:
        provenance["rotation_rate_s"] = float(rotation_rate_s)
    if float(u0_m_s) != JW_U0:
        provenance["jet_m_s"] = float(u0_m_s)
    return (
        ArwenGlobalState(atmosphere, surface, PhysicsState()),
        backend.asarray(surface_geopotential, dtype=dtype),
        provenance,
    )


class _HostBackend:
    """The minimal backend ``HybridCoordinate.pressure`` reads, in float64
    on the host: the case is a specification and is built once."""

    name = "numpy"
    xp = np
    float_dtype = np.float64
    complex_dtype = np.complex128

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def to_numpy(value):
        return np.asarray(value)


def _dry_surface(transform, surface_temperature) -> SurfaceState:
    """A surface that no scheme reads: the dry cases run with physics off,
    and the state still has to carry a valid one."""
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    shape = transform.grid.shape
    temperature = backend.asarray(
        np.broadcast_to(np.asarray(surface_temperature), shape).copy(),
        dtype=dtype,
    )
    from .statics import synthetic_surface_statics

    nsoil = 4
    soil = xp.stack([temperature for _ in range(nsoil)])
    land_fraction = np.zeros(shape)
    resolved = synthetic_surface_statics(
        land_fraction, np.broadcast_to(
            backend.to_numpy(temperature)[None], (nsoil, *shape)
        ).copy(),
    )
    resolved.pop("land_fraction")
    return SurfaceState(
        temperature_k=temperature,
        water_kg_m2=xp.zeros(shape, dtype=dtype),
        land_fraction=xp.zeros(shape, dtype=dtype),
        heat_capacity_j_m2_k=xp.full(shape, 4.0e7, dtype=dtype),
        soil_temperature_k=soil,
        soil_water_fraction=xp.full((nsoil, *shape), 0.25, dtype=dtype),
        accumulated_rain_kg_m2=xp.zeros(shape, dtype=dtype),
        accumulated_snow_kg_m2=xp.zeros(shape, dtype=dtype),
        accumulated_graupel_kg_m2=xp.zeros(shape, dtype=dtype),
        **{name: backend.asarray(value, dtype=dtype)
           for name, value in resolved.items()},
    )


# --------------------------------------------------------------------------
# An atmosphere at rest over terrain.
#
# Temperature that is a function of pressure alone, no wind, and a surface
# pressure that is the same profile's hydrostatic pressure at the height of
# the ground.  Every pressure surface is then level and isothermal, so the
# continuous equations hold this state still over ANY terrain.  What moves
# in a model is what its discrete pressure-gradient force, its hydrostatic
# sum and its spectral truncation of the ground leave unbalanced on
# terrain-following levels: the wind that grows is the core's own error
# over mountains, and with no terrain it is rounding.  This is the resting
# case of DCMIP 2012 (test 2-0) with the US standard atmosphere's lower
# profile, so the model top at 1 hPa sits in an isothermal stratosphere
# rather than at the 78 K a constant lapse would reach there.

#: Sea-level pressure and temperature, lapse rate and tropopause
#: temperature of the profile (the US standard atmosphere, 1976).
REST_SEA_LEVEL_PRESSURE_PA = 101325.0
REST_SEA_LEVEL_TEMPERATURE_K = 288.15
REST_LAPSE_K_M = 0.0065
REST_TROPOPAUSE_K = 216.65


def rest_atmosphere_temperature(pressure_pa, *,
                                gas_constant: float = DRY_AIR_GAS_CONSTANT):
    """Temperature of the resting profile at ``pressure_pa``, which is a
    function of pressure and nothing else."""
    p = np.asarray(pressure_pa, dtype=np.float64)
    exponent = gas_constant * REST_LAPSE_K_M / GRAVITY_M_S2
    return np.maximum(
        REST_SEA_LEVEL_TEMPERATURE_K * (p / REST_SEA_LEVEL_PRESSURE_PA) ** exponent,
        REST_TROPOPAUSE_K,
    )


def rest_atmosphere_surface_pressure(height_m, *,
                                     gas_constant: float = DRY_AIR_GAS_CONSTANT):
    """The resting profile's hydrostatic pressure at ``height_m`` above sea
    level: the closed form of the constant-lapse layer below the
    tropopause and of the isothermal layer above it."""
    z = np.asarray(height_m, dtype=np.float64)
    power = GRAVITY_M_S2 / (gas_constant * REST_LAPSE_K_M)
    z_trop = (REST_SEA_LEVEL_TEMPERATURE_K - REST_TROPOPAUSE_K) / REST_LAPSE_K_M
    p_trop = REST_SEA_LEVEL_PRESSURE_PA * (
        REST_TROPOPAUSE_K / REST_SEA_LEVEL_TEMPERATURE_K) ** power
    below = REST_SEA_LEVEL_PRESSURE_PA * (
        1.0 - REST_LAPSE_K_M * np.minimum(z, z_trop)
        / REST_SEA_LEVEL_TEMPERATURE_K) ** power
    above = p_trop * np.exp(
        -GRAVITY_M_S2 * (z - z_trop) / (gas_constant * REST_TROPOPAUSE_K))
    return np.where(z <= z_trop, below, above)


def terrain_on_model_grid(latitude_deg, longitude_deg, values, grid):
    """Periodic bilinear interpolation of a regular global latitude-longitude
    field (Earth's terrain height, say) onto the model's Gaussian grid.

    The analysis cold start regrids its terrain the same way (bilinear,
    then the spectral truncation), through a Rust door; this is plain
    numpy so a benchmark needs no door.  A field that is not regular, not
    a full longitude ring or does not reach the grid's outermost rows is
    refused by name.
    """
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    field = np.asarray(values, dtype=np.float64)
    if field.shape != (lat.size, lon.size):
        raise ValueError(
            f"terrain shape {field.shape} is not (latitude, longitude) = "
            f"({lat.size}, {lon.size})")
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1]
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    if (np.max(np.abs(dlat - dlat[0])) > 1.0e-6
            or np.max(np.abs(dlon - dlon[0])) > 1.0e-6 or dlat[0] <= 0.0):
        raise ValueError("terrain grid must be regular in latitude and longitude")
    if abs(float(dlon[0]) * lon.size - 360.0) > 1.0e-3:
        raise ValueError("terrain longitudes must close the globe")
    target_lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    target_lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    if target_lat.min() < lat[0] - 1.0e-9 or target_lat.max() > lat[-1] + 1.0e-9:
        raise ValueError("terrain latitudes do not reach the model grid's outer rows")
    row = np.clip(np.searchsorted(lat, target_lat) - 1, 0, lat.size - 2)
    wy = ((target_lat - lat[row]) / (lat[row + 1] - lat[row]))[:, None]
    x = np.mod(target_lon - lon[0], 360.0) / float(dlon[0])
    col = np.floor(x).astype(np.int64)
    wx = (x - col)[None, :]
    col = np.mod(col, lon.size)
    col1 = np.mod(col + 1, lon.size)
    lower = field[row][:, col] * (1.0 - wx) + field[row][:, col1] * wx
    upper = field[row + 1][:, col] * (1.0 - wx) + field[row + 1][:, col1] * wx
    return lower * (1.0 - wy) + upper * wy


def rest_atmosphere_initial_state(cfg, transform, terrain_height_m):
    """The resting atmosphere over ``terrain_height_m`` (metres, on the
    model grid), in this model's own variables: ``(bundle,
    surface_geopotential, provenance)``.

    The terrain passes through the model's spectral truncation first, as
    the analysis cold start's does, and the surface pressure is the
    profile's at the truncated height, so the start is the model's own
    best rest state and what it does next is the core's.
    """
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    grid = transform.grid
    height = np.asarray(terrain_height_m, dtype=np.float64)
    if height.shape != tuple(grid.shape):
        raise ValueError(
            f"terrain shape {height.shape} is not the model grid {grid.shape}")
    surface_geopotential = np.asarray(backend.to_numpy(transform.inverse(
        transform.forward(backend.asarray(GRAVITY_M_S2 * height, dtype=dtype))
    )), dtype=np.float64)
    ps = rest_atmosphere_surface_pressure(surface_geopotential / GRAVITY_M_S2)
    pressure = cfg.vertical.pressure(ps, _HostBackend())
    p_full = np.asarray(pressure["p_full"], dtype=np.float64)
    temperature = rest_atmosphere_temperature(p_full)
    theta = temperature / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA

    zeros_grid = xp.zeros((cfg.vertical.nlev, *grid.shape), dtype=dtype)
    zeros_spectral = xp.zeros(
        (cfg.vertical.nlev, *transform.spectral_shape),
        dtype=backend.complex_dtype,
    )
    atmosphere = MoistHybridState(
        vorticity=zeros_spectral.copy(),
        divergence=zeros_spectral.copy(),
        theta=transform.project(
            transform.forward(backend.asarray(theta, dtype=dtype))
        ),
        log_surface_pressure=lnps_from_anomaly_coefficients(
            transform, transform.project(transform.forward(backend.asarray(
                np.log(ps) - LNPS_REFERENCE, dtype=dtype)))
        ),
        qv=zeros_spectral.copy(),
        **{
            name: zeros_grid.copy()
            for name in ("qc", "qr", "qi", "qs", "qg",
                         "nc", "nr", "ni", "ns", "ng")
        },
    )
    surface = _dry_surface(transform, temperature[-1])
    provenance = {
        "mode": "rest_atmosphere",
        "case": "rest-over-terrain",
        "sea_level_pressure_pa": REST_SEA_LEVEL_PRESSURE_PA,
        "terrain_max_m": float(np.max(surface_geopotential) / GRAVITY_M_S2),
        "terrain_min_m": float(np.min(surface_geopotential) / GRAVITY_M_S2),
    }
    return (
        ArwenGlobalState(atmosphere, surface, PhysicsState()),
        backend.asarray(surface_geopotential, dtype=dtype),
        provenance,
    )


# --------------------------------------------------------------------------
# Held and Suarez 1994.

#: Relaxation rates, per second: the free atmosphere, the boundary layer,
#: and the Rayleigh friction of the boundary layer.
HS_KA_S = 1.0 / (40.0 * 86400.0)
HS_KS_S = 1.0 / (4.0 * 86400.0)
HS_KF_S = 1.0 / (1.0 * 86400.0)
#: The sigma the boundary-layer terms switch on at.
HS_SIGMA_B = 0.7
#: The equilibrium profile's constants.
HS_T_EQUATOR_K = 315.0
HS_T_FLOOR_K = 200.0
HS_DELTA_T_Y_K = 60.0
HS_DELTA_THETA_Z_K = 10.0


def held_suarez_equilibrium(p_full, sin_lat, cos_lat):
    """``(T_eq, k_T, k_v)`` of the Held and Suarez forcing.

    ``p_full`` is in Pa and the sigma the boundary-layer terms read is
    ``p / p0`` with ``p0`` the reference pressure, which is the case's own
    definition (its surface pressure is 1000 hPa by construction).
    """
    xp = _array_module(p_full)
    sigma = p_full / REFERENCE_PRESSURE_PA
    log_sigma = xp.log(sigma)
    equilibrium = xp.maximum(
        HS_T_FLOOR_K,
        (
            HS_T_EQUATOR_K
            - HS_DELTA_T_Y_K * sin_lat ** 2
            - HS_DELTA_THETA_Z_K * log_sigma * cos_lat ** 2
        ) * sigma ** KAPPA,
    )
    boundary = xp.maximum(0.0, (sigma - HS_SIGMA_B) / (1.0 - HS_SIGMA_B))
    k_t = HS_KA_S + (HS_KS_S - HS_KA_S) * boundary * cos_lat ** 4
    k_v = HS_KF_S * boundary
    return equilibrium, k_t, k_v


def _array_module(value):
    module = type(value).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def apply_held_suarez(model, bundle: ArwenGlobalState, dt_s: float):
    """One Held and Suarez forcing step, applied to a spectral state.

    Both terms are integrated BACKWARD in time (``x <- (x + dt k x_eq) /
    (1 + dt k)``) rather than forward.  The relaxation is a stiff linear
    term whose fastest rate is one over four days, and a forward step is
    stable at every time step this model runs; the backward form is used
    because it is stable at EVERY time step, so the thousand-day climate
    the case measures cannot depend on the step through the forcing's own
    stability rather than through the dynamics the gate is about.
    """
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    state = bundle.atmosphere
    g = model.grid_state(state, only=("temperature", "p_full", "u", "v"))
    sin_lat = backend.asarray(
        transform.grid.sin_lat[:, None], dtype=dtype
    )[None]
    cos_lat = backend.asarray(
        transform.grid.cos_lat[:, None], dtype=dtype
    )[None]
    equilibrium, k_t, k_v = held_suarez_equilibrium(
        g["p_full"], sin_lat, cos_lat
    )
    dt = dtype(float(dt_s))
    temperature = (g["temperature"] + dt * k_t * equilibrium) / (1.0 + dt * k_t)
    exner = (g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
    theta = transform.project(transform.forward(temperature / exner))
    del temperature, exner, equilibrium, k_t
    damping = 1.0 / (1.0 + dt * k_v)
    vorticity, divergence = model.vector.vordiv_from_wind(
        g["u"] * damping, g["v"] * damping
    )
    del damping, k_v, g
    advanced = state.with_fields(
        (vorticity, divergence, theta, state.log_surface_pressure, state.qv)
    )
    return ArwenGlobalState(advanced, bundle.surface, bundle.physics_state)


__all__ = [
    "HS_KA_S",
    "HS_KF_S",
    "HS_KS_S",
    "HS_SIGMA_B",
    "JW_SURFACE_PRESSURE_PA",
    "JW_U0",
    "REST_LAPSE_K_M",
    "REST_SEA_LEVEL_PRESSURE_PA",
    "REST_SEA_LEVEL_TEMPERATURE_K",
    "REST_TROPOPAUSE_K",
    "apply_held_suarez",
    "baroclinic_wave_initial_state",
    "held_suarez_equilibrium",
    "jablonowski_williamson_fields",
    "jablonowski_williamson_perturbation",
    "rest_atmosphere_initial_state",
    "rest_atmosphere_surface_pressure",
    "rest_atmosphere_temperature",
    "terrain_on_model_grid",
]
