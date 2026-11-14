"""The mixed-layer cooling response of a prescribed sea surface.

Design: docs/ARWEN_GLOBAL_OCEAN_RESPONSE.md.  Off unless ``[ocean]
mixed_layer_response = true``; a run with it off builds nothing of this
module and writes what it wrote before, byte for byte.

A prescribed sea surface (``[ocean] mode = "hold"``, ``"anomaly"``,
``"climatology"`` or ``"series"``) cannot cool under days of strong dry
offshore wind, so its evaporation keeps the rate of the first hour for as
long as the wind blows (the T383 50-day runs reached 51-55 kg m-2 a day off
Tampico, in the northern Red Sea and in the Gulf of Tehuantepec, against
the 60 kg m-2 a day ceiling that stops a run).  The response carries one
prognostic departure D from the prescribed field P on every open-ocean
column the boundary writes, and the skin written there is
``max(T_f, P + D)``:

    rho_w c_w h dD/dt = F - F_ref  -  rho_w c_w h D / tau_r,   |D| <= D_max

* F, the net heat into open water over the interval the physics held
  (arwen_global.ocean_slab.open_water_flux: absorbed shortwave, absorbed
  longwave less emission at the skin the step saw, less sensible and latent
  heat; the reference suite's held net surface flux), positive down;
* F_ref, the column's normal flux: the running mean of F limited to
  :data:`FLUX_LIMITS_W_M2`, a plain mean of everything since the response
  began until ``reference_days`` have passed and an exponential mean over
  ``reference_days`` after, then held within :data:`REFERENCE_LIMITS_W_M2`.
  A column whose flux does not change is never moved (F = F_ref exactly
  from the first step), and over a long run the mean of F - F_ref is zero,
  so the response is a departure from the prescribed field, not a drift
  away from it;
* h a fixed mixed-layer depth (``mixed_layer_depth_m``), tau_r the
  relaxation back to the prescribed field (``mixed_layer_relaxation_days``),
  D_max the bound (``mixed_layer_max_departure_k``).

The two limits keep a run that starts (or branches) inside an extreme event
from taking the event for normal: an instantaneous flux enters the mean at
no more than 1000 W m-2 of loss (normal open-water nights reach 600-900),
and the reference is held within -600..+300 W m-2 (the observed range of
monthly net surface flux over open water: western boundary currents in
January to the equatorial cold tongue).

Only open water carries D: a column that is land, ice or an inland lake at
a step (or that the prescribed field does not define) holds D = 0 and
F_ref is left as it was.  D and F_ref live in the physics state, float64,
so a checkpoint carries them (``physics__<name>``) and a restart resumes
them bit for bit; together they are two (nlat, nlon) planes.
"""
from __future__ import annotations

import math

import numpy as np

from .surface_series import SEA_WATER_FREEZING_K

#: physics_state.arrays names, (nlat, nlon) float64.
RESPONSE_DEPARTURE = "ocean_response_sst_departure_k"
RESPONSE_REFERENCE_FLUX = "ocean_response_reference_flux_w_m2"
RESPONSE_ARRAYS = (RESPONSE_DEPARTURE, RESPONSE_REFERENCE_FLUX)
#: physics_state.metadata key: {"started_s", "last_update_s", "advances"}.
METADATA_KEY = "ocean_response"
#: The identity block's key (inside physics_metadata["surface_boundary"]).
IDENTITY_KEY = "mixed_layer_response"
#: The instantaneous flux the running mean takes, W m-2 (positive down).
FLUX_LIMITS_W_M2 = (-1000.0, 1200.0)
#: The range the reference is held in, W m-2.
REFERENCE_LIMITS_W_M2 = (-600.0, 300.0)
SECONDS_PER_DAY = 86400.0


def _f64(value, xp):
    return xp.asarray(value, dtype=xp.float64)


def reference_flux_step(reference, flux, *, dt_s, elapsed_s, reference_days, xp):
    """F_ref after one interval of ``dt_s`` ending ``elapsed_s`` after the
    response began: ``F_ref + (clip(F) - F_ref) min(1, dt / min(elapsed,
    tau_m))``, tau_m = ``reference_days`` days (float64)."""
    window = min(float(elapsed_s), float(reference_days) * SECONDS_PER_DAY)
    factor = 1.0 if window <= 0.0 else min(1.0, float(dt_s) / window)
    lo, hi = FLUX_LIMITS_W_M2
    reference = _f64(reference, xp)
    return reference + (xp.clip(_f64(flux, xp), lo, hi) - reference) * factor


def departure_step(departure, flux, reference, *, depth_m, relaxation_days,
                   max_departure_k, dt_s, xp):
    """D after one explicit interval: ``D + dt ((F - clip(F_ref)) / (rho_w
    c_w h) - D / tau_r)``, held within +-``max_departure_k`` (float64)."""
    from .ocean_slab import RHO_CP_SEAWATER_J_M3_K

    lo, hi = REFERENCE_LIMITS_W_M2
    anomaly = _f64(flux, xp) - xp.clip(_f64(reference, xp), lo, hi)
    departure = _f64(departure, xp)
    tau = float(relaxation_days) * SECONDS_PER_DAY
    new = departure + float(dt_s) * (
        anomaly / (RHO_CP_SEAWATER_J_M3_K * float(depth_m)) - departure / tau)
    bound = float(max_departure_k)
    return xp.clip(new, -bound, bound)


def _resident(xp, value):
    from .spill import resident, spilled

    return resident(xp, value) if spilled(value) else value


def _module_of(value):
    from .surface_boundary import SurfaceBoundary

    return SurfaceBoundary._xp(value)


def without_response(block):
    """An identity block with the response taken out (None stays None)."""
    if block is None:
        return None
    out = {key: value for key, value in block.items() if key != IDENTITY_KEY}
    if isinstance(out.get("parameters"), dict):
        out["parameters"] = {key: value for key, value in out["parameters"].items()
                             if key != IDENTITY_KEY}
    return out


class MixedLayerResponse:
    """A prescribed provider (ocean_rules.RuleProvider, any mode, or
    surface_series.SeriesProvider) with the response on top, as a record
    provider of the boundary service (arwen_global.surface_boundary).

    The service calls :meth:`advance` first in every ``apply``; the SST the
    service then writes is the base provider's (under ``hold``, the
    cold-start skin on the columns that start as open ocean) plus D.  Every
    ice record is the base provider's."""

    kind = "response"

    def __init__(self, base, options, grid, *, start_time, cold_surface: dict,
                 physics_mode: str = "arwen-native"):
        from .statics import lake_columns, water_columns
        from .surface_series import _utc

        self.base = base
        self.options = options
        self.grid = grid
        self.start = _utc(start_time)
        self.physics_mode = str(physics_mode)
        self.hold = getattr(base, "kind", None) == "rule" and base.mode == "hold"
        self.depth_m = float(options.mixed_layer_depth_m)
        self.relaxation_days = float(options.mixed_layer_relaxation_days)
        self.max_departure_k = float(options.mixed_layer_max_departure_k)
        self.reference_days = float(options.mixed_layer_reference_days)
        f = {name: np.asarray(value) for name, value in cold_surface.items()}
        land = f["land_fraction"].astype(np.float32)
        lake = f["lake_fraction"].astype(np.float32)
        ice = f["sea_ice_fraction"].astype(np.float32)
        #: Under hold: the columns that start as open ocean, and their skin.
        self.open0 = (np.asarray(water_columns(land, np, sea_ice_fraction=ice))
                      & ~np.asarray(lake_columns(land, lake, np, sea_ice_fraction=ice)))
        self.held_skin = np.asarray(f["temperature_k"], dtype=np.float64)
        if self.hold:
            self.defined = self.open0
        else:
            defined = getattr(base, "defined", None)
            self.defined = (np.ones(land.shape, dtype=bool) if defined is None
                            else np.asarray(defined, dtype=bool))
        #: Set by the runner when a branch switches the response on: the
        #: parent carries no response arrays and the response starts there.
        self.start_at_branch = False
        self._device: dict = {}
        self._state = None
        self._last_t = None
        self.stats: dict[str, object] = {"advances_this_segment": 0,
                                         "started_at_branch": False}

    # -- helpers -------------------------------------------------------------

    @property
    def reference_suite(self) -> bool:
        return self.physics_mode == "reference"

    def _on(self, xp, name, value):
        key = (xp.__name__, name)
        if key not in self._device:
            self._device[key] = xp.asarray(value)
        return self._device[key]

    def _columns(self, surface, xp):
        """The open-ocean columns the boundary writes at this step: water,
        not frozen, not an inland lake, defined by the base provider."""
        from .statics import lake_columns, water_columns

        def get(name):
            return _resident(xp, getattr(surface, name))

        land, ice = get("land_fraction"), get("sea_ice_fraction")
        water = water_columns(land, xp, sea_ice_fraction=ice)
        lake = lake_columns(land, get("lake_fraction"), xp, sea_ice_fraction=ice)
        return water & ~lake & self._on(xp, "defined", self.defined)

    def _flux(self, state, xp):
        from .ocean_coupler import HELD_NET_FLUX
        from .ocean_slab import open_water_flux

        arrays = state.physics_state.arrays
        if HELD_NET_FLUX in arrays:
            return _f64(_resident(xp, arrays[HELD_NET_FLUX]), xp)

        def plane(name):
            return _f64(_resident(xp, arrays[name]), xp)

        def surface(name):
            return _f64(_resident(xp, getattr(state.surface, name)), xp)

        return _f64(open_water_flux(
            swdown=plane("swdown"), albedo=surface("albedo"), glw=plane("glw"),
            emissivity=surface("emissivity"), skin_k=surface("temperature_k"),
            hfx=plane("hfx"), qfx=plane("qfx"), xp=xp), xp)

    # -- the state -----------------------------------------------------------

    def initialize(self, state) -> None:
        """D = 0 and F_ref = 0 in ``state``'s physics namespace, the response
        beginning at ``state.time_s`` (the cold start, or a branch)."""
        from .ocean_coupler import HELD_NET_FLUX, _put_array

        xp = _module_of(state.surface.temperature_k)
        shape = tuple(np.shape(state.surface.land_fraction))
        for name in RESPONSE_ARRAYS:
            _put_array(state.physics_state, name, xp.zeros(shape, dtype=xp.float64))
        if self.reference_suite and HELD_NET_FLUX not in state.physics_state.arrays:
            _put_array(state.physics_state, HELD_NET_FLUX, xp.zeros(shape, dtype=xp.float64))
        t = float(state.time_s)
        state.physics_state.metadata[METADATA_KEY] = {
            "started_s": t, "last_update_s": t, "advances": 0}

    def check_state(self, state, label: str) -> None:
        """Refuse a restart that carries no response, unless the runner
        admitted it as a branch that switches the response on (which then
        starts here with D = 0)."""
        from .ocean_coupler import HELD_NET_FLUX

        arrays = state.physics_state.arrays
        missing = [name for name in RESPONSE_ARRAYS if name not in arrays]
        if self.reference_suite and HELD_NET_FLUX not in arrays:
            missing.append(HELD_NET_FLUX)
        if not missing and METADATA_KEY in state.physics_state.metadata:
            return
        if self.start_at_branch and len(missing) >= len(RESPONSE_ARRAYS) \
                and METADATA_KEY not in state.physics_state.metadata:
            self.initialize(state)
            self.stats["started_at_branch"] = True
            return
        raise ValueError(
            f"{label} carries no mixed-layer response ({', '.join(missing) or METADATA_KEY}"
            " missing); a restart must continue the boundary it was written with "
            "(--branch-from-other-config switches the response on at a checkpoint "
            "written without it)")

    # -- the step ------------------------------------------------------------

    def advance(self, state, *, model=None) -> None:
        """Step D and F_ref from the last update to the state's time with
        the fluxes the physics held over that interval."""
        from .ocean_coupler import _put_array

        self._state = state
        meta = state.physics_state.metadata.get(METADATA_KEY)
        if meta is None:
            raise ValueError("the mixed-layer response found no ocean_response record in "
                             "the physics state; the run was not started with it")
        t = float(state.time_s)
        dt = t - float(meta["last_update_s"])
        if dt <= 0.0:
            return
        arrays = state.physics_state.arrays
        xp = _module_of(state.surface.temperature_k)
        columns = self._columns(state.surface, xp)
        flux = self._flux(state, xp)
        reference = _f64(_resident(xp, arrays[RESPONSE_REFERENCE_FLUX]), xp)
        departure = _f64(_resident(xp, arrays[RESPONSE_DEPARTURE]), xp)
        stepped_reference = reference_flux_step(
            reference, flux, dt_s=dt, elapsed_s=t - float(meta["started_s"]),
            reference_days=self.reference_days, xp=xp)
        new_reference = xp.where(columns, stepped_reference, reference)
        new_departure = xp.where(columns, departure_step(
            departure, flux, new_reference, depth_m=self.depth_m,
            relaxation_days=self.relaxation_days, max_departure_k=self.max_departure_k,
            dt_s=dt, xp=xp), 0.0)
        _put_array(state.physics_state, RESPONSE_REFERENCE_FLUX, new_reference)
        _put_array(state.physics_state, RESPONSE_DEPARTURE, new_departure)
        state.physics_state.metadata[METADATA_KEY] = {
            **meta, "last_update_s": t, "advances": int(meta.get("advances", 0)) + 1}
        self.stats["advances_this_segment"] += 1

    # -- the provider contract -----------------------------------------------

    def record_index(self, t: float) -> int:
        return self.base.record_index(t)

    def sst_at(self, t: float, xp, dtype):
        """``(max(T_f, P + D) as dtype where D != 0, P elsewhere; the columns
        the base defines)``."""
        self._last_t = float(t)
        if self.hold:
            self.base.sst_at(t, xp, dtype)
            value = self._on(xp, "held_skin", self.held_skin)
            defined = self._on(xp, "defined", self.defined)
        else:
            value, defined = self.base.sst_at(t, xp, dtype)
            if value is None:
                return None, None
        if self._state is None:
            return xp.asarray(value).astype(dtype, copy=False), defined
        departure = _f64(_resident(xp, self._state.physics_state.arrays[RESPONSE_DEPARTURE]),
                         xp)
        base = _f64(value, xp)
        moved = xp.maximum(xp.float64(SEA_WATER_FREEZING_K), base + departure)
        return xp.where(departure != 0.0, moved, base).astype(dtype, copy=False), defined

    def ice_record(self, k: int, xp, surface, get):
        return self.base.ice_record(k, xp, surface, get)

    def record_sst_host(self, k: int):
        return self.base.record_sst_host(k)

    def record_valid(self, k: int):
        return self.base.record_valid(k)

    def last_info(self, t: float) -> dict:
        out = dict(self.base.last_info(t))
        meta = ({} if self._state is None
                else self._state.physics_state.metadata.get(METADATA_KEY, {}))
        out["response_advances"] = meta.get("advances")
        return out

    # -- identity, receipt and rows ------------------------------------------

    def parameters(self) -> dict:
        return {
            "depth_m": self.depth_m, "relaxation_days": self.relaxation_days,
            "max_departure_k": self.max_departure_k, "reference_days": self.reference_days,
            "flux_limits_w_m2": list(FLUX_LIMITS_W_M2),
            "reference_limits_w_m2": list(REFERENCE_LIMITS_W_M2),
        }

    def identity_without_response(self):
        """The restart identity the same run has with the response off."""
        if self.hold or not hasattr(self.base, "identity_block"):
            return None
        return without_response(self.base.identity_block())

    def identity_block(self) -> dict:
        if self.hold or not hasattr(self.base, "identity_block"):
            block = {"ocean_mode": self.options.mode, "sea_ice_mode": self.options.ice_mode,
                     "start_utc": self.start.isoformat().replace("+00:00", "Z")}
        else:
            block = dict(self.base.identity_block())
        block[IDENTITY_KEY] = self.parameters()
        return block

    def receipt(self) -> dict:
        out = dict(self.base.receipt())
        out[IDENTITY_KEY] = {
            **self.parameters(),
            "sst": "max(T_f, P + D) on open ocean, P the prescribed field, "
                   "rho_w c_w h dD/dt = F - F_ref - rho_w c_w h D / tau_r, |D| <= D_max",
            "flux_source": ("the reference suite's held net surface flux"
                            if self.reference_suite else
                            "the native suite's held radiation and surface-layer planes"),
            **self.stats,
        }
        return out

    def diagnostics(self, state) -> dict:
        """The base provider's rows (none under hold) and the response's:
        area means on the open-ocean columns of D and F_ref, D's extremes
        and the share of that area cooled by more than 1 K."""
        from .ocean_rules import area_mean
        from .surface_boundary import _host

        rows = ({"boundary_provider": "rule:hold/hold"} if self.hold
                or not hasattr(self.base, "diagnostics") else dict(self.base.diagnostics(state)))
        arrays = state.physics_state.arrays
        if RESPONSE_DEPARTURE not in arrays:
            return rows
        departure = _host(arrays[RESPONSE_DEPARTURE]).astype(np.float64)
        reference = _host(arrays[RESPONSE_REFERENCE_FLUX]).astype(np.float64)
        surface = state.surface
        columns = np.asarray(_host(self._columns(surface, _module_of(
            surface.temperature_k))), dtype=bool)
        mean = area_mean(self.grid, departure, columns)
        square = area_mean(self.grid, departure * departure, columns)
        cooled = area_mean(self.grid, (departure < -1.0).astype(np.float64), columns)
        rows.update({
            "ocean_response_mean_departure_k": mean,
            "ocean_response_rms_departure_k": None if square is None else math.sqrt(square),
            "ocean_response_min_departure_k": (float(departure[columns].min())
                                               if columns.any() else None),
            "ocean_response_max_departure_k": (float(departure[columns].max())
                                               if columns.any() else None),
            "ocean_response_area_cooled_1k_fraction": cooled,
            "ocean_response_mean_reference_flux_w_m2": area_mean(self.grid, reference,
                                                                 columns),
        })
        return rows


__all__ = [
    "FLUX_LIMITS_W_M2", "IDENTITY_KEY", "METADATA_KEY", "MixedLayerResponse",
    "REFERENCE_LIMITS_W_M2", "RESPONSE_ARRAYS", "RESPONSE_DEPARTURE",
    "RESPONSE_REFERENCE_FLUX", "departure_step", "reference_flux_step", "without_response",
]
