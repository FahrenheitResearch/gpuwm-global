"""The dynamic ocean in the run: ``[ocean] mode = "dynamic"`` (design 6).

:class:`DynamicOceanProvider` is the slab's provider
(arwen_global.ocean_coupler.SlabProvider) with the dynamic ocean on the
atmosphere columns it OWNS (ocean columns at least half covered by wet
ocean cells, arwen_global.ocean_dyn.remap): there the mixed-layer
temperature is the ocean's top-cell temperature, remapped once an hour, and
the sea ice takes its ocean-side heat from the ocean.  On every other ocean
column (north of 65N, closed seas, the coast cells the ocean grid does not
cover) the slab steps exactly as under ``mode = "slab"``.

Every atmosphere step (``advance``, before the model's step, the slab's
one-step lag):

* the slab advances everywhere; on the owned columns its temperature is
  then replaced by the held ocean SST and its ice is stepped again from the
  pre-step ice with the ocean's terms (the lead flux and the ocean's basal
  heat A F_bot melt or grow it; the frazil the ocean made in the last
  ocean step is its freeze deficit);
* the fluxes the physics held over the step are added, times dt, into
  float64 accumulators on the atmosphere grid: open-water non-shortwave and
  shortwave heat (the slab's F_open, split), -A F_bot and the energy a
  complete melt hands back on iced columns, the air stress on the open part
  (rho_a u*^2 along the 10 m wind, rho_a = 1.2 kg m-3), the ice cover, the
  fresh water ((1 - A) P - E, P from the rain accumulators, E from qfx or
  the lead tile's), and the ice volume grown (for the brine salt);
* at each whole hour since the start the means are remapped to the ocean
  grid (conservatively: the ocean receives exactly what the owned columns
  exchanged), one ocean step runs, and the new SST, the basal-heat
  potential and the frazil are remapped back.

Everything lives in the physics state (``ocean_dyn_*`` arrays and the
``ocean_dynamic`` metadata), so a checkpoint carries it and a restart in the
middle of an hour continues the partial sums bit for bit.
"""
from __future__ import annotations

from datetime import timedelta
import math

import numpy as np

from ..ocean_coupler import (ICE_FRACTION, ICE_VOLUME, SLAB_TEMPERATURE, SlabProvider,
                             _module_of, _put_array, _resident, _host, LEAD_ALBEDO, ENERGY_SOURCES,
                             ENERGY_STORED)
from .grid import OceanGrid
from .init import OceanInit, Targets
from .model import BOOKS, FREEZING_C, KELVIN, OceanModel, OceanParams, STATE_DTYPES
from .remap import Remap

PREFIX = "ocean_dyn_"
METADATA_KEY = "ocean_dynamic"
ACCUMULATORS = ("acc_heat_ns", "acc_heat_sw", "acc_taux", "acc_tauy", "acc_fw", "acc_ice",
                "acc_ice_growth")
HELD = ("sst_atm", "fbot_atm", "frazil_atm", "prev_precip", "prev_outflow")
#: The land's booked water exits (native_runtime: runoff the land step
#: debits from its stores), routed to the ocean.
OUTFLOW = "water_outflow_kg_m2"
#: The tape's ocean planes on the atmosphere grid (float32), refreshed hourly;
#: the ocean_dyn_ prefix keeps their dtype through a checkpoint and keeps
#: them out of the physics call.
TAPE_PREFIX = PREFIX + "tape_"
TAPE_NAMES = ("ssh_m", "u_m_s", "v_m_s", "hbl_m", "d20_m", "ohc300_j_m2")
AIR_DENSITY = 1.2
COUPLING_S = 3600.0


def state_names():
    return (tuple(PREFIX + n for n in STATE_DTYPES) + tuple(PREFIX + n for n in ACCUMULATORS + HELD)
            + tuple(TAPE_PREFIX + n for n in TAPE_NAMES))


def is_dynamic_array(name: str) -> bool:
    return name.removeprefix("physics__").startswith(PREFIX)


def params_from_options(options) -> OceanParams:
    from dataclasses import replace

    p = OceanParams()
    return replace(p,
                   sss_piston_m_per_day=options.dynamic_sss_restoring_m_per_day,
                   sst_restoring_days=options.dynamic_sst_restoring_days,
                   gm_kappa=float(options.dynamic_gm_kappa_m2_s))


class DynamicOceanProvider(SlabProvider):
    kind = "dynamic"

    def __init__(self, options, climatology, qflux, grid, *, start_time, cold_surface, **kw):
        super().__init__(options, climatology, qflux, grid, start_time=start_time,
                         cold_surface=cold_surface, **kw)
        self.ogrid = OceanGrid.load(options.dynamic_grid)
        if self.ogrid.sha256 != options.dynamic_grid_sha256:
            raise ValueError(f"the ocean grid {options.dynamic_grid} changed since the config "
                             "was read")
        self.oinit = OceanInit(options.dynamic_initial)
        if self.oinit.sha256 != options.dynamic_initial_sha256:
            raise ValueError(f"the ocean initial file {options.dynamic_initial} changed since the "
                             "config was read")
        if self.oinit.provenance.get("grid_sha256") not in (None, self.ogrid.sha256):
            raise ValueError("the ocean initial file was built on another ocean grid")
        self.oparams = params_from_options(options)
        self.remap = Remap(self.ogrid, atm_lat_deg=self.latitude_deg,
                           atm_weights=np.asarray(grid.quadrature_weights, dtype=np.float64),
                           atm_nlon=int(self.longitude_deg.size), atm_ocean=self.ocean,
                           radius_m=float(grid.radius_m))
        self.owned = self.remap.owned
        self._route = None
        self._model = {}
        self.stats.update({"owned_columns": int(self.owned.sum()), "ocean_steps_this_segment": 0,
                           "ocean_seconds_this_segment": 0.0, **self.remap.stats()})

    # -- the ocean model on the state's module --------------------------------
    def model(self, xp):
        key = xp.__name__
        if key not in self._model:
            targets = Targets(self.oinit, self.start, xp)
            self._model[key] = OceanModel(self.ogrid, self.oparams, xp,
                                          targets=targets.as_model_targets())
        return self._model[key]

    def _arr(self, state, name, xp):
        return _resident(xp, state.physics_state.arrays[PREFIX + name])

    def ocean_state(self, state, xp) -> dict:
        return {n: self._arr(state, n, xp) for n in STATE_DTYPES}

    # -- cold start ---------------------------------------------------------
    def initial_arrays(self, cold_surface: dict) -> dict:
        out = super().initial_arrays(cold_surface)
        a = self.oinit.arrays
        m = self.model(np)
        theta = np.asarray(a["theta"], dtype=np.float64).copy()
        # SST match: the analysis skin on owned open columns, remapped, minus
        # the ocean's top cell, spread over the mixed layer.
        skin = np.asarray(cold_surface["temperature_k"], dtype=np.float64)
        ice = np.asarray(cold_surface["sea_ice_fraction"], dtype=np.float64)
        open_owned = self.owned & ~(np.nan_to_num(ice) >= 0.5)
        num, _ = self.remap.to_ocean(np.where(open_owned, skin - KELVIN, 0.0), np)
        den, _ = self.remap.to_ocean(np.where(open_owned, 1.0, 0.0), np)
        g = self.ogrid
        ok = g.wet & (den > 0.5)
        target = np.where(ok, num / np.where(ok, den, 1.0), theta[0])
        delta = np.where(ok, target - theta[0], 0.0)
        mld = _mixed_layer_depth(theta, np.asarray(a["salt"]), g)
        z = g.z_t_nom[:, None, None]
        within = (z <= np.maximum(mld, 10.0)[None]) & g.tmask
        theta = np.where(within, theta + delta[None], theta)
        st = m.new_state(theta, a["salt"], a["u"], a["v"], a["eta"])
        for name, value in st.items():
            out[PREFIX + name] = np.asarray(value)
        shape = cold_surface["land_fraction"].shape
        for name in ACCUMULATORS + HELD:
            out[PREFIX + name] = np.zeros(shape, dtype=np.float64)
        sst = self.remap.to_atmosphere(theta[0] + KELVIN, np)
        out[PREFIX + "sst_atm"] = np.where(self.owned, sst, 0.0)
        out[PREFIX + "fbot_atm"] = np.zeros(shape)
        for name, value in self._tape_planes(st, np).items():
            out[name] = np.asarray(value)
        self._sst_match = {"columns": int(ok.sum()), "mean_delta_k": float(
            (delta * g.area * ok).sum() / max((g.area * ok).sum(), 1.0)),
            "rms_delta_k": float(np.sqrt((delta ** 2 * g.area * ok).sum() / max((g.area * ok).sum(), 1.0)))}
        return out

    def initialize(self, state) -> None:
        super().initialize(state)
        xp = _module_of(state.surface.temperature_k)
        arrays = state.physics_state.arrays
        _put_array(state.physics_state, PREFIX + "prev_precip", self._precip_total(state, xp))
        if OUTFLOW in arrays:
            _put_array(state.physics_state, PREFIX + "prev_outflow",
                       xp.asarray(_resident(xp, arrays[OUTFLOW]), dtype=xp.float64))
        state.physics_state.metadata[METADATA_KEY] = {
            "last_couple_s": float(state.time_s), "acc_s": 0.0, "couplings": 0,
            "sst_match": getattr(self, "_sst_match", None)}

    def check_state(self, state, label: str) -> None:
        super().check_state(state, label)
        missing = [n for n in state_names() if n not in state.physics_state.arrays]
        if missing or METADATA_KEY not in state.physics_state.metadata:
            raise ValueError(f"{label} carries no dynamic-ocean state ({', '.join(missing[:4]) or METADATA_KEY} "
                             "missing); a restart must continue the ocean it was written with")

    # -- per step -------------------------------------------------------------
    def _precip_total(self, state, xp):
        arrays = state.physics_state.arrays
        total = xp.zeros(self.ocean.shape, dtype=xp.float64)
        for name in ("rainnc", "rainc"):
            if name in arrays:
                total = total + xp.asarray(_resident(xp, arrays[name]), dtype=xp.float64)
        return total

    def advance(self, state, *, model=None) -> None:
        # Bind the state before any early return, as the slab's advance does:
        # the boundary service asks for the ice record right after this, and
        # an unbound provider skipped record zero at a cold start, so a
        # restart inside the first record re-applied the ice the straight run
        # never had (a coupled restart that was not bitwise).
        self._state = state
        meta_slab = state.physics_state.metadata.get("ocean_slab")
        if meta_slab is None:
            raise ValueError("the dynamic ocean found no ocean_slab record in the physics state")
        t = float(state.time_s)
        t0 = float(meta_slab["last_update_s"])
        dt = t - t0
        if dt <= 0.0:
            return
        xp = _module_of(state.surface.temperature_k)
        arrays = state.physics_state.arrays
        physics = state.physics_state

        def plane(name):
            return xp.asarray(_resident(xp, arrays[name]), dtype=xp.float64)

        owned = self._on(xp, "owned", self.owned)
        ocean = self._on(xp, "ocean", self.ocean)
        frac0 = plane(ICE_FRACTION)
        vol0 = plane(ICE_VOLUME)
        frozen = self._frozen(state, xp)
        f_open = xp.where(ocean, self._open_water_flux(state, xp, frozen), 0.0)
        basal, melt = self._ice_column_fluxes(state, xp, frozen)
        # 1. the slab everywhere (its own arithmetic, unchanged)
        super().advance(state, model=model)
        # 2. the owned columns: ice from the pre-step ice with the ocean's terms
        from ..sea_ice_thermo import ice_step

        iced = owned & (vol0 > 0.0)
        zero = xp.zeros_like(frac0)
        basal = zero if basal is None else xp.where(iced, basal, 0.0)
        melt = zero if melt is None else xp.where(iced, melt, 0.0)
        fbot = plane(PREFIX + "fbot_atm")
        frazil = plane(PREFIX + "frazil_atm")
        f_bot_cell = xp.where(iced, frac0 * fbot, 0.0)
        ice = ice_step(fraction=frac0, volume_m=vol0,
                       ocean_flux_w_m2=xp.where(iced, f_open + f_bot_cell, 0.0),
                       basal_conduction_w_m2=basal, top_melt_w_m2=melt,
                       freeze_deficit_j_m2=xp.where(owned, frazil, 0.0), dt_s=dt,
                       new_ice_thickness_m=self.new_ice_thickness_m, xp=xp)
        new_frac = xp.where(owned, ice.fraction, plane(ICE_FRACTION))
        new_vol = xp.where(owned, ice.volume_m, plane(ICE_VOLUME))
        _put_array(physics, ICE_FRACTION, new_frac)
        _put_array(physics, ICE_VOLUME, new_vol)
        _put_array(physics, SLAB_TEMPERATURE,
                   xp.where(owned, plane(PREFIX + "sst_atm"), plane(SLAB_TEMPERATURE)))
        # the slab's own books do not describe the owned columns
        _put_array(physics, ENERGY_SOURCES, xp.where(owned, 0.0, plane(ENERGY_SOURCES)))
        _put_array(physics, ENERGY_STORED, xp.where(owned, 0.0, plane(ENERGY_STORED)))
        _put_array(physics, PREFIX + "frazil_atm", xp.zeros_like(frazil))
        # 3. accumulate
        surface = state.surface
        albedo = xp.asarray(_resident(xp, surface.albedo), dtype=xp.float64)
        swdown = plane("swdown") if "swdown" in arrays else zero
        cover = xp.clip(xp.asarray(_resident(xp, surface.sea_ice_fraction), dtype=xp.float64), 0.0, 1.0)
        if frozen is None:
            sw_open = swdown * (1.0 - albedo)
        else:
            sw_open = xp.where(frozen, (1.0 - cover) * swdown * (1.0 - LEAD_ALBEDO),
                               swdown * (1.0 - albedo))
        open_ = owned & ~iced
        heat_ns = xp.where(open_, f_open - sw_open, 0.0) - f_bot_cell
        heat_ns = dt * heat_ns + xp.where(owned, ice.mixed_layer_heat_j_m2, 0.0)
        heat_sw = dt * xp.where(open_, sw_open, 0.0)
        ust = plane("ust") if "ust" in arrays else zero
        u10 = plane("u10") if "u10" in arrays else zero
        v10 = plane("v10") if "v10" in arrays else zero
        speed = xp.maximum(xp.sqrt(u10 * u10 + v10 * v10), 1.0e-3)
        stress = AIR_DENSITY * ust * ust
        a_cov = xp.where(owned, new_frac, 0.0)
        taux = dt * (1.0 - a_cov) * stress * u10 / speed
        tauy = dt * (1.0 - a_cov) * stress * v10 / speed
        precip_now = self._precip_total(state, xp)
        precip = xp.maximum(precip_now - plane(PREFIX + "prev_precip"), 0.0)
        qfx = plane("qfx") if "qfx" in arrays else zero
        if frozen is not None and "lead_qfx" in arrays:
            evap = xp.where(frozen, (1.0 - cover) * plane("lead_qfx"), qfx)
        else:
            evap = qfx
        fw = (1.0 - a_cov) * precip - dt * evap
        if OUTFLOW in arrays:
            # rivers: the land's water exits this step, each column's mass put on
            # its nearest owned ocean column at once (no delay; stage 1)
            now = plane(OUTFLOW)
            fw = fw + self._routed(now - plane(PREFIX + "prev_outflow"), xp)
            _put_array(physics, PREFIX + "prev_outflow", now)
        growth = xp.where(owned, new_vol - vol0, 0.0)
        for name, inc in (("acc_heat_ns", heat_ns), ("acc_heat_sw", heat_sw), ("acc_taux", taux),
                          ("acc_tauy", tauy), ("acc_fw", fw), ("acc_ice", dt * a_cov),
                          ("acc_ice_growth", growth)):
            _put_array(physics, PREFIX + name, plane(PREFIX + name) + xp.where(owned, inc, 0.0))
        _put_array(physics, PREFIX + "prev_precip", precip_now)
        meta = dict(state.physics_state.metadata[METADATA_KEY])
        meta["acc_s"] = float(meta["acc_s"]) + dt
        state.physics_state.metadata[METADATA_KEY] = meta
        # 4. the coupling hour
        last = float(meta["last_couple_s"])
        every = float(self.oparams.dt)  # the ocean step is the coupling interval
        if math.floor((t + 1.0e-6) / every) > math.floor((last + 1.0e-6) / every):
            self._couple(state, xp, t)

    def _couple(self, state, xp, t: float) -> None:
        import time as _time

        started = _time.perf_counter()
        physics = state.physics_state
        meta = dict(physics.metadata[METADATA_KEY])
        acc_s = float(meta["acc_s"])
        m = self.model(xp)
        rm = self.remap

        def mean_on_ocean(name):
            value = xp.asarray(_resident(xp, physics.arrays[PREFIX + name]), dtype=xp.float64)
            g, res = rm.to_ocean(value / acc_s, xp)
            return g, res

        q_ns, r_ns = mean_on_ocean("acc_heat_ns")
        q_sw, r_sw = mean_on_ocean("acc_heat_sw")
        taux, _ = mean_on_ocean("acc_taux")
        tauy, _ = mean_on_ocean("acc_tauy")
        fw, _ = mean_on_ocean("acc_fw")
        ice, _ = mean_on_ocean("acc_ice")
        growth, _ = rm.to_ocean(xp.asarray(_resident(xp, physics.arrays[PREFIX + "acc_ice_growth"]),
                                           dtype=xp.float64), xp)
        if abs(acc_s - m.p.dt) > 1.0e-6:
            raise ValueError(f"the dynamic ocean accumulated {acc_s:g} s for a {m.p.dt:g} s step")
        fc = {"q_ns": q_ns, "q_sw": q_sw, "taux": taux, "tauy": tauy, "fw": fw,
              "ice": xp.clip(ice, 0.0, 1.0), "ice_growth_m": growth, "time_s": t - 0.5 * acc_s}
        ost = self.ocean_state(state, xp)
        books_res = float(r_ns + r_sw) * acc_s
        new, diag = m.step(ost, fc)
        b = new["books"]
        b[BOOKS.index("remap_residual_heat_j")] = b[BOOKS.index("remap_residual_heat_j")] + books_res
        for name, value in new.items():
            _put_array(physics, PREFIX + name, value)
        owned = self._on(xp, "owned", self.owned)
        sst = rm.to_atmosphere(new["theta"][0] + KELVIN, xp)
        _put_array(physics, PREFIX + "sst_atm", xp.where(owned, sst, 0.0))
        fbot = rm.to_atmosphere(diag["f_bot_w_m2"], xp)
        _put_array(physics, PREFIX + "fbot_atm", xp.where(owned, fbot, 0.0))
        cov = self._on(xp, "coverage", rm.coverage)
        frazil = rm.to_atmosphere(diag["frazil_j_m2"], xp) * cov
        _put_array(physics, PREFIX + "frazil_atm", xp.where(owned, frazil, 0.0))
        for name, value in self._tape_planes(new, xp).items():
            _put_array(physics, name, value)
        # the slab temperature on owned columns follows at once
        _put_array(physics, SLAB_TEMPERATURE, xp.where(
            owned, physics.arrays[PREFIX + "sst_atm"], _resident(xp, physics.arrays[SLAB_TEMPERATURE])))
        for name in ACCUMULATORS:
            _put_array(physics, PREFIX + name, xp.zeros(self.ocean.shape, dtype=xp.float64))
        meta.update(last_couple_s=float(t), acc_s=0.0, couplings=int(meta["couplings"]) + 1)
        physics.metadata[METADATA_KEY] = meta
        if xp is not np:
            xp.cuda.Device().synchronize()
        self.stats["ocean_steps_this_segment"] += 1
        self.stats["ocean_seconds_this_segment"] += _time.perf_counter() - started

    def _routed(self, increment, xp):
        """A plane of water (kg m-2 of each column) moved, mass for mass, to
        each column's nearest owned column (an owned column keeps its own).
        Summed on the host with numpy's bincount, whose order is fixed, so a
        rerun and a restart are bitwise."""
        if self._route is None:
            from scipy.spatial import cKDTree

            nlat, nlon = self.ocean.shape
            la = np.deg2rad(np.repeat(self.latitude_deg, nlon))
            lo = np.deg2rad(np.tile(self.longitude_deg, nlat))
            xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
            owned = np.nonzero(self.owned.ravel())[0]
            _, near = cKDTree(xyz[owned]).query(xyz)
            route = owned[near]
            route[owned] = owned
            self._route = route
            self._area_flat = self.remap.area_a.ravel()
        inc = np.nan_to_num(_host(increment).astype(np.float64).ravel())
        mass = np.bincount(self._route, weights=inc * self._area_flat,
                           minlength=self._area_flat.size)
        return xp.asarray((mass / self._area_flat).reshape(self.ocean.shape))

    def _tape_planes(self, ost: dict, xp) -> dict:
        """The tape's ocean planes on the atmosphere grid (float32, 0 off the
        owned columns), from an ocean state: SSH, the top cell's current at
        the T points, the KPP depth, D20 and the 0-300 m heat content."""
        g = self.model(xp).g
        th = xp.asarray(ost["theta"])
        u = xp.asarray(ost["u"], dtype=xp.float64)
        v = xp.asarray(ost["v"], dtype=xp.float64)
        uT = 0.5 * (u[0] + xp.roll(u[0], 1, axis=-1))
        vT = v[0].copy()
        vT[1:] = 0.5 * (v[0][1:] + v[0][:-1])
        z = g["z_t_nom"]
        tm = g["tmask"]
        thm = xp.where(tm, th, -99.0)
        d20 = xp.zeros(th.shape[1:])
        found = xp.zeros(th.shape[1:], dtype=bool)
        for k in range(th.shape[0] - 1):
            cross = (thm[k] >= 20.0) & (thm[k + 1] < 20.0) & tm[k + 1] & ~found
            frac = (thm[k] - 20.0) / xp.where(cross, thm[k] - thm[k + 1], 1.0)
            d20 = xp.where(cross, z[k] + frac * (z[k + 1] - z[k]), d20)
            found = found | cross
        s = 1.0 + xp.asarray(ost["eta"]) / xp.where(g["wet"], g["H"], 1.0)
        h = g["dzt"] * s[None]
        top = xp.cumsum(h, axis=0) <= 300.0 + 1.0e-6
        from .grid import CP0, RHO0

        ohc = RHO0 * CP0 * (th * h * top).sum(axis=0)
        owned = self._on(xp, "owned", self.owned)
        out = {}
        for name, plane in (("ssh_m", ost["eta"]), ("u_m_s", uT), ("v_m_s", vT),
                            ("hbl_m", ost["hbl"]), ("d20_m", d20), ("ohc300_j_m2", ohc)):
            value = self.remap.to_atmosphere(xp.asarray(plane, dtype=xp.float64), xp)
            out[TAPE_PREFIX + name] = xp.where(owned, value, 0.0).astype(xp.float32)
        return out

    # -- identity, receipt, rows ----------------------------------------------
    def identity_block(self) -> dict:
        block = super().identity_block()
        block["ocean_mode"] = "dynamic"
        block["ocean_grid_sha256"] = self.ogrid.sha256
        block["ocean_initial_sha256"] = self.oinit.sha256
        block["ocean_parameters"] = self.oparams.identity()
        # The two files are identified by their SHA-256 (above and in the
        # parameters); their paths are not identity, or a run could not
        # continue on another host (as the slab drops its file paths).
        for name in ("dynamic_grid", "dynamic_initial"):
            block["parameters"].pop(name, None)
        return block

    def receipt(self) -> dict:
        out = super().receipt()
        out["provider"] = "dynamic"
        out["ocean_grid"] = self.ogrid.receipt()
        out["ocean_initial"] = {"sha256": self.oinit.sha256, "valid_utc": self.oinit.valid_utc,
                                "provenance": self.oinit.provenance}
        out["ocean_parameters"] = self.oparams.identity()
        out["ocean_coupling"] = {k: self.stats.get(k) for k in (
            "owned_columns", "partly_covered_unowned", "ocean_cells_filled_from_nearest",
            "ocean_steps_this_segment", "ocean_seconds_this_segment")}
        out["sst"] = ("the dynamic ocean's top cell on owned columns, remapped hourly; the "
                      "slab's mixed layer elsewhere")
        return out

    def diagnostics(self, state) -> dict:
        from .diagnostics import rows

        out = super().diagnostics(state)
        ost = {n: _host(state.physics_state.arrays[PREFIX + n]) for n in STATE_DTYPES}
        out.update(rows(self.ogrid, ost))
        b = ost["books"]
        out["ocean_remap_residual_heat_j"] = float(b[BOOKS.index("remap_residual_heat_j")])
        meta = state.physics_state.metadata.get(METADATA_KEY, {})
        out["ocean_couplings"] = int(meta.get("couplings", 0))
        return out


def _mixed_layer_depth(theta, salt, grid, drho=0.03):
    from .eos import density

    z = grid.z_t_nom[:, None, None]
    rho = density(theta, salt, 0.0 * z)
    ref = rho[0]
    deeper = (rho - ref[None] > drho) & grid.tmask
    k = np.argmax(deeper, axis=0)
    has = deeper.any(axis=0)
    return np.where(has, grid.z_t_nom[k], grid.H)


#: The tape names of the dynamic ocean's planes and their attributes
#: (wrfout_export writes them only when the state carries them).
TAPE_FIELDS = {"SSH": "ssh_m", "UOCE": "u_m_s", "VOCE": "v_m_s", "HBL": "hbl_m",
               "D20": "d20_m", "OHC300": "ohc300_j_m2"}
TAPE_FIELD_ATTRS = {
    "SSH": {"units": "m", "description": "the dynamic ocean's sea surface height (0 off its columns)"},
    "UOCE": {"units": "m s-1", "description": "the dynamic ocean's top-cell eastward current"},
    "VOCE": {"units": "m s-1", "description": "the dynamic ocean's top-cell northward current"},
    "HBL": {"units": "m", "description": "the dynamic ocean's KPP boundary-layer depth"},
    "D20": {"units": "m", "description": "depth of the 20 C isotherm (0 where the column has none)"},
    "OHC300": {"units": "J m-2", "description": "the dynamic ocean's 0-300 m heat content"},
}


def tape_fields(arrays: dict) -> dict:
    """The dynamic ocean's tape planes from a checkpoint's arrays
    (``physics__`` names or a namespace's own); an empty dict when the arrays
    carry none, so a tape of any other run is unchanged."""
    out = {}
    for tape, name in TAPE_FIELDS.items():
        for key in (f"physics__{TAPE_PREFIX}{name}", f"{TAPE_PREFIX}{name}"):
            if key in arrays:
                out[tape] = np.asarray(_host(arrays[key]), dtype=np.float64)
                break
    return out


__all__ = ["ACCUMULATORS", "DynamicOceanProvider", "HELD", "METADATA_KEY", "PREFIX",
           "TAPE_FIELDS", "TAPE_FIELD_ATTRS", "TAPE_NAMES", "TAPE_PREFIX", "is_dynamic_array",
           "params_from_options", "state_names", "tape_fields"]
