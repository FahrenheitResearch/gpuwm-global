"""The ocean and sea-ice rules: hold, anomaly and climatology.

Each rule is a closed form in model time from three things known at the
start: a model-grid monthly climatology (arwen_global.surface_climatology,
OISST v2.1 1991-2020), the starting state, and its parameters.  So a rule
needs no data later than the start, covers any length, and a restarted run
evaluates the same expression at the same instant as the straight run.

Symbols: t elapsed seconds; C_s(t), C_i(t) the climatological SST and ice
fraction, linear in time between mid-month values; T_f = 271.35 K.

**Starting anomaly.**  S_0 is the open-water SST at the start: the model's
own cold-start skin (``anomaly_reference = "start"``) or the mean of the 28
GDAS analyses of the 7 days before the start (``"mean-7d"``, a series built
with the analysis ice as land in the open-water rule's stencil, so an ice
point's skin never enters an open-water value), held at or above T_f (a
colder value is an ice or blended skin, not an SST).  On columns that are open
ocean at the start, A_0 = S_0 - C_s(0); on columns that start as ice, on
lakes, and where the climatology or S_0 has no value, A_0 = 0 and the column
takes the climatology.  A_0 = Abar + A', with Abar its area mean over those
columns and A' the rest, smoothed to T``anomaly_truncation`` (T42) and put
back to zero where A_0 is zero.

**Damping rate.**  r = r_x s + r_t (1 - s) with r_x = 1/``anomaly_damping_
days``, r_t = 1/``tropical_anomaly_damping_days`` (0 for "none") and s a
half-cosine ramp in |latitude| from 0 at the band's inner edge to 1 at its
outer edge.  Under ``tropical_persist_region = "pacific"`` the tropical rate
applies only between 120 E and 80 W, with a 10 degree half-cosine taper to
the extratropical rate outside them; elsewhere r = r_x.

* ``anomaly``:     S = max(T_f, C_s(t) + Abar g(t) + A' exp(-r t)), g = 1
  (``large_scale_offset = "persist"``) or exp(-r_x t) ("damp");
* ``climatology``: S = max(T_f, C_s(t) + (Abar + A') exp(-t/ramp));
* ``hold``:        the skin is not written (the cold-start SST, held).

**Sea ice**, at records t_k = k ``record_interval_hours`` on ocean columns
(land fraction at most one half, not a lake, in the climatology's ocean):
I = clip(C_i(t_k) + (I_0 - C_i(0)) exp(-t_k/tau), 0, 1), tau the ice
damping (``anomaly``) or the climatology ramp (``climatology``); 0 where the
column's open-water SST is at or above T_f + 2 K; a column that becomes ice
takes ``new_ice_thickness_m``, one that stays ice keeps its thickness.
``hold`` replaces nothing.  Lakes are never written: their skin follows its
own budget and their ice stays as it starts.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import math

import numpy as np

from .surface_series import SEA_WATER_FREEZING_K, SERIES_SCHEMA, _utc

SECONDS_PER_DAY = 86400.0
ICE_RULE_WARM_MARGIN_K = 2.0
#: The physics-metadata key of the restart identity block.
IDENTITY_KEY = "surface_boundary"
NINO34_LATITUDE = (-5.0, 5.0)
NINO34_LONGITUDE = (190.0, 240.0)
PACIFIC_LONGITUDE = (120.0, 280.0)
PACIFIC_TAPER_DEG = 10.0
TROPICS_DEG = 10.0
EXTRATROPICS_DEG = 30.0
REFERENCE_SPAN = timedelta(days=7)
REFERENCE_INTERVAL = timedelta(hours=6)


def _sha(array) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode()
                          + a.tobytes()).hexdigest()


def _area_weights(grid) -> np.ndarray:
    w = np.asarray(grid.quadrature_weights, dtype=np.float64)
    return np.broadcast_to(w[:, None], (w.size, int(grid.nlon)))


def area_mean(grid, plane, mask) -> float | None:
    weight = _area_weights(grid) * np.asarray(mask, dtype=np.float64)
    total = float(weight.sum())
    if total <= 0.0:
        return None
    return float((np.where(mask, plane, 0.0) * weight).sum() / total)


def _half_cosine(x):
    """0 at x <= 0, 1 at x >= 1, a half cosine between."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(math.pi * x))


def pacific_weight(longitude_deg) -> np.ndarray:
    """1 between 120 E and 80 W (280 E), 0 beyond a 10 degree half-cosine
    taper outside them."""
    lon = np.mod(np.asarray(longitude_deg, dtype=np.float64), 360.0)
    west, east = PACIFIC_LONGITUDE
    inside = (lon >= west) & (lon <= east)
    below = (west - lon) / PACIFIC_TAPER_DEG
    above = (lon - east) / PACIFIC_TAPER_DEG
    outside = np.where(lon < west, below, above)
    return np.where(inside, 1.0, 1.0 - _half_cosine(outside))


def damping_rate(latitude_deg, longitude_deg, options) -> np.ndarray:
    """r(phi, lambda) in 1/s on the (nlat, nlon) grid."""
    lat = np.abs(np.asarray(latitude_deg, dtype=np.float64))[:, None]
    lon = np.asarray(longitude_deg, dtype=np.float64)[None, :]
    r_x = 1.0 / (options.anomaly_damping_days * SECONDS_PER_DAY)
    tau_t = options.tropical_anomaly_damping_days
    r_t = 0.0 if tau_t is None else 1.0 / (tau_t * SECONDS_PER_DAY)
    inner, outer = options.tropical_band_deg
    s = _half_cosine((lat - inner) / (outer - inner)) * np.ones_like(lon)
    if options.tropical_persist_region == "pacific":
        p = pacific_weight(lon) * np.ones_like(lat)
        tropical = p * r_t + (1.0 - p) * r_x
    else:
        tropical = np.full(s.shape, r_t)
    return r_x * s + tropical * (1.0 - s)


def truncate(plane, grid, truncation, model_truncation):
    """``plane`` projected onto spherical harmonics up to ``truncation``
    on ``grid`` (float64, numpy); unchanged when ``truncation`` is None or
    not below the model's own."""
    if truncation is None or int(truncation) >= int(model_truncation):
        return np.asarray(plane, dtype=np.float64)
    from .spectral.transform import SphericalHarmonicTransform

    transform = SphericalHarmonicTransform.create(
        int(truncation), grid=grid, backend="numpy", precision="float64")
    return np.asarray(transform.inverse(transform.forward(
        np.asarray(plane, dtype=np.float64))), dtype=np.float64)


def reference_sst(path, *, start, nlat: int, nlon: int) -> tuple[np.ndarray, dict]:
    """S_0 from an anomaly-reference series: the mean of its records, each
    checked to lie in the 7 days before ``start``."""
    with np.load(path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        sst = np.asarray(data["sst_k"], dtype=np.float64)
        valid = [_utc(str(v)) for v in data["valid_utc"]]
    if provenance.get("schema") != SERIES_SCHEMA:
        raise ValueError(f"{path} is not a {SERIES_SCHEMA} series")
    if provenance.get("source_water") != "land-and-ice":
        raise ValueError(
            f"{path} was not built as an anomaly reference: its open-water rule read "
            "the analysis ice as water.  Build it with `gpuwm-global surface-series "
            "... --anomaly-reference`")
    if sst.shape[1:] != (nlat, nlon):
        raise ValueError(f"{path} is on a {sst.shape[1]}x{sst.shape[2]} grid, not "
                         f"this run's {nlat}x{nlon}")
    start = _utc(start)
    early = [v for v in valid if not (start - REFERENCE_SPAN <= v < start)]
    if early or not valid:
        raise ValueError(
            f"{path}: an anomaly reference holds analyses of the 7 days before the "
            f"run's start {start.isoformat()}; records "
            f"{', '.join(v.isoformat() for v in early) or 'none'} lie outside it")
    return sst.mean(axis=0), {
        "path": str(path), "records": len(valid),
        "first_utc": valid[0].isoformat().replace("+00:00", "Z"),
        "last_utc": valid[-1].isoformat().replace("+00:00", "Z"),
        "source": provenance.get("source"),
    }


class RuleProvider:
    """The hold, anomaly and climatology rules as a record provider of the
    boundary service (arwen_global.surface_boundary)."""

    kind = "rule"

    def __init__(self, options, climatology, grid, *, start_time, cold_surface: dict,
                 model_truncation: int, reference=None):
        """``cold_surface`` holds host planes of the cold-start surface:
        ``temperature_k``, ``land_fraction``, ``lake_fraction``,
        ``sea_ice_fraction`` and ``sea_ice_thickness_m``.  ``reference`` is
        ``(S_0 plane, record)`` from :func:`reference_sst`, or None for the
        cold-start skin."""
        from .statics import lake_columns, water_columns

        self.options = options
        self.mode = options.mode
        self.ice_mode = options.ice_mode
        self.clim = climatology
        self.grid = grid
        self.start = _utc(start_time)
        lat = np.asarray(grid.latitude_deg, dtype=np.float64)
        lon = np.asarray(grid.longitude_deg, dtype=np.float64)
        self.latitude_deg, self.longitude_deg = lat, lon
        if climatology is not None:
            climatology.check_grid(lat, lon)
        f = {name: np.asarray(value, dtype=np.float64) for name, value in cold_surface.items()}
        land, lake = f["land_fraction"], f["lake_fraction"]
        ice0 = f["sea_ice_fraction"]
        self.cold_skin = f["temperature_k"]
        water0 = np.asarray(water_columns(land.astype(np.float32), np,
                                          sea_ice_fraction=ice0.astype(np.float32)))
        lake0 = np.asarray(lake_columns(land.astype(np.float32), lake.astype(np.float32),
                                        np, sea_ice_fraction=ice0.astype(np.float32)))
        lake_any = np.asarray(lake_columns(land.astype(np.float32),
                                           lake.astype(np.float32), np))
        clim_ocean = (np.asarray(climatology.ocean, dtype=bool) if climatology is not None
                      else np.zeros(land.shape, dtype=bool))
        self.open0 = water0 & ~lake0
        #: The columns the ice rule writes: ocean water, never a lake.
        self.ice_columns = clim_ocean & (land.astype(np.float32) <= np.float32(0.5)) \
            & ~lake_any & ~lake0
        self.defined = clim_ocean & ~lake_any
        self.stats: dict[str, object] = {}
        self._device: dict = {}
        self._last_t = None
        self._record_planes = None
        self.r = None
        if self.mode in ("anomaly", "climatology"):
            self._build_anomaly(f, reference, model_truncation, water0, lake_any)
        if self.ice_mode in ("anomaly", "climatology"):
            c_i0 = climatology.at("sea_ice_fraction", self.start).astype(np.float64)
            self.ice_anomaly = np.where(self.ice_columns, ice0 - c_i0, 0.0)
            self.ice_tau_s = (options.sea_ice.anomaly_damping_days
                              if self.ice_mode == "anomaly"
                              else options.climatology_ramp_days) * SECONDS_PER_DAY
            self.stats["ice_start_anomaly_mean"] = area_mean(
                grid, self.ice_anomaly, self.ice_columns)
        self.record_s = options.sea_ice.record_interval_hours * 3600.0
        self.stats["future_surface_records"] = 0
        if self.mode in ("anomaly", "climatology"):
            first = self.sst_host(0.0)
            mask = self.open0 & self.defined
            change = np.where(mask, first - self.cold_skin, 0.0)
            self.stats["first_application_change_k"] = {
                "rms": (float(np.sqrt(area_mean(grid, change * change, mask)))
                        if mask.any() else None),
                "max_abs": float(np.abs(change).max()) if mask.any() else None,
            }

    # -- the starting anomaly ------------------------------------------------

    def _build_anomaly(self, f, reference, model_truncation, water0, lake_any) -> None:
        options, grid = self.options, self.grid
        if reference is None:
            s0, record = self.cold_skin, {"anomaly_reference": "start"}
        else:
            s0, record = reference
            record = {"anomaly_reference": "mean-7d", **record}
        c0 = self.clim.at("sst_k", self.start).astype(np.float64)
        sourced = np.isfinite(s0) & np.isfinite(c0)
        mask = self.open0 & self.defined & sourced
        # An open-water value below the freezing point of sea water is not
        # an SST: the analysis skin of a point with some ice is a blend of
        # ice and water skins (GDAS 2026-11-10: 252 K on open-water columns
        # at 76 N), and the cold start's own rule lets an ice skin into an
        # open-water column beside the ice (241 K at 82 N).  S_0 is held at
        # or above T_f, so neither becomes an anomaly the T42 smoothing
        # spreads over the water around it.
        frozen_source = mask & (s0 < SEA_WATER_FREEZING_K)
        s0 = np.where(frozen_source, SEA_WATER_FREEZING_K, s0)
        a0 = np.where(mask, s0 - c0, 0.0)
        abar = area_mean(grid, a0, mask)
        abar = 0.0 if abar is None else abar
        residual = np.where(mask, a0 - abar, 0.0)
        smoothed = truncate(residual, grid, options.anomaly_truncation, model_truncation)
        self.residual = np.where(mask, smoothed, 0.0)
        self.offset = np.where(mask, abar, 0.0)
        self.anomaly_columns = mask
        self.r = damping_rate(self.latitude_deg, self.longitude_deg, options)
        lat = self.latitude_deg[:, None] * np.ones((1, self.longitude_deg.size))
        box = self._box_mask()
        frozen0 = (np.asarray(f["land_fraction"]) <= 0.5) & ~water0 & ~lake_any
        self.stats.update({
            **record,
            "abar_k": abar,
            "nino34_start_anomaly_k": area_mean(grid, a0, mask & box),
            "tropical_mean_residual_k": area_mean(
                grid, self.residual, mask & (np.abs(lat) <= TROPICS_DEG)),
            "extratropical_mean_residual_k": area_mean(
                grid, self.residual, mask & (np.abs(lat) >= EXTRATROPICS_DEG)),
            "anomaly_truncation": ("none" if options.anomaly_truncation is None
                                   else int(options.anomaly_truncation)),
            "columns_with_anomaly": int(mask.sum()),
            "columns_whose_start_sst_was_raised_to_t_f": int(frozen_source.sum()),
            "columns_given_zero_anomaly": {
                "ice_at_start": int((frozen0 & self.defined).sum()),
                "lake": int((lake_any & (np.asarray(f["land_fraction"]) <= 0.5)).sum()),
                "no_climatology_or_source": int(
                    (self.open0 & ~lake_any & ~(self.defined & sourced)).sum()),
            },
        })

    def _box_mask(self):
        lat = self.latitude_deg[:, None]
        lon = np.mod(self.longitude_deg, 360.0)[None, :]
        return ((lat >= NINO34_LATITUDE[0]) & (lat <= NINO34_LATITUDE[1])
                & (lon >= NINO34_LONGITUDE[0]) & (lon <= NINO34_LONGITUDE[1]))

    # -- the closed forms ----------------------------------------------------

    def when(self, t: float):
        return self.start + timedelta(seconds=float(t))

    def _factors(self, t: float, xp, r):
        if self.mode == "climatology":
            fac = math.exp(-float(t) / (self.options.climatology_ramp_days * SECONDS_PER_DAY))
            return fac, fac
        g = 1.0
        if self.options.large_scale_offset == "damp":
            g = math.exp(-float(t) / (self.options.anomaly_damping_days * SECONDS_PER_DAY))
        return g, xp.exp(-r * float(t))

    def sst_host(self, t: float) -> np.ndarray:
        """The rule SST at ``t`` on the host, float64 (NaN off the
        climatology's ocean); under ``hold``, the held skin."""
        if self.mode == "hold":
            c = self.clim.at("sst_k", self.when(t)).astype(np.float64) \
                if self.clim is not None else self.cold_skin
            return np.where(self.open0, self.cold_skin,
                            np.maximum(SEA_WATER_FREEZING_K, c))
        c = self.clim.at("sst_k", self.when(t)).astype(np.float64)
        g, fac = self._factors(t, np, self.r)
        return np.maximum(SEA_WATER_FREEZING_K, c + self.offset * g + self.residual * fac)

    def _on(self, xp, name, value):
        key = (xp.__name__, name)
        if key not in self._device:
            self._device[key] = xp.asarray(value)
        return self._device[key]

    def _month_on(self, xp, i):
        key = (xp.__name__, "sst", i)
        if key not in self._device:
            stale = [k for k in self._device if k[1] == "sst"]
            if len(stale) >= 2:
                for k in stale:
                    del self._device[k]
            self._device[key] = xp.asarray(self.clim.sst_k[i])
        return self._device[key]

    def record_index(self, t: float) -> int:
        return int(float(t) // self.record_s + 1.0e-9)

    def sst_at(self, t: float, xp, dtype):
        self._last_t = float(t)
        if self.mode == "hold":
            return None, None
        from .surface_climatology import month_weights

        i0, i1, w = month_weights(self.when(t))
        a, b = self._month_on(xp, i0), self._month_on(xp, i1)
        c = (a * np.float32(1.0 - w) + b * np.float32(w)).astype(xp.float64)
        offset = self._on(xp, "offset", self.offset)
        residual = self._on(xp, "residual", self.residual)
        r = self._on(xp, "r", self.r)
        g, fac = self._factors(t, xp, r)
        value = xp.maximum(xp.float64(SEA_WATER_FREEZING_K), c + offset * g + residual * fac)
        defined = self._on(xp, "defined", self.defined)
        return xp.where(defined, value, xp.float64(SEA_WATER_FREEZING_K)).astype(
            dtype, copy=False), defined

    def ice_host(self, k: int, current_fraction, current_thickness):
        t = k * self.record_s
        c_i = self.clim.at("sea_ice_fraction", self.when(t)).astype(np.float64)
        fraction = np.clip(c_i + self.ice_anomaly * math.exp(-t / self.ice_tau_s), 0.0, 1.0)
        warm = self.sst_host(t) >= SEA_WATER_FREEZING_K + ICE_RULE_WARM_MARGIN_K
        fraction = np.where(warm, 0.0, fraction)
        fraction = np.where(self.ice_columns, fraction, current_fraction)
        new_frozen = fraction >= 0.5
        old_frozen = np.asarray(current_fraction) >= 0.5
        thickness = np.where(self.ice_columns & new_frozen & ~old_frozen,
                             self.options.sea_ice.new_ice_thickness_m, current_thickness)
        return fraction, thickness, int((self.ice_columns & new_frozen & ~old_frozen).sum())

    def ice_record(self, k: int, xp, surface, get):
        if self.ice_mode not in ("anomaly", "climatology"):
            return None, None, None
        from .surface_boundary import _host

        current = get(xp, surface, "sea_ice_fraction")
        dtype = current.dtype
        fraction, thickness, new_ice = self.ice_host(
            k, _host(current).astype(np.float64),
            _host(get(xp, surface, "sea_ice_thickness_m")).astype(np.float64))
        fraction = fraction.astype(dtype)
        thickness = thickness.astype(dtype)
        self._record_planes = (k, fraction)
        self.stats["new_ice_columns"] = int(self.stats.get("new_ice_columns", 0)) + new_ice
        return (xp.asarray(fraction), xp.asarray(thickness),
                self._on(xp, "ice_columns", self.ice_columns))

    def record_sst_host(self, k: int):
        return self.sst_host(k * self.record_s)

    def record_valid(self, k: int):
        return self.when(k * self.record_s)

    def last_info(self, t: float) -> dict:
        return {"time_s": float(t), "record": self.record_index(t)}

    # -- identity, receipt and the daily row ---------------------------------

    def identity_block(self) -> dict:
        block = {
            "ocean_mode": self.mode, "sea_ice_mode": self.ice_mode,
            "start_utc": self.start.isoformat().replace("+00:00", "Z"),
            "climatology_sha256": None if self.clim is None else self.clim.sha256,
            "parameters": self.options.receipt(),
        }
        block["parameters"].pop("climatology", None)
        block["parameters"].pop("anomaly_reference_series", None)
        if self.mode in ("anomaly", "climatology"):
            block["start_anomaly_sha256"] = _sha(np.stack([self.offset, self.residual]))
        if self.ice_mode in ("anomaly", "climatology"):
            block["ice_start_anomaly_sha256"] = _sha(self.ice_anomaly)
        return block

    def receipt(self) -> dict:
        return {
            "provider": "rule",
            **self.options.receipt(),
            "climatology_provenance": None if self.clim is None else {
                "sha256": self.clim.sha256,
                "inputs": self.clim.provenance.get("inputs"),
                "rows": self.clim.provenance.get("rows"),
                "truncation": self.clim.provenance.get("truncation"),
            },
            "start": dict(self.stats),
            "sst": {
                "hold": "the cold-start SST, not written",
                "anomaly": "max(T_f, C_s(t) + Abar g(t) + A' exp(-r t)) on open ocean",
                "climatology": "max(T_f, C_s(t) + (Abar + A') exp(-t/ramp)) on open ocean",
            }[self.mode],
            "sea_ice": {
                "hold": "the cold-start ice, not written",
                "anomaly": "clip(C_i(t) + (I_0 - C_i(0)) exp(-t/tau_i), 0, 1) at each "
                           "record; 0 where the SST is at or above T_f + 2 K",
                "climatology": "clip(C_i(t) + (I_0 - C_i(0)) exp(-t/ramp), 0, 1) at each "
                               "record; 0 where the SST is at or above T_f + 2 K",
            }[self.ice_mode],
            "lakes": "not written: the skin follows its own budget, the ice is held",
        }

    def diagnostics(self, state) -> dict:
        """The boundary rows of one output step, area means on the grid's
        quadrature, read from the state."""
        from .surface_climatology import month_weights
        from .statics import lake_columns, water_columns

        t = float(state.time_s)
        planes = cold_surface_planes(state.surface)
        skin = planes["temperature_k"].astype(np.float64)
        land = planes["land_fraction"].astype(np.float32)
        ice = planes["sea_ice_fraction"].astype(np.float32)
        lake = planes["lake_fraction"].astype(np.float32)
        ocean = (np.asarray(water_columns(land, np, sea_ice_fraction=ice))
                 & ~np.asarray(lake_columns(land, lake, np, sea_ice_fraction=ice))
                 & self.defined)
        grid = self.grid
        lat = self.latitude_deg[:, None] * np.ones((1, self.longitude_deg.size))
        rows: dict[str, object] = {
            "boundary_provider": f"rule:{self.mode}/{self.ice_mode}",
            "boundary_valid_utc": self.when(t).isoformat().replace("+00:00", "Z"),
        }
        if self.clim is not None:
            i0, i1, w = month_weights(self.when(t))
            c = self.clim.at("sst_k", self.when(t)).astype(np.float64)
            departure = np.where(ocean, skin - c, 0.0)
            rows.update({
                "boundary_climatology_months": [i0 + 1, i1 + 1],
                "boundary_climatology_weight": w,
                "boundary_open_ocean_mean_sst_k": area_mean(grid, skin, ocean),
                "boundary_mean_sst_anomaly_k": area_mean(grid, departure, ocean),
                "boundary_tropical_mean_sst_anomaly_k": area_mean(
                    grid, departure, ocean & (np.abs(lat) <= TROPICS_DEG)),
                "boundary_extratropical_mean_sst_anomaly_k": area_mean(
                    grid, departure, ocean & (np.abs(lat) >= EXTRATROPICS_DEG)),
                "boundary_nino34_sst_anomaly_k": area_mean(
                    grid, departure, ocean & self._box_mask()),
            })
        if self.r is not None:
            j = int(np.argmin(np.abs(np.mod(self.longitude_deg, 360.0) - 200.0)))
            for name, phi in (("0n", 0.0), ("20n", 20.0), ("45n", 45.0)):
                i = int(np.argmin(np.abs(self.latitude_deg - phi)))
                g, fac = self._factors(t, np, self.r[i, j])
                rows[f"boundary_damping_factor_{name}_200e"] = (
                    float(fac) if self.mode == "anomaly" else float(g))
        if self.mode != "hold" and self._last_t is not None:
            expected = self.sst_host(self._last_t)
            rows["boundary_max_abs_sst_departure_k"] = float(
                np.abs(np.where(ocean, skin - expected, 0.0)).max()) if ocean.any() else 0.0
        if self._record_planes is not None:
            _k, planes = self._record_planes
            rows["boundary_ice_record_mismatch_columns"] = int(
                (self.ice_columns & (ice != planes.astype(np.float32))).sum())
        areas = 4.0 * math.pi * float(grid.radius_m) ** 2 * (
            0.5 * _area_weights(grid)) / int(grid.nlon) / 1.0e12
        water = (land <= np.float32(0.5)) & ~np.asarray(lake_columns(land, lake, np))
        for hemisphere, rows_mask in (("nh", lat > 0.0), ("sh", lat < 0.0)):
            for threshold, label in ((0.5, "050"), (0.15, "015")):
                cover = water & rows_mask & (ice >= np.float32(threshold))
                rows[f"boundary_ice_extent_{hemisphere}_{label}_mkm2"] = float(
                    areas[cover].sum())
                rows[f"boundary_ice_area_{hemisphere}_{label}_mkm2"] = float(
                    (areas * ice)[cover].sum())
        return rows


def cold_surface_planes(surface) -> dict:
    """The host planes :class:`RuleProvider` reads from a cold state."""
    from .surface_boundary import _host

    def plane(name):
        value = getattr(surface, name, None)
        if value is None:
            return np.zeros(np.shape(_host(surface.land_fraction)))
        return _host(value)

    return {name: plane(name) for name in (
        "temperature_k", "land_fraction", "lake_fraction", "sea_ice_fraction",
        "sea_ice_thickness_m")}


def check_restart_identity(expected: dict | None, physics_metadata: dict, label: str) -> None:
    """Refuse a checkpoint whose boundary was another rule, another
    climatology or another starting anomaly, before any array is used."""
    found = (physics_metadata or {}).get(IDENTITY_KEY)
    if expected is None and found is None:
        return
    if expected != found:
        def brief(block):
            if block is None:
                return "none"
            return (f"{block.get('ocean_mode')}/{block.get('sea_ice_mode')}, climatology "
                    f"{str(block.get('climatology_sha256'))[:12]}, start "
                    f"{block.get('start_utc')}, anomaly "
                    f"{str(block.get('start_anomaly_sha256'))[:12]}")
        raise ValueError(
            f"{label} was written under another ocean boundary ({brief(found)}) than "
            f"this run's ({brief(expected)}); a restart must continue the boundary it "
            "was written with")


__all__ = [
    "IDENTITY_KEY", "RuleProvider", "area_mean", "check_restart_identity",
    "cold_surface_planes", "damping_rate", "pacific_weight", "reference_sst", "truncate",
]
