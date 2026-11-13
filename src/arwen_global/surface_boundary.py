"""The lower-boundary service: one machine, two kinds of record provider.

The machine is what ``surface_series.SurfaceSeriesUpdate`` always did, moved
here unchanged in its arithmetic:

* every step, the SST on the open-ocean columns (water, not frozen, not an
  inland lake: a lake's skin follows its own energy budget);
* at each ice record, the sea-ice fraction and thickness replaced on the
  provider's columns, and a column that crosses ``SEA_ICE_THRESHOLD``
  re-classed through the cold start's own rulebook
  (:func:`arwen_global.statics.resolve_surface_statics` at the record's
  valid time), after which the native runtime's per-band masks are dropped.

A provider answers what the machine writes:

* :class:`arwen_global.surface_series.SeriesProvider`: a dated series,
  linear between records, refused past its last record;
* :class:`arwen_global.ocean_rules.RuleProvider`: the hold, anomaly and
  climatology rules, closed form in time from a climatology file and the
  starting state, so they need no data later than the start and a
  restarted run reproduces the straight one.

Provider contract (duck-typed):

``record_index(t)``
    the ice record in force at model time ``t`` (s);
``sst_at(t, xp, dtype)``
    ``(plane, defined)``: the SST to write on open ocean at ``t``, and a
    boolean plane of the columns the provider defines (None: all), or
    ``(None, None)`` to leave the skin alone;
``ice_record(k, xp, surface, get)``
    ``(fraction, thickness, columns)`` for record ``k``: the planes, and
    the columns they replace (None: no ice change);
``record_sst_host(k)``
    the SST of record ``k`` on the host (float64), the skin a melting
    column takes;
``record_valid(k)``
    the record's valid time, at which the rulebook resolves a re-class;
``last_info(t)``, ``receipt()``.

A provider may also have ``advance(state, model=)``, called first in
every :meth:`SurfaceBoundary.apply`, before the record and the SST
(:class:`arwen_global.ocean_coupler.SlabProvider` steps the slab ocean
there); so may a ``coupler`` given beside the provider (the slab's
calibration, which deposits fluxes while the climatology rule writes the
surface).  A service with neither calls nothing more than it did.
"""
from __future__ import annotations

import time

import numpy as np

from .surface_series import SEA_WATER_FREEZING_K, invalidate_runtime_masks


def _host(array):
    """A host numpy copy of a device, parked or host plane."""
    from .spill import spilled

    if spilled(array):
        array = array.host
    return array.get() if hasattr(array, "get") else np.asarray(array)


class SurfaceBoundary:
    """Apply a record provider to a run's surface state every step."""

    def __init__(self, provider, *, statics_fields=None, statics_provenance=None,
                 terrain_height_m=None, latitude_deg=None, coupler=None):
        self.provider = provider
        self.coupler = coupler
        self._advancers = tuple(
            item for item in (coupler, provider)
            if item is not None and callable(getattr(item, "advance", None)))
        self.statics_fields = statics_fields
        self.statics_provenance = statics_provenance
        self.terrain = terrain_height_m
        self.latitude_deg = latitude_deg
        self._ice_record = None
        self._ocean = None
        self.counts = {"sst_steps": 0, "ice_records": 0, "ice_to_water": 0,
                       "water_to_ice": 0}
        self.seconds = 0.0
        self.last = {}

    @staticmethod
    def _xp(array):
        """The device module when the run is on a card (a parked plane is
        staged to the card to be worked on), numpy otherwise."""
        from .spill import spilled

        try:
            import cupy as cp
        except ImportError:  # pragma: no cover
            return np
        return cp if spilled(array) else cp.get_array_module(array)

    @staticmethod
    def _get(xp, surface, name):
        from .spill import resident

        return resident(xp, getattr(surface, name))

    @staticmethod
    def _put(surface, name, value) -> None:
        """Write a whole plane back: in place on the card, or the parked
        slot replaced through the tier (spill.SpilledArray.store), as the
        water fixer replaces its reservoir."""
        from .spill import spilled

        target = getattr(surface, name)
        if spilled(target):
            target.store(value)
        elif isinstance(target, np.ndarray):
            # On the host a checkpoint holds the state's own arrays (the
            # numpy backend's to_numpy is the identity) while its writer
            # thread hashes them, and a state's arrays are never written in
            # place: the plane is replaced.
            setattr(surface, name, np.array(value, dtype=target.dtype))
        else:
            target[...] = value

    def _open_ocean(self, surface, xp):
        from .statics import lake_columns, water_columns

        if self._ocean is None:
            land = self._get(xp, surface, "land_fraction")
            ice = self._get(xp, surface, "sea_ice_fraction")
            water = water_columns(land, xp, sea_ice_fraction=ice)
            lake = lake_columns(land, self._get(xp, surface, "lake_fraction"), xp,
                                sea_ice_fraction=ice)
            self._ocean = xp.ascontiguousarray(water & ~lake)
        return self._ocean

    def apply(self, state, *, model=None) -> None:
        started = time.perf_counter()
        for advancer in self._advancers:
            advancer.advance(state, model=model)
        surface = state.surface
        xp = self._xp(surface.temperature_k)
        t = float(state.time_s)
        k = self.provider.record_index(t)
        if self._ice_record != k:
            # Record zero at a cold start is the state's own ice; a record
            # re-applied after a restart is the one the straight run
            # applied, so either is the identity.
            self._apply_ice(state, model, k, xp)
            self._ice_record = k
        skin = self._get(xp, surface, "temperature_k")
        sst, defined = self.provider.sst_at(t, xp, skin.dtype)
        if sst is not None:
            ocean = self._open_ocean(surface, xp)
            if defined is not None:
                ocean = ocean & defined
            self._put(surface, "temperature_k", xp.where(ocean, sst, skin))
            self.counts["sst_steps"] += 1
        self.last = self.provider.last_info(t)
        self.seconds += time.perf_counter() - started

    def _apply_ice(self, state, model, k, xp) -> None:
        from .statics import SEA_ICE_THRESHOLD

        surface = state.surface
        new_ice, new_thick, columns = self.provider.ice_record(k, xp, surface, self._get)
        if columns is None:
            return
        ice_now = self._get(xp, surface, "sea_ice_fraction")
        threshold = np.float32(SEA_ICE_THRESHOLD)
        old_frozen = xp.asarray(ice_now, dtype=xp.float32) >= threshold
        new_frozen = xp.asarray(new_ice, dtype=xp.float32) >= threshold
        freeze = columns & new_frozen & ~old_frozen
        melt = columns & old_frozen & ~new_frozen
        n_freeze = int(xp.count_nonzero(freeze))
        n_melt = int(xp.count_nonzero(melt))
        if (n_freeze or n_melt) and self.statics_fields is not None:
            self._convert(surface, xp, k, new_ice, new_thick, freeze, melt)
        elif n_freeze or n_melt:
            # Without the statics there is no rulebook to re-class with;
            # the crossing columns keep their class and their old ice.
            new_ice = xp.where(freeze | melt, ice_now, new_ice)
            n_freeze = n_melt = 0
        self._put(surface, "sea_ice_fraction", xp.where(columns, new_ice, ice_now))
        self._put(surface, "sea_ice_thickness_m", xp.where(
            columns, new_thick, self._get(xp, surface, "sea_ice_thickness_m")))
        self.counts["ice_records"] += 1
        self.counts["water_to_ice"] += n_freeze
        self.counts["ice_to_water"] += n_melt
        if n_freeze or n_melt:
            self._ocean = None
            invalidate_runtime_masks(model)

    def _convert(self, surface, xp, k, new_ice, new_thick, freeze, melt) -> None:
        from .statics import resolve_surface_statics
        from .surface_seeding import sea_ice_initial_soil_temperature

        skin = _host(self._get(xp, surface, "temperature_k")).astype(np.float64)
        soil = _host(self._get(xp, surface, "soil_temperature_k")).astype(np.float64)
        sst = self.provider.record_sst_host(k)
        ice = _host(new_ice).astype(np.float32)
        thickness = _host(new_thick).astype(np.float64)
        melt_h, freeze_h = _host(melt), _host(freeze)
        skin = np.where(melt_h, sst,
                        np.where(freeze_h, np.minimum(skin, SEA_WATER_FREEZING_K), skin))
        resolved, _ = resolve_surface_statics(
            self.statics_fields, self.statics_provenance,
            valid_time=self.provider.record_valid(k).replace(tzinfo=None),
            latitude_deg=self.latitude_deg, terrain_height_m=self.terrain,
            soil_temperature_k=soil, skin_temperature_k=skin,
            sea_ice_fraction=ice, snow_water_kg_m2=np.zeros_like(ice))
        mask = xp.asarray(freeze_h | melt_h)
        for name in ("landuse_category", "soil_category_top", "soil_category_bottom",
                     "albedo", "emissivity", "roughness_m", "snow_albedo"):
            current = self._get(xp, surface, name)
            self._put(surface, name, xp.where(
                mask, xp.asarray(resolved[name].astype(current.dtype)), current))
        new_soil = sea_ice_initial_soil_temperature(
            soil, skin, ice.astype(np.float64), thickness, np.zeros_like(skin))
        new_soil = np.where(melt_h[None], sst[None], new_soil)
        current = self._get(xp, surface, "soil_temperature_k")
        self._put(surface, "soil_temperature_k", xp.where(
            mask[None], xp.asarray(new_soil.astype(current.dtype)), current))
        current = self._get(xp, surface, "temperature_k")
        self._put(surface, "temperature_k", xp.where(
            mask, xp.asarray(skin.astype(current.dtype)), current))

    def receipt(self) -> dict:
        out = {**self.provider.receipt(), **self.counts,
               "host_seconds": round(self.seconds, 3), "last": self.last}
        if self.coupler is not None:
            out["coupler"] = self.coupler.receipt()
        return out


__all__ = ["SurfaceBoundary"]
