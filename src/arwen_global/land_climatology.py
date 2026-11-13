"""The greenness march: vegetation, leaf area and background albedo by date.

Once per model day the vegetation fraction, the leaf area index and the
background albedo on land are the monthly climatologies of the statics
cache (GREENFRAC, LAI12M, ALBEDO12M) at the date, by WRF real's
``monthly_interp_to_date`` (MPAS ``config_greeness_update = 24:00:00``).
It moves Noah's albedo, emissivity and roughness through the vegetation
fraction; interannual phenology is not predicted, and the annual minimum
and maximum vegetation fraction are constant over a run, so they are not
re-resolved.

The date is the model day's start, ``start + day`` whole days, not the
step's instant.  Until this module the march lived in the series service
and took the instant of the first step it ran on, which after a restart in
the middle of a model day is the restart instant: a run started at 06, 12
or 18 UTC and restarted inside a day took another UTC day's vegetation
than the straight run (``monthly_interp_to_date`` reads whole UTC days).
Anchored at the day start, the march gives a restarted run the planes of
the straight run.  A straight run whose step lands on each day boundary
reads the same date either way.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from .surface_series import _utc

SECONDS_PER_DAY = 86400.0


def greenness_planes(statics_fields, when) -> dict[str, np.ndarray]:
    """The three land planes at ``when`` (a datetime), float64."""
    from gpuwm.static.build import monthly_interp_to_date

    f = statics_fields
    valid = when.replace(tzinfo=None)
    return {
        "vegetation_fraction": np.clip(monthly_interp_to_date(
            np.asarray(f["GREENFRAC"], dtype=np.float64), valid), 0.0, 1.0),
        "leaf_area_index": np.clip(monthly_interp_to_date(
            np.asarray(f["LAI12M"], dtype=np.float64), valid), 0.0, None),
        "background_albedo": np.clip(monthly_interp_to_date(
            np.asarray(f["ALBEDO12M"], dtype=np.float64), valid) / 100.0, 0.0, 1.0),
    }


class GreennessMarch:
    """Apply the monthly land climatologies once per model day."""

    def __init__(self, statics_fields, *, start_time):
        self.statics_fields = statics_fields
        self.start = _utc(start_time)
        self._day = None
        self.updates = 0
        self.last_date = None

    def anchor(self, t: float):
        """The date the march reads at model time ``t``: the model day's
        start."""
        day = int(float(t) // SECONDS_PER_DAY)
        return day, self.start + timedelta(days=day)

    def apply(self, state) -> None:
        from .surface_boundary import SurfaceBoundary

        day, when = self.anchor(state.time_s)
        if self._day == day:
            return
        surface = state.surface
        xp = SurfaceBoundary._xp(surface.temperature_k)
        get, put = SurfaceBoundary._get, SurfaceBoundary._put
        land = get(xp, surface, "land_fraction") > 0.5
        for name, plane in greenness_planes(self.statics_fields, when).items():
            current = get(xp, surface, name)
            put(surface, name, xp.where(land, xp.asarray(plane.astype(current.dtype)),
                                        current))
        self._day = day
        self.updates += 1
        self.last_date = when.isoformat().replace("+00:00", "Z")

    def receipt(self) -> dict:
        return {
            "rule": ("vegetation fraction, leaf area index and background albedo on "
                     "land from the statics' monthly climatologies, once per model "
                     "day, at the model day's start (monthly_interp_to_date)"),
            "greenness_updates": self.updates, "last_date": self.last_date,
        }


__all__ = ["GreennessMarch", "greenness_planes"]
