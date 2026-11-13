"""The stratospheric noise gate: small-scale kinetic energy between 10 and
100 hPa, read from the spectra the ledger already samples.

DIAGNOSTIC, OFF BY DEFAULT, UNCALIBRATED.  It is set by
``[insitu] stratospheric_noise_limit``; unset, nothing here runs and the
ledger and the receipt are what they were.  It never fails a run: its row
sits in the receipt's in-situ block, not among the gates the status is read
from, until it has been calibrated on quiet cases and on a sudden-warming
case (a limit tuned on quiet days would trip on a warming).

WHY THIS NUMBER.  A longer step puts its damage in the smallest scales of
the lower stratosphere first: on one T255 time-step sweep (one GDAS day,
the shipped split, CUDA) the top-decile kinetic energy at about 48 hPa,
against the 300 s run's at matched forecast time, read 3x at 450 s, 11x at
600 s, 40x at 900 s and 79x at 1200 s, while no gate the model had read it.  A running forecast has no
300 s twin to divide by, so the gate reads a quantity each run has on its
own: on every level whose reference pressure lies in the band, the
top-decile share of the level's kinetic energy (rotational plus divergent,
degrees from ``truncation - truncation // 10`` up, over the level's total),
and its maximum over the band's levels and over the run's samples.  The
share is scale-free, so one limit can serve a quiet day and a stormy one
better than an energy would; the sampled spectra are the ones the ledger
already computes from the dycore's own coefficients, so it costs no
transform.

``tools/arwen_global_strat_noise_replay.py`` replays written ledgers
through this same code.
"""
from __future__ import annotations

import math

import numpy as np

from .energy import LEVEL_BAND_REFERENCE_PS_PA

#: The default band, in hPa: 10 to 100 hPa, where the long-step noise sat.
DEFAULT_STRATOSPHERIC_BAND_HPA = (10.0, 100.0)


def stratospheric_levels(vertical, band_hpa=DEFAULT_STRATOSPHERIC_BAND_HPA,
                         reference_ps_pa: float = LEVEL_BAND_REFERENCE_PS_PA
                         ) -> list[tuple[int, float]]:
    """``(index, reference full pressure in Pa)`` of every model level whose
    full pressure on the reference column (the geometric mean of its two
    half levels at ``reference_ps_pa``, as the energy ledger's reference
    level is placed) lies inside ``band_hpa`` inclusive."""
    low, high = (100.0 * float(value) for value in band_hpa)
    a = np.asarray(vertical.a_half_pa, dtype=np.float64)
    b = np.asarray(vertical.b_half, dtype=np.float64)
    p_half = a + b * float(reference_ps_pa)
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    return [(int(k), float(p)) for k, p in enumerate(p_full) if low <= p <= high]


def top_decile_share(sample: dict, levels) -> tuple[float | None, int | None]:
    """The largest top-decile share of kinetic energy over ``levels`` in
    one spectra sample, and the level it is on."""
    worst, where = None, None
    for index, _pressure in levels:
        top = (sample["rot_top_decile_by_level"][index]
               + sample["div_top_decile_by_level"][index])
        total = (sample["rot_total_by_level"][index]
                 + sample["div_total_by_level"][index])
        if not (total > 0.0) or not math.isfinite(top):
            continue
        share = top / total
        if worst is None or share > worst:
            worst, where = share, index
    return worst, where


class StratosphericNoiseGate:
    """Reads every spectra sample and keeps the run's worst share."""

    def __init__(self, vertical, limit: float,
                 band_hpa=DEFAULT_STRATOSPHERIC_BAND_HPA):
        self.limit = float(limit)
        self.band_hpa = (float(band_hpa[0]), float(band_hpa[1]))
        self.levels = stratospheric_levels(vertical, self.band_hpa)
        self.samples = 0
        self.maximum: float | None = None
        self.maximum_at: dict | None = None
        self.first_over: dict | None = None
        self.last: dict | None = None

    def evaluate(self, sample: dict) -> float | None:
        share, level = top_decile_share(sample, self.levels)
        self.samples += 1
        if share is None:
            return None
        record = {
            "step": int(sample["step"]),
            "time_s": float(sample["time_s"]),
            "level": int(level),
            "reference_pressure_pa": dict(self.levels)[level],
            "share": float(share),
        }
        self.last = record
        if self.maximum is None or share > self.maximum:
            self.maximum, self.maximum_at = float(share), record
        if share > self.limit and self.first_over is None:
            self.first_over = record
        return float(share)

    def summary(self) -> dict:
        return {
            "definition": (
                "max over sampled steps and over the levels whose reference "
                "pressure lies in the band of the top-decile share of the "
                "level's kinetic energy (rotational + divergent); a "
                "diagnostic, uncalibrated, never part of the run's status"
            ),
            "band_hpa": list(self.band_hpa),
            "levels": [index for index, _ in self.levels],
            "level_reference_pressure_pa": [p for _, p in self.levels],
            "limit": self.limit,
            "samples": int(self.samples),
            "value": self.maximum,
            "at": self.maximum_at,
            "last": self.last,
            "first_over_limit": self.first_over,
            "passed": None if self.maximum is None else bool(self.maximum <= self.limit),
        }


__all__ = [
    "DEFAULT_STRATOSPHERIC_BAND_HPA",
    "StratosphericNoiseGate",
    "stratospheric_levels",
    "top_decile_share",
]
