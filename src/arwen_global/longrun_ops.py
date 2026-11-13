"""Long-run operations: checkpoint retention, time means, a daily summary,
and the disk and wall estimate a launch is checked against.

All of it reads the arrays the checkpoint writer already brought to the
host and writes files beside the checkpoints; none of it changes the
forecast's arithmetic, so the ``[output]`` table is outside the config
identity.

**Retention** (``keep_checkpoints = "newest:N"`` or ``"newest:N+month-
ends"``).  After each checkpoint is handed to the writer, the durable ones
beyond the newest N are deleted (with their sidecars), except a month-end
(the last checkpoint whose valid time lies in its calendar month, kept
under ``+month-ends``) and the checkpoint this segment restarted from.
Pruning runs during the segment: at most N durable checkpoints plus the
one being written are on disk.  Before a checkpoint is deleted its header
(step, time, valid time, file size, ``self_sha256``, the config and pins
hashes and every array's SHA-256) is appended to
``arwen_global_pruned_checkpoints.jsonl`` beside the checkpoints, so a
replay of any window can be checked against the state the run held there.

**Mean state** (``mean_state = "monthly"``, ``"daily"`` or both).  At every
output step the checkpoint's own atmosphere and surface arrays and the
physics namespace's 2-D planes are summed in float64 (complex128 for
spectral coefficients) into the open calendar window of the step's valid
time; when a step falls in the next window the closed one is written as
``arwen_global_mean_<YYYY-MM>.npz`` (or ``<YYYY-MM-DD>``) in the
checkpoint layout: the mean of those arrays over the window's output
instants, every other array (categories, the namespace's 3-D working
arrays, the second time level) as its last value, and
``physics_metadata["time_mean"]`` naming the window, the sample count and
the first and last sample.  A mean file is refused as a restart, and an
exported tape of it carries ``ARWEN_TIME_MEAN``: a field the exporter
derives nonlinearly (wind speed, temperature from potential temperature
and pressure) is then a function of means, not a mean of the function.
The open windows' sums ride beside each checkpoint in a sidecar
(``<checkpoint>.longrun.npz``), so a restarted run's means are the
straight run's.

**Daily summary** (``daily_summary = true``).  Per UTC day, from the 2-D
planes a checkpoint carries: 2 m temperature mean, minimum and maximum over
the day's output instants, 2 m humidity and 10 m wind means, outgoing
longwave mean, skin temperature (the SST on open ocean) and sea-ice
fraction means, snow cover mean, and the day's precipitation (the surface
microphysics buckets plus the cumulus accumulator, the last instant of the
day less the last instant of the day before).  Written as
``arwen_global_daily_<YYYY-MM-DD>.npz``.

Beside those, planes made on the host from the same checkpoint arrays
(``instant_planes``; the runner hands the summary a numpy synthesis of the
run's grid, ``SummaryGeometry``):

* surface stress, eastward and northward (N/m2, the force on the surface,
  rho u*^2 along the 10 m wind), for an ocean layer driven from the run;
* sea-level pressure (mean, minimum) and 850 hPa relative vorticity (mean,
  largest cyclonic value, the fraction of instants 850 hPa lay under
  ground), for a storm index;
* sensible and latent heat flux, cloud liquid and ice water paths
  (sampled means), and the exact interval means of the radiation
  accumulators over the precipitation's interval: TOA incoming, reflected,
  absorbed, outgoing longwave and net, the four surface streams and their
  net, and the cloud cover.  Clear-sky fluxes are not written (the
  radiation holds none).

Each file's metadata states every plane's rule, the radiation interval and
the area means of every plane with the planetary albedo, the surface's net
energy and the precipitation rate.  Z500, T850, U250 and V250 are not in it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np

SIDECAR_SUFFIX = ".longrun.npz"
MEAN_KEY = "time_mean"
_STEP_RE = re.compile(r"^arwen_global_step(\d{8})\.npz$")
#: The one measured checkpoint: T383 L40 native, 1,522,236,583 B
#: (the step-1764 checkpoint of a seven-day run, design section 0.8).
MEASURED_CHECKPOINT_BYTES = 1_522_236_583
MEASURED_CHECKPOINT_SHAPE = (576, 1152, 40)
#: Measured stepping cost, minutes of wall per model day (design 0.8):
#: T383 10.7 to 12.2 on an RTX 5070 Ti (planning figure 12); T255 4.3 on
#: an older tree (README, 2026-09-06).
MINUTES_PER_DAY = {255: 4.3, 383: 12.0}
#: Planes in one daily summary file (and, as float64 sums, in its sidecar).
DAILY_SUMMARY_PLANES = 34
MEAN_SIDECARS_PER_WINDOW = 1


def checkpoint_step(path: Path) -> int | None:
    m = _STEP_RE.match(Path(path).name)
    return None if m is None else int(m.group(1))


def sidecar_of(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name[:-len(".npz")] + SIDECAR_SUFFIX)


def _utc(value) -> datetime:
    from .surface_series import _utc as series_utc

    return series_utc(value)


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------

def checkpoints_to_prune(steps, *, dt_s: float, start, keep_newest: int,
                         keep_month_ends: bool, protected=()) -> list[int]:
    """The steps of ``steps`` (durable checkpoints) retention deletes."""
    ordered = sorted(set(int(s) for s in steps))
    keep = set(ordered[-keep_newest:]) if keep_newest else set()
    keep |= {int(s) for s in protected}
    if keep_month_ends and start is not None:
        start = _utc(start)

        def month(step):
            v = start + timedelta(seconds=step * float(dt_s))
            return (v.year, v.month)

        for this, after in zip(ordered, ordered[1:]):
            if month(after) != month(this):
                keep.add(this)
    return [s for s in ordered if s not in keep]


class CheckpointRetention:
    def __init__(self, outdir: Path, options, *, dt_s: float, start, protected=()):
        self.outdir = Path(outdir)
        self.options = options
        self.dt_s = float(dt_s)
        self.start = start
        self.protected = {Path(p).resolve() for p in protected}
        self.deleted: list[str] = []
        self.recorded = 0

    @property
    def enabled(self) -> bool:
        return self.options.keep_newest is not None

    def prune(self, in_flight: Path | None = None) -> list[Path]:
        if not self.enabled:
            return []
        durable = {}
        for path in self.outdir.glob("arwen_global_step*.npz"):
            step = checkpoint_step(path)
            if step is None or (in_flight is not None
                                and path.resolve() == Path(in_flight).resolve()):
                continue
            durable[step] = path
        protected = [s for s, p in durable.items() if p.resolve() in self.protected]
        gone = []
        for step in checkpoints_to_prune(
                durable, dt_s=self.dt_s, start=self.start,
                keep_newest=self.options.keep_newest,
                keep_month_ends=self.options.keep_month_ends, protected=protected):
            path = durable[step]
            # The header first: once the file is gone nothing can say which
            # state it held.
            self._record(path)
            path.unlink(missing_ok=True)
            sidecar_of(path).unlink(missing_ok=True)
            gone.append(path)
            self.deleted.append(path.name)
        return gone

    def _record(self, path: Path) -> None:
        """Append the pruned checkpoint's header to the record beside the
        checkpoints (``arwen_global_pruned_checkpoints.jsonl``, one JSON
        object per line, flushed and fsynced before the file is deleted)."""
        _append_jsonl(self.outdir / PRUNED_RECORD, pruned_header(
            path, start=self.start))
        self.recorded += 1

    def receipt(self) -> dict:
        return {"keep_checkpoints": self.options.keep_checkpoints,
                "deleted": len(self.deleted),
                "pruned_record": PRUNED_RECORD if self.recorded else None,
                "pruned_records_written": self.recorded}


#: The record of every checkpoint retention deleted, beside the checkpoints.
PRUNED_RECORD = "arwen_global_pruned_checkpoints.jsonl"


def pruned_header(path: Path, *, start=None) -> dict:
    """What a deleted checkpoint leaves behind: its step and time (and the
    valid time when the run is dated), its file size, and the hashes its own
    metadata carries (``self_sha256`` over the whole record, the config,
    pins and schema identities, and every array's SHA-256 by name), so a
    replay of the window that produced it can be checked against it byte for
    byte, array by array.  An unreadable header is recorded as unreadable,
    never skipped."""
    path = Path(path)
    header: dict = {"checkpoint": path.name}
    try:
        header["bytes"] = int(path.stat().st_size)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
        header.update({
            "step": int(metadata["step"]),
            "time_s": float(metadata["time_s"]),
            "self_sha256": metadata.get("self_sha256"),
            "config_hash": metadata.get("config_hash"),
            "pins_hash": metadata.get("pins_hash"),
            "schema": metadata.get("schema"),
            "physics_state_schema": metadata.get("physics_state_schema"),
            "run_trackers": metadata.get("run_trackers"),
            "array_sha256": {name: entry.get("sha256") for name, entry
                             in sorted((metadata.get("arrays") or {}).items())},
        })
        if start is not None:
            valid = _utc(start) + timedelta(seconds=float(metadata["time_s"]))
            header["valid_utc"] = valid.isoformat().replace("+00:00", "Z")
    except Exception as exc:  # noqa: BLE001 - a record is never lost to this
        header["unreadable"] = f"{type(exc).__name__}: {exc}"
    return header


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# time means and the daily summary
# ---------------------------------------------------------------------------

def window_key(kind: str, valid: datetime) -> str:
    return f"{valid:%Y-%m}" if kind == "monthly" else f"{valid:%Y-%m-%d}"


def _floating(array) -> bool:
    return np.issubdtype(np.asarray(array).dtype, np.floating) or \
        np.issubdtype(np.asarray(array).dtype, np.complexfloating)


def _averaged(name: str, value) -> bool:
    """The arrays a mean file averages: every floating atmosphere and
    surface array and the 2-D planes of the physics namespace (2 m and
    10 m fields, radiation and flux planes, accumulators).  The namespace's
    3-D working arrays and the second time level are the last instant's,
    so the file stays a complete archive that exports."""
    if not _floating(value):
        return False
    if name.startswith(("atmosphere__", "surface__")):
        return True
    return name.startswith("physics__") and np.ndim(value) <= 2


def _wide(array):
    a = np.asarray(array)
    return a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)


class _Window:
    def __init__(self, key: str):
        self.key = key
        self.count = 0
        self.sums: dict[str, np.ndarray] = {}
        self.last: dict[str, np.ndarray] = {}
        self.first_utc = None
        self.last_utc = None
        self.last_record = None

    def add(self, arrays: dict, valid: datetime, record: dict, *,
            mean_all: bool = False) -> None:
        # ``mean_all``: the daily summary's planes, renamed from their
        # checkpoint names, are all means.
        for name, value in arrays.items():
            if (mean_all and _floating(value)) or _averaged(name, value):
                wide = _wide(value)
                if name in self.sums:
                    self.sums[name] += wide
                else:
                    self.sums[name] = wide.copy()
            else:
                self.last[name] = np.array(value, copy=True)
        self.count += 1
        stamp = valid.isoformat().replace("+00:00", "Z")
        self.first_utc = self.first_utc or stamp
        self.last_utc = stamp
        self.last_record = record


_SUMMARY_MEANS = {
    "t2m_mean_k": "physics__t2", "q2m_mean_kg_kg": "physics__q2",
    "u10_mean_m_s": "physics__u10", "v10_mean_m_s": "physics__v10",
    "olr_mean_w_m2": "physics__olr", "skin_temperature_mean_k": "surface__surface_temperature_k",
    "sea_ice_fraction_mean": "surface__sea_ice_fraction", "snow_cover_mean": "physics__noah_snowc",
}
_PRECIP = ("surface__accumulated_rain_kg_m2", "surface__accumulated_snow_kg_m2",
           "surface__accumulated_graupel_kg_m2", "physics__rainc")


def accumulated_precipitation(arrays: dict) -> np.ndarray | None:
    parts = [arrays.get(name) for name in _PRECIP[:3]]
    if any(p is None for p in parts):
        return None
    total = sum(np.asarray(p, dtype=np.float64) for p in parts)
    if arrays.get(_PRECIP[3]) is not None:
        total = total + np.asarray(arrays[_PRECIP[3]], dtype=np.float64)
    return total


def _extreme(store: dict, name: str, plane, combine) -> None:
    plane = np.asarray(plane, dtype=np.float64)
    store[name] = plane.copy() if name not in store else combine(store[name], plane)


class SummaryGeometry:
    """What the daily summary's derived planes need besides a checkpoint's
    arrays: a host (numpy, float64, streaming) synthesis on the run's own
    Gaussian grid, the hybrid coefficients and the terrain geopotential.
    Everything it computes reads host copies the checkpoint writer already
    made; nothing reaches the model or the card."""

    def __init__(self, transform, a_half_pa, b_half, surface_geopotential):
        self.transform = transform
        self.grid = transform.grid
        self.a_half = np.asarray(a_half_pa, dtype=np.float64)
        self.b_half = np.asarray(b_half, dtype=np.float64)
        self.surface_geopotential = np.asarray(surface_geopotential, dtype=np.float64)

    @classmethod
    def for_run(cls, cfg, transform, surface_geopotential) -> "SummaryGeometry":
        from .spectral.transform import SphericalHarmonicTransform

        grid = transform.grid
        host = SphericalHarmonicTransform.create(
            int(cfg.truncation), nlat=int(grid.nlat), nlon=int(grid.nlon),
            dealias_factor=float(cfg.dealias_factor), radius_m=float(grid.radius_m),
            backend="numpy", precision="float64", streaming=True)
        if tuple(host.grid.shape) != tuple(grid.shape) or not np.allclose(
                host.grid.latitude_deg, grid.latitude_deg, rtol=0.0, atol=1.0e-9):
            raise ValueError("the daily summary's host synthesis is not on the run's grid")
        return cls(host, cfg.a_half_pa, cfg.b_half, surface_geopotential)

    def synthesize(self, coefficients) -> np.ndarray:
        return np.asarray(self.transform.inverse(
            np.asarray(coefficients).astype(np.complex128)), dtype=np.float64)

    def surface_pressure(self, arrays: dict) -> np.ndarray | None:
        lnps = arrays.get("atmosphere__log_surface_pressure")
        return None if lnps is None else np.exp(self.synthesize(lnps))


class InstantPlanes:
    """One output instant's derived planes: ``means`` summed into the day's
    mean, ``mins`` and ``maxs`` folded into the day's extremes."""

    def __init__(self):
        self.means: dict[str, np.ndarray] = {}
        self.mins: dict[str, np.ndarray] = {}
        self.maxs: dict[str, np.ndarray] = {}


#: Surface stress names (the daily summary) and their rule.  N/m2, the
#: force per unit area the atmosphere exerts on the surface, positive
#: eastward and northward (along the wind): the sign an ocean model's
#: momentum forcing takes (ERA5's eastward/northward turbulent surface
#: stress has it too).
SURFACE_STRESS_NAMES = ("surface_stress_eastward_mean_n_m2",
                        "surface_stress_northward_mean_n_m2")


def surface_stress(arrays: dict, ps) -> tuple[np.ndarray, np.ndarray] | None:
    """rho u*^2 along the 10 m wind, from planes one surface-layer call
    wrote together: the friction velocity ``ust`` and the 10 m wind (the
    surface layer's own 10 m wind has the lowest level's direction), with
    the near-surface density ps / (R_d T2 (1 + 0.608 q2)).  The PBL's own
    momentum flux carries a further |V1| / wspd <= 1 (the convective
    velocity in wspd) that a checkpoint does not hold; it matters only in
    light convective winds.  None when a plane is missing."""
    from .constants import DRY_AIR_GAS_CONSTANT, EPSILON

    names = ("physics__ust", "physics__u10", "physics__v10", "physics__t2", "physics__q2")
    if ps is None or any(arrays.get(name) is None for name in names):
        return None
    ust, u10, v10, t2, q2 = (np.asarray(arrays[name], dtype=np.float64) for name in names)
    virtual = t2 * (1.0 + (1.0 / EPSILON - 1.0) * q2)
    rho = np.asarray(ps, dtype=np.float64) / (DRY_AIR_GAS_CONSTANT * virtual)
    speed = np.hypot(u10, v10)
    scale = np.divide(rho * ust * ust, speed, out=np.zeros_like(speed), where=speed > 0.0)
    return scale * u10, scale * v10


#: The storm index's planes: sea-level pressure (mean and minimum over the
#: day's output instants) and 850 hPa relative vorticity (mean, and the
#: largest cyclonic value: vorticity times the sign of latitude, so a
#: cyclone is positive in either hemisphere), with the fraction of the
#: day's instants at which 850 hPa lay below a column's lowest full level.
STORM_LEVEL_PA = 85000.0
SLP_MEAN, SLP_MIN = "sea_level_pressure_mean_pa", "sea_level_pressure_min_pa"
VORTICITY_MEAN = "vorticity_850hpa_mean_per_s"
VORTICITY_CYCLONIC_MAX = "vorticity_850hpa_cyclonic_max_per_s"
BELOW_GROUND = "pressure_850hpa_below_ground_fraction"


def sea_level_pressure(arrays: dict, ps, geometry: SummaryGeometry) -> np.ndarray | None:
    """The scorecard chain of record's reduction (obs_scorecard
    ``mslp_reduction``: ps exp(g z / (R Tmean)), Tmean = T2 + 0.0065 z / 2,
    z the terrain height), from the checkpoint's surface pressure and 2 m
    temperature."""
    t2 = arrays.get("physics__t2")
    if ps is None or t2 is None:
        return None
    from .constants import GRAVITY_M_S2
    from .obs_scorecard import mslp_reduction

    return mslp_reduction(ps, geometry.surface_geopotential / GRAVITY_M_S2, t2)


def vorticity_at_pressure(arrays: dict, ps, geometry: SummaryGeometry,
                          target_pa: float = STORM_LEVEL_PA):
    """Relative vorticity (1/s) at ``target_pa``, linear in ln p between the
    bracketing full levels (``longrun_gates.interpolate_to_pressure``, the
    gates' own rule); a column whose lowest full level lies above the target
    holds its lowest level's value, and the second plane says where.  Only
    the levels from the lowest one lying above the target in every column
    down to the surface are synthesized, which gives every column the same
    bracketing levels the whole column would."""
    vorticity = arrays.get("atmosphere__vorticity")
    if ps is None or vorticity is None:
        return None
    from .longrun_gates import interpolate_to_pressure

    a, b = geometry.a_half, geometry.b_half
    ps = np.asarray(ps, dtype=np.float64)
    deepest = a + b * float(ps.max())
    above_everywhere = np.nonzero(np.sqrt(deepest[:-1] * deepest[1:]) < target_pa)[0]
    first = int(above_everywhere.max()) if above_everywhere.size else 0
    # Two levels at least, so the bracketing pair exists even where every
    # column lies under the target.
    first = max(0, min(first, len(a) - 3))
    p_half = a[first:, None, None] + b[first:, None, None] * ps[None]
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    field = geometry.synthesize(np.asarray(vorticity)[first:])
    value, _over_top = interpolate_to_pressure(np, field, p_full, target_pa,
                                               extrapolate="hold")
    return value, p_full[-1] < target_pa


#: How each derived plane of the daily summary is made, written into every
#: summary's metadata for the planes it holds.
SUMMARY_RULES = {
    SLP_MEAN: ("mean over the day's output instants of ps exp(g z / (R Tmean)), "
               "Tmean = T2 + 0.0065 z / 2 (the scorecard's reduction), Pa"),
    SLP_MIN: "minimum over the day's output instants of the same reduction, Pa",
    VORTICITY_MEAN: ("mean over the day's output instants of relative vorticity at "
                     "850 hPa, linear in ln p between full levels, 1/s; a column "
                     "whose lowest full level is above 850 hPa holds that level's"),
    VORTICITY_CYCLONIC_MAX: ("maximum over the day's output instants of 850 hPa "
                             "relative vorticity times the sign of latitude (cyclonic "
                             "positive in both hemispheres), 1/s"),
    BELOW_GROUND: ("fraction of the day's output instants at which 850 hPa lay below "
                   "the column's lowest full level (its vorticity is that level's)"),
    SURFACE_STRESS_NAMES[0]: (
        "mean over the day's output instants of rho u*^2 u10/|V10| (N/m2, positive "
        "eastward: the force on the surface); rho = ps / (R_d T2 (1 + 0.608 q2))"),
    SURFACE_STRESS_NAMES[1]: (
        "mean over the day's output instants of rho u*^2 v10/|V10| (N/m2, positive "
        "northward: the force on the surface); rho = ps / (R_d T2 (1 + 0.608 q2))"),
}


#: The energy and water terms (the test of the albedo fix).  Sampled at the
#: output instants and averaged: the surface-layer fluxes and the cloud
#: water paths.  Exact interval means from the namespace's radiation
#: accumulators (flux times dt summed over every physics call), over the
#: same interval as the day's precipitation: the last output instant of
#: the day before to the last of the day.
SENSIBLE, LATENT = "sensible_heat_flux_mean_w_m2", "latent_heat_flux_mean_w_m2"
LIQUID_WATER_PATH, ICE_WATER_PATH = ("liquid_water_path_mean_kg_m2",
                                     "ice_water_path_mean_kg_m2")
#: (summary name, accumulator) for the exact interval means.
RADIATION_INTERVAL_TERMS = (
    ("toa_incoming_shortwave_w_m2", "acc_sw_down_top_j_m2"),
    ("toa_reflected_shortwave_w_m2", "acc_sw_up_top_j_m2"),
    ("toa_outgoing_longwave_w_m2", "acc_lw_up_top_j_m2"),
    ("surface_downward_shortwave_w_m2", "acc_sw_down_surface_j_m2"),
    ("surface_upward_shortwave_w_m2", "acc_sw_up_surface_j_m2"),
    ("surface_downward_longwave_w_m2", "acc_lw_down_surface_j_m2"),
    ("surface_upward_longwave_w_m2", "acc_lw_up_surface_j_m2"),
    ("cloud_cover_fraction", "acc_total_cloud_cover_s"),
)
TOA_ABSORBED, TOA_NET = "toa_absorbed_shortwave_w_m2", "toa_net_w_m2"
SURFACE_NET_RADIATION = "surface_net_radiation_w_m2"
#: What the summary does not write, and why.
NOT_WRITTEN = {
    "clear_sky_radiation": (
        "not written: the radiation holds no clear-sky planes, so clear-sky "
        "fluxes would need a second radiation pass inside every radiation call, "
        "which is neither cheap nor outside the step this summary may not touch"),
}


def surface_flux_planes(arrays: dict) -> dict[str, np.ndarray]:
    """The held surface-layer fluxes at the instant, upward positive (WRF
    HFX; L_v QFX, the latent heat LR-4 books)."""
    from .constants import LATENT_HEAT_VAPORIZATION

    out = {}
    if arrays.get("physics__hfx") is not None:
        out[SENSIBLE] = np.asarray(arrays["physics__hfx"], dtype=np.float64)
    if arrays.get("physics__qfx") is not None:
        out[LATENT] = LATENT_HEAT_VAPORIZATION * np.asarray(
            arrays["physics__qfx"], dtype=np.float64)
    return out


def water_paths(arrays: dict, ps, geometry: SummaryGeometry) -> dict[str, np.ndarray]:
    """Cloud liquid (qc) and cloud ice (qi) water paths at the instant,
    int q dp/g in kg/m2 over the run's own hybrid layers; rain, snow and
    graupel are not cloud and are not in them."""
    from .constants import GRAVITY_M_S2

    if ps is None:
        return {}
    p_half = (geometry.a_half[:, None, None]
              + geometry.b_half[:, None, None] * np.asarray(ps, dtype=np.float64)[None])
    dp_g = (p_half[1:] - p_half[:-1]) / GRAVITY_M_S2
    out = {}
    for name, species in ((LIQUID_WATER_PATH, "atmosphere__qc"),
                          (ICE_WATER_PATH, "atmosphere__qi")):
        q = arrays.get(species)
        if q is not None and np.shape(q) == dp_g.shape:
            out[name] = np.sum(np.asarray(q, dtype=np.float64) * dp_g, axis=0)
    return out


def radiation_accumulators(arrays: dict, record: dict | None, time_s: float) -> dict | None:
    """The accumulators and their seconds at one instant, or None when the
    namespace or the record does not carry them."""
    seconds = ((record or {}).get("physics_metadata") or {}).get("radiation_accumulated_s")
    planes = {acc: arrays.get(f"physics__{acc}") for _name, acc in RADIATION_INTERVAL_TERMS}
    if seconds is None or any(p is None for p in planes.values()):
        return None
    return {"time_s": float(time_s), "seconds": float(seconds),
            "planes": {acc: np.asarray(p, dtype=np.float64) for acc, p in planes.items()}}


def interval_means(before: dict, last: dict) -> dict | None:
    """The exact interval-mean fluxes (W/m2; cloud cover as a fraction)
    between two accumulator records, with TOA absorbed and net and the
    surface net radiation; None when no radiation time elapsed."""
    seconds = float(last["seconds"]) - float(before["seconds"])
    if seconds <= 0.0:
        return None
    planes = {name: (last["planes"][acc] - before["planes"][acc]) / seconds
              for name, acc in RADIATION_INTERVAL_TERMS}
    planes[TOA_ABSORBED] = (planes["toa_incoming_shortwave_w_m2"]
                            - planes["toa_reflected_shortwave_w_m2"])
    planes[TOA_NET] = planes[TOA_ABSORBED] - planes["toa_outgoing_longwave_w_m2"]
    planes[SURFACE_NET_RADIATION] = (
        planes["surface_downward_shortwave_w_m2"] - planes["surface_upward_shortwave_w_m2"]
        + planes["surface_downward_longwave_w_m2"] - planes["surface_upward_longwave_w_m2"])
    return {"planes": planes, "from_time_s": float(before["time_s"]),
            "to_time_s": float(last["time_s"]), "seconds": seconds}


def _accum_header(accum: dict | None) -> dict | None:
    if accum is None:
        return None
    return {"time_s": accum["time_s"], "seconds": accum["seconds"],
            "names": sorted(accum["planes"])}


def summary_global_means(grid, out: dict, interval: dict | None) -> dict[str, float]:
    """Area means on the run's Gaussian quadrature of every plane the day's
    file holds, with the planetary albedo (reflected over incoming global
    means), the surface's net energy (net radiation less H and LE: exact
    radiation, sampled fluxes) and the precipitation rate over the
    radiation interval."""
    means = {name: float(grid.global_mean(np.asarray(plane, dtype=np.float64)))
             for name, plane in sorted(out.items())}
    incoming = means.get("toa_incoming_shortwave_w_m2")
    if incoming:
        means["planetary_albedo"] = means["toa_reflected_shortwave_w_m2"] / incoming
    if all(name in means for name in (SURFACE_NET_RADIATION, SENSIBLE, LATENT)):
        means["surface_net_energy_w_m2"] = (means[SURFACE_NET_RADIATION]
                                            - means[SENSIBLE] - means[LATENT])
    if "precipitation_kg_m2" in means and interval is not None:
        means["precipitation_mm_day"] = (means["precipitation_kg_m2"]
                                         * 86400.0 / interval["seconds"])
    return means


SUMMARY_RULES.update({
    SENSIBLE: "mean over the day's output instants of the held HFX, upward positive, W/m2",
    LATENT: "mean over the day's output instants of L_v times the held QFX, upward "
            "positive, W/m2",
    LIQUID_WATER_PATH: "mean over the day's output instants of int qc dp/g (cloud "
                       "liquid only), kg/m2",
    ICE_WATER_PATH: "mean over the day's output instants of int qi dp/g (cloud ice "
                    "only), kg/m2",
    **{name: (f"exact mean over the radiation interval of the {acc} accumulator's "
              "rate (difference over the accumulated radiation seconds)"
              + (", W/m2" if name != "cloud_cover_fraction" else ", a fraction"))
       for name, acc in RADIATION_INTERVAL_TERMS},
    TOA_ABSORBED: "incoming less reflected shortwave at the top, exact interval mean, W/m2",
    TOA_NET: "absorbed shortwave less outgoing longwave at the top, exact interval "
             "mean, W/m2",
    SURFACE_NET_RADIATION: "net shortwave plus net longwave at the surface, downward "
                           "positive, exact interval mean, W/m2",
})


def instant_planes(arrays: dict, geometry: SummaryGeometry | None) -> InstantPlanes:
    """The daily summary's planes derived from one instant's checkpoint
    arrays (host, float64).  A plane whose inputs the checkpoint does not
    carry is left out, never zero-filled."""
    out = InstantPlanes()
    out.means.update(surface_flux_planes(arrays))
    if geometry is None:
        return out
    ps = geometry.surface_pressure(arrays)
    out.means.update(water_paths(arrays, ps, geometry))
    stress = surface_stress(arrays, ps)
    if stress is not None:
        out.means.update(zip(SURFACE_STRESS_NAMES, stress))
    slp = sea_level_pressure(arrays, ps, geometry)
    if slp is not None:
        out.means[SLP_MEAN] = out.mins[SLP_MIN] = slp
    storm = vorticity_at_pressure(arrays, ps, geometry)
    if storm is not None:
        vorticity, below = storm
        hemisphere = np.where(np.asarray(geometry.grid.latitude_deg) >= 0.0, 1.0, -1.0)
        out.means[VORTICITY_MEAN] = vorticity
        out.maxs[VORTICITY_CYCLONIC_MAX] = vorticity * hemisphere[:, None]
        out.means[BELOW_GROUND] = below.astype(np.float64)
    return out


class LongRunProducts:
    """The mean-state windows and the daily summary of one run segment."""

    def __init__(self, outdir: Path, options, *, start, dt_s: float, config_hash: str,
                 physics_state_schema: str, semi_implicit_scheme: str, integrator: str,
                 geometry: "SummaryGeometry | None" = None):
        self.outdir = Path(outdir)
        # What the summary's derived planes need besides the checkpoint's
        # arrays (a host synthesis, the hybrid coefficients, the terrain);
        # without it only the planes the checkpoint carries are summarised.
        self.geometry = geometry
        self.geometry_note = ("on" if geometry is not None
                              else "off: no host synthesis was given")
        self.options = options
        self.start = _utc(start) if start is not None else None
        self.dt_s = float(dt_s)
        self.config_hash = config_hash
        self.schema = physics_state_schema
        self.semi_implicit_scheme = semi_implicit_scheme
        self.integrator = integrator
        self.windows: dict[str, _Window] = {}
        self.summary: _Window | None = None
        self.summary_mins: dict[str, np.ndarray] = {}
        self.summary_maxs: dict[str, np.ndarray] = {}
        self.precip_before: np.ndarray | None = None
        self.precip_last: np.ndarray | None = None
        # The radiation accumulators at the day's last instant and at the
        # last instant of the day before ({"time_s", "seconds", "planes"}).
        self.accum_before: dict | None = None
        self.accum_last: dict | None = None
        self.written: list[str] = []
        self.resumed_from = None

    @property
    def enabled(self) -> bool:
        return bool(self.options.mean_windows) or self.options.daily_summary

    def _valid(self, time_s: float) -> datetime:
        return self.start + timedelta(seconds=float(time_s))

    def add(self, arrays: dict, record: dict) -> None:
        """One output step's checkpoint arrays (host) and record."""
        if not self.enabled:
            return
        valid = self._valid(record["time_s"])
        for kind in self.options.mean_windows:
            key = window_key(kind, valid)
            window = self.windows.get(kind)
            if window is not None and window.key != key:
                self._write_mean(kind, window)
                window = None
            if window is None:
                window = self.windows[kind] = _Window(key)
            window.add(arrays, valid, record)
        if self.options.daily_summary:
            self._add_summary(arrays, valid, record)

    def _add_summary(self, arrays: dict, valid: datetime, record: dict | None = None) -> None:
        key = window_key("daily", valid)
        if self.summary is not None and self.summary.key != key:
            self._write_summary()
        if self.summary is None:
            self.summary = _Window(key)
            self.summary_mins, self.summary_maxs = {}, {}
        planes = {name: arrays[src] for name, src in _SUMMARY_MEANS.items()
                  if arrays.get(src) is not None}
        derived = instant_planes(arrays, self.geometry)
        planes.update(derived.means)
        self.summary.add(planes, valid, {}, mean_all=True)
        mins, maxs = dict(derived.mins), dict(derived.maxs)
        t2 = arrays.get("physics__t2")
        if t2 is not None:
            mins["t2m_min_k"] = maxs["t2m_max_k"] = np.asarray(t2, dtype=np.float64)
        for name, plane in mins.items():
            _extreme(self.summary_mins, name, plane, np.minimum)
        for name, plane in maxs.items():
            _extreme(self.summary_maxs, name, plane, np.maximum)
        precip = accumulated_precipitation(arrays)
        if precip is not None:
            self.precip_last = precip
        accumulators = radiation_accumulators(arrays, record, float(
            (valid - self.start).total_seconds()))
        if accumulators is not None:
            self.accum_last = accumulators

    def _write_summary(self) -> None:
        window = self.summary
        out = {name: (value / window.count).astype(np.float32)
               for name, value in window.sums.items()}
        for extremes in (self.summary_mins, self.summary_maxs):
            out.update({name: value.astype(np.float32) for name, value in extremes.items()})
        if self.precip_last is not None:
            before = self.precip_before if self.precip_before is not None else None
            if before is not None:
                out["precipitation_kg_m2"] = (self.precip_last - before).astype(np.float32)
            self.precip_before = self.precip_last
        interval = None
        if self.accum_last is not None:
            if self.accum_before is not None:
                interval = interval_means(self.accum_before, self.accum_last)
                if interval is not None:
                    out.update({name: plane.astype(np.float32)
                                for name, plane in interval["planes"].items()})
            self.accum_before = self.accum_last
        meta = {"schema": "gpuwm.arwen-global-daily-summary/v1", "day": window.key,
                "samples": window.count, "first_utc": window.first_utc,
                "last_utc": window.last_utc, "config_hash": self.config_hash,
                "precipitation": (
                    "the day's total: the last output instant of the day less the last "
                    "of the day before" if "precipitation_kg_m2" in out else
                    "absent: the day before is not in this run's record")}
        meta["rules"] = {name: rule for name, rule in SUMMARY_RULES.items() if name in out}
        meta["derived_planes"] = self.geometry_note
        meta["radiation_interval"] = (
            {key: interval[key] for key in ("from_time_s", "to_time_s", "seconds")}
            if interval is not None else
            "absent: the accumulators of the day before are not in this run's record")
        meta["not_written"] = NOT_WRITTEN
        if self.geometry is not None:
            meta["global_means"] = summary_global_means(self.geometry.grid, out, interval)
        path = self.outdir / f"arwen_global_daily_{window.key}.npz"
        _atomic_savez(path, metadata=np.array(json.dumps(meta, sort_keys=True)),
                      **{k: np.ascontiguousarray(v) for k, v in out.items()})
        self.written.append(path.name)
        self.summary = None

    def _write_mean(self, kind: str, window: _Window) -> Path:
        from .checkpoint import write_checkpoint_arrays

        arrays = {}
        for name, total in window.sums.items():
            reference = window.last_record["dtypes"].get(name)
            arrays[name] = (total / window.count).astype(reference or total.dtype)
        arrays.update(window.last)
        record = window.last_record
        physics_metadata = dict(record["physics_metadata"])
        physics_metadata[MEAN_KEY] = {
            "window": kind, "key": window.key, "samples": window.count,
            "first_sample_utc": window.first_utc, "last_sample_utc": window.last_utc,
            "rule": "the mean over the window's output instants (float64 sums) of "
                    "every floating atmosphere and surface array and of the physics "
                    "namespace's 2-D planes; integer arrays, the namespace's 3-D "
                    "working arrays and the second time level are the last "
                    "instant's; a field derived nonlinearly from these means is a "
                    "function of means, not a mean of the function",
        }
        path = self.outdir / f"arwen_global_mean_{window.key}.npz"
        write_checkpoint_arrays(
            path, arrays, step=int(record["step"]), time_s=float(record["time_s"]),
            physics_state_schema=self.schema, physics_metadata=physics_metadata,
            config_hash=self.config_hash, trackers=dict(record.get("trackers") or {}),
            semi_implicit_scheme=self.semi_implicit_scheme, integrator=self.integrator)
        self.written.append(path.name)
        return path

    def finish(self, *, run_complete: bool) -> None:
        """At the run's last step: close every window (a window the run
        ends inside is written with the samples it has).  A segment that
        ends earlier leaves its windows open in the last sidecar."""
        if not run_complete:
            return
        for kind, window in list(self.windows.items()):
            self._write_mean(kind, window)
        self.windows = {}
        if self.summary is not None:
            self._write_summary()

    # -- the sidecar ---------------------------------------------------------

    def write_sidecar(self, checkpoint: Path) -> None:
        if not self.enabled:
            return
        payload: dict[str, np.ndarray] = {}
        header = {"windows": {}, "summary": None,
                  "precip_before": self.precip_before is not None,
                  "precip_last": self.precip_last is not None,
                  "accum_before": _accum_header(self.accum_before),
                  "accum_last": _accum_header(self.accum_last)}
        for kind, w in self.windows.items():
            header["windows"][kind] = {
                "key": w.key, "count": w.count, "first_utc": w.first_utc,
                "last_utc": w.last_utc, "sums": sorted(w.sums), "last": sorted(w.last),
                "record": w.last_record}
            for name, value in w.sums.items():
                payload[f"{kind}__sum__{name}"] = value
            for name, value in w.last.items():
                payload[f"{kind}__last__{name}"] = value
        if self.summary is not None:
            s = self.summary
            header["summary"] = {"key": s.key, "count": s.count, "first_utc": s.first_utc,
                                 "last_utc": s.last_utc, "sums": sorted(s.sums),
                                 "mins": sorted(self.summary_mins),
                                 "maxs": sorted(self.summary_maxs)}
            for name, value in s.sums.items():
                payload[f"summary__sum__{name}"] = value
            for name, value in self.summary_mins.items():
                payload[f"summary__min__{name}"] = value
            for name, value in self.summary_maxs.items():
                payload[f"summary__max__{name}"] = value
        if self.precip_before is not None:
            payload["precip_before"] = self.precip_before
        if self.precip_last is not None:
            payload["precip_last"] = self.precip_last
        for which, accum in (("accum_before", self.accum_before),
                             ("accum_last", self.accum_last)):
            for name, value in (accum or {}).get("planes", {}).items():
                payload[f"{which}__{name}"] = value
        _atomic_savez(sidecar_of(checkpoint),
                      header=np.array(json.dumps(header, sort_keys=True)), **payload)

    def resume(self, checkpoint: Path) -> str:
        """Re-open the windows a restart checkpoint's sidecar carries."""
        if not self.enabled:
            return "off"
        path = sidecar_of(checkpoint)
        if not path.is_file():
            self.resumed_from = None
            return (f"no sidecar beside {Path(checkpoint).name}: the windows open at "
                    "the restart and their files say how many samples they hold")
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            for kind, h in header["windows"].items():
                if kind not in self.options.mean_windows:
                    continue
                w = _Window(h["key"])
                w.count, w.first_utc, w.last_utc = h["count"], h["first_utc"], h["last_utc"]
                w.last_record = h["record"]
                w.sums = {n: np.array(data[f"{kind}__sum__{n}"]) for n in h["sums"]}
                w.last = {n: np.array(data[f"{kind}__last__{n}"]) for n in h["last"]}
                self.windows[kind] = w
            s = header.get("summary")
            if s is not None and self.options.daily_summary:
                w = _Window(s["key"])
                w.count, w.first_utc, w.last_utc = s["count"], s["first_utc"], s["last_utc"]
                w.sums = {n: np.array(data[f"summary__sum__{n}"]) for n in s["sums"]}
                self.summary = w
                self.summary_mins = {n: np.array(data[f"summary__min__{n}"]) for n in s["mins"]}
                self.summary_maxs = {n: np.array(data[f"summary__max__{n}"])
                                     for n in s.get("maxs", s["mins"])}
                if "maxs" not in s:
                    # A sidecar written before the extremes were named by
                    # their output: the 2 m pair under "t2m".
                    self.summary_mins = {("t2m_min_k" if n == "t2m" else n): v
                                         for n, v in self.summary_mins.items()}
                    self.summary_maxs = {("t2m_max_k" if n == "t2m" else n): v
                                         for n, v in self.summary_maxs.items()}
            if header.get("precip_before"):
                self.precip_before = np.array(data["precip_before"])
            if header.get("precip_last"):
                self.precip_last = np.array(data["precip_last"])
            for which in ("accum_before", "accum_last"):
                h = header.get(which)
                if h is not None:
                    setattr(self, which, {
                        "time_s": h["time_s"], "seconds": h["seconds"],
                        "planes": {n: np.array(data[f"{which}__{n}"]) for n in h["names"]}})
        self.resumed_from = path.name
        return f"resumed from {path.name}"

    def receipt(self) -> dict:
        return {"mean_state": self.options.mean_state,
                "daily_summary": self.options.daily_summary,
                "written": list(self.written), "resumed_from": self.resumed_from,
                "derived_planes": self.geometry_note}


def _atomic_savez(path: Path, **arrays) -> None:
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def refuse_time_mean(metadata: dict, label: str) -> None:
    if MEAN_KEY in (metadata.get("physics_metadata") or {}):
        raise ValueError(f"{label} is a time-mean file, not a model state; it cannot "
                         "be a restart")


def time_mean_attrs(metadata: dict) -> dict:
    block = (metadata.get("physics_metadata") or {}).get(MEAN_KEY)
    if block is None:
        return {}
    return {"ARWEN_TIME_MEAN": json.dumps(block, sort_keys=True)}


# ---------------------------------------------------------------------------
# the estimate and the launch gate
# ---------------------------------------------------------------------------

def checkpoint_bytes(nlat: int, nlon: int, nlev: int) -> int:
    """Scaled from the one measured checkpoint by grid area and levels."""
    lat0, lon0, lev0 = MEASURED_CHECKPOINT_SHAPE
    return int(MEASURED_CHECKPOINT_BYTES * (nlat * nlon) / (lat0 * lon0) * nlev / lev0)


def estimate(cfg, *, start_s: float = 0.0, until_s: float | None = None) -> dict:
    """Disk and wall estimate of the span ``start_s`` .. ``until_s``."""
    from .spectral.grid import GaussianGrid

    grid = GaussianGrid.create(int(cfg.truncation), dealias_factor=float(cfg.dealias_factor))
    nlat, nlon = int(grid.nlat), int(grid.nlon)
    nlev = len(cfg.a_half_pa) - 1
    end = float(cfg.duration_s if until_s is None else until_s)
    span = max(0.0, end - float(start_s))
    out = cfg.output
    per = checkpoint_bytes(nlat, nlon, nlev)
    outputs = int(math.floor(span / float(cfg.output_interval_s) + 1.0e-9)) + 1
    days = span / 86400.0
    if out.keep_newest is None:
        kept = outputs
    else:
        months = int(math.ceil(days / 28.0)) + 1 if out.keep_month_ends else 0
        kept = min(outputs, out.keep_newest + 1 + months)
    windows = out.mean_windows
    sidecar = 2 * per * len(windows)
    mean_files = 0
    if "monthly" in windows:
        mean_files += int(math.ceil(days / 28.0)) + 1
    if "daily" in windows:
        mean_files += int(math.ceil(days)) + 1
    summary = (DAILY_SUMMARY_PLANES * nlat * nlon * 4 * (int(math.ceil(days)) + 1)
               if out.daily_summary else 0)
    sidecar_count = kept if windows or out.daily_summary else 0
    # The summary's float64 sums plus the radiation accumulators of the day
    # before and of the day's last instant.
    sidecar_each = sidecar + ((DAILY_SUMMARY_PLANES + 2 * len(RADIATION_INTERVAL_TERMS))
                              * nlat * nlon * 8 if out.daily_summary else 0)
    disk = kept * per + sidecar_count * sidecar_each + mean_files * per + summary
    rate = MINUTES_PER_DAY.get(int(cfg.truncation))
    return {
        "span_s": span, "checkpoint_bytes_each": per, "checkpoints_on_disk": kept,
        "sidecar_bytes_each": sidecar_each if sidecar_count else 0,
        "mean_files": mean_files, "daily_summary_bytes": summary,
        "disk_bytes": int(disk),
        "wall_s": None if rate is None else days * rate * 60.0,
        "basis": (
            "checkpoint bytes scaled by grid area and levels from the one measured "
            f"T383 L40 native checkpoint ({MEASURED_CHECKPOINT_BYTES:,} B); mean files "
            "as large as a checkpoint; sidecars hold float64 sums (twice a checkpoint "
            "per open window); retention keeps the newest N plus the one being "
            "written plus month-ends; wall from the measured per-day cost "
            "(T255 4.3 min, T383 12 min), none for other truncations; export and "
            "rebuild time excluded"),
    }


def launch_gate(cfg, outdir: Path, *, start_s: float, until_s: float | None) -> dict | None:
    """Refuse a launch whose estimate exceeds the free disk less the
    reserve.  Only for runs that state long-run outputs or run longer than
    384 h; returns the estimate it checked, or None."""
    from .boundary_options import LONG_RUN_THRESHOLD_S

    out = cfg.output
    stated = out.keep_newest is not None or out.mean_windows or out.daily_summary
    if not stated and float(cfg.duration_s) <= LONG_RUN_THRESHOLD_S:
        return None
    figure = estimate(cfg, start_s=start_s, until_s=until_s)
    target = Path(outdir)
    probe = target if target.exists() else target.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    reserve = int(out.disk_reserve_gb * 1.0e9)
    figure["free_bytes"] = int(free)
    figure["reserve_bytes"] = reserve
    if figure["disk_bytes"] > free - reserve:
        raise ValueError(
            f"this run would write about {figure['disk_bytes'] / 1e9:.1f} GB to {target} "
            f"and {free / 1e9:.1f} GB is free, {out.disk_reserve_gb:g} GB of it kept in "
            "reserve: set [output] keep_checkpoints = \"newest:2\", lengthen the output "
            "interval, or free space")
    return figure


__all__ = [
    "CheckpointRetention", "LongRunProducts", "checkpoint_bytes", "checkpoints_to_prune",
    "estimate", "launch_gate", "refuse_time_mean", "sidecar_of", "time_mean_attrs",
    "window_key",
]
