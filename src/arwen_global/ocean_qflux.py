"""The slab ocean's heat-flux correction: calibrated, stored and read.

Design: docs/ARWEN_GLOBAL_OCEAN.md.  The correction is derived from the
model's own run with the SST prescribed (``[ocean] mode = "climatology"``
with ``qflux_calibration = true``).  During that run the coupler computes,
every step and on every ocean column, the net heat F_sys the atmosphere
puts into the ocean-and-ice-mass system (exactly the flux a slab run would
consume) and :func:`deposit` adds F_sys dt into the two calendar-month bins
the climatology's month weights bracket the instant with, weighted by those
weights; the weights themselves (in seconds) go into a second array.  Both
arrays ride in the physics state, so a restarted calibration accumulates
exactly what the straight one does.

:func:`build_qflux` turns the last checkpoint of one or more calibration
runs into a file (:data:`QFLUX_SCHEMA`): F_cal(month) = sum / weight per
column, NaN where a month has no weight on a column, with each month's
coverage.  The file stores F_cal, not Q: Q is formed at run time from the
climatology's tendency and the run's own mixed-layer depth
(ocean_slab.qflux), so a change of depth bounds needs no new calibration.
A slab run refuses a file whose grid, physics identity or climatology is
not the run's, and a run that reaches a month the file does not cover.

What a calibration checkpoint must carry.  Its physics metadata's
``surface_boundary`` block (arwen_global.ocean_rules.IDENTITY_KEY) holds
the run's ``climatology_sha256`` and ``start_utc``, as every rule block
does, and under ``qflux_calibration`` the coupler's record
(arwen_global.ocean_coupler.SlabProvider.identity_block), of which this
module reads :data:`CALIBRATION_KEYS`: ``physics_identity``
(arwen_global.ocean_coupler.physics_identity, the same function a slab run
passes its own through to :meth:`QFluxFile.check`), ``grid`` as ``[nlat,
nlon]`` (a Gaussian grid is fixed by its shape) and ``spinup_days``.
Every key of the record must agree between runs whose sums are added.  Its
physics arrays hold the two accumulators under the names
arwen_global.ocean_coupler gives them (``QCAL_SUM``, ``QCAL_WEIGHT``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

QFLUX_SCHEMA = "gpuwm.arwen-global-slab-qflux/v1"
#: The keys of a calibration run's ``qflux_calibration`` identity record
#: that :func:`build_qflux` reads.
CALIBRATION_KEYS = ("physics_identity", "grid", "spinup_days")
#: Days at the start of a calibration run that are not deposited (the
#: atmosphere's adjustment to the prescribed SST).  The default of the
#: ``[ocean] qflux_calibration_spinup_days`` key.
DEFAULT_SPINUP_DAYS = 2.0
MONTHS = 12
SECONDS_PER_DAY = 86400.0
#: The fixed timestamp every member of a written file carries, so two builds
#: from the same inputs are the same bytes (as surface_climatology writes).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December")


class QFluxRefused(ValueError):
    """A calibration file or checkpoint that may not be used here; the
    message names what differs."""


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plain(value):
    """``value`` as plain JSON (tuples to lists), so two records compare equal
    exactly when their canonical JSON does."""
    return json.loads(_canonical(value))


def _sha256_file(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _month_list(months) -> str:
    return ", ".join(_MONTH_NAMES[m] for m in sorted(months))


def _differences(a: dict, b: dict) -> list[str]:
    keys = sorted(set(a) | set(b))
    return [key for key in keys if _canonical(a.get(key)) != _canonical(b.get(key))]


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QFluxFile:
    """A loaded calibration file.

    ``flux_w_m2`` and ``coverage_s`` are (12, nlat, nlon) float64; month
    index 0 is January.  ``provenance`` carries the schema, the grid, the
    physics identity of the calibration runs
    (arwen_global.physics.registry.identity_of_global_physics_options),
    their config hashes, the climatology SHA-256, the ice reference
    thicknesses, the spin-up and the package version."""

    path: Path
    sha256: str
    flux_w_m2: np.ndarray
    coverage_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    provenance: dict
    _device: dict = field(default_factory=dict, repr=False, compare=False)

    def _plane(self, month: int, xp):
        """Month ``month`` of F_cal and its finite mask on ``xp``, kept on the
        device once moved (twelve planes at most)."""
        key = (getattr(xp, "__name__", str(id(xp))), int(month))
        if key not in self._device:
            host = self.flux_w_m2[month]
            finite = np.isfinite(host)
            self._device[key] = (xp.asarray(np.where(finite, host, 0.0), dtype=xp.float64),
                                 xp.asarray(finite))
        return self._device[key]

    def at(self, when, xp=np):
        """F_cal at ``when``, W/m2, interpolated with the climatology's month
        weights (surface_climatology.month_weights); NaN columns read 0.

        A column with a value in only one of the two bracketing months reads
        that month's value (a weight on a NaN would read 0 for the missing
        month and bias the column toward 0); one with neither reads 0."""
        from .surface_climatology import month_weights

        i0, i1, w = month_weights(when)
        f0, ok0 = self._plane(i0, xp)
        f1, ok1 = self._plane(i1, xp)
        both = f0 * (1.0 - w) + f1 * w
        return xp.where(ok0 & ok1, both, xp.where(ok0, f0, xp.where(ok1, f1, 0.0)))

    def coverage_days(self) -> list[float]:
        """Each month's largest coverage over the columns, in days, January
        first (the receipt's month line)."""
        return [round(float(np.max(self.coverage_s[m])) / SECONDS_PER_DAY, 6)
                for m in range(MONTHS)]

    def check(self, *, latitude_deg, longitude_deg, physics_identity: dict,
              climatology_sha256: str, months: set[int]) -> None:
        """Refuse, naming what differs: another grid, another physics
        identity, another climatology, or a needed month (0-11) with no
        coverage on any ocean column."""
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        if lat.shape != self.latitude_deg.shape or lon.shape != self.longitude_deg.shape \
                or not np.allclose(lat, self.latitude_deg, rtol=0.0, atol=1.0e-9) \
                or not np.allclose(lon, self.longitude_deg, rtol=0.0, atol=1.0e-9):
            raise QFluxRefused(
                f"Q-flux file {self.path} is on a {self.latitude_deg.size}x"
                f"{self.longitude_deg.size} grid, not this run's {lat.size}x{lon.size} "
                "grid; calibrate on this run's grid")
        theirs = self.provenance.get("physics_identity") or {}
        ours = _plain(physics_identity)
        differ = _differences(theirs, ours)
        if differ:
            raise QFluxRefused(
                f"Q-flux file {self.path} was calibrated with another physics: "
                + ", ".join(differ) + " differ (" + "; ".join(
                    f"{key}: file {_canonical(theirs.get(key))}, run "
                    f"{_canonical(ours.get(key))}" for key in differ)
                + "); a physics change needs a new calibration")
        if self.provenance.get("climatology_sha256") != climatology_sha256:
            raise QFluxRefused(
                f"Q-flux file {self.path} was calibrated against the climatology with "
                f"SHA-256 {self.provenance.get('climatology_sha256')}, and this run's "
                f"climatology is {climatology_sha256}; Q is formed from the climatology "
                "the calibration prescribed, so the two must be the same file")
        uncovered = sorted(m for m in months if not float(np.max(self.coverage_s[int(m)])) > 0.0)
        if uncovered:
            covered = [m for m in range(MONTHS) if float(np.max(self.coverage_s[m])) > 0.0]
            raise QFluxRefused(
                f"Q-flux file {self.path} has no coverage in {_month_list(uncovered)}, "
                f"which this run reaches; it covers {_month_list(covered) or 'no month'}. "
                "Calibrate over those months (a calibration may be made of several runs)")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def deposit(sum_j_m2, weight_s, *, flux_w_m2, columns, when, dt_s: float, xp) -> None:
    """Add ``flux_w_m2 * dt_s`` (and ``dt_s``) into the two month bins of
    ``when`` with the month weights, on ``columns`` only, in place.

    ``sum_j_m2`` and ``weight_s`` are (12, nlat, nlon) float64."""
    from .surface_climatology import month_weights

    i0, i1, w = month_weights(when)
    dt = float(dt_s)
    columns = xp.asarray(columns, dtype=bool)
    energy = xp.where(columns, xp.asarray(flux_w_m2, dtype=xp.float64) * dt, 0.0)
    span = xp.where(columns, dt, 0.0)
    for month, weight in ((i0, 1.0 - w), (i1, w)):
        sum_j_m2[month] += energy * weight
        weight_s[month] += span * weight


def months_needed(start, duration_s: float) -> set[int]:
    """The month indices (0-11) whose month weight is above zero at some
    instant of [start, start + duration_s]."""
    from .ocean_slab import month_bracket
    from .surface_climatology import _utc

    duration = float(duration_s)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"a run's duration is a non-negative number of seconds, "
                         f"not {duration_s!r}")
    now = _utc(start)
    end = now + timedelta(seconds=duration)
    months: set[int] = set()
    while True:
        i0, i1, t0, t1 = month_bracket(now)
        # Every instant of [t0, t1) gives i0 a weight above zero; i1's weight
        # is above zero at every instant after t0.
        months.add(i0)
        if end > t0:
            months.add(i1)
        if end < t1:
            return months
        now = t1


def _calibration_arrays():
    from .ocean_coupler import QCAL_SUM, QCAL_WEIGHT

    return f"physics__{QCAL_SUM}", f"physics__{QCAL_WEIGHT}"


def _read_calibration(path) -> dict:
    """One checkpoint's accumulators and identity, verified."""
    from .checkpoint import read_checkpoint
    from .ocean_rules import IDENTITY_KEY

    metadata, arrays = read_checkpoint(path)
    sum_name, weight_name = _calibration_arrays()
    block = (metadata.get("physics_metadata") or {}).get(IDENTITY_KEY) or {}
    calibration = block.get("qflux_calibration")
    if sum_name not in arrays or weight_name not in arrays:
        raise QFluxRefused(
            f"{path} carries no calibration arrays ({sum_name}, {weight_name}); it is not "
            "the checkpoint of a run with [ocean] qflux_calibration = true")
    missing = ([] if not isinstance(calibration, dict)
               else [key for key in CALIBRATION_KEYS if key not in calibration])
    if not isinstance(calibration, dict) or missing:
        raise QFluxRefused(
            f"{path}: its boundary identity carries no qflux_calibration record"
            + (f" with {', '.join(missing)}" if missing else "")
            + ", so the physics, grid and spin-up it was calibrated with are unknown")
    grid = calibration["grid"]
    if not isinstance(grid, (list, tuple)) or len(grid) != 2 \
            or not all(isinstance(n, int) and n > 0 for n in grid):
        raise QFluxRefused(f"{path}: its calibration grid record {grid!r} is not [nlat, nlon]")
    total = np.asarray(arrays[sum_name], dtype=np.float64)
    weight = np.asarray(arrays[weight_name], dtype=np.float64)
    shape = (MONTHS, int(grid[0]), int(grid[1]))
    if total.shape != shape or weight.shape != shape:
        raise QFluxRefused(f"{path}: the calibration arrays are {total.shape} and "
                           f"{weight.shape}, not the {shape} its grid record says")
    return {
        "path": str(path), "sum": total, "weight": weight, "calibration": calibration,
        "climatology_sha256": block.get("climatology_sha256"),
        "start_utc": block.get("start_utc"),
        "config_hash": metadata["config_hash"], "self_sha256": metadata["self_sha256"],
        "step": int(metadata["step"]), "time_s": float(metadata["time_s"]),
    }


def _fill_rows(plane, defined):
    """``plane`` with every column off ``defined`` set to the mean of its
    row's defined columns, a row with none taking the nearest row's mean (a
    smooth surround for the spherical-harmonic projection, which then reads
    only the defined columns back)."""
    nlat = plane.shape[0]
    counts = defined.sum(axis=1)
    rows = np.flatnonzero(counts > 0)
    means = np.where(defined, plane, 0.0).sum(axis=1)[rows] / counts[rows]
    row_mean = np.interp(np.arange(nlat, dtype=np.float64), rows.astype(np.float64), means)
    return np.where(defined, plane, row_mean[:, None])


def _smooth(flux, coverage, *, truncation: int, grid):
    """Each covered month projected onto spherical harmonics up to
    ``truncation`` (ocean_rules.truncate, float64) and read back on its
    covered columns; a truncation at or above the grid's own content leaves
    the month as it is."""
    from .ocean_rules import truncate

    out = flux.copy()
    for month in range(MONTHS):
        defined = coverage[month] > 0.0
        if not defined.any():
            continue
        smoothed = truncate(_fill_rows(flux[month], defined), grid, int(truncation),
                            int(grid.truncation))
        out[month] = np.where(defined, smoothed, np.nan)
    return out


def _write_npz(path: Path, arrays: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=_ZIP_EPOCH)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_STORED
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    partial.replace(path)
    return _sha256_file(path)


def build_qflux(checkpoints, *, out, smoothing_truncation: int | None = None) -> dict:
    """Write the file from the last checkpoint of each calibration run (their
    sums and weights add); return its provenance.  Refuses checkpoints of
    other grids, physics identities or climatologies, and a checkpoint that
    carries no calibration arrays.  ``smoothing_truncation`` projects each
    month onto spherical harmonics up to that truncation over the ocean
    (None: unsmoothed).  Byte-reproducible from the same inputs.

    The runs are added in the order of their checkpoints' self SHA-256, so
    the bytes do not depend on the order they are named in.  The same
    checkpoint twice, or two checkpoints of one run (one config hash and one
    start), are refused: either would count the same days twice."""
    from ._version import __version__

    paths = [Path(p) for p in checkpoints]
    if not paths:
        raise ValueError("a Q-flux file is built from at least one calibration checkpoint")
    runs = sorted((_read_calibration(p) for p in paths), key=lambda r: r["self_sha256"])
    first = runs[0]
    seen: dict[tuple, str] = {}
    for run in runs:
        key = (run["config_hash"], run["start_utc"])
        if key in seen:
            raise QFluxRefused(
                f"{run['path']} and {seen[key]} are checkpoints of one run (config "
                f"{run['config_hash'][:12]}..., start {run['start_utc']}); a run's last "
                "checkpoint already holds all it accumulated, so name only that one")
        seen[key] = run["path"]
        differ = _differences(first["calibration"], run["calibration"])
        if differ:
            raise QFluxRefused(f"{run['path']} and {first['path']} were calibrated with "
                               f"different {', '.join(differ)}; their sums do not add")
        if run["climatology_sha256"] != first["climatology_sha256"]:
            raise QFluxRefused(
                f"{run['path']} prescribed the climatology {run['climatology_sha256']} and "
                f"{first['path']} prescribed {first['climatology_sha256']}; their sums do "
                "not add")
    if not first["climatology_sha256"]:
        raise QFluxRefused(f"{first['path']}: its boundary identity names no climatology")
    total = np.zeros_like(first["sum"])
    weight = np.zeros_like(first["weight"])
    for run in runs:
        total = total + run["sum"]
        weight = weight + run["weight"]
    calibration = first["calibration"]
    from .spectral.grid import GaussianGrid

    from .sea_ice_thermo import ICE_REFERENCE_THICKNESS_M

    nlat, nlon = (int(n) for n in calibration["grid"])
    # A Gaussian grid is fixed by its shape: the latitudes are the nlat
    # Gauss-Legendre nodes, the longitudes nlon equal steps from 0.
    grid = GaussianGrid.for_shape(nlat, nlon)
    latitude, longitude = grid.latitude_deg, grid.longitude_deg
    covered = weight > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.where(covered, total / np.where(covered, weight, 1.0), np.nan)
    if smoothing_truncation is not None:
        flux = _smooth(flux, weight, truncation=int(smoothing_truncation), grid=grid)
    coverage_days = [round(float(np.max(weight[m])) / SECONDS_PER_DAY, 6)
                     for m in range(MONTHS)]
    provenance = _plain({
        "schema": QFLUX_SCHEMA,
        "nlat": nlat, "nlon": nlon,
        "calibration": calibration,
        "physics_identity": calibration["physics_identity"],
        "climatology_sha256": first["climatology_sha256"],
        "ice_reference_thickness_m": dict(ICE_REFERENCE_THICKNESS_M),
        "spinup_days": calibration["spinup_days"],
        "smoothing_truncation": None if smoothing_truncation is None
        else int(smoothing_truncation),
        "runs": [{"config_hash": run["config_hash"], "checkpoint_self_sha256":
                  run["self_sha256"], "start_utc": run["start_utc"], "step": run["step"],
                  "time_s": run["time_s"]} for run in runs],
        "month_coverage_days": coverage_days,
        "package_version": __version__,
        "rule": ("calibration_flux_w_m2 = sum / weight per month and column, NaN where a "
                 "month has no weight; sums and weights of the runs added; the flux is "
                 "F_cal, the net heat the atmosphere put into the ocean-and-ice system "
                 "with the SST prescribed to the climatology, deposited with the "
                 "climatology's month weights after the spin-up"),
        "units": {"calibration_flux_w_m2": "W m-2", "coverage_s": "s"},
    })
    arrays = {
        "calibration_flux_w_m2": np.ascontiguousarray(flux, dtype=np.float64),
        "coverage_s": np.ascontiguousarray(weight, dtype=np.float64),
        "latitude_deg": np.ascontiguousarray(latitude, dtype=np.float64),
        "longitude_deg": np.ascontiguousarray(longitude, dtype=np.float64),
        "provenance": np.array(_canonical(provenance)),
    }
    _write_npz(Path(out), arrays)
    return provenance


def load_qflux(path) -> QFluxFile:
    """Read and check the schema of a file :func:`build_qflux` wrote."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "provenance" not in data.files:
            raise QFluxRefused(f"{path} is not a {QFLUX_SCHEMA} file: it has no provenance")
        provenance = json.loads(str(data["provenance"]))
        if provenance.get("schema") != QFLUX_SCHEMA:
            raise QFluxRefused(f"{path} is not a {QFLUX_SCHEMA} file")
        arrays = {name: np.array(data[name]) for name in data.files if name != "provenance"}
    missing = [name for name in ("calibration_flux_w_m2", "coverage_s", "latitude_deg",
                                 "longitude_deg") if name not in arrays]
    if missing:
        raise QFluxRefused(f"{path} carries no {missing[0]}")
    lat, lon = arrays["latitude_deg"], arrays["longitude_deg"]
    shape = (MONTHS, lat.size, lon.size)
    for name in ("calibration_flux_w_m2", "coverage_s"):
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float64:
            raise QFluxRefused(f"{path}: {name} is {value.dtype} {value.shape}, "
                               f"not float64 {shape}")
    coverage = arrays["coverage_s"]
    flux = arrays["calibration_flux_w_m2"]
    if not np.isfinite(coverage).all() or (coverage < 0.0).any():
        raise QFluxRefused(f"{path}: coverage_s is not finite and non-negative")
    if not np.array_equal(np.isfinite(flux), coverage > 0.0):
        raise QFluxRefused(f"{path}: calibration_flux_w_m2 is not defined exactly where "
                           "coverage_s is above zero")
    return QFluxFile(path=path, sha256=_sha256_file(path), flux_w_m2=flux,
                     coverage_s=coverage, latitude_deg=lat, longitude_deg=lon,
                     provenance=provenance)


def main(argv=None) -> int:
    """``gpuwm-global climatology qflux CHECKPOINT [CHECKPOINT ...] --out
    FILE [--smoothing-truncation N]``: the arguments after ``qflux``."""
    parser = argparse.ArgumentParser(
        prog="gpuwm-global climatology qflux",
        description="Write the slab ocean's heat-flux correction file "
                    f"({QFLUX_SCHEMA}) from the last checkpoint of one or more "
                    "calibration runs ([ocean] mode = \"climatology\" with "
                    "qflux_calibration = true); their sums and weights add.")
    parser.add_argument("checkpoints", nargs="+", metavar="CHECKPOINT",
                        help="the last checkpoint of each calibration run")
    parser.add_argument("--out", required=True, metavar="FILE",
                        help="the .npz file to write")
    parser.add_argument("--smoothing-truncation", type=int, default=None, metavar="N",
                        help="project each month onto spherical harmonics up to T N over "
                             "the ocean (default: unsmoothed)")
    args = parser.parse_args(argv)
    if args.smoothing_truncation is not None and args.smoothing_truncation < 1:
        parser.error("--smoothing-truncation is a truncation of at least 1")
    try:
        provenance = build_qflux(args.checkpoints, out=args.out,
                                 smoothing_truncation=args.smoothing_truncation)
    except (QFluxRefused, ValueError) as error:
        print(f"climatology qflux: refused: {error}")
        return 2
    path = Path(args.out)
    smoothing = provenance["smoothing_truncation"]
    print(f"climatology qflux: {len(provenance['runs'])} calibration run(s) on the "
          f"{provenance['nlat']}x{provenance['nlon']} Gaussian grid, climatology "
          f"{provenance['climatology_sha256'][:12]}..., spin-up "
          f"{provenance['spinup_days']:g} d, smoothing "
          f"{'none' if smoothing is None else f'T{smoothing}'}")
    print("climatology qflux: coverage (days, largest over the columns): " + ", ".join(
        f"{name[:3]} {days:g}" for name, days in zip(_MONTH_NAMES,
                                                     provenance["month_coverage_days"])))
    print(f"climatology qflux: {path} ({path.stat().st_size:,} B, SHA-256 "
          f"{_sha256_file(path)})")
    return 0


__all__ = [
    "CALIBRATION_KEYS", "DEFAULT_SPINUP_DAYS", "MONTHS", "QFLUX_SCHEMA", "QFluxFile",
    "QFluxRefused", "build_qflux", "deposit", "load_qflux", "main", "months_needed",
]
