"""Composition replay: transported fields added to a finished forecast.

A forecast's checkpoints hold its wind and its surface pressure at every
time it kept.  This module reads those, and nothing else, and carries a
table of extra fields (tags, an origin map, a point release, a field read
from a file such as an ozone analysis) through the same semi-Lagrangian
machinery the model's own grid tracers ride: the departure-point search of
:mod:`arwen_global.semilag.trajectory`, the quasi-monotone cubic gather of
:mod:`arwen_global.semilag.interpolate` and the Bermejo-Conde mass fixer of
:mod:`arwen_global.semilag.tracers`.  The physics is not run again and the
weather is not integrated again; the weather checkpoints are opened read
only and their bytes cannot change.

What it is and what it is not
-----------------------------
It is an OFFLINE transport.  The model advects its own species with the
wind of every step; a replay has the wind only at the times the run kept a
checkpoint, and between two of them it uses the straight line in time
between the two winds (and between the two layer thicknesses).  That is
the stored-field approach of offline chemistry-transport models, and its
error grows with the checkpoint interval: a run kept every step replays
its winds exactly at every step, a run kept every six hours replays a
six-hourly wind.  The receipt states the interval every sidecar was made
from, and :func:`replay` refuses an interval that is not a whole number
of substeps.

The trajectory is the two-time-level one the model's step solves, with
one difference that only an offline replay can have: the arrival-side
wind is the KNOWN wind at the end of the substep, not the extrapolation
``2 V(t) - V(t - dt)`` a forecast has to make.  The departure-side wind is
the wind at the start of the substep, interpolated to the departure point.

What the rows do not have: boundary-layer mixing, convective transport,
wet and dry removal.  A replay row is carried by the resolved wind alone.
That is a fair first answer above the boundary layer and outside deep
convection (stratospheric ozone, a volcanic plume at cruising height, an
origin map in the extratropical free troposphere) and is wrong inside
them; the module says so in every receipt (``processes``).

Why the weather cannot change
-----------------------------
Nothing in this module is imported by the forecast loop, nothing in it
writes a weather checkpoint, and the checkpoints are read through
``numpy.load`` with only the four arrays the flow needs materialized.  The
known-answer suite hashes every checkpoint file before and after a replay
(tests/test_arwen_global_composition.py).

Rows
----
``constant``  a uniform mixing ratio (``value``, default 1).  With the fixer
              off it measures the gather's interpolation error, which is
              zero to rounding for a constant; with the fixer on it
              measures the air-mass drift of the replayed flow.
``band``      one inside a latitude and pressure box at the first
              checkpoint, zero outside: a tag that says where air came from.
``origin``    three rows ``<name>_x``, ``_y``, ``_z``: the unit-sphere
              position of each point at the first checkpoint, stored as
              ``(1 + c) / 2`` because the fixer floors a field at zero and a
              Cartesian component is signed.
``release``   zero at the start; ``mass_kg`` emitted uniformly in mixing
              ratio over the air of a cylinder (``lon_deg``, ``lat_deg``,
              ``radius_km``, between ``p_top_pa`` and ``p_bottom_pa``) from
              model time ``start_s`` for ``duration_s``.  The float64 ledger
              holds the emitted mass beside the carried mass.
``field``     a ``(nlev, nlat, nlon)`` array from an ``.npz`` file on the
              model grid (``path``, ``array``, ``scale``), floored at zero.
              An ozone analysis on the model's levels is one.

The spec
--------
A TOML file with one ``[composition]`` table::

    [composition]
    schema = "gpuwm.arwen-global-composition/v1"
    substep_s = 300.0
    fixer = "bermejo_conde"

    [[composition.tracer]]
    name = "plume"
    kind = "release"
    lon_deg = 15.0
    lat_deg = 37.75
    radius_km = 60.0
    p_bottom_pa = 40000.0
    p_top_pa = 20000.0
    start_s = 0.0
    duration_s = 21600.0
    mass_kg = 1.0e9

The spec's own hash keys the result together with the weather
checkpoints' ``self_sha256``: a sidecar belongs to exactly one weather and
one spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
import tomllib
from typing import Any, Iterable, Sequence

import numpy as np

from .constants import GRAVITY_M_S2, GRID_TRACERS

#: The spec file's schema string.
SPEC_SCHEMA = "gpuwm.arwen-global-composition/v1"

#: The schema of every sidecar this module writes.
SIDECAR_SCHEMA = "gpuwm.arwen-global-composition-sidecar/v1"

#: A sidecar sits in the replay's output directory, named after the weather
#: checkpoint it belongs to with this suffix in place of ``.npz``.
SIDECAR_SUFFIX = ".composition.npz"

#: The replay's receipt, in the output directory.
RECEIPT_NAME = "arwen-global-composition-receipt.json"

#: The row kinds a spec may name.
TRACER_KINDS = ("constant", "band", "origin", "release", "field")

#: The mass fixers a replay accepts (semilag.tracers.TRACER_FIXERS less the
#: additive form, which needs the gather's clip deficit and buys nothing a
#: passive row needs).
REPLAY_FIXERS = ("bermejo_conde", "mass_proportional", "none")

#: The four checkpoint arrays a replay reads.  Nothing else is materialized.
FLOW_ARRAYS = (
    "atmosphere__vorticity", "atmosphere__divergence",
    "atmosphere__log_surface_pressure",
)

#: What a replay row is carried by, stated in every receipt.
PROCESSES = {
    "resolved_advection": True,
    "boundary_layer_mixing": False,
    "convective_transport": False,
    "wet_removal": False,
    "dry_deposition": False,
    "chemistry": False,
    "radiative_feedback": False,
}

_ROW_KEYS = {
    "constant": {"value"},
    "band": {"lat_min_deg", "lat_max_deg", "p_top_pa", "p_bottom_pa"},
    "origin": set(),
    "release": {"lon_deg", "lat_deg", "radius_km", "p_top_pa", "p_bottom_pa",
                "start_s", "duration_s", "mass_kg"},
    "field": {"path", "array", "scale"},
}
_ROW_REQUIRED = {
    "constant": set(),
    "band": {"lat_min_deg", "lat_max_deg"},
    "origin": set(),
    "release": {"lon_deg", "lat_deg", "radius_km", "p_top_pa",
                "p_bottom_pa", "duration_s", "mass_kg"},
    "field": {"path", "array"},
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class TracerRow:
    """One row of the spec: a name, a kind and the kind's parameters."""

    name: str
    kind: str
    params: tuple = ()

    def param(self, key: str, default=None):
        return dict(self.params).get(key, default)

    def stored_names(self) -> tuple[str, ...]:
        """The arrays this row is carried as (three for an origin row)."""
        if self.kind == "origin":
            return tuple(f"{self.name}_{axis}" for axis in "xyz")
        return (self.name,)

    def identity(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "params": {key: value for key, value in self.params}}


@dataclass(frozen=True)
class CompositionSpec:
    """The rows and the transport settings of one replay."""

    rows: tuple[TracerRow, ...]
    substep_s: float = 300.0
    fixer: str = "bermejo_conde"
    limiter: str = "quasi_monotone"
    trajectory_iterations: int = 3
    trajectory_convergence_cells: float = 0.01
    #: The file a ``field`` row's relative path is read against.  Not part of
    #: the identity: the identity holds the file's own sha256 instead.
    base_dir: str = field(default=".", compare=False)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a composition spec needs at least one tracer row")
        if not (math.isfinite(self.substep_s) and self.substep_s > 0.0):
            raise ValueError(
                f"composition substep_s must be positive and finite, got "
                f"{self.substep_s!r}")
        if self.fixer not in REPLAY_FIXERS:
            raise ValueError(
                "composition fixer must be one of "
                + ", ".join(repr(name) for name in REPLAY_FIXERS)
                + f", got {self.fixer!r}")
        if self.limiter not in ("quasi_monotone", "none"):
            raise ValueError(
                "composition limiter must be 'quasi_monotone' or 'none', got "
                f"{self.limiter!r}")
        if int(self.trajectory_iterations) < 1:
            raise ValueError("composition trajectory_iterations must be >= 1")
        names = [name for row in self.rows for name in row.stored_names()]
        if len(set(names)) != len(names):
            raise ValueError(
                f"composition rows carry duplicate names: {sorted(names)}")

    @property
    def stored_names(self) -> tuple[str, ...]:
        return tuple(name for row in self.rows for name in row.stored_names())

    def identity(self) -> dict:
        rows = []
        for row in self.rows:
            entry = row.identity()
            if row.kind == "field":
                entry["file_sha256"] = _file_sha256(self.field_path(row))
            rows.append(entry)
        return {
            "schema": SPEC_SCHEMA,
            "substep_s": float(self.substep_s),
            "fixer": self.fixer,
            "limiter": self.limiter,
            "trajectory_iterations": int(self.trajectory_iterations),
            "trajectory_convergence_cells": float(
                self.trajectory_convergence_cells),
            "rows": rows,
        }

    @property
    def spec_hash(self) -> str:
        return _sha256(self.identity())

    def field_path(self, row: TracerRow) -> Path:
        path = Path(str(row.param("path")))
        return path if path.is_absolute() else Path(self.base_dir) / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def _row_from_table(table: dict) -> TracerRow:
    if not isinstance(table, dict):
        raise ValueError("every [[composition.tracer]] entry must be a table")
    name = table.get("name")
    kind = table.get("kind")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"a composition tracer needs a name that is an identifier, got "
            f"{name!r}")
    if kind not in TRACER_KINDS:
        raise ValueError(
            f"composition tracer {name!r}: kind must be one of "
            + ", ".join(repr(item) for item in TRACER_KINDS)
            + f", got {kind!r}")
    params = {key: value for key, value in table.items()
              if key not in ("name", "kind")}
    unknown = sorted(set(params) - _ROW_KEYS[kind])
    if unknown:
        raise ValueError(
            f"composition tracer {name!r} ({kind}): unknown keys "
            f"{', '.join(unknown)}")
    missing = sorted(_ROW_REQUIRED[kind] - set(params))
    if missing:
        raise ValueError(
            f"composition tracer {name!r} ({kind}): missing keys "
            f"{', '.join(missing)}")
    clean: dict[str, object] = {}
    for key, value in params.items():
        if key in ("path", "array"):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"composition tracer {name!r}: {key} must be a string")
            clean[key] = value
        else:
            clean[key] = _number(value, f"composition tracer {name!r} {key}")
    if kind == "release":
        if clean["radius_km"] <= 0.0 or clean["duration_s"] <= 0.0:
            raise ValueError(
                f"composition tracer {name!r}: radius_km and duration_s must "
                "be positive")
        if clean["mass_kg"] < 0.0:
            raise ValueError(
                f"composition tracer {name!r}: mass_kg must be nonnegative")
        clean.setdefault("start_s", 0.0)
    if kind in ("release", "band") and "p_top_pa" in clean:
        if clean.get("p_bottom_pa", math.inf) <= clean["p_top_pa"]:
            raise ValueError(
                f"composition tracer {name!r}: p_bottom_pa must exceed "
                "p_top_pa (pressure falls upward)")
    if kind == "band" and clean["lat_max_deg"] <= clean["lat_min_deg"]:
        raise ValueError(
            f"composition tracer {name!r}: lat_max_deg must exceed lat_min_deg")
    return TracerRow(name=name, kind=kind, params=tuple(sorted(clean.items())))


def spec_from_mapping(raw: dict, *, base_dir: str | Path = ".") -> CompositionSpec:
    """The spec a ``[composition]`` table describes."""
    table = raw.get("composition") if "composition" in raw else raw
    if not isinstance(table, dict):
        raise ValueError("a composition spec is one [composition] table")
    allowed = {"schema", "substep_s", "fixer", "limiter",
               "trajectory_iterations", "trajectory_convergence_cells",
               "tracer"}
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown [composition] keys: {', '.join(unknown)}")
    schema = table.get("schema", SPEC_SCHEMA)
    if schema != SPEC_SCHEMA:
        raise ValueError(
            f"composition schema must be {SPEC_SCHEMA!r}, got {schema!r}")
    rows = tuple(_row_from_table(entry) for entry in table.get("tracer", ()))
    return CompositionSpec(
        rows=rows,
        substep_s=_number(table.get("substep_s", 300.0), "substep_s"),
        fixer=str(table.get("fixer", "bermejo_conde")),
        limiter=str(table.get("limiter", "quasi_monotone")),
        trajectory_iterations=int(table.get("trajectory_iterations", 3)),
        trajectory_convergence_cells=_number(
            table.get("trajectory_convergence_cells", 0.01),
            "trajectory_convergence_cells"),
        base_dir=str(base_dir),
    )


def load_spec(path: str | Path) -> CompositionSpec:
    """Read a composition spec from a TOML file."""
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    return spec_from_mapping(raw, base_dir=source.resolve().parent)


# ---------------------------------------------------------------------------
# The flow a checkpoint holds


@dataclass
class FlowFrame:
    """The advecting flow of one checkpoint, on the model grid.

    ``vx, vy, vz`` are the geocentric Cartesian wind in m/s and
    ``level_rate`` the vertical rate in level indices per second, as
    :class:`arwen_global.semilag.trajectory.CartesianWind` reads them;
    ``dp`` and ``p_full`` the layer thickness and full-level pressure.
    """

    time_s: float
    vx: Any
    vy: Any
    vz: Any
    level_rate: Any
    dp: Any
    p_full: Any

    def blend(self, other: "FlowFrame", weight: float, scalar) -> "FlowFrame":
        """The straight line in time between two frames.

        Written ``(1 - w) a + w b`` so ``w = 0`` returns ``a`` and ``w = 1``
        returns ``b`` bit for bit: a substep that ends on a checkpoint
        reads that checkpoint's own flow.
        """
        if weight == 0.0:
            return self
        if weight == 1.0:
            return other
        w = scalar(weight)
        keep = scalar(1.0 - weight)
        return FlowFrame(
            time_s=(1.0 - weight) * self.time_s + weight * other.time_s,
            **{name: keep * getattr(self, name) + w * getattr(other, name)
               for name in ("vx", "vy", "vz", "level_rate", "dp", "p_full")},
        )

    def wind(self):
        from .semilag.trajectory import CartesianWind

        return CartesianWind(vx=self.vx, vy=self.vy, vz=self.vz,
                             level_rate=self.level_rate)


def checkpoint_flow_header(path: str | Path) -> dict:
    """The metadata of a weather checkpoint, without its arrays."""
    from .checkpoint import read_checkpoint_header

    return read_checkpoint_header(path)


class FlowReader:
    """Turns weather checkpoints into :class:`FlowFrame` objects.

    It holds a dynamics-only :class:`arwen_global.dynamics.MoistHybridModel`
    built on the run's transform and vertical coordinate, with no physics,
    and asks it the questions the model's own step asks: the grid wind, the
    pressure volumes and the eta-coordinate mass flux.  Theta, vapour and
    the ten grid tracers do not enter any of them, so the state it builds
    carries zeros there (one shared array each).
    """

    def __init__(self, cfg):
        from .dynamics import MoistHybridModel
        from .runner import build_transform
        from .semilag.step import grid_tables

        transform = build_transform(cfg)
        self.cfg = cfg
        self.transform = transform
        backend = transform.backend
        self.backend = backend
        self.xp = backend.xp
        nlat, nlon = transform.grid.shape
        self.model = MoistHybridModel(
            transform=transform,
            vertical=cfg.vertical,
            surface_geopotential=np.zeros((nlat, nlon), dtype=np.float64),
            physics=None,
            integrator=cfg.integrator,
            semilag=cfg.semilag,
            synthesis_memo=False,
        )
        self.tables = grid_tables(self.model)
        nlev = int(self.model.nlev)
        self._zero_spectral = self.xp.zeros(
            (nlev, *transform.spectral_shape), dtype=backend.complex_dtype)
        self._zero_grid = self.xp.zeros(
            (nlev, nlat, nlon), dtype=backend.float_dtype)

    @property
    def shape(self) -> tuple[int, int, int]:
        nlat, nlon = self.transform.grid.shape
        return (int(self.model.nlev), int(nlat), int(nlon))

    def check_geometry(self, path: str | Path) -> None:
        """Refuse a checkpoint whose arrays are not this grid's."""
        with np.load(path, allow_pickle=False) as archive:
            for name in FLOW_ARRAYS:
                if name not in archive.files:
                    raise ValueError(
                        f"checkpoint {path} carries no {name}; it is not a "
                        "weather checkpoint a replay can read")
            vorticity = archive["atmosphere__vorticity"]
            want = (self.shape[0], *self.transform.spectral_shape)
            if tuple(vorticity.shape) != want:
                raise ValueError(
                    f"checkpoint {path} holds a {tuple(vorticity.shape)} "
                    f"vorticity where this geometry has {want}: the replay "
                    "config is not the run's grid")

    def frame(self, path: str | Path) -> FlowFrame:
        from .semilag.trajectory import cartesian_wind, level_rate_from_mass_flux
        from .state import MoistHybridState

        backend = self.backend
        xp = self.xp
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))
            arrays = {name: np.array(archive[name]) for name in FLOW_ARRAYS}
        state = MoistHybridState(
            vorticity=backend.asarray(arrays["atmosphere__vorticity"],
                                      dtype=backend.complex_dtype),
            divergence=backend.asarray(arrays["atmosphere__divergence"],
                                       dtype=backend.complex_dtype),
            theta=self._zero_spectral,
            log_surface_pressure=backend.asarray(
                arrays["atmosphere__log_surface_pressure"],
                dtype=backend.complex_dtype),
            qv=self._zero_spectral,
            **{name: self._zero_grid for name in GRID_TRACERS},
            time_s=float(metadata["time_s"]),
            step=int(metadata["step"]),
        )
        del arrays
        model = self.model
        sources = model.grid_sources(state, only=("u", "v", "dp", "p_full"))
        g, _stack, _names = model.grid_band(sources, model.whole_rows)
        _spectral, _div_mass, _ps_t, omega_half = model._mass_flux_and_omega(
            sources)
        level_rate = level_rate_from_mass_flux(
            omega_half, p_full=g["p_full"], xp=xp)
        vx, vy, vz = cartesian_wind(g["u"], g["v"], self.tables)
        frame = FlowFrame(
            time_s=float(metadata["time_s"]), vx=vx, vy=vy, vz=vz,
            level_rate=level_rate, dp=xp.array(g["dp"], copy=True),
            p_full=xp.array(g["p_full"], copy=True),
        )
        del g, sources, omega_half, state
        model.release_syntheses()
        return frame


# ---------------------------------------------------------------------------
# Rows on the grid


def _grid_geometry(transform):
    """Latitude and longitude of every grid point, radians, on the host."""
    grid = transform.grid
    lat = np.asarray(grid.lat_rad, dtype=np.float64)
    lon = np.arange(int(grid.nlon), dtype=np.float64) * (
        2.0 * math.pi / int(grid.nlon))
    return lat, lon


def _cell_area_m2(transform) -> np.ndarray:
    """Area of every grid cell in square metres, (nlat, 1), float64."""
    grid = transform.grid
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    fraction = weights / (2.0 * int(grid.nlon))
    return (4.0 * math.pi * float(grid.radius_m) ** 2 * fraction)[:, None]


def _great_circle_m(transform, lon_deg: float, lat_deg: float) -> np.ndarray:
    lat, lon = _grid_geometry(transform)
    lat0 = math.radians(lat_deg)
    lon0 = math.radians(lon_deg)
    dlat = lat[:, None] - lat0
    dlon = lon[None, :] - lon0
    hav = (np.sin(0.5 * dlat) ** 2
           + np.cos(lat[:, None]) * math.cos(lat0) * np.sin(0.5 * dlon) ** 2)
    return 2.0 * float(transform.grid.radius_m) * np.arcsin(
        np.sqrt(np.clip(hav, 0.0, 1.0)))


def _pressure_mask(p_full, top: float, bottom: float, xp):
    return (p_full >= top) & (p_full <= bottom)


def initial_rows(spec: CompositionSpec, frame: FlowFrame, transform) -> dict:
    """Every stored row at the first checkpoint, on the device."""
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    nlev = int(frame.dp.shape[0])
    nlat, nlon = transform.grid.shape
    lat, lon = _grid_geometry(transform)
    rows: dict[str, Any] = {}
    for row in spec.rows:
        if row.kind == "constant":
            rows[row.name] = xp.full((nlev, nlat, nlon),
                                     float(row.param("value", 1.0)),
                                     dtype=dtype)
        elif row.kind == "band":
            lat_deg = np.degrees(lat)
            inside = ((lat_deg >= row.param("lat_min_deg"))
                      & (lat_deg <= row.param("lat_max_deg")))
            plane = xp.asarray(
                np.broadcast_to(inside[:, None], (nlat, nlon)), dtype=dtype)
            mask = _pressure_mask(frame.p_full,
                                  float(row.param("p_top_pa", -math.inf)),
                                  float(row.param("p_bottom_pa", math.inf)),
                                  xp)
            rows[row.name] = xp.where(mask, plane[None], 0.0).astype(dtype)
        elif row.kind == "origin":
            planes = (
                np.cos(lat)[:, None] * np.cos(lon)[None, :],
                np.cos(lat)[:, None] * np.sin(lon)[None, :],
                np.broadcast_to(np.sin(lat)[:, None], (nlat, nlon)),
            )
            for axis, plane in zip("xyz", planes):
                stored = 0.5 * (1.0 + plane)
                rows[f"{row.name}_{axis}"] = xp.asarray(
                    np.broadcast_to(stored[None], (nlev, nlat, nlon)),
                    dtype=dtype).copy()
        elif row.kind == "release":
            rows[row.name] = xp.zeros((nlev, nlat, nlon), dtype=dtype)
        elif row.kind == "field":
            path = spec.field_path(row)
            with np.load(path, allow_pickle=False) as archive:
                name = str(row.param("array"))
                if name not in archive.files:
                    raise ValueError(
                        f"composition tracer {row.name!r}: {path} has no "
                        f"array {name!r}")
                value = np.asarray(archive[name], dtype=np.float64)
            if value.shape != (nlev, nlat, nlon):
                raise ValueError(
                    f"composition tracer {row.name!r}: {path}:{name} is "
                    f"{value.shape} where the model grid is "
                    f"{(nlev, nlat, nlon)}")
            value = np.maximum(value * float(row.param("scale", 1.0)), 0.0)
            rows[row.name] = xp.asarray(value, dtype=dtype)
        else:  # pragma: no cover - the spec reader refuses other kinds
            raise AssertionError(row.kind)
    return rows


@dataclass
class _Release:
    row: TracerRow
    footprint: Any  # (nlat, nlon) bool on the device

    @property
    def start(self) -> float:
        return float(self.row.param("start_s", 0.0))

    @property
    def stop(self) -> float:
        return self.start + float(self.row.param("duration_s"))

    def mass_between(self, t0: float, t1: float) -> float:
        overlap = max(0.0, min(t1, self.stop) - max(t0, self.start))
        return float(self.row.param("mass_kg")) * overlap / float(
            self.row.param("duration_s"))


def _releases(spec: CompositionSpec, transform) -> list[_Release]:
    xp = transform.backend.xp
    out = []
    for row in spec.rows:
        if row.kind != "release":
            continue
        distance = _great_circle_m(transform, float(row.param("lon_deg")),
                                   float(row.param("lat_deg")))
        footprint = distance <= 1000.0 * float(row.param("radius_km"))
        if not footprint.any():
            raise ValueError(
                f"composition release {row.name!r}: no grid column lies "
                f"within {row.param('radius_km')} km of its centre; give it a "
                "radius of at least one grid spacing")
        out.append(_Release(row=row, footprint=xp.asarray(footprint)))
    return out


def _emit(rows: dict, releases: Sequence[_Release], frame: FlowFrame,
          transform, t0: float, t1: float, ledger: dict) -> None:
    """Add each release's mass for ``[t0, t1)`` as a uniform mixing ratio
    over the air of its cylinder, and book it in float64."""
    if not releases:
        return
    backend = transform.backend
    xp = backend.xp
    area = xp.asarray(_cell_area_m2(transform), dtype=backend.float_dtype)
    for release in releases:
        mass = release.mass_between(t0, t1)
        if mass <= 0.0:
            continue
        row = release.row
        mask = (_pressure_mask(frame.p_full, float(row.param("p_top_pa")),
                               float(row.param("p_bottom_pa")), xp)
                & release.footprint[None])
        air = float(xp.sum(xp.where(mask, frame.dp * area[None], 0.0),
                           dtype=xp.float64)) / GRAVITY_M_S2
        if air <= 0.0:
            raise ValueError(
                f"composition release {row.name!r}: no model layer lies "
                f"between {row.param('p_top_pa')} and "
                f"{row.param('p_bottom_pa')} Pa inside its footprint")
        increment = backend.float_dtype(mass / air)
        rows[row.name] = rows[row.name] + xp.where(mask, increment, 0.0).astype(
            backend.float_dtype)
        ledger[row.name] = ledger.get(row.name, 0.0) + mass


def mass_kg(field, dp, transform) -> float:
    """A row's global mass in kg, reduced on the host in float64."""
    to_numpy = transform.backend.to_numpy
    value = np.asarray(to_numpy(field), dtype=np.float64)
    thickness = np.asarray(to_numpy(dp), dtype=np.float64)
    area = _cell_area_m2(transform)
    return float(np.sum(value * thickness * area[None]) / GRAVITY_M_S2)


# ---------------------------------------------------------------------------
# The replay


def _ordered_checkpoints(paths: Iterable[str | Path]) -> list[tuple[Path, dict]]:
    items = []
    for path in paths:
        source = Path(path)
        items.append((source, checkpoint_flow_header(source)))
    items.sort(key=lambda item: float(item[1]["time_s"]))
    times = [float(meta["time_s"]) for _path, meta in items]
    if len(items) < 2:
        raise ValueError("a replay needs at least two weather checkpoints")
    if len(set(times)) != len(times):
        raise ValueError("two weather checkpoints carry the same model time")
    hashes = {meta.get("config_hash") for _path, meta in items}
    if len(hashes) != 1:
        raise ValueError(
            "the weather checkpoints come from runs with different config "
            f"hashes ({sorted(str(h) for h in hashes)}); a replay reads one run")
    return items


def sidecar_path(outdir: str | Path, checkpoint: str | Path) -> Path:
    stem = Path(checkpoint).name
    if stem.endswith(".npz"):
        stem = stem[:-4]
    return Path(outdir) / f"{stem}{SIDECAR_SUFFIX}"


def _write_sidecar(path: Path, rows: dict, metadata: dict, to_numpy) -> None:
    arrays = {f"tracer__{name}": np.asarray(to_numpy(value))
              for name, value in rows.items()}
    record = dict(metadata)
    record["arrays"] = {name: {"shape": list(value.shape),
                               "dtype": str(value.dtype),
                               "sha256": hashlib.sha256(
                                   np.ascontiguousarray(value).tobytes()
                               ).hexdigest()}
                        for name, value in arrays.items()}
    record["self_sha256"] = _sha256(record)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, __metadata__=np.asarray(
            json.dumps(record, sort_keys=True, allow_nan=False)), **arrays)
    tmp.replace(path)


def read_sidecar(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """A sidecar's metadata and rows, verified against its own hashes."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {name.removeprefix("tracer__"): np.array(archive[name])
                  for name in archive.files if name.startswith("tracer__")}
    if metadata.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{path} is not a composition sidecar")
    claimed = metadata.pop("self_sha256", None)
    if claimed != _sha256(metadata):
        raise ValueError(f"composition sidecar {path} fails its self hash")
    for name, value in arrays.items():
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if metadata["arrays"].get(f"tracer__{name}", {}).get("sha256") != digest:
            raise ValueError(
                f"composition sidecar {path}: row {name} fails its hash")
    metadata["self_sha256"] = claimed
    return metadata, arrays


def replay(
    cfg,
    checkpoints: Sequence[str | Path],
    spec: CompositionSpec,
    outdir: str | Path,
    *,
    resume: str | Path | None = None,
    accept_geometry_config: bool = False,
    progress=None,
) -> dict:
    """Carry the spec's rows through the flow of ``checkpoints``.

    ``cfg`` is the run's config, or (with ``accept_geometry_config``) a
    config that carries the run's grid and vertical coordinate under a
    different hash: the opt-in is recorded in the receipt with both
    hashes, because a replay on a geometry the run did not integrate would
    otherwise read as the run's own.  ``resume`` is a sidecar this spec
    wrote at one of ``checkpoints``; the replay continues from it and the
    sidecars it writes are bit for bit the ones an unbroken replay writes.

    Returns the receipt, which is also written to ``outdir``.
    """
    from .semilag.tracers import STENCIL_FIXERS, advect, fix_mass
    from .semilag.trajectory import converged_departure_points

    started = time.perf_counter()
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    items = _ordered_checkpoints(checkpoints)
    weather_hash = str(items[0][1].get("config_hash"))
    geometry_mismatch = weather_hash != str(cfg.config_hash)
    if geometry_mismatch and not accept_geometry_config:
        raise ValueError(
            f"the weather checkpoints were written under config hash "
            f"{weather_hash} and the replay config hashes to "
            f"{cfg.config_hash}; pass the run's own config, or opt in with "
            "accept_geometry_config (--accept-geometry-config) and the "
            "receipt records both hashes")
    reader = FlowReader(cfg)
    for path, _meta in items:
        reader.check_geometry(path)
    transform = reader.transform
    backend = transform.backend
    xp = backend.xp
    scalar = backend.float_dtype
    to_numpy = backend.to_numpy
    tables = reader.tables
    spec_identity = spec.identity()
    spec_hash = _sha256(spec_identity)
    substep = float(spec.substep_s)
    for (path_a, meta_a), (path_b, meta_b) in zip(items, items[1:]):
        interval = float(meta_b["time_s"]) - float(meta_a["time_s"])
        count = interval / substep
        if abs(count - round(count)) > 1.0e-9 or round(count) < 1:
            raise ValueError(
                f"the interval {interval:g} s between {path_a.name} and "
                f"{path_b.name} is not a whole number of {substep:g} s "
                "substeps")

    releases = _releases(spec, transform)
    emitted: dict[str, float] = {}
    start_index = 0
    frame_a = reader.frame(items[0][0])
    if resume is None:
        rows = initial_rows(spec, frame_a, transform)
    else:
        metadata, stored = read_sidecar(resume)
        if metadata["spec_sha256"] != spec_hash:
            raise ValueError(
                f"composition sidecar {resume} was written under spec "
                f"{metadata['spec_sha256']}, not this spec ({spec_hash})")
        selves = [str(meta["self_sha256"]) for _path, meta in items]
        if metadata["weather"]["self_sha256"] not in selves:
            raise ValueError(
                f"composition sidecar {resume} belongs to a weather "
                "checkpoint that is not in this replay's list")
        start_index = selves.index(metadata["weather"]["self_sha256"])
        if start_index == len(items) - 1:
            raise ValueError(
                f"composition sidecar {resume} is at the last checkpoint; "
                "there is nothing left to replay")
        frame_a = reader.frame(items[start_index][0])
        rows = {name: xp.asarray(stored[name], dtype=scalar)
                for name in spec.stored_names}
        emitted = {name: float(value) for name, value
                   in metadata["ledger"]["emitted_kg"].items()}

    initial_mass = {name: mass_kg(value, frame_a.dp, transform)
                    for name, value in rows.items()}
    emitted_at_start = dict(emitted)
    sidecars: list[str] = []
    intervals: list[dict] = []

    def ledger_at(frame: FlowFrame) -> dict:
        return {
            "mass_kg": {name: mass_kg(value, frame.dp, transform)
                        for name, value in rows.items()},
            "emitted_kg": dict(emitted),
        }

    def write(index: int, frame: FlowFrame, interval_record: dict | None):
        path, meta = items[index]
        target = sidecar_path(output, path)
        _write_sidecar(target, rows, {
            "schema": SIDECAR_SCHEMA,
            "spec_sha256": spec_hash,
            "spec": spec_identity,
            "weather": {
                "checkpoint": path.name,
                "step": int(meta["step"]),
                "time_s": float(meta["time_s"]),
                "self_sha256": str(meta["self_sha256"]),
                "config_hash": weather_hash,
            },
            "replay_config_hash": str(cfg.config_hash),
            "processes": PROCESSES,
            "ledger": ledger_at(frame),
            "interval": interval_record,
        }, to_numpy)
        sidecars.append(str(target))

    if resume is None:
        write(0, frame_a, None)

    for index in range(start_index, len(items) - 1):
        path_b, meta_b = items[index + 1]
        frame_b = reader.frame(path_b)
        t_a = float(items[index][1]["time_s"])
        t_b = float(meta_b["time_s"])
        count = int(round((t_b - t_a) / substep))
        worst: dict[str, float] = {}
        displacement_cells = 0.0
        iterations = 0
        for sub in range(count):
            w0 = sub / count
            w1 = (sub + 1) / count
            before = frame_a.blend(frame_b, w0, scalar)
            after = frame_a.blend(frame_b, w1, scalar)
            t0 = t_a + sub * substep
            _emit(rows, releases, before, transform, t0, t0 + substep, emitted)
            stencil, diagnostics = converged_departure_points(
                before.wind(), tables, substep,
                iterations=int(spec.trajectory_iterations),
                limit_cells=float(spec.trajectory_convergence_cells),
                extrapolated=after.wind(),
            )
            displacement_cells = max(displacement_cells,
                                     float(diagnostics.displacement_max_cells))
            iterations = max(iterations, int(diagnostics.iterations))
            advected = advect(rows, stencil, monotone=spec.limiter)
            fixed, metrics = fix_mass(
                advected, rows, before.dp, after.dp, transform,
                scheme=spec.fixer,
                stencil=stencil if spec.fixer in STENCIL_FIXERS else None,
                consume=True,
            )
            del advected, stencil
            rows = fixed
            for key, value in metrics.items():
                if key.startswith("semilag_tracer_mass_fixer_relative__"):
                    name = key.rsplit("__", 1)[1]
                    worst[name] = max(worst.get(name, 0.0), float(value))
            del before, after
        record = {
            "from_checkpoint": items[index][0].name,
            "to_checkpoint": path_b.name,
            "interval_s": t_b - t_a,
            "substeps": count,
            "substep_s": substep,
            "trajectory_iterations_max": iterations,
            "displacement_max_cells": displacement_cells,
            "fixer_relative_max": worst,
        }
        intervals.append(record)
        frame_a = frame_b
        write(index + 1, frame_a, record)
        if progress is not None:
            progress(record)

    final = ledger_at(frame_a)
    # Carried mass against what the segment started with plus what it
    # emitted.  A resumed segment's starting mass already holds what was
    # emitted before it, so only this segment's emissions are added.
    drift = {}
    for name, start in initial_mass.items():
        expected = start + (final["emitted_kg"].get(name, 0.0)
                            - emitted_at_start.get(name, 0.0))
        drift[name] = (final["mass_kg"][name] - expected) / max(
            abs(expected), 1.0e-300)
    receipt = {
        "schema": "gpuwm.arwen-global-composition-receipt/v1",
        "status": "done",
        "spec_sha256": spec_hash,
        "spec": spec_identity,
        "weather_config_hash": weather_hash,
        "replay_config_hash": str(cfg.config_hash),
        "accept_geometry_config": bool(accept_geometry_config),
        "geometry_config_differs": bool(geometry_mismatch),
        "geometry": {
            "truncation": int(cfg.truncation),
            "grid": list(reader.shape),
            "precision": str(cfg.precision),
            "backend": str(cfg.backend),
        },
        "weather": [
            {"checkpoint": path.name, "step": int(meta["step"]),
             "time_s": float(meta["time_s"]),
             "self_sha256": str(meta["self_sha256"])}
            for path, meta in items
        ],
        "resumed_from": None if resume is None else str(resume),
        "processes": PROCESSES,
        "time_interpolation": (
            "linear in time between consecutive weather checkpoints; the "
            "arrival wind of each substep is the known wind at its end"),
        "intervals": intervals,
        "sidecars": sidecars,
        "ledger": {
            "initial_mass_kg": initial_mass,
            "final": final,
            "relative_mass_drift": drift,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return receipt


__all__ = [
    "CompositionSpec", "FlowFrame", "FlowReader", "PROCESSES", "RECEIPT_NAME",
    "REPLAY_FIXERS", "SIDECAR_SCHEMA", "SIDECAR_SUFFIX", "SPEC_SCHEMA",
    "TRACER_KINDS", "TracerRow", "initial_rows", "load_spec", "mass_kg",
    "read_sidecar", "replay", "sidecar_path", "spec_from_mapping",
]
