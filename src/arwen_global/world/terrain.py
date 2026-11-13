"""Terrain for the dynamics of an invented world.

An invented world (Pangea, a painted map) arrives as an elevation map on a
regular latitude-longitude grid, often coarser than the model's own (the
PALEOMAP maps are one degree, the T255 grid is about half a degree).  The
dynamics needs that map as a surface geopotential it can carry, and a
spectral model carries only what its truncation can represent.  Three
steps, each with a number the build prints:

1.  **Grid averaging.**  The map is averaged onto the cells of the model's
    Gaussian grid by area overlap (first-order conservative), so the mean
    height of the world is the map's own to rounding whether the map is
    finer or coarser than the grid.  Sea counts as height zero: the model's
    lower boundary over water is the sea surface, never the sea floor, so a
    map's bathymetry never reaches the dynamics.  The same overlap gives
    each cell its land fraction.

2.  **The spectral filter.**  A sharp truncation of a coastline or a ridge
    rings (Gibbs): the model then carries mountains in the open sea and
    pits in the plains.  The filter multiplies each spectral degree ``n``
    by ``exp(-(n (n + 1) / (n0 (n0 + 1))) ** order)`` (the shape of
    Sardeshmukh and Hoskins 1984 as written in Hoskins 1980's orography
    work; recalled, not re-read for this module, and nothing here is
    compared with a published number).  Degree 0 is untouched, so the
    filter never moves the world's mean height.  ``kind="none"`` is a
    sharp truncation, which is what an Earth analysis start does today.

3.  **The false-height report.**  Over open water (cells with no land at
    all in the map) the true height is zero, so whatever height is left
    there is the filter's and the truncation's.  The build prints the
    largest of it, where it is, and how much of the tallest peak the
    filter took.

Two helpers hand the result to the rest of the track: the terrain sampled
on any regular latitude-longitude grid (the invented-world frame of
package 2.6 is such a grid, and the cold start reads its terrain from the
frame), and a resting atmosphere over a given terrain with the means to
measure how much wind the dynamics makes out of it (the rest test).

Nothing here runs for Earth: an Earth start never calls this module and
the cold start's own truncation of the analysis terrain is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)

#: The filter kinds: ``"hoskins"`` (the degree-dependent damping above)
#: and ``"none"`` (a sharp truncation and nothing else).
FILTER_KINDS = ("hoskins", "none")

#: The default filter.  ``cutoff_fraction`` places ``n0`` at that fraction
#: of the truncation, so the same setting means the same thing at every
#: resolution; ``order`` is the exponent of the damping.
#:
#: MEASURED 2026-11-12 at T255 (``tools/arwen_global_world_terrain.py
#: report`` on the PALEOMAP one-degree maps, Scotese and Wright 2018), the
#: deepest false pit over water / on land, and the model's peak against the
#: map's:
#:
#:     map      filter          pit water   pit land    peak (map)
#:     250 Ma   sharp cut        -135 m      -114 m     2368 (2320)
#:     250 Ma   n0 = 1.0 T        -66 m       -54 m     2314
#:     250 Ma   n0 = 0.7 T        -62 m       -43 m     2248
#:     0 Ma     sharp cut        -447 m      -486 m     5947 (6000)
#:     0 Ma     n0 = 1.0 T       -248 m      -226 m     5809
#:     0 Ma     n0 = 0.7 T       -198 m      -167 m     5551
#:
#: n0 = 1.0 T halves the Gibbs pits on both maps and keeps the peak within
#: 3 percent; a stronger filter buys little more against the pits, costs
#: peak height and spreads the coasts further out to sea (the rms height
#: over open water rises again: 4.2, 3.7, 5.1 m on the 250 Ma map).
DEFAULT_CUTOFF_FRACTION = 1.0
DEFAULT_ORDER = 2

#: A cell is open water for the false-height report when no part of the
#: map inside it is land.  The largest height over open water sits next to
#: a coast (a cliff the grid cannot hold spreads into the first sea cells);
#: the report also gives it beyond these many cells from any land, where
#: what is left is ringing.
OFFSHORE_CELLS = 2
FAR_OFFSHORE_CELLS = 6

#: The resting atmosphere of the rest test: a constant-lapse troposphere
#: from sea level to the tropopause, isothermal above.  These are the
#: standard atmosphere's sea-level temperature, lapse rate and tropopause
#: temperature.  Temperature is a function of pressure alone, so in the
#: continuous equations the state is at rest over ANY terrain; whatever
#: wind the model makes is its own discretization's.
REST_SEA_LEVEL_TEMPERATURE_K = 288.15
REST_LAPSE_RATE_K_M = 0.0065
REST_STRATOSPHERE_TEMPERATURE_K = 216.65
REST_SEA_LEVEL_PRESSURE_PA = 1.0e5

#: The ring mountain of the rest test: the shape of the DCMIP 2012 test
#: 2-0 mountain (a cosine bell of radius 3 pi / 4 modulated by cos^2 rings
#: of width pi / 16, centred on the equator at 270 degrees east), 2,000 m
#: tall.  The shape is recalled; the numbers this package measures over it
#: are its own and are not compared with any published result.
RING_HEIGHT_M = 2000.0
RING_RADIUS_RAD = 3.0 * math.pi / 4.0
RING_WIDTH_RAD = math.pi / 16.0
RING_CENTRE_LON_RAD = 3.0 * math.pi / 2.0
RING_CENTRE_LAT_RAD = 0.0


# ----------------------------------------------------------------- filter

@dataclass(frozen=True)
class TerrainFilter:
    """The spectral filter applied to an invented world's terrain."""

    kind: str = "hoskins"
    cutoff_fraction: float = DEFAULT_CUTOFF_FRACTION
    order: int = DEFAULT_ORDER

    def __post_init__(self) -> None:
        if self.kind not in FILTER_KINDS:
            raise ValueError(
                f"terrain filter kind {self.kind!r} is not one of "
                + ", ".join(repr(k) for k in FILTER_KINDS))
        fraction = float(self.cutoff_fraction)
        if not math.isfinite(fraction) or not 0.05 <= fraction <= 4.0:
            raise ValueError(
                f"terrain filter cutoff_fraction {self.cutoff_fraction!r} "
                "must lie in 0.05..4: below it the filter flattens every "
                "mountain the grid can hold, above it the filter does "
                "nothing a sharp truncation does not")
        if isinstance(self.order, bool) or int(self.order) != self.order \
                or not 1 <= int(self.order) <= 8:
            raise ValueError(
                f"terrain filter order {self.order!r} must be an integer "
                "in 1..8")

    def cutoff_degree(self, truncation: int) -> float:
        return float(self.cutoff_fraction) * float(truncation)

    def factors(self, truncation: int) -> np.ndarray:
        """The multiplier of each degree ``0..truncation``; degree 0 is
        exactly one, so the world's mean height never moves."""
        t = int(truncation)
        n = np.arange(t + 1, dtype=np.float64)
        if self.kind == "none":
            return np.ones(t + 1, dtype=np.float64)
        n0 = self.cutoff_degree(t)
        ratio = (n * (n + 1.0)) / (n0 * (n0 + 1.0))
        out = np.exp(-ratio ** int(self.order))
        out[0] = 1.0
        return out

    def record(self, truncation: int | None = None) -> dict:
        out = {"kind": self.kind}
        if self.kind != "none":
            out["cutoff_fraction"] = float(self.cutoff_fraction)
            out["order"] = int(self.order)
            if truncation is not None:
                out["cutoff_degree"] = self.cutoff_degree(truncation)
                out["factor_at_truncation"] = float(
                    self.factors(truncation)[-1])
        return out

    def describe(self, truncation: int | None = None) -> str:
        if self.kind == "none":
            return "sharp truncation, no filter"
        text = (f"exp(-(n(n+1)/(n0(n0+1)))^{int(self.order)}), "
                f"n0 = {float(self.cutoff_fraction):g} x truncation")
        if truncation is not None:
            text += (f" = {self.cutoff_degree(truncation):.1f}, degree "
                     f"{int(truncation)} kept at "
                     f"{float(self.factors(truncation)[-1]):.3f}")
        return text


DEFAULT_FILTER = TerrainFilter()
NO_FILTER = TerrainFilter(kind="none")


# --------------------------------------------------------- grid averaging

def _source_axes(values, latitude_deg, longitude_deg):
    """The map's values south to north with one column per longitude, and
    the cell edges of both axes: latitude edges in sin(latitude), longitude
    edges in degrees (a full ring, possibly starting below zero)."""
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape != (lat.size, lon.size):
        raise ValueError(
            f"the elevation map has shape {data.shape}, expected "
            f"({lat.size}, {lon.size}) from its latitude and longitude")
    if lat.size < 2 or lon.size < 4:
        raise ValueError("the elevation map needs at least 2 latitudes and "
                         "4 longitudes")
    bad = int(data.size - np.count_nonzero(np.isfinite(data)))
    if bad:
        raise ValueError(
            f"the elevation map carries {bad} non-finite values; a masked "
            "or partial map cannot be a world's terrain")
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("the elevation map's coordinates are not finite")
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1]
    if np.any(np.diff(lat) <= 0.0):
        raise ValueError("the elevation map's latitudes are not monotone")
    if lat[0] < -90.0 - 1e-6 or lat[-1] > 90.0 + 1e-6:
        raise ValueError("the elevation map's latitudes leave -90..90")
    dlat = float(np.max(np.diff(lat)))
    if lat[0] > -90.0 + dlat or lat[-1] < 90.0 - dlat:
        raise ValueError(
            f"the elevation map covers latitudes {lat[0]:.3f} to "
            f"{lat[-1]:.3f}; a world's map must reach both poles")
    if np.any(np.diff(lon) <= 0.0):
        raise ValueError("the elevation map's longitudes are not ascending")
    # A grid-registered map repeats its first column 360 degrees on (the
    # PALEOMAP maps run -180 to 180); the repeat is the same place.
    if abs((lon[-1] - lon[0]) - 360.0) <= 1e-6:
        lon = lon[:-1]
        data = data[:, :-1]
    span = float(lon[-1] - lon[0])
    if span >= 360.0:
        raise ValueError("the elevation map's longitudes span more than "
                         "one ring")
    dlon = float(np.max(np.diff(lon)))
    gap = 360.0 - span
    if gap > 1.5 * dlon + 1e-9:
        raise ValueError(
            f"the elevation map's longitudes leave a {gap:.3f} degree gap; "
            "a world's map must be a full ring")
    mid_lat = 0.5 * (lat[1:] + lat[:-1])
    lat_edges = np.concatenate([[-90.0], mid_lat, [90.0]])
    sin_edges = np.sin(np.deg2rad(lat_edges))
    sin_edges[0], sin_edges[-1] = -1.0, 1.0
    ring = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
    lon_edges = 0.5 * (ring[1:] + ring[:-1])
    return data, sin_edges, lon_edges


def _gaussian_edges(grid):
    """Cell edges of the Gaussian grid: latitude edges in sin(latitude)
    from the cumulative quadrature weights (the weights ARE the cells'
    widths in sin(latitude)), longitude edges in degrees."""
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    mu_edges = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    mu_edges[-1] = 1.0
    nlon = int(grid.nlon)
    step = 360.0 / nlon
    lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    lon_edges = np.concatenate([lon - 0.5 * step, [lon[-1] + 0.5 * step]])
    return mu_edges, lon_edges


def _overlap(target_edges, source_edges, *, period=None):
    """``(len(target)-1, len(source)-1)`` overlap lengths of two partitions,
    normalised along each target row to a partition of unity."""
    t0 = target_edges[:-1, None]
    t1 = target_edges[1:, None]
    out = np.zeros((t0.size, source_edges.size - 1), dtype=np.float64)
    shifts = (0.0,) if period is None else (-period, 0.0, period)
    for shift in shifts:
        s0 = source_edges[None, :-1] + shift
        s1 = source_edges[None, 1:] + shift
        out += np.maximum(0.0, np.minimum(t1, s1) - np.maximum(t0, s0))
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("a model cell overlaps no cell of the map")
    return out / total


def average_to_grid(values, latitude_deg, longitude_deg, grid) -> np.ndarray:
    """``values`` (a map on a regular or irregular latitude-longitude grid,
    rows by latitude) averaged over each cell of the Gaussian ``grid`` by
    area overlap.  The area mean of the result equals the map's own area
    mean to rounding."""
    data, sin_edges, lon_edges = _source_axes(values, latitude_deg,
                                              longitude_deg)
    mu_edges, target_lon_edges = _gaussian_edges(grid)
    rows = _overlap(mu_edges, sin_edges)
    cols = _overlap(target_lon_edges, lon_edges, period=360.0)
    return rows @ data @ cols.T


def average_to_latlon(values, latitude_deg, longitude_deg,
                      target_latitude_deg, target_longitude_deg) -> np.ndarray:
    """``values`` averaged by area overlap onto the cells of another
    global latitude-longitude grid (a frame), whose cells are bounded
    half-way between its points; rows come back in the target's order."""
    data, sin_edges, lon_edges = _source_axes(values, latitude_deg,
                                              longitude_deg)
    t_lat = np.asarray(target_latitude_deg, dtype=np.float64).reshape(-1)
    t_lon = np.asarray(target_longitude_deg, dtype=np.float64).reshape(-1)
    descending = t_lat.size > 1 and t_lat[0] > t_lat[-1]
    if t_lon.size > 1 and abs((t_lon[-1] - t_lon[0]) - 360.0) <= 1e-6:
        raise ValueError("the frame repeats its first longitude; give one "
                         "ring of distinct points")
    _, t_sin, t_lon_edges = _source_axes(
        np.zeros((t_lat.size, t_lon.size)), t_lat, t_lon)
    rows = _overlap(t_sin, sin_edges)
    cols = _overlap(t_lon_edges, lon_edges, period=360.0)
    out = rows @ data @ cols.T
    return out[::-1] if descending else out


def map_area_mean(values, latitude_deg, longitude_deg) -> float:
    """The area mean of a map over its own cells (the number grid
    averaging must keep)."""
    data, sin_edges, lon_edges = _source_axes(values, latitude_deg,
                                              longitude_deg)
    area = np.diff(sin_edges)[:, None] * np.diff(lon_edges)[None, :]
    return float(np.sum(data * area) / np.sum(area))


# --------------------------------------------------------------- spectral

def host_transform(transform):
    """A float64 NumPy transform on the same Gaussian grid as ``transform``
    (the build runs on the host whatever the run's backend)."""
    from ..spectral.transform import SphericalHarmonicTransform

    backend = transform.backend
    if backend.name == "numpy" and np.dtype(backend.float_dtype) == np.float64:
        return transform
    grid = transform.grid
    return SphericalHarmonicTransform.create(
        int(transform.truncation), nlat=int(grid.nlat), nlon=int(grid.nlon),
        backend="numpy", precision="float64",
    )


def filter_coefficients(coefficients, truncation: int,
                        terrain_filter: TerrainFilter = DEFAULT_FILTER):
    """Spectral coefficients ``(..., n, m)`` with each degree scaled."""
    factors = terrain_filter.factors(int(truncation))
    return np.asarray(coefficients) * factors[:, None]


def sample_on_latlon(coefficients, truncation: int, latitude_deg,
                     longitude_deg, *, latitude_chunk: int = 48) -> np.ndarray:
    """The spectral field ``coefficients`` (``(T+1, T+1)``, this package's
    complex-orthonormal convention) evaluated on a regular longitude ring
    at any latitudes, rows in the order given.

    The longitudes must be one full regular ring; the latitudes may be
    anything in -90..90, poles included.  The values are the field's
    own at those points (a ring of ``2 T`` points or fewer is synthesized
    on a finer ring and subsampled, so nothing folds).  This is how an
    invented-world frame carries the filtered terrain: the cold start
    interpolates the frame to its own grid and truncates it again, which
    returns the field closely only when the frame is finer than the model
    grid (``tools/arwen_global_world_terrain.py frame`` measures it).

    MEASURED 2026-11-12, the 250 Ma map at T255 with the default filter,
    frame -> bilinear onto the Gaussian grid -> truncation, against the
    terrain the frame was sampled from: a one-degree frame (181 x 360)
    lands up to 249 m away (rms 9.8 m) and drops the peak from 2,314 to
    2,201 m; a quarter-degree frame (721 x 1440) lands up to 21 m away
    (rms 0.95 m) with the peak at 2,308 m.  An invented world's frame is
    therefore built at a quarter degree, not on its map's own grid.
    """
    from ..spectral.legendre import normalized_associated_legendre_values

    t = int(truncation)
    coeff = np.asarray(coefficients, dtype=np.complex128)
    if coeff.shape != (t + 1, t + 1):
        raise ValueError(f"coefficients have shape {coeff.shape}, expected "
                         f"({t + 1}, {t + 1})")
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    nlon = lon.size
    if nlon < 4:
        raise ValueError("the frame's longitude ring needs at least 4 points")
    step = np.diff(lon)
    if np.max(np.abs(step - 360.0 / nlon)) > 1e-6:
        raise ValueError("the frame's longitudes must be one regular ring "
                         "of 360 degrees")
    # Synthesize on a ring that holds every order, then take every
    # ``stride``-th point: the frame's own points, with nothing folded.
    stride = max(1, -(-(2 * t + 1) // nlon))
    ring = nlon * stride
    if np.any(np.abs(lat) > 90.0 + 1e-9):
        raise ValueError("the frame's latitudes leave -90..90")
    phase = np.exp(1j * np.arange(t + 1) * math.radians(float(lon[0])))
    out = np.empty((lat.size, nlon), dtype=np.float64)
    for start in range(0, lat.size, int(latitude_chunk)):
        stop = min(lat.size, start + int(latitude_chunk))
        sin_lat = np.clip(np.sin(np.deg2rad(lat[start:stop])), -1.0, 1.0)
        basis = normalized_associated_legendre_values(t, sin_lat)
        amplitude = np.einsum("nm,nmj->jm", coeff, basis) * phase[None, :]
        spectrum = np.zeros((stop - start, ring // 2 + 1), dtype=np.complex128)
        spectrum[:, : t + 1] = amplitude * ring
        out[start:stop] = np.fft.irfft(spectrum, n=ring, axis=1)[:, ::stride]
    return out


# ---------------------------------------------------------------- report

def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """``mask`` grown by ``cells`` in every direction (periodic in
    longitude, clipped at the poles)."""
    out = mask.copy()
    for _ in range(int(cells)):
        grown = out.copy()
        grown |= np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1)
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        out = grown
    return out


@dataclass(frozen=True)
class FalseHeightReport:
    """What the filter and the truncation leave behind, in metres."""

    truncation: int
    terrain_filter: dict
    open_water_cells: int
    largest_false_height_m: float
    largest_false_height_lat_deg: float
    largest_false_height_lon_deg: float
    highest_over_water_m: float
    lowest_over_water_m: float
    rms_over_water_m: float
    largest_offshore_false_height_m: float
    largest_far_offshore_false_height_m: float
    lowest_over_land_m: float
    map_peak_m: float
    averaged_peak_m: float
    model_peak_m: float
    map_mean_m: float
    averaged_mean_m: float
    model_mean_m: float
    land_share: float

    def record(self) -> dict:
        return {name: getattr(self, name)
                for name in self.__dataclass_fields__}

    def lines(self) -> list[str]:
        return [
            f"terrain at T{self.truncation}: filter "
            + TerrainFilter(**{k: v for k, v in self.terrain_filter.items()
                               if k in ("kind", "cutoff_fraction", "order")}
                            ).describe(self.truncation),
            f"  largest false height over open water: "
            f"{self.largest_false_height_m:.1f} m at "
            f"{self.largest_false_height_lat_deg:.2f}N "
            f"{self.largest_false_height_lon_deg:.2f}E "
            f"(range {self.lowest_over_water_m:.1f} to "
            f"{self.highest_over_water_m:.1f} m, rms "
            f"{self.rms_over_water_m:.1f} m over "
            f"{self.open_water_cells} open-water cells)",
            f"  more than {OFFSHORE_CELLS} cells from any land: "
            f"{self.largest_offshore_false_height_m:.1f} m; more than "
            f"{FAR_OFFSHORE_CELLS}: "
            f"{self.largest_far_offshore_false_height_m:.1f} m",
            f"  lowest height on land: {self.lowest_over_land_m:.1f} m",
            f"  peak: map {self.map_peak_m:.0f} m, grid average "
            f"{self.averaged_peak_m:.0f} m, model {self.model_peak_m:.0f} m",
            f"  mean height: map {self.map_mean_m:.3f} m, grid average "
            f"{self.averaged_mean_m:.3f} m, model {self.model_mean_m:.3f} m",
        ]

    def text(self) -> str:
        return "\n".join(self.lines())


def false_height_report(model_height_m, averaged_height_m, land_fraction,
                        grid, terrain_filter: TerrainFilter, *,
                        map_peak_m: float, map_mean_m: float
                        ) -> FalseHeightReport:
    model = np.asarray(model_height_m, dtype=np.float64)
    averaged = np.asarray(averaged_height_m, dtype=np.float64)
    land = np.asarray(land_fraction, dtype=np.float64)
    water = land <= 0.0
    lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    cell = np.broadcast_to(weights[:, None], model.shape)
    nan = float("nan")
    if np.any(water):
        over = np.where(water, np.abs(model), -1.0)
        j, i = np.unravel_index(int(np.argmax(over)), model.shape)
        largest = float(model[j, i])
        largest_lat, largest_lon = float(lat[j]), float(lon[i])
        highest = float(np.max(model[water]))
        lowest = float(np.min(model[water]))
        rms = float(np.sqrt(np.sum(model[water] ** 2 * cell[water])
                            / np.sum(cell[water])))
    else:
        largest = largest_lat = largest_lon = highest = lowest = rms = nan
    offshore = water & ~_dilate(~water, OFFSHORE_CELLS)
    largest_offshore = (float(np.max(np.abs(model[offshore])))
                        if np.any(offshore) else nan)
    far = water & ~_dilate(~water, FAR_OFFSHORE_CELLS)
    largest_far = float(np.max(np.abs(model[far]))) if np.any(far) else nan
    on_land = land > 0.0
    lowest_land = float(np.min(model[on_land])) if np.any(on_land) else nan
    return FalseHeightReport(
        truncation=int(grid.truncation),
        terrain_filter=terrain_filter.record(int(grid.truncation)),
        open_water_cells=int(np.count_nonzero(water)),
        largest_false_height_m=largest,
        largest_false_height_lat_deg=largest_lat,
        largest_false_height_lon_deg=largest_lon,
        highest_over_water_m=highest,
        lowest_over_water_m=lowest,
        rms_over_water_m=rms,
        largest_offshore_false_height_m=largest_offshore,
        largest_far_offshore_false_height_m=largest_far,
        lowest_over_land_m=lowest_land,
        map_peak_m=float(map_peak_m),
        averaged_peak_m=float(np.max(averaged)),
        model_peak_m=float(np.max(model)),
        map_mean_m=float(map_mean_m),
        averaged_mean_m=float(grid.global_mean(averaged)),
        model_mean_m=float(grid.global_mean(model)),
        land_share=float(grid.global_mean(land)),
    )


# ------------------------------------------------------------ the build

@dataclass(frozen=True)
class DynamicsTerrain:
    """An invented world's terrain as the dynamics will carry it.

    ``height_m`` is on the model's Gaussian grid, ``coefficients`` is the
    same field in the spectral basis (float64), ``averaged_m`` and
    ``land_fraction`` are the map averaged to the grid before any
    filtering, and ``report`` is the false-height report.
    """

    height_m: np.ndarray
    coefficients: np.ndarray
    averaged_m: np.ndarray
    land_fraction: np.ndarray
    report: FalseHeightReport
    terrain_filter: TerrainFilter = field(default=DEFAULT_FILTER)

    @property
    def surface_geopotential(self) -> np.ndarray:
        return GRAVITY_M_S2 * self.height_m

    def on_latlon(self, latitude_deg, longitude_deg) -> np.ndarray:
        """The same terrain on a regular latitude-longitude frame."""
        return sample_on_latlon(self.coefficients, self.report.truncation,
                                latitude_deg, longitude_deg)


def land_height(elevation_m, sea_level_m: float = 0.0):
    """``(height above the sea surface, land indicator)`` of a map: sea
    and anything at or below sea level is height zero, because the
    model's lower boundary over water is the water's surface."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    above = elevation - float(sea_level_m)
    land = above > 0.0
    return np.where(land, above, 0.0), land.astype(np.float64)


def dynamics_terrain(elevation_m, latitude_deg, longitude_deg, transform, *,
                     sea_level_m: float = 0.0,
                     terrain_filter: TerrainFilter = DEFAULT_FILTER
                     ) -> DynamicsTerrain:
    """Average a world's elevation map to the model grid, filter it in the
    spectral basis and report the false height it leaves.

    ``transform`` may be the run's own (any backend); the build runs on a
    float64 host transform on the same grid.  ``sea_level_m`` is the
    world's sea level against the map's datum.
    """
    host = host_transform(transform)
    grid = host.grid
    height, land = land_height(elevation_m, sea_level_m)
    averaged = average_to_grid(height, latitude_deg, longitude_deg, grid)
    land_fraction = np.clip(
        average_to_grid(land, latitude_deg, longitude_deg, grid), 0.0, 1.0)
    # Overlap weights of exact zeros give exact zeros; a cell that is all
    # sea in the map is open water in the report, and nowhere else.
    coeff = np.asarray(host.forward(averaged), dtype=np.complex128)
    coeff = np.asarray(host.project(filter_coefficients(
        coeff, host.truncation, terrain_filter)), dtype=np.complex128)
    model = np.asarray(host.inverse(coeff), dtype=np.float64)
    report = false_height_report(
        model, averaged, land_fraction, grid, terrain_filter,
        map_peak_m=float(np.max(height)),
        map_mean_m=map_area_mean(height, latitude_deg, longitude_deg),
    )
    return DynamicsTerrain(
        height_m=model, coefficients=coeff, averaged_m=averaged,
        land_fraction=land_fraction, report=report,
        terrain_filter=terrain_filter,
    )


#: The frame an invented world's terrain is handed to the cold start on:
#: regular, pole to pole, a quarter degree (see :func:`sample_on_latlon`
#: for what a one-degree frame loses).
FRAME_STEP_DEG = 0.25


def frame_planes(built: "DynamicsTerrain", elevation_m, latitude_deg,
                 longitude_deg, *, sea_level_m: float = 0.0,
                 step_deg: float = FRAME_STEP_DEG):
    """``(latitude_deg, longitude_deg, terrain_m, land_fraction)`` of the
    invented-world frame: a regular pole-to-pole grid of ``step_deg``,
    the filtered terrain sampled on it exactly, and the map's land
    averaged onto its cells by area overlap.  These are the four planes
    package 2.6's ``WorldMap`` takes."""
    step = float(step_deg)
    nlat = int(round(180.0 / step)) + 1
    nlon = int(round(360.0 / step))
    if abs((nlat - 1) * step - 180.0) > 1e-9 or abs(nlon * step - 360.0) > 1e-9:
        raise ValueError(f"a frame step of {step_deg} degrees does not "
                         "divide the globe")
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.arange(nlon, dtype=np.float64) * (360.0 / nlon)
    terrain = built.on_latlon(lat, lon)
    _, land = land_height(elevation_m, sea_level_m)
    land_fraction = np.clip(
        average_to_latlon(land, latitude_deg, longitude_deg, lat, lon),
        0.0, 1.0)
    return lat, lon, terrain, land_fraction


def dynamics_terrain_for_world(world, elevation_m, latitude_deg,
                               longitude_deg, transform, *,
                               terrain_filter: TerrainFilter = DEFAULT_FILTER
                               ) -> DynamicsTerrain:
    """:func:`dynamics_terrain` at the world's own sea level.  Refused for
    Earth: an Earth start keeps the analysis terrain and its own
    truncation, and no world code runs for it."""
    if getattr(world, "is_earth", True):
        raise ValueError(
            "the dynamics terrain of an invented world was asked for an "
            "Earth world; an Earth start keeps its analysis terrain")
    return dynamics_terrain(
        elevation_m, latitude_deg, longitude_deg, transform,
        sea_level_m=float(world.sea.level_m), terrain_filter=terrain_filter)


# ------------------------------------------------------- the rest test

def ring_mountain(grid, height_m: float = RING_HEIGHT_M) -> np.ndarray:
    """The ring mountain on ``grid``, in metres."""
    lat = np.asarray(grid.lat_rad, dtype=np.float64)[:, None]
    lon = np.asarray(grid.lon_rad, dtype=np.float64)[None, :]
    cos_r = (math.sin(RING_CENTRE_LAT_RAD) * np.sin(lat)
             + math.cos(RING_CENTRE_LAT_RAD) * np.cos(lat)
             * np.cos(lon - RING_CENTRE_LON_RAD))
    r = np.arccos(np.clip(cos_r, -1.0, 1.0))
    shape = 0.5 * (1.0 + np.cos(math.pi * r / RING_RADIUS_RAD)) \
        * np.cos(math.pi * r / RING_WIDTH_RAD) ** 2
    return float(height_m) * np.where(r < RING_RADIUS_RAD, shape, 0.0)


class _HostBackend:
    """The minimal backend ``HybridCoordinate.pressure`` reads, in float64
    on the host."""

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


def rest_temperature(p_pa):
    """Temperature of the resting atmosphere at pressure ``p_pa``."""
    exponent = DRY_AIR_GAS_CONSTANT * REST_LAPSE_RATE_K_M / GRAVITY_M_S2
    p = np.asarray(p_pa, dtype=np.float64)
    troposphere = REST_SEA_LEVEL_TEMPERATURE_K * (
        p / REST_SEA_LEVEL_PRESSURE_PA) ** exponent
    return np.maximum(troposphere, REST_STRATOSPHERE_TEMPERATURE_K)


def rest_surface_pressure(height_m):
    """Surface pressure of the resting atmosphere over ``height_m``; the
    closed form of a constant-lapse troposphere, refused above its
    tropopause."""
    height = np.asarray(height_m, dtype=np.float64)
    tropopause = (REST_SEA_LEVEL_TEMPERATURE_K
                  - REST_STRATOSPHERE_TEMPERATURE_K) / REST_LAPSE_RATE_K_M
    if float(np.max(height)) >= tropopause:
        raise ValueError(
            f"terrain of {float(np.max(height)):.0f} m reaches the resting "
            f"atmosphere's tropopause at {tropopause:.0f} m")
    exponent = GRAVITY_M_S2 / (DRY_AIR_GAS_CONSTANT * REST_LAPSE_RATE_K_M)
    return REST_SEA_LEVEL_PRESSURE_PA * (
        1.0 - REST_LAPSE_RATE_K_M * height
        / REST_SEA_LEVEL_TEMPERATURE_K) ** exponent


def resting_state(transform, vertical, surface_geopotential, surface,
                  physics_state=None):
    """``(bundle, model_surface_geopotential)``: an atmosphere at rest
    over ``surface_geopotential``, dry, in this model's own variables.

    The terrain is first put through the model's own truncation (a field
    that is already band-limited comes back unchanged to rounding), the
    surface pressure follows from it in closed form, and temperature is
    the resting profile at each model level's pressure.  ``surface`` and
    ``physics_state`` are carried through unchanged; the rest test runs
    with physics off and nothing reads them.
    """
    from ..constants import GRID_TRACERS
    from ..lnps_reference import LNPS_REFERENCE, lnps_from_anomaly_coefficients
    from ..state import ArwenGlobalState, MoistHybridState, PhysicsState

    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    host = host_transform(transform)
    phis = np.asarray(backend.to_numpy(surface_geopotential)
                      if not isinstance(surface_geopotential, np.ndarray)
                      else surface_geopotential, dtype=np.float64)
    phis = np.asarray(host.inverse(host.forward(phis)), dtype=np.float64)
    ps = rest_surface_pressure(phis / GRAVITY_M_S2)
    lnps = host.project(host.forward(np.log(ps) - LNPS_REFERENCE))
    # The level pressures follow the surface pressure the model will
    # actually carry (the truncated one), so the profile is the model's.
    ps_model = np.exp(np.asarray(host.inverse(lnps), dtype=np.float64)
                      + LNPS_REFERENCE)
    p_full = np.asarray(vertical.pressure(ps_model, _HostBackend())["p_full"],
                        dtype=np.float64)
    theta = rest_temperature(p_full) / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    nlev = int(vertical.nlev)
    zeros_spectral = xp.zeros((nlev, *transform.spectral_shape),
                              dtype=backend.complex_dtype)
    zeros_grid = xp.zeros((nlev, *transform.grid.shape), dtype=dtype)
    atmosphere = MoistHybridState(
        vorticity=zeros_spectral.copy(),
        divergence=zeros_spectral.copy(),
        theta=transform.project(transform.forward(
            backend.asarray(theta, dtype=dtype))),
        log_surface_pressure=lnps_from_anomaly_coefficients(
            transform, transform.project(backend.asarray(
                lnps, dtype=backend.complex_dtype))),
        qv=zeros_spectral.copy(),
        **{name: zeros_grid.copy() for name in GRID_TRACERS},
    )
    bundle = ArwenGlobalState(atmosphere, surface,
                              physics_state if physics_state is not None
                              else PhysicsState())
    return bundle, backend.asarray(phis, dtype=dtype)


def _wind_sample(model, bundle):
    host = model.transform.backend.to_numpy
    g = model.grid_state(bundle.atmosphere, only=("u", "v"))
    u = np.asarray(host(g["u"]), dtype=np.float64)
    v = np.asarray(host(g["v"]), dtype=np.float64)
    speed = np.hypot(u, v)
    k, j, i = np.unravel_index(int(np.argmax(speed)), speed.shape)
    grid = model.transform.grid
    weights = np.asarray(grid.quadrature_weights, dtype=np.float64)
    mean_square = float(np.sum((speed ** 2).mean(axis=(0, 2)) * weights)
                        / np.sum(weights))
    return {
        "max_wind_m_s": float(speed[k, j, i]),
        "level": int(k),
        "lat_deg": float(grid.latitude_deg[j]),
        "lon_deg": float(grid.longitude_deg[i]),
        "rms_wind_m_s": math.sqrt(mean_square),
    }


def rest_test(model, bundle, *, dt_s: float, days: float,
              samples_per_day: int = 24, progress=None) -> dict:
    """Step ``model`` from ``bundle`` for ``days`` and return the largest
    wind of each day (sampled ``samples_per_day`` times a day, at every
    level), with where it was."""
    dt = float(dt_s)
    steps_per_day = int(round(86400.0 / dt))
    if abs(steps_per_day * dt - 86400.0) > 1e-6:
        raise ValueError(f"dt {dt} s does not divide a day")
    every = max(1, steps_per_day // int(samples_per_day))
    total = int(round(float(days) * steps_per_day))
    rows = []
    day_best = None
    started = time.perf_counter()
    state = bundle
    for step in range(1, total + 1):
        state, _metrics = model.step(state, dt)
        if step % every == 0 or step == total:
            sample = _wind_sample(model, state)
            sample["hour"] = step * dt / 3600.0
            if day_best is None or sample["max_wind_m_s"] > day_best["max_wind_m_s"]:
                day_best = sample
            if progress is not None:
                progress(sample)
        if step % steps_per_day == 0 or step == total:
            day_best = dict(day_best)
            day_best["day"] = int(math.ceil(step / steps_per_day))
            rows.append(day_best)
            day_best = None
    return {
        "dt_s": dt,
        "days": float(days),
        "steps": total,
        "samples_per_day": int(samples_per_day),
        "wall_s": float(time.perf_counter() - started),
        "per_day": rows,
        "largest_m_s": max(r["max_wind_m_s"] for r in rows) if rows else 0.0,
        "final_state": state,
    }


__all__ = [
    "DEFAULT_CUTOFF_FRACTION",
    "DEFAULT_FILTER",
    "DEFAULT_ORDER",
    "DynamicsTerrain",
    "FILTER_KINDS",
    "FRAME_STEP_DEG",
    "FalseHeightReport",
    "FAR_OFFSHORE_CELLS",
    "NO_FILTER",
    "OFFSHORE_CELLS",
    "RING_HEIGHT_M",
    "TerrainFilter",
    "average_to_grid",
    "average_to_latlon",
    "dynamics_terrain",
    "dynamics_terrain_for_world",
    "false_height_report",
    "filter_coefficients",
    "frame_planes",
    "host_transform",
    "land_height",
    "map_area_mean",
    "rest_surface_pressure",
    "rest_temperature",
    "rest_test",
    "resting_state",
    "ring_mountain",
    "sample_on_latlon",
]
