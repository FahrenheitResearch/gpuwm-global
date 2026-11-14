"""The dynamic ocean's grid: rows, levels, sea floor, masks, metrics.

Horizontal (design 5): 360 columns of 1 deg longitude on an Arakawa C grid;
rows 1/3 deg for |lat| <= 10, a smooth (geometric) ramp to 1 deg at |lat| =
30, 1 deg poleward, from 78S to 65N (209 rows), every row edge snapped to a
whole arc-minute so the ETOPO 2022 60-arc-second cells aggregate exactly.
Vertical: 50 z* levels, 22 of 10 m to 220 m, then a tanh stretching to
5500 m; partial bottom cells (at least 20 % of the level and 5 m).

Index conventions, everywhere in this package (arrays are (nz, ny, nx), x
periodic, rows south to north, level 0 at the top):

* T points (k, j, i): cell centres;
* U points (k, j, i): the EAST face of cell (j, i), between i and i+1
  (periodic);
* V points (k, j, i): the NORTH face of cell (j, i), between j and j+1; the
  last row's V is a wall (0), as is the south face of row 0;
* interfaces k+1/2: the bottom of level k.

The grid file (``ocean-grid-v1.npz``) is written byte-reproducibly
(:func:`write_npz`: fixed zip timestamps, fixed member order) and enters a
run's identity by its SHA-256.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from ..spectral.constants import EARTH_RADIUS_M, EARTH_ROTATION_RATE_S, GRAVITY_M_S2

SCHEMA = "gpuwm.arwen-global-ocean-grid/v1"
#: Boussinesq reference density and heat capacity (design 4).
RHO0 = 1035.0
CP0 = 3992.0
GRAVITY = GRAVITY_M_S2
OMEGA = EARTH_ROTATION_RATE_S
RADIUS = EARTH_RADIUS_M
#: The default stage-1 layout.
SOUTH_DEG = -78.0
NORTH_DEG = 65.0
NLEV = 50
N_TEN_METRE = 22
BOTTOM_M = 5500.0
MIN_DEPTH_M = 30.0
#: The barotropic step's safety factor on the forward-backward limit.
BAROTROPIC_SAFETY = 0.7
COUPLING_S = 3600.0

#: Straits kept open by hand (design 5): name, latitude, longitude (deg E),
#: the sill depth (m) the opened cell is given at least.  Each opens the
#: one cell that holds the point.
STRAITS = (
    ("Gibraltar", 35.95, 354.4, 300.0),
    ("Bab-el-Mandeb", 12.6, 43.4, 150.0),
    # the strait cell and the Red Sea's last 1-deg cell touch only at a
    # corner; this opens the side between them
    ("Bab-el-Mandeb (north approach)", 13.2, 43.2, 150.0),
    # the same for the upper Gulf of California
    ("Gulf of California (Ballenas channel)", 28.6, 246.5, 200.0),
    ("Hormuz", 26.5, 56.4, 100.0),
    ("Lombok", -8.6, 115.8, 300.0),
    ("Ombai", -8.3, 125.0, 1000.0),
    ("Timor", -10.5, 125.5, 1250.0),
    ("Makassar (Dewakang sill)", -3.0, 118.5, 680.0),
    ("Florida", 24.0, 279.5, 800.0),
)


# -- rows and levels ---------------------------------------------------------

def _hemisphere_edges(pole_deg: float, n_ramp: int = 33) -> list[float]:
    """Row edges from 0 toward ``pole_deg`` (positive), arc-minute snapped:
    30 rows of 20' to 10 deg, ``n_ramp`` geometric rows to 30 deg, 1 deg to
    the pole edge."""
    edges = [0.0]
    for k in range(1, 31):
        edges.append(k / 3.0)
    # Geometric ramp: dphi_i = a q^i, last/first = 3, sum = 20 deg.
    q = 3.0 ** (1.0 / (n_ramp - 1))
    a = 20.0 * (q - 1.0) / (q ** n_ramp - 1.0)
    run = 10.0
    for i in range(n_ramp):
        run += a * q ** i
        edges.append(run)
    edges[-1] = 30.0
    lat = 30.0
    while lat < pole_deg - 1.0e-9:
        lat += 1.0
        edges.append(lat)
    # Snap to whole arc-minutes (the tropical and 1-deg edges are already).
    return [round(e * 60.0) / 60.0 for e in edges]


def row_edges(south_deg: float = SOUTH_DEG, north_deg: float = NORTH_DEG) -> np.ndarray:
    north = _hemisphere_edges(north_deg)
    south = _hemisphere_edges(-south_deg)
    edges = [-e for e in reversed(south)] + north[1:]
    out = np.asarray(edges, dtype=np.float64)
    if not np.all(np.diff(out) > 0):  # pragma: no cover - construction
        raise AssertionError("row edges are not increasing")
    return out


def level_edges(nlev: int = NLEV, n_ten: int = N_TEN_METRE, bottom_m: float = BOTTOM_M
                ) -> np.ndarray:
    """Nominal level interfaces (m, positive down), 0 .. ``bottom_m``: ``n_ten``
    levels of 10 m, then a tanh ramp of thickness whose ceiling is solved so
    the levels end exactly at ``bottom_m``."""
    top = 10.0 * n_ten
    n = nlev - n_ten
    k = np.arange(n, dtype=np.float64)
    ramp = 0.5 * (1.0 + np.tanh((k - 0.45 * n) / (0.22 * n)))
    ramp = (ramp - ramp[0]) / (ramp[-1] - ramp[0])
    lo, hi = 10.0, 2000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = float(np.sum(10.0 + (mid - 10.0) * ramp))
        if total > bottom_m - top:
            hi = mid
        else:
            lo = mid
    dz = 10.0 + (0.5 * (lo + hi) - 10.0) * ramp
    dz *= (bottom_m - top) / dz.sum()
    edges = np.concatenate([[0.0], np.cumsum(np.concatenate([np.full(n_ten, 10.0), dz]))])
    edges[-1] = bottom_m
    return edges


# -- the reproducible file ---------------------------------------------------

def write_npz(path: Path, arrays: dict) -> str:
    """Write ``arrays`` (name -> numpy array, or str stored as a 0-d unicode
    array) as an npz whose bytes depend only on the content; return the
    SHA-256."""
    path = Path(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            value = arrays[name]
            if isinstance(value, str):
                value = np.asarray(value)
            value = np.asarray(value)
            if not value.flags.c_contiguous:
                value = value.copy(order="C")
            member = io.BytesIO()
            np.lib.format.write_array(member, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue())
    data = buffer.getvalue()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def sha256_of(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# -- the runtime grid --------------------------------------------------------

@dataclass
class OceanGrid:
    """Grid geometry, masks and metrics (host numpy; :meth:`on` gives the
    same object's arrays on a device module)."""

    lon_edges_deg: np.ndarray
    lat_edges_deg: np.ndarray
    z_edges_m: np.ndarray
    kmt: np.ndarray            # (ny, nx) wet levels, 0 = land
    bottom_dz_m: np.ndarray    # (ny, nx) thickness of the bottom wet cell
    substeps: int = 0
    periodic: bool = True
    sha256: str | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lon_edges_deg = np.asarray(self.lon_edges_deg, dtype=np.float64)
        self.lat_edges_deg = np.asarray(self.lat_edges_deg, dtype=np.float64)
        self.z_edges_m = np.asarray(self.z_edges_m, dtype=np.float64)
        self.kmt = np.asarray(self.kmt, dtype=np.int32)
        self.bottom_dz_m = np.asarray(self.bottom_dz_m, dtype=np.float64)
        self.nx = self.lon_edges_deg.size - 1
        self.ny = self.lat_edges_deg.size - 1
        self.nz = self.z_edges_m.size - 1
        if self.kmt.shape != (self.ny, self.nx):
            raise ValueError("kmt does not match the rows and columns")
        self._derive()
        if not self.substeps:
            self.substeps = self.barotropic_substeps()
        self._device = {}

    # geometry --------------------------------------------------------------
    def _derive(self):
        R = RADIUS
        lam = np.deg2rad(self.lon_edges_deg)
        phi = np.deg2rad(self.lat_edges_deg)
        dlam = np.diff(lam)
        if not np.allclose(dlam, dlam[0]):
            raise ValueError("the ocean grid's columns must be uniform in longitude")
        self.dlam = float(dlam[0])
        self.lat_t = 0.5 * (self.lat_edges_deg[:-1] + self.lat_edges_deg[1:])
        self.lon_t = 0.5 * (self.lon_edges_deg[:-1] + self.lon_edges_deg[1:])
        phi_t = np.deg2rad(self.lat_t)
        self.phi_t = phi_t
        dz_nom = np.diff(self.z_edges_m)
        self.dz_nom = dz_nom
        self.z_t_nom = 0.5 * (self.z_edges_m[:-1] + self.z_edges_m[1:])
        nz, ny, nx = self.nz, self.ny, self.nx
        k = np.arange(nz)[:, None, None]
        kmt = self.kmt[None]
        dzt = np.where(k < kmt - 1, dz_nom[:, None, None], 0.0)
        dzt = np.where(k == kmt - 1, self.bottom_dz_m[None], dzt)
        self.dzt = dzt.astype(np.float64)
        self.tmask = self.dzt > 0.0
        self.H = self.dzt.sum(axis=0)
        east = np.roll(self.dzt, -1, axis=2)
        dzu = np.minimum(self.dzt, east)
        if not self.periodic:
            dzu[:, :, -1] = 0.0
        self.dzu = dzu.astype(np.float32)
        north = np.zeros_like(self.dzt)
        north[:, :-1, :] = self.dzt[:, 1:, :]
        self.dzv = np.minimum(self.dzt, north).astype(np.float32)
        self.umask = self.dzu > 0
        self.vmask = self.dzv > 0
        self.Hu = self.dzu.astype(np.float64).sum(axis=0)
        self.Hv = self.dzv.astype(np.float64).sum(axis=0)
        self.wet = self.kmt > 0
        # 1-D metrics by row, as (ny, 1).
        self.dxt = (R * np.cos(phi_t) * self.dlam)[:, None]
        self.dyt = (R * np.diff(phi))[:, None]
        self.area = (R * R * self.dlam * (np.sin(phi[1:]) - np.sin(phi[:-1])))[:, None]
        self.dxu = self.dxt
        self.dyu = self.dyt
        self.dxv = (R * np.cos(phi[1:]) * self.dlam)[:, None]
        dyv = np.ones(ny)
        dyv[:-1] = R * np.diff(phi_t)
        self.dyv = dyv[:, None]
        self.f_t = (2.0 * OMEGA * np.sin(phi_t))[:, None]
        self.f_u = self.f_t
        self.f_v = (2.0 * OMEGA * np.sin(phi[1:]))[:, None]
        self.beta_t = (2.0 * OMEGA * np.cos(phi_t) / R)[:, None]
        # Corner (q) points: (j+1/2, i+1/2), north-east corner of cell (j, i).
        self.f_q = self.f_v
        self.dxq = self.dxv
        self.dyq = self.dyv
        self.area_q = self.dxq * self.dyq

    def barotropic_substeps(self, step_s: float = COUPLING_S) -> int:
        H = np.where(self.wet, self.H, 0.0)
        c = np.sqrt(GRAVITY * H)
        inv = np.sqrt(1.0 / self.dxt ** 2 + 1.0 / self.dyt ** 2)
        with np.errstate(divide="ignore"):
            limit = np.where(self.wet, 1.0 / (c * inv + 1e-30), np.inf)
        dt = BAROTROPIC_SAFETY * float(limit.min())
        return int(math.ceil(step_s / dt))

    @property
    def cells(self) -> int:
        return int(self.tmask.sum())

    def volume(self, xp=np):
        return self.on(xp)["dzt"] * self.on(xp)["area"]

    def on(self, xp) -> dict:
        """The arrays the kernels read, on ``xp`` (cached per module)."""
        key = xp.__name__
        if key not in self._device:
            names = ("dzt", "dzu", "dzv", "tmask", "umask", "vmask", "H", "Hu", "Hv", "dxt",
                     "dyt", "area", "dxu", "dyu", "dxv", "dyv", "f_t", "f_u", "f_v", "f_q",
                     "dxq", "dyq", "area_q", "beta_t", "z_t_nom", "z_edges_m", "dz_nom",
                     "kmt", "wet", "lat_t", "lon_t")
            out = {}
            for name in names:
                value = getattr(self, name)
                out[name] = xp.asarray(value)
            out["lat_t2"] = xp.asarray(np.broadcast_to(self.lat_t[:, None], (self.ny, self.nx)).copy())
            out["lat_u2"] = out["lat_t2"]
            out["lat_v2"] = xp.asarray(np.broadcast_to(self.lat_edges_deg[1:, None],
                                                       (self.ny, self.nx)).copy())
            self._device[key] = out
        return self._device[key]

    def to_arrays(self) -> dict:
        return {
            "schema": SCHEMA,
            "lon_edges_deg": self.lon_edges_deg,
            "lat_edges_deg": self.lat_edges_deg,
            "z_edges_m": self.z_edges_m,
            "kmt": self.kmt.astype(np.int16),
            "bottom_dz_m": self.bottom_dz_m,
            "substeps": np.asarray(self.substeps, dtype=np.int32),
            "periodic": np.asarray(self.periodic),
            "provenance_json": json.dumps(self.provenance, sort_keys=True),
        }

    def write(self, path) -> str:
        digest = write_npz(Path(path), self.to_arrays())
        self.sha256 = digest
        return digest

    @classmethod
    def load(cls, path) -> "OceanGrid":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != SCHEMA:
                raise ValueError(f"{path} is not an ocean grid file ({SCHEMA})")
            grid = cls(lon_edges_deg=data["lon_edges_deg"], lat_edges_deg=data["lat_edges_deg"],
                       z_edges_m=data["z_edges_m"], kmt=data["kmt"],
                       bottom_dz_m=data["bottom_dz_m"], substeps=int(data["substeps"]),
                       periodic=bool(data["periodic"]),
                       provenance=json.loads(str(data["provenance_json"])))
        grid.sha256 = sha256_of(path)
        return grid

    def receipt(self) -> dict:
        return {"schema": SCHEMA, "sha256": self.sha256, "nx": self.nx, "ny": self.ny,
                "nz": self.nz, "wet_columns": int(self.wet.sum()), "wet_cells": self.cells,
                "substeps": self.substeps, "south_deg": float(self.lat_edges_deg[0]),
                "north_deg": float(self.lat_edges_deg[-1])}


def partial_bottom(depth_m: np.ndarray, z_edges: np.ndarray, *, min_depth_m: float = MIN_DEPTH_M):
    """``(kmt, bottom_dz)`` from a column depth: the bottom level holds the
    depth, its thickness at least 20 % of the level and 5 m."""
    depth = np.asarray(depth_m, dtype=np.float64)
    wet = depth > 0.0
    d = np.clip(np.where(wet, np.maximum(depth, min_depth_m), 0.0), 0.0, z_edges[-1])
    kb = np.searchsorted(z_edges, d, side="left") - 1          # level holding d
    kb = np.clip(kb, 0, z_edges.size - 2)
    top = z_edges[kb]
    dz = z_edges[kb + 1] - z_edges[kb]
    thick = np.maximum(d - top, np.maximum(0.2 * dz, 5.0))
    thick = np.minimum(thick, dz)
    kmt = np.where(wet, kb + 1, 0).astype(np.int32)
    return kmt, np.where(wet, thick, 0.0)


def idealized(*, lat_edges_deg, lon_edges_deg, z_edges_m, depth_m, periodic=True,
              substeps: int = 0) -> OceanGrid:
    """A grid from a given depth plane (0 = land), for the idealised tests."""
    kmt, bottom = partial_bottom(np.asarray(depth_m, dtype=np.float64),
                                 np.asarray(z_edges_m, dtype=np.float64), min_depth_m=0.0)
    return OceanGrid(lon_edges_deg=lon_edges_deg, lat_edges_deg=lat_edges_deg,
                     z_edges_m=z_edges_m, kmt=kmt, bottom_dz_m=bottom, periodic=periodic,
                     substeps=substeps, provenance={"idealized": True})


# -- the builder (Linux host CPU) --------------------------------------------

def _label_periodic(wet: np.ndarray) -> np.ndarray:
    """4-connected components of ``wet``, periodic in x."""
    from scipy import ndimage

    labels, n = ndimage.label(wet, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # Merge components that touch across the seam.
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for j in range(wet.shape[0]):
        a, b = labels[j, 0], labels[j, -1]
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(a) for a in range(n + 1)])
    return roots[labels]


def _etopo_rows(path: Path, lat0: float, lat1: float):
    """ETOPO 2022 60 s elevation rows covering [lat0, lat1] deg, re-ordered so
    column 0 starts at 0 deg E; returns (z (nrow, 21600) float32, first-row
    edge in arc-minutes from -90)."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        m0 = int(round((lat0 + 90.0) * 60.0))
        m1 = int(round((lat1 + 90.0) * 60.0))
        # Row r covers [-90 + r/60, -90 + (r+1)/60] when lat ascends from -89.99.
        if not (lat[0] < lat[-1] and abs(lat[0] - (-90.0 + 1.0 / 120.0)) < 1e-6):
            raise ValueError(f"{path}: unexpected latitude layout")
        if abs(lon[0] - (-180.0 + 1.0 / 120.0)) > 1e-6:
            raise ValueError(f"{path}: unexpected longitude layout")
        z = np.asarray(ds.variables["z"][m0:m1, :], dtype=np.float32)
    z = np.roll(z, -10800, axis=1)  # column 0 = 0..1/60 deg E
    return z, m0


def build_grid(bed_path, surface_path=None, *, south_deg=SOUTH_DEG, north_deg=NORTH_DEG,
               nlev=NLEV) -> tuple[OceanGrid, dict]:
    """The stage-1 grid from ETOPO 2022 60 s (the bed for depth; the surface
    elevation, when given, so ice shelves count as land).  Returns the grid
    and a report (rows, straits, basins removed)."""
    bed_path = Path(bed_path)
    lat_edges = row_edges(south_deg, north_deg)
    lon_edges = np.arange(361, dtype=np.float64)
    z_edges = level_edges(nlev)
    bed, m0 = _etopo_rows(bed_path, lat_edges[0], lat_edges[-1])
    surf = None
    if surface_path is not None:
        surf, _ = _etopo_rows(Path(surface_path), lat_edges[0], lat_edges[-1])
    ny, nx = lat_edges.size - 1, 360
    wet_frac = np.zeros((ny, nx))
    depth = np.zeros((ny, nx))
    sub_lat = -90.0 + (m0 + np.arange(bed.shape[0]) + 0.5) / 60.0
    weight_row = np.cos(np.deg2rad(sub_lat))
    for j in range(ny):
        a = int(round((lat_edges[j] + 90.0) * 60.0)) - m0
        b = int(round((lat_edges[j + 1] + 90.0) * 60.0)) - m0
        zb = bed[a:b].reshape(b - a, nx, 60).astype(np.float64)
        wet = zb < 0.0
        if surf is not None:
            wet &= surf[a:b].reshape(b - a, nx, 60) < 0.0
        w = weight_row[a:b][:, None, None] * np.ones_like(zb)
        ww = (w * wet).sum(axis=(0, 2))
        wet_frac[j] = ww / w.sum(axis=(0, 2))
        depth[j] = np.where(ww > 0, (w * wet * -zb).sum(axis=(0, 2)) / np.maximum(ww, 1e-30), 0.0)
    ocean = wet_frac >= 0.5
    H = np.where(ocean, np.maximum(depth, MIN_DEPTH_M), 0.0)
    lat_t = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    report: dict = {"rows": int(ny), "straits": []}
    for name, lat, lon, sill in STRAITS:
        j = int(np.searchsorted(lat_edges, lat) - 1)
        i = int(math.floor(lon)) % 360
        before = float(H[j, i])
        H[j, i] = max(H[j, i], sill)
        report["straits"].append({"name": name, "row": j, "col": i, "lat_t": float(lat_t[j]),
                                  "depth_before_m": before, "depth_after_m": float(H[j, i])})
    labels = _label_periodic(H > 0)
    j0 = int(np.argmin(np.abs(lat_t)))
    seed = labels[j0, 180]
    if seed == 0:
        raise ValueError("the equatorial Pacific seed cell is land")
    keep = labels == seed
    removed = {}
    for lab in np.unique(labels[(labels != 0) & ~keep]):
        cells = labels == lab
        jj, ii = np.nonzero(cells)
        removed[int(lab)] = {"cells": int(cells.sum()), "lat": float(lat_t[jj].mean()),
                             "lon": float(ii.mean() + 0.5)}
    report["basins_removed"] = sorted(removed.values(), key=lambda r: -r["cells"])
    H = np.where(keep, H, 0.0)
    for row in report["straits"]:
        row["connected"] = bool(keep[row["row"], row["col"]])
    kmt, bottom = partial_bottom(H, z_edges)
    provenance = {"etopo_bed_sha256": sha256_of(bed_path), "etopo_bed": bed_path.name,
                  "etopo_surface_sha256": None if surface_path is None else sha256_of(surface_path),
                  "straits": [list(s) for s in STRAITS], "min_depth_m": MIN_DEPTH_M,
                  "wet_fraction_threshold": 0.5}
    grid = OceanGrid(lon_edges_deg=lon_edges, lat_edges_deg=lat_edges, z_edges_m=z_edges,
                     kmt=kmt, bottom_dz_m=bottom, provenance=provenance)
    report.update(grid.receipt())
    return grid, report


__all__ = ["CP0", "GRAVITY", "OMEGA", "OceanGrid", "RADIUS", "RHO0", "SCHEMA", "STRAITS",
           "build_grid", "idealized", "level_edges", "partial_bottom", "row_edges",
           "sha256_of", "write_npz"]
