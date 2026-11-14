"""The dynamic ocean's initial state and climatological targets (design 10).

``build_init`` writes ``ocean-init-<YYYYMMDDHH>.npz`` (byte-reproducible,
schema :data:`SCHEMA`):

* temperature, salinity, velocity and free surface from NOAA GODAS pentads
  (GRIB1, arwen_global.ocean_dyn.grib1), interpolated linearly in time
  between the two pentads whose centres bracket the valid time (a pentad
  dated D is the mean of D-4 .. D, centred on D-2 days);
* each GODAS level creep-filled over land (iterative neighbour averaging),
  then interpolated linearly in depth to the model's level centres and
  bilinearly to its T, U and V points; below GODAS's deepest level (4478 m)
  the WOA23 1991-2020 annual climatology;
* the velocities' depth mean removed (the barotropic flow is the model's own
  to spin up; GODAS's on its own sea floor would start a barotropic shock),
  the free surface GODAS's SSHG less its area mean;
* GODAS gives u and v on the same latitude-longitude coordinates as T in its
  GRIB records; they are taken at those coordinates (a stated choice);
* the targets: WOA23 monthly sea-surface salinity (SSS restoring) and WOA23
  monthly temperature and salinity on the sponge rows (60-65N; monthly to
  1500 m, annual below).

``godas_forcing`` reads the pentads' surface fluxes (THFLX, UFLX, VFLX,
SLTFL) onto the ocean grid for the ocean-only driver.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .grid import OceanGrid, sha256_of, write_npz

SCHEMA = "gpuwm.arwen-global-ocean-init/v1"
GODAS_BASE = "https://ftp.cpc.ncep.noaa.gov/godas/pentad"


def creep_fill(field, valid=None, max_iter=2000):
    """Fill NaN (or ~valid) cells of a 2-D field by repeated averaging of
    their filled 4-neighbours (periodic in x)."""
    f = np.array(field, dtype=np.float64)
    ok = np.isfinite(f) if valid is None else (np.asarray(valid, bool) & np.isfinite(f))
    if not ok.any():
        return np.zeros_like(f)
    f = np.where(ok, f, 0.0)
    for _ in range(max_iter):
        if ok.all():
            break
        s = np.zeros_like(f)
        n = np.zeros_like(f)
        for sh, ax in ((1, 1), (-1, 1)):
            s += np.roll(f * ok, sh, axis=ax)
            n += np.roll(ok, sh, axis=ax)
        s[1:] += (f * ok)[:-1]
        n[1:] += ok[:-1]
        s[:-1] += (f * ok)[1:]
        n[:-1] += ok[1:]
        new = (~ok) & (n > 0)
        f = np.where(new, s / np.maximum(n, 1), f)
        ok = ok | new
    return f


def _interp_lat(field, src_lat, dst_lat):
    """Linear in latitude, clamped at the ends; field (..., nlat, nlon)."""
    idx = np.clip(np.searchsorted(src_lat, dst_lat) - 1, 0, src_lat.size - 2)
    t = np.clip((dst_lat - src_lat[idx]) / (src_lat[idx + 1] - src_lat[idx]), 0.0, 1.0)
    a = np.take(field, idx, axis=-2)
    b = np.take(field, idx + 1, axis=-2)
    return a + t[:, None] * (b - a)


def _interp_lon(field, src_lon, dst_lon):
    """Linear in longitude, periodic; src_lon uniform ascending."""
    d = src_lon[1] - src_lon[0]
    x = ((dst_lon - src_lon[0]) / d) % src_lon.size
    i0 = np.floor(x).astype(int) % src_lon.size
    i1 = (i0 + 1) % src_lon.size
    t = x - np.floor(x)
    return np.take(field, i0, axis=-1) * (1 - t) + np.take(field, i1, axis=-1) * t


def _interp_depth(field, src_z, dst_z):
    """Linear in depth (axis 0), constant beyond the ends."""
    out = np.empty((dst_z.size,) + field.shape[1:])
    for k, z in enumerate(dst_z):
        if z <= src_z[0]:
            out[k] = field[0]
        elif z >= src_z[-1]:
            out[k] = field[-1]
        else:
            j = int(np.searchsorted(src_z, z) - 1)
            t = (z - src_z[j]) / (src_z[j + 1] - src_z[j])
            out[k] = field[j] * (1 - t) + field[j + 1] * t
    return out


def pentad_center(date: datetime) -> datetime:
    return date - timedelta(days=2)


def pentad_dates(start: datetime, end: datetime, anchor=datetime(2026, 7, 29)):
    """GODAS pentad file dates whose centres cover [start, end] (pentads are
    every 5 days; ``anchor`` is one of them)."""
    d0 = anchor + timedelta(days=5 * int(np.floor(((start - anchor).days + 2) / 5.0)) - 5)
    out = []
    d = d0
    while pentad_center(d) <= end + timedelta(days=5):
        out.append(d)
        d += timedelta(days=5)
    return out


def bracket(dates, when):
    """``(d0, d1, w1)``: the two pentads whose centres bracket ``when`` and the
    weight of the second."""
    centers = [pentad_center(d) for d in dates]
    for a, b, ca, cb in zip(dates[:-1], dates[1:], centers[:-1], centers[1:]):
        if ca <= when <= cb:
            return a, b, (when - ca).total_seconds() / (cb - ca).total_seconds()
    raise ValueError(f"no two pentads bracket {when:%Y-%m-%d %H}Z (have {dates[0]:%Y%m%d}.."
                     f"{dates[-1]:%Y%m%d})")


def _woa(path, var):
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        v = ds.variables[var][0]
        z = np.asarray(ds.variables["depth"][:], dtype=np.float64)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        data = np.ma.filled(v.astype(np.float64), np.nan)
    # to 0..360 E
    order = np.argsort(lon % 360.0)
    return data[..., order], z, lat, (lon % 360.0)[order]


def _to_grid_3d(field, src_lat, src_lon, src_z, grid: OceanGrid, lats, lons):
    """Creep-fill each source level, then depth then horizontal interpolation
    to the given destination points (lats (ny,), lons (nx,))."""
    filled = np.stack([creep_fill(level) for level in field])
    col = _interp_depth(filled, src_z, grid.z_t_nom)
    return _interp_lon(_interp_lat(col, src_lat, lats), src_lon, lons)


def _to_grid_2d(field, src_lat, src_lon, lats, lons):
    return _interp_lon(_interp_lat(creep_fill(field), src_lat, lats), src_lon, lons)


def godas_state(paths_weights, grid: OceanGrid):
    """Time-weighted GODAS fields on the model grid: theta, salt (T points),
    u (U points), v (V points), eta."""
    from .grib1 import godas_fields

    acc = None
    for path, w in paths_weights:
        f = godas_fields(path)
        part = {k: f[k] for k in ("POT", "SALTY", "UOGRD", "VOGRD", "SSHG", "THFLX", "UFLX",
                                  "VFLX", "SLTFL")}
        if acc is None:
            acc = {k: w * v for k, v in part.items()}
            meta = f
        else:
            for k in acc:
                acc[k] = acc[k] + w * part[k]
    lat, lon, z = meta["lats"], meta["lons"], meta["depth_m"]
    lat_u, lon_u = grid.lat_t, grid.lon_edges_deg[1:]
    lat_v, lon_v = grid.lat_edges_deg[1:], grid.lon_t
    out = {
        "theta": _to_grid_3d(acc["POT"], lat, lon, z, grid, grid.lat_t, grid.lon_t),
        "salt": _to_grid_3d(acc["SALTY"], lat, lon, z, grid, grid.lat_t, grid.lon_t),
        "u": _to_grid_3d(np.where(np.isfinite(acc["UOGRD"]), acc["UOGRD"], 0.0), lat, lon, z,
                         grid, lat_u, lon_u),
        "v": _to_grid_3d(np.where(np.isfinite(acc["VOGRD"]), acc["VOGRD"], 0.0), lat, lon, z,
                         grid, lat_v, lon_v),
        "eta": _to_grid_2d(acc["SSHG"], lat, lon, grid.lat_t, grid.lon_t),
        "godas_depth_max_m": float(z[-1]),
    }
    return out


def _deep_fill(state, grid, woa_t00, woa_s00):
    """Below the deepest GODAS level, WOA23 annual."""
    zmax = state["godas_depth_max_m"]
    deep = grid.z_t_nom > zmax
    if not deep.any():
        return
    for key, path, var in (("theta", woa_t00, "t_an"), ("salt", woa_s00, "s_an")):
        data, z, lat, lon = _woa(path, var)
        col = _to_grid_3d(data, lat, lon, z, grid, grid.lat_t, grid.lon_t)
        state[key][deep] = col[deep]


def targets_from_woa(grid: OceanGrid, woa_dir, sponge_lat0=60.0):
    """Monthly SSS (12, ny, nx) and sponge-row T and S (12, nz, nrows, nx)."""
    woa_dir = Path(woa_dir)
    rows = np.nonzero(grid.lat_t > sponge_lat0)[0]
    lat_rows = grid.lat_t[rows]
    t_ann, z_ann, la, lo = _woa(woa_dir / "woa23_decav91C0_t00_01.nc", "t_an")
    s_ann, _, _, _ = _woa(woa_dir / "woa23_decav91C0_s00_01.nc", "s_an")
    sss = np.zeros((12, grid.ny, grid.nx))
    st = np.zeros((12, grid.nz, rows.size, grid.nx))
    ss = np.zeros_like(st)
    t_ann_rows = _to_grid_3d(t_ann, la, lo, z_ann, grid, lat_rows, grid.lon_t)
    s_ann_rows = _to_grid_3d(s_ann, la, lo, z_ann, grid, lat_rows, grid.lon_t)
    for m in range(12):
        tm, zm, _, _ = _woa(woa_dir / f"woa23_decav91C0_t{m + 1:02d}_01.nc", "t_an")
        sm, _, _, _ = _woa(woa_dir / f"woa23_decav91C0_s{m + 1:02d}_01.nc", "s_an")
        sss[m] = _to_grid_2d(sm[0], la, lo, grid.lat_t, grid.lon_t)
        tr = _to_grid_3d(tm, la, lo, zm, grid, lat_rows, grid.lon_t)
        sr = _to_grid_3d(sm, la, lo, zm, grid, lat_rows, grid.lon_t)
        deep = grid.z_t_nom > zm[-1]
        tr[deep] = t_ann_rows[deep]
        sr[deep] = s_ann_rows[deep]
        st[m], ss[m] = tr, sr
    return {"sss_month": sss.astype(np.float32), "sponge_t": st.astype(np.float32),
            "sponge_s": ss.astype(np.float32), "sponge_rows": rows.astype(np.int32)}


def build_init(grid: OceanGrid, *, valid: datetime, godas_dir, woa_dir, out_path,
               barotropic: str = "remove") -> dict:
    """``barotropic``: "remove" (the default) drops GODAS's depth-mean
    velocity, so the model spins its own barotropic flow up on its own sea
    floor; "godas" keeps GODAS's full velocity (measured: on this grid it
    drains the Amazon shelf, eta -40 m, in 2.5 days of the spin-up)."""
    godas_dir = Path(godas_dir)
    files = sorted(godas_dir.glob("godas.P.*.grb"))
    dates = [datetime.strptime(p.name.split(".")[2], "%Y%m%d") for p in files]
    d0, d1, w1 = bracket(dates, valid.replace(tzinfo=None))
    p0 = godas_dir / f"godas.P.{d0:%Y%m%d}.grb"
    p1 = godas_dir / f"godas.P.{d1:%Y%m%d}.grb"
    state = godas_state([(p0, 1.0 - w1), (p1, w1)], grid)
    woa_dir = Path(woa_dir)
    _deep_fill(state, grid, woa_dir / "woa23_decav91C0_t00_01.nc", woa_dir / "woa23_decav91C0_s00_01.nc")
    tm = grid.tmask
    # GODAS carries values below the surface freezing point under the ice
    # shelves' edges (-3.0 C seen at 72.5S 94.5W); the model's frazil works
    # at -1.8 C, so the start is clamped at -1.9 C (stated in the provenance).
    theta = np.where(tm, np.maximum(state["theta"], -1.9), 0.0)
    salt = np.where(tm, state["salt"], 0.0)
    u = np.where(grid.umask, state["u"], 0.0)
    v = np.where(grid.vmask, state["v"], 0.0)
    if barotropic == "remove":
        # Remove the depth mean of the velocities.
        for vel, dz, H in ((u, grid.dzu.astype(np.float64), grid.Hu),
                           (v, grid.dzv.astype(np.float64), grid.Hv)):
            mean = (vel * dz).sum(axis=0) / np.where(H > 0, H, 1.0)
            vel -= np.where(dz > 0, mean[None], 0.0)
    area = np.broadcast_to(grid.area, (grid.ny, grid.nx))
    eta = np.where(grid.wet, state["eta"], 0.0)
    eta = np.where(grid.wet, eta - (eta * area * grid.wet).sum() / (area * grid.wet).sum(), 0.0)
    targets = targets_from_woa(grid, woa_dir)
    prov = {"valid_utc": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "godas": [{"file": p0.name, "sha256": sha256_of(p0), "weight": 1.0 - w1},
                      {"file": p1.name, "sha256": sha256_of(p1), "weight": w1}],
            "woa23": {p.name: sha256_of(p) for p in sorted(woa_dir.glob("woa23_*.nc"))},
            "grid_sha256": grid.sha256, "theta_floor_c": -1.9,
            "velocity": ("GODAS full velocity" if barotropic == "godas" else
                         "GODAS baroclinic part (depth mean removed)"),
            "eta": "GODAS SSHG less its wet-area mean"}
    arrays = {"schema": SCHEMA, "theta": theta, "salt": salt, "u": u.astype(np.float32),
              "v": v.astype(np.float32), "eta": eta, "valid_utc": prov["valid_utc"],
              "provenance_json": json.dumps(prov, sort_keys=True), **targets}
    digest = write_npz(Path(out_path), arrays)
    return {"sha256": digest, **prov}


def init_from_state(grid: OceanGrid, *, base_init, state_path, out_path) -> dict:
    """An initial file from an ocean-only run's state (driver.save_state):
    temperature, salinity, velocity (barotropic flow included: it is the
    model's own) and free surface from the state, the targets from
    ``base_init``.  The valid time is the state's."""
    base = OceanInit(base_init)
    state_path = Path(state_path)
    with np.load(state_path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        theta, salt = d["theta"], d["salt"]
        u, v, eta = d["u"], d["v"], d["eta"]
    valid = meta["valid"]
    prov = {"valid_utc": valid if valid.endswith("Z") else valid + "Z",
            "from_state": {"file": state_path.name, "sha256": sha256_of(state_path),
                           "step": meta.get("step")},
            "base_init": {"file": Path(base_init).name, "sha256": base.sha256,
                          "provenance": base.provenance},
            "grid_sha256": grid.sha256,
            "velocity": "the ocean-only run's own (barotropic flow included)",
            "eta": "the ocean-only run's own"}
    arrays = {"schema": SCHEMA, "theta": np.asarray(theta, np.float64),
              "salt": np.asarray(salt, np.float64), "u": np.asarray(u, np.float32),
              "v": np.asarray(v, np.float32), "eta": np.asarray(eta, np.float64),
              "valid_utc": prov["valid_utc"], "provenance_json": json.dumps(prov, sort_keys=True)}
    for name in ("sss_month", "sponge_t", "sponge_s", "sponge_rows"):
        arrays[name] = base.arrays[name]
    digest = write_npz(Path(out_path), arrays)
    return {"sha256": digest, **prov}


class OceanInit:
    def __init__(self, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as d:
            if str(d["schema"]) != SCHEMA:
                raise ValueError(f"{path} is not a dynamic-ocean initial file ({SCHEMA})")
            self.arrays = {k: d[k] for k in d.files}
        self.path = path
        self.sha256 = sha256_of(path)
        self.valid_utc = str(self.arrays["valid_utc"])
        self.provenance = json.loads(str(self.arrays["provenance_json"]))


class Targets:
    """The SSS-restoring and sponge targets, linear between mid-months."""

    def __init__(self, init: OceanInit, start_utc: datetime, xp):
        self.start = start_utc.replace(tzinfo=None)
        self.xp = xp
        self.sss = init.arrays["sss_month"]
        self.t = init.arrays["sponge_t"]
        self.s = init.arrays["sponge_s"]
        self._cache = {}

    def _bracket(self, time_s):
        when = self.start + timedelta(seconds=float(time_s))
        mid = datetime(when.year, when.month, 15)
        if when < mid:
            m1 = (when.month - 2) % 12
            y = when.year if when.month > 1 else when.year - 1
            a = datetime(y, m1 + 1, 15)
            b = mid
            ia, ib = m1, when.month - 1
        else:
            ia = when.month - 1
            ib = when.month % 12
            y = when.year if when.month < 12 else when.year + 1
            a = mid
            b = datetime(y, ib + 1, 15)
        w = (when - a).total_seconds() / (b - a).total_seconds()
        return ia, ib, w

    def _get(self, arr, time_s, key):
        ia, ib, w = self._bracket(time_s)
        # one plane pair per day is plenty: cache by (key, ia, ib, rounded w)
        tag = (key, ia, ib, round(w, 3))
        if tag not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != key}
            val = (1.0 - w) * arr[ia].astype(np.float64) + w * arr[ib].astype(np.float64)
            self._cache[tag] = self.xp.asarray(val)
        return self._cache[tag]

    def sss_target(self, time_s):
        return self._get(self.sss, time_s, "sss")

    def sponge_target(self, time_s):
        return self._get(self.t, time_s, "t"), self._get(self.s, time_s, "s")

    def as_model_targets(self):
        return {"sss": self.sss_target, "sponge": self.sponge_target}


def godas_forcing(paths_weights, grid: OceanGrid):
    """Surface forcing on the ocean grid from (path, weight) pentads: q_ns =
    THFLX (all heat at the surface: GODAS carries no shortwave split), taux,
    tauy = UFLX, VFLX (stress on the ocean, N m-2, positive east/north),
    salt_flux = SLTFL (taken as psu m s-1 into the ocean)."""
    from .grib1 import godas_fields

    acc = None
    for path, w in paths_weights:
        f = godas_fields(path)
        part = {k: f[k] for k in ("THFLX", "UFLX", "VFLX", "SLTFL")}
        part["SST"] = f["POT"][0]
        if acc is None:
            acc = {k: w * v for k, v in part.items()}
            lat, lon = f["lats"], f["lons"]
        else:
            for k in acc:
                acc[k] = acc[k] + w * part[k]
    out = {}
    for k, name in (("THFLX", "q_ns"), ("UFLX", "taux"), ("VFLX", "tauy"), ("SLTFL", "salt_flux"),
                    ("SST", "godas_sst_c")):
        out[name] = np.where(grid.wet, _to_grid_2d(acc[k], lat, lon, grid.lat_t, grid.lon_t), 0.0)
    return out


__all__ = ["OceanInit", "SCHEMA", "Targets", "bracket", "build_init", "creep_fill",
           "godas_forcing", "godas_state", "pentad_center", "targets_from_woa"]
