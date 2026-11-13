"""Airliner heights: the broadcast GNSS-minus-pressure altitude as an
upper-air height observation, read from one day of an open ADS-B archive.

An aircraft's transponder broadcasts two altitudes.  The barometric one is
pressure altitude: the static pressure the aircraft measures, expressed as
the height of that pressure in the ICAO standard atmosphere (always on the
1013.25 hPa datum, whatever the cockpit altimeter is set to).  The
geometric one is the GNSS height, above the WGS84 ellipsoid for ADS-B
version 1 and later.  So every report pairs a pressure with a geometric
height, and the geometric height converted to geopotential height is the
height of that pressure surface at the aircraft: the quantity an analysis
calls Z at p.  This tool turns the pair into that observation and checks it
against reference fields and radiosondes.

The observation operator (``height_observation``):

* pressure from pressure altitude by the exact ICAO standard-atmosphere
  inversion (``isa_pressure_pa``; the ICAO layers to 32 km);
* ellipsoidal height minus the EGM96 geoid undulation (``Geoid``, a
  15-arc-minute grid read bilinearly) is the height above mean sea level;
* that height to geopotential height through WGS84 normal gravity and the
  latitude's effective radius (``geopotential_height_m``):
  Z = (g(phi) / g0) * R(phi) * H / (R(phi) + H), g0 = 9.80665 m s-2.

The model side (``ReferenceSampler``): a GRIB2 product's geopotential
height and temperature on isobaric levels, bilinear in the horizontal and
hypsometric in the vertical (temperature linear in ln p between the two
levels that bracket the report, with the residual against the upper
level's own height spread linearly in ln p so both levels are honoured
exactly), and linear in time between hourly products.  The GRIB2 reader
here (``read_grib2``) is plain numpy and decodes the templates the GFS
0.25 degree products use: grid 3.0, product 4.0, data 5.0, 5.2 and 5.3.

Subcommands:

``extract``  stream the day's split tar (``*.tar.aa``, ``*.tar.ab`` given in
            order) and write one row per kept trace point to an ``.npz``,
            with a JSON record of every rejection counted by cause.  Only the
            numbers are kept: the 24-bit address as an integer, the ICAO type
            designator, time, position, both altitudes, vertical rate,
            source, ADS-B version, GVA, NIC-baro, NACp and emitter category.
            No registration, owner or flight number is read into a row.
``geoid``    convert the EGM96 15' GeoTIFF the PROJ data CDN publishes
            (``us_nga_egm96_15.tif``) to the ``.npz`` the operator reads
            (needs ``tifffile`` and ``imagecodecs``; nothing else does).
``score``    the operator on every kept row, sampled against an hourly
            reference set and a longer-lead forecast set, with the
            statistics this idea's attackers asked for (out-of-sample
            airframe bias, airframe-mean stability, super-observations,
            the residual correlation between different airframes against
            separation, the information regression, the geoid regression
            per airframe, the radiosonde collocation) written to JSON, and
            the coverage and residual maps written to PNG (``matplotlib``).

The archive this was built for is adsb.lol's ``globe_history_2026`` (one
GitHub release per day, ODbL 1.0 and CC0, no account).  A derived database
that is published carries the ODbL's share-alike terms; this tool writes
statistics and pictures, and its row files are working files.

Nothing here imports the model: it runs on a CPU host with numpy, and the
radiosonde reader is ``tools/arwen_global_igra2_levels_csv.parse_igra2``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import struct
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ constants

FT_M = 0.3048
G0 = 9.80665                 # standard gravity; GRIB HGT is geopotential / G0
RD = 287.05                  # dry-air gas constant the GFS products use
ISA_R = 287.05287            # ICAO's air gas constant
ISA_P0 = 101325.0
ISA_T0 = 288.15
# ICAO standard-atmosphere layers to 32 km: (base geopotential height m,
# base temperature K, lapse rate K/m (temperature change per metre up),
# base pressure Pa).
ISA_LAYERS = (
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.0559),
    (20000.0, 216.65, 0.001, 5474.88867),
    (32000.0, 228.65, 0.0028, 868.018685),
)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_M = 0.00344978650684   # omega^2 a^2 b / GM
SOMIGLIANA_GE = 9.7803253359
SOMIGLIANA_K = 0.00193185265241
WGS84_E2 = 0.00669437999013

SOURCES = ("adsb_icao", "adsb_icao_nt", "adsr_icao", "tisb_icao", "adsc",
           "mlat", "other", "mode_s", "adsb_other", "adsr_other",
           "tisb_other", "tisb_trackfile", "unknown")
SOURCE_CODE = {name: i for i, name in enumerate(SOURCES)}


# ------------------------------------------------------ the observation side

def isa_pressure_pa(pressure_altitude_m):
    """Pressure (Pa) of an ICAO standard-atmosphere pressure altitude
    (geopotential metres), exact by layer."""
    h = np.asarray(pressure_altitude_m, dtype=np.float64)
    out = np.full(h.shape, np.nan)
    for i, (hb, tb, lapse, pb) in enumerate(ISA_LAYERS):
        top = ISA_LAYERS[i + 1][0] if i + 1 < len(ISA_LAYERS) else 47000.0
        sel = (h >= hb) & (h < top) if i else (h < top)
        dh = h[sel] - hb
        if lapse == 0.0:
            out[sel] = pb * np.exp(-G0 * dh / (ISA_R * tb))
        else:
            out[sel] = pb * (1.0 + lapse * dh / tb) ** (-G0 / (ISA_R * lapse))
    return out


def isa_pressure_altitude_m(pressure_pa):
    """The inverse of ``isa_pressure_pa``."""
    p = np.asarray(pressure_pa, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    for i, (hb, tb, lapse, pb) in enumerate(ISA_LAYERS):
        p_top = ISA_LAYERS[i + 1][3] if i + 1 < len(ISA_LAYERS) else 110.906
        sel = (p <= pb) & (p > p_top) if i else (p > p_top)
        if lapse == 0.0:
            out[sel] = hb - ISA_R * tb / G0 * np.log(p[sel] / pb)
        else:
            out[sel] = hb + tb / lapse * ((p[sel] / pb) ** (-ISA_R * lapse / G0) - 1.0)
    return out


def normal_gravity(lat_deg):
    """WGS84 normal gravity on the ellipsoid (Somigliana), m s-2."""
    s2 = np.sin(np.radians(np.asarray(lat_deg, dtype=np.float64))) ** 2
    return SOMIGLIANA_GE * (1.0 + SOMIGLIANA_K * s2) / np.sqrt(1.0 - WGS84_E2 * s2)


def effective_radius_m(lat_deg):
    """The radius that gives normal gravity's free-air gradient at a
    latitude: a / (1 + f + m - 2 f sin^2 phi)."""
    s2 = np.sin(np.radians(np.asarray(lat_deg, dtype=np.float64))) ** 2
    return WGS84_A / (1.0 + WGS84_F + WGS84_M - 2.0 * WGS84_F * s2)


def geopotential_height_m(height_msl_m, lat_deg):
    """Geopotential height (gpm, on G0) of a height above mean sea level."""
    h = np.asarray(height_msl_m, dtype=np.float64)
    r = effective_radius_m(lat_deg)
    return normal_gravity(lat_deg) / G0 * r * h / (r + h)


class Geoid:
    """A regular latitude-longitude geoid grid read bilinearly.

    The ``.npz`` carries ``lat`` (descending or ascending), ``lon`` (a
    uniform ring, any start) and ``n_m`` [lat, lon], the undulation of the
    geoid above the WGS84 ellipsoid in metres."""

    def __init__(self, lat, lon, n_m):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        n_m = np.asarray(n_m, dtype=np.float64)
        if lat[0] > lat[-1]:
            lat, n_m = lat[::-1], n_m[::-1]
        self.lat0, self.dlat = lat[0], lat[1] - lat[0]
        self.lon0, self.dlon = lon[0], lon[1] - lon[0]
        self.n_m = n_m
        self.nlat, self.nlon = n_m.shape

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(z["lat"], z["lon"], z["n_m"])

    def __call__(self, lat_deg, lon_deg):
        y = (np.asarray(lat_deg, dtype=np.float64) - self.lat0) / self.dlat
        x = np.mod(np.asarray(lon_deg, dtype=np.float64) - self.lon0, 360.0) / self.dlon
        i0 = np.clip(np.floor(y).astype(np.int64), 0, self.nlat - 2)
        fy = np.clip(y - i0, 0.0, 1.0)
        j0 = np.floor(x).astype(np.int64) % self.nlon
        fx = x - np.floor(x)
        j1 = (j0 + 1) % self.nlon
        g = self.n_m
        return ((1 - fy) * ((1 - fx) * g[i0, j0] + fx * g[i0, j1])
                + fy * ((1 - fx) * g[i0 + 1, j0] + fx * g[i0 + 1, j1]))


def height_observation(alt_baro_ft, alt_geom_ft, lat_deg, lon_deg, geoid):
    """(pressure Pa, geopotential height gpm) of each report: the pressure
    its barometric altitude names and the geopotential height of its GNSS
    height above the ellipsoid."""
    p = isa_pressure_pa(np.asarray(alt_baro_ft, dtype=np.float64) * FT_M)
    h_msl = np.asarray(alt_geom_ft, dtype=np.float64) * FT_M - geoid(lat_deg, lon_deg)
    return p, geopotential_height_m(h_msl, lat_deg)


# ------------------------------------------------------------ the GRIB2 side

def _u(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _s(b: bytes) -> int:
    """GRIB2 sign-magnitude integer."""
    value = int.from_bytes(b, "big")
    top = 1 << (8 * len(b) - 1)
    return -(value & (top - 1)) if value & top else value


def unpack_bits(buf: bytes, bit_offset: int, widths) -> np.ndarray:
    """Big-endian unsigned integers of the given bit widths (each at most
    56), packed back to back from ``bit_offset``."""
    widths = np.asarray(widths, dtype=np.int64)
    n = widths.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if widths.max(initial=0) > 56:
        raise ValueError("packed width above 56 bits")
    offs = bit_offset + np.concatenate(([0], np.cumsum(widths)[:-1]))
    arr = np.frombuffer(bytes(buf) + b"\0" * 8, dtype=np.uint8)
    byte = offs >> 3
    word = np.zeros(n, dtype=np.uint64)
    for k in range(8):
        word = (word << np.uint64(8)) | arr[byte + k].astype(np.uint64)
    shift = (64 - (offs & 7) - widths).astype(np.uint64)
    shift = np.minimum(shift, np.uint64(63))
    mask = (np.left_shift(np.uint64(1), widths.astype(np.uint64)) - np.uint64(1))
    out = (word >> shift) & mask
    out[widths == 0] = 0
    return out.astype(np.int64)


def _decode_data(s5: bytes, s7: bytes, npts: int) -> np.ndarray:
    template = _u(s5[9:11])
    ref = struct.unpack(">f", s5[11:15])[0]
    e = _s(s5[15:17])
    d = _s(s5[17:19])
    nbits = s5[19]
    data = s7[5:]
    if template == 0:
        if nbits == 0:
            x = np.zeros(npts, dtype=np.int64)
        else:
            x = unpack_bits(data, 0, np.full(npts, nbits))
    elif template in (2, 3):
        mvm = s5[22]
        if mvm != 0:
            raise ValueError(f"data template 5.{template} with missing-value management {mvm}")
        ng = _u(s5[31:35])
        ref_w, bits_w = s5[35], s5[36]
        ref_l, inc_l, last_l, bits_l = _u(s5[37:41]), s5[41], _u(s5[42:46]), s5[46]
        pos = 0
        order = 0
        if template == 3:
            order, nbytes = s5[47], s5[48]
            ival1 = _s(data[0:nbytes])
            ival2 = _s(data[nbytes:2 * nbytes]) if order == 2 else 0
            minsd = _s(data[order * nbytes:(order + 1) * nbytes])
            pos = (order + 1) * nbytes
        gref = unpack_bits(data, pos * 8, np.full(ng, nbits)) if nbits else np.zeros(ng, np.int64)
        pos += (ng * nbits + 7) // 8
        gw = unpack_bits(data, pos * 8, np.full(ng, bits_w)) + ref_w if bits_w else np.full(ng, ref_w, np.int64)
        pos += (ng * bits_w + 7) // 8
        gl = unpack_bits(data, pos * 8, np.full(ng, bits_l)) if bits_l else np.zeros(ng, np.int64)
        pos += (ng * bits_l + 7) // 8
        lengths = ref_l + gl * inc_l
        lengths[-1] = last_l
        if int(lengths.sum()) != npts:
            raise ValueError(f"group lengths sum to {int(lengths.sum())}, expected {npts}")
        x = np.repeat(gref, lengths) + unpack_bits(data, pos * 8, np.repeat(gw, lengths))
        if order == 1:
            x[1:] += minsd
            x[0] = ival1
            x = np.cumsum(x)
        elif order == 2:
            diff = np.empty(npts, dtype=np.int64)
            diff[0] = ival1
            if npts > 1:
                diff[1] = ival2 - ival1
            if npts > 2:
                diff[2:] = x[2:] + minsd
            diff[1:] = np.cumsum(diff[1:])
            x = np.cumsum(diff)
    else:
        raise ValueError(f"data representation template 5.{template} is not read here")
    return (ref + x.astype(np.float64) * 2.0 ** e) / 10.0 ** d


def read_grib2(path, want=None):
    """Messages of a GRIB2 file as dicts: ``discipline``, ``category``,
    ``number``, ``surface``, ``level`` (scaled value of the first fixed
    surface), ``reference`` and ``valid`` (datetimes), ``lat`` and ``lon``
    (1-D) and ``values`` [lat, lon] float32 on the file's own scan order.
    ``want(discipline, category, number, surface, level)`` filters before
    the data are decoded."""
    blob = Path(path).read_bytes()
    pos = 0
    out = []
    while True:
        start = blob.find(b"GRIB", pos)
        if start < 0:
            break
        if blob[start + 7] != 2:
            raise ValueError(f"{path}: GRIB edition {blob[start + 7]} at byte {start}")
        total = _u(blob[start + 8:start + 16])
        msg = blob[start:start + total]
        discipline = msg[6]
        p = 16
        sec = {}
        while p < total - 4:
            length, number = _u(msg[p:p + 4]), msg[p + 4]
            sec[number] = msg[p:p + length]
            p += length
            if number == 7:
                s1, s3, s4, s5, s6, s7 = (sec[k] for k in (1, 3, 4, 5, 6, 7))
                if _u(s4[7:9]) != 0:
                    raise ValueError(f"product template 4.{_u(s4[7:9])} is not read here")
                cat, num = s4[9], s4[10]
                surface = s4[22]
                level = _u(s4[24:28]) * 10.0 ** (-_s(s4[23:24]))
                if want is None or want(discipline, cat, num, surface, level):
                    if _u(s3[12:14]) != 0:
                        raise ValueError(f"grid template 3.{_u(s3[12:14])} is not read here")
                    if s6[5] != 255:
                        raise ValueError("a bitmap is not read here")
                    ni, nj = _u(s3[30:34]), _u(s3[34:38])
                    la1, lo1 = _s(s3[46:50]) * 1e-6, _s(s3[50:54]) * 1e-6
                    la2, lo2 = _s(s3[55:59]) * 1e-6, _s(s3[59:63]) * 1e-6
                    di, dj = _u(s3[63:67]) * 1e-6, _u(s3[67:71]) * 1e-6
                    scan = s3[71]
                    if scan & 0b11100000 not in (0, 0b01000000):
                        raise ValueError(f"scanning mode {scan:08b} is not read here")
                    ref = dt.datetime(_u(s1[12:14]), s1[14], s1[15], s1[16], s1[17], s1[18],
                                      tzinfo=dt.timezone.utc)
                    unit = s4[17]
                    step = {0: 60, 1: 3600, 2: 86400, 10: 3 * 3600, 11: 6 * 3600,
                            12: 12 * 3600, 13: 1}[unit]
                    lead = _u(s4[18:22]) * step
                    values = _decode_data(s5, s7, ni * nj).astype(np.float32).reshape(nj, ni)
                    lat = la1 + np.sign(la2 - la1) * dj * np.arange(nj)
                    lon = lo1 + di * np.arange(ni)
                    out.append({"discipline": discipline, "category": cat, "number": num,
                                "surface": surface, "level": level, "reference": ref,
                                "valid": ref + dt.timedelta(seconds=lead),
                                "lat": lat, "lon": lon, "values": values})
                sec = {k: v for k, v in sec.items() if k in (1, 2, 3)}
        pos = start + total
    return out


def _is_height_or_temperature(discipline, cat, num, surface, level):
    return discipline == 0 and surface == 100 and (cat, num) in ((3, 5), (0, 0))


class ReferenceSampler:
    """Geopotential height at a report's pressure, from one GRIB2 product's
    isobaric HGT and TMP."""

    def __init__(self, messages):
        hgt = {m["level"]: m for m in messages if (m["category"], m["number"]) == (3, 5)}
        tmp = {m["level"]: m for m in messages if (m["category"], m["number"]) == (0, 0)}
        levels = sorted(set(hgt) & set(tmp), reverse=True)    # high pressure first
        if len(levels) < 2:
            raise ValueError("fewer than two isobaric levels carry both HGT and TMP")
        first = hgt[levels[0]]
        self.valid = first["valid"]
        self.levels_pa = np.array(levels, dtype=np.float64)
        self.z = np.stack([hgt[p]["values"] for p in levels])
        self.t = np.stack([tmp[p]["values"] for p in levels])
        lat, lon = first["lat"], first["lon"]
        self.lat0, self.dlat = float(lat[0]), float(lat[1] - lat[0])
        self.lon0, self.dlon = float(lon[0]), float(lon[1] - lon[0])
        self.nlat, self.nlon = self.z.shape[1:]
        self.ring = abs(self.dlon * self.nlon - 360.0) < 1e-6

    @classmethod
    def from_path(cls, path, cache_dir=None):
        """The sampler of one GRIB2 file, through an uncompressed ``.npz``
        cache keyed by the file's name and size when ``cache_dir`` is set."""
        path = Path(path)
        if cache_dir is None:
            return cls(read_grib2(path, _is_height_or_temperature))
        cache = Path(cache_dir) / f"{path.name}.{path.stat().st_size}.npz"
        if cache.exists():
            self = cls.__new__(cls)
            with np.load(cache) as z:
                self.z, self.t, self.levels_pa = z["z"], z["t"], z["levels_pa"]
                (self.lat0, self.dlat, self.lon0, self.dlon) = (float(v) for v in z["grid"])
                self.valid = dt.datetime.fromtimestamp(float(z["valid"]), dt.timezone.utc)
            self.nlat, self.nlon = self.z.shape[1:]
            self.ring = abs(self.dlon * self.nlon - 360.0) < 1e-6
            return self
        self = cls(read_grib2(path, _is_height_or_temperature))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, z=self.z, t=self.t, levels_pa=self.levels_pa,
                 grid=np.array([self.lat0, self.dlat, self.lon0, self.dlon]),
                 valid=np.array(self.valid.timestamp()))
        return self

    def _corners(self, lat, lon):
        y = (np.asarray(lat, dtype=np.float64) - self.lat0) / self.dlat
        x = np.mod(np.asarray(lon, dtype=np.float64) - self.lon0, 360.0) / self.dlon
        i0 = np.clip(np.floor(y).astype(np.int64), 0, self.nlat - 2)
        fy = np.clip(y - i0, 0.0, 1.0)
        j0 = np.floor(x).astype(np.int64) % self.nlon
        fx = x - np.floor(x)
        j1 = (j0 + 1) % self.nlon
        return i0, j0, j1, fx, fy

    def horizontal(self, field, k, lat, lon):
        i0, j0, j1, fx, fy = self._corners(lat, lon)
        f = field
        return ((1 - fy) * ((1 - fx) * f[k, i0, j0] + fx * f[k, i0, j1])
                + fy * ((1 - fx) * f[k, i0 + 1, j0] + fx * f[k, i0 + 1, j1]))

    def height_at(self, p_pa, lat, lon):
        """Geopotential height (gpm) at pressure ``p_pa``; NaN outside the
        level range."""
        p = np.asarray(p_pa, dtype=np.float64)
        levels = self.levels_pa
        # k: the bracketing level of higher pressure (levels descend).
        k = np.searchsorted(-levels, -p, side="right") - 1
        inside = (k >= 0) & (k < len(levels) - 1) & np.isfinite(p)
        k = np.clip(k, 0, len(levels) - 2)
        z1 = self.horizontal(self.z, k, lat, lon)
        z2 = self.horizontal(self.z, k + 1, lat, lon)
        t1 = self.horizontal(self.t, k, lat, lon)
        t2 = self.horizontal(self.t, k + 1, lat, lon)
        return np.where(inside, hypsometric_between(p, levels[k], levels[k + 1], z1, z2, t1, t2), np.nan)


def hypsometric_between(p, p1, p2, z1, z2, t1, t2):
    """Height at ``p`` between two levels (p1 > p >= p2): temperature linear
    in ln p, and the hypsometric residual against z2 spread linearly in ln p,
    so the result is z1 at p1 and z2 at p2 exactly."""
    x = np.log(p1 / p)
    span = np.log(p1 / p2)
    coef = RD / G0
    zh = z1 + coef * (t1 * x + (t2 - t1) * x * x / (2.0 * span))
    zh_top = z1 + coef * span * 0.5 * (t1 + t2)
    return zh + (z2 - zh_top) * x / span


# ------------------------------------------------------------------ extract

class _Concat(io.RawIOBase):
    """The split parts of one tar read as one stream."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.index = 0
        self.handle = open(self.paths[0], "rb")

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            n = self.handle.readinto(buffer)
            if n:
                return n
            self.handle.close()
            self.index += 1
            if self.index >= len(self.paths):
                return 0
            self.handle = open(self.paths[self.index], "rb")


def _loads():
    try:
        import orjson
        return orjson.loads
    except ImportError:
        return json.loads


CATEGORY = {f"{a}{n}": 16 * i + n for i, a in enumerate("ABCD") for n in range(8)}


def trace_rows(doc, day_start, day_end, *, floor_ft, spacing_s, counts):
    """Kept rows of one trace document, with every point that is not kept
    counted under its cause in ``counts``."""
    icao = doc.get("icao", "")
    try:
        address = int(icao.lstrip("~"), 16)
    except ValueError:
        counts["trace_bad_address"] += 1
        return []
    if icao.startswith("~"):
        counts["trace_non_icao_address"] += 1
        return []
    base = float(doc["timestamp"])
    rows = []
    version = gva = nic_baro = nac_p = -1
    category = 255
    last_kept = -1e18
    for point in doc.get("trace", ()):
        counts["points"] += 1
        obj = point[8] if len(point) > 8 else None
        if isinstance(obj, dict):
            version = obj.get("version", version) if obj.get("version") is not None else version
            gva = obj.get("gva", gva) if obj.get("gva") is not None else gva
            nic_baro = obj.get("nic_baro", nic_baro) if obj.get("nic_baro") is not None else nic_baro
            nac_p = obj.get("nac_p", nac_p) if obj.get("nac_p") is not None else nac_p
            if obj.get("category") in CATEGORY:
                category = CATEGORY[obj["category"]]
        t = base + float(point[0])
        if not day_start <= t < day_end:
            counts["outside_day"] += 1
            continue
        baro = point[3]
        if baro == "ground":
            counts["on_ground"] += 1
            continue
        if not isinstance(baro, (int, float)):
            counts["baro_missing"] += 1
            continue
        flags = point[6] or 0
        if flags & 8:
            counts["altitude_is_geometric"] += 1
            continue
        if flags & 1:
            counts["position_stale"] += 1
            continue
        geom = point[10] if len(point) > 10 else None
        if not isinstance(geom, (int, float)):
            counts["geom_missing"] += 1
            continue
        source = point[9] if len(point) > 9 and point[9] else "unknown"
        if source != "adsb_icao":
            counts[f"source_{source}"] += 1
            continue
        if baro < floor_ft:
            counts["below_floor"] += 1
            continue
        if t - last_kept < spacing_s:
            counts["thinned"] += 1
            continue
        last_kept = t
        vrate = point[7] if isinstance(point[7], (int, float)) else float("nan")
        rows.append((address, t, point[1], point[2], baro, geom, vrate,
                     SOURCE_CODE.get(source, SOURCE_CODE["unknown"]),
                     version, gva, nic_baro, nac_p, category))
        counts["kept"] += 1
    return rows


ROW_FIELDS = (("address", np.uint32), ("t", np.float64), ("lat", np.float32),
              ("lon", np.float32), ("alt_baro_ft", np.int32), ("alt_geom_ft", np.int32),
              ("vrate_fpm", np.float32), ("source", np.uint8), ("version", np.int8),
              ("gva", np.int8), ("nic_baro", np.int8), ("nac_p", np.int8),
              ("category", np.uint8))


def extract(parts, day, out, *, floor_ft=5000.0, spacing_s=60.0, limit=None, log=sys.stderr):
    day_start = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp()
    day_end = day_start + 86400.0
    loads = _loads()
    counts = Counter()
    rows = []
    types = {}
    began = time.time()
    with tarfile.open(fileobj=io.BufferedReader(_Concat(parts), 1 << 20), mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if "/traces/" not in "/" + name.lstrip("./") and not name.lstrip("./").startswith("traces/"):
                counts["members_not_traces"] += 1
                continue
            counts["trace_files"] += 1
            raw = tar.extractfile(member).read()
            try:
                doc = loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            except Exception:
                counts["trace_unreadable"] += 1
                continue
            new = trace_rows(doc, day_start, day_end, floor_ft=floor_ft,
                             spacing_s=spacing_s, counts=counts)
            if new:
                rows.extend(new)
                types[new[0][0]] = str(doc.get("t") or "")
            if counts["trace_files"] % 5000 == 0:
                print(f"extract: {counts['trace_files']} traces, {counts['kept']} rows, "
                      f"{time.time() - began:.0f} s", file=log, flush=True)
            if limit and counts["trace_files"] >= limit:
                break
    arrays = {}
    for i, (name, dtype) in enumerate(ROW_FIELDS):
        arrays[name] = np.array([r[i] for r in rows], dtype=dtype)
    addr = np.array(sorted(types), dtype=np.uint32)
    arrays["type_address"] = addr
    arrays["type_designator"] = np.array([types[int(a)] for a in addr], dtype="U8")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    record = {"tool": "arwen_global_airliner_heights extract", "day": day,
              "parts": [str(p) for p in parts], "floor_ft": floor_ft,
              "spacing_s": spacing_s, "counts": dict(sorted(counts.items())),
              "aircraft_kept": int(np.unique(arrays["address"]).size),
              "seconds": round(time.time() - began, 1),
              "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


# -------------------------------------------------------------------- score

def sample_series(grib_paths, t, p, lat, lon, *, cache_dir=None, log=sys.stderr):
    """Height at each report's pressure and place, linear in time between
    the products' valid times (hourly products assumed); NaN where a report
    is not bracketed by two products or lies outside the level range.
    Also returns the products' valid times."""
    acc = np.zeros(t.size)
    wsum = np.zeros(t.size)
    valids = []
    for path in grib_paths:
        sampler = ReferenceSampler.from_path(path, cache_dir)
        v = sampler.valid.timestamp()
        valids.append(v)
        if len(valids) > 1 and v <= valids[-2]:
            raise ValueError("products must be given in increasing valid time")
        near = np.nonzero(np.abs(t - v) < 3600.0)[0]
        if near.size:
            w = 1.0 - np.abs(t[near] - v) / 3600.0
            z = sampler.height_at(p[near], lat[near], lon[near])
            ok = np.isfinite(z)
            acc[near[ok]] += w[ok] * z[ok]
            wsum[near[ok]] += w[ok]
        print(f"sample: {Path(path).name} valid {sampler.valid:%Y-%m-%d %H}Z, "
              f"{near.size} reports", file=log, flush=True)
    z = np.where(np.abs(wsum - 1.0) < 1e-6, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    return z, valids


def _robust_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [1, 5, 50, 95, 99])
    return {"n": int(x.size), "mean": round(float(x.mean()), 2), "std": round(float(x.std()), 2),
            "robust_std": round(_robust_std(x), 2), "p01": round(float(q[0]), 1),
            "p05": round(float(q[1]), 1), "median": round(float(q[2]), 2),
            "p95": round(float(q[3]), 1), "p99": round(float(q[4]), 1)}


def group_mean(keys, values, *, min_n=1):
    """Mean of ``values`` per key: (unique keys, means, counts)."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    v = values[order]
    uniq, start, count = np.unique(k, return_index=True, return_counts=True)
    sums = np.add.reduceat(v, start) if v.size else np.zeros(0)
    keep = count >= min_n
    return uniq[keep], (sums / count)[keep], count[keep]


def lookup(keys, table_keys, table_values, default=np.nan):
    """``table_values`` at each of ``keys`` (sorted ``table_keys``)."""
    idx = np.searchsorted(table_keys, keys)
    idx = np.clip(idx, 0, max(len(table_keys) - 1, 0))
    hit = (table_keys[idx] == keys) if len(table_keys) else np.zeros(keys.shape, bool)
    out = np.full(keys.shape, default, dtype=np.float64)
    out[hit] = table_values[idx[hit]]
    return out


def regression(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) == 0:
        return {"n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    return {"n": int(x.size), "slope": round(float(slope), 3),
            "intercept": round(float(intercept), 2),
            "correlation": round(float(np.corrcoef(x, y)[0, 1]), 3),
            "std_x": round(float(np.std(x)), 2), "std_y": round(float(np.std(y)), 2)}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def superobs(cell_keys, aircraft, residual, *, min_aircraft):
    """Super-observations: the mean over aircraft of each aircraft's mean
    residual in a cell.  Returns (cell keys, super-ob residual, aircraft
    count)."""
    pair = cell_keys.astype(np.int64) * (1 << 25) + aircraft.astype(np.int64)
    upair, pmean, _ = group_mean(pair, residual)
    cells = upair >> 25
    ucell, cmean, ccount = group_mean(cells, pmean)
    keep = ccount >= min_aircraft
    return ucell[keep], cmean[keep], ccount[keep]


def residual_correlation(t, lat, lon, aircraft, r, bins_km, *, per_hour=1500, seed=0):
    """Correlation of the residuals of DIFFERENT aircraft against their
    separation: per hour, each aircraft's hour-mean residual and position,
    then every pair of up to ``per_hour`` aircraft."""
    rng = np.random.default_rng(seed)
    hour = np.floor(t / 3600.0).astype(np.int64)
    key = hour * (1 << 25) + aircraft.astype(np.int64)
    ukey, rm, _ = group_mean(key, r)
    _, latm, _ = group_mean(key, lat.astype(np.float64))
    # longitude mean through the unit circle
    _, cx, _ = group_mean(key, np.cos(np.radians(lon)))
    _, sx, _ = group_mean(key, np.sin(np.radians(lon)))
    lonm = np.degrees(np.arctan2(sx, cx))
    hours = ukey >> 25
    var = np.var(rm)
    mean = np.mean(rm)
    sums = np.zeros(len(bins_km) - 1)
    counts = np.zeros(len(bins_km) - 1)
    for h in np.unique(hours):
        sel = np.nonzero(hours == h)[0]
        if sel.size > per_hour:
            sel = rng.choice(sel, per_hour, replace=False)
        a = rm[sel] - mean
        d = haversine_km(latm[sel][:, None], lonm[sel][:, None], latm[sel][None, :], lonm[sel][None, :])
        iu = np.triu_indices(sel.size, 1)
        dd = d[iu]
        prod = (a[:, None] * a[None, :])[iu]
        which = np.digitize(dd, bins_km) - 1
        ok = (which >= 0) & (which < len(bins_km) - 1)
        np.add.at(sums, which[ok], prod[ok])
        np.add.at(counts, which[ok], 1)
    rows = []
    for i in range(len(bins_km) - 1):
        rows.append({"from_km": bins_km[i], "to_km": bins_km[i + 1], "pairs": int(counts[i]),
                     "correlation": (round(float(sums[i] / counts[i] / var), 3)
                                     if counts[i] else None)})
    return {"aircraft_hours": int(rm.size), "variance_m2": round(float(var), 2), "bins": rows}


def layer_pairs(aircraft, t, p, *, min_ln=0.18, max_ln=0.30, max_dt=1200.0, max_lag=30):
    """Pairs of reports of one aircraft across a layer, as index arrays
    (i, j) into the inputs: j is the first later report, within ``max_dt``
    seconds, whose ln-pressure differs from i's by at least ``min_ln``; the
    pair is kept when that difference is at most ``max_ln`` and the pressure
    moved one way only between them (one climb or one descent)."""
    order = np.lexsort((t, aircraft))
    a, tt, lp = aircraft[order], t[order], np.log(p[order])
    n = a.size
    best = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    rising = np.ones(n, dtype=bool)      # pressure rising all the way (descent)
    falling = np.ones(n, dtype=bool)     # pressure falling all the way (climb)
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        i = np.arange(n - k)
        j = i + k
        step = lp[j] - lp[j - 1]
        rising[: n - k] &= step > 0
        falling[: n - k] &= step < 0
        alive = ((a[j] == a[i]) & (tt[j] - tt[i] <= max_dt)
                 & (rising[: n - k] | falling[: n - k]) & ~done[: n - k])
        d = np.abs(lp[j] - lp[i])
        reach = alive & (d >= min_ln)
        good = reach & (d <= max_ln)
        best[i[good]] = j[good]
        done[i[reach]] = True
        if not alive.any():
            break
    keep = best >= 0
    return order[np.nonzero(keep)[0]], order[best[keep]]


def layer_temperature_k(p_i, p_j, dz):
    """Layer-mean virtual temperature (K) of a thickness ``dz`` between
    pressures ``p_i`` and ``p_j``: g0 dz / (Rd ln(p_i / p_j))."""
    return G0 * dz / (RD * np.log(p_i / p_j))


def _land_mask(grib_path):
    for m in read_grib2(grib_path, lambda d, c, n, s, lv: (d, c, n, s) == (2, 0, 0, 1)):
        return m
    return None


def collocate_sondes(sonde_dirs, nominal, rows, *, radius_km, hours, dp_hpa, levels_hpa,
                     an_paths_by_valid, cache_dir=None, log=sys.stderr):
    """Radiosonde collocation at mandatory levels.  For each sounding and
    level: the reports within ``radius_km``, ``hours`` and ``dp_hpa`` of it,
    each aircraft's mean residual against the reference (x), and the
    sonde's own residual against the same reference at its level (y).  The
    aircraft-minus-sonde difference through the reference is x - y."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from arwen_global_igra2_levels_csv import parse_igra2, read_source

    soundings = []
    for directory in sonde_dirs:
        for path in sorted(Path(directory).glob("*-data-*.txt.zip")):
            soundings.extend(parse_igra2(read_source(path), set(nominal)))
    out = []
    for stamp in sorted(set(nominal)):
        valid = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        path = an_paths_by_valid.get(valid)
        if path is None:
            continue
        sampler = ReferenceSampler.from_path(path, cache_dir)
        here = [s for s in soundings if s["nominal"] == stamp]
        tv = valid.timestamp()
        near_t = np.abs(rows["t"] - tv) <= hours * 3600.0
        for lev in levels_hpa:
            near_p = np.abs(rows["p"] / 100.0 - lev) <= dp_hpa
            cand = np.nonzero(near_t & near_p & np.isfinite(rows["r"]))[0]
            if cand.size == 0:
                continue
            for s in here:
                level = s["levels"].get(int(round(lev * 100)))
                if not level or level["gph_m"] is None:
                    continue
                d = haversine_km(s["latitude"], s["longitude"], rows["lat"][cand], rows["lon"][cand])
                hit = cand[d <= radius_km]
                if hit.size == 0:
                    continue
                za = sampler.height_at(np.array([lev * 100.0]), np.array([s["latitude"]]),
                                       np.array([s["longitude"]]))[0]
                _, am, _ = group_mean(rows["aircraft"][hit], rows["r"][hit])
                corrected = hit[np.isfinite(rows["rc"][hit])]
                xc = (float(np.mean(group_mean(rows["aircraft"][corrected], rows["rc"][corrected])[1]))
                      if corrected.size else None)
                out.append({"station": s["station"], "nominal": stamp, "level_hpa": lev,
                            "lat": s["latitude"], "lon": s["longitude"],
                            "aircraft": int(am.size), "reports": int(hit.size),
                            "x_aircraft_minus_ref": float(np.mean(am)),
                            "x_corrected_aircraft_minus_ref": xc,
                            "y_sonde_minus_ref": float(level["gph_m"] - za)})
        print(f"sondes: {stamp} {len(here)} soundings, {len(out)} collocations so far",
              file=log, flush=True)
    return out


def score(args, log=sys.stderr):
    began = time.time()
    with np.load(args.rows) as z:
        rows = {k: z[k] for k in z.files}
    geoid = Geoid.load(args.geoid)
    n_all = rows["t"].size
    p, zobs = height_observation(rows["alt_baro_ft"], rows["alt_geom_ft"],
                                 rows["lat"], rows["lon"], geoid)
    t = rows["t"]
    lat = rows["lat"].astype(np.float64)
    lon = rows["lon"].astype(np.float64)
    pmin, pmax = args.p_min_hpa * 100.0, args.p_max_hpa * 100.0
    keep = (p >= pmin) & (p <= pmax)
    idx = np.nonzero(keep)[0]
    t, lat, lon, p, zobs = t[idx], lat[idx], lon[idx], p[idx], zobs[idx]
    aircraft = rows["address"][idx].astype(np.int64)
    version = rows["version"][idx]
    vrate = rows["vrate_fpm"][idx]
    category = rows["category"][idx]
    alt_baro = rows["alt_baro_ft"][idx]
    print(f"score: {idx.size} of {n_all} rows between {args.p_min_hpa} and {args.p_max_hpa} hPa",
          file=log, flush=True)

    zan, an_valids = sample_series(args.reference, t, p, lat, lon, cache_dir=args.cache_dir, log=log)
    zfc, _ = sample_series(args.forecast, t, p, lat, lon, cache_dir=args.cache_dir,
                           log=log) if args.forecast else (np.full(t.size, np.nan), [])
    r = zobs - zan
    result = {"tool": "arwen_global_airliner_heights score", "rows": str(args.rows),
              "geoid": str(args.geoid),
              "reference": [Path(x).name for x in args.reference],
              "forecast": [Path(x).name for x in (args.forecast or [])],
              "pressure_range_hpa": [args.p_min_hpa, args.p_max_hpa],
              "reports_in_range": int(idx.size), "rows_total": int(n_all)}

    # ---- gross check and classes
    finite = np.isfinite(r)
    gross = finite & (np.abs(r - np.nanmedian(r)) > args.gross_m)
    # 100 ft (Gillham) encoding: an airframe whose every barometric altitude
    # is a multiple of 100 ft over at least 20 reports
    u_air, frac100, n_air = group_mean(aircraft, (alt_baro % 100 == 0).astype(np.float64))
    gillham_air = u_air[(frac100 >= 0.98) & (n_air >= 20)]
    gillham = np.isin(aircraft, gillham_air)
    level = np.abs(np.nan_to_num(vrate, nan=1e9)) <= args.level_fpm
    good = finite & ~gross & ~gillham & (version >= 1)
    result["counts"] = {
        "reference_bracketed": int(finite.sum()),
        "gross_rejected": int(gross.sum()), "gross_m": args.gross_m,
        "gillham_aircraft": int(gillham_air.size), "gillham_reports": int((gillham & finite).sum()),
        "version_0_reports": int((finite & (version == 0)).sum()),
        "version_unknown_reports": int((finite & (version < 0)).sum()),
        "used_reports": int(good.sum()), "used_aircraft": int(np.unique(aircraft[good]).size),
        "level_flight_used": int((good & level).sum()),
    }
    bands = ((100, 250), (250, 350), (350, 500), (500, 700), (700, 850))
    raw = {}
    for lo, hi in bands:
        sel = good & (p >= lo * 100) & (p < hi * 100)
        raw[f"{lo}-{hi}hPa"] = summary(r[sel])
    for v in (0, 1, 2):
        sel = finite & ~gross & ~gillham & (version == v) & (p <= 350e2)
        raw[f"version_{v}_p<=350hPa"] = summary(r[sel])
    for name, code in (("A3_large", CATEGORY["A3"]), ("A5_heavy", CATEGORY["A5"]),
                       ("A1_light", CATEGORY["A1"])):
        sel = good & (category == code) & (p <= 350e2)
        raw[f"category_{name}_p<=350hPa"] = summary(r[sel])
    raw["all_used"] = summary(r[good])
    raw["level_p<=300hPa"] = summary(r[good & level & (p <= 300e2)])
    result["single_report_minus_reference_m"] = raw

    # ---- airframe bias: fitted on the cruise reports before the split hour,
    #      applied to the reports after it (never seen by the fit)
    cruise = good & level & (p <= args.cruise_hpa * 100.0)
    split = dt.datetime.fromisoformat(args.day).replace(tzinfo=dt.timezone.utc).timestamp() \
        + args.split_hour * 3600.0
    fit = cruise & (t < split)
    test = cruise & (t >= split + 3600.0 * args.gap_hours)
    ua, bias, nb = group_mean(aircraft[fit], r[fit], min_n=args.min_fit)
    corr_bias = lookup(aircraft, ua, bias)
    rc = r - corr_bias
    covered = test & np.isfinite(corr_bias)
    result["airframe_bias"] = {
        "cruise_hpa": args.cruise_hpa, "level_fpm": args.level_fpm,
        "split_hour_utc": args.split_hour, "gap_hours": args.gap_hours, "min_fit_reports": args.min_fit,
        "aircraft_fitted": int(ua.size),
        "fitted_bias_spread_m": summary(bias),
        "test_reports_raw": summary(r[test]),
        "test_reports_with_fitted_aircraft_raw": summary(r[covered]),
        "test_reports_corrected_out_of_sample": summary(rc[covered]),
        "test_fraction_with_a_fitted_aircraft": round(float(covered.sum() / max(test.sum(), 1)), 3),
    }
    # stability: the two halves of the day, each airframe's mean
    half = dt.datetime.fromisoformat(args.day).replace(tzinfo=dt.timezone.utc).timestamp() + 43200.0
    a1, m1, _ = group_mean(aircraft[cruise & (t < half)], r[cruise & (t < half)], min_n=args.min_fit)
    a2, m2, _ = group_mean(aircraft[cruise & (t >= half)], r[cruise & (t >= half)], min_n=args.min_fit)
    both, i1, i2 = np.intersect1d(a1, a2, return_indices=True)
    result["airframe_bias"]["halves_correlation"] = regression(m1[i1], m2[i2])
    # Same thing with each half's mean taken over the other aircraft in its cell and hour
    # removed is not attempted: the reference error is common to both halves only where
    # the same aircraft flies through the same weather.

    # ---- the fleet's mean residual hour by hour: a common-mode GNSS height
    #      error that moves with local time would show here (with the
    #      reference's own error at the hour)
    day0 = dt.datetime.fromisoformat(args.day).replace(tzinfo=dt.timezone.utc).timestamp()
    hour_of_day = np.floor((t - day0) / 3600.0).astype(np.int64)
    hourly = []
    for h in range(24):
        sel = cruise & (hour_of_day == h)
        corrected = sel & np.isfinite(corr_bias)
        hourly.append({"hour_utc": h, "reports": int(sel.sum()),
                       "raw_mean_m": round(float(np.mean(r[sel])), 2) if sel.any() else None,
                       "corrected_mean_m": (round(float(np.mean(rc[corrected])), 2)
                                            if corrected.any() else None),
                       "corrected_std_m": (round(float(np.std(rc[corrected])), 2)
                                           if corrected.any() else None)})
    result["cruise_residual_by_hour"] = hourly

    # ---- super-observations (bias-corrected, out of sample)
    lay = np.floor(np.log(p) / np.log(1.0 + args.layer_frac)).astype(np.int64)
    cell = ((np.floor((lat + 90) / args.cell_deg).astype(np.int64) * 1000
             + np.floor(np.mod(lon, 360) / args.cell_deg).astype(np.int64)) * 1000
            + (lay - lay.min())) * 100 + np.floor((t - day0) / 3600.0).astype(np.int64).clip(0, 99)
    so = {}
    for name, sel, res in (("raw_all_day", cruise, r), ("corrected_test", covered, rc)):
        for k in (1, 3, 5, 10):
            c, m, n = superobs(cell[sel], aircraft[sel], res[sel], min_aircraft=k)
            so[f"{name}_min{k}_aircraft"] = summary(m)
    result["superobs_minus_reference_m"] = so
    result["superobs_definition"] = {"cell_deg": args.cell_deg, "layer_ln_p_fraction": args.layer_frac,
                                     "time_bin_h": 1}

    # ---- residual correlation between different aircraft against separation
    bins = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]
    sel = covered
    result["residual_correlation_between_aircraft_corrected_test"] = residual_correlation(
        t[sel], lat[sel], lon[sel], aircraft[sel], rc[sel], bins)
    sel = cruise
    result["residual_correlation_between_aircraft_raw_all_day"] = residual_correlation(
        t[sel], lat[sel], lon[sel], aircraft[sel], r[sel], bins)

    # ---- information test: (O - B) regressed on (A - B)
    if args.forecast:
        ob = zobs - zfc
        ab = zan - zfc
        info = {"background": "the --forecast products at the same valid times"}
        info["single_cruise_raw"] = regression(ab[cruise], ob[cruise])
        info["single_test_corrected"] = regression(ab[covered], (ob - corr_bias)[covered])
        info["a_minus_b_cruise"] = summary(ab[cruise])
        for k in (3, 5):
            c1, m_ob, n1 = superobs(cell[covered], aircraft[covered], (ob - corr_bias)[covered], min_aircraft=k)
            c2, m_ab, n2 = superobs(cell[covered], aircraft[covered], ab[covered], min_aircraft=k)
            info[f"superobs_test_corrected_min{k}"] = regression(m_ab, m_ob)
        for k in (3, 5):
            c1, m_ob, _ = superobs(cell[cruise], aircraft[cruise], ob[cruise], min_aircraft=k)
            c2, m_ab, _ = superobs(cell[cruise], aircraft[cruise], ab[cruise], min_aircraft=k)
            info[f"superobs_raw_all_day_min{k}"] = regression(m_ab, m_ob)
        result["information_test"] = info

    # ---- geoid regression per airframe: slope 0 for heights above the
    #      ellipsoid, -1 for heights above the geoid broadcast as geometric
    nund = geoid(lat, lon)
    slopes = []
    sel_air = cruise
    order = np.argsort(aircraft[sel_air], kind="stable")
    a_s = aircraft[sel_air][order]
    n_s = nund[sel_air][order]
    r_s = r[sel_air][order]
    uniq, start, count = np.unique(a_s, return_index=True, return_counts=True)
    for u, s0, c in zip(uniq, start, count):
        if c < 30:
            continue
        nn = n_s[s0:s0 + c]
        rr = r_s[s0:s0 + c]
        if np.ptp(nn) < args.geoid_span_m:
            continue
        slopes.append(np.polyfit(nn, rr, 1)[0])
    slopes = np.array(slopes)
    result["geoid_regression_per_aircraft"] = {
        "aircraft": int(slopes.size), "min_geoid_span_m": args.geoid_span_m,
        "slope": summary(slopes),
        "near_0": int((np.abs(slopes) < 0.3).sum()),
        "near_minus_1": int((np.abs(slopes + 1) < 0.3).sum()),
    }

    # ---- layer-mean virtual temperature from climbs and descents: the
    #      difference of two heights of one aircraft, where an airframe's
    #      constant bias cancels (the quantity de Haan's method uses)
    sel = np.nonzero(good)[0]
    li, lj = layer_pairs(aircraft[sel], t[sel], p[sel], min_ln=args.layer_ln, max_ln=args.layer_ln + 0.12)
    li, lj = sel[li], sel[lj]
    t_obs = layer_temperature_k(p[li], p[lj], zobs[lj] - zobs[li])
    t_ref = layer_temperature_k(p[li], p[lj], zan[lj] - zan[li])
    dtk = t_obs - t_ref
    mid = np.sqrt(p[li] * p[lj])
    layer = {"min_ln_p": args.layer_ln, "max_ln_p": round(args.layer_ln + 0.12, 3),
             "pairs": int(li.size), "aircraft": int(np.unique(aircraft[li]).size),
             "obs_minus_reference_k": summary(dtk)}
    for lo, hi in ((150, 300), (300, 450), (450, 600), (600, 850)):
        s = (mid >= lo * 100) & (mid < hi * 100)
        layer[f"obs_minus_reference_k_mid_{lo}-{hi}hPa"] = summary(dtk[s])
    if args.forecast:
        t_fc = layer_temperature_k(p[li], p[lj], zfc[lj] - zfc[li])
        layer["information_test"] = regression(t_ref - t_fc, t_obs - t_fc)
        # super-observations: pairs averaged per 1-degree cell, hour and layer
        # midpoint band of 10 percent, then over aircraft
        lkey = ((np.floor((lat[li] + 90) / args.cell_deg).astype(np.int64) * 1000
                 + np.floor(np.mod(lon[li], 360) / args.cell_deg).astype(np.int64)) * 100
                + np.floor(np.log(mid) / np.log(1.1)).astype(np.int64) % 100) * 100 \
            + np.floor((t[li] - day0) / 3600.0).astype(np.int64).clip(0, 99)
        for k in (3, 5):
            _, m1, _ = superobs(lkey, aircraft[li], dtk, min_aircraft=k)
            _, m_ob, _ = superobs(lkey, aircraft[li], t_obs - t_fc, min_aircraft=k)
            _, m_ab, _ = superobs(lkey, aircraft[li], t_ref - t_fc, min_aircraft=k)
            layer[f"superobs_min{k}_obs_minus_reference_k"] = summary(m1)
            layer[f"superobs_min{k}_information_test"] = regression(m_ab, m_ob)
    result["layer_temperature"] = layer

    # ---- radiosondes
    if args.sondes:
        an_by_valid = {dt.datetime.fromtimestamp(v, dt.timezone.utc): path
                       for v, path in zip(an_valids, args.reference)}
        nominal = [f"{args.day}T00:00:00Z", f"{args.day}T12:00:00Z"]
        coll = collocate_sondes(
            args.sondes, nominal,
            {"t": t[cruise], "p": p[cruise], "lat": lat[cruise], "lon": lon[cruise],
             "aircraft": aircraft[cruise], "r": r[cruise], "rc": rc[cruise]},
            radius_km=args.sonde_km, hours=args.sonde_hours, dp_hpa=args.sonde_dp_hpa,
            levels_hpa=(200.0, 250.0, 300.0), an_paths_by_valid=an_by_valid, cache_dir=args.cache_dir, log=log)
        x = np.array([c["x_aircraft_minus_ref"] for c in coll])
        y = np.array([c["y_sonde_minus_ref"] for c in coll])
        xc = np.array([np.nan if c["x_corrected_aircraft_minus_ref"] is None
                       else c["x_corrected_aircraft_minus_ref"] for c in coll])
        sonde = {"radius_km": args.sonde_km, "hours": args.sonde_hours,
                 "dp_hpa": args.sonde_dp_hpa, "collocations": len(coll),
                 "note": "x: aircraft mean residual (no airframe correction) near the sounding; "
                         "x_corrected: the same with each airframe's bias fitted before the "
                         "split hour removed (the fit includes these reports); y: sonde minus "
                         "reference at the level; x - y: aircraft minus sonde through the reference"}
        if len(coll) >= 3:
            ok = np.isfinite(xc)
            sonde.update({"x": summary(x), "y": summary(y), "x_minus_y": summary(x - y),
                          "corr_xy": round(float(np.corrcoef(x, y)[0, 1]), 3),
                          "cov_xy_m2": round(float(np.cov(x, y)[0, 1]), 2),
                          "x_corrected": summary(xc[ok]), "y_where_corrected": summary(y[ok]),
                          "x_corrected_minus_y": summary(xc[ok] - y[ok]),
                          "corr_xcorrected_y": (round(float(np.corrcoef(xc[ok], y[ok])[0, 1]), 3)
                                                if ok.sum() >= 3 else None)})
            for lev in (200.0, 250.0, 300.0):
                s = np.array([c["level_hpa"] == lev for c in coll])
                sonde[f"x_minus_y_{int(lev)}hPa"] = summary((x - y)[s])
        result["radiosondes"] = sonde
        result["radiosonde_collocations"] = coll

    result["seconds"] = round(time.time() - began, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "airliner-heights-stats.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if not args.no_plots:
        land = _land_mask(args.reference[0])
        plots(out, land, lat, lon, p, r, rc, cruise, covered, good, args,
              result, zobs, zan, zfc, layer_dt=dtk)
    return result


def plots(out, land, lat, lon, p, r, rc, cruise, covered, good, args, result, zobs, zan, zfc,
          layer_dt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    def coast(ax):
        if land is None:
            return
        lo = np.mod(land["lon"] + 180.0, 360.0) - 180.0
        order = np.argsort(lo)
        ax.contour(lo[order], land["lat"], land["values"][:, order], levels=[0.5],
                   colors="0.35", linewidths=0.4)

    fig, axes = plt.subplots(2, 1, figsize=(13, 13), constrained_layout=True)
    ax = axes[0]
    h, xe, ye = np.histogram2d(np.mod(lon[good] + 180, 360) - 180, lat[good],
                               bins=(360, 180), range=((-180, 180), (-90, 90)))
    im = ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, cmap="viridis",
                       norm=LogNorm(vmin=1, vmax=max(h.max(), 10)), shading="flat")
    coast(ax)
    if "radiosonde_collocations" in result:
        sl = np.array([[c["lon"], c["lat"]] for c in result["radiosonde_collocations"]])
        if sl.size:
            ax.scatter(np.mod(sl[:, 0] + 180, 360) - 180, sl[:, 1], s=14, facecolors="none",
                       edgecolors="red", linewidths=0.7, label="radiosonde collocated")
            ax.legend(loc="lower left", fontsize=8)
    c = result["counts"]
    ax.set_title(f"{args.day}: ADS-B reports carrying both altitudes, {args.p_min_hpa:g} to "
                 f"{args.p_max_hpa:g} hPa, one per aircraft per {args.spacing_s:g} s\n"
                 f"{c['used_reports']:,} reports from {c['used_aircraft']:,} aircraft, "
                 f"counted per 1-degree cell; red rings: radiosondes collocated", fontsize=10)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="reports per 1-degree cell")
    ax = axes[1]
    sel = covered
    sums, xe, ye = np.histogram2d(np.mod(lon[sel] + 180, 360) - 180, lat[sel],
                                  bins=(180, 90), range=((-180, 180), (-90, 90)), weights=rc[sel])
    cnt, _, _ = np.histogram2d(np.mod(lon[sel] + 180, 360) - 180, lat[sel],
                               bins=(180, 90), range=((-180, 180), (-90, 90)))
    mean = np.where(cnt >= 20, sums / np.maximum(cnt, 1), np.nan)
    im = ax.pcolormesh(xe, ye, mean.T, cmap="RdBu_r", vmin=-20, vmax=20, shading="flat")
    coast(ax)
    ax.set_title(f"Aircraft minus reference height at the aircraft's pressure, cruise, airframe bias "
                 f"removed\n(fitted before {args.split_hour:02d}Z, applied after), mean per 2-degree "
                 f"cell with 20 or more reports", fontsize=10)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="m")
    fig.savefig(out / "airliner-coverage-map.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)
    ax = axes[0]
    for name, sel, res in (("raw", cruise, r), ("bias-corrected, out of sample", covered, rc)):
        x = res[sel]
        x = x[np.isfinite(x)]
        ax.hist(x, bins=np.arange(-80, 81, 2), histtype="step", density=True, label=f"{name} (n={x.size:,})")
    ax.set_xlabel("aircraft minus reference (m)"); ax.set_ylabel("density")
    ax.set_title(f"Cruise reports at or above {args.cruise_hpa:g} hPa, level flight")
    ax.legend(fontsize=8)
    ax = axes[1]
    key = "residual_correlation_between_aircraft_corrected_test"
    for label, k in (("bias-corrected, out of sample", key),
                     ("raw, all day", "residual_correlation_between_aircraft_raw_all_day")):
        b = result[k]["bins"]
        xs = [0.5 * (x["from_km"] + x["to_km"]) for x in b if x["correlation"] is not None]
        ys = [x["correlation"] for x in b if x["correlation"] is not None]
        ax.plot(xs, ys, "o-", label=label)
    ax.set_xscale("log"); ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("separation of two different aircraft (km)")
    ax.set_ylabel("correlation of hour-mean residuals")
    ax.set_title("Residual correlation between different aircraft")
    ax.legend(fontsize=8)
    ax = axes[2]
    if args.forecast:
        ab = zan - zfc
        ob = zobs - zfc - np.nan_to_num(r - rc)
        sel = covered & np.isfinite(ab) & np.isfinite(ob)
        ax.hexbin(ab[sel], ob[sel], gridsize=80, bins="log", extent=(-150, 150, -150, 150), cmap="Greys")
        ax.plot([-150, 150], [-150, 150], color="red", lw=0.8)
        info = result["information_test"]["single_test_corrected"]
        ax.set_title(f"Single reports, bias-corrected: slope {info.get('slope')}, "
                     f"r {info.get('correlation')}", fontsize=10)
        ax.set_xlabel("reference minus forecast (m)")
        ax.set_ylabel("aircraft minus forecast (m)")
    ax = axes[3]
    if layer_dt is not None and layer_dt.size:
        x = layer_dt[np.isfinite(layer_dt)]
        ax.hist(x, bins=np.arange(-6, 6.01, 0.2), histtype="step", density=True)
        lt = result["layer_temperature"]
        ax.set_title(f"Climbs and descents: layer-mean temperature\n{lt['pairs']:,} layers, "
                     f"std {lt['obs_minus_reference_k'].get('std')} K, robust "
                     f"{lt['obs_minus_reference_k'].get('robust_std')} K", fontsize=10)
        ax.set_xlabel("aircraft minus reference (K)")
    fig.savefig(out / "airliner-error-statistics.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="the day's split tar to a row file")
    e.add_argument("--parts", nargs="+", required=True, help="the split tar parts, in order")
    e.add_argument("--day", required=True, help="YYYY-MM-DD the archive covers")
    e.add_argument("--out", required=True, help=".npz row file (a .json record beside it)")
    e.add_argument("--floor-ft", type=float, default=5000.0)
    e.add_argument("--spacing-s", type=float, default=60.0)
    e.add_argument("--limit", type=int, default=None, help="stop after this many trace files")
    g = sub.add_parser("geoid", help="EGM96 15' GeoTIFF to the operator's .npz")
    g.add_argument("--tif", required=True)
    g.add_argument("--out", required=True)
    s = sub.add_parser("score", help="the operator against references and sondes")
    s.add_argument("--rows", required=True)
    s.add_argument("--geoid", required=True)
    s.add_argument("--day", required=True)
    s.add_argument("--reference", nargs="+", required=True,
                   help="hourly GRIB2 products with HGT and TMP on isobaric levels")
    s.add_argument("--forecast", nargs="*", default=None,
                   help="longer-lead GRIB2 products valid at the same hours (the background)")
    s.add_argument("--sondes", nargs="*", default=None, help="directories of IGRA2 station files")
    s.add_argument("--out", required=True, help="directory for the JSON and the PNGs")
    s.add_argument("--cache-dir", default=None,
                   help="keep each product's decoded HGT and TMP here (uncompressed .npz)")
    s.add_argument("--p-min-hpa", type=float, default=150.0)
    s.add_argument("--p-max-hpa", type=float, default=850.0)
    s.add_argument("--cruise-hpa", type=float, default=350.0)
    s.add_argument("--level-fpm", type=float, default=500.0)
    s.add_argument("--gross-m", type=float, default=300.0)
    s.add_argument("--split-hour", type=int, default=18)
    s.add_argument("--gap-hours", type=float, default=0.0,
                   help="test only reports this many hours after the split, so a flight fitted "
                        "before it is not scored minutes later")
    s.add_argument("--min-fit", type=int, default=10)
    s.add_argument("--cell-deg", type=float, default=1.0)
    s.add_argument("--layer-frac", type=float, default=0.1,
                   help="super-observation layer thickness as a fraction of pressure")
    s.add_argument("--geoid-span-m", type=float, default=20.0)
    s.add_argument("--layer-ln", type=float, default=0.18,
                   help="least ln-pressure depth of a climb or descent layer (0.18 is about "
                        "100 hPa at 550 hPa)")
    s.add_argument("--sonde-km", type=float, default=150.0)
    s.add_argument("--sonde-hours", type=float, default=1.0)
    s.add_argument("--sonde-dp-hpa", type=float, default=15.0)
    s.add_argument("--spacing-s", type=float, default=60.0, help="as extracted (for the map title)")
    s.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "extract":
        record = extract(args.parts, args.day, args.out, floor_ft=args.floor_ft,
                         spacing_s=args.spacing_s, limit=args.limit)
        print(json.dumps({k: record[k] for k in ("aircraft_kept", "seconds")}))
        return 0
    if args.command == "geoid":
        import tifffile
        grid = tifffile.imread(args.tif).astype(np.float32)
        nlat, nlon = grid.shape
        step = 180.0 / (nlat - 1)
        np.savez_compressed(args.out, lat=90.0 - step * np.arange(nlat),
                            lon=-180.0 + step * np.arange(nlon), n_m=grid)
        return 0
    result = score(args)
    print(json.dumps(result["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
