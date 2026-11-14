"""A small GRIB edition 1 reader: latitude-longitude grids, simple packing,
optional bitmap (WMO FM 92 GRIB edition 1, sections 0-5), written from the
WMO specification for the NOAA GODAS pentad files (no new dependency).

Each record yields its parameter number, level type and value, the valid
date, the grid and the unpacked field (NaN where the bitmap masks it):

    value = (R + X * 2^E) / 10^D

with R the reference value (IBM single-precision float), E the binary and D
the decimal scale factor (GRIB 1 sign-and-magnitude integers) and X the
packed integer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _u(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _s(b: bytes) -> int:
    """GRIB 1 signed integer: the top bit is the sign, the rest the magnitude."""
    v = _u(b)
    top = 1 << (8 * len(b) - 1)
    return -(v & (top - 1)) if v & top else v


def _ibm(b: bytes) -> float:
    v = _u(b)
    sign = -1.0 if v >> 31 else 1.0
    exponent = (v >> 24) & 0x7F
    mantissa = v & 0xFFFFFF
    return sign * mantissa / float(1 << 24) * 16.0 ** (exponent - 64)


@dataclass
class Record:
    parameter: int
    table: int
    level_type: int
    level: int
    date: tuple
    ni: int
    nj: int
    lat1: float
    lon1: float
    lat2: float
    lon2: float
    di: float
    dj: float
    scanning: int
    values: np.ndarray  # (nj, ni) south-to-north when the scanning flag says +j

    @property
    def lats(self) -> np.ndarray:
        step = (self.lat2 - self.lat1) / (self.nj - 1)
        return self.lat1 + step * np.arange(self.nj)

    @property
    def lons(self) -> np.ndarray:
        step = (self.lon2 - self.lon1) / (self.ni - 1)
        return self.lon1 + step * np.arange(self.ni)


def _unpack(data: bytes, nbits: int, count: int) -> np.ndarray:
    if nbits == 0:
        return np.zeros(count, dtype=np.float64)
    raw = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(raw)[: nbits * count].reshape(count, nbits)
    weights = (1 << np.arange(nbits - 1, -1, -1, dtype=np.uint64)).astype(np.uint64)
    return (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.float64)


def read(path) -> list[Record]:
    buf = Path(path).read_bytes()
    out = []
    pos = 0
    n = len(buf)
    while True:
        start = buf.find(b"GRIB", pos)
        if start < 0 or start + 8 > n:
            break
        total = _u(buf[start + 4:start + 7])
        if buf[start + 7] != 1:
            raise ValueError(f"{path}: record at {start} is GRIB edition {buf[start + 7]}, not 1")
        p = start + 8
        pds_len = _u(buf[p:p + 3])
        pds = buf[p:p + pds_len]
        table = pds[3]
        flag = pds[7]
        param = pds[8]
        ltype = pds[9]
        level = _u(pds[10:12])
        yy, mm, dd, hh = pds[12], pds[13], pds[14], pds[15]
        century = pds[24] if pds_len > 24 else 21
        year = (century - 1) * 100 + yy
        dscale = _s(pds[26:28])
        p += pds_len
        if not flag & 0x80:
            raise ValueError(f"{path}: record without a grid description section")
        gds_len = _u(buf[p:p + 3])
        gds = buf[p:p + gds_len]
        if gds[5] != 0:
            raise ValueError(f"{path}: grid type {gds[5]} is not latitude-longitude")
        ni, nj = _u(gds[6:8]), _u(gds[8:10])
        lat1, lon1 = _s(gds[10:13]) / 1000.0, _s(gds[13:16]) / 1000.0
        lat2, lon2 = _s(gds[17:20]) / 1000.0, _s(gds[20:23]) / 1000.0
        di, dj = _u(gds[23:25]) / 1000.0, _u(gds[25:27]) / 1000.0
        scan = gds[27]
        p += gds_len
        bitmap = None
        if flag & 0x40:
            bms_len = _u(buf[p:p + 3])
            if _u(buf[p + 4:p + 6]) != 0:
                raise ValueError(f"{path}: predefined bitmaps are not supported")
            bits = np.unpackbits(np.frombuffer(buf[p + 6:p + bms_len], dtype=np.uint8))
            bitmap = bits[: ni * nj].astype(bool)
            p += bms_len
        bds_len = _u(buf[p:p + 3])
        bds = buf[p:p + bds_len]
        if bds[3] & 0xF0:
            raise ValueError(f"{path}: only simple grid-point packing is supported")
        escale = _s(bds[4:6])
        ref = _ibm(bds[6:10])
        nbits = bds[10]
        count = int(bitmap.sum()) if bitmap is not None else ni * nj
        packed = _unpack(bds[11:], nbits, count)
        vals = (ref + packed * 2.0 ** escale) / 10.0 ** dscale
        field = np.full(ni * nj, np.nan)
        if bitmap is not None:
            field[bitmap] = vals
        else:
            field[:] = vals
        field = field.reshape(nj, ni)
        if scan & 0x80:  # -i
            field = field[:, ::-1]
            lon1, lon2 = lon2, lon1
        if not scan & 0x40:  # -j: north to south as stored
            field = field[::-1]
            lat1, lat2 = lat2, lat1
        out.append(Record(parameter=param, table=table, level_type=ltype, level=level,
                          date=(year, mm, dd, hh), ni=ni, nj=nj, lat1=lat1, lon1=lon1, lat2=lat2,
                          lon2=lon2, di=di, dj=dj, scanning=scan, values=field))
        pos = start + total
    return out


#: NCEP parameter numbers in the GODAS pentads (table 2 / 129), checked
#: against each file's own .inv listing by :func:`godas_fields`.
GODAS_PARAMETERS = {(2, 13): "POT", (2, 88): "SALTY", (2, 49): "UOGRD", (2, 50): "VOGRD", (2, 40): "DZDT",
                    (129, 198): "SSHG", (2, 124): "UFLX", (2, 125): "VFLX", (129, 202): "THFLX",
                    (129, 199): "SLTFL", (129, 195): "DBSS"}


def godas_fields(path) -> dict:
    """The pentad's fields: 3-D (40, nj, ni) POT (deg C), SALTY (psu), UOGRD,
    VOGRD, DZDT, the depths (m), and 2-D SSHG (m), UFLX, VFLX (N m-2),
    THFLX (W m-2), SLTFL, with ``lats``/``lons`` and the date.  The names
    come from the .inv beside the file when it exists (checked against the
    parameter table), so a mis-identified record is refused."""
    recs = read(path)
    inv = Path(str(path) + ".inv")
    names = None
    if inv.exists():
        names = [line.split(":")[3] for line in inv.read_text().splitlines() if line.strip()]
        if len(names) != len(recs):
            raise ValueError(f"{path}: {len(recs)} records but {len(names)} lines in {inv}")
    fields: dict = {}
    depths: dict = {}
    for i, r in enumerate(recs):
        name = GODAS_PARAMETERS.get((r.table, r.parameter), f"P{r.table}.{r.parameter}")
        if names is not None and names[i] != name:
            raise ValueError(f"{path}: record {i + 1} is parameter {r.parameter} ({name}) but "
                             f"the .inv says {names[i]}")
        if r.level_type == 160:  # depth below sea level (m)
            fields.setdefault(name, []).append(r.values)
            depths.setdefault(name, []).append(r.level)
        else:
            fields.setdefault(name, []).append(r.values)
    out = {"lats": recs[0].lats, "lons": recs[0].lons, "date": recs[0].date}
    for name, planes in fields.items():
        out[name] = np.stack(planes) if name in depths else planes[0]
    out["depth_m"] = np.asarray(depths.get("POT", []), dtype=np.float64)
    if "POT" in out:
        out["POT"] = out["POT"] - 273.15
    if "SALTY" in out:
        out["SALTY"] = out["SALTY"] * 1000.0
    return out


__all__ = ["GODAS_PARAMETERS", "Record", "godas_fields", "read"]
