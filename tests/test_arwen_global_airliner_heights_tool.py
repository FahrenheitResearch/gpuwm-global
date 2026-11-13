"""The airliner height operator (``tools/arwen_global_airliner_heights.py``)
against known answers, on the CPU, with planted inputs only.

* The ICAO standard atmosphere: published pressures at three flight levels
  and the layer bases, and the inversion round trip from 0 to 30 km.
* Geopotential height of a height above sea level: the 1976 US Standard
  Atmosphere's pair (11 000 gpm at 11 019.13 m geometric on its 6 356 766 m
  radius) and the WGS84 free-air gravity formula integrated numerically at
  three latitudes and three heights, both within 0.1 m.
* The operator end to end: an aircraft flying a pressure altitude through a
  standard atmosphere, with its GNSS height planted from the inverse of the
  conversion plus a planted geoid, reads its pressure altitude back as the
  geopotential height of its pressure.
* The vertical read: exact for an isothermal layer and for temperature
  linear in ln p, within 0.5 m for the standard troposphere on 50 hPa levels.
* The GRIB2 reader: messages built here with simple packing (5.0) and with
  complex packing and first- and second-order spatial differencing (5.3)
  decode to the planted field to the packing quantum.
* The trace reader: every cause of rejection is counted on a planted trace,
  and a split tar of gzip members reads across the split.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import arwen_global_airliner_heights as tool  # noqa: E402

FT = 0.3048


# ------------------------------------------------- the standard atmosphere

@pytest.mark.parametrize("flight_level, hpa", [
    (100, 696.8), (350, 238.4), (390, 196.8)])
def test_published_flight_level_pressures(flight_level, hpa):
    p = tool.isa_pressure_pa(flight_level * 100 * FT)
    assert abs(p / 100.0 - hpa) < 0.06


def test_layer_bases_and_round_trip():
    assert tool.isa_pressure_pa(0.0) == pytest.approx(101325.0, abs=1e-6)
    assert tool.isa_pressure_pa(11000.0) == pytest.approx(22632.06, abs=0.01)
    assert tool.isa_pressure_pa(20000.0) == pytest.approx(5474.89, abs=0.01)
    h = np.linspace(-300.0, 30000.0, 3001)
    back = tool.isa_pressure_altitude_m(tool.isa_pressure_pa(h))
    assert np.max(np.abs(back - h)) < 1e-6


# ------------------------------------------------- geopotential height

def test_us_standard_atmosphere_pair():
    # At the latitude where WGS84 normal gravity equals g0 (near 45.6 N) the
    # 1976 pair holds: 11 000 gpm at 11 019.13 m on a 6 356 766 m radius.
    lo, hi = 40.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tool.normal_gravity(mid) < tool.G0 else (lo, mid)
    lat = 0.5 * (lo + hi)
    assert abs(tool.normal_gravity(lat) - tool.G0) < 1e-9
    z = tool.geopotential_height_m(11019.13, lat)
    r0 = 6356766.0
    assert abs(z - r0 * 11019.13 / (r0 + 11019.13)) < 0.01
    assert abs(z - 11000.0) < 0.1      # the published geometric height is rounded


@pytest.mark.parametrize("lat", [0.0, 45.0, 80.0])
@pytest.mark.parametrize("height", [3000.0, 11000.0, 13000.0])
def test_closed_form_matches_integrated_free_air_gravity(lat, height):
    # WGS84 normal gravity above the ellipsoid (NIMA TR8350.2, eq. 4-3):
    # g(h) = g_s [1 - 2h/a (1 + f + m - 2 f sin^2 phi) + 3 h^2 / a^2].
    s2 = np.sin(np.radians(lat)) ** 2
    gs = tool.normal_gravity(lat)
    a, f, m = tool.WGS84_A, tool.WGS84_F, tool.WGS84_M
    hh = np.linspace(0.0, height, 20001)
    g = gs * (1 - 2 * hh / a * (1 + f + m - 2 * f * s2) + 3 * hh ** 2 / a ** 2)
    integral = np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(hh)) / tool.G0
    assert abs(tool.geopotential_height_m(height, lat) - integral) < 0.1


def test_equator_to_pole_difference_is_tens_of_metres():
    z_eq = tool.geopotential_height_m(11000.0, 0.0)
    z_pole = tool.geopotential_height_m(11000.0, 90.0)
    assert 50.0 < z_pole - z_eq < 65.0


# ------------------------------------------------- geoid and the operator

def _planted_geoid():
    lat = np.arange(90.0, -90.25, -0.25)
    lon = np.arange(-180.0, 180.0, 0.25)
    n = 30.0 * np.cos(np.radians(lat))[:, None] + 0.1 * lon[None, :]
    return tool.Geoid(lat, lon, n), lat, lon, n


def test_geoid_reads_bilinear_and_wraps():
    geoid, _, _, _ = _planted_geoid()
    assert geoid(10.0, 20.0) == pytest.approx(30.0 * np.cos(np.radians(10.0)) + 2.0, abs=1e-9)
    assert geoid(10.125, 20.125) == pytest.approx(
        0.5 * (30 * np.cos(np.radians(10.0)) + 30 * np.cos(np.radians(10.25))) + 2.0125, abs=1e-6)
    # 200 E is -160: the ring wraps
    assert geoid(0.0, 200.0) == pytest.approx(30.0 - 16.0, abs=1e-9)


def test_operator_reads_pressure_altitude_back_in_a_standard_atmosphere():
    geoid, _, _, _ = _planted_geoid()
    rng = np.random.default_rng(3)
    lat = rng.uniform(-70, 70, 200)
    lon = rng.uniform(-180, 180, 200)
    alt_baro_ft = rng.uniform(5000, 41000, 200)
    zp = alt_baro_ft * FT                     # in ISA, pressure altitude IS gpm
    ratio = tool.normal_gravity(lat) / tool.G0
    r = tool.effective_radius_m(lat)
    h_msl = zp * r / (ratio * r - zp)          # the inverse of the conversion
    alt_geom_ft = (h_msl + geoid(lat, lon)) / FT
    p, z = tool.height_observation(alt_baro_ft, alt_geom_ft, lat, lon, geoid)
    assert np.max(np.abs(z - zp)) < 0.01
    assert np.max(np.abs(p - tool.isa_pressure_pa(zp))) < 1e-6


# ------------------------------------------------- the vertical read

def test_hypsometric_read_is_exact_for_isothermal_and_log_linear_layers():
    coef = tool.RD / tool.G0
    p1, p2 = 30000.0, 25000.0
    p = np.array([30000.0, 29000.0, 27300.0, 25000.0])
    # isothermal 220 K
    z1 = 9000.0
    z2 = z1 + coef * 220.0 * np.log(p1 / p2)
    got = tool.hypsometric_between(p, p1, p2, z1, z2, 220.0, 220.0)
    assert np.max(np.abs(got - (z1 + coef * 220.0 * np.log(p1 / p)))) < 1e-6
    # T linear in ln p from 230 to 220 K
    span = np.log(p1 / p2)
    x = np.log(p1 / p)
    z2 = z1 + coef * span * 225.0
    want = z1 + coef * (230.0 * x - 10.0 * x * x / (2 * span))
    got = tool.hypsometric_between(p, p1, p2, z1, z2, 230.0, 220.0)
    assert np.max(np.abs(got - want)) < 1e-6


def _isa_sampler(levels_hpa, valid=None):
    lat = np.arange(90.0, -90.25, -0.25)[::40]
    lon = np.arange(0.0, 360.0, 0.25)[::40]
    msgs = []
    valid = valid or dt.datetime(2026, 11, 10, tzinfo=dt.timezone.utc)
    for hpa in levels_hpa:
        zp = float(tool.isa_pressure_altitude_m(hpa * 100.0))
        h = zp
        layer = [b for b in tool.ISA_LAYERS if b[0] <= h][-1]
        temp = layer[1] + layer[2] * (h - layer[0])
        for (cat, num), value in (((3, 5), zp), ((0, 0), temp)):
            msgs.append({"discipline": 0, "category": cat, "number": num, "surface": 100,
                         "level": hpa * 100.0, "reference": valid, "valid": valid,
                         "lat": lat, "lon": lon,
                         "values": np.full((lat.size, lon.size), value, np.float32)})
    return tool.ReferenceSampler(msgs)


def test_reference_sampler_reads_the_standard_troposphere_within_half_a_metre():
    sampler = _isa_sampler([850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150])
    p = np.linspace(84000.0, 26000.0, 300)
    z = sampler.height_at(p, np.full(p.size, 47.3), np.full(p.size, 351.1))
    want = tool.isa_pressure_altitude_m(p)
    assert np.max(np.abs(z - want)) < 0.5
    # outside the levels: NaN, never an extrapolation
    out = sampler.height_at(np.array([90000.0, 14000.0]), np.zeros(2), np.zeros(2))
    assert np.all(np.isnan(out))


# ------------------------------------------------- GRIB2 messages built here

def _sm(value: int, nbytes: int) -> bytes:
    """Sign-magnitude integer."""
    mag = abs(int(value))
    return (mag | ((1 << (8 * nbytes - 1)) if value < 0 else 0)).to_bytes(nbytes, "big")


def _pack(values, widths) -> bytes:
    bits = "".join(format(int(v), f"0{int(w)}b") if w else "" for v, w in zip(values, widths))
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""


def _message(field: np.ndarray, *, template: int, order: int = 2, d: int = 1,
             group: int = 7, level_pa: int = 25000, cat: int = 3, num: int = 5) -> bytes:
    nj, ni = field.shape
    ints = np.round(field.ravel() * 10 ** d).astype(np.int64)
    sec1 = bytes([0, 0, 0, 21, 1, 0, 7, 0, 0, 2, 1, 1]) + (2026).to_bytes(2, "big") + bytes([11, 10, 6, 0, 0, 0, 1])
    sec3 = (b"\0\0\0\0" + bytes([3, 0]) + (ni * nj).to_bytes(4, "big") + bytes([0, 0]) + (0).to_bytes(2, "big")
            + bytes([6]) + bytes(15) + ni.to_bytes(4, "big") + nj.to_bytes(4, "big") + bytes(8)
            + _sm(90_000_000, 4) + _sm(0, 4) + bytes([48])
            + _sm(90_000_000 - (nj - 1) * 10_000_000, 4) + _sm((ni - 1) * 10_000_000, 4)
            + (10_000_000).to_bytes(4, "big") + (10_000_000).to_bytes(4, "big") + bytes([0]))
    sec3 = len(sec3).to_bytes(4, "big") + sec3[4:]
    sec4 = (b"\0\0\0\0" + bytes([4]) + (0).to_bytes(2, "big") + (0).to_bytes(2, "big") + bytes([cat, num, 2, 0, 81])
            + bytes(3) + bytes([1]) + (6).to_bytes(4, "big") + bytes([100, 0]) + level_pa.to_bytes(4, "big")
            + bytes([255, 0]) + bytes(4))
    sec4 = len(sec4).to_bytes(4, "big") + sec4[4:]
    ref_min = int(ints.min()) if template == 0 else 0
    if template == 0:
        x = ints - ref_min
        nbits = max(int(x.max()).bit_length(), 1)
        data = _pack(x, [nbits] * x.size)
        tmpl = struct.pack(">f", float(ref_min)) + _sm(0, 2) + _sm(d, 2) + bytes([nbits, 0])
    else:
        if order == 1:
            ival = [int(ints[0])]
            diff = np.concatenate(([0], np.diff(ints)))
        else:
            ival = [int(ints[0]), int(ints[1])]
            diff = np.concatenate(([0, 0], ints[2:] - 2 * ints[1:-1] + ints[:-2]))
        minsd = int(diff[order:].min())
        x = diff - minsd
        x[:order] = 0
        groups = [x[i:i + group] for i in range(0, x.size, group)]
        grefs = [int(g.min()) for g in groups]
        widths = [int((g - gr).max()).bit_length() for g, gr in zip(groups, grefs)]
        lengths = [g.size for g in groups]
        nbits = max(max(grefs).bit_length(), 1)
        bits_w = max(max(widths).bit_length(), 1)
        bits_l = max(max(lengths).bit_length(), 1)
        nbytes = 4
        extra = b"".join(_sm(v, nbytes) for v in ival) + _sm(minsd, nbytes)
        data = (extra + _pack(grefs, [nbits] * len(grefs)) + _pack(widths, [bits_w] * len(widths))
                + _pack(lengths, [bits_l] * len(lengths))
                + _pack(np.concatenate([g - gr for g, gr in zip(groups, grefs)]),
                        np.concatenate([[w] * len(g) for g, w in zip(groups, widths)])))
        tmpl = (struct.pack(">f", 0.0) + _sm(0, 2) + _sm(d, 2) + bytes([nbits, 0, 1, 0])
                + bytes(8) + len(groups).to_bytes(4, "big") + bytes([0, bits_w])
                + (0).to_bytes(4, "big") + bytes([1]) + lengths[-1].to_bytes(4, "big")
                + bytes([bits_l, order, nbytes]))
    sec5 = (ni * nj).to_bytes(4, "big") + (template).to_bytes(2, "big") + tmpl
    sec5 = (len(sec5) + 5).to_bytes(4, "big") + bytes([5]) + sec5
    sec6 = (6).to_bytes(4, "big") + bytes([6, 255])
    sec7 = (len(data) + 5).to_bytes(4, "big") + bytes([7]) + data
    body = sec1 + sec3 + sec4 + sec5 + sec6 + sec7 + b"7777"
    return b"GRIB" + bytes([0, 0, 0, 2]) + (16 + len(body)).to_bytes(8, "big") + body


def _field(nj=19, ni=36):
    lat = 90.0 - 10.0 * np.arange(nj)
    lon = 10.0 * np.arange(ni)
    return (10500.0 + 300.0 * np.cos(np.radians(lat))[:, None]
            + 25.0 * np.sin(np.radians(2 * lon))[None, :]
            - 7.3 * np.cos(np.radians(lon + lat[:, None])))


@pytest.mark.parametrize("template, order", [(0, 0), (3, 1), (3, 2)])
def test_grib2_reader_decodes_planted_fields(tmp_path, template, order):
    field = _field()
    path = tmp_path / "planted.grib2"
    path.write_bytes(_message(field, template=template, order=order)
                     + _message(field - 100.0, template=template, order=order, level_pa=30000))
    msgs = tool.read_grib2(path)
    assert len(msgs) == 2
    m = msgs[0]
    assert (m["discipline"], m["category"], m["number"], m["surface"], m["level"]) == (0, 3, 5, 100, 25000.0)
    assert m["valid"] == dt.datetime(2026, 11, 10, 12, tzinfo=dt.timezone.utc)
    assert m["lat"][0] == 90.0 and m["lat"][-1] == -90.0 and m["lon"][1] == 10.0
    assert np.max(np.abs(m["values"] - field)) <= 0.05 + 1e-3
    assert np.max(np.abs(msgs[1]["values"] - (field - 100.0))) <= 0.05 + 1e-3
    only = tool.read_grib2(path, lambda d, c, n, s, lv: lv == 30000.0)
    assert len(only) == 1 and only[0]["level"] == 30000.0


def test_unpack_bits_reads_mixed_widths():
    values = [5, 0, 1023, 1, 77777, 3]
    widths = [3, 0, 10, 1, 17, 2]
    got = tool.unpack_bits(_pack(values, widths), 0, widths)
    assert got.tolist() == values


# ------------------------------------------------- the trace reader

DAY0 = dt.datetime(2026, 11, 10, tzinfo=dt.timezone.utc).timestamp()


def _trace(icao="4ca123", t0=DAY0):
    obj = {"type": "adsb_icao", "version": 2, "gva": 2, "nic_baro": 1, "nac_p": 9, "category": "A3"}
    return {"icao": icao, "timestamp": t0, "trace": [
        [0.0, 50.0, 5.0, 35000, 450, 90, 0, 0, obj, "adsb_icao", 35900, 0, 260, 0],    # kept
        [10.0, 50.0, 5.1, 35000, 450, 90, 0, 0, None, "adsb_icao", 35900, 0, 260, 0],  # thinned
        [70.0, 50.0, 5.3, 35025, 450, 90, 0, 64, None, "adsb_icao", 35925, 0, 260, 0],  # kept
        [80.0, 50.0, 5.3, "ground", 0, 90, 0, 0, None, "adsb_icao", None, 0, 0, 0],     # ground
        [140.0, 50.0, 5.4, 35000, 450, 90, 1, 0, None, "adsb_icao", 35900, 0, 260, 0],  # stale
        [200.0, 50.0, 5.5, 35000, 450, 90, 0, 0, None, "adsb_icao", None, 0, 260, 0],   # no geom
        [260.0, 50.0, 5.6, 35000, 450, 90, 0, 0, None, "mlat", 35900, 0, 260, 0],       # mlat
        [320.0, 50.0, 5.7, 4000, 250, 90, 0, -900, None, "adsb_icao", 4500, 0, 200, 0],  # below floor
        [380.0, 50.0, 5.8, 35000, 450, 90, 8, 0, None, "adsb_icao", 35900, 0, 260, 0],  # alt is geometric
        [90000.0, 50.0, 5.9, 35000, 450, 90, 0, 0, None, "adsb_icao", 35900, 0, 260, 0],  # next day
    ]}


def test_trace_rows_count_every_cause():
    from collections import Counter

    counts = Counter()
    rows = tool.trace_rows(_trace(), DAY0, DAY0 + 86400, floor_ft=5000, spacing_s=60, counts=counts)
    assert len(rows) == 2
    assert rows[0][0] == 0x4ca123 and rows[0][4] == 35000 and rows[0][5] == 35900
    assert rows[1][8:] == (2, 2, 1, 9, tool.CATEGORY["A3"])   # carried from the first object
    assert counts == Counter({"points": 10, "kept": 2, "thinned": 1, "on_ground": 1,
                              "position_stale": 1, "geom_missing": 1, "source_mlat": 1,
                              "below_floor": 1, "altitude_is_geometric": 1, "outside_day": 1})
    counts = Counter()
    assert tool.trace_rows(_trace(icao="~0a0b0c"), DAY0, DAY0 + 86400, floor_ft=5000,
                           spacing_s=60, counts=counts) == []
    assert counts["trace_non_icao_address"] == 1


def test_extract_reads_a_split_tar_of_gzip_members(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for i, icao in enumerate(("4ca123", "a0b1c2", "3c4d5e")):
            data = gzip.compress(json.dumps(_trace(icao=icao)).encode())
            info = tarfile.TarInfo(f"./traces/{icao[-2:]}/trace_full_{icao}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        blob = b"\0" * 5000
        info = tarfile.TarInfo("./heatmap/00.bin.ttf")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    raw = buf.getvalue()
    parts = [tmp_path / "day.tar.aa", tmp_path / "day.tar.ab"]
    parts[0].write_bytes(raw[:1700])
    parts[1].write_bytes(raw[1700:])
    record = tool.extract(parts, "2026-11-10", tmp_path / "rows.npz")
    assert record["counts"]["kept"] == 6
    assert record["counts"]["trace_files"] == 3
    assert record["counts"]["members_not_traces"] == 1
    with np.load(tmp_path / "rows.npz") as z:
        assert sorted(set(z["address"].tolist())) == [0x3c4d5e, 0x4ca123, 0xa0b1c2]
        assert z["alt_geom_ft"].tolist().count(35925) == 3


# ------------------------------------------------- climbs and descents

def test_layer_pairs_take_one_climb_and_skip_level_flight():
    # aircraft 1 climbs 5 percent in pressure a minute; aircraft 2 is level;
    # aircraft 3 climbs, levels off for a step, then climbs again
    p1 = 80000.0 * 0.95 ** np.arange(12)
    p2 = np.full(12, 25000.0)
    p3 = 80000.0 * 0.95 ** np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p = np.concatenate([p1, p2, p3])
    aircraft = np.repeat([1, 2, 3], 12)
    t = np.tile(np.arange(12) * 60.0, 3)
    i, j = tool.layer_pairs(aircraft, t, p, min_ln=0.18, max_ln=0.30)
    pairs = set(zip(i.tolist(), j.tolist()))
    # ln(1/0.95) = 0.0513 per step: four steps (0.205) is the first to reach 0.18
    assert (0, 4) in pairs and (7, 11) in pairs
    assert all(a < 12 or a >= 24 for a, _ in pairs)          # none from the level aircraft
    assert all(b - a == 4 for a, b in pairs)
    # aircraft 3's pause breaks the single-climb rule for pairs spanning it
    assert (24, 28) not in pairs and (26, 30) not in pairs and (28, 32) in pairs


def test_layer_temperature_reads_an_isothermal_layer():
    coef = tool.RD / tool.G0
    dz = coef * 230.0 * np.log(50000.0 / 40000.0)
    assert tool.layer_temperature_k(50000.0, 40000.0, dz) == pytest.approx(230.0, abs=1e-9)
    assert tool.layer_temperature_k(40000.0, 50000.0, -dz) == pytest.approx(230.0, abs=1e-9)


# ------------------------------------------------- the statistics helpers

def test_superobs_average_aircraft_first():
    cells = np.array([1, 1, 1, 1, 2, 2])
    aircraft = np.array([7, 7, 7, 8, 7, 9])
    resid = np.array([1.0, 1.0, 1.0, 5.0, 2.0, 4.0])
    c, m, n = tool.superobs(cells, aircraft, resid, min_aircraft=2)
    assert c.tolist() == [1, 2] and m.tolist() == [3.0, 3.0] and n.tolist() == [2, 2]
    c, m, n = tool.superobs(cells, aircraft, resid, min_aircraft=3)
    assert c.size == 0


def test_residual_correlation_finds_a_planted_common_mode():
    rng = np.random.default_rng(1)
    n_air = 400
    lat = rng.uniform(40, 55, n_air)
    lon = rng.uniform(-5, 20, n_air)
    t = np.full(n_air, DAY0 + 1800.0)
    noise = rng.normal(0, 5, n_air)
    common = 10.0 * np.sin(np.radians(lon) * 8)         # a wave about 3,400 km long
    out = tool.residual_correlation(t, lat, lon, np.arange(n_air), noise + common,
                                    [0, 100, 400, 3200])
    near = out["bins"][0]["correlation"]
    assert near > 0.3
    out = tool.residual_correlation(t, lat, lon, np.arange(n_air), noise, [0, 100, 400, 3200])
    assert abs(out["bins"][0]["correlation"]) < 0.2
