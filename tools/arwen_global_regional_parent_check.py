"""Measure whether a regional run is being fed by its Arwen Global parent.

``python tools/arwen_global_regional_parent_check.py track REGIONAL PARENT_FRAME [--label TEXT] [--json OUT]``
``python tools/arwen_global_regional_parent_check.py diff REGIONAL_A REGIONAL_B [--label TEXT] [--json OUT]``

``track`` compares one regional history file (a WRF-shaped ``wrfout``) with
one parent frame written by ``gpuwm-global export-mapped-parent``.  The
parent is interpolated here, independently of the regional preparation:
bilinear in latitude and longitude, linear in ln(p) to the regional mass
level pressures.  Regional winds are destaggered and rotated to earth
relative.  Values where the regional pressure lies below the parent's lowest
full level (regional terrain lower than the parent's) are left out.  The
result is bias and RMS of temperature, both wind components, vapour and
surface pressure by distance ``d`` in grid points from the nearest lateral
edge:

* ``spec``, ``d = 0``: the specified row, set from the boundary file;
* ``relax``, ``d = 1..4``: the relaxation rows, nudged to the boundary file;
* ``near``, ``d = 5..19``;
* ``inner``, ``d >= 60``: the interior, free to depart.

A run that is fed shows three things with this one instrument.  At the
initial time every zone sits at the same small residual, which is the
difference between this interpolation and the preparation's.  At a later
boundary time the ``spec`` row is still at that residual against the parent
frame of the same valid time while the interior has moved away.  And against
the FIRST parent frame the ``spec`` row is far away, because the boundary
moved with the parent.

``diff`` compares two regional history files of the same valid time, for a
fed run against a control whose boundary was held at the initial state: RMS
difference by the same zones and by the 20-point strip along each side, with
the mean wind into the domain between 850 and 500 hPa on that side.

Both need only numpy and netCDF4, and no device.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

GRAVITY_M_S2 = 9.80616
DRY_AIR_GAS_CONSTANT = 287.0
#: The parent's kappa (287/1004) and the regional engine's (2/7): each file's
#: potential temperature goes back to temperature with its own.
KAPPA_PARENT = 287.0 / 1004.0
KAPPA_REGIONAL = 2.0 / 7.0
REFERENCE_PRESSURE_PA = 1.0e5
THETA_OFFSET_K = 300.0
#: Lapse rate used to carry the parent's surface pressure to the regional
#: terrain, half the standard 6.5 K/km so the layer mean is what is used.
HALF_LAPSE_K_M = 0.00325

TRACK_FIELDS = ("T K", "u m/s", "v m/s", "qv g/kg")
DIFF_FIELDS = (("T", "T K"), ("u", "u m/s"), ("v", "v m/s"), ("q", "qv g/kg"),
               ("psfc", "psfc hPa"), ("t2", "T2 K"))


def edge_distance(ny: int, nx: int) -> np.ndarray:
    """Grid points from the nearest lateral edge, 0 on the outermost row."""
    jj, ii = np.mgrid[0:ny, 0:nx]
    return np.minimum(np.minimum(ii, nx - 1 - ii), np.minimum(jj, ny - 1 - jj))


def track_zones(distance: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("spec d=0", distance == 0),
        ("relax d=1-4", (distance >= 1) & (distance <= 4)),
        ("near d=5-19", (distance >= 5) & (distance <= 19)),
        ("inner d>=60", distance >= 60),
        ("all interior d>=5", distance >= 5),
    ]


def _times(dataset) -> str:
    return b"".join(dataset["Times"][0].tolist()).decode()


def read_regional(path: str) -> dict[str, object]:
    """Mass-point temperature, earth-relative wind, vapour and the surface
    from one regional history file."""
    from netCDF4 import Dataset

    with Dataset(path) as data:
        def get(name):
            return np.asarray(data[name][0], dtype=np.float64)

        pressure = get("P") + get("PB")
        u_stag, v_stag = get("U"), get("V")
        u_grid = 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])
        v_grid = 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])
        cosa, sina = get("COSALPHA"), get("SINALPHA")
        return {
            "lat": get("XLAT"), "lon": get("XLONG"), "p": pressure,
            "T": (get("T") + THETA_OFFSET_K)
                 * (pressure / REFERENCE_PRESSURE_PA) ** KAPPA_REGIONAL,
            "q": get("QVAPOR"),
            "u": u_grid * cosa - v_grid * sina,
            "v": v_grid * cosa + u_grid * sina,
            "u_grid": u_grid, "v_grid": v_grid,
            "psfc": get("PSFC"), "hgt": get("HGT"), "t2": get("T2"),
            "time": _times(data),
        }


def read_parent(path: str) -> dict[str, object]:
    from netCDF4 import Dataset

    with Dataset(path) as data:
        def get(name):
            return np.asarray(data[name][0], dtype=np.float64)

        pressure = get("PB")
        return {
            "lat": np.asarray(data["lat"][:], dtype=np.float64),
            "lon": np.asarray(data["lon"][:], dtype=np.float64),
            "p": pressure,
            "T": (get("T") + THETA_OFFSET_K)
                 * (pressure / REFERENCE_PRESSURE_PA) ** KAPPA_PARENT,
            "q": get("QVAPOR"), "u": get("U_EARTH"), "v": get("V_EARTH"),
            "psfc": get("PSFC"), "hgt": get("HGT"),
            "time": str(data.valid_time),
        }


def parent_at_regional_points(regional, parent) -> dict[str, np.ndarray]:
    """The parent frame at the regional mass points: ``T``, ``u``, ``v``,
    ``q`` (levels, y, x), ``psfc`` carried to the regional terrain, and
    ``ok``, false where the regional pressure is outside the parent's full
    levels."""
    lat, lon = regional["lat"], regional["lon"]
    plat, plon = parent["lat"], parent["lon"]
    fy = (lat - plat[0]) / (plat[1] - plat[0])
    fx = (lon - plon[0]) / (plon[1] - plon[0])
    if not (fy.min() >= 0 and fy.max() <= plat.size - 1
            and fx.min() >= 0 and fx.max() <= plon.size - 1):
        raise ValueError("the regional grid leaves the parent frame's window")
    j0 = np.clip(np.floor(fy).astype(int), 0, plat.size - 2)
    i0 = np.clip(np.floor(fx).astype(int), 0, plon.size - 2)
    wy = fy - j0
    wx = fx - i0

    def horizontal(field):
        return ((1 - wy) * (1 - wx) * field[..., j0, i0]
                + (1 - wy) * wx * field[..., j0, i0 + 1]
                + wy * (1 - wx) * field[..., j0 + 1, i0]
                + wy * wx * field[..., j0 + 1, i0 + 1])

    log_parent = np.log(horizontal(parent["p"]))      # (parent levels, y, x), bottom to top
    log_regional = np.log(regional["p"])               # (regional levels, y, x)
    count = log_parent.shape[0]
    columns = {name: horizontal(parent[name]) for name in ("T", "q", "u", "v")}

    def vertical(field):
        out = np.empty_like(log_regional)
        ok = np.empty(log_regional.shape, dtype=bool)
        for level in range(log_regional.shape[0]):
            # parent levels with higher pressure than this regional level
            below = (log_parent > log_regional[level][None]).sum(axis=0)
            k = np.clip(below - 1, 0, count - 2)[None]
            lower = np.take_along_axis(log_parent, k, 0)[0]
            upper = np.take_along_axis(log_parent, k + 1, 0)[0]
            weight = (lower - log_regional[level]) / (lower - upper)
            out[level] = ((1 - weight) * np.take_along_axis(field, k, 0)[0]
                          + weight * np.take_along_axis(field, k + 1, 0)[0])
            ok[level] = (below >= 1) & (below <= count - 1)
        return out, ok

    result: dict[str, np.ndarray] = {}
    for name, column in columns.items():
        result[name], result["ok"] = vertical(column)
    # parent surface pressure carried hydrostatically to the regional terrain
    rise = regional["hgt"] - horizontal(parent["hgt"])
    mean_temperature = columns["T"][0] - HALF_LAPSE_K_M * rise
    result["psfc"] = horizontal(parent["psfc"]) * np.exp(
        -GRAVITY_M_S2 * rise / (DRY_AIR_GAS_CONSTANT * mean_temperature))
    return result


def track(regional_path: str, parent_path: str, label: str = "") -> dict[str, object]:
    regional = read_regional(regional_path)
    parent = read_parent(parent_path)
    at = parent_at_regional_points(regional, parent)
    ok = at["ok"]
    differences = (
        ("T K", regional["T"] - at["T"]), ("u m/s", regional["u"] - at["u"]),
        ("v m/s", regional["v"] - at["v"]),
        ("qv g/kg", 1000.0 * (regional["q"] - at["q"])),
    )
    surface = (regional["psfc"] - at["psfc"]) / 100.0
    ny, nx = regional["lat"].shape
    result: dict[str, object] = {
        "regional": regional_path, "parent": parent_path,
        "regional_time": regional["time"], "parent_time": parent["time"],
        "label": label,
        "excluded_below_parent_fraction": float(1.0 - ok.mean()), "zones": {},
    }
    for zone, mask in track_zones(edge_distance(ny, nx)):
        selected = ok & mask[None]
        row: dict[str, object] = {"columns": int(mask.sum())}
        for name, difference in differences:
            values = difference[selected]
            row[name] = {"bias": float(values.mean()),
                         "rms": float(np.sqrt((values * values).mean()))}
        values = surface[mask]
        row["psfc hPa"] = {"bias": float(values.mean()),
                           "rms": float(np.sqrt((values * values).mean()))}
        result["zones"][zone] = row
    return result


def print_track(result: dict[str, object]) -> None:
    names = (*TRACK_FIELDS, "psfc hPa")
    print(f"## {result['label'] or result['regional_time']}: regional "
          f"{result['regional_time']} vs parent {result['parent_time']}")
    print("levels excluded (regional pressure below the parent's lowest full "
          f"level): {100 * result['excluded_below_parent_fraction']:.2f} % of grid values")
    print("| zone | columns | " + " | ".join(f"{n} bias | {n} rms" for n in names) + " |")
    print("|---|---|" + "---|" * (2 * len(names)))
    for zone, row in result["zones"].items():
        cells = []
        for name in names:
            cells += [f"{row[name]['bias']:+.3f}", f"{row[name]['rms']:.3f}"]
        print(f"| {zone} | {row['columns']} | " + " | ".join(cells) + " |")


def run_difference(path_a: str, path_b: str, label: str = "") -> dict[str, object]:
    """RMS difference of two regional history files of one valid time, by
    zone and by side.  Winds are compared on the model grid (grid relative)."""
    first, second = read_regional(path_a), read_regional(path_b)
    if first["time"] != second["time"]:
        raise ValueError(
            f"the two files are not of one valid time: {first['time']} and {second['time']}")
    state = []
    for run in (first, second):
        state.append({"T": run["T"], "u": run["u_grid"], "v": run["v_grid"],
                      "q": 1000.0 * run["q"], "psfc": run["psfc"] / 100.0,
                      "t2": run["t2"]})
    a, b = state
    ny, nx = first["psfc"].shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    distance = edge_distance(ny, nx)
    zones = [
        ("spec d=0", distance == 0),
        ("relax d=1-4", (distance >= 1) & (distance <= 4)),
        ("near d=5-19", (distance >= 5) & (distance <= 19)),
        ("inner d>=60", distance >= 60),
        ("west strip i<20", (ii < 20) & (jj >= 20) & (jj < ny - 20)),
        ("east strip", (ii >= nx - 20) & (jj >= 20) & (jj < ny - 20)),
        ("south strip j<20", (jj < 20) & (ii >= 20) & (ii < nx - 20)),
        ("north strip", (jj >= ny - 20) & (ii >= 20) & (ii < nx - 20)),
        ("whole domain", distance >= 0),
    ]
    layer = (first["p"] < 85000.0) & (first["p"] > 50000.0)
    inward = {"west": ("u", 1.0), "east": ("u", -1.0),
              "south": ("v", 1.0), "north": ("v", -1.0)}
    result: dict[str, object] = {"a": path_a, "b": path_b, "time": first["time"],
                                 "label": label, "zones": {}}
    for zone, mask in zones:
        row: dict[str, object] = {"columns": int(mask.sum())}
        for key, title in DIFF_FIELDS:
            values = (a[key] - b[key])[..., mask]
            row[title] = float(np.sqrt((values * values).mean()))
        row["max abs T K"] = float(np.abs((a["T"] - b["T"])[..., mask]).max())
        side = zone.split()[0]
        if side in inward:
            key, sign = inward[side]
            row["inward wind m/s"] = float(sign * a[key][layer & mask[None]].mean())
        result["zones"][zone] = row
    return result


def print_difference(result: dict[str, object]) -> None:
    print(f"## {result['label'] or result['time']}: RMS difference at {result['time']}")
    print("| zone | columns | " + " | ".join(f"{title} rms" for _key, title in DIFF_FIELDS)
          + " | max abs T K | inward wind of A, 850-500 hPa, m/s |")
    print("|---|---|" + "---|" * (len(DIFF_FIELDS) + 2))
    for zone, row in result["zones"].items():
        cells = [f"{row[title]:.3f}" for _key, title in DIFF_FIELDS]
        cells.append(f"{row['max abs T K']:.2f}")
        cells.append(f"{row['inward wind m/s']:+.1f}" if "inward wind m/s" in row else "")
        print(f"| {zone} | {row['columns']} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, first, second in (("track", "regional", "parent"), ("diff", "a", "b")):
        leg = sub.add_parser(name)
        leg.add_argument(first)
        leg.add_argument(second)
        leg.add_argument("--label", default="")
        leg.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if args.command == "track":
        result = track(args.regional, args.parent, args.label)
        print_track(result)
    else:
        result = run_difference(args.a, args.b, args.label)
        print_difference(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
