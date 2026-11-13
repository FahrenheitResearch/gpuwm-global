"""Published single-column cases run through the model's own physics suite.

This is the single-column bench of the scheme workshop: the shipped native
suite (RRTMGP, sfclay, YSU, Grell-Freitas, Morrison) is handed a column whose
atmosphere, large-scale forcing and surface are a published case's, and it
answers with its cloud, rain, surface fluxes and radiation.  The suite is the
same object the global model calls (:class:`ArwenCudaColumnSuite`), on the
same 40-level stack, at the same 300 s step with the same Strang split (the
suite over dt/2, the forcing over dt, the suite over dt/2), so a case scores
the scheme the forecast runs, at the vertical resolution the forecast has.

What the harness supplies in place of the dycore, and nothing else:

* the case's initial column, laid on the hybrid levels at the case's surface
  pressure (heights from the coordinate's own hydrostatic integral);
* the case's large-scale forcing between the two physics half-steps:
  vertical advection by the prescribed subsidence (first-order upwind in
  height, on theta and every water species), the prescribed horizontal
  advection of temperature and water, the geostrophic (Coriolis) term,
  wind nudging where the case asks for it, and the prescribed radiative
  tendency in the ``spec`` arm when the case prescribes one;
* the pressure velocity of the prescribed subsidence on the half levels
  (the exchange's ``omega_half_pa_s``, which Grell-Freitas reads);
* the case's surface: the sea-surface or skin temperature, and where the
  case prescribes the surface fluxes (and friction velocity or roughness),
  those values written over the surface layer's own on every call.

Two module substitutions do the last two jobs, through the suite's
``modules`` door (the same door the CPU tests use for fakes): the surface
layer is the shipped ``sfclay`` with the case's fluxes written over its
result, and in the ``spec`` arm of a case that prescribes its radiation the
shipped RRTMGP still runs (its fluxes are read as diagnostics) but its
heating is not applied.  Nothing in the package changes; a run of the model
never sees this file.

Arms.  ``spec`` is the case as published: its radiation (a prescribed
tendency, a formula in liquid water path, or none), its surface.  This is
the arm the published LES ranges score.  ``native-radiation`` drops the
case's own radiation and keeps the model's RRTMGP heating; it is allowed only
where the case states its radiation separately (``tend`` and ``formula``
cases, and ``on`` cases, where it is the spec arm).

The radiation clock.  RRTMGP's sun follows ``start_time_utc`` plus the run
time.  In the ``spec`` arm of a case whose radiation is prescribed, RRTMGP is
a diagnostic only, so ``sun="noon"`` may move its clock so that the middle
of the scoring window is local solar noon: the model's reflected sunlight
for the cloud the model made, which is the number the albedo error is about.

Above a case's top level.  The cases define the lowest few kilometres; the
model's column reaches 1 hPa.  Above the top of a temperature profile the
column joins the US standard atmosphere through an offset that decays with a
3 km e-folding height, vapour decays from its top value with a 2.5 km
e-folding height to a 3 ppm floor and never exceeds 30 percent of ice
saturation (so the joined column holds no cloud of its own), the winds and the geostrophic wind hold
their top values, and every prescribed tendency and the subsidence are zero.

Columns.  A case runs as ``ncol`` identical columns of one batch (default
two).  Identical columns must come back identical; the result records the
largest difference between them as a check.

Documented departures, all stated in each result's ``departures`` list:
the land cases (ARM, LBA, GABLS1) run as water columns to the cumulus and
PBL schemes, because their fluxes are prescribed and Noah's would overwrite
them; their surface layer runs its land branch through the substitution.
Advection of liquid-water potential temperature and of total water is
applied to theta and to vapour.

Usage (on a host with a card)::

    python tools/arwen_global_scm_cases.py run --case bomex \
        --dephy-root <dir holding BOMEX_REF/BOMEX_REF_DEF_driver.nc> \
        --arm spec --out <dir>

``--case`` is one of :data:`CASES`.  The DEPHY files come from the
community repository (https://github.com/GdR-DEPHY/DEPHY-SCM); DYCOMS-II RF01
is built here from Stevens et al. (2005), which DEPHY does not carry.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

RESULT_SCHEMA = "arwen-global-scm-result/v1"

#: Physical constants the harness itself uses.  The model's own come from
#: arwen_global.constants when the suite runs; these are the same values,
#: repeated so the case builders and their tests import nothing from the
#: package.
GRAVITY = 9.80665
RD = 287.04
RV = 461.5
CP = 1004.64
KAPPA_DRY = RD / CP
P0 = 100_000.0
LV = 2.5e6
EARTH_OMEGA = 7.292e-5
#: Standard atmosphere joined above a case's top level.
UPPER_JOIN_EFOLD_M = 3000.0
UPPER_VAPOUR_EFOLD_M = 2500.0
UPPER_VAPOUR_FLOOR = 3.0e-6
UPPER_RELATIVE_HUMIDITY = 0.3
#: A level is cloudy for the base and top heights when its cloud liquid
#: plus ice exceeds this specific content (kg/kg).
CLOUD_THRESHOLD = 1.0e-5


# --------------------------------------------------------------------------
# Case representation.


def _time_index(times: np.ndarray, t: float) -> tuple[int, int, float]:
    """Bracketing indices and weight of ``t`` in ``times`` (held at the ends)."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 1 or t <= times[0]:
        return 0, 0, 0.0
    if t >= times[-1]:
        last = times.size - 1
        return last, last, 0.0
    upper = int(np.searchsorted(times, t, side="right"))
    lower = upper - 1
    weight = (t - times[lower]) / (times[upper] - times[lower])
    return lower, upper, float(weight)


@dataclass(frozen=True)
class Profile:
    """A profile in height (m above the surface), piecewise linear in height
    and in time.  ``heights`` and ``values`` are ``(ntime, nlev)``.  Below
    the lowest height the lowest value holds; above the highest the value is
    zero (``above="zero"``, for tendencies and subsidence) or holds
    (``above="hold"``, for winds and initial profiles)."""

    times: np.ndarray
    heights: np.ndarray
    values: np.ndarray
    above: str = "hold"

    def __post_init__(self) -> None:
        times = np.atleast_1d(np.asarray(self.times, dtype=np.float64))
        heights = np.atleast_2d(np.asarray(self.heights, dtype=np.float64))
        values = np.atleast_2d(np.asarray(self.values, dtype=np.float64))
        if heights.shape[0] == 1 and values.shape[0] > 1:
            heights = np.repeat(heights, values.shape[0], axis=0)
        if heights.shape != values.shape or heights.shape[0] != times.size:
            raise ValueError(
                f"profile shapes disagree: times {times.shape}, heights "
                f"{heights.shape}, values {values.shape}")
        order = np.argsort(heights, axis=1, kind="stable")
        heights = np.take_along_axis(heights, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        if self.above not in ("zero", "hold"):
            raise ValueError(f"profile above={self.above!r}")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "heights", heights)
        object.__setattr__(self, "values", values)

    @classmethod
    def constant(cls, heights, values, above: str = "hold") -> "Profile":
        return cls(np.zeros(1), np.asarray(heights)[None], np.asarray(values)[None], above)

    @property
    def top_m(self) -> float:
        return float(self.heights[0, -1])

    def _one(self, index: int, z: np.ndarray) -> np.ndarray:
        h = self.heights[index]
        v = self.values[index]
        right = 0.0 if self.above == "zero" else v[-1]
        out = np.interp(z, h, v, left=v[0], right=right)
        if self.above == "zero":
            # The last defined level keeps its value; zero only strictly above.
            out = np.where(z > h[-1], 0.0, out)
        return out

    def at(self, t: float, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        lower, upper, weight = _time_index(self.times, float(t))
        a = self._one(lower, z)
        if upper == lower:
            return a
        return (1.0 - weight) * a + weight * self._one(upper, z)


@dataclass(frozen=True)
class Series:
    """A scalar series in time, linear between samples, held at the ends."""

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", np.atleast_1d(np.asarray(self.times, np.float64)))
        object.__setattr__(self, "values", np.atleast_1d(np.asarray(self.values, np.float64)))
        if self.times.shape != self.values.shape:
            raise ValueError("series times and values differ in length")

    @classmethod
    def constant(cls, value: float) -> "Series":
        return cls(np.zeros(1), np.asarray([value], np.float64))

    def at(self, t: float) -> float:
        lower, upper, weight = _time_index(self.times, float(t))
        return float((1.0 - weight) * self.values[lower] + weight * self.values[upper])


@dataclass
class SCMCase:
    """One single-column case, in the DEPHY conventions."""

    name: str
    title: str
    reference: str
    source: str
    latitude_deg: float
    longitude_deg: float
    start_utc: datetime
    duration_s: float
    window_s: tuple[float, float]
    surface_pressure_pa: float
    u0: Profile
    v0: Profile
    temperature_kind: str          # "ta", "theta" or "thetal"
    temperature0: Profile
    water_kind: str                # "qv", "qt", "rv" or "rt"
    water0: Profile
    radiation: str                 # "tend", "off", "on" or "formula"
    surface_type: str              # "ocean" or "land"
    orography_m: float = 0.0
    ug: Profile | None = None
    vg: Profile | None = None
    w: Profile | None = None
    t_adv_kind: str | None = None
    t_adv: Profile | None = None
    q_adv_kind: str | None = None
    q_adv: Profile | None = None
    t_rad_kind: str | None = None
    t_rad: Profile | None = None
    #: (column) -> theta tendency (K/s) for radiation = "formula".
    radiation_formula: Callable | None = None
    nudge_u: Profile | None = None
    nudge_v: Profile | None = None
    nudge_u_tau_s: float | None = None
    nudge_v_tau_s: float | None = None
    skin_kind: str | None = None   # "ts" or "thetas"
    skin: Series | None = None
    sensible_w_m2: Series | None = None
    latent_w_m2: Series | None = None
    ustar_m_s: Series | None = None
    z0_m: Series | None = None
    moisture_availability: Series | None = None
    departures: list[str] = field(default_factory=list)

    @property
    def coriolis_s(self) -> float:
        return 2.0 * EARTH_OMEGA * math.sin(math.radians(self.latitude_deg))

    @property
    def prescribed_fluxes(self) -> bool:
        return self.sensible_w_m2 is not None or self.latent_w_m2 is not None


# --------------------------------------------------------------------------
# DEPHY reader.

_TEMPERATURE_KINDS = ("ta", "theta", "thetal")
_WATER_KINDS = ("qv", "qt", "rv", "rt")


def _dephy_profile(arrays: Mapping[str, np.ndarray], name: str, *, above: str,
                   init: bool) -> Profile:
    values = np.asarray(arrays[name], dtype=np.float64)
    heights_key = f"zh_{name}"
    if heights_key not in arrays:
        raise ValueError(
            f"DEPHY variable {name!r} has no height coordinate {heights_key!r}; "
            "this harness reads cases defined on heights only")
    heights = np.asarray(arrays[heights_key], dtype=np.float64)
    times_key = "t0" if init else f"time_{name}"
    times = np.asarray(arrays.get(times_key, arrays["t0"]), dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if heights.ndim == 1:
        heights = heights[None]
    return Profile(times, heights, values, above)


def _dephy_series(arrays, name) -> Series | None:
    if name not in arrays:
        return None
    values = np.asarray(arrays[name], dtype=np.float64).reshape(-1)
    times = np.asarray(arrays.get(f"time_{name}", arrays["t0"]), dtype=np.float64).reshape(-1)
    if times.size != values.size:
        times = np.asarray(arrays["t0"], dtype=np.float64).reshape(-1)[: values.size]
    return Series(times, values)


def _flag(attrs, name) -> float:
    value = attrs.get(name, 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dephy_date(text: str) -> datetime:
    return datetime.strptime(str(text).strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def case_from_dephy(attrs: Mapping[str, object], arrays: Mapping[str, np.ndarray], *,
                    name: str, source: str, duration_s: float | None = None,
                    window_s: tuple[float, float] | None = None) -> SCMCase:
    """A case from a DEPHY-SCM ``*_DEF_driver.nc``'s global attributes and
    variables (read by :func:`read_netcdf`, or written by hand in a test)."""
    if "forc_wap" in attrs and _flag(attrs, "forc_wap"):
        raise ValueError(f"{name}: pressure-velocity forcing (forc_wap) is not read")
    if _flag(attrs, "forc_p") and not (_flag(attrs, "forc_z") or _flag(attrs, "forc_zh")):
        raise ValueError(f"{name}: forcing on pressure levels only is not read")
    for kind in ("nudging_ta", "nudging_theta", "nudging_thetal", "nudging_qv",
                 "nudging_qt", "nudging_rv", "nudging_rt"):
        if _flag(attrs, kind):
            raise ValueError(f"{name}: {kind} is not implemented")
    temperature_kind = next(
        (k for k in _TEMPERATURE_KINDS if _flag(attrs, f"ini_{k}") and k in arrays), None)
    water_kind = next(
        (k for k in _WATER_KINDS if _flag(attrs, f"ini_{k}") and k in arrays), None)
    if temperature_kind is None or water_kind is None:
        raise ValueError(f"{name}: no initial temperature or water profile flagged")
    start = _dephy_date(attrs["start_date"])
    end = _dephy_date(attrs["end_date"])
    duration = float(duration_s if duration_s is not None else (end - start).total_seconds())
    case = SCMCase(
        name=name,
        title=str(attrs.get("title", name)),
        reference=str(attrs.get("reference", "")),
        source=source,
        latitude_deg=float(np.asarray(arrays["lat"]).reshape(-1)[0]),
        longitude_deg=float(np.asarray(arrays["lon"]).reshape(-1)[0]),
        start_utc=start,
        duration_s=duration,
        window_s=tuple(window_s) if window_s is not None else (max(0.0, duration - 3600.0), duration),
        surface_pressure_pa=float(np.asarray(arrays["ps"]).reshape(-1)[0]),
        orography_m=float(np.asarray(arrays.get("orog", [0.0])).reshape(-1)[0]),
        u0=_dephy_profile(arrays, "ua", above="hold", init=True),
        v0=_dephy_profile(arrays, "va", above="hold", init=True),
        temperature_kind=temperature_kind,
        temperature0=_dephy_profile(arrays, temperature_kind, above="hold", init=True),
        water_kind=water_kind,
        water0=_dephy_profile(arrays, water_kind, above="hold", init=True),
        radiation=str(attrs.get("radiation", "off")),
        surface_type=str(attrs.get("surface_type", "ocean")),
    )
    if _flag(attrs, "forc_geo"):
        case.ug = _dephy_profile(arrays, "ug", above="hold", init=False)
        case.vg = _dephy_profile(arrays, "vg", above="hold", init=False)
    if _flag(attrs, "forc_wa"):
        case.w = _dephy_profile(arrays, "wa", above="zero", init=False)
    for kind in _TEMPERATURE_KINDS:
        if _flag(attrs, f"adv_{kind}"):
            case.t_adv_kind = kind
            case.t_adv = _dephy_profile(arrays, f"tn{kind}_adv", above="zero", init=False)
    for kind in _WATER_KINDS:
        if _flag(attrs, f"adv_{kind}"):
            case.q_adv_kind = kind
            case.q_adv = _dephy_profile(arrays, f"tn{kind}_adv", above="zero", init=False)
    if case.radiation == "tend":
        for kind in _TEMPERATURE_KINDS:
            if f"tn{kind}_rad" in arrays:
                case.t_rad_kind = kind
                case.t_rad = _dephy_profile(arrays, f"tn{kind}_rad", above="zero", init=False)
        if case.t_rad is None:
            raise ValueError(f"{name}: radiation='tend' and no tn*_rad variable")
    for axis in ("u", "v"):
        tau = _flag(attrs, f"nudging_{axis}a")
        if tau:
            setattr(case, f"nudge_{axis}", _dephy_profile(arrays, f"{axis}a_nud", above="hold", init=False))
            setattr(case, f"nudge_{axis}_tau_s", tau)
    temp_forcing = str(attrs.get("surface_forcing_temp", "none"))
    if temp_forcing == "surface_flux":
        case.sensible_w_m2 = _dephy_series(arrays, "hfss")
        case.latent_w_m2 = _dephy_series(arrays, "hfls")
        if case.latent_w_m2 is None:
            case.latent_w_m2 = Series.constant(0.0)
        case.skin_kind = "ts" if ("tskin" in arrays or "ts" in arrays) else None
        case.skin = _dephy_series(arrays, "tskin") or _dephy_series(arrays, "ts")
        if case.skin is None:
            case.departures.append(
                "no skin temperature in the case: the skin is diagnosed each call "
                "from the prescribed sensible heat flux through the surface "
                "layer's own heat exchange coefficient")
    elif temp_forcing == "ts":
        case.skin_kind = "ts"
        case.skin = _dephy_series(arrays, "ts_forc") or _dephy_series(arrays, "ts")
    elif temp_forcing == "thetas":
        case.skin_kind = "thetas"
        case.skin = _dephy_series(arrays, "thetas_forc") or _dephy_series(arrays, "thetas")
    else:
        raise ValueError(f"{name}: surface_forcing_temp={temp_forcing!r} is not read")
    wind_forcing = str(attrs.get("surface_forcing_wind", "none"))
    if wind_forcing == "ustar":
        case.ustar_m_s = _dephy_series(arrays, "ustar")
    elif wind_forcing == "z0":
        case.z0_m = _dephy_series(arrays, "z0")
    moisture = str(attrs.get("surface_forcing_moisture", "none"))
    if moisture == "beta":
        case.moisture_availability = _dephy_series(arrays, "beta")
    if case.surface_type != "ocean":
        case.departures.append(
            "land case run as water columns to YSU and the cumulus scheme "
            "(its fluxes are prescribed and Noah would overwrite them); the "
            "surface layer runs its land branch with the case's roughness")
    if case.t_adv_kind == "thetal" or case.q_adv_kind == "qt" or case.temperature_kind == "thetal":
        case.departures.append(
            "liquid-water potential temperature and total-water forcing applied "
            "to theta and vapour")
    return case


def read_netcdf(path: str | Path) -> tuple[dict, dict]:
    """Global attributes and variables of a DEPHY netCDF file, as numpy."""
    import netCDF4  # noqa: PLC0415 - only the card host needs it

    with netCDF4.Dataset(str(path)) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        arrays = {k: np.asarray(v[:], dtype=np.float64) for k, v in ds.variables.items()}
    return attrs, arrays


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# DYCOMS-II RF01 (Stevens et al. 2005, Mon. Wea. Rev. 133, 1443).

DYCOMS_F0 = 70.0
DYCOMS_F1 = 22.0
DYCOMS_KAPPA = 85.0
DYCOMS_ALPHA = 1.0
DYCOMS_DIVERGENCE = 3.75e-6
DYCOMS_CP = 1015.0
DYCOMS_ZI = 840.0
#: The inversion height the radiation formula reads is the height of the
#: 8 g/kg total-water isoline (Stevens et al. 2005).
DYCOMS_QT_ISOLINE = 8.0e-3
#: The height above which the formula's flux is held constant, a harness
#: choice: the formula describes the boundary layer and the free
#: troposphere just above the 840 m inversion, and its subsidence term
#: grows as z^(4/3) without bound.
DYCOMS_FORMULA_TOP_M = 1500.0


def dycoms_net_flux(z_half, liquid_layer_kg_m2, z_inversion, rho_inversion):
    """Net upward longwave flux (W/m2) on the half levels of the DYCOMS-II
    RF01 formula, top to bottom.  ``liquid_layer_kg_m2`` is each layer's
    liquid water path (``nlev``), ``z_half`` the half-level heights
    (``nlev + 1``)."""
    z_half = np.asarray(z_half, dtype=np.float64)
    layer = np.asarray(liquid_layer_kg_m2, dtype=np.float64)
    above = np.concatenate([[0.0], np.cumsum(layer)])          # path above half i
    below = layer.sum() - above                                 # path below half i
    flux = DYCOMS_F0 * np.exp(-DYCOMS_KAPPA * above) + DYCOMS_F1 * np.exp(-DYCOMS_KAPPA * below)
    dz = np.maximum(z_half - z_inversion, 0.0)
    flux = flux + np.where(
        z_half > z_inversion,
        rho_inversion * DYCOMS_CP * DYCOMS_DIVERGENCE * DYCOMS_ALPHA
        * (0.25 * dz ** (4.0 / 3.0) + z_inversion * dz ** (1.0 / 3.0)),
        0.0,
    )
    return flux


def _dycoms_radiation(column) -> np.ndarray:
    """Theta tendency (K/s) of the DYCOMS-II RF01 radiation on one column."""
    qt = column["qv"] + column["qc"]
    z = column["z_full"]
    # Lowest crossing of the 8 g/kg isoline going up (the arrays are top to
    # bottom, so walk from the surface).
    zi = DYCOMS_ZI
    for k in range(len(z) - 1, 0, -1):
        if qt[k] >= DYCOMS_QT_ISOLINE > qt[k - 1]:
            frac = (qt[k] - DYCOMS_QT_ISOLINE) / (qt[k] - qt[k - 1])
            zi = float(z[k] + frac * (z[k - 1] - z[k]))
            break
    rho = column["p_full"] / (RD * column["virtual_temperature"])
    rho_i = float(np.interp(zi, z[::-1], rho[::-1]))
    layer = column["qc"] * column["dp"] / GRAVITY
    # The formula describes the LES domain; above its top the flux is held
    # (no heating), where its subsidence term would otherwise grow without
    # bound into the stratosphere the model column carries.
    flux = dycoms_net_flux(np.minimum(column["z_half"], DYCOMS_FORMULA_TOP_M), layer, zi, rho_i)
    heating = -GRAVITY * (flux[:-1] - flux[1:]) / (DYCOMS_CP * column["dp"])
    return heating / column["exner"]


def dycoms_rf01() -> SCMCase:
    z = np.array([0.0, DYCOMS_ZI, DYCOMS_ZI + 1.0e-3, 1000.0, 1200.0, 1500.0, 2000.0, 3000.0])
    thetal = np.where(z <= DYCOMS_ZI, 289.0, 297.5 + np.maximum(z - DYCOMS_ZI, 0.0) ** (1.0 / 3.0))
    qt = np.where(z <= DYCOMS_ZI, 9.0e-3, 1.5e-3)
    zw = np.array([0.0, 3000.0])
    return SCMCase(
        name="dycoms-rf01",
        title="DYCOMS-II RF01 nocturnal marine stratocumulus",
        reference="Stevens et al. 2005, Mon. Wea. Rev. 133, 1443-1462",
        source="analytic specification of Stevens et al. (2005), transcribed in this file",
        latitude_deg=31.5,
        longitude_deg=-121.5,
        # Local night (the case is nocturnal; RRTMGP sees no sun).
        start_utc=datetime(2001, 7, 11, 7, 0, 0, tzinfo=timezone.utc),
        duration_s=4 * 3600.0,
        window_s=(3 * 3600.0, 4 * 3600.0),
        surface_pressure_pa=101_780.0,
        u0=Profile.constant([0.0, 3000.0], [7.0, 7.0]),
        v0=Profile.constant([0.0, 3000.0], [-5.5, -5.5]),
        temperature_kind="thetal",
        temperature0=Profile.constant(z, thetal),
        water_kind="qt",
        water0=Profile.constant(z, qt),
        radiation="formula",
        radiation_formula=_dycoms_radiation,
        surface_type="ocean",
        ug=Profile.constant([0.0, 3000.0], [7.0, 7.0]),
        vg=Profile.constant([0.0, 3000.0], [-5.5, -5.5]),
        # w = -D z up to the case top; above it the harness's zero.
        w=Profile.constant(zw, -DYCOMS_DIVERGENCE * zw, above="zero"),
        skin_kind="ts",
        skin=Series.constant(292.5),
        sensible_w_m2=Series.constant(15.0),
        latent_w_m2=Series.constant(115.0),
        departures=[
            "liquid-water potential temperature and total-water forcing applied "
            "to theta and vapour",
            "radiation formula's inversion density read from the column at the "
            "diagnosed 8 g/kg isoline",
            "radiation formula's flux held constant above 1500 m (no "
            "formula heating there)",
        ],
    )


#: The cases, the DEPHY file each is read from, its length and its scoring
#: window (seconds from the start).  Lengths and windows follow the cases'
#: intercomparison papers where they differ from the DEPHY file's end date.
CASES: dict[str, dict[str, object]] = {
    "dycoms-rf01": {"builder": dycoms_rf01},
    "bomex": {"dephy": "BOMEX_REF", "duration_s": 6 * 3600.0, "window_s": (3 * 3600.0, 6 * 3600.0)},
    "rico": {"dephy": "RICO_SHORT", "duration_s": 24 * 3600.0, "window_s": (20 * 3600.0, 24 * 3600.0)},
    "fire": {"dephy": "FIRE_REF", "duration_s": 48 * 3600.0, "window_s": (24 * 3600.0, 48 * 3600.0)},
    "armcu": {"dephy": "ARMCU_REF", "duration_s": None, "window_s": None},
    "lba": {"dephy": "LBA_REF", "duration_s": 6 * 3600.0, "window_s": (5 * 3600.0, 6 * 3600.0)},
    "gabls1": {"dephy": "GABLS1_REF", "duration_s": 9 * 3600.0, "window_s": (8 * 3600.0, 9 * 3600.0)},
}


#: The DEPHY-SCM commit the bench's readings were taken on, and each case
#: file's place in that repository and its sha256 there.  ``fetch`` takes
#: exactly these bytes; a run on any other file still runs and records the
#: file's own sha256 in its result.
DEPHY_REPOSITORY = "https://raw.githubusercontent.com/GdR-DEPHY/DEPHY-SCM"
DEPHY_COMMIT = "661b0328acd253f345715534986e65724c8919db"
DEPHY_FILES: dict[str, tuple[str, str]] = {
    "ARMCU_REF": ("ARMCU/REF", "cfaa32ab693125f0ca7cbb24bb5b1fa5f922920d27886d135cc4f64de792f740"),
    "BOMEX_REF": ("BOMEX/REF", "fdf2f4e89b79e23adc1d1a5e75192d03d480d616691b25ed029e8cc54c2c3ec9"),
    "FIRE_REF": ("FIRE/REF", "340cda1aae07dc772b74fc0c7cf4441849cf1a9e718c12a15818bb9ddbd74d91"),
    "GABLS1_REF": ("GABLS1/REF", "1b38d239476a5e546f8a2f4c7f41a3293222c5e6ce77733d82ee6eaabec84241"),
    "LBA_REF": ("LBA/REF", "fc61ad0905f15b10b9d7088523f57d303aced25145fc25d65a1f9e0979e3aa33"),
    "RICO_SHORT": ("RICO/SHORT", "d76e4de3df26c65100d2343ac8414f5b3283dabe5fe02915f0ecf82b1620749b"),
}


def dephy_path(dephy_root: str | Path, dephy: str) -> Path:
    """Where a case's ``*_DEF_driver.nc`` sits under ``dephy_root``.

    Two layouts are read: the one ``fetch`` writes (``BOMEX_REF/``) and a
    checkout of the DEPHY-SCM repository itself (``BOMEX/REF/``).
    """

    name = f"{dephy}_DEF_driver.nc"
    flat = Path(dephy_root) / dephy / name
    if flat.is_file() or dephy not in DEPHY_FILES:
        return flat
    nested = Path(dephy_root) / DEPHY_FILES[dephy][0] / name
    return nested if nested.is_file() else flat


def fetch_dephy(out: str | Path, *, opener: Callable | None = None) -> list[Path]:
    """Download the six DEPHY case files at :data:`DEPHY_COMMIT` into ``out``.

    Each file is checked against its pinned sha256 before it is kept; a
    file whose bytes differ is removed and the fetch refuses by name.  A
    file already present with the right bytes is not downloaded again.
    """

    import urllib.request

    opener = opener or (lambda url: urllib.request.urlopen(url, timeout=120).read())
    written = []
    for dephy, (place, digest) in sorted(DEPHY_FILES.items()):
        path = Path(out) / dephy / f"{dephy}_DEF_driver.nc"
        if path.is_file() and _sha256(path) == digest:
            written.append(path)
            continue
        url = f"{DEPHY_REPOSITORY}/{DEPHY_COMMIT}/{place}/{dephy}_DEF_driver.nc"
        data = opener(url)
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            raise ValueError(f"{url}: sha256 {got}, expected {digest}; nothing written")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(path)
    return written


def load_case(name: str, dephy_root: str | Path | None = None) -> SCMCase:
    spec = CASES[name]
    if "builder" in spec:
        return spec["builder"]()
    if dephy_root is None:
        raise ValueError(f"case {name!r} reads a DEPHY file: pass --dephy-root "
                         "(the `fetch` command downloads them)")
    path = dephy_path(dephy_root, str(spec["dephy"]))
    attrs, arrays = read_netcdf(path)
    case = case_from_dephy(
        attrs, arrays, name=name,
        source=f"DEPHY-SCM {attrs.get('case', spec['dephy'])} {path.name} sha256 {_sha256(path)}",
        duration_s=spec["duration_s"], window_s=spec["window_s"])
    if spec["window_s"] is None:
        case.window_s = (case.duration_s - 3600.0, case.duration_s)
    return case


# --------------------------------------------------------------------------
# Thermodynamics of the initial column.


def saturation_specific_humidity(temperature, pressure):
    """Tetens/Bolton saturation over liquid, as specific humidity."""
    t = np.asarray(temperature, dtype=np.float64)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    eps = RD / RV
    return eps * es / np.maximum(pressure - (1.0 - eps) * es, 1.0)


def saturation_specific_humidity_ice(temperature, pressure):
    """Magnus saturation over ice (Alduchov and Eskridge 1996), as specific
    humidity; used only to bound the vapour of the column above a case."""
    celsius = np.asarray(temperature, dtype=np.float64) - 273.15
    es = 611.21 * np.exp(22.587 * celsius / (celsius + 273.86))
    eps = RD / RV
    return eps * es / np.maximum(pressure - (1.0 - eps) * es, 1.0)


def saturation_adjusted_temperature(t_liquid, total_water, pressure, *, liquid_temperature=True):
    """The temperature T with T = t_liquid + (Lv/cp) max(qt - qs(T, p), 0)
    (``liquid_temperature``), by bisection: the left side minus the right is
    increasing in T, so the root is bracketed by t_liquid and t_liquid +
    (Lv/cp) qt.  Without ``liquid_temperature`` the temperature is the
    case's and only the split changes."""
    t_liquid = np.asarray(t_liquid, np.float64)
    if not liquid_temperature:
        return t_liquid.copy()
    qt = np.asarray(total_water, np.float64)
    low = t_liquid.copy()
    high = t_liquid + LV / CP * qt
    for _ in range(60):
        mid = 0.5 * (low + high)
        excess = mid - t_liquid - LV / CP * np.maximum(
            qt - saturation_specific_humidity(mid, pressure), 0.0)
        high = np.where(excess > 0.0, mid, high)
        low = np.where(excess > 0.0, low, mid)
    return 0.5 * (low + high)


def _standard_atmosphere_temperature(pressure_pa):
    from arwen_global.vertical import standard_atmosphere_temperature_k

    return np.asarray(standard_atmosphere_temperature_k(np.asarray(pressure_pa, np.float64)), np.float64)


def initial_column(case: SCMCase, coordinate, iterations: int = 6) -> dict[str, np.ndarray]:
    """The case's initial column on the hybrid levels, top to bottom:
    temperature, specific vapour and cloud liquid, winds, and the level
    pressures and heights.  Heights come from the coordinate's own
    hydrostatic integral of the column's virtual temperature, iterated."""
    a = np.asarray(coordinate.a_half_pa, np.float64)
    b = np.asarray(coordinate.b_half, np.float64)
    p_half = a + b * case.surface_pressure_pa
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    exner = (p_full / P0) ** KAPPA_DRY
    nlev = p_full.size
    temperature = _standard_atmosphere_temperature(p_full)
    qv = np.zeros(nlev)
    ql = np.zeros(nlev)
    z_top = case.temperature0.top_m
    water_top = case.water0.top_m
    for _ in range(iterations):
        tv = temperature * (1.0 + 0.61 * qv - ql)
        z_full, _z_half = column_heights(coordinate, tv, p_half)
        raw = case.temperature0.at(0.0, z_full)
        water = case.water0.at(0.0, z_full)
        if case.water_kind in ("rv", "rt"):
            water = water / (1.0 + water)
        if case.temperature_kind == "ta":
            t_case = raw
        elif case.temperature_kind == "theta":
            t_case = raw * exner
        else:
            t_case = raw * exner
        qv_case = water.copy()
        ql_case = np.zeros(nlev)
        if case.water_kind in ("qt", "rt"):
            # Total water: split by saturation adjustment at the case's
            # temperature (thetal: T = thetal exner + Lv/cp ql).
            t_adj = saturation_adjusted_temperature(
                t_case, water, p_full, liquid_temperature=case.temperature_kind == "thetal")
            ql_case = np.maximum(water - saturation_specific_humidity(t_adj, p_full), 0.0)
            qv_case = water - ql_case
            t_case = t_adj
        # Above the case's top: standard atmosphere joined with a decaying offset.
        t_std = _standard_atmosphere_temperature(p_full)
        zt = float(z_top)
        k_top = np.argmin(np.abs(z_full - zt))
        offset = float(t_case[k_top] - t_std[k_top])
        joined = t_std + offset * np.exp(-np.maximum(z_full - zt, 0.0) / UPPER_JOIN_EFOLD_M)
        temperature = np.where(z_full <= zt, t_case, joined)
        q_top = float(np.interp(water_top, z_full[::-1], qv_case[::-1]))
        upper_q = np.maximum(q_top * np.exp(-np.maximum(z_full - water_top, 0.0) / UPPER_VAPOUR_EFOLD_M),
                             UPPER_VAPOUR_FLOOR)
        # Never more than UPPER_RELATIVE_HUMIDITY of ice saturation, so the
        # joined column carries no cloud of its own.
        upper_q = np.minimum(upper_q, np.maximum(
            UPPER_RELATIVE_HUMIDITY * saturation_specific_humidity_ice(temperature, p_full),
            1.0e-7))
        qv = np.where(z_full <= water_top, np.maximum(qv_case, 0.0), upper_q)
        ql = np.where(z_full <= water_top, ql_case, 0.0)
    tv = temperature * (1.0 + 0.61 * qv - ql)
    z_full, z_half = column_heights(coordinate, tv, p_half)
    return {
        "p_half": p_half, "p_full": p_full, "dp": np.diff(p_half), "exner": exner,
        "temperature": temperature, "qv": qv, "qc": ql,
        "u": case.u0.at(0.0, z_full), "v": case.v0.at(0.0, z_full),
        "z_full": z_full, "z_half": z_half,
    }


def column_heights(coordinate, virtual_temperature, p_half):
    """Full- and half-level heights (m above the surface) of one column,
    from the coordinate's hydrostatic geopotential (full levels) and the
    layer integral R Tv ln(p_lower / p_upper) (half levels)."""
    tv = np.asarray(virtual_temperature, np.float64)
    ph = np.asarray(p_half, np.float64)
    phi = coordinate.hydrostatic_geopotential(
        tv[:, None, None], np.zeros((1, 1)), ph[:, None, None], gas_constant=RD)
    z_full = np.asarray(phi, np.float64)[:, 0, 0] / GRAVITY
    thickness = RD * tv * np.log(ph[1:] / ph[:-1]) / GRAVITY
    z_half = np.concatenate([np.cumsum(thickness[::-1])[::-1], [0.0]])
    return z_full, z_half


# --------------------------------------------------------------------------
# The large-scale forcing between the physics half-steps.


def subsidence_tendency(field_values, z_full, w):
    """-w d(field)/dz, first-order upwind in height, top-to-bottom arrays.
    ``field_values`` is ``(nlev, ...)``; ``z_full`` and ``w`` are ``(nlev,)``
    or broadcastable.  Zero at a level whose upwind neighbour is outside the
    column."""
    xp = _xp(field_values)
    f = field_values
    z = xp.asarray(z_full).reshape((-1,) + (1,) * (f.ndim - 1))
    ww = xp.asarray(w).reshape((-1,) + (1,) * (f.ndim - 1))
    up = xp.zeros_like(f)      # (f[k-1] - f[k]) / (z[k-1] - z[k]): from above
    down = xp.zeros_like(f)    # (f[k] - f[k+1]) / (z[k] - z[k+1]): from below
    up[1:] = (f[:-1] - f[1:]) / (z[:-1] - z[1:])
    down[:-1] = (f[:-1] - f[1:]) / (z[:-1] - z[1:])
    gradient = xp.where(ww < 0.0, up, down)
    return -ww * gradient


def rotate_toward_geostrophic(u, v, ug, vg, coriolis_s, dt_s):
    """Exact integration of du/dt = f (v - vg), dv/dt = -f (u - ug) over dt
    with the geostrophic wind held."""
    angle = coriolis_s * dt_s
    c, s = math.cos(angle), math.sin(angle)
    du = u - ug
    dv = v - vg
    return ug + c * du + s * dv, vg - s * du + c * dv


def _xp(value):
    if type(value).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415

        return cupy
    return np


# --------------------------------------------------------------------------
# Module substitutions for the suite's ``modules`` door.


class _Clock:
    """What the substitutions read: the case and the time of the call."""

    def __init__(self, case: SCMCase):
        self.case = case
        self.time_s = 0.0


def prescribed_sfclay(real_module, clock: _Clock):
    """The shipped surface layer with the case's surface written over it:
    on a prescribed-roughness case the roughness goes in and comes back
    unchanged; on a land case the land branch runs (xland 1, the case's
    moisture availability); prescribed fluxes and friction velocity replace
    the scheme's own on the way out."""

    def sfclay(*args, **kwargs):
        case = clock.case
        t = clock.time_s
        args = list(args)
        xp = _xp(args[0])
        if case.z0_m is not None:
            # Positional 9: znt.
            args[8] = xp.full_like(args[8], case.z0_m.at(t))
        if case.surface_type != "ocean":
            args[11] = xp.ones_like(args[11])          # xland: land branch
            if case.moisture_availability is not None:
                args[10] = xp.full_like(args[10], case.moisture_availability.at(t))
        result = real_module.sfclay(*args, **kwargs)
        if case.z0_m is not None:
            result.znt = xp.full_like(result.znt, case.z0_m.at(t))
        if case.sensible_w_m2 is not None:
            result.hfx = xp.full_like(result.hfx, case.sensible_w_m2.at(t))
        if case.latent_w_m2 is not None:
            result.qfx = xp.full_like(result.qfx, case.latent_w_m2.at(t) / LV)
        if case.ustar_m_s is not None:
            result.ust = xp.full_like(result.ust, case.ustar_m_s.at(t))
        return result

    return SimpleNamespace(
        sfclay=sfclay,
        __name__="tools.arwen_global_scm_cases.prescribed_sfclay",
        **{k: getattr(real_module, k) for k in dir(real_module)
           if not k.startswith("_") and k != "sfclay"},
    )


class _HeatingDropped:
    """A radiation result whose heating is zero and whose fluxes are the
    scheme's: the spec arm of a case that prescribes its own radiation."""

    def __init__(self, result, xp):
        self._result = result
        self.rthratenlw = xp.zeros_like(result.rthratenlw)
        self.rthratensw = xp.zeros_like(result.rthratensw)

    def __getattr__(self, name):
        return getattr(self._result, name)


def diagnostic_rrtmgp(real_module):
    """The shipped RRTMGP with its heating dropped (fluxes kept)."""

    held = {}

    class RRTMGPRadiation:
        def __init__(self, **kwargs):
            self._driver = real_module.RRTMGPRadiation(**kwargs)

        def __call__(self, **kwargs):
            result = self._driver(**kwargs)
            # The rates the scheme computed, kept for the readings (kernel
            # order, theta tendency K/s); the suite applies zeros.
            held["rad_rthratenlw"] = result.rthratenlw.copy()
            held["rad_rthratensw"] = result.rthratensw.copy()
            return _HeatingDropped(result, _xp(result.rthratenlw))

        def __getattr__(self, name):
            return getattr(self._driver, name)

    return SimpleNamespace(
        RRTMGPRadiation=RRTMGPRadiation,
        held_rates=held,
        __name__="tools.arwen_global_scm_cases.diagnostic_rrtmgp",
        **{k: getattr(real_module, k) for k in dir(real_module)
           if not k.startswith("_") and k != "RRTMGPRadiation"},
    )


# --------------------------------------------------------------------------
# The run.

#: The shipped configuration whose vertical stack and physics options the
#: bench runs (the 52 km forecast's).
PRODUCTION_CONFIG = "arwen_global_gdas_t255_native_sl_si_24h.toml"
#: The T255 Gaussian grid's ring length and (mean) ring spacing, for a
#: cumulus scheme that reads its grid length from the batch's rings.
T255_RING_COLUMNS = 768
T255_RING_SPACING_DEG = 180.0 / 384.0
#: The tiny shipped configuration whose cold state lends the bench a valid
#: water column's surface statics and physics-state metadata.
TEMPLATE_CONFIG = "arwen_global_moist_smoke.toml"


def production_setup(start_utc: datetime, *, dx_m: float | None = None):
    """The production vertical coordinate, step and native options, with
    the radiation clock set to ``start_utc``."""
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.vertical import HybridCoordinate

    cfg = load_config(config_root() / PRODUCTION_CONFIG)
    options = dict(cfg.native_adapter_options)
    options["start_time_utc"] = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if dx_m is not None:
        options["dx_m"] = float(dx_m)
    coordinate = HybridCoordinate(np.asarray(cfg.a_half_pa), np.asarray(cfg.b_half))
    return coordinate, float(cfg.dt_s), options


def template_water_column():
    """Surface statics and physics-state metadata of one water column of
    the tiny shipped configuration's cold state (numpy, no card)."""
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.runner import build_model_and_cold_state
    from arwen_global.statics import water_columns

    cfg = load_config(config_root() / TEMPLATE_CONFIG)
    _model, state = build_model_and_cold_state(cfg)
    surface = state.surface
    water = np.asarray(water_columns(np.asarray(surface.land_fraction)))
    if not water.any():
        raise RuntimeError("the template configuration has no water column")
    j, i = (int(v[0]) for v in np.nonzero(water))
    return surface, state.physics_state, (j, i)


@dataclass
class Bench:
    """One case on ``ncol`` identical columns, stepped through a suite."""

    case: SCMCase
    coordinate: object
    dt_s: float
    xp: object
    dtype: object
    ncol: int = 2
    arm: str = "spec"
    #: Rows of the batch.  One row of ``ncol`` columns at the case latitude
    #: by default; two rows (``ring_spacing_deg`` apart, centred on the case
    #: latitude) for a scheme that takes its grid length from the batch's
    #: latitude rings (New Tiedtke): with ``ncol`` the forecast's ring
    #: length, every column then has the forecast's spacing.
    nrow: int = 1
    ring_spacing_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.arm not in ("spec", "native-radiation"):
            raise ValueError(f"arm {self.arm!r}")
        if self.arm == "native-radiation" and self.case.radiation not in ("tend", "formula", "on"):
            raise ValueError(
                f"{self.case.name}: radiation={self.case.radiation!r} folds the "
                "radiation into the advective forcing or leaves it out; the "
                "native-radiation arm would count it twice or add what the case "
                "omits")
        column = initial_column(self.case, self.coordinate)
        self.column0 = column
        xp, dt = self.xp, self.dtype
        shape = (self.nrow, self.ncol)

        def volume(values):
            return xp.ascontiguousarray(xp.broadcast_to(
                xp.asarray(np.asarray(values)[:, None, None], dtype=dt),
                (len(values), *shape)))

        self.p_half = volume(column["p_half"])
        self.p_full = volume(column["p_full"])
        self.dp = volume(column["dp"])
        self.exner = volume(column["exner"])
        self.state = {
            "u": volume(column["u"]), "v": volume(column["v"]),
            "theta": volume(column["temperature"] / column["exner"]),
            "qv": volume(column["qv"]), "qc": volume(column["qc"]),
        }
        zeros = volume(np.zeros_like(column["qv"]))
        for name in ("qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"):
            self.state[name] = zeros.copy()
        offsets = (np.arange(self.nrow) - 0.5 * (self.nrow - 1)) * self.ring_spacing_deg
        self.latitude = xp.ascontiguousarray(xp.broadcast_to(
            xp.asarray((self.case.latitude_deg + offsets)[:, None], dtype=dt), shape))
        lon = self.case.longitude_deg % 360.0
        self.longitude = xp.full(shape, lon, dtype=dt)
        self.surface = None
        self.physics_state = None
        self.time_s = 0.0
        #: RRTMGP's own rates when the arm drops them (diagnostic_rrtmgp).
        self.held_rates = None

    # ---- surface and namespace -------------------------------------------------

    def seed_surface(self, template_surface, template_physics_state, column) -> None:
        from arwen_global.state import PhysicsState, SURFACE_ARRAY_NAMES, SurfaceState

        j, i = column
        xp, dt = self.xp, self.dtype
        shape = (self.nrow, self.ncol)

        def take(value):
            host = np.asarray(value)
            point = host[..., j, i]
            return xp.ascontiguousarray(xp.broadcast_to(
                xp.asarray(point[..., None, None], dtype=dt), (*point.shape, *shape)))

        members = {member: take(getattr(template_surface, member))
                   for member in SURFACE_ARRAY_NAMES}
        members["land_fraction"] = xp.zeros(shape, dtype=dt)
        members["sea_ice_fraction"] = xp.zeros(shape, dtype=dt)
        members["sea_ice_thickness_m"] = xp.zeros(shape, dtype=dt)
        members["lake_fraction"] = xp.zeros(shape, dtype=dt)
        self.surface = SurfaceState(**members)
        self.physics_state = PhysicsState(
            schema=template_physics_state.schema,
            arrays={name: take(value) for name, value in template_physics_state.arrays.items()},
            metadata=json.loads(json.dumps(template_physics_state.metadata)),
        )
        self._apply_surface(0.0)

    def _apply_surface(self, t: float) -> None:
        case = self.case
        xp, dt = self.xp, self.dtype
        if case.skin is not None:
            value = case.skin.at(t)
            if case.skin_kind == "thetas":
                value = value * (case.surface_pressure_pa / P0) ** KAPPA_DRY
            self.surface.temperature_k[...] = xp.asarray(value, dtype=dt)
            # The skin the soil layers carry follows it (a water column's
            # soil is never integrated; this keeps the template's consistent).
        elif case.sensible_w_m2 is not None:
            self.surface.temperature_k[...] = xp.asarray(self._flux_skin(t), dtype=dt)
        if case.z0_m is not None:
            self.surface.roughness_m[...] = xp.asarray(case.z0_m.at(t), dtype=dt)

    def _flux_skin(self, t: float) -> float:
        """The skin temperature a prescribed sensible heat flux implies
        through the surface layer's own heat exchange coefficient (``chs``,
        m/s, from the previous call; 0.01 m/s before the first):
        theta_skin = theta_lowest + H / (rho cp chs), within 20 K of the
        lowest level.  A case that prescribes its fluxes and no skin would
        otherwise hand sfclay and YSU a skin whose stability contradicts the
        flux they are given (a cold skin under an upward flux is a stable
        surface layer that mixes nothing out of the lowest level)."""
        theta1 = float(_host(self.state["theta"][-1, 0, 0]))
        exner1 = float(_host(self.exner[-1, 0, 0]))
        p_surface = float(_host(self.p_half[-1, 0, 0]))
        chs = 0.0
        arrays = self.physics_state.arrays if self.physics_state is not None else {}
        if "chs" in arrays:
            chs = float(_host(arrays["chs"])[0, 0])
        if not chs > 1.0e-4:
            chs = 0.01
        rho = p_surface / (RD * theta1 * exner1)
        theta_skin = theta1 + self.case.sensible_w_m2.at(t) / (rho * CP * chs)
        skin = theta_skin * (p_surface / P0) ** KAPPA_DRY
        lowest = theta1 * exner1
        return float(min(max(skin, lowest - 20.0), lowest + 20.0))

    # ---- exchange ---------------------------------------------------------------

    def _derived(self):
        xp = self.xp
        s = self.state
        temperature = s["theta"] * self.exner
        condensate = s["qc"] + s["qr"] + s["qi"] + s["qs"] + s["qg"]
        tv = temperature * (1.0 + 0.61 * s["qv"] - condensate)
        phis = xp.full(self.p_half.shape[1:], self.case.orography_m * GRAVITY, dtype=self.dtype)
        geopotential = self.coordinate.hydrostatic_geopotential(tv, phis, self.p_half)
        return temperature, tv, xp.asarray(geopotential, dtype=self.dtype)

    def _omega(self, t: float, z_half):
        xp, dt = self.xp, self.dtype
        nlev1 = self.p_half.shape[0]
        if self.case.w is None:
            return xp.zeros((nlev1, self.nrow, self.ncol), dtype=dt)
        w = self.case.w.at(t, z_half)
        temperature, tv, _ = self._derived()
        tv_host = np.asarray(_host(tv[:, 0, 0]), np.float64)
        tv_half = np.concatenate([[tv_host[0]], 0.5 * (tv_host[:-1] + tv_host[1:]), [tv_host[-1]]])
        p_half = np.asarray(_host(self.p_half[:, 0, 0]), np.float64)
        omega = -(p_half / (RD * tv_half)) * GRAVITY * w
        omega[0] = 0.0
        omega[-1] = 0.0
        return xp.ascontiguousarray(xp.broadcast_to(
            xp.asarray(omega[:, None, None], dtype=dt), (nlev1, self.nrow, self.ncol)))

    def heights(self):
        _t, tv, _g = self._derived()
        tv_host = np.asarray(_host(tv[:, 0, 0]), np.float64)
        return column_heights(self.coordinate, tv_host, np.asarray(_host(self.p_half[:, 0, 0]), np.float64))

    def exchange(self, time_s: float, dt_s: float):
        from arwen_global.physics.exchange import PhysicsExchange

        temperature, tv, geopotential = self._derived()
        _z_full, z_half = self.heights()
        s = self.state
        return PhysicsExchange.create(
            time_s=float(time_s), dt_s=float(dt_s),
            latitude_deg=self.latitude, longitude_deg=self.longitude,
            p_half=self.p_half, p_full=self.p_full, dp=self.dp, exner=self.exner,
            temperature=temperature, theta=s["theta"], virtual_temperature=tv,
            geopotential=geopotential, u=s["u"], v=s["v"],
            qv=s["qv"], qc=s["qc"], qr=s["qr"], qi=s["qi"], qs=s["qs"], qg=s["qg"],
            nc=s["nc"], nr=s["nr"], ni=s["ni"], ns=s["ns"], ng=s["ng"],
            surface=self.surface, physics_state=self.physics_state,
            omega_half_pa_s=self._omega(time_s, z_half),
        )

    def physics(self, suite, clock: _Clock, time_s: float, dt_s: float):
        self._apply_surface(time_s)
        clock.time_s = float(time_s)
        result = suite.step(self.exchange(time_s, dt_s))
        for name in self.state:
            self.state[name] = self.xp.ascontiguousarray(
                self.xp.asarray(getattr(result, name), dtype=self.dtype))
        self.surface = result.surface
        self.physics_state = result.physics_state
        return result

    # ---- forcing ----------------------------------------------------------------

    def column_host(self) -> dict[str, np.ndarray]:
        temperature, tv, _ = self._derived()
        z_full, z_half = self.heights()
        out = {name: np.asarray(_host(value[:, 0, 0]), np.float64) for name, value in self.state.items()}
        out.update(
            temperature=np.asarray(_host(temperature[:, 0, 0]), np.float64),
            virtual_temperature=np.asarray(_host(tv[:, 0, 0]), np.float64),
            p_full=np.asarray(_host(self.p_full[:, 0, 0]), np.float64),
            p_half=np.asarray(_host(self.p_half[:, 0, 0]), np.float64),
            dp=np.asarray(_host(self.dp[:, 0, 0]), np.float64),
            exner=np.asarray(_host(self.exner[:, 0, 0]), np.float64),
            z_full=z_full, z_half=z_half,
        )
        return out

    def forcing(self, t: float, dt_s: float) -> None:
        """Advance the state by the case's large-scale forcing over ``dt_s``,
        every term read at time ``t``."""
        case = self.case
        xp, dt = self.xp, self.dtype
        s = self.state
        z_full, _z_half = self.heights()
        exner = np.asarray(_host(self.exner[:, 0, 0]), np.float64)

        def vol(values):
            return xp.asarray(np.asarray(values, np.float64)[:, None, None], dtype=dt)

        theta_rate = np.zeros_like(z_full)
        if case.t_adv is not None:
            rate = case.t_adv.at(t, z_full)
            theta_rate += rate / exner if case.t_adv_kind == "ta" else rate
        if self.arm == "spec":
            if case.radiation == "tend" and case.t_rad is not None:
                rate = case.t_rad.at(t, z_full)
                theta_rate += rate / exner if case.t_rad_kind == "ta" else rate
            elif case.radiation == "formula":
                theta_rate += case.radiation_formula(self.column_host())
        q_rate = np.zeros_like(z_full)
        if case.q_adv is not None:
            rate = case.q_adv.at(t, z_full)
            if case.q_adv_kind in ("rv", "rt"):
                qv = np.asarray(_host(s["qv"][:, 0, 0]), np.float64)
                rate = rate * (1.0 - qv) ** 2
            q_rate += rate
        tendencies = {"theta": vol(theta_rate), "qv": vol(q_rate)}
        if case.w is not None:
            w = case.w.at(t, z_full)
            for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg"):
                sub = subsidence_tendency(s[name], z_full, w)
                tendencies[name] = tendencies.get(name, 0.0) + xp.asarray(sub, dtype=dt)
        for name, rate in tendencies.items():
            s[name] = xp.ascontiguousarray(s[name] + xp.asarray(dt_s, dtype=dt) * rate)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
            s[name] = xp.maximum(s[name], 0.0)
        if case.ug is not None or case.vg is not None:
            ug = vol(case.ug.at(t, z_full) if case.ug is not None else np.zeros_like(z_full))
            vg = vol(case.vg.at(t, z_full) if case.vg is not None else np.zeros_like(z_full))
            u, v = rotate_toward_geostrophic(s["u"], s["v"], ug, vg, case.coriolis_s, dt_s)
            s["u"], s["v"] = xp.ascontiguousarray(u), xp.ascontiguousarray(v)
        for axis in ("u", "v"):
            target = getattr(case, f"nudge_{axis}")
            if target is not None:
                tau = float(getattr(case, f"nudge_{axis}_tau_s"))
                weight = 1.0 - math.exp(-dt_s / tau)
                s[axis] = xp.ascontiguousarray(
                    s[axis] + weight * (vol(target.at(t, z_full)) - s[axis]))

    # ---- one step and the run --------------------------------------------------

    def step(self, suite, clock) -> list:
        t0 = self.time_s
        half = 0.5 * self.dt_s
        results = []
        if suite is not None:
            results.append(self.physics(suite, clock, t0, half))
        self.forcing(t0 + half, self.dt_s)
        if suite is not None:
            results.append(self.physics(suite, clock, t0 + half, half))
        self.time_s = t0 + self.dt_s
        return results


def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


# --------------------------------------------------------------------------
# Readings.

#: Persistent planes read every step, under the names the result records.
_PLANES = {
    "hfx": "sensible_w_m2", "ust": "ustar_m_s", "pblh": "pbl_height_m",
    "swupt": "toa_sw_up_w_m2", "swdnt": "toa_sw_down_w_m2", "olr": "olr_w_m2",
    "gsw": "surface_sw_net_w_m2", "glw": "surface_lw_down_w_m2",
    "cldfra_total": "radiation_cloud_cover",
}


def _inversion_height(col, ceiling_m: float = 5000.0) -> float:
    """Height of the half level with the largest potential-temperature
    gradient below ``ceiling_m`` (the LES papers' inversion height, on the
    model's levels)."""
    z = col["z_full"]
    theta = col["theta"]
    gradient = (theta[:-1] - theta[1:]) / (z[:-1] - z[1:])
    middle = 0.5 * (z[:-1] + z[1:])
    below = middle < ceiling_m
    if not below.any():
        return float("nan")
    k = int(np.argmax(np.where(below, gradient, -np.inf)))
    return float(middle[k])


def _surface_wind_angle(bench: Bench, col) -> float:
    """Angle (degrees, positive for backing toward low pressure in the
    northern hemisphere) from the lowest level's wind to the case's
    geostrophic wind at that height; NaN for a case without one."""
    case = bench.case
    if case.ug is None and case.vg is None:
        return float("nan")
    z = col["z_full"][-1:]
    ug = float(case.ug.at(bench.time_s, z)[0]) if case.ug is not None else 0.0
    vg = float(case.vg.at(bench.time_s, z)[0]) if case.vg is not None else 0.0
    turn = math.atan2(col["v"][-1], col["u"][-1]) - math.atan2(vg, ug)
    turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(turn)) * (1.0 if case.latitude_deg >= 0.0 else -1.0)


#: The namespace's convective rain accumulator (arwen_global.constants
#: CONVECTIVE_RAIN_ACCUMULATOR).
CONVECTIVE_RAIN = "rainc"


def _rain_accumulators(bench: Bench) -> tuple[float, float]:
    arrays = bench.physics_state.arrays
    grid = float(_host(bench.surface.accumulated_rain_kg_m2)[0, 0])
    convective = (float(_host(arrays[CONVECTIVE_RAIN])[0, 0])
                  if CONVECTIVE_RAIN in arrays else 0.0)
    return grid, convective


def step_readings(bench: Bench, rain_before: tuple[float, float]) -> dict[str, float]:
    col = bench.column_host()
    dp_g = col["dp"] / GRAVITY
    z = col["z_full"]
    cloudy = (col["qc"] + col["qi"]) > CLOUD_THRESHOLD
    arrays = bench.physics_state.arrays
    out = {
        "time_s": bench.time_s,
        "lwp_g_m2": float(1000.0 * (col["qc"] * dp_g).sum()),
        "rwp_g_m2": float(1000.0 * (col["qr"] * dp_g).sum()),
        "iwp_g_m2": float(1000.0 * ((col["qi"] + col["qs"] + col["qg"]) * dp_g).sum()),
        "cloud_base_m": float(z[cloudy].min()) if cloudy.any() else float("nan"),
        "cloud_top_m": float(z[cloudy].max()) if cloudy.any() else float("nan"),
        "condensate_cloud_cover": float(cloudy.any()),
        "inversion_height_m": _inversion_height(col),
        "surface_wind_angle_deg": _surface_wind_angle(bench, col),
    }
    for key, label in _PLANES.items():
        if key in arrays:
            out[label] = float(_host(arrays[key])[0, 0])
    if "qfx" in arrays:
        out["latent_w_m2"] = float(_host(arrays["qfx"])[0, 0]) * LV
    # The boundary layer's radiative heating, W/m2, below 1500 m: RRTMGP's
    # held rates (kernel order, bottom to top) whether or not this arm
    # applies them, and the case formula's on the same column.
    low = z < 1500.0
    weight = CP * col["exner"] * dp_g
    rates = bench.held_rates if bench.held_rates else arrays
    for lane, label in (("rad_rthratenlw", "rrtmgp_lw"), ("rad_rthratensw", "rrtmgp_sw")):
        if lane in rates:
            rate = np.asarray(_host(rates[lane])[:, 0, 0], np.float64)[::-1]
            out[f"{label}_heating_below_1500m_w_m2"] = float((rate * weight)[low].sum())
    if bench.case.radiation == "formula":
        rate = bench.case.radiation_formula(col)
        out["formula_heating_below_1500m_w_m2"] = float((rate * weight)[low].sum())
    # Surface rain is the microphysics' (the surface accumulator) plus the
    # cumulus scheme's (the namespace's RAINC), each as its own rate too.
    grid = float(_host(bench.surface.accumulated_rain_kg_m2)[0, 0])
    convective = (float(_host(arrays[CONVECTIVE_RAIN])[0, 0])
                  if CONVECTIVE_RAIN in arrays else 0.0)
    per_day = 86400.0 / bench.dt_s
    out["grid_rain_mm_day"] = (grid - rain_before[0]) * per_day
    out["convective_rain_mm_day"] = (convective - rain_before[1]) * per_day
    out["surface_rain_mm_day"] = out["grid_rain_mm_day"] + out["convective_rain_mm_day"]
    out["accumulated_rain_kg_m2"] = grid + convective
    out["_rain_accumulators"] = (grid, convective)
    swdn = out.get("toa_sw_down_w_m2", 0.0)
    out["toa_albedo"] = (out.get("toa_sw_up_w_m2", float("nan")) / swdn
                         if swdn > 1.0 else float("nan"))
    theta = np.asarray(_host(bench.state["theta"][:, 0, :]))
    qc = np.asarray(_host(bench.state["qc"][:, 0, :]))
    out["column_spread_theta_k"] = float(np.max(np.abs(theta - theta[:, :1])))
    out["column_spread_qc"] = float(np.max(np.abs(qc - qc[:, :1])))
    return out


def window_means(series: list[dict], window: tuple[float, float]) -> dict[str, float]:
    rows = [r for r in series if window[0] < r["time_s"] <= window[1] + 1.0e-6]
    out = {}
    for key in rows[0] if rows else []:
        if key == "time_s":
            continue
        values = np.asarray([r[key] for r in rows], np.float64)
        finite = values[np.isfinite(values)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
    out["window_rows"] = len(rows)
    return out


def run_case(case: SCMCase, *, arm: str = "spec", ncol: int = 2, sun: str = "case",
             dx_m: float | None = None, xp=None, dtype=None, suite=None,
             modules=None, options=None, coordinate=None, dt_s=None,
             template=None, max_steps: int | None = None, log=None,
             option_overrides: Mapping[str, object] | None = None,
             noon_at_s: float | None = None) -> dict:
    """Run one case and return the result record (profiles and series as
    lists).  The defaults run the shipped suite on the card; a test passes
    ``xp=numpy``, ``options``, ``modules`` (fakes), ``coordinate`` and
    ``dt_s``.  ``suite=False`` runs the forcing alone."""
    if sun not in ("case", "noon"):
        raise ValueError(f"sun={sun!r}")
    radiation_heating_dropped = arm == "spec" and case.radiation in ("tend", "formula", "off")
    if sun == "noon" and not radiation_heating_dropped:
        raise ValueError(
            "sun='noon' moves the radiation clock; it is allowed only where the "
            "model's radiation heating is not applied (the spec arm of a case "
            "with prescribed or no radiation)")
    clock_start = case.start_utc
    if sun == "noon":
        mid = (0.5 * (case.window_s[0] + case.window_s[1]) if noon_at_s is None
               else float(noon_at_s))
        # Local solar noon at the case longitude, on the case's date.
        noon_utc = datetime(case.start_utc.year, case.start_utc.month, case.start_utc.day,
                            tzinfo=timezone.utc) + timedelta(hours=12.0 - case.longitude_deg / 15.0)
        clock_start = noon_utc - timedelta(seconds=mid)
    if coordinate is None or dt_s is None or options is None:
        prod_coordinate, prod_dt, prod_options = production_setup(clock_start, dx_m=dx_m)
        coordinate = coordinate or prod_coordinate
        dt_s = dt_s or prod_dt
        if options is None:
            options = prod_options
    if option_overrides:
        options = {**dict(options), **dict(option_overrides)}
    if xp is None:
        import cupy as xp  # noqa: PLC0415
    if dtype is None:
        dtype = xp.float32
    nrow, spacing = 1, 0.0
    if str(dict(options).get("cumulus", "gf")).lower() == "ntiedtke":
        # New Tiedtke reads each column's grid length from the batch's
        # latitude rings: two rings of the T255 grid's length and spacing.
        nrow, ncol, spacing = 2, T255_RING_COLUMNS, T255_RING_SPACING_DEG
    bench = Bench(case=case, coordinate=coordinate, dt_s=float(dt_s), xp=xp, dtype=dtype,
                  ncol=ncol, arm=arm, nrow=nrow, ring_spacing_deg=spacing)
    surface, physics_state, column = template or template_water_column()
    bench.seed_surface(surface, physics_state, column)
    clock = _Clock(case)
    if suite is None:
        from arwen_global.physics.native_suite import ArwenCudaColumnSuite

        import importlib  # noqa: PLC0415

        resolved = dict(modules or {})

        def base(name):
            return resolved.get(name) or importlib.import_module(name)

        resolved["arwen_global.core.sfclay"] = prescribed_sfclay(
            base("arwen_global.core.sfclay"), clock)
        if radiation_heating_dropped:
            resolved["arwen_global.core.rrtmgp"] = diagnostic_rrtmgp(
                base("arwen_global.core.rrtmgp"))
            bench.held_rates = resolved["arwen_global.core.rrtmgp"].held_rates
        suite = ArwenCudaColumnSuite(options, array_module=xp, modules=resolved)
    elif suite is False:
        suite = None
    steps = int(round(case.duration_s / bench.dt_s))
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    series = [dict(step_readings_initial(bench), time_s=0.0)] if suite is None else []
    profiles = {"time_s": [], "theta": [], "qv": [], "qc": [], "qr": [], "qi": [],
                "u": [], "v": []}
    started = time.perf_counter()
    rain_before = _rain_accumulators(bench)
    failure = None
    for n in range(steps):
        try:
            bench.step(suite, clock)
        except Exception as exc:  # noqa: BLE001 - the column up to here is the reading
            failure = {"step": n + 1, "time_s": bench.time_s + bench.dt_s,
                       "error": f"{type(exc).__name__}: {exc}",
                       "column_before": {k: v.tolist() for k, v in bench.column_host().items()}}
            if log is not None:
                log(f"{case.name} {arm}: FAILED at step {n + 1}: {failure['error']}")
            break
        if suite is not None:
            reading = step_readings(bench, rain_before)
            rain_before = reading.pop("_rain_accumulators")
            series.append(reading)
        col = bench.column_host()
        profiles["time_s"].append(bench.time_s)
        for name in ("theta", "qv", "qc", "qr", "qi", "u", "v"):
            profiles[name].append(col[name].tolist())
        if log is not None and (n + 1) % max(1, steps // 8) == 0:
            last = series[-1] if series else {}
            log(f"{case.name} {arm}: step {n + 1}/{steps} t={bench.time_s / 3600:.2f} h "
                f"lwp={last.get('lwp_g_m2', float('nan')):.1f} g/m2")
    elapsed = time.perf_counter() - started
    final = bench.column_host()
    return {
        "schema": RESULT_SCHEMA,
        "case": case.name,
        "title": case.title,
        "reference": case.reference,
        "source": case.source,
        "arm": arm,
        "sun": sun,
        "noon_at_s": noon_at_s,
        "radiation_clock_start_utc": clock_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "radiation_heating_applied": not radiation_heating_dropped,
        "case_radiation": case.radiation,
        "failure": failure,
        "columns": ncol,
        "rows": nrow,
        "ring_spacing_deg": spacing,
        "dt_s": bench.dt_s,
        "steps": steps,
        "steps_completed": len(profiles["time_s"]),
        "duration_s": steps * bench.dt_s,
        "window_s": list(case.window_s),
        "wall_s": elapsed,
        "options": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                    for k, v in dict(options).items()},
        "departures": list(case.departures),
        "levels": {"z_full_m_initial": bench.column0["z_full"].tolist(),
                   "p_full_pa": bench.column0["p_full"].tolist(),
                   "z_full_m_final": final["z_full"].tolist()},
        "initial": {k: bench.column0[k].tolist() for k in ("temperature", "qv", "qc", "u", "v")},
        "series": series,
        "window_means": window_means(series, case.window_s) if series else {},
        # The whole run after its first hour, beside the case's own window:
        # a column that clouds over and clears again between the two is a
        # finding the window alone would hide.
        "run_means": window_means(series, (3600.0, steps * bench.dt_s)) if series else {},
        "profiles": profiles,
    }


def step_readings_initial(bench: Bench) -> dict[str, float]:
    col = bench.column_host()
    return {"lwp_g_m2": float(1000.0 * (col["qc"] * col["dp"] / GRAVITY).sum())}


# --------------------------------------------------------------------------
# Published reference ranges.  Each row names where its numbers were read.
# A row is the LES ensemble's range over the case's scoring window, and the
# model's reading it is scored against is the window mean of one series of
# the result (``model_key``).

#: (case, metric, model_key, low, central, high, how the numbers were read)
REFERENCES: tuple[tuple, ...] = (
    ("bomex", "total cloud cover", "radiation_cloud_cover", 0.08, 0.13, 0.17,
     "Siebesma et al. 2003 (J. Atmos. Sci. 60, 1201), p. 1204: the models' "
     "means over the last 3 h range from 0.08 to 0.17, ensemble mean 0.13"),
    ("bomex", "liquid water path (g/m2)", "lwp_g_m2", 3.0, 7.0, 13.0,
     "Siebesma et al. 2003 Fig. 2b, read from the figure: ensemble mean and "
     "its two-standard-deviation band over hours 3 to 6 (approximate)"),
    ("rico", "cloud cover", "radiation_cloud_cover", 0.09, 0.19, 0.38,
     "vanZanten et al. 2011 (JAMES 3, doi:10.1029/2011MS000056) Fig. 3: minimum, four-hour mean "
     "and maximum of the full ensemble over hours 20 to 24"),
    ("rico", "liquid water path (g/m2)", "lwp_g_m2", 4.8, 19.4, 47.7,
     "vanZanten et al. 2011 Fig. 3, as above"),
    ("rico", "rain water path (g/m2)", "rwp_g_m2", 0.1, 3.8, 20.6,
     "vanZanten et al. 2011 Fig. 3, as above"),
    ("rico", "surface sensible heat flux (W/m2)", "sensible_w_m2", 5.9, 8.5, 14.5,
     "vanZanten et al. 2011 Fig. 3, as above"),
    ("rico", "surface latent heat flux (W/m2)", "latent_w_m2", 104.0, 158.0, 189.0,
     "vanZanten et al. 2011 Fig. 3, as above"),
    ("rico", "cloud base height (m)", "cloud_base_m", 120.0, 517.0, 751.0,
     "vanZanten et al. 2011 Fig. 3, as above"),
    ("rico", "inversion height (m)", "inversion_height_m", 1919.0, 2254.0, 2645.0,
     "vanZanten et al. 2011 Fig. 3, as above (height of the largest potential "
     "temperature gradient)"),
    ("rico", "surface precipitation (mm/day)", "surface_rain_mm_day",
     0.0, 7.5 * 86400.0 / LV, 31.2 * 86400.0 / LV,
     "vanZanten et al. 2011 Table 3: the thirteen simulations' surface "
     "precipitation over the last four hours, 0.0 to 31.2 W/m2, median 7.5 "
     "W/m2, converted at Lv = 2.5e6 J/kg"),
    ("dycoms-rf01", "liquid water path (g/m2)", "lwp_g_m2", 5.0, 35.0, 65.0,
     "Stevens et al. 2005 ensemble as replotted by Pressel et al. 2017 (JAMES 9, "
     "doi:10.1002/2016MS000778) Fig. 3, read from the figure over hours 3 to 4: "
     "absolute range and ensemble mean (approximate; the interquartile range "
     "reads about 22 to 45)"),
    ("dycoms-rf01", "cloud fraction", "radiation_cloud_cover", 0.2, 0.88, 1.0,
     "Pressel et al. 2017 Fig. 2, read from the figure over hours 3 to 4: the "
     "Stevens et al. 2005 ensemble's absolute range and mean (approximate); "
     "Pressel et al. p. 1350 give the observed cover as 100 percent"),
    ("gabls1", "boundary-layer height (m)", "pbl_height_m", 160.0, 180.0, 200.0,
     "Holtslag et al. 2012 (ECMWF GABLS workshop overview, after Svensson and "
     "Holtslag 2009) Fig. 2b, read from the figure: the averaged LES with its "
     "standard-deviation bar (approximate); the model's reading is YSU's own "
     "PBL height, a different definition"),
    ("gabls1", "surface wind turning angle (deg)", "surface_wind_angle_deg", 34.0, 36.0, 38.0,
     "Holtslag et al. 2012 p. 13: averaged LES 36 degrees; the bar is read "
     "from Fig. 2b (approximate); the model's angle is its lowest level's"),
)


def score(record: dict) -> list[dict]:
    """The result's window means against the case's reference rows."""
    rows = []
    means = record.get("window_means", {})
    for case, metric, key, low, central, high, source in REFERENCES:
        if case != record["case"]:
            continue
        value = means.get(key, float("nan"))
        if not math.isfinite(value):
            verdict = "no reading"
        elif low <= value <= high:
            verdict = "inside"
        else:
            verdict = "below" if value < low else "above"
        rows.append({"metric": metric, "model": value, "low": low, "central": central,
                     "high": high, "verdict": verdict, "source": source})
    return rows


# --------------------------------------------------------------------------
# Command line.


def score_table(records: list[dict]) -> list[str]:
    """The ``score`` table's lines for these result records, header first.

    A case with no reference row (FIRE, ARM SGP, TRMM-LBA) prints one line
    saying so, so a reader of the table is never left to guess whether its
    verdicts were dropped or never read.
    """

    lines = ["| case | arm | options | metric | model | LES low | LES central | LES high | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for record in records:
        label = record.get("label") or "shipped"
        rows = score(record)
        if not rows:
            lines.append(f"| {record['case']} | {record['arm']} | {label} | (no published "
                         "range read for this case) | | | | | |")
        for row in rows:
            lines.append(f"| {record['case']} | {record['arm']} | {label} | {row['metric']} "
                         f"| {row['model']:.3g} | {row['low']:.3g} | {row['central']:.3g} "
                         f"| {row['high']:.3g} | {row['verdict']} |")
    return lines


def _summary_line(record: dict) -> str:
    m = record["window_means"]
    keys = ("lwp_g_m2", "rwp_g_m2", "cloud_base_m", "cloud_top_m", "radiation_cloud_cover",
            "surface_rain_mm_day", "sensible_w_m2", "latent_w_m2", "toa_sw_up_w_m2",
            "toa_albedo", "olr_w_m2", "pbl_height_m")
    return " ".join(f"{k}={m.get(k, float('nan')):.4g}" for k in keys)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run cases on this host's card")
    run.add_argument("--case", action="append", required=True,
                     help=f"one of {sorted(CASES)} (repeatable), or 'all'")
    run.add_argument("--dephy-root", default=None)
    run.add_argument("--arm", action="append", default=None,
                     help="spec (default) and/or native-radiation (repeatable)")
    run.add_argument("--sun", default="case", choices=("case", "noon"))
    run.add_argument("--noon-at-hours", type=float, default=None,
                     help="with --sun noon: the run hour put at local solar noon "
                          "(default the middle of the scoring window)")
    run.add_argument("--columns", type=int, default=2)
    run.add_argument("--dx-m", type=float, default=None)
    run.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                     help="override one native physics option (a scheme change to "
                          "score, e.g. cumulus=ntiedtke); the value is read as JSON "
                          "when it parses, else as a string")
    run.add_argument("--label", default=None,
                     help="a name for the option set, added to the result file names")
    run.add_argument("--out", required=True)
    scorer = sub.add_parser("score", help="tabulate result files against the reference rows")
    scorer.add_argument("results", nargs="+")
    summary = sub.add_parser("summary", help="one row of readings per result file")
    summary.add_argument("results", nargs="+")
    fetch = sub.add_parser("fetch", help="download the six DEPHY-SCM case files at the "
                                         "pinned commit, sha256-checked, into OUT")
    fetch.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        for path in fetch_dephy(args.out):
            print(f"{path}  sha256 {_sha256(path)}")
        return 0
    if args.command == "summary":
        keys = (("lwp_g_m2", "LWP g/m2"), ("radiation_cloud_cover", "cloud cover"),
                ("cloud_base_m", "base m"), ("cloud_top_m", "top m"),
                ("surface_rain_mm_day", "rain mm/day"),
                ("convective_rain_mm_day", "of it convective"), ("sensible_w_m2", "SH W/m2"),
                ("latent_w_m2", "LH W/m2"), ("toa_sw_up_w_m2", "TOA SW up W/m2"),
                ("toa_albedo", "TOA albedo (sunlit)"), ("olr_w_m2", "OLR W/m2"),
                ("rrtmgp_lw_heating_below_1500m_w_m2", "RRTMGP LW <1.5 km W/m2"),
                ("formula_heating_below_1500m_w_m2", "case formula <1.5 km W/m2"))
        print("| result | means over | " + " | ".join(label for _, label in keys) + " |")
        print("|---|---|" + "---|" * len(keys))
        for path in args.results:
            record = json.loads(Path(path).read_text())
            name = f"{record['case']} {record['arm']} {record.get('label') or 'shipped'} sun={record['sun']}"
            if record.get("failure"):
                name += f" (FAILED at step {record['failure']['step']})"
            for span, means in (("window", record.get("window_means", {})),
                                ("run after 1 h", record.get("run_means", {}))):
                cells = " | ".join(f"{means.get(k, float('nan')):.3g}" for k, _ in keys)
                print(f"| {name} | {span} | {cells} |")
        return 0
    if args.command == "score":
        records = [json.loads(Path(path).read_text()) for path in args.results]
        print("\n".join(score_table(records)))
        return 0
    names = sorted(CASES) if args.case == ["all"] else args.case
    arms = args.arm or ["spec"]
    overrides = {}
    for item in args.option:
        key, _, text = item.partition("=")
        try:
            overrides[key] = json.loads(text)
        except json.JSONDecodeError:
            overrides[key] = text
    suffix = f"-{args.label}" if args.label else ""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    template = template_water_column()
    status = 0
    finished = []
    for name in names:
        case = load_case(name, args.dephy_root)
        for arm in arms:
            if arm == "native-radiation" and case.radiation not in ("tend", "formula", "on"):
                print(f"{name} {arm}: skipped (case radiation {case.radiation!r})", flush=True)
                continue
            if arm == "native-radiation" and case.radiation == "on":
                print(f"{name} {arm}: skipped (the spec arm already runs RRTMGP)", flush=True)
                continue
            sun = args.sun if arm == "spec" and case.radiation != "on" else "case"
            try:
                record = run_case(case, arm=arm, ncol=args.columns, sun=sun, dx_m=args.dx_m,
                                  template=template, log=lambda s: print(s, flush=True),
                                  option_overrides=overrides,
                                  noon_at_s=(None if args.noon_at_hours is None
                                             else 3600.0 * args.noon_at_hours))
            except Exception as exc:  # noqa: BLE001 - one case's failure is a reading
                print(f"{name} {arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                (out / f"{name}-{arm}-{sun}{suffix}.failed.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n")
                status = 1
                continue
            record["label"] = args.label
            record["option_overrides"] = overrides
            path = out / f"{name}-{arm}-{sun}{suffix}.json"
            path.write_text(json.dumps(record, allow_nan=True))
            if record["failure"] is not None:
                status = 1
            print(f"{name} {arm} sun={sun}: {record['steps']} steps in {record['wall_s']:.1f} s; "
                  f"{_summary_line(record)}", flush=True)
            finished.append(record)
    if finished:
        # The run ends on its score: every finished case against the
        # published ranges, the table `score` prints from the result files.
        print("\n".join(["", *score_table(finished)]), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
