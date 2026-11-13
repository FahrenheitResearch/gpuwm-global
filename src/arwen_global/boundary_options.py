"""The ``[ocean]``, ``[sea_ice]``, ``[land]`` and ``[output]`` tables.

The ocean rule is the user's choice, one switch (``[ocean] mode``):

* ``hold``: the cold-start sea surface temperature for the whole run (what
  a run with no series has always done, now stated);
* ``anomaly``: the starting anomaly carried on an evolving 1991-2020
  climatology (arwen_global.ocean_rules), so an anomaly present at the
  start (an El Nino) is kept while the seasons advance;
* ``climatology``: the climatology after a stated ramp that removes the
  starting anomaly;
* ``series``: a dated SST and sea-ice series (``[surface_update] series``),
  refused past its last record;
* ``slab``: a mixed layer that answers the model's fluxes, kept near the
  observed climate by a heat-flux correction made from the model's own
  prescribed-SST run, with thermodynamic sea ice (arwen_global.ocean_coupler,
  docs/ARWEN_GLOBAL_OCEAN.md).  ``climatology`` with ``qflux_calibration``
  is that calibration run.

``[sea_ice] mode`` follows the ocean by default (``match``).  ``[land]
greenness`` says whether vegetation, leaf area and background albedo
follow their monthly climatology day by day or stay at the start.
``[output]`` holds the long-run products and the checkpoint retention; it
changes no arithmetic and is outside the config identity.

Identity follows the rule the series set (config.py, "joins the identity
only when set"): a table that changes no arithmetic leaves every config
hash as it was, files enter by content SHA-256 and never by path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path

OCEAN_MODES = ("hold", "anomaly", "climatology", "slab", "series")
SEA_ICE_MODES = ("match", "hold", "anomaly", "climatology", "thermodynamic", "series")
RULE_MODES = ("anomaly", "climatology")
GREENNESS_MODES = ("daily", "start")
PERSIST_REGIONS = ("pacific", "global")
OFFSET_RULES = ("persist", "damp")
ANOMALY_REFERENCES = ("start", "mean-7d")
#: A run longer than this with no ocean rule and no series is refused (the
#: GFS series horizon, the longest a dated forecast boundary reaches).
LONG_RUN_THRESHOLD_S = 384.0 * 3600.0

_OCEAN_KEYS = {
    "mode", "climatology", "anomaly_damping_days", "tropical_anomaly_damping_days",
    "tropical_band_deg", "tropical_persist_region", "large_scale_offset",
    "climatology_ramp_days", "anomaly_reference", "anomaly_reference_series",
    "anomaly_truncation",
}
_SLAB_KEYS = {"slab_target", "slab_relaxation_days", "slab_depth",
              "slab_depth_bounds_m", "slab_qflux"}
_CALIBRATION_KEYS = {"qflux_calibration", "qflux_calibration_spinup_days"}
SLAB_TARGETS = ("climatology", "anomaly")
#: The new-ice thickness under ``sea_ice.mode = "thermodynamic"`` when the
#: key is not given (the rules' default stays 1.0 m).
THERMODYNAMIC_NEW_ICE_THICKNESS_M = 0.5
_SEA_ICE_KEYS = {"mode", "anomaly_damping_days", "record_interval_hours",
                 "new_ice_thickness_m"}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _positive(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive, finite number")
    return number


def _choice(value, choices, name: str) -> str:
    text = str(value).strip().lower()
    if text not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
    return text


def _none_or(value, convert, name: str):
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return convert(value, name)


def _truncation(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a whole truncation >= 1 or \"none\"")
    return int(value)


def _file(value, source_dir: Path, name: str, hint: str) -> tuple[str, str]:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"{name} {path} does not exist; {hint}")
    return str(path.resolve()), _sha256(path)


@dataclass(frozen=True)
class SeaIceOptions:
    mode: str = "match"
    anomaly_damping_days: float = 60.0
    record_interval_hours: float = 24.0
    new_ice_thickness_m: float = 1.0


@dataclass(frozen=True)
class OceanOptions:
    #: Whether the config wrote an ``[ocean]`` table (receipt only).
    declared: bool = False
    mode: str = "hold"
    climatology: str | None = None
    climatology_sha256: str | None = None
    anomaly_damping_days: float = 90.0
    #: None is "none": persisted, undamped.
    tropical_anomaly_damping_days: float | None = None
    tropical_band_deg: tuple = (10.0, 30.0)
    tropical_persist_region: str = "pacific"
    large_scale_offset: str = "persist"
    climatology_ramp_days: float = 3.0
    anomaly_reference: str = "mean-7d"
    anomaly_reference_series: str | None = None
    anomaly_reference_series_sha256: str | None = None
    #: None is "none": the grid field, unsmoothed.
    anomaly_truncation: int | None = 42
    sea_ice: SeaIceOptions = field(default_factory=SeaIceOptions)
    #: Whether the run's ocean rule was implied (no table) or stated.
    implicit: bool = True
    #: The slab keys (``mode = "slab"`` only).  ``slab_depth`` is "woa18"
    #: or metres; ``slab_relaxation_days`` None is "none".
    slab_qflux: str | None = None
    slab_qflux_sha256: str | None = None
    slab_depth: object = "woa18"
    slab_depth_bounds_m: tuple = (10.0, 200.0)
    slab_relaxation_days: float | None = 365.0
    slab_target: str = "climatology"
    #: The calibration of the heat-flux correction (``mode = "climatology"``
    #: only).
    qflux_calibration: bool = False
    qflux_calibration_spinup_days: float = 2.0

    @property
    def ice_mode(self) -> str:
        """The sea-ice rule after ``match``: hold for hold, anomaly for
        anomaly, climatology for climatology, series for series,
        thermodynamic for slab."""
        if self.sea_ice.mode != "match":
            return self.sea_ice.mode
        return "thermodynamic" if self.mode == "slab" else self.mode

    @property
    def slab(self) -> bool:
        return self.mode == "slab"

    @property
    def needs_climatology(self) -> bool:
        return self.mode in RULE_MODES or self.ice_mode in RULE_MODES or self.slab

    @property
    def needs_anomaly(self) -> bool:
        """Whether the starting anomaly enters the arithmetic (the anomaly
        and climatology rules, and the slab's restoring target under
        ``slab_target = "anomaly"``)."""
        return self.mode in RULE_MODES or (
            self.slab and self.slab_target == "anomaly"
            and self.slab_relaxation_days is not None)

    def slab_receipt(self) -> dict:
        """The slab and thermodynamic-ice keys, or the calibration keys, as
        the identity and the receipt carry them ({} for every other run)."""
        if self.slab:
            return {
                "slab_qflux_sha256": self.slab_qflux_sha256,
                "slab_depth": self.slab_depth,
                "slab_depth_bounds_m": list(self.slab_depth_bounds_m),
                "slab_relaxation_days": ("none" if self.slab_relaxation_days is None
                                         else self.slab_relaxation_days),
                "slab_target": self.slab_target,
                "sea_ice_record_interval_hours": self.sea_ice.record_interval_hours,
                "new_ice_thickness_m": self.sea_ice.new_ice_thickness_m,
            }
        if self.qflux_calibration:
            return {"qflux_calibration": True,
                    "qflux_calibration_spinup_days": self.qflux_calibration_spinup_days}
        return {}

    @property
    def identity(self) -> dict | None:
        """The hash payload, or None where the arithmetic is the one a run
        with no table already has (hold with held ice; the series)."""
        ice = self.ice_mode
        if self.mode in ("hold", "series") and ice in ("hold", "series"):
            return None
        payload: dict[str, object] = {"mode": self.mode, "sea_ice_mode": ice,
                                      "climatology_sha256": self.climatology_sha256}
        if self.needs_anomaly:
            payload.update(
                anomaly_damping_days=self.anomaly_damping_days,
                tropical_anomaly_damping_days=self.tropical_anomaly_damping_days,
                tropical_band_deg=list(self.tropical_band_deg),
                tropical_persist_region=self.tropical_persist_region,
                large_scale_offset=self.large_scale_offset,
                climatology_ramp_days=self.climatology_ramp_days,
                anomaly_reference=self.anomaly_reference,
                anomaly_reference_series_sha256=(
                    self.anomaly_reference_series_sha256
                    if self.anomaly_reference == "mean-7d" else None),
                anomaly_truncation=self.anomaly_truncation,
            )
        if ice in RULE_MODES:
            payload.update(
                sea_ice_anomaly_damping_days=self.sea_ice.anomaly_damping_days,
                sea_ice_record_interval_hours=self.sea_ice.record_interval_hours,
                new_ice_thickness_m=self.sea_ice.new_ice_thickness_m,
            )
            if ice == "climatology":
                payload["climatology_ramp_days"] = self.climatology_ramp_days
        # The slab and the calibration join only when set, so every config
        # hash written before them stays as it was.
        payload.update(self.slab_receipt())
        return payload

    def receipt(self) -> dict:
        out = {
            "ocean_mode": self.mode, "sea_ice_mode": self.ice_mode,
            "stated": not self.implicit,
        }
        if self.needs_climatology:
            out["climatology"] = self.climatology
            out["climatology_sha256"] = self.climatology_sha256
        if self.needs_anomaly:
            out.update({
                "anomaly_damping_days": self.anomaly_damping_days,
                "tropical_anomaly_damping_days": (
                    "none" if self.tropical_anomaly_damping_days is None
                    else self.tropical_anomaly_damping_days),
                "tropical_band_deg": list(self.tropical_band_deg),
                "tropical_persist_region": self.tropical_persist_region,
                "large_scale_offset": self.large_scale_offset,
                "climatology_ramp_days": self.climatology_ramp_days,
                "anomaly_reference": self.anomaly_reference,
                "anomaly_reference_series": self.anomaly_reference_series,
                "anomaly_reference_series_sha256": self.anomaly_reference_series_sha256,
                "anomaly_truncation": ("none" if self.anomaly_truncation is None
                                       else self.anomaly_truncation),
            })
        if self.ice_mode in RULE_MODES:
            out.update({
                "sea_ice_anomaly_damping_days": self.sea_ice.anomaly_damping_days,
                "sea_ice_record_interval_hours": self.sea_ice.record_interval_hours,
                "new_ice_thickness_m": self.sea_ice.new_ice_thickness_m,
            })
        if self.slab or self.qflux_calibration:
            out.update(self.slab_receipt())
            if self.slab:
                out["slab_qflux"] = self.slab_qflux
        return out


@dataclass(frozen=True)
class LandOptions:
    #: None: the ocean mode's default (daily under a rule or a series,
    #: start under hold).
    greenness: str | None = None

    def resolved(self, ocean: OceanOptions) -> str:
        if self.greenness is not None:
            return self.greenness
        return "start" if ocean.mode == "hold" else "daily"

    def identity(self, ocean: OceanOptions) -> dict | None:
        """Joins only when it states something the mode would not do."""
        default = LandOptions().resolved(ocean)
        if self.greenness is None or self.greenness == default:
            return None
        return {"greenness": self.greenness}


@dataclass(frozen=True)
class OutputOptions:
    """Long-run products and checkpoint retention (arwen_global.means,
    arwen_global.retention).  Read-only instruments: outside the identity."""
    #: "all", "newest:N" or "newest:N+month-ends".
    keep_checkpoints: str = "all"
    #: "none", "monthly", "daily" or "monthly+daily".
    mean_state: str = "none"
    daily_summary: bool = False
    #: Refuse to start when the run's disk estimate exceeds the free space
    #: less this reserve.
    disk_reserve_gb: float = 20.0

    @property
    def mean_windows(self) -> tuple[str, ...]:
        if self.mean_state == "none":
            return ()
        return tuple(self.mean_state.split("+"))

    @property
    def keep_newest(self) -> int | None:
        if self.keep_checkpoints == "all":
            return None
        return int(self.keep_checkpoints.split(":", 1)[1].split("+", 1)[0])

    @property
    def keep_month_ends(self) -> bool:
        return self.keep_checkpoints.endswith("+month-ends")


#: The [output] keys this module reads.  The same table also carries the
#: checkpoint writer's ``checkpoint_compression`` and ``snapshot_buffer``,
#: which arwen_global.config reads and hands nothing of here.
OUTPUT_PRODUCT_KEYS = frozenset(
    {"keep_checkpoints", "mean_state", "daily_summary", "disk_reserve_gb"})


def parse_output(table: dict) -> OutputOptions:
    unknown = sorted(set(table) - OUTPUT_PRODUCT_KEYS)
    if unknown:
        raise ValueError(f"unknown keys in [output]: {', '.join(unknown)}")
    keep = str(table.get("keep_checkpoints", "all")).strip().lower()
    if keep != "all":
        head, _, tail = keep.partition("+")
        name, _, count = head.partition(":")
        if name != "newest" or not count.isdigit() or int(count) < 1 \
                or tail not in ("", "month-ends"):
            raise ValueError(
                "output.keep_checkpoints must be \"all\", \"newest:N\" (N >= 1) or "
                f"\"newest:N+month-ends\"; got {table.get('keep_checkpoints')!r}")
    mean = str(table.get("mean_state", "none")).strip().lower()
    if mean not in ("none", "monthly", "daily", "monthly+daily", "daily+monthly"):
        raise ValueError("output.mean_state must be none, monthly, daily or monthly+daily")
    if mean == "daily+monthly":
        mean = "monthly+daily"
    summary = table.get("daily_summary", False)
    if not isinstance(summary, bool):
        raise ValueError("output.daily_summary must be true or false")
    reserve = float(table.get("disk_reserve_gb", 20.0))
    if not math.isfinite(reserve) or reserve < 0.0:
        raise ValueError("output.disk_reserve_gb must be >= 0")
    return OutputOptions(keep_checkpoints=keep, mean_state=mean, daily_summary=summary,
                         disk_reserve_gb=reserve)


def parse_land(table: dict) -> LandOptions:
    unknown = sorted(set(table) - {"greenness"})
    if unknown:
        raise ValueError(f"unknown keys in [land]: {', '.join(unknown)}")
    if "greenness" not in table:
        return LandOptions()
    return LandOptions(_choice(table["greenness"], GREENNESS_MODES, "land.greenness"))


def parse_ocean(ocean_table: dict | None, ice_table: dict | None, *, source_dir: Path,
                surface_series: str | None, duration_s: float, initial_mode: str,
                statics_source: str) -> OceanOptions:
    """Both tables, checked against the run they belong to.

    ``ocean_table`` is None when the config has no ``[ocean]`` table: the
    mode is then the series when one is set, else hold (an implicit hold,
    which :func:`long_run_refusal` refuses on a long dated run).
    """
    ice_table = dict(ice_table or {})
    unknown = sorted(set(ice_table) - _SEA_ICE_KEYS)
    if unknown:
        raise ValueError(f"unknown keys in [sea_ice]: {', '.join(unknown)}")
    ice_mode = _choice(ice_table.get("mode", "match"), SEA_ICE_MODES, "sea_ice.mode")
    ocean_mode = (None if ocean_table is None
                  else str(dict(ocean_table).get("mode", "anomaly")).strip().lower())
    thermodynamic = ice_mode == "thermodynamic" or (
        ice_mode == "match" and ocean_mode == "slab")
    if ice_mode == "thermodynamic" and ocean_mode != "slab":
        raise ValueError(
            "sea_ice.mode = \"thermodynamic\" (ice that grows and melts from the "
            "model's fluxes) goes only with [ocean] mode = \"slab\": the ice and the "
            "water must answer the same energy")
    if ocean_mode == "slab" and ice_mode not in ("match", "thermodynamic"):
        raise ValueError(
            "[ocean] mode = \"slab\" carries its own sea ice (sea_ice.mode = "
            f"\"thermodynamic\", which \"match\" resolves to); {ice_mode!r} would "
            "prescribe ice under a prognostic SST")
    sea_ice = SeaIceOptions(
        mode=ice_mode,
        anomaly_damping_days=_positive(ice_table.get("anomaly_damping_days", 60.0),
                                       "sea_ice.anomaly_damping_days"),
        record_interval_hours=_positive(ice_table.get("record_interval_hours", 24.0),
                                        "sea_ice.record_interval_hours"),
        new_ice_thickness_m=_positive(
            ice_table.get("new_ice_thickness_m",
                          THERMODYNAMIC_NEW_ICE_THICKNESS_M if thermodynamic else 1.0),
            "sea_ice.new_ice_thickness_m"),
    )
    if ocean_table is None:
        mode = "series" if surface_series else "hold"
        if ice_table and ice_mode not in ("match", mode):
            raise ValueError(
                f"[sea_ice] mode = {ice_mode!r} needs an [ocean] table that names the "
                "ocean rule it goes with")
        options = OceanOptions(mode=mode, sea_ice=sea_ice, implicit=True)
        return _with_climatology(options, {}, source_dir)
    ocean_table = dict(ocean_table)
    slab_keys = sorted(set(ocean_table) & _SLAB_KEYS)
    calibration_keys = sorted(set(ocean_table) & _CALIBRATION_KEYS)
    unknown = sorted(set(ocean_table) - _OCEAN_KEYS - _SLAB_KEYS - _CALIBRATION_KEYS)
    if unknown:
        raise ValueError(f"unknown keys in [ocean]: {', '.join(unknown)}")
    mode = _choice(ocean_table.get("mode", "anomaly"), OCEAN_MODES, "ocean.mode")
    if slab_keys and mode != "slab":
        raise ValueError(
            f"{', '.join('ocean.' + key for key in slab_keys)} "
            f"{'is' if len(slab_keys) == 1 else 'are'} read only under [ocean] mode = "
            f"\"slab\"; this run's mode is {mode!r}")
    calibrate = ocean_table.get("qflux_calibration", False)
    if not isinstance(calibrate, bool):
        raise ValueError("ocean.qflux_calibration must be true or false")
    if calibration_keys and mode != "climatology":
        raise ValueError(
            "ocean.qflux_calibration (the prescribed-SST run a slab's heat-flux "
            "correction is made from) goes only with [ocean] mode = \"climatology\"; "
            f"this run's mode is {mode!r}")
    if "qflux_calibration_spinup_days" in ocean_table and not calibrate:
        raise ValueError("ocean.qflux_calibration_spinup_days is read only with "
                         "qflux_calibration = true")
    if mode == "series" and not surface_series:
        raise ValueError("[ocean] mode = \"series\" needs [surface_update] series "
                         "(or --surface-series)")
    if surface_series and mode != "series":
        raise ValueError(
            f"[ocean] mode = {mode!r} and a [surface_update] series both set the SST; "
            "set mode = \"series\" to follow the series, or remove the series")
    if ice_mode == "series" and mode != "series":
        raise ValueError("sea_ice.mode = \"series\" needs [ocean] mode = \"series\"")
    if mode == "series" and ice_mode not in ("match", "series"):
        raise ValueError(
            "under [ocean] mode = \"series\" the sea ice follows the series "
            "(sea_ice.mode = \"match\" or \"series\")")
    band = ocean_table.get("tropical_band_deg", [10.0, 30.0])
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        raise ValueError("ocean.tropical_band_deg must be two latitudes [inner, outer]")
    band = (float(band[0]), float(band[1]))
    if not (0.0 <= band[0] < band[1] <= 90.0):
        raise ValueError("ocean.tropical_band_deg must satisfy 0 <= inner < outer <= 90")
    options = OceanOptions(
        declared=True, mode=mode, implicit=False,
        anomaly_damping_days=_positive(ocean_table.get("anomaly_damping_days", 90.0),
                                       "ocean.anomaly_damping_days"),
        tropical_anomaly_damping_days=_none_or(
            ocean_table.get("tropical_anomaly_damping_days", "none"), _positive,
            "ocean.tropical_anomaly_damping_days"),
        tropical_band_deg=band,
        tropical_persist_region=_choice(
            ocean_table.get("tropical_persist_region", "pacific"), PERSIST_REGIONS,
            "ocean.tropical_persist_region"),
        large_scale_offset=_choice(ocean_table.get("large_scale_offset", "persist"),
                                   OFFSET_RULES, "ocean.large_scale_offset"),
        climatology_ramp_days=_positive(ocean_table.get("climatology_ramp_days", 3.0),
                                        "ocean.climatology_ramp_days"),
        anomaly_reference=_choice(ocean_table.get("anomaly_reference", "mean-7d"),
                                  ANOMALY_REFERENCES, "ocean.anomaly_reference"),
        anomaly_truncation=_none_or(ocean_table.get("anomaly_truncation", 42),
                                    _truncation, "ocean.anomaly_truncation"),
        sea_ice=sea_ice,
    )
    if mode == "slab":
        options = _with_slab(options, ocean_table, source_dir)
    elif calibrate:
        spinup = ocean_table.get("qflux_calibration_spinup_days", 2.0)
        if isinstance(spinup, bool) or not math.isfinite(float(spinup)) \
                or float(spinup) < 0.0:
            raise ValueError("ocean.qflux_calibration_spinup_days must be a finite "
                             "number of days >= 0")
        options = replace(options, qflux_calibration=True,
                          qflux_calibration_spinup_days=float(spinup))
    if options.needs_climatology and initial_mode != "analysis" \
            and statics_source != "real":
        raise ValueError(
            f"[ocean] mode = {mode!r} with sea_ice mode {options.ice_mode!r} reads a "
            "climatology built on real statics' ocean; this run has neither an "
            "analysis nor real statics")
    return _with_climatology(options, ocean_table, source_dir)


def _with_slab(options: OceanOptions, table: dict, source_dir: Path) -> OceanOptions:
    """The slab keys, checked; the Q-flux file enters by content SHA-256."""
    if "slab_qflux" not in table:
        raise ValueError(
            "[ocean] mode = \"slab\" needs slab_qflux = \"<path>\": the heat-flux "
            "correction `gpuwm-global climatology qflux` writes from the last "
            "checkpoint of a calibration run ([ocean] mode = \"climatology\" with "
            "qflux_calibration = true; docs/ARWEN_GLOBAL_OCEAN.md)")
    qflux, qflux_sha = _file(table["slab_qflux"], source_dir, "ocean.slab_qflux",
                             "write it with `gpuwm-global climatology qflux CHECKPOINT "
                             "--out <path>` from a calibration run")
    depth = table.get("slab_depth", "woa18")
    if isinstance(depth, str):
        if depth.strip().lower() != "woa18":
            raise ValueError("ocean.slab_depth must be \"woa18\" or a depth in metres; "
                             f"got {depth!r}")
        depth = "woa18"
    else:
        depth = _positive(depth, "ocean.slab_depth")
    bounds = table.get("slab_depth_bounds_m", [10.0, 200.0])
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("ocean.slab_depth_bounds_m must be two depths [lower, upper] in m")
    bounds = (_positive(bounds[0], "ocean.slab_depth_bounds_m"),
              _positive(bounds[1], "ocean.slab_depth_bounds_m"))
    if not bounds[0] < bounds[1]:
        raise ValueError("ocean.slab_depth_bounds_m must satisfy lower < upper")
    relax = _none_or(table.get("slab_relaxation_days", 365.0), _positive,
                     "ocean.slab_relaxation_days")
    target = _choice(table.get("slab_target", "climatology"), SLAB_TARGETS,
                     "ocean.slab_target")
    return replace(options, slab_qflux=qflux, slab_qflux_sha256=qflux_sha,
                   slab_depth=depth, slab_depth_bounds_m=bounds,
                   slab_relaxation_days=relax, slab_target=target)


def long_run_refusal(cfg) -> str | None:
    """Why ``cfg`` may not run, or None.

    A dated run (from an analysis) on real statics, longer than
    :data:`LONG_RUN_THRESHOLD_S`, with no ``[ocean]`` table and no series
    would hold its starting SST for the whole run without saying so.  It is
    refused; ``[ocean] mode = "hold"`` states the hold and runs.  Idealized
    and undated runs are not refused.
    """
    ocean = cfg.ocean
    if not ocean.implicit or ocean.mode != "hold" or getattr(cfg, "surface_series", None):
        return None
    if getattr(cfg, "initial_mode", None) != "analysis"             or getattr(cfg.statics, "source", None) != "real":
        return None
    if float(cfg.duration_s) <= LONG_RUN_THRESHOLD_S:
        return None
    days = float(cfg.duration_s) / 86400.0
    return (
        f"this run is {days:g} days long and states no ocean rule: the starting "
        f"SST would be held for {days:g} days.  Add an [ocean] table: mode = "
        "\"anomaly\" (the starting anomaly on an evolving climatology; build the "
        "climatology with `gpuwm-global climatology build CONFIG`), "
        "\"climatology\", \"series\" (with [surface_update] series) or \"hold\" "
        "to accept the held ocean, or \"slab\" (a mixed layer with thermodynamic "
        "sea ice; docs/ARWEN_GLOBAL_OCEAN.md).")


def with_series(ocean: OceanOptions) -> OceanOptions:
    """The options after a series is given on the command line."""
    from dataclasses import replace

    if ocean.mode == "series":
        return ocean
    if not ocean.implicit:
        raise ValueError(
            f"--surface-series and [ocean] mode = {ocean.mode!r} both set the SST; "
            "set mode = \"series\" in the config to follow the series")
    return replace(ocean, mode="series")


def _with_climatology(options: OceanOptions, table: dict, source_dir: Path) -> OceanOptions:
    updates: dict[str, object] = {}
    if options.needs_climatology:
        if "climatology" not in table:
            raise ValueError(
                f"[ocean] mode = {options.mode!r} (sea ice {options.ice_mode!r}) needs "
                "climatology = \"<path>\": the model-grid file `gpuwm-global "
                "climatology build CONFIG` writes (it prints the path)")
        path, digest = _file(table["climatology"], source_dir, "ocean.climatology",
                             "build it with `gpuwm-global climatology build CONFIG`")
        updates.update(climatology=path, climatology_sha256=digest)
    elif "climatology" in table:
        raise ValueError(
            f"ocean.climatology is read only by the anomaly and climatology rules; "
            f"mode {options.mode!r} with sea ice {options.ice_mode!r} reads none")
    if options.needs_anomaly and options.anomaly_reference == "mean-7d":
        if "anomaly_reference_series" not in table:
            raise ValueError(
                "ocean.anomaly_reference = \"mean-7d\" (the default) needs "
                "anomaly_reference_series = \"<path>\": the 28 GDAS analyses of the 7 "
                "days before the start, built with `gpuwm-global surface-series CONFIG "
                "--source gdas-analysis --cycle <start YYYYMMDDHH> --anomaly-reference "
                "--out <path>`; or set anomaly_reference = \"start\" to take the "
                "anomaly from the cold-start analysis alone")
        path, digest = _file(table["anomaly_reference_series"], source_dir,
                             "ocean.anomaly_reference_series",
                             "build it with `gpuwm-global surface-series CONFIG --source "
                             "gdas-analysis --cycle <start> --anomaly-reference --out <path>`")
        updates.update(anomaly_reference_series=path,
                       anomaly_reference_series_sha256=digest)
    elif "anomaly_reference_series" in table:
        raise ValueError("ocean.anomaly_reference_series is read only with "
                         "anomaly_reference = \"mean-7d\" under the anomaly or "
                         "climatology rule")
    return replace(options, **updates) if updates else options


__all__ = [
    "GREENNESS_MODES", "LONG_RUN_THRESHOLD_S", "LandOptions", "OCEAN_MODES",
    "OceanOptions", "OutputOptions", "RULE_MODES", "SEA_ICE_MODES", "SLAB_TARGETS",
    "SeaIceOptions", "THERMODYNAMIC_NEW_ICE_THICKNESS_M",
    "long_run_refusal", "parse_land", "parse_ocean", "parse_output", "with_series",
]
