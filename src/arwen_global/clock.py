"""The forecast clock: the one instant model time zero stands for.

WHY ONE SOURCE.  Three copies of the same instant used to travel apart:
the analysis frame's valid time (recorded, never compared), the native
suite's ``start_time_utc`` (a config literal that every shipped GDAS
config pinned to its own packaged cycle, and that no door rewrote when the
analysis changed), and the render door's ``--start-date`` (a free string
stamped on every tape).  A user who pointed a config at today's 12Z cycle
ran the sun of the packaged cycle: the radiation's hour angle and
declination came from one instant while the atmosphere was the analysis
at another, and the receipt reported ``pass``.  The reference suite had no
date at all and treated model time zero as 00 UTC at equinox, so the
quickstart's 18Z analysis ran its diurnal cycle six hours off.

THE RULE.  When the forecast starts from an analysis, the analysis frame's
valid time IS the clock.  A config that still states ``start_time_utc`` is
cross-checked against it and refused by name when the two disagree;
nothing silently wins.  Without an analysis, a stated ``start_time_utc`` is
the clock; without either, the clock is the idealized fixture
(:data:`IDEALIZED_FIXTURE`), declared as such in every receipt, and the
native suite (whose radiation needs a real date) refuses to build.

The clock rides three places so a reader never has to guess it: the model
(``model.forecast_clock``), the run receipt (``"forecast_clock"``), and
every checkpoint's physics metadata (:data:`FORECAST_CLOCK_KEY`), so a
render or a restart from a checkpoint is checked against the clock it was
integrated under.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import math

__all__ = [
    "ANALYSIS_VALID_TIME",
    "CONFIG_START_TIME",
    "ClockMismatchError",
    "FORECAST_CLOCK_KEY",
    "ForecastClock",
    "IDEALIZED_FIXTURE",
    "UNSTATED",
    "analysis_valid_time",
    "check_checkpoint_clock",
    "config_is_dated",
    "declared_start_time",
    "format_utc",
    "mismatch_sentence",
    "parse_utc_instant",
    "receipt_start_utc",
    "resolve_forecast_clock",
    "solar_declination_and_equation_of_time",
    "undated_render_reason",
]

#: The physics-metadata key every checkpoint of a dated run carries.
FORECAST_CLOCK_KEY = "forecast_clock"

#: Sources a clock can come from.
ANALYSIS_VALID_TIME = "analysis-valid-time"
CONFIG_START_TIME = "config-start_time_utc"
IDEALIZED_FIXTURE = "idealized-fixture"
UNSTATED = "unstated"

#: What the idealized fixture means, word for word in every receipt that
#: runs on it.  This is the clock the reference suite always had, kept
#: bit-identical for idealized test cases (analytic, baroclinic wave) that
#: have no date to take one from.
FIXTURE_STATEMENT = (
    "idealized test fixture: model time zero is 00 UTC at an equinox; the "
    "hour angle is model time plus longitude, the declination is zero and "
    "there is no equation of time"
)


class ClockMismatchError(ValueError):
    """Two copies of the forecast's start instant disagree."""


def parse_utc_instant(value) -> dt.datetime | None:
    """``value`` as an aware UTC datetime: a datetime (naive read as UTC,
    the engine's frame convention), an ISO-8601 string with ``Z`` or an
    offset, a ``YYYY-MM-DD HH:MM:SS`` string (``str`` of a naive frame
    time), or a WRF ``YYYY-MM-DD_HH:MM:SS``.  None for None."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        moment = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if hasattr(value, "astype") and "T" in text and len(text) > 19:
            text = text[:19]  # numpy datetime64 at sub-second resolution
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if len(text) >= 11 and text[10] == "_":
            text = text[:10] + "T" + text[11:]
        try:
            moment = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a UTC instant") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def format_utc(moment: dt.datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ``, the spelling ``start_time_utc`` uses."""
    return moment.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ForecastClock:
    """The instant model time zero stands for and where it came from.

    ``start_utc`` is None for the idealized fixture and for a run whose
    clock nobody states (no analysis, no physics that reads the sun)."""

    start_utc: dt.datetime | None
    source: str
    analysis_valid_time: dt.datetime | None = None
    declared_start_time: dt.datetime | None = None

    @property
    def dated(self) -> bool:
        return self.start_utc is not None

    @property
    def iso(self) -> str | None:
        return None if self.start_utc is None else format_utc(self.start_utc)

    def valid_at(self, time_s: float) -> dt.datetime:
        if self.start_utc is None:
            raise ValueError(
                f"this run's clock is {self.source}: it has no date, so model "
                "time cannot be placed in real time"
            )
        return self.start_utc + dt.timedelta(seconds=float(time_s))

    def metadata(self) -> dict[str, object]:
        """The checkpoint physics-metadata record (dated clocks only)."""
        return {"start_utc": self.iso, "source": self.source}

    def receipt(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "start_utc": self.iso,
            "source": self.source,
            "analysis_valid_time": (
                None if self.analysis_valid_time is None
                else format_utc(self.analysis_valid_time)
            ),
            "declared_start_time_utc": (
                None if self.declared_start_time is None
                else format_utc(self.declared_start_time)
            ),
        }
        if self.source == IDEALIZED_FIXTURE:
            payload["fixture"] = FIXTURE_STATEMENT
        return payload


def declared_start_time(cfg) -> dt.datetime | None:
    """The native suite's ``start_time_utc`` when the config states one."""
    options = getattr(cfg, "native_adapter_options", None) or {}
    value = options.get("start_time_utc") if isinstance(options, dict) else None
    if value is None:
        return None
    try:
        return parse_utc_instant(value)
    except ValueError as exc:
        raise ValueError(
            f"the physics options' start_time_utc {value!r} is not an "
            "ISO-8601 instant"
        ) from exc


def config_is_dated(cfg) -> bool:
    """Whether a run of ``cfg`` has a real start instant before it runs:
    it starts from an analysis (whose frame carries its valid time) or it
    states the native suite's ``start_time_utc``.  The render stage of a
    dated run takes its tapes' valid times from that clock and needs no
    ``--start-date``."""
    if getattr(cfg, "initial_mode", None) == "analysis":
        return True
    return declared_start_time(cfg) is not None


def undated_render_reason(cfg) -> str | None:
    """Why a render of ``cfg``'s checkpoints needs a typed ``--start-date``,
    or None when the run's own clock dates its tapes."""
    if config_is_dated(cfg):
        return None
    return ("this config starts from no analysis and states no physics "
            "start_time_utc, so its run has no instant to stamp a tape with "
            "(the idealized fixture), and no start date was given")


def analysis_valid_time(initial_provenance) -> dt.datetime | None:
    """The analysis frame's valid time from the cold start's provenance."""
    if not isinstance(initial_provenance, dict):
        return None
    if initial_provenance.get("mode") != "analysis":
        return None
    return parse_utc_instant(initial_provenance.get("valid_time"))


def _hours(delta: dt.timedelta) -> float:
    return delta.total_seconds() / 3600.0


def mismatch_sentence(what: str, stated: dt.datetime, clock: dt.datetime,
                      clock_name: str = "the analysis valid time") -> str:
    """The breakage a disagreeing instant would cause, in numbers."""
    offset_h = _hours(stated - clock)
    phase_h = math.fmod(offset_h, 24.0)
    if phase_h > 12.0:
        phase_h -= 24.0
    elif phase_h < -12.0:
        phase_h += 24.0
    dec_stated, _ = solar_declination_and_equation_of_time(stated)
    dec_clock, _ = solar_declination_and_equation_of_time(clock)
    return (
        f"{what} {format_utc(stated)} is not {clock_name} "
        f"{format_utc(clock)}: the solar geometry would run {offset_h:+g} h "
        f"from the atmosphere it heats (the diurnal cycle {phase_h:+g} h out "
        f"of phase, the declination off by "
        f"{math.degrees(dec_stated - dec_clock):+.1f} deg) and every valid "
        "time would be stamped from the wrong instant"
    )


def resolve_forecast_clock(cfg, initial_provenance) -> ForecastClock:
    """The run's clock, refused by name when its copies disagree.

    Analysis start: the frame's valid time; a stated ``start_time_utc``
    must equal it (delete the line to follow the analysis).  Otherwise a
    stated ``start_time_utc``; otherwise the idealized fixture, which the
    native suite refuses because its radiation needs a real date."""
    declared = declared_start_time(cfg)
    physics_mode = getattr(cfg, "physics_mode", "none")
    if getattr(cfg, "initial_mode", None) == "analysis":
        valid = analysis_valid_time(initial_provenance)
        if valid is None:
            if physics_mode in ("arwen-native", "reference"):
                raise ValueError(
                    "the analysis frame carries no valid time, so the "
                    "physics' solar geometry would have no instant to run on "
                    "and no render tape could be placed in time"
                )
            return ForecastClock(None, UNSTATED, declared_start_time=declared)
        if declared is not None and declared != valid:
            raise ClockMismatchError(
                mismatch_sentence(
                    "the config's physics start_time_utc", declared, valid)
                + ".  Delete start_time_utc from the config to follow the "
                "analysis, or point initial.analysis_grib at the cycle it "
                "names."
            )
        return ForecastClock(valid, ANALYSIS_VALID_TIME,
                             analysis_valid_time=valid,
                             declared_start_time=declared)
    if declared is not None:
        return ForecastClock(declared, CONFIG_START_TIME,
                             declared_start_time=declared)
    if physics_mode == "arwen-native":
        raise ValueError(
            "the native suite's radiation needs the real instant model time "
            "zero stands for (its declination and hour angle are dated), and "
            "this config has neither an analysis to take it from nor a "
            "physics start_time_utc: initialise from an analysis, or set "
            "start_time_utc"
        )
    return ForecastClock(None, IDEALIZED_FIXTURE)


def check_checkpoint_clock(physics_metadata, clock: ForecastClock, label: str) -> None:
    """Refuse a checkpoint integrated under a different clock than the
    model rebuilt from its config: its fields were heated by another sun
    and its valid times would be stamped from the wrong instant."""
    if not isinstance(physics_metadata, dict):
        return
    record = physics_metadata.get(FORECAST_CLOCK_KEY)
    if not isinstance(record, dict) or record.get("start_utc") is None:
        return
    written = parse_utc_instant(record["start_utc"])
    if clock.start_utc is None or written != clock.start_utc:
        raise ClockMismatchError(
            f"{label} was integrated from {format_utc(written)} but this "
            f"config's clock is {clock.iso or clock.source}; a state from "
            "another start would be resumed under, or stamped with, the "
            "wrong sun and the wrong valid times"
        )


def receipt_start_utc(receipt) -> dt.datetime | None:
    """The run's start instant from a run receipt: its forecast clock, a
    cycle receipt's own start, or for a receipt written before the clock
    existed, the config's native ``start_time_utc``.  None when none
    states one."""
    if not isinstance(receipt, dict):
        return None
    clock = receipt.get("forecast_clock")
    if isinstance(clock, dict) and clock.get("start_utc"):
        return parse_utc_instant(clock["start_utc"])
    # A cycle's receipt (written under the same name) records the instant
    # it cycled from under "cycle", and a fresh-door config no longer
    # states the literal, so without this a reader pointed at a cycle
    # directory could read no date at all.
    cycle = receipt.get("cycle")
    if isinstance(cycle, dict) and cycle.get("start_utc"):
        return parse_utc_instant(cycle["start_utc"])
    options = (receipt.get("config") or {}).get("native_adapter_options") or {}
    start = options.get("start_time_utc") if isinstance(options, dict) else None
    return parse_utc_instant(start) if start else None


def solar_declination_and_equation_of_time(moment: dt.datetime) -> tuple[float, float]:
    """WRF v4.6.1 ``radconst`` geometry at ``moment``: the declination in
    radians and the equation of time in minutes, from WRF's zero-based
    fractional julian day on its fixed 365-day orbit (the same arithmetic
    the native suite's RRTMGP clock runs, core/rrtmgp.py
    ``_cosine_zenith``)."""
    moment = moment.astimezone(dt.timezone.utc)
    hour = (moment.hour + moment.minute / 60.0 + moment.second / 3600.0
            + moment.microsecond / 3.6e9)
    julian = moment.timetuple().tm_yday - 1.0 + hour / 24.0
    degrad = math.pi / 180.0
    dpd = 360.0 / 365.0
    if julian >= 80.0:
        solar_longitude = dpd * (julian - 80.0)
    else:
        solar_longitude = dpd * (julian + 285.0)
    declination = math.asin(
        math.sin(23.5 * degrad) * math.sin(solar_longitude * degrad))
    da = 2.0 * math.pi * (julian - 1.0) / 365.0
    equation = 229.18 * (
        0.000075 + 0.001868 * math.cos(da) - 0.032077 * math.sin(da)
        - 0.014615 * math.cos(2.0 * da) - 0.04089 * math.sin(2.0 * da))
    return declination, equation
