"""The forecast clock: one instant, taken from the analysis, checked everywhere.

THE BREAKAGE THESE GATES PREVENT.  Three copies of the instant model time
zero stands for travelled apart: the analysis frame's valid time (recorded
in the receipt, never compared), the native suite's ``start_time_utc`` (a
config literal no door rewrote when the analysis changed, so a run of
today's 12Z cycle radiated on the packaged cycle's sun) and the render
door's ``--start-date`` (a free string stamped on every tape).  The
reference suite had no date at all: model time zero was 00 UTC at an
equinox, so the quickstart's 18Z analysis ran its diurnal cycle six hours
off.  Every gate here fails on the code that had those three copies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from arwen_global.config import load_config
from arwen_global.configs_dir import config_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_arwen_global_analysis_initial import (  # noqa: E402
    _analysis_cfg, _synthetic_frame,
)

UTC = timezone.utc


def _frame_at(valid: datetime):
    frame = _synthetic_frame()
    frame.valid_time = valid
    frame.source_cycle = valid
    return frame


@pytest.fixture
def analysis_at(monkeypatch):
    """Route every cold start through a synthetic analysis frame valid at
    the instant the test names, exactly as the GRIB decode would hand it."""
    from arwen_global import analysis_initial

    real = analysis_initial.analysis_initial_state
    holder: dict[str, datetime] = {}

    def fake(cfg, transform, frame=None, statics=None, **kwargs):
        return real(cfg, transform, frame=_frame_at(holder["valid"]),
                    statics=statics, **kwargs)

    monkeypatch.setattr(analysis_initial, "analysis_initial_state", fake)

    def set_valid(valid: datetime) -> None:
        holder["valid"] = valid

    return set_valid


def _native_analysis_cfg(tmp_path: Path, start_time_utc: str | None):
    """The shipped native T255 config as a user points it at a cycle: the
    analysis-mode identity, with or without its packaged start_time_utc."""
    text = (config_root() / "arwen_global_gdas_t255_native_sl_si_24h.toml").read_text(
        encoding="utf-8")
    packaged = 'start_time_utc = "2026-09-01T00:00:00Z", '
    assert packaged in text
    replacement = "" if start_time_utc is None else f'start_time_utc = "{start_time_utc}", '
    path = tmp_path / "native.toml"
    path.write_text(text.replace(packaged, replacement, 1), encoding="utf-8")
    return load_config(path)


def _provenance(valid: str) -> dict:
    return {"mode": "analysis", "valid_time": valid}


# ---------------------------------------------------------------------------
# GI-1 / GP-6: the native suite's clock is the analysis valid time
# ---------------------------------------------------------------------------


def test_a_native_config_dated_to_another_cycle_is_refused_by_name(tmp_path):
    """The shipped T255 config (start_time_utc 2026-09-01T00Z) pointed at a
    12Z analysis of 2026-10-04: the radiation would run twelve hours out of
    phase and on a September declination, and the run said ``pass``."""
    from arwen_global.clock import ClockMismatchError, resolve_forecast_clock

    cfg = _native_analysis_cfg(tmp_path, "2026-09-01T00:00:00Z")
    with pytest.raises(ClockMismatchError) as caught:
        resolve_forecast_clock(cfg, _provenance("2026-10-04 12:00:00"))
    message = str(caught.value)
    assert "2026-09-01T00:00:00Z" in message
    assert "2026-10-04T12:00:00Z" in message
    assert "-804 h" in message and "-12 h out of phase" in message
    assert "Delete start_time_utc" in message


def test_a_native_config_without_the_literal_runs_on_the_analysis_sun(tmp_path):
    from arwen_global.clock import ANALYSIS_VALID_TIME, resolve_forecast_clock
    from arwen_global.runner import build_physics

    cfg = _native_analysis_cfg(tmp_path, None)
    clock = resolve_forecast_clock(cfg, _provenance("2026-10-04T12:00:00.000000000"))
    assert clock.start_utc == datetime(2026, 10, 4, 12, tzinfo=UTC)
    assert clock.source == ANALYSIS_VALID_TIME
    bridge = build_physics(cfg, "numpy", clock)
    assert bridge.adapter_options["start_time_utc"] == "2026-10-04T12:00:00Z"

    # The literal that agrees with the analysis is accepted, and the
    # packaged configs (each pinned to its own cycle) keep running.
    same = _native_analysis_cfg(tmp_path, "2026-10-04T12:00:00Z")
    assert resolve_forecast_clock(
        same, _provenance("2026-10-04 12:00:00")).start_utc == clock.start_utc


def test_a_native_run_with_no_date_anywhere_is_refused():
    """No analysis and no start_time_utc: RRTMGP's declination and hour
    angle would have no instant to run on."""
    from types import SimpleNamespace

    from arwen_global.clock import resolve_forecast_clock

    cfg = SimpleNamespace(initial_mode="analytic", physics_mode="arwen-native",
                          native_adapter_options={})
    with pytest.raises(ValueError, match="needs the real instant"):
        resolve_forecast_clock(cfg, None)


def test_the_fresh_door_lets_the_fetched_cycle_date_the_run(tmp_path):
    """``fresh`` points the base config at the fetched cycle; the base's own
    start_time_utc used to ride along untouched, so the physics radiated on
    the packaged cycle while the observations ran on the fetched one."""
    from arwen_global.clock import resolve_forecast_clock
    from arwen_global.da_door import derive_fresh_config, dump_toml

    base = config_root() / "arwen_global_gdas_t255_native_sl_si_24h.toml"
    document = derive_fresh_config(
        base, analysis_grib="data/gdas.20261004/gdas.t12z.pgrb2.0p25.f000",
        analysis_mapping=None, duration_s=21600.0)
    options = document["physics"]["native_adapter_options"]
    assert "start_time_utc" not in options
    path = tmp_path / "fresh.toml"
    path.write_text(dump_toml(document), encoding="utf-8")
    cfg = load_config(path)
    clock = resolve_forecast_clock(cfg, _provenance("2026-10-04 12:00:00"))
    assert clock.iso == "2026-10-04T12:00:00Z"


def test_the_fresh_door_dates_an_analytic_native_base_with_its_start(tmp_path):
    """An analytic native base (the level-5 smoke, dated 2024-05-21) kept
    its own start_time_utc in the derived config, so the cycle's clock
    check refused every --start-utc but that literal, after the ensemble
    was built.  The derived config now states the --start-utc instant."""
    from datetime import timezone

    from arwen_global.clock import resolve_forecast_clock
    from arwen_global.cycle import reconcile_start_time
    from arwen_global.da_door import derive_fresh_config, dump_toml

    base = config_root() / "arwen_global_level5_native_smoke.toml"
    start = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    document = derive_fresh_config(
        base, analysis_grib=None, analysis_mapping=None, duration_s=21600.0,
        start_utc=start)
    assert (document["physics"]["native_adapter_options"]["start_time_utc"]
            == "2026-10-04T12:00:00Z")
    path = tmp_path / "fresh.toml"
    path.write_text(dump_toml(document), encoding="utf-8")
    clock = resolve_forecast_clock(load_config(path), None)
    assert reconcile_start_time(clock, "2026-10-04T12:00:00Z") == start


# ---------------------------------------------------------------------------
# GI-1: the reference suite (the quickstart's) takes the same dated clock
# ---------------------------------------------------------------------------


def _first_step_theta_increment(cfg, analysis_at, valid: datetime) -> np.ndarray:
    from arwen_global.runner import build_model_and_cold_state, build_transform

    analysis_at(valid)
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    before = np.asarray(transform.inverse(cold.atmosphere.theta))
    after_state, _ = model.step(cold, cfg.dt_s)
    after = np.asarray(transform.inverse(after_state.atmosphere.theta))
    return model, transform, after - before


def _phase_deg(field: np.ndarray, longitude_deg: np.ndarray) -> float:
    """Longitude of the first zonal harmonic's maximum of ``field``."""
    zonal = field.sum(axis=tuple(range(field.ndim - 1)))
    lon = np.deg2rad(longitude_deg)
    return math.degrees(math.atan2(float((zonal * np.sin(lon)).sum()),
                                   float((zonal * np.cos(lon)).sum())))


def test_the_reference_suite_radiates_on_the_analysis_hour(tmp_path, analysis_at):
    """Same atmosphere, analyses valid at 00Z and at 12Z: the step's theta
    increments differ by the shortwave heating alone, and that difference
    peaks where it is noon at 12 UTC (Greenwich), not on the dateline.  The
    undated suite gave the two runs bit-identical increments."""
    cfg = _analysis_cfg(tmp_path)
    assert cfg.physics_mode == "reference"
    model, transform, night = _first_step_theta_increment(
        cfg, analysis_at, datetime(2026, 3, 20, 0))
    assert model.physics.start_utc == datetime(2026, 3, 20, 0, tzinfo=UTC)
    _model, _transform, noon = _first_step_theta_increment(
        cfg, analysis_at, datetime(2026, 3, 20, 12))
    difference = noon - night
    assert np.abs(difference).max() > 0.0, (
        "the 00Z and 12Z analyses heated identically: the reference suite "
        "has no clock")
    longitude = np.asarray(transform.grid.longitude_deg)
    phase = _phase_deg(difference, longitude)
    assert abs(phase) < 30.0, phase


def test_the_idealized_fixture_is_stated_and_unchanged():
    """No analysis, no date: the reference suite keeps the 00 UTC equinox
    clock it always ran, bit for bit, and says so in the receipt."""
    from arwen_global.clock import FIXTURE_STATEMENT, IDEALIZED_FIXTURE
    from arwen_global.physics.reference import ReferencePhysics
    from arwen_global.runner import build_model_and_cold_state, build_transform

    cfg = load_config(config_root() / "arwen_global_moist_smoke.toml")
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(cfg, transform)
    assert model.forecast_clock.source == IDEALIZED_FIXTURE
    assert model.forecast_clock.receipt()["fixture"] == FIXTURE_STATEMENT
    assert model.physics.start_utc is None

    physics = ReferencePhysics("numpy")
    lat = np.deg2rad(np.linspace(-80.0, 80.0, 9))[:, None] * np.ones((1, 16))
    lon = np.deg2rad(np.arange(16) * 22.5)[None, :] * np.ones((9, 1))
    for time_s, dt in ((0.0, 10.0), (43_210.0, 300.0), (90_000.0, 60.0)):
        angle = 2.0 * math.pi * ((time_s + 0.5 * dt) % 86_400.0) / 86_400.0
        legacy = np.maximum(0.0, np.cos(lat) * np.cos(angle + lon - math.pi))
        assert np.array_equal(physics._cosine_zenith(lat, lon, time_s, dt, np), legacy)


def test_the_dated_reference_geometry_is_the_native_suites():
    """The dated reference clock is WRF's radconst geometry, the arithmetic
    the native RRTMGP clock runs: at 2026-06-21 12 UTC the subsolar point
    sits near 23.4 N and near the Greenwich meridian (equation of time
    about -1.2 min)."""
    from arwen_global.physics.reference import ReferencePhysics

    physics = ReferencePhysics("numpy", start_utc=datetime(2026, 6, 21, 12, tzinfo=UTC))
    lat_deg = np.linspace(-89.75, 89.75, 360)
    lon_deg = np.linspace(-179.75, 179.75, 720)
    lat = np.deg2rad(lat_deg)[:, None] * np.ones((1, lon_deg.size))
    lon = np.deg2rad(lon_deg)[None, :] * np.ones((lat_deg.size, 1))
    mu = physics._cosine_zenith(lat, lon, 0.0, 0.0, np)
    row, col = np.unravel_index(int(np.argmax(mu)), mu.shape)
    assert abs(lat_deg[row] - 23.4) < 0.5
    assert abs(lon_deg[col] - 0.30) < 0.5
    assert physics.identity["solar_clock"]["start_utc"] == "2026-06-21T12:00:00Z"


@pytest.mark.parametrize("hour, expected_lon", [(18, -89.70), (6, 90.30)])
def test_the_dated_reference_sun_moves_west_with_the_hour(hour, expected_lon):
    """The 12 UTC check above cannot see the hour angle's sign (a flipped
    sign also peaks at Greenwich then); at 18 UTC the subsolar point is
    near 90 W and at 06 UTC near 90 E, a quarter turn either side."""
    from arwen_global.physics.reference import ReferencePhysics

    physics = ReferencePhysics("numpy", start_utc=datetime(2026, 6, 21, hour, tzinfo=UTC))
    lat_deg = np.linspace(-89.75, 89.75, 360)
    lon_deg = np.linspace(-179.75, 179.75, 720)
    lat = np.deg2rad(lat_deg)[:, None] * np.ones((1, lon_deg.size))
    lon = np.deg2rad(lon_deg)[None, :] * np.ones((lat_deg.size, 1))
    mu = physics._cosine_zenith(lat, lon, 0.0, 0.0, np)
    _row, col = np.unravel_index(int(np.argmax(mu)), mu.shape)
    assert abs(lon_deg[col] - expected_lon) < 0.5


# ---------------------------------------------------------------------------
# The receipt, the checkpoints and the cycle carry the one clock
# ---------------------------------------------------------------------------


@pytest.fixture
def dated_run(tmp_path, analysis_at):
    from arwen_global.runner import run

    analysis_at(datetime(2026, 8, 30, 18))
    cfg = _analysis_cfg(tmp_path)
    outdir = tmp_path / "run"
    run(cfg, outdir)
    return cfg, outdir


def test_the_receipt_and_every_checkpoint_carry_the_analysis_clock(dated_run):
    from arwen_global.checkpoint import read_checkpoint_header
    from arwen_global.clock import FORECAST_CLOCK_KEY

    _cfg, outdir = dated_run
    receipt = json.loads((outdir / "arwen-global-receipt.json").read_text(encoding="utf-8"))
    clock = receipt["forecast_clock"]
    assert clock["start_utc"] == "2026-08-30T18:00:00Z"
    assert clock["source"] == "analysis-valid-time"
    assert clock["analysis_valid_time"] == "2026-08-30T18:00:00Z"
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    assert len(checkpoints) >= 2
    for path in checkpoints:
        header = read_checkpoint_header(path)
        assert header["physics_metadata"][FORECAST_CLOCK_KEY] == {
            "start_utc": "2026-08-30T18:00:00Z", "source": "analysis-valid-time"}


def test_a_restart_from_another_start_is_refused(dated_run, analysis_at, tmp_path):
    """The same config pointed at a later cycle cannot resume a checkpoint
    the earlier analysis started: its fields were heated by another sun."""
    from arwen_global.clock import ClockMismatchError
    from arwen_global.runner import run

    cfg, outdir = dated_run
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    analysis_at(datetime(2026, 8, 31, 0))
    with pytest.raises(ClockMismatchError, match="integrated from 2026-08-30T18:00:00Z"):
        run(cfg, tmp_path / "resumed", restart=checkpoints[1])


def test_the_cycle_start_is_held_to_the_run_clock():
    """``cycle --start-utc`` windows the observations; a value that is not
    the run's clock would window them against one instant while the physics
    and the analyses ran on another."""
    from arwen_global.clock import ANALYSIS_VALID_TIME, ClockMismatchError, ForecastClock
    from arwen_global.cycle import reconcile_start_time

    valid = datetime(2026, 10, 4, 12, tzinfo=UTC)
    clock = ForecastClock(valid, ANALYSIS_VALID_TIME, analysis_valid_time=valid)
    assert reconcile_start_time(clock, None) == valid
    assert reconcile_start_time(clock, "2026-10-04T12:00:00Z") == valid
    with pytest.raises(ClockMismatchError, match="--start-utc 2026-10-04T06:00:00Z"):
        reconcile_start_time(clock, "2026-10-04T06:00:00Z")


def test_build_physics_refuses_an_analysis_config_without_its_clock():
    """No silent fallback to the fixture sun or a literal for an
    analysis-initialised config handed no clock."""
    import pytest as _pytest
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.runner import build_physics

    cfg = load_config(str(config_root() / "arwen_global_t255_quickstart.toml"))
    assert cfg.initial_mode == "analysis" and cfg.physics_mode != "none"
    with _pytest.raises(ValueError, match="no forecast clock"):
        build_physics(cfg, "numpy")


def test_a_cycle_receipt_states_its_start_to_the_readers():
    """The five rewired readers take the start from the receipt; a cycle
    directory's receipt carries it under "cycle", and a fresh-door config
    no longer states the literal."""
    from arwen_global.clock import receipt_start_utc

    got = receipt_start_utc({"cycle": {"start_utc": "2026-09-25T12:00:00+00:00"},
                             "config": {"native_adapter_options": {}}})
    assert got == datetime(2026, 9, 25, 12, tzinfo=UTC)


def test_a_checkpoint_without_a_clock_is_stamped_out_loud(capsys):
    """A checkpoint written before checkpoints carried a clock was stamped
    with the config's clock silently; the stamp now names that it was taken
    on trust, and the operator is told."""
    from types import SimpleNamespace

    from arwen_global.clock import ANALYSIS_VALID_TIME, FORECAST_CLOCK_KEY, ForecastClock
    from arwen_global.runner import adopt_checkpoint_clock

    valid = datetime(2026, 9, 1, tzinfo=timezone.utc)
    model = SimpleNamespace(forecast_clock=ForecastClock(
        valid, ANALYSIS_VALID_TIME, analysis_valid_time=valid))
    state = SimpleNamespace(physics_state=SimpleNamespace(metadata={}))
    adopt_checkpoint_clock(model, state, "restart old.npz")
    record = state.physics_state.metadata[FORECAST_CLOCK_KEY]
    assert record["start_utc"] == "2026-09-01T00:00:00Z"
    assert record["stamped_on_checkpoint_without_clock"] == "restart old.npz"
    assert "carries no forecast clock" in capsys.readouterr().err


def test_an_undated_native_config_receipts_the_gases_of_its_analysis_clock():
    """A config that follows its analysis states no start_time_utc (the
    clock's own refusal tells the reader to delete it).  The run receipt's
    greenhouse-gas record read the date off the config literal alone, so
    every such run integrated its whole forecast and was then refused at
    its receipt ("the native suite has no forecast clock").  The record is
    dated by the clock the physics ran on."""
    import dataclasses

    from arwen_global.clock import ForecastClock
    from arwen_global.runner import _greenhouse_gas_receipt

    cfg = load_config(
        str(config_root() / "arwen_global_gdas_t255_native_sl_si_24h.toml"))
    options = dict(cfg.native_adapter_options)
    options.pop("start_time_utc", None)
    undated = dataclasses.replace(cfg, native_adapter_options=options)
    clock = ForecastClock(
        start_utc=datetime(2026, 10, 2, tzinfo=UTC), source="analysis")
    record = _greenhouse_gas_receipt(undated, clock)
    assert record is not None
    assert "2026" in json.dumps(record)
    # Without a clock an undated config still has no date to give.
    with pytest.raises(ValueError, match="no forecast clock"):
        _greenhouse_gas_receipt(undated, None)
