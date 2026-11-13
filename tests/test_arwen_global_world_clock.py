"""Package 3.1 of the other-worlds track: a world's day.

CPU only.  The sun's hour angle and the dynamics' rotation rate come from one
function (``arwen_global.world.clock.planet_rates``); a 24-hour day returns
the model's own numbers to the bit and builds nothing; the radiation-interval
advice restores the config's degrees of sun per call; the two local-solar-time
instruments refuse a run on a world whose day is not 24 hours, by name.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.spectral.constants import EARTH_ROTATION_RATE_S
from arwen_global.world import EARTH, parse_world
from arwen_global.world.clock import (
    ORBITAL_RATE_S,
    LocalTimeRefused,
    PlanetClock,
    checkpoint_day_length_h,
    metadata_day_length_h,
    planet_rates,
    receipt_day_length_h,
    record_day_length_h,
    refuse_local_time,
    rotation_rate_of,
)


def _world(sun: str, label: str = "fast"):
    import tomllib

    text = f'[world]\nlabel = "{label}"\n[world.sun]\n{sun}\n'
    return parse_world(tomllib.loads(text)["world"])


TWELVE = _world("day_length_h = 12.0")


# ------------------------------------------------------------- Earth is Earth


def test_a_24_hour_day_is_the_models_own_numbers_to_the_bit():
    solar, spin = planet_rates(24.0)
    assert spin == EARTH_ROTATION_RATE_S
    assert solar == 2.0 * math.pi / 86400.0
    assert rotation_rate_of(EARTH) == EARTH_ROTATION_RATE_S
    assert rotation_rate_of(None) == EARTH_ROTATION_RATE_S


def test_no_clock_is_built_for_earth_or_for_a_world_that_keeps_the_day():
    assert PlanetClock.from_world(None) is None
    assert PlanetClock.from_world(EARTH) is None
    assert PlanetClock.from_world(_world("day_length_h = 24.0", "earth-again")) is None
    tilted = _world("obliquity_deg = 90.0", "on-its-side")
    assert not tilted.is_earth
    assert PlanetClock.from_world(tilted) is None
    assert rotation_rate_of(tilted) == EARTH_ROTATION_RATE_S
    assert PlanetClock().extra_turn_deg(123456.0) == 0.0


# --------------------------------------------------------- one function, two uses


def test_the_spin_is_the_solar_rate_plus_the_models_orbital_rate():
    clock = PlanetClock.from_world(TWELVE)
    assert clock.day_length_h == 12.0
    solar, spin = planet_rates(12.0)
    assert clock.rotation_rate_s == spin == solar + ORBITAL_RATE_S
    assert solar == 2.0 * math.pi / 43200.0
    # The model's orbital rate is its Earth spin less its 24-hour solar rate:
    # about a 367-day year, positive.
    assert 1.9e-7 < ORBITAL_RATE_S < 2.0e-7
    # Not 24 / day_length_h times Earth's spin: 0.14 percent apart at 12 h.
    ratio = clock.rotation_rate_s / (TWELVE.sun.rotation_factor * EARTH_ROTATION_RATE_S)
    assert 0.998 < ratio < 0.9988
    # A slow spinner's day is mostly its orbit.
    slow = PlanetClock(2400.0)
    assert slow.rotation_rate_s > ORBITAL_RATE_S > 0.0


@pytest.mark.parametrize("day_h", [12.0, 18.0, 36.0, 96.0, 2400.0])
def test_the_sun_returns_to_the_same_meridian_after_exactly_one_day(day_h):
    """Earth's hour angle advances 360 degrees per 24 h of the calendar; the
    world's extra turn on top of it must make a whole turn per world day."""
    clock = PlanetClock(day_h)
    for days in (1, 2, 5):
        elapsed = days * day_h * 3600.0
        earth = 360.0 * elapsed / 86400.0
        total = (earth + clock.extra_turn_deg(elapsed)) % 360.0
        assert min(total, 360.0 - total) < 1e-6, (day_h, days, total)
    # Half a day on, the sun is on the far side.
    half = 0.5 * day_h * 3600.0
    total = (360.0 * half / 86400.0 + clock.extra_turn_deg(half)) % 360.0
    assert abs(total - 180.0) < 1e-6


def test_the_sub_solar_longitude_is_the_tapes_formula():
    """wrf_export.subsolar_point (package 1.4) turns Earth's sub-solar
    longitude by 360 x hours x (1/day - 1/24) degrees west; this clock's
    extra turn is the same angle, so the tape and the radiation agree."""
    for day_h in (12.0, 30.0):
        clock = PlanetClock(day_h)
        for seconds in (0.0, 3600.0, 10800.0, 172800.0, 3.1e6):
            for earth_lon in (-179.0, -12.5, 0.0, 88.0, 179.9):
                tape = earth_lon - 360.0 * (seconds / 3600.0) * (1.0 / day_h - 1.0 / 24.0)
                tape = (tape + 180.0) % 360.0 - 180.0
                mine = clock.subsolar_longitude_deg(earth_lon, seconds)
                gap = abs(((mine - tape) + 180.0) % 360.0 - 180.0)
                assert gap < 1e-7, (day_h, seconds, earth_lon, mine, tape)


def test_local_hour_on_the_worlds_own_clock():
    clock = PlanetClock(12.0)
    # At the start a place at Earth's local noon is at the world's noon.
    assert clock.local_solar_hour(12.0, 0.0) == pytest.approx(6.0)
    # Three hours on, Earth's local hour there is 15; the world's sun has
    # turned a quarter of its day: 6 + 3 = 9 of 12.
    assert clock.local_solar_hour(15.0, 3 * 3600.0) == pytest.approx(9.0)


# ------------------------------------------------------- the radiation interval


def test_the_radiation_interval_advice_restores_the_configs_degrees_per_call():
    advice = PlanetClock(12.0).radiation_interval_advice(3000.0, 300.0)
    assert advice["degrees_per_call"] == pytest.approx(25.0)
    assert advice["earth_degrees_per_call"] == pytest.approx(12.5)
    assert advice["recommended_radiation_interval_s"] == 1500.0
    assert advice["recommended_degrees_per_call"] == pytest.approx(12.5)
    assert "radiation_interval_s = 1500" in advice["sentence"]
    # Rounded DOWN to whole steps: 3000 x 18 / 24 = 2250 -> 7 steps of 300.
    assert PlanetClock(18.0).radiation_interval_advice(
        3000.0, 300.0)["recommended_radiation_interval_s"] == 2100.0
    # Never below one step: 600 s on a 12-hour day of 300 s steps gives 300,
    # and an interval that is already one step has nothing to recommend.
    assert PlanetClock(12.0).radiation_interval_advice(
        600.0, 300.0)["recommended_radiation_interval_s"] == 300.0
    assert PlanetClock(12.0).radiation_interval_advice(300.0, 300.0) is None
    # A day of 24 hours or longer already moves the sun no further per call.
    assert PlanetClock(24.0).radiation_interval_advice(3000.0, 300.0) is None
    assert PlanetClock(48.0).radiation_interval_advice(3000.0, 300.0) is None


def test_the_run_door_hands_the_dynamics_the_worlds_spin():
    """``runner._world_spin`` arrives with w3-1-for-w1-2.patch; until the
    integrator applies it this test has nothing to call."""
    from arwen_global import runner

    if not hasattr(runner, "_world_spin"):
        pytest.skip("runner._world_spin arrives with w3-1-for-w1-2.patch")
    cfg = SimpleNamespace(physics_mode="arwen-native", dt_s=300.0,
                          native_adapter_options={"radiation_interval_s": 3000.0})
    assert runner._world_spin(cfg, EARTH) == {}
    spin = runner._world_spin(cfg, TWELVE)
    assert spin == {"rotation_rate_s": PlanetClock(12.0).rotation_rate_s}


# --------------------------------------------------- the local-time refusals


def test_the_readers_take_24_hours_from_earth_and_from_old_records():
    assert record_day_length_h(None) == 24.0
    assert record_day_length_h(EARTH.record()) == 24.0
    assert record_day_length_h(TWELVE.record()) == 12.0
    assert metadata_day_length_h({"physics_metadata": {}}) == 24.0
    assert metadata_day_length_h({}) == 24.0
    assert metadata_day_length_h({"physics_metadata": {"world": TWELVE.record()}}) == 12.0
    assert receipt_day_length_h({}) == 24.0
    assert receipt_day_length_h({"world": TWELVE.record()}) == 12.0
    # The config-shaped stand-in receipt carries config_identity, whose
    # "world" is the world's identity (present only for a non-Earth world).
    assert receipt_day_length_h({"config": {"world": TWELVE.identity}}) == 12.0
    assert receipt_day_length_h({"config": {"truncation": 3}}) == 24.0
    refuse_local_time("diurnal_phase", 24.0, "anything")  # quiet


def _checkpoint(path, world_record=None, step=1):
    physics = {} if world_record is None else {"world": world_record}
    metadata = {"step": step, "time_s": 3600.0 * step, "physics_metadata": physics}
    np.savez(path, __metadata__=np.asarray(json.dumps(metadata)))
    return path


def test_the_diurnal_phase_instrument_refuses_a_12_hour_world(tmp_path):
    from arwen_global.diurnal_phase import read_checkpoint_series

    paths = [_checkpoint(tmp_path / f"arwen_global_step{n:08d}.npz", TWELVE.record(), n)
             for n in range(1, 6)]
    with pytest.raises(LocalTimeRefused, match=r"diurnal_phase refused: .* 12 hours.*24-hour day"):
        read_checkpoint_series(paths, start_utc_s=0.0)
    assert checkpoint_day_length_h(paths[0]) == 12.0
    # An Earth checkpoint (no world record) is not refused by the clock: it
    # reaches the instrument's own reading, which this stub cannot satisfy.
    earth = [_checkpoint(tmp_path / f"e{n}.npz", None, n) for n in range(1, 6)]
    assert checkpoint_day_length_h(earth[0]) == 24.0
    with pytest.raises(Exception) as caught:
        read_checkpoint_series(earth, start_utc_s=0.0)
    assert not isinstance(caught.value, LocalTimeRefused)


def test_the_trigger_diagnostic_refuses_a_12_hour_world(tmp_path):
    from arwen_global.trigger_diagnostic import measure_run

    run = tmp_path / "run"
    run.mkdir()
    (run / "arwen-global-receipt.json").write_text(json.dumps({"world": TWELVE.record()}))
    with pytest.raises(LocalTimeRefused, match=r"trigger_diagnostic refused: .* 12 hours"):
        measure_run(run, start_utc_s=0.0)
    # A receipt that names no world, over checkpoints that do: refused at
    # the checkpoints.
    (run / "arwen-global-receipt.json").write_text(json.dumps({}))
    for n in range(1, 3):
        _checkpoint(run / f"arwen_global_step{n:08d}.npz", TWELVE.record(), n)
    with pytest.raises(LocalTimeRefused, match=r"arwen_global_step00000001\.npz"):
        measure_run(run, start_utc_s=0.0)
