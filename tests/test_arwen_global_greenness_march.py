"""The greenness march (arwen_global.land_climatology) on known answers.

The march sets vegetation fraction, leaf area index and background albedo
on land to the statics' monthly climatologies at the model day's START,
once per model day.  Until it was its own service it read the instant of
the first step it ran on, so a run started at 18 UTC and restarted 12 h in
took the next UTC day's vegetation: 26.452 restarted against 26.129
straight on a field that is 10 x month (the review's greencheck case, start
2026-04-03 18 UTC).  These tests hold the planes to
``gpuwm.static.build.monthly_interp_to_date`` at the anchor date, count the
updates, and make that restart bitwise.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from gpuwm.static.build import monthly_interp_to_date

from arwen_global.land_climatology import GreennessMarch

START = datetime(2026, 4, 3, 18, tzinfo=timezone.utc)
SHAPE = (4, 6)


def _fields():
    months = np.arange(12, dtype=np.float64)[:, None, None] * np.ones((12, *SHAPE))
    return {"GREENFRAC": 0.05 * months, "LAI12M": 10.0 * months,
            "ALBEDO12M": 10.0 + months}


def _state(t):
    land = np.ones(SHAPE, np.float32)
    land[0] = 0.0  # one row of water, never written
    surface = SimpleNamespace(
        temperature_k=np.full(SHAPE, 280.0, np.float32), land_fraction=land,
        vegetation_fraction=np.full(SHAPE, -1.0, np.float32),
        leaf_area_index=np.full(SHAPE, -1.0, np.float32),
        background_albedo=np.full(SHAPE, -1.0, np.float32))
    return SimpleNamespace(surface=surface, time_s=float(t))


def test_the_planes_are_the_climatology_at_the_model_day_start():
    march = GreennessMarch(_fields(), start_time=START)
    state = _state(6 * 3600.0)
    march.apply(state)
    anchor = START.replace(tzinfo=None)
    expect = np.clip(monthly_interp_to_date(_fields()["LAI12M"], anchor), 0.0, None)
    np.testing.assert_array_equal(state.surface.leaf_area_index[1:],
                                  expect[1:].astype(np.float32))
    assert (state.surface.leaf_area_index[0] == -1.0).all()
    green = np.clip(monthly_interp_to_date(_fields()["GREENFRAC"], anchor), 0.0, 1.0)
    np.testing.assert_array_equal(state.surface.vegetation_fraction[1:],
                                  green[1:].astype(np.float32))
    albedo = np.clip(monthly_interp_to_date(_fields()["ALBEDO12M"], anchor) / 100.0, 0.0, 1.0)
    np.testing.assert_array_equal(state.surface.background_albedo[1:],
                                  albedo[1:].astype(np.float32))


def test_the_march_fires_once_per_model_day():
    march = GreennessMarch(_fields(), start_time=START)
    state = _state(0.0)
    for t in (0.0, 300.0, 6 * 3600.0, 86399.0):
        state.time_s = t
        march.apply(state)
    assert march.updates == 1
    state.time_s = 86400.0
    march.apply(state)
    assert march.updates == 2
    assert march.last_date == (START + timedelta(days=1)).isoformat().replace("+00:00", "Z")


def test_restarts_at_6_and_12_hours_from_18_utc_are_bitwise():
    straight = GreennessMarch(_fields(), start_time=START)
    a = _state(0.0)
    for t in np.arange(0.0, 18 * 3600.0, 300.0):
        a.time_s = float(t)
        straight.apply(a)
    for restart_h in (6, 12):
        resumed = GreennessMarch(_fields(), start_time=START)
        b = _state(0.0)
        for t in np.arange(restart_h * 3600.0, 18 * 3600.0, 300.0):
            b.time_s = float(t)
            resumed.apply(b)
        for name in ("vegetation_fraction", "leaf_area_index", "background_albedo"):
            np.testing.assert_array_equal(getattr(a.surface, name),
                                          getattr(b.surface, name))
    # The pre-fix rule (the instant of the first step) would have read the
    # next UTC day after the 12 h restart: the value the review measured.
    instant = monthly_interp_to_date(_fields()["LAI12M"],
                                     (START + timedelta(hours=12)).replace(tzinfo=None))
    assert abs(float(instant[0, 0]) - 26.452) < 5e-3
    assert abs(float(a.surface.leaf_area_index[1, 0]) - 26.129) < 5e-3
