"""A layer deeper than the RRTMGP tables is looked up at the tables' deepest pressure.

RRTMGP v1.9's gas tables end at 109,663.316 Pa of layer pressure.  The T127
year run of 2026-11-13 (coupled slab, the albedo pair) stopped at day 192.8,
step 27769, with ``play range [120.168411, 109701.883] Pa is outside allowed
range [1.00518357, 109663.316] Pa``: the T127 terrain rings down to -101 m on
the south-east Caspian shore (37.9 N 53.4 E) and a winter high sat on it.
The `side` world of the same night stopped the same way at 109,667.2 Pa.

These rows hold the fix (core/rrtmgp.py RRTMGP_PLAY_CEILING_PA); the rows that
need a card are in test_rrtmgp_pressure_ceiling_cuda.py:

* a layer between the tables' deepest pressure and 1,150 hPa reads exactly the
  interpolation coordinates the deepest pressure reads, on the device and in the
  float64 mirror, and the troposphere test keeps the true pressure;
* a layer inside the tables reads, bit for bit, what the untouched inline
  oracle kernel reads, so no run that stays inside the tables moves;
* the range gate now ends at 1,150 hPa, and still refuses past it.
"""
from __future__ import annotations

import numpy as np
import pytest

from arwen_global.core import npref, rrtmgp

DEEPEST_SEEN_PA = 109_701.883  # the year run's stop, 2026-11-13


def _tables(kind):
    try:
        return rrtmgp.load_gas_tables(kind)
    except (OSError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"RRTMGP tables not available: {exc}")


def _profile(tables):
    """Four columns, six layers bottom-to-top; the bottom layers straddle
    the tables' deepest pressure."""
    top = float(np.max(tables.press_ref))
    bottom = np.array([DEEPEST_SEEN_PA, top, 114_999.0, 100_000.0])
    above = np.array([90_000.0, 70_000.0, 50_000.0, 20_000.0, 5_000.0])
    play = np.empty((4, 6))
    play[:, 0] = bottom
    play[:, 1:] = above
    plev = np.empty((4, 7))
    plev[:, 0] = bottom + 300.0
    plev[:, 1:-1] = 0.5 * (play[:, :-1] + play[:, 1:])
    plev[:, -1] = 1_000.0
    tlay = np.tile(np.array([285.0, 278.0, 262.0, 245.0, 218.0, 225.0]),
                   (4, 1))
    vmr = np.zeros((4, 6, tables.ngas + 1))
    for gas, value in (("h2o", 8.0e-3), ("co2", 4.2e-4), ("o3", 3.0e-8),
                       ("n2o", 3.3e-7), ("ch4", 1.9e-6), ("o2", 0.209),
                       ("n2", 0.781), ("co", 1.0e-7)):
        if gas in tables.gas_index:
            vmr[:, :, tables.gas_index[gas]] = value
    return play, plev, tlay, vmr


def test_the_accepted_range_runs_from_the_table_floor_to_1150_hpa():
    lw, sw = _tables("lw"), _tables("sw")
    lower, upper = rrtmgp._play_bounds(lw, sw)
    assert lower == float(np.min(lw.press_ref)) == rrtmgp.RRTMGP_TOA_PRESSURE_PA
    assert upper == rrtmgp.RRTMGP_PLAY_CEILING_PA == 115_000.0
    # The ceiling sits past the tables, and past the deepest layer seen.
    assert float(np.max(lw.press_ref)) < DEEPEST_SEEN_PA < upper


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_the_float64_mirror_looks_a_deep_layer_up_at_the_last_pressure(kind):
    tables = _tables(kind)
    play, plev, tlay, vmr = _profile(tables)
    top = float(np.max(tables.press_ref))
    col_dry = npref.np_rrtmgp_col_dry(vmr[:, :, tables.gas_index["h2o"]], plev)
    col_gas = vmr * col_dry[:, :, None]
    col_gas[:, :, 0] = col_dry
    deep = npref._rrtmgp_interpolation(tables, play, tlay, col_gas)
    clipped = play.copy()
    clipped[:, 0] = np.minimum(clipped[:, 0], top)
    at_top = npref._rrtmgp_interpolation(tables, clipped, tlay, col_gas)
    for a, b in zip(deep, at_top):
        np.testing.assert_array_equal(a, b)
    # The table descends from the surface: the deep layers sit on its
    # first (deepest) level with the full weight, none past it.
    assert float(tables.press_ref[0]) == top
    jtemp, jpress, tropo = deep[:3]
    assert np.all(jpress[:3, 0] == 0)
    fmajor = deep[6]
    assert np.all(fmajor[:, 1, :, :3, 0, :] == 0.0)  # no weight on level 2
    assert np.all(tropo[:, 0])
    # The mirror's gas optics accept the stop's pressure ...
    out = npref.np_rrtmgp_gas_optics(tables, play, plev, tlay, vmr)
    assert np.all(np.isfinite(out.tau))
    # ... and refuse past the ceiling.
    bad = play.copy()
    bad[0, 0] = 115_001.0
    with pytest.raises(ValueError, match="outside allowed range"):
        npref.np_rrtmgp_gas_optics(tables, bad, plev + 2_000.0, tlay, vmr)

