"""The reduced radiation grid's maps (physics.radiation_grid), on the host."""
from types import SimpleNamespace

import numpy as np
import pytest

from arwen_global.physics.native_options import NativePhysicsOptions
from arwen_global.physics.radiation_grid import (
    FULL_GRID, ReducedRadiationGrid, full_grid_result, validate_strides)

F = np.float32


def _grid(ny=9, nx=16, sy=2, sx=2):
    return ReducedRadiationGrid(np, ny, nx, sy, sx)


def test_sampled_columns_come_back_exactly():
    grid = _grid()
    rng = np.random.default_rng(3)
    full = rng.standard_normal((4, 9, 16)).astype(F)
    reduced = grid.sub(full)
    assert reduced.shape == (4, 5, 8)
    back = grid.linear(reduced)
    assert back.shape == full.shape and back.dtype == F
    np.testing.assert_array_equal(back[:, ::2, ::2], full[:, ::2, ::2])


def test_linear_in_index_is_reproduced_and_longitude_wraps():
    grid = _grid(ny=9, nx=16)
    j, i = np.meshgrid(np.arange(9), np.arange(16), indexing="ij")
    field = (3.0 * j + 0.5 * i).astype(F)
    back = grid.linear(grid.sub(field))
    np.testing.assert_allclose(back[:, :15], field[:, :15], rtol=0, atol=1e-5)
    # the last longitude interpolates toward column 0 (periodic)
    np.testing.assert_allclose(back[:, 15], 0.5 * (field[:, 14] + field[:, 0]), atol=1e-5)


def test_odd_row_count_holds_the_last_reduced_row():
    grid = _grid(ny=8, nx=8)
    field = np.arange(64, dtype=F).reshape(8, 8)
    back = grid.linear(grid.sub(field))
    np.testing.assert_array_equal(back[7, ::2], field[6, ::2])


def test_shortwave_is_dark_where_the_model_column_is_dark():
    grid = _grid(ny=4, nx=8)
    mu_full = np.tile(np.array([0.6, 0.4, 0.2, 0.05, -0.1, -0.3, -0.2, 0.3], F), (4, 1))
    mu_r = grid.sub(mu_full)
    swdnt_r = np.where(mu_r > 0, F(1361.0) * mu_r, F(0.0)).astype(F)
    plane = lambda value: np.full((2, 4), value, F)  # noqa: E731
    result = SimpleNamespace(
        coszen=mu_r, swdown=swdnt_r * F(0.7), swdnt=swdnt_r, swupt=swdnt_r * F(0.3),
        rthratensw=np.stack([swdnt_r * F(1e-8)] * 3), rthratenlw=np.zeros((3, 2, 4), F),
        glw=plane(300.0), olr=plane(240.0), cldfra_total=plane(0.5))
    out = full_grid_result(
        grid, result, coszen_full=mu_full, albedo=np.full((4, 8), 0.2, F),
        emissivity=np.full((4, 8), 0.98, F), skin_temperature=np.full((4, 8), 288.0, F),
        stefan_boltzmann=5.670374419e-8)
    dark = mu_full <= 0
    for name in ("swdown", "swdnt", "swupt", "gsw"):
        assert np.all(getattr(out, name)[dark] == 0.0), name
    assert np.all(out.rthratensw[:, dark] == 0.0)
    lit = ~dark
    # the incident top flux is S0 * mu on every lit model column
    np.testing.assert_allclose(out.swdnt[lit], 1361.0 * mu_full[lit], rtol=1e-5)
    np.testing.assert_allclose(out.gsw, out.swdown * F(0.8), rtol=1e-6)
    np.testing.assert_allclose(out.lwupb, 0.98 * 5.670374419e-8 * 288.0 ** 4 + 0.02 * 300.0, rtol=1e-5)
    assert np.array_equal(out.coszen, mu_full)


def test_size_bounding_columns_take_the_nearest_reduced_column():
    grid = _grid(ny=4, nx=8)
    values = np.arange(8, dtype=F)
    out = grid.nearest(values)
    assert out.shape == (4, 8)
    assert out[1, 1] == values[0] and out[3, 7] == values[7]


def test_full_grid_is_the_identity_and_the_default_leaves_the_identity():
    value = np.ones((3, 4, 8), F)
    assert FULL_GRID.sub(value) is value and not FULL_GRID.reduced
    fields = NativePhysicsOptions.__dataclass_fields__
    assert fields["radiation_stride_lat"].default == 1
    assert fields["radiation_stride_lon"].default == 1


@pytest.mark.parametrize("lat,lon", [(0, 1), (1, 9), (2.0, 2), (True, 1)])
def test_strides_are_whole_numbers_one_to_eight(lat, lon):
    with pytest.raises(ValueError):
        validate_strides(lat, lon)


def test_a_stride_that_does_not_divide_the_longitudes_is_refused():
    with pytest.raises(ValueError):
        ReducedRadiationGrid(np, 4, 10, 1, 3)
