"""The experimental spectral orography filter (``[initial]
orography_filter_half_degree``, arwen_global.spectral.orography_filter).

Held here: the factor's shape (degree 0 kept, one half at n0, monotone
falling, close to one far below n0); the config key (off by default, a
valid value parsed, a negative, non-finite or unresolved half degree
refused by name, the analytic modes refusing the key, the hash unchanged
while off and moved when on); and the analysis start's terrain step, whose
off path is the unfiltered ``inverse(forward(phi))`` bit for bit and whose
on path is the filtered one.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.analysis_initial import (  # noqa: E402
    _global_regridder,
    analysis_initial_state,
)
from arwen_global.config import load_config  # noqa: E402
from arwen_global.constants import GRAVITY_M_S2  # noqa: E402
from arwen_global.runner import build_transform  # noqa: E402
from arwen_global.spectral.orography_filter import (  # noqa: E402
    filter_orography_coefficients,
    orography_filter_factor,
)
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402

from test_arwen_global_analysis_initial import (  # noqa: E402
    CONFIG, MAPPING, _synthetic_frame,
)

KEY = "orography_filter_half_degree"


# --- the factor ---------------------------------------------------------

@pytest.mark.parametrize("n0", [1, 5, 85, 400])
def test_factor_keeps_degree_zero_and_is_one_half_at_n0(n0):
    factor = orography_filter_factor(1023, n0)
    assert factor.shape == (1024,) and factor.dtype == np.float64
    assert factor[0] == 1.0
    assert factor[n0] == 0.5


def test_factor_falls_monotonically_and_is_one_far_below_n0():
    n0 = 200.0
    factor = orography_filter_factor(1023, n0)
    assert np.all(np.diff(factor) <= 0.0)
    # Strictly falling until it underflows to zero far above n0.
    positive = factor > 0.0
    assert np.all(np.diff(factor[positive]) < 0.0)
    # Close to one far below n0 (n = n0 / 10: 1 - 1e-4 or better) ...
    assert factor[20] > 1.0 - 1.0e-4
    assert factor[1] > 1.0 - 1.0e-8
    # ... and strongly damped above it.
    assert factor[400] < 1.0e-4
    assert 0.0 <= factor.min() and factor.max() == 1.0


def test_factor_matches_its_formula_at_a_fractional_half_degree():
    n0 = 42.5
    factor = orography_filter_factor(100, n0)
    n = np.arange(101, dtype=np.float64)
    expected = np.exp(-math.log(2.0) * (n * (n + 1.0) / (n0 * (n0 + 1.0))) ** 2)
    expected[0] = 1.0
    assert np.array_equal(factor, expected)
    assert factor[42] > 0.5 > factor[43]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_factor_refuses_a_half_degree_that_is_not_finite_and_positive(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        orography_filter_factor(21, bad)


# --- the coefficient step on a NumPy transform --------------------------

def _transform():
    return SphericalHarmonicTransform.create(
        truncation=21, dealias_factor=1.5, backend="numpy", precision="float64")


def _degree_power(coeff):
    return np.sum(np.abs(np.asarray(coeff)) ** 2, axis=1)


def test_off_returns_the_coefficients_untouched():
    transform = _transform()
    rng = np.random.default_rng(3)
    field = rng.standard_normal((transform.grid.nlat, transform.grid.nlon))
    coeff = transform.forward(field)
    kept = filter_orography_coefficients(coeff, transform, 0.0)
    assert kept is coeff
    assert np.array_equal(transform.inverse(kept), transform.inverse(coeff))


def test_on_damps_the_high_degrees_and_keeps_the_mean():
    transform = _transform()
    rng = np.random.default_rng(4)
    field = rng.standard_normal((transform.grid.nlat, transform.grid.nlon))
    coeff = transform.forward(field)
    n0 = 8.0
    filtered = filter_orography_coefficients(coeff, transform, n0)
    factor = orography_filter_factor(transform.truncation, n0)
    # Each degree's coefficients scale by its factor (the field is a
    # projection already, so project() moves nothing else).
    expected = transform.project(coeff * factor[:, None])
    assert np.allclose(filtered, expected, rtol=0.0, atol=1.0e-14)
    before, after = _degree_power(coeff), _degree_power(filtered)
    assert after[0] == pytest.approx(before[0], rel=1.0e-15)
    assert np.allclose(after, before * factor ** 2, rtol=1.0e-12, atol=1.0e-28)
    assert after[transform.truncation] < 1.0e-6 * before[transform.truncation]
    assert after[1] == pytest.approx(before[1], rel=1.0e-2)
    assert not np.array_equal(transform.inverse(filtered), transform.inverse(coeff))


# --- the config key -----------------------------------------------------

def _analysis_text(extra: str = "") -> str:
    """The smoke config as an analysis start (the analysis_initial gates'
    own construction), with ``extra`` added to ``[initial]``."""
    text = Path(CONFIG).read_text(encoding="utf-8")
    replaced = text.replace(
        "[initial]\n",
        "[initial]\nmode = \"analysis\"\n"
        "analysis_grib = \"unused.grib\"\n"
        f"analysis_mapping = \"{MAPPING}\"\n" + extra,
        1,
    )
    kept = []
    for line in replaced.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(key) for key in (
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m",
        )):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n" + '[statics]\nsource = "synthetic"\n'


def _load(tmp_path, extra: str = "", name: str = "analysis.toml"):
    path = tmp_path / name
    path.write_text(_analysis_text(extra), encoding="utf-8")
    return load_config(path)


def test_the_key_is_off_by_default_and_off_keeps_the_hash(tmp_path):
    absent = _load(tmp_path, name="absent.toml")
    assert absent.truncation == 3
    assert absent.orography_filter_half_degree == 0.0
    assert KEY not in absent.config_identity
    for spelling in ("0.0", "0", "-0.0"):
        off = _load(tmp_path, f"{KEY} = {spelling}\n", name=f"off{spelling}.toml")
        assert off.orography_filter_half_degree == 0.0
        assert KEY not in off.config_identity
        assert off.config_identity == absent.config_identity
        assert off.config_hash == absent.config_hash


def test_the_shipped_smoke_config_hash_does_not_see_the_field():
    shipped = load_config(CONFIG)
    assert shipped.orography_filter_half_degree == 0.0
    assert KEY not in shipped.config_identity


@pytest.mark.parametrize("value, parsed", [("2.0", 2.0), ("3", 3.0), ("0.5", 0.5)])
def test_a_valid_half_degree_parses_and_joins_the_hash(tmp_path, value, parsed):
    absent = _load(tmp_path, name="absent.toml")
    on = _load(tmp_path, f"{KEY} = {value}\n", name="on.toml")
    assert on.orography_filter_half_degree == parsed
    assert isinstance(on.orography_filter_half_degree, float)
    assert on.config_identity[KEY] == parsed
    assert on.config_hash != absent.config_hash
    # Everything but the key is the unfiltered config's identity.
    rest = dict(on.config_identity)
    del rest[KEY]
    assert rest == absent.config_identity


def test_two_half_degrees_hash_apart(tmp_path):
    one = _load(tmp_path, f"{KEY} = 2.0\n", name="a.toml")
    two = _load(tmp_path, f"{KEY} = 2.5\n", name="b.toml")
    assert one.config_hash != two.config_hash


@pytest.mark.parametrize("value, match", [
    ("-1.0", "0 < n0 <= truncation"),
    ("-0.5", "0 < n0 <= truncation"),
    ("3.5", "0 < n0 <= truncation"),
    ("4", "0 < n0 <= truncation"),
    ("nan", "must be finite"),
    ("inf", "must be finite"),
    ("-inf", "must be finite"),
    ("true", "must be a finite number"),
])
def test_an_unusable_half_degree_is_refused_by_name(tmp_path, value, match):
    with pytest.raises(ValueError, match=match) as raised:
        _load(tmp_path, f"{KEY} = {value}\n")
    assert KEY in str(raised.value)


def test_an_analytic_start_refuses_the_key(tmp_path):
    text = Path(CONFIG).read_text(encoding="utf-8").replace(
        "[initial]\n", f"[initial]\n{KEY} = 2.0\n", 1)
    path = tmp_path / "analytic.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown keys in \[initial\]: " + KEY):
        load_config(path)


# --- the analysis start's terrain step ----------------------------------

def _unfiltered_phi_model(frame, transform):
    """The shipped terrain step, written out: the regridded analysis
    geopotential through forward and inverse, nothing between."""
    regrid = _global_regridder(frame.latitude, frame.longitude, transform.grid)
    phi_src = GRAVITY_M_S2 * regrid(frame.fields["terrain_height"].values)
    return phi_src, transform.backend.to_numpy(
        transform.inverse(transform.forward(phi_src)))


def test_off_is_the_shipped_terrain_step_bit_for_bit(tmp_path):
    cfg = _load(tmp_path)
    transform = build_transform(cfg)
    frame = _synthetic_frame(terrain_zonal_wavenumber=2)
    _state, phi_model, provenance = analysis_initial_state(cfg, transform, frame=frame)
    _phi_src, expected = _unfiltered_phi_model(frame, transform)
    b = transform.backend
    assert np.array_equal(
        b.to_numpy(phi_model), b.to_numpy(b.asarray(expected, dtype=b.float_dtype)))
    assert "orography_filter" not in provenance


def test_on_filters_the_terrain_the_surface_pressure_moves_onto(tmp_path):
    off_cfg = _load(tmp_path, name="off.toml")
    on_cfg = _load(tmp_path, f"{KEY} = 1.0\n", name="on.toml")
    transform = build_transform(on_cfg)
    frame = _synthetic_frame(terrain_zonal_wavenumber=2)
    off_state, phi_off, _p = analysis_initial_state(off_cfg, transform, frame=frame)
    on_state, phi_on, provenance = analysis_initial_state(on_cfg, transform, frame=frame)
    b = transform.backend
    phi_src, _unfiltered = _unfiltered_phi_model(frame, transform)
    expected = b.to_numpy(transform.inverse(
        filter_orography_coefficients(transform.forward(phi_src), transform, 1.0)))
    assert np.allclose(b.to_numpy(phi_on), expected, rtol=0.0, atol=1.0e-9)
    assert not np.allclose(b.to_numpy(phi_on), b.to_numpy(phi_off))
    # Degree 0 is kept: the global-mean terrain is unchanged.
    mean_on = transform.backend.to_numpy(transform.forward(b.to_numpy(phi_on)))[0, 0]
    mean_off = transform.backend.to_numpy(transform.forward(b.to_numpy(phi_off)))[0, 0]
    assert mean_on == pytest.approx(mean_off, rel=1.0e-12)
    # The surface pressure moved onto the filtered terrain, not the other.
    ps_on = b.to_numpy(on_state.atmosphere.log_surface_pressure)
    ps_off = b.to_numpy(off_state.atmosphere.log_surface_pressure)
    assert not np.array_equal(ps_on, ps_off)
    assert provenance["orography_filter"]["half_degree"] == 1.0
