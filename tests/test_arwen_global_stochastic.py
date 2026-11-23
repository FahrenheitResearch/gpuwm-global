"""Stochastic physics (arwen_global.stochastic): SPPT and SKEB.

* The noise is Philox4x32-10: the Random123 known-answer vectors.
* The pattern law: unit point variance, the Gaussian correlation length and
  the AR(1) decorrelation time each match their settings, measured.
* A member's patterns are a pure function of (member, seed, clock).
* Off is the shipped run: same config hash, same bits.
* On, a run is reproducible, members differ, and a restart is bit exact.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.checkpoint import read_checkpoint  # noqa: E402
from arwen_global.config import load_config  # noqa: E402
from arwen_global.configs_dir import config_root as _shipped_configs  # noqa: E402
from arwen_global.runner import run  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from arwen_global.stochastic import (  # noqa: E402
    CLOCK,
    COEFFICIENTS,
    StochasticConfig,
    StochasticPhysics,
    complex_noise,
    philox4x32,
)

SMOKE = _shipped_configs() / "arwen_global_moist_smoke.toml"


def _config_text(table: str | None, *, truncation: int = 21) -> str:
    text = SMOKE.read_text(encoding="utf-8")
    text = text.replace("truncation = 3", f"truncation = {truncation}")
    text = text.replace("duration_s = 40.0", "duration_s = 60.0")
    return text + ("" if table is None else "\n[stochastic]\n" + table)


def _write(tmp_path, name, table, **kw):
    path = tmp_path / f"{name}.toml"
    path.write_text(_config_text(table, **kw), encoding="utf-8")
    return load_config(path)


# ---------------------------------------------------------------- noise


@pytest.mark.parametrize("counter,key,expected", [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF),
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
])
def test_philox_matches_the_random123_known_answers(counter, key, expected):
    words = philox4x32(*(np.uint64(c) for c in counter), key[0] | (key[1] << 32))
    assert tuple(int(w) for w in words) == expected


def test_noise_is_unit_complex_gaussian_on_the_triangle():
    noise = np.stack([complex_noise(40, 1234, d) for d in range(400)])
    tri = np.tri(41, dtype=bool)
    assert np.all(noise[:, ~tri] == 0)
    m0 = noise[:, 1:, 0]
    assert np.all(m0.imag == 0)
    assert abs(np.var(m0.real) - 1.0) < 0.04
    rows, cols = np.tril_indices(41, -1)
    keep = cols > 0  # m > 0 only; the m = 0 column is real
    off = noise[:, rows[keep], cols[keep]]
    assert abs(np.mean(np.abs(off) ** 2) - 1.0) < 0.02
    assert abs(np.var(off.real) - 0.5) < 0.02
    assert abs(np.mean(off.real * off.imag)) < 0.01


# ---------------------------------------------------------------- pattern law


def _single_scale(length_m, time_s, **kw):
    return StochasticConfig(sppt=True, sppt_scales=((1.0, length_m, time_s),), **kw)


def test_pattern_point_variance_and_correlation_length_match_the_settings():
    """500 km requested: the correlation along a meridian falls to
    exp(-1/2) at 500 km (5 %), and the point variance is one (3 %)."""
    transform = SphericalHarmonicTransform.create(170, backend="numpy", precision="float64")
    length = 500.0e3
    radius = transform.grid.radius_m
    lat = transform.grid.lat_rad
    fields = []
    for member in range(160):
        sp = StochasticPhysics(_single_scale(length, 21600.0, member=member), transform)
        coeff, clock = sp._empty_state()
        coeff, clock, _ = sp.advance(coeff, clock, 0.0)
        fields.append(np.asarray(transform.inverse(sp.spectral(coeff, 0))))
    f = np.stack(fields)                     # (draws, nlat, nlon)
    band = np.abs(np.rad2deg(lat)) < 60
    assert abs(np.mean(f[:, band] ** 2) - 1.0) < 0.03
    rows = np.flatnonzero(band)
    dlat = abs(lat[1] - lat[0])
    distances, correlations = [], []
    for k in range(1, 12):
        a = f[:, rows[:-k]]
        b = f[:, rows[:-k] + k]
        corr = np.mean(a * b) / math.sqrt(np.mean(a * a) * np.mean(b * b))
        distances.append(k * dlat * radius)
        correlations.append(corr)
    distances = np.array(distances)
    correlations = np.array(correlations)
    target = math.exp(-0.5)
    j = int(np.flatnonzero(correlations < target)[0])
    x0, x1 = distances[j - 1], distances[j]
    y0, y1 = correlations[j - 1], correlations[j]
    measured = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    assert abs(measured / length - 1.0) < 0.05, measured


def test_pattern_decorrelation_time_matches_the_setting():
    """Lag-one correlation of the coefficients exp(-dt/tau) (6 h, dt 1 h)."""
    transform = SphericalHarmonicTransform.create(63, backend="numpy", precision="float64")
    tau, dt = 21600.0, 3600.0
    sp = StochasticPhysics(_single_scale(1000.0e3, tau), transform)
    coeff, clock = sp._empty_state()
    series = []
    for k in range(400):
        coeff, clock, redrawn = sp.advance(coeff, clock, k * dt)
        assert redrawn
        series.append(coeff[0].copy())
    s = np.stack(series)                      # (time, 2, n, n)
    a, b = s[:-1].ravel(), s[1:].ravel()
    keep = (a != 0) | (b != 0)
    lag1 = np.sum(a[keep] * b[keep]) / math.sqrt(np.sum(a[keep] ** 2) * np.sum(b[keep] ** 2))
    assert abs(lag1 - math.exp(-dt / tau)) < 0.01, lag1
    # And the AR(1) keeps the stationary variance it started with.
    first, last = s[:50].var(), s[-50:].var()
    assert abs(last / first - 1.0) < 0.15


def test_the_same_clock_draws_nothing_and_the_clock_never_runs_back():
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    sp = StochasticPhysics(StochasticConfig(sppt=True), transform)
    coeff, clock = sp._empty_state()
    coeff, clock, _ = sp.advance(coeff, clock, 600.0)
    again, clock2, redrawn = sp.advance(coeff, clock, 600.0)
    assert not redrawn and again is coeff and clock2 is clock
    with pytest.raises(ValueError, match="backwards"):
        sp.advance(coeff, clock, 0.0)


def test_patterns_are_a_function_of_member_seed_and_clock():
    transform = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")

    def draws(**kw):
        sp = StochasticPhysics(StochasticConfig(sppt=True, skeb=True, **kw), transform)
        coeff, clock = sp._empty_state()
        out = []
        for t in (0.0, 600.0, 1200.0):
            coeff, clock, _ = sp.advance(coeff, clock, t)
            out.append(coeff)
        return np.stack(out)

    base = draws(member=3, seed=53)
    assert np.array_equal(base, draws(member=3, seed=53))
    assert not np.allclose(base, draws(member=4, seed=53))
    assert not np.allclose(base, draws(member=3, seed=54))


def test_the_three_scale_sppt_pattern_has_the_ecmwf_amplitude():
    """Default scales 0.52/0.18/0.06: point standard deviation sqrt(sum a^2)."""
    transform = SphericalHarmonicTransform.create(106, backend="numpy", precision="float64")
    values = []
    for member in range(60):
        sp = StochasticPhysics(StochasticConfig(sppt=True, member=member), transform)
        coeff, clock = sp._empty_state()
        coeff, clock, _ = sp.advance(coeff, clock, 0.0)
        values.append(np.asarray(transform.inverse(sp.sppt_spectral(coeff))))
    expected = math.sqrt(0.52 ** 2 + 0.18 ** 2 + 0.06 ** 2)
    assert abs(np.std(np.stack(values)) / expected - 1.0) < 0.06


# ---------------------------------------------------------------- the run


def _final(outdir, step=6):
    return read_checkpoint(outdir / f"arwen_global_step{step:08d}.npz")


def test_off_is_the_shipped_run_same_hash_same_bits(tmp_path):
    plain = _write(tmp_path, "plain", None)
    off = _write(tmp_path, "off", "sppt = false\nskeb = false\nmember = 7\n")
    assert off.stochastic is None
    assert off.config_hash == plain.config_hash
    run(plain, tmp_path / "plain")
    run(off, tmp_path / "off")
    _, a = _final(tmp_path / "plain")
    _, b = _final(tmp_path / "off")
    assert a.keys() == b.keys()
    assert not any(name.startswith("physics__stochastic_") for name in a)
    for name in a:
        assert np.array_equal(a[name], b[name]), name


ON = "sppt = true\nskeb = true\nmember = {member}\nseed = 53\n"


def test_on_is_reproducible_members_differ_and_restart_is_exact(tmp_path):
    plain = _write(tmp_path, "plain", None)
    m0 = _write(tmp_path, "m0", ON.format(member=0))
    m1 = _write(tmp_path, "m1", ON.format(member=1))
    assert len({plain.config_hash, m0.config_hash, m1.config_hash}) == 3
    run(plain, tmp_path / "plain")
    result = run(m0, tmp_path / "m0")
    run(m0, tmp_path / "m0-again")
    run(m1, tmp_path / "m1")
    _, a = _final(tmp_path / "m0")
    _, again = _final(tmp_path / "m0-again")
    _, other = _final(tmp_path / "m1")
    _, base = _final(tmp_path / "plain")
    for name in a:
        assert np.array_equal(a[name], again[name]), name
    clock = a["physics__" + CLOCK]
    assert clock[1] >= 6  # one draw per step at least
    assert a["physics__" + COEFFICIENTS].dtype == np.float64
    moved = [n for n in base if not np.array_equal(base[n], a[n])]
    assert any("vorticity" in n for n in moved) and any("theta" in n or "temperature" in n
                                                         for n in moved), moved
    assert any(not np.array_equal(a[n], other[n]) for n in a if "vorticity" in n)
    # Restart from the step-2 checkpoint: bit exact at step 6.
    run(m0, tmp_path / "resumed", restart=tmp_path / "m0" / "arwen_global_step00000002.npz")
    meta_a, full = _final(tmp_path / "m0")
    meta_b, resumed = _final(tmp_path / "resumed")
    assert full.keys() == resumed.keys()
    for name in full:
        assert np.array_equal(full[name], resumed[name]), name
    assert result is not None


def test_stochastic_table_refuses_unknown_keys_and_bad_values(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        _write(tmp_path, "bad", "sppt = true\nampltude = 2\n")
    with pytest.raises(ValueError, match="clip"):
        _write(tmp_path, "clip", "sppt = true\nclip = 1.5\n")
    with pytest.raises(ValueError, match="together"):
        _write(tmp_path, "scales", "sppt = true\nsppt_amplitudes = [0.5]\n")
    cfg = _write(tmp_path, "scales-ok", "sppt = true\nsppt_amplitudes = [0.8]\n"
                 "sppt_lengths_m = [450000.0]\nsppt_times_s = [21600.0]\nlogit = true\n")
    assert cfg.stochastic.sppt_scales == ((0.8, 450000.0, 21600.0),)
