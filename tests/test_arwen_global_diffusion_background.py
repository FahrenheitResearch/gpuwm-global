"""The background hyperdiffusion term ([diffusion] background_order /
background_e_folding_time_s_at_truncation, arwen_global.spectral.diffusion).

Off by default: every shipped config keeps its hash and the operator is the
single-term drain bit for bit.  On: the factor is the product of the two
exponentials, the keys come as a pair, under the hyperdiffusion only, and
the background order sits below the drain's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.config import load_config  # noqa: E402
from arwen_global.configs_dir import config_root as _shipped_configs  # noqa: E402
from arwen_global.runner import build_diffusion  # noqa: E402
from arwen_global.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402

T1023 = "arwen_global_gdas_t1023_native_sl_si_24h.toml"


def _transform(truncation: int = 42):
    return SphericalHarmonicTransform.create(truncation, backend="numpy", precision="float64")


def _with_diffusion(tmp_path: Path, extra: str) -> Path:
    text = (Path(_shipped_configs()) / T1023).read_text(encoding="utf-8")
    marker = "preserve_degree = 1\n"
    assert marker in text
    out = tmp_path / "t1023-bg.toml"
    out.write_text(text.replace(marker, marker + extra, 1), encoding="utf-8")
    return out


def test_off_is_the_single_term_drain_bit_for_bit():
    tr = _transform()
    a = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0)
    b = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0,
                                  background_order=0,
                                  background_e_folding_time_s_at_truncation=0.0)
    fa = np.asarray(tr.backend.to_numpy(a.factors(tr, 300.0)))
    fb = np.asarray(tr.backend.to_numpy(b.factors(tr, 300.0)))
    assert np.array_equal(fa, fb)


def test_on_is_the_product_of_the_two_terms():
    tr = _transform()
    drain = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0)
    bg_only = ExponentialHyperdiffusion(order=3, e_folding_time_s_at_truncation=1800.0)
    both = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0,
                                     background_order=3,
                                     background_e_folding_time_s_at_truncation=1800.0)
    to = tr.backend.to_numpy
    product = np.asarray(to(drain.factors(tr, 300.0)), dtype=np.float64) * np.asarray(
        to(bg_only.factors(tr, 300.0)), dtype=np.float64)
    got = np.asarray(to(both.factors(tr, 300.0)), dtype=np.float64)
    np.testing.assert_allclose(got, product, rtol=2e-6, atol=0.0)
    # degrees 0..preserve untouched; the truncation reads both rates
    assert got[0] == 1.0 and got[1] == 1.0
    np.testing.assert_allclose(got[-1], np.exp(-300.0 / 720.0 - 300.0 / 1800.0), rtol=2e-6)


def test_the_background_term_drains_the_mesoscale_band_and_leaves_the_synoptic_scales():
    """At T1023 the order-16 drain e-folds the coefficients in tens of
    thousands of hours at n = 700; order 3 at 1800 s does it in hours, and
    leaves n = 200 at hundreds of hours."""
    n = np.array([200.0, 500.0, 700.0, 900.0, 1023.0])
    lam = n * (n + 1.0) / (1023.0 * 1024.0)
    efold16_h = 720.0 / lam ** 16 / 3600.0
    efold3_h = 1800.0 / lam ** 3 / 3600.0
    assert efold16_h[2] > 1.0e4          # n = 700: nothing drains it today
    assert efold3_h[2] < 6.0              # n = 700: hours with the term
    assert efold3_h[0] > 5.0e3 / 60.0     # n = 200: > 80 h, synoptic scales kept


def test_shipped_t1023_hash_is_unchanged_by_the_new_keys_being_absent():
    cfg = load_config(Path(_shipped_configs()) / T1023)
    assert cfg.diffusion_background_order == 0
    ident = cfg.config_identity
    assert "diffusion_background_order" not in ident
    assert "diffusion_background_efold_s" not in ident


def test_a_config_that_turns_it_on_moves_the_hash_and_builds_the_term(tmp_path):
    base = load_config(Path(_shipped_configs()) / T1023)
    on = load_config(_with_diffusion(
        tmp_path, "background_order = 3\nbackground_e_folding_time_s_at_truncation = 1800.0\n"))
    assert on.diffusion_background_order == 3
    assert on.diffusion_background_efold_s == 1800.0
    assert on.config_hash != base.config_hash
    op = build_diffusion(on)
    assert isinstance(op, ExponentialHyperdiffusion)
    assert op.background_order == 3
    assert op.background_e_folding_time_s_at_truncation == 1800.0
    assert op.order == 16 and op.e_folding_time_s_at_truncation == 720.0


@pytest.mark.parametrize("extra", [
    "background_order = 3\n",
    "background_e_folding_time_s_at_truncation = 1800.0\n",
    "background_order = 0\nbackground_e_folding_time_s_at_truncation = 1800.0\n",
    "background_order = 3\nbackground_e_folding_time_s_at_truncation = 0.0\n",
    "background_order = 16\nbackground_e_folding_time_s_at_truncation = 1800.0\n",
])
def test_bad_background_settings_are_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(_with_diffusion(tmp_path, extra))


def test_momentum_only_acts_on_vorticity_and_divergence_and_leaves_the_rest_to_the_drain():
    tr = _transform()
    op = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0,
                                   background_order=3,
                                   background_e_folding_time_s_at_truncation=3600.0,
                                   background_fields="momentum")
    drain = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0)
    both = ExponentialHyperdiffusion(order=16, e_folding_time_s_at_truncation=720.0,
                                     background_order=3,
                                     background_e_folding_time_s_at_truncation=3600.0)
    assert op.background_acts_on("vorticity") and op.background_acts_on("divergence")
    for name in ("theta", "log_surface_pressure", "qv"):
        assert not op.background_acts_on(name)
        assert both.background_acts_on(name)
    assert not drain.background_acts_on("vorticity")
    to = tr.backend.to_numpy
    assert np.array_equal(np.asarray(to(op.factors(tr, 300.0, background=False))),
                          np.asarray(to(drain.factors(tr, 300.0))))
    assert np.array_equal(np.asarray(to(op.factors(tr, 300.0))),
                          np.asarray(to(both.factors(tr, 300.0))))
    with pytest.raises(ValueError):
        ExponentialHyperdiffusion(background_fields="temperature")


def test_background_fields_joins_the_identity_only_when_momentum(tmp_path):
    keys = "background_order = 3\nbackground_e_folding_time_s_at_truncation = 3600.0\n"
    plain = load_config(_with_diffusion(tmp_path, keys))
    (tmp_path / "a").mkdir()
    spelled = load_config(_with_diffusion(tmp_path / "a", keys + 'background_fields = "all"\n'))
    (tmp_path / "m").mkdir()
    momentum = load_config(_with_diffusion(tmp_path / "m", keys + 'background_fields = "momentum"\n'))
    assert plain.config_hash == spelled.config_hash
    assert "diffusion_background_fields" not in plain.config_identity
    assert momentum.config_identity["diffusion_background_fields"] == "momentum"
    assert momentum.config_hash != plain.config_hash
    op = build_diffusion(momentum)
    assert op.background_fields == "momentum"


@pytest.mark.parametrize("extra", [
    'background_fields = "momentum"\n',
    'background_order = 3\nbackground_e_folding_time_s_at_truncation = 3600.0\nbackground_fields = "theta"\n',
])
def test_bad_background_fields_are_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(_with_diffusion(tmp_path, extra))
