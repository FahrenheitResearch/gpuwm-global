"""Hortal's departure-side SETTLS and the vertical-rate limiter, as options.

``[semilag] extrapolation = "settls_departure"`` reads the extrapolated wind
and residual at the DEPARTURE point where the shipped ``"settls"`` reads them
at the arrival point, and ``level_rate_limiter`` keeps ``s^n`` where the
vertical rate changed sign between the two time levels.  Neither is a
default.  The gates here are the ones a reader of the option needs:

*   the shipped form and the shipped identity are untouched by the new keys;
*   the start-up step, which has no previous level, is the same step under
    both forms, bit for bit;
*   the new form changes the run once a previous level exists (an option
    that changed nothing would pass every other gate here);
*   a restart under the new form and the limiter is bit-exact, because every
    input it reads is already checkpointed;
*   the limiter does exactly what it says at a sign flip and nothing where
    the rate kept its sign.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.checkpoint import (  # noqa: E402
    read_checkpoint,
    state_from_checkpoint,
    trajectory_from_checkpoint,
    write_checkpoint,
)
from arwen_global.config import load_config  # noqa: E402
from arwen_global.semilag import SemiLagrangianOptions  # noqa: E402
from arwen_global.semilag.options import EXTRAPOLATIONS  # noqa: E402
from arwen_global.semilag.step import extrapolated_level_rate  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402

from test_arwen_global_semilag_step import (  # noqa: E402
    SL, _BASE, _baroclinic, _model, _transform, _write,
)

FIELDS = ("vorticity", "divergence", "theta", "log_surface_pressure", "qv",
          "qc")


def _options(**kwargs):
    return SemiLagrangianOptions(**kwargs)


def _run(options, steps, *, dt=300.0, truncation=21, nlev=20):
    transform = _transform(truncation)
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    bundle = _baroclinic(transform, vertical)
    model = _model(transform, vertical, options=options)
    for _ in range(steps):
        bundle, _m = model.step(bundle, dt)
    return transform, vertical, model, bundle


# ------------------------------------------------------------- the options

def test_the_shipped_form_keeps_its_name_and_its_identity(tmp_path):
    assert EXTRAPOLATIONS == ("settls", "settls_departure", "none")
    default = SemiLagrangianOptions()
    assert default.extrapolation == "settls"
    assert default.level_rate_limiter is False
    # Off, the limiter is absent from the identity, so every shipped config
    # hash is the hash it was before the key existed.
    assert "level_rate_limiter" not in default.identity
    shipped = load_config(_write(tmp_path, _BASE.format(
        integrator=SL, time_extra="", extra="")))
    spelled = load_config(_write(tmp_path, _BASE.format(
        integrator=SL, time_extra="",
        extra='[semilag]\nextrapolation = "settls"\n'
              "level_rate_limiter = false\n")))
    assert spelled.config_hash == shipped.config_hash
    assert "level_rate_limiter" not in shipped.config_identity["semilag"]


def test_the_new_form_and_the_limiter_get_their_own_identity(tmp_path):
    shipped = load_config(_write(tmp_path, _BASE.format(
        integrator=SL, time_extra="", extra="")))
    departure = load_config(_write(tmp_path, _BASE.format(
        integrator=SL, time_extra="",
        extra='[semilag]\nextrapolation = "settls_departure"\n')))
    limited = load_config(_write(tmp_path, _BASE.format(
        integrator=SL, time_extra="",
        extra='[semilag]\nextrapolation = "settls_departure"\n'
              "level_rate_limiter = true\n")))
    hashes = {shipped.config_hash, departure.config_hash, limited.config_hash}
    assert len(hashes) == 3
    assert limited.config_identity["semilag"]["level_rate_limiter"] is True
    assert (departure.config_identity["semilag"]["extrapolation"]
            == "settls_departure")


def test_a_limiter_with_nothing_to_limit_is_refused():
    with pytest.raises(ValueError, match="level_rate_limiter"):
        _options(extrapolation="none", level_rate_limiter=True)
    with pytest.raises(ValueError, match="boolean"):
        _options(level_rate_limiter="yes")


# ----------------------------------------------------------------- the step

def test_the_start_up_step_is_the_same_step_under_both_forms():
    """No previous level, nothing to extrapolate: the two forms and the
    limiter all take the shipped start-up step, bit for bit."""
    _t, _v, _m, shipped = _run(_options(), 1)
    _t, _v, _m, departure = _run(_options(
        extrapolation="settls_departure", level_rate_limiter=True), 1)
    for name in FIELDS:
        assert np.array_equal(
            np.asarray(getattr(shipped.atmosphere, name)),
            np.asarray(getattr(departure.atmosphere, name)),
        ), name


@pytest.mark.parametrize("options", [
    {"extrapolation": "settls_departure"},
    {"extrapolation": "settls", "level_rate_limiter": True},
])
def test_each_new_arm_changes_the_run_once_a_previous_level_exists(options):
    """The counter-instrument: past the start-up step the arm is a different
    run (an option that was parsed and changed nothing would pass the
    gates above)."""
    _t, _v, _m, shipped = _run(_options(), 4)
    _t, _v, _m, other = _run(_options(**options), 4)
    difference = float(np.max(np.abs(
        np.asarray(shipped.atmosphere.divergence)
        - np.asarray(other.atmosphere.divergence))))
    assert difference > 0.0
    # Same scheme order, so the same forecast to well inside the state's
    # own scale: a sign error in the swap reads as an O(1) difference.
    scale = float(np.max(np.abs(np.asarray(shipped.atmosphere.divergence))))
    assert difference < 0.25 * scale, (difference, scale)


def test_the_departure_form_restarts_bit_exact(tmp_path):
    """Every input the new form and the limiter read (u, v, s and the four
    residuals at t^{n-1}) is in the checkpoint already, so a run restarted
    at its midpoint is the uninterrupted run, bit for bit."""
    options = _options(extrapolation="settls_departure",
                       level_rate_limiter=True)
    transform, vertical, _model_u, uninterrupted = _run(options, 6)

    bundle = _baroclinic(transform, vertical)
    restarting = _model(transform, vertical, options=options)
    for _ in range(3):
        bundle, _m = restarting.step(bundle, 300.0)
    path = write_checkpoint(
        tmp_path / "midpoint.npz", bundle,
        config_hash="0" * 64, to_numpy=transform.backend.to_numpy,
        semi_implicit_scheme="vertical_modes", integrator=SL,
        trajectory=restarting.trajectory_state(),
    )
    metadata, arrays = read_checkpoint(
        path, semi_implicit_scheme="vertical_modes", integrator=SL)
    resumed_model = _model(transform, vertical, options=options)
    resumed = state_from_checkpoint(metadata, arrays, transform.backend)
    resumed_model.set_trajectory_state(
        trajectory_from_checkpoint(metadata, arrays, transform.backend))
    for _ in range(3):
        resumed, _m = resumed_model.step(resumed, 300.0)
    for name in FIELDS:
        assert np.array_equal(
            np.asarray(getattr(uninterrupted.atmosphere, name)),
            np.asarray(getattr(resumed.atmosphere, name)),
        ), f"{name} differs across the restart"


# -------------------------------------------------------------- the limiter

def test_the_limiter_keeps_the_current_rate_where_the_sign_flipped():
    current = np.array([[1.0, -2.0, 3.0, 0.0, -4.0]])
    previous = np.array([[0.5, 1.0, -1.0, 2.0, -1.0]])
    unlimited = extrapolated_level_rate(current, previous, xp=np,
                                        limited=False)
    assert np.array_equal(unlimited, 2.0 * current - previous)
    limited = extrapolated_level_rate(current, previous, xp=np, limited=True)
    # columns 1 and 2 flipped sign: they keep s^n; 0 and 4 did not; 3 is a
    # zero rate, which is no flip, and keeps the extrapolation.
    assert np.array_equal(limited, np.array([[1.5, -2.0, 3.0, -2.0, -7.0]]))
