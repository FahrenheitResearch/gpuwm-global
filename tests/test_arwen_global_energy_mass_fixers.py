"""The two long-run conservation switches of [repair] (2026-11-14).

* ``energy_fixer``: the dry total energy int (cp T + K) dp/g + Phi_s ps/g
  the state carries into the dynamics is restored after the dynamics,
  the diffusion and the mass fixer, by one uniform temperature increment.
  With no physics a step then conserves that energy to roundoff, where
  without it the semi-Lagrangian core, the hyperdiffusion and the mass
  fixer move it (the T127 year run's in-situ ledger: -4.9 W/m2).
* ``mass_fixer_target = "dry"``: the mass fixer holds the dry mass
  ps - g W instead of the total, so water the physics adds or removes at
  a fixed surface pressure moves the total, not the dry mass (LR-1).

Both are off by default, and off is the shipped arithmetic and the
shipped config hash.
"""
from __future__ import annotations

from dataclasses import replace
import math

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
from arwen_global.configs_dir import config_root as _shipped_configs  # noqa: E402
from arwen_global.constants import (  # noqa: E402
    DRY_TARGET_MASS_RELATIVE_DRIFT_FLOOR,
    GRAVITY_M_S2,
)
from arwen_global.dynamics import MoistHybridModel  # noqa: E402
from arwen_global.runner import run  # noqa: E402
from arwen_global.semi_implicit import VerticalModeSemiImplicit  # noqa: E402
from arwen_global.semilag import SEMILAG_INTEGRATORS, SemiLagrangianOptions  # noqa: E402
from arwen_global.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from arwen_global.state import ArwenGlobalState  # noqa: E402
from arwen_global.vertical import HybridCoordinate  # noqa: E402

from test_arwen_global_semilag_step import _baroclinic, _transform  # noqa: E402

SL = SEMILAG_INTEGRATORS[0]
SMOKE = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
#: The repair's water reading against the same water measured on the state
#: entering the dynamics: measured equal to 2e-16 (CPU), so anything
#: past roundoff is a reading of the wrong state or species set.
READING_TOLERANCE = 1.0e-12
#: The dry mass after a step under the repair's reading: off the cold value
#: by g times the adiabatic step's own water change, re-anchored each step.
#: Measured 2.8e-9 on the start-up step and 1e-10..5e-10 after (CPU).
DRY_STEP_TOLERANCE = 1.0e-8


def _model(transform, vertical, *, energy_fixer=False, target="total",
           terrain=True, vapour_fixer=False, positivity=False):
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    height = 1500.0 * np.exp(-((lat - 0.6) ** 2 + (lon - 2.0) ** 2) / 0.15)
    geopotential = GRAVITY_M_S2 * np.broadcast_to(
        height if terrain else 0.0 * height, transform.grid.shape).copy()
    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=geopotential, physics=None,
        diffusion=ExponentialHyperdiffusion(order=4, e_folding_time_s_at_truncation=3600.0),
        semi_implicit=VerticalModeSemiImplicit(off_centring_weight=0.55),
        integrator=SL, mass_fixer=True, mass_fixer_target=target,
        energy_fixer=energy_fixer, vapour_fixer=vapour_fixer, water_fixer=False,
        positivity_repair=positivity, maximum_cfl=1.0e9, sponge_base_pa=0.0,
        semilag=SemiLagrangianOptions(),
    )


def _energy(model, bundle):
    return model._dry_energy_reading(bundle.atmosphere)[0]


def _dry_mass(model, bundle):
    xp = model.transform.backend.xp
    ps = np.asarray(xp.exp(model.transform.inverse(bundle.atmosphere.log_surface_pressure)))
    return (model.transform.grid.global_mean(ps)
            - GRAVITY_M_S2 * model._global_mean_atmospheric_water(bundle.atmosphere))


def _total_mass(model, bundle):
    xp = model.transform.backend.xp
    ps = np.asarray(xp.exp(model.transform.inverse(bundle.atmosphere.log_surface_pressure)))
    return model.transform.grid.global_mean(ps)


def test_the_energy_fixer_makes_an_adiabatic_step_conserve_dry_energy():
    """No physics: the only energy the step may move is what its numerics
    move.  Off, the semi-Lagrangian step, the diffusion and the mass fixer
    over terrain move it (measured here, not assumed); on, every step
    returns the energy it was handed to 1e-12 of the energy."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    start = _baroclinic(transform, vertical)
    dt = 300.0

    off = _model(transform, vertical)
    on = _model(transform, vertical, energy_fixer=True)
    e0 = _energy(off, start)
    a = b = start
    drift_off = []
    for _ in range(6):
        a, _m = off.step(a, dt)
        b, metrics = on.step(b, dt)
        drift_off.append(_energy(off, a) - e0)
        assert abs(_energy(on, b) - e0) <= 1.0e-12 * e0
        assert math.isfinite(metrics["energy_fixer_w_m2"])
        assert math.isfinite(metrics["energy_fixer_delta_t_k"])
    # The instrument has something to measure: off, the step moves the
    # energy by far more than the on arm's tolerance.
    assert max(abs(d) for d in drift_off) > 1.0e-9 * e0
    # Off writes no fixer rows.
    assert "energy_fixer_w_m2" not in _m


def test_the_vapour_fixer_makes_an_adiabatic_step_conserve_vapour_and_energy():
    """The vapour column, read the way the fixer reads it, is the one the
    step was handed to 1e-12, with the energy fixer beside it holding the
    dry energy as before; off, the spectral vapour's semi-Lagrangian ride
    over the terrain moves it."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    start = _baroclinic(transform, vertical)
    dt = 300.0
    off = _model(transform, vertical)
    on = _model(transform, vertical, energy_fixer=True, vapour_fixer=True)
    reading0 = on._dry_energy_reading(start.atmosphere)
    assert off._dry_energy_reading(start.atmosphere)[3] is None
    a = b = start
    moved = []
    for _ in range(6):
        a, _m = off.step(a, dt)
        b, metrics = on.step(b, dt)
        ra = on._dry_energy_reading(a.atmosphere)
        rb = on._dry_energy_reading(b.atmosphere)
        moved.append(ra[3] - reading0[3])
        assert abs(rb[3] - reading0[3]) <= 1.0e-12 * reading0[3]
        assert abs(rb[0] - reading0[0]) <= 1.0e-12 * reading0[0]
        assert math.isfinite(metrics["vapour_fixer_kg_m2"])
        assert math.isfinite(metrics["vapour_fixer_latent_w_m2"])
    assert max(abs(d) for d in moved) > 1.0e-9 * reading0[3]
    assert "vapour_fixer_kg_m2" not in _m


def test_the_energy_fixer_moves_only_theta_and_only_its_level_means():
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    start = _baroclinic(transform, vertical)
    model = _model(transform, vertical, energy_fixer=True)
    reference = model._dry_energy_reading(start.atmosphere)
    cooled = start.atmosphere.with_fields([
        start.atmosphere.vorticity, start.atmosphere.divergence,
        start.atmosphere.theta * (1.0 - 1.0e-4),
        start.atmosphere.log_surface_pressure, start.atmosphere.qv])
    fixed, record = model._fix_energy(cooled, reference, 300.0)
    assert abs(model._dry_energy_reading(fixed)[0] - reference[0]) <= 1.0e-12 * reference[0]
    assert record["energy_fixer_delta_t_k"] > 0.0
    for name in ("vorticity", "divergence", "log_surface_pressure", "qv"):
        assert np.array_equal(getattr(fixed, name), getattr(cooled, name)), name
    change = np.asarray(fixed.theta) - np.asarray(cooled.theta)
    assert np.all(change[:, 1:, :] == 0.0) and np.all(change[:, 0, 1:] == 0.0)
    assert np.all(change[:, 0, 0].real > 0.0)


def test_the_dry_target_holds_dry_mass_while_water_comes_and_goes():
    """Water added at a fixed surface pressure (what the physics does when
    it evaporates) is dry air taken away under the total target, and
    total mass added under the dry target."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    start = _baroclinic(transform, vertical)
    dt = 300.0
    total = _model(transform, vertical)
    dry = _model(transform, vertical, target="dry")
    total.initialize_mass_target(start.atmosphere)
    dry.initialize_mass_target(start.atmosphere)
    d0 = _dry_mass(dry, start)
    m0 = _total_mass(total, start)
    a = b = start
    for _ in range(4):
        # "Evaporation": 0.2 kg/m2 of vapour per step, spread as a uniform
        # 5 percent rise of the mixing ratio (the surface pressure is not
        # touched, as the physics does not touch it).
        for name, bundle in (("a", a), ("b", b)):
            moist = bundle.atmosphere.with_fields([
                bundle.atmosphere.vorticity, bundle.atmosphere.divergence,
                bundle.atmosphere.theta, bundle.atmosphere.log_surface_pressure,
                bundle.atmosphere.qv * 1.05])
            moist = ArwenGlobalState(moist, bundle.surface, bundle.physics_state)
            if name == "a":
                a = moist
            else:
                b = moist
        a, _ma = total.step(a, dt)
        b, _mb = dry.step(b, dt)
        assert abs(_total_mass(total, a) - m0) <= 1.0e-9 * m0
        assert abs(_dry_mass(dry, b) - d0) <= 1.0e-9 * d0
    # The two targets part by g times the water added (tens of Pa here).
    assert _total_mass(dry, b) - _total_mass(total, a) > 1.0


def _mean_ps(model, atmosphere):
    xp = model.transform.backend.xp
    ps = np.asarray(xp.exp(model.transform.inverse(atmosphere.log_surface_pressure)))
    return model.transform.grid.global_mean(ps)


def test_the_dry_target_on_the_production_path_hands_the_fixer_the_right_water(monkeypatch):
    """The path every real run takes: the positivity repair on and the
    semi-Lagrangian step, so the fixer is handed the first repair's reading
    of the water instead of measuring it (the test above has the repair off
    and so only ever takes the measured branch).

    Held here, every step: the reading handed in is the water of the state
    entering the dynamics (a reading of the wrong state or the wrong
    species set would miss by far more); the fixed state's global-mean
    surface pressure is the dry target plus g times that reading; and the
    dry mass at the fixer and at the step's end stays at the cold value
    within what the adiabatic step's own water change moves it."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    start = _baroclinic(transform, vertical)
    dt = 300.0
    model = _model(transform, vertical, target="dry", positivity=True)
    model.initialize_mass_target(start.atmosphere)
    d0 = _dry_mass(model, start)
    target = float(model._target_mass_pa)
    assert abs(target - d0) <= 1.0e-12 * d0
    seen = {}
    real_dynamics = model.integrate_dynamics
    real_fix = model._fix_mass

    def dynamics(atmosphere, *args, **kwargs):
        seen["entering"] = atmosphere
        return real_dynamics(atmosphere, *args, **kwargs)

    def fix(state, water_kg_m2=None):
        fixed, offset = real_fix(state, water_kg_m2=water_kg_m2)
        seen["water"], seen["fixed"] = water_kg_m2, fixed
        return fixed, offset

    monkeypatch.setattr(model, "integrate_dynamics", dynamics)
    monkeypatch.setattr(model, "_fix_mass", fix)
    b = start
    for _ in range(6):
        b, metrics = model.step(b, dt)
        assert seen["water"] is not None, "the measured branch ran, not the hand-in"
        assert metrics["mass_fixer_atmospheric_water_kg_m2"] == seen["water"]
        entering = model._global_mean_atmospheric_water(seen["entering"])
        assert abs(seen["water"] - entering) <= READING_TOLERANCE * entering, (
            seen["water"], entering)
        held = _mean_ps(model, seen["fixed"])
        assert abs(held - (target + GRAVITY_M_S2 * seen["water"])) <= 1.0e-12 * held
        fixed_water = model._global_mean_atmospheric_water(seen["fixed"])
        dry_at_fixer = held - GRAVITY_M_S2 * fixed_water
        assert abs(dry_at_fixer - d0) <= DRY_STEP_TOLERANCE * d0, dry_at_fixer - d0
        assert abs(_dry_mass(model, b) - d0) <= DRY_STEP_TOLERANCE * d0


def _restart_pair(tmp_path, model_of):
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle = _baroclinic(transform, vertical)
    dt = 300.0
    straight = model_of(transform, vertical)
    straight.initialize_mass_target(bundle.atmosphere)
    uninterrupted = bundle
    for _ in range(6):
        uninterrupted, _m = straight.step(uninterrupted, dt)

    first = model_of(transform, vertical)
    first.initialize_mass_target(bundle.atmosphere)
    midpoint = bundle
    for _ in range(3):
        midpoint, _m = first.step(midpoint, dt)
    path = write_checkpoint(
        tmp_path / "midpoint.npz", midpoint,
        config_hash="0" * 64, to_numpy=transform.backend.to_numpy,
        semi_implicit_scheme="vertical_modes", integrator=SL,
        trajectory=first.trajectory_state(),
    )
    metadata, arrays = read_checkpoint(
        path, semi_implicit_scheme="vertical_modes", integrator=SL)
    second = model_of(transform, vertical)
    # The runner initializes the targets from the cold state on every
    # segment (build_model_and_cold_state); the same here.
    second.initialize_mass_target(bundle.atmosphere)
    resumed = state_from_checkpoint(metadata, arrays, transform.backend)
    second.set_trajectory_state(trajectory_from_checkpoint(metadata, arrays, transform.backend))
    for _ in range(3):
        resumed, _m = second.step(resumed, dt)
    return uninterrupted, resumed


@pytest.mark.parametrize("positivity", [False, True], ids=["measured", "repair_reading"])
def test_a_restart_with_both_switches_on_is_bitwise(tmp_path, positivity):
    uninterrupted, resumed = _restart_pair(
        tmp_path, lambda t, v: _model(t, v, energy_fixer=True, target="dry",
                                      vapour_fixer=True, positivity=positivity))
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure", "qv", "qc"):
        assert np.array_equal(np.asarray(getattr(uninterrupted.atmosphere, name)),
                              np.asarray(getattr(resumed.atmosphere, name))), name


def test_config_parses_refuses_and_keeps_the_shipped_hash(tmp_path):
    shipped = load_config(SMOKE)
    assert shipped.mass_fixer_target == "total" and shipped.energy_fixer is False
    text = open(SMOKE, encoding="utf-8").read()
    marker = "[repair]\n"
    assert marker in text

    def with_repair(extra):
        path = tmp_path / f"cfg{abs(hash(extra))}.toml"
        path.write_text(text.replace(marker, marker + extra), encoding="utf-8")
        return load_config(str(path))

    explicit_off = with_repair(
        'mass_fixer_target = "total"\nenergy_fixer = false\nvapour_fixer = false\n')
    assert explicit_off.config_hash == shipped.config_hash
    # The smoke gates mass at 1e-9: the dry target refuses that by name.
    with pytest.raises(ValueError, match="mass_relative_drift = 1e-09"):
        with_repair('mass_fixer_target = "dry"\n')
    shipped_text = text
    text = shipped_text.replace("mass_relative_drift = 1.0e-9\n", "")
    assert "mass_relative_drift" not in text
    # The total target with the gate unset keeps the shipped default.
    assert with_repair("").gate_mass_relative_drift == 1.0e-9
    on = with_repair('mass_fixer_target = "dry"\nenergy_fixer = true\nvapour_fixer = true\n')
    # Unset under the dry target, the gate is the floor; set at or above it,
    # it is kept.
    assert on.gate_mass_relative_drift == DRY_TARGET_MASS_RELATIVE_DRIFT_FLOOR == 1.0e-5
    text = shipped_text.replace("mass_relative_drift = 1.0e-9\n",
                                "mass_relative_drift = 1.0e-4\n")
    assert with_repair('mass_fixer_target = "dry"\n').gate_mass_relative_drift == 1.0e-4
    text = shipped_text
    assert on.mass_fixer_target == "dry" and on.energy_fixer is True and on.vapour_fixer is True
    assert on.config_hash != shipped.config_hash
    hashes = {with_repair(f"{key} = true\n").config_hash
              for key in ("energy_fixer", "vapour_fixer")}
    assert len(hashes) == 2 and not hashes & {shipped.config_hash, on.config_hash}
    with pytest.raises(ValueError, match="mass_fixer_target"):
        with_repair('mass_fixer_target = "wet"\n')
    bad = text.replace("mass_fixer = true", 'mass_fixer = false\nmass_fixer_target = "dry"')
    path = tmp_path / "bad.toml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="mass_fixer is false"):
        load_config(str(path))


def _dry_smoke(tmp_path, *, gate_line="", name="dry-smoke.toml"):
    """The shipped smoke under all three switches, through load_config,
    with the smoke's total-target mass gate replaced by ``gate_line``."""
    text = open(SMOKE, encoding="utf-8").read()
    text = text.replace("mass_relative_drift = 1.0e-9\n", gate_line)
    text = text.replace("[repair]\n", '[repair]\nmass_fixer_target = "dry"\n'
                        "energy_fixer = true\nvapour_fixer = true\n")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_config(str(path))


def test_the_branch_epoch_fingerprint():
    """branch_mass_epoch reads which quantity the parent held off the two
    states: this run's quantity moved and the other held means the parent
    held the other one; anything else keeps the cold target."""
    from arwen_global.runner import branch_mass_epoch

    dry_cfg = replace(load_config(SMOKE), mass_fixer_target="dry")
    total_cfg = load_config(SMOKE)
    cold = {"global_mean_surface_pressure_pa": 98500.0,
            "global_mean_dry_surface_pressure_pa": 98250.0}
    water_gained = {  # a total-held parent: the total held, 3 kg/m2 more water
        "global_mean_surface_pressure_pa": 98500.0,
        "global_mean_dry_surface_pressure_pa": 98250.0 - 3.0 * GRAVITY_M_S2}
    dry_held = {  # a dry-held parent with the same water gained
        "global_mean_surface_pressure_pa": 98500.0 + 3.0 * GRAVITY_M_S2,
        "global_mean_dry_surface_pressure_pa": 98250.0}
    drifted = {  # a parent that held neither (both moved)
        "global_mean_surface_pressure_pa": 98540.0,
        "global_mean_dry_surface_pressure_pa": 98260.0}
    moved = branch_mass_epoch(dry_cfg, cold, water_gained)
    assert moved == pytest.approx(3.0 * GRAVITY_M_S2 / 98250.0)
    assert branch_mass_epoch(total_cfg, cold, dry_held) == pytest.approx(
        3.0 * GRAVITY_M_S2 / 98500.0)
    for cfg, state in ((dry_cfg, dry_held), (total_cfg, water_gained),
                       (dry_cfg, drifted), (total_cfg, drifted),
                       (dry_cfg, cold), (total_cfg, cold)):
        assert branch_mass_epoch(cfg, cold, state) is None
    assert branch_mass_epoch(replace(total_cfg, mass_fixer=False), cold, dry_held) is None


def test_a_branch_onto_the_dry_target_from_a_total_run_takes_no_mass_shock(tmp_path):
    """A total-target run whose atmosphere has gained water (here 5 percent
    of its vapour, at a fixed surface pressure, as the physics books it)
    branched onto the dry target: the restart's dry mass sits g dW below the
    cold target, and without a new epoch the first step would take all of
    it out as one uniform ln ps offset.  With the epoch the first step's
    offset is the step's own, the receipt names the epoch, and the mass
    gate reads against the restart state's dry mass."""
    total_cfg = load_config(SMOKE)
    parent = tmp_path / "parent"
    run(total_cfg, parent)
    metadata, arrays = read_checkpoint(parent / "arwen_global_step00000002.npz")
    wetter = dict(arrays)
    wetter["atmosphere__qv"] = arrays["atmosphere__qv"] * 1.05
    from arwen_global.checkpoint import write_checkpoint_arrays
    source = tmp_path / "wetter.npz"
    write_checkpoint_arrays(
        source, wetter, step=int(metadata["step"]), time_s=float(metadata["time_s"]),
        physics_state_schema=metadata["physics_state_schema"],
        physics_metadata=metadata["physics_metadata"],
        config_hash=total_cfg.config_hash, trackers=metadata.get("run_trackers"),
        semi_implicit_scheme=total_cfg.semi_implicit_scheme,
        integrator=total_cfg.integrator)
    dry_cfg = _dry_smoke(tmp_path)
    branch = tmp_path / "branch"
    result = run(dry_cfg, branch, restart=source, branch_config=True)
    targets = result["restart_targets"]
    assert targets is not None and "other mass quantity" in targets["reason"]
    assert targets["mass_target_pa"] < targets["cold_mass_target_pa"] - 1.0
    assert targets["total_water_target_kg_m2"] == targets["cold_total_water_target_kg_m2"]
    gates = result["gates"]
    assert gates["mass_fixer_max_step_log_offset"]["value"] < 1.0e-6
    assert gates["mass_relative_drift"]["passed"], gates["mass_relative_drift"]
    # The same branch from the unmodified checkpoint (the parent held the
    # total and gained no water to speak of in four smoke steps) keeps the
    # cold target, as every branch did before.
    plain = run(dry_cfg, tmp_path / "plain",
                restart=parent / "arwen_global_step00000002.npz", branch_config=True)
    assert plain["restart_targets"] is None


def test_the_runner_with_both_switches_restarts_bitwise_and_gates_dry_mass(tmp_path):
    """The shipped smoke (Eulerian, reference physics, terrain) through the
    runner: the receipt's mass gate reads the dry mass the run holds, a
    restart from the midpoint reproduces the straight run's last
    checkpoint array for array, and the dry-mass row stays at its cold
    value while the total moves with the water.

    The config goes through load_config with the smoke's 1e-9 mass gate
    left unset, so the dry target's own default gate is the one the
    receipt reads: the fixer reads the water of the state entering the
    dynamics (the first repair's reading, free) and the row reads it after
    the second physics half, so the dry mass at an output instant carries
    that half step's E - P (1.1e-9 of the mass here, 1e-4 Pa), re-anchored
    every step and never accumulated."""
    cfg = _dry_smoke(tmp_path)
    assert cfg.gate_mass_relative_drift == DRY_TARGET_MASS_RELATIVE_DRIFT_FLOOR
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    result = run(cfg, full)
    assert result["status"] == "pass", result.get("gates")
    assert result["gates"]["mass_relative_drift"]["value"] < 1.0e-7
    run(cfg, resumed, restart=full / "arwen_global_step00000002.npz")
    meta_a, arrays_a = read_checkpoint(full / "arwen_global_step00000004.npz")
    meta_b, arrays_b = read_checkpoint(resumed / "arwen_global_step00000004.npz")
    assert meta_a["run_trackers"] == meta_b["run_trackers"]
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name
    import json
    rows = [json.loads(line) for line in (full / "diagnostics.jsonl").read_text().splitlines() if line]
    dry = [row["global_mean_dry_surface_pressure_pa"] for row in rows]
    assert max(dry) - min(dry) <= 1.0e-7 * dry[0]
