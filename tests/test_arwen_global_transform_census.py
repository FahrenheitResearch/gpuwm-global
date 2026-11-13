"""The transform diet ([transforms]): what each switch removes, counted, and
what it does to the bits, compared.

THE BREAKAGE THIS PREVENTS.  The semi-Lagrangian step made 61 Legendre
contractions over 2,573 level-planes a step (the census of the core-speed
reading, numpy T21 L40, reproduced as the first known answer below), and
some of them fed nothing: the grid vorticity and divergence the step never
reads, and the vorticity half of three mass-flux analyses.  A cut like that
is easy to lose again without anyone noticing, because the forecast does
not change when a transform is wasted.  So the census is a gate: the counts
per step are pinned, for the shipped step and for the dieted one, and the
test fails if either rises.

The second half is the bits.  Both switches keep every row of every call
and only narrow the stack (the GEMM's M), and whether a narrower M returns
the same bits is the BLAS library's choice on the CPU it runs on, not a
property of the switch.  `skip_unread_planes`' ten-step checkpoint is
compared byte for byte with the shipped run's: it held on every CPU
measured, GitHub's hosted runners included.  `divergence_only_mass_flux`'s
is compared to rounding: byte-identical on the hosts measured (AVX2
OpenBLAS kernels, numpy 2.4.6 and 2.5.4), it was NOT on GitHub's hosted
runners, where 0.1.8's publish failed on this gate (and three of its four
push-test jobs, ubuntu and Windows): 19 or 20 of the ten-step checkpoint's
52 arrays moved, alone and in the dieted run, and the float64 operator
test below found the divergence-only analysis not equal to the full one.
What it is held to is the guaranteed part, the same products and the same
subtraction in a narrower call: every float array of the ten-step
checkpoint within ROUNDING_TOLERANCE of its own largest value, and every
other array byte-identical.  Emulated on a host by moving every element of
every divergence-only analysis by up to one ulp, the ten-step checkpoint
moved the same 20 arrays the runner reported, by at most 7.0e-11 of their largest
value (the trajectory's ln ps tendency; 2.8e-12 for the divergence).  The
dieted run is still held to a restart-to-the-bit gate, which is a property
of one CPU and holds on every one.  A third cut, the assemble analysis of theta
and vapour in one call of 2 x nlev planes instead of fourteen six-plane
calls, moved M outside the measured bit-exact widths and moved bits on the
card, so it is not built.

Two more of the package's cuts are NOT built, and the counts below are why:
the in-situ ledger's syntheses are already reused (the next step's first
physics exchange reads them from the memo, so a run without the ledger
makes the same number of planes), and on an ordinary step no stacked
synthesis misses the memo while all of its fields were synthesized earlier
in the step, so a memo that served a subset of a cached stack would serve
nothing there; it would serve 2 x nlev planes once per output interval,
on the step after an output step.

CPU only: numpy float64, T21, 40 levels, the shipped sl_si options and the
reference physics; about ten seconds a run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.checkpoint import read_checkpoint  # noqa: E402
from arwen_global.config import load_config  # noqa: E402
from arwen_global.runner import run  # noqa: E402
from arwen_global.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from arwen_global.spectral.vector import VorticityDivergenceOperator  # noqa: E402

TREE = Path(__file__).resolve().parents[1]
NLEV = 40

#: The shipped step's counts per step (numpy, T21, L40, sl_si, reference
#: physics, Strang split, in-situ ledger on): the read-step-cost census.
SHIPPED_CALLS, SHIPPED_PLANES = 61, 2573
#: The same step with both switches on: as many calls, and 2 x 40 +
#: 3 x 2 x 40 fewer planes (the unread pair and the vorticity halves of the
#: three mass-flux analyses).
DIET_CALLS, DIET_PLANES = 61, 2253

_CONFIG = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "transform-census"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"
[grid]
truncation = 21
dealias_factor = 1.5
[time]
dt_s = 300.0
duration_s = 3000.0
output_interval_s = 1500.0
integrator = "sl_si"
minimum_fold_determinant = 0.2
[semilag]
quasi_monotone = true
trajectory_iterations = 3
[vertical]
coordinate = "surface_stretched"
nlev = 40
p_top_pa = 100.0
[initial]
surface_pressure_pa = 100000.0
surface_temperature_k = 286.0
top_temperature_k = 220.0
qv_surface = 0.006
zonal_wind_m_s = 2.0
perturbation_amplitude = 0.00001
zonal_wavenumber = 2
terrain_amplitude_m = 25.0
surface_water_kg_m2 = 500.0
[physics]
mode = "reference"
[reference_physics]
radiation = true
surface_fluxes = true
turbulence = true
convection = true
saturation_adjustment = true
microphysics = true
solar_constant_w_m2 = 1361.0
atmospheric_shortwave_absorptivity = 0.08
atmospheric_longwave_emissivity = 0.35
bulk_exchange_coefficient = 0.0005
background_diffusivity_m2_s = 0.05
pbl_diffusivity_m2_s = 2.0
pbl_depth_m = 1200.0
cloud_autoconversion_threshold = 0.001
cloud_autoconversion_time_s = 1800.0
ice_to_snow_time_s = 2400.0
snow_to_graupel_time_s = 4800.0
rain_fall_speed_m_s = 4.0
snow_fall_speed_m_s = 0.7
graupel_fall_speed_m_s = 2.0
soil_exchange_time_s = 86400.0
[diffusion]
enabled = true
order = 16
e_folding_time_s_at_truncation = 720.0
preserve_degree = 1
[semi_implicit]
enabled = true
weight = 1.0
off_centring_weight = 0.55
[repair]
mass_fixer = true
water_fixer = true
positivity_repair = true
"""

ARMS = {
    "shipped": {},
    "unread": {"skip_unread_planes": True},
    "divergence": {"divergence_only_mass_flux": True},
    "diet": {"skip_unread_planes": True, "divergence_only_mass_flux": True},
}


def _config(directory: Path, name: str, transforms: dict, extra: str = "") -> Path:
    text = _CONFIG
    if transforms:
        text += "[transforms]\n" + "".join(
            f"{key} = {'true' if value else 'false'}\n"
            for key, value in transforms.items())
    text += extra
    path = directory / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _hashes(outdir: Path, step: int) -> dict[str, str]:
    metadata, _ = read_checkpoint(outdir / f"arwen_global_step{step:08d}.npz")
    return {key: value["sha256"] for key, value in metadata["arrays"].items()}


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_transform_census",
        TREE / "tools" / "arwen_global_transform_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms(tool, tmp_path_factory):
    """Every arm run once, ten steps, steps 5 to 8 counted.  Step 5 writes
    output (and so runs the output diagnostics) and step 6 follows it;
    steps 7 and 8 are ordinary steps, which the step pins are about."""
    root = tmp_path_factory.mktemp("census")
    out = {}
    for name, transforms in ARMS.items():
        outdir = root / name
        table = tool.census_run(_config(root, name, transforms), outdir,
                                steps=4, warmup=4)
        out[name] = (outdir, table)
    outdir = root / "no-ledger"
    table = tool.census_run(
        _config(root, "no-ledger", {}, "[insitu]\nenabled = false\n"),
        outdir, steps=4, warmup=4)
    out["no-ledger"] = (outdir, table)
    return out


ORDINARY = (7, 8)


def _totals(table):
    steps = sorted(table)
    assert steps == [5, 6, *ORDINARY]
    totals = [table[step]["TOTAL"] for step in ORDINARY]
    # A step is a step: two consecutive ordinary steps count the same.
    assert totals[0] == totals[1]
    return totals[0], table[ORDINARY[0]]


# -- the counts -------------------------------------------------------------

def test_the_census_reproduces_the_shipped_step_count(arms):
    total, rows = _totals(arms["shipped"][1])
    assert (total["legendre_calls"], total["legendre_planes"]) == (
        SHIPPED_CALLS, SHIPPED_PLANES)
    # The assemble section: two wind analyses, theta and vapour in
    # thirteen chunks of six planes and one of two, and ln ps.
    assert rows["step.dynamics.assemble"]["legendre_calls"] == 17
    # The grid view the step builds carries four fields, two unread.
    assert rows["step.dynamics.grid_state"]["legendre_planes"] == 360


def test_the_dieted_step_makes_no_more_transforms_than_its_pin(arms):
    """The CI guard: a change that adds calls or planes to the dieted step
    fails here and must lower the pin with its own measurement."""
    total, rows = _totals(arms["diet"][1])
    assert total["legendre_calls"] <= DIET_CALLS
    assert total["legendre_planes"] <= DIET_PLANES
    assert rows["step.dynamics.grid_state"]["legendre_planes"] <= 360 - 2 * NLEV
    # And it is the cut it claims, not a lucky lower count: exactly.
    assert (total["legendre_calls"], total["legendre_planes"]) == (
        DIET_CALLS, DIET_PLANES)


def test_each_switch_removes_what_it_names_and_nothing_else(arms):
    shipped, _ = _totals(arms["shipped"][1])
    unread, rows = _totals(arms["unread"][1])
    assert shipped["legendre_calls"] == unread["legendre_calls"]
    assert shipped["legendre_planes"] - unread["legendre_planes"] == 2 * NLEV
    assert rows["step.dynamics.grid_state"]["legendre_planes"] == 360 - 2 * NLEV
    divergence, _ = _totals(arms["divergence"][1])
    assert shipped["legendre_calls"] == divergence["legendre_calls"]
    # Three mass-flux analyses a step: grid_state and both exchanges.
    assert shipped["legendre_planes"] - divergence["legendre_planes"] == 3 * 2 * NLEV
    # Neither touches the assemble analysis: thirteen six-plane chunks of
    # theta and vapour and one of two, as shipped.
    _diet, rows = _totals(arms["diet"][1])
    assert rows["step.dynamics.assemble"]["legendre_calls"] == 17


def test_the_ledger_costs_no_transform_the_step_would_not_make(arms):
    """The in-situ ledger's syntheses are the step's own: the next first
    physics exchange reads them from the memo.  Without the ledger the
    exchange makes them itself and the step's total is unchanged."""
    with_ledger, rows = _totals(arms["shipped"][1])
    without, rows_off = _totals(arms["no-ledger"][1])
    assert rows["step.observer"]["legendre_planes"] == 6 * NLEV
    assert "step.observer" not in rows_off
    assert without["legendre_planes"] == with_ledger["legendre_planes"]
    assert without["legendre_calls"] == with_ledger["legendre_calls"]


def test_no_synthesis_on_this_path_is_a_subset_of_a_cached_stack(arms):
    """What a subset memo could serve, counted: nothing, on the shipped
    step and on the dieted one.  Every stacked synthesis that misses the
    memo reads at least one array no earlier stack of the step held."""
    for name in ("shipped", "diet"):
        total, _ = _totals(arms[name][1])
        assert total["stack_requests"] > 0
        assert total["stack_planes_subset_servable"] == 0, name


def test_the_one_subset_a_cached_stack_could_serve_follows_an_output_step(arms):
    """The exception, counted so the page can state its size: an output
    step's diagnostics synthesize the full four-field view of the finished
    state and displace the ledger's (theta, qv) pair from the one-entry
    memo, so the next step's first exchange synthesizes that pair again.
    A subset memo would serve those 2 x nlev planes once per output
    interval and nothing on any other step."""
    table = arms["shipped"][1]
    assert table[5]["step.diagnostics"]["legendre_planes"] == 4 * NLEV
    assert table[6]["TOTAL"]["stack_planes_subset_servable"] == 2 * NLEV
    assert table[6]["TOTAL"]["legendre_planes"] == SHIPPED_PLANES + 2 * NLEV


# -- the bits ---------------------------------------------------------------

#: How far a divergence-only checkpoint array may sit from the shipped one,
#: as a fraction of the shipped array's largest magnitude: fourteen times
#: the 7.0e-11 the one-ulp emulation measured (module docstring).
ROUNDING_TOLERANCE = 1e-9


def test_skip_unread_planes_keeps_every_checkpoint_byte(arms):
    shipped = _hashes(arms["shipped"][0], 10)
    assert len(shipped) > 10
    assert _hashes(arms["unread"][0], 10) == shipped


@pytest.mark.parametrize("arm", ["divergence", "diet"])
def test_divergence_only_mass_flux_agrees_with_every_checkpoint_array_to_rounding(arms, arm):
    """Byte-identical on the hosts measured; to rounding on every CPU.

    0.1.8 asserted the bytes here and its publish failed on GitHub's
    hosted runner, whose BLAS kernel returns different bits for the
    narrower M.  An array that matches its hash is equal; one that does
    not is held to ROUNDING_TOLERANCE of its own largest value, and an
    integer or boolean array is always held to equality."""
    shipped_hashes = _hashes(arms["shipped"][0], 10)
    arm_hashes = _hashes(arms[arm][0], 10)
    assert len(shipped_hashes) > 10
    assert arm_hashes.keys() == shipped_hashes.keys()
    shipped = read_checkpoint(arms["shipped"][0] / "arwen_global_step00000010.npz")[1]
    dieted = read_checkpoint(arms[arm][0] / "arwen_global_step00000010.npz")[1]
    assert shipped.keys() == dieted.keys()
    for name in shipped:
        a = np.asarray(shipped[name])
        b = np.asarray(dieted[name])
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if np.array_equal(a, b):
            continue
        assert a.dtype.kind in "fc", f"{name}: a {a.dtype} array moved"
        assert np.all(np.isfinite(b)), name
        scale = float(np.max(np.abs(a)))
        assert scale > 0.0, name
        moved = float(np.max(np.abs(a - b)))
        assert moved <= ROUNDING_TOLERANCE * scale, (
            f"{name}: moved {moved:.3e}, {moved / scale:.3e} of its largest value")


def test_the_ledger_is_still_bit_neutral(arms):
    assert _hashes(arms["no-ledger"][0], 10) == _hashes(arms["shipped"][0], 10)


def test_the_dieted_run_restarts_to_the_bit(arms, tmp_path):
    outdir = arms["diet"][0]
    cfg = load_config(_config(tmp_path, "diet", ARMS["diet"]))
    resumed = tmp_path / "resumed"
    run(cfg, resumed, restart=outdir / "arwen_global_step00000005.npz")
    straight = read_checkpoint(outdir / "arwen_global_step00000010.npz")[1]
    again = read_checkpoint(resumed / "arwen_global_step00000010.npz")[1]
    assert straight.keys() == again.keys()
    for name in straight:
        assert np.array_equal(straight[name], again[name]), name


def test_the_dieted_banded_step_is_the_dieted_resident_step(arms, tmp_path):
    """The banded step reads the same switches (its grid view and its
    assemble chunk): at two latitude bands the dieted run's checkpoint is
    the dieted resident run's, byte for byte."""
    from dataclasses import replace

    cfg = replace(load_config(_config(tmp_path, "diet", ARMS["diet"])),
                  latitude_bands=2, duration_s=1500.0)
    banded = tmp_path / "banded"
    assert run(cfg, banded)["status"] == "pass"
    assert _hashes(banded, 5) == _hashes(arms["diet"][0], 5)


# -- the operator and the door ---------------------------------------------

@pytest.mark.parametrize("precision,tolerance", [("float64", 1e-13), ("float32", 1e-6)])
def test_divergence_from_fourier_is_the_full_analysis_divergence(precision, tolerance):
    """The narrower call against the full analysis, to rounding.

    Equal in float64 on the hosts measured and not on GitHub's hosted
    runners (0.1.8's publish and push tests).  In float32 the AVX2 OpenBLAS kernels move
    2,519 of the 19,360 coefficients by at most 2.3e-07 of the largest
    (1.9 float32 epsilons).  The float64 bound, about 450 float64
    epsilons, is a margin over that kind of difference and not a runner
    measurement: the failing assertion reported no size."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision=precision)
    vector = VorticityDivergenceOperator(transform)
    rng = np.random.default_rng(7)
    shape = (NLEV, *transform.spectral_shape)
    zeta = transform.project(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    div = transform.project(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    u, v = vector.wind_from_vordiv(zeta, div)
    waist = vector._wind_fourier(np.stack([u, v]))
    _zeta, full = vector._vordiv_from_fourier(waist)
    alone = vector.divergence_from_fourier(waist)
    assert alone.dtype == full.dtype and alone.shape == full.shape
    scale = float(np.max(np.abs(full)))
    assert scale > 0.0
    assert float(np.max(np.abs(alone - full))) <= tolerance * scale
    with pytest.raises(ValueError, match="leading length"):
        vector.divergence_from_fourier(waist.values[:1])


def test_the_transforms_table_is_off_by_default_and_carried_when_on(tmp_path):
    shipped = load_config(_config(tmp_path, "plain", {}))
    assert (shipped.skip_unread_planes,
            shipped.divergence_only_mass_flux) == (False, False)
    for key in ("skip_unread_planes", "divergence_only_mass_flux"):
        assert key not in shipped.config_identity
    named_off = load_config(_config(tmp_path, "off", {
        "skip_unread_planes": False, "divergence_only_mass_flux": False}))
    assert named_off.config_hash == shipped.config_hash
    hashes = {shipped.config_hash}
    for key in ("skip_unread_planes", "divergence_only_mass_flux"):
        cfg = load_config(_config(tmp_path, key, {key: True}))
        assert cfg.config_identity[key] is True
        hashes.add(cfg.config_hash)
    assert len(hashes) == 3
    # The cut that moved bits is not a key this table takes.
    for bad_key in ("fused_layout", "chunk_by_field"):
        bad = _config(tmp_path, f"bad-{bad_key}", {}, f"[transforms]\n{bad_key} = true\n")
        with pytest.raises(ValueError, match=r"unknown keys in \[transforms\]"):
            load_config(bad)
    text = _config(tmp_path, "typed", {}).read_text(encoding="utf-8")
    typed = tmp_path / "typed2.toml"
    typed.write_text(text + '[transforms]\nskip_unread_planes = "yes"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="transforms.skip_unread_planes"):
        load_config(typed)
