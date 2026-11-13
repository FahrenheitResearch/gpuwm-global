"""The twin-scan instrument plants exactly what it says and measures zero
where nothing was planted.

THE BREAKAGE THIS PREVENTS.  `tools/arwen_global_twin_scan.py` writes the
perturbed starts of an amplitude scan and reads their responses.  A planter
that moved a second array, rounded the unperturbed coefficients, or wrote a
file the restart door refuses would make every twin difference a statement
about the instrument; a reader with a wrong normalisation would print
plausible small numbers.  So on the tiny CPU configuration:

* a zero-amplitude start is byte-identical to the input in every array, is
  admitted by the package's own checkpoint reader under the run's config
  hash, and a restart from it lands on the cold run's last checkpoint byte
  for byte;
* the pattern has unit grid RMS by the package's own synthesis and
  quadrature, not by the formula the tool uses;
* a planted amplitude reads back as that RMS theta difference, a plus and
  minus pair at step 0 reads a relative nonlinearity of zero, and an arm
  compared with itself reads correlation 1 and ratio 1;
* the one-bit twin differs from its input in one element by one unit in
  the last place.

No device: numpy at T3, four levels.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.runner import build_model_and_cold_state, run

TREE = Path(__file__).resolve().parents[1]
CONFIG = _shipped_configs() / "arwen_global_moist_smoke.toml"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "_arwen_global_twin_scan", TREE / "tools" / "arwen_global_twin_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cold(tmp_path_factory):
    out = tmp_path_factory.mktemp("cold")
    run(load_config(str(CONFIG)), out, overwrite=True)
    return out


def _steps(directory: Path) -> dict[int, Path]:
    return {int(p.name[-12:-4]): p for p in sorted(directory.glob("*_step*.npz"))}


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files
                if name != "__metadata__"}


def _perturb(tool, base, out, amplitude):
    return tool.main(["perturb", str(base), str(out), "--config", str(CONFIG),
                      "--amplitude-k", repr(amplitude), "--max-degree", "3",
                      "--verify"])


def test_zero_amplitude_is_the_input_and_restarts_onto_the_cold_run(tool, cold, tmp_path):
    steps = _steps(cold)
    start = steps[0]
    zero = tmp_path / "zero.npz"
    assert _perturb(tool, start, zero, 0.0) == 0
    a, b = _arrays(start), _arrays(zero)
    assert set(a) == set(b)
    for name in a:
        assert a[name].dtype == b[name].dtype and a[name].tobytes() == b[name].tobytes(), name
    cfg = load_config(str(CONFIG))
    read_checkpoint(zero, expected_config_hash=cfg.config_hash,
                    semi_implicit_scheme=cfg.semi_implicit_scheme,
                    integrator=cfg.integrator)
    resumed = tmp_path / "resumed"
    run(cfg, resumed, restart=zero, overwrite=True)
    last = max(steps)
    x, y = _arrays(steps[last]), _arrays(_steps(resumed)[last])
    for name in x:
        assert x[name].tobytes() == y[name].tobytes(), name
    report = tool.cmd_measure(tool.build_parser().parse_args(
        ["measure", "--config", str(CONFIG), "--control", str(cold),
         "--arm", f"zero={resumed}"]))
    rows = [r for r in report["rows"] if r.get("arm") == "zero"]
    assert rows and all(r["identical"] and r["energy_j_kg"] == 0.0 for r in rows)


def test_pattern_has_unit_rms_by_the_package_synthesis(tool):
    cfg = load_config(str(CONFIG))
    model, _ = build_model_and_cold_state(cfg)
    transform = model.transform
    h = tool.horizontal_pattern(cfg.truncation, 7, 3)
    grid = np.asarray(transform.backend.to_numpy(transform.inverse(
        transform.backend.asarray(h, dtype=transform.backend.complex_dtype))))
    assert math.isclose(transform.grid.global_mean(grid ** 2), 1.0, rel_tol=1e-9)
    assert abs(transform.grid.global_mean(grid)) < 1e-12
    # the same seed and degree give the same numbers at a larger truncation
    big = tool.horizontal_pattern(10, 7, 3)
    assert np.array_equal(big[:4, :4], h) and not np.any(big[4:]) and not np.any(big[:, 4:])


def test_planted_amplitude_pair_and_compare_read_back(tool, cold, tmp_path):
    start = _steps(cold)[0]
    amplitude = 0.25
    arms = {}
    for name, a in (("plus", amplitude), ("minus", -amplitude)):
        d = tmp_path / name
        d.mkdir()
        _perturb(tool, start, d / start.name, a)
        arms[name] = d
    base, plus = _arrays(start), _arrays(arms["plus"] / start.name)
    changed = [n for n in base if base[n].tobytes() != plus[n].tobytes()]
    assert changed == ["atmosphere__theta"]
    report = tool.cmd_measure(tool.build_parser().parse_args(
        ["measure", "--config", str(CONFIG), "--control", str(cold),
         "--arm", f"plus={arms['plus']}", "--arm", f"minus={arms['minus']}",
         "--pair", "a=plus,minus", "--compare", "plus=plus"]))
    cfg = load_config(str(CONFIG))
    p_full, dp = tool.reference_levels(cfg)
    w = tool.vertical_profile(p_full, tool.DEFAULT_CENTRE_HPA, tool.DEFAULT_WIDTH_LN_P)
    expected = amplitude * math.sqrt(float(np.sum(dp * w * w) / np.sum(dp)))
    plus_row = next(r for r in report["rows"] if r.get("arm") == "plus" and r["step"] == 0)
    assert math.isclose(plus_row["rms_theta_k"], expected, rel_tol=1e-9)
    assert plus_row["kinetic_j_kg"] == 0.0 and plus_row["surface_j_kg"] == 0.0
    pair = next(r for r in report["rows"] if r.get("pair") == "a" and r["step"] == 0)
    assert pair["relative_nonlinearity"] < 1e-12
    same = next(r for r in report["rows"] if r.get("compare") == "plus~plus" and r["step"] == 0)
    assert math.isclose(same["correlation"], 1.0) and math.isclose(same["amplitude_ratio"], 1.0)
    assert all(math.isclose(v, 1.0) for v in same["band_correlation"].values())


def test_bitflip_moves_one_element_by_one_ulp(tool, cold, tmp_path):
    start = _steps(cold)[0]
    out = tmp_path / "bit.npz"
    assert tool.main(["bitflip", str(start), str(out), "--level", "1",
                      "--n", "2", "--m", "1"]) == 0
    a, b = _arrays(start), _arrays(out)
    for name in a:
        if name != "atmosphere__theta":
            assert a[name].tobytes() == b[name].tobytes(), name
    x, y = a["atmosphere__theta"], b["atmosphere__theta"]
    assert np.count_nonzero(x != y) == 1
    assert y[1, 2, 1].imag == x[1, 2, 1].imag
    assert abs(y[1, 2, 1].real - x[1, 2, 1].real) == np.spacing(abs(x[1, 2, 1].real))
    cfg = load_config(str(CONFIG))
    read_checkpoint(out, expected_config_hash=cfg.config_hash,
                    semi_implicit_scheme=cfg.semi_implicit_scheme,
                    integrator=cfg.integrator)


def test_a_later_checkpoint_is_refused(tool, cold, tmp_path):
    later = _steps(cold)[max(_steps(cold))]
    with pytest.raises(SystemExit):
        _perturb(tool, later, tmp_path / "x.npz", 0.1)
