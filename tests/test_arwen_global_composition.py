"""Composition replay: known answers on the tiny CPU configuration.

The replay (arwen_global.composition) carries extra fields through the
flow a finished run's checkpoints hold.  What it must never do is touch the
weather, and what it must do is carry a field the way the model's own
semi-Lagrangian transport would.  Each gate here names its answer:

*   every weather checkpoint is the same file, byte for byte, after a
    replay as before it;
*   a constant row stays the constant to rounding with the fixer off,
    because the gather's Lagrange weights sum to one;
*   with the Bermejo-Conde fixer on, each row's float64 mass is held to
    rounding over the whole replay, and a release's carried mass equals
    the mass it was given;
*   a replay cut in two at a checkpoint writes the sidecars an unbroken
    replay writes, bit for bit;
*   a row read from a file at rest under a zero wind comes back unchanged,
    bit for bit;
*   the refusals: a replay config whose hash is not the run's without the
    opt-in, an interval that is not a whole number of substeps, a sidecar
    of another spec.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from arwen_global.composition import (  # noqa: E402
    RECEIPT_NAME,
    SIDECAR_SUFFIX,
    FlowReader,
    initial_rows,
    load_spec,
    mass_kg,
    read_sidecar,
    replay,
    sidecar_path,
    spec_from_mapping,
)
from arwen_global.config import load_config  # noqa: E402
from arwen_global.runner import run  # noqa: E402


_CONFIG = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "composition-replay-t10"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"
[grid]
truncation = 10
[time]
dt_s = 600.0
duration_s = {duration}
output_interval_s = 1200.0
integrator = "sl_si"
[vertical]
coordinate = "pressure_blend"
nlev = 8
[initial]
zonal_wind_m_s = 30.0
perturbation_amplitude = 0.004
zonal_wavenumber = 3
[physics]
mode = "none"
"""


def _write_config(directory: Path, duration: float = 7200.0) -> Path:
    path = directory / "run.toml"
    path.write_text(_CONFIG.format(duration=duration), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def weather(tmp_path_factory):
    """A two-hour T10 run with a checkpoint every 20 minutes."""
    root = tmp_path_factory.mktemp("composition")
    cfg_path = _write_config(root)
    cfg = load_config(cfg_path)
    outdir = root / "run"
    run(cfg, outdir)
    checkpoints = sorted(outdir.glob("arwen_global_step*.npz"))
    assert len(checkpoints) == 7
    return cfg_path, cfg, checkpoints


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_SPEC = {
    "composition": {
        "substep_s": 600.0,
        "fixer": "bermejo_conde",
        "tracer": [
            {"name": "one", "kind": "constant"},
            {"name": "tropics", "kind": "band", "lat_min_deg": -20.0,
             "lat_max_deg": 20.0},
            {"name": "upper", "kind": "band", "lat_min_deg": 30.0,
             "lat_max_deg": 90.0, "p_top_pa": 0.0, "p_bottom_pa": 30000.0},
            {"name": "where", "kind": "origin"},
            {"name": "plume", "kind": "release", "lon_deg": 40.0,
             "lat_deg": 45.0, "radius_km": 1500.0, "p_top_pa": 20000.0,
             "p_bottom_pa": 60000.0, "start_s": 0.0, "duration_s": 3600.0,
             "mass_kg": 1.0e9},
        ],
    }
}


def test_a_replay_leaves_every_weather_checkpoint_byte_identical(weather, tmp_path):
    _cfg_path, cfg, checkpoints = weather
    before = {path.name: _file_sha(path) for path in checkpoints}
    spec = spec_from_mapping(_SPEC)
    receipt = replay(cfg, checkpoints, spec, tmp_path / "replay")
    after = {path.name: _file_sha(path) for path in checkpoints}
    assert before == after
    assert len(receipt["sidecars"]) == len(checkpoints)
    assert (tmp_path / "replay" / RECEIPT_NAME).exists()
    for path in checkpoints:
        assert sidecar_path(tmp_path / "replay", path).exists()


def test_the_bermejo_conde_fixer_holds_every_row_and_books_the_release(weather, tmp_path):
    _cfg_path, cfg, checkpoints = weather
    spec = spec_from_mapping(_SPEC)
    receipt = replay(cfg, checkpoints, spec, tmp_path / "replay")
    ledger = receipt["ledger"]
    # The release emitted exactly its mass over its hour.
    assert ledger["final"]["emitted_kg"]["plume"] == pytest.approx(1.0e9, rel=1e-12)
    for name, drift in ledger["relative_mass_drift"].items():
        assert abs(drift) < 1.0e-10, (name, drift)
    # The plume moved: its last sidecar is not its first emission.
    meta, rows = read_sidecar(receipt["sidecars"][-1])
    assert meta["weather"]["checkpoint"] == checkpoints[-1].name
    assert rows["plume"].max() > 0.0
    assert set(rows) == {"one", "tropics", "upper", "where_x", "where_y",
                         "where_z", "plume"}


def test_a_constant_row_stays_constant_with_the_fixer_off(weather, tmp_path):
    _cfg_path, cfg, checkpoints = weather
    spec = spec_from_mapping({"composition": {
        "substep_s": 600.0, "fixer": "none",
        "tracer": [{"name": "one", "kind": "constant", "value": 1.0}],
    }})
    receipt = replay(cfg, checkpoints, spec, tmp_path / "replay")
    _meta, rows = read_sidecar(receipt["sidecars"][-1])
    assert np.max(np.abs(rows["one"] - 1.0)) < 1.0e-12


def test_two_segments_write_the_sidecars_one_segment_writes(weather, tmp_path):
    _cfg_path, cfg, checkpoints = weather
    spec = spec_from_mapping(_SPEC)
    whole = replay(cfg, checkpoints, spec, tmp_path / "whole")
    first = replay(cfg, checkpoints[:4], spec, tmp_path / "split")
    second = replay(cfg, checkpoints, spec, tmp_path / "split",
                    resume=first["sidecars"][-1])
    assert len(second["sidecars"]) == 3
    for path in checkpoints:
        _ma, a = read_sidecar(sidecar_path(tmp_path / "whole", path))
        _mb, b = read_sidecar(sidecar_path(tmp_path / "split", path))
        assert set(a) == set(b)
        for name in a:
            assert a[name].tobytes() == b[name].tobytes(), (path.name, name)
    assert whole["ledger"]["final"] == second["ledger"]["final"]


def test_a_field_row_at_rest_comes_back_unchanged(weather, tmp_path):
    """A zero wind: the departure point is the arrival point and the gather
    reads the field at its own nodes."""
    _cfg_path, cfg, checkpoints = weather
    reader = FlowReader(cfg)
    nlev, nlat, nlon = reader.shape
    rng = np.random.default_rng(7)
    field = rng.uniform(0.0, 1.0, size=(nlev, nlat, nlon))
    source = tmp_path / "field.npz"
    np.savez(source, ozone=field)
    spec = spec_from_mapping({"composition": {
        "substep_s": 600.0, "fixer": "none",
        "tracer": [{"name": "o3", "kind": "field", "path": str(source),
                    "array": "ozone"}],
    }}, base_dir=tmp_path)
    frame = reader.frame(checkpoints[0])
    rows = initial_rows(spec, frame, reader.transform)
    assert rows["o3"].tobytes() == field.tobytes()
    from arwen_global.semilag.interpolate import zero_stencil
    from arwen_global.semilag.tracers import advect

    stencil = zero_stencil(reader.tables, nlev, xp=np, dtype=np.float64)
    moved = advect(rows, stencil, monotone="quasi_monotone")
    assert np.max(np.abs(moved["o3"] - field)) < 1.0e-14


def test_the_spec_file_round_trips_and_keys_the_result(tmp_path):
    path = tmp_path / "spec.toml"
    path.write_text(
        '[composition]\nsubstep_s = 600.0\n'
        '[[composition.tracer]]\nname = "one"\nkind = "constant"\n',
        encoding="utf-8")
    spec = load_spec(path)
    assert spec.stored_names == ("one",)
    other = spec_from_mapping({"composition": {
        "substep_s": 300.0, "tracer": [{"name": "one", "kind": "constant"}]}})
    assert spec.spec_hash != other.spec_hash
    with pytest.raises(ValueError, match="unknown keys"):
        spec_from_mapping({"composition": {"tracer": [
            {"name": "a", "kind": "constant", "colour": 1.0}]}})
    with pytest.raises(ValueError, match="duplicate"):
        spec_from_mapping({"composition": {"tracer": [
            {"name": "a", "kind": "constant"},
            {"name": "a", "kind": "constant"}]}})
    with pytest.raises(ValueError, match="fixer"):
        spec_from_mapping({"composition": {"fixer": "bermejo_conde_additive",
                           "tracer": [{"name": "a", "kind": "constant"}]}})


def test_the_refusals(weather, tmp_path):
    cfg_path, cfg, checkpoints = weather
    spec = spec_from_mapping(_SPEC)
    other = load_config(_write_config(tmp_path, duration=3600.0))
    assert other.config_hash != cfg.config_hash
    with pytest.raises(ValueError, match="accept_geometry_config"):
        replay(other, checkpoints, spec, tmp_path / "a")
    receipt = replay(other, checkpoints[:2], spec, tmp_path / "b",
                     accept_geometry_config=True)
    assert receipt["geometry_config_differs"] is True
    assert receipt["weather_config_hash"] == cfg.config_hash
    assert receipt["replay_config_hash"] == other.config_hash
    odd = spec_from_mapping({"composition": {
        "substep_s": 900.0, "tracer": [{"name": "a", "kind": "constant"}]}})
    with pytest.raises(ValueError, match="whole number"):
        replay(cfg, checkpoints, odd, tmp_path / "c")
    first = replay(cfg, checkpoints[:3], spec, tmp_path / "d")
    changed = spec_from_mapping({"composition": {
        "substep_s": 600.0, "tracer": [{"name": "one", "kind": "constant"}]}})
    with pytest.raises(ValueError, match="spec"):
        replay(cfg, checkpoints, changed, tmp_path / "d",
               resume=first["sidecars"][-1])


def test_the_ozone_driver_interpolates_linearly_in_log_pressure():
    """A field linear in ln p comes back exact on any target levels, and a
    level outside the source ladder takes the end value."""
    from tools.arwen_global_ozone_field import _vertical_interpolate

    source = np.array([100000.0, 50000.0, 10000.0, 1000.0, 100.0, 1.0])
    values = (3.0 + 0.5 * np.log(source))[:, None, None] * np.ones((1, 2, 3))
    target = np.array([80000.0, 20000.0, 300.0, 0.5])[:, None, None] * np.ones((1, 2, 3))
    target[0, 1, 2] = 120000.0
    got = _vertical_interpolate(source, values, target)
    want = 3.0 + 0.5 * np.log(np.clip(target, 1.0, 100000.0))
    assert np.max(np.abs(got - want)) < 1.0e-12


def test_a_sidecar_that_was_edited_is_refused(weather, tmp_path):
    _cfg_path, cfg, checkpoints = weather
    spec = spec_from_mapping(_SPEC)
    receipt = replay(cfg, checkpoints[:2], spec, tmp_path / "r")
    path = Path(receipt["sidecars"][-1])
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name]) for name in archive.files}
    values["tracer__one"][0, 0, 0] += 1.0e-6
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="fails its hash"):
        read_sidecar(path)
    assert path.name.endswith(SIDECAR_SUFFIX)
