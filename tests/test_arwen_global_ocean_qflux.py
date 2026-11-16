"""The heat-flux correction's calibration file (arwen_global.ocean_qflux).

Gate OC-5's Q-flux parts, on known answers:

* the month deposit and its read-back are exact: a constant flux deposited
  over a window that crosses a mid-month instant reads back as itself in
  both months it touched, and the weights add to the window's length;
* the months a run needs, across a mid-month instant, a year end and an
  instant that is exactly a mid-month one;
* the file's read at an instant, with a month missing on a column;
* the refusals: another grid, physics, climatology, a month without
  coverage, a checkpoint without calibration arrays or record, two
  checkpoints of one run, and runs of different physics or climatology;
* the build adds runs, is the same bytes in any order of its inputs, and
  its smoothing keeps a degree-5 pattern and removes a degree-18 one.

The checkpoints are given to the builder through
``arwen_global.checkpoint.read_checkpoint`` replaced by a table of
(metadata, arrays): the builder reads nothing else from them, and the
checkpoint format has its own tests.  One test writes a real checkpoint
(the smoke config's cold state carrying the accumulators) and builds from
it through the verified reader.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import numpy as np
import pytest

from arwen_global.ocean_coupler import QCAL_SUM, QCAL_WEIGHT
from arwen_global.ocean_qflux import (
    CALIBRATION_KEYS, QFLUX_SCHEMA, QFluxFile, QFluxRefused, build_qflux, deposit,
    load_qflux, main, months_needed,
)
from arwen_global.ocean_slab import month_bracket
from arwen_global.spectral.grid import GaussianGrid
from arwen_global.surface_climatology import mid_month

DT = 300.0
START = datetime(2026, 11, 10, 18, tzinfo=timezone.utc)
CLIM_SHA = "c" * 64
PHYSICS = {"physics_mode": "reference", "reference_physics": {"x": 1}}


def _grid():
    return GaussianGrid.create(21)


def _shape(grid):
    return int(grid.nlat), int(grid.nlon)


def test_deposit_and_read_back_are_exact_across_a_mid_month_instant():
    shape = (4, 6)
    total = np.zeros((12, *shape))
    weight = np.zeros((12, *shape))
    columns = np.ones(shape, dtype=bool)
    columns[0, 0] = False
    flux = np.full(shape, 37.25)
    flux[0, 0] = np.nan  # off the columns: never read
    when = START
    steps = 12 * 24 * 10  # ten days, across 16 November 00 UTC
    for _ in range(steps):
        deposit(total, weight, flux_w_m2=flux, columns=columns, when=when, dt_s=DT, xp=np)
        when += timedelta(seconds=DT)
    touched = sorted(int(m) for m in np.flatnonzero(weight[:, 1, 1] > 0.0))
    assert touched == [9, 10, 11]  # October, November, December
    for m in touched:
        np.testing.assert_allclose(total[m][columns] / weight[m][columns], 37.25,
                                   rtol=1.0e-13)
    np.testing.assert_allclose(weight.sum(axis=0)[columns], steps * DT, rtol=1.0e-13)
    assert not total[:, 0, 0].any() and not weight[:, 0, 0].any()
    assert np.isfinite(total).all()


def test_deposit_weights_are_the_month_weights():
    total = np.zeros((12, 1, 1))
    weight = np.zeros((12, 1, 1))
    i0, i1, t0, t1 = month_bracket(START)
    deposit(total, weight, flux_w_m2=np.full((1, 1), 10.0), columns=np.ones((1, 1), bool),
            when=START, dt_s=DT, xp=np)
    w = (START - t0).total_seconds() / (t1 - t0).total_seconds()
    assert weight[i0, 0, 0] == DT * (1.0 - w) and weight[i1, 0, 0] == DT * w
    assert total[i0, 0, 0] == 10.0 * DT * (1.0 - w)


def test_months_needed():
    assert months_needed(START, 10 * 86400.0) == {9, 10, 11}
    # From an instant that is exactly a mid-month one, for no time: that month.
    assert months_needed(mid_month(2026, 11), 0.0) == {10}
    # Up to exactly the next mid-month instant: those two.
    span = (mid_month(2026, 12) - mid_month(2026, 11)).total_seconds()
    assert months_needed(mid_month(2026, 11), span) == {10, 11}
    # Across a year end and the next mid-month instant.
    assert months_needed(datetime(2026, 12, 20, tzinfo=timezone.utc), 30 * 86400.0) == {11, 0, 1}
    assert months_needed(START, 400 * 86400.0) == set(range(12))
    with pytest.raises(ValueError):
        months_needed(START, -1.0)


def _qfile(grid, flux=None, coverage=None, *, provenance=None):
    shape = _shape(grid)
    if flux is None:
        flux = np.arange(12, dtype=np.float64)[:, None, None] + np.zeros((12, *shape))
    if coverage is None:
        coverage = np.where(np.isfinite(flux), 86400.0, 0.0)
    if provenance is None:
        provenance = {"schema": QFLUX_SCHEMA, "physics_identity": PHYSICS,
                      "climatology_sha256": CLIM_SHA}
    return QFluxFile(path=Path("q.npz"), sha256="0" * 64, flux_w_m2=flux,
                     coverage_s=coverage, latitude_deg=grid.latitude_deg,
                     longitude_deg=grid.longitude_deg, provenance=provenance)


def test_the_file_reads_with_the_month_weights_and_a_missing_month_reads_the_other():
    grid = _grid()
    flux = np.arange(12, dtype=np.float64)[:, None, None] * 10.0 + np.zeros((12, *_shape(grid)))
    i0, i1, t0, t1 = month_bracket(START)
    w = (START - t0).total_seconds() / (t1 - t0).total_seconds()
    flux[i1, 0, 0] = np.nan
    flux[i0, 0, 1] = np.nan
    flux[i0, 0, 2] = flux[i1, 0, 2] = np.nan
    out = _qfile(grid, flux).at(START)
    assert out.dtype == np.float64
    assert out[1, 1] == 10.0 * i0 * (1.0 - w) + 10.0 * i1 * w
    assert out[0, 0] == 10.0 * i0
    assert out[0, 1] == 10.0 * i1
    assert out[0, 2] == 0.0


def test_check_passes_its_own_run_and_refuses_each_difference():
    grid = _grid()
    q = _qfile(grid)
    kw = dict(latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
              physics_identity=PHYSICS, climatology_sha256=CLIM_SHA, months={9, 10, 11})
    q.check(**kw)
    other = GaussianGrid.create(31)
    with pytest.raises(QFluxRefused, match="grid"):
        q.check(**{**kw, "latitude_deg": other.latitude_deg,
                   "longitude_deg": other.longitude_deg})
    with pytest.raises(QFluxRefused, match="reference_physics"):
        q.check(**{**kw, "physics_identity": {**PHYSICS, "reference_physics": {"x": 2}}})
    with pytest.raises(QFluxRefused, match="climatology"):
        q.check(**{**kw, "climatology_sha256": "d" * 64})
    coverage = np.full((12, *_shape(grid)), 86400.0)
    coverage[[0, 1]] = 0.0
    flux = np.where(coverage > 0.0, 1.0, np.nan)
    thin = _qfile(grid, flux, coverage)
    thin.check(**kw)
    with pytest.raises(QFluxRefused, match="January, February"):
        thin.check(**{**kw, "months": {11, 0, 1}})


# ---------------------------------------------------------------------------
# the build, from checkpoints
# ---------------------------------------------------------------------------

def _run(grid, *, flux_by_month, days_by_month, config="a", start="2026-11-10T18:00:00Z",
         physics=PHYSICS, clim=CLIM_SHA, spinup=2.0, calibration=True):
    shape = _shape(grid)
    total = np.zeros((12, *shape))
    weight = np.zeros((12, *shape))
    for month, days in days_by_month.items():
        weight[month] = days * 86400.0
        total[month] = flux_by_month[month] * weight[month]
    block = {"ocean_mode": "climatology", "climatology_sha256": clim, "start_utc": start}
    if calibration:
        # The record the coupler writes (ocean_coupler.SlabProvider
        # .identity_block); the builder reads CALIBRATION_KEYS and compares
        # every key between runs.
        block["qflux_calibration"] = {"spinup_days": spinup, "grid": list(shape),
                                      "physics_identity": physics,
                                      "ocean_columns_sha256": "f" * 64}
        assert set(CALIBRATION_KEYS) <= set(block["qflux_calibration"])
    config_hash = hashlib.sha256(config.encode()).hexdigest()
    metadata = {"config_hash": config_hash, "step": 2880, "time_s": 864000.0,
                "self_sha256": hashlib.sha256((config + start).encode()).hexdigest(),
                "physics_metadata": {"surface_boundary": block}}
    arrays = {f"physics__{QCAL_SUM}": total, f"physics__{QCAL_WEIGHT}": weight}
    return metadata, arrays


@pytest.fixture
def checkpoints(monkeypatch, tmp_path):
    table = {}

    def fake_read(path, **_):
        metadata, arrays = table[str(Path(path).name)]
        return dict(metadata), {k: v.copy() for k, v in arrays.items()}

    import arwen_global.checkpoint

    monkeypatch.setattr(arwen_global.checkpoint, "read_checkpoint", fake_read)

    def put(name, record):
        table[name] = record
        return tmp_path / name

    return put


def test_the_build_adds_runs_and_reads_back(checkpoints, tmp_path):
    grid = _grid()
    shape = _shape(grid)
    rng = np.random.default_rng(3)
    fa = {10: rng.normal(0, 30, shape), 11: rng.normal(0, 30, shape)}
    fb = {10: rng.normal(0, 30, shape), 0: rng.normal(0, 30, shape)}
    a = checkpoints("a.npz", _run(grid, flux_by_month=fa, days_by_month={10: 6.0, 11: 4.0}))
    b = checkpoints("b.npz", _run(grid, flux_by_month=fb, days_by_month={10: 2.0, 0: 3.0},
                                  config="b"))
    out = tmp_path / "q.npz"
    provenance = build_qflux([a, b], out=out)
    q = load_qflux(out)
    assert provenance["schema"] == QFLUX_SCHEMA == q.provenance["schema"]
    np.testing.assert_allclose(q.flux_w_m2[10], (6.0 * fa[10] + 2.0 * fb[10]) / 8.0,
                               rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(q.flux_w_m2[11], fa[11], rtol=1.0e-12)
    np.testing.assert_allclose(q.flux_w_m2[0], fb[0], rtol=1.0e-12)
    assert np.isnan(q.flux_w_m2[[1, 2, 3, 4, 5, 6, 7, 8, 9]]).all()
    assert q.coverage_days()[10] == 8.0 and q.coverage_days()[3] == 0.0
    np.testing.assert_array_equal(q.latitude_deg, grid.latitude_deg)
    np.testing.assert_array_equal(q.longitude_deg, grid.longitude_deg)
    assert q.provenance["physics_identity"] == PHYSICS
    assert q.provenance["climatology_sha256"] == CLIM_SHA
    assert q.provenance["spinup_days"] == 2.0
    assert q.provenance["ice_reference_thickness_m"] == {"nh": 2.0, "sh": 1.0}
    assert len(q.provenance["runs"]) == 2
    q.check(latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
            physics_identity=PHYSICS, climatology_sha256=CLIM_SHA, months={10, 11})
    # The same inputs in the other order are the same bytes.
    again = tmp_path / "q2.npz"
    build_qflux([b, a], out=again)
    assert out.read_bytes() == again.read_bytes()


def test_the_build_refuses(checkpoints, tmp_path):
    grid = _grid()
    flux = {10: np.ones(_shape(grid))}
    days = {10: 1.0}
    good = checkpoints("good.npz", _run(grid, flux_by_month=flux, days_by_month=days))
    bare = _run(grid, flux_by_month=flux, days_by_month=days, config="bare")
    bare = (bare[0], {})
    cases = {
        "carries no calibration arrays": [checkpoints("bare.npz", bare)],
        "no qflux_calibration record": [checkpoints("norec.npz", _run(
            grid, flux_by_month=flux, days_by_month=days, config="n", calibration=False))],
        "checkpoints of one run": [good, checkpoints("later.npz", _run(
            grid, flux_by_month=flux, days_by_month={10: 2.0}))],
        "different physics": [good, checkpoints("phys.npz", _run(
            grid, flux_by_month=flux, days_by_month=days, config="p",
            physics={"physics_mode": "none"}))],
        "different spinup_days": [good, checkpoints("spin.npz", _run(
            grid, flux_by_month=flux, days_by_month=days, config="s", spinup=1.0))],
        "prescribed the climatology": [good, checkpoints("clim.npz", _run(
            grid, flux_by_month=flux, days_by_month=days, config="c", clim="e" * 64))],
    }
    for message, paths in cases.items():
        with pytest.raises(QFluxRefused, match=message):
            build_qflux(paths, out=tmp_path / "x.npz")
    assert not (tmp_path / "x.npz").exists()
    assert main([str(paths[0]), str(paths[1]), "--out", str(tmp_path / "x.npz")]) == 2
    assert main([str(good), "--out", str(tmp_path / "y.npz")]) == 0
    assert load_qflux(tmp_path / "y.npz").coverage_days()[10] == 1.0


def test_the_smoothing_keeps_degree_5_and_removes_degree_18(checkpoints, tmp_path):
    grid = GaussianGrid.create(31)
    lat, lon = np.meshgrid(np.radians(grid.latitude_deg), np.radians(grid.longitude_deg),
                           indexing="ij")
    low = 20.0 * np.cos(lat) ** 5 * np.cos(5 * lon)
    high = 20.0 * np.cos(lat) ** 18 * np.cos(18 * lon)
    path = checkpoints("s.npz", _run(grid, flux_by_month={10: low + high},
                                     days_by_month={10: 5.0}))
    out = tmp_path / "s.npz"
    provenance = build_qflux([path], out=out, smoothing_truncation=10)
    q = load_qflux(out)
    assert provenance["smoothing_truncation"] == 10
    np.testing.assert_allclose(q.flux_w_m2[10], low, rtol=0.0, atol=1.0e-9)
    assert np.isnan(q.flux_w_m2[3]).all()




def test_the_build_reads_a_real_checkpoint(tmp_path):
    """The real checkpoint path: a cold state of the shipped smoke config
    carrying the two accumulators and a calibration record, written by
    write_checkpoint and read back, verified, by the builder."""
    from arwen_global.checkpoint import write_checkpoint
    from arwen_global.config import load_config
    from arwen_global.configs_dir import config_root
    from arwen_global.runner import build_model_and_cold_state

    cfg = load_config(str(config_root() / "arwen_global_moist_smoke.toml"))
    model, state = build_model_and_cold_state(cfg)
    grid = model.transform.grid
    shape = (int(grid.nlat), int(grid.nlon))
    total = np.zeros((12, *shape))
    weight = np.zeros((12, *shape))
    flux = np.linspace(-80.0, 80.0, shape[0] * shape[1]).reshape(shape)
    when = START
    for _ in range(24):
        deposit(total, weight, flux_w_m2=flux, columns=np.ones(shape, bool), when=when,
                dt_s=3600.0, xp=np)
        when += timedelta(hours=1)
    state.physics_state.arrays[QCAL_SUM] = total
    state.physics_state.arrays[QCAL_WEIGHT] = weight
    state.physics_state.metadata["surface_boundary"] = {
        "ocean_mode": "climatology", "climatology_sha256": CLIM_SHA,
        "start_utc": "2026-11-10T18:00:00Z",
        "qflux_calibration": {"spinup_days": 0.0, "grid": list(shape),
                              "physics_identity": PHYSICS}}
    path = write_checkpoint(tmp_path / "cal.npz", state, config_hash=cfg.config_hash,
                            to_numpy=model.transform.backend.to_numpy)
    out = tmp_path / "q.npz"
    provenance = build_qflux([path], out=out)
    q = load_qflux(out)
    assert provenance["runs"][0]["config_hash"] == cfg.config_hash
    for month in (9, 10):  # 10-11 November: October and November
        np.testing.assert_allclose(q.flux_w_m2[month], flux, rtol=1.0e-12, atol=1.0e-12)
    assert abs(sum(q.coverage_days()) - 1.0) < 1.0e-12
    np.testing.assert_array_equal(q.latitude_deg, grid.latitude_deg)
    np.testing.assert_array_equal(q.longitude_deg, grid.longitude_deg)


# ---------------------------------------------------------------------------
# the second pass (ocean_qflux.second_pass): a known drift gives a known flux
# ---------------------------------------------------------------------------

def _second_pass_inputs(tmp_path, *, days=40, rate_k_day=-0.01):
    """A flat climatology (290 K, no ice, 50 m), a base file of 5 W/m2 against
    it, and one slab run's daily summaries whose open water cools by
    ``rate_k_day`` a day from 2026-03-01; one column holds ice, one is land."""
    import json

    from arwen_global.surface_climatology import write_climatology

    grid = _grid()
    shape = _shape(grid)
    ocean = np.ones(shape, dtype=bool)
    ocean[0, 0] = False
    planes = {"sst_k": np.full((12, *shape), 290.0), "sea_ice_fraction": np.zeros((12, *shape)),
              "mixed_layer_depth_m": np.full((12, *shape), 50.0)}
    clim = tmp_path / "clim.npz"
    sha = write_climatology(clim, planes, ocean=ocean, latitude_deg=grid.latitude_deg,
                            longitude_deg=grid.longitude_deg,
                            provenance={"schema": "gpuwm.arwen-global-surface-climatology/v1"})
    q = _qfile(grid, flux=np.full((12, *shape), 5.0),
               provenance={"schema": QFLUX_SCHEMA, "physics_identity": PHYSICS,
                           "climatology_sha256": sha})
    base = tmp_path / "base.npz"
    np.savez(base, calibration_flux_w_m2=q.flux_w_m2, coverage_s=q.coverage_s,
             latitude_deg=q.latitude_deg, longitude_deg=q.longitude_deg,
             provenance=np.array(json.dumps(q.provenance)))
    run = tmp_path / "run" / "L1"
    run.mkdir(parents=True)
    day0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for k in range(days):
        day = day0 + timedelta(days=k)
        ice = np.zeros(shape)
        ice[1, 1] = 0.5
        np.savez(run / f"arwen_global_daily_{day:%Y-%m-%d}.npz",
                 metadata=np.array(json.dumps({"day": f"{day:%Y-%m-%d}"})),
                 skin_temperature_mean_k=np.full(shape, 290.3 + rate_k_day * k),
                 sea_ice_fraction_mean=ice.astype(np.float32))
    return grid, clim, base, tmp_path / "run"


def test_the_second_pass_books_a_known_drift(tmp_path):
    from arwen_global.ocean_slab import RHO_CP_SEAWATER_J_M3_K

    _, clim, base, run = _second_pass_inputs(tmp_path)
    out = tmp_path / "q2.npz"
    assert main(["second-pass", str(base), "--climatology", str(clim), "--slab-run", str(run),
                 "--out", str(out), "--relaxation-days", "none"]) == 0
    q = load_qflux(out)
    expected = 5.0 + RHO_CP_SEAWATER_J_M3_K * 50.0 * (-0.01 / 86400.0)  # -23.69 W/m2 lacking
    record = q.provenance["second_pass"]
    # 2 March to 9 April instants: February (from the March instants before the 16th), March, April
    corrected = [m for m in range(12) if record["month_corrected_days"][m] > 0.0]
    assert corrected == [1, 2, 3]
    for month in corrected:
        np.testing.assert_allclose(q.flux_w_m2[month][2:, 2:], expected, rtol=0.0, atol=1.0e-9)
        assert q.flux_w_m2[month][0, 0] == 5.0  # land
        assert q.flux_w_m2[month][1, 1] == 5.0  # the run's ice
    assert (q.flux_w_m2[5] == 5.0).all()
    np.testing.assert_array_equal(q.coverage_s, load_qflux(base).coverage_s)
    assert q.provenance["physics_identity"] == PHYSICS


def test_the_second_pass_adds_the_restoring_and_refuses(tmp_path):
    from arwen_global.ocean_slab import RHO_CP_SEAWATER_J_M3_K

    _, clim, base, run = _second_pass_inputs(tmp_path, rate_k_day=0.0)
    out = tmp_path / "q2.npz"
    assert main(["second-pass", str(base), "--climatology", str(clim), "--slab-run", str(run),
                 "--out", str(out)]) == 0
    # held 0.3 K above a 365-day restoring: the restoring took rho c h 0.3 / tau out of it
    expected = 5.0 + RHO_CP_SEAWATER_J_M3_K * 50.0 * 0.3 / (365.0 * 86400.0)
    np.testing.assert_allclose(load_qflux(out).flux_w_m2[2][2:, 2:], expected, rtol=0.0,
                               atol=1.0e-9)
    (next(run.rglob("arwen_global_daily_2026-03-10.npz"))).unlink()
    assert main(["second-pass", str(base), "--climatology", str(clim), "--slab-run", str(run),
                 "--out", str(out)]) == 2


def test_the_second_pass_skips_a_runs_first_days(tmp_path):
    import json

    from arwen_global.ocean_slab import RHO_CP_SEAWATER_J_M3_K

    _, clim, base, run = _second_pass_inputs(tmp_path)
    # the start's own anomaly: 1 K above the steady drift on 1 March, gone by 11 March
    for path in sorted(run.rglob("arwen_global_daily_*.npz")):
        k = (datetime.strptime(path.name[19:29], "%Y-%m-%d") - datetime(2026, 3, 1)).days
        if k < 10:
            with np.load(path) as data:
                fields = dict(data)
            fields["skin_temperature_mean_k"] = fields["skin_temperature_mean_k"] + 0.1 * (10 - k)
            np.savez(path, **fields)
    steady = 5.0 + RHO_CP_SEAWATER_J_M3_K * 50.0 * (-0.01 / 86400.0)
    args = ["second-pass", str(base), "--climatology", str(clim), "--slab-run", str(run),
            "--relaxation-days", "none"]
    whole, skipped = tmp_path / "whole.npz", tmp_path / "skipped.npz"
    assert main(args + ["--out", str(whole)]) == 0
    assert main(args + ["--out", str(skipped), "--skip-days", "10"]) == 0
    # every day booked: the decay reads as heat lacking in March
    assert load_qflux(whole).flux_w_m2[2][2, 2] < steady - 10.0
    q = load_qflux(skipped)
    record = q.provenance["second_pass"]
    assert record["skip_days"] == 10
    assert record["runs"][0]["booked_from"] == "2026-03-11"
    assert record["runs"][0]["pairs_booked"] == 29
    corrected = [m for m in range(12) if record["month_corrected_days"][m] > 0.0]
    assert 2 in corrected
    for month in corrected:
        np.testing.assert_allclose(q.flux_w_m2[month][2:, 2:], steady, rtol=0.0, atol=1.0e-9)
    # the default is the whole run, and its provenance carries no skip record
    assert "skip_days" not in load_qflux(whole).provenance["second_pass"]
    assert "booked_from" not in json.dumps(load_qflux(whole).provenance)
    # a skip that leaves no pair is refused
    assert main(args + ["--out", str(skipped), "--skip-days", "39"]) == 2
    assert main(args + ["--out", str(skipped), "--skip-days", "-1"]) == 2
