"""The forcing operator (arwen_global.forcing): known answers on the
tiny CPU smoke configuration.

What is pinned here:
- the spellings and their refusals;
- a record round-trips by digest and refuses a tampered byte;
- one pull on a planted departure moves exactly the planted coefficient by
  exactly ``1 - exp(-dt/tau)`` of it, never degree 0 of ln ps, and the
  ledger reads the planted size back per field and level;
- a run pulled toward its own every-step record, and a run that only
  observes, are the free run bit for bit;
- a run pulled toward a record carrying a planted anomaly closes on it,
  the observe-only twin reads the anomaly back as its departure, and a
  restart in the middle of the pull continues it exactly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from arwen_global.checkpoint import read_checkpoint
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root
from arwen_global.forcing import (
    FORCING_HISTORY_KEY,
    FORCING_LEDGER_NAME,
    ForcingOperator,
    ForcingOptions,
    ForcingRecord,
    parse_forcing_spec,
    parse_record_spec,
    record_from_checkpoints,
)
from arwen_global.constants import SPECTRAL_FIELDS
from arwen_global.runner import build_model_and_cold_state, run

CONFIG = str(config_root() / "arwen_global_moist_smoke.toml")
FINAL = "arwen_global_step00000004.npz"


def _arrays(path):
    _meta, arrays = read_checkpoint(path)
    return arrays


def _assert_same_arrays(a: Path, b: Path) -> None:
    arrays_a, arrays_b = _arrays(a), _arrays(b)
    assert arrays_a.keys() == arrays_b.keys()
    for name in arrays_a:
        assert np.array_equal(arrays_a[name], arrays_b[name]), name


def test_spellings_and_refusals():
    path, options = parse_forcing_spec("rec.npz:tau_h=3;fields=theta,vorticity;zero_degree=22;full_degree=10;end_h=12")
    assert path == Path("rec.npz")
    assert options.tau_s == 3 * 3600.0 and options.fields == ("theta", "vorticity")
    assert (options.full_degree, options.zero_degree, options.end_s) == (10, 22, 12 * 3600.0)
    assert options.alpha(300.0) == pytest.approx(1.0 - math.exp(-300.0 / 10800.0), rel=1e-15)
    path, options = parse_forcing_spec(r"C:\runs\rec.npz:tau_h=inf")
    assert path == Path(r"C:\runs\rec.npz") and options.observe_only and options.alpha(300.0) == 0.0
    path, options = parse_forcing_spec("/data/rec.npz")
    assert options == ForcingOptions()
    assert parse_record_spec("out.npz:degree=21;every_h=1;dtype=complex64") == (
        Path("out.npz"), {"degree": 21, "every_s": 3600.0, "dtype": "complex64"})
    with pytest.raises(ValueError, match="unknown forcing option"):
        parse_forcing_spec("rec.npz:tau=3")
    with pytest.raises(ValueError, match="fields"):
        ForcingOptions(fields=("wind",))
    with pytest.raises(ValueError, match="degrees"):
        ForcingOptions(full_degree=10, zero_degree=10)
    with pytest.raises(ValueError, match="tau_s"):
        ForcingOptions(tau_s=0.0)
    with pytest.raises(ValueError, match="level_weights"):
        ForcingOptions(level_weights=(1.0, 2.0))


def _record(times, nlev=4, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    k = degree + 1
    fields = {}
    for name in SPECTRAL_FIELDS:
        shape = (len(times), k, k) if name == "log_surface_pressure" else (len(times), nlev, k, k)
        fields[name] = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return ForcingRecord(times_s=np.asarray(times, dtype=float), fields=fields, degree=degree,
                         source={"test": True})


def test_record_round_trip_digest_and_time_interpolation(tmp_path):
    record = _record([0.0, 10.0, 30.0])
    path = record.write(tmp_path / "rec.npz")
    back = ForcingRecord.read(path)
    assert back.sha256 == record.sha256
    for name in record.fields:
        assert np.array_equal(back.fields[name], record.fields[name])
    # A stored time returns the stored snapshot exactly; between two times
    # the record is their linear blend; outside, nothing.
    assert np.array_equal(back.at("theta", 10.0, 2), record.fields["theta"][1, :, :3, :3])
    blend = back.at("theta", 20.0, 3)
    assert np.allclose(blend, 0.5 * (record.fields["theta"][1] + record.fields["theta"][2]), rtol=0, atol=1e-15)
    assert back.at("theta", 30.5, 3) is None and back.bracket(-1.0) is None
    # One changed byte and the record refuses itself.
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name]) for name in archive.files}
    values["field__theta"][0, 0, 1, 0] += 1e-9
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="digest"):
        ForcingRecord.read(path)
    with pytest.raises(ValueError, match="strictly increasing"):
        _record([0.0, 0.0])


def test_one_pull_moves_the_planted_coefficients_by_the_known_fraction():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    transform = model.transform
    to_numpy = transform.backend.to_numpy
    k = cfg.truncation + 1
    fields = {name: np.array(to_numpy(getattr(state.atmosphere, name)), dtype=np.complex128)[None]
              for name in SPECTRAL_FIELDS}
    planted_theta = 0.75 - 0.25j          # level 1, degree 2, order 1
    planted_mean = 0.5                   # level 0, degree 0: a global mean
    fields["theta"][0, 1, 2, 1] += planted_theta
    fields["theta"][0, 0, 0, 0] += planted_mean
    fields["log_surface_pressure"][0, 0, 0] += 1e-3   # never pulled
    fields["log_surface_pressure"][0, 1, 0] += 2e-3   # pulled
    record = ForcingRecord(times_s=np.array([0.0]), fields=fields, degree=cfg.truncation, source={})
    tau = 4.0 * cfg.dt_s
    options = ForcingOptions(tau_s=tau, full_degree=cfg.truncation, zero_degree=cfg.truncation + 1,
                             fields=("theta", "log_surface_pressure", "vorticity"))
    op = ForcingOperator(record, options, transform=transform, dt_s=cfg.dt_s, nlev=cfg.vertical.nlev)
    alpha = 1.0 - math.exp(-cfg.dt_s / tau)
    assert op.alpha == pytest.approx(alpha, rel=1e-15)
    before = {name: np.array(to_numpy(getattr(state.atmosphere, name))) for name in SPECTRAL_FIELDS}
    pulled = op.apply(state, model)
    assert pulled is not state
    after = {name: np.array(to_numpy(getattr(pulled.atmosphere, name))) for name in SPECTRAL_FIELDS}
    moved = after["theta"] - before["theta"]
    assert moved[1, 2, 1] == pytest.approx(alpha * planted_theta, rel=1e-12)
    assert moved[0, 0, 0] == pytest.approx(alpha * planted_mean, rel=1e-12)
    mask = np.ones_like(moved, dtype=bool)
    mask[1, 2, 1] = mask[0, 0, 0] = False
    assert np.all(moved[mask] == 0)
    lnps = after["log_surface_pressure"] - before["log_surface_pressure"]
    assert lnps[0, 0] == 0 and lnps[1, 0] == pytest.approx(alpha * 2e-3, rel=1e-9)
    for name in ("vorticity", "divergence", "qv"):
        assert np.array_equal(after[name], before[name]), name
    assert pulled.physics_state.metadata[FORCING_HISTORY_KEY][0]["record_sha256"] == record.sha256
    # The ledger reads the plant back: grid RMS of a coefficient c at
    # order m > 0 is |c| sqrt(2 / 4 pi), at m = 0 |c| sqrt(1 / 4 pi).
    row = op.finish()
    assert row["steps"] == 1
    summary = op.ledger.summary()
    theta = summary["fields"]["theta"]["levels"]
    per_day = 86400.0 / cfg.dt_s
    want = abs(planted_theta) * math.sqrt(2.0 / (4.0 * math.pi))
    assert theta[1]["departure_rms"] == pytest.approx(want, rel=1e-9)
    assert theta[1]["pull_rms_per_day"] == pytest.approx(alpha * want * per_day, rel=1e-9)
    assert theta[0]["global_mean_pull_per_day"] == pytest.approx(
        alpha * planted_mean / math.sqrt(4.0 * math.pi) * per_day, rel=1e-9)
    assert theta[2]["departure_rms"] == 0.0 and theta[3]["pull_rms_per_day"] == 0.0
    assert summary["fields"]["vorticity"]["levels"][0]["departure_rms"] == 0.0


def test_pull_window_and_record_coverage_leave_the_state_alone():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    record = _record([100.0, 200.0], nlev=cfg.vertical.nlev, degree=cfg.truncation)
    op = ForcingOperator(record, ForcingOptions(tau_s=cfg.dt_s, full_degree=1, zero_degree=3),
                         transform=model.transform, dt_s=cfg.dt_s, nlev=cfg.vertical.nlev)
    assert op.apply(state, model) is state          # time 0 lies outside the record
    op2 = ForcingOperator(record, ForcingOptions(tau_s=cfg.dt_s, start_s=150.0, record_offset_s=150.0),
                          transform=model.transform, dt_s=cfg.dt_s, nlev=cfg.vertical.nlev)
    assert op2.apply(state, model) is state          # time 0 lies before the window
    assert op.ledger.skipped_uncovered == 1 and op2.ledger.skipped_outside_window == 1
    with pytest.raises(ValueError, match="levels"):
        ForcingOperator(_record([0.0], nlev=3, degree=3), ForcingOptions(),
                        transform=model.transform, dt_s=cfg.dt_s, nlev=cfg.vertical.nlev)
    with pytest.raises(ValueError, match="holds degrees up to 1"):
        ForcingOperator(_record([0.0], nlev=cfg.vertical.nlev, degree=1), ForcingOptions(),
                        transform=model.transform, dt_s=cfg.dt_s, nlev=cfg.vertical.nlev)


def _all_degrees(cfg, **kwargs):
    return ForcingOptions(full_degree=cfg.truncation, zero_degree=cfg.truncation + 1, **kwargs)


def test_a_run_pulled_toward_its_own_record_is_the_free_run_bit_for_bit(tmp_path):
    cfg = load_config(CONFIG)
    free = run(cfg, tmp_path / "free", forcing_record=(tmp_path / "self.npz", {"every_s": cfg.dt_s,
                                                                                "degree": cfg.truncation}))
    assert free["status"] == "pass"
    assert free["forcing_record"]["ntimes"] == 5
    record = ForcingRecord.read(tmp_path / "self.npz")
    assert record.sha256 == free["forcing_record"]["sha256"]
    # The same record cut afterwards from the run's checkpoints holds the
    # same snapshots at the checkpoints' times.
    cut = record_from_checkpoints(sorted((tmp_path / "free").glob("arwen_global_step*.npz")),
                                  degree=cfg.truncation)
    assert list(cut.times_s) == [0.0, 20.0, 40.0]
    for name in SPECTRAL_FIELDS:
        assert np.array_equal(cut.fields[name], record.fields[name][[0, 2, 4]]), name
    assert cut.source["config_hash"] == cfg.config_hash
    # ...and the tool the forcing page cites writes the same record.
    cutter = _load_tool("arwen_global_forcing_record.py")
    ckpts = [str(p) for p in sorted((tmp_path / "free").glob("arwen_global_step*.npz"))]
    assert cutter.main([*ckpts, "--out", str(tmp_path / "cut.npz"), "--degree", str(cfg.truncation)]) == 0
    assert ForcingRecord.read(tmp_path / "cut.npz").sha256 == cut.sha256
    pulled = run(cfg, tmp_path / "pulled",
                 forcing=(tmp_path / "self.npz", _all_degrees(cfg, tau_s=cfg.dt_s, fields=SPECTRAL_FIELDS)))
    observed = run(cfg, tmp_path / "observed",
                   forcing=(tmp_path / "self.npz", _all_degrees(cfg, tau_s=math.inf)))
    for other in ("pulled", "observed"):
        _assert_same_arrays(tmp_path / "free" / FINAL, tmp_path / other / FINAL)
        meta, _ = read_checkpoint(tmp_path / other / FINAL)
        assert FORCING_HISTORY_KEY not in meta["physics_metadata"]
    ledger = json.loads((tmp_path / "pulled" / FORCING_LEDGER_NAME).read_text())
    assert ledger["steps"] == 4
    assert all(row["departure_rms"] == 0.0 for row in ledger["fields"]["theta"]["levels"])
    # The shipped run's receipt carries neither key.
    assert "forcing" not in free and "forcing_record" in free
    assert pulled["forcing"]["steps"] == 4 and observed["forcing"]["observe_only"] is True


def _planted_record(tmp_path, cfg) -> Path:
    run(cfg, tmp_path / "truth", forcing_record=(tmp_path / "truth.npz", {"every_s": cfg.dt_s,
                                                                         "degree": cfg.truncation}))
    record = ForcingRecord.read(tmp_path / "truth.npz")
    record.fields["theta"][:, 2, 1, 0] += 1.0        # a planted large-scale anomaly
    record.source["planted"] = "theta level 2 degree 1 order 0 += 1"
    return record.write(tmp_path / "planted.npz")


def test_a_pulled_run_closes_on_a_planted_anomaly_and_restarts_exactly(tmp_path):
    cfg = load_config(CONFIG)
    planted = _planted_record(tmp_path, cfg)
    options = _all_degrees(cfg, tau_s=2.0 * cfg.dt_s, window_s=20.0)
    observe = _all_degrees(cfg, tau_s=math.inf, window_s=20.0)
    run(cfg, tmp_path / "observed", forcing=(planted, observe))
    forced = run(cfg, tmp_path / "forced", forcing=(planted, options))
    _assert_same_arrays(tmp_path / "truth" / FINAL, tmp_path / "observed" / FINAL)
    free_ledger = json.loads((tmp_path / "observed" / FORCING_LEDGER_NAME).read_text())
    forced_ledger = json.loads((tmp_path / "forced" / FORCING_LEDGER_NAME).read_text())
    # The observing twin is the truth run, so its departure is the plant.
    want = 1.0 / math.sqrt(4.0 * math.pi)
    level2 = free_ledger["fields"]["theta"]["levels"][2]
    assert level2["departure_rms"] == pytest.approx(want, rel=1e-9)
    assert level2["pull_rms_per_day"] == 0.0
    # The pulled run closes on it window by window.
    windows = [w["fields"]["theta"]["departure_rms"][2] for w in forced_ledger["windows"]]
    assert len(windows) == 2 and windows[1] < 0.6 * windows[0]
    assert windows[0] < want
    assert forced_ledger["fields"]["theta"]["levels"][2]["time_mean_pull_rms_per_day"] > 0.0
    meta, _ = read_checkpoint(tmp_path / "forced" / FINAL)
    history = meta["physics_metadata"][FORCING_HISTORY_KEY]
    assert len(history) == 1 and history[0]["from_step"] == 0
    assert forced["forcing"]["record"]["sha256"] == ForcingRecord.read(planted).sha256
    # A restart in the middle of the pull continues it exactly.
    run(cfg, tmp_path / "resumed", restart=tmp_path / "forced" / "arwen_global_step00000002.npz",
        forcing=(planted, options))
    _assert_same_arrays(tmp_path / "forced" / FINAL, tmp_path / "resumed" / FINAL)
    meta_b, _ = read_checkpoint(tmp_path / "resumed" / FINAL)
    assert meta_b["physics_metadata"] == meta["physics_metadata"]
    # The table tool the forcing page cites, on a known answer: set the
    # observing twin (the truth run bit for bit) in the forced slot and the
    # run pulled away from the truth in the free slot, and r reads exactly
    # one once the two have parted.
    tool = _table_tool()
    table = tool.twin_table(tmp_path / "truth", tmp_path / "forced", tmp_path / "observed",
                            offset_s=0.0, bands=[(0, 3)])
    assert [row["hours"] for row in table["rows"]] == [0.0, 20.0 / 3600.0, 40.0 / 3600.0]
    last = table["rows"][-1]["fields"]["theta"]
    assert last["forced_rms"] == [0.0] and last["free_rms"][0] > 0.0 and last["r"] == [1.0]
    ledger = tool.ledger_table(tmp_path / "forced", [500.0])
    assert ledger["steps"] == 4 and ledger["fields"]["theta"]["levels"]


def _table_tool():
    return _load_tool("arwen_global_forcing_table.py")


def _load_tool(name: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location("_" + name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
