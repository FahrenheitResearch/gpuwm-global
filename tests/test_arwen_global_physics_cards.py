"""The physics half-step is split across the cards, and the answer is the
one-card answer.

A ``--cards P`` run partitions grid space by latitude band.  The physics
runs only the bands a card owns (dynamics.apply_physics over
``BandPipeline.local_slices``); the waists, the plane accumulators and the
extrema cross the row exchange as they do for every other operator, the
grid tracers, the surface and the namespace are gathered whole once per
call, and every band's diagnostics and metadata reach every card before
the suite's ``finish`` merges them.  Every claim below runs the real step
with the native suite behind its fakes on the numpy backend, two ranks on
two threads over loopback TCP, because a mocked transport tests the mock:

PSPLIT-1  each card hands the suite its own bands and no other, and the
          cards together hand it every band of the schedule exactly once
PSPLIT-2  gate BIT-6 over the BIT-5 inventory: every checkpointed array
          and every scalar metric of two steps on two cards equals one
          card at the same band count, on the default semi-Lagrangian
          core and on the Eulerian IMEX core
PSPLIT-3  the same with the pinned host tier holding all three slices on
          both cards, so the gather fills the tier's host slots
PSPLIT-4  the component capture records the one-card vectors on two cards
PSPLIT-5  a card record crosses exactly, type for type, and a value the
          wire cannot carry exactly is refused by name
PSPLIT-6  a tracer floor refusal raised by one card's bands fires on every
          card, because the extrema are folded across the cards first
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from arwen_global import cards
from arwen_global.checkpoint import bundle_arrays
from arwen_global.config import load_config
from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global.insitu.capture import ComponentCapture, attach_capture
from arwen_global.physics.native_suite import ArwenCudaColumnSuite
from arwen_global.runner import build_model_and_cold_state, build_transform
from test_arwen_global_cards import _two_rank_tcp
from test_arwen_global_level5_native import _options
from test_arwen_global_physics_bands import _band_fakes

CONFIG = str(_shipped_configs() / "arwen_global_t21_baroclinic_ten_day.toml")
BANDS = 4
STEPS = 2


def _digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _scalars(metrics, prefix=""):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out.update(_scalars(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float, bool, str)) or value is None:
            out[f"{prefix}{key}"] = value
    return out


def _run(*, integrator=None, session=None, park=False, capture=False):
    """Two steps of the native suite; returns what each check reads."""
    # The flux-form transport's card halo must fit inside a card's 16
    # rows at T21; the semi-Lagrangian core never reads it, and one card
    # never does.
    cfg = replace(
        load_config(CONFIG), latitude_bands=BANDS, host_spill="off",
        card_halo_rows=8,
    )
    if integrator is not None:
        cfg = replace(cfg, integrator=integrator)
    transform = build_transform(cfg)
    model, state = build_model_and_cold_state(
        cfg, transform, card_session=session)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_band_fakes())
    model.physics = suite
    probe = None
    if capture:
        probe = ComponentCapture(
            model.transform.grid.quadrature_weights,
            exchange=model.pipeline.exchange,
        )
        attach_capture(suite, probe)
    if park:
        from arwen_global.sizing import SPILL_SLICES
        from arwen_global.spill import HostTier

        model.host_tier = HostTier(model.transform.backend.xp)
        model.host_tier.slices = list(SPILL_SLICES)
        model.spill_slices = tuple(SPILL_SLICES)
        state = model.park_persistent(state)
    seen = []
    original = suite.step

    def spy(exchange):
        seen.append(tuple(exchange.band))
        return original(exchange)

    suite.step = spy
    metrics = None
    records = []
    for _ in range(STEPS):
        if probe is not None:
            probe.begin_step()
        state, metrics = model.step(state, cfg.dt_s)
        if probe is not None:
            records.extend(
                (call, name, np.asarray(vector).copy())
                for call, name, vector in probe.drain()
            )
    gather = None
    if session is not None and session.world > 1:
        grid = model.transform.grid

        def gather(arrays):
            return cards.gather_named_arrays(session, arrays, grid.nlat, grid.nlon)

    arrays = bundle_arrays(
        state, model.transform.backend.to_numpy,
        model.trajectory_state(), card_gather=gather,
    )
    return {
        "inventory": {name: _digest(value) for name, value in arrays.items()},
        "scalars": _scalars(metrics),
        "metadata": state.physics_state.metadata,
        "bands_seen": seen,
        "local": [(r.start, r.stop) for r in model.pipeline.local_slices()],
        "schedule": [(r.start, r.stop) for r in model.pipeline.slices()],
        "capture": records,
    }


def _two_cards(**kwargs):
    nlat = build_transform(load_config(CONFIG)).grid.nlat

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, BANDS)
        return _run(session=session, **kwargs)

    return _two_rank_tcp(body)


def _same_answer(one, pair):
    for rank, out in pair.items():
        assert sorted(out["inventory"]) == sorted(one["inventory"])
        differing = [
            name for name in one["inventory"]
            if one["inventory"][name] != out["inventory"][name]
        ]
        assert differing == [], f"rank {rank}: {differing}"
        assert out["scalars"] == one["scalars"], f"rank {rank}"
        assert out["metadata"] == one["metadata"], f"rank {rank}"


# ---------------------------------------------------------------- PSPLIT-1


def test_psplit_1_each_card_runs_its_own_bands_and_no_other():
    pair = _two_cards(integrator="sl_si")
    schedule = pair[0]["schedule"]
    assert len(schedule) == BANDS
    union = []
    for rank, out in pair.items():
        local = out["local"]
        assert 0 < len(local) < BANDS
        # Two Strang halves a step, each over this card's bands only.
        assert out["bands_seen"] == local * (2 * STEPS), f"rank {rank}"
        union.extend(local)
    assert sorted(union) == schedule


# ---------------------------------------------------------------- PSPLIT-2


@pytest.mark.parametrize("integrator", ["sl_si", "imex_ssp3"])
def test_psplit_2_two_cards_return_the_one_card_checkpoint(integrator):
    one = _run(integrator=integrator)
    assert one["bands_seen"] == one["schedule"] * (2 * STEPS)
    pair = _two_cards(integrator=integrator)
    _same_answer(one, pair)
    assert len(one["inventory"]) > 100


# ---------------------------------------------------------------- PSPLIT-3


def test_psplit_3_the_gather_fills_the_tier_slots():
    one = _run(integrator="sl_si")
    pair = _two_cards(integrator="sl_si", park=True)
    _same_answer(one, pair)


# ---------------------------------------------------------------- PSPLIT-4


def test_psplit_4_the_capture_records_the_one_card_vectors():
    one = _run(integrator="sl_si", capture=True)
    pair = _two_cards(integrator="sl_si", capture=True)
    assert one["capture"]
    for rank, out in pair.items():
        assert [(c, n) for c, n, _ in out["capture"]] == [
            (c, n) for c, n, _ in one["capture"]], f"rank {rank}"
        for (_, name, left), (_, _, right) in zip(one["capture"], out["capture"]):
            assert np.array_equal(left, right), (rank, name)


# ---------------------------------------------------------------- PSPLIT-5


def test_psplit_5_a_record_crosses_exactly_and_type_for_type():
    record = [{
        "start": 12,
        "diagnostics": {
            "f": 0.1, "nan": float("nan"), "neg_zero": -0.0, "big": 2**70,
            "flag": True, "none": None, "s": "x",
            "f32": np.float32(1.0e-7), "i64": np.int64(-3),
            "tup": (1, 2.5, "a"), "lst": [1, [2, (3,)]],
            "arr": np.arange(6, dtype=np.float32).reshape(2, 3),
            "nested": {"k": {"v": 1.5}},
        },
    }]
    back = cards.decode_record(cards.encode_record(record))
    d0, d1 = record[0]["diagnostics"], back[0]["diagnostics"]
    assert back[0]["start"] == 12 and type(back[0]["start"]) is int
    for key in d0:
        left, right = d0[key], d1[key]
        assert type(left) is type(right), key
        if key == "nan":
            assert np.isnan(right)
        elif key == "arr":
            assert right.dtype == left.dtype and np.array_equal(left, right)
        else:
            assert left == right, key
    assert np.signbit(d1["neg_zero"])
    assert d1["f32"].dtype == np.float32
    with pytest.raises(TypeError, match="cannot cross between cards"):
        cards.encode_record({"s": {1, 2}})


def test_psplit_5_records_gather_in_rank_order_over_the_wire():
    def body(rank, transport):
        session = cards.CardSession(transport, 64, 4)
        return session.gather_records(
            [{"rank": rank, "v": np.float32(rank) / np.float32(3)}],
            name="probe")

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        got = out[rank]
        assert [entry[0]["rank"] for entry in got] == [0, 1]
        assert got[1][0]["v"] == np.float32(1) / np.float32(3)
        assert type(got[1 - rank][0]["v"]) is np.float32


def test_psplit_5_records_of_unequal_length_cross_an_equal_length_collective():
    """NCCL's all-gather moves one length from every rank; the ranks'
    records differ in length, so the gather must not hand the collective
    unequal payloads."""

    def body(rank, transport):
        post, collect = transport.post, transport.collect
        posted = {}

        def recording_post(tag, payload):
            posted[tag] = len(payload)
            return post(tag, payload)

        def equal_length_collect(tag, **kwargs):
            pieces = collect(tag, **kwargs)
            sizes = {len(p) for p in pieces if p is not None} | {posted[tag]}
            assert len(sizes) == 1, f"{tag}: payload lengths {sorted(sizes)}"
            return pieces

        transport.post = recording_post
        transport.collect = equal_length_collect
        session = cards.CardSession(transport, 64, 4)
        return session.gather_records(
            [{"rank": rank, "pad": "x" * (5 + 40 * rank)}], name="probe")

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        assert [entry[0]["pad"] for entry in out[rank]] == ["x" * 5, "x" * 45]


def test_psplit_6_a_floor_refusal_on_one_card_fires_on_every_card():
    """Each card folds only its own bands' tracer extrema, so a negative
    beyond roundoff in one card's band was seen by that card alone: it
    refused while its peer blocked in the next collective until the
    transport's timeout, which names a tag and not the physics breakage.
    The extrema are folded across the cards before the check, so every
    card refuses the same tracer at the same call."""
    from types import SimpleNamespace

    from arwen_global.dynamics import MoistHybridModel

    class Total:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float64)

        def total(self):
            return self.value

    def body(rank, transport):
        session = cards.CardSession(transport, 64, 4)
        stub = SimpleNamespace(
            transform=SimpleNamespace(backend=SimpleNamespace(
                xp=np, to_numpy=np.asarray, float_dtype=np.float64)),
            pipeline=SimpleNamespace(exchange=cards.RowExchange(session)),
            _card_world=lambda: 2,
        )
        # Rank 1's band holds the negative; rank 0's bands are clean.
        minima = {"qv": Total(-1.0e-3 if rank == 1 else 0.0)}
        maxima = {"qv": Total(1.0e-2)}
        try:
            MoistHybridModel._refuse_floored_tracers(stub, minima, maxima)
        except FloatingPointError as exc:
            return str(exc)
        return None

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        assert out[rank] is not None and "grid tracer qv" in out[rank]
