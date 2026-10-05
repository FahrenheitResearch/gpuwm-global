"""Gates CARD-1, BIT-6 and WIRE-1's arithmetic: the multi-card layer.

Three separable claims, and this file is where the two that need no GPU
are settled:

CARD-1  every module-level cache of a device object carries the device id
        in its key.  A process that opens two cards builds an entry on
        the first and hands it to a launch on the second, and CuPy's own
        error for that case ("The device where the array resides (0) is
        different from the current device (1)") only appears where it
        happens to check.  Neither node has two cards, so the device id
        is driven through a stub here and the real single-device arm runs
        on the cards; the stub is the arm that can distinguish a key that
        carries the id from one that does not, which is what the gate is
        about.

BIT-6   the band-to-card assignment is outside the arithmetic.  The band
        schedule is ``floor(k * nlat / B)`` whatever the card count, the
        reduction buffers are laid out by ``(nlat, shape)`` and filled in
        grid order, and a partial sum is added in ascending RANK order.
        The assertions here are on those three; the whole-model arm runs
        on the pair.

WIRE-1  the transport moves the bytes it says it moves and the ledger
        counts them.  The rate itself is a two-node measurement; the
        accounting is tested here over the loopback, where a wrong ledger
        would report a wrong rate on the real link too.

The transport is exercised for real over TCP in-process (two ranks, two
threads, loopback sockets), because a mocked transport tests the mock.
"""
from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from arwen_global.configs_dir import config_root as _shipped_configs
from arwen_global import cards
from arwen_global.bands import (
    BandPipeline,
    LatitudeAccumulator,
    PlaneAccumulator,
    band_edges,
)


# ---------------------------------------------------------------------
# CARD-1
# ---------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, ident):
        self.id = ident


class _FakeCuda:
    def __init__(self):
        self.current = 0

    def Device(self):  # noqa: N802 - CuPy's own spelling
        return _FakeDevice(self.current)


class _FakeCupy(types.ModuleType):
    """Just enough cupy for a cache key: a device id and a kernel factory."""

    def __init__(self):
        super().__init__("cupy")
        self.cuda = _FakeCuda()
        self.built = []

    def ElementwiseKernel(self, *args, **kwargs):  # noqa: N802
        made = ("kernel", len(self.built), self.cuda.current)
        self.built.append(made)
        return made


@pytest.fixture
def two_devices(monkeypatch):
    """A stub ``cupy`` whose current device the test moves.

    Neither node in this pair has two cards, so a process that holds a
    device-0 array and a device-1 array at once cannot be built here.
    What CAN be built, and what the defect actually is, is a cache key
    that does not distinguish the two: with this stub the caches are
    driven on device 0 and then on device 1 in one process, and a key
    missing the device id collapses the two entries into one.
    """
    fake = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return fake


def test_card1_the_backend_key_carries_the_device_and_numpy_has_none(two_devices):
    from arwen_global.spectral.backend import device_cache_key

    two_devices.cuda.current = 0
    assert device_cache_key(two_devices) == ("cupy", 0)
    two_devices.cuda.current = 1
    assert device_cache_key(two_devices) == ("cupy", 1)
    # NumPy has no device, so every CPU-backed cache keeps its old key.
    assert device_cache_key(np) == ("numpy",)


def test_card1_every_module_level_device_cache_builds_once_per_device(two_devices):
    """The gate proper: each cache, both devices, distinct entries.

    Six caches, named individually rather than swept, so a cache added
    later without a device id fails this file rather than a two-card run.
    """
    from arwen_global.physics import native_batch
    from arwen_global import transport as transport_module
    from arwen_global.spectral import fused

    for holder in (
        fused._CACHE, transport_module._KERNELS, native_batch._COLUMN_WATER_KERNEL,
    ):
        holder.clear()

    def build_all():
        return (
            fused.project_kernel(two_devices),
            transport_module._fused(two_devices, "periodic"),
            transport_module._fused(two_devices, "walled"),
            native_batch._column_water_kernel(two_devices),
        )

    two_devices.cuda.current = 0
    first = build_all()
    two_devices.cuda.current = 1
    second = build_all()
    two_devices.cuda.current = 0
    again = build_all()

    assert all(a != b for a, b in zip(first, second)), (
        "a cache handed device 1 the entry it built on device 0"
    )
    assert again == first, "device 0's entries were not reused"
    for holder, expected in (
        (fused._CACHE, 2), (transport_module._KERNELS, 4),
        (native_batch._COLUMN_WATER_KERNEL, 2),
    ):
        assert len(holder) == expected
        for key in holder:
            assert 0 in key or 1 in key, f"{key!r} carries no device id"


def test_card1_the_vertical_operator_serves_two_devices_from_one_build(two_devices):
    """The one cache deliberately NOT device-keyed, and why that is right.

    ``semi_implicit._OPERATORS`` holds an object of NumPy arrays; the three
    caches of DEVICE arrays it owns carry the device in their own keys.
    So one operator serves both cards and hands each its own matrices,
    and the NumPy eigen-decomposition behind it is not repeated per card.
    """
    from arwen_global.semi_implicit import (
        _OPERATORS,
        vertical_structure_operator,
    )
    from arwen_global.vertical import HybridCoordinate
    from arwen_global.spectral.backend import get_backend

    _OPERATORS.clear()
    vertical = HybridCoordinate.surface_stretched(40)
    one = vertical_structure_operator(vertical, 300.0, 1.0e5)
    two = vertical_structure_operator(vertical, 300.0, 1.0e5)
    assert one is two and len(_OPERATORS) == 1

    backend = get_backend("numpy", "float64")
    two_devices.cuda.current = 0
    one.device_matrices(backend)
    keys_after_numpy = set(one._device)
    assert keys_after_numpy == {("numpy", str(backend.float_dtype))}, (
        "the NumPy key must stay one element so a CPU run keeps its behaviour"
    )


# ---------------------------------------------------------------------
# The band assignment: BIT-6's arithmetic
# ---------------------------------------------------------------------


def test_the_band_schedule_does_not_move_with_the_card_count():
    """The schedule is a function of (nlat, bands) and of nothing else."""
    reference = band_edges(384, 8)
    for cards_count in (1, 2, 3, 4):
        weights = tuple(1.0 + i for i in range(cards_count))
        owners = cards.band_owners(384, 8, weights)
        assert len(owners) == 8
        assert band_edges(384, 8) == reference
        assert sorted(set(owners)) == list(range(cards_count))
        # contiguous runs: a rank's bands are consecutive
        for rank in set(owners):
            held = [k for k, r in enumerate(owners) if r == rank]
            assert held == list(range(held[0], held[-1] + 1))


def test_the_faster_card_is_given_more_rows():
    """Assignment by MEASURED throughput, and the reciprocal is the weight."""
    weights = cards.throughput_weights((357.4, 739.9))  # 5090, 5070 Ti
    assert weights[0] > weights[1]
    assert abs(sum(weights) - 1.0) < 1e-9
    owners = cards.band_owners(384, 16, weights)
    fast = sum(1 for r in owners if r == 0)
    assert fast > 16 - fast, "the 2.07x faster card took the smaller share"


def test_a_card_with_no_band_is_refused_by_name():
    with pytest.raises(ValueError, match="pays every exchange and computes"):
        cards.band_owners(384, 1, (1.0, 1.0))
    with pytest.raises(ValueError, match="pays every exchange and computes"):
        cards.band_owners(384, 4, (1.0, 1.0e-9))


def test_a_partial_sum_is_added_in_rank_order():
    """The one floating-point sum that crosses the wire, pinned to an order.

    Three values whose sum is order-dependent: 1 + 1e-16 + 1e-16 is 1.0
    left to right (each addend vanishes under the 1) and
    1.0000000000000002 right to left (the two small ones add first).
    Added in rank order the answer is the first one, every time.
    """
    parts = [np.float64(1.0), np.float64(1e-16), np.float64(1e-16)]
    assert cards.sum_in_rank_order(parts) == (parts[0] + parts[1]) + parts[2]
    assert cards.sum_in_rank_order(parts) != parts[0] + (parts[1] + parts[2])


def test_the_pipeline_hands_a_card_whole_bands_and_refuses_a_split_one():
    class _Exchange:
        world = 2

        def __init__(self, rows):
            self._rows = rows

        def owned_rows(self):
            return self._rows

    whole = BandPipeline(384, 8, exchange=_Exchange((0, 192)))
    assert [(s.start, s.stop) for s in whole.local_slices()] == [
        (0, 48), (48, 96), (96, 144), (144, 192)
    ]
    assert whole.local_rows() == (0, 192)
    assert len(whole.slices()) == 8, "the schedule itself must not move"
    with pytest.raises(ValueError, match="whole number of\n? *bands|whole bands"):
        BandPipeline(384, 8, exchange=_Exchange((0, 100)))


# ---------------------------------------------------------------------
# The transport, over real sockets
# ---------------------------------------------------------------------


def _two_rank_tcp(body, world: int = 2, attempts: int = 4):
    """Run ``body(rank, transport)`` on ``world`` threads over loopback.

    Retried on a connection failure: the helper picks a free port, closes
    it and reconnects, so another test in the same session can take the
    port in between.  That is the harness racing itself, not the
    transport, and a retry is the right answer to it.
    """
    last = None
    for _ in range(int(attempts)):
        try:
            return _two_rank_tcp_once(body, world)
        except (ConnectionError, OSError) as exc:
            last = exc
    raise last


def _two_rank_tcp_once(body, world: int = 2):
    import socket as _socket

    ports = []
    holders = []
    for _ in range(world - 1):
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        ports.append(sock.getsockname()[1])
        holders.append(sock)
    for sock in holders:
        sock.close()
    ports.append(0)
    addresses = [f"127.0.0.1:{p}" for p in ports]
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def run(rank):
        try:
            transport = cards.TcpCards(rank, world, addresses, chunk_bytes=1 << 16)
            try:
                results[rank] = body(rank, transport)
            finally:
                transport.close()
        except BaseException as exc:  # noqa: BLE001
            errors[rank] = exc

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120.0)
    if errors:
        raise errors[sorted(errors)[0]]
    return results


def test_the_tcp_transport_all_gathers_in_rank_order():
    def body(rank, transport):
        payload = bytes([rank]) * (3 + rank)
        return transport.all_gather("t", payload)

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        assert out[rank] == [b"\x00\x00\x00", b"\x01\x01\x01\x01"]


def test_wire1_the_ledger_counts_the_bytes_that_crossed():
    """WIRE-1's accounting: a wrong ledger reports a wrong rate."""
    size = 1 << 20

    def body(rank, transport):
        transport.all_gather("t", bytes(size))
        transport.ledger.mark_step()
        return transport.ledger.receipt()

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        row = out[rank]
        assert row["posted_bytes"] == size
        assert row["received_bytes"] == size
        assert row["steps"] == 1
        assert row["bytes_per_step"] == 2 * size
        assert row["achieved_bytes_s"] > 0.0
        assert row["wire_floor_gb_s"] == pytest.approx(2.0)


def test_wire1_the_ledger_is_read_after_the_sender_threads_catch_up():
    """The bytes a receipt reports are the bytes that crossed, every time.

    A post returns as soon as the frame is queued and the sender thread
    is what hands the bytes to the ledger, so the ledger trails the model
    by whatever is in flight -- which is the overlap working.  Reading it
    raw at that instant reports fewer bytes than crossed the wire:
    MEASURED, a one-megabyte all-gather read back 0 posted bytes on two
    of three attempts.  Every route a reader has must therefore drain
    first.  Repeated, because the defect this pins was intermittent.
    """
    size = 1 << 20

    def body(rank, transport):
        transport.all_gather("t", bytes(size))
        raw = transport.ledger.receipt()["posted_bytes"]
        transport.drain()
        return (raw, transport.ledger.receipt()["posted_bytes"])

    for _ in range(6):
        out = _two_rank_tcp(body)
        for rank in (0, 1):
            straight, after = out[rank]
            # The ledger drains itself, so the FIRST read is already the
            # wire's own count; the explicit drain changes nothing.
            assert straight == size
            assert after == size


def test_the_row_gather_assembles_the_buffer_one_card_would_have_built():
    """The seam every waist and every reduction buffer goes through."""
    nlat, width = 32, 5
    whole = np.arange(nlat * width, dtype=np.float64).reshape(nlat, width)

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 4, weights=(1.0, 1.0))
        first, last = session.local_rows()
        mine = np.full_like(whole, np.nan)
        mine[first:last] = whole[first:last]
        session.gather_rows(np, mine, 0, name="rows")
        return mine

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        assert np.array_equal(out[rank], whole), "a gathered buffer is not the globe"


def test_the_accumulators_reduce_the_globe_across_two_cards():
    """The reduction contract, over the wire: same number, both ranks, and
    the number a one-card run computes."""
    nlat, nlon = 24, 4
    plane = np.arange(nlat * nlon, dtype=np.float64).reshape(nlat, nlon)

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 4, weights=(1.0, 1.0))
        exchange = cards.RowExchange(session)
        pipeline = BandPipeline(nlat, 4, exchange=exchange)
        acc = PlaneAccumulator(np, (nlat, nlon), np.float64,
                               name="p", exchange=exchange)
        rows = LatitudeAccumulator(np, (nlat,), np.float64,
                                   name="l", exchange=exchange)
        for band in pipeline.local_slices():
            acc.add_band(band, plane[band])
            rows.add_band(band, plane[band].sum(axis=-1))
        return float(acc.total()), float(rows.total())

    out = _two_rank_tcp(body)
    reference = (float(plane.sum()), float(plane.sum(axis=-1).sum()))
    assert out[0] == out[1] == reference


def test_the_deep_halo_fills_exactly_the_rows_past_the_boundary():
    nlat, width = 40, 3
    whole = np.arange(nlat * 2, dtype=np.float64).reshape(nlat, 2)

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 4, weights=(1.0, 1.0))
        first, last = session.local_rows()
        mine = np.full_like(whole, np.nan)
        mine[first:last] = whole[first:last]
        session.exchange_halo(np, mine, 0, width, name="halo")
        return mine, (first, last)

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        mine, (first, last) = out[rank]
        lo = max(0, first - width)
        hi = min(nlat, last + width)
        assert np.array_equal(mine[lo:hi], whole[lo:hi]), "halo rows are not the globe's"
        # and nothing beyond the halo was fetched
        assert np.isnan(mine[hi:]).all() or hi == nlat
        assert np.isnan(mine[:lo]).all() or lo == 0


def test_mg5_the_halo_goes_to_the_two_neighbours_only():
    """MG-5: each edge is posted to the one rank that reads it.

    At four cards every rank used to post both of its edges to all three
    peers (the 2026-10-05 audit: 3.0x the neighbour-only need at world 4,
    7x at 8), so the halo bytes grew with the card count.  Now an interior
    rank posts exactly two edges and a polar rank one, at any world, and
    the window each rank fills is still the globe's own rows."""
    nlat, cols, width, world = 40, 2, 3, 4
    whole = np.arange(nlat * cols, dtype=np.float64).reshape(nlat, cols)
    edge = width * cols * 8

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 8, weights=(1.0,) * world)
        first, last = session.local_rows()
        mine = np.full_like(whole, np.nan)
        mine[first:last] = whole[first:last]
        session.exchange_halo(np, mine, 0, width, name="halo")
        transport.drain()
        posted = transport.ledger.receipt()["posted_by_exchange"]["halo"]["posted_bytes"]
        return mine, (first, last), posted

    out = _two_rank_tcp(body, world=world)
    for rank in range(world):
        mine, (first, last), posted = out[rank]
        neighbours = (rank > 0) + (rank < world - 1)
        assert posted == neighbours * edge, (rank, posted)
        lo, hi = max(0, first - width), min(nlat, last + width)
        assert np.array_equal(mine[lo:hi], whole[lo:hi])
        assert np.isnan(mine[:lo]).all() and np.isnan(mine[hi:]).all()


def test_mg5_a_halo_too_wide_for_any_card_is_refused_on_every_rank():
    """The width is checked against the narrowest card, so a neighbour that
    owns fewer rows than the halo refuses on EVERY rank instead of leaving
    the wider rank waiting on a peer that refused alone."""
    def body(rank, transport):
        # weights 3:1 over 40 rows: rank 1 owns 10 rows, rank 0 owns 30.
        session = cards.CardSession(transport, 40, 8, weights=(3.0, 1.0))
        with pytest.raises(ValueError, match="wider than the 10 rows card rank 1"):
            session.exchange_halo(np, np.zeros((40, 2)), 0, 12, name="halo")
        return True

    assert _two_rank_tcp(body) == {0: True, 1: True}


def test_a_halo_wider_than_a_card_is_refused_by_name():
    def body(rank, transport):
        session = cards.CardSession(transport, 40, 4, weights=(1.0, 1.0))
        with pytest.raises(ValueError, match="wider than the"):
            session.exchange_halo(np, np.zeros((40, 2)), 0, 40, name="halo")
        return True

    assert _two_rank_tcp(body) == {0: True, 1: True}


def test_a_missing_tag_names_the_breakage_rather_than_hanging():
    ready = threading.Event()

    def body(rank, transport):
        if rank == 0:
            with pytest.raises(TimeoutError, match="same tags in the same order"):
                transport.collect("never-posted", timeout_s=1.0)
            ready.set()
        else:
            # Rank 1 stays up: a peer that exits first would deliver an EOF
            # and the test would measure the dead-rank message instead.
            ready.wait(20.0)
        return True

    assert _two_rank_tcp(body)[0] is True


# ---------------------------------------------------------------------
# The launcher and the receipt
# ---------------------------------------------------------------------


def test_the_launcher_sets_nccl_ib_disable_and_the_receipt_records_it():
    env = cards.launch_environment("enp133s0f1np1")
    assert env["NCCL_IB_DISABLE"] == "1"
    assert env["NCCL_SOCKET_IFNAME"] == "enp133s0f1np1"


def test_a_single_card_session_is_the_two_card_one_with_a_world_of_one():
    session = cards.single_card_session(96, 4)
    assert session.world == 1
    assert session.local_rows() == (0, 96)
    assert session.gather_rows(np, np.zeros((96, 2)), 0).shape == (96, 2)
    row = session.receipt()
    assert row["cards"] == 1 and row["transport"] == "single"
    assert row["wire"]["posted_bytes"] == 0


def test_an_unrecognised_latitude_layout_is_refused_rather_than_written_stale():
    session = cards.single_card_session(8, 2)
    session.world = 2  # the refusal is the point, not the transport
    with pytest.raises(ValueError, match="does not recognise"):
        cards.gather_named_arrays(
            session, {"physics__odd": np.zeros((8, 3, 5))}, 8, 16)


# ---------------------------------------------------------------------
# WIRE-1 as a gate, not as a number in a receipt
# ---------------------------------------------------------------------


def _gate_fixture(tmp_path, **overrides):
    """The arguments ``run_gates`` needs, all of them comfortably inside
    their limits, so the only row that can move the verdict is the wire."""
    from arwen_global.config import load_config

    path = tmp_path / "wire-gate.toml"
    path.write_text("".join(line + chr(10) for line in (
        "[arwen_global]",
        'schema = "gpuwm.arwen-global-run/v1"',
        'name = "wire-gate"',
        'backend = "numpy"',
        'acknowledgement = "research-only-arwen-global-v1"',
        "[grid]", "truncation = 21",
        "[time]", "dt_s = 600.0", "duration_s = 600.0",
        "[vertical]", "nlev = 16", 'coordinate = "pressure_blend"',
    )), encoding="utf-8")
    cfg = load_config(path)
    payload = dict(
        cfg=cfg,
        transform_check={"roundtrip_relative_linf": 0.0,
                         "parseval_relative_error": 0.0},
        final_diag={"global_mean_surface_pressure_pa": 1.0,
                    "global_mean_total_water_kg_m2": 1.0},
        target_mass=1.0,
        target_water=1.0,
        trackers={"maximum_mass_fixer_log_offset": 0.0,
                  "maximum_global_water_fixer_kg_m2": 0.0,
                  "maximum_physics_water_repair_kg_m2": 0.0},
        # A config that says nothing about the core is a semi-Lagrangian run
        # (the default core, 2026-09-06), and a semi-Lagrangian run's receipt
        # carries that core's own gate rows; the runner supplies their
        # trackers and so does this fixture, because run_gates refuses to
        # judge such a run without them (a KeyError, not a silently missing
        # gate).
        supplementary={"maximum_positivity_fixer_relative": 0.0,
                       "maximum_semilag_lipschitz": 0.0,
                       "maximum_semilag_tracer_mass_fixer_relative": 0.0,
                       "maximum_semilag_tracer_mass_fixer_water_relative": 0.0,
                       "maximum_semilag_trajectory_move_cells": 0.0,
                       "minimum_semilag_fold_determinant": 1.0,
                       },
    )
    payload.update(overrides)
    return payload


def _wire_block(achieved_gb_s, world=2):
    return {"cards": world, "wire": {
        "achieved_gb_s": achieved_gb_s,
        "wire_floor_gb_s": cards.WIRE_FLOOR_BYTES_S / 1e9,
    }}


def test_wire1_fails_the_receipt_when_the_link_falls_below_the_floor(tmp_path):
    """The breakage: every interconnect figure this design was priced on
    was taken on IDLE cards, and MEASURED 2026-09-06 three of four
    rank-legs of the T255 two-card probe achieved 1.82 to 1.98 GB/s while
    both cards computed.  Without this row the receipt reports "pass" on
    a link the run did not get."""
    from arwen_global.runner import run_gates

    rows = run_gates(**_gate_fixture(tmp_path), cards=_wire_block(1.8174))
    gate = rows["two_card_wire_achieved_gb_s"]
    assert gate["direction"] == "floor"
    assert gate["value"] == pytest.approx(1.8174)
    assert gate["limit"] == pytest.approx(2.0)
    assert gate["passed"] is False
    assert not all(row["passed"] for row in rows.values())


def test_wire1_passes_above_the_floor_and_every_other_row_is_a_ceiling(tmp_path):
    from arwen_global.runner import run_gates

    rows = run_gates(**_gate_fixture(tmp_path), cards=_wire_block(2.3124))
    assert rows["two_card_wire_achieved_gb_s"]["passed"] is True
    assert all(row["passed"] for row in rows.values())
    # Two rows are floors: the wire's, and the semi-Lagrangian core's
    # trajectory fold determinant (the smallest det(I -+ (dt/2) J) met).
    floors = {n for n, r in rows.items() if r["direction"] == "floor"}
    assert floors == {"two_card_wire_achieved_gb_s",
                      "semilag_trajectory_fold_determinant"}
    ceilings = [n for n, r in rows.items() if r["direction"] == "ceiling"]
    assert len(ceilings) == len(rows) - 2


def test_a_single_card_run_has_no_wire_gate_to_fail(tmp_path):
    """A card that opened no socket moves no bytes, and a floor on zero
    would fail every single-card run in the tree."""
    from arwen_global.runner import run_gates

    rows = run_gates(**_gate_fixture(tmp_path), cards={"cards": 1, "transport": "single"})
    assert "two_card_wire_achieved_gb_s" not in rows
    assert all(row["passed"] for row in rows.values())


# ---------------------------------------------------------------------
# The order-m axis (design section 10, lane 7): the multi-card axis for
# the transform.  Orders are independent output indices of the Legendre
# contraction and are never reduced across the wire, so the split is
# bit-exact by construction.
# ---------------------------------------------------------------------


def test_order_band_owners_assigns_whole_bands_and_refuses_an_idle_rank():
    from arwen_global.spectral.legendre import band_bounds, DEFAULT_BAND

    bounds = band_bounds(255, DEFAULT_BAND)  # eight bands
    owners = cards.order_band_owners(bounds, (1.0, 1.0))
    assert len(owners) == len(bounds)
    # contiguous runs, both ranks used
    assert set(owners) == {0, 1}
    assert owners == tuple(sorted(owners)), "an order rank owns a contiguous run"
    # a truncation with fewer bands than cards is refused
    with pytest.raises(ValueError, match="cannot be shared"):
        cards.order_band_owners(band_bounds(20, DEFAULT_BAND), (1.0, 1.0))
    # an extreme weight that starves a rank is refused by name
    with pytest.raises(ValueError, match="no band"):
        cards.order_band_owners(bounds, (1.0, 1e-9))


def test_the_faster_card_is_given_more_orders():
    from arwen_global.spectral.legendre import band_bounds, DEFAULT_BAND

    bounds = band_bounds(255, DEFAULT_BAND)
    slow_first = cards.order_band_owners(bounds, (0.5, 1.0))
    # rank 1 is twice as fast, so it owns more of the (coefficient-weighted)
    # order bands than rank 0
    assert slow_first.count(1) >= slow_first.count(0)


def test_the_transform_refuses_a_partition_that_splits_a_band():
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    with pytest.raises(ValueError, match="whole Legendre bands"):
        SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=((0, 16),),  # half a 32-order band
        )


def test_the_transform_refuses_a_non_contiguous_partition():
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    with pytest.raises(ValueError, match="contiguous"):
        SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=((0, 32), (64, 86)),  # skips the middle band
        )


def test_a_partitioned_transform_holds_only_its_own_bands_tables():
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(255, backend="numpy", precision="float64")
    lower = SphericalHarmonicTransform.create(
        255, backend="numpy", precision="float64",
        order_partition=tuple(
            __import__("arwen_global.spectral.legendre", fromlist=["band_bounds"])
            .band_bounds(255, 32)[:4]),
    )
    # four of eight bands: the resident analysis+basis tables are a
    # fraction of the whole, which is the capacity the axis buys.
    assert lower._analysis.nbytes < whole._analysis.nbytes
    assert lower._basis.nbytes < whole._basis.nbytes


def _order_split_reference(truncation, seed):
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64")
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((2, whole.grid.nlat, whole.grid.nlon))
    coeff = whole.forward(field)
    grid = whole.inverse(coeff)
    return whole, field, coeff, grid


def test_the_order_split_transform_is_the_single_card_transform_over_the_wire():
    """BIT-5 on the order axis, in process: two ranks each hold half the
    Legendre table, each contracts its own orders, and the assembled
    analysis and synthesis are the single-card transform's bit for bit."""
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    whole, field, ref_coeff, ref_grid = _order_split_reference(85, 1)

    def body(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        exchange = cards.order_exchange_for(session, whole, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=exchange.owned_bounds())
        tr.order_exchange = exchange
        return tr.forward(field), tr.inverse(ref_coeff)

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        coeff, grid = out[rank]
        assert np.array_equal(coeff, ref_coeff), "order-split analysis is not the single-card spectrum"
        assert np.array_equal(grid, ref_grid), "order-split synthesis is not the single-card grid"


def test_the_order_axis_gathers_columns_and_never_sums_across_the_wire():
    """The assembled column set is a concatenation of disjoint order
    ranges: a placed piece, never an added one, so no floating-point sum
    crosses the wire and the schedule cannot move a bit."""
    whole, field, ref_coeff, _ = _order_split_reference(85, 2)
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    def body(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        exchange = cards.order_exchange_for(session, whole, weights=(1.0, 1.0))
        lo, hi = exchange.owned_order_range
        tr = SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=exchange.owned_bounds())
        # the compact contraction of the owned orders, before the gather
        waist = tr.fourier_waist(field)
        compact = tr._contract_orders(waist.take(), tr._analysis, lo, hi)
        return lo, hi, compact

    out = _two_rank_tcp(body)
    # each rank's compact columns equal exactly the reference columns for
    # its owned order range (m=0 realified on the whole afterwards)
    for rank in (0, 1):
        lo, hi, compact = out[rank]
        want = ref_coeff[..., lo:hi].copy()
        if lo == 0:
            # the m=0 realification is applied to the assembled whole, not
            # to the compact piece, so compare the piece pre-realification
            want[..., 0] = compact[..., 0]
        assert np.array_equal(compact, want)


# ---------------------------------------------------------------------
# The cross-card agreement refusal (lane 6's finding, made into a gate):
# a gather two-card run assembles a waist from rows computed on both
# cards, so it reproduces one card only where the cards agree.
# ---------------------------------------------------------------------


def test_the_agreement_check_refuses_a_pair_that_disagrees():
    # The run is refused by name at the contraction the cards disagree on,
    # with its direction and operand shape in the message.
    key = ("synthesis", "basis", (1, 24, 48), "complex64")
    with pytest.raises(ValueError, match="do not return the same bits"):
        cards._assert_card_hashes_agree(key, ["bb", "cc"])
    message = _raised_message(cards._assert_card_hashes_agree, key, ["bb", "cc"])
    assert "synthesis" in message and "(1, 24, 48)" in message


def test_the_agreement_check_passes_a_pair_that_agrees():
    cards._assert_card_hashes_agree(("analysis", "analysis", (8, 4, 8), "complex64"), ["aa", "aa"])  # no raise


def test_the_agreement_check_rides_the_runs_own_contractions_over_the_wire():
    """Two ranks contract the same inputs through a row exchange; the ledger
    records every distinct (direction, table, shape) once and the run
    passes.  A rank whose basis table differs by one unit of roundoff is
    refused by name at the first synthesis, before any waist is assembled
    from its rows.  A probe at fixed widths cannot do this: it refused a
    real pair at an eight-plane synthesis the step never presents."""
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    rng = np.random.default_rng(3)
    field = rng.standard_normal((3, whole.grid.nlat, whole.grid.nlon))
    coeff = whole.project(whole.forward(field))

    def body(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
        tr.row_exchange = cards.RowExchange(session)
        tr.forward(field)
        tr.inverse(coeff)
        tr.inverse(coeff)          # the same shape again: checked once
        tr.inverse(coeff[:1])      # a new shape: checked
        return session.agreement.receipt()

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        receipt = out[rank]
        assert receipt["agree"] is True
        directions = [row["direction"] for row in receipt["contractions"]]
        assert directions == ["analysis", "synthesis", "synthesis"]
        assert receipt["checked"] == 3
    assert out[0]["contractions"] == out[1]["contractions"]

    def body_disagree(rank, transport):
        session = cards.CardSession(transport, whole.grid.nlat, 4, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
        if rank == 1:
            # One band of the synthesis table moved by a few units of
            # roundoff: the kind of difference two cards' cuBLAS kernels
            # produce, applied to one rank only.
            m0, m1, block = tr._basis.blocks[0]
            tr._basis.blocks[0] = (m0, m1, block * (1.0 + 2.0 ** -40))
        tr.row_exchange = cards.RowExchange(session)
        try:
            tr.inverse(coeff)
        except ValueError as exc:
            return str(exc)
        return "no refusal"

    out = _two_rank_tcp(body_disagree)
    for rank in (0, 1):
        assert "do not return the same bits" in out[rank]
        assert "synthesis" in out[rank]


def _raised_message(fn, *args):
    try:
        fn(*args)
    except ValueError as exc:
        return str(exc)
    return ""


def test_a_model_run_on_the_order_axis_is_refused_by_name():
    import types
    from arwen_global.spectral.transform import SphericalHarmonicTransform
    from arwen_global.runner import open_card_session

    transform = SphericalHarmonicTransform.create(85, backend="numpy", precision="float64")
    cfg = types.SimpleNamespace(
        cards=2, card_axis="order", card_rank=0,
        card_addresses=("127.0.0.1:1", "127.0.0.1:2"),
        card_transport="tcp", card_weights=(), card_exchange="gather",
    )
    with pytest.raises(ValueError, match="card_axis='order' is not built for a model RUN"):
        open_card_session(cfg, transform, 4)


# -- the steps' own wall on every receipt ------------------------------------

def test_every_receipt_carries_the_steps_own_wall(tmp_path):
    """The capacity rows and the two-card ratio are defined on a STEP.

    The tier's receipt block carried a per-step wall only when something was
    parked; a resident run had no per-step figure at all and its step could
    only be read off the whole run's wall with the build, the cold start and
    the checkpoint writes inside it (the cards lane's T383 pair measurement,
    2026-09-07).  So every receipt carries the steps' own wall, pass or fail.
    """
    from dataclasses import replace
    from arwen_global.config import load_config
    from arwen_global.runner import run, _step_wall_receipt

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    cfg = replace(cfg, duration_s=cfg.dt_s * 3, output_interval_s=cfg.dt_s * 3)
    result = run(cfg, tmp_path / "run")
    wall = result["step_wall"]
    assert wall["steps_timed"] == 3
    assert wall["step_wall_seconds"] > 0.0
    assert wall["step_s"] == pytest.approx(wall["step_wall_seconds"] / 3, abs=1e-6)
    assert wall["step_wall_seconds"] <= result["wall_seconds"]
    # A run that died before its first step says so rather than dividing by it.
    assert _step_wall_receipt(0, 0.0)["step_s"] is None


def test_record_mode_carries_on_and_fails_the_gate_row_rather_than_refusing(tmp_path):
    """card_agreement='record': a disagreeing pair is not refused, the
    receipt lists the shape the cards disagreed on, and run_gates fails
    the two_card_contractions_agree row on it.  A timing device for unlike
    cards; the run's answer is neither card's and the receipt says so."""
    from arwen_global.spectral.transform import SphericalHarmonicTransform
    from arwen_global.runner import run_gates

    whole = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    rng = np.random.default_rng(5)
    field = rng.standard_normal((2, whole.grid.nlat, whole.grid.nlon))
    coeff = whole.project(whole.forward(field))

    def body(rank, transport):
        session = cards.CardSession(
            transport, whole.grid.nlat, 4, weights=(1.0, 1.0), agreement="record")
        tr = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
        if rank == 1:
            m0, m1, block = tr._basis.blocks[0]
            tr._basis.blocks[0] = (m0, m1, block * (1.0 + 2.0 ** -40))
        tr.row_exchange = cards.RowExchange(session)
        tr.forward(field)
        tr.inverse(coeff)
        return session.agreement.receipt()

    out = _two_rank_tcp(body)
    for rank in (0, 1):
        receipt = out[rank]
        assert receipt["mode"] == "record"
        assert receipt["agree"] is False
        assert [row["direction"] for row in receipt["contractions"]] == ["analysis"]
        assert [row["direction"] for row in receipt["disagreements"]] == ["synthesis"]
        assert receipt["disagreements"][0]["hashes"][0] != receipt["disagreements"][0]["hashes"][1]
    block = _wire_block(2.5)
    block["card_agreement"] = out[0]
    gates = run_gates(**_gate_fixture(tmp_path), cards=block)
    row = gates["two_card_contractions_agree"]
    assert row["passed"] is False and row["value"] == 1 and row["limit"] == 0
    block["card_agreement"] = {"agree": True, "disagreements": []}
    gates = run_gates(**_gate_fixture(tmp_path), cards=block)
    assert gates["two_card_contractions_agree"]["passed"] is True


def test_card_agreement_is_a_named_choice_outside_every_identity(tmp_path):
    from dataclasses import replace
    from arwen_global.config import load_config

    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    assert cfg.card_agreement == "refuse"
    recorded = replace(cfg, card_agreement="record")
    assert recorded.config_hash == cfg.config_hash
    text = open(str(_shipped_configs() / "arwen_global_moist_smoke.toml"), encoding="utf-8").read()
    assert "[memory]" not in text
    bad = tmp_path / "bad.toml"
    bad.write_text(text + chr(10) + '[memory]' + chr(10) + 'card_agreement = "ignore"' + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="card_agreement must be one of"):
        load_config(bad)


# -- transport selection (MG-3) ----------------------------------------------

def _free_addresses(world):
    import socket as _socket

    ports = []
    holders = []
    for _ in range(world - 1):
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        ports.append(sock.getsockname()[1])
        holders.append(sock)
    for sock in holders:
        sock.close()
    ports.append(0)
    return [f"127.0.0.1:{p}" for p in ports]


def _on_threads(world, run):
    results, errors = {}, {}

    def wrap(rank):
        try:
            results[rank] = run(rank)
        except BaseException as exc:  # noqa: BLE001
            errors[rank] = exc

    threads = [threading.Thread(target=wrap, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120.0)
    if errors:
        raise errors[sorted(errors)[0]]
    return results


def test_mg3_auto_runs_where_nccl_imports_instead_of_refusing(monkeypatch):
    """MG-3: ``auto`` used to raise NotImplementedError the moment
    ``cupy.cuda.nccl`` imported, which a rented CUDA image normally does, so
    every default multi-card run there died at the door.  ``auto`` now opens
    a transport that carries the exchange and the receipt names it."""
    monkeypatch.setattr(cards.NcclCards, "available", staticmethod(lambda: True))
    addresses = _free_addresses(2)

    def run(rank):
        transport = cards.open_transport(rank, 2, addresses, prefer="auto")
        try:
            pieces = transport.all_gather("t", bytes([rank + 1]) * 3)
            return pieces, transport.receipt()
        finally:
            transport.close()

    out = _on_threads(2, run)
    for rank in (0, 1):
        pieces, receipt = out[rank]
        assert pieces == [b"\x01" * 3, b"\x02" * 3]
        assert receipt["transport_requested"] == "auto"
        assert receipt["transport"] in ("tcp", "nccl")
        assert receipt["transport_reason"]


def test_mg3_an_explicit_nccl_that_cannot_be_honoured_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(cards.NcclCards, "available", staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match="cupy.cuda.nccl is not importable"):
        cards.open_transport(0, 2, ["127.0.0.1:1", "127.0.0.1:2"], prefer="nccl")


# -- one rank, one card (MG-4) -----------------------------------------------

_BOX8 = [f"127.0.0.1:{29500 + r}" for r in range(8)]


def test_mg4_eight_ranks_in_one_box_land_on_eight_cards():
    """MG-4: nothing chose the card, so eight ranks in one box all opened
    device 0.  Each rank now narrows CUDA_VISIBLE_DEVICES to its local rank."""
    chosen = []
    for rank in range(8):
        env = {}
        record = cards.place_rank(rank, 8, _BOX8, "cupy", env=env)
        assert record["placed"] and record["placed_by"] == "rank"
        assert env["CUDA_VISIBLE_DEVICES"] == str(rank)
        chosen.append(env["CUDA_VISIBLE_DEVICES"])
    assert len(set(chosen)) == 8


def test_mg4_two_hosts_count_local_ranks_per_host():
    addresses = ["10.0.0.1:1", "10.0.0.1:2", "10.0.0.2:1", "10.0.0.2:2"]
    assert [cards.local_rank_of(r, addresses) for r in range(4)] == [
        (0, 2), (1, 2), (0, 2), (1, 2)]
    env = {}
    cards.place_rank(3, 4, addresses, "cupy", env=env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_mg4_an_operator_choice_is_kept_and_a_short_list_is_refused():
    env = {"CUDA_VISIBLE_DEVICES": "5"}
    assert cards.place_rank(2, 8, _BOX8, "cupy", env=env)["placed_by"] == "operator"
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    cards.place_rank(1, 4, _BOX8[:4], "cupy", env=env)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    with pytest.raises(ValueError, match="names 2 cards"):
        cards.place_rank(2, 4, _BOX8[:4], "cupy", env=env)


def test_mg4_an_empty_visible_device_list_is_the_operator_hiding_every_card():
    """Set and empty hides every card; placement used to read it as unset
    and write the local rank, turning a deliberate "no card" into a card."""
    env = {"CUDA_VISIBLE_DEVICES": ""}
    record = cards.place_rank(1, 2, _BOX8[:2], "cupy", env=env)
    assert record["placed"] is False and "hidden" in record["reason"]
    assert env == {"CUDA_VISIBLE_DEVICES": ""}


def test_mg4_a_cpu_run_and_a_one_card_run_are_untouched():
    env = {}
    assert cards.place_rank(1, 2, _BOX8[:2], "numpy", env=env)["placed"] is False
    assert cards.place_rank(0, 1, _BOX8[:1], "cupy", env=env)["placed"] is False
    assert env == {}


def test_mg4_two_ranks_on_one_card_are_refused_by_uuid():
    def ident(rank, uuid):
        return {"device_index": 0, "name": "card",
                "uuid": uuid, "pci_bus_id": f"0000:0{rank}:00.0",
                "cuda_visible_devices": None}

    def same(rank, transport):
        with pytest.raises(ValueError, match="card ranks 0 and 1 both run on"):
            cards.check_card_placement(transport, ident(rank, "aa"))
        return True

    assert _two_rank_tcp(same) == {0: True, 1: True}

    def distinct(rank, transport):
        return cards.check_card_placement(transport, ident(rank, f"u{rank}"))

    out = _two_rank_tcp(distinct)
    assert [row["uuid"] for row in out[0]] == ["u0", "u1"]

    def cpu(rank, transport):
        return cards.check_card_placement(transport, None)

    assert _two_rank_tcp(cpu)[1] == [None, None]


def test_mg4_the_door_places_the_rank_before_the_sizer_reads_the_card(monkeypatch):
    """The gate's probe subprocess inherits the environment, so the rank's
    card has to be chosen before the gate runs, or every rank in a box is
    sized against device 0's free memory."""
    import argparse
    import os as _os
    from dataclasses import replace
    from arwen_global import cli, sizing
    from arwen_global.config import load_config

    seen = {}

    class Stop(Exception):
        pass

    def fake_gate(cfg, **_kw):
        seen["visible"] = _os.environ.get("CUDA_VISIBLE_DEVICES")
        raise Stop()

    import sys as _sys

    monkeypatch.setattr(sizing, "run_memory_gate", fake_gate)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # No device is touched: the placement is the environment alone.
    monkeypatch.delitem(_sys.modules, "cupy", raising=False)
    monkeypatch.setattr(cards, "_host_card_count", lambda: 8)
    cards._PLACED.clear()
    cfg = load_config(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
    cfg = replace(cfg, backend="cupy", cards=4, card_rank=3,
                  card_addresses=tuple(_BOX8[:4]))
    args = argparse.Namespace(config="x.toml")
    try:
        with pytest.raises(Stop):
            cli._size_the_run(args, cfg, "gpuwm global run")
        assert seen["visible"] == "3"
    finally:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        cards._PLACED.clear()


# -- the in-box device transport (MG-2) --------------------------------------
#
# A stand-in for cupy.cuda.nccl that moves REAL bytes between threads with
# memmove on the arrays' pointers, so the transport's grouped broadcasts and
# neighbour send/receive are exercised end to end on the CPU: the pieces,
# their order and their sizes are what a real communicator is handed.

class _FakeNccl:
    NCCL_UINT8 = 1

    def __init__(self, world, fail_rank=None, uid_bytes=False):
        self.world = world
        self.fail_rank = fail_rank
        # CuPy 14 returns the unique id as bytes and wants bytes back;
        # older CuPy used a tuple of ints.  Both must survive the mesh.
        self.uid = bytes(range(128)) if uid_bytes else tuple(range(-64, 64))
        self.barrier = threading.Barrier(world)
        self.groups = {}
        self.tls = threading.local()
        self.calls = {r: [] for r in range(world)}

    def get_unique_id(self):
        return self.uid

    def get_version(self):
        return 22800

    def NcclCommunicator(self, ndev, uid, rank):
        assert ndev == self.world and type(uid) is type(self.uid) and uid == self.uid
        if rank == self.fail_rank:
            raise RuntimeError("fake: ncclInternalError")
        self.tls.rank = rank
        return _FakeComm(self, rank)

    def groupStart(self):
        self.tls.ops = []

    def groupEnd(self):
        import ctypes

        rank = self.tls.rank
        ops = list(self.tls.ops)
        self.calls[rank].append([op[0] for op in ops])
        self.groups[rank] = ops
        self.barrier.wait()
        mine_b = [op for op in ops if op[0] == "bcast"]
        for k, (_, root, ptr, n) in enumerate(mine_b):
            if root == rank:
                continue
            theirs = [op for op in self.groups[root] if op[0] == "bcast"][k]
            assert theirs[1] == root and theirs[3] == n
            ctypes.memmove(ptr, theirs[2], n)
        for _, peer, ptr, n in (op for op in ops if op[0] == "recv"):
            sent = [op for op in self.groups[peer] if op[0] == "send" and op[1] == rank]
            assert len(sent) == 1 and sent[0][3] == n
            ctypes.memmove(ptr, sent[0][2], n)
        self.barrier.wait()


class _FakeComm:
    def __init__(self, fake, rank):
        self.fake, self.rank = fake, rank

    def broadcast(self, send, recv, n, dtype, root, stream):
        assert dtype == 1 and send == recv
        self.fake.tls.ops.append(("bcast", root, recv, n))

    def send(self, ptr, n, dtype, peer, stream):
        self.fake.tls.ops.append(("send", peer, ptr, n))

    def recv(self, ptr, n, dtype, peer, stream):
        self.fake.tls.ops.append(("recv", peer, ptr, n))

    def destroy(self):
        pass


def _on_fake_nccl(world, body, *, fake=None, device=lambda r: True, prefer="auto"):
    fake = fake or _FakeNccl(world)
    addresses = _free_addresses(world)

    def run(rank):
        transport = cards.open_transport(
            rank, world, addresses, prefer=prefer, device=device(rank),
            nccl=fake, device_xp=np)
        try:
            return body(rank, transport)
        finally:
            transport.close()

    return _on_threads(world, run), fake


def test_mg2_device_exchanges_ride_nccl_and_assemble_the_one_card_bytes():
    """MG-2: the row gather, the partial sum, the order columns and the halo
    take the device communicator, and every rank assembles exactly the
    bytes a one-card run holds.  Host payloads still cross the TCP mesh."""
    nlat, cols, world = 36, 3, 3
    rng = np.random.default_rng(11)
    whole = rng.standard_normal((nlat, cols))
    partials = [rng.standard_normal((4, 5)) for _ in range(world)]

    def body(rank, transport):
        session = cards.CardSession(transport, nlat, 6, weights=(1.0,) * world)
        first, last = session.local_rows()
        rows = np.full_like(whole, np.nan)
        rows[first:last] = whole[first:last]
        session.gather_rows(np, rows, 0, name="rows")
        halo = np.full_like(whole, np.nan)
        halo[first:last] = whole[first:last]
        session.exchange_halo(np, halo, 0, 2, name="halo")
        total = session.gather_partials(np, partials[rank], name="partial")
        host = transport.all_gather("host", bytes([rank]))
        return transport.name, rows, halo, (first, last), total, host, transport.receipt()

    out, fake = _on_fake_nccl(world, body, fake=_FakeNccl(world, uid_bytes=True))
    reference_total = cards.sum_in_rank_order(partials)
    for rank in range(world):
        name, rows, halo, (first, last), total, host, receipt = out[rank]
        assert name == "nccl"
        assert np.array_equal(rows, whole)
        lo, hi = max(0, first - 2), min(nlat, last + 2)
        assert np.array_equal(halo[lo:hi], whole[lo:hi])
        assert np.isnan(halo[:lo]).all() and np.isnan(halo[hi:]).all()
        assert total.tobytes() == reference_total.tobytes()
        assert host == [b"\x00", b"\x01", b"\x02"]
        assert receipt["transport"] == "nccl" and receipt["host_channel"] == "tcp"
        assert receipt["device_exchanges"] == 3
        # The first groups are the link probe at session open (a broadcast
        # from every root per repeat), kept out of the exchange count.
        probes = cards.NcclCards.LINK_PROBE_REPEATS
        assert fake.calls[rank][:probes] == [["bcast"] * world] * probes
        assert receipt["link_probe_gb_s"] is not None
        # gather and partial are broadcasts from every root; the halo is the
        # neighbours only: one send and one receive per neighbour
        assert fake.calls[rank][probes] == ["bcast"] * world
        neighbours = (rank > 0) + (rank < world - 1)
        assert sorted(fake.calls[rank][probes + 1]) == sorted(
            ["send", "recv"] * neighbours)


def test_mg2_the_transform_over_nccl_is_the_transform_over_tcp_bit_for_bit():
    """BIT-5 across transports: the same three-rank gather analysis returns
    one card's bits over the TCP mesh and over the device communicator,
    and so does the order-split analysis and synthesis."""
    from arwen_global.spectral.transform import SphericalHarmonicTransform

    whole = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
    rng = np.random.default_rng(3)
    field = rng.standard_normal((3, whole.grid.nlat, whole.grid.nlon))
    one_card = whole.forward(field)
    world = 3

    def rows(rank, transport):
        session = cards.CardSession(
            transport, whole.grid.nlat, 6, weights=(1.0,) * world)
        tr = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
        tr.row_exchange = cards.RowExchange(session)
        return tr.forward(field)

    over_tcp = _two_rank_tcp(rows, world=world)
    over_nccl, _ = _on_fake_nccl(world, rows)
    for rank in range(world):
        assert over_tcp[rank].tobytes() == one_card.tobytes()
        assert over_nccl[rank].tobytes() == one_card.tobytes()

    owhole, ofield, ref_coeff, ref_grid = _order_split_reference(85, 1)

    def orders(rank, transport):
        session = cards.CardSession(transport, owhole.grid.nlat, 4, weights=(1.0, 1.0))
        exchange = cards.order_exchange_for(session, owhole, weights=(1.0, 1.0))
        tr = SphericalHarmonicTransform.create(
            85, backend="numpy", precision="float64",
            order_partition=exchange.owned_bounds())
        tr.order_exchange = exchange
        return tr.forward(ofield), tr.inverse(ref_coeff)

    out, _ = _on_fake_nccl(2, orders)
    for rank in (0, 1):
        coeff, grid = out[rank]
        assert coeff.tobytes() == ref_coeff.tobytes()
        assert grid.tobytes() == ref_grid.tobytes()


def test_mg2_auto_falls_back_to_the_mesh_when_one_rank_cannot_open_nccl():
    def body(rank, transport):
        return transport.name, transport.receipt()["transport_reason"], \
            transport.all_gather("x", b"y")

    out, fake = _on_fake_nccl(3, body, device=lambda r: r != 1)
    for rank in range(3):
        name, reason, pieces = out[rank]
        assert name == "tcp" and "card ranks [1] cannot open NCCL" in reason
        assert pieces == [b"y"] * 3
    out, _ = _on_fake_nccl(2, body, fake=_FakeNccl(2, fail_rank=0))
    for rank in range(2):
        assert out[rank][0] == "tcp"
        assert "did not build on card ranks [0]" in out[rank][1]


def test_mg2_an_explicit_nccl_refuses_on_every_rank_when_one_cannot():
    def body(rank, transport):
        return transport.name

    with pytest.raises(RuntimeError, match="card_transport='nccl' was asked for"):
        _on_fake_nccl(2, body, device=lambda r: r == 0, prefer="nccl")


def test_mg4_more_ranks_than_the_host_has_cards_is_refused_by_name(monkeypatch):
    """Rank 1 of two on a one-card host used to be pointed at a device that
    does not exist (or, before placement, at the one card rank 0 holds)."""
    import sys as _sys

    monkeypatch.delitem(_sys.modules, "cupy", raising=False)
    monkeypatch.setattr(cards, "_host_card_count", lambda: 1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cards._PLACED.clear()
    try:
        with pytest.raises(ValueError, match="2 card ranks share host 'loopback' and it has 1 card:"):
            cards.place_rank(1, 2, _BOX8[:2], "cupy")
    finally:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        cards._PLACED.clear()


# -- the device path keeps the mesh's two safety properties ------------------


class _StuckEvent:
    done = False


class _AbortableComm:
    def __init__(self):
        self.aborted = False

    def abort(self):
        self.aborted = True

    def destroy(self):
        pass


class _MeshStub:
    rank, world = 0, 2

    def __init__(self):
        self.ledger = cards.WireLedger()

    def drain(self, timeout_s=None):
        pass

    def close(self):
        pass


def test_mg2_a_device_exchange_that_never_finishes_aborts_and_names_its_tag():
    """R17 on NCCL: a peer that died leaves a collective unfinished, which
    no NCCL call times out on.  The watchdog aborts the communicator past
    the timeout and the next exchange refuses naming the tag, instead of
    every survivor hanging in its next stream synchronise."""
    import time as _time

    comm = _AbortableComm()
    transport = cards.NcclCards(_MeshStub(), comm, _FakeNccl(2),
                                device_xp=np, timeout_s=0.2, probe=False)
    try:
        with transport._watch_lock:
            transport._watching.append(
                (_StuckEvent(), "semilag_tracer_qc#7", _time.monotonic()))
        deadline = _time.monotonic() + 10.0
        while transport._aborted is None and _time.monotonic() < deadline:
            _time.sleep(0.05)
        assert comm.aborted
        with pytest.raises(TimeoutError, match="semilag_tracer_qc#7"):
            transport.drain()
        with pytest.raises(TimeoutError, match="R17"):
            transport._group("next", [], 0, 0)
    finally:
        transport.close()


def test_mg2_ranks_running_different_device_exchanges_are_refused_by_name(
        monkeypatch):
    """Equal byte counts under different tags would swap bytes silently on
    NCCL; the running tag digest catches it at the next check."""
    monkeypatch.setattr(cards.NcclCards, "TAG_CHECK_EVERY", 1)
    payload = np.arange(6.0)

    def body(rank, transport):
        name = "waist_rows" if rank == 0 else "tracer_rows"
        try:
            transport.all_gather_arrays(name, np, payload, [payload.shape] * 2,
                                        payload.dtype)
        except RuntimeError as refusal:
            return str(refusal)
        return None

    out, _fake = _on_fake_nccl(2, body)
    for rank in range(2):
        assert out[rank] is not None and "different device exchanges" in out[rank]


def test_wire1_reads_the_device_link_probe_on_nccl_and_the_run_rate_on_tcp():
    from arwen_global.runner import wire_gate_row

    wire = {"achieved_gb_s": 0.03, "wire_floor_gb_s": 2.4}
    nccl = wire_gate_row({"cards": 2, "transport": "nccl",
                          "link_probe_gb_s": 21.5, "wire": wire})
    assert nccl["value"] == 21.5 and nccl["passed"]
    assert "probe" in nccl["measured"]
    tcp = wire_gate_row({"cards": 2, "transport": "tcp", "wire": wire})
    assert tcp["value"] == 0.03 and not tcp["passed"]
    assert wire_gate_row({"cards": 1, "wire": wire}) is None
