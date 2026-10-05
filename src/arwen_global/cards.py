"""Two cards, one answer: the transport, the band assignment, the ledger.

A second card is a SPEED device in this design, not a capacity device.
Capacity comes from latitude bands and, when the card is short, from the
pinned host tier; both of those live in :mod:`arwen_global.bands`
and neither needs a wire.  What a second card buys is that the bands are
executed by two processors instead of one, and what it costs is that the
Legendre contraction wants every latitude row on the card that runs it.
This module is that cost, made explicit and measured.

**One mechanism, two modes.**  Grid space is partitioned: every latitude
band belongs to exactly one rank for the life of the run, and that rank
computes it.  Spectral space is replicated: every rank holds the whole
spectral state, so the semi-implicit solve, the diffusion and every
per-coefficient operator run whole on every rank and never touch the
wire.  The two meet at the Fourier waist, and the meeting is where the
mode is chosen:

``gather`` (the default, bit-identical)
    Each rank fills the waist rows of its own bands, the ranks all-gather
    the ROWS, and every rank then runs the whole-width contraction on a
    complete waist.  The contraction sees ``K = N = nlat`` exactly as a
    one-card run does, so the coefficients are the one-card coefficients
    bit for bit and no pin moves.  It ships the analysis direction only:
    the synthesis starts from a replicated spectral state, so each rank
    contracts to its own rows without a gather.

``partial`` (approximate, carries its own pin)
    Each rank contracts its own rows, producing a PARTIAL Legendre sum,
    and the ranks all-gather those partial coefficient sets and add them
    locally in ascending rank order.  The contraction's ``K`` is the
    rank's row count rather than ``nlat``, which is a change of
    arithmetic and not of layout -- ``BandedLegendreTable`` carries the
    measurement that forces the point -- so this mode writes its own pin
    and a checkpoint written under it refuses to resume under the other.
    It moves a third of the bytes ``gather`` moves.

**All-gather, never all-reduce.**  A collective reduction picks its own
tree, and the tree moves with the rank count and the message size, so the
sum it computes is a function of the schedule.  Every exchange here is a
transport that moves bytes and nothing else; the addition happens
afterwards, locally, in ascending rank order, on every rank
(:func:`sum_in_rank_order`).  It costs nothing to insist on this:
MEASURED 2026-09-06, all-gather 3.087 GB/s against all-reduce 3.086 at
512 MB on this pair.

**The band assignment is outside the arithmetic, and gate BIT-6 says so.**
Band edges are ``floor(k * nlat / B)``, a pure function of ``(nlat, B)``
and never of the card count (:func:`arwen_global.bands.band_edges`).
Which rank runs which band changes who computes a row, not what the row
is, and the reduction contract in :mod:`~arwen_global.bands` folds
every global reduction over a resident buffer in grid order rather than
in completion order.  So the assignment is free to follow the measured
speed of each card, and BIT-6 runs three different assignments at one
band count and demands one answer.

**What the wire costs on the pair we own.**  MEASURED 2026-09-06 by the
interconnect lane: 3.09 GB/s on the 25 GbE between the nodes, with
``t = 0.30 ms + bytes / 3.09 GB/s`` fitting NCCL over sockets, NCCL over
RoCE, MPI over TCP and a hand-staged pipeline to within 1 percent of each
other.  8 MB is the operating point: at 1 MB the latency floor costs 12
percent of the rate and at 32 MB a single unpipelined chunk exposes 10 ms.
PCIe to the host is 23 times cheaper per byte than that wire
(H2D 55.6 / 56.8, D2H 42.3 / 42.6 GB/s MEASURED), which is why spilling
to the host is the capacity mechanism and the second card is not.

**Two transports, one rendezvous.**  Every multi-card run opens
:class:`TcpCards` first: the full mesh of sockets is the rendezvous, and
it carries every small host payload (the band profile, the agreement
digests, the associative folds, the closing barrier).  When every rank
runs on a CUDA device and ``cupy.cuda.nccl`` imports on every rank, the
ranks vote over that mesh, rank 0 broadcasts an NCCL unique id over it,
and :class:`NcclCards` carries the DEVICE exchanges (waist rows, halo
edges, partial sums, order columns) as device pointers on the compute
stream: no device-to-host copy, no ``tobytes``, no socket and no GIL in
the data path.  In one box that is NVLink or PCIe peer copies instead of
loopback TCP through host memory, which the 2026-10-05 audit MEASURED at
0.28 to 3.1 GB/s per rank.  NCCL still moves bytes only: the all-gather
is a grouped broadcast from each rank of its exact payload, the halo is a
grouped send/receive with the two neighbours, and every sum is taken
afterwards, locally, in ascending rank order, so the transport cannot
move a bit (BIT-5, BIT-6).  Where NCCL is absent on any rank, or the
backend is NumPy, the device exchanges stage through the host over TCP
exactly as before, and the receipt names which transport ran and why.
``NCCL_IB_DISABLE=1`` is set by the launcher and recorded in the receipt
so a socket run between nodes is never read as a fabric run.

**One rank, one card.**  A box with several cards runs one rank per card:
:func:`place_rank` narrows ``CUDA_VISIBLE_DEVICES`` to the rank's own card
(its LOCAL rank among the ranks sharing its host) before any CUDA context
exists, so the sizer's probe subprocess and the run read the same card,
and the transport refuses by name a run whose ranks report the same
device UUID.
"""
from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time

import numpy as np

from .bands import band_edges

#: The exchange chunk.  MEASURED 2026-09-06 on the 25 GbE pair: the fitted
#: curve ``t = 0.30 ms + bytes / 3.09 GB/s`` reaches 2.653 GB/s at 4 MB,
#: 2.927 at 16 MB and 3.086 at 512 MB, so 8 MB buys 96 percent of the
#: asymptotic rate at a chunk small enough that one unpipelined chunk
#: costs 2.6 ms rather than the 166 ms a 512 MB one would.
CHUNK_BYTES = 8 * 2**20

#: Gate WIRE-1's floor.  A two-card run whose achieved rate falls below
#: this is reporting a link that is not the link the design was priced on
#: -- every interconnect number was taken on idle cards, and R10 is the
#: risk that the rate falls while both cards compute.  Two thirds of the
#: MEASURED idle rate.
WIRE_FLOOR_BYTES_S = 2.0e9

#: The frame header: tag length, then payload length.
_HEADER = struct.Struct("!HQ")

#: How long a collect waits for a peer before it names the breakage.  A
#: two-card run has no fault tolerance (R17, accepted): a dead rank must
#: fail loudly and soon rather than hang until the harness reaps it.
DEFAULT_TIMEOUT_S = 900.0


# ---------------------------------------------------------------------
# The band assignment
# ---------------------------------------------------------------------


def band_owners(nlat: int, bands: int, weights) -> tuple[int, ...]:
    """Which rank runs each band, by measured per-card throughput.

    ``weights`` is one positive number per rank, larger meaning faster;
    the per-band cost profile taken at run start supplies them
    (:func:`throughput_weights`).  Each rank receives a CONTIGUOUS run of
    bands whose row count is as close to its share of the rows as whole
    bands allow, which keeps the one halo a meridional sweep needs to one
    boundary per rank pair rather than one per band.

    A rank that would receive no band is REFUSED rather than tolerated:
    an idle rank still takes part in every exchange, so it pays the wire
    and returns nothing, and the run would be slower than the one card it
    is meant to accelerate.  The caller is told to use fewer cards or
    more bands, which are the two things that fix it.
    """
    b = int(bands)
    p = len(weights)
    w = [float(v) for v in weights]
    if p < 1:
        raise ValueError("band_owners needs at least one rank weight")
    if any(not np.isfinite(v) or v <= 0.0 for v in w):
        raise ValueError(f"every card weight must be finite and positive, got {weights!r}")
    if b < p:
        raise ValueError(
            f"{b} latitude bands cannot be shared between {p} cards: a card "
            "with no band pays every exchange and computes nothing.  Raise "
            "latitude_bands to at least the card count, or run one card"
        )
    edges = band_edges(int(nlat), b)
    rows = [edges[k + 1] - edges[k] for k in range(b)]
    total = float(sum(rows))
    share = [v / sum(w) for v in w]
    # Target row boundaries in row space, then each band goes to the rank
    # whose interval holds its midpoint.  A pure function of (edges, w).
    bounds = []
    running = 0.0
    for r in range(p - 1):
        running += share[r] * total
        bounds.append(running)
    owners = []
    start = 0
    for k in range(b):
        middle = start + rows[k] / 2.0
        rank = 0
        while rank < p - 1 and middle > bounds[rank]:
            rank += 1
        owners.append(rank)
        start += rows[k]
    # The midpoint rule is monotone in k, so the runs are contiguous; an
    # extreme weight can still starve a rank, and that is the refusal above.
    held = {r: owners.count(r) for r in range(p)}
    empty = [r for r in range(p) if held[r] == 0]
    if empty:
        raise ValueError(
            f"card weights {weights!r} over {b} bands leave rank(s) {empty} "
            "with no band: a card with no band pays every exchange and "
            "computes nothing.  Raise latitude_bands, or run one card"
        )
    return tuple(owners)


def throughput_weights(band_ms) -> tuple[float, ...]:
    """Per-rank weights from the per-band device-millisecond profile.

    ``band_ms`` is one measured device time per rank for the SAME
    reference band, taken once at run start on each card.  Throughput is
    the reciprocal, so the faster card is given proportionally more rows.
    The receipt records the raw milliseconds beside the weights, because a
    ratio with no numerator has no tense.
    """
    values = [float(v) for v in band_ms]
    if not values or any(not np.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError(f"every per-band time must be finite and positive, got {band_ms!r}")
    inverse = [1.0 / v for v in values]
    scale = sum(inverse)
    # Quantised so a receipt and a second run of the same pair record the
    # same assignment; the sixth decimal is far below any real spread.
    return tuple(round(v / scale, 6) for v in inverse)


def order_band_owners(order_bounds, weights) -> tuple[int, ...]:
    """Which rank owns each Legendre ORDER band, by measured throughput.

    This is the order-m axis's assignment, dual to :func:`band_owners`.
    ``order_bounds`` is the ``[(m0, m1), ...]`` band schedule
    (:func:`arwen_global.spectral.legendre.band_bounds`), and every band is
    given WHOLE to exactly one rank: the Legendre contraction runs one
    strided-batched GEMM per band whose batch count is that band's order
    count, and cuBLAS picks its kernel by the batch, so splitting a band
    across cards would change the arithmetic where assigning whole bands
    does not (:class:`arwen_global.spectral.transform.BandedLegendreTable`
    carries the measurement).  So the order axis's grain is the band, and
    an order-split run holds only its own bands' tables -- which is the
    capacity the axis buys past the Legendre table wall.

    Each rank receives a CONTIGUOUS run of bands.  The cost of a band is
    the number of retained coefficients in it (the triangle is heavier at
    low order), so the split balances summed coefficients rather than band
    count.  A rank left with no band is REFUSED by name, exactly as the
    latitude axis refuses one: an idle rank pays every exchange and returns
    nothing.
    """
    bounds = [(int(a), int(b)) for a, b in order_bounds]
    if not bounds:
        raise ValueError("order_band_owners needs at least one order band")
    t1 = bounds[-1][1]
    w = [float(v) for v in weights]
    p = len(w)
    if p < 1:
        raise ValueError("order_band_owners needs at least one rank weight")
    if any(not np.isfinite(v) or v <= 0.0 for v in w):
        raise ValueError(f"every card weight must be finite and positive, got {weights!r}")
    if len(bounds) < p:
        raise ValueError(
            f"{len(bounds)} order bands cannot be shared between {p} cards on "
            "the order axis: a card with no band holds no table and computes "
            "nothing.  Raise the truncation, lower legendre_band, or run one "
            "card"
        )
    # Retained coefficients per band: order m carries degrees m..T, so a
    # band [m0, m1) carries sum_{m=m0}^{m1-1} (T + 1 - m).  A pure function
    # of the schedule, so a receipt and a rerun read the same owners.
    cost = [sum(t1 - m for m in range(m0, m1)) for m0, m1 in bounds]
    total = float(sum(cost))
    share = [v / sum(w) for v in w]
    bounds_row = []
    running = 0.0
    for r in range(p - 1):
        running += share[r] * total
        bounds_row.append(running)
    owners = []
    acc = 0.0
    for c in cost:
        middle = acc + c / 2.0
        rank = 0
        while rank < p - 1 and middle > bounds_row[rank]:
            rank += 1
        owners.append(rank)
        acc += c
    held = {r: owners.count(r) for r in range(p)}
    empty = [r for r in range(p) if held[r] == 0]
    if empty:
        raise ValueError(
            f"card weights {weights!r} over {len(bounds)} order bands leave "
            f"rank(s) {empty} with no band: a card with no order band holds no "
            "table and computes nothing.  Raise the truncation, lower "
            "legendre_band, or run one card"
        )
    return tuple(owners)


def measure_band_cost_ms(transform, rows: int, *, repeats: int = 7) -> float:
    """ How long ONE BAND of grid-space work costs on THIS card, in ms.

    The per-band cost profile the design asks for, taken once at run start
    and all-gathered so every rank assigns bands by the same numbers.  It
    is a measurement and not a table: the ratio between two cards is the
    quantity four capacity rows and every two-card figure move with (R5),
    and a table entry has no tense.

    What it times is the shape of the work a band actually does: a
    row-local FFT along longitude over a band-high stack, and a multiply
    over the same stack.  Not the whole step -- the whole step needs the
    model, and the assignment has to exist before the model is built --
    but the same operand shape and the same memory traffic, which is what
    the ratio depends on.  The BEST of ``repeats`` is taken, because a
    slow sample is another process on the card and a fast one cannot be.
    """
    xp = transform.backend.xp
    dtype = transform.backend.float_dtype
    nlon = int(transform.grid.nlon)
    # The stack is sized to a fixed 64 MiB of operand rather than to a
    # fixed level count.  MEASURED 2026-09-06 and CORRECTED: at a
    # sixteen-plane stack (2.4 MB at T255, 48 rows) the probe read
    # 0.3096 ms on the 5090 against 0.3025 on the 5070 Ti -- a two
    # percent inversion, because a 2.4 MB transform is bound by kernel
    # launch and not by the card.  A probe that cannot tell two cards
    # apart is a flawed instrument, and a flawed instrument is worse than
    # none: it would have split the rows evenly between a fast card and a
    # slow one and called that a measurement.
    per_plane = max(1, int(rows) * nlon * 4)
    height = max(16, (64 * 2**20) // per_plane)
    n = height * int(rows) * nlon
    field = xp.asarray(
        (np.arange(n, dtype=np.float64) % 1024.0).reshape(height, int(rows), nlon),
        dtype=dtype,
    )
    best = float("inf")
    for _ in range(int(repeats)):
        if transform.backend.name == "cupy":
            transform.backend.synchronize()
        started = time.perf_counter()
        spectrum = xp.fft.rfft(field, axis=-1)
        product = spectrum * spectrum.conj()
        total = xp.sum(product.real)
        if transform.backend.name == "cupy":
            transform.backend.synchronize()
        else:
            float(total)
        best = min(best, time.perf_counter() - started)
        del spectrum, product, total
    del field
    return best * 1e3


def sum_in_rank_order(parts):
    """Add gathered partials in ascending rank order, left to right.

    The one place a floating-point sum crosses the wire.  A collective
    all-reduce would add them in whatever order its tree chose, and the
    tree moves with the rank count and the message size, so the same two
    cards would return two different numbers for two different chunk
    sizes.  Here the order is the rank order and nothing else.
    """
    ordered = list(parts)
    if not ordered:
        raise ValueError("no partials to sum")
    total = ordered[0]
    for piece in ordered[1:]:
        total = total + piece
    return total


# ---------------------------------------------------------------------
# The wire ledger
# ---------------------------------------------------------------------


class WireLedger:
    """What crossed the wire, and how much of it the run had to wait for.

    Two different quantities, and gate WIRE-1 needs both.  ``posted`` is
    every byte the transport moved and ``sender_seconds`` is the time the
    sending threads spent inside ``sendall``, so their ratio is the rate
    ACHIEVED WHILE BOTH CARDS COMPUTE -- which is R10, the risk that the
    idle-card interconnect measurements do not survive a real run.
    ``exposed_seconds`` is the time the model itself blocked waiting for a
    peer, which is the only part of the wire that shows up in the step.
    """

    def __init__(self):
        self.exchanges = 0
        #: Posted bytes per exchange NAME (the tag without its counter).
        #: Which exchange the wire is actually spent on is the whole
        #: two-card verdict: a design that priced the Fourier waist and
        #: found the grid tracers is a design that measured the wrong
        #: thing, and this is what tells the two apart.
        self.by_name: dict[str, list] = {}
        self.posted_bytes = 0
        self.received_bytes = 0
        self.sender_seconds = 0.0
        self.exposed_seconds = 0.0
        self.steps = 0
        self._lock = threading.Lock()
        #: Set by the transport that feeds this ledger: a callable that
        #: returns once every posted frame has been recorded here.  The
        #: ledger owns the wait rather than each caller, because a reader
        #: that forgets it reads a number that is quietly too small and
        #: nothing in the receipt says so.
        self.drain_hook = None

    def record_send(self, nbytes: int, seconds: float, name: str = "") -> None:
        with self._lock:
            self.posted_bytes += int(nbytes)
            self.sender_seconds += float(seconds)
            if name:
                row = self.by_name.setdefault(name, [0, 0.0, 0])
                row[0] += int(nbytes)
                row[1] += float(seconds)
                row[2] += 1

    def record_exposure(self, nbytes: int, seconds: float) -> None:
        with self._lock:
            self.received_bytes += int(nbytes)
            self.exposed_seconds += float(seconds)
            self.exchanges += 1

    def mark_step(self) -> None:
        with self._lock:
            self.steps += 1

    @property
    def achieved_bytes_s(self) -> float:
        if self.sender_seconds <= 0.0:
            return 0.0
        return self.posted_bytes / self.sender_seconds

    def receipt(self) -> dict[str, object]:
        # The sender threads first: a post returns when the frame is
        # queued, and the bytes and the seconds are handed over here only
        # once it has left.  MEASURED 2026-09-06: without this wait a
        # one-megabyte all-gather read back 0 posted bytes on two of every
        # three attempts, and WIRE-1's rate, its bytes per step and its
        # per-exchange breakdown are all defined on that count.
        if self.drain_hook is not None:
            self.drain_hook()
        steps = max(1, int(self.steps))
        return {
            "exchanges": int(self.exchanges),
            "exchanges_per_step": self.exchanges / steps,
            "posted_bytes": int(self.posted_bytes),
            "received_bytes": int(self.received_bytes),
            "bytes_per_step": (self.posted_bytes + self.received_bytes) / steps,
            "sender_seconds": float(self.sender_seconds),
            "exposed_seconds": float(self.exposed_seconds),
            "exposed_ms_per_step": 1e3 * self.exposed_seconds / steps,
            "achieved_bytes_s": float(self.achieved_bytes_s),
            "achieved_gb_s": float(self.achieved_bytes_s / 1e9),
            "wire_floor_gb_s": WIRE_FLOOR_BYTES_S / 1e9,
            "steps": int(self.steps),
            "posted_by_exchange": {
                name: {
                    "posted_bytes": row[0],
                    "posted_mb_per_step": row[0] / 2**20 / steps,
                    "sender_seconds": row[1],
                    "calls_per_step": row[2] / steps,
                }
                for name, row in sorted(
                    self.by_name.items(), key=lambda kv: -kv[1][0])
            },
        }


# ---------------------------------------------------------------------
# The transports
# ---------------------------------------------------------------------


class CardTransport:
    """Move bytes between ranks.  Never reduce, never reorder, never add.

    Three calls carry the whole model: :meth:`post` hands a payload to the
    peers without waiting, :meth:`collect` waits for the peers' payloads
    for one tag, and :meth:`all_gather` is the two of them together for
    the cases with nothing to overlap.  A tag is a string every rank
    agrees on for the same exchange; posting different tags on different
    ranks is a scheduling defect and :meth:`collect` names it by timeout
    rather than hanging until the harness reaps the run.
    """

    name = "none"

    def __init__(self, rank: int, world: int, *, ledger: WireLedger | None = None):
        self.rank = int(rank)
        self.world = int(world)
        self.ledger = ledger if ledger is not None else WireLedger()
        #: What the run asked for and why this transport is the one that
        #: runs: the receipt carries both, so a rate is never read against
        #: the wrong transport.
        self.requested = self.name
        self.reason = ""

    def post(self, tag: str, payload) -> None:
        raise NotImplementedError

    def collect(self, tag: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> list:
        raise NotImplementedError

    def all_gather(self, tag: str, payload, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> list:
        self.post(tag, payload)
        pieces = self.collect(tag, timeout_s=timeout_s)
        pieces[self.rank] = payload
        return pieces

    def barrier(self, tag: str = "barrier") -> None:
        self.all_gather(tag, b"\x00")

    # -- point to point -------------------------------------------------
    def post_to(self, peer: int, tag: str, payload) -> None:
        """Hand a payload to ONE peer without waiting."""
        raise NotImplementedError

    def collect_from(self, tag: str, peers, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
        """Wait for ``tag`` from each of ``peers`` only; ``{peer: payload}``."""
        raise NotImplementedError

    # -- array exchanges ------------------------------------------------
    #
    # The base implementation stages through host bytes, which is what the
    # TCP transport runs.  The bytes that arrive are the bytes the peer
    # computed, so the transport never enters the answer.

    def all_gather_arrays(self, tag: str, xp, local, shapes, dtype) -> list:
        """Every rank's array, in rank order: ``shapes[r]`` is rank ``r``'s.

        The bytes that arrive are the bytes each rank computed, PLACED and
        never combined; any sum over the pieces is the caller's, in rank
        order (:func:`sum_in_rank_order`).  This rank's own entry is
        ``local`` itself.
        """
        self.post(tag, _to_host_bytes(xp, local))
        pieces = self.collect(tag)
        out = []
        for rank, piece in enumerate(pieces):
            if rank == self.rank:
                out.append(local)
                continue
            out.append(_array_from_host_bytes(xp, piece, tuple(shapes[rank]), dtype))
        return out

    def exchange_with(self, tag: str, xp, sends: dict, recv_shapes: dict, dtype) -> dict:
        """Point-to-point: send ``sends[peer]`` to each peer named there and
        receive one array of ``recv_shapes[peer]`` from each peer named
        there.  Only the named peers see a byte (the halo's two neighbours)."""
        for peer in sorted(sends):
            self.post_to(int(peer), tag, _to_host_bytes(xp, sends[peer]))
        got = self.collect_from(tag, sorted(recv_shapes))
        return {
            peer: _array_from_host_bytes(xp, got[peer], tuple(recv_shapes[peer]), dtype)
            for peer in sorted(recv_shapes)
        }

    def close(self) -> None:
        pass

    def drain(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Wait until the wire ledger has every byte this rank posted.

        A transport that records its sends inline has nothing to wait for
        and says so by doing nothing.
        """
        return None

    def receipt(self) -> dict[str, object]:
        return {
            "transport": self.name,
            "transport_requested": self.requested,
            "transport_reason": self.reason,
            "rank": self.rank,
            "cards": self.world,
        }


class SingleCard(CardTransport):
    """One card: every exchange is the identity, and nothing is allocated.

    The single-card path is not a special case bolted beside the two-card
    one; it is the two-card one with a world of one, so the resident run
    executes the same code and there is one implementation of the step
    rather than two that happen to agree.
    """

    name = "single"

    def __init__(self, ledger: WireLedger | None = None):
        super().__init__(0, 1, ledger=ledger)

    def post(self, tag: str, payload) -> None:
        return None

    def collect(self, tag: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> list:
        return [None]

    def all_gather(self, tag: str, payload, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> list:
        return [payload]

    def barrier(self, tag: str = "barrier") -> None:
        return None


class _Peer:
    """One socket, its sender thread and its received frames."""

    def __init__(self, rank: int, sock: socket.socket, ledger: WireLedger, chunk: int):
        self.rank = int(rank)
        self.sock = sock
        self.ledger = ledger
        self.chunk = int(chunk)
        self.frames: dict[str, bytes] = {}
        self.error: BaseException | None = None
        self._condition = threading.Condition()
        self._outbox: queue.Queue = queue.Queue()
        self._closing = False
        #: Frames posted whose bytes the sender thread has not yet handed
        #: to the ledger.  :meth:`drain` waits for this to reach zero.
        self._unrecorded = 0
        self._sender = threading.Thread(
            target=self._send_loop, name=f"cards-send-{rank}", daemon=True)
        self._receiver = threading.Thread(
            target=self._recv_loop, name=f"cards-recv-{rank}", daemon=True)
        self._sender.start()
        self._receiver.start()

    # -- sending ------------------------------------------------------
    def post(self, tag: str, payload: memoryview | bytes) -> None:
        with self._condition:
            self._unrecorded += 1
        self._outbox.put((tag, payload))

    def drain(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Block until every posted frame has reached the ledger.

        A post returns as soon as the frame is queued and the SENDER
        THREAD records its bytes and its seconds, so the ledger trails the
        model by however many frames are in flight.  That is what the
        overlap is for and it must not be paid per exchange, but a receipt
        read while frames are outstanding reports fewer bytes than crossed
        the wire, and WIRE-1's rate, its bytes per step and its per-
        exchange breakdown are all defined on that count.  MEASURED: a
        one-megabyte all-gather read back 0 posted bytes on two of every
        three attempts.  So the wait is taken exactly where the numbers
        are consumed.
        """
        deadline = time.perf_counter() + float(timeout_s)
        with self._condition:
            while self._unrecorded > 0:
                if self.error is not None:
                    raise RuntimeError(
                        f"card rank {self.rank} failed with "
                        f"{self._unrecorded} frames unsent: {self.error!r}"
                    ) from self.error
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"card rank {self.rank} still holds {self._unrecorded} "
                        f"posted frames after {timeout_s:.1f} s; the wire "
                        "ledger would report fewer bytes than crossed the wire"
                    )
                self._condition.wait(min(remaining, 0.05))

    def _send_loop(self) -> None:
        try:
            while True:
                item = self._outbox.get()
                if item is None:
                    return
                tag, payload = item
                raw = tag.encode("utf-8")
                view = memoryview(payload).cast("B")
                started = time.perf_counter()
                self.sock.sendall(_HEADER.pack(len(raw), view.nbytes) + raw)
                # 8 MB writes: the operating point of the MEASURED curve,
                # and small enough that the peer's first chunk is on its
                # card while the rest is still on the wire.
                for offset in range(0, view.nbytes, self.chunk):
                    self.sock.sendall(view[offset:offset + self.chunk])
                self.ledger.record_send(
                    view.nbytes, time.perf_counter() - started,
                    tag.rsplit("#", 1)[0])
                with self._condition:
                    self._unrecorded -= 1
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - reported to the model thread
            with self._condition:
                self.error = exc
                self._condition.notify_all()

    # -- receiving ----------------------------------------------------
    def _recv_exact(self, nbytes: int) -> bytes:
        chunks = []
        remaining = int(nbytes)
        while remaining > 0:
            piece = self.sock.recv(min(remaining, self.chunk))
            if not piece:
                raise ConnectionError(
                    f"card rank {self.rank} closed the connection with "
                    f"{remaining} bytes of a frame outstanding: a dead rank "
                    "kills a two-card run and this one is telling you which"
                )
            chunks.append(piece)
            remaining -= len(piece)
        return b"".join(chunks)

    def _recv_loop(self) -> None:
        try:
            while True:
                header = self._recv_exact(_HEADER.size)
                tag_len, payload_len = _HEADER.unpack(header)
                tag = self._recv_exact(tag_len).decode("utf-8")
                payload = self._recv_exact(payload_len)
                with self._condition:
                    self.frames[tag] = payload
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001
            with self._condition:
                if not self._closing:
                    self.error = exc
                self._condition.notify_all()

    def take(self, tag: str, timeout_s: float) -> bytes:
        deadline = time.perf_counter() + float(timeout_s)
        with self._condition:
            while tag not in self.frames:
                if self.error is not None:
                    raise RuntimeError(
                        f"card rank {self.rank} failed before delivering "
                        f"{tag!r}: {self.error!r}"
                    ) from self.error
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"waited {timeout_s:.0f} s for {tag!r} from card rank "
                        f"{self.rank}.  Every rank must post the same tags in "
                        "the same order; a rank that took a different branch "
                        "of the step is the usual cause"
                    )
                self._condition.wait(min(remaining, 1.0))
            return self.frames.pop(tag)

    def close(self) -> None:
        # DRAIN BEFORE SHUTTING DOWN.  A post is asynchronous, so a rank
        # that finished its last exchange can still have bytes in its own
        # outbox; closing the socket there delivers the peer an EOF in the
        # middle of a frame and the peer reports a dead rank for a run
        # that finished cleanly.  The sender thread processes its queue in
        # order, so joining it past the sentinel is the drain.
        self._closing = True
        self._outbox.put(None)
        self._sender.join(30.0)
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class TcpCards(CardTransport):
    """The transport that actually runs on this pair: TCP over the 25 GbE.

    A full mesh of one socket per rank pair, ``TCP_NODELAY`` on, each
    socket with its own sender and receiver thread so a post returns
    immediately and the bytes leave while the model computes the next
    band.  That overlap is the whole difference between the design's
    1.37x and its 1.06x, and it is the first thing this lane measures.

    Between two nodes over 25 GbE the interconnect lane measured NCCL
    over sockets, NCCL over RoCE, MPI over TCP and a hand-staged pipeline
    within 1 percent of each other, so this is the inter-node transport
    and costs nothing there.  It is ALSO the rendezvous and the host
    channel of every multi-card run: :class:`NcclCards` is built on top
    of it and keeps it for the small host payloads.  Inside one box it is
    the wrong device path (host staging through loopback, MEASURED
    0.28 to 3.1 GB/s per rank, 2026-10-05), which is what NCCL replaces
    wherever every rank can open it.
    """

    name = "tcp"

    def __init__(
        self,
        rank: int,
        world: int,
        addresses,
        *,
        chunk_bytes: int = CHUNK_BYTES,
        ledger: WireLedger | None = None,
        connect_timeout_s: float = 120.0,
        listen_timeout_s: float = 1800.0,
    ):
        super().__init__(rank, world, ledger=ledger)
        self.addresses = [self._split(a) for a in addresses]
        if len(self.addresses) != self.world:
            raise ValueError(
                f"{len(self.addresses)} rendezvous addresses for {self.world} "
                "cards: every rank needs one host:port, in rank order"
            )
        self.chunk_bytes = int(chunk_bytes)
        self._peers: dict[int, _Peer] = {}
        self.ledger.drain_hook = self.drain
        self._connect(connect_timeout_s, listen_timeout_s)

    @staticmethod
    def _split(address) -> tuple[str, int]:
        text = str(address)
        host, _, port = text.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"card address must be host:port, got {address!r}")
        return host, int(port)

    def _connect(self, timeout_s: float, listen_timeout_s: float) -> None:
        """Lower ranks listen, higher ranks dial: one socket per pair.

        THE LISTENER HAS A DEADLINE, and it has one because it once did
        not: on 2026-09-06 the second rank was refused by the memory gate
        before it dialled (its card had picked up a co-tenant), and the
        first rank sat in ``accept`` holding ten gigabytes of a shared
        card until it was killed by hand.  A rank that never arrives is a
        dead rank and it is named here rather than waited on.
        """
        host, port = self.addresses[self.rank]
        listener = None
        if self.rank < self.world - 1:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("", port))
            listener.listen(self.world)
            listener.settimeout(float(listen_timeout_s))
        try:
            for peer in range(self.rank):
                sock = self._dial(self.addresses[peer], timeout_s)
                sock.sendall(struct.pack("!I", self.rank))
                self._peers[peer] = _Peer(peer, sock, self.ledger, self.chunk_bytes)
            for _ in range(self.world - 1 - self.rank):
                try:
                    sock, _addr = listener.accept()
                except (TimeoutError, OSError) as exc:
                    raise ConnectionError(
                        f"no card dialled rank {self.rank} on port {port} "
                        f"within {listen_timeout_s:.0f} s: {exc!r}.  A rank "
                        "that was refused before it dialled -- the memory "
                        "gate on a card that picked up a co-tenant is the "
                        "way this happens -- leaves this one holding the "
                        "card for nothing, so it stops instead of waiting"
                    ) from exc
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                (peer,) = struct.unpack("!I", _recv_all(sock, 4))
                self._peers[int(peer)] = _Peer(
                    int(peer), sock, self.ledger, self.chunk_bytes)
        finally:
            if listener is not None:
                listener.close()
        del host

    @staticmethod
    def _dial(address, timeout_s: float) -> socket.socket:
        host, port = address
        deadline = time.perf_counter() + float(timeout_s)
        last = None
        while time.perf_counter() < deadline:
            try:
                sock = socket.create_connection((host, port), timeout=10.0)
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError as exc:
                last = exc
                time.sleep(0.25)
        raise ConnectionError(
            f"no card answered at {host}:{port} within {timeout_s:.0f} s: "
            f"{last!r}.  Every rank must be launched before the rendezvous "
            "timeout, and the address list must be in rank order"
        )

    def post(self, tag: str, payload) -> None:
        for peer in self._peers.values():
            peer.post(tag, payload)

    def collect(self, tag: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> list:
        started = time.perf_counter()
        pieces: list = [None] * self.world
        total = 0
        for rank in sorted(self._peers):
            payload = self._peers[rank].take(tag, timeout_s)
            pieces[rank] = payload
            total += len(payload)
        self.ledger.record_exposure(total, time.perf_counter() - started)
        return pieces

    def post_to(self, peer: int, tag: str, payload) -> None:
        self._peers[int(peer)].post(tag, payload)

    def collect_from(self, tag: str, peers, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
        started = time.perf_counter()
        got = {}
        total = 0
        for rank in sorted(int(p) for p in peers):
            payload = self._peers[rank].take(tag, timeout_s)
            got[rank] = payload
            total += len(payload)
        self.ledger.record_exposure(total, time.perf_counter() - started)
        return got

    def close(self) -> None:
        for peer in self._peers.values():
            peer.close()
        self._peers.clear()

    def drain(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Every posted frame accounted for, on every peer socket.

        Nothing in the step calls this.  The overlap the two-card speedup
        rests on IS the ledger trailing the model, and paying this wait
        per exchange would remove the thing being measured; it is paid
        where the numbers are read instead.
        """
        for peer in list(self._peers.values()):
            peer.drain(timeout_s)

    def receipt(self) -> dict[str, object]:
        self.drain()
        row = super().receipt()
        row["chunk_bytes"] = int(self.chunk_bytes)
        row["addresses"] = [f"{h}:{p}" for h, p in self.addresses]
        return row


def _recv_all(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        piece = sock.recv(remaining)
        if not piece:
            raise ConnectionError("peer closed during the rank handshake")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


class NcclCards(CardTransport):
    """NCCL as a DEVICE TRANSPORT: bytes only, no reduction, IB disabled.

    Built on an open :class:`TcpCards` mesh, which stays the rendezvous
    and the host channel: every host payload (the band profile, the
    placement table, the agreement digests, the associative folds, the
    closing barrier) still crosses it, and so does any exchange of a NumPy
    array (a checkpoint gather).  The DEVICE arrays the step exchanges --
    waist rows, partial sums, order columns, halo edges -- go through the
    communicator as device pointers on the current stream: no
    device-to-host copy, no ``tobytes``, no socket, no GIL.

    The answer cannot move.  An all-gather of unequal pieces is a grouped
    broadcast from every root of that root's exact bytes (``ncclUint8``),
    the halo is a grouped send/receive with the two neighbours only, and
    nothing is reduced on the wire: every sum is the caller's, afterwards,
    locally, in ascending rank order (:func:`sum_in_rank_order`), exactly
    as over TCP.  So one card, P cards over TCP and P cards over NCCL
    assemble the same bytes and BIT-5 and BIT-6 hold by construction.

    The wire ledger is fed from CUDA events around each group, resolved
    when the receipt is read (:meth:`drain`) rather than by synchronising
    the stream per exchange, so the device rate WIRE-1 grades is the rate
    the collectives ran at.  A collective cannot finish before its slowest
    rank arrives, so that time includes any wait on a peer still
    computing, and the rate is a LOWER bound on the link (MEASURED
    2026-10-05 across an RTX 4090 and an RTX 5070 Ti on the 2.5 GbE LAN:
    0.19 and 0.03 GB/s on the two ranks of one T31 run, the second rank
    being the one that waited).  There is no chunk: the 8 MB operating
    point was fitted to the 25 GbE curve and belongs to the TCP transport
    only.

    ``NCCL_IB_DISABLE=1`` is set by :func:`launch_environment` and
    recorded in the receipt: without it NCCL tries InfiniBand verbs on an
    Ethernet link between nodes and stalls rather than failing.
    """

    name = "nccl"

    @staticmethod
    def available() -> bool:
        try:
            from cupy.cuda import nccl  # type: ignore
        except Exception:  # noqa: BLE001
            return False
        return bool(getattr(nccl, "available", False))

    #: Device groups between two cross-rank checks of the tag sequence.
    TAG_CHECK_EVERY = 32
    #: Bytes each rank broadcasts in the device link probe at session open.
    LINK_PROBE_BYTES = 32 * 2**20
    #: Repeats of the link probe; the fastest is the link's rate.
    LINK_PROBE_REPEATS = 3

    def __init__(self, mesh: "TcpCards", communicator, nccl=None, *, device_xp=None,
                 timeout_s: float = DEFAULT_TIMEOUT_S, probe: bool = True):
        super().__init__(mesh.rank, mesh.world, ledger=mesh.ledger)
        self.mesh = mesh
        self._comm = communicator
        if nccl is None:
            from cupy.cuda import nccl  # type: ignore
        self._nccl = nccl
        if device_xp is None:
            import cupy as device_xp  # type: ignore
        #: The array module whose arrays take the device path; any other
        #: (NumPy) stages through the mesh.
        self._dev = device_xp
        self._uint8 = int(getattr(nccl, "NCCL_UINT8", 1))
        #: (start, end, posted, received, name) per group, resolved lazily.
        self._timings: list = []
        self.device_exchanges = 0
        self.ledger.drain_hook = self.drain
        # THE TWO SAFETY PROPERTIES THE TCP PATH HAS, kept on the device
        # path.  R17: a dead rank must fail loudly and soon.  A collective
        # has no timeout of its own, so a watchdog aborts the communicator
        # when a group stays unfinished past ``timeout_s``, and the next
        # exchange (or the receipt) raises naming the group's tag.  And the
        # tag: two ranks running different exchanges of equal byte counts
        # would swap bytes silently, so every rank folds each group's tag
        # and sizes into a running digest and the ranks compare it over the
        # host mesh every TAG_CHECK_EVERY groups.
        self._timeout_s = float(timeout_s)
        self._watch_lock = threading.Lock()
        self._watching: list = []
        self._aborted: str | None = None
        self._watch_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watch, name=f"nccl-watchdog-{self.rank}", daemon=True)
        self._watchdog.start()
        self._tag_digest = b""
        self._tag_groups = 0
        #: The device link's rate, measured at session open with every rank
        #: released together (gate WIRE-1 on NCCL).
        self.link_probe_gb_s: float | None = None
        if probe:
            self.link_probe_gb_s = self._probe_link()

    # -- the host channel: the mesh ----------------------------------
    def post(self, tag: str, payload) -> None:
        self.mesh.post(tag, payload)

    def collect(self, tag: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> list:
        return self.mesh.collect(tag, timeout_s=timeout_s)

    def post_to(self, peer: int, tag: str, payload) -> None:
        self.mesh.post_to(peer, tag, payload)

    def collect_from(self, tag: str, peers, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
        return self.mesh.collect_from(tag, peers, timeout_s=timeout_s)

    # -- device plumbing ---------------------------------------------
    def _on_device(self, xp) -> bool:
        return xp is self._dev

    def _ptr(self, array) -> int:
        data = getattr(array, "data", None)
        ptr = getattr(data, "ptr", None)
        if ptr is not None:
            return int(ptr)
        return int(array.ctypes.data)

    def _stream_ptr(self) -> int:
        cuda = getattr(self._dev, "cuda", None)
        if cuda is None:
            return 0
        return int(cuda.get_current_stream().ptr)

    def _mark(self):
        cuda = getattr(self._dev, "cuda", None)
        if cuda is None:
            return time.perf_counter()
        event = cuda.Event()
        event.record()
        return event

    def _elapsed_s(self, start, end) -> float:
        cuda = getattr(self._dev, "cuda", None)
        if cuda is None:
            return float(end - start)
        end.synchronize()
        return float(cuda.get_elapsed_time(start, end)) / 1e3

    def _settle(self, *, wait: bool) -> None:
        """Hand finished groups' bytes and seconds to the ledger."""
        cuda = getattr(self._dev, "cuda", None)
        keep = []
        for row in self._timings:
            start, end, posted, received, name = row
            if not wait and cuda is not None and not end.done:
                keep.append(row)
                continue
            seconds = self._elapsed_s(start, end)
            self.ledger.record_send(posted, seconds, name)
            self.ledger.record_exposure(received, seconds)
        self._timings = keep

    def _group(self, tag: str, ops, posted: int, received: int) -> None:
        self._raise_if_aborted()
        self._check_tags(tag, posted, received)
        self._settle(wait=False)
        start = self._mark()
        self._nccl.groupStart()
        try:
            for op in ops:
                op()
        finally:
            self._nccl.groupEnd()
        end = self._mark()
        self._timings.append((start, end, int(posted), int(received),
                              tag.rsplit("#", 1)[0]))
        with self._watch_lock:
            self._watching.append((end, tag, time.monotonic()))
        self.device_exchanges += 1

    # -- the two safety properties ------------------------------------
    def _check_tags(self, tag: str, posted: int, received: int) -> None:
        """Fold this group into the running tag digest; every
        TAG_CHECK_EVERY groups, refuse by name when the ranks disagree."""
        import hashlib

        name = tag.rsplit("#", 1)[0]
        self._tag_digest = hashlib.sha256(
            self._tag_digest + f"{name}|{int(posted)}|{int(received)}".encode()
        ).digest()
        self._tag_groups += 1
        if self._tag_groups % self.TAG_CHECK_EVERY:
            return
        digests = self.mesh.all_gather(
            f"nccl-tag-digest#{self._tag_groups}", self._tag_digest,
            timeout_s=self._timeout_s)
        differing = [r for r, d in enumerate(digests)
                     if bytes(d) != self._tag_digest]
        if differing:
            raise RuntimeError(
                f"the card ranks ran different device exchanges: by group "
                f"{self._tag_groups} (this rank's last: {name!r}) the tag "
                f"sequence of ranks {differing} differs from rank "
                f"{self.rank}'s, and NCCL would have swapped bytes between "
                "different exchanges without noticing")

    def _watch(self) -> None:
        while not self._watch_stop.wait(1.0):
            now = time.monotonic()
            with self._watch_lock:
                pending = []
                for end, tag, posted_at in self._watching:
                    if self._done(end):
                        continue
                    if now - posted_at > self._timeout_s:
                        self._aborted = (
                            f"the device exchange {tag!r} did not finish within "
                            f"{self._timeout_s:.0f} s; a peer rank is dead or "
                            "stuck, and the communicator was aborted so this "
                            "rank fails instead of hanging (R17)")
                        pending = []
                        break
                    pending.append((end, tag, posted_at))
                self._watching = pending
            if self._aborted is not None:
                abort = getattr(self._comm, "abort", None)
                if abort is not None:
                    try:
                        abort()
                    except Exception:  # noqa: BLE001 - the flag is the report
                        pass
                return

    @staticmethod
    def _done(end) -> bool:
        done = getattr(end, "done", None)
        return True if done is None else bool(done)

    def _raise_if_aborted(self) -> None:
        if self._aborted is not None:
            raise TimeoutError(self._aborted)

    def _probe_link(self) -> float:
        """The device link's rate in GB/s, every rank released together.

        In a run a collective cannot finish before its slowest rank
        arrives, so the event time a rank records includes its wait on
        peers still computing and the ledger's rate is only a lower bound
        on the link.  The probe is the link itself: a host barrier, then an
        all-gather of LINK_PROBE_BYTES from every rank timed on the device,
        the fastest of LINK_PROBE_REPEATS and then the fastest rank (the
        one that waited least).  It stays out of the wire ledger and the
        tag digest, which describe the run's own exchanges.
        """
        xp = self._dev
        nbytes = int(self.LINK_PROBE_BYTES)
        if getattr(xp, "cuda", None) is None:
            # A host stand-in (the CPU tests): the rate means nothing there,
            # and the exchange is exercised with a token payload.
            nbytes = 4096
        buffers = [xp.zeros(nbytes, dtype=np.uint8) for _ in range(self.world)]
        stream = self._stream_ptr()
        best = 0.0
        for repeat in range(int(self.LINK_PROBE_REPEATS)):
            self.mesh.all_gather(f"nccl-probe-barrier#{repeat}", bytes(1),
                                 timeout_s=self._timeout_s)
            start = self._mark()
            self._nccl.groupStart()
            try:
                for root, buf in enumerate(buffers):
                    ptr = self._ptr(buf)
                    self._comm.broadcast(ptr, ptr, nbytes, self._uint8, root,
                                         stream)
            finally:
                self._nccl.groupEnd()
            end = self._mark()
            seconds = self._elapsed_s(start, end)
            moved = nbytes * (self.world - 1)
            if seconds > 0.0:
                best = max(best, moved / seconds / 1e9)
        del buffers
        rates = self.mesh.all_gather(
            "nccl-probe-rate", f"{best:.9e}".encode("ascii"),
            timeout_s=self._timeout_s)
        return max(float(bytes(r).decode("ascii")) for r in rates)

    # -- the device exchanges ----------------------------------------
    def all_gather_arrays(self, tag: str, xp, local, shapes, dtype) -> list:
        if not self._on_device(xp):
            return super().all_gather_arrays(tag, xp, local, shapes, dtype)
        send = xp.ascontiguousarray(local)
        out = [send if r == self.rank else xp.empty(tuple(shapes[r]), dtype=dtype)
               for r in range(self.world)]
        stream = self._stream_ptr()
        ops = []
        for root, buf in enumerate(out):
            nbytes = int(buf.nbytes)
            if nbytes == 0:
                continue
            ptr = self._ptr(buf)
            ops.append(lambda ptr=ptr, nbytes=nbytes, root=root: self._comm.broadcast(
                ptr, ptr, nbytes, self._uint8, root, stream))
        received = sum(int(out[r].nbytes) for r in range(self.world) if r != self.rank)
        self._group(tag, ops, int(send.nbytes) * (self.world - 1), received)
        out[self.rank] = local
        return out

    def exchange_with(self, tag: str, xp, sends: dict, recv_shapes: dict, dtype) -> dict:
        if not self._on_device(xp):
            return super().exchange_with(tag, xp, sends, recv_shapes, dtype)
        stream = self._stream_ptr()
        outgoing = {int(p): xp.ascontiguousarray(a) for p, a in sends.items()}
        incoming = {int(p): xp.empty(tuple(shape), dtype=dtype)
                    for p, shape in recv_shapes.items()}
        ops = []
        for peer in sorted(outgoing):
            buf = outgoing[peer]
            if buf.nbytes:
                ptr, n = self._ptr(buf), int(buf.nbytes)
                ops.append(lambda ptr=ptr, n=n, peer=peer: self._comm.send(
                    ptr, n, self._uint8, peer, stream))
        for peer in sorted(incoming):
            buf = incoming[peer]
            if buf.nbytes:
                ptr, n = self._ptr(buf), int(buf.nbytes)
                ops.append(lambda ptr=ptr, n=n, peer=peer: self._comm.recv(
                    ptr, n, self._uint8, peer, stream))
        self._group(
            tag, ops,
            sum(int(b.nbytes) for b in outgoing.values()),
            sum(int(b.nbytes) for b in incoming.values()))
        return {peer: incoming[peer] for peer in sorted(incoming)}

    # -- lifecycle ----------------------------------------------------
    def drain(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._raise_if_aborted()
        self.mesh.drain(timeout_s)
        self._settle(wait=True)
        self._raise_if_aborted()

    def close(self) -> None:
        self._watch_stop.set()
        try:
            if self._aborted is None:
                self._settle(wait=True)
        finally:
            destroy = getattr(self._comm, "destroy", None)
            if destroy is not None:
                try:
                    destroy()
                except Exception:  # noqa: BLE001 - the mesh still closes
                    pass
            self.mesh.close()

    def receipt(self) -> dict[str, object]:
        self.drain()
        row = super().receipt()
        mesh = self.mesh.receipt()
        row["addresses"] = mesh.get("addresses")
        row["host_channel"] = "tcp"
        row["host_channel_chunk_bytes"] = mesh.get("chunk_bytes")
        row["device_exchanges"] = int(self.device_exchanges)
        row["link_probe_gb_s"] = self.link_probe_gb_s
        row["tag_checked_groups"] = int(
            self._tag_groups - self._tag_groups % self.TAG_CHECK_EVERY)
        try:
            row["nccl_version"] = int(self._nccl.get_version())
        except Exception:  # noqa: BLE001
            row["nccl_version"] = None
        row["nccl_ib_disable"] = os.environ.get("NCCL_IB_DISABLE")
        row["nccl_socket_ifname"] = os.environ.get("NCCL_SOCKET_IFNAME")
        return row


def _build_nccl(mesh: "TcpCards", want: bool, nccl=None, *, timeout_s: float = DEFAULT_TIMEOUT_S):
    """Vote over the mesh, then build the communicator, or say why not.

    Every rank must take the same branch or the first collective hangs, so
    the decision is taken from votes every rank sees: rank 0's NCCL unique
    id crosses the mesh only when every rank voted yes, and the
    communicator is kept only when every rank reports it built.  Returns
    ``(communicator or None, nccl module, reason)``.
    """
    import json

    vote = b"1" if want else b"0"
    votes = mesh.all_gather("nccl-vote", vote, timeout_s=timeout_s)
    missing = [r for r, v in enumerate(votes) if bytes(v) != b"1"]
    if missing:
        return None, None, (
            f"card ranks {missing} cannot open NCCL (no cupy.cuda.nccl, or "
            "a NumPy backend)")
    if nccl is None:
        from cupy.cuda import nccl  # type: ignore
    uid = b""
    first_error = ""
    if mesh.rank == 0:
        try:
            raw = nccl.get_unique_id()
            # CuPy 14 hands the id over as bytes, older CuPy as a tuple of
            # ints, and NcclCommunicator wants it back in the same type.
            if isinstance(raw, (bytes, bytearray)):
                uid = b"B" + bytes(raw)
            else:
                uid = b"T" + json.dumps([int(v) for v in raw]).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - the vote below carries it
            first_error = repr(exc)
    pieces = mesh.all_gather("nccl-uid", uid, timeout_s=timeout_s)
    if not pieces[0]:
        return None, None, "rank 0 could not draw an NCCL unique id: " + (
            first_error if mesh.rank == 0 else "see rank 0")
    head, body = bytes(pieces[0][:1]), bytes(pieces[0][1:])
    unique_id = body if head == b"B" else tuple(json.loads(body.decode("utf-8")))
    comm = None
    error = ""
    try:
        comm = nccl.NcclCommunicator(mesh.world, unique_id, mesh.rank)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    ready = mesh.all_gather(
        "nccl-ready", b"1" if comm is not None else error.encode("utf-8")[:512] or b"0",
        timeout_s=timeout_s)
    failed = [r for r, v in enumerate(ready) if bytes(v) != b"1"]
    if failed:
        if comm is not None:
            try:
                comm.destroy()
            except Exception:  # noqa: BLE001
                pass
        return None, None, (
            f"the NCCL communicator did not build on card ranks {failed}: "
            + "; ".join(bytes(ready[r]).decode("utf-8", "replace") for r in failed))
    return comm, nccl, "every rank opened NCCL"


def launch_environment(interface: str | None = None) -> dict[str, str]:
    """The environment a card rank is launched with, and the receipt records it.

    ``NCCL_IB_DISABLE=1`` unconditionally: the link between the nodes is
    25 GbE, and NCCL trying InfiniBand verbs on it stalls rather than
    failing.  Setting it on a run that never loads NCCL costs nothing and
    means the receipt of every two-card run carries the same field, so a
    reader never has to guess which transport a rate belongs to.
    """
    env = {"NCCL_IB_DISABLE": "1"}
    if interface:
        env["NCCL_SOCKET_IFNAME"] = str(interface)
    return env


def open_transport(
    rank: int,
    world: int,
    addresses=None,
    *,
    prefer: str = "auto",
    chunk_bytes: int = CHUNK_BYTES,
    ledger: WireLedger | None = None,
    device: bool = False,
    nccl=None,
    device_xp=None,
) -> CardTransport:
    """The transport for this rank: single card, NCCL on the TCP mesh, or TCP.

    Every multi-card run opens the TCP mesh first; it is the rendezvous.
    ``auto`` then takes NCCL for the device exchanges when every rank runs
    on a CUDA device and can open it, which in one box is NVLink or PCIe
    peer traffic instead of loopback TCP through host memory (MG-2), and
    keeps the mesh alone otherwise.  ``auto`` NEVER refuses: it used to
    raise as soon as ``cupy.cuda.nccl`` imported, which is the normal state
    of a rented CUDA box image, so every default multi-card run there died
    at the door (MG-3).  Only an explicit ``nccl`` that cannot be honoured
    is an error, because an operator who named a transport and got another
    would read its rate as the wrong link.  The receipt names the
    transport that ran, the one asked for, and why.

    ``device`` is whether this rank's arrays live on a CUDA device (the
    CuPy backend); ``nccl`` and ``device_xp`` replace ``cupy.cuda.nccl`` and
    ``cupy`` for a test.
    """
    if int(world) <= 1:
        return SingleCard(ledger=ledger)
    choice = str(prefer).strip().lower()
    if choice not in {"auto", "tcp", "nccl"}:
        raise ValueError(
            f"card_transport must be 'auto', 'tcp' or 'nccl', got {prefer!r}")
    if choice == "nccl" and nccl is None and not NcclCards.available():
        raise RuntimeError(
            "card_transport='nccl' was asked for and cupy.cuda.nccl is not "
            "importable in this environment (MEASURED 2026-09-06: CuPy 14.2.0 "
            "on CUDA 13 without nvidia-nccl-cu13 on either node).  Install "
            "nvidia-nccl-cu13 or select 'auto', which runs the TCP mesh "
            "where NCCL is absent"
        )
    if not addresses:
        raise ValueError(
            f"{world} cards need {world} rendezvous addresses in rank order"
        )
    mesh = TcpCards(
        rank, world, addresses, chunk_bytes=chunk_bytes, ledger=ledger)
    mesh.requested = choice
    if choice == "tcp":
        mesh.reason = "card_transport='tcp' was asked for"
        return mesh
    want = bool(device) and (nccl is not None or NcclCards.available())
    try:
        comm, module, reason = _build_nccl(mesh, want, nccl)
    except BaseException:
        mesh.close()
        raise
    if comm is None:
        if choice == "nccl":
            mesh.close()
            raise RuntimeError(
                f"card_transport='nccl' was asked for and {reason}; select "
                "'auto' to run the TCP mesh instead")
        mesh.reason = f"auto: {reason}, so the TCP mesh carries every exchange"
        return mesh
    transport = NcclCards(mesh, comm, module, device_xp=device_xp)
    transport.requested = choice
    transport.reason = (
        f"{choice}: {reason}; device exchanges on NCCL, host payloads on the "
        "TCP mesh")
    return transport


# ---------------------------------------------------------------------
# Placement: one rank, one card
# ---------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "::1", "[::1]", "0.0.0.0"}

#: The placement each rank took in this process, so the door and the run
#: report the first decision rather than the second call reading it back.
_PLACED: dict = {}


def _host_key(address) -> str:
    host = str(address).rpartition(":")[0].strip().lower()
    if host in _LOOPBACK_HOSTS or host.startswith("127."):
        return "loopback"
    return host


def _host_card_count() -> int | None:
    """How many CUDA cards this host has, asked WITHOUT opening CUDA here.

    ``CUDA_VISIBLE_DEVICES`` is read once, when CUDA initialises, so the
    count comes from ``nvidia-smi -L`` in a child process; ``None`` when
    that cannot answer, and the UUID check at the session still refuses
    two ranks on one card."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))


def local_rank_of(rank: int, addresses) -> tuple[int, int]:
    """``(local rank, ranks on this host)`` from the rendezvous list.

    The ranks that share this rank's host are the ranks whose rendezvous
    address names the same host (every loopback spelling is one host); the
    local rank is this rank's position among them in rank order.  A pure
    function of the address list every rank holds, so no discovery service
    is needed and every rank reaches the same table.
    """
    keys = [_host_key(a) for a in addresses]
    mine = keys[int(rank)]
    local = sum(1 for k in keys[:int(rank)] if k == mine)
    return local, sum(1 for k in keys if k == mine)


def place_rank(rank: int, world: int, addresses, backend: str, *, env=None) -> dict:
    """Put this rank on its own card, before any CUDA context exists.

    A box with several cards runs one rank per card, and nothing used to
    choose the card: every rank opened the current device, so ``--cards 8``
    in an eight-card box put all eight ranks on device 0, where the sizer's
    probe read the same free memory eight times, admitted all eight, and
    the ranks then competed for one card while seven idled (the 2026-10-05
    audit, MG-4).  So the rank narrows ``CUDA_VISIBLE_DEVICES`` to its
    LOCAL rank's card (:func:`local_rank_of`) and the sizer's probe
    subprocess, which inherits the environment, reads the card the run
    will use.

    An operator's own choice is kept: a single entry already in
    ``CUDA_VISIBLE_DEVICES`` is that operator pinning this process, and a
    list is the set of cards this host offers, of which the rank takes the
    local-rank-th.  A list too short for the ranks sharing the host is
    refused by name, because the rank past its end would land on a card
    another rank already holds.  Two ranks that end up on one card anyway
    are refused at the session by device UUID (:func:`check_card_placement`).

    Where CUDA was already initialised in this process (a library caller
    that touched the device before ``run``), narrowing the environment
    changes nothing, so the rank also selects its card by index when more
    than one is still visible.  A CPU run and a one-card run are untouched.
    """
    env = os.environ if env is None else env
    record = {"placed": False, "backend": str(backend)}
    if int(world) <= 1 or str(backend).strip().lower() != "cupy":
        return record
    addresses = tuple(str(a) for a in (addresses or ()))
    if len(addresses) != int(world):
        return record
    key = (int(rank), int(world), addresses)
    done = _PLACED.get(key)
    if (env is os.environ and done is not None
            and env.get("CUDA_VISIBLE_DEVICES") == done["cuda_visible_devices"]):
        # The door placed this rank already; the run asks again so a
        # library caller that skipped the door is placed too.
        return dict(done)
    local, sharing = local_rank_of(rank, addresses)
    before = env.get("CUDA_VISIBLE_DEVICES")
    entries = [e.strip() for e in (before or "").split(",") if e.strip()]
    if before is not None and not entries:
        # Set and empty is an operator hiding every card from this process.
        # Overwriting it with the local rank would turn a deliberate "no
        # card" into device 0; it is left as set, and the cupy backend then
        # says there is no device.
        record["reason"] = (
            "CUDA_VISIBLE_DEVICES is set and empty (every card hidden); "
            "left as the operator set it")
        return record
    if len(entries) == 1:
        chosen, placed_by = entries[0], "operator"
    elif entries:
        if local >= len(entries):
            raise ValueError(
                f"card rank {rank} is rank {local} of the {sharing} ranks on "
                f"host {_host_key(addresses[rank])!r}, and CUDA_VISIBLE_DEVICES "
                f"names {len(entries)} cards ({before}): this rank would land "
                "on a card another rank already holds, and the two would "
                "compete for its memory.  Name one card per rank on this host"
            )
        chosen, placed_by = entries[local], "rank-from-list"
    else:
        count = _host_card_count() if env is os.environ else None
        if count is not None and sharing > count:
            raise ValueError(
                f"{sharing} card ranks share host "
                f"{_host_key(addresses[rank])!r} and it has {count} "
                f"card{'s' if count != 1 else ''}: rank {rank} (local rank "
                f"{local}) would have no card of its own, and ranks sharing "
                "a card split its memory while the run reads as a "
                f"{sharing}-card run.  Launch at most {count} "
                f"rank{'s' if count != 1 else ''} on this host"
            )
        chosen, placed_by = str(local), "rank"
    env["CUDA_VISIBLE_DEVICES"] = chosen
    record.update({
        "placed": True,
        "local_rank": int(local),
        "ranks_on_host": int(sharing),
        "cuda_visible_devices_before": before,
        "cuda_visible_devices": chosen,
        "placed_by": placed_by,
    })
    import sys

    if env is os.environ and "cupy" in sys.modules:
        # CUDA may already be initialised in this process, in which case
        # the narrowed variable is read by nobody; select by index then.
        try:
            cp = sys.modules["cupy"]
            count = int(cp.cuda.runtime.getDeviceCount())
        except Exception:  # noqa: BLE001 - no driver: the backend says so later
            count = 0
        if count > 1:
            index = int(local)
            if index >= count:
                raise ValueError(
                    f"card rank {rank} wants device {index} and this process "
                    f"sees {count}: CUDA was initialised before the rank was "
                    "placed, so start the rank in a fresh process"
                )
            cp.cuda.Device(index).use()
            record["selected_device_index"] = index
    if env is os.environ:
        _PLACED[key] = dict(record)
    return record


def card_identity(xp) -> dict | None:
    """The card this rank computes on: name, PCI bus and UUID.

    No host name: the UUID is unique across machines, and the package
    reads no hostname into anything it writes.

    ``None`` under NumPy, where there is no card to share."""
    if xp is np or getattr(xp, "__name__", "") != "cupy":
        return None
    dev = xp.cuda.Device()
    props = xp.cuda.runtime.getDeviceProperties(int(dev.id))
    uuid = props.get("uuid", b"")
    if isinstance(uuid, (bytes, bytearray)):
        uuid = bytes(uuid).hex()
    name = props.get("name", b"")
    if isinstance(name, (bytes, bytearray)):
        name = bytes(name).split(b"\x00", 1)[0].decode("utf-8", "replace")
    try:
        pci = str(dev.pci_bus_id)
    except Exception:  # noqa: BLE001
        pci = ""
    return {
        "device_index": int(dev.id),
        "name": str(name),
        "uuid": str(uuid),
        "pci_bus_id": pci,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def check_card_placement(transport: "CardTransport", identity) -> list:
    """All-gather every rank's card and refuse two ranks on one card.

    Two ranks on one card pass every gate the run has (the answer is still
    the one-card answer) while the card's memory is split between them and
    the other cards idle, so it is refused by name here, at the session,
    with both ranks and the card named.  The list is the receipt's
    ``card_devices`` block.
    """
    import json

    if transport.world <= 1:
        return [identity]
    pieces = transport.all_gather(
        "card-placement", json.dumps(identity, sort_keys=True).encode("utf-8"))
    devices = [json.loads(bytes(piece).decode("utf-8")) for piece in pieces]
    seen: dict = {}
    for rank, row in enumerate(devices):
        if not row or not row.get("uuid"):
            continue
        key = row["uuid"]
        if key in seen:
            other = seen[key]
            raise ValueError(
                f"card ranks {other} and {rank} both run on {row.get('name')!r} "
                f"(UUID {key}, PCI {row.get('pci_bus_id')}): two ranks on one "
                "card split its memory "
                "and leave another card idle.  Launch one rank per card (the "
                "rank's local rank picks its card when CUDA_VISIBLE_DEVICES is "
                "unset or lists the host's cards)"
            )
        seen[key] = rank
    return devices


# ---------------------------------------------------------------------
# The session: what the model asks
# ---------------------------------------------------------------------

#: Exchange modes.  ``gather`` ships waist rows and keeps the contraction
#: whole, so the bits do not move; ``partial`` ships partial Legendre sums
#: and adds them in rank order, which is a third of the bytes and a change
#: of arithmetic that carries its own pin.
EXCHANGE_MODES = ("gather", "partial")


class CardSession:
    """One rank's view of a multi-card run: who owns what, and the wire.

    Built once at run start and handed to the model.  It owns the band
    ownership table, the transport, the ledger and the exchange mode, and
    it is the ONLY object that knows a second card exists -- the
    operators see a band pipeline whose local slices happen to be a
    subset, and the waist and the reduction accumulators see a row
    exchange they can call.
    """

    def __init__(
        self,
        transport: CardTransport,
        nlat: int,
        bands: int,
        *,
        exchange: str = "gather",
        weights=None,
        band_ms=None,
        agreement: str = "refuse",
    ):
        mode = str(exchange).strip().lower()
        if mode not in EXCHANGE_MODES:
            raise ValueError(
                f"card_exchange must be one of {EXCHANGE_MODES}, got {exchange!r}"
            )
        self.transport = transport
        self.rank = transport.rank
        self.world = transport.world
        self.exchange = mode
        self.nlat = int(nlat)
        #: The cross-card agreement ledger: every distinct contraction
        #: shape this run presents, checked once across the ranks.
        self.agreement = ContractionAgreement(self, agreement)
        self.bands = int(bands)
        self.band_ms = None if band_ms is None else tuple(float(v) for v in band_ms)
        if self.world == 1:
            self.weights = (1.0,)
            self.owners = tuple([0] * self.bands)
        else:
            if weights is None:
                weights = (
                    throughput_weights(self.band_ms) if self.band_ms
                    else tuple([1.0] * self.world)
                )
            self.weights = tuple(float(v) for v in weights)
            self.owners = band_owners(self.nlat, self.bands, self.weights)
        self._edges = band_edges(self.nlat, self.bands)
        self._counter = 0
        #: Every rank's card (:func:`check_card_placement`), and how this
        #: rank was placed (:func:`place_rank`); both ride the receipt.
        self.devices: list = []
        self.placement: dict = {}

    # -- ownership ----------------------------------------------------
    def local_band_indices(self) -> tuple[int, ...]:
        return tuple(k for k, r in enumerate(self.owners) if r == self.rank)

    def rows_of_rank(self, rank: int) -> tuple[int, int]:
        """``(first, last)`` grid rows a rank owns; the runs are contiguous."""
        held = [k for k, r in enumerate(self.owners) if r == int(rank)]
        if not held:
            raise ValueError(f"rank {rank} owns no band")
        return self._edges[held[0]], self._edges[held[-1] + 1]

    def local_rows(self) -> tuple[int, int]:
        return self.rows_of_rank(self.rank)

    def tag(self, name: str) -> str:
        """A monotone tag: the ranks post exchanges in the same order, so
        the counter is the same on every rank and a divergence shows up as
        a timeout naming the tag rather than as a silent mismatch."""
        self._counter += 1
        return f"{name}#{self._counter}"

    # -- the exchanges the model calls --------------------------------
    def gather_rows(self, xp, buffer, axis: int, *, name: str = "rows"):
        """Fill the rows this rank does not own from the ranks that do.

        ``buffer`` carries the whole latitude axis at ``axis`` and holds
        this rank's rows; every other rank's rows come back over the wire
        as the bytes that rank computed, so the assembled buffer is the
        buffer a one-card run would have built.  That is the whole
        bit-identity argument of the ``gather`` mode, and it holds only
        because every rank runs the same code on the same inputs for the
        rows it owns.
        """
        if self.world == 1:
            return buffer
        first, last = self.local_rows()
        local = _slice_rows(buffer, axis, first, last)
        shapes = []
        for rank in range(self.world):
            r0, r1 = self.rows_of_rank(rank)
            shapes.append(_rows_shape(buffer, axis, r1 - r0))
        pieces = self.transport.all_gather_arrays(
            self.tag(name), xp, local, shapes, buffer.dtype)
        for rank, piece in enumerate(pieces):
            if rank == self.rank:
                continue
            r0, r1 = self.rows_of_rank(rank)
            _slice_rows(buffer, axis, r0, r1)[...] = piece
        return buffer

    def exchange_halo(self, xp, buffer, axis: int, width: int, *, name: str = "halo"):
        """ Fill the ``width`` rows either side of this card's row range.

        The meridional tracer sweep reads ``q[j-2 .. j+2]`` through its van
        Leer slope, so ``n`` sub-steps read ``2n`` rows beyond the block,
        and at a CARD boundary those rows live on the other card.  The
        design's deep halo is exactly this: exchange ``2n`` rows ONCE and
        run every sub-step behind them, rather than exchanging two rows
        ``n`` times.  The redundant arithmetic the halo rows attract reads
        identical inputs and produces identical values, so the sweep stays
        bit-exact (gate HALO-1 proved the same argument inside one card).

        Poles are walls, not neighbours, so the window is clipped there
        and nothing is asked for beyond them.

        NEIGHBOURS ONLY.  Each rank's rows are one contiguous run and the
        runs follow rank order (:func:`band_owners`), so the rows below
        ``first`` belong to rank ``r - 1`` and the rows from ``last`` to
        rank ``r + 1``; with every rank holding at least ``width`` rows
        (refused otherwise, below) the whole window lies on that one
        neighbour.  So each edge goes to the one rank that reads it and to
        nobody else, and the halo bytes per rank are two edges whatever
        the card count (the 2026-10-05 audit MEASURED 3.0x the need at four
        cards when every edge went to every rank).
        """
        if self.world == 1 or int(width) <= 0:
            return buffer
        w = int(width)
        first, last = self.local_rows()
        # Checked over EVERY rank, not only this one: the rows each rank
        # owns are a pure function of the assignment every rank holds, so
        # every rank reaches the same verdict and none of them is left
        # waiting on a neighbour that refused alone.
        narrowest = min(
            (b - a, r) for r, (a, b) in
            ((r, self.rows_of_rank(r)) for r in range(self.world)))
        if w > narrowest[0]:
            own = last - first
            raise ValueError(
                f"a {w}-row halo is wider than the {narrowest[0]} rows card "
                f"rank {narrowest[1]} owns (this card owns {own}): the deep "
                "halo would reach past the neighbour into a third card's "
                "rows.  Use fewer cards, more rows, or a smaller sub-step "
                "count"
            )
        sends = {}
        recv_shapes = {}
        if self.rank > 0:
            sends[self.rank - 1] = _slice_rows(buffer, axis, first, first + w)
            recv_shapes[self.rank - 1] = _rows_shape(buffer, axis, w)
        if self.rank < self.world - 1:
            sends[self.rank + 1] = _slice_rows(buffer, axis, last - w, last)
            recv_shapes[self.rank + 1] = _rows_shape(buffer, axis, w)
        got = self.transport.exchange_with(
            self.tag(name), xp, sends, recv_shapes, buffer.dtype)
        if self.rank > 0:
            _slice_rows(buffer, axis, first - w, first)[...] = got[self.rank - 1]
        if self.rank < self.world - 1:
            _slice_rows(buffer, axis, last, last + w)[...] = got[self.rank + 1]
        return buffer

    def gather_partials(self, xp, local, *, name: str = "partial"):
        """All-gather a partial spectral sum and add in ascending rank order."""
        if self.world == 1:
            return local
        parts = self.transport.all_gather_arrays(
            self.tag(name), xp, local, [local.shape] * self.world, local.dtype)
        return sum_in_rank_order(parts)

    def fold_associative(self, xp, value, op: str, *, name: str = "fold"):
        """Fold an exactly associative reduction across the cards.

        Minimum, maximum, all and any are exact in any order, so the only
        thing this owes the answer is that every rank sees every rank's
        value; it still folds in ascending rank order so a receipt and a
        rerun read the same.
        """
        if self.world == 1:
            return value
        host = np.asarray(_to_numpy(xp, value))
        tag = self.tag(name)
        payload = host.tobytes()
        self.transport.post(tag, payload)
        pieces = self.transport.collect(tag)
        folded = host
        for rank, piece in enumerate(pieces):
            if rank == self.rank or piece is None:
                continue
            other = np.frombuffer(piece, dtype=host.dtype).reshape(host.shape)
            if op == "min":
                folded = np.minimum(folded, other)
            elif op == "max":
                folded = np.maximum(folded, other)
            elif op == "all":
                folded = folded & other
            elif op == "any":
                folded = folded | other
            else:
                raise ValueError(
                    f"{op!r} is not exactly associative across cards; a sum "
                    "must go through gather_partials in rank order"
                )
        return xp.asarray(folded) if xp is not np else folded

    def gather_records(self, record, *, name: str = "records") -> list:
        """All-gather one small host record per rank, in rank order.

        For the readings a band produces that are not arrays: the physics
        suite's per-band diagnostics and namespace metadata, which the
        suite's ``finish`` merges by rule over EVERY band of the call.  A
        card runs only its own bands, so the other cards' records have to
        reach it before the merge, and they have to arrive as the values
        the other card held, type for type: a float32 reading that came
        back a Python float, or a tuple that came back a list, would make
        a merged reading or a checkpointed metadata record differ from the
        one-card run's (gate BIT-6).  :func:`encode_record` is exact in
        that sense and refuses a value it cannot carry exactly.

        The bytes cross through the transport's own post and collect, so
        any transport that moves bytes moves these.  The ranks' records
        differ in length, and a collective all-gather (NCCL's) moves one
        length from every rank, so the lengths cross first and every
        payload is padded to the longest; each piece is cut back to its
        own length before it is decoded.  This rank's own entry is the
        record it was handed, not a decoded copy.
        """
        if self.world == 1:
            return [record]
        payload = encode_record(record)
        tag = self.tag(f"{name}_length")
        self.transport.post(tag, struct.pack("!Q", len(payload)))
        lengths = [
            len(payload) if rank == self.rank or piece is None
            else struct.unpack("!Q", bytes(piece))[0]
            for rank, piece in enumerate(self.transport.collect(tag))
        ]
        tag = self.tag(name)
        self.transport.post(tag, payload + b" " * (max(lengths) - len(payload)))
        pieces = self.transport.collect(tag)
        out = []
        for rank, piece in enumerate(pieces):
            if rank == self.rank or piece is None:
                out.append(record)
                continue
            out.append(decode_record(bytes(piece)[: lengths[rank]]))
        return out

    def mark_step(self) -> None:
        self.transport.ledger.mark_step()

    # -- the receipt --------------------------------------------------
    def wire_receipt(self) -> dict[str, object]:
        """The wire ledger, with every posted frame accounted for."""
        return self.transport.ledger.receipt()

    def receipt(self) -> dict[str, object]:
        row = dict(self.transport.receipt())
        row.update(
            {
                "card_exchange": self.exchange,
                "cards": int(self.world),
                "rank": int(self.rank),
                "band_owners": list(self.owners),
                "card_weights": list(self.weights),
                "band_profile_ms": None if self.band_ms is None else list(self.band_ms),
                "local_rows": list(self.local_rows()),
                "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
                "card_axis": "band",
                "card_devices": list(self.devices),
                "card_placement": dict(self.placement),
                "card_agreement": self.agreement.receipt(),
                "wire": self.wire_receipt(),
            }
        )
        return row

    def close(self) -> None:
        """Every rank reaches the same point, then the sockets go down.

        The barrier is what makes a clean finish look clean: without it
        the first rank to finish tears its sockets down while the other is
        still posting, and a run that completed reports a dead rank.
        """
        if self.world > 1:
            try:
                self.transport.barrier(self.tag("close"))
            except Exception:  # noqa: BLE001 - a failed run still closes
                pass
        self.transport.close()


def gather_named_arrays(session, arrays, nlat: int, nlon: int) -> dict:
    """ Assemble the whole-globe form of every grid array, for the checkpoint.

    A multi-card run partitions grid space, so each rank holds the rows of
    the bands it owns and nothing else; the checkpoint is the whole state
    and its per-array hashes are what gate BIT-5 is defined on, so the
    rows have to come together before anything is hashed.  Spectral
    arrays are already whole on every rank and are passed through
    untouched.

    An array that carries the latitude axis in a shape this does not
    recognise is REFUSED by name rather than written with the rows this
    rank happens to hold: a stale row in a checkpoint is a wrong restart,
    and a wrong restart that hashes cleanly is worse than a crash.
    """
    import numpy as _np

    if session is None or session.world <= 1:
        return arrays
    out = {}
    for name, value in arrays.items():
        array = _np.array(value, copy=True)
        shape = tuple(array.shape)
        if len(shape) >= 2 and shape[-2] == nlat and shape[-1] == nlon:
            axis = array.ndim - 2
        elif len(shape) >= 1 and shape[-1] == nlat:
            axis = array.ndim - 1
        elif nlat in shape:
            raise ValueError(
                f"checkpoint array {name!r} of shape {shape} carries the "
                f"latitude axis ({nlat} rows) somewhere this gather does not "
                "recognise, so a multi-card run would write the rows this "
                "card happens to hold and hash them as the globe.  Teach the "
                "gather the layout, or keep the array off the checkpoint"
            )
        else:
            out[name] = array
            continue
        session.gather_rows(_np, array, axis, name=f"checkpoint_{name}")
        out[name] = array
    return out


def single_card_session(nlat: int, bands: int) -> CardSession:
    """The session a one-card run holds: real object, no wire."""
    return CardSession(SingleCard(), nlat, bands)


#: The synthesis stack widths the card-agreement probe exercises.  Lane 6
#: MEASURED 2026-09-06 that the Legendre synthesis returns different bits on
#: an RTX 5070 Ti and an RTX 5090 for stack widths 1, 2, 4 and 8 at T127 and
#: T85, and the same bits at 40, because cuBLAS picks its kernel by the
#: GEMM's M -- the surface pressure is the width-1 case, so a run that
#: presents it is the one that diverges.  These are the widths a gather run
#: must find the cards agree on before it assembles a waist from their rows.
def _assert_card_hashes_agree(key, hashes) -> None:
    """Refuse a gather two-card run whose cards do not return the same bits.

    ``key`` names the contraction (its direction and the operand shape) and
    ``hashes`` is one digest per rank of that contraction's output.  A
    gather run assembles a Fourier waist out of latitude rows computed on
    two different cards, so it reproduces the single-card answer ONLY if
    the cards return the same bits for the same work (lane 6's finding, the
    precondition the design did not state).  This is that precondition
    made into a refusal: a pair that disagrees is stopped by name at the
    first contraction they disagree on, before the waist it feeds is
    assembled from both cards' rows.
    """
    first = hashes[0]
    if any(h != first for h in hashes[1:]):
        direction, shape = key[0], key[-2]
        raise ValueError(
            "the cards in this two-card gather run do not return the same "
            f"bits for the Legendre {direction} at operand shape {shape}: "
            + ", ".join(f"rank {r}: {h[:12]}" for r, h in enumerate(hashes))
            + ".  A gather run assembles a waist from rows computed on "
            "both cards, so it reproduces the single-card answer only "
            "where the cards agree, and agreement is a property of the "
            "operand shapes the configuration presents, not of the "
            "truncation (an RTX 5070 Ti and an RTX 5090 agree on every "
            "contraction of a T255 L40 step and disagree on an eight-plane "
            "synthesis at the same truncation, MEASURED 2026-09-06 and "
            "2026-09-07).  Run one card, or a pair that agrees on this "
            "configuration"
        )


class ContractionAgreement:
    """The cross-card agreement check, taken on the run's OWN contractions.

    A gather run reproduces one card only where the cards return the same
    bits for the same work, and which shapes agree is a property of the
    operand shapes the configuration presents (lane 6).  A probe at fixed
    widths therefore refuses pairs the run would reproduce (an eight-plane
    synthesis at T255 differs between an RTX 5070 Ti and an RTX 5090 while
    every contraction of the T255 L40 step agrees, MEASURED 2026-09-07) and
    passes pairs it would not.  So the check is taken on the contractions
    themselves: the first time a (direction, table, shape, dtype) is
    contracted, every rank hashes its output and the ranks all-gather the
    digests.  The analysis contracts an assembled waist that is identical
    on every rank and the synthesis contracts the replicated spectral
    state, so the outputs MUST agree, and one disagreement is the run
    refused by name.  Every later contraction of that shape is the same
    kernel on the same card, so it is checked once; the cost is one host
    hash per distinct shape per run.
    """

    def __init__(self, session, mode: str = "refuse"):
        self.session = session
        self.mode = str(mode).strip().lower()
        if self.mode not in ("refuse", "record"):
            raise ValueError(
                f"card_agreement must be 'refuse' or 'record', got {mode!r}")
        self.agreed: dict = {}
        #: The shapes the cards disagreed on, kept only under "record":
        #: the run carries on, the receipt lists them, and the run's
        #: two_card_contractions_agree gate row fails on their count.
        self.disagreed: dict = {}
        self._count = 0

    def check(self, xp, key, array) -> None:
        if key in self.agreed or key in self.disagreed:
            return
        import hashlib

        host = np.ascontiguousarray(_to_host_array(xp, array))
        digest = hashlib.sha256(
            host.dtype.str.encode() + str(host.shape).encode() + host.tobytes()
        ).hexdigest()
        n = self._count
        self._count += 1
        pieces = self.session.transport.all_gather(
            self.session.tag(f"card-agree-{n}"), digest.encode()
        )
        hashes = [piece.decode() for piece in pieces]
        if self.mode == "record" and any(h != hashes[0] for h in hashes[1:]):
            self.disagreed[key] = hashes
            return
        _assert_card_hashes_agree(key, hashes)
        self.agreed[key] = hashes

    def receipt(self) -> dict:
        return {
            "mode": self.mode,
            "checked": len(self.agreed) + len(self.disagreed),
            "contractions": [
                {"direction": key[0], "table": key[1], "shape": list(key[2]),
                 "dtype": key[3], "hash": hashes[0][:16]}
                for key, hashes in self.agreed.items()
            ],
            "disagreements": [
                {"direction": key[0], "table": key[1], "shape": list(key[2]),
                 "dtype": key[3],
                 "hashes": [h[:16] for h in hashes]}
                for key, hashes in self.disagreed.items()
            ],
            "agree": not self.disagreed,
        }


def _to_host_array(xp, array):
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


# ---------------------------------------------------------------------
# The row exchange the waist and the accumulators call
# ---------------------------------------------------------------------


class RowExchange:
    """The hook a Fourier waist and a reduction buffer call to fill in the
    rows this rank did not compute.

    Deliberately a tiny duck-typed object rather than an import:
    ``arwen_global.spectral`` is the layer below ``arwen_global``
    and must not learn that cards exist.  The transform holds one of these
    or holds ``None``, and the ``None`` case is the shipped single-card
    run with no branch taken and no cost.
    """

    def __init__(self, session: CardSession):
        self.session = session

    @property
    def world(self) -> int:
        return self.session.world

    def owned_rows(self) -> tuple[int, int]:
        return self.session.local_rows()

    def fill_rows(self, xp, buffer, axis: int, *, name: str = "rows"):
        return self.session.gather_rows(xp, buffer, axis, name=name)

    def fold(self, xp, value, op: str, *, name: str = "fold"):
        return self.session.fold_associative(xp, value, op, name=name)

    def halo(self, xp, buffer, axis: int, width: int, *, name: str = "halo"):
        return self.session.exchange_halo(xp, buffer, axis, width, name=name)

    def gather_partials(self, xp, local, *, name: str = "partial"):
        return self.session.gather_partials(xp, local, name=name)

    def gather_records(self, record, *, name: str = "records") -> list:
        return self.session.gather_records(record, name=name)

    def agree_once(self, xp, key, array) -> None:
        """The transform hands every contraction's output here once per
        distinct shape; the ranks must agree on it or the run is refused."""
        self.session.agreement.check(xp, key, array)


class OrderExchange:
    """The order-m multi-card axis: the hook the transform's contraction
    calls to gather the coefficient and waist columns of the orders this
    rank did not contract.

    Dual to :class:`RowExchange`.  The band (latitude) axis partitions GRID
    space and gathers waist ROWS; this axis partitions the Legendre ORDERS
    and gathers coefficient/waist COLUMNS.  Which one a two-card run uses
    is a truncation question, not a preference:

    * the band axis splits grid-space work -- the physics, the FFT, the
      transport -- and duplicates the whole-width Legendre contraction, so
      it is the SPEED axis for every truncation whose Legendre table fits
      one card;
    * the order axis splits the Legendre TABLE and the contraction and
      duplicates grid-space work, so it is the CAPACITY axis for a
      truncation whose whole table no longer fits one card (design section
      10: the only path past the table wall).

    The orders are independent output indices of the contraction and are
    never reduced across the wire -- the ranks own disjoint order ranges
    and the assembled column set is their concatenation -- so an order-split
    analysis or synthesis returns the single-card bits by construction.
    The one reduction over orders, the inverse longitude FFT, runs WHOLE on
    every rank after the gather, identically.
    """

    def __init__(self, session: CardSession, order_bounds, owners):
        self.session = session
        self.bounds = [(int(a), int(b)) for a, b in order_bounds]
        self.owners = tuple(int(r) for r in owners)
        if len(self.owners) != len(self.bounds):
            raise ValueError(
                f"{len(self.owners)} owners for {len(self.bounds)} order bands"
            )
        self.rank = session.rank
        self.world = session.world

    def _ranges(self) -> list[tuple[int, int]]:
        """``[(m_lo, m_hi), ...]`` order range each rank owns, rank order."""
        out = []
        for r in range(self.world):
            held = [k for k, o in enumerate(self.owners) if o == r]
            if not held:
                raise ValueError(
                    f"order rank {r} owns no band: an idle rank on the order "
                    "axis holds no table and computes nothing"
                )
            out.append((self.bounds[held[0]][0], self.bounds[held[-1]][1]))
        return out

    def owned_bounds(self) -> tuple:
        """The ``(m0, m1)`` bands this rank owns, for the transform build."""
        return tuple(
            self.bounds[k] for k, o in enumerate(self.owners) if o == self.rank
        )

    @property
    def owned_order_range(self) -> tuple[int, int]:
        return self._ranges()[self.rank]

    def gather_columns(self, xp, compact, width, *, name="orders"):
        """Assemble ``(*lead, X, width)`` from every rank's owned columns.

        ``compact`` is this rank's ``(*lead, X, m_hi - m_lo)``.  Each rank's
        owned order range is a pure function of the schedule, so a rank
        reconstructs the shape of every piece it receives; the pieces are
        disjoint order ranges and are PLACED, never added, so the assembled
        column set is the whole spectrum a single card produced.
        """
        if self.world == 1:
            return compact
        ranges = self._ranges()
        lead = tuple(compact.shape[:-2])
        x = compact.shape[-2]
        dtype = compact.dtype
        shapes = [(*lead, x, m_hi - m_lo) for m_lo, m_hi in ranges]
        pieces = self.session.transport.all_gather_arrays(
            self.session.tag(name), xp, compact, shapes, dtype)
        full = xp.empty((*lead, x, int(width)), dtype=dtype)
        for r, piece in enumerate(pieces):
            m_lo, m_hi = ranges[r]
            full[..., :, m_lo:m_hi] = piece
        return full


def order_exchange_for(session: CardSession, transform, weights=None) -> "OrderExchange":
    """The :class:`OrderExchange` for a transform, with the band-to-rank
    assignment the order axis runs under.

    The schedule is the transform's own Legendre band schedule; the owners
    follow the measured per-card throughput when ``weights`` is given and
    split the bands evenly otherwise.  Whole bands only, contiguous per
    rank, refuse an idle rank -- :func:`order_band_owners`.
    """
    from arwen_global.spectral.legendre import band_bounds

    bounds = band_bounds(transform.grid.truncation, transform.legendre_band)
    if session.world == 1:
        owners = tuple(0 for _ in bounds)
    else:
        w = tuple(weights) if weights else tuple(1.0 for _ in range(session.world))
        owners = order_band_owners(bounds, w)
    return OrderExchange(session, bounds, owners)


# ---------------------------------------------------------------------
# host staging
# ---------------------------------------------------------------------


def _slice_rows(buffer, axis: int, first: int, last: int):
    index = [slice(None)] * buffer.ndim
    index[axis] = slice(int(first), int(last))
    return buffer[tuple(index)]


def _rows_shape(buffer, axis: int, rows: int) -> tuple:
    shape = list(buffer.shape)
    shape[axis] = int(rows)
    return tuple(shape)


def _to_numpy(xp, value):
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def _to_host_bytes(xp, value) -> bytes:
    host = _to_numpy(xp, value)
    return np.ascontiguousarray(host).tobytes()


def _array_from_host_bytes(xp, payload: bytes, shape, dtype):
    host = np.frombuffer(payload, dtype=dtype).reshape(shape)
    return host if xp is np else xp.asarray(host)


def _from_host_bytes(xp, payload: bytes, target) -> None:
    host = np.frombuffer(payload, dtype=_dtype_of(target)).reshape(target.shape)
    target[...] = host if xp is np else xp.asarray(host)


def _dtype_of(array):
    return np.dtype(array.dtype)


# ---------------------------------------------------------------------
# small host records
# ---------------------------------------------------------------------


def _record_tree(value):
    """``value`` as a JSON tree that names every type it came in."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return {"np": np.dtype(value.dtype).str,
                "b": np.asarray(value).tobytes().hex()}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"nd": array.dtype.str, "s": list(array.shape),
                "b": array.tobytes().hex()}
    if isinstance(value, int):
        return {"i": str(int(value))}
    if isinstance(value, float):
        return {"f": float(value).hex()}
    if isinstance(value, tuple):
        return {"t": [_record_tree(v) for v in value]}
    if isinstance(value, list):
        return {"l": [_record_tree(v) for v in value]}
    if isinstance(value, dict):
        return {"d": [[_record_tree(k), _record_tree(v)]
                      for k, v in value.items()]}
    raise TypeError(
        f"a {type(value).__name__} cannot cross between cards as a record: "
        "only None, bool, int, float, str, tuple, list, dict and numpy "
        "scalars and arrays are carried exactly, and a reading that came "
        "back as a different type would make the merged reading differ "
        "from the one-card run's"
    )


def _record_value(tree):
    if tree is None or isinstance(tree, (bool, str)):
        return tree
    if "np" in tree:
        return np.frombuffer(bytes.fromhex(tree["b"]), dtype=np.dtype(tree["np"]))[0]
    if "nd" in tree:
        return np.frombuffer(
            bytes.fromhex(tree["b"]), dtype=np.dtype(tree["nd"])
        ).reshape(tree["s"]).copy()
    if "i" in tree:
        return int(tree["i"])
    if "f" in tree:
        return float.fromhex(tree["f"])
    if "t" in tree:
        return tuple(_record_value(v) for v in tree["t"])
    if "l" in tree:
        return [_record_value(v) for v in tree["l"]]
    if "d" in tree:
        return {_record_value(k): _record_value(v) for k, v in tree["d"]}
    raise ValueError(f"not a card record: {sorted(tree)}")


def encode_record(value) -> bytes:
    """Bytes for a small host record, exact in value and in type.

    JSON with every number and container tagged by its type, rather than
    pickle: a rank's socket is a network listener, and a payload decoded
    by pickle can run code, while this one can only build the types
    above.  Floats travel as their hexadecimal form, so every bit
    survives, NaN and negative zero included.
    """
    import json

    return json.dumps(_record_tree(value), separators=(",", ":")).encode()


def decode_record(payload: bytes):
    """The record :func:`encode_record` was handed."""
    import json

    return _record_value(json.loads(bytes(payload).decode()))


__all__ = [
    "CHUNK_BYTES",
    "CardSession",
    "CardTransport",
    "ContractionAgreement",
    "EXCHANGE_MODES",
    "NcclCards",
    "OrderExchange",
    "RowExchange",
    "SingleCard",
    "TcpCards",
    "WIRE_FLOOR_BYTES_S",
    "WireLedger",
    "band_owners",
    "card_identity",
    "check_card_placement",
    "decode_record",
    "encode_record",
    "launch_environment",
    "local_rank_of",
    "open_transport",
    "order_band_owners",
    "order_exchange_for",
    "place_rank",
    "single_card_session",
    "sum_in_rank_order",
    "throughput_weights",
]
