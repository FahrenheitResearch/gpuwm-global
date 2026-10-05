"""Three-dimensional interpolation at semi-Lagrangian departure points.

One :class:`Stencil` describes where every arrival point came from; every
advected field is then read at those points through the same index and weight
computation.  The stencil carries the departure COORDINATES rather than
twelve precomputed weights per point, for two reasons and both are measured:
twelve float32 weights plus three base indices is 566 MiB at T255 and 2.46 GiB
at T533 against 141 MiB and 616 MiB for three coordinate arrays, and the
kernel that made the cost model computes the weights once per point and reuses
them across all fourteen fields inside one launch anyway, so writing them to
memory would be the only time they were ever stored.

Directions, and what each one's weights are:

*   Zonal.  ``lon[i] = i*2*pi/nlon`` is uniform, so the weights are the
    equispaced cubic Lagrange weights and the stencil wraps modulo nlon.  It
    is never clamped: there is no zonal boundary.
*   Meridional.  The latitudes are Gauss-Legendre nodes and are NOT uniform,
    so the weights are the general four-point Lagrange weights on the actual
    node latitudes, built from a precomputed table of reciprocal
    denominators.  Treating them as equispaced is exact for no cubic at all
    and shows up first as a hemispherically antisymmetric error, which is
    why KERN-1 runs at a rotation angle of pi/2.  A departure point poleward
    of the outermost ring reads the reflected rows described in
    :mod:`arwen_global.semilag.tables`.
*   Vertical.  Four-point cubic Lagrange in the continuous level index, full
    level ``k`` at index ``k``, with the departure index clamped to the
    model's boundary INTERFACES ``[-1/2, nlev-1/2]`` and the stencil start
    clamped to ``[0, nlev-4]``, so the four levels nearest a boundary use
    the one-sided cubic through them.  The clamp names its breakage: a
    departure point above the lid interface has left the model, and
    extrapolating beyond it would make unbounded values exactly where the
    top sponge already has a rigid-lid reflection to fight.  The half
    layer between the outermost full level and its interface is real
    model air, though: the level-0 rate is half the first interior
    interface rate, so a parcel arriving at level 0 under descent departs
    from inside it.  Clamping that point to level 0, as this gather did
    until DYC-1, read the parcel's own value under descent and the colder
    level below under ascent, a first-order cooling of the lid of about
    -0.12 K per step per 0.001 levels of displacement on the default
    stack.  Inside the half layer the unlimited fields are held between
    the outermost level's value and the linear reconstruction at the
    interface (the column's own end slope), so the extrapolation creates
    no extremum that slope does not imply; the limited fields keep their
    cell box.

All three weight sets are normalized by their own sum.  That is what makes
the zero-displacement identity exact, at every point of the grid including
the polar rows and the vertical boundaries: at a grid point three of the four
Lagrange weights are exactly zero and the fourth is divided by itself.

Two horizontal widths.  ``order = 4`` is the tricubic gather above, the one
every kernel gate of record was measured on, and its arithmetic does not
move.  ``order = 6`` is the QUINTIC-horizontal gather: six-point Lagrange
weights in the zonal and meridional directions on the six-point tables
(three reflected rows beyond each pole) and the same four-point cubic in the
vertical.  It exists because an interpolation is a filter applied once per
step whatever the step is: MEASURED 2026-09-06 on the T255 forecast day at
dt = 300 s, the cubic gather kept 17 percent of the Eulerian core's 500 hPa
vorticity power at total wavenumbers 181 to 230 and 49 to 71 percent above
n = 60, and the four-point weights at a fifth of a cell lose 7.6 percent of
a 3.8-cell wave's amplitude per pass.  The six-point weights halve that
loss at the same cost in trajectory and tables and 2.25 times the taps.
The dynamical bundle reads it; the limited species keep the cubic, whose
limiter box is the same inner cell either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .tables import BandWindow, HaloEscape, SphericalGridTables
from ..spill import resident

#: Fields per launch.  Batching was measured free in time and worth 1.9 GiB
#: of transient at T533, because one index and weight computation is shared.
DEFAULT_BATCH = 16

#: Horizontal stencil widths the gather compiles: four-point cubic and
#: six-point quintic Lagrange.  The vertical is four-point under both.
ORDERS = (4, 6)


@dataclass(frozen=True)
class Stencil:
    """Departure coordinates of every arrival point of one grid."""

    #: departure position in CONTINUOUS ZONAL INDEX, any branch (the gather
    #: wraps).  An index rather than a longitude because the zonal nodes are
    #: uniform, so the index is the natural coordinate of the direction and,
    #: unlike ``lam/dlam`` computed inside the gather, it is exactly integral
    #: at a grid point.  That is what makes the zero-displacement identity
    #: exact instead of correct to a few ulp.
    xi: Any
    #: departure latitude, radians, clamped into [-pi/2, pi/2] by the gather
    phi: Any
    #: departure continuous full-level index
    level: Any
    tables: SphericalGridTables
    #: The latitude band these arrival points are, and the rows its source
    #: fields hold (tables.BandWindow), or None for the whole grid.  A
    #: windowed stencil covers the band's rows only and is read against
    #: fields that hold the band and its halo.
    window: BandWindow | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.xi.shape)  # type: ignore[return-value]

    @property
    def source_shape(self) -> tuple[int, int, int]:
        """The shape of a field this stencil reads."""
        nlev = self.shape[0]
        if self.window is None:
            return self.shape
        return self.window.source_shape(nlev)

    def __post_init__(self) -> None:
        shape = tuple(int(s) for s in self.xi.shape)
        if len(shape) != 3:
            raise ValueError(
                f"departure coordinates must be (nlev, nlat, nlon), got {shape}"
            )
        expected = (self.tables.shape if self.window is None
                    else (self.window.arrival_rows, self.tables.nlon))
        if shape[1:] != expected:
            raise ValueError(
                f"departure coordinates are {shape[1:]} where "
                f"{expected} were expected on a {self.tables.shape} grid"
            )
        if shape[0] < 4:
            raise ValueError(
                f"a four-point vertical stencil needs nlev >= 4, got {shape[0]}"
            )
        for name in ("phi", "level"):
            other = tuple(int(s) for s in getattr(self, name).shape)
            if other != shape:
                raise ValueError(
                    f"{name} is {other} where xi is {shape}"
                )


def zero_stencil(tables: SphericalGridTables, nlev: int, *, xp, dtype) -> Stencil:
    """The identity stencil: every arrival point departs from itself."""
    nlat, nlon = tables.shape
    lat = xp.asarray(tables.lat_ext_host[2:nlat + 2], dtype=dtype)
    index = xp.arange(nlon, dtype=dtype)
    xi = xp.broadcast_to(index[None, None, :], (nlev, nlat, nlon))
    phi = xp.broadcast_to(lat[None, :, None], (nlev, nlat, nlon))
    lev = xp.broadcast_to(
        xp.arange(nlev, dtype=dtype)[:, None, None], (nlev, nlat, nlon)
    )
    return Stencil(
        xi=xp.ascontiguousarray(xi),
        phi=xp.ascontiguousarray(phi),
        level=xp.ascontiguousarray(lev),
        tables=tables,
    )


def _is_cupy(xp) -> bool:
    return getattr(xp, "__name__", "") == "cupy"


#: The limiter shapes the gather compiles, and the bound each one holds.
#:
#: ``quasi_monotone``  [cell minimum, cell maximum].  No new extremum of
#:                     either sign, and nonnegative wherever the source
#:                     was, which is what a condensate species needs.
#: ``none``            no bound at all.  The counter-arm: MEASURED
#:                     2026-09-06, a six-hour T255 native run with no
#:                     limiter and no mass fixer was refused at its FIRST
#:                     step on a cloud water of -1.49e-5 against a maximum
#:                     of 1.02e-3.
#:
#: A third shape, [0, cell maximum], was built and MEASURED on 2026-09-06
#: and is not here.  On a condensate-shaped field it returned the same
#: array as the quasi-monotone shape at every point, because every
#: undershoot that shape lifts is already below zero on a field that is
#: zero nearly everywhere; on a field with a positive floor its mass error
#: was 2.15e-3 against the quasi-monotone shape's 8.04e-5.  A door with a
#: knob that is identical where it would help and 27 times worse where it
#: would not is a worse door.
LIMITERS = ("quasi_monotone", "none")

_LIMITER_SUFFIX = {"quasi_monotone": "qm"}


def _entry_point(dtype, limiter: str, deficit: bool = False,
                 order: int = 4, band: bool = False) -> str:
    kind = np.dtype(dtype)
    if kind == np.float32:
        suffix = "f32"
    elif kind == np.float64:
        suffix = "f64"
    else:
        raise ValueError(
            f"the semi-Lagrangian gather is compiled for float32 and float64 "
            f"only, got {kind}"
        )
    if limiter not in LIMITERS:
        raise ValueError(
            "the semi-Lagrangian gather's limiter must be one of "
            + ", ".join(repr(item) for item in LIMITERS)
            + f", got {limiter!r}"
        )
    if int(order) not in ORDERS:
        raise ValueError(
            "the semi-Lagrangian gather's horizontal order must be one of "
            + ", ".join(str(item) for item in ORDERS) + f", got {order!r}"
        )
    family = "sl_gather" if int(order) == 4 else "sl_gather5"
    if limiter == "none":
        if deficit:
            raise ValueError(
                "the clip deficit is the mass the limiter moved, so an "
                "unlimited gather has none to report; ask for a limiter or "
                "do not ask for the deficit"
            )
        return f"{family}{_band_tag(band)}_{suffix}"
    tag = _LIMITER_SUFFIX[limiter]
    if deficit:
        if int(order) != 4:
            raise ValueError(
                "the clip deficit is reported by the cubic gather only: the "
                "additive mass fixer reads it for the limited species, which "
                "ride the cubic stencil, and a quintic deficit would be a "
                "number nothing consumes"
            )
        return f"{family}_{tag}d{_band_tag(band)}_{suffix}"
    return f"{family}_{tag}{_band_tag(band)}_{suffix}"


def _band_tag(band: bool) -> str:
    """The entry-point infix of the latitude-band form (kernels.cu, BAND)."""
    return "_band" if band else ""


def _limiter_of(monotone) -> str:
    """The limiter a caller asked for, from either spelling.

    ``monotone`` accepts the two booleans it always did and, now, any
    name in :data:`LIMITERS`.  A boolean is not enough to say WHICH bound
    is wanted and the shapes differ by which mass they move, so the string
    is the spelling the model uses and the boolean is kept because the
    kernel gates and the case drivers read it that way.
    """
    if isinstance(monotone, str):
        return monotone
    return "quasi_monotone" if monotone else "none"


def gather_batch(
    fields: Sequence[Any],
    stencil: Stencil,
    *,
    monotone: Any = True,
    batch: int = DEFAULT_BATCH,
    deficit: bool = False,
    order: int = 4,
    defer_escape: bool = False,
) -> list:
    """Read every field of ``fields`` at the stencil's departure points.

    ``defer_escape`` leaves a band window's escape flag for the caller to
    check once after several launches (tables.BandWindow.raise_if_escaped)
    instead of reading it back after this one.

    ``monotone`` selects the limiter: ``True``/``False`` for the
    quasi-monotone shape and none, or a name from :data:`LIMITERS`.
    ``order`` selects the horizontal stencil width, 4 (cubic) or 6
    (quintic); the vertical is four-point cubic under both.

    With ``deficit=True`` (which needs a limiter) the return is a list of
    ``(value, clip_deficit)`` pairs instead of a list of values.
    The deficit is the signed mass the limiter moved at each point, raw
    minus kept: positive where it cut an overshoot away and negative where
    it lifted an undershoot.  The interpolated values are bit for bit the
    ones the plain quasi-monotone gather returns; the second array is the
    difference that gather computes and discards.
    """
    if not fields:
        return []
    tables = stencil.tables
    xp = tables.xp
    window = stencil.window
    source_shape = stencil.source_shape
    dtype = np.dtype(tables.dtype)
    for index, field in enumerate(fields):
        got = tuple(int(s) for s in field.shape)
        if got != source_shape:
            raise ValueError(
                f"field {index} is {got} where the stencil reads "
                f"{source_shape}"
            )
        if np.dtype(field.dtype) != dtype:
            raise ValueError(
                f"field {index} is {np.dtype(field.dtype)} where the grid "
                f"tables were built for {dtype}; a mixed-precision gather "
                f"would read the wrong bytes, not the wrong answer"
            )
    if int(batch) < 1:
        raise ValueError("batch must be >= 1")
    limiter = _limiter_of(monotone)
    order = int(order)
    # Validates the pairing on every backend, naming the reason.
    _entry_point(dtype, limiter, deficit, order)
    if order == 6 and tables.lat_ext6 is None:
        raise ValueError(
            f"the quintic gather needs six meridional rows and this grid has "
            f"{tables.nlat}; the six-point tables were not built"
        )
    if not _is_cupy(xp):
        # A parked field on the host backend IS its pinned host array.
        return _gather_numpy([resident(xp, field) for field in fields],
                             stencil, limiter=limiter,
                             deficit=deficit, order=order)

    nlev, nrows, nlon = stencil.shape
    nlat = tables.nlat
    kernel = _kernel_for(dtype, limiter, deficit, order, window is not None)
    npoints = nlev * nrows * nlon
    threads = 256
    blocks = (npoints + threads - 1) // threads
    scalar = dtype.type
    out: list = []
    if order == 6:
        rowoff = tables.rowoff6 if window is None else window.rowoff6
        geometry = (tables.lat_ext6, tables.mrden6, tables.mlut6,
                    rowoff, tables.rowshift6)
        bins, scale = tables.lookup_bins6, tables.lut_scale6
    else:
        rowoff = tables.rowoff if window is None else window.rowoff
        geometry = (tables.lat_ext, tables.mrden, tables.mlut,
                    rowoff, tables.rowshift)
        bins, scale = tables.lookup_bins, tables.lut_scale
    if window is not None and not defer_escape:
        window.clear_escape()
    escape = None if window is None else window.escape
    for start in range(0, len(fields), int(batch)):
        # A field the pinned host tier holds is staged HERE, one batch at
        # a time, and the staged copy is dropped with the batch's stack:
        # the tier's credit for a parked tracer is the original not
        # standing on the card between steps, and a gather that staged
        # all ten at once would put the whole slice back for the step
        # (the ten grid tracers at T533 are 1.9 GiB, and a gather that
        # staged them all before its first batch held them beside its own
        # two stacks on a 16 GB card: MEASURED 2026-09-07, RTX 5070 Ti,
        # T533 L40 with every slice parked, out of memory at 15.96 GB
        # inside this loop). A resident field passes through untouched
        # (spill.resident is the identity for one already on the card).
        chunk = [resident(xp, field) for field in fields[start:start + int(batch)]]
        count = len(chunk)
        src = xp.ascontiguousarray(xp.stack(chunk, axis=0))
        del chunk
        dst = xp.empty((count, *stencil.shape), dtype=src.dtype)
        tail = (
            np.int32(count), np.int32(nlev), np.int32(nlat),
            np.int32(nlon), np.int32(bins), scalar(scale),
        )
        if window is not None:
            tail = tail + (np.int32(window.source_rows),
                           np.int32(window.arrival_rows), escape)
        head = (src, stencil.xi, stencil.phi, stencil.level,
                *geometry, dst)
        if deficit:
            cut = xp.empty_like(dst)
            kernel((blocks,), (threads,), (*head, cut, *tail))
            out.extend((dst[index], cut[index])
                       for index in range(count))
        else:
            kernel((blocks,), (threads,), (*head, *tail))
            out.extend(dst[index] for index in range(count))
    if window is not None and not defer_escape:
        window.raise_if_escaped("gather")
    return out


def gather_linear_batch(fields: Sequence[Any], stencil: Stencil, *,
                        batch: int = DEFAULT_BATCH) -> list:
    """Every field of ``fields`` read TRILINEARLY at the departure points.

    The low-order interpolant inside the cubic gather's own limiter cell:
    the same two columns, the same bracketing pair of (extended) rows and
    the same clamped pair of levels the quasi-monotone box is built from,
    so the value lies in that box, is nonnegative wherever the source is,
    and is the field itself at a zero displacement.  It exists for the
    Bermejo-Conde mass fixer, whose weight is the distance between the
    high-order and this low-order value (``tracers.fix_mass``).
    """
    if not fields:
        return []
    if stencil.window is not None:
        # The trilinear kernel addresses rows through the whole grid's
        # offsets; a band's stencil read against them would sample the
        # wrong rows.  The banded step assembles the whole stencil before
        # the fixer reads it (semilag.step._semilag_step_banded).
        raise ValueError(
            "the trilinear gather reads whole-grid fields at a whole-grid "
            "stencil; a latitude band's stencil would address the wrong rows"
        )
    tables = stencil.tables
    xp = tables.xp
    shape = stencil.shape
    dtype = np.dtype(tables.dtype)
    if dtype == np.float32:
        name = "sl_linear_f32"
    elif dtype == np.float64:
        name = "sl_linear_f64"
    else:
        raise ValueError(
            f"the semi-Lagrangian gather is compiled for float32 and float64 "
            f"only, got {dtype}"
        )
    for index, field in enumerate(fields):
        got = tuple(int(s) for s in field.shape)
        if got != shape:
            raise ValueError(
                f"field {index} is {got} where the stencil is {shape}"
            )
        if np.dtype(field.dtype) != dtype:
            raise ValueError(
                f"field {index} is {np.dtype(field.dtype)} where the grid "
                f"tables were built for {dtype}; a mixed-precision gather "
                f"would read the wrong bytes, not the wrong answer"
            )
    if int(batch) < 1:
        raise ValueError("batch must be >= 1")
    if not _is_cupy(xp):
        return _linear_numpy([resident(xp, field) for field in fields],
                             stencil)
    from ._cuda import get_kernel

    kernel = get_kernel(name)
    nlev, nlat, nlon = shape
    npoints = nlev * nlat * nlon
    threads = 256
    blocks = (npoints + threads - 1) // threads
    out: list = []
    for start in range(0, len(fields), int(batch)):
        chunk = [resident(xp, field)
                 for field in fields[start:start + int(batch)]]
        count = len(chunk)
        src = xp.ascontiguousarray(xp.stack(chunk, axis=0))
        del chunk
        dst = xp.empty_like(src)
        kernel((blocks,), (threads,), (
            src, stencil.xi, stencil.phi, stencil.level,
            tables.lat_ext, tables.mlut, tables.rowoff, tables.rowshift, dst,
            np.int32(count), np.int32(nlev), np.int32(nlat), np.int32(nlon),
            np.int32(tables.lookup_bins), dtype.type(tables.lut_scale),
        ))
        del src
        out.extend(dst[index] for index in range(count))
    return out


def _kernel_for(dtype, limiter: str, deficit: bool = False, order: int = 4,
                band: bool = False):
    from ._cuda import get_kernel

    return get_kernel(_entry_point(dtype, limiter, deficit, order, band))


def gather(field, stencil: Stencil, *, monotone: Any = True, order: int = 4):
    """One field at the stencil's departure points."""
    return gather_batch([field], stencil, monotone=monotone, order=order)[0]


# ---------------------------------------------------------------------------
# The numpy specification.
#
# Written independently of the kernel and kept as the thing the kernel is
# graded against, on the tree's standing convention that a fused device path
# is a replacement for a numpy expression that stays in place as the
# specification.  It is chunked over levels because the fancy-index form
# materializes sixteen index arrays per point, which is 189 million entries
# at T255 in one go.


def _lagrange_equispaced(t: np.ndarray) -> np.ndarray:
    a = t
    b = t - 1.0
    c = t - 2.0
    d = t - 3.0
    w = np.stack([
        -b * c * d / 6.0,
        a * c * d * 0.5,
        -a * b * d * 0.5,
        a * b * c / 6.0,
    ], axis=-1)
    return w / np.sum(w, axis=-1, keepdims=True)


def _lagrange_equispaced6(t: np.ndarray) -> np.ndarray:
    """Six-point Lagrange weights on the nodes 0..5, normalized."""
    d = [t - float(q) for q in range(6)]
    cols = []
    for m in range(6):
        prod = np.ones_like(t)
        den = 1.0
        for q in range(6):
            if q == m:
                continue
            prod = prod * d[q]
            den *= float(m - q)
        cols.append(prod / den)
    w = np.stack(cols, axis=-1)
    return w / np.sum(w, axis=-1, keepdims=True)


def _gather_numpy(fields, stencil: Stencil, *, limiter: str,
                  deficit: bool = False, order: int = 4) -> list:
    fields = [resident(np, field) for field in fields]
    tables = stencil.tables
    window = stencil.window
    nlev, nrows, nlon = stencil.shape
    nlat = tables.nlat
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    nh = int(order)
    if nh == 6:
        lat_ext = tables.lat_ext6_host.astype(dtype)
        mrden = tables.mrden6_host.astype(dtype)
        rowoff = (tables.rowoff6_host if window is None
                  else window.rowoff6_host)
        rowshift = tables.rowshift6_host
    else:
        lat_ext = tables.lat_ext_host.astype(dtype)
        mrden = tables.mrden_host.astype(dtype)
        rowoff = tables.rowoff_host if window is None else window.rowoff_host
        rowshift = tables.rowshift_host
    ghost = nh // 2 - 1
    mid = nh // 2 - 1
    half_pi = scalar(np.pi / 2.0)
    # The source plane is the rows the fields hold; the output plane is
    # the rows the stencil computes.  They are one plane on the whole grid.
    plane = (nlat if window is None else window.source_rows) * nlon
    out_plane = nrows * nlon

    xis = np.asarray(stencil.xi)
    phi = np.asarray(stencil.phi)
    lev = np.asarray(stencil.level)
    flat = [np.ascontiguousarray(np.asarray(f)).reshape(-1) for f in fields]
    outs = [np.empty((nlev, out_plane), dtype=dtype) for _ in fields]
    cuts = [np.zeros((nlev, out_plane), dtype=dtype) for _ in fields]

    for k in range(nlev):
        p = np.clip(phi[k].reshape(-1), -half_pi, half_pi)
        xk = np.clip(lev[k].reshape(-1), scalar(-0.5), scalar(nlev - 0.5))
        below_lid = xk < scalar(0.0)
        outside = below_lid | (xk > scalar(nlev - 1))
        c_edge = np.where(below_lid, 0, 3)
        c_next = np.where(below_lid, 1, 2)
        xi = xis[k].reshape(-1)

        i0 = np.floor(xi).astype(np.int64) - ghost
        if nh == 6:
            wx = _lagrange_equispaced6(xi - i0.astype(dtype))
        else:
            wx = _lagrange_equispaced(xi - i0.astype(dtype))
        cols = (i0[:, None] + np.arange(nh)) % nlon
        cols_shift = (cols + nlon // 2) % nlon

        m0 = np.clip(
            np.searchsorted(lat_ext.astype(np.float64),
                            p.astype(np.float64), side="right") - 1,
            ghost, nlat + ghost,
        )
        sst = m0 - ghost
        nodes = lat_ext[sst[:, None] + np.arange(nh)]
        d = p[:, None] - nodes
        num = np.stack([
            np.prod(np.delete(d, m, axis=1), axis=1) for m in range(nh)
        ], axis=-1)
        wy = num * mrden[sst]
        wy = wy / np.sum(wy, axis=-1, keepdims=True)

        kb = np.clip(np.floor(xk).astype(np.int64), 0, nlev - 2)
        k0 = np.clip(kb - 1, 0, nlev - 4)
        kbox = kb - k0
        wz = _lagrange_equispaced(xk - k0.astype(dtype))

        offsets = rowoff[sst[:, None] + np.arange(nh)]
        if window is not None and bool((offsets < 0).any()):
            raise HaloEscape(window, "gather")
        rows = offsets // nlon
        shifted = rowshift[sst[:, None] + np.arange(nh)] > 0
        # (points, nh rows, nh cols) offsets inside one horizontal plane
        col_index = np.where(shifted[:, :, None], cols_shift[:, None, :],
                             cols[:, None, :])
        in_plane = rows[:, :, None] * nlon + col_index

        for fi, field in enumerate(flat):
            acc = np.zeros(p.shape, dtype=dtype)
            lo = np.full(p.shape, np.inf, dtype=dtype)
            hi = np.full(p.shape, -np.inf, dtype=dtype)
            f_edge = np.zeros(p.shape, dtype=dtype)
            f_next = np.zeros(p.shape, dtype=dtype)
            for c in range(4):
                index = (k0 + c)[:, None, None] * plane + in_plane
                vals = field[index]
                inner = np.einsum("pa,pra,pr->p", wx, vals, wy)
                acc = acc + wz[:, c] * inner
                f_edge = np.where(c_edge == c, inner, f_edge)
                f_next = np.where(c_next == c, inner, f_next)
                if limiter != "none":
                    inbox = (c == kbox) | (c == kbox + 1)
                    box = vals[:, mid:mid + 2, mid:mid + 2].reshape(-1, 4)
                    lo = np.where(inbox, np.minimum(lo, box.min(axis=1)), lo)
                    hi = np.where(inbox, np.maximum(hi, box.max(axis=1)), hi)
            if limiter == "none":
                # The half layer outside the outermost full level: the
                # one-sided cubic held between that level's value and the
                # linear reconstruction at the boundary interface.
                f_face = f_edge + scalar(0.5) * (f_edge - f_next)
                bounded = np.minimum(
                    np.maximum(acc, np.minimum(f_edge, f_face)),
                    np.maximum(f_edge, f_face),
                )
                acc = np.where(outside, bounded, acc)
            kept = acc
            if limiter == "quasi_monotone":
                kept = np.minimum(np.maximum(acc, lo), hi)
            elif limiter == "positive_definite":
                kept = np.minimum(np.maximum(acc, scalar(0.0)), hi)
            outs[fi][k] = kept
            if deficit:
                cuts[fi][k] = acc - kept

    values = [o.reshape(nlev, nrows, nlon) for o in outs]
    if not deficit:
        return values
    return list(zip(values, [c.reshape(nlev, nrows, nlon) for c in cuts]))


def _linear_numpy(fields, stencil: Stencil) -> list:
    """The numpy specification of :func:`gather_linear_batch`."""
    tables = stencil.tables
    nlev, nlat, nlon = stencil.shape
    dtype = np.dtype(tables.dtype)
    scalar = dtype.type
    lat_ext = tables.lat_ext_host.astype(dtype)
    rowoff = tables.rowoff_host
    rowshift = tables.rowshift_host
    half_pi = scalar(np.pi / 2.0)
    plane = nlat * nlon
    xis = np.asarray(stencil.xi)
    phi = np.asarray(stencil.phi)
    lev = np.asarray(stencil.level)
    flat = [np.ascontiguousarray(np.asarray(f)).reshape(-1) for f in fields]
    outs = [np.empty((nlev, plane), dtype=dtype) for _ in fields]
    for k in range(nlev):
        p = np.clip(phi[k].reshape(-1), -half_pi, half_pi)
        xk = np.clip(lev[k].reshape(-1), scalar(0.0), scalar(nlev - 1))
        xi = xis[k].reshape(-1)
        fi = np.floor(xi)
        ax = xi - fi
        c0 = fi.astype(np.int64) % nlon
        c1 = (c0 + 1) % nlon
        d0 = (c0 + nlon // 2) % nlon
        d1 = (c1 + nlon // 2) % nlon
        m0 = np.clip(
            np.searchsorted(lat_ext.astype(np.float64),
                            p.astype(np.float64), side="right") - 1,
            1, nlat + 1,
        )
        la = lat_ext[m0]
        lb = lat_ext[m0 + 1]
        ay = np.clip((p - la) / (lb - la), scalar(0.0), scalar(1.0))
        r0 = rowoff[m0]
        r1 = rowoff[m0 + 1]
        s0 = rowshift[m0] > 0
        s1 = rowshift[m0 + 1] > 0
        i00 = r0 + np.where(s0, d0, c0)
        i01 = r0 + np.where(s0, d1, c1)
        i10 = r1 + np.where(s1, d0, c0)
        i11 = r1 + np.where(s1, d1, c1)
        kb = np.clip(np.floor(xk).astype(np.int64), 0, nlev - 2)
        az = xk - kb.astype(dtype)
        o0 = kb * plane
        o1 = o0 + plane
        bx, by, bz = 1 - ax, 1 - ay, 1 - az
        for fi_, field in enumerate(flat):
            t0 = (by * (bx * field[o0 + i00] + ax * field[o0 + i01])
                  + ay * (bx * field[o0 + i10] + ax * field[o0 + i11]))
            t1 = (by * (bx * field[o1 + i00] + ax * field[o1 + i01])
                  + ay * (bx * field[o1 + i10] + ax * field[o1 + i11]))
            outs[fi_][k] = bz * t0 + az * t1
    return [o.reshape(nlev, nlat, nlon) for o in outs]
