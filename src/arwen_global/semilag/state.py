"""The second time level a two-time-level scheme carries.

Seven grid arrays.  Three of them are the previous step's advecting flow,
which the trajectory extrapolates (``V_ex = 2 V^n - V^{n-1}``); four are
the previous step's nonlinear residual, which the arrival side of the step
extrapolates the same way.  They live here rather than inside
:class:`arwen_global.state.MoistHybridState` because that class is
the PROGNOSTIC state and its field inventory is pinned in three places (the
pin document, the checkpoint's array namespaces, and every export); the
trajectory level is a property of the INTEGRATOR, and an integrator that
does not carry one writes no such arrays at all.

They are checkpointed all the same.  The device-qualification pin of
record is ``uninterrupted-versus-midpoint-restart-bit-exact-plus-device-
identity-v1``, and a restart that rebuilt the second level with a
non-extrapolated start-up step would not be the continuation of the
uninterrupted run: the pin would have to be overridden with an admission
of a weaker property.  Carrying the level instead costs 317 MiB at T255
and 1,381 MiB at T533 in float32, which is 4.5 and 5.1 percent of the
measured IMEX device peaks of 7.05 and 27.07 GiB (2026-09-06, the 32 GB host).

The wind is stored in the LOCAL east-north components, not the geocentric
Cartesian ones the trajectory reads.  The arrival points are the grid
points and never move, so the local basis at the arrival point is a
constant of the run and the conversion back is deterministic arithmetic on
the stored arrays; two arrays are 45 MiB less at T255 and 196 MiB less at
T533 than three.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The array names, in the order the checkpoint lists them.  Three of them
#: are the flow of the step that has just been taken and four are its
#: nonlinear residual; ``n_lnps`` is two-dimensional and the rest are not.
TRAJECTORY_FIELDS = (
    "u_prev", "v_prev", "s_prev", "n_u", "n_v", "n_theta", "n_lnps",
)
#: The one member with no level axis.
TRAJECTORY_SURFACE_FIELDS = ("n_lnps",)


@dataclass(frozen=True)
class TrajectoryState:
    """One step's worth of second-level state."""

    #: eastward and northward wind of the state the previous step's
    #: dynamics advected, at the grid points, m/s
    u_prev: Any
    v_prev: Any
    #: its vertical rate in continuous level indices per second
    s_prev: Any
    #: the previous step's nonlinear residual at the grid points: the
    #: advective-form tendency minus the semi-implicit linear operator
    n_u: Any
    n_v: Any
    n_theta: Any
    #: the same for ln surface pressure, two-dimensional
    n_lnps: Any

    def __post_init__(self) -> None:
        shape = tuple(int(s) for s in self.u_prev.shape)
        if len(shape) != 3:
            raise ValueError(
                f"the trajectory state's volume arrays must be "
                f"(nlev, nlat, nlon), got {shape}"
            )
        for name in TRAJECTORY_FIELDS:
            value = getattr(self, name)
            got = tuple(int(s) for s in value.shape)
            want = shape[1:] if name in TRAJECTORY_SURFACE_FIELDS else shape
            if got != want:
                raise ValueError(
                    f"trajectory member {name} is {got} where {want} was "
                    f"expected"
                )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.u_prev.shape)  # type: ignore[return-value]

    def arrays(self) -> dict[str, Any]:
        """The seven members by name, in ``TRAJECTORY_FIELDS`` order."""
        return {name: getattr(self, name) for name in TRAJECTORY_FIELDS}

    @classmethod
    def from_arrays(cls, arrays: dict[str, Any]) -> "TrajectoryState":
        missing = [name for name in TRAJECTORY_FIELDS if name not in arrays]
        if missing:
            raise ValueError(
                "the trajectory state needs all of "
                f"{list(TRAJECTORY_FIELDS)}; missing {missing}"
            )
        return cls(**{name: arrays[name] for name in TRAJECTORY_FIELDS})


@dataclass(frozen=True)
class RowTrajectory:
    """The second time level of ONE CARD of a multi-card run: the rows it
    owns and nothing else.

    The trajectory reads its previous level at the ARRIVAL point only (the
    extrapolated flow and the extrapolated residual), and a card computes
    the arrival points of its own rows, so a card never reads another
    card's rows of it.  Holding the globe on every card would cost the
    whole level P times over: 2.59 GiB a card at T799 L40 float32, twice
    over while a step builds the next level beside the last.

    :meth:`whole` lays the rows into whole-grid arrays (zero elsewhere) for
    the readers that want the globe, the checkpoint first among them,
    whose card gather fills the other cards' rows before anything is
    hashed (runner._card_gather_of, cards.gather_named_arrays).
    """

    first: int
    last: int
    nlat: int
    arrays: dict

    def rows(self, name: str, rows: slice, xp):
        """``name``'s values at grid ``rows``, which this card owns."""
        start = int(rows.start) - self.first
        stop = int(rows.stop) - self.first
        if start < 0 or stop > self.last - self.first:
            raise ValueError(
                f"trajectory rows {rows.start}..{rows.stop} are not this "
                f"card's {self.first}..{self.last}: a card reads its own "
                "arrival rows of the second time level and no other card's"
            )
        return xp.ascontiguousarray(self.arrays[name][..., start:stop, :])

    def whole(self, xp) -> TrajectoryState:
        out = {}
        for name, value in self.arrays.items():
            shape = (*value.shape[:-2], self.nlat, value.shape[-1])
            array = xp.zeros(shape, dtype=value.dtype)
            array[..., self.first:self.last, :] = value
            out[name] = array
        return TrajectoryState(**out)


__all__ = [
    "RowTrajectory",
    "TRAJECTORY_FIELDS",
    "TRAJECTORY_SURFACE_FIELDS",
    "TrajectoryState",
]
