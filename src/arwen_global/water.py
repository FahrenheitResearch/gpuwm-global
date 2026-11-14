"""Single accounting authority for Arwen Global water reservoirs."""
from __future__ import annotations

from .constants import GRAVITY_M_S2, LIQUID_WATER_DENSITY, WATER_SPECIES
from .spill import resident

SOIL_LAYER_THICKNESS_M = (0.10, 0.30, 0.60, 1.00)
# Native Noah fields in the Level-5 PhysicsState.  They are stores, unlike
# accumulated precipitation diagnostics, and therefore enter total water.
NATIVE_CANOPY_STORE_NAMES = ("noah_canwat", "canwat")
NATIVE_SNOW_STORE_NAMES = ("noah_snow", "snow")
# Noah's runoff accumulators are the water's LAST address inside the column
# system: SSTEP saturation excess and subsurface drainage leave the soil
# column into udrunoff, and infiltration excess leaves into sfcrunoff
# (noah.cu:563-579, 1543-1544), and nothing in the column ever reads them
# back -- physically that water has left for the rivers.  The v3 ledger
# counted them as held water at land_fraction pricing, which kept every
# per-call closure exact but sequestered a monotone-growing store inside
# the pinned conservation total: with total held water fixed to the
# cold-start target, every kg the runoff accumulators grew squeezed the
# atmosphere+reservoir global mean by the same kg, forever.  v4 books each
# priced runoff-store increment to the cumulative outflow account below at
# the moment the land step debits the reservoir for it; the runoff stores
# themselves (still monotone, still per unit LAND area -- the kernel is
# untouched) are counted minus outflow, i.e. at zero.  The driven-negative
# refusal stays on them: the outflow booking prices increments of
# max(store, 0) * land_fraction, so a store driven negative would underbook
# the next exit and launder the loss through the outflow account.
NATIVE_SURFACE_RUNOFF_STORE_NAMES = ("noah_sfcrunoff", "sfcrunoff")
NATIVE_SUBSURFACE_RUNOFF_STORE_NAMES = ("noah_udrunoff", "udrunoff")
# Cumulative booked exits (kg/m2 per column), maintained by the native land
# step.  Water conservation is held + outflow = constant: the run-level
# total_water_relative_drift gate and the global water fixer target both
# read total_water_column, which counts this account as a booked exit term.
WATER_OUTFLOW_NAMES = ("water_outflow_kg_m2",)
# The open-water exchange account (kg/m2 per column, signed, unbounded):
# the water an open-water column has handed the atmosphere or taken from it
# since the run began, P - E, plus the global water fixer's corrections
# there.  An ocean, a lake or the sea under ice is not a finite store, so
# on those columns (``exchange_columns``) the native suite's per-call
# closure and the global water fixer book here instead of debiting the
# explicit surface reservoir, which a long run would otherwise empty on
# every column whose evaporation exceeds its rain: the 500 kg/m2 most
# shipped configurations seed is 100 days at 5 mm a day of E - P, and the
# run refuses when a column reaches zero.  Counted by every water ledger
# beside the outflow account, so held water + outflow + exchange is the
# conserved total, and nothing the atmosphere reads changes: no native
# kernel reads the reservoir.  The reference suite does (it caps one call's
# evaporation by it) and keeps its finite reservoir; it is a test suite,
# not a configuration for long runs.  Present in the physics namespace
# exactly when the run books it (native physics,
# runner.books_open_water_exchange); a namespace without it is the
# finite-reservoir bookkeeping of 0.1.5.
#
# The account is float64 whatever the run's precision: it is a running sum
# of small increments that reaches hundreds of kg/m2 over a season, where a
# float32 plane's spacing (6.1e-5 at 500) is a third of a percent of one
# call's open-water evaporation at dt = 300 s.  Every call's move is formed
# in the reservoir's own precision and added exactly as that number, so the
# account is the sum of what the reservoir would have carried, without the
# rounding of a float32 running total; the checkpoint keeps its dtype.
OCEAN_EXCHANGE_NAMES = ("ocean_exchange_kg_m2",)
OCEAN_EXCHANGE_NAME = OCEAN_EXCHANGE_NAMES[0]
OCEAN_EXCHANGE_DTYPE = "float64"
# The runaway watch that stands where the reservoir refusal stood on open
# water.  The finite reservoir was also the run's only bound on a column
# that evaporates without limit (a lake column started from a 321 K land
# skin drained it in 117 h, analysis_initial.open_water_skin_temperature),
# and an unbounded account has none.  So the native suite sums each
# open-water column's evaporation (the surface moisture flux the PBL
# applied, qfx * dt per call, dew negative) over fixed 24 h blocks of model
# time from the start, in this float64 plane, reports the largest column,
# and refuses the run, naming the column, once a column's sum in a block
# passes the ceiling (native option open_water_evaporation_ceiling_kg_m2_day;
# open_water_evaporation_refusal = false reports without refusing).  The
# default ceiling is a provisional engineering value, not a measured one:
# 60 kg/m2 (mm) in a day is a mean latent flux of about 1,700 W/m2.
OPEN_WATER_EVAPORATION_NAME = "ocean_evaporation_day_kg_m2"
OPEN_WATER_EVAPORATION_BLOCK_S = 86400.0


# Store content below zero is invisible to this accounting: the clamp counts
# it as zero, the global-mean total drops by only the store's prior positive
# content, and the global water fixer then manufactures that loss as real
# surface water everywhere. Refusing above roundoff scale keeps a defect that
# drives a store negative visible instead of laundered.
# -1e-9 on the dimensionless soil fraction is >1e6 times float64 roundoff on
# an O(0.1) fraction and books at most 2e-6 kg/m2 over the 2 m soil stack,
# inside the 1e-8 relative total-water gate on the ~1e3 kg/m2 column.
SOIL_FRACTION_MINIMUM = -1.0e-9
# Native stores are kg/m2; -1e-8 matches the surface-water reservoir refusal
# threshold in physics/reference.py.
NATIVE_STORE_MINIMUM_KG_M2 = -1.0e-8


def _refuse_driven_negative(name, value, minimum, xp):
    # The floors above are derived for float64, but the only configuration
    # that owns these stores ships them at float32 (native physics refuses
    # any other precision), where one representable rounding step of an
    # O(0.1) fraction is ~1.8e-8 - already past the float64 floor.  The
    # refusal therefore never fires inside 64 ulps of the store's own dtype
    # at its own magnitude; a driven-negative defect sits orders beyond.
    eps = float(xp.finfo(value.dtype).eps) if hasattr(value, "dtype") else 2.3e-16
    scale = max(1.0, float(xp.max(xp.abs(value))))
    floor = min(float(minimum), -64.0 * eps * scale)
    smallest = float(xp.min(value))
    if smallest < floor:
        raise FloatingPointError(
            f"water store {name} driven negative: min={smallest:.9g} is below "
            f"{floor:g}; below-zero content is invisible to the water ledger "
            "and the global fixer would manufacture the loss as surface water"
        )


def _staged_arrays(xp, arrays):
    """The ledger's few named physics-namespace planes, on the card.

    The tier holds the whole native namespace (2.3 GiB at T533); the
    water ledger reads three planes out of it, so only those three are
    staged and the rest never touches the card for this call.
    """
    from .spill import spilled

    if not any(spilled(value) for value in arrays.values()):
        return arrays
    wanted = (
        *NATIVE_CANOPY_STORE_NAMES, *NATIVE_SNOW_STORE_NAMES,
        *NATIVE_SURFACE_RUNOFF_STORE_NAMES,
        *NATIVE_SUBSURFACE_RUNOFF_STORE_NAMES, *WATER_OUTFLOW_NAMES,
        *OCEAN_EXCHANGE_NAMES,
    )
    return {
        name: (resident(xp, value) if name in wanted else value)
        for name, value in arrays.items()
    }


def atmospheric_water_column(grid_state):
    return (
        sum(grid_state[name] for name in WATER_SPECIES)
        * grid_state["dp"]
        / GRAVITY_M_S2
    )


def soil_water_column(surface, xp):
    # The soil stack and the land mask may be held by the pinned host
    # tier; ``resident`` stages them for this reduction and they are
    # dropped with the call.  Four soil planes and one mask, 0.024 GiB at
    # T533: the ledger reads them once a step.
    fraction = resident(xp, surface.soil_water_fraction)
    land = resident(xp, surface.land_fraction)
    _refuse_driven_negative(
        "soil_water_fraction", fraction, SOIL_FRACTION_MINIMUM, xp,
    )
    thickness = xp.asarray(
        SOIL_LAYER_THICKNESS_M,
        dtype=fraction.dtype,
    )[:, None, None]
    return xp.sum(
        xp.maximum(fraction, 0.0)
        * thickness
        * LIQUID_WATER_DENSITY
        * land[None],
        axis=0,
    )


def native_store_column(physics_state, like, xp):
    """Held native-store water: canopy and snow, at full weight.

    The runoff accumulators are validated but counted at zero: every priced
    increment they take is booked to the outflow account by the land step
    (native_runtime._land_step), so counting the stores again would count
    exited water twice.  See the account notes above WATER_OUTFLOW_NAMES.
    """
    total = xp.zeros_like(like)
    arrays = _staged_arrays(xp, getattr(physics_state, "arrays", {}))
    exited = (
        NATIVE_SURFACE_RUNOFF_STORE_NAMES,
        NATIVE_SUBSURFACE_RUNOFF_STORE_NAMES,
    )
    for choices in (
        NATIVE_CANOPY_STORE_NAMES,
        NATIVE_SNOW_STORE_NAMES,
        *exited,
    ):
        name, value = next(
            ((name, arrays[name]) for name in choices if name in arrays),
            (None, None),
        )
        if value is None:
            continue
        if tuple(value.shape) != tuple(like.shape):
            raise ValueError(
                f"native water store has shape {value.shape}, expected {like.shape}"
            )
        _refuse_driven_negative(name, value, NATIVE_STORE_MINIMUM_KG_M2, xp)
        if choices in exited:
            continue
        total += xp.maximum(value, 0.0)
    return total


def water_outflow_column(physics_state, like, xp):
    """Cumulative booked exits (kg/m2 per column); zero when never booked.

    Monotone nonnegative by construction (the land step adds priced
    increments of monotone accumulators), so a negative value is a booking
    defect and refuses like a driven-negative store: below-zero outflow
    would raise the conservation total and the global fixer would drain the
    difference out of real surface water everywhere.
    """
    arrays = _staged_arrays(xp, getattr(physics_state, "arrays", {}))
    name, value = next(
        ((name, arrays[name]) for name in WATER_OUTFLOW_NAMES if name in arrays),
        (None, None),
    )
    if value is None:
        return xp.zeros_like(like)
    if tuple(value.shape) != tuple(like.shape):
        raise ValueError(
            f"water outflow account has shape {value.shape}, expected {like.shape}"
        )
    _refuse_driven_negative(name, value, NATIVE_STORE_MINIMUM_KG_M2, xp)
    return value


def exchange_columns(land_fraction, xp):
    """The columns whose surface water is the open-water exchange account:
    open water by the land/water rule of :func:`arwen_global.statics.
    xland_plane` WITHOUT the sea-ice override, so the ocean, inland lakes
    and the sea under its ice.  The land fraction is static, so the set is
    fixed for the run whatever the ice does; land ice is land.
    """
    from .statics import xland_plane

    return xland_plane(land_fraction, None, xp) >= xp.float32(1.5)


def ocean_exchange_column(physics_state, like, xp):
    """The open-water exchange account (kg/m2 per column); zero when the
    namespace does not carry one.

    Signed by construction (P - E since the start, negative where the
    column has evaporated more than it received), so unlike the stores and
    the outflow account no driven-negative refusal applies to it.
    """
    arrays = _staged_arrays(xp, getattr(physics_state, "arrays", {}))
    value = arrays.get(OCEAN_EXCHANGE_NAME)
    if value is None:
        return xp.zeros_like(like)
    if tuple(value.shape) != tuple(like.shape):
        raise ValueError(
            f"ocean exchange account has shape {value.shape}, expected {like.shape}"
        )
    return value


def book_open_water_exchange(surface, arrays, reservoir_before, xp):
    """Hand a physics call's reservoir movement on open water to the account.

    ``reservoir_before`` is the surface reservoir as the call received it
    and ``surface.water_kg_m2`` the reservoir after the call's own credits,
    debits and closure.  On :func:`exchange_columns` the difference moves
    into ``arrays[OCEAN_EXCHANGE_NAME]`` and the reservoir goes back to the
    value it entered with, so reservoir + account is the closed column the
    call computed and the reservoir there never moves.  Land columns are not
    touched.  A namespace without the account is left as it is (the
    finite-reservoir bookkeeping) and None is returned; otherwise the moved
    plane (zero on land).
    """
    account = arrays.get(OCEAN_EXCHANGE_NAME)
    if account is None:
        return None
    reservoir = surface.water_kg_m2
    water = exchange_columns(surface.land_fraction, xp)
    moved = xp.where(water, reservoir - reservoir_before, 0.0).astype(reservoir.dtype)
    arrays[OCEAN_EXCHANGE_NAME] = (account + moved.astype(account.dtype)).astype(
        account.dtype
    )
    surface.water_kg_m2 = xp.where(water, reservoir_before, reservoir).astype(
        reservoir.dtype
    )
    return moved


def total_water_column(bundle, grid_state, xp, atmospheric=None):
    """The water stores and booked exits of a bundle, per column.

    ``atmospheric`` lets a caller that assembled the atmospheric column
    itself -- a band pipeline, which never holds the levelled water
    volume at full size -- hand it in; the plane is the same one the
    level sum below produces, so the total is the same expression in the
    same order either way.
    """
    if atmospheric is None:
        atmospheric = xp.sum(atmospheric_water_column(grid_state), axis=0)
    explicit_surface = resident(xp, bundle.surface.water_kg_m2)
    soil = soil_water_column(bundle.surface, xp)
    native = native_store_column(bundle.physics_state, explicit_surface, xp)
    outflow = water_outflow_column(bundle.physics_state, explicit_surface, xp)
    exchange = ocean_exchange_column(bundle.physics_state, explicit_surface, xp)
    # Conservation statement: held water plus booked exits.  The global
    # water fixer target and the total_water_relative_drift gate read
    # this total, so runoff leaving the column system is a booked exit,
    # not a loss for the fixer to manufacture back, and the open-water
    # exchange account is the other booked exit (or entrance).  The sum
    # without the account is the 0.1.5 expression, unchanged.
    total = atmospheric + explicit_surface + soil + native + outflow
    if OCEAN_EXCHANGE_NAME in getattr(bundle.physics_state, "arrays", {}):
        total = total + exchange
    return {
        "atmospheric": atmospheric,
        "surface": explicit_surface,
        "soil": soil,
        "native": native,
        "outflow": outflow,
        "exchange": exchange,
        "total": total,
    }


__all__ = [
    "OCEAN_EXCHANGE_NAME",
    "OCEAN_EXCHANGE_NAMES",
    "SOIL_LAYER_THICKNESS_M",
    "WATER_OUTFLOW_NAMES",
    "atmospheric_water_column",
    "book_open_water_exchange",
    "exchange_columns",
    "native_store_column",
    "ocean_exchange_column",
    "soil_water_column",
    "total_water_column",
    "water_outflow_column",
]
