"""Vertical-order, precision and humidity-convention boundary for existing Arwen CUDA physics.

Humidity convention.  The global model carries every water field as
SPECIFIC humidity (kg per kg of moist air): the analysis init stores GDAS
SPFH (GRIB2 0/1/0) unconverted and the dycore's water ledger prices
column water as sum(q dp)/g.  The WRF-lineage kernels consume DRY MIXING
RATIO (kg per kg of dry air): noah.cu:998 forms q2k = qv1/(1+qv1), and
sfclay.cu:255-257 / morrison.cu:950 compare qv straight against
EP2*e/(p-e).  Handing q to them as if it were r read every saturation test
dry by q_v^2 -- 2.0% at q = 0.02, 1.0% at 0.01, a dry bias across the
tropical boundary layer (audit 2026-09-01 NB-3).  This module converts at
the bridge boundary, r_x = q_x / (1 - q_v) inbound for every water species
and q_x = r_x * (1 - q_v_in) outbound, so the kernels see mixing ratio while
the model keeps its convention.

The outbound factor is the INBOUND dry fraction d = 1 - q_v_in, held on
the batch (``NativeColumnBatch.dry_fraction``), not 1 / (1 + r_v) of the
returned vapour.  The physics call keeps dp fixed, so a layer's dry-air
mass d dp / g is fixed through it and every species' mass is r_x d dp / g.
Dividing by the post-kernel 1 + r_v instead re-priced every species
against a dry mass that changes with the vapour: a pure vapour-to-
condensate change, which conserves sum_x r_x, raised sum_x q_x dp / g by
about q_v times the condensed amount (one column on the CPU, 2026-11-15:
+0.151 kg/m2 on 10.04 kg/m2 condensed, 1.50%), so the rain books stopped
matching the atmosphere's loss.  With d held, a kernel that conserves
mixing-ratio water conserves the model's ledger to round-off and fallout
of F kg per kg of dry air is exactly F d dp / g in that ledger.  When the
kernel changes nothing the return trip is r = q / d, q' = r d: one float32
rounding each way, as the 1 / (1 + r_v) form was.  Number concentrations stay per kg of
moist air: the dycore rebuilds an untouched moment as raw + tendency so it
round-trips exactly, and a per-kg-of-dry-air rescale would turn that exact
round-trip into a one-ulp walk per call for a 2% shift in the implied mean
particle mass.  Stated divergence, not converted.

Water ledger.  ``atmospheric_water_kg_m2`` prices the batch's column water
in the MODEL's metric (specific humidity times dp/g) from the kernel-side
mixing ratios, so the suite's closure and the runtime's reservoir bookings
compare like with like on both sides of every kernel call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    NUMBER_MOMENTS,
    WATER_SPECIES,
)
from ..spectral.backend import device_cache_key
from ..spill import prefetch, resident, spilled
from ..state import PhysicsState, SURFACE_ARRAY_NAMES, SurfaceState, _copy_json
from ..vertical import HybridCoordinate
from .exchange import PhysicsExchange


#: The fused column-water kernel, one entry per (module, DEVICE).  See
#: :func:`~arwen_global.spectral.backend.device_cache_key`; gate CARD-1.
_COLUMN_WATER_KERNEL: dict[tuple, object] = {}
#: The fused column-increment kernel (cumulus rain on the model's mass),
#: one entry per (module, DEVICE), as above.
_COLUMN_INCREMENT_KERNEL: dict[tuple, object] = {}


def staged_surface(xp, surface):
    """The surface reservoirs on the card, whatever holds them.

    ``SurfaceState.copy`` is the batch's own private copy of the caller's
    surface; when the pinned host tier holds those planes, the copy IS
    the stage, one plane at a time, and the tier keeps the original.
    """
    if not any(
        spilled(getattr(surface, member)) for member in SURFACE_ARRAY_NAMES
    ):
        return surface.copy()
    def _one(member):
        value = getattr(surface, member)
        # A parked plane's stage IS the private copy; one already on the
        # card is copied as ``SurfaceState.copy`` copies it.
        return resident(xp, value) if spilled(value) else value.copy()

    return SurfaceState(**{
        member: _one(member) for member in SURFACE_ARRAY_NAMES
    })


def staged_physics_state(xp, physics_state):
    """The physics namespace on the card, whatever holds it.

    The same contract as :func:`staged_surface`: a suite that copies the
    caller's namespace before working on it gets the stage as its copy.
    """
    if not any(spilled(value) for value in physics_state.arrays.values()):
        return physics_state.copy()
    return PhysicsState(
        schema=physics_state.schema,
        arrays={
            name: (resident(xp, value) if spilled(value) else value.copy())
            for name, value in physics_state.arrays.items()
        },
        metadata=_copy_json(physics_state.metadata),
    )


def _column_water_kernel(xp):
    """``(sum_x r_x) d_in dp / g`` per cell in float64, fused.

    ``d_in`` is the batch's inbound dry fraction
    (:attr:`NativeColumnBatch.dry_fraction`), the layer's fixed dry-air
    share of dp; see the module docstring for why it is not 1 / (1 + r_v)
    of the current vapour.  Specification: the numpy branch of
    :meth:`NativeColumnBatch.atmospheric_water_kg_m2`.  ``-fmad=false``
    keeps each multiply and add rounded separately, as the ufunc chain
    rounds them; the species order is ``WATER_SPECIES``.
    """
    key = device_cache_key(xp)
    kernel = _COLUMN_WATER_KERNEL.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "float32 qv, float32 qc, float32 qr, float32 qi, float32 qs, "
            "float32 qg, float32 dry, float32 dp, float64 gravity",
            "float64 out",
            """
            double total = 0.0;
            total = total + (double)qv;
            total = total + (double)qc;
            total = total + (double)qr;
            total = total + (double)qi;
            total = total + (double)qs;
            total = total + (double)qg;
            out = total * (double)dry * (double)dp / gravity;
            """,
            "arwen_native_column_water_dry",
            options=("-fmad=false",),
        )
        _COLUMN_WATER_KERNEL[key] = kernel
    return kernel


def _column_increment_kernel(xp):
    """``increment d_in dp / g`` per cell in float64, fused: the
    specification is the numpy branch of
    :meth:`NativeColumnBatch.column_integral_kg_m2`; ``-fmad=false`` as
    in :func:`_column_water_kernel`."""
    key = device_cache_key(xp)
    kernel = _COLUMN_INCREMENT_KERNEL.get(key)
    if kernel is None:
        kernel = xp.ElementwiseKernel(
            "float32 increment, float32 dry, float32 dp, float64 gravity",
            "float64 out",
            "out = (double)increment * (double)dry * (double)dp / gravity;",
            "arwen_native_column_increment",
            options=("-fmad=false",),
        )
        _COLUMN_INCREMENT_KERNEL[key] = kernel
    return kernel


def _flipped_quotient(xp, source, dry):
    """``ascontiguousarray(float32(source)[::-1]) / dry`` in one pass.

    The level flip and the humidity conversion of one inbound water
    species were two volume passes (a flipped contiguous copy, then the
    quotient); one elementwise divide that reads the source flipped and
    writes a contiguous result is the same float32 divide per element.
    """
    value = xp.asarray(resident(xp, source), dtype=xp.float32)
    out = xp.empty(value.shape, dtype=xp.float32)
    xp.divide(value[::-1], dry, out=out)
    return out


def mixing_ratio_from_specific_humidity(species):
    """q -> r = q / (1 - q_v) for every per-kg water field in ``species``.

    ``species`` maps field names to arrays (or scalars) sharing one
    convention and must carry ``"qv"``.  Arithmetic stays in the inputs'
    dtype and array module.
    """
    dry = 1.0 - species["qv"]
    return {name: value / dry for name, value in species.items()}


def specific_humidity_from_mixing_ratio(species):
    """r -> q = r / (1 + r_v), the exact inverse of the conversion above.

    The inverse of ONE state's conversion only: across a physics call that
    moves vapour, the model's metric is :func:`specific_humidity_at_dry_fraction`
    with the inbound dry fraction (module docstring).
    """
    moist = 1.0 + species["qv"]
    return {name: value / moist for name, value in species.items()}


def specific_humidity_at_dry_fraction(species, dry):
    """r -> q = r * d for every per-kg water field, d the INBOUND dry
    fraction 1 - q_v_in: the bridge's outbound conversion, under which a
    physics call that conserves mixing-ratio water at fixed dp conserves
    sum_x q_x dp / g (module docstring)."""
    return {name: value * dry for name, value in species.items()}


@dataclass
class NativeColumnBatch:
    """Complete copy of one global exchange in native bottom-to-top FP32 order.

    Water species are held as dry mixing ratios (see the module docstring);
    everything else is the exchange's own convention.
    """

    xp: object
    arrays: dict[str, object]
    surface: object
    physics_state: object
    time_s: float
    dt_s: float
    #: The globe's rows this batch is (PhysicsExchange.band), or None for
    #: the whole grid; the runtime keys its per-grid state on it.
    band: tuple[int, int] | None = None
    #: The model-top pressure the exchange read over the whole grid
    #: (PhysicsExchange.model_top_pa), or None to take this batch's own.
    model_top_pa: float | None = None
    #: The inbound dry fraction d = 1 - q_v_in per cell (float32, native
    #: order), the divisor the inbound conversion used and the factor of
    #: the outbound one and of the ledger (module docstring).  Fixed for
    #: the batch's life: it is the layer's dry-air share of dp, which no
    #: step of a physics call changes.  A batch built directly from mixing
    #: ratios (None here) takes 1 / (1 + r_v) of the vapour it is built
    #: with, the same dry fraction of the state as handed in.
    dry_fraction: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.dry_fraction is None and "qv" in self.arrays:
            xp = self.xp
            qv = xp.asarray(self.arrays["qv"], dtype=xp.float32)
            self.dry_fraction = xp.ascontiguousarray(
                xp.float32(1.0) / (xp.float32(1.0) + qv))

    @classmethod
    def from_exchange(cls, exchange: PhysicsExchange, xp) -> "NativeColumnBatch":
        exchange.validate()

        # The exchange may carry arrays the pinned host tier holds rather
        # than the card (spill.SpilledArray): the grid tracers and the
        # surface reservoirs.  ``resident`` stages one, and the batch's
        # own copy below IS the device copy the caller would otherwise
        # have held for the whole run -- which is what makes the tier a
        # capacity mechanism rather than a second copy.  The staged array
        # dies with the expression that consumes it.
        def volume(value):
            staged = resident(xp, value)
            out = xp.ascontiguousarray(
                xp.asarray(staged, dtype=xp.float32)[::-1]
            )
            del staged
            return out

        def level(value):
            staged = resident(xp, value)
            out = xp.ascontiguousarray(
                xp.asarray(staged, dtype=xp.float32)[::-1]
            )
            del staged
            return out

        def plane(value):
            staged = resident(xp, value)
            out = xp.ascontiguousarray(xp.asarray(staged, dtype=xp.float32))
            del staged
            return out

        # No geopotential volume: the runtime never reads one (the cumulus
        # scheme's height coordinate is the terrain height built below and
        # the layer depths come from _dz), so at T533 float32 it was 0.19
        # GiB copied and held through every physics call for nothing.
        arrays = {
            "latitude_deg": plane(exchange.latitude_deg),
            "longitude_deg": plane(exchange.longitude_deg),
            "p_half": level(exchange.p_half),
            "p_full": volume(exchange.p_full),
            "dp": volume(exchange.dp),
            "exner": volume(exchange.exner),
            "temperature": volume(exchange.temperature),
            "theta": volume(exchange.theta),
            "virtual_temperature": volume(exchange.virtual_temperature),
            "u": volume(exchange.u),
            "v": volume(exchange.v),
        }
        # One species live at a time through the r = q / (1 - q_v)
        # conversion (mixing_ratio_from_specific_humidity, applied per
        # species with the same shared divisor): the six-volume specific
        # stack this replaced was a 1.15 GiB transient at T533 float32.
        qv = volume(exchange.qv)
        dry = 1.0 - qv
        # The tracers arrive in a known order, so each one's copy is
        # started while the one before it is being converted: the double
        # buffer, at the one loop that knows what it reads next.
        order = (*WATER_SPECIES, *NUMBER_MOMENTS)
        prefetch(getattr(exchange, order[1]))
        for index, name in enumerate(WATER_SPECIES):
            if index + 1 < len(order):
                prefetch(getattr(exchange, order[index + 1]))
            if name == "qv":
                arrays[name] = xp.ascontiguousarray(qv / dry)
                continue
            arrays[name] = _flipped_quotient(xp, getattr(exchange, name), dry)
        # The divisor stays on the batch: it is the outbound factor and the
        # ledger's (module docstring).
        dry_fraction = xp.ascontiguousarray(dry)
        del qv, dry
        for index, name in enumerate(NUMBER_MOMENTS):
            step = len(WATER_SPECIES) + index + 1
            if step < len(order):
                prefetch(getattr(exchange, order[step]))
            arrays[name] = volume(getattr(exchange, name))
        # Terrain height for the cumulus scheme's height coordinate
        # (gf.cu builds zo from ht upward and reads zo[kpbl] as WRF does).
        # The exchange carries full-level geopotential built hydrostatically
        # from the surface (dynamics._hydrostatic_geopotential), so the
        # surface value is the bottom layer's inverted with the same
        # half-layer integral (HybridCoordinate.half_layer_thickness, Tv
        # linear in ln p; DN-3): phi_s = phi_full - that thickness.  Taken
        # in the exchange's own precision before the float32 cast.
        phi_bottom = xp.asarray(resident(xp, exchange.geopotential))[-1]
        tv = xp.asarray(resident(xp, exchange.virtual_temperature))
        p_half = xp.asarray(resident(xp, exchange.p_half))
        p_full = xp.asarray(resident(xp, exchange.p_full))
        half = HybridCoordinate.half_layer_thickness(
            tv, p_full, xp.log(p_half[1:] / p_half[:-1]), xp
        )
        arrays["terrain_height_m"] = plane((phi_bottom - half[-1]) / GRAVITY_M_S2)
        if exchange.omega_half_pa_s is not None:
            arrays["omega_half"] = level(exchange.omega_half_pa_s)
        # The physics namespace is carried by reference in a private dict:
        # nothing on the batch writes into it (the persistent state the
        # runtime works on copies every array it touches first, see
        # native_state._array), so a deep copy here was one whole
        # namespace (1.5 GiB at T533 with 40 levels) held per call for
        # nothing.  The caller's arrays stay untouched, which the bridge's
        # fingerprint check proves after every call.
        return cls(
            xp=xp,
            arrays=arrays,
            surface=staged_surface(xp, exchange.surface),
            physics_state=PhysicsState(
                schema=exchange.physics_state.schema,
                arrays=dict(exchange.physics_state.arrays),
                metadata=_copy_json(exchange.physics_state.metadata),
            ),
            time_s=float(exchange.time_s),
            dt_s=float(exchange.dt_s),
            band=(None if exchange.band is None
                  else (int(exchange.band[0]), int(exchange.band[1]))),
            model_top_pa=(None if exchange.model_top_pa is None
                          else float(exchange.model_top_pa)),
            dry_fraction=dry_fraction,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.arrays["theta"].shape)

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.shape[1:]

    def to_global(self, value):
        return self.xp.ascontiguousarray(value[::-1])

    def specific_humidities(self, dtype=None) -> dict[str, object]:
        """The batch's water species in the model's convention (unreversed):
        q_x = r_x d_in, the outbound conversion (module docstring)."""
        xp = self.xp
        dtype = xp.float32 if dtype is None else dtype
        return specific_humidity_at_dry_fraction(
            {name: xp.asarray(self.arrays[name], dtype=dtype) for name in WATER_SPECIES},
            xp.asarray(self.dry_fraction, dtype=dtype),
        )

    def atmospheric_water_kg_m2(self):
        """Column water in the model's metric, sum_x q_x dp / g, in float64,
        with q_x = r_x d_in (the outbound conversion; module docstring).

        Every reservoir booking and the suite's closure read this, so a
        kernel-side change measured here is exactly what the model's own
        ledger (water.py atmospheric_water_column on the returned specific
        humidities) will see, up to the float32 cast of the return trip.
        Because d_in dp / g is the layer's fixed dry-air mass, a kernel
        that conserves mixing-ratio water leaves this reading unchanged to
        round-off, and fallout reads here as exactly the water that left.
        """
        xp = self.xp
        if hasattr(xp, "ElementwiseKernel"):
            # One fused pass on the device: the same float64 chain as the
            # numpy specification below (0 + r_qv, + r_qc, ... in
            # WATER_SPECIES order; times d_in, times dp, over g) with
            # multiply-add contraction off, so every rounding sits where
            # the specification rounds.  The species reduction over levels
            # stays the backend's own sum.  Why: the specification streams
            # six float32 volumes through float64 temporaries per call and
            # the runtime calls this up to eight times per physics call;
            # the fused pass reads the eight inputs once.
            out = xp.empty(self.arrays["dp"].shape, dtype=xp.float64)
            _column_water_kernel(xp)(
                self.arrays["qv"], self.arrays["qc"], self.arrays["qr"],
                self.arrays["qi"], self.arrays["qs"], self.arrays["qg"],
                self.dry_fraction, self.arrays["dp"], float(GRAVITY_M_S2), out,
            )
            return out.sum(axis=0)
        # The species sum is accumulated one float64 species at a time
        # (0 + r_qv, + r_qc, ... in WATER_SPECIES order), so at most one
        # converted species is live instead of six (2.3 GiB at T533 with
        # 40 levels).  This numpy chain is the specification of the fused
        # kernel above.
        total = 0
        for name in WATER_SPECIES:
            total = total + xp.asarray(self.arrays[name], dtype=xp.float64)
        dry = xp.asarray(self.dry_fraction, dtype=xp.float64)
        dp = xp.asarray(self.arrays["dp"], dtype=xp.float64)
        return (total * dry * dp / GRAVITY_M_S2).sum(axis=0)

    def column_integral_kg_m2(self, increment):
        """``sum_k increment_k d_in dp_k / g`` per column in float64: one
        water species' float32 increment (native order) priced on the
        model's mass exactly as :meth:`atmospheric_water_kg_m2` prices the
        species, so a scheme's applied increments integrate to the change
        that reading measures, up to the float32 rounding of the update.
        One fused pass on the device (one float64 volume, as the ledger's
        own), the same chain in numpy."""
        xp = self.xp
        increment = xp.asarray(increment, dtype=xp.float32)
        if hasattr(xp, "ElementwiseKernel"):
            out = xp.empty(increment.shape, dtype=xp.float64)
            _column_increment_kernel(xp)(
                increment, self.dry_fraction, self.arrays["dp"],
                float(GRAVITY_M_S2), out,
            )
            return out.sum(axis=0)
        return (
            increment.astype(xp.float64)
            * xp.asarray(self.dry_fraction, dtype=xp.float64)
            * xp.asarray(self.arrays["dp"], dtype=xp.float64)
            / GRAVITY_M_S2
        ).sum(axis=0)

    def global_prognostics(self, *, consume: bool = False) -> dict[str, object]:
        """The batch's prognostics back in the model's order and convention.

        ``consume=True`` releases each batch array as its global copy is
        made, for a caller that is done with the batch: the fourteen
        copies otherwise stood beside the fourteen originals (2.67 GiB
        each at T533 float32) at the end of every physics call.  Same
        values either way: q_x = r_x d_in per species (d_in the inbound
        dry fraction; module docstring), then the level flip.
        """
        xp = self.xp
        arrays = self.arrays
        dry = xp.asarray(self.dry_fraction, dtype=xp.float32)

        def take(name):
            return arrays.pop(name) if consume else arrays[name]

        out = {}
        for name in ("u", "v", "theta"):
            out[name] = self.to_global(take(name))
        # The product and the level flip in one pass (one elementwise
        # multiply reading both operands flipped and writing a contiguous
        # result): the same float32 multiply per element as multiplying
        # and then flipping, without the intermediate volume.
        dry_flipped = dry[::-1]
        for name in WATER_SPECIES:
            value = xp.asarray(take(name), dtype=xp.float32)
            result = xp.empty(value.shape, dtype=xp.float32)
            xp.multiply(value[::-1], dry_flipped, out=result)
            out[name] = result
            del value
        del dry, dry_flipped
        if consume:
            self.dry_fraction = None
        for name in NUMBER_MOMENTS:
            out[name] = self.to_global(take(name))
        return out

    def validate(self) -> None:
        shape = self.shape
        for name in ("u", "v", "theta", "p_full", "dp", "exner",
                     "temperature", *WATER_SPECIES,
                     *NUMBER_MOMENTS):
            value = self.arrays[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"native batch {name} shape mismatch")
            if value.dtype != self.xp.float32 or not value.flags.c_contiguous:
                raise TypeError(f"native batch {name} must be contiguous float32")
        dry = self.dry_fraction
        if dry is None or tuple(dry.shape) != shape:
            raise ValueError("native batch dry_fraction shape mismatch")
        if tuple(self.arrays["p_half"].shape) != (shape[0] + 1, *shape[1:]):
            raise ValueError("native batch p_half shape mismatch")
        omega = self.arrays.get("omega_half")
        if omega is not None and tuple(omega.shape) != (shape[0] + 1, *shape[1:]):
            raise ValueError("native batch omega_half shape mismatch")


__all__ = [
    "NativeColumnBatch",
    "staged_physics_state",
    "staged_surface",
    "mixing_ratio_from_specific_humidity",
    "specific_humidity_at_dry_fraction",
    "specific_humidity_from_mixing_ratio",
]
