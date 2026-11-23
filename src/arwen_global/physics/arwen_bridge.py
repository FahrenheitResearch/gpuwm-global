"""Admission boundary for reusing existing Arwen CUDA physics globally."""
from __future__ import annotations

from dataclasses import dataclass
import sys

from arwen_global.physics.registry import get_global_physics_adapter

from ..constants import NUMBER_MOMENTS, WATER_SPECIES
from ..spill import spilled
from .builtin_adapters import ensure_builtin_global_physics_adapters
from .exchange import PhysicsExchange, PhysicsResult


# Negativity admitted at the boundary, as a fraction of the field's own scale.
# float32 resolves a value to about 1.2e-7 of itself, so 1e-6 admits round-off
# accumulated over a handful of operations and nothing physical.  The scale
# floor of 1.0 keeps mixing ratios (order 1e-2) from being held to a tolerance
# finer than float32 resolves near unity.  Absolute floors cannot serve both
# families here: mixing ratios run to 1e-2 and number moments to 1e8.
_NEGATIVITY_RELATIVE_TOLERANCE = 1.0e-6


def _fingerprint(xp, value):
    """Cheap order-sensitive signature of one array, computed on its device.

    An array the pinned host tier holds is signed by its slot and the
    slot's write count instead.  That is a STRONGER check than the
    reduction below -- it catches a write that leaves the sum, the
    minimum and the maximum unchanged -- and it is the only affordable
    one: reducing the whole tier twice per physics call is 10 GiB of host
    reads a step at T533, on the host, where there is no bandwidth to
    spare.  It is also the right check: an adapter cannot reach a parked
    array except through the tier, and the tier counts every write.
    """
    if spilled(value):
        return ("parked", value.name, int(value.version), str(value.shape),
                str(value.dtype))
    total = float(xp.sum(value, dtype=xp.float64))
    minimum = float(xp.min(value))
    maximum = float(xp.max(value))
    return (str(tuple(value.shape)), str(value.dtype), total, minimum, maximum)


def _host_values(xp, scalars) -> list:
    """Device 0-d reductions as Python numbers, in ONE transfer.

    Each ``float(xp.sum(...))`` was its own device synchronisation, three
    per array and two signatures per physics call; the same reductions are
    stacked on the device and read back once.  Every value is widened to
    float64 on the device (exact for the float32 and float64 reductions
    and for integer extrema below 2**53), which is the Python float the
    scalar read produced.
    """
    if not scalars:
        return []
    import numpy

    # One stack per dtype, widened once, joined, read once, and put back
    # in the original order on the host.
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(scalars):
        groups.setdefault(str(value.dtype), []).append(index)
    parts, order = [], []
    for indices in groups.values():
        parts.append(xp.stack([scalars[i] for i in indices]).astype(xp.float64))
        order.extend(indices)
    stacked = parts[0] if len(parts) == 1 else xp.concatenate(parts)
    if hasattr(stacked, "get"):
        stacked = stacked.get()
    read = numpy.asarray(stacked).tolist()
    values = [0.0] * len(scalars)
    for position, index in enumerate(order):
        values[index] = float(read[position])
    return values


def _fingerprints(entries) -> dict[str, object]:
    """``_fingerprint`` of every ``(key, xp, value)`` entry, the device
    reductions of each array module read back in one transfer.  Same
    signatures, same order of keys."""
    rows: dict[str, object] = {}
    pending: dict[int, tuple[object, list]] = {}
    for key, xp, value in entries:
        if spilled(value):
            rows[key] = _fingerprint(xp, value)
            continue
        rows[key] = None
        _module, items = pending.setdefault(id(xp), (xp, []))
        items.append((key, value))
    for xp, items in pending.values():
        scalars = []
        for _key, value in items:
            scalars.extend((xp.sum(value, dtype=xp.float64), xp.min(value),
                            xp.max(value)))
        values = _host_values(xp, scalars)
        for index, (key, value) in enumerate(items):
            total, minimum, maximum = values[3 * index:3 * index + 3]
            rows[key] = (str(tuple(value.shape)), str(value.dtype), total,
                         minimum, maximum)
    return rows


def _array_module(value):
    # An array's own package is already imported wherever the array exists, so
    # this resolves cupy without the boundary importing it.
    name = type(value).__module__.split(".")[0]
    module = sys.modules.get(name)
    if module is not None and hasattr(module, "isfinite"):
        return module
    import numpy

    return numpy


@dataclass
class NativeArwenPhysicsBridge:
    adapter_name: str
    adapter_options: dict[str, object]

    def __post_init__(self) -> None:
        ensure_builtin_global_physics_adapters()
        registration = get_global_physics_adapter(self.adapter_name)
        self.registration = registration
        # The validator returns every normalized option (the build payload);
        # the receipt and the pins carry the registration's identity of
        # them (identity_of), which may drop a key whose value is the state
        # older checkpoints were written in (gf_updraft_only_when_downdraft_dry
        # = false, YSU's "wrf-layer" length).  The adapter runs the options
        # the operator wrote, laid over the normalized defaults: built from
        # the identity payload instead, it ran the timing lane's second
        # dissection arm switched on under a config and a receipt that both
        # said off, and ran the "fixed" YSU length under a config spelling
        # "wrf-layer".
        normalized = registration.validate_options(dict(self.adapter_options))
        self.adapter_options = {**normalized, **dict(self.adapter_options)}
        self.adapter = registration.factory(dict(self.adapter_options))
        if not callable(getattr(self.adapter, "step", None)):
            raise TypeError(
                f"native adapter {self.adapter_name!r} factory did not return "
                "an object with step()"
            )

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": "arwen-native",
            "adapter_name": self.registration.name,
            "contract_hash": self.registration.contract_hash,
            "admission_status": self.registration.contract["admission_status"],
            "device_evidence_sha256": self.registration.contract[
                "device_evidence_sha256"
            ],
            "scheme_identity": self.registration.contract["scheme_identity"],
            # The trajectory's name, not the build payload: an option the
            # adapter's identity drops stays out of the receipt as it stays
            # out of the config hash.
            "options": self.registration.identity_of(self.adapter_options),
        }

    def _caller_fingerprints(self, exchange: PhysicsExchange, *,
                             namespace: bool = True) -> dict[str, object]:
        xp = _array_module(exchange.theta)
        entries = []
        for name, value in exchange.prognostics().items():
            entries.append((f"prognostic.{name}", xp, value))
        for name, value in exchange.surface.arrays().items():
            entries.append((f"surface.{name}", xp, value))
        for name, value in (exchange.physics_state.arrays.items() if namespace else ()):
            entries.append((f"physics_state.{name}", _array_module(value), value))
        return _fingerprints(entries)

    def _refuse_caller_mutation(
        self, exchange: PhysicsExchange, before: dict[str, object], *,
        namespace: bool = True,
    ) -> None:
        after = self._caller_fingerprints(exchange, namespace=namespace)
        changed = sorted(
            name for name in before
            if name not in after or after[name] != before[name]
        )
        changed.extend(sorted(name for name in after if name not in before))
        if changed:
            raise ValueError(
                f"native adapter {self.adapter_name!r} wrote into the caller's "
                f"state: {', '.join(changed)}; the exchange carries the live "
                "committed bundle, so an in-place write corrupts the state the "
                "dycore is about to advance"
            )

    def _refuse_bad_values(self, result: PhysicsResult) -> None:
        # Every reduction the checks read is launched first and read back
        # in one transfer per array module (each bool()/float() of a device
        # reduction was its own synchronisation); the checks then run in
        # their original order on the read values, so the same field is
        # refused with the same sentence.  The field scale max|v| is read
        # as max(|min v|, |max v|), which is the same number for the
        # finite field the check reaches (abs is exact).
        signed = (*WATER_SPECIES, *NUMBER_MOMENTS)
        prognostics = list(result.prognostics().items())
        surfaces = list(result.surface.arrays().items())
        groups: dict[int, tuple[object, list]] = {}

        def ask(value, key, reduce):
            xp = _array_module(value)
            groups.setdefault(id(xp), (xp, []))[1].append((key, reduce(xp, value)))

        for name, value in prognostics:
            ask(value, ("finite", name), lambda xp, v: xp.all(xp.isfinite(v)))
            if name in signed:
                ask(value, ("min", name), lambda xp, v: xp.min(v))
                ask(value, ("max", name), lambda xp, v: xp.max(v))
        for name, value in surfaces:
            ask(value, ("surface", name), lambda xp, v: xp.all(xp.isfinite(v)))
        read: dict[tuple[str, str], float] = {}
        for xp, items in groups.values():
            for (key, _scalar), number in zip(
                    items, _host_values(xp, [scalar for _key, scalar in items])):
                read[key] = number
        for name, _value in prognostics:
            if not bool(read[("finite", name)]):
                raise FloatingPointError(
                    f"native adapter {self.adapter_name!r} returned non-finite "
                    f"{name}; it would enter the spectral transforms and only "
                    "surface later with no adapter named"
                )
            if name not in signed:
                continue
            minimum = read[("min", name)]
            if minimum >= 0.0:
                continue
            scale = max(1.0, abs(max(abs(minimum), abs(read[("max", name)]))))
            floor = -_NEGATIVITY_RELATIVE_TOLERANCE * scale
            if minimum < floor:
                raise FloatingPointError(
                    f"native adapter {self.adapter_name!r} returned negative "
                    f"{name}: {minimum:.9g} below {floor:.9g} at field scale "
                    f"{scale:.9g}"
                )
        for name, _value in surfaces:
            if not bool(read[("surface", name)]):
                raise FloatingPointError(
                    f"native adapter {self.adapter_name!r} returned non-finite "
                    f"surface field {name}"
                )

    #: Set by the model for a call whose input namespace is consumed
    #: (dynamics.step's second half on a resident run): the adapter works
    #: on it in place and the caller-mutation check skips that namespace
    #: (the prognostics and the surface are still checked).
    consume_input_namespace = False

    def step(self, exchange: PhysicsExchange) -> PhysicsResult:
        exchange.validate()
        consume = bool(self.consume_input_namespace)
        before = self._caller_fingerprints(exchange, namespace=not consume)
        self.adapter.consume_input_namespace = consume
        try:
            result = self.adapter.step(exchange)
        finally:
            self.adapter.consume_input_namespace = False
        if not isinstance(result, PhysicsResult):
            raise TypeError(
                f"native adapter {self.adapter_name!r} must return PhysicsResult"
            )
        expected = tuple(exchange.theta.shape)
        for name, value in result.prognostics().items():
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"native adapter result {name} shape {value.shape} != {expected}"
                )
        result.physics_state.validate()
        self._refuse_caller_mutation(exchange, before, namespace=not consume)
        self._refuse_bad_values(result)
        return result

    # The suite's deferred plane readings (native_suite: the host
    # arithmetic of a call's grid-mean readings runs where the model's step
    # says, on the same bytes); a suite without them takes neither.
    @property
    def defer_host_readings(self) -> bool:
        return bool(getattr(self.adapter, "defer_host_readings", False))

    @defer_host_readings.setter
    def defer_host_readings(self, value: bool) -> None:
        if hasattr(self.adapter, "resolve_pending_readings"):
            self.adapter.defer_host_readings = bool(value)

    def resolve_pending_readings(self) -> None:
        resolve = getattr(self.adapter, "resolve_pending_readings", None)
        if callable(resolve):
            resolve()

    def finish(self, band_diagnostics, band_metadata, planes, surface,
               physics_state, **context):
        """The call's diagnostics and metadata from its bands' results
        (physics.banding): the adapter's own merge when it declares one,
        the default otherwise."""
        finish = getattr(self.adapter, "finish", None)
        if callable(finish):
            return finish(band_diagnostics, band_metadata, planes, surface,
                          physics_state, **context)
        from .banding import default_finish

        return default_finish(band_diagnostics, band_metadata, planes,
                              surface, physics_state, **context)


__all__ = ["NativeArwenPhysicsBridge"]
