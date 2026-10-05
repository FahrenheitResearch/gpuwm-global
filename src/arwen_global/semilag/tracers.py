"""Semi-Lagrangian transport of the ten grid tracers, and its mass fixer.

The condensate species and number moments ride the SAME stencil as the
wind, the thermodynamic variable and the vapour: one departure-point
search and one index-and-weight computation serve every field of the
step, so ten tracers are ten more rows in a batched gather rather than a
second transport scheme with a second time step of its own.

That matters arithmetically as well as in cost.  The shipped flux-form
sweep is sub-cycled to a per-face Courant number of 0.25, and on the
outermost T255 ring, which is 326 m wide, a 10 m/s zonal wind at
dt = 300 s has a Courant number of 9.2 and needs 37 sub-steps; at T533
the same wind on a 75 m ring needs 160.  The flux form does not survive
the step this integrator exists to take.  It is not deleted, and
``transport.py`` is not edited: ``[semilag] tracer_scheme = "flux_form"``
still routes the tracers through it for the arm that compares the two.

What a semi-Lagrangian gather does not do is conserve mass.  It reads a
MIXING RATIO at a point; the layer thickness that mixing ratio is
measured against has changed underneath it, and nothing in the
interpolation knows that.  On top of that the quasi-monotone limiter cuts
the cubic's overshoot at every sharp maximum, and a condensate species is
almost all sharp maxima.

Both effects are MEASURED, on the T255 native arms of 2026-09-06 (the 16 GB host,
dt = 300 s, the full native suite).  The two arm sets run the tree's two
cases and the numbers below are not two points on one run: the six-hour
arms are the GDAS 2026-08-30 18Z case
(configs/verify/arwen_global_gdas_t255_native_sl_si_6h.toml) and the
forecast day is the GDAS 2026-09-01 00Z case.  The largest per-step
relative mass the fixer had to move was 5.96e-2 over six hours and 1.01e-1
over the forecast day, and the six-hour arm with the limiter TURNED OFF
reads 6.43e-2, which is the same number as the six-hour arm with it on.

That equality is the finding, and it took fixing the defect below to see
it.  The mass is not the limiter's; it is POSITIVITY's.  A cubic through a
field that is zero at three of its four stencil points undershoots below
zero on the shoulder of every maximum, and something has to lift it back,
because a negative mixing ratio is not a state this model has.  Turning
the limiter off does not stop the lift, it moves it: the negatives arrive
at the fixer's own floor instead of at the gather's clip, and the same
mass is created either way.  Turning BOTH off does stop it, and the run is
then refused at its first step on a cloud water of -1.49e-5 against a
maximum of 1.02e-3.

So the fixer's magnitude is the ADVECTION's error, it is the same to two
figures under every form below, and what separates the forms is not how
much they move but where they put it:

``bermejo_conde`` (default)
    The Bermejo-Conde mass fixer (Mon. Wea. Rev. 130, 423-430, 2002), in
    the form the IFS tracer mass fixers use (Geosci. Model Dev. 7,
    965-979, 2014, their Eq. 5 and 6): the correction at a point is
    ``lambda * w`` with the one-signed weight

        w = max(0, sgn(d) (q_L - q_H)) ** beta,   beta = 1

    where ``d`` is the mass to restore, ``q_H`` the limited cubic value
    the gather returned and ``q_L`` the TRILINEAR value at the same
    departure point inside the limiter's own cell
    (``interpolate.gather_linear_batch``).  The two interpolants agree
    wherever the field is resolved and part where it is not, which is
    where the interpolation and its positivity floor made the mass error,
    so the correction lands on the feature that made it and not on every
    point of the planet that holds the species.  The weight is one-signed
    so a surplus is taken only where the cubic rose above the trilinear
    value and a deficit restored only where it fell below it: every point
    moves TOWARD its trilinear value and is capped at reaching it, so the
    result stays between the two interpolants, inside the limiter's box
    and nonnegative (water filling: a capped point drops out and the rest
    is re-spread, a few passes).  MEASURED 2026-10-05 on T255: the
    two-signed weight ``|q_H - q_L| ** 1.5`` (the unlimited variant of
    the paper's section 3.2) emptied points holding a
    thousandth of a species' maximum and more, by taking a surplus from
    points whose cubic had already fallen below the trilinear value.
    What the weighted stage cannot place, which is the float roundoff of
    the sums and, rarely, a correction larger than every point's room,
    closes as one uniform relative factor over the species and is
    REPORTED as ``semilag_tracer_fixer_uniform_rescale__<name>``.
``mass_proportional``
    The previous default, kept as the counter-arm: one uniform RELATIVE
    adjustment of every point that holds the species, ``q (1 + d/W)``
    with ``W`` the species' mass.  Conservative, and wrong in place.  On
    the T255 water-closure receipts (2026-10-01) its per-step factor
    reached 0.116 for graupel, 0.057 for snow and 0.051 and 0.043 for
    their number moments in the worst step of a day, and 0.086 to 0.089 in
    a five-day run: every graupel shaft on the planet rescaled by up to 12
    percent in one step to pay for positivity mass created at the
    shoulders of other storms.  It was named ``bermejo_conde`` until
    2026-10-05; that name now means the published weighting above.
``bermejo_conde_additive``
    Two stages.  First the clip deficit, the signed mass the limiter moved
    at every point, which the gather reports at no extra arithmetic: the
    correction goes back exactly there, scaled by one global number and
    capped so it can never over-restore past the raw interpolated value or
    drive a point negative.  Whatever is left over closes multiplicatively
    as ``mass_proportional`` does.  MEASURED on the 2026-09-06 arms: that
    stage carries 1 to 6 percent of the correction under the default
    physics coupling and 100 percent under ``physics_coupling =
    "arrival"``, where the correction is an ADDITION and the deficit
    points the right way.  The reason it is small under the default is
    the direction above: the limiter's mass came from lifting points to
    zero, and a point that now holds exactly zero cannot give any of it
    back without going negative.
``proportional`` is RETIRED and refused by name.  It weighted the
correction by the signed advected value where ``mass_proportional``
weights it by the positive part, and once the positivity floor moved in
front of the mass measurement the two weightings became the same array at
every point of every field: MEASURED 2026-09-06 on the numpy path, bitwise
identical output with the limiter on AND off, on a spiky field and on a
cloudy one.  A door whose two values produce the same run and two
different config hashes is the flag-parsed-and-ignored failure this
package refuses everywhere else, so it is refused here rather than left
selectable.

Every form floors the advected field at zero BEFORE they measure it, and
report the mass that floor created per species.  A negative mixing ratio
is not a state the model has, so the floor is not optional; measuring the
mass after it is what makes the correction close.

The refusal in the middle of both names its breakage.  A species whose
entire mass has left the model cannot have that mass restored by a
weighted fixer, and a fixer that fell back to spreading it uniformly
would create the mass in clear air, which is precisely the failure the
level-wide spectral rescale used to have: 37 percent of the planet's
cloud water moved out of its columns on every positivity pass, four per
step, and grid-scale precipitation never reached the ground (the pin
document's ``water_repair`` v6 entry).
"""
from __future__ import annotations

from typing import Any

from .interpolate import Stencil, gather_batch, gather_linear_batch
from ..spill import resident

#: The fixer forms the ``[semilag] tracer_fixer`` door accepts.  ``none``
#: reports the drift and closes nothing; it is the counter-arm that says
#: what the fixer is worth, not a shipping choice.
TRACER_FIXERS = (
    "bermejo_conde", "mass_proportional", "bermejo_conde_additive", "none",
)

#: The forms that weigh the correction by the high-minus-low-order
#: interpolation difference and therefore need the departure stencil.
STENCIL_FIXERS = ("bermejo_conde",)

#: The Bermejo-Conde exponent on the interpolation difference.  The IFS
#: implementation runs 1, having found no benefit from larger values and
#: sharper, larger increments with them; this one follows it.
BERMEJO_CONDE_BETA = 1.0

#: Water-filling passes.  Each pass re-spreads what the capped points
#: could not take over the points that still have room before their
#: trilinear value; what is left after the last one is closed uniformly
#: and reported.
BERMEJO_CONDE_PASSES = 4

#: A point the fixer changed by more than this fraction of its own value
#: counts as touched in ``semilag_tracer_fixer_touched_mass_fraction``.
TOUCHED_RELATIVE = 1.0e-2

#: Names the door once accepted and now refuses, each with the reason.
#: A retired name is refused rather than dropped, because a config that
#: still carries it would otherwise run under a form it did not ask for.
RETIRED_TRACER_FIXERS = {
    "proportional": (
        "it weighted the correction by the signed advected value where "
        "'mass_proportional' weights it by the positive part, and every "
        "conservative form now floors the field at zero BEFORE it measures "
        "the mass, so the two weightings are the same array at every point: "
        "MEASURED 2026-09-06, bitwise identical output with the "
        "quasi-monotone limiter on and off, on a spiky field and on a cloudy "
        "one.  Selecting it produced the same run as 'mass_proportional' "
        "under a different config hash, which is the parsed-and-ignored "
        "failure this package refuses everywhere else"
    ),
}

#: The forms that read the limiter's clip deficit and therefore need the
#: gather to report it.
DEFICIT_FIXERS = ("bermejo_conde_additive",)


def area_weights(transform):
    """Fractional area of one grid cell, summing to one over the sphere.

    The same convention ``_transport_grid_tracers`` uses for its own
    global reductions, so a mass computed here and a mass computed there
    are the same number.
    """
    backend = transform.backend
    return backend.asarray(
        transform.grid.quadrature_weights, dtype=backend.float_dtype
    )[None, :, None] / (2.0 * transform.grid.nlon)


def _mass(xp, field, dp, cell) -> float:
    return float(xp.sum(field * dp * cell))


def advect(
    tracers: dict[str, Any], stencil: Stencil, *, monotone: bool = True,
    batch: int = 8, deficit: bool = False,
) -> dict[str, Any]:
    """Every grid tracer read at the stencil's departure points.

    With ``deficit=True`` each value is a ``(field, clip_deficit)`` pair.
    """
    names = list(tracers)
    if not names:
        return {}
    values = gather_batch(
        [tracers[name] for name in names], stencil,
        monotone=monotone, batch=int(batch), deficit=bool(deficit),
    )
    return dict(zip(names, values))


def _refuse_empty(name: str, delta: float) -> None:
    raise ValueError(
        f"the semi-Lagrangian mass fixer cannot restore {abs(delta):.6g} "
        f"kg/m2 of {name}: the advected field holds no mass anywhere on "
        "the sphere, so there is no air to put it back in, and spreading "
        "it uniformly would create the species in clear air"
    )


def _bermejo_conde(xp, value, low, delta: float, dp, cell):
    """The weighted stage of the Bermejo-Conde fixer.

    Returns the corrected field and the signed mass it placed.  ``value``
    is nonnegative (the floor ran first) and so is ``low``, the trilinear
    value of a nonnegative source.  The weight is one-signed: an ADDITION
    goes only where the high-order value fell below the low-order one and
    a REMOVAL only where it rose above it, so every point moves TOWARD its
    trilinear value.  Each point is capped at reaching it (``room``), so
    the corrected value stays between the two interpolants, inside the
    limiter's box and nonnegative; a point that reaches its cap drops out
    and the rest is re-spread over the others (water filling).
    """
    sign = 1.0 if delta > 0.0 else -1.0
    room = xp.maximum(sign * (low - value), 0.0)
    weight = room if BERMEJO_CONDE_BETA == 1.0 else room ** BERMEJO_CONDE_BETA
    remaining = abs(float(delta))
    placed = 0.0
    for _ in range(BERMEJO_CONDE_PASSES):
        capacity = _mass(xp, weight, dp, cell)
        if capacity <= 0.0 or remaining <= 0.0:
            break
        step = xp.minimum((remaining / capacity) * weight, room)
        moved = _mass(xp, step, dp, cell)
        value = value + sign * step
        room = room - step
        del step
        remaining -= moved
        placed += moved
        # A point that reached its trilinear value takes no more.
        weight = xp.where(room > 0.0, weight, 0.0)
    return value, sign * placed


def fix_mass(
    advected: dict[str, Any], before: dict[str, Any], dp_before, dp_after,
    transform, *, scheme: str = "bermejo_conde",
    deficits: dict[str, Any] | None = None,
    stencil: Stencil | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Restore each species' mass after a semi-Lagrangian advection.

    ``before`` are the mixing ratios the step started from and
    ``dp_before`` the thickness they were measured against; ``advected``
    are the same species at the departure points and ``dp_after`` the
    thickness of the layers they now sit in.  ``deficits`` carries the
    limiter's signed clip amount per species and is required by, and only
    read by, the additive form.  ``stencil`` is the departure stencil the
    species were gathered on; the Bermejo-Conde form reads the low-order
    interpolant of ``before`` on it, one species at a time.

    Returns the fixed tracers and, per species: the relative mass the
    fixer moved, that mass signed (positive where it put mass back), the
    fraction of it the clip deficit carried, the fraction the local
    (weighted) stage carried, the uniform relative factor the remainder
    applied to every point holding the species, and the fraction of the
    species' mass at points the fixer changed by more than
    ``TOUCHED_RELATIVE`` of their own value.
    """
    if scheme in RETIRED_TRACER_FIXERS:
        raise ValueError(
            f"the semi-Lagrangian tracer fixer {scheme!r} is retired: "
            + RETIRED_TRACER_FIXERS[scheme]
        )
    if scheme not in TRACER_FIXERS:
        raise ValueError(
            "the semi-Lagrangian tracer fixer must be one of "
            + ", ".join(repr(item) for item in TRACER_FIXERS)
            + f", got {scheme!r}"
        )
    additive = scheme == "bermejo_conde_additive"
    weighted = scheme in STENCIL_FIXERS
    if additive and deficits is None:
        raise ValueError(
            "the additive tracer fixer puts a species' mass back where the "
            "quasi-monotone limiter took it, so it cannot run without the "
            "limiter's clip deficit; gather with deficit=True, select "
            "tracer_fixer = 'bermejo_conde' and hand over the departure "
            "stencil, or select 'mass_proportional', which needs neither"
        )
    if weighted and stencil is None:
        raise ValueError(
            "the Bermejo-Conde tracer fixer weighs the correction by the "
            "difference between the high-order and the trilinear "
            "interpolant at each departure point, so it cannot run without "
            "the departure stencil; without it the weight does not exist "
            "and the correction would land on points the interpolation did "
            "not err at"
        )
    xp = transform.backend.xp
    cell = area_weights(transform)
    metrics: dict[str, float] = {}
    fixed: dict[str, Any] = {}
    for name, value in advected.items():
        # The before-state may be the tier's (a parked tracer): staged
        # one species at a time for its mass and dropped.
        source = resident(xp, before[name])
        target = _mass(xp, source, dp_before, cell)
        clamped = 0.0
        if scheme != "none":
            # The floor comes FIRST, and its mass is measured.
            #
            # Every conservative form here ends by clamping the mixing
            # ratio at zero, because a negative one is not a state the
            # model has.  Measuring the advected mass before that clamp
            # and correcting afterwards leaves the correction short by
            # exactly the negative part: the fixer then returns a field
            # whose mass is the target PLUS the mass the clamp created,
            # reports a small correction, and closes nothing.  It is
            # invisible while the limiter is on, because the gather is
            # then nonnegative and the clamp does nothing; it is the whole
            # error of the arm that runs with the limiter off, and that
            # arm is the counter-arm the limiter's value is read from.
            floored = xp.maximum(value, 0.0)
            clamped = (_mass(xp, floored, dp_after, cell)
                       - _mass(xp, value, dp_after, cell))
            value = floored
        metrics[f"semilag_tracer_positivity_clamp_kg_m2__{name}"] = float(clamped)
        after = _mass(xp, value, dp_after, cell)
        delta = target - after
        relative = abs(delta) / max(abs(target), 1.0e-30)
        metrics[f"semilag_tracer_mass_fixer_relative__{name}"] = float(relative)
        # SIGNED: positive where the fixer had to put mass back, negative
        # where it had to take mass away.  The two directions cancel across
        # species in the water budget and must not be added as magnitudes
        # there; the per-species accuracy row reads the relative number
        # above, which is a magnitude on purpose.
        metrics[f"semilag_tracer_mass_fixer_kg_m2__{name}"] = float(delta)
        metrics[f"semilag_tracer_clip_share__{name}"] = 0.0
        metrics[f"semilag_tracer_fixer_local_share__{name}"] = 0.0
        metrics[f"semilag_tracer_fixer_uniform_rescale__{name}"] = 0.0
        metrics[f"semilag_tracer_fixer_touched_mass_fraction__{name}"] = 0.0
        metrics[f"semilag_tracer_fixer_residual_relative__{name}"] = (
            0.0 if scheme != "none" else float(relative)
        )
        if scheme == "none" or delta == 0.0:
            fixed[name] = value
            continue
        start = value
        total = abs(delta)
        sign = 1.0 if delta > 0.0 else -1.0
        if additive:
            # Stage one.  The limiter's own record of where the mass went:
            # the deficit is raw minus kept, so a positive entry is mass
            # the clip removed and a negative entry mass it added.  Take
            # only the entries pointing the way the correction has to go,
            # and never take more from a point than it holds, so the stage
            # cannot drive a mixing ratio negative.
            room = xp.maximum(sign * deficits[name], 0.0)
            if sign < 0.0:
                room = xp.minimum(room, xp.maximum(value, 0.0))
            capacity = _mass(xp, room, dp_after, cell)
            if capacity > 0.0:
                # Capped at one: restoring the whole clip amount returns
                # the raw interpolated value, and past it the fixer would
                # be undoing more than the limiter ever did.
                lam = min(abs(delta) / capacity, 1.0)
                value = value + (sign * lam) * room
                share = float(lam * capacity / max(total, 1.0e-30))
                metrics[f"semilag_tracer_clip_share__{name}"] = share
                metrics[f"semilag_tracer_fixer_local_share__{name}"] = share
            del room
            value = xp.maximum(value, 0.0)
        elif weighted:
            low = gather_linear_batch([source], stencil)[0]
            value, moved = _bermejo_conde(xp, value, low, delta,
                                          dp_after, cell)
            del low
            metrics[f"semilag_tracer_fixer_local_share__{name}"] = float(
                abs(moved) / max(total, 1.0e-30)
            )
        del source
        # The uniform stage, and the whole of ``mass_proportional``.  The
        # remainder is measured afresh from the field as it now stands, so
        # whatever the stages above left (the float roundoff of their sums
        # included) closes here and the species' mass is the target.
        # Weights proportional to the mass already present, zero in air
        # that holds none of the species: a uniform RELATIVE adjustment of
        # the air that has it.  ``value`` is nonnegative here, so the
        # scaled field cannot leave the floor.
        weight = _mass(xp, value, dp_after, cell)
        rest = target - weight
        if weight == 0.0:
            if rest != 0.0:
                _refuse_empty(name, rest)
            fixed[name] = value
            continue
        factor = rest / weight
        metrics[f"semilag_tracer_fixer_uniform_rescale__{name}"] = float(
            abs(factor)
        )
        value = value * (1.0 + factor)
        # Where the correction landed: the fraction of the species' mass
        # (as advected) sitting at points the fixer changed by more than
        # TOUCHED_RELATIVE of their own value.  One for the uniform form
        # whenever its factor exceeds that, and the number that says how
        # local a weighted form actually was.
        held = _mass(xp, start, dp_after, cell)
        if held > 0.0:
            changed = xp.abs(value - start) > TOUCHED_RELATIVE * start
            metrics[
                f"semilag_tracer_fixer_touched_mass_fraction__{name}"
            ] = float(_mass(xp, xp.where(changed, start, 0.0),
                            dp_after, cell) / held)
            del changed
        del start
        # The closure the fixer exists for, read back from the field it
        # returns: the species' mass after the fix against the target, as
        # a fraction of the target.  Float roundoff of the reductions and
        # nothing else; a receipt that reads more than that has a fixer
        # that is not conserving.
        metrics[f"semilag_tracer_fixer_residual_relative__{name}"] = float(
            abs(_mass(xp, value, dp_after, cell) - target)
            / max(abs(target), 1.0e-30)
        )
        fixed[name] = value
    return fixed, metrics


__all__ = ["BERMEJO_CONDE_BETA", "DEFICIT_FIXERS", "RETIRED_TRACER_FIXERS",
           "STENCIL_FIXERS", "TRACER_FIXERS", "advect", "area_weights",
           "fix_mass"]
