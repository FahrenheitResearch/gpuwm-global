"""The two-time-level semi-Lagrangian semi-implicit step.

For each advected variable ``X``, with ``A`` an arrival point (which is a
grid point, always) and ``D`` its departure point::

    X^{n+1}_A - alpha dt (L X)^{n+1}_A = Q_X(D) + (dt/2) [2 N^n - N^{n-1}]_A
    Q_X = X^n + dt [ (1 - alpha) (L X)^n + (1/2) N^n ]

``L`` is the semi-implicit linear operator the shipped core already
carries and ``N = A - L`` the nonlinear residual of the advective-form
tendency.  This is the trapezoidal two-time-level scheme with the
arrival-side nonlinear term extrapolated to ``t^{n+1}`` (SETTLS-type) and
off-centred by ``alpha``.

For theta the material tendency ``A`` is zero (adiabatic flow), so
``N = -L theta`` and the WHOLE variable is gathered: a parcel's reference
change is then whatever the vertical stencil reads between its departure
and arrival levels, which is exactly consistent with what the same stencil
reads of the deviation from the reference.  Until 2026-09-06 the core
gathered ``theta - theta_ref(k)`` and carried the reference profile's
material tendency as a grid tendency in ``N``; :mod:`.rhs` records what
that did at the model lid and the measurement that retired it.

Pre-combining the whole departure-side bundle into ``Q_X`` before the
interpolation is what keeps this to ONE interpolated field per advected
variable: three Cartesian wind components, ``theta``, vapour, ten grid
tracers and one two-dimensional field for ``ln ps``.  Interpolating the
state and its tendencies separately would double that, and the gather is
the step's largest new cost.

The left-hand side is the shipped Helmholtz solve, unchanged.  Written
out, ``(I - tau L) y = R`` with ``tau = alpha dt`` is exactly
``VerticalModeSemiImplicit.solve_shifted``, the same routine the IMEX
integrator calls for its stage solves and with the same per-tau cached
inverse; the semi-Lagrangian core calls it ONCE per step where the IMEX
pair calls it twice.

What is NOT clipped, and why.  The quasi-monotone option bounds the
interpolated value by the values of the cell that surrounds the
departure point.  That is right for a positive-definite species: cloud
water has no business exceeding the cloud water around it.  It is wrong
for the dynamical bundle.  ``Q_u`` is a wind plus a tendency, so its
physical bounds are not its neighbours' values at all, and clipping a
wind to its neighbours removes the extremes of a field whose extremes
ARE the flow, every step, with no conservation and no measurement of
what was taken.  So the wind, ``theta'`` and ``ln ps`` are gathered
unlimited by default and vapour and the ten tracers are gathered
limited; ``[semilag] quasi_monotone_dynamics`` reaches the other arm so
the choice can be measured rather than argued.
"""
from __future__ import annotations

import numpy as np

from .halo import default_halo_rows
from .interpolate import Stencil, gather_batch
from .options import HORIZONTAL_ORDER, PHYSICS_ARRIVAL_WEIGHT
from .rhs import ReferenceProfile, advective_tendencies, linear_spectral_rows
from .state import (
    TRAJECTORY_FIELDS, TRAJECTORY_SURFACE_FIELDS, RowTrajectory, TrajectoryState,
)
from .tables import HaloEscape, SphericalGridTables
from .tracers import DEFICIT_FIXERS, STENCIL_FIXERS, area_weights, fix_mass
from .trajectory import (
    CartesianWind,
    LipschitzDiagnostics,
    TrajectoryDiagnostics,
    MAXIMUM_TRAJECTORY_ITERATIONS,
    cartesian_wind,
    converged_departure_points,
    convergence,
    departure_points,
    level_rate_from_mass_flux,
    lipschitz,
    refuse_trajectory_fold,
)
from .vectors import transport_to_arrival

from ..bands import PlaneAccumulator
from ..constants import CONDENSATE_SPECIES, GRAVITY_M_S2, GRID_TRACERS
from ..profile import profiler_of
from ..spill import resident, spilled
from ..state import MoistHybridState

#: The integrator names this package answers to.
SEMILAG_INTEGRATORS = ("sl_si",)


def grid_tables(model) -> SphericalGridTables:
    """The grid geometry of this model's transform, built once."""
    cached = getattr(model, "_semilag_tables", None)
    backend = model.transform.backend
    if cached is None or cached.dtype != backend.float_dtype:
        cached = SphericalGridTables.create(
            model.transform.grid, xp=backend.xp, dtype=backend.float_dtype
        )
        model._semilag_tables = cached
    return cached


def reference_profile(model) -> ReferenceProfile:
    """The semi-implicit reference column, built once."""
    cached = getattr(model, "_semilag_reference", None)
    if cached is None:
        cached = ReferenceProfile(model)
        model._semilag_reference = cached
    return cached


def _surface_stencil(stencil: Stencil, tables: SphericalGridTables):
    """The horizontal departure points of the LOWEST model level, as a
    four-level stencil with no vertical displacement.

    ``ln ps`` is two-dimensional and the gather is a tricubic on a volume,
    so the surface field is read through four identical levels with the
    departure level index pinned to an exact node.  Three of the four
    vertical Lagrange weights are then exactly zero and the fourth is
    exactly one, so the result is the plain bicubic, computed by the same
    compiled kernel every other field is read by rather than by a second
    interpolation with its own bracket search and its own polar map.  It
    costs a tenth of one volume field on a forty-level stack.
    """
    xp = tables.xp
    dtype = tables.dtype
    nlev, nlat, nlon = stencil.shape
    xi = xp.ascontiguousarray(
        xp.broadcast_to(stencil.xi[nlev - 1][None], (4, nlat, nlon))
    )
    phi = xp.ascontiguousarray(
        xp.broadcast_to(stencil.phi[nlev - 1][None], (4, nlat, nlon))
    )
    level = xp.ascontiguousarray(
        xp.broadcast_to(
            xp.arange(4, dtype=dtype)[:, None, None], (4, nlat, nlon)
        )
    )
    # A band's surface stencil is the band's rows and reads the band's
    # held rows, exactly as its volume stencil does.
    return Stencil(xi=xi, phi=phi, level=level, tables=tables,
                   window=stencil.window)


def _gather_surface(field, surface: Stencil, *, monotone: bool, order: int = 4):
    xp = surface.tables.xp
    _levels, nlat, nlon = surface.source_shape
    tiled = xp.ascontiguousarray(
        xp.broadcast_to(field[None], (4, nlat, nlon))
    )
    # A band's surface gather leaves its escape flag to the band's one
    # read-back (step._band_pass), like every other gather of the band.
    return gather_batch([tiled], surface, monotone=monotone, batch=1,
                        order=order,
                        defer_escape=surface.window is not None)[0][1]


def _physics_increment(model, atmosphere, pre, tables, *, riding: bool):
    """The FIRST physics half's increment, in grid space.

    Taken as a difference in SPECTRAL space and synthesized ONCE, rather
    than as the difference of two grid syntheses: the increment is five
    spectral fields and one synthesis, where the two-synthesis form would
    pay the 4.15 ms T255 stacked synthesis twice for the same numbers.
    The ten grid tracers are already grid fields and their increment is a
    plain subtraction with no transform at all.
    """
    transform = model.transform
    xp = transform.backend.xp
    nlev = int(model.nlev)
    d_u, d_v = model.vector.wind_from_vordiv(
        atmosphere.vorticity - pre.vorticity,
        atmosphere.divergence - pre.divergence,
    )
    stacked = xp.concatenate([
        atmosphere.theta - pre.theta, atmosphere.qv - pre.qv,
    ], axis=0)
    grid = model._chunked(transform.inverse, stacked)
    del stacked
    d_x, d_y, d_z = cartesian_wind(d_u, d_v, tables)
    return {
        "u": d_u, "v": d_v, "cartesian": (d_x, d_y, d_z),
        "theta": grid[:nlev], "qv": grid[nlev:],
        "lnps": transform.inverse(
            atmosphere.log_surface_pressure - pre.log_surface_pressure
        ),
        "tracers": _tracer_increment(atmosphere, pre) if riding else {},
    }


def _tracer_increment(atmosphere, pre) -> dict:
    """The physics half's increment of the ten grid tracers.

    REFUSED while the pinned host tier holds them: the physics half-step
    writes a parked tracer's slot in place (dynamics.apply_physics,
    band by band), so the state the half started from no longer exists
    as an array to subtract, and a difference taken against the slot
    would read zero on every tracer and couple no physics at all.  The
    shipped coupling ("advected") never takes this difference; the two
    that do run with the tier off.
    """
    parked = [name for name in GRID_TRACERS
              if spilled(getattr(atmosphere, name)) or spilled(getattr(pre, name))]
    if parked:
        raise ValueError(
            "semilag physics_coupling needs the grid tracers' physics "
            f"increment and the pinned host tier holds {', '.join(parked)}: "
            "the physics half-step writes a parked tracer's slot in place, "
            "so the pre-physics tracers no longer exist to subtract and the "
            "increment would read zero.  Run with [memory] host_spill = "
            "\"off\" or with [semilag] physics_coupling = \"advected\"."
        )
    return {
        name: getattr(atmosphere, name) - getattr(pre, name)
        for name in GRID_TRACERS
    }


def semilag_step(model, atmosphere: MoistHybridState, dt_s: float, mark=None):
    """One two-time-level semi-Lagrangian semi-implicit step of the
    adiabatic core, tracers included.

    Returns ``(advanced, metrics)`` in the shape
    :meth:`arwen_global.dynamics.MoistHybridModel.integrate_dynamics`
    contracts for, with the trajectory's own diagnostics beside the
    semi-implicit increment.

    A resident run (one band on one card, the shipped default) takes the
    whole-grid step.  Every other run takes the BANDED step
    (:func:`_semilag_step_banded`): the trajectory, the departure gather
    and the tendencies run one latitude band at a time behind a row halo,
    on the bands this card owns, and meet the replicated spectral state at
    the Fourier waist.  Both return the same bits (the band gates in
    tests/test_arwen_global_semilag_bands.py).
    """
    if model.pipeline.resident:
        return _semilag_step_resident(model, atmosphere, dt_s, mark=mark)
    return _semilag_step_banded(model, atmosphere, dt_s, mark=mark)


def _semilag_step_resident(model, atmosphere: MoistHybridState, dt_s: float,
                           mark=None):
    """The whole-grid step: one band covering every row, on one card."""
    options = model.semilag
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    scalar = backend.float_dtype
    dt = float(dt_s)
    alpha = float(model.semi_implicit.off_centring_weight)
    tables = grid_tables(model)
    reference = reference_profile(model)
    previous = model.trajectory_state()
    startup = previous is None
    prof = profiler_of(model)
    # How much of the first physics half's increment is moved from the
    # departure point to the arrival point (semilag.options): zero under
    # the plain Strang split, which is the default and pays nothing at
    # all, because the increment is then never separated from the state.
    arrival_weight = float(PHYSICS_ARRIVAL_WEIGHT[options.physics_coupling])
    # The dynamical bundle's horizontal stencil width (semilag.options
    # HORIZONTAL_INTERPOLATIONS); the limited species stay on the cubic.
    order = int(HORIZONTAL_ORDER[options.horizontal_interpolation])
    pre_physics = model.take_pre_physics() if arrival_weight else None
    if arrival_weight and pre_physics is None:
        raise ValueError(
            f"semilag.physics_coupling = {options.physics_coupling!r} needs "
            "the state as it stood before the first physics half, and the "
            "step did not hand one over; a coupling that silently fell back "
            "to 'advected' would produce a different run under the same "
            "config and say nothing"
        )

    with prof.section("grid_state"):
        # One set of sources feeds both the grid view and the mass-flux
        # pass, because the band loop keys its memo and its per-band
        # syntheses off the sources object rather than off a grid dict.
        sources = model.grid_sources(atmosphere)
        g, _stack, _names = model.grid_band(sources, model.whole_rows)
        _spec, _div_mass, ps_t, omega_half = model._mass_flux_and_omega(sources)

    with prof.section("trajectory"):
        level_rate = level_rate_from_mass_flux(
            omega_half, p_full=g["p_full"], xp=xp
        )
        vx, vy, vz = cartesian_wind(g["u"], g["v"], tables)
        wind = CartesianWind(vx=vx, vy=vy, vz=vz, level_rate=level_rate)
        if startup or options.extrapolation == "none":
            extrapolated = wind
        else:
            ex_u = 2.0 * g["u"] - previous.u_prev
            ex_v = 2.0 * g["v"] - previous.v_prev
            ex_x, ex_y, ex_z = cartesian_wind(ex_u, ex_v, tables)
            del ex_u, ex_v
            extrapolated = CartesianWind(
                vx=ex_x, vy=ex_y, vz=ex_z,
                level_rate=2.0 * level_rate - previous.s_prev,
            )
        # The gate reads the fold of the map the search below solves, so it
        # reads the same two winds: the extrapolated one at the arrival
        # point and the current one at the departure point.
        deformation = lipschitz(
            wind, tables, dt,
            extrapolated=None if extrapolated is wind else extrapolated,
        )
        refuse_trajectory_fold(deformation, model.minimum_fold_determinant)
        stencil, trajectory = converged_departure_points(
            wind, tables, dt,
            iterations=int(options.trajectory_iterations),
            limit_cells=options.trajectory_convergence_cells,
            extrapolated=extrapolated,
        )
        del extrapolated
        surface = _surface_stencil(stencil, tables)

    with prof.section("tendencies"):
        tendencies = advective_tendencies(
            model, atmosphere, g, omega_half, ps_t, reference
        )
        n_u = tendencies["a_u"] - tendencies["l_u"]
        n_v = tendencies["a_v"] - tendencies["l_v"]
        # Adiabatic flow: the material tendency of theta is ZERO, so its
        # nonlinear residual is minus the operator's own row and the
        # reference profile is read along the trajectory by the gather of
        # the whole variable (rhs.advective_tendencies says why the
        # reference-subtracted form was retired, with the measurement).
        n_theta = -tendencies["l_theta"]
        n_lnps = tendencies["a_lnps"] - tendencies["l_lnps"]
        half = scalar(0.5 * dt)
        lead = scalar((1.0 - alpha) * dt)
        bundle_u = g["u"] + lead * tendencies["l_u"] + half * n_u
        bundle_v = g["v"] + lead * tendencies["l_v"] + half * n_v
        bundle_theta = g["theta"] + lead * tendencies["l_theta"] + half * n_theta
        bundle_lnps = (
            g["logps"] + lead * tendencies["l_lnps"] + half * n_lnps
        )
        del tendencies
        bundle_x, bundle_y, bundle_z = cartesian_wind(
            bundle_u, bundle_v, tables
        )
        del bundle_u, bundle_v

    with prof.section("gather"):
        batch = int(options.gather_batch)
        gathered = gather_batch(
            [bundle_x, bundle_y, bundle_z, bundle_theta], stencil,
            monotone=bool(options.quasi_monotone_dynamics), batch=batch,
            order=order,
        )
        del bundle_x, bundle_y, bundle_z, bundle_theta
        departure_x, departure_y, departure_z, departure_theta = gathered
        del gathered
        # Under tracer_scheme = "flux_form" the ten tracers are left for
        # dynamics.step()'s Eulerian sweep and only vapour is gathered
        # here; the trial state carries the originals through the solve
        # by reference, exactly as it does on the IMEX path.
        riding = options.tracer_scheme == "semi_lagrangian"
        # The additive fixer puts a species' mass back where the limiter
        # took it, so it needs the limiter's own record of that: the
        # gather reports it out of arithmetic it already does, and the
        # values it returns are bit for bit the plain limited gather's.
        limiter = "quasi_monotone" if options.quasi_monotone else "none"
        want_deficit = bool(
            riding and limiter != "none"
            and options.tracer_fixer in DEFICIT_FIXERS
        )
        # The grid tracers may live in the pinned host tier (spill): the
        # gather stages a parked field one batch at a time and the copy
        # dies with the batch (interpolate.gather_batch).  MEASURED
        # 2026-09-07, T533 L40 sl_si with all three slices parked on an
        # RTX 5070 Ti: handed the slot itself, the kernel refused it by
        # type at step 1; handed all ten staged at once, the card ran out
        # at 15.96 GB inside the gather.
        # A tracer the pinned host tier holds is staged onto the card
        # here, where the gather reads it; under ``flux_form`` the ten are
        # NOT gathered and must stay parked, because they then ride through
        # the solve by reference and dynamics.step's Eulerian sweep is what
        # stages and re-parks them.  Staging them here as well would take
        # the tier off the tracers for the rest of the run.
        species = gather_batch(
            [g["qv"], *(
                (getattr(atmosphere, name) for name in GRID_TRACERS)
                if riding else ()
            )],
            stencil, monotone=limiter, batch=batch,
            deficit=want_deficit,
        )
        if want_deficit:
            departure_qv = species[0][0]
            advected = dict(zip(GRID_TRACERS,
                                (row[0] for row in species[1:])))
            deficits = dict(zip(GRID_TRACERS,
                                (row[1] for row in species[1:])))
        else:
            departure_qv = species[0]
            advected = (
                dict(zip(GRID_TRACERS, species[1:])) if riding
                else atmosphere.grid_tracers()
            )
            deficits = None
        del species
        departure_lnps = _gather_surface(
            bundle_lnps, surface, monotone=bool(options.quasi_monotone_dynamics),
            order=order,
        )
        del bundle_lnps
        arrival_u, arrival_v = transport_to_arrival(
            departure_x, departure_y, departure_z, stencil, tables
        )
        del departure_x, departure_y, departure_z

    with prof.section("physics_coupling"):
        # The first physics half ran at the grid points, which are the
        # arrival points, and its increment went into the state BEFORE the
        # gather, so under the default coupling a parcel reads it at its
        # departure point along with everything else.  The other two
        # couplings move a fraction of it to the arrival point, by taking
        # it back out where the gather put it and adding it where the
        # parcel lands:
        #
        #     X = gather(X_pre + I + ...)  +  w * (I_A - gather(I))
        #
        # with w = 1 for "arrival" and 1/2 for "trajectory_average".  What
        # it buys is that a field the physics has just created is not
        # interpolated on the step that created it; what it costs is one
        # synthesis of the increment and one more gathered field per
        # advected variable, and, for a physics SINK, that the removal
        # lands where it was not computed, which the positivity repair
        # then has to hold.  That is why the default is the plain split.
        if arrival_weight:
            inc = _physics_increment(
                model, atmosphere, pre_physics, tables, riding=riding
            )
            del pre_physics
            names = list(inc["tracers"])
            # The dynamical rows read the bundle's own stencil width and
            # the species rows the cubic, so what is taken back out is
            # what the gather put in, field by field.
            rows = gather_batch(
                [*inc["cartesian"], inc["theta"]], stencil,
                monotone=False, batch=batch, order=order,
            ) + gather_batch(
                [inc["qv"], *(inc["tracers"][name] for name in names)],
                stencil, monotone=False, batch=batch,
            )
            gathered_u, gathered_v = transport_to_arrival(
                rows[0], rows[1], rows[2], stencil, tables
            )
            weight = scalar(arrival_weight)
            arrival_u = arrival_u + weight * (inc["u"] - gathered_u)
            arrival_v = arrival_v + weight * (inc["v"] - gathered_v)
            del gathered_u, gathered_v
            departure_theta = departure_theta + weight * (
                inc["theta"] - rows[3]
            )
            departure_qv = departure_qv + weight * (inc["qv"] - rows[4])
            for index, name in enumerate(names):
                advected[name] = advected[name] + weight * (
                    inc["tracers"][name] - rows[5 + index]
                )
            departure_lnps = departure_lnps + weight * (
                inc["lnps"] - _gather_surface(
                    inc["lnps"], surface, monotone=False, order=order
                )
            )
            del inc, rows, names
    del surface
    # The Bermejo-Conde tracer fixer reads the trilinear interpolant of each
    # species on this stencil after the solve (tracers.fix_mass), one
    # species at a time.  Holding the three coordinate arrays across the
    # solve costs less than holding a second gathered array per species,
    # which is what reporting the weight out of the gather would cost.
    keep_stencil = bool(riding and options.tracer_fixer in STENCIL_FIXERS)
    if not keep_stencil:
        stencil = None

    with prof.section("assemble"):
        if startup or options.extrapolation == "none":
            arrival_u = arrival_u + half * n_u
            arrival_v = arrival_v + half * n_v
            departure_theta = departure_theta + half * n_theta
            departure_lnps = departure_lnps + half * n_lnps
        else:
            arrival_u = arrival_u + half * (2.0 * n_u - previous.n_u)
            arrival_v = arrival_v + half * (2.0 * n_v - previous.n_v)
            departure_theta = departure_theta + half * (
                2.0 * n_theta - previous.n_theta
            )
            departure_lnps = departure_lnps + half * (
                2.0 * n_lnps - previous.n_lnps
            )
        trial_zeta, trial_divergence = model.vector.vordiv_from_wind(
            arrival_u, arrival_v
        )
        del arrival_u, arrival_v
        stacked = xp.concatenate([departure_theta, departure_qv], axis=0)
        del departure_theta, departure_qv
        analysed = model._chunked(
            lambda block: transform.project(transform.forward(block)), stacked
        )
        del stacked
        nlev = int(model.nlev)
        trial = MoistHybridState(
            vorticity=trial_zeta,
            divergence=trial_divergence,
            theta=analysed[:nlev],
            log_surface_pressure=transform.project(
                transform.forward(departure_lnps)
            ),
            qv=analysed[nlev:],
            **advected,
            time_s=atmosphere.time_s,
            step=atmosphere.step,
        )
        del analysed, departure_lnps, trial_zeta, previous
        if mark is not None:
            mark("semilag_transport", trial)

    with prof.section("solve"):
        advanced = model.semi_implicit.solve_shifted(
            trial, transform, model.vertical, alpha * dt
        )
        increment = float(
            xp.max(xp.abs(advanced.divergence - trial_divergence))
        )
        del trial_divergence

    with prof.section("tracers"):
        del trial
        fixer: dict[str, float] = {}
        water_relative = 0.0
        if riding:
            after = model.grid_state(advanced, only=("dp",))
            # The mass fixer measures the advected species against the
            # species as they stood, and the water reading below reads the
            # same ones, so the pinned host tier is read ONCE here and the
            # staged copy serves both.  It is read here rather than held
            # from the gather because ten grid volumes are 1.9 GiB at T533
            # and holding them across the trajectory and the solve would
            # put them beside the step's own peak; two lines of the same
            # section is a lifetime the peak does not see.  A resident run
            # gets its own arrays back and copies nothing.
            before = {name: resident(xp, value) for name, value
                      in atmosphere.grid_tracers().items()}
            fixed, fixer = fix_mass(
                advected, before, g["dp"], after["dp"],
                transform, scheme=options.tracer_fixer, deficits=deficits,
                stencil=stencil,
            )
            # The advanced ten stay on the card from here to the second
            # physics half, which returns them to the tier's own slots
            # (dynamics._physics_result_to_state, and the sponge-only pass
            # on the dry route).  Parking them here as well would write
            # 1.9 GiB back at T533 and read it again a few operators
            # later, for a peak that is unchanged: the step's peak is in
            # the physics half, where one device copy of the ten exists
            # either way.
            advanced = advanced.with_grid_tracers(fixed)
            # The fixer's NET correction as a fraction of the atmosphere's
            # own water, which is the quantity the breakage is about: a
            # correction of four percent of a species whose global mass is
            # a millionth of the column water is four hundredths of a
            # millionth of the water, and the per-species relative number
            # on its own cannot say that.  The sum is SIGNED, because a
            # species the fixer fed and a species it starved are the same
            # water moving between them and not two losses; a per-species
            # magnitude belongs to the per-species row.  Number moments are
            # counts per kilogram, not water, and are not in this sum.  The
            # 1/g that turns a Pa-weighted mass into kg/m2 divides both
            # sides and is written on both so the two are the same quantity.
            column = _water_column_total(transform, _water_column_rows(
                xp, g["qv"], before, g["dp"]))
            moved = sum(
                value for name, value in fixer.items()
                if name.startswith("semilag_tracer_mass_fixer_kg_m2__")
                and name.rsplit("__", 1)[1] in CONDENSATE_SPECIES
            ) / GRAVITY_M_S2
            water_relative = abs(moved) / max(column, 1.0e-30)
            del fixed, after, before
        del advected, deficits, stencil

    model.set_trajectory_state(TrajectoryState(
        u_prev=g["u"], v_prev=g["v"], s_prev=level_rate,
        n_u=n_u, n_v=n_v, n_theta=n_theta, n_lnps=n_lnps,
    ))
    del g, n_u, n_v, n_theta, n_lnps, level_rate, wind, vx, vy, vz

    if mark is not None:
        mark("semilag_implicit", advanced)
    # The largest RELATIVE correction, over species.  Named explicitly
    # rather than taken as the maximum of the fixer's whole record: that
    # record also carries the correction in kg/m2 and the clip share, and a
    # number moment's correction in counts per kilogram is a number like
    # 2.6e9, which a bare max over the values reports as a relative
    # magnitude of 2.6e9 and fails every gate that reads it.
    fixer_max = max(
        (value for name, value in fixer.items()
         if name.startswith("semilag_tracer_mass_fixer_relative__")),
        default=0.0,
    )
    # The row dynamics.step() reads where the flux-form sweep's own
    # metrics would be.  The names are kept so a receipt of either arm
    # has the same shape; ``scheme`` says which produced them and the
    # values mean what they say.  A semi-Lagrangian gather crosses the
    # cells it crosses in ONE pass -- that is the whole point -- so the
    # sub-cycle counts are one and the displacement is reported under the
    # Courant names it is the semi-Lagrangian analogue of.  There is no
    # pseudo-density here and therefore no second continuity
    # discretization to disagree with the first, so that gap is zero by
    # construction rather than by measurement.
    transport = {
        "scheme": "semi_lagrangian",
        "max_courant_x": float(trajectory.displacement_max_cells),
        "max_courant_y": float(trajectory.displacement_max_cells),
        "max_courant_z": float(trajectory.displacement_max_levels),
        "substeps_x": 1, "substeps_y": 1, "substeps_z": 1,
        "floor_clip_kg_m2": 0.0,
        "pseudo_density_mismatch_relative": 0.0,
        "pseudo_density_mismatch_mean_relative": 0.0,
        "mass_fixer_max_relative": float(fixer_max),
        **fixer,
    }
    metrics = {
        **({"tracer_transport": transport} if riding else {}),
        "semi_implicit_max_divergence_increment_s1": increment,
        "semilag_startup_step": bool(startup),
        "semilag_off_centring_weight": alpha,
        "semilag_tracer_mass_fixer_relative": float(fixer_max),
        "semilag_tracer_mass_fixer_water_relative": float(water_relative),
        "semilag_tracer_limiter": limiter,
        **fixer,
        **deformation.as_dict(),
        **trajectory.as_dict(),
        # One when the configured search missed the convergence test and
        # this step was searched again at the most iterations
        # (trajectory.converged_departure_points), zero otherwise.
        "semilag_trajectory_retried": float(
            trajectory.iterations != int(options.trajectory_iterations)),
    }
    return advanced, metrics


def _water_column_rows(xp, qv, before, dp):
    """The band-local stage of the fixer's water reading: the column's
    vapour and condensate mass per unit area times g, one value per
    column of the rows handed in (a level sum, so column-local).

    The whole stage is :func:`_water_column_total`, one weighted sum over
    the assembled plane in grid order, so a run that presents the rows a
    band at a time reduces the same operand and reads the same number
    (bands.PlaneAccumulator states the contract).
    """
    return xp.sum(
        (qv + sum(before[name] for name in CONDENSATE_SPECIES)) * dp, axis=0
    )


def _water_column_total(transform, plane) -> float:
    xp = transform.backend.xp
    return float(xp.sum(plane * area_weights(transform)[0])) / GRAVITY_M_S2


def _previous(previous, name: str, rows: slice, xp):
    """The previous level's ``name`` at the band's own ``rows``: from a
    whole-grid level, or from this card's own rows of one."""
    from ..dynamics import band_view

    if isinstance(previous, RowTrajectory):
        return previous.rows(name, rows, xp)
    return band_view(xp, getattr(previous, name), rows)


def _fold_max(values) -> float:
    """The largest of ``values``, in the order given.

    Maximum is exact in floating point in any order, so folding the band
    maxima reads the number the whole-grid maximum reads.
    """
    out = None
    for value in values:
        out = float(value) if out is None else max(out, float(value))
    return 0.0 if out is None else out


def _fold_across_cards(exchange, xp, values: list[float]) -> list[float]:
    """The card-wide maxima of one card's band maxima (exact, any order)."""
    if exchange is None or int(getattr(exchange, "world", 1)) <= 1:
        return values
    folded = exchange.fold(np, np.asarray(values, dtype=np.float64), "max",
                           name="semilag_maxima")
    return [float(v) for v in np.asarray(folded).reshape(-1)]


#: The per-second Lipschitz maxima, and the trajectory maxima, in the order
#: the band fold carries them.
_LIPSCHITZ_MAXIMA = ("jacobian_spectral_s", "jacobian_frobenius_s",
                     "jacobian_horizontal_s", "jacobian_vertical_s",
                     "jacobian_balanced_s")
_TRAJECTORY_MAXIMA = ("move_max_m", "move_max_cells", "move_max_levels",
                      "displacement_max_m", "displacement_max_cells",
                      "displacement_max_levels")


#: The fold determinants, which the band fold carries as minima.
_FOLD_MINIMA = ("fold_determinant_arrival", "fold_determinant_departure")


class _BandsResult:
    """Every band of one card's pass over the step at one search count."""


def _previous_wind_with_neighbour_rows(previous, exchange, xp, nlat: int):
    """The previous level's wind and level rate on this card's rows and
    one row either side, for the fold gate's meridional derivative of the
    extrapolated wind at a card's first and last band.

    One row of three fields from each neighbouring card (the row exchange's
    neighbour-only halo); the rest of the level stays this card's own rows
    (:class:`RowTrajectory`), so the cost is two rows, not the globe.
    """
    if not isinstance(previous, RowTrajectory):
        return previous
    first, last = int(previous.first), int(previous.last)
    lo, hi = max(first - 1, 0), min(last + 1, nlat)
    arrays = {}
    for name in ("u_prev", "v_prev", "s_prev"):
        value = previous.arrays[name]
        whole = xp.zeros((*value.shape[:-2], nlat, value.shape[-1]),
                         dtype=value.dtype)
        whole[..., first:last, :] = value
        exchange.halo(xp, whole, whole.ndim - 2, 1,
                      name=f"semilag_previous_{name}")
        arrays[name] = xp.ascontiguousarray(whole[..., lo:hi, :])
        del whole
    return RowTrajectory(first=lo, last=hi, nlat=nlat, arrays=arrays)


def _folded_lipschitz(bands, dt: float, reference_m: float, maxima,
                      minima):
    """:class:`LipschitzDiagnostics` from the folded per-second maxima and
    fold-determinant minima, by the arithmetic :func:`trajectory.lipschitz`
    applies to its own."""
    sm, fm, hm, vm, bm = maxima
    arrival, departure = minima
    gated = max(hm, vm, bm)
    return LipschitzDiagnostics(
        dt_s=dt,
        reference_length_m=reference_m,
        lipschitz=dt * gated,
        lipschitz_frobenius=dt * fm,
        lipschitz_horizontal=dt * hm,
        jacobian_spectral_s=sm,
        jacobian_frobenius_s=fm,
        jacobian_horizontal_s=hm,
        lipschitz_vertical=dt * vm,
        jacobian_vertical_s=vm,
        lipschitz_mixed=dt * sm,
        lipschitz_balanced=dt * bm,
        jacobian_balanced_s=bm,
        fold_determinant_arrival=arrival,
        fold_determinant_departure=departure,
    )


def _assembled_dp(model, pressure_sources):
    """The whole grid's layer thickness, built a band at a time from the
    replicated surface-pressure plane.

    ``vertical.pressure`` is column-local, so each band's rows are the rows
    the whole call writes; assembling them costs one band's four pressure
    volumes at a time instead of the globe's (1.8 GiB at T799 float32).
    Every band of the schedule is built on every card, because the plane
    is replicated and the fixer reads the whole grid.
    """
    xp = model.transform.backend.xp
    nlat, nlon = model.transform.grid.shape
    out = xp.empty((model.nlev, nlat, nlon),
                   dtype=model.transform.backend.float_dtype)
    for rows in model.pipeline.slices():
        out[:, rows] = model._pressure_band(pressure_sources, rows)["dp"]
    return out


def _contract_chunked(transform, waist, limit: int):
    """``project(forward(.))`` of a stacked waist in chunks of ``limit``
    leading rows: the contractions :meth:`MoistHybridModel._chunked` hands
    the resident analysis, chunk for chunk (the chunk is the Legendre
    GEMM's M dimension, so it is arithmetic and not layout)."""
    xp = transform.backend.xp
    values = waist.take()
    count = int(values.shape[0])
    limit = max(1, int(limit))
    if count <= limit:
        return transform.project(transform.contract_values(values))
    out = None
    for start in range(0, count, limit):
        piece = transform.project(
            transform.contract_values(values[start:start + limit])
        )
        if out is None:
            out = xp.empty((count, *piece.shape[1:]), dtype=piece.dtype)
        out[start:start + piece.shape[0]] = piece
        del piece
    del values
    return out


class _BandResult:
    """What one band of the banded step hands back: its rows of every
    output, and its maxima.  Nothing is written anywhere until the band
    has finished without escaping its halo."""

    __slots__ = ("arrival_u", "arrival_v", "theta", "qv", "lnps",
                 "advected", "deficits", "trajectory", "water_rows",
                 "lipschitz", "maxima", "stencil")


def _band_pass(model, atmosphere, sources, mass_grid, window, *, tables,
               options, previous, previous_wind, extrapolate, gradient_waists,
               lnps_gradient_waists, linear_waist, increment, riding,
               limiter, want_deficit, keep_stencil, iterations, order, batch,
               dt, scalar):
    """One latitude band of the banded step, behind ``window``'s halo.

    Every expression below is :func:`_semilag_step_resident`'s, evaluated
    on the band's held rows (``window.source``) where it feeds the gather
    and on the band's own rows (``window.arrival``) where it is read at the
    arrival point.  Raises :class:`HaloEscape` when a departure point
    leaves the held rows.
    """
    from ..dynamics import band_view

    transform = model.transform
    xp = transform.backend.xp
    held = window.source
    rows = window.arrival
    own = window.local

    g, _stack, _names = model.grid_band(sources, held)
    _div_mass, ps_t, omega_half = model._mass_flux_band(mass_grid, held)

    def mine(value):
        """The band's own rows of a held-rows array, as a contiguous slab
        (a reduction over a strided band reads different bits: see
        dynamics.band_view)."""
        return xp.ascontiguousarray(value[..., own, :])

    level_rate = level_rate_from_mass_flux(
        omega_half, p_full=g["p_full"], xp=xp)
    vx, vy, vz = cartesian_wind(g["u"], g["v"], tables, rows=held)
    wind = CartesianWind(vx=vx, vy=vy, vz=vz, level_rate=level_rate)
    u_own = mine(g["u"])
    v_own = mine(g["v"])
    s_own = mine(level_rate)
    if extrapolate:
        # The extrapolated wind on the band's rows and one row either side:
        # the search reads it at the arrival point (the band's rows), and
        # the fold gate's meridional derivative reads the neighbour rows
        # too, exactly as the whole-grid gate reads them.
        nlat = int(tables.nlat)
        ext = slice(max(rows.start - 1, 0), min(rows.stop + 1, nlat))
        if ext.start < held.start or ext.stop > held.stop:
            raise HaloEscape(window, "extrapolated wind")
        ext_local = slice(ext.start - held.start, ext.stop - held.start)
        ex_window = tables.window(ext, rows)
        ex_u = (2.0 * xp.ascontiguousarray(g["u"][:, ext_local])
                - _previous(previous_wind, "u_prev", ext, xp))
        ex_v = (2.0 * xp.ascontiguousarray(g["v"][:, ext_local])
                - _previous(previous_wind, "v_prev", ext, xp))
        ex_x, ex_y, ex_z = cartesian_wind(ex_u, ex_v, tables, rows=ext)
        del ex_u, ex_v
        ex_rate = (2.0 * xp.ascontiguousarray(level_rate[:, ext_local])
                   - _previous(previous_wind, "s_prev", ext, xp))
        extrapolated_ext = CartesianWind(
            vx=ex_x, vy=ex_y, vz=ex_z, level_rate=ex_rate)
        del ex_x, ex_y, ex_z, ex_rate
        extrapolated = CartesianWind(*(
            xp.ascontiguousarray(field[:, ex_window.local])
            for field in extrapolated_ext.arrays()
        ))
    else:
        extrapolated_ext = ex_window = extrapolated = None
    # The gate reads the fold of the map the search solves, so it reads
    # the same two winds the search reads (the resident step's statement).
    deformation = lipschitz(wind, tables, dt, window=window,
                            extrapolated=extrapolated_ext,
                            extrapolated_window=ex_window)
    del extrapolated_ext
    # One escape flag for the band's kernels, read once at the band's end:
    # a read-back per launch stalls the stream four times a band.
    window.clear_escape()
    stencil, trajectory = departure_points(
        wind, tables, dt,
        iterations=int(iterations),
        extrapolated=extrapolated, window=window, defer_escape=True,
    )
    del extrapolated, wind, vx, vy, vz
    surface = _surface_stencil(stencil, tables)

    # The tendencies, on the held rows: the gradient and the linear rows
    # drain their waists there, and everything else is the band's columns.
    east, north = transform.gradient_band(
        gradient_waists, held.start, held.stop)
    l_u = -east[1]
    l_v = -north[1]
    grad_phi_east = east[0]
    grad_phi_north = north[0]
    grad_lnps_east, grad_lnps_north = transform.gradient_band(
        lnps_gradient_waists, held.start, held.stop)
    linear = transform.waist_band_to_grid(linear_waist, held.start, held.stop)
    l_theta = linear[:-1]
    l_lnps = linear[-1]
    del linear
    factor = model._pressure_gradient_factor(g["ps"], g["p_half"])
    force = model.gas_constant * g["virtual_temperature"] * factor
    del factor
    coriolis = model.coriolis[held]
    a_u = coriolis * g["v"] - grad_phi_east - force * grad_lnps_east
    a_v = -coriolis * g["u"] - grad_phi_north - force * grad_lnps_north
    del force, grad_phi_east, grad_phi_north, east, north
    a_lnps = (
        ps_t / g["ps"]
        + g["u"][-1] * grad_lnps_east
        + g["v"][-1] * grad_lnps_north
    )
    del grad_lnps_east, grad_lnps_north

    n_u = a_u - l_u
    n_v = a_v - l_v
    n_theta = -l_theta
    n_lnps = a_lnps - l_lnps
    half = scalar(0.5 * dt)
    alpha = float(model.semi_implicit.off_centring_weight)
    lead = scalar((1.0 - alpha) * dt)
    bundle_u = g["u"] + lead * l_u + half * n_u
    bundle_v = g["v"] + lead * l_v + half * n_v
    bundle_theta = g["theta"] + lead * l_theta + half * n_theta
    bundle_lnps = g["logps"] + lead * l_lnps + half * n_lnps
    del a_u, a_v, a_lnps, l_u, l_v, l_theta, l_lnps
    bundle_x, bundle_y, bundle_z = cartesian_wind(
        bundle_u, bundle_v, tables, rows=held)
    del bundle_u, bundle_v

    gathered = gather_batch(
        [bundle_x, bundle_y, bundle_z, bundle_theta], stencil,
        monotone=bool(options.quasi_monotone_dynamics), batch=batch,
        order=order, defer_escape=True,
    )
    del bundle_x, bundle_y, bundle_z, bundle_theta
    departure_x, departure_y, departure_z, departure_theta = gathered
    del gathered
    species = gather_batch(
        [g["qv"], *(
            (band_view(xp, getattr(atmosphere, name), held)
             for name in GRID_TRACERS)
            if riding else ()
        )],
        stencil, monotone=limiter, batch=batch, deficit=want_deficit,
        defer_escape=True,
    )
    if want_deficit:
        departure_qv = species[0][0]
        advected = dict(zip(GRID_TRACERS, (row[0] for row in species[1:])))
        deficits = dict(zip(GRID_TRACERS, (row[1] for row in species[1:])))
    else:
        departure_qv = species[0]
        advected = dict(zip(GRID_TRACERS, species[1:])) if riding else {}
        deficits = None
    del species
    departure_lnps = _gather_surface(
        bundle_lnps, surface, monotone=bool(options.quasi_monotone_dynamics),
        order=order,
    )
    del bundle_lnps
    arrival_u, arrival_v = transport_to_arrival(
        departure_x, departure_y, departure_z, stencil, tables
    )
    del departure_x, departure_y, departure_z

    if increment is not None:
        weight_value, inc = increment
        names = list(inc["tracers"])
        rows_gathered = gather_batch(
            [*(band_view(xp, part, held) for part in inc["cartesian"]),
             band_view(xp, inc["theta"], held)], stencil,
            monotone=False, batch=batch, order=order, defer_escape=True,
        ) + gather_batch(
            [band_view(xp, inc["qv"], held),
             *(band_view(xp, inc["tracers"][name], held) for name in names)],
            stencil, monotone=False, batch=batch, defer_escape=True,
        )
        gathered_u, gathered_v = transport_to_arrival(
            rows_gathered[0], rows_gathered[1], rows_gathered[2], stencil,
            tables,
        )
        weight = scalar(weight_value)
        arrival_u = arrival_u + weight * (
            band_view(xp, inc["u"], rows) - gathered_u)
        arrival_v = arrival_v + weight * (
            band_view(xp, inc["v"], rows) - gathered_v)
        del gathered_u, gathered_v
        departure_theta = departure_theta + weight * (
            band_view(xp, inc["theta"], rows) - rows_gathered[3]
        )
        departure_qv = departure_qv + weight * (
            band_view(xp, inc["qv"], rows) - rows_gathered[4])
        for index, name in enumerate(names):
            advected[name] = advected[name] + weight * (
                band_view(xp, inc["tracers"][name], rows)
                - rows_gathered[5 + index]
            )
        departure_lnps = departure_lnps + weight * (
            band_view(xp, inc["lnps"], rows) - _gather_surface(
                band_view(xp, inc["lnps"], held), surface, monotone=False,
                order=order,
            )
        )
        del rows_gathered, names
    del surface
    # The Bermejo-Conde fixer reads the trilinear interpolant at these
    # departure points once the globe is assembled (tracers.fix_mass), so
    # the band hands its coordinates back for the step to lay into the
    # whole stencil the resident step keeps.
    band_stencil = ((stencil.xi, stencil.phi, stencil.level)
                    if keep_stencil else None)
    del stencil

    n_u_own = mine(n_u)
    n_v_own = mine(n_v)
    n_theta_own = mine(n_theta)
    n_lnps_own = mine(n_lnps)
    del n_u, n_v, n_theta, n_lnps
    if not extrapolate:
        arrival_u = arrival_u + half * n_u_own
        arrival_v = arrival_v + half * n_v_own
        departure_theta = departure_theta + half * n_theta_own
        departure_lnps = departure_lnps + half * n_lnps_own
    else:
        arrival_u = arrival_u + half * (
            2.0 * n_u_own - _previous(previous, "n_u", rows, xp))
        arrival_v = arrival_v + half * (
            2.0 * n_v_own - _previous(previous, "n_v", rows, xp))
        departure_theta = departure_theta + half * (
            2.0 * n_theta_own - _previous(previous, "n_theta", rows, xp)
        )
        departure_lnps = departure_lnps + half * (
            2.0 * n_lnps_own - _previous(previous, "n_lnps", rows, xp)
        )

    window.raise_if_escaped("step")
    result = _BandResult()
    result.arrival_u = arrival_u
    result.arrival_v = arrival_v
    result.theta = departure_theta
    result.qv = departure_qv
    result.lnps = departure_lnps
    result.advected = advected
    result.deficits = deficits
    result.trajectory = {
        "u_prev": u_own, "v_prev": v_own, "s_prev": s_own,
        "n_u": n_u_own, "n_v": n_v_own, "n_theta": n_theta_own,
        "n_lnps": n_lnps_own,
    }
    result.water_rows = None
    if riding:
        before = {name: band_view(xp, getattr(atmosphere, name), rows)
                  for name in CONDENSATE_SPECIES}
        result.water_rows = _water_column_rows(
            xp, mine(g["qv"]), before, mine(g["dp"]))
        del before
    result.lipschitz = deformation
    result.maxima = trajectory
    result.stencil = band_stencil
    del g, level_rate
    return result


def _semilag_step_banded(model, atmosphere: MoistHybridState, dt_s: float,
                         mark=None):
    """The semi-Lagrangian step a latitude band at a time.

    THE SPLIT.  Spectral space is replicated on every card and grid space
    is partitioned by the band schedule (bands.BandPipeline, a pure
    function of ``(nlat, bands)``).  Each band this card owns synthesizes
    its own rows PLUS a row halo out of the replicated spectral state (the
    synthesis contracts whole and drains any rows it is asked for, so a
    halo row costs an inverse FFT and no wire) and computes its departure
    points, its gather and its tendencies there.  Its arrival
    rows go into the three Fourier waists the analysis contracts whole on
    every card (``K = nlat``), and the waist close is where the cards
    exchange rows.  The grid tracers are held whole on every card, so a
    band reads its halo rows of them locally; the band's advected rows are
    gathered across the cards once, before the mass fixer reads the globe.

    WHAT IS NOT MOVED.  Every value a band computes is the value the
    whole-grid step computes at that point: the kernels read the whole
    grid's weights and only re-address the rows (semilag.tables
    .BandWindow), the syntheses are the whole contraction drained a band
    at a time, and every reduction is a maximum (exact in any order) or
    runs on the assembled globe.  So the answer is a function of the
    schedule and not of the card count, and one card equals P cards.

    THE HALO.  :func:`arwen_global.semilag.halo.default_halo_rows` sizes
    it; a departure point that leaves it is refused by name at the kernel
    (:class:`HaloEscape`) and the band is recomputed behind a halo twice as
    wide, up to the whole grid, which cannot escape.
    """
    options = model.semilag
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    scalar = backend.float_dtype
    dt = float(dt_s)
    alpha = float(model.semi_implicit.off_centring_weight)
    tables = grid_tables(model)
    reference = reference_profile(model)
    previous = model.trajectory_held()
    startup = previous is None
    prof = profiler_of(model)
    pipeline = model.pipeline
    exchange = pipeline.exchange
    nlev = int(model.nlev)
    nlat, nlon = transform.grid.shape
    arrival_weight = float(PHYSICS_ARRIVAL_WEIGHT[options.physics_coupling])
    order = int(HORIZONTAL_ORDER[options.horizontal_interpolation])
    pre_physics = model.take_pre_physics() if arrival_weight else None
    if arrival_weight and pre_physics is None:
        raise ValueError(
            f"semilag.physics_coupling = {options.physics_coupling!r} needs "
            "the state as it stood before the first physics half, and the "
            "step did not hand one over; a coupling that silently fell back "
            "to 'advected' would produce a different run under the same "
            "config and say nothing"
        )
    riding = options.tracer_scheme == "semi_lagrangian"
    limiter = "quasi_monotone" if options.quasi_monotone else "none"
    want_deficit = bool(
        riding and limiter != "none"
        and options.tracer_fixer in DEFICIT_FIXERS
    )
    batch = int(options.gather_batch)
    extrapolate = not (startup or options.extrapolation == "none")
    half = scalar(0.5 * dt)

    with prof.section("grid_state"):
        sources = model.grid_sources(atmosphere)
        _spectral_mass, mass_grid = model._mass_flux_sources(sources)

    with prof.section("tendencies"):
        # The replicated halves of the tendencies, contracted once on every
        # card: the operator's linear rows and the two gradients.  The one
        # analysis among them, the geopotential's, is fed by this card's
        # own bands and closed across the cards.
        phi_l, x_t = linear_spectral_rows(model, atmosphere, reference)
        linear_waist = transform.contract_to_waist(
            transform.project(x_t), bands=pipeline.bands)
        del x_t
        phi_l_spectral = transform.project(phi_l)
        del phi_l
        geopotential_waist = transform.open_waist(
            (nlev,), bands=pipeline.bands)
        # The geopotential's own closure, read off the same syntheses: the
        # wind drains are not paid for a pass that does not read the wind.
        from ..dynamics import GridSources

        geopotential_sources = GridSources(
            atmosphere, model._grid_closure(("geopotential",))[0],
            stack=sources.stack, pressure=sources.pressure,
        )
        for rows in pipeline.local_slices():
            g, _stack, _names = model.grid_band(geopotential_sources, rows)
            geopotential_waist.fill_band(
                rows.start, rows.stop, g["geopotential"])
            del g
        del geopotential_sources
        geopotential_spectral = transform.project(
            transform.contract_waist(geopotential_waist.close()))
        del geopotential_waist
        gradient_waists = transform.gradient_waists(
            xp.stack([geopotential_spectral, phi_l_spectral]),
            bands=pipeline.bands,
        )
        del geopotential_spectral, phi_l_spectral
        lnps_gradient_waists = transform.gradient_waists(
            atmosphere.log_surface_pressure, bands=pipeline.bands)

    increment = None
    if arrival_weight:
        with prof.section("physics_coupling"):
            increment = (arrival_weight, _physics_increment(
                model, atmosphere, pre_physics, tables, riding=riding))
            del pre_physics

    # The next level: whole on one card, this card's own rows on several
    # (semilag.state.RowTrajectory).
    first, last = pipeline.local_rows()
    split = exchange is not None and int(getattr(exchange, "world", 1)) > 1
    held_rows = (last - first) if split else nlat
    offset = first if split else 0
    keep_stencil = bool(riding and options.tracer_fixer in STENCIL_FIXERS)
    # The fold gate reads the extrapolated wind one row past each band, so
    # on several cards the previous level's wind needs the neighbouring
    # card's edge row (one row, three fields, at the step's start).
    previous_wind = previous
    if extrapolate and split:
        previous_wind = _previous_wind_with_neighbour_rows(
            previous, exchange, xp, nlat)
    halo = default_halo_rows(nlat, dt, transform.grid.radius_m)

    def run_bands(iterations: int):
        """Every band this card owns, searched at ``iterations``."""
        out = _BandsResult()
        out.uv_waist = transform.open_waist((2, nlev), bands=pipeline.bands)
        out.stack_waist = transform.open_waist((2 * nlev,), bands=pipeline.bands)
        out.lnps_waist = transform.open_waist((), bands=pipeline.bands)
        out.trajectory_rows = {
            name: xp.zeros(
                (held_rows, nlon) if name in TRAJECTORY_SURFACE_FIELDS
                else (nlev, held_rows, nlon), dtype=scalar)
            for name in TRAJECTORY_FIELDS
        }
        out.advected = ({name: xp.zeros((nlev, nlat, nlon), dtype=scalar)
                         for name in GRID_TRACERS} if riding else None)
        out.deficits = ({name: xp.zeros((nlev, nlat, nlon), dtype=scalar)
                         for name in GRID_TRACERS} if want_deficit else None)
        out.stencil = ([xp.zeros((nlev, nlat, nlon), dtype=scalar)
                        for _ in range(3)] if keep_stencil else None)
        out.water = (PlaneAccumulator(xp, (nlat, nlon), scalar,
                                      name="semilag_fixer_water_column",
                                      exchange=exchange)
                     if riding else None)
        out.lipschitz_bands = []
        out.trajectory_bands = []
        out.widest = 0
        for rows in pipeline.local_slices():
            width = halo
            while True:
                held, _lead, _trail = pipeline.halo(rows, width)
                window = tables.window(held, rows)
                try:
                    band = _band_pass(
                        model, atmosphere, sources, mass_grid, window,
                        tables=tables, options=options, previous=previous,
                        previous_wind=previous_wind,
                        extrapolate=extrapolate,
                        gradient_waists=gradient_waists,
                        lnps_gradient_waists=lnps_gradient_waists,
                        linear_waist=linear_waist, increment=increment,
                        riding=riding, limiter=limiter,
                        want_deficit=want_deficit, keep_stencil=keep_stencil,
                        iterations=iterations, order=order, batch=batch,
                        dt=dt, scalar=scalar,
                    )
                    break
                except HaloEscape:
                    if held.start == 0 and held.stop == nlat:
                        raise
                    width = min(2 * width, nlat)
            out.widest = max(out.widest, width)
            out.uv_waist.fill_band(
                rows.start, rows.stop, xp.stack([band.arrival_u, band.arrival_v]))
            out.stack_waist.fill_band(
                rows.start, rows.stop,
                xp.concatenate([band.theta, band.qv], axis=0))
            out.lnps_waist.fill_band(rows.start, rows.stop, band.lnps)
            mine = slice(rows.start - offset, rows.stop - offset)
            for name, value in band.trajectory.items():
                out.trajectory_rows[name][..., mine, :] = value
            if riding:
                for name, value in band.advected.items():
                    out.advected[name][:, rows] = value
                if want_deficit:
                    for name, value in band.deficits.items():
                        out.deficits[name][:, rows] = value
                out.water.add_band(rows, band.water_rows)
            if keep_stencil:
                for whole, value in zip(out.stencil, band.stencil):
                    whole[:, rows] = value
            out.lipschitz_bands.append(band.lipschitz)
            out.trajectory_bands.append(band.maxima)
            del band
        out.trajectory_maxima = _fold_across_cards(exchange, xp, [
            _fold_max(getattr(d, name) for d in out.trajectory_bands)
            for name in _TRAJECTORY_MAXIMA
        ])
        return out

    configured = int(options.trajectory_iterations)
    limit = float(options.trajectory_convergence_cells)
    with prof.section("bands"):
        done = run_bands(configured)
        iterations = configured
        # The resident step's retry (trajectory.converged_departure_points),
        # decided on the folded maximum over every band and card so every
        # band is searched at the same count: the bits are the whole-grid
        # search's whether or not the retry fires.
        if (done.trajectory_maxima[_TRAJECTORY_MAXIMA.index("move_max_cells")]
                > limit and configured < MAXIMUM_TRAJECTORY_ITERATIONS):
            del done
            iterations = MAXIMUM_TRAJECTORY_ITERATIONS
            done = run_bands(iterations)
    model.semilag_halo_rows = (halo, done.widest)
    del gradient_waists, lnps_gradient_waists, linear_waist, increment
    del sources, mass_grid, previous_wind
    uv_waist, stack_waist, lnps_waist = (
        done.uv_waist, done.stack_waist, done.lnps_waist)
    trajectory_rows = done.trajectory_rows
    advected, deficits, water = done.advected, done.deficits, done.water
    band_stencil = done.stencil

    # The gates read the globe: every card's band maxima and minima, folded.
    lipschitz_maxima = _fold_across_cards(exchange, xp, [
        _fold_max(getattr(d, name) for d in done.lipschitz_bands)
        for name in _LIPSCHITZ_MAXIMA
    ])
    # Minima fold as the maxima of their negatives (exact in any order).
    fold_minima = [-value for value in _fold_across_cards(exchange, xp, [
        _fold_max(-getattr(d, name) for d in done.lipschitz_bands)
        for name in _FOLD_MINIMA
    ])]
    deformation = _folded_lipschitz(
        done.lipschitz_bands, dt, done.lipschitz_bands[0].reference_length_m,
        lipschitz_maxima, fold_minima)
    refuse_trajectory_fold(deformation, model.minimum_fold_determinant)
    trajectory = TrajectoryDiagnostics(iterations, dt, *done.trajectory_maxima)
    convergence(trajectory, limit)
    del done

    with prof.section("assemble"):
        trial_zeta, trial_divergence = model.vector._vordiv_from_fourier(
            uv_waist.close())
        del uv_waist
        analysed = _contract_chunked(
            transform, stack_waist.close(), model.spectral_chunk)
        del stack_waist
        lnps_spectral = transform.project(
            transform.contract_waist(lnps_waist.close()))
        del lnps_waist
        if riding and exchange is not None:
            # The band's advected rows, onto every card: the grid tracers
            # are held whole on every card (the physics reads the globe),
            # and the mass fixer below reads the assembled globe.
            for name in GRID_TRACERS:
                exchange.fill_rows(xp, advected[name], 1,
                                   name=f"semilag_tracer_{name}")
                if want_deficit:
                    exchange.fill_rows(xp, deficits[name], 1,
                                       name=f"semilag_deficit_{name}")
        if band_stencil is not None and exchange is not None:
            # The departure coordinates too: the Bermejo-Conde fixer reads
            # the trilinear interpolant of the assembled globe at them.
            for axis_name, value in zip(("xi", "phi", "level"), band_stencil):
                exchange.fill_rows(xp, value, 1,
                                   name=f"semilag_stencil_{axis_name}")
        trial = MoistHybridState(
            vorticity=trial_zeta,
            divergence=trial_divergence,
            theta=analysed[:nlev],
            log_surface_pressure=lnps_spectral,
            qv=analysed[nlev:],
            **(advected if riding else atmosphere.grid_tracers()),
            time_s=atmosphere.time_s,
            step=atmosphere.step,
        )
        del analysed, lnps_spectral, trial_zeta
        if mark is not None:
            mark("semilag_transport", trial)

    with prof.section("solve"):
        advanced = model.semi_implicit.solve_shifted(
            trial, transform, model.vertical, alpha * dt
        )
        increment_s1 = float(
            xp.max(xp.abs(advanced.divergence - trial_divergence))
        )
        del trial_divergence

    with prof.section("tracers"):
        del trial
        fixer: dict[str, float] = {}
        water_relative = 0.0
        if riding:
            dp_before = _assembled_dp(
                model, model._pressure_sources(atmosphere))
            dp_after = _assembled_dp(
                model, model._pressure_sources(advanced))
            # The species as they stood, handed over as the run holds them:
            # the fixer stages a parked one for its own mass and drops it
            # (tracers.fix_mass), so the ten are never on the card at once
            # beside the advected ten (4.29 GiB at T799 L40 float32).  The
            # water reading that also read them was taken band by band.
            stencil = (None if band_stencil is None else Stencil(
                xi=band_stencil[0], phi=band_stencil[1],
                level=band_stencil[2], tables=tables))
            band_stencil = None
            fixed, fixer = fix_mass(
                advected, atmosphere.grid_tracers(), dp_before, dp_after,
                transform, scheme=options.tracer_fixer, deficits=deficits,
                stencil=stencil,
            )
            del stencil
            del dp_before, dp_after
            advanced = advanced.with_grid_tracers(fixed)
            column = _water_column_total(transform, water.plane)
            moved = sum(
                value for name, value in fixer.items()
                if name.startswith("semilag_tracer_mass_fixer_kg_m2__")
                and name.rsplit("__", 1)[1] in CONDENSATE_SPECIES
            ) / GRAVITY_M_S2
            water_relative = abs(moved) / max(column, 1.0e-30)
            del fixed
        del advected, deficits, water, band_stencil

    model.set_trajectory_state(
        RowTrajectory(first=first, last=last, nlat=nlat,
                      arrays=trajectory_rows)
        if split else TrajectoryState(**trajectory_rows))
    del trajectory_rows

    if mark is not None:
        mark("semilag_implicit", advanced)
    fixer_max = max(
        (value for name, value in fixer.items()
         if name.startswith("semilag_tracer_mass_fixer_relative__")),
        default=0.0,
    )
    transport = {
        "scheme": "semi_lagrangian",
        "max_courant_x": float(trajectory.displacement_max_cells),
        "max_courant_y": float(trajectory.displacement_max_cells),
        "max_courant_z": float(trajectory.displacement_max_levels),
        "substeps_x": 1, "substeps_y": 1, "substeps_z": 1,
        "floor_clip_kg_m2": 0.0,
        "pseudo_density_mismatch_relative": 0.0,
        "pseudo_density_mismatch_mean_relative": 0.0,
        "mass_fixer_max_relative": float(fixer_max),
        **fixer,
    }
    metrics = {
        **({"tracer_transport": transport} if riding else {}),
        "semi_implicit_max_divergence_increment_s1": increment_s1,
        "semilag_startup_step": bool(startup),
        "semilag_off_centring_weight": alpha,
        "semilag_tracer_mass_fixer_relative": float(fixer_max),
        "semilag_tracer_mass_fixer_water_relative": float(water_relative),
        "semilag_tracer_limiter": limiter,
        **fixer,
        **deformation.as_dict(),
        **trajectory.as_dict(),
        # One when the configured search missed the convergence test and
        # every band was searched again at the most iterations, zero
        # otherwise (the resident step's statement).
        "semilag_trajectory_retried": float(
            trajectory.iterations != int(options.trajectory_iterations)),
    }
    return advanced, metrics


__all__ = ["SEMILAG_INTEGRATORS", "grid_tables", "reference_profile",
           "semilag_step"]
