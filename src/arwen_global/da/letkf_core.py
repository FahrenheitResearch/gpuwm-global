"""The LETKF arithmetic this package's filter used to borrow from the engine.

Until 0.2.5 the point LETKF (:mod:`arwen_global.da.letkf_point`), the
localisation builder and the twin experiment imported eight names from
``gpuwm.da.letkf``: the refusal class, the Gaspari-Cohn function, the array
namespace pick, the eigensolver choice and factorisation, and three device
housekeeping helpers.  gpuwm 3.0.3 removed that module with the engine's own
local data assimilation, so every command of this package failed at import,
``doctor`` and ``--help`` included, the moment the engine pin moved to 3.0.

The eight names are carried here, verbatim from gpuwm 2.9.13
(``gpuwm/da/letkf.py``, Apache-2.0, the same owner), so the filter computes
exactly what it computed on 2.9.13.  The one engine module they still reach
is ``gpuwm.core.jacobi_eigh``, the batched Jacobi eigensolver, which 3.0.3
keeps; ``arwen_global/data/engine-seam.json`` pins it.

Algorithm authority: Hunt, Kostelich and Szunyogh (2007), Physica D 230,
112-126, section 2.3; Gaspari and Cohn (1999), QJRMS 125, 723-757,
equation (4.10).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "LetkfError",
    "gaspari_cohn",
]


class LetkfError(ValueError):
    """Any refusal by this module.  Never raised for a merely hard problem."""


def _get_xp(*arrays):
    """numpy, or cupy when any input is a cupy array.

    No cupy import happens unless a cupy array actually shows up, which is
    what keeps ``-m "not gpu"`` accurate for callers of this module.
    """
    for a in arrays:
        mod = type(a).__module__
        if mod is not None and mod.split(".")[0] == "cupy":
            import cupy
            return cupy
    return np


# ---------------------------------------------------------------------------
# Gaspari-Cohn
# ---------------------------------------------------------------------------

def gaspari_cohn(distance, cutoff):
    """The Gaspari-Cohn compactly supported correlation function.

    Gaspari and Cohn (1999), *Construction of correlation functions in two
    and three dimensions*, QJRMS 125, 723-757, equation (4.10).  This is the
    fifth-order piecewise rational function that approximates a Gaussian
    while vanishing identically beyond a finite radius -- the property that
    makes it, rather than the Gaussian, the standard localisation kernel.

    Parameters
    ----------
    distance
        Separation, same units as ``cutoff``.  Scalar or array; negative
        values are treated by magnitude.
    cutoff
        The radius at which the function reaches zero AND STAYS THERE.  This
        is the ``2c`` of Gaspari and Cohn's own parameterisation, not their
        ``c``.  The two conventions differ by a factor of two and confusing
        them silently halves or doubles every localisation radius in a
        system, so this module names the unambiguous one: past ``cutoff``
        the weight is exactly ``0.0``, so ``cutoff`` is the distance beyond
        which an observation cannot influence an analysis point.  Gaspari
        and Cohn's ``c`` -- their e-folding-ish length scale, where the
        function takes the value 5/24 -- is ``cutoff / 2``.

    Returns
    -------
    Weights in [0, 1], with ``gaspari_cohn(0, c) == 1.0`` exactly and
    ``gaspari_cohn(d, c) == 0.0`` exactly for ``d >= c``.

    Notes
    -----
    Evaluated in the namespace of ``distance`` (numpy or cupy) and in that
    array's dtype, except that an integer or python-scalar input is promoted
    to float64.  The ``2/(3r)`` term of the outer branch is evaluated with a
    guarded denominator so that the vectorised form does not raise or warn
    on the ``r = 0`` elements it computes and then discards.
    """
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise LetkfError(
            f"Gaspari-Cohn cutoff must be finite and positive, got {cutoff!r}."
            " Use a radius larger than the domain diagonal to mean 'no"
            " localisation'; infinity is refused because it implies an"
            " infinite stencil."
        )
    xp = _get_xp(distance)
    d = xp.abs(xp.asarray(distance))
    if d.dtype.kind != "f":
        d = d.astype(xp.float64)
    one = d.dtype.type(1)

    # r is distance in units of Gaspari-Cohn's c = cutoff/2, so the two
    # polynomial branches are r <= 1 and 1 < r < 2, exactly as published.
    r = d * d.dtype.type(2.0 / cutoff)
    # Both polynomials are evaluated everywhere and selected between
    # afterwards, so an out-of-support distance still flows through r**5.
    # Clamping first keeps an infinite separation -- which is exactly what a
    # masked or out-of-domain slot looks like to a caller -- from producing
    # inf - inf and a RuntimeWarning on a value that is about to be
    # discarded.
    r = xp.minimum(r, d.dtype.type(2))
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r

    inner = (
        -r5 / 4 + r4 / 2 + r3 * (5.0 / 8.0) - r2 * (5.0 / 3.0) + one
    )
    # Guard only the reciprocal: r == 0 elements take the inner branch and
    # this value is discarded, but an unguarded 1/0 still warns and pollutes.
    r_safe = xp.where(r > 0, r, one)
    outer = (
        r5 / 12 - r4 / 2 + r3 * (5.0 / 8.0) + r2 * (5.0 / 3.0)
        - r * 5 + one * 4 - (2.0 / 3.0) / r_safe
    )

    # The outer branch is selected on r < 2, STRICTLY.  Analytically the
    # polynomial is zero at r = 2, but in floating point it evaluates to a
    # few ulp either side of zero, and the whole localisation guarantee --
    # "an increment beyond the cutoff is bitwise zero, not merely small" --
    # rests on the weight at and past the cutoff being the literal 0.0 that
    # drops the observation from the gridpoint's batch.  Selecting the
    # constant instead of the polynomial at r >= 2 is what makes that true
    # rather than nearly true.
    out = xp.where(r <= 1, inner, xp.where(r < 2, outer, xp.zeros_like(d)))
    # The published polynomial is nonnegative on [0, 2] but rounding can
    # produce -1e-17 within an ulp of r == 2; a negative localisation weight
    # would flip the sign of an observation's influence.
    return xp.maximum(out, xp.zeros_like(out))


# ---------------------------------------------------------------------------
# Device housekeeping
# ---------------------------------------------------------------------------

#: Substrings that identify an allocation failure in a device library's
#: message, lowercased.  Matched only on exceptions raised from the cupy
#: stack, so a LetkfError that merely *mentions* memory cannot trip this.
_MEMORY_ERROR_MARKERS = (
    "out of memory",
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
    "alloc_failed",
    "allocation failure",
)


def _is_device_memory_error(exc) -> bool:
    """Is this an allocation failure rather than a wrong answer?

    Allocation failures are the one exception class the chunk loop may
    retry at a smaller chunk: the analysis they interrupted was never
    produced, and a smaller chunk asks the same mathematical question with
    a smaller footprint.  Everything else -- a non-positive eigenvalue, a
    non-finite weight, a LetkfError -- would only be asked again at a
    different batch shape, and a wrong answer does not become right by
    being recomputed in smaller pieces.

    Recognition is by name and module along the cause chain, so this works
    with numpy alone on the path where cupy was never imported: a host
    ``MemoryError`` (numpy's ``_ArrayMemoryError`` is a subclass) is an
    allocation failure too, and the same degradation is the right answer
    for it.
    """

    seen = set()
    node = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, MemoryError):
            return True
        cls = type(node)
        root = (cls.__module__ or "").split(".")[0]
        if root in ("cupy", "cupy_backends"):
            if cls.__name__ == "OutOfMemoryError":
                return True
            text = str(node).lower()
            if any(marker in text for marker in _MEMORY_ERROR_MARKERS):
                return True
        node = node.__cause__ if node.__cause__ is not None \
            else node.__context__
    return False


def _sync_namespace(xp) -> None:
    """Wait for the namespace's queued work, so a timer means something.

    A no-op on numpy, which is synchronous.  On a device namespace every
    call in the analysis only ENQUEUES work, so a stopwatch read without
    this bills whatever it likes to whichever phase happens to block
    first -- typically the one that copies a result back, which is the
    phase after the expensive one.  Swallowed on failure for the same
    reason as :func:`_release_device_scratch`: this measures the analysis,
    it must never be able to fail one.
    """

    if xp is np:
        return
    try:
        xp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _release_device_scratch(xp) -> None:
    """Hand every idle pool block back to the driver, quietly.

    Called between a failed chunk attempt and its smaller retry.  Failure
    here is deliberately swallowed: if the context is too broken to
    synchronise, the retry itself will surface that as its own error,
    which is a better message than one from a cleanup helper.
    """

    if xp is np:
        return
    try:
        xp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    try:
        xp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The eigensolver
# ---------------------------------------------------------------------------

def _resolve_eigensolver(xp, members, solve_dtype, config):
    """Pick the eigensolver ONCE, before any chunk runs.

    Deciding per chunk would let a single analysis silently mix two solvers
    -- they agree to rounding, so the seam would be invisible in the
    increments and visible only as an irreproducible receipt.  Deciding here
    also means an unsatisfiable ``eigensolver="jacobi"`` refuses before the
    first gather rather than after the last one.
    """
    mode = str(config.eigensolver)
    if mode == "library":
        return "library"
    if xp is np:
        if mode == "jacobi":
            raise LetkfError(
                "eigensolver='jacobi' needs the analysis on the device: the"
                " kernel is CUDA and this analysis is running in numpy."
                "  Pass cupy arrays, or use eigensolver='auto', which means"
                " numpy's own LAPACK here and needs no CUDA library at all."
            )
        return "library"
    from gpuwm.core.jacobi_eigh import supported
    if supported(members, solve_dtype):
        return "jacobi"
    if mode == "jacobi":
        from gpuwm.core.jacobi_eigh import MAX_K, MIN_K
        raise LetkfError(
            f"eigensolver='jacobi' was required but this project's kernel"
            f" does not take R={members} in {np.dtype(solve_dtype).name}: it"
            f" supports {MIN_K} <= R <= {MAX_K} in float32 or float64."
            "  Use eigensolver='auto' to fall back to the array namespace's"
            " own solver, or reduce the ensemble size."
        )
    return "library"


def _eigendecompose(xp, amat, which):
    """``(evals, evecs, sweeps)`` for a batch of symmetric matrices, ascending.

    Both branches promise the same contract -- eigenvalues ascending,
    eigenvectors in the columns -- so everything downstream is written once.
    ``sweeps`` is 0 for the library branch, which does not report one.
    """
    if which == "jacobi":
        from gpuwm.core.jacobi_eigh import JacobiEighError, batched_eigh
        try:
            return batched_eigh(amat, return_sweeps=True)
        except JacobiEighError as exc:
            raise LetkfError(
                "this project's batched Jacobi eigensolver refused the"
                f" localised LETKF matrix (R-1)I/rho + C Yb: {exc}"
                "  No analysis was produced; do not treat the prior as one."
            ) from exc
    try:
        evals, evecs = xp.linalg.eigh(amat)
        return evals, evecs, 0
    except Exception as exc:                     # LinAlgError and backend kin
        if _is_device_memory_error(exc):
            # NOT wrapped: an allocation failure is the chunk loop's to
            # handle -- it retries at a smaller chunk -- and wrapping it in
            # LetkfError here would turn a recoverable footprint problem
            # into a terminal verdict about the mathematics.
            raise
        # The one failure mode no input check can pre-empt.  A raw
        # convergence message from LAPACK or cuSOLVER does not name the
        # analysis that did not happen, and this module promises that every
        # failure arrives as LetkfError.
        #
        # On the device this is also where a MISSING cuSOLVER lands, which is
        # a different problem wearing the same exception: elementwise CuPy
        # works, the gather worked, and only the factorisation failed.  Say
        # so, and name the setting that does not need it.
        raise LetkfError(
            "the batched eigensolver failed on the localised LETKF"
            f" matrix (R-1)I/rho + C Yb: {type(exc).__name__}: {exc}."
            "  This is eigensolver='library', which on the device is"
            " cuSOLVER -- a separate CUDA library from the one elementwise"
            " CuPy uses, so the rest of the analysis working says nothing"
            " about whether it is installed.  Run `gpuwm doctor` for a"
            " verdict on it, or use eigensolver='auto' (the default), which"
            " prefers this project's own kernel and needs no CUDA library."
            "  No analysis was produced; do not treat the prior as one."
        ) from exc
