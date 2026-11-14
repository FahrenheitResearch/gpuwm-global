"""Column solvers: the tridiagonal (Thomas) solve along the level axis, and
the implicit vertical diffusion every vertical process uses.

Arrays are (nz, ...) with the solve along axis 0; a row whose thickness is 0
(below the sea floor) is the identity row, so a column of any depth is
solved in the same pass."""
from __future__ import annotations


def thomas(a, b, c, d, xp):
    """Solve a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k (a_0 and c_{n-1}
    ignored)."""
    n = b.shape[0]
    cp = xp.empty_like(b)
    dp = xp.empty_like(d)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        m = b[k] - a[k] * cp[k - 1]
        cp[k] = c[k] / m
        dp[k] = (d[k] - a[k] * dp[k - 1]) / m
    x = xp.empty_like(d)
    x[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def implicit_diffusion(phi, h, kappa, dt, xp, *, top_flux=None, bottom_drag=None,
                       top_drag=None, w_adv=None, h_old=None, w_flux=None):
    """Backward-Euler vertical diffusion of ``phi`` (nz, ...) with cell
    thicknesses ``h`` (0 below the floor) and interface diffusivities
    ``kappa`` (nz, ...; entry k is the interface below level k, 0 at and
    below the floor).  ``top_flux`` (per unit area, units of phi m s-1, into
    the top cell) is added explicitly; ``bottom_drag``/``top_drag`` (s-1 m:
    a linear coefficient r with flux -r phi) act implicitly on the bottom
    wet cell / the top cell.  Conserves sum(h phi) up to the fluxes.

    Implicit upwind vertical advection, either form:

    * ``w_adv`` (nz, ...): the vertical velocity (m s-1, up) at the level
      centres, advective form (momentum);
    * ``w_flux`` (nz, ...): the volume flux per area (m s-1, up) through the
      interface below each level, flux form, with ``h_old`` the thickness
      before it acts (so ``h = h_old + dt (w_flux below - w_flux above)``):
      a field uniform in the vertical stays uniform and sum(h phi) is
      conserved (tracers)."""
    nz = phi.shape[0]
    wet = h > 0.0
    hh = xp.where(wet, h, 1.0)
    below = xp.zeros_like(h)
    below[:-1] = h[1:]
    delta = 0.5 * (h + below)
    has_below = wet & (below > 0.0)
    A = xp.where(has_below, dt * kappa / xp.where(has_below, delta, 1.0), 0.0)
    A_up = xp.zeros_like(A)
    A_up[1:] = A[:-1]
    a = -A_up
    c = -A
    b = hh + A_up + A
    if w_flux is not None:
        wb = xp.where(has_below, w_flux, 0.0)          # interface below k
        wt = xp.zeros_like(wb)
        wt[1:] = wb[:-1]                                 # interface above k
        b = b + dt * (xp.maximum(wt, 0.0) + xp.maximum(-wb, 0.0))
        a = a - dt * xp.maximum(-wt, 0.0)
        c = c - dt * xp.maximum(wb, 0.0)
        hh_rhs = xp.where(wet, h_old, 1.0)
        del wb, wt
    else:
        hh_rhs = hh
    if w_adv is not None:
        up = xp.where(wet, w_adv, 0.0)
        has_above = xp.zeros_like(wet)
        has_above[1:] = wet[:-1]
        d_dn = xp.where(has_below, delta, 1.0)
        d_up = xp.zeros_like(delta)
        d_up[1:] = delta[:-1]
        d_up = xp.where(has_above, d_up, 1.0)
        rise = xp.where(has_below, dt * hh * xp.maximum(up, 0.0) / d_dn, 0.0)
        sink = xp.where(has_above, dt * hh * xp.maximum(-up, 0.0) / d_up, 0.0)
        b = b + rise + sink
        c = c - rise
        a = a - sink
        del up, rise, sink, d_dn, d_up
    rhs = hh_rhs * phi
    if top_flux is not None:
        rhs[0] = rhs[0] + dt * top_flux
    if top_drag is not None:
        b[0] = b[0] + dt * top_drag
    if bottom_drag is not None:
        bottom = wet & ~has_below
        b = b + xp.where(bottom, dt * bottom_drag, 0.0)
    out = thomas(a, b, c, rhs, xp)
    return xp.where(wet, out, 0.0)


def by_rows(fn, nchunks: int, ny: int, _xp, *args, **kwargs):
    """Call the column function ``fn`` on blocks of rows and join the
    results, so its temporaries live one block at a time (the card's
    transient memory falls by the block count; the arithmetic of every
    column is unchanged).  An array argument whose axis -2 has ``ny``
    entries is sliced there; everything else is passed whole.  ``fn``
    returns an array, a tuple of arrays or a dict of arrays."""
    if nchunks <= 1:
        return fn(*args, **kwargs)
    edges = [round(ny * i / nchunks) for i in range(nchunks + 1)]

    def cut(a, j0, j1):
        nd = getattr(a, "ndim", None)
        if isinstance(nd, int) and nd >= 2 and a.shape[-2] == ny:
            return a[..., j0:j1, :]
        return a

    parts = []
    for j0, j1 in zip(edges[:-1], edges[1:]):
        if j1 <= j0:
            continue
        parts.append(fn(*[cut(a, j0, j1) for a in args],
                        **{k: cut(v, j0, j1) for k, v in kwargs.items()}))
    first = parts[0]
    if isinstance(first, dict):
        return {k: _xp.concatenate([p[k] for p in parts], axis=-2) for k in first}
    if isinstance(first, tuple):
        return tuple(_xp.concatenate([p[i] for p in parts], axis=-2) for i in range(len(first)))
    return _xp.concatenate(parts, axis=-2)


__all__ = ["by_rows", "implicit_diffusion", "thomas"]
