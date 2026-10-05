"""How many latitude rows a semi-Lagrangian band holds beyond its own.

A band of the semi-Lagrangian step computes the departure points of its
own rows and reads the advected fields there.  Those points lie up to one
step's travel away, so the band has to hold that many rows on either side
of its own, plus the rows its widest stencil reads around a departure
point.  The width is pure arithmetic on the grid and the step and never
enters a value the band computes: a band behind any halo wide enough
returns the same bits (semilag.tables.BandWindow), so the halo is a memory
and work figure, not an identity.

The figure:

    rows = ceil(dt * HALO_WIND_M_S / (pi a / nlat)) + STENCIL_ROWS + 1

``pi a / nlat`` is the mean meridional spacing of the Gaussian rows, and a
departure point's latitude moves by no more than its great-circle
distance, so the first term bounds the rows a parcel crosses for any wind
up to :data:`HALO_WIND_M_S`.  ``STENCIL_ROWS`` is the six-point quintic's
reach of three rows beyond its bracket (the widest stencil the step reads),
and the last row is the Lipschitz diagnostic's centred difference.  At
T533 (801 rows, 25.0 km) and 300 s that is 4 + 3 + 1 = 8 rows, the top of
the six to eight the audit of 2026-10-05 sized by hand; at T799 (1200
rows, 16.7 km) it is 9 at 300 s and 7 at 150 s.

A flow faster than :data:`HALO_WIND_M_S`, or a trajectory iterate that
overshoots, is not trusted to this figure: the band kernels flag any read
of a row the band does not hold (semilag.tables.HaloEscape) and the step
recomputes that band behind a halo twice as wide, which is exact for the
reason above.
"""
from __future__ import annotations

import math

#: The fastest wind the default halo is sized for, m/s.  The stratospheric
#: polar-night jet at the 1 hPa lid reaches about 200 m/s; 250 leaves a
#: quarter of headroom before the escape check widens a band.
HALO_WIND_M_S = 250.0

#: Rows beyond the bracket the widest gather reads: the six-point stencil
#: reads three rows on its far side (semilag.tables, the quintic tables).
STENCIL_ROWS = 3

#: The Lipschitz diagnostic's centred meridional difference.
DIFFERENCE_ROWS = 1


def default_halo_rows(nlat: int, dt_s: float, radius_m: float) -> int:
    """The halo a band of an ``nlat``-row grid holds at step ``dt_s``."""
    n = int(nlat)
    if n < 1:
        raise ValueError(f"nlat must be >= 1, got {nlat!r}")
    spacing = math.pi * float(radius_m) / n
    travel = math.ceil(abs(float(dt_s)) * HALO_WIND_M_S / spacing)
    return int(min(n, travel + STENCIL_ROWS + DIFFERENCE_ROWS))


__all__ = ["DIFFERENCE_ROWS", "HALO_WIND_M_S", "STENCIL_ROWS",
           "default_halo_rows"]
