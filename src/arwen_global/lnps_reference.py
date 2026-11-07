"""ln ps about a fixed reference, for the float32 semi-Lagrangian step.

A float32 plane of ln ps sits near 11.5, where one unit in the last place is
9.5e-7, or 0.095 Pa of surface pressure.  The semi-Lagrangian gather's
float32 Lagrange weights sum to one only to about that unit, so a gather of
the whole plane writes nearly the same sub-ulp error at the same point every
step of a steady flow, and the error adds up: the nine-day dry
Jablonowski-Williamson steady arm drifted 1.51 hPa at T255 in float32 against
0.25 hPa in float64 (CHANGELOG 0.1.3, Open).  About ln(p_ref) the plane is
O(0.1) and the same weights move it a hundred times less, with no float64
table or transform on the card.
"""
from __future__ import annotations

import math

from .constants import REFERENCE_PRESSURE_PA

__all__ = [
    "LNPS_REFERENCE",
    "lnps_anomaly_coefficients",
    "lnps_from_anomaly_coefficients",
]

#: ln of the reference surface pressure the semi-Lagrangian step advects
#: ln ps about.  A FIXED constant, never a property of the run's own state,
#: so a restart takes off and puts back exactly the bytes a continuous run
#: does.
LNPS_REFERENCE = math.log(REFERENCE_PRESSURE_PA)


def lnps_anomaly_coefficients(transform, coefficients):
    """``coefficients`` (spectral ln ps) with ln(p_ref) taken off (0, 0).

    The subtraction is EXACT in any IEEE precision: the (0, 0) coefficient
    of a planet's ln ps is sqrt(4 pi) times its global-mean ln ps, 40.4 to
    41.0 for every global-mean surface pressure between 890 and 1050 hPa,
    and the constant is 40.81, so the two are within a factor of two of
    each other (true for any mean above 3.2 hPa) and Sterbenz's lemma makes
    their difference representable.
    :func:`lnps_from_anomaly_coefficients` puts the same rounded constant
    back, so a state the step leaves unchanged comes back bit for bit.
    """
    return transform.add_grid_constant(coefficients, -LNPS_REFERENCE)


def lnps_from_anomaly_coefficients(transform, coefficients):
    """The inverse of :func:`lnps_anomaly_coefficients`."""
    return transform.add_grid_constant(coefficients, LNPS_REFERENCE)
