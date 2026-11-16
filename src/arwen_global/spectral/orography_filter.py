"""Isotropic spectral filter for the model orography (experimental).

``[initial] orography_filter_half_degree = n0`` multiplies the spectral
coefficients of the analysis surface geopotential by

    factor(n) = exp(-ln 2 * (n (n + 1) / (n0 (n0 + 1)))**2),   factor(0) = 1

before the model terrain is synthesized: half amplitude at total degree
n0, close to one well below it, falling fast above it, and the global mean
(degree 0) kept.  The coefficients are laid out [n, m] and the filtered
field is projected back onto the triangular truncation, as
``ExponentialHyperdiffusion.apply`` does.  ``n0 = 0`` is off and returns
the coefficients untouched, so the shipped terrain path is unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from .transform import SphericalHarmonicTransform


def orography_filter_factor(truncation: int, half_degree: float) -> np.ndarray:
    """The float64 factor for degrees ``0..truncation`` at half degree ``n0``.

    Pure NumPy, so it is testable without a device.  ``half_degree`` must
    be finite and positive; the config layer bounds it by the truncation.
    """
    n0 = float(half_degree)
    if not math.isfinite(n0) or n0 <= 0.0:
        raise ValueError(
            "orography filter half degree must be finite and positive "
            f"(got {half_degree!r}); 0 means off and builds no factor"
        )
    truncation = int(truncation)
    if truncation < 0:
        raise ValueError("truncation must be >= 0")
    degree = np.arange(truncation + 1, dtype=np.float64)
    ratio = degree * (degree + 1.0) / (n0 * (n0 + 1.0))
    factor = np.exp(-math.log(2.0) * ratio ** 2)
    factor[0] = 1.0
    return factor


def filter_orography_coefficients(coeff, transform: SphericalHarmonicTransform,
                                  half_degree: float):
    """``coeff`` ([n, m] on ``transform``) with the orography filter applied.

    ``half_degree == 0`` returns ``coeff`` itself, unchanged, so the off
    path is the shipped arithmetic bit for bit.
    """
    if float(half_degree) == 0.0:
        return coeff
    xp = transform.backend.xp
    factor = orography_filter_factor(transform.truncation, half_degree)
    return transform.project(
        coeff * xp.asarray(factor, dtype=transform.backend.float_dtype)[:, None])
