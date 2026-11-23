"""Exact exponential hyperdiffusion in total spherical wavenumber."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transform import SphericalHarmonicTransform

#: The spelling of [diffusion] background_fields.
BACKGROUND_FIELDS = ("all", "momentum")
#: The spectral fields "momentum" names.
MOMENTUM_FIELDS = ("vorticity", "divergence")


@dataclass(frozen=True)
class ExponentialHyperdiffusion:
    """exp(-(dt/tau) (n(n+1)/N(N+1))^order) per total degree n.

    ``background_order`` > 0 adds a second, lower-order term of the same
    form with its own e-folding time at the truncation, the way the
    operational spectral models drain the mesoscale band: IFS on its
    linear grid applies del^4 (order 2) at tau = 6 dt at the truncation
    (IFS Cy49r1 Part III, eq. 2.60), the spectral GFS an implicit del^8
    (order 4), FV3 GFS a sixth-order (order 3) divergence damping.  The
    order-16 drain alone bites only above about 0.85 N, so the band
    between it and the resolved cascade has no sink.  ``background_order``
    0 (the default) is exactly the single-term operator.
    """

    order: int = 4
    e_folding_time_s_at_truncation: float = 21_600.0
    preserve_degree: int = 1
    background_order: int = 0
    background_e_folding_time_s_at_truncation: float = 0.0
    #: "all": the background term acts on every diffused field, as the drain
    #: does.  "momentum": on vorticity and divergence only, the way FV3
    #: damps divergence and vorticity; temperature and moisture keep the drain
    #: alone, so the term does not diffuse them along the terrain-following
    #: model surfaces (IFS corrects its temperature diffusion to pressure
    #: surfaces for that reason, Part III below eq. 2.59).
    background_fields: str = "all"

    def __post_init__(self) -> None:
        if self.background_fields not in BACKGROUND_FIELDS:
            raise ValueError(f"background_fields must be one of {BACKGROUND_FIELDS}")
        if self.order < 1:
            raise ValueError("hyperdiffusion order must be >= 1")
        if not math.isfinite(self.e_folding_time_s_at_truncation) or self.e_folding_time_s_at_truncation <= 0:
            raise ValueError("e-folding time must be finite and positive")
        if self.preserve_degree < 0:
            raise ValueError("preserve_degree must be >= 0")
        if self.background_order < 0:
            raise ValueError("background_order must be >= 0 (0 is off)")
        if self.background_order > 0:
            tau = self.background_e_folding_time_s_at_truncation
            if not math.isfinite(tau) or tau <= 0:
                raise ValueError("background e-folding time must be finite and positive when the term is on")

    def background_acts_on(self, field_name: str) -> bool:
        """Whether the background term acts on the named spectral field."""
        if self.background_order == 0:
            return False
        return self.background_fields == "all" or field_name in MOMENTUM_FIELDS

    def factors(self, transform: SphericalHarmonicTransform, dt_s: float, *, background: bool = True):
        xp = transform.backend.xp
        n = np.arange(transform.truncation + 1, dtype=np.float64)
        lam = n * (n + 1.0)
        top = max(1.0, lam[-1])
        exponent = -(float(dt_s) / self.e_folding_time_s_at_truncation) * (lam / top) ** self.order
        if self.background_order > 0 and background:
            exponent = exponent - (
                float(dt_s) / self.background_e_folding_time_s_at_truncation
            ) * (lam / top) ** self.background_order
        factor = np.exp(exponent)
        factor[: self.preserve_degree + 1] = 1.0
        return xp.asarray(factor, dtype=transform.backend.float_dtype)

    def apply(self, coeff, transform: SphericalHarmonicTransform, dt_s: float, *, strength: float = 1.0,
              background: bool = True):
        if strength < 0 or not math.isfinite(strength):
            raise ValueError("diffusion strength must be finite and nonnegative")
        factor = self.factors(transform, dt_s, background=background) ** float(strength)
        return transform.project(coeff * factor[:, None])
