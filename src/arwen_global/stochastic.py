"""Stochastic physics for ensemble members: SPPT and SKEB on the spectral state.

Ported from this project's own stochastic physics for the gpuwm-hex model
line (project code, Apache-2.0, itself written from the published methods),
and re-expressed for a spectral model, where the patterns are spectral fields
and need no on-the-fly harmonic synthesis:

* SPPT: Buizza, Miller & Palmer (1999, QJRMS 125, 2887-2908) and Palmer et
  al. (2009, ECMWF Tech. Memo. 598).  The net physics change of u, v, T and
  qv over a physics call is multiplied by ``1 + mu*r``: ``r`` the sum of
  independent AR(1) spectral patterns (multi-scale SPPT), bounded so
  ``1 + r`` stays in ``[0, 2]`` (hard clip, or ``clip*tanh(r/clip)``), and
  ``mu`` a taper that is zero at the lowest level, ramps in over the lowest
  ``surface_m`` of height above ground and ramps out between
  ``top_start_m`` and ``top_end_m``.  Wherever the perturbed humidity would
  be negative or supersaturated (Bolton 1980 over water) the point keeps
  its unperturbed temperature and humidity change (Palmer et al. 2009).
* SKEB: Shutts (2005, QJRMS 131, 3079-3102) and Berner et al. (2009, JAS
  66, 603-626).  A streamfunction pattern whose rotational wind injects a
  fraction of the dissipated kinetic energy back into the resolved flow.
  The dissipation is the positive kinetic-energy sink of the UNPERTURBED
  physics wind change, area-mean per level; the vorticity forcing is the
  pattern's spectral Laplacian normalised by the pattern's expected
  rotational-wind RMS; a small thermal pattern (``thermal_fraction``) does
  the same for temperature.  Both are applied in spectral space.

Pattern law (Weaver & Courtier 2001, QJRMS 127, 1815-1846): a field of
spherical-harmonic coefficients with variance spectrum ``exp(-l(l+1)a)``,
``a = L**2/(2 R**2)``, normalised to unit point variance, so the spatial
correlation is ``exp(-r**2/(2 L**2))``; each coefficient is AR(1) in time
with lag correlation ``exp(-dt/tau)``.  Degree 0 is excluded (zero global
mean).  The noise is Philox4x32-10 (Random123, BSD-3-Clause, see
licenses/LICENSE-Random123.txt) keyed by (member, seed, process) and
counted by (mode, draw), so a member's patterns are a pure function of its
seed and of the clock, independent of card, band count and restart.

What hex had that is not here: the mesh synthesis and edge/vertex
geometry (a spectral model has neither), the limited-area masks, and the
LTS macro clock.  SHUM is not implemented.

The pattern state (coefficients, clock, draw counter) is carried in the
physics namespace under ``stochastic_*`` names, so checkpoints carry it and
a restart continues the same sequence bit for bit.  Off (the default) the
module is never imported by the step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np

SCHEMA = "arwen-global.stochastic/v1"
#: The physics-namespace prefix of the carried pattern state.
STATE_PREFIX = "stochastic_"
COEFFICIENTS = STATE_PREFIX + "coefficients"
CLOCK = STATE_PREFIX + "clock"
EARTH_RADIUS_M = 6371220.0
#: Omitted variance fraction of the automatic pattern truncation (hex).
TRUNCATION_TAIL = 1.0e-4
GRAVITY = 9.80665

#: ECMWF's operational three-scale SPPT (Palmer et al. 2009; Leutbecher et
#: al. 2017): amplitude, correlation length (m), decorrelation time (s).
ECMWF_SPPT_SCALES = (
    (0.52, 500.0e3, 6 * 3600.0),
    (0.18, 1000.0e3, 3 * 86400.0),
    (0.06, 2000.0e3, 30 * 86400.0),
)

_MASK32 = np.uint64(0xFFFFFFFF)


# ---------------------------------------------------------------- noise


def philox4x32(c0, c1, c2, c3, key: int):
    """Philox4x32-10 (Random123) on uint32 words held in uint64 arrays.

    The round function of hex ``stoch_philox`` (stochastic.cu), word for
    word; numpy's uint64 product of two 32-bit words is exact.
    """
    k0 = np.uint64(int(key) & 0xFFFFFFFF)
    k1 = np.uint64((int(key) >> 32) & 0xFFFFFFFF)
    m0 = np.uint64(0xD2511F53)
    m1 = np.uint64(0xCD9E8D57)
    w0 = np.uint64(0x9E3779B9)
    w1 = np.uint64(0xBB67AE85)
    s32 = np.uint64(32)
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) & _MASK32 for c in (c0, c1, c2, c3))
    for _ in range(10):
        p0 = m0 * c0
        p1 = m1 * c2
        c0, c1, c2, c3 = (
            ((p1 >> s32) ^ c1 ^ k0) & _MASK32,
            p1 & _MASK32,
            ((p0 >> s32) ^ c3 ^ k1) & _MASK32,
            p0 & _MASK32,
        )
        k0 = (k0 + w0) & _MASK32
        k1 = (k1 + w1) & _MASK32
    return c0, c1, c2, c3


def _box_muller(a, b):
    u = (a.astype(np.float64) + 0.5) * 2.0 ** -32
    v = (b.astype(np.float64) + 0.5) * 2.0 ** -32
    return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)


def complex_noise(degree: int, key: int, draw: int):
    """Unit complex Gaussian noise on the triangle ``(l, m<=l)``.

    m = 0: real N(0, 1).  m > 0: real and imaginary parts each N(0, 1/2),
    so the real field sum_l c_l0 Y_l0 + 2 Re sum_m c_lm Y_lm has the
    variance of a real-orthonormal expansion with unit-variance coefficients.
    """
    n = degree + 1
    index = np.arange(n * n, dtype=np.uint64)
    d = int(draw)
    x, y, z, w = philox4x32(index, np.uint64(d & 0xFFFFFFFF), np.uint64(0),
                            np.uint64((d >> 32) & 0xFFFFFFFF), key)
    re = _box_muller(x, y).reshape(n, n)
    im = _box_muller(z, w).reshape(n, n)
    out = (re + 1j * im) / math.sqrt(2.0)
    out[:, 0] = re[:, 0]
    out *= np.tri(n, dtype=bool)
    return out


# ---------------------------------------------------------------- law


def spectral_exponent(length_m: float, radius: float = EARTH_RADIUS_M) -> float:
    """``a`` of the variance spectrum ``exp(-l(l+1)a)`` whose correlation is
    ``exp(-r**2/(2 L**2))``: ``a = L**2/(2 R**2)`` (hex, Weaver & Courtier)."""
    ratio = float(length_m) / float(radius)
    return ratio * ratio / 2.0


def auto_degree(length_m: float, radius: float = EARTH_RADIUS_M,
                tail: float = TRUNCATION_TAIL) -> int:
    """Smallest degree whose omitted variance is below ``tail`` (8..512)."""
    a = spectral_exponent(length_m, radius)
    n = math.ceil((-1 + math.sqrt(1 + 4 * math.log(1 / tail) / a)) / 2)
    return int(min(512, max(8, n)))


def degree_sigma(length_m: float, degree: int, radius: float = EARTH_RADIUS_M):
    """Per-degree coefficient standard deviation for unit point variance.

    sum_l (2l+1) sigma_l**2 / (4 pi) = 1 (addition theorem), l = 1..degree.
    The l = 1 exponent is cancelled before normalising (hex), so a very long
    length keeps a finite dipole limit.
    """
    ell = np.arange(degree + 1, dtype=np.float64)
    a = spectral_exponent(length_m, radius)
    variance = np.exp(-(ell * (ell + 1) - 2.0) * a)
    variance[0] = 0.0
    total = np.sum((2 * ell + 1) * variance)
    return np.sqrt(variance * 4.0 * math.pi / total)


def backscatter_rate_scale(period_s: float, time_s: float) -> float:
    """Rate variance per unit energy of a held AR(1) pattern (hex).

    With update period h and phi = exp(-h/tau) the integrated
    autocovariance is h/2 (1+phi)/(1-phi), so sigma**2 = 2 D/h tanh(h/(2 tau)).
    """
    if not (math.isfinite(period_s) and period_s > 0 and math.isfinite(time_s) and time_s > 0):
        raise ValueError("backscatter period and time must be positive and finite")
    return 2.0 * math.tanh(period_s / (2.0 * time_s)) / period_s


# ---------------------------------------------------------------- config


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"stochastic.{name} must be positive and finite, got {value!r}")
    return value


@dataclass(frozen=True)
class StochasticConfig:
    """``[stochastic]``.  Everything off is the default and leaves the run's
    identity, arithmetic and checkpoints exactly as they were."""

    sppt: bool = False
    skeb: bool = False
    #: The member number and the ensemble seed: together they key every
    #: pattern this member draws.
    member: int = 0
    seed: int = 53
    #: SPPT scales: (amplitude, correlation length m, decorrelation time s).
    sppt_scales: tuple = ECMWF_SPPT_SCALES
    #: |r| bound; 1 keeps 1 + r in [0, 2] so a tendency never reverses.
    clip: float = 1.0
    #: False: hard clip (ECMWF); True: clip*tanh(r/clip) (hex's smooth bound).
    logit: bool = False
    surface_m: float = 500.0
    top_start_m: float = 18000.0
    top_end_m: float = 25000.0
    moist_guard: bool = True
    skeb_length_m: float = 500.0e3
    skeb_time_s: float = 6 * 3600.0
    #: Fraction of the dissipated kinetic energy injected back (hex 0.1).
    skeb_fraction: float = 0.1
    #: Fraction injected as the thermal pattern (hex 0.01).
    thermal_fraction: float = 0.01

    def __post_init__(self):
        for name in ("sppt", "skeb", "logit", "moist_guard"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"stochastic.{name} must be true or false")
        for name in ("member", "seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2 ** 63:
                raise ValueError(f"stochastic.{name} must be a nonnegative integer")
        scales = tuple(tuple(float(v) for v in scale) for scale in self.sppt_scales)
        if not 1 <= len(scales) <= 5:
            raise ValueError("stochastic SPPT takes one to five scales")
        for amp, length, tau in scales:
            _positive(amp, "sppt amplitude")
            _positive(length, "sppt length")
            _positive(tau, "sppt time")
        object.__setattr__(self, "sppt_scales", scales)
        for name in ("surface_m", "top_end_m", "skeb_length_m", "skeb_time_s"):
            _positive(getattr(self, name), name)
        if not (0 < float(self.clip) <= 1):
            raise ValueError("stochastic.clip must be in (0, 1] so SPPT cannot reverse a tendency")
        if not (0 <= float(self.top_start_m) < float(self.top_end_m)):
            raise ValueError("stochastic.top_start_m must be nonnegative and below top_end_m")
        for name in ("skeb_fraction", "thermal_fraction"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"stochastic.{name} must be in [0, 1]")

    @property
    def enabled(self) -> bool:
        return self.sppt or self.skeb

    def identity(self) -> dict | None:
        """The canonical settings for the config hash, None when off."""
        if not self.enabled:
            return None
        payload = asdict(self)
        payload["sppt_scales"] = [list(s) for s in self.sppt_scales]
        payload["schema"] = SCHEMA
        return payload

    def process_key(self, name: str) -> int:
        payload = f"{SCHEMA}:{self.member}:{self.seed}:{name}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def processes(self):
        """(name, length_m, time_s) of every pattern this config draws."""
        out = []
        if self.sppt:
            out += [(f"sppt{i}", length, tau)
                    for i, (_amp, length, tau) in enumerate(self.sppt_scales)]
        if self.skeb:
            out += [("skeb_psi", self.skeb_length_m, self.skeb_time_s),
                    ("skeb_thermal", self.skeb_length_m, self.skeb_time_s)]
        return out


def config_from_table(table) -> StochasticConfig:
    """``[stochastic]`` -> :class:`StochasticConfig` (unknown keys refused)."""
    table = dict(table or {})
    known = {
        "sppt", "skeb", "member", "seed", "sppt_amplitudes", "sppt_lengths_m",
        "sppt_times_s", "clip", "logit", "surface_m", "top_start_m", "top_end_m",
        "moist_guard", "skeb_length_m", "skeb_time_s", "skeb_fraction",
        "thermal_fraction",
    }
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"unknown [stochastic] keys: {', '.join(unknown)}")
    options = {k: table[k] for k in table if not k.startswith("sppt_")}
    lists = [table.get(k) for k in ("sppt_amplitudes", "sppt_lengths_m", "sppt_times_s")]
    if any(v is not None for v in lists):
        if any(v is None for v in lists) or len({len(v) for v in lists}) != 1:
            raise ValueError("stochastic.sppt_amplitudes, sppt_lengths_m and sppt_times_s "
                             "must be given together, of equal length")
        options["sppt_scales"] = tuple(zip(*lists))
    for name in ("clip", "surface_m", "top_start_m", "top_end_m", "skeb_length_m",
                 "skeb_time_s", "skeb_fraction", "thermal_fraction"):
        if name in options:
            options[name] = float(options[name])
    return StochasticConfig(**options)


# ---------------------------------------------------------------- patterns


@dataclass
class _Process:
    name: str
    length_m: float
    time_s: float
    degree: int
    key: int
    sigma: np.ndarray  # (degree+1,)


@dataclass
class CallContext:
    """What one physics call needs of the stochastic state."""
    sppt_grid: object = None        # (nlat, nlon) bounded-before pattern r
    psi: object = None              # unit SKEB streamfunction, complex (T+1, T+1)
    thermal: object = None          # unit SKEB thermal pattern, complex (T+1, T+1)
    curl_rms: float = 0.0
    period_s: float = 0.0
    state: dict = field(default_factory=dict)
    sink_rows: object = None        # LatitudeAccumulator (nlev, nlat)
    taper_rows: object = None       # LatitudeAccumulator (nlev, nlat)
    reverted: list = field(default_factory=list)
    points: int = 0
    record: dict = field(default_factory=dict)


class StochasticPhysics:
    """The member's pattern processes on one transform."""

    def __init__(self, config: StochasticConfig, transform):
        if not config.enabled:
            raise ValueError("StochasticPhysics is built only for an enabled config")
        self.config = config
        self.transform = transform
        radius = float(transform.grid.radius_m)
        self.radius = radius
        cap = int(transform.truncation)
        self.processes = []
        for name, length, tau in config.processes():
            degree = min(auto_degree(length, radius), cap)
            self.processes.append(_Process(
                name, float(length), float(tau), degree, config.process_key(name),
                degree_sigma(length, degree, radius)))
        self.max_degree = max(p.degree for p in self.processes)
        if config.skeb:
            p = self._process("skeb_psi")
            ell = np.arange(p.degree + 1, dtype=np.float64)
            # Expected RMS of one component of k x grad(psi) for the unit
            # pattern: E|grad psi|^2 = sum_l (2l+1) sigma_l^2 l(l+1)/(4 pi R^2),
            # half per component (isotropy).  hex: SphericalPattern.curl_rms.
            self.curl_rms = math.sqrt(float(np.sum(
                (2 * ell + 1) * p.sigma ** 2 * ell * (ell + 1))) / (8 * math.pi)) / radius
        else:
            self.curl_rms = 0.0

    def _process(self, name):
        for p in self.processes:
            if p.name == name:
                return p
        raise KeyError(name)

    # -- state ---------------------------------------------------------

    def _empty_state(self):
        n = self.max_degree + 1
        return (np.zeros((len(self.processes), 2, n, n), dtype=np.float64),
                np.array([-1.0, 0.0, 0.0], dtype=np.float64))

    def read_state(self, arrays, backend):
        coeff = arrays.get(COEFFICIENTS)
        clock = arrays.get(CLOCK)
        if coeff is None or clock is None:
            return self._empty_state()
        from .spill import SpilledArray

        def host(value):
            if isinstance(value, SpilledArray):
                return np.array(value.host, dtype=np.float64)
            return np.array(backend.to_numpy(value), dtype=np.float64)

        coeff, clock = host(coeff), host(clock)
        expected, _ = self._empty_state()
        if coeff.shape != expected.shape or clock.shape != (3,):
            raise ValueError(
                "the carried stochastic pattern state does not match this run's "
                f"[stochastic] settings (shape {coeff.shape}, expected {expected.shape}); "
                "a restart must keep the member's stochastic configuration")
        return coeff, clock

    def advance(self, coeff, clock, time_s: float):
        """Advance every process to ``time_s``; same clock, same pattern."""
        last, draw = float(clock[0]), int(clock[1])
        if last >= 0 and time_s == last:
            return coeff, clock, False
        if last >= 0 and time_s < last:
            raise ValueError("the stochastic pattern clock moved backwards; "
                             "restore its checkpoint")
        coeff = coeff.copy()
        for i, p in enumerate(self.processes):
            n = p.degree + 1
            noise = complex_noise(p.degree, p.key, draw) * p.sigma[:, None]
            current = coeff[i, 0, :n, :n] + 1j * coeff[i, 1, :n, :n]
            if last < 0:
                new = noise
            else:
                elapsed = (time_s - last) / p.time_s
                phi = math.exp(-elapsed)
                new = phi * current + math.sqrt(-math.expm1(-2 * elapsed)) * noise
            coeff[i, 0, :n, :n] = new.real
            coeff[i, 1, :n, :n] = new.imag
        period = 0.0 if last < 0 else time_s - last
        clock = np.array([time_s, draw + 1, period], dtype=np.float64)
        return coeff, clock, True

    def spectral(self, coeff, index: int):
        """Process ``index`` as a model spectral field (T+1, T+1)."""
        p = self.processes[index]
        n = p.degree + 1
        full = np.zeros(self.transform.spectral_shape, dtype=np.complex128)
        full[:n, :n] = coeff[index, 0, :n, :n] + 1j * coeff[index, 1, :n, :n]
        b = self.transform.backend
        return b.asarray(full, dtype=b.complex_dtype)

    def sppt_spectral(self, coeff):
        total = None
        for i, (amp, _l, _t) in enumerate(self.config.sppt_scales):
            piece = amp * self.spectral(coeff, i)
            total = piece if total is None else total + piece
        return total

    # -- one physics call -----------------------------------------------

    def begin(self, model, bundle, dt_s: float, step_dt_s: float | None):
        b = self.transform.backend
        coeff, clock = self.read_state(bundle.physics_state.arrays, b)
        coeff, clock, redrawn = self.advance(coeff, clock, float(bundle.time_s))
        ctx = CallContext()
        ctx.state = {COEFFICIENTS: b.asarray(coeff, dtype=np.float64),
                     CLOCK: b.asarray(clock, dtype=np.float64)}
        if self.config.sppt:
            ctx.sppt_grid = self.transform.inverse(self.sppt_spectral(coeff))
        if self.config.skeb:
            names = [p.name for p in self.processes]
            ctx.psi = self.spectral(coeff, names.index("skeb_psi"))
            ctx.thermal = self.spectral(coeff, names.index("skeb_thermal"))
            ctx.curl_rms = self.curl_rms
            period = float(step_dt_s or 0.0) or float(clock[2]) or float(dt_s)
            ctx.period_s = max(period, float(dt_s))
        if self.config.skeb or self.config.sppt:
            from .bands import LatitudeAccumulator

            xp = b.xp
            nlat = self.transform.grid.nlat
            exchange = model.pipeline.exchange
            if self.config.skeb:
                ctx.sink_rows = LatitudeAccumulator(
                    xp, (model.nlev, nlat), np.float64, name="stochastic_sink",
                    exchange=exchange)
                ctx.taper_rows = LatitudeAccumulator(
                    xp, (model.nlev, nlat), np.float64, name="stochastic_taper",
                    exchange=exchange)
        ctx.record = {"draw": int(clock[1]), "clock_s": float(clock[0]), "new_draw": bool(redrawn)}
        return ctx

    def taper(self, xp, geopotential, surface_geopotential):
        """mu(level, rows, lon): zero at the lowest level, linear in height
        above ground to one at ``surface_m``, linear to zero between
        ``top_start_m`` and ``top_end_m`` (hex StochasticPhysics.taper)."""
        c = self.config
        height = (geopotential - surface_geopotential[None]) / GRAVITY
        lower = xp.clip(height / c.surface_m, 0.0, 1.0)
        upper = xp.clip((c.top_end_m - height) / (c.top_end_m - c.top_start_m), 0.0, 1.0)
        mu = xp.minimum(lower, upper)
        mu[-1] = 0.0
        return mu

    def perturb_band(self, ctx, model, rows, inputs, result, exchange, *, dt_s: float,
                     fused: bool | None = None):
        """SPPT on one band's physics result, in place; SKEB's band readings.

        ``inputs`` holds the band's u, v, theta and qv as the suite received
        them.  Returns nothing; ``result`` u, v, theta, qv are replaced.  On
        cupy the whole band is one fused kernel (``fused``, default on for
        cupy); the array expressions below are its specification and the
        numpy path.
        """
        xp = self.transform.backend.xp
        dtype = self.transform.backend.float_dtype
        c = self.config
        if fused is None:
            fused = self.transform.backend.name == "cupy"
        if fused:
            return self._perturb_band_fused(ctx, model, rows, inputs, result, exchange,
                                            dt_s=dt_s)
        mu = self.taper(xp, exchange.geopotential, model.surface_geopotential[rows])
        du = xp.asarray(result.u, dtype=dtype) - inputs["u"]
        dv = xp.asarray(result.v, dtype=dtype) - inputs["v"]
        if ctx.sink_rows is not None:
            # hex _kinetic_sink: the positive KE sink of the UNPERTURBED
            # physics wind change, as a rate.
            sink = xp.maximum(-(inputs["u"] * du + inputs["v"] * dv) / dt_s, 0.0)
            nlon = sink.shape[-1]
            ctx.sink_rows.add_band(rows, xp.sum(sink.astype(xp.float64), axis=-1) / nlon)
            ctx.taper_rows.add_band(rows, xp.sum(mu.astype(xp.float64), axis=-1) / nlon)
        if not c.sppt:
            return
        r = ctx.sppt_grid[rows]
        if c.logit:
            bounded = c.clip * xp.tanh(r / c.clip)
        else:
            bounded = xp.clip(r, -c.clip, c.clip)
        factor = (mu * bounded[None]).astype(dtype)
        result.u = inputs["u"] + (1.0 + factor) * du
        result.v = inputs["v"] + (1.0 + factor) * dv
        del du, dv
        dtheta = xp.asarray(result.theta, dtype=dtype) - inputs["theta"]
        dq = xp.asarray(result.qv, dtype=dtype) - inputs["qv"]
        theta = inputs["theta"] + (1.0 + factor) * dtheta
        qv = inputs["qv"] + (1.0 + factor) * dq
        if c.moist_guard:
            # Palmer et al. (2009), hex stoch_moist_safe: negative or
            # supersaturated perturbed humidity reverts T and q at the point.
            temperature = theta * exchange.exner
            celsius = temperature - 273.15
            vapour = 611.2 * xp.exp(17.67 * celsius / (celsius + 243.5))
            p = exchange.p_full
            qsat = 0.62198 * vapour / xp.maximum(p - vapour, 1.0e-3 * p)
            unsafe = (factor != 0) & ((qv < 0) | (qv > qsat))
            theta = xp.where(unsafe, xp.asarray(result.theta, dtype=dtype), theta)
            qv = xp.where(unsafe, xp.asarray(result.qv, dtype=dtype), qv)
            ctx.reverted.append(xp.sum(unsafe))
            ctx.points += int(unsafe.size)
        result.theta = theta
        result.qv = qv

    def _perturb_band_fused(self, ctx, model, rows, inputs, result, exchange, *, dt_s: float):
        """:meth:`perturb_band` as one cupy elementwise kernel (same rules)."""
        import cupy as cp

        c = self.config
        dtype = np.dtype(self.transform.backend.float_dtype)
        f = dtype.type
        nlev = int(inputs["u"].shape[0])
        lowest = cp.zeros((nlev, 1, 1), dtype=dtype)
        lowest[-1] = 1
        r = ctx.sppt_grid[rows] if c.sppt else cp.zeros(inputs["u"].shape[1:], dtype=dtype)
        u, v, theta, qv, mu, sink, unsafe = _band_kernel(cp)(
            inputs["u"], inputs["v"], inputs["theta"], inputs["qv"],
            cp.asarray(result.u, dtype=dtype), cp.asarray(result.v, dtype=dtype),
            cp.asarray(result.theta, dtype=dtype), cp.asarray(result.qv, dtype=dtype),
            exchange.geopotential, model.surface_geopotential[rows], r,
            exchange.exner, exchange.p_full, lowest,
            f(c.surface_m), f(c.top_end_m), f(c.top_end_m - c.top_start_m), f(c.clip),
            np.int32(c.logit), np.int32(c.sppt), np.int32(c.moist_guard), f(1.0 / dt_s))
        if ctx.sink_rows is not None:
            nlon = sink.shape[-1]
            ctx.sink_rows.add_band(rows, cp.sum(sink, axis=-1, dtype=cp.float64) / nlon)
            ctx.taper_rows.add_band(rows, cp.sum(mu, axis=-1, dtype=cp.float64) / nlon)
        if c.sppt:
            result.u, result.v, result.theta, result.qv = u, v, theta, qv
            if c.moist_guard:
                ctx.reverted.append(cp.sum(unsafe, dtype=cp.int64))
                ctx.points += int(unsafe.size)

    def finish(self, ctx, zeta, theta_spec, *, dt_s: float):
        """SKEB's spectral forcing, applied to the analysed vorticity and
        theta of this call.  Returns (zeta, theta_spec, record)."""
        b = self.transform.backend
        xp = b.xp
        record = dict(ctx.record)
        if ctx.reverted:
            reverted = int(b.to_numpy(xp.sum(xp.stack(ctx.reverted))))
            record["sppt_moist_guard_reverted_fraction"] = reverted / max(ctx.points, 1)
        if self.config.skeb:
            weights = xp.asarray(0.5 * self.transform.grid.quadrature_weights, dtype=np.float64)
            dissipation = xp.maximum(ctx.sink_rows.total(weights), 0.0)   # (nlev,)
            level_taper = ctx.taper_rows.total(weights)                   # (nlev,)
            scale = backscatter_rate_scale(ctx.period_s, self.config.skeb_time_s)
            wind_amp = level_taper * xp.sqrt(scale * self.config.skeb_fraction * dissipation)
            heat_amp = level_taper * xp.sqrt(
                scale * self.config.thermal_fraction * 300.0 / 1004.0 * dissipation)
            lap = self.transform.laplacian(ctx.psi) / ctx.curl_rms
            forcing = (dt_s * wind_amp).astype(b.float_dtype)[:, None, None] * lap[None]
            zeta = self.transform.project(zeta + forcing.astype(zeta.dtype))
            heat = (dt_s * heat_amp).astype(b.float_dtype)[:, None, None] * ctx.thermal[None]
            theta_spec = self.transform.project(theta_spec + heat.astype(theta_spec.dtype))
            host = b.to_numpy(xp.stack([dissipation, wind_amp]))
            record["skeb_dissipation_w_kg_max"] = float(np.max(host[0]))
            record["skeb_wind_tendency_rms_max_m_s2"] = float(np.max(host[1]))
            record["skeb_period_s"] = float(ctx.period_s)
        return zeta, theta_spec, record

    def receipt(self) -> dict:
        return {
            "schema": SCHEMA,
            "config": self.config.identity(),
            "processes": [{"name": p.name, "length_m": p.length_m, "time_s": p.time_s,
                           "degree": p.degree, "key": f"{p.key:016x}"}
                          for p in self.processes],
            "correlation": "exp(-r**2/(2 L**2)) in space, exp(-dt/tau) in time",
            "ported_from": "this project's stochastic physics for the gpuwm-hex "
                           "model line, Apache-2.0 project code",
        }


_BAND_KERNEL = None


def _band_kernel(cp):
    """The fused SPPT/SKEB band kernel: taper, sink, bound, perturbation and
    moisture guard in one pass (the rules of :meth:`StochasticPhysics.perturb_band`)."""
    global _BAND_KERNEL
    if _BAND_KERNEL is None:
        _BAND_KERNEL = cp.ElementwiseKernel(
            "T ui, T vi, T ti, T qi, T uo, T vo, T to, T qo, T geo, T phis, T r, "
            "T exner, T p, T lowest, T surface_m, T top_end, T top_span, T clip, "
            "int32 logit, int32 sppt, int32 guard, T inv_dt",
            "T u, T v, T th, T q, T mu_out, T sink, int8 unsafe",
            """
            T h = (geo - phis) / (T)9.80665;
            T lower = min(max(h / surface_m, (T)0), (T)1);
            T upper = min(max((top_end - h) / top_span, (T)0), (T)1);
            T mu = lowest != (T)0 ? (T)0 : min(lower, upper);
            mu_out = mu;
            T du = uo - ui;
            T dv = vo - vi;
            T s = -(ui * du + vi * dv) * inv_dt;
            sink = s > (T)0 ? s : (T)0;
            unsafe = 0;
            if (!sppt) {
                u = uo; v = vo; th = to; q = qo;
            } else {
                T b = logit ? clip * tanh(r / clip) : min(max(r, -clip), clip);
                T f = mu * b;
                u = ui + ((T)1 + f) * du;
                v = vi + ((T)1 + f) * dv;
                T tp = ti + ((T)1 + f) * (to - ti);
                T qp = qi + ((T)1 + f) * (qo - qi);
                if (guard && f != (T)0) {
                    T tc = tp * exner - (T)273.15;
                    T e = (T)611.2 * exp((T)17.67 * tc / (tc + (T)243.5));
                    T qs = (T)0.62198 * e / max(p - e, (T)1.0e-3 * p);
                    if (qp < (T)0 || qp > qs) { tp = to; qp = qo; unsafe = 1; }
                }
                th = tp; q = qp;
            }
            """,
            "arwen_stochastic_band",
        )
    return _BAND_KERNEL


def state_digest(arrays, backend) -> str:
    """sha256 of the carried pattern state (tests and receipts)."""
    h = hashlib.sha256()
    for name in (COEFFICIENTS, CLOCK):
        if name in arrays:
            h.update(np.ascontiguousarray(backend.to_numpy(arrays[name])).tobytes())
    return h.hexdigest()


def describe(config: StochasticConfig) -> str:
    return json.dumps(config.identity(), sort_keys=True)
