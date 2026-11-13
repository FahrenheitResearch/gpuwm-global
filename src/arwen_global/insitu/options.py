"""[insitu] configuration: cadences for the in-situ ledger runtime.

The ledger is DIAGNOSTIC ONLY.  Nothing in this table enters
``config_hash`` and nothing here may change a prognostic bit (proved by
``tests/test_arwen_global_insitu.py::test_ledger_on_and_off_are_bit_identical``).
It ships enabled by default: every failure of the week of 2026-08-31 (the
day-2 jet event, the reservoir-floor death at hour 37.7, the polar sag) was
diagnosed post-mortem from three-hourly checkpoints, and the whole point of
the instrument is to have watched every step and to have snapshotted the
state at onset.

THE CADENCES IN SECONDS.  ``flush_every``, ``spectra_every`` and
``energy_every`` count STEPS, so at a longer step they sample less of the
forecast: the spectral tripwire compares each sample with a 6-sample
trailing mean, which is 5 h of forecast at 300 s and 15 h at 900 s.  Each
of the three may instead be given in seconds of forecast time
(``flush_every_s``, ``spectra_every_s``, ``energy_every_s``); the run then
samples at the same forecast times whatever its step, and the tripwire
window is the same length of forecast.  A cadence in seconds must be a
whole multiple of ``[time] dt_s`` (a cadence between steps cannot be
sampled), and a table that gives one cadence both ways is refused.  The
step is counted from the forecast start (the bundle's own step), so a
restarted segment samples the same steps a straight run does.  Unset, the
step counts stand exactly as before, and the ledger's header and the
receipt's in-situ block carry the same keys they always did.

THE STRATOSPHERIC NOISE GATE (``stratospheric_noise_limit``,
``stratospheric_noise_band_hpa``; ``insitu.stratosphere``) is off unless a
limit is set, and is a diagnostic: it reports in the receipt's in-situ
block and never moves the run's status.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

#: The step-counted cadences that may also be given in seconds.
SECONDS_CADENCES = ("flush_every", "spectra_every", "energy_every")


@dataclass(frozen=True)
class InsituOptions:
    enabled: bool = True
    # Every step's scalar ledger row is computed on the device and buffered
    # there; every ``flush_every`` steps ONE batched device-to-host read
    # lands the buffer in insitu.ndjson.  The tripwires are evaluated at
    # the flush, so a trip snapshot can lag onset by up to flush_every - 1
    # steps; flush_every = 1 makes the snapshot step-exact at the price of
    # one host read per step (a device sync on the cupy backend).
    flush_every: int = 10
    # Spectral kinetic energy by total degree (rotational and divergent,
    # per level) from the vorticity/divergence coefficients the dycore
    # already holds - no transform is paid, so this is nearly free.
    spectra_every: int = 10
    # Per-component physics tendency capture (global-mean and max-abs
    # change of theta, qv, u, v per component per call).  Each mark copies
    # the four fields on the device (the reference suite's convective
    # adjustment and the native kernels update in place, so a
    # reference-holding capture would read zero change) - measured at 0.3%
    # of a T63/40-level step on one CPU core (2026-09-01), so every step
    # is the default.
    tendencies_every: int = 1
    # Per-operator total-energy ledger (insitu.energy): every
    # ``energy_every`` steps the atmosphere is marked around each operator
    # of the step and the per-level net change of kinetic, internal and
    # potential energy is booked.  Marks at every step cost 5.4% of a
    # T255 native IMEX step and 5.9% of a split step on the card
    # (insitu.energy records the measurement), so the default samples
    # every 50 steps: 0.11% of the run.
    energy_every: int = 50
    # Rows of ledger history written beside a trip snapshot.
    history_rows: int = 50
    # A tripwire that trips writes a checkpoint snapshot of the state at
    # the flush it tripped in (the ONSET snapshot); each named tripwire
    # snapshots at most once per run, and this caps the total so a run
    # that trips everything cannot fill the disk with checkpoints.
    snapshot_on_trip: bool = True
    max_snapshots: int = 8
    tripwires: bool = True
    # The three cadences in seconds of forecast time (module docstring).
    # None keeps the step count above; set, it replaces that count once
    # the step is known (resolved_for_step).
    flush_every_s: float | None = None
    spectra_every_s: float | None = None
    energy_every_s: float | None = None
    # The stratospheric noise gate (insitu.stratosphere): a limit on the
    # top-decile share of kinetic energy on the levels in the band, read
    # at every spectra sample.  None (default) is off.  Diagnostic and
    # uncalibrated: it reports in the receipt's in-situ block and never
    # moves the run's status.
    stratospheric_noise_limit: float | None = None
    stratospheric_noise_band_hpa: tuple[float, float] = (10.0, 100.0)

    def __post_init__(self) -> None:
        for name in (
            "flush_every", "spectra_every", "tendencies_every",
            "energy_every", "history_rows", "max_snapshots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"insitu.{name} must be a positive integer")
        for name in SECONDS_CADENCES:
            value = getattr(self, f"{name}_s")
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(
                    f"insitu.{name}_s must be a positive number of seconds"
                )
        limit = self.stratospheric_noise_limit
        if limit is not None and (
                isinstance(limit, bool) or not isinstance(limit, (int, float))
                or not math.isfinite(float(limit)) or float(limit) <= 0.0):
            raise ValueError("insitu.stratospheric_noise_limit must be a positive number")
        band = self.stratospheric_noise_band_hpa
        if (not isinstance(band, (tuple, list)) or len(band) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(float(v)) for v in band)
                or not 0.0 < float(band[0]) < float(band[1])):
            raise ValueError(
                "insitu.stratospheric_noise_band_hpa must be two ascending "
                "positive pressures in hPa"
            )
        object.__setattr__(
            self, "stratospheric_noise_band_hpa", (float(band[0]), float(band[1]))
        )

    def cadence_steps(self, name: str, dt_s: float) -> int:
        """The cadence ``name`` (one of SECONDS_CADENCES) in steps of
        ``dt_s``: the step count when no seconds are set, else the seconds
        divided by the step, refused when that is not a whole number."""
        if name not in SECONDS_CADENCES:
            raise ValueError(f"insitu.{name} has no cadence in seconds")
        seconds = getattr(self, f"{name}_s")
        if seconds is None:
            return int(getattr(self, name))
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("the step must be a positive number of seconds")
        ratio = float(seconds) / dt_s
        steps = int(round(ratio))
        if steps < 1 or abs(ratio - steps) > 1.0e-9 * max(1.0, ratio):
            raise ValueError(
                f"insitu.{name}_s = {float(seconds):g} s is not a whole "
                f"number of {dt_s:g} s steps; a cadence between steps cannot "
                f"be sampled (the nearest whole cadence is "
                f"{max(1, steps) * dt_s:g} s)"
            )
        return steps

    def resolved_for_step(self, dt_s: float) -> "InsituOptions":
        """These options with every cadence given in seconds turned into
        its step count at ``dt_s``.  Without a cadence in seconds this is
        ``self`` unchanged."""
        if all(getattr(self, f"{name}_s") is None for name in SECONDS_CADENCES):
            return self
        return replace(self, **{
            name: self.cadence_steps(name, dt_s) for name in SECONDS_CADENCES
        })

    @property
    def identity(self) -> dict[str, object]:
        # A cadence in seconds appears only when set, so a ledger header
        # and a receipt written without one keep the keys they always had.
        payload = asdict(self)
        for name in SECONDS_CADENCES:
            if payload[f"{name}_s"] is None:
                del payload[f"{name}_s"]
        # Likewise the stratospheric noise gate, which is off unless set.
        if payload["stratospheric_noise_limit"] is None:
            del payload["stratospheric_noise_limit"]
            del payload["stratospheric_noise_band_hpa"]
        else:
            payload["stratospheric_noise_band_hpa"] = list(
                payload["stratospheric_noise_band_hpa"]
            )
        return payload


def insitu_options_from_table(table: dict) -> InsituOptions:
    allowed = set(InsituOptions.__dataclass_fields__)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in [insitu]: {', '.join(unknown)}")
    kwargs = {}
    for name in ("enabled", "snapshot_on_trip", "tripwires"):
        if name in table:
            if not isinstance(table[name], bool):
                raise ValueError(f"insitu.{name} must be true or false")
            kwargs[name] = table[name]
    for name in (
        "flush_every", "spectra_every", "tendencies_every", "energy_every",
        "history_rows", "max_snapshots",
    ):
        if name in table:
            kwargs[name] = table[name]
    for name in SECONDS_CADENCES:
        key = f"{name}_s"
        if key in table:
            if name in table:
                raise ValueError(
                    f"[insitu] gives {name} both in steps ({name}) and in "
                    f"seconds ({key}); give one"
                )
            kwargs[key] = table[key]
    if "stratospheric_noise_band_hpa" in table:
        if "stratospheric_noise_limit" not in table:
            raise ValueError(
                "[insitu] stratospheric_noise_band_hpa is read only by the "
                "stratospheric noise gate; set stratospheric_noise_limit too"
            )
        band = table["stratospheric_noise_band_hpa"]
        if not isinstance(band, list):
            raise ValueError(
                "insitu.stratospheric_noise_band_hpa must be two ascending "
                "positive pressures in hPa"
            )
        kwargs["stratospheric_noise_band_hpa"] = tuple(band)
    if "stratospheric_noise_limit" in table:
        kwargs["stratospheric_noise_limit"] = table["stratospheric_noise_limit"]
    return InsituOptions(**kwargs)


__all__ = ["InsituOptions", "SECONDS_CADENCES", "insitu_options_from_table"]
