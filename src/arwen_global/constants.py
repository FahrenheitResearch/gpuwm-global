"""Physical constants and serialized identities for Arwen Global."""
from __future__ import annotations

from arwen_global.spectral.constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EARTH_RADIUS_M,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    SECONDS_PER_DAY,
)

WATER_VAPOR_GAS_CONSTANT = 461.5
EPSILON = DRY_AIR_GAS_CONSTANT / WATER_VAPOR_GAS_CONSTANT
LATENT_HEAT_VAPORIZATION = 2.5e6
LATENT_HEAT_FUSION = 3.34e5
LATENT_HEAT_SUBLIMATION = LATENT_HEAT_VAPORIZATION + LATENT_HEAT_FUSION
STEFAN_BOLTZMANN = 5.670374419e-8
LIQUID_WATER_DENSITY = 1000.0

RESEARCH_ACKNOWLEDGEMENT = "research-only-arwen-global-v1"
NATIVE_PHYSICS_ACKNOWLEDGEMENT = "device-pending-arwen-native-physics-v1"
RUN_SCHEMA = "gpuwm.arwen-global-run/v1"
#: v3: the ten condensate and number-moment fields are real grid arrays
#: (nlev, nlat, nlon); v2 carried all fifteen as spectral coefficients.
CHECKPOINT_SCHEMA = "gpuwm.arwen-global-checkpoint/v3"
#: [output] checkpoint_compression: how a checkpoint's members sit in its
#: ``.npz`` container (checkpoint.write_checkpoint_arrays).  "none" stores
#: them, "deflate" is zlib level 6.  The arrays and every hash are the same.
CHECKPOINT_COMPRESSIONS = ("none", "deflate")
DEFAULT_CHECKPOINT_COMPRESSION = "none"
#: [output] snapshot_buffer: where a run's checkpoint is copied to on the
#: host (checkpoint.PinnedSnapshot): one page-locked buffer kept for the
#: run, or a fresh pageable array per checkpoint as before the option.
SNAPSHOT_BUFFERS = ("pinned", "pageable")
DEFAULT_SNAPSHOT_BUFFER = "pinned"
#: v4: a two-time-level integrator carries a second time level, and a
#: restart that rebuilt it from a start-up step would not be the
#: continuation of the uninterrupted run -- which is the property the
#: device-qualification pin asserts.  Written ONLY by a run whose
#: integrator carries one; every other run keeps writing v3, byte for
#: byte, and a v3 archive carrying a trajectory array is refused by name.
SEMILAG_CHECKPOINT_SCHEMA = "gpuwm.arwen-global-checkpoint/v4"
SPECTRAL_TRACER_CHECKPOINT_SCHEMA = "gpuwm.arwen-global-checkpoint/v2"
LEVEL4_CHECKPOINT_SCHEMA = "gpuwm.arwen-global-checkpoint/v1"
RECEIPT_SCHEMA = "gpuwm.arwen-global-receipt/v2"
EXPORT_SCHEMA = "gpuwm.arwen-global-parent-export/v3"
PHYSICS_EXCHANGE_SCHEMA = "gpuwm.arwen-global-physics-exchange/v2"
PHYSICS_ADAPTER_SCHEMA = "gpuwm.arwen-global-native-physics-adapter/v2"
PHYSICS_STATE_SCHEMA = "gpuwm.arwen-global-physics-state/v1"
NATIVE_DEVICE_EVIDENCE_SCHEMA = "gpuwm.arwen-global-native-device-evidence/v1"
NATIVE_CONTRACT_CANDIDATE_SCHEMA = "gpuwm.arwen-global-native-contract-candidate/v1"
LEVEL4_MIGRATION_SCHEMA = "gpuwm.arwen-global-level4-migration/v1"
ASSIMILATION_SCHEMA = "gpuwm.arwen-global-assimilation/v1"
REGIONAL_TARGET_SCHEMA = "gpuwm.arwen-global-regional-target/v1"
REGIONAL_FRAME_SCHEMA = "gpuwm.arwen-global-regional-frame/v1"
REGIONAL_SERIES_SCHEMA = "gpuwm.arwen-global-parent-series/v1"
REGIONAL_INSTALL_SCHEMA = "gpuwm.arwen-global-regional-install/v1"
REGIONAL_ATTACH_SCHEMA = "gpuwm.arwen-global-regional-attach/v1"

#: Physics-state array holding accumulated convective precipitation
#: (kg/m2, WRF's RAINC).  A cumulus scheme books its surface rain here and
#: the render tape's RAINC reads it; a physics state without it exports
#: zeros, WRF's own convention for a run with no cumulus scheme.
CONVECTIVE_RAIN_ACCUMULATOR = "rainc"

WATER_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER_MOMENTS = ("nc", "nr", "ni", "ns", "ng")
ADVECTED_TRACERS = (*WATER_SPECIES, *NUMBER_MOMENTS)
CONDENSATE_SPECIES = ("qc", "qr", "qi", "qs", "qg")
EFFECTIVE_RADIUS_FIELDS = ("effc", "effr", "effi", "effs")
#: The prognostic fields that live in the spectral basis: the dynamical
#: quartet and water vapor.  Vapor is smooth enough for a truncated basis
#: (its measured ringing fraction is 0.00 percent of its mass); the
#: condensate species and the number moments are not (finding
#: 2026-09-02: 37 percent of the planet's cloud water and 9-32 percent
#: of its rain, ice and graupel were moved out of their columns on every
#: positivity pass, four per step, and grid-scale precipitation never
#: reached the ground).
SPECTRAL_FIELDS = (
    "vorticity",
    "divergence",
    "theta",
    "log_surface_pressure",
    "qv",
)
#: The prognostic fields that live on the Gaussian grid, transported by
#: the positive-definite flux-form scheme (arwen_global.transport)
#: and never analysed into the spectral basis.
GRID_TRACERS = (*CONDENSATE_SPECIES, *NUMBER_MOMENTS)
#: Every prognostic field of the atmosphere in the state's own order
#: (the order every checkpoint, pin document and export lists them in).
PROGNOSTIC_FIELDS = (*SPECTRAL_FIELDS, *GRID_TRACERS)
LEVEL4_SPECTRAL_FIELDS = (
    "vorticity",
    "divergence",
    "theta",
    "log_surface_pressure",
    *WATER_SPECIES,
)

#: The Eulerian core's shipped step is the largest whole step that keeps the
#: strongest analysis day on disk at or under this fraction of the spectral
#: CFL gate (config.default_eulerian_step_s); the CFL refusal in dynamics
#: names the step this fraction would have admitted.  Here, with no imports,
#: because both the config parser and the dynamics read it.
DEFAULT_EULERIAN_STEP_RULE_FRACTION = 0.70

#: What the global mass fixer holds ([repair] mass_fixer_target): the
#: global-mean surface pressure ("total", shipped) or the global-mean dry
#: surface pressure ps - g W ("dry").  Here because the config parser and
#: the dynamics both read it.
MASS_FIXER_TARGETS = ("total", "dry")

#: The tightest [gates] mass_relative_drift a "dry" mass_fixer_target admits,
#: and its default there.  The fixer holds the dry mass of the state leaving
#: the dynamics; the receipt reads it after the second physics half, which
#: books that half step's E - P at a fixed surface pressure, so the output
#: row departs from the target by g (E - P) dt/2 and the next step re-anchors
#: it.  Measured on the 6-hourly global_mean_dry_surface_pressure_pa rows of
#: every lease's diagnostics.jsonl (fx/energy-mass proof arms, GDAS
#: 2026-08-01 00Z, all three switches, float32): 1.33e-6 at most over 60
#: T127 days (dt 600 s, 246 rows, worst at day 29.5) and 1.23e-6 over 10
#: T255 days (dt 300 s, 42 rows).  The shipped total-target gate (1e-9)
#: would fail on weather, not on the fixer.  Also the departure beyond which
#: a branch from a checkpoint that held the other quantity opens a new mass
#: epoch (runner.branch_mass_epoch).
DRY_TARGET_MASS_RELATIVE_DRIFT_FLOOR = 1.0e-5
