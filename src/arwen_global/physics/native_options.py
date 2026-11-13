"""Pure, fail-closed options for the built-in Arwen CUDA column suite."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math

from ..core.ysu_contract import YSU_FREE_ATMOSPHERE_MIXING_LENGTHS

from ..constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT
from .cloud_radiation_options import (
    CLOUD_INHOMOGENEITY_FACTORS,
    RADIATION_LIQUID_RADIUS_MODES,
)


# Option value -> the name the component order reports for the slot.
CUMULUS_COMPONENTS = {"gf": "gf", "own": "arwen-massflux-v1", "ntiedtke": "ntiedtke"}

# Morrison droplet-number modes: WRF's single NDCNST, or the declared
# land/ocean departure (see NativePhysicsOptions.morrison_droplet_number).
MORRISON_DROPLET_NUMBER_MODES = frozenset({"fixed", "land_ocean"})
#: RRTMGP gas name -> (the option that overrides it, its unit as a mole
#: fraction).  The order is the order a receipt lists them in.
GAS_OPTIONS = {
    "co2": ("trace_co2_ppm", 1.0e-6),
    "ch4": ("trace_ch4_ppb", 1.0e-9),
    "n2o": ("trace_n2o_ppb", 1.0e-9),
}

#: [world.air] key -> the option it sets (arwen_global.world.spec.Air).
WORLD_AIR_OPTIONS = {
    "co2_ppm": "trace_co2_ppm",
    "ch4_ppb": "trace_ch4_ppb",
    "n2o_ppb": "trace_n2o_ppb",
}

#: The options that leave the identity (and the normalized options) when
#: they are None; trace_co2_ppm is not among them because it has carried
#: its None into every config hash since it existed.
_DATED_UNLESS_SET = ("trace_ch4_ppb", "trace_n2o_ppb")

#: Open upper bound of the methane and nitrous oxide options, ppb: the
#: world table's own bound (arwen_global.world.spec.TRACE_RANGE_PPB).
TRACE_PPB_LIMIT = 1.0e5


@dataclass(frozen=True)
class NativePhysicsOptions:
    acknowledgement: str
    # The instant model time zero stands for (arwen_global.clock).  An
    # analysis start takes it from the analysis frame's valid time, so a
    # config that initialises from an analysis leaves it out; a config that
    # states it anyway is cross-checked against the frame and refused when
    # the two disagree.  Only a run with no analysis needs to state it.
    start_time_utc: str | None = None
    radiation: str = "rrtmgp"
    radiation_interval_s: float = 1800.0
    radiation_column_chunk: int = 12_500
    radiation_validation_mode: str = "fused"
    # What the shortwave does between two radiation calls
    # (arwen_global.physics.radiation_between): "hold" (default) applies
    # the call's planes unchanged until the next call, as every run before
    # the option did; "surface" is WRF's swint_opt = 1 on the surface
    # downward shortwave, as the regional engine runs it (the surface
    # follows the sun every physics call); "column" adds the shortwave
    # heating rate and the top-of-atmosphere shortwave scaled by the
    # cosine ratio.  Joins the identity only when not "hold", so every
    # config hash and checkpoint written without it keeps its bytes.
    radiation_between: str = "hold"
    # The ozone the radiation reads (core.rrtmgp.OZONE_CLIMATOLOGIES):
    # "cam-zonal-monthly" (default, 2026-11-06), the CAM climatology WRF
    # and MPAS-A ship, by date, latitude and pressure; "rfmip-median", the
    # one global-median profile on every column and date this suite read
    # before.  Joins the identity only when "rfmip-median" is NOT chosen,
    # i.e. the new default moves the hash (it changes the arithmetic).
    ozone: str = "cam-zonal-monthly"
    # Which year's greenhouse gases the radiation reads
    # (physics.greenhouse_gases.GHG_YEAR_RULES), spelled in a config as
    # [radiation] ghg_year.  "start" (default): the start time's year for
    # the whole run, the arithmetic of every checkpoint before the option
    # existed, so it leaves the identity.  "valid": re-selected at every
    # radiation call from the call's valid year; joins the identity.
    ghg_year: str = "start"
    # The liquid effective radius the radiation reads for Morrison cloud
    # (physics.cloud_radiation_options).  "scheme" (default): Morrison's own
    # effc, what this package has always radiated.  "wrf_relcalc": what
    # WRF v4.7.1's RRTMG radiates for Morrison, which it leaves out of
    # has_reqc (module_physics_init.F:1000-1024): relcalc's 14 um over ocean
    # and sea ice, 8 um over warm snow-free land.  Joins the identity only
    # when not "scheme".
    radiation_liquid_radius: str = "scheme"
    # Unresolved in-cloud variability in the radiation.  "none" (default):
    # homogeneous cloudy subcolumns, WRF's RRTMG.  "tiedtke1996": liquid and
    # ice in-cloud paths times 0.7 in both solvers (Cahalan et al. 1994;
    # Tiedtke 1996), a declared departure from WRF.  Joins the identity only
    # when not "none".
    radiation_cloud_inhomogeneity: str = "none"
    surface_layer: str = "sfclay"
    sfclay_option: int = 1
    # Land thermal-roughness closure of the surface layer (sfclay.cu
    # iz0tlnd): 0 the Carlson-Boland scalar roughness WRF runs by default
    # (z0h fixed near 4e-4 m under the viscous term, whatever the canopy),
    # 1 Chen-Zhang 2009, 2 Zilitinkevich with czil 0.1, 3 Zilitinkevich
    # with the bare-soil czil 0.8 weighted by the square of the bare
    # fraction (Zheng et al. 2012, the GFS form): a vegetated column
    # exchanges heat with z0h near z0m, bare soil with a large kB^-1.
    # Read under 3 from the surface state's vegetation fraction.  Outside
    # 0 the option joins the identity; at 0 no arithmetic changes and a
    # checkpoint keeps its hash.
    sfclay_iz0tlnd: int = 0
    land_surface: str = "noah"
    land_surface_interval_s: float = 60.0
    pbl: str = "ysu"
    ysu_topdown_pblmix: int = 1
    # The asymptotic mixing length of YSU's local diffusivity above the
    # boundary layer (arwen_global.core.ysu_contract).  WRF ties it to the layer
    # thickness (0.1 dz clipped to 30..300 m, bl_ysu.F90:1003), so on this
    # suite's 40-level stack, whose layers across the jet are 55 hPa (about
    # 1500 m) thick, the length reads 150 m and the diffusivity 25x what a
    # 300 m regional layer gets for the same shear and Richardson number;
    # measured on the 2026-09-01 00Z T255 control the physics drained the
    # 100 to 400 km kinetic energy at 237 hPa at 0.7 to 1.2 per day, 99.6
    # percent of it this operator (the energy ledger).
    # "wrf-layer" is WRF's rule, bit for bit (the regional model's only
    # setting) and the DEFAULT here.  "fixed" holds the length at WRF's own
    # rlam = 30 m at every spacing, the value WRF's rule gives every layer
    # thinner than 300 m; nothing inside the boundary layer changes.  It is
    # selectable, not default, and is reported as a workaround: its 24 h
    # T255 grade (2026-09-05) lifted the northern 250 km energy ratio at
    # 250 hPa from 0.67 to 0.85 and improved the 250 hPa speed bias, but
    # worsened the northern 250 hPa vector wind rmse against the GFS f024
    # by 0.16 m/s (4.41 to 4.56, the gap opening from hour 9) and lifted
    # the grid-limit ratio above the diffusion package's (0.053 against
    # 0.038); the rule that admits a default asked for both not to worsen.
    # Identity: the flag joins the config identity only as "fixed" (see
    # ``identity``), so every checkpoint written under WRF's rule keeps
    # its hash, bare or with "wrf-layer" spelled.
    ysu_free_atmosphere_mixing_length: str = "wrf-layer"
    # Cumulus parameterization, default ON (fixed means default).  The
    # suite runs at 25-52 km, where the resolved grid cannot carry deep
    # convection: without a scheme every convective column waits for
    # grid-scale saturation and rains through Morrison as a grid-point
    # storm (audit 2026-09-01, the sixth native component).  "gf" is the
    # scale-aware WRF-v4.6.1 Grell-Freitas kernel behind arwen_global.core.gf,
    # run between YSU and Morrison in WRF's order; "ntiedtke" is the
    # WRF-v4.6.1 New Tiedtke scheme behind arwen_global.core.ntiedtke (the
    # regional cu_physics = 16, bitwise against the frozen Fortran at all
    # 21 stages), in the same slot; "none" is the research arm without
    # either.  gf_ishallow follows the MPAS seam
    # (gpuwm/core/mpas_column_batch.py: native MPAS hardwires ishallow=1),
    # read only where cumulus="gf"; the closure is the 16-member ensemble
    # (clos_choice=0), the only arm with oracle coverage, and is not an
    # option here.  "own" is Arwen Global's own closure, arwen-massflux-v1
    # (arwen_global/physics/arwen_massflux.py): a scale-aware
    # deep/shallow mass-flux scheme written for this Gaussian grid, this
    # 40-level split and this step, with convective momentum transport,
    # in the same slot.
    cumulus: str = "gf"
    gf_ishallow: int = 1
    # Read only where cumulus="gf".  WRF's GFDRV rejects a column whose
    # downdraft cannot form (not negatively buoyant, or nothing to
    # evaporate: every deep, saturated column) and hands it to the grid
    # scale; on this 50 km column the grid scale cannot carry deep
    # convection and the hand-off is the grid-point storm (the warm-pool
    # cell of the 2026-09-01 control: 1,200 J/kg of CAPE, seventeen
    # saturated levels, 84 mm/h of grid-scale rain beside 0.06 mm/h from
    # the scheme; the ierr 7 exits and cup_dd_moisture's ierr 51
    # zero-denominator exit, which the control's 311 mm central-Pacific
    # storm cell at 13.3N 211E carried on all 24 calls of its hour-12
    # trace).  True runs the deep arm updraft-only in
    # those columns (gf.cu GF_UPDRAFT_ONLY_WHEN_DOWNDRAFT_DRY); False is
    # WRF's kernel word for word and is the identity every Grell-Freitas
    # checkpoint before this option existed was pinned with, so only True
    # joins the hash.  Default False by the timing lane's grade of
    # 2026-09-05 on the 2026-09-01 00Z T255 case (both arms carrying the
    # advective forcing lanes): the switch halves the grid-scale cells
    # above 200 mm/day (19 to 12 on the export grid, 27 on the control)
    # and lowers the CONUS T2 and MSLP errors by 0.01 K and 0.004 hPa
    # beyond the lanes alone, but takes the northern 250 km energy ratio
    # from 0.806 to 0.791 and the grid-limit ratio from 0.561 to 0.545
    # (0.825 / 0.584 on the control: 0.034 / 0.039 below it, past the
    # 0.03 bar the grade set) and leaves the CONUS diurnal composite at
    # 12.93 LST against Stage-IV's 14.60, so it stays an opt-in until a
    # remedy meets the bar; the forcing lanes ship by themselves.
    gf_updraft_only_when_downdraft_dry: bool = False
    # Read only where cumulus="gf".  The coarse-column form of the deep
    # arm (gf.cu GF_RESOLVED_CONVERGENCE_CLOSURE), a declared divergence
    # from WRF v4.6.1 for a 52 km column that carries the whole of deep
    # convection as resolved forcing: (1) neg_check's 300.01 K/day
    # per-level heating cap, a bound tuned where the grid resolves part of
    # the convection, yields to the latent heat of the column's own
    # resolved moisture convergence (the cap becomes the larger of the
    # two); (2) the downdraft sweeps continue past levels the downdraft's
    # beta profile reaches with no mass, carrying the environment there,
    # where WRF divides by zero and exits with ierr 51 (the control's
    # central-Pacific storm at 13.3N 211E on all 24 calls of its hour-12
    # trace, 10,900 columns per call over the planet); (3) a downdraft that
    # cannot form (ierr 7) leaves the updraft running, the
    # gf_updraft_only_when_downdraft_dry behaviour; (4) the cloud-base mass
    # flux is floored at the ensemble's own Kuo moisture-convergence member
    # (a quarter of the sixteen-member mean's weight in WRF) times the
    # Kuo-Anthes share of the converged moisture that rains out (1 - b,
    # b = 10 (1 - RH) clipped to 0..1, RH the cloud layer's mean relative
    # humidity: nothing under 90 percent, all of it at saturation), so a
    # saturated convecting column precipitates the moisture the grid
    # converges into it instead of handing it to the microphysics as a
    # grid-point storm, and an unsaturated one moistens as WRF's members
    # already do.  True joins the hash
    # (a different trajectory); False is WRF's kernel word for word and
    # keeps the identity every earlier Grell-Freitas checkpoint carried.
    # Default False by the mass-flux lane's grade of 2026-09-05 on the
    # 2026-09-01 00Z T255 case (three arms, this form the last): it takes
    # the grid-scale cells above 200 mm/day from 27 to 0 and the heaviest
    # from 448 to 177 mm with every surface and upper-air score inside the
    # 0.03 bar, but takes the northern 250 km energy ratio from 0.825 to
    # 0.792 (0.034, past the 0.03 bar the grade set), the global grid-scale
    # rain from 1.500 to 1.187 mm/day (0.313, past the 0.3 bar) and the
    # CONUS diurnal composite only to 13.00 LST against Stage-IV's 14.60
    # (the bar was within 1 h), so it is an OPT-IN, a workaround for the
    # grid-point storms until a form meets the bar; the closure reading and
    # its census ship with either value.  The surface gains of that grade
    # (CONUS T2 rmse 2.548 to 2.481 K, MSLP rmse 1.676 to 1.575 hPa) are
    # the dynamics forcing lanes', not the arm's: with the arm off on the
    # same tree they read 2.477 K and 1.588 hPa, the northern 250 km ratio
    # 0.806 and the grid-scale rain 1.418 mm/day.  The floor is compared
    # with the mean after the diurnal-cycle term is taken off it, so a
    # column that term silenced convects on the floor alone (1,298 of
    # 15,357 deep-active columns at the arm's hour 12; the census names
    # them).
    gf_resolved_convergence_closure: bool = False
    # Read only where cumulus="ntiedtke": True runs the New Tiedtke kernels
    # with classic Tiedtke's deep closure (the fixed 2400 s adjustment time
    # and the moisture-convergence first guess in place of the
    # resolution-scaled time and the geometric fraction;
    # docs/cumulus-new-tiedtke.md).  False leaves every result bit-identical
    # to the port.  Outside cumulus="ntiedtke" the flag is not part of the
    # identity (see ``identity``), so a Grell-Freitas checkpoint keeps its
    # hash.
    ntiedtke_tiedtke_closure: bool = False
    # Columns the Grell-Freitas seam packs per pass (arwen_global.core.gf
    # GF_COLUMN_CHUNK).  MEASURED on the T533 40-level batch (1,283,202
    # columns, RTX 5090 32 GB): packed whole, the scheme's input and output
    # blocks were 3.07 + 3.28 GB and the run died at its first cumulus
    # call.  Column-independent, so the chunk moves memory and nothing else.
    cumulus_column_chunk: int = 131_072
    microphysics: str = "morrison"
    morr_rimed_ice: int = 1
    # Morrison's fixed cloud droplet number.  "fixed" (the default) is WRF's
    # INUM=1 arm with its single NDCNST = 250 cm-3
    # (module_mp_morr_two_moment.F:271-276), every result bit-identical to
    # the port.  "land_ocean" is a DECLARED DEPARTURE from WRF's Morrison:
    # the constant becomes one value per column, the land value on a column
    # the model calls land (land_fraction above one half, the same split as
    # native_runtime._xland) and the ocean value on every other column,
    # sea ice included.  The pair defaults to MPAS-Thompson's landmask
    # prescription (300 cm-3 land, 100 cm-3 ocean,
    # mpas_atmphys_init_microphysics.F init_thompson_clouddroplets_forMPAS),
    # which is the maritime/continental guidance WRF's own Thompson carries
    # beside its Nt_c (module_mp_thompson.F:80-88, WRF 4.7.1).  The droplet
    # number sets the cloud size distribution, so the liquid effective
    # radius RRTMGP reads, and Khairoutdinov-Kogan autoconversion
    # (PRC ~ N^-1.79, morrison.cu:413-414); both move.  See
    # docs/morrison-droplet-number.md.
    morrison_droplet_number: str = "fixed"
    morrison_droplet_number_land_cm3: float = 300.0
    morrison_droplet_number_ocean_cm3: float = 100.0
    dx_m: float = 50_000.0
    # Measured and reported, never enforced: the suite writes
    # native_water_residual_exceeds_tolerance into its diagnostics and no gate
    # consumes it.  validate() therefore refuses a caller-chosen value rather
    # than accepting a threshold that does nothing.
    water_fix_tolerance_kg_m2: float = 1.0e-5
    energy_change_advisory_j_m2: float = 5.0e4
    initial_pbl_height_m: float = 800.0
    initial_soil_temperature_k: float = 285.0
    initial_soil_water_fraction: float = 0.30
    # The per-column vegetation and soil categories live in the surface
    # state (arwen_global.statics: real WPS_GEOG fields, or the
    # declared synthetic planet); the former vegetation_category /
    # soil_category constants here were that planet and are retired.
    land_ice_category: int = 15
    urban_category: int = 13
    use_monthly_albedo: bool = False
    read_lai_2d: bool = True
    noah_thermal_conductivity_option: int = 1
    wrf_rrtmg_compatibility: str = "none"
    # THE WELL-MIXED GREENHOUSE GASES the radiation (RRTMGP) reads, one
    # override per gas.  None (the default for each) radiates that gas's
    # present-day mole fraction for the run's valid year from the dated
    # NOAA GML table in physics/greenhouse_gases.py; a number overrides that
    # gas alone, on every column and level, for the whole run.  Until
    # 2026-10-05 the CO2 default was a fixed 369.55 ppm, the air of about
    # 2000.
    #
    # trace_co2_ppm: CO2 in ppm (micromoles per mole of dry air), finite
    # and inside (0, 1e4).  trace_ch4_ppb and trace_n2o_ppb: methane and
    # nitrous oxide in ppb (nanomoles per mole), finite and inside
    # (0, 1e5).  The bounds are open because RRTMGP refuses a mole
    # fraction of exactly zero (core.rrtmgp.trace_gases).  What the bounds
    # do NOT say: this radiation's response to a gas is checked only
    # across the RFMIP clear-sky experiments (the CO2 runs reach eight
    # times the pre-industrial 284.3 ppm, 2,274 ppm), by
    # tests/test_arwen_global_world_air_device.py against the published
    # RTE-RRTMGP reference fluxes of those experiments.  A CO2 above eight
    # times pre-industrial is accepted and radiated, and is UNVALIDATED.
    #
    # Identity: a gas left on its dated table (None) is not in the
    # identity, so every checkpoint written before the methane and nitrous
    # oxide options existed keeps its hash; a number joins it (a different
    # trajectory).  CO2 has been in the identity since it was an option,
    # None included, and stays there as it was.
    #
    # A WORLD sets the same three gases from its [world.air] table
    # (co2_ppm, ch4_ppb, n2o_ppb, each "dated" or a number):
    # :func:`world_air_options` writes them onto these options when the
    # physics is built, never into the config, so the world's air joins the
    # config hash once, through the world's own identity.  A gas named in
    # both places is refused ("name it in one place").
    trace_co2_ppm: float | None = None
    trace_ch4_ppb: float | None = None
    trace_n2o_ppb: float | None = None
    # Stratospheric cold-top floor: OFF by default (retired 2026-10-05),
    # kept as an option for research arms.  It was installed after the 384 h
    # native baseline arm (T255, GDAS 2026-09-01 00Z) died at hour 83.4
    # ("temperature outside research bounds: 139.995..298.852 K") by a slow
    # top sag with winds healthy.  That death does not reproduce on the
    # 0.1.3 tree (the dynamics' lid read fixed): MEASURED 2026-10-05, the
    # same case run 240 h with the floor off (T255 L40 sl_si, RTX 4090)
    # passes every gate, its top full level (about 1.2 hPa) holds a minimum
    # potential temperature of 1290 K (about 189 K) at 240 h against 1332 K
    # (195 K) floored, and its global mean differs from the floored run's
    # by 0.12 K of potential temperature; the global minimum temperature
    # stays at 178.9 to 185.6 K in both arms, set by the analysed Antarctic
    # vortex, never near the 140 K bound.  The two arms are identical
    # through hour 144.  With no breakage left to name, a default-on pull of
    # the model top toward 195 K is an unbooked heat source, not a guard.
    # The top level's remaining cooling under the physics, about 8 to 11 K
    # a day at level 0 once the longwave radiates the air above the lid,
    # stays open on its own (CHANGELOG 0.1.3 Open).  At its former 50 hPa
    # reach the floor also
    # warmed ANALYSED Antarctic vortex air (4.6 percent of the GDAS points
    # above 50 hPa are below 195 K), so an arm that turns it on keeps the
    # 5 hPa reach.  Same one-sided Held-Suarez-style pull as reference.py
    # _stratospheric_floor_pull; stratospheric_floor_k <= 0 disables.
    stratospheric_floor_k: float = 0.0
    stratospheric_floor_pa: float = 500.0
    stratospheric_relaxation_time_s: float = 1_800.0
    # The open-water runaway watch (water.OPEN_WATER_EVAPORATION_NAME): the
    # most one open-water column may evaporate in a 24 h block of model
    # time before the run is refused, and whether it is refused or only
    # reported.  60 kg/m2 a day (a mean latent flux of about 1,700 W/m2) is
    # a provisional engineering value until a long run has measured the
    # distribution.  Both are refusal settings, not arithmetic: outside the
    # identity at every value, so a run that only reports shares its
    # lineage with the one that refuses.
    open_water_evaporation_ceiling_kg_m2_day: float = 60.0
    open_water_evaporation_refusal: bool = True
    # Whether the forecast loop books open water to the exchange account
    # (runner.books_open_water_exchange).  False keeps the finite surface
    # reservoir on open water, the bookkeeping of 0.1.8 and every earlier
    # release, so with it off a shipped config writes the checkpoint
    # arrays 0.1.8 wrote; the runaway watch above runs only with
    # the account.  No native kernel reads the reservoir, so the atmosphere
    # is the same either way: bookkeeping, outside the identity, and a run
    # with it off refuses a state that carries an account
    # (dynamics.ensure_exchange_account).
    open_water_exchange: bool = True
    # Noah's snow age (WRF SNOTIME) carried between land calls, so a pack
    # ages and its albedo decays as ALCALC intends (noah.cu:154-173).  Off
    # rebuilds the age as zeros on every call: the arithmetic of 0.1.5 and
    # every earlier release, where each pack sat at its fresh-snow albedo.
    # It changes the numbers of every run with snow, so on it joins the
    # identity: a config that sets it hashes apart from one that does not,
    # and a checkpoint written under the zero-age arithmetic is refused
    # rather than resumed with different physics.  Off is the default, so
    # a shipped forecast keeps 0.1.8's hash and arithmetic; a long run sets
    # it (docs/ARWEN_GLOBAL_LONG_RUNS.md), because a pack that lies for
    # weeks should darken.
    snow_ageing: bool = False
    # Orographic gravity-wave drag (arwen_global.physics.orographic_drag):
    # "none" (the default) or "kim-gwdo", WRF's gwd_opt = 1, the KIM
    # orographic gravity-wave drag with flow blocking (Choi and Hong 2015),
    # carried from the engine (arwen_global.core.terrain_drag) and added to
    # YSU's momentum tendencies from the column state YSU read, as WRF's PBL
    # driver adds it.  It reads the sub-grid orography statistics of
    # arwen_global.orographic_statics (gpuwm-global statics CONFIG
    # --orographic) and the grid length dx_m above, one value for the globe.
    # Before this option the model had no gravity-wave drag of any kind.
    # OFF by default because nothing has measured it on this model yet: the
    # long-run validation (plan WPG-check) decides the default.  Joins the
    # identity only when not "none", so every earlier checkpoint keeps its
    # hash.
    orographic_drag: str = "none"

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "NativePhysicsOptions":
        if not isinstance(raw, dict):
            raise TypeError("native adapter options must be a mapping")
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown native adapter options: " + ", ".join(unknown)
            )
        try:
            value = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid native adapter options: {exc}") from exc
        value.validate()
        return value

    @property
    def normalized(self) -> dict[str, object]:
        """Every option, validated, with its default filled in: what the
        suite is built from.  ``from_mapping(normalized)`` rebuilds these
        options exactly; ``from_mapping(identity)`` does NOT (an option the
        identity drops comes back as its default), so nothing that runs a
        scheme may be built from the identity.

        A methane or nitrous oxide override left at None is left out: the
        normalized options are stored in the config and written into
        receipts, which must read as they did before those options existed,
        and ``from_mapping`` fills the None back in."""
        payload = asdict(self)
        for name in _DATED_UNLESS_SET:
            if payload[name] is None:
                del payload[name]
        return payload

    @property
    def identity(self) -> dict[str, object]:
        payload = asdict(self)
        # A gas on its dated table changes nothing an earlier checkpoint
        # was written under (the options did not exist); a number does.
        for name in _DATED_UNLESS_SET:
            if payload[name] is None:
                del payload[name]
        # The identity carries every normalised option and the config hash
        # (and so every checkpoint's restart admission) is built from it.
        # A scheme-scoped flag joins only under the scheme that reads it:
        # under cumulus="gf" or "none" the New Tiedtke closure flag changes
        # no arithmetic, so it must not move the hash of a Grell-Freitas
        # checkpoint (tests/test_arwen_global_vertical_surface_stretched.py
        # pins that hash), while under cumulus="ntiedtke" the two closures
        # are different trajectories and never share one.
        if self.cumulus_scheme != "ntiedtke":
            del payload["ntiedtke_tiedtke_closure"]
        # WRF's thickness-scaled length is the arithmetic every checkpoint
        # before 2026-09-05 was written under, when the option did not
        # exist: spelling it must keep those hashes.  "fixed" is a
        # different trajectory and joins.
        if self.ysu_free_atmosphere_mixing_length == "wrf-layer":
            del payload["ysu_free_atmosphere_mixing_length"]
        if self.cumulus_scheme != "gf" or not self.gf_updraft_only_when_downdraft_dry:
            # The WRF-faithful state keeps the identity Grell-Freitas
            # checkpoints carried before the option existed.
            del payload["gf_updraft_only_when_downdraft_dry"]
        if self.cumulus_scheme != "gf" or not self.gf_resolved_convergence_closure:
            del payload["gf_resolved_convergence_closure"]
        if self.sfclay_iz0tlnd == 0:
            del payload["sfclay_iz0tlnd"]
        if self.radiation_between == "hold":
            # The held planes are the arithmetic every checkpoint before
            # the option carried.
            del payload["radiation_between"]
        if self.orographic_drag == "none":
            # The arithmetic every checkpoint before the option carried.
            del payload["orographic_drag"]
        if self.ozone == "rfmip-median":
            # The arithmetic every checkpoint before the option carried.
            del payload["ozone"]
        if self.ghg_year == "start":
            # The selection every checkpoint before the option carried.
            del payload["ghg_year"]
        # WRF's single droplet number is the arithmetic every checkpoint
        # before the option carried; under it the two values are not read.
        if self.morrison_droplet_number == "fixed":
            del payload["morrison_droplet_number"]
            del payload["morrison_droplet_number_land_cm3"]
            del payload["morrison_droplet_number_ocean_cm3"]
        # The two radiation couplings at their defaults are the call every
        # checkpoint before the options carried.
        if self.radiation_liquid_radius == "scheme":
            del payload["radiation_liquid_radius"]
        if self.radiation_cloud_inhomogeneity == "none":
            del payload["radiation_cloud_inhomogeneity"]
        # A config that follows its analysis states no clock; the analysis
        # (initial.analysis_grib, in the identity) is what dates the run.
        if self.start_time_utc is None:
            del payload["start_time_utc"]
        # Ageing on is a new trajectory and always joins the hash; off (the
        # default) is the zero-age arithmetic of 0.1.8 and earlier, and
        # leaves the hash every checkpoint written under it carries.
        if not self.snow_ageing:
            del payload["snow_ageing"]
        del payload["open_water_evaporation_ceiling_kg_m2_day"]
        del payload["open_water_evaporation_refusal"]
        del payload["open_water_exchange"]
        return payload

    @property
    def orographic_drag_enabled(self) -> bool:
        return self.orographic_drag != "none"

    @property
    def cumulus_enabled(self) -> bool:
        return self.cumulus_scheme is not None

    @property
    def cumulus_scheme(self) -> str | None:
        """``"gf"``, ``"ntiedtke"``, ``"own"`` or None for cumulus="none"."""
        name = str(self.cumulus).lower()
        return None if name == "none" else name

    @property
    def uses_grell_freitas(self) -> bool:
        return self.cumulus_scheme == "gf"

    @property
    def cumulus_component(self) -> str | None:
        """The name the component order carries for the cumulus slot."""
        scheme = self.cumulus_scheme
        return None if scheme is None else CUMULUS_COMPONENTS[scheme]

    @property
    def start_time(self) -> datetime:
        text = self.start_time_utc
        if text is None:
            raise ValueError(
                "the native suite has no forecast clock: start_time_utc is "
                "filled from the analysis valid time when the model is built "
                "(arwen_global.clock), or stated in the config for a run with "
                "no analysis"
            )
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        value = datetime.fromisoformat(text)
        if value.tzinfo is None:
            raise ValueError("start_time_utc must carry an explicit UTC offset")
        return value.astimezone(timezone.utc)

    def _gas_overrides(self) -> dict[str, float]:
        """Mole fraction of every gas whose option is set, by RRTMGP name."""
        out: dict[str, float] = {}
        for gas, (name, scale) in GAS_OPTIONS.items():
            value = getattr(self, name)
            if value is not None:
                out[gas] = float(value) * scale
        return out

    def greenhouse_gas_overrides(self, valid: datetime | None = None
                                 ) -> dict[str, float]:
        """The well-mixed gas mole fractions handed to RRTMGP over its
        climatology: the dated present-day CO2, CH4 and N2O of ``valid``'s
        year (the start time when None), with ``trace_co2_ppm``,
        ``trace_ch4_ppb`` and ``trace_n2o_ppb`` over their gas wherever one
        is set."""
        from .greenhouse_gases import present_day_greenhouse_gases

        gases = present_day_greenhouse_gases(
            self.start_time if valid is None else valid)
        gases.update(self._gas_overrides())
        return gases

    def greenhouse_gas_record(self, duration_s: float | None = None
                              ) -> dict[str, object]:
        """The receipt's record of what the radiation ran with, its source
        named, and which gases an option overrode.  Under
        ``ghg_year = "valid"`` it needs the run's ``duration_s`` for the
        year-by-year schedule."""
        from datetime import timedelta

        from .greenhouse_gases import greenhouse_gas_record

        overrides = self._gas_overrides() or None
        end = None
        if self.ghg_year == "valid":
            if duration_s is None:
                raise ValueError(
                    "the ghg_year = \"valid\" record needs the run's duration")
            end = self.start_time + timedelta(seconds=float(duration_s))
        return greenhouse_gas_record(
            self.start_time, overrides, rule=self.ghg_year, end=end)

    def validate(self) -> None:
        if self.acknowledgement != NATIVE_PHYSICS_ACKNOWLEDGEMENT:
            raise ValueError(
                "native adapter acknowledgement must be exactly "
                f"{NATIVE_PHYSICS_ACKNOWLEDGEMENT!r}"
            )
        if self.start_time_utc is not None:
            _ = self.start_time
        required = {
            "radiation": (self.radiation, "rrtmgp"),
            "surface_layer": (self.surface_layer, "sfclay"),
            "land_surface": (self.land_surface, "noah"),
            "pbl": (self.pbl, "ysu"),
            "microphysics": (self.microphysics, "morrison"),
        }
        for name, (actual, admitted) in required.items():
            if str(actual).lower() != admitted:
                raise ValueError(
                    f"Level-5 built-in adapter admits {name}={admitted!r}, "
                    f"got {actual!r}"
                )
        if str(self.cumulus).lower() not in {*CUMULUS_COMPONENTS, "none"}:
            raise ValueError(
                "Level-5 built-in adapter admits cumulus='gf' (WRF-v4.6.1 "
                "Grell-Freitas, arwen_global/core/kernels/gf.cu), cumulus='ntiedtke' "
                "(WRF-v4.6.1 New Tiedtke, arwen_global/core/kernels/ntiedtke.cu), "
                "cumulus='own' (arwen-massflux-v1, arwen_global/physics/"
                "arwen_massflux.py) or cumulus='none', "
                f"got {self.cumulus!r}"
            )
        if self.gf_ishallow not in {0, 1} or isinstance(self.gf_ishallow, bool):
            raise ValueError("gf_ishallow must be 0 or 1")
        if not isinstance(self.ntiedtke_tiedtke_closure, bool):
            raise ValueError("ntiedtke_tiedtke_closure must be true or false")
        if not isinstance(self.gf_updraft_only_when_downdraft_dry, bool):
            raise ValueError("gf_updraft_only_when_downdraft_dry must be true or false")
        if not isinstance(self.snow_ageing, bool):
            raise ValueError("snow_ageing must be true or false")
        if not isinstance(self.gf_resolved_convergence_closure, bool):
            raise ValueError("gf_resolved_convergence_closure must be true or false")
        if self.orographic_drag not in {"none", "kim-gwdo"}:
            raise ValueError(
                "orographic_drag must be 'none' or 'kim-gwdo' (WRF gwd_opt = 1, "
                "arwen_global/core/kernels/terrain_drag.cu), got "
                f"{self.orographic_drag!r}")
        if self.ozone not in {"cam-zonal-monthly", "rfmip-median"}:
            raise ValueError("ozone must be 'cam-zonal-monthly' or 'rfmip-median'")
        from .radiation_between import validate_mode

        validate_mode(self.radiation_between)
        if self.ghg_year not in {"start", "valid"}:
            raise ValueError(
                f"ghg_year must be 'start' or 'valid', got {self.ghg_year!r}")
        if self.radiation_validation_mode not in {"fused", "full"}:
            raise ValueError("radiation_validation_mode must be 'fused' or 'full'")
        if self.radiation_liquid_radius not in RADIATION_LIQUID_RADIUS_MODES:
            raise ValueError(
                "radiation_liquid_radius must be one of "
                f"{sorted(RADIATION_LIQUID_RADIUS_MODES)}, got "
                f"{self.radiation_liquid_radius!r}"
            )
        if self.radiation_cloud_inhomogeneity not in CLOUD_INHOMOGENEITY_FACTORS:
            raise ValueError(
                "radiation_cloud_inhomogeneity must be one of "
                f"{sorted(CLOUD_INHOMOGENEITY_FACTORS)}, got "
                f"{self.radiation_cloud_inhomogeneity!r}"
            )
        if self.sfclay_option not in {1, 91}:
            raise ValueError("sfclay_option must be 1 or 91")
        if self.sfclay_iz0tlnd not in {0, 1, 2, 3} or isinstance(self.sfclay_iz0tlnd, bool):
            raise ValueError("sfclay_iz0tlnd must be 0, 1, 2 or 3")
        if self.ysu_topdown_pblmix not in {0, 1}:
            raise ValueError("ysu_topdown_pblmix must be 0 or 1")
        # The mode table is the CARRIED contract's, and the kernel that
        # switches on it is carried beside it.  Until 2026-09-09 a second
        # refusal stood here for the engine that could not run "fixed" -- its
        # launcher took no flag and its kernel one fewer argument -- and it
        # is retired because that kernel now travels with this package.
        if self.ysu_free_atmosphere_mixing_length not in YSU_FREE_ATMOSPHERE_MIXING_LENGTHS:
            raise ValueError(
                "ysu_free_atmosphere_mixing_length must be one of "
                f"{sorted(YSU_FREE_ATMOSPHERE_MIXING_LENGTHS)}, got "
                f"{self.ysu_free_atmosphere_mixing_length!r}"
            )
        if self.morr_rimed_ice not in {0, 1}:
            raise ValueError("morr_rimed_ice must be 0 or 1")
        if self.morrison_droplet_number not in MORRISON_DROPLET_NUMBER_MODES:
            raise ValueError(
                "morrison_droplet_number must be one of "
                f"{sorted(MORRISON_DROPLET_NUMBER_MODES)}, got "
                f"{self.morrison_droplet_number!r}"
            )
        for name in ("morrison_droplet_number_land_cm3",
                     "morrison_droplet_number_ocean_cm3"):
            raw = getattr(self, name)
            # Morrison's own droplet bounds are a 1-60 um mean diameter
            # clamp, not a number range; 1 to 2000 cm-3 spans every
            # published cloud regime (Thompson's Nt_c_max is 1999 cm-3).
            if (isinstance(raw, bool) or not isinstance(raw, (int, float))
                    or not math.isfinite(float(raw))
                    or not 1.0 <= float(raw) <= 2000.0):
                raise ValueError(f"{name} must be a number in [1, 2000] cm-3")
        if self.noah_thermal_conductivity_option not in {1, 2}:
            raise ValueError("noah_thermal_conductivity_option must be 1 or 2")
        if self.wrf_rrtmg_compatibility not in {"none", "wrf-v4.6.1-v1", "wrf-v4.6.1-v2"}:
            raise ValueError("unknown wrf_rrtmg_compatibility identity")
        for name in (
            "radiation_interval_s", "land_surface_interval_s", "dx_m",
            "water_fix_tolerance_kg_m2", "energy_change_advisory_j_m2",
            "initial_pbl_height_m", "initial_soil_temperature_k",
            "initial_soil_water_fraction",
            "stratospheric_floor_k", "stratospheric_floor_pa",
            "stratospheric_relaxation_time_s",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not math.isfinite(float(raw)):
                raise ValueError(f"{name} must be finite")
        if self.trace_co2_ppm is not None and (
            isinstance(self.trace_co2_ppm, bool)
            or not math.isfinite(float(self.trace_co2_ppm))
            or not 0.0 < float(self.trace_co2_ppm) < 1.0e4
        ):
            raise ValueError(
                "trace_co2_ppm must be None (the dated present-day table) "
                "or a finite concentration in (0, 1e4) ppm"
            )
        for name in ("trace_ch4_ppb", "trace_n2o_ppb"):
            raw = getattr(self, name)
            if raw is None:
                continue
            if (isinstance(raw, bool) or not isinstance(raw, (int, float))
                    or not math.isfinite(float(raw))
                    or not 0.0 < float(raw) < TRACE_PPB_LIMIT):
                raise ValueError(
                    f"{name} must be None (the dated present-day table) or "
                    f"a finite concentration in (0, {TRACE_PPB_LIMIT:g}) "
                    f"ppb, got {raw!r}"
                )
        if self.radiation_interval_s <= 0.0 or self.land_surface_interval_s <= 0.0:
            raise ValueError("native physics intervals must be positive")
        if self.stratospheric_floor_pa <= 0.0:
            raise ValueError("stratospheric_floor_pa must be positive")
        if self.stratospheric_relaxation_time_s <= 0.0:
            raise ValueError("stratospheric_relaxation_time_s must be positive")
        if self.radiation_column_chunk < 1:
            raise ValueError("radiation_column_chunk must be positive")
        if (
            isinstance(self.cumulus_column_chunk, bool)
            or int(self.cumulus_column_chunk) != self.cumulus_column_chunk
            or self.cumulus_column_chunk < 1
        ):
            raise ValueError("cumulus_column_chunk must be a positive integer")
        if self.dx_m <= 0.0 or self.water_fix_tolerance_kg_m2 < 0.0:
            raise ValueError("dx_m must be positive and water tolerance nonnegative")
        if self.energy_change_advisory_j_m2 < 0.0:
            raise ValueError("energy_change_advisory_j_m2 must be nonnegative")
        if not 0.0 <= self.initial_soil_water_fraction <= 1.0:
            raise ValueError("initial_soil_water_fraction must lie in [0,1]")
        default_water_tolerance = type(self).__dataclass_fields__[
            "water_fix_tolerance_kg_m2"
        ].default
        if self.water_fix_tolerance_kg_m2 != default_water_tolerance:
            raise ValueError(
                "water_fix_tolerance_kg_m2 is measured and reported, not "
                "enforced: no gate consumes it, so setting "
                f"{self.water_fix_tolerance_kg_m2!r} would change nothing about "
                "when the run refuses.  Read the measured residual from the "
                "receipt tracker maximum_native_water_residual_kg_m2 instead."
            )
        for name in ("land_ice_category", "urban_category"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("use_monthly_albedo", "read_lai_2d"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        ceiling = self.open_water_evaporation_ceiling_kg_m2_day
        if (isinstance(ceiling, bool) or not isinstance(ceiling, (int, float))
                or not math.isfinite(float(ceiling)) or float(ceiling) <= 0.0):
            raise ValueError(
                "open_water_evaporation_ceiling_kg_m2_day must be a finite "
                "positive kg/m2 per 24 h; open_water_evaporation_refusal = "
                "false reports the watch without refusing")
        if not isinstance(self.open_water_evaporation_refusal, bool):
            raise ValueError("open_water_evaporation_refusal must be true or false")
        if not isinstance(self.open_water_exchange, bool):
            raise ValueError("open_water_exchange must be true or false")


def world_air_options(options: Mapping[str, object], world) -> dict[str, object]:
    """The native adapter options a run on ``world`` radiates with.

    ``options`` is the config's ``[physics] native_adapter_options`` (raw
    or normalized); ``world`` its ``cfg.world`` (or None).  For Earth, for
    None and for a world whose air is Earth's, the result is a plain copy of
    ``options``: no world code changes an Earth run.  Otherwise each gas
    the world's ``[world.air]`` names as a number (co2_ppm, ch4_ppb,
    n2o_ppb) is written onto its option (trace_co2_ppm, trace_ch4_ppb,
    trace_n2o_ppb); a gas left "dated" keeps the option's own value.

    Refused by name: a gas the world sets that the options also set (the
    config reader refuses the CO2 pair at load; this is the same rule at the
    point the physics is built, for every gas).

    The result is NOT for the config: the world joins the config hash
    through its own identity, and writing its gases into the stored options
    as well would count them twice and make a config written back out from
    them carry the same gas in two places."""
    out = dict(options)
    if world is None or world.is_earth or not world.changes_air:
        return out
    for key, name in WORLD_AIR_OPTIONS.items():
        value = getattr(world.air, key)
        if isinstance(value, str):
            # "dated": the option (None, or an arm's own number) stands.
            continue
        if out.get(name) is not None:
            raise ValueError(
                f"world.air.{key} and [physics] native_adapter_options "
                f"{name} both set the {key.split('_')[0].upper()}: name it "
                "in one place")
        out[name] = float(value)
    return out


def world_air_physics_refusal(world, physics_mode: str) -> str | None:
    """Why ``world``'s air cannot be radiated under ``physics_mode``, or
    None.  Only the native suite's RRTMGP reads trace gases; the reference
    suite's gray radiation has no CO2, methane or nitrous oxide in it, and
    physics "none" has no radiation at all, so a changed air there would be
    a label on an unchanged run."""
    if world is None or world.is_earth or not world.changes_air:
        return None
    if physics_mode == "arwen-native":
        return None
    gases = ", ".join(
        f"world.{name}" for name in world.changed_settings()
        if name.startswith("air."))
    return (
        f"world {world.label!r} changes its air ({gases}), "
        f"and physics.mode = {physics_mode!r} radiates no trace gas: only "
        "the native suite's RRTMGP reads CO2, methane and nitrous oxide, so "
        "set physics.mode = \"arwen-native\" or leave [world.air] dated")


__all__ = [
    "CUMULUS_COMPONENTS", "GAS_OPTIONS", "NativePhysicsOptions",
    "TRACE_PPB_LIMIT", "WORLD_AIR_OPTIONS", "world_air_options",
    "world_air_physics_refusal",
]
