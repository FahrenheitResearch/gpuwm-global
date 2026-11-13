# Thin air and the carried physics: where the kernels read an absolute pressure

A world whose surface pressure is well below Earth's is listed as **not
supported** for the full physics. The reason given was that both convection
schemes and the microphysics carry thresholds written in absolute hPa or Pa,
which nobody had read one by one, so on thin air they would be silently wrong
rather than refused. This page is that reading. It lists every place the
convection and microphysics code compares or caps an absolute pressure, says
what each one decides, whether it reaches the model's tendencies at all, and
what it would have to scale with on a thinner world. It ends with what thin
air needs before the label can change.

Nothing here changes a kernel. It is a reading of the source this package
ships, held to that source by
`tests/test_arwen_global_worlds_pressure_audit.py`: every row below names the
exact source line it describes, and the test fails if a re-cut of the carried
kernels moves, removes or adds one.

```
python -m pytest tests/test_arwen_global_worlds_pressure_audit.py
```

## What was read

| Scheme | Option | Source |
|---|---|---|
| Grell-Freitas, deep and shallow (WRF v4.6.1) | `cumulus = "gf"`, the adapter default; no shipped config sets `cumulus` | `src/arwen_global/core/kernels/gf.cu` |
| New Tiedtke (WRF v4.6.1) | `cumulus = "ntiedtke"` | `src/arwen_global/core/kernels/ntiedtke.cu` |
| This package's own mass-flux scheme | `cumulus = "own"` | `src/arwen_global/physics/arwen_massflux.py` |
| Morrison two-moment microphysics (WRF v4.6.1) | the shipped microphysics | `src/arwen_global/core/kernels/morrison.cu` |

The device kernels were read rather than the Python wrappers because the
kernels are what runs; the wrappers convert units at the boundary (the
Grell-Freitas driver multiplies Pa by 0.01 to work in hPa, New Tiedtke works
in Pa throughout) and hold no threshold of their own.

## How to read the tables

**Live** means the threshold changes what the scheme hands back to the model.
**Inert** means changing its value, within the range the row gives, changes
nothing the scheme hands back: it is computed and captured for the parity
suites and nothing else reads it, or only its side of another threshold
matters. It is listed so that nobody spends effort scaling it.

The last column says what the threshold should scale with on a world whose
surface pressure is not Earth's:

- **scale with s**: the threshold is a pressure depth standing for a fraction
  of the column (a mixed layer, a cloud depth, an inhibition layer). On a
  world whose mean surface pressure is `p_world` it should be multiplied by
  one world-wide factor `s = p_world / 1013.25 hPa`. It must not be scaled by
  each column's own surface pressure: Earth's high ground already runs these
  schemes far below 1000 hPa, as WRF intends, and a per-column factor would
  change Earth's own results there.
- **replace**: an absolute pressure used for something a pressure is not, such
  as a flag. It should be replaced by what it stands for.
- **leave**: the pressure enters as the physics needs it (a density, a
  diffusivity), or as a ratio that is already relative.
- **mass**: a cap on a mass flux in kg m-2 s-1. A thinner column holds less
  mass, so the cap becomes looser; it should scale with s as well, but it is
  a ceiling rather than a trigger, and how often it binds was not measured
  here, so it comes second.

## Grell-Freitas (`gf.cu`)

| ID | Where | Value | What it decides | Live? | On a thinner world |
|---|---|---|---|---|---|
| GF-1 | deep: `gfd_cup_kbcon`, called at `iloop = 1` (module_cu_gf_deep.F:2722-2858, the cap set at :409-433) | `cap_maxs` 75 hPa | The largest pressure depth allowed between the updraft's source level (k22) and its level of free convection (kbcon). A deeper gap moves the source up one level and tries again; past `kbmax + 2` the column is refused (ierr 3). | live | scale with s |
| GF-2 | deep, same block | plus or minus 25 hPa | Over land only, the cap above is raised by 25 hPa where the surface buoyancy excess (`ztexec`) is positive and lowered by 25 hPa where it is negative. | live | scale with s |
| GF-3 | deep: `gfd_cup_kbcon` at `iloop = 2` and `3` (:1633-1654) | `cap_max_increment` 20 hPa | Two probes with the cap tightened by 20 and 40 hPa. Over water, a column that fails either has every closure member multiplied by `ens_adj` (`if (ierr2 > 0 || ierr3 > 0)`), but `ens_adj` is 1 on every path, and the probes' source and cloud-base levels are read by nothing else. | inert | leave |
| GF-4 | `gfd_cup_kbcon`, every `iloop` below 5 | floor 25 hPa | The tightened cap never falls below 25 hPa. At `iloop = 1` the cap is at least 50 hPa, so the floor can bind only in the inert probes of GF-3. | inert | leave |
| GF-5 | shallow: `gfd_cup_kbcon` at `iloop = 5` (module_cu_gf_sh.F:402-407) | 150 hPa | The largest gap allowed between the source level, or the boundary-layer top (GF-6), and the level of free convection. | live | scale with s |
| GF-6 | shallow, same call; the cap set at :370-383 | 200 hPa as a test | When the boundary-layer top index `kpbl` exceeds 3 the shallow cap is overwritten with the PRESSURE at the boundary-layer top, and `cap_max > 200` is how `gfd_cup_kbcon` tells that pressure from a depth: above 200 the gap is measured from the boundary-layer top. Where the pressure at the boundary-layer top is 200 hPa or less, the test takes it for the 125 hPa default and measures the gap from the source level instead. | live | replace by the condition it stands for (`kpbl > 3`) |
| GF-7 | shallow, the cap set at :287-298 and :241-256 | `cap_maxs` 125 hPa, increment 25 hPa | Never used as a depth: at `iloop = 5` the allowance is the fixed 150 hPa of GF-5 and the increment is not read. The 125 only makes `cap_max > 200` false when `kpbl <= 3`. | inert | leave (scaling it above 200 hPa would turn GF-6 on by mistake) |
| GF-8 | shallow: the first cloud top (:417-449) | 200 hPa | The shallow cloud top is the inversion found by GF-9 if it lies within 200 hPa above cloud base, otherwise the first level more than 200 hPa above cloud base. This is the shallow scheme's depth limit. | live | scale with s |
| GF-9 | `gfd_get_inversion_layers` (module_cu_gf_deep.F:4063-4159), called from the shallow arm | 100 hPa | Picks the temperature inversion nearest 100 hPa above cloud base (the source's "800 hPa slot"); it is the shallow top candidate of GF-8. | live | scale with s |
| GF-10 | same routine | 300 hPa | Picks the inversion nearest 300 hPa above cloud base (the "550 hPa slot"). No statement reads that slot afterwards, and the deep arm's call of the routine is captured only. | inert | leave |
| GF-11 | deep, after the first `gfd_cup_kbcon` (:660-685) | 150 hPa | `pmin_lev`, the first level more than 150 hPa above cloud base. Written to the capture and read by nothing else. | inert | leave |
| GF-12 | deep `gfd_cup_output_ens_3d`; shallow closure | 100 and 1 kg m-2 s-1 | The ceiling on the deep closure's mean cloud-base mass flux, and the shallow `xmbmax`. | live | mass |

Grell-Freitas also carries thresholds that look like pressure and are not.
They are heights and are left alone by any pressure scaling: `zkbmax` 4000 m
(deep) and 3000 m (shallow) for the highest source level, `z_detr` 1000 m,
`depth_min` 1000 m (the shallowest deep cloud, ierr 6) and `zcutdown` 4000 m.
The 1013.246 in `gfd_satvap` is the steam-point pressure of the Goff-Gratch
formula, a physical constant. `zkbc` is cloud-base height in thousands of
feet (the 3.281e-3 factor). The mass-flux shape (`gfd_get_zu_zd_pdf`) and the
shear term of the downdraft efficiency (`gfd_cup_dd_edt`) use pressure only
as ratios and pressure-weighted means, which carry over to any surface
pressure unchanged.

## New Tiedtke (`ntiedtke.cu`)

| ID | Where | Value | What it decides | Live? | On a thinner world |
|---|---|---|---|---|---|
| NT-1 | `zdnoprc`, three sites: both passes of `ntiedtke_cutypen` and `ntiedtke_cloud_depth` (cumastrn :562-590) | 200 hPa (2.0e4 Pa) | The line between shallow and deep convection. A shallow candidate deeper than 200 hPa is not shallow; a deep candidate shallower than 200 hPa is not deep; after the ascent, a cloud is re-typed deep or shallow by the same depth. | live | scale with s |
| NT-2 | `ntiedtke_cutypen`, deep pass (:1517-1746) | 350 hPa | Deep test parcels start only from levels within 350 hPa of the surface. | live | scale with s |
| NT-3 | same pass | 60 hPa | A deep test parcel starting within 60 hPa of the surface takes a mixed-layer mean; above that it takes the level's own values. | live | scale with s |
| NT-4 | same pass | 50 hPa | The depth of that mixed layer. | live | scale with s |
| NT-5 | `ntiedtke_closure` (:644-668) | 50 hPa | The boundary-layer forcing term subtracted from CAPE in the non-equilibrium closure (`zcape2`) is counted only when the departure level lies within 50 hPa of the surface. | live | scale with s |
| NT-6 | `ntiedtke_cuddrafn` (:2616-2621) | 60 hPa | The downdraft detrains through the lowest 60 hPa, linearly in pressure. | live | scale with s |
| NT-7 | `ntiedtke_cuascn` (:1903-1937) | 400 hPa (4.0e4 Pa), an absolute level | In a column with no convection, the cloud-top record is set to the lowest level above 400 hPa. It is written back to the cloud-top buffer and no later stage of this port reads it. | inert | leave |
| NT-8 | `ntiedtke_closure` (:670-694) | 5000 | The cap on CAPE in the deep closure. This port's CAPE is a buoyancy integrated over pressure in Pa, so the same buoyancy profile in a column with s times the mass gives s times the number, and the cap binds later on a thinner world. | live | scale with s |
| NT-9 | `ntiedtke_cuflxn`, sub-cloud evaporation (:3012-3057) | `sqrt(p / p_surface) / 5.09e-3` | The air-density factor of the rain-evaporation rate is the level's pressure over the column's own surface pressure, so it is relative. On a thinner world it gives Earth's factor at the same fraction of the column and does not see that the air is thinner. Whether WRF means it as an absolute density was not settled by this reading. | live | open: leave if relative is meant, otherwise use a fixed reference pressure |
| NT-10 | `NT_CMFCMAX`; the first-guess mass flux of `ntiedtke_mfub` | 1 and 0.01 kg m-2 s-1 | The mid-level cloud-base mass-flux ceiling, and the deep first guess where moisture convergence is absent under `ntiedtke_tiedtke_closure = true` (off by default). | live | mass |

New Tiedtke's other limits are already relative or are not pressures: the
mass-flux ceilings `zmfmax` are a layer's own mass over the time step, the
boundary-layer wind `upbl` is a pressure-weighted mean, the saturation
helpers divide by the level's own pressure, the mid-level trigger window is
500 m to 10 km in height, and the `1.0e5` in the closure's `zdh` floor is
J/kg per unit specific humidity, not Pa. The deep departure search runs over
the lower half of the model levels by index (`levels >= klev / 2 + 1`), which
ties it to the vertical layout rather than to a pressure.

## This package's own scheme (`arwen_massflux.py`)

| ID | Where | Value | What it decides | Live? | On a thinner world |
|---|---|---|---|---|---|
| OWN-1 | `MassFluxParameters.evaporation_coefficient`, used in the sub-cloud evaporation | 0.5 per 100 hPa at zero relative humidity | The fraction of falling precipitation evaporated per Pa of air it falls through. Per Pa is per unit mass of air crossed. | live | leave |

Every depth this scheme tests is in metres (`deep_minimum_depth_m`,
`shallow_maximum_depth_m`, `source_depth_min_m`, `source_depth_max_m`), and
its saturation helper caps vapour pressure at half the level's own pressure.
It has no absolute-pressure threshold.

## Morrison microphysics (`morrison.cu`)

| ID | Where | Value | What it decides | Live? | On a thinner world |
|---|---|---|---|---|---|
| M-1 | `MRHOSU`, the fall-speed density correction | air density at 850 hPa and 273.15 K | Fall speeds of rain, snow, graupel and ice are multiplied by `(MRHOSU / rho) ** 0.54` (0.35 for ice), and so are their speed caps. A ratio of densities: thinner air gives faster fall, as it should. | live | leave |
| M-2 | `dv`, vapour diffusivity | `8.794e-5 * T ** 1.81 / p` | Diffusion of vapour to and from every hydrometeor, inversely proportional to the absolute pressure, as the physics requires. | live | leave |
| M-3 | `slip`, contact freezing | `7.37 T / (288 * 10 * p) / 100` | The mean-free-path correction of aerosol diffusion, inversely proportional to the absolute pressure, as the physics requires. | live | leave |
| M-4 | saturation vapour pressure | `min(0.99 p, e_s)` | Keeps saturation vapour pressure under the level's own pressure. Relative. | live | leave |
| M-5 | droplet number and the ice-nucleus target | 250 per cm3 (or the land and ocean numbers of `morrison_droplet_number = "land_ocean"`); at most 500 per litre | Both are fixed per unit volume and divided by the air density to give a number per kg, so a thinner world gets more droplets and ice nuclei per kg of air. An Earth aerosol assumption, not a pressure threshold. | live | leave, and state it |

Morrison compares no pressure with a threshold anywhere (the test checks
this). Its sedimentation chooses its own number of sub-steps from the largest
fall-speed Courant number in the column, so faster fall in thin air costs
more sub-steps rather than breaking the step.

## Beyond the kernels: the vertical coordinate

| ID | Where | Value | What it decides | Live? | On a thinner world |
|---|---|---|---|---|---|
| V-1 | `HybridCoordinate.__post_init__` in `src/arwen_global/vertical.py` | surface pressures 500, 800, 1050 and 1200 hPa | Every hybrid coordinate is checked for strictly increasing half levels at these four surface pressures and no other. Nothing below a 500 hPa surface is checked. | live | extend the check to the world's lowest surface pressure before any run |

The coordinate is pure pressure above 100 hPa, so on a thin-air world the
levels above 100 hPa keep their Earth pressures while the hybrid levels
below crowd into whatever column is left.

## Shares of the column

What each live pressure-depth threshold is as a share of a 1013.25 hPa
column, and what the same unchanged number would be on a world whose surface
pressure is 500 hPa. The test recomputes both columns.

| ID | Threshold | Share of 1013.25 hPa | Share of 500 hPa |
|---|---|---|---|
| GF-1 | 75 hPa | 7.4 % | 15.0 % |
| GF-2 | 25 hPa | 2.5 % | 5.0 % |
| GF-5 | 150 hPa | 14.8 % | 30.0 % |
| GF-8 | 200 hPa | 19.7 % | 40.0 % |
| GF-9 | 100 hPa | 9.9 % | 20.0 % |
| NT-1 | 200 hPa | 19.7 % | 40.0 % |
| NT-2 | 350 hPa | 34.5 % | 70.0 % |
| NT-3 | 60 hPa | 5.9 % | 12.0 % |
| NT-4 | 50 hPa | 4.9 % | 10.0 % |
| NT-5 | 50 hPa | 4.9 % | 10.0 % |
| NT-6 | 60 hPa | 5.9 % | 12.0 % |

Read as an example, not a measurement: on a 500 hPa world New Tiedtke would
call a cloud shallow until it spans 40 % of the column, where on Earth it
calls it shallow up to about a fifth, and Grell-Freitas would allow an
inhibition layer twice as deep (by share) before refusing deep convection.
How much that moves rain and heating has not been run.

## What this decides

**Thin air stays "not supported".** The microphysics is not the obstacle:
Morrison has no absolute-pressure threshold and its four pressure uses are
the physics. This package's own scheme has none either. The obstacle is the
two carried convection schemes and the vertical coordinate. Grell-Freitas
holds six live absolute-pressure thresholds (GF-1, GF-2, GF-5, GF-6, GF-8 and
GF-9) and New Tiedtke seven (NT-1 to NT-6 and NT-8), besides the mass-flux
ceilings and the open question of NT-9; the vertical coordinate has never
been checked under a 500 hPa surface.

What would let thin air move to "illustration", in order:

1. One world-wide factor `s = p_world / 1013.25 hPa`, from a surface-pressure
   setting the `[world]` table does not have yet, multiplied into the live
   thresholds marked "scale with s": GF-1, GF-2, GF-5, GF-8 and GF-9 in
   Grell-Freitas; NT-1 to NT-6 and NT-8 in New Tiedtke. For Earth `s` is
   exactly 1, and a float multiply by exactly 1 returns its operand, so
   Earth's results would not move; but the kernel sources and their pins
   would, and the WRF parity suites would have to be run again with `s = 1`.
2. GF-6 replaced by the `kpbl > 3` condition it stands for.
3. NT-9 settled: relative or absolute.
4. The vertical coordinate checked at the world's lowest surface pressure
   (V-1).
5. A test that could fail: one sounding posed on Earth and again on a world
   at `s = 0.5`, with the same temperature against height. With the factor
   in place, both schemes should pick the same cloud base, cloud top and
   cloud type in height to within one level; without it, the shares table
   above says where they will not.

The mass-flux ceilings (GF-12, NT-10) can follow later. The cut-off below
which thin air is refused today is a judgment, not something this reading
measured, and nothing here moves it.
