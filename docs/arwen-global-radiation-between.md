# Arwen Global radiation between its calls: the shortwave follows the sun (experimental)

Module `src/arwen_global/physics/radiation_between.py`; option
`radiation_between` of the native suite (`native_adapter_options`); tests
`tests/test_radiation_between.py`.

RRTMGP runs once per `radiation_interval_s` (3000 s, ten 300 s steps, in the
shipped semi-Lagrangian configs) and its planes are held until the next
call. The call's sun is the interval's midpoint sun, so a held plane is
wrong by up to half an interval of sun motion: a column the call saw at
sunrise keeps its midpoint shortwave for the whole interval, and a column
past sunset keeps heating. `radiation_between` chooses what happens between
two calls.

| Value | What happens between calls |
|---|---|
| `"hold"` (the default) | the call's planes are applied unchanged until the next call, as every run before the option did |
| `"surface"` | the surface downward shortwave follows the sun every physics call, by WRF's `swint_opt = 1` fit; the heating rates stay the call's |
| `"column"` | `"surface"`, plus the shortwave heating rate and the two top-of-atmosphere shortwave planes scaled by the ratio of the current cosine of the solar zenith angle to the call's |

```toml
[physics]
mode = "arwen-native"
native_adapter_options = { ..., radiation_interval_s = 3000.0, radiation_between = "surface" }
```

Under `"hold"` no array is added, no arithmetic changes and the option is
left out of the adapter identity, so every config hash and checkpoint
written without it keeps its bytes. `"surface"` and `"column"` join the
identity, so a run under either has its own config hash.

## What `"surface"` does

Each full radiation call fits the column's surface downward shortwave as
`SWDOWN = Gx * coszen**gg`, the exponent from the log ratio between this
call and the previous one, clamped to [-0.5, 2.5]; a clamped or zero
exponent falls back to the ratio `coszen_now / coszen_ref`. Every physics
call then evaluates the fit at the current sun. A night column (either
cosine at or under 1e-4) is zero, and the absorbed part is
`GSW = SWDOWN * (1 - albedo)` on the step's live albedo. Noah, the sea-ice
and the lake surfaces read the updated planes. This is the fit of WRF
v4.6.1's `update_swinterp_parameters` / `interp_sw_radiation`
(`module_radiation_driver.F`), on the global flux only: the global solver
hands back no direct-beam plane, so WRF's `SWDDIR` half of the fit is not
carried.

The current sun is read at the physics call's model time, as WRF reads it
at the step's `xtime`, through the same cosine-zenith statements the
radiation call uses. Every reference the update reads is a persistent array
of the physics namespace, so a checkpoint carries it and a restart in the
middle of an interval continues bit for bit.

Clouds do not move between calls in either mode. The shortwave sees the
cloud through McICA samples, so a partly cloudy column's flux is not a
smooth function of its cloud fields that a per-step update could follow.

## What is tested

`tests/test_radiation_between.py`, on the CPU:

- `"hold"` is the default, stays out of the identity, adds no array and
  keeps the call's planes; `"surface"` and `"column"` join the identity; an
  unknown value is refused;
- the fit: the first call's guess is the cosine ratio, a second call
  recovers a planted power law, a clamped exponent falls back to the ratio,
  night columns are zero;
- the sun: local noon on the equator at the equinox is overhead, and the
  summer pole sees the sine of the tilt;
- through the suite: the surface follows the sun between calls under both
  modes, the radiation integrals advance by what was applied, `"column"`
  scales the heating and the top-of-atmosphere planes, and a suite
  restarted from the namespace continues bit for bit.

## Measured on a card

Measured 2026-11-12 on one RTX 4090: four 48-hour runs at T127 L40 of the
native suite on the semi-Lagrangian core at 300 s from the GDAS 2026-10-15
00Z analysis, output every three hours. Three call radiation every 3000 s
under `"hold"`, `"surface"` and `"column"`; the reference calls it every
step (`radiation_interval_s = 300`), so it carries no between-call error.
Each run's surface downward shortwave and land 2 m temperature (land
fraction at least one half, equatorward of 60 degrees) are compared with
the reference's at the same output, as an area-weighted RMS, and the means
below are over the 16 outputs of each run:

| | `"hold"` | `"surface"` | `"column"` |
|---|---:|---:|---:|
| surface downward shortwave against the reference, RMS (W/m2) | 72.3 | 69.5 | 69.8 |
| land 2 m temperature against the reference, RMS (K), first day | 0.306 | 0.272 | 0.275 |
| the same, second day | 0.453 | 0.434 | 0.432 |
| land 2 m temperature against the reference, mean (K) | +0.036 | +0.009 | +0.010 |
| top-of-atmosphere downward shortwave, 48 h run mean (W/m2; reference 343.74) | 343.74 | 343.74 | 343.39 |

Both update modes take about a tenth off the land 2 m temperature RMS
against the every-step reference on the first day and about a twentieth on
the second, and nearly remove its mean warm offset. The shortwave RMS
moves by 3 to 4 percent only: most of it is not what following the sun
removes (the runs' clouds differ, and clouds are held between calls; that
split was not measured). `"column"` reads 0.35 W/m2 less incident shortwave at
the top than the sun gives over two days, because a column dark at the
call stays dark until the next one. All four runs held 3.3 GiB of card at
the allocator's peak. This is one case at one truncation; it says what the
option did here, not what it does to a forecast's skill.

## Limits

- Clouds are held between calls.
- Only the global surface flux is fitted; there is no direct-beam plane.
- The longwave is held, as under `"hold"`.
- One case was measured; it says what the option did to that case, not
  what it does to a forecast's skill.
