# Stochastic physics for ensemble members (SPPT, SKEB)

`[stochastic]` gives each ensemble member its own model perturbations: SPPT (stochastically perturbed physics
tendencies) and SKEB (stochastic kinetic energy backscatter). Both are off by default. Off is the shipped run: the
config hash, the step arithmetic and the checkpoints are unchanged.

```toml
[stochastic]
sppt = true
skeb = true
member = 7      # or `gpuwm-global run cfg.toml --member 7`
seed = 53       # or `--ensemble-seed 20261116`
```

## Where it comes from

`arwen_global/stochastic.py` is a port of this project's own stochastic physics for the gpuwm-hex model line (project
code under Apache-2.0). It was written from the papers listed in NOTICE (Buizza et al. 1999, Palmer et al. 2009, Shutts 2005, Berner et al. 2009, Weaver &
Courtier 2001, Bolton 1980). No NOAA `stochastic_physics` code was read or used.

What was ported, and what changed for a spectral model:

| Piece | Hex | Here |
| --- | --- | --- |
| Pattern law | AR(1) real harmonic coefficients, variance spectrum `exp(-l(l+1)a)`, `a = L^2/(2R^2)`, unit point variance, l = 0 excluded, automatic degree (1e-4 omitted variance) | Same law, held as the model's own complex triangular coefficients. The degree is capped at the model truncation. |
| Noise | Philox4x32-10 (CUDA), Box-Muller, counter = (mode, draw), key = sha256 of member and seed | The same round function in exact numpy uint64 (it passes the Random123 known-answer vectors), keyed per process by sha256(schema, member, seed, process name). |
| Synthesis | Harmonic sum at every mesh cell and vertex | One `transform.inverse` of the summed SPPT pattern per physics call. SKEB stays spectral. |
| SPPT | Net physics tendency times `1 + mu*r`. Hard clip, or `clip*tanh(r/clip)`. Height-ramp taper. Moisture guard. | The same, applied to the physics call's change of u, v, theta and qv. The taper uses height above ground from the call's geopotential. |
| SKEB | Rotational wind of a streamfunction pattern, normalised by its expected RMS, amplitude `sqrt(2/h tanh(h/2tau) * fraction * D_k)`. `D_k` is the positive KE sink of the unperturbed physics wind tendency. Plus a thermal channel. | The same budget and normalisation. The forcing is the spectral Laplacian of the pattern added to the analysed vorticity (and the thermal pattern to theta). The level weight is the global mean of the SPPT taper. |
| State | Coefficients, clock and draw counter in the checkpoint | Carried in the physics namespace as `stochastic_coefficients` (float64) and `stochastic_clock`. Checkpoints carry them, and a restart continues the same sequence. |

Not ported: the mesh geometry, the limited-area masks and the LTS macro clock (they have no meaning in a spectral
model). SHUM is not implemented.

## Defaults

| Setting | Default | Source |
| --- | --- | --- |
| SPPT scales (amplitude, L, tau) | 0.52 / 500 km / 6 h, 0.18 / 1000 km / 3 d, 0.06 / 2000 km / 30 d | ECMWF three-scale SPPT |
| Bound | hard clip at 1 (`logit = true` gives the smooth bound) | Palmer et al. 2009 |
| Taper | 0 at the lowest level, ramps in over the lowest 500 m, ramps out between 18 and 25 km | hex |
| Moisture guard | on | Palmer et al. 2009 |
| SKEB pattern | 500 km, 6 h | hex |
| SKEB fractions | wind 0.1, thermal 0.01 | hex |

## Clock

A physics call at model time t advances every process to t. A second call at the same time (the Strang second half
of step n and the first half of step n+1 share t + dt, as do physics substeps) reuses the pattern without a new draw.
So a run draws once per step. The SKEB budget uses the step as the pattern's hold period.

## Measured pattern statistics (T383, 2026-11-16)

`tools/arwen_global_stochastic_stats.py --truncation 383 --draws 120 --members 8`, numpy on CPU. Correlation length is
the meridional distance at which the correlation falls to exp(-1/2), over 120 independent draws. Time is from the
pooled coefficient correlation after one decorrelation time of 300 s steps (8 members).

| Process | Degree | L asked -> measured | Point variance | tau asked -> measured | Lag-1 at 300 s (expected) |
| --- | --- | --- | --- | --- | --- |
| SPPT 1 | 55 | 500 -> 497.8 km | 1.002 | 6 -> 6.01 h | 0.98612 (0.98621) |
| SPPT 2 | 27 | 1000 -> 996.3 km | 0.996 | 72 -> 69.9 h | 0.99903 (0.99884) |
| SPPT 3 | 14 | 2000 -> 1908.5 km | 0.972 | 720 -> 762 h | 0.99986 (0.99988) |
| SKEB psi | 55 | 500 -> 498.9 km | 1.009 | 6 -> 6.14 h | 0.98593 (0.98621) |
| SKEB thermal | 55 | 500 -> 497.4 km | 0.996 | 6 -> 5.72 h | 0.98630 (0.98621) |

The summed SPPT pattern has a point standard deviation of 0.553 (0.554 expected from 0.52/0.18/0.06). After the clip at
1 it is 0.519, and 7.1 % of points are clipped. The 2000 km and 30-day rows carry the most sampling error: degree 14
has few modes, and only 8 members reach 30 days. Also, at L/R = 0.31 the heat-kernel spectrum is no longer exactly
Gaussian in distance.

## Tests

`tests/test_arwen_global_stochastic.py` (12 tests, CPU):

- Philox matches the Random123 known-answer vectors.
- The noise is unit complex Gaussian on the triangle.
- Correlation length within 5 % at T170, decorrelation time within 0.01 in lag-1 correlation, point variance within 3 %,
  and three-scale amplitude within 6 %.
- The same clock draws nothing, and a clock that moves backwards is refused.
- Patterns are a function of member, seed and clock only.
- Off: the config hash is unchanged and every checkpoint array is bit-identical to a run without the table.
- On: a run reproduces bit for bit, two members differ, and a restart from step 2 equals the straight run at step 6 in
  every array.
- The table refuses unknown keys and bad values.

## Cost and restart at T383 (RTX PRO 6000, 2026-11-16)

Setup: the shipped `arwen_global_gdas_t383_native_24h.toml` (T383 L40, native suite, sl_si at 300 s) from GDAS
2026-11-15 06Z, 24 steps, off and then on (`sppt = true`, `skeb = true`, member 1) on the same card.
`--profile-steps 16 --profile-warmup 4`.

| | off | on |
| --- | --- | --- |
| Profiled steps without radiation, median | 792 ms | 808 ms (+16 ms, +2.0 %) |
| `stochastic` profile sections (fused band kernel + SKEB spectral forcing) | | 1.75 ms per physics call, 3.5 ms per step |
| Profiled 16-step mean (3 radiation steps each, noisy) | 1122 ms | 1086 ms |
| 24-step mean, receipt `step_s`, warm-up steps included | 1.077 s | 1.236 s |
| Card peak (nvidia-smi) | 14,353 MiB | 14,391 MiB |

The steady cost is about 16 ms a step. Of that, 3.5 ms is the stochastic sections themselves. The rest is the pattern
advance (host Philox for five processes and one host read of the pattern state per call), one T383 synthesis per
call and the four input copies per band. The 24-step receipt mean is 0.16 s/step higher with stochastic on. That is
3.8 s over the 24 steps, and the profiled steady steps do not show it. I attribute it to first-call costs (compiling
the fused kernel, the first allocations), but did not prove it per step.

Restart: the on arm resumed from its step-12 checkpoint matches the straight run at step 24 in all 138 arrays, bit
for bit (cupy float32). After 2 h, the on arm differs from the off arm by up to 0.06 K in theta and 5e-5 kg/kg in qv.

On the cupy backend `perturb_band` is one elementwise kernel (`_band_kernel`). The array expressions are its
specification and the numpy path. The unfused path cost 23.8 ms per call on a shared RTX 5070 Ti
(`tools/arwen_global_stochastic_cost.py`), which is why the band work was fused.
