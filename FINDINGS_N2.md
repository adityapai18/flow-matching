# FINDINGS — SE(3) flow matching, Night 2: does night 1's headline survive a fair comparison?

## Verdict on night 1: **QUALIFIED** (its interpretation is withdrawn)

Night 1 found that M2 (Euclidean-6D) beats M1 (Riemannian) at every D. Those numbers are reproducible and
correct, but only for the Haar source distribution, and that prior handicaps exactly one model. Remove the
handicap and the sign flips:

- **Oracle prior without the cut locus (P3, Haar truncated at d0 ≤ 150°): M1 beats M2 at every D ≥ 30.**
  Mean endpoint error is 0.92/1.67/2.27/2.36° (M1) vs 1.79/2.39/3.55/3.81° (M2) at D = 30/90/150/175.
  That is 3.1/7.4/12.2/12.9 pooled seed-sd, Holm p ≤ 0.018. Pooled over D ∈ {30, 90, 150, 175}:
  1.80° vs 2.89°, 19 sd.
- **Deployable start-centered Gaussian prior (P2, σ=90°): M2 still wins on the overall mean**
  (2.5–4.9° vs 9.3–11.2° at D ≥ 30). But the win comes entirely from the remaining cut-locus tail:
  this prior still puts 32% of sources more than 150° from the target (16% beyond 165°). Stratified by d0, **M1 is better for
  d0 < 120°** (2.25° vs 3.54° at d0 < 90°, 5.9 sd), tied at 120–150°, and far worse beyond 150°.
- **Haar (P1, night 1):** M1 and M2 tie for d0 < 90° (2.65° vs 2.64°). M2 wins in every bin above that.
  Training with cut-locus samples present degrades M1 even on easy samples: its d0 < 90° error is 2.65°
  under Haar vs 1.33° under P3.

So "the manifold buys nothing" (night 1's H2 reading) is withdrawn. On this task, the manifold helps
whenever the source distribution keeps samples away from the SO(3) cut locus, and hurts badly when it
doesn't. Whether a *deployable* prior can do that at large D is open: a start-centered prior sits about D
from the target, so at D = 175° it lands on the cut locus by construction.

**Budget and width (Phase B).**
- **Training budget:** the M1-vs-M2 *sign* held from 20k to 80k steps under P2 (M2 better) and under P3
  (M1 better at D=150). Every gap shrank by half or more, and no model converged. At D=30 under P3, the
  80k gap is no longer distinguishable from zero.
- **Width:** the four-model ranking flipped again at width 2048 (single-seed diagnostic), so **no
  width-fixed ranking is safe.**
- **LR:** the equal-protocol LR pick was unstable for M1 under Haar (8/25 runs spiked at 3e-3, 0/25 at
  1e-3). It was also unstable for M2 at D=10 under P2 and P3 (1 diverged run of 5 each).

**Multimodality (Phase C).** It does not produce a clean flip. On the mode-quality metrics M1 is better at
both D, and at D=150 by large margins:
- share of samples within 10° of a mode: 0.34 vs 0.17;
- median nearest-mode error: 15.5° vs 24.4°;
- share of starts with all 3 modes covered: 0.52 vs 0.39.

On the distributional metric (W1 to the 3-mode target), M2 is better at both D (12.3 vs 14.1°;
34.9 vs 40.7°). No model is good at it: at most 61% of samples land within 10° of a mode.

**For night 3 (real robot data), tonight's evidence says:**
1. The result hinges on the source distribution's distance to the cut locus, so the prior must be reported
   and controlled.
2. Rankings are budget- and width-sensitive, so a single-configuration comparison is not publishable.
3. Report d0-stratified errors.

Whether that clears the gate is your call. In my reading, the comparison is now well-posed, but its answer
depends on the prior.


---

## Phase A — the source distribution

All four models were trained under every prior, with the same sampled rotation encoded into each model's
representation. Grid: 4 models × D ∈ {10, 30, 90, 150, 175} × 5 seeds per prior. **D = 60 and D = 120 were
dropped** to afford three priors. P1 reuses the night-1 runs, which are bit-identical on this grid.

- **P1 Haar:** night 1's prior.
- **P2 Gaussian:** R0·exp(ŵ), w ~ N(0, σ²I), centered on the conditioning start pose. σ ∈ {30, 60, 90} was
  piloted at D=90, and σ* = 90 was chosen by the pre-registered rule (geometric mean over models of val
  error). The rule is model-symmetric, but the pilot was strongly σ-dependent: at σ=30, M1 scored 1.6° vs
  M2's 13.8° (untuned); at σ=90, M1 9.2° vs M2 4.1°. See "What broke" §3.
- **P3 Haar-truncated:** Haar, rejecting sources with d0 > 150° from each waypoint's target. This needs the
  target, so it is an **oracle ablation**, not a deployable prior.

**LR search** was re-run per (model, prior) with the night-1 protocol, including the pre-registered
edge-extension rule. The extension fired for both P2 and P3. Selected LRs:
- M1: 3e-3 (P1) → **1e-3** (P2, P3). Its selected LR changes with the prior.
- M2: 3e-3 under all three.
- M3: 3e-3 (P1, P2) and 1e-3 (P3).
- M4: 3e-4 under all three.

Full tables: `results/n2/tables_n2.md`.

**Mean endpoint geodesic error (deg), NFE=64, test, mean ± sd over 5 seeds**

| Prior | Model | D=10 | D=30 | D=90 | D=150 | D=175 |
|---|---|---|---|---|---|---|
| P1 Haar | M1 | 18.03 ± 2.09 | 15.63 ± 1.65 | 9.49 ± 0.63 | 11.22 ± 0.60 | 10.67 ± 0.85 |
| | M2 | 5.96 ± 1.26 | 1.22 ± 0.04 | 2.38 ± 0.10 | 3.56 ± 0.18 | 3.69 ± 0.19 |
| | M3 | 10.44 ± 2.64 | 10.26 ± 0.27 | 18.08 ± 0.47 | 19.95 ± 0.54 | 19.60 ± 0.61 |
| | M4 | 4.41 ± 0.05 | 8.25 ± 0.14 | 8.90 ± 0.34 | 20.22 ± 0.80 | 21.76 ± 2.01 |
| P2 Gauss σ=90 | M1 | 9.44 ± 0.63 | 10.37 ± 0.75 | 9.34 ± 0.99 | 10.43 ± 0.20 | 11.23 ± 0.47 |
| | M2 | 17.67 ± 28.01* | 4.93 ± 1.22 | 2.46 ± 0.04 | 3.39 ± 0.17 | 3.78 ± 0.20 |
| | M3 | 18.27 ± 3.00 | 10.72 ± 0.56 | 18.28 ± 0.53 | 19.63 ± 1.14 | 20.51 ± 0.27 |
| | M4 | 4.38 ± 0.07 | 8.16 ± 0.11 | 8.88 ± 0.28 | 20.41 ± 1.04 | 20.59 ± 1.66 |
| P3 trunc 150° | M1 | **0.64 ± 0.05** | **0.92 ± 0.02** | **1.67 ± 0.03** | **2.27 ± 0.03** | **2.36 ± 0.03** |
| | M2 | 17.31 ± 28.57* | 1.79 ± 0.40 | 2.39 ± 0.13 | 3.55 ± 0.15 | 3.81 ± 0.16 |
| | M3 | 6.02 ± 0.13 | 10.53 ± 0.45 | 19.47 ± 0.60 | 21.73 ± 0.38 | 21.76 ± 0.58 |
| | M4 | 4.09 ± 0.02 | 7.83 ± 0.09 | 10.17 ± 0.38 | 20.26 ± 0.86 | 19.49 ± 0.90 |

\* One of the five M2 runs **diverged in training** (final loss 0.25 under P2 seed 4, 0.65 under P3 seed 2,
vs ~0.021 for the other runs; endpoint error 68°). The other four seeds give 4.5–6.0° (P2) and 3.6–5.3° (P3).
This is M2 at its protocol-selected LR (3e-3) at D=10, a level outside the tuning set. See "What broke" §2.

**Seed-paired M2 − M1 (mean error, deg, NFE 64): diff [95% CI], pooled seed-sd. Negative = M2 better.**

| Prior | D=10 | D=30 | D=90 | D=150 | D=175 |
|---|---|---|---|---|---|
| P1 Haar | −12.1 [−15.7, −8.4] (7.0) | −14.4 [−16.5, −12.4] (12.4) | −7.1 [−7.9, −6.3] (15.7) | −7.7 [−8.4, −6.9] (17.2) | −7.0 [−8.0, −6.0] (11.3) |
| P2 Gauss 90 | +8.2 [−26.9, +43.4] (0.4)* | −5.4 [−7.0, −3.9] (5.4) | −6.9 [−8.1, −5.6] (9.9) | −7.0 [−7.4, −6.7] (37.5) | −7.5 [−8.2, −6.7] (20.7) |
| P3 trunc 150 | +16.7 [−18.8, +52.2] (0.8)* | **+0.88 [+0.36, +1.39] (3.1)** | **+0.72 [+0.54, +0.90] (7.4)** | **+1.28 [+1.08, +1.49] (12.2)** | **+1.46 [+1.27, +1.65] (12.9)** |

### The headline table: d0-stratified endpoint error

d0 is the geodesic distance from the endpoint-waypoint source sample to the target. Errors are pooled over
D ∈ {30, 90, 150, 175}, at NFE=64, as mean ± sd over 5 seeds. **D=10 is excluded** because of the diverged
M2 runs; the D=10-inclusive version is in `tables_n2.md`. Bins are the same as night 1's `diag_tail.json`.
Source: `results/n2/strata_noD10.json`.

| Prior | d0 bin | share | M1 | M2 | M3 | M4 | M2 − M1 [95% CI] (sd) |
|---|---|---|---|---|---|---|---|
| P1 Haar | 0–90 | 0.19 | 2.65 ± 0.08 | 2.64 ± 0.12 | 16.3 | 14.8 | −0.01 [−0.22, +0.20] (0.1) |
| | 90–120 | 0.21 | 3.34 ± 0.10 | 2.77 ± 0.14 | 16.0 | 14.5 | −0.57 [−0.75, −0.40] (4.8) |
| | 120–150 | 0.28 | 4.92 ± 0.31 | 2.75 ± 0.10 | 17.1 | 14.9 | −2.17 [−2.58, −1.77] (9.5) |
| | 150–165 | 0.16 | 9.51 ± 0.67 | 2.70 ± 0.10 | 17.7 | 14.8 | −6.81 [−7.66, −5.96] (14.3) |
| | 165–175 | 0.11 | 25.4 ± 1.9 | 2.67 ± 0.15 | 18.2 | 14.9 | −22.7 [−25.1, −20.3] (16.5) |
| | 175–180 | 0.06 | 89.0 ± 4.5 | 2.72 ± 0.09 | 17.9 | 14.8 | −86.3 [−91.9, −80.7] (27.4) |
| | **all d0 < 150** | 0.67 | 3.81 ± 0.18 | 2.72 ± 0.11 | 16.5 | 14.8 | −1.09 [−1.37, −0.82] (7.3) |
| P2 Gauss 90 | 0–90 | 0.18 | **2.25 ± 0.06** | 3.54 ± 0.30 | 16.0 | 13.9 | **+1.29 [+0.87, +1.71] (5.9)** |
| | 90–120 | 0.22 | **2.82 ± 0.04** | 3.65 ± 0.31 | 15.9 | 14.6 | **+0.83 [+0.47, +1.19] (3.8)** |
| | 120–150 | 0.28 | 4.19 ± 0.13 | 3.74 ± 0.29 | 17.4 | 14.5 | −0.45 [−0.82, −0.08] (2.0) |
| | 150–165 | 0.16 | 7.65 ± 0.23 | 3.69 ± 0.39 | 18.9 | 15.2 | −3.97 [−4.69, −3.24] (12.3) |
| | 165–175 | 0.11 | 20.8 ± 0.9 | 3.53 ± 0.27 | 18.7 | 14.2 | −17.3 [−18.2, −16.4] (27.4) |
| | 175–180 | 0.05 | 84.9 ± 3.7 | 3.50 ± 0.31 | 19.1 | 15.3 | −81.4 [−86.1, −76.6] (30.9) |
| | **all d0 < 150** | 0.68 | **3.23 ± 0.07** | 3.65 ± 0.29 | 16.5 | 14.3 | **+0.43 [+0.06, +0.79] (2.0)** |
| P3 trunc 150 | 0–90 | 0.28 | **1.33 ± 0.07** | 2.80 ± 0.09 | 16.4 | 14.4 | **+1.47 [+1.39, +1.54] (19.1)** |
| | 90–120 | 0.31 | **1.58 ± 0.03** | 2.94 ± 0.08 | 17.8 | 14.2 | **+1.36 [+1.25, +1.47] (23.0)** |
| | 120–150 | 0.41 | **2.28 ± 0.02** | 2.91 ± 0.10 | 20.1 | 14.7 | **+0.63 [+0.51, +0.75] (9.2)** |
| | **all** | 1.00 | **1.80 ± 0.01** | 2.89 ± 0.08 | 18.4 | 14.4 | **+1.08 [+0.99, +1.18] (19.0)** |

How to read it:
1. **M2's error does not depend on d0, under any prior** (2.6–3.7° in every bin). M1's error rises steeply
   with d0, and catastrophically beyond 165°.
2. **M1's error at a given d0 depends on the training prior.** At d0 < 90°: 2.65° (Haar) → 2.25° (Gauss 90)
   → 1.33° (trunc 150). The cut-locus samples in training don't just produce a tail; they degrade the whole
   learned field. A comparison that only stratifies at test time still understates M1.
3. **M3 and M4 are insensitive to the prior.** They trail M2 at every D ≥ 30 under every prior, and trail
   M1 at every D under P3. Under P1/P2, M4 is slightly better than M1 at D ≤ 90. The prior matters only
   for the model that has a cut locus.

**Answer to Phase A.** Under P3, the one prior with no cut-locus samples at all, M2 does **not** beat M1:
M1 is better at every D ≥ 30, by 0.7–1.5° (3–13 sd). Under the deployable P2, M2 beats M1 overall
(5.4–7.5°, 5–38 sd at D ≥ 30), and the entire margin comes from the 32% of sources beyond 150°. On the
68% of samples with d0 < 150°, M1 is better by 0.43° (2.0 sd), and at d0 < 120° by 0.8–1.3° (3.8–5.9 sd).

## Phase B — budget stability

### B1 — step sweep (P2 Gaussian σ=90, M1 and M2, D ∈ {30, 150}, seeds 0–2, test NFE 64)

B1 was restricted to M1 and M2: the question is their sign, and this halved a ~5 h phase. The 20k point is
the Phase-A main run, which is bit-identical. Each model uses its own P2-selected LR (M1 1e-3, M2 3e-3)
with the cosine schedule stretched to the budget.

| steps | D | M1 | M2 | M2 − M1 [95% CI] (sd) | final loss M1 / M2 |
|---|---|---|---|---|---|
| 20k | 30 | 10.13 ± 0.12 | 4.95 ± 1.66 | −5.18 [−9.21, −1.15] (4.4) | 0.072 / 0.020 |
| 40k | 30 | 5.48 ± 0.62 | 0.75 ± 0.01 | −4.73 [−6.25, −3.21] (10.8) | 0.058 / 0.012 |
| 80k | 30 | 3.73 ± 0.55 | 0.59 ± 0.00 | −3.13 [−4.50, −1.77] (8.1) | 0.045 / 0.010 |
| 20k | 150 | 10.45 ± 0.26 | 3.47 ± 0.18 | −6.98 [−7.92, −6.04] (30.8) | 0.092 / 0.027 |
| 40k | 150 | 8.08 ± 0.28 | 2.41 ± 0.07 | −5.66 [−6.40, −4.93] (28.0) | 0.074 / 0.020 |
| 80k | 150 | 5.68 ± 0.30 | 2.01 ± 0.07 | −3.67 [−4.39, −2.95] (16.6) | 0.055 / 0.017 |

d0-stratified (`results/n2/b1_strata.json`):

| steps | D | d0 < 120: M1 / M2 | 120–150 | 150–165 | ≥ 165 |
|---|---|---|---|---|---|
| 20k | 30 | **2.17** / 4.84 | **4.46** / 5.07 | 8.95 / 5.23 | 47.2 / 4.8 |
| 80k | 30 | 0.84 / **0.59** | 1.10 / **0.61** | 1.61 / **0.58** | 20.1 / 0.6 |
| 20k | 150 | **2.87** / 3.41 | 3.94 / 3.56 | 7.03 / 3.44 | 42.6 / 3.5 |
| 80k | 150 | 1.95 / 2.02 | 2.16 / 2.03 | 2.90 / 1.93 | 23.3 / 2.1 |

- **The sign of the overall M1-vs-M2 comparison under P2 is stable from 20k to 80k (M2 better), but the
  gap halves** (−5.2 → −3.1° at D=30, −7.0 → −3.7° at D=150).
- **Training has not converged for either model.** Over the last quarter of an 80k run, loss still falls
  22% (M1) and 45% (M2). This metric includes the cosine LR decay, so it overstates how far from a plateau
  the models are, but neither curve is flat. Loss curves: `results/figures/n2_b1_loss.png`.
- **Phase A's stratified "M1 is better for d0 < 120° under P2" result is NOT budget-stable.** It was mostly
  M2 being undertrained at 20k. M2 at D=30 goes from 4.95° to 0.59°, with high 20k seed variance. By 80k,
  M2 is better than M1 in every d0 bin at D=30, and the two tie for d0 < 150° at D=150.
- M1's cut-locus tail shrinks with training (≥ 165°: 47 → 20° at D=30) but does not go away.
- **Untested at the time of B1:** whether the P3 result (M1 better by 19 sd at 20k) survives more steps.
  An 80k extension under P3 was queued after Phase C (see "B1 extension" below).

### B1 extension — does the P3 (cut-locus-free) M1 win survive 4× the steps?

This was added during the night, after B1 showed the P2-stratified M1 advantage was budget-dependent.
Same protocol as B1: P3 oracle prior, M1 and M2 with their P3-selected LRs, D ∈ {30, 150}, **3 seeds**
(as in B1). Source: `results/n2/b1x_trunc150.json`.

| D | steps | M1 | M2 | M2 − M1 [95% CI] (sd) | d0 < 90 / 90–120 / 120–150: M1 vs M2 |
|---|---|---|---|---|---|
| 30 | 20k | 0.91 ± 0.02 | 1.82 ± 0.53 | +0.91 [−0.44, +2.25] (2.4) | 0.63/0.78/1.19 vs 1.77/1.86/1.82 |
| 30 | 80k | 0.44 ± 0.02 | 0.52 ± 0.04 | +0.08 [−0.04, +0.20] (2.6) | 0.41/0.44/0.47 vs 0.50/0.53/0.54 |
| 150 | 20k | 2.28 ± 0.03 | 3.57 ± 0.16 | +1.29 [+0.83, +1.75] (11.5) | 1.73/2.02/2.86 vs 3.41/3.66/3.61 |
| 150 | 80k | **1.50 ± 0.03** | 1.98 ± 0.07 | **+0.48 [+0.37, +0.60] (8.5)** | 1.29/1.39/1.72 vs 1.93/2.02/1.99 |

- Under P3, **M1 stays ahead in every cell and every d0 bin at 80k, but the gap shrinks** as both models
  train: 0.91 → 0.08° at D=30, 1.29 → 0.48° at D=150.
- At D=30 the 80k gap is no longer distinguishable from zero (CI includes 0, n=3).
- At D=150 it is still clear (8.5 sd), and it is 24% of M2's error.
- This is the same pattern as under P2: more training narrows every M1-vs-M2 gap. **The robust,
  budget-stable statement is about the sign at large D under a cut-locus-free prior, not about magnitudes.**



### B2 — width sweep (night-1 diagnostic protocol: Haar, D=90, tuning seed 100, LR 1e-3, 10k steps, val NFE 32)

| width (params) | M1 | M2 | M3 | M4 | ranking (best first) |
|---|---|---|---|---|---|
| 512 (1.0–1.2M) | 23.7 | 45.1 | 60.6 | 33.4 | M1 < M4 < M2 < M3 |
| 1024 (3.6–3.9M) | 17.8 | 13.4 | 20.4 | 7.6 | M4 < M2 < M1 < M3 |
| **2048 (~14M)** | 15.8 | **2.9** | 12.8 | 4.6 | **M2 < M4 < M3 < M1** |

(M1 at width 256 was the superseded head and sat at chance; it is excluded.)

**The ranking flips again at 2048.** M2 overtakes M4, and M1 falls from third to last. By the brief's
criterion, **no width-fixed conclusion is safe, and the paper must say so.**
- These are single-seed diagnostic runs, not headline numbers.
- Under Haar, M2 gains 4.6× from 1024 → 2048 (13.4 → 2.9°) while M1 gains 1.1× (17.8 → 15.8°). That fits
  M1 being limited by the cut-locus tail (p95 61° at 2048) rather than by capacity. The M1-vs-M2 sign under
  Haar is the same at 1024 and 2048; the other pairwise orders are not.
- Width 2048 was not run under P2/P3. Whether M1's cut-locus-free advantage survives more capacity is
  untested.

### B3 — M1 learning-rate stability (Haar, 20k, test, NFE 64, 5 seeds)

| | D=10 | D=30 | D=90 | D=150 | D=175 | runs with a loss spike |
|---|---|---|---|---|---|---|
| M1 @ 3e-3 (equal-protocol pick) | 18.03 ± 2.09 | 15.63 ± 1.65 | 9.49 ± 0.63 | 11.22 ± 0.60 | 10.67 ± 0.85 | **8/25** |
| M1 @ 1e-3 | 13.48 ± 0.42 | 13.77 ± 0.61 | 11.01 ± 0.71 | 11.36 ± 0.57 | 10.58 ± 0.75 | **0/25** |
| M2 @ 3e-3 | 5.96 ± 1.26 | 1.22 ± 0.04 | 2.38 ± 0.10 | 3.56 ± 0.18 | 3.69 ± 0.19 | 0/25 |
| M2 − M1@1e-3 | −7.5 (8.0 sd) | −12.6 (29.2 sd) | −8.6 (16.9 sd) | −7.8 (18.5 sd) | −6.9 (12.6 sd) | |

**The equal-protocol LR selection produced an unstable configuration for one model.** Under Haar, the
protocol picked 3e-3 for M1 by a 0.6° val margin over 1e-3. At 3e-3, 8 of 25 M1 runs had loss spikes
(all at D ≤ 30 on this grid); at 1e-3 there were none. The stable LR is better at small D (−4.6° at D=10,
−1.9° at D=30) and 1.5° worse at D=90. By the "equal in effect" standard, night-1's M1 numbers at D ≤ 30
were therefore depressed by an unstable training configuration.

It does not change the Haar conclusion: M2 beats the stable M1 at every D, by 6.9–12.6° (8–29 sd).
Under P2 and P3, the protocol itself selected 1e-3 for M1, and those runs had no spikes (0/25 each).


## Phase C — multimodal target

**Data.** `MM_D{30,150}.npz` uses the same start distribution as the unimodal data. For each start there
are K=3 valid targets: rotations by D about the sampled axis a and about two axes at ±60° from a (mode
separation is in `results/synthetic/MM_D*_meta.json`). Each trajectory takes one mode uniformly at random
(counts 653/678/669 at D=150). Models are conditioned on R0 only, so they must represent all three modes.

**Protocol.**
- Prior P2 Gaussian σ=90 (see deviation 1).
- LR re-searched on the multimodal data with the same protocol, criterion = val W1 at NFE 32. The edge
  extension fired. Final LRs (all interior): M1 1e-3, M2 3e-3, M3 3e-3, M4 3e-4.
- 20k steps, 5 seeds, 200 test starts × 30 samples each.
- Metrics:
  - **W1**: exact optimal transport with geodesic cost between the 30 endpoint samples and the uniform
    3-mode distribution, per start;
  - **nearest-mode error**;
  - **mode coverage**: a sample "hits" a mode if it lands within 10° of it.

**Results (NFE 64, mean ± sd over 5 seeds; seed-paired M2 − M1, Holm over D)**

| metric | D | M1 | M2 | M3 | M4 | M2 − M1 [95% CI] (sd) |
|---|---|---|---|---|---|---|
| W1 to 3-mode target (deg) ↓ | 30 | 14.14 ± 0.41 | **12.31 ± 0.75** | 19.47 | 18.97 | −1.83 [−3.23, −0.44] (3.1) |
| | 150 | 40.74 ± 1.22 | **34.90 ± 0.25** | 44.87 | 44.82 | −5.84 [−7.10, −4.58] (6.7) |
| nearest-mode error, mean ↓ | 30 | 12.31 ± 0.45 | **10.68 ± 0.75** | 18.24 | 18.02 | −1.63 [−3.05, −0.21] (2.6) |
| | 150 | **23.07 ± 0.29** | 27.25 ± 0.62 | 39.50 | 39.66 | +4.18 [+3.50, +4.86] (8.6) |
| nearest-mode error, median ↓ | 30 | **7.80 ± 0.31** | 10.29 ± 0.76 | 16.82 | 17.05 | +2.49 [+1.33, +3.65] (4.3) |
| | 150 | **15.49 ± 0.57** | 24.37 ± 0.96 | 38.45 | 36.28 | +8.88 [+7.74, +10.03] (11.3) |
| samples within 10° of a mode ↑ | 30 | **0.611 ± 0.015** | 0.479 ± 0.058 | 0.135 | 0.126 | −0.13 (3.1) |
| | 150 | **0.341 ± 0.012** | 0.167 ± 0.008 | 0.024 | 0.044 | −0.17 (17.4) |
| starts with all 3 modes hit ↑ | 30 | **0.948 ± 0.010** | 0.859 ± 0.022 | 0.294 | 0.245 | −0.09 (5.3) |
| | 150 | **0.519 ± 0.027** | 0.385 ± 0.040 | 0.003 | 0.033 | −0.13 (3.9) |
| TV of mode shares from uniform ↓ | 30 | **0.182 ± 0.006** | 0.225 ± 0.005 | 0.378 | 0.404 | +0.04 (7.5) |
| | 150 | 0.354 ± 0.015 | 0.338 ± 0.008 | 0.594 | 0.537 | −0.02 (1.3), n.s. |

**d0-stratified nearest-mode error** (d0 = distance from the source to the *nearest* mode; recomputed
exactly from the deterministic evaluation sources by `src/strata_mm.py` → `results/n2/mm_strata.json`):

| D | d0 stratum (share) | M1 mean / median / on-mode | M2 mean / median / on-mode |
|---|---|---|---|
| 30 | d0 < 150 (0.84) | **8.39 / 6.72 / 0.69** | 10.70 / 10.30 / 0.48 |
| 30 | d0 ≥ 150 (0.16) | 33.25 / 18.77 / 0.20 | **10.61 / 10.27 / 0.48** |
| 150 | d0 < 150 (0.93) | **21.65 / 14.75 / 0.36** | 27.23 / 24.35 / 0.17 |
| 150 | d0 ≥ 150 (0.07) | 42.19 / 29.04 / 0.13 | **27.56 / 24.64 / 0.17** |

**Answer to Phase C.** Multimodality changes the ranking on some metrics but not others, so this is not
the clean flip that would have been the project's most important result.
- **On mode quality, M1 beats M2 at both D**, and at D=150 by a lot: twice the on-mode fraction
  (17.4 sd), a median error of 15.5 vs 24.4°, and more starts with every mode covered. Away from the cut
  locus (d0 < 150), M1 also wins on mean error at both D.
- **On W1, the distributional distance, M2 wins at both D** (3.1 and 6.7 sd). At D=30 this is consistent
  with M1's cut-locus tail (16% of sources). At D=150 only 7% of sources are beyond 150°, so the tail
  cannot account for a 5.8° W1 gap. **I have not identified why W1 and the mode-quality metrics disagree
  at D=150.** A plausible but untested explanation: W1 forces every sample onto a mode, so the cost of M1's
  off-mode samples matters more than its on-mode precision.
- **M3 and M4 essentially fail** to represent the modes at D=150: ≤ 4% of samples on a mode, and ≤ 3% of
  starts with all modes covered. That is consistent with their unimodal deficits.
- **No model represents the distribution well.** At D=150 the best on-mode fraction is 34%. This is a
  20k-step, width-1024 result, and the B1/B2 sensitivities apply.
- At NFE 8, M2 is far ahead on W1 (12.2 vs 20.8° at D=30, 36.1 vs 45.7° at D=150; 15 sd): M1 again needs more integration steps
  (`tables_n2.md`).


## Phase E — M3 mechanism (optional; partial)

`src/diag_m3.py` → `results/n2/diag_m3.json`, run on the night-1 Haar M3 models (NFE 64, all 5 seeds).

- **(a) Straight chords between quaternions in opposite half-spaces.** Canonicalization (w ≥ 0) does not
  make q0·q1 ≥ 0. For 28% of source/target pairs the dot product is negative, and the R⁴ chord passes near
  the origin, sweeping the long way round in rotation space. **This explains part of the deficit at small
  D**: at D=10/30, error is 15.4/16.3° when q0·q1 < 0 vs 10.0/9.4° when ≥ 0. **It explains none of it at
  D ≥ 90** (17.8 vs 18.8°, 20.1 vs 19.8°, 19.8 vs 19.4°).
- **(b) Normalization at the end.** Ruled out: correlation between error and | |q| − 1 | is 0.06.
- **Error vs angle α between q0 and q1 in R⁴** rises only mildly (13–15° for α < 90° vs 19–22° beyond),
  correlation 0.19.

Status: **not explained.** No candidate accounts for the flat ~18–20° error at D ≥ 90, or for the missing
step at D = 150/175. An untested hypothesis is that the network is capacity-limited on M3's target: M3
improves 37% from width 1024 to 2048 (20.4 → 12.8°, B2) without reaching M2. I stopped at the timebox.


## Deviations from the night-2 brief and from the pre-registration

1. **B1 and C ran under P2 (Gaussian σ=90), not the pre-registered rule's pick (Haar).** The rule
   (geometric mean over models of each model's selected-LR val score) chose Haar by 0.5% (8.107 vs 8.151,
   one tuning seed), effectively a tie. Haar is the prior this night exists to remove: the LR searches
   already showed it raises M1's tuning error from 1.45° (P3) to 12.9° while M2 moves 2.6 → 2.1°. I broke
   the tie toward the legal prior with no known single-model pathology. Recorded in
   `results/n2/best_prior.json` (`prereg_choice: haar, used: gauss90`). P2 does not remove the cut locus
   either (32% of sources beyond 150°), so B1 and C are still not fully "fair" to M1, and the stratified
   tables are the correction.
2. **B1 was restricted to M1 and M2.** The question is their sign, and this halved a ~5 h phase.
3. **B1 extension** (P3, 80k steps, M1/M2) was added after B1's result. It is not in the brief.
4. **σ*=90 came from the pre-registered rule, but σ was highly influential in the pilot**: at σ=30, M1 was
   at 1.6° vs M2 at 13.8° (untuned, LR 1e-3). The σ=30 arm (full LR search plus main grid) was queued at
   the lowest priority and **did not run**: 2 of 36 LR-search runs completed. I stopped it at 08:54 so the
   machine is free; `cd src && python night2.py run` resumes it, ~5 h.
5. **The Phase-A grid drops D=60 and D=120**, as the brief allowed.
6. **Session interruption.** Night 2 began in an earlier session (Sep 29) that ended mid-Phase A. This
   session (Oct 2) reviewed that code and its pre-registration, resumed from disk, and completed the rest.
   Everything was resumed from checkpoints; no run was restarted from scratch with different settings.

## What broke

1. **The pre-registered "best prior" rule picked the unfair prior** (deviation 1). A model-symmetric rule
   selected, by a statistical tie, the prior that handicaps one model. That is exactly the "equal in form,
   unequal in effect" failure this brief warned about.
2. **The equal-protocol LR produced unstable configurations for two models:**
   - **M1 under Haar at 3e-3:** 8/25 runs had loss spikes; 1e-3 had 0/25 (B3).
   - **M2 at 3e-3 at D=10 under P2 and P3:** one diverged run each (final loss 0.25 and 0.65 vs 0.021;
     endpoint error ~68°). D=10 is outside the tuning set.
   Both are reported, not tuned away. D=10 is excluded from the headline stratified table for this reason.
   The bottom row of `results/figures/n2_phaseA.png` *includes* D=10, which is why M2's bars there are wide.
3. **Phase A's stratified M1 advantage under P2 was not budget-stable** (B1). It largely reflected
   undertrained M2 at 20k. Every M1-vs-M2 gap shrinks with training, and nothing converged by 80k.
4. **The width ranking flipped again at 2048** (B2), on a single seed.
5. **Phase C's W1 and mode-quality metrics disagree at D=150, unexplained.**
6. **M3's deficit is only partly explained** (Phase E): chord-through-origin pairs at small D account for
   part of it; at D ≥ 90 it is unexplained.
7. **Scheduling.** The night-2 scheduler was killed once by the tool's background time limit (restarted
   detached with `nohup`, resuming from checkpoints). Once, lower-priority long jobs filled free slots
   ahead of Phase C's LR extension; that cost ~40 min, not correctness.
8. **The D=10 anomaly from night 1 persists for M2 under every prior**: the healthy D=10 runs are at
   3.6–6.9° vs ~1–2° at D=30. Still unexplained.

## Files
- `results/n2/tables_n2.md`: every table, generated by `src/analyze_n2.py`.
- `results/n2/strata_noD10.json`, `b1_strata.json`, `b1x_trunc150.json`, `mm_strata.json`: headline
  stratified numbers (`src/strata_n2.py`, `src/strata_mm.py`).
- `results/n2/preregistration.json`, `sigma_select.json`, `hparams_*.json`, `best_prior.json`: selection records.
- `results/figures/n2_phaseA.png`, `n2_b1_loss.png`.
- Night-1 `FINDINGS_SE3_N1.md`: headline revised in place; the original is kept in a "SUPERSEDED" block.

