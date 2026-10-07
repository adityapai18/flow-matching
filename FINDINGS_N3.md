# FINDINGS — SE(3) flow matching, Night 3: asymptotic or convergence-rate? Is there a deployable prior?

_Status: complete. Phases A and B finished in full; Phase C (optional) ran as a single-seed diagnostic.
Runs started 00:13 and finished 16:51 on 2026-10-07. The scheduler exited ("nothing left to run"), and no
worker processes remain (checked with `pgrep`)._

## Answers

**1. Gate (Q1): CONVERGENCE-RATE (b), on the only interpretable series. The pre-registered series could
not decide.**
- **Pre-registered protocol series (P3, M2 at its 20k-tuned LR 3e-3): (c) undecidable, for a structural
  reason.** M2 diverged in 3/3 seeds at 320k. The "with diverged" rule output is an artifact.
- **Matched-LR sensitivity arm (both models at 1e-3; specified before it ran): (b).**
  - Per-doubling gap ratios: 0.45 / 0.46 / 0.26 / −0.28. The last has CI [−0.60, −0.09].
  - The gap goes from +4.75° (20k) to −0.07° [−0.16, +0.02] (320k), and the error curves cross at
    ~275k steps [247k, 302k].
  - The brief's linear-in-log fit: −1.15°/doubling [−1.42, −0.88], extrapolated −2.6° at 1M. That fit is
    misspecified (a decaying gap cannot be a line); the ratios and the crossing are the measurements.
  - At 320k, M1's advantage is gone in every d0 band.
  - **320k is not converged either** (loss still falling ~37% in the last quarter), so any asymptotic
    claim is the extrapolation, not the measurement.
- **What the result is:** M1 is ~4× more sample-efficient at matched LR. It is not more accurate in the
  limit.
- **This retracts the "Riemannian advantage" framing** under the cut-locus-free prior. The project
  should be reframed around sample efficiency.
- **Two further findings:**
  - Under P2, M1's cut-locus tail (sources ≥ 165° from the target) shrinks from 42.6° to 5.3° by 320k.
    Most of the "structural" failure is also a rate effect, though not all of it.
  - The LR protocol tuned at 20k was unequal in effect at long budgets. It produced loss spikes in most
    long runs of both models, and outright divergence for M2.

**2. Prediction and deployable prior (Q2).**
- The d0 mechanism table confirmed the prediction at D=30. It partly contradicted "less at D=90": tight
  σ removes the cut locus there too. **It killed "not at all at D=175": tight σ makes cut-locus exposure
  worse** (≥ 165° share 18% → 67% from σ=90 to σ=15). The trained results followed the mechanism table,
  not the prediction (M1 at D=175: 14.2 → 11.1 → 26.0 → 36.3° from σ=90 to σ=15).
- **A deployable σ gives M1 a win at D=30 and D=90 (σ ≤ 60), and at no σ at D=175** (best: a tie at
  σ=60). σ **shifts** the crossover D, non-monotonically (below 30° at σ=90, 90–175° at σ ≤ 30, ~175° at
  σ=60); it does not merely rescale the gap.
- With each model at its own best deployable prior: M1 wins at D=30 (0.63 vs 1.22°) and D=90 (1.48 vs
  2.38°); M2 wins at D=175 (3.69 vs 11.07°).
- **Confound:** M2 degrades badly under tight priors (unexplained), which inflates M1's within-σ margins.

**3. Width (Phase C, single seed, diagnostic):** **Partly, at D=150 only. Single seed; diagnostic, not a headline.**
- At D=150, M1's cut-locus-free advantage survives 4× capacity but narrows: width 2048 gives 2.06° vs
  2.60° (gap +0.54°), against +1.48° at width 1024 with the same seed. Both models improve with width.
- At D=30 the cell is uninterpretable: **M2 failed to train at width 2048** (loss stuck at the trivial
  0.667 from step 1k; 79.5° error) at its width-1024-tuned LR of 3e-3. M1 there improved 0.92 → 0.67°.

**4. What broke:** see the "What broke" section at the end.

## Housekeeping (done before any science)

- **The 17 orphaned background tasks from the previous session were dead.** No `night2`, worker, or
  monitor process was running (`pgrep` empty). They were stale monitor/scheduler handles; nothing was killed.
- **No half-written state.** Every `ckpt.pt` under `results/runs/` loads with `torch.load` (9 files), every
  `*.json` parses, and there are no `*.tmp` files left from an interrupted atomic write.
- **The σ=30 arm was deleted, not resumed.** `results/runs/n2_hp_gauss30_lr0.0003/`: 10 run dirs,
  3 finished training (2 also fully evaluated, which is night 2's "2 of 36"), 5 with valid partial
  checkpoints. It *was* cleanly resumable, but night-3 Phase B supersedes it. Deleted 255 MB.
  `results/runs/n2_pilot_gauss30/` was **kept**: it is the night-2 σ-pilot run that fed
  `sigma_select.json`, and it is reused below as the LR = 1e-3 point of the Phase-B σ=30 search.
- The night-1 orphan checkpoints `results/runs/diag_w2048/*/ckpt.pt` (step 2000 of a killed run,
  superseded by `n2_diag_w2048`) were deleted. Their `train.json` files remain.

## Pre-registrations (written before the runs they govern)
- `results/n3/prereg_phaseA.json`: gap definition, fits, and the numeric decision rule for the gate
  (per-doubling gap ratio at 160k→320k; (a) if its CI lower bound > 0.75 with g(320k) > 0, (b) if the
  ratios are ≤ 0.6 with CI upper bound < 0.75, or the gap reaches 0; (c) otherwise). Written before any
  160k/320k run started. It also records the prior expectation from night-2 B1 (implied ratio ~0.61,
  leaning (b)).
- `results/n3/prediction.json`: the brief's Phase-B prediction verbatim, plus operational tests and the
  risk I noted in advance that a tight σ could *raise* cut-locus exposure at D=175.

## Phase B, step 1 — the d0 mechanism table (no training)

d0 is the geodesic distance from the endpoint-waypoint source sample to the target endpoint (the night-2
definition). It is computed from the exact evaluation sources: test split, 200 starts × 5 samples × seeds
0–2 = 3000 per cell. Source: `src/d0_mechanism.py` → `results/n3/d0_mechanism.json`.

| prior | D=30: mean d0 [p5, p95]; >150; >165 | D=90 | D=175 |
|---|---|---|---|
| Haar | 127 [59, 176]; 0.34; 0.17 | 126 [58, 175]; 0.32; 0.16 | 126 [56, 175]; 0.32; 0.17 |
| trunc150 (oracle) | 109 [53, 147]; 0; 0 | 107 [50, 147]; 0; 0 | 107 [49, 147]; 0; 0 |
| Gauss σ=15 | 38 [17, 60]; 0; 0 | 92 [68, 116]; 0; 0 | 168 [150, 179]; **0.95; 0.67** |
| Gauss σ=30 | 56 [22, 96]; 0; 0 | 98 [52, 143]; 0.03; 0.01 | 157 [124, 178]; **0.71; 0.40** |
| Gauss σ=60 | 98 [36, 166]; 0.11; 0.05 | 116 [48, 173]; 0.24; 0.12 | 139 [79, 177]; 0.43; 0.22 |
| Gauss σ=90 | 122 [49, 175]; 0.29; 0.14 | 125 [56, 175]; 0.32; 0.16 | 130 [62, 176]; 0.35; 0.18 |

**Against the prediction:**
- "d0 ≈ D" **holds only for tight σ** (σ=15: 38/92/168°). For σ ≥ 60 the sources spread out and d0
  regresses toward the Haar value (~126°) at every D.
- "Tightening σ helps M1 substantially at D=30": the mechanism **supports** it. σ ≤ 30 removes all
  cut-locus exposure (0% beyond 150°), down from 29% at σ=90.
- "Less at D=90": **partly contradicted.** σ=15 also removes all exposure at D=90 (0% beyond 150°;
  1–3% at σ=30). What differs from D=30 is the mean d0 (92° vs 38°), not the cut-locus share. If M1's
  error grows smoothly with d0 below 150° (it did in night 2), tight σ should still help at D=90, but less
  than at D=30, for that reason.
- "Not at all at D=175": **FAILED, in the harmful direction** flagged in advance. At D=175, tightening σ
  *raises* cut-locus exposure: the share beyond 165° goes 0.18 (σ=90) → 0.22 (σ=60) → 0.40 (σ=30) →
  0.67 (σ=15). The mechanism predicts tight σ will *hurt* M1 at D=175, not leave it unchanged.
- **The stronger structural statement follows from the measurement:** at D=175, every start-centered
  Gaussian puts at least 18% of sources beyond 165°. The least-exposed is the widest (σ=90 ≈ Haar's 17%).
  No σ approaches the oracle's 0%.

## Phase A, interim (05:30): long-budget training instability, an "equal in form, unequal in effect" finding

The 320k runs exposed something the ≤ 80k data could not show. **The LRs selected at 20k are unstable
when the cosine schedule is stretched 16×,** and catastrophically so for M2 at 3e-3. Source:
`src/stability_n3.py` → `results/n3/stability.json`. Spike = a 500-step mean loss > 1.5× the previous one.

| run (D=150) | LR | spikes | first spike (step; LR as fraction of peak) | final loss | diverged | endpoint err |
|---|---|---|---|---|---|---|
| P3 M2 s0, 320k | 3e-3 | 4 | 36k; **0.97** (loss → 1580, then flat at 0.736 = trivial predictor) | 0.736 | **yes** | **103.2°** |
| P3 M2 s1, 320k | 3e-3 | 26 | 109k; 0.74 | 0.083 | **yes** (borderline) | **26.9°** (5 non-finite) |
| P2 M2 s2, 320k (running) | 3e-3 | 1 | 61k; 0.91 (loss → 30.7, now 0.778) | — | likely | — |
| P2 M2 s0, s1, 320k | 3e-3 | 0 | — | 0.006 | no | 1.62°, 1.54° |
| P3 M1 s0, s1, 320k | 1e-3 | 2, 5 | 104k, 71k; 0.76, 0.89 | 0.015 | no (recovered) | 1.23°, 1.19° |
| P2 M1 s0, s1, 320k | 1e-3 | 1, 3 | 174k, 88k; 0.43, 0.82 | 0.034 | no (recovered) | 2.36°, 2.07° |
| **all 30 runs at ≤ 80k** | | **0** | | | | |

- Every first spike in the long runs struck while the LR was at 43–97% of its peak, mostly ≥ 74%. At
  ≤ 80k there were none. This measures, rather than assumes, the mechanism: a stretched cosine keeps the
  LR near its 20k-tuned peak for 4–16× longer.
- M1's spikes recover. M2's at 3e-3 sometimes do not. The protocol LR is therefore unequal in effect at
  long budgets, this time **against M2**.
- **The pre-registered divergence rule had to change.** "Final loss > 5× the cell median" fails when 2 of
  3 seeds are pathological. The reference became 5× the same model/prior's healthy 80k final loss
  (addendum_1 in `prereg_phaseA.json`).
- **Response, pre-registered before it ran (addendum_1):** a matched-LR sensitivity arm, M2 at 1e-3
  under P3 at all five budgets (20k–320k), seeds 0–2. Both models then use 1e-3 under P3. Lowering M2's LR
  can only help M2, so the arm works *against* the standing P3 result. The gate is reported (i) on the
  pre-registered protocol LRs, with and without diverged runs, and (ii) on this arm, labeled as a post-hoc
  sensitivity. The arm never becomes "the" gate.
- **Revised timeline:** ~100 worker-hours remain at 05:30, ≈ 12.5 h on 8 workers, so done ≈ 18:00. If the
  session ends earlier, the brief's A > B > D > C order governs, and what didn't run will be stated.

**Update 05:56.** P3 M2 seed 2 at 320k has also diverged (spike at 116k, LR 0.71× peak, loss stuck at 0.737 =
trivial predictor). So **all 3 P3 seeds of M2 at 3e-3 failed at 320k**. The P3 160k seed-0 M2 run is
stuck at trivial loss after a spike at 43k (LR 0.84× peak). P2 seed 2 at 320k (spike at 61k) has not
recovered. The pre-registered protocol series therefore cannot answer Q1 under P3: "without diverged" has
no M2 seeds left at 320k, and "with diverged" measures M2's collapse. The matched-LR arm was moved ahead of
the remaining protocol 160k runs (priority change only; nothing was killed or excluded). The diverged runs
are left to finish, so the "with diverged" numbers are measured rather than assumed.

## Phase A — asymptotic or convergence-rate? (THE GATE)

D=150, M1 vs M2, 3 seeds, 20k → 320k steps. Each budget is a fresh run with the cosine schedule stretched
to that budget. Gap g = err_M2 − err_M1 (mean endpoint error, NFE 64; positive = M1 better). Sources:
`src/analyze_n3_a.py` → `results/n3/phaseA.json`, `src/stability_n3.py` → `results/n3/stability.json`.
Pre-registration: `results/n3/prereg_phaseA.json` (rule, and addendum_1 for the matched-LR arm).

### Gate outcome, stated as the two pre-registered labels

- **(i) Pre-registered protocol series** (P3, night-2 LRs: M1 1e-3, M2 3e-3): **(c) UNDECIDABLE, for a
  structural reason, not noise.** M2 at its 20k-tuned LR **diverged in 3 of 3 seeds at 320k** and in 1 of
  3 at 160k. With diverged runs removed there is no usable seed pair at 320k, so the pre-registered
  question cannot be asked. The rule's "with diverged" output reads (b), but only because the 320k gap
  CI is [−33, +186]° (M2 errors of 103°, 27° and 103°). That is an artifact of M2's collapse and carries
  **no information about convergence**.
- **(ii) Matched-LR sensitivity arm** (post-hoc, pre-registered in addendum_1 before it ran; M2 at 1e-3,
  so both models use 1e-3, all 15 runs stable with zero spikes): **(b) CONVERGENCE-RATE, by both
  pre-registered clauses independently.**
  - The per-doubling ratios fall: 0.45 [0.44, 0.46] (20k→40k), 0.46 [0.43, 0.48], 0.26 [0.18, 0.29], then
    **−0.28 [−0.60, −0.09]** (160k→320k). The last CI's upper bound is below 0.75, so the ratio clause fires.
  - **g(320k) = −0.07° [−0.16, +0.02].** The gap has reached zero (all three seeds are slightly negative),
    so the second clause fires too.
- **The only interpretable series says convergence-rate; the pre-registered series could not decide.**
  Under the cut-locus-free oracle prior, M1's advantage is a **sample-efficiency (convergence-rate)
  effect, not an asymptotic one**. This retracts the project's current framing of a Riemannian
  "correctness" advantage, and is reported as such.

| steps | M1 (1e-3) | M2 (1e-3), matched arm | gap [95% CI] | M2 (3e-3), protocol | protocol gap |
|---|---|---|---|---|---|
| 20k | 2.28 ± 0.03 | 7.03 | **+4.75** [3.87, 5.63] | 3.57 | +1.29 [0.83, 1.75] |
| 40k | 1.76 | 3.89 | +2.13 [1.82, 2.44] | 2.43 | +0.66 [0.48, 0.85] |
| 80k | 1.50 | 2.48 | +0.99 [0.70, 1.27] | 1.98 | +0.48 [0.37, 0.60] |
| 160k | 1.34 | 1.59 | +0.25 [0.05, 0.46] | 1.45 (2 healthy) + 1 diverged | +0.15 (healthy) |
| 320k | **1.21** | **1.14** | **−0.07 [−0.16, +0.02]** | **3/3 diverged** (103°, 27°, 103°) | — |

**The arm's LR choice delayed the crossing; it did not cause it.** The two surviving protocol M2 runs at 160k
(3e-3: 1.45°, 1.46°) are *better* than the matched-LR M2 at 160k (1.59°). Lowering M2's LR slowed it, so
a stable M2 at 3e-3 would have overtaken M1 sooner. Read ~275k as an upper bound on the crossing, given
stability.

**Observed crossing (matched LR): ~275k steps [bootstrap 247k–302k; 100% of resamples cross].** The two
models' own error curves meet *inside* the measured range, not beyond it.

**The brief's fit, reported, then explained.** Gap vs log2(steps), OLS on the matched-LR series: slope
**−1.15°/doubling [t-CI −1.42, −0.88; bootstrap −1.24, −1.09]**, extrapolated gap at 1M steps **−2.6°
[−2.8, −2.5]**. That extrapolation is impossible: both models are already at ~1.2°, so a −2.6° gap would
need negative error. A straight line in log(steps) cannot describe a gap that decays toward zero; the
per-doubling ratios and the observed crossing are the measurements. Own curves: M1 −0.26°/doubling
[−0.31, −0.21], M2 −1.41°/doubling [−1.72, −1.09]. The fitted lines do not meet *ahead* of 320k because
they already crossed within it.

**d0-stratified (matched LR, P3; bands 0–90 / 90–120 / 120–150°):**

| steps | M1 | M2 |
|---|---|---|
| 20k | 1.73 / 2.02 / 2.86 | 6.57 / 7.18 / 7.25 |
| 80k | 1.29 / 1.39 / 1.72 | 2.34 / 2.54 / 2.54 |
| 320k | **1.13 / 1.16 / 1.30** | **1.14 / 1.14 / 1.13** |

At 320k, M1's advantage is gone **in every d0 band**: a tie below 120°, and M2 ahead at 120–150°. The
night-2 claim that M1's cut-locus-free advantage "lives in the easy-d0 bands" does not survive training.
It, too, is a rate effect. One structural trace remains: **M1's error still rises with d0 (1.13 → 1.30)
while M2's is flat**, the same signature as the cut-locus mechanism, attenuated.

**Sample efficiency (the quantity (b) reframes the project around).** At matched LR, M1 reaches 2.28° at
20k steps; M2 needs ~80k to reach a similar 2.48°, **about 4× the steps**. Against the protocol M2
(3e-3), M1 at 20k (2.28°) already beats M2 at 40k (2.43°), about 2× the steps.

**The matched-LR arm has its own confound.** At 20k, M2 at 1e-3 (7.03°) is far worse than M2 at 3e-3
(3.57°). Matching the LR removes one asymmetry (long-run stability) and introduces another (short-run
speed), which inflates M1's early lead in the arm. **No single LR treated the two models equally across
the whole budget range.** This is the night-3 instance of the project's "equal in form, unequal in effect"
thread. Both lead to the same endpoint: M1's lead shrinks every doubling under either M2 LR.

**Convergence check (the brief's 10% criterion).** The training loss still fell **~37% over the last
quarter of the 320k runs** (matched arm: M1 0.37, M2 0.37). This metric includes the cosine decay, which
inflates it, but it is far above 10%. **320k is also not converged, and the extrapolation is the claim,
not the measurement.** What *is* measured: the gap went from +4.75° to −0.07° across 20k → 320k, and
crossed zero inside the measured range.


**Final train loss and last-quarter relative loss drop at each budget (mean over seeds; M1 / M2).** The drop
includes the cosine LR decay to zero, so it overstates distance from a plateau; it is the night-2 metric.

| series | steps | n | final loss | last-quarter drop |
|---|---|---|---|---|
| P3 matched LR | 20k | 3 | 0.0476 / 0.0411 | 0.24 / 0.22 |
| P3 matched LR | 40k | 3 | 0.0389 / 0.0305 | 0.26 / 0.25 |
| P3 matched LR | 80k | 3 | 0.0305 / 0.0228 | 0.30 / 0.28 |
| P3 matched LR | 160k | 3 | 0.0194 / 0.0166 | 0.38 / 0.30 |
| P3 matched LR | 320k | 3 | 0.0151 / 0.0111 | 0.37 / 0.37 |
| P2 protocol, diverged removed | 20k | 3 | 0.0916 / 0.0267 | 0.17 / 0.38 |
| P2 protocol, diverged removed | 40k | 3 | 0.0742 / 0.0201 | 0.20 / 0.41 |
| P2 protocol, diverged removed | 80k | 3 | 0.0551 / 0.0169 | 0.23 / 0.43 |
| P2 protocol, diverged removed | 160k | 3 | 0.0450 / 0.0125 | 0.23 / 0.50 |
| P2 protocol, diverged removed | 320k | 2 | 0.0335 / 0.0059 | 0.28 / 0.64 |

Figure: `results/figures/n3_steps_gap.png` (gap vs budget for the matched-LR arm and both protocol series, with
diverged runs removed). Phase B figure: `results/figures/n3_sigma_gap.png`.

**Secondary (P2, Gaussian σ=90, protocol LRs):** the same picture from the other side.
- M2's lead shrinks every doubling: −6.98 (20k) → −5.66 → −3.67 → −2.83 (160k), then
  **−0.63° [−1.97, +0.71] at 320k** on the two healthy seeds. Seed 2 of M2 diverged (124°) and is
  excluded by the addendum rule.
- Ratios: 0.81, 0.65, 0.77, then 0.22 (n=2, no CI). **By the rule: (c) UNDECIDABLE** (n=2 at 320k, CI
  straddles 0). Reported, not picked.
- **The cut-locus failure is itself largely a rate effect.** M1's error on sources ≥ 165° from the target
  falls 42.6 → 34.5 → 23.3 → 18.3 → **5.3°** from 20k to 320k. Below 165°, the two models are tied in every
  band at 320k (M1 1.56–1.63°, M2 1.50–1.63°), so M2's remaining 0.63° lead comes entirely from that band.
  Nights 1–2 treated the cut-locus tail as structural. With 16× the training, most of it is learned away
  (5.3° vs M2's 1.6° in that band; not all of it).

| steps | M1 by d0 band: <90 / 90–120 / 120–150 / 150–165 / ≥165 | M2 (same bands) |
|---|---|---|
| 20k | 2.66 / 3.02 / 3.94 / 7.03 / **42.6** | 3.26 / 3.52 / 3.56 / 3.44 / 3.46 |
| 80k | 1.88 / 2.00 / 2.16 / 2.90 / **23.3** | 2.00 / 2.04 / 2.03 / 1.93 / 2.05 |
| 160k | 1.79 / 1.78 / 1.86 / 2.12 / **18.3** | 1.76 / 1.77 / 1.76 / 1.65 / 1.77 |
| 320k (n=2) | 1.58 / 1.56 / 1.62 / 1.63 / **5.3** | 1.60 / 1.63 / 1.59 / 1.50 / 1.58 |

**Phase A in one line:** with enough training, under either prior, the two models converge toward the
same error (~1.1–1.6° at D=150). M1 gets there faster when the source avoids the cut locus, and slower when
it does not. The differences nights 1–2 measured at 20k were differences in *rate*, including the
cut-locus tail.

## Phase B, step 2 — LR search per (model, σ) at D=90 (night-2 protocol, edge rule on M1+M2 jointly)

Val mean endpoint error (deg) at NFE 32, seed 100, 20k. Bold = selected. `results/n3/hparams_gauss*.json`.
Protocol-identical night-2 runs were reused: σ=30/60 at 1e-3 = the night-2 σ pilot runs, and σ=90 = the
D=90 subset of the night-2 σ=90 search.

| σ | M1: 1e-4 / 3e-4 / 1e-3 / 3e-3 / 1e-2 | M2: same LRs | picks |
|---|---|---|---|
| 15 | 9.57 / 4.63 / 1.91 / **1.86** / 5.32 | 20.07 / **13.06** / 16.91 / 16.77 / 83.7 | M1 3e-3, M2 3e-4 |
| 30 | — / 2.45 / 1.62 / **1.39** / 106.8 | — / 15.12 / **13.84** / 20.47 / 101.7 | M1 3e-3, M2 1e-3 |
| 60 | — / 8.95 / 4.72 / **4.36** / 117.6 | — / 12.04 / **11.74** / 12.14 / 116.6 | M1 3e-3, M2 1e-3 |
| 90 | 17.96 / **8.41** / 9.22 / 8.47 / 123.7 | 15.56 / 4.92 / 4.14 / **2.23** / 124.2 | M1 3e-4, M2 3e-3 |

- **The tuning stage already shows a reversal with σ at D=90.** With a tight source, M1 is at 1.4–1.9° and
  M2 at 13–14°. At σ=90, M2 is at 2.2° and M1 at 8.4°. M2's *degradation* with a tight source is not
  predicted by the cut-locus mechanism (M2 has no cut locus), and **I have no verified explanation for it**.
- **The σ=90 M1 pick differs from night 2.** Under tonight's D=90-only protocol it is 3e-4, not 1e-3, by a
  0.06° single-seed margin (8.41 vs 8.47). The 9 M1 σ=90 main runs therefore use 3e-4. The night-2 1e-3
  runs are reported beside them as a sensitivity: a tie decided inside seed noise.

## Phase B, step 3 — trained results (M1, M2; 3 seeds; 20k; test NFE 64)

Source: `src/analyze_n3_b.py` → `results/n3/phaseB.json`, figure `results/figures/n3_sigma_gap.png`.
Gap = M2 − M1, mean endpoint error (deg), seed-paired 95% CI; **positive = M1 better**.

| σ | D=30: M1 / M2; gap | D=90 | D=175 |
|---|---|---|---|
| 15 | 0.83 / 17.02; **+16.2 [15.5, 16.9]** | 1.92 / 13.03; **+11.1 [10.5, 11.7]** | 36.34 / 14.99; −21.4 [−23.4, −19.3] |
| 30 | 0.63 / 11.26; **+10.6 [9.8, 11.5]** | 1.48 / 13.34; **+11.9 [11.4, 12.3]** | 26.01 / 12.09; −13.9 [−16.8, −11.0] |
| 60 | 2.85 / 8.93; **+6.1 [3.7, 8.5]** | 4.41 / 12.42; **+8.0 [5.8, 10.3]** | 11.07 / 11.58; +0.5 [−3.2, +4.2] (tie) |
| 90 | 19.99 / 4.95; −15.0 [−19.7, −10.4] | 9.02 / 2.47; −6.6 [−11.0, −2.2] | 14.16 / 3.76; −10.4 [−10.8, −10.0] |

σ=90 LR sensitivity: with M1 at the night-2 LR (1e-3) instead of tonight's D=90-only pick (3e-4), the
gaps are −5.2 / −7.0 / −7.8. The sign is the same at every D, but the D=30 magnitude halves. A
single-seed LR tie (0.06°) moved M1's D=30 error from 10.1° to 20.0°.

**M1's d0-stratified error at D=175** (bands < 90 / 90–120 / 120–150 / 150–165 / ≥ 165):
- σ=15: — / — / 7.05 / 10.3 / **49.4**
- σ=30: 0.88 / 4.07 / 5.95 / 10.5 / **52.8**
- σ=60: 2.65 / 2.67 / 3.33 / 5.63 / **36.4**
- σ=90: 2.63 / 3.27 / 5.36 / 10.5 / **54.3**

The cut-locus band is where M1 loses at D=175, under every σ. M2 is flat across bands.

### Did the prediction hold?

| clause | outcome | evidence (M1 error across σ = 90 → 60 → 30 → 15) |
|---|---|---|
| "helps M1 substantially at D=30" | **held** | 20.0 → 2.85 → 0.63 → 0.83 (best at σ=30) |
| "less at D=90" | **held in ratio, not in kind** | 9.02 → 4.41 → 1.48 → 1.92: a ~6× improvement vs ~30× at D=30; still large, with the cut locus fully removed |
| "NOT AT ALL at D=175" | **FAILED, both ways** | 14.2 → 11.1 → 26.0 → 36.3: moderate tightening helps slightly, tight priors hurt a lot (≥ 165° share 18% → 22% → 40% → 67%, as the mechanism table predicted) |
| "no start-centered prior rescues M1 at large D" | **held** | best M1 at D=175 is 11.1° (σ=60), a tie with that σ's M2 and far from M2's best (3.69°, Haar) |

### Answer to Q2

- **Yes: a deployable prior gives M1 a win, at D ≤ 90.** At σ ≤ 60, M1 beats M2 at D=30 and D=90 by 6–16°
  (9–86 sd). **No σ gives M1 a win at D=175.** The best is a tie at σ=60.
- **σ shifts the crossover D; it does not just rescale the gap. The shift is non-monotone in σ.**
  - The D at which M1 stops winning is below 30° at σ=90, between 90° and 175° at σ=15–30, and about 175°
    (a tie) at σ=60.
  - Tight σ shifts the crossover up at small D but makes the large-D loss *worse*. The largest D at which
    M1 still ties is reached at an intermediate σ.
  - This follows the mechanism table directly. Tight σ puts d0 ≈ D: good when D is small, fatal when D is
    near 180°.
- **A large confound, flagged rather than resolved: M2 degrades badly under tight priors.**
  - M2's error at D=30 rises from 4.95° (σ=90) to 8.93° (σ=60), 11.26° (σ=30) and 17.02° (σ=15). Its
    raw orthogonality residual rises to 0.75 (vs 0.21 at σ=90), while translation stays fine.
  - The LR search found nothing better than 12–17° at D=90 for σ ≤ 60, across five LRs. The error is
    NFE-independent from NFE 4 to 64.
  - M2 has no cut locus, so the mechanism does not predict this. **I have no verified explanation.** An
    untested hypothesis: a low-entropy source makes all 32 waypoint states nearly identical at small t,
    and the 6D MLP fails to separate them.
  - Part of M1's "win" at σ ≤ 60 is therefore M2 losing.
- **The fair version: compare each model at its own best deployable prior** (20k steps, same task, same D):

  M2's numbers are 5 seeds under the night-1 protocol (LR tuned over 3 D). M1's are 3 seeds, tuned at D=90
  only. Both are 20k steps on the same test set.

  | D | M1 best (prior) | M2 best (prior) | winner |
  |---|---|---|---|
  | 30 | **0.63** (σ=30) | 1.22 (Haar, 5 seeds) | M1 |
  | 90 | **1.48** (σ=30) | 2.38 (Haar, 5 seeds) | M1 |
  | 175 | 11.07 (σ=60) | **3.69** (Haar, 5 seeds) | M2, by 3× |

  At D ≤ 90, M1's best deployable configuration beats M2's best configuration. M1's errors at σ=30 are its
  lowest of the whole project, below even the oracle prior's 0.92 / 1.67. At D=175, M2 wins and no
  deployable prior changes that. **That is the structural limit the brief predicted, confirmed by
  measurement rather than by the prediction's own reasoning.** (Phase A says these are 20k-step numbers;
  with longer training, gaps shrink. Phase B was not run at long budgets.)

## Phase C — width 2048 at the fair prior (optional; 1 seed; DIAGNOSTIC ONLY)

P3 (trunc150), D ∈ {30, 150}, seed 0, 20k steps. Each model uses its P3 LR, **tuned at width 1024**
(M1 1e-3, M2 3e-3). The comparison is against `n2_main_trunc150` seed 0 at width 1024 (same prior, LR,
budget, seed). Parameters: M1 14.19M, M2 13.80M.

| D | model | width 1024 (mean / p95) | width 2048 (mean / p95) | spikes (2048) |
|---|---|---|---|---|
| 30 | M1 | 0.92 / 1.79 | **0.67 / 1.37** | 0 |
| 30 | M2 | 1.31 / 2.33 | **79.5 / 127.1: failed to train** | 0 (the loss collapsed to 0.667, the trivial level, during warmup, before the spike counter's first window) |
| 150 | M1 | 2.26 / 4.46 | 2.06 / 4.34 | 0 |
| 150 | M2 | 3.74 / 6.90 | 2.60 / 4.92 | 0 |

- **D=150: the M1 advantage survives 4× capacity, narrower** (gap +1.48° at width 1024 → +0.54° at 2048,
  single seed). This is consistent with Phase A: more capacity, like more steps, closes the gap without
  (yet) reversing it at 20k.
- **D=30: uninterpretable.** M2 at 3e-3 never trained at width 2048. This is the same LR-transfer failure
  as Phase A, this time across width instead of budget, and again it hit M2.
- Night 2's four-model ranking flip at width 2048 was under Haar. Under P3 at D=150, the M1-vs-M2 sign did
  not flip at 2048 (one seed).


## What broke

1. **The LR protocol does not transfer across budgets.**
   - LRs tuned at 20k (cosine) produced loss spikes in most 160k/320k runs of *both* models (every first
     spike with the LR at 43–97% of its peak). There were none in 30 runs at ≤ 80k.
   - M1 at 1e-3 recovered every time. **M2 at 3e-3 diverged** in 3/3 P3 seeds at 320k, 1/3 at 160k, and
     1/3 P2 seeds at 320k.
   - An identical protocol was unequal in effect, this time against the baseline. Budget is confounded
     with time-at-peak-LR, and separating them needs per-budget tuning or a constant-then-anneal schedule.
     Neither was run.
2. **The pre-registered divergence rule failed.** "5× the cell median" breaks when two of three seeds
   diverge. It was replaced (addendum_1) by 5× the same model/prior's healthy 80k loss.
3. **The pre-registered Phase-A series could not answer the gate.** The answer comes from a sensitivity
   arm specified before it ran. It is labeled as such and kept separate from the protocol result.
4. **The matched-LR arm has its own asymmetry.** M2 at 1e-3 is slow at short budgets (7.0° vs 3.6° at
   20k), which inflates M1's early lead. No single LR treated both models equally across the budget range.
5. **The brief's requested fit is misspecified.** A linear fit of the gap in log(steps) extrapolates to an
   impossible −2.6° at 1M. The per-doubling ratios and the observed crossing replace it as the evidence.
6. **The Phase-B prediction failed at D=175, in the harmful direction.** Tightening σ concentrates sources
   at the antipode. This was caught by the mechanism table before training, and the trained results agree.
7. **M2 degrades under tight start-centered priors (8.9–17° at D ≤ 90 for σ ≤ 60).** Unexplained. The raw
   6D residual triples, translation stays fine, no LR helps, and the error does not depend on NFE. It
   confounds the within-σ Phase-B margins. The best-prior-per-model comparison sidesteps it, but the
   mechanism remains open.
8. **LR ties decided inside seed noise had large consequences.**
   - Under tonight's D=90-only protocol, the σ=90 M1 pick (3e-4) beat 1e-3 by 0.06° on one tuning seed.
     It doubled M1's D=30 test error (10.1 → 20.0°).
   - Both picks are reported. Tuning on one seed at one D is too thin for picks this consequential.
9. **Width transfer of the LR (Phase C).** M2 at its width-1024 LR failed to train at width 2048 (D=30).
   An LR tuned at one capacity is not validated at another.
10. **Seeds.** Phases A and B use 3 seeds (as briefed); CLAUDE.md asks for ≥ 5 on headline numbers. Phase C
   uses 1. P2's Phase-A gate is undecidable partly because n=2 at 320k after one divergence.
11. **Runtime.** Planned as one night, it took ~17 h of wall clock. The causes were the 320k runs (~5 h each
    on P3 with its rejection sampler), the 15-run matched-LR arm (~3 h of worker time), and three scheduler
    restarts to add or reprioritize tasks. In-flight runs resumed from checkpoints and lost ≤ 2k steps each;
    no run was restarted from scratch with different settings. A ~25-minute outage of the tool permission
    check delayed analysis, not training.
12. **Phase B's absolute numbers are 20k-step numbers.** Phase A shows every gap shrinks with training, so
    Phase-B margins (especially M1's within-σ wins) are not asymptotic claims.
