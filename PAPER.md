# What Decides Riemannian versus Euclidean Flow Matching on SO(3)? Source Priors, the Cut Locus, and Training Budget in a Controlled Study

_Draft. Synthetic study, single machine (CPU). All numbers come from `FINDINGS_SE3_N1.md`, `FINDINGS_N2.md`
and `FINDINGS_N3.md` in this repository, and every table can be regenerated from `results/`._

## Abstract

Riemannian flow matching on SO(3) is often reported to beat Euclidean flow matching for generating rotations
and rigid-body trajectories. Those Euclidean baselines usually parameterize rotations with quaternions or
Euler angles, which are discontinuous. We compare four conditional flow-matching models that differ *only*
in the rotation parameterization and interpolation path: geodesic flow on SO(3), and straight-line flow on
6D, quaternion and XYZ-Euler representations. Architecture, budget and tuning protocol are matched. The
task is a synthetic family of SE(3) trajectories whose angular displacement D we control from 10° to 175°.

**The answer to "which is better" is not determined by the method alone. It depends on the source (prior)
distribution, and the effect is large.**
- Under a uniform (Haar) prior, 6D Euclidean flow matching beats the Riemannian model at every D, by
  6.6–14.4° mean endpoint error.
- Removing source samples near the SO(3) cut locus reverses the sign. Under an oracle prior truncated at
  150°, the Riemannian model wins at every D ≥ 30° by 3–13 seed standard deviations.
- Under a deployable start-centered Gaussian prior, the outcome depends on the spread σ and on D, through
  how much source mass lands near the target's antipode.

The mechanism is specific and measurable. The geodesic conditional velocity field is discontinuous at the
cut locus, and every catastrophic (>45°) Riemannian error comes from a source sample more than 165° from
its target. Against quaternion and Euler baselines, Riemannian flow matching wins above crossover angles
of 54° and 94°, which reproduces the published direction for reasons unrelated to the manifold. Rankings
are also sensitive to training budget and network width: gaps between models shrink by roughly half per
doubling of steps, and one four-model ranking flips when width doubles.
Under the cut-locus-free prior, the surviving Riemannian advantage is a convergence-rate effect, not an
asymptotic one. At matched learning rate it is about 4× more sample-efficient, but the two models' errors
cross at ~275k steps. At the learning rate tuned for short budgets, the 6D model diverged on long
schedules, so an identical protocol was not equal in effect. A deployable start-centered prior lets the Riemannian model win at displacements up to 90°: with each model at its own best prior, 0.63° vs 1.22° at D=30 and 1.48° vs 2.38° at D=90. No deployable prior rescues it at D=175, where tight priors concentrate source mass on the cut locus.

We conclude that published comparisons are underdetermined unless they report the source distribution,
its distance to the cut locus, error stratified by that distance, and the training budget.

## 1. Introduction

Generating rotations is central to robot policy learning, pose estimation and molecular modeling. Two
families of generative models dominate. Riemannian models respect the geometry of SO(3) (or SE(3)) by
transporting probability mass along geodesics. Euclidean models embed rotations in R^d and transport along
straight lines. Riemannian flow-matching policies have been reported to give smoother trajectories with
fewer integration steps than Euclidean baselines (Braun et al., 2024).

Zhou et al. (2019) showed a confound in such comparisons. Every rotation representation in four or fewer
real dimensions is discontinuous, including quaternions (once their sign ambiguity is resolved) and Euler
angles, and networks regressing them make occasional errors of up to 180°. Continuous representations,
such as the 6D first-two-columns representation, avoid this. Zhou et al. also note that a discontinuous
representation may be harmless if the data stays away from its discontinuities. When a Riemannian model
beats a quaternion baseline, the win could come from the manifold, or from the baseline's discontinuity.

We designed a controlled study to separate these. We held fixed:
- the architecture (a 4-hidden-layer MLP over a 32-waypoint trajectory);
- the training budget, optimizer, schedule and LR-search protocol;
- the conditioning;
- the translation treatment;
- the stream of source samples (every model sees the same rotations, encoded into its own representation).

Only the rotation parameterization and interpolation path vary. The angular displacement D of the task
moves the data toward or away from each representation's singularities.

The study went through three rounds, each prompted by the previous one's confounds. Our first result
(6D beats Riemannian at every D) did not survive scrutiny. The Riemannian deficit came almost entirely
from one mechanism, the cut locus of SO(3) meeting a uniform source prior, which handicaps the geodesic
model alone. Changing the prior reversed the result. Training longer narrowed every gap, and changing the
network width reordered the models. We therefore present the comparison as *underdetermined by what the
literature typically reports*, and characterize what determines it.

**Contributions.**
1. A controlled four-way comparison isolating rotation parameterization in conditional flow matching, with
   crossover angles against discontinuous baselines (Section 5.1).
2. Identification and measurement of the cut-locus failure mode of geodesic flow matching under broad
   priors, with d0-stratified error tables (Section 5.2).
3. Evidence that the Riemannian-vs-Euclidean sign depends on the source prior (Section 5.3), on training
   budget and on width (Section 5.4), and only partially on multimodality (Section 5.5).
4. A pre-registered test of whether a deployable prior can avoid the cut locus at large displacement
   (Section 5.6). It can at D ≤ 90°, and it cannot at D=175°, where tightening the prior makes the cut-locus
   exposure worse, contrary to our prediction that it would have no effect.
5. Reporting recommendations (Section 7).

## 2. Related Work

**Rotation representations.** Zhou et al. (2019) proved that rotation representations in ≤ 4 real
dimensions are discontinuous with respect to SO(3), and showed empirically that the discontinuity causes
rare, very large regression errors. They proposed continuous 5D and 6D representations; the 6D one maps to
SO(3) by Gram-Schmidt on two 3-vectors. Our M2 uses it. Our M3 (quaternion, w ≥ 0 canonical) and M4
(intrinsic XYZ Euler) are the discontinuous baselines, and our D sweep tests Zhou et al.'s remark that
discontinuities matter only when the data approaches them.

**Flow matching.** Flow matching and rectified flow (Lipman et al., 2023; Liu et al., 2023) train a
time-dependent velocity field by regressing conditional velocities along a prescribed path between source
and data samples. With straight-line conditional paths, the target is linear in the current state.

**Riemannian flow matching.** Chen and Lipman (2024) extend flow matching to Riemannian manifolds, using
geodesic conditional paths on simple manifolds and premetric constructions on general ones. For an
embedded manifold they parameterize the vector field as an ambient output projected onto the tangent space.
We follow that recipe for M1; a naive direct body-frame output did not learn (Section 6). On SO(3) the
geodesic path is γ(t) = R0 exp(t log(R0ᵀR1)), whose log is discontinuous where R0 and R1 are antipodal (the
cut locus). That is the mechanism we identify in Section 5.2. Related SE(3)/SO(3) generative models for
protein backbones (e.g. FrameFlow, FoldFlow) use geodesic or related paths and often prefer non-uniform
priors. We do not evaluate them.

**Riemannian flow-matching policies.** Braun et al. (2024) apply Riemannian flow matching to robot
visuomotor and trajectory policies on manifolds including SO(3) and SE(3), and report smoother
trajectories with fewer integration steps than Euclidean baselines. Our study does not replicate their
setting. It asks a narrower question their comparison leaves open: how much of a Riemannian advantage
survives when the Euclidean baseline uses a continuous representation, and what else decides the sign.

## 3. Method

### 3.1 Task

For each D ∈ {10, 30, 60, 90, 120, 150, 175}°, we generate 2000 SE(3) trajectories of 32 waypoints
(split 1600/200/200):
- axis a ~ Uniform(S²), start tilt σ_s ~ U[10°, 20°];
- start R0 = exp(σ_s·a), p0 ~ N(0, 0.1²I);
- target R1 = R0·exp(D·a), p1 = p0 + 0.3·a.

So d(R0, R1) = D exactly, and the target is a deterministic function of the start: the axis can be read
from R0. That makes endpoint error well defined for a start-conditioned model.

Path types come in equal thirds, all with minimum-jerk timing: geodesic screw motion, a via-point detour,
and a rotational overshoot of 10–25% of D. Starts sit near the identity, so D controls how close the data
comes to each representation's singularities. No trajectory crosses the quaternion hemisphere boundary
below D=150 (31% do at D=150, 100% at D=175). Euler jumps appear from D=90 (3%) and reach 75% at D=175.

A **multimodal** variant gives each start three valid targets: rotation by D about a and about axes ±60°
from a. Each trajectory takes one target uniformly at random.

### 3.2 Models

There are four conditional flow-matching models, each predicting a velocity.

| | state | conditional path | ODE step | final map |
|---|---|---|---|---|
| M1 Riemannian | R ∈ SO(3) | geodesic R0 exp(t log(R0ᵀR1)) | R ← R exp(h·Ω) | none (stays on SO(3)) |
| M2 Euclidean-6D | first two columns of R | straight line | x ← x + h·v | Gram-Schmidt |
| M3 Euclidean-quat | canonical quaternion (w ≥ 0) | straight line | x ← x + h·v | normalize |
| M4 Euclidean-Euler | XYZ Euler angles | straight line | x ← x + h·v | Euler → matrix |

M1's network outputs an ambient 3×3 matrix per waypoint, and the velocity is its projection onto the
tangent space, Ω = vee(Rᵀ A) (Chen & Lipman, 2024).

Shared by all four:
- An MLP over the full trajectory state with 4 hidden layers of width 1024 and SiLU activations
  (3.6–3.9M parameters; the differences come only from I/O dimensions).
- Conditioning on the start pose (flattened R0 and p0) and on t.
- Translation handled as Euclidean, with a straight-line path.
- Adam, 500-step warmup, cosine decay, batch 256, gradient clipping at 1.0.

Per seed, all models share the minibatch indices, time samples and source rotations. All geometry
(log/exp, interpolation, ODE state updates, metrics) is computed in float64 on CPU.

### 3.3 Source distributions (priors)

- **P1 Haar:** uniform on SO(3), drawn independently per waypoint.
- **P2 Gaussian(σ):** R0·exp(ŵ) with w ~ N(0, σ²I), centered on the conditioning start pose. This is
  deployable, since the start is known at inference.
- **P3 Haar truncated at 150°:** Haar, rejecting samples more than 150° from that waypoint's target. This
  is an **oracle** (it needs the target) and is used only as an ablation.

### 3.4 Tuning protocol

For each model, prior and task, we searched the LR over {3e-4, 1e-3, 3e-3} at tuning displacements, on a
tuning seed disjoint from the evaluation seeds. The criterion was validation mean endpoint error at NFE=32.
A pre-registered rule extends the grid one step on a side for *all* models if *any* model's pick falls on
that edge. The extension fired in every search, and every final pick was interior. The step budget is 20k
unless stated otherwise.

### 3.5 Metrics

- **Endpoint geodesic error** (deg) between the generated last waypoint and the target, reported as mean,
  median, 95th percentile and max, over 200 test starts × 5 samples.
- **d0:** the geodesic distance from the endpoint-waypoint source sample to the target. We stratify errors
  by d0.
- **Orthogonality residual** ‖RᵀR − I‖ of the raw final state, before any projection.
- **Angular jerk.**
- **Multimodal metrics:** W1 (exact optimal transport, geodesic cost) between each start's 30 samples and
  the uniform 3-mode target; nearest-mode error; and mode coverage (a sample within 10° of a mode counts
  as a hit).
- **Statistics:** headline numbers use 5 seeds (budget sweeps use 3), with seed-paired differences,
  t-based CIs, and the gap expressed in pooled seed standard deviations.

## 4. Experimental scale

Every run trained on a single CPU (8 parallel workers). We used 140 main runs at the uniform prior,
200 more across two further priors, LR searches with 60 runs per prior, and step sweeps up to
320k steps. The scale limits what we can claim about convergence (Section 6).

## 5. Results

### 5.1 Against discontinuous baselines, the Riemannian model wins above a crossover angle

Under the uniform prior (all seven D, 5 seeds, NFE 64), the Riemannian model's mean endpoint error is
roughly flat in D (9.5–18.0°). The discontinuous baselines degrade with D. Quaternion (M3) error rises
from 10° to 20°, and Euler (M4) error rises from 4.4° to 21.8°. Euler's worst-case error jumps from 29° at
D=60 to 172° at D=90, exactly where the data first produces Euler discontinuities. This is the Zhou et al.
failure mode, visible as a phase change in the maximum error but not in the mean.

Seed-bootstrap crossover angles, where the baseline's mean error overtakes M1's: **54° [95% CI 50–59]
for quaternion** and **94° [92–98] for Euler**. (The CIs reflect seed variance only; the 30° grid spacing
is the larger uncertainty.) A study comparing a Riemannian model to quaternion or Euler baselines on
large-rotation tasks would therefore report a Riemannian win of up to 11° (7–19 seed-sd).

Against the continuous 6D baseline under the same prior, the sign reverses. M2 beats M1 at every D, by
6.6–14.4° (7–19 sd). The quaternion deficit is only partly explained. At small D it is driven by pairs
whose quaternion chord passes near the origin (q0·q1 < 0: 15.4° vs 10.0° error at D=10). At D ≥ 90 it is
flat and unexplained, and it shows no step at D=150/175 where sign flips enter the data.

### 5.2 The cut-locus mechanism

The uniform-prior Riemannian deficit is not spread across samples. Over 35,000 test samples, every M1
error above 45°, at every D, comes from a source sample more than 165° from its target:

| d0 (deg) | share | M1 mean err | M1 frac > 45° | M2 mean err |
|---|---|---|---|---|
| 0–90 | 0.18 | 2.5 | 0.000 | 3.0 |
| 90–120 | 0.21 | 3.4 | 0.000 | 3.1 |
| 120–150 | 0.28 | 5.2 | 0.000 | 3.1 |
| 150–165 | 0.16 | 10.3 | 0.000 | 3.1 |
| 165–175 | 0.11 | 27.0 | 0.143 | 3.1 |
| 175–180 | 0.06 | 91.7 | 0.803 | 3.1 |

Near the cut locus, log(R0ᵀR1) is discontinuous: the rotation axis flips sign at 180°. The conditional
velocity field therefore has opposite targets on either side of a measure-zero set, and a smooth network
regresses toward their average (≈ zero), so those samples stall. The Haar density ∝ (1 − cos θ) puts about
16% of source samples within 15° of the antipode. The straight-line path in 6D has no such discontinuity,
and M2's error is independent of d0 (correlation 0.02, vs 0.47 for M1). The quaternion and Euler models'
errors are also independent of d0.

(This is at 20k steps. Under the Gaussian σ=90 prior, the Riemannian model's error in the ≥ 165° band
falls from 42.6° at 20k to 5.3° at 320k; see §5.6. The failure is real, but mostly a rate effect.)

A second, less obvious effect: **training on cut-locus samples degrades the Riemannian model even on
far-from-cut-locus samples.** M1's error at d0 < 90° is 2.65° when trained under Haar, 2.25° under
Gaussian σ=90, and 1.33° under the truncated prior (pooled over D ∈ {30, 90, 150, 175}). Stratifying at
test time alone therefore understates what the cut locus costs.

### 5.3 The sign depends on the prior

We re-ran the comparison under three priors, re-tuning every model's LR under each.

**Mean endpoint error (deg), 5 seeds, NFE 64, 20k steps (§5.6 shows these gaps narrow with training):**

| prior | model | D=30 | D=90 | D=150 | D=175 |
|---|---|---|---|---|---|
| Haar | M1 | 15.63 ± 1.65 | 9.49 ± 0.63 | 11.22 ± 0.60 | 10.67 ± 0.85 |
| | M2 | **1.22 ± 0.04** | **2.38 ± 0.10** | **3.56 ± 0.18** | **3.69 ± 0.19** |
| Gauss σ=90 | M1 | 10.37 ± 0.75 | 9.34 ± 0.99 | 10.43 ± 0.20 | 11.23 ± 0.47 |
| | M2 | **4.93 ± 1.22** | **2.46 ± 0.04** | **3.39 ± 0.17** | **3.78 ± 0.20** |
| Trunc 150 (oracle) | M1 | **0.92 ± 0.02** | **1.67 ± 0.03** | **2.27 ± 0.03** | **2.36 ± 0.03** |
| | M2 | 1.79 ± 0.40 | 2.39 ± 0.13 | 3.55 ± 0.15 | 3.81 ± 0.16 |

**d0-stratified, pooled over D ∈ {30, 90, 150, 175}, M2 − M1 (positive = M1 better); all at 20k steps. See §5.6 for how these change with training:**

| prior | d0 < 90 | 90–120 | 120–150 | 150–165 | 165–175 | all d0 < 150 |
|---|---|---|---|---|---|---|
| Haar | −0.01 (0.1 sd) | −0.57 (4.8) | −2.17 (9.5) | −6.81 (14.3) | −22.7 (16.5) | −1.09 (7.3) |
| Gauss σ=90 | **+1.29 (5.9)** | **+0.83 (3.8)** | −0.45 (2.0) | −3.97 (12.3) | −17.3 (27.4) | +0.43 (2.0) |
| Trunc 150 | **+1.47 (19.1)** | **+1.36 (23.0)** | **+0.63 (9.2)** | — | — | **+1.08 (19.0)** |

The sign of the comparison flips with the prior. With no cut-locus samples, the Riemannian model is better
at every D ≥ 30 and in every d0 band, by 19 pooled seed-sd overall. Under the uniform and broad-Gaussian
priors, the 6D model wins overall, and its margin comes from source samples near the antipode. The
quaternion and Euler models are insensitive to the prior: only the model with a cut locus cares.
All of this is at 20k steps. Section 5.6 shows that the truncated-prior advantage is a convergence-rate
effect that vanishes by ~275k steps.

### 5.4 Budget and width sensitivity

**Training budget (20k → 40k → 80k steps, 3 seeds, D=150; extended to 320k in §5.6):**
- Under Gauss σ=90, the M2 − M1 gap goes −7.0 → −5.7 → −3.7° (M2 ahead, shrinking).
- Under the truncated prior, it goes +1.29 → +0.48° at 80k (M1 ahead, shrinking, 8.5 sd).
- At D=30 under the truncated prior, the 80k gap (+0.08°, CI [−0.04, +0.20]) is no longer distinguishable
  from zero.
- Night-2's d0-stratified "Riemannian is better at d0 < 120° under Gauss σ=90" is *not* budget-stable. By
  80k it is a tie at D=150 and an M2 win at D=30. It reflected an undertrained 6D model at 20k.
- No model converged: the training loss was still falling 22% (M1) and 45% (M2) over the last quarter of
  an 80k run.

**Width (single seed, Haar, D=90, 10k steps).** The four-model ranking changed at every width tested:

| width | ranking (best first) |
|---|---|
| 512 | M1 < M4 < M2 < M3 |
| 1024 | M4 < M2 < M1 < M3 |
| 2048 | M2 < M4 < M3 < M1 |

Under the truncated prior at D=150 (single seed), doubling width to 2048 narrowed the Riemannian
advantage (+1.48° → +0.54°) without reversing it. At D=30, the 6D model at its width-1024 learning rate
failed to train at width 2048, another case of an LR not transferring, this time across capacity.

The specified 256-wide network could not learn the task for any model: the Riemannian model sat at the
Haar chance level of 126°. We report results at width 1024 throughout, and treat any width-fixed ranking
as provisional.

**Learning-rate stability.** An identical LR protocol produced an unstable configuration for one model or
another:
- the Riemannian model at 3e-3 under Haar: 8/25 runs had loss spikes, versus 0/25 at 1e-3;
- the 6D model at 3e-3 at D=10: one diverged run in five, under each of two priors.

Equal protocols do not guarantee equal treatment.

### 5.5 Multimodal targets

With three valid targets per start (Gauss σ=90, 5 seeds), the comparison splits by metric:

| metric | D=30: M1 / M2 | D=150: M1 / M2 |
|---|---|---|
| samples within 10° of a mode ↑ | **0.61** / 0.48 | **0.34** / 0.17 |
| starts covering all 3 modes ↑ | **0.95** / 0.86 | **0.52** / 0.39 |
| median nearest-mode error ↓ | **7.8** / 10.3 | **15.5** / 24.4 |
| W1 to the 3-mode target ↓ | 14.1 / **12.3** | 40.7 / **34.9** |

The Riemannian model places more samples on modes and covers more starts' modes. The 6D model matches the
target *distribution* better in W1. At D=30, the cut-locus tail (16% of sources) accounts for the W1
difference. At D=150, only 7% of sources lie beyond 150°, and **we could not explain the disagreement.** A
plausible but untested account is that W1 assigns every sample to a mode, so the Riemannian model's
off-mode samples cost more than its on-mode precision gains. No model represents the distribution well
(at most 61% of samples on a mode at D=30, and 34% at D=150). The quaternion and Euler models essentially
fail at D=150 (≤ 4% on a mode).

### 5.6 Is the surviving advantage asymptotic, and can a deployable prior deliver it?

**The Riemannian advantage under the cut-locus-free prior is a convergence-rate effect.** We extended the
D=150 comparison under the truncated prior to 160k and 320k steps (3 seeds per budget, a fresh run per
budget, cosine schedule stretched to the budget).

The pre-registered series could not answer the question. At its 20k-tuned LR (3e-3), the 6D model
**diverged in all three seeds at 320k** (errors of 103°, 27° and 103°) and in one of three at 160k. Every
loss spike in the long runs began with the LR above 43% of its peak; none occurred in 30 runs at ≤ 80k.

A matched-LR arm (both models at 1e-3, specified before it ran) was stable in all 15 runs:

| steps | Riemannian | 6D | gap (6D − Riem.) [95% CI] |
|---|---|---|---|
| 20k | 2.28 | 7.03 | +4.75 [3.87, 5.63] |
| 40k | 1.76 | 3.89 | +2.13 [1.82, 2.44] |
| 80k | 1.50 | 2.48 | +0.99 [0.70, 1.27] |
| 160k | 1.34 | 1.59 | +0.25 [0.05, 0.46] |
| 320k | 1.21 | 1.14 | −0.07 [−0.16, +0.02] |

- The per-doubling gap ratios fall (0.45, 0.46, 0.26, −0.28), and the models' error curves cross at
  **~275k steps (bootstrap 247k–302k)**.
- At 320k the Riemannian advantage is gone in every d0 band (a tie below 120°; the 6D model is ahead at
  120–150°). The Riemannian model's error still rises with d0 while the 6D model's is flat.
- A linear fit of the gap in log(steps) gives −1.15°/doubling [−1.42, −0.88] and an extrapolated −2.6° at
  1M steps. That is impossible when both errors are ~1.2°: such a fit cannot represent a gap that decays
  to zero, and the ratios and observed crossing are the relevant measurements.
- Training had not converged at 320k (loss still falling ~37% over the last quarter), so the asymptotic
  statement is an extrapolation.
- What the data supports: **the Riemannian model is about 4× more sample-efficient at matched LR** (2.28°
  at 20k vs ~80k steps for the 6D model to reach 2.48°), **and that efficiency advantage, not a lower
  error floor, is what our earlier budgets measured.**
- The 6D model's surviving runs at its tuned LR were *better* at 160k than the matched-LR runs (1.45° vs
  1.59°). Lowering its LR therefore delayed the crossing, and ~275k is an upper bound given stability.
- Matching the LR has its own cost. It slows the 6D model early (7.03° at 20k vs 3.57° at its tuned LR),
  so no single LR treats both models equally across the budget range.

Under the deployable Gaussian prior (σ=90°, protocol LRs), the 6D model's lead shrank every doubling
(−7.0° at 20k to −2.8° at 160k, and −0.6° [−2.0, +0.7] at 320k on its two non-diverged seeds). The rule
calls this undecidable at n=2. **The Riemannian model's cut-locus tail is itself mostly a rate effect:**
its error on sources within 15° of the antipode fell from 42.6° at 20k to 5.3° at 320k. Below 165° the two
models were tied in every d0 band at 320k. Given enough training, both models approach the same error
under either prior. What differs is how fast they get there.

**The d0 mechanism of a start-centered prior.** A Gaussian centered at the start pose puts sources about σ
from R0, while the target sits D from R0. We measured the resulting d0 distribution directly from the
sampler (3000 sources per cell):

| prior | D=30: share of d0 > 165° | D=90 | D=175 |
|---|---|---|---|
| Gauss σ=15 | 0.00 | 0.00 | **0.67** |
| Gauss σ=30 | 0.00 | 0.01 | **0.40** |
| Gauss σ=60 | 0.05 | 0.12 | 0.22 |
| Gauss σ=90 | 0.14 | 0.16 | 0.18 |
| Haar | 0.17 | 0.16 | 0.17 |

At small and moderate D, a tight prior removes the cut locus entirely. At D=175 it does the opposite:
tightening σ *concentrates* sources at the antipode, from 18% beyond 165° at σ=90 to 67% at σ=15. Our
pre-registered prediction said tightening σ would help "not at all" at D=175. The measurement says it
should hurt.

**Trained results** (both models, LR re-tuned per σ, 3 seeds, 20k; gap = 6D − Riemannian, positive =
Riemannian better):

| σ | D=30 | D=90 | D=175 |
|---|---|---|---|
| 15 | +16.2 [15.5, 16.9] | +11.1 [10.5, 11.7] | −21.4 [−23.4, −19.3] |
| 30 | +10.6 [9.8, 11.5] | +11.9 [11.4, 12.3] | −13.9 [−16.8, −11.0] |
| 60 | +6.1 [3.7, 8.5] | +8.0 [5.8, 10.3] | +0.5 [−3.2, +4.2] |
| 90 | −15.0 [−19.7, −10.4] | −6.6 [−11.0, −2.2] | −10.4 [−10.8, −10.0] |

- **σ shifts the displacement at which the Riemannian model stops winning, and the shift is non-monotone.**
  It is below 30° at σ=90, between 90° and 175° at σ=15–30, and about 175° (a tie) at σ=60.
- At D=175, the Riemannian model's error rises from 11.1° (σ=60) to 36.3° (σ=15), concentrated in sources
  beyond 165°, exactly as the d0 table predicts.
- **A confound we could not explain:** the 6D model also degrades under tight priors (17° at D=30 with
  σ=15, versus 1.2° under Haar). Its raw outputs drift far off the Stiefel manifold (residual 0.75 vs
  0.21), and no learning rate in a five-point grid fixes it.
- **The cleanest comparison therefore puts each model at its own best deployable prior** (6D: 5 seeds,
  night-1 protocol; Riemannian: 3 seeds, tuned at D=90). At D=30 and
  D=90, the Riemannian model at σ=30 (0.63°, 1.48°) beats the 6D model at its best prior, Haar (1.22°,
  2.38°). At D=175, the 6D model under Haar (3.69°) beats every Riemannian configuration we found (best:
  11.1°).
- These are 20k-step numbers. Phase A shows the gaps narrow with training.

## 6. Limitations

- **Synthetic task.** One task family: deterministic or three-mode targets, smooth paths, starts near
  identity. Real robot data has noisier, higher-entropy action distributions, and observations richer than
  a start pose. The crossover angles depend on our choice to center the data at the identity. With
  uniformly distributed starts, the discontinuous baselines would sit near their singularities at every D.
- **Nothing converged.** Every budget we trained (up to 320k steps) was still improving. All gaps
  shrank with training. Our magnitudes describe a fixed, finite budget, and any statement about the limit
  is an extrapolation. At 320k the loss was still falling ~37% per last quarter.
- **The cut-locus-free prior is an oracle.** The truncated prior needs the target. It isolates the
  mechanism; it is not a usable method. The deployable start-centered Gaussian gives the Riemannian model a win only at D ≤ 90°, and is confounded
  there by an unexplained degradation of the 6D model under tight priors (Section 5.6).
- **Small seed counts.** Headline numbers use 5 seeds; budget sweeps use 3; width and Phase-C width
  results use 1. Tuning used a single seed, and several LR picks were decided by margins within seed noise
  (e.g. 0.06° for the Riemannian model under Gauss σ=90 at D=90). The 5-seed CIs capture model-init and
  sampling variance, not dataset resampling: each D has one fixed dataset.
- **Width and architecture.** Width-2048 results are single-seed, and at D=30 one model failed to train
  at its width-1024 learning rate. Our backbone is a trajectory-level MLP. The specified 256-wide network failed
  outright, and
  rankings changed with width. Temporal convolutional or transformer backbones, common in practice, were
  not tested and might change the capacity story entirely.
- **One Riemannian design.** We tested geodesic conditional paths with an ambient-projection head and
  exp-map Euler integration. Alternatives might avoid the cut-locus failure: data-prediction
  parameterizations, cut-locus-aware paths, non-geodesic premetrics, higher-order integrators, or priors
  concentrated around the *target* side. We did not test them, so our results bound this design, not the
  family.
- **Euclidean details.** M2's raw outputs drift far off the Stiefel manifold (orthogonality residual
  0.05–0.5 before Gram-Schmidt), and we never re-project during integration. A Euclidean model that
  re-projects might behave differently.
- **Unexplained results:** the quaternion deficit at D ≥ 90; the W1-vs-mode-quality disagreement at
  D=150; a D=10 anomaly where the 6D model is worse than at D=30 under every prior.
- **Budget is confounded with time-at-peak-LR.** Our step sweeps stretch a cosine schedule while keeping
  the LR tuned at 20k. At 160k–320k steps this produced loss spikes in most runs of both models, every one
  starting with the LR above 43% of its peak, and none in 30 runs at ≤ 80k. For the 6D model at 3e-3 some
  runs diverged outright. Separating "more steps" from "longer at a high LR" would need a
  constant-LR-then-anneal design or per-budget re-tuning, which we did not run. We report a matched-LR
  sensitivity arm instead (Section 5.6).
- **Analysis drift.** The study evolved over three rounds, each responding to the last. We pre-registered
  selection rules and predictions before the runs they governed, and report the one case where we broke a
  pre-registered rule (a 0.5% tie that selected the prior known to handicap one model). The overall path
  was still adaptive, and some analyses (e.g. d0 stratification) were chosen after seeing the first
  results.

## 7. Recommendations for practitioners

A paper comparing Riemannian and Euclidean flow-matching (or diffusion) policies on rotations should report
at least the following for the comparison to be interpretable:

1. **The source distribution, and its distance to the cut locus.** Report the distribution of d0, the
   geodesic distance between source and target, and the share of sources beyond ~150°. Under a uniform
   prior about a third of sources are beyond 150°, which handicaps the geodesic model alone.
2. **Error stratified by d0**, not just the mean. Report the tail too (95th percentile and max, or the
   fraction above a threshold). Rare catastrophic errors decide most of these comparisons.
3. **A continuous Euclidean baseline** (6D or 9D with projection), not only quaternions or Euler angles.
   Against discontinuous baselines, a Riemannian win above some angular displacement is expected
   regardless of the manifold.
4. **The angular extent of the data relative to each representation's singularities**, e.g. the share of
   trajectories crossing a canonical-quaternion sign flip or an Euler gimbal region.
5. **Training budget and convergence evidence.** Report loss curves and results at several budgets. In our
   study, a 4.75° Riemannian advantage at 20k steps shrank every doubling and crossed zero at ~275k: a
   single-budget comparison measured sample efficiency, not final accuracy.
6. **Per-model hyperparameter search, and its stability.** Report the LR grid, each model's pick, and
   whether the pick produced loss spikes or divergence. An identical protocol can be unequal in effect.
7. **Per-budget LR validation in any step sweep.** Re-tune or re-validate the LR at every budget, and
   report spike and divergence counts per budget. An LR tuned at a short budget can be stable there and
   divergent at 16× the steps. In our sweep that happened to one model and not the other.
8. **Capacity sensitivity:** results at at least two widths, or a statement that rankings were not
   checked for width dependence.
9. **For multimodal tasks, both distributional (W1) and mode-quality metrics.** They can disagree.

## References

- Braun, M., Jaquier, N., Rozo, L., Asfour, T. (2024). Riemannian Flow Matching Policy for Robot Motion
  Learning. IROS 2024.
- Chen, R. T. Q., Lipman, Y. (2024). Flow Matching on General Geometries. ICLR 2024.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. (2023). Flow Matching for Generative
  Modeling. ICLR 2023.
- Liu, X., Gong, C., Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with
  Rectified Flow. ICLR 2023.
- Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H. (2019). On the Continuity of Rotation Representations in
  Neural Networks. CVPR 2019.
- (Protein-backbone SE(3) flow models, e.g. FrameFlow and FoldFlow, are mentioned for context only.)

_Citation details (author lists, venues; in particular the spelling of the RFMP first author) were written
from memory and must be checked against the papers before submission._
