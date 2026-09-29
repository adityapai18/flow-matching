# SE(3) flow matching: does the manifold actually buy anything?

Riemannian flow matching policies transport along geodesics on SE(3) instead of
Euclidean straight lines, and report smoother trajectories with fewer
integration steps than Euclidean baselines. But the Euclidean baselines in that
literature typically parameterize rotations with quaternions or Euler angles —
representations Zhou et al. (2019) showed are *discontinuous* and hard for
neural networks to learn, with errors up to 180° on some samples, against under
2° for the continuous 6D representation.

So the comparison that decides the question has not been run. This repo runs it.

**H1.** Euclidean flow matching with a 6D rotation representation matches
Riemannian flow matching, and the Riemannian advantage appears only above some
angular-displacement threshold.

**H2.** The published Riemannian-over-Euclidean gap is substantially an artifact
of the baseline's rotation parameterization, not of the manifold geometry.

Everything runs on CPU on a laptop. Four models, tiny MLPs, ~11 min per run.

---

## Status

**Night 1 complete** (140/140 runs, 0 failures, 5 seeds per cell). **Night 2 in
progress.**

Night 1's headline as originally written — *"Euclidean-6D beats Riemannian at
every angular displacement"* — **is confounded and should not be cited.** The
diagnostic that found the confound is in this repo and is described below. The
fair comparison is what night 2 is for.

What survives night 1 is reported under "Results that stand." What does not is
under "The confound."

---

## Setup

Four conditional flow matching models, identical architecture (4×1024 SiLU MLP),
optimizer, schedule, batch size, step budget, conditioning, and source samples.
Per seed, all four see the same minibatch order, the same `t` values, and the
same source draws — common random numbers. Translation is Euclidean and handled
identically in all four. Only the rotation state, interpolation path and ODE
step differ:

| | state | path | ODE step | projection |
|---|---|---|---|---|
| **M1** Riemannian | SO(3) matrix | geodesic (screw) | exp-map | on-manifold throughout |
| **M2** Euclid-6D | 6D (Zhou et al.) | straight line | Euler | Gram-Schmidt at end only |
| **M3** Euclid-quat | canonical quaternion | straight line | Euler | normalize at end only |
| **M4** Euclid-Euler | XYZ Euler | straight line | Euler | structural |

Learning rate was searched per model over the same grid, with a rule registered
*before* selection ran that extended the grid by one step on both sides if any
optimum landed on an edge. It did, for all four; the extension ran; all final
optima are interior. No other hyperparameter was searched for any model.

**Task.** For each angular displacement `D ∈ {10, 30, 60, 90, 120, 150, 175}°`,
2000 SE(3) trajectories of 32 waypoints, split 1600/200/200. Start poses sit
10–20° from identity; targets are exactly `D` away along a sampled axis. Path
types in equal thirds: geodesic screw motion, via-point detour, rotational
overshoot. Stored as float64 rotation matrices (max orthogonality residual
3.8e-15) so the dataset is representation-agnostic — storing quaternions would
bake the sign ambiguity into the data and confound the experiment.

Starts near identity is a deliberate choice: the discontinuities sit at fixed
places relative to identity (quaternion hemisphere boundary at 180°, XYZ gimbal
lock at ±90° middle angle), so `D` controls how close the data comes to them.
This is the *favorable* case for quaternions and Euler angles at small `D`. With
Haar-uniform starts, every `D` would straddle the singularities.

---

## Results that stand

### 1. The parameterization flips the conclusion (supports H2)

Against the baselines the literature actually uses, Riemannian wins above a
crossover angle:

| Riemannian vs | crossover | 95% CI |
|---|---|---|
| M3 quaternion | **54°** | 50–59° |
| M4 XYZ Euler | **94°** | 92–98° |

Above those angles M1 beats M3/M4 by 7–11°. At the same angles, M2 (6D) beats M1
by 6.6–7.7°. **A study comparing Riemannian flow matching to a quaternion or
Euler baseline at D ≳ 90° would report a clear Riemannian win. With 6D, the sign
reverses.**

(Crossovers from a seed-paired bootstrap, 10k resamples, linear interpolation
between grid levels. The CI reflects seed variance only — the 30° grid spacing
and the interpolation are the larger, unquantified uncertainty.)

### 2. Zhou's discontinuity mechanism is visible for Euler, not for quaternions

M4's max error jumps from 29° at D=60 to **172°** at D=90 — exactly where the
data first crosses Euler jumps (3.4% of trajectories at D=90, 0% at D=60). Its
*mean* at D=90 is 8.9°, better than M1's. The mean would put the Euler failure at
~120°; the max puts it at 90°. **Report worst case, not just means.**

For quaternions the mechanism is *not* visible as a step. M3 degrades smoothly
and does not break at D=150/175 where canonical-sign flips first appear (31% and
100% of trajectories). M3's errors are elevated at every D, including D ≤ 60
where no flips exist. Its deficit is not mainly the hemisphere discontinuity, and
the mechanism is currently unexplained.

### 3. The Riemannian model needs *more* integration steps, not fewer

At NFE=1, M1's mean error is 69–71° against M2's 10–11°. M2 plateaus by NFE 4–8;
M1 improves monotonically and is **still improving at NFE=64**. M1 ranks last at
NFE ≤ 4 for every D.

This runs against the "fewer integration steps" claim, with a caveat: nothing
here is converged (see Limitations), and M1 might close more of the gap at NFE
> 64, which was not measured.

### 4. Off-manifold drift does not predict endpoint error

M1 stays on SO(3) to machine precision, as designed. M2 drifts far off (raw
residual 0.05–0.17) and M3 further (final `|q|` off by 10–20%). But M2 is the
most accurate model after Gram-Schmidt. **The raw residual is a real
measurement and it is not the thing that matters** — a useful negative result
for anyone reaching for manifold constraints on the grounds of numerical
tidiness alone.

---

## The confound

Night 1's headline said M2 beats M1 everywhere. The tail diagnostic
(`results/diag_tail.json`) says why, and undoes it.

Across all 35,000 M1 test samples, endpoint error correlates 0.47 with `d0`, the
initial geodesic distance between the source sample and the target. For M2–M4
that correlation is ≤ 0.06.

| d0 | n | M1 mean err | M1 frac > 45° | M2 mean err |
|---|---|---|---|---|
| 0–90° | 6403 | **2.5°** | 0.000 | 3.0° |
| 90–120° | 7377 | 3.4° | 0.000 | 3.1° |
| 150–165° | 5591 | 10.3° | 0.000 | 3.1° |
| 165–175° | 3851 | 27.0° | 0.143 | 3.1° |
| 175–180° | 1979 | 91.7° | 0.803 | 3.1° |

**Every M1 error above 45°, at every D, comes from a source sample more than
165° from its target. Conditioned on d0 < 90°, M1 (2.5°) beats M2 (3.0°).**

Near 180°, `log(R₀ᵀR₁)` is discontinuous — the axis flips sign — so the
conditional velocity field has a discontinuity. The MLP smooths it toward zero
and those samples stall. Haar density ∝ (1 − cos θ) puts ~16% of source samples
past 165°. M2's straight line in 6D has no cut locus, so the same samples cost it
nothing.

This is a property of geodesic flow matching **with a uniform prior**, not of the
manifold. It also means the experimental design handicapped exactly one model:
equal architecture and equal budget are not equal treatment when the source
distribution is pathological for one path type and benign for another. The RFMP
paper itself notes performance improves with a Euclidean Gaussian base
distribution rather than a uniform one.

Night 2 re-runs the sweep under three priors — Haar, a concentrated Gaussian
around the conditioning start pose, and Haar truncated at d0 < 150° — applied to
all four models, with the LR search repeated per (model, prior).

---

## Deviations from the original spec

Recorded because two of them were found by inspecting a failure, not planned.

1. **Width 1024, not 256** — applied identically to all four. The specified
   4×256 MLP cannot fit this task for *any* parameterization (M1 sat at exactly
   Haar-chance, 126°). The velocity target requires a time-gated pass-through of
   a 192–384-dim state, and how badly a narrow net fails depends on each
   representation's state dimension, so a 256-wide comparison measures capacity,
   not geometry. **The model ranking flips between width 512 and 1024**, so any
   narrower network would have decided the answer by capacity.
2. **M1 output head: tangent projection of an ambient output.** The first M1
   emitted the body-frame tangent vector directly, a target that forces the
   network to learn multiplication by its own input. It stayed at chance at
   every width. The final M1 emits an ambient 3×3 matrix and projects
   analytically onto the tangent space — the standard parameterization for an
   embedded manifold. Geometry unchanged; oracle tests pass for both heads.
3. **20k steps, set by wall clock, not convergence.** Loss was still decreasing.
4. **M1 training instability.** At the protocol-selected LR (3e-3), 10 of 35 M1
   runs had a loss spike; M2–M4 had 0 in 105 runs. M1 at 1e-3 was only 0.6° worse
   in tuning. A protocol identical in *form* produced an unstable choice for one
   model — being re-checked in night 2.

---

## Limitations

- **Synthetic task, and the target is a deterministic function of the start.**
  Flow matching's claimed advantage is modeling multimodal action distributions;
  on a deterministic target this is regression with extra steps. A multimodal
  variant is in night 2.
- **Not converged, and the ranking is not budget-stable.** It flipped between the
  10k diagnostic and the 20k runs, and between width 512 and 1024.
- **Seed variance covers model init plus training and eval noise, not dataset
  resampling.** One fixed dataset per D.
- **Rotation and translation share one network and one loss**, so a hard rotation
  target degrades translation too — a real coupling confound, small for M2/M3 and
  larger for M1/M4.
- No real robot data yet.

---

## Geometry primitives

`src/geometry.py`, 47 tests in `tests/`. scipy is used as an independent oracle
for the quaternion, Euler and exp conventions. Two tests failed on the first
run; both were diagnosed before anything else ran:

- A near-π tolerance was miscalibrated — the generic `log` branch cannot deliver
  1e-12 just below the mandated `π − 1e-4` boundary. Test corrected; primitive
  unchanged.
- **A real weakness:** single-pass Gram-Schmidt left a 1.5e-13 orthogonality
  residual when the two 6D columns were nearly parallel. `sixd_to_matrix` now
  does a second pass ("twice is enough"); residual is at 1e-16 for arbitrary
  input.

`geodesic_distance(R, R)` is *exactly* 0.0 for all 10,000 test rotations, because
`R₁ᵀR₂` is formed with a fixed symmetric summation order.

**Quaternion convention:** scalar-first `(w,x,y,z)`, canonical representative has
`w > 0`, ties broken on the first nonzero of `(x,y,z)`. Applied per rotation
inside `matrix_to_quat`, before any Euclidean operation.

---

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install torch numpy scipy matplotlib pytest

.venv/bin/python -m pytest tests -q          # 47 geometry + flow oracle tests
cd src
../.venv/bin/python make_data.py             # -> results/synthetic/
../.venv/bin/python run_experiments.py tune  # LR search + edge extension
../.venv/bin/python run_experiments.py main  # 4 models x 7 D x 5 seeds
../.venv/bin/python analyze.py               # tables, stats, figures
../.venv/bin/python crossover.py             # crossover angles
../.venv/bin/python diag_tail.py             # cut-locus diagnostic
../.venv/bin/python timing.py                # clean wall-clock (run alone)
```

Every step is resumable — finished runs skip, partial training resumes from
`ckpt.pt`, evaluation resumes per NFE. All numbers in `results/summary.json`,
tables in `results/summary_tables.md`, log in `results/run.log`.

CPU float64 throughout the geometry path. Rotation composition and log/exp maps
accumulate error badly in float16, and MPS linalg is unreliable.

---

## Open questions

1. **Does M2 still beat M1 under a prior that does not handicap M1?** The
   d0-stratified numbers say probably not. Night 2 settles it.
2. **Does the ranking survive a converged budget?** Two independent signals say
   the current ordering may be a budget artifact.
3. **Does multimodality change the answer?** This is where Riemannian flow
   matching has its strongest theoretical case, and the current task does not
   test it.
4. **Why is M3 (quaternion) bad everywhere?** Its deficit does not follow Zhou's
   predicted step at the hemisphere boundary. Unexplained.
5. **Why is D=10 harder than D=30** for M1, M2 and M3? Possibly a near-degenerate
   target distribution at that scale — untested.

## References

- Zhou et al., *On the Continuity of Rotation Representations in Neural
  Networks*, CVPR 2019. [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)
- Braun et al., *Riemannian Flow Matching Policy for Robot Motion Learning*,
  IROS 2024. [arXiv:2403.10672](https://arxiv.org/abs/2403.10672)
- Chen & Lipman, *Flow Matching on General Geometries*, ICLR 2024.