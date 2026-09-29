# Project: SE(3) flow matching — does the manifold actually buy anything?

Testing whether Riemannian flow matching on SE(3) outperforms Euclidean flow
matching with a properly-chosen rotation representation (6D, Zhou et al. 2019),
and where the crossover is as a function of angular displacement.

## Hardware — non-negotiable
- 16GB Apple Silicon Mac, MPS backend. No CUDA, no cloud, no API inference.
- PyTorch only. No JAX, no MuJoCo-MJX, no CUDA-dependent packages.
- Models here are TINY (3-5 layer MLPs, <1M params). If something needs a GPU
  cluster, it is out of scope — say so rather than shrinking it silently.
- All geometry math in float64 on CPU. Rotation composition and log/exp maps
  accumulate error badly in float16, and MPS linalg is unreliable.
- Training may use MPS; evaluation geometry must be CPU float64.

## Numerical rules — the whole project depends on these
- SO(3) log map is singular at theta=0 and theta=pi. Always use the
  small-angle Taylor branch below 1e-4 and the near-pi branch above pi-1e-4.
  Write these branches explicitly and unit-test them.
- Never compare rotations with elementwise L2. Use geodesic distance:
  d(R1,R2) = ||log(R1^T R2)||.
- Quaternions have a sign ambiguity (q and -q are the same rotation). Resolve
  it consistently before ANY Euclidean operation, and document the convention.
- Check orthogonality residual ||R^T R - I|| BEFORE any re-projection. That
  residual is a primary measurement, not an intermediate to clean up.

## Engineering rules
- Every script resumable from disk state. Skip work already cached.
- Fix and log the random seed. Every headline number needs >=5 seeds with
  mean and standard deviation — single-seed results are not reportable.
- Write metrics to JSON incrementally, never only at the end.
- Log to results/run.log with timestamps.
- On exception: log it, save partial state, continue to the next condition.

## Standing prohibitions
- Never tune a baseline worse than the proposed method. The Euclidean-6D
  baseline gets the SAME architecture, parameter count, training budget, and
  hyperparameter search as the Riemannian model. Unequal effort invalidates
  the entire result.
- Never add features not asked for.
- Negative results get written up as negative results.
