"""Crossover angle where a Euclidean baseline's mean endpoint error overtakes M1's (NFE=64).
Seed-paired bootstrap (10k resamples of the 5 seeds); linear interpolation of the paired
gap (baseline - M1) between grid levels at its first sign change from negative to positive."""
import numpy as np

from analyze import arr, load
from common import D_LEVELS, MODELS, RESULTS, write_json


def first_crossing(gap):
    for j in range(len(gap) - 1):
        if gap[j] < 0 <= gap[j + 1]:
            return D_LEVELS[j] + (D_LEVELS[j + 1] - D_LEVELS[j]) * (-gap[j]) / (gap[j + 1] - gap[j])
    return np.nan


def main():
    ev, _, _ = load()
    A = arr(ev, "geo_err_deg.mean")
    rng = np.random.default_rng(0)
    out = {}
    for i, m in enumerate(MODELS[1:], start=1):
        gap = A[i] - A[0]                                  # [D, seed], paired by seed
        point = first_crossing(gap.mean(-1))
        boots = [first_crossing(gap[:, rng.integers(0, 5, 5)].mean(-1)) for _ in range(10000)]
        boots = np.array(boots)
        ok = ~np.isnan(boots)
        out[m] = dict(point_deg=None if np.isnan(point) else float(point), frac_boot_with_crossing=float(ok.mean()),
                      ci95_deg=[float(np.percentile(boots[ok], 2.5)), float(np.percentile(boots[ok], 97.5))] if ok.any() else None)
        print(m, out[m])
    write_json(RESULTS / "crossover.json", out)


if __name__ == "__main__":
    main()
