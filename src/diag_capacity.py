"""Capacity diagnostic (not part of the headline sweep): does widening the MLP
fix the failure seen under the specified 4x256 architecture? Identical for all models."""
import sys
from common import MODELS
from run_experiments import run_pool

WIDTHS = [int(w) for w in sys.argv[1:]] or [1024, 2048]
if __name__ == "__main__":
    tasks = [dict(tag=f"diag_w{w}", model=m, D=90, seed=100, lr=1e-3, steps=10000, arch=dict(width=w, depth=4),
                  evals=[("val", [8, 32], 2, "eval_val.json")]) for w in sorted(WIDTHS, reverse=True) for m in MODELS]
    run_pool(tasks, "diag_capacity")
