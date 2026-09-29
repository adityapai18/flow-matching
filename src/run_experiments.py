"""Orchestrates pilot (step budget), hyperparameter search, and the main sweep.

Usage:
  python run_experiments.py pilot     # step-budget pilot, identical for all models
  python run_experiments.py hpsearch  # LR grid, identical for all models
  python run_experiments.py select    # pick per-model LR from hpsearch -> results/hparams.json
  python run_experiments.py tune      # hpsearch + select + pre-registered grid-edge extension
  python run_experiments.py main      # 4 models x 7 D x 5 seeds, test-set eval over NFE sweep

Every task is resumable (training checkpoints + per-NFE JSON). A failing task is
logged, its partial state kept, and the pool moves on to the next condition.
"""
import itertools
import multiprocessing as mp
import sys
import time
import traceback

import torch

from common import D_LEVELS, EVAL_SEEDS, MODELS, NFES, RESULTS, RUNS_DIR, get_logger, read_json, write_json

N_WORKERS = int(__import__("os").environ.get("N_WORKERS", 8))
TUNE_SEED = 100                      # disjoint from the evaluation seeds 0-4
TUNE_D = [30, 90, 150]
LR_GRID = [3e-4, 1e-3, 3e-3]
PILOT_STEPS = [5000, 15000, 30000]
PILOT_D = [90, 150]
PILOT_LR = 1e-3
SELECT_NFE = "32"
BUDGET_PATH = RESULTS / "budget.json"
HPARAMS_PATH = RESULTS / "hparams.json"


def run_task(task):
    torch.set_num_threads(1)
    import fm                                           # import in worker
    log = get_logger()
    run_dir = RUNS_DIR / task["tag"] / f'{task["model"]}_D{task["D"]}_s{task["seed"]}'
    try:
        fm.train_one(task["model"], task["D"], task["seed"], task["lr"], task["steps"], run_dir,
                     **task.get("arch", {}))
        for split, nfes, k, name in task["evals"]:
            fm.evaluate(task["model"], task["D"], task["seed"], run_dir, split, nfes, k, name)
        (run_dir / "ERROR.txt").unlink(missing_ok=True)
        return task["tag"], run_dir.name, "ok"
    except Exception as e:
        log.error(f"[task] {task['tag']}/{run_dir.name} FAILED: {e!r}\n{traceback.format_exc()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "ERROR.txt").write_text(traceback.format_exc())
        return task["tag"], run_dir.name, f"error: {e!r}"


def run_pool(tasks, label):
    log = get_logger()
    log.info(f"[{label}] {len(tasks)} tasks on {N_WORKERS} workers")
    t0 = time.time()
    status = []
    with mp.get_context("spawn").Pool(N_WORKERS, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
            status.append(r)
            log.info(f"[{label}] {i + 1}/{len(tasks)} {r[0]}/{r[1]}: {r[2]}  ({time.time() - t0:.0f}s)")
    bad = [s for s in status if s[2] != "ok"]
    log.info(f"[{label}] finished: {len(status) - len(bad)} ok, {len(bad)} failed")
    return status


def pilot():
    tasks = [dict(tag=f"pilot_steps{s}", model=m, D=D, seed=TUNE_SEED, lr=PILOT_LR, steps=s,
                  evals=[("val", [8, 32], 2, "eval_val.json")])
             for s in sorted(PILOT_STEPS, reverse=True) for D in PILOT_D for m in MODELS]
    run_pool(tasks, "pilot")


def hpsearch(lrs=None):
    steps = read_json(BUDGET_PATH)["steps"]
    tasks = [dict(tag=f"hp_lr{lr:g}", model=m, D=D, seed=TUNE_SEED, lr=lr, steps=steps,
                  evals=[("val", NFES, 2, "eval_val.json")])
             for D, lr, m in itertools.product(TUNE_D, lrs or LR_GRID, MODELS)]
    run_pool(tasks, "hpsearch")


def grid_done():
    """LRs whose runs are complete for every model and tuning D."""
    out = []
    for d in sorted(RUNS_DIR.glob("hp_lr*")):
        lr = float(d.name[len("hp_lr"):])
        ok = all((r := read_json(d / f"{m}_D{D}_s{TUNE_SEED}" / "eval_val.json")) and SELECT_NFE in r["by_nfe"]
                 for m in MODELS for D in TUNE_D)
        if ok:
            out.append(lr)
    return sorted(out)


def tune():
    """Pre-registered rule, identical for all models: run the base grid; if ANY model's
    selected LR is at a grid edge, extend the grid one step (x10/3 spacing) on that side for
    ALL models, and re-select. At most one extension per side."""
    log = get_logger()
    hpsearch()
    for _ in range(2):
        chosen = select()
        grid = grid_done()
        ext = []
        if any(v == max(grid) for v in chosen.values()) and max(grid) < 1e-2:
            ext.append(1e-2 if max(grid) == 3e-3 else max(grid) * 10 / 3)
        if any(v == min(grid) for v in chosen.values()) and min(grid) > 1e-4:
            ext.append(1e-4 if min(grid) == 3e-4 else min(grid) * 3 / 10)
        if not ext:
            return
        log.info(f"[tune] selected LR on grid edge {chosen}; extending grid for ALL models with {ext}")
        hpsearch(ext)
    select()


def select():
    """Identical protocol for every model: minimize mean val endpoint geodesic
    error (deg) at NFE=32, averaged over the tuning D levels."""
    log = get_logger()
    table, chosen = {}, {}
    grid = grid_done()
    for m in MODELS:
        table[m] = {}
        for lr in grid:
            vals = []
            for D in TUNE_D:
                r = read_json(RUNS_DIR / f"hp_lr{lr:g}" / f"{m}_D{D}_s{TUNE_SEED}" / "eval_val.json")
                vals.append(r["by_nfe"][SELECT_NFE]["geo_err_deg"]["mean"] if r and SELECT_NFE in r["by_nfe"]
                            else float("inf"))
            table[m][f"{lr:g}"] = dict(per_D=dict(zip(map(str, TUNE_D), vals)), mean=sum(vals) / len(vals))
        chosen[m] = float(min(grid, key=lambda lr: table[m][f"{lr:g}"]["mean"]))
    out = dict(protocol=f"per-model LR from grid {grid} (base {LR_GRID}, extended one step for ALL models "
                        f"if any model's pick was on a grid edge); criterion = mean val endpoint geodesic error at "
                        f"NFE={SELECT_NFE}, averaged over D in {TUNE_D}; tuning seed {TUNE_SEED}; "
                        f"identical grid/budget/criterion for all four models",
               steps=read_json(BUDGET_PATH)["steps"], table=table, chosen_lr=chosen)
    write_json(HPARAMS_PATH, out)
    log.info(f"[select] grid {grid}; chosen LRs: {chosen}")
    return chosen


def main_sweep():
    hp = read_json(HPARAMS_PATH)
    steps = hp["steps"]
    # interleave models so contention (P vs E cores) is spread evenly across models
    tasks = [dict(tag="main", model=m, D=D, seed=s, lr=hp["chosen_lr"][m], steps=steps,
                  evals=[("test", NFES, 5, "eval_test.json")])
             for s in EVAL_SEEDS for D in D_LEVELS for m in MODELS]
    run_pool(tasks, "main")


if __name__ == "__main__":
    {"pilot": pilot, "hpsearch": hpsearch, "tune": tune, "select": select, "main": main_sweep}[sys.argv[1]]()
