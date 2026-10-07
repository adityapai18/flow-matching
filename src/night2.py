"""Night-2 orchestrator: every phase is derived from disk state, so this script is resumable
and can be killed/restarted at any point. Tasks become ready as their dependencies (pilot
selections, LR searches) complete; one CPU pool runs them, highest priority first.

  python night2.py run        schedule everything that is ready, until nothing is left
  python night2.py status     print what is done / pending

DEVICE: everything trains on CPU (8 workers, as night 1). MPS was benchmarked first: the machine
saturates at ~245-270 aggregate steps/s in every CPU/MPS worker mix tried (the per-step float64
geometry and Python overhead are CPU-bound), so MPS added no throughput, only a device confound.
CPU training is deterministic per seed, so B1's 20k point IS the Phase-A main run for seeds 0-2.
Evaluation (geometry AND network forward) is CPU float64/float32, as in night 1.

PRE-REGISTERED RULES (fixed before any night-2 result existed; also in results/n2/preregistration.json):
  sigma pilot : D=90, tuning seed 100, LR 1e-3 for all models (night-1 diagnostic LR), 20k steps.
                sigma* = argmin over {30,60,90} of the GEOMETRIC MEAN over the 4 models of val mean
                endpoint error at NFE 32 (scale-free, so no single model's error dominates).
  LR search   : night-1 protocol verbatim, per (model, prior): base grid {3e-4,1e-3,3e-3}, D in
                {30,90,150}, seed 100, criterion = val mean endpoint error at NFE 32 averaged over D;
                if ANY model's pick is on a grid edge, extend that side one step for ALL models
                (1e-2 top, 1e-4 bottom), at most once per side.
  best prior  : among the priors legal at inference (P1 Haar, P2 Gauss at sigma*; P3 needs the target):
                argmin of the geometric mean over models of each model's selected-LR search score.
                Uses only tuning-seed validation runs, never test runs.
  Phase C LR  : same rule on the multimodal data at the best prior, D in {30,150}, criterion = val
                mean W1 (geodesic, deg) at NFE 32 averaged over D.
"""
import concurrent.futures as cf
import itertools
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import torch

from common import EVAL_SEEDS, MODELS, NFES, RESULTS, RUNS_DIR, get_logger, read_json, write_json

N2 = RESULTS / "n2"
N_CPU = int(os.environ.get("N2_CPU_WORKERS", 8))

D_N2 = [10, 30, 90, 150, 175]            # reduced grid: D=60 and D=120 dropped to afford 3 priors
TUNE_SEED, TUNE_D, SELECT_NFE = 100, [30, 90, 150], "32"
BASE_GRID, LR_MIN, LR_MAX = [3e-4, 1e-3, 3e-3], 1e-4, 1e-2
STEPS = 20000
SIGMAS = [30, 60, 90]
P3 = {"kind": "haar_trunc", "max_deg": 150}
HAAR = {"kind": "haar"}
B1_D, B1_SEEDS, B1_STEPS = [30, 150], [0, 1, 2], [40000, 80000]
B1_MODELS = ["M1_riemannian", "M2_euclid6d"]   # the B1 question is the M1-vs-M2 sign; halves a ~5 h phase
MM_D, MM_K, MM_NFES, MM_TUNE_NFES = [30, 150], 30, [8, 32, 64], [32]
VAL_EVAL = ("uni", "val", NFES, 2, "eval_val.json", None)
TEST_EVAL = ("uni", "test", NFES, 5, "eval_test.json", 64)


def gauss(s):
    return {"kind": "gauss", "sigma_deg": s}


def pname(prior):
    import fm
    return fm.prior_name(prior)


# ----------------------------------------------------------------------------- tasks

def task(tag, model, D, seed, lr, steps, device, prio, evals, prior=HAAR, dataset=None, arch=None):
    return dict(tag=tag, model=model, D=D, seed=seed, lr=lr, steps=steps, device=device, prio=prio,
                evals=evals, prior=prior, dataset=dataset, arch=arch or {},
                key=f"{tag}/{model}_D{D}_s{seed}")


def run_dir(t):
    return RUNS_DIR / t["tag"] / f'{t["model"]}_D{t["D"]}_s{t["seed"]}'


def task_done(t):
    rd = run_dir(t)
    info = read_json(rd / "train.json")
    if not (info and info.get("done") and (rd / "model.pt").exists()):
        return False
    for kind, split, nfes, k, name, smp in t["evals"]:
        r = read_json(rd / name)
        if not r or any(str(n) not in r["by_nfe"] for n in nfes):
            return False
        if smp and not (rd / name.replace(".json", "_samples.npz")).exists():
            return False
    return True


def run_task(t):
    torch.set_num_threads(1)
    import fm
    log = get_logger()
    rd = run_dir(t)
    try:
        fm.train_one(t["model"], t["D"], t["seed"], t["lr"], t["steps"], rd, device=t["device"],
                     prior=t["prior"], dataset=t["dataset"], **t["arch"])
        for kind, split, nfes, k, name, smp in t["evals"]:
            if kind == "uni":
                fm.evaluate(t["model"], t["D"], t["seed"], rd, split, nfes, k, name,
                            prior=t["prior"], dataset=t["dataset"], samples_nfe=smp)
            else:
                fm.evaluate_mm(t["model"], t["D"], t["seed"], rd, split, nfes, k, name,
                               prior=t["prior"], samples_nfe=smp)
        (rd / "ERROR.txt").unlink(missing_ok=True)
        return t["key"], "ok"
    except Exception as e:
        log.error(f"[n2] {t['key']} FAILED: {e!r}\n{traceback.format_exc()}")
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "ERROR.txt").write_text(traceback.format_exc())
        return t["key"], f"error: {e!r}"


# ----------------------------------------------------------------------------- selection rules

def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def score_uni(rd):
    r = read_json(rd / "eval_val.json")
    return r["by_nfe"][SELECT_NFE]["geo_err_deg"]["mean"] if r and SELECT_NFE in r["by_nfe"] else None


def score_mm(rd):
    r = read_json(rd / "eval_mm_val.json")
    return r["by_nfe"][SELECT_NFE]["w1_deg"]["mean"] if r and SELECT_NFE in r["by_nfe"] else None


def lr_search(name, make_tasks, tune_d, scorer, out_path):
    """Night-1 LR protocol (base grid + pre-registered one-step edge extension for ALL models).
    make_tasks(lr) -> tasks for every model x tune_d at that LR.
    Returns (tasks still needed, final selection dict or None)."""
    final = read_json(out_path)
    if final:
        return [], final
    needed = list(BASE_GRID)
    while True:
        missing = [t for lr in needed for t in make_tasks(lr) if not task_done(t)]
        if missing:
            return missing, None
        grid = sorted(needed)
        table = {m: {f"{lr:g}": [scorer(run_dir(t)) for t in make_tasks(lr) if t["model"] == m] for lr in grid}
                 for m in MODELS}
        mean = {m: {lr: sum(v) / len(v) for lr, v in table[m].items()} for m in MODELS}
        chosen = {m: float(min(grid, key=lambda lr: mean[m][f"{lr:g}"])) for m in MODELS}
        ext = []
        if any(v == max(grid) for v in chosen.values()) and max(grid) < LR_MAX:
            ext.append(LR_MAX if max(grid) == 3e-3 else max(grid) * 10 / 3)
        if any(v == min(grid) for v in chosen.values()) and min(grid) > LR_MIN:
            ext.append(LR_MIN if min(grid) == 3e-4 else min(grid) * 3 / 10)
        if not ext:
            out = dict(search=name, grid=grid, base_grid=BASE_GRID, tune_d=tune_d, seed=TUNE_SEED,
                       criterion_nfe=SELECT_NFE, per_D=table, mean=mean, chosen_lr=chosen,
                       score={m: mean[m][f"{chosen[m]:g}"] for m in MODELS},
                       edge_extended=[lr for lr in grid if lr not in BASE_GRID])
            write_json(out_path, out)
            get_logger().info(f"[n2] LR search {name} final: {chosen}  grid {grid}")
            return [], out
        get_logger().info(f"[n2] LR search {name}: pick on grid edge {chosen}; extending for ALL models {ext}")
        needed += [lr for lr in ext if lr not in needed]


def p1_hparams():
    hp = read_json(RESULTS / "hparams.json")
    return dict(search="haar (night 1, reused)", chosen_lr=hp["chosen_lr"],
                score={m: hp["table"][m][f'{hp["chosen_lr"][m]:g}']["mean"] for m in MODELS})


def pilot_tasks():
    return [task(f"n2_pilot_gauss{s}", m, 90, TUNE_SEED, 1e-3, STEPS, "cpu", 0,
                 [("uni", "val", [8, 32], 2, "eval_val.json", None)], prior=gauss(s))
            for s in SIGMAS for m in MODELS]


def sigma_selection():
    out = read_json(N2 / "sigma_select.json")
    if out:
        return [], out
    ts = pilot_tasks()
    missing = [t for t in ts if not task_done(t)]
    if missing:
        return missing, None
    err = {s: {t["model"]: score_uni(run_dir(t)) for t in ts if t["prior"]["sigma_deg"] == s} for s in SIGMAS}
    gm = {s: geomean(list(err[s].values())) for s in SIGMAS}
    best = min(SIGMAS, key=lambda s: gm[s])
    out = dict(rule="argmin_sigma geomean_models(val mean err, NFE 32), D=90, seed 100, LR 1e-3, 20k",
               val_err=err, geomean=gm, sigma_star=best)
    write_json(N2 / "sigma_select.json", out)
    get_logger().info(f"[n2] sigma pilot: geomean {gm} -> sigma* = {best}")
    return [], out


def tune_tasks(prior, device, prio):
    tag = f"n2_hp_{pname(prior)}"
    return lambda lr: [task(f"{tag}_lr{lr:g}", m, D, TUNE_SEED, lr, STEPS, device, prio, [VAL_EVAL], prior=prior)
                       for D in TUNE_D for m in MODELS]


def main_tasks(prior, hp, device, prio, tag=None):
    tag = tag or f"n2_main_{pname(prior)}"
    return [task(tag, m, D, s, hp["chosen_lr"][m], STEPS, device, prio + 0.01 * s, [TEST_EVAL], prior=prior)
            for s in EVAL_SEEDS for D in D_N2 for m in MODELS]


def mm_tune_tasks(prior, device, prio):
    tag = f"n2_mmhp_{pname(prior)}"
    return lambda lr: [task(f"{tag}_lr{lr:g}", m, D, TUNE_SEED, lr, STEPS, device, prio,
                            [("mm", "val", MM_TUNE_NFES, MM_K, "eval_mm_val.json", None)],
                            prior=prior, dataset=f"MM_D{D}")
                       for D in MM_D for m in MODELS]


# ----------------------------------------------------------------------------- dependency graph

def ready_tasks():
    """Everything runnable now, given what is on disk. Priority: lower runs first."""
    log = get_logger()
    out = []
    # --- Phase A: P1 (Haar) main runs are cached from night 1; add per-sample d0 files (cheap)
    out += [dict(t, prio=2) for t in main_tasks(HAAR, p1_hparams(), "cpu", 2, tag="main")]
    # --- Phase A: sigma pilot -> P2 LR search -> P2 main (CPU)
    miss, sig = sigma_selection()
    out += miss
    hp = {"haar": p1_hparams()}
    if sig:
        p2 = gauss(sig["sigma_star"])
        miss, hp2 = lr_search(pname(p2), tune_tasks(p2, "cpu", 1), TUNE_D, score_uni,
                              N2 / f"hparams_{pname(p2)}.json")
        out += miss
        if hp2:
            hp[pname(p2)] = hp2
            out += main_tasks(p2, hp2, "cpu", 3)
    # --- Phase A: P3 LR search -> P3 main
    miss, hp3 = lr_search(pname(P3), tune_tasks(P3, "cpu", 1), TUNE_D, score_uni, N2 / "hparams_trunc150.json")
    out += miss
    if hp3:
        out += main_tasks(P3, hp3, "cpu", 3)
    # --- Phase B2: width 2048 capacity diagnostic, night-1 protocol (Haar, D=90, seed 100, LR 1e-3, 10k)
    out += [task("n2_diag_w2048", m, 90, TUNE_SEED, 1e-3, 10000, "cpu", 4,
                 [("uni", "val", [8, 32], 2, "eval_val.json", None)], arch=dict(width=2048, depth=4))
            for m in MODELS]
    # --- Phase B3: M1 at LR 1e-3 under Haar, same grid/seeds as the P1 main runs (which used 3e-3)
    out += [task("n2_b3_M1_lr0.001", "M1_riemannian", D, s, 1e-3, STEPS, "cpu", 4.5, [TEST_EVAL])
            for s in EVAL_SEEDS for D in D_N2]
    # --- best prior -> Phase B1 and Phase C
    best = read_json(N2 / "best_prior.json")
    if not best and len(hp) == 2:
        gm = {k: geomean(list(v["score"].values())) for k, v in hp.items()}
        name = min(gm, key=gm.get)
        best = dict(rule="argmin over legal priors of geomean_models(selected-LR val score)", geomean=gm,
                    best=name, prior=HAAR if name == "haar" else gauss(sig["sigma_star"]),
                    chosen_lr=hp[name]["chosen_lr"])
        write_json(N2 / "best_prior.json", best)
        log.info(f"[n2] best prior: {gm} -> {name}")
    if best and "used" not in best and best["best"] == "haar" and "gauss90" in hp:
        # DEVIATION from the pre-registered rule (night 2, session 2): the rule picked Haar on a 0.5%
        # geomean margin (8.107 vs 8.151, one tuning seed = a tie). Haar is the prior Phase A exists to
        # remove: it produces M1's cut-locus tail (M1 tuning score 12.9 deg under Haar vs 1.45 under
        # trunc150, while M2 moves 2.1 -> 2.6). Running B1/C under it would be equal in form, unequal
        # in effect. Tie broken toward the legal prior WITHOUT a known single-model pathology.
        best.update(prereg_choice="haar", used="gauss90", prior=gauss(sig["sigma_star"]),
                    chosen_lr=hp["gauss90"]["chosen_lr"],
                    deviation="tie (0.5%) broken away from Haar: Haar handicaps M1 only (cut locus)")
        write_json(N2 / "best_prior.json", best)
    if best:
        bp, bname = best["prior"], best.get("used", best["best"])
        out += [task(f"n2_b1_{bname}_steps{S}", m, D, s, best["chosen_lr"][m], S, "cpu",
                     5 + B1_STEPS.index(S) * 0.1 + 0.01 * s, [TEST_EVAL], prior=bp)
                for S in B1_STEPS for s in B1_SEEDS for D in B1_D for m in B1_MODELS]
        miss, hpc = lr_search(f"mm_{bname}", mm_tune_tasks(bp, "cpu", 6), MM_D, score_mm,
                              N2 / f"hparams_mm_{bname}.json")
        out += miss
        if hpc:
            out += [task(f"n2_mm_{bname}", m, D, s, hpc["chosen_lr"][m], STEPS, "cpu", 7 + 0.01 * s,
                         [("mm", "test", MM_NFES, MM_K, "eval_mm_test.json", 64)], prior=bp, dataset=f"MM_D{D}")
                    for s in EVAL_SEEDS for D in MM_D for m in MODELS]
    # --- B1 extension (session 2, added after B1 under gauss90 showed M1's d0<120 advantage was not
    #     budget-stable): does the P3 (cut-locus-free) M1 win survive 4x the steps? Oracle prior, M1/M2,
    #     80k only, own (P3) selected LRs. Priority below Phase C.
    if hp3:
        out += [task("n2_b1x_trunc150_steps80000", m, D, s, hp3["chosen_lr"][m], 80000, "cpu", 7.5 + 0.01 * s,
                     [TEST_EVAL], prior=P3)
                for s in B1_SEEDS for D in B1_D for m in B1_MODELS]
    # --- sensitivity arm (NOT part of the pre-registered selection; lowest priority, leftover compute only):
    #     the sigma pilot showed sigma=30 favors M1 (1.6 vs M2 13.8 deg, untuned) while the pre-registered
    #     rule picked sigma*=90. Run sigma=30 through the identical LR protocol + main grid so the headline
    #     can say whether the ranking depends on sigma after tuning.
    if sig and sig["sigma_star"] != 30:
        p30 = gauss(30)
        miss, hp30 = lr_search(pname(p30), tune_tasks(p30, "cpu", 8), TUNE_D, score_uni,
                               N2 / f"hparams_{pname(p30)}.json")
        out += miss
        if hp30:
            out += main_tasks(p30, hp30, "cpu", 9)
    return out


# ----------------------------------------------------------------------------- scheduler

def preregister():
    path = N2 / "preregistration.json"
    if not path.exists():
        write_json(path, dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), rules=__doc__.split("PRE-REGISTERED")[1]))


def run():
    log = get_logger()
    preregister()
    ctx = mp.get_context("spawn")
    pools = {"cpu": cf.ProcessPoolExecutor(N_CPU, mp_context=ctx, max_tasks_per_child=1)}
    cap = {"cpu": N_CPU}
    running, tried = {}, set()
    t0 = time.time()
    log.info(f"[n2] scheduler start: {N_CPU} cpu workers")
    while True:
        ready = [t for t in ready_tasks() if t["key"] not in tried and not task_done(t)]
        for dev in pools:
            busy = sum(1 for t in running.values() if t["device"] == dev)
            for t in sorted((t for t in ready if t["device"] == dev), key=lambda t: t["prio"])[:cap[dev] - busy]:
                running[pools[dev].submit(run_task, t)] = t
                tried.add(t["key"])
        if not running:
            break
        done, _ = cf.wait(list(running), return_when=cf.FIRST_COMPLETED)
        for f in done:
            t = running.pop(f)
            try:
                key, status = f.result()
            except Exception as e:                          # worker died (OOM, MPS crash...)
                key, status = t["key"], f"worker crash: {e!r}"
                log.error(f"[n2] {key} {status}")
            log.info(f"[n2] {t['device']} {key}: {status}  ({time.time() - t0:.0f}s, {len(running)} running)")
    log.info("[n2] scheduler: nothing left to run")


def status():
    ts = ready_tasks()
    from collections import Counter
    c = Counter((t["tag"], task_done(t)) for t in ts)
    for tag in sorted({t["tag"] for t in ts}):
        print(f"{tag:40s} done {c[(tag, True)]:4d}  pending {c[(tag, False)]:4d}")
    for f in sorted(N2.glob("*.json")):
        print("  ", f.name)


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1]]()
