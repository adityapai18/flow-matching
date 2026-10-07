"""Night-3 orchestrator. Reuses night-2's task runner, scheduler, evaluation and checkpointing unchanged;
only the task list differs. Resumable from disk state like night 2.

  python night3.py run       schedule everything ready, longest jobs first
  python night3.py status

Phase A  M1/M2, D=150, priors P3 (trunc150) and P2 (gauss90), steps {160k, 320k}, seeds 0-2, night-2
         prior-selected LRs. Trained FRESH (see results/n3/prereg_phaseA.json for why).
Phase B  M1/M2, sigma in {15,30,60,90}, D in {30,90,175}, seeds 0-2, 20k. LR: night-2 protocol per
         (model, sigma) at D=90 ONLY, seed 100, base grid {3e-4,1e-3,3e-3}, pre-registered one-step edge
         extension applied to M1 and M2 jointly. Protocol-identical existing runs are reused:
           sigma=30/60 @ 1e-3  -> n2_pilot_gauss{30,60}   (D=90, seed 100, 20k, val NFE [8,32], k=2)
           sigma=90 @ any LR   -> n2_hp_gauss90_lr*       (D=90 subset of the night-2 search)
           sigma=90 main       -> n2_main_gauss90 when the D=90-only pick equals the night-2 pick
Phase C  width 2048, P3, D in {30,150}, M1/M2, seed 0, 20k, P3 LRs tuned at width 1024. Diagnostic only.
"""
import sys

import night2 as N
from common import read_json, write_json, get_logger
from night2 import P3, TEST_EVAL, TUNE_SEED, gauss, run_dir, score_uni, task, task_done

N3 = N.RESULTS / "n3"
AB_MODELS = ["M1_riemannian", "M2_euclid6d"]
SEEDS = [0, 1, 2]
A_D, A_STEPS = 150, [320000, 160000]
SIGMAS_B, D_B = [15, 30, 60, 90], [30, 90, 175]
LR_BASE, LR_MIN, LR_MAX = [3e-4, 1e-3, 3e-3], 1e-4, 1e-2
VAL_832 = ("uni", "val", [8, 32], 2, "eval_val.json", None)


def hp2(name):
    return read_json(N.N2 / f"hparams_{name}.json")["chosen_lr"]


def phase_a():
    out = []
    for pname, prior, hp in [("trunc150", P3, hp2("trunc150")), ("gauss90", gauss(90), hp2("gauss90"))]:
        for S in A_STEPS:
            prio = 1 if S == 320000 else 2
            out += [task(f"n3_A_{pname}_steps{S}", m, A_D, s, hp[m], S, "cpu", prio + 0.01 * s, [TEST_EVAL],
                         prior=prior) for s in SEEDS for m in AB_MODELS]
    # Matched-LR sensitivity arm (prereg_phaseA.json addendum_1): M2 at 1e-3 under P3, all five budgets
    out += [task(f"n3_A_trunc150_M2lr1e-3_steps{S}", "M2_euclid6d", A_D, s, 1e-3, S, "cpu", 1.5 + 0.01 * s,
                 [TEST_EVAL], prior=P3) for S in (320000, 160000, 80000, 40000, 20000) for s in SEEDS]
    # P3 has no 40k point from night 2 (B1 ran 40k only under gauss90); fill it so both priors have 5 budgets
    out += [task("n3_A_trunc150_steps40000", m, A_D, s, hp2("trunc150")[m], 40000, "cpu", 2.5 + 0.01 * s,
                 [TEST_EVAL], prior=P3) for s in SEEDS for m in AB_MODELS]
    return out


def search_tag(sigma, lr):
    if sigma == 90:
        return f"n2_hp_gauss90_lr{lr:g}"
    if sigma in (30, 60) and lr == 1e-3:
        return f"n2_pilot_gauss{sigma}"
    return f"n3_hp_gauss{sigma}_lr{lr:g}"


def search_tasks(sigma, lr):
    return [task(search_tag(sigma, lr), m, 90, TUNE_SEED, lr, N.STEPS, "cpu", 3, [VAL_832], prior=gauss(sigma))
            for m in AB_MODELS]


def lr_search_b(sigma):
    path = N3 / f"hparams_gauss{sigma}.json"
    final = read_json(path)
    if final:
        return [], final
    needed = list(LR_BASE)
    while True:
        missing = [t for lr in needed for t in search_tasks(sigma, lr) if not task_done(t)]
        if missing:
            return missing, None
        grid = sorted(needed)
        score = {m: {f"{lr:g}": score_uni(run_dir(t)) for lr in grid for t in search_tasks(sigma, lr) if t["model"] == m}
                 for m in AB_MODELS}
        chosen = {m: float(min(grid, key=lambda lr: score[m][f"{lr:g}"])) for m in AB_MODELS}
        ext = []
        if any(v == max(grid) for v in chosen.values()) and max(grid) < LR_MAX:
            ext.append(LR_MAX)
        if any(v == min(grid) for v in chosen.values()) and min(grid) > LR_MIN:
            ext.append(LR_MIN)
        if not ext:
            out = dict(sigma=sigma, protocol="D=90 only, seed 100, 20k, val mean err NFE 32; edge rule on M1+M2 jointly",
                       grid=grid, score=score, chosen_lr=chosen, tags={f"{lr:g}": search_tag(sigma, lr) for lr in grid})
            write_json(path, out)
            get_logger().info(f"[n3] LR search gauss{sigma} final: {chosen}  grid {grid}")
            return [], out
        needed += [lr for lr in ext if lr not in needed]


def main_tag(sigma, model, lr):
    if sigma == 90 and lr == hp2("gauss90")[model]:
        return "n2_main_gauss90"                     # bit-identical config already run (seeds 0-4)
    return f"n3_B_gauss{sigma}_lr{lr:g}"


def phase_b():
    out = []
    for sigma in SIGMAS_B:
        miss, hp = lr_search_b(sigma)
        out += miss
        if hp:
            out += [task(main_tag(sigma, m, hp["chosen_lr"][m]), m, D, s, hp["chosen_lr"][m], N.STEPS, "cpu",
                         4 + 0.01 * s, [TEST_EVAL], prior=gauss(sigma))
                    for s in SEEDS for D in D_B for m in AB_MODELS]
    return out


def phase_c():
    hp = hp2("trunc150")
    return [task("n3_C_w2048_trunc150", m, D, 0, hp[m], N.STEPS, "cpu", 5, [TEST_EVAL], prior=P3,
                 arch=dict(width=2048, depth=4)) for D in (30, 150) for m in AB_MODELS]


def ready_tasks():
    return phase_a() + phase_b() + phase_c()


def status():
    from collections import Counter
    ts = ready_tasks()
    c = Counter((t["tag"], task_done(t)) for t in ts)
    for tag in sorted({t["tag"] for t in ts}):
        print(f"{tag:36s} done {c[(tag, True)]:4d}  pending {c[(tag, False)]:4d}")


if __name__ == "__main__":
    N3.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "run":
        N.ready_tasks = ready_tasks          # night-2 scheduler, night-3 task list
        N.run()
    else:
        status()
