"""Night-3 training-stability record for every D=150 budget-sweep run (Phase A + matched-LR arm).
Spike = a 500-step mean loss > 1.5x the previous one (night-2 definition). LR fraction of peak at the first
spike from the actual schedule (500-step warmup, cosine to 0 over the run). Diverged = final loss > 5x the
final loss of the same model/prior/LR at 80k (prereg addendum_1; the cell-median rule fails here).
Output: results/n3/stability.json"""
import glob
import math
import os

from common import RUNS_DIR, read_json, write_json
from night3 import N3

WARMUP = 500


def lr_frac(step, total):
    if step < WARMUP:
        return (step + 1) / WARMUP
    return 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / max(1, total - WARMUP)))


REF80 = {("trunc150", "M1_riemannian", 1e-3): "n2_b1x_trunc150_steps80000", ("trunc150", "M2_euclid6d", 3e-3): "n2_b1x_trunc150_steps80000",
         ("gauss90", "M1_riemannian", 1e-3): "n2_b1_gauss90_steps80000", ("gauss90", "M2_euclid6d", 3e-3): "n2_b1_gauss90_steps80000",
         ("trunc150", "M2_euclid6d", 1e-3): "n3_A_trunc150_M2lr1e-3_steps80000"}


def ref_final(prior, m, lr):
    tag = REF80.get((prior, m, lr))
    if not tag:
        return None
    vals = []
    for s in (0, 1, 2):
        t = read_json(RUNS_DIR / tag / f"{m}_D150_s{s}" / "train.json")
        if t and t.get("done"):
            vals.append(t["hist"][-1][1])
    return min(vals) if vals else None


rows = []
pats = ["n2_main_trunc150", "n2_main_gauss90", "n2_b1_gauss90_steps*", "n2_b1x_trunc150_steps*", "n3_A_*"]
for pat in pats:
    for d in sorted(glob.glob(str(RUNS_DIR / pat))):
        tag = os.path.basename(d)
        prior = "trunc150" if "trunc150" in tag else "gauss90"
        for m in ("M1_riemannian", "M2_euclid6d"):
            for s in (0, 1, 2):
                t = read_json(RUNS_DIR / tag / f"{m}_D150_s{s}" / "train.json")
                if not t or not t.get("hist"):
                    continue
                h = t["hist"]
                L = [x[1] for x in h]
                spikes = [(h[i][0], L[i - 1], L[i]) for i in range(1, len(L)) if L[i] > 1.5 * L[i - 1]]
                ref = ref_final(prior, m, t["lr"])
                fin = L[-1]
                e = read_json(RUNS_DIR / tag / f"{m}_D150_s{s}" / "eval_test.json")
                err = e["by_nfe"]["64"]["geo_err_deg"]["mean"] if e and "64" in e["by_nfe"] else None
                rows.append(dict(tag=tag, prior=prior, model=m, seed=s, lr=t["lr"], steps=t["steps"], step_now=t["step"],
                                 done=bool(t.get("done")), n_spikes=len(spikes),
                                 first_spike_step=spikes[0][0] if spikes else None,
                                 lr_frac_at_first_spike=round(lr_frac(spikes[0][0], t["steps"]), 3) if spikes else None,
                                 max_spike_loss=max(x[2] for x in spikes) if spikes else None, final_loss=fin,
                                 diverged=bool(ref and t.get("done") and fin > 5 * ref), err=err))
write_json(N3 / "stability.json", rows)
print(f"{'tag':38s} {'model':5s} s  lr      steps  now    spk first  lrfrac  maxspike  final    div  err")
for r in rows:
    if r["steps"] < 160000 and r["n_spikes"] == 0:
        continue
    print(f"{r['tag']:38s} {r['model'][:2]:5s} {r['seed']} {r['lr']:<7g} {r['steps']//1000:4d}k {r['step_now']//1000:4d}k "
          f"{r['n_spikes']:3d} {str(r['first_spike_step']//1000 if r['first_spike_step'] else '-'):>5s} "
          f"{str(r['lr_frac_at_first_spike'] or '-'):>6s} {str(round(r['max_spike_loss'],2) if r['max_spike_loss'] else '-'):>9s} "
          f"{r['final_loss']:.4f} {'YES' if r['diverged'] else '-':>4s} {('%.2f' % r['err']) if r['err'] else '-'}")
short = [r for r in rows if r["steps"] <= 80000]
print(f"\nruns at <= 80k: {len(short)}, with any spike: {sum(r['n_spikes'] > 0 for r in short)}")
