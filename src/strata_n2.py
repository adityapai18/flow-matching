"""Night-2 headline: d0-stratified endpoint error per (model, prior), excluding D=10 (where M2 at its
selected LR diverged in 1/5 runs under P2 and P3; reported separately). Per-seed bin means pooled over
D in {30, 90, 150, 175}; seed-paired M2 - M1 difference with t-CI. Output results/n2/strata_noD10.json"""
import math
import numpy as np
from scipy import stats
from common import MODELS, RUNS_DIR, write_json
from night2 import N2

BINS = [0, 90, 120, 150, 165, 175, 180.01]
LAB = ["0–90", "90–120", "120–150", "150–165", "165–175", "175–180", "all d0<150", "all"]
DS = [30, 90, 150, 175]
TAGS = {"P1 Haar": "main", "P2 Gauss σ=90": "n2_main_gauss90", "P3 Haar-trunc150": "n2_main_trunc150"}


def cell(tag, m, s):
    e, d = [], []
    for D in DS:
        z = np.load(RUNS_DIR / tag / f"{m}_D{D}_s{s}" / "eval_test_samples.npz")
        e.append(z["err_deg"]); d.append(z["d0_deg"])
    e, d = np.concatenate(e), np.concatenate(d)
    sels = [(d >= BINS[b]) & (d < BINS[b + 1]) for b in range(6)] + [d < 150, np.ones_like(d, bool)]
    return [float(e[x].mean()) if x.any() else np.nan for x in sels], [float(x.mean()) for x in sels]


out = {}
for pn, tag in TAGS.items():
    A = {m: np.array([cell(tag, m, s)[0] for s in range(5)]) for m in MODELS}     # seed x bin
    share = cell(tag, "M1_riemannian", 0)[1]
    rows = []
    for b, lb in enumerate(LAB):
        r = dict(bin=lb, share=share[b], **{m: [float(np.nanmean(A[m][:, b])), float(np.nanstd(A[m][:, b], ddof=1))] for m in MODELS})
        d = A["M2_euclid6d"][:, b] - A["M1_riemannian"][:, b]
        if not np.isnan(d).any():
            se = d.std(ddof=1) / math.sqrt(5); tc = stats.t.ppf(.975, 4)
            psd = math.sqrt(.5 * (A["M2_euclid6d"][:, b].var(ddof=1) + A["M1_riemannian"][:, b].var(ddof=1)))
            r["M2_minus_M1"] = dict(diff=float(d.mean()), ci=[float(d.mean() - tc * se), float(d.mean() + tc * se)],
                                    p=float(stats.ttest_rel(A["M2_euclid6d"][:, b], A["M1_riemannian"][:, b]).pvalue),
                                    sd_units=float(abs(d.mean()) / psd) if psd > 0 else float("inf"))
        rows.append(r)
    out[pn] = rows
    print(f"\n{pn} (D in {DS})")
    for r in rows:
        c = r.get("M2_minus_M1")
        print(f"  {r['bin']:12s} share {r['share']:.3f}  M1 {r['M1_riemannian'][0]:6.2f}±{r['M1_riemannian'][1]:.2f}  M2 {r['M2_euclid6d'][0]:6.2f}±{r['M2_euclid6d'][1]:.2f}  M3 {r['M3_quat'][0]:6.2f}  M4 {r['M4_euler'][0]:6.2f}"
              + (f"  M2-M1 {c['diff']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] p={c['p']:.2g} ({c['sd_units']:.1f} sd)" if c else ""))
write_json(N2 / "strata_noD10.json", out)
