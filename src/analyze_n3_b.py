"""Night-3 Phase B analysis + night-3 figures. Runs on partial data (cells report n).

Per (sigma, D): overall mean endpoint error (NFE 64, test), d0-stratified error, seed-paired M2 - M1 with t-CI
(positive = M1 better). Sensitivity: sigma=90 M1 at the night-2 LR (1e-3) beside the D=90-only pick.
Outputs: results/n3/phaseB.json, results/figures/n3_sigma_gap.png, results/figures/n3_steps_gap.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from common import RESULTS, RUNS_DIR, read_json, write_json  # noqa: E402
from night3 import AB_MODELS, D_B, N3, SEEDS, SIGMAS_B, main_tag  # noqa: E402

BINS = [(0, 90), (90, 120), (120, 150), (150, 165), (165, 181)]
BLAB = ["0–90", "90–120", "120–150", "150–165", "≥165"]
SERIES = {30: "#2a78d6", 90: "#eb6834", 175: "#1baf7a"}          # categorical slots 1-3 (validated all-pairs)
MARK = {30: "o", 90: "s", 175: "^"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def cell(tag, m, D):
    errs, strata = [], []
    for s in SEEDS:
        rd = RUNS_DIR / tag / f"{m}_D{D}_s{s}"
        e = read_json(rd / "eval_test.json")
        if not (e and "64" in e["by_nfe"] and (rd / "eval_test_samples.npz").exists()):
            return None
        z = np.load(rd / "eval_test_samples.npz")
        errs.append(e["by_nfe"]["64"]["geo_err_deg"]["mean"])
        strata.append([float(z["err_deg"][(z["d0_deg"] >= a) & (z["d0_deg"] < b)].mean())
                       if ((z["d0_deg"] >= a) & (z["d0_deg"] < b)).any() else np.nan for a, b in BINS])
    return np.array(errs), np.array(strata)


def paired(e1, e2):
    d = e2 - e1
    se = d.std(ddof=1) / math.sqrt(len(d))
    tc = stats.t.ppf(0.975, len(d) - 1)
    psd = math.sqrt(0.5 * (e1.var(ddof=1) + e2.var(ddof=1)))
    return dict(gap=float(d.mean()), ci=[float(d.mean() - tc * se), float(d.mean() + tc * se)],
                sd_units=float(d.mean() / psd) if psd > 0 else None,
                p=float(stats.ttest_rel(e2, e1).pvalue) if d.std() > 0 else None)


def phase_b():
    out = {"cells": {}, "hparams": {}}
    for sig in SIGMAS_B:
        hp = read_json(N3 / f"hparams_gauss{sig}.json")
        if not hp:
            continue
        out["hparams"][sig] = hp["chosen_lr"]
        for D in D_B:
            c = {m: cell(main_tag(sig, m, hp["chosen_lr"][m]), m, D) for m in AB_MODELS}
            if any(v is None for v in c.values()):
                continue
            (e1, s1), (e2, s2) = c["M1_riemannian"], c["M2_euclid6d"]
            rec = dict(M1=[float(e1.mean()), float(e1.std(ddof=1))], M2=[float(e2.mean()), float(e2.std(ddof=1))],
                       **paired(e1, e2),
                       strata={BLAB[b]: dict(M1=float(np.nanmean(s1[:, b])), M2=float(np.nanmean(s2[:, b])),
                                             **(paired(s1[:, b], s2[:, b]) if not np.isnan(s1[:, b]).any() else {}))
                               for b in range(len(BINS)) if not np.isnan(s1[:, b]).all()})
            if sig == 90 and hp["chosen_lr"]["M1_riemannian"] != 1e-3:          # LR sensitivity, night-2 pick
                alt = cell("n2_main_gauss90", "M1_riemannian", D)
                if alt is not None:
                    rec["M1_at_1e-3"] = [float(alt[0].mean()), float(alt[0].std(ddof=1))]
                    rec["gap_vs_M1_at_1e-3"] = paired(alt[0], e2)
            out["cells"][f"{sig}_{D}"] = rec
    return out


def style(ax, xl, yl):
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.axhline(0, color=INK2, lw=1, ls="--")


def fig_sigma(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for D in D_B:
        pts = [(s, res["cells"][f"{s}_{D}"]) for s in SIGMAS_B if f"{s}_{D}" in res["cells"]]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1]["gap"] for p in pts]
        lo = [p[1]["gap"] - p[1]["ci"][0] for p in pts]
        hi = [p[1]["ci"][1] - p[1]["gap"] for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[D], lw=2, marker=MARK[D], ms=8, capsize=3,
                    markeredgecolor="white", markeredgewidth=1, label=f"D = {D}°")
        ax.annotate(f"D={D}°", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9,
                    va="center")
    ax.set_xticks(SIGMAS_B)
    style(ax, "Start-centered Gaussian prior σ (deg)", "M2 − M1 mean endpoint error (deg)\n> 0: Riemannian better")
    ax.set_title("Riemannian vs 6D gap by prior width (seed-paired 95% CI, 3 seeds, 20k)", loc="left", color=INK,
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "n3_sigma_gap.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_steps():
    a = read_json(N3 / "phaseA.json")
    if not a:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [  # key, color, marker, label (categorical slots 1-3, validated all-pairs)
        ("trunc150_matchedLR", "#2a78d6", "o", "Trunc-150 (oracle), both at LR 1e-3 [matched-LR arm]"),
        ("trunc150__without_diverged", "#eb6834", "s", "Trunc-150, protocol LRs, diverged runs removed"),
        ("gauss90__without_diverged", "#1baf7a", "^", "Gaussian σ=90°, protocol LRs, diverged runs removed"),
    ]
    for key, col, mk, lab in series:
        t = [r for r in a[key]["table"] if r.get("n", 0) >= 2]
        xs = [r["steps"] / 1000 for r in t]
        ys = [r["gap"] for r in t]
        lo = [r["gap"] - r["gap_ci"][0] for r in t]
        hi = [r["gap_ci"][1] - r["gap"] for r in t]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, lw=2, marker=mk, ms=8, capsize=3, markeredgecolor="white",
                    markeredgewidth=1, label=lab)
        short = {"trunc150_matchedLR": "matched LR", "trunc150__without_diverged": "trunc protocol",
                 "gauss90__without_diverged": "Gauss σ=90"}[key]
        ax.annotate(short, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9,
                    va="center")
    ax.annotate("trunc protocol @320k: all 3 M2 runs diverged (not plotted)", (320, 0.5), xytext=(-10, 30),
                textcoords="offset points", ha="right", color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xscale("log", base=2)
    ticks = [20, 40, 80, 160, 320]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}k" for t in ticks])
    style(ax, "Training steps (log scale)", "M2 − M1 mean endpoint error (deg)\n> 0: Riemannian better")
    ax.set_title("D=150°: gap vs training budget (seed-paired 95% CI, 3 seeds; n=2 where one pair diverged)",
                 loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "n3_steps_gap.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    res = phase_b()
    write_json(N3 / "phaseB.json", res)
    print("LRs:", res["hparams"])
    for k, r in res["cells"].items():
        sig, D = k.split("_")
        print(f"σ={sig:>2} D={D:>3}: M1 {r['M1'][0]:6.2f}±{r['M1'][1]:.2f}  M2 {r['M2'][0]:6.2f}±{r['M2'][1]:.2f}  "
              f"M2−M1 {r['gap']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] ({r['sd_units']:.1f} sd)"
              + (f"  | M1@1e-3 {r['M1_at_1e-3'][0]:.2f} gap {r['gap_vs_M1_at_1e-3']['gap']:+.2f}" if "M1_at_1e-3" in r else ""))
        print("      strata:", {b: (round(v["M1"], 2), round(v["M2"], 2)) for b, v in r["strata"].items()})
    fig_sigma(res)
    fig_steps()
