"""Night-2 analysis: every number in FINDINGS_N2.md. Runs on partial results (missing -> NaN,
and every table reports n). Outputs:
  results/n2/summary_n2.json    all aggregated numbers
  results/n2/tables_n2.md       markdown tables
  results/figures/n2_*.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import COLOR, INK, INK2, MARKER, SHORT, holm, paired, style  # noqa: E402
from common import EVAL_SEEDS, MODELS, RESULTS, RUNS_DIR, read_json, write_json  # noqa: E402
from night2 import B1_D, B1_SEEDS, B1_STEPS, D_N2, MM_D, N2  # noqa: E402

HEAD = "64"
BINS = [0, 90, 120, 150, 165, 175, 180.01]            # = diag_tail.json
BIN_LABELS = ["0–90", "90–120", "120–150", "150–165", "165–175", "175–180"]
FIG = RESULTS / "figures"
P3NAME = "trunc150"
B1_ALL = [20000] + sorted(B1_STEPS)


def b1_tag(bn, S):
    """20k point of the step sweep = the Phase-A main runs at that prior (CPU is deterministic per seed)."""
    if S == 20000:
        return "main" if bn == "haar" else f"n2_main_{bn}"
    return f"n2_b1_{bn}_steps{S}"


def priors():
    sig = read_json(N2 / "sigma_select.json")
    out = [("haar", "main", "P1 Haar")]
    if sig:
        s = sig["sigma_star"]
        out.append((f"gauss{s}", f"n2_main_gauss{s}", f"P2 Gauss σ={s}°"))
    out.append((P3NAME, f"n2_main_{P3NAME}", "P3 Haar-trunc 150°"))
    if sig and sig["sigma_star"] != 30 and (RUNS_DIR / "n2_main_gauss30").exists():
        out.append(("gauss30", "n2_main_gauss30", "P2′ Gauss σ=30° (sensitivity arm, not pre-registered)"))
    return out


def ms(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), len(x)


def f(mu, sd, n=None, p=2):
    if isinstance(mu, tuple):
        mu, sd, n = mu
    if math.isnan(mu):
        return "—"
    s = f"{mu:.{p}f} ± {sd:.{p}f}" if not math.isnan(sd) else f"{mu:.{p}f}"
    return s + (f" (n={n})" if n is not None and n != 5 else "")


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def load_arr(tag, metric, Ds, seeds, name="eval_test.json", nfe=HEAD, models=MODELS):
    A = np.full((len(models), len(Ds), len(seeds)), np.nan)
    for i, m in enumerate(models):
        for j, D in enumerate(Ds):
            for k, s in enumerate(seeds):
                r = read_json(RUNS_DIR / tag / f"{m}_D{D}_s{s}" / name)
                if r and nfe in r["by_nfe"]:
                    A[i, j, k] = get(r["by_nfe"][nfe], metric)
    return A


def spikes(tag, models=MODELS, Ds=D_N2, seeds=EVAL_SEEDS):
    """Night-1 definition: any 500-step window mean > 1.5x the previous one."""
    out = {}
    for m in models:
        n_sp, n = 0, 0
        for D in Ds:
            for s in seeds:
                t = read_json(RUNS_DIR / tag / f"{m}_D{D}_s{s}" / "train.json")
                if t and t.get("done"):
                    h = [x[1] for x in t["hist"]]
                    n += 1
                    n_sp += any(h[i] > 1.5 * h[i - 1] for i in range(1, len(h)))
        out[m] = [n_sp, n]
    return out


def paired_table(A, Ds):
    """Seed-paired diffs vs M1 (index 0), Holm over the D levels, per comparison."""
    out = {}
    for i, m in enumerate(MODELS[1:], start=1):
        cs = {str(D): paired(A, 0, i, j) for j, D in enumerate(Ds)}
        ph = holm([cs[str(D)].get("p", 1.0) for D in Ds])
        for D, p in zip(Ds, ph):
            cs[str(D)]["p_holm"] = float(p)
        out[m] = cs
    return out


def cell(c):
    if c.get("n", 0) < 2:
        return "—"
    return (f"{c['diff_mean']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] p={c['p_holm']:.2g} "
            f"({abs(c['diff_over_seed_sd']):.1f} sd)" + ("" if c["n"] == 5 else f" n={c['n']}"))


# ----------------------------------------------------------------------------- Phase A

def strata(tag, Ds, seeds=EVAL_SEEDS):
    """-> per-seed bin means pooled over Ds: S[model, bin, seed], counts N[model, bin, seed];
    plus per-D: SD[model, D, bin, seed]; plus cumulative 'd0 < 150' and 'all'."""
    nb = len(BINS) - 1
    S = np.full((4, nb + 2, len(seeds)), np.nan)
    N = np.zeros((4, nb + 2, len(seeds)), int)
    SD = np.full((4, len(Ds), nb, len(seeds)), np.nan)
    for i, m in enumerate(MODELS):
        for k, s in enumerate(seeds):
            es, ds = [], []
            for j, D in enumerate(Ds):
                p = RUNS_DIR / tag / f"{m}_D{D}_s{s}" / "eval_test_samples.npz"
                if not p.exists():
                    continue
                z = np.load(p)
                es.append(z["err_deg"]); ds.append(z["d0_deg"])
                for b in range(nb):
                    sel = (z["d0_deg"] >= BINS[b]) & (z["d0_deg"] < BINS[b + 1])
                    if sel.any():
                        SD[i, j, b, k] = z["err_deg"][sel].mean()
            if len(es) != len(Ds):            # pooled numbers only from complete seeds
                continue
            e, d = np.concatenate(es), np.concatenate(ds)
            for b in range(nb):
                sel = (d >= BINS[b]) & (d < BINS[b + 1])
                N[i, b, k] = sel.sum()
                if sel.any():
                    S[i, b, k] = e[sel].mean()
            for b, sel in ((nb, d < 150), (nb + 1, np.ones_like(d, bool))):
                N[i, b, k] = sel.sum()
                S[i, b, k] = e[sel].mean() if sel.any() else np.nan
    return S, N, SD


def phase_a(summary, L):
    Ls = []
    summary["A"] = {}
    for pn, tag, label in priors():
        A = load_arr(tag, "geo_err_deg.mean", D_N2, EVAL_SEEDS)
        S, N, SD = strata(tag, D_N2)
        rec = dict(tag=tag, label=label, mean_err={}, p95={}, max={}, frac_gt45={}, strata={}, strata_by_D={},
                   paired_mean=paired_table(A, D_N2), spikes=spikes(tag))
        for key, dst in (("geo_err_deg.p95", "p95"), ("geo_err_deg.max", "max"), ("frac_err_above_45deg", "frac_gt45"),
                         ("geo_err_deg.mean", "mean_err"), ("geo_err_deg.median", "median")):
            B = A if key == "geo_err_deg.mean" else load_arr(tag, key, D_N2, EVAL_SEEDS)
            rec[dst] = {m: {str(D): ms(B[i, j]) for j, D in enumerate(D_N2)} for i, m in enumerate(MODELS)}
        labels = BIN_LABELS + ["all d0 < 150", "all"]
        rec["strata"] = {m: {lb: dict(zip(("mean", "sd", "n_seeds"), ms(S[i, b])), n_samples=int(N[i, b].sum()))
                             for b, lb in enumerate(labels)} for i, m in enumerate(MODELS)}
        rec["strata_paired_M2_vs_M1"] = {lb: paired(S, 0, 1, b) for b, lb in enumerate(labels)}
        rec["strata_by_D"] = {m: {str(D): {lb: ms(SD[i, j, b]) for b, lb in enumerate(BIN_LABELS)}
                                  for j, D in enumerate(D_N2)} for i, m in enumerate(MODELS)}
        rec["strata_paired_M2_vs_M1_by_D"] = {str(D): {lb: paired(SD[:, j], 0, 1, b) for b, lb in enumerate(BIN_LABELS)}
                                              for j, D in enumerate(D_N2)}
        rec["d0_bin_fraction"] = {lb: float(N[0, b].sum() / max(1, N[0, -1].sum())) for b, lb in enumerate(labels)}
        summary["A"][pn] = rec

        Ls.append(f"\n### {label}  (`runs/{tag}`)\n")
        Ls.append(f"**Mean endpoint geodesic error (deg), NFE={HEAD}, test, mean ± sd over seeds**\n")
        Ls.append("| Model | " + " | ".join(f"D={D}" for D in D_N2) + " |")
        Ls.append("|---|" + "---|" * len(D_N2))
        for i, m in enumerate(MODELS):
            Ls.append(f"| {SHORT[m]} | " + " | ".join(f(*rec["mean_err"][m][str(D)]) for D in D_N2) + " |")
        for key, nm, p in (("p95", "95th percentile", 1), ("max", "Max", 0), ("frac_gt45", "Fraction > 45°", 3)):
            Ls.append(f"\n**{nm}**\n")
            Ls.append("| Model | " + " | ".join(f"D={D}" for D in D_N2) + " |")
            Ls.append("|---|" + "---|" * len(D_N2))
            for m in MODELS:
                Ls.append(f"| {SHORT[m]} | " + " | ".join(f(*rec[key][m][str(D)], p=p) for D in D_N2) + " |")
        Ls.append("\n**Seed-paired difference vs M1, mean error (deg): diff [95% CI], Holm p over the 5 D, "
                  "|diff|/pooled seed sd**\n")
        Ls.append("| vs M1 | " + " | ".join(f"D={D}" for D in D_N2) + " |")
        Ls.append("|---|" + "---|" * len(D_N2))
        for m in MODELS[1:]:
            Ls.append(f"| {SHORT[m]} | " + " | ".join(cell(rec["paired_mean"][m][str(D)]) for D in D_N2) + " |")
        Ls.append(f"\n**d0-stratified mean error (deg), pooled over D ∈ {D_N2}, NFE={HEAD}; per-seed bin means, "
                  "mean ± sd over seeds. d0 = geodesic distance from the last-waypoint source sample to the target endpoint.**\n")
        Ls.append("| d0 bin | share of samples | " + " | ".join(SHORT[m] for m in MODELS) + " | M2 − M1 (seed-paired) |")
        Ls.append("|---|---|" + "---|" * 4 + "---|")
        for b, lb in enumerate(labels):
            c = rec["strata_paired_M2_vs_M1"][lb]
            pc = (f"{c['diff_mean']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] ({abs(c['diff_over_seed_sd']):.1f} sd)"
                  if c.get("n", 0) >= 2 else "—")
            Ls.append(f"| {lb} | {rec['d0_bin_fraction'][lb]:.3f} | "
                      + " | ".join(f(*ms(S[i, b])) for i in range(4)) + f" | {pc} |")
        sp = rec["spikes"]
        Ls.append("\nLoss spikes (runs with any 500-step mean > 1.5× the previous): "
                  + ", ".join(f"{m[:2]} {sp[m][0]}/{sp[m][1]}" for m in MODELS))
    L.append("\n## Phase A — source distribution\n")
    L += lr_tables()
    L += Ls


def lr_tables():
    L = []
    sig = read_json(N2 / "sigma_select.json")
    if sig:
        L.append("\n**P2 σ pilot (D=90, seed 100, LR 1e-3, 20k; val mean error at NFE 32, deg)**\n")
        L.append("| σ | " + " | ".join(SHORT[m] for m in MODELS) + " | geometric mean |")
        L.append("|---|" + "---|" * 5)
        for s in sorted(sig["val_err"], key=float):
            L.append(f"| {s}° | " + " | ".join(f"{sig['val_err'][s][m]:.2f}" for m in MODELS)
                     + f" | {sig['geomean'][s]:.2f} |")
        L.append(f"\nσ* = {sig['sigma_star']}°\n")
    hp1 = read_json(RESULTS / "hparams.json")
    tabs = [("P1 Haar (night 1)", {m: {lr: v["mean"] for lr, v in hp1["table"][m].items()} for m in MODELS},
             hp1["chosen_lr"])]
    for p in sorted(N2.glob("hparams_*.json")):
        h = read_json(p)
        tabs.append((h["search"], h["mean"], h["chosen_lr"]))
    for name, mean, chosen in tabs:
        lrs = sorted({lr for m in MODELS for lr in mean[m]}, key=float)
        L.append(f"\n**LR search: {name}** (val mean {'W1' if name.startswith('mm') else 'endpoint error'} "
                 f"at NFE 32, deg, averaged over the tuning D; bold = selected)\n")
        L.append("| LR | " + " | ".join(SHORT[m] for m in MODELS) + " |")
        L.append("|---|" + "---|" * 4)
        for lr in lrs:
            L.append(f"| {lr} | " + " | ".join(
                (f"**{mean[m][lr]:.2f}**" if float(lr) == chosen[m] else f"{mean[m][lr]:.2f}") if lr in mean[m] else "—"
                for m in MODELS) + " |")
    best = read_json(N2 / "best_prior.json")
    if best:
        L.append(f"\nBest legal prior (pre-registered rule: geometric mean over models of selected-LR val score): "
                 + ", ".join(f"{k} {v:.2f}" for k, v in best["geomean"].items()) + f" → **{best['best']}**"
                 + (f". **Deviation:** used **{best['used']}** for Phases B1/C ({best['deviation']}).\n" if best.get("used") else "\n"))
    return L


# ----------------------------------------------------------------------------- Phase B

def phase_b(summary, L):
    best = read_json(N2 / "best_prior.json")
    L.append("\n## Phase B — budget stability\n")
    summary["B"] = {}
    if best:
        bn = best.get("used", best["best"])
        L.append(f"\n### B1 step sweep (prior {bn}, seeds {B1_SEEDS}, NFE {HEAD}; 20k = Phase-A main runs)\n")
        L.append("| steps | D | " + " | ".join(SHORT[m] for m in MODELS) + " | M2 − M1 (paired) | final loss M1 / M2 |")
        L.append("|---|---|" + "---|" * 4 + "---|---|")
        b1 = {}
        for S in B1_ALL:
            tag = b1_tag(bn, S)
            A = load_arr(tag, "geo_err_deg.mean", B1_D, B1_SEEDS)
            FL = np.full((4, len(B1_D), len(B1_SEEDS)), np.nan)
            for i, m in enumerate(MODELS):
                for j, D in enumerate(B1_D):
                    for k, s in enumerate(B1_SEEDS):
                        t = read_json(RUNS_DIR / tag / f"{m}_D{D}_s{s}" / "train.json")
                        if t and t.get("done"):
                            FL[i, j, k] = t["final_loss"]
            b1[S] = dict(err={m: {str(D): ms(A[i, j]) for j, D in enumerate(B1_D)} for i, m in enumerate(MODELS)},
                         final_loss={m: {str(D): ms(FL[i, j]) for j, D in enumerate(B1_D)} for i, m in enumerate(MODELS)},
                         paired={str(D): paired(A, 0, 1, j) for j, D in enumerate(B1_D)},
                         spikes=spikes(tag, Ds=B1_D, seeds=B1_SEEDS))
            for j, D in enumerate(B1_D):
                c = b1[S]["paired"][str(D)]
                pc = (f"{c['diff_mean']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] ({abs(c['diff_over_seed_sd']):.1f} sd)"
                      if c.get("n", 0) >= 2 else "—")
                L.append(f"| {S // 1000}k | {D} | " + " | ".join(f(*ms(A[i, j])[:2]) for i in range(4))
                         + f" | {pc} | {f(*ms(FL[0, j]), p=3)} / {f(*ms(FL[1, j]), p=3)} |")
        summary["B"]["B1"] = b1
        # loss flattening: relative drop over the last 25% of steps, and final loss vs budget
        flat = {}
        for S in B1_ALL:
            tag = b1_tag(bn, S)
            for m in MODELS:
                drops = []
                for D in B1_D:
                    for s in B1_SEEDS:
                        t = read_json(RUNS_DIR / tag / f"{m}_D{D}_s{s}" / "train.json")
                        if t and t.get("done"):
                            h = np.array(t["hist"])
                            q = h[h[:, 0] > 0.75 * S, 1]
                            h1 = h[(h[:, 0] > 0.5 * S) & (h[:, 0] <= 0.75 * S), 1]
                            drops.append(1 - q.mean() / h1.mean())
                flat.setdefault(str(S), {})[m] = ms(drops)
        summary["B"]["B1_loss_drop_last_quarter_vs_third"] = flat
        L.append("\nRelative loss drop, mean(last 25% of steps) vs mean(50–75%), per model (mean ± sd over D × seeds):\n")
        L.append("| steps | " + " | ".join(SHORT[m] for m in MODELS) + " |")
        L.append("|---|" + "---|" * 4)
        for S in B1_ALL:
            L.append(f"| {S // 1000}k | " + " | ".join(f(*flat[str(S)][m][:2], p=3) for m in MODELS) + " |")
        loss_figure(bn)

    # B2 width
    rows = [("256 (15k steps, superseded M1 head)", {m: f"pilot_steps15000/{m}_D90_s100" for m in MODELS}),
            ("512", {m: f"diag_w512/{m}_D90_s100" for m in MODELS}),
            ("1024", {**{m: f"diag_w1024/{m}_D90_s100" for m in MODELS},
                      "M1_riemannian": "diag_w1024_M1proj/M1_riemannian_D90_s100"}),
            ("2048 (night 2)", {m: f"n2_diag_w2048/{m}_D90_s100" for m in MODELS})]
    L.append("\n### B2 width sweep (Haar, D=90, seed 100, LR 1e-3, 10k steps; val mean error, NFE 32, deg)\n")
    L.append("| width | " + " | ".join(SHORT[m] for m in MODELS) + " | ranking (best first) | params (M1) |")
    L.append("|---|" + "---|" * 4 + "---|---|")
    b2 = {}
    for name, paths in rows:
        v, npar = {}, "—"
        for m in MODELS:
            r = read_json(RUNS_DIR / paths[m] / "eval_val.json")
            v[m] = r["by_nfe"]["32"]["geo_err_deg"]["mean"] if r and "32" in r["by_nfe"] else float("nan")
        t = read_json(RUNS_DIR / paths["M1_riemannian"] / "train.json")
        if t:
            npar = f"{t['n_params'] / 1e6:.2f}M"
        b2[name] = v
        rank = " < ".join(m[:2] for m in sorted(MODELS, key=lambda m: v[m])) if not any(map(math.isnan, v.values())) else "—"
        L.append(f"| {name} | " + " | ".join(f"{v[m]:.1f}" for m in MODELS) + f" | {rank} | {npar} |")
    summary["B"]["B2"] = b2

    # B3 M1 LR stability
    A3 = load_arr("n2_b3_M1_lr0.001", "geo_err_deg.mean", D_N2, EVAL_SEEDS, models=["M1_riemannian"])
    A1 = load_arr("main", "geo_err_deg.mean", D_N2, EVAL_SEEDS)
    sp3 = spikes("n2_b3_M1_lr0.001", models=["M1_riemannian"])["M1_riemannian"]
    sp1 = spikes("main", models=["M1_riemannian"])["M1_riemannian"]
    L.append("\n### B3 M1 learning-rate stability (Haar, 20k, CPU, test, NFE 64)\n")
    L.append("| | " + " | ".join(f"D={D}" for D in D_N2) + " | runs with a loss spike |")
    L.append("|---|" + "---|" * len(D_N2) + "---|")
    L.append("| M1 @ 3e-3 (protocol pick, night 1) | " + " | ".join(f(*ms(A1[0, j])) for j in range(len(D_N2)))
             + f" | {sp1[0]}/{sp1[1]} |")
    L.append("| M1 @ 1e-3 | " + " | ".join(f(*ms(A3[0, j])) for j in range(len(D_N2))) + f" | {sp3[0]}/{sp3[1]} |")
    L.append("| M2 @ 3e-3 (protocol pick) | " + " | ".join(f(*ms(A1[1, j])) for j in range(len(D_N2))) + " | 0 by night-1 count |")
    Ab = np.stack([A3[0], A1[1]])                      # M1@1e-3 vs M2
    pb = {str(D): paired(Ab, 0, 1, j) for j, D in enumerate(D_N2)}
    ph = holm([pb[str(D)].get("p", 1.0) for D in D_N2])
    for D, p in zip(D_N2, ph):
        pb[str(D)]["p_holm"] = float(p)
    L.append("| M2 − M1@1e-3 (paired) | " + " | ".join(cell(pb[str(D)]) for D in D_N2) + " | |")
    summary["B"]["B3"] = dict(m1_lr1e3={str(D): ms(A3[0, j]) for j, D in enumerate(D_N2)},
                              m1_lr3e3={str(D): ms(A1[0, j]) for j, D in enumerate(D_N2)},
                              spikes_1e3=sp3, spikes_3e3=sp1, paired_M2_vs_M1_1e3=pb)


def loss_figure(bn):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ls = {20000: ":", 40000: "--", 80000: "-"}
    for ax, D in zip(axs, B1_D):
        for m in MODELS[:2] + MODELS[2:]:
            for S in B1_ALL:
                hs = []
                for s in B1_SEEDS:
                    t = read_json(RUNS_DIR / b1_tag(bn, S) / f"{m}_D{D}_s{s}" / "train.json")
                    if t and t.get("done"):
                        hs.append(np.array(t["hist"]))
                if not hs:
                    continue
                n = min(len(h) for h in hs)
                h = np.mean([h[:n, 1] for h in hs], 0)
                ax.plot(hs[0][:n, 0] / 1000, h, color=COLOR[m], ls=ls[S], lw=1.6,
                        label=f"{SHORT[m].split()[0]} {S // 1000}k" if D == B1_D[0] else None)
        style(ax, "training step (thousands)", "training loss (500-step mean)", log=True)
        ax.set_title(f"D = {D}°, prior {bn}; mean over {len(B1_SEEDS)} seeds", loc="left")
    axs[0].legend(frameon=False, fontsize=7, ncol=4)
    fig.suptitle("B1: loss curves at 20k (dotted) / 40k (dashed) / 80k (solid) steps; cosine LR decay to 0 in every run",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "n2_b1_loss.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- Phase C

MM_METRICS = [("w1_deg.mean", "W1 to the 3-mode target (deg) ↓", 2),
              ("nearest_mode_err_deg.mean", "Nearest-mode error, mean (deg) ↓", 2),
              ("nearest_mode_err_deg.median", "Nearest-mode error, median (deg) ↓", 2),
              ("frac_on_mode", "Fraction of samples within 10° of a mode ↑", 3),
              ("coverage_all_modes", "Starts with all 3 modes hit (of 30 samples) ↑", 3),
              ("modes_hit_mean", "Modes hit per start (of 3) ↑", 2),
              ("mode_props_min_mean", "Smallest mode share per start (uniform = 0.333) ↑", 3),
              ("tv_from_uniform_mean", "TV distance of mode shares from uniform ↓", 3)]


def phase_c(summary, L):
    best = read_json(N2 / "best_prior.json")
    L.append("\n## Phase C — multimodal target\n")
    if not best:
        return
    tag = f"n2_mm_{best.get('used', best['best'])}"
    summary["C"] = {"tag": tag}
    for nfe in ("64", "8"):
        L.append(f"\n**{tag}, test (200 starts × 30 samples), NFE={nfe}, mean ± sd over seeds**\n")
        L.append("| metric | D | " + " | ".join(SHORT[m] for m in MODELS) + " | M2 − M1 (paired, Holm over D) |")
        L.append("|---|---|" + "---|" * 4 + "---|")
        for key, name, p in MM_METRICS:
            A = load_arr(tag, key, MM_D, EVAL_SEEDS, name="eval_mm_test.json", nfe=nfe)
            pt = paired_table(A, MM_D)
            summary["C"].setdefault(nfe, {})[key] = dict(
                agg={m: {str(D): ms(A[i, j]) for j, D in enumerate(MM_D)} for i, m in enumerate(MODELS)}, paired=pt)
            for j, D in enumerate(MM_D):
                c = pt["M2_euclid6d"][str(D)]
                L.append(f"| {name} | {D} | " + " | ".join(f(*ms(A[i, j]), p=p) for i in range(4))
                         + f" | {cell(c) if c.get('n', 0) >= 2 else '—'} |")
    # pooled mode proportions
    L.append("\n**Pooled mode shares (modes 0 / +60° / −60°), NFE 64, mean over seeds; data share for reference**\n")
    L.append("| D | " + " | ".join(SHORT[m] for m in MODELS) + " |")
    L.append("|---|" + "---|" * 4)
    for D in MM_D:
        cells = []
        for m in MODELS:
            ps = [read_json(RUNS_DIR / tag / f"{m}_D{D}_s{s}" / "eval_mm_test.json") for s in EVAL_SEEDS]
            ps = [r["by_nfe"]["64"]["mode_props_pooled"] for r in ps if r and "64" in r["by_nfe"]
                  and r["by_nfe"]["64"]["mode_props_pooled"]]
            cells.append(" / ".join(f"{x:.2f}" for x in np.mean(ps, 0)) if ps else "—")
        L.append(f"| {D} | " + " | ".join(cells) + " |")
    spk = spikes(tag, Ds=MM_D)
    summary["C"]["spikes"] = spk
    L.append("\nLoss spikes: " + ", ".join(f"{m[:2]} {spk[m][0]}/{spk[m][1]}" for m in MODELS))


# ----------------------------------------------------------------------------- figures

def fig_phase_a(summary):
    ps = [p for p in priors() if p[0] in summary["A"]]
    fig, axs = plt.subplots(2, len(ps), figsize=(5 * len(ps), 8.5), sharey="row", squeeze=False)
    x = np.array(D_N2)
    for c, (pn, tag, label) in enumerate(ps):
        rec = summary["A"][pn]
        ax = axs[0, c]
        for m in MODELS:
            mu = np.array([rec["mean_err"][m][str(D)][0] for D in D_N2])
            sd = np.array([rec["mean_err"][m][str(D)][1] for D in D_N2])
            ax.fill_between(x, np.maximum(mu - sd, 0.5 * mu), mu + sd, color=COLOR[m], alpha=0.18, lw=0)
            ax.plot(x, mu, color=COLOR[m], marker=MARKER[m], ms=7, lw=2, markeredgecolor="white",
                    markeredgewidth=1, label=SHORT[m])
            if not np.isnan(mu[-1]):
                ax.annotate(SHORT[m].split()[0], (x[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                            fontsize=8, va="center", color=INK)
        style(ax, "Angular displacement D (deg)", "Mean endpoint error (deg)", log=True)
        ax.set_xticks(x)
        ax.set_title(f"{label}", loc="left")
        ax = axs[1, c]
        labels = BIN_LABELS
        xb = np.arange(len(labels))
        for k, m in enumerate(MODELS):
            mu = np.array([rec["strata"][m][lb]["mean"] for lb in labels])
            sd = np.array([rec["strata"][m][lb]["sd"] for lb in labels])
            ax.errorbar(xb + (k - 1.5) * 0.14, mu, yerr=sd, fmt=MARKER[m], color=COLOR[m], ms=6, lw=1.2,
                        capsize=0, markeredgecolor="white", label=SHORT[m])
        style(ax, "d0 = source-to-target distance (deg)", "Mean endpoint error in bin (deg)", log=True)
        ax.set_xticks(xb)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("d0-stratified, pooled over D", loc="left")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Phase A: endpoint error by source prior (NFE {HEAD}, ±1 sd over 5 seeds)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "n2_phaseA.png", dpi=150)
    plt.close(fig)


def main():
    summary, L = {}, ["# Night-2 tables (generated by src/analyze_n2.py)\n"]
    phase_a(summary, L)
    phase_b(summary, L)
    phase_c(summary, L)
    write_json(N2 / "summary_n2.json", summary)
    (N2 / "tables_n2.md").write_text("\n".join(L) + "\n")
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    fig_phase_a(summary)
    print("wrote", N2 / "tables_n2.md")


if __name__ == "__main__":
    main()
