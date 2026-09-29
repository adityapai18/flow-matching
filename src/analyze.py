"""Stage 4: aggregate the main sweep, statistics, figures, markdown tables.

Outputs:
  results/summary.json           every aggregated number used in FINDINGS
  results/summary_tables.md      markdown tables (pasted into FINDINGS)
  results/figures/*.png
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

import fm  # noqa: E402
from common import D_LEVELS, EVAL_SEEDS, MODELS, NFES, RESULTS, RUNS_DIR, read_json, write_json  # noqa: E402

HEAD_NFE = 64
FIG = RESULTS / "figures"
SHORT = {"M1_riemannian": "M1 Riemannian", "M2_euclid6d": "M2 Euclid-6D", "M3_quat": "M3 Euclid-quat",
         "M4_euler": "M4 Euclid-Euler"}
COLOR = {"M1_riemannian": "#2a78d6", "M2_euclid6d": "#eb6834", "M3_quat": "#1baf7a", "M4_euler": "#eda100"}
MARKER = {"M1_riemannian": "o", "M2_euclid6d": "s", "M3_quat": "^", "M4_euler": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def load(tag="main"):
    ev, tr, missing = {}, {}, []
    for m in MODELS:
        for D in D_LEVELS:
            for s in EVAL_SEEDS:
                rd = RUNS_DIR / tag / f"{m}_D{D}_s{s}"
                e, t = read_json(rd / "eval_test.json"), read_json(rd / "train.json")
                if e and all(str(n) in e["by_nfe"] for n in NFES):
                    ev[m, D, s], tr[m, D, s] = e, t
                else:
                    missing.append(f"{m}_D{D}_s{s}")
    return ev, tr, missing


def arr(ev, metric, nfe=HEAD_NFE):
    """-> array [model, D, seed] (NaN where missing)."""
    A = np.full((len(MODELS), len(D_LEVELS), len(EVAL_SEEDS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, D in enumerate(D_LEVELS):
            for k, s in enumerate(EVAL_SEEDS):
                if (m, D, s) in ev:
                    A[i, j, k] = get(ev[m, D, s]["by_nfe"][str(nfe)], metric)
    return A


def ms(x):
    return float(np.nanmean(x)), float(np.nanstd(x, ddof=1)) if np.sum(~np.isnan(x)) > 1 else float("nan")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(A, i_ref, i_other, j):
    """Seed-paired comparison (same data, same training/eval noise streams per seed)."""
    a, b = A[i_ref, j], A[i_other, j]
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = b[ok] - a[ok]
    n = len(d)
    if n < 2:
        return dict(n=n)
    se = d.std(ddof=1) / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    p = float(stats.ttest_rel(b[ok], a[ok]).pvalue) if d.std() > 0 else (0.0 if d.mean() != 0 else 1.0)
    pooled_seed_sd = math.sqrt(0.5 * (a[ok].var(ddof=1) + b[ok].var(ddof=1)))
    return dict(n=n, diff_mean=float(d.mean()), ci95=[float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
                p=p, ref_mean=float(a[ok].mean()), other_mean=float(b[ok].mean()),
                pooled_seed_sd=pooled_seed_sd,
                diff_over_seed_sd=float(d.mean() / pooled_seed_sd) if pooled_seed_sd > 0 else float("inf"),
                ratio=float(b[ok].mean() / a[ok].mean()) if a[ok].mean() > 0 else float("nan"))


def gt_jerk():
    out = {}
    for D in D_LEVELS:
        R = fm.load_split(D, "test")["R"]
        out[D] = float(fm.angular_jerk_deg(R).mean())
    return out


# ----------------------------------------------------------------------------- figures

def style(ax, xlabel, ylabel, log=False):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2)


def band_plot(ax, x, A, label_last=False):
    for i, m in enumerate(MODELS):
        mu, sd = np.nanmean(A[i], -1), np.nanstd(A[i], -1, ddof=1)
        lo = np.maximum(mu - sd, 0.5 * mu)          # keep bands finite on log axes
        ax.fill_between(x, lo, mu + sd, color=COLOR[m], alpha=0.18, lw=0)
        ax.plot(x, mu, color=COLOR[m], lw=2, marker=MARKER[m], ms=6, label=SHORT[m],
                markeredgecolor="white", markeredgewidth=1)
        if label_last:
            ax.annotate(SHORT[m].split()[0], (x[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")


def figures(ev, summary):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    x = np.array(D_LEVELS)

    # 1. headline: geodesic endpoint error vs D (mean / median / p95 / max)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, title) in zip(axs.flat, [("mean", "Mean"), ("median", "Median"), ("p95", "95th percentile"),
                                           ("max", "Max (linear axis)")]):
        band_plot(ax, x, arr(ev, f"geo_err_deg.{key}"), label_last=True)
        style(ax, "Angular displacement D (deg)", "Endpoint geodesic error (deg)", log=key != "max")
        if key == "max":
            ax.set_ylim(0, 185)
        ax.set_title(f"{title} error, NFE={HEAD_NFE}", loc="left")
        ax.set_xticks(x)
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Endpoint geodesic error vs angular displacement (bands: ±1 sd over 5 seeds)", x=0.01, ha="left",
                 color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_D.png", dpi=150)
    plt.close(fig)

    # 2. NFE sweep: one panel per D
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for ax, (j, D) in zip(axs.flat, enumerate(D_LEVELS)):
        A = np.stack([arr(ev, "geo_err_deg.mean", n)[:, j] for n in NFES], 1)   # model, nfe, seed
        band_plot(ax, np.array(NFES), A)
        ax.set_xscale("log", base=2)
        style(ax, "NFE", "Mean endpoint error (deg)", log=True)
        ax.set_title(f"D = {D}°", loc="left")
        ax.set_xticks(NFES)
        ax.set_xticklabels(NFES)
    axs.flat[-1].axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(h, l, frameon=False, loc="center")
    fig.suptitle("Mean endpoint geodesic error vs integration steps (±1 sd over 5 seeds)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "nfe_sweep.png", dpi=150)
    plt.close(fig)

    # 3. orthogonality residual (pre-projection) and 4. jerk
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    band_plot(axs[0], x, arr(ev, "orth_residual.mean"), label_last=True)
    style(axs[0], "Angular displacement D (deg)", "||RᵀR − I||_F, raw output", log=True)
    axs[0].set_title(f"Orthogonality residual before projection, NFE={HEAD_NFE}", loc="left")
    axs[0].set_xticks(x)
    band_plot(axs[1], x, arr(ev, "jerk_deg_mean"), label_last=True)
    gj = summary["gt_jerk_deg"]
    axs[1].plot(x, [gj[str(D)] if str(D) in gj else gj[D] for D in D_LEVELS], color=INK2, lw=1.5, ls="--",
                label="ground-truth test trajectories")
    style(axs[1], "Angular displacement D (deg)", "Mean angular jerk (deg / step³)", log=True)
    axs[1].set_title(f"Trajectory smoothness, NFE={HEAD_NFE}", loc="left")
    axs[1].set_xticks(x)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "orth_and_jerk.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables

def fmt(mu, sd, p=2):
    return f"{mu:.{p}f} ± {sd:.{p}f}"


def fmt_e(mu, sd):
    return f"{mu:.1e} ± {sd:.1e}"


def tables(ev, tr, summary):
    L = []
    for key, name in [("mean", "Mean"), ("median", "Median"), ("p95", "95th pct"), ("max", "Max")]:
        A = arr(ev, f"geo_err_deg.{key}")
        L.append(f"\n**{name} endpoint geodesic error (deg), NFE={HEAD_NFE}, mean ± sd over 5 seeds**\n")
        L.append("| Model | " + " | ".join(f"D={D}" for D in D_LEVELS) + " |")
        L.append("|---|" + "---|" * len(D_LEVELS))
        for i, m in enumerate(MODELS):
            L.append(f"| {SHORT[m]} | " + " | ".join(fmt(*ms(A[i, j])) for j in range(len(D_LEVELS))) + " |")
    for key, name in [("frac_err_above_10deg", "Fraction of samples with error > 10°"),
                      ("frac_err_above_45deg", "Fraction of samples with error > 45°")]:
        A = arr(ev, key)
        L.append(f"\n**{name}, NFE={HEAD_NFE}**\n")
        L.append("| Model | " + " | ".join(f"D={D}" for D in D_LEVELS) + " |")
        L.append("|---|" + "---|" * len(D_LEVELS))
        for i, m in enumerate(MODELS):
            L.append(f"| {SHORT[m]} | " + " | ".join(fmt(*ms(A[i, j]), 3) for j in range(len(D_LEVELS))) + " |")

    # paired comparisons vs M1
    for key in ["mean", "p95", "max"]:
        L.append(f"\n**Seed-paired difference vs M1 in {key} error (deg), NFE={HEAD_NFE}: diff [95% CI], "
                 f"Holm-adjusted p over the 7 D levels, |diff| / pooled seed sd**\n")
        L.append("| Model | " + " | ".join(f"D={D}" for D in D_LEVELS) + " |")
        L.append("|---|" + "---|" * len(D_LEVELS))
        for m in MODELS[1:]:
            cells = []
            for j, D in enumerate(D_LEVELS):
                c = summary["paired_vs_M1"][key][m][str(D)]
                cells.append(f"{c['diff_mean']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] "
                             f"p={c['p_holm']:.2g} ({abs(c['diff_over_seed_sd']):.1f} sd)")
            L.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")

    # NFE table (mean error) per D
    L.append("\n**Mean endpoint error (deg) vs NFE, mean ± sd over 5 seeds**\n")
    L.append("| D | Model | " + " | ".join(f"NFE={n}" for n in NFES) + " |")
    L.append("|---|---|" + "---|" * len(NFES))
    for j, D in enumerate(D_LEVELS):
        for i, m in enumerate(MODELS):
            L.append(f"| {D} | {SHORT[m]} | " + " | ".join(fmt(*ms(arr(ev, 'geo_err_deg.mean', n)[i, j]), 1)
                                                         for n in NFES) + " |")

    # ranking by NFE
    L.append("\n**Model ranking by mean error (best first), seed-averaged**\n")
    L.append("| D | " + " | ".join(f"NFE={n}" for n in NFES) + " |")
    L.append("|---|" + "---|" * len(NFES))
    for j, D in enumerate(D_LEVELS):
        cells = []
        for n in NFES:
            mu = np.nanmean(arr(ev, "geo_err_deg.mean", n)[:, j], -1)
            cells.append(" < ".join(MODELS[i][:2] for i in np.argsort(mu)))
        L.append(f"| {D} | " + " | ".join(cells) + " |")

    for key, name, f in [("orth_residual.mean", "Orthogonality residual ||RᵀR−I||_F of RAW output (mean over waypoints)", fmt_e),
                         ("orth_residual.max", "Orthogonality residual of RAW output (max over waypoints & samples)", fmt_e),
                         ("jerk_deg_mean", "Mean angular jerk (deg/step³)", fmt),
                         ("trans_err.mean", "Translation endpoint error (m)", lambda a, b: fmt(a, b, 3)),
                         ("n_nonfinite", "Non-finite outputs (count of 1000)", fmt)]:
        A = arr(ev, key)
        L.append(f"\n**{name}, NFE={HEAD_NFE}**\n")
        L.append("| Model | " + " | ".join(f"D={D}" for D in D_LEVELS) + " |")
        L.append("|---|" + "---|" * len(D_LEVELS))
        for i, m in enumerate(MODELS):
            L.append(f"| {SHORT[m]} | " + " | ".join(f(*ms(A[i, j])) for j in range(len(D_LEVELS))) + " |")
        if key == "jerk_deg_mean":
            L.append("| Ground truth (test) | " + " | ".join(f"{summary['gt_jerk_deg'][str(D)]:.2f}"
                                                          for D in D_LEVELS) + " |")
    L.append("\n**Orthogonality residual of RAW output (mean) vs NFE, D=175**\n")
    L.append("| Model | " + " | ".join(f"NFE={n}" for n in NFES) + " |")
    L.append("|---|" + "---|" * len(NFES))
    for i, m in enumerate(MODELS):
        L.append(f"| {SHORT[m]} | " + " | ".join(fmt_e(*ms(arr(ev, 'orth_residual.mean', n)[i, -1])) for n in NFES) + " |")

    L.append("\n**Training (in-sweep, 8 parallel workers on 4P+6E cores) wall-clock seconds, mean ± sd over 35 runs**\n")
    L.append("| Model | params | train s |")
    L.append("|---|---|---|")
    for m in MODELS:
        secs = [tr[k]["train_seconds"] for k in tr if k[0] == m]
        L.append(f"| {SHORT[m]} | {tr[next(k for k in tr if k[0] == m)]['n_params']:,} | {fmt(np.mean(secs), np.std(secs, ddof=1), 0)} |")
    (RESULTS / "summary_tables.md").write_text("\n".join(L) + "\n")


def main():
    ev, tr, missing = load()
    summary = dict(head_nfe=HEAD_NFE, n_runs=len(ev), missing=missing)
    summary["gt_jerk_deg"] = {str(k): v for k, v in gt_jerk().items()}
    summary["paired_vs_M1"] = {}
    for key in ["mean", "median", "p95", "max"]:
        A = arr(ev, f"geo_err_deg.{key}")
        summary["paired_vs_M1"][key] = {}
        for i, m in enumerate(MODELS[1:], start=1):
            cs = {str(D): paired(A, 0, i, j) for j, D in enumerate(D_LEVELS)}
            ph = holm([cs[str(D)].get("p", 1.0) for D in D_LEVELS])
            for D, p in zip(D_LEVELS, ph):
                cs[str(D)]["p_holm"] = float(p)
            summary["paired_vs_M1"][key][m] = cs
    # per-NFE paired M2 vs M1 on mean error
    summary["paired_M2_vs_M1_by_nfe"] = {
        str(n): {str(D): paired(arr(ev, "geo_err_deg.mean", n), 0, 1, j) for j, D in enumerate(D_LEVELS)}
        for n in NFES}
    summary["agg"] = {}
    for key in ["geo_err_deg.mean", "geo_err_deg.median", "geo_err_deg.p95", "geo_err_deg.max",
                "frac_err_above_10deg", "frac_err_above_45deg", "orth_residual.mean", "orth_residual.max",
                "jerk_deg_mean", "trans_err.mean", "n_nonfinite"]:
        summary["agg"][key] = {str(n): {m: {str(D): ms(arr(ev, key, n)[i, j]) for j, D in enumerate(D_LEVELS)}
                                        for i, m in enumerate(MODELS)} for n in NFES}
    write_json(RESULTS / "summary.json", summary)
    if ev:
        tables(ev, tr, summary)
        figures(ev, summary)
    print(f"runs complete: {len(ev)}/140; missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
