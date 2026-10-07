"""Night-3 Phase A analysis (pre-registered in results/n3/prereg_phaseA.json). Runs on partial data.

Gap g(S) = mean_seeds(err_M2 - err_M1), D=150, NFE 64, seeds 0-2 (positive = M1 better).
Budgets 20k..320k; sources:
  gauss90 : 20k n2_main_gauss90 | 40k n2_b1_gauss90_steps40000 | 80k n2_b1_gauss90_steps80000 | 160k/320k n3_A_gauss90_*
  trunc150: 20k n2_main_trunc150 | 40k n3_A_trunc150_steps40000 | 80k n2_b1x_trunc150_steps80000 | 160k/320k n3_A_trunc150_*
Output: results/n3/phaseA.json
"""
import math

import numpy as np
from scipy import stats

from common import RUNS_DIR, read_json, write_json
from night3 import N3

D, SEEDS, MODELS = 150, [0, 1, 2], ["M1_riemannian", "M2_euclid6d"]
BUDGETS = [20000, 40000, 80000, 160000, 320000]
_TAGS = {
    "trunc150": {20000: "n2_main_trunc150", 40000: "n3_A_trunc150_steps40000", 80000: "n2_b1x_trunc150_steps80000",
                 160000: "n3_A_trunc150_steps160000", 320000: "n3_A_trunc150_steps320000"},
    "gauss90": {20000: "n2_main_gauss90", 40000: "n2_b1_gauss90_steps40000", 80000: "n2_b1_gauss90_steps80000",
                160000: "n3_A_gauss90_steps160000", 320000: "n3_A_gauss90_steps320000"},
}
TAGS = {p: {S: {m: t for m in ["M1_riemannian", "M2_euclid6d"]} for S, t in v.items()} for p, v in _TAGS.items()}
# matched-LR sensitivity arm (prereg addendum_1): protocol M1 (1e-3) vs M2 at 1e-3, P3
TAGS["trunc150_matchedLR"] = {S: {"M1_riemannian": _TAGS["trunc150"][S],
                                  "M2_euclid6d": f"n3_A_trunc150_M2lr1e-3_steps{S}"} for S in BUDGETS}
D0_BINS = [(0, 90), (90, 120), (120, 150), (150, 165), (165, 181)]
NBOOT = 10000
RNG = np.random.default_rng(0)


def run(tag, m, s):
    rd = RUNS_DIR / tag / f"{m}_D{D}_s{s}"
    e, t = read_json(rd / "eval_test.json"), read_json(rd / "train.json")
    if not (e and t and t.get("done") and "64" in e["by_nfe"]):
        return None
    L = np.array([h[1] for h in t["hist"]])
    n = len(L)
    q3, q4 = L[n // 2: 3 * n // 4].mean(), L[3 * n // 4:].mean()
    out = dict(err=e["by_nfe"]["64"]["geo_err_deg"]["mean"], final_loss=float(L[-1]),
               last_quarter_drop=float((q3 - q4) / q3), loss_max=float(L[2:].max()))
    smp = rd / "eval_test_samples.npz"
    if smp.exists():
        z = np.load(smp)
        out["d0_strata"] = [float(z["err_deg"][(z["d0_deg"] >= a) & (z["d0_deg"] < b)].mean())
                            if ((z["d0_deg"] >= a) & (z["d0_deg"] < b)).any() else None for a, b in D0_BINS]
    return out


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    n = len(x)
    if n < 3:
        return a, b, (np.nan, np.nan)
    resid = y - (a + b * x)
    se = math.sqrt(resid @ resid / (n - 2) / ((x - x.mean()) @ (x - x.mean())))
    tc = stats.t.ppf(0.975, n - 2)
    return a, b, (b - tc * se, b + tc * se)


def diverged_seeds():
    st = read_json(N3 / "stability.json") or []
    return {(r["tag"], r["model"], r["seed"]) for r in st if r["diverged"]}


def analyze(prior, drop_diverged=False):
    R = {S: {m: [run(TAGS[prior][S][m], m, s) for s in SEEDS] for m in MODELS} for S in BUDGETS}
    have = [S for S in BUDGETS if all(r is not None for m in MODELS for r in R[S][m])]
    if not have:
        return dict(budgets=[], table=[])
    div = diverged_seeds()
    # seed pairs to keep per budget (drop the pair if either model's run diverged)
    keep = {S: [s for s in SEEDS if not (drop_diverged and any((TAGS[prior][S][m], m, s) in div for m in MODELS))]
            for S in have}
    E = {m: np.array([[R[S][m][i]["err"] if i in keep[S] else np.nan for i in range(3)] for S in have]) for m in MODELS}
    G = E["M2_euclid6d"] - E["M1_riemannian"]
    x = np.log2(np.array(have) / 1000.0)
    res = dict(budgets=have, table=[])
    res["seeds_used"] = {S: keep[S] for S in have}
    for k, S in enumerate(have):
        g = G[k][~np.isnan(G[k])]
        e1, e2 = (E[m][k][~np.isnan(E[m][k])] for m in MODELS)
        n = len(g)
        if n == 0:
            res["table"].append(dict(steps=S, n=0, note="no usable seed pairs (all diverged)"))
            continue
        if n >= 2:
            se = g.std(ddof=1) / math.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            ci = [float(g.mean() - tc * se), float(g.mean() + tc * se)]
            psd = math.sqrt(0.5 * (e1.var(ddof=1) + e2.var(ddof=1)))
        else:
            ci, psd = [None, None], 0
        row = dict(steps=S, n=n, M1=[float(e1.mean()), float(e1.std(ddof=1)) if n > 1 else None],
                   M2=[float(e2.mean()), float(e2.std(ddof=1)) if n > 1 else None],
                   gap=float(g.mean()), gap_ci=ci, gap_sd_units=float(g.mean() / psd) if psd > 0 else None)
        for m in MODELS:
            rr = [R[S][m][i] for i in keep[S]]
            row[f"{m[:2]}_final_loss"] = float(np.mean([r["final_loss"] for r in rr]))
            row[f"{m[:2]}_last_quarter_drop"] = float(np.mean([r["last_quarter_drop"] for r in rr]))
            row[f"{m[:2]}_loss_max_over_final"] = float(np.max([r["loss_max"] / r["final_loss"] for r in rr]))
            if all("d0_strata" in r for r in rr):
                row[f"{m[:2]}_d0_strata"] = [float(np.mean([r["d0_strata"][b] for r in rr]))
                                            if all(r["d0_strata"][b] is not None for r in rr) else None
                                            for b in range(len(D0_BINS))]
        res["table"].append(row)
    if len(have) < 2 or any(len(keep[S]) < 2 for S in have):
        res["note"] = "fits/ratios skipped: < 2 budgets or a budget with < 2 usable seed pairs"
        return res
    # --- per-doubling ratios, bootstrap over seeds jointly across budgets
    idx = RNG.integers(0, 3, size=(NBOOT, 3))
    Gb = np.nanmean(G[:, idx], -1)                                            # budget x boot
    ratios = []
    for k in range(len(have) - 1):
        if have[k + 1] != 2 * have[k]:
            continue
        r = np.nanmean(G[k + 1]) / np.nanmean(G[k])
        rb = Gb[k + 1] / Gb[k]
        ratios.append(dict(frm=have[k], to=have[k + 1], ratio=float(r),
                           ci=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]))
    res["per_doubling"] = ratios
    # --- observed sign change of the gap between consecutive budgets (log2-linear interpolation)
    def crossing(gm):
        for k in range(len(gm) - 1):
            if np.sign(gm[k]) != np.sign(gm[k + 1]) and gm[k] != 0:
                f = gm[k] / (gm[k] - gm[k + 1])
                return float(1000 * 2 ** (x[k] + f * (x[k + 1] - x[k])))
        return None
    cpt = crossing(np.nanmean(G, -1))
    cb = [crossing(Gb[:, i]) for i in range(NBOOT)]
    cbf = [c for c in cb if c is not None]
    res["observed_crossing"] = dict(steps=cpt, frac_boot_with_crossing=float(len(cbf) / NBOOT),
                                    ci=[float(np.percentile(cbf, 2.5)), float(np.percentile(cbf, 97.5))] if len(cbf) > 20 else None)
    # --- brief's fit: gap vs log2(steps), all budget x seed points
    xs = np.repeat(x, 3)
    ok = ~np.isnan(G.ravel())
    a, b, ci = ols(xs[ok], G.ravel()[ok])
    bb = [np.polyfit(x, Gb[:, i], 1)[0] for i in range(NBOOT) if not np.isnan(Gb[:, i]).any()]
    x1m = math.log2(1000.0)
    ext = [np.polyval(np.polyfit(x, Gb[:, i], 1), x1m) for i in range(2000) if not np.isnan(Gb[:, i]).any()]
    res["gap_fit"] = dict(intercept=float(a), slope_per_doubling=float(b), slope_ci_t=[float(ci[0]), float(ci[1])],
                          slope_ci_boot=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                          gap_at_1M=float(a + b * x1m),
                          gap_at_1M_ci_boot=[float(np.percentile(ext, 2.5)), float(np.percentile(ext, 97.5))],
                          zero_crossing_steps=float(1000 * 2 ** (-a / b)) if b != 0 and -a / b > x.max() else None)
    # --- power law on |g| (log|g| vs log2 S): pure power law implies a zero asymptote
    gm = np.nanmean(G, -1)
    if np.all(gm * np.sign(gm.mean()) > 0):
        lg = np.log(np.abs(gm))
        pa, pb, pci = ols(x, lg)
        res["abs_gap_powerlaw"] = dict(exponent_per_doubling=float(pb), ratio_per_doubling=float(math.exp(pb)),
                                       ratio_ci_t=[float(math.exp(pci[0])), float(math.exp(pci[1]))])
    # --- each model's own curve vs log2(steps); intersection
    fits = {m: ols(xs[~np.isnan(E[m].ravel())], E[m].ravel()[~np.isnan(E[m].ravel())]) for m in MODELS}
    a1, b1, _ = fits["M1_riemannian"]
    a2, b2, _ = fits["M2_euclid6d"]

    def meet(a1, b1, a2, b2):
        if b1 == b2:
            return None
        xm = (a2 - a1) / (b1 - b2)
        return xm if xm > x.max() else None

    xm = meet(a1, b1, a2, b2)
    boots = []
    for i in range(NBOOT):
        y1, y2 = np.nanmean(E["M1_riemannian"][:, idx[i]], -1), np.nanmean(E["M2_euclid6d"][:, idx[i]], -1)
        if np.isnan(y1).any() or np.isnan(y2).any():
            continue
        f1, f2 = np.polyfit(x, y1, 1), np.polyfit(x, y2, 1)
        boots.append(meet(f1[1], f1[0], f2[1], f2[0]))
    fwd = [v for v in boots if v is not None]
    res["own_curves"] = dict(
        M1=dict(intercept=float(a1), slope_per_doubling=float(b1), slope_ci=list(map(float, fits["M1_riemannian"][2]))),
        M2=dict(intercept=float(a2), slope_per_doubling=float(b2), slope_ci=list(map(float, fits["M2_euclid6d"][2]))),
        meet_steps=float(1000 * 2 ** xm) if xm is not None else None,
        frac_boot_meeting_ahead=float(len(fwd) / max(1, len(boots))),
        meet_steps_ci=[float(1000 * 2 ** np.percentile(fwd, 2.5)), float(1000 * 2 ** np.percentile(fwd, 97.5))] if len(fwd) > 20 else None)
    return res


def gate(res):
    """Pre-registered rule (prereg_phaseA.json), applied to P3."""
    g320 = next((r for r in res["table"] if r["steps"] == 320000), None)
    if g320 is not None and g320.get("n", 0) == 0:
        return "(c) UNDECIDABLE (no usable seed pairs at 320k: every M2 run diverged)"
    pd = {r["frm"]: r for r in res.get("per_doubling", [])}
    if 160000 not in pd or 80000 not in pd or g320 is None:
        return "pending (needs the full 20k-320k series)"
    r80, r160 = pd[80000], pd[160000]
    early_pos = all(r["gap"] > 0 for r in res["table"] if r["steps"] <= 80000)
    if g320["gap_ci"][0] is None:
        return "(c) UNDECIDABLE (fewer than 2 usable seed pairs at 320k)"
    if r160["ci"][0] > 0.75 and g320["gap_ci"][0] > 0:
        return "(a) ASYMPTOTIC"
    if (r80["ratio"] <= 0.6 and r160["ratio"] <= 0.6 and r160["ci"][1] < 0.75) or \
            (early_pos and g320["gap_ci"][0] <= 0):
        return "(b) CONVERGENCE-RATE"
    return "(c) UNDECIDABLE"


if __name__ == "__main__":
    out = {p: analyze(p) for p in TAGS}
    out.update({f"{p}__without_diverged": analyze(p, True) for p in TAGS})
    out["gate"] = {
        "(i) protocol P3, WITH diverged runs": gate(out["trunc150"]),
        "(i) protocol P3, WITHOUT diverged runs": gate(out["trunc150__without_diverged"]),
        "(ii) matched-LR P3 (post-hoc sensitivity), WITH diverged": gate(out["trunc150_matchedLR"]),
        "(ii) matched-LR P3 (post-hoc sensitivity), WITHOUT diverged": gate(out["trunc150_matchedLR__without_diverged"]),
        "secondary P2, WITH diverged": gate(out["gauss90"]),
        "secondary P2, WITHOUT diverged": gate(out["gauss90__without_diverged"])}
    write_json(N3 / "phaseA.json", out)
    for p in [k for k in out if k != "gate"]:
        print(f"\n== {p}  seeds used: {out[p].get('seeds_used')}")
        for r in out[p]["table"]:
            if r.get("n", 1) == 0:
                print(f"  {r['steps']//1000:4d}k n=0  {r['note']}")
                continue
            print(f"  {r['steps']//1000:4d}k n={r['n']} M1 {r['M1'][0]:6.3f}  M2 {r['M2'][0]:6.3f}  "
                  f"gap {r['gap']:+.3f} {r['gap_ci']}  "
                  f"lossdrop M1 {r['M1_last_quarter_drop']:.3f} M2 {r['M2_last_quarter_drop']:.3f}")
        for r in out[p].get("per_doubling", []):
            print(f"  ratio {r['frm']//1000}k->{r['to']//1000}k: {r['ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
        if "gap_fit" in out[p]:
            gf = out[p]["gap_fit"]
            print(f"  gap fit: slope/doubling {gf['slope_per_doubling']:+.3f} t-CI {gf['slope_ci_t']} boot {gf['slope_ci_boot']}; "
                  f"gap@1M {gf['gap_at_1M']:+.3f} {gf['gap_at_1M_ci_boot']}; zero crossing {gf['zero_crossing_steps']}")
            print(f"  own curves: {out[p]['own_curves']}")
            print(f"  |g| power law: {out[p].get('abs_gap_powerlaw')}")
            print(f"  observed crossing: {out[p].get('observed_crossing')}")
    print("\nGATE:")
    for k, v in out["gate"].items():
        print(f"  {k}: {v}")
