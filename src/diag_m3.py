"""Phase E (night 2, optional): why is M3 (canonical quaternion) worse at EVERY D?

Candidate mechanisms, each tested on the night-1 main M3 models with their exact evaluation samples (NFE 64):
  (a) source/target pairing in R^4: canonicalization (w >= 0) puts q0 and q1 in one hemisphere but does
      NOT make q0.q1 >= 0. When q0.q1 < 0 the straight chord q0 -> q1 passes close to the origin
      (min |q_t| = cos(alpha/2), alpha = angle(q0, q1) in R^4) and traverses the LONG way round in
      rotation space (rotation-angle swept = 2 alpha > 180 deg for alpha > 90). Independent of D because
      the source is Haar -> predicts an error elevated at every D, driven by alpha, not by D.
  (b) normalization at the end: the raw final |q| is off by 10-20% (night 1). If normalization were
      the problem, error would track | |q| - 1 |; direction error is what normalization cannot fix.
  (c) the Euclidean-to-rotation gain: a perpendicular error e in R^4 costs 2e/|q| rad of rotation,
      vs ~e for a 6D column. A constant gain would raise error uniformly but not create alpha-dependence.
Output: results/n2/diag_m3.json
"""
import numpy as np
import torch

import fm
import geometry as G
from common import EVAL_SEEDS, RESULTS, RUNS_DIR, write_json
from flows import get_rep

DS = [10, 30, 90, 150, 175]
ALPHA_BINS = [0, 30, 60, 90, 120, 150, 180.01]
NFE, K = 64, 5


def main():
    torch.set_num_threads(2)
    out = {"alpha_bins_deg": ALPHA_BINS, "by_model": {}}
    for m in ["M3_quat", "M2_euclid6d", "M1_riemannian"]:
        rep = get_rep(m)
        rows = {k: [] for k in ("err", "alpha", "normdev", "D", "minnorm")}
        for D in DS:
            data = fm.load_split(D, "test")
            cond = data["cond"].repeat_interleave(K, 0)
            R1 = data["R"][:, -1].repeat_interleave(K, 0)
            q1 = G.matrix_to_quat(R1)
            for s in EVAL_SEEDS:
                net = fm.load_net(m, RUNS_DIR / "main" / f"{m}_D{D}_s{s}")
                R0n, p0n = fm.eval_noise(D, s, cond.shape[0])
                q0 = G.matrix_to_quat(R0n[:, -1])
                # integrate, tracking |q_t| of the last waypoint (M3 only)
                x, p, h, mins = rep.encode(R0n), p0n, 1.0 / NFE, torch.full((cond.shape[0],), np.inf, dtype=torch.float64)
                with torch.no_grad():
                    for k in range(NFE):
                        t = torch.full((cond.shape[0],), k * h, dtype=torch.float64)
                        v, vp = fm.split_output(rep, x, net(fm.net_input(rep, x, p, cond, t)))
                        x, p = rep.step(x, v, h), p + h * vp
                        if m == "M3_quat":
                            mins = torch.minimum(mins, x[:, -1].norm(dim=-1))
                err = torch.rad2deg(G.geodesic_distance(rep.to_matrix(x)[:, -1], R1))
                rows["err"].append(err.numpy())
                rows["alpha"].append(torch.rad2deg(torch.arccos((q0 * q1).sum(-1).clamp(-1, 1))).numpy())
                rows["normdev"].append(((x[:, -1].norm(dim=-1) - 1).abs() if m == "M3_quat"
                                        else torch.zeros(cond.shape[0], dtype=torch.float64)).numpy())
                rows["minnorm"].append(mins.numpy())
                rows["D"].append(np.full(cond.shape[0], D))
        r = {k: np.concatenate(v) for k, v in rows.items()}
        bins = []
        for lo, hi in zip(ALPHA_BINS[:-1], ALPHA_BINS[1:]):
            sel = (r["alpha"] >= lo) & (r["alpha"] < hi)
            bins.append(dict(bin=[lo, hi], n=int(sel.sum()), share=float(sel.mean()),
                             mean_err=float(r["err"][sel].mean()), median_err=float(np.median(r["err"][sel])),
                             by_D={str(D): float(r["err"][sel & (r["D"] == D)].mean()) if (sel & (r["D"] == D)).any()
                                   else None for D in DS}))
        neg = r["alpha"] > 90
        rec = dict(alpha_bins=bins, corr_err_alpha=float(np.corrcoef(r["err"], r["alpha"])[0, 1]),
                   frac_pairs_negative_dot=float(neg.mean()),
                   mean_err_dot_pos=float(r["err"][~neg].mean()), mean_err_dot_neg=float(r["err"][neg].mean()),
                   by_D={str(D): dict(all=float(r["err"][r["D"] == D].mean()),
                                      dot_pos=float(r["err"][(r["D"] == D) & ~neg].mean()),
                                      dot_neg=float(r["err"][(r["D"] == D) & neg].mean())) for D in DS})
        if m == "M3_quat":
            rec["corr_err_normdev"] = float(np.corrcoef(r["err"], r["normdev"])[0, 1])
            rec["corr_err_minnorm"] = float(np.corrcoef(r["err"], r["minnorm"])[0, 1])
            q = np.quantile(r["minnorm"], [0, 0.25, 0.5, 0.75, 1])
            rec["err_by_minnorm_quartile"] = [dict(range=[float(a), float(b)],
                                                   mean_err=float(r["err"][(r["minnorm"] >= a) & (r["minnorm"] <= b)].mean()))
                                              for a, b in zip(q[:-1], q[1:])]
        out["by_model"][m] = rec
        write_json(RESULTS / "n2" / "diag_m3.json", out)
        print(m, "corr(err, alpha)", round(rec["corr_err_alpha"], 3), "| dot>0:", round(rec["mean_err_dot_pos"], 2),
              "dot<0:", round(rec["mean_err_dot_neg"], 2), "| share dot<0", round(rec["frac_pairs_negative_dot"], 3))
        for b in bins:
            print("   alpha", b["bin"], b["n"], round(b["mean_err"], 2), {k: (round(v, 1) if v else v) for k, v in b["by_D"].items()})


if __name__ == "__main__":
    main()
