"""Diagnostic for the catastrophic-error tail ("what broke"), on the main-sweep models.

Hypothesis: M1's rare catastrophic endpoint errors come from source samples that
start near the cut locus of the target (initial geodesic distance ~ 180 deg), where
the geodesic conditional velocity is discontinuous (axis flips sign). For the same
source samples, report endpoint error binned by initial source-to-target distance.
Uses the exact evaluation source samples (paired across models). NFE = 64.
"""
import numpy as np
import torch

import fm
import geometry as G
from common import D_LEVELS, EVAL_SEEDS, MODELS, RESULTS, RUNS_DIR, write_json
from flows import get_rep

BINS = [0, 90, 120, 150, 165, 175, 180.01]
NFE = 64
K = 5


def main():
    torch.set_num_threads(4)
    out = {"bins_deg": BINS, "nfe": NFE, "by_model": {}}
    for m in MODELS:
        rep = get_rep(m)
        errs, dists = [], []
        per_D = {}
        for D in D_LEVELS:
            e_D, d_D = [], []
            for s in EVAL_SEEDS:
                rd = RUNS_DIR / "main" / f"{m}_D{D}_s{s}"
                if not (rd / "model.pt").exists():
                    continue
                net = fm.load_net(m, rd)
                data = fm.load_split(D, "test")
                cond = data["cond"].repeat_interleave(K, 0)
                R1 = data["R"][:, -1].repeat_interleave(K, 0)
                R0n, p0n = fm.eval_noise(D, s, cond.shape[0])
                x, _ = fm.integrate(rep, net, cond, rep.encode(R0n), p0n, NFE)
                e = torch.rad2deg(G.geodesic_distance(rep.to_matrix(x)[:, -1], R1))
                d0 = torch.rad2deg(G.geodesic_distance(R0n[:, -1], R1))
                e_D.append(e.numpy()); d_D.append(d0.numpy())
            if e_D:
                e_D, d_D = np.concatenate(e_D), np.concatenate(d_D)
                per_D[str(D)] = dict(frac_gt45=float((e_D > 45).mean()),
                                     frac_gt45_given_d0_gt165=float((e_D[d_D > 165] > 45).mean()),
                                     frac_gt45_given_d0_le165=float((e_D[d_D <= 165] > 45).mean()))
                errs.append(e_D); dists.append(d_D)
        e, d = np.concatenate(errs), np.concatenate(dists)
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = (d >= lo) & (d < hi)
            rows.append(dict(bin=[lo, hi], n=int(sel.sum()),
                             mean_err=float(e[sel].mean()) if sel.any() else None,
                             median_err=float(np.median(e[sel])) if sel.any() else None,
                             frac_gt45=float((e[sel] > 45).mean()) if sel.any() else None))
        out["by_model"][m] = dict(bins=rows, per_D=per_D,
                                  corr_err_vs_d0=float(np.corrcoef(e, d)[0, 1]))
        write_json(RESULTS / "diag_tail.json", out)
        print(m, [(r["bin"], r["n"], None if r["frac_gt45"] is None else round(r["frac_gt45"], 3)) for r in rows])


if __name__ == "__main__":
    main()
