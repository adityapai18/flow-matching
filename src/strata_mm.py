"""Phase C d0-stratification. evaluate_mm saves per-sample nearest-mode error and hit flags but not d0;
the evaluation sources are deterministic (fm.eval_noise), so d0 = geodesic distance from the endpoint
source sample to its NEAREST mode is recomputed exactly here (no network needed). Output: results/n2/mm_strata.json"""
import json
import numpy as np
import torch
import fm
import geometry as G
from common import MODELS, RUNS_DIR, write_json
from night2 import N2

TAG, DS, SEEDS, K = "n2_mm_gauss90", [30, 150], range(5), 30
out = {}
for D in DS:
    data = fm.load_split(D, "test", f"MM_D{D}")
    n = data["cond"].shape[0]
    prior = read = json.load(open(RUNS_DIR / TAG / f"M1_riemannian_D{D}_s0" / "eval_mm_test.json"))["prior"]
    for s in SEEDS:
        R0n, _ = fm.eval_noise(D, s, n * K, prior, data["R"][:, 0].repeat_interleave(K, 0))
        modes = data["modes_R1"].repeat_interleave(K, 0)
        d0 = torch.rad2deg(G.geodesic_distance(R0n[:, -1, None].expand(-1, 3, 3, 3), modes)).min(-1).values.numpy()
        for m in MODELS:
            z = np.load(RUNS_DIR / TAG / f"{m}_D{D}_s{s}" / "eval_mm_test_samples.npz")
            near, hit = z["nearest"], z["hit"]
            for lab, sel in [("d0<150", d0 < 150), ("d0>=150", d0 >= 150), ("all", d0 >= 0)]:
                out.setdefault(f"D{D}", {}).setdefault(lab, {}).setdefault(m, []).append(
                    [float(near[sel].mean()), float(np.median(near[sel])), float(hit[sel].mean()), float(sel.mean())])
for D, v in out.items():
    for lab, mm in v.items():
        print(D, f"{lab:8s}", "  ".join(f"{m[:2]} mean {np.mean([x[0] for x in r]):5.2f} med {np.mean([x[1] for x in r]):5.2f} on-mode {np.mean([x[2] for x in r]):.3f}" for m, r in mm.items()),
              f" share {np.mean([x[3] for x in mm['M1_riemannian']]):.2f}")
write_json(N2 / "mm_strata.json", out)
