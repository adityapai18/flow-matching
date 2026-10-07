"""Night-3 Phase B mechanism table (no training). d0 = geodesic distance from the ENDPOINT-waypoint source
sample to the target endpoint (night-2 definition), using the exact evaluation sources (fm.eval_noise),
test split, 5 samples/start, seeds 0-2. Output: results/n3/d0_mechanism.json"""
import numpy as np
import torch
import fm
import geometry as G
from common import write_json
from night2 import HAAR, P3, gauss
from night3 import N3

torch.set_num_threads(1)
K, SEEDS = 5, [0, 1, 2]
out = {}
for D in [30, 90, 175]:
    data = fm.load_split(D, "test")
    n = data["cond"].shape[0]
    Rc = data["R"][:, 0].repeat_interleave(K, 0)
    R1 = data["R"][:, -1].repeat_interleave(K, 0)
    Rt = data["R"].repeat_interleave(K, 0)
    for name, prior in [("haar", HAAR), ("trunc150", P3)] + [(f"gauss{s}", gauss(s)) for s in (15, 30, 60, 90)]:
        d0 = []
        for s in SEEDS:
            R0n, _ = fm.eval_noise(D, s, n * K, prior, Rc, Rt)
            d0.append(torch.rad2deg(G.geodesic_distance(R0n[:, -1], R1)).numpy())
        d0 = np.concatenate(d0)
        out.setdefault(name, {})[str(D)] = dict(mean=float(d0.mean()), median=float(np.median(d0)), sd=float(d0.std()),
                                                p05=float(np.percentile(d0, 5)), p95=float(np.percentile(d0, 95)),
                                                gt150=float((d0 > 150).mean()), gt165=float((d0 > 165).mean()),
                                                lt90=float((d0 < 90).mean()), n=int(d0.size))
for name, v in out.items():
    print(f"{name:9s}", "  ".join(f"D={D}: mean {r['mean']:6.1f} [p5 {r['p05']:5.1f}, p95 {r['p95']:5.1f}] >150 {r['gt150']:.3f} >165 {r['gt165']:.3f} <90 {r['lt90']:.3f}" for D, r in v.items()))
write_json(N3 / "d0_mechanism.json", out)
