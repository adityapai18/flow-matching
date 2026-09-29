"""Clean wall-clock measurements, run ALONE (no other jobs), 1 CPU thread.

- Training: ms per optimizer step (incl. CPU float64 geometry), per model.
- Inference: seconds per trajectory at each NFE, batched (1000) and single (batch 1),
  using the trained main-sweep models at D=90 for seeds 0-4 (mean ± sd over seeds).
"""
import shutil
import tempfile
import time

import numpy as np
import torch

import fm
from common import EVAL_SEEDS, MODELS, NFES, RESULTS, RUNS_DIR, get_logger, read_json, write_json
from flows import get_rep

D_TIME = 90
OUT = RESULTS / "timing.json"


def train_ms_per_step(model, reps=3, steps=1000):
    out = []
    for r in range(reps):
        d = tempfile.mkdtemp()
        try:
            fm.train_one(model, D_TIME, 1000 + r, 1e-3, 200, d, ckpt_every=10**9)       # warm-up
            shutil.rmtree(d)
            t0 = time.perf_counter()
            fm.train_one(model, D_TIME, 1000 + r, 1e-3, steps, d, ckpt_every=10**9)
            out.append(1000 * (time.perf_counter() - t0) / steps)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return out


def infer_seconds(model, seed, nfe, batch):
    rep = get_rep(model)
    net = fm.load_net(model, RUNS_DIR / "main" / f"{model}_D{D_TIME}_s{seed}")
    data = fm.load_split(D_TIME, "test")
    cond = data["cond"][:batch] if batch <= len(data["cond"]) else data["cond"].repeat(batch // len(data["cond"]) + 1, 1)[:batch]
    R0, p0 = fm.eval_noise(D_TIME, seed, batch)
    x0 = rep.encode(R0)
    fm.integrate(rep, net, cond, x0, p0, nfe)                                          # warm-up
    reps = 3 if batch > 1 else 20
    t0 = time.perf_counter()
    for _ in range(reps):
        x, p = fm.integrate(rep, net, cond, x0, p0, nfe)
        rep.to_matrix(x)
    return (time.perf_counter() - t0) / reps / batch


def main():
    torch.set_num_threads(1)
    log = get_logger()
    res = read_json(OUT, default={"note": "1 CPU thread, run alone", "D": D_TIME, "train_ms_per_step": {},
                                  "infer_s_per_traj": {}})
    for m in MODELS:
        if m not in res["train_ms_per_step"]:
            v = train_ms_per_step(m)
            res["train_ms_per_step"][m] = dict(values=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
            write_json(OUT, res)
            log.info(f"[timing] train {m}: {np.mean(v):.2f} ms/step")
    for m in MODELS:
        res["infer_s_per_traj"].setdefault(m, {})
        for batch in (1000, 1):
            for nfe in NFES:
                key = f"b{batch}_nfe{nfe}"
                if key in res["infer_s_per_traj"][m]:
                    continue
                v = [infer_seconds(m, s, nfe, batch) for s in EVAL_SEEDS]
                res["infer_s_per_traj"][m][key] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
                write_json(OUT, res)
        log.info(f"[timing] infer {m} done")


if __name__ == "__main__":
    main()
