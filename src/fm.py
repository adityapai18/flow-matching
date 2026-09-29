"""Stage 2/3 core: conditional flow matching training, ODE sampling, metrics.

Identical for all four models: MLP (4 hidden layers x 256, SiLU), Adam,
warmup + cosine LR schedule, batch size, step budget, conditioning (start pose
as flattened R0 (9) + scaled p0 (3)), timestep input, translation handling,
and the source distribution (Haar rotations, N(0, I) scaled translations).
Only the rotation parameterization / interpolation path (flows.py) differs.

Precision: every geometric operation (noise sampling, representation
conversion, interpolation, log/exp, ODE state updates, metrics) is CPU float64.
Only the network forward/backward runs in float32.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import geometry as G
from common import DATA_DIR, N_WAYPOINTS, get_logger, read_json, seed_everything, write_json
from flows import get_rep

T = N_WAYPOINTS
P_SCALE = 0.3          # translations are divided by this inside the model (identical for all)
HIDDEN = 1024          # spec said 256; see FINDINGS "Deviations" (capacity diagnostic)
N_HIDDEN_LAYERS = 4
BATCH = 256
WARMUP = 500
CLIP = 1.0
f64 = torch.float64


# ----------------------------------------------------------------------------- model

class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=HIDDEN, depth=N_HIDDEN_LAYERS):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_net(rep, width=HIDDEN, depth=N_HIDDEN_LAYERS):
    d_in = T * rep.feat_dim + T * 3 + 12 + 1
    d_out = T * (rep.out_dim + 3)
    return MLP(d_in, d_out, width, depth)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------- data

def load_split(D, split, dataset=None):
    """dataset: npz stem; default f"D{D}" (night-1 unimodal data). "MM_D{D}" = night-2 multimodal."""
    z = np.load(DATA_DIR / f"{dataset or f'D{D}'}.npz")
    idx = z[f"{split}_idx"]
    R = torch.from_numpy(z["R"][idx])                      # (n, T, 3, 3) float64
    p = torch.from_numpy(z["p"][idx]) / P_SCALE            # (n, T, 3)
    cond = torch.cat([R[:, 0].flatten(-2), p[:, 0]], -1)    # (n, 12)
    out = dict(R=R, p=p, cond=cond, path_type=torch.from_numpy(z["path_type"][idx]))
    if "modes_R1" in z:                                      # multimodal: all K valid endpoint rotations
        out["modes_R1"] = torch.from_numpy(z["modes_R1"][idx])
        out["mode"] = torch.from_numpy(z["mode"][idx])
    return out


def net_input(rep, x, p, cond, t):
    """float64 -> float32 network input. t: (B,)"""
    B = cond.shape[0]
    return torch.cat([rep.features(x).reshape(B, -1), p.reshape(B, -1), cond, t[:, None]], -1).float()


def split_output(rep, x, out):
    """Network output -> (rotation state velocity, translation velocity), CPU float64.
    Differentiable (used in the training loss as well as for sampling)."""
    B = out.shape[0]
    out = out.cpu().double()                  # cpu first: MPS has no float64
    rot = out[:, :T * rep.out_dim].reshape(B, T, rep.out_dim)
    return rep.velocity(x, rot), out[:, T * rep.out_dim:].reshape(B, T, 3)


HAAR = {"kind": "haar"}


def prior_name(prior):
    prior = prior or HAAR
    return {"haar": lambda: "haar", "gauss": lambda: f"gauss{prior['sigma_deg']:g}",
            "haar_trunc": lambda: f"trunc{prior['max_deg']:g}"}[prior["kind"]]()


def sample_rotations(prior, n, g, R_cond=None, R_tgt=None):
    """Source rotations (n, T, 3, 3), float64, for night-2's three prior families. The SAME
    rotations are encoded into every model's representation (common random numbers).
      haar        night-1 prior; draws are bit-identical to night 1.
      gauss       R_cond exp(hat(w)), w ~ N(0, sigma^2 I_3) i.i.d. per waypoint; R_cond (n,3,3)
                  is the conditioning start pose. Legal at inference (start pose is known).
      haar_trunc  Haar, rejection-resampled until d(source_k, target_k) <= max_deg for every
                  waypoint k. Needs the TARGET (n,T,3,3): an oracle ablation, not a deployable prior."""
    prior = prior or HAAR
    kind = prior["kind"]
    if kind == "haar":
        return G.random_rotation(n * T, g).reshape(n, T, 3, 3)
    if kind == "gauss":
        w = torch.randn(n, T, 3, dtype=f64, generator=g) * math.radians(prior["sigma_deg"])
        return R_cond[:, None] @ G.so3_exp(w)
    if kind == "haar_trunc":
        R = G.random_rotation(n * T, g).reshape(n, T, 3, 3)
        lim = math.radians(prior["max_deg"])
        bad = G.geodesic_distance(R, R_tgt) > lim
        while bad.any():
            R[bad] = G.random_rotation(int(bad.sum()), g)
            bad = G.geodesic_distance(R, R_tgt) > lim
        return R
    raise ValueError(prior)


def sample_source(rep, n, g, prior=None, R_cond=None, R_tgt=None):
    """Same source for all models: prior rotations (encoded) + N(0, I) translations.
    Returns (encoded rotation state, translations, source rotation matrices)."""
    Rs = sample_rotations(prior, n, g, R_cond, R_tgt)
    return rep.encode(Rs), torch.randn(n, T, 3, dtype=f64, generator=g), Rs


# ----------------------------------------------------------------------------- training

def lr_lambda(steps):
    def f(s):
        if s < WARMUP:
            return (s + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, steps - WARMUP)))
    return f


def train_one(model_name, D, seed, lr, steps, run_dir, device="cpu", ckpt_every=2000,
              width=HIDDEN, depth=N_HIDDEN_LAYERS, prior=None, dataset=None):
    """Resumable. Returns train-info dict. Skips if model.pt already exists."""
    log = get_logger()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path, final_path, ckpt_path = run_dir / "train.json", run_dir / "model.pt", run_dir / "ckpt.pt"
    info = read_json(info_path)
    if final_path.exists() and info and info.get("done"):
        return info

    rep = get_rep(model_name)
    data = load_split(D, "train", dataset)
    R_all = data["R"]
    x1_all = rep.encode(R_all)                              # float64, encoded once
    p1_all, cond_all = data["p"], data["cond"]
    n_train = cond_all.shape[0]

    seed_everything(seed)
    net = build_net(rep, width, depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps))
    g = torch.Generator().manual_seed(10_000 * seed + D)    # training-noise stream
    start, elapsed, hist = 0, 0.0, []

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)   # RNG states must stay on CPU
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        g.set_state(ck["gen"]); torch.set_rng_state(ck["torch_rng"])
        start, elapsed, hist = ck["step"], ck["elapsed"], ck["hist"]
        log.info(f"[train] resume {run_dir.name} at step {start}")

    info = dict(model=model_name, D=D, seed=seed, lr=lr, steps=steps, batch=BATCH, device=str(device),
                n_params=n_params(net), hidden=width, n_hidden_layers=depth, warmup=WARMUP,
                clip=CLIP, noise_seed=10_000 * seed + D, prior=prior or HAAR, dataset=dataset or f"D{D}",
                done=False)
    write_json(info_path, {**info, "step": start, "hist": hist})

    net.train()
    t0, run_loss = time.time(), 0.0
    for step in range(start, steps):
        idx = torch.randint(n_train, (BATCH,), generator=g)
        x0, p0, _ = sample_source(rep, BATCH, g, prior, R_all[idx, 0], R_all[idx])
        t = torch.rand(BATCH, dtype=f64, generator=g)
        xt, u = rep.path(x0, x1_all[idx], t[:, None].expand(BATCH, T))
        p1 = p1_all[idx]
        pt = (1 - t)[:, None, None] * p0 + t[:, None, None] * p1
        inp = net_input(rep, xt, pt, cond_all[idx], t).to(device)
        v, vp = split_output(rep, xt, net(inp))
        loss = torch.cat([(v - u).reshape(BATCH, -1), (vp - (p1 - p0)).reshape(BATCH, -1)], -1).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step(); sched.step()
        run_loss += loss.item()
        if (step + 1) % 500 == 0:
            hist.append([step + 1, run_loss / 500])
            run_loss = 0.0
        if (step + 1) % ckpt_every == 0 and step + 1 < steps:
            el = elapsed + time.time() - t0
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=g.get_state(), torch_rng=torch.get_rng_state(),
                            step=step + 1, elapsed=el, hist=hist), ckpt_path)
            write_json(info_path, {**info, "step": step + 1, "train_seconds": el, "hist": hist})
    elapsed += time.time() - t0
    tail = steps % 500
    if tail and steps > start:
        hist.append([steps, run_loss / min(tail, steps - start)])
    if not math.isfinite(hist[-1][1]):
        log.warning(f"[train] {run_dir.name}: non-finite loss")
    torch.save(net.state_dict(), final_path)
    info.update(done=True, step=steps, train_seconds=elapsed, final_loss=hist[-1][1] if hist else None, hist=hist)
    write_json(info_path, info)
    if ckpt_path.exists():
        ckpt_path.unlink()
    log.info(f"[train] done {run_dir.name} lr={lr} loss={info['final_loss']:.4f} {elapsed:.0f}s")
    return info


def load_net(model_name, run_dir, device="cpu"):
    rep = get_rep(model_name)
    info = read_json(Path(run_dir) / "train.json")
    net = build_net(rep, info["hidden"], info["n_hidden_layers"])
    net.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location="cpu"))
    return net.to(device).eval()


# ----------------------------------------------------------------------------- sampling

@torch.no_grad()
def integrate(rep, net, cond, x0, p0, nfe, device="cpu"):
    """Fixed-step ODE from t=0 to 1 with `nfe` network evaluations.
    M1 uses exp-map steps (stays on SO(3)); M2-M4 plain Euler steps with NO
    intermediate re-projection. Returns RAW final states."""
    x, p, h, B = x0, p0, 1.0 / nfe, cond.shape[0]
    for k in range(nfe):
        t = torch.full((B,), k * h, dtype=f64)
        v, vp = split_output(rep, x, net(net_input(rep, x, p, cond, t).to(device)))
        x = rep.step(x, v, h)
        p = p + h * vp
    return x, p


# ----------------------------------------------------------------------------- metrics

def angular_jerk_deg(R):
    """Mean over waypoints of ||third difference of orientation||, in degrees.
    omega_k = log(R_k^T R_{k+1}) (body-frame increments); jerk = 2nd diff of omega.
    Input (n, T, 3, 3) -> (n,)"""
    om = G.so3_log(G.compose_T(R[:, :-1], R[:, 1:]))
    j = om[:, 2:] - 2 * om[:, 1:-1] + om[:, :-2]
    return torch.rad2deg(j.norm(dim=-1)).mean(-1)


def _stats(x):
    x = x.double()
    return dict(mean=float(x.mean()), median=float(x.median()), p95=float(torch.quantile(x, 0.95)),
                max=float(x.max()), std=float(x.std()))


def endpoint_metrics(rep, x, p, R1_true, p1_true):
    """All metrics for one batch of generated trajectories (raw final ODE states)."""
    raw = rep.raw_matrix(x)
    res = G.orthogonality_residual(raw)                    # (n, T) BEFORE projection
    R = rep.to_matrix(x)
    finite = torch.isfinite(R).flatten(-3).all(-1) & torch.isfinite(p).flatten(-2).all(-1)
    n_nonfinite = int((~finite).sum())
    err = torch.full((R.shape[0],), 180.0, dtype=f64)      # non-finite output counted as 180 deg
    if finite.any():
        err[finite] = torch.rad2deg(G.geodesic_distance(R[finite, -1], R1_true[finite]))
    terr = (p[:, -1] - p1_true).norm(dim=-1) * P_SCALE
    jerk = angular_jerk_deg(R[finite]) if finite.any() else torch.tensor([float("nan")], dtype=f64)
    res_f = res[torch.isfinite(res)]
    return dict(
        geo_err_deg=_stats(err),
        frac_err_above_10deg=float((err > 10).double().mean()),
        frac_err_above_45deg=float((err > 45).double().mean()),
        trans_err=_stats(terr[torch.isfinite(terr)]) if torch.isfinite(terr).any() else None,
        orth_residual=dict(mean=float(res_f.mean()), p95=float(torch.quantile(res_f, 0.95)),
                           max=float(res_f.max()), endpoint_mean=float(res[:, -1][torch.isfinite(res[:, -1])].mean())),
        jerk_deg_mean=float(jerk.mean()),
        n_nonfinite=n_nonfinite,
        n=int(R.shape[0]),
    )


def eval_noise(D, seed, n, prior=None, R_cond=None, R_tgt=None):
    """Evaluation source samples. Same stream for every model => paired comparison.
    With the default Haar prior this is bit-identical to night 1."""
    g = torch.Generator().manual_seed(900_000 + 1000 * seed + D)
    R0 = sample_rotations(prior, n, g, R_cond, R_tgt)
    return R0, torch.randn(n, T, 3, dtype=f64, generator=g)


def evaluate(model_name, D, seed, run_dir, split, nfes, k_samples, out_name, device="cpu",
             prior=None, dataset=None, samples_nfe=None):
    """Resumable per-NFE evaluation; writes JSON after every NFE.
    samples_nfe: also save per-sample endpoint error and d0 (distance from the last-waypoint
    source sample to the target endpoint, as in diag_tail.py) at that NFE, to <out>_samples.npz."""
    run_dir = Path(run_dir)
    out_path = run_dir / out_name
    smp_path = run_dir / out_name.replace(".json", "_samples.npz")
    res = read_json(out_path, default={"model": model_name, "D": D, "seed": seed, "split": split,
                                       "k_samples": k_samples, "prior": prior or HAAR, "by_nfe": {}})
    todo = [n for n in nfes if str(n) not in res["by_nfe"]]
    want_smp = samples_nfe is not None and not smp_path.exists()
    if not todo and not want_smp:
        return res
    rep = get_rep(model_name)
    net = load_net(model_name, run_dir, device)
    data = load_split(D, split, dataset)
    cond = data["cond"].repeat_interleave(k_samples, 0)
    R1 = data["R"][:, -1].repeat_interleave(k_samples, 0)
    p1 = data["p"][:, -1].repeat_interleave(k_samples, 0)
    R0n, p0n = eval_noise(D, seed, cond.shape[0], prior, data["R"][:, 0].repeat_interleave(k_samples, 0),
                          data["R"].repeat_interleave(k_samples, 0))
    x0 = rep.encode(R0n)
    if want_smp:
        x, _ = integrate(rep, net, cond, x0, p0n, samples_nfe, device)
        err = torch.full((cond.shape[0],), 180.0, dtype=f64)
        R = rep.to_matrix(x)[:, -1]
        ok = torch.isfinite(R).flatten(-2).all(-1)
        err[ok] = torch.rad2deg(G.geodesic_distance(R[ok], R1[ok]))
        d0 = torch.rad2deg(G.geodesic_distance(R0n[:, -1], R1))
        tmp = smp_path.with_suffix(".tmp.npz")
        np.savez(tmp, err_deg=err.numpy(), d0_deg=d0.numpy(), nfe=np.array(samples_nfe))
        tmp.replace(smp_path)
    for nfe in todo:
        t0 = time.time()
        x, p = integrate(rep, net, cond, x0, p0n, nfe, device)
        dt = time.time() - t0
        m = endpoint_metrics(rep, x, p, R1, p1)
        m["infer_seconds_per_traj"] = dt / cond.shape[0]
        res["by_nfe"][str(nfe)] = m
        write_json(out_path, res)
    return res


# ----------------------------------------------------------------------------- multimodal (night 2, Phase C)

ON_MODE_DEG = 10.0     # a sample "hits" its nearest mode if within this geodesic distance


def _w1_to_modes(dist_deg, k):
    """Exact W1 (geodesic cost, deg) between k equally weighted samples and the uniform
    distribution over K modes, for one start. dist_deg: (k, K); k must be divisible by K."""
    from scipy.optimize import linear_sum_assignment
    K = dist_deg.shape[1]
    C = np.repeat(dist_deg, k // K, axis=1)            # each mode duplicated k/K times -> square
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


def mm_metrics(R_end, modes_R1, n_starts, k):
    """R_end (n*k, 3, 3) projected endpoints, grouped by start; modes_R1 (n, K, 3, 3)."""
    K = modes_R1.shape[1]
    Rm = modes_R1.repeat_interleave(k, 0)                                      # (n*k, K, 3, 3)
    ok = torch.isfinite(R_end).flatten(-2).all(-1)
    dist = torch.full((R_end.shape[0], K), 180.0, dtype=f64)
    if ok.any():
        dist[ok] = torch.rad2deg(G.geodesic_distance(R_end[ok, None].expand(-1, K, 3, 3), Rm[ok]))
    near, assign = dist.min(-1)
    hit = near < ON_MODE_DEG
    counts = torch.zeros(n_starts, K, dtype=torch.long)
    sidx = torch.arange(R_end.shape[0]) // k
    counts.index_put_((sidx[hit], assign[hit]), torch.ones(int(hit.sum()), dtype=torch.long), accumulate=True)
    n_hit = counts.sum(-1)
    have = n_hit > 0
    props = counts[have].double() / n_hit[have, None]
    tv = 0.5 * (props - 1.0 / K).abs().sum(-1)                                 # per start, among on-mode samples
    dn = dist.reshape(n_starts, k, K).numpy()
    w1 = np.array([_w1_to_modes(dn[i], k) for i in range(n_starts)])
    pooled = counts.sum(0).double()
    return dict(
        nearest_mode_err_deg=_stats(near),
        frac_on_mode=float(hit.double().mean()),
        coverage_all_modes=float((counts > 0).all(-1).double().mean()),     # starts with every mode hit >= 1
        modes_hit_mean=float((counts > 0).sum(-1).double().mean()),
        mode_props_pooled=(pooled / pooled.sum()).tolist() if pooled.sum() > 0 else None,
        mode_props_min_mean=float(props.min(-1).values.mean()) if have.any() else 0.0,
        tv_from_uniform_mean=float(tv.mean()) if have.any() else 1.0,
        w1_deg=_stats(torch.from_numpy(w1)),
        n_nonfinite=int((~ok).sum()), n=int(R_end.shape[0]), n_starts=n_starts, k=k,
    ), dict(nearest=near.numpy(), assign=assign.numpy(), hit=hit.numpy(), w1=w1)


def evaluate_mm(model_name, D, seed, run_dir, split, nfes, k_samples, out_name, device="cpu",
                prior=None, samples_nfe=None):
    """Multimodal evaluation: k_samples draws per start (k divisible by K=3); resumable per NFE."""
    run_dir = Path(run_dir)
    out_path = run_dir / out_name
    smp_path = run_dir / out_name.replace(".json", "_samples.npz")
    res = read_json(out_path, default={"model": model_name, "D": D, "seed": seed, "split": split,
                                       "k_samples": k_samples, "prior": prior or HAAR,
                                       "on_mode_deg": ON_MODE_DEG, "by_nfe": {}})
    todo = [n for n in nfes if str(n) not in res["by_nfe"]]
    if not todo and (samples_nfe is None or smp_path.exists()):
        return res
    rep = get_rep(model_name)
    net = load_net(model_name, run_dir, device)
    data = load_split(D, split, f"MM_D{D}")
    n = data["cond"].shape[0]
    cond = data["cond"].repeat_interleave(k_samples, 0)
    R0n, p0n = eval_noise(D, seed, cond.shape[0], prior, data["R"][:, 0].repeat_interleave(k_samples, 0))
    x0 = rep.encode(R0n)
    for nfe in sorted(set(todo) | ({samples_nfe} if samples_nfe and not smp_path.exists() else set())):
        t0 = time.time()
        x, _ = integrate(rep, net, cond, x0, p0n, nfe, device)
        dt = time.time() - t0
        m, per = mm_metrics(rep.to_matrix(x)[:, -1], data["modes_R1"], n, k_samples)
        res_orth = G.orthogonality_residual(rep.raw_matrix(x)[:, -1])
        m["orth_residual_endpoint_mean"] = float(res_orth[torch.isfinite(res_orth)].mean())
        m["infer_seconds_per_traj"] = dt / cond.shape[0]
        if nfe == samples_nfe and not smp_path.exists():
            tmp = smp_path.with_suffix(".tmp.npz")
            np.savez(tmp, nfe=np.array(nfe), data_mode=data["mode"].numpy(), **per)
            tmp.replace(smp_path)
        if str(nfe) not in res["by_nfe"]:
            res["by_nfe"][str(nfe)] = m
            write_json(out_path, res)
    return res
