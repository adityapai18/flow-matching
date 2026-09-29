"""Stage 1: synthetic SE(3) trajectory datasets with controlled angular displacement D.

Task definition (per trajectory):
  axis a ~ Uniform(S^2), start tilt sigma ~ U[10, 20] deg
  start   g0 = (R0, p0),  R0 = exp(sigma * a),  p0 ~ N(0, 0.1^2 I)
  target  g1 = (R1, p1),  R1 = R0 exp(D * a),   p1 = p0 + L * a,  L = 0.3
  => d(R0, R1) = D exactly, |p1 - p0| = L exactly, and g1 is a deterministic
     function of g0 (the axis is recoverable from R0), so "geodesic error to the
     ground-truth endpoint" is well defined for a model conditioned on g0 only.
  Starts sit near the identity, so D controls how far the data reaches toward
  the representation singularities (quaternion hemisphere boundary at 180 deg
  from identity, Euler XYZ gimbal lock at middle angle +-90 deg).

Path types (cyclic assignment, shuffled -> 667/667/666):
  0 geodesic : screw motion g0 * se3_exp(s(tau) * se3_log(g0^-1 g1))
  1 via      : geodesic * detour bump sin^2(pi tau) through a via pose at tau=0.5;
               rotation detour U[0.15, 0.30]*D about a random axis perpendicular to a,
               translation detour U[0.15, 0.30]*L perpendicular to a
  2 overshoot: R0 exp(psi(tau) D a), psi = s + A tau^3 (1-tau)^2 with max psi = 1 + o,
               o ~ U[0.10, 0.25]; translation follows the minimum-jerk line
  s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 (minimum-jerk timing), tau = k/31.

Saved as rotation MATRICES (float64) plus translations. No representation
conversion happens here.
"""
import math
import sys

import numpy as np
import torch

import geometry as G
from common import D_LEVELS, DATA_DIR, N_WAYPOINTS, get_logger, read_json, write_json

N_TRAJ = 2000
L_TRANS = 0.3
P0_STD = 0.1
SIGMA_RANGE_DEG = (10.0, 20.0)
DETOUR_FRAC = (0.15, 0.30)
OVERSHOOT_FRAC = (0.10, 0.25)
f64 = torch.float64


def data_seed(D):
    return 1000 + D


def min_jerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def unit_perp(a, g):
    """Random unit vectors perpendicular to each row of a."""
    r = torch.randn(a.shape, dtype=f64, generator=g)
    r = r - (r * a).sum(-1, keepdim=True) * a
    return r / r.norm(dim=-1, keepdim=True)


def overshoot_gain(o, n_grid=4001):
    """Solve A so that max_tau [s(tau) + A tau^3 (1-tau)^2] = 1 + o (bisection, vectorized)."""
    tau = torch.linspace(0, 1, n_grid, dtype=f64)
    s, h = min_jerk(tau), tau ** 3 * (1 - tau) ** 2
    lo, hi = torch.zeros_like(o), torch.full_like(o, 100.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        peak = (s[None] + mid[:, None] * h[None]).max(-1).values
        big = peak > 1 + o
        hi = torch.where(big, mid, hi)
        lo = torch.where(big, lo, mid)
    return 0.5 * (lo + hi)


MM_OFFSETS_DEG = (0.0, 60.0, -60.0)   # night-2 multimodal: mode axes at these angles from a


def mode_axes(a):
    """K=3 rotation axes per start, a deterministic function of the start axis a (n,3) -> (n,K,3):
    a itself, and a tilted by +/-60 deg about e1 = z x a / |z x a|. e1 is singular at a = +/-z
    (a tangent line field on S^2 must have singularities), so the mode set turns quickly
    for the ~1.5% of starts within 10 deg of the poles."""
    z = torch.tensor([0.0, 0.0, 1.0], dtype=f64).expand_as(a)
    e1 = torch.linalg.cross(z, a)
    small = e1.norm(dim=-1) < 1e-6
    e1 = torch.where(small[:, None], torch.linalg.cross(torch.tensor([1.0, 0.0, 0.0], dtype=f64).expand_as(a), a), e1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = torch.linalg.cross(e1, a)
    phi = torch.deg2rad(torch.tensor(MM_OFFSETS_DEG, dtype=f64))
    return torch.cos(phi)[None, :, None] * a[:, None] + torch.sin(phi)[None, :, None] * e2[:, None]


def generate(D_deg, multimodal=False):
    """multimodal=False reproduces night 1 bit-for-bit. multimodal=True (night 2, Phase C): same starts
    (a, sigma, p0) and splits, but the rotation is about b = mode_axes(a)[k], with k ~ U{0,1,2} drawn from a
    separate stream; translation still moves along a (only the rotation is multimodal)."""
    g = torch.Generator().manual_seed(data_seed(D_deg))
    n, T = N_TRAJ, N_WAYPOINTS
    D = math.radians(D_deg)
    tau = torch.linspace(0, 1, T, dtype=f64)
    s = min_jerk(tau)                                      # (T,)
    bump = torch.sin(math.pi * tau) ** 2                   # (T,)

    a = torch.randn(n, 3, dtype=f64, generator=g)
    a = a / a.norm(dim=-1, keepdim=True)
    sigma = torch.deg2rad(SIGMA_RANGE_DEG[0] + (SIGMA_RANGE_DEG[1] - SIGMA_RANGE_DEG[0])
                          * torch.rand(n, dtype=f64, generator=g))
    R0 = G.so3_exp(a * sigma[:, None])
    p0 = P0_STD * torch.randn(n, 3, dtype=f64, generator=g)
    if multimodal:
        axes = mode_axes(a)
        mode = torch.randint(len(MM_OFFSETS_DEG), (n,), generator=torch.Generator().manual_seed(5000 + D_deg))
        b = axes[torch.arange(n), mode]
    else:
        b = a
    R1 = R0 @ G.so3_exp(b * D)
    p1 = p0 + L_TRANS * a

    ptype = torch.arange(n) % 3
    ptype = ptype[torch.randperm(n, generator=g)]

    # (0) geodesic screw in SE(3), body-frame twist
    xi = G.se3_log(G.compose_T(R0, R1), (G.mT(R0) @ (p1 - p0).unsqueeze(-1)).squeeze(-1))
    Rrel, prel = G.se3_exp(xi[:, None, :] * s[None, :, None])           # (n,T,3,3), (n,T,3)
    R_geo = R0[:, None] @ Rrel
    p_geo = p0[:, None] + (R0[:, None] @ prel.unsqueeze(-1)).squeeze(-1)

    # (1) via-point detour
    m_rot = D * (DETOUR_FRAC[0] + (DETOUR_FRAC[1] - DETOUR_FRAC[0]) * torch.rand(n, dtype=f64, generator=g))
    m_tr = L_TRANS * (DETOUR_FRAC[0] + (DETOUR_FRAC[1] - DETOUR_FRAC[0]) * torch.rand(n, dtype=f64, generator=g))
    delta = unit_perp(b, g) * m_rot[:, None]
    dtr = unit_perp(a, g) * m_tr[:, None]
    R_via = R_geo @ G.so3_exp(delta[:, None, :] * bump[None, :, None])
    p_via = p_geo + dtr[:, None, :] * bump[None, :, None]

    # (2) rotational overshoot
    o = OVERSHOOT_FRAC[0] + (OVERSHOOT_FRAC[1] - OVERSHOOT_FRAC[0]) * torch.rand(n, dtype=f64, generator=g)
    A = overshoot_gain(o)
    psi = s[None] + A[:, None] * (tau ** 3 * (1 - tau) ** 2)[None]      # (n,T)
    R_ov = R0[:, None] @ G.so3_exp(b[:, None, :] * (D * psi)[..., None])
    p_ov = p0[:, None] + L_TRANS * a[:, None, :] * s[None, :, None]

    sel = ptype[:, None, None, None]
    R = torch.where(sel == 0, R_geo, torch.where(sel == 1, R_via, R_ov))
    p = torch.where(ptype[:, None, None] == 0, p_geo, torch.where(ptype[:, None, None] == 1, p_via, p_ov))

    perm = torch.randperm(n, generator=g)
    split = {"train": perm[:1600], "val": perm[1600:1800], "test": perm[1800:]}
    out = dict(R=R, p=p, ptype=ptype, axis=a, sigma=sigma, R0=R0, p0=p0, R1=R1, p1=p1,
               overshoot=o, split=split)
    if multimodal:
        out.update(mode=mode, mode_axes=axes, modes_R1=R0[:, None] @ G.so3_exp(axes * D))
    return out


def dataset_stats(d, D_deg):
    """Checks + descriptive stats: how close does this level come to each representation's singularities?"""
    R, p = d["R"], d["p"]
    n, T = R.shape[:2]
    Rf = R.reshape(-1, 3, 3)
    ang_from_I = torch.rad2deg(G.so3_angle(Rf)).reshape(n, T)
    q = G.matrix_to_quat(Rf).reshape(n, T, 4)
    flips = ((q[:, 1:] * q[:, :-1]).sum(-1) < 0)                       # canonical sign jumps between waypoints
    e = G.matrix_to_euler(Rf).reshape(n, T, 3)
    beta = torch.rad2deg(e[..., 1]).abs()
    ejump = (e[:, 1:] - e[:, :-1]).abs().amax(-1) > math.pi / 2          # Euler discontinuity between waypoints
    return {
        "D_deg": D_deg,
        "n": n,
        "start_to_target_deg_max_err": float((torch.rad2deg(G.geodesic_distance(d["R0"], d["R1"])) - D_deg).abs().max()),
        "endpoint_matches_target_max_rad": float(G.geodesic_distance(R[:, -1], d["R1"]).max()),
        "startpoint_matches_start_max_rad": float(G.geodesic_distance(R[:, 0], d["R0"]).max()),
        "endpoint_trans_err_max": float((p[:, -1] - d["p1"]).norm(dim=-1).max()),
        "orthogonality_residual_max": float(G.orthogonality_residual(Rf).max()),
        "path_type_counts": [int((d["ptype"] == k).sum()) for k in range(3)],
        "angle_from_identity_deg": {"max": float(ang_from_I.max()), "p95": float(ang_from_I.quantile(0.95)),
                                    "endpoint_mean": float(ang_from_I[:, -1].mean())},
        "frac_traj_with_canonical_quat_sign_flip": float(flips.any(-1).double().mean()),
        "frac_traj_endpoint_quat_w_below_0.1": float((q[:, -1, 0] < 0.1).double().mean()),
        "frac_traj_with_euler_jump": float(ejump.any(-1).double().mean()),
        "frac_traj_with_euler_beta_above_80deg": float((beta > 80).any(-1).double().mean()),
        "max_euler_beta_deg": float(beta.max()),
    }


def mode_stats(d):
    """Multimodal extras: how far apart are the K valid endpoints of one start?"""
    M = d["modes_R1"]
    K = M.shape[1]
    sep = torch.stack([torch.rad2deg(G.geodesic_distance(M[:, i], M[:, j]))
                       for i in range(K) for j in range(i + 1, K)], -1)
    return {"mode_counts": [int((d["mode"] == k).sum()) for k in range(K)],
            "mode_offsets_deg": list(MM_OFFSETS_DEG),
            "mode_separation_deg": {"min": float(sep.min()), "mean": float(sep.mean()), "max": float(sep.max())},
            "endpoint_is_its_mode_max_rad": float(G.geodesic_distance(
                M[torch.arange(M.shape[0]), d["mode"]], d["R1"]).max())}


def main():
    """python make_data.py [D ...]        night-1 unimodal levels -> D{D}.npz
       python make_data.py mm [D ...]     night-2 multimodal levels -> MM_D{D}.npz"""
    log = get_logger()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    mm = bool(args) and args[0] == "mm"
    if mm:
        args = args[1:]
    levels = [int(x) for x in args] or ([30, 150] if mm else D_LEVELS)
    for D in levels:
        stem = f"MM_D{D}" if mm else f"D{D}"
        out = DATA_DIR / f"{stem}.npz"
        meta_path = DATA_DIR / f"{stem}_meta.json"
        if out.exists() and read_json(meta_path):
            log.info(f"[data] {stem} cached, skipping")
            continue
        try:
            d = generate(D, multimodal=mm)
            stats = dataset_stats(d, D)
            assert stats["start_to_target_deg_max_err"] < 1e-8, stats
            assert stats["endpoint_matches_target_max_rad"] < 1e-10, stats
            assert stats["startpoint_matches_start_max_rad"] < 1e-10, stats
            assert stats["endpoint_trans_err_max"] < 1e-10, stats
            extra = {}
            if mm:
                stats.update(mode_stats(d))
                assert stats["endpoint_is_its_mode_max_rad"] < 1e-10, stats
                extra = dict(mode=d["mode"].numpy(), mode_axes=d["mode_axes"].numpy(), modes_R1=d["modes_R1"].numpy())
            np.savez_compressed(
                out, R=d["R"].numpy(), p=d["p"].numpy(), path_type=d["ptype"].numpy(),
                axis=d["axis"].numpy(), sigma=d["sigma"].numpy(), overshoot=d["overshoot"].numpy(),
                train_idx=d["split"]["train"].numpy(), val_idx=d["split"]["val"].numpy(),
                test_idx=d["split"]["test"].numpy(), D_deg=np.array(D), **extra)
            stats["seed"] = data_seed(D)
            write_json(meta_path, stats)
            log.info(f"[data] {stem} saved: {stats}")
        except Exception as e:  # log, continue with next level
            log.exception(f"[data] {stem} FAILED: {e}")


if __name__ == "__main__":
    main()
