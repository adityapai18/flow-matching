"""Night-2 additions: source priors, multimodal data, multimodal metrics. CPU float64."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fm  # noqa: E402
import geometry as G  # noqa: E402
import make_data  # noqa: E402
from flows import get_rep  # noqa: E402

f64 = torch.float64
T = fm.T


def _targets(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return G.random_rotation(n * T, g).reshape(n, T, 3, 3)


def test_haar_prior_is_bit_identical_to_night1():
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    old = G.random_rotation(50 * T, g1).reshape(50, T, 3, 3)
    new = fm.sample_rotations(fm.HAAR, 50, g2)
    assert torch.equal(old, new)
    assert torch.equal(torch.randn(3, generator=g1), torch.randn(3, generator=g2))   # stream position too


def test_truncated_haar_respects_bound_and_is_deterministic():
    Rt = _targets(400)
    prior = {"kind": "haar_trunc", "max_deg": 150}
    a = fm.sample_rotations(prior, 400, torch.Generator().manual_seed(3), R_tgt=Rt)
    b = fm.sample_rotations(prior, 400, torch.Generator().manual_seed(3), R_tgt=Rt)
    assert torch.equal(a, b)
    d = torch.rad2deg(G.geodesic_distance(a, Rt))
    assert d.max() <= 150 + 1e-9
    assert d.max() > 140                                   # truncation, not collapse
    assert G.orthogonality_residual(a).max() < 1e-12
    # truncated Haar is Haar conditioned on d <= 150: CDF(theta) = (theta - sin theta) / (5pi/6 - sin(5pi/6))
    th = torch.deg2rad(d.flatten())
    for q in (60, 90, 120):
        x = math.radians(q)
        want = (x - math.sin(x)) / (5 * math.pi / 6 - 0.5)
        assert abs(float((th < x).double().mean()) - want) < 0.01


def test_gauss_prior_centered_on_conditioning_pose():
    n = 2000
    Rc = G.random_rotation(n, torch.Generator().manual_seed(1))
    prior = {"kind": "gauss", "sigma_deg": 30}
    R = fm.sample_rotations(prior, n, torch.Generator().manual_seed(2), R_cond=Rc)
    assert R.shape == (n, T, 3, 3)
    w = G.so3_log(G.compose_T(Rc[:, None].expand_as(R), R))   # body-frame offsets from R_cond
    s = math.radians(30)
    assert abs(float(w.mean())) < 0.01
    assert abs(float(w.std()) - s) / s < 0.02              # |w| < pi almost surely at 30 deg
    assert G.orthogonality_residual(R).max() < 1e-12


def test_sample_source_encodes_the_same_rotations_for_every_model():
    Rc, Rt = G.random_rotation(8, torch.Generator().manual_seed(4)), _targets(8)
    for prior in (fm.HAAR, {"kind": "gauss", "sigma_deg": 60}, {"kind": "haar_trunc", "max_deg": 150}):
        outs = []
        for m in ("M1_riemannian", "M2_euclid6d", "M3_quat", "M4_euler"):
            rep = get_rep(m)
            x0, p0, Rs = fm.sample_source(rep, 8, torch.Generator().manual_seed(9), prior, Rc, Rt)
            assert G.geodesic_distance(rep.to_matrix(x0), Rs).max() < 1e-9
            outs.append((Rs, p0))
        for Rs, p0 in outs[1:]:
            assert torch.equal(Rs, outs[0][0]) and torch.equal(p0, outs[0][1])


def test_mode_axes_geometry():
    a = torch.randn(500, 3, dtype=f64, generator=torch.Generator().manual_seed(5))
    a = a / a.norm(dim=-1, keepdim=True)
    a[0] = torch.tensor([0.0, 0.0, 1.0], dtype=f64)        # pole: fallback branch
    b = make_data.mode_axes(a)
    assert torch.allclose(b.norm(dim=-1), torch.ones(500, 3, dtype=f64), atol=1e-12)
    assert torch.allclose(b[:, 0], a, atol=1e-15)
    ang = torch.rad2deg(torch.arccos((b[:, 1:] * a[:, None]).sum(-1).clamp(-1, 1)))
    assert torch.allclose(ang, torch.full_like(ang, 60.0), atol=1e-6)
    assert torch.isfinite(b).all()


def test_multimodal_generator_keeps_starts_and_hits_modes():
    uni, mm = make_data.generate(30), make_data.generate(30, multimodal=True)
    assert torch.equal(uni["R0"], mm["R0"]) and torch.equal(uni["p0"], mm["p0"]) and torch.equal(uni["p1"], mm["p1"])
    assert torch.equal(uni["split"]["test"], mm["split"]["test"])
    n = mm["R"].shape[0]
    assert G.geodesic_distance(mm["R"][:, -1], mm["modes_R1"][torch.arange(n), mm["mode"]]).max() < 1e-10
    assert (torch.rad2deg(G.geodesic_distance(mm["R0"][:, None].expand(-1, 3, 3, 3), mm["modes_R1"])) - 30).abs().max() < 1e-8


def _modes(n, K=3, D=math.radians(90), seed=0):
    g = torch.Generator().manual_seed(seed)
    R0 = G.random_rotation(n, g)
    ax = torch.eye(3, dtype=f64)[None].expand(n, K, 3)
    return R0[:, None] @ G.so3_exp(ax * D)


def test_mm_metrics_perfect_sampler():
    n, k = 20, 30
    M = _modes(n)
    R_end = M[:, torch.arange(k) % 3].reshape(n * k, 3, 3)            # 10 exact samples per mode per start
    m, _ = fm.mm_metrics(R_end, M, n, k)
    assert m["coverage_all_modes"] == 1.0 and m["frac_on_mode"] == 1.0
    assert m["w1_deg"]["max"] < 1e-5 and m["nearest_mode_err_deg"]["max"] < 1e-5
    assert m["tv_from_uniform_mean"] < 1e-12
    assert np.allclose(m["mode_props_pooled"], [1 / 3] * 3)


def test_mm_metrics_mode_collapse():
    n, k = 20, 30
    M = _modes(n)
    R_end = M[:, [0] * k].reshape(n * k, 3, 3)                         # every sample on mode 0
    m, _ = fm.mm_metrics(R_end, M, n, k)
    assert m["nearest_mode_err_deg"]["max"] < 1e-5                    # precise ...
    assert m["coverage_all_modes"] == 0.0 and m["modes_hit_mean"] == 1.0
    assert abs(m["tv_from_uniform_mean"] - 2 / 3) < 1e-12
    sep = torch.rad2deg(G.geodesic_distance(M[:, 0], M[:, 1])).mean()
    assert abs(m["w1_deg"]["mean"] - 2 / 3 * float(sep)) < 1e-6        # ... but 2/3 of the mass must move


def test_mm_metrics_nonfinite_counts_as_miss():
    n, k = 2, 3
    M = _modes(n)
    R_end = M[:, [0, 1, 2]].reshape(n * k, 3, 3).clone()
    R_end[0] = float("nan")
    m, _ = fm.mm_metrics(R_end, M, n, k)
    assert m["n_nonfinite"] == 1 and m["coverage_all_modes"] == 0.5
