"""Stage 0 unit tests. Every downstream result depends on these passing.

Run: .venv/bin/python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as SciRot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import geometry as G  # noqa: E402

N = 10000
TOL = 1e-10
f64 = torch.float64


def gen(seed):
    return torch.Generator().manual_seed(seed)


def rand_rots(n=N, seed=0):
    return G.random_rotation(n, gen(seed))


def rand_axes(n, seed):
    a = torch.randn(n, 3, dtype=f64, generator=gen(seed))
    return a / a.norm(dim=-1, keepdim=True)


def maxabs(x):
    return x.abs().max().item()


def edge_rotations():
    """Identity, exact pi rotations, near-0, near-pi, gimbal-lock poses."""
    I = torch.eye(3, dtype=f64)
    pis = [torch.diag(torch.tensor(d, dtype=f64)) for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    axes = rand_axes(50, 99)
    near0 = G.so3_exp(axes * 1e-9)
    nearpi = G.so3_exp(axes * (math.pi - 1e-9))
    exactpi = G.so3_exp(axes * math.pi)
    lock = G.euler_to_matrix(torch.tensor([[0.3, math.pi / 2, -1.1], [-2.0, -math.pi / 2, 0.7]], dtype=f64))
    return torch.cat([I[None], torch.stack(pis), near0, nearpi, exactpi, lock])


# ----------------------------------------------------------------------------- dtype guard

def test_rejects_non_float64():
    with pytest.raises(TypeError):
        G.so3_log(torch.eye(3, dtype=torch.float32))
    with pytest.raises(TypeError):
        G.so3_exp(torch.zeros(3, dtype=torch.float16))


# ----------------------------------------------------------------------------- exp / log

def test_exp_log_roundtrip_random():
    R = rand_rots()
    assert maxabs(G.so3_exp(G.so3_log(R)) - R) < TOL


def test_exp_log_roundtrip_edges():
    R = edge_rotations()
    w = G.so3_log(R)
    assert torch.isfinite(w).all()
    assert maxabs(G.so3_exp(w) - R) < TOL


def test_log_exp_roundtrip_random():
    axes = rand_axes(N, 1)
    th = torch.rand(N, dtype=f64, generator=gen(2)) * (math.pi - 1e-3)
    w = axes * th[:, None]
    assert maxabs(G.so3_log(G.so3_exp(w)) - w) < TOL


def test_exp_matches_scipy():
    w = torch.randn(1000, 3, dtype=f64, generator=gen(3))
    ref = SciRot.from_rotvec(w.numpy()).as_matrix()
    assert np.abs(G.so3_exp(w).numpy() - ref).max() < TOL


def test_log_at_identity_is_exactly_zero():
    w = G.so3_log(torch.eye(3, dtype=f64)[None])
    assert (w == 0).all()


def test_log_small_angle_branch():
    axes = rand_axes(200, 4)
    ths = torch.cat([torch.tensor([0.0], dtype=f64), torch.logspace(-14, -3, 199, dtype=f64)])
    w_true = axes * ths[:, None]
    w = G.so3_log(G.so3_exp(w_true))
    assert torch.isfinite(w).all()
    # absolute error at machine level, relative error small wherever theta is not tiny
    assert maxabs(w - w_true) < 1e-15
    big = ths > 1e-8
    rel = (w - w_true).norm(dim=-1)[big] / ths[big]
    assert rel.max().item() < 1e-8


def test_log_continuous_across_small_branch_boundary():
    a = rand_axes(20, 5)
    for d in [1e-12, 1e-10, 1e-8]:
        below = a * (G.SMALL_ANGLE * (1 - d))
        above = a * (G.SMALL_ANGLE * (1 + d))
        wb = G.so3_log(G.so3_exp(below))
        wa = G.so3_log(G.so3_exp(above))
        # each side exact; the jump across the boundary equals the true input jump
        assert maxabs(wb - below) < 1e-17
        assert maxabs(wa - above) < 1e-17
        assert maxabs((wa - wb) - (above - below)) < 1e-17


def test_log_near_pi_branch():
    axes = rand_axes(300, 6)
    deltas = torch.logspace(-13, -4.1, 300, dtype=f64)   # all strictly inside the near-pi branch
    th = math.pi - deltas
    w_true = axes * th[:, None]
    R = G.so3_exp(w_true)
    w = G.so3_log(R)
    assert torch.isfinite(w).all()
    assert maxabs(w.norm(dim=-1) - th) < 1e-6
    assert maxabs(w - w_true) < 1e-6          # correct axis AND sign for theta < pi
    assert maxabs(G.so3_exp(w) - R) < TOL


def test_log_exactly_pi():
    axes = rand_axes(300, 7)
    R = G.so3_exp(axes * math.pi)
    w = G.so3_log(R)
    assert torch.isfinite(w).all()
    assert maxabs(w.norm(dim=-1) - math.pi) < 1e-6
    # at exactly pi, +axis and -axis are the same rotation; either is correct
    err = torch.minimum((w - axes * math.pi).norm(dim=-1), (w + axes * math.pi).norm(dim=-1))
    assert err.max().item() < 1e-6
    assert maxabs(G.so3_exp(w) - R) < TOL
    for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        Rd = torch.diag(torch.tensor(d, dtype=f64))[None]
        wd = G.so3_log(Rd)
        assert abs(wd.norm().item() - math.pi) < 1e-12
        assert maxabs(G.so3_exp(wd) - Rd) < TOL


def test_log_continuous_across_near_pi_boundary():
    # Just below the boundary the generic formula theta/sin(theta) * vee(R) is
    # used; its error is bounded by eps * pi / sin(1e-4) ~ 7e-12 (measured 2e-12).
    # The dedicated branch above the boundary is accurate to ~1e-15.
    a = rand_axes(2000, 8)
    for d in [1e-12, 1e-10, 1e-8]:
        below = a * (G.NEAR_PI - d)
        above = a * (G.NEAR_PI + d)
        wb = G.so3_log(G.so3_exp(below))
        wa = G.so3_log(G.so3_exp(above))
        assert maxabs(wb - below) < 1e-11
        assert maxabs(wa - above) < 1e-14
        assert maxabs((wa - wb) - (above - below)) < 1e-11


# ----------------------------------------------------------------------------- geodesic distance

def test_geodesic_distance_self_is_exactly_zero():
    R = torch.cat([rand_rots(), edge_rotations()])
    d = G.geodesic_distance(R, R)
    assert (d == 0).all(), f"nonzero self-distance: max {d.max().item()}"


def test_geodesic_distance_known_axis_angle():
    R = rand_rots(N, 9)
    axes = rand_axes(N, 10)
    th = torch.rand(N, dtype=f64, generator=gen(11)) * math.pi
    R2 = R @ G.so3_exp(axes * th[:, None])
    assert maxabs(G.geodesic_distance(R, R2) - th) < TOL
    # exact values for canonical cases
    I = torch.eye(3, dtype=f64)[None]
    for ang in [1e-8, 1e-3, 0.5, math.pi / 2, 3.0, math.pi - 1e-7, math.pi]:
        Rz = G.so3_exp(torch.tensor([[0, 0, ang]], dtype=f64))
        assert abs(G.geodesic_distance(I, Rz).item() - ang) < 1e-12


def test_geodesic_distance_symmetric_and_invariant():
    A, B, C = rand_rots(1000, 12), rand_rots(1000, 13), rand_rots(1000, 14)
    d = G.geodesic_distance(A, B)
    assert maxabs(d - G.geodesic_distance(B, A)) < TOL
    assert maxabs(d - G.geodesic_distance(C @ A, C @ B)) < TOL      # left invariance
    assert maxabs(d - G.geodesic_distance(A @ C, B @ C)) < TOL      # right invariance (bi-invariant metric)


# ----------------------------------------------------------------------------- SE(3)

def test_se3_exp_log_roundtrip():
    R = torch.cat([rand_rots(), edge_rotations()])
    p = torch.randn(R.shape[0], 3, dtype=f64, generator=gen(15))
    xi = G.se3_log(R, p)
    assert torch.isfinite(xi).all()
    R2, p2 = G.se3_exp(xi)
    assert maxabs(R2 - R) < TOL
    assert maxabs(p2 - p) < TOL


def test_se3_log_exp_roundtrip_including_small_angles():
    axes = rand_axes(N, 16)
    th = torch.cat([torch.logspace(-14, -3, 1000, dtype=f64),
                    torch.rand(N - 1000, dtype=f64, generator=gen(17)) * (math.pi - 1e-3)])
    xi = torch.cat([axes * th[:, None], torch.randn(N, 3, dtype=f64, generator=gen(18))], -1)
    assert maxabs(G.se3_log(*G.se3_exp(xi)) - xi) < TOL


def test_se3_screw_along_axis():
    a = rand_axes(100, 19)
    xi = torch.cat([a * 2.0, a * 0.3], -1)        # v parallel to omega: pure screw
    R, p = G.se3_exp(xi)
    assert maxabs(p - a * 0.3) < TOL


# ----------------------------------------------------------------------------- quaternions

def test_quat_matches_scipy_convention():
    q = G.canonicalize_quat(torch.randn(1000, 4, dtype=f64, generator=gen(20)))
    q = q / q.norm(dim=-1, keepdim=True)
    ref = SciRot.from_quat(q.numpy(), scalar_first=True).as_matrix()
    assert np.abs(G.quat_to_matrix(q).numpy() - ref).max() < TOL


def test_quat_roundtrip():
    R = torch.cat([rand_rots(), edge_rotations()])
    q = G.matrix_to_quat(R)
    assert torch.isfinite(q).all()
    assert maxabs(G.quat_to_matrix(q) - R) < TOL
    # canonical quat -> matrix -> quat is the identity, except exactly w == 0
    # where the tie-break is re-applied (still deterministic)
    q2 = G.matrix_to_quat(G.quat_to_matrix(q))
    assert maxabs(q2 - q) < TOL


def test_quat_sign_ambiguity_same_matrix():
    q = torch.randn(N, 4, dtype=f64, generator=gen(21))
    assert maxabs(G.quat_to_matrix(q) - G.quat_to_matrix(-q)) < 1e-15


def test_quat_canonicalization_deterministic():
    q = torch.randn(N, 4, dtype=f64, generator=gen(22))
    # hand-made ties: w == 0, and w == x == 0, and w == x == y == 0
    ties = torch.tensor([[0.0, 0.6, -0.8, 0.0], [0.0, -0.6, 0.8, 0.0], [0.0, 0.0, -0.6, 0.8],
                         [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]], dtype=f64)
    q = torch.cat([q, ties])
    c1, c2 = G.canonicalize_quat(q), G.canonicalize_quat(-q)
    assert (c1 == c2).all()                                  # q and -q -> same representative
    assert (G.canonicalize_quat(c1) == c1).all()             # idempotent
    assert (c1[:, 0] >= 0).all()
    w0 = c1[:, 0] == 0
    first = c1[w0][:, 1:]
    first_nz = first[torch.arange(first.shape[0]), (first != 0).to(torch.int64).argmax(-1)]
    assert (first_nz > 0).all()
    # matrix_to_quat always canonical, and bit-for-bit repeatable
    R = rand_rots(1000, 23)
    qa, qb = G.matrix_to_quat(R), G.matrix_to_quat(R.clone())
    assert (qa == qb).all() and (qa[:, 0] >= 0).all()


# ----------------------------------------------------------------------------- 6D

def test_sixd_roundtrip():
    R = torch.cat([rand_rots(), edge_rotations()])
    x = G.matrix_to_sixd(R)
    assert maxabs(G.sixd_to_matrix(x) - R) < TOL
    assert maxabs(G.matrix_to_sixd(G.sixd_to_matrix(x)) - x) < TOL


def test_sixd_gram_schmidt_projects_arbitrary_input():
    x = torch.randn(N, 6, dtype=f64, generator=gen(24))
    R = G.sixd_to_matrix(x)
    assert G.orthogonality_residual(R).max().item() < 1e-14
    assert maxabs(torch.linalg.det(R) - 1) < 1e-14


# ----------------------------------------------------------------------------- Euler

def test_euler_matches_scipy_convention():
    e = torch.rand(1000, 3, dtype=f64, generator=gen(25)) * 2 - 1
    ref = SciRot.from_euler("XYZ", e.numpy()).as_matrix()      # uppercase = intrinsic
    assert np.abs(G.euler_to_matrix(e).numpy() - ref).max() < TOL


def test_euler_roundtrip():
    R = torch.cat([rand_rots(), edge_rotations()])
    e = G.matrix_to_euler(R)
    assert torch.isfinite(e).all()
    assert maxabs(G.euler_to_matrix(e) - R) < TOL
    assert (e[:, 1].abs() <= math.pi / 2).all()


def test_euler_angles_roundtrip_away_from_lock():
    g = gen(26)
    a = (torch.rand(N, dtype=f64, generator=g) * 2 - 1) * (math.pi - 1e-6)
    b = (torch.rand(N, dtype=f64, generator=g) * 2 - 1) * (math.pi / 2 - 1e-3)
    c = (torch.rand(N, dtype=f64, generator=g) * 2 - 1) * (math.pi - 1e-6)
    e = torch.stack([a, b, c], -1)
    assert maxabs(G.matrix_to_euler(G.euler_to_matrix(e)) - e) < 1e-9


# ----------------------------------------------------------------------------- residuals

def test_orthogonality_residuals():
    R = rand_rots()
    assert G.orthogonality_residual(R).max().item() < 1e-14
    q = torch.randn(1000, 4, dtype=f64, generator=gen(27))
    n = q.norm(dim=-1)
    res = G.orthogonality_residual(G.quat_to_matrix_homogeneous(q))
    assert maxabs(res - math.sqrt(3) * (n ** 4 - 1).abs()) < 1e-9
    x = G.matrix_to_sixd(R)
    assert G.orthogonality_residual(G.sixd_to_matrix_raw(x)).max().item() < 1e-14
