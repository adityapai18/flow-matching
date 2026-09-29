"""SO(3) / SE(3) geometry primitives. CPU float64 ONLY (enforced).

Conventions (every downstream module relies on these):
- Rotation matrices act on column vectors, shape (..., 3, 3).
- Poses are pairs (R, p) with x_world = R @ x_body + p. Twists xi = (omega, v)
  are 6-vectors, rotation first.
- Quaternions are scalar-first (w, x, y, z), Hamilton convention, unit norm.
- QUATERNION SIGN CONVENTION: q and -q are the same rotation. The canonical
  representative has w > 0. If w == 0 exactly, the first nonzero component of
  (x, y, z) is made positive. `matrix_to_quat` always returns canonical
  quaternions, and `canonicalize_quat` is applied before any Euclidean
  operation on quaternions anywhere in this project.
- 6D representation (Zhou et al. 2019): the first two COLUMNS of R,
  concatenated [R[:, 0], R[:, 1]]. Recovered by Gram-Schmidt.
- Euler angles: intrinsic XYZ, R = Rx(a) @ Ry(b) @ Rz(c), with
  b in [-pi/2, pi/2] and a, c in (-pi, pi]. At gimbal lock (cos b == 0 to
  working precision) we set c = 0 and put all of the in-plane angle into a.
"""
import math

import torch

SMALL_ANGLE = 1e-4               # Taylor branch below this angle
NEAR_PI = math.pi - 1e-4         # dedicated antipodal branch above this angle
GIMBAL_EPS = 1e-12               # cos(b) below this is treated as gimbal lock


def _check(*xs):
    for x in xs:
        if not torch.is_tensor(x) or x.dtype != torch.float64 or x.device.type != "cpu":
            raise TypeError("geometry requires CPU float64 tensors, got "
                            f"{getattr(x, 'dtype', type(x))} on {getattr(x, 'device', '?')}")


def _eye(shape):
    return torch.eye(3, dtype=torch.float64).expand(*shape, 3, 3)


def hat(w):
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix [w]x."""
    _check(w)
    z = torch.zeros_like(w[..., 0])
    x, y, zz = w[..., 0], w[..., 1], w[..., 2]
    return torch.stack([
        torch.stack([z, -zz, y], -1),
        torch.stack([zz, z, -x], -1),
        torch.stack([-y, x, z], -1),
    ], -2)


def vee(W):
    """(..., 3, 3) skew part -> (..., 3). Uses the antisymmetric part of W."""
    _check(W)
    return 0.5 * torch.stack([
        W[..., 2, 1] - W[..., 1, 2],
        W[..., 0, 2] - W[..., 2, 0],
        W[..., 1, 0] - W[..., 0, 1],
    ], -1)


def mT(R):
    return R.transpose(-1, -2)


def compose_T(R1, R2):
    """R1^T @ R2 with a fixed, symmetric summation order.

    For R1 == R2 the result is exactly symmetric in floating point, which makes
    geodesic_distance(R, R) exactly 0 (not merely ~1e-16).
    """
    return (R1.unsqueeze(-1) * R2.unsqueeze(-2)).sum(-3)


# ----------------------------------------------------------------------------- SO(3)

def so3_exp(w):
    """Rotation vector (..., 3) -> rotation matrix (..., 3, 3). Rodrigues."""
    _check(w)
    th2 = (w * w).sum(-1)
    th = th2.sqrt()
    small = th < SMALL_ANGLE
    ths = torch.where(small, torch.ones_like(th), th)
    # A = sin(t)/t, B = (1 - cos t)/t^2 = 2 sin^2(t/2)/t^2 (no cancellation)
    A = torch.where(small, 1 - th2 / 6 + th2 * th2 / 120, torch.sin(ths) / ths)
    B = torch.where(small, 0.5 - th2 / 24 + th2 * th2 / 720,
                    2 * torch.sin(ths / 2) ** 2 / ths ** 2)
    K = hat(w)
    return _eye(w.shape[:-1]) + A[..., None, None] * K + B[..., None, None] * (K @ K)


def so3_angle(R):
    """Rotation angle in [0, pi], computed with atan2 (accurate at 0 and pi)."""
    _check(R)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    s = vee(R).norm(dim=-1)            # = sin(theta)
    c = 0.5 * (tr - 1.0)               # = cos(theta)
    return torch.atan2(s, c)


def so3_log(R):
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3), |result| in [0, pi].

    Three explicit branches:
      theta <  1e-4       : Taylor series of theta/sin(theta)
      1e-4 <= theta <= pi-1e-4 : closed form theta/sin(theta) * vee(R)
      theta >  pi - 1e-4  : axis from the symmetric part (R + R^T)/2 - cos(theta) I
                            = (1 - cos theta) a a^T, sign fixed by vee(R) = sin(theta) a.
    """
    _check(R)
    batch = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    w = vee(Rf)                                  # sin(theta) * a
    s = w.norm(dim=-1)
    c = 0.5 * (Rf[:, 0, 0] + Rf[:, 1, 1] + Rf[:, 2, 2] - 1.0)
    th = torch.atan2(s, c)
    out = torch.empty_like(w)

    small = th < SMALL_ANGLE
    near_pi = th > NEAR_PI
    mid = ~(small | near_pi)

    if small.any():
        t2 = th[small] ** 2
        out[small] = w[small] * (1 + t2 / 6 + 7 * t2 * t2 / 360)[:, None]
    if mid.any():
        out[mid] = w[mid] * (th[mid] / s[mid])[:, None]
    if near_pi.any():
        Rp, cp, wp, tp = Rf[near_pi], c[near_pi], w[near_pi], th[near_pi]
        B = 0.5 * (Rp + mT(Rp)) - cp[:, None, None] * _eye(cp.shape)
        diag = torch.diagonal(B, dim1=-2, dim2=-1)
        k = diag.argmax(-1)
        col = B[torch.arange(B.shape[0]), :, k]          # column k: (1-c) a_k a
        a = col / col.norm(dim=-1, keepdim=True)
        flip = (a * wp).sum(-1) < 0                      # match sign of sin(theta)*a
        a = torch.where(flip[:, None], -a, a)
        out[near_pi] = a * tp[:, None]
    return out.reshape(*batch, 3)


def geodesic_distance(R1, R2):
    """d(R1, R2) = ||log(R1^T R2)|| in radians."""
    _check(R1, R2)
    return so3_log(compose_T(R1, R2)).norm(dim=-1)


# ----------------------------------------------------------------------------- SE(3)

def _se3_V_coeffs(th):
    """B = (1-cos t)/t^2, C = (t - sin t)/t^3 with Taylor branch below SMALL_ANGLE."""
    th2 = th * th
    small = th < SMALL_ANGLE
    ths = torch.where(small, torch.ones_like(th), th)
    B = torch.where(small, 0.5 - th2 / 24 + th2 * th2 / 720,
                    2 * torch.sin(ths / 2) ** 2 / ths ** 2)
    C = torch.where(small, 1.0 / 6 - th2 / 120 + th2 * th2 / 5040,
                    (ths - torch.sin(ths)) / ths ** 3)
    return B, C


def se3_exp(xi):
    """Twist (..., 6) = (omega, v) -> (R (..., 3, 3), p (..., 3))."""
    _check(xi)
    w, v = xi[..., :3], xi[..., 3:]
    R = so3_exp(w)
    B, C = _se3_V_coeffs(w.norm(dim=-1))
    K = hat(w)
    V = _eye(w.shape[:-1]) + B[..., None, None] * K + C[..., None, None] * (K @ K)
    return R, (V @ v.unsqueeze(-1)).squeeze(-1)


def se3_log(R, p):
    """(R, p) -> twist (..., 6) = (omega, v)."""
    _check(R, p)
    w = so3_log(R)
    th = w.norm(dim=-1)
    th2 = th * th
    small = th < SMALL_ANGLE
    ths = torch.where(small, torch.ones_like(th), th)
    # V^{-1} = I - K/2 + E K^2,  E = 1/t^2 - cot(t/2)/(2t)
    E = torch.where(small, 1.0 / 12 + th2 / 720 + th2 * th2 / 30240,
                    1 / ths ** 2 - torch.cos(ths / 2) / torch.sin(ths / 2) / (2 * ths))
    K = hat(w)
    Vinv = _eye(w.shape[:-1]) - 0.5 * K + E[..., None, None] * (K @ K)
    v = (Vinv @ p.unsqueeze(-1)).squeeze(-1)
    return torch.cat([w, v], -1)


# ----------------------------------------------------------------------------- quaternions

def canonicalize_quat(q):
    """Pick the canonical sign of q ~ -q (see module docstring). Deterministic."""
    _check(q)
    w = q[..., 0]
    xyz = q[..., 1:]
    # first nonzero of (x, y, z): index of first True in (xyz != 0)
    nz = xyz != 0
    first = torch.where(nz.any(-1), nz.to(torch.int64).argmax(-1), torch.zeros_like(w, dtype=torch.int64))
    first_val = torch.gather(xyz, -1, first.unsqueeze(-1)).squeeze(-1)
    neg = (w < 0) | ((w == 0) & (first_val < 0))
    return torch.where(neg.unsqueeze(-1), -q, q)


def quat_to_matrix(q):
    """Quaternion (w,x,y,z) -> rotation matrix. Normalizes q first."""
    _check(q)
    q = q / q.norm(dim=-1, keepdim=True)
    return quat_to_matrix_homogeneous(q)


def quat_to_matrix_homogeneous(q):
    """Homogeneous (un-normalized) quaternion map: returns |q|^2 * R(q/|q|).

    Used to measure the orthogonality residual of RAW quaternion model output
    before normalization: R^T R = |q|^4 I.
    """
    _check(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z], -1),
    ], -2)


def matrix_to_quat(R):
    """Rotation matrix -> CANONICAL unit quaternion (w,x,y,z). Shepperd's method."""
    _check(R)
    m = R
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    t = torch.stack([1 + m00 + m11 + m22,
                     1 + m00 - m11 - m22,
                     1 - m00 + m11 - m22,
                     1 - m00 - m11 + m22], -1)
    k = t.argmax(-1)
    r = 0.5 * t.clamp_min(0).sqrt()                    # |w|,|x|,|y|,|z| candidates
    d = 4 * r.clamp_min(1e-300)
    a21, a02, a10 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    s01, s02, s12 = m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1]
    cands = torch.stack([
        torch.stack([r[..., 0], a21 / d[..., 0], a02 / d[..., 0], a10 / d[..., 0]], -1),
        torch.stack([a21 / d[..., 1], r[..., 1], s01 / d[..., 1], s02 / d[..., 1]], -1),
        torch.stack([a02 / d[..., 2], s01 / d[..., 2], r[..., 2], s12 / d[..., 2]], -1),
        torch.stack([a10 / d[..., 3], s02 / d[..., 3], s12 / d[..., 3], r[..., 3]], -1),
    ], -2)
    q = torch.gather(cands, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    return canonicalize_quat(q)


# ----------------------------------------------------------------------------- 6D

def sixd_to_matrix(x):
    """6D (..., 6) = [a1, a2] -> rotation matrix via Gram-Schmidt (columns b1,b2,b3)."""
    _check(x)
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # second pass ("twice is enough"): single-pass classical GS leaves an
    # O(eps / angle(a1, a2)) residual when a1, a2 are nearly parallel
    b2 = b2 - (b1 * b2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)


def sixd_to_matrix_raw(x):
    """Un-projected 6D -> [a1, a2, a1 x a2]. For the pre-projection orthogonality residual."""
    _check(x)
    a1, a2 = x[..., :3], x[..., 3:]
    return torch.stack([a1, a2, torch.linalg.cross(a1, a2, dim=-1)], -1)


def matrix_to_sixd(R):
    _check(R)
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


# ----------------------------------------------------------------------------- Euler XYZ

def euler_to_matrix(e):
    """Intrinsic XYZ Euler (a, b, c) -> R = Rx(a) Ry(b) Rz(c)."""
    _check(e)
    a, b, c = e.unbind(-1)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    return torch.stack([
        torch.stack([cb * cc, -cb * sc, sb], -1),
        torch.stack([ca * sc + sa * sb * cc, ca * cc - sa * sb * sc, -sa * cb], -1),
        torch.stack([sa * sc - ca * sb * cc, sa * cc + ca * sb * sc, ca * cb], -1),
    ], -2)


def matrix_to_euler(R):
    """R -> intrinsic XYZ Euler (a, b, c). Gimbal lock: c = 0."""
    _check(R)
    cb = torch.hypot(R[..., 0, 0], R[..., 0, 1])
    b = torch.atan2(R[..., 0, 2], cb)
    lock = cb < GIMBAL_EPS
    a = torch.where(lock, torch.atan2(R[..., 2, 1], R[..., 1, 1]),
                    torch.atan2(-R[..., 1, 2], R[..., 2, 2]))
    c = torch.where(lock, torch.zeros_like(b), torch.atan2(-R[..., 0, 1], R[..., 0, 0]))
    return torch.stack([a, b, c], -1)


# ----------------------------------------------------------------------------- misc

def orthogonality_residual(R):
    """||R^T R - I||_F. Call on RAW output, before any projection."""
    _check(R)
    return (mT(R) @ R - _eye(R.shape[:-2])).flatten(-2).norm(dim=-1)


def random_rotation(n, generator=None):
    """Haar-uniform rotations (n, 3, 3) via normalized Gaussian quaternions."""
    q = torch.randn(n, 4, dtype=torch.float64, generator=generator)
    return quat_to_matrix(q)
