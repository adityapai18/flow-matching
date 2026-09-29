"""The four rotation parameterizations. Everything here is CPU float64.

Each class defines, for ONE waypoint's rotation (batched over leading dims):
  encode(R)          data/noise rotation -> flow state
  path(x0, x1, t)    conditional path state x_t and regression target u_t
  step(x, v, h)      one ODE step of size h with network velocity v
  features(x)        what the network sees of the state
  velocity(x, out)   network output (out_dim per waypoint) -> state velocity (vel_dim)
  raw_matrix(x)      matrix implied by the RAW state, before any projection
  to_matrix(x)       final (projected) rotation matrix

Translation is NOT handled here: it is Euclidean and identical for all models.
Source distribution is the same for all four: Haar-uniform rotations, mapped
into each representation with `encode`.
"""
import torch

import geometry as G


class Riemannian:
    """M1: state on SO(3); geodesic path R_t = R0 exp(t log(R0^T R1));
    velocity is the constant body-frame tangent log(R0^T R1); exp-map Euler step.

    Vector-field parameterization follows the Riemannian FM recipe for embedded
    manifolds (Chen & Lipman 2023): the network outputs an ambient 3x3 matrix A
    and the velocity is its projection onto the tangent space at R,
    R skew(R^T A), i.e. body-frame Omega = vee(R^T A)."""
    name = "M1_riemannian"
    state_shape = (3, 3)
    feat_dim = 9
    out_dim = 9
    vel_dim = 3

    def encode(self, R):
        return R

    def path(self, x0, x1, t):
        u = G.so3_log(G.compose_T(x0, x1))
        return x0 @ G.so3_exp(t[..., None] * u), u

    def step(self, x, v, h):
        return x @ G.so3_exp(h * v)

    def features(self, x):
        return x.flatten(-2)

    def velocity(self, x, out):
        return G.vee(G.mT(x) @ out.reshape(*out.shape[:-1], 3, 3))

    def raw_matrix(self, x):
        return x

    def to_matrix(self, x):
        return x                      # stays on SO(3) by construction; no projection applied


class _Euclidean:
    """Straight-line path x_t = (1-t) x0 + t x1, target x1 - x0, Euler step."""
    def path(self, x0, x1, t):
        t = t[..., None]
        return (1 - t) * x0 + t * x1, x1 - x0

    def step(self, x, v, h):
        return x + h * v

    def features(self, x):
        return x

    def velocity(self, x, out):
        return out


class Euclid6D(_Euclidean):
    """M2: 6D (first two columns); Gram-Schmidt only at the end."""
    name = "M2_euclid6d"
    state_shape = (6,)
    feat_dim = out_dim = vel_dim = 6

    def encode(self, R):
        return G.matrix_to_sixd(R)

    def raw_matrix(self, x):
        return G.sixd_to_matrix_raw(x)          # [a1, a2, a1 x a2]

    def to_matrix(self, x):
        return G.sixd_to_matrix(x)


class Quat(_Euclidean):
    """M3: canonical quaternion (w >= 0, see geometry.py); normalize only at the end."""
    name = "M3_quat"
    state_shape = (4,)
    feat_dim = out_dim = vel_dim = 4

    def encode(self, R):
        return G.matrix_to_quat(R)              # canonicalized

    def raw_matrix(self, x):
        return G.quat_to_matrix_homogeneous(x)  # |q|^2 R(q/|q|)

    def to_matrix(self, x):
        return G.quat_to_matrix(x)


class Euler(_Euclidean):
    """M4: intrinsic XYZ Euler angles. Any angle triple is a valid rotation,
    so the pre-projection orthogonality residual is structurally ~1e-16."""
    name = "M4_euler"
    state_shape = (3,)
    feat_dim = out_dim = vel_dim = 3

    def encode(self, R):
        return G.matrix_to_euler(R)

    def raw_matrix(self, x):
        return G.euler_to_matrix(x)

    def to_matrix(self, x):
        return G.euler_to_matrix(x)


REPS = {c.name: c for c in (Riemannian, Euclid6D, Quat, Euler)}


def get_rep(name):
    return REPS[name]()
