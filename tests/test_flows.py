"""Correctness of the four flow parameterizations with an ORACLE velocity field.

For a point-mass target x1, the marginal velocity equals the conditional one:
  M1: v(R, t) = log(R^T R1) / (1 - t)   (body frame)
  M2-M4: v(x, t) = (x1 - x) / (1 - t)
Integrating it with the model's own step rule must land exactly on x1.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import geometry as G  # noqa: E402
from flows import REPS, get_rep  # noqa: E402

f64 = torch.float64


def oracle_v(rep, x, x1, t):
    if rep.name == "M1_riemannian":
        return G.so3_log(G.compose_T(x, x1)) / (1 - t)
    return (x1 - x) / (1 - t)


@pytest.mark.parametrize("name", list(REPS))
@pytest.mark.parametrize("nfe", [1, 2, 7, 64])
def test_oracle_integration_hits_target(name, nfe):
    rep = get_rep(name)
    g = torch.Generator().manual_seed(0)
    R0 = G.random_rotation(500, g)
    R1 = G.random_rotation(500, g)
    x, x1 = rep.encode(R0), rep.encode(R1)
    h = 1.0 / nfe
    for k in range(nfe):
        x = rep.step(x, oracle_v(rep, x, x1, k * h), h)
    err = G.geodesic_distance(rep.to_matrix(x), R1)
    assert err.max().item() < 1e-9


@pytest.mark.parametrize("name", list(REPS))
def test_path_endpoints_and_velocity(name):
    rep = get_rep(name)
    g = torch.Generator().manual_seed(1)
    R0, R1 = G.random_rotation(500, g), G.random_rotation(500, g)
    x0, x1 = rep.encode(R0), rep.encode(R1)
    z = torch.zeros(500, dtype=f64)
    xa, _ = rep.path(x0, x1, z)
    xb, _ = rep.path(x0, x1, z + 1)
    assert G.geodesic_distance(rep.to_matrix(xa), R0).max() < 1e-9
    assert G.geodesic_distance(rep.to_matrix(xb), R1).max() < 1e-9
    # finite-difference velocity along the path equals the regression target
    t = torch.rand(500, dtype=f64, generator=g) * 0.9
    dt = 1e-6
    xt, u = rep.path(x0, x1, t)
    xt2, _ = rep.path(x0, x1, t + dt)
    if name == "M1_riemannian":
        fd = G.so3_log(G.compose_T(xt, xt2)) / dt      # body-frame velocity
    else:
        fd = (xt2 - xt) / dt
    assert (fd - u).abs().max().item() < 1e-5
