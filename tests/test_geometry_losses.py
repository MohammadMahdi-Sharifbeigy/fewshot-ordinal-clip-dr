import math

import pytest
import torch
import torch.nn.functional as F

from src.models.geometry_losses import (
    anchor_loss,
    cosine_distance_matrix,
    inverse_sqrt_freq_lambdas,
    ladder_loss,
    ladder_satisfied_frac,
    path_loss,
    path_losses,
    step_angles_deg,
)


def arc(K=5, D=8, span=math.pi / 2):
    # K points on a great circle, equally spaced: perfectly ordinal & smooth
    th = torch.linspace(0, span, K)
    t = torch.zeros(K, D)
    t[:, 0] = th.cos()
    t[:, 1] = th.sin()
    return t


def test_ladder_zero_on_wide_arc():
    assert ladder_loss(arc(span=math.pi * 0.9), margin=0.0).item() == 0.0
    assert ladder_satisfied_frac(arc()) == 1.0


def test_ladder_positive_when_order_swapped():
    t = arc()[[0, 2, 1, 3, 4]]
    assert ladder_loss(t, margin=0.05).item() > 0
    assert ladder_satisfied_frac(t) < 1.0


def test_ladder_margin_term():
    # equal distances -> each pair violated by exactly (Δ'-Δ)m
    t = F.normalize(torch.eye(3), dim=-1)  # K=3, all pairwise d = 1
    # pairs: i=0: (Δ=1, Δ'=2) -> margin*1 ; i=1: none ; total = 0.05
    assert math.isclose(ladder_loss(t, margin=0.05).item(), 0.05, rel_tol=1e-5)


def test_path_closed_form_on_equally_spaced_arc():
    # great-circle step δ: second difference = 2(cosδ-1)·t_k, adjacent steps differ by angle δ
    d = math.pi / 2 / 4
    s, a = path_losses(arc())
    assert math.isclose(s.item(), 3 * (2 * (1 - math.cos(d))) ** 2, rel_tol=1e-4)
    assert math.isclose(a.item(), 3 * (1 - math.cos(d)), rel_tol=1e-4)


def test_path_detects_zigzag():
    t = arc()
    t[2] = F.normalize(t[2] + torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0]), dim=0)
    s, a = path_losses(t)
    s0, a0 = path_losses(arc())
    assert s > s0 and a > a0
    assert torch.isclose(path_loss(t, "both"), s + a)


def test_path_mode_invalid():
    with pytest.raises(ValueError):
        path_loss(arc(), "bogus")


def test_anchor_loss_values():
    t = arc()
    assert anchor_loss(t, t, torch.ones(5)).item() < 1e-6
    t2 = arc()
    t2[0] = -t2[0]  # cos = -1 -> 2*λ*2 = 4
    assert math.isclose(anchor_loss(t2, arc(), torch.ones(5)).item(), 4.0, rel_tol=1e-5)


def test_inverse_sqrt_freq_lambdas():
    lam = inverse_sqrt_freq_lambdas([100, 25, 25, 4, 4])
    assert math.isclose(lam.mean().item(), 1.0, rel_tol=1e-6)
    assert lam[3] > lam[1] > lam[0] and math.isclose((lam[3] / lam[0]).item(), 5.0, rel_tol=1e-5)


def test_inverse_sqrt_freq_lambdas_rejects_zero_count():
    with pytest.raises(ValueError):
        inverse_sqrt_freq_lambdas([10, 0, 5, 5, 5])


def test_losses_are_differentiable():
    t = arc().requires_grad_(True)
    (ladder_loss(t) + path_loss(t, "both") + anchor_loss(t, arc(), torch.ones(5))).backward()
    assert t.grad is not None


def test_step_angles_and_distance_matrix():
    assert len(step_angles_deg(arc())) == 3
    D = cosine_distance_matrix(arc())
    assert D.shape == (5, 5) and torch.allclose(D.diag(), torch.zeros(5), atol=1e-6)
