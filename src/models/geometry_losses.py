"""Ordinal geometry losses on L2-normalised grade embeddings t̂ ∈ S^{D-1}, [K, D].

All functions expect already-normalised embeddings (the caller normalises in-graph).
d(t̂_i, t̂_j) = 1 − ⟨t̂_i, t̂_j⟩ is the cosine distance.
"""
import math
from typing import Sequence

import torch


def cosine_distance_matrix(t_hat: torch.Tensor) -> torch.Tensor:
    """[K, K] matrix of cosine distances 1 − t̂ t̂ᵀ."""
    return 1.0 - t_hat @ t_hat.T


def _ladder_pairs(K: int):
    """(i, Δ, Δ') with 1 ≤ Δ < Δ' ≤ K−1−i: grade i must be closer to i+Δ than to i+Δ'."""
    for i in range(K):
        for near in range(1, K - i):
            for far in range(near + 1, K - i):
                yield i, near, far


def ladder_loss(t_hat: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Σ max(0, d(i, i+Δ) − d(i, i+Δ') + (Δ' − Δ)·m)."""
    D = cosine_distance_matrix(t_hat)
    terms = [
        torch.relu(D[i, i + near] - D[i, i + far] + (far - near) * margin)
        for i, near, far in _ladder_pairs(t_hat.shape[0])
    ]
    if not terms:
        return t_hat.new_zeros(())
    return torch.stack(terms).sum()


@torch.no_grad()
def ladder_satisfied_frac(t_hat: torch.Tensor) -> float:
    """Fraction of ladder pairs with d(i, i+Δ) < d(i, i+Δ') (no margin)."""
    D = cosine_distance_matrix(t_hat)
    pairs = list(_ladder_pairs(t_hat.shape[0]))
    if not pairs:
        return 1.0
    ok = sum(bool(D[i, i + near] < D[i, i + far]) for i, near, far in pairs)
    return ok / len(pairs)


def path_losses(t_hat: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """(L_smooth, L_align) for the path t̂_0 → … → t̂_{K−1}.

    L_smooth = Σ_k ‖(t̂_{k+1} − t̂_k) − (t̂_k − t̂_{k−1})‖²
    L_align  = Σ_k (1 − cos(v_k, v_{k−1})),  v_k = t̂_{k+1} − t̂_k
    """
    v = t_hat[1:] - t_hat[:-1]  # [K-1, D]
    if v.shape[0] < 2:
        zero = t_hat.new_zeros(())
        return zero, zero
    second = v[1:] - v[:-1]
    l_smooth = (second ** 2).sum()
    cos = (v[1:] * v[:-1]).sum(-1) / (v[1:].norm(dim=-1) * v[:-1].norm(dim=-1) + eps)
    l_align = (1.0 - cos).sum()
    return l_smooth, l_align


def path_loss(t_hat: torch.Tensor, mode: str = "both") -> torch.Tensor:
    """Path loss selected by `mode` ∈ {"smooth", "align", "both"}."""
    if mode not in ("smooth", "align", "both"):
        raise ValueError(f"Unknown path_mode: {mode}")
    l_smooth, l_align = path_losses(t_hat)
    if mode == "smooth":
        return l_smooth
    if mode == "align":
        return l_align
    return l_smooth + l_align


def anchor_loss(t_hat: torch.Tensor, anchors_hat: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Σ_k λ_k ‖t̂_k − t̂⁰_k‖² = Σ_k 2λ_k (1 − ⟨t̂_k, t̂⁰_k⟩)."""
    cos = (t_hat * anchors_hat.to(t_hat)).sum(-1)
    return (2.0 * lambdas.to(t_hat) * (1.0 - cos)).sum()


def inverse_sqrt_freq_lambdas(counts: Sequence[int]) -> torch.Tensor:
    """λ_k ∝ 1/√N_k, normalised to mean 1."""
    counts = [int(c) for c in counts]
    if any(c <= 0 for c in counts):
        raise ValueError(f"inverse_sqrt_freq needs a positive count for every grade, got {counts}")
    lam = torch.tensor([1.0 / math.sqrt(c) for c in counts], dtype=torch.float32)
    return lam / lam.mean()


@torch.no_grad()
def step_angles_deg(t_hat: torch.Tensor) -> list[float]:
    """Angles (degrees) between consecutive path steps v_k and v_{k−1}."""
    v = t_hat[1:] - t_hat[:-1]
    cos = (v[1:] * v[:-1]).sum(-1) / (v[1:].norm(dim=-1) * v[:-1].norm(dim=-1) + 1e-8)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).tolist()
