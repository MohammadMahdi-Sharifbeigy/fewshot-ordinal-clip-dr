import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingAwareLoss(nn.Module):
    """CLIP-DR Ranking Aware Loss — fully vectorized.

    For each sample with ground-truth rank j, enforces that:
      - sim[j] > sim[j+1] > ... (rightward monotonicity)
      - sim[j] > sim[j-1] > ... (leftward monotonicity)

    Each pairwise constraint uses -log σ(s_k − s_{k±1}).
    """

    def __init__(self, num_ranks: int = 5, tau: float = 1.0):
        super().__init__()
        self.num_ranks = num_ranks
        self.tau = tau

    def forward(self, sim: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sim:     [B, K]  — cosine similarity logits (already scaled)
            targets: [B]     — ground-truth rank indices in [0, K-1]
        Returns:
            Scalar loss.
        """
        B, K = sim.shape
        s = sim / self.tau  # [B, K]

        # --- Rightward loss: for each j' in [target, K-2], penalise s[j'] < s[j'+1] ---
        # diff_right[b, r] = s[b, r] - s[b, r+1]  for r in [0, K-2]
        diff_right = s[:, :-1] - s[:, 1:]  # [B, K-1]

        # Mask: include pair (b, r) only when r >= target[b]
        rank_idx = torch.arange(K - 1, device=sim.device).unsqueeze(0)  # [1, K-1]
        mask_right = rank_idx >= targets.unsqueeze(1)  # [B, K-1]

        right_loss = (-F.logsigmoid(diff_right) * mask_right).sum(dim=1)  # [B]

        # --- Leftward loss: for each j' in [1, target], penalise s[j'] < s[j'-1] ---
        # diff_left[b, r] = s[b, r] - s[b, r-1]  for r in [1, K-1]
        diff_left = s[:, 1:] - s[:, :-1]  # [B, K-1]  (s[r] - s[r-1])

        # Mask: include pair (b, r) only when r <= target[b]
        # pair index r here corresponds to the pair (r, r-1), r in [1, K-1]
        rank_idx_left = torch.arange(1, K, device=sim.device).unsqueeze(0)  # [1, K-1]
        mask_left = rank_idx_left <= targets.unsqueeze(1)  # [B, K-1]

        left_loss = (-F.logsigmoid(diff_left) * mask_left).sum(dim=1)  # [B]

        return (right_loss + left_loss).mean()


class OrdinalCLIPKLLoss(nn.Module):
    """OrdinalCLIP's text-to-image KL loss (repos/OrdinalCLIP/ordinalclip/runner/runner.py).

    Each rank prototype's similarities over the batch, softmax-normalised, are
    pulled towards a near-uniform distribution over the images of that rank.
    Ranks absent from the batch are ignored.
    """

    def __init__(self, num_ranks: int = 5):
        super().__init__()
        self.num_ranks = num_ranks
        self.kl = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, K] logit-scaled similarities
            targets: [B]    ground-truth ranks
        """
        y_t = F.one_hot(targets, self.num_ranks).t().float()  # [K, B]
        present = y_t.sum(-1) > 0
        target = (y_t * 10.0).softmax(-1)
        target[~present] = 0
        return self.kl(F.log_softmax(logits.t(), dim=-1), target) / present.sum()


class CrossModalPairwiseLoss(nn.Module):
    """L2RCLIP Cross-Modal Pairwise CE Loss — fully vectorized.

    Penalises images being closer to wrong-rank prototypes, weighted by
    the ordinal distance |y_i − j|.

    Similarities are computed on the same scale as the main logits
    (logit_scale · cos / tau). On raw cosines (logit_scale = 1) the pairwise
    differences are ~0.01, so −log σ(Δ) collapses to a near-constant term
    with almost no ranking signal.
    """

    def __init__(self, num_ranks: int = 5, tau: float = 1.0):
        super().__init__()
        self.num_ranks = num_ranks
        self.tau = tau

    def forward(
        self,
        image_feats: torch.Tensor,
        rank_prototypes: torch.Tensor,
        targets: torch.Tensor,
        logit_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            image_feats:     [B, d]   — L2-normalised image embeddings
            rank_prototypes: [K, d]   — L2-normalised text embeddings
            targets:         [B]      — ground-truth rank indices in [0, K-1]
            logit_scale:     CLIP logit scale (exp of the learnt log-scale)
        Returns:
            Scalar loss.
        """
        B = image_feats.size(0)
        K = self.num_ranks

        sim = logit_scale * image_feats @ rank_prototypes.T / self.tau  # [B, K]

        # Ground-truth similarities: sim_yi[b] = sim[b, targets[b]]
        sim_yi = sim[torch.arange(B, device=sim.device), targets]  # [B]

        # Pairwise CE: -log σ(sim_yi - sim_j) for each j != y_i
        # Broadcast: [B, 1] - [B, K] = [B, K]
        pce = -F.logsigmoid(sim_yi.unsqueeze(1) - sim)  # [B, K]

        # Ordinal distance weights: |y_i - j|
        # [1, K] and [B, 1] broadcast to [B, K]
        ranks = torch.arange(K, device=sim.device).unsqueeze(0)  # [1, K]
        weights = (targets.unsqueeze(1) - ranks).abs().float()    # [B, K]

        # Zero out the diagonal (j == y_i term — no self-penalty)
        diag_mask = torch.ones(B, K, device=sim.device)
        diag_mask[torch.arange(B, device=sim.device), targets] = 0.0

        loss = (weights * pce * diag_mask).sum(dim=1).mean()
        return loss
