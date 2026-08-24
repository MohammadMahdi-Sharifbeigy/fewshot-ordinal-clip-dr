import torch
import torch.nn as nn


class SMS(nn.Module):
    """Similarity Matrix Smoothing (SMS) calibration module.

    After each training epoch, computes per-rank mean similarity statistics
    and smooths them with a Gaussian kernel over the rank axis. At inference,
    the raw similarity columns are shifted toward the smoothed mean to reduce
    overconfidence artefacts.

    The calibration formula (column-wise mean shift):
        S̃[:, j] = S[:, j] - μ_j + μ̃_j
    where μ_j is the raw per-rank mean and μ̃_j = Σ_{j'} K(j,j') μ_{j'} is the
    kernel-smoothed mean. This is numerically stable and cannot produce NaN.
    """

    def __init__(self, num_ranks: int = 5, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.num_ranks = num_ranks

        # Build symmetric Gaussian kernel over rank axis — pure torch, no numpy
        idx = torch.arange(num_ranks, dtype=torch.float32)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()   # [K, K]
        # Zero out entries beyond half the kernel window
        window = kernel_size // 2
        kernel = torch.where(
            dist <= window,
            torch.exp(-dist ** 2 / (2.0 * sigma ** 2)),
            torch.zeros_like(dist),
        )
        # Row-normalise so each row sums to 1
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        self.register_buffer("kernel", kernel)  # [K, K]

        # Running per-rank mean similarities (updated each epoch)
        self.register_buffer("mu_raw", torch.zeros(num_ranks))    # [K]
        self.register_buffer("mu_smooth", torch.zeros(num_ranks)) # [K]

        self.reset_stats()

    # ------------------------------------------------------------------
    # Stats accumulation (called during training_step)
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        """Reset accumulators at the start of each training epoch."""
        # Store (sum, count) per rank for streaming mean computation
        self._sum = [torch.zeros(self.num_ranks) for _ in range(self.num_ranks)]
        self._cnt = [0] * self.num_ranks

    def update_stats(self, S: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate per-rank column-mean statistics.

        Args:
            S:       [B, K] raw similarity matrix
            targets: [B]    ground-truth rank labels
        """
        S_detached = S.detach().cpu()
        targets_cpu = targets.cpu()
        for rank in range(self.num_ranks):
            mask = targets_cpu == rank        # [B] bool
            if mask.any():
                self._sum[rank] = self._sum[rank] + S_detached[mask].sum(dim=0)
                self._cnt[rank] += mask.sum().item()

    # ------------------------------------------------------------------
    # Epoch-end smoothing update
    # ------------------------------------------------------------------

    def compute_smoothed_stats(self) -> None:
        """Compute kernel-smoothed per-rank means at epoch end."""
        device = self.kernel.device

        mu = torch.zeros(self.num_ranks, device=device)
        for rank in range(self.num_ranks):
            if self._cnt[rank] > 0:
                mu[rank] = (
                    self._sum[rank].to(device) / self._cnt[rank]
                ).mean()  # scalar: mean over K columns for this rank's samples

        self.mu_raw = mu                          # [K]
        self.mu_smooth = self.kernel @ mu         # [K]  — kernel-smoothed means

    # ------------------------------------------------------------------
    # Inference calibration
    # ------------------------------------------------------------------

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """Apply column-wise mean-shift calibration.

        Args:
            S: [B, K] similarity matrix
        Returns:
            S_tilde: [B, K] calibrated similarities
        """
        # Skip calibration if we haven't gathered any stats yet
        if self.mu_raw.abs().sum() < 1e-9:
            return S

        # Column-wise shift: pull each column toward its smoothed mean
        # S̃[:, j] = S[:, j] - μ_raw[j] + μ_smooth[j]
        shift = self.mu_smooth - self.mu_raw  # [K]
        return S + shift.unsqueeze(0)         # [B, K]
