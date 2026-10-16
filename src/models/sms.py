import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


class SMS(nn.Module):
    """Similarity Matrix Smooth (SMS), following CLIP-DR (Yu et al., MICCAI 2024).

    CLIP-DR adapts Feature Distribution Smoothing (FDS, Yang et al., ICML 2021)
    to the K-dim similarity vector s_i of each image. For every grade j it keeps
    the per-dimension mean μ_j and variance Σ_j of the similarity vectors of
    grade-j images, smooths both over neighbouring grades with a Gaussian kernel,
    and re-normalises the training similarities of grade-j images:

        s̃_i = (s_i − μ_j) · sqrt(clip(Σ̃_j / Σ_j, 0.5, 2)) + μ̃_j,   y_i = j

    Statistics are refreshed once per epoch (momentum-averaged over epochs) and
    held fixed within an epoch. The calibration needs labels, so it is applied
    to the training similarities before the losses only; at inference the
    similarities pass through unchanged, as in FDS.

    Defaults (Gaussian kernel, size 5, sigma 2, momentum 0.9, clip [0.5, 2])
    are the values in the official CLIP-DR code (clipdr/runnerr/fds.py,
    utils_fds.calibrate_mean_var). There, FDS uses bucket_start=3, which skips
    grades 0-2 (a leftover from FDS's age-regression setting), and the calls
    are commented out; here every grade is its own bucket.
    """

    def __init__(
        self,
        num_ranks: int = 5,
        kernel_size: int = 5,
        sigma: float = 2.0,
        momentum: float = 0.9,
        clip_min: float = 0.5,
        clip_max: float = 2.0,
    ):
        super().__init__()
        self.num_ranks = num_ranks
        self.momentum = momentum
        self.clip_min = clip_min
        self.clip_max = clip_max
        # Reflect padding needs pad < number of grades
        self.half_ks = min((kernel_size - 1) // 2, num_ranks - 1)
        kernel_size = 2 * self.half_ks + 1

        # Same construction as FDS: Gaussian-filtered delta, normalised to sum 1
        base = np.zeros(kernel_size, dtype=np.float32)
        base[self.half_ks] = 1.0
        window = gaussian_filter1d(base, sigma=sigma)
        self.register_buffer("kernel", torch.tensor(window / window.sum(), dtype=torch.float32))

        K = num_ranks
        # Rows: grade of the image, columns: similarity dimension
        self.register_buffer("running_mean", torch.zeros(K, K))
        self.register_buffer("running_var", torch.ones(K, K))
        self.register_buffer("smoothed_mean", torch.zeros(K, K))
        self.register_buffer("smoothed_var", torch.ones(K, K))
        self.register_buffer("seen", torch.zeros(K, dtype=torch.bool))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

        self.reset_stats()

    # ------------------------------------------------------------------
    # Per-epoch accumulation
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        """Clear the current-epoch accumulators."""
        K = self.num_ranks
        self._sum = torch.zeros(K, K, dtype=torch.float64)
        self._sq_sum = torch.zeros(K, K, dtype=torch.float64)
        self._cnt = torch.zeros(K, dtype=torch.float64)

    def update_stats(self, S: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate per-grade statistics of the raw training similarities.

        Args:
            S:       [B, K] similarity matrix (logit-scaled)
            targets: [B]    ground-truth grades
        """
        S = S.detach().double().cpu()
        onehot = F.one_hot(targets.detach().cpu().long(), self.num_ranks).double()  # [B, K]
        self._sum += onehot.T @ S
        self._sq_sum += onehot.T @ (S * S)
        self._cnt += onehot.sum(dim=0)

    def compute_smoothed_stats(self) -> None:
        """End of epoch: update running stats, then smooth them over grades."""
        device = self.running_mean.device
        has = self._cnt > 0
        if not has.any():
            return

        cnt = self._cnt.clamp(min=1).unsqueeze(1)
        mean = self._sum / cnt
        var = (self._sq_sum / cnt - mean * mean).clamp(min=0.0)
        # Unbiased variance when a grade has more than one sample (as in FDS)
        unbias = torch.where(self._cnt > 1, self._cnt / (self._cnt - 1).clamp(min=1), torch.ones_like(self._cnt))
        var = var * unbias.unsqueeze(1)

        mean = mean.float().to(device)
        var = var.float().to(device)
        has = has.to(device)

        factor = 0.0 if self.num_updates.item() == 0 else self.momentum
        upd_mean = (1 - factor) * mean + factor * self.running_mean
        upd_var = (1 - factor) * var + factor * self.running_var
        # Grades seen for the first time take the current statistics directly
        first = has & ~self.seen
        upd_mean[first] = mean[first]
        upd_var[first] = var[first]

        self.running_mean[has] = upd_mean[has]
        self.running_var[has] = upd_var[has]
        self.seen |= has
        self.num_updates += 1

        self.smoothed_mean.copy_(self._smooth_over_grades(self.running_mean))
        self.smoothed_var.copy_(self._smooth_over_grades(self.running_var))

    def _smooth_over_grades(self, stats: torch.Tensor) -> torch.Tensor:
        """Convolve [K_grades, K_dims] statistics with the kernel along the grade axis."""
        x = stats.T.unsqueeze(1)  # [K_dims, 1, K_grades]
        x = F.pad(x, (self.half_ks, self.half_ks), mode="reflect")
        x = F.conv1d(x, self.kernel.view(1, 1, -1))
        return x.squeeze(1).T  # [K_grades, K_dims]

    # ------------------------------------------------------------------
    # Calibration (training only)
    # ------------------------------------------------------------------

    def forward(self, S: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """Calibrate the similarities of each image with its grade's statistics.

        Args:
            S:       [B, K] similarity matrix
            targets: [B] ground-truth grades; None (inference) returns S unchanged
        Returns:
            [B, K] calibrated similarities
        """
        if targets is None or self.num_updates.item() == 0:
            return S

        targets = targets.long()
        m1 = self.running_mean[targets].to(S.dtype)
        v1 = self.running_var[targets].to(S.dtype)
        m2 = self.smoothed_mean[targets].to(S.dtype)
        v2 = self.smoothed_var[targets].to(S.dtype)

        valid = (v1 > 1e-10) & (v2 >= 0) & self.seen[targets].unsqueeze(1)
        factor = torch.where(valid, v2 / v1.clamp(min=1e-10), torch.ones_like(v1))
        factor = factor.clamp(self.clip_min, self.clip_max)
        S_cal = (S - m1) * factor.sqrt() + m2
        return torch.where(valid, S_cal, S)
