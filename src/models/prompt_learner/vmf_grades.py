"""Grades as von Mises–Fisher distributions in CLIP's joint space (Direction 3).

Grade k is vMF(μ_k, κ_k) on the unit sphere: μ_k is its mean direction (learned
directly, as in Direction 1, initialised from the zero-shot description
ensemble of Direction 2) and κ_k its concentration. A low κ_k is a broad,
ambiguous grade; a high κ_k a sharply defined one.

Scoring an image embedding z:
- "likelihood" (default): the vMF-mixture posterior
      logit_k = κ_k μ_kᵀz + log C_D(κ_k).
  With equal κ this is CLIP's softmax with scale κ; a broad grade spreads
  probability onto its neighbours, which is the ordinal smoothing that SMS adds
  post hoc.
- "expected": the expected similarity to a prototype drawn from the grade,
      logit_k = s · A_D(κ_k)/A_D(κ⁰) · μ_kᵀz,
  i.e. a per-grade temperature equal to CLIP's scale s at κ⁰.

κ modes: "per_grade" (one κ per grade), "shared" (one κ for all grades) or
"fixed" (κ = κ⁰, not learned).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry_losses import cosine_distance_matrix, ladder_satisfied_frac, step_angles_deg
from ..vmf import log_vmf_normalizer, mean_resultant_length
from .text_lora_prototypes import PROMPT_SETS


@torch.no_grad()
def ensemble_init_means(clip_model, prompt_set: str, num_ranks: int) -> torch.Tensor:
    """Zero-shot grade prototypes [K, D]: normalised mean of each grade's normalised descriptions."""
    import clip

    groups = PROMPT_SETS[prompt_set]
    if len(groups) != num_ranks:
        raise ValueError(f"prompt_set '{prompt_set}' defines {len(groups)} grades, got num_ranks={num_ranks}")
    device = clip_model.text_projection.device
    means = []
    for sentences in groups:
        emb = clip_model.encode_text(clip.tokenize(sentences, truncate=False).to(device)).float()
        means.append(F.normalize(F.normalize(emb, dim=-1).mean(0), dim=0))
    return torch.stack(means).cpu()


class VMFGradeDistributions(nn.Module):
    """vMF grade distributions; forward() returns the normalised means μ̂ [K, D]."""

    direct_prototypes = True

    def __init__(self, init_means: torch.Tensor, kappa_init: float = 100.0,
                 kappa_mode: str = "per_grade", scoring: str = "likelihood"):
        super().__init__()
        if kappa_mode not in ("per_grade", "shared", "fixed"):
            raise ValueError(f"Unknown kappa_mode: {kappa_mode}")
        if scoring not in ("likelihood", "expected"):
            raise ValueError(f"Unknown scoring: {scoring}")
        init = F.normalize(init_means.detach().float(), dim=-1, eps=1e-8)
        K, D = init.shape
        self.num_ranks, self.dim = K, D
        self.kappa_mode, self.scoring = kappa_mode, scoring
        self.t = nn.Parameter(init.clone())
        self.register_buffer("anchors", init.clone())
        self.register_buffer("anchor_lambdas", torch.ones(K))
        self.register_buffer("log_kappa0", torch.tensor(math.log(kappa_init)))
        n_kappa = {"per_grade": K, "shared": 1, "fixed": 1}[kappa_mode]
        log_kappa = torch.full((n_kappa,), math.log(kappa_init))
        if kappa_mode == "fixed":
            self.register_buffer("log_kappa", log_kappa)
        else:
            self.log_kappa = nn.Parameter(log_kappa)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.t, dim=-1, eps=1e-8)

    def kappa(self) -> torch.Tensor:
        """Concentrations [K]."""
        return self.log_kappa.exp().expand(self.num_ranks)

    def score(self, image_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        """Logits [B, K] for L2-normalised image features [B, D]."""
        mu = self().to(image_features.dtype)
        cos = (image_features @ mu.T).float()
        kappa = self.kappa().float()
        if self.scoring == "likelihood":
            # log C_D(κ) is ~ +2000 at κ = 100: subtract the class-independent
            # log C_D(κ⁰) in float64 so the logits keep CLIP's scale and precision
            log_c = log_vmf_normalizer(kappa.double(), self.dim) - log_vmf_normalizer(self.log_kappa0.exp().double(), self.dim)
            return kappa * cos + log_c.float()
        kappa0 = self.log_kappa0.exp().float()
        temp = mean_resultant_length(kappa, self.dim) / mean_resultant_length(kappa0, self.dim)
        return logit_scale.float() * temp * cos

    def kappa_prior(self) -> torch.Tensor:
        """Σ_k (log κ_k − log κ⁰)²: keeps the concentrations near the start in the few-shot regime."""
        return ((self.log_kappa - self.log_kappa0) ** 2).sum()

    def set_anchor_lambdas(self, lambdas: torch.Tensor) -> None:
        self.anchor_lambdas.copy_(lambdas.to(self.anchor_lambdas))

    def no_decay_parameters(self) -> list:
        """The sphere-projected means: no weight decay (it would only shrink their norm)."""
        return [self.t]

    def kappa_parameters(self) -> list:
        """The log-concentrations: own learning rate (train.kappa_lr), no weight decay.

        At the general lr (1e-4) log κ would move by about 1% over a whole run.
        """
        return [self.log_kappa] if isinstance(self.log_kappa, nn.Parameter) else []

    @torch.no_grad()
    def renormalize_(self) -> None:
        self.t.div_(self.t.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def diagnostics(self) -> dict:
        mu = self().float()
        return {
            "distance_matrix": cosine_distance_matrix(mu).tolist(),
            "step_angles_deg": step_angles_deg(mu),
            "anchor_drift": (1.0 - (mu * self.anchors.float()).sum(-1)).tolist(),
            "ladder_satisfied_frac": ladder_satisfied_frac(mu),
            "kappa": self.kappa().float().tolist(),
        }
