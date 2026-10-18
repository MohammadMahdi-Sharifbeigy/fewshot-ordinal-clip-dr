"""Ordinal losses on the description embeddings of Direction 2."""
import torch
import torch.nn.functional as F


def description_order_loss(desc: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor,
                           logit_scale, rank_criterion) -> torch.Tensor:
    """Each description must be closest to its own grade and less similar the further a grade is.

    desc:       [N, D] normalised description embeddings
    labels:     [N]    grade of each description
    prototypes: [K, D] normalised grade prototypes
    Loss = CE(s, y) + ranking-aware(s, y) with s = logit_scale · desc · prototypesᵀ, i.e. the
    image-side CE + CLIP-DR ranking pair applied to the descriptions, which keeps each
    grade's description cloud separated and ordered along the grade axis.
    """
    s = logit_scale * desc @ prototypes.T
    return F.cross_entropy(s, labels) + rank_criterion(s, labels)
