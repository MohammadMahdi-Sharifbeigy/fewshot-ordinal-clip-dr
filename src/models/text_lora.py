"""LoRA for CLIP's text transformer (Direction 2).

CLIP's text blocks use nn.MultiheadAttention with one packed q/k/v projection,
which PEFT cannot target per projection. Following CLIP-LoRA (Zanella & Ben
Ayed, CVPRW 2024), each block's attention is replaced by a module that keeps
the frozen weights and adds low-rank updates

    W' x = W x + (α / r) · B A dropout(x),   B initialised to zero,

to the chosen projections (q, k, v and optionally the output projection o).
With B = 0 the block computes exactly the original attention.
"""
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_PARAM_TAG = "tlora_"  # text-LoRA parameter names contain this tag


class LoRAMultiheadAttention(nn.Module):
    """Self-attention with frozen CLIP weights plus LoRA updates (sequence-first layout)."""

    def __init__(self, mha: nn.MultiheadAttention, r: int, alpha: float, dropout: float,
                 targets: Sequence[str] = ("q", "k", "v")):
        super().__init__()
        if not mha._qkv_same_embed_dim or mha.batch_first:
            raise ValueError("expected CLIP's packed, sequence-first nn.MultiheadAttention")
        unknown = set(targets) - {"q", "k", "v", "o"}
        if unknown:
            raise ValueError(f"Unknown LoRA targets: {sorted(unknown)}")
        self.base = mha
        self.embed_dim, self.num_heads = mha.embed_dim, mha.num_heads
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        d = self.embed_dim
        # Same device / dtype as the frozen weights (CLIP may already be on the GPU)
        factory = {"device": mha.in_proj_weight.device, "dtype": mha.in_proj_weight.dtype}
        self.tlora_A = nn.ParameterDict()
        self.tlora_B = nn.ParameterDict()
        for t in targets:
            a = torch.empty(r, d, **factory)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.tlora_A[t] = nn.Parameter(a)
            self.tlora_B[t] = nn.Parameter(torch.zeros(d, r, **factory))

    def _delta(self, x: torch.Tensor, target: str):
        if target not in self.tlora_A:
            return None
        a, b = self.tlora_A[target], self.tlora_B[target]
        return F.linear(F.linear(self.dropout(x), a.to(x.dtype)), b.to(x.dtype)) * self.scaling

    def _project(self, x, weight, bias, target):
        y = F.linear(x, weight, bias)
        delta = self._delta(x, target)
        return y if delta is None else y + delta.to(y.dtype)

    def forward(self, query, key=None, value=None, need_weights=False, attn_mask=None, **kwargs):
        # CLIP's text blocks call attn(x, x, x, ...): self-attention on [L, N, d]
        x = query
        L, N, d = x.shape
        w, b = self.base.in_proj_weight, self.base.in_proj_bias
        q = self._project(x, w[:d], b[:d], "q")
        k = self._project(x, w[d:2 * d], b[d:2 * d], "k")
        v = self._project(x, w[2 * d:], b[2 * d:], "v")

        h, hd = self.num_heads, d // self.num_heads
        def split(t):  # [L, N, d] -> [N, h, L, hd]
            return t.reshape(L, N, h, hd).permute(1, 2, 0, 3)
        mask = attn_mask.to(q.dtype) if attn_mask is not None and attn_mask.is_floating_point() else attn_mask
        o = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
        o = o.permute(2, 0, 1, 3).reshape(L, N, d)
        out = self._project(o, self.base.out_proj.weight, self.base.out_proj.bias, "o")
        return out, None


def attach_text_lora(clip_model, r: int = 2, alpha: float = 1.0, dropout: float = 0.25,
                     targets: Sequence[str] = ("q", "k", "v")) -> int:
    """Wrap every attention block of CLIP's text transformer; returns the number of LoRA parameters."""
    for block in clip_model.transformer.resblocks:
        if not isinstance(block.attn, LoRAMultiheadAttention):
            block.attn = LoRAMultiheadAttention(block.attn, r, alpha, dropout, targets)
    return sum(p.numel() for n, p in clip_model.transformer.named_parameters() if LORA_PARAM_TAG in n)


def text_lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """The text-LoRA parameters inside `module`."""
    return [p for n, p in module.named_parameters() if LORA_PARAM_TAG in n]
