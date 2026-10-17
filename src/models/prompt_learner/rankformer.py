import torch
import torch.nn as nn

from .ordinalclip_head import context_embeddings, interpolation_weights, prompt_template


class RankFormerPromptLearner(nn.Module):
    """Learnable prompts [SOS][CTX x n][RANK][EOS] with a RankFormer on the rank tokens.

    rank_param:
        "interpolated" — rank tokens interpolate `num_base` learnable base
                         embeddings with OrdinalCLIP's weighting. This is the
                         same parametrisation as OrdinalCLIPHead, so removing
                         the RankFormer ablates the attention block only.
        "free"         — one independent learnable token per rank.
    block:
        "paper"  — L2RCLIP (Wang et al., NeurIPS 2023):
                   R' = (1 - α) R + α · FFN(MSA(LN(R))), attention across ranks.
                   α (the "residual ratio") is not reported in the paper, so it is
                   a config value. The FFN output layer starts at zero, so at
                   initialisation R' = (1 - α) R.
        "simple" — earlier variant: R' = R + MSA(R) (no LN / FFN).
    """

    def __init__(
        self,
        clip_model,
        num_ranks,
        num_context_tokens=16,
        classnames=None,
        attn_heads=4,
        rank_param="interpolated",
        num_base=3,
        block="paper",
        alpha=0.5,
        init_context=None,
        full_stop=False,
        interpolation="linear",
    ):
        super().__init__()
        self.num_ranks = num_ranks
        self.rank_param = rank_param
        self.block = block
        self.alpha = alpha

        # dtype of CLIP model
        dtype = clip_model.dtype
        self.dtype = dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Global context tokens: random or initialised from a sentence
        # [num_context_tokens, ctx_dim]
        ctx_vectors, num_context_tokens = context_embeddings(clip_model, num_context_tokens, init_context, dtype)
        self.num_context_tokens = num_context_tokens
        self.ctx = nn.Parameter(ctx_vectors)

        # Rank tokens
        if rank_param == "interpolated":
            base_vectors = torch.empty(num_base, ctx_dim, dtype=dtype)
            nn.init.normal_(base_vectors, std=0.02)
            self.base_embeddings = nn.Parameter(base_vectors)
            self.register_buffer("interp_weights", interpolation_weights(num_ranks, num_base, dtype, interpolation))
        elif rank_param == "free":
            # [num_ranks, ctx_dim]
            rank_vectors = torch.empty(num_ranks, ctx_dim, dtype=dtype)
            nn.init.normal_(rank_vectors, std=0.02)
            self.rank_embeddings = nn.Parameter(rank_vectors)
        else:
            raise ValueError(f"Unknown rank_param: {rank_param}")

        # RankFormer
        self.attn = nn.MultiheadAttention(embed_dim=ctx_dim, num_heads=attn_heads, batch_first=True, dtype=dtype)
        if block == "paper":
            self.ln = nn.LayerNorm(ctx_dim, dtype=dtype)
            self.ffn = nn.Sequential(
                nn.Linear(ctx_dim, 4 * ctx_dim, dtype=dtype),
                nn.GELU(),
                nn.Linear(4 * ctx_dim, ctx_dim, dtype=dtype),
            )
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)
        elif block != "simple":
            raise ValueError(f"Unknown RankFormer block: {block}")

        tokenized_prompts, prefix, suffix = prompt_template(clip_model, num_ranks, num_context_tokens, full_stop, dtype)
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", prefix)  # SOS
        self.register_buffer("token_suffix", suffix)  # (.) EOS and padding

    def rank_tokens(self) -> torch.Tensor:
        """Rank tokens before the RankFormer, [num_ranks, ctx_dim]."""
        if self.rank_param == "interpolated":
            return self.interp_weights @ self.base_embeddings
        return self.rank_embeddings

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num_ranks, -1, -1)

        # Token-level attention across the rank tokens: [1, num_ranks, ctx_dim]
        r_tokens = self.rank_tokens().unsqueeze(0)

        if self.block == "paper":
            h = self.ln(r_tokens)
            attn_out, _ = self.attn(h, h, h)
            r_hat = (1.0 - self.alpha) * r_tokens + self.alpha * self.ffn(attn_out)
        else:
            attn_out, _ = self.attn(r_tokens, r_tokens, r_tokens)
            r_hat = r_tokens + attn_out

        r_hat = r_hat.squeeze(0).unsqueeze(1)  # [num_ranks, 1, ctx_dim]

        prefix = self.token_prefix
        suffix = self.token_suffix

        # Concat: [SOS] + [CTX] + [RANK] + [EOS...]
        prompts = torch.cat(
            [
                prefix,  # [num_ranks, 1, ctx_dim]
                ctx,     # [num_ranks, num_context_tokens, ctx_dim]
                r_hat,   # [num_ranks, 1, ctx_dim]
                suffix,  # [num_ranks, *, ctx_dim]
            ],
            dim=1,
        )
        return prompts, self.tokenized_prompts
