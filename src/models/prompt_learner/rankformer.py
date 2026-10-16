import torch
import torch.nn as nn

from .ordinalclip_head import interpolation_weights


class RankFormerPromptLearner(nn.Module):
    """Learnable prompts [SOS][CTX x n][RANK][EOS] with a RankFormer on the rank tokens.

    rank_param:
        "interpolated" — rank tokens are linear interpolations of `num_base`
                         learnable base embeddings (OrdinalCLIP). This is the
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
    ):
        super().__init__()
        self.num_ranks = num_ranks
        self.num_context_tokens = num_context_tokens
        self.rank_param = rank_param
        self.block = block
        self.alpha = alpha

        # dtype of CLIP model
        dtype = clip_model.dtype
        self.dtype = dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Global context tokens
        # [num_context_tokens, ctx_dim]
        ctx_vectors = torch.empty(num_context_tokens, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Rank tokens
        if rank_param == "interpolated":
            base_vectors = torch.empty(num_base, ctx_dim, dtype=dtype)
            nn.init.normal_(base_vectors, std=0.02)
            self.base_embeddings = nn.Parameter(base_vectors)
            self.register_buffer("interp_weights", interpolation_weights(num_ranks, num_base, dtype))
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

        import clip
        # Create a dummy prompt of exactly length: num_context_tokens + 1 (for rank)
        dummy_prompt = " ".join(["X"] * (num_context_tokens + 1))
        prompts = [dummy_prompt for _ in range(num_ranks)]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        device = clip_model.token_embedding.weight.device

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_context_tokens + 1:, :])  # EOS and padding

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
