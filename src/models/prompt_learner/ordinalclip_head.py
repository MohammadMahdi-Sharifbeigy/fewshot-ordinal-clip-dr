import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolation_weights(num_ranks: int, num_base: int, dtype=torch.float32) -> torch.Tensor:
    """Linear interpolation weights [num_ranks, num_base] mapping base to rank embeddings."""
    weights = torch.zeros((num_ranks, num_base), dtype=dtype)
    for i in range(num_ranks):
        # map i in [0, num_ranks-1] to pos in [0, num_base-1]
        pos = i * (num_base - 1) / max(1, num_ranks - 1)
        left = int(pos)
        right = min(left + 1, num_base - 1)
        alpha = pos - left
        if left == right:
            weights[i, left] = 1.0
        else:
            weights[i, left] = 1.0 - alpha
            weights[i, right] = alpha
    return weights


class OrdinalCLIPHead(nn.Module):
    """
    OrdinalCLIP's plain interpolated rank-prompt head (Ablation only)
    """
    def __init__(self, clip_model, num_ranks, num_base=3, num_context_tokens=16, classnames=None):
        super().__init__()
        self.num_ranks = num_ranks
        self.num_base = num_base
        self.num_context_tokens = num_context_tokens
        
        dtype = clip_model.dtype
        self.dtype = dtype
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Global context tokens
        ctx_vectors = torch.empty(num_context_tokens, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Base rank embeddings
        base_vectors = torch.empty(num_base, ctx_dim, dtype=dtype)
        nn.init.normal_(base_vectors, std=0.02)
        self.base_embeddings = nn.Parameter(base_vectors)
        
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
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num_ranks, -1, -1)
            
        # Interpolate ranks
        # Weighting: simple linear distance based interpolation
        weights = interpolation_weights(self.num_ranks, self.num_base, self.dtype).to(self.base_embeddings.device)

        # [num_ranks, num_base] @ [num_base, ctx_dim] -> [num_ranks, ctx_dim]
        rank_embeddings = weights @ self.base_embeddings
        rank_embeddings = rank_embeddings.unsqueeze(1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat(
            [
                prefix,  # [num_ranks, 1, ctx_dim]
                ctx,     # [num_ranks, num_context_tokens, ctx_dim]
                rank_embeddings, # [num_ranks, 1, ctx_dim]
                suffix,  # [num_ranks, *, ctx_dim]
            ],
            dim=1,
        )
        return prompts, self.tokenized_prompts
