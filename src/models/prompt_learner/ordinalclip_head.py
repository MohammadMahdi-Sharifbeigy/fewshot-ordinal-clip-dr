import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if classnames is None:
            classnames = [f"grade {i}" for i in range(num_ranks)]
            
        prompts = [f"a photo of a {name}" for name in classnames]
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
        weights = torch.zeros((self.num_ranks, self.num_base), dtype=self.dtype, device=self.base_embeddings.device)
        for i in range(self.num_ranks):
            # map i in [0, num_ranks-1] to pos in [0, num_base-1]
            pos = i * (self.num_base - 1) / max(1, self.num_ranks - 1)
            left = int(pos)
            right = min(left + 1, self.num_base - 1)
            alpha = pos - left
            if left == right:
                weights[i, left] = 1.0
            else:
                weights[i, left] = 1.0 - alpha
                weights[i, right] = alpha
                
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
