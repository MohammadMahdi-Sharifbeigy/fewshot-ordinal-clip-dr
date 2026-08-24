import torch
import torch.nn as nn

class RankFormerPromptLearner(nn.Module):
    def __init__(self, clip_model, num_ranks, num_context_tokens=16, classnames=None, attn_heads=4):
        super().__init__()
        self.num_ranks = num_ranks
        self.num_context_tokens = num_context_tokens
        
        # dtype of CLIP model
        dtype = clip_model.dtype
        self.dtype = dtype
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Global context tokens
        # [num_context_tokens, ctx_dim]
        ctx_vectors = torch.empty(num_context_tokens, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Rank-specific tokens
        # [num_ranks, ctx_dim]
        rank_vectors = torch.empty(num_ranks, ctx_dim, dtype=dtype)
        nn.init.normal_(rank_vectors, std=0.02)
        self.rank_embeddings = nn.Parameter(rank_vectors)
        
        # RankFormer Attention
        self.attn = nn.MultiheadAttention(embed_dim=ctx_dim, num_heads=attn_heads, batch_first=True, dtype=dtype)
        
        # "A photo of a [rank]"
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
            
        rank_embeddings = self.rank_embeddings.unsqueeze(1) # [num_ranks, 1, ctx_dim]
        
        # Token-level attention across rank embeddings
        # We want rank tokens to attend to each other
        # [num_ranks, 1, ctx_dim] -> [1, num_ranks, ctx_dim]
        r_tokens = self.rank_embeddings.unsqueeze(0)
        
        # Self attention: query=r_tokens, key=r_tokens, value=r_tokens
        attn_out, _ = self.attn(r_tokens, r_tokens, r_tokens)
        
        # Residual blending
        r_hat = r_tokens + attn_out  # [1, num_ranks, ctx_dim]
        r_hat = r_hat.squeeze(0).unsqueeze(1) # [num_ranks, 1, ctx_dim]
        
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
