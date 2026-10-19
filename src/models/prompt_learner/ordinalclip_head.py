import torch
import torch.nn as nn


# OrdinalCLIP (Li et al., NeurIPS 2022), repos/OrdinalCLIP/ordinalclip/models/
# prompt_leaners/rank_prompt_learner.py: every rank is a weighted sum of ALL base
# rank embeddings, with weights decaying with the distance |i - p_b| between the
# rank and each base position p_b = linspace(0, K - 1, num_base).
INTERPOLATION_FUNCTIONS = {
    "linear": lambda d, num_ranks: 1.0 - d / (num_ranks - 1),
    "inv_prop": lambda d, _: 1.0 / (d + 1e-5),
    "normal": lambda d, _: torch.exp(-d * d),
}


def interpolation_weights(num_ranks: int, num_base: int, dtype=torch.float32, interpolation: str = "linear") -> torch.Tensor:
    """OrdinalCLIP interpolation weights [num_ranks, num_base]; each row sums to 1.

    interpolation="none" (num_base == num_ranks) gives one free token per rank,
    i.e. CLIP-DR's PlainPromptLearner.
    """
    if interpolation == "none":
        if num_base != num_ranks:
            raise ValueError(f"interpolation='none' needs num_base == num_ranks, got {num_base} and {num_ranks}")
        return torch.eye(num_ranks, dtype=dtype)
    if interpolation not in INTERPOLATION_FUNCTIONS:
        raise ValueError(f"Unknown interpolation: {interpolation}")
    ranks = torch.arange(num_ranks, dtype=torch.float64)[:, None].repeat(1, num_base)
    if num_base == 1:
        base_pos = torch.linspace(0, num_ranks - 1, 3, dtype=torch.float64)[1:2]
    else:
        base_pos = torch.linspace(0, num_ranks - 1, num_base, dtype=torch.float64)
    weights = INTERPOLATION_FUNCTIONS[interpolation](torch.abs(ranks - base_pos[None]), num_ranks)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(dtype)


def context_embeddings(clip_model, num_context_tokens, init_context, dtype):
    """Context tokens: random (std 0.02) or the word embeddings of `init_context`.

    As in OrdinalCLIP, a text initialisation sets the number of context tokens
    to the length of the tokenised sentence. Returns (embeddings, num_tokens).
    """
    if init_context:
        import clip
        tokens = clip.tokenize(init_context.replace("_", " "))[0]
        n = int(torch.argmax(tokens).item()) - 1  # tokens between <sot> and <eot>
        with torch.no_grad():
            emb = clip_model.token_embedding(tokens.to(clip_model.token_embedding.weight.device)).type(dtype)
        return emb[1:1 + n].detach().clone(), n
    ctx_dim = clip_model.ln_final.weight.shape[0]
    ctx = torch.empty(num_context_tokens, ctx_dim, dtype=dtype)
    nn.init.normal_(ctx, std=0.02)
    return ctx, num_context_tokens


def prompt_template(clip_model, num_ranks, num_context_tokens, full_stop, dtype):
    """Token ids and the fixed [SOS] / [(.) EOS pad] embeddings around [CTX][RANK].

    full_stop=True gives OrdinalCLIP's sentence <sot> ctx rank . <eot>.
    """
    import clip
    dummy_prompt = " ".join(["X"] * (num_context_tokens + 1)) + ("." if full_stop else "")
    tokenized_prompts = torch.cat([clip.tokenize(dummy_prompt) for _ in range(num_ranks)])
    device = clip_model.token_embedding.weight.device
    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)
    prefix = embedding[:, :1, :]                           # SOS
    suffix = embedding[:, 1 + num_context_tokens + 1:, :]  # (.) EOS and padding
    return tokenized_prompts, prefix, suffix


class OrdinalCLIPHead(nn.Module):
    """
    OrdinalCLIP's interpolated rank-prompt head: learnable context + base rank
    embeddings interpolated to all ranks (OrdinalCLIP's RankPromptLearner).
    """
    def __init__(self, clip_model, num_ranks, num_base=3, num_context_tokens=16, classnames=None,
                 init_context=None, full_stop=False, interpolation="linear"):
        super().__init__()
        self.num_ranks = num_ranks
        self.num_base = num_base

        dtype = clip_model.dtype
        self.dtype = dtype

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Global context tokens
        ctx_vectors, num_context_tokens = context_embeddings(clip_model, num_context_tokens, init_context, dtype)
        self.num_context_tokens = num_context_tokens
        self.ctx = nn.Parameter(ctx_vectors)

        # Base rank embeddings
        base_vectors = torch.empty(num_base, ctx_dim, dtype=dtype)
        nn.init.normal_(base_vectors, std=0.02)
        self.base_embeddings = nn.Parameter(base_vectors)
        self.register_buffer("interp_weights", interpolation_weights(num_ranks, num_base, dtype, interpolation))

        tokenized_prompts, prefix, suffix = prompt_template(clip_model, num_ranks, num_context_tokens, full_stop, dtype)
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", prefix)  # SOS
        self.register_buffer("token_suffix", suffix)  # (.) EOS and padding

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num_ranks, -1, -1)

        # [num_ranks, num_base] @ [num_base, ctx_dim] -> [num_ranks, ctx_dim]
        rank_embeddings = self.interp_weights @ self.base_embeddings
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
