import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry_losses import cosine_distance_matrix, ladder_satisfied_frac, step_angles_deg


# Starting sentences for the five DR grades (ICDR scale)
GRADE_PROMPTS = {
    "icdr_clinical": [
        "A fundus photograph showing no diabetic retinopathy, with normal retinal vasculature "
        "and no microaneurysms or hemorrhages.",
        "A fundus photograph showing mild non-proliferative diabetic retinopathy, characterized "
        "by the presence of microaneurysms only.",
        "A fundus photograph showing moderate non-proliferative diabetic retinopathy, with multiple "
        "microaneurysms, dot and blot hemorrhages, hard exudates, or cotton wool spots, less than severe.",
        "A fundus photograph showing severe non-proliferative diabetic retinopathy, with extensive "
        "intraretinal hemorrhages in four quadrants, venous beading, or intraretinal microvascular "
        "abnormalities.",
        "A fundus photograph showing proliferative diabetic retinopathy, characterized by "
        "neovascularization on the optic disc or elsewhere, vitreous hemorrhage, or preretinal hemorrhage.",
    ],
    "plain": [
        "A fundus photograph with no diabetic retinopathy.",
        "A fundus photograph with mild non-proliferative diabetic retinopathy.",
        "A fundus photograph with moderate non-proliferative diabetic retinopathy.",
        "A fundus photograph with severe non-proliferative diabetic retinopathy.",
        "A fundus photograph with proliferative diabetic retinopathy.",
    ],
}


@torch.no_grad()
def build_init_embeddings(clip_model, init_mode: str, num_ranks: int, seed: int) -> torch.Tensor:
    """Initial grade embeddings [K, D], L2-normalised, float32 on CPU.

    "icdr_clinical" / "plain": the grade sentences through the frozen CLIP text encoder.
    "random_ortho":            K seeded random orthonormal vectors (no text encoder).
    """
    D = clip_model.text_projection.shape[1]
    if init_mode == "random_ortho":
        g = torch.Generator().manual_seed(int(seed))
        q, _ = torch.linalg.qr(torch.randn(D, num_ranks, generator=g))
        return q.T.contiguous().float()

    if init_mode not in GRADE_PROMPTS:
        raise ValueError(f"Unknown init_mode: {init_mode}")
    sentences = GRADE_PROMPTS[init_mode]
    if num_ranks != len(sentences):
        raise ValueError(f"init_mode '{init_mode}' defines {len(sentences)} grades, got num_ranks={num_ranks}")

    import clip

    # truncate=False: clip.tokenize raises if a sentence exceeds the 77-token context
    tokens = clip.tokenize(sentences, truncate=False).to(clip_model.text_projection.device)
    emb = clip_model.encode_text(tokens).float()
    return F.normalize(emb, dim=-1, eps=1e-8).cpu()


class OrdinalGeometryEmbeddings(nn.Module):
    """Grade embeddings t_0 … t_{K-1} learned directly in CLIP's joint space.

    forward() returns the L2-normalised prototypes [K, D]; the text encoder is not
    called. The anchors t̂⁰ are the normalised initial embeddings.
    """

    direct_prototypes = True

    def __init__(self, init_embeddings: torch.Tensor):
        super().__init__()
        init = F.normalize(init_embeddings.detach().float(), dim=-1, eps=1e-8)
        self.t = nn.Parameter(init.clone())
        self.register_buffer("anchors", init.clone())
        self.register_buffer("anchor_lambdas", torch.ones(init.shape[0]))

    def forward(self) -> torch.Tensor:
        return F.normalize(self.t, dim=-1, eps=1e-8)

    def set_anchor_lambdas(self, lambdas: torch.Tensor) -> None:
        self.anchor_lambdas.copy_(lambdas.to(self.anchor_lambdas))

    @torch.no_grad()
    def renormalize_(self) -> None:
        self.t.div_(self.t.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def diagnostics(self) -> dict:
        t_hat = self().float()
        return {
            "distance_matrix": cosine_distance_matrix(t_hat).tolist(),
            "step_angles_deg": step_angles_deg(t_hat),
            "anchor_drift": (1.0 - (t_hat * self.anchors.float()).sum(-1)).tolist(),
            "ladder_satisfied_frac": ladder_satisfied_frac(t_hat),
        }
