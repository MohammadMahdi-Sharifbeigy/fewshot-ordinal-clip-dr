"""Grade prototypes from clinical descriptions through a LoRA-adapted text encoder (Direction 2).

Each grade k has several descriptions s_{k,1..M}. The (LoRA-adapted) CLIP text
encoder embeds them, and the grade prototype is the normalised mean of its
normalised description embeddings (prompt ensembling as in CuPL):

    e_{k,m} = f_θ(s_{k,m}) / ‖·‖,    t_k = mean_m e_{k,m} / ‖·‖

Only the text-LoRA weights (src/models/text_lora.py) are learned; with LoRA
switched off the prototypes are the frozen zero-shot ensemble. The zero-shot
prototypes are kept as anchors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry_losses import cosine_distance_matrix, ladder_satisfied_frac, step_angles_deg
from .ordinal_geometry import GRADE_PROMPTS

GRADE_NAMES = [
    "no diabetic retinopathy",
    "mild non-proliferative diabetic retinopathy",
    "moderate non-proliferative diabetic retinopathy",
    "severe non-proliferative diabetic retinopathy",
    "proliferative diabetic retinopathy",
]

# Several clinical descriptions per grade, following the International Clinical
# Diabetic Retinopathy (ICDR) severity scale and its typical fundus findings.
ICDR_ENSEMBLE = [
    [
        "A fundus photograph showing no diabetic retinopathy, with normal retinal vasculature "
        "and no microaneurysms or hemorrhages.",
        "A healthy retina with a clear optic disc, normal blood vessels and no signs of diabetic retinopathy.",
        "A color fundus image without any microaneurysms, hemorrhages, exudates or new vessels.",
        "A normal fundus photograph of a diabetic patient with no retinal lesions.",
        "Fundus photography graded as ICDR level 0, no apparent retinopathy.",
    ],
    [
        "A fundus photograph showing mild non-proliferative diabetic retinopathy, characterized "
        "by the presence of microaneurysms only.",
        "A retina with a few small red dots from microaneurysms and no other diabetic lesions.",
        "A color fundus image with scattered microaneurysms only, the earliest sign of diabetic retinopathy.",
        "Mild diabetic retinopathy with isolated microaneurysms and no hemorrhages or exudates.",
        "Fundus photography graded as ICDR level 1, mild non-proliferative retinopathy.",
    ],
    [
        "A fundus photograph showing moderate non-proliferative diabetic retinopathy, with multiple "
        "microaneurysms, dot and blot hemorrhages, hard exudates, or cotton wool spots, less than severe.",
        "A retina with several dot and blot hemorrhages, hard exudates and cotton wool spots.",
        "A color fundus image with more than microaneurysms, such as intraretinal hemorrhages and yellow "
        "hard exudates, but no severe signs.",
        "Moderate diabetic retinopathy with hemorrhages and exudates in fewer than four quadrants.",
        "Fundus photography graded as ICDR level 2, moderate non-proliferative retinopathy.",
    ],
    [
        "A fundus photograph showing severe non-proliferative diabetic retinopathy, with extensive "
        "intraretinal hemorrhages in four quadrants, venous beading, or intraretinal microvascular "
        "abnormalities.",
        "A retina with many intraretinal hemorrhages in all four quadrants and definite venous beading.",
        "A color fundus image with intraretinal microvascular abnormalities and widespread hemorrhages "
        "but no new vessels.",
        "Severe diabetic retinopathy meeting the 4-2-1 rule, without neovascularization.",
        "Fundus photography graded as ICDR level 3, severe non-proliferative retinopathy.",
    ],
    [
        "A fundus photograph showing proliferative diabetic retinopathy, characterized by "
        "neovascularization on the optic disc or elsewhere, vitreous hemorrhage, or preretinal hemorrhage.",
        "A retina with abnormal new blood vessels growing on the optic disc or along the vascular arcades.",
        "A color fundus image with neovascularization, fibrovascular proliferation or vitreous hemorrhage.",
        "Proliferative diabetic retinopathy with preretinal hemorrhage and fragile new vessels.",
        "Fundus photography graded as ICDR level 4, proliferative retinopathy.",
    ],
]

PROMPT_SETS = {
    "icdr_ensemble": ICDR_ENSEMBLE,
    "icdr_clinical": [[s] for s in GRADE_PROMPTS["icdr_clinical"]],
    "plain": [[s] for s in GRADE_PROMPTS["plain"]],
    # CLIP-LoRA's template with the grade names
    "clip_template": [[f"a photo of a {name}."] for name in GRADE_NAMES],
}


class TextLoRAPrototypes(nn.Module):
    """Grade prototypes [K, D] from a description ensemble through CLIP's (LoRA) text encoder.

    The module holds no parameters: the text-LoRA weights live inside the text
    transformer. forward() also stores the description embeddings of the last
    call (`last_desc`, `desc_labels`) for the description-ordering loss.
    """

    direct_prototypes = True

    def __init__(self, clip_model, text_encoder, prompt_set: str = "icdr_ensemble", num_ranks: int = 5):
        super().__init__()
        if prompt_set not in PROMPT_SETS:
            raise ValueError(f"Unknown prompt_set: {prompt_set}")
        groups = PROMPT_SETS[prompt_set]
        if len(groups) != num_ranks:
            raise ValueError(f"prompt_set '{prompt_set}' defines {len(groups)} grades, got num_ranks={num_ranks}")
        import clip

        sentences = [s for g in groups for s in g]
        labels = [k for k, g in enumerate(groups) for _ in g]
        device = clip_model.token_embedding.weight.device
        # truncate=False: clip.tokenize raises if a description exceeds 77 tokens
        self.register_buffer("tokens", clip.tokenize(sentences, truncate=False).to(device))
        self.register_buffer("desc_labels", torch.tensor(labels, dtype=torch.long, device=device))
        self.num_ranks = num_ranks
        self.sentences = sentences
        # Frozen CLIP parts, kept out of this module's parameters on purpose
        object.__setattr__(self, "_token_embedding", clip_model.token_embedding)
        object.__setattr__(self, "_text_encoder", text_encoder)

        self.last_desc = None
        self._cache = None
        with torch.no_grad():
            anchors = self._prototypes(self._encode().float())
        self.register_buffer("anchors", anchors.clone())
        self.register_buffer("anchor_lambdas", torch.ones(num_ranks, device=device))

    def _encode(self) -> torch.Tensor:
        prompts = self._token_embedding(self.tokens)
        return F.normalize(self._text_encoder(prompts, self.tokens), dim=-1, eps=1e-8)

    def _prototypes(self, desc: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(self.desc_labels, self.num_ranks).to(desc.dtype)  # [N, K]
        mean = (onehot.T @ desc) / onehot.sum(0).unsqueeze(1)
        return F.normalize(mean, dim=-1, eps=1e-8)

    def forward(self) -> torch.Tensor:
        # In eval mode the prototypes do not change between batches: compute them once
        if not self.training and self._cache is not None:
            return self._cache
        desc = self._encode()
        self.last_desc = desc
        protos = self._prototypes(desc)
        if not self.training:
            self._cache = protos.detach()
        return protos

    def train(self, mode: bool = True):
        self._cache = None
        return super().train(mode)

    def set_anchor_lambdas(self, lambdas: torch.Tensor) -> None:
        self.anchor_lambdas.copy_(lambdas.to(self.anchor_lambdas))

    @torch.no_grad()
    def diagnostics(self) -> dict:
        was_training = self.training
        self.eval()
        t_hat = self().float()
        desc = self.last_desc.float() if self.last_desc is not None else self._encode().float()
        self.train(was_training)
        own = (desc * t_hat[self.desc_labels]).sum(-1)
        spread = [float((1.0 - own[self.desc_labels == k]).mean()) for k in range(self.num_ranks)]
        return {
            "distance_matrix": cosine_distance_matrix(t_hat).tolist(),
            "step_angles_deg": step_angles_deg(t_hat),
            "anchor_drift": (1.0 - (t_hat * self.anchors.float()).sum(-1)).tolist(),
            "ladder_satisfied_frac": ladder_satisfied_frac(t_hat),
            "description_spread": spread,
        }
