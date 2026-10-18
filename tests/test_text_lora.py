import glob

import clip
import pytest
import torch
import torch.nn.functional as F

from src.models.clip_backbone import TextEncoder
from src.models.losses import RankingAwareLoss
from src.models.prompt_learner.text_lora_prototypes import PROMPT_SETS, TextLoRAPrototypes
from src.models.text_lora import LoRAMultiheadAttention, attach_text_lora, text_lora_parameters
from src.models.text_losses import description_order_loss


@pytest.fixture(scope="module")
def clip_cpu():
    model, _ = clip.load("RN50", device="cpu")
    return model.float().eval()


def _fresh(clip_cpu):
    model, _ = clip.load("RN50", device="cpu")
    return model.float().eval()


def test_prompt_sets_fit_clip_context():
    for name, groups in PROMPT_SETS.items():
        assert len(groups) == 5, name
        clip.tokenize([s for g in groups for s in g], truncate=False)  # raises if > 77 tokens


def test_lora_attention_with_zero_b_matches_clip(clip_cpu):
    tokens = clip.tokenize(["a fundus photograph with mild diabetic retinopathy.", "a healthy retina."])
    with torch.no_grad():
        ref = clip_cpu.encode_text(tokens)
    model = _fresh(clip_cpu)
    attach_text_lora(model, r=2, alpha=1.0, dropout=0.25, targets=["q", "k", "v", "o"])
    assert all(isinstance(b.attn, LoRAMultiheadAttention) for b in model.transformer.resblocks)
    model.eval()
    with torch.no_grad():
        out = model.encode_text(tokens)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()


def test_only_lora_weights_get_gradients(clip_cpu):
    model = _fresh(clip_cpu)
    n = attach_text_lora(model, r=2)
    assert n == 12 * 3 * 2 * 2 * 512  # layers * targets * (A, B) * r * width
    for p in model.parameters():
        p.requires_grad = False
    for p in text_lora_parameters(model):
        p.requires_grad = True
        if p.shape[0] == 512:  # make B non-zero so A also receives gradient
            torch.nn.init.normal_(p, std=0.02)
    model.train()
    protos = TextLoRAPrototypes(model, TextEncoder(model), "icdr_ensemble")
    protos.train()
    protos().sum().backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad and all("tlora_" in n for n in with_grad)
    assert len(with_grad) == len(text_lora_parameters(model))


def test_prototypes_are_normalised_description_means(clip_cpu):
    model = _fresh(clip_cpu)
    attach_text_lora(model, r=2)
    protos = TextLoRAPrototypes(model, TextEncoder(model), "icdr_ensemble").eval()
    with torch.no_grad():
        t = protos()
        desc = F.normalize(model.encode_text(protos.tokens).float(), dim=-1)
    labels = protos.desc_labels
    expected = torch.stack([F.normalize(desc[labels == k].mean(0), dim=0) for k in range(5)])
    assert torch.allclose(t, expected, atol=1e-4)
    # LoRA starts at zero, so the anchors are the zero-shot ensemble
    assert torch.allclose(protos.anchors, expected, atol=1e-4)
    assert protos() is protos._cache  # eval mode: computed once
    protos.train()
    assert protos._cache is None
    d = protos.diagnostics()
    assert set(d) >= {"distance_matrix", "ladder_satisfied_frac", "description_spread"}


def test_description_order_loss_prefers_ordered_clouds():
    torch.manual_seed(0)
    protos = F.normalize(torch.randn(5, 16), dim=-1)
    labels = torch.arange(5).repeat_interleave(3)
    near = F.normalize(protos[labels] + 0.05 * torch.randn(15, 16), dim=-1)
    shuffled = F.normalize(protos[labels.flip(0)] + 0.05 * torch.randn(15, 16), dim=-1)
    rank = RankingAwareLoss(num_ranks=5)
    assert description_order_loss(near, labels, protos, 100.0, rank) < description_order_loss(shuffled, labels, protos, 100.0, rank)


@pytest.mark.parametrize("config", sorted(glob.glob("configs/direction2/*.yaml"))
                         + ["configs/baselines/cliplora.yaml", "configs/baselines/cliplora_ordinal.yaml"])
def test_direction2_configs_train_one_step(config):
    from scripts.train import load_config
    from src.models.runner import FewShotOrdinalRunner

    cfg = load_config(config, ["data.k_shot=1", "data.seed=1"])
    runner = FewShotOrdinalRunner(cfg)
    lora_on = cfg.model.prompt_learner.text_lora.get("enabled", True)
    trainable = {n for n, p in runner.named_parameters() if p.requires_grad}
    text_trainable = {n for n in trainable if "clip_model.transformer" in n}
    assert text_trainable == ({n for n in trainable if "tlora_" in n} if lora_on else set())
    assert bool(text_trainable) == lora_on

    runner.train()
    device = next(runner.parameters()).device
    images, targets = torch.randn(5, 3, 224, 224, device=device), torch.arange(5, device=device)
    runner.log = lambda *a, **k: None  # no trainer attached
    loss = runner.training_step((images, targets, None, None), 0)
    assert torch.isfinite(loss)
    loss.backward()
    if lora_on:
        assert any(p.grad is not None for p in text_lora_parameters(runner.clip_model))
