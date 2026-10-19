"""Baseline configs reproduce the published methods (checked against their code)."""
import torch

from scripts.train import load_config
from src.models.prompt_learner.ordinalclip_head import interpolation_weights


def test_clipdr_matched_is_clipdr():
    # CLIP-DR's PlainPromptLearner: 10 context tokens, one free token per grade, full stop;
    # loss CE + KL + ranking-aware; SMS off as in its released training code
    c = load_config("configs/baselines/clipdr_matched.yaml")
    pl = c.model.prompt_learner
    assert pl.name == "ordinalclip_rank_prompt_learner" and pl.interpolation == "none" and pl.num_base == 5
    assert pl.num_tokens_for_context == 10 and pl.full_stop and pl.init_context is None
    assert c.loss.w_kl == 1.0 and c.loss.lambda_rank == 1.0 and c.loss.mu_pairwise == 0.0
    assert not c.model.sms.enabled and c.model.image_encoder.lora.enabled
    assert "DeepDRiD" in c.data.manifest


def test_clipdr_official_recipe():
    c = load_config("configs/baselines/clipdr_official.yaml")
    assert c.model.image_encoder.full_finetune and not c.model.image_encoder.lora.enabled
    assert c.train.optimizer == "radam" and c.train.weight_decay == 0.0 and c.train.epochs == 31
    assert c.data.augment == "ordinalclip" and c.data.batch_query == 64
    assert c.model.prompt_learner.interpolation == "none"


def test_plain_interpolation_is_identity():
    assert torch.equal(interpolation_weights(5, 5, interpolation="none"), torch.eye(5))


def test_every_baseline_uses_deepdrid():
    import glob
    for path in glob.glob("configs/baselines/*.yaml"):
        assert "DeepDRiD" in load_config(path).data.manifest, path
