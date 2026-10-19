"""L2RCLIP (RankFormer + pairwise loss) is kept out of our model and lives in its own configs."""
from scripts.train import load_config

L2RCLIP_LEARNER = "rankformer_prompt_learner"


def test_our_model_has_no_l2rclip_parts():
    cfg = load_config("configs/deepdrid_cfp.yaml")
    assert cfg.model.prompt_learner.name != L2RCLIP_LEARNER
    assert cfg.loss.mu_pairwise == 0.0


def test_l2rclip_alone():
    cfg = load_config("configs/baselines/l2rclip_ours_impl.yaml")
    assert cfg.model.prompt_learner.name == L2RCLIP_LEARNER and cfg.loss.mu_pairwise > 0
    # none of our parts
    assert not cfg.model.sms.enabled and cfg.loss.lambda_rank == 0.0
    assert "DeepDRiD" in cfg.data.manifest


def test_ours_plus_l2rclip():
    ours = load_config("configs/deepdrid_cfp.yaml")
    cfg = load_config("configs/ours_plus_l2rclip.yaml")
    assert cfg.model.prompt_learner.name == L2RCLIP_LEARNER and cfg.loss.mu_pairwise > 0
    # everything else is our model
    assert cfg.model.sms.enabled == ours.model.sms.enabled
    assert cfg.loss.lambda_rank == ours.loss.lambda_rank
    assert cfg.model.image_encoder.lora.enabled == ours.model.image_encoder.lora.enabled
    assert cfg.data.manifest == ours.data.manifest
