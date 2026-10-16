import glob
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("train", Path("scripts/train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

FILES = sorted(glob.glob("configs/direction1/*.yaml")) + [
    f"configs/baselines/{n}.yaml" for n in ("clipdr_matched", "l2rclip_matched", "clap_matched", "clap_native")
]


def test_all_configs_load_with_full_keys():
    assert len(FILES) == 14
    for f in FILES:
        c = train.load_config(f)
        assert c.model.prompt_learner.num_ranks == 5 and "DeepDRiD" in c.data.manifest, f
        assert c.train.epochs >= 100 and c.model.image_encoder.name == "RN50", f


def test_key_switches():
    full = train.load_config("configs/direction1/d1_full.yaml")
    assert full.model.prompt_learner.init_mode == "icdr_clinical" and full.loss.w_ladder == 1
    assert train.load_config("configs/direction1/d1_no_path.yaml").loss.w_path == 0
    native = train.load_config("configs/baselines/clap_native.yaml")
    assert native.train.optimizer == "sgd" and native.model.image_encoder.lora.enabled is False
    assert native.model.prompt_learner.anchor_weighting == "clap"


def test_matched_baselines_use_lora():
    for n in ("clipdr_matched", "l2rclip_matched", "clap_matched"):
        c = train.load_config(f"configs/baselines/{n}.yaml")
        assert c.model.image_encoder.lora.enabled is True and c.train.optimizer == "adamw", n
