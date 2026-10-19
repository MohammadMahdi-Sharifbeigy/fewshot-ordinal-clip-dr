"""The shared result cache: runs are keyed by what can change the result, nothing else."""
import json

from src.utils.experiment import _register, config_key, find_runs, load_run_config, run_dir


def test_key_ignores_data_location_k_and_seed():
    a = load_run_config("configs/deepdrid_cfp.yaml", 4, 1)
    b = load_run_config("configs/deepdrid_cfp.yaml", 8, 3, data_dir="/elsewhere/data/DeepDRiD",
                        overrides=["data.num_workers=4"])
    assert config_key(a) == config_key(b)
    assert run_dir("R", a).name == "CFP_4shot_s1" and run_dir("R", b).name == "CFP_8shot_s3"


def test_key_changes_with_anything_that_matters():
    base = load_run_config("configs/deepdrid_cfp.yaml", 4, 1)
    for ov in (["train.epochs=50"], ["model.sms.enabled=false"], ["loss.lambda_rank=0.0"], ["data.augment=ordinalclip"]):
        assert config_key(load_run_config("configs/deepdrid_cfp.yaml", 4, 1, overrides=ov)) != config_key(base), ov
    assert config_key(load_run_config("configs/baselines/clipdr_matched.yaml", 4, 1)) != config_key(base)


def test_same_config_through_overrides_or_file_shares_a_key():
    # The main notebook builds variants from overrides; direction notebooks use config files
    via_file = load_run_config("configs/ours_plus_l2rclip.yaml", 4, 1)
    via_overrides = load_run_config("configs/deepdrid_cfp.yaml", 4, 1, overrides=[
        "model.prompt_learner.name=rankformer_prompt_learner", "loss.mu_pairwise=1.0"])
    assert config_key(via_file) == config_key(via_overrides)


def test_find_runs_by_name(tmp_path):
    cfg = load_run_config("configs/deepdrid_cfp.yaml", 8, 2)
    _register(tmp_path, cfg, "Ours (Full)")
    _register(tmp_path, cfg, "Ours (master)")
    d = run_dir(tmp_path, cfg)
    d.mkdir(parents=True)
    json.dump({"qwk": 0.5, "mae": 0.7}, open(d / "test_stats.json", "w"))
    for name in ("Ours (Full)", "Ours (master)"):
        df = find_runs(tmp_path, [name])
        assert len(df) == 1 and df.iloc[0]["k"] == 8 and df.iloc[0]["Seed"] == 2 and df.iloc[0]["qwk"] == 0.5
    assert find_runs(tmp_path, ["Direction 1"]).empty
