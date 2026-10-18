import glob
import math

import mpmath as mp
import pytest
import torch
import torch.nn.functional as F

from src.models.prompt_learner.vmf_grades import VMFGradeDistributions
from src.models.vmf import log_bessel_iv, log_vmf_normalizer, mean_resultant_length


@pytest.mark.parametrize("kappa", [1.0, 10.0, 50.0, 100.0, 300.0, 1000.0])
def test_log_bessel_matches_mpmath(kappa):
    mp.mp.dps = 50
    nu = 511
    ref = float(mp.log(mp.besseli(nu, kappa)))
    ours = float(log_bessel_iv(nu, torch.tensor(kappa)))
    assert abs(ours - ref) <= 1e-8 * abs(ref)
    a_ref = float(mp.besseli(nu + 1, kappa) / mp.besseli(nu, kappa))
    assert abs(float(mean_resultant_length(torch.tensor(kappa, dtype=torch.float64), 1024)) - a_ref) < 1e-8


def test_normalizer_is_differentiable():
    k = torch.tensor([20.0, 100.0, 400.0], requires_grad=True)
    log_vmf_normalizer(k, 1024).sum().backward()
    assert torch.isfinite(k.grad).all() and (k.grad < 0).all()  # C_D decreases with κ for κ ≪ D


def _module(**kw):
    torch.manual_seed(0)
    return VMFGradeDistributions(F.normalize(torch.randn(5, 1024), dim=-1), **kw)


def test_equal_kappa_reduces_to_clip_softmax():
    m = _module(kappa_init=100.0, kappa_mode="per_grade")
    z = F.normalize(torch.randn(8, 1024), dim=-1)
    clip_probs = F.softmax(100.0 * z @ m().T, dim=-1)
    assert torch.allclose(F.softmax(m.score(z, torch.tensor(100.0)), dim=-1), clip_probs, atol=1e-5)
    assert float(m.kappa_prior().detach()) == 0.0


def test_expected_scoring_at_kappa0_is_clip_logits():
    m = _module(kappa_init=100.0, scoring="expected")
    z = F.normalize(torch.randn(4, 1024), dim=-1)
    assert torch.allclose(m.score(z, torch.tensor(100.0)), 100.0 * z @ m().T, atol=1e-4)


def test_broad_grade_takes_probability_from_ambiguous_images():
    m = _module(kappa_init=100.0)
    z = F.normalize(torch.randn(16, 1024), dim=-1)  # far from every mean
    before = F.softmax(m.score(z, torch.tensor(100.0)), dim=-1)[:, 2].mean()
    with torch.no_grad():
        m.log_kappa[2] = math.log(60.0)  # grade 2 broader
    after = F.softmax(m.score(z, torch.tensor(100.0)), dim=-1)[:, 2].mean()
    assert after > before


@pytest.mark.parametrize("mode,n_trainable_kappa", [("per_grade", 5), ("shared", 1), ("fixed", 0)])
def test_kappa_modes(mode, n_trainable_kappa):
    m = _module(kappa_mode=mode)
    kappa_params = [p for n, p in m.named_parameters() if "kappa" in n]
    assert sum(p.numel() for p in kappa_params) == n_trainable_kappa
    assert m.kappa().shape == (5,)
    assert [id(p) for p in m.no_decay_parameters()] == [id(m.t)]
    assert {id(p) for p in m.kappa_parameters()} == {id(p) for p in kappa_params}


@pytest.mark.parametrize("config", sorted(glob.glob("configs/direction3/*.yaml")))
def test_direction3_configs_train_and_evaluate(config):
    from scripts.train import load_config
    from src.models.runner import FewShotOrdinalRunner

    cfg = load_config(config, ["data.k_shot=1", "data.seed=1"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = FewShotOrdinalRunner(cfg).to(device)  # Lightning moves the model the same way
    runner.log = lambda *a, **k: None  # no trainer attached
    images, targets = torch.randn(5, 3, 224, 224, device=device), torch.arange(5, device=device)

    runner.train()
    loss = runner.training_step((images, targets, None, None), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert runner.prompt_learner.t.grad is not None
    if cfg.model.prompt_learner.vmf.get("kappa_mode", "per_grade") != "fixed":
        assert runner.prompt_learner.log_kappa.grad is not None

    runner.eval()
    runner._shared_eval_step((images, targets, None, None))
    runner._shared_eval_epoch_end("test")
    m = runner.latest_metrics
    assert {"qwk", "ece", "nll", "brier"} <= set(m) and math.isfinite(m["nll"]) and 0 <= m["brier"] <= 2


def test_concentrations_get_their_own_learning_rate():
    from scripts.train import load_config
    from src.models.runner import FewShotOrdinalRunner

    cfg = load_config("configs/direction3/d3_full.yaml", ["data.k_shot=1", "data.seed=1"])
    runner = FewShotOrdinalRunner(cfg)
    opt = runner.configure_optimizers()["optimizer"]
    groups = {id(p): g for g in opt.param_groups for p in g["params"]}
    kappa_group = groups[id(runner.prompt_learner.log_kappa)]
    # initial_lr: the base rate before the warm-up schedule scales it
    assert kappa_group["initial_lr"] == cfg.train.kappa_lr and kappa_group["weight_decay"] == 0.0
    assert groups[id(runner.prompt_learner.t)]["weight_decay"] == 0.0
