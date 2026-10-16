import torch
from omegaconf import OmegaConf

from src.models.prompt_learner.ordinal_geometry import OrdinalGeometryEmbeddings
from src.models.runner import clap_lambdas, population_grade_counts, resolve_loss_weights, unpack_batch


def test_resolve_loss_weights_backcompat():
    w = resolve_loss_weights(OmegaConf.create({"lambda_rank": 1.0, "mu_pairwise": 0.5, "tau": 1.0}))
    assert w == {"w_ce": 1.0, "w_clipdr": 1.0, "w_l2r": 0.5, "w_sms": 1.0,
                 "w_ladder": 0.0, "w_path": 0.0, "w_anchor": 0.0}


def test_resolve_loss_weights_new_keys_win():
    w = resolve_loss_weights(OmegaConf.create({"lambda_rank": 1.0, "w_clipdr": 0.0, "w_ladder": 2.0, "w_sms": 0}))
    assert w["w_clipdr"] == 0.0 and w["w_ladder"] == 2.0 and w["w_sms"] == 0.0


def test_population_grade_counts(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,modality,grade,lesion,lr,split\na,CFP,0,,0,train\nb,CFP,0,,0,train\n"
                 "c,CFP,4,,0,train\nd,CFP,1,,0,test\n")
    assert population_grade_counts(str(p), 5) == [2, 0, 0, 0, 1]


def test_renorm_after_optimizer_step():
    m = OrdinalGeometryEmbeddings(torch.nn.functional.normalize(torch.randn(5, 16), dim=-1))
    opt = torch.optim.AdamW([{"params": [m.t], "weight_decay": 0.0}], lr=0.5)
    (m() @ torch.randn(16)).sum().backward()
    opt.step()
    m.renormalize_()
    assert m.t.requires_grad and torch.allclose(m.t.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_clap_lambdas_mean_true_class_probability():
    probs = torch.tensor([[0.8, 0.2, 0.0], [0.6, 0.4, 0.0], [0.1, 0.3, 0.6]])
    targets = torch.tensor([0, 0, 2])
    lam = clap_lambdas(probs, targets, 3)
    # grade 0: mean(0.8, 0.6); grade 1: no images -> 1; grade 2: 0.6
    assert torch.allclose(lam, torch.tensor([0.7, 1.0, 0.6]))


def test_unpack_batch_tuple_and_dict():
    img, tgt = torch.zeros(2, 3), torch.tensor([1, 2])
    a = unpack_batch((img, tgt, "x", "y"))
    b = unpack_batch({"image": img, "target": tgt})
    assert a[0] is img and a[1] is tgt and b[0] is img and b[1] is tgt
