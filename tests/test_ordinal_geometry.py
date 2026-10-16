import pytest
import torch

from src.models.prompt_learner.ordinal_geometry import (
    GRADE_PROMPTS,
    OrdinalGeometryEmbeddings,
    build_init_embeddings,
)


class FakeClip(torch.nn.Module):
    """Stands in for a CLIP model: only text_projection (for D) and encode_text are used."""

    def __init__(self, D=16):
        super().__init__()
        self.text_projection = torch.nn.Parameter(torch.zeros(8, D))

    def encode_text(self, tokens):  # deterministic per sentence
        g = torch.Generator().manual_seed(int(tokens.sum()))
        return torch.randn(tokens.shape[0], self.text_projection.shape[1], generator=g) * 3


def test_prompts_verbatim():
    assert GRADE_PROMPTS["plain"][0] == "A fundus photograph with no diabetic retinopathy."
    assert GRADE_PROMPTS["icdr_clinical"][1].endswith("characterized by the presence of microaneurysms only.")
    assert all(len(v) == 5 for v in GRADE_PROMPTS.values())


@pytest.mark.parametrize("mode", ["icdr_clinical", "plain", "random_ortho"])
def test_init_is_unit_norm(mode):
    t = build_init_embeddings(FakeClip(), mode, 5, seed=1)
    assert t.shape == (5, 16) and torch.allclose(t.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_random_ortho_orthonormal_and_seeded():
    a = build_init_embeddings(FakeClip(), "random_ortho", 5, seed=3)
    assert torch.allclose(a @ a.T, torch.eye(5), atol=1e-5)
    assert torch.equal(a, build_init_embeddings(FakeClip(), "random_ortho", 5, seed=3))


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_init_embeddings(FakeClip(), "nope", 5, seed=1)


def test_text_mode_needs_five_ranks():
    with pytest.raises(ValueError):
        build_init_embeddings(FakeClip(), "plain", 4, seed=1)


def test_parameter_starts_on_sphere():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16) * 4)
    assert torch.allclose(m.t.detach().norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(m.t.detach(), m.anchors)


def test_forward_is_unit_norm_and_differentiable():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16) * 4)
    out = m()
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
    out[:, 0].sum().backward()
    g = m.t.grad
    assert g is not None and g.abs().sum() > 0
    # gradient of a normalised output is orthogonal to t (angular only)
    assert torch.allclose((g * m.t.detach()).sum(-1), torch.zeros(5), atol=1e-4)


def test_anchors_normalised_and_frozen():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16) * 4)
    assert torch.allclose(m.anchors.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert "anchors" not in dict(m.named_parameters())


def test_renormalize_and_diagnostics():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16))
    with torch.no_grad():
        m.t.mul_(7)
    m.renormalize_()
    assert torch.allclose(m.t.norm(dim=-1), torch.ones(5), atol=1e-5)
    d = m.diagnostics()
    assert set(d) == {"distance_matrix", "step_angles_deg", "anchor_drift", "ladder_satisfied_frac"}


def test_set_anchor_lambdas():
    m = OrdinalGeometryEmbeddings(torch.randn(5, 16))
    m.set_anchor_lambdas(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert m.anchor_lambdas.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
