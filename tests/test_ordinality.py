import torch
import torch.nn as nn
import clip
from src.models.prompt_learner.rankformer import RankFormerPromptLearner
from src.models.prompt_learner.ordinalclip_head import OrdinalCLIPHead

class DummyCLIP:
    def __init__(self):
        self.dtype = torch.float32
        class LNWeight:
            def __init__(self):
                self.shape = [512]
        self.ln_final = type('obj', (object,), {'weight': LNWeight()})()
        self.token_embedding = nn.Embedding(49408, 512)

def test_ordinality_shapes():
    # We just test the output shapes to ensure they are drop-in replacements
    clip_model = DummyCLIP()

    num_ranks = 5
    rf = RankFormerPromptLearner(clip_model, num_ranks=num_ranks)
    oc = OrdinalCLIPHead(clip_model, num_ranks=num_ranks)

    p1, t1 = rf()
    p2, t2 = oc()

    assert p1.shape[0] == num_ranks
    assert p1.shape == p2.shape
    assert t1.shape == t2.shape


def test_rankformer_shares_ordinalclip_rank_parametrisation():
    # With interpolated ranks, RankFormer starts from exactly the OrdinalCLIP
    # rank tokens, so removing it ablates only the attention block
    clip_model = DummyCLIP()
    rf = RankFormerPromptLearner(clip_model, num_ranks=5, rank_param="interpolated", block="paper", alpha=0.5)
    oc = OrdinalCLIPHead(clip_model, num_ranks=5)
    oc.base_embeddings.data.copy_(rf.base_embeddings.data)
    oc.ctx.data.copy_(rf.ctx.data)

    p_rf, _ = rf()
    p_oc, _ = oc()
    rank_pos = 1 + rf.num_context_tokens
    # FFN output starts at zero, so R' = (1 - alpha) R at initialisation
    assert torch.allclose(p_rf[:, rank_pos], 0.5 * p_oc[:, rank_pos], atol=1e-6)
    assert torch.allclose(p_rf[:, :rank_pos], p_oc[:, :rank_pos])

    for name in ["free", "interpolated"]:
        for block in ["paper", "simple"]:
            p, _ = RankFormerPromptLearner(clip_model, num_ranks=5, rank_param=name, block=block)()
            assert p.shape == p_oc.shape
