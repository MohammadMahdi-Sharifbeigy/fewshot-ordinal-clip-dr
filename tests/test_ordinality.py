import torch
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
        
    def token_embedding(self, x):
        return torch.randn(x.size(0), x.size(1), 512)

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
