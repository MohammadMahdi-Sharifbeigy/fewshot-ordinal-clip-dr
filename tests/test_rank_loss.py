import torch
from src.models.losses import RankingAwareLoss

def test_ranking_aware_loss():
    num_ranks = 5
    criterion = RankingAwareLoss(num_ranks=num_ranks, tau=1.0)
    
    # Batch size 1, true rank 2
    # perfectly ordered similarities
    sim_ordered = torch.tensor([[0.0, 1.0, 2.0, 1.0, 0.0]])
    target = torch.tensor([2])
    
    loss_ordered = criterion(sim_ordered, target)
    
    # reversed (incorrect) similarities
    sim_reversed = torch.tensor([[2.0, 1.0, 0.0, 1.0, 2.0]])
    
    loss_reversed = criterion(sim_reversed, target)
    
    assert loss_ordered < loss_reversed
    
    # Test gradients flow
    sim = torch.tensor([[0.5, 1.0, 2.0, 1.5, 0.5]], requires_grad=True)
    loss = criterion(sim, target)
    loss.backward()
    assert sim.grad is not None
