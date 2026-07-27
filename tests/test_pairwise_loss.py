import torch
from src.models.losses import CrossModalPairwiseLoss

def test_pairwise_loss():
    num_ranks = 5
    criterion = CrossModalPairwiseLoss(num_ranks=num_ranks, tau=1.0)
    
    image_feats = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True) # D=3
    rank_prototypes = torch.tensor([
        [1.0, 0.0, 0.0], # rank 0
        [0.8, 0.2, 0.0], # rank 1
        [0.0, 1.0, 0.0], # rank 2
        [-0.8, 0.2, 0.0], # rank 3
        [-1.0, 0.0, 0.0], # rank 4
    ], requires_grad=True)
    
    # True rank is 0, image_feats exactly matches rank 0
    target = torch.tensor([0])
    
    loss = criterion(image_feats, rank_prototypes, target)
    
    loss.backward()
    
    assert image_feats.grad is not None
    assert rank_prototypes.grad is not None
    
    # If image features are exactly the correct rank prototype, and very far from others, loss should be small
    # Let's verify it increases if we move image_feats closer to a wrong prototype
    
    image_feats_wrong = torch.tensor([[-1.0, 0.0, 0.0]]) # Matches rank 4
    loss_wrong = criterion(image_feats_wrong, rank_prototypes, target)
    
    assert loss_wrong > loss
