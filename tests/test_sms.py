import torch
from src.models.sms import SMS

def test_sms():
    sms = SMS(num_ranks=5, kernel_size=3)
    
    # Uniform similarities
    S = torch.ones((10, 5))
    targets = torch.randint(0, 5, (10,))
    
    # Update stats
    sms.update_stats(S, targets)
    sms.compute_smoothed_stats()
    
    S_tilde = sms(S)
    
    assert S_tilde.shape == S.shape
    
    # The output on uniform inputs should still be mostly uniform/identity (centered around mu=1)
    # Check that shape is preserved and gradients can theoretically flow
    S.requires_grad = True
    S_tilde = sms(S)
    S_tilde.sum().backward()
    assert S.grad is not None
