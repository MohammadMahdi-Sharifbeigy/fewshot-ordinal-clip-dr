import torch
from src.models.sms import SMS


def _epoch(sms, S, targets):
    sms.reset_stats()
    sms.update_stats(S, targets)
    sms.compute_smoothed_stats()


def test_sms_identity_before_stats_and_at_inference():
    sms = SMS(num_ranks=5)
    S = torch.randn(10, 5)
    targets = torch.arange(10) % 5

    # No statistics yet -> unchanged
    assert torch.equal(sms(S, targets), S)

    _epoch(sms, torch.randn(50, 5) * 3, torch.arange(50) % 5)

    # Inference (no labels) -> unchanged
    assert torch.equal(sms(S), S)


def test_sms_calibrates_towards_smoothed_stats():
    torch.manual_seed(0)
    sms = SMS(num_ranks=5)
    targets = torch.arange(500) % 5
    # Grade j has mean j and a grade-dependent spread
    S = targets.float().unsqueeze(1) + torch.randn(500, 5) * (1 + targets.float().unsqueeze(1))
    _epoch(sms, S, targets)

    out = sms(S, targets)
    assert out.shape == S.shape
    for j in range(5):
        cal = out[targets == j]
        # Calibrated mean of each grade equals its smoothed mean
        assert torch.allclose(cal.mean(0), sms.smoothed_mean[j], atol=1e-4)
    # Smoothing pulls the extreme grades' means towards their neighbours
    assert sms.smoothed_mean[0].mean() > sms.running_mean[0].mean()
    assert sms.smoothed_mean[4].mean() < sms.running_mean[4].mean()


def test_sms_momentum_and_gradients():
    sms = SMS(num_ranks=5, momentum=0.9)
    targets = torch.arange(20) % 5
    _epoch(sms, torch.zeros(20, 5) + torch.randn(20, 5), targets)
    first = sms.running_mean.clone()
    _epoch(sms, torch.full((20, 5), 10.0) + torch.randn(20, 5), targets)
    # Second epoch only moves the running mean 10% of the way
    expected = 0.1 * (sms.running_mean.new_full((5, 5), 10.0)) + 0.9 * first
    assert torch.allclose(sms.running_mean, expected, atol=1.0)

    S = torch.randn(20, 5, requires_grad=True)
    sms(S, targets).sum().backward()
    assert S.grad is not None and torch.all(S.grad > 0)
