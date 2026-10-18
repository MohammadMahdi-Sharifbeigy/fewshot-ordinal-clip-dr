"""von Mises–Fisher (vMF) helpers on the unit sphere S^{D-1} (Direction 3).

vMF(z; μ, κ) = C_D(κ) exp(κ μᵀz),   C_D(κ) = κ^{ν} / ((2π)^{ν+1} I_ν(κ)),   ν = D/2 − 1.

CLIP embeddings have D = 1024 (RN50), so ν = 511. The modified Bessel function
I_ν is evaluated with the uniform asymptotic (Debye) expansion, which is
accurate for large ν, smooth and differentiable in κ:

    log I_ν(κ) ≈ η − ½ log(2π) − ¼ log(ν² + κ²) + log(1 + u₁(t)/ν),
    η = √(ν² + κ²) + ν log(κ / (ν + √(ν² + κ²))),   t = ν / √(ν² + κ²),
    u₁(t) = (3t − 5t³) / 24.
"""
import math

import torch


def log_bessel_iv(nu: float, kappa: torch.Tensor) -> torch.Tensor:
    """log I_ν(κ) for large order ν (uniform asymptotic expansion, first correction term)."""
    kappa = kappa.double()
    r = torch.sqrt(nu * nu + kappa * kappa)
    t = nu / r
    eta = r + nu * torch.log(kappa / (nu + r))
    u1 = (3.0 * t - 5.0 * t ** 3) / 24.0
    return eta - 0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(r) + torch.log1p(u1 / nu)


def log_vmf_normalizer(kappa: torch.Tensor, dim: int) -> torch.Tensor:
    """log C_D(κ) of a vMF on S^{D-1}."""
    nu = dim / 2.0 - 1.0
    k = kappa.double()
    out = nu * torch.log(k) - (nu + 1.0) * math.log(2.0 * math.pi) - log_bessel_iv(nu, k)
    return out.to(kappa.dtype)


def mean_resultant_length(kappa: torch.Tensor, dim: int) -> torch.Tensor:
    """A_D(κ) = I_{D/2}(κ) / I_{D/2−1}(κ) = E[μᵀx] for x ~ vMF(μ, κ)."""
    nu = dim / 2.0 - 1.0
    k = kappa.double()
    return torch.exp(log_bessel_iv(nu + 1.0, k) - log_bessel_iv(nu, k)).to(kappa.dtype)
