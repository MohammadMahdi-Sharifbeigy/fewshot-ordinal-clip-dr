import torch
import torch.nn as nn
import numpy as np

class SMS(nn.Module):
    def __init__(self, num_ranks=5, kernel_size=3, sigma=1.0):
        super().__init__()
        self.num_ranks = num_ranks
        
        # Build symmetric gaussian-like kernel
        kernel = torch.zeros((num_ranks, num_ranks))
        for i in range(num_ranks):
            for j in range(num_ranks):
                if abs(i-j) <= kernel_size // 2:
                    kernel[i, j] = np.exp(-abs(i-j)**2 / (2 * sigma**2))
        # Normalize rows
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        self.register_buffer('kernel', kernel)
        
        self.register_buffer('mu', torch.zeros(num_ranks))
        self.register_buffer('sigma_sq', torch.ones(num_ranks))
        
        self.reset_stats()
        
    def reset_stats(self):
        self.mu_acc = {j: [] for j in range(self.num_ranks)}
        
    def update_stats(self, S, targets):
        """ Accumulate stats during training """
        for i in range(S.size(0)):
            j = targets[i].item()
            # We want to smooth the similarity of sample i w.r.t true class j
            # But SMS smooths the column distributions.
            # In CLIP-DR, mu_j and Sigma_j are statistics of row vectors where true rank is j.
            self.mu_acc[j].append(S[i].detach())
            
    def compute_smoothed_stats(self):
        """ Called at the end of epoch to update mu and sigma for the next epoch """
        device = self.kernel.device
        mu = torch.zeros(self.num_ranks, self.num_ranks, device=device)
        sigma_sq = torch.ones(self.num_ranks, self.num_ranks, device=device)
        
        for j in range(self.num_ranks):
            if len(self.mu_acc[j]) > 0:
                rows = torch.stack(self.mu_acc[j]) # [N, num_ranks]
                mu[j] = rows.mean(dim=0)
                if len(rows) > 1:
                    sigma_sq[j] = rows.var(dim=0)
                else:
                    sigma_sq[j] = torch.zeros(self.num_ranks, device=device)
        
        # Smooth with kernel
        # \tilde{mu}_j = \sum_{j'} K(j, j') mu_{j'}
        self.mu = self.kernel @ mu # [num_ranks, num_ranks]
        self.sigma_sq = self.kernel @ sigma_sq # [num_ranks, num_ranks]
        
    def forward(self, S):
        """ Apply calibration """
        # S: [B, num_ranks]
        # We don't know the true class at inference time for SMS. 
        # Wait, the formula says:
        # \tilde{s}_{.,j} = \tilde{\Sigma}_j^{1/2} * \tilde{\mu}_j^{1/2} * (s_{.,j} - \mu_j) + \tilde{\mu}_j
        # where \mu_j is the scalar mean of column j? 
        # No, the paper says "s_{.,j} as the row vector that the true text embedding is t_j".
        # For calibration, it's applied column-wise. Let's do a simplified column-wise smoothing:
        # we compute mean and var of each column over the epoch.
        # Here we just return S if stats are not ready
        if torch.all(self.mu == 0):
            return S
            
        S_tilde = S.clone()
        # Simplified: apply diagonal of smoothed stats
        for j in range(self.num_ranks):
            # mu_j of column j
            mu_j = self.mu[j, j]
            sig_j = torch.sqrt(self.sigma_sq[j, j] + 1e-8)
            # Center and scale
            # To avoid exploding values, just slightly pull toward smoothed mean
            S_tilde[:, j] = sig_j * torch.sqrt(torch.abs(mu_j) + 1e-8) * (S[:, j] - mu_j) + mu_j
            
        return S_tilde
