import torch
import torch.nn as nn
import torch.nn.functional as F

class RankingAwareLoss(nn.Module):
    """CLIP-DR Ranking Aware Loss"""
    def __init__(self, num_ranks=5, tau=1.0):
        super().__init__()
        self.num_ranks = num_ranks
        self.tau = tau
        
    def forward(self, sim, targets):
        """
        sim: [B, num_ranks] similarity matrix
        targets: [B] ground truth ranks
        """
        B = sim.size(0)
        total_loss = 0.0
        
        for i in range(B):
            j = targets[i].item()
            
            # rightward: sum_{j'=j..K-2} -log softmax(sim[j'], sim[j'+1])
            right_loss = 0.0
            for jp in range(j, self.num_ranks - 1):
                s1 = sim[i, jp] / self.tau
                s2 = sim[i, jp + 1] / self.tau
                right_loss += -torch.log(torch.exp(s1) / (torch.exp(s1) + torch.exp(s2) + 1e-8))
                
            # leftward: sum_{j'=2..j} -log softmax(sim[j'], sim[j'-1])
            left_loss = 0.0
            for jp in range(j, 0, -1):
                s1 = sim[i, jp] / self.tau
                s2 = sim[i, jp - 1] / self.tau
                left_loss += -torch.log(torch.exp(s1) / (torch.exp(s1) + torch.exp(s2) + 1e-8))
                
            total_loss += (right_loss + left_loss)
            
        return total_loss / max(B, 1)

class CrossModalPairwiseLoss(nn.Module):
    """L2RCLIP Cross-Modal Pairwise Loss"""
    def __init__(self, num_ranks=5, tau=1.0):
        super().__init__()
        self.num_ranks = num_ranks
        self.tau = tau
        
    def forward(self, image_feats, rank_prototypes, targets):
        """
        image_feats: [B, d]
        rank_prototypes: [num_ranks, d]
        targets: [B]
        """
        B = image_feats.size(0)
        sim = image_feats @ rank_prototypes.T  # [B, num_ranks]
        probs = F.softmax(sim / self.tau, dim=1) # [B, num_ranks]
        
        loss = 0.0
        for i in range(B):
            img_feat = image_feats[i]
            y_i = targets[i].item()
            
            for j in range(self.num_ranks):
                if j == y_i:
                    continue
                    
                # Weight by label space distance
                w_ij = abs(y_i - j)
                
                # Cross-modal features: hard mean = rank_prototypes[j], soft mean = weighted avg of images
                hard_mean = rank_prototypes[j]
                
                # Diversity estimation using meanNN (simplified for single batch)
                # D_pce ≈ - ||img_feat - hard_mean||^2
                # Note: The exact formulation in L2RCLIP uses entire dataset statistics.
                # For batch processing, we approximate CPCE as bringing image closer to correct rank
                # and pushing away from incorrect ranks proportional to rank distance.
                
                # Simple approximation: log softmax over distance
                # The paper uses: L_CPCE(x_i, t_j, y_i)
                sim_i_yi = sim[i, y_i] / self.tau
                sim_i_j = sim[i, j] / self.tau
                
                # Pairwise CE: -log( exp(s_yi) / (exp(s_yi) + exp(s_j)) )
                pce = -torch.log(torch.exp(sim_i_yi) / (torch.exp(sim_i_yi) + torch.exp(sim_i_j) + 1e-8))
                loss += w_ij * pce
                
        return loss / max(B, 1)
