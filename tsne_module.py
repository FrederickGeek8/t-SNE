import torch
import torch.nn as nn
import torch.nn.functional as F


class tSNE(nn.Module):
    def __init__(self, size, latent_dim=2):
        super(tSNE, self).__init__()
        self.embedding = nn.Embedding(size, latent_dim)
        self.degrees_of_freedom = 1.0

    def forward(self, p):
        all_y = self.embedding.weight
        dist = F.pdist(all_y).pow(2)
        dist = (1. + dist).pow(-1.0 * self.degrees_of_freedom)
        q = dist / dist.sum()

        # log_loss = pij * (torch.log(dist) - torch.log(qij))
        log_loss = p.dot(torch.log(p) - torch.log(q))

        return log_loss.sum()
