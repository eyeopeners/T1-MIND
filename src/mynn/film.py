
import torch, torch.nn as nn
class FiLM(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_dim)
        self.beta  = nn.Linear(cond_dim, in_dim)
    def forward(self, x, c):
        g = self.gamma(c); b = self.beta(c)
        return x * (1 + g) + b
