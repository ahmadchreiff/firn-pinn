"""
Simple PyTorch MLP mapping (z, t) -> V. No PDE logic here.
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden=4, width=64):
        super().__init__()
        # To be implemented later
        pass

    def forward(self, x):
        pass
