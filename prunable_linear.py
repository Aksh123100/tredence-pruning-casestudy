
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    A linear layer where every weight has a learnable gate.
    gate ≈ 0  →  weight is pruned (silenced)
    gate ≈ 1  →  weight is fully active
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)
