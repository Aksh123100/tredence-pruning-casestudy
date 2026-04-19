
import torch
import torch.nn as nn
from prunable_linear import PrunableLinear

class PruningNet(nn.Module):
    """
    A 3-layer network using PrunableLinear layers.
    CIFAR-10 input: 3 x 32 x 32 = 3072 features
    Output: 10 classes
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten image: (batch, 3, 32, 32) → (batch, 3072)
        return self.net(x)


def sparsity_loss(model):
    """
    L1 penalty on all gate values across all PrunableLinear layers.
    Encourages gates to go to 0 (pruning).
    """
    total = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.sum()
    return total
