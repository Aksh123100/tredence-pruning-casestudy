
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
from prunable_linear import PrunableLinear
from model import PruningNet, sparsity_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('outputs', exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_set  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


def train(lam, warmup_epochs=5, prune_epochs=20):
    """
    Two-phase training:
      Phase 1 (warmup): train with lambda=0 so the network learns which weights
        matter before sparsity kicks in.
      Phase 2 (pruning): add sparsity penalty; CE gradient now protects important
        weights while unimportant gates drift to 0.
    """
    model = PruningNet().to(device)
    # gate_scores get a higher lr so they move far enough for sigmoid < 0.01
    gate_params  = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3},
        {'params': gate_params,  'lr': 1e-1},
    ])
    ce_loss = nn.CrossEntropyLoss()
    total_epochs = warmup_epochs + prune_epochs

    for epoch in range(total_epochs):
        # Phase 1: no sparsity; Phase 2: full lambda
        effective_lam = 0.0 if epoch < warmup_epochs else lam
        phase = "warmup" if epoch < warmup_epochs else "pruning"

        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = ce_loss(pred, y) + effective_lam * sparsity_loss(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [{phase}] Epoch {epoch+1}/{total_epochs} — loss: {total_loss/len(train_loader):.4f}")

    return model


def evaluate(model, threshold=0.01):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    accuracy = correct / total

    pruned, all_w = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate_scores)
            pruned += (g < threshold).sum().item()
            all_w  += g.numel()
    sparsity = pruned / all_w
    return accuracy, sparsity


def plot_gates(model, lam):
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate_scores).detach().cpu().flatten().tolist()
            all_gates.extend(g)

    plt.figure(figsize=(8, 4))
    plt.hist(all_gates, bins=50, color='steelblue', edgecolor='white')
    plt.title(f'Gate value distribution — lambda={lam}')
    plt.xlabel('Gate value (0 = pruned, 1 = active)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'outputs/gates_lambda_{lam}.png')
    plt.close()
    print(f"Plot saved to outputs/gates_lambda_{lam}.png")


if __name__ == '__main__':
    results = []
    for lam in [0.001, 0.01, 0.1]:
        print(f"\nTraining with lambda = {lam}")
        model = train(lam, warmup_epochs=5, prune_epochs=20)
        acc, spar = evaluate(model)
        plot_gates(model, lam)
        results.append((lam, acc, spar))
        print(f"lambda={lam} | accuracy={acc:.2%} | sparsity={spar:.2%}")

    print("\n--- Final Results ---")
    print(f"{'Lambda':<10} {'Accuracy':<12} {'Sparsity'}")
    for lam, acc, spar in results:
        print(f"{lam:<10} {acc:<12.2%} {spar:.2%}")
