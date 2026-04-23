# Self-Pruning Neural Network

Case study submission for **Tredence Analytics — AI Engineering Internship**.

A feed-forward network that **learns to prune its own weights during training** via learnable gate parameters and L1 sparsity regularization on CIFAR-10.

## How it works

Each weight in every linear layer has a paired learnable `gate_score`.
The gate is computed as `sigmoid(gate_score)` and multiplies the weight element-wise.
An L1 penalty on all gate values is added to the cross-entropy loss:

```
Total Loss = CrossEntropyLoss + λ * Σ sigmoid(gate_scores)
```

This drives unimportant gates toward zero during training, effectively removing those weights from the network without any post-training step.

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Aksh123100/tredence-pruning-casestudy.git
cd tredence-pruning-casestudy
```

### 2. Install dependencies
```bash
pip install torch torchvision matplotlib
```

### 3. Run training
```bash
python train.py
```

CIFAR-10 will download automatically on first run. Training sweeps over λ = [0.001, 0.01, 0.1] and prints accuracy + sparsity for each.

### 4. View results
- Printed to terminal at the end of training
- Gate distribution plots saved to `outputs/gates_lambda_<λ>.png`
- Full analysis in [report.md](report.md)

## Files

| File | Description |
|------|-------------|
| `prunable_linear.py` | Custom `PrunableLinear` layer |
| `model.py` | Network architecture + `sparsity_loss` |
| `train.py` | Training, evaluation, and visualization |
| `report.md` | Analysis and results |
| `outputs/` | Gate distribution plots |
