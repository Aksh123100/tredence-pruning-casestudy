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

## Files

| File | Description |
|------|-------------|
| `prunable_linear.py` | Custom `PrunableLinear` layer |
| `model.py` | Network architecture + `sparsity_loss` |
| `train.py` | Training, evaluation, and visualization |
| `report.md` | Analysis and results |

## Run

```bash
pip install torch torchvision matplotlib
python train.py
```

Results and gate distribution plots are saved to `outputs/`.

## Results summary

See [report.md](report.md) for the full analysis and λ trade-off table.
