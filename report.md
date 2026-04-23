# Self-Pruning Neural Network — Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The gate values are computed as `sigmoid(gate_scores)`, which maps any real number to the range (0, 1).
The L1 penalty adds `λ * Σ|gate_i|` to the loss. Because sigmoid outputs are always positive,
`|gate_i| = gate_i`, so this simply becomes `λ * Σ gate_i` — the sum of all gate values.

Minimizing this term pushes the optimizer to reduce gate values toward zero.
The reason L1 (rather than L2) works so well for sparsity is geometric:
the L1 ball has corners on the axes, so the optimal solution under a constraint tends to land
exactly on an axis where many coordinates are zero.
In practice, the gradient of the L1 penalty with respect to each gate is a constant `λ`,
providing a steady "pressure" that accumulates over training and eventually collapses
unimportant gates to zero, even when the gate is already small.
L2, by contrast, applies a gradient proportional to the gate value itself — that gradient
shrinks to near-zero as the gate shrinks, so it never quite reaches zero.

The combined loss is:

```
Total Loss = CrossEntropyLoss(predictions, labels) + λ * Σ sigmoid(gate_scores)
```

The classification loss fights to keep important gates open (because closing them hurts accuracy),
while the sparsity term fights to close every gate.
The resulting equilibrium is a sparse network where only the weights that meaningfully
contribute to accuracy survive.

---

## Results

> **Note:** Run `python train.py` to populate these values.
> Results are saved to `outputs/` and printed to stdout at the end of training.

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|------------|---------------|--------------------|
| 0.0001     | ~49%          | ~25%               |
| 0.001      | ~46%          | ~65%               |
| 0.01       | ~38%          | ~88%               |

*Above values are representative estimates; replace with actual numbers after running.*

### Interpretation

- **Low λ (0.0001):** The sparsity penalty is weak. The network retains most weights and
  achieves near-peak accuracy, but little pruning occurs.
- **Medium λ (0.001):** A clear trade-off — roughly half the weights are pruned, and accuracy
  drops modestly. This is typically the sweet spot.
- **High λ (0.01):** Aggressive pruning drives sparsity above ~85%, but the network loses
  significant capacity and accuracy degrades noticeably.

---

## Gate Value Distribution Plot

The `outputs/gates_lambda_<λ>.png` files show histograms of all gate values after training.

For the **best model (medium λ)** a successful result exhibits:
- A large spike near **0** — the pruned weights whose gates the sparsity loss collapsed.
- A secondary cluster near **0.7–1.0** — the surviving weights that the classification loss
  kept active.
- Very few values in the middle, showing clean binary-like separation.

This bimodal distribution is the hallmark of effective learned pruning.

---

## Code Structure

| File | Purpose |
|------|---------|
| `prunable_linear.py` | Custom `PrunableLinear` layer with learnable gate mechanism |
| `model.py` | `PruningNet` architecture + `sparsity_loss` helper |
| `train.py` | Data loading, training loop, evaluation, and visualization |
| `outputs/` | Saved gate distribution plots (`gates_lambda_<λ>.png`) |

### Running the experiment

```bash
python train.py
```

Requires: `torch`, `torchvision`, `matplotlib`
