# ML-Explore

Implementing a neural network framework from scratch — based on [toy_ml_framework](https://github.com/sidracha/toy_ml_framework).

No libraries. Every concept built by hand.

---

## Topic List

| # | Folder | What to implement |
|---|--------|-------------------|
| 1 | `01_tensor/` | Tensor data structure: data, grad, shape, stride; strided indexing; permute; transpose |
| 2 | `02_matmul/` | 2D matrix multiply (GEMM), batched GEMM, bias add |
| 3 | `03_autograd/` | Computational graph (predecessor edges, node ownership); topological sort; backward traversal; gradient accumulation |
| 4 | `03_autograd/grad_fns.py` | Chain-rule backward functions for every op: add, sub, mul, div, pow, exp, matmul, bias add, transpose |
| 5 | `04_activations/` | ReLU, Sigmoid, Softmax — forward AND backward |
| 6 | `04_activations/` | Concat along a dim — forward AND backward |
| 7 | `05_layers/` | Abstract Layer interface: `forward`, `zero_grad`, `gradient_descent_step` |
| 8 | `05_layers/linear.py` | Linear layer: `Wx + b`, random weight init |
| 9 | `05_layers/mlp.py` | MLP: stacked Linear layers with activations |
| 10 | `06_loss/mse.py` | MSE Loss — forward AND backward |
| 11 | `07_optimizer/sgd.py` | SGD optimizer: `zero_grad`, `step` |
| 12 | `08_attention/` | Scaled dot-product attention: `QK^T / sqrt(d)` → softmax → `V` |
| 13 | `08_attention/` | Multi-head self-attention: Q/K/V projections, per-head compute, output projection |
| 14 | `08_attention/` | Attention block: Self-Attention + MLP |
| 15 | `09_training/` | Training loop: forward → loss → backward → optimizer step |

## Suggested Build Order

Build bottom-up — each layer depends on the one before it:

```
Tensor → Element-wise ops → GEMM
       ↓
Computational graph + backward fns
       ↓
Activations (ReLU, Sigmoid, Softmax, Concat)
       ↓
Layer abstraction → Linear → MLP
       ↓
MSE Loss + SGD
       ↓
Attention → Attention Block
       ↓
Training loop
```

## Rules

- No external libraries (no numpy, torch, etc.)
- Each file implements one concept end-to-end
- Implement forward AND backward for every operation
