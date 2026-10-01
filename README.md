# ML-Explore

Implementing a neural network framework from scratch in C++.
Based on [toy_ml_framework](https://github.com/sidracha/toy_ml_framework).

No libraries. Every concept built by hand.

---

## Structure

```
ML-Explore/
├── include/        # Header files (.h)  — class/function declarations
├── src/            # Source files (.cc) — implementations
└── Makefile
```

## Topic List

| # | Header | Source | What to implement |
|---|--------|--------|-------------------|
| 1 | `tensor.h` | `tensor.cc` | Tensor class: data, grad, shape, stride; strided indexing; permute; transpose |
| 2 | `tensor_ops.h` | `tensor_ops.cc` | Element-wise ops: +, -, *, /, pow, exp |
| 3 | `calc.h` | `calc.cc` | Odometer (n-dim index iteration); 2D GEMM; batched GEMM; bias add |
| 4 | `tensor_grad.h` | `tensor_grad.cc` | Backward fns for every op: add, sub, mul, div, pow, matmul, bias add, transpose |
| 5 | `nn.h` | `nn.cc` | ReLU, Sigmoid, Softmax, Concat — forward + backward |
| 6 | `layer.h` | *(header-only)* | Abstract Layer: `forward`, `zero_grad`, `gradient_descent_step`; SequentialLayer; Model |
| 7 | `linear.h` | `linear.cc` | Linear layer (Wx + b); LinearReLU; LinearSigmoid; MLP |
| 8 | `losses.h` | `losses.cc` | MSE Loss — forward + backward |
| 9 | `optimizer.h` | `optimizer.cc` | SGD: `zero_grad`, `step` |
| 10 | `attention.h` | `attention.cc` | Scaled dot-product attention; multi-head SelfAttn; SelfAttnBlock |
| 11 | `train_loop.h` | `train_loop.cc` | Data generation; training loop |
| 12 | *(none)* | `main.cc` | Entry point — wire everything together |

## Build Order

```
tensor → tensor_ops → calc
       ↓
tensor_grad  (backward fns for all ops above)
       ↓
nn  (activations)
       ↓
layer → linear → losses → optimizer
       ↓
attention → train_loop → main
```

## C++ Concepts You'll Need

- Classes, constructors, operator overloading
- `std::vector`, `std::shared_ptr`, `std::unique_ptr`
- `std::function` (for storing backward fns)
- Templates (SequentialLayer::register_layer)
- Virtual functions / abstract base classes
- Move semantics
