# ML-Explore

Implementing ML fundamentals from scratch — no libraries, no shortcuts.

Loosely inspired by [toy_ml_framework](https://github.com/sidracha/toy_ml_framework).

---

## Structure

```
ML-Explore/
├── 01_linear_algebra/          # Vectors, matrices, decompositions, eigenvalues
├── 02_calculus_autodiff/       # Numerical diff, computational graph, autograd
├── 03_probability_stats/       # Distributions, sampling, MLE/MAP, Bayes
├── 04_optimization/            # SGD, momentum, Adam, second-order methods
├── 05_classical_ml/            # Linear/logistic regression, trees, SVM, boosting
├── 06_unsupervised/            # K-means, PCA, GMM, hierarchical clustering
├── 07_neural_networks/         # Activations, layers, loss, backprop, MLP
├── 08_deep_learning/           # CNN, RNN, LSTM, attention, transformer
└── utils/                      # Data loading, metrics, visualization helpers
```

## Suggested Order

1. **Linear Algebra** — everything builds on this
2. **Calculus & Autodiff** — understand gradients before using them
3. **Optimization** — implement gradient descent from scratch
4. **Probability & Stats** — foundation for probabilistic models
5. **Classical ML** — apply the above; implement each algorithm by hand
6. **Unsupervised** — clustering and dimensionality reduction
7. **Neural Networks** — manual backprop before any framework
8. **Deep Learning** — CNNs, RNNs, attention — build on the NN layer

## Rules

- No `numpy`, `scipy`, `sklearn`, or any ML/math library (unless explicitly noted)
- Each file implements one concept end-to-end
- Test your implementations against known outputs
