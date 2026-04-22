# Self-Pruning Neural Network using L1 Regularization

## Overview
This project implements a self-pruning neural network where weights are dynamically pruned during training using learnable gates.

Each weight is multiplied by a sigmoid-based gate:
- Gate ≈ 0 → weight pruned
- Gate ≈ 1 → weight retained

L1 regularization is applied on gate values to encourage sparsity.

## Key Idea
Instead of manually pruning weights after training, the model learns which connections to keep or remove during training itself.

Total Loss:
Loss = Classification Loss + λ × Sparsity Loss

- Classification Loss → CrossEntropy
- Sparsity Loss → Sum of gate values

## Model Architecture
- Input: CIFAR-10 images (32×32×3)
- Fully Connected Network:
  - FC1: 3072 → 512
  - FC2: 512 → 256
  - FC3: 256 → 10

Custom Layer:
- `PrunableLinear`
  - Applies sigmoid(gate_score) to weights
  - Enables soft pruning

## Dataset
- CIFAR-10
- 10 classes
- Standard normalization applied


## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|------------------|--------------------|
| 0.0    | 53.87            | 0.00               |
| 0.0001 | 55.39            | 86.82              |
| 0.0005 | 52.40            | 99.09              |

## Analysis

- λ = 0.0  
  → No pruning (baseline model)

- λ = 0.0001  
  → Best balance  
  → 86.82% weights pruned  
  → Accuracy improved to 55.39%

- λ = 0.0005  
  → Extreme pruning (99.09%)  
  → Accuracy drops (underfitting)

This shows a clear **sparsity vs accuracy tradeoff**.


## Gate Distribution

![Gate Distribution](gate_distribution.png)

- Most values are near 0 → pruned weights  
- Few active weights remain → efficient model  


## Conclusion

L1 regularization on gate values successfully enables self-pruning.

- Moderate λ achieves high sparsity with good accuracy
- Excessive λ leads to over-pruning and performance loss

This validates that neural networks can learn sparse structures during training without explicit pruning steps.
