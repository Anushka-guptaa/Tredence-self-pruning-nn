# Self-Pruning Neural Network using L1 Regularization

## Overview
This project implements a **self-pruning neural network** that learns to remove unimportant weights *during training* using learnable gates.  
Each weight is multiplied by a sigmoid-based gate:
- Gate ≈ 0 → weight is pruned
- Gate ≈ 1 → weight is retained

L1 regularization is applied on the gate values to encourage sparsity.

## Why L1 Penalty on the Sigmoid Gates Encourages Sparsity
The total loss is:  
**Loss = CrossEntropyLoss + λ × SparsityLoss**

where **SparsityLoss = sum of all gate values** (L1 norm of `sigmoid(gate_scores)`).  

Because the L1 norm heavily penalizes non-zero values, the optimizer is strongly encouraged to drive most `gate_scores` towards large negative numbers.  
This makes `sigmoid(gate_score)` → **0**, effectively pruning those weights while keeping the entire process fully differentiable and trainable end-to-end.

## Model Architecture
- **Dataset**: CIFAR-10 (32×32×3 images, 10 classes)
- **Custom Layer**: `PrunableLinear` (replaces `nn.Linear`)
- **Network**:
  - FC1: 3072 → 512
  - FC2: 512 → 256
  - FC3: 256 → 10
- Activation: ReLU
- Optimizer: Adam (lr=0.001)

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|--------------------|
| 0.0    | 53.87             | 0.00               |
| 0.0001 | 55.39             | 86.82              |
| 0.0005 | 52.40             | 99.09              |

## Analysis
- **λ = 0.0** → No pruning (baseline model)
- **λ = 0.0001** → **Best balance**  
  → 86.82% weights pruned  
  → Accuracy actually improved to **55.39%**
- **λ = 0.0005** → Extreme pruning (99.09%)  
  → Slight accuracy drop due to over-pruning

This clearly demonstrates the **sparsity vs accuracy trade-off**.

## Gate Distribution (Best Model)
![Gate Distribution](gate_distribution.png)

The plot shows:
- A very large spike near **0** (pruned weights)
- A smaller cluster of values away from 0 (active weights)

This is exactly the desired bimodal distribution for successful self-pruning.

## Conclusion
L1 regularization on sigmoid gates successfully enables the network to **prune itself during training**.  
Moderate λ values achieve very high sparsity (86%+) with no loss in accuracy — proving that neural networks can learn sparse architectures dynamically without any post-training pruning step.

## How to Run
```bash
pip install torch torchvision torchaudio matplotlib numpy
python self_pruning_nn.py
