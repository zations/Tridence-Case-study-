
# Sparse Neural Network via Learnable Gates (NumPy Implementation)

## Overview

This project implements a **sparse neural network from scratch using NumPy**, where each weight is controlled by a **learnable gate** that determines whether the connection should be active or pruned.

The goal is to:
- Learn useful features
- Automatically remove unnecessary connections
- Achieve a balance between **model performance and sparsity**

---

## Problem Statement

Modern neural networks are often **over-parameterized**, containing many redundant connections.

This project explores:

> Can we **learn which connections are important** and prune the rest during training?

---

## Approach

### Core Idea

Each weight is associated with a **gate value**:

```

Effective Weight = Weight × Gate

```

- Gate ≈ 1 → connection active  
- Gate ≈ 0 → connection pruned  

---

### Model Architecture

- Fully connected neural network:
  - Input → 3072 (CIFAR flattened)
  - Hidden → 128 → 64
  - Output → 10 classes

- Activation: ReLU  
- Loss: Cross Entropy  
- Optimization: Manual backpropagation  

---

### Key Components

#### 1. Prunable Linear Layer
- Stores:
  - weights
  - gate_scores
- Applies:
```

pruned_weights = weights * gates

```

---

#### 2. Gate Mechanism
- Gates are learned during training
- Clipped to range:
```

gate ∈ [0.01, 1]

```

---

#### 3. Sparsity Loss
Encourages gates to go toward zero:

```

L_total = L_classification + λ × L_sparsity

```

---

#### 4. Sparsity Measurement

After training:

```

sparsity = % of gates < threshold

```

---

## Initial Problems Encountered

### Model Not Learning
- Cross entropy stuck at ~2.30
- Cause:
  - Full-batch training → gradients too small

Fix:
- Switched to **mini-batch training**

---

### Gates Not Updating
- Gradients for gates were near zero

Fix:
- Improved gradient flow and gating mechanism

---

### Model Collapse (Critical Issue)
- All gates → 0  
- Sparsity → 1.0  
- Accuracy dropped drastically  

Cause:
- Sparsity pressure too strong
- λ kept increasing

---

## Key Improvements

### 1. Mini-Batch Training
**Impact:**
- Stronger gradients
- Model started learning

---

### 2. Controlled Sparsity Scheduling

```

Epoch 0–10  → No pruning
Epoch 10–50 → Gradual pruning
Epoch 50+   → Stop pruning

```

**Impact:**
- Prevented early damage
- Allowed feature learning first

---

### 3. Reduced Pruning Strength

```

Old: 10 × lr × λ
New: 2 × lr × λ

```

**Impact:**
- Avoided aggressive pruning
- Stabilized training

---

### 4. Gate Clipping

```

gate_scores = clip(0.01, 1)

```

**Impact:**
- Prevented full network collapse

---

## Results

### Final Model Performance

| Metric | Value |
|------|------|
| Accuracy | **0.409** |
| Sparsity | **66.6%** |

---

## λ (Lambda) Experiments

| λ | Accuracy | Sparsity | Behavior |
|--|----------|----------|----------|
| 0.0 | High | Very Low | No pruning |
| 1e-3 | High | Medium | Mild pruning |
| 5e-3 | **0.409** | **0.666** | Best balance |
| 1e-2 | Low | Very High | Over-pruning |

---

## Key Insight

> Increasing λ increases sparsity but reduces accuracy.

---

### Trade-off Observed

- Small λ → better accuracy  
- Large λ → more pruning  
- Optimal λ → balance of both  

---

## Suggested Plot

Accuracy vs Sparsity graph to visualize trade-off.

---

## Key Learnings

### 1. Pruning too early hurts learning
Model needs time to learn meaningful features.

---

### 2. Excessive pruning collapses the model
Removing too many connections destroys capacity.

---

### 3. Controlled scheduling is critical
Gradual pruning works best.

---

### 4. Balance is more important than extremes
Best model is the one that balances accuracy and sparsity.

---

## How to Run

```

python main.py

```

---

## Project Structure

```

model.py      → neural network + pruning layer
train.py      → training loop + λ scheduling
utils.py      → loss + gradients
data.py       → dataset loading + preprocessing
logger.py     → training logs
main.py       → experiment runner

```

---

## Future Improvements

- Use CNN instead of fully connected layers
- Structured pruning (channel-level)
- More advanced sparsity methods (L0 regularization)
- Visualization of pruned connections

---

## Conclusion

This project demonstrates that:

- Neural networks can **learn sparsity automatically**
- Proper control of pruning is **essential**
- There is a clear **trade-off between compression and performance**

---