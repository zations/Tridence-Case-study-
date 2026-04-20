import numpy as np

def softmax(x):
    """Convert logits to probabilities with numerical stability."""
    # Subtract max for numerical stability (prevents overflow)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs, y_true):
    """Compute cross-entropy loss for classification.
    
    Args:
        probs: (batch_size, num_classes) - predicted probabilities
        y_true: (batch_size,) - true class labels (integers)
    """
    n = probs.shape[0]
    # Get probability of true class for each sample
    log_probs = -np.log(probs[range(n), y_true] + 1e-9)
    return np.mean(log_probs)

def cross_entropy_grad(probs, y_true):
    """Compute gradient of cross-entropy loss w.r.t. logits."""
    n = probs.shape[0]
    grad = probs.copy()
    grad[range(n), y_true] -= 1  # Subtract 1 from true class probability
    return grad / n

def sparsity_loss(model):
    """Measure average gate activation - loss pushes gates toward 0."""
    total = 0
    count = 0
    
    # Average all gates across all layers
    for layer in model.layers:
        total += np.sum(layer.gates)
        count += layer.gates.size
    
    return total / count   # Normalized average gate value

def sparsity_grad(layer):
    """Compute gradient to push gates toward 0 for sparsity."""
    # dL/dgates = 1 (pushing gates toward lower values)
    d_gates = np.ones_like(layer.gates)

    # For numerical stability with gate values in [0, 1]
    sigmoid_grad = layer.gates * (1 - layer.gates)

    return d_gates * sigmoid_grad

def accuracy(model, X, y):
    """Compute classification accuracy on dataset."""
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)  # Get class with highest logit
    return np.mean(preds == y)

def compute_sparsity(model, threshold=0.25):
    """Compute sparsity: percentage of gates below threshold (pruned connections)."""
    total = 0
    active = 0

    for layer in model.layers:
        total += layer.gates.size
        # Count gates above threshold as active connections
        active += np.sum(layer.gates > threshold)

    sparsity = 1 - (active / total)  # Percentage of pruned connections
    return sparsity