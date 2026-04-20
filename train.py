import numpy as np
from utils import softmax, cross_entropy, cross_entropy_grad, sparsity_loss, sparsity_grad
from logger import Logger


def train(
    model,
    X,
    y,
    epochs=50,
    lr=0.5,
    base_lambda=8e-3,
    batch_size=64,
    log_file="training_log.md"
):
    """Train model with pruning via learnable gates and controlled sparsity scheduling."""
    logger = Logger(log_file)

    num_samples = X.shape[0]

    for epoch in range(epochs):
        # Lambda schedule: no pruning initially (epochs 0-10), gradual pruning (10-50), no pruning after
        # This allows model to learn features first before applying sparsity pressure
        if epoch < 10:
            lam = 0                                           # No sparsity pressure
        elif epoch < 50:
            lam = base_lambda * ((epoch - 10) / 40)          # Linearly increase sparsity
        else:
            lam = 0                                           # Stop pruning pressure
        
        # Shuffle training data for mini-batch training
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_ce = 0

        # Mini-batch training loop
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            logits = model.forward(X_batch)
            probs = softmax(logits)

            # Compute losses: classification loss + sparsity regularization
            ce_loss = cross_entropy(probs, y_batch)
            sp_loss = sparsity_loss(model)  # Encourages gates toward 0
            total_loss = ce_loss + lam * sp_loss

            epoch_ce += ce_loss

            # Backward pass for classification loss
            d_out = cross_entropy_grad(probs, y_batch)
            model.backward(d_out, lr)

            # Additional sparsity update: push gates toward 0 when lambda > 0
            for layer in model.layers:
                d_gate = sparsity_grad(layer)              # Gradient pushing gates to 0
                layer.gate_scores -= 2 * lr * lam * d_gate # Scale factor of 2 prevents over-pruning

        # Average cross-entropy loss for the epoch
        epoch_ce /= (num_samples // batch_size)

        # Compute pruning metrics
        avg_gate = np.mean([np.mean(layer.gates) for layer in model.layers])  # Average gate value
        min_gate = np.min([np.min(layer.gates) for layer in model.layers])    # Minimum gate value

        # Gradient logs (last batch values)
        grad_logs = []
        for i, layer in enumerate(model.layers):
            grad_logs.append(
                f"L{i}: gradW={layer.last_grad_w:.2e}, gradG={layer.last_grad_g:.2e}"
            )

        # Final log line
        log_text = (
            f"Epoch {epoch} | "
            f"CE: {epoch_ce:.4f} | "
            f"SP: {sp_loss:.4f} | "
            f"AvgGate: {avg_gate:.4f} | MinGate: {min_gate:.4f} | "
            f"Lambda: {lam:.6f} | "
            + " | ".join(grad_logs)
        )

        logger.log(log_text)

    logger.close()