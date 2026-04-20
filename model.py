import numpy as np

class PrunableLinear:
    """Linear layer with learnable pruning gates for each weight."""
    
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with small random values for stable training
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros((out_features,))

        # Initialize gate scores in [0, 1] range (determines which connections stay active)
        self.gate_scores = np.random.rand(out_features, in_features)

    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """Forward pass: apply gated weights to input."""
        self.X = X

        # Clip gates to [0, 1] - gate value determines if connection is active (1) or pruned (0)
        self.gates = np.clip(self.gate_scores, 0, 1)
        # Apply gates to weights: pruned_weight = weight * gate
        self.pruned_weights = self.weights * self.gates

        # Standard linear transformation with gated weights
        output = X @ self.pruned_weights.T + self.bias
        return output

    def backward(self, d_out, lr):
        """Backward pass: compute gradients for weights, gates, and bias."""
        # Compute gradient of pruned weights
        d_pruned_weights = d_out.T @ self.X

        # Gradient w.r.t. weights: incorporates gate effects (gates scale weight updates)
        d_weights = d_pruned_weights * self.gates
        # Gradient w.r.t. gates: how much each gate affects output
        d_gates = d_pruned_weights * self.weights

        d_gate_scores = d_gates
        d_bias = np.sum(d_out, axis=0)

        # Update parameters with learning rate
        self.weights -= lr * d_weights
        self.bias -= lr * d_bias
        self.gate_scores -= lr * d_gate_scores

        # Keep gate scores in valid range [0.01, 1] to prevent full pruning and numerical issues
        self.gate_scores = np.clip(self.gate_scores, 0.01, 1)
        d_input = d_out @ self.pruned_weights
        self.last_grad_w = np.mean(np.abs(d_weights))
        self.last_grad_g = np.mean(np.abs(d_gate_scores))
        # print("grad W:", np.mean(np.abs(d_weights)))
        # print("grad G:", np.mean(np.abs(d_gate_scores)))
        return d_input

class SimpleNN:
    """Neural network with prunable layers for CIFAR-10 classification."""
    
    def __init__(self):
        # Architecture: 3072 (CIFAR-10 flattened) -> 256 -> 128 -> 10 (classes)
        self.layers = [
            PrunableLinear(3072, 256),   # Input layer
            PrunableLinear(256, 128),    # Hidden layer
            PrunableLinear(128, 10)      # Output layer
        ]

    def forward(self, X):
        """Forward pass through all layers with ReLU activation (except output)."""
        self.activations = []
        
        # Pass through hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            X = layer.forward(X)
            self.activations.append(X)  # Store pre-ReLU for backward pass
            X = np.maximum(0, X)        # ReLU activation
        
        # Output layer (no ReLU - raw logits)
        X = self.layers[-1].forward(X)
        return X

    def backward(self, d_out, lr):
        """Backward pass through all layers, computing ReLU gradients."""
        # Process layers in reverse order
        for i in reversed(range(len(self.layers))):
            
            if i < len(self.layers) - 1:
                # ReLU backward: gradient is 1 where activation > 0, else 0
                relu_grad = self.activations[i] > 0
                d_out = d_out * relu_grad
            
            # Backprop through layer and update parameters
            d_out = self.layers[i].backward(d_out, lr)