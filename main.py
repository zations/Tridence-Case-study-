import numpy as np
from model import SimpleNN
from train import train
from data import load_cifar10, preprocess
from utils import accuracy, compute_sparsity

# Load CIFAR-10 dataset
X_train, y_train, X_test, y_test = load_cifar10()

# Preprocess: normalize pixel values to [-1, 1]
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Use subset for faster training/testing (optional)
X_train = X_train[:5000]
y_train = y_train[:5000]

X_test = X_test[:1000]
y_test = y_test[:1000]

# Test different sparsity regularization strengths (lambda values)
lambda_values = [0.0, 1e-3, 5e-3, 1e-2]

results = []

# Train and evaluate model for each lambda value
for lam in lambda_values:
    print(f"\nRunning for lambda = {lam}")

    # Create fresh model for each experiment
    model = SimpleNN()

    # Train with current lambda
    train(
        model,
        X_train,
        y_train,
        epochs=100,
        lr=0.1,
        base_lambda=lam,
        log_file=f"log_lambda_{lam}.md"
    )

    # Evaluate on test set
    acc = accuracy(model, X_test, y_test)
    sp = compute_sparsity(model, threshold=0.3)  # Sparsity with 0.3 threshold

    print(f"Lambda: {lam} | Acc: {acc:.4f} | Sparsity: {sp:.4f}")

    results.append((lam, acc, sp))

# Save results to markdown file
with open("results.md", "w") as f:
    f.write("| Lambda | Accuracy | Sparsity |\n")
    f.write("|--------|----------|----------|\n")
    
    for lam, acc, sp in results:
        f.write(f"| {lam} | {acc:.4f} | {sp:.4f} |\n")