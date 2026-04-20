import numpy as np
import pickle
import os
import urllib.request
import tarfile


def download_cifar():
    """Download CIFAR-10 dataset if not already present."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"

    if not os.path.exists("cifar-10-batches-py"):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)

        print("Extracting CIFAR-10...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()


def load_batch(file):
    """Load a single CIFAR-10 batch file.
    
    Returns:
        data: (10000, 3072) - pixel values
        labels: (10000,) - class labels
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    data = data_dict[b'data']       # (10000, 3072)
    labels = data_dict[b'labels']

    return data, labels


def load_cifar10():
    """Load full CIFAR-10 dataset (train + test).
    
    Returns:
        X_train: (50000, 3072) - training images
        y_train: (50000,) - training labels
        X_test: (10000, 3072) - test images
        y_test: (10000,) - test labels
    """
    download_cifar()

    X_train = []
    y_train = []

    # Load all 5 training batches
    for i in range(1, 6):
        data, labels = load_batch(f'cifar-10-batches-py/data_batch_{i}')
        X_train.append(data)
        y_train.extend(labels)

    X_train = np.concatenate(X_train)   # (50000, 3072)
    y_train = np.array(y_train)

    # Load test batch
    X_test, y_test = load_batch('cifar-10-batches-py/test_batch')
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def preprocess(X):
    """Normalize pixel values to [-1, 1] range.
    
    Converts from [0, 255] → [0, 1] → [-1, 1]
    """
    X = X / 255.0                  # Normalize to [0, 1]
    X = (X - 0.5) / 0.5            # Shift to [-1, 1]
    return X

