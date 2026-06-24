"""Accuracy and convergence metrics for Stage 1."""

import torch
import torch.nn.functional as F


def classification_accuracy(network, data_loader, device: torch.device) -> float:
    """Direct classification accuracy via eval-mode forward pass + argmax.

    Matches the reference evaluation in Bogacz-Group/PredictiveCoding notebook.
    """
    network.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = network(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


def linear_probe_accuracy(network, train_loader, test_loader,
                          device: torch.device, layer_idx: int = 0,
                          max_train: int = None) -> float:
    """Train a logistic-regression linear probe on frozen PC representations.

    Extracts representations at `layer_idx` via network.get_representation(),
    trains a logistic regression with sklearn, and returns test accuracy.

    layer_idx=0: first hidden layer activations (H-dim, after ReLU)
    layer_idx=1: second hidden layer activations (H-dim, after ReLU)
    max_train: if set, limits training samples (for fast per-epoch probe).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    def _extract(loader, max_samples=None):
        feats, labels = [], []
        n_collected = 0
        network.eval()
        with torch.no_grad():
            for x, y in loader:
                if max_samples is not None and n_collected >= max_samples:
                    break
                x = x.to(device)
                r = network.get_representation(x, layer_idx=layer_idx).cpu().numpy()
                feats.append(r)
                labels.append(y.numpy())
                n_collected += len(y)
        return np.concatenate(feats)[:max_samples], np.concatenate(labels)[:max_samples]

    X_train, y_train = _extract(train_loader, max_samples=max_train)
    X_test,  y_test  = _extract(test_loader)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    iters = 500 if max_train is not None else 2000
    clf = LogisticRegression(max_iter=iters, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def convergence_gap(energies: list) -> float:
    """Fractional energy reduction: (E_0 - E_final) / E_0."""
    if not energies or energies[0] == 0:
        return 0.0
    return (energies[0] - energies[-1]) / energies[0]


def is_monotonically_decreasing(energies: list, tolerance: float = 1e-6) -> bool:
    """True if energy never increases by more than `tolerance` between consecutive steps."""
    for i in range(1, len(energies)):
        if energies[i] > energies[i - 1] + tolerance:
            return False
    return True
