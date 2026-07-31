"""Linear probing with a clean train/val protocol."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score


def regression_probe(
    X: np.ndarray, Y: np.ndarray, val_frac: float = 0.2, seed: int = 0
) -> dict:
    """Linear regression X -> Y. Returns train/val R² (uniform average over targets)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    reg = LinearRegression().fit(X[tr], Y[tr])
    return {
        "r2_train": float(r2_score(Y[tr], reg.predict(X[tr]))),
        "r2_val": float(r2_score(Y[va], reg.predict(X[va]))),
        "pred_val": reg.predict(X[va]),
        "idx_val": va,
    }


def classification_probe(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int = 0, C: float = 1.0
) -> dict:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    clf = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
    return {
        "acc_train": float(accuracy_score(y[tr], clf.predict(X[tr]))),
        "acc_val": float(accuracy_score(y[va], clf.predict(X[va]))),
    }
