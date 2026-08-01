"""Probe monitors: runtime read-access to residual stream state.

A monitor is a fitted linear probe that reads a concept from the residual stream at
a specific layer. It includes its own fit quality (r2_val or acc_val) for the report.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from miabstraction.probes import classification_probe, regression_probe

from loom.spec import Monitor, WeaveSpec


class ProbeMonitor:
    """A fitted linear probe that reads a value from residual activations."""

    def __init__(
        self,
        name: str,
        concept: str,
        layer: int,
        probe: nn.Linear,
        fit_quality: dict[str, float],
    ):
        self.name = name
        self.concept = concept
        self.layer = layer
        self.probe = probe  # Linear layer mapping residual (d,) -> output
        self.fit_quality = fit_quality  # {"r2_val": ...} or {"acc_val": ...}

    def read(self, residuals: torch.Tensor) -> torch.Tensor:
        """Read the monitored value from a residual activation.

        Args:
            residuals: (B, d) or (d,) tensor, the residual activation at this layer.

        Returns:
            Predicted value(s): (B, out_dim) or (out_dim,).
        """
        was_1d = residuals.ndim == 1
        if was_1d:
            residuals = residuals.unsqueeze(0)

        self.probe.eval()
        with torch.no_grad():
            out = self.probe(residuals)

        return out.squeeze(0) if was_1d else out


def fit_monitors(
    model: nn.Module,
    spec: WeaveSpec,
    calib_tokens: torch.Tensor,
    ground_truth: dict[str, "np.ndarray"],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, ProbeMonitor]:
    """Fit linear probes on a calibration dataset.

    For each monitor in the spec, collect residual activations and ground-truth
    labels, then fit a linear regression or classification probe.

    Args:
        model: TinyTransformer with the collect=True capability.
        spec: WeaveSpec defining monitors.
        calib_tokens: (n_samples, seq_len) int64 token tensor.
        device: device to run on.

    Returns:
        Dict[monitor_name] -> fitted ProbeMonitor.
    """
    monitors_dict = {}

    if not spec.monitors:
        return monitors_dict

    # Collect residual activations
    model.to(device).eval()
    calib_tokens = calib_tokens.to(device)

    with torch.no_grad():
        logits, residuals_list = model(calib_tokens, collect=True)
    # residuals_list: list of (B, L, d) tensors, one per layer

    # For now, use the last layer for all monitors (can generalize later)
    # Flatten across batch and time dimension
    resid_last = residuals_list[-1]  # (B, L, d)
    B, L, d = resid_last.shape
    X = resid_last.reshape(B * L, d).cpu().numpy()

    for monitor in spec.monitors:
        if monitor.kind != "probe":
            continue

        # Ground truth is REQUIRED. A monitor fitted to synthetic labels reports a
        # confident R^2 while reading nothing (red-team FINDING 4D) — the exact
        # confident-wrong-answer failure this project catalogued. No fallback.
        if monitor.concept not in ground_truth:
            raise ValueError(
                f"Monitor '{monitor.name}' needs ground-truth labels for concept "
                f"'{monitor.concept}' (shape (n_samples*seq_len, k) aligned with "
                "calib_tokens). Refusing to fit a probe to synthetic labels."
            )
        y = np.asarray(ground_truth[monitor.concept])
        if y.shape[0] != B * L:
            raise ValueError(
                f"Ground truth for '{monitor.concept}' has {y.shape[0]} rows; "
                f"expected {B * L} (calib batch x seq_len)."
            )
        if y.ndim > 1:
            fit_result = regression_probe(X, y, val_frac=0.2, seed=0)
            quality = {"r2_val": fit_result["r2_val"], "r2_train": fit_result["r2_train"]}
        else:
            fit_result = classification_probe(X, y, val_frac=0.2, seed=0)
            quality = {"acc_val": fit_result["acc_val"], "acc_train": fit_result["acc_train"]}

        # Fit the linear probe layer
        probe = nn.Linear(d, y.shape[1] if y.ndim > 1 else 1)
        probe.to(device)

        # Actually train the probe on collected data
        opt = torch.optim.Adam(probe.parameters(), lr=0.01)
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
        y_torch = torch.tensor(y, dtype=torch.float32 if y.ndim > 1 else torch.long, device=device)

        if y.ndim > 1:
            # Regression
            loss_fn = nn.MSELoss()
        else:
            # Classification
            loss_fn = nn.CrossEntropyLoss()

        for _ in range(10):
            opt.zero_grad()
            pred = probe(X_torch)
            loss = loss_fn(pred, y_torch)
            loss.backward()
            opt.step()

        monitors_dict[monitor.name] = ProbeMonitor(
            name=monitor.name,
            concept=monitor.concept,
            layer=-1,  # last layer
            probe=probe,
            fit_quality=quality,
        )

    return monitors_dict
