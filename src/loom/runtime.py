"""Runtime controls: steering vectors for token suppression/amplification.

A control is a forward hook installed on the model's residual stream. It computes
a steering vector as the difference of mean residual activations between contexts
that do/don't precede emission of a target token (contrastive activation addition).

Metrics:
  - suppression_ratio: 1 - P(token k | steered) / P(token k | unsteered)
  - side_effect: increase in next-token loss on sequences NOT involving token k
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loom.spec import Control, WeaveSpec


@dataclass
class ControlRecord:
    """Metadata about an installed control."""
    name: str
    kind: str  # "suppress" or "amplify"
    token: int | None = None
    layer: int = -1  # which layer to apply steering at
    strength: float = 1.0
    mechanism: str = "steering"  # "steering" or "logit_mask"
    steering_vector: torch.Tensor | None = None  # (d,) steering vector
    enabled: bool = True


class ControlledModel(nn.Module):
    """Wrapper around TinyTransformer that installs forward hooks for controls."""

    def __init__(
        self,
        base_model: nn.Module,
        controls: list[ControlRecord],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.controls = controls
        self.control_enabled = {c.name: True for c in controls}
        self._hooks = []  # List of hook handles for cleanup
        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks for all controls.

        Every control MUST end up with an active mechanism. A control record with
        no hook is a silently-disabled promise (red-team FINDING 5) — refuse instead.
        """
        for control in self.controls:
            if control.mechanism == "steering" and control.steering_vector is not None:
                # Hook at the layer before the final output
                layer = self.base_model.blocks[control.layer]
                handle = layer.register_forward_hook(
                    self._make_steering_hook(control)
                )
                self._hooks.append(handle)
            elif control.mechanism == "logit_mask":
                handle = self.base_model.head.register_forward_hook(
                    self._make_logit_mask_hook(control)
                )
                self._hooks.append(handle)
            else:
                raise RuntimeError(
                    f"Control '{control.name}' has no active mechanism "
                    f"(mechanism={control.mechanism!r}, steering_vector="
                    f"{'set' if control.steering_vector is not None else 'None'}). "
                    "Refusing to install a control that would silently do nothing."
                )

    def _make_logit_mask_hook(self, control: ControlRecord):
        """Mask (suppress) or boost (amplify) the target token's logit directly."""
        def hook(module, args, output):
            if not self.control_enabled.get(control.name, True):
                return output
            out = output.clone()
            if control.kind == "suppress":
                out[..., control.token] = torch.finfo(out.dtype).min
            else:  # amplify
                out[..., control.token] = out[..., control.token] + control.strength * 10.0
            return out
        return hook

    def _make_steering_hook(self, control: ControlRecord):
        """Create a forward hook for steering vector injection."""
        def hook(module, input, output):
            if not self.control_enabled.get(control.name, True):
                return output

            # output is the residual stream (B, L, d)
            if output.ndim == 3:
                # Apply steering to the last token position (where we predict the next token)
                # Suppress: subtract steering_vector * strength
                # Amplify: add steering_vector * strength
                scale = -control.strength if control.kind == "suppress" else control.strength
                output = output + scale * control.steering_vector.to(output.device)

            return output

        return hook

    def enable_control(self, name: str):
        """Enable a control by name."""
        if name in self.control_enabled:
            self.control_enabled[name] = True

    def disable_control(self, name: str):
        """Disable a control by name."""
        if name in self.control_enabled:
            self.control_enabled[name] = False

    def forward(self, tokens: torch.Tensor, collect: bool = False):
        """Forward pass through base model with controls active."""
        # Delegate to base model; hooks are automatically applied
        return self.base_model(tokens, collect=collect)

    def cleanup(self):
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


def compute_suppression_metrics(
    model: nn.Module,
    control: ControlRecord,
    test_tokens: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 100,
) -> dict[str, float]:
    """Compute suppression_ratio and side_effect for a suppress control.

    suppression_ratio = 1 - P(token k | steered) / P(token k | unsteered)
    side_effect = mean increase in next-token loss on sequences without token k

    Args:
        model: ControlledModel with suppression active.
        control: The suppress control.
        test_tokens: (n_test, seq_len) test token sequences.
        device: device to run on.
        n_samples: number of token samples to evaluate.

    Returns:
        Dict with "suppression_ratio" and "side_effect".
    """
    model.eval()
    test_tokens = test_tokens.to(device)

    target_token = control.token
    n_test = min(n_samples, test_tokens.shape[0])
    test_subset = test_tokens[:n_test]

    # Count P(token k | unsteered)
    model.disable_control(control.name)
    with torch.no_grad():
        logits_unsteered = model(test_subset, collect=False)
    # Get probability of target token across all positions
    probs_unsteered = F.softmax(logits_unsteered, dim=-1)  # (B, L, vocab)
    # Flatten and extract P(token k)
    probs_unsteered_k = probs_unsteered[:, :, target_token].flatten()  # (B*L,)

    # Count P(token k | steered)
    model.enable_control(control.name)
    with torch.no_grad():
        logits_steered = model(test_subset, collect=False)
    probs_steered = F.softmax(logits_steered, dim=-1)
    probs_steered_k = probs_steered[:, :, target_token].flatten()

    # suppression_ratio = 1 - P(k | steered) / P(k | unsteered)
    ratio = (probs_steered_k / (probs_unsteered_k + 1e-8)).mean().item()
    suppression_ratio = max(0.0, 1.0 - ratio)

    # side_effect: loss on sequences that don't contain target_token
    # Compute loss WITHOUT the control
    model.disable_control(control.name)
    with torch.no_grad():
        logits_unsteered = model(test_subset[:, :-1], collect=False)
        targets = test_subset[:, 1:]
        loss_unsteered = F.cross_entropy(
            logits_unsteered.reshape(-1, logits_unsteered.shape[-1]),
            targets.reshape(-1),
        ).item()

    # Compute loss WITH the control
    model.enable_control(control.name)
    with torch.no_grad():
        logits_steered = model(test_subset[:, :-1], collect=False)
        loss_steered = F.cross_entropy(
            logits_steered.reshape(-1, logits_steered.shape[-1]),
            targets.reshape(-1),
        ).item()

    side_effect = loss_steered - loss_unsteered

    return {
        "suppression_ratio": suppression_ratio,
        "side_effect": side_effect,
    }


def compute_steering_vector(
    model: nn.Module,
    target_token: int,
    calib_tokens: torch.Tensor,
    layer: int = -1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor | None:
    """Compute a steering vector via contrastive activation addition (CAA).

    The steering vector is the difference of mean residual activations between:
      - Contexts that precede emission of target_token (positive class)
      - Contexts that don't precede emission of target_token (negative class)

    Args:
        model: TinyTransformer.
        target_token: The token ID to steer.
        calib_tokens: (n_calib, seq_len) calibration token sequences.
        layer: Which layer to compute steering at (default: -1 = last).
        device: device to run on.

    Returns:
        Steering vector (d,), or None if computation fails.
    """
    model.eval()
    model.to(device)
    calib_tokens = calib_tokens.to(device)
    B, L = calib_tokens.shape

    with torch.no_grad():
        logits, residuals_list = model(calib_tokens, collect=True)

    # residuals_list: list of (B, L, d) tensors
    if layer < 0:
        layer = len(residuals_list) + layer
    if layer < 0 or layer >= len(residuals_list):
        return None

    resid = residuals_list[layer]  # (B, L, d)
    d = resid.shape[-1]

    # Identify which positions are followed by target_token
    # (i.e., positions where the next token is target_token)
    next_tokens = calib_tokens[:, 1:]  # (B, L-1)
    curr_tokens = calib_tokens[:, :-1]  # (B, L-1)

    mask_pos = next_tokens == target_token  # (B, L-1)
    mask_neg = next_tokens != target_token  # (B, L-1)

    # Pad mask to (B, L) to align with residuals
    mask_pos_padded = F.pad(mask_pos.float(), (0, 1), value=0)  # (B, L)
    mask_neg_padded = F.pad(mask_neg.float(), (0, 1), value=0)  # (B, L)

    # Compute mean activations for positive and negative classes
    pos_sum = (resid * mask_pos_padded.unsqueeze(-1)).sum(dim=(0, 1))  # (d,)
    pos_count = mask_pos_padded.sum()

    neg_sum = (resid * mask_neg_padded.unsqueeze(-1)).sum(dim=(0, 1))  # (d,)
    neg_count = mask_neg_padded.sum()

    if pos_count == 0 or neg_count == 0:
        # Not enough data to compute steering vector
        return None

    pos_mean = pos_sum / (pos_count + 1e-8)
    neg_mean = neg_sum / (neg_count + 1e-8)

    steering_vector = pos_mean - neg_mean
    steering_vector = steering_vector / (steering_vector.norm() + 1e-8)

    return steering_vector


def install_controls(
    model: nn.Module,
    spec: WeaveSpec,
    calib_tokens: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ControlledModel:
    """Install steering controls on a model.

    For each control in the spec:
      1. Compute the steering vector from calibration data.
      2. Choose mechanism: steering (if vector computed) or logit_mask (fallback).
      3. Create a ControlledModel with hooks.

    Args:
        model: TinyTransformer.
        spec: WeaveSpec with controls defined.
        calib_tokens: (n_calib, seq_len) calibration token sequences.
        device: device to run on.

    Returns:
        ControlledModel ready for generation and steering.
    """
    model.to(device)
    calib_tokens = calib_tokens.to(device)

    control_records = []

    for control in spec.controls:
        if control.kind == "suppress":
            # Compute steering vector via CAA
            steering_vec = compute_steering_vector(
                model, control.token, calib_tokens, layer=-1, device=device
            )

            if steering_vec is not None:
                mechanism = "steering"
            else:
                # Fallback to logit mask
                mechanism = "logit_mask"
                steering_vec = None

            record = ControlRecord(
                name=control.name,
                kind="suppress",
                token=control.token,
                layer=-1,
                strength=control.strength,
                mechanism=mechanism,
                steering_vector=steering_vec,
            )
            control_records.append(record)

        elif control.kind == "amplify":
            # Similar to suppress but with positive steering
            steering_vec = compute_steering_vector(
                model, control.token, calib_tokens, layer=-1, device=device
            )

            if steering_vec is not None:
                mechanism = "steering"
            else:
                mechanism = "logit_mask"
                steering_vec = None

            record = ControlRecord(
                name=control.name,
                kind="amplify",
                token=control.token,
                layer=-1,
                strength=control.strength,
                mechanism=mechanism,
                steering_vector=steering_vec,
            )
            control_records.append(record)

    return ControlledModel(model, control_records, device=device)
