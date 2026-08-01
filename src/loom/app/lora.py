"""Parameter-efficient fine-tuning with Low-Rank Adaptation (LoRA).

Injects low-rank adapters into attention projections while keeping base weights frozen.
Trainable parameters are < 2% of the full model, allowing safe fine-tuning without
destroying the base model's capabilities.

Core API:
  - attach_lora(model, rank, alpha, targets) -> handles
  - lora_parameters(model) -> list of trainable parameters
  - merge_or_detach(model, handles, mode) -> model
"""
from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class LoRAHandle:
    """Handle to a single LoRA adapter for cleanup/merging."""
    layer_name: str
    module: nn.Module
    adapter_a: nn.Parameter
    adapter_b: nn.Parameter
    scale: float


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter.

    Forward pass: output = base(x) + scale * adapter_b @ adapter_a @ x
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze base weights
        for param in self.base.parameters():
            param.requires_grad = False

        # Create adapters
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # A: in_features -> rank
        self.adapter_a = nn.Parameter(
            torch.zeros(in_features, rank, dtype=base_linear.weight.dtype)
        )
        # B: rank -> out_features
        self.adapter_b = nn.Parameter(
            torch.zeros(rank, out_features, dtype=base_linear.weight.dtype)
        )

        # Initialize adapters: A ~ N(0, 1/rank), B ~ 0
        # This ensures adapter contribution starts near zero
        nn.init.kaiming_uniform_(self.adapter_a, a=0, mode='fan_out')
        self.adapter_b.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with base + adapter."""
        base_out = self.base(x)
        # Adapter: (... x in) @ (in x rank) @ (rank x out) = (... x out)
        adapter_out = torch.matmul(x, self.adapter_a)  # (... x rank)
        adapter_out = torch.matmul(adapter_out, self.adapter_b)  # (... x out)
        return base_out + self.scale * adapter_out


class LoRAConv1D(nn.Module):
    """Wraps a frozen transformer Conv1D with a low-rank adapter.

    Conv1D from transformers is actually a linear layer with weight shape (in, out).
    Forward pass: output = base(x) + scale * (x @ adapter_a) @ adapter_b
    """

    def __init__(self, base_conv1d, rank: int, alpha: float):
        super().__init__()
        self.base = base_conv1d
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze base weights
        for param in self.base.parameters():
            param.requires_grad = False

        # Conv1D has weight of shape (in_features, out_features)
        # but accessed as .weight with shape (nf, weight_shape[0])
        # nf is the output feature dimension
        out_features = self.base.nf  # Output features
        in_features = self.base.weight.shape[0]  # Input features

        # A: in_features -> rank
        self.adapter_a = nn.Parameter(
            torch.zeros(in_features, rank, dtype=self.base.weight.dtype)
        )
        # B: rank -> out_features
        self.adapter_b = nn.Parameter(
            torch.zeros(rank, out_features, dtype=self.base.weight.dtype)
        )

        # Initialize
        nn.init.kaiming_uniform_(self.adapter_a, a=0, mode='fan_out')
        self.adapter_b.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with base + adapter."""
        base_out = self.base(x)
        # Adapter contribution
        adapter_out = torch.matmul(x, self.adapter_a)  # (... x rank)
        adapter_out = torch.matmul(adapter_out, self.adapter_b)  # (... x out)
        return base_out + self.scale * adapter_out


def attach_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    targets: Optional[list[str]] = None,
) -> list[LoRAHandle]:
    """Attach LoRA adapters to attention projections in the model.

    Args:
        model: The base model (frozen)
        rank: Rank of the low-rank update
        alpha: Scaling factor (typically 2 * rank)
        targets: List of layer name patterns to adapt (e.g., ['q_proj', 'v_proj']).
                If None, auto-detect common attention projection names.

    Returns:
        List of LoRAHandle objects for later cleanup/merging.

    Note:
        - Base weights are frozen after attachment
        - Trainable parameters added are approximately 2 * rank * (in_features + out_features) per layer
        - Scales adapter output by alpha/rank to keep contributions balanced
    """
    if targets is None:
        # Common attention projection names across model families
        # Llama/Meta models: q_proj, v_proj, k_proj, o_proj, gate_proj
        # GPT-2/Phi: c_attn, c_proj, c_fc
        # Gemma: q_proj, v_proj, o_proj
        targets = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'fc1', 'fc2', 'gate_proj',
                   'c_attn', 'c_proj', 'c_fc']

    handles = []

    def _should_attach(name: str) -> bool:
        """Check if this layer matches our target patterns."""
        for pattern in targets:
            if pattern in name:
                return True
        return False

    for name, module in model.named_modules():
        # Check if this is a layer we should wrap with LoRA
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = type(module).__name__ == 'Conv1D'  # Check by name to avoid import

        if (is_linear or is_conv1d) and _should_attach(name):
            # Replace this layer with a LoRA-wrapped version
            parts = name.split('.')
            if len(parts) == 1:
                # Top-level attribute on the model
                parent = model
                child_name = parts[0]
            else:
                # Nested attribute
                parent_name = '.'.join(parts[:-1])
                child_name = parts[-1]
                # Navigate to parent and replace
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)

            # Create appropriate wrapped module
            if is_linear:
                lora_module = LoRALinear(module, rank=rank, alpha=alpha)
            else:  # is_conv1d
                lora_module = LoRAConv1D(module, rank=rank, alpha=alpha)

            setattr(parent, child_name, lora_module)

            handles.append(LoRAHandle(
                layer_name=name,
                module=lora_module,
                adapter_a=lora_module.adapter_a,
                adapter_b=lora_module.adapter_b,
                scale=lora_module.scale,
            ))

    return handles


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Get all trainable LoRA parameters (adapter_a and adapter_b weights).

    Returns:
        List of adapter parameters that should be trained.
    """
    params = []
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1D)):
            params.append(module.adapter_a)
            params.append(module.adapter_b)
    return params


def freeze_base_weights(model: nn.Module) -> None:
    """Freeze all non-adapter parameters in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.base.parameters():
                param.requires_grad = False
        elif not isinstance(module, nn.Parameter):
            # For non-LoRA modules, freeze everything
            for param in module.parameters():
                if not _is_adapter_param(model, param):
                    param.requires_grad = False


def _is_adapter_param(model: nn.Module, param: nn.Parameter) -> bool:
    """Check if a parameter is part of a LoRA adapter."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if param is module.adapter_a or param is module.adapter_b:
                return True
    return False


def merge_or_detach(
    model: nn.Module,
    handles: list[LoRAHandle],
    mode: str = "detach",
) -> nn.Module:
    """Clean up LoRA adapters after training.

    Args:
        model: Model with attached LoRA
        handles: List of LoRAHandle objects from attach_lora()
        mode: 'merge' to merge adapters into base weights (unfreeze),
              'detach' to remove adapters and restore base module

    Returns:
        Model with adapters either merged or removed.
    """
    if mode == "merge":
        # Merge adapters into base weights and unfreeze
        for handle in handles:
            lora_mod = handle.module
            if isinstance(lora_mod, LoRALinear):
                # Merge: W_final = W_base + scale * (adapter_b.T @ adapter_a.T)
                # adapter_a: (in_features, rank)
                # adapter_b: (rank, out_features)
                # We want delta of shape (out_features, in_features) to match W.shape
                # delta = (out_features x rank) @ (rank x in_features) = (out_features x in_features)
                with torch.no_grad():
                    delta = lora_mod.scale * torch.matmul(
                        lora_mod.adapter_b.T,  # (out_features x rank)
                        lora_mod.adapter_a.T  # (rank x in_features)
                    )  # (out_features x in_features)
                    lora_mod.base.weight.data.add_(delta)  # Add to base weight (out x in)

                # Unfreeze base weights
                for param in lora_mod.base.parameters():
                    param.requires_grad = True

            elif isinstance(lora_mod, LoRAConv1D):
                # Merge for Conv1D: W_final = W_base + scale * (adapter_a @ adapter_b)
                # adapter_a: (in_features, rank)
                # adapter_b: (rank, out_features)
                # Conv1D weight shape: (in_features, out_features)
                # delta = (in_features x rank) @ (rank x out_features) = (in_features x out_features)
                with torch.no_grad():
                    delta = lora_mod.scale * torch.matmul(
                        lora_mod.adapter_a,  # (in_features x rank)
                        lora_mod.adapter_b  # (rank x out_features)
                    )  # (in_features x out_features)
                    lora_mod.base.weight.data.add_(delta)  # Add to base weight

                # Unfreeze base weights
                for param in lora_mod.base.parameters():
                    param.requires_grad = True

    elif mode == "detach":
        # Restore original base modules (no merge)
        for handle in handles:
            lora_mod = handle.module
            if isinstance(lora_mod, (LoRALinear, LoRAConv1D)):
                # Restore original base module
                parts = handle.layer_name.split('.')
                if len(parts) == 1:
                    # Top-level
                    parent = model
                    child_name = parts[0]
                else:
                    parent_name = '.'.join(parts[:-1])
                    child_name = parts[-1]
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                setattr(parent, child_name, lora_mod.base)

                # Unfreeze base weights
                for param in lora_mod.base.parameters():
                    param.requires_grad = True

    return model


def get_adapter_info(model: nn.Module) -> dict:
    """Get statistics about attached LoRA adapters.

    Returns:
        Dict with counts and parameter statistics.
    """
    n_adapters = 0
    adapter_params = 0
    base_params = 0

    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1D)):
            n_adapters += 1
            # Count adapter params
            adapter_params += module.adapter_a.numel() + module.adapter_b.numel()
            # Count base params
            base_params += sum(p.numel() for p in module.base.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "n_adapters": n_adapters,
        "adapter_params": adapter_params,
        "base_params": base_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / total_params, 6) if total_params > 0 else 0,
        "adapter_ratio": round(adapter_params / total_params, 6) if total_params > 0 else 0,
    }
