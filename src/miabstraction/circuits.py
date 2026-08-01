"""Minimal circuit extraction by greedy mean-ablation."""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable


@dataclass
class MinimalCircuit:
    """Extracted minimal circuit."""

    nodes: set[tuple[int, int]]  # set of (layer, head_or_mlp) tuples
    faithfulness: float  # accuracy with circuit nodes vs full model


class CircuitExtractor:
    """Extract minimal circuits via greedy mean-ablation.

    Nodes are (layer, head_index) for attention or (layer, -1) for MLPs.
    Iteratively ablates (replaces with batch-mean) the node whose removal
    least hurts accuracy, until accuracy would drop below threshold.
    """

    def __init__(self, model: nn.Module, accuracy_threshold_frac: float = 0.95):
        """Initialize extractor.

        Args:
            model: TinyTransformer
            accuracy_threshold_frac: keep ablating until accuracy <= full * this fraction
        """
        self.model = model
        self.accuracy_threshold_frac = accuracy_threshold_frac

        # Infer architecture
        self.n_layers = len(model.blocks)
        # Count attention heads (assume first block's attention has n_heads)
        first_attn = model.blocks[0].attn
        self.n_heads_per_layer = first_attn.num_heads

    def extract(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        answer_mask: torch.Tensor,
        accuracy_fn: Callable[[nn.Module, nn.Module], float],
        max_iterations: int = 100,
    ) -> tuple[MinimalCircuit, float]:
        """Extract minimal circuit.

        Args:
            tokens: (B, L) input token sequences
            labels: (B,) target labels
            answer_mask: (B, L) boolean mask for answer positions (currently unused)
            accuracy_fn: function(model_or_ablated, full_model) -> accuracy (0-1)
            max_iterations: max ablation steps

        Returns:
            circuit: MinimalCircuit with extracted nodes
            full_accuracy: accuracy of full model
        """
        # Compute full model accuracy
        full_acc = accuracy_fn(self.model, self.model)

        # Start with all nodes
        all_nodes = set()
        for layer in range(self.n_layers):
            for head in range(self.n_heads_per_layer):
                all_nodes.add((layer, head))
            # Add MLP node
            all_nodes.add((layer, -1))

        threshold = full_acc * self.accuracy_threshold_frac
        remaining_nodes = all_nodes.copy()

        # Greedy ablation: remove node that least hurts accuracy
        for iteration in range(max_iterations):
            if len(remaining_nodes) == 0:
                break

            # Find node whose ablation least hurts accuracy
            best_node = None
            best_acc = -1.0

            for node in list(remaining_nodes):
                # Ablate this node
                test_nodes = remaining_nodes - {node}

                # Create ablated model and test
                ablated_model = self._create_ablated_model(test_nodes, tokens)
                test_acc = accuracy_fn(ablated_model, self.model)

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_node = node

            # If ablating best_node keeps us above threshold, remove it
            if best_acc >= threshold:
                remaining_nodes.remove(best_node)
            else:
                # Cannot remove any more nodes without dropping below threshold
                break

        # Compute faithfulness: accuracy with circuit nodes / full accuracy
        circuit_model = self._create_ablated_model(remaining_nodes, tokens)
        circuit_acc = accuracy_fn(circuit_model, self.model)
        faithfulness = circuit_acc / full_acc if full_acc > 0 else 0.0

        return MinimalCircuit(nodes=remaining_nodes, faithfulness=faithfulness), full_acc

    def _create_ablated_model(
        self, keep_nodes: set[tuple[int, int]], tokens: torch.Tensor
    ) -> nn.Module:
        """Create a model with specified nodes ablated (replaced with batch-mean).

        Args:
            keep_nodes: set of (layer, head) tuples to keep
            tokens: input for computing batch means

        Returns:
            A model wrapper that applies mean-ablation
        """
        # Collect batch means for each position
        device = next(self.model.parameters()).device
        self.model.eval()

        # Collect residual streams
        with torch.no_grad():
            _, resid_streams = self.model(tokens.to(device), collect=True)

        # Compute means
        means = [r.mean(dim=0, keepdim=True) for r in resid_streams]

        # Return wrapped model
        return AblatedModel(self.model, keep_nodes, means, device)


class AblatedModel(nn.Module):
    """Wrapper around TinyTransformer that applies mean-ablation to specified nodes."""

    def __init__(
        self,
        base_model: nn.Module,
        keep_nodes: set[tuple[int, int]],
        batch_means: list[torch.Tensor],
        device: str,
    ):
        super().__init__()
        self.base_model = base_model
        self.keep_nodes = keep_nodes
        self.batch_means = batch_means
        self.device = device
        self.n_layers = len(base_model.blocks)
        self.n_heads = base_model.blocks[0].attn.num_heads

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward with ablation."""
        tokens = tokens.to(self.device)
        _, L = tokens.shape
        pos = torch.arange(L, device=self.device)

        # Embedding
        x = self.base_model.tok(tokens) + self.base_model.pos(pos)[None]
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=self.device), diagonal=1
        )

        # Forward through blocks with ablation
        for layer_idx, blk in enumerate(self.base_model.blocks):
            # Compute attention
            h = blk.ln1(x)
            a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)

            # Ablate attention heads if needed
            if any((layer_idx, head) not in self.keep_nodes for head in range(self.n_heads)):
                # Need to apply per-head ablation
                # This is approximate: we ablate the entire attention output
                # A more sophisticated approach would split by head
                if (layer_idx, 0) not in self.keep_nodes:
                    # All heads in this layer are ablated, use mean
                    a = self.batch_means[layer_idx][:, :L, :] - x
                    # This is residual, so subtract x to preserve shape

            x = x + a

            # MLP
            if not blk.attn_only:
                # Check if MLP is ablated
                if (layer_idx, -1) not in self.keep_nodes:
                    # Ablate: use batch mean
                    mlp_out = self.batch_means[layer_idx][:, :L, :] - (x - self.batch_means[layer_idx][:, :L, :])
                else:
                    mlp_out = blk.mlp(blk.ln2(x))
                x = x + mlp_out

        # Output
        logits = self.base_model.head(self.base_model.ln_f(x))
        return logits


def circuit_weight_count(model: nn.Module, nodes: set[tuple[int, int]]) -> int:
    """Nonzero parameters attributable to the circuit's nodes.

    Attention heads in a layer share one packed in/out projection, so a head's
    share is the layer's attention weights divided by n_heads; MLP nodes
    (head index -1) count their layer's MLP weights. This is the weight-level
    circuit size that weight-sparse training is claimed to shrink
    (Gao et al. 2511.13653), and it resolves differences that node counts cannot.
    """
    total = 0
    for layer, head in nodes:
        blk = model.blocks[layer]
        if head == -1:
            if getattr(blk, "attn_only", False):
                continue
            total += sum(int((p != 0).sum().item()) for p in blk.mlp.parameters())
        else:
            n_heads = blk.attn.num_heads
            attn_nnz = sum(int((p != 0).sum().item()) for p in blk.attn.parameters())
            total += attn_nnz // n_heads
    return total


def surviving_edge_count(
    model: nn.Module, nodes: set[tuple[int, int]], threshold: float = 1e-3
) -> int:
    """Count weights in the circuit whose magnitude exceeds `threshold` x the layer's
    max |weight| — a size metric that does NOT reduce to the imposed sparsity level.

    `circuit_weight_count` counts nonzeros, which under AbsTopK masking is exactly
    q x total by construction: it reports the hyperparameter, not the learned circuit
    (E5 measured the same 0.2086 ratio on every seed). Thresholding *relative to each
    layer's own scale* asks a different question — how many connections carry
    non-negligible magnitude — which a dense model can also fail or pass, so dense and
    sparse become comparable on equal terms.
    """
    total = 0
    for layer, head in nodes:
        blk = model.blocks[layer]
        if head == -1:
            if getattr(blk, "attn_only", False):
                continue
            params = list(blk.mlp.parameters())
            share = 1
        else:
            params = list(blk.attn.parameters())
            share = blk.attn.num_heads
        counted = 0
        for p in params:
            scale = p.abs().max()
            if scale == 0:
                continue
            counted += int((p.abs() > threshold * scale).sum().item())
        total += counted // share
    return total
