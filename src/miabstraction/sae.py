"""Sparse Autoencoder (SAE) with top-k activation.

Top-k sparsity is enforced by selecting the k largest absolute values of the
latent activations, zeroing the rest. The reconstruction is computed from these
sparse latents via an untied decoder.

Metrics:
  - FVU (Fraction of Variance Unexplained): residual variance / total variance
  - L0: mean number of non-zero latents per sample
  - Dead latent fraction: fraction of dictionary dimensions never activated
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """Sparse autoencoder with top-k activation and untied decoder.

    Architecture:
      x (d_model) -> encoder (Linear d->m) -> top-k -> latents (m)
                                           -> decoder (Linear m->d) -> recon (d)
    """

    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model)

    def forward(
        self, x: torch.Tensor, return_latents: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with top-k sparsity.

        Args:
            x: (... , d_model) input activations
            return_latents: if True, return (reconstruction, latents)

        Returns:
            recon: (..., d_model) reconstructed activations
            latents: (..., dict_size) sparse latent activations (if return_latents=True)
        """
        # Flatten all but last dimension
        shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        # Encode
        z = self.encoder(x_flat)  # (batch, dict_size)

        # Top-k sparsity
        topk_vals, topk_idx = torch.topk(torch.abs(z), self.k, dim=1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(1, topk_idx, z.gather(1, topk_idx))

        # Decode
        recon = self.decoder(z_sparse)  # (batch, d_model)

        # Reshape back to original shape
        recon = recon.reshape(shape)
        if return_latents:
            z_sparse = z_sparse.reshape(*shape[:-1], self.dict_size)
            return recon, z_sparse

        return recon

    def compute_metrics(self, x: torch.Tensor) -> dict:
        """Compute FVU, L0, and dead-latent fraction on a batch.

        Args:
            x: (batch, d_model) or (batch, seq_len, d_model)

        Returns:
            dict with keys: fvu, l0, dead_frac
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        with torch.no_grad():
            recon, latents_flat = self(x, return_latents=True)
            latents_flat = latents_flat.reshape(-1, self.dict_size)

            # FVU: fraction of variance unexplained
            residual = x_flat - recon.reshape(-1, self.d_model)
            var_residual = residual.var(dim=0).sum()
            var_total = x_flat.var(dim=0).sum()
            fvu = (var_residual / (var_total + 1e-8)).item()

            # L0: mean number of non-zero latents
            nnz = (latents_flat != 0).sum(dim=1).float().mean()
            l0 = nnz.item()

            # Dead latent fraction: neuron is dead if never non-zero
            never_active = ((latents_flat != 0).sum(dim=0) == 0).float().mean()
            dead_frac = never_active.item()

        return {"fvu": fvu, "l0": l0, "dead_frac": dead_frac}


def train_sae(
    x: torch.Tensor,
    dict_size: int,
    k: int,
    lr: float = 0.01,
    steps: int = 1000,
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 64,
    log_every: int = 100,
) -> tuple[SAE, dict]:
    """Train a sparse autoencoder on activations.

    Args:
        x: (n_samples,) + shape, input activations (will be flattened to (N, d_model))
        dict_size: dictionary size (m)
        k: top-k sparsity level
        lr: learning rate for Adam
        steps: training steps
        seed: random seed
        device: device to train on
        batch_size: batch size for training
        log_every: log metrics every N steps

    Returns:
        sae: trained SAE model
        metrics: dict with keys:
          - fvu_final: final FVU on training data
          - l0_mean: final L0
          - dead_frac: final dead latent fraction
          - loss_history: list of losses per step
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Flatten x to (N, d_model)
    shape = x.shape
    if len(shape) == 1:
        d_model = 1
        x_flat = x.reshape(-1, 1)
    else:
        d_model = shape[-1]
        x_flat = x.reshape(-1, d_model)

    x_flat = x_flat.to(device)

    # Initialize SAE
    sae = SAE(d_model=d_model, dict_size=dict_size, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    # Training loop
    loss_history = []
    rng = np.random.default_rng(seed)

    for step in range(steps):
        # Random batch
        idx = rng.choice(x_flat.shape[0], size=batch_size, replace=True)
        batch = x_flat[idx]

        # Forward pass
        recon = sae(batch)

        # Reconstruction loss
        loss = F.mse_loss(recon, batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())

        if (step + 1) % log_every == 0 or step == steps - 1:
            sae.eval()
            with torch.no_grad():
                m = sae.compute_metrics(x_flat)
            sae.train()

    # Final metrics on full training set
    sae.eval()
    metrics_final = sae.compute_metrics(x_flat)

    result = {
        "fvu_final": metrics_final["fvu"],
        "l0_mean": metrics_final["l0"],
        "dead_frac": metrics_final["dead_frac"],
        "loss_history": [float(l) for l in loss_history],
    }

    return sae, result
