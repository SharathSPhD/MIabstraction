"""E3: SAE random-transformer control (H3).

Train SAEs on a trained TinyTransformer vs. an untrained (random) control.
Measure if standard SAE metrics (FVU, L0, dead-fraction) distinguish the two.

H3 supported iff metrics do NOT separate trained vs random by >3 sigma.
If they do separate, it means SAE metrics ARE sensitive to learned structure.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.mess3 import belief_states, mess3_matrices, sample_sequences
from ..models import TinyTransformer, train_lm
from ..sae import train_sae


def collect_resid(
    model: TinyTransformer, tokens: torch.Tensor, layer: int, device: str, batch: int = 256
) -> np.ndarray:
    """Collect residual stream from a specific layer, flattened."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch):
            _, resid = model(tokens[i : i + batch].to(device), collect=True)
            outs.append(resid[layer].float().cpu().numpy())
    return np.concatenate(outs)  # (n_samples, seq_len, d_model)


def compute_selectivity(
    latents: np.ndarray, beliefs: np.ndarray, top_k: int
) -> float:
    """Compute selectivity: max class-conditional activation for top-k latents.

    Args:
        latents: (n_samples, n_positions, dict_size) sparse latent activations
        beliefs: (n_samples, n_positions, n_states) belief state probabilities
        top_k: consider top-k latents by activation magnitude

    Returns:
        mean selectivity over top-k latents
    """
    n_samples, n_pos, dict_size = latents.shape
    n_states = beliefs.shape[-1]

    # Quantize beliefs to argmax class (3 classes for Mess3)
    belief_class = beliefs.argmax(axis=2)  # (n_samples, n_pos)

    # Flatten
    latents_flat = latents.reshape(-1, dict_size)  # (n_samples*n_pos, dict_size)
    belief_class_flat = belief_class.reshape(-1)  # (n_samples*n_pos,)

    # Get top-k active latents by total activation
    mean_activation = np.abs(latents_flat).mean(axis=0)  # (dict_size,)
    top_latents = np.argsort(-mean_activation)[:top_k]

    selectivity_scores = []
    for lat_idx in top_latents:
        lat_activation = np.abs(latents_flat[:, lat_idx])  # (n_samples*n_pos,)

        # For each belief class, compute mean activation
        class_means = []
        for c in range(n_states):
            mask = belief_class_flat == c
            if mask.sum() > 0:
                class_means.append(lat_activation[mask].mean())
            else:
                class_means.append(0)

        # Selectivity = max class conditional / overall mean
        overall_mean = lat_activation.mean()
        if overall_mean > 1e-8:
            selectivity = max(class_means) / overall_mean
        else:
            selectivity = 0

        selectivity_scores.append(selectivity)

    return float(np.mean(selectivity_scores))


def run(cfg: ExperimentConfig) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"

    # Generate Mess3 data
    T = mess3_matrices(**cfg.data.get("mess3", {}))
    n_seq, L = cfg.data["n_seq"], cfg.data["seq_len"]
    seqs = sample_sequences(T, n_seq, L, rng)
    tokens = torch.from_numpy(seqs)

    # Train model
    model = TinyTransformer(vocab=3, **cfg.model)
    losses = train_lm(model, tokens, device=dev, **cfg.train)

    # Create untrained control (same architecture, random init)
    control = TinyTransformer(vocab=3, **cfg.model)

    # Collect residual-stream activations from specified layer
    model.to(dev)
    control.to(dev)

    resid_layer = cfg.analysis.get("resid_layer", 1)
    n_probe = cfg.analysis.get("n_probe_seq", 2000)
    probe_seqs = sample_sequences(T, n_probe, L, rng)
    probe_tokens = torch.from_numpy(probe_seqs)

    resid_trained = collect_resid(model, probe_tokens[:, :-1], resid_layer, dev)
    resid_random = collect_resid(control, probe_tokens[:, :-1], resid_layer, dev)

    # Flatten for SAE training: (n_samples * seq_len, d_model)
    d_model = resid_trained.shape[-1]
    resid_trained_flat = resid_trained.reshape(-1, d_model)
    resid_random_flat = resid_random.reshape(-1, d_model)

    # SAE hyperparameters
    dict_size = cfg.analysis.get("sae_dict_size", 256)
    k = cfg.analysis.get("sae_k", 32)
    sae_lr = cfg.analysis.get("sae_lr", 0.01)
    sae_steps = cfg.analysis.get("sae_steps", 1000)
    n_seeds = cfg.analysis.get("n_sae_seeds", 5)

    # Train multiple SAEs with different seeds for both trained and random
    metrics_trained = {"fvu": [], "l0": [], "dead_frac": []}
    metrics_random = {"fvu": [], "l0": [], "dead_frac": []}

    for seed_offset in range(n_seeds):
        seed = cfg.seed + seed_offset + 1000  # Offset to avoid collision with model seed

        # Train on learned model residuals
        sae_t, m_t = train_sae(
            torch.from_numpy(resid_trained_flat).float(),
            dict_size=dict_size,
            k=k,
            lr=sae_lr,
            steps=sae_steps,
            seed=seed,
            device=dev,
        )
        metrics_trained["fvu"].append(m_t["fvu_final"])
        metrics_trained["l0"].append(m_t["l0_mean"])
        metrics_trained["dead_frac"].append(m_t["dead_frac"])

        # Train on random model residuals
        sae_r, m_r = train_sae(
            torch.from_numpy(resid_random_flat).float(),
            dict_size=dict_size,
            k=k,
            lr=sae_lr,
            steps=sae_steps,
            seed=seed,
            device=dev,
        )
        metrics_random["fvu"].append(m_r["fvu_final"])
        metrics_random["l0"].append(m_r["l0_mean"])
        metrics_random["dead_frac"].append(m_r["dead_frac"])

    # Compute statistics
    def stats(values):
        return float(np.mean(values)), float(np.std(values))

    fvu_t_mean, fvu_t_std = stats(metrics_trained["fvu"])
    fvu_r_mean, fvu_r_std = stats(metrics_random["fvu"])
    l0_t_mean, l0_t_std = stats(metrics_trained["l0"])
    l0_r_mean, l0_r_std = stats(metrics_random["l0"])
    dead_t_mean, dead_t_std = stats(metrics_trained["dead_frac"])
    dead_r_mean, dead_r_std = stats(metrics_random["dead_frac"])

    # Compute separation in units of sigma (combined std)
    # Separation = (mean_A - mean_B) / sqrt(std_A^2 + std_B^2)
    def separation_sigma(m1, s1, m2, s2):
        if s1 + s2 < 1e-8:
            return 0
        return abs(m1 - m2) / np.sqrt(s1**2 + s2**2)

    fvu_sep = separation_sigma(fvu_t_mean, fvu_t_std, fvu_r_mean, fvu_r_std)
    l0_sep = separation_sigma(l0_t_mean, l0_t_std, l0_r_mean, l0_r_std)
    dead_sep = separation_sigma(dead_t_mean, dead_t_std, dead_r_mean, dead_r_std)

    # Compute selectivity on SAE from trained model
    burn = cfg.analysis.get("burn_in", 8)
    belief_data = belief_states(T, probe_seqs)  # (n_probe, seq_len, n_states)
    selectivity_top_k = cfg.analysis.get("selectivity_top_k", 64)

    # Need to get latents from trained SAE on positions >= burn_in
    with torch.no_grad():
        resid_t_tensor = torch.from_numpy(resid_trained).float().to(dev)
        _, latents_t = sae_t(resid_t_tensor, return_latents=True)
        latents_t = latents_t.cpu().numpy()  # (n_probe, seq_len-1, dict_size)

    # Note: resid_trained has seq_len-1 positions (no last token), but belief_data
    # has seq_len positions. Align by taking beliefs for positions 0:seq_len-1
    belief_data_aligned = belief_data[:, :-1, :]  # (n_probe, seq_len-1, n_states)

    selectivity = compute_selectivity(
        latents_t[:, burn:, :],
        belief_data_aligned[:, burn:, :],
        top_k=selectivity_top_k,
    )

    # H3 is supported if metrics do NOT separate trained vs random by >3 sigma
    # (meaning SAE metrics are NOT sensitive to learned structure)
    all_seps = [fvu_sep, l0_sep, dead_sep]
    supports = all(sep <= 3.0 for sep in all_seps)

    result = {
        "hypothesis": cfg.hypothesis,
        "supports": supports,
        "final_loss": float(np.mean(losses[-50:])),
        "fvu_mean_trained": fvu_t_mean,
        "fvu_std_trained": fvu_t_std,
        "fvu_mean_random": fvu_r_mean,
        "fvu_std_random": fvu_r_std,
        "fvu_separation_sigma": fvu_sep,
        "l0_mean_trained": l0_t_mean,
        "l0_std_trained": l0_t_std,
        "l0_mean_random": l0_r_mean,
        "l0_std_random": l0_r_std,
        "l0_separation_sigma": l0_sep,
        "dead_frac_mean_trained": dead_t_mean,
        "dead_frac_std_trained": dead_t_std,
        "dead_frac_mean_random": dead_r_mean,
        "dead_frac_std_random": dead_r_std,
        "dead_frac_separation_sigma": dead_sep,
        "selectivity_trained": selectivity,
        "fvu_mean": fvu_t_mean,
        "fvu_std": fvu_t_std,
        "l0_mean": l0_t_mean,
        "l0_std": l0_t_std,
        "dead_frac_mean": dead_t_mean,
        "dead_frac_std": dead_t_std,
        "config_hash": cfg.hash(),
        "runtime_s": round(time.time() - t0, 1),
        "device": dev,
    }

    d = cfg.result_dir()
    (d / "result.json").write_text(json.dumps(result, indent=2))
    _plot_comparison(result, d)

    return result


def _plot_comparison(result, d):
    """Plot SAE metrics comparison: trained vs random."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["FVU", "L0", "Dead Fraction"]
    trained_means = [
        result["fvu_mean_trained"],
        result["l0_mean_trained"],
        result["dead_frac_mean_trained"],
    ]
    trained_stds = [
        result["fvu_std_trained"],
        result["l0_std_trained"],
        result["dead_frac_std_trained"],
    ]
    random_means = [
        result["fvu_mean_random"],
        result["l0_mean_random"],
        result["dead_frac_mean_random"],
    ]
    random_stds = [
        result["fvu_std_random"],
        result["l0_std_random"],
        result["dead_frac_std_random"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(1)
    width = 0.35

    for i, (ax, metric, t_mean, t_std, r_mean, r_std) in enumerate(
        zip(axes, metrics, trained_means, trained_stds, random_means, random_stds)
    ):
        ax.bar(x - width / 2, [t_mean], width, label="Trained", yerr=[t_std], capsize=5)
        ax.bar(x + width / 2, [r_mean], width, label="Random", yerr=[r_std], capsize=5)
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([""])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("E3: SAE Metrics Comparison (Trained vs Random Model)")
    fig.savefig(d / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
