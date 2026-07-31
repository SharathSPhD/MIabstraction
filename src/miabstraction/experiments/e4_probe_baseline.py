"""E4: Linear probe vs. SAE probe (H4).

Compare logistic regression probes on:
  (a) raw residual activations
  (b) SAE latent activations (from E3 trained SAE)

Two concepts:
  1. Belief-state argmax region (3-class classification)
  2. Belief-state entropy above/below median (2-class classification)

H4 supported iff raw-activation probe >= SAE probe - 1pt accuracy on both concepts.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.mess3 import belief_states, mess3_matrices, sample_sequences
from ..models import TinyTransformer, train_lm
from ..probes import classification_probe
from ..sae import train_sae


def collect_resid(
    model: TinyTransformer, tokens: torch.Tensor, layer: int, device: str, batch: int = 256
) -> np.ndarray:
    """Collect residual stream from a specific layer."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch):
            _, resid = model(tokens[i : i + batch].to(device), collect=True)
            outs.append(resid[layer].float().cpu().numpy())
    return np.concatenate(outs)


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
    model.to(dev)

    # Collect residual-stream activations
    resid_layer = cfg.analysis.get("resid_layer", 1)
    n_eval = cfg.analysis.get("n_eval_seq", 2000)
    eval_seqs = sample_sequences(T, n_eval, L, rng)
    eval_tokens = torch.from_numpy(eval_seqs)

    resid_raw = collect_resid(model, eval_tokens[:, :-1], resid_layer, dev)
    d_model = resid_raw.shape[-1]

    # Flatten and get belief states
    burn = cfg.analysis.get("burn_in", 8)
    belief_data = belief_states(T, eval_seqs)  # (n_eval, L, 3)

    # CRITICAL ALIGNMENT:
    # resid_raw from model(tokens[:, :-1]) has shape (n_eval, L-1, d_model)
    # resid_raw[:, t] encodes prefix tokens[0..t], corresponding to beliefs[:, t]
    # Since resid_raw has L-1 positions, align with beliefs[:, :-1] (drop last belief)
    resid_aligned = resid_raw[:, burn:, :]  # (n_eval, L-1-burn, d_model)
    beliefs_aligned = belief_data[:, burn:-1, :]  # (n_eval, L-1-burn, 3) — note :-1 to match resid

    # Flatten activations and beliefs to (samples, d_model/dict_size)
    resid_flat = resid_aligned.reshape(-1, d_model)
    beliefs_flat = beliefs_aligned.reshape(-1, 3)  # 3 states for Mess3

    # Concept 1: Belief-state argmax (3-class)
    y_belief_class = beliefs_flat.argmax(axis=1)

    # Concept 2: Belief-state entropy above/below median (2-class)
    belief_entropy = -np.sum(beliefs_flat * np.log(beliefs_flat + 1e-8), axis=1)
    entropy_median = np.median(belief_entropy)
    y_entropy_class = (belief_entropy > entropy_median).astype(int)

    # Train SAE on raw activations (using training data)
    train_seqs = sample_sequences(T, n_seq, L, rng)
    train_tokens = torch.from_numpy(train_seqs)
    resid_train = collect_resid(model, train_tokens[:, :-1], resid_layer, dev)
    # Apply same alignment to training data
    resid_train_aligned = resid_train[:, burn:, :]
    resid_train_flat = resid_train_aligned.reshape(-1, d_model)

    dict_size = cfg.analysis.get("sae_dict_size", 256)
    k = cfg.analysis.get("sae_k", 32)
    sae_lr = cfg.analysis.get("sae_lr", 0.01)
    sae_steps = cfg.analysis.get("sae_steps", 1000)
    sae_seed = cfg.analysis.get("sae_seed", 1000)

    sae_model, _ = train_sae(
        torch.from_numpy(resid_train_flat).float(),
        dict_size=dict_size,
        k=k,
        lr=sae_lr,
        steps=sae_steps,
        seed=sae_seed,
        device=dev,
    )

    # Get SAE latents on evaluation data
    with torch.no_grad():
        resid_eval_tensor = torch.from_numpy(resid_flat).float().to(dev)
        _, latents_sae = sae_model(resid_eval_tensor, return_latents=True)
        latents_sae = latents_sae.cpu().numpy()

    # Test probes with StandardScaler to handle unstandardized activations
    from sklearn.preprocessing import StandardScaler

    # Standardize activations for better probe convergence
    scaler_raw = StandardScaler()
    resid_flat_scaled = scaler_raw.fit_transform(resid_flat)

    scaler_sae = StandardScaler()
    latents_sae_scaled = scaler_sae.fit_transform(latents_sae)

    # Concept 1: Belief-state argmax (3-class)
    raw_probe_belief = classification_probe(resid_flat_scaled, y_belief_class, seed=cfg.seed, C=1.0)
    sae_probe_belief = classification_probe(latents_sae_scaled, y_belief_class, seed=cfg.seed, C=1.0)

    raw_acc_belief = raw_probe_belief["acc_val"]
    sae_acc_belief = sae_probe_belief["acc_val"]

    # Concept 2: Entropy above/below median (2-class)
    raw_probe_entropy = classification_probe(resid_flat_scaled, y_entropy_class, seed=cfg.seed, C=1.0)
    sae_probe_entropy = classification_probe(latents_sae_scaled, y_entropy_class, seed=cfg.seed, C=1.0)

    raw_acc_entropy = raw_probe_entropy["acc_val"]
    sae_acc_entropy = sae_probe_entropy["acc_val"]

    # H4: raw probe >= SAE probe - 1pt on both concepts
    raw_belief_wins = raw_acc_belief >= (sae_acc_belief - 0.01)
    raw_entropy_wins = raw_acc_entropy >= (sae_acc_entropy - 0.01)
    supports = raw_belief_wins and raw_entropy_wins

    result = {
        "hypothesis": cfg.hypothesis,
        "supports": supports,
        "final_loss": float(np.mean(losses[-50:])),
        "raw_probe_acc_belief": raw_acc_belief,
        "sae_probe_acc_belief": sae_acc_belief,
        "raw_probe_acc_entropy": raw_acc_entropy,
        "sae_probe_acc_entropy": sae_acc_entropy,
        "raw_vs_sae_delta_belief": raw_acc_belief - sae_acc_belief,
        "raw_vs_sae_delta_entropy": raw_acc_entropy - sae_acc_entropy,
        "entropy_median": float(entropy_median),
        "config_hash": cfg.hash(),
        "runtime_s": round(time.time() - t0, 1),
        "device": dev,
    }

    d = cfg.result_dir()
    (d / "result.json").write_text(json.dumps(result, indent=2))
    _plot_comparison(result, d)

    return result


def _plot_comparison(result, d):
    """Plot probe accuracy comparison: raw vs SAE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    concepts = ["Belief State\nArgmax (3-class)", "Entropy\nAbove/Below (2-class)"]
    raw_accs = [result["raw_probe_acc_belief"], result["raw_probe_acc_entropy"]]
    sae_accs = [result["sae_probe_acc_belief"], result["sae_probe_acc_entropy"]]

    x = np.arange(len(concepts))
    width = 0.35

    for i, (ax, concept, raw, sae) in enumerate(zip(axes, concepts, raw_accs, sae_accs)):
        ax.bar(x[i] - width / 2, raw, width, label="Raw Activations")
        ax.bar(x[i] + width / 2, sae, width, label="SAE Latents")
        ax.set_ylabel("Validation Accuracy")
        ax.set_ylim([0, 1.0])
        ax.set_xticks([x[i]])
        ax.set_xticklabels([concept])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add delta text
        delta = raw - sae
        color = "green" if delta >= -0.01 else "red"
        ax.text(x[i], max(raw, sae) + 0.05, f"Δ={delta:.3f}", ha="center", color=color)

    fig.suptitle("E4: Linear Probe Comparison (Raw vs SAE)")
    fig.savefig(d / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
