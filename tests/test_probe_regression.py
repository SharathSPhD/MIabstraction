"""Regression test for probe pipeline alignment bugs.

This test catches position/label misalignment issues that cause
probe accuracies to collapse to chance levels.

Bug pattern: residual at position t (from tokens[:, :-1]) encodes prefix
tokens[0..t], which should align with beliefs[:, t] (computed from full
sequence). Using different slicing orders or off-by-one errors gives
exactly-chance accuracy on belief-state decoding, contradicting E1 findings.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from miabstraction.config import ExperimentConfig
from miabstraction.data.mess3 import belief_states, mess3_matrices, sample_sequences
from miabstraction.models import TinyTransformer, train_lm


def test_belief_argmax_probe_above_baseline():
    """
    Regression test: belief-state argmax probe MUST exceed 60% accuracy
    on trained model. E1 guarantees linear decodability with R²≈0.90,
    so logistic regression should easily exceed 33% (chance for 3-class).

    If this fails, check position/label alignment:
    - resid[:, :-1] has L-1 positions (from tokens[:, :-1])
    - beliefs[:, :] has L positions
    - Must align: resid[:, t] with beliefs[:, t] for all valid t
    - Use beliefs[:, :-1] or resid with careful indexing
    """
    cfg = ExperimentConfig.load("configs/e3_sae_control.yaml")
    cfg.data["n_seq"] = 500  # Fast training
    cfg.data["seq_len"] = 16
    cfg.train["steps"] = 200

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate and train
    T = mess3_matrices(**cfg.data.get("mess3", {}))
    n_seq, L = cfg.data["n_seq"], cfg.data["seq_len"]
    seqs = sample_sequences(T, n_seq, L, rng)
    tokens = torch.from_numpy(seqs)

    model = TinyTransformer(vocab=3, **cfg.model)
    losses = train_lm(model, tokens, device=dev, **cfg.train)

    # Verify training occurred
    assert losses[-1] < losses[0] * 0.9, "Model did not train"

    model.eval()
    model.to(dev)

    # Generate eval sequences
    eval_seqs = sample_sequences(T, 200, L, rng)
    eval_tokens = torch.from_numpy(eval_seqs)

    # Collect residuals (from tokens[:, :-1])
    with torch.no_grad():
        logits, resid = model(eval_tokens[:, :-1].to(dev), collect=True)
        resid_raw = resid[1].float().cpu().numpy()  # (200, L-1, d_model)

    # Collect beliefs (from full sequences)
    beliefs = belief_states(T, eval_seqs)  # (200, L, n_states)

    # CRITICAL: Align positions
    # resid_raw[:, t] encodes prefix tokens[0..t]
    # This corresponds to beliefs[:, t]
    # Since resid_raw has L-1 positions (0..L-2), align with beliefs[:, 0..L-2]
    burn = 8
    resid_aligned = resid_raw[:, burn:, :]  # (200, L-1-burn, d_model)
    beliefs_aligned = beliefs[:, burn:-1, :]  # (200, L-1-burn, n_states) — must use :-1 to match resid positions

    # Verify alignment
    assert resid_aligned.shape[1] == beliefs_aligned.shape[1], \
        f"Position mismatch: resid {resid_aligned.shape[1]} vs beliefs {beliefs_aligned.shape[1]}"

    # Extract features and labels
    X = resid_aligned.reshape(-1, resid_aligned.shape[-1])
    y = beliefs_aligned.argmax(axis=2).reshape(-1)

    # Train/val split
    n = len(X)
    perm = rng.permutation(n)
    n_val = int(n * 0.2)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Probe with StandardScaler (prevent unstandardized data issues)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    clf.fit(X_scaled[train_idx], y[train_idx])

    train_acc = accuracy_score(y[train_idx], clf.predict(X_scaled[train_idx]))
    val_acc = accuracy_score(y[val_idx], clf.predict(X_scaled[val_idx]))

    print(f"Belief-state argmax probe: train={train_acc:.4f}, val={val_acc:.4f}")

    # REGRESSION: Must exceed 60% (far above 33% chance for 3-class)
    # If this fails, position/label alignment is broken
    assert val_acc > 0.60, \
        f"Belief probe accuracy {val_acc:.4f} is at chance (33%). " \
        f"Check position/label alignment: resid has {resid_aligned.shape[1]} positions, " \
        f"beliefs has {beliefs_aligned.shape[1]} positions."
