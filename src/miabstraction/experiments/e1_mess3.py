"""E1: belief-state geometry (H1).

Train a small transformer on Mess3 sequences; probe every residual-stream layer
for the ground-truth belief states. H1 supported iff best val R² >= 0.7 AND the
trained model beats an untrained control by a wide margin.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.mess3 import belief_states, mess3_matrices, sample_sequences
from ..models import TinyTransformer, train_lm
from ..probes import regression_probe


def collect_resid(model: TinyTransformer, tokens: torch.Tensor, device: str,
                  batch: int = 256) -> list[np.ndarray]:
    """Residual stream per layer, flattened over (seq, pos>=burn_in)."""
    model.eval()
    outs: list[list[np.ndarray]] = []
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch):
            _, resid = model(tokens[i : i + batch].to(device), collect=True)
            for li, r in enumerate(resid):
                if len(outs) <= li:
                    outs.append([])
                outs[li].append(r.float().cpu().numpy())
    return [np.concatenate(o) for o in outs]


def run(cfg: ExperimentConfig) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"

    T = mess3_matrices(**cfg.data.get("mess3", {}))
    n_seq, L = cfg.data["n_seq"], cfg.data["seq_len"]
    seqs = sample_sequences(T, n_seq, L, rng)
    tokens = torch.from_numpy(seqs)

    model = TinyTransformer(vocab=3, **cfg.model)
    control = TinyTransformer(vocab=3, **cfg.model)  # untrained, same init family

    losses = train_lm(model, tokens, device=dev, **cfg.train)

    # probe on held-out sequences
    n_probe = cfg.analysis.get("n_probe_seq", 2000)
    probe_seqs = sample_sequences(T, n_probe, L, rng)
    beliefs = belief_states(T, probe_seqs)
    burn = cfg.analysis.get("burn_in", 2)
    # inputs to model are tokens[:, :-1]; resid position t encodes prefix ..t
    pt = torch.from_numpy(probe_seqs)
    Y = beliefs[:, burn:-1, :].reshape(-1, 3)

    layer_r2, control_r2 = [], []
    for m, store in ((model, layer_r2), (control, control_r2)):
        m.to(dev)
        resid = collect_resid(m, pt[:, :-1], dev)
        for r in resid:
            X = r[:, burn:, :].reshape(-1, r.shape[-1])
            res = regression_probe(X, Y, seed=cfg.seed)
            store.append(res["r2_val"])

    # recent-token baseline: one-hot of last k tokens -> beliefs. Any representation
    # (including a random reservoir) that merely stores recent tokens is bounded by this.
    k = cfg.analysis.get("window_k", 8)
    n, Lm1 = probe_seqs.shape[0], probe_seqs.shape[1] - 1
    onehot = np.eye(3)[probe_seqs[:, :-1]]  # (n, L-1, 3)
    Xw = np.stack(
        [
            np.concatenate([onehot[:, t - k + 1 : t + 1].reshape(n, -1)], axis=1)
            for t in range(burn, Lm1)
            if t - k + 1 >= 0
        ],
        axis=1,
    )
    t_start = max(burn, k - 1)
    Yw = beliefs[:, t_start:-1, :].reshape(-1, 3)
    window_r2 = regression_probe(
        Xw.reshape(-1, Xw.shape[-1]), Yw, seed=cfg.seed
    )["r2_val"]

    best = float(max(layer_r2))
    best_control = float(max(control_r2))
    supports = best >= 0.7 and best > best_control and best > window_r2
    result = {
        "hypothesis": cfg.hypothesis,
        "supports": supports,
        "final_loss": float(np.mean(losses[-50:])),
        "r2_val_by_layer": [float(v) for v in layer_r2],
        "r2_val_by_layer_control": [float(v) for v in control_r2],
        "best_r2": best,
        "best_r2_control": best_control,
        "r2_window_baseline": float(window_r2),
        "window_k": k,
        "leak_budget": float(1 - best),
        "config_hash": cfg.hash(),
        "runtime_s": round(time.time() - t0, 1),
        "device": dev,
    }

    d = cfg.result_dir()
    (d / "result.json").write_text(json.dumps(result, indent=2))
    _plot_geometry(model, pt, beliefs, burn, dev, int(np.argmax(layer_r2)), d)
    return result


def _plot_geometry(model, pt, beliefs, burn, dev, best_layer, d):
    """Project best layer's residual onto belief simplex via the probe and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    resid = collect_resid(model, pt[:, :-1], dev)[best_layer]
    X = resid[:, burn:, :].reshape(-1, resid.shape[-1])
    Y = beliefs[:, burn:-1, :].reshape(-1, 3)
    pred = LinearRegression().fit(X, Y).predict(X)

    def simplex_xy(b):
        # barycentric -> 2D
        v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
        return b @ v

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, B, title in ((axes[0], Y, "ground-truth beliefs"),
                         (axes[1], pred, "linear readout of residual stream")):
        xy = simplex_xy(B)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, alpha=0.25,
                   c=B, edgecolors="none")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"E1 Mess3 belief-state geometry (layer {best_layer})")
    fig.savefig(d / "belief_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
