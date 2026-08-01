"""E6 — SAEs on a real pretrained LM vs the same architecture at random init.

E3 asked whether standard SAE quality metrics separate a trained toy model from an
untrained one, and refuted H3 in an unexpected direction: the random network's
low-dimensional activations were EASIER to reconstruct. The Heap et al. control was
designed for full-scale pretrained transformers, though, so the honest version of the
question runs there: a downloaded Qwen against the same config at random init, on real
text, with the SAE trained identically on each.

This is a replication probe, not a new preregistered hypothesis: the result records
which way the separation runs and how far outside seed noise it is, and updates
nothing by itself.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..config import ExperimentConfig
from ..sae import train_sae


def chunk_tokens(ids: list[int], seq_len: int, n_seq: int) -> np.ndarray:
    """First n_seq non-overlapping windows of seq_len tokens. Refuses to pad, wrap or
    repeat: a pipeline that silently invents data trains on noise and reports numbers
    earned on nothing."""
    need = seq_len * n_seq
    if len(ids) < need:
        raise ValueError(f"need {need} tokens for {n_seq} sequences of {seq_len}, "
                         f"only {len(ids)} available")
    return np.asarray(ids[:need], dtype=np.int64).reshape(n_seq, seq_len)


def separation(trained: list[dict], control: list[dict]) -> dict:
    """Per metric: mean/std for each arm, the direction of the gap stated in words,
    and a flag when a metric shows zero variance — a result that reproduces perfectly
    may be reproducing a constant."""
    out: dict = {}
    for key in trained[0]:
        t = np.array([r[key] for r in trained], dtype=float)
        c = np.array([r[key] for r in control], dtype=float)
        gap = float(t.mean() - c.mean())
        if key == "fvu":
            direction = ("control_easier_to_reconstruct" if gap > 0
                         else "trained_easier_to_reconstruct")
        else:
            direction = "trained_higher" if gap > 0 else "control_higher"
        out[key] = {
            "trained_mean": float(t.mean()), "trained_std": float(t.std()),
            "control_mean": float(c.mean()), "control_std": float(c.std()),
            "gap": gap, "direction": direction,
            "zero_variance_flag": bool(t.std() < 1e-12 and c.std() < 1e-12),
        }
    return out


@torch.no_grad()
def collect_resid(model, tokens: torch.Tensor, layer: int, device: str,
                  batch: int = 8) -> np.ndarray:
    """Hidden states at `layer` (hidden_states index, so 0 is the embedding),
    gathered in float32 whatever the model computes in."""
    model.eval()
    outs = []
    for i in range(0, len(tokens), batch):
        b = tokens[i:i + batch].to(device)
        hs = model(input_ids=b, output_hidden_states=True).hidden_states[layer]
        outs.append(hs.float().cpu().numpy())
    return np.concatenate(outs)


def run(cfg: ExperimentConfig) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    name = cfg.model["name"]
    layer = int(cfg.model["resid_layer"])

    tok = AutoTokenizer.from_pretrained(name)
    trained = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    trained.to(dev)

    # The control: the same architecture with everything it learned removed. This is
    # the Heap et al. construction in its intended habitat.
    torch.manual_seed(cfg.seed)
    control = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))
    control.to(torch.bfloat16).to(dev)

    from datasets import load_dataset
    ds = load_dataset(cfg.data["dataset"], cfg.data.get("config"),
                      split=cfg.data.get("split", "train"))
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    tokens = torch.from_numpy(chunk_tokens(ids, cfg.data["seq_len"],
                                           cfg.data["n_seq"]))

    resid_t = collect_resid(trained, tokens, layer, dev)
    resid_c = collect_resid(control, tokens, layer, dev)
    del trained, control
    torch.cuda.empty_cache()

    d_model = resid_t.shape[-1]
    a = cfg.analysis
    burn = int(a.get("burn_in", 4))
    xs_t = torch.from_numpy(resid_t[:, burn:, :].reshape(-1, d_model))
    xs_c = torch.from_numpy(resid_c[:, burn:, :].reshape(-1, d_model))

    rows_t, rows_c = [], []
    for s in range(int(a.get("n_sae_seeds", 3))):
        for xs, rows in ((xs_t, rows_t), (xs_c, rows_c)):
            sae, _ = train_sae(xs, int(a["sae_dict_size"]), int(a["sae_k"]),
                               lr=float(a.get("sae_lr", 1e-3)),
                               steps=int(a.get("sae_steps", 1000)),
                               seed=1000 + s, device=dev,
                               batch_size=int(a.get("sae_batch", 1024)))
            rows.append(sae.compute_metrics(xs[:20000].to(dev)))
            del sae
            torch.cuda.empty_cache()

    sep = separation(rows_t, rows_c)
    fvu = sep["fvu"]
    replicates = fvu["direction"] == "control_easier_to_reconstruct"
    return {
        "experiment": cfg.name, "hypothesis": cfg.hypothesis, "seed": cfg.seed,
        "model": name, "resid_layer": layer, "d_model": d_model,
        "n_tokens_probed": int(xs_t.shape[0]),
        "trained": rows_t, "control": rows_c, "separation": sep,
        "toy_result_replicates": bool(replicates),
        "note": ("On the toy model the refutation of H3 ran through the control being "
                 "easier to reconstruct; this records whether a full-scale pretrained "
                 "model shows the same direction. No Bayesian update follows from an "
                 "unpreregistered replication probe."),
        "wall_clock_s": round(time.time() - t0, 1),
    }
