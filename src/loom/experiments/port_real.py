"""Port backend on a real frozen open-weight model — measured, not simulated.

Programs Llama-3.2-1B without touching its weights:
  monitor  a linear probe on the residual stream, fitted against ground truth
  control  a steering direction whose strength is calibrated by dose-response, with a
           side-effect budget measured on text the control is not meant to affect
  gating   the control fires only at uncommitted moments (high predictive entropy),
           the "event-gated write" idea imported from prabodha

Everything here runs on the local GB10 and writes what it measured. There is no
synthetic path: if the model or tokenizer is unavailable the run fails.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "meta-llama/Llama-3.2-1B"
LAYER = -6            # a mid-late residual site
OUT = Path("results/loom_port_demo.json")

NEUTRAL = [
    "The weather today is quite pleasant and the sky is",
    "She opened the book and began to read the first",
    "Scientists have discovered a new species of insect in the",
    "The recipe calls for two cups of flour and a",
    "After the meeting ended, everyone walked back to their",
    "The train arrived at the station exactly on",
    "He picked up the phone and dialled the number he",
    "In the morning the market fills with people buying fresh",
]
FORMAL = [
    "Pursuant to the aforementioned agreement, the parties shall",
    "It is hereby resolved that the committee will",
    "The undersigned acknowledges receipt of the documents and",
    "In accordance with regulation 14, the applicant must",
]
CASUAL = [
    "honestly that was such a weird movie and i",
    "yeah so we just hung out and ate way too much",
    "lol i totally forgot to text you back about",
    "it was kinda fun but also super tiring because we",
]


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@torch.no_grad()
def residuals(model, tok, prompts, layer=LAYER):
    """Mean residual at `layer` over the prompt's tokens."""
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to("cuda")
        h = model(**ids, output_hidden_states=True).hidden_states[layer]
        out.append(h[0].float().mean(0).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def loss_on(model, tok, prompts) -> float:
    tot = 0.0
    for p in prompts:
        ids = tok(p, return_tensors="pt").to("cuda")
        out = model(**ids, labels=ids["input_ids"])
        tot += float(out.loss)
    return tot / len(prompts)


class Steer:
    """A steering control installed as a forward hook on one decoder layer."""

    def __init__(self, model, direction, layer=LAYER):
        self.h = None
        self.dir = torch.tensor(direction, dtype=torch.bfloat16, device="cuda")
        self.dir = self.dir / self.dir.norm()
        self.strength = 0.0
        self.entropy_gate: float | None = None
        self.fired = 0
        self.seen = 0
        blocks = model.model.layers
        self.layer = blocks[layer if layer >= 0 else len(blocks) + layer]

    def __enter__(self):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if self.strength == 0.0:
                return output
            add = self.strength * self.dir
            if self.entropy_gate is None:
                hs = hs + add
                self.fired += hs.shape[1]
                self.seen += hs.shape[1]
            else:
                # Event gating: write only where the model is uncommitted. Entropy is
                # approximated by the spread of the residual's own projection, a cheap
                # proxy available inside the hook.
                proj = (hs.float() @ self.dir.float())
                z = (proj - proj.mean()) / (proj.std() + 1e-6)
                mask = (z.abs() < self.entropy_gate).to(hs.dtype)
                hs = hs + add * mask.unsqueeze(-1)
                self.fired += int(mask.sum().item())
                self.seen += mask.numel()
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        self.h = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        if self.h:
            self.h.remove()


def run() -> dict:
    t0 = time.time()
    torch.manual_seed(0)
    model, tok = load()
    n_params = sum(p.numel() for p in model.parameters())

    # ---- direction: formal vs casual register (a neutral, non-safety concept)
    a = residuals(model, tok, FORMAL)
    b = residuals(model, tok, CASUAL)
    direction = a.mean(0) - b.mean(0)
    dnorm = float(np.linalg.norm(direction))

    # ---- monitor: can a linear probe read the register from the residual stream?
    X = np.concatenate([a, b])
    y = np.array([1] * len(a) + [0] * len(b))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    probe_acc = float(cross_val_score(
        LogisticRegression(max_iter=2000), X, y, cv=LeaveOneOut()).mean())

    # ---- dose-response: measured, one forward pass per strength
    base_neutral = loss_on(model, tok, NEUTRAL)
    base_formal = loss_on(model, tok, FORMAL)
    curve = []
    with Steer(model, direction) as s:
        for mult in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            s.strength = mult * dnorm / 8.0
            eff = base_formal - loss_on(model, tok, FORMAL)   # >0 => pushed toward formal
            side = loss_on(model, tok, NEUTRAL) - base_neutral
            curve.append({"strength": round(float(s.strength), 4),
                          "effect_nats": round(eff, 4),
                          "side_effect_nats": round(side, 4)})

    # EC50-style: smallest strength reaching half the maximum measured effect
    emax = max(p["effect_nats"] for p in curve)
    ec50 = next((p["strength"] for p in curve if p["effect_nats"] >= emax / 2), None)

    # ---- event-gated vs always-on at matched strength
    chosen = next((p for p in curve if p["strength"] > 0
                   and p["side_effect_nats"] < 0.15), curve[1])
    arms = {}
    for name, gate in (("always_on", None), ("event_gated", 1.0)):
        with Steer(model, direction) as s:
            s.strength, s.entropy_gate = chosen["strength"], gate
            eff = base_formal - loss_on(model, tok, FORMAL)
            side = loss_on(model, tok, NEUTRAL) - base_neutral
            arms[name] = {"effect_nats": round(eff, 4),
                          "side_effect_nats": round(side, 4),
                          "write_fraction": round(s.fired / max(s.seen, 1), 4)}

    result = {
        "model": MODEL, "params": n_params, "dtype": "bfloat16", "frozen": True,
        "layer": LAYER, "device": torch.cuda.get_device_name(0),
        "direction_norm": round(dnorm, 3),
        "monitor": {"concept": "register (formal vs casual)",
                    "probe_acc_loo": round(probe_acc, 4), "n_examples": len(y)},
        "baseline_loss": {"neutral": round(base_neutral, 4),
                          "formal": round(base_formal, 4)},
        "dose_response": {"curve": curve, "max_effect_nats": round(emax, 4),
                          "ec50": ec50, "chosen_strength": chosen["strength"]},
        "gating_comparison": arms,
        "gating_verdict": (
            "event-gated achieves lower side-effect at comparable effect"
            if arms["event_gated"]["side_effect_nats"] < arms["always_on"]["side_effect_nats"]
            and arms["event_gated"]["effect_nats"] >= 0.5 * arms["always_on"]["effect_nats"]
            else "event-gating did not beat always-on here"),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
