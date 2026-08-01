"""Execute a Loom program's mech-interp operations against any substrate.

Every operation here goes through `loom.backends.Backend`, so the same program runs on
a model built from an architecture spec and on a model downloaded from the open-weight
ecosystem. Where an operation cannot be realized on a given substrate, the executor
refuses it with a reason — the way a compiler reports that an intrinsic is unavailable
on the selected target, rather than silently emitting something else.

Operations (the language's verbs):
    realize   bring the target into being (build or load)
    read      fit a probe monitor on a feature          -> probe_acc
    amplify   dose-calibrated steering along a feature  -> effect, side_effect
    install   link a separately compiled circuit        -> the circuit's own gate
    assert    measure a gate; failure fails the build
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .backends import Backend, ModelHandle, for_target
from .stdlib import require_circuit, require_feature


@dataclass
class OpResult:
    op: str
    ok: bool
    metrics: dict = field(default_factory=dict)
    diagnosis: str = ""


@dataclass
class RunReport:
    target: str
    backend: str
    substrate: dict
    ops: list[OpResult] = field(default_factory=list)
    gates: list[dict] = field(default_factory=list)
    passed: bool = False
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target": self.target, "backend": self.backend,
            "substrate": self.substrate,
            "ops": [{"op": o.op, "ok": o.ok, "metrics": o.metrics,
                     "diagnosis": o.diagnosis} for o in self.ops],
            "gates": self.gates, "passed": self.passed,
            "wall_clock_s": round(self.wall_clock_s, 1),
        }


# ------------------------------------------------------------------ measurement

@torch.no_grad()
def _resid(be: Backend, m: ModelHandle, batches, layer: int, dev: str) -> np.ndarray:
    """Mean residual at `layer` for each input — the feature's read site."""
    out = []
    cap: dict = {}
    h = be.residual_hook(m, layer, lambda hs: cap.__setitem__("h", hs))
    try:
        for b in batches:
            be.logits(m, b.to(dev))
            hs = cap["h"]
            out.append(hs.float().mean(1).cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out)


@torch.no_grad()
def _loss(be: Backend, m: ModelHandle, batches, dev: str) -> float:
    tot, n = 0.0, 0
    for b in batches:
        tot += float(be.forward_loss(m, b.to(dev)))
        n += 1
    return tot / max(n, 1)


class _Steer:
    """A steering write on the residual stream, through the backend's hook."""

    def __init__(self, be: Backend, m: ModelHandle, direction: np.ndarray, layer: int):
        self.be, self.m, self.layer = be, m, layer
        d = torch.tensor(direction, dtype=torch.float32)
        self.dir = (d / d.norm())
        self.strength = 0.0
        self.h = None

    def __enter__(self):
        def fn(hs):
            if self.strength == 0.0:
                return None
            return hs + (self.strength * self.dir.to(hs.device, hs.dtype))
        self.h = self.be.residual_hook(self.m, self.layer, fn)
        return self

    def __exit__(self, *a):
        if self.h:
            self.h.remove()


# -------------------------------------------------------------------- the ops

def op_read(be, m, dev, layer, contrast_a, contrast_b) -> OpResult:
    """Fit a probe on a contrastive feature: can this substrate's stream be read?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    A = _resid(be, m, contrast_a, layer, dev)
    B = _resid(be, m, contrast_b, layer, dev)
    X = np.concatenate([A, B])
    y = np.array([1] * len(A) + [0] * len(B))
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), X, y,
                                cv=LeaveOneOut()).mean())
    return OpResult("read", True, {"probe_acc": round(acc, 4),
                                   "layer": layer, "n": int(len(y))})


def op_amplify(be, m, dev, layer, contrast_a, contrast_b, neutral,
               side_effect_budget: float) -> tuple[OpResult, np.ndarray]:
    """Dose-calibrated steering: sweep strength, pick the smallest that has an effect
    while staying inside the side-effect budget (prayoga's calibration, generalized)."""
    A = _resid(be, m, contrast_a, layer, dev)
    B = _resid(be, m, contrast_b, layer, dev)
    direction = A.mean(0) - B.mean(0)
    dnorm = float(np.linalg.norm(direction))

    base_target = _loss(be, m, contrast_a, dev)
    base_neutral = _loss(be, m, neutral, dev)
    curve, chosen = [], None
    with _Steer(be, m, direction, layer) as s:
        for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
            s.strength = mult * dnorm / 4.0
            eff = base_target - _loss(be, m, contrast_a, dev)
            side = _loss(be, m, neutral, dev) - base_neutral
            curve.append({"strength": round(float(s.strength), 4),
                          "effect": round(eff, 4), "side_effect": round(side, 4)})
            if chosen is None and eff > 0 and side < side_effect_budget:
                chosen = curve[-1]
    emax = max(p["effect"] for p in curve)
    ec50 = next((p["strength"] for p in curve if p["effect"] >= emax / 2), None)
    ok = chosen is not None
    return (OpResult(
        "amplify", ok,
        {"dose_curve": curve, "ec50": ec50, "max_effect": round(emax, 4),
         "chosen": chosen, "side_effect_budget": side_effect_budget},
        "" if ok else (
            f"No steering strength produced an effect while keeping the side-effect "
            f"under {side_effect_budget}. The feature is present but not usable as a "
            "control on this substrate at this layer.")),
        direction)


def op_install(be, m, dev, circuit_name: str, host_vocab: int, seq_len: int) -> OpResult:
    """Link a separately compiled circuit — refused if outside its verified envelope."""
    spec = require_circuit(circuit_name)
    env = spec.envelope
    problems = []
    if host_vocab > env.get("vocab_max", 10**9):
        problems.append(
            f"host vocabulary {host_vocab:,} exceeds the {env['vocab_max']} this "
            "circuit was verified for")
    if seq_len > env.get("len_max", 10**9):
        problems.append(
            f"sequence length {seq_len} exceeds the {env['len_max']} it was verified for")
    if problems:
        return OpResult(
            "install", False, {"circuit": circuit_name, "envelope": env},
            "Circuit not installed: " + "; ".join(problems) +
            ". The circuit is verified, but not for this substrate — re-verify it at "
            "this scale before linking, or compile a circuit for this target.")
    return OpResult("install", True, {"circuit": circuit_name, "envelope": env,
                                      "gates": spec.gates})


# ---------------------------------------------------------------------- driver

def run_program(target_spec: dict, data: dict, dev: str = "cuda",
                layer: int = -2, side_effect_budget: float = 0.15,
                train_batches=None, train_steps: int = 400) -> RunReport:
    """Execute hardened.loom's operations against one substrate.

    `train_batches` is supplied only for a substrate built from scratch: a freshly
    initialized model has no features to read, so the program's monitor would fail at
    chance. Training is how the compiler realizes a declared feature on that target —
    a lowering step, not part of the program.
    """
    t0 = time.time()
    be = for_target(target_spec)
    m = be.realize(target_spec)
    m.to(dev)

    if train_batches:
        opt = torch.optim.AdamW(m.module.parameters(), lr=3e-4)
        m.module.train()
        for step in range(train_steps):
            b = train_batches[step % len(train_batches)].to(dev)
            loss = be.forward_loss(m, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.module.parameters(), 1.0); opt.step()
        m.module.eval()
    rep = RunReport(
        target=target_spec.get("name", target_spec.get("kind", "?")),
        backend=be.name,
        substrate={"params": m.n_params, "layers": m.n_layers, "width": m.width,
                   "vocab": m.vocab,
                   "family": m.meta.get("family", target_spec.get("kind")),
                   "pattern": m.meta.get("pattern")})

    r_read = op_read(be, m, dev, layer, data["a"], data["b"])
    rep.ops.append(r_read)
    rep.gates.append({"gate": "probe_acc > 0.7",
                      "measured": r_read.metrics["probe_acc"],
                      "passed": r_read.metrics["probe_acc"] > 0.7})

    r_amp, _ = op_amplify(be, m, dev, layer, data["a"], data["b"], data["neutral"],
                          side_effect_budget)
    rep.ops.append(r_amp)
    eff = r_amp.metrics["max_effect"]
    rep.gates.append({"gate": "steer_effect > 0.0", "measured": eff,
                      "passed": bool(eff > 0.0)})

    r_inst = op_install(be, m, dev, "induction", m.vocab, data["seq_len"])
    rep.ops.append(r_inst)
    rep.gates.append({"gate": "circuit_gate(induction)", "measured": r_inst.ok,
                      "passed": r_inst.ok, "note": r_inst.diagnosis})

    rep.passed = all(g["passed"] for g in rep.gates)
    rep.wall_clock_s = time.time() - t0
    return rep
