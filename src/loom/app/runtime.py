"""The compiled artifact, and how to run it.

A build produces a directory, not just a file of numbers. It holds the weights, the hooks
that implement the app's behaviour, the plan that produced them, and the measurements that
were taken. `loom run` loads that directory and gives you a prompt.

The hooks matter as much as the weights: a style, a prohibition and a guardrail are
realized as steering writes on the residual stream, so they live in the artifact as
directions plus the strengths the compiler calibrated. Loading the artifact reinstalls
them, which is why the model you talk to behaves like the program said it would rather
than like the base model it started from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class InstalledControl:
    """A steering write the compiler calibrated and the artifact carries."""
    name: str
    kind: str                 # style | prohibition | guardrail | invariant
    layer: int
    strength: float
    direction: list[float] = field(default_factory=list)
    side_effect: float | None = None
    joint_scale: float | None = None   # how far the joint pass backed this control off

    def tensor(self, device, dtype):
        d = torch.tensor(self.direction, dtype=torch.float32, device=device)
        return (d / d.norm()).to(dtype)


class LoomModel:
    """A built application: a model with its compiled behaviour attached."""

    def __init__(self, module, tokenizer, controls: list[InstalledControl],
                 plan: dict, report: dict, device: str = "cuda"):
        self.module = module
        self.tokenizer = tokenizer
        self.controls = controls
        self.plan = plan
        self.report = report
        self.device = device
        self._handles: list = []
        self._install()

    def _blocks(self):
        for path in (("model", "layers"), ("transformer", "h"), ("blocks",)):
            obj = self.module
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError("cannot locate this model's blocks to attach controls")

    def _install(self) -> None:
        blocks = self._blocks()
        for c in self.controls:
            idx = c.layer if c.layer >= 0 else len(blocks) + c.layer

            def hook(mod, args, output, _c=c):
                hs = output[0] if isinstance(output, tuple) else output
                v = _c.tensor(hs.device, hs.dtype) * _c.strength
                new = hs + v
                return (new,) + output[1:] if isinstance(output, tuple) else new

            self._handles.append(blocks[idx].register_forward_hook(hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def respond(self, prompt: str, max_new_tokens: int = 60) -> str:
        if self.tokenizer is None:
            return "(this build has no tokenizer; it is a token-level model)"
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.module.generate(**ids, max_new_tokens=max_new_tokens,
                                   do_sample=True, temperature=0.7,
                                   pad_token_id=getattr(self.tokenizer, "eos_token_id",
                                                        None))
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() or "(no continuation)"

    def describe(self) -> str:
        caps = self.plan.get("capabilities", [])
        lines = [f"app: {self.plan.get('app')}  on {self.plan.get('target')}",
                 f"built by: {self.plan.get('substrate')}"]
        for c in caps:
            lines.append(f"  {c['capability']:<44} via {c['strategy']}")
        if self.controls:
            lines.append("active controls:")
            for c in self.controls:
                lines.append(f"  {c.name} ({c.kind}) at layer {c.layer}, "
                             f"strength {c.strength:.3g}")
        return "\n".join(lines)


def load_artifact(path: str | Path, device: str = "cuda") -> LoomModel:
    """Load a built application. Everything needed is in the directory."""
    root = Path(path)
    plan = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else {}
    report = json.loads((root / "report.json").read_text()) if (root / "report.json").exists() else {}

    controls = [InstalledControl(**c) for c in report.get("controls", [])]

    weights = root / "model"
    if weights.exists():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        module = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16)
        module.to(device).eval()
        try:
            tok = AutoTokenizer.from_pretrained(weights)
        except Exception:
            tok = None
        return LoomModel(module, tok, controls, plan, report, device)

    base = report.get("base_model")
    if base:
        # An open-weight build may carry only its controls: the base is downloaded and
        # the compiled behaviour is reattached on load. That is the artifact being small
        # because most of it is already on your disk, not because it is incomplete.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        module = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        module.to(device).eval()
        tok = AutoTokenizer.from_pretrained(base)
        return LoomModel(module, tok, controls, plan, report, device)

    raise FileNotFoundError(
        f"{root} has no model/ directory and its report names no base model. "
        "This artifact was planned but never built — run `loom build` without --dry-run.")
