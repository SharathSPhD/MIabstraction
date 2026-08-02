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
        # The model's own conversation format, when it has one. A chat-tuned model
        # handed a bare string may simply emit EOS — Gemma verified as "(no
        # continuation)" three times on a guardrail its gate had measurably trained —
        # and a user meets the artifact through chat, so verification must too.
        from .steering_ops import _as_chat
        text_in = _as_chat(self.tokenizer, "", prompt)
        ids = self.tokenizer(text_in, return_tensors="pt").to(self.device)
        n_in = ids["input_ids"].shape[1]
        out = self.module.generate(**ids, max_new_tokens=max_new_tokens,
                                   do_sample=True, temperature=0.7,
                                   pad_token_id=getattr(self.tokenizer, "eos_token_id",
                                                        None))
        text = self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        return text.strip() or "(no continuation)"

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


def _reapply_adapter(module, path: Path, base: str) -> bool:
    """Fold the build's adapter back into the freshly downloaded weights.

    Without this, loading an artifact gives you the base model the build started from,
    while the report next to it says the knowledge capability passed. The measurement
    would be true of a model that existed only inside the build process, which is the
    same failure as reporting a number nobody can reproduce.
    """
    if not path.exists():
        return False
    # weights_only: an artifact is a thing you hand to someone else, so loading one must
    # not be able to run code. The blob is tensors, floats and strings by construction.
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("base_model") and blob["base_model"] != base:
        raise ValueError(
            f"{path} holds an adapter trained on {blob['base_model']}, but this artifact "
            f"names {base}. Refusing to graft one model's adaptation onto another.")
    by_name = dict(module.named_modules())
    for layer_name, t in blob.get("adapters", {}).items():
        mod = by_name.get(layer_name)
        if mod is None or not hasattr(mod, "weight"):
            raise ValueError(
                f"{path} adapts {layer_name!r}, which {base} does not have. The artifact "
                "and the base model have drifted apart.")
        a, b, scale = t["a"], t["b"], t["scale"]
        w = mod.weight
        # nn.Linear stores (out, in); transformers' Conv1D stores (in, out). For a square
        # projection the shapes cannot tell you which, so the build records the layout
        # and this reads it rather than guessing.
        if t.get("layout") == "LoRAConv1D":
            delta = scale * torch.matmul(a, b)
        else:
            delta = scale * torch.matmul(b.T, a.T)
        if delta.shape != w.shape:
            raise ValueError(
                f"adapter for {layer_name!r} produces a {tuple(delta.shape)} delta for a "
                f"{tuple(w.shape)} weight; the artifact does not match this model.")
        with torch.no_grad():
            w.data.add_(delta.to(device=w.device, dtype=w.dtype))
    return True


class _BPETokenizerAdapter:
    """Give a `tokenizers.Tokenizer` the small slice of the HF interface the runtime
    uses, so one LoomModel serves both substrates."""

    def __init__(self, tk):
        self.tk = tk
        v = tk.get_vocab()
        self.eos_token_id = v.get("<EOS>", 0)
        self.eos_token = "<EOS>"
        self.chat_template = None          # a base LM has no conversation format

    def __call__(self, text, return_tensors=None, **kw):
        ids = self.tk.encode(text).ids
        if return_tensors == "pt":
            import torch as _t
            return {"input_ids": _t.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)


class _ScratchLM(LoomModel):
    """A from-scratch artifact. It completes text rather than answering turns,
    because that is what it is — and saying so is better than dressing a base
    language model up as a chat model."""

    @torch.no_grad()
    def respond(self, prompt: str, max_new_tokens: int = 60) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        out = ids
        for _ in range(max_new_tokens):
            logits = self.module(out[:, -self.max_len:])
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if int(nxt) == getattr(self.tokenizer, "eos_token_id", -1):
                break
        text = self.tokenizer.decode(out[0].tolist())
        return text[len(prompt):].strip() or text.strip() or "(no continuation)"


def _load_scratch(weights: Path, controls, plan, report, device: str) -> LoomModel:
    from miabstraction.models import TinyTransformer
    blob = torch.load(weights / "weights.pt", map_location="cpu", weights_only=True)
    a = blob["arch"]
    module = TinyTransformer(
        vocab=int(a.get("vocab", blob.get("vocab_size", 16000))),
        d_model=int(a.get("width", 384)), n_layers=int(a.get("layers", 6)),
        n_heads=int(a.get("heads", 6)), max_len=int(a.get("ctx", 512)))
    module.load_state_dict(blob["state_dict"])
    module.to(device).eval()

    tok = None
    tj = weights / "tokenizer.json"
    if tj.exists():
        from tokenizers import Tokenizer
        tok = _BPETokenizerAdapter(Tokenizer.from_file(str(tj)))

    lm = _ScratchLM(module, tok, controls, plan, report, device)
    lm.max_len = int(a.get("ctx", 512))
    return lm


def load_artifact(path: str | Path, device: str = "cuda") -> LoomModel:
    """Load a built application. Everything needed is in the directory."""
    root = Path(path)
    plan = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else {}
    report = json.loads((root / "report.json").read_text()) if (root / "report.json").exists() else {}

    controls = [InstalledControl(**c) for c in report.get("controls", [])]

    weights = root / "model"

    # A model this compiler made from nothing. There is no upstream repository to
    # fetch it from, so the artifact carries the weights themselves — and loading it
    # is how "the app lets you work with the model you built" stops being a promise.
    if (weights / "weights.pt").exists():
        return _load_scratch(weights, controls, plan, report, device)

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
        # An open-weight build carries its adapter and its controls, not a copy of the
        # weights: the base is downloaded and the compiled behaviour is put back on load.
        # That is the artifact being small because most of it is already on your disk,
        # not because it is incomplete.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        module = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        module.to(device).eval()
        tok = AutoTokenizer.from_pretrained(base)
        # Every adapter the build saved, not just the knowledge one: an escalated
        # guardrail lives in adapter_guardrail.pt, and an artifact that reapplied only
        # adapter.pt would verify as refusing in the report while the loaded model
        # refuses nothing.
        applied = [p.name for p in sorted(root.glob("adapter*.pt"))
                   if _reapply_adapter(module, p, base)]
        report = {**report, "adapters_reapplied": applied}
        return LoomModel(module, tok, controls, plan, report, device)

    raise FileNotFoundError(
        f"{root} has no model/ directory and its report names no base model. "
        "This artifact was planned but never built — run `loom build` without --dry-run.")
