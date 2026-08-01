"""Backend interface — the seam that makes a Loom program architecture-agnostic.

A Loom program names a `target`. That target may be an architecture the compiler builds
from scratch, or an existing open-weight model someone else trained. Every stage above
this module — pretrain, finetune, align, graft, merge, quantize — is written against the
interface here and never against a particular transformer. That is what lets one line
change the substrate:

    target arch = decoder(layers=12, width=768, heads=12);   // ScratchBackend
    target arch = load("Qwen/Qwen3-0.6B");                   // HFBackend

The interface is deliberately small. Anything a stage needs that is not here is a sign the
stage has grown a dependency on a specific architecture, which is the thing this design
exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn


@dataclass
class ModelHandle:
    """A model plus what a stage needs to know about it, without knowing its class."""
    module: nn.Module
    name: str
    backend: str
    n_params: int
    n_layers: int
    width: int
    vocab: int
    tokenizer: Any = None
    meta: dict = field(default_factory=dict)

    def to(self, device: str) -> "ModelHandle":
        self.module.to(device)
        return self


@runtime_checkable
class Backend(Protocol):
    """What every substrate must provide. Nothing above this may assume more."""

    name: str

    def realize(self, spec: dict) -> ModelHandle:
        """Build from an architecture spec, or load existing weights."""
        ...

    def blocks(self, m: ModelHandle) -> list[nn.Module]:
        """The transformer blocks in order — the unit of structural surgery."""
        ...

    def residual_hook(self, m: ModelHandle, layer: int,
                      fn: Callable) -> Any:
        """Attach a read/write hook to the residual stream after `layer`.

        Returns a handle with .remove(). This is what monitors, controls, and the
        linker all use; implementing it is what makes those work on any substrate.
        """
        ...

    def forward_loss(self, m: ModelHandle, batch: torch.Tensor,
                     labels: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token loss for a batch. Stages train through this, not through
        architecture-specific forward signatures."""
        ...

    def logits(self, m: ModelHandle, batch: torch.Tensor) -> torch.Tensor:
        ...

    def save(self, m: ModelHandle, path: str) -> None:
        ...


# --------------------------------------------------------------------------- scratch

class ScratchBackend:
    """Substrate built from an architecture spec — the from-scratch path."""

    name = "scratch"

    def realize(self, spec: dict) -> ModelHandle:
        from miabstraction.models import TinyTransformer
        d = int(spec.get("width", 384))
        layers = int(spec.get("layers", 6))
        heads = int(spec.get("heads", 6))
        vocab = int(spec.get("vocab", 16000))
        ctx = int(spec.get("ctx", 512))
        mod = TinyTransformer(vocab=vocab, d_model=d, n_layers=layers,
                              n_heads=heads, max_len=ctx)
        return ModelHandle(
            module=mod, name=spec.get("name", "scratch-decoder"), backend=self.name,
            n_params=sum(p.numel() for p in mod.parameters()),
            n_layers=layers, width=d, vocab=vocab,
            meta={"spec": spec})

    def blocks(self, m: ModelHandle) -> list[nn.Module]:
        return list(m.module.blocks)

    def residual_hook(self, m: ModelHandle, layer: int, fn: Callable):
        blocks = self.blocks(m)
        idx = layer if layer >= 0 else len(blocks) + layer

        def wrapper(mod, args, output):
            # Only a Tensor return replaces the block's output. A read-only callback
            # returns None (or anything else) and must not be able to corrupt the
            # forward pass by accident — both backends behave identically here.
            new = fn(output)
            return new if isinstance(new, torch.Tensor) else output

        return blocks[idx].register_forward_hook(wrapper)

    def forward_loss(self, m: ModelHandle, batch, labels=None) -> torch.Tensor:
        import torch.nn.functional as F
        x = batch[:, :-1] if labels is None else batch
        y = batch[:, 1:] if labels is None else labels
        logits = m.module(x)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
            ignore_index=-100)

    def logits(self, m: ModelHandle, batch) -> torch.Tensor:
        return m.module(batch)

    def save(self, m: ModelHandle, path: str) -> None:
        torch.save({"state_dict": m.module.state_dict(), "meta": m.meta}, path)


# -------------------------------------------------------------------------------- hf

class HFBackend:
    """Substrate loaded from the open-weight ecosystem.

    Block access is resolved by probing the common container names rather than
    hard-coding one family's layout, so Llama, Gemma, Qwen, GPT-2 and friends work
    through the same code path.
    """

    name = "hf"
    _BLOCK_PATHS = (
        ("model", "layers"),        # llama, qwen, gemma, mistral, deepseek
        ("transformer", "h"),       # gpt2, gptj
        ("model", "decoder", "layers"),
        ("gpt_neox", "layers"),
    )

    def realize(self, spec: dict) -> ModelHandle:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = spec["name"]
        dtype = getattr(torch, spec.get("dtype", "bfloat16"))
        mod = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        tok = AutoTokenizer.from_pretrained(name)
        cfg = mod.config
        return ModelHandle(
            module=mod, name=name, backend=self.name,
            n_params=sum(p.numel() for p in mod.parameters()),
            n_layers=getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)),
            width=getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)),
            vocab=getattr(cfg, "vocab_size", 0),
            tokenizer=tok, meta={"config": cfg.to_dict() if hasattr(cfg, "to_dict") else {}})

    def blocks(self, m: ModelHandle) -> list[nn.Module]:
        for path in self._BLOCK_PATHS:
            obj = m.module
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue
        raise NotImplementedError(
            f"Cannot locate transformer blocks in {m.name} "
            f"({type(m.module).__name__}). Add its layout to HFBackend._BLOCK_PATHS — "
            "this is the only place architecture layout is allowed to be known.")

    def residual_hook(self, m: ModelHandle, layer: int, fn: Callable):
        blocks = self.blocks(m)
        idx = layer if layer >= 0 else len(blocks) + layer

        def wrapper(mod, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            new = fn(hs)
            if not isinstance(new, torch.Tensor):
                return output
            return (new,) + output[1:] if isinstance(output, tuple) else new

        return blocks[idx].register_forward_hook(wrapper)

    def forward_loss(self, m: ModelHandle, batch, labels=None) -> torch.Tensor:
        out = m.module(input_ids=batch,
                       labels=batch if labels is None else labels)
        return out.loss

    def logits(self, m: ModelHandle, batch) -> torch.Tensor:
        return m.module(input_ids=batch).logits

    def save(self, m: ModelHandle, path: str) -> None:
        m.module.save_pretrained(path)
        if m.tokenizer is not None:
            m.tokenizer.save_pretrained(path)


REGISTRY: dict[str, Backend] = {"scratch": ScratchBackend(), "hf": HFBackend()}


def for_target(spec: dict) -> Backend:
    """Pick the backend a `target` declaration implies.

    `load("org/model")` -> hf; `decoder(...)`/`gptbert(...)` -> scratch. The program
    does not name a backend; the compiler infers it from the target, which is why the
    same source compiles to either substrate.
    """
    kind = spec.get("kind", "decoder")
    return REGISTRY["hf" if kind == "load" else "scratch"]
