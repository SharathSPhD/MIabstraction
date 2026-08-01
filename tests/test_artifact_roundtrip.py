"""A build's artifact must reproduce the model the build verified.

The failure this guards against reported itself as a success: the compiler merged its
adapter into weights that existed only inside that process, wrote a report saying the
knowledge capability passed, and left `loom run` loading the untouched base model. The
number was true of a model nobody could load, which is the same thing as a number nobody
can reproduce.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from loom.app.build_open import continued_pretraining, _base_fingerprint
from loom.app.runtime import _reapply_adapter

CORPUS = "data/domains/medical/corpus.txt"


def _gpt2():
    return AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)


def test_the_saved_adapter_reproduces_the_built_model(tmp_path):
    tok = AutoTokenizer.from_pretrained("gpt2")
    built = _gpt2()
    out = continued_pretraining(built, tok, CORPUS, "cpu", steps=3, lr=5e-5,
                                merge=True, save_adapter_to=str(tmp_path / "adapter.pt"),
                                base_name="gpt2")
    assert out["ran"], out.get("reason")
    assert out["adapter_saved_to"]
    assert out["base_weights_unchanged"], "training moved the frozen base"

    # A fresh copy of the base, with only the artifact to go on.
    reloaded = _gpt2()
    assert _base_fingerprint(reloaded) != _base_fingerprint(built), (
        "the build did not change the model at all, so this proves nothing")
    assert _reapply_adapter(reloaded, tmp_path / "adapter.pt", "gpt2")
    assert _base_fingerprint(reloaded) == _base_fingerprint(built)


def test_an_adapter_from_another_model_is_refused(tmp_path):
    tok = AutoTokenizer.from_pretrained("gpt2")
    m = _gpt2()
    continued_pretraining(m, tok, CORPUS, "cpu", steps=2, lr=5e-5, merge=True,
                          save_adapter_to=str(tmp_path / "adapter.pt"),
                          base_name="gpt2")
    try:
        _reapply_adapter(_gpt2(), tmp_path / "adapter.pt", "meta-llama/Llama-3.2-1B")
    except ValueError as e:
        assert "Refusing" in str(e)
    else:
        raise AssertionError("grafted one model's adaptation onto another")


def test_a_missing_adapter_is_not_silently_a_success(tmp_path):
    assert _reapply_adapter(_gpt2(), tmp_path / "nope.pt", "gpt2") is False


def test_loading_an_artifact_cannot_execute_code(tmp_path):
    """An artifact is a thing you hand to someone else."""
    import inspect
    from loom.app import runtime
    src = inspect.getsource(runtime._reapply_adapter)
    assert "weights_only=True" in src


def test_square_projections_are_not_transposed(tmp_path):
    """Where the bug hid. nn.Linear stores (out, in) and Conv1D stores (in, out); on a
    square projection the shapes cannot tell you which, so a reader guessing from shape
    transposes the delta on exactly the layers where nothing looks wrong. gpt2's
    attention c_proj is 768x768."""
    import torch as _t
    from loom.app.lora import attach_lora, merge_or_detach

    m = _gpt2()
    square = [n for n, mod in m.named_modules()
              if hasattr(mod, "weight") and getattr(mod.weight, "ndim", 0) == 2
              and mod.weight.shape[0] == mod.weight.shape[1]]
    assert square, "this model has no square projection, so the test proves nothing"

    handles = attach_lora(m, rank=2, alpha=4.0)
    for h in handles:                       # make every adapter contribute
        _t.nn.init.normal_(h.adapter_b, std=0.01)
    blob = {"base_model": "gpt2", "rank": 2, "alpha": 4.0,
            "adapters": {h.layer_name: {"a": h.adapter_a.detach().cpu(),
                                        "b": h.adapter_b.detach().cpu(),
                                        "scale": h.scale,
                                        "layout": type(h.module).__name__}
                         for h in handles}}
    _t.save(blob, tmp_path / "adapter.pt")
    merge_or_detach(m, handles, mode="merge")

    fresh = _gpt2()
    _reapply_adapter(fresh, tmp_path / "adapter.pt", "gpt2")
    by_name = dict(fresh.named_modules())
    for name in square:
        a = by_name[name].weight
        b = dict(m.named_modules())[name].weight
        assert _t.allclose(a, b, atol=1e-6), f"{name} was reapplied transposed"


def test_merging_leaves_the_model_structurally_unchanged():
    """A merged model has to be what it was before, or save_pretrained and every
    consumer that walks named_modules quietly sees `...c_attn.base.weight`."""
    from loom.app.lora import attach_lora, merge_or_detach
    m = _gpt2()
    before = {n for n, _ in m.named_parameters()}
    handles = attach_lora(m, rank=2, alpha=4.0)
    assert {n for n, _ in m.named_parameters()} != before
    merge_or_detach(m, handles, mode="merge")
    assert {n for n, _ in m.named_parameters()} == before
