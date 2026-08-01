"""The backend seam: stages must work on any substrate through one interface."""
import pytest
import torch

from loom.backends import REGISTRY, Backend, HFBackend, ScratchBackend, for_target

SPEC = {"kind": "decoder", "layers": 2, "width": 64, "heads": 2, "vocab": 128, "ctx": 64}


def test_backends_satisfy_the_protocol():
    for b in REGISTRY.values():
        assert isinstance(b, Backend)


def test_target_kind_selects_the_backend():
    """The program names a target, never a backend — the compiler infers it."""
    assert for_target({"kind": "decoder"}).name == "scratch"
    assert for_target({"kind": "gptbert"}).name == "scratch"
    assert for_target({"kind": "load", "name": "gpt2"}).name == "hf"


def test_scratch_realize_reports_shape():
    m = ScratchBackend().realize(SPEC)
    assert m.n_layers == 2 and m.width == 64 and m.vocab == 128
    assert m.n_params > 0 and m.backend == "scratch"


def test_scratch_blocks_are_the_surgery_unit():
    b = ScratchBackend()
    m = b.realize(SPEC)
    assert len(b.blocks(m)) == 2


def test_scratch_forward_loss_and_logits():
    b = ScratchBackend()
    m = b.realize(SPEC)
    toks = torch.randint(0, 128, (3, 16))
    loss = b.forward_loss(m, toks)
    assert loss.ndim == 0 and loss.item() > 0
    assert b.logits(m, toks).shape == (3, 16, 128)


def test_residual_hook_can_read_and_write():
    """Monitors, controls and the linker all ride on this one capability."""
    b = ScratchBackend()
    m = b.realize(SPEC)
    toks = torch.randint(0, 128, (2, 8))
    seen = {}

    h = b.residual_hook(m, 0, lambda hs: seen.setdefault("shape", tuple(hs.shape)))
    b.logits(m, toks)
    h.remove()
    assert seen["shape"] == (2, 8, 64)

    base = b.logits(m, toks).clone()
    h = b.residual_hook(m, 0, lambda hs: hs * 0)
    changed = b.logits(m, toks)
    h.remove()
    assert not torch.allclose(base, changed), "hook must be able to WRITE, not just read"


def test_scratch_save_roundtrip(tmp_path):
    b = ScratchBackend()
    m = b.realize(SPEC)
    p = tmp_path / "m.pt"
    b.save(m, str(p))
    assert p.exists() and "state_dict" in torch.load(p, weights_only=True)


def test_hf_block_layouts_cover_the_named_families():
    """The only place architecture layout may be known is this table — and it must
    cover the families the project targets."""
    paths = {p[-1] for p in HFBackend._BLOCK_PATHS}
    assert "layers" in paths      # llama, qwen, gemma, mistral, deepseek
    assert "h" in paths           # gpt2 family


def test_hf_unknown_layout_fails_loudly():
    """A model whose blocks cannot be found must raise, not silently return nothing."""
    class Odd(torch.nn.Module):
        pass

    from loom.backends import ModelHandle
    m = ModelHandle(module=Odd(), name="odd/model", backend="hf",
                    n_params=0, n_layers=0, width=0, vocab=0)
    with pytest.raises(NotImplementedError, match="Cannot locate transformer blocks"):
        HFBackend().blocks(m)
