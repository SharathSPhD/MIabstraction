import torch

from miabstraction.models import TinyTransformer, train_lm


def test_forward_shapes():
    m = TinyTransformer(vocab=3, d_model=32, n_layers=2, n_heads=2)
    x = torch.randint(0, 3, (4, 10))
    logits, resid = m(x, collect=True)
    assert logits.shape == (4, 10, 3)
    assert len(resid) == 2
    assert resid[0].shape == (4, 10, 32)


def test_causal_masking():
    """Future tokens must not influence past logits."""
    m = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=2)
    m.eval()
    x1 = torch.randint(0, 5, (1, 12))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 5
    with torch.no_grad():
        l1, l2 = m(x1), m(x2)
    assert torch.allclose(l1[0, :-1], l2[0, :-1], atol=1e-5)
    assert not torch.allclose(l1[0, -1], l2[0, -1], atol=1e-5)


def test_training_reduces_loss():
    torch.manual_seed(0)
    # learnable structure: token t+1 = token t (copy task)
    seq = torch.randint(0, 4, (256, 1)).repeat(1, 16)
    m = TinyTransformer(vocab=4, d_model=32, n_layers=1, n_heads=2)
    losses = train_lm(m, seq, steps=200, batch_size=64, lr=1e-3, device="cpu")
    assert losses[-1] < losses[0] * 0.5
