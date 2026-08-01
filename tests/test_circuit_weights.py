import torch

from miabstraction.circuits import circuit_weight_count
from miabstraction.models import TinyTransformer


def test_weight_count_scales_with_nodes():
    m = TinyTransformer(vocab=8, d_model=32, n_layers=2, n_heads=2)
    one = circuit_weight_count(m, {(0, 0)})
    two = circuit_weight_count(m, {(0, 0), (0, 1)})
    assert one > 0
    assert two == 2 * one  # heads in a layer share the packed projection equally


def test_mlp_node_counted():
    m = TinyTransformer(vocab=8, d_model=32, n_layers=2, n_heads=2)
    head_only = circuit_weight_count(m, {(0, 0)})
    with_mlp = circuit_weight_count(m, {(0, 0), (0, -1)})
    assert with_mlp > head_only


def test_zeroed_weights_are_not_counted():
    m = TinyTransformer(vocab=8, d_model=32, n_layers=1, n_heads=2)
    before = circuit_weight_count(m, {(0, 0), (0, -1)})
    with torch.no_grad():
        for p in m.blocks[0].mlp.parameters():
            p.zero_()
    after = circuit_weight_count(m, {(0, 0), (0, -1)})
    assert after < before


def test_attn_only_model_has_no_mlp_weights():
    m = TinyTransformer(vocab=8, d_model=32, n_layers=1, n_heads=2, attn_only=True)
    assert circuit_weight_count(m, {(0, -1)}) == 0
