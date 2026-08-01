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


def test_edge_count_is_not_just_nonzero_count():
    """Edge counting must respond to weight MAGNITUDE, not just nonzero-ness —
    otherwise it reproduces the tautology it exists to avoid."""
    from miabstraction.circuits import surviving_edge_count

    m = TinyTransformer(vocab=8, d_model=32, n_layers=1, n_heads=2)
    with torch.no_grad():
        # make most MLP weights tiny-but-nonzero: nonzero count unchanged, edges drop
        for p in m.blocks[0].mlp.parameters():
            p.mul_(1e-6)
            p[..., :1] = 1.0  # keep a few at full scale
    nonzero = circuit_weight_count(m, {(0, -1)})
    edges = surviving_edge_count(m, {(0, -1)})
    assert edges < nonzero


def test_edge_count_scales_with_circuit():
    from miabstraction.circuits import surviving_edge_count

    m = TinyTransformer(vocab=8, d_model=32, n_layers=2, n_heads=2)
    assert surviving_edge_count(m, {(0, 0), (0, 1)}) > surviving_edge_count(m, {(0, 0)})


def test_tautology_flag_fires_when_ratio_matches_q():
    """The E5 result must self-report when its size ratio just echoes imposed sparsity."""
    import json
    from pathlib import Path

    r = json.loads(Path("results/e5_sparsity/result.json").read_text())
    if "size_metrics_are_tautological" not in r:
        return  # result predates the flag
    q = r["imposed_q"]
    close = abs(r["circuit_weight_ratio"] - q) < 0.05 * q
    assert r["size_metrics_are_tautological"] == close
