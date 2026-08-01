"""E6 — the SAE control experiment where the control has its intended meaning.

On the toy model, H3 refuted itself in an unexpected direction: a random network's
low-dimensional activations were EASIER to reconstruct than a trained one's. The Heap
et al. random-transformer control was designed for full-scale pretrained models, so the
replication has to run on one. These tests cover the pure parts; the GPU run is driven
by configs/e6_real_lm_sae.yaml.
"""
import numpy as np
import pytest

from miabstraction.experiments.e6_real_lm_sae import chunk_tokens, separation


def test_chunking_is_exact_and_drops_the_remainder():
    ids = list(range(103))
    chunks = chunk_tokens(ids, seq_len=10, n_seq=8)
    assert chunks.shape == (8, 10)
    np.testing.assert_array_equal(chunks[0], np.arange(10))
    np.testing.assert_array_equal(chunks[7], np.arange(70, 80))


def test_chunking_refuses_to_invent_data():
    """Fewer tokens than requested sequences must be an error, not a silent repeat —
    fallbacks that substitute invented data are how a pipeline trains on noise."""
    with pytest.raises(ValueError, match="only"):
        chunk_tokens(list(range(50)), seq_len=10, n_seq=8)


def test_separation_reports_direction_and_spread():
    trained = [{"fvu": 0.30, "l0": 32.0, "dead_frac": 0.1},
               {"fvu": 0.34, "l0": 32.0, "dead_frac": 0.1}]
    control = [{"fvu": 0.10, "l0": 32.0, "dead_frac": 0.2},
               {"fvu": 0.12, "l0": 32.0, "dead_frac": 0.2}]
    s = separation(trained, control)
    assert s["fvu"]["trained_mean"] == pytest.approx(0.32)
    assert s["fvu"]["control_mean"] == pytest.approx(0.11)
    # The sign convention is spelled out so a reader cannot mistake which way the
    # separation ran — that mistake is exactly how H3 was almost mis-scored.
    assert s["fvu"]["direction"] == "control_easier_to_reconstruct"
    assert s["fvu"]["trained_std"] > 0


def test_separation_with_zero_variance_is_flagged():
    """A metric that reproduces perfectly may be reproducing a constant."""
    rows = [{"fvu": 0.2, "l0": 32.0, "dead_frac": 0.0}] * 3
    s = separation(rows, rows)
    assert s["fvu"]["zero_variance_flag"] is True
