"""Test the constructed+polished induction head backend."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from loom.construct_then_polish import compile_induction
from miabstraction.experiments.e2_induction import (
    copy_region_losses,
    gapped_doubled_sequences,
    prefix_matching_score,
)


@pytest.mark.timeout(300)
def test_constructed_polish_gates(tmp_path):
    """Constructed + polished model passes induction gates."""
    print("\n=== PHASE 1: PURE CONSTRUCTION ===")

    # Test pure construction first
    model_constructed = compile_induction(vocab=10, d_model=256, max_len=48, polish_steps=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_constructed.to(device)

    rng = np.random.default_rng(42)
    seqs, gaps = gapped_doubled_sequences(32, 12, 6, 10, rng)
    tokens = torch.from_numpy(seqs).to(device)

    score_pre = prefix_matching_score(model_constructed, tokens, gaps, copy_len=12)
    loss_first_pre, loss_second_pre = copy_region_losses(model_constructed, tokens, gaps, copy_len=12)

    print(f"  Prefix score (pre-polish): {score_pre:.4f}")
    print(f"  ICL loss (pre-polish): {loss_second_pre:.4f}")

    print("\n=== PHASE 2: POLISH (200 steps) ===")

    # Now with polish (use more steps if needed to reach gates)
    model_polished = compile_induction(vocab=10, d_model=256, max_len=48, polish_steps=500, polish_lr=0.01)
    model_polished.to(device)

    score_post = prefix_matching_score(model_polished, tokens, gaps, copy_len=12)
    loss_first_post, loss_second_post = copy_region_losses(model_polished, tokens, gaps, copy_len=12)

    print(f"  Prefix score (post-polish): {score_post:.4f}")
    print(f"  ICL loss (post-polish): {loss_second_post:.4f}")

    results = {
        "construction": {
            "prefix_score": float(score_pre),
            "icl_loss": float(loss_second_pre),
            "first_copy_loss": float(loss_first_pre),
            "passes_prefix_gate_gt_0_5": score_pre > 0.5,
            "passes_icl_gate_lt_0_5": loss_second_pre < 0.5,
        },
        "after_polish": {
            "prefix_score": float(score_post),
            "icl_loss": float(loss_second_post),
            "first_copy_loss": float(loss_first_post),
            "passes_prefix_gate_gt_0_5": score_post > 0.5,
            "passes_icl_gate_lt_0_5": loss_second_post < 0.5,
        },
        "improvement": {
            "prefix_score_delta": float(score_post - score_pre),
            "icl_loss_delta": float(loss_second_post - loss_second_pre),
        },
    }

    print(json.dumps(results, indent=2))

    # Save results
    results_file = tmp_path / "loom_constructed_demo.json"
    results_file.write_text(json.dumps(results, indent=2))

    # Check gates after polish
    assert results["after_polish"]["passes_prefix_gate_gt_0_5"], \
        f"Failed: prefix_score > 0.5 (post-polish: {score_post})"
    assert results["after_polish"]["passes_icl_gate_lt_0_5"], \
        f"Failed: icl_loss < 0.5 (post-polish: {loss_second_post})"
