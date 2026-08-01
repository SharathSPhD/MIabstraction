"""Test pure hand-constructed induction head with Hadamard codes.

Verification gates:
- prefix_score > 0.5 (attention mass on correct positions)
- icl_argmax_accuracy > 0.9 (next-token prediction on second copy)
- icl_loss reported (should be well below 0.5 with strong unembedding)

Uses vocab=20, copy_len=24 to minimize spurious matches (~1.2 per sequence).
Verifies layer-by-layer before composition.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from loom.constructed import compile_induction
from miabstraction.experiments.e2_induction import (
    attention_patterns,
    copy_region_losses,
    gapped_doubled_sequences,
    prefix_matching_score,
)


@pytest.mark.timeout(120)
def test_layer1_prev_token_attention():
    """Verify layer 1 attention is shifted identity on positions."""
    model = compile_induction(vocab=20, d_model=256, max_len=64)

    # Create simple test: [0, 1, 2, 3, ...]
    tokens = torch.arange(20, dtype=torch.long)[None, :20]

    pats = attention_patterns(model, tokens)
    layer1_attn = pats[0][0, 0]  # batch 0, head 0

    # Layer 1 should attend to p-1
    for p in range(1, 20):
        max_pos = layer1_attn[p].argmax().item()
        assert max_pos == p - 1, f"Position {p}: attended to {max_pos}, expected {p-1}"
        weight = layer1_attn[p, p-1].item()
        assert weight > 0.9, f"Attention weight at p-1 should be high, got {weight}"

    print("✓ Layer 1: Previous-token attention verified")


@pytest.mark.timeout(300)
def test_constructed_induction_gates():
    """Test pure hand construction on induction task with proper gates."""
    print("\n" + "="*70)
    print("PURE HAND-CONSTRUCTED INDUCTION HEAD TEST")
    print("="*70)

    model = compile_induction(vocab=20, d_model=256, max_len=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Test data: vocab=20, copy_len=24 minimizes spurious matches
    rng = np.random.default_rng(42)
    copy_len = 24
    max_gap = 12
    n_test = 64

    seqs, gaps = gapped_doubled_sequences(n_test, copy_len, max_gap, 20, rng)
    tokens = torch.from_numpy(seqs).to(device)

    print(f"\nTest data:")
    print(f"  Vocab: 20, Copy length: {copy_len}, Max gap: {max_gap}")
    print(f"  Expected spurious token matches: ~{copy_len/20:.2f} per sequence")
    print(f"  Test sequences: {n_test}")

    # GATE 1: Prefix-matching score > 0.5
    score = prefix_matching_score(model, tokens, gaps, copy_len)
    score_pass = score > 0.5
    print(f"\nGate 1: Prefix-matching score")
    print(f"  Value: {score:.4f}")
    print(f"  Requirement: > 0.5")
    print(f"  Status: {'✓ PASS' if score_pass else '✗ FAIL'}")

    # GATE 2: ICL argmax accuracy > 0.9
    with torch.no_grad():
        logits = model(tokens[:, :-1])
        preds = logits.argmax(dim=-1)

    icl_accuracy = 0.0
    icl_count = 0
    for b in range(tokens.shape[0]):
        g = int(gaps[b])
        start_idx = copy_len + g - 1
        end_idx = min(2*copy_len + g - 1, preds.shape[1])
        if start_idx < preds.shape[1]:
            start_tok = copy_len + g
            end_tok = min(2*copy_len + g, tokens.shape[1])
            pred_slice = preds[b, start_idx:end_idx]
            tok_slice = tokens[b, start_tok:end_tok]
            if len(pred_slice) > 0 and len(tok_slice) > 0:
                min_len = min(len(pred_slice), len(tok_slice))
                matches = (pred_slice[:min_len] == tok_slice[:min_len]).sum().item()
                icl_accuracy += matches
                icl_count += min_len

    if icl_count > 0:
        icl_accuracy /= icl_count

    accuracy_pass = icl_accuracy > 0.9
    print(f"\nGate 2: ICL argmax accuracy on second copy")
    print(f"  Value: {icl_accuracy:.4f}")
    print(f"  Requirement: > 0.9")
    print(f"  Status: {'✓ PASS' if accuracy_pass else '✗ FAIL'}")

    # Reported metric: ICL loss
    loss_first, loss_second = copy_region_losses(model, tokens, gaps, copy_len)
    print(f"\nReported metric: ICL loss")
    print(f"  Value: {loss_second:.4f}")
    print(f"  Uniform baseline: {np.log(20):.4f}")
    print(f"  Note: With strong attention + unembedding, should be << 0.5")

    # Per-layer analysis
    pats = attention_patterns(model, tokens)
    per_layer_scores = []
    for i, pat in enumerate(pats):
        layer_score = 0.0
        count = 0
        for batch_idx in range(tokens.shape[0]):
            g = gaps[batch_idx]
            for p in range(copy_len + g + 1, min(2*copy_len + g, tokens.shape[1])):
                target_k = p - (copy_len + g) + 1
                if target_k >= 0 and target_k < tokens.shape[1]:
                    attn_at_target = pat[batch_idx, 0, p, target_k].item()
                    layer_score += attn_at_target
                    count += 1
        if count > 0:
            layer_score /= count
        per_layer_scores.append(layer_score)

    print(f"\nPer-layer prefix scores:")
    for i, s in enumerate(per_layer_scores):
        print(f"  Layer {i}: {s:.4f}")

    # Compile results for JSON
    results = {
        "construction_pure": True,
        "model_config": {
            "vocab": 20,
            "d_model": 256,
            "n_layers": 2,
            "n_heads": 2,
            "attn_only": True,
            "max_len": 64,
        },
        "verification": {
            "sequence_config": {
                "copy_len": copy_len,
                "max_gap": max_gap,
                "spurious_matches_expected": float(copy_len / 20),
            },
            "prefix_score": float(score),
            "icl_loss": float(loss_second),
            "icl_argmax_accuracy": float(icl_accuracy),
            "first_copy_loss": float(loss_first),
            "uniform_baseline": float(np.log(20)),
            "per_layer_scores": [float(s) for s in per_layer_scores],
        },
        "gates": {
            "prefix_score_gt_0_5": {
                "value": float(score),
                "passes": score_pass,
            },
            "icl_argmax_accuracy_gt_0_9": {
                "value": float(icl_accuracy),
                "passes": accuracy_pass,
            },
        },
        "polish_steps": 0,
    }

    # Save results
    results_dir = Path("/home/sharaths/projects/MIabstraction-loomC/results")
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / "loom_constructed_demo.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_file}")

    print("\n" + "="*70)
    if score_pass and accuracy_pass:
        print("FINAL STATUS: ALL GATES PASSED ✓✓✓")
        results["status"] = "PASS"
    elif score_pass:
        print("FINAL STATUS: Partial (prefix gate passed) →→→ needs pure construction iteration")
        results["status"] = "PARTIAL"
    else:
        print("FINAL STATUS: Construction needs refinement")
        results["status"] = "NEEDS_WORK"
    print("="*70)

    # Assert gates
    assert score_pass, f"prefix_score {score:.4f} failed gate > 0.5"
    assert accuracy_pass, f"icl_argmax_accuracy {icl_accuracy:.4f} failed gate > 0.9"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
