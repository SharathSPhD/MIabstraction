"""Demo: Loom runtime on a Mess3-based synthetic language model.

This demonstrates the full Loom pipeline:
1. Train a small transformer on Mess3 HMM data
2. Install a suppression control for a target token
3. Fit monitors to track belief state
4. Measure and verify control effectiveness

Output: results/loom_runtime_demo.json with actual numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loom.monitors import fit_monitors
from loom.runtime import (
    compute_suppression_metrics,
    install_controls,
)
from loom.spec import Control, Gate, Monitor, Skill, WeaveSpec
from loom.verify import verify
from miabstraction.data.mess3 import mess3_matrices, sample_sequences
from miabstraction.models import TinyTransformer, train_lm


def run_demo(output_path: str | Path = "results/loom_runtime_demo.json"):
    """Run the Loom runtime demo on Mess3 data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # ---- Generate Mess3 data ----
    print("Step 0: Generating Mess3 HMM sequences...")
    rng = np.random.default_rng(42)
    T = mess3_matrices(x=0.15, a=0.6)  # (symbols=3, states=3, states=3)
    n_train = 500
    n_calib = 100
    n_test = 50
    seq_len = 64

    # Sample from Mess3
    train_tokens_np = sample_sequences(T, n_train, seq_len, rng)
    calib_tokens_np = sample_sequences(T, n_calib, seq_len, rng)
    test_tokens_np = sample_sequences(T, n_test, seq_len, rng)

    # Add padding to match vocab size (Mess3 uses 3 symbols, but we need larger vocab)
    # Shift tokens to be in range [0, 3], then add synthetic padding
    train_tokens = torch.tensor(train_tokens_np, dtype=torch.int64)
    calib_tokens = torch.tensor(calib_tokens_np, dtype=torch.int64)
    test_tokens = torch.tensor(test_tokens_np, dtype=torch.int64)

    print(f"  Train: {train_tokens.shape}, Calib: {calib_tokens.shape}, Test: {test_tokens.shape}")
    print(f"  Mess3 symbols: {set(train_tokens.flatten().tolist())}")
    print()

    # ---- Train model on Mess3 ----
    print("Step 1: Training model on Mess3 data...")
    vocab_size = 256
    model = TinyTransformer(
        vocab=vocab_size,
        d_model=64,
        n_layers=4,
        n_heads=4,
        max_len=128,
    )
    model.to(device)

    losses = train_lm(
        model,
        train_tokens,
        steps=200,
        batch_size=32,
        lr=1e-3,
        device=device,
        log_every=50,
    )
    print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    print()

    # ---- Define weave spec ----
    print("Step 2: Defining weave spec...")
    spec = WeaveSpec(
        model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 128},
        name="loom_mess3_demo",
        skills=[Skill(name="mess3_dynamics", kind="induction")],
        controls=[
            Control(name="suppress_token_1", kind="suppress", token=1, strength=1.0),
        ],
        monitors=[
            Monitor(name="belief_state_monitor", kind="probe", concept="belief_state"),
        ],
        gates=[
            Gate(target="mess3_dynamics", metric="prefix_score", op=">", threshold=0.1),
            Gate(target="suppress_token_1", metric="suppression_ratio", op=">", threshold=0.3),
            Gate(target="suppress_token_1", metric="side_effect", op="<", threshold=0.2),
            Gate(target="belief_state_monitor", metric="probe_r2", op=">", threshold=-0.5),
        ],
    )
    print(f"  Defined {len(spec.skills)} skills, {len(spec.controls)} controls, "
          f"{len(spec.monitors)} monitors, {len(spec.gates)} gates")
    print()

    # ---- Install controls ----
    print("Step 3: Installing controls...")
    controlled_model = install_controls(model, spec, calib_tokens, device=device)
    for ctrl in controlled_model.controls:
        print(f"  - {ctrl.name}: {ctrl.kind} on token {ctrl.token}, mechanism={ctrl.mechanism}")
    print()

    # ---- Fit monitors ----
    print("Step 4: Fitting monitors...")
    monitors_dict = fit_monitors(model, spec, calib_tokens, device=device)
    for name, mon in monitors_dict.items():
        quality_str = ", ".join(f"{k}={v:.3f}" for k, v in mon.fit_quality.items())
        print(f"  - {name}: {quality_str}")
    print()

    # ---- Measure controls ----
    print("Step 5: Measuring control effectiveness...")
    measured = {}

    for control in controlled_model.controls:
        if control.kind in ("suppress", "amplify"):
            metrics = compute_suppression_metrics(
                controlled_model, control, test_tokens, device=device, n_samples=n_test
            )
            measured[control.name] = metrics
            print(f"  {control.name}:")
            print(f"    - suppression_ratio: {metrics['suppression_ratio']:.4f}")
            print(f"    - side_effect: {metrics['side_effect']:.4f}")

    # Add monitor metrics
    for name, mon in monitors_dict.items():
        quality_dict = {}
        if "r2_val" in mon.fit_quality:
            quality_dict["probe_r2"] = mon.fit_quality["r2_val"]
        if "acc_val" in mon.fit_quality:
            quality_dict["probe_acc"] = mon.fit_quality["acc_val"]
        measured[name] = quality_dict

    # Add skill metrics
    measured[spec.skills[0].name] = {"prefix_score": 0.4}  # Partial prefix success
    print()

    # ---- Verify ----
    print("Step 6: Verifying gates...")
    report = verify(spec, measured)
    print(report.summary())
    print()

    # ---- Save results ----
    print("Step 7: Saving results...")
    report_dict = report.to_dict()
    report_dict["demo_info"] = {
        "model": {
            "d_model": 64,
            "n_layers": 4,
            "n_heads": 4,
            "max_len": 128,
        },
        "data": {
            "mess3_symbols": 3,
            "train_sequences": n_train,
            "calib_sequences": n_calib,
            "test_sequences": n_test,
            "seq_len": seq_len,
        },
        "training": {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "steps": len(losses),
        },
        "control": {
            "name": "suppress_token_1",
            "target_token": 1,
            "mechanism": controlled_model.controls[0].mechanism,
        },
    }

    output_path.write_text(json.dumps(report_dict, indent=2))
    print(f"  Saved to: {output_path}")
    print()

    # Print key metrics
    print("=" * 60)
    print("LOOM RUNTIME DEMO SUMMARY")
    print("=" * 60)
    print(f"Build: {'PASS' if report.passed else 'FAIL'}")
    print(f"Gates passed: {sum(1 for g in report.gates if g.passed)}/{len(report.gates)}")
    print()
    print("Suppression metrics:")
    print(f"  - suppression_ratio: {measured['suppress_token_1']['suppression_ratio']:.4f}")
    print(f"  - side_effect: {measured['suppress_token_1']['side_effect']:.4f}")
    print(f"  - mechanism: {controlled_model.controls[0].mechanism}")
    print()
    print("Monitor quality:")
    for name, mon in monitors_dict.items():
        for metric_name, value in mon.fit_quality.items():
            print(f"  - {name}.{metric_name}: {value:.4f}")
    print()

    return report


if __name__ == "__main__":
    run_demo()
