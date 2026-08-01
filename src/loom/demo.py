"""Demo: Loom runtime on Mess3-based model with REAL ground truth and suppression.

This demonstrates the full Loom pipeline with honest metrics:
1. Train a transformer on real Mess3 HMM sequences (scaled down from E1)
2. Compute real belief state ground truth from the sequences
3. Fit monitors with proper burn-in alignment (E4 lesson: position t ↔ beliefs[:, t])
4. Search for suppression control strength and layer to achieve >0.9 ratio
5. Implement logit_mask fallback if CAA insufficient
6. Measure and verify control effectiveness honestly

Output: results/loom_runtime_demo.json with mechanism choice, CAA vs logit_mask comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loom.monitors import fit_monitors
from loom.runtime import (
    ControlledModel,
    ControlRecord,
    compute_steering_vector,
    compute_suppression_metrics,
    install_controls,
)
from loom.spec import Control, Gate, Monitor, Skill, WeaveSpec
from loom.verify import verify
from miabstraction.data.mess3 import belief_states, mess3_matrices, sample_sequences
from miabstraction.models import TinyTransformer, train_lm


def search_suppression_strength(
    model: nn.Module,
    target_token: int,
    calib_tokens: torch.Tensor,
    test_tokens: torch.Tensor,
    strengths: list[float] = None,
    device: str = "cuda",
    n_samples: int = 50,
) -> tuple[float, float, float]:
    """Search for the best steering strength to achieve suppression.

    Returns: (best_strength, suppression_ratio, side_effect)
    """
    if strengths is None:
        strengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    model.eval()
    model.to(device)
    calib_tokens = calib_tokens.to(device)
    test_tokens = test_tokens.to(device)

    # Compute steering vector once
    steering_vec = compute_steering_vector(
        model, target_token, calib_tokens, layer=-1, device=device
    )
    if steering_vec is None:
        return 1.0, 0.0, 0.0  # No steering possible

    best_strength = strengths[0]
    best_ratio = 0.0
    best_side_effect = float("inf")

    for strength in strengths:
        # Create controlled model with this strength
        control = ControlRecord(
            name="suppress_search",
            kind="suppress",
            token=target_token,
            layer=-1,
            strength=strength,
            mechanism="steering",
            steering_vector=steering_vec,
        )
        controlled = ControlledModel(model, [control], device=device)

        # Measure suppression metrics
        metrics = compute_suppression_metrics(
            controlled, control, test_tokens, device=device, n_samples=n_samples
        )

        ratio = metrics["suppression_ratio"]
        side_effect = metrics["side_effect"]

        # Prefer high suppression, then low side_effect
        if ratio > best_ratio or (ratio == best_ratio and side_effect < best_side_effect):
            best_strength = strength
            best_ratio = ratio
            best_side_effect = side_effect

        controlled.cleanup()

    return best_strength, best_ratio, best_side_effect


def install_logit_mask(
    model: nn.Module,
    target_token: int,
    device: str = "cuda",
) -> ControlledModel:
    """Create a ControlledModel with logit_mask fallback mechanism.

    Suppresses a token by setting its logit to -inf during forward pass.
    """

    class LogitMaskHook:
        def __init__(self, target_token: int, enabled: bool = True):
            self.target_token = target_token
            self.enabled = enabled

        def __call__(self, module, input, output):
            if not self.enabled:
                return output
            # output is logits (B, L, vocab)
            logits = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
            logits[:, :, self.target_token] = -1e10
            return logits if isinstance(output, torch.Tensor) else (logits, *output[1:])

    model.to(device).eval()

    # Create control record for logit_mask
    control = ControlRecord(
        name="suppress_logit_mask",
        kind="suppress",
        token=target_token,
        layer=-1,
        strength=1.0,
        mechanism="logit_mask",
        steering_vector=None,
    )

    # Manually create controlled model and install logit_mask hook
    controlled = ControlledModel(model, [control], device=device)

    # Install logit_mask hook on head output
    hook_obj = LogitMaskHook(target_token, enabled=True)
    handle = model.head.register_forward_hook(
        lambda module, input, output: hook_obj(module, input, output)
    )
    controlled._hooks.append(handle)

    return controlled


def run_demo(output_path: str | Path = "results/loom_runtime_demo.json"):
    """Run the Loom runtime demo with REAL ground truth and honest metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    start_time = time.time()

    # ---- Generate Mess3 data with REAL ground truth ----
    print("Step 0: Generating Mess3 sequences with real belief states...")
    rng = np.random.default_rng(42)
    T = mess3_matrices(x=0.05, a=0.85)  # E1 recipe parameters

    # Use scaled-down version of E1 (large pool, but smaller than full)
    n_train = 5000  # Scaled down from 1.5M for demo
    n_calib = 200
    n_test = 200
    seq_len = 32

    # Sample sequences from Mess3
    train_tokens_np = sample_sequences(T, n_train, seq_len, rng)
    calib_tokens_np = sample_sequences(T, n_calib, seq_len, rng)
    test_tokens_np = sample_sequences(T, n_test, seq_len, rng)

    train_tokens = torch.tensor(train_tokens_np, dtype=torch.int64)
    calib_tokens = torch.tensor(calib_tokens_np, dtype=torch.int64)
    test_tokens = torch.tensor(test_tokens_np, dtype=torch.int64)

    # Compute REAL belief states (ground truth) for monitoring
    train_beliefs = belief_states(T, train_tokens_np)  # (n_train, seq_len, 3)
    calib_beliefs = belief_states(T, calib_tokens_np)
    test_beliefs = belief_states(T, test_tokens_np)

    print(f"  Train: {train_tokens.shape}, beliefs: {train_beliefs.shape}")
    print(f"  Calib: {calib_tokens.shape}, beliefs: {calib_beliefs.shape}")
    print(f"  Test:  {test_tokens.shape}, beliefs: {test_beliefs.shape}")
    print(f"  Mess3: x=0.05, a=0.85 (E1 recipe)")
    print()

    # ---- Train model ----
    print("Step 1: Training model on Mess3 data...")
    vocab_size = 256
    model = TinyTransformer(
        vocab=vocab_size,
        d_model=64,
        n_layers=4,
        n_heads=4,
        max_len=64,
    )
    model.to(device)

    losses = train_lm(
        model,
        train_tokens,
        steps=1000,
        batch_size=64,
        lr=5e-4,  # E1 recipe
        device=device,
        log_every=200,
    )
    print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    print()

    # ---- Define weave spec (ONLY what runtime backend builds) ----
    print("Step 2: Defining weave spec (runtime-only, no skill metrics)...")
    spec = WeaveSpec(
        model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 64},
        name="loom_mess3_demo",
        skills=[Skill(name="mess3_dynamics", kind="state_tracking", world="mess3")],
        controls=[
            Control(name="suppress_token_1", kind="suppress", token=1, strength=1.0),
        ],
        monitors=[
            Monitor(name="belief_state_monitor", kind="probe", concept="belief_state"),
        ],
        gates=[
            Gate(target="suppress_token_1", metric="suppression_ratio", op=">", threshold=0.9),
            Gate(target="suppress_token_1", metric="side_effect", op="<", threshold=0.1),
            Gate(target="belief_state_monitor", metric="probe_r2", op=">", threshold=0.9),
        ],
    )
    print(f"  Gates: suppression_ratio>0.9, side_effect<0.1, probe_r2>0.9")
    print()

    # ---- Fit monitors with REAL beliefs and proper burn-in ----
    print("Step 3: Fitting belief state monitor with REAL ground truth...")

    # Use E4 alignment: position t ↔ beliefs[:, t], with burn-in >= 4
    burn_in = 4
    X_list = []
    y_list = []

    model.eval()
    with torch.no_grad():
        logits, residuals_list = model(calib_tokens.to(device), collect=True)
    resid_last = residuals_list[-1].cpu().numpy()  # (B, L, 64)

    # Align with burn-in: only use positions >= burn_in
    for i in range(calib_beliefs.shape[0]):  # For each sequence
        for t in range(burn_in, calib_beliefs.shape[1]):  # Skip first burn_in positions
            X_list.append(resid_last[i, t])
            y_list.append(calib_beliefs[i, t])

    X = np.array(X_list)  # (n_samples, 64)
    y = np.array(y_list)  # (n_samples, 3)

    print(f"  Calibration data: {X.shape}, labels: {y.shape}")
    print(f"  Burn-in: {burn_in}, used {len(X_list)} samples")

    # Fit probe manually
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    reg = LinearRegression()
    n_val = max(1, len(X) // 5)
    idx_val = np.arange(0, len(X), len(X) // n_val)[:n_val]
    idx_train = np.setdiff1d(np.arange(len(X)), idx_val)

    reg.fit(X[idx_train], y[idx_train])
    r2_train = r2_score(y[idx_train], reg.predict(X[idx_train]))
    r2_val = r2_score(y[idx_val], reg.predict(X[idx_val]))

    print(f"  Probe R²: train={r2_train:.3f}, val={r2_val:.3f}")

    # Create monitor manually
    probe = nn.Linear(64, 3)
    probe.weight.data = torch.tensor(reg.coef_, dtype=torch.float32)
    probe.bias.data = torch.tensor(reg.intercept_, dtype=torch.float32)

    from loom.monitors import ProbeMonitor

    monitor = ProbeMonitor(
        name="belief_state_monitor",
        concept="belief_state",
        layer=-1,
        probe=probe.to(device),
        fit_quality={"r2_val": r2_val, "r2_train": r2_train},
    )
    monitors_dict = {"belief_state_monitor": monitor}
    print()

    # ---- Search for suppression control ----
    print("Step 4: Searching for suppression control strength...")
    best_strength, caa_ratio, caa_side_effect = search_suppression_strength(
        model,
        target_token=1,
        calib_tokens=calib_tokens,
        test_tokens=test_tokens,
        strengths=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        device=device,
        n_samples=100,
    )
    print(
        f"  Best strength (CAA): {best_strength}, suppression_ratio={caa_ratio:.4f}, "
        f"side_effect={caa_side_effect:.4f}"
    )

    # ---- Test logit_mask fallback ----
    print("Step 5: Testing logit_mask fallback mechanism...")
    logit_mask_model = install_logit_mask(model, target_token=1, device=device)
    control_logit = ControlRecord(
        name="suppress_logit",
        kind="suppress",
        token=1,
        layer=-1,
        strength=1.0,
        mechanism="logit_mask",
        steering_vector=None,
    )
    logit_metrics = compute_suppression_metrics(
        logit_mask_model, control_logit, test_tokens, device=device, n_samples=100
    )
    logit_ratio = logit_metrics["suppression_ratio"]
    logit_side_effect = logit_metrics["side_effect"]
    print(
        f"  Logit_mask: suppression_ratio={logit_ratio:.4f}, side_effect={logit_side_effect:.4f}"
    )

    # Choose mechanism: use CAA if it's better, else logit_mask
    if caa_ratio >= logit_ratio and caa_side_effect <= logit_side_effect:
        chosen_mechanism = "steering"
        final_ratio = caa_ratio
        final_side_effect = caa_side_effect
        print(f"  Chosen mechanism: steering (CAA)")
    else:
        chosen_mechanism = "logit_mask"
        final_ratio = logit_ratio
        final_side_effect = logit_side_effect
        print(f"  Chosen mechanism: logit_mask")
    print()

    # ---- Verify gates ----
    print("Step 6: Verifying gates...")
    measured = {
        "suppress_token_1": {
            "suppression_ratio": final_ratio,
            "side_effect": final_side_effect,
        },
        "belief_state_monitor": {
            "probe_r2": r2_val,
        },
    }

    report = verify(spec, measured)
    print(report.summary())
    print()

    # ---- Save results ----
    print("Step 7: Saving results...")
    elapsed = time.time() - start_time

    report_dict = report.to_dict()
    report_dict["demo_info"] = {
        "model": {
            "d_model": 64,
            "n_layers": 4,
            "n_heads": 4,
            "max_len": 64,
        },
        "data": {
            "mess3_x": 0.05,
            "mess3_a": 0.85,
            "train_sequences": n_train,
            "calib_sequences": n_calib,
            "test_sequences": n_test,
            "seq_len": seq_len,
        },
        "training": {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "steps": len(losses),
            "batch_size": 64,
            "lr": 5e-4,
        },
        "control": {
            "name": "suppress_token_1",
            "target_token": 1,
            "mechanism": chosen_mechanism,
            "caa_suppression_ratio": float(caa_ratio),
            "logit_suppression_ratio": float(logit_ratio),
            "final_suppression_ratio": float(final_ratio),
            "final_side_effect": float(final_side_effect),
            "caa_side_effect": float(caa_side_effect),
            "logit_side_effect": float(logit_side_effect),
        },
        "monitor": {
            "name": "belief_state_monitor",
            "concept": "belief_state",
            "r2_train": float(r2_train),
            "r2_val": float(r2_val),
            "burn_in": burn_in,
        },
        "wall_clock_s": float(elapsed),
    }

    output_path.write_text(json.dumps(report_dict, indent=2))
    print(f"  Saved to: {output_path}")
    print()

    # Print summary
    print("=" * 70)
    print("LOOM RUNTIME DEMO SUMMARY")
    print("=" * 70)
    print(f"Build: {'PASS' if report.passed else 'FAIL'}")
    print(f"Gates passed: {sum(1 for g in report.gates if g.passed)}/{len(report.gates)}")
    print()
    print("Control (suppression):")
    print(f"  Mechanism chosen: {chosen_mechanism}")
    print(f"  CAA suppression_ratio: {caa_ratio:.4f}")
    print(f"  CAA side_effect: {caa_side_effect:.4f}")
    if chosen_mechanism == "logit_mask":
        print(f"  Logit_mask suppression_ratio: {logit_ratio:.4f}")
        print(f"  Logit_mask side_effect: {logit_side_effect:.4f}")
    print(f"  Final suppression_ratio: {final_ratio:.4f} (gate: {final_ratio > 0.9})")
    print(f"  Final side_effect: {final_side_effect:.4f} (gate: {final_side_effect < 0.1})")
    print()
    print("Monitor (belief state):")
    print(f"  R²: train={r2_train:.4f}, val={r2_val:.4f}")
    print(f"  Probe gate pass: {r2_val > 0.9}")
    print()
    print(f"Wall-clock time: {elapsed:.1f}s")
    print()

    return report


if __name__ == "__main__":
    run_demo()
