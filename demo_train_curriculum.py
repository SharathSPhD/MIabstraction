#!/usr/bin/env python3
"""Demonstrate multi-skill curriculum training with honest gates.

Uses E2-proven recipe:
- Attention-only architecture (induction requires it)
- 200k diverse sequences for induction
- 20k steps, batch 256, lr 1e-3
- All gates must pass or we report honest refusal with diagnosis
"""
import json
import time
from pathlib import Path

import torch

from loom.curriculum import compile_curriculum, train
from loom.spec import load_weave


def main():
    t_start = time.time()

    # Load the demo weave
    weave_path = Path(__file__).parent / "demo_curriculum.yaml"
    spec = load_weave(weave_path)

    print(f"Loading weave: {weave_path}")
    print(f"Skills: {[s.name for s in spec.skills]}")
    print(f"Gates:")
    for g in spec.gates:
        print(f"  {g.describe()}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Compile curriculum with E2-proven hyperparameters
    print("Compiling curriculum...")
    plan = compile_curriculum(
        spec,
        max_steps=20000,  # E2 used 20000
        batch_size=256,   # E2 used 256
        lr=1e-3,          # E2 used 0.001
        device=device,
    )

    print(f"Vocabulary: {plan.vocab_plan.total_vocab} tokens")
    print(f"Attention-only: {plan.attn_only}")
    for skill_name, n_seq in {k: v["n_seq_train"] for k, v in plan.datasets.items()}.items():
        print(f"  {skill_name}: {n_seq:,} sequences")
    print(f"Max steps: {plan.max_steps}, batch: {plan.batch_size}, lr: {plan.lr}\n")

    # Train
    print("Training...")
    model, metrics, losses = train(spec, plan, device=device)

    t_elapsed = time.time() - t_start

    print("\nFinal metrics:")
    for skill_name, skill_metrics in metrics.items():
        print(f"  {skill_name}:")
        for metric_name, value in skill_metrics.items():
            print(f"    {metric_name}: {value:.6f}")

    # Check gates
    print("\nGate status:")
    gate_pass = True
    gates_by_target = {}
    for g in spec.gates:
        if g.target not in gates_by_target:
            gates_by_target[g.target] = []
        gates_by_target[g.target].append(g)

    for skill_name, gates in gates_by_target.items():
        skill_metrics = metrics.get(skill_name, {})
        for gate in gates:
            metric_name = gate.metric
            if metric_name in skill_metrics:
                value = skill_metrics[metric_name]
                passes = gate.holds(value)
                status = "PASS" if passes else "FAIL"
                print(f"  {gate.describe()}: {value:.6f} [{status}]")
                gate_pass = gate_pass and passes
            else:
                print(f"  {gate.describe()}: MISSING METRIC")
                gate_pass = False

    # Save results
    print(f"\n{'All gates PASSED!' if gate_pass else 'Some gates FAILED (honest refusal)'}")
    print(f"Training time: {t_elapsed:.1f}s ({t_elapsed/60:.1f}m)")

    # Build result structure
    result = {
        "weave": spec.name,
        "skills": [s.name for s in spec.skills],
        "device": device,
        "model_config": spec.model,
        "vocab_total": plan.vocab_plan.total_vocab,
        "attn_only": plan.attn_only,
        "training": {
            "max_steps": plan.max_steps,
            "batch_size": plan.batch_size,
            "lr": plan.lr,
            "actual_steps": len(losses),
            "wall_clock_seconds": round(t_elapsed, 1),
        },
        "per_skill_metrics": metrics,
        "gates": [
            {
                "target": g.target,
                "metric": g.metric,
                "op": g.op,
                "threshold": g.threshold,
                "actual": metrics.get(g.target, {}).get(g.metric),
                "passes": g.holds(metrics.get(g.target, {}).get(g.metric, float('-inf')))
                if g.metric in metrics.get(g.target, {}) else False,
            }
            for g in spec.gates
        ],
        "all_gates_pass": gate_pass,
        "refusal_diagnosis": None if gate_pass else "See per-gate passes above for diagnosis",
        "loss_samples": losses[::max(1, len(losses)//100)],  # Sample losses for visualization
    }

    # Save to exact path required by coordinator
    results_path = Path(__file__).parent / "results" / "loom_curriculum_demo.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {results_path}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    exit(main())
