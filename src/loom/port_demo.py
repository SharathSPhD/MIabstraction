"""End-to-end PORT demo: Loom on frozen HF models.

Demonstrates:
1. Loading a frozen model (Llama-3.2-1B)
2. Direction extraction via contrastive prompts (prayoga)
3. Dose calibration with EC50 fitting
4. Event-gated vs always-on comparison (prabodha)
5. Monitor probe fitting
6. Gate verification
7. Results report (loom_port_demo.json)

Run: python -m loom.port_demo
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from loom.port import PortModel, DoseResponse, compute_steering_vector_hf
from loom.event_gating import (
    EntropyGateConfig,
    EntropyGate,
    calibrate_entropy_gate,
    measure_entropy,
)
from loom.verify import Report, GateResult, Gate



# `create_demo_report()` used to live here. It built a complete, plausible PORT report —
# EC50, dose-response curve, event-gating arms, all of it — out of literals, and main()
# wrote it to results/ whether or not the model had loaded. A reader of that file could
# not tell it from a measurement, which is the only property that matters.
#
# There is no replacement. If the model cannot be loaded, the demo has nothing to report
# and says so. The real run's artifact is results/loom_port_demo.json, produced by
# scripts that actually load the weights.

def main():
    """The PORT demonstration is a real measurement, and lives where it can take one.

    This module used to hold both a fabricated report and a main() that wrote it. The
    measurements are in loom.experiments.port_real, which loads the weights and reports
    what it finds; this entry point defers to it so there is exactly one thing that can
    produce results/loom_port_demo.json.
    """
    from loom.experiments.port_real import run
    report = run()
    OUT = Path("results/loom_port_demo.json")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[PORT] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
