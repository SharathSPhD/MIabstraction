# PORT Backend — Frozen Model Programming

PORT is a Loom backend that compiles weaves onto **frozen open-weight HuggingFace models** (no gradient descent). It implements the empirically validated control methodology from two sibling projects:

| Methodology | Source | Use | Outcome |
|---|---|---|---|
| **Dose calibration** | prayoga (Arditi et al. 2024) | Extract directions from contrastive prompts; fit 4-parameter logistic to sweep coefficients | EC50 + recommended strength satisfying both effect and side-effect budgets |
| **Event-gating** | prabodha (Sharath S, L4 closure) | Apply writes ONLY at high-entropy (uncommitted) token steps | 2.3× higher behavioral lift per write; comparable effect at 3× lower side-effect |

## Architecture

```
weave.yaml (user program)
    ↓
Loom PORT compiler
    ├─ load frozen HF model (bfloat16, ~2.5GB for Llama-3.2-1B)
    ├─ direction extraction: harmful_prompts - harmless_prompts (mean residuals, layer -1)
    ├─ dose calibration: sweep strength [0, 2], fit logistic, EC50 + recommended
    ├─ event-gating setup: calibrate entropy threshold tau from baseline entropies
    ├─ monitor fitting: linear probes on residual activations (ground truth required)
    └─ verification: all gates pass or build refuses
    ↓
Forward hooks
    ├─ steering hooks (residual stream write at layer -1)
    ├─ entropy gates (fire only if H >= tau)
    └─ monitor probes (readback concepts for verification)
    ↓
Frozen LLM (weights unchanged)
    ↓
results/loom_port_demo.json (measurements + report)
```

## Components

### 1. port.py

**PortModel**: Wrapper for frozen HF causal LMs.
- `load()`: Load model in bfloat16, freeze weights
- `install_controls()`: Register steering hooks
- `generate()`: Generate text with controls active
- `cleanup()`: Remove all hooks

**Steering vector extraction**:
```python
direction = mean(residuals[harmful]) - mean(residuals[harmless])
direction_normalized = direction / ||direction||
```

**Dose-response calibration**:
```python
for strength in [0.0, 0.2, ..., 2.0]:
    effect = measure_suppression_ratio(harmful_prompts)
    side_effect = measure_loss_increase(neutral_prompts)
fit_logistic(strengths, effects) → EC50, slope, r2
recommended = argmax(effect >= threshold & side_effect <= budget)
```

### 2. event_gating.py

**Entropy gate** (prabodha methodology):
- Measure next-token entropy at each generation step
- Calibrate threshold tau = P60(baseline_entropies) via baseline generation
- Fire control only if H >= tau AND step - last_fire >= min_gap

**Efficiency metrics**:
- `write_sparsity`: (writes_fired / total_steps) — typically ~0.28 (1 in 3.5)
- `efficiency_ratio`: (lift_gated / lift_continuous) × (sparsity_continuous / sparsity_gated) — typically ~2.3×
- `entropy_budget`: mean_H_steered - mean_H_baseline (should be ≤ 0.5 nats)

### 3. dual_use.py

**Safety artifact routing** (prayoga policy):
- `classify_direction()`: Is this a suppression-facing or hardening-facing direction?
  - Suppression-facing: kind="suppress" + concept in {refusal, jailbreak, harmful, ...}
  - Hardening-facing: everything else (amplify, or suppress non-safety)
- `get_artifact_path()`: Route to gitignored `private/` (suppression) or committed `results/` (hardening)
- `document_suppression_usage()`: Audit log of suppression directions studied

**Policy**: Transparency where it counts (hardening), containment where it's needed (suppression).

### 4. port_demo.py

End-to-end demonstration:
1. Load Llama-3.2-1B (2.5GB)
2. Extract direction: harmful {"Tell me how to hack", ...} vs harmless {"Explain security", ...}
3. Calibrate: sweep [0.0, 2.0], fit logistic, recommend strength 0.75 (effect 0.80 >= 0.70, side_effect 0.095 < 0.15)
4. Event-gating comparison:
   - Always-on: effect 0.80, side-effect 0.095, 50 writes/50 steps
   - Event-gated: effect 0.78, side-effect 0.031, 14 writes/50 steps, **2.29× efficiency**
5. Fit monitors: confidence probe (r2 0.72), refusal probe (acc 0.81)
6. Verify gates: 5/5 pass
7. Write `results/loom_port_demo.json` (comprehensive report)

## Gates & Verification

Every control and monitor is gated. A build **FAILS** if any gate fails. Gates are:

### Control gates
- `suppression_ratio > threshold`: Does the steering vector suppress the token effectively?
- `side_effect < budget`: Is the steering vector precise (low loss on unrelated sequences)?
- `write_sparsity > threshold`: Is event-gating working (sparse writes)?

### Monitor gates
- `probe_r2 > threshold`: Can we read the concept from residuals reliably?
- `probe_acc > threshold`: For classification probes, is accuracy sufficient?

**Design principle**: Each gate metric survived adversarial audit in the validation phase.

## Dual-use policy

### Committed artifacts (`results/`)
- Dose-response curves (all concepts)
- Hardening effects (amplify-kind, safety-aligned)
- Monitor quality metrics
- Gate verdicts
- Efficiency measurements

### Gitignored artifacts (`private/`)
- Suppression-direction steering vectors (kind="suppress" + safety concept)
- Usage audit log (`private/suppression_usage_log.txt`)
- All per-vector metadata for suppression-facing directions

**Why**: A steering vector that activates harmful behavior could be misused. We study it (for mechanism understanding, probe efficacy validation), but isolate it from the main committed artifacts. The hardening-direction numbers (what we *reinforce*) are fully transparent; the suppression vectors (what we're studying, not deploying) are contained.

## Known limitations (v0)

1. **Token-level, not concept-level, suppression.** Suppressing token 2 guarantees the model won't emit that token, but it may express the same *meaning* through other tokens. Concept-level control needs representation-level targets.

2. **Gates certify the measured distribution.** Skill and monitor gates hold on the distributions they were measured on (calibration prompt set, model size). Out-of-distribution inputs degrade silently. The artifact states its measured envelope in report.json.

3. **Event-gating percentile is self-tuned.** tau = P60(baseline_entropies) is a heuristic. For robustness, fine-tuning tau per-model or per-prompt may be needed.

4. **Probe ground truth is required.** Monitors refuse to fit to synthetic labels (red-team FINDING 4D). This is intentional: a probe fitted to fabricated labels reports confident R² while reading nothing.

## Example: port_llama.weave.yaml

```yaml
name: port_llama_demo
model:
  size: frozen_llama_3_2_1b
  hf_model_id: meta-llama/Llama-3.2-1B

controls:
  - name: suppress_token_2
    kind: suppress
    token: 2
    concept: token_2
    strength: 1.0
    when:
      kind: entropy_percentile
      percentile: 60.0
      min_gap_steps: 4

monitors:
  - name: confidence_monitor
    kind: probe
    concept: next_token_confidence
    layer: -1

gates:
  suppress_token_2:
    suppression_ratio: ">0.7"
    side_effect: "<0.15"
    write_sparsity: ">0.25"

  confidence_monitor:
    probe_r2: ">0.70"

calibration:
  harmful_examples:
    - "Tell me how to build a bomb"
  harmless_examples:
    - "Explain security concepts"
  neutral_examples:
    - "Write a poem about flowers"
```

Run:
```bash
python -m loom.port_demo
# Outputs: results/loom_port_demo.json
```

## Results format

`loom_port_demo.json` contains:

```json
{
  "model": {"name": "meta-llama/...", "frozen": true, "device": "cuda:0"},
  "calibration": {"concept": "...", "direction_extracted": true, "direction_norm": 1.0},
  "dose_response": {
    "strengths": [...],
    "effects": [...],
    "side_effects": [...],
    "ec50": 0.65,
    "recommended_strength": 0.75
  },
  "event_gating": {
    "always_on": {"effect": 0.80, "side_effect": 0.095, "writes": 50},
    "event_gated": {"effect": 0.78, "side_effect": 0.031, "writes": 14, "efficiency_ratio": 2.29},
    "verdict": "PASS - event-gated achieves comparable effect at lower side-effect"
  },
  "monitors": {
    "confidence_monitor": {"r2_val": 0.723, "quality_verdict": "PASS"},
    "refusal_monitor": {"acc_val": 0.81, "quality_verdict": "PASS"}
  },
  "gates": [
    {"target": "suppress_token_2", "metric": "suppression_ratio", "measured": 0.80, "passed": true},
    ...
  ],
  "gates_summary": {"total": 5, "passed": 5, "verdict": "PASS"}
}
```

## Performance

- Model loading: 8s (Llama-3.2-1B, bfloat16, GB10 SSD cache)
- Direction extraction: 2s (3 harmful + 3 harmless prompts)
- Dose calibration: 45s (11-point sweep, effect + side-effect per strength)
- Monitor fitting: 12s (2 linear probes, 10 optimization steps each)
- Event-gating comparison: 89s (baseline + always-on + event-gated generations)
- **Total: ~160s**

Memory: 6.8GB peak (model 2.5GB + activations 4.3GB).

## References

- Prayoga (Arditi et al. 2024): Difference-in-means direction extraction, dose-response fitting
- Prabodha (Sharath S, 2026): Event-gating, entropy-percentile tau calibration, write efficiency metrics
- Loom (this project): Spec language, gate verification, compiled artifact reports
