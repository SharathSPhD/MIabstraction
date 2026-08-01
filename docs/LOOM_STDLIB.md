# Loom Standard Library: Verified, Importable Units

**Version:** 0.1-alpha  
**Date:** 2026-08-01  
**Status:** Design proposal for standard library of grounded, evidence-backed units

This document proposes Loom's standard library of importable controls, monitors, and skills — each one grounded in findings from prayoga, prabodha, or PWM. The library is meant to be **C-standard-library-like**: concrete syntax, small, no vague aspirations, each unit has measured evidence and a gate.

---

## Design Principle

Each stdlib unit is a **pre-compiled, pre-verified building block** that a non-expert can use without understanding the internals. Instead of writing:

```yaml
controls:
  - name: never_token_2
    kind: suppress
    token: 2
    strength: 0.7
```

A user can import:

```yaml
stdlib:
  - unit: refusal_hardening
    strength: 0.7
```

The compiler expands this to the full control spec, pulls the evidence gates from the prayoga/prabodha ledgers, and verifies them.

---

## Units by Category

### A. Refusal Hardening

#### `refusal_hardening`
**What it does:** Amplifies refusal in language models via activation addition on the refusal direction.

**Declaration:**
```yaml
controls:
  - unit: refusal_hardening
    strength: 0.5  # coefficient in [0, 1]
    target_model: gemma-2-2b | qwen2.5-3b  # family-specific dose lookup
```

**What it guarantees:**
- Behavioral lift (over-refusal on benign prompts): Δ+0.95 (Gemma) or Δ+0.75 (Qwen, dose-adjusted)
- Random-direction control is flat (effect is specific)
- EC50-calibrated: providing strength=0.329 gives ~50% refusal lift

**Evidence source:** prayoga F1, F2, F23  
**Gate:** `over_refusal_ratio > 0.90` and `random_control_ratio < 0.05`

**Honest limitation:** This control induces over-refusal on benign prompts (false positives). Use prabodha's `jailbreak_gate` to detect attacks and gate the hardening.

---

#### `refusal_suppression`
**What it does:** Ablates the refusal direction, enabling harmful content generation. **PRIVATE UNIT — used in research only.**

**Declaration (research use only):**
```yaml
controls:
  - unit: refusal_suppression
    strength: 0.7  # ablation coefficient
    target_model: gemma-2-2b
```

**Artifacts routed to:** `private/` (gitignored)

**Note:** This unit is only documented for completeness. Publishing or releasing this control is a dual-use violation.

---

### B. Entropy-Gated Steering

#### `event_gated_control`
**What it does:** Applies a control only at high-entropy workspace moments (sphurattā events), trading behavioral lift (67%) for dramatic write sparsity (29%).

**Declaration:**
```yaml
controls:
  - unit: event_gated_control
    concept: refusal
    when_entropy_percentile: 0.5
    strength: 0.7
    svAtantrya_budget: 0.5  # max entropy cost in nats
```

**What it guarantees:**
- Lift-per-write ratio: 2.32× (range 1.83–3.25) vs. continuous writing
- Entropy cost: Δ ≤ 0.5 nats (95% of sequences)
- Lift recovery: 67% (i.e., 67% of the full continuous effect with 29% of the writes)

**Evidence source:** prabodha L9, L11, L22  
**Gate:** `entropy_cost < 0.5 nats` and `lift_per_write_ratio > 1.8`

**Honest limitation:** Write sparsity (29%) means you lose 33% of the potential behavioral lift. This trade is worthwhile for computationally-constrained settings but not for maximum safety.

---

### C. Jailbreak Detection & Gating

#### `jailbreak_gate`
**What it does:** Recognizes prompt-level jailbreak attacks at the activation level and routes to targeted hardening only, eliminating benign over-refusal.

**Declaration:**
```yaml
monitors:
  - unit: jailbreak_gate
    target_model: gemma-2-2b | llama-3.2-1b  # models where the gap exists
    detection_threshold: 0.7

controls:
  - name: conditional_hardening
    unit: refusal_hardening
    strength: 0.8
    gate_on: jailbreak_gate  # only fire if gate detects an attack
```

**What it guarantees:**
- Attack ASR reduction: 0.50 → 0.25 (Gemma) or 0.25 → 0.083 (Llama)
- Benign over-refusal: 0.0 (zero false-positive hardening on harmless inputs)
- Prediction of success: If activation-level benign/attack gap exists at the read layer, hardening works. Otherwise, it backfires.

**Evidence source:** prabodha moat proof (Gemma, Llama); honest failures (Qwen, SmolLM)  
**Gate:** `attack_asr_reduction > 0.2` and `benign_over_refusal_ratio < 0.1` and `gap_criterion_met == true`

**Honest limitation:** This unit only works on models where benign and attack activations form a clean gap. On Qwen and SmolLM, the gap overlaps and hardening backfires. Loom refuses to deploy this unit on models where the gap test fails.

**Criterion (built-in model-dependent check):**
```python
if activation_gap_exists(model):  # gap measured at read layer
    return jailbreak_gate_compiled
else:
    raise WeaveError(
        f"Model {model} fails gap criterion. Hardening would backfire. "
        "See prabodha gates for per-model results."
    )
```

---

### D. Side-Module Steering

#### `side_module_control`
**What it does:** Trains a small neural network side-channel that reads frozen-model activations and emits steering signals (logit biases), without modifying the original weights.

**Declaration:**
```yaml
controls:
  - unit: side_module_control
    concept: refusal
    direction: amplify
    side_module_config:
      hidden_dim: 256
      training_iters: 10000
    target_model: gemma-2-2b
```

**What it guarantees:**
- Suppression/amplification accuracy: measured on held-out test set
- Zero weight modification: original model stays frozen
- Portability: side-module can be extracted and re-applied to other models with re-calibration

**Evidence source:** PWM H5a (positive on internal reward), PWM H5b (honest negative on text quality)  
**Gate:** `control_accuracy > 0.85` (e.g., suppression_ratio > 0.85 on held-out set)

**Honest limitation — CRITICAL:** PWM's H5b measured bridge-bias on a text-only scorer and found **no text-quality lift** (g = -0.47, below zero). This unit improves **control accuracy only**, not benign-prompt text quality. Do not expect text-quality gains.

**Falsifiable claim:** If benign-prompt quality drops > 10% compared to uncontrolled baseline, the control is too strong and Loom refuses the build.

---

### E. Induction (Hand-Compiled Circuit)

#### `induction_circuit`
**What it does:** Pre-compiled induction head — the model learns to copy repeated trigrams in-context without training.

**Declaration:**
```yaml
skills:
  - unit: induction_circuit
    max_copy_len: 8
    min_sequence_len: 16
```

**What it guarantees:**
- Prefix score (how much of the pattern the model copies before learning) > 0.5
- ICL loss (in-context copying loss) < 0.5
- Works on sequences where copy_len ≤ max_len - num_layers

**Evidence source:** MIabstraction E2 (hand-compiled)  
**Gate:** `prefix_score > 0.5` and `icl_loss < 0.5`

**Honest limitation:** This unit only works on sequences longer than 2×max_copy_len. On very short sequences (< 6 tokens), the prefix_score is undefined. Loom warns if you declare induction on short sequences.

---

### F. State Tracking (Belief State Monitor)

#### `belief_state_monitor`
**What it does:** Linear probe on residual stream that reads the model's internal belief state during sequential decision-making.

**Declaration:**
```yaml
monitors:
  - unit: belief_state_monitor
    world: mess3
    state_dim: 3
```

**What it guarantees:**
- Probe R² on held-out Mess3 sequences ≥ 0.9 (reads true belief, not synthetic noise)
- Probe margin ≥ 0.1 (distinct class separation)
- Transfer to new environment parameters

**Evidence source:** prabodha (corrected from REDTEAM finding 4); MIabstraction  
**Gate:** `probe_r2 > 0.9` and `probe_margin > 0.1`

**Honest limitation:** This monitor is fitted on ground-truth posteriors from the Mess3 environment, not synthetic labels. If you use a different world model, the monitor needs re-fitting.

---

## Standard Library Registry Format

Each unit in the registry is a JSON entry:

```json
{
  "name": "refusal_hardening",
  "category": "refusal",
  "version": "1.0",
  "status": "stable",
  "evidence": {
    "source": "prayoga",
    "findings": ["F1", "F2", "F23"],
    "files": [
      "/home/sharaths/projects/prayoga/docs/FINDINGS.md",
      "/home/sharaths/projects/prayoga/data/findings_registry.json"
    ]
  },
  "declaration_syntax": "...",
  "gates": [
    {
      "name": "over_refusal_ratio",
      "target": "> 0.90",
      "metric": "behavioral lift on benign prompts"
    },
    {
      "name": "random_control_ratio",
      "target": "< 0.05",
      "metric": "specificity to refusal direction"
    }
  ],
  "honest_limitations": [
    "Induces over-refusal on benign inputs (false positives)",
    "EC50 is family-dependent (~0.14 Qwen vs. ~0.24 Gemma)"
  ],
  "dual_use_route": "results/ (aggregate) + private/ (raw vectors)"
}
```

---

## Compiler Integration: From Declaration to Verification

When a user writes:

```yaml
controls:
  - unit: refusal_hardening
    strength: 0.7

gates:
  refusal_hardening:
    over_refusal_ratio: ">0.90"
    random_control_ratio: "<0.05"
```

Loom's compiler does:

1. **Lookup:** Find `refusal_hardening` in the stdlib registry
2. **Expand:** Replace the unit declaration with the full control spec (with inherited gates)
3. **Merge:** User-declared gates override defaults (e.g., higher threshold > 0.95)
4. **Build:** Compile the control via the appropriate backend (curriculum, constructed, or runtime)
5. **Verify:** Measure the gates and report pass/fail with evidence
6. **Report:** Publish `report.json` with per-unit evidence, including citations to prayoga/prabodha/PWM findings

---

## Backwards Compatibility

The standard library is **additive**, not breaking. Old weaves using explicit control specs still work:

```yaml
controls:
  - name: never_token_2
    kind: suppress
    token: 2
    strength: 0.7
```

New weaves can use the library:

```yaml
controls:
  - unit: refusal_hardening
    strength: 0.7
```

Both compile to the same runtime code; the library version just inherits evidence gates from the ledger.

---

## Future Extensions

Future stdlib units will be added as new verified mechanisms emerge:

- `truthfulness_probe` (prayoga F22 — truth-of-input feature)
- `activation_sparsity_control` (future: sparse steering)
- `model_interpretability_circuit` (future: hand-compiled circuits per concept)

Each new unit goes through the same evidence→gate→registry flow.

---

## Summary: What The Standard Library Gives a Non-Expert

1. **Names instead of mechanism details:** `refusal_hardening` instead of "ablate layer 7 by α=0.7"
2. **Inherited gates:** The unit carries its evidence from prayoga/prabodha; user doesn't rewrite verification
3. **Honest limitations:** Each unit documents its caveats (over-refusal risk, model-dependence, etc.)
4. **Dual-use routing:** Artifacts are automatically routed to public or private storage
5. **Per-model calibration:** Loom handles family-dependent dose lookup (Gemma vs. Qwen)
6. **Model-dependent gating:** Jailbreak gate refuses to deploy on models where it backfires

This is the "C standard library" for transformers: small, verified, usable without expertise.

---

## Files

- `src/loom/stdlib.py` — registry and unit expansion logic (to be implemented)
- `tests/test_loom_stdlib.py` — unit tests for each stdlib entry (to be implemented)
- `benchmarks/stdlib_gates/` — gate evidence for each unit (to be populated from sibling projects)
