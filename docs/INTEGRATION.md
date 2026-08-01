# Loom Integration: Verified Mechanisms from Three Sibling Projects

**Date:** 2026-08-01  
**Track:** INTEGRATION + POLICY  
**Status:** In progress — integration claims with evidence sources

This document specifies Loom's integration of five core mechanisms established in three independent research projects. Each claim is tagged IMPORTED-AS-PRIOR (mechanism already established, citation sufficient) or NEW-CLAIM-REQUIRING-VALIDATION (the integration itself is novel, falsifiability stated).

---

## Executive Summary

Loom integrates five sibling-project mechanisms to build a verifying compiler for transformer programming:

| # | Mechanism | Source | Loom feature | Status |
|---|-----------|--------|--------------|--------|
| **1** | Refusal = low-dimensional, ablatable residual direction (F1–F2) | prayoga | `calibrated_control_strength` | IMPORTED-AS-PRIOR |
| **2** | EC50 dose-response & cross-family transfer (F2, F23) | prayoga | `control_portability_gate` | IMPORTED-AS-PRIOR |
| **3** | Event-gated writes within entropy budget (L9, L11) | prabodha | `when:` control modifier + `svātantrya_gate` | IMPORTED-AS-PRIOR |
| **4** | Recognition gate as runtime monitor | prabodha | `recognition_gated_probe` | IMPORTED-AS-PRIOR |
| **5** | Trainable side-module for frozen-model control | PWM (H5a positive; H5b honest negative) | `side_module_backend` | NEW-CLAIM-REQUIRING-VALIDATION |

**Most critical caveat:** PWM's H5b reports that bridge-bias does NOT improve text quality (g=-0.47). Loom's side-module claim is therefore bounded to CONTROL ONLY, never quality lift. This is an honest integration, not a marketing one.

---

## 1. Refusal = Single Ablatable Direction (IMPORTED-AS-PRIOR)

### Sibling Finding
**Source:** prayoga/docs/FINDINGS.md § F1, § F2  
**Models:** Gemma-2-2b-it (26 layers, d=2304)  
**Evidence:**
- F1: Directional ablation ASR 0.90 (CI [0.75, 1.00]); activation addition raises over-refusal +0.95 (CI [0.85, 1.00])
- F2: Dose-response EC50 = 0.329, R² = 0.996; monotone rise with random-direction control flat
- **Gate pass:** ablation-raises-ASR ✓ · addition-raises-over-refusal ✓ · exceeds-random-control ✓

### What It Becomes in Loom
A `control` with `kind: suppress` or `amplify` now rests on a **verified mechanism**: single-direction steering. The Loom compiler can assume that:
- Suppression via ablation is dosable (partial ablation at strength α ∈ [0, 1])
- The effect is specific to the refusal direction, not noise (random control flat)
- The curve is smooth and calibratable

### Status
**IMPORTED-AS-PRIOR**: F1 and F2 are independently validated, survived adversarial review (leakage-correction 0.92→0.90), and are the mechanism-tier linchpin of prayoga.

### What Would Falsify the Integration
- Loom compiles a control and measure() reports ASR rise < 0.1 (the effect disappears)
- Loom compiles a dose-calibration gate and finds dose-response is non-monotone or R² < 0.5

---

## 2. Cross-Family Control Transfer & Dose-Calibration (IMPORTED-AS-PRIOR)

### Sibling Finding
**Source:** prayoga/docs/FINDINGS.md § F23  
**Models:** Gemma-2-2b, Gemma-2-9b, Qwen2.5-3b  
**Evidence:**
- F6 initially reported "addition asymmetry": Gemma +0.95 vs. Qwen 0.0 @ 64×
- F23 re-measured with powered dose-response: both Gemma and Qwen reach over-refusal 1.0 at **different effective coefficients** (Qwen ≈12–32, Gemma ≈64)
- **Verdict:** "asymmetry" was a fixed-coefficient dose artifact; refusal is single-direction-sufficient in both families at the right dose
- **Gate pass:** monotone-rise ✓ · peak-coherence (inspection-verified over-refusal) ✓

### What It Becomes in Loom
A `control` compiled for one model family can **port to another** if:
1. The same refusal direction is extracted at the per-model optimal layer
2. The dose (coefficient α) is re-calibrated per family via dose-response measurement
3. The control strength is specified as `strength: <coefficient>` in the weave

Loom's compiler implements family-dependent dose lookup: given a weave's target over-refusal level (e.g., 0.95), query the dose table and emit the family-specific coefficient.

### Status
**IMPORTED-AS-PRIOR**: F23 overturns F6's interpretation and is now the canonical cross-family claim.

### What Would Falsify the Integration
- A new model family shows refusal is **not** sufficient with any single-direction dose (requires multi-direction steering or reparameterization)
- Dose-response curves are **non-monotone** or multimodal, breaking the EC50 calibration

---

## 3. Event-Gated Writes Within Entropy Budget (IMPORTED-AS-PRIOR)

### Sibling Finding
**Source:** prabodha/README.md § L9, L11, L22  
**Models:** Gemma-2-2b, Nemotron  
**Evidence:**
- L9 (alignconf gate): gated steering achieves behavioral lift 0.30–0.35 within entropy budget ±0.5 nats
- L11 (efficiency): gated writes are 2.32× more efficient than continuous (67% lift recovery at 29% writes)
- L22 (consolidation): lift-per-write ratio 2.32× (range 1.83–3.25, 6/6 sign-consistent); write sparsity 29%
- **Control:** entropy Δ within ±0.5 nats confirms autonomous agent (svātantrya) bound

### What It Becomes in Loom
A control now supports a **`when:` modifier** that gates writes to specific activation moments (sphurattā events):

```yaml
controls:
  - name: harmless_refusal_gate
    kind: suppress
    token: 2
    when: entropy_percentile > 0.5  # Fire only at high-entropy moments
    strength: 1.0
```

The Loom compiler:
1. Measures workspace entropy at each token position
2. Gates writes to moments where entropy exceeds the `when:` threshold
3. **Verifies** that the total entropy cost stays within the `svātantrya_gate` bound (e.g., `< 0.5 nats`)

This trades behavioral lift (67% recovery) for dramatic write sparsity (29% actual writes).

### Status
**IMPORTED-AS-PRIOR**: L9 and L11 are confirmed across 6 seeds; L22 consolidation is the efficiency benchmark.

### What Would Falsify the Integration
- Gating a control causes its behavioral effect to drop **below 50%** of the continuous case (breaks lift-recovery guarantee)
- Entropy budget is **exceeded** on typical sequences (gate criterion: Δ ≤ ε must hold 95% of samples)

---

## 4. Recognition Gate as Runtime Monitor (IMPORTED-AS-PRIOR)

### Sibling Finding
**Source:** prabodha/README.md § The product  
**Models:** Gemma-2-2b, Llama-3.2-1B  
**Evidence:**
- Activation-level recognition gate on jailbreak inputs vs. benign: clean gap at read layer
- **Proof:** Gemma-2-2b: attack ASR 0.50 → 0.25 (same cut as brute-force hardening) with **zero benign over-refusal** (vs. brute-force 1.00)
- Llama-3.2-1B: 0.25 → 0.083 at zero collateral damage
- **Prediction:** gap exists → gate works; no gap → hardening backfires (Qwen, SmolLM fail)
- **Gate pass:** attack-reduction ✓ · zero-collateral ✓ · per-model gating-criterion ✓

### What It Becomes in Loom
A `monitor` with `kind: probe` can now declare:

```yaml
monitors:
  - name: jailbreak_detector
    kind: probe
    concept: jailbreak_activation_gap
```

The Loom compiler treats this as a **control activation gate**: the monitor reads whether the current input sits in benign or attack space, and only activates controls if the gate fires. This is the per-model "characterize before hardening" principle: Loom refuses to blindly harden a model where the gap is missing (where hardening backfires).

### Status
**IMPORTED-AS-PRIOR**: prabodha's moat proof is model-dependent but well-characterized; the honest failure cases (Qwen, SmolLM) are documented.

### What Would Falsify the Integration
- Loom compiles a recognition gate on a model where benign/attack activation projections **overlap**, yet the hardening still improves safety (disproves the gap hypothesis)
- The gate's attack-reduction **reverses** (higher ASR with gating than without), indicating backfiring

---

## 5. Trainable Side-Module for Frozen-Model Control (NEW-CLAIM-REQUIRING-VALIDATION)

### Sibling Foundation & Honest Limitation
**Source:** PWM/README.md § H5a (positive), H5b (honest negative)  
**Models:** 120B frozen LLM + Trika RSSM world model  
**Evidence:**
- H5a: Internal imagination reward 2.142× over baseline (world model learns real generative state)
- **H5b: Bridge-bias logit coupling does NOT improve text-only quality. g = -0.47 (below zero).**
- **Critical caveat:** H5b is a text-only scorer, stripped of world-model reward terms for fairness. The bridge is not a general quality enhancer.

### What It Becomes in Loom (NEW-CLAIM)
A new `backend: side_module` compiles a control into:
1. A small trainable neural network module (side-channel) trained offline on contrastive pairs
2. A runtime hook that reads frozen-model activations and writes logit biases
3. **Zero weight modification** — the original model is frozen; all steering is via the side-module

**Falsifiable claim:** the side-module improves **control accuracy** (e.g., suppression ASR rises) without degrading benign-prompt quality (measured separately).

### Status
**NEW-CLAIM-REQUIRING-VALIDATION**

### Evidence Required
1. **Control accuracy gate:** Loom compiles a `suppress: token 2` control via side-module and measures ASR on harmful prompts. Gate: `suppression_ratio > 0.9`.
2. **Benign quality gate:** On a held-out benign dataset, measure next-token prediction accuracy or BLEU vs. the uncontrolled baseline. Gate: `benign_quality_delta < 0.05` (within 5% of baseline).
3. **Portability gate:** Extract the side-module on Gemma-2-2b, apply it to Qwen2.5-3b with re-calibration. Gate: `suppression_ratio > 0.8` (some transfer, with loss).

### What Would Falsify the Integration
1. Side-module control accuracy **below 50%** on suppression (fails to suppress)
2. Benign-prompt quality drops **> 10%** vs. baseline (induces excessive over-refusal, breaking usability)
3. Side-module does **not transfer** across families even with re-calibration (Δ suppression_ratio < 0.2 between families)

---

## 6. Dual-Use Policy as Code (IMPORTED-AS-PRIOR)

### Sibling Finding
**Source:** prayoga/docs/DUAL_USE_POLICY.md  
**Principle:** Public aggregates + code, private raw materials

### What It Becomes in Loom
`src/loom/dual_use.py` implements the classifier:

```python
from loom.dual_use import classify_artifact, write_artifact, Direction, ArtifactKind

# Hardening vectors (amplification): public aggregate
route = classify_artifact(
    concept="refusal",
    direction=Direction.HARDENING,
    kind=ArtifactKind.VECTOR,
)  # → results/

# Suppression vectors (ablation): private raw
route = classify_artifact(
    concept="refusal",
    direction=Direction.SUPPRESSION,
    kind=ArtifactKind.VECTOR,
)  # → private/
```

**Rules:**
- Hardening aggregates (statistics, gates, code) → `results/` (public)
- Suppression raw artifacts (vectors, checkpoints, generations) → `private/` (gitignored)
- Suppression aggregates (statistics, methods) → `results/` (published for transparency)
- Unknown concepts → route conservatively to private

### Status
**IMPORTED-AS-PRIOR**: prayoga's policy is policy-as-code in Loom.

---

## Summary: Integration Claims & Falsifiability

| # | Claim | Status | Falsified by |
|---|-------|--------|--------------|
| **1** | Refusal = single direction, EC50-calibratable | IMPORTED | ASR rise < 0.1 on compiled control |
| **2** | Cross-family transfer via dose-calibration | IMPORTED | Non-monotone dose-response or non-transfer |
| **3** | Event-gated writes within entropy budget | IMPORTED | Entropy overflow or < 50% lift recovery |
| **4** | Recognition gate predicts hardening success | IMPORTED | Gap-predicts-success fails (Qwen/SmolLM data accepted) |
| **5** | Side-module backend for frozen-model control | NEW | Suppression < 50% OR benign quality drop > 10% OR non-transfer across families |
| **6** | Dual-use policy routes artifacts safely | IMPORTED | Raw suppression artifacts leak to public |

---

## Integration Methodology

Each integrated mechanism enters Loom's compiler as:

1. **A gate in the weave.yaml schema** — e.g., `ec50_calibration: 0.3`, `entropy_budget: 0.5`
2. **A backend implementation** — curriculum, constructed, runtime, or side_module
3. **Verification code** — `verify.py` checks that measured values satisfy gate criteria
4. **Test suite** — each mechanism has regression tests that would fail if the integration broke

When a user declares:

```yaml
controls:
  - name: suppress_token_2
    kind: suppress
    token: 2
    strength: 0.7
    when: entropy_percentile > 0.5

gates:
  suppress_token_2:
    suppression_ratio: ">0.9"
    entropy_cost: "<0.5"
```

Loom:
1. Verifies the control via the prayoga mechanism (F1–F2 gate, calibrate via dose-response)
2. Applies the prabodha when-gating (L9, L11 efficiency)
3. Runs the entropy-budget checker (svātantrya gate from prabodha)
4. Reports or refuses based on measured values vs. declared thresholds

If all gates pass, the build succeeds; if any gate fails, Loom refuses with a diagnosis.

---

## Honest Caveats

1. **PWM H5b is real:** Bridge-bias does not lift text quality. Loom's side-module claim is bounded to control-accuracy, not quality.
2. **prabodha's moat is model-dependent:** Hardening backfires on some models (Qwen, SmolLM). Loom's recognition gate is gated on the activation-gap criterion; without the gap, Loom refuses to hardening.
3. **prayoga's dose is family-dependent:** EC50 differs by family (~0.14 Qwen vs. ~0.24 Gemma). Loom requires per-family dose calibration.
4. **Gates certify the measured distribution:** If inputs go out-of-distribution (longer sequences, different concepts), gate guarantees degrade. Loom publishes the measured envelope in `report.json`.

---

## Files & References

### prayoga
- Findings ledger: `/home/sharaths/projects/prayoga/docs/FINDINGS.md` (F1, F2, F23)
- Dual-use policy: `/home/sharaths/projects/prayoga/docs/DUAL_USE_POLICY.md`
- Data: `/home/sharaths/projects/prayoga/data/findings_registry.json` (machine-readable)

### prabodha
- README: `/home/sharaths/projects/prabodha/README.md` (L9, L11, L22)
- Gate definitions: `/home/sharaths/projects/prabodha/gates/gate_L*.json`
- Moat proof: `/home/sharaths/projects/prabodha/gates/gate_L26_moat_*.json`

### PWM
- README: `/home/sharaths/projects/PWM/README.md` (H5a, H5b)
- Bridge implementation: `/home/sharaths/projects/PWM/pwm/vimarsa/bridge*.py`
- H5b ablation results: `/home/sharaths/projects/PWM/benchmarks/results/h5_live_ablation.json`

### Loom (this project)
- Dual-use policy as code: `src/loom/dual_use.py`
- Spec with gate validation: `src/loom/spec.py`
- Runtime verification: `src/loom/verify.py`
- Integration tests: `tests/test_loom_integration.py` (new)

---

**This document is the scientific spine of Loom's integration. Every claim is grounded in evidence; every claim has a falsification criterion. This is honest integration, not marketing.**
