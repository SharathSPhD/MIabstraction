# Loom Redteam Report: Adversarial Audit of Gate Promises

**Date:** 2026-08-01  
**Target:** Loom transformer programming layer (v0)  
**Branch:** loom/redteam  
**Audit Type:** Adversarial red-team attack on gate verification system

---

## Executive Summary

Loom's core promise is: **"A weave declares WHAT the model must do, and the compiler verifies it with gates before the build succeeds."** This is marketed as a guarantee to non-experts.

The audit finds **5 findings** across 5 attack surfaces, ranging from MED to HIGH severity:

1. **FINDING 1 (HIGH):** Vacuous gates (e.g., `>-1`) are accepted and always pass
2. **FINDING 2 (MED-HIGH):** Constructed circuits degrade silently on edge-case sequences
3. **FINDING 3 (HIGH):** Suppression controls can be evaded by semantic equivalents
4. **FINDING 4 (HIGH):** Belief-state monitor fitted to synthetic labels, measures nothing
5. **FINDING 5 (MED):** Logit-mask fallback mechanism not implemented, silent control failure

**Verdict:** Loom's gate promise is **currently dishonest**. The layman's guarantee is false. A weave with gates can pass verification despite failing to deliver the promised capabilities.

---

## Findings Table

| Finding | Severity | Issue | One-Liner Fix | Test |
|---------|----------|-------|---------------|------|
| 1A | HIGH | Vacuous gates `>-1` always pass | Validate thresholds against metric semantics in spec.py | `test_finding_1_spec_attacks.py::test_gate_greater_than_negative_one_always_passes` |
| 1B | MED | Negative R² thresholds nonsensical | Reject thresholds outside [0,1] for normalized metrics | `test_finding_1_spec_attacks.py::test_negative_threshold_gates_are_allowed` |
| 2A | MED-HIGH | Sequences < 3 tokens have no trigram context | Document in constructed backend that prefix_score is undefined for len < 3 | `test_finding_2_constructed_edge_cases.py::test_very_short_sequences_no_trigram` |
| 2B | MED | All-same-token sequences pass silently | Measure accuracy separately for degenerate cases | `test_finding_2_constructed_edge_cases.py::test_all_same_token_sequence` |
| 2C | MED | Long copy_len exceeds lookback window | Warn when copy_len > max_len - n_layers | `test_finding_2_constructed_edge_cases.py::test_copy_len_longer_than_lookback` |
| 2D | MED | Repeated bigrams cause attention ambiguity | Measure accuracy on self-collision test | `test_finding_2_constructed_edge_cases.py::test_repeated_bigram_collisions` |
| 3A | HIGH | Model expresses suppressed token semantically | Measure downstream semantic equivalence, not just logits | N/A |
| 3B | HIGH | side_effect metric doesn't detect evasion | Use contrastive distributions (with/without trigger) | N/A |
| 4D | **CRITICAL** | belief_state monitor fitted to random synthetic labels | Use actual ground-truth environment labels (Mess3 posterior) | `test_finding_4_monitor_trust.py::test_synthetic_labels_produce_low_r2` |
| 5A | MED | logit_mask mechanism not implemented in ControlledModel | Add logit-masking hook alongside steering hook | `test_finding_5_unimplemented_fallback.py::test_logit_mask_mechanism_not_implemented` |
| 5B | MED | Steering computation failure silently disables control | Fall back to logit_mask or fail the build explicitly | `test_finding_5_unimplemented_fallback.py::test_steering_vector_none_causes_silent_failure` |

---

## Detailed Findings

### FINDING 1: Vacuous Gates (HIGH)

**Attack Surface:** Spec parsing and gate validation  
**Vulnerability:** The gate regex accepts thresholds that are vacuous (always true/false) or nonsensical.

```python
_GATE_RE = re.compile(r"^\s*([<>])\s*=?\s*([-+0-9.eE]+)\s*$")
```

Examples of vacuous gates that pass validation:
- `prefix_score: ">-1"` — any non-negative score passes (almost all real values pass)
- `side_effect: "<999"` — any realistic side effect passes
- `probe_r2: ">-0.5"` — R² should be in [0, 1], but negative thresholds are accepted

**Impact:** The weave author can declare thresholds that the build is guaranteed to pass. The gate becomes theater — it promises nothing.

**Severity:** HIGH — the layman's guarantee is false.

**Reproduction:**
```bash
pytest tests/redteam/test_finding_1_spec_attacks.py::test_gate_greater_than_negative_one_always_passes -v
```

**Fix Recommendation:**
1. Add metric-aware threshold validation in `spec.py::parse_gate_expr()`
2. For `>` gates on normalized metrics [0,1]: reject threshold < 0 or > 1
3. For `<` gates on side_effect: reject threshold < 0 or > 1
4. For `>` gates on suppression_ratio: reject threshold < 0 or > 1

---

### FINDING 2: Constructed Circuit Edge Cases (MED-HIGH)

**Attack Surface:** Constructed backend trigram induction circuit  
**Vulnerability:** The trigram matcher degrades silently on edge cases with no clear diagnosis.

Edge cases discovered:

| Case | Behavior | Metric Impact |
|------|----------|---------------|
| Sequences len < 3 | No trigram context, random predictions | prefix_score undefined |
| All-same-token [k,k,k,...] | Trigram (k,k,k) repeats everywhere, attention ambiguous | icl_argmax_acc still high (lucky) |
| Repeated bigrams [1,2,1,2,...] | Bigram (1,2) repeats, trigram breaks ties → all position 1 matches | accuracy degrades unpredictably |
| copy_len > max_len - layers | Attention can't reach source positions | accuracy drops to near random |

**Impact:** The compiled model claims high accuracy on these edge cases, but behavior is undefined or adversarial. A weave author asking for induction on short sequences or repeated patterns gets false confidence.

**Severity:** MED-HIGH — accuracy metrics are misleading on edge cases.

**Reproduction:**
```bash
pytest tests/redteam/test_finding_2_constructed_edge_cases.py -v
```

**Fix Recommendations:**
1. Add validation in `spec.py` to reject induction skills with copy_len > max_len
2. Document that prefix_score is only meaningful for sequences len > 6
3. Add diagnostic metrics: measure accuracy separately on degenerate cases
4. Measure accuracy only on the "safe" range of the sequence

---

### FINDING 3: Control Evasion (HIGH)

**Attack Surface:** Suppress/amplify controls and suppression_ratio metric  
**Vulnerability:** Suppression masks logits but doesn't change the model's underlying intent.

A model learned to emit `token=2` for concept X can be trained to emit `token=3` instead if that token is masked. The suppression_ratio gate still passes (low P(2)), but the concept is still expressed.

Moreover, the side_effect measurement uses random test sequences. If the model's learned behavior is context-dependent (e.g., only emit token 2 after token 1), the suppression might not hurt on random sequences where token 1 never appears.

**Impact:** A weave declaring "never emit token 2" can still express the semantics of token 2 through other tokens. The gate is bypassed.

**Severity:** HIGH — the control doesn't actually constrain behavior.

**Reproduction:**
```bash
pytest tests/redteam/test_finding_3_control_evasion.py -v
```

**Fix Recommendations:**
1. Measure suppression on **semantically equivalent tokens** (requires token embeddings or external semantic model)
2. Use **contrastive side_effect evaluation**: measure loss on sequences that trigger the learned behavior vs. sequences that don't
3. Document that suppression_ratio only guarantees logit suppression, not semantic suppression

---

### FINDING 4: Belief-State Monitor Fitted to Synthetic Labels (CRITICAL)

**Attack Surface:** Monitor fitting in `monitors.py`  
**Vulnerability:** The belief_state monitor is fitted to RANDOM labels, not ground truth.

```python
# From monitors.py line 105
if monitor.concept == "belief_state":
    y = np.random.dirichlet([1, 1, 1], size=B * L)  # SYNTHETIC!
    fit_result = regression_probe(X, y, val_frac=0.2, seed=0)
    quality = {"r2_val": fit_result["r2_val"]}
```

The probe is fitted to random belief states, not the true posterior from the Mess3 environment. This means:
- The probe measures the model's ability to fit random noise
- R² on synthetic labels is meaningless
- The monitor doesn't read belief state; it measures overfitting

**Impact:** A weave author declares `probe_r2: >0.9` expecting the monitor to read the model's belief state. In reality, the gate just checks if the probe can overfit to noise.

**Severity:** CRITICAL — the gate promise is false. The layman thinks the monitor works; it doesn't.

**Reproduction:**
```bash
pytest tests/redteam/test_finding_4_monitor_trust.py::test_probe_with_synthetic_labels_is_suspicious -v
```

**Fix Recommendation:**
1. Generate ground-truth belief states from the Mess3 environment (sample from posterior)
2. Fit probe on true posterior, not random dirichlet
3. Verify probe R² on held-out Mess3 sequences with known transitions
4. Document the ground-truth source in the monitor's fit_quality

---

### FINDING 5: Unimplemented Logit-Mask Fallback (MED)

**Attack Surface:** Runtime control fallback mechanism  
**Vulnerability:** The ControlledModel only handles `mechanism=="steering"`. If steering_vector computation fails, the control is silently disabled.

From `runtime.py::install_controls()`:
```python
if steering_vec is not None:
    mechanism = "steering"
else:
    mechanism = "logit_mask"  # Fallback — but ControlledModel doesn't handle this!
```

From `runtime.py::_install_hooks()`:
```python
for control in self.controls:
    if control.mechanism == "steering" and control.steering_vector is not None:
        # Install hook
    # else: do nothing! logit_mask is not handled
```

**Impact:** When CAA steering_vector computation fails (e.g., insufficient positive/negative samples), the control falls back to logit_mask. But logit_mask is never applied. The control is silently inactive.

The weave author thinks the suppress control is active. It's not.

**Severity:** MED — safety mechanism silently fails.

**Reproduction:**
```bash
pytest tests/redteam/test_finding_5_unimplemented_fallback.py -v
```

**Fix Recommendations:**
1. Implement logit_mask hook: add hook to the unembedding layer that sets `logits[:, :, token] = -inf`
2. Or: fail the build with a clear diagnosis if steering_vec computation fails
3. Do NOT silently disable the control

---

## Summary of Attack Surfaces

| Surface | Findings | Status |
|---------|----------|--------|
| GATE GAMING (spec validation) | 1A, 1B | Gates accept vacuous thresholds |
| CONSTRUCTED CIRCUITS | 2A-2D | Degradation undetected on edge cases |
| CONTROL EVASION | 3A, 3B | Semantic evasion not measured |
| MONITOR TRUST | 4D | Fitted to synthetic noise, not ground truth |
| SPEC ATTACKS | 1 | (covered above) |
| FALLBACK MECHANISMS | 5A, 5B | Unimplemented, silently disabled |

---

## Honest Assessment: Can Loom's Gate Promise Be Trusted?

**No.** The current implementation does not deliver on its core promise:

> "The compiler refuses to build capabilities it is not required to verify."
> (From README.md)

Counter-evidence:
- **FINDING 1:** Gates can be vacuous and always pass (e.g., `>-1`)
- **FINDING 4:** The belief_state monitor measures overfitting to noise, not state tracking
- **FINDING 5:** Safety controls can be silently disabled with no error
- **FINDING 3:** Suppression can be evaded through semantic equivalents

A weave with gates passes verification despite failing to deliver the promised behavior.

### What a Layman Believes
> "I've declared the gates. The compiler verified them. The model definitely has these capabilities."

### What's Actually True
> "The gates may be vacuous, the monitors may be measuring noise, and the controls may be inactive. You got the verification theater without actual verification."

---

## Most Important Fix

**PRIORITY 1 (FIX IMMEDIATELY):** Fix FINDING 4 — the belief_state monitor is currently useless.

Currently, the probe is fitted to `np.random.dirichlet()`, which has no correlation with the model's actual internal state. This makes the gate completely meaningless. Replace with:

```python
# Instead of random labels:
from miabstraction.worlds import Mess3

# Generate true belief states from the environment
world = Mess3(p_x=spec.p_x, p_a=spec.p_a)
true_posteriors = []
for seq in calib_sequences:
    posterior = world.posterior_given_observations(seq)
    true_posteriors.append(posterior)

y = np.array(true_posteriors)  # True ground truth, not random noise
fit_result = regression_probe(X, y, val_frac=0.2, seed=0)
```

This single fix elevates FINDING 4 from "CRITICAL gate is theater" to "monitor measures actual belief state."

---

## Recommendations by Priority

### P0 (Ship-blocker)
- [ ] Fix FINDING 4: Replace synthetic labels with ground-truth posteriors
- [ ] Implement or remove FINDING 5: Either implement logit_mask or fail the build

### P1 (Critical)
- [ ] Fix FINDING 1: Validate gates against metric semantics (reject `>-1`, `<999`, etc.)
- [ ] Fix FINDING 3: Document semantic evasion risk; measure contrastive side_effect

### P2 (Important)
- [ ] Fix FINDING 2: Add edge-case diagnostics; warn on degenerate sequences
- [ ] Add cross-metric consistency checks: suppress_ratio + side_effect should agree on causality

### P3 (Nice-to-have)
- [ ] Add random seed validation (reproducibility of steering_vector, probes)
- [ ] Document precision loss in gates (floating-point comparisons)

---

## Test Execution

All redteam tests are in `tests/redteam/`:

```bash
cd /home/sharaths/projects/MIabstraction-redteam

# Run all findings
pytest tests/redteam/ -v

# Run specific findings
pytest tests/redteam/test_finding_1_spec_attacks.py -v
pytest tests/redteam/test_finding_2_constructed_edge_cases.py -v
pytest tests/redteam/test_finding_3_control_evasion.py -v
pytest tests/redteam/test_finding_4_monitor_trust.py -v
pytest tests/redteam/test_finding_5_unimplemented_fallback.py -v
```

All tests are marked as `@pytest.mark.xfail(reason="...")` for genuine unfixed holes. Passing tests document the bugs; xfailing tests indicate they're known issues.

---

**Report Generated By:** Claude Code Red-Team Agent  
**Branch:** loom/redteam  
**Ready for:** Handoff to main team for fixes
