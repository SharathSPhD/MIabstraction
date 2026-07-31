# MIabstraction — Specification

**Goal:** empirically discover and validate the *abstraction layer* of transformer LLMs,
treating the research synthesis in `docs/` as the hypothesis source. The central claim under
test (from `docs/research1.md`):

> The transformer abstraction layer is not a single clean "ISA" but a **plural, leaky,
> multi-layer stack**: weights → parameter components → features → circuits → algorithms →
> behaviors — where each arrow leaks (Spolsky), leaks must be *budgeted* not eliminated, and
> the strongest layers are weight-structure and representation geometry, not SAE features.

This spec is the contract: every experiment is config-driven (YAML in `configs/`), test-driven
(pytest in `tests/`), and produces machine-readable results (`results/<exp>/result.json`) that
update the hypothesis registry (`results/hypotheses.json`).

## Hypotheses (falsifiable claims)

| ID | Layer | Claim | Experiment | Falsified if |
|----|-------|-------|-----------|--------------|
| H1 | Representation geometry | A transformer trained on an HMM (Mess3) linearly embeds the belief-state simplex of the data process in its residual stream. | E1 | Linear probe R² from residual stream to ground-truth belief states < 0.7 (train/val split), or no better than an untrained-model control. |
| H2 | Circuits / development | Induction capability forms as a discrete phase transition, visible as a sharp rise in prefix-matching score co-timed with an in-context-learning loss drop. | E2 | Prefix-matching score rises only gradually (no identifiable transition window < 20% of training) or never exceeds 0.6. |
| H3 | Features (SAE reckoning) | Standard SAE quality metrics (reconstruction, sparsity, downstream CE) do **not** clearly separate SAEs trained on a trained model vs a randomly-initialized one. | E3 | Metrics separate trained vs random cleanly (> 3σ on all metrics) — which would *rehabilitate* SAE metrics. Either outcome is informative. |
| H4 | Features vs baselines | For a *known* concept, a logistic-regression probe on raw activations matches or beats an SAE-feature probe. | E4 | SAE probe beats raw-activation probe by > 5 pts accuracy consistently. |
| H5 | Weights (imposed ISA) | Training with weight sparsity yields a *smaller, more faithful* minimal circuit for an algorithmic task than a matched dense model, at similar task performance. | E5 | Sparse-model circuit is not smaller, or is smaller but unfaithful (mean-ablation sufficiency < dense). |

Verdict semantics: each experiment writes `supports: true/false`, effect sizes, and a
*leak budget* (quantified failure fraction) into its result.json. `VALIDATION.md` aggregates.

## Architecture

- `src/miabstraction/` — library code
  - `config.py` — YAML → dataclass experiment configs (single schema, versioned)
  - `models.py` — small hook-instrumented decoder-only transformer (from scratch, no HF dependency for core; HF models optional)
  - `data/` — generators: `mess3.py` (HMM + mixed-state presentation), `induction.py`, `algo.py`
  - `sae.py` — top-k sparse autoencoder
  - `probes.py` — linear/logistic probes with train/val protocol
  - `circuits.py` — ablation-based minimal-circuit extraction, faithfulness metrics
  - `design.py` — active-inference experiment selector: hypothesis registry with priors,
    expected-information-gain scoring of candidate experiments
  - `runner.py` — `python -m miabstraction.runner configs/eX.yaml` → results/
- `configs/` — one YAML per experiment; *all* hyperparameters live here, none in code
- `tests/` — pytest; every module has tests written before/with implementation
- `results/` — run artifacts (json + png), committed for provenance
- `docs/` — research inputs and syntheses

## Constraints

- Runs on a single NVIDIA GB10 (aarch64, unified memory); venv: `.venv` (links CUDA torch from `~/.venvs/prabhasa-gb10`).
- Determinism: every config carries `seed`; results record git SHA + config hash.
- No experiment may exceed ~30 GPU-minutes without checkpointing.

## Method stack (process, not code)

- **Spec-driven:** this file is the source of truth; changes to claims require a commit touching SPEC.md.
- **Config-driven:** experiments are fully described by YAML; the runner takes no other input.
- **TDD:** red → green on every module.
- **Active inference:** `design.py` selects next experiment by expected information gain over H1–H5 posteriors.
- **TRIZ:** the core engineering contradiction — *interpretability vs capability* (and *imposed vs discovered structure*) — is analyzed with the TRIZ engine; inventive principles feed variant experiments.
- **Attractor-flow:** divergent exploration when an experiment stalls; convergence mode to close.
- **Ralph loop:** iterate until all tests pass and every hypothesis has a grounded verdict.
