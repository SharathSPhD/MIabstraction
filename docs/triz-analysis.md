# TRIZ Analysis: Transformer Interpretability Engineering Contradictions

## Executive Summary

The MIabstraction project faces three core engineering contradictions in mechanistic interpretability research. Using TRIZ (Theory of Inventive Problem-Solving), we map these contradictions to engineering parameters, retrieve recommended inventive principles from the TRIZ 40 matrix, and translate them into five concrete, testable experiment proposals. The top three proposals are scored against the Ideal Final Result (IFR) criterion.

---

## Contradiction 1: Interpretability vs. Model Capability

**Engineering Frame:**
- **Improving:** Interpretability / Observability (TRIZ Parameter 18: Illumination Intensity)
  - Ability to explain model decisions and reverse-engineer computations
  - Transparency of internal mechanisms and representations
  
- **Worsening:** Model Capability / Measurement Accuracy (TRIZ Parameter 28: Measurement Accuracy)
  - Model performance, accuracy, and downstream task quality
  - Loss reduction and prediction fidelity

**Problem Statement:**
As we impose sparsity, weight constraints, or clean architectural priors to make models interpretable, we typically pay a capability cost: loss increases, downstream accuracy drops, or the model must be substantially smaller. The classical tradeoff.

**TRIZ Matrix Recommendation:**
Cell (18, 28) → Principles: **11 (Beforehand Cushioning), 15 (Dynamics), 32 (Color Changes)**

---

### Principle 11: Beforehand Cushioning
*Prepare emergency means in advance so when disruption occurs recovery is cheap and controlled.*

**Experiment Proposal 1A: Adaptive Sparsity Budget with Capability-Aware Annealing**

*Rationale:* Instead of fixing a global sparsity ratio (e.g., 1-in-1000 weights nonzero), assign a *per-layer sparsity budget* that is initially loose and anneals only after a capability checkpoint is reached. Each layer gets emergency budget slack; if loss diverges, the scheduler recovers budget automatically.

*Config Parameters:*
- Base sparsity target (e.g., 95%)
- Per-layer budget allocation strategy (uniform vs. adaptive by layer diameter)
- Capability threshold (e.g., "halt annealing until val-loss matches dense baseline")
- Budget recovery schedule (e.g., add 5% slack to any layer where loss-delta exceeds threshold)

*Measurement:*
- Plot: Sparsity (X) vs. downstream accuracy and interpretability-circuit fidelity (Y) over training
- Success metric: Sparse model matches dense baseline accuracy *and* circuit fidelity > 0.8 (mean-ablation necessity)

---

### Principle 15: Dynamics
*Make characteristics adaptive or self-tuning so the system finds better operating points as conditions change.*

**Experiment Proposal 1B: Online Constraint Temperature Scheduling**

*Rationale:* Treat interpretability constraints (sparsity, top-k gating, weight-structure priors) as temperature-controlled systems. Early in training, use low temperature (tight constraints) to bias the learner toward modular structure; as training progresses and loss stabilizes, increase temperature (relax constraints) to recover capability. This is a *dynamic trade-off knob* that adapts to training dynamics.

*Config Parameters:*
- Initial constraint temperature (e.g., high sparsity penalty)
- Cooling/heating schedule (e.g., inverse-sigmoid annealing)
- Loss-stability detector (e.g., window-based variance)
- Capability recovery threshold (e.g., if loss plateaus, increase temperature)

*Measurement:*
- Track: Constraint temperature, sparsity, loss, and circuit fidelity over training steps
- Success metric: Sparse model achieves > 95% of dense-baseline accuracy while maintaining sparsity > 95% and circuit discovery rate > 60%

---

### Principle 32: Color Changes
*Change color, transparency, or optical properties to signal state, reduce glare, or reveal hidden information.*

**Experiment Proposal 1C: Per-Module Interpretability-Capability Scorecard**

*Rationale:* Train two parallel SAEs (or transcoders) on the sparse and dense models' activations. Score each module (attn / mlp layer) on three dimensions: sparsity achieved, feature-reconstruction quality, and downstream-loss contribution. Display as a live heatmap where each cell is colored by the dominance of interpretability vs. capability in that module. This reveals which layers are being "sacrificed" and which are resilient, informing targeted constraint relaxation.

*Config Parameters:*
- Module types to track (layer-wise: attn, mlp; or finer: qkv, out proj, etc.)
- SAE hyperparameters (dict size, sparsity)
- Scorecard dimensions (weights: sparsity, reconstruction, loss-delta)

*Measurement:*
- Generate heatmaps for every N training steps
- Identify "divergence regions" (modules where capability drops most sharply with sparsity increase)
- Success metric: Heatmap analysis reveals ≥3 distinct module clusters (high-resilience, balanced, high-sacrifice); targeted relaxation in high-sacrifice layers recovers > 5% accuracy

---

## Contradiction 2: Imposed vs. Discovered Structure

**Engineering Frame:**
- **Improving:** Imposed Structure / Design Enforced Modularity (TRIZ Parameter 12: Shape)
  - Designed architectural constraints (e.g., sparsity patterns, block structure)
  - Clean interfaces, enforced module boundaries, deterministic routing
  
- **Worsening:** Discovered Structure / Faithful Representation (TRIZ Parameter 24: Loss of Information)
  - Fidelity to actual learned computation
  - Structural authenticity—do the discovered circuits *actually* explain how the model computes?

**Problem Statement:**
Weight-sparse models (Gao et al. 2025) and parameter-decomposition methods (Apollo) yield interpretable circuits—but an Anthropic-Fellows replication showed they can be "interpretable yet unfaithful." Imposing structure (sparsity, decomposition) creates clean-looking ISAs that don't reflect true computation. Conversely, fully discovered structure (from pure training) is leaky and messy but authentic.

**TRIZ Analysis:**
Relevant principles: **1 (Segmentation), 2 (Taking Out), 3 (Local Quality)**

---

### Principle 1: Segmentation
*Divide the object into independent parts so each segment can be optimized, replaced, or scaled on its own.*

**Experiment Proposal 2A: Two-Stage Structure Validation**

*Rationale:* Instead of imposing sparsity from scratch, first train a dense model, reverse-engineer its circuits (via circuit tracing or perturbation analysis), extract the *empirically discovered* minimal subgraph, and *then* impose that structure (freeze those weights, zero others). This makes imposed structure faithfulness-testable: if freezing the discovered subgraph preserves loss, the imposed structure is faithful.

*Config Parameters:*
- Stage 1: Train dense model on task (e.g., induction, algorithmic task)
- Stage 2: Extract minimal circuit via:
  - Causal-intervention approach (ablate individual weights, measure loss delta)
  - OR activation-pattern clustering (SAE / transcoder to find key activation paths)
- Stage 3: Freeze discovered circuit, zero remaining weights
- Stage 4 (optional): Fine-tune frozen architecture on different task to test transferability

*Measurement:*
- Success metric 1: Stage 3 fine-tuning loss matches Stage 1 loss within 2%
- Success metric 2: Circuit subgraph size < 10% of dense model weights
- Success metric 3: Transfer to held-out task: accuracy drop < 5% (faithfulness of discovered structure)

---

### Principle 2: Taking Out
*Extract a disturbing element or isolate only what is necessary so the core system stays stable.*

**Experiment Proposal 2B: Superposition-Aware Sparse Decomposition**

*Rationale:* Before imposing weight sparsity, use a parameter-decomposition method (Apollo / SPD style) to separate features residing in superposition. This "Taking Out" step recovers hidden structure that pure weight sparsity would destroy or hide. Then impose sparsity *on the decomposed feature space* rather than the raw weights, ensuring imposed structure aligns with discovered features.

*Config Parameters:*
- Decomposition method: APD (Attribution-based Parameter Decomposition) or SPD (Stochastic Parameter Decomposition)
- Target: Decompose each weight matrix into K rank-1 (or low-rank) component features
- Sparsity target: Applied to component activations, not raw weights
- Fidelity threshold: Only keep components whose ablation changes loss by > threshold

*Measurement:*
- Component count: Should be 10-50x smaller than neuron count (true feature extraction)
- Faithfulness: Ablation-based necessity of top-K components (should be > 0.9 cumulative for top 10%)
- Circuit recovery: Extracted component circuits < 5% weight count, fidelity > 0.8
- Success metric: Sparse-decomposed model matches dense baseline accuracy; component-based circuits are simpler and more faithful than weight-sparse circuits

---

### Principle 3: Local Quality
*Move from uniform structure to differentiated structure so each region operates under conditions best suited to its role.*

**Experiment Proposal 2C: Layer-Adaptive Structure Imposition**

*Rationale:* Not all weights are equal. QK-circuits (attention patterns) carry information about *what* to attend to and are hard to reverse-engineer; they benefit from lighter constraint. Value-head circuits and MLP output projections are more linear and decomposable; they can tolerate tighter sparsity. Implement a *layer-role-aware* sparsity scheduler that assigns different target sparsity levels to different module types based on their functional classification.

*Config Parameters:*
- Layer classification: attention-qk, attention-v, attention-out, mlp-up, mlp-down, mlp-out (etc.)
- Functional role inference: Use intrinsic dimensionality, SAE-feature counts, or ablation-based importance scoring to classify roles
- Per-role sparsity budget: e.g., qk:50%, v:85%, mlp-up:95%, mlp-out:80%
- Sparsity enforcement: Top-k weights or magnitude threshold, layer-specific

*Measurement:*
- Sparsity variance across layers: Should cluster into role-based groups
- Loss impact: Per-layer loss sensitivity to sparsity (measure via layer-wise ablation at different sparsity targets)
- Success metric: Non-uniform sparsity achieves comparable accuracy to uniform sparsity *with lower average sparsity* (e.g., 90% uniform → 87% average non-uniform, same accuracy)

---

## Contradiction 3: Abstraction Usefulness vs. Discovery Power

**Engineering Frame:**
- **Improving:** Abstraction Usefulness / Actionability on Known Concepts (TRIZ Parameter 35: Adaptability)
  - Ability to probe, steer, and act on pre-specified features and behaviors
  - Efficacy of behavioral control (e.g., refusal directions, steering)
  
- **Worsening:** Discovery Power / Novelty Exploration (TRIZ Parameter 18 proxy: Conceptual Reach)
  - Ability to find unknown concepts and hidden objectives not specified in advance
  - Exploratory capacity and surprise-factor in discovered mechanisms

**Problem Statement:**
Research 1.md notes: "interp's value is currently *concentrated* in discovery/auditing and *thin* in acting-on-known-concepts." SAE-feature probes underperform simple linear probes on known-concept steering; but SAEs excel at finding *novel* features. The tradeoff: build a tool optimized for known-concept control, and it becomes brittle for discovery. Build for novelty, and it fails on reproducible steering tasks.

**TRIZ Matrix Recommendation:**
Cell (35, 18) → Principles: **6 (Universality), 22 (Blessing in Disguise), 26 (Copying), 1 (Segmentation)**

---

### Principle 6: Universality
*Design a part or subsystem to perform multiple functions so fewer specialized components are needed.*

**Experiment Proposal 3A: Dual-Mode Adaptive Probe**

*Rationale:* Instead of separate tools (linear probes for control, SAEs for discovery), build one probe system that is:
1. **Known-concept mode:** Fast linear regression on any pre-specified behavioral axis (refusal, honesty, power-seeking)
2. **Discovery mode:** Online SAE training on activations not explained by the known-concept probes, surfacing residual latent factors

The system is *universal* because it serves both roles; it's *adaptive* because it learns which features are "known" vs. "novel" online.

*Config Parameters:*
- Known-concept: List of pre-labeled behavioral dimensions (e.g., refusal_score, honesty_score) with ground-truth labels
- Discovery target: Residual unexplained by known-concept probes
- Online adaptation: Add newly discovered features to the "known" set as they stabilize across seeds

*Measurement:*
- Known-concept performance: Probe accuracy on held-out refusal/honesty/jailbreak detection vs. baseline
- Discovery power: Number of novel discovered features per dataset, their prevalence and interpretability rating
- Success metric: Dual-mode probe matches linear-probe accuracy on known concepts *and* discovers ≥5 novel features with > 80% cross-seed reproducibility

---

### Principle 22: Blessing in Disguise
*Turn harmful factors into useful ones by harnessing side effects, noise, or failures as signals and drivers of improvement.*

**Experiment Proposal 3B: Failure-Driven Discovery via Probe Disagreement**

*Rationale:* Train ensemble of probes (linear, MLP, SAE) on the same known-concept task. When they disagree sharply, rather than averaging them out, *investigate the disagreement*. Disagreement signals that the feature space is complex or contains hidden concepts. Use disagreement regions to prioritize SAE training and feature discovery—allocate more resources to activations where probes diverge.

*Config Parameters:*
- Ensemble: 5 probe types (linear, logistic, MLP-2layer, SAE-topk, SAE-gated)
- Disagreement metric: Pairwise KL divergence or confidence variance across ensemble
- Discovery allocation: Rank activations by disagreement; run SAE discovery on top-K disagreement regions
- Boosting: Each probe iteration re-weights to reduce disagreement on future batches

*Measurement:*
- Disagreement distribution: Histogram of probe variance across activation space
- Discovery yield: Novel features found in high-disagreement vs. low-disagreement regions (should be 3-5x higher)
- Success metric: Failure-driven discovery finds ≥3x more novel features than random SAE allocation; novel features match ground-truth concept labels (if available) > 70% of the time

---

### Principle 26: Copying
*Use inexpensive or simplified copies when originals are inaccessible, risky, or costly to move.*

**Experiment Proposal 3C: Lightweight Proxy Probes**

*Rationale:* Train a full SAE on the model's activations (discovery-focused, computationally expensive, high-fidelity). Then train fast linear probes (copies) that predict:
1. SAE latent activations (copy of feature space)
2. Known behavioral dimensions (copy of control axis)

Deploy the lightweight proxies in inference; keep the reference SAE for periodic recalibration. This gives you two benefits: (a) known-concept probes with low latency, (b) SAE reference for discovery mode and validation.

*Config Parameters:*
- Reference SAE: Full dictionary size, sparsity target (e.g., 1000 latents, 99% sparse)
- Lightweight copies: Linear regressors on raw activations to predict top-K SAE latents
- Validation: Periodically compare proxy predictions to reference; recalibrate if divergence exceeds threshold

*Measurement:*
- Proxy fidelity: Linear-proxy prediction of SAE latents (R²)
- Known-concept speed: Latency of linear-proxy steering vs. SAE-based steering
- Accuracy trade-off: Steering accuracy (known-concept) via proxy vs. full SAE (should be > 95% of SAE accuracy)
- Success metric: Lightweight proxies achieve 90%+ of known-concept accuracy with 10x faster inference; SAE reference still enables discovery mode

---

## Scoring of Top 3 Proposals Against Ideal Final Result

**IFR Rubric (0-4 scale):**
- (i) **Leverages existing system resources** — no new infrastructure required
- (ii) **Minimal cost** — no significant additional expense in compute, storage, or time
- (iii) **No new problems** — solution does not introduce new degradations or side effects
- (iv) **Self-resolving** — the contradiction disappears by design; the system naturally reaches optimality

---

### Proposal 1A: Adaptive Sparsity Budget with Capability-Aware Annealing

**IFR Score: 2/4 (Partial IFR)**

**Criteria Met:**
- ✓ Minimal cost: Uses existing training infrastructure (scheduler modifications)
- ✓ No new problems: Gradual annealing is a well-tested technique; no novel side effects expected

**Criteria Unmet:**
- ✗ Leverages existing resources: Requires per-layer budget tracking and adaptive scheduler (new state management)
- ✗ Self-resolving: The contradiction is mitigated but not resolved—sparsity-capability tradeoff remains fundamental

**Rationale:** This proposal provides practical mitigation through staged constraint relaxation, but requires new instrumentation and does not fundamentally eliminate the interpretability-capability tension. It's an engineering workaround, not an IFR solution.

---

### Proposal 2A: Two-Stage Structure Validation

**IFR Score: 2/4 (Partial IFR)**

**Criteria Met:**
- ✓ Minimal cost: Uses existing circuit extraction and ablation methods
- ✓ No new problems: Freezing and testing discovered subgraphs is a standard validation technique

**Criteria Unmet:**
- ✗ Leverages existing resources: Requires two full training runs (dense then sparse), doubling compute cost
- ✗ Self-resolving: Tests faithfulness post-hoc but does not eliminate the imposed-vs-discovered tension

**Rationale:** This proposal makes the imposed-discovered tradeoff *testable* and *quantifiable*, which is valuable for research but does not resolve the fundamental contradiction—some degree of unfaithfulness is inherent to any imposed structure.

---

### Proposal 3A: Dual-Mode Adaptive Probe

**IFR Score: 3/4 (Near-IFR)**

**Criteria Met:**
- ✓ Leverages existing resources: Combines standard linear regression and existing SAE architectures; no new infrastructure
- ✓ Minimal cost: Linear probes are cheap; SAE training is amortized (one reference model per dataset)
- ✓ No new problems: Linear-SAE ensembles are well-understood; no novel failure modes expected

**Criteria Unmet:**
- ✗ Self-resolving: The probe remains a tool; it does not eliminate the usefulness-discovery tradeoff, only offers both modes

**Rationale:** This proposal comes closest to IFR by *unifying* the two functions (known-concept control and novelty discovery) into one adaptive system. The contradiction does not disappear, but it transforms from a "choose one" problem into a "enable both with graceful tradeoffs" problem. The online adaptation (features graduating from novel to known) is the self-resolving component, though incomplete.

---

## Synthesis and Recommendations

### Summary of Five Concrete Proposals

| # | Title | Principle | Contradiction | IFR Score | Key Innovation |
|---|-------|-----------|---|---|---|
| 1A | Adaptive Sparsity Budget | Beforehand Cushioning (11) | C1: Interp vs Capability | 2/4 | Per-layer budget slack; auto-recovery on loss divergence |
| 1B | Online Constraint Temperature | Dynamics (15) | C1: Interp vs Capability | 2/4 | Constraint temperature as a dynamic knob, annealed online |
| 1C | Per-Module Scorecard | Color Changes (32) | C1: Interp vs Capability | 2/4 | Visualize module-wise interp-capability tradeoff; targeted relaxation |
| 2A | Two-Stage Structure Validation | Segmentation (1) | C2: Imposed vs Discovered | 2/4 | Discover first, validate faithfulness, then impose structure |
| 2B | Superposition-Aware Decomposition | Taking Out (2) | C2: Imposed vs Discovered | 2/4 | Decompose superposed features before sparsification |
| 2C | Layer-Adaptive Sparsity | Local Quality (3) | C2: Imposed vs Discovered | 2/4 | Role-specific sparsity targets; respect module functions |
| 3A | Dual-Mode Adaptive Probe | Universality (6) | C3: Usefulness vs Discovery | **3/4** | Unified probe serving both known-concept and discovery modes |
| 3B | Failure-Driven Discovery | Blessing in Disguise (22) | C3: Usefulness vs Discovery | 2/4 | Use probe disagreement to prioritize discovery |
| 3C | Lightweight Proxy Probes | Copying (26) | C3: Usefulness vs Discovery | 2/4 | Reference SAE + fast linear proxies for speed + discovery tradeoff |

### High-Priority Experiment Track

**Recommendation:** Prioritize **Proposals 3A, 1A, and 2A** as the first experimental wave:

1. **3A (Dual-Mode Adaptive Probe)** – Highest IFR score (3/4); addresses the most impactful contradiction (usefulness vs. discovery), directly tied to deployment safety and auditing.
2. **1A (Adaptive Sparsity Budget)** – Practical mitigation of the classic interpretability-capability tradeoff; lowest implementation risk.
3. **2A (Two-Stage Structure Validation)** – Resolves unfaithfulness concerns in weight-sparse models; enables rigorous circuit verification.

Each proposal should be implemented as a separate config in `configs/e[X].yaml` with corresponding tests in `tests/test_design.py` and `tests/test_[module].py`.

---

## References

- TRIZ principles retrieved from: mcp__plugin_triz-engine_triz-knowledge
- Principles used: 1 (Segmentation), 2 (Taking Out), 3 (Local Quality), 6 (Universality), 11 (Beforehand Cushioning), 15 (Dynamics), 22 (Blessing in Disguise), 26 (Copying), 32 (Color Changes)
- Research grounding: docs/research1.md (The Instruction Set That Wasn't Designed)
- TRIZ matrix cell mappings:
  - (18, 28): Illumination Intensity → Measurement Accuracy [Principles: 11, 15, 32]
  - (35, 18): Adaptability → Illumination Intensity [Principles: 1, 6, 22, 26]

---

## Next Steps

1. **Implement proposals 1A, 2A, 3A** as experiment configs
2. **TDD cycle:** Write tests first for each proposal's success metrics
3. **Active-inference scoring:** Use `design.py` to rank experiments by expected information gain over H1-H5
4. **Execution:** Run experiments on MIabstraction testbed (Mess3, induction, algorithmic tasks)
5. **Evaluation:** Update `results/hypotheses.json` with outcomes; flag contradictions resolved vs. unresolved
