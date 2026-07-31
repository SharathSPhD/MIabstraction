# Research4: Mid-2026 Transformer Interpretability Landscape & Implementation Guidance

## Executive Summary

Mid-2026 marks a critical juncture in transformer mechanistic interpretability. Four major developments reshape the experimental landscape for MIabstraction:

1. **Parameter-level decomposition** (APD → SPD → VPD, Goodfire 2025-26) now scales to extract neuron-circuit mappings directly from weights
2. **Weight-sparse circuits** (OpenAI circuit-sparsity, Gao et al. 2511.13653) offer a third abstraction layer between dense architectures and hand-crafted probes
3. **SAE reliability crisis & mitigation** (Heap et al. 2501.17727; Kantamneni et al. 2502.16681): random-transformer controls are now mandatory, linear probes often outperform SAEs
4. **Belief-state geometry** is empirically validated (Shai et al. 2405.15943, Piotrowski et al. 2025) and reproducible with published code, making E1 tractable on small models

---

## 1. Attribution-based & Stochastic Parameter Decomposition (2025-26 State)

### Core Methods: APD → SPD → VPD

**Attribution-based Parameter Decomposition (APD)**
- Decomposes network parameters into interpretable components faithful to original parameters
- Extremely computationally expensive; sensitive to hyperparameters
- Impractical for models >10M parameters

**Stochastic Parameter Decomposition (SPD)** (Released June 2025, arXiv:2506.20790)
- Scalable variant of APD; robust to hyperparameter variation
- Decomposes weight matrices into sparse linear combinations of parameter vectors
- Successfully decomposed models up to ~100M parameters
- Avoids shrinkage issues; better identifies ground-truth mechanisms in toy models
- **Key advantage**: Works on parameters, not activations → captures learned algorithmic structure directly
- **Code**: https://github.com/goodfire-ai/spd and https://github.com/goodfire-ai/param-decomp

**Adversarial Parameter Decomposition (VPD)** (April 2026)
- Improves on both SPD and APD; represents frontier method
- Unclear scaling properties or release timeline for public code

### Recommendation for MIabstraction

**Use SPD as reference baseline** for comparing with circuit-sparse and SAE-based decompositions. If budget permits, run mini SPD experiment (e.g., 1-2 layer toy transformer) to establish feasibility. Otherwise, use published SPD results as benchmark.

---

## 2. Weight-Sparse Transformers & Circuit-Sparsity (2025-26 State)

### OpenAI Circuit-Sparsity (Gao et al., arXiv:2511.13653)

**Architecture & Training**
- Decoder-only GPT-2 style transformers
- **Sparsity mechanism**: L0 penalty on weights + AbsTopK activation masking + calibrated annealing schedule
- **Extreme sparsity**: ~1 in 1000 weights non-zero in final models
- **Model sizes tested**: Tens to hundreds of millions of non-zero parameters
- **Task**: Python code generation

**Key Technical Parameters** (from paper abstract):
- Three-lever optimization: (1) L0 weight penalty, (2) AbsTopK activation function, (3) annealing schedule
- Scaling shows: **capability-interpretability tradeoff** (sparser → more interpretable but lower task performance)
- **Larger sparse models improve Pareto frontier** (improve both capability and interpretability)

**Published Artifacts**
- Model: `openai/circuit-sparsity` on HuggingFace (419.1M params, Python code task)
- Code: https://github.com/openai/circuit_sparsity (Streamlit visualizer, test suite)
- Demo space: https://huggingface.co/spaces/mrfakename/circuit-sparsity
- **Training hyperparameters**: Not yet publicly released (marked as "[todo]" in repo README)

**Bridges Experiment (Preliminary)**
- Maps sparse circuits onto dense model checkpoints via activation "bridges"
- Goal: extract interpretable circuits from frontier models without retraining
- Status: Preliminary results; full scaling results pending

### Recommendation for E5 (Weight-Sparse Circuit Size Replication)

**Feasibility: MEDIUM**
- **Baseline**: Use released `openai/circuit-sparsity` model as reference
- **Mini-replication target**: 4-8 layer GPT-2 style transformer on toy task (e.g., repeated copying, parentheses matching)
- **Key hyperparameters to replicate**:
  - L0 weight penalty: estimate ~0.001-0.01 (inversely proportional to desired sparsity)
  - AbsTopK: keep top k% of activations alive per layer (estimate k=10-20% for interpretability-capability balance)
  - Annealing schedule: start without L0 penalty, anneal in over 10-20% of training
  - Batch size: 32-64
  - Steps: 100k-500k depending on task complexity
- **Metrics**:
  - Non-zero weight count per layer
  - Validation accuracy or task loss
  - Circuit interpretability (manual inspection of neuron connectivity)
  - Capability-interpretability Pareto points (plot accuracy vs. sparsity)

---

## 3. Belief-State Geometry & Computational Mechanics (E1 Guidance)

### Core Paper: Shai et al., arXiv:2405.15943

**Main Finding**
Transformers represent belief-state geometries in their residual stream, even for non-trivial fractal structures. The geometry is linear: belief probability simplex appears directly as a simplex in residual stream coordinates.

**Mess3 HMM Dataset & Experimental Setup**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Process | Mess3 (3-state HMM) | Generates from {A, B, C} |
| Observation vocab | {A, B, C} | Non-unifilar: state transition graph ≠ observation graph |
| MSP geometry | Fractal simplex (2D) | Non-trivial complexity |
| Alternative process | RRXOR (5-state) | Tests alternative geometries |

**Model Architecture**

| Component | Specification | Notes |
|-----------|---------------|-------|
| d_model | 64 | Residual stream dimension |
| n_layers | 4 | Depth |
| n_heads | 1 per layer | Single head per layer (minimal) |
| head_dim | 8 | 64/1 = 64 per head |
| d_mlp | 256 | Hidden layer dimension |
| activation | ReLU | Standard |
| context_length | 10 tokens | Sufficient for belief state convergence |
| attention | Causal masking | Standard autoregressive |

**Training Setup**

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | SGD | Standard choice |
| Learning rate | 0.01 | Relatively high; tune if needed |
| Batch size | 64 | Reasonable for small model |
| Total epochs | 1,000,000 | Very long; expect convergence much earlier (~100k) |
| Weight decay | 0 (none) | No regularization |
| Data distribution | Stationary | Sequences drawn from process equilibrium |
| Validation | Cross-validation | Report robust metrics |

**Key Metrics & Evaluation**

1. **Geometric fit**: PCA on final residual stream activations; fit simplex to belief coordinates
   - Metric: R² of belief coordinate reconstruction from residual stream
   - Target: R² > 0.9 for Mess3
2. **Layer-by-layer emergence**: Plot belief coordinate leakage per layer
   - Expectation: full geometry in layer 4 (final layer)
3. **Fractal structure**: Compare predicted vs observed MSP geometry
4. **Generalization**: Test on held-out sequences; measure consistency

**GitHub Code**
- https://github.com/danibalcells/belief-state-transformers
- Contains reproducible experiments, Mess3 generation, and geometric fitting code

### Recommendation for E1

**Feasibility: HIGH** (already published + code available)
- Use the referenced GitHub code as baseline
- Mini-replication: reproduce Mess3 experiment exactly on a single GPU (<<1 hour on single L4)
- Variance study: train 5 random seeds; measure consistency of geometry emergence
- **Must-validate metrics**: (1) simplex fit R² per layer, (2) belief coordinate correlation, (3) training dynamics (when does geometry stabilize?)

---

## 4. SAE Critique, Defenses & Reliability Assessment (E3 & E4 Guidance)

### E3: SAE Random-Transformer Control (Heap et al., arXiv:2501.17727)

**Main Finding**
SAEs trained on randomly initialized transformers produce **nearly identical quality metrics** to those trained on actual trained models. This suggests SAE metrics alone cannot validate that features capture learned computations.

**Critical Empirical Results**
- Auto-interpretability scores: ~95% correlated between random and trained
- Reconstruction loss: similar magnitude for both
- Effect size: consistent across model sizes and layers
- **Implication**: high SAE quality metrics are insufficient proxies for mechanistic interpretability

**Safeguards the Paper Recommends**

1. **Always include random baselines**
   - Train SAE on randomly initialized version of your model
   - Compare all metrics (reconstruction, sparsity, auto-interp) side-by-side
   - Only claim feature discovery if trained >> random

2. **Use targeted measures of feature abstractness**
   - Measure if features activate on semantically similar inputs (not just training examples)
   - Test on held-out data with semantic shifts
   - Metric: feature consistency under input paraphrase or style transfer

3. **Combine with downstream tasks**
   - Verify features improve performance on auxiliary tasks (e.g., probing for target attribute)
   - Linear probe on top of SAE features should beat linear probe on raw activations
   - Use Kantamneni et al. setup (see below)

### E4: Linear Probe vs. SAE Probe (Kantamneni et al., arXiv:2502.16681)

**Main Finding**
When applied to LLM activation probing, **SAE-based probes do not outperform simple logistic regression baselines** across four challenging regimes: data scarcity, class imbalance, label noise, covariate shift.

**Quantitative Results**
- Baseline logistic regression on raw activations: 88% intervention success rate
- SAE-based probes: 41% intervention success rate
- Even with max/mean aggregation for multi-token probing, attention-based logistic regression removes most SAE advantage
- Conclusion: basic baselines are sufficient and more interpretable

**Experimental Regimes & Setup**

| Regime | Setup | SAE Performance |
|--------|-------|-----------------|
| Data scarcity | Few labeled examples | Underperforms |
| Class imbalance | Unbalanced label distribution | Underperforms |
| Label noise | Noisy/unreliable annotations | Underperforms |
| Covariate shift | Test domain differs from train | Underperforms |

**Why Linear Probes Win**
- Lower parameter count: less overfitting in low-data regimes
- Interpretability: linear probe weights directly show direction in activation space
- Robustness: SAE training introduces additional hyperparameters (sparsity, dict size) that must be tuned

**Code & Reproduction**
- GitHub: https://github.com/sae-probes/sae-probes (official benchmark)
- Also see: https://github.com/JoshEngels/SAE-Probing

### Recommendation for E3 & E4

**E3 (SAE Random-Transformer Control)**
- **Experimental design**: 
  - Train SAE on trained model AND on random-initialized version of same architecture
  - Measure: (1) reconstruction loss, (2) sparsity distribution, (3) auto-interpretability scores
  - Compare via violin plots
  - Claim feature discovery only if trained >> random with p < 0.01
- **Safeguard**: Use sae-lens library (v6.46.1, latest as of July 2026)
- **Model**: Pythia-70M or Pythia-160M for tractability on single GPU
- **Hyperparameters**:
  - Dictionary size: 4x * d_model (e.g., 256 for d_model=64)
  - Sparsity loss (beta): 0.1-1.0 (tune via validation)
  - Batch size: 32-128
  - Learning rate: 1e-4 (Adam)
  - Steps: 10k-50k

**E4 (Linear Probe vs. SAE Probe)**
- **Experimental design**:
  - Task: Factual recall (is "Paris" capital of France?) on language model activations
  - Probe at multiple layers
  - Compare: (1) logistic regression, (2) SAE-derived features, (3) raw activations
  - Metrics: validation AUC, intervention success rate (causal tracing), interpretability (cosine similarity of probe weight to known features)
- **Model**: Pythia-1B or Gemma-2B
- **Hyperparameters**:
  - SAE dict size: 8x * d_model
  - Sparsity: 10-100 (tune)
  - Logistic regression: L2 regularization, C=0.1-1.0

---

## 5. Induction Head Phase Transition (E2 Guidance)

### Core Papers

**Emergence & Phase Transition Dynamics**
- Induction heads implement A B ... A → B copy/repeat pattern
- Sharp phase transition during pretraining: loss plateaus, then sudden drop when induction circuits mature
- Only 3 dimensions control head emergence; learned in O(N²) time where N = context length

**Recent Advances (2025-26)**
- Circuit parameters converge sequentially: readout → match strength → token routing
- Phase transition timing is **data-dependent**: class count and label frequency predictively modulate transition sharpness
- Dual-route structure in large models: token-level verbatim copy + concept-level multi-token abstraction

**Available Code**
- https://github.com/danibalcells/belief-state-transformers (includes induction head studies as submodule)
- Emergent Mind topic pages with reproducible experiments

### Recommendation for E2

**Feasibility: MEDIUM-HIGH**

**Experimental setup**:
1. **Synthetic induction task**: Train transformer on sequences of form [A₁, B₁, A₂, B₂, ..., Aₙ, Bₙ] with n=2-5
   - Vocabulary size: 10-100 tokens
   - Context length: n+5 (leave room for final prediction)
   - Sequence length: 1000-10000 examples

2. **Model architecture**:
   - d_model: 64-128
   - n_layers: 2-4
   - n_heads: 2-4 (must be >1 to measure head-specific behavior)
   - Context: 5-10 tokens

3. **Training**:
   - Optimizer: Adam, lr=1e-3
   - Batch size: 32
   - Steps: 50k-100k
   - Log validation loss every 1000 steps

4. **Measurement**:
   - **Phase transition**: Plot validation loss per epoch; identify sharp knee (transition point)
   - **Circuit analysis**: Measure attention pattern from [Aᵢ] to [Bᵢ] in layer-2 heads
   - **Interpretability**: Do top attention heads implement prefix matching (high attention to matching A)?
   - **Dimensionality**: PCA on attention weight matrix; how many dimensions encode induction pattern?

**Key Hyperparameters to Vary**:
- Number of classes (affects Q/K match learning speed)
- In-context label frequency (affects V-copy learning)
- Layer depth (2 vs 4; should see depth-dependent emergence)

**Metrics**:
- Training step at which phase transition occurs
- Sharpness of transition (gradient of loss curve)
- Head-wise attention pattern alignment (cosine sim. to theoretical induction pattern)
- Generalization to longer sequences not seen in training

---

## 6. HuggingFace Model & Dataset Availability (Mid-2026 Landscape)

### Models Available for Experiments

**Weight-Sparse Models**
- `openai/circuit-sparsity` (419.1M params, Python code task, GPT-2 style)
  - Status: Released December 2025
  - Use for: E5 baseline, bridges experiments
  - Availability: ✅ HuggingFace Model Hub

**SAE Weights (Interpretability)**
- `mwhanna/gemma-scope-attn-saes-16k` (attention layer SAEs for Gemma)
- `Anish-1101/gemma-2-9b-it-sae-scoped-coding` (Gemma-2 SAEs, April 2026)
- `Anish-1101/gemma-3-12b-it-sae-scoped-coding` (Gemma-3 SAEs, April 2026)
  - Status: Fresh releases (early 2026)
  - Use for: SAE comparison studies
  - Availability: ✅ HuggingFace Model Hub

**Base Language Models (For Probing & SAE Training)**
- `EleutherAI/pythia-70m` / `pythia-160m` / `pythia-1b`
  - Status: Well-established, stable checkpoints
  - Use for: E1, E3, E4 experiments (small, fast)
  - Availability: ✅ HuggingFace
  - Memory: <4GB for Pythia-1B on GPU

- `gpt2` (OpenAI GPT-2, 124M params)
  - Status: Baseline
  - Use for: comparison baseline
  - Availability: ✅ HuggingFace
  - Memory: <500MB

**Gemma Series (Alternative to Pythia)**
- `google/gemma-2-9b-it`, `google/gemma-3-12b-it`
  - Status: Modern, high-quality
  - Use for: E4 linear probe comparisons
  - Availability: ✅ HuggingFace (requires auth)
  - Memory: ~20GB for 9B, ~40GB for 12B

### Datasets & Code

**Belief-State Geometry (E1)**
- Mess3 HMM generation code: https://github.com/danibalcells/belief-state-transformers
  - Generates synthetic sequences; no HuggingFace dataset needed
  - Status: ✅ Available, reproducible

**SAE Training & Probing**
- sae-lens library: `pip install sae-lens` (v6.46.1 as of July 2026)
  - Includes pre-trained SAEs for Pythia, Gemma, Llama
  - Status: ✅ Maintained, actively updated
  - Code: https://github.com/decoderesearch/SAELens

**Computational Mechanics Datasets**
- No dedicated HuggingFace dataset found for Mess3/HMM tasks
- **Workaround**: Generate synthetic sequences on-the-fly (standard practice; see belief-state-transformers repo)

---

## 7. 2026 New Abstraction Layers & Recent Breakthroughs

### Mechanistic Interpretability as Established Field

Mechanistic interpretability now organized into **three abstraction layers**:
1. **Neurons**: Individual weight parameters and activations
2. **Circuits**: Subcircuits combining multiple neurons (cross-layer, cross-head)
3. **Algorithms**: End-to-end mechanisms (induction, in-context learning, arithmetic)

### Parameter-Level Abstractions (New in 2025-26)

**Stochastic Parameter Decomposition (SPD)** as intermediate layer:
- **Where it sits**: Between raw parameters and circuits
- **What it does**: Factors weight matrices into interpretable basis vectors
- **Advantage**: Captures learned source-code-like algorithmic structure
- **Scaling**: Up to ~100M parameter models

**Adversarial Parameter Decomposition (VPD)** as frontier:
- Improves on SPD robustness and accuracy
- Emerging standard; code not yet public

### Recognition & Impact

- **MIT 2026 Breakthrough Technology**: Mechanistic interpretability
- Signals mainstream academic acceptance of the field
- Growing infrastructure: mlxterp (Apple Silicon), sae-lens (v6+), circuit-sparsity tooling

### Tools Ecosystem (as of July 2026)

| Tool | Purpose | Status | GB10 Feasible |
|------|---------|--------|---|
| sae-lens | SAE training & analysis | ✅ Maintained | Yes |
| circuit-sparsity | Weight-sparse circuits | ✅ Released | Yes |
| TransformerLens | Activation tracing | ✅ Standard | Yes |
| param-decomp (SPD) | Parameter decomposition | ✅ Released | Yes, small models |
| transformer-circuits | Automated circuit discovery | ✅ Available | Partial |

---

## 8. Implementation Priorities for MIabstraction

### Experiment-by-Experiment Roadmap

#### E1: Belief-State Geometry (Mess3 HMM)
- **Effort**: Low (code available)
- **Novelty**: Medium (reproducibility study)
- **Priority**: HIGH (validates computational mechanics approach)
- **Deliverable**: Reproduce Shai et al. + variance across seeds
- **Timeline**: 1-2 weeks, <$5 compute

#### E2: Induction Head Phase Transition
- **Effort**: Medium (some code to write)
- **Novelty**: Medium (extend prior work to data-dependent regimes)
- **Priority**: MEDIUM (core circuit phenomenon)
- **Deliverable**: Phase transition plots + circuit dimensionality analysis
- **Timeline**: 2-3 weeks, ~$10-20 compute

#### E3: SAE Random-Transformer Control
- **Effort**: Medium (mostly plumbing with sae-lens)
- **Novelty**: High (applying recent critique to new model)
- **Priority**: HIGH (methodological soundness)
- **Deliverable**: Comparative metrics; validated feature claims
- **Timeline**: 2-3 weeks, ~$20-30 compute

#### E4: Linear Probe vs SAE Probe
- **Effort**: Medium (reimplementation of Kantamneni benchmark)
- **Novelty**: High (establishes when SAEs worth effort)
- **Priority**: MEDIUM-HIGH (practical guidance)
- **Deliverable**: Intervention success rates + interpretability comparison
- **Timeline**: 3-4 weeks, ~$30-40 compute

#### E5: Weight-Sparse Circuit Size
- **Effort**: High (requires training with L0 penalty + annealing)
- **Novelty**: Medium (mini-replication of OpenAI work)
- **Priority**: MEDIUM (establishes capability-interpretability frontier)
- **Deliverable**: Pareto curves (accuracy vs sparsity) + neuron interpretability
- **Timeline**: 4-6 weeks, ~$50-100 compute (depends on model size)

### Sequential Dependency Graph
```
E1 (Mess3) → E2 (Induction) → E3 (SAE Control)
                                  ↓
                             E4 (Probe Comparison)
                             
E5 (Weight-Sparse) [independent, can run in parallel]
```

**Recommended execution order**: E1 → E3 → E4 (tightly coupled); E2 and E5 in parallel

---

## 9. Critical 2026 Methodological Shifts

### Must-Do Safeguards (Non-Negotiable)

1. **Random-transformer control (E3 mandate)**
   - Do not claim feature discovery without random baseline
   - SAE metrics alone insufficient

2. **Baseline comparisons (E4 mandate)**
   - Always compare to linear probes
   - Report effect size vs. baseline, not just absolute performance

3. **Reproducibility & seed variance**
   - Report mean ± std over ≥3 random seeds
   - Plot error bars on all quantitative claims

4. **Code & artifact release**
   - Publish experiment configs (training hyperparameters)
   - Release trained model checkpoints where feasible
   - Use container/environment reproducibility (e.g., conda env)

### Emerging Best Practices

- **Parameter-level thinking**: Decompose weights alongside activations (SPD as reference)
- **Geometric perspectives**: Fit geometric structures (simplices, manifolds) to validate learned representations
- **Phase transitions**: Study sharpness and data dependence of emergent phenomena
- **Circuit-to-algorithm mapping**: Link hand-crafted circuits to algorithmic abstractions (induction, copying, arithmetic)

---

## Sources

- [Stochastic Parameter Decomposition (arXiv:2506.20790)](https://arxiv.org/abs/2506.20790)
- [OpenAI Circuit-Sparsity Paper](https://cdn.openai.com/pdf/41df8f28-d4ef-43e9-aed2-823f9393e470/circuit-sparsity-paper.pdf)
- [Weight-sparse transformers have interpretable circuits (arXiv:2511.13653)](https://arxiv.org/abs/2511.13653)
- [OpenAI Circuit-Sparsity Model on HuggingFace](https://hf.co/openai/circuit-sparsity)
- [OpenAI circuit_sparsity GitHub](https://github.com/openai/circuit_sparsity/)
- [Transformers Represent Belief State Geometry (arXiv:2405.15943)](https://arxiv.org/abs/2405.15943)
- [belief-state-transformers GitHub](https://github.com/danibalcells/belief-state-transformers)
- [Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers (arXiv:2501.17727)](https://arxiv.org/abs/2501.17727)
- [Are Sparse Autoencoders Useful? A Case Study in Sparse Probing (arXiv:2502.16681)](https://arxiv.org/pdf/2502.16681)
- [SAE Probes GitHub](https://github.com/sae-probes/sae-probes)
- [SAE-Probing GitHub](https://github.com/JoshEngels/SAE-Probing)
- [Towards Scalable Parameter Decomposition (Goodfire Research)](https://www.goodfire.ai/research/stochastic-param-decomp)
- [Goodfire param-decomp GitHub](https://github.com/goodfire-ai/param-decomp)
- [Goodfire SPD GitHub](https://github.com/goodfire-ai/spd)
- [SAE Lens PyPI](https://pypi.org/project/sae-lens/)
- [SAELens GitHub](https://github.com/decoderesearch/SAELens)
- [Induction Heads Emergence on Emergent Mind](https://www.emergentmind.com/topics/induction-heads-in-transformers)
- [How Transformers Implement Induction Heads (arXiv:2410.11474)](https://arxiv.org/abs/2410.11474)
- [Mechanistic Interpretability for Neural Networks Survey (arXiv:2607.07316)](https://arxiv.org/abs/2607.07316)
- [Open Problems in Mechanistic Interpretability (arXiv:2501.16496)](https://arxiv.org/abs/2501.16496)
- [Stop Probing, Start Coding (arXiv:2603.28744)](https://arxiv.org/abs/2603.28744)
- [Untangling Neural Network Mechanisms: Goodfire's Lee Sharkey Interview (Cognitive Revolution)](https://www.cognitiverevolution.ai/untangling-neural-network-mechanisms-goodfires-lee-sharkey-on-parameter-based-interpretability/)
- [Gemma-Scope SAE on HuggingFace](https://hf.co/mwhanna/gemma-scope-attn-saes-16k)
- [Pythia Models on HuggingFace](https://huggingface.co/EleutherAI)
- [Constrained Belief Updates Explain Geometric Structures (arXiv:2502.01954)](https://arxiv.org/abs/2502.01954)
- [Finding Belief Geometries with Sparse Autoencoders (arXiv:2604.02685)](https://arxiv.org/abs/2604.02685)
