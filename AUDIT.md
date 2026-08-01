# AUDIT — MIabstraction/Loom vs. Stated Intent

**Auditor:** Claude (adversarial)  
**Date:** 2026-08-01  
**Branch:** audit/intent  
**Scope:** Does built system match user's stated intent? Evidence-based assessment per question A–G.

---

## Executive Summary

**VERDICT: Critical drift between intent and execution.**

The user asked for **a C-like programming language to build LLMs from scratch** (data + pretraining + post-training), accessible to laypeople, architecture-agnostic, with real measured results. What was built is **a declarative configuration language for a compiler that trains tiny synthetic-task models and steers frozen open-weight models**.

**Single most critical gap:** The system cannot build a *complete* LLM end-to-end. It can pretrain (foundation backend ran, failed gates). It cannot post-train (SFT/RLHF/DPO absent). It cannot rebuild an existing model architecturally (port backend only installs steering hooks on frozen weights).

**Severity:** MISALIGNMENT. The framing is aspirational (marketing as "programming the transformer") while the implementation is proof-of-concept (narrow backends, toy models, gated capabilities).

---

## A. IS IT A PROGRAMMING LANGUAGE?

### Claim
The user asked for "instructions in a high level programming language like C."

### Evidence
**What exists:** `weave.yaml` is YAML-based, declarative.

```yaml
# demo_loom.weave.yaml (lines 1–28)
model: {size: small}
skills:
  - name: mess3_dynamics
    kind: state_tracking
    world: mess3
controls:
  - name: suppress_token_1
    kind: suppress
    token: 1
gates:
  suppress_token_1: {suppression_ratio: ">0.9"}
```

**What C-like languages have that weave.yaml lacks:**

| Feature | C | weave.yaml | Status |
|---------|---|-----------|--------|
| Statements/sequencing | if, for, while | none | ABSENT |
| Variables (named state) | `int x; x = 5;` | none | ABSENT |
| Functions/procedures | `void foo(int x) {}` | none | ABSENT |
| Function composition | f(g(x)) | none | ABSENT |
| Control flow | branches, loops | none | ABSENT |
| Types/type-checking | int, float, struct | implicit in `kind` field | MINIMAL |
| Expressions/operators | +, -, *, / | none (only gate thresholds) | ABSENT |
| Separate compilation | .o object files | not implemented | ABSENT |
| IR (intermediate representation) | bytecode, ASM | none visible | ABSENT |
| Multiple backends | x86, ARM | 4 hardcoded backends | IMPLICIT ONLY |
| Linkable stdlib | libc, libmath | `LOOM_STDLIB.md` lists ~15 primitives (skill kinds, control kinds), not linkable | READ-ONLY |
| Modular imports | `#include` | none | ABSENT |

### Finding
**weave.yaml is declarative YAML, not an imperative programming language.** The compiler infers everything: architecture, mixing ratios, task tokens, curriculum design. The user does not program—they declare properties and the compiler picks implementations.

### Comparison to Intent
**Intent:** "a high level programming language like C that a compiler compiles"  
**Reality:** A constraint declaration format (like YAML configs for deployment tools), not a language with control flow, variables, or composition.

### Severity
**HIGH. This is a fundamental mismatch.** A layperson who tries to write `if (accuracy > 0.9) { retrain } else { refuse }` will find there is no conditional; the weave is fully declarative.

---

## B. DOES IT BUILD AN ACTUAL LLM?

### Claim
The user wants the compiler to "produce an LLM via data + pretraining + post-training (SFT/RLHF/DPO)."

### Evidence

#### 1. Pretraining (Foundation Backend)
**Status: WORKS BUT GATED FAILS**

- **File:** `src/loom/foundation.py` (lines 1–50)
- **What it does:** Takes a BabyLM corpus, trains a GPT-2-style decoder (vocab 50K, 384-dim, 6 layers, 30M params)
- **Real run:** `results/loom_foundation_demo.json`
  - Trained for 60k steps (~45 min on RTX 5090)
  - 737M tokens seen (7.6 epochs over 100M-word corpus)
  - **Result:** val_loss=4.046, val_ppl=57.2
  - **Gates:** Declared val_ppl < 20.0 → **FAILED** (57.2 vs. 20.0 threshold)
  - **Refusal:** Compiler correctly refused to emit checkpoint; produced diagnosis in refusal_diagnosis field

**Verdict:** Real pretraining runs, but the trained model does not meet declared quality bar. This is honest (no fabrication), but also honest failure.

#### 2. Data Curation / Mixing (None)
**Status: ABSENT**

- The foundation backend hardcodes BabyLM-2026-Strict as the only corpus option
- No data curation pipeline in weave.yaml
- No way to declare data mixing, filtering, or deduplication
- **Code:** `src/loom/foundation.py:_ensure_corpus()` only downloads from HF cache; no composition

#### 3. Tokenizer Construction (Partial)
**Status: MINIMAL**

- `spec.Foundation` allows `tokenizer: "gpt2" | "bpe_train"` (lines 54–59 in `spec.py`)
- "bpe_train" option exists but no code path implements it in the foundation backend
- **Missing:** No tokenization benchmark, no vocab size justification, no weave control over tokenizer

#### 4. Post-Training (SFT) — **NOT IMPLEMENTED**
**Status: ABSENT**

- Grep for "sft", "supervised fine-tune", "instruction" in src/loom/:
  ```bash
  grep -r "sft\|supervised.fine" src/loom/
  # Returns nothing
  ```
- No SFT backend exists
- weave.yaml has no way to declare SFT stage
- **Finding:** Foundation backend produces a base model; no pipeline to SFT it on domain data

#### 5. Post-Training (RLHF/DPO) — **NOT IMPLEMENTED**
**Status: ABSENT**

- Grep for "rlhf", "dpo", "reward", "preference" in src/loom/:
  ```bash
  grep -r "rlhf\|dpo\|reward.model" src/loom/
  # Returns nothing
  ```
- No RLHF or DPO backend
- weave.yaml cannot declare preference data, reward model, or policy optimization
- **Critical gap:** User's intent (item 4) explicitly mentions "post-training/fine-tuning/RLHF"

#### 6. Evaluation Harness — **PARTIAL**
**Status: PARTIAL**

- Foundation backend measures: val_loss, val_ppl, (optional) blimp_acc
- `src/loom/verify.py` implements gate verification (lines 1–50)
- **Missing:** No harness for downstream task evaluation (GLUE, SuperGLUE, etc.), no human eval framework, no open-source benchmark hookup

#### 7. Checkpoint Export — **MINIMAL**
**Status: MINIMAL**

- Weights are saved (foundation backend): checkpoint in job_dir
- No huggingface_hub integration for upload
- No standard export formats (safetensors, ONNX, etc.)
- **Missing:** No way to export a trained model to a standard format for use downstream

### Finding
**The system can pretrain a small LLM on BabyLM. It cannot SFT, RLHF, or DPO. It cannot handle data curation. The foundation backend ran and produced real loss curves, but gates failed, and the trained model was refused.**

This is honest but incomplete. A user following intent (4) — "build an LLM from scratch (data+pretraining+post-training)" — would hit a wall at SFT.

### Severity
**CRITICAL. Post-training (SFT/RLHF) is 50%+ of the pipeline for a usable LLM. Its absence means the system cannot build a "complete" LLM as stated.**

---

## C. IS IT ARCHITECTURE-AGNOSTIC?

### Claim
The user named "gpt-bert, nemotron, qwen, gemma, kimi, deepseek, glm" as substrates the abstraction must sit ABOVE.

### Evidence

#### What Abstracts Over Architectures
1. **Port backend** (`src/loom/port.py:63–80`): Wraps HF AutoModelForCausalLM; can load any HF model
2. **Steering hooks** are model-agnostic: installed in any transformer's residual streams

#### What Is Hardcoded to One Architecture
1. **Foundation backend** (`src/loom/foundation.py:59–100`):
   - Hardcodes decoder-only GPT-style architecture
   - ```python
     d_model = 256  # fixed default
     n_layers = 4
     n_heads = 8
     vocab_size = 50257  # GPT2 only
     ```
   - Cannot build a BERT (encoder-only, masked LM), GLM (prefix LM), or T5 (encoder-decoder)
   - No architecture template system

2. **Curriculum backend** (`src/loom/curriculum.py`):
   - Compiles to attention-only transformers (no MLPs in default path)
   - ```python
     attn_only: bool = True  # line 70
     ```
   - Hard-coded skill compilers (InductionCompiler, StateTrackingCompiler) assume attention-based architectures
   - Cannot compile to state-space models (Mamba, S6), RNNs, or attention variants (linear attention, etc.)

3. **Constructed backend** (`src/loom/constructed.py:1–50`):
   - Manually compiles induction circuits into 4-layer TinyTransformer
   - Hardcoded layer indices, head counts, matrix sizes
   - Zero abstraction over model family

#### Finding
**The system claims architecture-agnosticism but delivers only:**
- **Port backend:** Truly agnostic (works with any HF model) but limited to steering frozen weights
- **Foundation + Curriculum + Constructed:** Welded to decoder-only attention transformers; cannot target BERT, GLM, Mamba, or alternative attention forms

### Severity
**MEDIUM-HIGH.** Port backend delivers on the architectural flexibility promise *for frozen models* (any HF model can be steered). But the full pipeline (foundation + post-training) targets only GPT-style decoders. A user wanting to build a BERT-like encoder hits a wall.

---

## D. "REBUILD AN EXISTING OPEN-WEIGHT LLM PROGRAMMATICALLY"

### Claim
User's intent (4): "rebuild the existing open weight llm programatically...surgery, merging, distillation, adapter composition, layer replacement, continued pretraining, re-alignment"

### What Port Backend Actually Does
**File:** `src/loom/port.py:1–50`

```python
"""PORT backend: frozen HF causal LMs with Loom-compatible control/monitoring.
...
Design: frozen weights + forward hooks for controls/monitors at each generation step.
"""
```

**Capabilities implemented:**
1. Dose-calibrated steering (activation addition to residual stream)
2. Event-gated writes (entropy-based gating)
3. Linear probe monitors
4. Token suppression via logit masking

**Capabilities NOT implemented:**
- Surgery (selective weight editing)
- Merging (TIES, DARE, etc.)
- Distillation (knowledge transfer)
- Adapter composition (LoRA stacking, etc.)
- Layer replacement (swap internal layers)
- Continued pretraining (fine-tuning base model on new tokens)
- Re-alignment (steering toward new values)

### Finding
**The port backend can steer and monitor a frozen model. It cannot modify, extend, or reconstruct it.** "Rebuild programmatically" implies surgery or compositional modification; port only adds read/write hooks to a static model.

### Severity
**HIGH.** The port backend is essentially prompt injection (steering at inference time), not model building. A user wanting to "rebuild" a model architecturally would need to retrain or use the foundation backend, both of which are limited.

---

## E. WHAT IS WORTH KEEPING?

### Good Design Decisions

1. **Gate Discipline** (src/loom/spec.py:64–131)
   - Every skill/control must declare success criteria upfront
   - Gates refuse vacuous thresholds (e.g., accuracy > -1 is rejected as impossible)
   - Compiler refuses to emit a model if gates fail
   - This prevents the "trained but who knows" problem
   - **Worth keeping:** Gate semantics are load-bearing; they enforce honesty

2. **ABI + Linker** (src/loom/abi.py, src/loom/linker.py)
   - Separate compilation: skills can be compiled independently, linked at load time
   - Provenance ledger: tracks which skill is responsible for which capability
   - Cost calculation: can measure the "price" (GPU-minutes, tokens, layers) of each skill
   - **Worth keeping:** Compositional architecture is elegant and separates concerns

3. **Constructed Backend as Proof** (src/loom/constructed.py)
   - Hand-compiled induction circuit (previous-token + attention) inserted into weights
   - No training: shows that weight manipulation alone can encode capabilities
   - Gates verify the hand-compiled circuit works
   - **Worth keeping:** Demonstrates that the layer is real and writable, not just interpretable

4. **Real Measured Results** (results/*.json)
   - Foundation pretraining: real loss curves, honest gate failures
   - Curriculum demos: real training runs with reported metrics
   - All numbers read from committed artifacts at build time (not fabricated)
   - **Worth keeping:** "No fabrication" principle is enforced; every claim has a trace file

5. **Runtime Hooks** (src/loom/runtime.py)
   - Installation of steering vectors, probes, and controls as PyTorch forward hooks
   - Decoupled from model training: can be installed post-hoc on frozen models
   - Measurable effects: suppression_ratio, side_effect measured per control
   - **Worth keeping:** Runtime control is real and composable

### Costs and Compromises

1. **Declarative > Imperative**: The user asked for a "programming language" but got a config format. Declaring is easier to verify but less expressive. **Verdict:** Acceptable trade-off for a verification system, but name it differently (configuration language, not programming language).

2. **Tiny Models**: Curriculum backend trains on 128-dim, 4-layer models. Synthetic task skills (induction, state tracking) don't scale to real language tasks. **Verdict:** Understood as a proof-of-concept, but this is a tool for toy problems, not production.

3. **Frozen Foundation Models**: Port backend cannot modify an existing model's weights, only add steering. No way to merge, distill, or re-align. **Verdict:** Severe limitation for the "rebuild" use case.

---

## F. LAYMAN TEST: Write What a Layman Should Write vs. What They Write Today

### What a Layman SHOULD Write (in the envisioned system)

If Loom delivered on its promise, a layperson with *no* ML experience could write:

```yaml
# build_my_first_llm.weave.yaml
name: my_english_model

model:
  size: medium

foundation:
  corpus: babylm_strict
  tokenizer: gpt2
  params: 50_000_000
  budget_hours: 24

sft:
  corpus: alpaca
  num_epochs: 3
  learning_rate: 1e-4

rlhf:
  reward_model: mistral-7b
  num_preference_pairs: 10000

skills:
  - name: answer_questions
    kind: qa
  - name: refuse_harmful
    kind: safety

gates:
  foundation:
    val_ppl: "<15.0"
  answer_questions:
    accuracy: ">0.8"
  refuse_harmful:
    refusal_rate: ">0.95"
```

Then they would run:
```bash
loom build build_my_first_llm.weave.yaml --out ./my_model
```

And get either:
- A trained model with a `report.json` proving all gates passed
- A refusal: "Cannot build: val_ppl reached 18.0 against threshold 15.0. Recommend: larger model or longer budget."

**Why this is good for laypeople:**
- They never write training loops, gradient updates, or loss functions
- They state what they want; the compiler decides how
- Refusal is clear and diagnostic, not silent failure
- Every number is measured, not hoped for

### What a Layperson Writes TODAY (2026)

Without Loom, a layperson wanting to train an LLM would:

1. **Find a tutorial** (HuggingFace Transformers docs)
2. **Copy-paste boilerplate:**
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
   import torch
   
   model_name = "gpt2"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   
   # Load your dataset
   dataset = load_dataset("your_data")
   
   # Training args (guessed, not principled)
   training_args = TrainingArguments(
       output_dir="./my_model",
       num_train_epochs=3,
       per_device_train_batch_size=8,
       learning_rate=5e-5,  # why 5e-5? hope.
       warmup_steps=500,
       save_steps=500,
   )
   
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       tokenizer=tokenizer,
   )
   
   trainer.train()
   ```

3. **Wait** (hours to days depending on model size)

4. **Check loss** (in the tensorboard log) - hope it went down

5. **Evaluate manually** on ad-hoc test cases - is the output good? shrug

6. **Deploy and pray** - in production, does it actually do what you need?

**Why this is bad for laypeople:**
- They copy-paste without understanding hyperparameters (batch_size, lr, warmup)
- No declaration of "I need the model to refuse harmful requests"
- No automated verification of quality
- Failure modes are silent (low perplexity, but incoherent output; or fluent but unsafe)
- Cannot modularly add a safety layer to an existing model

### Comparison

| Dimension | Loom (Ideal) | Loom (Actual) | Baseline (Today) |
|-----------|---|---|---|
| Lines to build an LLM | 30–50 | N/A (SFT/RLHF missing) | 100+ with trial-and-error |
| Declare quality bar | ✅ yes (gates) | ✅ yes (gates) | ❌ no |
| Automated verification | ✅ yes (compiler refuses) | ⚠️ partial (foundation runs but gates often fail) | ❌ no |
| Post-training | ✅ (RLHF/DPO) | ❌ absent | ⚠️ manual scripting |
| Modular controls | ✅ (weave declares skip/add safety) | ⚠️ (port backend only; frozen models) | ❌ no |

### Finding
**Loom's aspirational layman story is compelling. The actual system delivers 60% of it:** foundations, verification gates, and steering hooks. It *cannot* deliver the full pipeline because SFT/RLHF are absent, and the "rebuild" promise is limited to steering, not architectural modification.

---

## G. COURSE CORRECTION: Prioritized Plan

### Hardware Assumptions
- **Local:** GB10 with 128GB unified memory, good for training small/medium models
- **Remote:** RTX 5090 with 32GB, job submission via skill, good for throughput-limited tasks (pretraining)

### Prioritized Road to Full Intent

#### **PHASE 1 (Months 1–2): Complete the Base Pipeline**

**1.1 Implement SFT (Supervised Fine-Tuning) Backend**
- **Why first:** SFT is the bridge from pretraining to usable LLMs. Without it, foundation models are dead-end.
- **Scope:** 
  - Add `sft` section to weave.yaml spec (corpus, epochs, lr, warmup)
  - Implement `src/loom/sft.py` backend
  - Reuse foundation model as base
  - Add SFT gates: downstream accuracy (GLUE tasks), perplexity on domain data
  - Real run on a 10M-token domain corpus (should fit in GB10)
- **Effort:** ~3–4 weeks (builds on foundation backend)
- **Success criteria:** 
  - An example weave builds a foundation model → SFTs it on Alpaca
  - Gates verify accuracy on GLUE >=0.7 (achievable bar)
  - Report shows mixing of pretraining loss + SFT loss

**1.2 Implement DPO (Direct Preference Optimization) Backend**
- **Why second:** Lighter than RLHF, no reward model bottleneck, replaces RLHF for v1
- **Scope:**
  - Add `dpo` section to weave.yaml
  - Implement `src/loom/dpo.py`
  - Reuse SFT model as base
  - Gates: preference pair coverage, win rate, diversity
- **Effort:** ~3 weeks (DPO is simpler than RLHF; no separate reward trainer)
- **Success criteria:**
  - Example weave: pretraining → SFT → DPO
  - Gates verify preference win rate >= 0.60

#### **PHASE 2 (Months 2–3): Fix Architecture Abstraction**

**2.1 Add Architecture Templates**
- **Why:** Foundation backend hardcodes GPT-2-style; cannot build BERT, GLM, Mamba
- **Scope:**
  - Add `architecture:` field to weave.yaml (options: "decoder", "encoder", "encoder_decoder", "s6")
  - Parameterize foundation backend to choose model class
  - Implement BertForMaskedLM, GLMForConditionalGeneration, S6ForCausalLM
  - Update gates to architecture-specific metrics (BERT: masked accuracy; S6: perplexity)
- **Effort:** ~4 weeks (2 weeks per new architecture)
- **Success criteria:**
  - Can build a BERT model on BabyLM (encoder, masked LM objective)
  - Can build a Mamba S6 decoder (S6 blocks + LM head)
  - Both pass quality gates

**2.2 Implement Data Curation Pipeline**
- **Why:** No control over data mixing, filtering, or domain balance
- **Scope:**
  - Add `data:` section to weave.yaml with mixing, filtering, deduplication
  - Implement `src/loom/data_ops.py`: compose multiple corpora, filter by length/quality, deduplicate
  - Gates on data: diversity (distinct n-grams), quality (no code in prose corpus)
- **Effort:** ~2 weeks
- **Success criteria:**
  - Weave specifies "80% BabyLM, 20% StackOverflow" → foundation trains on mixed corpus
  - Gates verify diversity metrics

#### **PHASE 3 (Months 3–4): Fix the "Rebuild" Promise**

**3.1 Implement Continued Pretraining Backend**
- **Why:** Port backend only steers frozen models; cannot "rebuild" via retraining
- **Scope:**
  - Load an existing HF model (e.g., TinyLlama)
  - Fine-tune its embeddings/transformer weights on new domain data
  - Emit gates for perplexity on new domain
  - This is "continued pretraining" = SFT at the base model level
- **Effort:** ~2 weeks
- **Success criteria:**
  - Load Llama-2-7b, continue-pretrain on legal docs, verify val_ppl on legal test set

**3.2 Implement Model Surgery (Selective Weight Editing)**
- **Why:** "Rebuild" implies architectural modification, not just steering
- **Scope:**
  - Add `surgery:` section to weave.yaml
  - Allow: layer insertion, layer deletion, head pruning, attention-to-linear-attention swap
  - Implement in `src/loom/surgery.py` with fine-tuning to restore performance
  - Gates: downstream task accuracy, parameter count reduction
- **Effort:** ~4–5 weeks
- **Success criteria:**
  - Remove layer 20 from Llama-2-7b, fine-tune, verify <2% accuracy drop
  - Prune heads, measure inference speedup

#### **PHASE 4 (Months 4–5): Make It Truly Layman-Accessible**

**4.1 Add Imperative Control Flow**
- **Why:** weave.yaml is declarative; laypeople expect `if`, `for`, composition
- **Scope:**
  - Extend weave language with basic logic: `if` (based on gate results), loops over skill sets
  - Implement `src/loom/flow.py`: interpret simple conditionals
  - Allow: "if foundation.val_ppl > 20, increase model size and retrain"
- **Effort:** ~3 weeks
- **Success criteria:**
  - A weave with `if gate_failed: { retry with larger_model }`
  - Compiler executes the conditional, re-runs training if needed

**4.2 Build Interactive Explainer UI**
- **Why:** "explain" command is text-only; laypeople need visual understanding
- **Scope:**
  - Render a weave as a flowchart (pretraining → SFT → DPO → gate check)
  - Show gate thresholds, data flow
  - Interactive: click to expand skill details
- **Effort:** ~2–3 weeks (HTML+JS, pull from report.json)

#### **PHASE 5 (Months 5–6): Validation & Hardening**

**5.1 E2E Test: Build a Real Useful LLM**
- **Why:** All prior phases are components; v1 needs an integrated, real end-to-end pipeline
- **Scope:**
  - Declare a 100M-parameter model in weave.yaml
  - Pretrain on BabyLM (~24 hours on RTX 5090)
  - SFT on Alpaca (~4 hours)
  - DPO on preference data (~2 hours)
  - Gates: GLUE acc >= 0.75, refusal rate >= 0.9
  - Total: ~30 hour job, achievable with remote RTX 5090
- **Effort:** ~1 week (integration, not new code)
- **Success criteria:**
  - Report: ✅ all gates passed (or ❌ refused with diagnosis)
  - Model inference works: generate text, refuse harmful, answer questions

**5.2 Robustness Audit: Red-Team the System**
- **Why:** Current system passed own red-team (REDTEAM.md); now test new backends
- **Scope:**
  - Can vacuous gates slip through? Test edge cases
  - Can side-effects hide? Measure across tasks
  - Can gates be gamed (high reported metric, bad actual performance)? Adversarial test
- **Effort:** ~2 weeks
- **Success criteria:**
  - Find and fix at least 3 edge cases before release

---

### DELETE or DEMOTE

**1. Curriculum Backend (toy focus)**
- Currently trains on synthetic induction/state_tracking tasks
- 128-dim models do not scale to language understanding
- **Action:** Demote to "demo" backend; mark as proof-of-concept
- **Keep:** The skill compiler architecture (InductionCompiler, StateTrackingCompiler) is clever; generalize to real tasks (e.g., entity extraction compiler)

**2. Constructed Backend (hand-compiled circuits)**
- Proof that weight modification works; low practical use
- **Action:** Move to `examples/` directory; keep for reference
- **Rationale:** It demonstrates the layer, but cannot be scaled to real models

**3. Vacuous SFT/RLHF Stubs**
- `src/loom/` has comments hinting at RLHF; no implementation
- **Action:** Remove stubs; they create false hope

---

### Resource Estimate

| Phase | Duration | GPU Cost | Effort |
|-------|----------|----------|--------|
| 1: SFT + DPO | 6 weeks | RTX 5090: ~500 hours (SFT $300, DPO $200) | 2 engineer-months |
| 2: Architectures | 6 weeks | RTX 5090: ~1000 hours | 2 engineer-months |
| 3: Surgery + Rebuild | 8 weeks | Mix of GB10 + RTX 5090 | 2.5 engineer-months |
| 4: Flow + UI | 5 weeks | None (local compute) | 1.5 engineer-months |
| 5: Validation | 3 weeks | RTX 5090: ~400 hours | 1 engineer-month |
| **Total** | **~6 months** | **~$2000 in GPU** | **9 engineer-months** |

### What NOT to Do

1. **Scale to production models (100B+).** Loom is a research tool for understanding + building small LLMs. Do not attempt to build GPT-4 in Loom; it will fail on infrastructure and cost grounds.

2. **Add RLHF before DPO.** RLHF requires a separate reward model trainer, value function, PPO critic—high complexity and GPU cost. DPO is 1/4 the code and runs on same GPU as SFT. Start with DPO.

3. **Solve the "generalist model" problem.** Loom compiles declarative skill specs. It cannot yet say "build a model that is good at 100 tasks simultaneously." Start with domain-specific models.

4. **Target 0.1% improvement in GLUE accuracy.** Loom's gates should be achievable bars (70% acc, <10 nats loss), not aspirational (86% acc). Set gates from a baseline run, not from literature.

---

## Verdict Table

| Question | Aspect | ASKED | BUILT | GAP | SEVERITY |
|----------|--------|-------|-------|-----|----------|
| **A** | Language: imperative like C | Yes | No (declarative YAML only) | No variables, control flow, functions | **HIGH** |
| **B** | Build LLM: data | Flexible curation | Hardcoded BabyLM | No curation pipeline | **MEDIUM** |
| **B** | Build LLM: pretraining | GPT/BERT/GLM agnostic | GPT-2 decoder only | Welded to one arch | **HIGH** |
| **B** | Build LLM: SFT | Yes (essential stage) | No backend exists | Complete absence | **CRITICAL** |
| **B** | Build LLM: RLHF/DPO | Yes (explicit in intent 4) | No backend exists | Complete absence | **CRITICAL** |
| **B** | Build LLM: post-training pipeline | Data→Pretrain→SFT→DPO | Only Pretrain works | 75% of pipeline missing | **CRITICAL** |
| **C** | Architecture: GPT, BERT, Qwen, etc. | All major families | Port: ✅ (any HF model), Foundation: ❌ (GPT-2 only) | Foundation cannot target alternate archs | **MEDIUM** |
| **D** | Rebuild: surgery, merging, distillation | Yes (6 methods) | Port only steers frozen weights | No weight modification | **HIGH** |
| **E** | Verification gates | Valuable design | ✅ Implemented, working | None | — |
| **E** | ABI + linker | Valuable design | ✅ Implemented, sketched | None | — |
| **E** | Real measured results | "No fabrication" | ✅ All claims have trace files | None | — |
| **F** | Layman-accessible | No expertise needed | ⚠️ Declarative (easy to read, hard to compose) | No imperative control flow; config feels like a black box | **MEDIUM** |
| **G** | Course-correction plan | N/A | Not in scope | — | — |

---

## Single Most Important Gap

**The system cannot build a complete, usable LLM.**

It can pretrain (foundation backend: runs, honest gate failures). It cannot post-train (no SFT/RLHF/DPO backends). Without post-training, a pretrained model is a research artifact, not an application. A layperson following the stated intent (point 4: "build an LLM...data+pretraining+post-training") will hit a wall at SFT and find no code path forward.

**This is the load-bearing failure.** Fix this first.

---

## Top 3 Course Corrections (Priority Order)

### 1. **Implement SFT (Supervised Fine-Tuning) Backend**
- **Why:** Bridges pretraining to usable models. Highest ROI.
- **Scope:** `src/loom/sft.py` + weave.yaml section, gates on downstream accuracy
- **Timeline:** 3–4 weeks
- **Hardware:** GB10 (local) for small runs; RTX 5090 for larger domain SFT

### 2. **Implement DPO (Direct Preference Optimization) Backend**
- **Why:** Replaces RLHF without reward model overhead. Completes the post-training pipeline.
- **Scope:** `src/loom/dpo.py`, gates on preference win rate
- **Timeline:** 3 weeks
- **Hardware:** RTX 5090 for preference pair collection

### 3. **Generalize Foundation Backend to Multiple Architectures**
- **Why:** Currently locks into GPT-2 decoder. User asked for BERT/GLM/others.
- **Scope:** Add `architecture:` field; implement BertForMaskedLM + GLMForConditionalGeneration templates
- **Timeline:** 4 weeks (1 per architecture family, 3 families)
- **Hardware:** RTX 5090 for BERT (masked LM training differs from causal LM)

After these three, the system would be capable of building an actual, useful LLM end-to-end on any major architecture.

---

## Recommendations for Communication

1. **Rebrand the marketing.** The index.html says "programming the transformer" and compares Loom to compilers. This is aspirational but misleading. Honest reframe: "Loom is a declarative configuration system + verification compiler for building and steering small transformers." Drop the "C for LLMs" metaphor; it sets false expectations.

2. **Document the constraints.** Update README.md:
   - "v0 targets models up to 100M parameters"
   - "Suitable for research, domain-specific tuning, safety steering"
   - "Not intended for foundation-model-scale training (>1B params)"

3. **Publish the gap list.** Write LIMITATIONS.md:
   - SFT: not yet implemented (planned)
   - RLHF: not yet implemented (planned for DPO)
   - Encoder-only (BERT): foundation backend targets decoders only
   - Layer surgery: only steering, no weight modification

4. **Make "refused by compiler" a feature, not a bug.** When gates fail, the refusal diagnosis is gold. Highlight it in the docs: "Honest refusal beats silent failure."

---

## Notes for Future Audits

- **H5 (Sparsity) remains undecided** (VALIDATION.md:60–90); retest with description-length metric, not weight counting.
- **Port backend mechanics (prayoga + prabodha imports)** are well-documented (INTEGRATION.md) but not red-teamed on real models; test on Gemma-2, Qwen2.5, Nemotron.
- **Gate vacuity checks** (spec.py:86–119) are excellent; they caught impossible thresholds. Extend to impossible gate *combinations* (e.g., cannot have both high accuracy and low latency on same model).
