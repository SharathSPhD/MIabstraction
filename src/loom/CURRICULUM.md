# Loom Curriculum Backend

The curriculum backend transforms a `WeaveSpec` (the Loom language) into a trainable, verifiable multi-task learning setup.

## Architecture

```
WeaveSpec (YAML)
    ↓
compile_curriculum() → CurriculumPlan
    ↓
train(spec, plan, device) → (model, metrics, losses)
```

Each skill declares:
- A **generator**: creates synthetic sequences for that task
- An **evaluator**: measures gate metrics (e.g., `prefix_score`, `probe_r2`, `accuracy`)

All skills train on a single shared `TinyTransformer` with per-skill vocabulary ranges.

## Task Compilers

### InductionCompiler

**Task**: Learn to copy patterns via in-context learning (attention heads).

**Sequence structure**: 
```
[copy_len tokens] [gap] [copy_len tokens] [padding]
```
The second copy is variable-gap to defeat positional shortcuts.

**Metrics**:
- `prefix_score`: Mean attention mass on induction targets, max over heads/layers. Range [0, 1].
- `icl_loss`: Cross-entropy loss on the second-copy region (nats).

**Gate example**: `prefix_score > 0.5, icl_loss < 0.5`

---

### StateTrackingCompiler

**Task**: Track hidden state of a Markov process (Mess3).

**Sequence structure**:
```
Tokens sampled from Mess3 HMM with x=0.05, a=0.85.
Ground truth: belief state P(hidden_state | tokens[:t+1]) for each position.
```

**Metrics**:
- `probe_r2`: Linear probe R² from best layer residual to ground-truth beliefs. Range [0, 1].
- `probe_r2_incremental`: R² gain beyond an 8-token window reservoir baseline (defeats reservoir fallback).
- `best_layer`: Which layer was probed.

**Gate example**: `probe_r2 > 0.9`

---

### ClassifyCompiler

**Task**: Predict parity (even/odd) of a target token's occurrences in a sequence.

**Sequence structure**:
```
[32 content tokens] [MARKER] [target_token_id] [parity_answer_token]
```
Next-token prediction must learn that the parity answer depends on counting occurrences of `target_token_id` in the content.

**Metrics**:
- `accuracy`: Fraction of correct parity predictions (range [0, 1]).

**Gate example**: `accuracy > 0.90`

---

## Vocabulary Plan

Skills share ONE model vocabulary. The plan allocates:

```python
plan.vocab_plan.skills = {
    "copy_patterns": {"token_start": 10, "token_end": 266, "n_tokens": 256},
    "state_track": {"token_start": 266, "token_end": 522, "n_tokens": 256},
    "parity": {"token_start": 522, "token_end": 778, "n_tokens": 256},
}
plan.vocab_plan.task_tokens = {
    "copy_patterns": 0,  # Unique task prefix token
    "state_track": 1,
    "parity": 2,
}
plan.vocab_plan.total_vocab = 778
```

Each skill's generator/evaluator respects these ranges.

---

## API

### compile_curriculum()

```python
def compile_curriculum(
    spec: WeaveSpec,
    max_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
) -> CurriculumPlan:
```

**Input**: A `WeaveSpec` from `load_weave(weave_path)`.

**Output**: `CurriculumPlan` with:
- Vocabulary allocation
- Per-skill dataset configs
- Mixing weights (uniform by default)
- Training hyperparameters
- Gate thresholds

**Usage**:
```python
from loom.spec import load_weave
from loom.curriculum import compile_curriculum, train

spec = load_weave("my_weave.yaml")
plan = compile_curriculum(spec, max_steps=5000, batch_size=64, lr=1e-3, device="cuda")
model, metrics, losses = train(spec, plan, device="cuda")
```

---

### train()

```python
def train(
    spec: WeaveSpec,
    plan: CurriculumPlan,
    device: str = "cuda",
) -> tuple[TinyTransformer, dict, list]:
```

**Training loop**:
1. Initialize TinyTransformer with `plan.vocab_plan.total_vocab`
2. For each step: sample a skill, sample a batch, forward, backward, step
3. Evaluate all skills every 100 steps
4. Early stop if all gates pass (after step 500)

**Output**:
- `model`: Trained TinyTransformer
- `metrics`: dict[skill_name -> dict[metric_name -> float]]
- `losses`: Training loss history

**Example**:
```python
model, metrics, losses = train(spec, plan, device="cuda")
print(metrics)
# {
#     "copy_patterns": {"prefix_score": 0.62, "icl_loss": 0.41},
#     "state_track": {"probe_r2": 0.91, "probe_r2_incremental": 0.05, "best_layer": 2},
#     "parity": {"accuracy": 0.95},
# }
```

---

### evaluate_curriculum()

```python
def evaluate_curriculum(
    model: TinyTransformer,
    plan: CurriculumPlan,
    device: str,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
```

Evaluate all skills on new data (used during training for gate checking).

---

### check_gates()

```python
def check_gates(
    metrics: dict[str, dict[str, float]],
    gate_metrics: dict[str, dict[str, dict]],
) -> bool:
```

Return `True` iff all gates hold. Used for early stopping.

---

## Example Weave

```yaml
name: two_skill_demo
model: {size: small}  # d_model=64, n_layers=4, n_heads=4, max_len=128

skills:
  - name: patterns
    kind: induction
  - name: hidden_state
    kind: state_tracking
    world: mess3

gates:
  patterns:
    prefix_score: ">0.5"
    icl_loss: "<0.5"
  hidden_state:
    probe_r2: ">0.9"

seed: 42
```

Then:
```python
spec = load_weave("two_skill_demo.yaml")
plan = compile_curriculum(spec, max_steps=5000)
model, metrics, losses = train(spec, plan)
```

---

## Design Notes

1. **Fixed-offset trap avoided**: Induction uses variable-gap sequences so models must learn content-based attention, not positional shortcuts.

2. **Incremental probes**: State tracking includes `probe_r2_incremental` to detect and penalize reservoir fallback (fitting only recent history).

3. **Hand-solvable tasks**: Each task is verified to be achievable by a trained solver (see `tests/test_curriculum_hand_solved.py`).

4. **Shared model, separate metrics**: One transformer, per-skill generators/evaluators. This mirrors the Loom philosophy: compile, verify, report.

5. **Early stopping on gates**: Training halts when all gates pass, wasting no compute on over-training.

---

## Testing

```bash
# Run all curriculum tests
pytest tests/test_loom_curriculum.py tests/test_curriculum_hand_solved.py -v

# Run the demo
python demo_train_curriculum.py
```

Tests cover:
- Generator well-posedness (sequences are valid)
- Evaluator alignment (metrics correlate with learning)
- End-to-end training on CPU and GPU
- Hand-trained solvers achieve metric ceilings
