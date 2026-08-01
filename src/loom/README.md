# Loom — a programming layer for transformers

Loom is the abstraction layer this project set out to discover: the missing middle between
"collect data, run gradient descent, hope" and the application. It plays the role that
assembly + compiler play for CPUs, built on the empirically validated substrate from the
MIabstraction experiments:

| Validated result | What it becomes in Loom |
|---|---|
| H1: residual stream linearly encodes world state (R²=0.998) | **variables** — named, probe-readable state |
| H2: capabilities form as discrete circuits (induction) | **instructions** — skills that verifiably exist |
| H4: linear probes read known concepts as well as anything | **the debugger** — monitors on internal state |
| Steering / activation intervention | **write access** — runtime controls |
| Weight sparsity ~free at task accuracy | structure can be **imposed**, not just found |

## The stack

```
Application        "a model that copies patterns, tracks state, never says token 2"
   ↓  weave.yaml   (the program — written by a non-expert)
Loom compiler      parse → plan → build (per-skill backend) → verify gates → report
   ↓
Backends           curriculum   (compile to a training objective)
                   constructed  (compile directly to weights, no training)
                   runtime      (compile to probes + steering hooks)
   ↓
TinyTransformer    weights + hooks     ("machine code")
   ↓
GPU
```

## The language (weave.yaml)

A *weave* declares four things. The user never mentions gradients, losses, layers, or
tokens' embedding geometry:

```yaml
model: {size: small}          # small|medium — Loom picks architecture

skills:                       # capabilities the model MUST have
  - name: copy_patterns
    kind: induction           # in-context copying of repeated structure
  - name: track_state
    kind: state_tracking      # maintain belief over a hidden process
    world: mess3
  - name: parity
    kind: classify            # answer a query about the sequence
    concept: token_parity

controls:                     # runtime write-access
  - name: never_token_2
    kind: suppress
    token: 2

monitors:                     # runtime read-access (the debugger)
  - name: state_monitor
    kind: probe
    concept: belief_state

gates:                        # the promise — build FAILS if any gate fails
  copy_patterns: {prefix_score: ">0.5", icl_loss: "<0.5"}
  track_state:   {probe_r2: ">0.9"}
  parity:        {accuracy: ">0.95"}
  never_token_2: {suppression_ratio: ">0.9", side_effect: "<0.1"}
```

`loom build weave.yaml` returns either a **verified artifact** (weights + hooks +
`report.json` describing what was installed, with measured evidence per gate) or a
**refusal with diagnosis**. There is no "trained but who knows" state. That is the
ralph-loop promise implemented as a compiler property.

## Why gates are the load-bearing idea

Every MIabstraction experiment produced at least one *confident wrong answer* before its
controls caught it (reservoir probes, positional shortcuts, tautological metrics). A layer
for non-experts cannot assume the user will catch these. So verification is not a testing
afterthought — it is the compiler's contract, and each gate metric is one that survived
adversarial audit in the validation phase (incremental-R² probes, mechanism-level prefix
scores, side-effect budgets on controls).

## Backends

- **curriculum** (default): compiles the skill set into a mixed synthetic curriculum and a
  multi-task training run. The compiler owns task tokens, mixing ratios, and stopping.
- **constructed**: compiles a skill *directly into weights* (Tracr-style, hand-derived
  circuits — e.g. the previous-token + induction head composition), no training. Proof that
  the layer can write machine code, not just steer SGD.
- **runtime**: compiles controls/monitors into forward hooks — steering vectors computed
  from contrastive activations, probes fitted on collected activations.

## Status

v0 targets the primitives above on TinyTransformer-scale models, verified end-to-end on a
single GPU. The point is not scale — it is the existence proof that a declared program plus
a verifying compiler can replace expertise for this class of customization.

## Known limitations (red-teamed, documented rather than hidden)

The adversarial audit (REDTEAM.md) found five holes. Three are FIXED and locked by
regression tests (vacuous gates refused; monitors demand ground-truth labels; controls
without an active mechanism refuse to install). Two are inherent limits of v0, stated
here because a limitation you know about is a spec, and one you don't is a trap:

- **Token-level, not concept-level, suppression.** `suppress: token` guarantees the
  token is not emitted; it does not stop the model expressing the same *meaning* through
  other tokens. Concept-level control needs representation-level targets and its own
  gate design.
- **Gates certify the verified distribution.** Skill and monitor gates hold on the
  distributions they were measured on (declared copy lengths, gap ranges, world
  parameters). Out-of-distribution inputs degrade silently. Verification-carrying
  artifacts state their measured envelope in report.json; using a model outside it is
  outside the promise.
