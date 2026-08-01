# Loom — a language for constructing language models

## What this is

A high-level language whose programs *build LLMs*. You write source code describing data,
pretraining, and post-training; a compiler type-checks it, lowers it to an intermediate
representation, and executes it against a real substrate — either an architecture built
from scratch or an existing open-weight model.

The relationship to C is structural, not cosmetic:

| C | Loom |
|---|---|
| source statements | stages you write and sequence |
| types (`int`, `char*`) | `corpus`, `tokenizer`, `target`, `model`, `evalset` |
| functions | `pretrain`, `finetune`, `align`, `merge`, `graft`, `quantize` |
| standard library | recipes: `std.pretrain.chinchilla`, `std.align.dpo` |
| separate compilation + linker | verified units + the residual-stream ABI |
| LLVM IR | the stage graph (typed DAG) |
| target ISA (x86, ARM) | target architecture (a from-scratch decoder, Qwen, Gemma, Nemotron, DeepSeek, GLM, GPT-BERT) |
| `assert.h` | `assert` statements — gates, measured, build-failing |

The last row of that table is the whole point. **The same program compiles to different
transformer substrates**, because everything above `target` is written against
architecture-independent operations. That is the abstraction layer.

## A complete program

```loom
// smalltalk.loom — a small conversational model, built from nothing.

target arch = decoder(layers=12, width=768, heads=12, ctx=1024);

corpus speech = data.text("babylm:childes") + data.text("babylm:switchboard");
corpus prose  = data.text("babylm:simple_wiki").filter(len > 64);
corpus mix    = speech * 0.7 + prose * 0.3;

tokenizer tk = tokenizer.bpe(mix, vocab=16000);

model m = pretrain(arch, mix, tk) {
    tokens    = 800M;
    optimizer = adamw(lr=6e-4, wd=0.1);
    schedule  = cosine(warmup=2%);
};

assert perplexity(m, mix.heldout) < 60;      // measured, or the build fails

m = finetune(m, data.chat("dialogues.jsonl")) {
    epochs = 2;
    lr     = 2e-5;
};

m = align(m, data.prefs("preferences.jsonl")) {
    algo = dpo;
    beta = 0.1;
};

assert winrate(m, baseline=m.before_align) > 0.55;

export m to "smalltalk-v1";
```

The same file, with **one line changed**, rebuilds an existing open-weight model instead:

```loom
target arch = load("Qwen/Qwen3-0.6B");   // everything below is unchanged
```

That is what "the abstraction sits above the architecture" has to mean in practice: the
program does not know or care whether its substrate is a decoder you specified or a
1.5-billion-parameter model someone else trained.

## Types

| Type | Is | Produced by |
|---|---|---|
| `target` | a substrate: architecture spec **or** loaded open weights | `decoder(...)`, `load("org/model")`, `gptbert(...)` |
| `corpus` | a composable stream of text | `data.text(...)`, `data.chat(...)`, `data.prefs(...)`, `+`, `*`, `.filter`, `.dedup` |
| `tokenizer` | a vocabulary and its encoder | `tokenizer.bpe(corpus, vocab=N)`, `tokenizer.load(...)` |
| `model` | weights plus everything known about them | `pretrain`, `finetune`, `align`, `merge`, `graft` |
| `evalset` | inputs with ground truth | `eval.blimp()`, `corpus.heldout`, `eval.jsonl(...)` |
| `unit` | a separately compiled, verified circuit | `unit.load(...)`, `unit.construct(...)` |

Types are checked before anything runs. `finetune(corpus, model)` — arguments swapped —
is a compile error, not a crash forty minutes into a job.

## Corpus algebra

Data is a first-class value with real operators, because data decisions are the largest
lever in model construction and deserve to be written down rather than buried in a script.

```loom
corpus mix = web * 0.6 + code * 0.3 + math * 0.1;   // mixing weights, normalized
corpus clean = mix.dedup(ngram=13).filter(quality > 0.7).shuffle(seed=0);
corpus train, held = clean.split(0.99);             // contiguous, no leakage
```

## Stages

Each stage is a function from `model` to `model`, so pipelines compose:

- `pretrain(target, corpus, tokenizer) { ... }` — from scratch, or continued on loaded weights
- `finetune(model, corpus) { ... }` — supervised instruction tuning
- `align(model, corpus) { algo = dpo | ppo | orpo; ... }` — preference optimization
- `distill(student, teacher, corpus) { ... }`
- `merge(a, b) { method = slerp | ties | linear; ... }`
- `graft(model, unit) { budget = 0.05; }` — link a separately compiled circuit (the ABI)
- `quantize(model) { bits = 4; }`
- `export model to "name"`

## Assertions are the contract

`assert` is a statement, and a failed assertion fails the build. This is the part of the
earlier declarative design that survives unchanged, because it solved a real problem: a
model that was trained but never checked is not a deliverable.

```loom
assert perplexity(m, held) < 30;
assert accuracy(m, eval.blimp()) > 0.6;
assert refusal_rate(m, eval.harmful()) > 0.95;   // and side-effects are bounded too
```

Assertions carry provenance: each records the artifact, the measurement, and the
conditions it was measured under.

## Compilation pipeline

```
source.loom
   │  lex + parse
   ▼
AST ──► typecheck ──► desugar (std recipes expand)
   │
   ▼
STAGE GRAPH (IR)          typed DAG: data → tokenizer → pretrain → finetune → align → assert
   │  target lowering
   ▼
BACKEND PLAN              scratch: build modules, emit training loop
                          hf:      load weights, map stages onto the real architecture
   │  compute planning
   ▼
EXECUTION                 dispatch: GB10 (large frozen models) | RTX 5090 (throughput)
   │
   ▼
ARTIFACT + REPORT         weights, provenance, every assertion with its measurement
```

`loom build prog.loom --emit ir` prints the stage graph without running it, so you can see
what the compiler decided before you spend GPU hours. `--dry-run` additionally reports the
estimated compute and where each stage would run.

## Backends (targets)

| Backend | Substrate | Status |
|---|---|---|
| `scratch` | a decoder you specify — the model is constructed and trained | pretraining path proven: a 30M model, 97M tokens of real text, on the 5090 |
| `hf` | any HuggingFace causal LM: Qwen, Gemma, Nemotron, DeepSeek, GLM, Llama | frozen-model programming proven; full stage support is the current build |
| `gptbert` | the masked/causal hybrid used in BabyLM work | planned |

A backend must implement a small interface — build or load, expose blocks and the residual
stream, run a training step, save. Nothing above that interface knows which one it is.

## What the earlier declarative form becomes

`weave.yaml` is not discarded — it becomes the easy front door. A weave desugars into a
Loom program:

```yaml
skills: [{name: chat, kind: instruct}]
gates:  {chat: {winrate: ">0.55"}}
```
desugars to a program that loads a default target, applies `finetune` with a standard
recipe, and asserts the gate. Same compiler, same IR, same execution. Declarative for
people who want defaults; the language for people who want control — which is exactly the
relationship between a build file and the language it builds.
