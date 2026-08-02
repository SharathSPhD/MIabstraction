# Loom — a programming language for building language models

## The premise

Building a language model — from scratch, or by adapting an open-weight one — is out of
reach for anyone who is not an AI engineer. Not because the goal is unclear, but because
the path from "I want a model that does X" to a working model runs through data curation,
architecture choice, pretraining schedules, fine-tuning, preference optimization, and
safety tuning. Each is a specialism.

C did this once before. Machine code was out of reach for anyone who was not an electrical
engineer, until a language existed whose constructs were *what programmers think about* —
variables, functions, loops — and a compiler that turned those into instructions for
whatever processor was in the machine. The programmer stopped thinking about registers.

Loom is that language for language models. You write what the model must know, how it must
behave, and what it must never do. The compiler decides everything else, and hands you a
model.

## The stack

Four levels. Only the top one is written by a person.

```
L3  APPLICATION SOURCE          app Tutor { knows ...; speaks ...; never ...; }
      │                          what a person writes: knowledge, behaviour, guardrails
      │  semantic analysis
L2  CAPABILITY GRAPH            knowledge(corpus) · style(patient) · refuse(off-topic)
      │                          architecture-free statement of what the model must do
      │  lowering through the mechanistic-interpretability layer
L1  MECH-INTERP IR              feature(style) · steer(dose) · circuit(retrieval)
      │                          the assembly language of transformers — validated units
      │  strategy selection against the substrate's capability table
L0  SUBSTRATE OPERATIONS        pretrain · finetune · adapter · graft · hook
      │                          only the operations this substrate actually permits
      ▼
    WEIGHTS + HOOKS → a model you can run
```

L1 is the layer this project spent its first half validating: features are real and
readable, circuits are real and installable, steering is real and dose-dependent. It is
Loom's assembly language. A person never writes it, exactly as a C programmer never writes
`mov`.

## Substrates have different capabilities, and the compiler knows it

This is the crux, and it is the same problem a C compiler solves when targeting two
processors. x86 has an instruction ARM lacks; the compiler does not refuse, it selects a
different lowering that achieves the same semantics.

| Lever | from-scratch | open-weight |
|---|---|---|
| choose architecture | yes | **no — fixed by whoever trained it** |
| choose tokenizer | yes | no |
| pretrain on chosen data | yes | only as continued pretraining |
| fine-tune | yes | yes |
| install a compiled circuit | yes | yes, within its verified envelope |
| steer a feature at runtime | yes | yes |
| attach an adapter | yes | yes |

So `knows facts from "docs/"` lowers differently on each: on a from-scratch target it
enters the pretraining mixture; on an open-weight target the architecture is frozen, so the
compiler reaches the same capability through continued pretraining, an adapter, or a
retrieval circuit — whichever the strategy table says is cheapest and sufficient.

**The program does not change. The lowering does.**

## The language (L3)

```loom
app Tutor {
    knows   facts from "corpus/*.txt";           // what it must know
    knows   how to explain step by step;

    speaks  patient, concise;                    // how it must behave
    always  cites the source it used;
    never   discusses pricing;                   // what it must not do
    refuses topics outside its corpus;

    expect  answers("what is X?") mentions "definition";   // how you know it worked
    expect  refuses("what is your pricing?");
}

build Tutor on scratch(size = small);            // every lever available
build Tutor on "Qwen/Qwen3-0.6B";                // architecture fixed
```

Nothing in that program mentions a gradient, a layer, a learning rate, an adapter rank, or
a preference pair. Those are the compiler's business, the way register allocation is a C
compiler's business.

Each clause has a defined meaning at L2:

| You write | Capability | Typically lowered to |
|---|---|---|
| `knows facts from <corpus>` | knowledge(corpus) | pretraining mixture · continued pretraining · retrieval circuit |
| `knows how to <skill>` | skill(name) | curriculum · fine-tuning · installed circuit |
| `speaks <style>` | style(traits) | style feature + calibrated steering · fine-tuning |
| `always <constraint>` | invariant(c) | monitor + steering, verified on held-out data |
| `never <topic>` | prohibition(t) | topic feature + suppression, with a side-effect bound |
| `refuses <class>` | guardrail(class) | refusal feature amplified to a calibrated dose |
| `expect <behaviour>` | acceptance test | measured after the build; reported, not assumed |

## What the compiler does

1. **Parse and analyse** the app into a capability graph — no architecture anywhere.
2. **Plan**: for each capability, consult the substrate's capability table and choose a
   realization strategy. Record the choice and the reason.
3. **Lower** each strategy into mech-interp operations (L1) and then into substrate
   operations (L0).
4. **Execute** on the available hardware, choosing the machine per job.
5. **Verify** each `expect` against held-out data, and emit the model with its report.

The output is a directory containing weights, hooks, the plan that produced them, and the
measurements. `loom run <app>` gives you a prompt.

## What each substrate actually does

Stated plainly, because the two are not the same claim and only one of them is
"creating a language model".

**From scratch** (`scratch(demo)`, `scratch(flagship)`) — no downloaded weights are
involved. The compiler chooses the architecture from the program's declared demands,
trains a tokenizer on the program's own corpus, pretrains, and can build the host
*inside a compiled circuit's verified envelope* so the circuit can be grafted. The
artifact carries the weights, because there is no upstream repository to fetch them
from later. This is the substrate the project's central claim is about.

**Open weight** (any cached model) — the architecture and tokenizer were fixed by
whoever trained the weights. What the compiler does here is choose, per capability,
between a LoRA adapter and a calibrated steering write, having first measured what
the capability is worth in nats and consulted a ledger of what a write has actually
delivered on this base model. That is fine-tuning and inference-time steering, and
calling it anything else would be untrue. What is not standard is the decision
procedure: measure the gap before choosing the lever, refuse a lever the measurement
says is too small, verify on the composed model, and refuse to ship a control that
only moves the probes it was derived from.

**Both** carry declared policy without compiling it. See below.

## Why this is not a wrapper

A wrapper maps each clause to one fixed recipe. The compiler measures first and
chooses, per substrate and per capability, and reports what it chose and why. Two
things it does that a training script does not: it **links a separately compiled
circuit** through the ABI — the induction unit is grafted with no gradient taken
anywhere, fires only where its condition holds, and is refused outright when the host
falls outside the envelope it was verified on — and it **declines to build refusal
into weights at all**, because every dose strong enough to refuse off-subject
questions also refused in-subject ones.

Circuits, the ABI and the linker were research components for most of this project's
life; `src/loom/app/linking.py` is where a build path finally calls them, and the
numbers it produces are in the build reports rather than in an experiment folder.

## Composition, on both substrates

An abstraction layer that can only ship one skill per model is a library of one book, so
composition is measured on both substrates rather than assumed:

- **Constructed (scratch)**: trigram induction and a succession rule compiled into one
  3-layer weight set (`compile_composed`), the second skill living entirely in the 32
  residual dimensions the induction memory map left unused. Measured
  (`results/loom_composed_demo.json`): succession exact at 1.0; induction 0.909 alone and
  0.909 composed; on letter traffic the composed model's logits match the single-skill
  model's to 7e-13 with identical argmax — in exact arithmetic they are equal, since the
  only thing LayerNorm adds to an untouched block is a scalar shift and a zero-mean read
  code annihilates it; where the two skills disagree, the declared arbitration wins; a
  random model fails both gates at chance.

- **Open-weight**: a clinic build composes a knowledge adapter and calibrated
  steering controls in one model. It no longer composes a guardrail adapter: refusal
  is not compiled here. Controls are calibrated jointly (three
  controls each measuring zero side-effect alone destroyed generation once composed), and
  every expectation is verified on the composed artifact, not on any capability alone.
