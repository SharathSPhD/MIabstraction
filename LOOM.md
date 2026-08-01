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

## Why this is not a wrapper

A wrapper would map each clause to one fixed recipe. The compiler chooses, per substrate,
per capability, and reports what it chose — which is the difference between a build system
and a compiler. And because L1 is the mech-interp layer, the compiler can realize behaviour
by *editing the model's internals* — installing a verified circuit, steering a measured
feature — not only by throwing data at it. That is the capability no existing tool has,
and it is why the abstraction layer had to be validated before the language could exist.
