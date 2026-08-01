# Loom grammar and IR — the implementable spec

Deliberately small. Everything here is checkable by a parser and a type checker; nothing
requires the compiler to guess what the author meant.

## Lexical

```
ident     ::= [A-Za-z_][A-Za-z0-9_]*
number    ::= digits ('.' digits)? ('%' | [KMBT])?     // 2%, 800M, 3e-4, 0.7
string    ::= '"' ... '"'
comment   ::= '//' ... EOL
```

`800M`, `1.5B`, `2%` are literals with units, because token counts and warmup fractions are
where silent unit errors hide.

## Grammar (EBNF)

```ebnf
program     ::= stmt* ;

stmt        ::= decl | assign | assert_ | export | import_ ;

decl        ::= type ident '=' expr ';' ;
type        ::= 'target' | 'corpus' | 'tokenizer' | 'model' | 'evalset' | 'unit' ;
assign      ::= ident '=' expr ';' ;                    // rebind, same type
assert_     ::= 'assert' expr cmp expr ';' ;
export      ::= 'export' ident 'to' string ';' ;
import_     ::= 'import' string ';' ;                   // another .loom file

expr        ::= call | binop | member | literal | ident ;
call        ::= path '(' args? ')' block? ;             // block = stage parameters
path        ::= ident ('.' ident)* ;
args        ::= arg (',' arg)* ;
arg         ::= (ident '=')? expr ;                     // positional or named
block       ::= '{' (ident '=' expr ';')* '}' ;
binop       ::= expr ('+' | '*') expr ;                 // corpus algebra
member      ::= expr '.' ident ('(' args? ')')? ;
cmp         ::= '<' | '>' | '<=' | '>=' | '==' ;
```

That is the entire language. A stage call with a block is the workhorse form:

```loom
model m = pretrain(arch, mix, tk) { tokens = 800M; optimizer = adamw(lr=6e-4); };
```

## Type rules (the checker must enforce these before execution)

```
decoder(layers:int, width:int, heads:int, ctx:int)          -> target
load(name:string)                                           -> target
data.text(spec:string)                                      -> corpus
data.chat(path:string)                                      -> corpus[chat]
data.prefs(path:string)                                     -> corpus[pref]
corpus + corpus                                             -> corpus
corpus * number                                             -> corpus        // weight
corpus.filter(pred) | .dedup(..) | .shuffle(..)             -> corpus
corpus.split(frac)                                          -> (corpus, corpus)
tokenizer.bpe(corpus, vocab:int)                            -> tokenizer
pretrain(target, corpus, tokenizer)                         -> model
finetune(model, corpus[chat])                               -> model
align(model, corpus[pref])                                  -> model
graft(model, unit)                                          -> model
merge(model, model)                                         -> model
perplexity(model, corpus|evalset)                           -> number
accuracy(model, evalset)                                    -> number
```

Corpus *kinds* (`chat`, `pref`) are part of the type: `align` requires preference data, and
handing it plain text is a compile error. This catches the single most common way an
alignment run wastes a day.

## The IR: a typed stage graph

Every program lowers to a DAG. Nodes are stages; edges carry typed artifacts. The IR is
what backends consume, and what `--emit ir` prints.

```json
{
  "nodes": [
    {"id": "n0", "op": "data.text",  "args": {"spec": "babylm:childes"}, "type": "corpus"},
    {"id": "n1", "op": "corpus.mix", "inputs": ["n0","n2"], "weights": [0.7,0.3]},
    {"id": "n3", "op": "tokenizer.bpe", "inputs": ["n1"], "args": {"vocab": 16000}},
    {"id": "n4", "op": "target.decoder", "args": {"layers":12,"width":768}},
    {"id": "n5", "op": "pretrain", "inputs": ["n4","n1","n3"],
                  "args": {"tokens": 8e8, "lr": 6e-4}, "type": "model"},
    {"id": "n6", "op": "assert", "inputs": ["n5"],
                  "args": {"metric":"perplexity","op":"<","threshold":60}}
  ],
  "targets": {"backend": "scratch", "device_plan": {"n5": "rtx5090"}}
}
```

Properties the IR must have:
- **Topologically ordered and side-effect free to inspect.** Printing it costs nothing.
- **Complete.** Everything needed to execute is in the graph; no hidden globals.
- **Backend-agnostic.** `pretrain` says what, not how. Lowering picks the how.
- **Costed.** Each node carries an estimated compute so the planner can place it.

## Backend interface

A backend implements exactly this. Nothing above it may assume a particular architecture.

```python
class Backend(Protocol):
    name: str
    def realize(self, target_node) -> ModelHandle: ...       # build or load
    def blocks(self, m: ModelHandle) -> list: ...            # uniform block access
    def residual_hook(self, m, layer: int, fn): ...          # read/write the stream
    def train_step(self, m, batch, opt_cfg) -> float: ...
    def evaluate(self, m, evalset, metric: str) -> float: ...
    def save(self, m, path: str) -> None: ...
```

`scratch` implements `realize` by constructing modules from a `decoder(...)` spec.
`hf` implements it by `AutoModelForCausalLM.from_pretrained(...)`. Both expose blocks and
the residual stream the same way, which is what lets `graft` (the ABI/linker) and monitors
work identically on a model you built and a model you downloaded.

## Compile-time errors worth having

| Program | Error |
|---|---|
| `align(m, data.text("x"))` | alignment needs preference data; `data.text` produces plain corpus |
| `pretrain(mix, arch, tk)` | argument 1 must be `target`, got `corpus` |
| `assert perplexity(m, held) < 0` | perplexity is ≥ 1; this assertion can never pass |
| a `model` never exported or asserted | dead result: nothing consumes it |
| `tokens = 800` | suspiciously small; did you mean `800M`? (warning, not error) |

## Execution and provenance

Executing the IR produces an artifact directory:

```
build/smalltalk-v1/
  model/                 weights, tokenizer, config
  ir.json                the graph that was executed
  report.json            every assertion, its measurement, pass/fail
  provenance.json        source hash, git sha, GPU, wall clock, tokens seen
```

A build that fails an assertion writes the report and exits nonzero. It does not write a
model directory, so a failing model cannot be picked up by accident.
