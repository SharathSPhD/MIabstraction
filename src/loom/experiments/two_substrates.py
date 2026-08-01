"""One program, two substrates.

`examples/hardened.loom` is compiled and executed twice with nothing changed but the
target — the way a C source file is built for two instruction sets:

    A   nemotron_h(...)                 a hybrid state-space/attention stack, built here
    B   load("meta-llama/Llama-3.2-1B") 1.24B parameters, downloaded, frozen

The program's operations are written against features and circuits, so neither run
knows what it is running on. Where an operation cannot be realized on a substrate, the
executor says so and the gate fails — which is the useful behaviour, not a shortfall.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..execute import run_program
from ..lang.frontend import compile_source

SRC = "examples/hardened.loom"
OUT = Path("results/loom_two_substrates.json")

# Contrastive text for the open-weight substrate: two registers of real English.
FORMAL = ["Pursuant to the aforementioned agreement, the parties shall",
          "It is hereby resolved that the committee will convene",
          "The undersigned acknowledges receipt of the documents and",
          "In accordance with regulation 14, the applicant must submit"]
CASUAL = ["honestly that was such a weird movie and i",
          "yeah so we just hung out and ate way too much",
          "lol i totally forgot to text you back about",
          "it was kinda fun but also super tiring because we"]
NEUTRAL = ["The weather today is quite pleasant and the sky is",
           "Water boils at one hundred degrees Celsius at sea",
           "She opened the book and began to read the first",
           "The train arrived at the station exactly on time"]


def _tok_batches(texts, tok, dev):
    return [tok(t, return_tensors="pt")["input_ids"] for t in texts]


def _synthetic_registers(vocab: int, n: int, L: int, seed: int):
    """Two distinguishable token 'registers' for the from-scratch substrate: two
    different first-order processes over the same vocabulary. Real structure, and
    the same shape of task the open-weight run faces — tell A from B, then steer."""
    rng = np.random.default_rng(seed)
    P_a = rng.dirichlet(np.ones(vocab) * 0.3, size=vocab)
    P_b = rng.dirichlet(np.ones(vocab) * 0.3, size=vocab)
    P_n = rng.dirichlet(np.ones(vocab) * 0.3, size=vocab)

    def draw(P, k):
        out = np.empty((k, L), dtype=np.int64)
        out[:, 0] = rng.integers(0, vocab, size=k)
        for t in range(1, L):
            cdf = np.cumsum(P[out[:, t - 1]], axis=1)
            out[:, t] = (rng.random(k)[:, None] < cdf).argmax(axis=1)
        return [torch.from_numpy(out[i:i + 1]) for i in range(k)]

    return draw(P_a, n), draw(P_b, n), draw(P_n, n)


def _train_set(vocab: int, L: int, seed: int, n: int = 256):
    """Real structure for the substrate to learn: a mixture of both registers, so the
    model has a reason to represent the distinction the program asks it to read."""
    a, b, _ = _synthetic_registers(vocab, n // 2, L, seed)
    import torch as _t
    pool = a + b
    return [_t.cat(pool[i:i + 16]) for i in range(0, len(pool) - 16, 16)]


def run(dev: str = "cuda") -> dict:
    t0 = time.time()
    graph = compile_source(SRC)          # the SAME source for both runs
    n_nodes = len(graph.nodes) if hasattr(graph, "nodes") else None

    reports = {}

    # ---- substrate A: built here, hybrid architecture, and TRAINED.
    # Vocabulary is 31 so the substrate sits inside the induction circuit's verified
    # envelope; an untrained model would fail the probe gate at chance, which is the
    # gate working, not a bug.
    vocab, L = 31, 48
    a, b, neutral = _synthetic_registers(vocab, 12, L, seed=0)
    spec_a = {"kind": "nemotron_h", "name": "nemotron_h(w=128,l=8,attn_every=4)",
              "width": 128, "layers": 8, "heads": 4, "attention_every": 4,
              "vocab": vocab, "ctx": 64}
    reports["A_from_scratch"] = run_program(
        spec_a, {"a": a, "b": b, "neutral": neutral, "seq_len": L},
        dev=dev, layer=-2, train_batches=_train_set(vocab, L, seed=0)).to_dict()

    # ---- substrate B: downloaded, frozen
    from transformers import AutoTokenizer
    name = "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(name)
    spec_b = {"kind": "load", "name": name, "dtype": "bfloat16"}
    reports["B_open_weight"] = run_program(
        spec_b,
        {"a": _tok_batches(FORMAL, tok, dev), "b": _tok_batches(CASUAL, tok, dev),
         "neutral": _tok_batches(NEUTRAL, tok, dev), "seq_len": 16},
        dev=dev, layer=-6).to_dict()

    result = {
        "source": SRC,
        "claim": "the same program, compiled to two different transformer substrates",
        "ir_nodes": n_nodes,
        "substrates": reports,
        "identical_source": True,
        "wall_clock_s": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    for k, v in r["substrates"].items():
        print(f"\n=== {k}: {v['target']} ({v['backend']}) ===")
        print(f"    {v['substrate']}")
        for g in v["gates"]:
            mark = "PASS" if g["passed"] else "FAIL"
            print(f"    [{mark}] {g['gate']}: {g['measured']}")
            if g.get("note"):
                print(f"           {g['note']}")
        print(f"    build: {'PASSED' if v['passed'] else 'REFUSED'}")
