"""Two hand-compiled skills in one weight set, measured.

Every number here comes from running the composed model. The claims worth paying for:
succession is exact on its alphabet; induction keeps its gate inside the composed
weights; letter traffic cannot tell the composed model from the single-skill one; and
on the one kind of traffic where the two skills disagree, arbitration goes the way the
semantics say. A random model is run through the same gates because a gate a random
model passes proves nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from loom.constructed import compile_composed, compile_induction
from miabstraction.experiments.e2_induction import gapped_doubled_sequences
from miabstraction.models import TinyTransformer

VOCAB, N_CYCLE, COPY, GAP, MAXLEN = 20, 6, 22, 16, 60
CYCLE = list(range(VOCAB - N_CYCLE, VOCAB))


def succ(v: int) -> int:
    return CYCLE[(CYCLE.index(v) + 1) % N_CYCLE]


@torch.no_grad()
def succession_accuracy(model) -> float:
    seq = [CYCLE[i % N_CYCLE] for i in range(24)]
    logits = model(torch.tensor([seq]))
    hits = sum(int(logits[0, p].argmax().item() == succ(seq[p]))
               for p in range(len(seq) - 1))
    return hits / (len(seq) - 1)


@torch.no_grad()
def induction_accuracy(model, seed: int = 1) -> float:
    seqs, gaps = gapped_doubled_sequences(32, COPY, GAP, VOCAB - N_CYCLE,
                                          np.random.default_rng(seed))
    logits = model(torch.from_numpy(seqs)[:, :-1])
    hits = total = 0
    for b in range(seqs.shape[0]):
        g = int(gaps[b])
        for p in range(COPY + g + 2, 2 * COPY + g):
            hits += int(logits[b, p - 1].argmax().item() == seqs[b, p])
            total += 1
    return hits / total


@torch.no_grad()
def main() -> int:
    composed = compile_composed(vocab=VOCAB, n_cycle=N_CYCLE, max_len=MAXLEN)
    alone = compile_induction(vocab=VOCAB, max_len=MAXLEN)
    torch.manual_seed(0)
    rand = TinyTransformer(vocab=VOCAB, d_model=192, n_layers=3, n_heads=1,
                           max_len=MAXLEN, attn_only=True)

    # Letter traffic: the second skill must be invisible.
    seqs, _ = gapped_doubled_sequences(16, COPY, GAP, VOCAB - N_CYCLE,
                                       np.random.default_rng(0))
    tokens = torch.from_numpy(seqs)
    a, b = composed(tokens), alone(tokens)
    divergence = float((a - b).abs().max())
    argmax_identical = bool(torch.equal(a.argmax(-1), b.argmax(-1)))

    # The disagreement case: ...a b c ... a b -> induction says c, succession succ(b).
    va, vb, vc = CYCLE[3], CYCLE[0], CYCLE[4]
    seq = [va, vb, vc, 1, 2, 3, 4, 5, va, vb]
    pred = int(composed(torch.tensor([seq]))[0, len(seq) - 1].argmax())

    report = {
        "what": "trigram induction + succession compiled into one 3-layer weight set",
        "memory_map": {"induction": "TOK/POS/BUF1/BUF2 (dims 0..160)",
                       "succession": "CYCLE_BLOCK (dims 160..192)"},
        "nonzero_weights": int(sum((p != 0).sum() for p in composed.parameters())),
        "succession_accuracy": succession_accuracy(composed),
        "induction_acc_composed": induction_accuracy(composed),
        "induction_acc_alone": induction_accuracy(alone),
        "letter_traffic_max_logit_divergence": divergence,
        "letter_traffic_argmax_identical": argmax_identical,
        "arbitration_when_skills_disagree": {
            "sequence": seq, "induction_would_say": vc,
            "succession_demands": succ(vb), "model_predicted": pred,
            "succession_won": bool(pred == succ(vb))},
        "random_model_control": {
            "succession_accuracy": succession_accuracy(rand),
            "induction_accuracy": induction_accuracy(rand),
            "note": "a gate a random model also passes proves nothing"},
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/loom_composed_demo.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
