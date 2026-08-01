"""Two hand-compiled skills in one weight set, and the proof they do not interfere.

Composition is the point of an abstraction layer: a compiler that can only ship one
skill per model has a library of one book. The composed model carries the trigram
induction circuit exactly as compiled, plus a succession skill (a deterministic
next-symbol rule over a designated cycle alphabet) living entirely in the residual
block induction never touches.

The non-interference claim: on traffic containing no cycle tokens the composed model
is the single-skill model. In exact arithmetic the logits are equal — the succession
skill reads coordinates that are zero on that traffic, and the only thing LayerNorm
adds to them is a scalar shift, which a zero-mean read code annihilates. In floats
the annihilation happens in summation order, leaving ~1e-14 of residue, so the test
asserts equality to 1e-9 and identical argmax everywhere rather than bit equality.
"""
import numpy as np
import pytest
import torch

from loom.constructed import CYCLE_BLOCK, compile_composed, compile_induction
from miabstraction.experiments.e2_induction import gapped_doubled_sequences
from miabstraction.models import TinyTransformer

VOCAB, N_CYCLE, COPY, GAP = 20, 6, 22, 16
MAXLEN = 60
CYCLE = list(range(VOCAB - N_CYCLE, VOCAB))       # the last n tokens form the cycle


@pytest.fixture(scope="module")
def composed():
    return compile_composed(vocab=VOCAB, n_cycle=N_CYCLE, max_len=MAXLEN)


@pytest.fixture(scope="module")
def induction_only():
    return compile_induction(vocab=VOCAB, max_len=MAXLEN)


def _succ(v: int) -> int:
    i = CYCLE.index(v)
    return CYCLE[(i + 1) % N_CYCLE]


def test_letter_traffic_is_unchanged(composed, induction_only):
    """On sequences with no cycle tokens the second skill must be invisible. The
    residue of LayerNorm's scalar shift against a zero-mean read code is float
    summation-order noise (~1e-14); anything above 1e-9 would be a real read."""
    rng = np.random.default_rng(0)
    seqs, _ = gapped_doubled_sequences(16, COPY, GAP, VOCAB - N_CYCLE, rng)
    tokens = torch.from_numpy(seqs)
    with torch.no_grad():
        a = composed(tokens)
        b = induction_only(tokens)
    assert float((a - b).abs().max()) < 1e-9, (
        "the composed model's logits differ on traffic the second skill "
        "should never see")
    assert torch.equal(a.argmax(-1), b.argmax(-1)), (
        "the second skill changed a prediction on traffic it should never see")


def test_succession_is_exact_on_cycle_traffic(composed):
    """Walking the cycle: every next-token prediction is the successor, always."""
    seq = [CYCLE[i % N_CYCLE] for i in range(24)]
    tokens = torch.tensor([seq])
    with torch.no_grad():
        logits = composed(tokens)
    for p in range(len(seq) - 1):
        pred = logits[0, p].argmax().item()
        assert pred == _succ(seq[p]), (
            f"at position {p}, token {seq[p]}: predicted {pred}, "
            f"successor is {_succ(seq[p])}")


def test_succession_wins_when_induction_disagrees(composed):
    """The adversarial case: a repeated digram of cycle tokens gives the induction
    head a confident (and wrong, under the composed semantics) prediction. Priority
    on cycle tokens belongs to succession, and this is where that is proven."""
    a, b, c = CYCLE[3], CYCLE[0], CYCLE[4]
    assert _succ(b) != c, "pick tokens so the two skills actually disagree"
    # ... a b c ... a b ?   induction says c; succession says succ(b)
    seq = [a, b, c, 1, 2, 3, 4, 5, a, b]
    tokens = torch.tensor([seq])
    with torch.no_grad():
        logits = composed(tokens)
    pred = logits[0, len(seq) - 1].argmax().item()
    assert pred == _succ(b), (
        f"induction won the arbitration: predicted {pred}, "
        f"succession demands {_succ(b)}")


def test_induction_still_works_in_the_composed_model(composed):
    """The first skill keeps its gate inside the composed weights, measured on its
    own traffic (letters only, where the semantics are unambiguous)."""
    rng = np.random.default_rng(1)
    seqs, gaps = gapped_doubled_sequences(32, COPY, GAP, VOCAB - N_CYCLE, rng)
    tokens = torch.from_numpy(seqs)
    with torch.no_grad():
        logits = composed(tokens[:, :-1])
    hits = total = 0
    for bidx in range(seqs.shape[0]):
        g = int(gaps[bidx])
        # predictable second-copy targets x[2..]: positions COPY+g+2 .. 2*COPY+g-1
        for p in range(COPY + g + 2, 2 * COPY + g):
            hits += int(logits[bidx, p - 1].argmax().item() == seqs[bidx, p])
            total += 1
    assert hits / total > 0.9, f"icl argmax accuracy {hits / total}"


def test_untrained_model_passes_neither_gate():
    """The Heap et al. control: any gate a random model also passes proves nothing."""
    torch.manual_seed(0)
    rand = TinyTransformer(vocab=VOCAB, d_model=192, n_layers=3, n_heads=1,
                           max_len=MAXLEN, attn_only=True)
    seq = [CYCLE[i % N_CYCLE] for i in range(24)]
    with torch.no_grad():
        logits = rand(torch.tensor([seq]))
    hits = sum(int(logits[0, p].argmax().item() == _succ(seq[p]))
               for p in range(len(seq) - 1))
    assert hits / (len(seq) - 1) < 0.5, "a random model walks the cycle; gate is vacuous"


def test_cycle_block_is_the_reserved_space():
    """The second skill lives in the block the memory map called unused; if this
    moves, the bit-identity claim needs re-deriving, so pin it."""
    assert CYCLE_BLOCK == slice(160, 192)
