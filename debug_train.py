#!/usr/bin/env python3
"""Debug training to see what's happening."""
import numpy as np
import torch

from loom.curriculum import InductionCompiler, allocate_vocabulary, compile_curriculum
from loom.spec import Skill, WeaveSpec
from miabstraction.models import TinyTransformer
from miabstraction.seeding import set_determinism


def main():
    seed = 42
    set_determinism(seed, strict=True)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a simple induction compiler
    compiler = InductionCompiler(copy_len=8, max_gap=16, vocab_offset=0)

    # Generate dataset
    seqs, gaps = compiler.generator(n_seq=128, rng=rng)
    tokens = torch.from_numpy(seqs).to(device)

    print(f"Token shape: {tokens.shape}")
    print(f"Gaps: {gaps[:5]}")
    print(f"Vocab: 256, Sequence length: {tokens.shape[1]}")

    # Create a small model - use attention-only for induction
    model = TinyTransformer(vocab=256, d_model=64, n_layers=4, n_heads=4, max_len=128, attn_only=True).to(device)

    # Train with debugging
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    print("\nTraining...")
    for step in range(250):
        batch_idx = rng.integers(0, len(tokens), size=32)
        batch = tokens[batch_idx]

        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 256),
            batch[:, 1:].reshape(-1),
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == 99:
            model.eval()
            metrics = compiler.evaluator(model, tokens[:64], gaps[:64], device)
            print(f"Step {step}: loss={loss.item():.4f}, prefix_score={metrics['prefix_score']:.4f}, icl_loss={metrics['icl_loss']:.4f}")
            model.train()


if __name__ == "__main__":
    main()
