"""Verify bracket matching task is well-posed: explicit stack solver."""
import torch
import numpy as np
from miabstraction.data.algo import BracketMatchingDataset


def explicit_bracket_solver(tokens, answer_pos):
    """Solve bracket matching with explicit stack algorithm.

    Openers: 0, 1
    Closers: 2, 3 (2 closes 0, 3 closes 1)
    Distractors: 4-9

    Returns: expected closing token (2 or 3) at answer_pos
    """
    stack = []

    # Process tokens up to answer position
    for t in range(answer_pos):
        tok = tokens[t].item()
        if tok == 0:  # opener type 0
            stack.append(0)
        elif tok == 1:  # opener type 1
            stack.append(1)
        elif tok == 2:  # closer type 0
            if stack and stack[-1] == 0:
                stack.pop()
        elif tok == 3:  # closer type 1
            if stack and stack[-1] == 1:
                stack.pop()
        # Else: distractor, ignore

    # At answer position, what should we predict?
    if not stack:
        return None  # No unmatched opener - task is ill-defined

    expected_opener = stack[-1]
    return expected_opener + 2  # Convert to closer token ID


def verify_dataset():
    """Verify dataset by comparing generated labels to explicit solver."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=100, seed=0)

    mismatches = 0
    ill_defined = 0

    for i in range(len(ds)):
        seq, mask, generated_label = ds.get_with_mask(i)
        ans_pos = mask.nonzero(as_tuple=True)[0].item()

        # Solve with explicit algorithm
        expected_label = explicit_bracket_solver(seq, ans_pos)

        if expected_label is None:
            ill_defined += 1
            continue

        if expected_label != generated_label:
            mismatches += 1
            print(f"Sample {i}: MISMATCH")
            print(f"  seq: {seq.tolist()}")
            print(f"  ans_pos: {ans_pos}")
            print(f"  generated label: {generated_label}")
            print(f"  expected label: {expected_label}")

    print(f"\n=== VERIFICATION RESULTS ===")
    print(f"Total samples: {len(ds)}")
    print(f"Ill-defined (no unmatched opener): {ill_defined}")
    print(f"Mismatches (label != solver): {mismatches}")
    print(f"Match rate: {(len(ds) - mismatches - ill_defined) / len(ds) * 100:.1f}%")

    if mismatches == 0 and ill_defined == 0:
        print("\n✓ Task is well-posed! Dataset labels are correct.")
        return True
    else:
        print("\n✗ Task has issues!")
        return False


if __name__ == "__main__":
    verify_dataset()
