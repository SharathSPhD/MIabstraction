"""Build an application on an open-weight substrate and test its expectations."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .capability import App, Kind
from .lowering import Choice, plan
from .parse import parse_program
from .substrate import profile_for
from .exec_open import continued_pretraining, finetune_refusals
from ..backends import HFBackend


def build_and_test(program_path: str, target: str, out_dir: str, device: str = "cuda") -> dict:
    """Build an application, train with knowledge and behavior, and test expectations.

    Args:
        program_path: Path to .loom file
        target: HuggingFace model ID (should be Instruct variant for best results)
        out_dir: Output directory for results
        device: CUDA device to use

    Returns:
        Dict with build info, training results, and test pass/fail status.
    """
    t0 = time.time()
    prog = parse_program(program_path)
    app: App = next(iter(prog.apps.values()))
    spec = {"kind": "load", "name": target}
    sub = profile_for(spec)
    choices: list[Choice] = plan(app.capabilities, sub)

    # Load model
    tok = AutoTokenizer.from_pretrained(target)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16)
    model.to(device).eval()

    # Load corpus if available
    corpus_dir = Path(program_path).parent / "corpus"
    corpus = []
    if corpus_dir.exists():
        for txt_file in sorted(corpus_dir.glob("*.txt")):
            corpus.extend(txt_file.read_text().split("\n"))
        corpus = [line.strip() for line in corpus if line.strip()][:100]  # Limit to 100 lines

    training_results = {}

    # Train knowledge capability
    knowledge_caps = [c for c in choices if c.capability.kind == Kind.KNOWLEDGE]
    if knowledge_caps and corpus:
        ch = knowledge_caps[0]
        if ch.strategy.name == "continued_pretraining":
            print(f"Training: {ch.capability.name}")
            result = continued_pretraining(
                type('Handle', (), {'module': model, 'name': target, 'tokenizer': tok})(),
                corpus, steps=50, rank=8, alpha=16.0
            )
            training_results["knowledge"] = result
            if "error" in result:
                print(f"Knowledge training error: {result['error']}")

    # Train refusal/behavior capabilities
    guardrail_caps = [c for c in choices if c.capability.kind == Kind.GUARDRAIL]
    for ch in guardrail_caps:
        if ch.strategy.name == "finetune_refusals":
            print(f"Training: {ch.capability.name}")
            result = finetune_refusals(
                type('Handle', (), {'module': model, 'name': target, 'tokenizer': tok})(),
                refuse_topic=ch.capability.name, steps=20
            )
            training_results[ch.capability.name] = result
            if "error" in result:
                print(f"Refusal training error: {result['error']}")

    # Test expectations
    model.eval()
    test_results = {}
    expectations_passed = 0

    for exp in app.expectations:
        if exp.kind == "refuses":
            # Test that model refuses
            prompt = f"Q: {exp.probe} A:"
            input_ids = tok(prompt, return_tensors="pt").to(device).input_ids
            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9
                )
            generated_text = tok.decode(output[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):]

            # Check if it's a refusal (contains words like "cannot", "not able", etc.)
            refusal_words = ["cannot", "not able", "cannot help", "cannot provide", "unable", "no information"]
            is_refusal = any(word in answer.lower() for word in refusal_words)

            test_results[f"refuse: {exp.probe}"] = {
                "expected": "refusal",
                "generated": answer.strip()[:100],
                "passed": is_refusal,
            }
            if is_refusal:
                expectations_passed += 1

        else:  # kind == "answers"
            # Test that model answers and mentions the key phrase
            prompt = f"Q: {exp.probe} A:"
            input_ids = tok(prompt, return_tensors="pt").to(device).input_ids
            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9
                )
            generated_text = tok.decode(output[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):]

            # Check if contains the required word
            contains_word = exp.contains.lower() in answer.lower()

            test_results[f"answers: {exp.probe}"] = {
                "expected": f"mentions '{exp.contains}'",
                "generated": answer.strip()[:200],
                "passed": contains_word,
            }
            if contains_word:
                expectations_passed += 1

    # Write results
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report = {
        "app": app.name,
        "base_model": target,
        "substrate": sub.id,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_params": sum(p.numel() for p in model.parameters()),
        "corpus_size": len(corpus),
        "training": training_results,
        "expectations": {
            "total": len(app.expectations),
            "passed": expectations_passed,
            "results": test_results,
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }

    (out_path / "build_results.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    prog = sys.argv[1] if len(sys.argv) > 1 else "examples/tutor.loom"
    tgt = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-1B-Instruct"
    out = sys.argv[3] if len(sys.argv) > 3 else "build/Tutor-lora"
    r = build_and_test(prog, tgt, out)
    print(json.dumps(r, indent=2)[:3000])
    print(f"\nExpectations: {r['expectations']['passed']}/{r['expectations']['total']} passed")
