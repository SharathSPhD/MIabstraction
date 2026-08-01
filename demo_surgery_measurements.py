#!/usr/bin/env python3
"""MODEL SURGERY demo with REAL measurements on cached models.

This script runs each surgery operation on actual HuggingFace models,
records before/after metrics (perplexity, loss, memory, latency),
and writes results to results/loom_surgery_demo.json.

CRITICAL: No invented numbers. Every metric is measured, or recorded as null
with the reason if measurement fails.
"""

import json
import time
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BertForMaskedLM
from src.loom.stages.surgery import (
    merge, prune_layers, lora, apply_lora, distill, quantize,
    measure_perplexity, measure_memory, _slerp, _ties_merge
)


# Test corpus: short texts to keep measurements fast
EVAL_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a field of artificial intelligence.",
    "Language models transform text through neural networks.",
    "Deep learning trains models on large datasets.",
    "Python is a popular language for AI research.",
]


def measure_model_stats(model: nn.Module, texts: List[str], model_name: str,
                       device: str = "cuda") -> Dict[str, Any]:
    """Measure a model's perplexity, memory, and other stats."""
    stats = {
        "model": model_name,
        "perplexity": None,
        "memory_mb": None,
        "perplexity_error": None,
    }

    # Memory
    try:
        stats["memory_mb"] = measure_memory(model)
    except Exception as e:
        stats["memory_error"] = str(e)

    # Perplexity
    try:
        ppl = measure_perplexity(model, texts, device=device)
        if math.isfinite(ppl):
            stats["perplexity"] = round(float(ppl), 4)
        else:
            stats["perplexity_error"] = "non-finite result"
    except Exception as e:
        stats["perplexity_error"] = str(e)

    return stats


def demo_merge() -> Dict[str, Any]:
    """Demo: Merge two models of the same architecture.

    Test: Load Llama-3.2-1B and its Instruct variant, merge them.
    """
    print("\n" + "="*70)
    print("DEMO 1: MODEL MERGING")
    print("="*70)

    result = {
        "op": "merge",
        "timestamp": datetime.now().isoformat(),
        "models": ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"],
        "measurements": [],
        "errors": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("Loading models...")
        model_a = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map=device
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, device_map=device
        )

        # Measure originals
        print("Measuring model A (base)...")
        stats_a = measure_model_stats(model_a, EVAL_CORPUS, "Llama-3.2-1B", device=device)

        print("Measuring model B (instruct)...")
        stats_b = measure_model_stats(model_b, EVAL_CORPUS, "Llama-3.2-1B-Instruct", device=device)

        result["measurements"].append({"role": "base", **stats_a})
        result["measurements"].append({"role": "target", **stats_b})

        # Merge: linear, slerp, ties at different alpha values
        for method in ["linear", "slerp", "ties"]:
            for alpha in [0.3, 0.5, 0.7]:
                print(f"Merging with {method}, alpha={alpha}...")
                start = time.time()

                merged, meta = merge(model_a, model_b, method=method, alpha=alpha)

                elapsed = time.time() - start

                # Measure merged model
                stats_merged = measure_model_stats(merged, EVAL_CORPUS, f"Merged({method}, α={alpha})", device=device)
                stats_merged["method"] = method
                stats_merged["alpha"] = alpha
                stats_merged["wall_clock_seconds"] = round(elapsed, 2)

                result["measurements"].append(stats_merged)

                print(f"  Perplexity: {stats_merged.get('perplexity', 'error')}")

                del merged  # Free memory

    except Exception as e:
        result["errors"].append(f"Merge demo failed: {str(e)}")

    return result


def demo_pruning() -> Dict[str, Any]:
    """Demo: Prune layers from a model.

    Test: Load Gemma-2-2B, progressively prune layers, measure degradation.
    """
    print("\n" + "="*70)
    print("DEMO 2: LAYER PRUNING")
    print("="*70)

    result = {
        "op": "prune",
        "timestamp": datetime.now().isoformat(),
        "model": "google/gemma-2-2b",
        "measurements": [],
        "errors": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("Loading Gemma-2-2B...")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map=device
        )

        n_layers = len(model.model.layers)
        print(f"Model has {n_layers} layers")

        # Measure original
        print("Measuring original model...")
        stats_orig = measure_model_stats(model, EVAL_CORPUS, f"Gemma-2-2B (full, {n_layers} layers)", device=device)
        result["measurements"].append({"layers_kept": n_layers, **stats_orig})

        # Prune progressively
        for keep_pct in [0.75, 0.5, 0.25]:
            n_keep = max(1, int(n_layers * keep_pct))
            keep_indices = list(range(n_keep))

            print(f"Pruning to {n_keep}/{n_layers} layers ({keep_pct*100:.0f}%)...")
            start = time.time()

            pruned = prune_layers(model, keep_indices)

            elapsed = time.time() - start

            stats_pruned = measure_model_stats(pruned, EVAL_CORPUS, f"Gemma-2-2B (pruned to {n_keep} layers)", device=device)
            stats_pruned["layers_kept"] = n_keep
            stats_pruned["wall_clock_seconds"] = round(elapsed, 2)

            result["measurements"].append(stats_pruned)

            print(f"  Perplexity: {stats_pruned.get('perplexity', 'error')}")

            del pruned  # Free memory

    except Exception as e:
        result["errors"].append(f"Pruning demo failed: {str(e)}")

    return result


def demo_lora() -> Dict[str, Any]:
    """Demo: Add LoRA adapter to a model.

    Test: Load SmolLM2-1.7B, add LoRA, measure parameter count.
    """
    print("\n" + "="*70)
    print("DEMO 3: LoRA ADAPTERS")
    print("="*70)

    result = {
        "op": "lora",
        "timestamp": datetime.now().isoformat(),
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "measurements": [],
        "errors": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("Loading SmolLM2-1.7B-Instruct...")
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.bfloat16, device_map=device
        )

        # Measure original
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"Original model parameters: {orig_params:,}")

        stats_orig = {
            "role": "base",
            "parameters": orig_params,
            "trainable_parameters": orig_params,
            "perplexity": None,
        }

        result["measurements"].append(stats_orig)

        # Add LoRA adapters
        for rank in [8, 16, 32]:
            print(f"Adding LoRA (rank={rank}) to q_proj, v_proj...")
            start = time.time()

            adapter = lora(model, rank=rank, targets=["q_proj", "v_proj"])

            elapsed = time.time() - start

            # Count LoRA parameters
            lora_params = sum(A.numel() + B.numel() for A, B in adapter.lora_weights.values())
            trainable_params = lora_params  # Only LoRA params are trainable

            stats_lora = {
                "rank": rank,
                "parameters_total": orig_params + lora_params,
                "parameters_lora": lora_params,
                "trainable_parameters": trainable_params,
                "trainable_pct": 100.0 * trainable_params / (orig_params + lora_params),
                "wall_clock_seconds": round(elapsed, 2),
            }

            result["measurements"].append(stats_lora)

            print(f"  LoRA params: {lora_params:,} ({100*lora_params/(orig_params+lora_params):.2f}%)")

            del adapter  # Free memory

    except Exception as e:
        result["errors"].append(f"LoRA demo failed: {str(e)}")

    return result


def demo_distillation() -> Dict[str, Any]:
    """Demo: Distill knowledge from teacher to student.

    Test: Use Gemma-2-2B as teacher, smaller GPT-2 as student.
    """
    print("\n" + "="*70)
    print("DEMO 4: KNOWLEDGE DISTILLATION")
    print("="*70)

    result = {
        "op": "distill",
        "timestamp": datetime.now().isoformat(),
        "teacher": "google/gemma-2-2b",
        "student": "gpt2",
        "measurements": [],
        "errors": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("Loading teacher (Gemma-2-2B)...")
        teacher = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map=device
        )

        print("Loading student (GPT-2)...")
        student = AutoModelForCausalLM.from_pretrained(
            "gpt2", torch_dtype=torch.bfloat16, device_map=device
        )

        # Measure student before distillation
        print("Measuring student before distillation...")
        stats_before = measure_model_stats(student, EVAL_CORPUS, "GPT-2 (before)", device=device)
        stats_before["stage"] = "before_distillation"
        result["measurements"].append(stats_before)

        # Distill
        print("Running distillation (1 epoch)...")
        start = time.time()

        student_trained, distill_meta = distill(
            student, teacher, corpus=EVAL_CORPUS * 2, epochs=1,
            temperature=4.0, alpha=0.5, batch_size=2, device=device
        )

        elapsed = time.time() - start

        # Measure after distillation
        print("Measuring student after distillation...")
        stats_after = measure_model_stats(student_trained, EVAL_CORPUS, "GPT-2 (after)", device=device)
        stats_after["stage"] = "after_distillation"
        stats_after["wall_clock_seconds"] = round(elapsed, 2)
        stats_after["distillation_loss"] = distill_meta.get("distillation_loss", [])

        result["measurements"].append(stats_after)

        print(f"  Perplexity before: {stats_before.get('perplexity', 'error')}")
        print(f"  Perplexity after: {stats_after.get('perplexity', 'error')}")

        del teacher, student_trained  # Free memory

    except Exception as e:
        result["errors"].append(f"Distillation demo failed: {str(e)}")

    return result


def demo_quantization() -> Dict[str, Any]:
    """Demo: Quantize a model to int8.

    Test: Load gpt2, quantize to int8, measure perplexity and memory.
    """
    print("\n" + "="*70)
    print("DEMO 5: QUANTIZATION (INT8)")
    print("="*70)

    result = {
        "op": "quantize",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2",
        "measurements": [],
        "errors": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("Loading GPT-2...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", torch_dtype=torch.bfloat16, device_map=device
        )

        # Measure original
        print("Measuring original (bfloat16)...")
        stats_orig = measure_model_stats(model, EVAL_CORPUS, "GPT-2 (bfloat16)", device=device)
        stats_orig["dtype"] = "bfloat16"
        result["measurements"].append(stats_orig)

        # Quantize to int8
        print("Quantizing to int8...")
        start = time.time()

        quantized, quant_meta = quantize(model, bits=8)

        elapsed = time.time() - start

        # Measure quantized
        print("Measuring quantized (int8)...")
        stats_quant = measure_model_stats(quantized, EVAL_CORPUS, "GPT-2 (int8)", device=device)
        stats_quant["dtype"] = "int8"
        stats_quant["wall_clock_seconds"] = round(elapsed, 2)
        stats_quant["quantized_params"] = quant_meta.get("quantized_params", 0)
        stats_quant["total_params"] = quant_meta.get("total_params", 0)

        result["measurements"].append(stats_quant)

        print(f"  Perplexity (bfloat16): {stats_orig.get('perplexity', 'error')}")
        print(f"  Perplexity (int8): {stats_quant.get('perplexity', 'error')}")
        print(f"  Memory reduction: {stats_orig.get('memory_mb', 0):.1f} MB -> {stats_quant.get('memory_mb', 0):.1f} MB")

        del quantized  # Free memory

    except Exception as e:
        result["errors"].append(f"Quantization demo failed: {str(e)}")

    return result


def main():
    """Run all surgery demos and write results."""
    print("="*70)
    print("LOOM MODEL SURGERY — REAL MEASUREMENTS ON CACHED MODELS")
    print("="*70)

    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "demos": [],
    }

    # Run each demo
    results["demos"].append(demo_merge())
    torch.cuda.empty_cache()

    results["demos"].append(demo_pruning())
    torch.cuda.empty_cache()

    results["demos"].append(demo_lora())
    torch.cuda.empty_cache()

    results["demos"].append(demo_distillation())
    torch.cuda.empty_cache()

    results["demos"].append(demo_quantization())
    torch.cuda.empty_cache()

    # Write results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "loom_surgery_demo.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"Results written to {output_file}")
    print("="*70)

    # Print summary
    print("\nSUMMARY:")
    for demo in results["demos"]:
        op = demo.get("op")
        errors = demo.get("errors", [])
        n_measurements = len(demo.get("measurements", []))
        print(f"  {op:15s}: {n_measurements} measurements, {len(errors)} errors")


if __name__ == "__main__":
    main()
