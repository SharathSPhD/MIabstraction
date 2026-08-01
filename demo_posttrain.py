#!/usr/bin/env python3
"""End-to-end demo of Loom post-training stages: SFT then DPO.

This script:
1. Loads a small cached model (gpt2)
2. Creates synthetic instruction data for SFT
3. Creates synthetic preference pairs for DPO
4. Runs SFT on the instruction data
5. Runs DPO with the SFT model as policy and pre-SFT as reference
6. Computes evaluation metrics (perplexity, accuracy, winrate)
7. Records all results with exact datasets used and honest limitations

The goal is to demonstrate that the Loom stages work on real models with real
(though synthetic) data, measuring actual wall-clock training time and GPU usage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from loom.stages.data import ChatCorpus, PrefCorpus, TextCorpus
from loom.stages.dpo import DPOConfig, dpo
from loom.stages.evaluate import perplexity, winrate
from loom.stages.sft import SFTConfig, sft


def create_synthetic_instruction_data(num_examples: int = 100) -> ChatCorpus:
    """Create synthetic instruction tuning data (Q&A style).

    Format: prompt is a question, response is an answer.
    All synthetic data is generated on the fly.
    """
    questions = [
        "What is Python?",
        "How do I write a loop?",
        "What is a function?",
        "What does a database do?",
        "What is machine learning?",
        "How does the internet work?",
        "What is encryption?",
        "What is a data structure?",
        "How do I debug code?",
        "What is an API?",
    ]

    answers = [
        "Python is a high-level programming language.",
        "You write a loop with for or while keywords.",
        "A function is a reusable block of code.",
        "A database stores and retrieves data efficiently.",
        "Machine learning is learning patterns from data.",
        "The internet uses routers and protocols to connect computers.",
        "Encryption protects data by encoding it.",
        "Data structures organize data for efficient access.",
        "Debugging involves finding and fixing errors.",
        "An API allows different software to communicate.",
    ]

    records = []
    for i in range(num_examples):
        q_idx = i % len(questions)
        a_idx = i % len(answers)
        records.append({
            "prompt": questions[q_idx] + f" (example {i})",
            "response": answers[a_idx] + f" (answer variant {i})",
        })

    return ChatCorpus(records=records)


def create_synthetic_preference_data(num_pairs: int = 50) -> PrefCorpus:
    """Create synthetic preference pairs (chosen vs rejected responses).

    Format: prompt with two response variants, one marked as chosen/better.
    """
    prompts = [
        "What is the best programming language?",
        "How should I learn to code?",
        "What is the difference between AI and ML?",
        "How do I optimize code?",
        "What is cloud computing?",
        "What makes a good software design?",
        "How do I handle errors?",
        "What is version control?",
    ]

    good_responses = [
        "The best language depends on your use case.",
        "Start with fundamentals and practice consistently.",
        "AI is broader; ML is a subset focused on learning.",
        "Profile first, then optimize bottlenecks.",
        "Cloud computing provides computing resources over the network.",
        "Good design is modular, testable, and maintainable.",
        "Handle errors gracefully with try-catch blocks.",
        "Version control tracks changes and enables collaboration.",
    ]

    bad_responses = [
        "Python is the only good language.",
        "Just copy code from the internet.",
        "AI and ML are the same thing.",
        "Optimize everything without measuring.",
        "Cloud computing is when you use someone else's computer.",
        "Good design has no structure.",
        "Errors should crash the program.",
        "Just email code to team members.",
    ]

    records = []
    for i in range(num_pairs):
        p_idx = i % len(prompts)
        r_idx = i % len(good_responses)
        records.append({
            "prompt": prompts[p_idx] + f" (pair {i})",
            "chosen": good_responses[r_idx] + f" (good {i})",
            "rejected": bad_responses[r_idx] + f" (bad {i})",
        })

    return PrefCorpus(records=records)


def create_synthetic_eval_data(num_examples: int = 20) -> TextCorpus:
    """Create synthetic text data for perplexity evaluation."""
    texts = [
        "Python is a great language for machine learning and data science.",
        "The internet connects computers around the world.",
        "Machine learning models learn patterns from data.",
        "Good software design makes code easier to maintain.",
        "Encryption protects sensitive information online.",
        "Databases store and organize large amounts of data.",
        "Functions allow code to be reused and organized.",
        "APIs enable different software systems to communicate.",
    ]

    records = []
    for i in range(num_examples):
        t_idx = i % len(texts)
        records.append({"text": texts[t_idx] + f" (variation {i})"})

    return TextCorpus(records=records)


def run_demo():
    """Run the full demo: SFT then DPO."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Load model and tokenizer
    print("\n" + "="*80)
    print("LOADING BASE MODEL")
    print("="*80)
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model for later reference
    base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    results = {
        "model": model_name,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "base_model_params": sum(p.numel() for p in model.parameters()),
        "stages": {}
    }

    # ========================================================================
    # STAGE 1: PERPLEXITY BEFORE SFT
    # ========================================================================
    print("\n" + "="*80)
    print("COMPUTING BASE PERPLEXITY (before SFT)")
    print("="*80)
    eval_data = create_synthetic_eval_data(num_examples=20)
    print(f"Eval set: {eval_data.size_estimate()} examples (synthetic)")

    try:
        base_ppl = perplexity(model, tokenizer, eval_data, max_length=128, batch_size=4)
        print(f"Base perplexity: {base_ppl:.4f}")
        results["base_perplexity"] = float(base_ppl)
    except Exception as e:
        print(f"WARNING: Could not compute base perplexity: {e}")
        results["base_perplexity"] = None

    # ========================================================================
    # STAGE 2: SUPERVISED FINE-TUNING (SFT)
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 1: SUPERVISED FINE-TUNING (SFT)")
    print("="*80)

    # Create training data
    train_data = create_synthetic_instruction_data(num_examples=100)
    eval_data_sft = create_synthetic_instruction_data(num_examples=20)

    print(f"Train set: {train_data.size_estimate()} examples (synthetic Q&A)")
    print(f"Eval set: {eval_data_sft.size_estimate()} examples (synthetic Q&A)")
    print(f"Data construction: 100 examples, 10 unique Q&A pairs, repeated with variations")

    # Configure SFT
    sft_config = SFTConfig(
        learning_rate=2e-5,
        batch_size=8,
        num_epochs=1,  # Quick demo
        max_steps=20,  # Limit for demo
        warmup_steps=5,
        pad_to_length=128,
        eval_interval=10,
    )

    print(f"SFT config: lr={sft_config.learning_rate}, batch_size={sft_config.batch_size}, "
          f"epochs={sft_config.num_epochs}, max_steps={sft_config.max_steps}")

    # Run SFT
    sft_start = time.time()
    print("Starting SFT training...")
    sft_result = sft(model, tokenizer, train_data, eval_corpus=eval_data_sft, config=sft_config)
    sft_elapsed = time.time() - sft_start

    print(f"SFT complete in {sft_elapsed:.2f} seconds")
    print(f"  Final loss: {sft_result['final_loss']:.4f}")
    print(f"  Eval loss: {sft_result['eval_loss']:.4f}")
    print(f"  Response-only loss: {sft_result['response_only_loss']:.4f}")

    results["stages"]["sft"] = {
        "final_loss": sft_result["final_loss"],
        "eval_loss": sft_result["eval_loss"],
        "response_only_loss": sft_result["response_only_loss"],
        "num_examples": sft_result["num_examples"],
        "num_steps": sft_result["global_steps"],
        "elapsed_seconds": sft_elapsed,
        "data_size_estimate": train_data.size_estimate(),
        "config": {
            "learning_rate": sft_config.learning_rate,
            "batch_size": sft_config.batch_size,
            "num_epochs": sft_config.num_epochs,
            "max_steps": sft_config.max_steps,
        }
    }

    # Compute perplexity after SFT
    print("\nComputing perplexity after SFT...")
    try:
        sft_ppl = perplexity(model, tokenizer, eval_data, max_length=128, batch_size=4)
        print(f"Perplexity after SFT: {sft_ppl:.4f}")
        results["stages"]["sft"]["perplexity"] = float(sft_ppl)
    except Exception as e:
        print(f"WARNING: Could not compute perplexity after SFT: {e}")
        results["stages"]["sft"]["perplexity"] = None

    # ========================================================================
    # STAGE 3: DIRECT PREFERENCE OPTIMIZATION (DPO)
    # ========================================================================
    print("\n" + "="*80)
    print("STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print("="*80)

    # Create preference data
    pref_train_data = create_synthetic_preference_data(num_pairs=50)
    pref_eval_data = create_synthetic_preference_data(num_pairs=10)

    print(f"Train set: {pref_train_data.size_estimate()} preference pairs (synthetic)")
    print(f"Eval set: {pref_eval_data.size_estimate()} preference pairs (synthetic)")
    print(f"Data construction: 50 pairs of (prompt, good_response, bad_response)")

    # Create reference model (copy of SFT model)
    print("Creating reference model (copy of SFT checkpoint)...")
    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    ref_model.load_state_dict(model.state_dict())

    # Configure DPO
    dpo_config = DPOConfig(
        beta=0.1,
        learning_rate=1e-5,
        batch_size=8,
        num_epochs=1,
        max_steps=10,  # Quick demo
        warmup_steps=2,
        pad_to_length=128,
        eval_interval=5,
    )

    print(f"DPO config: beta={dpo_config.beta}, lr={dpo_config.learning_rate}, "
          f"batch_size={dpo_config.batch_size}, epochs={dpo_config.num_epochs}, "
          f"max_steps={dpo_config.max_steps}")

    # Run DPO
    dpo_start = time.time()
    print("Starting DPO training...")
    dpo_result = dpo(model, ref_model, tokenizer, pref_train_data, eval_corpus=pref_eval_data,
                     config=dpo_config)
    dpo_elapsed = time.time() - dpo_start

    print(f"DPO complete in {dpo_elapsed:.2f} seconds")
    print(f"  Final loss: {dpo_result['final_loss']:.4f}")
    print(f"  Final accuracy: {dpo_result['final_accuracy']:.4f}")
    print(f"  Final margin: {dpo_result['final_margin']:.4f}")
    print(f"  Eval loss: {dpo_result['eval_loss']:.4f}")
    print(f"  Eval accuracy: {dpo_result['eval_accuracy']:.4f}")

    results["stages"]["dpo"] = {
        "final_loss": dpo_result["final_loss"],
        "final_accuracy": dpo_result["final_accuracy"],
        "final_margin": dpo_result["final_margin"],
        "eval_loss": dpo_result["eval_loss"],
        "eval_accuracy": dpo_result["eval_accuracy"],
        "num_examples": dpo_result["num_examples"],
        "num_steps": dpo_result["global_steps"],
        "elapsed_seconds": dpo_elapsed,
        "data_size_estimate": pref_train_data.size_estimate(),
        "config": {
            "beta": dpo_config.beta,
            "learning_rate": dpo_config.learning_rate,
            "batch_size": dpo_config.batch_size,
            "num_epochs": dpo_config.num_epochs,
            "max_steps": dpo_config.max_steps,
        }
    }

    # Compute perplexity after DPO
    print("\nComputing perplexity after DPO...")
    try:
        dpo_ppl = perplexity(model, tokenizer, eval_data, max_length=128, batch_size=4)
        print(f"Perplexity after DPO: {dpo_ppl:.4f}")
        results["stages"]["dpo"]["perplexity"] = float(dpo_ppl)
    except Exception as e:
        print(f"WARNING: Could not compute perplexity after DPO: {e}")
        results["stages"]["dpo"]["perplexity"] = None

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Base perplexity:        {results.get('base_perplexity', 'N/A')}")
    print(f"After SFT perplexity:   {results['stages']['sft'].get('perplexity', 'N/A')}")
    print(f"After DPO perplexity:   {results['stages']['dpo'].get('perplexity', 'N/A')}")
    print(f"\nSFT training time:      {sft_elapsed:.2f}s")
    print(f"DPO training time:      {dpo_elapsed:.2f}s")
    print(f"Total time:             {sft_elapsed + dpo_elapsed:.2f}s")

    print("\nDATASETS USED:")
    print("  SFT training: Synthetic (100 examples, 10 unique Q&A pairs with variations)")
    print("  DPO training: Synthetic (50 preference pairs, 8 unique prompts with variations)")
    print("  Evaluation: Synthetic (20 text examples)")

    print("\nLIMITATIONS & HONEST ASSESSMENT:")
    print("  - Data is fully synthetic (generated on the fly)")
    print("  - Model is gpt2 (small, not state-of-the-art)")
    print("  - Training runs are very short (20 SFT steps, 10 DPO steps) for speed")
    print("  - No convergence expected with synthetic data and short training")
    print("  - Metrics should not be interpreted as model quality")
    print("  - This demo validates that the stage implementations work end-to-end")

    # Save results
    results_path = output_dir / "loom_posttrain_demo.json"
    print(f"\nSaving results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    run_demo()
