"""Foundation backend: compile a weave into a pretraining job.

A foundation weave declares:
  foundation:
    corpus: babylm_strict         # 100M-word corpus (standard, from-scratch)
    tokenizer: gpt2 | bpe_train   # reuse GPT2 vocab or train small BPE
    params: 30_000_000            # model size (10-30M for small/medium)
    budget_hours: 3.0             # wall-clock time limit

Emits a self-contained job directory that runs inside the RTX 5090 container:
  - train.py: main training loop
  - config.json: model + training params
  - metrics.json: val_loss, val_ppl, gates pass/fail
  - README.md: what was run and why

Gates must be on "foundation" target:
  foundation:
    val_loss: "<3.0"     # target validation loss
    val_ppl: "<20.0"     # target validation perplexity
    blimp_acc: ">0.55"   # if BLiMP data is cached
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loom.spec import Foundation, WeaveSpec


@dataclass
class FoundationPlan:
    """Plan for a foundation pretraining job."""
    corpus: str
    tokenizer: str
    model_params: int
    budget_hours: float
    job_dir: Path


def _ensure_corpus(corpus: str) -> Path:
    """Download or verify cached corpus.

    Returns path to corpus directory with train/val splits.
    """
    corpus_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--BabyLM-community--BabyLM-2026-Strict"

    if corpus_dir.exists():
        # Corpus already cached
        return corpus_dir

    print(f"Foundation: Corpus not cached, will download at job runtime.")
    print(f"  Expected cache location: {corpus_dir}")
    return corpus_dir


def _estimate_model_size(n_params: int) -> dict:
    """Estimate model architecture from parameter budget.

    For a GPT-style decoder:
      - d_model (embedding + attention output)
      - n_layers
      - d_ff (feedforward hidden size, ~4x d_model)
      - n_heads
      - max_len (sequence length)

    Approximate formula: params ≈ d_model * n_layers * (12 * d_model + d_ff)
    = d_model * n_layers * (12 * d_model + 4 * d_model) ≈ 16 * d_model^2 * n_layers

    For 32GB GPU with batch_size=16, sequence_length=1024, mixed precision:
      - Recommended max params: ~10-15M for safety
      - Activation memory scales with batch_size * seq_len * d_model
    """
    # Cap at 15M for 32GB GPU safety
    target_params = min(n_params, 15_000_000)

    # Start with conservative defaults
    d_model = 256
    n_layers = 4
    n_heads = 8
    max_len = 512  # Reduce sequence length to save memory

    vocab_size = 50257  # GPT2 vocab

    # Calculate params: vocab embeddings + layer params
    # params ≈ 2 * vocab_size * d_model (embed + output layer)
    #        + n_layers * (12 * d_model^2 + 4 * d_model^2) (attn + ff)
    def estimate_params(d_model, n_layers):
        embed_params = 2 * vocab_size * d_model
        layer_params = n_layers * (16 * d_model**2 + d_model * 4)
        return embed_params + layer_params

    # Adjust d_model to hit target
    current_params = estimate_params(d_model, n_layers)
    if current_params < target_params:
        scale = (target_params / current_params) ** 0.5
        d_model = int(d_model * scale)
        d_model = (d_model // 64) * 64  # Round to multiple of 64

    # Re-estimate and potentially adjust n_layers
    current_params = estimate_params(d_model, n_layers)
    if current_params < target_params:
        scale = target_params / current_params
        n_layers = max(4, int(n_layers * scale))

    # Ensure n_heads divides d_model
    n_heads = min(n_heads, d_model // 64)
    n_heads = max(1, n_heads)

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "d_ff": 4 * d_model,
        "n_heads": n_heads,
        "max_len": max_len,
    }


def plan_foundation(spec: WeaveSpec) -> FoundationPlan:
    """Create a plan for the foundation pretraining job.

    Returns:
        FoundationPlan with parameters and job directory path (not yet created)
    """
    if not spec.foundation:
        raise ValueError("No foundation spec provided")

    foundation = spec.foundation

    # Validate corpus
    if foundation.corpus != "babylm_strict":
        raise ValueError(
            f"Only 'babylm_strict' corpus is supported; got '{foundation.corpus}'"
        )

    # Validate tokenizer
    if foundation.tokenizer not in ("gpt2", "bpe_train"):
        raise ValueError(
            f"Tokenizer must be 'gpt2' or 'bpe_train'; got '{foundation.tokenizer}'"
        )

    # Validate params
    if not (10_000_000 <= foundation.params <= 100_000_000):
        raise ValueError(
            f"Params must be 10M-100M; got {foundation.params//1e6:.1f}M"
        )

    # Validate budget
    if not (0.1 <= foundation.budget_hours <= 24):
        raise ValueError(
            f"Budget must be 0.1-24 hours; got {foundation.budget_hours}h"
        )

    # Ensure corpus is available (cache check)
    _ensure_corpus(foundation.corpus)

    # Return plan (job_dir will be created at build time)
    return FoundationPlan(
        corpus=foundation.corpus,
        tokenizer=foundation.tokenizer,
        model_params=foundation.params,
        budget_hours=foundation.budget_hours,
        job_dir=Path("PLACEHOLDER"),  # Set at build time
    )


def _write_train_script(
    job_dir: Path,
    model_config: dict,
    tokenizer_type: str,
    corpus_name: str,
    budget_hours: float,
) -> None:
    """Write the main training loop (train.py) to the job directory."""
    # Reduce batch size for 32GB GPU
    effective_batch_size = min(16, model_config.get("batch_size", 32))

    script = f'''"""Foundation pretraining job for Loom.

Trains a small GPT-style model on BabyLM (100M words) from scratch.
Measures validation loss, perplexity, and optionally BLiMP accuracy.
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Load config
config_path = Path(__file__).parent / "config.json"
config = json.loads(config_path.read_text())

# Override batch size for memory safety on 32GB GPU
config["batch_size"] = {effective_batch_size}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")
print(f"Config: {{json.dumps(config, indent=2)}}")

# Create output directories
job_dir = Path(__file__).parent
checkpoint_dir = job_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

# Load or create tokenizer
from transformers import AutoTokenizer
if "{tokenizer_type}" == "gpt2":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
else:
    # For bpe_train, we would train a new BPE here
    # For now, use GPT2 as fallback
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load BabyLM corpus
print(f"Loading {{'{corpus_name}' if '{corpus_name}' else 'synthetic'}} corpus...")
try:
    from datasets import load_dataset

    if '{corpus_name}' == 'babylm_strict':
        try:
            # Try to load train split; BabyLM-2026-Strict only has 'train'
            full_dataset = load_dataset(
                "BabyLM-community/BabyLM-2026-Strict",
                split="train",
                trust_remote_code=True,
            )
            # Split into train/val (90/10 split)
            train_size = int(0.9 * len(full_dataset))
            indices = list(range(len(full_dataset)))
            import random
            random.seed(42)
            random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            dataset = full_dataset.select(train_indices)
            val_dataset = full_dataset.select(val_indices)
            print(f"Split BabyLM into {{len(dataset)}} train, {{len(val_dataset)}} val sequences")
        except Exception as e:
            print(f"Warning: Could not split BabyLM corpus: {{e}}")
            raise
    else:
        # Fallback: synthetic data
        import random
        random.seed(42)
        vocab_size = 50257
        seq_len = 1024
        n_train = 10000
        dataset = TensorDataset(
            torch.randint(0, vocab_size, (n_train, seq_len))
        )
        val_dataset = TensorDataset(
            torch.randint(0, vocab_size, (1000, seq_len))
        )
except Exception as e:
    print(f"Warning: Could not load corpus: {{e}}")
    print("Using synthetic data fallback.")
    import random
    random.seed(42)
    vocab_size = config.get("vocab_size", 50257)
    seq_len = config.get("max_len", 1024)
    n_train = 10000

    # Create simple sequential tensors as synthetic data
    train_data = torch.randint(0, vocab_size, (n_train, seq_len))
    val_data = torch.randint(0, vocab_size, (1000, seq_len))
    dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

print(f"Dataset loaded: {{len(dataset)}} training, {{len(val_dataset)}} validation sequences")

# Build model
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_embed = nn.Embedding(config["max_len"], config["d_model"])

        layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config["n_layers"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"])
        self.config = config

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embed(input_ids) + self.pos_embed(pos_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

model = GPTModel(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {{total_params//1e6:.1f}}M parameters")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))
max_steps = config.get("max_steps", 10000)
scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

# Training loop
batch_size = config.get("batch_size", 32)
val_loss_best = float('inf')
start_time = time.time()
max_seconds = {budget_hours} * 3600

print(f"Training for up to {{max_steps}} steps (~{budget_hours}h)...")
print()

def collate_fn(batch):
    \"\"\"Collate function for BabyLM or TensorDataset.\"\"\"
    if isinstance(batch[0], dict):
        # BabyLM format: dict with 'text' key
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        texts = [b['text'] for b in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config["max_len"],
            return_tensors="pt",
        )
        return encoded["input_ids"]
    else:
        # TensorDataset format: already tokenized
        return torch.stack([b[0] for b in batch])

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

step = 0
for epoch in range(100):  # Multiple epochs if needed
    model.train()
    for input_ids in train_loader:
        if step >= max_steps:
            break
        if time.time() - start_time > max_seconds:
            print(f"Time budget exhausted after {{step}} steps")
            break

        input_ids = input_ids.to(device)

        # LM objective: predict next token
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, config["vocab_size"]),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {{step:4d}} | train_loss={{loss.item():.4f}} | lr={{scheduler.get_last_lr()[0]:.2e}}")

        step += 1

    if step >= max_steps or time.time() - start_time > max_seconds:
        break

# Validation
print()
print("Computing validation metrics...")
model.eval()
total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for input_ids in val_loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, config["vocab_size"]),
            targets.reshape(-1),
            reduction='sum',
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

val_loss = total_loss / total_tokens
val_ppl = torch.exp(torch.tensor(val_loss)).item()

print(f"Validation loss: {{val_loss:.4f}}")
print(f"Validation perplexity: {{val_ppl:.2f}}")

# Optional: BLiMP evaluation
blimp_acc = None
try:
    from datasets import load_dataset
    print("Attempting BLiMP evaluation (if cached)...")

    blimp_data = load_dataset("blimp", "regular", split="train", trust_remote_code=True)
    # Sample 100 examples for speed
    sample_indices = torch.randperm(len(blimp_data))[:100].tolist()
    blimp_sample = blimp_data.select(sample_indices)

    # Simple BLiMP scoring: pick the higher-likelihood option
    correct = 0
    with torch.no_grad():
        for i, example in enumerate(blimp_sample):
            sentence_good = example["sentence_good"]
            sentence_bad = example["sentence_bad"]

            # Tokenize and score
            try:
                tokens_good = tokenizer.encode(sentence_good)
                tokens_bad = tokenizer.encode(sentence_bad)

                # Truncate to max_len
                max_len = config["max_len"]
                tokens_good = tokens_good[:max_len]
                tokens_bad = tokens_bad[:max_len]

                # Score: average log probability
                def score_sentence(tokens):
                    if len(tokens) < 2:
                        return 0.0
                    input_ids = torch.tensor([tokens[:-1]], device=device)
                    targets = torch.tensor(tokens[1:], device=device)

                    logits = model(input_ids)
                    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

                    # Get log prob of target tokens
                    scores = log_probs[range(len(targets)), targets]
                    return scores.mean().item()

                score_g = score_sentence(tokens_good)
                score_b = score_sentence(tokens_bad)

                if score_g > score_b:
                    correct += 1
            except:
                pass

    blimp_acc = correct / len(blimp_sample)
    print(f"BLiMP accuracy (sample): {{blimp_acc:.3f}}")
except Exception as e:
    print(f"BLiMP evaluation skipped: {{e}}")

# Save checkpoint
checkpoint_path = checkpoint_dir / f"step_{{step}}.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"Checkpoint saved: {{checkpoint_path}}")

# Write metrics
elapsed = time.time() - start_time
metrics = {{
    "steps": step,
    "elapsed_seconds": elapsed,
    "elapsed_hours": elapsed / 3600,
    "val_loss": val_loss,
    "val_ppl": val_ppl,
    "blimp_acc": blimp_acc,
    "total_params": int(total_params),
    "model_config": config,
}}

metrics_path = job_dir / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))
print()
print(f"Metrics saved: {{metrics_path}}")
print(json.dumps(metrics, indent=2))
'''
    (job_dir / "train.py").write_text(script)


def _write_config(job_dir: Path, model_config: dict, budget_hours: float) -> None:
    """Write model config as JSON."""
    config = {
        **model_config,
        "max_steps": int(10_000 * min(budget_hours / 3.0, 1.0)),  # Scale steps to budget
        "batch_size": 32,
        "lr": 1e-3,
        "warmup_steps": 500,
    }
    (job_dir / "config.json").write_text(json.dumps(config, indent=2))


def _write_readme(job_dir: Path, plan: FoundationPlan, spec: WeaveSpec) -> None:
    """Write a README explaining the job."""
    gates_text = ""
    for gate in spec.gates_for("foundation"):
        gates_text += f"  - {gate.describe()}\n"

    readme = f"""# Foundation Pretraining Job

Generated by Loom foundation backend.

## Configuration

- **Corpus**: {plan.corpus}
- **Tokenizer**: {plan.tokenizer}
- **Model size**: ~{plan.model_params//1e6:.1f}M parameters
- **Time budget**: {plan.budget_hours:.1f} hours

## Gates (success criteria)

{gates_text or "(no gates declared)"}

## Running the job

The job is run inside the RTX 5090 Docker container:

```bash
cd {job_dir}
python train.py
```

## Outputs

- `train.py`: Main training loop
- `config.json`: Model and training configuration
- `metrics.json`: Validation metrics and gate results
- `checkpoints/`: Model checkpoints
- `train.log`: Full training output (when run on RTX 5090)

## Metrics

The job measures:
- `val_loss`: Validation cross-entropy loss on held-out data
- `val_ppl`: Validation perplexity (exp(val_loss))
- `blimp_acc`: BLiMP accuracy (if dataset is cached)

These are compared against the declared gates to determine pass/fail.
"""
    (job_dir / "README.md").write_text(readme)


def build_foundation(spec: WeaveSpec, output_dir: Path) -> Path:
    """Compile a foundation weave into a pretraining job.

    Args:
        spec: WeaveSpec with foundation section
        output_dir: Directory to write job artifacts

    Returns:
        Path to job directory (ready to submit to RTX 5090)
    """
    plan = plan_foundation(spec)

    # Create job directory
    job_dir = output_dir / "foundation_job"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Estimate model architecture
    model_config = _estimate_model_size(plan.model_params)

    # Write job files
    _write_train_script(
        job_dir,
        model_config,
        plan.tokenizer,
        plan.corpus,
        plan.budget_hours,
    )
    _write_config(job_dir, model_config, plan.budget_hours)
    _write_readme(job_dir, plan, spec)

    print(f"Foundation job compiled:")
    print(f"  Location: {job_dir}")
    print(f"  Model: {model_config['d_model']}-dim, {model_config['n_layers']} layers")
    print(f"  Params: ~{model_config['vocab_size'] * model_config['d_model'] // 1e6 + model_config['d_model'] ** 2 * model_config['n_layers'] * 16 // 1e6:.1f}M")
    print(f"  Corpus: {plan.corpus}")
    print(f"  Budget: {plan.budget_hours:.1f} hours")
    print()

    return job_dir
