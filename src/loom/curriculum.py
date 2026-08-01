"""Curriculum backend: compile skills to training objectives.

Transforms a WeaveSpec into a multi-task training curriculum with per-skill
token ranges, task generators, and evaluators that verify gate metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from miabstraction.data.mess3 import belief_states, mess3_matrices, sample_sequences
from miabstraction.probes import regression_probe
from miabstraction.seeding import set_determinism


@dataclass
class VocabularyPlan:
    """Vocabulary allocation across skills in a shared model."""
    vocab_base: int  # Base vocabulary size (shared padding tokens, BOS, EOS)
    skills: dict[str, dict]  # skill_name -> {token_start, token_end, n_tokens}
    task_tokens: dict[str, int]  # skill_name -> task_token_id
    total_vocab: int  # cumulative vocabulary size

    def to_dict(self) -> dict:
        return {
            "vocab_base": self.vocab_base,
            "skills": self.skills,
            "task_tokens": self.task_tokens,
            "total_vocab": self.total_vocab,
        }


@dataclass
class CurriculumPlan:
    """Training plan: datasets, mixing ratios, hyperparameters."""
    spec: dict  # WeaveSpec serialized
    vocab_plan: VocabularyPlan
    datasets: dict[str, dict]  # skill_name -> {generator_config, eval_config}
    mixing_weights: dict[str, float]  # skill_name -> weight in training
    max_steps: int
    batch_size: int
    lr: float
    seed: int
    device: str
    gate_metrics: dict[str, dict]  # skill_name -> {metric_name -> Gate threshold}

    def to_dict(self) -> dict:
        return {
            "vocab_plan": self.vocab_plan.to_dict(),
            "datasets": self.datasets,
            "mixing_weights": self.mixing_weights,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.seed,
            "device": self.device,
            "gate_metrics": self.gate_metrics,
        }


class InductionCompiler:
    """Compile induction skill: gapped doubled sequences."""

    def __init__(
        self,
        copy_len: int = 8,
        max_gap: int = 16,
        vocab_offset: int = 0,
        task_token: int = -1,
    ):
        self.copy_len = copy_len
        self.max_gap = max_gap
        self.vocab_offset = vocab_offset
        self.task_token = task_token

    def generator(
        self, n_seq: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate gapped doubled sequences.
        Returns (seqs, gaps) where seqs[i, j] is in [vocab_offset, vocab_offset+vocab).
        """
        vocab = 256  # local vocab for the task
        L = 2 * self.copy_len + self.max_gap
        seqs = rng.integers(0, vocab, size=(n_seq, L), dtype=np.int64)
        seqs += self.vocab_offset  # offset into global vocabulary
        gaps = rng.integers(0, self.max_gap + 1, size=n_seq)

        for i in range(n_seq):
            g = gaps[i]
            seqs[i, self.copy_len + g : 2 * self.copy_len + g] = seqs[i, :self.copy_len]
        return seqs, gaps

    def evaluator(
        self,
        model,
        tokens: torch.Tensor,
        gaps: np.ndarray,
        device: str,
    ) -> dict[str, float]:
        """Evaluate prefix_score and icl_loss."""
        model.eval()
        with torch.no_grad():
            # Prefix matching score: attention mass on induction target
            prefix_score = self._prefix_matching_score(model, tokens, gaps)

            # ICL loss: loss on second copy region
            icl_loss = self._icl_loss(model, tokens, gaps)

        return {
            "prefix_score": float(prefix_score),
            "icl_loss": float(icl_loss),
        }

    def _prefix_matching_score(
        self, model, tokens: torch.Tensor, gaps: np.ndarray
    ) -> float:
        """Mean attention mass on induction target, max over heads & layers.

        For an untrained model, this will be near 0. As the model learns induction,
        the score should rise toward 1 as attention heads learn to implement the
        pattern-matching attention.
        """
        B, L = tokens.shape
        device = tokens.device

        # Manually extract attention patterns using the forward pass
        pos = torch.arange(L, device=device)
        x = model.tok(tokens) + model.pos(pos)[None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        scores = []
        for blk in model.blocks:
            h = blk.ln1(x)
            # Extract attention weights
            _, w = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False)
            # Compute the residual update for the next iteration
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + attn_out
            if not blk.attn_only:
                x = x + blk.mlp(blk.ln2(x))

            # w: (B, H, L, L) - extract induction pattern
            masses = []
            for i in range(B):
                g = int(gaps[i])
                # Second copy positions where we want to attend to first copy
                q = torch.arange(self.copy_len + g + 1, 2 * self.copy_len + g, device=device)
                # Corresponding first copy positions (off by 1 for next-token copying)
                k = q - (self.copy_len + g) + 1
                if len(q) > 0 and len(k) > 0:
                    # Attention mass: average over query positions, max over heads
                    attn_mass = w[i, :, q, k].mean(dim=0)  # (H,)
                    masses.append(attn_mass.max().item())
                else:
                    masses.append(0.0)
            if masses:
                scores.append(np.mean(masses))

        return max(scores) if scores else 0.0

    def _icl_loss(self, model, tokens: torch.Tensor, gaps: np.ndarray) -> float:
        """Loss on second copy region."""
        logits = model(tokens[:, :-1])
        ce = F.cross_entropy(logits.transpose(1, 2), tokens[:, 1:], reduction="none")
        losses = []
        for i in range(tokens.shape[0]):
            g = int(gaps[i])
            # Second copy spans positions copy_len+g to 2*copy_len+g-1
            second_loss = ce[i, self.copy_len + g : 2 * self.copy_len + g - 1].mean()
            losses.append(second_loss)
        return torch.stack(losses).mean().item()


class StateTrackingCompiler:
    """Compile state tracking (Mess3): predict belief states from hidden process."""

    def __init__(
        self,
        seq_len: int = 32,
        x: float = 0.05,
        a: float = 0.85,
        vocab_offset: int = 0,
        task_token: int = -1,
    ):
        self.seq_len = seq_len
        self.x = x
        self.a = a
        self.vocab_offset = vocab_offset
        self.task_token = task_token

    def generator(
        self, n_seq: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Mess3 sequences and ground-truth belief states."""
        T = mess3_matrices(self.x, self.a)
        tokens = sample_sequences(T, n_seq, self.seq_len, rng)
        tokens += self.vocab_offset
        beliefs = belief_states(T, tokens - self.vocab_offset)
        return tokens, beliefs

    def evaluator(
        self,
        model,
        tokens: torch.Tensor,
        beliefs: np.ndarray,
        device: str,
    ) -> dict[str, float]:
        """Evaluate probe_r2 and incremental-aware variant."""
        model.eval()
        with torch.no_grad():
            logits, resid = model(tokens[:, :-1], collect=True)

        # Fit linear probe from best layer to ground-truth beliefs
        best_r2 = 0.0
        best_layer = 0
        for layer_idx, layer_resid in enumerate(resid):
            # layer_resid: (B, L, d_model)
            X = layer_resid.cpu().numpy().reshape(-1, layer_resid.shape[-1])
            Y = beliefs[:, :-1, :].reshape(-1, beliefs.shape[-1])
            probe_result = regression_probe(X, Y, val_frac=0.2, seed=0)
            if probe_result["r2_val"] > best_r2:
                best_r2 = probe_result["r2_val"]
                best_layer = layer_idx

        # Incremental R² (beyond 8-token window baseline)
        probe_r2_incremental = self._incremental_r2(
            resid[best_layer], beliefs, window_size=8
        )

        return {
            "probe_r2": float(max(0.0, best_r2)),  # clip to [0, 1]
            "probe_r2_incremental": float(max(0.0, probe_r2_incremental)),
            "best_layer": best_layer,
        }

    def _incremental_r2(
        self, layer_resid: torch.Tensor, beliefs: np.ndarray, window_size: int
    ) -> float:
        """R² gain beyond window-baseline reservoir."""
        X = layer_resid.cpu().numpy().reshape(-1, layer_resid.shape[-1])
        Y = beliefs[:, :-1, :].reshape(-1, beliefs.shape[-1])

        # Baseline: fit only on reservoir (first window_size tokens)
        B, L, d = layer_resid.shape
        window_mask = np.zeros(X.shape[0], dtype=bool)
        for i in range(B):
            window_mask[i * L : i * L + window_size] = True

        X_reservoir = X[window_mask]
        Y_reservoir = Y[window_mask]
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        baseline_reg = LinearRegression().fit(X_reservoir, Y_reservoir)
        baseline_r2 = r2_score(Y, baseline_reg.predict(X))

        # Full fit
        full_reg = LinearRegression().fit(X, Y)
        full_r2 = r2_score(Y, full_reg.predict(X))

        return max(0.0, full_r2 - baseline_r2)


class ClassifyCompiler:
    """Compile classify (token parity): predict parity of a token in the sequence.

    The task: predict a parity token (0 or 1) that follows a special marker.
    Sequence structure: [content...] [PARITY_MARKER] [parity_answer]
    The model must count occurrences of a target token and output its parity.
    """

    def __init__(
        self,
        seq_len: int = 32,
        vocab_offset: int = 256,  # Use high token IDs for special markers
        task_token: int = -1,
    ):
        self.seq_len = seq_len
        self.vocab_offset = vocab_offset
        self.task_token = task_token
        self.PARITY_MARKER = vocab_offset  # Special marker for parity task
        self.PARITY_0 = vocab_offset + 1  # Parity is 0
        self.PARITY_1 = vocab_offset + 2  # Parity is 1

    def generator(
        self, n_seq: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate sequences with parity labels.

        Format: [content tokens] [MARKER] [target_token_id]
        The next token after target_token_id should be the parity (0 or 1).
        """
        seqs = np.zeros((n_seq, self.seq_len + 3), dtype=np.int64)
        answers = np.zeros(n_seq, dtype=np.int64)

        for i in range(n_seq):
            # Pick a target token to count parity of (use low token IDs)
            target_token = rng.integers(0, 32)
            # Fill sequence with content
            seqs[i, :self.seq_len] = rng.integers(0, 32, size=self.seq_len, dtype=np.int64)
            # Add parity marker
            seqs[i, self.seq_len] = self.PARITY_MARKER
            # Add target token ID
            seqs[i, self.seq_len + 1] = target_token

            # Count parity of target_token in sequence
            count = np.sum(seqs[i, :self.seq_len] == target_token)
            parity = count % 2
            answers[i] = parity
            # Add the parity token (will be predicted by next-token prediction)
            seqs[i, self.seq_len + 2] = self.PARITY_0 + parity

        return seqs, answers

    def evaluator(
        self,
        model,
        tokens: torch.Tensor,
        answers,
        device: str,
    ) -> dict[str, float]:
        """Evaluate accuracy on parity prediction."""
        model.eval()
        with torch.no_grad():
            logits = model(tokens[:, :-1])

        # The parity answer is at position seq_len + 2
        # So we look at the logits at position seq_len + 1 (off-by-one due to next-token)
        answer_pos = self.seq_len + 1
        answer_logits = logits[:, answer_pos, :]  # (B, vocab)

        # Get predictions
        preds = answer_logits.argmax(dim=-1).cpu().numpy()
        expected_tokens = self.PARITY_0 + (answers % 2)

        # Handle both numpy array and torch tensor
        if isinstance(answers, torch.Tensor):
            answers = answers.cpu().numpy()

        accuracy = np.mean(preds == expected_tokens)

        return {
            "accuracy": float(accuracy),
        }


def allocate_vocabulary(
    skills: list,
    base_vocab: int = 10,
) -> VocabularyPlan:
    """Allocate vocabulary ranges to skills."""
    plan = VocabularyPlan(
        vocab_base=base_vocab,
        skills={},
        task_tokens={},
        total_vocab=base_vocab,
    )

    token_idx = base_vocab
    for skill in skills:
        task_token = token_idx
        token_idx += 1
        token_start = token_idx
        token_count = 256  # Each task gets 256 tokens
        token_end = token_idx + token_count

        plan.skills[skill.name] = {
            "token_start": token_start,
            "token_end": token_end,
            "n_tokens": token_count,
        }
        plan.task_tokens[skill.name] = task_token
        token_idx = token_end

    plan.total_vocab = token_idx
    return plan


def compile_curriculum(
    spec,
    max_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
) -> CurriculumPlan:
    """Compile a WeaveSpec into a CurriculumPlan."""
    from loom.spec import WeaveSpec

    # Allocate vocabulary
    vocab_plan = allocate_vocabulary(spec.skills)

    # Build per-skill compilers and configs
    datasets = {}
    mixing_weights = {}
    gate_metrics = {}

    skill_names = [s.name for s in spec.skills]
    for skill in spec.skills:
        vocab_offset = vocab_plan.skills[skill.name]["token_start"]

        if skill.kind == "induction":
            compiler = InductionCompiler(
                copy_len=8,
                max_gap=16,
                vocab_offset=vocab_offset,
                task_token=vocab_plan.task_tokens[skill.name],
            )
            datasets[skill.name] = {
                "kind": "induction",
                "compiler": compiler,
                "n_seq_train": 256,
                "n_seq_eval": 64,
            }
            mixing_weights[skill.name] = 1.0 / len(skill_names)
            gate_metrics[skill.name] = {
                m.metric: {"op": m.op, "threshold": m.threshold}
                for m in spec.gates_for(skill.name)
            }

        elif skill.kind == "state_tracking":
            compiler = StateTrackingCompiler(
                seq_len=32,
                x=0.05,
                a=0.85,
                vocab_offset=vocab_offset,
                task_token=vocab_plan.task_tokens[skill.name],
            )
            datasets[skill.name] = {
                "kind": "state_tracking",
                "compiler": compiler,
                "n_seq_train": 128,
                "n_seq_eval": 32,
            }
            mixing_weights[skill.name] = 1.0 / len(skill_names)
            gate_metrics[skill.name] = {
                m.metric: {"op": m.op, "threshold": m.threshold}
                for m in spec.gates_for(skill.name)
            }

        elif skill.kind == "classify":
            compiler = ClassifyCompiler(
                seq_len=32,
                vocab_offset=vocab_offset,
                task_token=vocab_plan.task_tokens[skill.name],
            )
            datasets[skill.name] = {
                "kind": "classify",
                "compiler": compiler,
                "n_seq_train": 256,
                "n_seq_eval": 64,
            }
            mixing_weights[skill.name] = 1.0 / len(skill_names)
            gate_metrics[skill.name] = {
                m.metric: {"op": m.op, "threshold": m.threshold}
                for m in spec.gates_for(skill.name)
            }

    return CurriculumPlan(
        spec=spec.__dict__,
        vocab_plan=vocab_plan,
        datasets=datasets,
        mixing_weights=mixing_weights,
        max_steps=max_steps,
        batch_size=batch_size,
        lr=lr,
        seed=spec.seed,
        device=device,
        gate_metrics=gate_metrics,
    )


def train(
    spec,
    plan: CurriculumPlan,
    device: str = "cuda",
) -> tuple:
    """Train a single TinyTransformer on multi-task curriculum.

    Returns (model, per_skill_metrics) where per_skill_metrics[skill_name]
    contains the evaluated gate metrics.
    """
    from miabstraction.models import TinyTransformer
    from miabstraction.seeding import set_determinism

    set_determinism(spec.seed, strict=True)

    # Create model
    d_model = spec.model["d_model"]
    n_layers = spec.model["n_layers"]
    n_heads = spec.model["n_heads"]
    max_len = spec.model["max_len"]
    vocab = plan.vocab_plan.total_vocab

    model = TinyTransformer(
        vocab=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        attn_only=False,  # With MLPs as in the spec
    ).to(device)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=plan.lr)
    rng = np.random.default_rng(plan.seed)

    # Prepare datasets for all skills
    skill_data = {}
    for skill_name, dataset_cfg in plan.datasets.items():
        compiler = dataset_cfg["compiler"]
        tokens, labels = compiler.generator(dataset_cfg["n_seq_train"], rng)
        skill_data[skill_name] = {
            "tokens": torch.from_numpy(tokens).to(device),
            "labels": labels if isinstance(labels, np.ndarray) else torch.from_numpy(labels).to(device),
            "compiler": compiler,
        }

    # Training loop with early stopping
    losses_log = []
    skill_metrics_log = {name: {} for name in plan.datasets.keys()}

    for step in range(plan.max_steps):
        # Sample skill and batch
        skill_name = rng.choice(list(plan.datasets.keys()))
        skill_tokens = skill_data[skill_name]["tokens"]

        # Sample batch
        batch_idx = rng.integers(0, len(skill_tokens), size=plan.batch_size)
        batch = skill_tokens[batch_idx]

        # Forward pass
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
        )

        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses_log.append(loss.item())

        # Evaluate gates periodically
        if step % 100 == 0 or step == plan.max_steps - 1:
            metrics_dict = evaluate_curriculum(model, plan, device, rng)
            for skill_name, metrics in metrics_dict.items():
                skill_metrics_log[skill_name] = metrics

            # Check if all gates pass
            all_pass = check_gates(metrics_dict, plan.gate_metrics)
            if all_pass and step > 500:  # Early stopping after reaching threshold
                print(f"All gates passed at step {step}!")
                break

    # Final evaluation
    final_metrics = evaluate_curriculum(model, plan, device, rng)

    return model, final_metrics, losses_log


def evaluate_curriculum(
    model, plan: CurriculumPlan, device: str, rng: np.random.Generator
) -> dict:
    """Evaluate all skills in the curriculum."""
    metrics = {}

    model.eval()
    with torch.no_grad():
        for skill_name, dataset_cfg in plan.datasets.items():
            compiler = dataset_cfg["compiler"]
            # Generate eval dataset
            tokens, labels = compiler.generator(dataset_cfg["n_seq_eval"], rng)
            tokens = torch.from_numpy(tokens).to(device)

            if dataset_cfg["kind"] == "induction":
                metrics[skill_name] = compiler.evaluator(model, tokens, labels, device)
            elif dataset_cfg["kind"] == "state_tracking":
                metrics[skill_name] = compiler.evaluator(model, tokens, labels, device)
            elif dataset_cfg["kind"] == "classify":
                labels = torch.from_numpy(labels).to(device)
                metrics[skill_name] = compiler.evaluator(model, tokens, labels, device)

    return metrics


def check_gates(metrics: dict, gate_metrics: dict) -> bool:
    """Check if all gates pass."""
    for skill_name, skill_gates in gate_metrics.items():
        if skill_name not in metrics:
            return False
        for metric_name, gate_spec in skill_gates.items():
            if metric_name not in metrics[skill_name]:
                return False
            value = metrics[skill_name][metric_name]
            op = gate_spec["op"]
            threshold = gate_spec["threshold"]

            if op == ">":
                if value <= threshold:
                    return False
            elif op == "<":
                if value >= threshold:
                    return False

    return True
