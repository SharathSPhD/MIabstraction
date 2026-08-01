"""Curriculum backend: compile skills to training objectives.

Based on E2-proven recipe: attention-only models, high sequence diversity,
and honest gate metrics that are reachable with proper hyperparameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from miabstraction.data.mess3 import belief_states, mess3_matrices, sample_sequences
from miabstraction.probes import regression_probe
from miabstraction.seeding import set_determinism


@dataclass
class VocabularyPlan:
    """Vocabulary allocation across skills in a shared model."""
    vocab_base: int  # Base vocabulary size
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
    vocab_plan: VocabularyPlan
    datasets: dict[str, dict]  # skill_name -> {compiler, n_seq_train, n_seq_eval}
    mixing_weights: dict[str, float]
    max_steps: int
    batch_size: int
    lr: float
    seed: int
    device: str
    gate_metrics: dict[str, dict]  # skill_name -> {metric_name -> {op, threshold}}
    attn_only: bool  # Use attention-only architecture

    def to_dict(self) -> dict:
        return {
            "vocab_plan": self.vocab_plan.to_dict(),
            "datasets": {k: {"kind": v.get("kind"), "n_seq_train": v.get("n_seq_train")}
                         for k, v in self.datasets.items()},
            "mixing_weights": self.mixing_weights,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.seed,
            "device": self.device,
            "gate_metrics": self.gate_metrics,
            "attn_only": self.attn_only,
        }


class InductionCompiler:
    """Compile induction skill: gapped doubled sequences (E2-proven)."""

    def __init__(
        self,
        copy_len: int = 24,
        max_gap: int = 16,
        vocab_size: int = 20,
        vocab_offset: int = 0,
        task_token: int = -1,
    ):
        self.copy_len = copy_len
        self.max_gap = max_gap
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.task_token = task_token

    def generator(
        self, n_seq: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate gapped doubled sequences. Variable gap defeats positional shortcuts."""
        L = 2 * self.copy_len + self.max_gap
        seqs = rng.integers(0, self.vocab_size, size=(n_seq, L), dtype=np.int64)
        seqs += self.vocab_offset
        gaps = rng.integers(0, self.max_gap + 1, size=n_seq)

        for i in range(n_seq):
            g = gaps[i]
            seqs[i, self.copy_len + g : 2 * self.copy_len + g] = seqs[i, :self.copy_len]
        return seqs, gaps

    def evaluator(
        self, model, tokens: torch.Tensor, gaps: np.ndarray, device: str
    ) -> dict[str, float]:
        """Evaluate prefix_score and icl_loss."""
        model.eval()
        with torch.no_grad():
            prefix_score = self._prefix_matching_score(model, tokens, gaps)
            icl_loss = self._icl_loss(model, tokens, gaps)
        return {"prefix_score": float(prefix_score), "icl_loss": float(icl_loss)}

    def _prefix_matching_score(
        self, model, tokens: torch.Tensor, gaps: np.ndarray
    ) -> float:
        """Attention mass on induction targets, max over heads & layers."""
        B, L = tokens.shape
        device = tokens.device

        pos = torch.arange(L, device=device)
        x = model.tok(tokens) + model.pos(pos)[None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        scores = []
        for blk in model.blocks:
            h = blk.ln1(x)
            _, w = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False)
            attn_out, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + attn_out
            if not blk.attn_only:
                x = x + blk.mlp(blk.ln2(x))

            masses = []
            for i in range(B):
                g = int(gaps[i])
                q = torch.arange(self.copy_len + g + 1, 2 * self.copy_len + g, device=device)
                k = q - (self.copy_len + g) + 1
                if len(q) > 0 and len(k) > 0:
                    attn_mass = w[i, :, q, k].mean(dim=0)
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
            second_loss = ce[i, self.copy_len + g : 2 * self.copy_len + g - 1].mean()
            losses.append(second_loss)
        return torch.stack(losses).mean().item()


class StateTrackingCompiler:
    """Compile state tracking (Mess3)."""

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
        self, model, tokens: torch.Tensor, beliefs: np.ndarray, device: str
    ) -> dict[str, float]:
        """Evaluate probe_r2 and incremental variant."""
        model.eval()
        with torch.no_grad():
            logits, resid = model(tokens[:, :-1], collect=True)

        best_r2 = 0.0
        for layer_idx, layer_resid in enumerate(resid):
            X = layer_resid.cpu().numpy().reshape(-1, layer_resid.shape[-1])
            Y = beliefs[:, :-1, :].reshape(-1, beliefs.shape[-1])
            probe_result = regression_probe(X, Y, val_frac=0.2, seed=0)
            if probe_result["r2_val"] > best_r2:
                best_r2 = probe_result["r2_val"]

        return {"probe_r2": float(max(0.0, best_r2))}


class ClassifyCompiler:
    """Compile classify (token parity) task."""

    def __init__(
        self,
        seq_len: int = 32,
        vocab_offset: int = 256,
        task_token: int = -1,
    ):
        self.seq_len = seq_len
        self.vocab_offset = vocab_offset
        self.task_token = task_token
        self.PARITY_MARKER = vocab_offset
        self.PARITY_0 = vocab_offset + 1
        self.PARITY_1 = vocab_offset + 2

    def generator(
        self, n_seq: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate parity sequences."""
        seqs = np.zeros((n_seq, self.seq_len + 3), dtype=np.int64)
        answers = np.zeros(n_seq, dtype=np.int64)

        for i in range(n_seq):
            target_token = rng.integers(0, 32)
            seqs[i, :self.seq_len] = rng.integers(0, 32, size=self.seq_len, dtype=np.int64)
            seqs[i, self.seq_len] = self.PARITY_MARKER
            seqs[i, self.seq_len + 1] = target_token
            count = np.sum(seqs[i, :self.seq_len] == target_token)
            parity = count % 2
            answers[i] = parity
            seqs[i, self.seq_len + 2] = self.PARITY_0 + parity

        return seqs, answers

    def evaluator(self, model, tokens: torch.Tensor, answers, device: str) -> dict[str, float]:
        """Evaluate accuracy on parity prediction."""
        model.eval()
        with torch.no_grad():
            logits = model(tokens[:, :-1])

        answer_pos = self.seq_len + 1
        answer_logits = logits[:, answer_pos, :]
        preds = answer_logits.argmax(dim=-1).cpu().numpy()

        if isinstance(answers, torch.Tensor):
            answers = answers.cpu().numpy()

        expected_tokens = self.PARITY_0 + (answers % 2)
        accuracy = np.mean(preds == expected_tokens)

        return {"accuracy": float(accuracy)}


def allocate_vocabulary(skills: list, base_vocab: int = 10) -> VocabularyPlan:
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
        token_count = 256
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
    max_steps: int = 20000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
) -> CurriculumPlan:
    """Compile a WeaveSpec into a CurriculumPlan using E2-proven hyperparameters."""
    vocab_plan = allocate_vocabulary(spec.skills)

    datasets = {}
    mixing_weights = {}
    gate_metrics = {}
    has_induction = False

    skill_names = [s.name for s in spec.skills]
    n_skills = len(skill_names)

    for skill in spec.skills:
        vocab_offset = vocab_plan.skills[skill.name]["token_start"]

        if skill.kind == "induction":
            has_induction = True
            compiler = InductionCompiler(
                copy_len=24, max_gap=16, vocab_size=20,
                vocab_offset=vocab_offset,
                task_token=vocab_plan.task_tokens[skill.name],
            )
            datasets[skill.name] = {
                "kind": "induction",
                "compiler": compiler,
                "n_seq_train": 200000,  # E2 uses 200k!
                "n_seq_eval": 256,
            }
            mixing_weights[skill.name] = 1.0 / n_skills
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
                "n_seq_train": 2048,
                "n_seq_eval": 256,
            }
            mixing_weights[skill.name] = 1.0 / n_skills
            gate_metrics[skill.name] = {
                m.metric: {"op": m.op, "threshold": m.threshold}
                for m in spec.gates_for(skill.name)
            }

        elif skill.kind == "classify":
            compiler = ClassifyCompiler(
                seq_len=32,
                vocab_offset=300,
                task_token=vocab_plan.task_tokens[skill.name],
            )
            datasets[skill.name] = {
                "kind": "classify",
                "compiler": compiler,
                "n_seq_train": 4096,
                "n_seq_eval": 256,
            }
            mixing_weights[skill.name] = 1.0 / n_skills
            gate_metrics[skill.name] = {
                m.metric: {"op": m.op, "threshold": m.threshold}
                for m in spec.gates_for(skill.name)
            }

    return CurriculumPlan(
        vocab_plan=vocab_plan,
        datasets=datasets,
        mixing_weights=mixing_weights,
        max_steps=max_steps,
        batch_size=batch_size,
        lr=lr,
        seed=spec.seed,
        device=device,
        gate_metrics=gate_metrics,
        attn_only=has_induction,  # Use attn_only if induction is present
    )


def train(
    spec,
    plan: CurriculumPlan,
    device: str = "cuda",
) -> tuple:
    """Train a single TinyTransformer on multi-task curriculum."""
    from miabstraction.models import TinyTransformer

    set_determinism(spec.seed, strict=True)

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
        attn_only=plan.attn_only,
    ).to(device)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=plan.lr)
    rng = np.random.default_rng(plan.seed)

    # Pre-generate training data for all skills
    skill_data = {}
    print("Generating training data...")
    for skill_name, dataset_cfg in plan.datasets.items():
        print(f"  {skill_name}: {dataset_cfg['n_seq_train']} sequences...")
        compiler = dataset_cfg["compiler"]
        tokens, labels = compiler.generator(dataset_cfg["n_seq_train"], rng)
        skill_data[skill_name] = {
            "tokens": torch.from_numpy(tokens).to(device),
            "labels": labels if isinstance(labels, np.ndarray) else torch.from_numpy(labels).to(device),
            "compiler": compiler,
        }

    print("Training...")
    losses_log = []
    skill_metrics_log = {name: {} for name in plan.datasets.keys()}
    eval_step = 100

    for step in range(plan.max_steps):
        # Sample skill and batch
        skill_name = rng.choice(list(plan.datasets.keys()))
        skill_tokens = skill_data[skill_name]["tokens"]

        batch_idx = rng.integers(0, len(skill_tokens), size=plan.batch_size)
        batch = skill_tokens[batch_idx]

        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses_log.append(loss.item())

        if step % eval_step == 0 or step == plan.max_steps - 1:
            metrics_dict = evaluate_curriculum(model, plan, device, rng)
            for skill_name, metrics in metrics_dict.items():
                skill_metrics_log[skill_name] = metrics

            all_pass = check_gates(metrics_dict, plan.gate_metrics)
            status = " [ALL GATES PASS]" if all_pass else ""
            print(f"Step {step}: loss={loss.item():.4f}{status}")

            if all_pass and step > 500:
                print(f"Early stop at step {step}")
                break

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
