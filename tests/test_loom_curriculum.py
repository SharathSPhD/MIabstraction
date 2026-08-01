"""Tests for the curriculum backend."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from loom.curriculum import (
    ClassifyCompiler,
    CompileRefusal,
    InductionCompiler,
    MajorityCompiler,
    StateTrackingCompiler,
    allocate_vocabulary,
    check_gates,
    compile_curriculum,
    evaluate_curriculum,
    train,
)
from loom.spec import Skill, WeaveSpec, load_weave


@pytest.fixture
def simple_weave():
    """Minimal weave spec for testing."""
    return WeaveSpec(
        model={"d_model": 64, "n_layers": 2, "n_heads": 4, "max_len": 64},
        skills=[
            Skill(name="test_induction", kind="induction"),
        ],
        gates=[],  # Gates handled separately in plan
    )


class TestInductionCompiler:
    """Test the induction task compiler."""

    def test_generator_produces_valid_sequences(self):
        """Verify generator produces well-formed gapped doubled sequences."""
        compiler = InductionCompiler(copy_len=4, max_gap=8, vocab_offset=0)
        rng = np.random.default_rng(42)
        seqs, gaps = compiler.generator(n_seq=10, rng=rng)

        assert seqs.shape == (10, 16)  # 2*4 + 8
        assert gaps.shape == (10,)
        assert np.all(gaps >= 0) and np.all(gaps <= 8)

        # Verify structure: copy_len + gap + copy_len + (max_gap-gap)
        for i in range(10):
            g = gaps[i]
            first_copy = seqs[i, :4]
            second_copy = seqs[i, 4 + g : 8 + g]
            np.testing.assert_array_equal(first_copy, second_copy)

    def test_generator_with_vocab_offset(self):
        """Verify vocab offset is applied."""
        compiler = InductionCompiler(copy_len=4, max_gap=8, vocab_offset=256)
        rng = np.random.default_rng(42)
        seqs, _ = compiler.generator(n_seq=5, rng=rng)

        assert np.all(seqs >= 256)
        assert np.all(seqs < 512)

    def test_evaluator_baseline(self):
        """Verify evaluator returns reasonable metrics."""
        from miabstraction.models import TinyTransformer

        compiler = InductionCompiler(copy_len=4, max_gap=8, vocab_offset=0)
        rng = np.random.default_rng(42)
        seqs, gaps = compiler.generator(n_seq=8, rng=rng)
        tokens = torch.from_numpy(seqs)

        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2)
        model.eval()

        metrics = compiler.evaluator(model, tokens, gaps, device="cpu")

        assert "prefix_score" in metrics
        assert "icl_loss" in metrics
        assert 0 <= metrics["prefix_score"] <= 1
        assert metrics["icl_loss"] >= 0


class TestStateTrackingCompiler:
    """Test the state tracking (Mess3) compiler."""

    def test_generator_produces_valid_sequences(self):
        """Verify generator produces valid Mess3 sequences and beliefs."""
        compiler = StateTrackingCompiler(seq_len=16, x=0.05, a=0.85, vocab_offset=0)
        rng = np.random.default_rng(42)
        tokens, beliefs = compiler.generator(n_seq=8, rng=rng)

        assert tokens.shape == (8, 16)
        assert beliefs.shape == (8, 16, 3)  # 3 hidden states
        assert np.all(np.abs(beliefs.sum(axis=-1) - 1.0) < 1e-5)  # Probabilities sum to 1

    def test_evaluator_baseline(self):
        """Verify evaluator returns probe_r2 metrics."""
        from miabstraction.models import TinyTransformer

        compiler = StateTrackingCompiler(seq_len=16, x=0.05, a=0.85, vocab_offset=0)
        rng = np.random.default_rng(42)
        tokens, beliefs = compiler.generator(n_seq=8, rng=rng)
        tokens = torch.from_numpy(tokens)

        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2)
        model.eval()

        metrics = compiler.evaluator(model, tokens, beliefs, device="cpu")

        assert "probe_r2" in metrics
        assert 0 <= metrics["probe_r2"] <= 1


class TestClassifyCompiler:
    """Test the classify compiler — plan-time refusal for known-hard concepts."""

    def test_parity_raises_compile_refusal(self):
        """Verify token_parity raises CompileRefusal at plan time."""
        with pytest.raises(CompileRefusal, match="token_parity"):
            ClassifyCompiler(seq_len=16, vocab_offset=256, concept="token_parity")

    def test_majority_generator_produces_valid_sequences(self):
        """Verify majority compiler generator produces valid sequences."""
        compiler = MajorityCompiler(seq_len=16, vocab_offset=256)
        rng = np.random.default_rng(42)
        seqs, answers = compiler.generator(n_seq=10, rng=rng)

        assert seqs.shape == (10, 19)  # seq_len + marker + token + answer
        assert answers.shape == (10,)
        assert np.all((answers == 0) | (answers == 1))

    def test_majority_evaluator_baseline(self):
        """Verify majority compiler evaluator returns accuracy metric."""
        from miabstraction.models import TinyTransformer

        compiler = MajorityCompiler(seq_len=16, vocab_offset=256)
        rng = np.random.default_rng(42)
        seqs, answers = compiler.generator(n_seq=8, rng=rng)
        tokens = torch.from_numpy(seqs)

        model = TinyTransformer(vocab=512, d_model=32, n_layers=2, n_heads=2, max_len=64)
        model.eval()

        metrics = compiler.evaluator(model, tokens, answers, device="cpu")

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestVocabularyPlan:
    """Test vocabulary allocation."""

    def test_allocate_vocabulary_no_overlap(self):
        """Verify skills get non-overlapping token ranges."""
        skills = [
            Skill(name="skill1", kind="induction"),
            Skill(name="skill2", kind="state_tracking", world="mess3"),
            Skill(name="skill3", kind="classify", concept="majority"),
        ]

        plan = allocate_vocabulary(skills, base_vocab=10)

        # Check no overlap
        ranges = [
            (plan.skills["skill1"]["token_start"], plan.skills["skill1"]["token_end"]),
            (plan.skills["skill2"]["token_start"], plan.skills["skill2"]["token_end"]),
            (plan.skills["skill3"]["token_start"], plan.skills["skill3"]["token_end"]),
        ]

        for i, (s1, e1) in enumerate(ranges):
            for j, (s2, e2) in enumerate(ranges):
                if i != j:
                    assert e1 <= s2 or e2 <= s1  # No overlap

    def test_task_tokens_unique(self):
        """Verify each skill gets a unique task token."""
        skills = [
            Skill(name="s1", kind="induction"),
            Skill(name="s2", kind="state_tracking", world="mess3"),
        ]

        plan = allocate_vocabulary(skills, base_vocab=10)
        task_tokens = list(plan.task_tokens.values())
        assert len(task_tokens) == len(set(task_tokens))


class TestCompileCurriculum:
    """Test curriculum compilation."""

    def test_compile_single_skill(self):
        """Verify compilation of a single-skill weave."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 2, "n_heads": 4, "max_len": 64},
            skills=[Skill(name="induction_test", kind="induction")],
            gates=[],
            seed=42,
        )

        plan = compile_curriculum(spec, max_steps=100, batch_size=16)

        assert plan.vocab_plan.total_vocab > 256
        assert "induction_test" in plan.datasets
        assert "induction_test" in plan.mixing_weights
        assert abs(plan.mixing_weights["induction_test"] - 1.0) < 1e-6

    def test_compile_multi_skill(self):
        """Verify compilation of multi-skill weave."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 2, "n_heads": 4, "max_len": 128},
            skills=[
                Skill(name="ind", kind="induction"),
                Skill(name="state", kind="state_tracking", world="mess3"),
                Skill(name="majority", kind="classify", concept="majority"),
            ],
            gates=[],
            seed=42,
        )

        plan = compile_curriculum(spec, max_steps=100, batch_size=16)

        assert len(plan.datasets) == 3
        assert abs(sum(plan.mixing_weights.values()) - 1.0) < 1e-6


class TestCheckGates:
    """Test gate checking logic."""

    def test_check_gates_all_pass(self):
        """Verify gate pass detection."""
        metrics = {
            "skill1": {"prefix_score": 0.6, "icl_loss": 0.3},
        }
        gate_metrics = {
            "skill1": {
                "prefix_score": {"op": ">", "threshold": 0.5},
                "icl_loss": {"op": "<", "threshold": 0.5},
            }
        }

        assert check_gates(metrics, gate_metrics)

    def test_check_gates_some_fail(self):
        """Verify gate failure detection."""
        metrics = {
            "skill1": {"prefix_score": 0.4, "icl_loss": 0.3},
        }
        gate_metrics = {
            "skill1": {
                "prefix_score": {"op": ">", "threshold": 0.5},
                "icl_loss": {"op": "<", "threshold": 0.5},
            }
        }

        assert not check_gates(metrics, gate_metrics)


class TestSmokeTrain:
    """Smoke tests for training."""

    @pytest.mark.timeout(60)
    def test_tiny_training_loop(self):
        """Test that training loop runs without errors."""
        from loom.spec import Gate

        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
            skills=[Skill(name="induction_smoke", kind="induction")],
            gates=[
                Gate("induction_smoke", "prefix_score", ">", 0.1),
                Gate("induction_smoke", "icl_loss", "<", 5.0),
            ],
            seed=42,
        )

        plan = compile_curriculum(spec, max_steps=50, batch_size=8, device="cpu")
        model, metrics, losses = train(spec, plan, device="cpu")

        assert model is not None
        assert "induction_smoke" in metrics
        assert len(losses) > 0
        assert metrics["induction_smoke"]["prefix_score"] >= 0

    @pytest.mark.timeout(60)
    def test_multi_skill_training_smoke(self):
        """Test multi-skill training smoke test - induction only."""
        from loom.spec import Gate

        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 96},
            skills=[
                Skill(name="ind", kind="induction"),
            ],
            gates=[
                Gate("ind", "prefix_score", ">", 0.05),
                Gate("ind", "icl_loss", "<", 5.0),
            ],
            seed=42,
        )

        plan = compile_curriculum(spec, max_steps=50, batch_size=8, device="cpu")
        model, metrics, losses = train(spec, plan, device="cpu")

        assert model is not None
        assert "ind" in metrics
        assert metrics["ind"]["prefix_score"] >= 0
