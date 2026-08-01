"""Tests for DPO loss implementation."""

import torch
import pytest

from loom.stages.dpo import dpo_loss


class TestDPOLoss:
    """Test the DPO loss implementation."""

    def test_dpo_loss_shape(self):
        """Test that DPO loss returns correct shape."""
        batch_size = 4
        log_probs_chosen = torch.randn(batch_size)
        log_probs_rejected = torch.randn(batch_size)
        log_probs_ref_chosen = torch.randn(batch_size)
        log_probs_ref_rejected = torch.randn(batch_size)

        loss, metrics = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        assert loss.shape == torch.Size([])  # Scalar
        assert isinstance(metrics, dict)
        assert "implicit_margin_mean" in metrics
        assert "accuracy" in metrics

    def test_dpo_loss_hand_computed(self):
        """Test DPO loss against hand-computed values on simple case."""
        # Simple case: 2 examples
        # Example 1: policy prefers chosen, ref prefers rejected
        log_probs_chosen = torch.tensor([0.0, -1.0])
        log_probs_rejected = torch.tensor([-1.0, 0.0])
        log_probs_ref_chosen = torch.tensor([-2.0, -1.0])
        log_probs_ref_rejected = torch.tensor([-1.0, -2.0])

        beta = 0.1
        loss, metrics = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=beta
        )

        # Hand-compute for first example:
        # reward = 0 - (-1) = 1
        # reward_ref = -2 - (-1) = -1
        # reward_diff = 1 - (-1) = 2
        # loss_1 = log(1 + exp(-0.1 * 2)) = log(1 + exp(-0.2))
        import math
        expected_loss_1 = math.log(1 + math.exp(-0.1 * 2))

        # For second example:
        # reward = -1 - 0 = -1
        # reward_ref = -1 - (-2) = 1
        # reward_diff = -1 - 1 = -2
        # loss_2 = log(1 + exp(-0.1 * -2)) = log(1 + exp(0.2))
        expected_loss_2 = math.log(1 + math.exp(0.1 * 2))

        expected_avg = (expected_loss_1 + expected_loss_2) / 2

        # Check that computed loss is close to hand-computed
        assert abs(loss.item() - expected_avg) < 1e-5

    def test_dpo_loss_prefers_chosen(self):
        """Test that DPO loss encourages preferring chosen over rejected."""
        # Create examples where policy assigns equal prob to chosen/rejected
        log_probs_chosen = torch.tensor([0.0, 0.0])
        log_probs_rejected = torch.tensor([0.0, 0.0])
        log_probs_ref_chosen = torch.tensor([-1.0, -1.0])
        log_probs_ref_rejected = torch.tensor([-1.0, -1.0])

        # When policy and ref are equal but ref is low, loss should be low
        loss_equal, _ = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        # Now policy prefers rejected (bad)
        log_probs_rejected_bad = torch.tensor([1.0, 1.0])
        loss_bad, _ = dpo_loss(
            log_probs_chosen, log_probs_rejected_bad,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        # Loss should be higher when policy prefers rejected
        assert loss_bad > loss_equal

    def test_dpo_accuracy_metric(self):
        """Test accuracy metric in DPO loss."""
        # Case 1: policy correctly prefers chosen
        log_probs_chosen = torch.tensor([0.5, 0.3])
        log_probs_rejected = torch.tensor([0.0, 0.0])
        log_probs_ref_chosen = torch.tensor([0.0, 0.0])
        log_probs_ref_rejected = torch.tensor([0.0, 0.0])

        loss, metrics = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        # Accuracy should be 1.0 (both examples have chosen > rejected)
        assert metrics["accuracy"] == 1.0

    def test_dpo_accuracy_half_correct(self):
        """Test accuracy when half are correct."""
        log_probs_chosen = torch.tensor([0.5, -0.5])
        log_probs_rejected = torch.tensor([0.0, 0.0])
        log_probs_ref_chosen = torch.tensor([0.0, 0.0])
        log_probs_ref_rejected = torch.tensor([0.0, 0.0])

        loss, metrics = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        # Accuracy should be 0.5 (first is correct, second is not)
        assert abs(metrics["accuracy"] - 0.5) < 1e-6

    def test_dpo_beta_effect(self):
        """Test that beta controls optimization strength."""
        log_probs_chosen = torch.tensor([0.1, 0.1])
        log_probs_rejected = torch.tensor([-0.1, -0.1])
        log_probs_ref_chosen = torch.tensor([0.0, 0.0])
        log_probs_ref_rejected = torch.tensor([0.0, 0.0])

        loss_low_beta, _ = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.01
        )

        loss_high_beta, _ = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=1.0
        )

        # With higher beta, reward difference is scaled more, so loss should differ
        # (Small reward_diff = 0.2 scaled by 0.01 vs 1.0)
        assert loss_low_beta != loss_high_beta

    def test_dpo_loss_is_scalar(self):
        """Test that DPO loss is a scalar tensor."""
        batch_size = 8
        log_probs_chosen = torch.randn(batch_size)
        log_probs_rejected = torch.randn(batch_size)
        log_probs_ref_chosen = torch.randn(batch_size)
        log_probs_ref_rejected = torch.randn(batch_size)

        loss, _ = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_dpo_metrics_dict_complete(self):
        """Test that metrics dict has all required fields."""
        log_probs_chosen = torch.randn(4)
        log_probs_rejected = torch.randn(4)
        log_probs_ref_chosen = torch.randn(4)
        log_probs_ref_rejected = torch.randn(4)

        loss, metrics = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        required_metrics = {
            "implicit_margin_mean",
            "implicit_margin_std",
            "chosen_margin_mean",
            "rejected_margin_mean",
            "accuracy",
        }
        assert set(metrics.keys()) >= required_metrics

        # Check that all metrics are floats
        for k, v in metrics.items():
            assert isinstance(v, (int, float))


class TestDPOGradients:
    """Test that DPO loss produces reasonable gradients."""

    def test_loss_backward(self):
        """Test that loss backward works."""
        log_probs_chosen = torch.randn(4, requires_grad=True)
        log_probs_rejected = torch.randn(4, requires_grad=True)
        log_probs_ref_chosen = torch.randn(4, requires_grad=False)
        log_probs_ref_rejected = torch.randn(4, requires_grad=False)

        loss, _ = dpo_loss(
            log_probs_chosen, log_probs_rejected,
            log_probs_ref_chosen, log_probs_ref_rejected,
            beta=0.1
        )

        loss.backward()

        # Gradients should exist
        assert log_probs_chosen.grad is not None
        assert log_probs_rejected.grad is not None
