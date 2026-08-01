"""
FINDING 3: Control Evasion (HIGH SEVERITY)

The suppress control masks token logits to prevent the model from emitting a token.
However, a model can still "express" the suppressed token through other tokens that
are semantically equivalent or through subtle shifts in the distribution.

Test cases:
1. Does masking token 2 actually prevent the model from expressing concept "2"?
2. Can the model use related tokens (e.g., token 3 when token 2 is masked)?
3. Does the side_effect measurement detect collateral damage on OOD sequences?
"""
import pytest
import torch
import torch.nn as nn
from loom.runtime import ControlledModel, ControlRecord, compute_suppression_metrics

class TestControlEvasion:
    """Test whether suppression controls can be evaded."""

    def test_suppression_may_shift_distribution_not_prevent(self):
        """FINDING 3A: Suppression doesn't change the model's underlying intent.

        Even if suppression works perfectly (logits for token 2 become -inf),
        the model's internal state still "wants" to emit something similar.

        This is a more subtle failure mode: the model learns to emit token 3
        instead of token 2 if they're semantically equivalent.

        The suppression_ratio gate passes, but the semantic meaning hasn't changed.
        """
        print("FINDING 3A: Suppression changes P(token), not intent")
        print("  Model learns token2 -> concept X")
        print("  Suppression prevents token 2")
        print("  Model learns token3 -> concept X instead")
        print("  Suppression passes, but concept still expressed")
        print("  Severity: HIGH - the gate doesn't actually constrain behavior")

    def test_side_effect_may_not_detect_distribution_shift(self):
        """FINDING 3B: side_effect measures loss on sequences WITHOUT token 2.

        But if the model has learned a strong association between token 1 -> token 2,
        the control might not harm sequences that never include token 1.

        Conversely, on sequences that DO include token 1, the model's state
        is confused by the suppression, leading to higher loss.

        This is hard to detect because we're averaging over all sequences.
        """
        # This is a deeper issue about measurement: side_effect is computed on
        # sequences that don't CONTAIN token 2, but if the model's learned behavior
        # is token1 -> token2, the suppression only hurts when token1 appears.

        print("FINDING 3B: side_effect metric averages across distributions")
        print("  If token suppression is learned, it only harms sequences with the trigger.")
        print("  This may be unmeasurable on random test distributions.")

    @pytest.mark.xfail(reason="HIGH: Evasion may not be detected by simple metrics")
    def test_model_trained_to_evade_suppression(self):
        """FINDING 3C: A model could be trained to evade the suppression control.

        If we train a model knowing that token 2 will be suppressed at test time,
        the model learns to emit token 3 instead (which has the same semantic value).

        The suppression_ratio gate would still pass (low P(2)), but the control
        did not actually constrain the model's behavior.
        """
        # This requires training, so we'll just document it as a theoretical risk
        print("FINDING 3C: A model trained to evade known suppressions passes all gates")
        print("  The weave author gains false confidence the suppression works")
        print("  Severity: HIGH - the layman's guarantee is false")
        pytest.xfail("This requires adversarial training to demonstrate")

    def test_control_enabled_disabled_toggle(self):
        """Verify that the ControlledModel can enable/disable controls."""
        class SimpleModel(nn.Module):
            def __init__(self, vocab=10, d_model=64, n_layers=2):
                super().__init__()
                self.embed = nn.Linear(1, d_model)
                self.blocks = nn.ModuleList([
                    nn.Linear(d_model, d_model) for _ in range(n_layers)
                ])
                self.head = nn.Linear(d_model, vocab)

            def forward(self, x, collect=False):
                B, L = x.shape
                # Dummy forward: create residual stream
                h = self.embed(torch.ones((B, L, 1), device=x.device))
                residuals = []
                for blk in self.blocks:
                    h = blk(h)
                    residuals.append(h)
                logits = self.head(h)
                if collect:
                    return logits, residuals
                return logits

        model = SimpleModel(vocab=10, d_model=64, n_layers=2)

        # Use a steering vector (so the hook actually gets installed)
        steering_vec = torch.randn(64)
        steering_vec = steering_vec / steering_vec.norm()

        control = ControlRecord(
            name="suppress_2",
            kind="suppress",
            token=2,
            layer=-1,
            mechanism="steering",  # Use steering so the hook gets installed
            steering_vector=steering_vec,
        )

        controlled = ControlledModel(model, [control])
        test_seq = torch.zeros((1, 5), dtype=torch.long)

        # Control enabled
        controlled.enable_control("suppress_2")
        with torch.no_grad():
            logits_suppressed = controlled(test_seq)

        # Control disabled
        controlled.disable_control("suppress_2")
        with torch.no_grad():
            logits_unsuppressed = controlled(test_seq)

        # The logits should be different when control is toggled
        diff = (logits_suppressed - logits_unsuppressed).abs().sum()
        assert diff > 0, "Logits should differ when control is toggled"

        print("FINDING 3D: Control toggling affects model output")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
