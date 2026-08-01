"""
FINDING 5: Unimplemented Fallback Mechanism (MED SEVERITY)

The runtime.py code specifies that controls can use either:
1. mechanism="steering" (via steering_vector injection)
2. mechanism="logit_mask" (fallback if steering_vector computation fails)

However, ControlledModel._install_hooks() only handles mechanism=="steering".
The "logit_mask" mechanism is never actually applied.

This means: if CAA steering_vector computation fails, the control is silently
disabled. The build passes with no steering control active, even though the
weave declared one.

Test case:
- Compute steering vector fails (return None)
- Mechanism falls back to "logit_mask"
- ControlledModel doesn't install any hook (mechanism != "steering")
- Control is inactive, but no error is raised
"""
import pytest
import torch
import torch.nn as nn
from loom.runtime import ControlRecord, ControlledModel

class TestUnimplementedFallback:
    """Test that the logit_mask fallback mechanism is not implemented."""

    def test_logit_mask_mechanism_not_implemented(self):
        """FINDING 5A: mechanism='logit_mask' is not handled by ControlledModel."""
        class SimpleModel(nn.Module):
            def __init__(self, vocab=10):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(vocab))
                self.logits.data[2] = 5.0

            def forward(self, x, collect=False):
                B, L = x.shape
                logits = self.logits.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
                if collect:
                    return logits, []
                return logits

        model = SimpleModel(vocab=10)

        # Create a control with logit_mask mechanism (the fallback)
        control = ControlRecord(
            name="suppress_2",
            kind="suppress",
            token=2,
            mechanism="logit_mask",  # This is the fallback!
            steering_vector=None,  # No steering vector
        )

        controlled = ControlledModel(model, [control])

        # The ControlledModel should NOT have installed any hooks
        # because the mechanism is "logit_mask", not "steering"
        test_seq = torch.zeros((1, 5), dtype=torch.long)

        with torch.no_grad():
            logits = controlled(test_seq)

        probs = torch.softmax(logits, dim=-1)
        p_token_2 = probs[0, 0, 2].item()

        # With logit_mask, token 2 should be impossible (prob = 0)
        # But since it's not implemented, token 2 still has high probability
        print(f"FINDING 5A: With logit_mask mechanism, P(token=2) = {p_token_2:.4f}")

        if p_token_2 > 0.4:
            print("  BUG: logit_mask is not implemented, control is inactive!")
            print("  The weave declares a suppress control, but it doesn't work")
            print("  Severity: MED - silent failure of safety mechanism")

    def test_steering_vector_none_causes_silent_failure(self):
        """FINDING 5B: When steering_vector computation returns None, the control is disabled.

        From runtime.py:
            if steering_vec is not None:
                mechanism = "steering"
            else:
                mechanism = "logit_mask"  # Fallback - but not implemented!
                steering_vec = None

        If the fallback mechanism isn't implemented, the control is silently disabled.
        """
        class SimpleModel(nn.Module):
            def __init__(self, vocab=10):
                super().__init__()
                self.logits = nn.Parameter(torch.ones(vocab) * 0.5)
                self.logits.data[2] = 5.0

            def forward(self, x, collect=False):
                B, L = x.shape
                logits = self.logits.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
                if collect:
                    return logits, []
                return logits

        model = SimpleModel(vocab=10)

        # Simulate the case where steering_vec computation would fail
        # In practice, this happens when CAA can't find enough positive/negative samples
        control = ControlRecord(
            name="suppress_2",
            kind="suppress",
            token=2,
            layer=-1,
            mechanism="logit_mask",  # Fallback kicked in
            steering_vector=None,  # No steering vector available
        )

        controlled = ControlledModel(model, [control])

        # The control should suppress token 2, but it won't because
        # logit_mask is not implemented
        test_seq = torch.zeros((1, 10), dtype=torch.long)
        controlled.enable_control("suppress_2")

        with torch.no_grad():
            logits = controlled(test_seq)

        probs = torch.softmax(logits, dim=-1)

        # If logit_mask were implemented, P(2) would be 0
        # But since it's not, P(2) is still high
        p_2 = probs[0, 0, 2].item()

        print(f"FINDING 5B: Control fails silently, P(2)={p_2:.4f} instead of 0")
        print("  The weave author thinks the control is active, but it's not")
        print("  This is a gate bypass vulnerability")

    @pytest.mark.xfail(reason="MED: Logit mask fallback not implemented")
    def test_fix_logit_mask_mechanism(self):
        """Suggestion: implement logit_mask in ControlledModel."""
        print("FIX: Add logit_mask handling to ControlledModel._install_hooks()")
        print("  Add a separate hook that modifies logits directly")
        print("  Set logits[:, :, token] = -inf before softmax")
        print("  Or implement at the layer level before unembedding")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
