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
        """FIXED: logit_mask now installs a real hook that masks the token."""
        import torch
        from loom.runtime import ControlRecord, ControlledModel
        from miabstraction.models import TinyTransformer
        model = TinyTransformer(vocab=5, d_model=32, n_layers=1, n_heads=2)
        rec = ControlRecord(name="c", kind="suppress", token=2,
                            mechanism="logit_mask", steering_vector=None)
        cm = ControlledModel(model, [rec], device="cpu")
        with torch.no_grad():
            logits = cm(torch.randint(0, 5, (2, 8)))
        assert (logits[..., 2] < -1e30).all()

    def test_steering_vector_none_causes_silent_failure(self):
        """FIXED: a control with no active mechanism is refused, never silent."""
        import pytest as _pytest
        from loom.runtime import ControlRecord, ControlledModel
        from miabstraction.models import TinyTransformer
        model = TinyTransformer(vocab=5, d_model=32, n_layers=1, n_heads=2)
        rec = ControlRecord(name="c", kind="suppress", token=2,
                            mechanism="steering", steering_vector=None)
        with _pytest.raises(RuntimeError):
            ControlledModel(model, [rec], device="cpu")

    def test_fix_logit_mask_mechanism(self):
        """Suggestion: implement logit_mask in ControlledModel."""
        print("FIX: Add logit_mask handling to ControlledModel._install_hooks()")
        print("  Add a separate hook that modifies logits directly")
        print("  Set logits[:, :, token] = -inf before softmax")
        print("  Or implement at the layer level before unembedding")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
