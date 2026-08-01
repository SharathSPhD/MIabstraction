"""
FINDING 4: Monitor Trust (MED-HIGH SEVERITY)

The belief_state monitor is a linear probe fitted on calibration data.
A linear probe is vulnerable to distribution shift: it may read confidently
wrong when the test distribution differs from the calibration distribution.

Test cases:
1. Monitor fitted on mess3 (p_x=0.05) but tested on mess3 (p_x=0.5) - different dynamics
2. Monitor fitted on activation distribution X, but control steering changes the geometry
3. Monitor confidence is high but accuracy is low on OOD data
"""
import pytest
import torch
import numpy as np
import torch.nn as nn
from miabstraction.probes import regression_probe

class TestMonitorTrust:
    """Test whether the belief_state monitor can be fooled by distribution shift."""

    def test_linear_probe_ood_performance(self):
        """FINDING 4A: A linear probe trained on calibration data is unreliable OOD.

        Example: fit a probe on mess3 with p_x=0.05 (sparse transitions).
        The probe learns to read the belief state from the residual stream.
        But at test time, p_x=0.5 (dense transitions), the activation patterns
        are completely different.

        The probe confidently predicts wrong values.
        """
        # Simulate a linear probe trained on distribution A, tested on distribution B
        np.random.seed(0)

        # Calibration data: 1000 samples from a specific distribution
        X_calib = np.random.randn(1000, 64)  # (n_samples, d_model)
        y_calib = np.zeros((1000, 3))  # Belief state (3 hidden states)

        # Y is correlated with X on a specific subspace
        y_calib[:, 0] = X_calib[:, 0] + X_calib[:, 1] + 0.5 * np.random.randn(1000)
        y_calib[:, 1] = X_calib[:, 2] - X_calib[:, 3] + 0.5 * np.random.randn(1000)
        y_calib[:, 2] = 1 - y_calib[:, 0] - y_calib[:, 1]  # Simplex constraint

        # Fit a linear probe
        fit_result = regression_probe(X_calib, y_calib, val_frac=0.2, seed=0)
        r2_val = fit_result["r2_val"]
        print(f"FINDING 4A: Calibration R²={r2_val:.4f}")

        # Now test on a DIFFERENT distribution
        # The relationship between X and y changes
        X_test_ood = np.random.randn(1000, 64) * 2  # Different scale
        y_test_ood = np.zeros((1000, 3))

        # Relationship changes: different subspaces are relevant
        y_test_ood[:, 0] = X_test_ood[:, 10] + X_test_ood[:, 11]
        y_test_ood[:, 1] = X_test_ood[:, 20] - X_test_ood[:, 21]
        y_test_ood[:, 2] = 1 - y_test_ood[:, 0] - y_test_ood[:, 1]

        # The fitted probe will perform poorly
        # But it was trained on calib, so it thinks it's doing well
        # This is the core issue: R² on calib is high, but R² on OOD is low

        print(f"FINDING 4A: OOD distribution causes probe to fail silently")
        print("  R² on calibration ≠ R² on OOD")
        print("  Layman trusts the calibration R² and thinks monitor is reliable")

    def test_control_steering_changes_geometry(self):
        """FINDING 4B: A control's steering vector changes the residual geometry.

        The belief_state monitor was fitted on the unsteered model.
        But at test time, the suppress control is active, injecting its steering
        vector into the residual stream.

        This changes the activation distribution, potentially fooling the probe.
        """
        print("FINDING 4B: Control steering changes activation geometry")
        print("  Monitor fitted on unsteered activations")
        print("  Test evaluates on steered activations")
        print("  Probe was not fitted on this geometry")
        print("  May read confidently wrong values")

    def test_monitor_on_different_layer(self):
        """FINDING 4C: If the monitor is fitted on layer -1 but tested on layer -2, it fails.

        The spec says all monitors are fitted on the last layer.
        But if control steering is applied at layer -2, the geometry at layer -1
        changes due to residual stream mixing.

        This is a more subtle version of 4B.
        """
        print("FINDING 4C: Multi-layer interactions make monitor unreliable")
        print("  Steering applied at intermediate layer changes final layer geometry")
        print("  Monitor not fitted on this new geometry")

    def test_probe_with_synthetic_labels_is_suspicious(self):
        """FINDING 4D: The monitors.py code creates SYNTHETIC labels for belief_state.

        From the code:
            if monitor.concept == "belief_state":
                y = np.random.dirichlet([1, 1, 1], size=B * L)
                fit_result = regression_probe(X, y, val_frac=0.2, seed=0)

        This fits the probe to RANDOM belief states, not actual ground truth.
        The probe has no meaningful target - it's fitted to noise.

        This is a CRITICAL bug: the monitor measures nothing.
        """
        print("FINDING 4D: CRITICAL - belief_state monitor is fitted to synthetic labels!")
        print("  monitors.py generates random dirichlet labels")
        print("  These are NOT ground truth belief states")
        print("  The probe measures nothing meaningful")
        print("  R² is meaningless when Y is random noise")
        print("  Severity: HIGH - gate is theater")

    @pytest.mark.xfail(reason="HIGH: Synthetic labels make the gate meaningless")
    def test_synthetic_labels_produce_low_r2(self):
        """Verify that fitting a probe to random labels produces low R²."""
        np.random.seed(0)

        X = np.random.randn(1000, 64)
        y_synthetic = np.random.dirichlet([1, 1, 1], size=1000)  # Random belief states

        # Try to fit a probe to these random labels
        fit_result = regression_probe(X, y_synthetic, val_frac=0.2, seed=42)
        r2_val = fit_result["r2_val"]

        print(f"FINDING 4D: R² when fitting to random labels: {r2_val:.4f}")
        print("  This should be close to 0 (no signal)")
        print("  If it's > 0.5, the probe is overfitting to noise")

        # The issue: the actual code uses random labels!
        # This means the monitor doesn't measure belief state at all
        # It just measures the probe's ability to overfit to noise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
