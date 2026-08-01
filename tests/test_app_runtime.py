"""The artifact: a built app is a directory you can load and talk to."""
import json

import pytest
import torch

from loom.app.runtime import InstalledControl, LoomModel, load_artifact


class _Block(torch.nn.Module):
    def forward(self, x):
        return x * 1.0


class _Tiny(torch.nn.Module):
    """Minimal stand-in with a `blocks` container, so runtime logic is testable
    without downloading a model."""
    def __init__(self, width=8, n=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block() for _ in range(n))
        self.width = width

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def _control(strength=2.0, width=8, layer=-1):
    return InstalledControl(name="polite", kind="style", layer=layer,
                            strength=strength, direction=[1.0] + [0.0] * (width - 1),
                            side_effect=0.02)


def test_controls_are_reattached_on_load_and_change_the_forward_pass():
    """The compiled behaviour lives in the artifact, not in the caller's memory."""
    m = _Tiny()
    x = torch.zeros(1, 4, 8)
    plain = m(x).clone()

    lm = LoomModel(m, None, [_control()], plan={}, report={}, device="cpu")
    steered = m(x)
    assert not torch.allclose(plain, steered), "a loaded control must actually apply"
    assert steered[..., 0].abs().sum() > 0


def test_detach_restores_the_base_behaviour():
    m = _Tiny()
    x = torch.zeros(1, 4, 8)
    lm = LoomModel(m, None, [_control()], plan={}, report={}, device="cpu")
    lm.detach()
    assert torch.allclose(m(x), torch.zeros_like(x))


def test_zero_strength_control_is_a_no_op():
    m = _Tiny()
    x = torch.randn(1, 4, 8)
    base = m(x).clone()
    LoomModel(m, None, [_control(strength=0.0)], plan={}, report={}, device="cpu")
    torch.testing.assert_close(m(x), base)


def test_describe_reports_the_plan_and_the_active_controls():
    m = _Tiny()
    plan = {"app": "Tutor", "target": "x/y", "substrate": "open_weight",
            "capabilities": [{"capability": "speak patient", "strategy": "steer_style_feature"}]}
    lm = LoomModel(m, None, [_control()], plan=plan, report={}, device="cpu")
    text = lm.describe()
    assert "Tutor" in text and "steer_style_feature" in text
    assert "polite" in text and "style" in text


def test_planned_but_unbuilt_artifact_says_so(tmp_path):
    """A --dry-run artifact must not masquerade as a model."""
    (tmp_path / "plan.json").write_text(json.dumps({"app": "T"}))
    with pytest.raises(FileNotFoundError, match="planned but never built"):
        load_artifact(tmp_path, device="cpu")
