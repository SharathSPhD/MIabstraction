"""Regression tests for the red-team findings that were fixed.

Each test encodes an exploit from REDTEAM.md and asserts Loom now refuses or
behaves correctly. If any of these start failing, a fixed hole has reopened.
"""
import numpy as np
import pytest
import torch

from loom.monitors import fit_monitors
from loom.runtime import ControlRecord, ControlledModel
from loom.spec import WeaveError, parse_gate_expr
from miabstraction.models import TinyTransformer


# ---- FINDING 1A: vacuous gates -------------------------------------------------

@pytest.mark.parametrize("metric,expr", [
    ("prefix_score", ">-1"),      # any score passes
    ("prefix_score", ">0"),       # any nonzero score passes
    ("accuracy", ">1.5"),         # impossible
    ("suppression_ratio", "<2"),  # any ratio passes
    ("icl_loss", "<-0.5"),        # impossible for a non-negative loss
    ("probe_r2", ">nan"),         # non-finite
])
def test_vacuous_gates_refused(metric, expr):
    with pytest.raises(WeaveError):
        parse_gate_expr("t", metric, expr)


def test_real_gates_still_accepted():
    assert parse_gate_expr("t", "prefix_score", ">0.5").threshold == 0.5
    assert parse_gate_expr("t", "icl_loss", "<0.5").op == "<"
    assert parse_gate_expr("t", "side_effect", "<0.1").holds(0.05)


# ---- FINDING 4D: monitors fitted to synthetic labels ---------------------------

def _spec_with_monitor():
    from loom.spec import Monitor, WeaveSpec
    return WeaveSpec(model={}, monitors=[
        Monitor(name="m", kind="probe", concept="belief_state")])


def test_monitor_refuses_without_ground_truth():
    model = TinyTransformer(vocab=3, d_model=32, n_layers=1, n_heads=2)
    toks = torch.randint(0, 3, (4, 8))
    with pytest.raises(ValueError, match="synthetic"):
        fit_monitors(model, _spec_with_monitor(), toks, ground_truth={},
                     device="cpu")


def test_monitor_refuses_misaligned_ground_truth():
    model = TinyTransformer(vocab=3, d_model=32, n_layers=1, n_heads=2)
    toks = torch.randint(0, 3, (4, 8))
    bad = {"belief_state": np.random.dirichlet([1, 1, 1], size=7)}  # wrong rows
    with pytest.raises(ValueError, match="rows"):
        fit_monitors(model, _spec_with_monitor(), toks, bad, device="cpu")


# ---- FINDING 5: silently-disabled controls -------------------------------------

def test_control_without_mechanism_refused():
    model = TinyTransformer(vocab=5, d_model=32, n_layers=1, n_heads=2)
    rec = ControlRecord(name="c", kind="suppress", token=2,
                        mechanism="steering", steering_vector=None)
    with pytest.raises(RuntimeError, match="silently"):
        ControlledModel(model, [rec], device="cpu")


def test_logit_mask_actually_masks():
    model = TinyTransformer(vocab=5, d_model=32, n_layers=1, n_heads=2)
    rec = ControlRecord(name="c", kind="suppress", token=2,
                        mechanism="logit_mask", steering_vector=None)
    cm = ControlledModel(model, [rec], device="cpu")
    toks = torch.randint(0, 5, (3, 10))
    with torch.no_grad():
        logits = cm(toks)
    assert (logits[..., 2] < -1e30).all(), "masked token must be unsampleable"
    cm.disable_control("c")
    with torch.no_grad():
        logits2 = cm(toks)
    assert (logits2[..., 2] > -1e30).all(), "disable must restore the token"
