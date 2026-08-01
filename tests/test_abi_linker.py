import numpy as np
import pytest
import torch

from loom.abi import (
    ABIError,
    Envelope,
    Mode,
    ReadKind,
    Unit,
    WriteKind,
    allocate_frames,
    check_no_clobber,
)
from loom.linker import LinkedModel, host_loss, in_band_cost, link
from miabstraction.models import TinyTransformer


def _unit(name, mode=Mode.OUT_OF_BAND, dims=None, core=None):
    return Unit(
        name=name,
        core=core or TinyTransformer(vocab=8, d_model=32, n_layers=1, n_heads=2),
        read=ReadKind.TOKENS,
        write=WriteKind.LOGITS,
        mode=mode,
        reserved_dims=dims,
        envelope=Envelope(vocab=8, max_len=64),
    )


# ---- ABI guarantees -----------------------------------------------------------

def test_out_of_band_units_never_collide():
    """The point of the convention: private address spaces cannot overlap."""
    check_no_clobber([_unit("a"), _unit("b"), _unit("c")])  # must not raise


def test_in_band_overlap_refused():
    a = _unit("a", Mode.IN_BAND, (0, 16))
    b = _unit("b", Mode.IN_BAND, (8, 24))
    with pytest.raises(ABIError, match="overlapping"):
        check_no_clobber([a, b])


def test_in_band_without_declared_frame_refused():
    with pytest.raises(ABIError, match="reserved_dims"):
        check_no_clobber([_unit("a", Mode.IN_BAND, None)])


def test_frame_allocator_assigns_disjoint_frames():
    units = [_unit("a", Mode.IN_BAND), _unit("b", Mode.IN_BAND)]
    allocate_frames(units, host_width=64, widths={"a": 16, "b": 16})
    assert units[0].reserved_dims == (0, 16)
    assert units[1].reserved_dims == (16, 32)


def test_frame_allocator_refuses_when_out_of_capacity():
    units = [_unit("a", Mode.IN_BAND), _unit("b", Mode.IN_BAND)]
    with pytest.raises(ABIError, match="out of in-band capacity"):
        allocate_frames(units, host_width=20, widths={"a": 16, "b": 16})


def test_capacity_cost_differs_by_mode():
    """The whole TRIZ claim in one assertion: out-of-band logit writes cost 0 host dims."""
    oob = _unit("oob")
    ib = _unit("ib", Mode.IN_BAND, (0, 24))
    assert oob.write_rank(64) == 0
    assert ib.write_rank(64) == 24


def test_envelope_flags_out_of_range_use():
    env = Envelope(vocab=8, max_len=32)
    assert env.check(8, 32) == []
    assert any("vocab" in w for w in env.check(50, 32))
    assert any("length" in w for w in env.check(8, 99))


# ---- linker mechanics ---------------------------------------------------------

@pytest.fixture
def host_and_data():
    torch.manual_seed(0)
    host = TinyTransformer(vocab=8, d_model=32, n_layers=2, n_heads=2, max_len=64)
    toks = torch.randint(0, 8, (32, 24))
    return host, toks


def test_linked_model_preserves_shape_and_can_be_disabled(host_and_data):
    host, toks = host_and_data
    u = _unit("u")
    u.gain = 0.0
    lm = LinkedModel(host, [u], device="cpu")
    with torch.no_grad():
        base = host(toks[:, :-1])
        out = lm(toks[:, :-1])
    assert out.shape == base.shape
    torch.testing.assert_close(out, base)  # zero gain must be a no-op


def test_gain_zero_is_identity_but_gain_nonzero_changes_output(host_and_data):
    host, toks = host_and_data
    u = _unit("u")
    u.gain = 1.0
    lm = LinkedModel(host, [u], device="cpu")
    with torch.no_grad():
        assert not torch.allclose(lm(toks[:, :-1]), host(toks[:, :-1]))


def test_in_band_reservation_costs_the_host(host_and_data):
    """Reserving host dimensions must measurably degrade the host — the price
    out-of-band linking exists to avoid."""
    host, toks = host_and_data
    base = host_loss(host, toks, "cpu")
    reserved = in_band_cost(host, toks, "cpu", n_dims=16)
    assert reserved > base


def test_link_refuses_when_unit_cannot_pass(host_and_data):
    host, toks = host_and_data
    u = _unit("u")
    lm, report = link(host, [u], toks, {"u": lambda m: {"passed": False}},
                      device="cpu", budget=0.05)
    assert lm is None and not report.linked
    assert "Link refused" in report.diagnosis


def test_link_succeeds_and_reports_evidence(host_and_data):
    host, toks = host_and_data
    u = _unit("u")
    lm, report = link(host, [u], toks, {"u": lambda m: {"passed": True, "score": 1.0}},
                      device="cpu", budget=10.0)  # generous budget: mechanics test
    assert lm is not None and report.linked
    assert report.gains["u"] > 0
    assert report.capacity_cost["u"] == 0
    assert report.host_loss_before > 0


def test_gates_are_verified_in_the_composed_model(host_and_data):
    """A unit that passes alone but fails alongside another must not be certified.

    Units write to a shared output and interfere — measured at -0.14 accuracy for one
    unit when a second was linked beside it. Verifying only the solo trials would
    certify a composition that does not work.
    """
    host, toks = host_and_data
    a, b = _unit("a"), _unit("b")
    calls = {"n": 0}

    def gate_a(m):
        # passes while solving alone, fails once both units are installed
        calls["n"] += 1
        composed = len(getattr(m, "units", [])) > 1
        return {"passed": not composed, "score": 0.4 if composed else 0.9}

    lm, report = link(host, [a, b], toks,
                      {"a": gate_a, "b": lambda m: {"passed": True}},
                      device="cpu", budget=50.0)
    assert lm is None and not report.linked
    assert "interfere" in report.diagnosis
