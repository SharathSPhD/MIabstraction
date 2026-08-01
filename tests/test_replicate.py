import json
import sys
import types

import pytest

from miabstraction.replicate import DECISIVE_METRIC, replicate


@pytest.fixture
def fake_experiment(monkeypatch, tmp_path):
    """Install a stub experiment module whose metric varies with seed."""
    mod = types.ModuleType("miabstraction.experiments.stub_exp")

    def run(cfg):
        # seed 0,1 support; seed 2 does not
        supports = cfg.seed < 2
        return {"hypothesis": "H2", "supports": supports,
                "final_prefix_score": 0.7 - 0.2 * cfg.seed}

    mod.run = run
    monkeypatch.setitem(sys.modules, "miabstraction.experiments.stub_exp", mod)

    cfg_file = tmp_path / "stub.yaml"
    cfg_file.write_text(
        f"name: stub_exp\nhypothesis: H2\nseed: 0\nout_dir: {tmp_path}\n"
    )
    return str(cfg_file)


def test_metric_registry_covers_all_hypotheses():
    assert set(DECISIVE_METRIC) == {"H1", "H2", "H3", "H4", "H5"}


def test_replicate_reports_spread_and_disagreement(fake_experiment, tmp_path):
    s = replicate(fake_experiment, [0, 1, 2])
    assert s["n_seeds"] == 3
    assert s["n_supporting"] == 2
    assert s["replicates"] is False  # seeds disagree -> not replicated
    assert s["metric_std"] > 0
    written = json.loads(
        (tmp_path / "seeds" / "stub_exp_replication.json").read_text()
    )
    assert written["n_supporting"] == 2


def test_unanimous_support_counts_as_replicated(fake_experiment):
    s = replicate(fake_experiment, [0, 1])
    assert s["n_supporting"] == 2
    assert s["replicates"] is True


def test_invariant_metric_is_flagged(monkeypatch, tmp_path):
    """A metric that never moves across seeds must be flagged, not reported as stable."""
    import sys
    import types

    mod = types.ModuleType("miabstraction.experiments.const_exp")
    mod.run = lambda cfg: {"hypothesis": "H5", "supports": True,
                           "circuit_weight_ratio": 0.2086}  # same for every seed
    monkeypatch.setitem(sys.modules, "miabstraction.experiments.const_exp", mod)
    cfg_file = tmp_path / "const.yaml"
    cfg_file.write_text(f"name: const_exp\nhypothesis: H5\nseed: 0\nout_dir: {tmp_path}\n")

    s = replicate(str(cfg_file), [0, 1, 2])
    assert s["metric_invariant_across_seeds"] is True
    assert "tautological" in s["invariance_warning"]


def test_varying_metric_not_flagged(fake_experiment):
    s = replicate(fake_experiment, [0, 1, 2])
    assert s["metric_invariant_across_seeds"] is False
    assert s["invariance_warning"] is None
