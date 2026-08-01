import json

from miabstraction.synthesize import likelihood_ratio, update_registry


def _write(tmp_path, rel, data):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_likelihood_ratio_direction():
    assert likelihood_ratio("H1", True) > 1
    assert likelihood_ratio("H1", False) < 1


def test_update_registry_moves_posteriors_both_ways(tmp_path):
    _write(tmp_path, "e1/result.json", {"hypothesis": "H1", "supports": True})
    _write(tmp_path, "e3/result.json", {"hypothesis": "H3", "supports": False})
    reg_path = tmp_path / "hypotheses.json"

    posteriors = update_registry(tmp_path, reg_path)

    assert posteriors["H1"] > 0.5   # supported -> belief rises
    assert posteriors["H3"] < 0.5   # refuted -> belief falls
    assert reg_path.exists()
    saved = json.loads(reg_path.read_text())
    # untouched hypotheses stay at the prior — independence is preserved
    assert saved["hypotheses"]["H2"]["posterior"] == 0.5
