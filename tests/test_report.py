import json

from miabstraction.report import collect, render, verdict_table


def _write(tmp_path, rel, data):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


def test_collect_and_verdicts(tmp_path):
    _write(tmp_path, "e1/result.json",
           {"hypothesis": "H1", "supports": False, "best_r2": 0.5})
    _write(tmp_path, "strong/e1/result.json",
           {"hypothesis": "H1", "supports": True, "best_r2": 0.9,
            "leak_budget": 0.1})
    _write(tmp_path, "e2/result.json",
           {"hypothesis": "H2", "supports": True})

    results = collect(tmp_path)
    assert len(results) == 3
    by_h = verdict_table(results)
    assert by_h["H1"]["supports"] is True  # strong run preferred
    assert by_h["H2"]["supports"] is True


def test_render_includes_pending(tmp_path):
    _write(tmp_path, "e1/result.json",
           {"hypothesis": "H1", "supports": True, "best_r2": 0.9})
    md = render(tmp_path)
    assert "H5" in md and "pending" in md
    assert "✅" in md
