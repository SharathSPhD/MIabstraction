import json

from miabstraction.report import render, replication_status


def _write(tmp_path, rel, data):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_replication_status_collected(tmp_path):
    _write(tmp_path, "seeds/e1_replication.json",
           {"hypothesis": "H1", "metric": "best_incremental_r2",
            "metric_mean": 0.0319, "metric_std": 0.0002,
            "n_supporting": 2, "n_seeds": 2, "replicates": True})
    reps = replication_status(tmp_path)
    assert reps["H1"]["n_supporting"] == 2


def test_render_shows_replication_and_single_seed(tmp_path):
    _write(tmp_path, "e1/result.json", {"hypothesis": "H1", "supports": True})
    _write(tmp_path, "e5/result.json", {"hypothesis": "H5", "supports": True})
    _write(tmp_path, "seeds/e1_replication.json",
           {"hypothesis": "H1", "metric": "best_incremental_r2",
            "metric_mean": 0.0319, "metric_std": 0.0002,
            "n_supporting": 2, "n_seeds": 2, "replicates": True})

    md = render(tmp_path)
    assert "2/2 seeds" in md          # replicated hypothesis reports its seeds
    assert "single seed" in md        # unreplicated one is called out, not hidden
