"""Build the grammar corpus the Tutor program teaches from.

Real third-party material: 29,700 Sanskrit verb derivations (dhātu, lakāra, prayoga,
puruṣa, vacana → surface form) from a published morphology dataset, rendered as the
sentences a grammar tutor would actually say. The point is that the Tutor program
declares *what the model must know* and the compiler puts it in the weights — so the
corpus has to be big enough to learn from, not a page of notes.

Run: .venv/bin/python scripts/fetch_grammar_domain.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LIMIT = 400_000
SRC = ("/home/sharaths/.cache/huggingface/hub/"
       "datasets--preetammukherjee--sanskrit_morph_prakriya/snapshots/"
       "ad4f3685554951927fbb181a85f0019682bf7e8c/default/train/0000.parquet")

FIELDS = {"Dhātu": "root", "Lakāra": "tense", "Prayoga": "voice",
          "Puruṣa": "person", "Vacana": "number"}


def sentence(spec: str, form: str) -> str:
    """One derivation, said the way a tutor says it."""
    parts = {}
    for chunk in spec.split(","):
        if ":" not in chunk:
            continue
        k, v = (x.strip() for x in chunk.split(":", 1))
        parts[FIELDS.get(k, k)] = v
    root = parts.get("root", "?")
    return (f"The root {root} in the {parts.get('tense','?')} tense, "
            f"{parts.get('voice','?')} voice, {parts.get('person','?')} person "
            f"{parts.get('number','?')} number, gives the form {form}.")


def main() -> int:
    import pyarrow.parquet as pq
    t = pq.read_table(SRC)
    specs = t.column("llm_input").to_pylist()
    forms = t.column("surface_form_vidyut").to_pylist()

    lines, total = [], 0
    for spec, form in zip(specs, forms):
        s = sentence(spec, form)
        lines.append(s)
        total += len(s) + 1
        if total > LIMIT:
            break

    text = "\n".join(lines)[:LIMIT]
    d = Path("data/domains/grammar")
    d.mkdir(parents=True, exist_ok=True)
    (d / "corpus.txt").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (d / "manifest.json").write_text(json.dumps({
        "source": "https://huggingface.co/datasets/preetammukherjee/"
                  "sanskrit_morph_prakriya",
        "source_note": "Sanskrit verb derivations (Pāṇinian prakriyā), rendered as "
                       "tutor sentences; 29,700 rows available, truncated to the "
                       "corpus budget",
        "license": "CC-BY-4.0",
        "retrieved": datetime.now(timezone.utc).isoformat(),
        "corpus_size_bytes": len(text.encode()),
        "num_documents": len(lines),
        "corpus_sha256": sha,
        "corpus_file": "corpus.txt",
        "is_specialist": True,
    }, indent=2))
    (d / "contrast.json").write_text(json.dumps({
        "in_domain": [
            "The root gam in the present tense gives the form gacchati.",
            "A kāraka is the role a noun plays in relation to the action.",
            "The instrumental case marks the karaṇa kāraka.",
            "Sandhi joins the final sound of one word to the initial of the next.",
            "The optative mood expresses what should or might happen.",
        ],
        "out_of_domain": [
            "Quarterly revenue grew eight percent on stronger cloud demand.",
            "Preheat the oven to 220 degrees before baking.",
            "The defendant moved to suppress the evidence.",
            "Tighten the drain plug to the specified torque.",
            "The patient presented with acute symptoms.",
        ]}, indent=2))
    print(f"grammar: {len(text):,} chars, {len(lines):,} derivations, sha {sha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
