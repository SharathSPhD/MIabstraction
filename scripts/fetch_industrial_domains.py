"""Build the legal and literature corpora from real third-party sources.

Legal: lex_glue/case_hold (Chalkidis et al., CC BY-SA 4.0) — the CaseHOLD task's
citing-context passages, real US court opinions. Literature: Project Gutenberg prose
(via the BabyLM 2026 strict corpus) plus public-domain poetry (merve/poetry rows with
pre-1900 authors). Both replace wikipedia fallbacks; both manifests say exactly what
was used and hash it. Fintech (finance-alpaca) and engineering (arXiv) already carry
real corpora and are not touched.

Run: .venv/bin/python scripts/fetch_industrial_domains.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LIMIT = 400_000     # bytes of corpus per domain, matching the medical build


def write_domain(name: str, text: str, source: str, note: str, license_: str,
                 n_docs: int, contrast: dict) -> None:
    d = Path(f"data/domains/{name}")
    d.mkdir(parents=True, exist_ok=True)
    text = text[:LIMIT]
    (d / "corpus.txt").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (d / "manifest.json").write_text(json.dumps({
        "source": source, "source_note": note, "license": license_,
        "retrieved": datetime.now(timezone.utc).isoformat(),
        "corpus_size_bytes": len(text.encode()), "num_documents": n_docs,
        "corpus_sha256": sha, "corpus_file": "corpus.txt",
        "is_specialist": True,
    }, indent=2))
    (d / "contrast.json").write_text(json.dumps(contrast, indent=2))
    print(f"{name}: {len(text):,} chars, {n_docs} documents, sha {sha[:12]}")


def legal() -> None:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("coastalcph/lex_glue", "case_hold/train-00000-of-00001.parquet",
                        repo_type="dataset")
    t = pq.read_table(p)
    ctx = t.column("context").to_pylist()
    docs, total = [], 0
    for c in ctx:
        c = " ".join(c.split())
        if len(c) < 200:
            continue
        docs.append(c)
        total += len(c)
        if total > LIMIT:
            break
    write_domain(
        "legal", "\n".join(docs),
        "https://huggingface.co/datasets/coastalcph/lex_glue (case_hold)",
        "CaseHOLD citing contexts from real US court opinions (Zheng et al. 2021, "
        "via LexGLUE)",
        "Apache-2.0 (CaseHOLD release); the opinions are US public records",
        len(docs),
        {"in_domain": [
            "The court held that the contract was unenforceable for lack of "
            "consideration.",
            "Summary judgment is appropriate where no genuine dispute of material "
            "fact exists.",
            "The appellant bears the burden of demonstrating reversible error.",
            "The statute of limitations begins to run when the claim accrues.",
            "A motion to dismiss tests the legal sufficiency of the complaint.",
        ], "out_of_domain": [
            "The recipe calls for two cups of flour and a pinch of salt.",
            "The team won the championship after a dramatic overtime.",
            "Photosynthesis converts sunlight into chemical energy.",
            "The album debuted at number one on the charts.",
            "Regular oil changes extend the life of an engine.",
        ]})


def literature() -> None:
    guten = sorted(Path.home().glob(
        ".cache/huggingface/hub/datasets--BabyLM-community--BabyLM-2026-Strict/"
        "snapshots/*/gutenberg.train.txt"))
    if not guten:
        raise SystemExit("gutenberg.train.txt not in the HF cache")
    prose = guten[0].read_text(errors="ignore")[: LIMIT // 2]

    blobs = sorted(Path.home().glob(
        ".cache/huggingface/hub/datasets--merve--poetry/blobs/*"),
        key=lambda p: p.stat().st_size, reverse=True)
    poems, n_poems = [], 0
    for b in blobs:
        head = b.read_bytes()[:64]
        if not head.startswith(b"author,"):
            continue
        rows = csv.DictReader(io.StringIO(b.read_text(errors="ignore")))
        for r in rows:
            # Public-domain only: the Renaissance-tagged rows (Shakespeare et al.).
            if (r.get("age") or "").strip() != "Renaissance":
                continue
            body = (r.get("content") or "").strip()
            if len(body) < 100:
                continue
            poems.append(f"{r.get('poem name', '').strip()}\n{body}")
            n_poems += 1
        break
    if not poems:
        raise SystemExit("no public-domain poetry rows found in the cached csv")
    text = prose + "\n\n" + "\n\n".join(poems)
    write_domain(
        "literature", text,
        "Project Gutenberg via BabyLM-2026-Strict + "
        "https://huggingface.co/datasets/merve/poetry (Renaissance rows only)",
        "Out-of-copyright prose and poetry; no living author's text included, "
        "which is the point of the Stylist program's guardrail",
        "public domain", n_poems + 1,
        {"in_domain": [
            "The prose carries the scene without ornament, and the rhythm does the "
            "rest.",
            "Her letters read like weather: plain at first, then suddenly turning.",
            "The stanza closes on a half rhyme that keeps the grief unresolved.",
            "The narrator withholds the name until the sentence cannot.",
            "Each chapter opens in the middle of a gesture.",
        ], "out_of_domain": [
            "Quarterly revenue grew eight percent on stronger cloud demand.",
            "Tighten the drain plug to the specified torque.",
            "The defendant moved to suppress the evidence.",
            "Preheat the oven to 220 degrees before baking.",
            "The router assigns addresses over DHCP.",
        ]})


if __name__ == "__main__":
    legal()
    literature()
    sys.exit(0)
