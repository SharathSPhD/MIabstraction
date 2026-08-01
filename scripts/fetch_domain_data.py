#!/usr/bin/env python3
"""
Fetch and prepare specialist-domain data for Loom demos.

This script builds genuine open-source corpora and creates the manifest
and contrast files needed for Loom application development.

Domains:
- literature: Wikipedia articles (300 docs)
- medical: Biomedical texts from Wikipedia
- legal: Legal topic articles from Wikipedia
- finance: Financial and economics texts from Wikipedia
- history: Historical articles from Wikipedia
- engineering: Computer science and technical documentation

All corpora use Wikipedia/WikiText-2 data under CC-BY-SA-3.0 license.

Usage:
    python scripts/fetch_domain_data.py

Output:
    data/domains/<domain>/corpus.txt      (*.txt in .gitignore)
    data/domains/<domain>/manifest.json   (committed to git)
    data/domains/<domain>/contrast.json   (committed to git)
"""

import json
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Tuple


def sha256_file(filepath: str) -> str:
    """Compute SHA256 of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_manifest(
    corpus_path: str,
    source_url: str,
    license_name: str,
    retrieval_date: str,
    num_docs: int,
    description: str
) -> dict:
    """Create a manifest.json for a domain."""
    size = Path(corpus_path).stat().st_size
    manifest = {
        "source": source_url,
        "source_note": description,
        "license": license_name,
        "retrieved": retrieval_date,
        "corpus_size_bytes": size,
        "num_documents": num_docs,
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_file": "corpus.txt"
    }
    return manifest


def fetch_literature() -> Tuple[str, dict]:
    """Fetch literature domain (Wikipedia articles)."""
    from datasets import load_dataset

    print("[1/6] LITERATURE - WikiText")

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    lit_texts = []
    for example in wt:
        text = example.get("text", "").strip()
        if text and len(text) > 200:
            lit_texts.append(text)

    lit_corpus = "data/domains/literature/corpus.txt"
    Path(lit_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(lit_corpus, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(lit_texts[:300]))

    manifest = create_manifest(
        lit_corpus,
        "https://huggingface.co/datasets/wikitext",
        "CC-BY-SA-3.0",
        datetime.now().isoformat(),
        min(300, len(lit_texts)),
        "Wikipedia articles from WikiText-2 dataset"
    )

    print(f"  ✓ {manifest['num_documents']} documents, {manifest['corpus_size_bytes']:,} bytes")
    return lit_corpus, manifest


def fetch_medical() -> Tuple[str, dict]:
    """Fetch medical domain (biomedical Wikipedia articles)."""
    from datasets import load_dataset

    print("[3/6] MEDICAL - Biomedical texts")

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    medical_keywords = ["disease", "medical", "health", "treatment", "symptom", "diagnosis", "therapy", "drug", "medicine"]
    med_texts = []

    for example in wt:
        text = example.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in medical_keywords) and len(text) > 200:
            med_texts.append(text.strip())

    med_corpus = "data/domains/medical/corpus.txt"
    Path(med_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(med_corpus, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(med_texts[:150]))

    manifest = create_manifest(
        med_corpus,
        "https://huggingface.co/datasets/wikitext",
        "CC-BY-SA-3.0",
        datetime.now().isoformat(),
        min(150, len(med_texts)),
        "Biomedical and medical texts extracted from Wikipedia"
    )

    print(f"  ✓ {manifest['num_documents']} documents, {manifest['corpus_size_bytes']:,} bytes")
    return med_corpus, manifest


def fetch_legal() -> Tuple[str, dict]:
    """Fetch legal domain (law and legal topic Wikipedia articles)."""
    from datasets import load_dataset

    print("[4/6] LEGAL - Legal texts")

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    legal_keywords = ["law", "legal", "court", "judge", "statute", "contract", "agreement", "liability", "tort", "jurisdiction"]
    legal_texts = []

    for example in wt:
        text = example.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in legal_keywords) and len(text) > 200:
            legal_texts.append(text.strip())

    legal_corpus = "data/domains/legal/corpus.txt"
    Path(legal_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(legal_corpus, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(legal_texts[:150]))

    manifest = create_manifest(
        legal_corpus,
        "https://huggingface.co/datasets/wikitext",
        "CC-BY-SA-3.0",
        datetime.now().isoformat(),
        min(150, len(legal_texts)),
        "Legal texts extracted from Wikipedia law and legal topic articles"
    )

    print(f"  ✓ {manifest['num_documents']} documents, {manifest['corpus_size_bytes']:,} bytes")
    return legal_corpus, manifest


def fetch_finance() -> Tuple[str, dict]:
    """Fetch finance domain (finance and economics Wikipedia articles)."""
    from datasets import load_dataset

    print("[5/6] FINANCE - Financial texts")

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    finance_keywords = ["finance", "financial", "stock", "market", "investment", "bond", "equity", "asset", "return", "risk", "trading", "portfolio"]
    finance_texts = []

    for example in wt:
        text = example.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in finance_keywords) and len(text) > 200:
            finance_texts.append(text.strip())

    fin_corpus = "data/domains/fintech/corpus.txt"
    Path(fin_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(fin_corpus, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(finance_texts[:150]))

    manifest = create_manifest(
        fin_corpus,
        "https://huggingface.co/datasets/wikitext",
        "CC-BY-SA-3.0",
        datetime.now().isoformat(),
        min(150, len(finance_texts)),
        "Financial and economics texts extracted from Wikipedia"
    )

    print(f"  ✓ {manifest['num_documents']} documents, {manifest['corpus_size_bytes']:,} bytes")
    return fin_corpus, manifest


def fetch_history() -> Tuple[str, dict]:
    """Fetch history domain (historical Wikipedia articles)."""
    from datasets import load_dataset

    print("[6/6] HISTORY - Historical texts")

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    history_keywords = ["history", "historical", "war", "century", "ancient", "medieval", "empire", "revolution", "dynasty", "era", "period"]
    history_texts = []

    for example in wt:
        text = example.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in history_keywords) and len(text) > 200:
            history_texts.append(text.strip())

    hist_corpus = "data/domains/history/corpus.txt"
    Path(hist_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(hist_corpus, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(history_texts[:150]))

    manifest = create_manifest(
        hist_corpus,
        "https://huggingface.co/datasets/wikitext",
        "CC-BY-SA-3.0",
        datetime.now().isoformat(),
        min(150, len(history_texts)),
        "Historical texts extracted from Wikipedia history articles"
    )

    print(f"  ✓ {manifest['num_documents']} documents, {manifest['corpus_size_bytes']:,} bytes")
    return hist_corpus, manifest


def fetch_engineering() -> Tuple[str, dict]:
    """Fetch engineering domain (CS and technical Wikipedia articles)."""
    from datasets import load_dataset

    print("[2/6] ENGINEERING - Engineering and CS texts")

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    cs_keywords = ["algorithm", "computation", "network", "system", "software", "hardware", "programming", "processor", "optimization", "architecture", "data structure"]
    eng_texts = []

    for example in wt:
        text = example.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in cs_keywords) and len(text) > 200:
            eng_texts.append(text.strip())

    # Add some technical documentation-like content
    technical_docs = [
        "API Documentation: RESTful endpoints support JSON and XML serialization formats. Authentication uses OAuth 2.0 with bearer tokens.",
        "Algorithm Analysis: The quicksort algorithm has average-case time complexity of O(n log n) and requires O(log n) auxiliary space.",
        "Network Protocol: TCP/IP stack manages end-to-end communication through three-way handshake and sliding window flow control.",
        "Database Design: Normalization reduces data redundancy by decomposing relations into third normal form.",
        "Compiler Theory: Lexical analysis produces tokens, syntax analysis constructs parse trees, and semantic analysis checks type correctness.",
    ]
    eng_texts.extend(technical_docs)

    eng_corpus = "data/domains/engineering/corpus.txt"
    Path(eng_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(eng_corpus, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(eng_texts[:200]))

    manifest = create_manifest(
        eng_corpus,
        "https://huggingface.co/datasets/wikitext + CS documentation",
        "CC-BY-SA-3.0",
        datetime.now().isoformat(),
        min(200, len(eng_texts)),
        "Engineering and computer science texts from Wikipedia and technical documentation"
    )

    print(f"  ✓ {manifest['num_documents']} documents, {manifest['corpus_size_bytes']:,} bytes")
    return eng_corpus, manifest


def create_contrast_sets() -> None:
    """Create contrast sets for all domains."""
    print("\n[7/6] Creating contrast sets...")

    # Literature
    literature_in = [
        "The moonlight fell upon the water, casting long silver pathways across the lake.",
        "She turned the page slowly, savoring each word of the novel.",
        "The protagonist's journey across the mountains tested her resolve.",
        "Poetry is the art of expressing complex emotions through carefully chosen words.",
        "The narrative unfolds through multiple perspectives and unreliable narrators.",
    ]
    literature_out = [
        "The compiler optimizes register allocation automatically.",
        "Cardiac output is calculated as heart rate times stroke volume.",
        "The market valuation increased by 15% in the last quarter.",
        "Trademark law protects brand identifiers from unauthorized use.",
        "Fermentation converts glucose to ethanol in anaerobic conditions.",
    ]
    contrast = {"in_domain": literature_in, "out_of_domain": literature_out}
    with open("data/domains/literature/contrast.json", 'w') as f:
        json.dump(contrast, f, indent=2)

    # Medical
    medical_in = [
        "Hypertension is a chronic condition characterized by elevated blood pressure.",
        "The diagnosis was confirmed through blood tests and imaging studies.",
        "Treatment options include medication, lifestyle changes, and surgery.",
        "The patient presented with acute symptoms requiring emergency intervention.",
        "Pharmacological therapy is the first-line treatment for this condition.",
    ]
    medical_out = [
        "The algorithm achieves 99.5% accuracy on the test set.",
        "The court ruled that the contract was unenforceable due to fraud.",
        "Stock prices rose 8% following the earnings announcement.",
        "The novel explores themes of identity and belonging.",
        "Renewable energy sources reduce carbon emissions effectively.",
    ]
    contrast = {"in_domain": medical_in, "out_of_domain": medical_out}
    with open("data/domains/medical/contrast.json", 'w') as f:
        json.dump(contrast, f, indent=2)

    # Engineering
    engineering_in = [
        "The computational complexity of this algorithm is O(n log n).",
        "Neural networks consist of layers of interconnected nodes.",
        "GPU acceleration significantly improves training speed for deep learning models.",
        "Parallel processing enables distributed computation across multiple cores.",
        "Machine learning models optimize loss functions through gradient descent.",
    ]
    engineering_out = [
        "Shakespeare's works revolutionized English literature.",
        "The merger increased shareholder value significantly.",
        "Antibiotics are used to treat bacterial infections.",
        "Constitutional law governs the structure of government.",
        "Solar panels convert sunlight into electrical energy.",
    ]
    contrast = {"in_domain": engineering_in, "out_of_domain": engineering_out}
    with open("data/domains/engineering/contrast.json", 'w') as f:
        json.dump(contrast, f, indent=2)

    # Legal
    legal_in = [
        "The plaintiff must establish liability under tort law principles.",
        "Statutory interpretation requires examining legislative intent and plain meaning.",
        "Jurisdiction is determined by the court system and venue rules.",
        "Contract enforceability depends on offer, acceptance, and consideration.",
        "The defendant's motion to dismiss was denied by the appellate court.",
    ]
    legal_out = [
        "The protagonist struggled with internal moral conflicts.",
        "Insulin regulates blood glucose through receptor binding.",
        "The Federal Reserve adjusted interest rates to control inflation.",
        "Quantum entanglement violates classical physics principles.",
        "Photosynthesis converts light energy into chemical energy.",
    ]
    contrast = {"in_domain": legal_in, "out_of_domain": legal_out}
    with open("data/domains/legal/contrast.json", 'w') as f:
        json.dump(contrast, f, indent=2)

    # Finance
    finance_in = [
        "The portfolio returned 12% annually, outperforming the benchmark index.",
        "Diversification reduces portfolio risk through asset allocation.",
        "The derivative contract provides leverage in commodities trading.",
        "Federal Reserve monetary policy affects interest rates and inflation.",
        "Equity valuations are calculated using price-to-earnings multiples.",
    ]
    finance_out = [
        "The novel's narrative technique employs unreliable narrators.",
        "Antibiotic resistance develops through natural selection.",
        "Constitutional amendments require congressional approval.",
        "Machine learning models require extensive data preprocessing.",
        "Historical civilizations rose and fell due to economic factors.",
    ]
    contrast = {"in_domain": finance_in, "out_of_domain": finance_out}
    with open("data/domains/fintech/contrast.json", 'w') as f:
        json.dump(contrast, f, indent=2)

    # History
    history_in = [
        "The fall of the Roman Empire marked the transition to the Medieval period.",
        "The Industrial Revolution transformed agricultural societies into industrial economies.",
        "World War II reshaped geopolitical boundaries and power dynamics globally.",
        "Ancient civilizations developed sophisticated systems of governance and administration.",
        "The Renaissance was characterized by renewed interest in classical learning.",
    ]
    history_out = [
        "Machine learning uses neural networks for pattern recognition.",
        "The court decided that the statute was unconstitutional.",
        "Diabetes is managed through insulin therapy and lifestyle changes.",
        "The stock market collapsed due to speculation and poor regulations.",
        "Photosynthesis is essential for oxygen production in ecosystems.",
    ]
    contrast = {"in_domain": history_in, "out_of_domain": history_out}
    with open("data/domains/history/contrast.json", 'w') as f:
        json.dump(contrast, f, indent=2)

    print("  ✓ All contrast sets created")


def save_manifests(manifests: dict) -> None:
    """Save all manifests to JSON files."""
    for domain, manifest in manifests.items():
        manifest_file = f"data/domains/{domain}/manifest.json"
        Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)


def main():
    """Fetch all domain data."""
    print("=" * 60)
    print("LOOM DOMAIN DATA FETCHER")
    print("=" * 60)
    print()

    manifests = {}

    try:
        # Fetch all domains
        _, manifests['literature'] = fetch_literature()
        _, manifests['engineering'] = fetch_engineering()
        _, manifests['medical'] = fetch_medical()
        _, manifests['legal'] = fetch_legal()
        _, manifests['fintech'] = fetch_finance()
        _, manifests['history'] = fetch_history()

        # Save manifests
        save_manifests(manifests)

        # Create contrast sets
        create_contrast_sets()

        print("\n" + "=" * 60)
        print("SUCCESS - All domains ready!")
        print("=" * 60)
        print("\nTo use in a Loom app:")
        print('  knows from "data/domains/<domain>/corpus.txt";')
        print("\nTo load in Python:")
        print("  from src.loom.data_sources import load_domain, load_contrast")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
