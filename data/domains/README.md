# Domain-Specific Corpora for Loom

This directory contains real open-source specialist-domain data for Loom demos. Each domain includes a corpus of authentic texts, a manifest describing the source and licensing, and contrast sets for feature extraction.

## Overview

| Domain | Documents | Size | License | Usable Offline | Status |
|--------|-----------|------|---------|----------------|--------|
| Literature | 300 | 240 KB | CC-BY-SA-3.0 | Yes | Complete |
| Medical | 150 | 101 KB | CC-BY-SA-3.0 | Yes | Complete |
| Legal | 150 | 139 KB | CC-BY-SA-3.0 | Yes | Complete |
| Finance | 150 | 132 KB | CC-BY-SA-3.0 | Yes | Complete |
| History | 150 | 129 KB | CC-BY-SA-3.0 | Yes | Complete |
| Engineering | 200 | 173 KB | CC-BY-SA-3.0 | Yes | Complete |

## Domains

### Literature
**Source:** Wikipedia articles (WikiText-2 dataset)  
**License:** CC-BY-SA-3.0  
**Documents:** 300 Wikipedia articles  
**Size:** ~240 KB  

**Demonstrates:** Storytelling, narrative structure, creative language, semantic coherence, stylistic consistency.

**Suitable for:** Building models that generate or analyze creative writing, essays, and literary critique.

### Medical
**Source:** Wikipedia biomedical articles (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 150 medical and health-related articles  
**Size:** ~101 KB  

**Demonstrates:** Technical terminology, precision in clinical language, accuracy in health information, structured domain knowledge.

**Suitable for:** Building medical Q&A systems, clinical documentation assistants, health literacy tools.

### Legal
**Source:** Wikipedia law and legal topic articles (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 150 legal and constitutional law articles  
**Size:** ~139 KB  

**Demonstrates:** Complex legal terminology, precise statute interpretation, citation patterns, structural legal reasoning.

**Suitable for:** Building legal research assistants, contract analysis tools, compliance checkers.

### Finance
**Source:** Wikipedia finance and economics articles (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 150 financial, investment, and economics articles  
**Size:** ~132 KB  

**Demonstrates:** Quantitative precision, risk assessment language, market analysis terminology, investment strategies.

**Suitable for:** Building financial advisory assistants, portfolio analysis tools, market sentiment analysis.

### History
**Source:** Wikipedia history articles (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 150 historical and civilizational articles  
**Size:** ~129 KB  

**Demonstrates:** Temporal narrative structures, causal explanations, historical context, primary source integration.

**Suitable for:** Building historical research tools, timeline generators, civilizational analysis systems.

### Engineering
**Source:** Wikipedia CS/engineering articles + technical documentation (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 200 computer science and engineering articles  
**Size:** ~173 KB  

**Demonstrates:** Algorithm analysis, system architecture, technical precision, formal notation, optimization techniques.

**Suitable for:** Building code documentation assistants, algorithm explanation tools, technical tutoring systems.

## File Structure

Each domain has three files:

```
data/domains/<domain>/
├── corpus.txt           # Plain text corpus (NOT in git, regenerable)
├── manifest.json        # Metadata: source, license, checksum, size
└── contrast.json        # Feature extraction sets (in-git)
```

### corpus.txt
Plain text file with documents separated by `\n\n`. Each domain contains 150-300 documents.

**File is listed in .gitignore** because corpora are multi-MB when domains expand. Regenerate from script.

### manifest.json
Metadata for reproducibility and attribution:
```json
{
  "source": "https://huggingface.co/datasets/wikitext",
  "source_note": "Description of what the corpus contains",
  "license": "CC-BY-SA-3.0",
  "retrieved": "2026-08-01T19:35:00",
  "corpus_size_bytes": 239549,
  "num_documents": 300,
  "corpus_sha256": "abc123...",
  "corpus_file": "corpus.txt"
}
```

- `source`: URL or reference to the data source
- `license`: Permissive open license (CC-BY*, Apache 2.0, MIT, etc.)
- `corpus_sha256`: Checksum to verify integrity
- `num_documents`: Count for verification

### contrast.json
In-domain and out-of-domain examples for Loom's contrastive feature extraction:

```json
{
  "in_domain": [
    "Example sentence typical of this domain...",
    "Another authentic example..."
  ],
  "out_of_domain": [
    "Example from a different domain...",
    "Another out-of-domain example..."
  ]
}
```

Contrast sets are small (5-10 pairs each) and committed to git. They guide Loom's control layer extraction during compilation.

## Regenerating Corpora

### From Cache (No Network Required)
All corpora are built from the local WikiText-2 dataset cache at `~/.cache/huggingface/datasets/wikitext/`.

Run:
```bash
cd /home/sharaths/projects/MIabstraction-domain-data
.venv/bin/python scripts/fetch_domain_data.py
```

### From Network (Requires Internet)
If caches are invalidated:
```bash
cd /home/sharaths/projects/MIabstraction-domain-data
HF_DATASETS_CACHE=~/.cache/huggingface/datasets .venv/bin/python scripts/fetch_domain_data.py
```

The script:
1. Loads WikiText-2 from HuggingFace (or cache)
2. Extracts domain-specific articles using keyword filtering
3. Validates and generates `corpus.txt` + `manifest.json`
4. Verifies SHA256 checksums

## Usage in Loom

### Writing a Loom Application

```loom
app MedicalAdvisor {
    knows from "data/domains/medical/corpus.txt";
    knows how to explain diagnoses step by step;
    
    speaks empathetic, precise;
    never discusses non-medical topics;
    
    expects answers("What is hypertension?") mentions "blood pressure";
}

build MedicalAdvisor on "meta-llama/Llama-3.2-1B";
```

### Loading Domains Programmatically

```python
from src.loom.data_sources import load_domain, load_contrast, available_domains

# List available domains
domains = available_domains()
# ['literature', 'medical', 'legal', 'finance', 'history', 'engineering']

# Load a domain corpus
text, manifest = load_domain("medical")
print(f"Loaded {manifest['num_documents']} documents")

# Load contrast sets for feature extraction
in_domain, out_domain = load_contrast("medical")
```

## Attribution and Licensing

All corpora are built from permissively licensed open sources:

- **WikiText-2**: CC-BY-SA-3.0 (Wikipedia content)
  - https://huggingface.co/datasets/wikitext
  - Creative Commons Attribution-ShareAlike 3.0 Unported

When using Loom models built on these corpora, preserve the CC-BY-SA-3.0 license notice in your model documentation.

## Adding New Domains

To add a new domain:

1. **Fetch data** from a permissively licensed open source
2. **Create** `data/domains/<name>/corpus.txt` (min 100 KB for demos)
3. **Write** `data/domains/<name>/manifest.json` with source, license, and SHA256
4. **Add** 5-10 in-domain and out-of-domain examples to `data/domains/<name>/contrast.json`
5. **Update** the loader in `src/loom/data_sources.py`
6. **Document** in this README

**License requirement:** New domains MUST use permissively licensed sources (CC-BY*, Apache 2.0, MIT, Public Domain, etc.). Do NOT add proprietary or restricted data.

## Testing

Run the test suite:
```bash
cd /home/sharaths/projects/MIabstraction-domain-data
pytest tests/test_data_sources.py -v
```

Tests verify:
- All manifests present and valid
- SHA256 checksums match corpus files
- Contrast sets are non-empty and disjoint
- Loaders raise clear errors for missing domains
- All domains are accessible offline

