# Domain-Specific Corpora for Loom

This directory contains genuine open-source data for Loom application demos. Specialist-domain corpora are preferred; when unavailable, Wikipedia fallback is explicitly marked in manifests so demos can honestly disclose their data source.

Each domain includes a corpus of authentic texts, a manifest describing the source and licensing, and contrast sets for feature extraction.

## Overview

| Domain | Docs | Size | License | Source | Offline |
|--------|------|------|---------|--------|---------|
| **Medical** | 300 | 273 KB | CC-BY-4.0 | MedQuAD (specialist) | Yes |
| **Engineering** | 300 | 442 KB | CC-BY-4.0 | arXiv API (specialist) | Yes |
| **Finance** | 300 | 350 KB | CC-BY-4.0 | finance-alpaca (specialist) | Yes |
| *Literature* | 300 | 240 KB | CC-BY-SA | Wikipedia (fallback) | Yes |
| *Legal* | 150 | 135 KB | CC-BY-SA | Wikipedia (fallback) | Yes |
| *History* | 150 | 132 KB | CC-BY-SA | Wikipedia (fallback) | Yes |

**Bold = specialist source. Italics = Wikipedia fallback.** Each manifest carries `is_specialist: true|false`, so models built on non-specialist domains can honestly disclose this.

## Domains

### Medical (Specialist ✓)
**Source:** MedQuAD (Hugging Face `lavita/MedQuAD`)  
**License:** CC-BY-4.0  
**Documents:** 300 medical Q&A pairs  
**Size:** 273 KB  
**Specialist:** Yes

**Demonstrates:** Medical terminology, clinical precision, question-answer structure, evidence-based health information.

**Suitable for:** Building medical Q&A systems, clinical documentation assistants, health literacy tools.

### Engineering (Specialist ✓)
**Source:** arXiv API (Computer Science, EESS categories)  
**License:** CC-BY-4.0  
**Documents:** 300 paper abstracts  
**Size:** 442 KB  
**Specialist:** Yes

**Demonstrates:** Algorithm analysis, formal notation, technical precision, research methodology, system design concepts.

**Suitable for:** Building code documentation assistants, algorithm explanation tools, research summarization systems.

### Finance (Specialist ✓)
**Source:** finance-alpaca (Hugging Face `gbharti/finance-alpaca`)  
**License:** CC-BY-4.0  
**Documents:** 300 financial instruction-answer pairs  
**Size:** 350 KB  
**Specialist:** Yes

**Demonstrates:** Financial terminology, quantitative reasoning, investment concepts, risk assessment.

**Suitable for:** Building financial advisory assistants, investment analysis tools, market terminology models.

### Literature (Wikipedia Fallback)
**Source:** Wikipedia articles (WikiText-2 dataset)  
**License:** CC-BY-SA-3.0  
**Documents:** 300 Wikipedia articles  
**Size:** 240 KB  
**Specialist:** No — attempted Project Gutenberg, unavailable

**Demonstrates:** Narrative structure, creative language, semantic coherence (at general encyclopedia level, not specialist literature).

**Note:** Use for demos only; not suitable for literary analysis systems claiming specialist knowledge.

### Legal (Wikipedia Fallback)
**Source:** Wikipedia law/legal topics (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 150 Wikipedia articles  
**Size:** 135 KB  
**Specialist:** No — attempted pile-of-law and casehold, no longer available (dataset scripts unsupported)

**Demonstrates:** Legal terminology at encyclopedia level (not authentic case law or statutes).

**Note:** Use for demos only; not suitable for legal analysis claiming specialist grounding.

### History (Wikipedia Fallback)
**Source:** Wikipedia history articles (WikiText-2 subset)  
**License:** CC-BY-SA-3.0  
**Documents:** 150 Wikipedia articles  
**Size:** 132 KB  
**Specialist:** No — attempted Wikisource and Chronicling America, require OAuth/specialized APIs

**Demonstrates:** Historical narrative at encyclopedia level (not primary sources).

**Note:** Use for demos only; not suitable for historical analysis systems.

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
Metadata for reproducibility, attribution, and transparency:
```json
{
  "source": "https://huggingface.co/datasets/lavita/MedQuAD",
  "source_note": "Medical Question-Answer pairs from medical literature",
  "license": "CC-BY-4.0",
  "retrieved": "2026-08-01T19:43:16.318309",
  "corpus_size_bytes": 272886,
  "num_documents": 300,
  "corpus_sha256": "a499c9a12c44f40f3bdf6bb6ed83844f5d12bb38b4cbe7c51614dd7f285c592b",
  "corpus_file": "corpus.txt",
  "is_specialist": true,
  "attempted_sources": null
}
```

Required fields:
- `source`: URL or reference to the data source
- `license`: Permissive open license (CC-BY*, Apache 2.0, MIT, CC0-1.0, etc.)
- `corpus_sha256`: Checksum to verify integrity
- `num_documents`: Count for verification
- `is_specialist`: **bool** — True if from specialist source, False if Wikipedia fallback
- `attempted_sources`: (optional) List of attempted specialist sources with errors (if applicable)

**Transparency contract:** When `is_specialist: false`, demos must disclose this in documentation so users know the model is built on general Wikipedia rather than specialist texts. The `attempted_sources` list records why specialist sources were unavailable.

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

