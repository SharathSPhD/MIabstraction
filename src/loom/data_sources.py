"""
Loom domain data sources.

Loads genuine specialist-domain corpora and contrast sets for Loom application demos.
Each domain includes:
- corpus.txt: Plain text corpus of domain documents
- manifest.json: Source, license, checksum, and metadata
- contrast.json: In-domain and out-of-domain examples for feature extraction
"""

import json
from pathlib import Path
from typing import Tuple, List

# Domain directory: data/domains/
DOMAINS_DIR = Path(__file__).parent.parent.parent / "data" / "domains"


def available_domains() -> List[str]:
    """
    List all available domains.

    Returns:
        List of domain names (e.g., ['literature', 'medical', 'legal', ...])

    Raises:
        FileNotFoundError: If domains directory does not exist.
    """
    if not DOMAINS_DIR.exists():
        raise FileNotFoundError(
            f"Domains directory not found: {DOMAINS_DIR}\n"
            f"Run: scripts/fetch_domain_data.py"
        )

    domains = [d.name for d in DOMAINS_DIR.iterdir() if d.is_dir()]
    return sorted(domains)


def _validate_domain_exists(domain: str) -> Path:
    """
    Validate that a domain exists and return its path.

    Args:
        domain: Domain name (e.g., 'medical')

    Returns:
        Path to the domain directory

    Raises:
        FileNotFoundError: If domain does not exist with helpful message.
    """
    domain_path = DOMAINS_DIR / domain

    if not domain_path.exists():
        available = available_domains()
        raise FileNotFoundError(
            f"Domain '{domain}' not found.\n"
            f"Available domains: {available}\n"
            f"To fetch missing domains, run: scripts/fetch_domain_data.py"
        )

    return domain_path


def load_domain(domain: str) -> Tuple[str, dict]:
    """
    Load a domain's corpus and manifest.

    Args:
        domain: Domain name (e.g., 'medical', 'literature', 'legal', ...)

    Returns:
        Tuple of (corpus_text, manifest_dict)
        - corpus_text: Plain text of all documents in the domain
        - manifest_dict: Metadata including source, license, sha256, size, num_documents

    Raises:
        FileNotFoundError: If domain or its files don't exist.
        ValueError: If corpus_sha256 doesn't match the actual file.
    """
    domain_path = _validate_domain_exists(domain)

    corpus_file = domain_path / "corpus.txt"
    manifest_file = domain_path / "manifest.json"

    # Verify corpus exists
    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_file}\n"
            f"Run: scripts/fetch_domain_data.py"
        )

    # Verify manifest exists
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_file}\n"
            f"Each domain must have a manifest.json"
        )

    # Load manifest
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_text = f.read()

    # Verify integrity
    import hashlib
    actual_sha256 = hashlib.sha256(corpus_text.encode('utf-8')).hexdigest()
    expected_sha256 = manifest.get('corpus_sha256')

    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Corpus integrity check failed for '{domain}'.\n"
            f"Expected SHA256: {expected_sha256}\n"
            f"Actual SHA256:   {actual_sha256}\n"
            f"The corpus may have been corrupted or modified."
        )

    return corpus_text, manifest


def load_contrast(domain: str) -> Tuple[List[str], List[str]]:
    """
    Load contrast sets (in-domain and out-of-domain examples) for feature extraction.

    These examples guide Loom's control layer extraction during compilation.
    In-domain examples are authentic to the domain; out-of-domain are from other domains.

    Args:
        domain: Domain name (e.g., 'medical', 'literature', 'legal', ...)

    Returns:
        Tuple of (in_domain_examples, out_of_domain_examples)
        - Each is a list of example sentences/passages

    Raises:
        FileNotFoundError: If domain or contrast.json doesn't exist.
        ValueError: If contrast sets are empty or invalid.
    """
    domain_path = _validate_domain_exists(domain)

    contrast_file = domain_path / "contrast.json"

    if not contrast_file.exists():
        raise FileNotFoundError(
            f"Contrast set not found: {contrast_file}\n"
            f"Each domain must have a contrast.json with 'in_domain' and 'out_of_domain' examples"
        )

    with open(contrast_file, 'r') as f:
        contrast = json.load(f)

    in_domain = contrast.get('in_domain', [])
    out_of_domain = contrast.get('out_of_domain', [])

    # Validate
    if not in_domain:
        raise ValueError(f"Contrast set for '{domain}' has empty 'in_domain' list")
    if not out_of_domain:
        raise ValueError(f"Contrast set for '{domain}' has empty 'out_of_domain' list")

    # Check for disjointness (no duplicates between in and out)
    in_set = set(in_domain)
    out_set = set(out_of_domain)
    overlap = in_set & out_set
    if overlap:
        raise ValueError(
            f"Contrast sets for '{domain}' overlap. "
            f"Examples must be unique between in_domain and out_of_domain: {overlap}"
        )

    return in_domain, out_of_domain


def load_manifest(domain: str) -> dict:
    """
    Load just the manifest metadata for a domain (without loading the full corpus).

    Args:
        domain: Domain name (e.g., 'medical')

    Returns:
        Manifest dict with keys: source, source_note, license, retrieved,
        corpus_size_bytes, num_documents, corpus_sha256, corpus_file

    Raises:
        FileNotFoundError: If domain or manifest doesn't exist.
    """
    domain_path = _validate_domain_exists(domain)

    manifest_file = domain_path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    with open(manifest_file, 'r') as f:
        return json.load(f)


def domain_info(domain: str) -> dict:
    """
    Get comprehensive info about a domain.

    Returns:
        Dict with keys: name, path, manifest, has_corpus, has_contrast, num_documents, size_bytes
    """
    domain_path = _validate_domain_exists(domain)

    manifest = load_manifest(domain)

    corpus_file = domain_path / "corpus.txt"
    contrast_file = domain_path / "contrast.json"

    return {
        "name": domain,
        "path": str(domain_path),
        "manifest": manifest,
        "has_corpus": corpus_file.exists(),
        "has_contrast": contrast_file.exists(),
        "num_documents": manifest.get('num_documents', 0),
        "size_bytes": manifest.get('corpus_size_bytes', 0),
        "license": manifest.get('license'),
        "source": manifest.get('source'),
    }


def list_domains_info() -> List[dict]:
    """
    List info for all available domains.

    Returns:
        List of dicts (one per domain) with keys from domain_info()
    """
    return [domain_info(d) for d in available_domains()]


if __name__ == '__main__':
    # Quick test
    print("Available domains:", available_domains())

    for domain in available_domains():
        info = domain_info(domain)
        print(f"\n{domain}:")
        print(f"  Documents: {info['num_documents']}")
        print(f"  Size: {info['size_bytes']:,} bytes")
        print(f"  License: {info['license']}")
        print(f"  Source: {info['source']}")
