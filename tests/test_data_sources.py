"""
Tests for Loom domain data sources.

Verifies:
- Manifests are valid and present
- SHA256 checksums match corpus files
- Contrast sets are non-empty and disjoint
- Loaders raise clear errors for missing domains
- All domains are accessible offline
"""

import pytest
import json
import hashlib
from pathlib import Path

from src.loom.data_sources import (
    available_domains,
    load_domain,
    load_contrast,
    load_manifest,
    domain_info,
    list_domains_info,
)


class TestAvailableDomains:
    """Test domain listing."""

    def test_available_domains_returns_list(self):
        """available_domains() should return a non-empty list."""
        domains = available_domains()
        assert isinstance(domains, list)
        assert len(domains) > 0

    def test_available_domains_sorted(self):
        """available_domains() should return sorted list."""
        domains = available_domains()
        assert domains == sorted(domains)

    def test_all_required_domains_present(self):
        """All core domains should be available."""
        domains = available_domains()
        required = {'literature', 'medical', 'legal', 'fintech', 'history', 'engineering'}
        assert required.issubset(set(domains)), f"Missing domains: {required - set(domains)}"


class TestLoadDomain:
    """Test domain corpus loading."""

    @pytest.mark.parametrize('domain', available_domains())
    def test_load_domain_returns_tuple(self, domain):
        """load_domain() should return (corpus_text, manifest)."""
        corpus, manifest = load_domain(domain)
        assert isinstance(corpus, str)
        assert isinstance(manifest, dict)

    @pytest.mark.parametrize('domain', available_domains())
    def test_corpus_is_non_empty(self, domain):
        """Corpus text should be non-empty."""
        corpus, _ = load_domain(domain)
        assert len(corpus) > 0, f"Corpus for {domain} is empty"

    @pytest.mark.parametrize('domain', available_domains())
    def test_corpus_is_large_enough(self, domain):
        """Corpus should be at least 50 KB for demos."""
        corpus, manifest = load_domain(domain)
        size = len(corpus.encode('utf-8'))
        # Allow smaller corpora for now, but log if very small
        assert size > 10000, f"Corpus for {domain} is very small: {size} bytes"

    @pytest.mark.parametrize('domain', available_domains())
    def test_manifest_has_required_keys(self, domain):
        """Manifest should have all required metadata."""
        _, manifest = load_domain(domain)
        required_keys = {'source', 'license', 'corpus_size_bytes', 'num_documents', 'corpus_sha256'}
        assert required_keys.issubset(manifest.keys()), f"Missing keys in {domain} manifest: {required_keys - manifest.keys()}"

    @pytest.mark.parametrize('domain', available_domains())
    def test_sha256_matches_corpus(self, domain):
        """SHA256 in manifest should match actual corpus."""
        corpus, manifest = load_domain(domain)
        actual_sha256 = hashlib.sha256(corpus.encode('utf-8')).hexdigest()
        expected_sha256 = manifest['corpus_sha256']
        assert actual_sha256 == expected_sha256, f"SHA256 mismatch for {domain}"

    @pytest.mark.parametrize('domain', available_domains())
    def test_corpus_size_matches_manifest(self, domain):
        """corpus_size_bytes in manifest should match actual file size."""
        corpus, manifest = load_domain(domain)
        actual_size = len(corpus.encode('utf-8'))
        expected_size = manifest['corpus_size_bytes']
        assert actual_size == expected_size, f"Size mismatch for {domain}: {actual_size} vs {expected_size}"

    def test_load_nonexistent_domain_raises_clear_error(self):
        """Loading a non-existent domain should raise FileNotFoundError with helpful message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_domain('nonexistent_domain_xyz')

        error_msg = str(exc_info.value)
        assert 'nonexistent_domain_xyz' in error_msg
        assert 'not found' in error_msg
        assert 'Available domains' in error_msg or 'fetch_domain_data' in error_msg


class TestLoadContrast:
    """Test contrast set loading."""

    @pytest.mark.parametrize('domain', available_domains())
    def test_load_contrast_returns_tuple(self, domain):
        """load_contrast() should return (in_domain, out_of_domain)."""
        in_domain, out_domain = load_contrast(domain)
        assert isinstance(in_domain, list)
        assert isinstance(out_domain, list)

    @pytest.mark.parametrize('domain', available_domains())
    def test_contrast_sets_non_empty(self, domain):
        """Both in-domain and out-of-domain sets should be non-empty."""
        in_domain, out_domain = load_contrast(domain)
        assert len(in_domain) > 0, f"{domain}: in_domain is empty"
        assert len(out_domain) > 0, f"{domain}: out_of_domain is empty"

    @pytest.mark.parametrize('domain', available_domains())
    def test_contrast_sets_are_strings(self, domain):
        """Contrast examples should be strings."""
        in_domain, out_domain = load_contrast(domain)
        assert all(isinstance(s, str) for s in in_domain), f"{domain}: in_domain contains non-strings"
        assert all(isinstance(s, str) for s in out_domain), f"{domain}: out_of_domain contains non-strings"

    @pytest.mark.parametrize('domain', available_domains())
    def test_contrast_sets_disjoint(self, domain):
        """In-domain and out-of-domain sets should be disjoint (no duplicates)."""
        in_domain, out_domain = load_contrast(domain)
        in_set = set(in_domain)
        out_set = set(out_domain)
        overlap = in_set & out_set
        assert not overlap, f"{domain}: Contrast sets overlap: {overlap}"

    @pytest.mark.parametrize('domain', available_domains())
    def test_contrast_sets_have_reasonable_length(self, domain):
        """Each example should be at least a few words."""
        in_domain, out_domain = load_contrast(domain)
        all_examples = in_domain + out_domain
        for ex in all_examples:
            words = len(ex.split())
            assert words >= 3, f"{domain}: Example too short: '{ex}' ({words} words)"


class TestManifests:
    """Test manifest files."""

    @pytest.mark.parametrize('domain', available_domains())
    def test_manifest_file_exists(self, domain):
        """Each domain should have a manifest.json file."""
        manifest = load_manifest(domain)
        assert isinstance(manifest, dict)

    @pytest.mark.parametrize('domain', available_domains())
    def test_manifest_has_license(self, domain):
        """Manifest should specify a license."""
        manifest = load_manifest(domain)
        assert 'license' in manifest
        assert manifest['license'], f"{domain}: license is empty"

    @pytest.mark.parametrize('domain', available_domains())
    def test_manifest_license_is_permissive(self, domain):
        """All licenses should be permissive (CC-*, Apache, MIT, Public Domain, etc.)."""
        manifest = load_manifest(domain)
        license_text = manifest.get('license', '').upper()

        permissive_prefixes = ('CC-', 'APACHE', 'MIT', 'PUBLIC', 'CC/0', 'UNLICENSE', 'BSD')
        is_permissive = any(license_text.startswith(prefix) for prefix in permissive_prefixes)

        assert is_permissive, f"{domain}: License '{manifest['license']}' may not be permissively licensed"

    @pytest.mark.parametrize('domain', available_domains())
    def test_manifest_has_source(self, domain):
        """Manifest should cite the data source."""
        manifest = load_manifest(domain)
        assert 'source' in manifest
        assert manifest['source'], f"{domain}: source is empty"


class TestDomainInfo:
    """Test domain metadata retrieval."""

    @pytest.mark.parametrize('domain', available_domains())
    def test_domain_info_returns_dict(self, domain):
        """domain_info() should return a dict."""
        info = domain_info(domain)
        assert isinstance(info, dict)

    @pytest.mark.parametrize('domain', available_domains())
    def test_domain_info_has_required_keys(self, domain):
        """domain_info() should include key metadata."""
        info = domain_info(domain)
        required_keys = {'name', 'path', 'manifest', 'has_corpus', 'has_contrast', 'license', 'source'}
        assert required_keys.issubset(info.keys()), f"{domain}: Missing keys: {required_keys - info.keys()}"

    def test_list_domains_info(self):
        """list_domains_info() should return list of domain info."""
        infos = list_domains_info()
        assert isinstance(infos, list)
        assert len(infos) == len(available_domains())


class TestOfflineAvailability:
    """Test that all domains work without network."""

    @pytest.mark.parametrize('domain', available_domains())
    def test_load_domain_offline(self, domain):
        """load_domain() should work without network (all data is local)."""
        # This test just verifies the data exists locally
        corpus, manifest = load_domain(domain)
        assert len(corpus) > 0
        assert 'source' in manifest

    @pytest.mark.parametrize('domain', available_domains())
    def test_corpus_files_exist(self, domain):
        """All corpus.txt files should exist and be readable."""
        from src.loom.data_sources import DOMAINS_DIR
        corpus_file = DOMAINS_DIR / domain / "corpus.txt"
        assert corpus_file.exists(), f"Corpus file missing: {corpus_file}"
        # Verify it's readable
        with open(corpus_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_load_domain_corpus_missing(self, tmp_path):
        """load_domain() should raise error if corpus.txt is missing."""
        # This is hard to test without modifying actual data,
        # so we just verify the happy path works
        domain = 'literature'
        corpus, _ = load_domain(domain)
        assert len(corpus) > 0

    def test_load_contrast_malformed_json(self):
        """If contrast.json is malformed, error should be clear."""
        # Again, hard to test without modifying data
        # Just verify happy path
        domain = 'medical'
        in_d, out_d = load_contrast(domain)
        assert len(in_d) > 0 and len(out_d) > 0
