from __future__ import annotations

from pathlib import Path

from openwam_front import _source_identity
from source_identity import (
    INTEGRATED_BASE_REVISION,
    INTEGRATED_REPOSITORY,
    LEGACY_REPOSITORY,
    LEGACY_REVISION,
    fingerprint_integrated,
    normalize_source,
)


def test_integrated_identity_covers_applied_patch_and_inventory():
    identity = fingerprint_integrated(Path(__file__).resolve().parents[2])
    assert identity["repository"] == INTEGRATED_REPOSITORY
    assert identity["base_revision"] == INTEGRATED_BASE_REVISION
    assert len(identity["revision"]) == 40
    assert identity["patch_sha256"]
    assert "velabot/source_files.json" in identity["files_sha256"]


def test_legacy_clean_source_keeps_optional_patch_identity():
    source = normalize_source(
        {
            "repository": LEGACY_REPOSITORY,
            "revision": LEGACY_REVISION,
            "patch_sha256": None,
            "files_sha256": {},
        }
    )
    assert "base_revision" not in source
    assert source["patch_sha256"] is None
    assert source["files_sha256"] == {}


def test_front_round_trips_integrated_identity_fields():
    source = fingerprint_integrated(Path(__file__).resolve().parents[2])
    normalized = _source_identity(source)
    assert normalized["repository"] == INTEGRATED_REPOSITORY
    assert normalized["base_revision"] == INTEGRATED_BASE_REVISION
    assert normalized["revision"] == source["revision"]
