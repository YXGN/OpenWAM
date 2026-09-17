"""Source identity checks for legacy and integrated OpenWAM layouts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

LEGACY_REPOSITORY = "OpenWAM-Official/OpenWAM"
LEGACY_REVISION = "c171854a18a2acbeaadf46a746c4118c0e0c00f7"
INTEGRATED_REPOSITORY = "YXGN/OpenWAM"
INTEGRATED_BASE_REVISION = "059cbfbfc068355f3d60fe9b014ff6469d1e94a3"
INTEGRATED_INVENTORY = Path("velabot/source_files.json")
INTEGRATED_PATCH = Path("velabot/patches/openwam-moon-cake-training.patch")

CORE_SOURCE_FILES = (
    "configs/train.yaml",
    "configs/dataloader/moon_cake.yaml",
    "openwam/dataloader/moon_cake.py",
    "openwam/dataloader/moon_cake_safety.py",
    "openwam/dataloader/registry.py",
    "openwam/deploy/delta_addback.py",
    "openwam/deploy/engine.py",
    "openwam/train/openwam_trainer.py",
    "openwam/train/utils/bounded_optimizer_groups.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=check,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"source fingerprint git command failed in {repo_root}: {' '.join(args)}") from exc


def _nul_paths(data: bytes) -> set[str]:
    return {item.decode("utf-8") for item in data.split(b"\0") if item}


def _changed_paths(repo_root: Path, base_revision: str) -> set[str]:
    committed = _git(repo_root, "diff", "--name-only", "-z", "--no-renames", base_revision, "--").stdout
    untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z").stdout
    return _nul_paths(committed) | _nul_paths(untracked)


def _validate_relative(relative: str, *, label: str = "source identity") -> str:
    if not isinstance(relative, str) or not relative or relative == "." or "\\" in relative:
        raise ValueError(f"{label} path is not canonical repository-relative: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or path.as_posix() != relative or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} path is not canonical repository-relative: {relative!r}")
    return relative


def _inside(repo_root: Path, relative: str) -> Path:
    try:
        _validate_relative(relative)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    path = repo_root / relative
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"source identity path escapes repository: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"source identity file is missing: {path}")
    return path


def load_inventory(repo_root: Path) -> tuple[str, ...]:
    inventory_path = _inside(repo_root, INTEGRATED_INVENTORY.as_posix())
    try:
        value = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read source inventory: {inventory_path}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError("velabot/source_files.json must be a non-empty string array")
    paths = tuple(value)
    if len(set(paths)) != len(paths) or tuple(sorted(paths)) != paths:
        raise RuntimeError("velabot/source_files.json must be sorted and duplicate-free")
    for relative in paths:
        try:
            _validate_relative(relative, label="source inventory")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    return paths


def _validate_common(source: dict[str, Any]) -> dict[str, Any]:
    allowed = {"repository", "base_revision", "revision", "patch_sha256", "files_sha256"}
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError(f"source has unknown fields: {unknown}")
    if "patch_sha256" not in source or "files_sha256" not in source:
        raise ValueError("source requires patch_sha256 and files_sha256")
    for key in ("repository", "revision"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"source.{key} must be a non-empty string")
    patch_sha256 = source.get("patch_sha256")
    if patch_sha256 is not None and (
        not isinstance(patch_sha256, str)
        or len(patch_sha256) != 64
        or patch_sha256 != patch_sha256.lower()
        or any(char not in "0123456789abcdef" for char in patch_sha256)
    ):
        raise ValueError("source.patch_sha256 must be null or a lowercase SHA-256 hex digest")
    files = source.get("files_sha256")
    if not isinstance(files, dict):
        raise ValueError("source.files_sha256 must be an object")
    for relative, digest in files.items():
        if (
            not isinstance(relative, str)
            or _invalid_relative(relative)
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("source.files_sha256 contains an invalid relative path or digest")
    return {
        "repository": source["repository"],
        **({"base_revision": source["base_revision"]} if "base_revision" in source else {}),
        "revision": source["revision"],
        "patch_sha256": patch_sha256,
        "files_sha256": dict(files),
    }


def _invalid_relative(relative: str) -> bool:
    try:
        _validate_relative(relative, label="source.files_sha256")
    except ValueError:
        return True
    return False


def normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    """Validate a legacy or integrated source object without inspecting a checkout."""

    normalized = _validate_common(source)
    revision = normalized["revision"]
    if len(revision) != 40 or revision != revision.lower() or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("source.revision must be a lowercase 40-character commit")
    if normalized["repository"] == LEGACY_REPOSITORY:
        if "base_revision" in normalized:
            raise ValueError("legacy source must not declare base_revision")
        if normalized["revision"] != LEGACY_REVISION:
            raise ValueError(f"legacy source revision must be {LEGACY_REVISION}")
    elif normalized["repository"] == INTEGRATED_REPOSITORY:
        if normalized.get("base_revision") != INTEGRATED_BASE_REVISION:
            raise ValueError(f"integrated source base_revision must be {INTEGRATED_BASE_REVISION}")
        if not normalized["patch_sha256"] or not normalized["files_sha256"]:
            raise ValueError("integrated source requires patch_sha256 and files_sha256")
    else:
        raise ValueError("source repository must be OpenWAM-Official/OpenWAM or YXGN/OpenWAM")
    return normalized


def fingerprint_integrated(repo_root: Path) -> dict[str, Any]:
    """Create an integrated identity from the checkout and explicit inventory."""

    revision = _git(repo_root, "rev-parse", "HEAD").stdout.decode().strip()
    if not revision or len(revision) != 40:
        raise RuntimeError(f"integrated checkout HEAD is not a full revision: {revision!r}")
    ancestry = _git(repo_root, "merge-base", "--is-ancestor", INTEGRATED_BASE_REVISION, revision, check=False)
    if ancestry.returncode:
        raise RuntimeError(f"integrated checkout must descend from {INTEGRATED_BASE_REVISION}")
    inventory = load_inventory(repo_root)
    expected = set(inventory)
    required = set(CORE_SOURCE_FILES) | {INTEGRATED_INVENTORY.as_posix(), INTEGRATED_PATCH.as_posix()}
    if not required <= expected:
        raise RuntimeError(f"source inventory is missing required paths: {sorted(required - expected)}")
    changed = _changed_paths(repo_root, INTEGRATED_BASE_REVISION)
    unknown = sorted(changed - expected)
    if unknown:
        raise RuntimeError(f"integrated checkout has unregistered changes: {unknown}")
    files_sha256 = {relative: sha256_file(_inside(repo_root, relative)) for relative in inventory}
    patch_path = _inside(repo_root, INTEGRATED_PATCH.as_posix())
    reverse = _git(repo_root, "apply", "--reverse", "--check", str(patch_path), check=False)
    if reverse.returncode:
        raise RuntimeError("integrated training patch is not applied; reverse check failed")
    return {
        "repository": INTEGRATED_REPOSITORY,
        "base_revision": INTEGRATED_BASE_REVISION,
        "revision": revision,
        "patch_sha256": sha256_file(patch_path),
        "files_sha256": files_sha256,
    }


def verify_integrated(repo_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Verify an integrated manifest against the current checkout."""

    normalized = normalize_source(source)
    if normalized["repository"] != INTEGRATED_REPOSITORY:
        raise ValueError("integrated verification requires source.repository=YXGN/OpenWAM")
    actual = fingerprint_integrated(repo_root)
    if actual["revision"] != normalized["revision"]:
        raise RuntimeError(
            f"integrated checkout revision differs: expected {normalized['revision']}, got {actual['revision']}"
        )
    if actual["patch_sha256"] != normalized["patch_sha256"]:
        raise RuntimeError("integrated patch SHA-256 differs from source identity")
    if actual["files_sha256"] != normalized["files_sha256"]:
        raise RuntimeError("integrated source file hashes differ from source identity")
    return normalized


def fingerprint_legacy(repo_root: Path, patch_path: Path, source_files: tuple[str, ...]) -> dict[str, Any]:
    """Preserve the external c171 patch identity used by the legacy layout."""

    revision = _git(repo_root, "rev-parse", "HEAD").stdout.decode().strip()
    if revision != LEGACY_REVISION:
        raise RuntimeError(f"legacy checkout must be {LEGACY_REVISION}, got {revision}")
    reverse = _git(repo_root, "apply", "--reverse", "--check", str(patch_path), check=False)
    if reverse.returncode:
        raise RuntimeError("legacy training patch is not applied; reverse check failed")
    expected = set(source_files)
    changed = _changed_paths(repo_root, LEGACY_REVISION)
    if changed != expected:
        raise RuntimeError(
            f"legacy checkout has unregistered changes: expected {sorted(expected)}, got {sorted(changed)}"
        )
    return {
        "repository": LEGACY_REPOSITORY,
        "revision": revision,
        "patch_sha256": sha256_file(patch_path),
        "files_sha256": {relative: sha256_file(_inside(repo_root, relative)) for relative in source_files},
    }


def fingerprint_source(repo_root: Path, patch_path: Path, legacy_files: tuple[str, ...]) -> dict[str, Any]:
    """Fingerprint either the integrated repo or the legacy external checkout."""

    if (repo_root / INTEGRATED_INVENTORY).is_file():
        return fingerprint_integrated(repo_root)
    return fingerprint_legacy(repo_root, patch_path, legacy_files)
