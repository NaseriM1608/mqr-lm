"""Verify transported data artifacts and bind checkpoints to tokenizer semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .data import ByteTokenizer


def file_sha256(path: str | Path) -> str:
    """Hash a file in bounded chunks, including large memory-mapped token files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_tokenizer_sha256() -> str:
    descriptor = {"kind": "utf8_bytes", **asdict(ByteTokenizer())}
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def tokenizer_identity(path_or_none: str | Path | None = None) -> str:
    """Return a semantic byte-tokenizer SHA or the exact BPE artifact SHA.

    The fixed byte mapping has one identity regardless of descriptor formatting.
    A BPE mapping depends on its full artifact, including merges and processing.
    """
    if path_or_none is None:
        return _byte_tokenizer_sha256()
    path = Path(path_or_none)
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid tokenizer JSON: {path}.") from exc
    if not isinstance(descriptor, dict):
        raise ValueError("Tokenizer JSON must contain an object.")
    if descriptor.get("kind") == "utf8_bytes":
        try:
            ByteTokenizer(**{key: descriptor[key] for key in asdict(ByteTokenizer())})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Byte tokenizer descriptor does not match the fixed UTF-8 byte mapping.") from exc
        return _byte_tokenizer_sha256()
    return file_sha256(path)


def raw_byte_identity() -> dict[str, str]:
    return {"kind": "raw_byte", "tokenizer_sha256": _byte_tokenizer_sha256()}


def validate_tokenizer(identity: dict, path: str | Path | None = None) -> None:
    """Reject an absent or incompatible tokenizer identity before interpreting IDs."""
    if not isinstance(identity, dict) or not isinstance(identity.get("tokenizer_sha256"), str):
        raise ValueError("Checkpoint does not declare a tokenizer identity.")
    if tokenizer_identity(path) != identity["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity does not match the checkpoint; vocabulary size alone is insufficient.")


def prepared_identity(data_dir: str | Path) -> dict[str, str]:
    """Verify all advertised files without requiring the original source JSONL.

    Required artifacts are tokenizer.json, train.npy and validation.npy. Hashes
    bind the files to the preparation manifest; they are integrity checks, not a
    signature proving who produced a dataset. External source revisions remain
    documentary metadata and are not fetched or read here.
    """
    root = Path(data_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid preparation manifest: {manifest_path}.") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Preparation manifest must declare an artifact 'files' mapping.")
    files = manifest["files"]
    required = {"tokenizer.json", "train.npy", "validation.npy"}
    if not required.issubset(files):
        missing = ", ".join(sorted(required - files.keys()))
        raise ValueError(f"Preparation manifest is missing required artifacts: {missing}.")
    for name, expected in files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Manifest artifact names must be nonempty relative paths.")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or Path(name).is_absolute():
            raise ValueError(f"Manifest artifact path escapes the prepared directory: {name!r}.")
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise ValueError(f"Manifest artifact has no SHA-256: {name!r}.")
        if not path.is_file():
            raise ValueError(f"Prepared artifact is missing: {name!r}.")
        if "bytes" in expected and path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Prepared artifact size mismatch: {name!r}.")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Prepared artifact SHA-256 mismatch: {name!r}.")
    return {
        "kind": "prepared",
        "tokenizer_sha256": tokenizer_identity(root / "tokenizer.json"),
        "manifest_sha256": file_sha256(manifest_path),
    }
