import hashlib
import json

import numpy as np
import pytest

from mqr.data import ByteTokenizer
from mqr.prepare import prepare_jsonl
from mqr.provenance import (
    file_sha256,
    prepared_identity,
    raw_byte_identity,
    tokenizer_identity,
    validate_tokenizer,
)


def make_bundle(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [
        {"id": "train-1", "language": "en", "split": "train", "text": "Training text uses an independent document."},
        {"id": "validation-1", "language": "fa", "split": "validation", "text": "این متن جداگانه برای ارزیابی است."},
    ]
    source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    output = tmp_path / "prepared"
    prepare_jsonl(source, output)
    return source, output


def test_prepared_bundle_verifies_after_source_jsonl_is_removed(tmp_path):
    source, output = make_bundle(tmp_path)
    expected = prepared_identity(output)
    source.unlink()
    assert prepared_identity(output) == expected
    assert expected["tokenizer_sha256"] == raw_byte_identity()["tokenizer_sha256"]
    assert expected["manifest_sha256"] == file_sha256(output / "manifest.json")
    validate_tokenizer(expected, output / "tokenizer.json")
    validate_tokenizer(expected, None)


def test_byte_identity_is_independent_of_descriptor_whitespace_and_key_order(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    ByteTokenizer().save(first)
    descriptor = json.loads(first.read_text(encoding="utf-8"))
    second.write_text(json.dumps(dict(reversed(list(descriptor.items()))), separators=(",", ":")), encoding="utf-8")
    assert file_sha256(first) != file_sha256(second)
    assert tokenizer_identity(first) == tokenizer_identity(second) == tokenizer_identity(None)
    validate_tokenizer(raw_byte_identity(), second)


def test_same_size_bpe_vocabulary_replacement_is_rejected(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    # Identity hashes the whole BPE artifact without loading optional Tokenizers.
    first.write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}}), encoding="utf-8")
    second.write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 1, "b": 0}, "merges": []}}), encoding="utf-8")
    identity = {"kind": "prepared", "tokenizer_sha256": tokenizer_identity(first)}
    validate_tokenizer(identity, first)
    with pytest.raises(ValueError, match="Tokenizer identity"):
        validate_tokenizer(identity, second)


@pytest.mark.parametrize("name", ["tokenizer.json", "train.npy", "validation.npy", "test.npy", "documents.jsonl"])
def test_every_advertised_artifact_is_verified(tmp_path, name):
    _, output = make_bundle(tmp_path)
    path = output / name
    content = bytearray(path.read_bytes())
    content[-1] ^= 1  # Keep file size fixed so the SHA check must detect this.
    path.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepared_identity(output)


def test_incomplete_manifest_and_missing_artifact_are_rejected(tmp_path):
    _, output = make_bundle(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["train.npy"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        prepared_identity(output)
    manifest["files"]["train.npy"] = {"sha256": file_sha256(output / "train.npy")}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output / "validation.npy").unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        prepared_identity(output)


def test_manifest_paths_cannot_reference_files_outside_prepared_bundle(tmp_path):
    _, output = make_bundle(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../outside.txt"] = {"sha256": file_sha256(outside)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        prepared_identity(output)


def test_file_sha256_reads_large_artifacts_in_chunks(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(600_000, dtype=np.uint32))
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_changed_byte_mapping_and_absent_checkpoint_identity_are_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    ByteTokenizer().save(path)
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor["pad_id"] = 255
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed UTF-8 byte mapping"):
        tokenizer_identity(path)
    with pytest.raises(ValueError, match="does not declare"):
        validate_tokenizer({}, None)
