import json

import numpy as np
import pytest

from mqr.balance import balance_prepared
from mqr.data import ByteTokenizer, load_tokenizer
from mqr.prepare import LANGUAGES, SPLITS, prepare_jsonl
from mqr.provenance import file_sha256, prepared_identity


def source_bundle(tmp_path, *, test_languages=LANGUAGES):
    records = []
    for split in SPLITS:
        for language_index, language in enumerate(LANGUAGES):
            if split == "test" and language not in test_languages:
                continue
            for document_index in range(2):
                # Unequal lengths cause both whole-document preservation and a
                # final cropped document, using distinguishable content per split.
                text = f"{split} {language} {document_index} " + "abcdefghij " * (language_index + 1)
                records.append({"id": f"{split}-{language}-{document_index}", "text": text,
                                "language": language, "split": split, "source": "authored-test", "revision": "r1"})
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    prepared = tmp_path / "source-prepared"
    prepare_jsonl(path, prepared)
    return prepared


def metadata(directory):
    return [json.loads(line) for line in (directory / "documents.jsonl").read_text(encoding="utf-8").splitlines()]


def test_balanced_streams_have_exact_shares_offsets_and_source_spans(tmp_path):
    source = source_bundle(tmp_path)
    output = tmp_path / "balanced"
    manifest = balance_prepared(source, output, seed=17)
    source_documents = {row["id"]: row for row in metadata(source)}
    selected = metadata(output)
    seen_ids = set()
    cropped = []
    for split in SPLITS:
        original = np.load(source / f"{split}.npy", mmap_mode="r")
        stream = np.load(output / f"{split}.npy", mmap_mode="r")
        quota = min(summary["tokens"] for summary in json.loads((source / "manifest.json").read_text())["splits"][split]["languages"].values())
        assert stream.size == len(LANGUAGES) * quota
        assert manifest["balance"]["splits"][split]["tokens_per_language"] == quota
        assert {row["tokens"] for row in manifest["splits"][split]["languages"].values()} == {quota}
        position = 0
        for language in manifest["balance"]["splits"][split]["language_order"]:
            summary = manifest["splits"][split]["languages"][language]
            assert summary["token_start"] == position
            assert summary["token_end"] == position + quota
            position += quota
        for row in (row for row in selected if row["split"] == split):
            assert row["id"] not in seen_ids
            seen_ids.add(row["id"])
            span = row["source_token_span"]
            assert span["split"] == split
            start, end = row["token_start"], row["token_start"] + row["tokens"]
            np.testing.assert_array_equal(stream[start:end], original[span["token_start"]:span["token_end"]])
            assert row["source_document_metadata"] == source_documents[row["id"]]
            assert row["source_document_metadata"]["split"] == split
            if row["complete_document"]:
                assert int(stream[end - 1]) == ByteTokenizer().eos_id
                assert row["eos_preserved"]
                assert row["content_sha256"] == source_documents[row["id"]]["content_sha256"]
            else:
                cropped.append(row)
                assert row["content_sha256"] is None
                assert row["utf8_bytes"] is None
                assert row["tokens"] < source_documents[row["id"]]["tokens"]
                assert not row["eos_preserved"]
        assert manifest["splits"][split]["documents"] == sum(row["split"] == split for row in selected)
        assert manifest["splits"][split]["utf8_bytes"] is None
        original._mmap.close()
        stream._mmap.close()
    assert cropped


def test_tokenizer_provenance_hashes_determinism_and_no_aliasing(tmp_path):
    source = source_bundle(tmp_path)
    before = {name: file_sha256(source / name) for name in ("manifest.json", "documents.jsonl", "tokenizer.json", "train.npy", "validation.npy", "test.npy")}
    first, second = tmp_path / "first", tmp_path / "second"
    one = balance_prepared(source, first, seed=9)
    two = balance_prepared(source, second, seed=9)
    assert one == two
    assert prepared_identity(first)["tokenizer_sha256"] == prepared_identity(source)["tokenizer_sha256"]
    assert one["source_preparation"]["manifest_sha256"] == before["manifest.json"]
    assert one["files"]["source_manifest.json"]["sha256"] == before["manifest.json"]
    assert one["tokenizer"]["vocab_size"] == load_tokenizer(first / "tokenizer.json").vocab_size
    assert (first / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    for name, expected in one["files"].items():
        assert file_sha256(first / name) == expected["sha256"]
        assert not (first / name).samefile(source / ("manifest.json" if name == "source_manifest.json" else name))
    changed = np.load(first / "train.npy", mmap_mode="r+")
    changed[0] = (int(changed[0]) + 1) % 259
    changed.flush()
    changed._mmap.close()
    assert before == {name: file_sha256(source / name) for name in before}
    assert file_sha256(second / "train.npy") == two["files"]["train.npy"]["sha256"]


def test_empty_test_split_allowed_but_partial_language_split_rejected(tmp_path):
    source = source_bundle(tmp_path, test_languages=())
    output = tmp_path / "balanced"
    manifest = balance_prepared(source, output)
    assert np.load(output / "test.npy").size == 0
    assert manifest["splits"]["test"]["tokens"] == 0
    nested = tmp_path / "partial"
    nested.mkdir()
    partial = source_bundle(nested, test_languages=("en", "de"))
    with pytest.raises(ValueError, match="every selected language"):
        balance_prepared(partial, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_output_immutable_and_input_integrity_verified(tmp_path):
    source = source_bundle(tmp_path)
    output = tmp_path / "balanced"
    balance_prepared(source, output)
    with pytest.raises(FileExistsError):
        balance_prepared(source, output)
    with (source / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="mismatch"):
        balance_prepared(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_source_span_overlap_rejected_even_with_updated_artifact_hash(tmp_path):
    source = source_bundle(tmp_path)
    rows = metadata(source)
    rows[1]["token_start"] = rows[0]["token_start"]
    documents = source / "documents.jsonl"
    documents.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["documents.jsonl"] = {"sha256": file_sha256(documents), "bytes": documents.stat().st_size}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="gaps or overlap|EOS"):
        balance_prepared(source, tmp_path / "invalid")


def test_balancing_never_uses_cross_split_documents(tmp_path):
    source = source_bundle(tmp_path)
    output = tmp_path / "balanced"
    balance_prepared(source, output)
    for row in metadata(output):
        assert row["id"].startswith(row["split"] + "-")
        assert row["source_token_span"]["split"] == row["split"]
        assert row["source_document_metadata"]["split"] == row["split"]


def test_bpe_actual_vocabulary_and_tokenizer_identity_are_preserved(tmp_path):
    pytest.importorskip("tokenizers")
    source_bundle(tmp_path)
    source = tmp_path / "source-bpe"
    prepared = prepare_jsonl(tmp_path / "source.jsonl", source, tokenizer_kind="bpe", vocab_size=300)
    output = tmp_path / "balanced-bpe"
    result = balance_prepared(source, output)
    assert result["tokenizer"] == prepared["tokenizer"]
    assert result["tokenizer"]["vocab_size"] == load_tokenizer(output / "tokenizer.json").vocab_size
    assert file_sha256(source / "tokenizer.json") == file_sha256(output / "tokenizer.json")
    assert prepared_identity(source)["tokenizer_sha256"] == prepared_identity(output)["tokenizer_sha256"]
