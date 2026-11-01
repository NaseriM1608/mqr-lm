import json

import pytest

from mqr.audit import audit_data
import mqr.audit as audit_module
from mqr.balance import balance_prepared
from mqr.data import ByteTokenizer
from mqr.prepare import prepare_jsonl
from mqr.provenance import file_sha256


def _bundle(tmp_path, tokenizer_kind="byte"):
    texts = {
        "en": "The quiet river passes a library and a station. ",
        "de": "Grüße aus Köln: Eine Leserin öffnet ein schönes Buch. ",
        "ar": "نقرأ كتابا جديدا عن المدينة الجميلة هذا الصباح. ",
        "fa": "در کتابخانه یک داستان تازه می‌خوانیم و یادداشت می‌نویسیم. ",
    }
    records = [{"id": f"{language}-{split}", "language": language, "split": split,
                "text": f"{split}\n{text * (index + 2)}", "source": "audit-fixture", "revision": "a" * 40}
               for split in ("train", "validation", "test")
               for index, (language, text) in enumerate(texts.items())]
    source = tmp_path / "selected.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepare_jsonl(source, prepared, tokenizer_kind=tokenizer_kind, vocab_size=300)
    return source, prepared, records


@pytest.mark.parametrize("tokenizer_kind", ["byte", "bpe"])
def test_audit_reconciles_counts_roundtrip_and_zwnj_without_test_sampling(tmp_path, tokenizer_kind):
    if tokenizer_kind == "bpe":
        pytest.importorskip("tokenizers")
    source, prepared, records = _bundle(tmp_path, tokenizer_kind)
    report = audit_data(source, prepared)
    assert report["selected_documents"]["documents"] == 12
    assert report["tokenizer"]["all_256_byte_symbols_present"]
    assert report["tokenizer"]["unk_token"] is None
    assert report["sample_checks"]["unicode_probe_roundtrip"]
    assert report["sample_checks"]["persian_zwnj_probe_preserved"]
    for language, checks in report["sample_checks"]["by_language"].items():
        assert set(checks) == {"train", "validation"}
        assert checks["train"]["roundtrip_passed"] == 1
        if language == "fa":
            assert checks["train"]["zwnj_characters_preserved"] > 0
    first = records[0]
    summary = report["prepared"]["by_language"][first["language"]][first["split"]]
    assert summary["utf8_bytes"] == len(first["text"].encode("utf-8"))
    assert summary["whitespace_words"] == len(first["text"].split())
    assert summary["tokens_excluding_eos"] == summary["tokens"] - 1
    assert summary["tokens_per_whitespace_word"] == summary["tokens_excluding_eos"] / len(first["text"].split())
    assert "whitespace" in report["fertility_definition"]
    assert report["warnings"]  # Deliberately tiny fixture cannot support a 512-token context.


def test_balanced_audit_verifies_source_spans_and_reports_exact_shares(tmp_path):
    source, prepared, _ = _bundle(tmp_path)
    balanced = tmp_path / "balanced"
    balance_prepared(prepared, balanced, seed=5)
    report = audit_data(source, prepared, balanced)
    assert report["balanced"]["source_token_spans_verified"] == 12
    for split in ("train", "validation", "test"):
        shares = report["balanced"]["token_shares"][split]
        assert shares["exactly_equal"]
        assert set(shares["shares"].values()) == {0.25}
    cropped = [stats for slices in report["balanced"]["by_language"].values() for stats in slices.values() if stats["cropped_documents"]]
    assert cropped
    assert all(stats["utf8_bytes"] is None and stats["tokens_per_whitespace_word"] is None for stats in cropped)


def test_audit_rejects_source_file_change_before_using_document_statistics(tmp_path):
    source, prepared, _ = _bundle(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input SHA"):
        audit_data(source, prepared)


def test_audit_rejects_tampered_prepared_artifact(tmp_path):
    source, prepared, _ = _bundle(tmp_path)
    path = prepared / "train.npy"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_data(source, prepared)


def test_balanced_logical_source_span_error_is_rejected_even_with_valid_file_hash(tmp_path):
    source, prepared, _ = _bundle(tmp_path)
    balanced = tmp_path / "balanced"
    balance_prepared(prepared, balanced)
    path = balanced / "documents.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["source_token_span"]["token_start"] += 1
    rows[0]["source_token_span"]["token_end"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = balanced / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["documents.jsonl"] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source span"):
        audit_data(source, prepared, balanced)


def test_sample_roundtrip_detects_a_decoder_that_removes_persian_zwnj(tmp_path, monkeypatch):
    class LossyByteTokenizer(ByteTokenizer):
        def decode(self, ids):
            return super().decode(ids).replace("\u200c", "")

    source, prepared, _ = _bundle(tmp_path)
    monkeypatch.setattr(audit_module, "load_tokenizer", lambda path: LossyByteTokenizer())
    with pytest.raises(ValueError, match="round-trip changed"):
        audit_data(source, prepared)
