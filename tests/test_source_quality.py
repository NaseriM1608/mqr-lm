import hashlib
import json
from pathlib import Path

import pytest

from mqr.source_quality import (FastTextLID, audit_sources, calibrate_lid,
                                domain_of, scoring_spans, select_review_samples,
                                sha256_file, text_statistics)


def write_raw(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def row(document_id="a", text="A coherent English sentence.", language="en", **kwargs):
    return {"id": document_id, "text": text, "language": language,
            "source": "publisher", "revision": "1" * 40, "source_shard": "shard.parquet",
            "source_row": 3, "url": "https://Example.com/page", **kwargs}


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_raw_orthography_unicode_counts_not_normalized():
    text = "می‌روم ی ي ک ك ۱۲٣\ufffd\x00\u200d"
    result = text_statistics(text, "fa")
    assert result["zwnj"] == result["zwj"] == 1
    assert result["arabic_yeh"] == result["arabic_kaf"] == 1
    assert result["persian_yeh"] == 2
    assert result["persian_keheh"] == 1
    assert result["persian_digits"] == 2
    assert result["arabic_indic_digits"] == 1
    assert result["replacement_characters"] == result["nul_characters"] == 1
    assert result["utf8_bytes"] == len(text.encode("utf-8"))
    assert result["control_and_format_characters"]["U+200C"] == 1
    assert "encoding_or_nul" in result["review_flags"]


def test_unicode_line_separator_is_inside_original_jsonl_text(tmp_path):
    text = "first\u2028second\u0085third\u2029fourth"
    raw = write_raw(tmp_path / "raw.jsonl", [row(text=text), row("other", "another document")])
    result = audit_sources([raw], tmp_path / "audit")
    assert result["summary"]["documents"] == 2
    reviews = read_jsonl(tmp_path / "audit/review_samples.jsonl")
    assert any(review["excerpt"]["parts"] == [text] for review in reviews)


def test_repetition_counts_extra_occurrences_not_all_repeated_content():
    stats = text_statistics("alpha beta gamma delta epsilon\nalpha beta gamma delta epsilon", "en")
    assert stats["repeated_line_extra_fraction"] == .5
    assert stats["repeated_word_5gram_extra_slots"] == 1
    assert stats["word_5gram_slots"] == 6
    assert "repetition" in stats["review_flags"]


def test_arabic_script_does_not_assert_language():
    stats = text_statistics("یہ اردو زبان میں ایک مثال ہے", "fa")
    assert stats["expected_script_letter_share"] == 1
    assert "language" not in stats
    assert "persian_without_zwnj" in stats["review_flags"]


@pytest.mark.parametrize("url, expected", [(None, "[missing]"), ("", "[missing]"),
    ("https://WWW.Example.com/path?q=x", "www.example.com"),
    ("fa.wikipedia.org/wiki/example", "fa.wikipedia.org"), ("http://[broken", "[invalid]")])
def test_domain_exact_host(url, expected):
    assert domain_of(url) == expected


def test_spans_do_not_overlap_or_claim_full_long_documents():
    text = "a" * 50_000
    spans = scoring_spans(text)
    assert len(spans) == 3 and sum(end - start for start, end in spans) == 12_288
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
    assert scoring_spans("abc") == [(0, 3)]


def test_report_domain_bytes_differ_from_document_share_and_raw_unchanged(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [row("a", "x" * 1000), row("b", "y" * 100, url="https://small.example/z")])
    before = raw.read_bytes()
    summary = audit_sources([raw], tmp_path / "audit", ordinary=1, risk=1)
    domains = {entry["slice"][1]: entry for entry in summary["slices"]["language_domain"]}
    assert domains["example.com"]["document_share_within_language"] == .5
    assert domains["example.com"]["byte_share_within_language"] == 1000 / 1100
    assert "tokens_excluding_eos" not in domains["example.com"]
    assert raw.read_bytes() == before
    ledger = read_jsonl(tmp_path / "audit/documents.jsonl")
    assert ledger[0]["text_sha256"] == hashlib.sha256(b"x" * 1000).hexdigest()
    for document in ledger:
        with raw.open("rb") as handle:
            handle.seek(document["raw_location"]["byte_offset"])
            recovered = json.loads(handle.readline())
        assert recovered["id"] == document["id"]
    manifest = json.loads((tmp_path / "audit/artifact_manifest.json").read_text())
    for filename, identity in manifest["artifacts"].items():
        assert sha256_file(tmp_path / "audit" / filename) == identity["sha256"]


def test_optional_token_counts_are_explicit_and_domain_weighted(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [row("a", "one two three"), row("b", "four", url="https://other.test")])
    report = audit_sources([raw], tmp_path / "audit", token_counter=lambda text: len(text.split()),
                           tokenizer_identity={"sha256": "frozen"})
    domains = {entry["slice"][1]: entry for entry in report["slices"]["language_domain"]}
    assert domains["example.com"]["token_share_within_language"] == .75
    assert report["tokenizer"] == {"sha256": "frozen"}


@pytest.mark.parametrize("change", [{"id": ""}, {"language": "ur"}, {"text": None},
    {"source_language_score": float("nan")}, {"source_language_score": True},
    {"source_language_score": 1.1}, {"text_sha256": "wrong"}])
def test_invalid_raw_rejected_without_output(tmp_path, change):
    raw = write_raw(tmp_path / "raw.jsonl", [{**row(), **change}])
    with pytest.raises(ValueError):
        audit_sources([raw], tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_duplicate_ids_and_duplicate_input_paths_rejected(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        audit_sources([raw], tmp_path / "audit")
    with pytest.raises(ValueError, match="distinct"):
        audit_sources([raw, raw], tmp_path / "audit")


def test_existing_output_is_not_overwritten(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [row()])
    output = tmp_path / "audit"
    output.mkdir()
    marker = output / "important"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        audit_sources([raw], output)
    assert marker.read_text() == "preserve"


def test_exact_duplicate_text_detected_across_languages_and_sources(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [row("a"), row("b", language="de", source="other")])
    result = audit_sources([raw], tmp_path / "audit")
    assert result["exact_text_duplicate_groups"][0]["ids"] == ["a", "b"]


def test_review_seeded_ordinary_and_targeted_slate_are_distinct():
    documents = [{"id": str(index), "language": "fa", "review_strata": ["source:web", "persian_without_zwnj"]} for index in range(30)]
    a = select_review_samples(documents, seed=4, ordinary=5, risk=5)
    b = select_review_samples(list(reversed(documents)), seed=4, ordinary=5, risk=5)
    assert a == b
    assert len(a) == len({item["id"] for item in a}) == 10
    assert sum(item["review_selection"]["kind"] == "ordinary" for item in a) == 5
    assert a[0]["review_selection"]["inclusion_fraction"] == 5 / 30
    assert a != select_review_samples(documents, seed=5, ordinary=5, risk=5)


def test_review_quality_flags_are_not_drowned_out_by_domain_strata():
    documents = [{"id": str(index), "language": "ar", "review_strata": ["domain:" + str(index)]} for index in range(100)]
    documents[-1]["review_strata"].append("local_lid_disagreement")
    slate = select_review_samples(documents, seed=0, ordinary=0, risk=5)
    assert slate[0]["id"] == "99"
    assert slate[0]["review_selection"]["selected_stratum"] == "local_lid_disagreement"


def test_empty_inputs_are_not_successfully_certified(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [])
    with pytest.raises(ValueError, match="no documents"):
        audit_sources([raw], tmp_path / "audit")


class FakeClassifier:
    identity = {"sha256": "model-identity"}
    def score(self, text):
        language = "fa" if "فارسی" in text else "en"
        return {"top_language": language, "top_score": .75, "top3": [], "spans": [(0, len(text))], "window_predictions": []}


def test_lid_controls_record_failures_and_do_not_become_filter(tmp_path):
    classifier = FakeClassifier()
    controls = [{"id": "positive", "text": "فارسی", "expected_language": "fa"},
                {"id": "failure", "text": "English", "expected_language": "de"},
                {"id": "mixed", "text": "names 123", "expected_language": None}]
    calibration = calibrate_lid(classifier, controls)
    assert calibration["labelled_controls"] == 2 and calibration["correct_controls"] == 1
    raw = write_raw(tmp_path / "raw.jsonl", [row("arabic", "An English intrusion", "ar")])
    report = audit_sources([raw], tmp_path / "audit", classifier=classifier, calibration=calibration)
    assert report["summary"]["documents"] == 1
    assert report["summary"]["local_lid"]["disagrees_with_declared_documents"] == 1
    review = read_jsonl(tmp_path / "audit/review_samples.jsonl")
    assert "local_lid_disagreement" in review[0]["review_strata"]


def test_lid_requires_matching_calibration(tmp_path):
    raw = write_raw(tmp_path / "raw.jsonl", [row()])
    with pytest.raises(ValueError, match="calibration"):
        audit_sources([raw], tmp_path / "audit", classifier=FakeClassifier())
    with pytest.raises(ValueError, match="calibration"):
        audit_sources([raw], tmp_path / "audit", classifier=FakeClassifier(), calibration={"model": {"sha256": "other"}})


def test_model_hash_is_verified_before_loading_or_importing(tmp_path):
    path = tmp_path / "fake.bin"
    path.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="hash"):
        FastTextLID(path, "0" * 64)
