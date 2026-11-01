import json
import random
import string

import pytest

from mqr.corpus import LANGUAGES, canonical_url, normalize_text, select_corpus
from mqr.prepare import prepare_jsonl
from mqr.provenance import file_sha256


def _write(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def records():
    records = []
    for language_index, language in enumerate(LANGUAGES):
        for index in range(20):
            rng = random.Random(language_index * 1000 + index)
            words = ["".join(rng.choices(string.ascii_lowercase, k=9)) for _ in range(40)]
            records.append({"id": f"{language}-{index}", "language": language, "split": "provider-train",
                            "text": " ".join(words), "source": f"fixture/{language}", "revision": "a" * 40,
                            "url": f"https://example.test/{language}/{index}"})
    return records


def test_selection_is_order_independent_and_emits_preparer_compatible_splits(tmp_path, records):
    source = tmp_path / "source.jsonl"
    _write(source, records)
    first = tmp_path / "first"
    manifest = select_corpus(source, first, seed=42)
    _write(source, list(reversed(records)))
    second = tmp_path / "second"
    repeated = select_corpus(source, second, seed=42)
    assert manifest["files"] == repeated["files"]
    rows = [json.loads(line) for line in (first / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(records)
    assert len({row["id"] for row in rows}) == len(rows)
    original = {row["id"]: row for row in records}
    assert all(row["text"] == original[row["id"]]["text"] for row in rows)
    assert all(row["source_split"] == "provider-train" for row in rows)
    assert all(row["url"] == original[row["id"]]["url"] for row in rows)
    for language in LANGUAGES:
        summary = manifest["retained_by_language"][language]
        assert summary["documents"] == 20
        assert [summary["splits"][split]["documents"] for split in ("train", "validation", "test")] == [16, 2, 2]
    assert manifest["benchmark_exclusions"]["missing_benchmarks"] == ["farstail", "flores200", "opus100", "xnli"]
    prepared = prepare_jsonl(first / "documents.jsonl", tmp_path / "tokens")
    assert prepared["splits"]["train"]["documents"] == 64
    assert prepared["splits"]["validation"]["documents"] == 8
    assert prepared["splits"]["test"]["documents"] == 8


def test_exact_near_and_repeated_id_documents_are_removed_globally(tmp_path, records):
    base = records[0]
    exact = {**base, "id": "en-exact", "text": " \n " + base["text"].upper().replace(" ", "\t") + "  ", "url": "https://example.test/exact"}
    near = {**base, "id": "en-near", "text": base["text"] + " extraworda extrawordb", "url": "https://example.test/near"}
    source = tmp_path / "source.jsonl"
    _write(source, [*records, exact, near, base])
    output = tmp_path / "selected"
    manifest = select_corpus(source, output)
    drops = manifest["dropped_documents"]
    assert drops["duplicate_id"] == 1
    assert drops.get("exact_duplicate", 0) + drops.get("near_duplicate", 0) == 2
    rows = [json.loads(line) for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(records)
    assert len({normalize_text(row["text"]) for row in rows}) == len(rows)
    assert len({row["id"] for row in rows} & {base["id"], exact["id"], near["id"]}) == 1
    assert "can miss" in manifest["deduplication"]["limitations"]


def test_benchmark_ngram_exclusion_and_missing_family_coverage_are_explicit(tmp_path, records):
    source = tmp_path / "source.jsonl"
    _write(source, records)
    phrase = " ".join(records[0]["text"].split()[5:20])
    exclusions = tmp_path / "exclusions.jsonl"
    _write(exclusions, [{"id": "devtest-1", "benchmark": "FLORES-200", "source": "official-fixture", "revision": "b" * 40,
                         "text": phrase.upper().replace(" ", " , ")}])
    output = tmp_path / "selected"
    manifest = select_corpus(source, output, exclusions_path=exclusions)
    assert manifest["dropped_documents"]["benchmark_overlap"] == 1
    selected_ids = {json.loads(line)["id"] for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()}
    assert records[0]["id"] not in selected_ids
    coverage = manifest["benchmark_exclusions"]
    assert coverage["benchmark_document_counts"] == {"flores200": 1}
    assert coverage["missing_benchmarks"] == ["farstail", "opus100", "xnli"]
    assert coverage["files"][0]["sha256"] == file_sha256(exclusions)
    with pytest.raises(ValueError, match="every expected benchmark"):
        select_corpus(source, tmp_path / "strict-missing", exclusions_path=exclusions, research_ready=True)


def test_coverage_gate_requires_exclusions_and_never_claims_comprehensive_readiness(tmp_path, records):
    source = tmp_path / "source.jsonl"
    _write(source, records)
    with pytest.raises(ValueError, match="nonempty exclusions"):
        select_corpus(source, tmp_path / "no-exclusions", research_ready=True)
    exclusions = tmp_path / "exclusions.jsonl"
    _write(exclusions, [{"benchmark": benchmark, "text": f"independent benchmark {benchmark} example sentence not present in source documents"}
                        for benchmark in ("FLORES-200", "XNLI", "FarsTail", "OPUS-100")])
    manifest = select_corpus(source, tmp_path / "screened", exclusions_path=exclusions, research_ready=True)
    assert manifest["benchmark_exclusions"]["complete_requested_family_coverage"]
    assert manifest["artifact_kind"] == "screened_bounded_corpus_sample"
    assert "no comprehensive" in manifest["quality_scope"]
    assert manifest["requested_benchmark_coverage_gate"]


def test_input_revision_must_be_pinned_and_conflicting_ids_are_rejected(tmp_path, records):
    source = tmp_path / "source.jsonl"
    records[0]["revision"] = "main"
    _write(source, records)
    with pytest.raises(ValueError, match="40-character"):
        select_corpus(source, tmp_path / "unpinned")
    records[0]["revision"] = "a" * 40
    _write(source, [*records, {**records[0], "text": "different document text"}])
    with pytest.raises(ValueError, match="ID reused"):
        select_corpus(source, tmp_path / "conflicting-id")


def test_failed_retention_gate_does_not_publish_partial_outputs(tmp_path, records):
    source = tmp_path / "source.jsonl"
    _write(source, records)
    output = tmp_path / "too-small"
    with pytest.raises(ValueError, match="Insufficient retained"):
        select_corpus(source, output, min_documents_per_language=21)
    assert not output.exists()


def test_existing_output_is_immutable(tmp_path, records):
    source = tmp_path / "source.jsonl"
    _write(source, records)
    output = tmp_path / "selected"
    select_corpus(source, output)
    before = file_sha256(output / "documents.jsonl")
    with pytest.raises(FileExistsError):
        select_corpus(source, output)
    assert file_sha256(output / "documents.jsonl") == before


def test_canonical_urls_preserve_content_queries_and_path_case():
    first = "HTTP://WWW.Example.test/Forum/View.php?page=3&id=12&utm_source=test#section"
    second = "https://example.test/Forum/View.php?gclid=abc&id=12&page=3&fbclid=def"
    assert canonical_url(first) == canonical_url(second)
    assert canonical_url(first) != canonical_url(second.replace("id=12", "id=13"))
    assert canonical_url(first) != canonical_url(second.replace("/Forum/", "/forum/"))
    assert canonical_url("https://example.test/a%2Fb") != canonical_url("https://example.test/a/b")
    assert canonical_url("https://example.test/a%3Fb") != canonical_url("https://example.test/a?b")
    assert canonical_url("https://example.test/%252F") != canonical_url("https://example.test/%2F")
    assert canonical_url(None) is None
    assert canonical_url("") is None


def test_cross_source_wikipedia_url_duplicates_removed_despite_changed_text(tmp_path, records):
    records[0]["url"] = "http://fa.wikipedia.org/wiki/%DA%A9%D8%AA%D8%A7%D8%A8_%D9%86%D9%88#old"
    records[1]["url"] = "https://WWW.fa.wikipedia.org/wiki/کتاب نو?utm_medium=test#new"
    records[1]["source"] = "wikimedia/wikipedia"
    records[1]["revision"] = "b" * 40
    assert records[0]["text"] != records[1]["text"]
    assert canonical_url(records[0]["url"]) == canonical_url(records[1]["url"])
    source = tmp_path / "source.jsonl"
    _write(source, records)
    output = tmp_path / "selected"
    manifest = select_corpus(source, output)
    assert manifest["dropped_documents"]["duplicate_url"] == 1
    example = next(item for item in manifest["drop_examples_first_100"] if item["reason"] == "duplicate_url")
    assert {example["id"], example["duplicate_of"]} == {records[0]["id"], records[1]["id"]}
    rows = [json.loads(line) for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    selected_ids = {row["id"] for row in rows}
    assert len(selected_ids & {records[0]["id"], records[1]["id"]}) == 1
    assert len(rows) == len(records) - 1


def test_distinct_forum_page_query_ids_remain_separate(tmp_path, records):
    records[0]["url"] = "https://example.test/forum/view.php?id=10"
    records[1]["url"] = "https://example.test/forum/view.php?id=11"
    records[2]["url"] = None
    records[3]["url"] = None
    source = tmp_path / "source.jsonl"
    _write(source, records)
    manifest = select_corpus(source, tmp_path / "selected")
    assert manifest["retained_documents"] == len(records)
    assert manifest["dropped_documents"].get("duplicate_url", 0) == 0
