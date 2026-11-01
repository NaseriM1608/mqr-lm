import io
import json

import pytest

import mqr.fetch as fetch


def lock_file(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"corpus": [dict(language=lang, repo="test/repo", revision="a" * 40,
                                               files=["data.parquet"]) for lang in ("en", "de", "ar", "fa")]}))
    return path


def test_bounded_reader_does_not_allow_whole_file_or_over_budget():
    reader = fetch.BoundedReader(io.BytesIO(b"abcdef"), 4)
    assert reader.read(3) == b"abc"
    reader.seek(0)
    assert reader.read(1) == b"a"
    with pytest.raises(ValueError, match="budget"):
        reader.read(1)
    with pytest.raises(ValueError, match="budget"):
        reader.read()


def test_fetch_keeps_whole_original_documents_and_pinned_provenance(tmp_path, monkeypatch):
    def rows(source, budget):
        for i, text in enumerate(["", "short", source["language"] + " valid text " * 30, "second " * 60]):
            yield {"text": text, "id": str(i)}, {"path": "data.parquet", "row": i, "parquet_bytes_requested": 100}
    monkeypatch.setattr(fetch, "_iter_rows", rows)
    output = tmp_path / "raw"
    manifest = fetch.fetch_corpus(lock_file(tmp_path), output, bytes_per_language=100, min_document_bytes=100)
    assert len(manifest["languages"]) == 4
    for entry in manifest["languages"]:
        data = [json.loads(x) for x in (output / entry["file"]).read_text().splitlines()]
        assert len(data) == 1
        assert data[0]["revision"] == "a" * 40
        assert data[0]["text"].startswith(entry["language"])
        assert entry["rejected_length_or_empty"] == 2
    with pytest.raises(ValueError, match="already exists"):
        fetch.fetch_corpus(lock_file(tmp_path), output)


def test_fetch_failure_never_publishes_incomplete_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_iter_rows", lambda *args: iter([]))
    with pytest.raises(ValueError, match="exhausted"):
        fetch.fetch_corpus(lock_file(tmp_path), tmp_path / "raw", bytes_per_language=10)
    assert not (tmp_path / "raw").exists()


def test_fetch_rejects_mutable_revision(tmp_path):
    path = lock_file(tmp_path)
    data = json.loads(path.read_text())
    data["corpus"][0]["revision"] = "main"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="immutable"):
        fetch.fetch_corpus(path, tmp_path / "raw")


def test_fetch_subset_avoids_redownloading_other_languages(tmp_path, monkeypatch):
    called = []
    def rows(source, budget):
        called.append(source["language"])
        yield {"text": "native text " * 30}, {"path": "data.parquet", "row": 0, "parquet_bytes_requested": 100}
    monkeypatch.setattr(fetch, "_iter_rows", rows)
    result = fetch.fetch_corpus(lock_file(tmp_path), tmp_path / "fa_only", bytes_per_language=100, languages=("fa",))
    assert called == ["fa"]
    assert [row["language"] for row in result["languages"]] == ["fa"]
    assert not (tmp_path / "fa_only/en.jsonl").exists()
