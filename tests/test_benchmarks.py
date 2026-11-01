import json
from pathlib import Path
import shutil

import pytest

import mqr.benchmarks as benchmarks
from mqr.benchmarks import fetch_benchmarks
from mqr.provenance import file_sha256

REVISION = "a" * 40


def write_lock(path, entries):
    path.write_text(json.dumps({"benchmarks": entries}), encoding="utf-8")


def hf_entry(benchmark="FLORES200", language="en", split="dev", path="data/part.parquet", fields=None):
    entry = {"benchmark": benchmark, "split": split, "repo": "test/benchmark",
             "revision": REVISION, "path": path, "format": "parquet", "text_fields": fields or ["sentence"]}
    if language is not None:
        entry["language"] = language
    return entry


def parquet(path, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)


def install_offline_download(monkeypatch, mapping):
    calls = []

    def download(url, destination, **kwargs):
        calls.append((url, kwargs))
        source = next(path for suffix, path in mapping.items() if url.endswith(suffix))
        shutil.copyfile(source, destination)
        return {"bytes": destination.stat().st_size, "sha256": file_sha256(destination), "attempts": 1}

    monkeypatch.setattr(benchmarks, "_download", download)
    return calls


def exclusions(directory):
    return [json.loads(line) for line in (directory / "exclusions.jsonl").read_text(encoding="utf-8").splitlines()]


def test_parquet_translation_tsv_fields_and_exact_family_dedup(tmp_path, monkeypatch):
    first = tmp_path / "sentences.parquet"
    parquet(first, [{"sentence": "Same exact sentence.", "label": "never inspected"},
                    {"sentence": "Same exact sentence.", "label": "irrelevant"},
                    {"sentence": "  Keep original spacing.  ", "label": None}])
    pair = tmp_path / "pairs.parquet"
    parquet(pair, [{"translation": {"en": "A bilingual example.", "de": "Ein zweisprachiges Beispiel."}}])
    tsv = tmp_path / "farstail.tsv"
    tsv.write_text("premise\thypothesis\tlabel\nیک پیش‌فرض\tیک فرضیه\tignored\n", encoding="utf-8")
    lock = tmp_path / "sources.json"
    write_lock(lock, [hf_entry(path="sentences.parquet"),
                     hf_entry("OPUS100", None, "validation", "pairs.parquet", ["translation"]),
                     {"benchmark": "FarsTail", "split": "train", "format": "tsv",
                      "url": f"https://raw.githubusercontent.com/test/repo/{REVISION}/farstail.tsv",
                      "text_fields": ["premise", "hypothesis"]}])
    calls = install_offline_download(monkeypatch, {"sentences.parquet": first, "pairs.parquet": pair, "farstail.tsv": tsv})
    output = tmp_path / "exclusions"
    manifest = fetch_benchmarks(lock, output)
    records = exclusions(output)
    assert len(records) == 6
    assert len({record["id"] for record in records}) == len(records)
    assert {record["language"] for record in records if record["benchmark"] == "OPUS100"} == {"en", "de"}
    assert {record["language"] for record in records if record["benchmark"] == "FarsTail"} == {"fa"}
    assert "  Keep original spacing.  " in {record["text"] for record in records}
    assert not any("label" in record or "ignored" in record["text"] for record in records)
    assert manifest["sources"][0]["duplicate_texts"] == 1
    assert manifest["sources"][0]["rows_scanned"] == 3
    assert manifest["coverage"]["FLORES200"]["en"]["dev"] == 3
    assert manifest["emitted_coverage"]["FLORES200"]["en"]["dev"] == 2
    assert all(record["revision"] == REVISION for record in records)
    assert len(calls) == 3
    assert manifest["scope"]["missing_or_zero_text_units"]
    for name, artifact in manifest["files"].items():
        assert file_sha256(output / name) == artifact["sha256"]


def test_same_source_downloaded_once_and_dedup_is_per_family(tmp_path, monkeypatch):
    source = tmp_path / "same.parquet"
    parquet(source, [{"sentence": "Shared example."}])
    lock = tmp_path / "sources.json"
    write_lock(lock, [hf_entry(path="same.parquet"),
                     hf_entry(split="devtest", path="same.parquet"),
                     hf_entry("XNLI", "en", "validation", "same.parquet")])
    calls = install_offline_download(monkeypatch, {"same.parquet": source})
    output = tmp_path / "output"
    manifest = fetch_benchmarks(lock, output)
    assert len(calls) == 1
    assert manifest["unique_downloads"] == 1
    assert len(exclusions(output)) == 2
    assert manifest["sources"][1]["duplicate_texts"] == 1


@pytest.mark.parametrize("change", [{"revision": "main"}, {"path": "../bad.parquet"},
                                    {"language": "xx"}, {"text_fields": []}, {"format": "json"}])
def test_invalid_lock_rejected_before_network(tmp_path, monkeypatch, change):
    lock = tmp_path / "sources.json"
    entry = hf_entry()
    entry.update(change)
    write_lock(lock, [entry])
    monkeypatch.setattr(benchmarks, "_download", lambda *args, **kwargs: pytest.fail("Unexpected network"))
    with pytest.raises(ValueError):
        fetch_benchmarks(lock, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_byte_limit_and_failed_download_leave_no_published_directory(tmp_path, monkeypatch):
    source = tmp_path / "data.parquet"
    parquet(source, [{"sentence": "A real text column."}])
    lock = tmp_path / "sources.json"
    write_lock(lock, [hf_entry(path="data.parquet")])
    install_offline_download(monkeypatch, {"data.parquet": source})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="byte limit"):
        fetch_benchmarks(lock, output, max_file_bytes=1)
    assert not output.exists()

    def fail(*args, **kwargs):
        raise OSError("offline test failure")

    monkeypatch.setattr(benchmarks, "_download", fail)
    with pytest.raises(OSError, match="offline test failure"):
        fetch_benchmarks(lock, output)
    assert not output.exists()


def test_missing_text_column_fails_instead_of_false_coverage(tmp_path, monkeypatch):
    source = tmp_path / "wrong.parquet"
    parquet(source, [{"other": "not the declared schema"}])
    lock = tmp_path / "sources.json"
    write_lock(lock, [hf_entry(path="wrong.parquet")])
    install_offline_download(monkeypatch, {"wrong.parquet": source})
    with pytest.raises(ValueError, match="text fields missing"):
        fetch_benchmarks(lock, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_output_refusal_and_deterministic_manifest(tmp_path, monkeypatch):
    source = tmp_path / "data.parquet"
    parquet(source, [{"sentence": "A reproducible text."}])
    lock = tmp_path / "sources.json"
    write_lock(lock, [hf_entry(path="data.parquet")])
    install_offline_download(monkeypatch, {"data.parquet": source})
    first, second = tmp_path / "first", tmp_path / "second"
    one = fetch_benchmarks(lock, first)
    two = fetch_benchmarks(lock, second)
    assert one == two
    assert (first / "exclusions.jsonl").read_bytes() == (second / "exclusions.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        fetch_benchmarks(lock, first)


def test_real_downloader_is_streamed_bounded_and_retries(tmp_path, monkeypatch):
    class Response:
        headers = {}

        def __init__(self, payload):
            import io
            self.stream = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, count):
            assert count <= 65_536
            return self.stream.read(count)

    attempts = []

    def open_url(request, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise OSError("temporary connection error")
        return Response(b"bounded payload")

    monkeypatch.setattr(benchmarks, "urlopen", open_url)
    monkeypatch.setattr(benchmarks.time, "sleep", lambda seconds: None)
    path = tmp_path / "download"
    result = benchmarks._download("https://example.com/file", path, max_bytes=100, timeout=3.0, retries=1)
    assert result["attempts"] == 2
    assert attempts == [3.0, 3.0]
    assert path.read_bytes() == b"bounded payload"
    with pytest.raises(ValueError, match="byte limit"):
        benchmarks._download("https://example.com/file", path, max_bytes=3, timeout=3.0, retries=0)
    assert not path.exists()


def test_tar_text_regular_members_are_byte_pinned_and_cached_once(tmp_path, monkeypatch):
    import io
    import tarfile

    archive = tmp_path / "flores.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, text in (("flores200_dataset/dev/eng_Latn.dev", "English example.\nSecond English.\n"),
                           ("flores200_dataset/dev/deu_Latn.dev", "Deutsches Beispiel.\n")):
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
    url = "https://example.com/flores.tar.gz"
    entries = [{"benchmark": "FLORES200", "language": language, "split": "dev",
                "url": url, "revision": "publisher-original-release", "expected_sha256": file_sha256(archive),
                "format": "tar_text", "member": f"flores200_dataset/dev/{code}.dev", "text_fields": ["sentence"]}
               for language, code in (("en", "eng_Latn"), ("de", "deu_Latn"))]
    lock = tmp_path / "sources.json"
    write_lock(lock, entries)
    calls = install_offline_download(monkeypatch, {"flores.tar.gz": archive})
    cache = tmp_path / "cache"
    first = fetch_benchmarks(lock, tmp_path / "first", cache_dir=cache)
    second = fetch_benchmarks(lock, tmp_path / "second", cache_dir=cache)
    assert len(calls) == 1
    assert first["unique_downloads"] == second["unique_downloads"] == 1
    assert first["all_artifacts_predeclared_pinned"]
    assert second["network_downloaded_bytes"] == 0
    assert all(source["cache_hit"] for source in second["sources"])
    assert {row["text"] for row in exclusions(tmp_path / "first")} == {"English example.", "Second English.", "Deutsches Beispiel."}
    assert {row["language"] for row in exclusions(tmp_path / "first")} == {"en", "de"}
    assert not (tmp_path / "first" / "flores200_dataset").exists()


@pytest.mark.parametrize("failure", ["digest", "missing_member", "symlink", "unsafe_path"])
def test_tar_member_and_content_pin_validation(tmp_path, monkeypatch, failure):
    import io
    import tarfile

    archive = tmp_path / "flores.tar.gz"
    member = "flores200_dataset/dev/eng_Latn.dev"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo(member)
        if failure == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            output.addfile(info)
        else:
            payload = b"A benchmark sentence.\n"
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
    entry = {"benchmark": "FLORES200", "language": "en", "split": "dev",
             "url": "https://example.com/flores.tar.gz", "revision": "original-release",
             "format": "tar_text", "member": member, "text_fields": ["sentence"],
             "expected_sha256": "0" * 64 if failure == "digest" else file_sha256(archive)}
    if failure == "missing_member":
        entry["member"] = "flores200_dataset/dev/missing.dev"
    if failure == "unsafe_path":
        entry["member"] = "../escape"
    lock = tmp_path / "sources.json"
    write_lock(lock, [entry])
    install_offline_download(monkeypatch, {"flores.tar.gz": archive})
    with pytest.raises(ValueError):
        fetch_benchmarks(lock, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "escape").exists()
