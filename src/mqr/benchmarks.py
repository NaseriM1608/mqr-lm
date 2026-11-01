"""Fetch pinned public benchmark texts for a bounded exclusion audit.

This collector reads only declared text fields and does not analyze labels.
Its manifest describes file coverage and limitations; collecting these files
does not establish comprehensive benchmark decontamination.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import tarfile
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .provenance import file_sha256

FAMILIES = ("FLORES200", "XNLI", "MNLI", "OPUS100", "FarsTail")
LANGUAGES = ("en", "de", "ar", "fa")
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_INTENDED_SCOPE = {
    "FLORES200": {language: ("dev", "devtest") for language in LANGUAGES},
    "XNLI": {language: ("validation", "test") for language in ("en", "de", "ar")},
    "MNLI": {"en": ("validation_matched", "validation_mismatched")},
    "OPUS100": {language: ("validation", "test") for language in LANGUAGES},
    "FarsTail": {"fa": ("train", "validation", "test")},
}


def _download(url: str, destination: Path, *, max_bytes: int, timeout: float, retries: int) -> dict:
    """Stream one HTTPS file with bounded attempts, size and request timeout."""
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "mqr-benchmark-exclusions/1"})
            digest = hashlib.sha256()
            size = 0
            with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
                advertised = response.headers.get("Content-Length")
                if advertised is not None and int(advertised) > max_bytes:
                    raise ValueError("Benchmark download exceeds its byte limit.")
                while True:
                    block = response.read(min(65_536, max_bytes - size + 1))
                    if not block:
                        break
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("Benchmark download exceeds its byte limit.")
                    output.write(block)
                    digest.update(block)
            return {"bytes": size, "sha256": digest.hexdigest(), "attempts": attempt + 1}
        except ValueError:
            destination.unlink(missing_ok=True)
            raise
        except OSError:
            destination.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2.0, 0.5 * (2 ** attempt)))
    raise RuntimeError("Download retry loop completed without a result.")


def _source(entry: dict) -> dict:
    if not isinstance(entry, dict) or entry.get("benchmark") not in FAMILIES:
        raise ValueError(f"Each benchmark source must name a family from {FAMILIES}.")
    if not isinstance(entry.get("split"), str) or not entry["split"].strip():
        raise ValueError("Benchmark sources must declare a split.")
    if entry.get("format") not in ("parquet", "tsv", "tar_text"):
        raise ValueError("Benchmark source format must be parquet, tsv or tar_text.")
    fields = entry.get("text_fields")
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields) or len(fields) != len(set(fields)):
        raise ValueError("text_fields must be a nonempty list of distinct column names.")
    language = entry.get("language")
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"Benchmark language must be one of {LANGUAGES}.")
    if language is None and entry["benchmark"] in ("MNLI", "FarsTail"):
        language = "en" if entry["benchmark"] == "MNLI" else "fa"
    if language is None and fields != ["translation"]:
        raise ValueError("Non-translation benchmark sources must specify language.")
    expected_sha = entry.get("expected_sha256")
    if expected_sha is not None and (not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha)):
        raise ValueError("expected_sha256 must be a 64-hex content digest.")
    expected_sha = expected_sha.lower() if expected_sha is not None else None
    if entry["format"] == "tar_text":
        member = entry.get("member")
        if not isinstance(member, str) or not member or PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts or "\\" in member or "\x00" in member:
            raise ValueError("tar_text requires a safe relative member path.")
        if len(fields) != 1:
            raise ValueError("tar_text uses exactly one text_fields column for each member line.")
    revision = entry.get("revision")
    if "url" in entry:
        url = entry["url"]
        if not isinstance(url, str):
            raise ValueError("Benchmark URL must be a string.")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Benchmark URLs must use HTTPS without embedded credentials.")
        path_parts = parsed.path.split("/")
        pinned_parts = [part for part in path_parts if _SHA.fullmatch(part)]
        if revision is None and len(pinned_parts) == 1:
            revision = pinned_parts[0]
        if revision is None and expected_sha is not None:
            revision = f"sha256:{expected_sha}"
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("Benchmark URLs require a pinned revision or an explicit release label.")
        revision_pinned = bool(_SHA.fullmatch(revision) and revision in path_parts)
        pinning = "sha256" if expected_sha is not None else ("revision_in_url" if revision_pinned else "unversioned_url_recorded_after_fetch")
    else:
        repo, path = entry.get("repo"), entry.get("path")
        if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", repo):
            raise ValueError("HF sources require a valid dataset repo name.")
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError("HF benchmark sources require an immutable 40-hex revision.")
        if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path:
            raise ValueError("HF source paths must be safe relative POSIX paths.")
        url = f"https://huggingface.co/datasets/{quote(repo, safe='/')}/resolve/{revision}/{quote(path, safe='/')}"
        pinning = "sha256_and_revision" if expected_sha is not None else "revision_in_url"
    return {**entry, "language": language, "revision": revision, "source_url": url,
            "expected_sha256": expected_sha, "pinning": pinning}


def _acquire(url: str, destination: Path, *, max_bytes: int, timeout: float, retries: int,
             expected_sha256: str | None, cache_dir: Path | None) -> dict:
    """Reuse verified URL-keyed artifacts; cache publication is independent of the run."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cached = cache_dir / f"{key}.data" if cache_dir is not None else None
    sidecar = cache_dir / f"{key}.json" if cache_dir is not None else None
    if cached is not None and cached.is_file() and sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            valid = (metadata.get("url") == url and metadata.get("bytes") == cached.stat().st_size
                     and metadata.get("sha256") == file_sha256(cached)
                     and (expected_sha256 is None or metadata.get("sha256") == expected_sha256))
        except (ValueError, OSError, AttributeError):
            valid = False
        if valid:
            if metadata["bytes"] > max_bytes:
                raise ValueError("Cached benchmark file exceeds its byte limit.")
            shutil.copyfile(cached, destination)
            return {"bytes": metadata["bytes"], "sha256": metadata["sha256"],
                    "attempts": 0, "cache_hit": True}
    details = _download(url, destination, max_bytes=max_bytes, timeout=timeout, retries=retries)
    size, digest = destination.stat().st_size, file_sha256(destination)
    if size > max_bytes:
        raise ValueError("Benchmark download exceeds its byte limit.")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Benchmark artifact SHA-256 does not match expected_sha256.")
    details = {**details, "bytes": size, "sha256": digest, "cache_hit": False}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mqr-cache-", dir=cache_dir) as temporary:
            temporary_path = Path(temporary)
            shutil.copyfile(destination, temporary_path / "artifact.data")
            (temporary_path / "artifact.json").write_text(json.dumps({"url": url, **details}, sort_keys=True) + "\n", encoding="utf-8")
            (temporary_path / "artifact.data").replace(cached)
            (temporary_path / "artifact.json").replace(sidecar)
    return details


def _rows(path: Path, source: dict, *, max_member_bytes: int = 64 * 1024 * 1024):
    fields = source["text_fields"]
    if source["format"] == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Install pyarrow to read benchmark Parquet files.") from exc
        parquet = pq.ParquetFile(path)
        try:
            missing = set(fields) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Benchmark Parquet text fields missing: {sorted(missing)}.")
            position = 0
            for batch in parquet.iter_batches(batch_size=1024, columns=fields):
                for row in batch.to_pylist():
                    yield position, row
                    position += 1
        finally:
            parquet.close()
    elif source["format"] == "tar_text":
        try:
            with tarfile.open(path, mode="r:*") as archive:
                try:
                    member = archive.getmember(source["member"])
                except KeyError as exc:
                    raise ValueError(f"Benchmark archive member not found: {source['member']}.") from exc
                if not member.isfile():
                    raise ValueError("Benchmark archive member must be a regular file.")
                if member.size < 0 or member.size > max_member_bytes:
                    raise ValueError("Benchmark archive member exceeds its byte limit.")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("Benchmark archive member could not be read.")
                with io.TextIOWrapper(extracted, encoding="utf-8") as lines:
                    for position, line in enumerate(lines):
                        yield position, {fields[0]: line.rstrip("\r\n")}
        except tarfile.TarError as exc:
            raise ValueError("Invalid benchmark tar archive.") from exc
    else:
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, [])
            if len(header) != len(set(header)) or not set(fields).issubset(header):
                raise ValueError("Benchmark TSV header has duplicate or missing text columns.")
            indices = {field: header.index(field) for field in fields}
            for position, values in enumerate(reader):
                if not values:
                    continue
                if len(values) != len(header):
                    raise ValueError("Benchmark TSV row does not match its header length.")
                yield position, {field: values[index] for field, index in indices.items()}


def _texts(row: dict, source: dict):
    for field in source["text_fields"]:
        value = row[field]
        if isinstance(value, dict):
            for language in sorted(value):
                if language in LANGUAGES:
                    yield field, language, value[language]
        else:
            if source["language"] is None:
                raise ValueError("Expected a translation dictionary containing language/text values.")
            yield field, source["language"], value


def fetch_benchmarks(
    sources_path: str | Path,
    output_dir: str | Path,
    *,
    max_file_bytes: int = 64 * 1024 * 1024,
    max_total_bytes: int = 512 * 1024 * 1024,
    timeout: float = 30.0,
    retries: int = 2,
    cache_dir: str | Path | None = None,
) -> dict:
    """Fetch a pinned file list and publish exact benchmark exclusion texts.

    Downloads are retained with hashes. Duplicate exact UTF-8 texts are emitted
    once per benchmark family; first occurrence retains its split provenance.
    Translation dictionaries emit both source and target language strings.
    """
    sources_path, output_dir = Path(sources_path).resolve(), Path(output_dir).resolve()
    cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; use a new directory.")
    if any(type(value) is not int or value < 1 for value in (max_file_bytes, max_total_bytes)):
        raise ValueError("Download byte limits must be positive integers.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
        raise ValueError("timeout must be positive and at most 300 seconds.")
    if type(retries) is not int or not 0 <= retries <= 5:
        raise ValueError("retries must be an integer from 0 to 5.")
    lock_hash = file_sha256(sources_path)
    lock = json.loads(sources_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(lock.get("benchmarks"), list) or not lock["benchmarks"]:
        raise ValueError("Source lock must contain a nonempty benchmarks list.")
    sources = [_source(entry) for entry in lock["benchmarks"]]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        root = Path(temporary).resolve()
        if root.parent != output_dir.parent:
            raise ValueError("Temporary directory escaped the output parent.")
        downloads = root / "downloads"
        downloads.mkdir()
        shutil.copyfile(sources_path, root / "sources.lock.json")
        downloaded = {}
        seen = defaultdict(set)
        coverage = {}
        emitted_coverage = {}
        source_results = []
        total_bytes = 0
        total_texts = 0
        with (root / "exclusions.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for source_index, source in enumerate(sources):
                url = source["source_url"]
                if url not in downloaded:
                    remaining = max_total_bytes - total_bytes
                    if remaining < 1:
                        raise ValueError("Benchmark downloads exceed the total byte limit.")
                    local = f"downloads/{source_index:03d}-{source['benchmark']}.{source['format']}"
                    details = _acquire(url, root / local, max_bytes=min(max_file_bytes, remaining),
                                       timeout=float(timeout), retries=retries,
                                       expected_sha256=source["expected_sha256"], cache_dir=cache_dir)
                    # Validate the downloaded file independently of the transport.
                    size = (root / local).stat().st_size
                    if size > min(max_file_bytes, remaining):
                        raise ValueError("Benchmark download exceeds its byte limit.")
                    downloaded[url] = {**details, "local_file": local, "bytes": size, "sha256": file_sha256(root / local)}
                    total_bytes += size
                details = downloaded[url]
                if source["expected_sha256"] is not None and details["sha256"] != source["expected_sha256"]:
                    raise ValueError("Shared benchmark artifact SHA-256 does not match expected_sha256.")
                counts = {"rows_scanned": 0, "candidate_texts": 0, "emitted_texts": 0, "duplicate_texts": 0, "empty_texts": 0}
                local_coverage = {}
                local_seen = {}
                for row_index, row in _rows(root / details["local_file"], source, max_member_bytes=max_file_bytes):
                    counts["rows_scanned"] += 1
                    for field, language, text in _texts(row, source):
                        counts["candidate_texts"] += 1
                        if text is None or (isinstance(text, str) and not text.strip()):
                            counts["empty_texts"] += 1
                            continue
                        if not isinstance(text, str):
                            raise ValueError("Declared benchmark text fields must contain strings or language/string dictionaries.")
                        benchmark = source["benchmark"]
                        local_seen[language] = local_seen.get(language, 0) + 1
                        seen_language_counts = coverage.setdefault(benchmark, {}).setdefault(language, {})
                        seen_language_counts[source["split"]] = seen_language_counts.get(source["split"], 0) + 1
                        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if text_hash in seen[benchmark]:
                            counts["duplicate_texts"] += 1
                            continue
                        seen[benchmark].add(text_hash)
                        record = {"text": text, "benchmark": benchmark,
                                  "id": f"{benchmark}:{source_index}:{row_index}:{field}:{language}",
                                  "source": url, "revision": source["revision"], "split": source["split"], "language": language}
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        counts["emitted_texts"] += 1
                        total_texts += 1
                        local_coverage[language] = local_coverage.get(language, 0) + 1
                        family = emitted_coverage.setdefault(benchmark, {})
                        language_counts = family.setdefault(language, {})
                        language_counts[source["split"]] = language_counts.get(source["split"], 0) + 1
                source_results.append({**source, **details, **counts, "nonempty_texts_by_language": local_seen,
                                       "emitted_texts_by_language": local_coverage})
        if file_sha256(sources_path) != lock_hash or file_sha256(root / "sources.lock.json") != lock_hash:
            raise ValueError("Source lock changed while benchmark collection was running.")
        missing_scope = []
        for family, languages in _INTENDED_SCOPE.items():
            for language, splits in languages.items():
                for split in splits:
                    if not coverage.get(family, {}).get(language, {}).get(split):
                        missing_scope.append({"benchmark": family, "language": language, "split": split})
        files = {}
        for path in (root / "sources.lock.json", root / "exclusions.jsonl", *(root / item["local_file"] for item in downloaded.values())):
            files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        manifest = {
            "schema_version": 1, "purpose": "pinned public text exclusions for bounded benchmark overlap auditing",
            "source_lock": {"path": str(sources_path), "sha256": lock_hash},
            "download_limits": {"max_file_bytes": max_file_bytes, "max_total_bytes": max_total_bytes,
                                "timeout_seconds": float(timeout), "retries_after_first_attempt": retries},
            "unique_downloads": len(downloaded), "downloaded_bytes": total_bytes,
            "network_downloaded_bytes": sum(item["bytes"] for item in downloaded.values() if not item.get("cache_hit")),
            "all_artifacts_predeclared_pinned": all(source["pinning"] != "unversioned_url_recorded_after_fetch" for source in sources),
            "text_records": total_texts, "deduplication": "exact UTF-8 text once per benchmark family; first source/split retained",
            "labels": "only declared text columns are selected; benchmark labels are neither inspected nor analyzed",
            "coverage": coverage, "coverage_counting": "nonempty text fields before exact deduplication, by family/language/split",
            "emitted_coverage": emitted_coverage, "sources": source_results, "files": files,
            "scope": {"intended": _INTENDED_SCOPE, "missing_or_zero_text_units": missing_scope,
                      "complete_for_declared_file_list": True,
                      "unversioned_sources_without_expected_sha256": [source["source_url"] for source in sources if source["pinning"] == "unversioned_url_recorded_after_fetch"],
                      "limitations": [
                          "Pinned files define the fetched scope; this is not a comprehensive contamination guarantee.",
                          "A publisher archive release label is documentary metadata; only expected_sha256 pins its bytes before fetching, while the manifest records the observed SHA-256.",
                          "FLORES dev/devtest covers four selected languages; XNLI validation/test covers en/de/ar, not Persian.",
                          "MNLI scope is public validation matched/mismatched; training and hidden test text are not claimed excluded.",
                          "OPUS100 scope is validation/test text in the selected English-centered language pairs; FarsTail scope is all three public splits.",
                          "Exact-text collection does not detect paraphrases, near duplicates, translations or contamination in private/unavailable sources.",
                          "Deduplication across splits retains only the first provenance record; per-source counts report duplicates and the original pinned files remain available.",
                      ]},
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        root.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, help="Persistent verified URL-keyed raw artifact cache")
    args = parser.parse_args()
    manifest = fetch_benchmarks(args.sources, args.output, max_file_bytes=args.max_file_bytes,
                                max_total_bytes=args.max_total_bytes, timeout=args.timeout, retries=args.retries,
                                cache_dir=args.cache_dir)
    print(json.dumps({"output": str(args.output.resolve()), "downloaded_bytes": manifest["downloaded_bytes"],
                      "text_records": manifest["text_records"], "coverage": manifest["coverage"],
                      "scope": manifest["scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
