"""Fetch a bounded prefix of immutable corpus shards using HTTP range reads.

This starter selection is for pipeline/GPU readiness, not a representative
sample of the full corpora. Corpus screening and token balancing are separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tempfile

from .provenance import file_sha256


class BoundedReader:
    """Bound bytes requested by Parquet, without downloading a whole shard."""
    def __init__(self, wrapped, limit: int):
        self.wrapped, self.limit, self.bytes_read = wrapped, limit, 0

    @property
    def closed(self):
        return self.wrapped.closed

    def read(self, size=-1):
        if size < 0 or size > self.limit - self.bytes_read:
            raise ValueError("Corpus range-read budget exceeded; increase --max-read-bytes explicitly.")
        result = self.wrapped.read(size)
        self.bytes_read += len(result)
        return result

    def seek(self, *args):
        return self.wrapped.seek(*args)

    def tell(self):
        return self.wrapped.tell()

    def readable(self):
        return True

    def seekable(self):
        return True


def _iter_rows(source: dict, max_read_bytes: int):
    try:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install the corpus extra: pip install -e '.[corpus]'.") from exc
    fs = HfFileSystem(token=False)
    for shard in source["files"]:
        name = shard["path"] if isinstance(shard, dict) else shard
        remote = f"datasets/{source['repo']}@{source['revision']}/{name}"
        with fs.open(remote, "rb", block_size=1_048_576, cache_type="readahead") as handle:
            reader = BoundedReader(handle, max_read_bytes)
            parquet = pq.ParquetFile(reader)
            columns = [column for column in ("text", "id", "url", "language_score")
                       if column in parquet.schema_arrow.names]
            if "text" not in columns:
                raise ValueError(f"Source shard has no text column: {remote}")
            row_number = 0
            for batch in parquet.iter_batches(batch_size=128, columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    yield row, {"path": name, "row": row_number,
                                "parquet_bytes_requested": reader.bytes_read}
                    row_number += 1


def _validate_lock(lock: dict) -> list[dict]:
    sources = lock.get("corpus", [])
    if not isinstance(sources, list) or {s.get("language") for s in sources} != {"en", "de", "ar", "fa"} or len(sources) != 4:
        raise ValueError("Source lock must contain exactly one corpus source for en, de, ar and fa.")
    for source in sources:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", source.get("revision", "")):
            raise ValueError("Corpus revisions must be immutable 40-character commit SHAs.")
        if not source.get("repo") or not source.get("files"):
            raise ValueError("Each corpus source requires a repo and explicit shard files.")
        for shard in source["files"]:
            name = shard.get("path") if isinstance(shard, dict) else shard
            if not isinstance(name, str) or not name.endswith(".parquet") or ".." in name.split("/") or name.startswith("/"):
                raise ValueError("Corpus shard paths must be relative Parquet paths.")
    return sources


def fetch_corpus(sources_path: str | Path, output_dir: str | Path, *, bytes_per_language: int = 5_242_880,
                 min_document_bytes: int = 256, max_document_bytes: int = 131_072,
                 max_read_bytes: int = 134_217_728, workers: int = 2,
                 languages: tuple[str, ...] | None = None) -> dict:
    """Fetch whole documents until each language reaches its UTF-8 byte target.

    The read budget counts Parquet read requests per shard, not HTTP overhead or
    the range reader's small read-ahead cache. No full shard hash is verified.
    Immutable repository revisions and shard LFS hashes identify remote inputs;
    SHA256 of every selected local JSONL is recorded and later rechecked.
    """
    if bytes_per_language <= 0 or min_document_bytes <= 0 or max_document_bytes < min_document_bytes or max_read_bytes <= 0 or not 1 <= workers <= 4:
        raise ValueError("Invalid corpus byte limits or worker count.")
    sources_path, output_dir = Path(sources_path), Path(output_dir)
    lock = json.loads(sources_path.read_text(encoding="utf-8"))
    sources = _validate_lock(lock)
    if languages is not None:
        if not languages or len(set(languages)) != len(languages) or not set(languages).issubset({"en", "de", "ar", "fa"}):
            raise ValueError("Selected languages must be a nonempty unique subset of en/de/ar/fa.")
        sources = [source for source in sources if source["language"] in languages]
    if output_dir.exists():
        raise ValueError("Corpus output already exists; choose a new output directory.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".tmp-", dir=output_dir.parent) as temporary:
        root = Path(temporary)

        def fetch_one(source):
            language = source["language"]
            destination = root / f"{language}.jsonl"
            accepted_bytes, count, scanned, rejected = 0, 0, 0, 0
            last_location = None
            rows = _iter_rows(source, max_read_bytes)
            try:
                with destination.open("w", encoding="utf-8", newline="\n") as out:
                    for row, location in rows:
                        scanned += 1
                        text = row.get("text")
                        length = len(text.encode("utf-8")) if isinstance(text, str) else 0
                        if length < min_document_bytes or length > max_document_bytes or not text.strip():
                            rejected += 1
                            continue
                        original_id = row.get("id") or hashlib.sha256(text.encode("utf-8")).hexdigest()
                        record = {"id": f"{source['repo']}:{language}:{original_id}", "text": text,
                                  "language": language, "source": source["repo"], "revision": source["revision"],
                                  "url": row.get("url"), "source_shard": location["path"],
                                  "source_row": location["row"], "source_language_score": row.get("language_score")}
                        out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        accepted_bytes += length
                        count += 1
                        last_location = location
                        if accepted_bytes >= bytes_per_language:
                            break
            finally:
                close = getattr(rows, "close", None)
                if close is not None:
                    close()
            if accepted_bytes < bytes_per_language:
                raise ValueError(f"Explicit shards exhausted for {language}: {accepted_bytes} < {bytes_per_language} bytes.")
            result = {"language": language, "documents": count, "utf8_bytes": accepted_bytes,
                      "rows_scanned": scanned, "rejected_length_or_empty": rejected,
                      "last_source_location": last_location, "file": destination.name,
                      "sha256": file_sha256(destination), "source": source}
            print(json.dumps({"event": "language_fetched", "language": language,
                              "documents": count, "utf8_bytes": accepted_bytes}), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            fetched = list(executor.map(fetch_one, sources))
        (root / "sources.json").write_bytes(sources_path.read_bytes())
        manifest = {"schema_version": 1, "scope": "bounded real-data readiness sample",
                    "selection": "deterministic prefix of explicit immutable shards; not globally representative",
                    "bytes_per_language_target": bytes_per_language, "min_document_bytes": min_document_bytes,
                    "max_document_bytes": max_document_bytes, "max_parquet_read_bytes_per_shard": max_read_bytes,
                    "sources_sha256": file_sha256(sources_path), "languages": fetched,
                    "remote_hash_verification": "immutable revision and recorded shard metadata; full remote shards not downloaded/hashed",
                    "contamination_screened": False}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        root.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bytes-per-language", type=int, default=5_242_880)
    parser.add_argument("--max-read-bytes", type=int, default=134_217_728)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--languages", nargs="+", choices=("en", "de", "ar", "fa"))
    args = parser.parse_args()
    fetch_corpus(args.sources, args.output, bytes_per_language=args.bytes_per_language,
                 max_read_bytes=args.max_read_bytes, workers=args.workers,
                 languages=tuple(args.languages) if args.languages else None)


if __name__ == "__main__":
    main()
