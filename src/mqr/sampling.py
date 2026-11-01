"""Reproducible, bounded diagnostic sampling of immutable Parquet sources.

This is a stratified source audit, not a representative unweighted web sample.
Every sampled row retains original text and its complete selection provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
from typing import Callable

from .provenance import file_sha256


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _write_immutable(path: Path, value):
    encoded = (_dump(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)


def _rng(seed: int, *parts: str):
    digest = hashlib.sha256(_dump([seed, *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def choose_shards(files: list[dict], count: int, seed: int, source_key: str) -> list[dict]:
    """Uniform sample without replacement, independent of inventory input order."""
    if count <= 0 or not files or len({f["path"] for f in files}) != len(files):
        raise ValueError("Need positive count and a nonempty unique file inventory.")
    ordered = sorted(files, key=lambda f: f["path"])
    selected = _rng(seed, source_key, "shards").sample(ordered, min(count, len(ordered)))
    return sorted(selected, key=lambda f: f["path"])


def choose_windows(row_group_rows: list[int], window_rows: int, seed: int,
                   source_key: str, shard_path: str) -> list[dict]:
    """Choose one group uniformly per third and one contiguous window uniformly.

    The row-group thirds use floor boundaries. Empty thirds are explicit and
    yield no sample. A row's conditional inclusion is generally unequal near
    the group edges; ``row_inclusion_probability`` supplies the exact value.
    """
    if window_rows <= 0 or not row_group_rows or any(n <= 0 for n in row_group_rows):
        raise ValueError("Row groups and window size must be positive.")
    output = []
    n_groups = len(row_group_rows)
    for index, name in enumerate(("head", "middle", "tail")):
        low, high = n_groups * index // 3, n_groups * (index + 1) // 3
        if low == high:
            continue
        rng = _rng(seed, source_key, shard_path, name)
        group = rng.randrange(low, high)
        size = row_group_rows[group]
        width = min(window_rows, size)
        start = rng.randrange(size - width + 1)
        output.append({"stratum": name, "row_group_start": low,
                       "row_group_stop_exclusive": high,
                       "row_group": group, "row_group_rows": size,
                       "row_group_first_row": sum(row_group_rows[:group]),
                       "window_start": start, "window_rows": width,
                       "row_group_selection_probability": 1 / (high - low),
                       "window_start_probability_given_group": 1 / (size - width + 1)})
    return output


def row_inclusion_probability(group_rows: int, width: int, row: int) -> float:
    """Probability that a uniformly placed fixed window contains this row."""
    if not 0 <= row < group_rows or not 0 < width <= group_rows:
        raise ValueError("Invalid row/window.")
    possible = group_rows - width + 1
    containing = max(0, min(row, possible - 1) - max(0, row - width + 1) + 1)
    return containing / possible


class ReadBudgetExceeded(ValueError):
    pass


class BudgetReader:
    """No read-ahead: bound logical read requests, including failed requests.

    Counts are persisted before delegation; acquisition retries never reclaim
    a failed request. Internal HTTP retries, headers/TLS and server-side
    overhead are outside this logical Parquet-request bound.
    """
    def __init__(self, wrapped, limit: int, *, used: int = 0,
                 reserve: Callable[[int], None] | None = None):
        self.wrapped, self.limit, self.bytes_requested = wrapped, limit, used
        self.reserve = reserve
        if limit <= 0 or not 0 <= used <= limit:
            raise ValueError("Invalid read budget.")

    @property
    def closed(self):
        return self.wrapped.closed

    def read(self, size=-1):
        if size < 0 or size > self.limit - self.bytes_requested:
            raise ReadBudgetExceeded("Parquet logical-read budget exceeded; no substitute is selected.")
        if self.reserve:
            self.reserve(size)
        self.bytes_requested += size
        return self.wrapped.read(size)

    def seek(self, *args):
        return self.wrapped.seek(*args)

    def tell(self):
        return self.wrapped.tell()

    def readable(self):
        return True

    def seekable(self):
        return True


def enumerate_inventory(source: dict, *, api=None) -> dict:
    """Lock complete publisher-listed file metadata within a declared prefix."""
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("Revision must be an immutable lowercase commit SHA.")
    prefix = source["prefix"]
    if prefix.startswith("/") or ".." in prefix.split("/"):
        raise ValueError("Unsafe source prefix.")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=False)
    files = []
    for item in api.list_repo_tree(source["repo"], repo_type="dataset",
                                   revision=source["revision"], path_in_repo=prefix,
                                   recursive=False, expand=False):
        if not item.path.endswith(".parquet"):
            continue
        lfs = getattr(item, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if isinstance(lfs, dict):
            sha = lfs.get("sha256")
        files.append({"path": item.path, "size": int(item.size),
                      "lfs_sha256": sha, "blob_id": getattr(item, "blob_id", None)})
    files.sort(key=lambda f: f["path"])
    if not files or len({f["path"] for f in files}) != len(files):
        raise ValueError("Empty or duplicate Parquet inventory.")
    return {"source": source, "files": files, "inventory_files": len(files),
            "scope": "all publisher-listed Parquet files under declared prefix at pinned revision",
            "remote_hash_verification": "publisher hashes identify inputs; full shards not downloaded/hashed"}


def default_sources(primary_path, supplement_path) -> list[dict]:
    primary = json.loads(Path(primary_path).read_text(encoding="utf-8"))["corpus"]
    supplement = json.loads(Path(supplement_path).read_text(encoding="utf-8"))["corpus"]
    prefixes = {"en": "sample/10BT", "de": "data/deu_Latn/train",
                "ar": "data/arb_Arab/train", "fa": "data/fas_Arab/train"}
    result = [{"key": s["language"] + "__web", "language": s["language"],
               "kind": "web", "repo": s["repo"], "revision": s["revision"],
               "prefix": prefixes[s["language"]], "shards": 8, "window_rows": 256,
               "max_read_bytes_per_shard": 128 * 1024**2} for s in primary]
    native = next(s for s in supplement if s["language"] == "fa")
    result.append({"key": "fa__native", "language": "fa", "kind": "native",
                   "repo": native["repo"], "revision": native["revision"],
                   "prefix": "20231101.fa", "shards": 4, "window_rows": 128,
                   "max_read_bytes_per_shard": 128 * 1024**2})
    return result


def _acquire(sources: list[dict], output_dir: str | Path, *, seed: int = 20261023,
             workers: int = 4) -> dict:
    """Resume exact selected strata; immutable completion evidence is rechecked."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable(root / "request.json", {"sources": sources, "seed": seed,
                     "selection": "uniform distinct shards; uniform group per third; uniform contiguous window",
                     "scope": "diagnostic stratified source audit, not an unweighted corpus prevalence estimate"})
    event_lock = threading.Lock()

    def event(value):
        with event_lock, (root / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_dump(value) + "\n")

    tasks = []
    inventories = []
    for source in sources:
        path = root / "inventory" / (source["key"] + ".json")
        if path.exists():
            inventory = json.loads(path.read_text(encoding="utf-8"))
            if inventory["source"] != source:
                raise ValueError("Source inventory request changed.")
        else:
            inventory = enumerate_inventory(source)
            _write_immutable(path, inventory)
        inventories.append(inventory)
        selected = choose_shards(inventory["files"], source["shards"], seed, source["key"])
        _write_immutable(root / "selection" / (source["key"] + ".json"),
                         {"inventory_sha256": file_sha256(path), "selected": selected,
                          "shard_selection_probability": len(selected) / len(inventory["files"])})
        tasks.extend((source, shard, index, len(selected) / len(inventory["files"]))
                     for index, shard in enumerate(selected))
        print(_dump({"event": "inventory_locked", "source": source["key"],
                     "inventory_shards": len(inventory["files"]), "selected_shards": len(selected)}), flush=True)

    def fetch_shard(task):
        source, shard, ordinal, shard_probability = task
        key = f"{source['key']}__{ordinal:02d}"
        done_path = root / "shards" / (key + ".json")
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            metadata = root / "metadata" / (key + ".json")
            if file_sha256(metadata) != done["metadata_sha256"]:
                raise ValueError("Completed shard metadata changed.")
            for item in done["strata"]:
                if file_sha256(root / item["file"]) != item["sha256"]:
                    raise ValueError("Completed stratum content changed.")
            return done
        usage_path = root / "usage" / (key + ".jsonl")
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        prior = sum(json.loads(line)["requested"] for line in usage_path.read_text(encoding="utf-8").splitlines()) if usage_path.exists() else 0

        def reserve(size):
            with usage_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(_dump({"requested": size}) + "\n")
                handle.flush()

        metadata_path = root / "metadata" / (key + ".json")
        fs = HfFileSystem(token=False)
        remote = f"datasets/{source['repo']}@{source['revision']}/{shard['path']}"
        strata = []
        try:
            # A zero block size selects HfFileSystem's non-seekable streaming
            # reader. A positive size with cache_type=none preserves seeks and
            # makes each requested range explicit without read-ahead caching.
            with fs.open(remote, "rb", block_size=1_048_576, cache_type="none") as wrapped:
                reader = BudgetReader(wrapped, source["max_read_bytes_per_shard"], used=prior, reserve=reserve)
                parquet = pq.ParquetFile(reader, pre_buffer=False)
                row_counts = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
                columns = [c for c in ("text", "id", "url", "language_score", "title") if c in parquet.schema_arrow.names]
                if "text" not in columns:
                    raise ValueError("Missing text column.")
                metadata = {"source": source, "publisher_shard": shard,
                            "num_rows": parquet.metadata.num_rows, "row_group_rows": row_counts,
                            "columns": parquet.schema_arrow.names,
                            "row_group_compressed_bytes": [sum(parquet.metadata.row_group(i).column(j).total_compressed_size for j in range(parquet.metadata.num_columns)) for i in range(parquet.num_row_groups)]}
                _write_immutable(metadata_path, metadata)
                windows = choose_windows(row_counts, source["window_rows"], seed, source["key"], shard["path"])
                _write_immutable(root / "windows" / (key + ".json"), windows)
                for window in windows:
                    stem = key + "__" + window["stratum"]
                    path = root / "strata" / (stem + ".jsonl")
                    marker = root / "strata" / (stem + ".complete.json")
                    if marker.exists():
                        completed = json.loads(marker.read_text(encoding="utf-8"))
                        if file_sha256(path) != completed["sha256"]:
                            raise ValueError("Completed stratum content changed.")
                        strata.append(completed)
                        continue
                    event({"event": "stratum_started", "source": source["key"], "shard": shard["path"], **window})
                    before = reader.bytes_requested
                    table = parquet.read_row_group(window["row_group"], columns=columns, use_threads=False)
                    rows = table.slice(window["window_start"], window["window_rows"]).to_pylist()
                    output_rows = []
                    for i, row in enumerate(rows):
                        text = row.get("text")
                        group_row = window["window_start"] + i
                        global_row = window["row_group_first_row"] + group_row
                        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None
                        record = {"id": f"{source['repo']}@{source['revision']}:{shard['path']}:{global_row}",
                                  "text": text, "language": source["language"], "source": source["repo"],
                                  "source_kind": source["kind"], "revision": source["revision"],
                                  "url": row.get("url"), "source_id": row.get("id"), "source_title": row.get("title"),
                                  "source_shard": shard["path"], "source_row": global_row,
                                  "source_row_group": window["row_group"], "source_language_score": row.get("language_score"),
                                  "text_sha256": text_sha,
                                  "sampling": {**window, "seed": seed, "shard_selection_probability": shard_probability,
                                      "row_inclusion_probability_given_group": row_inclusion_probability(window["row_group_rows"], window["window_rows"], group_row),
                                      "scope": "declared inventory, group third, fixed-window design; no unweighted global prevalence claim"}}
                        output_rows.append(_dump(record))
                    encoded = ("\n".join(output_rows) + "\n").encode("utf-8")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        if path.read_bytes() != encoded:
                            raise ValueError("Unmarked stratum differs on deterministic replay.")
                    else:
                        with path.open("xb") as handle:
                            handle.write(encoded)
                    completed = {"file": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                                 "rows": len(rows), "utf8_text_bytes": sum(len(r["text"].encode("utf-8")) for r in rows if isinstance(r.get("text"), str)),
                                 "parquet_logical_bytes_requested": reader.bytes_requested - before, "window": window}
                    _write_immutable(marker, completed)
                    strata.append(completed)
                    event({"event": "stratum_completed", "source": source["key"], "shard": shard["path"], **completed})
                    print(_dump({"event": "stratum_completed", "source": source["key"], "shard": ordinal, "stratum": window["stratum"], "rows": len(rows), "requested_bytes": reader.bytes_requested}), flush=True)
            result = {"key": key, "source": source, "shard": shard,
                      "metadata_sha256": file_sha256(metadata_path), "strata": strata,
                      "parquet_logical_bytes_requested": reader.bytes_requested, "status": "complete"}
            _write_immutable(done_path, result)
            return result
        except Exception as exc:
            # Preserve the category only: remote error messages can contain signed URLs.
            event({"event": "shard_failed", "key": key, "error_type": type(exc).__name__,
                   "status": "incomplete", "substitution": False})
            print(_dump({"event": "shard_failed", "key": key, "error_type": type(exc).__name__}), flush=True)
            return {"key": key, "source": source, "shard": shard, "status": "incomplete",
                    "error_type": type(exc).__name__, "strata": strata,
                    "parquet_logical_bytes_requested": sum(json.loads(line)["requested"] for line in usage_path.read_text(encoding="utf-8").splitlines()) if usage_path.exists() else 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_shard, tasks))
    complete = all(result["status"] == "complete" for result in results)
    raw_files = []
    if complete:
        for source in sources:
            path = root / "raw" / (source["key"] + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            chunks = [root / item["file"] for result in results if result["source"]["key"] == source["key"] for item in result["strata"]]
            encoded = b"".join(part.read_bytes() for part in chunks)
            if path.exists():
                if path.read_bytes() != encoded:
                    raise ValueError("Merged raw file changed.")
            else:
                with path.open("xb") as handle:
                    handle.write(encoded)
            raw_files.append({"source": source["key"], "file": path.relative_to(root).as_posix(),
                              "sha256": file_sha256(path), "bytes": path.stat().st_size,
                              "rows": sum(r["rows"] for result in results if result["source"]["key"] == source["key"] for r in result["strata"])})
    report = {"schema_version": 1, "status": "complete" if complete else "incomplete",
              "scope": "bounded diagnostic stratified source audit; no global representativeness claim",
              "request_sha256": file_sha256(root / "request.json"), "seed": seed,
              "logical_read_accounting": "requests persisted before reading; includes acquisition retries; no client read-ahead; excludes internal HTTP retries and HTTP/TLS overhead",
              "parquet_logical_bytes_requested": sum(r["parquet_logical_bytes_requested"] for r in results),
              "shards": results, "raw_files": raw_files}
    if complete:
        _write_immutable(root / "manifest.json", report)
    else:
        (root / "incomplete.json").write_text(_dump(report) + "\n", encoding="utf-8")
    return report


def acquire(sources: list[dict], output_dir: str | Path, *, seed: int = 20261023,
            workers: int = 4) -> dict:
    """Acquire with one writer; stale locks require explicit operator review.

    Resume uses exactly the original request, frozen plans, completed stratum
    hashes and cumulative logical-read reservations. It never substitutes a
    different shard, row group, prefix or seed for a failed stratum.
    """
    if not sources or not 1 <= workers <= 8:
        raise ValueError("Need sources and between one and eight workers.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "acquisition.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ValueError("Acquisition writer lock exists; verify its owner before resuming.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_dump({"pid": os.getpid()}) + "\n")
        return _acquire(sources, root, seed=seed, workers=workers)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-sources", default="configs/data/readiness_sources.json")
    parser.add_argument("--supplement-sources", default="configs/data/readiness_supplement_sources.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    report = acquire(default_sources(args.primary_sources, args.supplement_sources), args.output,
                     seed=args.seed, workers=args.workers)
    print(_dump({"event": "acquisition_finished", "status": report["status"],
                 "requested_bytes": report["parquet_logical_bytes_requested"]}), flush=True)
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
