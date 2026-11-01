"""Select an auditable bounded corpus from pinned local document JSONL files.

This is a sample preparation tool, not a complete web-corpus quality or benchmark
contamination audit. It never downloads data and never changes document text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

from .provenance import file_sha256

LANGUAGES = ("en", "de", "ar", "fa")
SPLITS = ("train", "validation", "test")
EXPECTED_BENCHMARKS = ("flores200", "xnli", "farstail", "opus100")
_PRIME = (1 << 61) - 1


def normalize_text(text: str) -> str:
    """Comparison only: NFC, case folding and whitespace folding."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def canonical_url(url: str | None) -> str | None:
    """Conservatively identify the same HTTP page across upstream snapshots.

    Paths remain case sensitive. Encoded reserved separators remain escaped,
    while Unicode path spellings normalize to the same NFC representation.
    """
    if not url or not url.strip():
        return None
    try:
        parts = urlsplit(url.strip())
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname or parts.username is not None:
            return None
        host = parts.hostname.lower().removeprefix("www.")
        port = parts.port
        if ":" in host:
            host = f"[{host}]"
        if port is not None and port not in (80, 443):
            host += f":{port}"

        def decode_path_run(match):
            decoded = bytes.fromhex(match.group().replace("%", "")).decode("utf-8")
            return "".join(character if ord(character) > 127 or character.isalnum() or character in "-._~ "
                           else f"%{ord(character):02X}" for character in decoded)

        path = parts.path or "/"
        if re.search(r"%(?![0-9a-fA-F]{2})", path):
            return None
        path = unicodedata.normalize("NFC", re.sub(r"(?:%[0-9a-fA-F]{2})+", decode_path_run, path))
        plain_host = parts.hostname.lower().removeprefix("www.")
        if (plain_host == "wikipedia.org" or plain_host.endswith(".wikipedia.org")) and path.startswith("/wiki/"):
            path = path.replace("_", " ")
        path = quote(path, safe="/:@!$&'()*+,;=-._~%")
        query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True, errors="strict")
                 if not key.casefold().startswith("utm_") and key.casefold() not in ("fbclid", "gclid")]
        query_string = urlencode(sorted(query))
        return f"https://{host}{path}" + (f"?{query_string}" if query_string else "")
    except (ValueError, UnicodeError):
        # Unparseable/non-HTTP addresses receive no identity, rather than an
        # ambiguous key that might remove an unrelated document.
        return None


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize_text(text), flags=re.UNICODE)


def _shingles(words: Sequence[str], size: int) -> set[int]:
    if not words:
        return set()
    size = min(size, len(words))
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[index:index + size]).encode("utf-8"), digest_size=8).digest(),
            "little",
        ) % _PRIME
        for index in range(len(words) - size + 1)
    }


def _document_hash(seed: int, language: str, document_id: str) -> str:
    key = json.dumps([seed, language, document_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    document_id: str
    language: str
    source: str
    revision: str
    source_split: str | None
    url: str | None
    path: Path
    offset: int
    text_sha256: str
    utf8_bytes: int
    normalized_sha256: str


def _read_text(handles: dict[Path, BinaryIO], document: Document) -> str:
    handle = handles[document.path]
    handle.seek(document.offset)
    text = json.loads(handle.readline())["text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document.text_sha256:
        raise ValueError("A raw input document changed during corpus selection.")
    return text


def _scan_inputs(paths: list[Path]) -> tuple[list[Document], dict, Counter]:
    documents: list[Document] = []
    seen_ids: dict[str, Document] = {}
    sources: dict[str, dict] = {}
    drops: Counter = Counter()
    for path in paths:
        with path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Document at {path}:{line_number} must be an object.")
                for field in ("id", "language", "text", "source", "revision"):
                    if not isinstance(record.get(field), str) or not record[field].strip():
                        raise ValueError(f"Document at {path}:{line_number} needs a nonempty '{field}'.")
                if record["language"] not in LANGUAGES:
                    raise ValueError(f"Unsupported corpus language {record['language']!r}; expected {LANGUAGES}.")
                if re.fullmatch(r"[0-9a-fA-F]{40}", record["revision"]) is None:
                    raise ValueError("Corpus revision must be an immutable 40-character hexadecimal SHA.")
                for field in ("url", "split"):
                    if record.get(field) is not None and not isinstance(record[field], str):
                        raise ValueError(f"Optional document '{field}' must be a string.")
                encoded = record["text"].encode("utf-8")
                document = Document(
                    record["id"], record["language"], record["source"], record["revision"].lower(),
                    record.get("split"), record.get("url"), path, offset,
                    hashlib.sha256(encoded).hexdigest(), len(encoded),
                    hashlib.sha256(normalize_text(record["text"]).encode("utf-8")).hexdigest(),
                )
                source_key = f"{document.source}@{document.revision}"
                source_stats = sources.setdefault(source_key, {"source": document.source, "revision": document.revision,
                                                              "input_documents": 0, "input_utf8_bytes": 0})
                source_stats["input_documents"] += 1
                source_stats["input_utf8_bytes"] += document.utf8_bytes
                previous = seen_ids.get(document.document_id)
                if previous is not None:
                    if (previous.text_sha256, previous.language) != (document.text_sha256, document.language):
                        raise ValueError(f"Document ID reused for different text or language: {document.document_id!r}.")
                    drops["duplicate_id"] += 1
                    continue
                seen_ids[document.document_id] = document
                documents.append(document)
    return documents, sources, drops


def _assign_splits(
    documents: Sequence[Document], seed: int, train_fraction: float, validation_fraction: float
) -> dict[str, str]:
    by_language: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_language[document.language].append(document)
    assignment = {}
    for language in LANGUAGES:
        ordered = sorted(by_language[language], key=lambda doc: _document_hash(seed, language, doc.document_id))
        count = len(ordered)
        if count < 3:
            raise ValueError(f"Need at least three raw documents for language {language!r} to assign all splits.")
        train_count = min(count - 2, max(1, int(count * train_fraction)))
        validation_count = min(count - train_count - 1, max(1, int(count * validation_fraction)))
        for index, document in enumerate(ordered):
            split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
            assignment[document.document_id] = split
    return assignment


def _benchmark_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class BenchmarkExclusions:
    def __init__(self, path: str | Path | None, ngram_size: int, expected: Sequence[str]) -> None:
        self.ngram_size = ngram_size
        self.exact: set[str] = set()
        self.ngrams: dict[int, set[int]] = defaultdict(set)
        self.files: list[dict] = []
        self.documents: Counter = Counter()
        self.upstream_sources: dict[tuple[str, str, str], int] = defaultdict(int)
        self.short_exact_only = 0
        self.metadata_without_revision = 0
        if path is not None:
            path = Path(path).resolve()
            paths = sorted(candidate for candidate in path.rglob("*") if candidate.suffix in (".jsonl", ".txt")) if path.is_dir() else [path]
            for input_path in paths:
                if input_path.suffix not in (".jsonl", ".txt"):
                    raise ValueError("Benchmark exclusions must be JSONL, plain text, or a directory containing those files.")
                self.files.append({"path": str(input_path), "sha256": file_sha256(input_path), "bytes": input_path.stat().st_size})
                with input_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        record = json.loads(line) if input_path.suffix == ".jsonl" else {"text": line, "benchmark": input_path.stem}
                        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                            raise ValueError(f"Invalid benchmark text at {input_path}:{line_number}.")
                        text = record["text"]
                        if not text.strip():
                            continue
                        benchmark = record.get("benchmark")
                        if not isinstance(benchmark, str) or not _benchmark_name(benchmark):
                            raise ValueError("Each exclusion must identify its benchmark family.")
                        benchmark = _benchmark_name(benchmark)
                        self.documents[benchmark] += 1
                        self.upstream_sources[(benchmark, str(record.get("source", "")), str(record.get("revision", "")))] += 1
                        self.metadata_without_revision += not bool(record.get("revision"))
                        words = _words(text)
                        self.exact.add(hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest())
                        if len(words) < 5:
                            self.short_exact_only += 1
                        else:
                            size = min(ngram_size, len(words))
                            self.ngrams[size].update(_shingles(words, size))
        self.expected = sorted({_benchmark_name(name) for name in expected})
        self.missing = sorted(set(self.expected) - self.documents.keys())

    def matches(self, words: Sequence[str]) -> bool:
        if hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest() in self.exact:
            return True
        for size, excluded in self.ngrams.items():
            if len(words) >= size and not _shingles(words, size).isdisjoint(excluded):
                return True
        return False

    def report(self) -> dict:
        return {
            "files": self.files,
            "expected_benchmarks": self.expected,
            "benchmark_document_counts": dict(sorted(self.documents.items())),
            "missing_benchmarks": self.missing,
            "complete_requested_family_coverage": bool(self.documents) and not self.missing,
            "ngram_size": self.ngram_size,
            "shorter_sentences": "5 through n-1 words: whole sentence ngram; fewer than 5 words: exact document only",
            "short_exact_only_texts": self.short_exact_only,
            "records_without_upstream_revision": self.metadata_without_revision,
            "upstream_sources": [{"benchmark": benchmark, "source": source, "revision": revision, "texts": count}
                                 for (benchmark, source, revision), count in sorted(self.upstream_sources.items())],
            "normalization": "NFC and casefold; Unicode letter/digit words; whitespace and punctuation ignored for word ngrams",
            "scope": "Only supplied texts and exact contiguous normalized word ngrams; no paraphrase, translation, or exhaustive benchmark-overlap guarantee.",
        }


class NearDuplicateIndex:
    """64 deterministic MinHash values in 16 bands, then exact shingle Jaccard."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        rng = random.Random(0)
        self.coefficients = [(rng.randrange(1, _PRIME), rng.randrange(_PRIME)) for _ in range(64)]
        self.buckets: dict[tuple, set[str]] = defaultdict(set)
        self.documents: dict[str, set[int]] = {}

    def _signature(self, shingles: set[int]) -> tuple[int, ...]:
        return tuple(min((a * value + b) % _PRIME for value in shingles) for a, b in self.coefficients)

    def duplicate(self, shingles: set[int]) -> tuple[str | None, tuple[int, ...]]:
        if not shingles:
            return None, ()
        signature = self._signature(shingles)
        candidates: set[str] = set()
        for band in range(16):
            candidates.update(self.buckets.get((band, signature[band * 4:band * 4 + 4]), ()))
        for document_id in sorted(candidates):
            other = self.documents[document_id]
            if min(len(shingles), len(other)) / max(len(shingles), len(other)) < self.threshold:
                continue
            intersection = len(shingles & other)
            if intersection / (len(shingles) + len(other) - intersection) >= self.threshold:
                return document_id, signature
        return None, signature

    def add(self, document_id: str, shingles: set[int], signature: tuple[int, ...]) -> None:
        if not signature:
            return
        self.documents[document_id] = shingles
        for band in range(16):
            self.buckets[(band, signature[band * 4:band * 4 + 4])].add(document_id)


def select_corpus(
    input_paths: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 0,
    exclusions_path: str | Path | None = None,
    research_ready: bool = False,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    min_documents_per_language: int = 10,
    near_duplicate_threshold: float = 0.85,
    ngram_size: int = 5,
    benchmark_ngram_size: int = 13,
    expected_benchmarks: Sequence[str] = EXPECTED_BENCHMARKS,
    stable_splits: bool = False,
    split_ledger_path: str | Path | None = None,
) -> dict:
    """Assign splits before filtering, screen documents, and atomically publish.

    Inputs must use globally unique IDs, language codes en/de/ar/fa, and pinned
    40-hex upstream revisions. All four languages need documents in all splits.
    Existing source splits are recorded separately; they do not dictate the new
    pretraining splits. ``research_ready`` checks supplied benchmark-family
    coverage only and never certifies the corpus as contamination free.
    """
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]
    paths = sorted({Path(path).resolve() for path in input_paths})
    output_dir = Path(output_dir).resolve()
    if not paths:
        raise ValueError("Provide at least one local input JSONL file.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}.")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Train/validation fractions must be positive and sum to less than one.")
    if min_documents_per_language < 3 or ngram_size < 1 or benchmark_ngram_size < 5:
        raise ValueError("Need at least 3 documents/language, positive near-duplicate ngram size, and benchmark ngram size >=5.")
    if not math.isfinite(near_duplicate_threshold) or not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1].")
    exclusions = BenchmarkExclusions(exclusions_path, benchmark_ngram_size, expected_benchmarks)
    if research_ready and (not exclusions.documents or not exclusions.expected or exclusions.missing):
        raise ValueError("Research coverage flag requires nonempty exclusions for every expected benchmark family; missing: " + ", ".join(exclusions.missing))
    inputs = [{"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in paths]
    documents, sources, drops = _scan_inputs(paths)
    ledger = None
    previous_ledger_sha = file_sha256(split_ledger_path) if split_ledger_path is not None else None
    if split_ledger_path is not None and not stable_splits:
        raise ValueError("An existing split ledger requires stable_splits")
    if stable_splits:
        from .holdouts import stable_assignments
        identities = [{"id": doc.document_id, "language": doc.language,
                       "content_sha256": doc.text_sha256, "normalized_sha256": doc.normalized_sha256,
                       "canonical_url": canonical_url(doc.url)} for doc in documents]
        assignment, ledger = stable_assignments(identities, seed=seed, train_fraction=train_fraction,
                                                validation_fraction=validation_fraction, previous=split_ledger_path)
    else:
        assignment = _assign_splits(documents, seed, train_fraction, validation_fraction)
    assigned_counts: dict[str, Counter] = defaultdict(Counter)
    for document in documents:
        assigned_counts[document.language][assignment[document.document_id]] += 1
    # Independent order prevents processing all train documents before held-out
    # documents merely because split assignment itself sorts document hashes.
    documents.sort(key=lambda doc: _document_hash(seed + 1, doc.language, doc.document_id))
    exact_seen: dict[str, str] = {}
    urls_seen: dict[str, str] = {}
    near_index = NearDuplicateIndex(near_duplicate_threshold)
    accepted: list[Document] = []
    counts: dict[str, dict] = {language: {"documents": 0, "utf8_bytes": 0, "splits": {split: {"documents": 0, "utf8_bytes": 0} for split in SPLITS}} for language in LANGUAGES}
    drop_examples: list[dict] = []
    with ExitStack() as stack:
        handles = {path: stack.enter_context(path.open("rb")) for path in paths}
        for document in documents:
            text = _read_text(handles, document)
            words = _words(text)
            normalized_sha = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
            page_key = canonical_url(document.url)
            duplicate_of = urls_seen.get(page_key) if page_key is not None else None
            reason = "duplicate_url" if duplicate_of is not None else None
            if reason is None:
                duplicate_of = exact_seen.get(normalized_sha)
                reason = "exact_duplicate" if duplicate_of is not None else None
            if reason is None and exclusions.matches(words):
                reason = "benchmark_overlap"
            shingles = _shingles(words, ngram_size) if reason is None else set()
            signature = ()
            if reason is None:
                duplicate_of, signature = near_index.duplicate(shingles)
                if duplicate_of is not None:
                    reason = "near_duplicate"
            if reason is not None:
                drops[reason] += 1
                if len(drop_examples) < 100:
                    drop_examples.append({"id": document.document_id, "language": document.language,
                                          "assigned_split": assignment[document.document_id], "reason": reason,
                                          "duplicate_of": duplicate_of})
                continue
            accepted.append(document)
            exact_seen[normalized_sha] = document.document_id
            if page_key is not None:
                urls_seen[page_key] = document.document_id
            near_index.add(document.document_id, shingles, signature)
            summary = counts[document.language]
            summary["documents"] += 1
            summary["utf8_bytes"] += document.utf8_bytes
            summary["splits"][assignment[document.document_id]]["documents"] += 1
            summary["splits"][assignment[document.document_id]]["utf8_bytes"] += document.utf8_bytes
            source_summary = sources[f"{document.source}@{document.revision}"]
            source_summary["retained_documents"] = source_summary.get("retained_documents", 0) + 1
            source_summary["retained_utf8_bytes"] = source_summary.get("retained_utf8_bytes", 0) + document.utf8_bytes
        for language, summary in counts.items():
            if summary["documents"] < min_documents_per_language or any(not split["documents"] for split in summary["splits"].values()):
                raise ValueError(f"Insufficient retained documents for {language!r}: {summary}; fetch more raw documents rather than reassigning filtered held-out documents.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
            temporary_path = Path(temporary).resolve()
            if temporary_path.parent != output_dir.parent:
                raise ValueError("Temporary artifact directory escaped the output parent.")
            output_path = temporary_path / "documents.jsonl"
            with output_path.open("w", encoding="utf-8", newline="\n") as output:
                for document in accepted:
                    record = {"id": document.document_id, "text": _read_text(handles, document),
                              "language": document.language, "split": assignment[document.document_id],
                              "source": document.source, "revision": document.revision}
                    if document.url is not None:
                        record["url"] = document.url
                    if document.source_split is not None:
                        record["source_split"] = document.source_split
                    handle = handles[document.path]
                    handle.seek(document.offset)
                    original = json.loads(handle.readline())
                    for key in ("source_shard", "source_row", "source_row_group", "source_language_score", "sampling"):
                        if key in original:
                            record[key] = original[key]
                    if stable_splits:
                        record["raw_provenance"] = {"input_path": str(document.path), "offset": document.offset,
                                                    "content_sha256": document.text_sha256}
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if ledger is not None:
                (temporary_path / "split_ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            for source in [*inputs, *exclusions.files]:
                if file_sha256(source["path"]) != source["sha256"]:
                    raise ValueError("Raw input or benchmark exclusion file changed during corpus selection.")
            if split_ledger_path is not None and file_sha256(split_ledger_path) != previous_ledger_sha:
                raise ValueError("Frozen split ledger changed during corpus selection")
            manifest = {
                "schema_version": 1,
                "artifact_kind": "screened_bounded_corpus_sample",
                "input_files": inputs,
                "sources": list(sorted(sources.values(), key=lambda entry: (entry["source"], entry["revision"]))),
                "seed": seed,
                "split_assignment": {"method": ("stable identity-connected hash thresholds with persistent ledger; assigned before filtering" if stable_splits else "per-language sort of SHA256([seed, language, document_id]); allocated before filtering; filtered documents are never reassigned"),
                                     "train_fraction": train_fraction, "validation_fraction": validation_fraction,
                                     "assigned_documents": {language: dict(assigned_counts[language]) for language in LANGUAGES},
                                     "stability": ("persistent ledger; conflicting frozen identity connections fail" if stable_splits else "reproducible for the same immutable input snapshot; adding documents can change percentile boundaries"),
                                     "previous_ledger_sha256": previous_ledger_sha},
                "retained_by_language": counts,
                "retained_documents": len(accepted),
                "retained_utf8_bytes": sum(document.utf8_bytes for document in accepted),
                "dropped_documents": dict(sorted(drops.items())),
                "drop_examples_first_100": drop_examples,
                "deduplication": {"exact": "NFC, casefold, whitespace-folded document SHA256; compared across every split and language",
                                  "url": "Across all sources/languages/splits: HTTP/HTTPS equivalent; lowercase hostname and optional www removed; NFC Unicode path spelling; fragments and utm_*/fbclid/gclid removed; remaining query pairs sorted and retained; Wikipedia /wiki/ underscores equal spaces; original output URL unchanged",
                                  "near": "64-value deterministic MinHash LSH, 16 bands of 4; candidates confirmed with Jaccard over hashed word shingles",
                                  "near_word_ngram_size": ngram_size, "near_jaccard_threshold": near_duplicate_threshold,
                                  "minhash_seed": 0, "shingle_hash": "BLAKE2b-64 modulo (2**61 - 1)",
                                  "limitations": "Approximate LSH can miss near duplicates; URL matching cannot identify redirects, renamed pages, mirrors or absent/unparseable URLs, and conservatively removes same-URL revisions even when text differs. Query sorting assumes pair order is not content-defining. No semantic/paraphrase deduplication or exhaustive cross-source audit."},
                "benchmark_exclusions": exclusions.report(),
                "requested_benchmark_coverage_gate": research_ready,
                "text_normalization": "none in emitted text; comparison normalization only",
                "quality_scope": "Pinned local sample, document-level split isolation and limited overlap screening; no comprehensive language-ID, quality, PII, licensing or production-readiness certification.",
                "files": {"documents.jsonl": {"sha256": file_sha256(output_path), "bytes": output_path.stat().st_size}},
            }
            if ledger is not None:
                ledger_file = temporary_path / "split_ledger.json"
                manifest["files"]["split_ledger.json"] = {"sha256": file_sha256(ledger_file), "bytes": ledger_file.stat().st_size}
            (temporary_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            temporary_path.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--research-ready", action="store_true", help="Require requested supplied benchmark-family coverage; does not certify contamination freedom.")
    parser.add_argument("--expected-benchmarks", nargs="+", default=list(EXPECTED_BENCHMARKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-documents-per-language", type=int, default=10)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--stable-splits", action="store_true")
    parser.add_argument("--split-ledger", type=Path)
    args = parser.parse_args()
    try:
        manifest = select_corpus(args.input, args.output, seed=args.seed, exclusions_path=args.exclusions,
                                 research_ready=args.research_ready, train_fraction=args.train_fraction,
                                 validation_fraction=args.validation_fraction,
                                 min_documents_per_language=args.min_documents_per_language,
                                 near_duplicate_threshold=args.near_duplicate_threshold,
                                 expected_benchmarks=args.expected_benchmarks,
                                 stable_splits=args.stable_splits, split_ledger_path=args.split_ledger)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.resolve()), "retained_by_language": manifest["retained_by_language"],
                      "dropped_documents": manifest["dropped_documents"], "benchmark_exclusions": manifest["benchmark_exclusions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
