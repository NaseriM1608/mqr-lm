"""Prepare explicitly split multilingual JSONL without network access.

Each line has id, text, language (en/de/ar/fa), and split
(train/validation/test). Optional source and revision strings are recorded.
The caller is responsible for source licensing and benchmark contamination
screening; this module rejects repeated IDs and duplicate documents.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import tempfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .data import BPETokenizer, ByteTokenizer, load_tokenizer

SPLITS = ("train", "validation", "test")
LANGUAGES = ("en", "de", "ar", "fa")
RESEARCH_SPECIAL_TOKENS = (
    "<|pad|>", "<|eos|>", "<|bos|>",
    "<|lang_en|>", "<|lang_de|>", "<|lang_ar|>", "<|lang_fa|>",
    "<|translate|>", "<|nli|>",
    "<|entailment|>", "<|neutral|>", "<|contradiction|>",
)


@dataclass(frozen=True)
class Document:
    document_id: str
    language: str
    split: str
    offset: int
    byte_count: int
    content_sha256: str
    source: str | None
    revision: str | None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    seen_content: dict[str, str] = {}
    counts: Counter[str] = Counter()
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
                raise ValueError(f"Invalid UTF-8 JSON at line {line_number}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            for field in ("id", "text", "language", "split"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"Line {line_number} requires a nonempty string '{field}'.")
            document_id, text = record["id"], record["text"]
            if document_id in seen_ids:
                raise ValueError(f"Duplicate document ID: {document_id!r}.")
            if record["split"] not in SPLITS:
                raise ValueError(f"Invalid split at line {line_number}; choose {SPLITS}.")
            if record["language"] not in LANGUAGES:
                raise ValueError(f"Invalid language at line {line_number}; choose {LANGUAGES}.")
            for field in ("source", "revision"):
                if record.get(field) is not None and not isinstance(record[field], str):
                    raise ValueError(f"Optional '{field}' must be a string at line {line_number}.")
            encoded = text.encode("utf-8")
            content_sha = hashlib.sha256(encoded).hexdigest()
            # This key detects whitespace/NFC variants without altering text fed
            # to the tokenizer or conflating Persian/Arabic letter distinctions.
            duplicate_key = hashlib.sha256(
                unicodedata.normalize("NFC", " ".join(text.split())).encode("utf-8")
            ).hexdigest()
            if duplicate_key in seen_content:
                raise ValueError(
                    f"Duplicate document content: {document_id!r} and "
                    f"{seen_content[duplicate_key]!r}; possible split leakage."
                )
            seen_ids.add(document_id)
            seen_content[duplicate_key] = document_id
            counts[record["split"]] += 1
            documents.append(
                Document(
                    document_id, record["language"], record["split"], offset,
                    len(encoded), content_sha, record.get("source"), record.get("revision"),
                )
            )
    if not counts["train"] or not counts["validation"]:
        raise ValueError("Input must contain separate, explicitly assigned train and validation documents.")
    return documents


def _read_text(handle: BinaryIO, document: Document) -> str:
    handle.seek(document.offset)
    text = json.loads(handle.readline())["text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document.content_sha256:
        raise ValueError("Input changed while preparation was running.")
    return text


def _balanced_texts(
    path: Path,
    documents: list[Document],
    byte_limit: int,
    seed: int,
    selected_bytes: dict[str, int],
    fitting_records: list[dict] | None = None,
) -> Iterator[str]:
    by_language: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        if document.split == "train":
            by_language[document.language].append(document)
    quota = min(byte_limit, *(sum(doc.byte_count for doc in docs) for docs in by_language.values()))
    rng = random.Random(seed)
    for language in sorted(by_language):
        rng.shuffle(by_language[language])
        selected_bytes[language] = 0
    iterators = {language: iter(docs) for language, docs in by_language.items()}
    with path.open("rb") as handle:
        while iterators:
            for language in sorted(list(iterators)):
                remaining = quota - selected_bytes[language]
                document = next(iterators[language], None)
                if remaining < 1 or document is None:
                    del iterators[language]
                    continue
                encoded = _read_text(handle, document).encode("utf-8")
                # Truncate only the tokenizer fitting sample, at a UTF-8 boundary.
                text = encoded[:remaining].decode("utf-8", errors="ignore")
                if not text:
                    del iterators[language]
                    continue
                selected_bytes[language] += len(text.encode("utf-8"))
                if fitting_records is not None:
                    fitting_records.append({"id": document.document_id, "language": language,
                                            "content_sha256": document.content_sha256,
                                            "sample_utf8_bytes": len(text.encode("utf-8")),
                                            "sample_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                            "full_document_utf8_bytes": document.byte_count})
                yield text


def _train_bpe(
    path: Path,
    documents: list[Document],
    vocab_size: int,
    seed: int,
    byte_limit: int,
    fitting_records: list[dict] | None = None,
) -> tuple[BPETokenizer, dict[str, int]]:
    if vocab_size < 256 + len(RESEARCH_SPECIAL_TOKENS):
        raise ValueError(f"Research BPE requires at least {256 + len(RESEARCH_SPECIAL_TOKENS)} entries for bytes and reserved markers.")
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise ImportError("Install the 'tokenizers' package to prepare BPE data.") from exc
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(RESEARCH_SPECIAL_TOKENS),
        show_progress=False,
    )
    selected_bytes: dict[str, int] = {}
    tokenizer.train_from_iterator(
        _balanced_texts(path, documents, byte_limit, seed, selected_bytes, fitting_records), trainer=trainer
    )
    return BPETokenizer(tokenizer), selected_bytes


def prepare_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 65_536,
    tokenizer_kind: str = "byte",
    seed: int = 0,
    tokenizer_max_bytes_per_language: int = 100_000_000,
    *, tokenizer_path: str | Path | None = None,
) -> dict:
    """Build immutable .npy streams and a content-addressed preparation manifest.

    Document order and language proportions follow the input; they are not
    automatically balanced. Only BPE fitting uses balanced training-language
    byte samples. Both architectures must consume these exact same streams.
    """
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; use a new directory.")
    if tokenizer_kind not in ("byte", "bpe"):
        raise ValueError("tokenizer_kind must be 'byte' or 'bpe'.")
    if tokenizer_max_bytes_per_language < 1:
        raise ValueError("tokenizer_max_bytes_per_language must be positive.")
    source_sha = file_sha256(input_path)
    documents = _scan_documents(input_path)
    selected_bytes: dict[str, int] = {}
    fitting_records: list[dict] = []
    frozen_path = Path(tokenizer_path).resolve() if tokenizer_path is not None else None
    frozen_sha = file_sha256(frozen_path) if frozen_path is not None else None
    if frozen_path is not None:
        tokenizer = load_tokenizer(frozen_path)
        if (tokenizer_kind == "bpe") != isinstance(tokenizer, BPETokenizer):
            raise ValueError("Frozen tokenizer kind differs from tokenizer_kind")
        if tokenizer_kind == "bpe" and tokenizer.vocab_size != vocab_size:
            raise ValueError("Frozen tokenizer vocabulary size differs from requested vocab_size")
        if tokenizer_kind == "bpe":
            markers = [tokenizer.tokenizer.token_to_id(token) for token in RESEARCH_SPECIAL_TOKENS]
            if None in markers or len(set(markers)) != len(markers):
                raise ValueError("Frozen research BPE lacks distinct reserved markers")
    elif tokenizer_kind == "bpe":
        tokenizer, selected_bytes = _train_bpe(
            input_path, documents, vocab_size, seed, tokenizer_max_bytes_per_language, fitting_records
        )
    else:
        tokenizer = ByteTokenizer()
    token_counts: list[int] = []
    with input_path.open("rb") as handle:
        for document in documents:
            token_counts.append(len(tokenizer.encode(_read_text(handle, document))))
    split_counts = {
        split: sum(count for doc, count in zip(documents, token_counts) if doc.split == split)
        for split in SPLITS
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only when all files and their manifest have been written successfully.
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary, ExitStack() as mappings:
        temporary_path = Path(temporary).resolve()
        if temporary_path.parent != output_dir.parent:
            raise ValueError("Temporary artifact directory escaped the output parent.")
        if frozen_path is not None:
            (temporary_path / "tokenizer.json").write_bytes(frozen_path.read_bytes())
        else:
            tokenizer.save(temporary_path / "tokenizer.json")
        streams = {}
        for split, count in split_counts.items():
            stream = np.lib.format.open_memmap(
                temporary_path / f"{split}.npy", mode="w+", dtype="<u4", shape=(count,)
            )
            mappings.callback(stream._mmap.close)
            streams[split] = stream
        offsets = {split: 0 for split in SPLITS}
        stats: dict[str, dict] = {split: {"documents": 0, "utf8_bytes": 0, "tokens": 0, "languages": {}} for split in SPLITS}
        id_digest = hashlib.sha256()
        with input_path.open("rb") as handle, (temporary_path / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as metadata_file:
            for document, expected_count in zip(documents, token_counts):
                ids = tokenizer.encode(_read_text(handle, document))
                if len(ids) != expected_count:
                    raise ValueError("Tokenization changed between preparation passes.")
                start = offsets[document.split]
                streams[document.split][start : start + expected_count] = ids
                offsets[document.split] += expected_count
                metadata = {
                    "id": document.document_id,
                    "language": document.language,
                    "split": document.split,
                    "content_sha256": document.content_sha256,
                    "utf8_bytes": document.byte_count,
                    "tokens": expected_count,
                    "token_start": start,
                    "source": document.source,
                    "revision": document.revision,
                }
                metadata_file.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                id_digest.update(json.dumps(document.document_id, ensure_ascii=False).encode("utf-8") + b"\n")
                summary = stats[document.split]
                for counter, value in (("documents", 1), ("utf8_bytes", document.byte_count), ("tokens", expected_count)):
                    summary[counter] += value
                language_summary = summary["languages"].setdefault(document.language, {"documents": 0, "utf8_bytes": 0, "tokens": 0})
                for counter, value in (("documents", 1), ("utf8_bytes", document.byte_count), ("tokens", expected_count)):
                    language_summary[counter] += value
        for stream in streams.values():
            stream.flush()
        # Windows cannot rename a directory containing an open memory map.
        mappings.close()
        if file_sha256(input_path) != source_sha:
            raise ValueError("Input changed while preparation was running.")
        if frozen_path is not None and (file_sha256(frozen_path) != frozen_sha or file_sha256(temporary_path / "tokenizer.json") != frozen_sha):
            raise ValueError("Frozen tokenizer changed while preparation was running")
        files = {}
        for artifact in ("tokenizer.json", "documents.jsonl", *(f"{split}.npy" for split in SPLITS)):
            artifact_path = temporary_path / artifact
            files[artifact] = {"sha256": file_sha256(artifact_path), "bytes": artifact_path.stat().st_size}
        if fitting_records:
            fitting_path = temporary_path / "tokenizer_fit.jsonl"
            fitting_path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in fitting_records), encoding="utf-8")
            files[fitting_path.name] = {"sha256": file_sha256(fitting_path), "bytes": fitting_path.stat().st_size}
        manifest = {
            "schema_version": 1,
            "input": {"path": str(input_path), "sha256": source_sha, "bytes": input_path.stat().st_size},
            "document_ids_sha256": id_digest.hexdigest(),
            "tokenizer": {
                "kind": tokenizer_kind,
                "requested_vocab_size": vocab_size if tokenizer_kind == "bpe" else 259,
                "vocab_size": tokenizer.vocab_size,
                "pad_id": tokenizer.pad_id,
                "eos_id": tokenizer.eos_id,
                "bos_id": tokenizer.bos_id,
                "reserved_markers": ({token: tokenizer.tokenizer.token_to_id(token) for token in RESEARCH_SPECIAL_TOKENS}
                                     if tokenizer_kind == "bpe" else {}),
                "marker_encoding_policy": "literal corpus spellings encode as ordinary text; task adapters must insert reserved IDs explicitly",
                "fitting_split": ("external_frozen; see source provenance" if frozen_path is not None else "train") if tokenizer_kind == "bpe" else None,
                "frozen_source": {"path": str(frozen_path), "sha256": frozen_sha} if frozen_path is not None else None,
                "balanced_fitting_utf8_bytes": selected_bytes,
                "fitting_byte_limit_per_language": tokenizer_max_bytes_per_language,
                "tokenizers_version": importlib.metadata.version("tokenizers") if tokenizer_kind == "bpe" else None,
            },
            "seed": seed,
            "normalization": "none; original text preserved",
            "deduplication": "reject repeated IDs or NFC/whitespace-normalized duplicate document text across all splits",
            "contamination_audit": "document duplicates only; external benchmark overlap and near-duplicate audit still required",
            "packing": "input document order; EOS appended to each document; attention may cross EOS",
            "language_sampling": "input proportions; no automatic token balancing",
            "dtype": "uint32_little_endian",
            "splits": stats,
            "files": files,
        }
        (temporary_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_path.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Explicitly split document JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing directories are rejected.")
    parser.add_argument("--tokenizer-kind", choices=("byte", "bpe"), default="byte")
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer-max-bytes-per-language", type=int, default=100_000_000)
    parser.add_argument("--tokenizer", type=Path, help="Reuse an already frozen tokenizer without refitting; preserve its original fitting provenance.")
    args = parser.parse_args()
    manifest = prepare_jsonl(
        args.input, args.output, args.vocab_size, args.tokenizer_kind,
        args.seed, args.tokenizer_max_bytes_per_language,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps({"output": str(args.output.resolve()), "splits": manifest["splits"], "tokenizer": manifest["tokenizer"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
