"""Create an immutable, exactly token-balanced derivative of a prepared bundle.

Source documents retain their assigned splits. Each language contributes the
same number of tokens within a split, without repetition. Only the final
selected document in a language can be cropped; no synthetic EOS is inserted.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from .data import load_tokenizer
from .prepare import LANGUAGES, SPLITS
from .provenance import file_sha256, prepared_identity

_COPY_TOKENS = 1_048_576


def _read_documents(path: Path, streams: dict, languages: tuple[str, ...], eos_id: int) -> dict:
    grouped = {split: {language: [] for language in languages} for split in SPLITS}
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid document JSON at line {line_number}.") from exc
            if not isinstance(document, dict):
                raise ValueError("Document metadata must contain JSON objects.")
            document_id = document.get("id")
            if not isinstance(document_id, str) or not document_id or document_id in seen_ids:
                raise ValueError("Document IDs must be nonempty and unique across all splits.")
            seen_ids.add(document_id)
            content_hash = document.get("content_sha256")
            if content_hash is not None:
                if not isinstance(content_hash, str) or content_hash in seen_hashes:
                    raise ValueError("Duplicate or invalid document content hashes; possible split leakage.")
                seen_hashes.add(content_hash)
            split, language = document.get("split"), document.get("language")
            if split not in SPLITS or language not in languages:
                raise ValueError("Document metadata has an unexpected split or language.")
            start, count = document.get("token_start"), document.get("tokens")
            if type(start) is not int or start < 0 or type(count) is not int or count < 1:
                raise ValueError("Document token offsets/counts must be nonnegative/positive integers.")
            if start + count > streams[split].size:
                raise ValueError("Document token span exceeds its source split.")
            if document.get("complete_document", True) and int(streams[split][start + count - 1]) != eos_id:
                raise ValueError("Complete source documents must end in the tokenizer's EOS token.")
            grouped[split][language].append(document)
    for split in SPLITS:
        documents = [document for entries in grouped[split].values() for document in entries]
        position = 0
        for document in sorted(documents, key=lambda entry: entry["token_start"]):
            if document["token_start"] != position:
                raise ValueError("Document spans must cover each source split exactly, without gaps or overlap.")
            position += document["tokens"]
        if position != streams[split].size:
            raise ValueError("Document spans do not cover their source split.")
        for entries in grouped[split].values():
            entries.sort(key=lambda entry: entry["token_start"])
    return grouped


def _summary() -> dict:
    return {"documents": 0, "complete_documents": 0, "truncated_documents": 0,
            "tokens": 0, "utf8_bytes": 0, "known_complete_document_utf8_bytes": 0}


def _count_document(summary: dict, document: dict) -> None:
    summary["documents"] += 1
    summary["tokens"] += document["tokens"]
    if document["complete_document"]:
        summary["complete_documents"] += 1
        known_bytes = document.get("utf8_bytes")
        if type(known_bytes) is int:
            summary["known_complete_document_utf8_bytes"] += known_bytes
            if summary["utf8_bytes"] is not None:
                summary["utf8_bytes"] += known_bytes
        else:
            summary["utf8_bytes"] = None
    else:
        summary["truncated_documents"] += 1
        # Token prefixes need not coincide with UTF-8 boundaries. Do not invent
        # text byte counts or retain the original full-document content hash.
        summary["utf8_bytes"] = None


def balance_prepared(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
    languages: tuple[str, ...] = LANGUAGES,
) -> dict:
    """Copy deterministic language prefixes into a new token-balanced bundle.

    A split's per-language quota is the minimum available token count, with
    training and validation requiring every selected language. An entirely
    empty test split is allowed. Language blocks are shuffled using ``seed``;
    document order within a block follows source token order. Exact source
    spans, cropped-document status and per-language output offsets are recorded.
    Only bounded token chunks are copied; source and output streams are memmaps.
    """
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; use a new directory.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    languages = tuple(languages)
    if not languages or len(languages) != len(set(languages)) or any(language not in LANGUAGES for language in languages):
        raise ValueError(f"languages must be distinct supported language codes from {LANGUAGES}.")
    identity = prepared_identity(input_dir)
    source_manifest_path = input_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    required = {"documents.jsonl", "test.npy"}
    if not required.issubset(source_manifest["files"]):
        raise ValueError("Balancing requires source documents.jsonl and test.npy artifacts.")
    tokenizer = load_tokenizer(input_dir / "tokenizer.json")
    declared_tokenizer = source_manifest.get("tokenizer", {})
    if declared_tokenizer.get("vocab_size") != tokenizer.vocab_size or declared_tokenizer.get("eos_id") != tokenizer.eos_id:
        raise ValueError("Source manifest vocabulary/EOS does not match its tokenizer.")
    with ExitStack() as source_mappings:
        streams = {}
        for split in SPLITS:
            stream = np.load(input_dir / f"{split}.npy", mmap_mode="r", allow_pickle=False)
            source_mappings.callback(stream._mmap.close)
            if stream.ndim != 1 or stream.dtype != np.dtype("<u4"):
                raise ValueError("Source token streams must be one-dimensional little-endian uint32 arrays.")
            for start in range(0, stream.size, _COPY_TOKENS):
                if np.any(stream[start:start + _COPY_TOKENS] >= tokenizer.vocab_size):
                    raise ValueError("Source token ID exceeds the tokenizer's actual vocabulary.")
            streams[split] = stream
        grouped = _read_documents(input_dir / "documents.jsonl", streams, languages, tokenizer.eos_id)
        rng = random.Random(seed)
        split_info = {}
        for split in SPLITS:
            available = {language: sum(document["tokens"] for document in grouped[split][language]) for language in languages}
            if not all(available.values()) and (split != "test" or any(available.values())):
                raise ValueError(f"Split {split!r} must contain every selected language; only a wholly empty test split is allowed.")
            order = list(languages)
            rng.shuffle(order)
            quota = min(available.values())
            split_info[split] = {"available_tokens": available, "tokens_per_language": quota, "language_order": order}
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary, ExitStack() as output_mappings:
            temporary_path = Path(temporary).resolve()
            if temporary_path.parent != output_dir.parent:
                raise ValueError("Temporary artifact directory escaped the output parent.")
            # Copies, rather than links, keep source artifacts immutable if the
            # derivative is moved, replaced or accidentally edited later.
            shutil.copyfile(input_dir / "tokenizer.json", temporary_path / "tokenizer.json")
            shutil.copyfile(source_manifest_path, temporary_path / "source_manifest.json")
            fitting_files = [name for name in ("tokenizer_fit.jsonl",) if name in source_manifest["files"]]
            for name in fitting_files:
                shutil.copyfile(input_dir / name, temporary_path / name)
            output_streams = {}
            for split in SPLITS:
                count = split_info[split]["tokens_per_language"] * len(languages)
                stream = np.lib.format.open_memmap(temporary_path / f"{split}.npy", mode="w+", dtype="<u4", shape=(count,))
                output_mappings.callback(stream._mmap.close)
                output_streams[split] = stream
            statistics = {}
            id_digest = hashlib.sha256()
            with (temporary_path / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as metadata_file:
                for split in SPLITS:
                    summary = _summary()
                    summary["languages"] = {}
                    quota = split_info[split]["tokens_per_language"]
                    position = 0
                    for language in split_info[split]["language_order"]:
                        language_start = position
                        language_summary = _summary()
                        remaining = quota
                        for source_document in grouped[split][language]:
                            if remaining == 0:
                                break
                            count = min(remaining, source_document["tokens"])
                            source_start = source_document["token_start"]
                            for chunk in range(0, count, _COPY_TOKENS):
                                size = min(_COPY_TOKENS, count - chunk)
                                output_streams[split][position + chunk:position + chunk + size] = streams[split][source_start + chunk:source_start + chunk + size]
                            complete = count == source_document["tokens"] and source_document.get("complete_document", True)
                            metadata = {
                                "id": source_document["id"], "language": language, "split": split,
                                "token_start": position, "tokens": count,
                                "complete_document": complete,
                                "eos_preserved": int(streams[split][source_start + count - 1]) == tokenizer.eos_id,
                                "content_sha256": source_document.get("content_sha256") if complete else None,
                                "utf8_bytes": source_document.get("utf8_bytes") if complete else None,
                                "source": source_document.get("source"), "revision": source_document.get("revision"),
                                "source_token_span": {"split": split, "token_start": source_start, "token_end": source_start + count},
                                "source_document_metadata": copy.deepcopy(source_document),
                            }
                            metadata_file.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                            id_digest.update(json.dumps(metadata["id"], ensure_ascii=False).encode("utf-8") + b"\n")
                            _count_document(summary, metadata)
                            _count_document(language_summary, metadata)
                            position += count
                            remaining -= count
                        if remaining:
                            raise ValueError("Source metadata did not supply the selected token quota.")
                        language_summary["token_start"] = language_start
                        language_summary["token_end"] = position
                        summary["languages"][language] = language_summary
                    statistics[split] = summary
            for stream in output_streams.values():
                stream.flush()
            output_mappings.close()
            if prepared_identity(input_dir) != identity:
                raise ValueError("Source prepared bundle changed during balancing.")
            files = {}
            for name in ("tokenizer.json", "documents.jsonl", "source_manifest.json", *fitting_files, *(f"{split}.npy" for split in SPLITS)):
                artifact = temporary_path / name
                files[name] = {"sha256": file_sha256(artifact), "bytes": artifact.stat().st_size}
            if files["source_manifest.json"]["sha256"] != identity["manifest_sha256"]:
                raise ValueError("Copied source manifest does not match its immutable identity.")
            manifest = copy.deepcopy(source_manifest)
            manifest.update({
                "document_ids_sha256": id_digest.hexdigest(), "seed": seed,
                "packing": "seeded whole-language blocks per split; source document order within each block; attention may cross documents and language boundaries",
                "language_sampling": "exactly equal token counts per selected language in every split; no repetition",
                "splits": statistics, "files": files,
                "source_preparation": {"directory": str(input_dir), **identity, "manifest_artifact": "source_manifest.json"},
                "balance": {
                    "algorithm": "equal-token-prefix-per-language-v1", "seed": seed,
                    "languages": list(languages), "splits": split_info,
                    "selection": "minimum available per-language token count within each split; source document order; no repetition",
                    "cropped_document_policy": "crop only the final selected document prefix at the exact quota; preserve source tokens; never insert or replace EOS; null cropped text hashes and byte counts",
                    "source_spans": "documents.jsonl source_token_span indexes the corresponding source prepared split, with exclusive token_end",
                    "evaluation_slices": "splits.<split>.languages.<language>.token_start/token_end are contiguous per-language slices, with exclusive token_end",
                },
            })
            (temporary_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            temporary_path.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Verified source prepared bundle")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = balance_prepared(args.input, args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "splits": manifest["splits"],
                      "balance": manifest["balance"], "vocab_size": manifest["tokenizer"]["vocab_size"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
