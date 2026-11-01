"""Check prepared multilingual data integrity and tokenizer readiness on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import ByteTokenizer, load_tokenizer
from .prepare import LANGUAGES, SPLITS
from .provenance import file_sha256, prepared_identity

_SAMPLE_LIMIT = 20
_CHUNK_SIZE = 1_048_576
_PROBE = "A\tB\nC  Grüße e\u0301 العربية فارسی می\u200cروم ۱۲۳۴۵۶۷۸۹۰ 😀 漢字 <|eos|>"


@dataclass(frozen=True)
class SelectedDocument:
    document_id: str
    language: str
    split: str
    offset: int
    content_sha256: str
    utf8_bytes: int
    words: int


def _summary() -> dict:
    return {"documents": 0, "utf8_bytes": 0, "tokens": 0,
            "tokens_excluding_eos": 0, "whitespace_words": 0}


def _finish_summary(summary: dict) -> None:
    words = summary["whitespace_words"]
    summary["tokens_per_whitespace_word"] = summary["tokens_excluding_eos"] / words if words else None


def _scan_selected(path: Path) -> dict[str, SelectedDocument]:
    documents = {}
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            document_id = record.get("id")
            text = record.get("text")
            if not isinstance(document_id, str) or document_id in documents or not isinstance(text, str):
                raise ValueError("Selected document IDs must be unique strings and text must be a string.")
            language, split = record.get("language"), record.get("split")
            if language not in LANGUAGES or split not in SPLITS:
                raise ValueError("Selected document language or split is unsupported.")
            encoded = text.encode("utf-8")
            documents[document_id] = SelectedDocument(document_id, language, split, offset,
                                                       hashlib.sha256(encoded).hexdigest(), len(encoded), len(text.split()))
    return documents


def _open_streams(stack: ExitStack, directory: Path, vocab_size: int) -> dict:
    streams = {}
    for split in SPLITS:
        stream = np.load(directory / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        stack.callback(stream._mmap.close)
        if stream.ndim != 1 or stream.dtype != np.dtype("<u4"):
            raise ValueError("Token streams must be one-dimensional little-endian uint32 arrays.")
        for start in range(0, len(stream), _CHUNK_SIZE):
            if np.any(stream[start:start + _CHUNK_SIZE] >= vocab_size):
                raise ValueError("A prepared token ID exceeds the tokenizer vocabulary.")
        streams[split] = stream
    return streams


def _read_metadata(path: Path, streams: dict) -> dict[str, dict]:
    documents = {}
    by_split = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            document = json.loads(line)
            document_id = document.get("id")
            if not isinstance(document_id, str) or document_id in documents:
                raise ValueError("Prepared document IDs are missing or duplicated.")
            split, language = document.get("split"), document.get("language")
            start, count = document.get("token_start"), document.get("tokens")
            if split not in SPLITS or language not in LANGUAGES:
                raise ValueError("Prepared metadata has an unsupported split or language.")
            if type(start) is not int or start < 0 or type(count) is not int or count < 1:
                raise ValueError("Prepared token spans must have nonnegative starts and positive lengths.")
            documents[document_id] = document
            by_split[split].append(document)
    for split in SPLITS:
        position = 0
        for document in sorted(by_split[split], key=lambda item: item["token_start"]):
            if document["token_start"] != position:
                raise ValueError("Document token spans must cover the entire split without gaps or overlaps.")
            position += document["tokens"]
        if position != len(streams[split]):
            raise ValueError("Prepared metadata does not cover its token stream exactly.")
    return documents


def _tokenizer_policy(path: Path, tokenizer) -> dict:
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(tokenizer, ByteTokenizer):
        return {"kind": "utf8_bytes", "vocab_size": 259, "all_256_byte_symbols_present": True,
                "missing_byte_symbols": 0, "unk_token": None, "unk_id": None,
                "added_vocab_ids": {"<|pad|>": tokenizer.pad_id, "<|eos|>": tokenizer.eos_id, "<|bos|>": tokenizer.bos_id},
                "byte_representation": "IDs 0 through 255 are UTF-8 bytes", "normalizer": None}
    from tokenizers.pre_tokenizers import ByteLevel
    alphabet = set(ByteLevel.alphabet())
    vocab = tokenizer.tokenizer.get_vocab()
    model = descriptor.get("model", {})
    unk = model.get("unk_token")
    return {"kind": "bpe", "vocab_size": tokenizer.vocab_size,
            "all_256_byte_symbols_present": alphabet.issubset(vocab),
            "missing_byte_symbols": len(alphabet - vocab.keys()),
            "unk_token": unk, "unk_id": vocab.get(unk) if unk is not None else None,
            "added_vocab_ids": {entry["content"]: entry["id"] for entry in descriptor.get("added_tokens", [])},
            "model_byte_fallback_flag": model.get("byte_fallback", False),
            "byte_representation": "ByteLevel alphabet; full alphabet provides byte coverage independently of BPE model byte_fallback flag",
            "pre_tokenizer": descriptor.get("pre_tokenizer"), "decoder": descriptor.get("decoder"),
            "normalizer": descriptor.get("normalizer")}


def _sample_checks(selected_path: Path, selected: dict, metadata: dict, streams: dict, tokenizer, policy: dict) -> dict:
    checks = {}
    unk_id = policy["unk_id"]
    with selected_path.open("rb") as handle:
        for language in LANGUAGES:
            checks[language] = {}
            for split in ("train", "validation"):
                candidates = [document for document in selected.values() if document.language == language and document.split == split]
                candidates.sort(key=lambda document: hashlib.sha256(document.document_id.encode("utf-8")).hexdigest())
                count, zwnj, observed_bytes, unk_count = 0, 0, set(), 0
                for document in candidates[:_SAMPLE_LIMIT]:
                    handle.seek(document.offset)
                    text = json.loads(handle.readline())["text"]
                    ids = tokenizer.encode(text)
                    entry = metadata[document.document_id]
                    start, length = entry["token_start"], entry["tokens"]
                    if ids != streams[split][start:start + length].tolist():
                        raise ValueError(f"Sample encoding disagrees with stored token span: {document.document_id!r}.")
                    if not ids or ids[-1] != tokenizer.eos_id:
                        raise ValueError("Sample encoding did not append EOS.")
                    decoded = tokenizer.decode(ids)
                    if decoded != text:
                        raise ValueError(f"Tokenizer round-trip changed sampled document {document.document_id!r}.")
                    if text.count("\u200c") != decoded.count("\u200c"):
                        raise ValueError("Tokenizer changed Persian ZWNJ in a sampled document.")
                    count += 1
                    zwnj += text.count("\u200c")
                    observed_bytes.update(text.encode("utf-8"))
                    if unk_id is not None:
                        unk_count += ids.count(unk_id)
                checks[language][split] = {"documents": count, "roundtrip_passed": count,
                                          "stored_token_span_agreement_passed": count,
                                          "zwnj_characters_preserved": zwnj,
                                          "observed_utf8_byte_values": len(observed_bytes),
                                          "unk_tokens": unk_count}
    probe = _PROBE + " " + " ".join(policy["added_vocab_ids"])
    probe_ids = tokenizer.encode(probe, add_eos=False)
    if tokenizer.decode(probe_ids) != probe:
        raise ValueError("Tokenizer Unicode/whitespace/ZWNJ probe did not round-trip exactly.")
    if set(policy["added_vocab_ids"].values()).intersection(probe_ids):
        raise ValueError("Literal special-token text unexpectedly encoded as a control token.")
    return {"selection": "up to 20 lowest SHA256(document ID) documents per language and train/validation split",
            "by_language": checks, "unicode_probe_roundtrip": True, "persian_zwnj_probe_preserved": True,
            "probe_scope": "tabs/newlines/repeated spaces, German accents, decomposed accent, Arabic, Persian ZWNJ/digits, CJK, emoji, literal EOS spelling",
            "test_policy": "test text receives mechanical byte/word metadata counts only; no test encoding, decoding, content sampling or performance evaluation"}


def _audit_balanced(directory: Path, prepared_identity_value: dict, base_metadata: dict,
                    base_streams: dict, selected: dict, tokenizer, warnings: list[str], stack: ExitStack) -> dict:
    identity = prepared_identity(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if identity["tokenizer_sha256"] != prepared_identity_value["tokenizer_sha256"]:
        raise ValueError("Balanced tokenizer does not match source prepared tokenizer.")
    if manifest.get("source_preparation", {}).get("manifest_sha256") != prepared_identity_value["manifest_sha256"]:
        raise ValueError("Balanced bundle is derived from a different preparation manifest.")
    if file_sha256(directory / "source_manifest.json") != prepared_identity_value["manifest_sha256"]:
        raise ValueError("Balanced source-manifest artifact does not match the source preparation.")
    streams = _open_streams(stack, directory, tokenizer.vocab_size)
    metadata = _read_metadata(directory / "documents.jsonl", streams)
    ranges = defaultdict(list)
    stats = {language: {split: {"documents": 0, "complete_documents": 0, "cropped_documents": 0,
                               "tokens": 0, "utf8_bytes": 0, "whitespace_words": 0,
                               "complete_document_tokens_excluding_eos": 0,
                               "complete_document_whitespace_words": 0} for split in SPLITS} for language in LANGUAGES}
    for document_id, entry in metadata.items():
        if document_id not in base_metadata:
            raise ValueError("Balanced metadata references an unknown source document.")
        base = base_metadata[document_id]
        split, language = entry["split"], entry["language"]
        count, target_start = entry["tokens"], entry["token_start"]
        span = entry.get("source_token_span", {})
        source_start = base["token_start"]
        if (split, language) != (base["split"], base["language"]) or span != {
            "split": split, "token_start": source_start, "token_end": source_start + count
        } or count > base["tokens"]:
            raise ValueError("Balanced source span is not a prefix of its source document in the same split.")
        for offset in range(0, count, _CHUNK_SIZE):
            size = min(_CHUNK_SIZE, count - offset)
            if not np.array_equal(streams[split][target_start + offset:target_start + offset + size],
                                  base_streams[split][source_start + offset:source_start + offset + size]):
                raise ValueError("Balanced tokens disagree with their declared source spans.")
        complete = count == base["tokens"]
        if entry.get("complete_document") is not complete:
            raise ValueError("Balanced complete-document flag disagrees with token cropping.")
        if entry.get("eos_preserved") is not (int(streams[split][target_start + count - 1]) == tokenizer.eos_id):
            raise ValueError("Balanced EOS-preserved flag disagrees with stored tokens.")
        summary = stats[language][split]
        summary["documents"] += 1
        summary["tokens"] += count
        ranges[(language, split)].append((target_start, target_start + count))
        if complete:
            if entry.get("content_sha256") != base["content_sha256"] or entry.get("utf8_bytes") != base["utf8_bytes"]:
                raise ValueError("Balanced complete document lost its source content identity.")
            summary["complete_documents"] += 1
            summary["complete_document_tokens_excluding_eos"] += count - 1
            summary["complete_document_whitespace_words"] += selected[document_id].words
            if summary["utf8_bytes"] is not None:
                summary["utf8_bytes"] += selected[document_id].utf8_bytes
                summary["whitespace_words"] += selected[document_id].words
        else:
            if entry.get("content_sha256") is not None or entry.get("utf8_bytes") is not None:
                raise ValueError("Cropped token spans must not claim complete text hashes or byte counts.")
            summary["cropped_documents"] += 1
            summary["utf8_bytes"] = summary["whitespace_words"] = None
    shares = {}
    for split in SPLITS:
        counts = {language: stats[language][split]["tokens"] for language in LANGUAGES}
        if len(set(counts.values())) != 1:
            raise ValueError("Balanced token shares are not exactly equal across all four languages.")
        expected_quota = manifest.get("balance", {}).get("splits", {}).get(split, {}).get("tokens_per_language")
        if any(count != expected_quota for count in counts.values()):
            raise ValueError("Balanced observed token counts disagree with declared quotas.")
        total = sum(counts.values())
        shares[split] = {"tokens": counts, "shares": {language: count / total if total else None for language, count in counts.items()},
                         "exactly_equal": True}
        for language in LANGUAGES:
            summary = stats[language][split]
            complete_words = summary["complete_document_whitespace_words"]
            summary["complete_document_tokens_per_whitespace_word"] = summary["complete_document_tokens_excluding_eos"] / complete_words if complete_words else None
            summary["tokens_per_whitespace_word"] = None if summary["cropped_documents"] else summary["complete_document_tokens_per_whitespace_word"]
            declared = manifest.get("splits", {}).get(split, {}).get("languages", {}).get(language, {})
            for key in ("documents", "tokens", "utf8_bytes"):
                if declared.get(key) != summary[key]:
                    raise ValueError(f"Balanced manifest {key} disagrees with observed metadata.")
            position = declared.get("token_start")
            for start, end in sorted(ranges[(language, split)]):
                if start != position:
                    raise ValueError("Balanced per-language spans are not contiguous or disagree with declared slice starts.")
                position = end
            if position != declared.get("token_end"):
                raise ValueError("Balanced language metadata disagrees with its declared slice end.")
            if split in ("train", "validation") and summary["tokens"] <= 512:
                warnings.append(f"Balanced {language}/{split} has {summary['tokens']} tokens; it cannot supply a full 512-token shifted window.")
    if prepared_identity(directory) != identity:
        raise ValueError("Balanced artifacts changed while the audit was running.")
    return {"identity": identity, "by_language": stats, "token_shares": shares,
            "cropped_text_policy": "No decoded-byte or whitespace-word counts invented for a cropped token prefix; full-slice fertility is null if any document is cropped.",
            "source_token_spans_verified": len(metadata)}


def audit_data(selected_jsonl: str | Path, prepared_dir: str | Path, balanced_dir: str | Path | None = None) -> dict:
    """Verify data identities and round-trip samples, without test performance checks."""
    selected_path, prepared_dir = Path(selected_jsonl).resolve(), Path(prepared_dir).resolve()
    identity = prepared_identity(prepared_dir)
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    selected_sha = file_sha256(selected_path)
    if manifest.get("input", {}).get("sha256") != selected_sha:
        raise ValueError("Prepared input SHA does not match the supplied selected documents JSONL.")
    tokenizer = load_tokenizer(prepared_dir / "tokenizer.json")
    if manifest.get("tokenizer", {}).get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("Prepared manifest vocabulary disagrees with its tokenizer.")
    if manifest.get("tokenizer", {}).get("kind") == "bpe" and manifest["tokenizer"].get("fitting_split") != "train":
        raise ValueError("BPE preparation must declare training-only tokenizer fitting.")
    selected = _scan_selected(selected_path)
    warnings: list[str] = []
    policy = _tokenizer_policy(prepared_dir / "tokenizer.json", tokenizer)
    if not policy["all_256_byte_symbols_present"]:
        warnings.append("Tokenizer does not expose the full ByteLevel alphabet; universal UTF-8 byte coverage is unconfirmed.")
    if policy["unk_token"] is not None:
        warnings.append("Tokenizer has an UNK token; inspect sample unknown-token counts before freezing it.")
    if policy["normalizer"] is not None:
        warnings.append("Tokenizer declares normalization; only the sampled texts and Unicode probe were checked for exact preservation.")
    stats = {language: {split: _summary() for split in SPLITS} for language in LANGUAGES}
    with ExitStack() as stack:
        streams = _open_streams(stack, prepared_dir, tokenizer.vocab_size)
        metadata = _read_metadata(prepared_dir / "documents.jsonl", streams)
        if selected.keys() != metadata.keys():
            raise ValueError("Prepared metadata document IDs differ from selected documents.")
        for document_id, document in selected.items():
            entry = metadata[document_id]
            if (entry["language"], entry["split"], entry.get("content_sha256"), entry.get("utf8_bytes")) != (
                document.language, document.split, document.content_sha256, document.utf8_bytes
            ):
                raise ValueError("Prepared document metadata does not match selected source content.")
            if int(streams[document.split][entry["token_start"] + entry["tokens"] - 1]) != tokenizer.eos_id:
                raise ValueError("Prepared complete document does not end with EOS.")
            summary = stats[document.language][document.split]
            summary["documents"] += 1
            summary["utf8_bytes"] += document.utf8_bytes
            summary["tokens"] += entry["tokens"]
            summary["tokens_excluding_eos"] += entry["tokens"] - 1
            summary["whitespace_words"] += document.words
        for language in LANGUAGES:
            for split in SPLITS:
                summary = stats[language][split]
                _finish_summary(summary)
                declared = manifest.get("splits", {}).get(split, {}).get("languages", {}).get(language, {})
                for key in ("documents", "utf8_bytes", "tokens"):
                    if declared.get(key, 0) != summary[key]:
                        raise ValueError(f"Prepared manifest {key} disagrees with actual document metadata.")
                if split in ("train", "validation") and summary["tokens"] <= 512:
                    warnings.append(f"Prepared {language}/{split} has {summary['tokens']} tokens; it cannot supply a full 512-token shifted window.")
        sample_checks = _sample_checks(selected_path, selected, metadata, streams, tokenizer, policy)
        balanced = _audit_balanced(Path(balanced_dir).resolve(), identity, metadata, streams, selected, tokenizer, warnings, stack) if balanced_dir is not None else None
    if file_sha256(selected_path) != selected_sha or prepared_identity(prepared_dir) != identity:
        raise ValueError("Data artifacts changed while the audit was running.")
    return {"schema_version": 1, "status": "passed_with_warnings" if warnings else "passed",
            "selected_documents": {"path": str(selected_path), "sha256": selected_sha, "documents": len(selected)},
            "prepared": {"identity": identity, "by_language": stats}, "balanced": balanced,
            "fertility_definition": "sum(encoded text tokens excluding one appended EOS per complete document) / sum(len(original_text.split())); whitespace words are not language-aware linguistic words",
            "tokenizer": policy, "sample_checks": sample_checks, "warnings": warnings,
            "scope": "Artifact integrity, mechanical corpus statistics and bounded tokenizer checks; no model performance, test-text sample inspection, or comprehensive corpus-quality judgment."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--balanced", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_data(args.documents, args.prepared, args.balanced)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "warnings": report["warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
