"""Audit immutable raw JSONL text without changing, normalizing or selecting it.

Scripts and classifier scores are diagnostic review signals, never language
certificates. Source-sampling probabilities belong to the acquisition ledger;
this report describes only the supplied sample, without population weighting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlsplit

LANGUAGES = ("en", "de", "ar", "fa")
_BOILERPLATE = {
    "en": ("accept cookies", "privacy policy", "all rights reserved", "javascript"),
    "de": ("cookies akzeptieren", "datenschutzerklärung", "alle rechte vorbehalten", "javascript"),
    "ar": ("سياسة الخصوصية", "جميع الحقوق محفوظة", "ملفات تعريف الارتباط", "جافا سكريبت"),
    "fa": ("تمامی حقوق محفوظ", "کلیه حقوق محفوظ", "سیاست حفظ حریم", "جاوا اسکریپت"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def domain_of(url: object) -> str:
    """Return the exact lower-case hostname; never guess registrable domains."""
    if not isinstance(url, str) or not url.strip():
        return "[missing]"
    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        return parsed.hostname.lower().rstrip(".") if parsed.hostname else "[invalid]"
    except ValueError:
        return "[invalid]"


@lru_cache(maxsize=8192)
def _script(character: str) -> str:
    if not character.isalpha():
        return "nonletter"
    name = unicodedata.name(character, "")
    for script in ("ARABIC", "LATIN", "CYRILLIC", "HEBREW", "GREEK", "DEVANAGARI"):
        if script in name:
            return script.lower()
    if "CJK" in name or "IDEOGRAPH" in name:
        return "han"
    return "other_letter"


def _quantiles(values: list[int | float]) -> dict:
    if not values:
        return {key: None for key in ("min", "p10", "p50", "p90", "p99", "max")}
    ordered = sorted(values)
    def at(fraction):
        index = (len(ordered) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"min": ordered[0], "p10": at(.1), "p50": at(.5), "p90": at(.9), "p99": at(.99), "max": ordered[-1]}


def text_statistics(text: str, language: str) -> dict:
    """Mechanical signals only. Original text is not returned or rewritten."""
    characters = Counter(text)
    scripts = Counter()
    controls = Counter()
    for character, count in characters.items():
        scripts[_script(character)] += count
        category = unicodedata.category(character)
        if category in {"Cc", "Cf"} and character not in "\t\n\r":
            controls[f"U+{ord(character):04X}"] += count
    words = text.split()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counts = Counter(lines)
    repeated_line_chars = sum(len(line) * (count - 1) for line, count in line_counts.items())
    line_chars = sum(map(len, lines))
    ngrams = Counter(tuple(words[index:index + 5]) for index in range(max(0, len(words) - 4)))
    ngram_slots = max(0, len(words) - 4)
    duplicate_ngram_slots = sum(count - 1 for count in ngrams.values())
    lower = text.lower()
    result = {
        "characters": len(text), "utf8_bytes": len(text.encode("utf-8")),
        "whitespace_words": len(words), "nonempty_lines": len(lines),
        "script_characters": dict(sorted(scripts.items())),
        "letters": sum(count for script, count in scripts.items() if script != "nonletter"),
        "control_and_format_characters": dict(sorted(controls.items())),
        "replacement_characters": characters["\ufffd"], "nul_characters": characters["\0"],
        "zwnj": characters["\u200c"], "zwj": characters["\u200d"],
        "arabic_yeh": characters["ي"], "persian_yeh": characters["ی"],
        "arabic_kaf": characters["ك"], "persian_keheh": characters["ک"],
        "arabic_indic_digits": sum(characters[chr(value)] for value in range(0x660, 0x66A)),
        "persian_digits": sum(characters[chr(value)] for value in range(0x6F0, 0x6FA)),
        "repeated_line_extra_characters": repeated_line_chars,
        "repeated_line_extra_fraction": repeated_line_chars / line_chars if line_chars else 0.0,
        "repeated_word_5gram_extra_slots": duplicate_ngram_slots,
        "word_5gram_slots": ngram_slots,
        "repeated_word_5gram_extra_fraction": duplicate_ngram_slots / ngram_slots if ngram_slots else 0.0,
        "longest_identical_character_run": max((len(match.group()) for match in re.finditer(r"(.)\1+", text, re.DOTALL)), default=1 if text else 0),
        "boilerplate_phrase_counts": {phrase: lower.count(phrase) for phrase in _BOILERPLATE.get(language, ()) if phrase in lower},
        "possible_mojibake_marker_counts": {marker: text.count(marker) for marker in ("Ã", "Â", "â€", "ðŸ") if marker in text},
    }
    flags = []
    if not text.strip():
        flags.append("empty")
    if result["replacement_characters"] or result["nul_characters"]:
        flags.append("encoding_or_nul")
    if result["possible_mojibake_marker_counts"]:
        flags.append("possible_mojibake_marker")
    if result["repeated_line_extra_fraction"] >= .2 or result["repeated_word_5gram_extra_fraction"] >= .2:
        flags.append("repetition")
    if result["longest_identical_character_run"] >= 20:
        flags.append("character_run")
    if result["boilerplate_phrase_counts"]:
        flags.append("boilerplate_phrase")
    if len(words) < 50:
        flags.append("short_under_50_whitespace_words")
    expected_script = "arabic" if language in {"ar", "fa"} else "latin"
    letters = result["letters"]
    expected_share = scripts[expected_script] / letters if letters else 0.0
    result["expected_script_letter_share"] = expected_share
    if expected_share < .8:
        flags.append("script_mixing_or_mismatch")
    if language == "fa" and not result["zwnj"]:
        flags.append("persian_without_zwnj")
    result["review_flags"] = flags
    return result


def scoring_spans(text: str, window: int = 4096) -> list[tuple[int, int]]:
    if len(text) <= 3 * window:
        return [(0, len(text))]
    middle = (len(text) - window) // 2
    return [(0, window), (middle, middle + window), (len(text) - window, len(text))]


class FastTextLID:
    """Local official lid.176 artifact; confidence is not calibrated probability."""
    def __init__(self, model_path: str | Path, expected_sha256: str):
        path = Path(model_path)
        if sha256_file(path) != expected_sha256:
            raise ValueError("Language-ID model hash differs from the pinned identity.")
        try:
            import fasttext
        except ImportError as exc:
            raise ImportError("Local LID requires the separately installed fasttext module.") from exc
        self.model = fasttext.load_model(str(path))
        self.identity = {"path": str(path.resolve()), "sha256": expected_sha256,
                         "bytes": path.stat().st_size, "model": "official fastText lid.176",
                         "documentation": "https://fasttext.cc/docs/en/language-identification.html",
                         "scoring": "up to three nonoverlapping 4096-character spans; whitespace joined only in the scoring copy",
                         "score_scope": "uncalibrated classifier score; diagnostic only; no automatic filtering"}

    def score(self, text: str) -> dict:
        spans = scoring_spans(text)
        segment_texts = [" ".join(text[start:end].split()) for start, end in spans]
        combined = " ".join(segment_texts)
        if not combined:
            return {"top_language": None, "top_score": None, "top3": [], "spans": spans, "window_predictions": []}
        def predict(value):
            labels, scores = self.model.predict(value, k=3)
            return [{"language": label.removeprefix("__label__"), "score": float(score)} for label, score in zip(labels, scores)]
        top = predict(combined)
        windows = [predict(value)[0] if value else {"language": None, "score": None} for value in segment_texts]
        return {"top_language": top[0]["language"], "top_score": top[0]["score"], "top3": top,
                "spans": spans, "window_predictions": windows}


def calibrate_lid(classifier, controls: list[dict]) -> dict:
    """Check declared local controls; this does not estimate corpus accuracy."""
    if (not controls or any(not isinstance(row.get("id"), str) or not row["id"] for row in controls)
            or len({row.get("id") for row in controls}) != len(controls)):
        raise ValueError("Calibration controls need distinct IDs and must be nonempty.")
    records, confusion = [], Counter()
    for row in controls:
        if not isinstance(row.get("text"), str) or "expected_language" not in row:
            raise ValueError("A control needs text and an explicit expected_language (possibly null).")
        result = classifier.score(row["text"])
        expected = row["expected_language"]
        correct = result["top_language"] == expected if expected is not None else None
        if expected is not None:
            confusion[(expected, result["top_language"])] += 1
        records.append({**row, "prediction": result, "correct": correct})
    labelled = [record for record in records if record["correct"] is not None]
    return {"schema_version": 1, "status": "diagnostic_controls_executed", "model": classifier.identity,
            "controls_sha256": hashlib.sha256(_canonical(controls)).hexdigest(),
            "labelled_controls": len(labelled), "correct_controls": sum(record["correct"] for record in labelled),
            "confusion": [{"expected": expected, "predicted": predicted, "count": count}
                          for (expected, predicted), count in sorted(confusion.items(), key=lambda item: str(item[0]))],
            "records": records,
            "limits": "Authored native and confusable controls are a sanity check, not held-out corpus calibration; uncertain/mixed controls have no forced language. Scores do not certify correctness."}


def _review_excerpt(text: str) -> dict:
    if len(text) <= 2400:
        spans = [(0, len(text))]
    else:
        midpoint = (len(text) - 600) // 2
        spans = [(0, 1200), (midpoint, midpoint + 600), (len(text) - 600, len(text))]
    return {"character_spans": spans, "parts": [text[start:end] for start, end in spans], "complete_document": len(text) <= 2400}


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).hexdigest()


def select_review_samples(documents: list[dict], *, seed: int, ordinary: int, risk: int) -> list[dict]:
    """Seeded ordinary probability sample plus explicit targeted risk slate."""
    selected = []
    for language in LANGUAGES:
        rows = sorted((row for row in documents if row["language"] == language), key=lambda row: _rank(seed, row["id"]))
        ordinary_rows = rows[:ordinary]
        seen = {row["id"] for row in ordinary_rows}
        selected.extend({**row, "review_selection": {"kind": "ordinary", "population_documents": len(rows),
                         "sampled_documents": len(ordinary_rows), "inclusion_fraction": len(ordinary_rows) / len(rows),
                         "rule": "lowest SHA256(seed,NUL,id) within language"}} for row in ordinary_rows)
        pools = defaultdict(list)
        for row in rows:
            if row["id"] in seen:
                continue
            for flag in row["review_strata"]:
                pools[flag].append(row)
        cursors, chosen = Counter(), []
        while len(chosen) < risk:
            progressed = False
            # Quality flags must remain represented even with thousands of domains.
            strata = sorted(pools, key=lambda value: (2 if value.startswith("domain:") else 1 if value.startswith("source:") else 0, value))
            for stratum in strata:
                while cursors[stratum] < len(pools[stratum]) and pools[stratum][cursors[stratum]]["id"] in seen:
                    cursors[stratum] += 1
                if cursors[stratum] >= len(pools[stratum]):
                    continue
                row = pools[stratum][cursors[stratum]]
                cursors[stratum] += 1
                seen.add(row["id"])
                chosen.append({**row, "review_selection": {"kind": "targeted_risk", "selected_stratum": stratum,
                               "stratum_eligible_after_ordinary": len(pools[stratum]),
                               "rule": "round robin sorted strata, seeded rank, unique IDs; overlaps make this a targeted slate, not a prevalence sample"}})
                progressed = True
                if len(chosen) >= risk:
                    break
            if not progressed:
                break
        selected.extend(chosen)
    return selected


def _aggregate(rows: list[dict]) -> dict:
    result = {"documents": len(rows), "utf8_bytes": sum(row["statistics"]["utf8_bytes"] for row in rows),
              "characters": sum(row["statistics"]["characters"] for row in rows),
              "whitespace_words": sum(row["statistics"]["whitespace_words"] for row in rows),
              "length_utf8_bytes": _quantiles([row["statistics"]["utf8_bytes"] for row in rows]),
              "length_whitespace_words": _quantiles([row["statistics"]["whitespace_words"] for row in rows]),
              "upstream_language_score": _quantiles([row["upstream_language_score"] for row in rows if row["upstream_language_score"] is not None]),
              "upstream_language_score_missing": sum(row["upstream_language_score"] is None for row in rows)}
    if all(row.get("tokens_excluding_eos") is not None for row in rows):
        result["tokens_excluding_eos"] = sum(row["tokens_excluding_eos"] for row in rows)
    for key in ("replacement_characters", "nul_characters", "zwnj", "zwj", "arabic_yeh", "persian_yeh", "arabic_kaf", "persian_keheh", "arabic_indic_digits", "persian_digits"):
        result[key] = sum(row["statistics"][key] for row in rows)
        result[key + "_documents"] = sum(row["statistics"][key] > 0 for row in rows)
    result["zwnj_per_million_characters"] = result["zwnj"] * 1e6 / result["characters"] if result["characters"] else 0.0
    result["script_characters"] = dict(sum((Counter(row["statistics"]["script_characters"]) for row in rows), Counter()))
    result["control_and_format_characters"] = dict(sum((Counter(row["statistics"]["control_and_format_characters"]) for row in rows), Counter()))
    result["possible_mojibake_marker_counts"] = dict(sum((Counter(row["statistics"]["possible_mojibake_marker_counts"]) for row in rows), Counter()))
    result["review_flag_documents"] = dict(Counter(flag for row in rows for flag in row["statistics"]["review_flags"]))
    scored = [row for row in rows if row.get("lid") is not None]
    result["local_lid"] = {"scored_documents": len(scored),
                           "top_language_documents": dict(Counter(row["lid"]["top_language"] or "[empty]" for row in scored)),
                           "disagrees_with_declared_documents": sum(row["lid"]["top_language"] != row["language"] for row in scored),
                           "any_window_disagrees_with_declared_documents": sum(any(window["language"] != row["language"] for window in row["lid"]["window_predictions"]) for row in scored),
                           "score_under_0_8_documents": sum((row["lid"]["top_score"] or 0) < .8 for row in scored)}
    return result


def audit_sources(inputs: list[str | Path], output: str | Path, *, seed: int = 0,
                  ordinary: int = 50, risk: int = 50, classifier=None,
                  calibration: dict | None = None, token_counter=None, tokenizer_identity=None) -> dict:
    if ordinary < 0 or risk < 0:
        raise ValueError("Review counts must be nonnegative.")
    paths = sorted({Path(path).resolve() for path in inputs}, key=str)
    if not paths or len(paths) != len(inputs):
        raise ValueError("Supply distinct nonempty raw input paths.")
    output = Path(output)
    if output.exists():
        raise ValueError("Quality output already exists; choose a new output directory.")
    if classifier is not None and (not calibration or calibration.get("model", {}).get("sha256") != classifier.identity["sha256"]):
        raise ValueError("LID requires the calibration report for the same pinned artifact.")
    code_sha256 = sha256_file(__file__)
    identities = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths]
    documents, ids, texts, text_hashes = [], set(), {}, defaultdict(list)
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
                raw = json.loads(line.decode("utf-8"))
                document_id, text, language = raw.get("id"), raw.get("text"), raw.get("language")
                if not isinstance(document_id, str) or not document_id or document_id in ids or not isinstance(text, str) or language not in LANGUAGES:
                    raise ValueError(f"Invalid or duplicate raw document at {path}:{line_number}.")
                ids.add(document_id)
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if raw.get("text_sha256", content_hash) != content_hash:
                    raise ValueError("Raw text_sha256 disagrees with the original text.")
                score = raw.get("source_language_score")
                if score is not None and (not isinstance(score, (float, int)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("Upstream language scores must be null or finite numbers in [0,1].")
                statistics = text_statistics(text, language)
                source = str(raw.get("source") or "[missing]")
                row = {"id": document_id, "language": language, "source": source,
                       "source_kind": raw.get("source_kind"), "source_title": raw.get("source_title"),
                       "revision": raw.get("revision"), "source_shard": raw.get("source_shard"),
                       "source_row": raw.get("source_row"), "source_row_group": raw.get("source_row_group"),
                       "url": raw.get("url"), "domain": domain_of(raw.get("url")),
                       "upstream_language_score": score, "text_sha256": content_hash,
                       "raw_location": {"path": str(path), "line": line_number, "byte_offset": offset},
                       "sampling_provenance": {key: value for key, value in raw.items() if key.startswith("sampling") or key in {"stratum", "window", "inclusion_probability", "acquisition_key"}},
                       "statistics": statistics, "lid": classifier.score(text) if classifier else None,
                       "tokens_excluding_eos": token_counter(text) if token_counter else None}
                strata = statistics["review_flags"] + ["source:" + source, "domain:" + row["domain"]]
                if row["lid"]:
                    if row["lid"]["top_language"] != language:
                        strata.append("local_lid_disagreement")
                    if any(window["language"] != language for window in row["lid"]["window_predictions"]):
                        strata.append("local_lid_window_disagreement")
                    if (row["lid"]["top_score"] or 0) < .8:
                        strata.append("local_lid_score_under_0_8")
                row["review_strata"] = sorted(set(strata))
                documents.append(row)
                texts[document_id] = _review_excerpt(text)
                text_hashes[content_hash].append(document_id)
    if not documents:
        raise ValueError("Raw inputs contain no documents.")
    for identity in identities:
        if sha256_file(identity["path"]) != identity["sha256"]:
            raise ValueError("A raw input changed while it was being audited.")
    if sha256_file(__file__) != code_sha256:
        raise ValueError("The source-quality code changed while auditing.")
    grouped = {}
    for name, key_fn in (("language", lambda row: (row["language"],)),
                         ("language_source", lambda row: (row["language"], row["source"])),
                         ("language_source_kind", lambda row: (row["language"], row["source"], row["source_kind"])),
                         ("language_source_shard", lambda row: (row["language"], row["source"], row["source_shard"])),
                         ("language_source_shard_stratum", lambda row: (row["language"], row["source"], row["source_shard"], row["sampling_provenance"].get("sampling", {}).get("stratum"))),
                         ("language_domain", lambda row: (row["language"], row["domain"]))):
        groups = defaultdict(list)
        for row in documents:
            groups[key_fn(row)].append(row)
        entries = []
        for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
            aggregate = _aggregate(rows)
            if name == "language_domain":
                aggregate = {key: value for key, value in aggregate.items() if key in {
                    "documents", "utf8_bytes", "characters", "whitespace_words", "tokens_excluding_eos",
                    "zwnj", "zwnj_documents", "replacement_characters", "local_lid"}}
            entries.append({"slice": list(key), **aggregate})
        grouped[name] = entries
    by_language = {row["slice"][0]: row for row in grouped["language"]}
    alerts = []
    for domain in grouped["language_domain"]:
        total = by_language[domain["slice"][0]]
        domain["document_share_within_language"] = domain["documents"] / total["documents"]
        domain["byte_share_within_language"] = domain["utf8_bytes"] / total["utf8_bytes"] if total["utf8_bytes"] else 0.0
        if "tokens_excluding_eos" in domain:
            domain["token_share_within_language"] = domain["tokens_excluding_eos"] / total["tokens_excluding_eos"] if total["tokens_excluding_eos"] else 0.0
        if domain["byte_share_within_language"] > .1:
            alerts.append({"kind": "domain_over_10_percent_bytes_review_alert", "language": domain["slice"][0],
                           "domain": domain["slice"][1], "byte_share": domain["byte_share_within_language"]})
    summary = {"schema_version": 1, "scope": "Immutable supplied raw sample; no population prevalence estimate, automatic filtering, normalization or quality certification",
               "inputs": identities, "code_sha256": code_sha256,
               "tokenizer": tokenizer_identity, "local_lid": classifier.identity if classifier else None,
               "lid_calibration_sha256": hashlib.sha256(_canonical(calibration)).hexdigest() if calibration else None,
               "summary": _aggregate(documents), "slices": grouped, "review_alerts": alerts,
               "exact_text_duplicate_groups": [{"text_sha256": key, "ids": value} for key, value in sorted(text_hashes.items()) if len(value) > 1],
               "near_duplicate_scope": "Not computed here; the separate corpus MinHash/exclusion audit handles supplied-corpus screening",
               "review_selection": {"seed": seed, "ordinary_per_language": ordinary, "risk_per_language": risk,
                                     "ordinary_scope": "seeded within-language document sample; byte-weighted review not implied",
                                     "risk_scope": "targeted round-robin source/domain/quality strata; no prevalence inference"},
               "metric_notes": {"script": "Unicode letter names only; Arabic script includes Persian, Urdu and other languages",
                                "whitespace_words": "Mechanical whitespace counts, not linguistic word segmentation",
                                "repetition": "Extra exact stripped-line characters or exact whitespace-word 5gram slots; no approximate semantic claim",
                                "encoding": "U+FFFD/NUL flags do not identify all mojibake or upstream deletion",
                                "lid": "Scores and 0.8 review alert are uncalibrated; no acceptance threshold or script-based language assignment",
                                "token_counts": "Optional frozen-tokenizer counts exclude EOS; absent unless explicitly supplied"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".tmp-", dir=output.parent) as temporary:
        root = Path(temporary)
        source_bytes = Path(__file__).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != code_sha256:
            raise ValueError("The source-quality code changed before report freeze.")
        (root / "source_quality.py").write_bytes(source_bytes)
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (root / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in documents:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        reviews = select_review_samples(documents, seed=seed, ordinary=ordinary, risk=risk)
        with (root / "review_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in reviews:
                handle.write(json.dumps({**row, "excerpt": texts[row["id"]], "semantic_review": None}, ensure_ascii=False, allow_nan=False) + "\n")
        if calibration:
            (root / "lid_calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (root / "artifact_manifest.json").write_text(json.dumps({"artifacts": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(root.iterdir())}}, indent=2) + "\n", encoding="utf-8")
        root.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ordinary", type=int, default=50)
    parser.add_argument("--risk", type=int, default=50)
    parser.add_argument("--lid-model")
    parser.add_argument("--lid-sha256")
    parser.add_argument("--lid-controls")
    parser.add_argument("--tokenizer", help="Optional frozen Tokenizers JSON; does not fit or modify it")
    args = parser.parse_args()
    if any((args.lid_model, args.lid_sha256, args.lid_controls)) and not all((args.lid_model, args.lid_sha256, args.lid_controls)):
        parser.error("--lid-model, --lid-sha256 and --lid-controls must be supplied together")
    classifier = FastTextLID(args.lid_model, args.lid_sha256) if args.lid_model else None
    calibration = calibrate_lid(classifier, json.loads(Path(args.lid_controls).read_text(encoding="utf-8"))) if classifier else None
    token_counter, identity = None, None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
        tokenizer.encode_special_tokens = True
        token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
        identity = {"path": str(Path(args.tokenizer).resolve()), "sha256": sha256_file(args.tokenizer), "eos_included": False}
    summary = audit_sources(args.input, args.output, seed=args.seed, ordinary=args.ordinary, risk=args.risk,
                            classifier=classifier, calibration=calibration, token_counter=token_counter,
                            tokenizer_identity=identity)
    print(json.dumps({"output": str(Path(args.output).resolve()), "documents": summary["summary"]["documents"],
                      "utf8_bytes": summary["summary"]["utf8_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
