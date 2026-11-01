import json

import numpy as np
import pytest

from mqr.data import load_tokenizer
from mqr.prepare import prepare_jsonl
from mqr.provenance import file_sha256


def source(path, suffix=""):
    rows = [{"id": language + split + suffix, "language": language, "split": split,
             "text": text + " " + split + suffix}
            for language, text in (("en", "A book about rivers and cities."),
                                   ("de", "Ein Buch über Flüsse und Städte."),
                                   ("ar", "كتاب عن الأنهار والمدن الجميلة."),
                                   ("fa", "کتابی دربارهٔ رودها و شهرهای زیبا می‌خوانیم."))
            for split in ("train", "validation", "test")]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def test_reused_bpe_is_byte_identical_and_never_refitted(tmp_path, monkeypatch):
    original = tmp_path / "original.jsonl"
    source(original)
    fitted = tmp_path / "fitted"
    first = prepare_jsonl(original, fitted, vocab_size=512, tokenizer_kind="bpe")
    fitting = [json.loads(line) for line in (fitted / "tokenizer_fit.jsonl").open(encoding="utf-8")]
    assert all("train" in row["id"] for row in fitting)
    for language in ("en", "de", "ar", "fa"):
        assert sum(row["sample_utf8_bytes"] for row in fitting if row["language"] == language) == first["tokenizer"]["balanced_fitting_utf8_bytes"][language]
    later = tmp_path / "later.jsonl"
    rows = source(later, " independent")
    monkeypatch.setattr("mqr.prepare._train_bpe", lambda *a, **k: pytest.fail("Frozen BPE was refitted"))
    destination = tmp_path / "reused"
    result = prepare_jsonl(later, destination, vocab_size=first["tokenizer"]["vocab_size"],
                           tokenizer_kind="bpe", tokenizer_path=fitted / "tokenizer.json")
    assert file_sha256(destination / "tokenizer.json") == file_sha256(fitted / "tokenizer.json")
    assert result["tokenizer"]["balanced_fitting_utf8_bytes"] == {}
    assert result["tokenizer"]["frozen_source"]["sha256"] == file_sha256(fitted / "tokenizer.json")
    tokenizer = load_tokenizer(fitted / "tokenizer.json")
    for split in ("train", "validation", "test"):
        expected = sum([tokenizer.encode(row["text"]) for row in rows if row["split"] == split], [])
        assert np.load(destination / (split + ".npy")).tolist() == expected


def test_reused_tokenizer_must_match_kind_and_actual_vocabulary(tmp_path):
    input_path = tmp_path / "source.jsonl"
    source(input_path)
    output = tmp_path / "fitted"
    first = prepare_jsonl(input_path, output, vocab_size=512, tokenizer_kind="bpe")
    with pytest.raises(ValueError, match="kind differs"):
        prepare_jsonl(input_path, tmp_path / "badkind", tokenizer_path=output / "tokenizer.json")
    with pytest.raises(ValueError, match="vocabulary size differs"):
        prepare_jsonl(input_path, tmp_path / "badsize", vocab_size=first["tokenizer"]["vocab_size"] + 1,
                      tokenizer_kind="bpe", tokenizer_path=output / "tokenizer.json")
