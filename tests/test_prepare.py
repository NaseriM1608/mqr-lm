import json

import numpy as np
import pytest

from mqr.data import ByteTokenizer, TokenStream, load_tokenizer
from mqr.prepare import file_sha256, prepare_jsonl


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def records():
    return [
        {"id": "en-train", "language": "en", "split": "train", "text": "The quiet river flows near the village.", "source": "authored-smoke", "revision": "v1"},
        {"id": "de-train", "language": "de", "split": "train", "text": "Am Morgen lesen wir ein Buch über die Stadt."},
        {"id": "ar-train", "language": "ar", "split": "train", "text": "نقرأ كتابا جديدا في المكتبة هذا الصباح."},
        {"id": "fa-train", "language": "fa", "split": "train", "text": "امروز در کتابخانه یک داستان تازه می‌خوانیم."},
        {"id": "en-valid", "language": "en", "split": "validation", "text": "An unseen sentence reserved for validation."},
        {"id": "fa-test", "language": "fa", "split": "test", "text": "این جمله فقط برای آزمون نهایی نگه داشته می‌شود."},
    ]


def test_prepare_separates_documents_and_records_reproducible_artifacts(tmp_path, records):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, records)
    output = tmp_path / "prepared"
    manifest = prepare_jsonl(source, output)
    tokenizer = ByteTokenizer()
    for split in ("train", "validation", "test"):
        expected = sum([tokenizer.encode(record["text"]) for record in records if record["split"] == split], [])
        actual = np.load(output / f"{split}.npy", mmap_mode="r")
        assert actual.tolist() == expected
        assert actual.dtype == np.dtype("<u4")
        assert manifest["splits"][split]["tokens"] == len(expected)
    metadata = [json.loads(line) for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in metadata] == [record["id"] for record in records]
    assert all("text" not in record for record in metadata)
    assert metadata[0]["source"] == "authored-smoke"
    assert metadata[0]["revision"] == "v1"
    for name, artifact in manifest["files"].items():
        assert file_sha256(output / name) == artifact["sha256"]
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    repeated = prepare_jsonl(source, tmp_path / "prepared-again")
    assert repeated == manifest
    assert load_tokenizer(output / "tokenizer.json").vocab_size == 259


@pytest.mark.parametrize("duplicate", ["id", "content", "whitespace"])
def test_duplicate_ids_and_cross_split_content_rejected_before_writing(tmp_path, records, duplicate):
    if duplicate == "id":
        records[-1]["id"] = records[0]["id"]
    elif duplicate == "content":
        records[-1]["text"] = records[0]["text"]
    else:
        records[-1]["text"] = "  " + records[0]["text"].replace(" ", "\n") + "  "
    source = tmp_path / "source.jsonl"
    write_jsonl(source, records)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_jsonl(source, output)
    assert not output.exists()


def test_splits_must_be_explicit_and_include_validation(tmp_path, records):
    source = tmp_path / "source.jsonl"
    del records[0]["split"]
    write_jsonl(source, records)
    with pytest.raises(ValueError, match="split"):
        prepare_jsonl(source, tmp_path / "missing-split")
    records[0]["split"] = "train"
    write_jsonl(source, [record for record in records if record["split"] == "train"])
    with pytest.raises(ValueError, match="validation"):
        prepare_jsonl(source, tmp_path / "missing-validation")


def test_output_directory_is_immutable_and_empty_test_split_is_allowed(tmp_path, records):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [record for record in records if record["split"] != "test"])
    output = tmp_path / "prepared"
    prepare_jsonl(source, output)
    assert np.load(output / "test.npy").size == 0
    with pytest.raises(FileExistsError):
        prepare_jsonl(source, output)
    with pytest.raises(ValueError, match="context_length"):
        TokenStream.from_file(output / "test.npy", 2)


def test_bpe_fits_only_balanced_training_text_and_retains_byte_coverage(tmp_path, records):
    pytest.importorskip("tokenizers")
    source = tmp_path / "source.jsonl"
    write_jsonl(source, records)
    first_output = tmp_path / "first"
    manifest = prepare_jsonl(source, first_output, vocab_size=300, tokenizer_kind="bpe", seed=8)
    markers = manifest["tokenizer"]["reserved_markers"]
    assert len(markers) == 12
    assert len(set(markers.values())) == 12
    assert markers["<|lang_fa|>"] is not None
    tokenizer = load_tokenizer(first_output / "tokenizer.json")
    literal = "literal <|entailment|> سلام"
    ids = tokenizer.encode(literal, add_eos=False)
    assert markers["<|entailment|>"] not in ids
    assert tokenizer.decode(ids) == literal
    selected = manifest["tokenizer"]["balanced_fitting_utf8_bytes"]
    assert set(selected) == {"en", "de", "ar", "fa"}
    assert max(selected.values()) - min(selected.values()) <= 3
    records[-1]["text"] = "HELDOUTTOKENONLY " * 100
    records[-2]["text"] = "VALIDATIONTOKENONLY " * 100
    write_jsonl(source, records)
    second_output = tmp_path / "second"
    prepare_jsonl(source, second_output, vocab_size=300, tokenizer_kind="bpe", seed=8)
    assert file_sha256(first_output / "tokenizer.json") == file_sha256(second_output / "tokenizer.json")
    tokenizer = load_tokenizer(first_output / "tokenizer.json")
    text = "Unseen 字符 🌍 العربية فارسی <|eos|>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.eos_id not in tokenizer.encode("<|eos|>", add_eos=False)
