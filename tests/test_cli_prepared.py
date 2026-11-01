"""Exercise the prepared-data contract through the public command-line API."""

import json
import math
import shutil

import pytest

from mqr.cli import main
from mqr.config import ExperimentConfig, HeadConfig, ModelConfig, TrainConfig
from mqr.data import load_tokenizer
from mqr.prepare import prepare_jsonl
from mqr.provenance import file_sha256, prepared_identity
from mqr.training import load_model


def _prepare_bundle(tmp_path, tokenizer_kind):
    text_by_language = {
        "en": (
            "A reader studies the river and writes a short story.",
            "The gardener watches clouds above the mountain.",
            "Tomorrow students will describe a new photograph.",
        ),
        "de": (
            "Eine Leserin betrachtet den Fluss und schreibt eine kurze Geschichte.",
            "Der Gärtner beobachtet die Wolken über dem Berg.",
            "Morgen beschreiben die Schülerinnen ein neues Foto.",
        ),
        "ar": (
            "يقرأ الطالب قصة قصيرة ثم يكتب ملاحظة عن النهر.",
            "تراقب البستانية الغيوم فوق الجبل في الصباح.",
            "سيصف التلاميذ صورة جديدة في الفصل غدا.",
        ),
        "fa": (
            "یک دانشجو داستان کوتاهی می‌خواند و درباره رودخانه یادداشت می‌نویسد.",
            "باغبان هر صبح ابرهای بالای کوه را تماشا می‌کند.",
            "فردا دانش‌آموزان یک عکس تازه را در کلاس توصیف می‌کنند.",
        ),
    }
    records = [
        {"id": f"{language}-{split}", "language": language, "split": split,
         "text": texts[split_index]}
        for split_index, split in enumerate(("train", "validation", "test"))
        for language, texts in text_by_language.items()
    ]
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    manifest = prepare_jsonl(source, output, tokenizer_kind=tokenizer_kind, vocab_size=300)
    # The GPU host should need only the prepared bundle, not the original JSONL.
    source.unlink()
    return output, manifest["tokenizer"]["vocab_size"]


def _replacement_bundle(original, destination):
    shutil.copytree(original, destination)
    tokenizer_path = destination / "tokenizer.json"
    descriptor = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = descriptor["model"]["vocab"]
    first, second = sorted((token for token in vocab if vocab[token] >= 3), key=vocab.get)[:2]
    vocab[first], vocab[second] = vocab[second], vocab[first]
    tokenizer_path.write_text(json.dumps(descriptor, ensure_ascii=False), encoding="utf-8")
    # This is a different but internally valid bundle, rather than a broken hash.
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["tokenizer.json"] = {
        "sha256": file_sha256(tokenizer_path), "bytes": tokenizer_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return destination


@pytest.mark.parametrize("tokenizer_kind", ["byte", "bpe"])
def test_prepared_bundle_training_evaluation_and_tokenizer_binding(tmp_path, capsys, tokenizer_kind):
    if tokenizer_kind == "bpe":
        pytest.importorskip("tokenizers")
    data_dir, vocab_size = _prepare_bundle(tmp_path, tokenizer_kind)
    config = ExperimentConfig(
        ModelConfig(vocab_size=vocab_size, layers=1, d_model=16, attention_heads=2,
                    ffn_dim=32, context_length=8),
        HeadConfig(kind="mqr", num_queries=2, d_e=8, d_c=16),
        TrainConfig(seed=11, batch_size=1, max_steps=2, warmup_steps=0,
                    loss_chunk_positions=4, eval_every=0, checkpoint_every=0, num_threads=1),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    run_dir = tmp_path / "run"
    train_args = ["train", "--config", str(config_path), "--data-dir", str(data_dir),
                  "--run-dir", str(run_dir)]
    assert main([*train_args, "--stop-after", "1"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["completed_steps"] == 1
    expected_identity = prepared_identity(data_dir)
    _, _, payload = load_model(first["checkpoint"])
    assert payload["data_identity"] == expected_identity
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["data_identity"] == expected_identity

    assert main([*train_args, "--resume", first["checkpoint"]]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["completed_steps"] == 2
    assert resumed["predicted_tokens"] == 16
    assert main(["evaluate", "--checkpoint", resumed["checkpoint"], "--data-dir", str(data_dir),
                 "--split", "test", "--batches", "2", "--batch-size", "1"]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["predicted_tokens"] == 16
    assert math.isfinite(metrics["full_nll"]) and metrics["full_nll"] > 0
    assert 0 <= metrics["accuracy"] <= 1

    if tokenizer_kind == "bpe":
        different = _replacement_bundle(data_dir, tmp_path / "different-tokenizer")
        assert load_tokenizer(different / "tokenizer.json").vocab_size == vocab_size
        assert prepared_identity(different)["tokenizer_sha256"] != expected_identity["tokenizer_sha256"]
        with pytest.raises(SystemExit) as evaluation_error:
            main(["evaluate", "--checkpoint", resumed["checkpoint"], "--data-dir", str(different),
                  "--batches", "1"])
        assert evaluation_error.value.code == 2
        assert "Evaluation tokenizer does not match" in capsys.readouterr().err
        with pytest.raises(SystemExit) as generation_error:
            main(["generate", "--checkpoint", resumed["checkpoint"],
                  "--tokenizer", str(different / "tokenizer.json"), "--prompt", "A reader"])
        assert generation_error.value.code == 2
        assert "Tokenizer identity does not match" in capsys.readouterr().err
        with pytest.raises(SystemExit) as resume_error:
            main(["train", "--config", str(config_path), "--data-dir", str(different),
                  "--run-dir", str(run_dir), "--resume", resumed["checkpoint"]])
        assert resume_error.value.code == 2
        assert "identity" in capsys.readouterr().err
