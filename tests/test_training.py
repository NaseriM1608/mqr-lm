import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mqr.config import ExperimentConfig, ModelConfig, HeadConfig, TrainConfig
from mqr.data import TokenStream
from mqr.model import LanguageModel
from mqr.training import exact_cross_entropy, run_training, load_model, evaluate


def tiny_config(kind="mqr", steps=8, dropout=0.0):
    return ExperimentConfig(
        ModelConfig(vocab_size=16, layers=1, d_model=16, attention_heads=2,
                    ffn_dim=32, context_length=8, dropout=dropout),
        HeadConfig(kind=kind, d_e=8, d_c=16, num_queries=4 if kind == "mqr" else 1),
        TrainConfig(seed=7, batch_size=2, max_steps=steps, learning_rate=0.003,
                    warmup_steps=2, loss_chunk_positions=3, checkpoint_every=0,
                    eval_every=2, eval_batches=1, num_threads=2),
    )


@pytest.mark.parametrize("kind", ["dense", "direct", "mqr"])
def test_chunked_loss_and_gradients_match_full_ce(kind):
    torch.manual_seed(10)
    head = LanguageModel(tiny_config(kind)).head
    reference = copy.deepcopy(head)
    features = torch.randn(2, 5, 16, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    targets = torch.randint(0, 16, (2, 5))
    targets[0, 2] = -100
    actual = exact_cross_entropy(head, features, targets, chunk_positions=3)
    expected = F.cross_entropy(reference.score_all(reference_features).reshape(-1, 16), targets.reshape(-1))
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(features.grad, reference_features.grad)
    for parameter, other in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad)


def test_all_masked_loss_is_zero_with_backward():
    head = LanguageModel(tiny_config()).head
    features = torch.randn(2, 3, 16, requires_grad=True)
    loss = exact_cross_entropy(head, features, torch.full((2, 3), -100))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(features.grad) == 0


def test_interrupted_resume_matches_uninterrupted_with_dropout(tmp_path):
    config = tiny_config(dropout=0.1)
    train = TokenStream(torch.arange(16).repeat(8), 8)
    validation = TokenStream(torch.arange(15, -1, -1).repeat(8), 8)
    whole = run_training(config, train, validation, tmp_path / "whole")
    part = run_training(config, train, validation, tmp_path / "split", stop_after=3)
    resumed = run_training(config, train, validation, tmp_path / "split", resume=part["checkpoint"])
    _, _, full_state = load_model(whole["checkpoint"])
    _, _, resumed_state = load_model(resumed["checkpoint"])
    assert resumed["completed_steps"] == config.training.max_steps
    assert resumed_state["predicted_tokens"] == full_state["predicted_tokens"]
    assert torch.equal(resumed_state["sampler_rng"], full_state["sampler_rng"])
    for name, weight in full_state["model"].items():
        assert torch.equal(weight, resumed_state["model"][name]), name


def test_resume_rejects_schedule_or_data_change(tmp_path):
    config = tiny_config()
    stream = TokenStream(torch.arange(16).repeat(8), 8)
    result = run_training(config, stream, None, tmp_path / "run", stop_after=2)
    changed = replace(config, training=replace(config.training, max_steps=9))
    with pytest.raises(ValueError, match="configuration"):
        run_training(changed, stream, None, tmp_path / "run", resume=result["checkpoint"])
    other = TokenStream(torch.arange(15, -1, -1).repeat(8), 8)
    with pytest.raises(ValueError, match="fingerprints"):
        run_training(config, other, None, tmp_path / "run", resume=result["checkpoint"])


def test_tiny_model_learns_repeated_target_and_evaluation_is_repeatable(tmp_path):
    config = tiny_config(steps=35)
    config = replace(config, training=replace(config.training, learning_rate=0.01, eval_every=0))
    stream = TokenStream(torch.full((128,), 3), 8)
    result = run_training(config, stream, None, tmp_path / "overfit")
    events = [json.loads(line) for line in (tmp_path / "overfit" / "metrics.jsonl").read_text().splitlines()]
    losses = [event["full_nll"] for event in events if event["event"] == "train"]
    assert losses[-1] < losses[0] * 0.25
    model, _, _ = load_model(result["checkpoint"])
    before = torch.get_rng_state().clone()
    first = evaluate(model, stream, batches=2)
    second = evaluate(model, stream, batches=2)
    assert first == second
    assert torch.equal(before, torch.get_rng_state())
    assert first["accuracy"] == 1.0


def test_existing_run_requires_explicit_resume(tmp_path):
    config = tiny_config()
    stream = TokenStream(torch.arange(16).repeat(8), 8)
    run_training(config, stream, None, tmp_path / "run", stop_after=1)
    with pytest.raises(ValueError, match="already exists"):
        run_training(config, stream, None, tmp_path / "run", stop_after=1)


def test_validation_context_and_resume_tokenizer_identity(tmp_path):
    config = tiny_config()
    stream = TokenStream(torch.arange(16).repeat(8), 8)
    shorter = TokenStream(torch.arange(15, -1, -1).repeat(8), 4)
    with pytest.raises(ValueError, match="Validation stream context"):
        run_training(config, stream, shorter, tmp_path / "bad")
    identity = {"kind": "prepared", "tokenizer_sha256": "original", "manifest_sha256": "abc"}
    result = run_training(config, stream, None, tmp_path / "run", stop_after=1, data_identity=identity)
    with pytest.raises(ValueError, match="identity"):
        run_training(config, stream, None, tmp_path / "run", resume=result["checkpoint"],
                     data_identity={**identity, "tokenizer_sha256": "different"})


def test_rejected_corrupt_log_resume_preserves_manifest(tmp_path):
    config = tiny_config()
    stream = TokenStream(torch.arange(16).repeat(8), 8)
    folder = tmp_path / "run"
    result = run_training(config, stream, None, folder, stop_after=1)
    original_manifest = (folder / "manifest.json").read_bytes()
    log = folder / "metrics.jsonl"
    corrupt_log = b"invalid interior record\n" + log.read_bytes()
    log.write_bytes(corrupt_log)
    with pytest.raises(ValueError, match="interior metrics JSON"):
        run_training(config, stream, None, folder, resume=result["checkpoint"])
    assert (folder / "manifest.json").read_bytes() == original_manifest
    assert log.read_bytes() == corrupt_log


def test_failed_manifest_replacement_preserves_previous_manifest(tmp_path, monkeypatch):
    import mqr.training as training

    config = tiny_config()
    stream = TokenStream(torch.arange(16).repeat(8), 8)
    folder = tmp_path / "run"
    result = run_training(config, stream, None, folder, stop_after=1)
    original_manifest = (folder / "manifest.json").read_bytes()
    original_replace = training.os.replace
    temporary_paths = []

    def fail_manifest_replace(source, destination):
        if Path(destination).name == "manifest.json":
            temporary_paths.append(Path(source))
            assert json.loads(Path(source).read_text(encoding="utf-8"))["implementation"] == "MQR-v1"
            raise OSError("simulated manifest replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(training.os, "replace", fail_manifest_replace)
    for _ in range(2):
        with pytest.raises(OSError, match="simulated manifest"):
            run_training(config, stream, None, folder, resume=result["checkpoint"])
        assert (folder / "manifest.json").read_bytes() == original_manifest
    assert len(set(temporary_paths)) == 2
