import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from mqr import confirmation as c
from mqr.data import ByteTokenizer
from mqr.provenance import file_sha256
from mqr.tracking import read_authoritative_metrics


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    ByteTokenizer().save(root / "tokenizer.json")
    splits = {}
    for seed, split in enumerate(("train", "validation", "test")):
        values = np.random.default_rng(seed + 220).integers(0, 256, size=36000, dtype=np.uint32)
        np.save(root / (split + ".npy"), values)
        splits[split] = {"tokens": len(values), "languages": {
            language: {"token_start": index * 9000, "token_end": (index + 1) * 9000, "tokens": 9000}
            for index, language in enumerate(c.LANGUAGES)}}
    files = {name: {"bytes": (root / name).stat().st_size, "sha256": file_sha256(root / name)}
             for name in ("train.npy", "validation.npy", "test.npy", "tokenizer.json")}
    (root / "manifest.json").write_text(json.dumps({"splits": splits, "files": files}), encoding="utf-8")
    return root


def test_configs_preserve_controls_fresh_schedule_and_three_seeds():
    for seed in (0, 1, 2):
        for arm in c.ARMS:
            config = c.confirmation_config(65536, arm, seed)
            assert config.training.max_steps == 4000 and config.training.warmup_steps == 50
            assert config.training.batch_size * config.training.gradient_accumulation == 32
            assert config.model.layers == 12 and config.model.context_length == 512
            assert config.head.temperature == 1 and not config.head.normalize_queries
            assert config.head.num_queries == (1 if arm in ("dense", "direct1_scale") else 4)
    with pytest.raises(ValueError, match="seed"):
        c.confirmation_config(65536, "dense", 3)


def test_balanced_sampling_replays_and_never_crosses_language_boundaries(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    tokens = np.load(prepared / "train.npy")
    first = torch.Generator().manual_seed(101)
    replay = torch.Generator().manual_seed(101)
    for _ in range(5):
        x, y, records = c.sample_global(tokens, manifest, first, context_length=512)
        rx, ry, rerecords = c.sample_global(tokens, manifest, replay, context_length=512)
        assert torch.equal(x, rx) and torch.equal(y, ry) and records == rerecords
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert Counter(row["language"] for row in records) == {key: 8 for key in c.LANGUAGES}
        for index, row in enumerate(records):
            span = manifest["splits"]["train"]["languages"][row["language"]]
            assert span["token_start"] <= row["token_start"] < row["token_end"] <= span["token_end"]
            assert y[index].tolist() == tokens[row["token_start"] + 1:row["token_end"]].tolist()


def test_volume_gate_prevents_relabeling_the_readiness_sample(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    with pytest.raises(ValueError, match="Broader data gate"):
        c.validate_volume(manifest)


def test_all_five_tiny_arms_share_backbone_windows_and_keep_checkpoints(prepared, tmp_path):
    backbones, windows = set(), []
    for arm in c.ARMS:
        output = tmp_path / arm
        result = c.run_confirmation(prepared, output, arm=arm, seed=0, device="cpu", cpu_test=True, stop_after=2)
        assert result["completed_steps"] == 2 and result["successful_tokens"] == 512
        assert result["checkpoint"]["cpu_reload"] and result["final_state_audit"]["finite"]
        initialization = json.loads((output / "initialization.json").read_text())
        backbones.add(initialization["initial_backbone_sha256"])
        assert initialization["transform"]["applied"] == arm.endswith("_scale")
        changed = set(initialization["changed_parameters"])
        assert changed <= ({"query_projection.weight", "query_projection.bias"} if arm.endswith("_scale") else set())
        if arm.endswith("_scale"):
            assert "query_projection.weight" in changed
        for name in initialization["head_parameters_before"]:
            if name not in changed:
                assert initialization["head_parameters_before"][name] == initialization["head_parameters_after"][name]
        events = read_authoritative_metrics(output / "metrics.jsonl")
        windows.append([row["window_sha256"] for row in events if row["event"] == "train"])
        assert [row["step"] for row in events if row["event"] == "validation"] == [0, 1, 2]
    assert len(backbones) == 1 and all(value == windows[0] for value in windows)


def test_cpu_checkpoint_resume_matches_uninterrupted_training_with_dropout(prepared, tmp_path, monkeypatch):
    original_config = c.confirmation_config
    def dropout_config(*args, **kwargs):
        config = original_config(*args, **kwargs)
        return replace(config, model=replace(config.model, dropout=.15))
    monkeypatch.setattr(c, "confirmation_config", dropout_config)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    args = dict(arm="mqr_scale", seed=2, device="cpu", cpu_test=True)
    c.run_confirmation(prepared, whole, stop_after=4, **args)
    c.run_confirmation(prepared, resumed, stop_after=2, **args)
    c.run_confirmation(prepared, resumed, stop_after=4, resume=True, **args)
    first = torch.load(whole / "last.pt", weights_only=True)
    second = torch.load(resumed / "last.pt", weights_only=True)
    for key in first["model"]:
        assert torch.equal(first["model"][key], second["model"][key]), key
    for key, state in first["optimizer"]["state"].items():
        for label, tensor in state.items():
            assert torch.equal(tensor, second["optimizer"]["state"][key][label])
    assert torch.equal(first["sampler_rng"], second["sampler_rng"])
    events = read_authoritative_metrics(resumed / "metrics.jsonl")
    assert [row["step"] for row in events if row["event"] == "train"] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="frozen"):
        c.run_confirmation(prepared, resumed, stop_after=4, resume=True, chunk=64, **args)


def test_amp_retry_keeps_same_successful_global_batch(prepared, tmp_path, monkeypatch):
    original = c.optimizer_attempt
    calls = []
    def with_skip(model, optimizer, scaler, config, x, y, completed):
        calls.append(c.window_digest(x, y))
        if len(calls) == 1:
            return {"skipped": True, "attempted_tokens": int(y.numel()), "full_nll": 1.0}
        return original(model, optimizer, scaler, config, x, y, completed)
    monkeypatch.setattr(c, "optimizer_attempt", with_skip)
    result = c.run_confirmation(prepared, tmp_path / "retry", arm="dense", seed=0,
                                device="cpu", cpu_test=True, stop_after=2)
    assert calls[0] == calls[1] and calls[1] != calls[2]
    assert result["amp_overflows"] == 1 and result["attempted_tokens"] == 768


def test_costs_include_abandoned_work_and_mark_unclosed_sessions(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"session_id": "first", "event": "start", "elapsed_session_seconds": 1},
            {"session_id": "first", "event": "train", "step": 2, "elapsed_session_seconds": 5},
            {"session_id": "second", "event": "resume", "step": 1, "elapsed_session_seconds": 2},
            {"session_id": "second", "event": "session_end", "elapsed_session_seconds": 4}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = c.recorded_costs(path)
    assert result["recorded_worker_wall_seconds"] == 9
    assert not result["complete_wall_accounting"]
