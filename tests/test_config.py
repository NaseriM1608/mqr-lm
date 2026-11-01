import json

import pytest

from mqr.config import ExperimentConfig, HeadConfig, ModelConfig, TrainConfig


def test_config_json_roundtrip(tmp_path):
    config = ExperimentConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert ExperimentConfig.from_json(path) == config
    assert ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize("payload", [
    {"mystery": 1}, {"model": {"d_modle": 32}}, {"head": {"M": 4}},
    {"training": {"gradient_accumulation_steps": 2}}, {"model": []},
    {"training": {"betas": "0.9,0.95"}}, [],
])
def test_reject_unknown_fields_or_wrong_sections(payload):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(payload)


@pytest.mark.parametrize("kwargs", [
    {"layers": 0}, {"d_model": 30, "attention_heads": 4},
    {"d_model": 12, "attention_heads": 4}, {"context_length": 0},
    {"dropout": 1.0}, {"dropout": float("nan")}, {"vocab_size": True},
])
def test_reject_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"kind": "unknown"}, {"num_queries": 0}, {"d_e": -2},
    {"temperature": 0}, {"temperature": float("inf")}, {"normalize_queries": 1},
])
def test_reject_invalid_head_config(kwargs):
    with pytest.raises(ValueError):
        HeadConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"device": "auto"}, {"device": "cpu", "precision": "bf16"},
    {"precision": "int8"}, {"batch_size": 0}, {"learning_rate": 0},
    {"learning_rate": float("nan")}, {"weight_decay": -1}, {"betas": (0.9, 1)},
    {"min_lr_ratio": 1.1}, {"loss_chunk_positions": 0}, {"seed": -1},
])
def test_reject_invalid_train_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_cuda_precision_and_disabled_intervals_are_valid():
    config = TrainConfig(device="cuda:0", precision="bf16", checkpoint_every=0, eval_every=0)
    assert config.precision == "bf16"
