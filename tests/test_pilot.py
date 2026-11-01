import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mqr.config import ExperimentConfig
from mqr.data import ByteTokenizer
from mqr.model import LanguageModel
from mqr.pilot import (ARMS, LANGUAGES, audit_checkpoint, bank_tensors, build_window_bank,
                       evaluate_bank, finite_state_audit, pilot_configs, retain_checkpoint, run_pilot)
from mqr.provenance import file_sha256
from mqr.training import load_model, set_seed


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    ByteTokenizer().save(root / "tokenizer.json")
    manifests = {}
    for index, split in enumerate(("train", "validation")):
        tokens = np.random.default_rng(index + 110).integers(0, 256, size=36_000, dtype=np.uint32)
        np.save(root / f"{split}.npy", tokens)
        manifests[split] = {"tokens": len(tokens), "languages": {
            language: {"token_start": number * 9000, "token_end": (number + 1) * 9000, "tokens": 9000}
            for number, language in enumerate(LANGUAGES)}}
    files = {name: {"bytes": (root / name).stat().st_size, "sha256": file_sha256(root / name)}
             for name in ("train.npy", "validation.npy", "tokenizer.json")}
    (root / "manifest.json").write_text(json.dumps({"splits": manifests, "files": files}), encoding="utf-8")
    return root


def test_registered_plan_keeps_architecture_controls_and_new_fresh_schedule(prepared, tmp_path):
    output = tmp_path / "not-created"
    plan = run_pilot(data_dir=prepared, output_root=output, seed=1, device="cuda:1", microbatch=4, dry_run=True)
    assert not output.exists()
    assert tuple(plan["arms"]) == ARMS
    assert plan["absolute_stop_after"] == 128
    assert plan["evaluation_steps"] == [0, 16, 32, 64, 128]
    assert plan["fresh_weights_only"]
    assert plan["required_study_seeds"] == [0, 1]
    assert plan["primary_endpoint"]["step"] == 128
    assert plan["diagnostic_support_rule"]["minimum_nll_improvement_each_seed"] == 0.05
    assert plan["validation_bank"]["predicted_tokens"] == 32768
    configs = {name: ExperimentConfig.from_dict(value) for name, value in plan["resolved_configs"].items()}
    for name, config in configs.items():
        assert config.model.layers == 12 and config.model.d_model == 512 and config.model.context_length == 512
        assert config.training.seed == 1 and config.training.device == "cuda:1"
        assert config.training.batch_size == 4 and config.training.gradient_accumulation == 8
        assert config.training.max_steps == 256 and config.training.warmup_steps == 16
        assert config.training.learning_rate == 3e-4
        assert config.head.temperature == 1
        assert not config.head.normalize_queries and not config.head.normalize_output_rows
    assert configs["dense"].head.d_e == 512
    assert configs["direct1"].head.num_queries == 1
    assert configs["direct4"].head.num_queries == 4
    assert configs["mqr"].head.d_c == 256
    assert configs["mqr"] == configs["mqr_scale"]
    assert configs["direct4"] == configs["direct4_scale"]
    other = run_pilot(data_dir=prepared, output_root=output, seed=0, microbatch=8, dry_run=True)
    assert plan["validation_bank"] == other["validation_bank"]
    assert plan["train_probe_bank"] == other["train_probe_bank"]
    assert plan["checkpoint_retention"] == ["mqr", "mqr_scale"]
    assert other["checkpoint_retention"] == ["dense", "mqr", "mqr_scale"]


def test_banks_are_nonoverlapping_language_bounded_and_reject_changed_content(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    tokens = np.load(prepared / "validation.npy")
    bank = build_window_bank(tokens, manifest, split="validation", context_length=512,
                             windows_per_language=16, seed=42)
    spans = []
    for record in bank["records"]:
        lower, upper = record["token_start"], record["token_end"]
        language = manifest["splits"]["validation"]["languages"][record["language"]]
        assert language["token_start"] <= lower < upper <= language["token_end"]
        assert upper - lower == 513
        assert not any(lower < end and upper > start for start, end in spans)
        spans.append((lower, upper))
    x, y = bank_tensors(tokens, bank)
    assert x.shape == y.shape == (64, 512)
    assert torch.equal(x[:, 1:], y[:, :-1])
    tokens[bank["records"][0]["token_start"]] += 1
    with pytest.raises(ValueError, match="content differs"):
        bank_tensors(tokens, bank)
    with pytest.raises(ValueError, match="training or validation"):
        build_window_bank(tokens, manifest, split="test", context_length=8, windows_per_language=1, seed=0)
    manifest["splits"]["validation"]["languages"]["de"]["token_start"] = 0
    with pytest.raises(ValueError, match="overlap"):
        build_window_bank(tokens, manifest, split="validation", context_length=8, windows_per_language=1, seed=0)


def test_evaluation_predictions_and_macro_average_do_not_change_with_microbatch(prepared):
    config = pilot_configs(vocab_size=259, device="cpu", cpu_test=True)["mqr"]
    set_seed(0)
    model = LanguageModel(config).train()
    tokens = np.load(prepared / "validation.npy")
    manifest = json.loads((prepared / "manifest.json").read_text())
    bank = build_window_bank(tokens, manifest, split="validation", context_length=8,
                             windows_per_language=2, seed=42)
    one = evaluate_bank(model, tokens, bank, microbatch=1, chunk_positions=3)
    four = evaluate_bank(model, tokens, bank, microbatch=4, chunk_positions=3)
    assert model.training
    assert one["predicted_tokens"] == four["predicted_tokens"] == 64
    assert one["macro_nll"] == pytest.approx(four["macro_nll"], abs=1e-6)
    assert one["macro_nll"] == pytest.approx(one["full_nll"])
    for language in LANGUAGES:
        assert one["per_language"][language]["predicted_tokens"] == 16
        assert len(one["per_language"][language]["window_nll"]) == 2
        assert one["per_language"][language]["full_nll"] == pytest.approx(four["per_language"][language]["full_nll"], abs=1e-6)


def test_tiny_seven_arm_pilot_preserves_seed_data_and_audits_only_retained_checkpoints(prepared, tmp_path):
    output = tmp_path / "run"
    report = run_pilot(data_dir=prepared, output_root=output, device="cpu", cpu_test=True,
                        microbatch=8, stop_after=2)
    assert report["status"] == "complete"
    assert not report["model_quality_claim"] and not report["throughput_comparison_claim"]
    assert len(list(output.rglob("last.pt"))) == 3
    dense = report["results"]["dense"]
    dense_std = dense["initialization"]["before"]["aggregate_logits"]["centered_std"]
    for name, result in report["results"].items():
        assert result["completed_steps"] == 2
        assert result["successful_tokens"] == result["attempted_tokens"] == 512
        assert result["amp_overflows"] == 0
        assert result["successful_window_sha256"] == dense["successful_window_sha256"]
        assert result["initialization"]["initial_backbone_sha256"] == dense["initialization"]["initial_backbone_sha256"]
        assert set(result["evaluations"]) == {"0", "2"}
        assert result["final_state_audit"]["finite"]
        assert result["final_state_audit"]["optimizer_tensors"] > 0
        initialization = result["initialization"]
        if name.endswith("_scale"):
            transform = initialization["transform"]
            assert transform["applied"] and transform["gain"] > 1
            assert transform["transformed_parameters"] == ["query_projection.weight", "query_projection.bias"]
            assert transform["changed_parameters"] == ["query_projection.weight"]
            assert initialization["after"]["aggregate_logits"]["centered_std"] == pytest.approx(dense_std, rel=1e-5)
            original = report["results"][name.removesuffix("_scale")]["initialization"]
            assert initialization["config_sha256"] == original["config_sha256"]
            assert initialization["before"] == original["before"]
        else:
            assert not initialization["transform"]["applied"]
        events = [json.loads(line) for line in (output / name / "metrics.jsonl").read_text().splitlines()]
        first = next(event for event in events if event["event"] == "train")
        assert first["unscaled_preclip_gradient_groups"]["backbone"]["unscaled_preclip_rms"] > 0
        checkpoint = result["checkpoint"]
        assert checkpoint["retained"] == retain_checkpoint(name, 0)
        if checkpoint["retained"]:
            path = output / name / "last.pt"
            assert checkpoint["cpu_reload"] and checkpoint["finite_model_optimizer_scaler"]
            assert checkpoint["sha256"] == file_sha256(path)
            assert checkpoint["initialization_sha256"] == file_sha256(output / name / "initialization.json")
            loaded, config, state = load_model(path)
            assert config.to_dict() == initialization["canonical_config"]
            assert state["completed_steps"] == 2 and state["predicted_tokens"] == 512
            assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
            assert state["validation_fingerprint"] == report["validation_fingerprint"]
    with pytest.raises(ValueError, match="fresh output"):
        run_pilot(data_dir=prepared, output_root=output, device="cpu", cpu_test=True, stop_after=2)


def test_amp_retry_reuses_batch_and_preserves_successful_stream(prepared, tmp_path, monkeypatch):
    import mqr.pilot as pilot
    real = pilot.optimizer_attempt
    seen = []

    def first_attempt_skips(model, optimizer, scaler, config, x, y, completed_steps, *, before_clip=None):
        seen.append((x.clone(), y.clone()))
        if len(seen) == 1:
            return {"full_nll": 6.0, "gradient_norm": None, "learning_rate": 3e-4 / 16,
                    "skipped": True, "attempted_tokens": x.numel(), "scaler_scale": 32768}
        return real(model, optimizer, scaler, config, x, y, completed_steps, before_clip=before_clip)

    monkeypatch.setattr(pilot, "optimizer_attempt", first_attempt_skips)
    report = run_pilot(data_dir=prepared, output_root=tmp_path / "retry", device="cpu", cpu_test=True,
                        microbatch=8, stop_after=2, arms=("dense", "mqr"))
    assert torch.equal(seen[0][0], seen[1][0]) and torch.equal(seen[0][1], seen[1][1])
    dense, mqr = report["results"]["dense"], report["results"]["mqr"]
    assert dense["amp_overflows"] == 1 and mqr["amp_overflows"] == 0
    assert dense["attempted_tokens"] == 768 and dense["successful_tokens"] == 512
    assert dense["successful_window_sha256"] == mqr["successful_window_sha256"]


def test_finite_audit_rejects_corrupted_optimizer_and_checkpoint(prepared, tmp_path):
    config = pilot_configs(vocab_size=259, device="cpu", cpu_test=True, microbatch=8)["dense"]
    model = LanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter = next(model.parameters())
    optimizer.state[parameter]["test"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        finite_state_audit(model, optimizer)
    report = run_pilot(data_dir=prepared, output_root=tmp_path / "audit", device="cpu", cpu_test=True,
                        microbatch=8, stop_after=1, arms=("dense",))
    path = Path(report["results"]["dense"]["checkpoint"]["path"])
    payload = torch.load(path, weights_only=True)
    next(iter(payload["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(FloatingPointError, match="optimizer"):
        audit_checkpoint(path, config, completed_steps=1, predicted_tokens=256, data_identity=report["data_identity"])


@pytest.mark.parametrize("kwargs,match", [
    ({"seed": 2}, "seeds 0 and 1"),
    ({"microbatch": 3}, "divide"),
    ({"stop_after": 120}, "128"),
    ({"arms": ("dense", "mqr")}, "all seven"),
    ({"stage": "test"}, "learn or init"),
])
def test_registered_gpu_protocol_rejects_unregistered_changes(prepared, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_pilot(data_dir=prepared, output_root=tmp_path / "invalid", dry_run=True, **kwargs)


def test_init_only_audit_has_no_training_or_checkpoint_outputs(prepared, tmp_path):
    output = tmp_path / "init"
    report = run_pilot(data_dir=prepared, output_root=output, device="cpu", cpu_test=True,
                        stage="init", stop_after=1, arms=("dense", "mqr", "mqr_scale"))
    assert report["absolute_stop_after"] == 0 and report["checkpoint_retention"] == []
    assert not list(output.rglob("last.pt"))
    assert all(result["completed_steps"] == 0 for result in report["results"].values())
    assert all(result["evaluations"] == {} for result in report["results"].values())
