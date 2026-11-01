import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mqr.config import ExperimentConfig, HeadConfig, ModelConfig, TrainConfig
from mqr.data import TokenStream
from mqr.prepare import prepare_jsonl
from mqr.training import run_training
from mqr.tuning import batch_config, benchmark_config, choose_shared_batch, optimizer_attempt, profile_config, run_sweep


def tiny(kind="mqr"):
    return ExperimentConfig(
        ModelConfig(vocab_size=17, layers=1, d_model=16, attention_heads=2, ffn_dim=32, context_length=8),
        HeadConfig(kind=kind, d_e=8, d_c=16, num_queries=1 if kind == "dense" else 2),
        TrainConfig(batch_size=2, gradient_accumulation=2, max_steps=20, warmup_steps=2,
                    checkpoint_every=0, eval_every=0, loss_chunk_positions=5, num_threads=2))


def token_stream():
    return TokenStream((torch.arange(255) * 7) % 17, 8, 17)


@pytest.mark.parametrize("kind", ["dense", "mqr"])
def test_direct_benchmark_matches_existing_training_updates(kind, tmp_path):
    config = tiny(kind)
    stream = token_stream()
    expected = run_training(config, stream, None, tmp_path / "trainer", stop_after=3)
    checkpoint = torch.load(expected["checkpoint"], map_location="cpu", weights_only=True)
    result = benchmark_config(config, stream, tmp_path / "benchmark", warmup_steps=1,
                              measured_steps=2, return_state=True)
    assert result["completed_steps"] == 3
    for name, value in checkpoint["model"].items():
        torch.testing.assert_close(result["model_state"][name], value, rtol=1e-6, atol=1e-7)
    assert result["checkpoint_files_written"] == 0
    assert not list((tmp_path / "benchmark").rglob("*.pt"))
    assert result["successful_measured_tokens"] == 2 * 4 * 8
    assert result["amp_overflows"] == 0 and result["timing_eligible"]


def test_batch_grid_preserves_windows_order_and_optimizer_schedule(tmp_path):
    base = tiny()
    results = []
    for microbatch in (1, 2, 4):
        config = batch_config(base, microbatch, global_batch=4)
        assert config.training.max_steps == base.training.max_steps
        assert config.training.warmup_steps == base.training.warmup_steps
        results.append(benchmark_config(config, token_stream(), tmp_path / f"b{microbatch}",
                                        warmup_steps=1, measured_steps=2, return_state=True))
    assert all(result["window_sha256"] == results[0]["window_sha256"] for result in results)
    for result in results[1:]:
        for name, value in results[0]["model_state"].items():
            torch.testing.assert_close(result["model_state"][name], value, rtol=2e-5, atol=2e-7)
    with pytest.raises(ValueError, match="divisor"):
        batch_config(base, 3, 4)


def test_shared_batch_requires_both_orders_stable_amp_and_identical_windows():
    jobs = []
    for batch in (1, 2):
        for head in ("dense", "mqr"):
            for repeat in (0, 1):
                jobs.append({"batch_size": batch, "head": head, "repeat": repeat, "status": "complete",
                             "summary": {"timing_eligible": True, "window_sha256": ["same", "windows"],
                                         "memory_headroom_eligible": True,
                                         "weighted_tokens_per_second": 100 * batch}})
    assert choose_shared_batch(jobs, [1, 2], 2) == 2
    jobs[-1]["summary"]["timing_eligible"] = False
    assert choose_shared_batch(jobs, [1, 2], 2) == 1
    jobs[-1]["summary"]["timing_eligible"] = True
    jobs[-1]["summary"]["memory_headroom_eligible"] = False
    assert choose_shared_batch(jobs, [1, 2], 2) == 1
    jobs[-1]["summary"]["memory_headroom_eligible"] = True
    jobs[-1]["summary"]["window_sha256"] = ["different"]
    with pytest.raises(ValueError, match="identical token windows"):
        choose_shared_batch(jobs, [1, 2], 2)


def test_amp_skipped_attempt_is_retained_and_disqualifies_timing(tmp_path, monkeypatch):
    import mqr.tuning as tuning
    original = tuning.optimizer_attempt
    calls = []

    def skip_once(model, optimizer, scaler, config, x, y, completed):
        calls.append(completed)
        if len(calls) == 2:
            return {"full_nll": 1.0, "gradient_norm": None, "learning_rate": config.training.learning_rate,
                    "skipped": True, "attempted_tokens": int(y.numel()), "scaler_scale": 32768}
        return original(model, optimizer, scaler, config, x, y, completed)

    monkeypatch.setattr(tuning, "optimizer_attempt", skip_once)
    result = benchmark_config(tiny(), token_stream(), tmp_path / "retry", warmup_steps=1, measured_steps=1)
    events = [json.loads(line) for line in (tmp_path / "retry" / "attempts.jsonl").read_text().splitlines()]
    assert result["amp_overflows"] == 1 and result["timing_eligible"] is False
    assert result["completed_steps"] == 2 and len(events) == 3
    measured = [event for event in events if event["phase"] == "measured"]
    assert len(measured) == 2 and measured[0]["skipped"]
    assert result["weighted_tokens_per_second"] == 32 / sum(event["attempt_seconds"] for event in measured)


def test_optional_gradient_diagnostics_observe_unclipped_gradients(monkeypatch):
    import mqr.tuning as tuning
    from mqr.model import LanguageModel
    config = tiny()
    model = LanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                  betas=config.training.betas, weight_decay=config.training.weight_decay)
    x, y = token_stream().sample(4, torch.Generator().manual_seed(1))
    observed = []
    original = torch.nn.utils.clip_grad_norm_

    def callback(current):
        assert current is model and any(parameter.grad is not None for parameter in model.parameters())
        observed.append("callback")

    def clip(*args, **kwargs):
        observed.append("clip")
        return original(*args, **kwargs)

    monkeypatch.setattr(tuning.torch.nn.utils, "clip_grad_norm_", clip)
    optimizer_attempt(model, optimizer, None, config, x, y, 0, before_clip=callback)
    assert observed == ["callback", "clip"]


def prepared_fixture(tmp_path):
    source = tmp_path / "documents.jsonl"
    source.write_text("\n".join(json.dumps(item) for item in (
        {"id": "train", "language": "en", "split": "train", "text": "Distinct training words. " * 40},
        {"id": "valid", "language": "de", "split": "validation", "text": "Separate validation words. " * 40})))
    root = tmp_path / "prepared"
    prepare_jsonl(source, root)
    return root


def test_sweep_dry_run_needs_no_gpu_and_exposes_bounded_fixed_batch_protocol(tmp_path):
    bundle = prepared_fixture(tmp_path)
    plan = run_sweep(data_dir=bundle, output_root=tmp_path / "unused", dry_run=True)
    assert plan["global_batch_sequences"] == 32 and plan["global_batch_tokens"] == 16384
    assert plan["repeat_head_order"] == [["dense", "mqr"], ["mqr", "dense"]]
    assert plan["budget_seconds"] == 3600 and plan["profiles_separate_from_timing"]
    assert plan["full_lr_schedule_preserved"] and not (tmp_path / "unused").exists()


def test_oom_in_clean_worker_stops_larger_shared_candidates(tmp_path, monkeypatch):
    import mqr.tuning as tuning
    bundle = prepared_fixture(tmp_path)
    monkeypatch.setattr(tuning, "inspect_device", lambda device: {"device": device})
    calls = []

    def oom(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"status": "oom", "error": "simulated CUDA OOM"}))
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(tuning.subprocess, "run", oom)
    result = run_sweep(data_dir=bundle, output_root=tmp_path / "sweep", batches=(1, 2, 4),
                       budget_seconds=60, worker_timeout=30, profiles=False)
    assert len(calls) == 1
    assert result["jobs"][0]["status"] == "oom"
    assert result["best_shared_batch"] is None
    assert "stopped for both heads" in result["stop_reason"]
    assert result["protocol_completed"] is False


def test_common_oom_boundary_is_complete_only_after_an_eligible_smaller_batch(tmp_path, monkeypatch):
    import mqr.tuning as tuning
    bundle = prepared_fixture(tmp_path)
    monkeypatch.setattr(tuning, "inspect_device", lambda device: {"device": device})
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        if len(calls) <= 4:
            summary = {"status": "complete", "timing_eligible": True, "memory_headroom_eligible": True,
                       "weighted_tokens_per_second": 100, "window_sha256": ["identical"]}
            code = 0
        else:
            summary, code = {"status": "oom"}, 3
        (output / "summary.json").write_text(json.dumps(summary))
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(tuning.subprocess, "run", worker)
    result = run_sweep(data_dir=bundle, output_root=tmp_path / "bounded", batches=(1, 2, 4),
                       budget_seconds=60, worker_timeout=30, profiles=False)
    assert len(calls) == 5 and result["best_shared_batch"] == 1
    assert result["status"] == "complete_with_shared_oom_boundary" and result["protocol_completed"]


@pytest.mark.parametrize("failure", ["failed", "timeout"])
def test_unexpected_worker_failures_are_not_reported_as_successful_oom_boundaries(tmp_path, monkeypatch, failure):
    import mqr.tuning as tuning
    bundle = prepared_fixture(tmp_path)
    monkeypatch.setattr(tuning, "inspect_device", lambda device: {"device": device})

    def worker(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"status": "failed", "error": "unexpected backend error"}))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(tuning.subprocess, "run", worker)
    result = run_sweep(data_dir=bundle, output_root=tmp_path / "bad", batches=(1,),
                       budget_seconds=60, worker_timeout=30, profiles=False)
    assert result["status"] == "failed_trial_" + failure
    assert result["protocol_completed"] is False and not result["profiles"]


def test_failed_cuda_profile_never_completes_the_protocol(tmp_path, monkeypatch):
    import mqr.tuning as tuning
    bundle = prepared_fixture(tmp_path)
    monkeypatch.setattr(tuning, "inspect_device", lambda device: {"device": device})

    def worker(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        if "profile-worker" in command:
            summary, code = {"status": "failed", "error": "CUDA activity requested but no CUDA operation time recorded"}, 1
        else:
            summary = {"status": "complete", "timing_eligible": True, "memory_headroom_eligible": True,
                       "weighted_tokens_per_second": 100, "window_sha256": ["identical"]}
            code = 0
        (output / "summary.json").write_text(json.dumps(summary))
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(tuning.subprocess, "run", worker)
    result = run_sweep(data_dir=bundle, output_root=tmp_path / "profile-failure", batches=(1,),
                       budget_seconds=600, worker_timeout=30, profiles=True)
    assert result["best_shared_batch"] == 1
    assert result["status"] == "failed_or_incomplete_profile" and result["protocol_completed"] is False
    assert result["profiles"][0]["summary"]["error"].startswith("CUDA activity requested")


def test_small_cpu_worker_runs_in_an_isolated_process_without_checkpoints(tmp_path):
    bundle = prepared_fixture(tmp_path)
    config = replace(tiny(), model=replace(tiny().model, vocab_size=259))
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "worker"
    result = subprocess.run([sys.executable, "-m", "mqr.tuning", "worker", "--config", str(config_path),
                             "--data-dir", str(bundle), "--output-dir", str(output),
                             "--warmup-steps", "1", "--measured-steps", "1", "--cpu-test"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "complete"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful_measured_tokens"] == 32
    assert not list(output.rglob("*.pt"))


def test_cpu_profiler_outputs_real_trace_and_distinguishes_workloads(tmp_path):
    result = profile_config(tiny(), token_stream(), tmp_path / "profiles", warmup_steps=1)
    assert result["instrumentation_excluded_from_sweep_timing"] is True
    for scope, description in result["scopes"].items():
        trace = json.loads((tmp_path / "profiles" / description["trace"]).read_text())
        averages = json.loads((tmp_path / "profiles" / description["key_averages"]).read_text())
        assert trace["traceEvents"] and averages["operations"]
        assert description["cuda_activity_requested"] is False
    assert result["scopes"]["full_training"]["positions"] == 32
    assert result["scopes"]["head_only"]["positions"] == 16


def test_large_traces_are_explicitly_dropped_but_key_averages_survive(tmp_path):
    result = profile_config(tiny(), token_stream(), tmp_path / "limited", warmup_steps=0, trace_budget_bytes=1)
    assert result["retained_trace_bytes"] == 0
    assert not list((tmp_path / "limited").glob("*.trace.json"))
    for scope in result["scopes"].values():
        assert scope["trace"] is None and scope["trace_retained"] is False
        assert "dropped" in scope["trace_retention"]
        assert (tmp_path / "limited" / scope["key_averages"]).is_file()


def test_tuning_notebook_is_valid_and_bounded():
    notebook = json.loads(Path("notebooks/mqr_tuning.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    joined = "\n".join(sources)
    assert "RUN_TUNING = True" in joined
    assert "budget_seconds=3600" in joined
    assert "--no-deps" in joined and "tokenizers==0.22.2" in joined
    assert 'if not result["protocol_completed"]:' in joined
    assert joined.index("shutil.make_archive") < joined.index('if not result["protocol_completed"]:')
