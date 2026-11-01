"""Bounded, isolated batching and profiler diagnostics; no model checkpoints.

The fixed global batch is sampled once per optimizer attempt, then sliced in
order. Accumulation changes scheduling, not the token windows or LR schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import TokenStream, load_tokenizer
from .kaggle import inspect_device, session_configs
from .model import LanguageModel
from .provenance import file_sha256, prepared_identity
from .training import autocast_context, exact_cross_entropy, learning_rate_at, resolve_device, set_seed


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_fingerprints() -> dict:
    source_dir = Path(__file__).parent
    return {name: file_sha256(source_dir / name) for name in ("tuning.py", "training.py", "heads.py", "model.py")}


def batch_config(config: ExperimentConfig, microbatch: int, global_batch: int = 32) -> ExperimentConfig:
    if type(microbatch) is not int or type(global_batch) is not int or microbatch < 1 or global_batch < 1 or global_batch % microbatch:
        raise ValueError("Microbatch must be a positive divisor of the fixed global batch.")
    if config.model.dropout != 0:
        raise ValueError("This batching protocol requires dropout=0 to isolate microbatch scheduling.")
    return replace(config, training=replace(config.training, batch_size=microbatch,
                                            gradient_accumulation=global_batch // microbatch))


def window_digest(x: torch.Tensor, y: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(x.numpy().tobytes())
    digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def optimizer_attempt(model, optimizer, scaler, config: ExperimentConfig,
                      x: torch.Tensor, y: torch.Tensor, completed_steps: int, *, before_clip=None) -> dict:
    """Match training.py's CE, weighting, AMP, clipping and optimizer semantics."""
    tc = config.training
    device = next(model.parameters()).device
    if x.shape != y.shape or x.shape[0] != tc.batch_size * tc.gradient_accumulation:
        raise ValueError("An optimizer attempt requires one complete fixed global batch.")
    total_valid = int((y != -100).sum().item())
    if not total_valid:
        raise ValueError("Optimizer attempt contains no predicted tokens.")
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate_at(completed_steps, tc)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_value = 0.0
    for offset in range(0, len(x), tc.batch_size):
        micro_x = x[offset:offset + tc.batch_size].to(device)
        micro_y = y[offset:offset + tc.batch_size].to(device)
        with autocast_context(device, tc.precision):
            features, _ = model.backbone(micro_x)
            mean_loss = exact_cross_entropy(model.head, features, micro_y, tc.loss_chunk_positions)
            weighted = mean_loss * ((micro_y != -100).sum() / total_valid)
        if not torch.isfinite(weighted).item():
            raise FloatingPointError("Nonfinite exact training loss.")
        if scaler is None:
            weighted.backward()
        else:
            scaler.scale(weighted).backward()
        loss_value += weighted.detach().item()
    if scaler is not None:
        scaler.unscale_(optimizer)
    if before_clip is not None:
        before_clip(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip,
                                               error_if_nonfinite=scaler is None)
    old_scale = scaler.get_scale() if scaler is not None else None
    if scaler is None:
        optimizer.step()
        skipped = False
    else:
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
    return {"full_nll": loss_value, "gradient_norm": float(grad_norm) if math.isfinite(float(grad_norm)) else None,
            "learning_rate": lr, "skipped": skipped, "attempted_tokens": total_valid,
            "scaler_scale": scaler.get_scale() if scaler is not None else None}


def _state(config):
    torch.set_num_threads(config.training.num_threads)
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    model = LanguageModel(config).to(device).train()
    tc = config.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, betas=tc.betas,
                                  weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if tc.precision == "fp16" else None
    generator = torch.Generator(device="cpu").manual_seed(tc.seed + 101)
    return model, optimizer, scaler, generator, device


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_config(config: ExperimentConfig, stream: TokenStream, output_dir: str | Path,
                     *, warmup_steps: int = 3, measured_steps: int = 8,
                     data_identity: dict | None = None, return_state: bool = False) -> dict:
    """Small direct benchmark; optional CPU-only state return is for parity tests."""
    if warmup_steps < 0 or measured_steps < 1 or warmup_steps + measured_steps > min(32, config.training.max_steps):
        raise ValueError("Invalid bounded warmup/measured steps.")
    if stream.context_length != config.model.context_length or stream.max_token_id >= config.model.vocab_size:
        raise ValueError("Token stream does not match the benchmark configuration.")
    if return_state and config.training.device != "cpu":
        raise ValueError("State return is restricted to small CPU parity tests.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.to_dict())
    provenance = {"config": config.to_dict(), "data_identity": data_identity,
                  "train_fingerprint": stream.fingerprint,
                  "source_sha256": source_fingerprints(),
                  "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                  "torch_cuda_build": torch.version.cuda, "device": config.training.device,
                                  "gpu": torch.cuda.get_device_name(config.training.device)
                                  if config.training.device.startswith("cuda:") else None,
                                  "hardware_vram_bytes": torch.cuda.get_device_properties(config.training.device).total_memory
                                  if config.training.device.startswith("cuda:") else None},
                  "sampling": "one seeded CPU global batch per attempt, sliced in original sequence order",
                  "checkpoint_policy": "no checkpoints written; workers restart from identical seed/config"}
    write_json(output / "provenance.json", provenance)
    model, optimizer, scaler, generator, device = _state(config)
    global_batch = config.training.batch_size * config.training.gradient_accumulation
    target_steps = warmup_steps + measured_steps
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    completed, attempts, consecutive_overflows = 0, [], 0
    with (output / "attempts.jsonl").open("x", encoding="utf-8") as events:
        while completed < target_steps:
            if len(attempts) >= target_steps + 20:
                raise FloatingPointError("AMP attempt budget exceeded.")
            _synchronize(device)
            started = time.perf_counter()
            x, y = stream.sample(global_batch, generator)
            digest = window_digest(x, y)
            result = optimizer_attempt(model, optimizer, scaler, config, x, y, completed)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            phase = "warmup" if completed < warmup_steps else "measured"
            if result["skipped"]:
                consecutive_overflows += 1
            else:
                completed += 1
                consecutive_overflows = 0
            event = {"attempt": len(attempts) + 1, "completed_steps": completed, "phase": phase,
                     "window_sha256": digest, "attempt_seconds": elapsed,
                     "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                     **result}
            attempts.append(event)
            events.write(json.dumps(event, allow_nan=False) + "\n")
            events.flush()
            if consecutive_overflows >= 20:
                raise FloatingPointError("20 consecutive AMP overflows.")
    measured = [event for event in attempts if event["phase"] == "measured"]
    measured_success = [event for event in measured if not event["skipped"]]
    total_seconds = sum(event["attempt_seconds"] for event in measured)
    successful_tokens = sum(event["attempted_tokens"] for event in measured_success)
    overflows = sum(event["skipped"] for event in attempts)
    capacity = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    summary = {"status": "complete", "config": config.to_dict(), "warmup_successful_steps": warmup_steps,
               "source_sha256": provenance["source_sha256"], "data_identity": data_identity,
               "train_fingerprint": stream.fingerprint,
               "measured_successful_steps": len(measured_success), "completed_steps": completed,
               "successful_measured_tokens": successful_tokens,
               "weighted_tokens_per_second": successful_tokens / total_seconds,
               "median_successful_step_seconds": statistics.median(event["attempt_seconds"] for event in measured_success),
               "amp_overflows": overflows, "timing_eligible": overflows == 0,
               "timing_scope": "synchronized attempts including data sampling/hash/transfer and all AMP retries; excludes setup/profiling/I/O",
               "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "peak_reserved_vram_bytes": peak_reserved, "hardware_vram_bytes": capacity,
               "reserved_capacity_fraction": peak_reserved / capacity if capacity is not None else None,
               "memory_headroom_eligible": peak_reserved <= 0.90 * capacity if capacity is not None else None,
               "memory_headroom_policy": "peak reserved allocator memory <=90% physical GPU capacity; other processes/driver memory excluded",
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "window_sha256": [event["window_sha256"] for event in attempts],
               "quality_claim": False, "checkpoint_files_written": 0}
    write_json(output / "summary.json", summary)
    if return_state:
        return {**summary, "model_state": {name: value.detach().clone() for name, value in model.state_dict().items()},
                "attempts": attempts}
    return summary


def _prepared(config, data_dir):
    root = Path(data_dir)
    identity = prepared_identity(root)
    actual_vocab = load_tokenizer(root / "tokenizer.json").vocab_size
    if actual_vocab != config.model.vocab_size:
        raise ValueError("Verified tokenizer vocabulary differs from worker config.")
    return TokenStream.from_file(root / "train.npy", config.model.context_length, actual_vocab), identity


def profile_config(config: ExperimentConfig, stream: TokenStream, output_dir: str | Path,
                   *, warmup_steps: int = 3, data_identity: dict | None = None,
                   trace_budget_bytes: int = 100 * 1024**2) -> dict:
    """Separate instrumented run: one full update and one detached-head update."""
    if not 0 <= warmup_steps <= 8:
        raise ValueError("Profiler warmup must be between zero and eight steps.")
    if type(trace_budget_bytes) is not int or trace_budget_bytes < 1:
        raise ValueError("Profiler trace retention budget must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, optimizer, scaler, generator, device = _state(config)
    global_batch = config.training.batch_size * config.training.gradient_accumulation
    for step in range(warmup_steps):
        x, y = stream.sample(global_batch, generator)
        result = optimizer_attempt(model, optimizer, scaler, config, x, y, step)
        if result["skipped"]:
            raise FloatingPointError("Profiler warmup overflow; choose a stable configuration.")
    _synchronize(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    scopes = {}
    retained_trace_bytes = 0
    for scope in ("full_training", "head_only"):
        x, y = stream.sample(global_batch, generator)
        if scope == "head_only":
            x, y = x[:config.training.batch_size], y[:config.training.batch_size].to(device)
            with torch.no_grad(), autocast_context(device, config.training.precision):
                features, _ = model.backbone(x.to(device))
            features = features.detach().requires_grad_(True)
        _synchronize(device)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=False, with_stack=False) as profiler:
            with torch.profiler.record_function("mqr." + scope):
                if scope == "full_training":
                    update = optimizer_attempt(model, optimizer, scaler, config, x, y, warmup_steps)
                    if update["skipped"]:
                        raise FloatingPointError("Profiled optimizer update overflowed.")
                else:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, config.training.precision):
                        loss = exact_cross_entropy(model.head, features, y, config.training.loss_chunk_positions)
                        loss = loss / config.training.gradient_accumulation
                    (scaler.scale(loss) if scaler is not None else loss).backward()
                _synchronize(device)
        trace = output / (scope + ".trace.json")
        profiler.export_chrome_trace(str(trace))
        generated_trace_bytes = trace.stat().st_size
        retain = retained_trace_bytes + generated_trace_bytes <= trace_budget_bytes
        if retain:
            retained_trace_bytes += generated_trace_bytes
        else:
            trace.unlink()
        rows = [{"key": item.key, "count": item.count, "cpu_time_total_us": item.cpu_time_total,
                 "self_cpu_time_total_us": item.self_cpu_time_total,
                 "device_time_total_us": getattr(item, "device_time_total", 0),
                 "self_device_time_total_us": getattr(item, "self_device_time_total", 0)}
                for item in profiler.key_averages()]
        write_json(output / (scope + ".key_averages.json"), {"operations": rows})
        scopes[scope] = {"trace": trace.name if retain else None, "trace_generated_bytes": generated_trace_bytes,
                         "trace_retained": retain,
                         "trace_retention": "complete" if retain else "dropped: exceeds per-worker trace retention budget; key averages retained",
                         "key_averages": scope + ".key_averages.json",
                         "positions": global_batch * config.model.context_length if scope == "full_training"
                         else config.training.batch_size * config.model.context_length,
                         "cuda_activity_requested": device.type == "cuda",
                         "cuda_time_recorded": any(row["device_time_total_us"] > 0 for row in rows),
                         "scope": "one global optimizer update" if scope == "full_training"
                         else "one microbatch exact head CE forward/backward, weighted as one global-batch slice; detached backbone features; no optimizer update"}
    cuda_missing = device.type == "cuda" and any(not scope["cuda_time_recorded"] for scope in scopes.values())
    summary = {"status": "failed" if cuda_missing else "complete", "config": config.to_dict(), "data_identity": data_identity,
               "source_sha256": source_fingerprints(), "train_fingerprint": stream.fingerprint,
               "instrumentation_excluded_from_sweep_timing": True,
               "profiling_options": {"record_shapes": True, "profile_memory": False, "with_stack": False},
               "trace_retention_budget_bytes": trace_budget_bytes, "retained_trace_bytes": retained_trace_bytes,
               "error": "CUDA activity requested but no CUDA operation time recorded" if cuda_missing else None,
               "scopes": scopes, "quality_claim": False}
    write_json(output / "summary.json", summary)
    return summary


def choose_shared_batch(jobs: list[dict], batches: list[int], repeats: int) -> int | None:
    eligible = []
    common_windows = None
    for batch in batches:
        selected = [job for job in jobs if job["batch_size"] == batch and job["status"] == "complete"
                    and job["summary"].get("timing_eligible")
                    and job["summary"].get("memory_headroom_eligible") is True]
        if len(selected) != 2 * repeats or {(job["head"], job["repeat"]) for job in selected} != {
                (head, repeat) for head in ("dense", "mqr") for repeat in range(repeats)}:
            continue
        digests = [job["summary"]["window_sha256"] for job in selected]
        if any(value != digests[0] for value in digests):
            raise ValueError("Common candidate workers did not consume identical token windows.")
        if common_windows is not None and digests[0] != common_windows:
            raise ValueError("Different microbatch candidates did not consume identical token windows.")
        common_windows = digests[0]
        rate = math.exp(sum(math.log(job["summary"]["weighted_tokens_per_second"]) for job in selected) / len(selected))
        eligible.append((rate, batch))
    return max(eligible)[1] if eligible else None


def run_sweep(*, data_dir: str | Path, output_root: str | Path, device: str = "cuda:0",
              batches: tuple[int, ...] = (1, 2, 4, 8, 16, 32), repeats: int = 2,
              warmup_steps: int = 3, measured_steps: int = 8, budget_seconds: int = 3600,
              worker_timeout: int = 240, profiles: bool = True, dry_run: bool = False) -> dict:
    if not 60 <= budget_seconds <= 7200 or not 30 <= worker_timeout <= 600:
        raise ValueError("Budget must be 60–7200 seconds; each worker timeout 30–600 seconds.")
    if repeats != 2 or warmup_steps < 1 or measured_steps < 2 or warmup_steps + measured_steps > 32:
        raise ValueError("Use two reversed-order repeats and at most 32 total successful steps per worker.")
    if not batches or tuple(sorted(set(batches))) != tuple(batches) or any(batch not in (1, 2, 4, 8, 16, 32) for batch in batches):
        raise ValueError("Batch grid must be an increasing subset of 1,2,4,8,16,32.")
    data_root = Path(data_dir).resolve()
    identity = prepared_identity(data_root)
    vocab = load_tokenizer(data_root / "tokenizer.json").vocab_size
    base = session_configs("prepared", device, vocab_size=vocab)
    output = Path(output_root).resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("Outputs must be outside the read-only prepared data bundle.")
    plan = {"protocol": "fixed-global32-microbatch-sweep-v1", "global_batch_sequences": 32,
            "global_batch_tokens": 16384, "batches": list(batches), "repeats": repeats,
            "repeat_head_order": [["dense", "mqr"], ["mqr", "dense"]],
            "warmup_successful_steps": warmup_steps, "measured_successful_steps": measured_steps,
            "budget_seconds": budget_seconds, "worker_timeout_seconds": worker_timeout,
            "data_identity": identity, "vocab_size": vocab, "device": device,
            "source_sha256": source_fingerprints(),
            "checkpoint_policy": "none; isolated workers start from identical seeds",
            "full_lr_schedule_preserved": True, "profiles_separate_from_timing": True,
            "implementation_note": "current heads include semantics-preserving unit-temperature and M=1 fast paths added after original calibration; batch1 is rerun here",
            "selection_memory_headroom_policy": "all four candidate workers must have peak_reserved<=90% hardware VRAM",
            "total_profiler_trace_retention_limit_bytes": 200 * 1024**2,
            "quality_claim": False, "dry_run": dry_run}
    if dry_run:
        return plan
    plan["hardware"] = inspect_device(device)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    deadline = time.monotonic() + budget_seconds
    jobs, profile_jobs = [], []
    stop_reason = None
    halt_kind = None

    def launch(name, config, mode, remaining):
        config_path = output / (name + ".config.json")
        write_json(config_path, config.to_dict())
        command = [sys.executable, "-m", "mqr.tuning", mode, "--config", str(config_path),
                   "--data-dir", str(data_root), "--output-dir", str(output / name),
                   "--warmup-steps", str(warmup_steps)]
        if mode == "worker":
            command += ["--measured-steps", str(measured_steps)]
        started = time.monotonic()
        timeout = min(worker_timeout, max(1, int(remaining)))
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                status = "complete" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        summary_path = output / name / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if status != "timeout":
            status = "complete" if status == "complete" and summary.get("status") == "complete" else summary.get("status", "failed")
            if status == "complete" and process.returncode != 0:
                status = "failed"
        return {"status": status, "summary": summary,
                "directory": name, "wall_seconds": time.monotonic() - started, "timeout_seconds": timeout}

    for batch in batches:
        for repeat in range(repeats):
            order = ("dense", "mqr") if repeat == 0 else ("mqr", "dense")
            for head in order:
                remaining = deadline - time.monotonic()
                reserve = 2 * worker_timeout if profiles else 0
                if remaining <= reserve + 30:
                    stop_reason = "wall-clock budget reserved for profiling or exhausted"
                    halt_kind = "budget"
                    break
                name = f"b{batch:02d}_{head}_r{repeat}"
                print("Starting " + name, flush=True)
                job = launch(name, batch_config(base[head], batch), "worker", remaining - reserve)
                jobs.append({"batch_size": batch, "head": head, "repeat": repeat, **job})
                write_json(output / "progress.json", {"jobs": jobs, "stop_reason": stop_reason})
                if job["status"] != "complete":
                    stop_reason = f"{job['status']} at batch={batch} head={head}; larger shared candidates stopped for both heads"
                    halt_kind = job["status"]
                    break
            if stop_reason:
                break
        if stop_reason:
            break
    best = choose_shared_batch(jobs, list(batches), repeats)
    unexpected_trial_failure = halt_kind not in (None, "oom", "budget")
    if profiles and best is not None and not unexpected_trial_failure:
        for head in ("dense", "mqr"):
            remaining = deadline - time.monotonic()
            if remaining < 30:
                halt_kind = "budget"
                stop_reason = "wall-clock budget exhausted before both profiles completed"
                break
            name = f"profile_b{best:02d}_{head}"
            print("Profiling " + name + " (excluded from sweep timings)", flush=True)
            profile_jobs.append({"head": head, "batch_size": best,
                                 **launch(name, batch_config(base[head], best), "profile-worker", remaining)})
            if profile_jobs[-1]["status"] != "complete":
                break
    if unexpected_trial_failure:
        status = "failed_trial_" + str(halt_kind)
    elif best is None:
        status = "no_common_eligible_configuration"
    elif profiles and (len(profile_jobs) != 2 or any(job["status"] != "complete" for job in profile_jobs)):
        status = "failed_or_incomplete_profile"
    elif halt_kind == "budget":
        status = "incomplete_budget"
    elif halt_kind == "oom":
        status = "complete_with_shared_oom_boundary"
    else:
        status = "complete"
    protocol_completed = status in ("complete", "complete_with_shared_oom_boundary")
    summary = {**plan, "jobs": jobs, "profiles": profile_jobs, "best_shared_batch": best,
               "status": status, "protocol_completed": protocol_completed,
               "selection_rule": "largest geometric mean across both heads/order repeats, requiring zero overflows, identical windows and >=10% reserved-memory headroom",
               "stop_reason": stop_reason, "total_wall_seconds": budget_seconds - (deadline - time.monotonic()),
               "interpretation": "configuration-specific cost diagnostic; no convergence, ANN or inference-speed claim"}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--data-dir", required=True)
    sweep.add_argument("--output-root", required=True)
    sweep.add_argument("--device", default="cuda:0")
    sweep.add_argument("--budget-seconds", type=int, default=3600)
    sweep.add_argument("--worker-timeout", type=int, default=240)
    sweep.add_argument("--batches", default="1,2,4,8,16,32")
    sweep.add_argument("--no-profiles", action="store_true")
    sweep.add_argument("--dry-run", action="store_true")
    for name in ("worker", "profile-worker"):
        worker = sub.add_parser(name)
        worker.add_argument("--config", required=True)
        worker.add_argument("--data-dir", required=True)
        worker.add_argument("--output-dir", required=True)
        worker.add_argument("--warmup-steps", type=int, default=3)
        worker.add_argument("--measured-steps", type=int, default=8)
        worker.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "sweep":
        result = run_sweep(data_dir=args.data_dir, output_root=args.output_root, device=args.device,
                           budget_seconds=args.budget_seconds, worker_timeout=args.worker_timeout,
                           batches=tuple(int(value) for value in args.batches.split(",")),
                           profiles=not args.no_profiles, dry_run=args.dry_run)
        print(json.dumps({key: result.get(key) for key in ("status", "best_shared_batch", "stop_reason", "total_wall_seconds", "dry_run")}, indent=2))
        return 0 if result.get("protocol_completed") or result.get("dry_run") else 1
    config = ExperimentConfig.from_json(args.config)
    output = Path(args.output_dir)
    try:
        if args.cpu_test:
            if config.training.device != "cpu" or config.model.layers > 2 or config.model.d_model > 64 or config.model.vocab_size > 1024:
                raise ValueError("CPU tests require an explicitly tiny CPU configuration.")
        else:
            inspect_device(config.training.device)
        stream, identity = _prepared(config, args.data_dir)
        if args.command == "worker":
            result = benchmark_config(config, stream, output, warmup_steps=args.warmup_steps,
                                      measured_steps=args.measured_steps, data_identity=identity)
        else:
            result = profile_config(config, stream, output, warmup_steps=args.warmup_steps, data_identity=identity)
        print(json.dumps({"status": result["status"], "output": str(output)}))
        return 0 if result["status"] == "complete" else 1
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        status = "oom" if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower() else "failed"
        write_json(output / "summary.json", {"status": status, "error": str(exc), "config": config.to_dict()})
        print(f"{status}: {exc}", file=sys.stderr)
        return 3 if status == "oom" else 1


if __name__ == "__main__":
    raise SystemExit(main())
