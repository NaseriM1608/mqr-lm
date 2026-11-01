"""Fresh, checkpointed five-arm learning confirmation on an approved snapshot.

The first real endpoint is 1,000 updates of a predeclared 4,000-update schedule.
This module never resumes the old readiness pilot under a changed schedule.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, HeadConfig, ModelConfig, TrainConfig
from .data import TokenStream, load_tokenizer
from .head_diagnostics import calibrate_initial_query_gain, diagnose_head
from .kaggle import inspect_device
from .model import LanguageModel
from .pilot import (audit_checkpoint, bank_tensors, build_window_bank, evaluate_bank,
                    finite_state_audit, probe_features, _tensor_state_digest)
from .provenance import file_sha256, prepared_identity
from .tracking import prepare_metrics_append
from .training import _restore_rng, resolve_device, save_checkpoint, set_seed
from .tuning import optimizer_attempt, window_digest

VERSION = "mqr-broader-learning-confirmation-v1"
ARMS = ("dense", "mqr", "mqr_scale", "direct1_scale", "direct4_scale")
LANGUAGES = ("en", "de", "ar", "fa")
EVALUATION_STEPS = (0, 50, 125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def confirmation_config(vocab_size, arm, seed, *, device="cuda:0", chunk=512, cpu_test=False):
    if arm not in ARMS or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Confirmation requires a declared arm and seed0,1,2")
    if chunk not in (64, 512):
        raise ValueError("Freeze execution from the bounded chunk64/512 candidates")
    if cpu_test and device != "cpu":
        raise ValueError("CPU correctness mode requires device=cpu")
    model = (ModelConfig(vocab_size=vocab_size, layers=1, d_model=16, attention_heads=2,
                         ffn_dim=32, context_length=8) if cpu_test else
             ModelConfig(vocab_size=vocab_size, layers=12, d_model=512, attention_heads=8,
                         ffn_dim=2048, context_length=512))
    training = TrainConfig(seed=seed, device=device, precision="fp32" if device == "cpu" else "fp16",
                           batch_size=8, gradient_accumulation=4, max_steps=8 if cpu_test else 4000,
                           warmup_steps=2 if cpu_test else 50, learning_rate=3e-4,
                           min_lr_ratio=.1, checkpoint_every=250, eval_every=0,
                           loss_chunk_positions=chunk, num_threads=2)
    head = HeadConfig(kind="dense" if arm == "dense" else "direct" if arm.startswith("direct") else "mqr",
                      d_e=model.d_model if arm == "dense" else 8 if cpu_test else 128,
                      d_c=12 if cpu_test else 256, num_queries=1 if arm in ("dense", "direct1_scale") else 4)
    config = ExperimentConfig(model, head, training)
    config.validate()
    return config


def sample_global(tokens, manifest, generator, *, context_length):
    """Exactly eight windows per language, shuffled in one fixed global batch."""
    records, arrays = [], []
    spans = manifest["splits"]["train"]["languages"]
    for language in LANGUAGES:
        start, end = spans[language]["token_start"], spans[language]["token_end"]
        if not 0 <= start < end <= len(tokens) or end - start < context_length + 1:
            raise ValueError("Invalid/short language training span")
        offsets = torch.randint(start, end - context_length, (8,), generator=generator).tolist()
        for offset in offsets:
            values = np.asarray(tokens[offset:offset + context_length + 1], dtype=np.int64)
            arrays.append(values)
            records.append({"language": language, "token_start": offset,
                            "token_end": offset + context_length + 1})
    order = torch.randperm(32, generator=generator).tolist()
    values = np.stack([arrays[index] for index in order])
    return (torch.from_numpy(values[:, :-1].copy()), torch.from_numpy(values[:, 1:].copy()),
            [records[index] for index in order])


def validate_volume(manifest, *, cpu_test=False):
    if cpu_test:
        return
    for split in ("train", "validation", "test"):
        spans = manifest["splits"][split]["languages"]
        for language in LANGUAGES:
            count = spans[language]["token_end"] - spans[language]["token_start"]
            minimum = 4_096_000 if split == "train" else 131_328
            if count < minimum:
                raise ValueError(f"Broader data gate failed: {split}/{language} {count} < {minimum}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def recorded_costs(metrics):
    """Count all recorded sessions, including work abandoned after a rollback.

    An interrupted session has only a lower bound through its last durable
    event. Missing tail time is never silently called a complete cost record.
    """
    sessions = {}
    with Path(metrics).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            session = sessions.setdefault(row["session_id"], {"recorded_wall_seconds": 0.0,
                "closed": False, "peak_allocated_vram_bytes": None, "peak_reserved_vram_bytes": None})
            session["recorded_wall_seconds"] = max(session["recorded_wall_seconds"], row.get("elapsed_session_seconds", 0.0))
            session["closed"] |= row["event"] == "session_end"
            for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
                if row.get(key) is not None:
                    session[key] = max(session[key] or 0, row[key])
    return {"sessions": sessions, "recorded_worker_wall_seconds": sum(row["recorded_wall_seconds"] for row in sessions.values()),
            "complete_wall_accounting": all(row["closed"] for row in sessions.values()),
            "scope": "All raw sessions, including abandoned attempts, setup, validation and checkpoint work; interrupted sessions give lower bounds through last durable event. GPU reservation/billing hours are separate."}


def _initialize(config, arm, train, train_bank):
    # A dense counterfactual defines the same training-only target for every arm.
    set_seed(config.training.seed)
    dense_config = replace(config, head=replace(config.head, kind="dense", d_e=config.model.d_model, num_queries=1))
    device = resolve_device(config.training.device)
    reference = LanguageModel(dense_config).to(device)
    reference_sha = _tensor_state_digest(reference.backbone.state_dict())
    features, targets = probe_features(reference, train.tokens, train_bank)
    target_std = diagnose_head(reference.head, features, targets)["aggregate_logits"]["centered_std"]
    del features, targets, reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(config.training.seed)
    model = LanguageModel(config).to(device).train()
    before_parameters = {name: {"sha256": _tensor_state_digest({name: parameter}), "shape": list(parameter.shape),
                                 "elements": parameter.numel()} for name, parameter in model.head.named_parameters()}
    backbone_sha = _tensor_state_digest(model.backbone.state_dict())
    if backbone_sha != reference_sha:
        raise ValueError("Counterfactual and candidate backbones differ")
    features, targets = probe_features(model, train.tokens, train_bank)
    before = diagnose_head(model.head, features, targets)
    transform = {"applied": False, "query_projection_gain": 1.0}
    if arm.endswith("_scale"):
        transform = {**calibrate_initial_query_gain(model.head, features, target_logit_std=target_std), "applied": True}
    after = diagnose_head(model.head, features, targets)
    after_parameters = {name: {"sha256": _tensor_state_digest({name: parameter}), "shape": list(parameter.shape),
                                "elements": parameter.numel()} for name, parameter in model.head.named_parameters()}
    changed = [name for name in before_parameters if before_parameters[name] != after_parameters[name]]
    allowed = {"query_projection.weight", "query_projection.bias"} if arm.endswith("_scale") else set()
    if set(changed) - allowed:
        raise ValueError("Initialization changed parameters outside the declared final projection")
    del features, targets
    model.zero_grad(set_to_none=True)
    # Equal stochastic training RNG origins; initialization consumes different draws.
    set_seed(config.training.seed + 100_000)
    return model, {"arm": arm, "initial_backbone_sha256": backbone_sha,
                   "dense_target_logit_std": target_std, "before": before, "after": after,
                   "head_parameters_before": before_parameters, "head_parameters_after": after_parameters,
                   "changed_parameters": changed,
                   "transform": transform, "training_probe_bank_sha256": train_bank["bank_sha256"]}


def run_confirmation(data_dir, output_dir, *, arm, seed, device="cuda:0", chunk=512,
                     stop_after=1000, cpu_test=False, dry_run=False, resume=False,
                     approval_path=None, preflight=False):
    root, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Learning outputs must be outside prepared data")
    identity = prepared_identity(root)
    tokenizer = load_tokenizer(root / "tokenizer.json")
    config = confirmation_config(tokenizer.vocab_size, arm, seed, device=device, chunk=chunk, cpu_test=cpu_test)
    if not cpu_test and tokenizer.vocab_size != 65536:
        raise ValueError("Full confirmation requires the frozen65536-token vocabulary")
    if type(stop_after) is not int or not 1 <= stop_after <= config.training.max_steps:
        raise ValueError("Invalid absolute stop endpoint")
    if preflight and (stop_after != 2 or resume):
        raise ValueError("GPU preflight is a fresh two-update prefix of the fixed schedule")
    if not cpu_test and stop_after not in ((2,) if preflight else (1000, 2000, 4000)):
        raise ValueError("Only preregistered confirmation endpoints are allowed")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validate_volume(manifest, cpu_test=cpu_test)
    train = TokenStream.from_file(root / "train.npy", config.model.context_length, tokenizer.vocab_size)
    validation = TokenStream.from_file(root / "validation.npy", config.model.context_length, tokenizer.vocab_size)
    if train.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation streams are identical")
    approval = json.loads(Path(approval_path).read_text(encoding="utf-8")) if approval_path else None
    if not cpu_test and not dry_run and (not approval or approval.get("status") != "passed"
            or approval.get("data_identity") != identity or approval.get("approved_for_learning") is not True):
        raise ValueError("A passed, data-bound source/preparation approval is required before GPU learning")
    train_bank = build_window_bank(train.tokens, manifest, split="train", context_length=config.model.context_length,
                                   windows_per_language=1, seed=2026102301)
    val_bank = build_window_bank(validation.tokens, manifest, split="validation", context_length=config.model.context_length,
                                 windows_per_language=1 if cpu_test else 256, seed=2026102302)
    plan = {"protocol_version": VERSION, "arm": arm, "seed": seed, "required_seeds": [0, 1, 2],
            "required_arms": list(ARMS), "config": config.to_dict(), "data_identity": identity,
            "train_fingerprint": train.fingerprint, "validation_fingerprint": validation.fingerprint,
            "train_probe_bank": train_bank, "validation_bank": val_bank, "source_sha256": source_hashes(),
            "approval_sha256": file_sha256(approval_path) if approval_path else None,
            "cpu_correctness_test_only": cpu_test, "loss_backend": "reference_exact_full_vocabulary_ce",
            "memory_ceiling_fraction": .9,
            "sampling": "exactly8windows/language/global32; shuffled, shared successful batches within seed",
            "schedule_horizon": config.training.max_steps, "first_endpoint": 4 if cpu_test else 1000,
            "timing_claim": "worker wall and synchronized attempts; parallel jobs are not isolated throughput evidence"}
    if dry_run:
        return plan
    if device == "cpu" and not cpu_test:
        raise ValueError("Full learning requires CUDA; use cpu_test only for software checks")
    torch.set_num_threads(config.training.num_threads)
    hardware = inspect_device(device, cpu_test=cpu_test)
    device_object = resolve_device(device)
    started = time.perf_counter()
    if resume:
        previous = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in plan.items()):
            raise ValueError("Resume changes the frozen experiment/data/source/config")
        if previous.get("hardware") != hardware:
            raise ValueError("Resume changes the recorded hardware/software environment")
        initialization = json.loads((output / "initialization.json").read_text(encoding="utf-8"))
        set_seed(seed)
        model = LanguageModel(config).to(device_object).train()
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", {**plan, "hardware": hardware})
        model, initialization = _initialize(config, arm, train, train_bank)
        write_json(output / "initialization.json", initialization)
    tc = config.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, betas=tc.betas, weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if tc.precision == "fp16" else None
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    completed, successful_tokens, attempted_tokens, overflows = 0, 0, 0, 0
    if resume:
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        state = payload["extra_state"]
        if (payload["config"] != config.to_dict() or payload["data_identity"] != identity
                or payload["train_fingerprint"] != train.fingerprint
                or payload["validation_fingerprint"] != validation.fingerprint
                or state["plan_sha256"] != digest(plan)
                or state["initialization_sha256"] != file_sha256(output / "initialization.json")):
            raise ValueError("Checkpoint differs from the frozen confirmation run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler"])
        generator.set_state(payload["sampler_rng"])
        completed, successful_tokens = payload["completed_steps"], payload["predicted_tokens"]
        attempted_tokens, overflows = state["attempted_tokens"], state["amp_overflows"]
        _restore_rng(payload["rng"])
        del payload
    if completed > stop_after:
        raise ValueError("Stop endpoint precedes checkpoint")
    setup_seconds = time.perf_counter() - started
    metrics = output / "metrics.jsonl"
    prepare_metrics_append(metrics)
    session = uuid.uuid4().hex
    milestones = set(EVALUATION_STEPS if not cpu_test else (0, 1, 2, 4, 8)) | {stop_after}
    if device_object.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_object)
    with metrics.open("a", encoding="utf-8") as events:
        def emit(value):
            memory = {"peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device_object),
                      "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device_object)} if device_object.type == "cuda" else {}
            events.write(json.dumps({"session_id": session, "elapsed_session_seconds": time.perf_counter() - started,
                                     **memory, **value}, allow_nan=False) + "\n")
            events.flush()

        def evaluate_now():
            _sync(device_object)
            before = time.perf_counter()
            result = evaluate_bank(model, validation.tokens, val_bank, microbatch=1,
                                   precision=tc.precision, chunk_positions=chunk)
            _sync(device_object)
            emit({"event": "validation", "step": completed, "wall_seconds": time.perf_counter() - before, **result})
            print(f"confirmation seed={seed} arm={arm} step={completed} macro_nll={result['macro_nll']:.6f}", flush=True)

        def persist():
            before = time.perf_counter()
            save_checkpoint(output / "last.pt", model=model, optimizer=optimizer, config=config,
                            completed_steps=completed, predicted_tokens=successful_tokens,
                            generator=generator, train_fingerprint=train.fingerprint,
                            validation_fingerprint=validation.fingerprint, scaler=scaler, data_identity=identity,
                            extra_state={"plan_sha256": digest(plan), "initialization_sha256": file_sha256(output / "initialization.json"),
                                         "attempted_tokens": attempted_tokens, "amp_overflows": overflows})
            emit({"event": "checkpoint", "step": completed, "wall_seconds": time.perf_counter() - before})

        emit({"event": "resume" if resume else "start", "step": completed, "setup_seconds": setup_seconds,
              "hardware": hardware})
        if not resume:
            evaluate_now()
        pending = None
        consecutive = 0
        while completed < stop_after:
            _sync(device_object)
            before = time.perf_counter()
            if pending is None:
                pending = sample_global(train.tokens, manifest, generator, context_length=config.model.context_length)
            x, y, records = pending
            batch_sha = window_digest(x, y)
            result = optimizer_attempt(model, optimizer, scaler, config, x, y, completed)
            _sync(device_object)
            elapsed = time.perf_counter() - before
            if device_object.type == "cuda" and torch.cuda.max_memory_reserved(device_object) > .9 * hardware["vram_bytes"]:
                raise MemoryError("Confirmation exceeds the frozen90% reserved-VRAM ceiling")
            attempted_tokens += result["attempted_tokens"]
            if result["skipped"]:
                overflows += 1
                consecutive += 1
                emit({"event": "amp_overflow", "step": completed, "attempt_seconds": elapsed,
                      "window_sha256": batch_sha, **result})
                if overflows >= 40 or consecutive >= 20:
                    raise FloatingPointError("Confirmation exceeded its AMP retry allowance")
                continue
            consecutive = 0
            completed += 1
            successful_tokens += result["attempted_tokens"]
            emit({"event": "train", "step": completed, "window_sha256": batch_sha,
                  "windows": records, "attempt_seconds": elapsed, "attempted_tokens_total": attempted_tokens,
                  "successful_tokens_total": successful_tokens,
                  "per_language_predicted_tokens": {language: 8 * config.model.context_length for language in LANGUAGES}, **result})
            pending = None
            if completed in milestones:
                evaluate_now()
            if completed % 250 == 0 or completed == stop_after:
                persist()
        finite = finite_state_audit(model, optimizer, scaler)
    checkpoint = audit_checkpoint(output / "last.pt", config, completed_steps=completed,
                                  predicted_tokens=successful_tokens, data_identity=identity)
    # Include final reload/finiteness audit time in a durable per-session wall.
    with metrics.open("a", encoding="utf-8") as events:
        emit({"event": "session_end", "step": completed})
    summary = {"protocol_version": VERSION, "status": "complete", "arm": arm, "seed": seed,
               "stage": "preflight_prefix" if preflight else "confirmation",
               "absolute_stop_after": stop_after, "completed_steps": completed,
               "successful_tokens": successful_tokens, "attempted_tokens": attempted_tokens,
               "amp_overflows": overflows, "data_identity": identity, "plan_sha256": digest(plan),
               "initialization_sha256": file_sha256(output / "initialization.json"),
               "final_state_audit": finite, "checkpoint": checkpoint,
               "session_wall_seconds": time.perf_counter() - started,
               "all_session_costs": recorded_costs(metrics),
               "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device_object) if device_object.type == "cuda" else None,
               "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device_object) if device_object.type == "cuda" else None,
               "isolated_throughput_claim": False}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk", type=int, choices=(64, 512), default=512)
    parser.add_argument("--stop-after", type=int, default=1000)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Run a fresh two-update GPU prefix, retained for exact continuation under the same plan.")
    args = parser.parse_args()
    result = run_confirmation(args.data_dir, args.output_dir, arm=args.arm, seed=args.seed,
                              device=args.device, chunk=args.chunk, stop_after=args.stop_after,
                              cpu_test=args.cpu_test, dry_run=args.dry_run, resume=args.resume,
                              approval_path=args.approval, preflight=args.preflight)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
