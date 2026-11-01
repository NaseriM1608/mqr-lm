"""Exact likelihood training, evaluation and resumable experiment state."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ExperimentConfig
from .data import TokenStream
from .model import LanguageModel
from .tracking import prepare_metrics_append


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose a CPU config or a CUDA PyTorch environment.")
    return device


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed-precision training currently requires CUDA.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; choose FP16 with scaling or FP32.")
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def exact_cross_entropy(head, features: torch.Tensor, targets: torch.Tensor,
                        chunk_positions: int = 64, checkpoint_chunks: bool = True) -> torch.Tensor:
    """Full-vocabulary CE with recomputation of per-chunk head activations.

    Targets are already shifted; -100 masks a position. Chunk checkpointing
    retains the backbone graph but avoids retaining every component-logit slab.
    """
    if features.shape[:-1] != targets.shape:
        raise ValueError("Targets must match the feature leading dimensions.")
    if chunk_positions <= 0:
        raise ValueError("chunk_positions must be positive.")
    flat_f = features.reshape(-1, features.shape[-1])
    flat_y = targets.reshape(-1)
    count = (flat_y != -100).sum()
    if count.item() == 0:
        return features.sum() * 0.0

    def summed_loss(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(head.score_all(f).float(), y, reduction="sum", ignore_index=-100)

    total = features.new_zeros((), dtype=torch.float32)
    for offset in range(0, flat_y.numel(), chunk_positions):
        f = flat_f[offset:offset + chunk_positions]
        y = flat_y[offset:offset + chunk_positions]
        if checkpoint_chunks and torch.is_grad_enabled():
            part = checkpoint(summed_loss, f, y, use_reentrant=False)
        else:
            part = summed_loss(f, y)
        total = total + part
    return total / count


def learning_rate_at(step: int, config) -> float:
    """Rate for zero-based next optimizer step, tied to the full run schedule."""
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    span = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step + 1 - config.warmup_steps) / span))
    factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return config.learning_rate * factor


@torch.inference_mode()
def evaluate(model: LanguageModel, stream: TokenStream, *, batches: int = 4,
             batch_size: int = 2, seed: int = 123, precision: str = "fp32",
             chunk_positions: int = 64) -> dict[str, float | int]:
    if batches <= 0 or batch_size <= 0 or chunk_positions <= 0:
        raise ValueError("Evaluation batch/chunk sizes must be positive.")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nll_sum, correct, count = 0.0, 0, 0
    try:
        for _ in range(batches):
            x, y = stream.sample(batch_size, generator)
            x, y = x.to(device), y.to(device)
            with autocast_context(device, precision):
                features, _ = model.backbone(x)
                features, y = features.reshape(-1, features.shape[-1]), y.reshape(-1)
                for offset in range(0, y.numel(), chunk_positions):
                    chunk_y = y[offset:offset + chunk_positions]
                    logits = model.head.score_all(features[offset:offset + chunk_positions]).float()
                    valid = chunk_y != -100
                    nll_sum += F.cross_entropy(logits, chunk_y, ignore_index=-100, reduction="sum").item()
                    correct += ((logits.argmax(-1) == chunk_y) & valid).sum().item()
                    count += valid.sum().item()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation has no predicted positions.")
    nll = nll_sum / count
    return {"full_nll": nll, "perplexity": math.exp(nll) if nll < 700 else float("inf"),
            "accuracy": correct / count, "predicted_tokens": count}


def _rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])]}


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))


def save_checkpoint(path: Path, *, model, optimizer, config: ExperimentConfig,
                    completed_steps: int, predicted_tokens: int, generator: torch.Generator,
                    train_fingerprint: str, validation_fingerprint: str | None,
                    scaler=None, data_identity: dict | None = None,
                    extra_state: dict | None = None) -> None:
    payload = {"format_version": 1, "config": config.to_dict(), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "completed_steps": completed_steps,
               "predicted_tokens": predicted_tokens, "sampler_rng": generator.get_state(),
               "rng": _rng_state(), "schedule": {"completed_steps": completed_steps,
                                                   "max_steps": config.training.max_steps},
               "scaler": scaler.state_dict() if scaler is not None else None,
               "train_fingerprint": train_fingerprint, "validation_fingerprint": validation_fingerprint,
               "data_identity": data_identity or {"kind": "token_ids_only"}}
    if extra_state is not None:
        payload["extra_state"] = extra_state
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(path: str | Path, device: str = "cpu") -> tuple[LanguageModel, ExperimentConfig, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    config = ExperimentConfig.from_dict(payload["config"])
    model = LanguageModel(config).to(resolve_device(device))
    model.load_state_dict(payload["model"])
    return model, config, payload


def _resume_config(config: ExperimentConfig) -> dict:
    result = config.to_dict()
    # Moving a run or adjusting reporting frequency does not change its schedule.
    for key in ("device", "num_threads", "checkpoint_every", "eval_every", "eval_batches"):
        result["training"].pop(key, None)
    return result


def run_training(config: ExperimentConfig, train_stream: TokenStream,
                 validation_stream: TokenStream | None, run_dir: str | Path, *,
                 resume: str | Path | None = None, stop_after: int | None = None,
                 data_identity: dict | None = None) -> dict:
    config.validate()
    tc = config.training
    torch.set_num_threads(tc.num_threads)
    device = resolve_device(tc.device)
    set_seed(tc.seed)
    if train_stream.context_length != config.model.context_length:
        raise ValueError("Training stream context length differs from the model configuration.")
    if train_stream.max_token_id >= config.model.vocab_size:
        raise ValueError("Training token IDs exceed model vocabulary.")
    if validation_stream is not None:
        if validation_stream.context_length != config.model.context_length:
            raise ValueError("Validation stream context length differs from the model configuration.")
        if validation_stream.max_token_id >= config.model.vocab_size:
            raise ValueError("Validation token IDs exceed model vocabulary.")
        if validation_stream.fingerprint == train_stream.fingerprint:
            raise ValueError("Training and validation token streams are identical.")
    run_dir = Path(run_dir)
    if resume is None and (run_dir / "manifest.json").exists():
        raise ValueError("Run already exists; choose a new directory or pass --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, betas=tuple(tc.betas),
                                  weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if tc.precision == "fp16" else None
    generator = torch.Generator(device="cpu").manual_seed(tc.seed + 101)
    completed_steps, predicted_tokens = 0, 0
    validation_fingerprint = validation_stream.fingerprint if validation_stream is not None else None
    data_identity = dict(data_identity) if data_identity is not None else {"kind": "token_ids_only"}
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        saved_config = ExperimentConfig.from_dict(payload["config"])
        if _resume_config(config) != _resume_config(saved_config):
            raise ValueError("Resume configuration changes the architecture, data schedule or optimization.")
        if payload["train_fingerprint"] != train_stream.fingerprint or payload["validation_fingerprint"] != validation_fingerprint:
            raise ValueError("Resume data fingerprints do not match the checkpoint.")
        if payload.get("data_identity", {"kind": "token_ids_only"}) != data_identity:
            raise ValueError("Resume tokenizer/prepared-data identity does not match the checkpoint.")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        generator.set_state(payload["sampler_rng"])
        completed_steps, predicted_tokens = payload["completed_steps"], payload["predicted_tokens"]
        _restore_rng(payload["rng"])
    limit = tc.max_steps if stop_after is None else min(tc.max_steps, stop_after)
    if limit < completed_steps:
        raise ValueError("stop_after precedes the checkpoint step.")
    session_id = uuid.uuid4().hex
    manifest = {"implementation": "MQR-v1", "config": config.to_dict(), "session_id": session_id,
                "config_sha256": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
                "train_fingerprint": train_stream.fingerprint, "validation_fingerprint": validation_fingerprint,
                "data_identity": data_identity,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": np.__version__, "platform": platform.platform(),
                                "device": str(device), "cpu_threads": tc.num_threads,
                                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                "resumed_from": str(resume) if resume is not None else None}
    prepare_metrics_append(run_dir / "metrics.jsonl")
    manifest_path = run_dir / "manifest.json"
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.{session_id}.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_temporary, manifest_path)
    events = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def emit(event: dict) -> None:
        events.write(json.dumps({"session_id": session_id, **event}, allow_nan=False) + "\n")
        events.flush()

    def persist(name: str = "last.pt") -> None:
        save_checkpoint(run_dir / name, model=model, optimizer=optimizer, config=config,
                        completed_steps=completed_steps, predicted_tokens=predicted_tokens,
                        generator=generator, train_fingerprint=train_stream.fingerprint,
                        validation_fingerprint=validation_fingerprint, scaler=scaler, data_identity=data_identity)

    emit({"event": "resume" if resume else "start", "step": completed_steps, "predicted_tokens": predicted_tokens})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    initial_steps = completed_steps
    overflows = 0
    last_eval = None
    try:
        model.train()
        while completed_steps < limit:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate_at(completed_steps, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batches = [train_stream.sample(tc.batch_size, generator) for _ in range(tc.gradient_accumulation)]
            total_valid = sum((y != -100).sum().item() for _, y in batches)
            if total_valid == 0:
                raise ValueError("Training step contains no predicted tokens.")
            step_loss = 0.0
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                with autocast_context(device, tc.precision):
                    features, _ = model.backbone(x)
                    mean_loss = exact_cross_entropy(model.head, features, y, tc.loss_chunk_positions)
                    weighted = mean_loss * ((y != -100).sum() / total_valid)
                if not torch.isfinite(weighted).item():
                    raise FloatingPointError("Nonfinite training loss; checkpoint and inspect scales/data.")
                if scaler is not None:
                    scaler.scale(weighted).backward()
                else:
                    weighted.backward()
                step_loss += weighted.detach().item()
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip,
                                                       error_if_nonfinite=scaler is None)
            old_scale = scaler.get_scale() if scaler is not None else None
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
            else:
                optimizer.step()
                skipped = False
            predicted_tokens += total_valid
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_start
            if skipped:
                overflows += 1
                emit({"event": "amp_overflow", "step": completed_steps, "predicted_tokens": predicted_tokens,
                      "scale": scaler.get_scale()})
                if overflows >= 20:
                    raise FloatingPointError("20 consecutive AMP overflows; inspect scales or use BF16/FP32.")
                continue
            overflows = 0
            completed_steps += 1
            emit({"event": "train", "step": completed_steps, "predicted_tokens": predicted_tokens,
                  "full_nll": step_loss, "learning_rate": lr, "gradient_norm": float(grad_norm),
                  "step_seconds": elapsed, "step_tokens_s": total_valid / elapsed,
                  "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
            if validation_stream is not None and tc.eval_every and completed_steps % tc.eval_every == 0:
                last_eval = evaluate(model, validation_stream, batches=tc.eval_batches, batch_size=tc.batch_size,
                                     seed=tc.seed + 202, precision=tc.precision, chunk_positions=tc.loss_chunk_positions)
                emit({"event": "validation", "step": completed_steps, **last_eval})
            if tc.checkpoint_every and completed_steps % tc.checkpoint_every == 0:
                persist()
        persist()
    finally:
        events.close()
    summary = {"completed_steps": completed_steps, "predicted_tokens": predicted_tokens,
               "session_id": session_id,
               "session_optimizer_steps": completed_steps - initial_steps,
               "session_seconds": time.perf_counter() - start_time, "checkpoint": str(run_dir / "last.pt"),
               "last_validation": last_eval}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
